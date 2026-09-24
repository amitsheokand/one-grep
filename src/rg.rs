//! Managed ripgrep path: exact text + regex over workspace files.
//!
//! No index required. Exhaustive by default, gitignore-aware, results
//! carry file-oriented locations for terminal reading or agent context.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use grep::{
    regex::RegexMatcherBuilder,
    searcher::{BinaryDetection, SearcherBuilder, Sink, SinkMatch},
};
use ignore::{WalkBuilder, WalkState};
use serde::Serialize;

use crate::{Error, engine};

/// One matching line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Hit {
    /// File containing the match.
    pub path: PathBuf,
    /// 1-based line number.
    pub line: u64,
    /// Line text without trailing newline.
    pub text: String,
}

/// Search options.
#[derive(Debug, Clone)]
pub struct Options {
    /// Treat pattern as regex; otherwise match literally.
    pub regex: bool,
    /// Case-insensitive matching.
    pub case_insensitive: bool,
    /// Extra ignore-style globs (e.g. `["!target/*"]`).
    pub globs: Vec<String>,
    /// Language filter (`rust`, `python`, `nix`, `markdown`, or an
    /// extension like `rs`). Empty means all files.
    pub langs: Vec<String>,
    /// Maximum hits to collect.
    pub limit: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            regex: false,
            case_insensitive: false,
            globs: Vec::new(),
            langs: Vec::new(),
            limit: 100,
        }
    }
}

/// Languages accepted by [`Options::langs`].
pub const SUPPORTED_LANGS: &[&str] = &[
    "rust",
    "python",
    "typescript",
    "go",
    "java",
    "nix",
    "markdown",
];

/// Map a `--lang` value to file extensions, à la `ast-grep --lang`.
/// Accepts canonical names and bare extensions (`rs`, `py`, `md`).
#[must_use]
pub fn lang_extensions(lang: &str) -> Option<&'static [&'static str]> {
    match lang.trim().to_ascii_lowercase().as_str() {
        "rust" | "rs" => Some(&["rs"]),
        "python" | "py" => Some(&["py"]),
        "typescript" | "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "mts" | "cts" => {
            Some(&["ts", "mts", "cts", "tsx", "jsx", "js", "mjs", "cjs"])
        }
        "go" => Some(&["go"]),
        "java" => Some(&["java"]),
        "nix" => Some(&["nix"]),
        "markdown" | "md" => Some(&["md", "markdown"]),
        _ => None,
    }
}

/// Resolve [`Options::langs`] to an extension set, or fail closed naming
/// what is supported. Unknown languages are caller errors, never silent
/// match-nothings.
fn resolve_lang_exts(langs: &[String]) -> Result<Option<Vec<String>>, Error> {
    if langs.is_empty() {
        return Ok(None);
    }
    let mut exts = Vec::new();
    for lang in langs {
        match lang_extensions(lang) {
            Some(list) => exts.extend(list.iter().map(ToString::to_string)),
            None => {
                return Err(Error::InvalidInput(format!(
                    "unknown --lang `{lang}` (supported: {})",
                    SUPPORTED_LANGS.join(", ")
                )));
            }
        }
    }
    exts.sort();
    exts.dedup();
    Ok(Some(exts))
}

/// Whether `path` passes a language allowlist. Empty `langs` passes
/// everything; unknown entries fail closed like [`Options::langs`].
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for unknown languages.
pub fn matches_lang(path: &Path, langs: &[String]) -> Result<bool, Error> {
    let Some(exts) = resolve_lang_exts(langs)? else {
        return Ok(true);
    };
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    Ok(exts.iter().any(|e| *e == ext))
}

/// Fail fast on unknown languages before any retrieval runs.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for unknown languages.
pub fn validate_langs(langs: &[String]) -> Result<(), Error> {
    resolve_lang_exts(langs).map(|_| ())
}

/// Post-retrieval include-glob filter with the same whitelist semantics as
/// the walker overrides: a hit passes unless ignored, and when positive
/// globs exist it must match one. `!`-prefixed globs exclude.
#[derive(Clone, Debug)]
pub struct GlobFilter {
    matcher: Option<ignore::overrides::Override>,
    positive: bool,
}

impl GlobFilter {
    /// Build from caller globs. Empty input matches everything.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Walk`] for malformed glob patterns.
    pub fn build(root: &Path, globs: &[String]) -> Result<Self, Error> {
        if globs.is_empty() {
            return Ok(Self {
                matcher: None,
                positive: false,
            });
        }
        let mut builder = ignore::overrides::OverrideBuilder::new(root);
        for glob in globs {
            builder.add(glob)?;
        }
        Ok(Self {
            matcher: Some(builder.build()?),
            positive: globs.iter().any(|g| !g.starts_with('!')),
        })
    }

    /// Whether a hit path survives the filter.
    #[must_use]
    pub fn keep(&self, path: &Path) -> bool {
        let Some(matcher) = &self.matcher else {
            return true;
        };
        let matched = matcher.matched(path, false);
        if matched.is_ignore() {
            return false;
        }
        if matched.is_whitelist() {
            return true;
        }
        !self.positive
    }
}

/// Stopwords dropped when turning an NL query into live-grep terms.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "we",
    "what", "when", "where", "which", "who", "why", "with", "you",
];

/// Note shown when `search` degrades to live rg because `.one-grep` is missing.
#[must_use]
pub fn unindexed_note(workspace: &Path) -> String {
    let root = workspace.display();
    format!(
        "mode: rg-fallback (no index at {}). Indexing is available for better ranking: one-grep index {root} && one-grep embed {root}",
        engine::index_dir(workspace).display()
    )
}

/// Distinctive tokens from a natural-language query for live-grep fallback.
#[must_use]
pub fn fallback_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = query
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/')))
        .filter(|t| t.len() >= 2 && !is_stopword(t))
        .map(ToOwned::to_owned)
        .collect();
    terms.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    terms.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    let specific: Vec<String> = terms
        .iter()
        .filter(|t| identifier_like(t))
        .cloned()
        .collect();
    if !specific.is_empty() {
        terms = specific;
    }
    terms.truncate(8);
    terms
}

fn is_stopword(term: &str) -> bool {
    STOPWORDS.iter().any(|w| term.eq_ignore_ascii_case(w))
}

fn identifier_like(term: &str) -> bool {
    term.bytes()
        .any(|c| matches!(c, b'-' | b'_' | b'/' | b'.') || c.is_ascii_uppercase())
}

/// Strip one layer of shell-style surrounding quotes (`"..."` or `'...'`).
///
/// Agents often copy a shell-quoted pattern (`one-grep rg "foo bar" .`) into
/// the MCP `rg` JSON field, where no shell runs. Without this, the literal
/// search looks for the quote characters themselves and returns zero hits,
/// which reads as "MCP regex fails on quoted patterns". Stripping here makes
/// CLI and MCP agree: pass raw text, surrounding quotes are ignored.
///
/// Returns the original string when there is no clean outer pair (empty
/// inner text, or the inner text still contains the same quote char, e.g. a
/// real quoted-string search like `"foo" OR "bar"`).
#[must_use]
pub fn normalize_pattern(pattern: &str) -> &str {
    let trimmed = pattern.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        let (first, last) = (bytes[0], bytes[bytes.len() - 1]);
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            let inner = &trimmed[1..trimmed.len() - 1];
            if !inner.trim().is_empty() && !inner.contains(first as char) {
                return inner;
            }
        }
    }
    trimmed
}

fn regex_escape(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for c in term.chars() {
        if matches!(
            c,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Live grep for an unindexed `search` query: distinctive tokens, case-insensitive.
///
/// # Errors
///
/// Returns [`Error`] when the workspace cannot be walked or the pattern is bad.
pub fn fallback_search(workspace: &Path, query: &str, limit: usize) -> Result<Vec<Hit>, Error> {
    let terms = fallback_terms(query);
    let (pattern, regex) = if terms.is_empty() {
        (query.trim().to_owned(), false)
    } else {
        (
            terms
                .iter()
                .map(|t| regex_escape(t))
                .collect::<Vec<_>>()
                .join("|"),
            true,
        )
    };
    if pattern.is_empty() {
        return Ok(Vec::new());
    }
    search(
        workspace,
        &pattern,
        &Options {
            regex,
            case_insensitive: true,
            globs: Vec::new(),
            langs: Vec::new(),
            limit,
        },
    )
}

/// Search `workspace` for `pattern`, returning up to `options.limit` hits.
///
/// `workspace` is a directory walked in parallel, or a single file (like
/// ripgrep: harnesses sometimes map a file `path` onto the workspace slot).
/// The walk runs on all cores (`ignore` parallel walker); hits are sorted
/// by `(path, line)` and truncated, so output is deterministic across runs.
/// `limit` caps output lines, not files scanned. Surrounding shell-style
/// quotes are ignored via [`normalize_pattern`], so `"foo bar"` and
/// `foo bar` agree between CLI and MCP. An empty pattern after
/// normalization returns no hits instead of matching every line.
///
/// # Errors
///
/// Returns [`Error`] when the pattern fails to compile, a `--lang` is
/// unknown, the workspace is missing, or a file cannot be searched.
pub fn search(workspace: &Path, pattern: &str, options: &Options) -> Result<Vec<Hit>, Error> {
    if !workspace.exists() {
        return Err(Error::InvalidInput(format!(
            "workspace does not exist: {}",
            workspace.display()
        )));
    }
    let effective = normalize_pattern(pattern);
    if effective.is_empty() {
        return Ok(Vec::new());
    }
    let lang_exts = resolve_lang_exts(&options.langs)?;
    let matcher = RegexMatcherBuilder::new()
        .case_insensitive(options.case_insensitive)
        .fixed_strings(!options.regex)
        .build(effective)?;

    if workspace.is_file() {
        return search_one_file(workspace, &matcher, options.limit);
    }

    let mut builder = WalkBuilder::new(workspace);
    builder
        .hidden(true)
        .parents(true)
        .git_ignore(true)
        .require_git(false);
    if !options.globs.is_empty() {
        let mut overrides = ignore::overrides::OverrideBuilder::new(workspace);
        for glob in &options.globs {
            overrides.add(glob)?;
        }
        builder.overrides(overrides.build()?);
    }
    let limit = options.limit;
    let hits = Mutex::new(Vec::<Hit>::new());
    builder.build_parallel().run(|| {
        let matcher = &matcher;
        let hits = &hits;
        let lang_exts = &lang_exts;
        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .binary_detection(BinaryDetection::quit(b'\x00'))
            .build();
        Box::new(move |entry: Result<ignore::DirEntry, ignore::Error>| {
            let Ok(path) = entry.map(|e| e.path().to_path_buf()) else {
                return WalkState::Continue;
            };
            if !path.is_file() {
                return WalkState::Continue;
            }
            if let Some(exts) = lang_exts {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if !exts.iter().any(|e| *e == ext) {
                    return WalkState::Continue;
                }
            }
            // Skip unreadable files instead of failing the whole search.
            let mut sink = VecSink {
                path: path.clone(),
                out: Vec::new(),
            };
            if searcher.search_path(matcher, &path, &mut sink).is_err() {
                return WalkState::Continue;
            }
            if !sink.out.is_empty() {
                hits.lock().expect("hits mutex").extend(sink.out);
            }
            WalkState::Continue
        })
    });
    let mut hits = hits.into_inner().expect("hits mutex");
    hits.sort_by(|a, b| a.path.cmp(&b.path).then_with(|| a.line.cmp(&b.line)));
    hits.truncate(limit);
    Ok(hits)
}

/// Search a single file (no walk). Skip unreadable files quietly, like the
/// walker path does.
fn search_one_file(
    path: &Path,
    matcher: &grep::regex::RegexMatcher,
    limit: usize,
) -> Result<Vec<Hit>, Error> {
    let mut searcher = SearcherBuilder::new()
        .line_number(true)
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .build();
    let mut sink = VecSink {
        path: path.to_path_buf(),
        out: Vec::new(),
    };
    let _ = searcher.search_path(matcher, path, &mut sink);
    sink.out.truncate(limit);
    Ok(sink.out)
}

/// Per-file sink used by the parallel walker (merged under a mutex after).
struct VecSink {
    path: PathBuf,
    out: Vec<Hit>,
}

impl Sink for VecSink {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &grep::searcher::Searcher,
        mat: &SinkMatch<'_>,
    ) -> Result<bool, std::io::Error> {
        let text = String::from_utf8_lossy(mat.bytes());
        self.out.push(Hit {
            path: self.path.clone(),
            line: mat.line_number().unwrap_or(0),
            text: text.trim_end_matches(['\r', '\n']).to_owned(),
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn workspace_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, contents) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("mkdir");
            }
            let mut file = std::fs::File::create(&path).expect("create");
            file.write_all(contents.as_bytes()).expect("write");
        }
        dir
    }

    #[test]
    fn literal_search_finds_lines_with_locations() {
        let dir = workspace_with(&[("a.txt", "hello world\nfoo bar\n")]);
        let hits = search(dir.path(), "foo", &Options::default()).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 2);
        assert_eq!(hits[0].text, "foo bar");
        assert_eq!(hits[0].path, dir.path().join("a.txt"));
    }

    #[test]
    fn regex_search_matches_pattern() {
        let dir = workspace_with(&[("a.txt", "foo123\nbar\n")]);
        let options = Options {
            regex: true,
            ..Options::default()
        };
        let hits = search(dir.path(), r"foo\d+", &options).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "foo123");
    }

    #[test]
    fn literal_search_ignores_regex_meta() {
        let dir = workspace_with(&[("a.txt", "foo.bar\nfooxbar\n")]);
        let hits = search(dir.path(), "foo.bar", &Options::default()).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "foo.bar");
    }

    #[test]
    fn gitignored_files_are_skipped() {
        let dir = workspace_with(&[
            (".gitignore", "secret.txt\n"),
            ("secret.txt", "password hunter2\n"),
            ("notes.txt", "password hunter2\n"),
        ]);
        let hits = search(dir.path(), "hunter2", &Options::default()).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, dir.path().join("notes.txt"));
    }

    #[test]
    fn single_file_root_searches_that_file() {
        let dir = workspace_with(&[("a.txt", "needle here\n"), ("b.txt", "needle here\n")]);
        let hits =
            search(&dir.path().join("a.txt"), "needle", &Options::default()).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, dir.path().join("a.txt"));
    }

    #[test]
    fn non_directory_is_invalid_input() {
        let err =
            search(Path::new("/no/such/dir"), "x", &Options::default()).expect_err("must fail");
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn bad_regex_is_bad_pattern() {
        let dir = workspace_with(&[("a.txt", "x\n")]);
        let options = Options {
            regex: true,
            ..Options::default()
        };
        let err = search(dir.path(), "(", &options).expect_err("must fail");
        assert!(matches!(err, Error::BadPattern(_)));
    }

    #[test]
    fn fallback_terms_drop_stopwords_and_keep_hyphenated() {
        let terms = fallback_terms("where is the one-grep home manager module");
        assert_eq!(terms, vec!["one-grep"]);
    }

    #[test]
    fn fallback_search_finds_hits_without_index() {
        let dir = workspace_with(&[("notes.txt", "one-grep hybrid search\n")]);
        let hits =
            fallback_search(dir.path(), "where is one-grep configured", 10).expect("fallback");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, dir.path().join("notes.txt"));
        assert!(unindexed_note(dir.path()).contains("Indexing is available"));
    }

    #[test]
    fn normalize_strips_one_outer_quote_pair() {
        assert_eq!(normalize_pattern("\"foo bar\""), "foo bar");
        assert_eq!(normalize_pattern("'foo bar'"), "foo bar");
        assert_eq!(normalize_pattern("  \"foo bar\"  "), "foo bar");
        assert_eq!(normalize_pattern("foo bar"), "foo bar");
        // Ambiguous / empty: keep original (trimmed) so we never match-all.
        assert_eq!(normalize_pattern("\"\""), "\"\"");
        assert_eq!(normalize_pattern("\"   \""), "\"   \"");
        assert_eq!(
            normalize_pattern("\"foo\" OR \"bar\""),
            "\"foo\" OR \"bar\""
        );
    }

    #[test]
    fn quoted_literal_finds_same_hits_as_raw() {
        let dir = workspace_with(&[("a.txt", "comment lies here\n")]);
        for pattern in ["comment lies", "\"comment lies\"", "'comment lies'"] {
            let hits = search(dir.path(), pattern, &Options::default()).expect("search");
            assert_eq!(hits.len(), 1, "{pattern}");
        }
    }

    #[test]
    fn empty_after_normalize_returns_no_hits() {
        let dir = workspace_with(&[("a.txt", "anything\n")]);
        for pattern in ["", "   ", "\"\""] {
            let hits = search(dir.path(), pattern, &Options::default()).expect("search");
            assert!(hits.is_empty(), "{pattern}");
        }
    }

    /// Forall shape (Kani: `kani::any::<String>()`, assume valid UTF-8,
    /// assert idempotence): stripping is a fixed point. Without a Kani
    /// binary here, the sampled cases below stand in for the harness.
    #[test]
    fn normalize_is_idempotent() {
        for pattern in [
            "\"foo bar\"",
            "'foo bar'",
            "  \"foo bar\"  ",
            "foo bar",
            "\"\"",
            "\"   \"",
            "\"foo\" OR \"bar\"",
            "",
        ] {
            let once = normalize_pattern(pattern);
            assert_eq!(normalize_pattern(once), once, "{pattern}");
        }
    }

    #[test]
    fn lang_filter_restricts_to_extensions() {
        let dir = workspace_with(&[
            ("main.rs", "needle here\n"),
            ("notes.txt", "needle here\n"),
            ("app.py", "needle here\n"),
        ]);
        let rust_only = Options {
            langs: vec!["rust".to_owned()],
            ..Options::default()
        };
        let hits = search(dir.path(), "needle", &rust_only).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, dir.path().join("main.rs"));
        // Extension aliases agree with canonical names.
        let alias = Options {
            langs: vec!["rs".to_owned(), "py".to_owned()],
            ..Options::default()
        };
        let hits = search(dir.path(), "needle", &alias).expect("search");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn unknown_lang_fails_closed() {
        let dir = workspace_with(&[("a.rs", "x\n")]);
        let options = Options {
            langs: vec!["cobol".to_owned()],
            ..Options::default()
        };
        let err = search(dir.path(), "x", &options).expect_err("must fail");
        assert!(matches!(err, Error::InvalidInput(_)));
        assert!(err.to_string().contains("rust"));
    }

    #[test]
    fn parallel_search_is_sorted_and_deterministic() {
        let dir = workspace_with(&[
            ("b.txt", "same\nsame\n"),
            ("a.txt", "same\n"),
            ("sub/c.txt", "same\n"),
        ]);
        let first = search(dir.path(), "same", &Options::default()).expect("search");
        assert_eq!(first.len(), 4);
        let mut sorted = first.clone();
        sorted.sort_by(|a, b| a.path.cmp(&b.path).then_with(|| a.line.cmp(&b.line)));
        assert_eq!(first, sorted);
        let second = search(dir.path(), "same", &Options::default()).expect("search");
        assert_eq!(first, second);
    }

    #[test]
    fn hit_serializes_with_path_line_text() {
        let dir = workspace_with(&[("a.txt", "hello\n")]);
        let hits = search(dir.path(), "hello", &Options::default()).expect("search");
        let value = serde_json::to_value(&hits).expect("json");
        assert_eq!(value[0]["line"], 1);
        assert_eq!(value[0]["text"], "hello");
        assert!(value[0]["path"].as_str().unwrap().ends_with("a.txt"));
    }

    #[test]
    fn matches_lang_accepts_names_and_extensions() {
        use std::path::Path;
        for (lang, path, want) in [
            ("rust", "src/main.rs", true),
            ("rs", "src/main.rs", true),
            ("python", "src/main.rs", false),
            ("markdown", "docs/a.md", true),
            ("md", "docs/a.markdown", true),
            ("nix", "hosts/x.nix", true),
            ("rust", "no-extension", false),
        ] {
            assert_eq!(
                matches_lang(Path::new(path), &[lang.to_owned()]).expect("valid"),
                want,
                "{lang} {path}"
            );
        }
        assert!(matches_lang(Path::new("a.rs"), &[]).expect("empty passes"));
        let err = matches_lang(Path::new("a.rs"), &["cobol".to_owned()]).expect_err("unknown");
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn glob_filter_whitelist_and_negation() {
        use std::path::Path;
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let keep_all = GlobFilter::build(root, &[]).expect("empty");
        assert!(keep_all.keep(Path::new("a.rs")));
        let rs_only = GlobFilter::build(root, &["*.rs".to_owned()]).expect("glob");
        assert!(rs_only.keep(&root.join("a.rs")));
        assert!(!rs_only.keep(&root.join("a.txt")));
        let not_target = GlobFilter::build(root, &["!target/*".to_owned()]).expect("neg");
        assert!(not_target.keep(&root.join("src/a.rs")));
        assert!(!not_target.keep(&root.join("target/a.rs")));
    }
}
