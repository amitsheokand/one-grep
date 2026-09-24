//! MCP server: `search` + `rg` tools over stdio or loopback HTTP.
//!
//! HTTP binds 127.0.0.1 only and requires `Authorization: Bearer <token>`.

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use rmcp::{
    ErrorData as McpError, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::{
        stdio,
        streamable_http_server::{StreamableHttpService, session::local::LocalSessionManager},
    },
};

use crate::{embed::FastembedProvider, fuse, index, lsp, rg, route};
use serde::Deserialize;

/// Max chars of chunk text per tool hit.
const TEXT_CAP: usize = 1200;

/// Output format for MCP tools. `text` (default) is the citable
/// `path:line` rendering; `json` returns
/// `{"notes": [...], "hits": [...]}` for machine parsing — robust against
/// paths containing `:` that break line splitting. Budget caps apply to
/// both: format never widens the result set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Format {
    Text,
    Json,
}

impl Format {
    fn opt(raw: Option<Format>) -> Self {
        raw.unwrap_or(Self::Text)
    }

    fn envelope<T: serde::Serialize>(&self, notes: &[String], hits: &T) -> Option<String> {
        match self {
            Self::Text => None,
            Self::Json => Some(serde_json::json!({"notes": notes, "hits": hits}).to_string()),
        }
    }
}

fn text_block(text: String) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(text)])
}

fn check_root(root: &str) -> Result<PathBuf, McpError> {
    let path = PathBuf::from(root);
    if !path.is_absolute() {
        return Err(McpError::invalid_params(
            format!("root must be absolute: {root}"),
            None,
        ));
    }
    if !path.is_dir() {
        return Err(McpError::invalid_params(
            format!("root is not a directory: {root}"),
            None,
        ));
    }
    Ok(path)
}

fn internal(e: impl std::fmt::Display) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

/// Env override for the serving-log path (tests point it at a tmp file).
pub const ENV_SERVING_LOG: &str = "ONE_GREP_SERVING_LOG";

/// Best-effort per-call serving log (`~/.one-grep/serving.log`, JSONL).
/// One row per MCP call: tool, root, query, hits returned, response
/// chars, notes, latency. Never fails the call — errors are swallowed.
/// A pi-side correlator joins these rows with subsequent `read` calls to
/// answer "hits → did they still read the file?".
fn log_call(
    tool: &str,
    root: &str,
    query: &str,
    hits: usize,
    chars: usize,
    notes: &[String],
    latency_ms: f64,
) {
    let path = std::env::var_os(ENV_SERVING_LOG)
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|h| h.join(".one-grep").join("serving.log"))
        });
    let Some(path) = path else { return };
    let query: String = query.chars().take(500).collect();
    let row = serde_json::json!({
        "ts": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "tool": tool,
        "root": root,
        "query": query,
        "hits": hits,
        "chars": chars,
        "notes": notes,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    });
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    use std::io::Write as _;
    let _ = writeln!(file, "{row}");
}

/// Caller errors (unknown `--lang`, bad glob) are `invalid_params`; engine
/// failures stay `internal_error`. Fails closed either way.
fn rg_error(e: crate::Error) -> McpError {
    match e {
        crate::Error::InvalidInput(msg) => McpError::invalid_params(msg, None),
        crate::Error::BadPattern(e) => McpError::invalid_params(format!("bad pattern: {e}"), None),
        other => internal(other),
    }
}

/// Resolve a relative-or-absolute file against the workspace root,
/// rejecting lexical escapes. Existence is checked by the caller.
fn resolve_in_root(root: &Path, raw: &str) -> Option<PathBuf> {
    let joined = if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        root.join(raw)
    };
    let mut normal = PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::Prefix(prefix) => normal.push(prefix.as_os_str()),
            std::path::Component::RootDir => normal.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normal.pop();
            }
            std::path::Component::Normal(part) => normal.push(part),
        }
    }
    normal.starts_with(root).then_some(normal)
}

fn render(
    hit_path: &Path,
    start: u64,
    end: u64,
    breadcrumb: &str,
    score: &str,
    source: &str,
    text: &str,
) -> String {
    let mut short = text;
    if short.len() > TEXT_CAP {
        let mut end = TEXT_CAP;
        while !short.is_char_boundary(end) {
            end -= 1;
        }
        short = &short[..end];
    }
    // One header line: newlines in crumbs would break the line budget.
    let crumb: String = breadcrumb
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    format!(
        "{}:{start}-{end} [{crumb}] ({score}) source={source}\n{short}",
        hit_path.display()
    )
}

fn provider() -> Result<&'static FastembedProvider, McpError> {
    static PROVIDER: OnceLock<Result<FastembedProvider, String>> = OnceLock::new();
    PROVIDER
        .get_or_init(|| FastembedProvider::load().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| McpError::internal_error(format!("embedding model unavailable: {e}"), None))
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchParams {
    /// Absolute workspace root.
    root: String,
    /// Natural-language query; also used for BM25 ranking.
    query: String,
    /// Exact anchors folded into lexical ranking.
    fts: Option<Vec<String>>,
    /// Fuse vector similarity with BM25 (default true).
    fuse: Option<bool>,
    /// Ranking backend for `search_ranked` (`jev` default with unranked
    /// fallback, `jina` local ONNX, `llama` llama.cpp server). Ignored by
    /// unranked `search`.
    rank: Option<crate::embed::RankKind>,
    /// Restrict hits to languages (`rust`, `python`, `typescript`, `go`,
    /// `java`, `nix`, `markdown`).
    /// Applied to every path, including the exact-`rg` shortcut.
    lang: Option<Vec<String>>,
    /// Restrict hits to ignore-style globs (whitelist, `!` negates).
    globs: Option<Vec<String>>,
    /// Output format: `text` (default) or `json` (`{"notes","hits"}`).
    format: Option<Format>,
    /// Structural pattern fused as a third RRF list (needs `ast_lang` and
    /// the `ast-grep` binary on PATH). Applies to hybrid retrieval.
    ast_pattern: Option<String>,
    /// Language for `ast_pattern` (same vocabulary as `lang`).
    ast_lang: Option<String>,
    /// Max hits (default 10, max 50).
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct DefinitionParams {
    /// Absolute workspace root.
    root: String,
    /// File containing the reference, relative to root or absolute.
    path: String,
    /// 1-based line number of the reference.
    line: u32,
    /// 1-based character (Unicode scalar) of the reference on the line.
    character: u32,
    /// Server command override (default `rust-analyzer` from PATH).
    server: Option<String>,
    /// Output format: `text` (default) or `json` (`{"notes","hits"}`).
    format: Option<Format>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RgParams {
    /// Absolute workspace root.
    root: String,
    /// Pattern text (literal unless `regex`). Pass raw text without shell
    /// quotes; one outer `"..."` / `'...'` pair is ignored.
    pattern: String,
    /// Treat pattern as regex.
    regex: Option<bool>,
    /// Interpret the pattern structurally via ast-grep (needs exactly one
    /// `lang`; requires the `ast-grep` binary on PATH).
    structural: Option<bool>,
    /// Case-insensitive matching.
    case_insensitive: Option<bool>,
    /// Restrict to languages (`rust`, `python`, `typescript`, `go`,
    /// `java`, `nix`, `markdown`, or extensions like `rs`). Unknown values
    /// are rejected.
    lang: Option<Vec<String>>,
    /// Restrict to ignore-style globs (whitelist, `!` negates).
    globs: Option<Vec<String>>,
    /// Output format: `text` (default) or `json` (`{"notes","hits"}`).
    format: Option<Format>,
    /// Max hits (default 100, max 500).
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SkillParams {
    /// Task description; Jev ranks the skill library when a key is set.
    task: String,
    /// Max skills to return (default 2, max 5).
    limit: Option<usize>,
    /// Skill library root (default `ONE_GREP_SKILLS_DIR` or `~/.local/share/agent-skills`).
    dir: Option<String>,
    /// Output format: `text` (default) or `json` (`{"notes","hits"}`).
    format: Option<Format>,
}

#[derive(Clone)]
pub struct OneGrep {
    // Read by `#[tool_handler]` expansion; lint cannot see through the macro.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl OneGrep {
    /// Create the server.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Hybrid workspace search for intent and concepts, ranked with file:line cites. Standalone Rust paths (foo::Bar) and quoted literals (\"...\" / '...') use exact rg lookup unless fts anchors are supplied. Single identifier tokens go BM25, never hybrid. `lang`/`globs` filter hits on every path. Falls back to BM25 when no vector store exists, and to live rg when the workspace is not indexed."
    )]
    async fn search(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let limit = p.limit.unwrap_or(10).clamp(1, 50);
        let format = Format::opt(p.format);
        let fts: &[String] = p.fts.as_deref().unwrap_or(&[]);
        let filter = HitFilter::build(&root, p.lang.clone(), p.globs.clone())?;
        let started = std::time::Instant::now();
        match route::route(&p.query, fts) {
            route::Route::Literal(pattern) => {
                return self
                    .rg(Parameters(RgParams {
                        root: p.root.clone(),
                        pattern,
                        regex: Some(false),
                        structural: None,
                        case_insensitive: Some(false),
                        lang: p.lang.clone(),
                        globs: p.globs.clone(),
                        format: p.format,
                        limit: Some(limit),
                    }))
                    .await;
            }
            route::Route::Identifier(term) => {
                // Exact beats fuzzy: single tokens go BM25, never hybrid.
                let indexed_now = index::is_indexed(&root);
                let mut notes = Vec::new();
                if indexed_now && index::is_stale(&root) {
                    notes.push(index::stale_note(&root));
                }
                if !indexed_now {
                    notes.push(rg::unindexed_note(&root));
                }
                if indexed_now {
                    let mut hits = index::search(&root, &term, limit).map_err(internal)?;
                    hits.retain(|h| filter.keep(&h.path));
                    if let Some(envelope) = format.envelope(&notes, &hits) {
                        log_call(
                            "search",
                            &p.root,
                            &term,
                            hits.len(),
                            envelope.len(),
                            &notes,
                            started.elapsed().as_secs_f64(),
                        );
                        return Ok(text_block(envelope));
                    }
                    let lines: Vec<String> = hits.iter().map(render_ranked).collect();
                    let body = lines.join("\n---\n");
                    let text = if notes.is_empty() {
                        body
                    } else {
                        format!("{}\n\n{body}", notes.join("\n"))
                    };
                    log_call(
                        "search",
                        &p.root,
                        &term,
                        lines.len(),
                        text.len(),
                        &notes,
                        started.elapsed().as_secs_f64(),
                    );
                    return Ok(text_block(text));
                }
                let mut hits = rg::fallback_search(&root, &term, limit).map_err(internal)?;
                hits.retain(|h| filter.keep(&h.path));
                if let Some(envelope) = format.envelope(&notes, &hits) {
                    log_call(
                        "search",
                        &p.root,
                        &term,
                        hits.len(),
                        envelope.len(),
                        &notes,
                        started.elapsed().as_secs_f64(),
                    );
                    return Ok(text_block(envelope));
                }
                let lines: Vec<String> = hits.iter().map(render_live).collect();
                let body = lines.join("\n---\n");
                let text = format!("{}\n\n{body}", notes.join("\n"));
                log_call(
                    "search",
                    &p.root,
                    &term,
                    lines.len(),
                    text.len(),
                    &notes,
                    started.elapsed().as_secs_f64(),
                );
                return Ok(text_block(text));
            }
            route::Route::Intent(_) => {}
        }
        let mut query = p.query.clone();
        if let Some(fts) = &p.fts {
            query.push(' ');
            query.push_str(&fts.join(" "));
        }
        let indexed = index::is_indexed(&root);
        let fuse = p.fuse.unwrap_or(true);
        // Structural third list: needs both halves, and a binary behind it.
        let ast = match (&p.ast_pattern, &p.ast_lang) {
            (None, None) => None,
            (Some(pattern), Some(lang)) => {
                if !crate::astg::available() {
                    return Err(McpError::invalid_params(
                        "ast-grep not found on PATH (or ONE_GREP_AST_GREP)",
                        None,
                    ));
                }
                Some((pattern.clone(), lang.clone()))
            }
            _ => {
                return Err(McpError::invalid_params(
                    "ast_pattern and ast_lang must be set together",
                    None,
                ));
            }
        };
        // Did retrieval actually consume vectors? Only then is their
        // staleness worth a note.
        let mut used_vectors = false;
        let mut fused: Vec<fuse::FusedHit> = if !indexed {
            fuse::collect(&root, &query, limit, false, None).map_err(internal)?
        } else if fuse {
            match provider() {
                Ok(provider) => {
                    used_vectors = true;
                    fuse::hybrid(&root, &query, limit, Some(provider), None).map_err(internal)?
                }
                Err(_) => fuse::collect(&root, &query, limit, false, None).map_err(internal)?,
            }
        } else {
            fuse::collect(&root, &query, limit, false, None).map_err(internal)?
        };
        if let Some((pattern, lang)) = ast {
            let found = crate::astg::search(&root, &pattern, &lang, limit).map_err(rg_error)?;
            let refs: Vec<(PathBuf, u64)> = found.into_iter().map(|h| (h.path, h.line)).collect();
            fuse::apply_ast(&mut fused, &refs);
            fused.truncate(limit);
        }
        fused.retain(|h| filter.keep(&h.path));
        let mut notes = Vec::new();
        if indexed && index::is_stale(&root) {
            notes.push(index::stale_note(&root));
        }
        if used_vectors && crate::vectors::is_stale(&root) {
            notes.push(crate::vectors::stale_note(&root));
        }
        if !indexed {
            notes.push(rg::unindexed_note(&root));
        }
        if let Some(envelope) = format.envelope(&notes, &fused) {
            log_call(
                "search",
                &p.root,
                &query,
                fused.len(),
                envelope.len(),
                &notes,
                started.elapsed().as_secs_f64(),
            );
            return Ok(text_block(envelope));
        }
        let lines: Vec<String> = fused.iter().map(render_fused).collect();
        let body = lines.join("\n---\n");
        let text = if notes.is_empty() {
            body
        } else {
            format!("{}\n\n{body}", notes.join("\n"))
        };
        log_call(
            "search",
            &p.root,
            &query,
            lines.len(),
            text.len(),
            &notes,
            started.elapsed().as_secs_f64(),
        );
        Ok(text_block(text))
    }

    #[tool(
        description = "Intent search with Jev ranking inside the tool: retrieve a shortlist, score each candidate, return top-k only. Pool never enters context. Falls back to unranked retrieval if Jev is unavailable. Exact anchors still belong on `rg`."
    )]
    async fn search_ranked(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let limit = p.limit.unwrap_or(10).clamp(1, 50);
        let format = Format::opt(p.format);
        let fts: &[String] = p.fts.as_deref().unwrap_or(&[]);
        let filter = HitFilter::build(&root, p.lang.clone(), p.globs.clone())?;
        let started = std::time::Instant::now();
        let cls = route::route(&p.query, fts);
        if let route::Route::Literal(pattern) = &cls {
            return self
                .rg(Parameters(RgParams {
                    root: p.root.clone(),
                    pattern: pattern.clone(),
                    regex: Some(false),
                    structural: None,
                    case_insensitive: Some(false),
                    lang: p.lang.clone(),
                    globs: p.globs.clone(),
                    format: p.format,
                    limit: Some(limit),
                }))
                .await;
        }
        // Single identifiers go lexical even when ranked: the meter rescores
        // a BM25 shortlist instead of paying for vectors.
        let lexical_only = matches!(cls, route::Route::Identifier(_));
        let ast = match (&p.ast_pattern, &p.ast_lang) {
            (None, None) => None,
            (Some(pattern), Some(lang)) => {
                if !crate::astg::available() {
                    return Err(McpError::invalid_params(
                        "ast-grep not found on PATH (or ONE_GREP_AST_GREP)",
                        None,
                    ));
                }
                Some((pattern.clone(), lang.clone()))
            }
            _ => {
                return Err(McpError::invalid_params(
                    "ast_pattern and ast_lang must be set together",
                    None,
                ));
            }
        };
        let mut query = p.query.clone();
        if let Some(fts) = &p.fts {
            query.push(' ');
            query.push_str(&fts.join(" "));
        }
        let fuse = p.fuse.unwrap_or(true);
        let indexed = index::is_indexed(&root);
        // Default preserves history: Jev with unranked fallback. Explicit
        // backends run their own meter instead.
        let rank = p.rank.unwrap_or(crate::embed::RankKind::Jev);
        let query_for_rank = query.clone();
        let root_for_rank = root.clone();
        // Identifier queries stay lexical (router); only the fused path
        // consumes vectors, so only it can report them stale.
        let want_vectors = indexed && fuse && !lexical_only;
        let (note, hits): (String, Vec<fuse::FusedHit>) = tokio::task::spawn_blocking(move || {
            let mut hits = if !indexed || lexical_only {
                fuse::collect(&root_for_rank, &query_for_rank, limit, false, None)
            } else if fuse {
                match provider() {
                    Ok(p) => fuse::collect(&root_for_rank, &query_for_rank, limit, true, Some(p)),
                    Err(_) => fuse::collect(&root_for_rank, &query_for_rank, limit, false, None),
                }
            } else {
                fuse::collect(&root_for_rank, &query_for_rank, limit, false, None)
            }?;
            if let Some((pattern, lang)) = &ast {
                let found = crate::astg::search(&root_for_rank, pattern, lang, limit)?;
                let refs: Vec<(PathBuf, u64)> =
                    found.into_iter().map(|h| (h.path, h.line)).collect();
                fuse::apply_ast(&mut hits, &refs);
                hits.truncate(limit);
            }
            // Filter before the meter: rankers score winners only, and
            // `limit` bounds what the caller ever sees.
            hits.retain(|h| filter.keep(&h.path));
            hits.truncate(limit);
            if rank == crate::embed::RankKind::Jev {
                let (status, hits) = crate::jev::rerank_hits(&query_for_rank, hits);
                return Ok::<_, crate::Error>((status.note, hits));
            }
            let label = match rank {
                crate::embed::RankKind::Jina => "jina",
                crate::embed::RankKind::Llama => "llama",
                crate::embed::RankKind::Jev => unreachable!("jev handled above"),
            };
            let reranker: Box<dyn crate::embed::Rerank> = match rank {
                crate::embed::RankKind::Jina => match crate::embed::JinaReranker::load() {
                    Ok(r) => Box::new(r),
                    Err(e) => {
                        return Ok((format!("rank: fallback ({e})"), hits));
                    }
                },
                crate::embed::RankKind::Llama => match crate::embed::LlamaReranker::from_env() {
                    Ok(r) => Box::new(r),
                    Err(e) => {
                        return Ok((format!("rank: fallback ({e})"), hits));
                    }
                },
                crate::embed::RankKind::Jev => unreachable!("jev handled above"),
            };
            match fuse::rescore(&query_for_rank, &mut hits, reranker.as_ref()) {
                Ok(()) => Ok((format!("rank: {label}"), hits)),
                Err(e) => Ok((format!("rank: fallback ({e})"), hits)),
            }
        })
        .await
        .map_err(|e| internal(e.to_string()))?
        .map_err(internal)?;
        let mut notes = vec![note];
        if indexed && index::is_stale(&root) {
            notes.push(index::stale_note(&root));
        }
        if want_vectors && crate::vectors::is_stale(&root) {
            notes.push(crate::vectors::stale_note(&root));
        }
        if !indexed {
            notes.push(rg::unindexed_note(&root));
        }
        if let Some(envelope) = format.envelope(&notes, &hits) {
            log_call(
                "search_ranked",
                &p.root,
                &p.query,
                hits.len(),
                envelope.len(),
                &notes,
                started.elapsed().as_secs_f64(),
            );
            return Ok(text_block(envelope));
        }
        let body = hits
            .iter()
            .map(render_fused)
            .collect::<Vec<_>>()
            .join("\n---\n");
        let text = format!("{}\n\n{body}", notes.join("\n"));
        log_call(
            "search_ranked",
            &p.root,
            &p.query,
            hits.len(),
            text.len(),
            &notes,
            started.elapsed().as_secs_f64(),
        );
        Ok(text_block(text))
    }

    #[tool(
        description = "Rust definition navigation via rust-analyzer: jump from a reference to its definition. Takes a 1-based line and character. Returns path:line evidence; targets outside the workspace are marked external. Needs saved files and a resolvable rust-analyzer."
    )]
    async fn definition(
        &self,
        Parameters(p): Parameters<DefinitionParams>,
    ) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let started = std::time::Instant::now();
        let file = resolve_in_root(&root, &p.path).ok_or_else(|| {
            McpError::invalid_params(format!("path escapes workspace root: {}", p.path), None)
        })?;
        if p.line < 1 || p.character < 1 {
            return Err(McpError::invalid_params(
                "line and character are 1-based",
                None,
            ));
        }
        let options = lsp::Options {
            command: p.server.unwrap_or_else(|| lsp::DEFAULT_COMMAND.to_owned()),
            ..Default::default()
        };
        let targets = lsp::definition(&options, &root, &file, p.line, p.character)
            .await
            .map_err(internal)?;
        let format = Format::opt(p.format);
        if targets.is_empty() {
            let note = format!(
                "no definition found at {}:{}:{} (rust-analyzer)",
                file.display(),
                p.line,
                p.character
            );
            if let Some(envelope) = format.envelope(&[note.clone()], &Vec::<lsp::Target>::new()) {
                log_call(
                    "definition",
                    &p.root,
                    &p.path,
                    0,
                    envelope.len(),
                    &[],
                    started.elapsed().as_secs_f64(),
                );
                return Ok(text_block(envelope));
            }
            log_call(
                "definition",
                &p.root,
                &p.path,
                0,
                note.len(),
                &[],
                started.elapsed().as_secs_f64(),
            );
            return Ok(text_block(note));
        }
        if let Some(envelope) = format.envelope(&[], &targets) {
            log_call(
                "definition",
                &p.root,
                &p.path,
                targets.len(),
                envelope.len(),
                &[],
                started.elapsed().as_secs_f64(),
            );
            return Ok(text_block(envelope));
        }
        let lines: Vec<String> = targets
            .iter()
            .map(|t| {
                let scope = if t.outside_root {
                    "external"
                } else {
                    "workspace"
                };
                format!(
                    "{}:{}-{} ({scope})",
                    t.path.display(),
                    t.start_line,
                    t.end_line
                )
            })
            .collect();
        let text = lines.join("\n");
        log_call(
            "definition",
            &p.root,
            &p.path,
            targets.len(),
            text.len(),
            &[],
            started.elapsed().as_secs_f64(),
        );
        Ok(text_block(text))
    }

    #[tool(
        description = "Exact text or regex search over workspace files (no index needed). Gitignore-aware. Returns path:line:text hits. Pass raw pattern text without shell quotes; a single outer \"...\" or '...' pair is stripped. Literal unless `regex` is true. `lang` restricts to languages (rust, python, typescript, go, java, nix, markdown); `globs` are include globs (`!` negates). `structural` runs the pattern through ast-grep instead (needs exactly one `lang`)."
    )]
    async fn rg(&self, Parameters(p): Parameters<RgParams>) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let started = std::time::Instant::now();
        let format = Format::opt(p.format);
        if p.structural.unwrap_or(false) {
            if p.regex.unwrap_or(false) {
                return Err(McpError::invalid_params(
                    "`structural` and `regex` cannot be combined",
                    None,
                ));
            }
            let langs = p.lang.clone().unwrap_or_default();
            let [lang] = langs.as_slice() else {
                return Err(McpError::invalid_params(
                    "`structural` needs exactly one `lang`",
                    None,
                ));
            };
            if !crate::astg::available() {
                return Err(McpError::invalid_params(
                    "ast-grep not found on PATH (or ONE_GREP_AST_GREP)",
                    None,
                ));
            }
            let limit = p.limit.unwrap_or(100).clamp(1, 500);
            let hits = crate::astg::search(&root, &p.pattern, lang, limit).map_err(rg_error)?;
            if let Some(envelope) = format.envelope(&[], &hits) {
                log_call(
                    "rg",
                    &p.root,
                    &p.pattern,
                    hits.len(),
                    envelope.len(),
                    &[],
                    started.elapsed().as_secs_f64(),
                );
                return Ok(text_block(envelope));
            }
            let lines: Vec<String> = hits
                .iter()
                .map(|h| {
                    format!(
                        "{}:{}:{}",
                        h.path.display(),
                        h.line,
                        h.text.lines().next().unwrap_or("")
                    )
                })
                .collect();
            let text = lines.join("\n");
            log_call(
                "rg",
                &p.root,
                &p.pattern,
                hits.len(),
                text.len(),
                &[],
                started.elapsed().as_secs_f64(),
            );
            return Ok(text_block(text));
        }
        let options = rg::Options {
            regex: p.regex.unwrap_or(false),
            case_insensitive: p.case_insensitive.unwrap_or(false),
            globs: p.globs.unwrap_or_default(),
            langs: p.lang.unwrap_or_default(),
            limit: p.limit.unwrap_or(100).clamp(1, 500),
        };
        let hits = rg::search(&root, &p.pattern, &options).map_err(rg_error)?;
        if let Some(envelope) = format.envelope(&[], &hits) {
            return Ok(text_block(envelope));
        }
        let lines: Vec<String> = hits
            .iter()
            .map(|h| format!("{}:{}:{}", h.path.display(), h.line, h.text))
            .collect();
        let text = lines.join("\n");
        log_call(
            "rg",
            &p.root,
            &p.pattern,
            hits.len(),
            text.len(),
            &[],
            started.elapsed().as_secs_f64(),
        );
        Ok(text_block(text))
    }

    #[tool(
        description = "Route a task to the best matching agent skill from the on-disk skill library. Jev ranks inside the tool when a TypeSafe key is configured; otherwise lexical fallback with overlap floor. Returns skill name, absolute path to SKILL.md, score, and a one-line description — never the skill body."
    )]
    async fn skill(
        &self,
        Parameters(p): Parameters<SkillParams>,
    ) -> Result<CallToolResult, McpError> {
        let started = std::time::Instant::now();
        let limit = p.limit.unwrap_or(2).clamp(1, 5);
        let dir_for_log = p.dir.clone().unwrap_or_default();
        let dir_path = p.dir.as_deref().map(PathBuf::from);
        let format = Format::opt(p.format);
        let task = p.task.clone();
        let result = tokio::task::spawn_blocking(move || {
            crate::skill::run(&task, dir_path.as_deref(), limit)
        })
        .await
        .map_err(|e| internal(e.to_string()))?;
        let mut notes = result.notes.clone();
        notes.insert(0, result.rank_note.clone());
        if let Some(envelope) = format.envelope(
            &notes,
            &serde_json::json!({
                "no_match": result.no_match,
                "hits": result.hits,
            }),
        ) {
            log_call(
                "skill",
                &dir_for_log,
                &p.task,
                result.hits.len(),
                envelope.len(),
                &notes,
                started.elapsed().as_secs_f64(),
            );
            return Ok(text_block(envelope));
        }
        let text = result.format_text();
        log_call(
            "skill",
            &dir_for_log,
            &p.task,
            result.hits.len(),
            text.len(),
            &notes,
            started.elapsed().as_secs_f64(),
        );
        Ok(text_block(text))
    }
}

impl Default for OneGrep {
    fn default() -> Self {
        Self::new()
    }
}

/// Post-retrieval `--lang` / `--glob` filter for `search` hits.
/// Validation fails fast (unknown langs, bad globs) as caller errors.
#[derive(Clone, Debug)]
struct HitFilter {
    langs: Vec<String>,
    globs: rg::GlobFilter,
}

impl HitFilter {
    fn build(
        root: &Path,
        lang: Option<Vec<String>>,
        globs: Option<Vec<String>>,
    ) -> Result<Self, McpError> {
        let langs = lang.unwrap_or_default();
        rg::validate_langs(&langs).map_err(rg_error)?;
        let globs = rg::GlobFilter::build(root, &globs.unwrap_or_default())
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        Ok(Self { langs, globs })
    }

    fn keep(&self, path: &Path) -> bool {
        self.globs.keep(path) && rg::matches_lang(path, &self.langs).unwrap_or(false)
    }
}

fn render_ranked(h: &index::RankedHit) -> String {
    render(
        &h.path,
        h.start,
        h.end,
        &h.breadcrumb,
        &format!("{:.2}", h.score),
        "bm25",
        &h.text,
    )
}

fn render_fused(h: &fuse::FusedHit) -> String {
    render(
        &h.path,
        h.start,
        h.end,
        &h.breadcrumb,
        &format!("{:.4}", h.score),
        fuse::source(h),
        &h.text,
    )
}

fn render_live(h: &rg::Hit) -> String {
    render(
        &h.path,
        h.line,
        h.line,
        "rg-fallback",
        "live",
        "rg",
        &h.text,
    )
}

#[tool_handler]
impl rmcp::ServerHandler for OneGrep {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        // Stamp the package version so harness MCP panels show whether the
        // running server matches the installed binary (rebuilds need a
        // consumer restart; the version makes staleness visible).
        info.server_info.version = env!("CARGO_PKG_VERSION").to_owned();
        info.instructions = Some(
            "Local-first hybrid workspace search. Prefer `search_ranked` for intent \
             (retrieve + Jev inside the tool; only top-k winners enter context). \
             Use `search` for the raw fused pool. Use `rg` for exact text, symbols, \
             or regex. Use `definition` to jump from a Rust reference to its \
             definition. Need a specialized skill (API docs, domain workflow)? Call \
             `skill` with the task first. Cite path:start-end evidence; to read more, \
             read only the cited line range, never the whole file."
                .into(),
        );
        info
    }
}

/// Serve over stdio (local MCP clients).
///
/// # Errors
///
/// Returns [`crate::Error`] when transport setup fails.
pub async fn serve_stdio() -> Result<(), crate::Error> {
    OneGrep::new()
        .serve(stdio())
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))?
        .waiting()
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))?;
    Ok(())
}

/// Serve Streamable HTTP on 127.0.0.1 with bearer auth.
///
/// # Errors
///
/// Returns [`crate::Error`] when binding or serving fails.
pub async fn serve_http(port: u16, token: &str) -> Result<(), crate::Error> {
    let service = StreamableHttpService::new(
        || Ok(OneGrep::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    );
    let token = token.to_owned();
    let app = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn(
            move |req: axum::http::Request<axum::body::Body>, next: axum::middleware::Next| {
                let token = token.clone();
                async move {
                    let ok = req
                        .headers()
                        .get(axum::http::header::AUTHORIZATION)
                        .and_then(|v| v.to_str().ok())
                        == Some(format!("Bearer {token}").as_str());
                    if ok {
                        Ok::<_, (axum::http::StatusCode, String)>(next.run(req).await)
                    } else {
                        Err((
                            axum::http::StatusCode::UNAUTHORIZED,
                            "missing or invalid bearer token".to_owned(),
                        ))
                    }
                }
            },
        ));
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))?;
    tracing::info!("serving MCP on 127.0.0.1:{port}/mcp");
    axum::serve(listener, app)
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_info_carries_package_version() {
        use rmcp::ServerHandler as _;
        let info = OneGrep::new().get_info();
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
        assert!(!info.server_info.version.is_empty());
    }

    #[test]
    fn tool_router_lists_skill() {
        assert!(OneGrep::new().tool_router.map.contains_key("skill"));
    }

    #[tokio::test]
    async fn skill_tool_output_excludes_skill_body() {
        let library = tempfile::tempdir().expect("library");
        crate::skill::write_fixture(library.path()).expect("fixture");
        let home = tempfile::tempdir().expect("home");
        let prev_home = std::env::var_os("HOME");
        let prev_key = std::env::var(crate::jev::ENV_API_KEY).ok();
        unsafe {
            std::env::set_var("HOME", home.path());
            std::env::remove_var(crate::jev::ENV_API_KEY);
        }
        let result = OneGrep::new()
            .skill(Parameters(SkillParams {
                task: "WinRT API reference".into(),
                limit: Some(3),
                dir: Some(library.path().to_string_lossy().into_owned()),
                format: None,
            }))
            .await
            .expect("skill");
        unsafe {
            match prev_key {
                Some(v) => std::env::set_var(crate::jev::ENV_API_KEY, v),
                None => std::env::remove_var(crate::jev::ENV_API_KEY),
            }
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("winrt-lookup"), "{body}");
        assert!(!body.contains("SECRET_BODY_MARKER"), "{body}");
        assert!(
            body.contains("/quoted-skill/SKILL.md"),
            "expected absolute path in output: {body}"
        );
        let path_line = body
            .lines()
            .find(|l| l.contains("winrt-lookup"))
            .expect("hit line");
        let path_token = path_line.split_whitespace().nth(1).expect("path");
        assert!(Path::new(path_token).is_absolute(), "{path_token}");
    }

    #[test]
    fn literal_and_identifier_take_different_paths_unindexed() {
        // Branch-observable routing contract (no index, so no model load):
        // literals return bare rg hits; identifiers return rg-fallback
        // lines with the indexing note.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.txt"), "supersonic_ferret\n").expect("write");
        let root = workspace.path().to_string_lossy().into_owned();
        let body = |query: &str| {
            let result = rt
                .block_on(OneGrep::new().search(Parameters(SearchParams {
                    root: root.clone(),
                    query: query.into(),
                    fts: None,
                    fuse: Some(false),
                    rank: None,
                    lang: None,
                    globs: None,
                    ast_pattern: None,
                    ast_lang: None,
                    format: None,
                    limit: None,
                })))
                .expect("search");
            let text = serde_json::to_value(result).expect("response");
            text["content"][0]["text"].as_str().unwrap().to_owned()
        };
        let literal = body("\"supersonic_ferret\"");
        assert!(literal.contains("a.txt:1:supersonic_ferret"), "{literal}");
        assert!(!literal.contains("Indexing is available"), "{literal}");
        let identifier = body("supersonic_ferret");
        assert!(identifier.contains("supersonic_ferret"), "{identifier}");
        assert!(identifier.contains("Indexing is available"), "{identifier}");
    }

    #[tokio::test]
    async fn intent_and_mixed_queries_use_rg_fallback_when_unindexed() {
        let workspace = tempfile::tempdir().expect("workspace");
        for (query, fts) in [
            ("where is authentication handled", None),
            ("foo::Bar", Some(vec!["implementation".into()])),
            ("find definition of foo::Bar", None),
        ] {
            let result = OneGrep::new()
                .search(Parameters(SearchParams {
                    root: workspace.path().to_string_lossy().into_owned(),
                    query: query.into(),
                    fts,
                    fuse: Some(false),
                    rank: None,
                    lang: None,
                    globs: None,
                    ast_pattern: None,
                    ast_lang: None,
                    format: None,
                    limit: None,
                }))
                .await
                .expect("unindexed intent search falls back to live rg");
            let text = serde_json::to_value(result).expect("response");
            let body = text["content"][0]["text"].as_str().unwrap();
            assert!(body.contains("Indexing is available"), "{query}: {body}");
        }
    }

    #[tokio::test]
    async fn search_ranked_unindexed_emits_rank_header() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.txt"), "supersonic_ferret\n").expect("write");
        let result = OneGrep::new()
            .search_ranked(Parameters(SearchParams {
                root: workspace.path().to_string_lossy().into_owned(),
                query: "where is supersonic_ferret".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: None,
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: None,
                limit: Some(5),
            }))
            .await
            .expect("ranked search");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("rank:"), "{body}");
        assert!(body.contains("Indexing is available"), "{body}");
    }

    #[test]
    fn definition_paths_stay_inside_the_workspace() {
        let root = Path::new("/ws");
        assert_eq!(
            resolve_in_root(root, "src/main.rs"),
            Some(PathBuf::from("/ws/src/main.rs"))
        );
        assert_eq!(
            resolve_in_root(root, "/ws/src/main.rs"),
            Some(PathBuf::from("/ws/src/main.rs"))
        );
        assert_eq!(resolve_in_root(root, "../escape.rs"), None);
        assert_eq!(resolve_in_root(root, "a/../../escape.rs"), None);
        assert_eq!(resolve_in_root(root, "/etc/passwd"), None);
    }

    #[tokio::test]
    async fn definition_rejects_bad_locations_before_spawning() {
        let workspace = tempfile::tempdir().expect("workspace");
        let root = workspace.path().to_string_lossy().into_owned();
        let err = OneGrep::new()
            .definition(Parameters(DefinitionParams {
                root: root.clone(),
                path: "../escape.rs".into(),
                line: 1,
                character: 1,
                server: None,
                format: None,
            }))
            .await
            .expect_err("escape must fail");
        assert!(err.message.contains("escapes"), "{err:?}");
        let err = OneGrep::new()
            .definition(Parameters(DefinitionParams {
                root,
                path: "a.rs".into(),
                line: 0,
                character: 1,
                server: None,
                format: None,
            }))
            .await
            .expect_err("line 0 must fail");
        assert!(err.message.contains("1-based"), "{err:?}");
    }

    #[tokio::test]
    async fn definition_missing_server_is_an_error_not_empty() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.rs"), "fn a() {}\n").expect("fixture");
        let err = OneGrep::new()
            .definition(Parameters(DefinitionParams {
                root: workspace.path().to_string_lossy().into_owned(),
                path: "a.rs".into(),
                line: 1,
                character: 1,
                server: Some("one-grep-definitely-no-such-server".into()),
                format: None,
            }))
            .await
            .expect_err("missing server must fail");
        assert!(err.message.contains("could not start"), "{err:?}");
    }

    #[tokio::test]
    async fn exact_search_routes_to_rg_without_an_index() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("view.txt"),
            "foo::Bar\nx:Name=\"SaveButton\"\n{Binding User.Name}\n",
        )
        .expect("fixture");
        for (query, expected) in [
            ("foo::Bar", "view.txt:1:foo::Bar"),
            ("\"x:Name\"", "view.txt:2:x:Name"),
            ("\"{Binding User.Name}\"", "view.txt:3:{Binding User.Name}"),
        ] {
            let result = OneGrep::new()
                .search(Parameters(SearchParams {
                    root: workspace.path().to_string_lossy().into_owned(),
                    query: query.into(),
                    fts: None,
                    fuse: Some(false),
                    rank: None,
                    lang: None,
                    globs: None,
                    ast_pattern: None,
                    ast_lang: None,
                    format: None,
                    limit: None,
                }))
                .await
                .expect("exact search needs no index");
            let text = serde_json::to_value(result).expect("response");
            assert!(
                text["content"][0]["text"]
                    .as_str()
                    .unwrap()
                    .contains(expected)
            );
        }
    }

    /// Forall shape (Kani: arbitrary path string, assert no escape): every
    /// `..` that would leave `root` resolves to `None`. Sampled below.
    #[test]
    fn resolve_in_root_never_escapes() {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path();
        for raw in [
            "../escape.rs",
            "a/../../escape.rs",
            "/etc/passwd",
            "..",
            "sub/../../../x",
        ] {
            assert!(resolve_in_root(root, raw).is_none(), "{raw}");
        }
        for raw in ["a.rs", "sub/b.rs", "./a.rs"] {
            let resolved = resolve_in_root(root, raw).expect("inside");
            assert!(resolved.starts_with(root), "{raw}");
        }
    }

    #[test]
    fn rg_error_maps_caller_mistakes_to_invalid_params() {
        let err = rg_error(crate::Error::InvalidInput("unknown --lang `cobol`".into()));
        assert!(err.message.contains("cobol"));
        let err = rg_error(crate::Error::InvalidInput("x".into()));
        assert!(err.message.contains('x'));
    }

    #[tokio::test]
    async fn rg_lang_filter_reaches_the_handler() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("main.rs"), "needle\n").expect("fixture");
        std::fs::write(workspace.path().join("notes.txt"), "needle\n").expect("fixture");
        let result = OneGrep::new()
            .rg(Parameters(RgParams {
                root: workspace.path().to_string_lossy().into_owned(),
                pattern: "needle".into(),
                regex: None,
                structural: None,
                case_insensitive: None,
                lang: Some(vec!["rust".into()]),
                globs: None,
                format: None,
                limit: None,
            }))
            .await
            .expect("rg");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("main.rs"), "{body}");
        assert!(!body.contains("notes.txt"), "{body}");
        // Unknown language fails closed as invalid params, not empty output.
        let err = OneGrep::new()
            .rg(Parameters(RgParams {
                root: workspace.path().to_string_lossy().into_owned(),
                pattern: "needle".into(),
                regex: None,
                structural: None,
                case_insensitive: None,
                lang: Some(vec!["cobol".into()]),
                globs: None,
                format: None,
                limit: None,
            }))
            .await
            .expect_err("unknown lang must fail");
        assert!(err.message.contains("cobol"), "{err:?}");
    }

    /// Offline default (1.4): with no key anywhere, `search_ranked` makes
    /// no network attempt and preserves retrieval order. The sandbox
    /// redirects HOME to an empty dir so key files cannot leak in.
    #[tokio::test]
    async fn search_ranked_offline_preserves_retrieval_order() {
        let home = tempfile::tempdir().expect("home");
        let prev_home = std::env::var_os("HOME");
        let prev_key = std::env::var_os(crate::jev::ENV_API_KEY);
        // SAFETY: restored below; Jev callers fail closed to the same
        // fallback order on error, so concurrent tests keep their verdicts.
        unsafe {
            std::env::set_var("HOME", home.path());
            std::env::remove_var(crate::jev::ENV_API_KEY);
        }
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("a.txt"),
            "offline canary bravo\ncanary alpha\n",
        )
        .expect("fixture");
        std::fs::write(workspace.path().join("b.txt"), "canary charlie\n").expect("fixture");
        crate::index::sync(workspace.path()).expect("sync");
        let root = workspace.path().to_string_lossy().into_owned();
        let expected: Vec<String> =
            crate::fuse::collect(workspace.path(), "canary query terms here", 5, false, None)
                .expect("collect")
                .iter()
                .map(|h| h.path.to_string_lossy().into_owned())
                .collect();
        let result = OneGrep::new()
            .search_ranked(Parameters(SearchParams {
                root,
                query: "canary query terms here".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: None,
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: None,
                limit: Some(5),
            }))
            .await
            .expect("ranked");
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            if let Some(v) = prev_key {
                std::env::set_var(crate::jev::ENV_API_KEY, v);
            }
        }
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("rank: fallback"), "{body}");
        let mut cursor = 0;
        for path in &expected {
            let rel = cursor;
            let found = body[rel..].find(path.as_str()).map(|i| rel + i);
            let pos = found.unwrap_or_else(|| panic!("{path} missing in order: {body}"));
            assert!(pos >= cursor, "{path} out of order: {body}");
            cursor = pos + path.len();
        }
    }

    #[tokio::test]
    async fn search_lang_filter_applies_to_lexical_hits() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("main.rs"),
            "needle_haystack_unique here\n",
        )
        .expect("fixture");
        std::fs::write(
            workspace.path().join("notes.txt"),
            "needle_haystack_unique here\n",
        )
        .expect("fixture");
        crate::index::sync(workspace.path()).expect("sync");
        let root = workspace.path().to_string_lossy().into_owned();
        let result = OneGrep::new()
            .search(Parameters(SearchParams {
                root: root.clone(),
                query: "needle_haystack_unique".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: Some(vec!["rust".into()]),
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: None,
                limit: None,
            }))
            .await
            .expect("search");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("main.rs"), "{body}");
        assert!(!body.contains("notes.txt"), "{body}");
        let err = OneGrep::new()
            .search(Parameters(SearchParams {
                root,
                query: "needle_haystack_unique".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: Some(vec!["cobol".into()]),
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: None,
                limit: None,
            }))
            .await
            .expect_err("unknown lang must fail fast");
        assert!(err.message.contains("cobol"), "{err:?}");
    }

    /// Output budget (4.1): at most `limit` chunks, each text capped at
    /// TEXT_CAP, every header one line carrying a source tag.
    #[tokio::test]
    async fn search_output_respects_chunk_budget() {
        let workspace = tempfile::tempdir().expect("workspace");
        for name in ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"] {
            let filler: String = "budgettoken filler prose ".repeat(200);
            std::fs::write(
                workspace.path().join(name),
                format!("budgettoken\n{filler}\n"),
            )
            .expect("fixture");
        }
        crate::index::sync(workspace.path()).expect("sync");
        let result = OneGrep::new()
            .search(Parameters(SearchParams {
                root: workspace.path().to_string_lossy().into_owned(),
                query: "budgettoken filler prose documentation".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: None,
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: None,
                limit: Some(3),
            }))
            .await
            .expect("search");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        let chunks: Vec<&str> = body.split("\n---\n").collect();
        assert_eq!(chunks.len(), 3, "{body}");
        for chunk in &chunks {
            let (header, text) = chunk.split_once('\n').expect("header + text");
            assert!(!header.contains("---"), "{header}");
            let sources = ["source=bm25", "source=vec", "source=bm25+vec", "source=rg"];
            assert!(
                sources.iter().any(|s| header.contains(s)),
                "no source tag: {header}"
            );
            assert!(text.chars().count() <= TEXT_CAP, "cap blown: {header}");
        }
    }

    /// Fake `ast-grep` binary printing `stdout` for any invocation.
    fn fake_ast_grep(bin: &Path, stdout: &str) {
        let script = bin.join("ast-grep");
        std::fs::write(&script, format!("#!/bin/sh\nprintf '%s' '{stdout}'\n")).expect("script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }
        // SAFETY: under `env_lock`; restored by the caller.
        unsafe {
            std::env::set_var(crate::astg::ENV_BIN, script.to_string_lossy().into_owned());
        }
    }

    fn unpoint_ast_grep() {
        // SAFETY: under `env_lock`; restores ambient state.
        unsafe {
            std::env::remove_var(crate::astg::ENV_BIN);
        }
    }

    #[tokio::test]
    async fn rg_structural_uses_ast_grep_when_present() {
        let _guard = crate::astg::tests::env_lock();
        let bin = tempfile::tempdir().expect("tempdir");
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.rs"), "fn apply() {}\n").expect("fixture");
        let file = workspace.path().join("a.rs").to_string_lossy().into_owned();
        fake_ast_grep(
            bin.path(),
            &format!(
                r#"[{{"file": "{file}", "range": {{"start": {{"line": 0}}}}, "text": "fn apply() {{}}"}}]"#
            ),
        );
        let result = OneGrep::new()
            .rg(Parameters(RgParams {
                root: workspace.path().to_string_lossy().into_owned(),
                pattern: "fn $F".into(),
                regex: None,
                structural: Some(true),
                case_insensitive: None,
                lang: Some(vec!["rust".into()]),
                globs: None,
                format: None,
                limit: None,
            }))
            .await
            .expect("structural rg");
        unpoint_ast_grep();
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("a.rs:1:fn apply()"), "{body}");
    }

    /// `rank=llama` scores through a llama.cpp `--rerank` server and
    /// reports its own header; a dead endpoint fails closed to retrieval
    /// order instead of erroring the search.
    #[tokio::test]
    async fn search_ranked_llama_backend_round_trips() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.txt"), "llama probe alpha\n").expect("fixture");
        std::fs::write(workspace.path().join("b.txt"), "llama probe bravo\n").expect("fixture");
        crate::index::sync(workspace.path()).expect("sync");
        let root = workspace.path().to_string_lossy().into_owned();
        let url = fake_llama_rerank(
            r#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#,
        );
        let prev = std::env::var_os(crate::embed::ENV_RERANK_URL);
        // SAFETY: distinct variable, restored below; other tests do not
        // read it (Jev paths use TYPESAFE_*).
        unsafe {
            std::env::set_var(crate::embed::ENV_RERANK_URL, &url);
        }
        let server = OneGrep::new();
        let ranked = |rank| {
            server.search_ranked(Parameters(SearchParams {
                root: root.clone(),
                query: "llama probe documentation".into(),
                fts: None,
                fuse: Some(false),
                rank,
                lang: None,
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: None,
                limit: Some(5),
            }))
        };
        let result = ranked(Some(crate::embed::RankKind::Llama))
            .await
            .expect("ranked");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("rank: llama"), "{body}");
        // Mock scores index 1 at 0.9 and index 0 at 0.1: winners first
        // regardless of which file the index returned first.
        assert!(
            body.find("(0.9000)").unwrap() < body.find("(0.1000)").unwrap(),
            "{body}"
        );
        assert!(body.contains("a.txt") && body.contains("b.txt"), "{body}");
        // Dead endpoint: fallback header, retrieval order, still success.
        unsafe {
            std::env::set_var(crate::embed::ENV_RERANK_URL, "http://127.0.0.1:9");
        }
        let result = ranked(Some(crate::embed::RankKind::Llama))
            .await
            .expect("ranked");
        unsafe {
            match prev {
                Some(v) => std::env::set_var(crate::embed::ENV_RERANK_URL, v),
                None => std::env::remove_var(crate::embed::ENV_RERANK_URL),
            }
        }
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("rank: fallback"), "{body}");
    }

    #[tokio::test]
    async fn search_ast_pattern_fuses_third_list() {
        let _guard = crate::astg::tests::env_lock();
        let bin = tempfile::tempdir().expect("tempdir");
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("a.rs"),
            "fn apply() {}\n// structural needle prose here\n",
        )
        .expect("fixture");
        crate::index::sync(workspace.path()).expect("sync");
        let file = workspace.path().join("a.rs").to_string_lossy().into_owned();
        fake_ast_grep(
            bin.path(),
            &format!(
                r#"[{{"file": "{file}", "range": {{"start": {{"line": 0}}}}, "text": "fn apply() {{}}"}}]"#
            ),
        );
        let root = workspace.path().to_string_lossy().into_owned();
        let result = OneGrep::new()
            .search(Parameters(SearchParams {
                root: root.clone(),
                query: "structural needle prose documentation".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: None,
                globs: None,
                ast_pattern: Some("fn $F".into()),
                ast_lang: Some("rust".into()),
                format: None,
                limit: None,
            }))
            .await
            .expect("search");
        unpoint_ast_grep();
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("a.rs"), "{body}");
        // No binary, explicit pattern: fail closed, not silent skip.
        // Point at a missing binary (not unpoint) so the test holds even
        // where a real ast-grep is installed.
        unsafe {
            std::env::set_var(crate::astg::ENV_BIN, "/no/such/ast-grep-binary");
        }
        let err = OneGrep::new()
            .search(Parameters(SearchParams {
                root,
                query: "structural needle prose documentation".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: None,
                globs: None,
                ast_pattern: Some("fn $F".into()),
                ast_lang: Some("rust".into()),
                format: None,
                limit: None,
            }))
            .await
            .expect_err("absent ast-grep must fail");
        unpoint_ast_grep();
        assert!(err.message.contains("ast-grep"), "{err:?}");
    }

    /// std-only fake llama-server: one canned `/v1/rerank` response.
    /// Returns the base URL. Distinct env var from the ast-grep fakes,
    /// so no lock is shared with them.
    fn fake_llama_rerank(body: &'static str) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = vec![0u8; 65536];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).expect("write");
        });
        format!("http://{addr}")
    }

    /// Serving log: one JSONL row per call, opt-out path via env.
    /// Proves the serving side of "hits → did they still read?".
    #[tokio::test]
    async fn serving_log_appends_json_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("serving.log");
        // SAFETY: unique path in this var; no other test reads it.
        unsafe {
            std::env::set_var(ENV_SERVING_LOG, log.to_string_lossy().into_owned());
        }
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.txt"), "logged_token here\n").expect("fixture");
        OneGrep::new()
            .rg(Parameters(RgParams {
                root: workspace.path().to_string_lossy().into_owned(),
                pattern: "logged_token".into(),
                regex: None,
                structural: None,
                case_insensitive: None,
                lang: None,
                globs: None,
                format: None,
                limit: None,
            }))
            .await
            .expect("rg");
        unsafe {
            std::env::remove_var(ENV_SERVING_LOG);
        }
        let content = std::fs::read_to_string(&log).expect("log written");
        let row: serde_json::Value = serde_json::from_str(content.trim()).expect("json row");
        assert_eq!(row["tool"], "rg");
        assert_eq!(row["query"], "logged_token");
        assert_eq!(row["hits"], 1);
        assert!(row["chars"].as_u64().unwrap() > 0);
        assert!(row["latency_ms"].as_f64().unwrap() >= 0.0);
    }

    /// `format=json` returns a machine envelope on every tool; text stays
    /// the default. Notes ride along so nothing agents rely on is lost.
    #[tokio::test]
    async fn format_json_envelope_round_trips() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.rs"), "jsonprobe_token here\n").expect("fixture");
        crate::index::sync(workspace.path()).expect("sync");
        let root = workspace.path().to_string_lossy().into_owned();
        // search (identifier path): notes + typed hits.
        let result = OneGrep::new()
            .search(Parameters(SearchParams {
                root: root.clone(),
                query: "jsonprobe_token".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: None,
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: Some(Format::Json),
                limit: None,
            }))
            .await
            .expect("search json");
        let text = serde_json::to_value(result).expect("response");
        let envelope: serde_json::Value =
            serde_json::from_str(text["content"][0]["text"].as_str().unwrap())
                .expect("envelope parses");
        assert_eq!(envelope["hits"].as_array().unwrap().len(), 1);
        assert!(
            envelope["hits"][0]["path"]
                .as_str()
                .unwrap()
                .ends_with("a.rs")
        );
        // rg: same envelope, line hits.
        let result = OneGrep::new()
            .rg(Parameters(RgParams {
                root: root.clone(),
                pattern: "jsonprobe_token".into(),
                regex: None,
                structural: None,
                case_insensitive: None,
                lang: None,
                globs: None,
                format: Some(Format::Json),
                limit: None,
            }))
            .await
            .expect("rg json");
        let text = serde_json::to_value(result).expect("response");
        let envelope: serde_json::Value =
            serde_json::from_str(text["content"][0]["text"].as_str().unwrap())
                .expect("envelope parses");
        assert_eq!(envelope["hits"][0]["line"], 1);
        // search_ranked offline: rank note preserved inside the envelope.
        let result = OneGrep::new()
            .search_ranked(Parameters(SearchParams {
                root: root.clone(),
                query: "jsonprobe_token prose documentation".into(),
                fts: None,
                fuse: Some(false),
                rank: None,
                lang: None,
                globs: None,
                ast_pattern: None,
                ast_lang: None,
                format: Some(Format::Json),
                limit: Some(5),
            }))
            .await
            .expect("ranked json");
        let text = serde_json::to_value(result).expect("response");
        let envelope: serde_json::Value =
            serde_json::from_str(text["content"][0]["text"].as_str().unwrap())
                .expect("envelope parses");
        assert!(
            envelope["notes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|n| n.as_str().unwrap().contains("rank:")),
            "{envelope}"
        );
    }
}
