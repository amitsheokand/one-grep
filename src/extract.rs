//! Extraction + chunking: symbols for code, sections for prose.
//!
//! Each [`Chunk`] carries its source location so indexed hits can cite
//! `path:start-end`. Supports tree-sitter symbols for known languages,
//! heading sections for Markdown, and fixed windows as fallback.

use std::path::{Path, PathBuf};

use crate::Error;

/// Extractor version. Bump when chunking rules change; the index sync
/// treats a mismatch as a full reindex (chunk IDs of unchanged content
/// stay stable, so vectors re-embed incrementally).
pub const EXTRACT_VERSION: &str = "2";

/// How a chunk was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    /// A language symbol (function, struct, class, ...).
    Symbol,
    /// A Markdown section.
    Section,
    /// Fixed-size line window (fallback).
    Window,
}

/// One retrievable unit of a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// File the chunk came from, as passed in.
    pub path: PathBuf,
    /// 1-based first line.
    pub start: u64,
    /// 1-based last line (inclusive).
    pub end: u64,
    /// How the chunk was derived.
    pub kind: ChunkKind,
    /// Symbol path (`mod > Type > method`) or heading stack.
    pub breadcrumb: String,
    /// Chunk text including the signature/heading lines.
    pub text: String,
}

/// Read `path` and extract chunks, dispatching on file extension.
///
/// Binary files (containing NUL) yield no chunks.
///
/// # Errors
///
/// Returns [`Error`] when the file cannot be read.
pub fn extract_file(path: &Path) -> Result<Vec<Chunk>, Error> {
    extract_file_with(path, DEFAULT_WINDOW)
}

/// Lines longer than this mark a file as generated/data: skipped.
const MAX_LINE_CHARS: usize = 8192;

/// Read `path` and extract with an explicit fallback window.
///
/// Binary files and generated files (huge lines) yield no chunks.
///
/// # Errors
///
/// Returns [`Error`] when the file cannot be read.
pub fn extract_file_with(path: &Path, window: Window) -> Result<Vec<Chunk>, Error> {
    let bytes = std::fs::read(path)?;
    if bytes.contains(&0) {
        return Ok(Vec::new());
    }
    let text = String::from_utf8_lossy(&bytes);
    if text.lines().any(|l| l.len() > MAX_LINE_CHARS) {
        return Ok(Vec::new());
    }
    Ok(extract_with(path, &text, window))
}

/// Extract chunks from in-memory `text` for `path`, dispatching on extension.
#[must_use]
pub fn extract(path: &Path, text: &str) -> Vec<Chunk> {
    extract_with(path, text, DEFAULT_WINDOW)
}

/// Window `(size, step)` for fallback chunking.
pub type Window = (usize, usize);

/// Default fallback window: 150 lines, 15 overlap.
pub const DEFAULT_WINDOW: Window = (150, 135);
/// Tight fallback window for embedding: 50 lines, 10 overlap.
pub const EMBED_WINDOW: Window = (50, 40);
/// File-header comment blocks longer than this are cut: license boilerplate
/// is searchable enough in its first lines (Run 12: headers must exist,
/// but need not be whole).
pub const HEADER_CAP_LINES: usize = 30;
/// Symbols longer than this are split (Run 12: 130–250-line symbols dilute
/// BM25 length-norm and mismatch the 50-line embed windows). Parts keep
/// the parent breadcrumb with a `[i/n]` suffix. Sections stay whole:
/// splitting prose breaks section coherence.
pub const SYMBOL_CAP_LINES: usize = 80;
/// Step for symbol splits (15-line overlap, like the fallback windows).
pub const SYMBOL_STEP_LINES: usize = 65;

/// Leading `#` / `//` comment block as its own chunk (Run 12: file-header
/// prose like "Exclusive MLX lane switcher" otherwise vanishes — symbol
/// extractors only attach comments to bindings). Skips shebangs and blank
/// preamble; Markdown is excluded (headings are content there).
fn header_chunk(path: &Path, text: &str, extension: &str) -> Option<Chunk> {
    let marker: &str = match extension {
        "rs" | "ts" | "tsx" | "mts" | "cts" | "js" | "jsx" | "mjs" | "cjs" | "go" | "java" => "//",
        "py" | "nix" | "sh" | "bash" | "zsh" => "#",
        _ => return None,
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut first = None;
    let mut last = 0;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("#!") {
            continue;
        }
        if trimmed.starts_with(marker) {
            if first.is_none() {
                first = Some(i);
            }
            last = i;
            continue;
        }
        break;
    }
    let start = first?;
    let end = last.min(start + HEADER_CAP_LINES - 1);
    Some(Chunk {
        path: path.to_path_buf(),
        start: start as u64 + 1,
        end: end as u64 + 1,
        kind: ChunkKind::Section,
        breadcrumb: "(header)".to_owned(),
        text: lines[start..=end].join("\n"),
    })
}

/// Extract with an explicit fallback window (symbols/sections unaffected).
#[must_use]
pub fn extract_with(path: &Path, text: &str, window: Window) -> Vec<Chunk> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let mut chunks = match extension {
        "rs" => symbols(
            path,
            text,
            tree_sitter_rust::LANGUAGE.into(),
            &[
                "function_item",
                "struct_item",
                "enum_item",
                "union_item",
                "trait_item",
                "impl_item",
                "mod_item",
                "const_item",
                "static_item",
                "type_item",
                "macro_definition",
            ],
            window,
        ),
        "py" => symbols(
            path,
            text,
            tree_sitter_python::LANGUAGE.into(),
            &["function_definition", "class_definition"],
            window,
        ),
        "ts" | "mts" | "cts" => symbols(
            path,
            text,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            &[
                "function_declaration",
                "generator_function_declaration",
                "class_declaration",
                "abstract_class_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            ],
            window,
        ),
        "tsx" | "jsx" | "js" | "mjs" | "cjs" => symbols(
            path,
            text,
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            &[
                "function_declaration",
                "generator_function_declaration",
                "class_declaration",
                "abstract_class_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            ],
            window,
        ),
        "go" => symbols(
            path,
            text,
            tree_sitter_go::LANGUAGE.into(),
            &[
                "function_declaration",
                "method_declaration",
                // `type_spec` carries the name; `type_declaration` is the
                // unnamed wrapper around it.
                "type_spec",
            ],
            window,
        ),
        "java" => symbols(
            path,
            text,
            tree_sitter_java::LANGUAGE.into(),
            &[
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "annotation_type_declaration",
            ],
            window,
        ),
        "md" | "markdown" => sections(path, text),
        "nix" => nix_symbols(path, text, window),
        _ => windows(path, text, window),
    };
    if let Some(header) = header_chunk(path, text, extension) {
        chunks.insert(0, header);
    }
    split_oversized(path, text, &mut chunks);
    chunks
}

/// Split over-long symbol chunks into overlapping parts. Parts inherit the
/// parent breadcrumb with a `[i/n]` suffix so symbol context survives the
/// split and chunk IDs stay deterministic.
fn split_oversized(path: &Path, text: &str, chunks: &mut Vec<Chunk>) {
    let total_lines = text.lines().count() as u64;
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks.drain(..) {
        if chunk.kind != ChunkKind::Symbol || chunk.end <= chunk.start + SYMBOL_CAP_LINES as u64 {
            out.push(chunk);
            continue;
        }
        let span = chunk.end - chunk.start + 1;
        let n = ((span - SYMBOL_CAP_LINES as u64 + SYMBOL_STEP_LINES as u64 - 1)
            / SYMBOL_STEP_LINES as u64
            + 1) as usize;
        let mut start = chunk.start;
        for i in 0..n {
            let end = (start + SYMBOL_CAP_LINES as u64 - 1)
                .min(chunk.end)
                .min(total_lines.max(1));
            out.push(Chunk {
                path: path.to_path_buf(),
                start,
                end,
                kind: ChunkKind::Symbol,
                breadcrumb: format!("{} [{}/{}]", chunk.breadcrumb, i + 1, n),
                text: slice_lines(text, (start - 1) as usize, (end - 1) as usize),
            });
            if end >= chunk.end {
                break;
            }
            start += SYMBOL_STEP_LINES as u64;
        }
    }
    *chunks = out;
}

fn node_text<'b>(node: tree_sitter::Node<'_>, bytes: &'b [u8]) -> &'b str {
    node.utf8_text(bytes).unwrap_or("")
}

/// Nix: every `binding` (`attrpath = value`) is a chunk; nested attrsets
/// extend the breadcrumb (`zsh > plugins`). Dotted attrpaths stay dotted.
fn nix_symbols(path: &Path, text: &str, window: Window) -> Vec<Chunk> {
    let mut parser = tree_sitter::Parser::new();
    if parser
        .set_language(&tree_sitter_nix::LANGUAGE.into())
        .is_err()
    {
        return windows(path, text, window);
    }
    let Some(tree) = parser.parse(text, None) else {
        return windows(path, text, window);
    };
    if tree.root_node().has_error() {
        return windows(path, text, window);
    }
    let bytes = text.as_bytes();
    let mut chunks = Vec::new();
    let mut scope: Vec<String> = Vec::new();
    nix_walk(tree.root_node(), bytes, &mut scope, path, text, &mut chunks);
    if chunks.is_empty() {
        return windows(path, text, window);
    }
    chunks
}

/// Prepend contiguous leading `#` comment lines (up to 10) to a chunk.
fn with_leading_comments(text: &str, start_row: usize, end_row: usize) -> (usize, String) {
    let lines: Vec<&str> = text.lines().collect();
    let mut first = start_row;
    let mut count = 0;
    while first > 0 && count < 10 && lines[first - 1].trim_start().starts_with('#') {
        first -= 1;
        count += 1;
    }
    (first, slice_lines(text, first, end_row))
}

fn nix_walk(
    node: tree_sitter::Node<'_>,
    bytes: &[u8],
    scope: &mut Vec<String>,
    path: &Path,
    text: &str,
    chunks: &mut Vec<Chunk>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "binding" {
            let name = child
                .named_children(&mut child.walk())
                .find(|n| n.kind() == "attrpath")
                .map(|n| node_text(n, bytes).trim().to_owned())
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| "binding".to_owned());
            scope.push(name);
            // Leading `#` comments document the binding (like rustdoc);
            // without them symbol chunks lose all prose context.
            let (start_row, body) =
                with_leading_comments(text, child.start_position().row, child.end_position().row);
            // Single-line uncommented bindings nested inside a container add
            // BM25 noise (`enable = true` x100); the container chunk already
            // covers them. Root singletons are still emitted.
            let bare_single =
                start_row == child.start_position().row && start_row == child.end_position().row;
            if !(bare_single && scope.len() > 1) {
                chunks.push(Chunk {
                    path: path.to_path_buf(),
                    start: start_row as u64 + 1,
                    end: child.end_position().row as u64 + 1,
                    kind: ChunkKind::Symbol,
                    breadcrumb: scope.join(" > "),
                    text: body,
                });
            }
            nix_walk(child, bytes, scope, path, text, chunks);
            scope.pop();
        } else {
            nix_walk(child, bytes, scope, path, text, chunks);
        }
    }
}

fn symbol_name(node: tree_sitter::Node<'_>, bytes: &[u8]) -> Option<String> {
    if let Some(name) = node.child_by_field_name("name") {
        return Some(node_text(name, bytes).to_owned());
    }
    // `impl Foo` / `impl Trait for Foo` have a `type` field instead of a name.
    if node.kind() == "impl_item"
        && let Some(ty) = node.child_by_field_name("type")
    {
        return Some(format!("impl {}", node_text(ty, bytes)));
    }
    None
}

fn slice_lines(text: &str, start_row: usize, end_row: usize) -> String {
    text.lines()
        .skip(start_row)
        .take(end_row.saturating_sub(start_row) + 1)
        .collect::<Vec<_>>()
        .join("\n")
}

fn symbols(
    path: &Path,
    text: &str,
    language: tree_sitter::Language,
    kinds: &[&str],
    window: Window,
) -> Vec<Chunk> {
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&language).is_err() {
        return windows(path, text, window);
    }
    let Some(tree) = parser.parse(text, None) else {
        return windows(path, text, window);
    };
    let bytes = text.as_bytes();
    let mut chunks = Vec::new();
    let mut scope: Vec<String> = Vec::new();
    walk(
        tree.root_node(),
        bytes,
        kinds,
        &mut scope,
        path,
        text,
        &mut chunks,
    );
    if chunks.is_empty() {
        return windows(path, text, window);
    }
    chunks
}

fn walk(
    node: tree_sitter::Node<'_>,
    bytes: &[u8],
    kinds: &[&str],
    scope: &mut Vec<String>,
    path: &Path,
    text: &str,
    chunks: &mut Vec<Chunk>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if kinds.contains(&child.kind()) {
            let name = symbol_name(child, bytes).unwrap_or_else(|| child.kind().to_owned());
            scope.push(name);
            chunks.push(Chunk {
                path: path.to_path_buf(),
                start: child.start_position().row as u64 + 1,
                end: child.end_position().row as u64 + 1,
                kind: ChunkKind::Symbol,
                breadcrumb: scope.join(" > "),
                text: slice_lines(text, child.start_position().row, child.end_position().row),
            });
            walk(child, bytes, kinds, scope, path, text, chunks);
            scope.pop();
        } else {
            // Recurse through non-symbol nodes (e.g. source file root).
            walk(child, bytes, kinds, scope, path, text, chunks);
        }
    }
}

/// Split Markdown on ATX headings; text before the first heading becomes a
/// `(preamble)` chunk when non-blank.
fn sections(path: &Path, text: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut headings: Vec<(usize, u8, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
        if (1..=6).contains(&hashes) && trimmed.as_bytes().get(hashes) == Some(&b' ') {
            headings.push((i, hashes as u8, trimmed[hashes + 1..].trim().to_owned()));
        }
    }
    if headings.is_empty() {
        return windows(path, text, DEFAULT_WINDOW);
    }
    let mut chunks = Vec::new();
    if headings[0].0 > 0 {
        let body = lines[..headings[0].0].join("\n");
        if !body.trim().is_empty() {
            chunks.push(Chunk {
                path: path.to_path_buf(),
                start: 1,
                end: headings[0].0 as u64,
                kind: ChunkKind::Section,
                breadcrumb: "(preamble)".to_owned(),
                text: body,
            });
        }
    }
    for (n, (line, level, title)) in headings.iter().enumerate() {
        let end = headings
            .iter()
            .skip(n + 1)
            .find(|(_, l, _)| *l <= *level)
            .map_or(lines.len(), |(l, _, _)| *l)
            - 1;
        let mut stack: Vec<&str> = Vec::new();
        for (_, l, t) in headings.iter().take(n + 1) {
            while stack.len() >= *l as usize {
                stack.pop();
            }
            stack.push(t);
        }
        let _ = title;
        chunks.push(Chunk {
            path: path.to_path_buf(),
            start: *line as u64 + 1,
            end: end as u64 + 1,
            kind: ChunkKind::Section,
            breadcrumb: stack.join(" > "),
            text: lines[*line..=end].join("\n"),
        });
    }
    chunks
}

/// Fixed windows of `window.0` lines stepping `window.1`, with overlap.
fn windows(path: &Path, text: &str, window: Window) -> Vec<Chunk> {
    let (size, step) = window;
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let end = (start + size).min(lines.len()) - 1;
        chunks.push(Chunk {
            path: path.to_path_buf(),
            start: start as u64 + 1,
            end: end as u64 + 1,
            kind: ChunkKind::Window,
            breadcrumb: format!("lines {}-{}", start + 1, end + 1),
            text: lines[start..=end].join("\n"),
        });
        if end + 1 >= lines.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_symbols_carry_breadcrumbs_and_lines() {
        let text = "mod foo {\n    pub struct Bar;\n\n    impl Bar {\n        pub fn baz(&self) {}\n    }\n}\n";
        let chunks = extract(Path::new("lib.rs"), text);
        let names: Vec<&str> = chunks.iter().map(|c| c.breadcrumb.as_str()).collect();
        assert!(names.contains(&"foo"), "{names:?}");
        assert!(names.contains(&"foo > Bar"), "{names:?}");
        assert!(names.contains(&"foo > impl Bar > baz"), "{names:?}");
        let baz = chunks
            .iter()
            .find(|c| c.breadcrumb == "foo > impl Bar > baz")
            .expect("baz chunk");
        assert_eq!((baz.start, baz.end), (5, 5));
        assert!(baz.text.contains("pub fn baz"));
        assert!(chunks.iter().all(|c| c.kind == ChunkKind::Symbol));
    }

    #[test]
    fn python_class_and_method() {
        let text = "class Greeter:\n    def hello(self):\n        return 1\n";
        let chunks = extract(Path::new("a.py"), text);
        let names: Vec<&str> = chunks.iter().map(|c| c.breadcrumb.as_str()).collect();
        assert!(names.contains(&"Greeter"), "{names:?}");
        assert!(names.contains(&"Greeter > hello"), "{names:?}");
    }

    #[test]
    fn typescript_function_and_interface() {
        let text = "export interface LaneConfig {\n  retries: number;\n}\n\nexport function apply_lane_defaults(config: LaneConfig): boolean {\n  return true;\n}\n";
        let chunks = extract(Path::new("lane.ts"), text);
        let names: Vec<&str> = chunks.iter().map(|c| c.breadcrumb.as_str()).collect();
        assert!(names.contains(&"LaneConfig"), "{names:?}");
        assert!(names.contains(&"apply_lane_defaults"), "{names:?}");
        assert!(chunks.iter().all(|c| c.kind == ChunkKind::Symbol));
    }

    #[test]
    fn go_function_and_type() {
        let text = "package lane\n\ntype Config struct {\n\tRetries int\n}\n\nfunc Apply(cfg Config) bool {\n\treturn true\n}\n";
        let chunks = extract(Path::new("lane.go"), text);
        let names: Vec<&str> = chunks.iter().map(|c| c.breadcrumb.as_str()).collect();
        assert!(names.contains(&"Config"), "{names:?}");
        assert!(names.contains(&"Apply"), "{names:?}");
    }

    #[test]
    fn header_comment_becomes_own_chunk() {
        let text = "# Exclusive MLX lane switcher.\n{ pkgs }: {\n  name = \"x\";\n}\n";
        let chunks = extract(Path::new("lane.nix"), text);
        let header = chunks
            .iter()
            .find(|c| c.breadcrumb == "(header)")
            .expect("header chunk");
        assert_eq!((header.start, header.end), (1, 1));
        assert!(header.text.contains("Exclusive"));
        assert_eq!(header.kind, ChunkKind::Section);
    }

    #[test]
    fn header_skips_shebang_and_blank_preamble() {
        let text = "#!/usr/bin/env python3\n\n# Real header.\nimport os\n";
        let chunks = extract(Path::new("a.py"), text);
        let header = chunks
            .iter()
            .find(|c| c.breadcrumb == "(header)")
            .expect("header chunk");
        assert_eq!((header.start, header.end), (3, 3));
        assert!(!header.text.contains("usr/bin"));
    }

    #[test]
    fn header_absent_without_leading_comments() {
        let chunks = extract(Path::new("a.py"), "import os\n");
        assert!(chunks.iter().all(|c| c.breadcrumb != "(header)"));
    }

    #[test]
    fn header_caps_long_boilerplate() {
        let mut text = String::new();
        for i in 0..40 {
            text.push_str(&format!("# license line {i}\n"));
        }
        text.push_str("x = 1\n");
        let chunks = extract(Path::new("a.nix"), &text);
        let header = chunks
            .iter()
            .find(|c| c.breadcrumb == "(header)")
            .expect("header chunk");
        assert_eq!((header.start, header.end), (1, HEADER_CAP_LINES as u64));
    }

    #[test]
    fn oversized_symbol_splits_with_inherited_crumbs() {
        let mut text = String::from("fn big() {\n");
        for i in 0..100 {
            text.push_str(&format!("    let x{i} = {i};\n"));
        }
        text.push_str("}\n");
        let chunks = extract(Path::new("a.rs"), &text);
        let parts: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| c.breadcrumb.contains("big ["))
            .collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].breadcrumb, "big [1/2]");
        assert_eq!(parts[1].breadcrumb, "big [2/2]");
        assert!(parts.iter().all(|c| c.kind == ChunkKind::Symbol));
        // Full coverage with overlap, no gaps.
        assert_eq!(parts[0].start, 1);
        assert!(parts[0].end >= parts[1].start - 15);
        assert_eq!(parts[1].end, 102);
        assert!(parts[0].text.contains("let x0"));
        assert!(parts[1].text.contains("let x99"));
    }

    #[test]
    fn small_symbol_stays_whole() {
        let chunks = extract(Path::new("a.rs"), "fn tiny() {}\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].breadcrumb, "tiny");
    }

    #[test]
    fn markdown_has_no_header_chunk() {
        let chunks = extract(Path::new("a.md"), "# Title\n\nbody\n");
        assert!(chunks.iter().all(|c| c.breadcrumb != "(header)"));
    }

    #[test]
    fn java_class_and_method() {
        let text = "class Lane {\n  boolean apply() {\n    return true;\n  }\n}\n";
        let chunks = extract(Path::new("Lane.java"), text);
        let names: Vec<&str> = chunks.iter().map(|c| c.breadcrumb.as_str()).collect();
        assert!(names.contains(&"Lane"), "{names:?}");
        assert!(names.contains(&"Lane > apply"), "{names:?}");
    }

    #[test]
    fn markdown_sections_nest() {
        let text = "# T\nintro\n## A\na1\n### B\nb1\n## C\nc1\n";
        let chunks = extract(Path::new("doc.md"), text);
        let names: Vec<&str> = chunks.iter().map(|c| c.breadcrumb.as_str()).collect();
        assert_eq!(names, vec!["T", "T > A", "T > A > B", "T > C"]);
        let b = &chunks[2];
        assert_eq!((b.start, b.end), (5, 6));
    }

    #[test]
    fn markdown_preamble_kept() {
        let text = "pre\n# T\nbody\n";
        let chunks = extract(Path::new("doc.md"), text);
        assert_eq!(chunks[0].breadcrumb, "(preamble)");
        assert_eq!(chunks[0].text, "pre");
    }

    #[test]
    fn fallback_windows_cover_long_files() {
        let text = (1..=400)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = extract(Path::new("data.txt"), &text);
        assert_eq!(chunks.len(), 3);
        assert_eq!((chunks[0].start, chunks[0].end), (1, 150));
        assert_eq!((chunks[1].start, chunks[1].end), (136, 285));
        assert_eq!((chunks[2].start, chunks[2].end), (271, 400));
    }

    #[test]
    fn binary_file_yields_no_chunks() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bin.dat");
        std::fs::write(&path, [0x00, 0x01, 0x02]).expect("write");
        assert!(extract_file(&path).expect("extract").is_empty());
    }

    #[test]
    fn nix_bindings_nest_and_dot() {
        let text = "{\n  direnv = {\n    # direnv integration\n    enable = true;\n    nix-direnv.enable = true;\n  };\n  zsh.enable = true;\n}\n";
        let chunks = extract(Path::new("mod.nix"), text);
        let names: Vec<&str> = chunks.iter().map(|c| c.breadcrumb.as_str()).collect();
        assert!(names.contains(&"direnv"), "{names:?}");
        // Commented single-line nested bindings are kept (prose context).
        assert!(names.contains(&"direnv > enable"), "{names:?}");
        // Bare single-line nested bindings are covered by the container.
        assert!(!names.contains(&"direnv > nix-direnv.enable"), "{names:?}");
        assert!(names.contains(&"zsh.enable"), "{names:?}");
        let enable = chunks
            .iter()
            .find(|c| c.breadcrumb == "direnv > enable")
            .expect("enable chunk");
        assert_eq!((enable.start, enable.end), (3, 4));
        assert!(enable.text.contains("direnv integration"));
        assert!(chunks.iter().all(|c| c.kind == ChunkKind::Symbol));
    }

    #[test]
    fn nix_lambda_file_parses() {
        let text = "{ pkgs }:\n{\n  foo = pkgs.bar;\n}\n";
        let chunks = extract(Path::new("mod.nix"), text);
        assert!(chunks.iter().any(|c| c.breadcrumb == "foo"), "{chunks:?}");
    }

    #[test]
    fn nix_leading_comments_attach() {
        let text = "{\n# Local secrets here\nsecrets = {\n  token = \"x\";\n};\n}\n";
        let chunks = extract(Path::new("mod.nix"), text);
        let secrets = chunks
            .iter()
            .find(|c| c.breadcrumb == "secrets")
            .expect("secrets chunk");
        assert_eq!(secrets.start, 2);
        assert!(secrets.text.contains("Local secrets here"));
    }

    #[test]
    fn tight_windows_cover_short_files() {
        let text = (1..=120)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = extract_with(Path::new("data.txt"), &text, EMBED_WINDOW);
        assert_eq!(chunks.len(), 3);
        assert_eq!((chunks[0].start, chunks[0].end), (1, 50));
        assert_eq!((chunks[1].start, chunks[1].end), (41, 90));
        assert_eq!((chunks[2].start, chunks[2].end), (81, 120));
    }
}
