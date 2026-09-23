//! Call-chain chunks: 2-hop caller→callee paths as compositional units.
//!
//! The Papers' transfer: atoms (symbols) compose into reasoning chains.
//! A chain chunk joins a caller with one resolved callee so data-flow and
//! call-chain questions match a single citable unit.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{Error, engine, extract};

/// Cap on chain chunks per workspace (call sites explode combinatorially).
const MAX_CHAINS: usize = 8000;

struct CallSite {
    caller: usize,
    callee_name: String,
}

struct Parsed {
    rel: String,
    chunks: Vec<extract::Chunk>,
    calls: Vec<CallSite>,
}

fn simple_name(breadcrumb: &str) -> &str {
    breadcrumb.rsplit(" > ").next().unwrap_or(breadcrumb)
}

fn is_call(node: tree_sitter::Node<'_>) -> bool {
    // tree-sitter-rust says `call_expression`, tree-sitter-python says `call`.
    matches!(node.kind(), "call_expression" | "call")
}

fn callee_of(node: tree_sitter::Node<'_>, bytes: &[u8]) -> Option<String> {
    let func = node.child_by_field_name("function")?;
    match func.kind() {
        "identifier" => Some(func.utf8_text(bytes).unwrap_or("").to_owned()),
        // `x.foo()` (field_expression/attribute) and `a::b()`/`A.B()`
        // (scoped_identifier): the callee is the last segment.
        "field_expression" | "attribute" | "scoped_identifier" => {
            let mut cursor = func.walk();
            func.named_children(&mut cursor)
                .last()
                .map(|n| n.utf8_text(bytes).unwrap_or("").to_owned())
        }
        _ => None,
    }
}

/// Parse one file: symbol chunks plus call sites attributed to the
/// innermost enclosing symbol chunk.
fn parse_file(
    rel: &str,
    path: &Path,
    text: &str,
    language: tree_sitter::Language,
    kinds: &[&str],
) -> Option<Parsed> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(text, None)?;
    if tree.root_node().has_error() {
        return None;
    }
    let bytes = text.as_bytes();
    // Reuse the standard extractor for chunk spans, then attribute calls.
    let chunks = extract::extract(path, text);
    if chunks.is_empty() {
        return None;
    }
    // Map each call to the innermost symbol chunk containing it via spans.
    let mut calls = Vec::new();
    // caller index per tree node resolved lazily: walk with chunk scope.
    let mut work: Vec<(tree_sitter::Node<'_>, Option<usize>)> = vec![(tree.root_node(), None)];
    while let Some((node, caller)) = work.pop() {
        let caller = if kinds.contains(&node.kind()) {
            chunk_containing(&chunks, node.start_position().row, node.end_position().row).or(caller)
        } else {
            caller
        };
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if is_call(child) {
                if let (Some(idx), Some(name)) = (caller, callee_of(child, bytes)) {
                    if !name.is_empty() {
                        calls.push(CallSite {
                            caller: idx,
                            callee_name: name,
                        });
                    }
                }
            }
            work.push((child, caller));
        }
    }
    Some(Parsed {
        rel: rel.to_owned(),
        chunks,
        calls,
    })
}

fn chunk_containing(chunks: &[extract::Chunk], start_row: usize, end_row: usize) -> Option<usize> {
    // Innermost = smallest span containing the node.
    chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| (c.start as usize) <= start_row + 1 && (c.end as usize) >= end_row + 1)
        .min_by_key(|(_, c)| c.end - c.start)
        .map(|(i, _)| i)
}

/// Extract 2-hop chain chunks across `workspace` (Rust + Python).
///
/// # Errors
///
/// Returns [`Error`] when the workspace cannot be walked.
pub fn extract_workspace(workspace: &Path) -> Result<Vec<extract::Chunk>, Error> {
    let mut parsed: Vec<Parsed> = Vec::new();
    for entry in engine::walker(workspace).filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "rs" && ext != "py" {
            continue;
        }
        let rel = path
            .strip_prefix(workspace)
            .map_err(|_| {
                Error::InvalidInput(format!("path escapes workspace: {}", path.display()))
            })?
            .to_string_lossy()
            .into_owned();
        let bytes = std::fs::read(path)?;
        if bytes.contains(&0) {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        let (language, kinds): (tree_sitter::Language, &[&str]) = match ext {
            "rs" => (
                tree_sitter_rust::LANGUAGE.into(),
                &["function_item", "mod_item", "impl_item"],
            ),
            _ => (
                tree_sitter_python::LANGUAGE.into(),
                &["function_definition", "class_definition"],
            ),
        };
        if let Some(p) = parse_file(&rel, path, &text, language, kinds) {
            parsed.push(p);
        }
    }

    // Resolution tables.
    let mut by_file_name: HashMap<(String, String), usize> = HashMap::new();
    let mut global: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (fi, p) in parsed.iter().enumerate() {
        for (ci, c) in p.chunks.iter().enumerate() {
            if c.kind != extract::ChunkKind::Symbol {
                continue;
            }
            let name = simple_name(&c.breadcrumb).to_owned();
            // Strip `impl X` wrappers to the type name for method lookup.
            let short = name
                .strip_prefix("impl ")
                .unwrap_or(&name)
                .split_whitespace()
                .next()
                .unwrap_or(&name)
                .to_owned();
            by_file_name.insert((p.rel.clone(), name.clone()), ci);
            global.entry(short).or_default().push((fi, ci));
        }
    }

    let mut chains = Vec::new();
    // Repeated call sites (same caller span + callee) produce byte-identical
    // chunks. Each duplicate would steal a pool slot and stack another RRF
    // term onto the same fused slot (Run 12: triple lexical credit for one
    // logical chunk). Keep the first, drop the rest.
    let mut seen = std::collections::HashSet::new();
    'outer: for (fi, p) in parsed.iter().enumerate() {
        for call in &p.calls {
            if chains.len() >= MAX_CHAINS {
                break 'outer;
            }
            let Some(caller) = p.chunks.get(call.caller) else {
                continue;
            };
            // Same-file match first, else unique global match.
            let target = by_file_name
                .get(&(p.rel.clone(), call.callee_name.clone()))
                .map(|&ci| (fi, ci))
                .or_else(|| {
                    let v = global.get(&call.callee_name)?;
                    (v.len() == 1).then(|| v[0])
                });
            let Some((tfi, tci)) = target else { continue };
            if tfi == fi {
                if let Some(tc) = parsed[tfi].chunks.get(tci) {
                    if tc.start == caller.start && tc.end == caller.end {
                        continue; // Self-call: no chain.
                    }
                }
            }
            let callee = &parsed[tfi].chunks[tci];
            let key = (
                p.rel.clone(),
                caller.start,
                caller.end,
                simple_name(&callee.breadcrumb).to_owned(),
            );
            if !seen.insert(key) {
                continue;
            }
            chains.push(extract::Chunk {
                path: PathBuf::from(&p.rel),
                start: caller.start,
                end: caller.end,
                kind: extract::ChunkKind::Symbol,
                breadcrumb: format!(
                    "{} > calls > {}",
                    caller.breadcrumb,
                    simple_name(&callee.breadcrumb)
                ),
                text: format!("{}\n--- calls ---\n{}", caller.text, callee.text),
            });
        }
    }
    Ok(chains)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, contents) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("mkdir");
            }
            std::fs::write(&path, contents).expect("write");
        }
        dir
    }

    #[test]
    fn resolves_same_file_call() {
        let dir = workspace_with(&[(
            "lib.rs",
            "fn fetch() -> i32 { load() }\nfn load() -> i32 { 1 }\n",
        )]);
        let chains = extract_workspace(dir.path()).expect("chains");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].breadcrumb, "fetch > calls > load");
        assert!(chains[0].text.contains("fn fetch"));
        assert!(chains[0].text.contains("fn load"));
    }

    #[test]
    fn unique_global_match_across_files() {
        let dir = workspace_with(&[
            ("a.rs", "fn main() { helper() }\n"),
            ("b.rs", "fn helper() {}\n"),
        ]);
        let chains = extract_workspace(dir.path()).expect("chains");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].breadcrumb, "main > calls > helper");
    }

    #[test]
    fn ambiguous_names_do_not_chain() {
        let dir = workspace_with(&[
            ("a.rs", "fn main() { helper() }\n"),
            ("b.rs", "fn helper() {}\n"),
            ("c.rs", "fn helper() {}\n"),
        ]);
        let chains = extract_workspace(dir.path()).expect("chains");
        assert!(chains.is_empty());
    }

    #[test]
    fn repeated_call_sites_emit_one_chain() {
        let dir = workspace_with(&[(
            "a.py",
            "def caller():\n    callee()\n    callee()\n    callee()\n\ndef callee():\n    pass\n",
        )]);
        let chains = extract_workspace(dir.path()).expect("chains");
        let matching: Vec<_> = chains
            .iter()
            .filter(|c| c.breadcrumb == "caller > calls > callee")
            .collect();
        assert_eq!(matching.len(), 1, "{chains:?}");
    }

    #[test]
    fn method_call_resolves_by_name() {
        let dir = workspace_with(&[(
            "lib.py",
            "class A:\n    def run(self):\n        self.step()\n    def step(self):\n        pass\n",
        )]);
        let chains = extract_workspace(dir.path()).expect("chains");
        assert!(
            chains
                .iter()
                .any(|c| c.breadcrumb == "A > run > calls > step")
                || chains.iter().any(|c| c.breadcrumb == "run > calls > step"),
            "{chains:?}"
        );
    }
}
