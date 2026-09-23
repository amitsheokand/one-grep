//! Hybrid retrieval: BM25 + vector ranks fused with RRF.
//!
//! Falls back to lexical-only when no vector store exists yet, and to
//! live rg when the workspace has no index.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{
    Error,
    embed::{EmbedProvider, Rerank},
    index, rg, vectors,
};

/// RRF smoothing constant.
const RRF_K: f64 = 20.0;
/// Lexical terms weigh double: exact matches outrank fuzzy ones, while
/// vector-only discovery still surfaces when lexical misses entirely.
const LEXICAL_WEIGHT: f64 = 2.0;
/// Fused candidates rescored by the reranker.
const RERANK_DEPTH: usize = 20;

/// One fused hit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FusedHit {
    /// Absolute file path.
    pub path: PathBuf,
    /// 1-based first line.
    pub start: u64,
    /// 1-based last line (inclusive).
    pub end: u64,
    /// Symbol path or heading stack.
    pub breadcrumb: String,
    /// Chunk text.
    pub text: String,
    /// Fused RRF score.
    pub score: f64,
    /// 1-based lexical rank, when present.
    pub lexical_rank: Option<usize>,
    /// 1-based vector rank, when present.
    pub vector_rank: Option<usize>,
    /// 1-based structural (ast-grep) rank, when present.
    pub ast_rank: Option<usize>,
}

/// Retrieval source for output budgets: which list(s) produced the hit.
/// Live-`rg` fallbacks carry neither rank. Structural-only hits are "ast";
/// mixed hits keep their lexical/vector label (the ast term only boosts).
#[must_use]
pub fn source(hit: &FusedHit) -> &'static str {
    if hit.lexical_rank.is_none() && hit.vector_rank.is_none() && hit.ast_rank.is_some() {
        return "ast";
    }
    match (hit.lexical_rank, hit.vector_rank) {
        (Some(_), Some(_)) => "bm25+vec",
        (Some(_), None) => "bm25",
        (None, Some(_)) => "vec",
        (None, None) => "rg",
    }
}

fn chunk_key(path: &Path, start: u64, end: u64, breadcrumb: &str) -> String {
    format!("{}:{start}-{end}:{breadcrumb}", path.display())
}

/// Fuse two rank lists with reciprocal rank fusion, optionally rescored
/// by a cross-encoder reranker over the top candidates.
///
/// # Errors
///
/// Returns [`Error`] when lexical search fails or the query cannot be
/// embedded. Unindexed workspaces return live rg hits instead of erroring.
pub fn hybrid(
    workspace: &Path,
    query: &str,
    limit: usize,
    provider: Option<&dyn EmbedProvider>,
    reranker: Option<&dyn Rerank>,
) -> Result<Vec<FusedHit>, Error> {
    if !index::is_indexed(workspace) {
        return Ok(from_rg(rg::fallback_search(workspace, query, limit)?));
    }
    let fetch = (limit * 10).max(50);
    let lexical = index::search(workspace, query, fetch)?;

    let mut vector_ranks: Vec<(String, index::RankedHit)> = Vec::new();
    if let Some(provider) = provider {
        // Skip vectors when the provider does not match the store, instead
        // of scoring against truncated dimensions.
        let dims_ok = vectors::store_dims(workspace)
            .map(|dims| dims.is_none_or(|d| d == provider.dims()))
            .unwrap_or(true);
        if dims_ok {
            let items = vectors::load_items(workspace)?;
            if !items.is_empty() {
                let query_text = match provider.query_prefix() {
                    Some(prefix) => format!("{prefix}{query}"),
                    None => query.to_owned(),
                };
                let query_vec = provider
                    .embed(&[query_text.as_str()])?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        Error::Embed("provider returned no vector for query".to_owned())
                    })?;
                let norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                let query_vec: Vec<f32> = if norm > 0.0 {
                    query_vec.iter().map(|x| x / norm).collect()
                } else {
                    query_vec
                };
                let refs: Vec<&vectors::VectorItem> = items.iter().collect();
                let ranked = vectors::topk(&refs, &query_vec, fetch);
                for (i, _) in ranked.iter().enumerate() {
                    let item = refs[ranked[i].0];
                    vector_ranks.push((
                        chunk_key(
                            &workspace.join(&item.rel),
                            item.start,
                            item.end,
                            &item.breadcrumb,
                        ),
                        index::RankedHit {
                            path: workspace.join(&item.rel),
                            start: item.start,
                            end: item.end,
                            breadcrumb: item.breadcrumb.clone(),
                            text: item.text.clone(),
                            score: 0.0,
                        },
                    ));
                }
            }
        }
    }

    let mut fused: Vec<FusedHit> = Vec::new();
    let mut by_key: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut upsert = |key: String,
                      hit: &index::RankedHit,
                      term: f64,
                      lexical_rank: Option<usize>,
                      vector_rank: Option<usize>| {
        if let Some(&i) = by_key.get(&key) {
            let slot = &mut fused[i];
            slot.score += term;
            if slot.lexical_rank.is_none() {
                slot.lexical_rank = lexical_rank;
            }
            if slot.vector_rank.is_none() {
                slot.vector_rank = vector_rank;
            }
            return;
        }
        by_key.insert(key, fused.len());
        fused.push(FusedHit {
            path: hit.path.clone(),
            start: hit.start,
            end: hit.end,
            breadcrumb: hit.breadcrumb.clone(),
            text: hit.text.clone(),
            score: term,
            lexical_rank,
            vector_rank,
            ast_rank: None,
        });
    };

    for (i, hit) in lexical.iter().enumerate() {
        let key = chunk_key(&hit.path, hit.start, hit.end, &hit.breadcrumb);
        upsert(
            key,
            hit,
            LEXICAL_WEIGHT * rrf(Some(i + 1)),
            Some(i + 1),
            None,
        );
    }
    // Vector windows are tighter than lexical chunks, so keys rarely match.
    // Credit one vector term per slot to the most-overlapping lexical chunk
    // in the same file (extra overlapping windows are redundant votes, not
    // independent evidence); vector-only regions become their own entries.
    for (i, (_, hit)) in vector_ranks.iter().enumerate() {
        let slot = fused
            .iter_mut()
            .filter(|s| s.path == hit.path && s.vector_rank.is_none())
            .max_by_key(|s| overlap(s.start, s.end, hit.start, hit.end));
        if let Some(slot) = slot {
            if overlap(slot.start, slot.end, hit.start, hit.end) > 0 {
                slot.score += rrf(Some(i + 1));
                slot.vector_rank = Some(i + 1);
                continue;
            }
        }
        fused.push(FusedHit {
            path: hit.path.clone(),
            start: hit.start,
            end: hit.end,
            breadcrumb: hit.breadcrumb.clone(),
            text: hit.text.clone(),
            score: rrf(Some(i + 1)),
            lexical_rank: None,
            vector_rank: Some(i + 1),
            ast_rank: None,
        });
    }

    // Exact beats fuzzy on ties: lexical rank first, then vector rank,
    // then structural rank.
    fused.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| cmp_rank(a.lexical_rank, b.lexical_rank))
            .then_with(|| cmp_rank(a.vector_rank, b.vector_rank))
            .then_with(|| cmp_rank(a.ast_rank, b.ast_rank))
    });

    if let Some(reranker) = reranker {
        rerank_top(query, &mut fused, reranker)?;
    }
    fused.truncate(limit);
    Ok(fused)
}

/// Rescore the top candidates with a cross-encoder; unranked tails keep
/// fused order behind rescored heads.
fn rerank_top(query: &str, fused: &mut Vec<FusedHit>, reranker: &dyn Rerank) -> Result<(), Error> {
    let depth = fused.len().min(RERANK_DEPTH);
    if depth == 0 {
        return Ok(());
    }
    let docs: Vec<String> = fused[..depth]
        .iter()
        .map(|h| format!("{} {}", h.breadcrumb, truncate_doc(&h.text)))
        .collect();
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    // Already best-first from the reranker.
    let order: Vec<(usize, f32)> = reranker.rerank(query, &refs)?;
    let mut rescored: Vec<FusedHit> = Vec::with_capacity(fused.len());
    for (orig, score) in &order {
        let mut hit = fused[*orig].clone();
        hit.score = f64::from(*score);
        rescored.push(hit);
    }
    // Any candidate the reranker dropped keeps fused order at the tail.
    let seen: std::collections::HashSet<usize> = order.iter().map(|(i, _)| *i).collect();
    for (i, hit) in fused.iter().enumerate() {
        if !seen.contains(&i) && rescored.len() < fused.len() {
            rescored.push(hit.clone());
        }
    }
    *fused = rescored;
    Ok(())
}

fn truncate_doc(text: &str) -> &str {
    const CAP: usize = 1000;
    if text.len() <= CAP {
        return text;
    }
    let mut end = CAP;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn cmp_rank(a: Option<usize>, b: Option<usize>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn rrf(rank: Option<usize>) -> f64 {
    rank.map_or(0.0, |r| 1.0 / (RRF_K + r as f64))
}

/// Inclusive line overlap of two ranges.
fn overlap(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> u64 {
    (a_end.min(b_end) + 1).saturating_sub(a_start.max(b_start))
}

/// Fuse structural (ast-grep) hits as a third RRF list over an existing
/// shortlist. Each `(path, line)` credits the overlapping fused slot in
/// the same file (enumeration order is the ast rank); non-overlapping
/// regions become their own entries. An empty list is a no-op, so absent
/// ast-grep means bit-identical results.
///
/// The caller re-truncates to its limit afterwards.
pub fn apply_ast(fused: &mut Vec<FusedHit>, ast: &[(PathBuf, u64)]) {
    for (i, (path, line)) in ast.iter().enumerate() {
        let rank = i + 1;
        let slot = fused
            .iter_mut()
            .filter(|s| &s.path == path && s.ast_rank.is_none())
            .max_by_key(|s| overlap(s.start, s.end, *line, *line));
        if let Some(slot) = slot {
            if overlap(slot.start, slot.end, *line, *line) > 0 {
                slot.score += rrf(Some(rank));
                slot.ast_rank = Some(rank);
                continue;
            }
        }
        fused.push(FusedHit {
            path: path.clone(),
            start: *line,
            end: *line,
            breadcrumb: "ast-grep".to_owned(),
            text: String::new(),
            score: rrf(Some(rank)),
            lexical_rank: None,
            vector_rank: None,
            ast_rank: Some(rank),
        });
    }
    fused.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| cmp_rank(a.lexical_rank, b.lexical_rank))
            .then_with(|| cmp_rank(a.vector_rank, b.vector_rank))
            .then_with(|| cmp_rank(a.ast_rank, b.ast_rank))
    });
}

/// Retrieve a shortlist: live rg when unindexed, hybrid or BM25 otherwise.
///
/// # Errors
///
/// Returns [`Error`] when walk, index, or embedding fails.
pub fn collect(
    workspace: &Path,
    query: &str,
    limit: usize,
    fuse: bool,
    provider: Option<&dyn EmbedProvider>,
) -> Result<Vec<FusedHit>, Error> {
    if !index::is_indexed(workspace) {
        return Ok(from_rg(rg::fallback_search(workspace, query, limit)?));
    }
    if fuse {
        return hybrid(workspace, query, limit, provider, None);
    }
    Ok(index::search(workspace, query, limit)?
        .into_iter()
        .enumerate()
        .map(|(i, hit)| FusedHit {
            path: hit.path,
            start: hit.start,
            end: hit.end,
            breadcrumb: hit.breadcrumb,
            text: hit.text,
            score: f64::from(hit.score),
            lexical_rank: Some(i + 1),
            vector_rank: None,
            ast_rank: None,
        })
        .collect())
}

fn from_rg(hits: Vec<rg::Hit>) -> Vec<FusedHit> {
    hits.into_iter()
        .map(|hit| FusedHit {
            path: hit.path,
            start: hit.line,
            end: hit.line,
            breadcrumb: "rg-fallback".to_owned(),
            text: hit.text,
            score: 0.0,
            // Unranked live hits: no list claims them, so `source` is "rg".
            lexical_rank: None,
            vector_rank: None,
            ast_rank: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectors::tests::TestProvider;

    fn workspace_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, contents) in files {
            std::fs::write(dir.path().join(name), contents).expect("write");
        }
        dir
    }

    #[test]
    fn collect_without_index_uses_rg() {
        let dir = workspace_with(&[("a.txt", "supersonic_ferret zoology\n")]);
        let hits = collect(dir.path(), "supersonic_ferret", 10, true, None).expect("collect");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].breadcrumb, "rg-fallback");
    }

    #[test]
    fn hybrid_without_index_falls_back_to_rg() {
        let dir = workspace_with(&[("a.txt", "supersonic_ferret zoology\n")]);
        let hits =
            hybrid(dir.path(), "where is supersonic_ferret", 10, None, None).expect("hybrid");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].breadcrumb, "rg-fallback");
        // Unranked live hits claim no list; the budget tags them "rg".
        assert_eq!(hits[0].lexical_rank, None);
        assert_eq!(hits[0].vector_rank, None);
        assert_eq!(source(&hits[0]), "rg");
    }

    #[test]
    fn hybrid_without_provider_is_lexical_only() {
        let dir = workspace_with(&[("a.txt", "supersonic_ferret zoology\n")]);
        index::sync(dir.path()).expect("sync");
        let hits = hybrid(dir.path(), "supersonic_ferret", 10, None, None).expect("hybrid");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].lexical_rank, Some(1));
        assert_eq!(hits[0].vector_rank, None);
    }

    #[test]
    fn hybrid_fuses_both_ranks() {
        let dir = workspace_with(&[("a.txt", "supersonic_ferret zoology\n")]);
        index::sync(dir.path()).expect("sync");
        vectors::sync(dir.path(), &TestProvider).expect("vectors");
        let hits = hybrid(
            dir.path(),
            "supersonic_ferret",
            10,
            Some(&TestProvider),
            None,
        )
        .expect("hybrid");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].lexical_rank, Some(1));
        assert_eq!(hits[0].vector_rank, Some(1));
        // Present in both lists: weighted sum of both RRF terms,
        // computed from the constant so the test tracks retunes.
        let expected = 2.0 / (RRF_K + 1.0) + 1.0 / (RRF_K + 1.0);
        assert!((hits[0].score - expected).abs() < 1e-9);
    }

    /// Mock reranker: reverses candidate order with descending scores.
    struct ReverseReranker;

    impl crate::embed::Rerank for ReverseReranker {
        fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>, Error> {
            Ok((0..docs.len()).rev().map(|i| (i, i as f32)).collect())
        }
    }

    #[test]
    fn reranker_reorders_and_limits() {
        let dir = workspace_with(&[
            ("a.txt", "supersonic_ferret zoology\n"),
            ("b.txt", "supersonic_ferret safari\n"),
        ]);
        index::sync(dir.path()).expect("sync");
        let plain = hybrid(dir.path(), "supersonic_ferret", 2, None, None).expect("hybrid");
        assert_eq!(plain.len(), 2);
        let hits = hybrid(
            dir.path(),
            "supersonic_ferret",
            1,
            None,
            Some(&ReverseReranker),
        )
        .expect("hybrid");
        assert_eq!(hits.len(), 1);
        // Reversed: the reranked head is the plain tail.
        assert_eq!(hits[0].path, plain[1].path);
    }

    #[test]
    fn apply_ast_empty_is_noop() {
        let mut fused = vec![FusedHit {
            path: std::path::PathBuf::from("/ws/a.rs"),
            start: 1,
            end: 10,
            breadcrumb: "f".to_owned(),
            text: "body".to_owned(),
            score: 1.0,
            lexical_rank: Some(1),
            vector_rank: None,
            ast_rank: None,
        }];
        let before = fused.clone();
        apply_ast(&mut fused, &[]);
        assert_eq!(fused, before);
    }

    #[test]
    fn apply_ast_boosts_overlapping_slot_and_adds_new_ones() {
        let mut fused = vec![
            FusedHit {
                path: std::path::PathBuf::from("/ws/a.rs"),
                start: 1,
                end: 10,
                breadcrumb: "f".to_owned(),
                text: "body".to_owned(),
                score: 0.01,
                lexical_rank: Some(2),
                vector_rank: None,
                ast_rank: None,
            },
            FusedHit {
                path: std::path::PathBuf::from("/ws/b.rs"),
                start: 1,
                end: 5,
                breadcrumb: "g".to_owned(),
                text: "other".to_owned(),
                score: 0.005,
                lexical_rank: Some(1),
                vector_rank: None,
                ast_rank: None,
            },
        ];
        // Line 3 overlaps the a.rs slot: RRF term lifts it above b.rs.
        // Line 99 matches nothing: becomes its own entry tagged "ast".
        apply_ast(
            &mut fused,
            &[
                (std::path::PathBuf::from("/ws/a.rs"), 3),
                (std::path::PathBuf::from("/ws/c.rs"), 99),
            ],
        );
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].path, std::path::PathBuf::from("/ws/a.rs"));
        assert_eq!(fused[0].ast_rank, Some(1));
        assert_eq!(source(&fused[0]), "bm25");
        let added = fused
            .iter()
            .find(|h| h.path.ends_with("c.rs"))
            .expect("added");
        assert_eq!(source(added), "ast");
        assert_eq!(added.ast_rank, Some(2));
    }

    #[test]
    fn fused_hit_serializes_for_json_output() {
        let hit = FusedHit {
            path: std::path::PathBuf::from("/ws/a.rs"),
            start: 3,
            end: 9,
            breadcrumb: "m > f".to_owned(),
            text: "body".to_owned(),
            score: 1.5,
            lexical_rank: Some(1),
            vector_rank: None,
            ast_rank: None,
        };
        let value = serde_json::to_value(&hit).expect("json");
        assert_eq!(value["start"], 3);
        assert_eq!(value["breadcrumb"], "m > f");
        assert_eq!(value["score"], 1.5);
    }
}
