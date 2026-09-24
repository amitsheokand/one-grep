//! Vector store: chunk embeddings persisted as JSON beside the index.
//!
//! [`sync`] embeds chunks missing from the store and drops stale entries.
//! [`topk`] ranks items by cosine similarity (brute force; HNSW later).

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{Error, embed::EmbedProvider, engine, extract};

/// Store filename inside the index dir.
const STORE: &str = "vectors.json";
/// Marker written when the index moves on without the vectors (e.g. `watch`
/// re-syncs the index only). Cleared by the next successful [`sync`].
const STALE_MARKER: &str = "vectors.stale";
/// Texts per embedding call. Small: batches pad to the longest sequence,
/// so big batches of long code chunks waste most work on padding.
const BATCH: usize = 16;
/// Embeddings truncate past the model window anyway; cut client-side to
/// avoid tokenizing megabytes. ~1500 chars ≈ 380 tokens: signature plus
/// head of body, and keeps batch padding small.
const EMBED_CHARS: usize = 1500;

/// One stored chunk + its vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorItem {
    /// Workspace-relative path string.
    pub rel: String,
    /// 1-based first line.
    pub start: u64,
    /// 1-based last line (inclusive).
    pub end: u64,
    /// Chunk kind label.
    pub kind: String,
    /// Symbol path or heading stack.
    pub breadcrumb: String,
    /// Chunk text.
    pub text: String,
    /// L2-normalized embedding.
    pub embedding: Vec<f32>,
}

/// Persisted store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Store {
    model: String,
    dims: usize,
    items: HashMap<String, VectorItem>,
}

/// Outcome of [`sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Chunks embedded.
    pub embedded: usize,
    /// Stale chunks dropped.
    pub removed: usize,
    /// Total chunks stored.
    pub total: usize,
}

fn store_path(workspace: &Path) -> PathBuf {
    engine::index_dir(workspace).join(STORE)
}

fn load(workspace: &Path) -> Result<Store, Error> {
    let path = store_path(workspace);
    if !path.exists() {
        return Ok(Store::default());
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn save(workspace: &Path, store: &Store) -> Result<(), Error> {
    std::fs::write(store_path(workspace), serde_json::to_vec(store)?)?;
    Ok(())
}

fn chunk_id(rel: &str, start: u64, end: u64, breadcrumb: &str) -> String {
    format!("{rel}\0{start}\0{end}\0{breadcrumb}")
}

/// Whether the vector store is marked stale (index moved without it).
#[must_use]
pub fn is_stale(workspace: &Path) -> bool {
    engine::index_dir(workspace).join(STALE_MARKER).exists()
}

/// Mark the store stale. Called when the index re-syncs without embedding.
pub fn mark_stale(workspace: &Path) -> Result<(), Error> {
    std::fs::create_dir_all(engine::index_dir(workspace))?;
    std::fs::write(engine::index_dir(workspace).join(STALE_MARKER), "stale\n")?;
    Ok(())
}

/// Human note pointing at the fix. Shown next to hybrid results.
#[must_use]
pub fn stale_note(workspace: &Path) -> String {
    format!(
        "vectors: stale (index changed since last embed; run: one-grep embed {})",
        workspace.display()
    )
}

/// Queue a chunk for embedding unless the store already holds the same
/// text under the same id. Same spans with different text (edited body)
/// re-embed instead of silently serving yesterday's vector.
fn queue_chunk(
    store: &Store,
    current_ids: &mut HashSet<String>,
    pending: &mut Vec<(String, VectorItem)>,
    rel: String,
    start: u64,
    end: u64,
    kind: String,
    breadcrumb: String,
    text: String,
) {
    let id = chunk_id(&rel, start, end, &breadcrumb);
    current_ids.insert(id.clone());
    if pending.iter().any(|(pid, _)| pid == &id) {
        return;
    }
    if let Some(stored) = store.items.get(&id) {
        if stored.text == text {
            return;
        }
    }
    pending.push((
        id,
        VectorItem {
            rel,
            start,
            end,
            kind,
            breadcrumb,
            text,
            embedding: Vec::new(),
        },
    ));
}

/// Cosine similarity of L2-normalized vectors (dot product).
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cut to the embed window on a char boundary.
fn truncate(text: &str) -> &str {
    if text.len() <= EMBED_CHARS {
        return text;
    }
    let mut end = EMBED_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Embed chunks missing from the store; drop entries for gone files or a
/// changed model. Reuses the same walker filters as lexical indexing.
///
/// # Errors
///
/// Returns [`Error`] when the workspace is not a directory, files cannot
/// be read, or the provider fails.
pub fn sync(workspace: &Path, provider: &dyn EmbedProvider) -> Result<Stats, Error> {
    if !workspace.is_dir() {
        return Err(Error::InvalidInput(format!(
            "workspace is not a directory: {}",
            workspace.display()
        )));
    }
    std::fs::create_dir_all(engine::index_dir(workspace))?;
    let mut store = load(workspace)?;
    if store.model != provider.name() || store.dims != provider.dims() {
        store = Store {
            model: provider.name().to_owned(),
            dims: provider.dims(),
            items: HashMap::new(),
        };
    }

    let mut pending: Vec<(String, VectorItem)> = Vec::new();
    let mut current_ids = HashSet::new();
    for entry in engine::walker(workspace).filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(workspace)
            .map_err(|_| {
                Error::InvalidInput(format!("path escapes workspace: {}", path.display()))
            })?
            .to_string_lossy()
            .into_owned();
        for chunk in extract::extract_file_with(path, extract::EMBED_WINDOW)? {
            let kind = match chunk.kind {
                extract::ChunkKind::Symbol => "symbol".to_owned(),
                extract::ChunkKind::Section => "section".to_owned(),
                extract::ChunkKind::Window => "window".to_owned(),
            };
            queue_chunk(
                &store,
                &mut current_ids,
                &mut pending,
                rel.clone(),
                chunk.start,
                chunk.end,
                kind,
                chunk.breadcrumb,
                chunk.text,
            );
        }
    }

    // Chain chunks share IDs with the lexical index (same rel/span/crumb).
    for chunk in crate::chains::extract_workspace(workspace)? {
        let rel = chunk.path.to_string_lossy().into_owned();
        queue_chunk(
            &store,
            &mut current_ids,
            &mut pending,
            rel,
            chunk.start,
            chunk.end,
            "chain".to_owned(),
            chunk.breadcrumb,
            chunk.text,
        );
    }

    let mut embedded = 0;
    let total_pending = pending.len();
    let started = std::time::Instant::now();
    for (n, batch) in pending.chunks(BATCH).enumerate() {
        let texts: Vec<&str> = batch.iter().map(|(_, item)| truncate(&item.text)).collect();
        let vectors = provider.embed(&texts)?;
        for ((id, mut item), vector) in batch.iter().cloned().zip(vectors) {
            item.embedding = normalize(vector);
            store.items.insert(id, item);
            embedded += 1;
        }
        // Checkpoint so interrupted runs resume instead of restarting.
        if n % 10 == 9 {
            save(workspace, &store)?;
            let rate = embedded as f64 / started.elapsed().as_secs_f64().max(0.1);
            eprintln!("embed {embedded}/{total_pending} chunks ({rate:.0}/s)");
        }
    }

    let before = store.items.len();
    store.items.retain(|id, _| current_ids.contains(id));
    let removed = before - store.items.len();
    let total = store.items.len();
    save(workspace, &store)?;
    // A successful sync owns the marker: vectors match the tree again.
    let _ = std::fs::remove_file(engine::index_dir(workspace).join(STALE_MARKER));
    Ok(Stats {
        embedded,
        removed,
        total,
    })
}

/// Rank stored chunks by cosine similarity to `query`, best first.
#[must_use]
pub fn topk(store_items: &[&VectorItem], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = store_items
        .iter()
        .enumerate()
        .map(|(i, item)| (i, cosine(&item.embedding, query)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored
}

/// Load items for querying, sorted by `(rel, start, end, breadcrumb)` so the
/// order is identical in every process (the store is a `HashMap`, whose
/// iteration order varies per process and would otherwise leak into rank ties).
pub fn load_items(workspace: &Path) -> Result<Vec<VectorItem>, Error> {
    let mut items: Vec<VectorItem> = load(workspace)?.items.into_values().collect();
    // Full key order (store ids are rel/start/end/breadcrumb): ties are
    // impossible, so this order is a pure function of store content.
    items.sort_by(|a, b| {
        a.rel
            .cmp(&b.rel)
            .then_with(|| a.start.cmp(&b.start))
            .then_with(|| a.end.cmp(&b.end))
            .then_with(|| a.breadcrumb.cmp(&b.breadcrumb))
    });
    Ok(items)
}

/// Model label the store was built with, if any.
pub fn store_model(workspace: &Path) -> Result<Option<String>, Error> {
    let store = load(workspace)?;
    Ok((!store.items.is_empty()).then_some(store.model))
}

/// Stored dimensionality, if any.
pub fn store_dims(workspace: &Path) -> Result<Option<usize>, Error> {
    let store = load(workspace)?;
    Ok((!store.items.is_empty()).then_some(store.dims))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Deterministic 2-D provider for tests (no model download).
    pub struct TestProvider;

    impl EmbedProvider for TestProvider {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
            Ok(texts
                .iter()
                .map(|t| {
                    let x = t.len() as f32;
                    vec![x, 1.0]
                })
                .collect())
        }

        fn dims(&self) -> usize {
            2
        }

        fn name(&self) -> &str {
            "test/const-2d"
        }
    }

    #[test]
    fn cosine_known_values() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn sync_embeds_and_second_sync_is_noop() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.txt"), "hello world\n").expect("write");
        let stats = sync(dir.path(), &TestProvider).expect("sync");
        assert_eq!(stats.embedded, 1);
        assert_eq!(stats.total, 1);
        let again = sync(dir.path(), &TestProvider).expect("resync");
        assert_eq!(again.embedded, 0);
        assert_eq!(again.removed, 0);
    }

    #[test]
    fn sync_drops_deleted_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "hello\n").expect("write");
        sync(dir.path(), &TestProvider).expect("sync");
        std::fs::remove_file(&path).expect("remove");
        let stats = sync(dir.path(), &TestProvider).expect("resync");
        assert_eq!(stats.removed, 1);
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn sync_reembeds_edited_text_under_same_spans() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "hello world\nsecond line\n").expect("write");
        let first = sync(dir.path(), &TestProvider).expect("sync");
        assert!(first.embedded > 0);
        // Same line count (same spans), different text: must re-embed,
        // not silently serve yesterday's vector.
        std::fs::write(&path, "goodbye world\nsecond line\n").expect("rewrite");
        let second = sync(dir.path(), &TestProvider).expect("resync");
        assert_eq!(second.embedded, first.embedded);
        let items = load_items(dir.path()).expect("load");
        assert!(items.iter().any(|i| i.text.contains("goodbye")));
        assert!(!items.iter().any(|i| i.text.contains("hello world")));
        // Unchanged third sync is a noop again.
        let third = sync(dir.path(), &TestProvider).expect("resync");
        assert_eq!(third.embedded, 0);
    }

    #[test]
    fn stale_marker_round_trips_through_sync() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.txt"), "hello\n").expect("write");
        sync(dir.path(), &TestProvider).expect("sync");
        assert!(!is_stale(dir.path()));
        mark_stale(dir.path()).expect("mark");
        assert!(is_stale(dir.path()));
        assert!(stale_note(dir.path()).contains("one-grep embed"));
        sync(dir.path(), &TestProvider).expect("resync");
        assert!(!is_stale(dir.path()));
    }

    #[test]
    fn topk_orders_by_similarity() {
        let items = [
            VectorItem {
                rel: "a".to_owned(),
                start: 1,
                end: 1,
                kind: "window".to_owned(),
                breadcrumb: String::new(),
                text: String::new(),
                embedding: vec![1.0, 0.0],
            },
            VectorItem {
                rel: "b".to_owned(),
                start: 1,
                end: 1,
                kind: "window".to_owned(),
                breadcrumb: String::new(),
                text: String::new(),
                embedding: vec![0.0, 1.0],
            },
        ];
        let refs: Vec<&VectorItem> = items.iter().collect();
        let ranked = topk(&refs, &[1.0, 0.0], 2);
        assert_eq!(ranked[0].0, 0);
        assert_eq!(ranked[1].0, 1);
    }

    #[test]
    fn topk_ties_break_by_input_order() {
        let mk = |rel: &str| VectorItem {
            rel: rel.to_owned(),
            start: 1,
            end: 1,
            kind: "window".to_owned(),
            breadcrumb: String::new(),
            text: String::new(),
            embedding: vec![1.0, 0.0],
        };
        // Identical scores: `b` first in input stays first in output,
        // so rank ties can never depend on HashMap iteration order.
        let items = [mk("b"), mk("a")];
        let refs: Vec<&VectorItem> = items.iter().collect();
        let ranked = topk(&refs, &[1.0, 0.0], 2);
        assert_eq!((ranked[0].0, ranked[1].0), (0, 1));
    }

    #[test]
    fn load_items_returns_sorted_spans() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("b.txt"), "one\ntwo\nthree\n").expect("write");
        std::fs::write(dir.path().join("a.txt"), "uno\ndos\n").expect("write");
        sync(dir.path(), &TestProvider).expect("sync");
        let items = load_items(dir.path()).expect("load");
        let keys: Vec<(&str, u64, u64)> = items
            .iter()
            .map(|i| (i.rel.as_str(), i.start, i.end))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
