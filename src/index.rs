//! Indexed retrieval: BM25 over extracted chunks with manifest freshness.
//!
//! The index lives at `<workspace>/.one-grep/`. [`sync`] incrementally
//! re-indexes only new, changed, or removed files; [`search`] runs a BM25
//! query over chunk text and breadcrumbs.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use tantivy::{
    Index, Term,
    collector::TopDocs,
    directory::MmapDirectory,
    doc,
    query::QueryParser,
    schema::{STORED, STRING, Schema, TEXT, Value},
};

use crate::{Error, engine, extract};

/// Files larger than this are recorded but not indexed.
const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024;
/// Writer heap in bytes.
const WRITER_HEAP: usize = 50_000_000;
/// Manifest filename inside the index dir.
const MANIFEST: &str = "manifest.json";
/// Extractor version marker inside the index dir.
const EXTRACT_VERSION_FILE: &str = "extract.version";

/// Whether the index was built by the current extractor. Missing or
/// mismatched markers mean chunking rules changed since indexing.
#[must_use]
pub fn extractor_current(workspace: &Path) -> bool {
    std::fs::read_to_string(engine::index_dir(workspace).join(EXTRACT_VERSION_FILE))
        .map(|v| v.trim() == extract::EXTRACT_VERSION)
        .unwrap_or(false)
}

fn stamp_extractor(workspace: &Path) -> Result<(), Error> {
    std::fs::write(
        engine::index_dir(workspace).join(EXTRACT_VERSION_FILE),
        format!("{}\n", extract::EXTRACT_VERSION),
    )?;
    Ok(())
}

/// One ranked chunk hit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankedHit {
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
    /// BM25 score.
    pub score: f32,
}

/// Outcome of [`sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Files walked.
    pub scanned: usize,
    /// Files (re-)indexed.
    pub upserted: usize,
    /// Chunks written.
    pub chunks: usize,
    /// Files removed from the index.
    pub removed: usize,
}

#[derive(Debug, Clone, Copy)]
struct Fields {
    path: tantivy::schema::Field,
    start: tantivy::schema::Field,
    end: tantivy::schema::Field,
    kind: tantivy::schema::Field,
    breadcrumb: tantivy::schema::Field,
    text: tantivy::schema::Field,
}

fn schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let fields = Fields {
        path: builder.add_text_field("path", STRING | STORED),
        start: builder.add_u64_field("start", STORED),
        end: builder.add_u64_field("end", STORED),
        kind: builder.add_text_field("kind", STRING | STORED),
        breadcrumb: builder.add_text_field("breadcrumb", TEXT | STORED),
        text: builder.add_text_field("text", TEXT | STORED),
    };
    (builder.build(), fields)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FileMeta {
    mtime_secs: u64,
    mtime_nanos: u32,
    len: u64,
}

fn file_meta(path: &Path) -> Result<FileMeta, Error> {
    let metadata = std::fs::metadata(path)?;
    let mtime = metadata.modified()?;
    let duration = mtime
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    Ok(FileMeta {
        mtime_secs: duration.as_secs(),
        mtime_nanos: duration.subsec_nanos(),
        len: metadata.len(),
    })
}

fn manifest_path(workspace: &Path) -> PathBuf {
    engine::index_dir(workspace).join(MANIFEST)
}

fn load_manifest(workspace: &Path) -> Result<HashMap<String, FileMeta>, Error> {
    let path = manifest_path(workspace);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn save_manifest(workspace: &Path, manifest: &HashMap<String, FileMeta>) -> Result<(), Error> {
    std::fs::write(manifest_path(workspace), serde_json::to_vec(manifest)?)?;
    Ok(())
}

fn kind_label(kind: extract::ChunkKind) -> &'static str {
    match kind {
        extract::ChunkKind::Symbol => "symbol",
        extract::ChunkKind::Section => "section",
        extract::ChunkKind::Window => "window",
    }
}

/// Incrementally index `workspace`: new/changed files are (re-)indexed,
/// removed files are dropped. Creates the index on first run.
///
/// # Errors
///
/// Returns [`Error`] when the workspace is not a directory or index,
/// filesystem, or manifest operations fail.
pub fn sync(workspace: &Path) -> Result<Stats, Error> {
    if !workspace.is_dir() {
        return Err(Error::InvalidInput(format!(
            "workspace is not a directory: {}",
            workspace.display()
        )));
    }
    let dir = engine::index_dir(workspace);
    std::fs::create_dir_all(&dir)?;
    let (schema, fields) = schema();
    let index = Index::open_or_create(MmapDirectory::open(&dir)?, schema)?;
    let mut writer = index.writer(WRITER_HEAP)?;

    let mut manifest = load_manifest(workspace)?;
    // Extractor upgrade: chunking rules changed, so mtime matches no
    // longer prove chunks are current. Re-extract everything; chunk IDs
    // of unchanged content are stable, so vectors stay incremental.
    if !extractor_current(workspace) {
        manifest.clear();
    }
    let mut seen = Vec::new();
    let mut stats = Stats::default();

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
        seen.push(rel.clone());
        stats.scanned += 1;
        let meta = file_meta(path)?;
        if manifest.get(&rel) == Some(&meta) {
            continue;
        }
        writer.delete_term(Term::from_field_text(fields.path, &rel));
        let mut chunks = 0;
        if meta.len <= MAX_FILE_BYTES {
            for chunk in extract::extract_file(path)? {
                chunks += 1;
                writer.add_document(doc!(
                    fields.path => rel.clone(),
                    fields.start => chunk.start,
                    fields.end => chunk.end,
                    fields.kind => kind_label(chunk.kind),
                    fields.breadcrumb => chunk.breadcrumb,
                    fields.text => chunk.text,
                ))?;
            }
        }
        manifest.insert(rel, meta);
        stats.upserted += 1;
        stats.chunks += chunks;
    }

    for rel in manifest
        .keys()
        .filter(|k| !seen.contains(k))
        .cloned()
        .collect::<Vec<_>>()
    {
        writer.delete_term(Term::from_field_text(fields.path, &rel));
        manifest.remove(&rel);
        stats.removed += 1;
    }

    // Chain docs reference callee text, so any file change can stale them.
    // Rebuild all chains when anything changed (noop syncs skip this).
    if stats.upserted > 0 || stats.removed > 0 {
        writer.delete_term(Term::from_field_text(fields.kind, "chain"));
        let mut chains = 0;
        for chunk in crate::chains::extract_workspace(workspace)? {
            let rel = chunk.path.to_string_lossy().into_owned();
            chains += 1;
            writer.add_document(doc!(
                fields.path => rel,
                fields.start => chunk.start,
                fields.end => chunk.end,
                fields.kind => "chain",
                fields.breadcrumb => chunk.breadcrumb,
                fields.text => chunk.text,
            ))?;
        }
        stats.chunks += chains;
    }

    writer.commit()?;
    save_manifest(workspace, &manifest)?;
    stamp_extractor(workspace)?;
    Ok(stats)
}

/// Whether `<workspace>/.one-grep/` exists (BM25 index present).
#[must_use]
pub fn is_indexed(workspace: &Path) -> bool {
    engine::index_dir(workspace).is_dir()
}

/// Whether the index no longer matches the tree: any walked file whose
/// (mtime, len) differs from the manifest, or any manifest entry with no
/// file behind it. A missing manifest is not "stale" — callers already
/// report that case as unindexed. An extractor upgrade is stale too.
#[must_use]
pub fn is_stale(workspace: &Path) -> bool {
    if !manifest_path(workspace).exists() {
        return false;
    }
    if !extractor_current(workspace) {
        return true;
    }
    let manifest = match load_manifest(workspace) {
        Ok(m) => m,
        Err(_) => return true,
    };
    let mut seen = std::collections::HashSet::new();
    for entry in engine::walker(workspace).filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(rel) = path
            .strip_prefix(workspace)
            .map(|r| r.to_string_lossy().into_owned())
        else {
            return true;
        };
        seen.insert(rel.clone());
        let meta = match file_meta(path) {
            Ok(m) => m,
            Err(_) => return true,
        };
        if manifest.get(&rel) != Some(&meta) {
            return true;
        }
    }
    manifest.keys().any(|k| !seen.contains(k))
}

/// Human note pointing at the fix. Shown next to index-backed results.
#[must_use]
pub fn stale_note(workspace: &Path) -> String {
    format!(
        "index: stale (tree changed since last sync; run: one-grep index {})",
        workspace.display()
    )
}

/// Run a BM25 query over chunk text and breadcrumbs.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when the workspace has no index or the
/// query fails to parse, and [`Error`] when search itself fails.
pub fn search(workspace: &Path, query: &str, limit: usize) -> Result<Vec<RankedHit>, Error> {
    let dir = engine::index_dir(workspace);
    if !is_indexed(workspace) {
        return Err(Error::InvalidInput(format!(
            "workspace is not indexed (no {}); run index first",
            dir.display()
        )));
    }
    let (_schema, fields) = schema();
    let index = Index::open(MmapDirectory::open(&dir)?)?;
    let reader = index.reader()?;
    reader.reload()?;
    let searcher = reader.searcher();
    let parser = QueryParser::for_index(&index, vec![fields.text, fields.breadcrumb]);
    let parsed = parser
        .parse_query(query)
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    let top = searcher.search(&parsed, &TopDocs::with_limit(limit))?;
    let mut hits = Vec::with_capacity(top.len());
    for (score, addr) in top {
        let retrieved: tantivy::TantivyDocument = searcher.doc(addr)?;
        let text = |f| {
            retrieved
                .get_first(f)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned()
        };
        let rel = text(fields.path);
        hits.push(RankedHit {
            path: workspace.join(&rel),
            start: retrieved
                .get_first(fields.start)
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            end: retrieved
                .get_first(fields.end)
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            breadcrumb: text(fields.breadcrumb),
            text: text(fields.text),
            score,
        });
    }
    Ok(hits)
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
    fn sync_then_search_finds_symbol_text() {
        let dir = workspace_with(&[("lib.rs", "pub fn supersonic_ferret() {}\n")]);
        let stats = sync(dir.path()).expect("sync");
        assert_eq!(stats.upserted, 1);
        assert_eq!(stats.chunks, 1);
        let hits = search(dir.path(), "supersonic_ferret", 10).expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].breadcrumb, "supersonic_ferret");
        assert_eq!((hits[0].start, hits[0].end), (1, 1));
    }

    #[test]
    fn second_sync_is_noop() {
        let dir = workspace_with(&[("a.txt", "alpha bravo\n")]);
        sync(dir.path()).expect("sync");
        let stats = sync(dir.path()).expect("resync");
        assert_eq!(stats.upserted, 0);
        assert_eq!(stats.removed, 0);
    }

    #[test]
    fn changed_file_is_reindexed() {
        let dir = workspace_with(&[("a.txt", "alpha bravo\n")]);
        sync(dir.path()).expect("sync");
        assert_eq!(search(dir.path(), "bravo", 10).expect("search").len(), 1);
        std::fs::write(dir.path().join("a.txt"), "alpha charlie\n").expect("write");
        let stats = sync(dir.path()).expect("resync");
        assert_eq!(stats.upserted, 1);
        assert!(search(dir.path(), "bravo", 10).expect("search").is_empty());
        assert_eq!(search(dir.path(), "charlie", 10).expect("search").len(), 1);
    }

    #[test]
    fn deleted_file_is_dropped() {
        let dir = workspace_with(&[("a.txt", "unique_zephyr_word\n")]);
        sync(dir.path()).expect("sync");
        std::fs::remove_file(dir.path().join("a.txt")).expect("remove");
        let stats = sync(dir.path()).expect("resync");
        assert_eq!(stats.removed, 1);
        assert!(
            search(dir.path(), "unique_zephyr_word", 10)
                .expect("search")
                .is_empty()
        );
    }

    #[test]
    fn search_without_index_is_invalid_input() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(!is_indexed(dir.path()));
        let err = search(dir.path(), "x", 10).expect_err("must fail");
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn is_indexed_after_sync() {
        let dir = workspace_with(&[("a.txt", "alpha\n")]);
        assert!(!is_indexed(dir.path()));
        sync(dir.path()).expect("sync");
        assert!(is_indexed(dir.path()));
    }

    #[test]
    fn extractor_upgrade_forces_full_reindex() {
        let dir = workspace_with(&[("a.txt", "alpha\n"), ("b.txt", "beta\n")]);
        sync(dir.path()).expect("sync");
        assert!(extractor_current(dir.path()));
        // Simulate an index built by older chunking rules.
        std::fs::write(engine::index_dir(dir.path()).join("extract.version"), "0\n")
            .expect("tamper");
        assert!(!extractor_current(dir.path()));
        assert!(is_stale(dir.path()));
        let stats = sync(dir.path()).expect("resync");
        assert_eq!(stats.upserted, 2);
        assert!(extractor_current(dir.path()));
        assert!(!is_stale(dir.path()));
    }

    #[test]
    fn stale_lifecycle_touch_resync() {
        let dir = workspace_with(&[("a.txt", "alpha\n")]);
        // No manifest yet: unindexed, not stale (separate note owns that).
        assert!(!is_stale(dir.path()));
        sync(dir.path()).expect("sync");
        assert!(!is_stale(dir.path()));
        // Touch a file with different bytes: dirty.
        std::fs::write(dir.path().join("a.txt"), "alpha plus more\n").expect("write");
        assert!(is_stale(dir.path()));
        assert!(stale_note(dir.path()).contains("one-grep index"));
        // Removed file: dirty too.
        std::fs::remove_file(dir.path().join("a.txt")).expect("remove");
        assert!(is_stale(dir.path()));
        // Re-sync clears it.
        sync(dir.path()).expect("resync");
        assert!(!is_stale(dir.path()));
    }
}
