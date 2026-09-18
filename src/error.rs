//! Crate error type (library side uses `thiserror`, binaries map to `anyhow`).

use thiserror::Error;

/// All errors produced by `one_grep`.
#[derive(Debug, Error)]
pub enum Error {
    /// Filesystem or workspace access failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Caller passed an invalid workspace path or query.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Pattern failed to compile.
    #[error("bad pattern: {0}")]
    BadPattern(#[from] grep::regex::Error),
    /// Workspace walk failed.
    #[error("walk: {0}")]
    Walk(#[from] ignore::Error),
    /// Index read/write failed.
    #[error("index: {0}")]
    Index(#[from] tantivy::TantivyError),
    /// Index directory could not be opened.
    #[error("index dir: {0}")]
    IndexDir(#[from] tantivy::directory::error::OpenDirectoryError),
    /// Manifest read/write failed.
    #[error("manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    /// Embedding failed.
    #[error("embed: {0}")]
    Embed(String),
    /// TypeSafe / Jev ranking failed. Messages never include the API key.
    #[error("jev: {0}")]
    Jev(String),
}
