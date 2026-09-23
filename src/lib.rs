//! one-grep library root.
//!
//!mirrors zg engine boundary: indexing + retrieval paths behind one API
//! so CLI and MCP server share behavior.

pub mod chains;
pub mod embed;
pub mod engine;
pub mod error;
pub mod eval;
pub mod extract;
pub mod fuse;
pub mod index;
pub mod install;
pub mod jev;
pub mod lsp;
pub mod mcp;
pub mod rg;
pub mod route;
pub mod vectors;
pub mod watch;

pub use error::Error;
