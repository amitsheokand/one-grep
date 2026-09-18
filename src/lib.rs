//! one-grep library root.
//!
//!mirrors zg engine boundary: indexing + retrieval paths behind one API
//! so CLI and MCP server share behavior.

pub mod chains;
pub mod embed;
pub mod engine;
pub mod error;
pub mod extract;
pub mod fuse;
pub mod index;
pub mod install;
pub mod jev;
pub mod mcp;
pub mod rg;
pub mod vectors;
pub mod watch;

pub use error::Error;
