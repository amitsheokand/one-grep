use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "one-grep", about = "Local-first hybrid workspace search")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Index a workspace for hybrid search.
    Index { path: std::path::PathBuf },
    /// Query an indexed workspace (BM25 over chunks).
    Query {
        /// Query text.
        query: String,
        /// Workspace root to search.
        #[arg(long, default_value = ".")]
        path: std::path::PathBuf,
        /// Maximum hits to print.
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Fuse BM25 with vector similarity (RRF). Loads the ONNX model
        /// matching the vector store.
        #[arg(long)]
        hybrid: bool,
        /// Rescore fused top-20 with a cross-encoder (implies hybrid).
        #[arg(long)]
        rerank: bool,
    },
    /// Embed workspace chunks for hybrid search.
    Embed {
        path: std::path::PathBuf,
        /// Embedding model: minilm (default), arctic-m, gemma-300m.
        #[arg(long, default_value = "minilm")]
        model: String,
    },
    /// Watch a workspace and re-sync the index on changes.
    Watch { path: std::path::PathBuf },
    /// Dump extracted chunks as JSONL (for mining/eval).
    DumpChunks { path: std::path::PathBuf },
    /// Search workspace files directly (no index needed).
    Rg {
        /// Pattern to search for.
        pattern: String,
        /// Workspace root to search.
        #[arg(default_value = ".")]
        path: std::path::PathBuf,
        /// Treat pattern as regex instead of literal text.
        #[arg(long)]
        regex: bool,
        /// Case-insensitive matching.
        #[arg(long)]
        case_insensitive: bool,
        /// Maximum hits to print.
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Serve the MCP server.
    Serve {
        /// Use stdio transport instead of HTTP.
        #[arg(long)]
        stdio: bool,
        /// Loopback port for HTTP mode.
        #[arg(long, default_value_t = one_grep::install::DEFAULT_PORT)]
        port: u16,
    },
    /// Register one-grep with an agent harness.
    Install {
        /// Agent target: opencode, cursor, pi, muse, hermes, command-code.
        #[arg(long, default_value = "opencode")]
        target: String,
        /// Register the HTTP endpoint instead of stdio (opencode only).
        #[arg(long)]
        http: bool,
        /// Loopback port for `--http`.
        #[arg(long, default_value_t = one_grep::install::DEFAULT_PORT)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Index { path } => {
            let stats = one_grep::index::sync(&path)?;
            println!(
                "indexed {}: {} scanned, {} upserted, {} chunks, {} removed",
                path.display(),
                stats.scanned,
                stats.upserted,
                stats.chunks,
                stats.removed
            );
        }
        Command::Query {
            query,
            path,
            limit,
            hybrid,
            rerank,
        } => {
            if !one_grep::index::is_indexed(&path) {
                println!("{}", one_grep::rg::unindexed_note(&path));
                for hit in one_grep::rg::fallback_search(&path, &query, limit)? {
                    println!(
                        "{}:{}-{} [rg-fallback] (live)\n{}",
                        hit.path.display(),
                        hit.line,
                        hit.line,
                        hit.text
                    );
                }
                return Ok(());
            }
            if hybrid || rerank {
                let provider = match one_grep::vectors::store_model(&path)? {
                    Some(name) => one_grep::embed::FastembedProvider::load_model(
                        one_grep::embed::OnnxModel::parse_stored(&name)?,
                    )?,
                    None => one_grep::embed::FastembedProvider::load()?,
                };
                let reranker = rerank
                    .then(one_grep::embed::JinaReranker::load)
                    .transpose()?;
                for hit in one_grep::fuse::hybrid(
                    &path,
                    &query,
                    limit,
                    Some(&provider),
                    reranker.as_ref().map(|r| r as &dyn one_grep::embed::Rerank),
                )? {
                    println!(
                        "{}:{}-{} [{}] ({:.4})\n{}",
                        hit.path.display(),
                        hit.start,
                        hit.end,
                        hit.breadcrumb,
                        hit.score,
                        hit.text
                    );
                }
                return Ok(());
            }
            for hit in one_grep::index::search(&path, &query, limit)? {
                println!(
                    "{}:{}-{} [{}] ({:.2})\n{}",
                    hit.path.display(),
                    hit.start,
                    hit.end,
                    hit.breadcrumb,
                    hit.score,
                    hit.text
                );
            }
        }
        Command::Embed { path, model } => {
            let provider = one_grep::embed::FastembedProvider::load_model(
                one_grep::embed::OnnxModel::parse(&model)?,
            )?;
            let stats = one_grep::vectors::sync(&path, &provider)?;
            println!(
                "embedded {}: {} new, {} removed, {} total",
                path.display(),
                stats.embedded,
                stats.removed,
                stats.total
            );
        }
        Command::Watch { path } => {
            one_grep::watch::run(&path)?;
        }
        Command::DumpChunks { path } => {
            for entry in one_grep::engine::walker(&path).filter_map(Result::ok) {
                let file = entry.path();
                if !file.is_file() {
                    continue;
                }
                let rel = file
                    .strip_prefix(&path)
                    .map_err(|_| anyhow::anyhow!("path escapes workspace: {}", file.display()))?
                    .to_string_lossy()
                    .into_owned();
                for chunk in one_grep::extract::extract_file(file)? {
                    println!(
                        "{}",
                        serde_json::json!({
                            "path": rel,
                            "start": chunk.start,
                            "end": chunk.end,
                            "kind": match chunk.kind {
                                one_grep::extract::ChunkKind::Symbol => "symbol",
                                one_grep::extract::ChunkKind::Section => "section",
                                one_grep::extract::ChunkKind::Window => "window",
                            },
                            "breadcrumb": chunk.breadcrumb,
                            "text": chunk.text,
                        })
                    );
                }
            }
        }
        Command::Rg {
            pattern,
            path,
            regex,
            case_insensitive,
            limit,
        } => {
            let options = one_grep::rg::Options {
                regex,
                case_insensitive,
                globs: Vec::new(),
                limit,
            };
            for hit in one_grep::rg::search(&path, &pattern, &options)? {
                println!("{}:{}:{}", hit.path.display(), hit.line, hit.text);
            }
        }
        Command::Serve { stdio, port } => {
            if stdio {
                one_grep::mcp::serve_stdio().await?;
            } else {
                let token = one_grep::install::ensure_token()?;
                one_grep::mcp::serve_http(port, &token).await?;
            }
        }
        Command::Install { target, http, port } => {
            let path = one_grep::install::install(&target, http, port)?;
            println!("registered one-grep in {}", path.display());
        }
    }
    Ok(())
}
