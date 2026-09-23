use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "one-grep",
    version,
    about = "Local-first hybrid workspace search"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Ranking backend for `query`. An enum (not a string) so illegal values
/// are rejected by the parser: the type checker closes the port before any
/// solver or meter runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RankBackend {
    /// TypeSafe Jev Nouls over the closed shortlist.
    Jev,
    /// Local Jina cross-encoder over fused top-20.
    Jina,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Index a workspace for hybrid search.
    Index { path: std::path::PathBuf },
    /// Query an indexed workspace (BM25 over chunks).
    #[command(alias = "search")]
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
        /// Rescore fused top-20 with a local cross-encoder (implies hybrid).
        /// Conflicts with `--rank`; prefer `--rank jina`.
        #[arg(long, conflicts_with = "rank")]
        rerank: bool,
        /// Rank the retrieval shortlist inside the tool.
        #[arg(long, value_enum)]
        rank: Option<RankBackend>,
        /// Fuse ast-grep structural hits as a third RRF list (implies
        /// hybrid; needs `--ast-lang` and the `ast-grep` binary on PATH).
        #[arg(long, value_name = "PATTERN")]
        ast: Option<String>,
        /// Language for `--ast` (same vocabulary as `rg --lang`).
        #[arg(long, value_name = "LANG")]
        ast_lang: Option<String>,
        /// Emit hits as a JSON array instead of human-readable text.
        #[arg(long)]
        json: bool,
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
        /// Pattern to search for (raw text; one outer "..." / '...' pair is ignored).
        pattern: String,
        /// Workspace root to search.
        #[arg(default_value = ".")]
        path: std::path::PathBuf,
        /// Treat pattern as regex instead of literal text.
        #[arg(long)]
        regex: bool,
        /// Interpret the pattern structurally via ast-grep (needs exactly
        /// one `--lang`; requires the `ast-grep` binary on PATH).
        #[arg(long)]
        structural: bool,
        /// Case-insensitive matching.
        #[arg(long)]
        case_insensitive: bool,
        /// Restrict to languages (`rust`, `python`, `typescript`, `go`,
        /// `java`, `nix`, `markdown`, or extensions like `rs`).
        /// Repeatable.
        #[arg(long)]
        lang: Vec<String>,
        /// Restrict to ignore-style globs (whitelist, `!` negates).
        /// Repeatable.
        #[arg(long)]
        glob: Vec<String>,
        /// Maximum hits to print.
        #[arg(long, default_value_t = 100)]
        limit: usize,
        /// Emit hits as a JSON array instead of `path:line:text` lines.
        #[arg(long)]
        json: bool,
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

/// Print a value as a JSON array for `--json` machine output.
fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// First line of a possibly multi-line match, for `path:line:text` output.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or("")
}

/// Fuse `--ast` structural hits as a third RRF list. No-op without `--ast`.
fn fuse_ast(
    path: &std::path::Path,
    hits: &mut Vec<one_grep::fuse::FusedHit>,
    limit: usize,
    ast: &Option<String>,
    ast_lang: &Option<String>,
) -> Result<()> {
    let (Some(pattern), Some(lang)) = (ast, ast_lang) else {
        return Ok(());
    };
    let found = one_grep::astg::search(path, pattern, lang, limit)?;
    let refs: Vec<(std::path::PathBuf, u64)> =
        found.into_iter().map(|h| (h.path, h.line)).collect();
    one_grep::fuse::apply_ast(hits, &refs);
    hits.truncate(limit);
    Ok(())
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
            rank,
            ast,
            ast_lang,
            json,
        } => {
            // `--rerank` and `--rank` conflict at parse time, so this match
            // is exhaustive: illegal combos are unwired, not runtime errors.
            let backend = match (rerank, rank) {
                (true, None) => Some(RankBackend::Jina),
                (false, r) => r,
                (true, Some(_)) => {
                    unreachable!("clap conflicts_with rejects --rerank with --rank")
                }
            };
            let rank_jev = backend == Some(RankBackend::Jev);
            let rank_jina = backend == Some(RankBackend::Jina);
            // `--ast` implies hybrid retrieval: the structural hits fuse as
            // a third RRF list over the fused shortlist.
            if ast.is_some() && ast_lang.is_none() {
                anyhow::bail!("--ast needs --ast-lang");
            }
            if ast.is_some() && !one_grep::astg::available() {
                anyhow::bail!("ast-grep not found on PATH (or ONE_GREP_AST_GREP)");
            }
            let hybrid = hybrid || ast.is_some();
            let indexed = one_grep::index::is_indexed(&path);
            // Shared router: explicit flags force intent; otherwise exact
            // anchors go rg and single tokens go BM25, like MCP `search`.
            if !hybrid && !rank_jev && !rank_jina {
                match one_grep::route::route(&query, &[]) {
                    one_grep::route::Route::Literal(pattern) => {
                        let options = one_grep::rg::Options {
                            regex: false,
                            case_insensitive: false,
                            globs: Vec::new(),
                            langs: Vec::new(),
                            limit,
                        };
                        let hits = one_grep::rg::search(&path, &pattern, &options)?;
                        if json {
                            print_json(&hits)?;
                            return Ok(());
                        }
                        for hit in hits {
                            println!("{}:{}:{}", hit.path.display(), hit.line, hit.text);
                        }
                        return Ok(());
                    }
                    one_grep::route::Route::Identifier(term) => {
                        if indexed {
                            let hits = one_grep::index::search(&path, &term, limit)?;
                            if json {
                                if one_grep::index::is_stale(&path) {
                                    eprintln!("{}", one_grep::index::stale_note(&path));
                                }
                                print_json(&hits)?;
                                return Ok(());
                            }
                            if one_grep::index::is_stale(&path) {
                                println!("{}", one_grep::index::stale_note(&path));
                            }
                            for hit in hits {
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
                            return Ok(());
                        }
                        // Unindexed identifiers fall through to live rg below.
                    }
                    one_grep::route::Route::Intent(_) => {}
                }
            }
            if rank_jev {
                let mut hits = if indexed && (hybrid || rank_jev) {
                    let provider = match one_grep::vectors::store_model(&path)? {
                        Some(name) => one_grep::embed::FastembedProvider::load_model(
                            one_grep::embed::OnnxModel::parse_stored(&name)?,
                        )?,
                        None => one_grep::embed::FastembedProvider::load()?,
                    };
                    one_grep::fuse::collect(&path, &query, limit, true, Some(&provider))?
                } else {
                    one_grep::fuse::collect(&path, &query, limit, false, None)?
                };
                fuse_ast(&path, &mut hits, limit, &ast, &ast_lang)?;
                let q = query.clone();
                let (status, hits) = tokio::task::spawn_blocking(move || {
                    one_grep::jev::rerank_hits(&q, hits)
                })
                .await?;
                if json {
                    eprintln!("{}", status.note);
                    if !indexed {
                        eprintln!("{}", one_grep::rg::unindexed_note(&path));
                    }
                    if indexed && one_grep::index::is_stale(&path) {
                        eprintln!("{}", one_grep::index::stale_note(&path));
                    }
                    if indexed && one_grep::vectors::is_stale(&path) {
                        eprintln!("{}", one_grep::vectors::stale_note(&path));
                    }
                    print_json(&hits)?;
                    return Ok(());
                }
                println!("{}", status.note);
                if !indexed {
                    println!("{}", one_grep::rg::unindexed_note(&path));
                }
                if indexed && one_grep::index::is_stale(&path) {
                    println!("{}", one_grep::index::stale_note(&path));
                }
                if indexed && one_grep::vectors::is_stale(&path) {
                    println!("{}", one_grep::vectors::stale_note(&path));
                }
                for hit in hits {
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
            if !indexed {
                let live = one_grep::rg::fallback_search(&path, &query, limit)?;
                // Structural fusion applies even without an index: the live
                // hits become single-line fused slots first.
                if ast.is_some() {
                    let mut fused: Vec<one_grep::fuse::FusedHit> = live
                        .into_iter()
                        .map(|h| one_grep::fuse::FusedHit {
                            path: h.path,
                            start: h.line,
                            end: h.line,
                            breadcrumb: "rg-fallback".to_owned(),
                            text: h.text,
                            score: 0.0,
                            lexical_rank: None,
                            vector_rank: None,
                            ast_rank: None,
                        })
                        .collect();
                    fuse_ast(&path, &mut fused, limit, &ast, &ast_lang)?;
                    if json {
                        eprintln!("{}", one_grep::rg::unindexed_note(&path));
                        print_json(&fused)?;
                        return Ok(());
                    }
                    println!("{}", one_grep::rg::unindexed_note(&path));
                    for hit in fused {
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
                let hits = live;
                if json {
                    eprintln!("{}", one_grep::rg::unindexed_note(&path));
                    print_json(&hits)?;
                    return Ok(());
                }
                println!("{}", one_grep::rg::unindexed_note(&path));
                for hit in hits {
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
            if hybrid || rank_jina {
                let provider = match one_grep::vectors::store_model(&path)? {
                    Some(name) => one_grep::embed::FastembedProvider::load_model(
                        one_grep::embed::OnnxModel::parse_stored(&name)?,
                    )?,
                    None => one_grep::embed::FastembedProvider::load()?,
                };
                let reranker = rank_jina
                    .then(one_grep::embed::JinaReranker::load)
                    .transpose()?;
                let mut hits = one_grep::fuse::hybrid(
                    &path,
                    &query,
                    limit,
                    Some(&provider),
                    reranker.as_ref().map(|r| r as &dyn one_grep::embed::Rerank),
                )?;
                fuse_ast(&path, &mut hits, limit, &ast, &ast_lang)?;
                if json {
                    if one_grep::index::is_stale(&path) {
                        eprintln!("{}", one_grep::index::stale_note(&path));
                    }
                    if one_grep::vectors::is_stale(&path) {
                        eprintln!("{}", one_grep::vectors::stale_note(&path));
                    }
                    print_json(&hits)?;
                    return Ok(());
                }
                if one_grep::index::is_stale(&path) {
                    println!("{}", one_grep::index::stale_note(&path));
                }
                if one_grep::vectors::is_stale(&path) {
                    println!("{}", one_grep::vectors::stale_note(&path));
                }
                for hit in hits {
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
            let hits = one_grep::index::search(&path, &query, limit)?;
            if json {
                if one_grep::index::is_stale(&path) {
                    eprintln!("{}", one_grep::index::stale_note(&path));
                }
                print_json(&hits)?;
                return Ok(());
            }
            if one_grep::index::is_stale(&path) {
                println!("{}", one_grep::index::stale_note(&path));
            }
            for hit in hits {
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
            structural,
            case_insensitive,
            lang,
            glob,
            limit,
            json,
        } => {
            if structural {
                if regex {
                    anyhow::bail!("--structural and --regex cannot be combined");
                }
                let [lang] = lang.as_slice() else {
                    anyhow::bail!("--structural needs exactly one --lang");
                };
                if !one_grep::astg::available() {
                    anyhow::bail!("ast-grep not found on PATH (or ONE_GREP_AST_GREP)");
                }
                let hits = one_grep::astg::search(&path, &pattern, lang, limit)?;
                if json {
                    print_json(&hits)?;
                    return Ok(());
                }
                for hit in hits {
                    println!(
                        "{}:{}:{}",
                        hit.path.display(),
                        hit.line,
                        first_line(&hit.text)
                    );
                }
                return Ok(());
            }
            let options = one_grep::rg::Options {
                regex,
                case_insensitive,
                globs: glob,
                langs: lang,
                limit,
            };
            let hits = one_grep::rg::search(&path, &pattern, &options)?;
            if json {
                print_json(&hits)?;
                return Ok(());
            }
            for hit in hits {
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
