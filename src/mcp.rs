//! MCP server: `search` + `rg` tools over stdio or loopback HTTP.
//!
//! HTTP binds 127.0.0.1 only and requires `Authorization: Bearer <token>`.

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use rmcp::{
    ErrorData as McpError, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::{
        stdio,
        streamable_http_server::{StreamableHttpService, session::local::LocalSessionManager},
    },
};

use crate::{embed::FastembedProvider, fuse, index, lsp, rg, route};
use serde::Deserialize;

/// Max chars of chunk text per tool hit.
const TEXT_CAP: usize = 1200;

fn check_root(root: &str) -> Result<PathBuf, McpError> {
    let path = PathBuf::from(root);
    if !path.is_absolute() {
        return Err(McpError::invalid_params(
            format!("root must be absolute: {root}"),
            None,
        ));
    }
    if !path.is_dir() {
        return Err(McpError::invalid_params(
            format!("root is not a directory: {root}"),
            None,
        ));
    }
    Ok(path)
}

fn internal(e: impl std::fmt::Display) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

/// Caller errors (unknown `--lang`, bad glob) are `invalid_params`; engine
/// failures stay `internal_error`. Fails closed either way.
fn rg_error(e: crate::Error) -> McpError {
    match e {
        crate::Error::InvalidInput(msg) => McpError::invalid_params(msg, None),
        crate::Error::BadPattern(e) => McpError::invalid_params(format!("bad pattern: {e}"), None),
        other => internal(other),
    }
}

/// Resolve a relative-or-absolute file against the workspace root,
/// rejecting lexical escapes. Existence is checked by the caller.
fn resolve_in_root(root: &Path, raw: &str) -> Option<PathBuf> {
    let joined = if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        root.join(raw)
    };
    let mut normal = PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::Prefix(prefix) => normal.push(prefix.as_os_str()),
            std::path::Component::RootDir => normal.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normal.pop();
            }
            std::path::Component::Normal(part) => normal.push(part),
        }
    }
    normal.starts_with(root).then_some(normal)
}

fn render(
    hit_path: &Path,
    start: u64,
    end: u64,
    breadcrumb: &str,
    score: &str,
    text: &str,
) -> String {
    let mut short = text;
    if short.len() > TEXT_CAP {
        let mut end = TEXT_CAP;
        while !short.is_char_boundary(end) {
            end -= 1;
        }
        short = &short[..end];
    }
    format!(
        "{}:{start}-{end} [{breadcrumb}] ({score})\n{short}",
        hit_path.display()
    )
}

fn provider() -> Result<&'static FastembedProvider, McpError> {
    static PROVIDER: OnceLock<Result<FastembedProvider, String>> = OnceLock::new();
    PROVIDER
        .get_or_init(|| FastembedProvider::load().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| McpError::internal_error(format!("embedding model unavailable: {e}"), None))
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchParams {
    /// Absolute workspace root.
    root: String,
    /// Natural-language query; also used for BM25 ranking.
    query: String,
    /// Exact anchors folded into lexical ranking.
    fts: Option<Vec<String>>,
    /// Fuse vector similarity with BM25 (default true).
    fuse: Option<bool>,
    /// Restrict hits to languages (`rust`, `python`, `nix`, `markdown`).
    /// Applied to every path, including the exact-`rg` shortcut.
    lang: Option<Vec<String>>,
    /// Restrict hits to ignore-style globs (whitelist, `!` negates).
    globs: Option<Vec<String>>,
    /// Max hits (default 10, max 50).
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct DefinitionParams {
    /// Absolute workspace root.
    root: String,
    /// File containing the reference, relative to root or absolute.
    path: String,
    /// 1-based line number of the reference.
    line: u32,
    /// 1-based character (Unicode scalar) of the reference on the line.
    character: u32,
    /// Server command override (default `rust-analyzer` from PATH).
    server: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct RgParams {
    /// Absolute workspace root.
    root: String,
    /// Pattern text (literal unless `regex`). Pass raw text without shell
    /// quotes; one outer `"..."` / `'...'` pair is ignored.
    pattern: String,
    /// Treat pattern as regex.
    regex: Option<bool>,
    /// Case-insensitive matching.
    case_insensitive: Option<bool>,
    /// Restrict to languages (`rust`, `python`, `nix`, `markdown`, or
    /// extensions like `rs`). Unknown values are rejected.
    lang: Option<Vec<String>>,
    /// Restrict to ignore-style globs (whitelist, `!` negates).
    globs: Option<Vec<String>>,
    /// Max hits (default 100, max 500).
    limit: Option<usize>,
}

#[derive(Clone)]
pub struct OneGrep {
    // Read by `#[tool_handler]` expansion; lint cannot see through the macro.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl OneGrep {
    /// Create the server.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Hybrid workspace search for intent and concepts, ranked with file:line cites. Standalone Rust paths (foo::Bar) and quoted literals (\"...\" / '...') use exact rg lookup unless fts anchors are supplied. Single identifier tokens go BM25, never hybrid. `lang`/`globs` filter hits on every path. Falls back to BM25 when no vector store exists, and to live rg when the workspace is not indexed."
    )]
    async fn search(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let limit = p.limit.unwrap_or(10).clamp(1, 50);
        let fts: &[String] = p.fts.as_deref().unwrap_or(&[]);
        let filter = HitFilter::build(&root, p.lang.clone(), p.globs.clone())?;
        match route::route(&p.query, fts) {
            route::Route::Literal(pattern) => {
                return self
                    .rg(Parameters(RgParams {
                        root: p.root.clone(),
                        pattern,
                        regex: Some(false),
                        case_insensitive: Some(false),
                        lang: p.lang.clone(),
                        globs: p.globs.clone(),
                        limit: Some(limit),
                    }))
                    .await;
            }
            route::Route::Identifier(term) => {
                // Exact beats fuzzy: single tokens go BM25, never hybrid.
                let indexed_now = index::is_indexed(&root);
                let lines: Vec<String> = if indexed_now {
                    lexical(&root, &term, limit, &filter)?
                } else {
                    rg_fallback_lines(&root, &term, limit, &filter)?
                };
                let body = lines.join("\n---\n");
                let mut notes = Vec::new();
                if indexed_now && index::is_stale(&root) {
                    notes.push(index::stale_note(&root));
                }
                let text = if indexed_now {
                    if notes.is_empty() {
                        body
                    } else {
                        format!("{}\n\n{body}", notes.join("\n"))
                    }
                } else {
                    format!("{}\n\n{body}", rg::unindexed_note(&root))
                };
                return Ok(CallToolResult::success(vec![ContentBlock::text(text)]));
            }
            route::Route::Intent(_) => {}
        }
        let mut query = p.query.clone();
        if let Some(fts) = &p.fts {
            query.push(' ');
            query.push_str(&fts.join(" "));
        }
        let indexed = index::is_indexed(&root);
        let fuse = p.fuse.unwrap_or(true);
        // Did retrieval actually consume vectors? Only then is their
        // staleness worth a note.
        let mut used_vectors = false;
        let lines: Vec<String> = if !indexed {
            rg_fallback_lines(&root, &query, limit, &filter)?
        } else if fuse {
            match provider() {
                Ok(provider) => {
                    used_vectors = true;
                    fuse::hybrid(&root, &query, limit, Some(provider), None)
                        .map_err(internal)?
                        .iter()
                        .filter(|h| filter.keep(&h.path))
                        .map(render_fused)
                        .collect()
                }
                Err(_) => lexical(&root, &query, limit, &filter)?,
            }
        } else {
            lexical(&root, &query, limit, &filter)?
        };
        let body = lines.join("\n---\n");
        let mut notes = Vec::new();
        if indexed && index::is_stale(&root) {
            notes.push(index::stale_note(&root));
        }
        if used_vectors && crate::vectors::is_stale(&root) {
            notes.push(crate::vectors::stale_note(&root));
        }
        let text = if indexed {
            if notes.is_empty() {
                body
            } else {
                format!("{}\n\n{body}", notes.join("\n"))
            }
        } else {
            format!("{}\n\n{body}", rg::unindexed_note(&root))
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
    }

    #[tool(
        description = "Intent search with Jev ranking inside the tool: retrieve a shortlist, score each candidate, return top-k only. Pool never enters context. Falls back to unranked retrieval if Jev is unavailable. Exact anchors still belong on `rg`."
    )]
    async fn search_ranked(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let limit = p.limit.unwrap_or(10).clamp(1, 50);
        let fts: &[String] = p.fts.as_deref().unwrap_or(&[]);
        let filter = HitFilter::build(&root, p.lang.clone(), p.globs.clone())?;
        let cls = route::route(&p.query, fts);
        if let route::Route::Literal(pattern) = &cls {
            return self
                .rg(Parameters(RgParams {
                    root: p.root.clone(),
                    pattern: pattern.clone(),
                    regex: Some(false),
                    case_insensitive: Some(false),
                    lang: p.lang.clone(),
                    globs: p.globs.clone(),
                    limit: Some(limit),
                }))
                .await;
        }
        // Single identifiers go lexical even when ranked: the meter rescores
        // a BM25 shortlist instead of paying for vectors.
        let lexical_only = matches!(cls, route::Route::Identifier(_));
        let mut query = p.query.clone();
        if let Some(fts) = &p.fts {
            query.push(' ');
            query.push_str(&fts.join(" "));
        }
        let fuse = p.fuse.unwrap_or(true);
        let indexed = index::is_indexed(&root);
        let query_for_rank = query.clone();
        let root_for_rank = root.clone();
        // Identifier queries stay lexical (router); only the fused path
        // consumes vectors, so only it can report them stale.
        let want_vectors = indexed && fuse && !lexical_only;
        let (status, hits) = tokio::task::spawn_blocking(move || {
            let mut hits = if !indexed || lexical_only {
                fuse::collect(&root_for_rank, &query_for_rank, limit, false, None)
            } else if fuse {
                match provider() {
                    Ok(p) => fuse::collect(&root_for_rank, &query_for_rank, limit, true, Some(p)),
                    Err(_) => fuse::collect(&root_for_rank, &query_for_rank, limit, false, None),
                }
            } else {
                fuse::collect(&root_for_rank, &query_for_rank, limit, false, None)
            }?;
            // Filter before the meter: Jev scores winners only, and
            // `limit` bounds what the caller ever sees.
            hits.retain(|h| filter.keep(&h.path));
            hits.truncate(limit);
            Ok::<_, crate::Error>(crate::jev::rerank_hits(&query_for_rank, hits))
        })
        .await
        .map_err(|e| internal(e.to_string()))?
        .map_err(internal)?;
        let body = hits
            .iter()
            .map(render_fused)
            .collect::<Vec<_>>()
            .join("\n---\n");
        let mut notes = vec![status.note.clone()];
        if indexed && index::is_stale(&root) {
            notes.push(index::stale_note(&root));
        }
        if want_vectors && crate::vectors::is_stale(&root) {
            notes.push(crate::vectors::stale_note(&root));
        }
        if !indexed {
            notes.push(rg::unindexed_note(&root));
        }
        let text = format!("{}\n\n{body}", notes.join("\n"));
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
    }

    #[tool(
        description = "Rust definition navigation via rust-analyzer: jump from a reference to its definition. Takes a 1-based line and character. Returns path:line evidence; targets outside the workspace are marked external. Needs saved files and a resolvable rust-analyzer."
    )]
    async fn definition(
        &self,
        Parameters(p): Parameters<DefinitionParams>,
    ) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let file = resolve_in_root(&root, &p.path).ok_or_else(|| {
            McpError::invalid_params(format!("path escapes workspace root: {}", p.path), None)
        })?;
        if p.line < 1 || p.character < 1 {
            return Err(McpError::invalid_params(
                "line and character are 1-based",
                None,
            ));
        }
        let options = lsp::Options {
            command: p.server.unwrap_or_else(|| lsp::DEFAULT_COMMAND.to_owned()),
            ..Default::default()
        };
        let targets = lsp::definition(&options, &root, &file, p.line, p.character)
            .await
            .map_err(internal)?;
        if targets.is_empty() {
            return Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "no definition found at {}:{}:{} (rust-analyzer)",
                file.display(),
                p.line,
                p.character
            ))]));
        }
        let lines: Vec<String> = targets
            .iter()
            .map(|t| {
                let scope = if t.outside_root {
                    "external"
                } else {
                    "workspace"
                };
                format!(
                    "{}:{}-{} ({scope})",
                    t.path.display(),
                    t.start_line,
                    t.end_line
                )
            })
            .collect();
        Ok(CallToolResult::success(vec![ContentBlock::text(
            lines.join("\n"),
        )]))
    }

    #[tool(
        description = "Exact text or regex search over workspace files (no index needed). Gitignore-aware. Returns path:line:text hits. Pass raw pattern text without shell quotes; a single outer \"...\" or '...' pair is stripped. Literal unless `regex` is true. `lang` restricts to languages (rust, python, nix, markdown); `globs` are include globs (`!` negates)."
    )]
    async fn rg(&self, Parameters(p): Parameters<RgParams>) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let options = rg::Options {
            regex: p.regex.unwrap_or(false),
            case_insensitive: p.case_insensitive.unwrap_or(false),
            globs: p.globs.unwrap_or_default(),
            langs: p.lang.unwrap_or_default(),
            limit: p.limit.unwrap_or(100).clamp(1, 500),
        };
        let lines: Vec<String> = rg::search(&root, &p.pattern, &options)
            .map_err(rg_error)?
            .iter()
            .map(|h| format!("{}:{}:{}", h.path.display(), h.line, h.text))
            .collect();
        Ok(CallToolResult::success(vec![ContentBlock::text(
            lines.join("\n"),
        )]))
    }
}

impl Default for OneGrep {
    fn default() -> Self {
        Self::new()
    }
}

fn lexical(
    root: &Path,
    query: &str,
    limit: usize,
    filter: &HitFilter,
) -> Result<Vec<String>, McpError> {
    Ok(index::search(root, query, limit)
        .map_err(internal)?
        .iter()
        .filter(|h| filter.keep(&h.path))
        .map(render_ranked)
        .collect())
}

fn rg_fallback_lines(
    root: &Path,
    query: &str,
    limit: usize,
    filter: &HitFilter,
) -> Result<Vec<String>, McpError> {
    Ok(rg::fallback_search(root, query, limit)
        .map_err(internal)?
        .iter()
        .filter(|h| filter.keep(&h.path))
        .map(render_live)
        .collect())
}

/// Post-retrieval `--lang` / `--glob` filter for `search` hits.
/// Validation fails fast (unknown langs, bad globs) as caller errors.
#[derive(Clone, Debug)]
struct HitFilter {
    langs: Vec<String>,
    globs: rg::GlobFilter,
}

impl HitFilter {
    fn build(
        root: &Path,
        lang: Option<Vec<String>>,
        globs: Option<Vec<String>>,
    ) -> Result<Self, McpError> {
        let langs = lang.unwrap_or_default();
        rg::validate_langs(&langs).map_err(rg_error)?;
        let globs = rg::GlobFilter::build(root, &globs.unwrap_or_default())
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
        Ok(Self { langs, globs })
    }

    fn keep(&self, path: &Path) -> bool {
        self.globs.keep(path) && rg::matches_lang(path, &self.langs).unwrap_or(false)
    }
}

fn render_ranked(h: &index::RankedHit) -> String {
    render(
        &h.path,
        h.start,
        h.end,
        &h.breadcrumb,
        &format!("{:.2}", h.score),
        &h.text,
    )
}

fn render_fused(h: &fuse::FusedHit) -> String {
    render(
        &h.path,
        h.start,
        h.end,
        &h.breadcrumb,
        &format!("{:.4}", h.score),
        &h.text,
    )
}

fn render_live(h: &rg::Hit) -> String {
    render(&h.path, h.line, h.line, "rg-fallback", "live", &h.text)
}

#[tool_handler]
impl rmcp::ServerHandler for OneGrep {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        info.instructions = Some(
            "Local-first hybrid workspace search. Prefer `search_ranked` for intent \
             (retrieve + Jev inside the tool; only top-k winners enter context). \
             Use `search` for the raw fused pool. Use `rg` for exact text, symbols, \
             or regex. Use `definition` to jump from a Rust reference to its \
             definition. Cite path:line evidence."
                .into(),
        );
        info
    }
}

/// Serve over stdio (local MCP clients).
///
/// # Errors
///
/// Returns [`crate::Error`] when transport setup fails.
pub async fn serve_stdio() -> Result<(), crate::Error> {
    OneGrep::new()
        .serve(stdio())
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))?
        .waiting()
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))?;
    Ok(())
}

/// Serve Streamable HTTP on 127.0.0.1 with bearer auth.
///
/// # Errors
///
/// Returns [`crate::Error`] when binding or serving fails.
pub async fn serve_http(port: u16, token: &str) -> Result<(), crate::Error> {
    let service = StreamableHttpService::new(
        || Ok(OneGrep::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    );
    let token = token.to_owned();
    let app = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn(
            move |req: axum::http::Request<axum::body::Body>, next: axum::middleware::Next| {
                let token = token.clone();
                async move {
                    let ok = req
                        .headers()
                        .get(axum::http::header::AUTHORIZATION)
                        .and_then(|v| v.to_str().ok())
                        == Some(format!("Bearer {token}").as_str());
                    if ok {
                        Ok::<_, (axum::http::StatusCode, String)>(next.run(req).await)
                    } else {
                        Err((
                            axum::http::StatusCode::UNAUTHORIZED,
                            "missing or invalid bearer token".to_owned(),
                        ))
                    }
                }
            },
        ));
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}"))
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))?;
    tracing::info!("serving MCP on 127.0.0.1:{port}/mcp");
    axum::serve(listener, app)
        .await
        .map_err(|e| crate::Error::InvalidInput(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_and_identifier_take_different_paths_unindexed() {
        // Branch-observable routing contract (no index, so no model load):
        // literals return bare rg hits; identifiers return rg-fallback
        // lines with the indexing note.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.txt"), "supersonic_ferret\n").expect("write");
        let root = workspace.path().to_string_lossy().into_owned();
        let body = |query: &str| {
            let result = rt
                .block_on(OneGrep::new().search(Parameters(SearchParams {
                    root: root.clone(),
                    query: query.into(),
                    fts: None,
                    fuse: Some(false),
                    lang: None,
                    globs: None,
                    limit: None,
                })))
                .expect("search");
            let text = serde_json::to_value(result).expect("response");
            text["content"][0]["text"].as_str().unwrap().to_owned()
        };
        let literal = body("\"supersonic_ferret\"");
        assert!(literal.contains("a.txt:1:supersonic_ferret"), "{literal}");
        assert!(!literal.contains("Indexing is available"), "{literal}");
        let identifier = body("supersonic_ferret");
        assert!(identifier.contains("supersonic_ferret"), "{identifier}");
        assert!(identifier.contains("Indexing is available"), "{identifier}");
    }

    #[tokio::test]
    async fn intent_and_mixed_queries_use_rg_fallback_when_unindexed() {
        let workspace = tempfile::tempdir().expect("workspace");
        for (query, fts) in [
            ("where is authentication handled", None),
            ("foo::Bar", Some(vec!["implementation".into()])),
            ("find definition of foo::Bar", None),
        ] {
            let result = OneGrep::new()
                .search(Parameters(SearchParams {
                    root: workspace.path().to_string_lossy().into_owned(),
                    query: query.into(),
                    fts,
                    fuse: Some(false),
                    lang: None,
                    globs: None,
                    limit: None,
                }))
                .await
                .expect("unindexed intent search falls back to live rg");
            let text = serde_json::to_value(result).expect("response");
            let body = text["content"][0]["text"].as_str().unwrap();
            assert!(body.contains("Indexing is available"), "{query}: {body}");
        }
    }

    #[tokio::test]
    async fn search_ranked_unindexed_emits_rank_header() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.txt"), "supersonic_ferret\n").expect("write");
        let result = OneGrep::new()
            .search_ranked(Parameters(SearchParams {
                root: workspace.path().to_string_lossy().into_owned(),
                query: "where is supersonic_ferret".into(),
                fts: None,
                fuse: Some(false),
                lang: None,
                globs: None,
                limit: Some(5),
            }))
            .await
            .expect("ranked search");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("rank:"), "{body}");
        assert!(body.contains("Indexing is available"), "{body}");
    }

    #[test]
    fn definition_paths_stay_inside_the_workspace() {
        let root = Path::new("/ws");
        assert_eq!(
            resolve_in_root(root, "src/main.rs"),
            Some(PathBuf::from("/ws/src/main.rs"))
        );
        assert_eq!(
            resolve_in_root(root, "/ws/src/main.rs"),
            Some(PathBuf::from("/ws/src/main.rs"))
        );
        assert_eq!(resolve_in_root(root, "../escape.rs"), None);
        assert_eq!(resolve_in_root(root, "a/../../escape.rs"), None);
        assert_eq!(resolve_in_root(root, "/etc/passwd"), None);
    }

    #[tokio::test]
    async fn definition_rejects_bad_locations_before_spawning() {
        let workspace = tempfile::tempdir().expect("workspace");
        let root = workspace.path().to_string_lossy().into_owned();
        let err = OneGrep::new()
            .definition(Parameters(DefinitionParams {
                root: root.clone(),
                path: "../escape.rs".into(),
                line: 1,
                character: 1,
                server: None,
            }))
            .await
            .expect_err("escape must fail");
        assert!(err.message.contains("escapes"), "{err:?}");
        let err = OneGrep::new()
            .definition(Parameters(DefinitionParams {
                root,
                path: "a.rs".into(),
                line: 0,
                character: 1,
                server: None,
            }))
            .await
            .expect_err("line 0 must fail");
        assert!(err.message.contains("1-based"), "{err:?}");
    }

    #[tokio::test]
    async fn definition_missing_server_is_an_error_not_empty() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("a.rs"), "fn a() {}\n").expect("fixture");
        let err = OneGrep::new()
            .definition(Parameters(DefinitionParams {
                root: workspace.path().to_string_lossy().into_owned(),
                path: "a.rs".into(),
                line: 1,
                character: 1,
                server: Some("one-grep-definitely-no-such-server".into()),
            }))
            .await
            .expect_err("missing server must fail");
        assert!(err.message.contains("could not start"), "{err:?}");
    }

    #[tokio::test]
    async fn exact_search_routes_to_rg_without_an_index() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("view.txt"),
            "foo::Bar\nx:Name=\"SaveButton\"\n{Binding User.Name}\n",
        )
        .expect("fixture");
        for (query, expected) in [
            ("foo::Bar", "view.txt:1:foo::Bar"),
            ("\"x:Name\"", "view.txt:2:x:Name"),
            ("\"{Binding User.Name}\"", "view.txt:3:{Binding User.Name}"),
        ] {
            let result = OneGrep::new()
                .search(Parameters(SearchParams {
                    root: workspace.path().to_string_lossy().into_owned(),
                    query: query.into(),
                    fts: None,
                    fuse: Some(false),
                    lang: None,
                    globs: None,
                    limit: None,
                }))
                .await
                .expect("exact search needs no index");
            let text = serde_json::to_value(result).expect("response");
            assert!(
                text["content"][0]["text"]
                    .as_str()
                    .unwrap()
                    .contains(expected)
            );
        }
    }

    /// Forall shape (Kani: arbitrary path string, assert no escape): every
    /// `..` that would leave `root` resolves to `None`. Sampled below.
    #[test]
    fn resolve_in_root_never_escapes() {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path();
        for raw in [
            "../escape.rs",
            "a/../../escape.rs",
            "/etc/passwd",
            "..",
            "sub/../../../x",
        ] {
            assert!(resolve_in_root(root, raw).is_none(), "{raw}");
        }
        for raw in ["a.rs", "sub/b.rs", "./a.rs"] {
            let resolved = resolve_in_root(root, raw).expect("inside");
            assert!(resolved.starts_with(root), "{raw}");
        }
    }

    #[test]
    fn rg_error_maps_caller_mistakes_to_invalid_params() {
        let err = rg_error(crate::Error::InvalidInput("unknown --lang `cobol`".into()));
        assert!(err.message.contains("cobol"));
        let err = rg_error(crate::Error::InvalidInput("x".into()));
        assert!(err.message.contains('x'));
    }

    #[tokio::test]
    async fn rg_lang_filter_reaches_the_handler() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("main.rs"), "needle\n").expect("fixture");
        std::fs::write(workspace.path().join("notes.txt"), "needle\n").expect("fixture");
        let result = OneGrep::new()
            .rg(Parameters(RgParams {
                root: workspace.path().to_string_lossy().into_owned(),
                pattern: "needle".into(),
                regex: None,
                case_insensitive: None,
                lang: Some(vec!["rust".into()]),
                globs: None,
                limit: None,
            }))
            .await
            .expect("rg");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("main.rs"), "{body}");
        assert!(!body.contains("notes.txt"), "{body}");
        // Unknown language fails closed as invalid params, not empty output.
        let err = OneGrep::new()
            .rg(Parameters(RgParams {
                root: workspace.path().to_string_lossy().into_owned(),
                pattern: "needle".into(),
                regex: None,
                case_insensitive: None,
                lang: Some(vec!["cobol".into()]),
                globs: None,
                limit: None,
            }))
            .await
            .expect_err("unknown lang must fail");
        assert!(err.message.contains("cobol"), "{err:?}");
    }

    /// Offline default (1.4): with no key anywhere, `search_ranked` makes
    /// no network attempt and preserves retrieval order. The sandbox
    /// redirects HOME to an empty dir so key files cannot leak in.
    #[tokio::test]
    async fn search_ranked_offline_preserves_retrieval_order() {
        let home = tempfile::tempdir().expect("home");
        let prev_home = std::env::var_os("HOME");
        let prev_key = std::env::var_os(crate::jev::ENV_API_KEY);
        // SAFETY: restored below; Jev callers fail closed to the same
        // fallback order on error, so concurrent tests keep their verdicts.
        unsafe {
            std::env::set_var("HOME", home.path());
            std::env::remove_var(crate::jev::ENV_API_KEY);
        }
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("a.txt"),
            "offline canary bravo\ncanary alpha\n",
        )
        .expect("fixture");
        std::fs::write(workspace.path().join("b.txt"), "canary charlie\n").expect("fixture");
        crate::index::sync(workspace.path()).expect("sync");
        let root = workspace.path().to_string_lossy().into_owned();
        let expected: Vec<String> =
            crate::fuse::collect(workspace.path(), "canary query terms here", 5, false, None)
                .expect("collect")
                .iter()
                .map(|h| h.path.to_string_lossy().into_owned())
                .collect();
        let result = OneGrep::new()
            .search_ranked(Parameters(SearchParams {
                root,
                query: "canary query terms here".into(),
                fts: None,
                fuse: Some(false),
                lang: None,
                globs: None,
                limit: Some(5),
            }))
            .await
            .expect("ranked");
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            if let Some(v) = prev_key {
                std::env::set_var(crate::jev::ENV_API_KEY, v);
            }
        }
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("rank: fallback"), "{body}");
        let mut cursor = 0;
        for path in &expected {
            let rel = cursor;
            let found = body[rel..].find(path.as_str()).map(|i| rel + i);
            let pos = found.unwrap_or_else(|| panic!("{path} missing in order: {body}"));
            assert!(pos >= cursor, "{path} out of order: {body}");
            cursor = pos + path.len();
        }
    }

    #[tokio::test]
    async fn search_lang_filter_applies_to_lexical_hits() {
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(
            workspace.path().join("main.rs"),
            "needle_haystack_unique here\n",
        )
        .expect("fixture");
        std::fs::write(
            workspace.path().join("notes.txt"),
            "needle_haystack_unique here\n",
        )
        .expect("fixture");
        crate::index::sync(workspace.path()).expect("sync");
        let root = workspace.path().to_string_lossy().into_owned();
        let result = OneGrep::new()
            .search(Parameters(SearchParams {
                root: root.clone(),
                query: "needle_haystack_unique".into(),
                fts: None,
                fuse: Some(false),
                lang: Some(vec!["rust".into()]),
                globs: None,
                limit: None,
            }))
            .await
            .expect("search");
        let text = serde_json::to_value(result).expect("response");
        let body = text["content"][0]["text"].as_str().unwrap();
        assert!(body.contains("main.rs"), "{body}");
        assert!(!body.contains("notes.txt"), "{body}");
        let err = OneGrep::new()
            .search(Parameters(SearchParams {
                root,
                query: "needle_haystack_unique".into(),
                fts: None,
                fuse: Some(false),
                lang: Some(vec!["cobol".into()]),
                globs: None,
                limit: None,
            }))
            .await
            .expect_err("unknown lang must fail fast");
        assert!(err.message.contains("cobol"), "{err:?}");
    }
}
