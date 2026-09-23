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

use crate::{embed::FastembedProvider, fuse, index, lsp, rg};
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

fn exact_pattern(query: &str) -> Option<&str> {
    let query = query.trim();
    for quote in ['"', '\''] {
        if let Some(literal) = query
            .strip_prefix(quote)
            .and_then(|s| s.strip_suffix(quote))
        {
            return (!literal.trim().is_empty() && !literal.contains(quote)).then_some(literal);
        }
    }
    let identifier = |part: &str| {
        let mut chars = part.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    (query.contains("::") && query.split("::").all(identifier)).then_some(query)
}

fn internal(e: impl std::fmt::Display) -> McpError {
    McpError::internal_error(e.to_string(), None)
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
        description = "Hybrid workspace search for intent and concepts, ranked with file:line cites. Standalone Rust paths (foo::Bar) and quoted literals (\"...\" / '...') use exact rg lookup unless fts anchors are supplied. Falls back to BM25 when no vector store exists, and to live rg when the workspace is not indexed."
    )]
    async fn search(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let limit = p.limit.unwrap_or(10).clamp(1, 50);
        if p.fts.as_ref().is_none_or(|anchors| anchors.is_empty()) {
            if let Some(pattern) = exact_pattern(&p.query) {
                return self
                    .rg(Parameters(RgParams {
                        root: p.root.clone(),
                        pattern: pattern.to_owned(),
                        regex: Some(false),
                        case_insensitive: Some(false),
                        limit: Some(limit),
                    }))
                    .await;
            }
        }
        let mut query = p.query.clone();
        if let Some(fts) = &p.fts {
            query.push(' ');
            query.push_str(&fts.join(" "));
        }
        let indexed = index::is_indexed(&root);
        let fuse = p.fuse.unwrap_or(true);
        let lines: Vec<String> = if !indexed {
            rg_fallback_lines(&root, &query, limit)?
        } else if fuse {
            match provider() {
                Ok(provider) => fuse::hybrid(&root, &query, limit, Some(provider), None)
                    .map_err(internal)?
                    .iter()
                    .map(|h| {
                        render(
                            &h.path,
                            h.start,
                            h.end,
                            &h.breadcrumb,
                            &format!("{:.4}", h.score),
                            &h.text,
                        )
                    })
                    .collect(),
                Err(_) => lexical(&root, &query, limit)?,
            }
        } else {
            lexical(&root, &query, limit)?
        };
        let body = lines.join("\n---\n");
        let text = if indexed {
            body
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
        if p.fts.as_ref().is_none_or(|anchors| anchors.is_empty()) {
            if let Some(pattern) = exact_pattern(&p.query) {
                return self
                    .rg(Parameters(RgParams {
                        root: p.root.clone(),
                        pattern: pattern.to_owned(),
                        regex: Some(false),
                        case_insensitive: Some(false),
                        limit: Some(limit),
                    }))
                    .await;
            }
        }
        let mut query = p.query.clone();
        if let Some(fts) = &p.fts {
            query.push(' ');
            query.push_str(&fts.join(" "));
        }
        let fuse = p.fuse.unwrap_or(true);
        let indexed = index::is_indexed(&root);
        let query_for_rank = query.clone();
        let root_for_rank = root.clone();
        let (status, hits) = tokio::task::spawn_blocking(move || {
            let hits = if !indexed {
                fuse::collect(&root_for_rank, &query_for_rank, limit, false, None)
            } else if fuse {
                match provider() {
                    Ok(p) => fuse::collect(&root_for_rank, &query_for_rank, limit, true, Some(p)),
                    Err(_) => fuse::collect(&root_for_rank, &query_for_rank, limit, false, None),
                }
            } else {
                fuse::collect(&root_for_rank, &query_for_rank, limit, false, None)
            }?;
            Ok::<_, crate::Error>(crate::jev::rerank_hits(&query_for_rank, hits))
        })
        .await
        .map_err(|e| internal(e.to_string()))?
        .map_err(internal)?;
        let body = hits
            .iter()
            .map(|h| {
                render(
                    &h.path,
                    h.start,
                    h.end,
                    &h.breadcrumb,
                    &format!("{:.4}", h.score),
                    &h.text,
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n");
        let mut text = format!("{}\n\n{body}", status.note);
        if !indexed {
            text = format!("{}\n{}\n\n{body}", status.note, rg::unindexed_note(&root));
        }
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
        description = "Exact text or regex search over workspace files (no index needed). Gitignore-aware. Returns path:line:text hits. Pass raw pattern text without shell quotes; a single outer \"...\" or '...' pair is stripped. Literal unless `regex` is true."
    )]
    async fn rg(&self, Parameters(p): Parameters<RgParams>) -> Result<CallToolResult, McpError> {
        let root = check_root(&p.root)?;
        let options = rg::Options {
            regex: p.regex.unwrap_or(false),
            case_insensitive: p.case_insensitive.unwrap_or(false),
            globs: Vec::new(),
            limit: p.limit.unwrap_or(100).clamp(1, 500),
        };
        let lines: Vec<String> = rg::search(&root, &p.pattern, &options)
            .map_err(internal)?
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

fn lexical(root: &Path, query: &str, limit: usize) -> Result<Vec<String>, McpError> {
    Ok(index::search(root, query, limit)
        .map_err(internal)?
        .iter()
        .map(|h| {
            render(
                &h.path,
                h.start,
                h.end,
                &h.breadcrumb,
                &format!("{:.2}", h.score),
                &h.text,
            )
        })
        .collect())
}

fn rg_fallback_lines(root: &Path, query: &str, limit: usize) -> Result<Vec<String>, McpError> {
    Ok(rg::fallback_search(root, query, limit)
        .map_err(internal)?
        .iter()
        .map(|h| render(&h.path, h.line, h.line, "rg-fallback", "live", &h.text))
        .collect())
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
    fn exact_routing_requires_an_unambiguous_anchor() {
        for query in [
            "foo::Bar",
            " foo::Bar ",
            "\"x:Name\"",
            "\"{Binding User.Name}\"",
            "'x:Name'",
            "'comment lies'",
        ] {
            assert!(exact_pattern(query).is_some(), "{query}");
        }
        for query in [
            "where is authentication handled",
            "find definition of foo::Bar",
            "foo::Bar example",
            "foo::",
            "::Bar",
            "foo:::Bar",
            "https://example.com",
            "\"\"",
            "\"   \"",
            "''",
            "\"foo\" OR \"bar\"",
            "'foo' OR 'bar'",
            "Bar",
        ] {
            assert_eq!(exact_pattern(query), None, "{query}");
        }
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
}
