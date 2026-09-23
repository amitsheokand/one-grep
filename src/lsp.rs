//! Rust definition navigation over LSP stdio (`PACKET-ideasearch.md`).
//!
//! Increment 1 covers `textDocument/definition` against `rust-analyzer`
//! only. One server is reused per workspace + command; requests carry
//! explicit timeouts; 1-based caller positions are translated to the
//! UTF-16 offsets LSP uses.
//!
//! Servers read saved files from disk; unsaved buffer contents are
//! invisible until written. Holding the pool lock across a request
//! serializes concurrent navigation; acceptable for this increment.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::OnceLock,
    time::Duration,
};

use lsp_types::{GotoDefinitionResponse, Location, LocationLink};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};
use url::Url;

use crate::Error;

/// Default server command (resolved via `PATH`).
pub const DEFAULT_COMMAND: &str = "rust-analyzer";
/// Cap on a single LSP frame body.
const MAX_FRAME: usize = 64 << 20;

/// Dial and request options.
#[derive(Debug, Clone)]
pub struct Options {
    /// Server executable (argv[0]); override for tests.
    pub command: String,
    /// Extra argv entries (e.g. server flags).
    pub args: Vec<String>,
    /// Budget for spawn + `initialize` handshake.
    pub init_timeout: Duration,
    /// Budget for one navigation request.
    pub request_timeout: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            command: DEFAULT_COMMAND.to_owned(),
            args: Vec::new(),
            init_timeout: Duration::from_secs(60),
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// One navigation hit. Lines are 1-based; characters stay LSP UTF-16
/// offsets and are only used for display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Target {
    /// Absolute file path.
    pub path: PathBuf,
    /// 1-based first line.
    pub start_line: u32,
    /// 1-based last line (inclusive).
    pub end_line: u32,
    /// Target escapes the workspace root.
    pub outside_root: bool,
}

fn lsp_error(msg: impl std::fmt::Display) -> Error {
    Error::Lsp(msg.to_string())
}

/// Count UTF-16 code units of the first `char_1 - 1` characters, clamped
/// to the line length (mirrors server-side clamping).
fn char_to_utf16_units(line_text: &str, char_1: u32) -> usize {
    line_text
        .chars()
        .take(char_1.saturating_sub(1) as usize)
        .map(|c| c.len_utf16())
        .sum()
}

fn file_uri(path: &Path) -> Result<Url, Error> {
    Url::from_file_path(path)
        .map_err(|()| lsp_error(format!("not a file path: {}", path.display())))
}

fn normalize_targets(
    root: &Path,
    response: Option<GotoDefinitionResponse>,
) -> Result<Vec<Target>, Error> {
    let mut targets = Vec::new();
    let mut push = |uri: &Url, start_line: u32, end_line: u32| -> Result<(), Error> {
        let path = uri
            .to_file_path()
            .map_err(|()| lsp_error(format!("server returned non-file URI: {uri}")))?;
        targets.push(Target {
            outside_root: !path.starts_with(root),
            path,
            start_line: start_line + 1,
            end_line: end_line + 1,
        });
        Ok(())
    };
    match response {
        None => Ok(targets),
        Some(GotoDefinitionResponse::Scalar(Location { uri, range })) => {
            push(&uri, range.start.line, range.end.line)?;
            Ok(targets)
        }
        Some(GotoDefinitionResponse::Array(locations)) => {
            for Location { uri, range } in &locations {
                push(uri, range.start.line, range.end.line)?;
            }
            Ok(targets)
        }
        Some(GotoDefinitionResponse::Link(links)) => {
            for LocationLink {
                target_uri,
                target_range,
                ..
            } in &links
            {
                push(target_uri, target_range.start.line, target_range.end.line)?;
            }
            Ok(targets)
        }
    }
}

/// JSON-RPC client over a piped stdio server.
struct Client {
    root: PathBuf,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    child: Child,
    next_id: i64,
}

impl Client {
    /// Spawn the server and run the `initialize` handshake. The stored
    /// root is canonicalized so `outside_root` checks survive symlinked
    /// parents (e.g. macOS `/var` → `/private/var`).
    async fn spawn(command: &str, args: &[String], root: &Path) -> Result<Self, Error> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .current_dir(root)
            .spawn()
            .map_err(|e| lsp_error(format!("could not start `{command}`: {e}")))?;
        let canonical = tokio::fs::canonicalize(root)
            .await
            .map_err(|e| lsp_error(format!("could not resolve {}: {e}", root.display())))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| lsp_error("server stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| lsp_error("server stdout unavailable"))?;
        let root_uri = file_uri(root)?;
        let mut client = Self {
            root: canonical,
            stdin,
            reader: BufReader::new(stdout),
            child,
            next_id: 1,
        };
        let params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri.as_str(),
            "capabilities": {
                "textDocument": {"definition": {"linkSupport": true}},
                "general": {"positionEncodings": ["utf-16"]},
            },
        });
        client.request("initialize", params).await?;
        client.notify("initialized", json!({})).await?;
        Ok(client)
    }

    async fn send(&mut self, body: &Value) -> Result<(), Error> {
        let text = serde_json::to_string(body).map_err(|e| lsp_error(e.to_string()))?;
        let frame = format!("Content-Length: {}\r\n\r\n{text}", text.len());
        self.stdin
            .write_all(frame.as_bytes())
            .await
            .map_err(|e| lsp_error(e.to_string()))
    }

    async fn recv(&mut self) -> Result<Value, Error> {
        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            self.reader
                .read_line(&mut line)
                .await
                .map_err(|e| lsp_error(e.to_string()))?;
            let line = line.trim();
            if line.is_empty() {
                break;
            }
            if let Some(value) = line.strip_prefix("Content-Length:") {
                content_length = value.trim().parse().ok();
            }
        }
        let len = content_length.ok_or_else(|| lsp_error("server frame without Content-Length"))?;
        if len > MAX_FRAME {
            return Err(lsp_error(format!("server frame too large: {len} bytes")));
        }
        let mut buf = vec![0u8; len];
        self.reader
            .read_exact(&mut buf)
            .await
            .map_err(|e| lsp_error(e.to_string()))?;
        serde_json::from_slice(&buf).map_err(|e| lsp_error(e.to_string()))
    }

    /// Send a request; answer server-to-client requests with `null` so a
    /// progress handshake never deadlocks the wait.
    async fn request(&mut self, method: &str, params: Value) -> Result<Value, Error> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await?;
        loop {
            let msg = self.recv().await?;
            if msg.get("id") == Some(&json!(id)) {
                if let Some(error) = msg.get("error") {
                    return Err(Error::LspServer(format!("{method}: {error}")));
                }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
            if msg.get("id").is_some() && msg.get("method").is_some() {
                self.send(&json!({"jsonrpc": "2.0", "id": msg["id"], "result": null}))
                    .await?;
            }
        }
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<(), Error> {
        self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
            .await
    }

    /// `textDocument/definition` at a 1-based line / 1-based character
    /// (Unicode scalar) position.
    async fn definition(
        &mut self,
        file: &Path,
        line_1: u32,
        char_1: u32,
    ) -> Result<Vec<Target>, Error> {
        let line_0 = line_1
            .checked_sub(1)
            .ok_or_else(|| lsp_error("line must be >= 1"))?;
        if char_1 < 1 {
            return Err(lsp_error("character must be >= 1"));
        }
        let text = tokio::fs::read_to_string(file)
            .await
            .map_err(|e| lsp_error(format!("could not read {}: {e}", file.display())))?;
        let line_text = text.lines().nth(line_0 as usize).ok_or_else(|| {
            lsp_error(format!("line {line_1} out of range in {}", file.display()))
        })?;
        let params = json!({
            "textDocument": {"uri": file_uri(file)?.as_str()},
            "position": {"line": line_0, "character": char_to_utf16_units(line_text, char_1)},
        });
        let result: Option<GotoDefinitionResponse> =
            serde_json::from_value(self.request("textDocument/definition", params).await?)
                .map_err(|e| lsp_error(e.to_string()))?;
        normalize_targets(&self.root, result)
    }

    async fn shutdown(mut self) {
        let _ = self.request("shutdown", Value::Null).await;
        let _ = self.notify("exit", Value::Null).await;
        let _ = self.child.kill().await;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ServerKey {
    command: String,
    args: Vec<String>,
    root: PathBuf,
}

static POOL: OnceLock<Mutex<HashMap<ServerKey, Client>>> = OnceLock::new();

fn pool() -> &'static Mutex<HashMap<ServerKey, Client>> {
    POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drop a dead server and shut it down in the background.
fn evict(guard: &mut HashMap<ServerKey, Client>, key: &ServerKey) {
    if let Some(client) = guard.remove(key) {
        tokio::spawn(async move { client.shutdown().await });
    }
}

/// Definition lookup through the per-workspace server pool. Spawns the
/// server on first use; drops a dead server so the next call respawns it.
pub async fn definition(
    options: &Options,
    root: &Path,
    file: &Path,
    line_1: u32,
    char_1: u32,
) -> Result<Vec<Target>, Error> {
    let canonical = tokio::fs::canonicalize(root)
        .await
        .map_err(|e| lsp_error(format!("could not resolve {}: {e}", root.display())))?;
    let key = ServerKey {
        command: options.command.clone(),
        args: options.args.clone(),
        root: canonical,
    };
    // Servers compare URIs against canonical VFS paths; resolve symlinks
    // (e.g. macOS `/var` → `/private/var`) before handshake and lookup.
    let file = tokio::fs::canonicalize(file)
        .await
        .map_err(|e| lsp_error(format!("could not resolve {}: {e}", file.display())))?;
    let mut guard = pool().lock().await;
    if !guard.contains_key(&key) {
        let spawn = Client::spawn(&options.command, &options.args, &key.root);
        match timeout(options.init_timeout, spawn).await {
            Ok(Ok(client)) => {
                guard.insert(key.clone(), client);
            }
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(lsp_error(format!(
                    "server `{}` initialize timed out after {:?}",
                    options.command, options.init_timeout
                )));
            }
        }
    }
    let client = guard.get_mut(&key).expect("pooled server");
    let result = timeout(
        options.request_timeout,
        client.definition(&file, line_1, char_1),
    )
    .await;
    match result {
        Ok(Ok(targets)) => Ok(targets),
        // A live server answering with an error keeps its pool slot; only
        // transport failures and timeouts evict (with background shutdown
        // so evicted servers never leak).
        Ok(Err(e @ Error::LspServer(_))) => Err(e),
        Ok(Err(e)) => {
            evict(&mut guard, &key);
            Err(e)
        }
        Err(_) => {
            evict(&mut guard, &key);
            Err(lsp_error(format!(
                "definition request timed out after {:?}",
                options.request_timeout
            )))
        }
    }
}

/// Shut down every pooled server. Tests must call this; the long-lived
/// MCP process keeps servers until exit (idle eviction is a follow-up).
pub async fn shutdown_all() {
    let mut guard = pool().lock().await;
    let servers: Vec<ServerKey> = guard.keys().cloned().collect();
    for key in servers {
        if let Some(client) = guard.remove(&key) {
            client.shutdown().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_translation_counts_astral_code_units() {
        // 'a' = 1 unit, 'é' = 1 unit, '𐀀' (U+10000) = 2 units, 'z' = 1 unit.
        let line = "aé𐀀z";
        assert_eq!(char_to_utf16_units(line, 1), 0);
        assert_eq!(char_to_utf16_units(line, 2), 1);
        assert_eq!(char_to_utf16_units(line, 3), 2);
        assert_eq!(char_to_utf16_units(line, 4), 4);
        assert_eq!(char_to_utf16_units(line, 5), 5);
        // Past end-of-line clamps instead of erroring.
        assert_eq!(char_to_utf16_units(line, 99), 5);
    }

    #[test]
    fn file_uri_round_trip() {
        let path = Path::new("/tmp/one-grep-lsp/main.rs");
        let uri = file_uri(path).expect("uri");
        assert_eq!(uri.to_file_path().expect("path"), path);
    }

    #[test]
    fn location_link_normalizes_to_target_range() {
        let response = serde_json::from_value::<Option<GotoDefinitionResponse>>(json!([
            {
                "originSelectionRange": {
                    "start": {"line": 5, "character": 12},
                    "end": {"line": 5, "character": 18},
                },
                "targetUri": "file:///tmp/one-grep-lsp/main.rs",
                "targetRange": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 2, "character": 1},
                },
                "targetSelectionRange": {
                    "start": {"line": 0, "character": 3},
                    "end": {"line": 0, "character": 9},
                },
            }
        ]))
        .expect("parse");
        let targets =
            normalize_targets(Path::new("/tmp/one-grep-lsp"), response).expect("normalize");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].path, Path::new("/tmp/one-grep-lsp/main.rs"));
        assert_eq!((targets[0].start_line, targets[0].end_line), (1, 3));
        assert!(!targets[0].outside_root);
    }

    #[test]
    fn targets_outside_root_are_marked() {
        let response = serde_json::from_value::<Option<GotoDefinitionResponse>>(json!({
            "uri": "file:///usr/local/rustlib/src/lib.rs",
            "range": {
                "start": {"line": 10, "character": 0},
                "end": {"line": 10, "character": 5},
            },
        }))
        .expect("parse");
        let targets = normalize_targets(Path::new("/tmp/ws"), response).expect("normalize");
        assert!(targets[0].outside_root);
    }

    #[tokio::test]
    async fn missing_server_errors_distinctly() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").expect("fixture");
        let options = Options {
            command: "one-grep-definitely-no-such-server".to_owned(),
            init_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(5),
            ..Default::default()
        };
        let err = definition(&options, dir.path(), &dir.path().join("a.rs"), 1, 1)
            .await
            .expect_err("missing server must fail");
        let msg = err.to_string();
        assert!(msg.contains("could not start"), "{msg}");
    }

    #[tokio::test]
    async fn silent_server_times_out() {
        // `sleep` holds stdio open without answering, so the handshake
        // never completes. (An echoing stand-in like `cat` would fake it.)
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").expect("fixture");
        let options = Options {
            command: "sleep".to_owned(),
            args: vec!["5".to_owned()],
            init_timeout: Duration::from_millis(500),
            request_timeout: Duration::from_millis(500),
        };
        let err = definition(&options, dir.path(), &dir.path().join("a.rs"), 1, 1)
            .await
            .expect_err("silent server must time out");
        assert!(err.to_string().contains("timed out"), "{err}");
        // No shutdown_all here: the timed-out spawn never entered the pool,
        // and shutting down would kill other tests' pooled servers.
    }

    /// Locate a usable rust-analyzer: `PATH`, then `rustup which`.
    fn discover_server() -> Option<String> {
        if std::process::Command::new("rust-analyzer")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Some("rust-analyzer".to_owned());
        }
        let output = std::process::Command::new("rustup")
            .args(["which", "rust-analyzer"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let path = String::from_utf8(output.stdout).ok()?;
        let path = path.trim().to_owned();
        (!path.is_empty()).then_some(path)
    }

    #[tokio::test]
    async fn rust_analyzer_resolves_same_file_definition() {
        let Some(server) = discover_server() else {
            eprintln!("skipping: no rust-analyzer on PATH or via rustup");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"navtest\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
        )
        .expect("manifest");
        std::fs::write(
            dir.path().join("src/main.rs"),
            "fn target() -> u32 {\n    42\n}\n\nfn main() {\n    let _ = target();\n}\n",
        )
        .expect("source");
        let options = Options {
            command: server,
            init_timeout: Duration::from_secs(120),
            request_timeout: Duration::from_secs(120),
            ..Default::default()
        };
        let file = dir.path().join("src/main.rs");
        // Workspace loading runs in the background; the server answers
        // "file not found" until the cargo project is loaded.
        let start = std::time::Instant::now();
        let targets = loop {
            // A warming server answers "file not found" or empty; this
            // fixture has a definition, so both mean "not yet loaded".
            // Any other error is genuine and fails fast.
            match definition(&options, dir.path(), &file, 6, 13).await {
                Ok(targets) if !targets.is_empty() => break targets,
                Ok(_) => {
                    assert!(
                        start.elapsed() < Duration::from_secs(90),
                        "server never resolved definition in {file:?}"
                    );
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Err(e)
                    if e.to_string().contains("file not found")
                        || e.to_string().contains("content modified") =>
                {
                    assert!(
                        start.elapsed() < Duration::from_secs(90),
                        "server never loaded {file:?}: {e}"
                    );
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Err(e) => panic!("definition: {e}"),
            }
        };
        shutdown_all().await;
        assert!(!targets.is_empty(), "expected at least one definition");
        let canonical = std::fs::canonicalize(&file).expect("canonical");
        let hit = targets
            .iter()
            .find(|t| t.path == canonical)
            .expect("same-file target");
        assert_eq!(hit.start_line, 1);
        assert!(!hit.outside_root);
    }
}
