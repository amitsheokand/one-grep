//! Optional ast-grep bridge: structural patterns as a third RRF list.
//!
//! Never a hard dependency. The `ast-grep` binary is probed on `PATH`
//! (overridable with `ONE_GREP_AST_GREP` for tests); when absent, callers
//! keep their current results untouched. When present, `search` runs
//! `ast-grep run --pattern … --lang … --json` over the workspace and
//! returns `(path, 1-based line, text)` hits in ast-grep's order.
//!
//! Verified against ast-grep 0.45.1: `--json` emits an array of
//! `{file, range: {start: {line (0-based)}}, text}`; exit 0/1 with `[]`
//! both mean no matches.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::Serialize;

use crate::Error;

/// Binary name probed on `PATH`.
pub const BINARY: &str = "ast-grep";
/// Env override for the binary path (tests point it at a fake script).
pub const ENV_BIN: &str = "ONE_GREP_AST_GREP";

/// One structural hit: file, 1-based line, matched text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AstHit {
    pub path: PathBuf,
    pub line: u64,
    pub text: String,
}

#[derive(Debug, Deserialize)]
struct AstJson {
    #[serde(default)]
    file: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    range: AstRange,
}

#[derive(Debug, Default, Deserialize)]
struct AstRange {
    #[serde(default)]
    start: AstPoint,
}

#[derive(Debug, Default, Deserialize)]
struct AstPoint {
    #[serde(default)]
    line: u64,
}

fn binary() -> String {
    std::env::var(ENV_BIN)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| BINARY.to_owned())
}

/// Whether the bridge can run here. No spawning: `PATH` lookup only.
#[must_use]
pub fn available() -> bool {
    if let Ok(custom) = std::env::var(ENV_BIN) {
        return !custom.trim().is_empty() && Path::new(custom.trim()).exists();
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths)
            .any(|dir| dir.join(BINARY).is_file() || dir.join(format!("{BINARY}.exe")).is_file())
    })
}

/// Map our `--lang` vocabulary to ast-grep language names. Verified live:
/// `rust`, `python`, `typescript`, `tsx`, `go`, `java`, `nix`,
/// `markdown` are all accepted; extensions fold to their canonical name.
fn ast_lang(lang: &str) -> Option<&'static str> {
    match lang.trim().to_ascii_lowercase().as_str() {
        "rust" | "rs" => Some("rust"),
        "python" | "py" => Some("python"),
        "typescript" | "ts" | "mts" | "cts" => Some("typescript"),
        "tsx" | "jsx" | "js" | "mjs" | "cjs" => Some("tsx"),
        "go" => Some("go"),
        "java" => Some("java"),
        "nix" => Some("nix"),
        "markdown" | "md" => Some("markdown"),
        _ => None,
    }
}

/// Run a structural pattern over `workspace`, returning at most `limit`
/// hits in ast-grep's order.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for unknown languages, unparseable
/// output, or ast-grep failures (stderr tail included), and [`Error::Io`]
/// when the binary cannot spawn.
pub fn search(
    workspace: &Path,
    pattern: &str,
    lang: &str,
    limit: usize,
) -> Result<Vec<AstHit>, Error> {
    if !workspace.is_dir() {
        return Err(Error::InvalidInput(format!(
            "workspace is not a directory: {}",
            workspace.display()
        )));
    }
    let Some(ast_lang) = ast_lang(lang) else {
        return Err(Error::InvalidInput(format!(
            "unknown --lang `{lang}` for structural search (supported: {})",
            crate::rg::SUPPORTED_LANGS.join(", ")
        )));
    };
    let output = std::process::Command::new(binary())
        .arg("run")
        .arg("--pattern")
        .arg(pattern)
        .arg("--lang")
        .arg(ast_lang)
        .arg("--json")
        .arg("--")
        .arg(workspace)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    match serde_json::from_str::<Vec<AstJson>>(&stdout) {
        Ok(items) => Ok(items
            .into_iter()
            .take(limit.max(1))
            .map(|item| AstHit {
                path: PathBuf::from(item.file),
                line: item.range.start.line + 1,
                text: item.text,
            })
            .collect()),
        Err(_) => {
            let mut detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if detail.len() > 300 {
                detail.truncate(300);
            }
            Err(Error::InvalidInput(format!(
                "ast-grep failed (status {}): {}",
                output.status,
                if detail.is_empty() {
                    "unparseable output"
                } else {
                    &detail
                }
            )))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Serializes the tests that point `ENV_BIN` at fake scripts: they
    /// share one process-global variable. Also used by MCP tests.
    static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    pub(crate) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .expect("env lock")
    }

    fn fake_ast_grep(dir: &Path, stdout: &str, status: i32) {
        // Fake `ast-grep`: ignores args, prints canned JSON, exits `status`.
        let script = dir.join("ast-grep");
        std::fs::write(
            &script,
            format!("#!/bin/sh\nprintf '%s' '{stdout}'\nexit {status}\n",),
        )
        .expect("script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }
        // SAFETY: restored by the caller; no other test reads this var.
        unsafe {
            std::env::set_var(ENV_BIN, script.to_string_lossy().into_owned());
        }
    }

    fn unpoint() {
        // SAFETY: test-only; restores the ambient state.
        unsafe {
            std::env::remove_var(ENV_BIN);
        }
    }

    #[test]
    fn parses_json_hits_with_one_based_lines() {
        let _guard = env_lock();
        let bin = tempfile::tempdir().expect("tempdir");
        let ws = tempfile::tempdir().expect("ws");
        fake_ast_grep(
            bin.path(),
            r#"[{"file": "x.ts", "range": {"start": {"line": 4}}, "text": "hit"}]"#,
            0,
        );
        let hits = search(ws.path(), "p", "typescript", 10).expect("search");
        unpoint();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 5);
        assert_eq!(hits[0].text, "hit");
    }

    #[test]
    fn empty_array_is_no_match_not_error() {
        let _guard = env_lock();
        let bin = tempfile::tempdir().expect("tempdir");
        let ws = tempfile::tempdir().expect("tempdir");
        fake_ast_grep(bin.path(), "[]", 1);
        let hits = search(ws.path(), "zzz", "rust", 10).expect("search");
        unpoint();
        assert!(hits.is_empty());
    }

    #[test]
    fn unparseable_output_is_caller_error() {
        let _guard = env_lock();
        let bin = tempfile::tempdir().expect("tempdir");
        let ws = tempfile::tempdir().expect("tempdir");
        fake_ast_grep(bin.path(), "not json", 2);
        let err = search(ws.path(), "p", "rust", 10).expect_err("must fail");
        unpoint();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn unknown_lang_fails_before_spawning() {
        let ws = tempfile::tempdir().expect("tempdir");
        let err = search(ws.path(), "p", "cobol", 10).expect_err("must fail");
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn absent_binary_reports_unavailable() {
        let _guard = env_lock();
        // SAFETY: restored immediately; see above.
        unsafe {
            std::env::set_var(ENV_BIN, "/no/such/ast-grep-binary");
        }
        assert!(!available());
        unpoint();
    }
}
