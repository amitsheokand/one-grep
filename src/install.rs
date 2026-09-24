//! Agent installs: bearer token + harness MCP registration.

use std::path::{Path, PathBuf};

use rand::RngExt as _;

use crate::Error;

/// Default HTTP port for `serve`.
pub const DEFAULT_PORT: u16 = 3210;

fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from)
}

fn global_dir() -> PathBuf {
    home_dir().join(".one-grep")
}

/// Prefer `~/.local/bin/one-grep` when present; otherwise the running binary.
fn resolve_binary() -> Result<String, Error> {
    let local = home_dir().join(".local").join("bin").join("one-grep");
    if local.exists() {
        return Ok(local.to_string_lossy().into_owned());
    }
    Ok(std::env::current_exe()?.to_string_lossy().into_owned())
}

/// Read the bearer token, generating and storing one (mode 0600) on first use.
///
/// # Errors
///
/// Returns [`Error`] when the token cannot be read or written.
pub fn ensure_token() -> Result<String, Error> {
    ensure_token_in(&global_dir())
}

fn ensure_token_in(dir: &std::path::Path) -> Result<String, Error> {
    let path = dir.join("token");
    if path.exists() {
        return Ok(std::fs::read_to_string(&path)?.trim().to_owned());
    }
    let token: String = rand::rng()
        .sample_iter(&rand::distr::Alphanumeric)
        .take(48)
        .map(char::from)
        .collect();
    std::fs::create_dir_all(dir)?;
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(dir.join("token"))?;
        use std::io::Write as _;
        file.write_all(token.as_bytes())?;
    }
    Ok(token)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Opencode,
    Cursor,
    Pi,
    Muse,
    Hermes,
    CommandCode,
}

fn parse_target(raw: &str) -> Result<Target, Error> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "opencode" => Ok(Target::Opencode),
        "cursor" => Ok(Target::Cursor),
        "pi" => Ok(Target::Pi),
        "muse" => Ok(Target::Muse),
        "hermes" => Ok(Target::Hermes),
        "command-code" | "commandcode" | "command_code" => Ok(Target::CommandCode),
        other => Err(Error::InvalidInput(format!(
            "unknown target: {other} (supported: opencode, cursor, pi, muse, hermes, command-code)"
        ))),
    }
}

fn read_json_object(path: &Path) -> Result<serde_json::Value, Error> {
    if path.exists() {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if value.is_object() {
            Ok(value)
        } else {
            Err(Error::InvalidInput(format!(
                "{} is not a JSON object",
                path.display()
            )))
        }
    } else {
        Ok(serde_json::json!({}))
    }
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn upsert_mcp_map(
    config: &mut serde_json::Value,
    map_key: &str,
    entry: serde_json::Value,
    path_label: &str,
) -> Result<(), Error> {
    let root = config
        .as_object_mut()
        .ok_or_else(|| Error::InvalidInput(format!("{path_label} is not an object")))?;
    let servers = root
        .entry(map_key.to_owned())
        .or_insert_with(|| serde_json::json!({}));
    servers
        .as_object_mut()
        .ok_or_else(|| Error::InvalidInput(format!("{path_label} {map_key} is not an object")))?
        .insert("one-grep".to_owned(), entry);
    Ok(())
}

fn stdio_command_args(exe: &str) -> (String, Vec<&'static str>) {
    (exe.to_owned(), vec!["serve", "--stdio"])
}

fn cursor_like_entry(exe: &str) -> serde_json::Value {
    let (command, args) = stdio_command_args(exe);
    serde_json::json!({ "command": command, "args": args })
}

fn command_code_entry(exe: &str) -> serde_json::Value {
    let (command, args) = stdio_command_args(exe);
    serde_json::json!({
        "transport": "stdio",
        "enabled": true,
        "command": command,
        "args": args,
    })
}

fn muse_legacy_entry(exe: &str) -> serde_json::Value {
    let (command, args) = stdio_command_args(exe);
    serde_json::json!({
        "command": command,
        "args": args,
        "enabled": true,
        "mode": "optional",
    })
}

fn opencode_config_path() -> PathBuf {
    home_dir()
        .join(".config")
        .join("opencode")
        .join("opencode.json")
}

/// Register one-grep in OpenCode's `mcp` section.
///
/// `http` selects the loopback+bearer entry; otherwise a stdio entry using
/// the resolved binary is written. Existing keys are preserved.
///
/// # Errors
///
/// Returns [`Error`] when the config cannot be read or written.
pub fn install_opencode(http: bool, port: u16) -> Result<PathBuf, Error> {
    let path = opencode_config_path();
    let mut config = read_json_object(&path)?;
    let exe = resolve_binary()?;
    let entry = if http {
        serde_json::json!({
            "type": "remote",
            "url": format!("http://127.0.0.1:{port}/mcp"),
            "headers": { "Authorization": format!("Bearer {}", ensure_token()?) },
            "enabled": true,
        })
    } else {
        serde_json::json!({
            "type": "local",
            "command": [exe, "serve", "--stdio"],
            "enabled": true,
        })
    };
    upsert_mcp_map(&mut config, "mcp", entry, "opencode.json")?;
    write_json(&path, &config)?;
    Ok(path)
}

fn install_cursor() -> Result<PathBuf, Error> {
    let path = home_dir().join(".cursor").join("mcp.json");
    let mut config = read_json_object(&path)?;
    let exe = resolve_binary()?;
    upsert_mcp_map(
        &mut config,
        "mcpServers",
        cursor_like_entry(&exe),
        "cursor mcp.json",
    )?;
    write_json(&path, &config)?;
    Ok(path)
}

fn install_pi() -> Result<PathBuf, Error> {
    let path = home_dir().join(".pi").join("agent").join("mcp.json");
    let mut config = read_json_object(&path)?;
    let exe = resolve_binary()?;
    upsert_mcp_map(
        &mut config,
        "mcpServers",
        cursor_like_entry(&exe),
        "pi mcp.json",
    )?;
    write_json(&path, &config)?;
    Ok(path)
}

fn install_muse() -> Result<PathBuf, Error> {
    let path = home_dir()
        .join(".config")
        .join("muse")
        .join("settings.json");
    let mut config = read_json_object(&path)?;
    let exe = resolve_binary()?;
    upsert_mcp_map(
        &mut config,
        "mcpServers",
        cursor_like_entry(&exe),
        "muse settings.json",
    )?;
    upsert_mcp_map(
        &mut config,
        "mcp_servers",
        muse_legacy_entry(&exe),
        "muse settings.json",
    )?;
    write_json(&path, &config)?;
    Ok(path)
}

fn install_command_code() -> Result<PathBuf, Error> {
    let path = home_dir().join(".commandcode").join("mcp.json");
    let mut config = read_json_object(&path)?;
    let exe = resolve_binary()?;
    upsert_mcp_map(
        &mut config,
        "mcpServers",
        command_code_entry(&exe),
        "commandcode mcp.json",
    )?;
    write_json(&path, &config)?;
    Ok(path)
}

/// Upsert `one-grep` under `mcp_servers` without round-tripping the whole YAML
/// document (preserves anchors/comments elsewhere).
fn upsert_hermes_one_grep(content: &str, exe: &str) -> String {
    let block =
        format!("  one-grep:\n    command: {exe}\n    args:\n      - serve\n      - --stdio\n");
    let lines: Vec<&str> = content.lines().collect();
    let Some(mcp_idx) = lines.iter().position(|l| l.trim_end() == "mcp_servers:") else {
        let mut out = content.trim_end().to_owned();
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("mcp_servers:\n");
        out.push_str(&block);
        return out;
    };

    let mut end_section = lines.len();
    for (i, line) in lines.iter().enumerate().skip(mcp_idx + 1) {
        if line.is_empty() {
            continue;
        }
        // Next top-level key ends the section.
        if !line.starts_with(' ') && !line.starts_with('\t') && line.contains(':') {
            end_section = i;
            break;
        }
    }

    let mut one_start = None;
    let mut one_end = None;
    for (i, line) in lines.iter().enumerate().take(end_section).skip(mcp_idx + 1) {
        if *line == "  one-grep:" || line.starts_with("  one-grep:") {
            one_start = Some(i);
            let mut j = i + 1;
            while j < end_section {
                let next = lines[j];
                if next.starts_with("  ")
                    && !next.starts_with("   ")
                    && !next.starts_with("  \t")
                    && next.contains(':')
                    && !next.starts_with("  -")
                {
                    break;
                }
                if next.is_empty() {
                    // keep blank lines inside block until a peer key
                    let mut k = j + 1;
                    while k < end_section && lines[k].is_empty() {
                        k += 1;
                    }
                    if k < end_section {
                        let peek = lines[k];
                        if peek.starts_with("  ")
                            && !peek.starts_with("   ")
                            && peek.contains(':')
                            && !peek.starts_with("  -")
                        {
                            break;
                        }
                    }
                }
                j += 1;
            }
            one_end = Some(j);
            break;
        }
    }

    let mut out: Vec<String> = Vec::with_capacity(lines.len() + 6);
    if let (Some(start), Some(end)) = (one_start, one_end) {
        out.extend(lines[..start].iter().map(|s| (*s).to_owned()));
        out.extend(block.lines().map(str::to_owned));
        out.extend(lines[end..].iter().map(|s| (*s).to_owned()));
    } else {
        out.extend(lines[..=mcp_idx].iter().map(|s| (*s).to_owned()));
        out.extend(block.lines().map(str::to_owned));
        out.extend(lines[mcp_idx + 1..].iter().map(|s| (*s).to_owned()));
    }
    let mut text = out.join("\n");
    if content.ends_with('\n') && !text.ends_with('\n') {
        text.push('\n');
    }
    text
}

fn install_hermes() -> Result<PathBuf, Error> {
    let path = home_dir().join(".hermes").join("config.yaml");
    let exe = resolve_binary()?;
    let content = if path.exists() {
        std::fs::read_to_string(&path)?
    } else {
        String::new()
    };
    let updated = upsert_hermes_one_grep(&content, &exe);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, updated)?;
    Ok(path)
}

/// Register `one-grep` with a supported agent harness.
///
/// Targets: `opencode`, `cursor`, `pi`, `muse`, `hermes`, `command-code`
/// (aliases: `commandcode`, `command_code`). `--http` is only valid for
/// `opencode`.
///
/// # Errors
///
/// Returns [`Error`] for unknown targets, unsupported `--http`, or I/O/JSON.
pub fn install(target: &str, http: bool, port: u16) -> Result<PathBuf, Error> {
    let target = parse_target(target)?;
    if http && target != Target::Opencode {
        return Err(Error::InvalidInput(
            "--http is only supported for --target opencode".to_owned(),
        ));
    }
    match target {
        Target::Opencode => install_opencode(http, port),
        Target::Cursor => install_cursor(),
        Target::Pi => install_pi(),
        Target::Muse => install_muse(),
        Target::Hermes => install_hermes(),
        Target::CommandCode => install_command_code(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_stable_and_sized() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = ensure_token_in(dir.path()).expect("generate");
        let again = ensure_token_in(dir.path()).expect("read");
        assert_eq!(first, again);
        assert_eq!(first.len(), 48);
    }

    #[test]
    fn parse_target_aliases() {
        assert_eq!(parse_target("command-code").unwrap(), Target::CommandCode);
        assert_eq!(parse_target("commandcode").unwrap(), Target::CommandCode);
        assert_eq!(parse_target("Cursor").unwrap(), Target::Cursor);
        assert!(parse_target("unknown").is_err());
    }

    #[test]
    fn hermes_upsert_inserts_and_replaces() {
        let base = "model:\n  default: x\nmcp_servers:\n  zvec_grep:\n    url: http://127.0.0.1:7999/mcp\n";
        let once = upsert_hermes_one_grep(base, "/bin/one-grep");
        assert!(once.contains("  one-grep:\n    command: /bin/one-grep\n"));
        assert!(once.contains("  zvec_grep:\n    url: http://127.0.0.1:7999/mcp\n"));
        let twice = upsert_hermes_one_grep(&once, "/opt/one-grep");
        assert_eq!(twice.matches("  one-grep:").count(), 1);
        assert!(twice.contains("command: /opt/one-grep"));
        assert!(!twice.contains("command: /bin/one-grep"));
    }

    #[test]
    fn hermes_upsert_creates_section() {
        let out = upsert_hermes_one_grep("model:\n  default: x\n", "/bin/one-grep");
        assert!(out.contains("mcp_servers:\n  one-grep:\n    command: /bin/one-grep\n"));
    }
}
