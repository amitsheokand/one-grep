//! TypeSafe Jev ranking over a closed retrieval shortlist.
//!
//! one-grep retrieves; Jev scores each candidate with a Noul and an
//! `exists` Noul (does any candidate answer). The pool never needs to
//! enter the agent window — callers return top-k only.
//!
//! Key chain matches the public Jev MCP wrapper: `TYPESAFE_API_KEY`, then
//! `~/.config/typesafe.env`, then `~/.config/environment.d/60-typesafe.conf`.
//! Values are never interpolated into logs or errors.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use serde_json::{Value, json};

use crate::{Error, embed::Rerank, fuse::FusedHit};

/// Process-env key name (also the assignment key in config files).
pub const ENV_API_KEY: &str = "TYPESAFE_API_KEY";
const DOTENV_REL: &str = ".config/typesafe.env";
const ENVIRONMENTD_REL: &str = ".config/environment.d/60-typesafe.conf";
const ENV_BASE_URL: &str = "TYPESAFE_BASE_URL";
const ENV_MODEL: &str = "JEV_MCP_MODEL";
const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
const DEFAULT_MODEL: &str = "jev-1.13.0";
const STATE_CHARS: usize = 800;
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Outcome of an optional Jev pass.
#[derive(Debug, Clone, PartialEq)]
pub struct RankStatus {
    /// `jev` or `fallback`.
    pub mode: &'static str,
    /// Header line for CLI / MCP (never contains the API key).
    pub note: String,
    /// Absolute P(any candidate answers), when Jev ran.
    pub exists: Option<f64>,
}

impl RankStatus {
    fn jev(exists: Option<f64>) -> Self {
        // Per-action setpoint lives next to the reading: below 0.30 the
        // shortlist is returned unchanged but flagged low-confidence, so a
        // caller scales to warn/verify instead of auto-acting.
        let note = match exists {
            Some(p) if p < 0.30 => format!("rank: jev exists={p:.2} (low — verify before acting)"),
            Some(p) => format!("rank: jev exists={p:.2}"),
            None => "rank: jev".to_owned(),
        };
        Self {
            mode: "jev",
            note,
            exists,
        }
    }

    fn fallback(reason: &str) -> Self {
        Self {
            mode: "fallback",
            note: format!("rank: fallback ({reason})"),
            exists: None,
        }
    }
}

/// Resolve `TYPESAFE_API_KEY` from env, then the two config files.
///
/// # Errors
///
/// Returns [`Error::Jev`] naming sources, never key values.
pub fn resolve_api_key_from_env() -> Result<String, Error> {
    let env_value = std::env::var(ENV_API_KEY).ok();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let dotenv = home.as_ref().map(|h| h.join(DOTENV_REL));
    let environmentd = home.as_ref().map(|h| h.join(ENVIRONMENTD_REL));
    resolve_api_key(
        env_value.as_deref(),
        dotenv.as_deref(),
        environmentd.as_deref(),
    )
}

/// Testable chain: env string, then dotenv path, then environment.d path.
pub fn resolve_api_key(
    env_value: Option<&str>,
    dotenv_path: Option<&Path>,
    environmentd_path: Option<&Path>,
) -> Result<String, Error> {
    if let Some(key) = nonempty(env_value) {
        return Ok(key);
    }
    if let Some(key) = read_key_file(dotenv_path)? {
        return Ok(key);
    }
    if let Some(key) = read_key_file(environmentd_path)? {
        return Ok(key);
    }
    Err(Error::Jev(format!(
        "missing `{ENV_API_KEY}` (checked process env, ~/{DOTENV_REL}, ~/{ENVIRONMENTD_REL})"
    )))
}

fn read_key_file(path: Option<&Path>) -> Result<Option<String>, Error> {
    let Some(path) = path else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|_| Error::Jev(format!("could not read {}", path.display())))?;
    Ok(parse_assignment_value(&text, ENV_API_KEY).and_then(|v| nonempty(Some(v.as_str()))))
}

fn nonempty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// Parse `KEY=value` / `export KEY=value` lines; last matching assignment wins.
#[must_use]
pub fn parse_assignment_value(contents: &str, key: &str) -> Option<String> {
    let mut found = None;
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export").map_or(line, |rest| {
            if rest.starts_with(char::is_whitespace) {
                rest.trim_start()
            } else {
                line
            }
        });
        let Some((lhs, rhs)) = line.split_once('=') else {
            continue;
        };
        if lhs.trim() != key {
            continue;
        }
        found = Some(unquote(rhs.trim()));
    }
    found.filter(|v| !v.is_empty())
}

fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return value[1..value.len() - 1].to_owned();
        }
    }
    value.to_owned()
}

/// HTTP System One client used as a [`Rerank`] backend.
pub struct JevReranker {
    client: reqwest::blocking::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl std::fmt::Debug for JevReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevReranker")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl JevReranker {
    /// Load key + model from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Jev`] when the key is missing or the HTTP client
    /// cannot be built.
    pub fn from_env() -> Result<Self, Error> {
        let api_key = resolve_api_key_from_env()?;
        let base_url = std::env::var(ENV_BASE_URL).unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        let model = std::env::var(ENV_MODEL).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        let client = reqwest::blocking::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .map_err(|e| Error::Jev(e.to_string()))?;
        Ok(Self {
            client,
            api_key,
            base_url: base_url.trim_end_matches('/').to_owned(),
            model,
        })
    }

    fn evaluate(&self, state: Value, questions: Value) -> Result<Value, Error> {
        let url = format!("{}/v1/systemone", self.base_url);
        let body = json!({
            "state": state,
            "model": self.model,
            "questions": questions,
        });
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .map_err(|e| Error::Jev(sanitize_http(&e.to_string())))?;
        let status = response.status();
        let text = response
            .text()
            .map_err(|e| Error::Jev(sanitize_http(&e.to_string())))?;
        if !status.is_success() {
            return Err(Error::Jev(format!(
                "systemone HTTP {status}: {}",
                sanitize_http(&truncate_err(&text))
            )));
        }
        serde_json::from_str(&text).map_err(|e| Error::Jev(e.to_string()))
    }
}

fn sanitize_http(msg: &str) -> String {
    // Strip accidental bearer tokens if a library interpolates headers.
    msg.split("Bearer ")
        .enumerate()
        .map(|(i, part)| {
            if i == 0 {
                part.to_owned()
            } else {
                let rest = part
                    .find(|c: char| c.is_whitespace() || c == '"' || c == ',')
                    .map_or("", |idx| &part[idx..]);
                format!("Bearer <redacted>{rest}")
            }
        })
        .collect()
}

fn truncate_err(text: &str) -> String {
    const CAP: usize = 180;
    let trimmed = text.trim();
    if trimmed.len() <= CAP {
        return trimmed.to_owned();
    }
    let mut end = CAP;
    while !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &trimmed[..end])
}

impl Rerank for JevReranker {
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>, Error> {
        rerank_nouls(query, docs, |state, questions| {
            let answers = self.evaluate(state, questions)?;
            Ok(answers.get("answers").cloned().unwrap_or(Value::Null))
        })
    }
}

/// Score each doc with a Noul; sort highest first. `exists` is recorded
/// by the caller via [`rerank_hits`].
///
/// Meter discipline: one Noul per candidate plus one `exists` Noul, each an
/// absolute P(yes) with no confidence field. Questions share the same state
/// (query + candidate list) but must not read each other's answers: every
/// per-candidate instruction judges that candidate alone, never by rank
/// against the others. A missing Noul fails closed to 0.0.
fn rerank_nouls(
    query: &str,
    docs: &[&str],
    mut evaluate: impl FnMut(Value, Value) -> Result<Value, Error>,
) -> Result<Vec<(usize, f32)>, Error> {
    if docs.is_empty() {
        return Ok(Vec::new());
    }
    let mut candidates = Vec::with_capacity(docs.len());
    let mut questions = serde_json::Map::new();
    questions.insert(
        "exists".into(),
        json!({
            "type": "noul",
            "instructions": "Does any candidate supply evidence that answers the query?",
            "criteria": {
                "true": "At least one candidate states or implements what the query asks for.",
                "false": "No candidate addresses the query."
            }
        }),
    );
    for (i, doc) in docs.iter().enumerate() {
        let id = format!("c{i}");
        candidates.push(json!({
            "id": id,
            "text": truncate_state(doc),
        }));
        questions.insert(
            format!("c{i}"),
            json!({
                "type": "noul",
                "instructions": format!("Does candidate c{i} answer the query directly?"),
                "criteria": {
                    "true": "This candidate states or implements what the query asks for.",
                    "false": "This candidate is only loosely related, or is an example rather than the answer."
                }
            }),
        );
    }
    let state = json!({ "query": query, "candidates": candidates });
    let answers = evaluate(state, Value::Object(questions))?;
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(docs.len());
    for i in 0..docs.len() {
        let noul = answers
            .get(format!("c{i}"))
            .and_then(|a| a.get("noul"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        scored.push((i, noul as f32));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(scored)
}

fn noul_exists(answers: &Value) -> Option<f64> {
    answers
        .get("exists")
        .and_then(|a| a.get("noul"))
        .and_then(Value::as_f64)
}

fn truncate_state(text: &str) -> &str {
    if text.len() <= STATE_CHARS {
        return text;
    }
    let mut end = STATE_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Reorder hits with Jev when a key is configured; otherwise leave order
/// and report fallback. Never fails the search. Captures the `exists` Noul
/// from the same evaluate call as the per-candidate scores.
#[must_use]
pub fn rerank_hits(query: &str, hits: Vec<FusedHit>) -> (RankStatus, Vec<FusedHit>) {
    if hits.is_empty() {
        return (RankStatus::fallback("empty shortlist"), hits);
    }
    let reranker = match JevReranker::from_env() {
        Ok(r) => r,
        Err(e) => return (RankStatus::fallback(&e.to_string()), hits),
    };
    let docs: Vec<String> = hits
        .iter()
        .map(|h| format!("{} {}", h.breadcrumb, h.text))
        .collect();
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let mut exists = None;
    let order = match rerank_nouls(query, &refs, |state, questions| {
        let body = reranker.evaluate(state, questions)?;
        let answers = body.get("answers").cloned().unwrap_or(Value::Null);
        exists = noul_exists(&answers);
        Ok(answers)
    }) {
        Ok(order) => order,
        Err(e) => return (RankStatus::fallback(&e.to_string()), hits),
    };
    let mut rescored = Vec::with_capacity(hits.len());
    let mut seen = std::collections::HashSet::new();
    for (orig, score) in &order {
        if *orig >= hits.len() {
            continue;
        }
        seen.insert(*orig);
        let mut hit = hits[*orig].clone();
        hit.score = f64::from(*score);
        rescored.push(hit);
    }
    for (i, hit) in hits.iter().enumerate() {
        if !seen.contains(&i) {
            rescored.push(hit.clone());
        }
    }
    (RankStatus::jev(exists), rescored)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMMY_ENV: &str = "dummy-ts-key-env";
    const DUMMY_DOTENV: &str = "dummy-ts-key-dotenv";

    #[test]
    fn parses_export_and_quotes() {
        let text = "export TYPESAFE_API_KEY=\"dummy-ts-key-dotenv\"\n";
        assert_eq!(
            parse_assignment_value(text, ENV_API_KEY).as_deref(),
            Some(DUMMY_DOTENV)
        );
    }

    #[test]
    fn missing_key_errors_list_sources_not_values() {
        let err = resolve_api_key(None, None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(ENV_API_KEY));
        assert!(msg.contains(DOTENV_REL));
        assert!(!msg.contains(DUMMY_ENV));
        assert!(!msg.contains(DUMMY_DOTENV));
    }

    #[test]
    fn chain_prefers_process_env() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dotenv = dir.path().join("typesafe.env");
        std::fs::write(&dotenv, format!("TYPESAFE_API_KEY={DUMMY_DOTENV}\n")).unwrap();
        let key = resolve_api_key(Some(DUMMY_ENV), Some(&dotenv), None).expect("env");
        assert_eq!(key, DUMMY_ENV);
    }

    #[test]
    fn rerank_nouls_orders_by_score() {
        let docs = ["alpha", "bravo", "charlie"];
        let order = rerank_nouls("q", &docs, |_state, _q| {
            Ok(json!({
                "exists": {"type": "noul", "noul": 0.9},
                "c0": {"type": "noul", "noul": 0.1},
                "c1": {"type": "noul", "noul": 0.8},
                "c2": {"type": "noul", "noul": 0.4},
            }))
        })
        .expect("rank");
        assert_eq!(order, vec![(1, 0.8), (2, 0.4), (0, 0.1)]);
    }

    #[test]
    fn low_exists_flags_verify_band() {
        assert!(RankStatus::jev(Some(0.12)).note.contains("low"));
        assert!(!RankStatus::jev(Some(0.85)).note.contains("low"));
        // Missing reading fails closed: no exists value, no approval signal.
        assert_eq!(RankStatus::jev(None).note, "rank: jev");
    }

    #[test]
    fn sanitize_http_redacts_bearer() {
        let msg = sanitize_http("Authorization: Bearer super-secret-key rest");
        assert!(msg.contains("<redacted>"));
        assert!(!msg.contains("super-secret-key"));
    }

    /// The `TYPESAFE_BASE_URL` override points the ranker at any
    /// SystemOne-wire-compatible endpoint (hosted Jev, LocalJev, OpenJev).
    /// A std-only mock proves the full round trip without network.
    #[test]
    fn jev_reranker_round_trips_systemone_wire() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let body = r#"{"answers":{"exists":{"type":"noul","noul":0.9},"c0":{"type":"noul","noul":0.2},"c1":{"type":"noul","noul":0.7}}}"#;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = vec![0u8; 65536];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).expect("write");
        });
        let prev_key = std::env::var(ENV_API_KEY).ok();
        let prev_url = std::env::var(ENV_BASE_URL).ok();
        // SAFETY: no other test in this binary reads these vars
        // concurrently in a way that changes its verdict ( Jev callers all
        // fail closed to the same fallback order on error).
        unsafe {
            std::env::set_var(ENV_API_KEY, "mock-key");
            std::env::set_var(ENV_BASE_URL, format!("http://{addr}"));
        }
        let order = JevReranker::from_env().and_then(|r| r.rerank("q", &["alpha", "bravo"]));
        unsafe {
            match prev_key {
                Some(v) => std::env::set_var(ENV_API_KEY, v),
                None => std::env::remove_var(ENV_API_KEY),
            }
            match prev_url {
                Some(v) => std::env::set_var(ENV_BASE_URL, v),
                None => std::env::remove_var(ENV_BASE_URL),
            }
        }
        handle.join().expect("mock server");
        assert_eq!(order.expect("rerank"), vec![(1, 0.7), (0, 0.2)]);
    }
}
