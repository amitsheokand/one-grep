//! Route a task to installed agent skills via Jev or lexical fallback.
//!
//! Scans `<library>/*/SKILL.md` (YAML front matter only — never the body).

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use serde::Serialize;
use serde_json::Value;

use crate::{
    Error,
    jev::{JevReranker, RankStatus, rerank_nouls},
};

pub const ENV_SKILLS_DIR: &str = "ONE_GREP_SKILLS_DIR";
const DEFAULT_SKILLS_REL: &str = ".local/share/agent-skills";
const DESC_CAP: usize = 200;
const EXISTS_FLOOR: f64 = 0.5;
const SCORE_FLOOR: f32 = 0.5;
const LEXICAL_SCORE_FLOOR: f32 = 0.2;

/// One skill entry from front matter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
    /// Absolute path to `SKILL.md` (canonical when the OS allows).
    pub skill_path: PathBuf,
}

/// A ranked skill returned to callers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SkillHit {
    pub name: String,
    pub path: String,
    pub score: f32,
    pub description: String,
}

/// Outcome of [`run`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SkillResult {
    pub rank_note: String,
    pub notes: Vec<String>,
    pub hits: Vec<SkillHit>,
    /// Set when Jev `exists` or per-candidate scores fail the floor.
    pub no_match: bool,
}

impl SkillResult {
    /// Human-readable text (rank header, notes, hits or `no matching skill`).
    #[must_use]
    pub fn format_text(&self) -> String {
        let mut lines = vec![self.rank_note.clone()];
        if !self.notes.is_empty() {
            lines.extend(self.notes.clone());
        }
        if self.no_match {
            lines.push("no matching skill".to_owned());
            return lines.join("\n");
        }
        for hit in &self.hits {
            lines.push(format!("{}  {}  ({:.4})", hit.name, hit.path, hit.score));
            lines.push(hit.description.clone());
        }
        lines.join("\n")
    }
}

/// Resolve the skill library directory.
#[must_use]
pub fn resolve_library_dir(dir: Option<&Path>) -> PathBuf {
    if let Some(d) = dir {
        return d.to_path_buf();
    }
    if let Ok(raw) = std::env::var(ENV_SKILLS_DIR) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|h| h.join(DEFAULT_SKILLS_REL))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SKILLS_REL))
}

/// Scan `dir/*/SKILL.md`, parse front matter, collect skip notes.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] when the directory cannot be read.
pub fn scan_library(dir: &Path) -> Result<(Vec<SkillEntry>, Vec<String>), Error> {
    let mut notes = Vec::new();
    if !dir.exists() {
        notes.push(format!("skill library not found: {}", dir.display()));
        return Ok((Vec::new(), notes));
    }
    if !dir.is_dir() {
        return Err(Error::InvalidInput(format!(
            "skill library is not a directory: {}",
            dir.display()
        )));
    }
    let mut folders = Vec::new();
    let read_dir = std::fs::read_dir(dir)
        .map_err(|e| Error::InvalidInput(format!("could not read {}: {e}", dir.display())))?;
    for entry in read_dir {
        let entry = entry.map_err(|e| Error::InvalidInput(e.to_string()))?;
        let path = entry.path();
        let meta =
            std::fs::symlink_metadata(&path).map_err(|e| Error::InvalidInput(e.to_string()))?;
        let is_dir = if meta.file_type().is_symlink() {
            path.metadata().map(|m| m.is_dir()).unwrap_or(false)
        } else {
            meta.is_dir()
        };
        if is_dir {
            folders.push(path);
        }
    }
    folders.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .cmp(b.file_name().unwrap_or_default())
    });
    let mut skills = Vec::new();
    for folder in folders {
        let skill_md = folder.join("SKILL.md");
        if !skill_md.is_file() {
            continue;
        }
        let folder_name = folder
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let rel_label = format!("{folder_name}/SKILL.md");
        let skill_path = std::fs::canonicalize(&skill_md).unwrap_or(skill_md);
        let text = std::fs::read_to_string(&skill_path)
            .map_err(|e| Error::InvalidInput(format!("{}: {e}", skill_path.display())))?;
        match parse_skill_md(&text, &folder_name) {
            Ok(Some(entry)) => {
                skills.push(SkillEntry {
                    name: entry.name,
                    description: entry.description,
                    skill_path,
                });
            }
            Ok(None) => {
                notes.push(format!("skipped {rel_label}: missing description"));
            }
            Err(e) => {
                notes.push(format!("skipped {rel_label}: {e}"));
            }
        }
    }
    Ok((skills, notes))
}

struct ParsedFront {
    name: String,
    description: String,
}

fn parse_skill_md(text: &str, folder_name: &str) -> Result<Option<ParsedFront>, String> {
    let body = text.trim_start();
    if !body.starts_with("---") {
        return Err("missing YAML front matter".into());
    }
    let rest = body.strip_prefix("---").unwrap_or(body);
    let rest = rest.trim_start_matches('\n').trim_start_matches('\r');
    let end = rest
        .find("\n---")
        .ok_or_else(|| "unclosed front matter".to_string())?;
    let front = &rest[..end];
    let mut name = None;
    let mut description = None;
    let lines: Vec<&str> = front.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim();
            let value = value.trim();
            if key == "name" {
                name = Some(parse_scalar(value, &lines, &mut i)?.trim().to_owned());
                i += 1;
                continue;
            }
            if key == "description" {
                description = Some(parse_scalar(value, &lines, &mut i)?);
                i += 1;
                continue;
            }
        }
        i += 1;
    }
    let name = name.unwrap_or_else(|| folder_name.to_owned());
    let Some(description) = description.filter(|d| !d.trim().is_empty()) else {
        return Ok(None);
    };
    Ok(Some(ParsedFront {
        name,
        description: collapse_one_line(&description),
    }))
}

fn parse_scalar(initial: &str, lines: &[&str], i: &mut usize) -> Result<String, String> {
    if initial.is_empty() {
        return Ok(String::new());
    }
    if initial == ">" || initial == "|" {
        return parse_block(initial, lines, i);
    }
    if (initial.starts_with('"') && initial.ends_with('"'))
        || (initial.starts_with('\'') && initial.ends_with('\''))
    {
        return Ok(unquote_scalar(initial));
    }
    Ok(initial.to_owned())
}

fn parse_block(kind: &str, lines: &[&str], i: &mut usize) -> Result<String, String> {
    let folded = kind == ">";
    let mut parts = Vec::new();
    let mut j = *i + 1;
    while j < lines.len() {
        let line = lines[j];
        if line.is_empty() {
            j += 1;
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }
        parts.push(line.trim_start().to_owned());
        j += 1;
    }
    *i = j.saturating_sub(1);
    if folded {
        Ok(parts.join(" ").trim().to_owned())
    } else {
        Ok(parts.join("\n"))
    }
}

fn unquote_scalar(value: &str) -> String {
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

fn collapse_one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cap_description(text: &str) -> String {
    let one = collapse_one_line(text);
    if one.len() <= DESC_CAP {
        return one;
    }
    let mut end = DESC_CAP;
    while end > 0 && !one.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &one[..end])
}

/// Rank skills for `task` and return up to `limit` hits.
#[must_use]
pub fn run(task: &str, dir: Option<&Path>, limit: usize) -> SkillResult {
    let limit = limit.clamp(1, 5);
    let library = resolve_library_dir(dir);
    let library = std::fs::canonicalize(&library).unwrap_or(library);
    let (entries, scan_notes) = match scan_library(&library) {
        Ok(pair) => pair,
        Err(e) => {
            return SkillResult {
                rank_note: RankStatus::fallback(&e.to_string()).note,
                notes: vec![e.to_string()],
                hits: Vec::new(),
                no_match: true,
            };
        }
    };
    if entries.is_empty() {
        let status = RankStatus::fallback("empty library");
        return SkillResult {
            rank_note: status.note,
            notes: scan_notes,
            hits: Vec::new(),
            no_match: true,
        };
    }
    let (status, hits) = rank_entries(task, &entries, limit);
    let no_match = hits.is_empty();
    SkillResult {
        rank_note: status.note,
        notes: scan_notes,
        hits,
        no_match,
    }
}

fn rank_entries(task: &str, entries: &[SkillEntry], limit: usize) -> (RankStatus, Vec<SkillHit>) {
    let reranker = match JevReranker::from_env() {
        Ok(r) => r,
        Err(e) => return lexical_rank(task, entries, limit, RankStatus::fallback(&e.to_string())),
    };
    let docs: Vec<String> = entries
        .iter()
        .map(|e| format!("{} {}", e.name, e.description))
        .collect();
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let mut exists = None;
    let order = match rerank_nouls(task, &refs, |state, questions| {
        let body = reranker.evaluate(state, questions)?;
        let answers = body.get("answers").cloned().unwrap_or(Value::Null);
        exists = answers
            .get("exists")
            .and_then(|a| a.get("noul"))
            .and_then(Value::as_f64);
        Ok(answers)
    }) {
        Ok(order) => order,
        Err(e) => return lexical_rank(task, entries, limit, RankStatus::fallback(&e.to_string())),
    };
    let status = RankStatus::jev(exists);
    if exists.is_some_and(|p| p < EXISTS_FLOOR) {
        return (status, Vec::new());
    }
    let hits = order
        .into_iter()
        .filter(|(_, score)| *score >= SCORE_FLOOR)
        .take(limit)
        .map(|(idx, score)| entry_to_hit(&entries[idx], score))
        .collect::<Vec<_>>();
    (status, hits)
}

fn lexical_rank(
    task: &str,
    entries: &[SkillEntry],
    limit: usize,
    status: RankStatus,
) -> (RankStatus, Vec<SkillHit>) {
    let task_tokens = tokens(task);
    if task_tokens.is_empty() {
        return (status, Vec::new());
    }
    let mut scored: Vec<(usize, f32, usize)> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let doc_tokens = tokens(&format!("{} {}", e.name, e.description));
            let overlap = task_tokens.intersection(&doc_tokens).count();
            let score = overlap as f32 / task_tokens.len() as f32;
            (i, score, overlap)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| entries[a.0].name.cmp(&entries[b.0].name))
    });
    let hits = scored
        .into_iter()
        .filter(|(_, score, overlap)| *overlap >= 1 && *score >= LEXICAL_SCORE_FLOOR)
        .take(limit)
        .map(|(idx, score, _)| entry_to_hit(&entries[idx], score))
        .collect();
    (status, hits)
}

fn entry_to_hit(entry: &SkillEntry, score: f32) -> SkillHit {
    SkillHit {
        name: entry.name.clone(),
        path: entry.skill_path.display().to_string(),
        score,
        description: cap_description(&entry.description),
    }
}

fn tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1 && !STOPWORDS.contains(t))
        .map(str::to_owned)
        .collect()
}

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with", "is", "are", "be",
    "this", "that", "from", "as", "at", "by", "it", "do", "does", "how", "what", "when", "where",
    "which", "who", "whom", "into", "via", "use", "using", "need", "your", "you",
];

#[cfg(test)]
pub(crate) fn write_fixture(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir.join("plain-skill"))?;
    std::fs::write(
        dir.join("plain-skill/SKILL.md"),
        "---\nname: plain-api\n\
description: Plain skill for REST API documentation lookup\n\
---\nSECRET_BODY_MARKER_PLAIN\n",
    )?;
    std::fs::create_dir_all(dir.join("quoted-skill"))?;
    std::fs::write(
        dir.join("quoted-skill/SKILL.md"),
        "---\nname: winrt-lookup\n\
description: \"Quoted WinRT API reference workflow\"\n\
---\nSECRET_BODY_MARKER_QUOTED\n",
    )?;
    std::fs::create_dir_all(dir.join("folded-skill"))?;
    std::fs::write(
        dir.join("folded-skill/SKILL.md"),
        "---\nname: folded-deploy\n\
description: >\n  Folded deployment checklist for\n  production releases\n\
---\nSECRET_BODY_MARKER_FOLDED\n",
    )?;
    std::fs::create_dir_all(dir.join("no-desc-skill"))?;
    std::fs::write(
        dir.join("no-desc-skill/SKILL.md"),
        "---\nname: empty-desc\n---\nSECRET_BODY_MARKER_NODEC\n",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use crate::jev::{ENV_API_KEY, ENV_BASE_URL, rerank_nouls};

    #[test]
    fn parses_plain_quoted_and_folded_descriptions() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path()).expect("fixture");
        let (skills, notes) = scan_library(dir.path()).expect("scan");
        assert_eq!(skills.len(), 3, "{skills:?} {notes:?}");
        assert!(notes.iter().any(|n| n.contains("no-desc-skill")));
        let plain = skills
            .iter()
            .find(|s| s.name == "plain-api")
            .expect("plain");
        assert!(plain.description.contains("REST API"));
        let quoted = skills
            .iter()
            .find(|s| s.name == "winrt-lookup")
            .expect("quoted");
        assert!(quoted.description.contains("WinRT"));
        let folded = skills
            .iter()
            .find(|s| s.name == "folded-deploy")
            .expect("folded");
        assert!(folded.description.contains("deployment checklist"));
    }

    #[test]
    fn fallback_ranking_is_deterministic_and_picks_winrt() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path()).expect("fixture");
        let home = tempfile::tempdir().expect("home");
        let prev_home = std::env::var_os("HOME");
        let prev_key = std::env::var(ENV_API_KEY).ok();
        unsafe {
            std::env::set_var("HOME", home.path());
            std::env::remove_var(ENV_API_KEY);
        }
        let r1 = run("WinRT API reference workflow", Some(dir.path()), 2);
        let r2 = run("WinRT API reference workflow", Some(dir.path()), 2);
        unsafe {
            match prev_key {
                Some(v) => std::env::set_var(ENV_API_KEY, v),
                None => std::env::remove_var(ENV_API_KEY),
            }
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        assert!(r1.rank_note.contains("fallback"));
        assert!(!r1.no_match);
        assert_eq!(r1.hits[0].name, "winrt-lookup");
        assert_eq!(r1.hits, r2.hits);
        let path = Path::new(&r1.hits[0].path);
        assert!(path.is_absolute(), "{}", r1.hits[0].path);
        assert!(
            r1.hits[0].path.ends_with("quoted-skill/SKILL.md"),
            "{}",
            r1.hits[0].path
        );
    }

    #[test]
    fn fallback_no_overlap_returns_no_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path()).expect("fixture");
        let home = tempfile::tempdir().expect("home");
        let prev_home = std::env::var_os("HOME");
        let prev_key = std::env::var(ENV_API_KEY).ok();
        unsafe {
            std::env::set_var("HOME", home.path());
            std::env::remove_var(ENV_API_KEY);
        }
        let r = run("quantum gardening underwater zzz", Some(dir.path()), 3);
        unsafe {
            match prev_key {
                Some(v) => std::env::set_var(ENV_API_KEY, v),
                None => std::env::remove_var(ENV_API_KEY),
            }
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        assert!(r.no_match);
        assert!(r.hits.is_empty());
        assert!(r.format_text().contains("no matching skill"));
    }

    #[test]
    fn limit_bounds_hits() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path()).expect("fixture");
        let home = tempfile::tempdir().expect("home");
        let prev_home = std::env::var_os("HOME");
        let prev_key = std::env::var(ENV_API_KEY).ok();
        unsafe {
            std::env::set_var("HOME", home.path());
            std::env::remove_var(ENV_API_KEY);
        }
        let r = run("API documentation", Some(dir.path()), 1);
        unsafe {
            match prev_key {
                Some(v) => std::env::set_var(ENV_API_KEY, v),
                None => std::env::remove_var(ENV_API_KEY),
            }
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        assert_eq!(r.hits.len(), 1);
    }

    #[test]
    fn missing_library_is_a_note_not_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("nope");
        let r = run("anything", Some(&missing), 2);
        assert!(r.notes.iter().any(|n| n.contains("not found")));
        assert!(r.no_match);
    }

    #[test]
    fn jev_floor_drops_low_exists_and_orders_by_score() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path()).expect("fixture");
        let (entries, _) = scan_library(dir.path()).expect("scan");
        let docs: Vec<String> = entries
            .iter()
            .map(|e| format!("{} {}", e.name, e.description))
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let order = rerank_nouls("WinRT", &refs, |_state, _q| {
            Ok(serde_json::json!({
                "exists": {"type": "noul", "noul": 0.9},
                "c0": {"type": "noul", "noul": 0.4},
                "c1": {"type": "noul", "noul": 0.2},
                "c2": {"type": "noul", "noul": 0.95},
            }))
        })
        .expect("rank");
        assert_eq!(order[0].0, 2);
        let low_exists = rerank_nouls("q", &refs, |_state, _q| {
            Ok(serde_json::json!({
                "exists": {"type": "noul", "noul": 0.2},
                "c0": {"type": "noul", "noul": 0.9},
            }))
        })
        .expect("rank");
        assert!(!low_exists.is_empty());
        let (entries2, _) = scan_library(dir.path()).expect("scan");
        let body = r#"{"answers":{"exists":{"type":"noul","noul":0.2},"c0":{"type":"noul","noul":0.9},"c1":{"type":"noul","noul":0.95},"c2":{"type":"noul","noul":0.4}}}"#;
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
        unsafe {
            std::env::set_var(ENV_API_KEY, "mock-key");
            std::env::set_var(ENV_BASE_URL, format!("http://{addr}"));
        }
        let r = run("WinRT", Some(dir.path()), 3);
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
        handle.join().expect("mock");
        assert!(r.rank_note.contains("jev"));
        assert!(r.no_match, "{r:?}");
        assert!(entries2.len() >= 2);
    }

    #[test]
    fn jev_orders_high_scores_when_exists_passes() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path()).expect("fixture");
        let body = r#"{"answers":{"exists":{"type":"noul","noul":0.9},"c0":{"type":"noul","noul":0.4},"c1":{"type":"noul","noul":0.2},"c2":{"type":"noul","noul":0.95}}}"#;
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
        unsafe {
            std::env::set_var(ENV_API_KEY, "mock-key");
            std::env::set_var(ENV_BASE_URL, format!("http://{addr}"));
        }
        let r = run("WinRT API", Some(dir.path()), 2);
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
        handle.join().expect("mock");
        assert!(!r.no_match);
        assert_eq!(r.hits[0].name, "winrt-lookup");
        assert!(r.hits[0].score >= 0.5);
    }

    #[test]
    fn output_never_contains_skill_body() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_fixture(dir.path()).expect("fixture");
        let home = tempfile::tempdir().expect("home");
        let prev_home = std::env::var_os("HOME");
        let prev_key = std::env::var(ENV_API_KEY).ok();
        unsafe {
            std::env::set_var("HOME", home.path());
            std::env::remove_var(ENV_API_KEY);
        }
        let r = run("API", Some(dir.path()), 3);
        unsafe {
            match prev_key {
                Some(v) => std::env::set_var(ENV_API_KEY, v),
                None => std::env::remove_var(ENV_API_KEY),
            }
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        let text = r.format_text();
        assert!(!text.contains("SECRET_BODY_MARKER"));
    }
}
