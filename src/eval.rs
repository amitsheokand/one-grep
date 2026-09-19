//! Phase-1 labeled retrieval evaluation (`PACKET-ideasearch.md`).
//!
//! Versioned, self-contained query set over a synthetic fixture workspace.
//! Measures first-stage lexical recall@k, MRR, top-1 accuracy, and
//! empty-result correctness before any ranking or model changes. Receipts
//! carry provenance (eval version, package version, per-query ranks) without
//! embedding source snippets.

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::{Error, index};

/// Eval set version. Bump when cases or fixtures change.
pub const EVAL_VERSION: &str = "ideasearch-v1";
/// Rank depth for recall reporting.
pub const RECALL_K: usize = 5;
/// Rank depth for the wider recall signal.
pub const RECALL_WIDE_K: usize = 20;

/// Query category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryKind {
    Exact,
    Intent,
    Unanswerable,
}

/// One labeled case. `expected` is a path substring identifying the target
/// file; `None` means the workspace genuinely has no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryCase {
    pub id: &'static str,
    pub kind: QueryKind,
    pub query: &'static str,
    pub expected: Option<&'static str>,
}

/// Baseline set: exact identifiers, paraphrased intent, and one
/// genuinely unanswerable query. Each answerable case uses terms unique
/// to its target file so the lexical baseline is deterministic.
pub const CASES: &[QueryCase] = &[
    QueryCase {
        id: "exact-auth-symbol",
        kind: QueryKind::Exact,
        query: "authenticate_user",
        expected: Some("src/auth.rs"),
    },
    QueryCase {
        id: "exact-route-symbol",
        kind: QueryKind::Exact,
        query: "route_request",
        expected: Some("src/router.rs"),
    },
    QueryCase {
        id: "intent-login-credentials",
        kind: QueryKind::Intent,
        query: "how does login verify credentials",
        expected: Some("src/auth.rs"),
    },
    QueryCase {
        id: "intent-route-dispatch",
        kind: QueryKind::Intent,
        query: "which handler dispatches incoming requests to lanes",
        expected: Some("src/router.rs"),
    },
    QueryCase {
        id: "intent-secret-storage",
        kind: QueryKind::Intent,
        query: "where do tokens live so they never get committed",
        expected: Some("docs/secrets.md"),
    },
    QueryCase {
        id: "unanswerable-nonsense",
        kind: QueryKind::Unanswerable,
        query: "zxqv klaxon theremin calibration",
        expected: None,
    },
];

/// Per-query outcome (no source text retained).
#[derive(Debug, Clone, Serialize)]
pub struct QueryOutcome {
    pub id: &'static str,
    #[serde(rename = "type")]
    pub kind: QueryKind,
    pub query: &'static str,
    pub expected: Option<&'static str>,
    /// 1-based rank of the first hit matching `expected`.
    pub rank: Option<usize>,
    pub hits: usize,
    pub latency_ms: f64,
}

/// Aggregate baseline report.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub eval_version: &'static str,
    pub package_version: &'static str,
    pub queries: usize,
    pub answerable: usize,
    pub unanswerable: usize,
    pub recall_at_5: f64,
    pub recall_at_20: f64,
    pub mrr: f64,
    pub top1_accuracy: f64,
    pub empty_correct: usize,
    pub empty_total: usize,
    pub false_abstentions: usize,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub outcomes: Vec<QueryOutcome>,
}

/// Build the synthetic fixture workspace the cases are labeled against.
pub fn fixture_workspace(dir: &Path) -> Result<(), Error> {
    let files: &[(&str, &str)] = &[
        (
            "src/auth.rs",
            "/// login credential verification\n\
             /// Verifies a login by checking the supplied credentials.\n\
             pub fn authenticate_user(login: &str, credentials: &str) -> bool {\n\
             let _ = (login, credentials);\n\
             true\n\
             }\n",
        ),
        (
            "src/router.rs",
            "/// Request lane dispatch.\n\
             /// The handler dispatches each incoming request to its lane.\n\
             pub fn route_request(incoming: &str) -> &str {\n\
             let _ = incoming;\n\
             \"lane\"\n\
             }\n",
        ),
        (
            "docs/secrets.md",
            "# Secrets\n\
             \n\
             Store tokens in the OS keychain so they never get committed.\n\
             Never check tokens into git.\n",
        ),
    ];
    for (rel, contents) in files {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, contents)?;
    }
    Ok(())
}

/// Run the labeled set against an indexed workspace with lexical search.
pub fn evaluate(workspace: &Path, cases: &[QueryCase], limit: usize) -> Result<Report, Error> {
    let mut outcomes = Vec::with_capacity(cases.len());
    for case in cases {
        let start = Instant::now();
        let hits = index::search(workspace, case.query, limit)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let rank = case.expected.and_then(|expected| {
            hits.iter()
                .position(|h| h.path.to_string_lossy().contains(expected))
                .map(|i| i + 1)
        });
        outcomes.push(QueryOutcome {
            id: case.id,
            kind: case.kind,
            query: case.query,
            expected: case.expected,
            rank,
            hits: hits.len(),
            latency_ms,
        });
    }
    Ok(report(&outcomes))
}

fn report(outcomes: &[QueryOutcome]) -> Report {
    let answerable: Vec<&QueryOutcome> = outcomes.iter().filter(|o| o.expected.is_some()).collect();
    let unanswerable: Vec<&QueryOutcome> =
        outcomes.iter().filter(|o| o.expected.is_none()).collect();
    let hit_within = |k: usize| {
        answerable
            .iter()
            .filter(|o| o.rank.is_some_and(|r| r <= k))
            .count()
    };
    let reciprocal_sum: f64 = answerable
        .iter()
        .map(|o| o.rank.map_or(0.0, |r| 1.0 / r as f64))
        .sum();
    let empty_correct = unanswerable.iter().filter(|o| o.hits == 0).count();
    let false_abstentions = answerable.iter().filter(|o| o.hits == 0).count();
    let mut latencies: Vec<f64> = outcomes.iter().map(|o| o.latency_ms).collect();
    latencies.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        if latencies.is_empty() {
            return 0.0;
        }
        let idx = ((latencies.len() as f64 * q).ceil() as usize).saturating_sub(1);
        latencies[idx.min(latencies.len() - 1)]
    };
    let denom = answerable.len().max(1) as f64;
    Report {
        eval_version: EVAL_VERSION,
        package_version: env!("CARGO_PKG_VERSION"),
        queries: outcomes.len(),
        answerable: answerable.len(),
        unanswerable: unanswerable.len(),
        recall_at_5: hit_within(RECALL_K) as f64 / denom,
        recall_at_20: hit_within(RECALL_WIDE_K) as f64 / denom,
        mrr: reciprocal_sum / denom,
        top1_accuracy: hit_within(1) as f64 / denom,
        empty_correct,
        empty_total: unanswerable.len(),
        false_abstentions,
        latency_p50_ms: quantile(0.5),
        latency_p95_ms: quantile(0.95),
        outcomes: outcomes.to_vec(),
    }
}

/// JSON-serializable receipt for a report.
#[must_use]
pub fn receipt(report: &Report) -> serde_json::Value {
    serde_json::to_value(report).unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indexed_fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        fixture_workspace(dir.path()).expect("fixture");
        index::sync(dir.path()).expect("sync");
        dir
    }

    #[test]
    fn baseline_fixture_meets_thresholds() {
        let dir = indexed_fixture();
        let report = evaluate(dir.path(), CASES, RECALL_WIDE_K).expect("evaluate");
        assert_eq!(report.eval_version, EVAL_VERSION);
        assert_eq!(report.answerable, 5);
        assert_eq!(report.unanswerable, 1);
        assert_eq!(report.recall_at_5, 1.0);
        assert_eq!(report.recall_at_20, 1.0);
        assert_eq!(report.top1_accuracy, 1.0);
        assert_eq!(report.mrr, 1.0);
        assert_eq!(report.empty_correct, report.empty_total);
        assert_eq!(report.false_abstentions, 0);
    }

    #[test]
    fn receipt_carries_provenance_without_source_text() {
        let dir = indexed_fixture();
        let report = evaluate(dir.path(), CASES, RECALL_WIDE_K).expect("evaluate");
        let value = receipt(&report);
        assert_eq!(value["eval_version"], EVAL_VERSION);
        assert_eq!(value["queries"], CASES.len());
        assert!(
            value["outcomes"]
                .as_array()
                .is_some_and(|o| o.len() == CASES.len())
        );
        let text = value.to_string();
        assert!(!text.contains("Verifies a login"));
        assert!(!text.contains("OS keychain"));
    }

    #[test]
    fn unanswerable_query_reports_empty_not_absent() {
        let dir = indexed_fixture();
        let report = evaluate(dir.path(), CASES, RECALL_WIDE_K).expect("evaluate");
        let outcome = report
            .outcomes
            .iter()
            .find(|o| o.kind == QueryKind::Unanswerable)
            .expect("unanswerable outcome");
        assert_eq!(outcome.rank, None);
        assert_eq!(outcome.hits, 0);
    }
}
