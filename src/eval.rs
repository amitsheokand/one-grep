//! Phase-1 labeled retrieval evaluation (`PACKET-ideasearch.md`).
//!
//! Versioned, self-contained query set over a synthetic fixture workspace.
//! Measures first-stage lexical recall@k, MRR, top-1 accuracy, and
//! empty-result correctness before any ranking or model changes. Receipts
//! carry provenance (eval version, package version, per-query ranks) without
//! embedding source snippets.
//!
//! `ideasearch-v2` (frozen gate): keyword / paraphrase / symbol /
//! call-chain cases over a fixture that includes one TypeScript file, so
//! the gate also covers the non-tree-sitter fallback path. The gate test
//! pins the measured paraphrase R@3 floor: any change that drops concept
//! recall below it fails the build.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::{Error, index};

/// Eval set version. Bump when cases or fixtures change.
pub const EVAL_VERSION: &str = "ideasearch-v1";
/// Frozen gate set version. Bump only with a tasklist decision + new floors.
pub const EVAL_VERSION_V2: &str = "ideasearch-v2";
/// Rank depth for recall reporting.
pub const RECALL_K: usize = 5;
/// Rank depth for the wider recall signal.
pub const RECALL_WIDE_K: usize = 20;
/// Gate depths: R@1 (precision head) and R@3 (agent-visible shortlist).
pub const RECALL_1: usize = 1;
pub const RECALL_3: usize = 3;

/// Query category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryKind {
    Exact,
    Intent,
    Unanswerable,
    Keyword,
    Paraphrase,
    Symbol,
    Chain,
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
    pub recall_at_1: f64,
    pub recall_at_3: f64,
    pub recall_at_5: f64,
    pub recall_at_20: f64,
    /// Per-kind recall@3, keyed by serde kind name (`keyword`,
    /// `paraphrase`, `symbol`, `chain`, …). Unanswerable kinds are absent.
    pub kind_recall_at_3: BTreeMap<String, f64>,
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

/// Frozen gate set (`ideasearch-v2`): keyword / paraphrase / symbol /
///
/// call-chain cases over a wider fixture. `src/lane.ts` is TypeScript on
/// purpose: chunking has no TS grammar, so the `.ts` file exercises the
/// window-fallback path agents actually hit in JS/TS repos. Each
/// answerable case uses terms unique to its target file, except the
/// paraphrases, which share (almost) nothing by design.
pub const CASES_V2: &[QueryCase] = &[
    QueryCase {
        id: "kw-auth-term",
        kind: QueryKind::Keyword,
        query: "authenticate_user",
        expected: Some("src/auth.rs"),
    },
    QueryCase {
        id: "kw-lane-config",
        kind: QueryKind::Keyword,
        query: "LaneConfig",
        expected: Some("src/lane.ts"),
    },
    QueryCase {
        id: "kw-keychain",
        kind: QueryKind::Keyword,
        query: "keychain",
        expected: Some("docs/secrets.md"),
    },
    QueryCase {
        id: "para-signin",
        kind: QueryKind::Paraphrase,
        query: "how does sign-in check a password",
        expected: Some("src/auth.rs"),
    },
    QueryCase {
        id: "para-build-output",
        kind: QueryKind::Paraphrase,
        query: "where do build outputs go",
        expected: Some("docs/deploy.md"),
    },
    QueryCase {
        id: "para-tokens",
        kind: QueryKind::Paraphrase,
        query: "where do tokens live so they never get committed",
        expected: Some("docs/secrets.md"),
    },
    QueryCase {
        id: "sym-route",
        kind: QueryKind::Symbol,
        query: "route_request",
        expected: Some("src/router.rs"),
    },
    QueryCase {
        id: "sym-lane-defaults",
        kind: QueryKind::Symbol,
        query: "apply_lane_defaults",
        expected: Some("src/lane.ts"),
    },
    QueryCase {
        id: "chain-apply",
        kind: QueryKind::Chain,
        query: "what does apply_request call",
        expected: Some("src/caller.rs"),
    },
    QueryCase {
        id: "chain-callers",
        kind: QueryKind::Chain,
        query: "which functions call apply_lane_defaults",
        expected: Some("src/caller.rs"),
    },
    QueryCase {
        id: "unanswerable-nonsense",
        kind: QueryKind::Unanswerable,
        query: "zxqv klaxon theremin calibration",
        expected: None,
    },
    QueryCase {
        id: "unanswerable-refunds",
        kind: QueryKind::Unanswerable,
        query: "how do refunds work",
        expected: None,
    },
];

/// Build the v2 fixture workspace the gate cases are labeled against.
pub fn fixture_workspace_v2(dir: &Path) -> Result<(), Error> {
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
            "/// Request dispatch.\n\
             /// The handler dispatches each incoming request to its lane.\n\
             pub fn route_request(incoming: &str) -> &str {\n\
             let _ = incoming;\n\
             \"lane\"\n\
             }\n",
        ),
        (
            "src/caller.rs",
            "/// Lane setup entry point.\n\
             /// Delegates lane setup to apply_lane_defaults.\n\
             pub fn apply_request(name: &str) -> bool {\n\
             let _ = name;\n\
             apply_lane_defaults()\n\
             }\n",
        ),
        (
            "src/lane.ts",
            "// Lane default settings.\n\
             export interface LaneConfig {\n\
             retries: number;\n\
             }\n\
             export function apply_lane_defaults(config: LaneConfig): boolean {\n\
             return config.retries >= 0;\n\
             }\n",
        ),
        (
            "docs/secrets.md",
            "# Secrets\n\
             \n\
             Store tokens in the OS keychain so they never get committed.\n\
             Never check tokens into git.\n",
        ),
        (
            "docs/deploy.md",
            "# Deploy\n\
             \n\
             Release artifacts land in the vault directory.\n\
             Promote a build by copying its artifacts to the vault.\n",
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
    let mut kind_recall_at_3 = BTreeMap::new();
    {
        let mut by_kind: BTreeMap<&'static str, Vec<&QueryOutcome>> = BTreeMap::new();
        for o in &answerable {
            by_kind.entry(kind_name(o.kind)).or_default().push(*o);
        }
        for (kind, group) in &by_kind {
            let hit = group
                .iter()
                .filter(|o| o.rank.is_some_and(|r| r <= RECALL_3))
                .count();
            kind_recall_at_3.insert((*kind).to_owned(), hit as f64 / group.len().max(1) as f64);
        }
    }
    Report {
        eval_version: EVAL_VERSION,
        package_version: env!("CARGO_PKG_VERSION"),
        queries: outcomes.len(),
        answerable: answerable.len(),
        unanswerable: unanswerable.len(),
        recall_at_1: hit_within(RECALL_1) as f64 / denom,
        recall_at_3: hit_within(RECALL_3) as f64 / denom,
        recall_at_5: hit_within(RECALL_K) as f64 / denom,
        recall_at_20: hit_within(RECALL_WIDE_K) as f64 / denom,
        kind_recall_at_3,
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

/// Serde name for a [`QueryKind`], used as the [`Report::kind_recall_at_3`] key.
#[must_use]
pub fn kind_name(kind: QueryKind) -> &'static str {
    match kind {
        QueryKind::Exact => "exact",
        QueryKind::Intent => "intent",
        QueryKind::Unanswerable => "unanswerable",
        QueryKind::Keyword => "keyword",
        QueryKind::Paraphrase => "paraphrase",
        QueryKind::Symbol => "symbol",
        QueryKind::Chain => "chain",
    }
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

    fn indexed_fixture_v2() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        fixture_workspace_v2(dir.path()).expect("fixture");
        index::sync(dir.path()).expect("sync");
        dir
    }

    /// Merge gate (`ideasearch-v2`): concept R@3 must not drop below the
    /// pinned floor. Floors are measured lexical baselines, not ambitions:
    /// raise them only when the retrieval path genuinely improves.
    ///
    /// Measured 2026-09-24 on the frozen fixture (lexical BM25):
    /// paraphrase R@3 = 2/3, overall R@3 = 9/10.
    #[test]
    fn gate_v2_recall_floors_hold() {
        const PARAPHRASE_R3_FLOOR: f64 = 2.0 / 3.0;
        const OVERALL_R3_FLOOR: f64 = 0.9;
        const EPS: f64 = 1e-9;
        let dir = indexed_fixture_v2();
        let report = evaluate(dir.path(), CASES_V2, RECALL_WIDE_K).expect("evaluate");
        assert_eq!(report.eval_version, EVAL_VERSION);
        assert_eq!(report.answerable, 10);
        assert_eq!(report.unanswerable, 2);
        assert_eq!(report.empty_correct, report.empty_total);
        assert_eq!(report.false_abstentions, 0);
        let paraphrase = report
            .kind_recall_at_3
            .get("paraphrase")
            .copied()
            .unwrap_or(0.0);
        assert!(
            paraphrase + EPS >= PARAPHRASE_R3_FLOOR,
            "concept R@3 dropped: {paraphrase:.3} < {PARAPHRASE_R3_FLOOR:.3}"
        );
        assert!(
            report.recall_at_3 + EPS >= OVERALL_R3_FLOOR,
            "overall R@3 dropped: {:.3} < {OVERALL_R3_FLOOR:.3}",
            report.recall_at_3
        );
    }
}
