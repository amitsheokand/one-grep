# Intent search and Rust LSP integration

## Goal

Improve retrieval decisions and symbol navigation before adding another embedder. Keep exact lookup deterministic, semantic retrieval intent-driven, and model judgments optional.

## Routing contract

- Known exact anchors (`x:Name`, binding paths, symbols, literals): use `rg` when locating occurrences is sufficient. Do not replace exact lookup with semantic ranking.
- Intent without an identifier: use `search` to retrieve a shortlist.
- Definition, references, or implementation of a symbol at a known location: use LSP navigation. A symbol-like query alone is not enough to resolve scope reliably.
- Mixed intent and anchors: discover candidates with `search`, then verify anchors or navigate their relationships with `rg` or LSP.
- An empty exact lookup is not proof that the concept is absent. Report which evidence source was checked.

## Rust LSP integration

Integrate `rust-analyzer` through a Rust LSP client exposed by the MCP layer. Keep language navigation separate from retrieval ranking and expose provenance in results.

Initial capabilities:

- `textDocument/definition`
- `textDocument/references`
- `textDocument/implementation`

Inputs must identify the workspace, document URI, and source position. Translate positions using the negotiated encoding; normalize LSP `Location` and `LocationLink` results into navigable file/range evidence.

Manage initialization, workspace roots, document synchronization, request cancellation, timeouts, and shutdown. Reuse a server per workspace rather than spawning one per request. Discover capabilities before calling optional methods. Surface missing servers, unsupported methods, and stale or unavailable source positions distinctly from an authoritative empty result.

Rust is the first supported language. `rust-analyzer` does not resolve XAML controls or framework binding paths: those still require exact search or a future language/framework-specific provider. Do not present the Rust integration as a general solution to “find the definition of this control.”

Preserve local-first behavior. Make executable selection and workspace trust explicit; do not silently install servers or enable project code execution. Validate returned locations against configured workspace/dependency navigation policy.

## Shortlist intelligence

Run optional judgments only after existing retrieval builds a bounded shortlist, initially 10–20 candidates.

### Exists

Ask whether any shortlisted candidate supplies evidence answering the query. Return `supported`, `unsupported`, or `uncertain` according to empirically evaluated thresholds.

An unsupported shortlist means “not established by these candidates,” not “does not exist in the workspace.” Preserve the distinction in MCP output.

### Where

Choose a candidate ID from the shortlist, with explicit `none` and `uncertain` outcomes. Validate IDs against the submitted candidate set. Preserve source citations rather than generating locations.

### Typed relevance facets

Evaluate separately:

- Answers the query directly.
- Defines the requested symbol or behavior.
- Provides an example rather than an implementation.

Combine facets with explicit weights in code. Keep original retrieval scores and ordering available for diagnostics and fallback. Do not treat model scores as interchangeable with RRF scores.

### Backends and fallback

- Opt-in TypeSafe backend: external transmission of query/snippets requires explicit configuration.
- Optional Open Jev backend: separately hosted GPU inference, not a lightweight local embedding replacement.
- Backend-neutral typed response validation; unsupported probability capabilities remain explicit.
- Open Jev restricted-softmax scores are uncalibrated and are not a substitute for Jev probability semantics.
- Bound request size, concurrency, latency, and cost. On timeout, malformed response, invalid candidate ID, or backend error, return original retrieval order with fallback status.
- Calibrate abstention thresholds on held-out workspace examples. Confidence is not proof of correctness or a security boundary.
- Avoid logging source snippets or credentials by default.

## Evidence and acceptance criteria

Build labeled workspace queries before implementing ranking changes. Include exact identifiers, paraphrased intent, Rust definitions/references/implementations, ambiguous names, examples versus implementations, and genuinely unanswerable queries.

Measure:

- Recall@20 for first-stage retrieval.
- MRR and top-1 accuracy before and after shortlist judgments.
- Empty-result correctness and false-abstention rate.
- End-to-end p50/p95 latency, fallback rate, and backend cost.
- LSP navigation accuracy with duplicate names, multiple crates, UTF-16 position conversion, and modified documents.

Retain reproducible receipts: query-set version, revision, backend/model version, shortlist IDs, judgments, timing, and aggregate metrics. Keep private source content out of published receipts.

Tests must cover deterministic routing, unavailable LSP servers, cancellation, unsupported capabilities, invalid locations, shortlist-only absence claims, invalid candidate IDs, and stable fallback ordering.

## Delivery order

1. Establish labeled queries and baseline receipts.
2. Formalize exact/intent/navigation routing and result provenance.
3. Integrate Rust LSP navigation with `rust-analyzer`.
4. Add opt-in `exists`/`where` evaluation over existing shortlists with validated outputs and fallback.
5. Add typed relevance facets and compare against baseline.
6. Consider another embedder only if evidence shows first-stage recall remains the limiting factor.

## Sources

- https://github.com/JoshuaSP/open-jev
- https://docs.typesafe.ai/introduction
- https://docs.typesafe.ai/cookbooks/rerank_typesafe.md
- https://docs.typesafe.ai/confidence.md
- https://docs.typesafe.ai/model-jaggedness/jev-1.13.md
