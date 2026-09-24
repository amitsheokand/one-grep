# Changelog

## v0.2.0 (2026-09-24)

50 commits since v0.1.0. Themes: concept-first retrieval, output budget,
agent tooling, and the measurement to prove it.

### Retrieval that wins intent

- Shared query router (`src/route.rs`): quoted/`foo::Bar` → exact `rg`,
  single tokens → BM25, natural language → hybrid. Same table for CLI
  `query` and MCP `search` (CLI parity hole closed).
- Frozen eval gate (`ideasearch-v2`, 12 cases: keyword / paraphrase /
  symbol / call-chain + TypeScript fixture): paraphrase R@3 pinned at
  2/3, overall 9/10 — merges fail on regression.
- Chunking fixes from the Run 12 diagnosis: file-header comment chunks,
  80-line symbol cap with breadcrumb inheritance, chain-duplicate dedupe
  (duplicates stacked RRF terms — found via an impossible score),
  extractor version migration (stale indexes report, then upgrade).
- Fusion: RRF k 60→20 after a clean A/B (keyword 8/10→9/10, no regress).
- Tree-sitter grammars for TypeScript, Go, Java (Rust/Python/Nix/Markdown
  kept); `--lang` covers all seven.

### Rank backends (generic, local-first)

- `RankKind`: `jev` (hosted Nouls) / `jina` (local ONNX) / `llama`
  (llama.cpp `--rerank` server, e.g. bge-reranker-v2-m3 on Vulkan) —
  shared by CLI `--rank` and MCP `search_ranked rank`.
- `LlamaReranker` over verified `POST /v1/rerank`; blocking client kept
  off the async runtime. Load/endpoint failures fall back to retrieval
  order, never an error. Default stays retrieval order: no local ranker
  beats the gate yet (proven by test).
- Bake-offs on 10 keyword + 4 concept queries: Jina 0/4 concepts,
  bge-reranker 2/4 on R9700 Vulkan (~1.1 s/q), Laya judge 1/4 at 5–17 s/q
  on CPU (Run 13/14). CLM assessed on paper (Run 15): relative scores
  can't drive fail-closed gates; no NVIDIA path here.

### Output budget + machine surface

- Every chunk line carries `source=` (`bm25`/`vec`/`bm25+vec`/`rg`/`ast`),
  one-line crumbs, 1200-char text cap; pinned by test.
- MCP `format=json` envelope (`{"notes","hits"}`) on all tools;
  `rg` accepts a single file as root (ripgrep semantics).
- New tools: `context` (expand a citation to its enclosing symbol),
  `skill` (route tasks to on-disk SKILL.md libraries, Jev or lexical
  fallback, hardened front-matter parsing), `definition` unchanged.
- Freshness contracts: `index: stale` (mtime manifest + extractor
  version), `vectors: stale` marker (`watch` marks, `embed` clears);
  incremental embed re-embeds edited spans only.
- Serving log (`~/.one-grep/serving.log`, JSONL, emails/keys redacted,
  atomic appends, tests quarantined) + pi `tool_call` correlator for the
  hits→read join. P8 first cut: 874 real rows, 98.7% `rg`.
- Cursor `beforeReadFile` pilot hook (deny unbounded reads over 400
  lines) + range-read docs (no `peek` tool).
- Package version stamped into the MCP handshake; LICENSE file added;
  `one-grep --version` works.

### Fixes

- `query --rank llama` no longer panics (blocking client off-runtime).
- Serving-log interleaving under concurrency; test pollution of the prod log.
- `rg` file-as-root mapping confusion now resolves instead of erroring.
