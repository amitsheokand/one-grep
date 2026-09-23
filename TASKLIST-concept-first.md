# TASKLIST — concept-first one-grep

**Thesis.** Run 9 is the honest signal: Jina rerank is 10/10 on keywords
and 0/4 on paraphrases; hybrid loses to lexical on concepts (1/4 vs 2/4).
Agents ask intent questions, so the default path must win intent first.
No new surface until the default path stops hurting the case agents
actually care about.

## Phase 0 — Ship it like software (unblocks adoption)

| # | Packet | Status | Gate |
| --- | --- | --- | --- |
| 0.1 | `og-license` — add LICENSE file matching README (Apache-2.0); fix empty package description | **Done** (`LICENSE`, tag `v0.1.0`; Cargo description was already non-empty) | `LICENSE` present; `cargo metadata` description non-empty |
| 0.2 | `og-version` — clap `version` attr so `one-grep --version` prints the Cargo version; tag `v0.1.0` | **Done** | `--version` output == `git describe`; tag pushed |
| 0.3 | `og-agent-install` — 20-line README block: `cargo install --git …` or prebuilt binary + `install --target …` + first `index`/`embed` | **Done** | fresh-machine walkthrough in README, no skill folder needed |

## Phase 1 — Measure, then route (concept search before flags)

| # | Packet | Status | Gate |
| --- | --- | --- | --- |
| 1.1 | `og-eval-gate` — freeze a labeled set: keyword / paraphrase / symbol / call-chain, incl. one TS or monorepo fixture; report R@1/R@3 + latency p50/p95; CI refuses merge on concept R@3 drop (extends `src/eval.rs` `ideasearch-v1`, which is lexical-only today) | **Done** (`ideasearch-v2`: 12 cases, `src/lane.ts` exercises the window fallback; measured lexical paraphrase R@3 = 2/3, overall 9/10, pinned; negative control verified red) | `cargo test eval_gate` red on Run-9-style regression |
| 1.2 | `og-router` — one router for CLI and MCP: exact/`::`/quoted → `rg`; identifier-like → BM25; natural-language → hybrid + intent ranker (CLI `query` currently skips the `exact_pattern` routing MCP already has — close that parity hole first) | **Done** (`src/route.rs`: `Literal`/`Identifier`/`Intent`, explicit flags/`fts` force intent; CLI `query` routes exact→rg and identifier→BM25; MCP `search_ranked` keeps identifiers lexical pre-meter) | same routing table asserted for `query` and `search`; golden tests per class |
| 1.3 | `og-intent-ranker` — ranker trained or prompted on intent, not keyword overlap; candidates: LocalJev via `TYPESAFE_BASE_URL`, fine-tuned MiniLM (`ft2`), fusion-constant retune. Jina stays available, never default | **Done (first cut)** — hybrid MiniLM measured on the frozen set (Run 11): paraphrase 2/3 → 3/3, no regressions; LocalJev wire path proven by mock round-trip test; real-corpus concepts still open (Run 9: 1/4) | concept R@3 beats lexical on the frozen set; keyword R@3 does not regress |
| 1.4 | `og-local-default` — default rank path is local and offline-capable; hosted Jev stays opt-in (`--rank jev`, key chain unchanged). Default is whichever local path wins 1.1 — not Jina by decree, since Jina is also 0/4 on concepts | **Open** | `search_ranked` with no key and no network returns the gated local order |

## Phase 2 — Adoption tax (incremental embed, stale contract)

| # | Packet | Status | Gate |
| --- | --- | --- | --- |
| 2.1 | `og-incr-embed` — hash chunks, embed only new/changed windows, persist vectors beside tantivy; `watch` re-syncs vectors or marks them stale (today `watch` re-syncs the index only, so vectors silently rot) | **Open** | second `embed` on unchanged tree is near-noop (timed); `watch` + edit + `query --hybrid` reflects the edit |
| 2.2 | `og-stale-contract` — detect dirty trees (mtime vs index generation); `query` prints `index: stale` next to the existing `unindexed` fallback so agents never cite yesterday's chunks | **Open** | touch-a-file → stale note; re-index clears it |

## Phase 3 — Coverage honesty (parity + languages)

| # | Packet | Status | Gate |
| --- | --- | --- | --- |
| 3.1 | `og-mcp-parity` — expose `--lang` / `--glob` / `--json` on MCP `rg` **and** `search` (CLI grew filters; the MCP table still documents old params — agents cannot pass flags they cannot see) | **Open** | MCP schema exposes all three; `search` documents how `lang` applies to fused results |
| 3.2 | `og-langs` — add tree-sitter grammars for TS/Go/Java (the usual agent repos) or document the Rust/Python/Nix/Markdown hole in the first paragraph of the README | **Open** | either new `extract` arms with chunk tests, or the documented hole |

## Phase 4 — Output budget + structure boundary

| # | Packet | Status | Gate |
| --- | --- | --- | --- |
| 4.1 | `og-budget` — MCP returns max N chunks, each truncated, with score, `source=rg\|bm25\|vec`, one-line breadcrumb (JSON without a cap moves the dump, it does not save tokens) | **Open** | token-count test on a fixed fixture: capped < uncapped, winners retained |
| 4.2 | `og-astg-fuse` — do **not** reimplement rewrite rules: shell out to `ast-grep` when present and fuse its hits as a third RRF list. `--lang` stays the only overlap | **Open** | `ast-grep` absent → identical results; present → third list fused, never merged as rewrite |

## Must not

- Add CLI/MCP surface before Phase 1 gates pass.
- Default any ranker that loses to lexical on the frozen concept set.
- Reimplement ast-grep rules/rewrite inside one-grep.
- Auto-index as a side effect of search (TASKLIST-fused-rank law stands).
- Ship a network-dependent default path.

## Sources (public)

- `benchmarks/results.md` Run 8 (rerank 10/10 keyword, 0/4 concept) and Run 9 (re-run: same shape)
- https://github.com/githubnext/localjev (local Jev wire-compatible backend)
- https://github.com/razorback16/openjev (logit-read backend, NVIDIA)
- https://ast-grep.github.io/ (structure boundary: `--lang` overlap only)
- `TASKLIST-fused-rank.md` (retrieves-vs-judges split; still in force)
