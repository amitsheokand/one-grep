# TASKLIST — fused retrieve + rank

**Thesis.** Retrieval is cheap and local. Judgment is cheap at the API
and expensive in the agent window. Rank inside the tool so candidate
pools never enter context — only winners do.

## Tool-use law

| Job | Tool | Not |
| --- | --- | --- |
| Exact symbol, literal, `foo::Bar`, `"quoted"` | `rg` | ranking |
| Intent / where / how | `search_ranked` (MCP) or `query --rank jev` | dumping the fused pool into chat |
| Raw fused shortlist (debug) | `search` / `query --hybrid` | default agent path |
| Rank a list the agent already has | `jev_find` MCP | one-grep |
| Screen fetched text | `jev_screen` MCP | one-grep |

one-grep **retrieves**. Jev **judges a closed shortlist**. Jev does not
search the tree, generate text, count, or do math.

## Economy

- Judgment at the tool boundary. One TypeSafe call: shared state + one
  Noul per candidate + an `exists` Noul (does any candidate answer).
- Return top-k only. Prefix `rank: jev exists=…` or `rank: fallback (…)`.
- Missing/invalid/unreachable Jev → original retrieval order, never an
  empty success that pretends the tree has no match.
- Do not auto-index. Unindexed trees stay on live `rg`, then optional Jev.
- Never print, log, or commit `TYPESAFE_API_KEY`. Errors name sources,
  not values.

Key chain: process env `TYPESAFE_API_KEY`, then `~/.config/typesafe.env`,
then `~/.config/environment.d/60-typesafe.conf`. Model pin: `JEV_MCP_MODEL`
(default `jev-1.13.0`).

## Packets

| # | Packet | Status | Gate |
| --- | --- | --- | --- |
| 1 | `og-jev-client` — key resolve + System One POST, no key leakage | **Done** | unit tests; error strings omit values |
| 2 | `og-rank-cli` — `query --rank jev` (`search` alias); keep `--rerank` as local Jina | **Done** | unindexed + indexed paths; fallback on missing key |
| 3 | `og-search-ranked-mcp` — MCP `search_ranked`; pool stays inside the tool | **Done** | top-k + rank header; `search` still returns the fused pool |
| 4 | `og-rules` — INTEGRATION.md + MCP instructions: prefer `search_ranked` for intent | **Done** | no “dump the pool then rank in chat” guidance |
| 5 | `og-release` — tag/release so consumers can bump the package | **Open** | GitHub tag; this repo does not vendor consumer configs |

## Must not

- Depend on private crates or product trees.
- Call Jev over the whole workspace.
- Replace exact `rg` lookup with ranking.
- Auto-index as a side effect of search.

## Sources (public)

- https://docs.typesafe.ai/cookbooks/rerank_typesafe.md
- https://docs.typesafe.ai/api.md
- https://docs.typesafe.ai/confidence.md
- https://docs.typesafe.ai/model-jaggedness/jev-1.13.md
