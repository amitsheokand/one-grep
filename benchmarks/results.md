# Bench: one-grep vs rg vs zg

Corpus: `nixos-config` copy (13 MB, `.git` excluded). Release binary,
best-of-3 latency, recall@3. Harness `benchmarks/bench.py`.
Caveat: `rg -l` file order is nondeterministic, so its recall@3 jitters
±1 across runs.

## Run 1 (2026-09-02): baseline

8 keyword queries.

| system | mean ms | recall@3 |
|---|---|---|
| rg -l | 7.3 | 7/8 |
| one-grep lexical (tantivy BM25) | 5.3 | 6/8 |
| one-grep hybrid (MiniLM + RRF k=60) | 67.4 | 4/8 |
| zg 0.2.1 (`query --human`) | 253.6 | 6/8 |

Index: one-grep 0.99 s, embed 12.55 s.

## Run 2: what fixed hybrid

Suspects were small vector windows + shallow RRF fetch (`limit*2`).
Ablation result: shrinking vector windows to 50/40 **alone dropped**
keyword hybrid to 2/8. Root cause: lexical index (150-line windows)
and vector store (50-line windows) produced different chunk keys, so
RRF dual-list accumulation never fired — pure noise addition.

Fix: keep tight vector windows, fuse by **line-overlap mapping**
(vector hit credits the most-overlapping same-file lexical chunk;
vector-only regions become own entries), fetch depth 50/side.

## Run 3 (2026-09-02): after fix, + 4 paraphrased concept queries

| system | mean ms | recall@3 |
|---|---|---|
| rg -l | 7.3 | 5/8 |
| one-grep lexical | 5.1 | 6/8 |
| one-grep hybrid | 68.7 | **7/8** |
| zg | 248.6 | 6/8 |

| concept set (4) | mean ms | recall@3 |
|---|---|---|
| one-grep lexical | 5.4 | 1/4 |
| one-grep hybrid | 69.7 | **3/4** |
| zg | 249.8 | 1/4 |

Hybrid-only wins: "where do I put tokens so they never get committed"
→ home-manager.nix secrets section; "make the terminal prompt show git
status with pretty colors" → powerlevel10k (run 2). Per-query cost:
hybrid ~69 ms (MiniLM-L6 query embed) vs zg ~249 ms.

## Run 4 (2026-09-02): model trade-off (MiniLM vs arctic-m)

`embed --model minilm|arctic-m|gemma-300m`; store auto-resets on switch;
`--hybrid` auto-loads the stored model. Arctic **requires** its BGE-style
query prefix (`Represent this sentence...`); without it keyword hybrid
was 4/8, with it 6/8.

| model | kw recall@3 | concept recall@3 | query ms | embed s |
|---|---|---|---|---|
| minilm (22M, 384d) | 7/8 | 3/4 | ~69 | ~14 |
| arctic-m (109M, 768d) | 6/8 | 2/4 | ~216 | ~79 |

Verdict: MiniLM stays default — better recall here at 3x less latency.
Arctic's MTEB pedigree did not transfer (matches the opencode-memory
eval's warning that pedigree ≠ corpus fit). gemma-300m wired but not
benched (expect ~500 ms/query CPU; try only if quality stalls).
Qwen3-0.6B unavailable in fastembed-rs 6.0.2 enum (573 MB, CPU-slow
per community reports) — revisit via `try_new_from_user_defined` if
a quality ceiling is hit.

## Run 5 (2026-09-02): fusion discipline + hipfire real-world

Hipfire R9700 (Rust monorepo, 127 MB, 1013 `.rs` files) exposed the
failure mode: with 46k chunks, unweighted RRF + stacked vector credits
let vector noise bury lexical rank-1 (BM25 29.97 → absent from hybrid
top-10). Fixes:

- One vector credit per fused slot (overlapping windows are redundant
  votes, not independent evidence).
- Lexical weight 2x in RRF; ties break toward lexical rank.
- Fetch stays 50/side.

Nixos rerun: keyword hybrid 7/8 held; concept hybrid 1/4 (down from 3/4
— the stacked-credit wins for "pretty prompt"/"no secrets" no longer
inflate; zg also scores 1/4 on this set, all systems ≤1). Deliberately
not chased further: tuning fusion constants against 12 queries is
overfit territory. The discipline (exact beats fuzzy, semantics
rescores) is the principled choice at 46k-chunk scale.

Hipfire qualitative (8 queries, hybrid): SERVE.md, QUANTIZE.md,
reap README+plan.rs, redline_vs_hip.md all top-3 — docs surface for
concept queries as designed. Symbol-heavy code queries still favor
lexical; hybrid matches it except one case (fused_qkv code symbols).

## Perf notes (real-world scale)

- `embed` was 5 chunks/s: batch-128 padded every batch to the longest
  4000-char sequence. Batch 16 + truncate 1500 chars → 34/s.
- Added: default excludes (lockfiles, `target/`, `node_modules/`,
  `.gguf/.jsonl`/images), >8k-char-line skip (generated/data),
  embed checkpoints every 10 batches, stderr progress rate.
- Release profile switched to thin LTO (fat LTO relinked ~7 min).
- Hipfire full: index 40k chunks, embed 46k vectors.

## Run 6 (2026-09-02): cross-encoder rerank (Jina v1-turbo, top-20)

`query --rerank` rescores fused top-20, keeps tail order.

| keyword (8) | ms | recall@3 |
|---|---|---|
| hybrid | 78 | 7/8 |
| **rerank** | 612 | **5/8** |

| concept (4) | ms | recall@3 |
|---|---|---|
| hybrid | 79 | 1/4 |
| **rerank** | 610 | **0/4** |

Verdict: net negative, kept behind flag (default off). Mechanism:
rerank docs are head-truncated 150-line windows, so the answer text
is often past the cut (launchd config sits mid-window; cross-encoder
never sees the term). It demoted 3 true hits fusion had right, all
scores negative. Rerank needs focused candidates first — same
section/symbol chunking fix as below. Individual wins exist
(p10k.zsh for prompt query, agent-profiles for lanes) but the
file-exact metric and the head-truncation loss dominate.

## Run 7 (2026-09-02): nix symbols + comment attach + noise filter

tree-sitter-nix `binding` chunks with `a > b.c` breadcrumbs, leading
`#` comments attached (rustdoc-style), bare single-line nested
bindings dropped (container covers them).

| keyword (8) | ms | recall@3 |
|---|---|---|
| lex | 5.6 | 5/8 |
| hybrid | 100 | 5/8 |
| **rerank** | 623 | **8/8** |
| zg | 291 | 7/8 |

| concept (4) | ms | recall@3 |
|---|---|---|
| lex | 5.9 | 2/4 |
| hybrid | 106 | 1/4 |
| rerank | 701 | 0/4 |
| zg | 290 | 1/4 |

Trade accepted: symbols split BM25 term co-occurrence (lex 7→5/8 on
prose-heavy config; e.g. launchd chunk outranks mlx-mac.nix for "mlx
server"), but focused candidates fixed rerank (5→8/8, best overall).
Primary interfaces (hybrid/rerank, agentic) win; lex-only and rg
remain for exact lookup. Concepts still weak everywhere (all ≤2/4) —
needs corpus strategy beyond ranking constants; parked.

## Run 8 (2026-09-02): stolen ideas — stratification, split, chains

Paper transfers applied:

- **Difficulty levels + dev/held split** (ICD-Bench method): queries
  labeled L1 (exact term) / L2 (partial) / L3 (paraphrase); dev tunes,
  held verifies. Harness `SPLIT=dev|held|all`.
- **Call-chain chunks** (compositional primitives): tree-sitter call
  extraction (Rust `call_expression`, Python `call`, incl. `attribute`
  receivers), same-file-first else unique-global resolution, ambiguous
  names skipped. Breadcrumb `caller > calls > callee`, kind `chain`,
  shared IDs across lexical/vector stores. Index rebuilds chains on
  any change (callee-text staleness); vectors prune via id-set.

| keyword (10, +2 chain Qs) | ms | recall@3 |
|---|---|---|
| rg | 5.4 | 7/10 |
| lex | 5.4 | 6/10 |
| hybrid | 99 | 7/10 |
| **rerank** | 602 | **10/10** |
| zg | 251 | 8/10 |

Held-out only: rerank 5/5, hybrid 4/5, zg 4/5 — tuning generalizes,
no overfit signal. Chain queries hit via `calls` breadcrumbs live
(`apply_request > calls > apply_lane_defaults` rank 2 pre-rerank).

## Next

- Qwen3-0.6B via `try_new_from_user_defined` if quality ceiling hit.
- Phase A pair mining (done): `dump-chunks` + `benchmarks/mine.py`
  synthesize (query, chunk, BM25-hard-negatives) with the mlx compact
  lane. 186 triples (md/nix/py/shell mix, avg 12-word queries, 4.6
  negs each). Baseline MiniLM hybrid on synthetic set: R@1 0.47,
  R@3 0.71, R@10 0.86 — real headroom, valid train+eval set.
  Next: Phase B fine-tune (sentence-transformers, MNR loss, ONNX
  export via existing user-defined path).

## Run 10 — Phase B at volume: 1111 triples (2026-09-02)

Added tokio (325) + windows-rs (600, 607k-chunk pool) + fresh nixos
(186, self-contained pos_text). 889 train / 223 held. Same recipe.

Pure-vector held (17k capped pool): base 0.37/0.52/0.66 →
**ft2 0.52/0.67/0.78** (+15pp R@1/R@3, ±3pp noise → conclusive).
Exported single-file 87 MB ONNX; `embed --model` verified live.
Lesson: volume + hard negatives beat model size; MiniLM stays.

148 train / 38 held triples. MNRL, 10 epochs, 35 s on M4 CPU.
Manual torch.onnx export (optimum/transformers clash) merged to
single-file 87 MB ONNX; `embed --model <dir>` via
`try_new_from_user_defined` (mean pooling, dims auto-probed).

Pure-vector held: base R@3 0.58 → custom **0.71** (+13pp).
Hybrid held: base 0.42/0.63/0.82 → custom **0.45/0.68/0.82**.
Direction consistent, n=38 noisy (±8pp). Next: scale pairs with
windows-rs + big Rust crates before claiming the win.
- Training corpora beyond nixos/hipfire: windows-rs + other big Rust
  crates (symbol-dense, Apache/MIT) for pair mining volume.

## Changelog (post-bench)

- `watch <path>`: foreground notify-based re-sync, 2 s debounce,
  skips index/build/vcs dirs. Live-verified (burst → single sync).
- Harness `rg -l` now `--sort path` (deterministic recall).

## Run 12 — why real concepts fail (2026-09-24, research, no code)

Question: Run 9 shows hybrid 1/4 vs lexical 2/4 on nixos concepts while
Run 11 (toy fixture) shows hybrid sweeping paraphrase 3/3. Diagnosed per
query on the Run 9 workspace (index + MiniLM vectors intact), top-10 and
top-50 both backends, `--json` ranks inspected.

**pretty prompt** (lex hit@3, hyb miss@3): not a retrieval failure. The
true chunk (home-manager.nix:318) sits at lex=3, vec=29 → fused #6 at
0.0430, behind AGENTS.md (lex=7, vec=12) at 0.0437. With RRF k=60,
vector ranks 12→29 differ by 0.003 — the fusion cannot separate them.
Mechanism: **RRF score compression**. Direction: smaller k (spread) or
score normalization, judged by the v2 gate + bench.

**never both** (all miss@50): the answer file (mlx-lane.nix) is absent
from both top-50s. Its discriminating word ("Exclusive", line-1 header
comment) is **not in any chunk**: `nix_symbols` attaches leading `#`
comments to bindings only, so file-header prose vanishes. BM25 is then
correct given its index — the corpus lies by omission. Direction: emit
the file-header comment block as its own chunk (small, gate-testable).

**no secrets** (all miss@50): the answer ("Local secrets (gitignored,
never committed)") drowns inside a 130-line `zsh > initContent` symbol
chunk (lines 30–160, ~4 KB). Length-norm dilution buries it under
tighter false friends (`estimate_tokens`). Same file also yields two
overlapping chunks (14–161 and 30–160) — redundant slots. Direction:
cap symbol chunk size (split with breadcrumb inheritance) and prefer
non-overlapping coverage. Chunk audit: longest index chunks run
130–253 lines (test files, flake outputs).

**tokens sense ambiguity** (no secrets, second-order): even retrieved,
"tokens" matches LLM-token code (`estimate_tokens`), not API secrets.
No ranking constant fixes word sense. Direction: semantic judge
experiment — LocalJev rerank over the fused shortlist on the 4 real
concepts (wiring already proven by mock test; needs a local endpoint).

Not pursued: tuning fusion constants against 4 queries (overfit
territory, per Run 5 discipline). Each direction above gets its own
gate check (v2 paraphrase floor + Run 9 re-run) before merging.

## Run 12 fix verified — header chunks (2026-09-24)

Implemented: leading `#` / `//` comment blocks become `(header)`
`Section` chunks (capped at 30 lines; Markdown excluded, shebangs
skipped), plus an extractor version marker (`.one-grep/extract.version`)
so chunking upgrades force a full re-extract — old indexes report
`index: stale` instead of silently missing new chunk types. Chunk IDs of
unchanged content are stable, so vectors stay incremental.

Live on the Run 9 workspace: reindex auto-triggered by the version
mismatch (217 upserted, 2929 → 3090 chunks, +5.5%); "never both" went
from absent-in-top-50 to lexical #4 (`mlx-lane.nix:1-5 [(header)]`) and
hybrid #6. Incremental embed for the 184 new chunks took seconds
(20/s), not minutes. Recall@3 still misses (#4, behind test-file false
friends) — the corpus gap is closed; what remains is ranking, for the
RRF-spread experiment next.

## P8 — serving-side numbers, first cut (2026-09-24)

`~/.one-grep/serving.log`, 1232 rows: 6 corrupt (0.5%, concurrent-append
interleave — fixed: single-syscall appends + mutex, proven by
`serving_log_concurrent_appends_stay_whole`), 350 test-pollution rows
(/tmp roots — fixed: test builds stay out unless the env var is set),
874 real rows. Real mix: **rg 863 (98.7%), search_ranked 9, skill 2,
search 0, context 0**. Output p50 155 chars, p95 4650; latency sub-ms
across the board. Reads side: pi correlator log absent — no pi sessions
have run with the extension yet (pi needs a restart to load new
extensions), so "did they still read" stays open; the DDE 43-vs-14
stands as the only reads datum.

Two readings: (1) agents reach for `rg`, not `search_ranked` — the
"make ranked the default" advice has an adoption problem confirmed by
data, not just taste; (2) output sizes are small (p95 4.6 KB), so the
read problem really is *count × re-reads across turns*, not single
dumps — consistent with the turns×context cost model.

## Run 13 note — hits→read instrumentation (2026-09-24)

Serving side: every MCP call appends `{ts, tool, root, query, hits,
chars, notes, latency_ms}` to `~/.one-grep/serving.log`. Harness side:
`~/.pi/agent/extensions/one-grep-hits.ts` logs one-grep search calls and
`read` calls (pre-call hooks only — no result hook assumed) to
`~/.pi/agent/obs/one-grep-hits.log`; cited paths come from the serving
log, joined offline. First metric to watch: searches after which no
full-file read follows.

## Run 13 — bge-reranker-v2-m3 on R9700 Vulkan as `--rank llama` (2026-09-24)

llama-server built from source with `-DGGML_VULKAN=ON` (nix shell:
cmake, ninja, gcc, gnumake, vulkan-headers, vulkan-loader, shaderc,
spirv-headers, pkg-config; needed explicit `-DVulkan_*` paths plus
`-isystem` spirv-headers for `spirv.hpp`). Both GPUs visible
(iGPU RAPHAEL + R9700 GFX1201); serving on `--device Vulkan1`.
Model: `gpustack/bge-reranker-v2-m3-GGUF` Q8_0 (636 MB) in
`~/work/models`, served with `--embedding --pooling rank`. 1-doc
rerank: 0.77s CPU → 0.16s Vulkan. one-grep side needed one fix: the
blocking reqwest client cannot even be *built* on the tokio runtime —
`query --rank llama` now builds + runs inside `spawn_blocking`.

Bake-off (same 10 keyword + 4 concept, `query --rank llama`, ~1.1 s/q):

| set | lex | hyb | jina (Run 9) | **llama (Vulkan)** |
|---|---|---|---|---|
| keyword (10) | 7/10 | 8/10 | — | **9/10** |
| concept (4) | 2/4 | 1/4 | 0/4 | **2/4** |

Per-query concepts: never-both:H (header chunk + cross-encoder agree),
pretty-prompt:M, save-session:H, no-secrets:M. Keyword: only
chain-apply misses (reranker prefers a *test* calling `apply_request`
over its definition — debatable, metric says miss).

Reads: cross-encoders beat Jina everywhere and tie lexical on concepts,
but `estimate_tokens` still wins no-secrets (scores go negative: the
model finds nothing fitting). Word sense needs a Noul judge, not a
stronger ranker — Laya comparison stays open. R9700 Vulkan serving is
proven for this path; CPU needs ~10 tok/s (≈8 min per 20-doc call),
hence the 90 s client budget is Vulkan-first.

## Run 14 — Laya as the Noul judge via the SystemOne seam (2026-09-24)

Laya 0.3.11 (`pip install "laya[serve]"`, torch CPU — needed
`LD_LIBRARY_PATH` to a nix gcc lib for `libstdc++.so.6`), served with
`LAYA_PORT=8001 LAYA_DEVICE=cpu LAYA_MODELS=english LAYA_PRELOAD=1`.
one-grep pointed unmodified at it: `TYPESAFE_BASE_URL`,
`TYPESAFE_API_KEY=local`, `JEV_MCP_MODEL=english`. Same 10+4, `query
--rank jev`:

| set | lex | hyb | llama (Run 13) | **laya (CPU)** |
|---|---|---|---|---|
| keyword (10) | 7/10 | 8/10 | 9/10 | **8/10** |
| concept (4) | 2/4 | 1/4 | 2/4 | **1/4** |

Laya rerank is byte-identical in outcome to hybrid on all 14: the
server runs (5–17 s/query on CPU — long chunk states in one forward
pass) and scores honestly (`rank: jev exists=0.68` on no-secrets), but
the false friends score ~0.55–0.58 instead of ~0.1, so order never
changes. Two readings, both useful: (1) our Noul question template
("states or implements what the query asks for") is too blunt to force
sense disambiguation — the judge is only as sharp as its criteria;
(2) `exists=0.68` with no true answer in the pool is overconfident,
matching the calibration caution in Laya's own docs. Next levers in
order: sharper per-query criteria, then a bigger shortlist (limit 8–10
gives the judge more to choose from), then ROCm torch (no override
hacks needed on this GPU per operator note — untried).

## Run 12 find — duplicate chain chunks stacked RRF terms (2026-09-24)

While testing RRF spread, one fused slot showed score 0.1464 for
lex=20/vec=None — impossible under single-count RRF (2×rrf(20) = 0.05
at k=20). The lexical list contained the same
(path, span, breadcrumb) chunk 3×: `chains::extract_workspace` emits one
chunk per call site, so 3 calls to `apply_lane_defaults` inside
`apply_request` produced 3 byte-identical chain docs, and fusion added
an RRF term per copy (triple lexical credit). Every prior number on a
re-synced workspace carries this distortion. Fixed: dedupe identical
chains at extraction (`repeated_call_sites_emit_one_chain`), extractor
version 3. Lesson: Run 5's one-credit discipline applies to duplicate
docs as well as overlapping windows.

## Run 12 experiment — RRF k=60 vs k=20 on clean data (2026-09-24)

Invalidated once (above), then re-run on a fresh workspace (217 files,
header chunks + splits + dedupe in, full embed). Single-pass recall
(deterministic, no sampling noise):

| backend | k=60 keyword | k=60 concept | k=20 keyword | k=20 concept |
|---|---|---|---|---|
| lexical | 7/10 | 2/4 | 7/10 | 2/4 |
| hybrid | 8/10 | 1/4 | **9/10** | 1/4 |

k=20 flips `stop gemma` (mechanism checked: spread separates vec-4/6
from vec-20+ instead of compressing them), regresses nothing, leaves
lexical untouched by construction. Kept: short candidate lists
(fetch ≤ 500, effective top-50) need less smoothing than TREC-scale
full rankings, and the v2 gate still holds. The motivating
pretty-prompt case did not move (vec-29 stays buried) — that one needs
the semantic judge, not more spread.

## Run 12 negative — term-overlap routing (2026-09-24)

Hypothesis: route NL queries by distinctive-term coverage (fraction of
non-stopword terms present anywhere in the index) — high overlap →
lexical, low overlap → hybrid. Dry-run on all 14 bench queries:
misses measure 1.00 coverage (mlx server, rust setup, ssh hosts, never
both, no secrets), hits range 0.80–1.00. No separation — presence is
near-universal because BM25 always finds *something*; the question is
whether the *right file* ranks, which presence cannot see. Rejected
without code churn (one `/tmp` script, since removed). Score margins
and lex/vec top-1 agreement were eyed next but also fail clean
separation on n=14; per Run 5 discipline, no threshold gets picked
from this sample. NL stays hybrid-routed; the outstanding lever
remains the judge (Laya bake-off).

## Run 12 fix verified — symbol size cap (2026-09-24)

Implemented: `Symbol` chunks spanning >80 lines split into overlapping
parts (80/65, breadcrumb `parent [i/n]`); sections stay whole;
extractor version bumped to 2 (auto full re-extract, stable IDs keep
vectors incremental). Live: the 130-line `initContent` is now
`zsh > initContent [1/2]` (30–109) + `[2/2]` (95–160); the distinctive
sub-query "never committed" surfaces the secrets answer at #8
(previously buried past top-30 inside the giant chunk). The full NL
query still drowns in `tokens` TF-noise (`estimate_tokens`) — dilution
fixed, sense disambiguation outstanding (LocalJev experiment). "never
both" holds #4 (no regression). v2 gate untouched by construction
(fixture files are tiny; floors re-verified green).

## Run 11 — intent paths on the frozen v2 fixture (2026-09-24)

`ideasearch-v2`: 12 cases (3 keyword / 3 paraphrase / 2 symbol /
2 chain / 2 unanswerable) over a 6-file synthetic fixture incl. one
TypeScript file. Lexical via `eval::evaluate`, hybrid MiniLM via the
ignored `measure_hybrid_v2_reports` test (vectors synced in tmpdir).

| backend | R@1 | R@3 | keyword | paraphrase | symbol | chain | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| lexical BM25 | 0.80 | 0.90 | 1.0 | 0.667 | 1.0 | 1.0 | — | — |
| hybrid MiniLM | 0.80 | **1.00** | 1.0 | **1.0** | 1.0 | 1.0 | 6.0 ms | 9.7 ms |

Reads: hybrid fixes exactly the pinned weak spot (the third paraphrase)
with no keyword/symbol/chain regression — the 1.3 gate as specified.
Caveat, stated plainly: this is a 10-answerable toy. Real-corpus Run 9
says the opposite on nixos concepts (hybrid 1/4 vs lexical 2/4). So the
claim is narrow: MiniLM bridges short-distance paraphrase
("password"→"credentials"); genuine intent ("where do build outputs go"
at repo scale) still needs more. Next candidate: LocalJev as the ranker
— its wire path (`TYPESAFE_BASE_URL` → any SystemOne endpoint) is now
covered by a std-only mock round-trip test in `src/jev.rs`, no network.

## Run 9 (2026-09-24): re-run on Linux after quote + plant changes

Same harness (`benchmarks/bench.py`, best-of-3, recall@3), fresh release
binary with current code. Corpus: same `nixos-config`, now 22M / 217 files
(2929 chunks fresh-indexed). Index 0.19 s; `embed --model minilm` ~4.5 min
for 2868 chunks (~11/s). `zg` not installed on this box — zg columns
skipped. Machine: 60 GB Linux box under concurrent build load, so absolute
latencies run higher than the September Mac runs; relative order holds.

| keyword (10) | mean ms | recall@3 |
|---|---|---|
| rg -l | 2.1 | 8/10 |
| one-grep lexical | 7.2 | 7/10 |
| one-grep hybrid (MiniLM) | 207.1 | 8/10 |
| one-grep rerank (Jina top-20) | 1199.2 | **10/10** |

| concept (4) | mean ms | recall@3 |
|---|---|---|
| one-grep lexical | 7.0 | 2/4 |
| one-grep hybrid | 201.7 | 1/4 |
| one-grep rerank | 1082.3 | 0/4 |

Reads: rerank still sweeps keywords (10/10) and still adds nothing on
concepts (0/4) — same shape as Run 8, one machine later. Hybrid beats
lexical on keywords (8/10 vs 7/10) and trails it on concepts (1/4 vs
2/4). Hybrid latency here (~200 ms vs ~69–99 ms in September) is box
noise (loaded CPU, cold caches), not a code regression: the retrieval
path is unchanged since Run 8; intervening commits touched CLI parsing,
quote normalization in `rg`, and Jev note text only.
