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
