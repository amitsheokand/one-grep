# one-grep

Local-first hybrid workspace search in Rust: exact `rg` + BM25 lexical index
(tantivy) + ONNX vector similarity with RRF fusion, exposed as a CLI and as an
in-tree MCP server.

Public repo: [github.com/amitsheokand/one-grep](https://github.com/amitsheokand/one-grep).
Crate/binary name: **`one-grep`**. License: Apache-2.0. MSRV: Rust 1.85.

## What it does

* **No-index exact search** (`rg`): gitignore-aware literal-or-regex search
  over files, walked on all cores with deterministic `(path, line)` order.
  Literal by default; one outer `"..."` / `'...'` pair is stripped so
  shell and MCP callers agree. `--lang` restricts to
  `rust`/`python`/`typescript`/`go`/`java`/`nix`/`markdown`
  (ast-grep-style); `--glob` adds
  include globs (`!` negates). `--json` emits a `[{path,line,text}]`
  array. Limit 1–500, default 100.
* **Indexed lexical search** (`index` + `query`): tree-sitter chunking
  (Rust/Python/TypeScript/Go/Java/Nix symbols, file-header comment blocks,
  Markdown sections, sliding windows, 2-hop call-chains) into a tantivy
  BM25 index under
  `<workspace>/.one-grep/`.
  Lexical window is 150 lines / 135 step; vector window is 50 / 40.
* **Hybrid retrieval** (`--hybrid`): MiniLM-class ONNX embeddings run
  in-process via fastembed, fused with BM25 by RRF (fetch depth 50/side,
  lexical 2x weight). Optional local cross-encoder rescore
  (`--rerank` / `--rank jina`, Jina v1-turbo top-20).
* **MCP server** (`serve`): 4 tools over stdio or `127.0.0.1:3210/mcp`
  (bearer token in `~/.one-grep/token`, mode 0600):
  `search`, `search_ranked`, `definition`, `rg`.
* **Agent wiring** (`install`, Nix module): idempotent upsert of the
  `one-grep` entry into 6 harnesses, preserving peer servers.
* **Maintenance**: `watch` (2s-debounce re-sync), `embed` (model sync),
  `dump-chunks` (JSONL for mining/eval), `eval` baseline.

## Measured numbers

Harness: `benchmarks/bench.py`, release binary, best-of-3 latency,
recall@3. Corpus Runs 1–8: `nixos-config` copy (13 MB, `.git` excluded);
Run 9: same repo grown to 22 MB / 217 files. `rg -l` file order is
nondeterministic, so its recall jitters ±1.
See `benchmarks/results.md` for full runs.

| Query set | one-grep lexical | one-grep hybrid (MiniLM) | `rg -l` | `zg query --human` |
| :--- | :--- | :--- | :--- | :--- |
| 8 keyword (Run 3) | 5.1 ms, 6/8 | 68.7 ms, **7/8** | 7.3 ms, 5/8 | 248.6 ms, 6/8 |
| 4 concept paraphrase (Run 3) | 5.4 ms, 1/4 | 69.7 ms, **3/4** | — | 249.8 ms, 1/4 |
| 10 keyword + chain (Run 8) | 5.4 ms, 6/10 | 99 ms, 7/10 | 5.4 ms, 7/10 | 251 ms, 8/10 |
| same 10, + Jina rerank top-20 | — | 602 ms, **10/10** | — | — |
| 10 keyword (Run 9, Linux re-run) | 7.2 ms, 7/10 | 207 ms, 8/10 | 2.1 ms, 8/10 | n/a (not installed) |
| 4 concept (Run 9) | 7.0 ms, 2/4 | 201.7 ms, 1/4 | — | n/a |

Model trade-off (Run 4, same corpus):

| Model | Params / dims | keyword R@3 | concept R@3 | query ms | embed time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `minilm` (default) | 22M / 384 | 7/8 | 3/4 | ~69 | ~14 s |
| `arctic-m` | 109M / 768 | 6/8 | 2/4 | ~216 | ~79 s |
| `gemma-300m` | 300M / 768 | not benched | not benched | ~500 (est. CPU) | — |

Fine-tune note (Run 10, 1111 mined triples, 889 train / 223 held):
pure-vector held R@1/R@3/R@10 base 0.37/0.52/0.66 → ft2 **0.52/0.67/0.78**.
Scale case: Hipfire Rust monorepo (127 MB, 1013 `.rs` files) indexes ~40k
chunks / 46k vectors.

Test suite: `cargo test --lib` — 80 passed, 0 failed (includes `rg`,
`index`, `fuse`, `vectors`, `mcp`, `lsp`, `eval`).

## Requirements

* Rust 1.85+ (`cargo build --release`), or Nix: `nix build .#one-grep`.
* ONNX Runtime: Nix build links system `onnxruntime`; non-Nix builds use
  the bundled fastembed runtime (first `embed` downloads the model to
  `~/.cache/one-grep/`).
* `rust-analyzer` on `PATH` only for the MCP `definition` tool.

## Quickstart

```bash
cargo build --release
./target/release/one-grep index <path>
./target/release/one-grep embed <path> [--model minilm|arctic-m|gemma-300m|<dir>]
./target/release/one-grep query "where is auth handled?" --path <path> [--limit 10] [--hybrid] [--rerank] [--rank jev|jina|llama] [--rank-endpoint http://127.0.0.1:8080] [--json]
# `search` is an alias of `query`
./target/release/one-grep rg "pattern" <path> [--regex] [--case-insensitive] [--lang rust|python|typescript|go|java|nix|markdown] [--glob '*.rs'] [--limit 100] [--json]
./target/release/one-grep watch <path>
./target/release/one-grep dump-chunks <path>
./target/release/one-grep serve --stdio        # or HTTP on 127.0.0.1:3210
./target/release/one-grep install --target opencode|cursor|pi|muse|hermes|command-code [--http --port 3210]
```

Notes:

* `query` without an index prints an `rg-fallback` live-grep result instead
  of failing.
* Rank backends (`--rank` / `search_ranked rank`): `jev` (hosted Nouls),
  `jina` (local ONNX cross-encoder, downloads on first use), `llama`
  (llama.cpp server with `--rerank`, e.g. `bge-reranker-v2-m3` on Vulkan;
  endpoint `ONE_GREP_RERANK_URL` or `--rank-endpoint`, default
  `http://127.0.0.1:8080`). Load or endpoint failures fall back to
  retrieval order, never an error.
* `rg` is literal unless `--regex`. `query --rank jev` needs
  `TYPESAFE_API_KEY` (else `~/.config/typesafe.env`), model
  `JEV_MCP_MODEL` default `jev-1.13.0`; without a key it emits a
  `rank: fallback` note.
* `install --http` is valid for `opencode` only; otherwise stdio entries
  using `~/.local/bin/one-grep` when present.

## Agent install (fresh machine, ~2 min)

```bash
cargo install --git https://github.com/amitsheokand/one-grep
one-grep --version                                   # expect one-grep 0.1.0
one-grep install --target opencode                   # or cursor|pi|muse|hermes|command-code
one-grep index ~/my-repo && one-grep embed ~/my-repo # first embed downloads MiniLM once
one-grep query "where is auth handled?" --path ~/my-repo --hybrid
```

No API key needed until `query --rank jev` / `search_ranked` with Jev;
without one they emit `rank: fallback` and keep retrieval order.

## MCP tools

Server instructions: prefer `search_ranked` for intent (retrieve + Jev
inside the tool, only top-k enters context), `search` for the raw fused
pool, `rg` for exact text/symbols/regex. Cite `path:line` evidence.

| Tool | Params | Returns |
| :--- | :--- | :--- |
| `search` | `root*`, `query*`, `fts?`, `fuse?=true`, `lang?`, `globs?`, `limit?=10` (1–50) | `path:start-end [breadcrumb] (score) source=bm25\|vec\|bm25+vec\|rg` chunks (text capped, 1-line crumb); `foo::Bar` and `"quoted"` / `'quoted'` route to exact `rg` unless `fts` is set; single tokens go BM25 |
| `search_ranked` | same as `search` | same pool rescored by Jev, top-k only, `rank: jev exists=…` or `rank: fallback (reason)` header |
| `definition` | `root*`, `path*`, `line*` (1-based), `character*` (1-based), `server?` (default `rust-analyzer`) | `path:start-end (workspace\|external)`; escapes rejected, missing server is an error |
| `rg` | `root*`, `pattern*`, `regex?=false`, `structural?=false`, `case_insensitive?=false`, `lang?`, `globs?`, `format?=text`, `limit?=100` (1–500) | `path:line:text` lines (or `{"notes","hits"}` with `format=json`), gitignore-aware |

## Install targets

| Target | Config file |
| :--- | :--- |
| `opencode` | `~/.config/opencode/opencode.json` (`--http` optional) |
| `cursor` | `~/.cursor/mcp.json` |
| `pi` | `~/.pi/agent/mcp.json` |
| `muse` | `~/.config/muse/settings.json` |
| `hermes` | `~/.hermes/config.yaml` |
| `command-code` | `~/.commandcode/mcp.json` (`commandcode`, `command_code` aliases accepted) |

Nix alternative: `nix build github:amitsheokand/one-grep` or import
`homeManagerModules.one-grep` (`programs.one-grep.mcp.<target>.enable`).
Details: [INTEGRATION.md](INTEGRATION.md), [nix/README.md](nix/README.md).

How it is wired on a Pi / Home Manager laptop:
[nixos-config `docs/coding-agent-stack.md`](https://github.com/amitsheokand/nixos-config/blob/main/docs/coding-agent-stack.md).

## Privacy

* Embeddings run locally (fastembed ONNX); no embedding API is called.
* `serve` HTTP binds `127.0.0.1` only and requires
  `Authorization: Bearer <token>`.
* Only `search_ranked` / `query --rank jev` makes a network call, to the
  configured Jev model; the key is never logged.

## Alternatives

Honest substitutes, depending on which half of one-grep you need:

* **Exact text search**: [ripgrep](https://github.com/BurntSushi/ripgrep) —
  the baseline. one-grep's `rg` will never beat it on raw latency
  (Run 9: 2.1 ms vs 7.2 ms); use `rg` when you know the literal text.
* **Structural search**: [ast-grep](https://github.com/ast-grep/ast-grep)
  (Rust, tree-sitter) — matches AST shape instead of text
  (`$A && $A()` patterns, rewrite rules). one-grep does not reimplement
  this: `rg --structural -p 'pattern' --lang rust` and
  `query --hybrid --ast 'pattern' --ast-lang rust` shell out to the
  `ast-grep` binary when present (fused as a third RRF list, tagged
  `source=ast`) and fail closed when absent.
* **Hybrid codebase search**: `zg` (`zvec-grep`, Node/TypeScript cousin)
  — same BM25+vector idea, ~250 ms
  per query in our bench vs ~70–200 ms for one-grep hybrid. one-grep is
  the faster native port with the in-tree MCP server.
* **Structural search over MCP**: [ast-grep-mcp](https://github.com/ast-grep/ast-grep-mcp)
  — the ast-grep team's own MCP server. Use it side-by-side when agents
  need deep structural rules; use one-grep's `search` for intent and
  `rg --structural` for one-off shape queries.
* **Combined engines**: [`ox-core`](https://crates.io/crates/ox-core)
  (`ox-codes`) — ripgrep + tree-sitter + ast-grep as an HTTP service
  with rewrite and dataflow analysis. Heavier than one-grep; pick it
  when you need rewrite/codemod, not just retrieval.
* **Jev ranking backend**: the default is the hosted
  [TypeSafe Jev API](https://typesafe.ai/) (`TYPESAFE_API_KEY`). Local
  options speak the same `POST /v1/systemone` wire protocol, selectable
  via `TYPESAFE_BASE_URL`:
  * [LocalJev](https://github.com/githubnext/localjev) — Bun +
    DiffusionGemma through an OpenAI-compatible endpoint (oMLX).
    Wire-compatible, but probabilities are prompted/self-reported rather
    than logit-read, so check calibration on your workload before
    trusting low-`exists` bands.
  * [OpenJev](https://github.com/razorback16/openjev) — patched vLLM
    backend with a structured logit read; needs NVIDIA hardware.
* **Local rerank without Jev at all**: `query --rerank` / `--rank jina` / `--rank llama` (llama.cpp `--rerank` server)
  rescores with an on-device Jina cross-encoder — no key, no network,
  ~1.1 s per query. In our bench it sweeps keywords (10/10) but adds
  nothing on paraphrased concepts (0/4).

## Docs

* Architecture, CLI table, harness JSON shapes: [INTEGRATION.md](INTEGRATION.md)
* Nix module/overlay: [nix/README.md](nix/README.md)
* Benchmarks, ablations, fine-tune: [benchmarks/results.md](benchmarks/results.md)
