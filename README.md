# one-grep

Local-first hybrid workspace search in Rust. It combines exact `rg`,
a BM25 lexical index (tantivy), and ONNX vector similarity with RRF
fusion. It ships as a CLI and an in-tree MCP server.

Public repo: [github.com/amitsheokand/one-grep](https://github.com/amitsheokand/one-grep).
Crate/binary name: **`one-grep`**. License: Apache-2.0. MSRV: Rust 1.85.

## What it does

* **No-index exact search** (`rg`): searches files with literal-or-regex
  text matching. It respects `.gitignore`. It walks all cores and returns
  hits in deterministic `(path, line)` order. Literal matching is the
  default. The search strips one outer `"..."` / `'...'` pair, so shell
  and MCP callers agree. `--lang` restricts matches to
  `rust`/`python`/`typescript`/`go`/`java`/`nix`/`markdown`
  (ast-grep-style). `--glob` adds
  include globs (`!` negates). `--json` emits a `[{path,line,text}]`
  array. Limit is 1–500. Default is 100.
* **Indexed lexical search** (`index` + `query`): tree-sitter splits code
  into chunks (symbols in Rust/Python/TypeScript/Go/Java/Nix, file-header
  comment blocks, Markdown sections, sliding windows, 2-hop call-chains).
  The chunks land in a tantivy BM25 index under
  `<workspace>/.one-grep/`.
  Lexical windows cover 150 lines with a 135-line step. Vector windows
  cover 50 lines with a 40-line step.
* **Hybrid retrieval** (`--hybrid`): MiniLM-class ONNX embeddings run
  in-process through fastembed. RRF fuses them with BM25 (fetch depth
  50 per side, lexical weight 2x). An optional local cross-encoder
  rescores the top results
  (`--rerank` / `--rank jina`, Jina v1-turbo top-20).
* **MCP server** (`serve`): 6 tools over stdio or HTTP on
  `127.0.0.1:3210`. HTTP requires a bearer token
  (`~/.one-grep/token`, mode 0600).
  Tools: `search`, `search_ranked`, `definition`, `rg`, `skill`, `context`.
* **Agent wiring** (`install`, Nix module): `install` upserts the
  `one-grep` entry into 6 harnesses. It preserves peer servers. The
  upsert is idempotent.
* **Maintenance**: `watch` re-syncs with a 2s debounce. `embed` syncs
  models. `dump-chunks` emits JSONL for mining and eval. `eval` runs
  the baseline.

## Measured numbers

Harness: `benchmarks/bench.py`, release binary, best-of-3 latency,
recall@3. Corpus Runs 1–8 use a `nixos-config` copy (13 MB, `.git`
excluded). Run 9 uses the same repo grown to 22 MB / 217 files.
`rg -l` file order is nondeterministic, so its recall jitters ±1.
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
At scale, the Hipfire Rust monorepo (127 MB, 1013 `.rs` files) indexes
~40k chunks and 46k vectors.

Test suite: `cargo test --lib` — 147 passed, 0 failed (covers retrieval,
ranking, MCP tools, and eval).

## Requirements

* Rust 1.85+ (`cargo build --release`), or Nix: `nix build .#one-grep`.
* ONNX Runtime: Nix builds link the system onnxruntime. Other builds use
  the bundled fastembed runtime. The first `embed` downloads the model
  to `~/.cache/one-grep/`.
* Only the MCP `definition` tool needs `rust-analyzer` on `PATH`.

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
./target/release/one-grep context <file> <line> [--json]
./target/release/one-grep skill "task description" [--dir ~/.local/share/agent-skills] [--limit 2] [--json]
./target/release/one-grep serve --stdio        # or HTTP on 127.0.0.1:3210
./target/release/one-grep install --target opencode|cursor|pi|muse|hermes|command-code [--http --port 3210]
```

Notes:

* `query` without an index prints an `rg-fallback` live-grep result instead
  of failing.
* Rank backends (`--rank` / `search_ranked rank`): `jev` (hosted Nouls),
  `jina` (local ONNX cross-encoder, downloads on first use), `llama`
  (llama.cpp server with `--rerank`, e.g. `bge-reranker-v2-m3` on Vulkan.
  endpoint `ONE_GREP_RERANK_URL` or `--rank-endpoint`, default
  `http://127.0.0.1:8080`). Load or endpoint failures fall back to
  retrieval order, never an error. Serve it with:
  `llama-server -m bge-reranker-v2-m3-Q8_0.gguf --embedding --pooling rank`.
  Add `--device Vulkan` / `--n-gpu-layers all` on AMD. On CPU a
  20-doc pool takes tens of seconds.
* Matching is literal unless you pass `--regex`. `query --rank jev` needs
  `TYPESAFE_API_KEY` (else `~/.config/typesafe.env`). Model comes from
  `JEV_MCP_MODEL`, default `jev-1.13.0`. Without a key the call emits a
  `rank: fallback` note.
* Use `--http` with `opencode` only. Other targets use stdio entries. The
  installer uses `~/.local/bin/one-grep` when present.

## Agent install (fresh machine, ~2 min)

```bash
cargo install --git https://github.com/amitsheokand/one-grep
one-grep --version                                   # expect one-grep 0.2.0
one-grep install --target opencode                   # or cursor|pi|muse|hermes|command-code
one-grep index ~/my-repo && one-grep embed ~/my-repo # first embed downloads MiniLM once
one-grep query "where is auth handled?" --path ~/my-repo --hybrid
```

You need no API key until `query --rank jev` or `search_ranked` with Jev.
Without a key they emit `rank: fallback` and keep retrieval order.

## MCP tools

Server instructions: prefer `search_ranked` for intent (retrieve + Jev
inside the tool, only top-k enters context). Use `search` for the raw
fused pool. Use `rg` for exact text, symbols, or regex. Cite
`path:start-end` evidence.

Every hit cites `path:start-end`. To read more, read only the cited
line range — never the whole file. If the range is insufficient, narrow
the query instead of widening the read.

| Tool | Params | Returns |
| :--- | :--- | :--- |
| `search` | `root*`, `query*`, `fts?`, `fuse?=true`, `lang?`, `globs?`, `limit?=10` (1–50) | `path:start-end [breadcrumb] (score) source=bm25\|vec\|bm25+vec\|rg` chunks (text capped, 1-line crumb); `foo::Bar` and `"quoted"` / `'quoted'` route to exact `rg` unless the caller passes `fts`; single tokens go BM25 |
| `search_ranked` | same as `search` | same pool rescored by Jev, top-k only, `rank: jev exists=…` or `rank: fallback (reason)` header |
| `definition` | `root*`, `path*`, `line*` (1-based), `character*` (1-based), `server?` (default `rust-analyzer`) | `path:start-end (workspace\|external)`; escapes rejected, missing server is an error |
| `rg` | `root*`, `pattern*`, `regex?=false`, `structural?=false`, `case_insensitive?=false`, `lang?`, `globs?`, `format?=text`, `limit?=100` (1–500) | `path:line:text` lines (or `{"notes","hits"}` with `format=json`), gitignore-aware |
| `context` | `root*`, `path*`, `line*`, `format?=text` | enclosing symbol chunk (`path:start-end [breadcrumb] (kind)`) — expands a citation instead of a whole-file read |
| `skill` | `task*`, `limit?=2` (1–5), `dir?`, `format?=text` | `name  /absolute/path/.../SKILL.md  (score)` plus one-line description; Jev ranks when the caller sets a key, else lexical fallback (requires token overlap); never the SKILL.md body |

## Skill library

Keep rarely used skills under `~/.local/share/agent-skills`. Override with
`ONE_GREP_SKILLS_DIR` or the MCP/CLI `dir` param. Use one folder per skill
with a `SKILL.md` front matter (`name`, `description`). Harnesses that load
only `~/.cursor/skills` (or similar) skip loading every skill on every
turn. Call `skill` with the task first. Then read the winning absolute
`SKILL.md` path yourself.

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

How a Pi / Home Manager laptop wires it:
[nixos-config `docs/coding-agent-stack.md`](https://github.com/amitsheokand/nixos-config/blob/main/docs/coding-agent-stack.md).

## Privacy

* Embeddings run locally through fastembed ONNX. The code calls no
  embedding API.
* In HTTP mode the server binds `127.0.0.1` only. It requires
  `Authorization: Bearer <token>`.
* Only `search_ranked` / `query --rank jev` calls the network. It calls
  the configured Jev model. It never logs the key.
* Every MCP call appends one JSONL row to `~/.one-grep/serving.log`
  (`ONE_GREP_SERVING_LOG` overrides): tool, query, hits, chars, notes,
  latency. A pi extension (`~/.pi/agent/extensions/one-grep-hits.ts`,
  experimental) logs search and read calls beside it. An offline join of
  the two logs answers "hits → did they still read the file?".

## Alternatives

Substitutes, depending on which half of one-grep you need. Each entry
states where one-grep loses, with the measurement that earns the claim:

* **Exact text search**: [ripgrep](https://github.com/BurntSushi/ripgrep) —
  ripgrep is the baseline. one-grep's `rg` never beats it on raw latency.
  Run 9 measured 2.1 ms against 7.2 ms. Use `rg` when you know the
  literal text.
* **Structural search**: [ast-grep](https://github.com/ast-grep/ast-grep)
  (Rust, tree-sitter) — it matches AST shape instead of text
  (`$A && $A()` patterns, rewrite rules). one-grep does not reimplement
  this. `rg --structural -p 'pattern' --lang rust` and
  `query --hybrid --ast 'pattern' --ast-lang rust` shell out to the
  `ast-grep` binary when present. The hits fuse as a third RRF list
  tagged `source=ast`. Without the binary the call fails closed.
* **Hybrid codebase search**: `zg` (`zvec-grep`, Node/TypeScript cousin)
  — it follows the same BM25+vector idea. Our bench measured ~250 ms
  per query against ~70–200 ms for one-grep hybrid. one-grep is the
  faster native port. It includes the in-tree MCP server.
* **Structural search over MCP**: [ast-grep-mcp](https://github.com/ast-grep/ast-grep-mcp)
  — the ast-grep team's own MCP server. Use it side-by-side when agents
  need deep structural rules. Use one-grep's `search` for intent and
  `rg --structural` for one-off shape queries.
* **Judges**: [Laya](https://github.com/NandhaKishorM/laya) serves local,
  calibrated Nouls. We measured Laya as our ranker through
  `TYPESAFE_BASE_URL` (Run 14).
  [CLM](https://github.com/Contrastive-LM/CLM) runs faster on NVIDIA
  with relative scores. We track it for its hard-negative training
  recipe (Run 15).
* **Combined engines**: [`ox-core`](https://crates.io/crates/ox-core)
  (`ox-codes`) — ripgrep + tree-sitter + ast-grep as an HTTP service
  with rewrite and dataflow analysis. It is heavier than one-grep. Pick
  it when you need rewrite or codemod, not just retrieval.
* **Jev ranking backend**: the default is the hosted
  [TypeSafe Jev API](https://typesafe.ai/) (`TYPESAFE_API_KEY`). Local
  options speak the same `POST /v1/systemone` wire protocol, selectable
  through `TYPESAFE_BASE_URL`:
  * [LocalJev](https://github.com/githubnext/localjev) — Bun +
    DiffusionGemma through an OpenAI-compatible endpoint (oMLX).
    It is wire-compatible. The model reports its own probabilities
    instead of reading logits. Check calibration on your workload
    before you trust low-`exists` bands.
  * [OpenJev](https://github.com/razorback16/openjev) — patched vLLM
    backend with a structured logit read. It needs NVIDIA hardware.
* **Local rerank without Jev at all**: `query --rerank` / `--rank jina` / `--rank llama` (llama.cpp `--rerank` server)
  rescores on-device (Jina cross-encoder or a llama.cpp server). It needs
  no key and no network. Our bench measured ~1.1 s per query. In our bench it sweeps
  keywords (10/10). It adds nothing on paraphrased concepts (0/4).

## Docs

* Architecture, CLI table, and harness JSON shapes live in [INTEGRATION.md](INTEGRATION.md).
* Nix module and overlay live in [nix/README.md](nix/README.md).
* Benchmarks, ablations, and fine-tune notes live in [benchmarks/results.md](benchmarks/results.md).
* Build history (packets, receipts, task lists) lives in [docs/dev/](docs/dev/).
