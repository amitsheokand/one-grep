# one-grep

Local-first hybrid workspace search: ripgrep + BM25 + ONNX embeddings, with an
in-tree MCP server (`search` + `rg`).

Public repo: [github.com/amitsheokand/one-grep](https://github.com/amitsheokand/one-grep).
The CLI and crate are **`one-grep`**.

```sh
one-grep index /path/to/workspace
one-grep embed /path/to/workspace
one-grep query "where is authentication handled?" --path /path/to/workspace --hybrid
one-grep serve --stdio
```

Nix: `nix build github:amitsheokand/one-grep` or import
`homeManagerModules.one-grep`. Details: [INTEGRATION.md](INTEGRATION.md),
[nix/README.md](nix/README.md).

How it is wired into a Pi / Home Manager laptop:
[nixos-config `docs/coding-agent-stack.md`](https://github.com/amitsheokand/nixos-config/blob/main/docs/coding-agent-stack.md).

## Quickstart

```bash
cargo build --release
./target/release/one-grep index <path>
./target/release/one-grep query "concept" --path <path> --hybrid
./target/release/one-grep rg "pattern" <path>
./target/release/one-grep serve --stdio        # or --port 3210 for HTTP on 127.0.0.1
./target/release/one-grep install --target opencode|cursor|pi|muse|hermes|command-code
```

## Install targets

| Target | Config |
| :--- | :--- |
| `opencode` | `~/.config/opencode/opencode.json` (`--http` optional) |
| `cursor` | `~/.cursor/mcp.json` |
| `pi` | `~/.pi/agent/mcp.json` |
| `muse` | `~/.config/muse/settings.json` |
| `hermes` | `~/.hermes/config.yaml` |
| `command-code` | `~/.commandcode/mcp.json` |

Idempotent upsert; peer servers preserved. Details in `INTEGRATION.md` §4–§5.
