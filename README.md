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
