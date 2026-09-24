# one-grep Integration & Architecture Guide

`one-grep` is a local-first hybrid workspace search engine built in Rust. It combines BM25 lexical search with ONNX vector embeddings and reciprocal rank fusion (RRF) to provide fast, privacy-preserving semantic and lexical codebase search over an in-tree Model Context Protocol (MCP) server or CLI.

---

## 1. Binary & Path Setup

- **Source Workspace**: `~/dev/one-grep`
- **Release Binary**: `~/dev/one-grep/target/release/one-grep`
- **Installed Symlink on PATH**: `~/.local/bin/one-grep -> /Users/amitsheokand/dev/one-grep/target/release/one-grep`
- **Cache / Token Directory**: `~/.one-grep/`
  - `~/.one-grep/token` (0600 mode bearer token for HTTP mode)
  - `~/.cache/one-grep/` (fastembed ONNX models cache)

### Note on `one-grep` vs `zg`
- `one-grep`: Rust implementation (`target/release/one-grep`), self-contained, native fastembed ONNX runtime, in-tree MCP stdio/HTTP server.
- `zg`: Node/TypeScript `zvec-grep` cousin (`~/.local/bin/zg`). Do not confuse; `one-grep` is the primary high-performance engine for this architecture.

---

## 2. CLI Reference

```bash
one-grep <COMMAND>
```

| Subcommand | Arguments / Flags | Description |
| :--- | :--- | :--- |
| `index` | `<path>` | Indexes workspace files for BM25 and chunk extraction. Stores index in `<path>/.onegrep/`. |
| `query` / `search` | `<query> [--path <path>] [--limit <n>] [--hybrid] [--rerank] [--rank jev\|jina]` | BM25 query; `--hybrid` fuses vectors; `--rank jev` reranks the shortlist with TypeSafe inside the tool (top-k only). `search` is an alias. |
| `embed` | `<path> [--model minilm\|arctic-m\|gemma-300m]` | Computes embeddings for extracted chunks using local ONNX fastembed. |
| `watch` | `<path>` | Watches workspace and incrementally updates index on file changes. |
| `dump-chunks` | `<path>` | Dumps extracted code chunks as JSONL. |
| `rg` | `<pattern> [<path>] [--regex] [--case-insensitive] [--limit <n>]` | Instant gitignore-aware text/regex search directly over files without indexing. |
| `serve` | `[--stdio] [--port 3210]` | Serves the Model Context Protocol (MCP) server over stdio or Streamable HTTP on loopback `127.0.0.1:<port>/mcp`. |
| `install` | `[--target opencode/cursor/pi/muse/hermes/command-code] [--http] [--port 3210]` | Idempotent upsert of `one-grep` MCP stdio entry into the harness config (`--http` only for `opencode`). |
| `skill` | `"<task>" [--dir <path>] [--limit N] [--json]` | Route a task to the best matching skill under `ONE_GREP_SKILLS_DIR` or `~/.local/share/agent-skills` (Jev when keyed, else lexical fallback). Hits cite an **absolute** path to `SKILL.md`; no token overlap → `no matching skill`. |

---

## 2.1 `one-grep install`

```bash
one-grep install --target <opencode|cursor|pi|muse|hermes|command-code>
# aliases: commandcode, command_code
# OpenCode HTTP (optional):
one-grep install --target opencode --http [--port 3210]
```

Writes/upserts the `one-grep` entry shapes documented in §4 (stdio → `~/.local/bin/one-grep serve --stdio` when that symlink exists). Idempotent.

---

## 3. MCP Surface (`src/mcp.rs`)

The MCP server exposes search, definition, `rg`, and `skill` tools over both `stdio` and loopback HTTP (`127.0.0.1:3210/mcp` with Bearer authentication):

### Server Instructions
> "Local-first hybrid workspace search. Prefer `search_ranked` for intent (retrieve + Jev inside the tool; only top-k winners enter context). Use `search` for the raw fused pool. Use `rg` for exact text, symbols, or regex. Cite path:line evidence."

### Tool: `search`
Hybrid workspace search combining semantic discovery with BM25 lexical ranking. Returns the fused shortlist (pool enters the caller).
- **Parameters**:
  - `root` (string, required): Absolute workspace directory path.
  - `query` (string, required): Natural language query or concept description.
  - `fts` (array of strings, optional): Exact keywords/anchors folded into lexical ranking.
  - `fuse` (boolean, optional, default `true`): Whether to fuse vector similarity with BM25.
  - `limit` (integer, optional, default 10, max 50): Maximum result chunks.
- **Returns**: Formatted snippets with `path:start_line-end_line [symbol_breadcrumb] (score)` citations.

### Tool: `search_ranked`
Same retrieval as `search`, then TypeSafe Jev scores each candidate inside the tool (`exists` Noul + one Noul per hit). Returns top-k only, prefixed with `rank: jev exists=…` or `rank: fallback (reason)` if the key is missing or the API fails. Exact anchors (`foo::Bar`, `"quoted"`) still route to `rg`.

Key: `TYPESAFE_API_KEY`, else `~/.config/typesafe.env`, else `~/.config/environment.d/60-typesafe.conf`. Model: `JEV_MCP_MODEL` (default `jev-1.13.0`). Never log the key.

### Tool: `rg`
Exact string or regular expression search across workspace files with gitignore filtering.
- **Parameters**:
  - `root` (string, required): Absolute workspace directory path.
  - `pattern` (string, required): Text pattern or regex to search.
  - `regex` (boolean, optional, default `false`): Enable regex syntax.
  - `case_insensitive` (boolean, optional, default `false`): Case-insensitive match.
  - `limit` (integer, optional, default 100, max 500): Maximum matching lines.
- **Returns**: `path:line:matching_text` lines.

---

## 4. Harness Configuration Matrix

All configurations use non-invasive user-directory configurations:

### 1. Cursor Agent CLI (`~/.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "one-grep": {
      "command": "/Users/amitsheokand/.local/bin/one-grep",
      "args": [
        "serve",
        "--stdio"
      ]
    }
  }
}
```

### 2. OpenCode (`~/.config/opencode/opencode.json`)
```json
{
  "mcp": {
    "one-grep": {
      "type": "local",
      "command": [
        "/Users/amitsheokand/.local/bin/one-grep",
        "serve",
        "--stdio"
      ],
      "enabled": true
    }
  }
}
```

### 3. Pi (`~/.pi/agent/mcp.json`)
Managed via `pi-mcp-adapter`:
```json
{
  "settings": {
    "toolPrefix": "server",
    "idleTimeout": 10
  },
  "mcpServers": {
    "one-grep": {
      "command": "/Users/amitsheokand/.local/bin/one-grep",
      "args": [
        "serve",
        "--stdio"
      ]
    }
  }
}
```

### 4. Muse (`~/.config/muse/settings.json`)
```json
{
  "mcpServers": {
    "one-grep": {
      "command": "/Users/amitsheokand/.local/bin/one-grep",
      "args": [
        "serve",
        "--stdio"
      ]
    }
  },
  "mcp_servers": {
    "one-grep": {
      "command": "/Users/amitsheokand/.local/bin/one-grep",
      "args": [
        "serve",
        "--stdio"
      ],
      "enabled": true,
      "mode": "optional"
    }
  }
}
```

### 5. Hermes (`~/.hermes/config.yaml`)
```yaml
mcp_servers:
  one-grep:
    command: /Users/amitsheokand/.local/bin/one-grep
    args:
      - serve
      - --stdio
```

### 6. Command-Code GOAT (`~/.commandcode/mcp.json`)
```json
{
  "mcpServers": {
    "one-grep": {
      "transport": "stdio",
      "enabled": true,
      "command": "/Users/amitsheokand/.local/bin/one-grep",
      "args": [
        "serve",
        "--stdio"
      ]
    }
  }
}
```

---

## 5. Nix / Home Manager (OSS module)

Declarative packaging lives in-tree under `nix/` (import into a consumer
home-manager / nixos config — do not bake private host paths into the module).

See **`nix/README.md`** for full options and examples.

```nix
# flake input → overlay + HM module
imports = [ inputs.one-grep.homeManagerModules.one-grep ];
nixpkgs.overlays = [ inputs.one-grep.overlays.default ];

programs.one-grep = {
  enable = true;
  package = pkgs.one-grep;
  mcp = {
    cursor.enable = true;
    opencode.enable = true;  # stdio local
    pi.enable = true;
    muse.enable = true;
    hermes.enable = true;
    commandCode.enable = true;
  };
};
```

| Flake output | Role |
| :--- | :--- |
| `packages.<system>.one-grep` | rustPlatform package |
| `overlays.default` | `pkgs.one-grep` |
| `homeManagerModules.one-grep` | `programs.one-grep.*` |

MCP paths are **options** (defaults match §4 / `one-grep install`). Activation
upserts only the `one-grep` entry so peer servers remain intact.

---

## 6. Privacy & Security Constraints

- **Local Execution**: All embeddings run locally via FastEmbed ONNX runtime (e.g. `bge-small-en-v1.5` / `minilm`).
- **No Cloud/China Endpoints**: No external inference or embedding API endpoints are called.
- **Localhost Only**: In HTTP mode (`one-grep serve`), socket binds strictly to `127.0.0.1`.
- **Bearer Token**: HTTP endpoints reject requests missing `Authorization: Bearer <token>` (`~/.one-grep/token` generated with mode `0600`).
