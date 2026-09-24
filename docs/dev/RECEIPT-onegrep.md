# Receipt: T-aimac-onegrep Integration

## Status: COMPLETE

**Target**: `ai-mac` (macOS arm64)  
**Binary Symlink**: `~/.local/bin/one-grep -> /Users/amitsheokand/dev/one-grep/target/release/one-grep`  
**Repo**: `~/dev/one-grep`

---

## 1. Harness Wiring Summary

| Harness | Configuration File | Format | Transport | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Cursor Agent CLI** | `~/.cursor/mcp.json` | JSON | `stdio` | Verified (`one-grep: ready`, tools: `rg`, `search`) |
| **OpenCode** | `~/.config/opencode/opencode.json` | JSON | `stdio` / local | Verified (updated from debug to release symlink) |
| **Pi** | `~/.pi/agent/mcp.json` | JSON | `stdio` | Verified (`mcpServers` entry active via `pi-mcp-adapter`) |
| **Muse** | `~/.config/muse/settings.json` | JSON | `stdio` | Verified (`mcpServers` & `mcp_servers` entries active) |
| **Hermes** | `~/.hermes/config.yaml` | YAML | `stdio` | Verified (`hermes mcp test one-grep`: connected 275ms, 2 tools discovered) |
| **Command-Code GOAT** | `~/.commandcode/mcp.json` | JSON | `stdio` | Verified (`command-code mcp list`: enabled) |

---

## 2. How Agents Invoke `one-grep`

### Via MCP (Model Context Protocol)
All agent harnesses have access to two tools:
1. `search`: Hybrid BM25 + ONNX vector embedding search with RRF score ranking.
   ```json
   {
     "root": "/Users/amitsheokand/dev/one-grep",
     "query": "FastembedProvider model loading",
     "limit": 5,
     "fuse": true
   }
   ```
2. `rg`: Exact string and regex ripgrep search without needing an index.
   ```json
   {
     "root": "/Users/amitsheokand/dev/one-grep",
     "pattern": "FastembedProvider",
     "limit": 10
   }
   ```

### Via CLI Direct Invocation
```bash
# Index a workspace
one-grep index /path/to/workspace

# Search indexed workspace
one-grep query "search query" --path /path/to/workspace --limit 10 --hybrid

# Quick text/regex search
one-grep rg "pattern" /path/to/workspace
```

---

## 3. Smoke Verification Proofs

### Proof 1: CLI Index & Query
```bash
$ one-grep index .
indexed .: 23 scanned, 22 upserted, 490 chunks, 1 removed

$ one-grep query "FastembedProvider" --path . --limit 2
./src/embed.rs:35-40 [FastembedProvider] (9.83)
pub struct FastembedProvider {
    model: Mutex<fastembed::TextEmbedding>,
    which: OnnxModel,
    name: String,
    dims: usize,
}
./src/embed.rs:201-226 [impl FastembedProvider] (8.08)
impl EmbedProvider for FastembedProvider {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
...
```

### Proof 2: CLI `rg`
```bash
$ one-grep rg "FastembedProvider" . --limit 4
./src/embed.rs:3://! [`FastembedProvider`] runs a tiny ONNX model in-process (default path,
./src/embed.rs:35:pub struct FastembedProvider {
./src/embed.rs:123:impl FastembedProvider {
./src/embed.rs:201:impl EmbedProvider for FastembedProvider {
```

### Proof 3: Hermes MCP Connection Test
```bash
$ hermes mcp test one-grep
  Testing 'one-grep'...
  Transport: stdio → /Users/amitsheokand/.local/bin/one-grep
  Auth: none
  ✓ Connected (275ms)
  ✓ Tools discovered: 2

    rg                                   Exact text or regex search over workspace files (no ind...
    search                               Hybrid workspace search: semantic discovery fused with ...
```

### Proof 4: Cursor Agent CLI MCP Discovery
```bash
$ cursor-agent mcp list
headroom: Error: Connection failed
zvec_grep: ready
one-grep: ready

$ cursor-agent mcp list-tools one-grep
Tools for one-grep (2):
- rg (case_insensitive, limit, pattern, regex, root)
- search (fts, fuse, limit, query, root)
```

### Proof 5: Command-Code MCP Registration
```bash
$ command-code mcp list
MCP Servers

  NAME      TYPE   SCOPE    AUTH  STATUS
  one-grep  stdio  user     -     enabled

Total: 1 server
```

### Proof 6: MCP Protocol Stdio End-to-End
Invoking JSON-RPC 2.0 handshake and tool execution over `one-grep serve --stdio`:
```json
// Initialize response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": { "tools": {} },
    "serverInfo": { "name": "rmcp", "version": "2.2.0" },
    "instructions": "Local-first hybrid workspace search. Prefer `search` for intent/concepts (it fuses semantic + BM25 ranks); use `rg` to verify exact text, symbols, or regex. Cite path:line evidence."
  }
}

// tools/list response
TOOLS: ["rg", "search"]

// tools/call search
Response returned citation: /Users/amitsheokand/dev/one-grep/src/embed.rs:165-198 (score 0.0659)
```

### Proof 7: HTTP Loopback & Bearer Auth
```bash
$ python3 -c "... test http 127.0.0.1:3210/mcp ..."
UNAUTH HTTP STATUS: 401
AUTH TOKEN VERIFIED: length 48
```

---

## 4. Gaps & Blockers for Duduk / Future Work

1. **`one-grep install` Target Expansion**:
   `src/install.rs` currently implements `--target opencode` only. Expanding `install.rs` to write target configs for `cursor`, `pi`, `muse`, `hermes`, and `commandcode` natively would streamline single-command onboarding across seats.
2. **NixOS / Vaayu Packaging**:
   ai-mac configurations were placed directly into user directories (`~/.cursor/mcp.json`, `~/.config/opencode/opencode.json`, `~/.pi/agent/mcp.json`, `~/.config/muse/settings.json`, `~/.hermes/config.yaml`, `~/.commandcode/mcp.json`). For systems with strict declarative home-manager setups, creating a shared Nix module or home-manager overlay will be beneficial when deploying to Vaayu/NixOS.
3. **Privacy Assurance**:
   Local FastEmbed ONNX runtime verified. No external outbound network requests or China-hosted endpoints used during indexing or searching.
