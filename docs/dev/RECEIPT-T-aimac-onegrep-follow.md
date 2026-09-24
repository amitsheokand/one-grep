# Receipt: T-aimac-onegrep-follow

## Status: GREEN (COMPLETE)

**Target**: `ai-mac` (macOS arm64)
**Repos**: `~/dev/one-grep` + `~/dev/nixos-config`
**No push. Local commits only.** (nixos-config `origin` = GitHub; coordinator fetches.)

Follow-ups from `docs/dev/RECEIPT-onegrep.md` §4: (1) expand `one-grep install --target`,
(2) shared Home Manager module for NixOS + vaayu.

---

## 1. Install targets (goal 1)

`src/install.rs` now registers every harness on this machine, not just `opencode`.
Selected with `--target <name>`; binary resolves to `~/.local/bin/one-grep` when
present, else the running `current_exe()`.

| Target | Config path | Entry shape |
| :--- | :--- | :--- |
| `opencode` | `~/.config/opencode/opencode.json` | `mcp.one-grep` = `{type: local, command: [exe, serve, --stdio], enabled}` (or `--http` remote+bearer) |
| `cursor` | `~/.cursor/mcp.json` | `mcpServers.one-grep` = `{command, args:[serve,--stdio]}` |
| `pi` | `~/.pi/agent/mcp.json` | `mcpServers.one-grep` = `{command, args}` |
| `muse` | `~/.config/muse/settings.json` | upserts **both** `mcpServers` and `mcp_servers` |
| `hermes` | `~/.hermes/config.yaml` | surgical YAML upsert under `mcp_servers:` (preserves anchors/comments) |
| `command-code` | `~/.commandcode/mcp.json` | `mcpServers.one-grep` = `{transport: stdio, enabled, command, args}` |

Aliases: `commandcode`, `command_code`. `--http` is `opencode`-only (errors otherwise).

---

## 2. Shared Home Manager module (goal 2)

Wiring is shared so NixOS (incl. vaayu) and Darwin stop hand-editing client
configs. Source of truth is the one-grep repo; `nixos-config` vendors it.

| Piece | Path |
| :--- | :--- |
| Upstream module | `~/dev/one-grep/nix/home-manager/one-grep.nix` → flake output `homeManagerModules.one-grep` |
| Upstream package | `~/dev/one-grep/nix/package.nix` (+ `flake.nix` overlay) |
| nixos-config wrapper | `~/dev/nixos-config/modules/shared/one-grep.nix` |
| nixos-config vendored module | `~/dev/nixos-config/modules/shared/one-grep-module.nix` |
| nixos-config overlay | `~/dev/nixos-config/overlays/one-grep.nix` → `pkgs.one-grep` |
| Imported by | `modules/nixos/home-manager.nix`, `modules/nixos/home-manager-vaayu.nix`, `modules/darwin/home-manager.nix` |

### What changed in this pass

Previously the wrapper was parked at `enable = false` (no public remote → package
builds only from a sibling `../one-grep` checkout, impure). It is now **enabled**
without an impure package dependency:

- `enable = true`, `package = null`, `installPackage = false`
- `command = "${config.home.homeDirectory}/.local/bin/one-grep"` — same path the
  CLI's own `install` prefers; no literal home path in the public repo
- all six `mcp.*.enable = true`
- new module option **`checkCommand`** (upstream + vendored, mirrored): activation
  runs `command -v <command>` and **skips registration with a warning** when the
  binary is absent, so hosts that do not have one-grep yet get no dead MCP
  entries. `pkgs.one-grep` remains for hosts that build the sibling checkout
  (`package = pkgs.one-grep; installPackage = true;`).

Activation still upserts only the `one-grep` key after `writeBoundary`, so peer
servers survive — including Headroom's `force`-written `~/.cursor/mcp.json`.

---

## 3. How to apply

```sh
# 1. Provide the binary on the host (path the module looks for):
cargo build --release && ln -sf "$PWD/target/release/one-grep" ~/.local/bin/one-grep

# 2. Rebuild the host (NixOS: nixos-rebuild; Mac: nix-darwin)
nix run .#build-switch        # nixos-config app
# then restart the agent / MCP client
```

Non-declarative alternative on any host:
`one-grep install --target opencode|cursor|pi|muse|hermes|command-code`

Hosts without the binary rebuild cleanly: activation prints
`one-grep: <path> not found; skipping MCP registration` and writes nothing.

---

## 4. Verification

| Check | Result |
| :--- | :--- |
| `one-grep install --help` | Targets listed; `--http` documented |
| Isolated-`HOME` install of all 6 targets | 6 files written, correct shape (see §1) |
| Negative: `--target cursor --http` / `--target bogus` | Error, exit 1 |
| `cargo test` (one-grep) | **40 passed**, exit 0 |
| `nix flake check --no-build` (one-grep) | **all checks passed** (aarch64-darwin) |
| `nix eval` `programs.one-grep` on garfield / vaayu / aarch64-darwin | `enable=true`, `checkCommand=true`, `command=~/.local/bin/one-grep`, all `mcp.*.enable=true` |
| Generated activation `bash -n` | Syntax OK |
| Guard — command missing | Skip branch: warning, **0 files** written |
| Guard — command present | Execute branch: JSON targets written; Hermes helper store path realises on switch |
| Hermes helper (extracted, run) | Insert + idempotent (1 entry) + preserves peers, `&id001` anchor, comments, top-level keys |
| Peer preservation (execute) | Cursor keeps `headroom` + `zvec_grep`; pi/muse/opencode/command-code peers kept; rerun → 1 `one-grep` entry |

Note: `nix flake check` on nixos-config itself fails on a **pre-existing** Asahi
firmware assertion for `vaayu` (needs `--impure`) — unrelated to one-grep, which
is why verification was done per-option/per-host plus a `bash -n` of the
generated activation.

---

## 5. Files

**one-grep**
- `src/install.rs` — multi-target install (committed earlier as `e692f7e`)
- `nix/home-manager/one-grep.nix` — `checkCommand` guard option
- `nix/README.md` — `checkCommand` documented
- `docs/dev/RECEIPT-T-aimac-onegrep-follow.md` — this receipt

**nixos-config**
- `modules/shared/one-grep.nix` — enabled wrapper (command route + guard)
- `modules/shared/one-grep-module.nix` — vendored; `checkCommand` guard
- `modules/nixos/home-manager.nix`, `modules/nixos/home-manager-vaayu.nix`,
  `modules/darwin/home-manager.nix` — import call sites pass `config`
- `AGENTS.md` — one-grep wiring section

---

## 6. Still queued (out of scope)

1. **Public remote for one-grep** — then the package route becomes clean
   (flake input or `fetchFromGitHub`); the `command` route stays as fallback.
2. **Full `nix build .#one-grep` on Linux** (agni/vaayu) — package drv evaluates
   on Darwin; binary build not proven here.
3. **China/PAYG policy, farm crates, hipfire GPU** — not touched.
