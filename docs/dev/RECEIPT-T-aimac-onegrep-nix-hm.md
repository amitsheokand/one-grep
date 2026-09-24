# Receipt: T-aimac-onegrep-nix-hm

## Status: GREEN (COMPLETE — local milestone)

**Target**: `ai-mac` (macOS arm64)  
**Repo**: `~/dev/one-grep`  
**Commit message**: `onegrep: home-manager module (OSS-safe)`  
**No push.**

---

## What landed

OSS-safe Nix flake + Home Manager module for declarative `one-grep` install and
optional MCP registration (cursor / opencode / pi / muse / hermes / command-code).

| Piece | Path |
| :--- | :--- |
| Flake | `flake.nix` + `flake.lock` (nixpkgs nixos-unstable) |
| Package | `nix/package.nix` (`rustPlatform`, `ORT_STRATEGY=system` + nixpkgs `onnxruntime`) |
| HM module | `nix/home-manager/one-grep.nix` → `programs.one-grep` |
| Docs | `nix/README.md` + `INTEGRATION.md` §5 |

### Module capabilities

- Overlay exposes `pkgs.one-grep`; flake `packages.<system>.one-grep`
- `programs.one-grep.enable` + optional `home.packages` via `installPackage`
- `command` override when package comes from cargo/dev symlink
- `mcp.*.enable` + overridable relative `path` options (no hostnames / private strings)
- Activation upserts only the `one-grep` entry (jq for JSON; Python for Hermes YAML)

### OSS / privacy

- Scanned `nix/` + `flake.nix`: no Advait / Vaayu / agni / duduk / user home paths
- Module lives in **one-grep** for import; **nixos-config / advait farm not edited**

---

## Validation

| Check | Result |
| :--- | :--- |
| `nix flake check --no-build` | **PASS** (aarch64-darwin; linux systems omitted on Darwin) |
| `nix eval .#packages.aarch64-darwin.one-grep.drvPath` | **PASS** |
| `nix-instantiate --parse` on module + package | **PASS** |
| `nix build .#checks.aarch64-darwin.module-file` | **PASS** (file presence) |
| `nix build .#one-grep` (full rust compile) | **Queued validate-on-nixos** (agni/vaayu) — Darwin compile skipped as heavy; cargo/ORT fetch not exercised here |

---

## Files

- `flake.nix`, `flake.lock`
- `nix/package.nix`
- `nix/home-manager/one-grep.nix`
- `nix/README.md`
- `INTEGRATION.md` (new §5 Nix / Home Manager)
- `docs/dev/RECEIPT-T-aimac-onegrep-nix-hm.md` (this receipt)

---

## Still Queued / out of scope

1. Full `nix build .#one-grep` on Linux (agni/vaayu) — package drv evaluates on Darwin; binary build not proven.
2. Wire module into consumer nixos-config hosts (import only; not done here).
3. Advait farm Hard parks — not touched.
4. OpenCode HTTP/bearer via HM — stdio only in module (CLI `--http` remains).
