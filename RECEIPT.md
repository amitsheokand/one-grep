sdkrun_v1

## Packet T-og-skill-route

### Files changed (net lines)

| File | + | - |
| :--- | --: | --: |
| `src/skill.rs` | new | |
| `src/lib.rs` | 1 | 0 |
| `src/jev.rs` | 5 | 5 |
| `src/mcp.rs` | ~100 | ~2 |
| `src/main.rs` | ~34 | ~0 |
| `README.md` | ~14 | ~2 |
| `INTEGRATION.md` | 2 | 1 |

### Gates

- `cargo test --lib`: **129 passed; 0 failed; 1 ignored**
- `cargo build --release`: ok (via `nix develop .#one-grep`)

### Sample `one-grep skill` (fallback, fixture)

```
rank: fallback (jev: missing `TYPESAFE_API_KEY` (checked process env, ~/.config/typesafe.env, ~/.config/environment.d/60-typesafe.conf))
winrt-lookup  quoted-skill/SKILL.md  (1.0000)
Quoted WinRT API reference workflow
plain-api  plain-skill/SKILL.md  (0.2500)
Plain skill for REST API documentation lookup
```

### Outside fence

Nothing required outside the packet fence.
