sdkrun_v1

## Packet T-og-skill-route (review fix)

### Changes

- Lexical fallback: require ≥1 shared non-stopword token and score ≥ 0.2; otherwise `no_match` + `no matching skill`.
- Hit paths: canonical absolute filesystem path to each `SKILL.md`.

### Gates

- `cargo test --lib`: **130 passed; 0 failed; 1 ignored**
- `cargo build --release`: ok (via `nix develop .#one-grep`)

### Sample `one-grep skill` (fallback, isolated HOME, no key)

**Matching task:**

```
rank: fallback (jev: missing `TYPESAFE_API_KEY` (checked process env, ~/.config/typesafe.env, ~/.config/environment.d/60-typesafe.conf))
winrt-lookup  /tmp/.../quoted-skill/SKILL.md  (1.0000)
Quoted WinRT API reference workflow
plain-api  /tmp/.../plain-skill/SKILL.md  (0.2500)
Plain skill for REST API documentation lookup
```

**Non-matching task (`quantum gardening underwater zzz`):**

```
rank: fallback (jev: missing `TYPESAFE_API_KEY` (checked process env, ~/.config/typesafe.env, ~/.config/environment.d/60-typesafe.conf))
no matching skill
```

### Outside fence

Nothing.
