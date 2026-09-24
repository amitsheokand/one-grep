# RECEIPT-001 — T-og-annot

## Gates

Command (from `PACKET.md`):

```bash
nix develop -c cargo test -q && nix develop -c cargo fmt --check
```

### `cargo test -q` (tail)

```
running 147 tests
..........i............................................................................ 87/147
............................................................
test result: ok. 146 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.60s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

### `cargo fmt --check` (tail)

```
warning: Git tree '/home/amitsheokand/work/worktrees/one-grep/T-og-annot' is dirty
```

(exit 0, no diff output)

## Done

- All six `#[tool(...)]` handlers in `src/mcp.rs` declare `title` and `annotations(...)` (`read_only_hint = true`; `open_world_hint = true` on `search_ranked` and `skill`).
- `every_tool_declares_read_only_hint` lists tools via `tool_router.list_all()` and asserts `read_only_hint == Some(true)`.
- Serving-log unit tests use thread-local log paths so parallel `cargo test` does not pollute `ONE_GREP_SERVING_LOG`.
- `src/install.rs` rustfmt-only fix so `cargo fmt --check` passes on this branch.
