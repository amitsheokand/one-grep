# RECEIPT-001 — T-og-query-lenient

## Gates

```bash
nix develop -c cargo test -q && nix develop -c cargo fmt --check
```

### `cargo test -q` (tail)

```
running 153 tests
.........i............................................................................. 87/153
..................................................................
test result: ok. 152 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.66s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

### `cargo fmt --check` (tail)

```
warning: Git tree '/home/amitsheokand/work/worktrees/one-grep/T-og-query-lenient' is dirty
```

(exit 0, no diff output)

## Done

- BM25 `index::search` uses `QueryParser::parse_query_lenient` so agent prose (`thread:`, colons, unbalanced quotes/parens, leading `-`) no longer returns `InvalidInput`.
- `text_hit_body` in `src/mcp.rs`: text-format hit-list tools emit `0 hits` when the result set is empty (including `rg`).
- Tests: `bm25_lenient_query_syntax_does_not_error`, `rg_text_format_zero_hits_is_not_empty`.
