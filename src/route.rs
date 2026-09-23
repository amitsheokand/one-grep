//! Query router: one classification shared by CLI `query` and MCP `search`.
//!
//! Classes, checked in order:
//!
//! 1. [`Route::Literal`] — a quoted `"..."` / `'...'` anchor (unambiguous:
//!    non-empty, no inner quote of the same kind) or a standalone
//!    `foo::Bar` path. Goes to exact `rg`, never ranked.
//! 2. [`Route::Identifier`] — a single identifier-like token. Goes to BM25
//!    lexical: exact beats fuzzy (benchmarks Run 5), so no vectors and no
//!    meter are spent on what the index answers directly.
//! 3. [`Route::Intent`] — everything else (natural language, mixed
//!    queries, anything with `fts` anchors). Goes to hybrid + ranker.
//!
//! Explicit caller flags beat heuristics: `fts` anchors (MCP) or
//! `--hybrid` / `--rank` (CLI) force [`Route::Intent`].

/// Where a query should execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// Exact literal text for `rg` (quotes already stripped).
    Literal(String),
    /// Single identifier-like token for BM25 lexical search.
    Identifier(String),
    /// Natural language for hybrid retrieval + ranking.
    Intent(String),
}

fn unquote(query: &str) -> Option<&str> {
    for quote in ['"', '\''] {
        if let Some(literal) = query
            .strip_prefix(quote)
            .and_then(|s| s.strip_suffix(quote))
        {
            if !literal.trim().is_empty() && !literal.contains(quote) {
                return Some(literal);
            }
        }
    }
    None
}

fn is_standalone_path(query: &str) -> bool {
    let identifier = |part: &str| {
        let mut chars = part.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    query.contains("::") && query.split("::").all(identifier)
}

fn is_identifier_token(query: &str) -> bool {
    let mut chars = query.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
}

/// Classify `query`. Non-empty `fts` anchors force [`Route::Intent`]:
/// the caller already supplied exact terms, so the shortcut is skipped.
#[must_use]
pub fn route(query: &str, fts: &[String]) -> Route {
    if !fts.is_empty() {
        return Route::Intent(query.to_owned());
    }
    let trimmed = query.trim();
    if let Some(literal) = unquote(trimmed) {
        return Route::Literal(literal.to_owned());
    }
    if is_standalone_path(trimmed) {
        return Route::Literal(trimmed.to_owned());
    }
    if is_identifier_token(trimmed) {
        return Route::Identifier(trimmed.to_owned());
    }
    Route::Intent(query.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_routing_table() {
        // Literal: quoted anchors and standalone paths.
        for query in [
            "foo::Bar",
            " foo::Bar ",
            "\"x:Name\"",
            "\"{Binding User.Name}\"",
            "'x:Name'",
            "'comment lies'",
        ] {
            assert!(matches!(route(query, &[]), Route::Literal(_)), "{query}");
        }
        // Identifier: single tokens go lexical, never ranked.
        for query in [
            "apply_request",
            "LaneConfig",
            "route_request",
            "src/auth.rs",
            "one-grep",
        ] {
            assert!(matches!(route(query, &[]), Route::Identifier(_)), "{query}");
        }
        // Intent: mixed, multi-word, ambiguous, or fts-anchored.
        for query in [
            "where is authentication handled",
            "find definition of foo::Bar",
            "foo::Bar example",
            "foo::",
            "::Bar",
            "foo:::Bar",
            "https://example.com",
            "\"\"",
            "\"   \"",
            "\"foo\" OR \"bar\"",
            "'foo' OR 'bar'",
            "",
            "   ",
        ] {
            assert!(matches!(route(query, &[]), Route::Intent(_)), "{query}");
        }
        // Explicit anchors beat heuristics.
        for query in ["foo::Bar", "\"x:Name\"", "apply_request"] {
            assert!(
                matches!(route(query, &["extra".into()]), Route::Intent(_)),
                "{query}"
            );
        }
    }

    #[test]
    fn literal_carries_stripped_text() {
        assert_eq!(
            route("\"x:Name\"", &[]),
            Route::Literal("x:Name".to_owned())
        );
        assert_eq!(
            route("foo::Bar", &[]),
            Route::Literal("foo::Bar".to_owned())
        );
    }
}
