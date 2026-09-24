#!/usr/bin/env python3
"""Cursor/Muse beforeReadFile gate: deny unbounded reads of large files.

A whole-file read of a large file is the 91% tool doing the most damage:
every line gets repaid on later turns. This hook allows the read only when
it is bounded (offset/limit present) or the file is small. Denials carry
an agent_message pointing at ranged reads, so the agent retries narrow
instead of failing.

Fail open everywhere: unknown payload shapes, missing files, and hook
errors all allow the read through. Exit 0 always; the permission field
carries the decision (exit code 2 would also block, but explicit JSON
is friendlier to review).
"""

import json
import os
import sys

# Mirrors toolgate's WHOLE_FILE_LIMIT_LINES: one budget, two enforcers.
WHOLE_FILE_LIMIT_LINES = 400


def extract_str(obj, *keys):
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def extract_num(obj, *keys):
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value >= 0:
            return value
    return None


def allow():
    print(json.dumps({"permission": "allow"}))


def deny(total, path):
    print(
        json.dumps(
            {
                "permission": "deny",
                "agent_message": (
                    f"Read denied: {path} has {total} lines; whole-file reads "
                    f"over {WHOLE_FILE_LIMIT_LINES} lines are capped. Re-read "
                    f"with offset/limit for the cited range only "
                    f"(e.g. the path:start-end from search), or call the "
                    f"one-grep `context` tool for the enclosing symbol."
                ),
            }
        )
    )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        allow()
        return
    try:
        # Cursor native and Claude-compatible shapes.
        tool = payload.get("tool") or payload.get("tool_name") or ""
        if isinstance(tool, dict):
            tool = tool.get("name", "")
        outer_input = payload.get("input")
        if outer_input is None:
            outer_input = payload.get("tool_input", {})
        path = extract_str(outer_input, "path", "file_path", "file", "filename")
        if path is None:
            # Also accept a bare path argument shape.
            path = extract_str(payload, "path", "file_path", "file")
        if path is None:
            allow()
            return
        if extract_num(outer_input, "offset", "start", "offset_lines") is not None:
            allow()  # bounded read: has a start
            return
        if extract_num(outer_input, "limit", "end", "limit_lines", "count") is not None:
            allow()  # bounded read: has an end
            return
        try:
            total = 0
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    total += chunk.count(b"\n")
                    if total > WHOLE_FILE_LIMIT_LINES:
                        break
        except OSError:
            allow()  # let the read tool report missing files itself
            return
        if total <= WHOLE_FILE_LIMIT_LINES:
            allow()
            return
        deny(total, path)
    except Exception:
        allow()


if __name__ == "__main__":
    # Live pilot: gate is on unless explicitly disabled per-machine.
    if os.environ.get("ONE_GREP_READ_GATE", "1") == "0":
        allow()
    else:
        main()
