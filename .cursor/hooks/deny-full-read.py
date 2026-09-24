#!/usr/bin/env python3
"""beforeReadFile gate for whole-file reads (Cursor/Muse hook shape).

A whole-file read of a large file is the 91% tool doing the most damage:
every line gets repaid on later turns. Modes via ONE_GREP_READ_GATE:

- unset or "observe" (default): log the decision, allow everything.
  Observe first; enforce only after the log says the rule fires cleanly.
- "enforce": deny unbounded reads over the limit, with an agent_message
  pointing at ranged reads.
- "off" or "0": allow silently, log nothing.

Decisions append as JSONL to ~/.pi/agent/obs/read-gate.log so an offline
join measures what enforcing would have blocked. Fail open everywhere:
unknown payloads, missing files, and hook errors all allow the read.
"""

import json
import os
import sys
import time

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


def log_decision(path, total, bounded, decision):
    try:
        home = os.environ.get("HOME", "")
        if not home:
            return
        row = {
            "ts": time.time(),
            "path": path,
            "lines": total,
            "bounded": bounded,
            "decision": decision,
        }
        with open(f"{home}/.pi/agent/obs/read-gate.log", "a") as handle:
            handle.write(json.dumps(row) + "\n")
    except Exception:
        pass


def decide(path, bounded, total):
    mode = os.environ.get("ONE_GREP_READ_GATE", "observe")
    if mode in ("off", "0"):
        return "allow", False
    if bounded or total <= WHOLE_FILE_LIMIT_LINES:
        log_decision(path, total, bounded, "allow")
        return "allow", True
    log_decision(path, total, bounded, "deny" if mode == "enforce" else "would-deny")
    if mode == "enforce":
        return "deny", True
    return "allow", True


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
            path = extract_str(payload, "path", "file_path", "file")
        if path is None:
            allow()
            return
        bounded = (
            extract_num(outer_input, "offset", "start", "offset_lines") is not None
            or extract_num(outer_input, "limit", "end", "limit_lines", "count")
            is not None
        )
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
        verdict, _logged = decide(path, bounded, total)
        if verdict == "deny":
            deny(total, path)
        else:
            allow()
    except Exception:
        allow()


if __name__ == "__main__":
    main()
