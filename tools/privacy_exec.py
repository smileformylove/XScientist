#!/usr/bin/env python3
"""Run a command while removing credentials and host paths from its output."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai_scientist.utils.privacy import redact_sensitive_text  # noqa: E402


def run(command: Sequence[str]) -> int:
    """Stream sanitized combined output and preserve the child's exit code."""

    argv = list(command)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print("privacy_exec: command is required", file=sys.stderr)
        return 2
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        print(
            "privacy_exec: failed to start command: " + redact_sensitive_text(str(exc)),
            file=sys.stderr,
        )
        return 127

    assert process.stdout is not None
    try:
        for line in process.stdout:
            sys.stdout.write(redact_sensitive_text(line))
            sys.stdout.flush()
    except KeyboardInterrupt:
        process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()
    finally:
        process.stdout.close()
    return process.wait()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return run(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
