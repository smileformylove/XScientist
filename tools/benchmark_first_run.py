#!/usr/bin/env python3
"""Run XScientist's public, zero-cost first-run benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from xscientist.benchmark import benchmark_first_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=None)
    parser.add_argument(
        "--profile",
        choices=["balanced", "discovery", "publication"],
        default="balanced",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--output", default=None, help="optional JSON report path")
    args = parser.parse_args()
    payload = benchmark_first_run(
        args.workspace,
        profile=args.profile,
        max_seconds=args.max_seconds,
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
