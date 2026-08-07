#!/usr/bin/env python3
"""Audit publishable repository content without printing matched values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai_scientist.utils.privacy import (
    PrivacyFinding,
    format_privacy_findings,
    privacy_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = privacy_report(
        args.path,
        include_untracked=args.include_untracked,
        history=args.history,
    )
    if args.as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif report["ok"]:
        print("Privacy audit: clean (matched values were never displayed)")
    else:
        print(
            format_privacy_findings(
                PrivacyFinding(**finding) for finding in report["findings"]
            ),
            file=sys.stderr,
        )
        print(
            f"Privacy audit: {report['finding_count']} finding(s); "
            "matched values were not displayed",
            file=sys.stderr,
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
