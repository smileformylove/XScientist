#!/usr/bin/env python3
"""Fail closed when a release tag and repository metadata disagree."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.engineering_checks import (
    _project_version,
    check_version_metadata,
)  # noqa: E402


def validate_release(tag: str, root: Path = REPOSITORY_ROOT) -> list[str]:
    errors = check_version_metadata(root)
    expected = f"v{_project_version(root)}"
    if tag != expected:
        errors.append(f"release tag must be {expected}, got {tag or '<empty>'}")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(
        r"^## \[Unreleased\]\s*(.*?)(?=^## \[|\Z)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    if match and re.search(r"^(?:### |[-*] )", match.group(1), re.MULTILINE):
        errors.append(
            "CHANGELOG.md still has Unreleased entries; move them into the "
            "tagged version section"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    errors = validate_release(args.tag)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"[OK] release metadata matches {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
