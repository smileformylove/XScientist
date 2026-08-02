#!/usr/bin/env python3
"""Build distributions after removing only known generated build state."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATED_TARGETS = (
    REPOSITORY_ROOT / "build",
    REPOSITORY_ROOT / "compat" / "xscientist.egg-info",
)


def clean_generated_build_state() -> list[str]:
    removed: list[str] = []
    allowed = {
        (REPOSITORY_ROOT / "build").resolve(),
        (REPOSITORY_ROOT / "compat" / "xscientist.egg-info").resolve(),
    }
    for target in GENERATED_TARGETS:
        resolved = target.resolve()
        if resolved not in allowed:
            raise ValueError(f"refusing unexpected cleanup target: {resolved}")
        if not target.exists():
            continue
        shutil.rmtree(target)
        removed.append(str(target.relative_to(REPOSITORY_ROOT)))
    return removed


def build_distribution(*, outdir: str | Path, isolation: bool = True) -> int:
    clean_generated_build_state()
    command = [
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--wheel",
        "--outdir",
        str(Path(outdir).expanduser().resolve()),
    ]
    if not isolation:
        command.append("--no-isolation")
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="dist")
    parser.add_argument("--no-isolation", action="store_true")
    parsed = parser.parse_args(argv)
    return build_distribution(outdir=parsed.outdir, isolation=not parsed.no_isolation)


if __name__ == "__main__":
    raise SystemExit(main())
