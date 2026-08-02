#!/usr/bin/env python3
"""Record and compare cold-import performance without touching runtime code."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
DEFAULT_PROFILES = {
    "public_cli_import": "import xscientist.cli",
    "project_import": "import ai_scientist.apps.project",
    "batch_import": "import ai_scientist.apps.batch",
    "daemon_import": "import ai_scientist.apps.daemon",
    "quality_pipeline_import": "import ai_scientist.utils.high_quality_pipeline",
}


class PerformanceRegressionError(ValueError):
    """Raised when benchmark inputs or results are invalid."""


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PerformanceRegressionError("cannot summarize an empty sample")
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _measurement_script(snippet: str) -> str:
    return (
        "import json, resource, time\n"
        "started = time.perf_counter()\n"
        f"exec({snippet!r}, {{}})\n"
        "elapsed = time.perf_counter() - started\n"
        "rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n"
        "print(json.dumps({'seconds': elapsed, 'max_rss_native': rss}))\n"
    )


def _measure_once(snippet: str, *, cwd: Path) -> dict[str, float]:
    completed = subprocess.run(
        [sys.executable, "-c", _measurement_script(snippet)],
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PerformanceRegressionError(
            f"profile failed for {snippet!r}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        return {
            "seconds": float(payload["seconds"]),
            "max_rss_native": float(payload["max_rss_native"]),
        }
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PerformanceRegressionError(
            f"invalid measurement output for {snippet!r}"
        ) from exc


def record_profiles(
    *,
    cwd: str | Path,
    repeats: int = 7,
    profiles: dict[str, str] | None = None,
) -> dict[str, Any]:
    if repeats < 3:
        raise PerformanceRegressionError("at least three repeats are required")
    root = Path(cwd).expanduser().resolve()
    results: dict[str, Any] = {}
    for name, snippet in dict(profiles or DEFAULT_PROFILES).items():
        samples = [_measure_once(snippet, cwd=root) for _ in range(repeats)]
        seconds = [item["seconds"] for item in samples]
        rss = [item["max_rss_native"] for item in samples]
        results[name] = {
            "snippet": snippet,
            "repeat_count": repeats,
            "median_seconds": statistics.median(seconds),
            "p90_seconds": _percentile(seconds, 0.90),
            "median_max_rss_native": statistics.median(rss),
            "samples": samples,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_epoch": time.time(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "rss_unit": "bytes" if sys.platform == "darwin" else "kibibytes",
        },
        "profiles": results,
    }


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_time_regression: float = 0.05,
    maximum_rss_regression: float = 0.05,
    minimum_time_slack_seconds: float = 0.005,
    minimum_rss_slack_native: float = 2048.0,
) -> dict[str, Any]:
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise PerformanceRegressionError("baseline schema is unsupported")
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise PerformanceRegressionError("candidate schema is unsupported")
    if baseline.get("environment") != candidate.get("environment"):
        raise PerformanceRegressionError(
            "baseline and candidate must use the same Python and platform environment"
        )
    baseline_profiles = baseline.get("profiles") or {}
    candidate_profiles = candidate.get("profiles") or {}
    if set(baseline_profiles) != set(candidate_profiles):
        raise PerformanceRegressionError("profile sets do not match")

    failures: list[str] = []
    profiles: dict[str, Any] = {}
    for name in sorted(baseline_profiles):
        before = baseline_profiles[name]
        after = candidate_profiles[name]
        before_time = float(before["median_seconds"])
        after_time = float(after["median_seconds"])
        before_rss = float(before["median_max_rss_native"])
        after_rss = float(after["median_max_rss_native"])
        allowed_time = before_time * (1.0 + maximum_time_regression) + (
            minimum_time_slack_seconds
        )
        allowed_rss = before_rss * (1.0 + maximum_rss_regression) + (
            minimum_rss_slack_native
        )
        time_passed = after_time <= allowed_time
        rss_passed = after_rss <= allowed_rss
        if not time_passed:
            failures.append(f"{name}:time")
        if not rss_passed:
            failures.append(f"{name}:rss")
        profiles[name] = {
            "baseline_seconds": before_time,
            "candidate_seconds": after_time,
            "allowed_seconds": allowed_time,
            "baseline_max_rss_native": before_rss,
            "candidate_max_rss_native": after_rss,
            "allowed_max_rss_native": allowed_rss,
            "time_passed": time_passed,
            "rss_passed": rss_passed,
        }
    return {"passed": not failures, "failures": failures, "profiles": profiles}


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise PerformanceRegressionError("benchmark JSON must be an object")
    return payload


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--output", required=True)
    record_parser.add_argument("--repeats", type=int, default=7)
    record_parser.add_argument("--cwd", default=".")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--max-time-regression", type=float, default=0.05)
    compare_parser.add_argument("--max-rss-regression", type=float, default=0.05)
    compare_parser.add_argument("--time-slack-ms", type=float, default=5.0)
    compare_parser.add_argument("--rss-slack-native", type=float, default=2048.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    if parsed.command == "record":
        result = record_profiles(cwd=parsed.cwd, repeats=parsed.repeats)
        _save_json(parsed.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    comparison = compare_results(
        _load_json(parsed.baseline),
        _load_json(parsed.candidate),
        maximum_time_regression=parsed.max_time_regression,
        maximum_rss_regression=parsed.max_rss_regression,
        minimum_time_slack_seconds=parsed.time_slack_ms / 1000.0,
        minimum_rss_slack_native=parsed.rss_slack_native,
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
