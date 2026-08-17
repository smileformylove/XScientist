"""Reproducible, provider-free first-run usability benchmark."""

from __future__ import annotations

import platform
import tempfile
import time
from pathlib import Path
from typing import Any

from ._version import __version__
from .demo import create_autopilot_demo
from .workspace_status import build_workspace_status


def benchmark_first_run(
    workspace: str | Path | None = None,
    *,
    profile: str = "balanced",
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Measure the deterministic offline journey from empty dir to status."""

    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be greater than zero")
    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="xscientist-first-run-")
        root = Path(temporary.name) / "study"
    else:
        root = Path(workspace).expanduser().resolve()
    started = time.perf_counter()
    try:
        demo = create_autopilot_demo(root, profile=profile, language="en")
        status = build_workspace_status(root, language="en")
        duration = time.perf_counter() - started
        threshold_passed = max_seconds is None or duration <= max_seconds
        return {
            "schema": "xscientist.first-run-benchmark.v1",
            "ok": bool(status["ok"] and threshold_passed),
            "version": __version__,
            "runtime": {
                "python": platform.python_version(),
                "system": platform.system().lower(),
            },
            "profile": profile,
            "duration_seconds": round(duration, 3),
            "max_seconds": max_seconds,
            "threshold_passed": threshold_passed,
            "network_used": False,
            "provider_used": False,
            "model_cost_usd": 0.0,
            "research": {
                "dag_nodes": demo["dag"]["nodes"],
                "dag_relations": demo["dag"]["relations"],
                "closure": demo["dag"]["closure"],
                "run_started": status["run"]["started"],
                "budget_available": status["budget"]["available"],
                "next_step": (status["next_steps"] or [{}])[0].get("code"),
            },
            "workspace_retained": workspace is not None,
            "host_paths_disclosed": False,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = ["benchmark_first_run"]
