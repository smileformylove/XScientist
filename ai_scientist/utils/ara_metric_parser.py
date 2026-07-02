"""Metric extraction + comparison helpers for ARA re-execution verification.

Extracted out of ``run_ara_fork.py`` so ``ara_reexec.py`` (a proper module)
doesn't have to reach into a top-level CLI script via importlib. Both the CLI
and the reexec module now import from here.

Also the natural home to grow multi-metric parsing when we need it — right
now we only support a single scalar metric per run, but re-executed
experiments routinely emit several.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ARA_METRIC={...} JSON marker — the canonical, machine-friendly format.
# See SPEC.md §8 for the contract.
_MARKER_RE = re.compile(r"ARA_METRIC\s*=\s*(\{.*\})\s*$", re.MULTILINE)

# Fallback: "metric: 0.42" style trailing line. Kept for legacy scripts.
_TAIL_RE = re.compile(
    r"^\s*metric\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def parse_metric_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract a machine-readable metric from a re-executed run.

    Strategy, in order:
      1. Any line matching ``ARA_METRIC={"name": "...", "value": 0.42}``
         (JSON after the equals sign). Freshest match wins so mid-run
         printouts don't outrank the final one.
      2. A trailing line of the form ``metric: <float>`` (case-insensitive).

    If neither matches we return ``{"available": False}`` so the caller can
    downgrade gracefully to a soft comparison. We never raise on malformed
    input — a broken marker is the same as no marker.
    """
    match = None
    for candidate in _MARKER_RE.finditer(stdout or ""):
        match = candidate  # keep the last one
    if match:
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict) and "value" in payload:
                return {"available": True, **payload}
        except json.JSONDecodeError:
            pass

    tail = None
    for candidate in _TAIL_RE.finditer(stdout or ""):
        tail = candidate
    if tail:
        try:
            return {"available": True, "value": float(tail.group(1)), "source": "text_tail"}
        except ValueError:
            pass

    return {"available": False}


def compare_metrics(
    recorded: Any, fresh: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    """Diff a re-executed metric against the recorded one.

    Statuses:
      - ``no_metric_parsed``    — we couldn't pull a value out of stdout.
      - ``missing_recorded_value`` — the ARA had no metric to compare against.
      - ``non_numeric_metric``  — value present but not float-coercible.
      - ``compared``            — good comparison; ``within_tolerance`` set.
    """
    if not fresh.get("available"):
        return {"status": "no_metric_parsed", "delta": None, "within_tolerance": None}
    recorded_value = None
    if isinstance(recorded, dict):
        recorded_value = recorded.get("value")
    elif isinstance(recorded, (int, float)):
        recorded_value = recorded
    fresh_value = fresh.get("value")
    if recorded_value is None or fresh_value is None:
        return {"status": "missing_recorded_value", "delta": None, "within_tolerance": None}
    try:
        delta = float(fresh_value) - float(recorded_value)
    except (TypeError, ValueError):
        return {"status": "non_numeric_metric", "delta": None, "within_tolerance": None}
    return {
        "status": "compared",
        "recorded_value": recorded_value,
        "fresh_value": fresh_value,
        "delta": delta,
        "within_tolerance": abs(delta) <= tolerance,
    }
