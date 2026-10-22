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
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

# ARA_METRIC={...} JSON marker — the canonical, machine-friendly format.
# See SPEC.md §8 for the contract.
_MARKER_RE = re.compile(r"ARA_METRIC\s*=\s*(\{.*\})\s*$", re.MULTILINE)

# Fallback: "metric: 0.42" style trailing line. Kept for legacy scripts.
_TAIL_RE = re.compile(
    r"^\s*metric\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def numeric_metric_fraction(value: Any) -> tuple[Fraction | None, str | None]:
    """Return an exact base-10 value and a stable error category."""
    if isinstance(value, bool):
        return None, "non_numeric_metric"
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None, "non_numeric_metric"
    if not decimal_value.is_finite():
        return None, "non_finite_metric"
    return Fraction(decimal_value), None


def validate_metric_tolerance(tolerance: Any) -> Fraction:
    """Validate an absolute metric tolerance without losing precision."""
    value, error = numeric_metric_fraction(tolerance)
    if error is not None or value is None or value < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    return value


def _serializable_delta(delta: Fraction) -> int | float | str:
    """Keep exact integer deltas and avoid emitting JSON Infinity."""
    if delta.denominator == 1:
        return delta.numerator
    try:
        as_float = float(delta)
    except OverflowError:
        return f"{delta.numerator}/{delta.denominator}"
    if math.isfinite(as_float):
        return as_float
    return f"{delta.numerator}/{delta.denominator}"


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
            payload = json.loads(
                match.group(1),
                parse_constant=_reject_json_constant,
            )
            if isinstance(payload, dict) and "value" in payload:
                return {"available": True, **payload}
        except (json.JSONDecodeError, ValueError):
            pass

    tail = None
    for candidate in _TAIL_RE.finditer(stdout or ""):
        tail = candidate
    if tail:
        try:
            return {
                "available": True,
                "value": float(tail.group(1)),
                "source": "text_tail",
            }
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
      - ``metric_name_mismatch`` — the two named metrics have different names.
      - ``metric_direction_mismatch`` — the two metrics disagree on direction.
      - ``non_numeric_metric``  — value present but not float-coercible.
      - ``non_finite_metric``   — a numeric value is NaN or infinite.
      - ``compared``            — good comparison; ``within_tolerance`` set.
    """
    exact_tolerance = validate_metric_tolerance(tolerance)
    if not fresh.get("available"):
        return {"status": "no_metric_parsed", "delta": None, "within_tolerance": None}
    recorded_value = None
    recorded_name = None
    if isinstance(recorded, dict):
        recorded_value = recorded.get("value")
        recorded_name = recorded.get("name")
    elif isinstance(recorded, (int, float)):
        recorded_value = recorded
    fresh_value = fresh.get("value")
    if recorded_value is None or fresh_value is None:
        return {
            "status": "missing_recorded_value",
            "delta": None,
            "within_tolerance": None,
        }
    fresh_name = fresh.get("name")
    if (
        recorded_name not in (None, "")
        and fresh_name not in (None, "")
        and str(recorded_name) != str(fresh_name)
    ):
        return {
            "status": "metric_name_mismatch",
            "recorded_name": str(recorded_name),
            "fresh_name": str(fresh_name),
            "delta": None,
            "within_tolerance": None,
        }
    recorded_maximize = recorded.get("maximize") if isinstance(recorded, dict) else None
    fresh_maximize = fresh.get("maximize")
    if (
        recorded_maximize is not None
        and fresh_maximize is not None
        and (
            type(recorded_maximize) is not bool
            or type(fresh_maximize) is not bool
            or recorded_maximize is not fresh_maximize
        )
    ):
        return {
            "status": "metric_direction_mismatch",
            "recorded_maximize": recorded_maximize,
            "fresh_maximize": fresh_maximize,
            "delta": None,
            "within_tolerance": None,
        }
    recorded_exact, recorded_error = numeric_metric_fraction(recorded_value)
    fresh_exact, fresh_error = numeric_metric_fraction(fresh_value)
    error = recorded_error or fresh_error
    if error is not None or recorded_exact is None or fresh_exact is None:
        return {"status": error, "delta": None, "within_tolerance": None}
    exact_delta = fresh_exact - recorded_exact
    return {
        "status": "compared",
        "recorded_value": recorded_value,
        "fresh_value": fresh_value,
        "delta": _serializable_delta(exact_delta),
        "within_tolerance": abs(exact_delta) <= exact_tolerance,
    }
