"""Lightweight validation of deterministic-evaluation hash bindings.

This module deliberately has no numerical dependencies. Protocol and artifact
consumers can verify an existing evaluation receipt without importing NumPy;
running the evaluator itself still requires the numerical optional extras.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def evaluation_hash_binding(report: Any) -> dict[str, Any] | None:
    """Return the stable subset that binds a verified evaluation into a node hash."""

    if (
        not isinstance(report, Mapping)
        or report.get("status") != "verified"
        or report.get("trust_tier") != "deterministic_verified"
    ):
        return None
    report_without_result_hash = dict(report)
    recorded_result_hash = report_without_result_hash.pop("result_hash", None)
    try:
        expected_result_hash = _canonical_hash(report_without_result_hash)
    except (TypeError, ValueError):
        return None
    if not recorded_result_hash or recorded_result_hash != expected_result_hash:
        return None
    input_info = report.get("input")
    if not isinstance(input_info, Mapping):
        return None
    required = {
        "schema_version": report.get("schema_version"),
        "evaluator_version": report.get("evaluator_version"),
        "evaluator_hash": report.get("evaluator_hash"),
        "input_hash": input_info.get("sha256"),
        "result_hash": recorded_result_hash,
    }
    if not all(required.values()):
        return None
    for key in ("evaluator_hash", "input_hash", "result_hash"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(required[key])):
            return None
    return required


__all__ = ["evaluation_hash_binding"]
