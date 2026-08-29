"""Lightweight validation of deterministic-evaluation hash bindings.

This module deliberately has no numerical dependencies. Protocol and artifact
consumers can verify an existing evaluation receipt without importing NumPy;
running the evaluator itself still requires the numerical optional extras.
"""

from __future__ import annotations

import hashlib
import json
import math
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


def evaluation_comparison_contract(report: Any) -> dict[str, Any] | None:
    """Return the evaluator and per-dataset identities required for comparison."""

    binding = evaluation_hash_binding(report)
    if binding is None or not isinstance(report, Mapping):
        return None
    if report.get("schema_version") != "deterministic_evaluation.v2":
        return None
    if (
        report.get("verification_scope") != "artifact_internal_consistency"
        or report.get("ground_truth_authority") != "research_agent_artifact"
    ):
        return None
    if report.get("comparison_ready") is not True:
        return None
    selected_metric = report.get("selected_metric")
    datasets = report.get("datasets")
    if not isinstance(selected_metric, str) or not selected_metric:
        return None
    if not isinstance(datasets, list) or not datasets:
        return None
    metric_payload = report.get("metric")
    metric_names = (
        metric_payload.get("metric_names")
        if isinstance(metric_payload, Mapping)
        else None
    )
    if not isinstance(metric_names, list) or len(metric_names) != 1:
        return None
    metric_entry = metric_names[0]
    if (
        not isinstance(metric_entry, Mapping)
        or metric_entry.get("metric_name") != selected_metric
        or not isinstance(metric_entry.get("data"), list)
    ):
        return None
    metric_dataset_names = [
        item.get("dataset_name")
        for item in metric_entry["data"]
        if isinstance(item, Mapping)
    ]
    if len(metric_dataset_names) != len(metric_entry["data"]):
        return None
    if len(metric_dataset_names) != len(set(metric_dataset_names)):
        return None
    metric_values: dict[str, float] = {}
    for metric_item in metric_entry["data"]:
        final_value = metric_item.get("final_value")
        best_value = metric_item.get("best_value")
        if (
            isinstance(final_value, bool)
            or not isinstance(final_value, (int, float))
            or not math.isfinite(float(final_value))
            or best_value != final_value
        ):
            return None
        metric_values[metric_item["dataset_name"]] = float(final_value)

    identities: dict[str, dict[str, Any]] = {}
    total_samples = 0
    for item in datasets:
        if not isinstance(item, Mapping):
            return None
        name = item.get("dataset_name")
        target_hash = item.get("target_sha256")
        split_identity = item.get("split_identity")
        samples = item.get("samples")
        target_dtype = item.get("target_dtype")
        target_shape = item.get("target_shape")
        sample_ids_hash = item.get("sample_ids_sha256")
        input_fingerprints_hash = item.get("input_fingerprints_sha256")
        input_identity_source = item.get("input_identity_source")
        value = item.get("value")
        if (
            not isinstance(name, str)
            or not name
            or name in identities
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(target_hash)) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(split_identity)) is None
            or isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples <= 0
            or not isinstance(target_dtype, str)
            or not target_dtype
            or not isinstance(target_shape, list)
            or len(target_shape) != 1
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in target_shape
            )
            or math.prod(target_shape) != samples
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(sample_ids_hash)) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(input_fingerprints_hash))
            is None
            or input_identity_source != "evaluator_computed_from_evaluation_inputs.v1"
            or item.get("comparison_ready") is not True
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or metric_values.get(name) != float(value)
        ):
            return None
        expected_split = _canonical_hash(
            {
                "dataset_path": name,
                "target_sha256": target_hash,
                "sample_ids_sha256": sample_ids_hash,
                "input_fingerprints_sha256": input_fingerprints_hash,
                "input_identity_source": input_identity_source,
                "samples": samples,
            }
        )
        if expected_split != split_identity:
            return None
        identities[name] = {
            "target_sha256": target_hash,
            "split_identity": split_identity,
            "samples": samples,
            "target_dtype": target_dtype,
            "target_shape": list(target_shape),
            "sample_ids_sha256": sample_ids_hash,
            "input_fingerprints_sha256": input_fingerprints_hash,
            "input_identity_source": input_identity_source,
        }
        total_samples += samples
    if set(metric_dataset_names) != set(identities):
        return None
    sample_count = report.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count != total_samples
    ):
        return None
    return {
        "schema_version": binding["schema_version"],
        "evaluator_version": binding["evaluator_version"],
        "evaluator_hash": binding["evaluator_hash"],
        "selected_metric": selected_metric,
        "verification_scope": report["verification_scope"],
        "ground_truth_authority": report["ground_truth_authority"],
        "datasets": identities,
    }


__all__ = ["evaluation_comparison_contract", "evaluation_hash_binding"]
