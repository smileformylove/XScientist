from __future__ import annotations

"""Benchmark-gated promotion for agent, prompt, tool, and search changes."""

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    artifact_path,
    save_contract_artifact,
    update_pipeline_artifact,
)

SCHEMA_VERSION = 1
COMPONENT_TYPES = {
    "agent_scaffold",
    "prompt",
    "tool",
    "model_routing",
    "search_policy",
    "verification_policy",
}
DEFAULT_METRIC_SPECS = {
    "objective_score": {
        "direction": "higher",
        "max_mean_regression": 0.0,
        "max_confident_regression": 0.02,
        "minimum_improvement": 0.01,
        "promotion_signal": True,
    },
    "reproducibility_rate": {
        "direction": "higher",
        "max_mean_regression": 0.0,
        "max_confident_regression": 0.02,
        "minimum_improvement": 0.0,
        "promotion_signal": False,
    },
    "false_discovery_rate": {
        "direction": "lower",
        "max_mean_regression": 0.0,
        "max_confident_regression": 0.01,
        "minimum_improvement": 0.0,
        "promotion_signal": False,
    },
    "cost_per_task": {
        "direction": "lower",
        "max_mean_regression": 0.10,
        "max_confident_regression": 0.15,
        "minimum_improvement": 0.0,
        "promotion_signal": False,
    },
    "latency_seconds": {
        "direction": "lower",
        "max_mean_regression": 0.20,
        "max_confident_regression": 0.25,
        "minimum_improvement": 0.0,
        "promotion_signal": False,
    },
}
DEFAULT_POLICY = {
    "minimum_hidden_tasks": 5,
    "confidence_z": 1.96,
    "minimum_canary_observations": 20,
    "maximum_canary_error_rate_delta": 0.0,
    "maximum_canary_quality_regression": 0.01,
    "require_safety_pass": True,
    "require_integrity_pass": True,
    "require_reproducibility_pass": True,
    "metric_specs": DEFAULT_METRIC_SPECS,
}


class EvolutionGateError(ValueError):
    """Raised when an evolution candidate bypasses a required control."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_content_hash(value: Any) -> bool:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        return False
    try:
        int(text.split(":", 1)[1], 16)
    except ValueError:
        return False
    return True


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _candidate_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"created_at", "candidate_hash", "status"}
    }


def build_evolution_candidate(
    *,
    candidate_id: str,
    component_type: str,
    base_version: str,
    candidate_version: str,
    base_artifact_hash: str,
    candidate_artifact_hash: str,
    rollback_ref: str,
    proposed_by: str,
    change_summary: str,
) -> dict[str, Any]:
    """Describe a shadow-only candidate without applying it to production."""

    component = str(component_type or "").strip()
    if component not in COMPONENT_TYPES:
        raise EvolutionGateError(
            f"unsupported component_type={component!r}; expected {sorted(COMPONENT_TYPES)}"
        )
    if not _is_content_hash(base_artifact_hash) or not _is_content_hash(
        candidate_artifact_hash
    ):
        raise EvolutionGateError("base and candidate artifacts require sha256 hashes")
    if base_artifact_hash == candidate_artifact_hash:
        raise EvolutionGateError("candidate artifact is identical to the baseline")
    required_strings = {
        "candidate_id": candidate_id,
        "base_version": base_version,
        "candidate_version": candidate_version,
        "rollback_ref": rollback_ref,
        "proposed_by": proposed_by,
        "change_summary": change_summary,
    }
    missing = [
        name for name, value in required_strings.items() if not str(value or "").strip()
    ]
    if missing:
        raise EvolutionGateError("missing candidate fields: " + ", ".join(missing))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": str(candidate_id).strip(),
        "component_type": component,
        "base_version": str(base_version).strip(),
        "candidate_version": str(candidate_version).strip(),
        "base_artifact_hash": base_artifact_hash,
        "candidate_artifact_hash": candidate_artifact_hash,
        "rollback_ref": str(rollback_ref).strip(),
        "proposed_by": str(proposed_by).strip(),
        "change_summary": str(change_summary).strip(),
        "status": "shadow_only",
        "created_at": _now_iso(),
    }
    payload["candidate_hash"] = _canonical_hash(_candidate_core(payload))
    return payload


def validate_evolution_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("component_type") not in COMPONENT_TYPES:
        errors.append("component_type_invalid")
    if payload.get("status") != "shadow_only":
        errors.append("candidate_not_shadow_only")
    if not _is_content_hash(payload.get("base_artifact_hash")):
        errors.append("base_artifact_hash_invalid")
    if not _is_content_hash(payload.get("candidate_artifact_hash")):
        errors.append("candidate_artifact_hash_invalid")
    if payload.get("base_artifact_hash") == payload.get("candidate_artifact_hash"):
        errors.append("candidate_artifact_unchanged")
    for field in (
        "candidate_id",
        "base_version",
        "candidate_version",
        "proposed_by",
        "change_summary",
    ):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field}_missing")
    if not str(payload.get("rollback_ref") or "").strip():
        errors.append("rollback_ref_missing")
    if payload.get("candidate_hash") != _canonical_hash(_candidate_core(payload)):
        errors.append("candidate_hash_mismatch")
    return {"ok": not errors, "errors": errors}


def _merge_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_POLICY)
    incoming = deepcopy(policy or {})
    unknown = sorted(set(incoming) - set(DEFAULT_POLICY))
    if unknown:
        raise EvolutionGateError("unknown policy fields: " + ", ".join(unknown))
    metric_overrides = incoming.pop("metric_specs", {})
    if not isinstance(metric_overrides, dict):
        raise EvolutionGateError("metric_specs must be an object")
    resolved.update(incoming)
    resolved["metric_specs"] = deepcopy(DEFAULT_METRIC_SPECS)
    for metric, spec in metric_overrides.items():
        if not isinstance(spec, dict):
            raise EvolutionGateError(f"metric policy must be an object: {metric}")
        resolved["metric_specs"].setdefault(metric, {})
        resolved["metric_specs"][metric].update(spec or {})
    for gate in (
        "require_safety_pass",
        "require_integrity_pass",
        "require_reproducibility_pass",
    ):
        if not resolved.get(gate):
            raise EvolutionGateError(f"hard gate cannot be disabled: {gate}")
    stricter_minimums = {
        "minimum_hidden_tasks": DEFAULT_POLICY["minimum_hidden_tasks"],
        "confidence_z": DEFAULT_POLICY["confidence_z"],
        "minimum_canary_observations": DEFAULT_POLICY["minimum_canary_observations"],
    }
    for field, floor in stricter_minimums.items():
        value = _finite_number(resolved.get(field))
        if value is None or value < float(floor):
            raise EvolutionGateError(f"policy may not weaken {field} below {floor}")
    stricter_maximums = {
        "maximum_canary_error_rate_delta": DEFAULT_POLICY[
            "maximum_canary_error_rate_delta"
        ],
        "maximum_canary_quality_regression": DEFAULT_POLICY[
            "maximum_canary_quality_regression"
        ],
    }
    for field, ceiling in stricter_maximums.items():
        value = _finite_number(resolved.get(field))
        if value is None or value > float(ceiling):
            raise EvolutionGateError(f"policy may not weaken {field} above {ceiling}")
    for metric, spec in resolved["metric_specs"].items():
        baseline_spec = DEFAULT_METRIC_SPECS.get(metric)
        if spec.get("direction") not in {"higher", "lower"}:
            raise EvolutionGateError(f"invalid direction for metric {metric}")
        if baseline_spec and spec.get("direction") != baseline_spec["direction"]:
            raise EvolutionGateError(
                f"policy may not reverse metric direction: {metric}"
            )
        for field in ("max_mean_regression", "max_confident_regression"):
            value = _finite_number(spec.get(field))
            ceiling = float((baseline_spec or {}).get(field, 0.0))
            if value is None or value > ceiling:
                raise EvolutionGateError(
                    f"policy may not weaken {metric}.{field} above {ceiling}"
                )
        improvement = _finite_number(spec.get("minimum_improvement"))
        improvement_floor = float((baseline_spec or {}).get("minimum_improvement", 0.0))
        if improvement is None or improvement < improvement_floor:
            raise EvolutionGateError(
                f"policy may not weaken {metric}.minimum_improvement "
                f"below {improvement_floor}"
            )
        expected_signal = bool((baseline_spec or {}).get("promotion_signal", False))
        if bool(spec.get("promotion_signal")) != expected_signal:
            raise EvolutionGateError(
                f"policy may not change promotion_signal for metric {metric}"
            )
    return resolved


def _signed_relative_delta(
    baseline: float, candidate: float, *, direction: str
) -> float:
    denominator = max(abs(baseline), 1e-12)
    if direction == "higher":
        return (candidate - baseline) / denominator
    if direction == "lower":
        return (baseline - candidate) / denominator
    raise EvolutionGateError(f"unsupported metric direction={direction!r}")


def _metric_evaluation(
    samples: list[dict[str, Any]],
    *,
    metric: str,
    spec: dict[str, Any],
    confidence_z: float,
) -> dict[str, Any]:
    deltas: list[float] = []
    missing_tasks: list[str] = []
    for sample in samples:
        baseline = _finite_number((sample.get("baseline") or {}).get(metric))
        candidate = _finite_number((sample.get("candidate") or {}).get(metric))
        if baseline is None or candidate is None:
            missing_tasks.append(str(sample.get("task_id") or "unknown"))
            continue
        deltas.append(
            _signed_relative_delta(
                baseline,
                candidate,
                direction=str(spec.get("direction") or "higher"),
            )
        )
    average = mean(deltas) if deltas else float("-inf")
    standard_error = (
        stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) >= 2 else float("inf")
    )
    lower_bound = average - confidence_z * standard_error
    max_mean_regression = float(spec.get("max_mean_regression") or 0.0)
    max_confident_regression = float(spec.get("max_confident_regression") or 0.0)
    complete = len(deltas) == len(samples) and bool(samples)
    passed = (
        complete
        and average >= -max_mean_regression
        and lower_bound >= -max_confident_regression
    )
    improvement = (
        passed
        and lower_bound >= float(spec.get("minimum_improvement") or 0.0)
        and bool(spec.get("promotion_signal"))
    )
    return {
        "metric": metric,
        "direction": spec.get("direction"),
        "sample_count": len(deltas),
        "missing_tasks": missing_tasks,
        "mean_relative_improvement": (
            round(average, 6) if math.isfinite(average) else None
        ),
        "standard_error": (
            round(standard_error, 6) if math.isfinite(standard_error) else None
        ),
        "confidence_lower_bound": (
            round(lower_bound, 6) if math.isfinite(lower_bound) else None
        ),
        "passed": passed,
        "promotion_signal_passed": improvement,
    }


def _criterion(
    criterion_id: str, passed: bool, detail: str, *, required: bool = True
) -> dict[str, Any]:
    return {
        "id": criterion_id,
        "passed": bool(passed),
        "required": bool(required),
        "detail": detail,
    }


def _gate_hash_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Return every decision-bearing field covered by the gate hash."""

    return {
        "schema_version": report.get("schema_version"),
        "candidate": report.get("candidate"),
        "policy": report.get("policy"),
        "benchmark_hash": report.get("benchmark_hash"),
        "benchmark_task_count": report.get("benchmark_task_count"),
        "decision": report.get("decision"),
        "production_promotion_allowed": report.get("production_promotion_allowed"),
        "criteria": report.get("criteria"),
        "required_failures": report.get("required_failures"),
        "metric_results": report.get("metric_results"),
    }


def build_evolution_gate(
    candidate: dict[str, Any],
    benchmark_samples: Iterable[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether a shadow candidate may advance to a bounded canary."""

    resolved_policy = _merge_policy(policy)
    samples = [dict(item) for item in benchmark_samples if isinstance(item, dict)]
    candidate_check = validate_evolution_candidate(candidate)
    task_ids = [str(item.get("task_id") or "").strip() for item in samples]
    hidden = bool(samples) and all(item.get("split") == "hidden" for item in samples)
    unique_tasks = (
        bool(task_ids) and all(task_ids) and len(task_ids) == len(set(task_ids))
    )
    minimum_tasks = max(int(resolved_policy.get("minimum_hidden_tasks") or 1), 1)
    safety_passed = all(bool(item.get("safety_pass")) for item in samples)
    integrity_passed = all(bool(item.get("integrity_pass")) for item in samples)
    reproducibility_passed = all(
        bool(item.get("reproducibility_pass")) for item in samples
    )
    metric_results = {
        metric: _metric_evaluation(
            samples,
            metric=metric,
            spec=spec,
            confidence_z=float(resolved_policy.get("confidence_z") or 1.96),
        )
        for metric, spec in resolved_policy["metric_specs"].items()
    }
    metrics_passed = bool(metric_results) and all(
        item["passed"] for item in metric_results.values()
    )
    promotion_signal = any(
        item["promotion_signal_passed"] for item in metric_results.values()
    )
    criteria = [
        _criterion(
            "candidate_integrity",
            candidate_check["ok"],
            ", ".join(candidate_check["errors"]) or "candidate hash valid",
        ),
        _criterion(
            "hidden_benchmark",
            hidden,
            "all benchmark samples must use split=hidden",
        ),
        _criterion(
            "minimum_task_count",
            len(samples) >= minimum_tasks,
            f"tasks={len(samples)}/{minimum_tasks}",
        ),
        _criterion(
            "paired_unique_tasks",
            unique_tasks,
            "task IDs are present and unique",
        ),
        _criterion(
            "safety_regression",
            safety_passed,
            "all safety checks pass",
        ),
        _criterion(
            "integrity_regression",
            integrity_passed,
            "all research-integrity checks pass",
        ),
        _criterion(
            "reproducibility_regression",
            reproducibility_passed,
            "all reproduction checks pass",
        ),
        _criterion(
            "metric_regression",
            metrics_passed,
            ", ".join(
                metric for metric, item in metric_results.items() if not item["passed"]
            )
            or "all metric confidence bounds pass",
        ),
        _criterion(
            "objective_improvement",
            promotion_signal,
            "at least one scientific objective improves beyond its confidence threshold",
        ),
        _criterion(
            "rollback_available",
            bool(str(candidate.get("rollback_ref") or "").strip()),
            str(candidate.get("rollback_ref") or "missing"),
        ),
    ]
    required_failures = [
        item["id"] for item in criteria if item["required"] and not item["passed"]
    ]
    decision = "promote_to_canary" if not required_failures else "hold"
    benchmark_hash = _canonical_hash(
        {"candidate_hash": candidate.get("candidate_hash"), "samples": samples}
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "candidate": deepcopy(candidate),
        "policy": resolved_policy,
        "benchmark_hash": benchmark_hash,
        "benchmark_task_count": len(samples),
        "decision": decision,
        "production_promotion_allowed": False,
        "criteria": criteria,
        "required_failures": required_failures,
        "metric_results": metric_results,
    }
    report["gate_hash"] = _canonical_hash(_gate_hash_payload(report))
    return report


def approve_production_promotion(
    gate_report: dict[str, Any],
    canary_report: dict[str, Any],
    *,
    approver_id: str,
) -> dict[str, Any]:
    """Authorize production only after a successful, rollback-tested canary."""

    policy = gate_report.get("policy") or {}
    candidate = gate_report.get("candidate") or {}
    approver = str(approver_id or "").strip()
    candidate_check = validate_evolution_candidate(candidate)
    gate_hash_valid = gate_report.get("gate_hash") == _canonical_hash(
        _gate_hash_payload(gate_report)
    )
    observations = _finite_number(canary_report.get("observation_count"))
    error_rate_delta = _finite_number(canary_report.get("error_rate_delta"))
    quality_delta = _finite_number(canary_report.get("quality_delta"))
    criteria = [
        _criterion(
            "shadow_gate_integrity",
            gate_hash_valid and candidate_check["ok"],
            (
                "gate and candidate hashes valid"
                if gate_hash_valid and candidate_check["ok"]
                else "shadow gate or candidate was modified after evaluation"
            ),
        ),
        _criterion(
            "shadow_gate",
            gate_report.get("decision") == "promote_to_canary",
            str(gate_report.get("decision") or "missing"),
        ),
        _criterion(
            "candidate_hash_binding",
            canary_report.get("candidate_hash") == candidate.get("candidate_hash"),
            "canary runs the exact shadow-tested candidate",
        ),
        _criterion(
            "canary_status",
            canary_report.get("status") == "passed",
            str(canary_report.get("status") or "missing"),
        ),
        _criterion(
            "canary_observations",
            observations is not None
            and observations.is_integer()
            and observations >= int(policy.get("minimum_canary_observations") or 20),
            (
                f"observations={observations}/"
                f"{int(policy.get('minimum_canary_observations') or 20)}"
            ),
        ),
        _criterion(
            "canary_error_rate",
            error_rate_delta is not None
            and error_rate_delta
            <= float(policy.get("maximum_canary_error_rate_delta") or 0.0),
            f"delta={error_rate_delta}",
        ),
        _criterion(
            "canary_quality",
            quality_delta is not None
            and quality_delta
            >= -float(policy.get("maximum_canary_quality_regression") or 0.01),
            f"delta={quality_delta}",
        ),
        _criterion(
            "incident_free",
            not list(canary_report.get("incidents") or []),
            f"incidents={len(canary_report.get('incidents') or [])}",
        ),
        _criterion(
            "rollback_tested",
            bool(canary_report.get("rollback_tested")),
            "rollback path was exercised during canary",
        ),
        _criterion(
            "independent_approval",
            bool(approver) and approver != str(candidate.get("proposed_by") or ""),
            f"approver={approver or 'missing'}",
        ),
    ]
    failures = [item["id"] for item in criteria if not item["passed"]]
    decision = "approved" if not failures else "blocked"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "candidate": deepcopy(candidate),
        "shadow_gate_hash": gate_report.get("gate_hash"),
        "canary_report": deepcopy(canary_report),
        "approver_id": approver or None,
        "decision": decision,
        "production_promotion_allowed": decision == "approved",
        "criteria": criteria,
        "required_failures": failures,
        "rollback_ref": candidate.get("rollback_ref"),
    }
    payload["promotion_hash"] = _canonical_hash(
        {
            "candidate_hash": candidate.get("candidate_hash"),
            "shadow_gate_hash": gate_report.get("gate_hash"),
            "canary_report": canary_report,
            "approver_id": approver,
            "decision": decision,
        }
    )
    return payload


def save_evolution_gate(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    producer: str,
) -> str:
    output_path = save_contract_artifact(
        project_root,
        "evolution_gate",
        payload,
        producer=producer,
        depends_on=["self_evolution", "stage_standards"],
        warnings=list(payload.get("required_failures") or []),
    )
    accepted = payload.get("decision") in {"promote_to_canary", "approved"}
    if not accepted:
        update_pipeline_artifact(
            project_root,
            "evolution_gate",
            status="blocked",
            producer=producer,
            depends_on=["self_evolution", "stage_standards"],
            warnings=list(payload.get("required_failures") or []),
            recovery_hint=(
                "Keep the candidate shadow-only, repair benchmark regressions, and rerun."
            ),
        )
    history_path = artifact_path(project_root, "evolution_gate").with_name(
        "evolution_gate_history.jsonl"
    )
    append_jsonl_artifact(
        history_path,
        {
            "generated_at": payload.get("generated_at"),
            "candidate_id": (payload.get("candidate") or {}).get("candidate_id"),
            "candidate_hash": (payload.get("candidate") or {}).get("candidate_hash"),
            "decision": payload.get("decision"),
            "production_promotion_allowed": bool(
                payload.get("production_promotion_allowed")
            ),
            "gate_hash": payload.get("gate_hash"),
            "promotion_hash": payload.get("promotion_hash"),
            "required_failures": list(payload.get("required_failures") or []),
        },
    )
    return output_path


__all__ = [
    "EvolutionGateError",
    "approve_production_promotion",
    "build_evolution_candidate",
    "build_evolution_gate",
    "save_evolution_gate",
    "validate_evolution_candidate",
]
