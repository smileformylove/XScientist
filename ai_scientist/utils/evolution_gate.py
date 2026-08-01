from __future__ import annotations

"""Constitution-bound, evidence-gated promotion for autonomous system changes."""

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
from ai_scientist.utils.science_constitution import (
    assert_science_constitution_intact,
)

SCHEMA_VERSION = 2
MUTABLE_COMPONENT_SCOPES = {
    "agent_scaffold": "agents/",
    "prompt": "prompts/",
    "tool": "tools/",
    "model_routing": "model_routing/",
    "search_policy": "search/",
    "resource_allocation": "resource_allocation/",
    "failure_recovery": "failure_recovery/",
}
COMPONENT_TYPES = set(MUTABLE_COMPONENT_SCOPES)
PROTECTED_COMPONENT_TYPES = {
    "science_constitution",
    "epistemic_graph_history",
    "raw_evidence",
    "sealed_benchmarks",
    "evaluation_policy",
    "verification_policy",
    "identity_and_approval_rules",
    "safety_boundaries",
}
PROTECTED_SCOPE_TOKENS = tuple(sorted(PROTECTED_COMPONENT_TYPES))
RISK_TIERS = {"low", "moderate", "high"}
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
    "minimum_prospective_tasks": 1,
    "minimum_distinct_evaluator_stacks": 2,
    "minimum_ablation_effect": 0.005,
    "minimum_ablation_samples_per_dimension": 1,
    "confidence_z": 1.96,
    "minimum_canary_observations": 20,
    "minimum_real_research_projects": 3,
    "minimum_high_risk_human_approvers": 2,
    "maximum_canary_error_rate_delta": 0.0,
    "maximum_canary_quality_regression": 0.01,
    "require_safety_pass": True,
    "require_integrity_pass": True,
    "require_reproducibility_pass": True,
    "metric_specs": DEFAULT_METRIC_SPECS,
}
CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_id",
    "component_type",
    "constitution_hash",
    "base_version",
    "candidate_version",
    "base_artifact_hash",
    "candidate_artifact_hash",
    "rollback_ref",
    "proposed_by",
    "change_summary",
    "change_scope",
    "applicability_domains",
    "failure_taxonomy_refs",
    "ablation_dimensions",
    "provenance_hashes",
    "risk_tier",
    "status",
    "created_at",
    "candidate_hash",
}
GATE_FIELDS = {
    "schema_version",
    "generated_at",
    "constitution_hash",
    "candidate",
    "policy",
    "ablation_report",
    "benchmark_hash",
    "benchmark_samples",
    "benchmark_task_count",
    "benchmark_layer_counts",
    "decision",
    "production_promotion_allowed",
    "criteria",
    "required_failures",
    "metric_results",
    "gate_hash",
}
ROLLBACK_FIELDS = {
    "schema_version",
    "generated_at",
    "candidate_hash",
    "rollback_ref",
    "expected_artifact_hash",
    "restored_artifact_hash",
    "execution_log_hash",
    "executed_by",
    "trigger",
    "exercise_only",
    "status",
    "receipt_hash",
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


def _parse_utc_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _strings(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(item).strip() for item in (values or []) if str(item).strip()})


def _content_hashes(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(item) for item in (values or []) if _is_content_hash(item)})


def _candidate_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key != "candidate_hash"
    }


def build_evolution_candidate(
    *,
    constitution: dict[str, Any],
    candidate_id: str,
    component_type: str,
    base_version: str,
    candidate_version: str,
    base_artifact_hash: str,
    candidate_artifact_hash: str,
    rollback_ref: str,
    proposed_by: str,
    change_summary: str,
    change_scope: Iterable[str],
    applicability_domains: Iterable[str],
    failure_taxonomy_refs: Iterable[str],
    ablation_dimensions: Iterable[str],
    provenance_hashes: Iterable[str],
    risk_tier: str = "moderate",
) -> dict[str, Any]:
    """Describe a shadow-only candidate without applying it to production."""

    assert_science_constitution_intact(constitution)
    component = str(component_type or "").strip()
    if component in PROTECTED_COMPONENT_TYPES:
        raise EvolutionGateError(
            f"protected component may not self-evolve: {component}"
        )
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
    if str(base_version).strip() == str(candidate_version).strip():
        raise EvolutionGateError("candidate version must differ from the baseline")
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
    proposer = str(proposed_by).strip()
    if not proposer.startswith(("agent:", "service:", "human:")):
        raise EvolutionGateError(
            "proposed_by must use an agent:, service:, or human: identity namespace"
        )
    scope = _strings(change_scope)
    expected_prefix = MUTABLE_COMPONENT_SCOPES[component]
    if not scope or any(not item.startswith(expected_prefix) for item in scope):
        raise EvolutionGateError(
            f"change_scope for {component} must stay under {expected_prefix!r}"
        )
    if any(token in item.lower() for item in scope for token in PROTECTED_SCOPE_TOKENS):
        raise EvolutionGateError(
            "change_scope intersects a constitution-protected asset"
        )
    domains = _strings(applicability_domains)
    failure_refs = _strings(failure_taxonomy_refs)
    ablations = _strings(ablation_dimensions)
    provenance_values = _strings(provenance_hashes)
    provenance = _content_hashes(provenance_values)
    if not domains or not failure_refs or not ablations:
        raise EvolutionGateError(
            "applicability domains, failure taxonomy refs, and ablation dimensions are required"
        )
    if not provenance or len(provenance) != len(provenance_values):
        raise EvolutionGateError("all provenance references must be SHA-256 hashes")
    risk = str(risk_tier or "").strip()
    if risk not in RISK_TIERS:
        raise EvolutionGateError(f"unsupported risk_tier={risk!r}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": str(candidate_id).strip(),
        "component_type": component,
        "constitution_hash": constitution["constitution_hash"],
        "base_version": str(base_version).strip(),
        "candidate_version": str(candidate_version).strip(),
        "base_artifact_hash": base_artifact_hash,
        "candidate_artifact_hash": candidate_artifact_hash,
        "rollback_ref": str(rollback_ref).strip(),
        "proposed_by": proposer,
        "change_summary": str(change_summary).strip(),
        "change_scope": scope,
        "applicability_domains": domains,
        "failure_taxonomy_refs": failure_refs,
        "ablation_dimensions": ablations,
        "provenance_hashes": provenance,
        "risk_tier": risk,
        "status": "shadow_only",
        "created_at": _now_iso(),
    }
    payload["candidate_hash"] = _canonical_hash(_candidate_core(payload))
    return payload


def validate_evolution_candidate(
    payload: dict[str, Any],
    *,
    constitution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if set(payload) != CANDIDATE_FIELDS:
        errors.append("candidate_fields_invalid")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
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
    if payload.get("base_version") == payload.get("candidate_version"):
        errors.append("candidate_version_unchanged")
    if not _is_content_hash(payload.get("constitution_hash")):
        errors.append("constitution_hash_invalid")
    if constitution is not None:
        try:
            assert_science_constitution_intact(constitution)
        except ValueError:
            errors.append("constitution_invalid")
        if payload.get("constitution_hash") != constitution.get("constitution_hash"):
            errors.append("constitution_binding_mismatch")
    for field in (
        "candidate_id",
        "base_version",
        "candidate_version",
        "proposed_by",
        "change_summary",
    ):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field}_missing")
    if not str(payload.get("proposed_by") or "").startswith(
        ("agent:", "service:", "human:")
    ):
        errors.append("proposed_by_namespace_invalid")
    if _parse_utc_timestamp(payload.get("created_at")) is None:
        errors.append("created_at_invalid")
    if not str(payload.get("rollback_ref") or "").strip():
        errors.append("rollback_ref_missing")
    component = str(payload.get("component_type") or "")
    scope = _strings(payload.get("change_scope"))
    expected_prefix = MUTABLE_COMPONENT_SCOPES.get(component)
    if (
        not scope
        or expected_prefix is None
        or any(not item.startswith(expected_prefix) for item in scope)
    ):
        errors.append("change_scope_invalid")
    if any(token in item.lower() for item in scope for token in PROTECTED_SCOPE_TOKENS):
        errors.append("protected_scope_intersection")
    for field in (
        "applicability_domains",
        "failure_taxonomy_refs",
        "ablation_dimensions",
    ):
        if not _strings(payload.get(field)):
            errors.append(f"{field}_missing")
    provenance = payload.get("provenance_hashes")
    if (
        not isinstance(provenance, list)
        or not provenance
        or any(not _is_content_hash(item) for item in provenance)
    ):
        errors.append("provenance_hashes_invalid")
    if payload.get("risk_tier") not in RISK_TIERS:
        errors.append("risk_tier_invalid")
    if payload.get("candidate_hash") != _canonical_hash(_candidate_core(payload)):
        errors.append("candidate_hash_mismatch")
    return {"ok": not errors, "errors": errors}


def _ablation_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"generated_at", "report_hash"}
    }


def build_ablation_report(
    candidate: dict[str, Any],
    samples: Iterable[dict[str, Any]],
    *,
    minimum_effect: float = float(DEFAULT_POLICY["minimum_ablation_effect"]),
    minimum_samples_per_dimension: int = int(
        DEFAULT_POLICY["minimum_ablation_samples_per_dimension"]
    ),
) -> dict[str, Any]:
    """Attribute gains to every declared change dimension before shadow promotion."""

    candidate_check = validate_evolution_candidate(candidate)
    if not candidate_check["ok"]:
        raise EvolutionGateError(
            "candidate invalid: " + ", ".join(candidate_check["errors"])
        )
    effect_floor = _finite_number(minimum_effect)
    if effect_floor is None or effect_floor < float(
        DEFAULT_POLICY["minimum_ablation_effect"]
    ):
        raise EvolutionGateError("minimum ablation effect may not be weakened")
    if minimum_samples_per_dimension < int(
        DEFAULT_POLICY["minimum_ablation_samples_per_dimension"]
    ):
        raise EvolutionGateError("minimum ablation samples may not be weakened")
    rows = [deepcopy(item) for item in samples if isinstance(item, dict)]
    dimensions = _strings(candidate.get("ablation_dimensions"))
    dimension_results: dict[str, Any] = {}
    failures: list[str] = []
    for dimension in dimensions:
        selected = [item for item in rows if item.get("dimension") == dimension]
        task_ids = _strings(item.get("task_id") for item in selected)
        contributions: list[float] = []
        for item in selected:
            full_score = _finite_number(item.get("full_candidate_score"))
            ablated_score = _finite_number(item.get("ablated_score"))
            if (
                full_score is None
                or ablated_score is None
                or not _is_content_hash(item.get("full_run_hash"))
                or not _is_content_hash(item.get("ablated_run_hash"))
                or item.get("full_run_hash") == item.get("ablated_run_hash")
            ):
                continue
            denominator = max(abs(ablated_score), 1e-12)
            contributions.append((full_score - ablated_score) / denominator)
        enough = (
            len(selected) >= minimum_samples_per_dimension
            and len(contributions) == len(selected)
            and len(task_ids) == len(selected)
        )
        average = mean(contributions) if contributions else float("-inf")
        passed = enough and average >= effect_floor
        if not passed:
            failures.append(f"dimension:{dimension}")
        dimension_results[dimension] = {
            "sample_count": len(contributions),
            "mean_relative_contribution": (
                round(average, 6) if math.isfinite(average) else None
            ),
            "minimum_effect": effect_floor,
            "passed": passed,
        }
    unexpected_dimensions = sorted(
        {
            str(item.get("dimension") or "").strip()
            for item in rows
            if str(item.get("dimension") or "").strip() not in dimensions
        }
    )
    if unexpected_dimensions:
        failures.append("undeclared_dimensions")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "candidate_hash": candidate["candidate_hash"],
        "minimum_effect": effect_floor,
        "minimum_samples_per_dimension": minimum_samples_per_dimension,
        "samples": rows,
        "dimension_results": dimension_results,
        "unexpected_dimensions": unexpected_dimensions,
        "passed": not failures,
        "required_failures": failures,
    }
    payload["report_hash"] = _canonical_hash(_ablation_core(payload))
    return payload


def validate_ablation_report(
    payload: dict[str, Any] | None,
    *,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    report = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if report.get("candidate_hash") != candidate.get("candidate_hash"):
        errors.append("candidate_binding_mismatch")
    if report.get("report_hash") != _canonical_hash(_ablation_core(report)):
        errors.append("report_hash_mismatch")
    try:
        expected = build_ablation_report(
            candidate,
            report.get("samples") or [],
            minimum_effect=float(report.get("minimum_effect")),
            minimum_samples_per_dimension=int(
                report.get("minimum_samples_per_dimension")
            ),
        )
        if _ablation_core(report) != _ablation_core(expected):
            errors.append("report_semantics_mismatch")
    except (EvolutionGateError, TypeError, ValueError):
        errors.append("report_reconstruction_failed")
    return {"passed": not errors, "errors": errors}


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
        "minimum_prospective_tasks": DEFAULT_POLICY["minimum_prospective_tasks"],
        "minimum_distinct_evaluator_stacks": DEFAULT_POLICY[
            "minimum_distinct_evaluator_stacks"
        ],
        "minimum_ablation_effect": DEFAULT_POLICY["minimum_ablation_effect"],
        "minimum_ablation_samples_per_dimension": DEFAULT_POLICY[
            "minimum_ablation_samples_per_dimension"
        ],
        "confidence_z": DEFAULT_POLICY["confidence_z"],
        "minimum_canary_observations": DEFAULT_POLICY["minimum_canary_observations"],
        "minimum_real_research_projects": DEFAULT_POLICY[
            "minimum_real_research_projects"
        ],
        "minimum_high_risk_human_approvers": DEFAULT_POLICY[
            "minimum_high_risk_human_approvers"
        ],
    }
    for field, floor in stricter_minimums.items():
        value = _finite_number(resolved.get(field))
        if value is None or value < float(floor):
            raise EvolutionGateError(f"policy may not weaken {field} below {floor}")
    for field in (
        "minimum_hidden_tasks",
        "minimum_prospective_tasks",
        "minimum_distinct_evaluator_stacks",
        "minimum_ablation_samples_per_dimension",
        "minimum_canary_observations",
        "minimum_real_research_projects",
        "minimum_high_risk_human_approvers",
    ):
        if not isinstance(resolved.get(field), int) or isinstance(
            resolved.get(field), bool
        ):
            raise EvolutionGateError(f"policy field must be an integer: {field}")
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
        "constitution_hash": report.get("constitution_hash"),
        "candidate": report.get("candidate"),
        "policy": report.get("policy"),
        "ablation_report": report.get("ablation_report"),
        "benchmark_hash": report.get("benchmark_hash"),
        "benchmark_samples": report.get("benchmark_samples"),
        "benchmark_task_count": report.get("benchmark_task_count"),
        "benchmark_layer_counts": report.get("benchmark_layer_counts"),
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
    constitution: dict[str, Any],
    ablation_report: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether a shadow candidate may advance to a bounded canary."""

    assert_science_constitution_intact(constitution)
    resolved_policy = _merge_policy(policy)
    samples = [deepcopy(item) for item in benchmark_samples if isinstance(item, dict)]
    candidate_check = validate_evolution_candidate(candidate, constitution=constitution)
    ablation_check = validate_ablation_report(ablation_report, candidate=candidate)
    task_ids = [str(item.get("task_id") or "").strip() for item in samples]
    hidden = bool(samples) and all(item.get("split") == "hidden" for item in samples)
    unique_tasks = (
        bool(task_ids) and all(task_ids) and len(task_ids) == len(set(task_ids))
    )
    minimum_tasks = max(int(resolved_policy.get("minimum_hidden_tasks") or 1), 1)
    layer_counts = {
        layer: sum(1 for item in samples if item.get("evaluation_layer") == layer)
        for layer in ("sealed", "prospective")
    }
    prospective_count = layer_counts["prospective"]
    task_hashes_valid = bool(samples) and all(
        _is_content_hash(item.get("task_hash")) for item in samples
    )
    candidate_created_at = _parse_utc_timestamp(candidate.get("created_at"))
    benchmark_frozen_times = [
        _parse_utc_timestamp(item.get("benchmark_frozen_at")) for item in samples
    ]
    frozen_before_candidate = (
        bool(samples)
        and candidate_created_at is not None
        and all(timestamp is not None for timestamp in benchmark_frozen_times)
        and all(
            item.get("frozen_before_candidate") is True
            and timestamp < candidate_created_at
            for item, timestamp in zip(samples, benchmark_frozen_times)
            if timestamp is not None
        )
    )
    now = datetime.now(timezone.utc)

    def prospective_custody_valid(item: dict[str, Any]) -> bool:
        not_before = _parse_utc_timestamp(item.get("resolution_not_before"))
        resolved_at = _parse_utc_timestamp(item.get("resolved_at"))
        return (
            item.get("prospective_resolved") is True
            and _is_content_hash(item.get("prospective_protocol_hash"))
            and _is_content_hash(item.get("resolution_attestation_hash"))
            and not_before is not None
            and resolved_at is not None
            and resolved_at >= not_before
            and resolved_at <= now
        )

    benchmark_custody = bool(samples) and all(
        (
            item.get("evaluation_layer") == "sealed"
            and _is_content_hash(item.get("custodian_attestation_hash"))
        )
        or (
            item.get("evaluation_layer") == "prospective"
            and prospective_custody_valid(item)
        )
        for item in samples
    )
    evaluator_stacks = _strings(item.get("evaluator_stack_id") for item in samples)
    independent_stacks = bool(samples) and all(
        str(item.get("producer_stack_id") or "").strip()
        and str(item.get("evaluator_stack_id") or "").strip()
        and item.get("producer_stack_id") != item.get("evaluator_stack_id")
        and _is_content_hash(item.get("producer_stack_hash"))
        and _is_content_hash(item.get("evaluator_stack_hash"))
        and item.get("producer_stack_hash") != item.get("evaluator_stack_hash")
        for item in samples
    )
    candidate_domains = set(_strings(candidate.get("applicability_domains")))
    sample_domains = set(_strings(item.get("domain") for item in samples))
    domain_coverage = bool(sample_domains) and (
        "all" in candidate_domains or sample_domains.issubset(candidate_domains)
    )
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
            "constitution_binding",
            candidate.get("constitution_hash") == constitution.get("constitution_hash"),
            "candidate is bound to the intact project constitution",
        ),
        _criterion(
            "ablation_attribution",
            ablation_check["passed"] and ablation_report.get("passed") is True,
            ", ".join(
                ablation_check["errors"]
                or list(ablation_report.get("required_failures") or [])
            )
            or "every declared change dimension has positive attributed value",
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
            "opaque_task_hashes",
            task_hashes_valid,
            "every hidden or prospective task is content-addressed",
        ),
        _criterion(
            "benchmark_precommitment",
            frozen_before_candidate,
            "benchmark tasks were frozen before the candidate existed",
        ),
        _criterion(
            "prospective_tasks",
            prospective_count
            >= int(resolved_policy.get("minimum_prospective_tasks") or 1),
            (
                f"tasks={prospective_count}/"
                f"{int(resolved_policy.get('minimum_prospective_tasks') or 1)}"
            ),
        ),
        _criterion(
            "benchmark_custody",
            benchmark_custody,
            "sealed and prospective tasks carry the required attestations",
        ),
        _criterion(
            "independent_evaluator_stacks",
            independent_stacks
            and len(evaluator_stacks)
            >= int(resolved_policy.get("minimum_distinct_evaluator_stacks") or 2),
            (
                f"distinct_stacks={len(evaluator_stacks)}/"
                f"{int(resolved_policy.get('minimum_distinct_evaluator_stacks') or 2)}"
            ),
        ),
        _criterion(
            "applicability_domain_coverage",
            domain_coverage,
            f"candidate_domains={sorted(candidate_domains)}, task_domains={sorted(sample_domains)}",
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
        "constitution_hash": constitution["constitution_hash"],
        "candidate": deepcopy(candidate),
        "policy": resolved_policy,
        "ablation_report": deepcopy(ablation_report),
        "benchmark_hash": benchmark_hash,
        "benchmark_samples": samples,
        "benchmark_task_count": len(samples),
        "benchmark_layer_counts": layer_counts,
        "decision": decision,
        "production_promotion_allowed": False,
        "criteria": criteria,
        "required_failures": required_failures,
        "metric_results": metric_results,
    }
    report["gate_hash"] = _canonical_hash(_gate_hash_payload(report))
    return report


def validate_evolution_gate(
    payload: dict[str, Any] | None,
    *,
    constitution: dict[str, Any],
) -> dict[str, Any]:
    report = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if set(report) != GATE_FIELDS:
        errors.append("gate_fields_invalid")
    try:
        assert_science_constitution_intact(constitution)
    except ValueError:
        errors.append("constitution_invalid")
    if report.get("constitution_hash") != constitution.get("constitution_hash"):
        errors.append("constitution_binding_mismatch")
    if report.get("gate_hash") != _canonical_hash(_gate_hash_payload(report)):
        errors.append("gate_hash_mismatch")
    try:
        expected = build_evolution_gate(
            report.get("candidate") or {},
            report.get("benchmark_samples") or [],
            constitution=constitution,
            ablation_report=report.get("ablation_report") or {},
            policy=report.get("policy") or {},
        )
        if _gate_hash_payload(report) != _gate_hash_payload(expected):
            errors.append("gate_semantics_mismatch")
    except (EvolutionGateError, TypeError, ValueError):
        errors.append("gate_reconstruction_failed")
    return {"passed": not errors, "errors": errors}


def _rollback_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"generated_at", "receipt_hash"}
    }


def build_rollback_receipt(
    candidate: dict[str, Any],
    *,
    restored_artifact_hash: str,
    execution_log_hash: str,
    executed_by: str,
    trigger: str = "canary_exercise",
) -> dict[str, Any]:
    """Record a content-verified rollback exercise for the exact candidate."""

    check = validate_evolution_candidate(candidate)
    if not check["ok"]:
        raise EvolutionGateError("candidate invalid: " + ", ".join(check["errors"]))
    if not _is_content_hash(restored_artifact_hash) or not _is_content_hash(
        execution_log_hash
    ):
        raise EvolutionGateError("rollback artifact and execution log require hashes")
    actor = str(executed_by or "").strip()
    event = str(trigger or "").strip()
    if not actor.startswith(("service:", "human:")) or not event:
        raise EvolutionGateError(
            "rollback trigger and a service: or human: actor are required"
        )
    verified = restored_artifact_hash == candidate.get("base_artifact_hash")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "candidate_hash": candidate["candidate_hash"],
        "rollback_ref": candidate["rollback_ref"],
        "expected_artifact_hash": candidate["base_artifact_hash"],
        "restored_artifact_hash": restored_artifact_hash,
        "execution_log_hash": execution_log_hash,
        "executed_by": actor,
        "trigger": event,
        "exercise_only": True,
        "status": "verified" if verified else "failed",
    }
    payload["receipt_hash"] = _canonical_hash(_rollback_core(payload))
    return payload


def validate_rollback_receipt(
    payload: dict[str, Any] | None,
    *,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    receipt = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if set(receipt) != ROLLBACK_FIELDS:
        errors.append("rollback_fields_invalid")
    if receipt.get("candidate_hash") != candidate.get("candidate_hash"):
        errors.append("candidate_binding_mismatch")
    if receipt.get("rollback_ref") != candidate.get("rollback_ref"):
        errors.append("rollback_ref_mismatch")
    if receipt.get("expected_artifact_hash") != candidate.get("base_artifact_hash"):
        errors.append("expected_artifact_mismatch")
    if receipt.get("restored_artifact_hash") != candidate.get("base_artifact_hash"):
        errors.append("restored_artifact_mismatch")
    if receipt.get("status") != "verified" or receipt.get("exercise_only") is not True:
        errors.append("rollback_not_verified")
    if receipt.get("receipt_hash") != _canonical_hash(_rollback_core(receipt)):
        errors.append("receipt_hash_mismatch")
    return {"passed": not errors, "errors": errors}


def _canary_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"generated_at", "canary_hash"}
    }


def build_canary_report(
    candidate: dict[str, Any],
    rollback_receipt: dict[str, Any],
    *,
    executed_by: str,
    observation_count: int,
    error_rate_delta: float,
    quality_delta: float,
    real_research_project_ids: Iterable[str],
    project_run_hashes: dict[str, str],
    incidents: Iterable[str] | None = None,
    long_tail_pass: bool,
    common_mode_failure_pass: bool,
    out_of_distribution_pass: bool,
) -> dict[str, Any]:
    """Build a hash-bound canary report over real research workloads."""

    candidate_check = validate_evolution_candidate(candidate)
    if not candidate_check["ok"]:
        raise EvolutionGateError(
            "candidate invalid: " + ", ".join(candidate_check["errors"])
        )
    actor = str(executed_by or "").strip()
    if not actor.startswith(("service:", "human:")):
        raise EvolutionGateError(
            "canary executor must use a service: or human: identity namespace"
        )
    observations = _finite_number(observation_count)
    error_delta = _finite_number(error_rate_delta)
    quality_change = _finite_number(quality_delta)
    projects = _strings(real_research_project_ids)
    run_hashes = {
        str(project_id): str(run_hash)
        for project_id, run_hash in (project_run_hashes or {}).items()
    }
    project_evidence_valid = set(run_hashes) == set(projects) and all(
        _is_content_hash(value) for value in run_hashes.values()
    )
    incident_rows = _strings(incidents)
    rollback_check = validate_rollback_receipt(rollback_receipt, candidate=candidate)
    status = (
        "passed"
        if observations is not None
        and observations.is_integer()
        and observations > 0
        and error_delta is not None
        and quality_change is not None
        and projects
        and project_evidence_valid
        and not incident_rows
        and rollback_check["passed"]
        and long_tail_pass is True
        and common_mode_failure_pass is True
        and out_of_distribution_pass is True
        else "blocked"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "candidate_hash": candidate["candidate_hash"],
        "executed_by": actor,
        "status": status,
        "observation_count": int(observations) if observations is not None else None,
        "error_rate_delta": error_delta,
        "quality_delta": quality_change,
        "real_research_project_ids": projects,
        "project_run_hashes": run_hashes,
        "incidents": incident_rows,
        "long_tail_pass": long_tail_pass is True,
        "common_mode_failure_pass": common_mode_failure_pass is True,
        "out_of_distribution_pass": out_of_distribution_pass is True,
        "rollback_receipt": deepcopy(rollback_receipt),
    }
    payload["canary_hash"] = _canonical_hash(_canary_core(payload))
    return payload


def validate_canary_report(
    payload: dict[str, Any] | None,
    *,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    report = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if report.get("candidate_hash") != candidate.get("candidate_hash"):
        errors.append("candidate_binding_mismatch")
    if report.get("canary_hash") != _canonical_hash(_canary_core(report)):
        errors.append("canary_hash_mismatch")
    try:
        expected = build_canary_report(
            candidate,
            report.get("rollback_receipt") or {},
            executed_by=str(report.get("executed_by") or ""),
            observation_count=int(report.get("observation_count")),
            error_rate_delta=float(report.get("error_rate_delta")),
            quality_delta=float(report.get("quality_delta")),
            real_research_project_ids=report.get("real_research_project_ids") or [],
            project_run_hashes=report.get("project_run_hashes") or {},
            incidents=report.get("incidents") or [],
            long_tail_pass=report.get("long_tail_pass") is True,
            common_mode_failure_pass=report.get("common_mode_failure_pass") is True,
            out_of_distribution_pass=report.get("out_of_distribution_pass") is True,
        )
        if _canary_core(report) != _canary_core(expected):
            errors.append("canary_semantics_mismatch")
    except (EvolutionGateError, TypeError, ValueError):
        errors.append("canary_reconstruction_failed")
    return {"passed": not errors, "errors": errors}


def _promotion_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"generated_at", "promotion_hash"}
    }


def approve_production_promotion(
    gate_report: dict[str, Any],
    canary_report: dict[str, Any],
    *,
    constitution: dict[str, Any],
    approver_ids: Iterable[str] | None = None,
    approver_id: str | None = None,
) -> dict[str, Any]:
    """Authorize production after attributed, diverse, rollback-tested canary work."""

    policy = gate_report.get("policy") or {}
    candidate = gate_report.get("candidate") or {}
    raw_approvers = (
        [approver_ids] if isinstance(approver_ids, str) else list(approver_ids or [])
    )
    approvers = _strings([*raw_approvers, *([approver_id] if approver_id else [])])
    candidate_check = validate_evolution_candidate(candidate, constitution=constitution)
    gate_check = validate_evolution_gate(gate_report, constitution=constitution)
    canary_check = validate_canary_report(canary_report, candidate=candidate)
    observations = _finite_number(canary_report.get("observation_count"))
    error_rate_delta = _finite_number(canary_report.get("error_rate_delta"))
    quality_delta = _finite_number(canary_report.get("quality_delta"))
    project_ids = _strings(canary_report.get("real_research_project_ids"))
    project_run_hashes = (
        canary_report.get("project_run_hashes")
        if isinstance(canary_report.get("project_run_hashes"), dict)
        else {}
    )
    required_approvers = (
        int(policy.get("minimum_high_risk_human_approvers") or 2)
        if candidate.get("risk_tier") == "high"
        else 1
    )
    canary_executor = str(canary_report.get("executed_by") or "")
    independent_humans = (
        len(approvers) >= required_approvers
        and all(item.startswith("human:") for item in approvers)
        and str(candidate.get("proposed_by") or "") not in approvers
        and canary_executor not in approvers
    )
    rollback_check = validate_rollback_receipt(
        canary_report.get("rollback_receipt"), candidate=candidate
    )
    criteria = [
        _criterion(
            "shadow_gate_integrity",
            gate_check["passed"] and candidate_check["ok"],
            ", ".join(gate_check["errors"] or candidate_check["errors"])
            or "gate semantics, candidate, and constitution binding are valid",
        ),
        _criterion(
            "shadow_gate",
            gate_report.get("decision") == "promote_to_canary",
            str(gate_report.get("decision") or "missing"),
        ),
        _criterion(
            "canary_integrity",
            canary_check["passed"],
            ", ".join(canary_check["errors"]) or "canary semantics and hash valid",
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
            "real_research_canary",
            len(project_ids) >= int(policy.get("minimum_real_research_projects") or 3),
            (
                f"projects={len(project_ids)}/"
                f"{int(policy.get('minimum_real_research_projects') or 3)}"
            ),
        ),
        _criterion(
            "real_research_evidence",
            set(project_run_hashes) == set(project_ids)
            and all(_is_content_hash(value) for value in project_run_hashes.values()),
            "every canary project has a content-addressed run artifact",
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
            "long_tail_stability",
            canary_report.get("long_tail_pass") is True,
            "long-tail workloads pass",
        ),
        _criterion(
            "common_mode_failure_resistance",
            canary_report.get("common_mode_failure_pass") is True,
            "independent failure-stack challenge passes",
        ),
        _criterion(
            "out_of_distribution_stability",
            canary_report.get("out_of_distribution_pass") is True,
            "out-of-distribution research workloads pass",
        ),
        _criterion(
            "rollback_verified",
            rollback_check["passed"],
            ", ".join(rollback_check["errors"])
            or "rollback restored the exact baseline artifact",
        ),
        _criterion(
            "independent_approval",
            independent_humans,
            f"human_approvers={len(approvers)}/{required_approvers}",
        ),
    ]
    failures = [item["id"] for item in criteria if not item["passed"]]
    decision = "approved" if not failures else "blocked"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "constitution_hash": constitution.get("constitution_hash"),
        "candidate": deepcopy(candidate),
        "gate_report": deepcopy(gate_report),
        "shadow_gate_hash": gate_report.get("gate_hash"),
        "canary_report": deepcopy(canary_report),
        "approver_ids": approvers,
        "decision": decision,
        "production_promotion_allowed": decision == "approved",
        "criteria": criteria,
        "required_failures": failures,
        "rollback_ref": candidate.get("rollback_ref"),
    }
    payload["promotion_hash"] = _canonical_hash(_promotion_core(payload))
    return payload


def validate_production_promotion(
    payload: dict[str, Any] | None,
    *,
    constitution: dict[str, Any],
) -> dict[str, Any]:
    report = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if report.get("promotion_hash") != _canonical_hash(_promotion_core(report)):
        errors.append("promotion_hash_mismatch")
    try:
        expected = approve_production_promotion(
            report.get("gate_report") or {},
            report.get("canary_report") or {},
            constitution=constitution,
            approver_ids=report.get("approver_ids") or [],
        )
        if _promotion_core(report) != _promotion_core(expected):
            errors.append("promotion_semantics_mismatch")
    except (EvolutionGateError, TypeError, ValueError):
        errors.append("promotion_reconstruction_failed")
    return {"passed": not errors, "errors": errors}


def save_evolution_gate(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    constitution: dict[str, Any],
    producer: str,
) -> str:
    if payload.get("promotion_hash"):
        artifact_check = validate_production_promotion(
            payload, constitution=constitution
        )
    else:
        artifact_check = validate_evolution_gate(payload, constitution=constitution)
    if not artifact_check["passed"]:
        raise EvolutionGateError(
            "evolution artifact invalid: " + ", ".join(artifact_check["errors"])
        )
    dependencies = ["science_constitution", "self_evolution", "stage_standards"]
    output_path = save_contract_artifact(
        project_root,
        "evolution_gate",
        payload,
        producer=producer,
        depends_on=dependencies,
        warnings=list(payload.get("required_failures") or []),
    )
    accepted = payload.get("decision") in {"promote_to_canary", "approved"}
    if not accepted:
        update_pipeline_artifact(
            project_root,
            "evolution_gate",
            status="blocked",
            producer=producer,
            depends_on=dependencies,
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
            "constitution_hash": payload.get("constitution_hash"),
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
    "COMPONENT_TYPES",
    "PROTECTED_COMPONENT_TYPES",
    "EvolutionGateError",
    "approve_production_promotion",
    "build_ablation_report",
    "build_canary_report",
    "build_evolution_candidate",
    "build_evolution_gate",
    "build_rollback_receipt",
    "save_evolution_gate",
    "validate_ablation_report",
    "validate_canary_report",
    "validate_evolution_candidate",
    "validate_evolution_gate",
    "validate_production_promotion",
    "validate_rollback_receipt",
]
