from __future__ import annotations

"""Independent, hash-bound governance for scientific evaluation."""

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
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

SCHEMA_VERSION = 1
EVALUATION_ROLES = (
    "researcher",
    "verifier",
    "benchmark_custodian",
    "approver",
)
EVALUATION_LAYERS = ("public", "sealed", "prospective", "external")
REQUIRED_PROMOTION_LAYERS = ("sealed", "prospective", "external")
METRIC_SPECS = {
    "objective_quality": {"direction": "higher", "threshold": 0.60},
    "worst_domain_quality": {"direction": "higher", "threshold": 0.50},
    "reproducibility_rate": {"direction": "higher", "threshold": 0.80},
    "false_discovery_rate": {"direction": "lower", "threshold": 0.10},
    "calibration_error": {"direction": "lower", "threshold": 0.15},
    "information_gain": {"direction": "higher", "threshold": 0.01},
}
CORE_EVALUATION_POLICY = {
    "policy_id": "xscientist-independent-evaluation",
    "version": "1.0.0",
    "roles": list(EVALUATION_ROLES),
    "required_promotion_layers": list(REQUIRED_PROMOTION_LAYERS),
    "metric_specs": deepcopy(METRIC_SPECS),
    "hard_gates": ["integrity_pass", "safety_pass", "reproducibility_pass"],
    "public_evaluation_can_promote": False,
    "minimum_distinct_verifiers": 2,
    "external_verifier_must_be_distinct": True,
    "approver_must_be_human": True,
    "all_roles_must_be_disjoint": True,
}
CHARTER_FIELDS = {
    "schema_version",
    "charter_id",
    "status",
    "generated_at",
    "constitution_hash",
    "policy",
    "policy_hash",
    "assignments",
    "charter_hash",
}
BENCHMARK_FIELDS = {
    "schema_version",
    "benchmark_id",
    "charter_hash",
    "layer",
    "status",
    "candidate_hash",
    "task_hashes",
    "task_count",
    "custodian_id",
    "answer_key_hash",
    "resolution_condition",
    "resolution_not_before",
    "external_organization_id",
    "external_protocol_hash",
    "raw_tasks_in_manifest",
    "raw_answers_in_manifest",
    "created_at",
    "benchmark_hash",
}


class EvaluationGovernanceError(ValueError):
    """Raised when evaluation independence or artifact integrity is violated."""


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


def _identities(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(item).strip() for item in (values or []) if str(item).strip()})


def _finite_metric(value: Any) -> float | None:
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


def _charter_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "charter_id": payload.get("charter_id"),
        "status": payload.get("status"),
        "constitution_hash": payload.get("constitution_hash"),
        "policy": payload.get("policy"),
        "policy_hash": payload.get("policy_hash"),
        "assignments": payload.get("assignments"),
    }


def build_evaluation_charter(
    constitution: dict[str, Any],
    *,
    assignments: dict[str, Iterable[str]],
) -> dict[str, Any]:
    """Lock mutually exclusive scientific evaluation authorities."""

    assert_science_constitution_intact(constitution)
    normalized = {role: _identities(assignments.get(role)) for role in EVALUATION_ROLES}
    missing = [role for role, principals in normalized.items() if not principals]
    if missing:
        raise EvaluationGovernanceError(
            "missing evaluation authorities: " + ", ".join(missing)
        )
    all_principals = [
        principal for principals in normalized.values() for principal in principals
    ]
    if len(all_principals) != len(set(all_principals)):
        raise EvaluationGovernanceError(
            "researcher, verifier, custodian, and approver identities must be disjoint"
        )
    if len(normalized["verifier"]) < 2:
        raise EvaluationGovernanceError(
            "at least two verifier identities are required for external independence"
        )
    if any(not principal.startswith("human:") for principal in normalized["approver"]):
        raise EvaluationGovernanceError(
            "approver identities must use the human: namespace"
        )
    seed = {
        "constitution_hash": constitution["constitution_hash"],
        "assignments": normalized,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "charter_id": "evaluation-charter:"
        + _canonical_hash(seed).split(":", 1)[1][:16],
        "status": "locked",
        "generated_at": _now_iso(),
        "constitution_hash": constitution["constitution_hash"],
        "policy": deepcopy(CORE_EVALUATION_POLICY),
        "policy_hash": _canonical_hash(CORE_EVALUATION_POLICY),
        "assignments": normalized,
    }
    payload["charter_hash"] = _canonical_hash(_charter_core(payload))
    return payload


def validate_evaluation_charter(
    payload: dict[str, Any] | None,
    *,
    constitution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    charter = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if charter.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if set(charter) != CHARTER_FIELDS:
        errors.append("charter_fields_invalid")
    if charter.get("status") != "locked":
        errors.append("charter_not_locked")
    if charter.get("policy") != CORE_EVALUATION_POLICY:
        errors.append("evaluation_policy_modified")
    if charter.get("policy_hash") != _canonical_hash(CORE_EVALUATION_POLICY):
        errors.append("evaluation_policy_hash_mismatch")
    if not _is_content_hash(charter.get("constitution_hash")):
        errors.append("constitution_hash_invalid")
    if constitution is not None:
        try:
            assert_science_constitution_intact(constitution)
        except ValueError:
            errors.append("constitution_invalid")
        if charter.get("constitution_hash") != constitution.get("constitution_hash"):
            errors.append("constitution_binding_mismatch")
    assignments = (
        charter.get("assignments")
        if isinstance(charter.get("assignments"), dict)
        else {}
    )
    normalized = {role: _identities(assignments.get(role)) for role in EVALUATION_ROLES}
    for role, principals in normalized.items():
        if not principals:
            errors.append(f"authority_missing:{role}")
    all_principals = [item for values in normalized.values() for item in values]
    if len(all_principals) != len(set(all_principals)):
        errors.append("authority_overlap")
    if len(normalized["verifier"]) < 2:
        errors.append("distinct_verifiers_insufficient")
    if any(not principal.startswith("human:") for principal in normalized["approver"]):
        errors.append("approver_not_human")
    if charter.get("charter_hash") != _canonical_hash(_charter_core(charter)):
        errors.append("charter_hash_mismatch")
    return {"passed": not errors, "errors": errors}


def assert_evaluation_charter_valid(
    payload: dict[str, Any] | None,
    *,
    constitution: dict[str, Any] | None = None,
) -> None:
    check = validate_evaluation_charter(payload, constitution=constitution)
    if not check["passed"]:
        raise EvaluationGovernanceError(
            "evaluation charter invalid: " + ", ".join(check["errors"])
        )


def _benchmark_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"created_at", "benchmark_hash"}
    }


def build_benchmark_manifest(
    charter: dict[str, Any],
    *,
    benchmark_id: str,
    layer: str,
    candidate_hash: str,
    task_hashes: Iterable[str],
    custodian_id: str,
    answer_key_hash: str | None = None,
    resolution_condition: str | None = None,
    resolution_not_before: str | None = None,
    external_organization_id: str | None = None,
    external_protocol_hash: str | None = None,
) -> dict[str, Any]:
    """Create an opaque benchmark custody manifest without storing answers."""

    assert_evaluation_charter_valid(charter)
    evaluation_layer = str(layer or "").strip()
    custodian = str(custodian_id or "").strip()
    if evaluation_layer not in EVALUATION_LAYERS:
        raise EvaluationGovernanceError(
            f"unsupported evaluation layer={evaluation_layer!r}"
        )
    if custodian not in charter["assignments"]["benchmark_custodian"]:
        raise EvaluationGovernanceError(
            "custodian is not assigned by the locked charter"
        )
    if not _is_content_hash(candidate_hash):
        raise EvaluationGovernanceError("candidate_hash must be a SHA-256 content hash")
    tasks = sorted({str(item) for item in task_hashes if _is_content_hash(item)})
    if not tasks:
        raise EvaluationGovernanceError("at least one opaque task hash is required")
    if evaluation_layer == "sealed" and not _is_content_hash(answer_key_hash):
        raise EvaluationGovernanceError("sealed evaluation requires answer_key_hash")
    if evaluation_layer == "prospective" and (
        not str(resolution_condition or "").strip()
        or _parse_utc_timestamp(resolution_not_before) is None
    ):
        raise EvaluationGovernanceError(
            "prospective evaluation requires a resolution condition and not-before time"
        )
    if evaluation_layer == "external" and (
        not str(external_organization_id or "").strip()
        or not _is_content_hash(external_protocol_hash)
    ):
        raise EvaluationGovernanceError(
            "external evaluation requires organization ID and protocol hash"
        )
    status_by_layer = {
        "public": "public",
        "sealed": "sealed",
        "prospective": "unresolved",
        "external": "external_custody",
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": str(benchmark_id or "").strip(),
        "charter_hash": charter["charter_hash"],
        "layer": evaluation_layer,
        "status": status_by_layer[evaluation_layer],
        "candidate_hash": candidate_hash,
        "task_hashes": tasks,
        "task_count": len(tasks),
        "custodian_id": custodian,
        "answer_key_hash": answer_key_hash if evaluation_layer == "sealed" else None,
        "resolution_condition": (
            str(resolution_condition).strip()
            if evaluation_layer == "prospective"
            else None
        ),
        "resolution_not_before": (
            str(resolution_not_before).strip()
            if evaluation_layer == "prospective"
            else None
        ),
        "external_organization_id": (
            str(external_organization_id).strip()
            if evaluation_layer == "external"
            else None
        ),
        "external_protocol_hash": (
            external_protocol_hash if evaluation_layer == "external" else None
        ),
        "raw_tasks_in_manifest": False,
        "raw_answers_in_manifest": False,
        "created_at": _now_iso(),
    }
    if not payload["benchmark_id"]:
        raise EvaluationGovernanceError("benchmark_id is required")
    payload["benchmark_hash"] = _canonical_hash(_benchmark_core(payload))
    return payload


def validate_benchmark_manifest(
    payload: dict[str, Any] | None,
    *,
    charter: dict[str, Any],
) -> dict[str, Any]:
    manifest = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    charter_check = validate_evaluation_charter(charter)
    if not charter_check["passed"]:
        errors.append("charter_invalid")
    layer = manifest.get("layer")
    if layer not in EVALUATION_LAYERS:
        errors.append("layer_invalid")
    if manifest.get("charter_hash") != charter.get("charter_hash"):
        errors.append("charter_binding_mismatch")
    if set(manifest) != BENCHMARK_FIELDS:
        errors.append("benchmark_fields_invalid")
    if manifest.get("custodian_id") not in (charter.get("assignments") or {}).get(
        "benchmark_custodian", []
    ):
        errors.append("custodian_not_authorized")
    if not _is_content_hash(manifest.get("candidate_hash")):
        errors.append("candidate_hash_invalid")
    tasks = (
        manifest.get("task_hashes")
        if isinstance(manifest.get("task_hashes"), list)
        else []
    )
    if not tasks or any(not _is_content_hash(item) for item in tasks):
        errors.append("task_hashes_invalid")
    if manifest.get("task_count") != len(tasks):
        errors.append("task_count_mismatch")
    if manifest.get("raw_tasks_in_manifest") is not False:
        errors.append("raw_tasks_forbidden")
    if manifest.get("raw_answers_in_manifest") is not False:
        errors.append("raw_answers_forbidden")
    if layer == "sealed" and not _is_content_hash(manifest.get("answer_key_hash")):
        errors.append("answer_key_hash_missing")
    if layer == "prospective" and (
        not str(manifest.get("resolution_condition") or "").strip()
        or _parse_utc_timestamp(manifest.get("resolution_not_before")) is None
    ):
        errors.append("prospective_resolution_missing")
    if layer == "external" and (
        not str(manifest.get("external_organization_id") or "").strip()
        or not _is_content_hash(manifest.get("external_protocol_hash"))
    ):
        errors.append("external_custody_missing")
    if manifest.get("benchmark_hash") != _canonical_hash(_benchmark_core(manifest)):
        errors.append("benchmark_hash_mismatch")
    return {"passed": not errors, "errors": errors}


def _criterion(criterion_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"id": criterion_id, "passed": bool(passed), "detail": detail}


def _run_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"generated_at", "run_hash"}
    }


def build_evaluation_run(
    charter: dict[str, Any],
    benchmark: dict[str, Any],
    *,
    evaluator_id: str,
    results: dict[str, Any],
    evaluator_input_hash: str,
    evaluator_result_hash: str,
    custodian_attestation_hash: str | None = None,
    resolution_attestation_hash: str | None = None,
    external_attestation_hash: str | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen candidate on one governed benchmark layer."""

    assert_evaluation_charter_valid(charter)
    benchmark_check = validate_benchmark_manifest(benchmark, charter=charter)
    if not benchmark_check["passed"]:
        raise EvaluationGovernanceError(
            "benchmark invalid: " + ", ".join(benchmark_check["errors"])
        )
    evaluator = str(evaluator_id or "").strip()
    if evaluator not in charter["assignments"]["verifier"]:
        raise EvaluationGovernanceError("evaluator is not assigned as a verifier")
    if not _is_content_hash(evaluator_input_hash) or not _is_content_hash(
        evaluator_result_hash
    ):
        raise EvaluationGovernanceError(
            "evaluator inputs and results require SHA-256 hashes"
        )
    metrics = results.get("metrics") if isinstance(results.get("metrics"), dict) else {}
    criteria = [
        _criterion(
            "integrity",
            results.get("integrity_pass") is True,
            "research-integrity checks pass",
        ),
        _criterion(
            "safety",
            results.get("safety_pass") is True,
            "safety checks pass",
        ),
        _criterion(
            "reproducibility",
            results.get("reproducibility_pass") is True,
            "reproduction checks pass",
        ),
    ]
    metric_results: dict[str, Any] = {}
    for metric, spec in METRIC_SPECS.items():
        value = _finite_metric(metrics.get(metric))
        threshold = float(spec["threshold"])
        passed = value is not None and (
            value >= threshold if spec["direction"] == "higher" else value <= threshold
        )
        metric_results[metric] = {
            "value": value,
            "direction": spec["direction"],
            "threshold": threshold,
            "passed": passed,
        }
        criteria.append(
            _criterion(
                f"metric:{metric}",
                passed,
                f"value={value}, {spec['direction']} threshold={threshold}",
            )
        )
    layer = str(benchmark["layer"])
    if layer == "sealed":
        criteria.append(
            _criterion(
                "sealed_custodian_attestation",
                _is_content_hash(custodian_attestation_hash),
                "custodian attests scoring without exposing answers",
            )
        )
    if layer == "prospective":
        resolution_not_before = _parse_utc_timestamp(
            benchmark.get("resolution_not_before")
        )
        criteria.append(
            _criterion(
                "prospective_resolution",
                _is_content_hash(resolution_attestation_hash)
                and results.get("prospective_resolved") is True,
                "future outcome resolved under the frozen protocol",
            )
        )
        criteria.append(
            _criterion(
                "prospective_embargo_elapsed",
                resolution_not_before is not None
                and datetime.now(timezone.utc) >= resolution_not_before,
                "evaluation cannot resolve before its frozen not-before time",
            )
        )
    if layer == "external":
        criteria.append(
            _criterion(
                "external_attestation",
                _is_content_hash(external_attestation_hash),
                "external organization attests independent evaluation",
            )
        )
    failures = [item["id"] for item in criteria if not item["passed"]]
    decision = "pass" if not failures else "blocked"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "charter_hash": charter["charter_hash"],
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": benchmark["benchmark_hash"],
        "benchmark": deepcopy(benchmark),
        "layer": layer,
        "candidate_hash": benchmark["candidate_hash"],
        "evaluator_id": evaluator,
        "evaluator_input_hash": evaluator_input_hash,
        "evaluator_result_hash": evaluator_result_hash,
        "custodian_attestation_hash": custodian_attestation_hash,
        "resolution_attestation_hash": resolution_attestation_hash,
        "external_attestation_hash": external_attestation_hash,
        "results": deepcopy(results),
        "criteria": criteria,
        "metric_results": metric_results,
        "decision": decision,
        "required_failures": failures,
    }
    payload["run_hash"] = _canonical_hash(_run_core(payload))
    return payload


def validate_evaluation_run(
    payload: dict[str, Any] | None,
    *,
    charter: dict[str, Any],
) -> dict[str, Any]:
    run = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if run.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if run.get("charter_hash") != charter.get("charter_hash"):
        errors.append("charter_binding_mismatch")
    benchmark = run.get("benchmark") if isinstance(run.get("benchmark"), dict) else {}
    benchmark_check = validate_benchmark_manifest(benchmark, charter=charter)
    if not benchmark_check["passed"]:
        errors.append("benchmark_invalid")
    if run.get("benchmark_id") != benchmark.get("benchmark_id"):
        errors.append("benchmark_id_mismatch")
    if run.get("benchmark_hash") != benchmark.get("benchmark_hash"):
        errors.append("benchmark_hash_mismatch")
    if run.get("candidate_hash") != benchmark.get("candidate_hash"):
        errors.append("candidate_binding_mismatch")
    if run.get("layer") not in EVALUATION_LAYERS:
        errors.append("layer_invalid")
    if run.get("evaluator_id") not in (charter.get("assignments") or {}).get(
        "verifier", []
    ):
        errors.append("evaluator_not_authorized")
    for field in (
        "benchmark_hash",
        "candidate_hash",
        "evaluator_input_hash",
        "evaluator_result_hash",
    ):
        if not _is_content_hash(run.get(field)):
            errors.append(f"{field}_invalid")
    criteria = run.get("criteria") if isinstance(run.get("criteria"), list) else []
    failures = [
        str(item.get("id"))
        for item in criteria
        if isinstance(item, dict) and not item.get("passed")
    ]
    expected_decision = "pass" if criteria and not failures else "blocked"
    if run.get("decision") != expected_decision:
        errors.append("decision_inconsistent")
    if list(run.get("required_failures") or []) != failures:
        errors.append("failure_list_inconsistent")
    if run.get("run_hash") != _canonical_hash(_run_core(run)):
        errors.append("run_hash_mismatch")
    try:
        expected = build_evaluation_run(
            charter,
            benchmark,
            evaluator_id=str(run.get("evaluator_id") or ""),
            results=deepcopy(run.get("results") or {}),
            evaluator_input_hash=str(run.get("evaluator_input_hash") or ""),
            evaluator_result_hash=str(run.get("evaluator_result_hash") or ""),
            custodian_attestation_hash=run.get("custodian_attestation_hash"),
            resolution_attestation_hash=run.get("resolution_attestation_hash"),
            external_attestation_hash=run.get("external_attestation_hash"),
        )
        if _run_core(run) != _run_core(expected):
            errors.append("run_semantics_mismatch")
    except (EvaluationGovernanceError, TypeError, ValueError):
        errors.append("run_reconstruction_failed")
    return {"passed": not errors, "errors": errors}


def _decision_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"generated_at", "decision_hash"}
    }


def build_evaluation_decision(
    charter: dict[str, Any],
    runs: Iterable[dict[str, Any]],
    *,
    candidate_hash: str,
    candidate_producer_id: str,
    approver_id: str,
    scope_node_ids: Iterable[str],
) -> dict[str, Any]:
    """Approve scientific promotion only after all independent layers pass."""

    assert_evaluation_charter_valid(charter)
    rows = [deepcopy(item) for item in runs if isinstance(item, dict)]
    producer = str(candidate_producer_id or "").strip()
    approver = str(approver_id or "").strip()
    scopes = _identities(scope_node_ids)
    run_checks = [validate_evaluation_run(item, charter=charter) for item in rows]
    passing_by_layer = {
        layer: [
            item
            for item, check in zip(rows, run_checks)
            if check["passed"]
            and item.get("layer") == layer
            and item.get("decision") == "pass"
            and item.get("candidate_hash") == candidate_hash
        ]
        for layer in EVALUATION_LAYERS
    }
    internal_verifiers = {
        str(item.get("evaluator_id"))
        for layer in ("sealed", "prospective")
        for item in passing_by_layer[layer]
    }
    external_verifiers = {
        str(item.get("evaluator_id")) for item in passing_by_layer["external"]
    }
    all_verifiers = internal_verifiers | external_verifiers
    criteria = [
        _criterion(
            "candidate_hash",
            _is_content_hash(candidate_hash),
            "candidate is content-addressed",
        ),
        _criterion(
            "candidate_producer",
            producer in charter["assignments"]["researcher"],
            f"producer={producer or 'missing'}",
        ),
        _criterion(
            "human_approver",
            approver in charter["assignments"]["approver"]
            and approver.startswith("human:"),
            f"approver={approver or 'missing'}",
        ),
        _criterion(
            "scope_nodes",
            bool(scopes),
            f"scope_node_count={len(scopes)}",
        ),
        _criterion(
            "run_integrity",
            bool(rows) and all(check["passed"] for check in run_checks),
            "every run is hash-valid and charter-bound",
        ),
    ]
    for layer in REQUIRED_PROMOTION_LAYERS:
        criteria.append(
            _criterion(
                f"layer:{layer}",
                bool(passing_by_layer[layer]),
                f"passing_runs={len(passing_by_layer[layer])}",
            )
        )
    criteria.extend(
        [
            _criterion(
                "distinct_verifiers",
                len(all_verifiers)
                >= int(CORE_EVALUATION_POLICY["minimum_distinct_verifiers"]),
                f"distinct_verifiers={len(all_verifiers)}",
            ),
            _criterion(
                "external_verifier_independence",
                bool(external_verifiers)
                and external_verifiers.isdisjoint(internal_verifiers),
                "external evaluation uses a different verifier identity",
            ),
            _criterion(
                "authority_separation",
                bool(producer)
                and bool(approver)
                and producer != approver
                and producer not in all_verifiers
                and approver not in all_verifiers,
                "producer, verifiers, and approver are distinct",
            ),
        ]
    )
    failures = [item["id"] for item in criteria if not item["passed"]]
    decision = "approved" if not failures else "blocked"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "charter": deepcopy(charter),
        "charter_hash": charter["charter_hash"],
        "constitution_hash": charter["constitution_hash"],
        "candidate_hash": candidate_hash,
        "candidate_producer_id": producer,
        "approver_id": approver,
        "scope_node_ids": scopes,
        "runs": rows,
        "criteria": criteria,
        "required_failures": failures,
        "decision": decision,
        "claim_promotion_allowed": decision == "approved",
    }
    payload["decision_hash"] = _canonical_hash(_decision_core(payload))
    return payload


def validate_evaluation_decision(payload: dict[str, Any] | None) -> dict[str, Any]:
    report = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    charter = report.get("charter") if isinstance(report.get("charter"), dict) else {}
    charter_check = validate_evaluation_charter(charter)
    if not charter_check["passed"]:
        errors.append("charter_invalid")
    if report.get("charter_hash") != charter.get("charter_hash"):
        errors.append("charter_hash_mismatch")
    runs = report.get("runs") if isinstance(report.get("runs"), list) else []
    if not runs:
        errors.append("runs_missing")
    else:
        for index, run in enumerate(runs):
            check = validate_evaluation_run(run, charter=charter)
            if not check["passed"]:
                errors.append(f"run_invalid:{index}")
    criteria = (
        report.get("criteria") if isinstance(report.get("criteria"), list) else []
    )
    failures = [
        str(item.get("id"))
        for item in criteria
        if isinstance(item, dict) and not item.get("passed")
    ]
    expected_decision = "approved" if criteria and not failures else "blocked"
    if report.get("decision") != expected_decision:
        errors.append("decision_inconsistent")
    if bool(report.get("claim_promotion_allowed")) != (expected_decision == "approved"):
        errors.append("promotion_flag_inconsistent")
    if list(report.get("required_failures") or []) != failures:
        errors.append("failure_list_inconsistent")
    if not _is_content_hash(report.get("candidate_hash")):
        errors.append("candidate_hash_invalid")
    if not _identities(report.get("scope_node_ids")):
        errors.append("scope_nodes_missing")
    if report.get("decision_hash") != _canonical_hash(_decision_core(report)):
        errors.append("decision_hash_mismatch")
    try:
        expected = build_evaluation_decision(
            charter,
            runs,
            candidate_hash=str(report.get("candidate_hash") or ""),
            candidate_producer_id=str(report.get("candidate_producer_id") or ""),
            approver_id=str(report.get("approver_id") or ""),
            scope_node_ids=report.get("scope_node_ids") or [],
        )
        if _decision_core(report) != _decision_core(expected):
            errors.append("decision_semantics_mismatch")
    except (EvaluationGovernanceError, TypeError, ValueError):
        errors.append("decision_reconstruction_failed")
    return {"passed": not errors, "errors": errors}


def assert_scientific_promotion_allowed(
    report: dict[str, Any] | None,
    *,
    node_id: str | None = None,
) -> None:
    check = validate_evaluation_decision(report)
    payload = report if isinstance(report, dict) else {}
    if not check["passed"] or payload.get("decision") != "approved":
        raise EvaluationGovernanceError(
            "scientific promotion blocked: "
            + ", ".join(
                check["errors"] or payload.get("required_failures") or ["unknown"]
            )
        )
    if node_id and str(node_id) not in _identities(payload.get("scope_node_ids")):
        raise EvaluationGovernanceError(
            f"scientific promotion report does not cover node_id={node_id!r}"
        )


def save_evaluation_charter(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    producer: str,
) -> str:
    assert_evaluation_charter_valid(payload)
    return save_contract_artifact(
        project_root,
        "evaluation_charter",
        payload,
        producer=producer,
        depends_on=["science_constitution"],
        notes="Locked, mutually exclusive evaluation authorities.",
    )


def save_benchmark_manifest(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    charter: dict[str, Any],
    producer: str,
) -> str:
    check = validate_benchmark_manifest(payload, charter=charter)
    if not check["passed"]:
        raise EvaluationGovernanceError(
            "benchmark invalid: " + ", ".join(check["errors"])
        )
    path = artifact_path(project_root, "evaluation_benchmarks")
    append_jsonl_artifact(path, payload)
    update_pipeline_artifact(
        project_root,
        "evaluation_benchmarks",
        status="ready",
        producer=producer,
        depends_on=["evaluation_charter"],
        notes="Append-only opaque benchmark custody manifests.",
    )
    return str(path)


def save_evaluation_report(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    producer: str,
) -> str:
    check = validate_evaluation_decision(payload)
    approved = check["passed"] and payload.get("decision") == "approved"
    output = save_contract_artifact(
        project_root,
        "evaluation_report",
        payload,
        producer=producer,
        depends_on=["evaluation_charter", "evaluation_benchmarks"],
        warnings=list(check["errors"] or payload.get("required_failures") or []),
        recovery_hint=(
            None
            if approved
            else "Complete sealed, prospective, and independent external evaluation."
        ),
    )
    if not approved:
        update_pipeline_artifact(
            project_root,
            "evaluation_report",
            status="blocked",
            producer=producer,
            depends_on=["evaluation_charter", "evaluation_benchmarks"],
            warnings=list(check["errors"] or payload.get("required_failures") or []),
            recovery_hint=(
                "Complete sealed, prospective, and independent external evaluation."
            ),
        )
    return output


__all__ = [
    "CORE_EVALUATION_POLICY",
    "EVALUATION_LAYERS",
    "EVALUATION_ROLES",
    "EvaluationGovernanceError",
    "assert_evaluation_charter_valid",
    "assert_scientific_promotion_allowed",
    "build_benchmark_manifest",
    "build_evaluation_charter",
    "build_evaluation_decision",
    "build_evaluation_run",
    "save_benchmark_manifest",
    "save_evaluation_charter",
    "save_evaluation_report",
    "validate_benchmark_manifest",
    "validate_evaluation_charter",
    "validate_evaluation_decision",
    "validate_evaluation_run",
]
