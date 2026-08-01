from __future__ import annotations

"""Fail-closed contracts for confirmatory autonomous research.

Exploratory runs remain cheap and permissive. Results may only enter the
manuscript as confirmed evidence after a preregistration is locked and an
independent, clean-room verifier passes every required criterion below.
"""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.pipeline_contracts import (
    save_contract_artifact,
    update_pipeline_artifact,
)

SCHEMA_VERSION = 1
PLACEHOLDER_VALUES = {
    "dataset_to_be_selected",
    "primary_task_metric",
    "strong_existing_baseline",
    "unspecified",
    "unknown",
}
SUPPORTED_CORRECTIONS = {"bonferroni", "holm", "benjamini-hochberg"}


class ResearchIntegrityError(ValueError):
    """Raised when a research artifact would weaken an integrity guarantee."""


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


def _is_placeholder(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or text in PLACEHOLDER_VALUES or "to_be_selected" in text


def _direction_for_metric(metric: str) -> str:
    lowered = str(metric or "").lower()
    if any(
        token in lowered
        for token in ("loss", "error", "rmse", "mse", "mae", "latency", "cost")
    ):
        return "lower_is_better"
    return "higher_is_better"


def _registration_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key
        not in {
            "created_at",
            "locked_at",
            "registration_hash",
            "status",
            "deviations",
        }
    }


def build_preregistration(
    idea_card: dict[str, Any],
    research_plan: dict[str, Any],
    *,
    alpha: float = 0.05,
    minimum_independent_seeds: int = 3,
    minimum_independent_reproductions: int = 1,
) -> dict[str, Any]:
    """Create a draft preregistration from existing planning artifacts."""

    tasks = [
        item for item in research_plan.get("tasks") or [] if isinstance(item, dict)
    ]
    outcomes: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        metric = str(task.get("metric") or "").strip()
        outcomes.append(
            {
                "task_id": str(task.get("task_id") or f"task_{index}"),
                "dataset": str(task.get("dataset") or "").strip(),
                "metric": metric,
                "baseline": str(task.get("baseline") or "").strip(),
                "direction": _direction_for_metric(metric),
                "primary": index == 0,
                "minimum_effect": None,
            }
        )

    alternative = str(
        idea_card.get("core_hypothesis")
        or idea_card.get("title")
        or idea_card.get("name")
        or ""
    ).strip()
    primary = outcomes[0] if outcomes else {}
    baseline = str(primary.get("baseline") or "the declared baseline")
    metric = str(primary.get("metric") or "the primary metric")
    failure_criteria = [
        str(item).strip()
        for item in idea_card.get("failure_criteria") or []
        if str(item).strip()
    ]
    preregistration_id = (
        "prereg_"
        + _canonical_hash(
            {
                "idea_id": idea_card.get("idea_id"),
                "plan_id": research_plan.get("plan_id"),
                "alternative": alternative,
                "outcomes": outcomes,
            }
        ).split(":", 1)[1][:16]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "preregistration_id": preregistration_id,
        "idea_id": idea_card.get("idea_id"),
        "plan_id": research_plan.get("plan_id"),
        "created_at": _now_iso(),
        "locked_at": None,
        "registered_by": None,
        "status": "draft",
        "registration_hash": None,
        "research_question": str(idea_card.get("title") or alternative).strip(),
        "hypotheses": {
            "alternative": alternative,
            "null": f"The proposed intervention does not improve {metric} over {baseline}.",
            "falsifiers": failure_criteria
            or ["The primary outcome fails to exceed the preregistered baseline."],
        },
        "outcomes": outcomes,
        "analysis_plan": {
            "alpha": float(alpha),
            "multiple_comparison_correction": "holm",
            "family_size": max(len(outcomes), 1),
            "minimum_independent_seeds": max(int(minimum_independent_seeds), 1),
            "minimum_independent_reproductions": max(
                int(minimum_independent_reproductions), 1
            ),
            "interim_looks": 0,
            "stopping_rule": (
                "Stop only after the preregistered seed count is complete, a kill "
                "criterion fires, or the fixed execution budget is exhausted."
            ),
        },
        "data_policy": {
            "holdout_required": True,
            "blind_evaluation": True,
            "test_labels_visible_to_research_agent": False,
            "split_hashes": {},
        },
        "controls": {
            "negative_controls": [
                "Evaluate the unchanged baseline under the identical data split and seed set."
            ],
            "confounders": [
                "dataset split drift",
                "test-set leakage",
                "unequal baseline tuning",
                "seed selection",
            ],
        },
        "deviations": [],
    }


def validate_preregistration(
    payload: dict[str, Any], *, require_locked: bool = False
) -> dict[str, Any]:
    """Validate completeness and, optionally, immutability of a registration."""

    errors: list[str] = []
    warnings: list[str] = []
    hypotheses = (
        payload.get("hypotheses") if isinstance(payload.get("hypotheses"), dict) else {}
    )
    outcomes = [
        item for item in payload.get("outcomes") or [] if isinstance(item, dict)
    ]
    analysis = (
        payload.get("analysis_plan")
        if isinstance(payload.get("analysis_plan"), dict)
        else {}
    )
    data_policy = (
        payload.get("data_policy")
        if isinstance(payload.get("data_policy"), dict)
        else {}
    )
    controls = (
        payload.get("controls") if isinstance(payload.get("controls"), dict) else {}
    )

    if not str(payload.get("research_question") or "").strip():
        errors.append("research_question_missing")
    if not str(hypotheses.get("alternative") or "").strip():
        errors.append("alternative_hypothesis_missing")
    if not str(hypotheses.get("null") or "").strip():
        errors.append("null_hypothesis_missing")
    if not hypotheses.get("falsifiers"):
        errors.append("falsification_criteria_missing")
    if not outcomes or not any(item.get("primary") for item in outcomes):
        errors.append("primary_outcome_missing")
    for outcome in outcomes:
        task_id = str(outcome.get("task_id") or "unknown")
        for field in ("dataset", "metric", "baseline"):
            if _is_placeholder(outcome.get(field)):
                errors.append(f"{task_id}_{field}_unresolved")
    alpha = analysis.get("alpha")
    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not 0 < float(alpha) < 1
    ):
        errors.append("invalid_alpha")
    correction = str(analysis.get("multiple_comparison_correction") or "")
    if correction not in SUPPORTED_CORRECTIONS:
        errors.append("multiple_comparison_correction_missing")
    if int(analysis.get("minimum_independent_seeds") or 0) < 1:
        errors.append("minimum_seed_count_missing")
    if not str(analysis.get("stopping_rule") or "").strip():
        errors.append("stopping_rule_missing")
    if not data_policy.get("holdout_required") or not data_policy.get(
        "blind_evaluation"
    ):
        errors.append("blind_holdout_policy_missing")
    if data_policy.get("test_labels_visible_to_research_agent") is not False:
        errors.append("test_label_boundary_not_enforced")
    if not controls.get("negative_controls"):
        errors.append("negative_control_missing")

    if require_locked:
        split_hashes = data_policy.get("split_hashes") or {}
        task_ids = {str(item.get("task_id") or "") for item in outcomes}
        if not task_ids or any(
            not _is_content_hash(split_hashes.get(task_id)) for task_id in task_ids
        ):
            errors.append("dataset_splits_not_hash_locked")
        if payload.get("status") != "locked" or not payload.get("locked_at"):
            errors.append("preregistration_not_locked")
        expected_hash = _canonical_hash(_registration_core(payload))
        if payload.get("registration_hash") != expected_hash:
            errors.append("registration_hash_mismatch")
    elif payload.get("status") != "locked":
        warnings.append("draft_registration_cannot_authorize_confirmatory_claims")

    return {
        "ok": not errors,
        "status": "ready" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def lock_preregistration(
    payload: dict[str, Any],
    *,
    split_hashes: dict[str, str],
    registered_by: str,
) -> dict[str, Any]:
    """Seal a registration; subsequent edits invalidate its content hash."""

    locked = deepcopy(payload)
    locked.setdefault("data_policy", {})["split_hashes"] = dict(split_hashes)
    locked["registered_by"] = str(registered_by or "").strip()
    if not locked["registered_by"]:
        raise ResearchIntegrityError("registered_by is required")
    locked["status"] = "locked"
    locked["locked_at"] = _now_iso()
    locked["registration_hash"] = _canonical_hash(_registration_core(locked))
    report = validate_preregistration(locked, require_locked=True)
    if not report["ok"]:
        raise ResearchIntegrityError(
            "Cannot lock incomplete preregistration: " + ", ".join(report["errors"])
        )
    return locked


def _criterion(
    criterion_id: str, passed: bool, detail: str, *, required: bool = True
) -> dict[str, Any]:
    return {
        "id": criterion_id,
        "passed": bool(passed),
        "required": bool(required),
        "detail": detail,
    }


def build_verification_report(
    preregistration: dict[str, Any],
    experiment_records: Iterable[dict[str, Any]],
    *,
    verifier_id: str,
    clean_room: bool,
) -> dict[str, Any]:
    """Evaluate whether experiment evidence is safe to promote into claims."""

    records = [dict(item) for item in experiment_records if isinstance(item, dict)]
    confirmatory = [
        item
        for item in records
        if str(item.get("study_phase") or "").lower() == "confirmatory"
        or bool(item.get("entered_storyline"))
    ]
    reproductions = [
        item for item in records if bool(item.get("independent_reproduction"))
    ]
    outcomes = [
        item for item in preregistration.get("outcomes") or [] if isinstance(item, dict)
    ]
    outcome_by_task = {str(item.get("task_id")): item for item in outcomes}
    analysis = preregistration.get("analysis_plan") or {}
    split_hashes = (preregistration.get("data_policy") or {}).get("split_hashes") or {}
    minimum_seeds = max(int(analysis.get("minimum_independent_seeds") or 1), 1)
    minimum_reproductions = max(
        int(analysis.get("minimum_independent_reproductions") or 1), 1
    )
    producer_ids = {
        str(item.get("producer_id") or "").strip()
        for item in confirmatory
        if str(item.get("producer_id") or "").strip()
    }

    prereg_check = validate_preregistration(preregistration, require_locked=True)
    task_coverage = all(
        any(str(record.get("task_id")) == task_id for record in confirmatory)
        for task_id in outcome_by_task
    ) and bool(outcome_by_task)
    fidelity = bool(confirmatory) and all(
        str(record.get("dataset") or "")
        == str(outcome_by_task.get(str(record.get("task_id")), {}).get("dataset") or "")
        and str(record.get("metric") or "")
        == str(outcome_by_task.get(str(record.get("task_id")), {}).get("metric") or "")
        and str(record.get("baseline_ref") or "")
        == str(
            outcome_by_task.get(str(record.get("task_id")), {}).get("baseline") or ""
        )
        for record in confirmatory
        if str(record.get("task_id")) in outcome_by_task
    )
    split_integrity = bool(confirmatory) and all(
        record.get("dataset_split_hash") == split_hashes.get(str(record.get("task_id")))
        for record in confirmatory
    )
    deterministic = bool(confirmatory) and all(
        record.get("metric_provenance") == "deterministic_verified"
        and _is_content_hash(record.get("evaluator_input_hash"))
        and _is_content_hash(record.get("evaluator_result_hash"))
        for record in confirmatory
    )
    blind = bool(confirmatory) and all(
        record.get("holdout_access") == "verifier_only" for record in confirmatory
    )
    seed_coverage = bool(outcome_by_task) and all(
        len(
            {
                record.get("seed")
                for record in confirmatory
                if str(record.get("task_id")) == task_id
                and record.get("seed") is not None
            }
        )
        >= minimum_seeds
        for task_id in outcome_by_task
    )
    verifier_independent = bool(verifier_id) and all(
        verifier_id != producer_id for producer_id in producer_ids
    )
    reproduction_links = {
        str(item.get("replicates_record_id") or "").strip()
        for item in reproductions
        if str(item.get("replicates_record_id") or "").strip()
        and bool(item.get("clean_room"))
        and str(item.get("verifier_id") or "") == verifier_id
    }
    primary_record_ids = {
        str(item.get("record_id") or "").strip()
        for item in confirmatory
        if str(item.get("record_id") or "").strip()
    }
    reproduction_ok = (
        len(reproduction_links & primary_record_ids) >= minimum_reproductions
    )
    deviation_ok = not any(
        not bool(item.get("approved_before_unblinding"))
        for item in preregistration.get("deviations") or []
        if isinstance(item, dict)
    )

    criteria = [
        _criterion(
            "locked_preregistration",
            prereg_check["ok"],
            ", ".join(prereg_check["errors"]) or "locked and hash-valid",
        ),
        _criterion(
            "confirmatory_records", bool(confirmatory), f"count={len(confirmatory)}"
        ),
        _criterion(
            "task_coverage", task_coverage, f"registered_tasks={len(outcome_by_task)}"
        ),
        _criterion(
            "protocol_fidelity",
            fidelity,
            "dataset, metric, and baseline match preregistration",
        ),
        _criterion(
            "split_integrity",
            split_integrity,
            "all confirmatory runs use the locked split hashes",
        ),
        _criterion(
            "deterministic_metrics",
            deterministic,
            "metrics and evaluator inputs/results are content-addressed",
        ),
        _criterion(
            "blind_holdout", blind, "research producers cannot access holdout labels"
        ),
        _criterion(
            "seed_coverage",
            seed_coverage,
            f"minimum independent seeds per task={minimum_seeds}",
        ),
        _criterion(
            "independent_verifier",
            verifier_independent,
            f"verifier={verifier_id or 'missing'}",
        ),
        _criterion(
            "clean_room", bool(clean_room), "verification runs in a fresh environment"
        ),
        _criterion(
            "independent_reproduction",
            reproduction_ok,
            f"linked clean-room reproductions={len(reproduction_links & primary_record_ids)}/{minimum_reproductions}",
        ),
        _criterion(
            "deviation_control",
            deviation_ok,
            "all deviations approved before unblinding",
        ),
    ]
    failures = [
        item["id"] for item in criteria if item["required"] and not item["passed"]
    ]
    status = "verified" if not failures else "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "preregistration_id": preregistration.get("preregistration_id"),
        "registration_hash": preregistration.get("registration_hash"),
        "verifier_id": str(verifier_id or "").strip() or None,
        "clean_room": bool(clean_room),
        "status": status,
        "claim_promotion_allowed": status == "verified",
        "criteria": criteria,
        "required_failures": failures,
        "confirmatory_record_ids": [item.get("record_id") for item in confirmatory],
        "reproduction_record_ids": [item.get("record_id") for item in reproductions],
        "report_hash": _canonical_hash(
            {
                "registration_hash": preregistration.get("registration_hash"),
                "verifier_id": verifier_id,
                "clean_room": bool(clean_room),
                "criteria": criteria,
            }
        ),
    }


def assert_claim_promotion_allowed(report: dict[str, Any]) -> None:
    if report.get("status") != "verified" or not report.get("claim_promotion_allowed"):
        failures = ", ".join(report.get("required_failures") or []) or "unknown"
        raise ResearchIntegrityError(
            "Confirmatory claims are blocked by the verification gate: " + failures
        )


def save_preregistration(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    producer: str,
) -> str:
    check = validate_preregistration(
        payload, require_locked=payload.get("status") == "locked"
    )
    output_path = save_contract_artifact(
        project_root,
        "preregistration",
        payload,
        producer=producer,
        depends_on=["idea_cards", "research_plan"],
        warnings=check["errors"] + check["warnings"],
        recovery_hint=(
            None
            if check["ok"] and payload.get("status") == "locked"
            else "Resolve datasets, metrics, baselines and lock split hashes before confirmatory execution."
        ),
    )
    if not (check["ok"] and payload.get("status") == "locked"):
        update_pipeline_artifact(
            project_root,
            "preregistration",
            status="blocked",
            producer=producer,
            depends_on=["idea_cards", "research_plan"],
            warnings=check["errors"] + check["warnings"],
            recovery_hint=(
                "Resolve datasets, metrics, baselines and lock split hashes before "
                "confirmatory execution."
            ),
        )
    return output_path


def save_verification_report(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    producer: str,
) -> str:
    verified = payload.get("status") == "verified"
    output_path = save_contract_artifact(
        project_root,
        "verification_report",
        payload,
        producer=producer,
        depends_on=["preregistration", "experiment_registry"],
        warnings=list(payload.get("required_failures") or []),
        recovery_hint=(
            None
            if verified
            else "Run blind, deterministic, multi-seed confirmation and clean-room reproduction."
        ),
    )
    if not verified:
        update_pipeline_artifact(
            project_root,
            "verification_report",
            status="blocked",
            producer=producer,
            depends_on=["preregistration", "experiment_registry"],
            warnings=list(payload.get("required_failures") or []),
            recovery_hint=(
                "Run blind, deterministic, multi-seed confirmation and clean-room "
                "reproduction."
            ),
        )
    return output_path


__all__ = [
    "ResearchIntegrityError",
    "assert_claim_promotion_allowed",
    "build_preregistration",
    "build_verification_report",
    "lock_preregistration",
    "save_preregistration",
    "save_verification_report",
    "validate_preregistration",
]
