from __future__ import annotations

"""Fail-closed contracts for confirmatory autonomous research.

Exploratory runs remain cheap and permissive. Results may only enter the
manuscript as confirmed evidence after a preregistration is locked and an
independent, clean-room verifier passes every required criterion below.
"""

import hashlib
import json
import math
import re
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

# Keep this list in one place so a hand-written verification report cannot omit
# a criterion that the publication gate silently assumes exists.
VERIFICATION_REQUIRED_CRITERIA = frozenset(
    {
        "locked_preregistration",
        "estimand_specification",
        "missing_data_policy",
        "null_result_interpretation",
        "confirmatory_records",
        "confirmatory_attempt_completeness",
        "record_id_integrity",
        "producer_identity",
        "registration_linkage",
        "registered_task_ids_only",
        "task_coverage",
        "protocol_fidelity",
        "split_integrity",
        "deterministic_metrics",
        "recomputed_outputs",
        "durable_records",
        "statistical_comparison",
        "blind_holdout",
        "seed_coverage",
        "independent_verifier",
        "clean_room",
        "independent_reproduction",
        "seed_independence",
        "deviation_control",
    }
)

# Generic directories and reports are useful provenance, but they are not
# result artifacts. A confirmatory record must point to at least one concrete
# output/metric/prediction file (or an explicit content-addressed manifest).
RESULT_ARTIFACT_KEYS = frozenset(
    {
        "result",
        "results",
        "metric",
        "metrics",
        "evaluation",
        "evaluation_result",
        "predictions",
        "output",
        "outputs",
        "artifact",
        "artifacts",
        "input",
        "inputs",
        "evaluator_input",
        "evaluator_result",
        "dataset_split",
    }
)

# ``verification_output_hash`` is specifically the evaluator's result, not a
# hash of an input, split description, or arbitrary provenance file.  Keep the
# accepted aliases narrower than ``RESULT_ARTIFACT_KEYS`` so a record cannot
# satisfy the recomputation check by pointing at an unrelated artifact.
RESULT_OUTPUT_ARTIFACT_KEYS = frozenset(
    {
        "result",
        "results",
        "metric",
        "metrics",
        "evaluation",
        "evaluation_result",
        "predictions",
        "output",
        "outputs",
        "artifact",
        "artifacts",
        "evaluator_result",
    }
)


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
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", text))


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_int(value: Any, default: int = 0) -> int:
    """Return a strictly positive JSON integer, otherwise ``default``.

    ``int(1.5)`` and ``int(True)`` are convenient in application code but are
    unsafe at an integrity boundary: they silently rewrite a malformed
    preregistration.  Counts are therefore accepted only as integers or
    canonical integer strings.
    """

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"\+?\d+", value.strip()):
        try:
            parsed = int(value.strip())
        except (TypeError, ValueError):
            return default
    else:
        return default
    return parsed if parsed > 0 else default


def _object_or_empty(value: Any) -> dict[str, Any]:
    """Normalize a possibly malformed JSON object without raising."""

    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    """Normalize a possibly malformed JSON list without raising."""

    return value if isinstance(value, list) else []


def _canonical_seed(value: Any) -> str | None:
    """Normalize seed identifiers and reject containers/bools/ambiguous text."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        return None
    try:
        return str(int(text))
    except (TypeError, ValueError):
        return None


def _first_numeric(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present numeric value, preserving legitimate zeroes."""

    for key in keys:
        if key in mapping and _is_number(mapping.get(key)):
            return mapping.get(key)
    return None


def _result_artifact_paths(
    record: dict[str, Any], base_folder: str | Path | None = None
) -> list[tuple[str, Path]]:
    """Resolve result paths and reject paths outside the verification root."""

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    root = Path(base_folder).expanduser().resolve() if base_folder is not None else None
    resolved_paths: list[tuple[str, Path]] = []
    for key, value in artifacts.items():
        key_text = str(key).lower()
        if key_text not in RESULT_ARTIFACT_KEYS:
            continue
        # Accept the small manifest shape used by external runners without
        # treating a hash-only entry as a file.  The callers below always
        # verify the resolved bytes, so this is only path normalization.
        if isinstance(value, dict):
            values = value.get("paths") or value.get("path") or value.get("file")
            values = values if isinstance(values, list) else [values]
        else:
            values = value if isinstance(value, list) else [value]
        for candidate in values:
            if not isinstance(candidate, (str, Path)) or not str(candidate).strip():
                continue
            path = Path(candidate).expanduser()
            if root is not None and not path.is_absolute():
                path = root / path
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if root is not None:
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
            resolved_paths.append((key_text, resolved))
    return resolved_paths


def _result_artifact_manifest_present(
    record: dict[str, Any], base_folder: str | Path | None = None
) -> bool:
    """Check for a concrete result artifact without trusting status text."""

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    for _key, path in _result_artifact_paths(record, base_folder):
        try:
            # Empty result files can be legitimate (for example, a valid
            # zero-row prediction shard).  Recomputed-output and lineage
            # checks still require a content hash, so merely allowing the
            # regular file here does not make an empty placeholder evidence.
            if path.is_file():
                return True
        except OSError:
            continue
    return False


def _verification_metric_matches(record: dict[str, Any]) -> bool:
    summary = record.get("result_summary")
    return isinstance(summary, dict) and record.get(
        "verification_metric_hash"
    ) == _canonical_hash(summary)


def _verification_output_matches(
    record: dict[str, Any], base_folder: str | Path | None = None
) -> bool:
    expected = str(record.get("verification_output_hash") or "").strip()
    evaluator_result_hash = str(record.get("evaluator_result_hash") or "").strip()
    # The output produced by the clean-room evaluator must be the same byte
    # stream whose lineage is recorded as ``evaluator_result_hash``.  Merely
    # finding a matching hash on an unrelated artifact is not sufficient.
    if (
        not _is_content_hash(expected)
        or not _is_content_hash(evaluator_result_hash)
        or expected != evaluator_result_hash
    ):
        return False
    for key, path in _result_artifact_paths(record, base_folder):
        if key not in RESULT_OUTPUT_ARTIFACT_KEYS:
            continue
        try:
            if path.is_file():
                digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                if digest == expected:
                    return True
        except OSError:
            continue
    return False


def _deterministic_lineage_matches(
    record: dict[str, Any], base_folder: str | Path | None = None
) -> bool:
    """Validate evaluator hashes against concrete artifact bytes when possible."""

    input_hash = record.get("evaluator_input_hash")
    result_hash = record.get("evaluator_result_hash")
    if not (_is_content_hash(input_hash) and _is_content_hash(result_hash)):
        return False
    required = {
        "evaluator_input": str(input_hash),
        "evaluator_result": str(result_hash),
    }
    aliases = {
        "evaluator_input": {"evaluator_input", "input", "inputs"},
        "evaluator_result": {
            "evaluator_result",
            "output",
            "outputs",
            "result",
            "results",
        },
    }
    for key, expected in required.items():
        matched = False
        for path_key, path in _result_artifact_paths(record, base_folder):
            if path_key not in aliases[key]:
                continue
            try:
                if path.is_file():
                    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest == expected:
                        matched = True
                        break
            except OSError:
                continue
        if not matched:
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


NULL_RESULT_STATUSES = frozenset(
    {
        "supports_effect",
        "supports_no_meaningful_effect",
        "inconclusive",
        "opposes_effect",
    }
)


def _validate_analysis_plan_contract(
    analysis: dict[str, Any], outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate the parts of a protocol that determine interpretation.

    A preregistration that only fixes alpha and a seed count still leaves the
    estimand, missing-data handling, and null-result interpretation open to
    post-hoc choices. Keep these checks structural so downstream runners can
    consume the same contract without parsing prose.
    """

    errors: list[str] = []
    estimand = analysis.get("estimand")
    estimand_ok = isinstance(estimand, dict)
    if not estimand_ok:
        errors.append("estimand_not_object")
        estimand = {}
    estimand_fields = (
        "definition",
        "population",
        "intervention",
        "comparator",
        "outcome",
        "summary_measure",
        "timepoint",
        "unit",
    )
    for field in estimand_fields:
        if _is_placeholder(estimand.get(field)):
            errors.append(f"estimand_{field}_missing")

    missing_policy = analysis.get("missing_data_policy")
    missing_policy_ok = isinstance(missing_policy, dict)
    if not missing_policy_ok:
        errors.append("missing_data_policy_not_object")
        missing_policy = {}
    for field in ("rule", "primary_analysis", "sensitivity_analysis"):
        if _is_placeholder(missing_policy.get(field)):
            errors.append(f"missing_data_{field}_missing")
    if missing_policy.get("report_counts") is not True:
        errors.append("missing_data_report_counts_missing")
    max_missing_fraction = missing_policy.get("max_missing_fraction")
    if (
        not _is_number(max_missing_fraction)
        or not 0 <= float(max_missing_fraction) <= 1
    ):
        errors.append("missing_data_max_fraction_invalid")

    null_policy = analysis.get("null_result_policy")
    null_policy_ok = isinstance(null_policy, dict)
    if not null_policy_ok:
        errors.append("null_result_policy_not_object")
        null_policy = {}
    allowed = null_policy.get("allowed_interpretations")
    allowed_values = (
        {str(item).strip() for item in allowed if str(item).strip()}
        if isinstance(allowed, list)
        else set()
    )
    if "inconclusive" not in allowed_values:
        errors.append("null_result_inconclusive_status_missing")
    if not str(null_policy.get("interpretation_status") or "").strip():
        errors.append("null_result_interpretation_status_missing")
    elif str(null_policy.get("interpretation_status")).strip() not in allowed_values:
        errors.append("null_result_interpretation_status_invalid")
    if not str(null_policy.get("zero_result_rule") or "").strip():
        errors.append("null_result_rule_missing")
    if not str(null_policy.get("support_no_effect_requires") or "").strip():
        errors.append("null_result_support_rule_missing")
    threshold = null_policy.get("near_zero_threshold")
    if not _is_number(threshold) or float(threshold) < 0:
        errors.append("null_result_threshold_invalid")
    for field in ("equivalence_margin", "precision_target"):
        value = null_policy.get(field)
        if value is not None and (
            not _is_number(value) or float(value) <= 0
        ):
            errors.append(f"null_result_{field}_invalid")
    if (
        str(null_policy.get("interpretation_status") or "").strip()
        == "supports_no_meaningful_effect"
        and null_policy.get("equivalence_margin") is None
        and null_policy.get("precision_target") is None
    ):
        errors.append("null_result_no_effect_margin_missing")

    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "estimand_ok": estimand_ok and not any(
            error.startswith("estimand_") for error in errors
        ),
        "missing_data_policy_ok": missing_policy_ok and not any(
            error.startswith("missing_data_") for error in errors
        ),
        "null_result_policy_ok": null_policy_ok and not any(
            error.startswith("null_result_") for error in errors
        ),
    }


def _record_comparison(record: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    summary = record.get("result_summary")
    if not isinstance(summary, dict):
        return None, None, None, {}
    comparison = summary.get("comparison")
    source = comparison if isinstance(comparison, dict) else summary
    candidate = _first_numeric(source, ("candidate", "candidate_mean"))
    if candidate is None and source is summary:
        candidate = _first_numeric(
            summary, ("metric_mean", "metric_value", "value", "final_metric")
        )
    baseline = _first_numeric(source, ("baseline", "baseline_mean"))
    if baseline is None and source is summary:
        baseline = _first_numeric(
            summary, ("baseline_metric_mean", "baseline_value", "baseline_mean")
        )
    delta = _first_numeric(source, ("delta", "delta_vs_baseline"))
    if delta is None and source is summary:
        delta = _first_numeric(summary, ("delta_vs_baseline", "metric_delta", "delta"))
    return candidate, baseline, delta, summary


def _result_interval(summary: dict[str, Any]) -> tuple[float, float] | None:
    for key in (
        "delta_confidence_interval",
        "confidence_interval_delta",
        "confidence_interval",
    ):
        interval = summary.get(key)
        if (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and all(_is_number(value) for value in interval)
            and float(interval[0]) <= float(interval[1])
        ):
            return float(interval[0]), float(interval[1])
    return None


def classify_result_interpretation(
    record: dict[str, Any],
    outcome: dict[str, Any] | None = None,
    analysis_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a result without turning a near-zero estimate into no effect."""

    outcome = outcome if isinstance(outcome, dict) else {}
    analysis = analysis_plan if isinstance(analysis_plan, dict) else {}
    policy = _object_or_empty(analysis.get("null_result_policy"))
    candidate, baseline, delta, summary = _record_comparison(record)
    threshold = outcome.get("null_effect_threshold")
    if not _is_number(threshold):
        threshold = policy.get("near_zero_threshold")
    if not _is_number(threshold) or float(threshold) < 0:
        threshold = 0.0
    if not _is_number(delta):
        return {
            "status": "not_evaluable",
            "valid": False,
            "claim_safe": False,
            "is_null": False,
            "delta": None,
            "threshold": float(threshold),
            "reason": "baseline_comparison_missing",
        }

    delta_value = float(delta)
    is_null = abs(delta_value) <= float(threshold)
    allowed = {
        str(item).strip()
        for item in (policy.get("allowed_interpretations") or [])
        if str(item).strip()
    }
    declared = str(summary.get("interpretation_status") or "").strip()
    if not is_null:
        direction = str(outcome.get("direction") or "higher_is_better")
        improves = delta_value > 0 if direction != "lower_is_better" else delta_value < 0
        derived_status = "supports_effect" if improves else "opposes_effect"
        valid = not declared or declared == derived_status
        return {
            "status": declared or derived_status,
            "valid": valid and (not declared or declared in allowed),
            "claim_safe": valid and derived_status == "supports_effect",
            "is_null": False,
            "delta": delta_value,
            "threshold": float(threshold),
            "reason": "non_null_effect_estimate",
        }

    if not declared:
        return {
            "status": "inconclusive",
            "valid": False,
            "claim_safe": False,
            "is_null": True,
            "delta": delta_value,
            "threshold": float(threshold),
            "reason": "explicit_interpretation_status_required",
        }
    if declared not in allowed or declared not in NULL_RESULT_STATUSES:
        return {
            "status": "inconclusive",
            "valid": False,
            "claim_safe": False,
            "is_null": True,
            "delta": delta_value,
            "threshold": float(threshold),
            "reason": "interpretation_status_not_allowed",
        }
    if declared == "inconclusive":
        return {
            "status": declared,
            "valid": True,
            "claim_safe": False,
            "is_null": True,
            "delta": delta_value,
            "threshold": float(threshold),
            "reason": "precision_or_equivalence_evidence_not_sufficient",
        }
    if declared != "supports_no_meaningful_effect":
        return {
            "status": declared,
            "valid": False,
            "claim_safe": False,
            "is_null": True,
            "delta": delta_value,
            "threshold": float(threshold),
            "reason": "near_zero_result_cannot_support_effect_claim",
        }

    margin = policy.get("equivalence_margin")
    if not _is_number(margin):
        margin = outcome.get("meaningful_effect_margin")
    precision_target = policy.get("precision_target")
    interval = _result_interval(summary)
    equivalence_test = summary.get("equivalence_test")
    equivalence_passed = (
        isinstance(equivalence_test, dict)
        and equivalence_test.get("passed") is True
    ) or summary.get("equivalence_passed") is True
    interval_support = bool(
        interval
        and _is_number(margin)
        and interval[0] >= -float(margin)
        and interval[1] <= float(margin)
    )
    precision_support = bool(
        interval
        and _is_number(precision_target)
        and interval[1] - interval[0] <= 2 * float(precision_target)
    )
    supported = equivalence_passed or interval_support or precision_support
    return {
        "status": declared,
        "valid": supported,
        "claim_safe": supported,
        "is_null": True,
        "delta": delta_value,
        "threshold": float(threshold),
        "reason": (
            "equivalence_or_precision_rule_satisfied"
            if supported
            else "equivalence_or_precision_evidence_missing"
        ),
        "evidence": {
            "equivalence_test_passed": equivalence_passed,
            "confidence_interval": list(interval) if interval else None,
            "equivalence_margin": margin if _is_number(margin) else None,
            "precision_target": (
                precision_target if _is_number(precision_target) else None
            ),
        },
    }


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


def _protocol_fidelity_hash(preregistration: dict[str, Any], task_id: str) -> str:
    """Hash the locked protocol fields that a confirmatory run must follow."""

    outcomes = [
        item
        for item in _list_or_empty(preregistration.get("outcomes"))
        if isinstance(item, dict)
        and str(item.get("task_id") or "").strip() == str(task_id).strip()
    ]
    analysis_plan = _object_or_empty(preregistration.get("analysis_plan"))
    data_policy = _object_or_empty(preregistration.get("data_policy"))
    controls = _object_or_empty(preregistration.get("controls"))
    return _canonical_hash(
        {
            "outcome": outcomes[0] if outcomes else None,
            "analysis_plan": analysis_plan,
            "data_policy": data_policy,
            "controls": controls,
        }
    )


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
                "null_effect_threshold": 0.0,
                "meaningful_effect_margin": None,
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
            "minimum_independent_seeds": _positive_int(minimum_independent_seeds, 1),
            "minimum_independent_reproductions": _positive_int(
                minimum_independent_reproductions, 1
            ),
            "interim_looks": 0,
            "stopping_rule": (
                "Stop only after the preregistered seed count is complete, a kill "
                "criterion fires, or the fixed execution budget is exhausted."
            ),
            "estimand": {
                "definition": (
                    "The mean difference in the primary outcome between the "
                    "proposed intervention and the declared baseline on the "
                    "locked evaluation population at the fixed endpoint."
                ),
                "population": "all examples in the locked evaluation split",
                "intervention": "the proposed intervention under test",
                "comparator": baseline,
                "outcome": metric,
                "summary_measure": "difference in mean task metric",
                "timepoint": "single preregistered evaluation endpoint",
                "unit": "task-level evaluation aggregate",
            },
            "missing_data_policy": {
                "rule": (
                    "Do not silently impute or drop observations; record missing "
                    "counts and reasons for every task."
                ),
                "primary_analysis": "complete_case_with_missingness_report",
                "sensitivity_analysis": (
                    "report a conservative missing-as-failure sensitivity analysis "
                    "when missingness is nonzero"
                ),
                "report_counts": True,
                "max_missing_fraction": 0.0,
            },
            "null_result_policy": {
                "interpretation_status": "inconclusive",
                "allowed_interpretations": [
                    "supports_effect",
                    "supports_no_meaningful_effect",
                    "inconclusive",
                    "opposes_effect",
                ],
                "near_zero_threshold": 0.0,
                "equivalence_margin": None,
                "precision_target": None,
                "support_no_effect_requires": (
                    "an equivalence test or a confidence interval fully inside "
                    "a preregistered meaningful-effect margin"
                ),
                "zero_result_rule": (
                    "A zero or near-zero estimate without equivalence or adequate "
                    "precision is inconclusive and does not support a no-effect claim."
                ),
            },
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
        "external_validity": {
            "required": False,
            "status": "not_assessed",
            "target_conditions": [],
            "assessment_method": "",
            "evidence_artifacts": [],
        },
        "deviations": [],
    }


def validate_preregistration(
    payload: dict[str, Any], *, require_locked: bool = False
) -> dict[str, Any]:
    """Validate completeness and, optionally, immutability of a registration."""

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["preregistration_not_object"],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    raw_hypotheses = payload.get("hypotheses")
    if not isinstance(raw_hypotheses, dict):
        errors.append("hypotheses_not_object")
    hypotheses = raw_hypotheses if isinstance(raw_hypotheses, dict) else {}

    raw_outcomes = payload.get("outcomes")
    if raw_outcomes is None:
        raw_outcomes = []
    if not isinstance(raw_outcomes, list):
        errors.append("outcomes_not_list")
        raw_outcomes = []
    elif any(not isinstance(item, dict) for item in raw_outcomes):
        errors.append("outcome_entry_not_object")
    outcomes = [item for item in raw_outcomes if isinstance(item, dict)]

    raw_analysis = payload.get("analysis_plan")
    if not isinstance(raw_analysis, dict):
        errors.append("analysis_plan_not_object")
    analysis = raw_analysis if isinstance(raw_analysis, dict) else {}

    raw_data_policy = payload.get("data_policy")
    if not isinstance(raw_data_policy, dict):
        errors.append("data_policy_not_object")
    data_policy = raw_data_policy if isinstance(raw_data_policy, dict) else {}

    raw_controls = payload.get("controls")
    if not isinstance(raw_controls, dict):
        errors.append("controls_not_object")
    controls = raw_controls if isinstance(raw_controls, dict) else {}

    raw_external_validity = payload.get("external_validity")
    if raw_external_validity is not None and not isinstance(raw_external_validity, dict):
        errors.append("external_validity_not_object")
        external_validity: dict[str, Any] = {}
    else:
        external_validity = (
            raw_external_validity if isinstance(raw_external_validity, dict) else {}
        )
    if external_validity.get("required") is True:
        if str(external_validity.get("status") or "").strip() not in {
            "assessed",
            "verified",
        }:
            errors.append("external_validity_required_but_unassessed")
        if not str(external_validity.get("assessment_method") or "").strip():
            errors.append("external_validity_method_missing")
        if not isinstance(external_validity.get("evidence_artifacts"), list) or not external_validity.get(
            "evidence_artifacts"
        ):
            errors.append("external_validity_evidence_missing")

    raw_deviations = payload.get("deviations")
    if raw_deviations is not None and not isinstance(raw_deviations, list):
        errors.append("deviations_not_list")
    elif isinstance(raw_deviations, list) and any(
        not isinstance(item, dict) for item in raw_deviations
    ):
        errors.append("deviation_entry_not_object")

    if not str(payload.get("research_question") or "").strip():
        errors.append("research_question_missing")
    if not str(hypotheses.get("alternative") or "").strip():
        errors.append("alternative_hypothesis_missing")
    if not str(hypotheses.get("null") or "").strip():
        errors.append("null_hypothesis_missing")
    falsifiers = hypotheses.get("falsifiers")
    if not isinstance(falsifiers, list) or not any(
        isinstance(item, str) and item.strip() for item in falsifiers
    ):
        errors.append("falsification_criteria_missing")
    elif any(not isinstance(item, str) or not item.strip() for item in falsifiers):
        errors.append("falsification_criteria_invalid")
    # Do not let truthy strings such as ``"false"`` silently become a primary
    # outcome.  Registration fields are serialized JSON booleans and must be
    # interpreted as such at the integrity boundary.
    primary_count = sum(item.get("primary") is True for item in outcomes)
    if not outcomes or primary_count == 0:
        errors.append("primary_outcome_missing")
    elif primary_count != 1:
        errors.append("primary_outcome_count_invalid")
    task_ids = []
    for outcome in outcomes:
        task_id = str(outcome.get("task_id") or "unknown")
        task_ids.append(str(outcome.get("task_id") or "").strip())
        if not str(outcome.get("task_id") or "").strip():
            errors.append(f"{task_id}_task_id_missing")
        for field in ("dataset", "metric", "baseline"):
            if _is_placeholder(outcome.get(field)):
                errors.append(f"{task_id}_{field}_unresolved")
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate_task_ids")
    alpha = analysis.get("alpha")
    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not math.isfinite(float(alpha))
        or not 0 < float(alpha) < 1
    ):
        errors.append("invalid_alpha")
    correction = str(analysis.get("multiple_comparison_correction") or "")
    if correction not in SUPPORTED_CORRECTIONS:
        errors.append("multiple_comparison_correction_missing")
    if _positive_int(analysis.get("minimum_independent_seeds")) < 1:
        errors.append("minimum_seed_count_missing")
    if _positive_int(analysis.get("minimum_independent_reproductions")) < 1:
        errors.append("minimum_reproduction_count_missing")
    if not str(analysis.get("stopping_rule") or "").strip():
        errors.append("stopping_rule_missing")
    analysis_contract = _validate_analysis_plan_contract(analysis, outcomes)
    errors.extend(analysis_contract["errors"])
    if (
        data_policy.get("holdout_required") is not True
        or data_policy.get("blind_evaluation") is not True
    ):
        errors.append("blind_holdout_policy_missing")
    if data_policy.get("test_labels_visible_to_research_agent") is not False:
        errors.append("test_label_boundary_not_enforced")
    negative_controls = controls.get("negative_controls")
    if not isinstance(negative_controls, list) or not any(
        isinstance(item, str) and item.strip() for item in negative_controls
    ):
        errors.append("negative_control_missing")
    elif any(
        not isinstance(item, str) or not item.strip() for item in negative_controls
    ):
        errors.append("negative_control_invalid")

    # Task IDs are the join key across registration, registry and claim graph.
    # Reject missing/duplicate IDs instead of allowing dict construction to
    # silently overwrite one outcome with another.
    task_ids = [str(item.get("task_id") or "").strip() for item in outcomes]
    if any(not task_id for task_id in task_ids):
        errors.append("outcome_task_id_missing")
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate_outcome_task_id")

    if require_locked:
        if payload.get("schema_version") != SCHEMA_VERSION:
            errors.append("schema_version_invalid")
        if not str(payload.get("preregistration_id") or "").strip():
            errors.append("preregistration_id_missing")
        if not str(payload.get("registered_by") or "").strip():
            errors.append("registered_by_missing")
        locked_at = str(payload.get("locked_at") or "").strip()
        if locked_at:
            try:
                parsed_locked_at = datetime.fromisoformat(
                    locked_at.replace("Z", "+00:00")
                )
                if parsed_locked_at.tzinfo is None:
                    errors.append("locked_at_timezone_missing")
            except ValueError:
                errors.append("locked_at_invalid")
        else:
            errors.append("locked_at_invalid")
        raw_split_hashes = data_policy.get("split_hashes")
        if raw_split_hashes is None:
            split_hashes = {}
        elif isinstance(raw_split_hashes, dict):
            split_hashes = raw_split_hashes
        else:
            split_hashes = {}
            errors.append("dataset_split_hashes_not_object")
        task_id_set = {task_id for task_id in task_ids if task_id}
        if not task_id_set or any(
            not _is_content_hash(split_hashes.get(task_id)) for task_id in task_id_set
        ):
            errors.append("dataset_splits_not_hash_locked")
        if set(str(key) for key in split_hashes) != task_id_set:
            errors.append("dataset_split_task_mismatch")
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
        "analysis_plan_contract": analysis_contract,
    }


def lock_preregistration(
    payload: dict[str, Any],
    *,
    split_hashes: dict[str, str],
    registered_by: str,
) -> dict[str, Any]:
    """Seal a registration; subsequent edits invalidate its content hash."""

    if not isinstance(payload, dict):
        raise ResearchIntegrityError("preregistration payload must be an object")
    if not isinstance(split_hashes, dict):
        raise ResearchIntegrityError("split_hashes must be an object")
    locked = deepcopy(payload)
    data_policy = locked.get("data_policy")
    if not isinstance(data_policy, dict):
        raise ResearchIntegrityError("data_policy must be an object")
    data_policy["split_hashes"] = dict(split_hashes)
    locked["data_policy"] = data_policy
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


def _verification_report_hash_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable portion of a verification report.

    Timestamps are intentionally excluded. Critical status, record coverage,
    criteria, and linkage fields are included so changing any of them requires
    a new audit history entry instead of silently reusing an old hash.
    """

    def _string_list(value: Any) -> list[Any]:
        return list(value) if isinstance(value, list) else []

    return {
        "schema_version": payload.get("schema_version", SCHEMA_VERSION),
        "preregistration_id": payload.get("preregistration_id"),
        "registration_hash": payload.get("registration_hash"),
        "verifier_id": str(payload.get("verifier_id") or "").strip() or None,
        "clean_room": payload.get("clean_room") is True,
        "status": payload.get("status"),
        "claim_promotion_allowed": payload.get("claim_promotion_allowed") is True,
        "confirmatory_record_ids": _string_list(payload.get("confirmatory_record_ids")),
        "reproduction_record_ids": _string_list(payload.get("reproduction_record_ids")),
        "reproduction_task_counts": payload.get("reproduction_task_counts") or {},
        "conclusion_interpretation": payload.get("conclusion_interpretation") or {},
        "analysis_plan_contract": payload.get("analysis_plan_contract") or {},
        "records_hash": payload.get("records_hash"),
        "registry_hash": payload.get("registry_hash"),
        "evidence_snapshot_hash": payload.get("evidence_snapshot_hash"),
        "manuscript_hash": payload.get("manuscript_hash"),
        "criteria": _string_list(payload.get("criteria")),
        "required_failures": _string_list(payload.get("required_failures")),
    }


def _valid_reproduction_link(
    record: dict[str, Any],
    *,
    primary_records_by_id: dict[str, dict[str, Any]],
    registration_id: str,
    split_hashes: dict[str, Any],
    preregistration: dict[str, Any],
    verifier_id: str,
    verification_root: str | Path | None,
) -> bool:
    """Check that a reproduction is an independent replay of a primary run.

    A reproduction is not just another file with a ``replicates_record_id``.
    It must carry the same registered task and locked protocol, use the same
    blind split, and identify a producer distinct from the primary producer.
    Keeping this contract in one helper prevents the report builder and its
    callers from accidentally validating only a subset of the linkage.
    """

    record_id = str(record.get("record_id") or "").strip()
    primary_id = str(record.get("replicates_record_id") or "").strip()
    primary = primary_records_by_id.get(primary_id)
    if not record_id or not primary or record_id in primary_records_by_id:
        return False
    task_id = str(record.get("task_id") or "").strip()
    primary_task_id = str(primary.get("task_id") or "").strip()
    if not task_id or task_id != primary_task_id:
        return False
    for field in ("dataset", "metric", "baseline_ref"):
        if str(record.get(field) or "") != str(primary.get(field) or ""):
            return False
    if str(record.get("preregistration_id") or "").strip() != registration_id:
        return False
    if record.get("protocol_fidelity_hash") != _protocol_fidelity_hash(
        preregistration, task_id
    ):
        return False
    if record.get("dataset_split_hash") != split_hashes.get(task_id):
        return False
    if record.get("holdout_access") != "verifier_only":
        return False
    producer_id = str(record.get("producer_id") or "").strip()
    primary_producers = {
        str(item.get("producer_id") or "").strip()
        for item in primary_records_by_id.values()
        if str(item.get("producer_id") or "").strip()
    }
    if not producer_id or producer_id in primary_producers:
        return False
    return (
        str(record.get("verifier_id") or "").strip() == verifier_id
        and record.get("clean_room") is True
        and str(record.get("status") or "").lower() in {"completed", "verified"}
        and bool(record.get("finished_at"))
        and not record.get("error_type")
        and _result_artifact_manifest_present(record, verification_root)
        and record.get("verification_recomputed") is True
        and _deterministic_lineage_matches(record, verification_root)
        and _verification_metric_matches(record)
        and _verification_output_matches(record, verification_root)
    )


def build_verification_report(
    preregistration: dict[str, Any],
    experiment_records: Iterable[dict[str, Any]],
    *,
    verifier_id: str,
    clean_room: bool,
    verification_root: str | Path | None = None,
    evidence_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether experiment evidence is safe to promote into claims.

    ``verification_root`` lets a verifier resolve relative artifact paths
    without depending on the process working directory. A verified report must
    resolve readable artifact files; a hash-only manifest is never promoted.
    """

    normalized_verifier_id = str(verifier_id or "").strip()
    # ``clean_room`` is a security-relevant assertion.  Treat only the JSON
    # boolean ``true`` as evidence; values such as ``"false"`` must not be
    # promoted through Python truthiness coercion.
    clean_room_verified = clean_room is True
    if not isinstance(preregistration, dict):
        preregistration = {}
    if experiment_records is None:
        records = []
    else:
        try:
            records = [dict(item) for item in experiment_records if isinstance(item, dict)]
        except TypeError:
            # A malformed registry should produce a blocked report, not an
            # exception that prevents the CLI from showing recovery actions.
            records = []
    confirmatory = [
        item
        for item in records
        if str(item.get("study_phase") or "").lower() == "confirmatory"
        and str(item.get("status") or "").lower() in {"completed", "verified"}
    ]
    confirmatory_attempts = [
        item
        for item in records
        if str(item.get("study_phase") or "").lower() == "confirmatory"
    ]
    incomplete_confirmatory_attempts = [
        item
        for item in confirmatory_attempts
        if str(item.get("status") or "").lower() not in {"completed", "verified"}
    ]
    reproductions = [
        item for item in records if item.get("independent_reproduction") is True
    ]
    outcomes = [
        item
        for item in _list_or_empty(preregistration.get("outcomes"))
        if isinstance(item, dict)
    ]
    outcome_by_task = {str(item.get("task_id")): item for item in outcomes}
    raw_analysis = preregistration.get("analysis_plan")
    analysis = raw_analysis if isinstance(raw_analysis, dict) else {}
    raw_data_policy = preregistration.get("data_policy")
    data_policy = raw_data_policy if isinstance(raw_data_policy, dict) else {}
    raw_split_hashes = data_policy.get("split_hashes")
    split_hashes = raw_split_hashes if isinstance(raw_split_hashes, dict) else {}
    minimum_seeds = _positive_int(analysis.get("minimum_independent_seeds"), 1)
    minimum_reproductions = _positive_int(
        analysis.get("minimum_independent_reproductions"), 1
    )
    producer_ids = {
        str(item.get("producer_id") or "").strip()
        for item in confirmatory
        if str(item.get("producer_id") or "").strip()
    }
    confirmatory_record_ids = [
        str(item.get("record_id") or "").strip() for item in confirmatory
    ]
    relevant_record_ids = confirmatory_record_ids + [
        str(item.get("record_id") or "").strip() for item in reproductions
    ]
    record_id_integrity = (
        bool(confirmatory)
        and all(relevant_record_ids)
        and len(relevant_record_ids) == len(set(relevant_record_ids))
    )
    unknown_task_records = [
        item
        for item in confirmatory
        if str(item.get("task_id") or "") not in outcome_by_task
    ]
    producer_identity = bool(confirmatory) and all(
        str(item.get("producer_id") or "").strip() for item in confirmatory
    )
    registered_task_ids = {str(item.get("task_id") or "").strip() for item in outcomes}
    confirmatory_task_ids = {
        str(item.get("task_id") or "").strip() for item in confirmatory
    }
    registration_id = str(preregistration.get("preregistration_id") or "").strip()
    registration_linkage = (
        bool(confirmatory)
        and bool(registration_id)
        and all(
            str(record.get("preregistration_id") or "").strip() == registration_id
            for record in confirmatory
        )
    )
    no_unregistered_records = bool(confirmatory) and confirmatory_task_ids.issubset(
        registered_task_ids
    )

    prereg_check = validate_preregistration(preregistration, require_locked=True)
    analysis_contract = prereg_check.get("analysis_plan_contract") or {
        "ok": False,
        "errors": ["analysis_plan_contract_missing"],
        "estimand_ok": False,
        "missing_data_policy_ok": False,
        "null_result_policy_ok": False,
    }
    task_coverage = (
        all(
            any(str(record.get("task_id")) == task_id for record in confirmatory)
            for task_id in outcome_by_task
        )
        and bool(outcome_by_task)
        and no_unregistered_records
    )
    fidelity = bool(confirmatory) and all(
        str(record.get("task_id") or "") in outcome_by_task
        and str(record.get("dataset") or "")
        == str(outcome_by_task.get(str(record.get("task_id")), {}).get("dataset") or "")
        and str(record.get("metric") or "")
        == str(outcome_by_task.get(str(record.get("task_id")), {}).get("metric") or "")
        and str(record.get("baseline_ref") or "")
        == str(
            outcome_by_task.get(str(record.get("task_id")), {}).get("baseline") or ""
        )
        and record.get("protocol_fidelity_hash")
        == _protocol_fidelity_hash(preregistration, str(record.get("task_id") or ""))
        for record in confirmatory
    )
    split_integrity = bool(confirmatory) and all(
        str(record.get("task_id") or "") in outcome_by_task
        and record.get("dataset_split_hash")
        == split_hashes.get(str(record.get("task_id")))
        for record in confirmatory
    )
    deterministic = bool(confirmatory) and all(
        record.get("metric_provenance") == "deterministic_verified"
        and _is_content_hash(record.get("evaluator_input_hash"))
        and _is_content_hash(record.get("evaluator_result_hash"))
        and _deterministic_lineage_matches(record, verification_root)
        for record in confirmatory
    )
    durable_records = bool(confirmatory) and all(
        bool(record.get("finished_at"))
        and not record.get("error_type")
        and _result_artifact_manifest_present(record, verification_root)
        for record in confirmatory
    )

    def _has_statistical_comparison(record: dict[str, Any]) -> bool:
        summary = record.get("result_summary")
        if not isinstance(summary, dict):
            return False
        p_value = summary.get("p_value")
        standard_error = summary.get("standard_error")
        interval = summary.get("confidence_interval")
        interval_ok = (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and all(_is_number(value) for value in interval)
            and float(interval[0]) <= float(interval[1])
        )
        uncertainty = (
            any(
                _is_number(summary.get(key))
                for key in ("effect_size", "p_value", "standard_error")
            )
            or interval_ok
        )
        if p_value is not None and (
            not _is_number(p_value) or not 0 <= float(p_value) <= 1
        ):
            return False
        if standard_error is not None and (
            not _is_number(standard_error) or float(standard_error) < 0
        ):
            return False
        if interval is not None and not interval_ok:
            return False
        comparison = summary.get("comparison")
        if isinstance(comparison, dict):
            candidate = _first_numeric(comparison, ("candidate", "candidate_mean"))
            baseline = _first_numeric(comparison, ("baseline", "baseline_mean"))
            delta = _first_numeric(comparison, ("delta", "delta_vs_baseline"))
            comparison_ok = all(
                _is_number(value) for value in (candidate, baseline, delta)
            )
        else:
            candidate = _first_numeric(
                summary, ("metric_mean", "metric_value", "value", "final_metric")
            )
            baseline = _first_numeric(
                summary, ("baseline_metric_mean", "baseline_value", "baseline_mean")
            )
            delta = _first_numeric(
                summary, ("delta_vs_baseline", "metric_delta", "delta")
            )
            comparison_ok = all(
                _is_number(value) for value in (candidate, baseline, delta)
            )
        if not comparison_ok:
            return False
        return (
            uncertainty
            and abs(float(candidate) - float(baseline) - float(delta)) <= 1e-3
        )

    statistical_comparison = bool(outcome_by_task) and all(
        any(
            _has_statistical_comparison(record)
            for record in confirmatory
            if str(record.get("task_id") or "") == task_id
        )
        for task_id in outcome_by_task
    )
    recomputed_outputs = bool(confirmatory) and all(
        record.get("verification_recomputed") is True
        and bool(
            str(
                record.get("verification_command")
                or record.get("verification_method")
                or ""
            ).strip()
        )
        and _verification_metric_matches(record)
        and _verification_output_matches(record, verification_root)
        for record in confirmatory
    )
    blind = bool(confirmatory) and all(
        record.get("holdout_access") == "verifier_only" for record in confirmatory
    )
    seed_coverage = bool(outcome_by_task) and all(
        len(
            {
                _canonical_seed(record.get("seed"))
                for record in confirmatory
                if str(record.get("task_id")) == task_id
                and _canonical_seed(record.get("seed")) is not None
            }
        )
        >= minimum_seeds
        for task_id in outcome_by_task
    )
    seed_independence = bool(outcome_by_task)
    for task_id in outcome_by_task:
        task_records = [
            record
            for record in confirmatory
            if str(record.get("task_id") or "") == task_id
        ]
        input_hashes = [
            str(record.get("evaluator_input_hash") or "")
            for record in task_records
        ]
        valid_input_hashes = all(_is_content_hash(value) for value in input_hashes)
        seed_commitments_valid = True
        for record in task_records:
            seed = _canonical_seed(record.get("seed"))
            input_hash = str(record.get("evaluator_input_hash") or "")
            if seed is None or not _is_content_hash(input_hash):
                seed_commitments_valid = False
                continue
            expected_commitment = _canonical_hash(
                {"seed": seed, "evaluator_input_hash": input_hash}
            )
            explicit_commitment = record.get("seed_input_commitment")
            if explicit_commitment is not None and explicit_commitment != expected_commitment:
                seed_commitments_valid = False
        # A copied evaluator input cannot become an independent seed run by
        # changing only the JSON seed label.
        seed_independence = seed_independence and valid_input_hashes and len(
            set(input_hashes)
        ) == len(input_hashes) and seed_commitments_valid
    verifier_independent = bool(normalized_verifier_id) and all(
        normalized_verifier_id != producer_id for producer_id in producer_ids
    )
    primary_records_by_id = {
        str(record.get("record_id") or "").strip(): record
        for record in confirmatory
        if str(record.get("record_id") or "").strip()
    }
    valid_reproduction_records = [
        item
        for item in reproductions
        if _valid_reproduction_link(
            item,
            primary_records_by_id=primary_records_by_id,
            registration_id=registration_id,
            split_hashes=split_hashes,
            preregistration=preregistration,
            verifier_id=normalized_verifier_id,
            verification_root=verification_root,
        )
    ]
    valid_reproduction_links = {
        str(item.get("record_id") or "").strip()
        for item in valid_reproduction_records
    }
    reproduction_task_counts = {
        task_id: sum(
            str(item.get("task_id") or "") == task_id
            for item in valid_reproduction_records
        )
        for task_id in outcome_by_task
    }
    reproduction_ok = bool(outcome_by_task) and all(
        count >= minimum_reproductions for count in reproduction_task_counts.values()
    )
    deviations = _list_or_empty(preregistration.get("deviations"))
    deviation_ok = not any(
        item.get("approved_before_unblinding") is not True
        for item in deviations
        if isinstance(item, dict)
    ) and all(isinstance(item, dict) for item in deviations)

    interpretation_by_task: dict[str, dict[str, Any]] = {}
    null_result_ok = True
    inconclusive_task_ids: list[str] = []
    unsupported_no_effect_task_ids: list[str] = []
    conclusion_statuses: list[str] = []
    for task_id, outcome in outcome_by_task.items():
        task_records = [
            record
            for record in confirmatory
            if str(record.get("task_id") or "") == task_id
        ]
        details = [
            {
                "record_id": str(record.get("record_id") or ""),
                **classify_result_interpretation(record, outcome, analysis),
            }
            for record in task_records
        ]
        null_details = [item for item in details if item.get("is_null") is True]
        if any(item.get("valid") is not True for item in null_details):
            null_result_ok = False
        if any(item.get("status") == "inconclusive" for item in details) or any(
            item.get("status") == "not_evaluable" for item in details
        ):
            task_status = "inconclusive"
            inconclusive_task_ids.append(task_id)
        elif any(
            item.get("status") == "supports_no_meaningful_effect" for item in details
        ):
            task_status = "supports_no_meaningful_effect"
            if any(item.get("valid") is not True for item in null_details):
                unsupported_no_effect_task_ids.append(task_id)
        elif details and all(
            item.get("status") == "supports_effect" for item in details
        ):
            task_status = "supports_effect"
        elif details and all(
            item.get("status") == "opposes_effect" for item in details
        ):
            task_status = "opposes_effect"
        else:
            task_status = "mixed"
        conclusion_statuses.append(task_status)
        interpretation_by_task[task_id] = {
            "status": task_status,
            "record_count": len(details),
            "details": details,
        }
    if not interpretation_by_task:
        conclusion_status = "not_evaluable"
    elif "inconclusive" in conclusion_statuses or "mixed" in conclusion_statuses:
        conclusion_status = "inconclusive"
    elif "opposes_effect" in conclusion_statuses:
        conclusion_status = "opposes_effect"
    elif all(
        status == "supports_no_meaningful_effect" for status in conclusion_statuses
    ):
        conclusion_status = "supports_no_meaningful_effect"
    else:
        conclusion_status = "supports_effect"
    conclusion_interpretation = {
        "status": conclusion_status,
        "by_task": interpretation_by_task,
        "inconclusive_task_ids": sorted(set(inconclusive_task_ids)),
        "unsupported_no_effect_task_ids": sorted(set(unsupported_no_effect_task_ids)),
        "null_result_records_checked": sum(
            len(
                [
                    item
                    for item in details.get("details") or []
                    if item.get("is_null") is True
                ]
            )
            for details in interpretation_by_task.values()
        ),
    }

    criteria = [
        _criterion(
            "locked_preregistration",
            prereg_check["ok"],
            ", ".join(prereg_check["errors"]) or "locked and hash-valid",
        ),
        _criterion(
            "estimand_specification",
            analysis_contract.get("estimand_ok") is True,
            "estimand defines population, intervention, comparator, outcome, and measure",
        ),
        _criterion(
            "missing_data_policy",
            analysis_contract.get("missing_data_policy_ok") is True,
            "missing observations have a primary rule, sensitivity rule, and count reporting",
        ),
        _criterion(
            "null_result_interpretation",
            analysis_contract.get("null_result_policy_ok") is True and null_result_ok,
            (
                "near-zero results are explicitly marked inconclusive or supported "
                "by preregistered equivalence/precision evidence"
            ),
        ),
        _criterion(
            "confirmatory_records", bool(confirmatory), f"count={len(confirmatory)}"
        ),
        _criterion(
            "confirmatory_attempt_completeness",
            not incomplete_confirmatory_attempts,
            "all confirmatory attempts are completed or verified",
        ),
        _criterion(
            "record_id_integrity",
            record_id_integrity,
            "confirmatory records have unique, non-empty record_id values",
        ),
        _criterion(
            "producer_identity",
            producer_identity,
            "confirmatory records identify their producer",
        ),
        _criterion(
            "registration_linkage",
            registration_linkage,
            "confirmatory records reference the locked preregistration",
        ),
        _criterion(
            "registered_task_ids_only",
            not unknown_task_records,
            f"unknown_confirmatory_tasks={len(unknown_task_records)}",
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
            "recomputed_outputs",
            recomputed_outputs,
            "clean-room verifier recomputed metrics and output artifacts",
        ),
        _criterion(
            "durable_records",
            durable_records,
            "confirmatory records contain finished, persisted artifacts",
        ),
        _criterion(
            "statistical_comparison",
            statistical_comparison,
            "confirmatory records include uncertainty and baseline comparison",
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
            "seed_independence",
            seed_independence,
            "each seed has a distinct content-addressed evaluator input",
        ),
        _criterion(
            "independent_verifier",
            verifier_independent,
            f"verifier={normalized_verifier_id or 'missing'}",
        ),
        _criterion(
            "clean_room",
            clean_room_verified,
            "verification runs in a fresh environment",
        ),
        _criterion(
            "independent_reproduction",
            reproduction_ok,
            f"linked clean-room reproductions={len(valid_reproduction_links)}/{minimum_reproductions}",
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
    registry_hash = None
    registry_records_match = True
    snapshot_payload = evidence_snapshot if isinstance(evidence_snapshot, dict) else None
    if verification_root is not None:
        root = Path(verification_root).expanduser().resolve()
        registry_path = root / "experiment_registry.jsonl"
        if registry_path.is_file():
            try:
                raw_registry = registry_path.read_bytes()
                registry_hash = "sha256:" + hashlib.sha256(raw_registry).hexdigest()
                registry_rows = []
                for line in raw_registry.decode("utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            registry_records_match = False
                            break
                        registry_rows.append(row)
                registry_records_match = registry_records_match and (
                    _canonical_hash(registry_rows) == _canonical_hash(records)
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                registry_records_match = False
        if snapshot_payload is None and (
            registry_path.is_file() or (root / "latex" / "template.tex").is_file()
        ):
            try:
                from ai_scientist.utils.evidence_snapshot import build_evidence_snapshot

                snapshot_payload = build_evidence_snapshot(root, records=records)
            except (OSError, TypeError, ValueError):
                snapshot_payload = None
    registry_content_criterion = _criterion(
        "registry_content_binding",
        registry_records_match,
        "verification records match the on-disk registry byte stream",
        required=registry_hash is not None,
    )
    criteria.append(registry_content_criterion)
    if registry_content_criterion["required"] and not registry_content_criterion["passed"]:
        failures.append(registry_content_criterion["id"])
    failures = sorted(set(failures))
    status = "verified" if not failures else "blocked"
    records_hash = _canonical_hash(records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "preregistration_id": preregistration.get("preregistration_id"),
        "registration_hash": preregistration.get("registration_hash"),
        "verifier_id": normalized_verifier_id or None,
        "clean_room": clean_room_verified,
        "status": status,
        "claim_promotion_allowed": status == "verified"
        and conclusion_status
        in {"supports_effect", "supports_no_meaningful_effect"},
        "criteria": criteria,
        "required_failures": failures,
        "confirmatory_record_ids": [item.get("record_id") for item in confirmatory],
        "reproduction_record_ids": [item.get("record_id") for item in reproductions],
        "reproduction_task_counts": reproduction_task_counts,
        "analysis_plan_contract": analysis_contract,
        "conclusion_interpretation": conclusion_interpretation,
        "records_hash": records_hash,
        "registry_hash": registry_hash,
        "evidence_snapshot_hash": (
            snapshot_payload.get("snapshot_hash") if snapshot_payload else None
        ),
        "manuscript_hash": (
            snapshot_payload.get("manuscript_hash") if snapshot_payload else None
        ),
    }
    report["report_hash"] = _canonical_hash(_verification_report_hash_payload(report))
    return report


def assert_claim_promotion_allowed(report: dict[str, Any]) -> None:
    # Keep this public assertion as strict as the publication gate.  A
    # hand-written report containing the truthy string ``"false"`` must not
    # authorize claims through Python's generic truthiness rules.
    if (
        report.get("status") != "verified"
        or report.get("claim_promotion_allowed") is not True
    ):
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
