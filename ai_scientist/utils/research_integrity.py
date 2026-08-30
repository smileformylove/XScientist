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
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.principal_identity import canonical_principal
from ai_scientist.utils.pipeline_contracts import (
    save_contract_artifact,
    update_pipeline_artifact,
)

SCHEMA_VERSION = 1
ADAPTIVE_STATE_FREEZE_SCHEMA = "xscientist.adaptive-state-freeze.v1"
PLACEHOLDER_VALUES = {
    "dataset_to_be_selected",
    "primary_task_metric",
    "strong_existing_baseline",
    "unspecified",
    "unknown",
}
SUPPORTED_CORRECTIONS = {"bonferroni", "holm", "benjamini-hochberg"}
EVIDENCE_PORTFOLIO_ROLES = frozenset({"primary", "ablation", "robustness"})

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
        if value is not None and (not _is_number(value) or float(value) <= 0):
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
        "estimand_ok": estimand_ok
        and not any(error.startswith("estimand_") for error in errors),
        "missing_data_policy_ok": missing_policy_ok
        and not any(error.startswith("missing_data_") for error in errors),
        "null_result_policy_ok": null_policy_ok
        and not any(error.startswith("null_result_") for error in errors),
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
        improves = (
            delta_value > 0 if direction != "lower_is_better" else delta_value < 0
        )
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
        isinstance(equivalence_test, dict) and equivalence_test.get("passed") is True
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


def _adaptive_state_freeze_core(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable scientific state captured at phase transition.

    The wall-clock timestamp is provenance, not identity. Everything that can
    change what the confirmatory experiment means is content-addressed below.
    """

    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"frozen_at", "state_hash"}
    }


def _adaptive_state_scientific_hashes(
    preregistration: dict[str, Any],
) -> dict[str, str]:
    hypotheses = _object_or_empty(preregistration.get("hypotheses"))
    outcomes = [
        item
        for item in _list_or_empty(preregistration.get("outcomes"))
        if isinstance(item, dict)
    ]
    analysis_plan = _object_or_empty(preregistration.get("analysis_plan"))
    data_policy = _object_or_empty(preregistration.get("data_policy"))
    controls = _object_or_empty(preregistration.get("controls"))
    return {
        "hypothesis_hash": _canonical_hash(hypotheses),
        "method_hash": _canonical_hash(
            {
                "idea_id": preregistration.get("idea_id"),
                "plan_id": preregistration.get("plan_id"),
                "research_question": preregistration.get("research_question"),
                "hypotheses": hypotheses,
                "outcomes": outcomes,
                "controls": controls,
            }
        ),
        "protocol_hash": _canonical_hash(
            {
                "outcomes": outcomes,
                "analysis_plan": analysis_plan,
                "data_policy": data_policy,
                "controls": controls,
            }
        ),
    }


def _adaptive_research_state_hash(
    *,
    code_state_hash: str,
    memory_state_hash: str,
    evaluator_spec_hash: str,
) -> str:
    """Bind the independently addressable execution components as one state."""

    return _canonical_hash(
        {
            "kind": "confirmatory_research_state",
            "code_state_hash": code_state_hash,
            "memory_state_hash": memory_state_hash,
            "evaluator_spec_hash": evaluator_spec_hash,
        }
    )


def derive_adaptive_state_hashes(
    preregistration: dict[str, Any],
    *,
    research_vcs_head: str,
) -> dict[str, str]:
    """Derive every host-owned adaptive component from one inspectable state.

    A caller-provided digest is not an attestation: an agent can invent a
    syntactically valid ``sha256:...`` value without committing the bytes that
    produced it. These identities are deterministic functions of the locked
    scientific contract and the Research VCS commit that owns the code and
    research memory. The publication gate separately proves that the commit
    exists and is bound to a valid Research VCS checkpoint.
    """

    if not isinstance(preregistration, dict):
        raise ResearchIntegrityError("preregistration payload must be an object")
    frozen_head = str(research_vcs_head or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40,64}", frozen_head):
        raise ResearchIntegrityError(
            "adaptive state freeze requires a 40-64 character Research VCS head"
        )

    scientific = _adaptive_state_scientific_hashes(preregistration)
    outcomes = [
        deepcopy(item)
        for item in _list_or_empty(preregistration.get("outcomes"))
        if isinstance(item, dict)
    ]
    analysis_plan = deepcopy(_object_or_empty(preregistration.get("analysis_plan")))
    data_policy = deepcopy(_object_or_empty(preregistration.get("data_policy")))
    controls = deepcopy(_object_or_empty(preregistration.get("controls")))

    code_state_hash = _canonical_hash(
        {
            "kind": "research_vcs_code_state",
            "research_vcs_head": frozen_head,
        }
    )
    memory_state_hash = _canonical_hash(
        {
            "kind": "research_vcs_memory_state",
            "research_vcs_head": frozen_head,
            "idea_id": preregistration.get("idea_id"),
            "plan_id": preregistration.get("plan_id"),
            "hypothesis_hash": scientific["hypothesis_hash"],
            "method_hash": scientific["method_hash"],
        }
    )
    evaluator_spec_hash = _canonical_hash(
        {
            "kind": "confirmatory_evaluator_spec",
            "research_vcs_head": frozen_head,
            "protocol_hash": scientific["protocol_hash"],
            "outcomes": outcomes,
            "analysis_plan": analysis_plan,
            "data_policy": data_policy,
            "controls": controls,
        }
    )
    return {
        "code_state_hash": code_state_hash,
        "memory_state_hash": memory_state_hash,
        "evaluator_spec_hash": evaluator_spec_hash,
        "research_state_hash": _adaptive_research_state_hash(
            code_state_hash=code_state_hash,
            memory_state_hash=memory_state_hash,
            evaluator_spec_hash=evaluator_spec_hash,
        ),
    }


def _find_upward(
    base_folder: str | Path,
    predicate,
) -> Path | None:
    """Return the nearest ancestor accepted by ``predicate``."""

    start = Path(base_folder).expanduser().resolve()
    if not start.is_dir():
        start = start.parent
    for candidate in (start, *start.parents):
        try:
            if predicate(candidate):
                return candidate
        except OSError:
            continue
    return None


def attest_adaptive_state_freeze(
    base_folder: str | Path,
    payload: Any,
    *,
    preregistration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify an adaptive freeze against the host-owned Research VCS repo.

    This routine is deliberately exception-safe because it runs at a
    publication boundary: an unreadable repository or malformed commit must
    become a blocker, never crash the quality workflow or be treated as a
    successful attestation.
    """

    errors: list[str] = []
    internal = validate_adaptive_state_freeze(
        payload,
        preregistration=preregistration,
    )
    if internal.get("ok") is not True:
        errors.extend(str(item) for item in internal.get("errors") or [])
    if not isinstance(payload, dict) or not isinstance(preregistration, dict):
        return {
            "ok": False,
            "status": "blocked",
            "errors": sorted(set(errors or ["research_vcs_freeze_missing"])),
            "repository_root": None,
        }

    repository_root = _find_upward(
        base_folder,
        lambda candidate: (candidate / ".git").exists()
        and (candidate / ".xscientist").is_dir(),
    )
    if repository_root is None:
        errors.append("research_vcs_repository_not_found")
        return {
            "ok": False,
            "status": "blocked",
            "errors": sorted(set(errors)),
            "repository_root": None,
        }

    frozen_head = str(payload.get("research_vcs_head") or "").strip().lower()
    try:
        # Import lazily so these contracts remain usable without importing the
        # public CLI implementation during module initialization.
        from xscientist.research_git import (
            list_research_objects,
            load_research_object,
            research_blame,
            show_checkpoint,
        )

        shown = show_checkpoint(repository_root, frozen_head)
        resolved = str(shown.get("commit") or "").strip().lower()
        if resolved != frozen_head:
            errors.append("research_vcs_head_resolution_mismatch")
        if shown.get("checkpoint_hash_valid") is not True:
            errors.append("research_vcs_checkpoint_hash_invalid")
        checkpoint = shown.get("checkpoint")
        if not isinstance(checkpoint, dict) or not _is_content_hash(
            checkpoint.get("content_hash")
        ):
            errors.append("research_vcs_checkpoint_binding_invalid")

        registrations = list_research_objects(
            repository_root,
            kind="preregistration",
            state="locked",
            max_objects=4096,
            max_bytes=64 * 1024 * 1024,
        )
        matching_registrations = [
            item
            for item in registrations
            if isinstance(item, dict) and item.get("payload") == preregistration
        ]
        if len(matching_registrations) != 1:
            errors.append("research_vcs_preregistration_object_missing_or_ambiguous")
        else:
            registration_object = matching_registrations[0]
            registration_id = str(registration_object.get("object_id") or "")
            registration_relations = [
                item
                for item in registration_object.get("relations") or []
                if isinstance(item, dict)
                and item.get("type") == "depends_on"
                and str(item.get("target") or "").startswith("rso-")
            ]
            if len(registration_relations) != 1:
                errors.append("research_vcs_preregistration_plan_binding_invalid")
            else:
                plan_id = str(registration_relations[0].get("target") or "")
                plan_object = load_research_object(repository_root, plan_id)
                plan_payload = plan_object.get("payload")
                if (
                    plan_object.get("kind") != "research_plan"
                    or not isinstance(plan_payload, dict)
                    or plan_payload.get("plan_id") != preregistration.get("plan_id")
                ):
                    errors.append("research_vcs_preregistration_plan_binding_invalid")
                else:
                    registration_origin = research_blame(
                        repository_root,
                        registration_id,
                        commit="HEAD",
                    )
                    plan_origin = research_blame(
                        repository_root,
                        plan_id,
                        commit="HEAD",
                    )
                    origin_commit = str(
                        (registration_origin.get("origin") or {}).get("commit") or ""
                    ).lower()
                    plan_origin_commit = str(
                        (plan_origin.get("origin") or {}).get("commit") or ""
                    ).lower()
                    origin_shown = show_checkpoint(repository_root, origin_commit)
                    origin_checkpoint = origin_shown.get("checkpoint")
                    expected_paths = {
                        str(
                            (registration_origin.get("object") or {}).get("path") or ""
                        ),
                        str((plan_origin.get("object") or {}).get("path") or ""),
                    }
                    if (
                        origin_commit != plan_origin_commit
                        or origin_shown.get("checkpoint_hash_valid") is not True
                        or not isinstance(origin_checkpoint, dict)
                    ):
                        errors.append("research_vcs_preregistration_origin_invalid")
                    else:
                        if (
                            str(origin_checkpoint.get("parent_commit") or "").lower()
                            != frozen_head
                        ):
                            errors.append(
                                "research_vcs_preregistration_parent_mismatch"
                            )
                        if (
                            not all(expected_paths)
                            or set(origin_checkpoint.get("changed_paths") or [])
                            != expected_paths
                        ):
                            errors.append(
                                "research_vcs_preregistration_checkpoint_paths_invalid"
                            )
    except Exception:
        errors.append("research_vcs_head_not_resolvable")

    try:
        expected = derive_adaptive_state_hashes(
            preregistration,
            research_vcs_head=frozen_head,
        )
    except (ResearchIntegrityError, TypeError, ValueError):
        expected = {}
        errors.append("research_vcs_component_derivation_failed")
    for field, expected_hash in expected.items():
        if payload.get(field) != expected_hash:
            errors.append(f"research_vcs_{field}_mismatch")

    return {
        "ok": not errors,
        "status": "ready" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "repository_root": str(repository_root),
        "research_vcs_head": frozen_head if not errors else None,
    }


def _hash_file_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_empirical_data_manifest(base_folder: str | Path) -> dict[str, Any]:
    """Attest the nearest project data contract and immutable snapshot bytes."""

    errors: list[str] = []
    project_root = _find_upward(
        base_folder,
        lambda candidate: (candidate / "00_config" / "data_manifest.json").is_file(),
    )
    if project_root is None:
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["data_manifest_not_found"],
            "mode": None,
            "project_root": None,
        }

    manifest_path = project_root / "00_config" / "data_manifest.json"
    try:
        if manifest_path.is_symlink():
            raise ValueError("manifest_symlink_forbidden")
        if manifest_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("manifest_too_large")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["data_manifest_unreadable"],
            "mode": None,
            "project_root": str(project_root),
        }
    if not isinstance(manifest, dict):
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["data_manifest_not_object"],
            "mode": None,
            "project_root": str(project_root),
        }

    mode = str(manifest.get("mode") or "").strip()
    required_fields = {
        "schema_version",
        "mode",
        "ready",
        "source_path_disclosed",
        "snapshot_id",
        "file_count",
        "total_bytes",
        "files",
        "scientific_boundary",
        "manifest_hash",
    }
    if not required_fields.issubset(manifest):
        errors.append("data_manifest_required_fields_missing")
    if manifest.get("schema_version") != "xscientist.data-contract.v1":
        errors.append("data_manifest_schema_invalid")
    manifest_core = {
        key: deepcopy(value)
        for key, value in manifest.items()
        if key != "manifest_hash"
    }
    try:
        expected_manifest_hash = canonical_content_hash(manifest_core)
    except (TypeError, ValueError):
        expected_manifest_hash = ""
    if (
        not _is_content_hash(manifest.get("manifest_hash"))
        or manifest.get("manifest_hash") != expected_manifest_hash
    ):
        errors.append("data_manifest_hash_mismatch")
    if manifest.get("ready") is not True:
        errors.append("data_manifest_not_ready")
    if manifest.get("source_path_disclosed") is not False:
        errors.append("data_manifest_source_disclosure_invalid")
    if not str(manifest.get("scientific_boundary") or "").strip():
        errors.append("data_manifest_scientific_boundary_missing")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        errors.append("data_manifest_files_invalid")
        raw_files = []
    files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in raw_files:
        if not isinstance(item, dict):
            errors.append("data_manifest_file_entry_invalid")
            continue
        relative_text = str(item.get("path") or "")
        relative = PurePosixPath(relative_text)
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not relative_text
            or relative.is_absolute()
            or "\\" in relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            errors.append("data_manifest_file_path_invalid")
            continue
        if relative_text in seen_paths:
            errors.append("data_manifest_file_path_duplicate")
            continue
        seen_paths.add(relative_text)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append("data_manifest_file_size_invalid")
            continue
        if not _is_content_hash(digest):
            errors.append("data_manifest_file_hash_invalid")
            continue
        files.append(
            {
                "path": relative_text,
                "size_bytes": size,
                "sha256": str(digest),
            }
        )
    if files != sorted(files, key=lambda item: item["path"]):
        errors.append("data_manifest_files_not_canonical")
    if manifest.get("file_count") != len(files):
        errors.append("data_manifest_file_count_mismatch")
    if manifest.get("total_bytes") != sum(item["size_bytes"] for item in files):
        errors.append("data_manifest_total_bytes_mismatch")

    if mode == "synthetic_explicit":
        if raw_files or manifest.get("snapshot_id") is not None:
            errors.append("synthetic_data_contract_malformed")
        errors.append("synthetic_data_not_empirical")
    elif mode != "content_addressed_snapshot_read_only":
        errors.append("empirical_data_mode_invalid")
    else:
        snapshot_id = str(manifest.get("snapshot_id") or "")
        try:
            expected_snapshot_id = canonical_content_hash({"files": raw_files})
        except (TypeError, ValueError):
            expected_snapshot_id = ""
        snapshot_id_valid = _is_content_hash(snapshot_id)
        if not snapshot_id_valid or snapshot_id != expected_snapshot_id:
            errors.append("data_snapshot_id_mismatch")
        if not files:
            errors.append("data_snapshot_files_empty")
        snapshot = (
            project_root
            / ".ara-store"
            / "datasets"
            / snapshot_id.removeprefix("sha256:")
            if snapshot_id_valid
            else None
        )
        if snapshot is None or snapshot.is_symlink() or not snapshot.is_dir():
            errors.append("data_snapshot_not_found")
        else:
            actual_paths: set[str] = set()
            try:
                for candidate in snapshot.rglob("*"):
                    if candidate.is_symlink():
                        errors.append("data_snapshot_symlink_forbidden")
                        continue
                    if candidate.is_dir():
                        if candidate.stat().st_mode & 0o222:
                            errors.append("data_snapshot_not_read_only")
                        continue
                    if candidate.is_file():
                        actual_paths.add(candidate.relative_to(snapshot).as_posix())
                if snapshot.stat().st_mode & 0o222:
                    errors.append("data_snapshot_not_read_only")
            except OSError:
                errors.append("data_snapshot_scan_failed")
            expected_paths = {item["path"] for item in files}
            if actual_paths != expected_paths:
                errors.append("data_snapshot_file_set_mismatch")
            for item in files:
                candidate = snapshot.joinpath(*PurePosixPath(item["path"]).parts)
                try:
                    if (
                        not candidate.is_file()
                        or candidate.stat().st_size != item["size_bytes"]
                        or _hash_file_bytes(candidate) != item["sha256"]
                    ):
                        errors.append("data_snapshot_file_hash_mismatch")
                except OSError:
                    errors.append("data_snapshot_file_unreadable")

    return {
        "ok": not errors,
        "status": "ready" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "mode": mode or None,
        "project_root": str(project_root),
        "manifest_hash": manifest.get("manifest_hash") if not errors else None,
        "snapshot_id": manifest.get("snapshot_id") if not errors else None,
    }


def build_adaptive_state_freeze(
    preregistration: dict[str, Any],
    *,
    research_vcs_head: str,
    code_state_hash: str,
    memory_state_hash: str,
    evaluator_spec_hash: str,
    research_state_hash: str,
) -> dict[str, Any]:
    """Freeze adaptive exploration before any confirmatory observation.

    Every component identity is derived from the locked scientific contract
    and Research VCS head.  The explicit parameters remain part of the public
    contract so older callers fail loudly when they submit invented digests.
    """

    supplied = {
        "code_state_hash": code_state_hash,
        "memory_state_hash": memory_state_hash,
        "evaluator_spec_hash": evaluator_spec_hash,
        "research_state_hash": research_state_hash,
    }
    frozen_head = str(research_vcs_head or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40,64}", frozen_head):
        raise ResearchIntegrityError(
            "adaptive state freeze requires a 40-64 character Research VCS head"
        )
    invalid = [name for name, value in supplied.items() if not _is_content_hash(value)]
    if invalid:
        raise ResearchIntegrityError(
            "adaptive state freeze requires content hashes for: "
            + ", ".join(sorted(invalid))
        )
    expected_research_state_hash = _adaptive_research_state_hash(
        code_state_hash=code_state_hash,
        memory_state_hash=memory_state_hash,
        evaluator_spec_hash=evaluator_spec_hash,
    )
    if research_state_hash != expected_research_state_hash:
        raise ResearchIntegrityError(
            "research_state_hash must bind code_state_hash, memory_state_hash, "
            "and evaluator_spec_hash"
        )
    derived = derive_adaptive_state_hashes(
        preregistration,
        research_vcs_head=frozen_head,
    )
    mismatched = [
        field for field, expected in derived.items() if supplied[field] != expected
    ]
    if mismatched:
        raise ResearchIntegrityError(
            "adaptive state component hashes must be derived from the locked "
            "preregistration and Research VCS head: " + ", ".join(sorted(mismatched))
        )

    freeze = {
        "schema": ADAPTIVE_STATE_FREEZE_SCHEMA,
        "status": "frozen",
        "phase_boundary": "adaptive_exploration_to_confirmatory",
        "frozen_at": _now_iso(),
        "frozen_before_confirmatory": True,
        "post_freeze_adaptation_allowed": False,
        "research_vcs_head": frozen_head,
        **_adaptive_state_scientific_hashes(preregistration),
        **supplied,
    }
    freeze["state_hash"] = _canonical_hash(_adaptive_state_freeze_core(freeze))
    return freeze


def validate_adaptive_state_freeze(
    payload: Any,
    *,
    preregistration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a phase-boundary receipt without trusting its status text."""

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "status": "blocked",
            "errors": ["adaptive_state_freeze_not_object"],
        }
    errors: list[str] = []
    if payload.get("schema") != ADAPTIVE_STATE_FREEZE_SCHEMA:
        errors.append("adaptive_state_freeze_schema_invalid")
    if payload.get("status") != "frozen":
        errors.append("adaptive_state_not_frozen")
    if payload.get("phase_boundary") != "adaptive_exploration_to_confirmatory":
        errors.append("adaptive_state_phase_boundary_invalid")
    if payload.get("frozen_before_confirmatory") is not True:
        errors.append("adaptive_state_not_frozen_before_confirmation")
    if payload.get("post_freeze_adaptation_allowed") is not False:
        errors.append("post_freeze_adaptation_not_blocked")
    if not re.fullmatch(
        r"[0-9a-f]{40,64}",
        str(payload.get("research_vcs_head") or "").strip().lower(),
    ):
        errors.append("adaptive_state_research_vcs_head_invalid")
    frozen_at = str(payload.get("frozen_at") or "").strip()
    try:
        parsed = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            errors.append("adaptive_state_frozen_at_timezone_missing")
    except ValueError:
        errors.append("adaptive_state_frozen_at_invalid")
    for field in (
        "hypothesis_hash",
        "method_hash",
        "protocol_hash",
        "code_state_hash",
        "memory_state_hash",
        "evaluator_spec_hash",
        "research_state_hash",
        "state_hash",
    ):
        if not _is_content_hash(payload.get(field)):
            errors.append(f"adaptive_state_{field}_invalid")
    expected_research_state_hash = _adaptive_research_state_hash(
        code_state_hash=str(payload.get("code_state_hash") or ""),
        memory_state_hash=str(payload.get("memory_state_hash") or ""),
        evaluator_spec_hash=str(payload.get("evaluator_spec_hash") or ""),
    )
    if payload.get("research_state_hash") != expected_research_state_hash:
        errors.append("adaptive_state_research_state_components_mismatch")
    expected_hash = _canonical_hash(_adaptive_state_freeze_core(payload))
    if payload.get("state_hash") != expected_hash:
        errors.append("adaptive_state_freeze_hash_mismatch")
    if isinstance(preregistration, dict):
        expected_scientific_hashes = _adaptive_state_scientific_hashes(preregistration)
        for field, expected in expected_scientific_hashes.items():
            if payload.get(field) != expected:
                errors.append(f"adaptive_state_{field}_does_not_match_registration")
        try:
            expected_components = derive_adaptive_state_hashes(
                preregistration,
                research_vcs_head=str(payload.get("research_vcs_head") or ""),
            )
        except (ResearchIntegrityError, TypeError, ValueError):
            errors.append("adaptive_state_component_derivation_failed")
        else:
            for field, expected in expected_components.items():
                if payload.get(field) != expected:
                    errors.append(f"adaptive_state_{field}_does_not_match_registration")
    return {
        "ok": not errors,
        "status": "ready" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "state_hash": payload.get("state_hash") if not errors else None,
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
            "adaptive_state_freeze": preregistration.get("adaptive_state_freeze"),
        }
    )


def _outcome_transformation_contract(outcome: dict[str, Any]) -> dict[str, Any]:
    """Return the locked intervention/stress semantics for one outcome."""

    role = str(outcome.get("evidence_role") or "").strip().lower()
    if not role:
        return {}
    return {
        "evidence_role": role,
        "paired_control_task_id": (
            str(outcome.get("paired_control_task_id") or "").strip() or None
        ),
        "intervention_variant": (
            str(outcome.get("intervention_variant") or "").strip() or None
        ),
        "stress_condition": (
            str(outcome.get("stress_condition") or "").strip() or None
        ),
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
        evidence_role = str(task.get("evidence_role") or "").strip().lower()
        outcome = {
            "task_id": str(task.get("task_id") or f"task_{index}"),
            "dataset": str(task.get("dataset") or "").strip(),
            "metric": metric,
            "baseline": str(task.get("baseline") or "").strip(),
            "direction": _direction_for_metric(metric),
            "primary": index == 0,
            "evidence_role": evidence_role or None,
            "paired_control_task_id": (
                str(task.get("paired_control_task_id") or "").strip() or None
            ),
            "intervention_variant": (
                str(task.get("intervention_variant") or "").strip() or None
            ),
            "stress_condition": (
                str(task.get("stress_condition") or "").strip() or None
            ),
            "minimum_effect": None,
            "null_effect_threshold": 0.0,
            "meaningful_effect_margin": None,
        }
        transformation_contract = _outcome_transformation_contract(outcome)
        outcome["transformation_contract"] = transformation_contract or None
        outcome["transformation_contract_hash"] = (
            _canonical_hash(transformation_contract)
            if transformation_contract
            else None
        )
        outcomes.append(outcome)

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
        "evidence_portfolio": deepcopy(
            research_plan.get("evidence_portfolio")
            if isinstance(research_plan.get("evidence_portfolio"), dict)
            else {"required": False, "required_roles": []}
        ),
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
    if raw_external_validity is not None and not isinstance(
        raw_external_validity, dict
    ):
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
        if not isinstance(
            external_validity.get("evidence_artifacts"), list
        ) or not external_validity.get("evidence_artifacts"):
            errors.append("external_validity_evidence_missing")

    raw_deviations = payload.get("deviations")
    if raw_deviations is not None and not isinstance(raw_deviations, list):
        errors.append("deviations_not_list")
    elif isinstance(raw_deviations, list) and any(
        not isinstance(item, dict) for item in raw_deviations
    ):
        errors.append("deviation_entry_not_object")

    campaign = payload.get("confirmatory_campaign")
    if campaign is not None:
        if not isinstance(campaign, dict):
            errors.append("confirmatory_campaign_not_object")
        else:
            queue_contract = campaign.get("queue_contract")
            if not isinstance(queue_contract, dict):
                errors.append("confirmatory_queue_contract_missing")
            else:
                contract_core = {
                    key: value
                    for key, value in queue_contract.items()
                    if key != "queue_contract_hash"
                }
                expected_contract_hash = _canonical_hash(contract_core)
                if (
                    queue_contract.get("schema")
                    != "xscientist.confirmatory-queue-contract.v1"
                    or queue_contract.get("queue_contract_hash")
                    != expected_contract_hash
                    or campaign.get("queue_contract_hash") != expected_contract_hash
                ):
                    errors.append("confirmatory_queue_contract_hash_invalid")
                contract_tasks = queue_contract.get("tasks")
                if not isinstance(contract_tasks, list) or any(
                    not isinstance(item, dict) for item in contract_tasks
                ):
                    errors.append("confirmatory_queue_contract_tasks_invalid")
                else:
                    contract_task_ids: list[str] = []
                    for contract_task in contract_tasks:
                        task_id = str(contract_task.get("task_id") or "").strip()
                        contract_task_ids.append(task_id)
                        task_core = {
                            key: value
                            for key, value in contract_task.items()
                            if key != "task_contract_hash"
                        }
                        command_template = contract_task.get("record_command_template")
                        if (
                            not task_id
                            or contract_task.get("task_contract_hash")
                            != _canonical_hash(task_core)
                            or not isinstance(command_template, list)
                            or "{TERMINAL_STATUS}" not in command_template
                            or "completed" in command_template
                        ):
                            errors.append("confirmatory_queue_task_contract_invalid")
                    outcome_ids = {
                        str(item.get("task_id") or "").strip() for item in outcomes
                    }
                    if (
                        len(contract_task_ids) != len(set(contract_task_ids))
                        or set(contract_task_ids) != outcome_ids
                    ):
                        errors.append("confirmatory_queue_task_coverage_invalid")

    freeze_report: dict[str, Any] | None = None
    if "adaptive_state_freeze" in payload:
        freeze_report = validate_adaptive_state_freeze(
            payload.get("adaptive_state_freeze"),
            preregistration=payload,
        )
        errors.extend(freeze_report.get("errors") or [])

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

    raw_portfolio = payload.get("evidence_portfolio")
    if raw_portfolio is not None and not isinstance(raw_portfolio, dict):
        errors.append("evidence_portfolio_not_object")
        portfolio: dict[str, Any] = {}
    else:
        portfolio = raw_portfolio if isinstance(raw_portfolio, dict) else {}
    outcome_by_id = {
        str(item.get("task_id") or "").strip(): item
        for item in outcomes
        if str(item.get("task_id") or "").strip()
    }
    roles = {
        str(item.get("evidence_role") or "").strip().lower()
        for item in outcomes
        if str(item.get("evidence_role") or "").strip()
    }
    if roles - EVIDENCE_PORTFOLIO_ROLES:
        errors.append("evidence_role_invalid")
    if roles:
        role_primary = [
            item
            for item in outcomes
            if str(item.get("evidence_role") or "").strip().lower() == "primary"
        ]
        if len(role_primary) != 1:
            errors.append("evidence_primary_role_count_invalid")
        elif role_primary[0].get("primary") is not True:
            errors.append("evidence_primary_role_mismatch")
        if any(
            item.get("primary") is True
            and str(item.get("evidence_role") or "").strip().lower() != "primary"
            for item in outcomes
        ):
            errors.append("primary_flag_role_mismatch")
    if portfolio.get("required") is True:
        raw_required_roles = portfolio.get("required_roles")
        required_roles = (
            {
                str(item).strip().lower()
                for item in raw_required_roles
                if str(item).strip()
            }
            if isinstance(raw_required_roles, list)
            else set()
        )
        if not required_roles or required_roles - EVIDENCE_PORTFOLIO_ROLES:
            errors.append("evidence_portfolio_required_roles_invalid")
        elif not required_roles.issubset(roles):
            errors.append("evidence_portfolio_role_missing")
    for outcome in outcomes:
        task_id = str(outcome.get("task_id") or "unknown").strip() or "unknown"
        role = str(outcome.get("evidence_role") or "").strip().lower()
        if not role:
            continue
        expected_contract = _outcome_transformation_contract(outcome)
        if outcome.get("transformation_contract") != expected_contract:
            errors.append(f"{task_id}_transformation_contract_mismatch")
        if outcome.get("transformation_contract_hash") != _canonical_hash(
            expected_contract
        ):
            errors.append(f"{task_id}_transformation_contract_hash_mismatch")
        pair_id = str(outcome.get("paired_control_task_id") or "").strip()
        intervention = str(outcome.get("intervention_variant") or "").strip()
        stress = str(outcome.get("stress_condition") or "").strip()
        if role == "primary":
            if pair_id:
                errors.append(f"{task_id}_primary_control_pair_disallowed")
            if not intervention:
                errors.append(f"{task_id}_primary_intervention_missing")
        elif role == "ablation":
            if (
                not pair_id
                or str(outcome_by_id.get(pair_id, {}).get("evidence_role") or "")
                .strip()
                .lower()
                != "primary"
            ):
                errors.append(f"{task_id}_ablation_primary_pair_invalid")
            if not intervention:
                errors.append(f"{task_id}_ablation_intervention_missing")
        elif role == "robustness":
            if (
                not pair_id
                or str(outcome_by_id.get(pair_id, {}).get("evidence_role") or "")
                .strip()
                .lower()
                != "primary"
            ):
                errors.append(f"{task_id}_robustness_primary_pair_invalid")
            if not stress:
                errors.append(f"{task_id}_robustness_stress_condition_missing")
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
        "adaptive_state_freeze": freeze_report,
    }


def lock_preregistration(
    payload: dict[str, Any],
    *,
    split_hashes: dict[str, str],
    registered_by: str,
    freeze_inputs: dict[str, str] | None = None,
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
    if freeze_inputs is not None:
        required_freeze_inputs = {
            "research_vcs_head",
            "code_state_hash",
            "memory_state_hash",
            "evaluator_spec_hash",
            "research_state_hash",
        }
        unknown = set(freeze_inputs) - required_freeze_inputs
        missing = required_freeze_inputs - set(freeze_inputs)
        if unknown or missing:
            details = []
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            if unknown:
                details.append("unknown=" + ",".join(sorted(unknown)))
            raise ResearchIntegrityError(
                "invalid adaptive state freeze inputs: " + "; ".join(details)
            )
        locked["adaptive_state_freeze"] = build_adaptive_state_freeze(
            locked,
            **freeze_inputs,
        )
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
        "data_manifest_hash": payload.get("data_manifest_hash"),
        "data_snapshot_id": payload.get("data_snapshot_id"),
        "trajectory_binding_hash": payload.get("trajectory_binding_hash"),
        "research_vcs_frozen_head": payload.get("research_vcs_frozen_head"),
        "research_vcs_lineage_head": payload.get("research_vcs_lineage_head"),
        "research_vcs_attempt_object_ids": _string_list(
            payload.get("research_vcs_attempt_object_ids")
        ),
        "research_vcs_binding_object_ids": _string_list(
            payload.get("research_vcs_binding_object_ids")
        ),
        "research_vcs_checkpoint_hashes": _string_list(
            payload.get("research_vcs_checkpoint_hashes")
        ),
        "failed_record_ids": _string_list(payload.get("failed_record_ids")),
        "research_vcs_disposition_object_ids": _string_list(
            payload.get("research_vcs_disposition_object_ids")
        ),
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
    try:
        producer_principal = canonical_principal(
            producer_id, label="reproduction producer_id"
        )
        primary_producers = {
            canonical_principal(item.get("producer_id"), label="primary producer_id")
            for item in primary_records_by_id.values()
        }
    except (TypeError, ValueError):
        return False
    if producer_principal in primary_producers:
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
            records = [
                dict(item) for item in experiment_records if isinstance(item, dict)
            ]
        except TypeError:
            # A malformed registry should produce a blocked report, not an
            # exception that prevents the CLI from showing recovery actions.
            records = []
    confirmatory = [
        item
        for item in records
        if item.get("record_type") != "attempt_disposition"
        and str(item.get("study_phase") or "").lower() == "confirmatory"
        and str(item.get("status") or "").lower() in {"completed", "verified"}
    ]
    reproductions = [
        item
        for item in records
        if item.get("record_type") != "attempt_disposition"
        and item.get("independent_reproduction") is True
    ]
    # Publication closure covers every confirmatory or reproduction row,
    # including unsuccessful attempts.  A failed reproduction is scientific
    # history, not an optional attachment that can disappear from completeness
    # or authority calculations.
    publication_attempts = [
        item
        for item in records
        if item.get("record_type") != "attempt_disposition"
        and (
            str(item.get("study_phase") or "").lower() == "confirmatory"
            or item.get("independent_reproduction") is True
        )
    ]
    incomplete_publication_attempts = [
        item
        for item in publication_attempts
        if str(item.get("status") or "").lower() not in {"completed", "verified"}
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
    declared_data_manifest_hash = str(
        data_policy.get("data_manifest_hash") or ""
    ).strip()
    declared_data_snapshot_id = str(data_policy.get("data_snapshot_id") or "").strip()
    if verification_root is not None:
        try:
            data_attestation = validate_empirical_data_manifest(verification_root)
        except Exception:
            data_attestation = {
                "ok": False,
                "errors": ["empirical_data_attestation_failed_closed"],
            }
    else:
        data_attestation = {
            "ok": False,
            "errors": ["verification_root_missing"],
        }
    host_data_manifest_hash = str(data_attestation.get("manifest_hash") or "").strip()
    host_data_snapshot_id = str(data_attestation.get("snapshot_id") or "").strip()
    data_binding_required = bool(
        declared_data_manifest_hash
        or declared_data_snapshot_id
        or data_attestation.get("ok") is True
    )
    trajectory_required = isinstance(preregistration.get("adaptive_state_freeze"), dict)
    minimum_seeds = _positive_int(analysis.get("minimum_independent_seeds"), 1)
    minimum_reproductions = _positive_int(
        analysis.get("minimum_independent_reproductions"), 1
    )
    producer_ids = {
        str(item.get("producer_id") or "").strip()
        for item in publication_attempts
        if str(item.get("producer_id") or "").strip()
    }
    confirmatory_record_ids = [
        str(item.get("record_id") or "").strip() for item in confirmatory
    ]
    relevant_record_ids = [
        str(item.get("record_id") or "").strip() for item in publication_attempts
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
        str(item.get("producer_id") or "").strip() for item in publication_attempts
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
        dispersion = any(
            _is_number(summary.get(key)) and float(summary.get(key)) >= 0
            for key in (
                "standard_deviation",
                "metric_std",
                "std",
                "variance",
                "dispersion",
            )
        )
        replicate_values = summary.get("replicate_values")
        replicate_uncertainty = (
            isinstance(replicate_values, (list, tuple))
            and len(replicate_values) >= 2
            and all(_is_number(value) for value in replicate_values)
        )
        # An effect-size point estimate describes magnitude, not uncertainty.
        # Require an interval, error/dispersion estimate, replicate distribution,
        # or an explicit valid hypothesis-test result.
        uncertainty = bool(
            interval_ok
            or _is_number(p_value)
            or _is_number(standard_error)
            or dispersion
            or replicate_uncertainty
        )
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
            str(record.get("evaluator_input_hash") or "") for record in task_records
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
            if (
                explicit_commitment is not None
                and explicit_commitment != expected_commitment
            ):
                seed_commitments_valid = False
        # A copied evaluator input cannot become an independent seed run by
        # changing only the JSON seed label.
        seed_independence = (
            seed_independence
            and valid_input_hashes
            and len(set(input_hashes)) == len(input_hashes)
            and seed_commitments_valid
        )
    try:
        verifier_principal = canonical_principal(
            normalized_verifier_id, label="verifier_id"
        )
        producer_principals = {
            canonical_principal(producer_id, label="producer_id")
            for producer_id in producer_ids
        }
        verifier_independent = verifier_principal not in producer_principals
    except (TypeError, ValueError):
        verifier_independent = False
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
        str(item.get("record_id") or "").strip() for item in valid_reproduction_records
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
    # The legacy deviations array is intentionally outside registration_hash,
    # so a post-result edit can truthfully describe a deviation but cannot
    # prove that approval preceded unblinding.  Publication-oriented frozen
    # work therefore fails closed on any such entry until a future typed
    # pre-approval object can prove its origin checkpoint.
    deviation_ok = (
        not deviations
        if trajectory_required
        else not any(
            item.get("approved_before_unblinding") is not True
            for item in deviations
            if isinstance(item, dict)
        )
        and all(isinstance(item, dict) for item in deviations)
    )

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

    data_bound_records = [*confirmatory, *reproductions]
    data_records_bound = bool(confirmatory) and all(
        record.get("data_manifest_hash") == host_data_manifest_hash
        and record.get("data_snapshot_id") == host_data_snapshot_id
        for record in data_bound_records
    )
    data_contract_binding = (
        data_attestation.get("ok") is True
        and _is_content_hash(host_data_manifest_hash)
        and _is_content_hash(host_data_snapshot_id)
        and declared_data_manifest_hash == host_data_manifest_hash
        and declared_data_snapshot_id == host_data_snapshot_id
        and data_records_bound
    )

    if trajectory_required and verification_root is not None:
        try:
            from ai_scientist.utils.trajectory_binding import (
                attest_structured_trajectory,
            )

            trajectory_attestation = attest_structured_trajectory(
                verification_root,
                preregistration,
                records,
            )
        except Exception as exc:
            trajectory_attestation = {
                "ok": False,
                "errors": [
                    str(exc) or "structured_trajectory_attestation_failed_closed"
                ],
            }
    elif trajectory_required:
        trajectory_attestation = {
            "ok": False,
            "errors": ["structured_trajectory_verification_root_missing"],
        }
    else:
        trajectory_attestation = {
            "ok": True,
            "errors": [],
            "trajectory_hash": None,
            "disposed_attempt_record_ids": [],
            "publication_blocking_attempt_record_ids": [],
            "publication_ready": True,
        }
    publication_blocking_attempt_ids = set(
        trajectory_attestation.get("publication_blocking_attempt_record_ids") or []
    )
    confirmatory_attempts_complete = (
        trajectory_attestation.get("ok") is True
        and trajectory_attestation.get("publication_ready") is True
        and not publication_blocking_attempt_ids
        if trajectory_required
        else not incomplete_publication_attempts
    )

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
            confirmatory_attempts_complete,
            (
                "every unsuccessful confirmatory attempt has an immutable, "
                "trajectory-bound disposition"
                if trajectory_required
                else "all confirmatory attempts are completed or verified"
            ),
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
            (
                "frozen publication work has no mutable legacy deviations; "
                "post-freeze failures use typed trajectory dispositions"
                if trajectory_required
                else "all deviations approved before unblinding"
            ),
        ),
    ]
    criteria.append(
        _criterion(
            "data_contract_binding",
            data_contract_binding if data_binding_required else True,
            (
                "locked preregistration and every confirmatory/reproduction "
                "record bind the host-verified data manifest and snapshot"
            ),
            required=data_binding_required,
        )
    )
    criteria.append(
        _criterion(
            "trajectory_binding",
            trajectory_attestation.get("ok") is True
            and trajectory_attestation.get("publication_ready") is True
            and not publication_blocking_attempt_ids,
            ", ".join(trajectory_attestation.get("errors") or [])
            or (
                "publication blockers="
                + ",".join(sorted(publication_blocking_attempt_ids))
                if publication_blocking_attempt_ids
                else ""
            )
            or (
                "every confirmatory/reproduction registry row maps bijectively to "
                "one typed attempt and origin checkpoint, and every failed attempt "
                "has a publication-resolving bound outcome"
            ),
            required=trajectory_required,
        )
    )
    failures = [
        item["id"] for item in criteria if item["required"] and not item["passed"]
    ]
    status = "verified" if not failures else "blocked"
    registry_hash = None
    registry_records_match = True
    snapshot_payload = (
        evidence_snapshot if isinstance(evidence_snapshot, dict) else None
    )
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
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
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
    if (
        registry_content_criterion["required"]
        and not registry_content_criterion["passed"]
    ):
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
        and conclusion_status in {"supports_effect", "supports_no_meaningful_effect"},
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
        "data_manifest_hash": (
            host_data_manifest_hash if data_attestation.get("ok") is True else None
        ),
        "data_snapshot_id": (
            host_data_snapshot_id if data_attestation.get("ok") is True else None
        ),
        "trajectory_binding_hash": trajectory_attestation.get("trajectory_hash"),
        "research_vcs_frozen_head": trajectory_attestation.get("frozen_head"),
        "research_vcs_lineage_head": trajectory_attestation.get("lineage_head"),
        "research_vcs_attempt_object_ids": trajectory_attestation.get(
            "attempt_object_ids"
        )
        or [],
        "research_vcs_binding_object_ids": trajectory_attestation.get(
            "binding_object_ids"
        )
        or [],
        "research_vcs_checkpoint_hashes": trajectory_attestation.get(
            "checkpoint_hashes"
        )
        or [],
        "failed_record_ids": trajectory_attestation.get("failed_record_ids") or [],
        "research_vcs_disposition_object_ids": trajectory_attestation.get(
            "disposition_object_ids"
        )
        or [],
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
    "ADAPTIVE_STATE_FREEZE_SCHEMA",
    "ResearchIntegrityError",
    "assert_claim_promotion_allowed",
    "attest_adaptive_state_freeze",
    "build_adaptive_state_freeze",
    "build_preregistration",
    "build_verification_report",
    "derive_adaptive_state_hashes",
    "lock_preregistration",
    "save_preregistration",
    "save_verification_report",
    "validate_adaptive_state_freeze",
    "validate_empirical_data_manifest",
    "validate_preregistration",
]
