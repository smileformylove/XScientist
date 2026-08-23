"""Auditable research-opportunity discovery and allocation.

This module is the XScientist analogue of the *Find → Attempt → Recommend*
pattern described by FAR (arXiv:2608.16977).  It deliberately reuses the
existing Research VCS object kinds instead of changing the built-in semantic
profiles: old repositories therefore remain readable while the new
``protocol_kind`` values make the funnel explicit.

The module is domain-agnostic.  It does not call a model, claim that a problem
is novel, or infer a human score.  It records the complete candidate set,
requires an explicit outcome for every attempt, requires a provenance-disjoint
evaluator for judgments, and keeps allocation estimates separate from
validated results.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash

from .research_authority import require_independent_evaluator
from .research_commands import _ensure_direct_save_is_safe, _finish
from .research_git import ResearchGitError
from .research_vcs import ResearchRepository

FAR_DIRECTION_PROTOCOL = "xscientist.far-research-direction.v1"
FAR_POOL_PROTOCOL = "xscientist.far-opportunity-pool.v1"
FAR_ATTEMPT_PROTOCOL = "xscientist.far-opportunity-attempt.v1"
FAR_JUDGMENT_PROTOCOL = "xscientist.far-opportunity-judgment.v1"
FAR_GRADE_PROTOCOL = "xscientist.far-opportunity-grade.v1"
FAR_ALLOCATION_PROTOCOL = "xscientist.far-allocation-plan.v1"
FAR_SUMMARY_PROTOCOL = "xscientist.far-funnel-summary.v1"

MAX_OPPORTUNITY_TEXT = 4096
MAX_OPPORTUNITY_REFS = 64
MAX_OPPORTUNITY_REF_TEXT = 2048
MAX_OPPORTUNITY_REF_DEPTH = 2
MAX_OPPORTUNITY_POLICY_FIELDS = 32
_FORBIDDEN_REF_TOKENS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "raw_response",
    "secret",
    "token",
)

OPPORTUNITY_SOURCE_STATUSES = ("open", "solved", "invalid", "unknown")
OPPORTUNITY_OUTCOMES = ("known", "new", "fix", "none")
JUDGMENT_VERDICTS = ("pass", "fail", "known")
GRADE_LEVELS = ("known", "minor", "substantial")
ALLOCATION_OBJECTIVES = ("artifact_yield", "importance_yield", "best_artifact")
# The FAR paper's allocation derivation distinguishes the probability of an
# accepted attempt from the probability that an accepted attempt is graded as
# publishable.  Keep that distinction explicit at the API boundary.  The
# default preserves the historical XScientist behaviour (multiply the two
# caller-supplied factors), but records the assumption in every allocation
# object instead of silently treating a missing factor as calibrated data.
PROBABILITY_SEMANTICS = (
    "conditional_artifact_given_success",
    "joint_artifact_probability",
)


def _text(value: Any, *, label: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ResearchGitError(f"{label} is required")
    if len(result) > MAX_OPPORTUNITY_TEXT:
        raise ResearchGitError(
            f"{label} exceeds the {MAX_OPPORTUNITY_TEXT}-character audit limit"
        )
    return result


def _optional_text(value: Any, *, label: str) -> str:
    if value in (None, ""):
        return ""
    return _text(value, label=label)


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchGitError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ResearchGitError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise ResearchGitError(f"{label} must be >= {minimum}")
    return result


def _probability(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ResearchGitError(f"{label} must be between 0 and 1")
    return round(result, 6)


def _ordinal(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
        raise ResearchGitError(f"{label} must be an integer from 0 to 4")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResearchGitError(f"{label} must be a positive integer")
    return value


def _bounded_json_value(
    value: Any,
    *,
    label: str,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    """Validate a small JSON-compatible metadata tree without leaking secrets.

    ``candidate_policy`` and independence receipts are metadata, not free-form
    payload channels.  Keeping this validator separate from ``_refs`` lets us
    preserve receipt hashes byte-for-byte while still rejecting recursive,
    oversized, non-finite, or credential-bearing values.
    """

    if depth > MAX_OPPORTUNITY_REF_DEPTH:
        raise ResearchGitError(f"{label} is nested too deeply")
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and len(value) > MAX_OPPORTUNITY_REF_TEXT:
            raise ResearchGitError(f"{label} is too long")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 10**18:
            raise ResearchGitError(f"{label} contains an oversized integer")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchGitError(f"{label} must contain only finite numbers")
        return value
    identity = id(value)
    if identity in seen:
        raise ResearchGitError(f"{label} contains a recursive reference")
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            if len(value) > MAX_OPPORTUNITY_POLICY_FIELDS:
                raise ResearchGitError(
                    f"{label} has too many fields (max {MAX_OPPORTUNITY_POLICY_FIELDS})"
                )
            result: dict[str, Any] = {}
            for key in sorted(value, key=str):
                normalized_key = str(key)
                lowered = normalized_key.lower()
                if any(token in lowered for token in _FORBIDDEN_REF_TOKENS):
                    raise ResearchGitError(
                        f"{label} contains a credential or raw-response field"
                    )
                if len(normalized_key) > 128:
                    raise ResearchGitError(f"{label} has an oversized field name")
                result[normalized_key] = _bounded_json_value(
                    value[key],
                    label=f"{label}.{normalized_key}",
                    depth=depth + 1,
                    seen=seen,
                )
            return result
        if isinstance(value, (list, tuple)):
            if len(value) > MAX_OPPORTUNITY_REFS:
                raise ResearchGitError(f"{label} has too many values")
            return [
                _bounded_json_value(
                    item,
                    label=f"{label}[{index}]",
                    depth=depth + 1,
                    seen=seen,
                )
                for index, item in enumerate(value)
            ]
    finally:
        seen.discard(identity)
    raise ResearchGitError(f"{label} must be JSON-compatible")


def _object_ids(value: Any, *, label: str) -> list[str]:
    """Normalize canonical local Research VCS object IDs for a payload."""

    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ResearchGitError(f"{label} must be an array")
    if len(value) > MAX_OPPORTUNITY_REFS:
        raise ResearchGitError(
            f"{label} exceeds the {MAX_OPPORTUNITY_REFS}-object limit"
        )
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or re.fullmatch(r"rso-[0-9a-f]{16}", item) is None:
            raise ResearchGitError(
                f"{label}[{index}] must be a canonical Research VCS object id"
            )
        if item in result:
            continue
        result.append(item)
    return sorted(result)


def _timestamp(value: Any, *, label: str) -> str:
    if value in (None, ""):
        return _now_iso()
    if not isinstance(value, str):
        raise ResearchGitError(f"{label} must be an ISO-8601 string")
    normalized = value.strip()
    if len(normalized) > 128:
        raise ResearchGitError(f"{label} is too long")
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchGitError(f"{label} must be an ISO-8601 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResearchGitError(f"{label} must include an explicit timezone")
    return normalized


def _resolve_local_object_ids(
    repository: ResearchRepository,
    values: Any,
    *,
    label: str,
) -> list[str]:
    """Resolve full IDs, prefixes, and ``@latest`` selectors before hashing."""

    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise ResearchGitError(f"{label} must be an array")
    if len(values) > MAX_OPPORTUNITY_REFS:
        raise ResearchGitError(
            f"{label} exceeds the {MAX_OPPORTUNITY_REFS}-object limit"
        )
    resolved: set[str] = set()
    for index, selector in enumerate(values):
        if not isinstance(selector, str) or not selector.strip():
            raise ResearchGitError(f"{label}[{index}] must be an object selector")
        item = repository.get(selector.strip())
        resolved.add(str(item["object_id"]))
    return sorted(resolved)


def _optional_gate_override(
    *,
    allow_stage_override: bool,
    override_reason: str,
    label: str,
) -> dict[str, Any] | None:
    """Normalize an explicit FAR-stage override without hiding it.

    The normal path is deliberately strict.  A caller may still preserve an
    unusual audit record (for example, a retrospective review of a known
    result), but only when it supplies a durable, human-readable reason that
    is copied into the content hash.
    """

    if not isinstance(allow_stage_override, bool):
        raise ResearchGitError("allow_stage_override must be a boolean")
    reason = " ".join(str(override_reason or "").split())
    if not allow_stage_override:
        if reason:
            raise ResearchGitError(
                f"{label} override_reason requires allow_stage_override=True"
            )
        return None
    if not reason:
        raise ResearchGitError(
            f"{label} stage override requires a non-empty override_reason"
        )
    if len(reason) > MAX_OPPORTUNITY_TEXT:
        raise ResearchGitError(
            f"{label} override_reason exceeds the {MAX_OPPORTUNITY_TEXT}-character audit limit"
        )
    return {"allowed": True, "reason": reason}


def _refs(value: Any, *, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ResearchGitError(f"{label} must be an array")
    if len(value) > MAX_OPPORTUNITY_REFS:
        raise ResearchGitError(
            f"{label} exceeds the {MAX_OPPORTUNITY_REFS}-reference audit limit"
        )

    def _safe_ref(item: Any, *, depth: int, item_label: str) -> Any:
        if depth > MAX_OPPORTUNITY_REF_DEPTH:
            raise ResearchGitError(f"{item_label} is nested too deeply")
        if isinstance(item, str):
            result = " ".join(item.split())
            if len(result) > MAX_OPPORTUNITY_REF_TEXT:
                raise ResearchGitError(f"{item_label} is too long")
            if not result:
                raise ResearchGitError(f"{item_label} cannot be empty")
            return result
        if isinstance(item, Mapping):
            if len(item) > 32:
                raise ResearchGitError(f"{item_label} has too many fields")
            result: dict[str, Any] = {}
            for key in sorted(item, key=str):
                normalized_key = str(key)
                lowered = normalized_key.lower()
                if any(token in lowered for token in _FORBIDDEN_REF_TOKENS):
                    raise ResearchGitError(
                        f"{item_label} contains a credential or raw-response field"
                    )
                if len(normalized_key) > 128:
                    raise ResearchGitError(f"{item_label} has an oversized field name")
                result[normalized_key] = _safe_ref(
                    item[key],
                    depth=depth + 1,
                    item_label=f"{item_label}.{normalized_key}",
                )
            if not any(
                str(result.get(key) or "").strip()
                for key in ("id", "url", "doi", "source_id")
            ):
                raise ResearchGitError(
                    f"{item_label} needs an id, url, doi, or source_id"
                )
            return result
        if isinstance(item, (list, tuple)):
            if len(item) > MAX_OPPORTUNITY_REFS:
                raise ResearchGitError(f"{item_label} has too many values")
            return [
                _safe_ref(child, depth=depth + 1, item_label=f"{item_label}[{index}]")
                for index, child in enumerate(item)
            ]
        raise ResearchGitError(f"{item_label} must be a string or object")

    rows: list[Any] = []
    for index, item in enumerate(value):
        rows.append(_safe_ref(item, depth=0, item_label=f"{label}[{index}]"))
    return rows


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_core(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**dict(core), field: canonical_content_hash(core)}


def _candidate_core(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    candidate_id = _text(raw.get("candidate_id"), label=f"candidate {index + 1} id")
    question = _text(
        raw.get("question") or raw.get("statement") or raw.get("summary"),
        label=f"candidate {candidate_id} question",
    )
    source_status = str(raw.get("source_status") or "unknown").strip().lower()
    if source_status not in OPPORTUNITY_SOURCE_STATUSES:
        raise ResearchGitError(
            f"candidate {candidate_id} source_status must be one of "
            + ", ".join(OPPORTUNITY_SOURCE_STATUSES)
        )
    source_refs = _refs(
        raw.get("source_refs"), label=f"candidate {candidate_id} source_refs"
    )
    source_object_ids = _object_ids(
        raw.get("source_object_ids"),
        label=f"candidate {candidate_id} source_object_ids",
    )
    core: dict[str, Any] = {
        "candidate_id": candidate_id,
        "question": question,
        "source_refs": source_refs,
        "source_object_ids": source_object_ids,
        "source_status": source_status,
        "source_complete": bool(source_refs or source_object_ids),
        "lineage_bound": bool(source_object_ids),
    }
    # FAR uses continuous [0, 1] difficulty/importance estimates.  Keep the
    # scale explicit instead of silently converting them to local ordinals.
    for field in ("difficulty", "importance"):
        if raw.get(field) is not None:
            core[field] = _probability(
                raw[field], label=f"candidate {candidate_id} {field}"
            )
    for field in ("difficulty_ordinal", "importance_ordinal"):
        if raw.get(field) is not None:
            core[field] = _ordinal(
                raw[field], label=f"candidate {candidate_id} {field}"
            )
    for field in (
        "expected_success_probability",
        "expected_artifact_probability",
        "expected_importance",
    ):
        if raw.get(field) is not None:
            core[field] = _probability(
                raw[field], label=f"candidate {candidate_id} {field}"
            )
    for field in ("cost", "risk"):
        if raw.get(field) is not None:
            core[field] = _finite(
                raw[field], label=f"candidate {candidate_id} {field}", minimum=0.0
            )
    probability_semantics = raw.get("expected_artifact_probability_semantics")
    if probability_semantics is not None:
        probability_semantics = str(probability_semantics).strip().lower()
        if probability_semantics not in {
            "conditional_on_success",
            "joint",
        }:
            raise ResearchGitError(
                f"candidate {candidate_id} expected_artifact_probability_semantics "
                "must be conditional_on_success or joint"
            )
        core["expected_artifact_probability_semantics"] = probability_semantics
    if raw.get("notes") not in (None, ""):
        core["notes"] = _text(raw["notes"], label=f"candidate {candidate_id} notes")
    return _hash_core(core, "candidate_hash")


def normalize_opportunity_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and deterministically normalize a complete candidate set."""

    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ResearchGitError("opportunity candidate set must be a non-empty array")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise ResearchGitError(f"candidate {index + 1} must be a JSON object")
        rows.append(_candidate_core(raw, index=index))
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ResearchGitError("opportunity candidate ids must be unique")
    return sorted(rows, key=lambda row: row["candidate_id"])


def build_research_direction(
    *,
    direction_id: str,
    statement: str,
    objective: str,
    domain: str = "",
    success_definition: str = "",
    constraints: Sequence[str] = (),
    source_refs: Sequence[Any] = (),
    candidate_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a direction-level contract without asserting a solution."""

    normalized_id = _text(direction_id, label="research direction id")
    if isinstance(constraints, (str, bytes)) or not isinstance(constraints, Sequence):
        raise ResearchGitError("research direction constraints must be an array")
    if not isinstance(candidate_policy, Mapping) and candidate_policy is not None:
        raise ResearchGitError("research direction candidate_policy must be an object")
    bounded_policy = _bounded_json_value(
        dict(candidate_policy or {}), label="research direction candidate_policy"
    )
    core: dict[str, Any] = {
        "protocol_kind": FAR_DIRECTION_PROTOCOL,
        "direction_id": normalized_id,
        "question": _text(statement, label="research direction statement"),
        "objective": _text(objective, label="research direction objective"),
        "domain": _optional_text(domain, label="research direction domain"),
        "success_definition": _optional_text(
            success_definition, label="research direction success_definition"
        ),
        "constraints": [
            _text(item, label="research direction constraint") for item in constraints
        ],
        "source_refs": _refs(source_refs, label="research direction source_refs"),
        "candidate_policy": bounded_policy,
    }
    return _hash_core(core, "goal_hash")


def build_opportunity_pool(
    *,
    direction_id: str,
    candidates: Sequence[Mapping[str, Any]],
    complete_candidate_set: bool = True,
    extraction_notes: str = "",
) -> dict[str, Any]:
    """Build the immutable Find-stage output and expose missing coverage."""

    normalized = normalize_opportunity_candidates(candidates)
    if not isinstance(complete_candidate_set, bool):
        raise ResearchGitError("complete_candidate_set must be a boolean")
    complete = complete_candidate_set
    status_counts = dict(
        sorted(Counter(row["source_status"] for row in normalized).items())
    )
    core: dict[str, Any] = {
        "protocol_kind": FAR_POOL_PROTOCOL,
        "direction_id": _text(direction_id, label="opportunity pool direction_id"),
        "candidates": normalized,
        "candidate_count": len(normalized),
        "candidate_set_complete": complete,
        "source_status_counts": status_counts,
        "source_coverage": "complete" if complete else "partial",
        "lineage_complete": all(row["lineage_bound"] for row in normalized),
        "extraction_notes": _optional_text(
            extraction_notes, label="opportunity extraction_notes"
        ),
    }
    core["candidate_set_hash"] = canonical_content_hash(normalized)
    return _hash_core(core, "pool_hash")


def _candidate_probability(candidate: Mapping[str, Any]) -> float | None:
    value = candidate.get("expected_success_probability")
    if value is None:
        return None
    return _probability(value, label="expected_success_probability")


def _candidate_importance(candidate: Mapping[str, Any]) -> float | None:
    value = candidate.get("expected_importance")
    if value is not None:
        return _probability(value, label="expected_importance")
    ordinal = candidate.get("importance_ordinal")
    if ordinal is not None:
        return round(_ordinal(ordinal, label="importance") / 4.0, 6)
    continuous = candidate.get("importance")
    if continuous is not None:
        return _probability(continuous, label="importance")
    return None


def rank_opportunity_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    objective: str = "artifact_yield",
    max_attempts: int | None = None,
    calibration_status: str = "declared_inputs_not_calibrated",
    probability_semantics: str = "conditional_artifact_given_success",
) -> dict[str, Any]:
    """Rank candidates using only explicitly supplied, finite estimates.

    Missing primary probabilities are never inferred from model confidence or
    paper counts.  Under the backward-compatible conditional semantics only,
    an omitted conditional artifact factor uses an explicit neutral ``1.0``
    assumption, recorded per row and in the allocation payload; joint-
    probability mode never performs that fallback.  Unscored rows remain
    visible and ineligible for automatic selection.
    """

    if objective not in ALLOCATION_OBJECTIVES:
        raise ResearchGitError(
            "allocation objective must be one of " + ", ".join(ALLOCATION_OBJECTIVES)
        )
    if probability_semantics not in PROBABILITY_SEMANTICS:
        raise ResearchGitError(
            "probability_semantics must be one of " + ", ".join(PROBABILITY_SEMANTICS)
        )
    if max_attempts is not None:
        max_attempts = _positive_int(max_attempts, label="max_attempts")
    rows: list[dict[str, Any]] = []
    assumptions: set[str] = set()
    for candidate in normalize_opportunity_candidates(candidates):
        probability = _candidate_probability(candidate)
        importance = _candidate_importance(candidate)
        artifact_probability = candidate.get("expected_artifact_probability")
        if artifact_probability is not None:
            artifact_probability = _probability(
                artifact_probability, label="expected_artifact_probability"
            )
        artifact_probability_assumed = False
        source_status_eligible = candidate.get("source_status") == "open"
        source_status_reason = (
            "" if source_status_eligible else "source_status_not_open"
        )
        candidate_probability_semantics = str(
            candidate.get("expected_artifact_probability_semantics")
            or (
                "joint"
                if probability_semantics == "joint_artifact_probability"
                else "conditional_on_success"
            )
        )
        expected_candidate_semantics = (
            "joint"
            if probability_semantics == "joint_artifact_probability"
            else "conditional_on_success"
        )
        if candidate_probability_semantics != expected_candidate_semantics:
            rows.append(
                {
                    **candidate,
                    "allocation_score": None,
                    "allocation_importance": importance,
                    "allocation_eligible": False,
                    "allocation_reason": "probability_semantics_mismatch",
                    "allocation_objective": objective,
                    "probability_semantics": probability_semantics,
                    "candidate_probability_semantics": candidate_probability_semantics,
                    "probability_formula": "not_comparable",
                    "artifact_probability_assumed": False,
                    "source_status_eligible": source_status_eligible,
                }
            )
            continue
        if probability_semantics == "joint_artifact_probability":
            # In this mode expected_artifact_probability is already the joint
            # probability p(c) from FAR's allocation derivation.  Multiplying
            # by expected_success_probability would double-count success.
            score_probability = artifact_probability
            probability_formula = "expected_artifact_probability"
            if score_probability is None:
                score = None
                reason = "missing_explicit_joint_artifact_probability"
            elif objective == "artifact_yield":
                score = round(score_probability, 6)
            elif importance is None:
                score = None
                reason = "missing_explicit_importance"
            else:
                score = round(score_probability * importance, 6)
        elif probability is None:
            score = None
            reason = "missing_explicit_success_probability"
            probability_formula = "expected_success_probability * expected_artifact_probability_given_success"
        else:
            # This is the backward-compatible/default contract: the first
            # factor is P(accepted attempt), and the second is P(publishable |
            # accepted).  If the conditional factor is omitted, retain the
            # old neutral assumption but make it visible in the object.
            artifact_rate = artifact_probability
            if artifact_rate is None:
                artifact_rate = 1.0
                artifact_probability_assumed = True
                assumptions.add(
                    "missing_expected_artifact_probability_assumed_one_for_conditional_semantics"
                )
            probability_formula = "expected_success_probability * expected_artifact_probability_given_success"
            score_probability = round(probability * artifact_rate, 6)
            if objective == "artifact_yield":
                score = score_probability
            elif importance is None:
                score = None
                reason = "missing_explicit_importance"
            else:
                score = round(score_probability * importance, 6)
        if score is not None and not source_status_eligible:
            score = None
            reason = source_status_reason
        if score is not None:
            reason = (
                "eligible_from_declared_inputs_with_explicit_artifact_assumption"
                if artifact_probability_assumed
                else "eligible_from_declared_inputs"
            )
        rows.append(
            {
                **candidate,
                "allocation_score": score,
                "allocation_importance": importance,
                "allocation_eligible": score is not None and source_status_eligible,
                "allocation_reason": reason,
                "allocation_objective": objective,
                "probability_semantics": probability_semantics,
                "candidate_probability_semantics": candidate_probability_semantics,
                "probability_formula": probability_formula,
                "artifact_probability_assumed": artifact_probability_assumed,
                "source_status_eligible": source_status_eligible,
            }
        )
    rows.sort(
        key=lambda row: (
            row["allocation_score"] is None,
            -(
                (row["allocation_importance"] or 0.0)
                if objective == "best_artifact"
                else (row["allocation_score"] or 0.0)
            ),
            -(row["allocation_score"] or 0.0),
            row["candidate_id"],
        )
    )
    eligible_seen = 0
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        if row["allocation_eligible"]:
            row["selected"] = max_attempts is None or eligible_seen < max_attempts
            eligible_seen += 1
        else:
            row["selected"] = False
    core = {
        "protocol_kind": FAR_ALLOCATION_PROTOCOL,
        "objective": objective,
        "max_attempts": max_attempts,
        "calibration_status": _text(
            calibration_status, label="allocation calibration_status"
        ),
        "probability_semantics": probability_semantics,
        "probability_assumptions": sorted(assumptions),
        "candidate_set": rows,
        "eligible_count": sum(row["allocation_eligible"] for row in rows),
        "selected_count": sum(row["selected"] for row in rows),
        "selection_policy": (
            "deterministic importance-first heuristic from declared estimates "
            "for best_artifact; this is not FAR's calibrated submodular greedy "
            "policy; keep unscored candidates visible"
            if objective == "best_artifact"
            else "rank declared finite expected yield; keep unscored candidates "
            "visible; any neutral conditional-factor assumption is recorded"
        ),
    }
    return _hash_core(core, "allocation_hash")


def build_opportunity_attempt(
    *,
    pool_id: str,
    candidate_id: str,
    outcome: str,
    summary: str,
    evidence_refs: Sequence[Any] = (),
    evidence_object_ids: Sequence[str] = (),
    runner: str = "",
    attempted_at: str = "",
) -> dict[str, Any]:
    """Build one explicit attempt, including a negative/empty result."""

    normalized_outcome = _text(outcome, label="opportunity outcome").lower()
    if normalized_outcome not in OPPORTUNITY_OUTCOMES:
        raise ResearchGitError(
            "opportunity outcome must be one of " + ", ".join(OPPORTUNITY_OUTCOMES)
        )
    core: dict[str, Any] = {
        "protocol_kind": FAR_ATTEMPT_PROTOCOL,
        "pool_id": _text(pool_id, label="opportunity attempt pool_id"),
        "candidate_id": _text(candidate_id, label="opportunity attempt candidate_id"),
        "status": "completed",
        "outcome": normalized_outcome,
        "summary": _text(summary, label="opportunity attempt summary"),
        "evidence_refs": _refs(
            evidence_refs, label="opportunity attempt evidence_refs"
        ),
        "evidence_object_ids": _object_ids(
            evidence_object_ids, label="opportunity attempt evidence_object_ids"
        ),
        "runner": _optional_text(runner, label="opportunity attempt runner"),
        "attempted_at": _timestamp(
            attempted_at, label="opportunity attempt attempted_at"
        ),
    }
    return _hash_core(core, "attempt_hash")


def build_opportunity_judgment(
    *,
    attempt_id: str,
    verdict: str,
    evaluator_id: str,
    summary: str,
    independence_receipt: Mapping[str, Any],
    evidence_refs: Sequence[Any] = (),
    evidence_object_ids: Sequence[str] = (),
    target_outcome: str | None = None,
    allow_stage_override: bool = False,
    override_reason: str = "",
) -> dict[str, Any]:
    """Build an independent judge decision without calling it ground truth.

    FAR judges claimed ``NEW`` outcomes.  A retrospective review of another
    outcome is possible only through a content-hashed, reasoned override.
    """

    normalized_verdict = _text(verdict, label="opportunity judgment verdict").lower()
    if normalized_verdict not in JUDGMENT_VERDICTS:
        raise ResearchGitError(
            "opportunity judgment verdict must be one of "
            + ", ".join(JUDGMENT_VERDICTS)
        )
    normalized_target = (
        " ".join(str(target_outcome or "").split()).lower()
        if target_outcome is not None
        else None
    )
    if normalized_target is not None and normalized_target not in OPPORTUNITY_OUTCOMES:
        raise ResearchGitError(
            "opportunity judgment target_outcome must be one of "
            + ", ".join(OPPORTUNITY_OUTCOMES)
        )
    stage_override = _optional_gate_override(
        allow_stage_override=allow_stage_override,
        override_reason=override_reason,
        label="opportunity judgment",
    )
    if (
        normalized_target is not None
        and normalized_target != "new"
        and stage_override is None
    ):
        raise ResearchGitError(
            "FAR stage gate permits judgment only for outcome=new; pass "
            "allow_stage_override=True with override_reason for an audit exception"
        )
    if not isinstance(independence_receipt, Mapping):
        raise ResearchGitError(
            "opportunity judgment independence_receipt must be an object"
        )
    receipt = _bounded_json_value(
        dict(independence_receipt), label="opportunity judgment independence_receipt"
    )
    if receipt.get("identity_verified") is not False:
        raise ResearchGitError(
            "opportunity judgment must preserve the declared, not verified, independence receipt"
        )
    if receipt.get("receipt_hash") != canonical_content_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    ):
        raise ResearchGitError(
            "opportunity judgment independence receipt hash mismatch"
        )
    core: dict[str, Any] = {
        "protocol_kind": FAR_JUDGMENT_PROTOCOL,
        "attempt_id": _text(attempt_id, label="opportunity judgment attempt_id"),
        "evaluator_id": _text(evaluator_id, label="opportunity judgment evaluator_id"),
        "verdict": normalized_verdict,
        "summary": _text(summary, label="opportunity judgment summary"),
        "evidence_refs": _refs(
            evidence_refs, label="opportunity judgment evidence_refs"
        ),
        "evidence_object_ids": _object_ids(
            evidence_object_ids, label="opportunity judgment evidence_object_ids"
        ),
        "independence_receipt": receipt,
        "status": "completed",
        "verdict_status": (
            "accepted" if normalized_verdict in {"pass", "known"} else "rejected"
        ),
        "decision": normalized_verdict,
    }
    if normalized_target is not None:
        core["target_outcome"] = normalized_target
    if stage_override is not None:
        core["stage_gate_override"] = stage_override
    return _hash_core(core, "judgment_hash")


def build_opportunity_grade(
    *,
    judgment_id: str,
    grade: str,
    evaluator_id: str,
    summary: str,
    independence_receipt: Mapping[str, Any],
    evidence_refs: Sequence[Any] = (),
    evidence_object_ids: Sequence[str] = (),
    target_verdict: str | None = None,
    allow_stage_override: bool = False,
    override_reason: str = "",
) -> dict[str, Any]:
    """Build an importance grade; it is not a publication or human score.

    A grade normally follows a judged PASS/KNOWN result.  Any other use is
    preserved only with an explicit override reason.
    """

    normalized_grade = _text(grade, label="opportunity grade").lower()
    if normalized_grade not in GRADE_LEVELS:
        raise ResearchGitError(
            "opportunity grade must be one of " + ", ".join(GRADE_LEVELS)
        )
    normalized_target = (
        " ".join(str(target_verdict or "").split()).lower()
        if target_verdict is not None
        else None
    )
    if normalized_target is not None and normalized_target not in JUDGMENT_VERDICTS:
        raise ResearchGitError(
            "opportunity grade target_verdict must be one of "
            + ", ".join(JUDGMENT_VERDICTS)
        )
    stage_override = _optional_gate_override(
        allow_stage_override=allow_stage_override,
        override_reason=override_reason,
        label="opportunity grade",
    )
    if (
        normalized_target is not None
        and normalized_target not in {"pass", "known"}
        and stage_override is None
    ):
        raise ResearchGitError(
            "FAR stage gate permits grading only after verdict=pass or known; pass "
            "allow_stage_override=True with override_reason for an audit exception"
        )
    if not isinstance(independence_receipt, Mapping):
        raise ResearchGitError(
            "opportunity grade independence_receipt must be an object"
        )
    receipt = _bounded_json_value(
        dict(independence_receipt), label="opportunity grade independence_receipt"
    )
    if receipt.get("identity_verified") is not False:
        raise ResearchGitError(
            "opportunity grade must preserve the declared, not verified, independence receipt"
        )
    if receipt.get("receipt_hash") != canonical_content_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    ):
        raise ResearchGitError("opportunity grade independence receipt hash mismatch")
    core: dict[str, Any] = {
        "protocol_kind": FAR_GRADE_PROTOCOL,
        "judgment_id": _text(judgment_id, label="opportunity grade judgment_id"),
        "evaluator_id": _text(evaluator_id, label="opportunity grade evaluator_id"),
        "grade": normalized_grade,
        "summary": _text(summary, label="opportunity grade summary"),
        "evidence_refs": _refs(evidence_refs, label="opportunity grade evidence_refs"),
        "evidence_object_ids": _object_ids(
            evidence_object_ids, label="opportunity grade evidence_object_ids"
        ),
        "independence_receipt": receipt,
        "status": "completed",
        "decision": normalized_grade,
    }
    if normalized_target is not None:
        core["target_verdict"] = normalized_target
    if stage_override is not None:
        core["stage_gate_override"] = stage_override
    return _hash_core(core, "grade_hash")


def build_opportunity_funnel_summary(
    *,
    pool: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]] = (),
    judgments: Sequence[Mapping[str, Any]] = (),
    grades: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Summarize funnel coverage with explicit orphan/missing rows."""

    if not isinstance(pool, Mapping) or pool.get("protocol_kind") != FAR_POOL_PROTOCOL:
        raise ResearchGitError("funnel summary requires a FAR opportunity pool")
    candidates = pool.get("candidates")
    if not isinstance(candidates, list):
        raise ResearchGitError("opportunity pool candidates must be an array")
    if any(not isinstance(row, Mapping) for row in candidates):
        raise ResearchGitError("opportunity pool contains a malformed candidate row")
    candidate_ids = {
        str(row.get("candidate_id") or "")
        for row in candidates
        if str(row.get("candidate_id") or "")
    }
    candidate_id_values = [
        str(row.get("candidate_id") or "")
        for row in candidates
        if str(row.get("candidate_id") or "")
    ]
    duplicate_candidate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate_id_values).items()
        if count > 1
    )
    malformed_candidate_ids = sorted(
        index
        for index, row in enumerate(candidates)
        if not str(row.get("candidate_id") or "")
    )
    if any(not isinstance(row, Mapping) for row in attempts):
        raise ResearchGitError("funnel attempts must contain only JSON objects")
    if any(not isinstance(row, Mapping) for row in judgments):
        raise ResearchGitError("funnel judgments must contain only JSON objects")
    if any(not isinstance(row, Mapping) for row in grades):
        raise ResearchGitError("funnel grades must contain only JSON objects")
    attempt_rows = [dict(row) for row in attempts]
    judgment_rows = [dict(row) for row in judgments]
    grade_rows = [dict(row) for row in grades]

    def _id(row: Mapping[str, Any], *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    pool_object_id = _id(pool, "object_id", "pool_id")
    malformed_attempt_rows = sorted(
        index
        for index, row in enumerate(attempt_rows)
        if not _id(row, "object_id", "attempt_id")
    )
    malformed_judgment_rows = sorted(
        index
        for index, row in enumerate(judgment_rows)
        if not _id(row, "object_id", "judgment_id")
    )
    malformed_grade_rows = sorted(
        index
        for index, row in enumerate(grade_rows)
        if not _id(row, "object_id", "grade_id")
    )
    attempt_object_ids = [
        _id(row, "object_id", "attempt_id")
        for row in attempt_rows
        if _id(row, "object_id", "attempt_id")
    ]
    duplicate_attempt_object_ids = sorted(
        object_id
        for object_id, count in Counter(attempt_object_ids).items()
        if count > 1
    )
    judgment_object_ids = [
        _id(row, "object_id", "judgment_id")
        for row in judgment_rows
        if _id(row, "object_id", "judgment_id")
    ]
    duplicate_judgment_object_ids = sorted(
        object_id
        for object_id, count in Counter(judgment_object_ids).items()
        if count > 1
    )
    grade_object_ids = [
        _id(row, "object_id", "grade_id")
        for row in grade_rows
        if _id(row, "object_id", "grade_id")
    ]
    duplicate_grade_object_ids = sorted(
        object_id for object_id, count in Counter(grade_object_ids).items() if count > 1
    )
    attempt_by_candidate: dict[str, dict[str, Any]] = {}
    duplicate_attempt_candidate_ids: set[str] = set()
    for row in attempt_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        if candidate_id in attempt_by_candidate:
            duplicate_attempt_candidate_ids.add(candidate_id)
        elif candidate_id:
            attempt_by_candidate[candidate_id] = row
    attempted_ids = set(attempt_by_candidate)
    outcomes = Counter(str(row.get("outcome") or "unknown") for row in attempt_rows)
    attempt_by_id = {
        _id(row, "object_id", "attempt_id"): row
        for row in attempt_rows
        if _id(row, "object_id", "attempt_id")
    }
    attempt_ids = set(attempt_by_id)
    judged_attempt_ids = {
        _id(row, "attempt_id") for row in judgment_rows if _id(row, "attempt_id")
    }
    graded_judgment_ids = {
        _id(row, "judgment_id") for row in grade_rows if _id(row, "judgment_id")
    }
    missing = sorted(candidate_ids - attempted_ids)
    orphan_attempts = sorted(attempted_ids - candidate_ids)
    needs_judgment = sorted(
        _id(row, "object_id", "attempt_id")
        for row in attempt_rows
        if row.get("outcome") == "new" and _id(row, "object_id", "attempt_id")
    )
    missing_judgments = sorted(set(needs_judgment) - judged_attempt_ids)
    judgment_attempt_ids = [_id(row, "attempt_id") for row in judgment_rows]
    duplicate_judgment_attempt_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(judgment_attempt_ids).items()
        if candidate_id and count > 1
    )
    orphan_judgment_attempt_ids = sorted(
        attempt_id for attempt_id in judged_attempt_ids if attempt_id not in attempt_ids
    )
    override_by_attempt = {
        _id(row, "attempt_id"): row.get("stage_gate_override")
        for row in judgment_rows
        if _id(row, "attempt_id")
    }
    unexpected_judgment_attempt_ids = sorted(
        attempt_id
        for attempt_id in judged_attempt_ids
        if attempt_id in attempt_by_id
        and attempt_by_id[attempt_id].get("outcome") != "new"
        and not (
            isinstance(override_by_attempt.get(attempt_id), Mapping)
            and override_by_attempt[attempt_id].get("allowed") is True
        )
    )
    judgment_target_mismatch_ids = sorted(
        attempt_id
        for attempt_id in judged_attempt_ids
        if attempt_id in attempt_by_id
        and any(
            _id(row, "attempt_id") == attempt_id
            and row.get("target_outcome")
            not in (None, attempt_by_id[attempt_id].get("outcome"))
            for row in judgment_rows
        )
    )
    judgment_id_set = {
        _id(row, "object_id", "judgment_id")
        for row in judgment_rows
        if _id(row, "object_id", "judgment_id")
    }
    judgment_by_id = {
        _id(row, "object_id", "judgment_id"): row
        for row in judgment_rows
        if _id(row, "object_id", "judgment_id")
    }
    required_grade_ids = sorted(
        judgment_id
        for judgment_id, row in judgment_by_id.items()
        if str(row.get("verdict") or "").lower() in {"pass", "known"}
    )
    missing_grades = sorted(set(required_grade_ids) - graded_judgment_ids)
    grade_judgment_ids = [_id(row, "judgment_id") for row in grade_rows]
    duplicate_grade_judgment_ids = sorted(
        judgment_id
        for judgment_id, count in Counter(grade_judgment_ids).items()
        if judgment_id and count > 1
    )
    orphan_grade_judgment_ids = sorted(
        judgment_id
        for judgment_id in graded_judgment_ids
        if judgment_id not in judgment_id_set
    )
    grade_override_by_judgment = {
        _id(row, "judgment_id"): row.get("stage_gate_override")
        for row in grade_rows
        if _id(row, "judgment_id")
    }
    unexpected_grade_judgment_ids = sorted(
        judgment_id
        for judgment_id in graded_judgment_ids
        if judgment_id in judgment_by_id
        and str(judgment_by_id[judgment_id].get("verdict") or "").lower()
        not in {"pass", "known"}
        and not (
            isinstance(grade_override_by_judgment.get(judgment_id), Mapping)
            and grade_override_by_judgment[judgment_id].get("allowed") is True
        )
    )
    grade_target_mismatch_ids = sorted(
        judgment_id
        for judgment_id in graded_judgment_ids
        if judgment_id in judgment_by_id
        and any(
            _id(row, "judgment_id") == judgment_id
            and row.get("target_verdict")
            not in (None, judgment_by_id[judgment_id].get("verdict"))
            for row in grade_rows
        )
    )
    candidate_lineage_unbound_ids = sorted(
        str(row.get("candidate_id") or "")
        for row in candidates
        if str(row.get("candidate_id") or "") and not row.get("source_object_ids")
    )
    candidate_lineage_mismatch_ids = sorted(
        str(row.get("candidate_id") or "")
        for row in candidates
        if str(row.get("candidate_id") or "")
        and row.get("lineage_bound") is not bool(row.get("source_object_ids"))
    )
    lineage_complete = (
        pool.get("lineage_complete") is True
        and not candidate_lineage_unbound_ids
        and not candidate_lineage_mismatch_ids
    )
    pool_mismatch_attempt_ids = sorted(
        _id(row, "object_id", "attempt_id")
        for row in attempt_rows
        if pool_object_id
        and row.get("pool_id") not in (None, "", pool_object_id)
        and _id(row, "object_id", "attempt_id")
    )
    review_complete = (
        not missing_judgments
        and not orphan_judgment_attempt_ids
        and not duplicate_judgment_attempt_ids
        and not unexpected_judgment_attempt_ids
        and not judgment_target_mismatch_ids
        and not missing_grades
        and not orphan_grade_judgment_ids
        and not duplicate_grade_judgment_ids
        and not unexpected_grade_judgment_ids
        and not grade_target_mismatch_ids
        and not duplicate_attempt_object_ids
        and not duplicate_judgment_object_ids
        and not duplicate_grade_object_ids
        and not malformed_attempt_rows
        and not malformed_judgment_rows
        and not malformed_grade_rows
        and not pool_mismatch_attempt_ids
    )
    core: dict[str, Any] = {
        "protocol_kind": FAR_SUMMARY_PROTOCOL,
        "direction_id": pool.get("direction_id"),
        "pool_id": pool.get("pool_id") or pool.get("object_id"),
        "candidate_count": len(candidate_ids),
        "candidate_set_complete": pool.get("candidate_set_complete") is True,
        "lineage_complete": lineage_complete,
        "candidate_lineage_unbound_ids": candidate_lineage_unbound_ids,
        "candidate_lineage_mismatch_ids": candidate_lineage_mismatch_ids,
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "attempt_count": len(attempt_rows),
        "attempted_candidate_count": len(attempted_ids & candidate_ids),
        "unattempted_candidate_ids": missing,
        "orphan_attempt_candidate_ids": orphan_attempts,
        "duplicate_attempt_candidate_ids": sorted(duplicate_attempt_candidate_ids),
        "malformed_candidate_indexes": malformed_candidate_ids,
        "malformed_attempt_indexes": malformed_attempt_rows,
        "duplicate_attempt_object_ids": duplicate_attempt_object_ids,
        "pool_mismatch_attempt_ids": pool_mismatch_attempt_ids,
        "outcome_counts": dict(sorted(outcomes.items())),
        "judgment_count": len(judgment_rows),
        "missing_judgment_attempt_ids": missing_judgments,
        "duplicate_judgment_attempt_ids": duplicate_judgment_attempt_ids,
        "orphan_judgment_attempt_ids": orphan_judgment_attempt_ids,
        "unexpected_judgment_attempt_ids": unexpected_judgment_attempt_ids,
        "judgment_target_mismatch_ids": judgment_target_mismatch_ids,
        "malformed_judgment_indexes": malformed_judgment_rows,
        "duplicate_judgment_object_ids": duplicate_judgment_object_ids,
        "grade_count": len(grade_rows),
        "missing_grade_judgment_ids": missing_grades,
        "duplicate_grade_judgment_ids": duplicate_grade_judgment_ids,
        "orphan_grade_judgment_ids": orphan_grade_judgment_ids,
        "unexpected_grade_judgment_ids": unexpected_grade_judgment_ids,
        "grade_target_mismatch_ids": grade_target_mismatch_ids,
        "malformed_grade_indexes": malformed_grade_rows,
        "duplicate_grade_object_ids": duplicate_grade_object_ids,
        "review_complete": review_complete,
        "unmatched_grade_judgment_ids": sorted(graded_judgment_ids - judgment_id_set),
        "funnel_complete": not missing
        and pool.get("candidate_set_complete") is True
        and lineage_complete
        and not orphan_attempts
        and not duplicate_attempt_candidate_ids
        and review_complete
        and not malformed_candidate_ids
        and not duplicate_candidate_ids
        and not malformed_attempt_rows
        and not malformed_judgment_rows
        and not malformed_grade_rows
        and not pool_mismatch_attempt_ids,
        "quality_claim_allowed": False,
        "human_baseline_claim_allowed": False,
        "note": "Funnel coverage and declared judgments are process evidence, not a quality or human score.",
    }
    return _hash_core(core, "summary_hash")


def _resolve_protocol(
    repository: ResearchRepository,
    selector: str,
    *,
    kind: str,
    protocol: str,
    label: str,
) -> dict[str, Any]:
    item = repository.get(selector)
    if (
        item.get("kind") != kind
        or (item.get("payload") or {}).get("protocol_kind") != protocol
    ):
        raise ResearchGitError(f"{label} reference has the wrong protocol kind")
    return item


def save_research_direction(
    repo: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Persist a direction-level contract as a ``research_goal`` object."""

    repository = ResearchRepository(repo)
    commit = bool(kwargs.pop("commit", True))
    message = kwargs.pop("message", None)
    _ensure_direct_save_is_safe(repository, commit=commit)
    payload = build_research_direction(**kwargs)
    result = repository.record("research_goal", payload, state="locked")
    return _finish(
        repository,
        result,
        stage="ideation",
        subject=message or "lock research direction for opportunity discovery",
        status="locked",
        commit=commit,
    )


def save_opportunity_pool(
    repo: str,
    *,
    direction_id: str,
    candidates: Sequence[Mapping[str, Any]],
    complete_candidate_set: bool = True,
    extraction_notes: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Persist the complete Find-stage candidate set."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    direction = _resolve_protocol(
        repository,
        direction_id,
        kind="research_goal",
        protocol=FAR_DIRECTION_PROTOCOL,
        label="opportunity direction",
    )
    # Resolve selectors once, before the candidate/pool hashes are computed.
    # This prevents a short selector or @latest alias from becoming an
    # unverifiable string in the immutable lineage.
    prepared_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ResearchGitError(f"candidate {index + 1} must be a JSON object")
        prepared = dict(candidate)
        if "source_object_ids" in prepared:
            prepared["source_object_ids"] = _resolve_local_object_ids(
                repository,
                prepared.get("source_object_ids"),
                label=f"candidate {index + 1} source_object_ids",
            )
        prepared_candidates.append(prepared)
    payload = build_opportunity_pool(
        direction_id=str(direction["object_id"]),
        candidates=prepared_candidates,
        complete_candidate_set=complete_candidate_set,
        extraction_notes=extraction_notes,
    )
    payload["text"] = "Opportunity pool for " + str(direction["object_id"])
    source_object_ids = sorted(
        {
            str(source_id)
            for candidate in payload["candidates"]
            for source_id in candidate.get("source_object_ids") or []
        }
    )
    source_relations: list[dict[str, str]] = []
    allowed_source_kinds = {
        "search_receipt",
        "source_snapshot",
        "passage_evidence",
        "question",
    }
    for source_id in source_object_ids:
        source = repository.get(source_id)
        if source.get("kind") not in allowed_source_kinds:
            raise ResearchGitError(
                "opportunity source_object_ids must reference a search receipt, "
                "source snapshot, passage evidence, or question"
            )
        source_roles = [
            str(candidate.get("candidate_id"))
            for candidate in payload["candidates"]
            if str(source_id)
            in {str(item) for item in candidate.get("source_object_ids") or []}
        ]
        for candidate_id in source_roles:
            source_relations.append(
                {
                    "type": "derived_from",
                    "target": str(source["object_id"]),
                    "role": f"literature_source:{candidate_id}",
                }
            )
    payload["lineage_complete"] = bool(source_object_ids) and all(
        bool(candidate.get("source_object_ids")) for candidate in payload["candidates"]
    )
    payload["pool_hash"] = canonical_content_hash(
        {key: value for key, value in payload.items() if key != "pool_hash"}
    )
    result = repository.record(
        "question",
        payload,
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": str(direction["object_id"]),
                "role": "direction",
            },
            *source_relations,
        ],
    )
    return _finish(
        repository,
        result,
        stage="ideation",
        subject=message or "record complete research opportunity pool",
        status="locked",
        commit=commit,
    )


def save_opportunity_attempt(
    repo: str,
    *,
    pool_id: str,
    candidate_id: str,
    outcome: str,
    summary: str,
    evidence_refs: Sequence[Any] = (),
    evidence_object_ids: Sequence[str] = (),
    runner: str = "",
    attempted_at: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Persist one attempt; ``none`` is a first-class negative result."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    pool = _resolve_protocol(
        repository,
        pool_id,
        kind="question",
        protocol=FAR_POOL_PROTOCOL,
        label="opportunity pool",
    )
    candidates = (pool.get("payload") or {}).get("candidates") or []
    if candidate_id not in {
        str(row.get("candidate_id")) for row in candidates if isinstance(row, Mapping)
    }:
        raise ResearchGitError(
            "opportunity attempt candidate is not in the locked pool"
        )
    payload = build_opportunity_attempt(
        pool_id=str(pool["object_id"]),
        candidate_id=candidate_id,
        outcome=outcome,
        summary=summary,
        evidence_refs=evidence_refs,
        evidence_object_ids=_resolve_local_object_ids(
            repository,
            evidence_object_ids,
            label="opportunity attempt evidence_object_ids",
        ),
        runner=runner,
        attempted_at=attempted_at,
    )
    evidence_relations = [
        {"type": "derived_from", "target": object_id, "role": "attempt_evidence"}
        for object_id in payload.get("evidence_object_ids") or []
    ]
    result = repository.record(
        "experiment_attempt",
        payload,
        state="completed",
        relations=[
            {
                "type": "depends_on",
                "target": str(pool["object_id"]),
                "role": "opportunity_pool",
            },
            *evidence_relations,
        ],
    )
    return _finish(
        repository,
        result,
        stage="experiment",
        subject=message or "record opportunity attempt outcome",
        status="completed",
        commit=commit,
    )


def save_opportunity_judgment(
    repo: str,
    *,
    attempt_id: str,
    verdict: str,
    evaluator_id: str,
    summary: str,
    evidence_refs: Sequence[Any] = (),
    evidence_object_ids: Sequence[str] = (),
    allow_stage_override: bool = False,
    override_reason: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Persist an evaluator-disjoint judgment for a claimed NEW result."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    attempt = _resolve_protocol(
        repository,
        attempt_id,
        kind="experiment_attempt",
        protocol=FAR_ATTEMPT_PROTOCOL,
        label="opportunity attempt",
    )
    attempt_payload = attempt.get("payload") or {}
    if attempt_payload.get("status") != "completed":
        raise ResearchGitError("FAR judgment requires a completed opportunity attempt")
    target_outcome = str(attempt_payload.get("outcome") or "").lower()
    stage_override = _optional_gate_override(
        allow_stage_override=allow_stage_override,
        override_reason=override_reason,
        label="opportunity judgment",
    )
    if target_outcome != "new" and stage_override is None:
        raise ResearchGitError(
            "FAR stage gate permits judgment only for attempt outcome=new; pass "
            "allow_stage_override=True with override_reason for an audit exception"
        )
    resolved_evidence_ids = _resolve_local_object_ids(
        repository,
        evidence_object_ids,
        label="opportunity judgment evidence_object_ids",
    )
    receipt = require_independent_evaluator(
        repository,
        evaluator_id=evaluator_id,
        target_ids=[str(attempt["object_id"]), *resolved_evidence_ids],
        label="opportunity judgment",
    )
    payload = build_opportunity_judgment(
        attempt_id=str(attempt["object_id"]),
        verdict=verdict,
        evaluator_id=evaluator_id,
        summary=summary,
        independence_receipt=receipt,
        evidence_refs=evidence_refs,
        evidence_object_ids=resolved_evidence_ids,
        target_outcome=target_outcome,
        allow_stage_override=allow_stage_override,
        override_reason=override_reason,
    )
    evidence_relations = [
        {"type": "derived_from", "target": object_id, "role": "judgment_evidence"}
        for object_id in payload.get("evidence_object_ids") or []
    ]
    result = repository.record(
        "review",
        payload,
        # A declared actor-disjointness receipt is not identity verification or
        # scientific ground truth.  Keep the immutable review completed and
        # expose the verdict in the payload instead of promoting it to a gate.
        state="completed",
        actor={"actor_id": evaluator_id, "authority": "independent_evaluator"},
        relations=[
            {
                "type": "evaluates",
                "target": str(attempt["object_id"]),
                "role": "opportunity_attempt",
            },
            *evidence_relations,
        ],
    )
    return _finish(
        repository,
        result,
        stage="review",
        subject=message or "record independent opportunity judgment",
        status="completed",
        commit=commit,
    )


def save_opportunity_grade(
    repo: str,
    *,
    judgment_id: str,
    grade: str,
    evaluator_id: str,
    summary: str,
    evidence_refs: Sequence[Any] = (),
    evidence_object_ids: Sequence[str] = (),
    allow_stage_override: bool = False,
    override_reason: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Persist a second, independent importance grade."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    judgment = _resolve_protocol(
        repository,
        judgment_id,
        kind="review",
        protocol=FAR_JUDGMENT_PROTOCOL,
        label="opportunity judgment",
    )
    judgment_payload = judgment.get("payload") or {}
    if judgment_payload.get("status") != "completed":
        raise ResearchGitError("FAR grade requires a completed opportunity judgment")
    target_verdict = str(judgment_payload.get("verdict") or "").lower()
    stage_override = _optional_gate_override(
        allow_stage_override=allow_stage_override,
        override_reason=override_reason,
        label="opportunity grade",
    )
    if target_verdict not in {"pass", "known"} and stage_override is None:
        raise ResearchGitError(
            "FAR stage gate permits grading only after verdict=pass or known; pass "
            "allow_stage_override=True with override_reason for an audit exception"
        )
    resolved_evidence_ids = _resolve_local_object_ids(
        repository,
        evidence_object_ids,
        label="opportunity grade evidence_object_ids",
    )
    receipt = require_independent_evaluator(
        repository,
        evaluator_id=evaluator_id,
        target_ids=[str(judgment["object_id"]), *resolved_evidence_ids],
        label="opportunity grade",
    )
    payload = build_opportunity_grade(
        judgment_id=str(judgment["object_id"]),
        grade=grade,
        evaluator_id=evaluator_id,
        summary=summary,
        independence_receipt=receipt,
        evidence_refs=evidence_refs,
        evidence_object_ids=resolved_evidence_ids,
        target_verdict=target_verdict,
        allow_stage_override=allow_stage_override,
        override_reason=override_reason,
    )
    evidence_relations = [
        {"type": "derived_from", "target": object_id, "role": "grade_evidence"}
        for object_id in payload.get("evidence_object_ids") or []
    ]
    result = repository.record(
        "review",
        payload,
        # The evaluator is provenance-disjoint by declaration, but identity
        # and scientific correctness are not independently verified here.
        # Keep the grade completed rather than promoting it to verified.
        state="completed",
        actor={"actor_id": evaluator_id, "authority": "independent_evaluator"},
        relations=[
            {
                "type": "evaluates",
                "target": str(judgment["object_id"]),
                "role": "opportunity_judgment",
            },
            *evidence_relations,
        ],
    )
    return _finish(
        repository,
        result,
        stage="review",
        subject=message or "record independent opportunity importance grade",
        status="completed",
        commit=commit,
    )


def save_opportunity_allocation(
    repo: str,
    *,
    pool_id: str,
    objective: str = "artifact_yield",
    max_attempts: int | None = None,
    calibration_status: str = "declared_inputs_not_calibrated",
    probability_semantics: str = "conditional_artifact_given_success",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Persist a transparent allocation decision over a locked open pool.

    Allocation is intentionally fail-closed at the persistence boundary.  A
    partial extraction or a candidate whose literature status is not ``open``
    must be repaired/rechecked before it can consume an attempt budget.  The
    pure ranking helper remains permissive so callers can inspect provisional
    rows without accidentally turning them into a locked plan.
    """

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    pool = _resolve_protocol(
        repository,
        pool_id,
        kind="question",
        protocol=FAR_POOL_PROTOCOL,
        label="opportunity pool",
    )
    pool_payload = pool.get("payload") or {}
    if pool_payload.get("candidate_set_complete") is not True:
        raise ResearchGitError(
            "opportunity allocation requires a complete candidate set; "
            "record --incomplete pools for inspection only"
        )
    candidates = pool_payload.get("candidates") or []
    non_open = sorted(
        str(row.get("candidate_id") or "")
        for row in candidates
        if isinstance(row, Mapping) and row.get("source_status") != "open"
    )
    if non_open:
        raise ResearchGitError(
            "opportunity allocation requires every candidate source_status=open; "
            f"recheck or remove non-open candidates: {', '.join(non_open)}"
        )
    ranking = rank_opportunity_candidates(
        candidates,
        objective=objective,
        max_attempts=max_attempts,
        calibration_status=calibration_status,
        probability_semantics=probability_semantics,
    )
    payload = {
        "protocol_kind": FAR_ALLOCATION_PROTOCOL,
        "budget": "opportunity attempts",
        # ``None`` means no explicit cap; serializing it as zero would falsely
        # imply that the locked plan allocates no attempts.
        "limits": {"max_attempts": max_attempts},
        "information_value_required": True,
        "pool_id": str(pool["object_id"]),
        "candidate_set_complete": pool_payload.get("candidate_set_complete") is True,
        "lineage_complete": pool_payload.get("lineage_complete") is True,
        "allocation_scope": "complete_open_candidate_pool",
        **ranking,
    }
    # Rebind the pure ranking hash to the persisted payload.  This makes the
    # repository object self-contained while retaining the same deterministic
    # candidate ordering as the pure function.
    payload["allocation_hash"] = canonical_content_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"allocation_hash", "budget_hash"}
        }
    )
    payload["budget_hash"] = canonical_content_hash(
        {key: value for key, value in payload.items() if key != "budget_hash"}
    )
    result = repository.record(
        "resource_budget",
        payload,
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": str(pool["object_id"]),
                "role": "opportunity_pool",
            }
        ],
    )
    return _finish(
        repository,
        result,
        stage="plan",
        subject=message or "lock transparent opportunity allocation",
        status="locked",
        commit=commit,
    )


def inspect_opportunity_funnel(repo: str | Any, pool_id: str) -> dict[str, Any]:
    """Read a pool and its typed descendants without treating missing data as zero."""

    repository = (
        repo if isinstance(repo, ResearchRepository) else ResearchRepository(repo)
    )
    pool = _resolve_protocol(
        repository,
        pool_id,
        kind="question",
        protocol=FAR_POOL_PROTOCOL,
        label="opportunity pool",
    )
    pool_object_id = str(pool["object_id"])
    attempts = [
        item
        for item in repository.objects(kind="experiment_attempt")
        if (item.get("payload") or {}).get("protocol_kind") == FAR_ATTEMPT_PROTOCOL
        and pool_object_id
        in {
            str(relation.get("target") or "")
            for relation in item.get("relations") or []
        }
    ]
    attempt_ids = {str(item["object_id"]) for item in attempts}
    judgments = [
        item
        for item in repository.objects(kind="review")
        if (item.get("payload") or {}).get("protocol_kind") == FAR_JUDGMENT_PROTOCOL
        and any(
            str(relation.get("target") or "") in attempt_ids
            for relation in item.get("relations") or []
        )
    ]
    judgment_ids = {str(item["object_id"]) for item in judgments}
    grades = [
        item
        for item in repository.objects(kind="review")
        if (item.get("payload") or {}).get("protocol_kind") == FAR_GRADE_PROTOCOL
        and any(
            str(relation.get("target") or "") in judgment_ids
            for relation in item.get("relations") or []
        )
    ]
    summary = build_opportunity_funnel_summary(
        pool={**(pool.get("payload") or {}), "object_id": pool_object_id},
        attempts=[
            {**(item.get("payload") or {}), "object_id": item["object_id"]}
            for item in attempts
        ],
        judgments=[
            {**(item.get("payload") or {}), "object_id": item["object_id"]}
            for item in judgments
        ],
        grades=[
            {**(item.get("payload") or {}), "object_id": item["object_id"]}
            for item in grades
        ],
    )
    return {
        "pool": pool,
        "attempts": attempts,
        "judgments": judgments,
        "grades": grades,
        "summary": summary,
    }


__all__ = [
    "ALLOCATION_OBJECTIVES",
    "FAR_ALLOCATION_PROTOCOL",
    "FAR_ATTEMPT_PROTOCOL",
    "FAR_DIRECTION_PROTOCOL",
    "FAR_GRADE_PROTOCOL",
    "FAR_JUDGMENT_PROTOCOL",
    "FAR_POOL_PROTOCOL",
    "FAR_SUMMARY_PROTOCOL",
    "GRADE_LEVELS",
    "JUDGMENT_VERDICTS",
    "OPPORTUNITY_OUTCOMES",
    "OPPORTUNITY_SOURCE_STATUSES",
    "PROBABILITY_SEMANTICS",
    "build_opportunity_attempt",
    "build_opportunity_funnel_summary",
    "build_opportunity_grade",
    "build_opportunity_judgment",
    "build_opportunity_pool",
    "build_research_direction",
    "inspect_opportunity_funnel",
    "normalize_opportunity_candidates",
    "rank_opportunity_candidates",
    "save_opportunity_allocation",
    "save_opportunity_attempt",
    "save_opportunity_grade",
    "save_opportunity_judgment",
    "save_opportunity_pool",
    "save_research_direction",
]
