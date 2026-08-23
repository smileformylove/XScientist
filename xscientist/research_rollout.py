"""Auditable research-policy rollouts.

Faraday's most portable idea is a division of labour: a research policy chooses
the next action while a coding tool executes it, and a task-specific evaluator
checks whether the result is scientifically meaningful.  This module records
that contract without running a model, retaining prompts, or treating an LLM
judge as ground truth.

The builders are deliberately pure and bounded.  They can be used by an agent
runtime, imported from an external runner, or persisted as one immutable
``research_rollout`` Research VCS object.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema

from .research_commands import _ensure_direct_save_is_safe, _finish
from .research_git import ResearchGitError
from .research_vcs import ResearchRepository

ROLLOUT_SCHEMA_VERSION = "xscientist.research-rollout.v1"
ROLLOUT_PROFILE_URI = "https://xscientist.io/profiles/research-rollout/v1"
MAX_TOOL_CALLS = 256
MAX_TURNS = 512
MAX_EVALUATIONS = 32
MAX_TEXT = 512
MAX_ID = 128

RUBRIC_DIMENSIONS = (
    "result_fidelity",
    "claim_support",
    "implementation_fidelity",
    "resource_efficiency",
    "scientific_integrity",
)
TURN_TYPES = ("plan", "delegate", "execute", "inspect", "repair", "judge", "stop")
TOOL_ROLES = ("research_policy", "coding_executor", "evaluator", "other")
TOOL_OUTCOMES = ("success", "failed", "timeout", "skipped", "unknown")
TOOL_DECISIONS = ("plan", "delegate", "execute", "inspect", "repair", "stop")
ROLLOUT_OUTCOMES = ("completed", "failed", "timed_out", "cancelled")
SPLITS = ("train", "development", "validation", "test", "holdout", "external")
EVALUATOR_AUTHORITIES = (
    "independent_evaluator",
    "human",
    "deterministic_gate",
    "research_agent",
)

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_FORBIDDEN_KEYS = re.compile(
    r"(?:prompt|api[_-]?key|access[_-]?token|auth[_-]?token|password|secret|"
    r"raw[_-]?(?:prompt|response|output|input)|stdout|stderr|credential)",
    re.IGNORECASE,
)


class ResearchRolloutError(ValueError):
    """Raised when a rollout would make an unverifiable or unsafe record."""


def _text(value: Any, *, label: str, limit: int = MAX_TEXT) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ResearchRolloutError(f"{label} cannot be empty")
    if len(normalized) > limit:
        raise ResearchRolloutError(f"{label} exceeds {limit} characters")
    return normalized


def _optional_text(value: Any, *, label: str, limit: int = MAX_TEXT) -> str | None:
    if value in (None, ""):
        return None
    return _text(value, label=label, limit=limit)


def _hash(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not _HASH_RE.fullmatch(normalized):
        raise ResearchRolloutError(f"{label} must use sha256:<64 lowercase hex>")
    return normalized


def _score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchRolloutError(f"{label} must be a finite number in [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ResearchRolloutError(f"{label} must be a finite number in [0, 1]")
    return round(parsed, 6)


def _nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchRolloutError(f"{label} must be a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ResearchRolloutError(f"{label} must be a finite non-negative number")
    return round(parsed, 6)


def _bounded_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResearchRolloutError(f"{label} must be an integer >= {minimum}")
    return value


def _assert_safe_keys(row: Mapping[str, Any], *, label: str) -> None:
    for key in row:
        if _FORBIDDEN_KEYS.search(str(key)):
            raise ResearchRolloutError(f"{label} contains a sensitive field")


def _name(value: Any, *, label: str) -> str:
    normalized = _text(value, label=label, limit=MAX_ID)
    if not _NAME_RE.fullmatch(normalized):
        raise ResearchRolloutError(f"{label} contains unsupported characters")
    return normalized


def _hash_payload(value: Any, *, label: str) -> str:
    try:
        return canonical_content_hash(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResearchRolloutError(f"{label} cannot be canonically hashed") from exc


def build_replication_rubric(
    *,
    task_id: str,
    task_hash: str,
    split: str = "test",
    dimensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a task-specific, evaluator-facing five-dimension rubric.

    Scores are intentionally not produced here.  The rubric is a locked
    evaluation contract; a later evaluator may provide observations against
    it.  ``quality_claim_allowed`` remains false until a project-specific
    independent promotion gate is satisfied.
    """

    task = _name(task_id, label="task_id")
    digest = _hash(task_hash, label="task_hash")
    normalized_split = str(split or "test").strip().lower()
    if normalized_split not in SPLITS:
        raise ResearchRolloutError(f"split must be one of: {', '.join(SPLITS)}")
    overrides = dimensions or {}
    if not isinstance(overrides, Mapping):
        raise ResearchRolloutError("dimensions must be a mapping")
    defaults = {
        "result_fidelity": "The observed result matches the blinded reference artifact or metric.",
        "claim_support": "The result supports the task's stated scientific claim without overreach.",
        "implementation_fidelity": "The implementation tests the intended method rather than a shortcut.",
        "resource_efficiency": "The work respects the declared time and compute budget.",
        "scientific_integrity": "The record avoids hard-coded outputs, leakage, and unsupported conclusions.",
    }
    rows: list[dict[str, Any]] = []
    for dimension in RUBRIC_DIMENSIONS:
        raw = overrides.get(dimension, {})
        if isinstance(raw, str):
            description = _text(raw, label=f"rubric {dimension}")
            weight = 1.0 / len(RUBRIC_DIMENSIONS)
        elif isinstance(raw, Mapping):
            _assert_safe_keys(raw, label=f"rubric {dimension}")
            description = _text(
                raw.get("description") or defaults[dimension],
                label=f"rubric {dimension} description",
            )
            weight = _score(
                raw.get("weight", 1.0 / len(RUBRIC_DIMENSIONS)),
                label=f"rubric {dimension} weight",
            )
        else:
            raise ResearchRolloutError(f"rubric {dimension} must be a string or object")
        rows.append(
            {
                "id": dimension,
                "description": description,
                "weight": weight,
                "score_min": 0.0,
                "score_max": 1.0,
            }
        )
    total_weight = sum(float(row["weight"]) for row in rows)
    if total_weight <= 0:
        raise ResearchRolloutError("rubric weights must sum to a positive value")
    for row in rows:
        row["weight"] = round(float(row["weight"]) / total_weight, 6)
    core = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "task_id": task,
        "task_hash": digest,
        "split": normalized_split,
        "reference_visibility": "evaluator_only",
        "dimensions": rows,
        "scoring_scale": {"min": 0.0, "max": 1.0},
        "quality_claim_allowed": False,
    }
    rubric_hash = _hash_payload(core, label="rubric")
    return {
        **core,
        "rubric_id": "rubric-" + rubric_hash.split(":", 1)[1][:16],
        "rubric_hash": rubric_hash,
    }


def _validate_rubric(
    rubric: Mapping[str, Any],
    *,
    task_id: str | None = None,
    task_hash: str | None = None,
    split: str | None = None,
) -> None:
    if not isinstance(rubric, Mapping):
        raise ResearchRolloutError("rubric must be an object")
    if rubric.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
        raise ResearchRolloutError("rubric schema_version is invalid")
    if rubric.get("reference_visibility") != "evaluator_only":
        raise ResearchRolloutError("rubric reference_visibility must be evaluator_only")
    if rubric.get("quality_claim_allowed") is not False:
        raise ResearchRolloutError("rubric quality claims must remain disabled")
    required = {
        "task_id",
        "task_hash",
        "split",
        "dimensions",
        "scoring_scale",
        "rubric_id",
        "rubric_hash",
    }
    missing = sorted(required - set(rubric))
    if missing:
        raise ResearchRolloutError("rubric is missing: " + ", ".join(missing))
    if rubric.get("scoring_scale") != {"min": 0.0, "max": 1.0}:
        raise ResearchRolloutError("rubric scoring_scale must be [0, 1]")
    if task_id is not None and rubric.get("task_id") != task_id:
        raise ResearchRolloutError("rubric task_id does not match rollout task")
    if task_hash is not None and rubric.get("task_hash") != task_hash:
        raise ResearchRolloutError("rubric task_hash does not match rollout task")
    if split is not None and rubric.get("split") != split:
        raise ResearchRolloutError("rubric split does not match rollout split")
    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != len(RUBRIC_DIMENSIONS):
        raise ResearchRolloutError("rubric must contain exactly five dimensions")
    dimension_ids = [
        str(row.get("id") or "") for row in dimensions if isinstance(row, Mapping)
    ]
    if dimension_ids != list(RUBRIC_DIMENSIONS):
        raise ResearchRolloutError(
            "rubric dimensions must use the published five-dimension order"
        )
    weights: list[float] = []
    for dimension, row in zip(RUBRIC_DIMENSIONS, dimensions):
        if not isinstance(row, Mapping):
            raise ResearchRolloutError(
                f"rubric dimension {dimension} must be an object"
            )
        if row.get("score_min") != 0.0 or row.get("score_max") != 1.0:
            raise ResearchRolloutError(
                f"rubric dimension {dimension} has an invalid scale"
            )
        _text(row.get("description"), label=f"rubric {dimension} description")
        weights.append(_score(row.get("weight"), label=f"rubric {dimension} weight"))
    if sum(weights) <= 0:
        raise ResearchRolloutError("rubric weights must sum to a positive value")
    core = {
        key: rubric[key]
        for key in (
            "schema_version",
            "task_id",
            "task_hash",
            "split",
            "reference_visibility",
            "dimensions",
            "scoring_scale",
            "quality_claim_allowed",
        )
    }
    expected_hash = _hash_payload(core, label="rubric")
    if rubric.get("rubric_hash") != expected_hash:
        raise ResearchRolloutError("rubric_hash does not match rubric contents")
    if rubric.get("rubric_id") != "rubric-" + expected_hash.split(":", 1)[1][:16]:
        raise ResearchRolloutError("rubric_id does not match rubric_hash")


def build_tool_delegation_trace(
    calls: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Normalize metadata-only outer-policy → coding-tool calls."""

    if calls is None:
        calls = ()
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise ResearchRolloutError("tool_delegations must be an array")
    if len(calls) > MAX_TOOL_CALLS:
        raise ResearchRolloutError(f"tool_delegations exceeds {MAX_TOOL_CALLS} calls")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed = {
        "call_id",
        "sequence",
        "role",
        "tool",
        "provider",
        "model",
        "decision",
        "outcome",
        "input_hash",
        "output_hash",
        "budget_before_seconds",
        "budget_after_seconds",
        "follow_up_required",
    }
    for index, raw in enumerate(calls):
        if not isinstance(raw, Mapping):
            raise ResearchRolloutError("each tool delegation must be an object")
        _assert_safe_keys(raw, label="tool delegation")
        unknown = set(raw) - allowed
        if unknown:
            if any(_FORBIDDEN_KEYS.search(str(key)) for key in unknown):
                raise ResearchRolloutError("tool delegation contains a sensitive field")
            raise ResearchRolloutError(
                "tool delegation has unsupported fields: " + ", ".join(sorted(unknown))
            )
        call_id = _name(raw.get("call_id") or f"call-{index + 1}", label="call_id")
        if call_id in seen:
            raise ResearchRolloutError(f"duplicate tool call id: {call_id}")
        seen.add(call_id)
        sequence = _bounded_int(raw.get("sequence", index), label="tool sequence")
        role = str(raw.get("role") or "coding_executor").strip().lower()
        if role not in TOOL_ROLES:
            raise ResearchRolloutError(f"unsupported tool role: {role}")
        decision = str(raw.get("decision") or "execute").strip().lower()
        if decision not in TOOL_DECISIONS:
            raise ResearchRolloutError(f"unsupported tool decision: {decision}")
        outcome = str(raw.get("outcome") or "unknown").strip().lower()
        if outcome not in TOOL_OUTCOMES:
            raise ResearchRolloutError(f"unsupported tool outcome: {outcome}")
        before = (
            _nonnegative(raw["budget_before_seconds"], label="budget_before_seconds")
            if "budget_before_seconds" in raw
            else None
        )
        after = (
            _nonnegative(raw["budget_after_seconds"], label="budget_after_seconds")
            if "budget_after_seconds" in raw
            else None
        )
        if before is not None and after is not None and after > before:
            raise ResearchRolloutError(
                "budget_after_seconds cannot exceed budget_before_seconds"
            )
        output_hash = (
            _hash(raw["output_hash"], label="output_hash")
            if raw.get("output_hash") not in (None, "")
            else None
        )
        if outcome == "success" and output_hash is None:
            raise ResearchRolloutError("successful tool calls require output_hash")
        row = {
            "call_id": call_id,
            "sequence": sequence,
            "role": role,
            "tool": _name(raw.get("tool") or "coding-tool", label="tool"),
            "provider": _optional_text(
                raw.get("provider"), label="provider", limit=128
            ),
            "model": _optional_text(raw.get("model"), label="model", limit=128),
            "decision": decision,
            "outcome": outcome,
            "input_hash": (
                _hash(raw["input_hash"], label="input_hash")
                if raw.get("input_hash") not in (None, "")
                else None
            ),
            "output_hash": output_hash,
            "budget_before_seconds": before,
            "budget_after_seconds": after,
            "follow_up_required": raw.get("follow_up_required", False),
        }
        if not isinstance(row["follow_up_required"], bool):
            raise ResearchRolloutError("follow_up_required must be boolean")
        rows.append(row)
    rows.sort(key=lambda row: (row["sequence"], row["call_id"]))
    return {
        "calls": rows,
        "call_count": len(rows),
        "trace_hash": _hash_payload(rows, label="tool delegation trace"),
        "trace_scope": "metadata_only",
        "tool_swap_claim_allowed": False,
    }


def build_turn_credit_summary(
    turns: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Assign observational post-hoc credit from recorded reward deltas.

    This is intentionally not a causal attribution or an RL implementation.
    A runtime may use the summary to construct a training dataset, but the
    persisted object always declares the credit as observational.
    """

    if turns is None:
        turns = ()
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ResearchRolloutError("turns must be an array")
    if len(turns) > MAX_TURNS:
        raise ResearchRolloutError(f"turns exceeds {MAX_TURNS} entries")
    allowed = {"turn_id", "index", "type", "outcome", "reward_before", "reward_after"}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    has_reward = False
    complete_reward = True
    for position, raw in enumerate(turns):
        if not isinstance(raw, Mapping):
            raise ResearchRolloutError("each turn must be an object")
        _assert_safe_keys(raw, label="turn")
        unknown = set(raw) - allowed
        if unknown:
            if any(_FORBIDDEN_KEYS.search(str(key)) for key in unknown):
                raise ResearchRolloutError("turn contains a sensitive field")
            raise ResearchRolloutError(
                "turn has unsupported fields: " + ", ".join(sorted(unknown))
            )
        turn_id = _name(raw.get("turn_id") or f"turn-{position + 1}", label="turn_id")
        if turn_id in seen:
            raise ResearchRolloutError(f"duplicate turn id: {turn_id}")
        seen.add(turn_id)
        turn_type = str(raw.get("type") or "execute").strip().lower()
        if turn_type not in TURN_TYPES:
            raise ResearchRolloutError(f"unsupported turn type: {turn_type}")
        outcome = str(raw.get("outcome") or "observed").strip().lower()
        if len(outcome) > 64:
            raise ResearchRolloutError("turn outcome exceeds 64 characters")
        before = raw.get("reward_before")
        after = raw.get("reward_after")
        if before is None or after is None:
            complete_reward = False
            delta = None
        else:
            before = _score(before, label="reward_before")
            after = _score(after, label="reward_after")
            has_reward = True
            delta = round(max(0.0, after - before), 6)
        rows.append(
            {
                "turn_id": turn_id,
                "index": _bounded_int(raw.get("index", position), label="turn index"),
                "type": turn_type,
                "outcome": outcome,
                "reward_before": before,
                "reward_after": after,
                "positive_delta": delta,
            }
        )
    positive_total = sum(float(row["positive_delta"] or 0.0) for row in rows)
    for row in rows:
        delta = row["positive_delta"]
        row["credit"] = (
            round(float(delta) / positive_total, 6)
            if positive_total > 0 and delta is not None
            else None
        )
    if has_reward:
        method = "post_hoc_positive_delta_v1"
    else:
        method = "not_available"
    core = {
        "turns": rows,
        "turn_count": len(rows),
        "credit_method": method,
        "credit_scope": "observational",
        "causal_claim_allowed": False,
        "reward_trace_complete": bool(rows) and complete_reward,
        "positive_credit_total": round(positive_total, 6),
    }
    return {**core, "turn_credit_hash": _hash_payload(core, label="turn credit")}


def evaluate_replication_rollout(
    rubric: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Aggregate bounded evaluator observations against a locked rubric."""

    _validate_rubric(rubric)
    dimensions = rubric.get("dimensions")
    if evaluations is None:
        evaluations = ()
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)):
        raise ResearchRolloutError("evaluations must be an array")
    if len(evaluations) > MAX_EVALUATIONS:
        raise ResearchRolloutError(f"evaluations exceeds {MAX_EVALUATIONS} samples")
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    allowed = {"sample_id", "evaluator_id", "authority", "scores", "evidence_refs"}
    for position, raw in enumerate(evaluations):
        if not isinstance(raw, Mapping):
            raise ResearchRolloutError("each evaluation must be an object")
        _assert_safe_keys(raw, label="evaluation")
        unknown = set(raw) - allowed
        if unknown:
            if any(_FORBIDDEN_KEYS.search(str(key)) for key in unknown):
                raise ResearchRolloutError("evaluation contains a sensitive field")
            raise ResearchRolloutError(
                "evaluation has unsupported fields: " + ", ".join(sorted(unknown))
            )
        sample_id = _name(
            raw.get("sample_id") or f"sample-{position + 1}", label="sample_id"
        )
        if sample_id in seen_samples:
            raise ResearchRolloutError(f"duplicate evaluation sample: {sample_id}")
        seen_samples.add(sample_id)
        authority = str(raw.get("authority") or "independent_evaluator").strip().lower()
        if authority not in EVALUATOR_AUTHORITIES:
            raise ResearchRolloutError(f"unsupported evaluator authority: {authority}")
        scores = raw.get("scores")
        if not isinstance(scores, Mapping):
            raise ResearchRolloutError("evaluation scores must be an object")
        if set(scores) != set(RUBRIC_DIMENSIONS):
            raise ResearchRolloutError("evaluation must score every rubric dimension")
        score_row = {
            dimension: _score(scores[dimension], label=f"score {dimension}")
            for dimension in RUBRIC_DIMENSIONS
        }
        refs = raw.get("evidence_refs") or []
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise ResearchRolloutError("evaluation evidence_refs must be an array")
        normalized_refs = [
            _text(ref, label="evidence_ref", limit=256) for ref in refs[:32]
        ]
        rows.append(
            {
                "sample_id": sample_id,
                "evaluator_id": _name(
                    raw.get("evaluator_id") or "evaluator", label="evaluator_id"
                ),
                "authority": authority,
                "scores": score_row,
                "evidence_refs": sorted(set(normalized_refs)),
            }
        )
    means: dict[str, float | None] = {}
    for dimension in RUBRIC_DIMENSIONS:
        values = [row["scores"][dimension] for row in rows]
        means[dimension] = round(sum(values) / len(values), 6) if values else None
    weights = {str(row["id"]): float(row["weight"]) for row in dimensions}
    sample_overall = [
        round(
            sum(
                row["scores"][dimension] * weights[dimension]
                for dimension in RUBRIC_DIMENSIONS
            ),
            6,
        )
        for row in rows
    ]
    overall = (
        round(sum(sample_overall) / len(sample_overall), 6) if sample_overall else None
    )
    summary = {
        "status": "not_evaluated" if not rows else "complete",
        "sample_count": len(rows),
        "distinct_evaluator_count": len({row["evaluator_id"] for row in rows}),
        "dimension_means": means,
        "overall_mean": overall,
        "overall_disagreement": (
            round(max(sample_overall) - min(sample_overall), 6)
            if sample_overall
            else None
        ),
        "evaluation_scope": "observational",
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "judge_reference_only": True,
    }
    core = {"evaluations": rows, "summary": summary}
    return {**core, "evaluation_hash": _hash_payload(core, label="evaluation")}


def build_research_rollout(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Build one schema-valid, metadata-only research episode."""

    if not isinstance(episode, Mapping):
        raise ResearchRolloutError("rollout episode must be an object")
    _assert_safe_keys(episode, label="rollout episode")
    task_id = _name(episode.get("task_id"), label="task_id")
    task_hash = _hash(episode.get("task_hash"), label="task_hash")
    split = str(episode.get("split") or "test").strip().lower()
    outcome = str(episode.get("outcome") or "completed").strip().lower()
    if split not in SPLITS:
        raise ResearchRolloutError(f"split must be one of: {', '.join(SPLITS)}")
    if outcome not in ROLLOUT_OUTCOMES:
        raise ResearchRolloutError(
            f"outcome must be one of: {', '.join(ROLLOUT_OUTCOMES)}"
        )
    time_budget = _nonnegative(
        episode.get("time_budget_seconds"), label="time_budget_seconds"
    )
    if time_budget <= 0:
        raise ResearchRolloutError("time_budget_seconds must be positive")
    rubric = episode.get("rubric")
    if rubric is None:
        rubric = build_replication_rubric(
            task_id=task_id, task_hash=task_hash, split=split
        )
    _validate_rubric(rubric, task_id=task_id, task_hash=task_hash, split=split)
    tool_trace = build_tool_delegation_trace(episode.get("tool_delegations") or ())
    turn_credit = build_turn_credit_summary(episode.get("turns") or ())
    evaluation = evaluate_replication_rollout(rubric, episode.get("evaluations") or ())
    core = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "profile": ROLLOUT_PROFILE_URI,
        "task_id": task_id,
        "task_hash": task_hash,
        "split": split,
        "time_budget_seconds": time_budget,
        "outcome": outcome,
        "policy_contract": {
            "policy_id": "xscientist.research-policy-tool-delegation.v1",
            "decision_owner": "research_policy",
            "execution_owner": "coding_executor",
            "tool_selection_recorded": True,
            "quality_claim_allowed": False,
        },
        "rubric": dict(rubric),
        "tool_delegations": tool_trace["calls"],
        "tool_trace_hash": tool_trace["trace_hash"],
        "turn_credit": turn_credit,
        "evaluation": evaluation,
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "evaluation_scope": "observational",
    }
    rollout_hash = _hash_payload(core, label="rollout")
    payload = {**core, "rollout_hash": rollout_hash}
    try:
        validate_json(payload, load_schema("research_rollout"))
    except ValidationError as exc:
        raise ResearchRolloutError(
            f"rollout payload is invalid: {exc.message}"
        ) from exc
    return payload


def assess_tool_swap_compatibility(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Check whether two rollouts share a comparison boundary.

    This only reports eligibility.  It never turns a tool swap into a quality
    claim, because evaluator and execution stochasticity still need a separate
    benchmark protocol.
    """

    reasons: list[str] = []
    for field, label in (
        ("task_hash", "task_hash"),
        ("rubric", "rubric"),
        ("split", "split"),
        ("time_budget_seconds", "time_budget_seconds"),
    ):
        left = reference.get(field)
        right = candidate.get(field)
        if (
            field == "rubric"
            and isinstance(left, Mapping)
            and isinstance(right, Mapping)
        ):
            left = left.get("rubric_hash")
            right = right.get("rubric_hash")
        if left != right:
            reasons.append(f"{label}_mismatch")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "official_comparable": False,
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "comparison_scope": "boundary_only",
    }


def save_research_rollout(
    repo: str,
    episode: Mapping[str, Any],
    *,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Persist a rollout as a completed/failed Research VCS object."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    payload = build_research_rollout(episode)
    state = {
        "completed": "completed",
        "failed": "failed",
        "timed_out": "timed_out",
        "cancelled": "cancelled",
    }[payload["outcome"]]
    result = repository.record(
        "research_rollout",
        payload,
        state=state,
        actor={"actor_id": "xscientist:rollout", "authority": "recorder"},
    )
    return _finish(
        repository,
        result,
        stage="experiment",
        subject=message or f"record research rollout {payload['task_id']}",
        status=state,
        commit=commit,
    )


__all__ = [
    "EVALUATOR_AUTHORITIES",
    "ResearchRolloutError",
    "ROLLOUT_PROFILE_URI",
    "ROLLOUT_SCHEMA_VERSION",
    "RUBRIC_DIMENSIONS",
    "assess_tool_swap_compatibility",
    "build_replication_rubric",
    "build_research_rollout",
    "build_tool_delegation_trace",
    "build_turn_credit_summary",
    "evaluate_replication_rollout",
    "save_research_rollout",
]
