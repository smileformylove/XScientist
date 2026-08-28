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

from ai_scientist.protocol.attestation import ATTESTATION_SCHEMA, verify_attestation
from ai_scientist.protocol.canonical_json import (
    CANONICAL_JSON_PROFILE,
    canonical_content_hash,
)
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
STRATEGY_BUDGET_POLICY = "xscientist.research-policy-budget.v1"
COMPARISON_BOUNDARY_POLICY = "xscientist.rollout-comparison-boundary.v1"
EVALUATOR_AUTHORITIES = (
    "independent_evaluator",
    "human",
    "deterministic_gate",
    "research_agent",
)
INDEPENDENCE_POLICY = "xscientist.provenance-actor-disjoint.v1"
INDEPENDENCE_ASSURANCE = "declared_actor_disjointness"
INDEPENDENCE_ATTESTATION_PURPOSE = "research_rollout_independent_evaluation"

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_PRINCIPAL_RE = re.compile(
    r"^(?:agent|service|human):[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$"
)
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


def _principal(value: Any, *, label: str) -> str:
    """Normalize a signer identity accepted by the attestation protocol."""

    normalized = _name(value, label=label)
    if not _PRINCIPAL_RE.fullmatch(normalized):
        raise ResearchRolloutError(f"{label} must use agent:, service:, or human:")
    return normalized


def _hash_payload(value: Any, *, label: str) -> str:
    try:
        return canonical_content_hash(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ResearchRolloutError(f"{label} cannot be canonically hashed") from exc


def _normalize_attestation_envelope(value: Any) -> dict[str, Any]:
    """Validate the bounded envelope shape without claiming signature trust."""

    if not isinstance(value, Mapping):
        raise ResearchRolloutError("independence attestation must be an object")
    expected = {
        "schema_version",
        "canonicalization",
        "purpose",
        "identity",
        "key_id",
        "algorithm",
        "issued_at",
        "payload_hash",
        "signature",
        "attestation_hash",
    }
    if set(value) != expected:
        raise ResearchRolloutError("independence attestation fields are invalid")
    algorithm = str(value.get("algorithm") or "").strip().lower()
    if algorithm not in {"hmac-sha256", "ed25519"}:
        raise ResearchRolloutError("independence attestation algorithm is invalid")
    purpose = _text(
        value.get("purpose"), label="independence attestation purpose", limit=128
    )
    if purpose != INDEPENDENCE_ATTESTATION_PURPOSE:
        raise ResearchRolloutError("independence attestation purpose is invalid")
    signature = _text(
        value.get("signature"), label="independence attestation signature", limit=16384
    )
    schema_version = _text(
        value.get("schema_version"),
        label="independence attestation schema_version",
        limit=128,
    )
    if schema_version != ATTESTATION_SCHEMA:
        raise ResearchRolloutError("independence attestation schema is invalid")
    canonicalization = _text(
        value.get("canonicalization"),
        label="independence attestation canonicalization",
        limit=128,
    )
    if canonicalization != CANONICAL_JSON_PROFILE:
        raise ResearchRolloutError(
            "independence attestation canonicalization is invalid"
        )
    return {
        "schema_version": schema_version,
        "canonicalization": canonicalization,
        "purpose": purpose,
        "identity": _principal(
            value.get("identity"), label="independence attestation identity"
        ),
        "key_id": _name(value.get("key_id"), label="independence attestation key_id"),
        "algorithm": algorithm,
        "issued_at": _text(
            value.get("issued_at"),
            label="independence attestation issued_at",
            limit=64,
        ),
        "payload_hash": _hash(
            value.get("payload_hash"),
            label="independence attestation payload_hash",
        ),
        "signature": signature,
        "attestation_hash": _hash(
            value.get("attestation_hash"),
            label="independence attestation attestation_hash",
        ),
    }


def build_independence_attestation_payload(
    *,
    evaluator_id: str,
    evaluator_identity: str,
    target_hashes: Sequence[str],
    producer_actor_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the exact canonical payload an independent evaluator signs."""

    normalized_evaluator = _name(evaluator_id, label="evaluator_id")
    normalized_identity = _principal(evaluator_identity, label="evaluator_identity")
    if (
        not isinstance(target_hashes, Sequence)
        or isinstance(target_hashes, (str, bytes))
        or not target_hashes
        or len(target_hashes) > 32
    ):
        raise ResearchRolloutError("target_hashes must contain 1-32 hashes")
    normalized_targets = sorted(
        {_hash(item, label="target_hash") for item in target_hashes}
    )
    if (
        not isinstance(producer_actor_ids, Sequence)
        or isinstance(producer_actor_ids, (str, bytes))
        or not producer_actor_ids
        or len(producer_actor_ids) > 128
    ):
        raise ResearchRolloutError("producer_actor_ids must contain 1-128 ids")
    normalized_producers = sorted(
        {_name(item, label="producer_actor_id") for item in producer_actor_ids}
    )
    if normalized_evaluator in normalized_producers:
        raise ResearchRolloutError("evaluator must be disjoint from producers")
    return {
        "policy": INDEPENDENCE_POLICY,
        "assurance": INDEPENDENCE_ASSURANCE,
        "evaluator_id": normalized_evaluator,
        "evaluator_identity": normalized_identity,
        "target_hashes": normalized_targets,
        "producer_actor_ids": normalized_producers,
    }


def _normalize_independence_receipt(
    value: Any,
    *,
    evaluator_id: str,
    evidence_refs: Sequence[str],
) -> dict[str, Any]:
    """Validate an optional actor-disjoint evaluator receipt.

    A role label such as ``independent_evaluator`` is only an assertion.  The
    receipt makes the assertion auditable by binding the evaluator identity to
    the exact content-addressed artifacts it inspected and by recording the
    producer identities it was checked against.  The optional attestation is
    only trusted later by ``audit_research_rollout`` when a caller supplies a
    matching trust store; the legacy ``identity_verified`` boolean is retained
    as an observational declaration and never grants verification authority.
    """

    if not isinstance(value, Mapping):
        raise ResearchRolloutError("independence_receipt must be an object")
    _assert_safe_keys(value, label="independence receipt")
    allowed = {
        "policy",
        "assurance",
        "identity_verified",
        "evaluator_id",
        "target_hashes",
        "producer_actor_ids",
        "evaluator_identity",
        "attestation",
        "receipt_hash",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ResearchRolloutError(
            "independence receipt has unsupported fields: " + ", ".join(sorted(unknown))
        )
    if value.get("policy") != INDEPENDENCE_POLICY:
        raise ResearchRolloutError("independence receipt policy is invalid")
    if value.get("assurance") != INDEPENDENCE_ASSURANCE:
        raise ResearchRolloutError("independence receipt assurance is invalid")
    if not isinstance(value.get("identity_verified"), bool):
        raise ResearchRolloutError(
            "independence receipt identity_verified must be boolean"
        )
    normalized_evaluator = _name(
        value.get("evaluator_id"), label="independence receipt evaluator_id"
    )
    if normalized_evaluator != evaluator_id:
        raise ResearchRolloutError(
            "independence receipt evaluator_id does not match evaluation"
        )
    raw_targets = value.get("target_hashes")
    if (
        not isinstance(raw_targets, Sequence)
        or isinstance(raw_targets, (str, bytes))
        or not raw_targets
        or len(raw_targets) > 32
    ):
        raise ResearchRolloutError(
            "independence receipt target_hashes must contain 1-32 hashes"
        )
    target_hashes = sorted(
        {_hash(item, label="independence target_hash") for item in raw_targets}
    )
    normalized_refs = set(evidence_refs)
    if not normalized_refs or not set(target_hashes).issubset(normalized_refs):
        raise ResearchRolloutError(
            "independence receipt targets must be included in evaluation evidence_refs"
        )
    raw_producers = value.get("producer_actor_ids")
    if (
        not isinstance(raw_producers, Sequence)
        or isinstance(raw_producers, (str, bytes))
        or not raw_producers
        or len(raw_producers) > 128
    ):
        raise ResearchRolloutError(
            "independence receipt producer_actor_ids must contain 1-128 ids"
        )
    producer_actor_ids = sorted(
        {_name(item, label="independence producer_actor_id") for item in raw_producers}
    )
    if normalized_evaluator in producer_actor_ids:
        raise ResearchRolloutError(
            "independence receipt evaluator must be disjoint from producers"
        )
    core: dict[str, Any] = {
        "policy": INDEPENDENCE_POLICY,
        "assurance": INDEPENDENCE_ASSURANCE,
        "identity_verified": bool(value["identity_verified"]),
        "evaluator_id": normalized_evaluator,
        "target_hashes": target_hashes,
        "producer_actor_ids": producer_actor_ids,
    }
    raw_identity = value.get("evaluator_identity")
    raw_attestation = value.get("attestation")
    if (raw_identity is None) != (raw_attestation is None):
        raise ResearchRolloutError(
            "independence receipt evaluator_identity and attestation must appear together"
        )
    if raw_identity is not None:
        evaluator_identity = _principal(
            raw_identity, label="independence receipt evaluator_identity"
        )
        attestation = _normalize_attestation_envelope(raw_attestation)
        if attestation["identity"] != evaluator_identity:
            raise ResearchRolloutError(
                "independence receipt signer identity does not match attestation"
            )
        attestation_payload = build_independence_attestation_payload(
            evaluator_id=normalized_evaluator,
            evaluator_identity=evaluator_identity,
            target_hashes=target_hashes,
            producer_actor_ids=producer_actor_ids,
        )
        if attestation["payload_hash"] != _hash_payload(
            attestation_payload, label="independence attestation payload"
        ):
            raise ResearchRolloutError(
                "independence attestation payload hash does not match receipt"
            )
        core["evaluator_identity"] = evaluator_identity
        core["attestation"] = attestation
    expected_hash = _hash_payload(core, label="independence receipt")
    if value.get("receipt_hash") != expected_hash:
        raise ResearchRolloutError(
            "independence receipt hash does not match its contents"
        )
    return {**core, "receipt_hash": expected_hash}


def build_independence_receipt(
    *,
    evaluator_id: str,
    evaluator_identity: str,
    target_hashes: Sequence[str],
    producer_actor_ids: Sequence[str],
    attestation: Mapping[str, Any],
    identity_verified: bool = False,
) -> dict[str, Any]:
    """Build a hash-bound evaluator receipt around an attestation envelope.

    This builder validates structure and payload binding only.  Signature trust
    is intentionally deferred to :func:`audit_research_rollout`, which requires
    a local trust store.  ``identity_verified`` is a legacy observational field
    and has no authority even when set to true.
    """

    if not isinstance(identity_verified, bool):
        raise ResearchRolloutError("identity_verified must be boolean")
    binding = build_independence_attestation_payload(
        evaluator_id=evaluator_id,
        evaluator_identity=evaluator_identity,
        target_hashes=target_hashes,
        producer_actor_ids=producer_actor_ids,
    )
    core = {
        "policy": binding["policy"],
        "assurance": binding["assurance"],
        "identity_verified": identity_verified,
        "evaluator_id": binding["evaluator_id"],
        "target_hashes": binding["target_hashes"],
        "producer_actor_ids": binding["producer_actor_ids"],
        "evaluator_identity": binding["evaluator_identity"],
        "attestation": _normalize_attestation_envelope(attestation),
    }
    receipt = {
        **core,
        "receipt_hash": _hash_payload(core, label="independence receipt"),
    }
    return _normalize_independence_receipt(
        receipt,
        evaluator_id=binding["evaluator_id"],
        evidence_refs=binding["target_hashes"],
    )


def build_comparison_boundary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the harness/resource boundary used for tool comparisons.

    Task and rubric hashes alone do not establish a fair tool swap.  This
    optional contract records the harness, resource envelope, evaluator
    protocol, network policy, and seed policy.  It is deliberately a
    *comparison boundary*, not an outcome or quality certificate.
    """

    if not isinstance(value, Mapping):
        raise ResearchRolloutError("comparison_boundary must be an object")
    _assert_safe_keys(value, label="comparison boundary")
    allowed = {
        "harness_id",
        "resource_fingerprint",
        "evaluator_protocol_hash",
        "starting_artifact_hash",
        "network_policy",
        "seed_policy",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ResearchRolloutError(
            "comparison boundary has unsupported fields: " + ", ".join(sorted(unknown))
        )
    required = {
        "harness_id",
        "resource_fingerprint",
        "evaluator_protocol_hash",
        "network_policy",
        "seed_policy",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ResearchRolloutError(
            "comparison boundary is missing: " + ", ".join(missing)
        )
    core = {
        "policy": COMPARISON_BOUNDARY_POLICY,
        "harness_id": _name(value["harness_id"], label="harness_id"),
        "resource_fingerprint": _hash(
            value["resource_fingerprint"], label="resource_fingerprint"
        ),
        "evaluator_protocol_hash": _hash(
            value["evaluator_protocol_hash"], label="evaluator_protocol_hash"
        ),
        "starting_artifact_hash": (
            _hash(value["starting_artifact_hash"], label="starting_artifact_hash")
            if value.get("starting_artifact_hash") not in (None, "")
            else None
        ),
        "network_policy": _name(value["network_policy"], label="network_policy"),
        "seed_policy": _name(value["seed_policy"], label="seed_policy"),
    }
    return {
        **core,
        "boundary_hash": _hash_payload(core, label="comparison boundary"),
        "comparison_scope": "harness_resource_evaluator_bound",
    }


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
    # Quantize with a largest-remainder allocation so the published six-decimal
    # weights always total exactly one million units. Independent rounding can
    # otherwise produce 1.000001 and make a perfect score violate the schema.
    precision_units = 1_000_000
    scaled_weights = [
        float(row["weight"]) / total_weight * precision_units for row in rows
    ]
    allocated_units = [math.floor(value) for value in scaled_weights]
    remaining_units = precision_units - sum(allocated_units)
    remainder_order = sorted(
        range(len(rows)),
        key=lambda index: (
            -(scaled_weights[index] - allocated_units[index]),
            index,
        ),
    )
    for index in remainder_order[:remaining_units]:
        allocated_units[index] += 1
    for row, units in zip(rows, allocated_units):
        row["weight"] = units / precision_units
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
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ResearchRolloutError("rubric weights must sum to 1")
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
            if raw.get("budget_before_seconds") is not None
            else None
        )
        after = (
            _nonnegative(raw["budget_after_seconds"], label="budget_after_seconds")
            if raw.get("budget_after_seconds") is not None
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


def build_strategy_budget_summary(
    *,
    time_budget_seconds: float,
    tool_delegations: Sequence[Mapping[str, Any]] | None = None,
    outcome: str = "completed",
) -> dict[str, Any]:
    """Summarise policy decisions, budget accounting, and failure recovery.

    Faraday-style rollouts need a visible boundary between *choosing* an
    action and *executing* it.  The original delegation trace recorded each
    call, but left consumers to infer whether calls were ordered, whether the
    declared budget was actually observed, and whether a failed tool call was
    repaired.  This deterministic projection makes those checks explicit.

    The summary is observational metadata.  It does not infer hidden work,
    impute missing budget values, or turn a recovery pattern into a quality or
    causal claim.  Missing budget fields therefore produce ``partial`` (or
    ``not_available`` for an empty trace), while discontinuities are reported
    as ``inconsistent`` rather than silently repaired.
    """

    declared_budget = _nonnegative(time_budget_seconds, label="time_budget_seconds")
    if declared_budget <= 0:
        raise ResearchRolloutError("time_budget_seconds must be positive")
    normalized_outcome = str(outcome or "completed").strip().lower()
    if normalized_outcome not in ROLLOUT_OUTCOMES:
        raise ResearchRolloutError(
            f"outcome must be one of: {', '.join(ROLLOUT_OUTCOMES)}"
        )

    trace = build_tool_delegation_trace(tool_delegations or ())
    rows = list(trace["calls"])
    decision_counts = {decision: 0 for decision in TOOL_DECISIONS}
    outcome_counts = {value: 0 for value in TOOL_OUTCOMES}
    role_counts = {role: 0 for role in TOOL_ROLES}
    for row in rows:
        decision_counts[row["decision"]] += 1
        outcome_counts[row["outcome"]] += 1
        role_counts[row["role"]] += 1

    # A compact, non-sensitive strategy projection.  The full hashes remain
    # in ``tool_delegations``; this view is sufficient for dashboards and
    # comparison reports without copying input/output material.
    decision_trace = [
        {
            "sequence": row["sequence"],
            "decision": row["decision"],
            "role": row["role"],
            "outcome": row["outcome"],
            "budget_before_seconds": row["budget_before_seconds"],
            "budget_after_seconds": row["budget_after_seconds"],
            "follow_up_required": row["follow_up_required"],
        }
        for row in rows
    ]

    # Check the budget as an accounting chain.  A missing value is not zero:
    # it means that this episode cannot establish a complete budget claim.
    paired = [
        row
        for row in rows
        if row["budget_before_seconds"] is not None
        and row["budget_after_seconds"] is not None
    ]
    violations: list[str] = []
    if not rows:
        accounting_status = "not_available"
    elif len(paired) != len(rows):
        accounting_status = "partial"
        violations.append("missing_call_budget_boundary")
    else:
        first_before = float(rows[0]["budget_before_seconds"])
        if len({int(row["sequence"]) for row in rows}) != len(rows):
            violations.append("duplicate_sequence")
        for previous, current in zip(rows, rows[1:]):
            if (
                abs(
                    float(current["budget_before_seconds"])
                    - float(previous["budget_after_seconds"])
                )
                > 1e-6
            ):
                violations.append("non_contiguous_budget")
                break
        if first_before > declared_budget + 1e-6:
            violations.append("initial_budget_exceeds_declared")
        elif first_before < declared_budget - 1e-6:
            # Work before the first recorded call is real budget use, but it
            # is not attributable to a tool delegation in this trace.
            violations.append("unobserved_budget_prefix")
        accounting_status = (
            "inconsistent"
            if any(
                item
                in {
                    "non_contiguous_budget",
                    "duplicate_sequence",
                    "initial_budget_exceeds_declared",
                }
                for item in violations
            )
            else ("partial" if violations else "complete")
        )

    first_before = (
        float(paired[0]["budget_before_seconds"])
        if paired and len(paired) == len(rows)
        else None
    )
    last_after = (
        float(paired[-1]["budget_after_seconds"])
        if paired and len(paired) == len(rows)
        else None
    )
    observed_consumed = (
        round(max(0.0, first_before - last_after), 6)
        if first_before is not None and last_after is not None
        else None
    )
    unobserved_prefix = (
        round(max(0.0, declared_budget - first_before), 6)
        if first_before is not None and first_before < declared_budget
        else None
    )
    budget_utilization = (
        round(observed_consumed / declared_budget, 6)
        if observed_consumed is not None
        else None
    )
    if first_before is None:
        within_declared = None
    elif accounting_status == "inconsistent":
        within_declared = (
            False if "initial_budget_exceeds_declared" in violations else None
        )
    else:
        within_declared = first_before <= declared_budget + 1e-6
    exhausted = last_after <= 1e-6 if last_after is not None else None

    # Failures are paired with the next explicit repair/delegation action in
    # sequence order.  A stop is an acknowledged terminal response, not a
    # successful recovery.  We only require a response when the trace says so;
    # this keeps old traces valid while making omitted policy explicit.
    failures = [
        (index, row)
        for index, row in enumerate(rows)
        if row["outcome"] in {"failed", "timeout"}
    ]
    used_recovery: set[int] = set()
    recovered_count = 0
    failed_recovery_count = 0
    stopped_count = 0
    non_terminal_stop_count = sum(row["decision"] == "stop" for row in rows[:-1])
    required_count = 0
    unrecovered_required = 0
    for failure_index, failure in failures:
        if failure["follow_up_required"]:
            required_count += 1
        resolved = False
        for index in range(failure_index + 1, len(rows)):
            if index in used_recovery:
                continue
            response = rows[index]
            if response["decision"] == "stop":
                used_recovery.add(index)
                if index == len(rows) - 1:
                    stopped_count += 1
                    resolved = True
                    break
                # A stop only resolves a failure when it is terminal. Continuing
                # to execute afterwards contradicts the recorded decision.
                continue
            if response["decision"] not in {"repair", "delegate"}:
                continue
            used_recovery.add(index)
            if response["outcome"] == "success":
                recovered_count += 1
                resolved = True
                break
            failed_recovery_count += 1
        if not resolved and failure["follow_up_required"]:
            unrecovered_required += 1

    if non_terminal_stop_count:
        recovery_status = "needs_recovery"
    elif not failures:
        recovery_status = "not_triggered"
    elif unrecovered_required or non_terminal_stop_count:
        recovery_status = "needs_recovery"
    elif required_count:
        recovery_status = "satisfied"
    else:
        recovery_status = "failure_observed_without_required_follow_up"

    if normalized_outcome in {"timed_out", "cancelled"} or exhausted is True:
        next_action = "stop"
    elif unrecovered_required or non_terminal_stop_count:
        next_action = "repair"
    elif failures:
        next_action = "inspect"
    elif rows and rows[-1]["outcome"] == "success":
        next_action = "inspect"
    else:
        next_action = "delegate"

    # Keep ownership mismatches visible without rejecting historical traces:
    # runtimes may intentionally use one tool for more than one role.
    expected_roles = {
        "plan": {"research_policy"},
        "delegate": {"research_policy"},
        "repair": {"research_policy"},
        "stop": {"research_policy"},
        "execute": {"coding_executor", "other"},
        "inspect": {"coding_executor", "evaluator", "other"},
    }
    ownership_mismatches = sum(
        row["decision"] in expected_roles
        and row["role"] not in expected_roles[row["decision"]]
        for row in rows
    )

    core = {
        "policy_id": STRATEGY_BUDGET_POLICY,
        "decision_owner": "research_policy",
        "execution_owner": "coding_executor",
        "declared_outcome": normalized_outcome,
        "declared_budget_seconds": round(declared_budget, 6),
        "decision_counts": decision_counts,
        "outcome_counts": outcome_counts,
        "role_counts": role_counts,
        "decision_trace": decision_trace,
        "budget_accounting": {
            "status": accounting_status,
            "observed_call_count": len(rows),
            "accounted_call_count": len(paired),
            "first_budget_before_seconds": (
                round(first_before, 6) if first_before is not None else None
            ),
            "last_budget_after_seconds": (
                round(last_after, 6) if last_after is not None else None
            ),
            "observed_consumed_seconds": observed_consumed,
            "unobserved_prefix_seconds": unobserved_prefix,
            "budget_utilization": budget_utilization,
            "remaining_seconds": (
                round(last_after, 6) if last_after is not None else None
            ),
            "within_declared_budget": within_declared,
            "exhausted": exhausted,
            "violations": violations,
        },
        "failure_recovery": {
            "policy": "repair_or_delegate_after_failed_or_timeout_v1",
            "failure_count": len(failures),
            "required_recovery_count": required_count,
            "recovered_count": recovered_count,
            "failed_recovery_count": failed_recovery_count,
            "stopped_count": stopped_count,
            "non_terminal_stop_count": non_terminal_stop_count,
            "unrecovered_required_count": unrecovered_required,
            "status": recovery_status,
        },
        "ownership": {
            "mismatch_count": ownership_mismatches,
            "status": "consistent" if ownership_mismatches == 0 else "mixed",
        },
        "next_action": next_action,
        "summary_scope": "observational",
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
    }
    return {
        **core,
        "strategy_budget_hash": _hash_payload(core, label="strategy budget summary"),
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
    allowed = {
        "sample_id",
        "evaluator_id",
        "authority",
        "scores",
        "evidence_refs",
        "independence_receipt",
    }
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
        if len(refs) > 32:
            raise ResearchRolloutError("evaluation evidence_refs exceeds 32 references")
        normalized_refs = [_text(ref, label="evidence_ref", limit=256) for ref in refs]
        evaluator_id = _name(
            raw.get("evaluator_id") or "evaluator", label="evaluator_id"
        )
        row = {
            "sample_id": sample_id,
            "evaluator_id": evaluator_id,
            "authority": authority,
            "scores": score_row,
            "evidence_refs": sorted(set(normalized_refs)),
        }
        if raw.get("independence_receipt") is not None:
            row["independence_receipt"] = _normalize_independence_receipt(
                raw["independence_receipt"],
                evaluator_id=evaluator_id,
                evidence_refs=row["evidence_refs"],
            )
        rows.append(row)
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
    strategy_budget = build_strategy_budget_summary(
        time_budget_seconds=time_budget,
        tool_delegations=tool_trace["calls"],
        outcome=outcome,
    )
    comparison_boundary = episode.get("comparison_boundary")
    if comparison_boundary is not None:
        comparison_boundary = build_comparison_boundary(comparison_boundary)
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
        "strategy_budget_summary": strategy_budget,
        "turn_credit": turn_credit,
        "evaluation": evaluation,
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "evaluation_scope": "observational",
    }
    if comparison_boundary is not None:
        core["comparison_boundary"] = comparison_boundary
    rollout_hash = _hash_payload(core, label="rollout")
    payload = {**core, "rollout_hash": rollout_hash}
    try:
        validate_json(payload, load_schema("research_rollout"))
    except ValidationError as exc:
        raise ResearchRolloutError(
            f"rollout payload is invalid: {exc.message}"
        ) from exc
    return payload


def _audit_issue(code: str, field: str, message: str) -> dict[str, str]:
    """Create a payload-free, stable audit issue."""

    def bounded(value: Any, limit: int) -> str:
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    return {
        "code": bounded(code, 128),
        "field": bounded(field, 256),
        "message": bounded(message, 512),
    }


def _audit_exception_message(exc: Exception) -> str:
    """Return a bounded, payload-free diagnostic for an audit exception."""

    return type(exc).__name__


def rollout_producer_actor_ids(rollout: Mapping[str, Any]) -> list[str]:
    """Return canonical producer IDs available in rollout metadata.

    Human-readable provider/model labels may contain spaces or other characters
    that are not legal principal IDs.  Those values are represented by a
    canonical hash so every schema-valid rollout can still be bound by a signed
    independence receipt without copying the display label into that receipt.
    """

    identities: set[str] = set()

    def add_identity(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        identities.add(
            text
            if _NAME_RE.fullmatch(text)
            else canonical_content_hash({"producer_metadata": text})
        )

    contract = rollout.get("policy_contract")
    if isinstance(contract, Mapping):
        for key in ("policy_id", "decision_owner", "execution_owner"):
            add_identity(contract.get(key))
    for call in rollout.get("tool_delegations") or ():
        if not isinstance(call, Mapping):
            continue
        for key in ("role", "tool", "provider", "model"):
            add_identity(call.get(key))
    return sorted(identities)


def audit_research_rollout(
    rollout: Mapping[str, Any],
    *,
    evidence_hashes: Sequence[str] | None = None,
    trust_store: Mapping[str, Mapping[str, Any]] | None = None,
    max_attestation_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Audit a rollout before it can enter a scientific verification gate.

    ``build_research_rollout`` intentionally accepts incomplete exploratory
    episodes so runtimes can preserve failures.  This second, explicit gate is
    fail-closed: a completed episode is *not* verification-eligible unless its
    content hashes, task boundary, artifact references, successful executor,
    and cryptographically attested actor-disjoint evaluator receipt all agree.
    The function returns only bounded metadata and never promotes a quality or
    causal claim.

    ``evidence_hashes`` is an optional resolver supplied by a repository/index.
    When present, every evaluation reference must resolve to one of those
    hashes; without it the audit can establish syntactic content addressing but
    reports that artifact existence was not checked.

    ``trust_store`` uses the existing protocol attestation trust-store shape.
    A receipt's self-declared ``identity_verified`` field is never sufficient:
    the evaluator must sign the receipt binding and the signature must verify
    against this local trust store.  ``max_attestation_age_seconds`` can impose
    an optional freshness limit without changing the immutable rollout.
    """

    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    checks: dict[str, bool] = {
        "schema_valid": False,
        "rollout_hash_valid": False,
        "rubric_binding_valid": False,
        "tool_trace_hash_valid": False,
        "turn_credit_hash_valid": False,
        "evaluation_hash_valid": False,
        "task_boundary_valid": False,
        "comparison_boundary_valid": True,
        "budget_boundary_valid": False,
        "successful_executor_present": False,
        "executor_artifact_bound": False,
        "evidence_refs_content_addressed": False,
        "evidence_refs_resolved": evidence_hashes is not None,
        "independent_evaluator_receipt_valid": False,
        "independent_evaluator_attestation_valid": False,
        "strategy_budget_summary_valid": False,
        "strategy_budget_accounting_complete": False,
        "failure_recovery_satisfied": False,
    }
    if not isinstance(rollout, Mapping):
        blockers.append(
            _audit_issue("invalid_rollout", "rollout", "rollout must be an object")
        )
        return _rollout_audit_result(rollout, checks, blockers, warnings)

    if trust_store is not None and not isinstance(trust_store, Mapping):
        blockers.append(
            _audit_issue(
                "trust_store_invalid",
                "trust_store",
                "trust store must be a mapping keyed by key_id",
            )
        )
        trust_store = None
    if max_attestation_age_seconds is not None and (
        isinstance(max_attestation_age_seconds, bool)
        or not isinstance(max_attestation_age_seconds, int)
        or max_attestation_age_seconds < 0
    ):
        blockers.append(
            _audit_issue(
                "attestation_age_limit_invalid",
                "max_attestation_age_seconds",
                "attestation age limit must be a non-negative integer",
            )
        )
        max_attestation_age_seconds = None

    try:
        _assert_safe_keys(rollout, label="rollout")
        validate_json(rollout, load_schema("research_rollout"))
        checks["schema_valid"] = True
    except Exception as exc:
        blockers.append(
            _audit_issue("schema_invalid", "rollout", _audit_exception_message(exc))
        )

    task_hash = rollout.get("task_hash")
    split = rollout.get("split")
    task_id = rollout.get("task_id")
    rubric = rollout.get("rubric")
    if isinstance(rubric, Mapping):
        try:
            _validate_rubric(
                rubric,
                task_id=str(task_id or ""),
                task_hash=str(task_hash or ""),
                split=str(split or ""),
            )
            checks["rubric_binding_valid"] = True
        except Exception as exc:
            blockers.append(
                _audit_issue(
                    "rubric_binding_invalid", "rubric", _audit_exception_message(exc)
                )
            )
    else:
        blockers.append(
            _audit_issue("rubric_missing", "rubric", "rollout has no rubric object")
        )

    calls = rollout.get("tool_delegations")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        calls = []
        blockers.append(
            _audit_issue(
                "tool_trace_invalid",
                "tool_delegations",
                "tool_delegations must be an array",
            )
        )
    try:
        rebuilt_trace = build_tool_delegation_trace(calls)
        if rebuilt_trace.get("trace_hash") == rollout.get("tool_trace_hash"):
            checks["tool_trace_hash_valid"] = True
        else:
            blockers.append(
                _audit_issue(
                    "tool_trace_hash_mismatch",
                    "tool_trace_hash",
                    "tool trace hash does not match recorded calls",
                )
            )
    except Exception as exc:
        blockers.append(
            _audit_issue(
                "tool_trace_invalid",
                "tool_delegations",
                _audit_exception_message(exc),
            )
        )

    turn_payload = rollout.get("turn_credit")
    if isinstance(turn_payload, Mapping):
        raw_turns = turn_payload.get("turns") or []
        if isinstance(raw_turns, Sequence) and not isinstance(raw_turns, (str, bytes)):
            turn_inputs = [
                {
                    key: row.get(key)
                    for key in (
                        "turn_id",
                        "index",
                        "type",
                        "outcome",
                        "reward_before",
                        "reward_after",
                    )
                }
                for row in raw_turns
                if isinstance(row, Mapping)
            ]
            try:
                rebuilt_credit = build_turn_credit_summary(turn_inputs)
                if rebuilt_credit.get("turn_credit_hash") == turn_payload.get(
                    "turn_credit_hash"
                ):
                    checks["turn_credit_hash_valid"] = True
                else:
                    blockers.append(
                        _audit_issue(
                            "turn_credit_hash_mismatch",
                            "turn_credit",
                            "turn credit hash does not match its source rewards",
                        )
                    )
            except Exception as exc:
                blockers.append(
                    _audit_issue(
                        "turn_credit_invalid",
                        "turn_credit",
                        _audit_exception_message(exc),
                    )
                )
        else:
            blockers.append(
                _audit_issue(
                    "turn_credit_invalid", "turn_credit.turns", "turns must be an array"
                )
            )
    else:
        blockers.append(
            _audit_issue(
                "turn_credit_missing", "turn_credit", "turn credit summary is missing"
            )
        )

    evaluation_payload = rollout.get("evaluation")
    evaluation_rows: Sequence[Mapping[str, Any]] = []
    if isinstance(evaluation_payload, Mapping):
        raw_evaluations = evaluation_payload.get("evaluations") or []
        if isinstance(raw_evaluations, Sequence) and not isinstance(
            raw_evaluations, (str, bytes)
        ):
            evaluation_rows = [
                row for row in raw_evaluations if isinstance(row, Mapping)
            ]
        if isinstance(rubric, Mapping):
            try:
                rebuilt_evaluation = evaluate_replication_rollout(
                    rubric, evaluation_rows
                )
                if rebuilt_evaluation.get("evaluation_hash") == evaluation_payload.get(
                    "evaluation_hash"
                ):
                    checks["evaluation_hash_valid"] = True
                else:
                    blockers.append(
                        _audit_issue(
                            "evaluation_hash_mismatch",
                            "evaluation",
                            "evaluation hash does not match evaluator observations",
                        )
                    )
            except Exception as exc:
                blockers.append(
                    _audit_issue(
                        "evaluation_invalid",
                        "evaluation",
                        _audit_exception_message(exc),
                    )
                )
    else:
        blockers.append(
            _audit_issue(
                "evaluation_missing", "evaluation", "evaluation summary is missing"
            )
        )

    core = {key: value for key, value in rollout.items() if key != "rollout_hash"}
    try:
        expected_rollout_hash = _hash_payload(core, label="rollout")
        if expected_rollout_hash == rollout.get("rollout_hash"):
            checks["rollout_hash_valid"] = True
        else:
            blockers.append(
                _audit_issue(
                    "rollout_hash_mismatch",
                    "rollout_hash",
                    "rollout hash does not match its immutable payload",
                )
            )
    except Exception as exc:
        blockers.append(
            _audit_issue(
                "rollout_hash_invalid", "rollout_hash", _audit_exception_message(exc)
            )
        )

    # The rubric/task/split tuple is the benchmark boundary.  A result from a
    # different hidden task or split must never be treated as a tool swap.
    if isinstance(rubric, Mapping) and (
        rubric.get("task_id") != task_id
        or rubric.get("task_hash") != task_hash
        or rubric.get("split") != split
    ):
        blockers.append(
            _audit_issue(
                "task_boundary_mismatch",
                "rubric",
                "rubric task_id/task_hash/split do not match rollout boundary",
            )
        )
    else:
        checks["task_boundary_valid"] = True

    comparison_boundary = rollout.get("comparison_boundary")
    if comparison_boundary is not None:
        checks["comparison_boundary_valid"] = False
        if isinstance(comparison_boundary, Mapping):
            try:
                boundary_input = {
                    key: comparison_boundary.get(key)
                    for key in (
                        "harness_id",
                        "resource_fingerprint",
                        "evaluator_protocol_hash",
                        "starting_artifact_hash",
                        "network_policy",
                        "seed_policy",
                    )
                }
                rebuilt_boundary = build_comparison_boundary(boundary_input)
                if canonical_content_hash(rebuilt_boundary) == canonical_content_hash(
                    dict(comparison_boundary)
                ):
                    checks["comparison_boundary_valid"] = True
                else:
                    blockers.append(
                        _audit_issue(
                            "comparison_boundary_mismatch",
                            "comparison_boundary",
                            "comparison boundary hash does not match its fields",
                        )
                    )
            except Exception as exc:
                blockers.append(
                    _audit_issue(
                        "comparison_boundary_invalid",
                        "comparison_boundary",
                        _audit_exception_message(exc),
                    )
                )
        else:
            blockers.append(
                _audit_issue(
                    "comparison_boundary_invalid",
                    "comparison_boundary",
                    "comparison boundary must be an object",
                )
            )
    else:
        warnings.append(
            _audit_issue(
                "comparison_boundary_not_supplied",
                "comparison_boundary",
                "harness/resource/evaluator parity was not declared",
            )
        )

    budget = rollout.get("time_budget_seconds")
    budget_valid = True
    numeric_budget = (
        isinstance(budget, (int, float))
        and not isinstance(budget, bool)
        and math.isfinite(float(budget))
        and float(budget) > 0
    )
    if not numeric_budget:
        budget_valid = False
        blockers.append(
            _audit_issue(
                "budget_invalid",
                "time_budget_seconds",
                "time budget must be a finite positive number",
            )
        )
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            budget_valid = False
            continue
        for key in ("budget_before_seconds", "budget_after_seconds"):
            value = call.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                budget_valid = False
                blockers.append(
                    _audit_issue(
                        "budget_value_invalid",
                        f"tool_delegations[{index}].{key}",
                        "tool call budget must be a finite non-negative number",
                    )
                )
                continue
            if value is not None and numeric_budget and float(value) > float(budget):
                budget_valid = False
                blockers.append(
                    _audit_issue(
                        "budget_boundary_exceeded",
                        f"tool_delegations[{index}].{key}",
                        "tool call budget exceeds rollout time budget",
                    )
                )
    sequences = [
        call.get("sequence")
        for call in calls
        if isinstance(call, Mapping) and call.get("sequence") is not None
    ]
    hashable_sequences = [
        sequence
        for sequence in sequences
        if isinstance(sequence, int) and not isinstance(sequence, bool)
    ]
    if len(hashable_sequences) != len(sequences) or len(sequences) != len(
        set(hashable_sequences)
    ):
        budget_valid = False
        blockers.append(
            _audit_issue(
                "tool_sequence_ambiguous",
                "tool_delegations",
                "tool calls contain duplicate sequence numbers",
            )
        )
    checks["budget_boundary_valid"] = budget_valid

    # Rebuild the deterministic strategy projection instead of trusting its
    # persisted hash.  This is the bridge between the policy layer and the
    # tool trace: a report must not be able to edit the decision summary while
    # leaving the underlying calls unchanged.
    strategy_summary = rollout.get("strategy_budget_summary")
    if isinstance(strategy_summary, Mapping):
        try:
            rebuilt_strategy = build_strategy_budget_summary(
                time_budget_seconds=budget,
                tool_delegations=calls,
                outcome=str(rollout.get("outcome") or ""),
            )
            if canonical_content_hash(rebuilt_strategy) == canonical_content_hash(
                dict(strategy_summary)
            ):
                checks["strategy_budget_summary_valid"] = True
                accounting = rebuilt_strategy.get("budget_accounting") or {}
                accounting_complete = (
                    accounting.get("status") == "complete"
                    and accounting.get("within_declared_budget") is True
                )
                checks["strategy_budget_accounting_complete"] = accounting_complete
                recovery = rebuilt_strategy.get("failure_recovery") or {}
                recovery_satisfied = recovery.get("status") != "needs_recovery"
                checks["failure_recovery_satisfied"] = recovery_satisfied
                if str(rollout.get("outcome") or "") == "completed":
                    if not accounting_complete:
                        blockers.append(
                            _audit_issue(
                                "completed_with_incomplete_budget_accounting",
                                "strategy_budget_summary.budget_accounting",
                                "completed rollout requires contiguous, complete budget accounting within the declared boundary",
                            )
                        )
                    if not recovery_satisfied:
                        non_terminal_stops = int(
                            recovery.get("non_terminal_stop_count") or 0
                        )
                        blockers.append(
                            _audit_issue(
                                (
                                    "completed_with_non_terminal_stop"
                                    if non_terminal_stops
                                    else "completed_with_unrecovered_required_failure"
                                ),
                                "strategy_budget_summary.failure_recovery",
                                (
                                    "completed rollout continued after a recorded stop decision"
                                    if non_terminal_stops
                                    else "completed rollout has a required failure response that was not recovered or terminally stopped"
                                ),
                            )
                        )
            else:
                blockers.append(
                    _audit_issue(
                        "strategy_budget_summary_mismatch",
                        "strategy_budget_summary",
                        "strategy budget summary does not match rollout budget and calls",
                    )
                )
        except (
            ResearchRolloutError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            blockers.append(
                _audit_issue(
                    "strategy_budget_summary_invalid",
                    "strategy_budget_summary",
                    _audit_exception_message(exc),
                )
            )
    elif str(rollout.get("outcome") or "") == "completed":
        blockers.append(
            _audit_issue(
                "completed_without_strategy_budget_summary",
                "strategy_budget_summary",
                "completed rollout requires the deterministic strategy-budget projection",
            )
        )
    else:
        warnings.append(
            _audit_issue(
                "strategy_budget_summary_missing",
                "strategy_budget_summary",
                "non-completed rollout has no strategy-budget projection",
            )
        )

    successful_executors = [
        call
        for call in calls
        if isinstance(call, Mapping)
        and call.get("role") == "coding_executor"
        and call.get("outcome") == "success"
        and _HASH_RE.fullmatch(str(call.get("output_hash") or ""))
    ]
    checks["successful_executor_present"] = bool(successful_executors)

    all_refs: set[str] = set()
    refs_valid = True
    # With no resolver we can only establish syntactic content addressing;
    # an explicit resolver is required for a verification-eligible completed
    # rollout.  When one is supplied, start optimistic and fail closed on the
    # first missing/invalid entry.
    refs_resolved = evidence_hashes is not None
    known_hashes: set[str] = set()
    if evidence_hashes is not None:
        try:
            known_hashes = {
                _hash(item, label="evidence_hash") for item in evidence_hashes
            }
        except ResearchRolloutError as exc:
            blockers.append(
                _audit_issue(
                    "evidence_resolver_invalid",
                    "evidence_hashes",
                    _audit_exception_message(exc),
                )
            )
            refs_resolved = False
    valid_independent = 0
    producer_identities = set(rollout_producer_actor_ids(rollout))
    executor_hashes = {
        str(call.get("output_hash"))
        for call in successful_executors
        if call.get("output_hash")
    }
    for index, row in enumerate(evaluation_rows):
        refs = row.get("evidence_refs") or []
        if not refs:
            refs_valid = False
            blockers.append(
                _audit_issue(
                    "evaluation_without_evidence",
                    f"evaluation.evaluations[{index}].evidence_refs",
                    "every evaluation must bind at least one artifact hash",
                )
            )
        for ref in refs:
            ref_text = str(ref)
            all_refs.add(ref_text)
            if not _HASH_RE.fullmatch(ref_text):
                refs_valid = False
                blockers.append(
                    _audit_issue(
                        "evidence_ref_not_content_addressed",
                        f"evaluation.evaluations[{index}].evidence_refs",
                        "evidence references must use sha256:<64 lowercase hex>",
                    )
                )
            elif evidence_hashes is not None and ref_text not in known_hashes:
                refs_resolved = False
                blockers.append(
                    _audit_issue(
                        "evidence_ref_unresolved",
                        f"evaluation.evaluations[{index}].evidence_refs",
                        "evidence reference is not present in the supplied resolver",
                    )
                )
        if row.get("authority") != "independent_evaluator":
            continue
        receipt = row.get("independence_receipt")
        try:
            normalized_receipt = _normalize_independence_receipt(
                receipt,
                evaluator_id=str(row.get("evaluator_id") or ""),
                evidence_refs=[str(ref) for ref in refs],
            )
            evaluator_identity = normalized_receipt.get("evaluator_identity")
            attestation = normalized_receipt.get("attestation")
            if not evaluator_identity or not isinstance(attestation, Mapping):
                raise ResearchRolloutError(
                    "independence receipt requires a signed evaluator attestation"
                )
            if trust_store is None:
                raise ResearchRolloutError(
                    "independence receipt requires a local attestation trust store"
                )
            declared_producers = set(normalized_receipt["producer_actor_ids"])
            if not producer_identities.issubset(declared_producers):
                raise ResearchRolloutError(
                    "independence receipt does not cover all rollout producer metadata"
                )
            if not executor_hashes.intersection(
                set(normalized_receipt["target_hashes"])
            ):
                raise ResearchRolloutError(
                    "independence receipt must target a successful executor artifact"
                )
            if str(row.get("evaluator_id") or "") in producer_identities:
                raise ResearchRolloutError(
                    "independent evaluator identity collides with rollout producer metadata"
                )
            if str(evaluator_identity) in producer_identities:
                raise ResearchRolloutError(
                    "independent evaluator signer collides with rollout producer metadata"
                )
            signed_payload = build_independence_attestation_payload(
                evaluator_id=normalized_receipt["evaluator_id"],
                evaluator_identity=str(evaluator_identity),
                target_hashes=normalized_receipt["target_hashes"],
                producer_actor_ids=normalized_receipt["producer_actor_ids"],
            )
            verification = verify_attestation(
                attestation,
                signed_payload,
                trust_store=trust_store,
                purpose=INDEPENDENCE_ATTESTATION_PURPOSE,
                identity=str(evaluator_identity),
                max_age_seconds=max_attestation_age_seconds,
            )
            if verification.get("ok") is not True:
                raise ResearchRolloutError(
                    "independence evaluator attestation is not trusted"
                )
            valid_independent += 1
        except (ResearchRolloutError, TypeError, ValueError) as exc:
            blockers.append(
                _audit_issue(
                    "independent_evaluator_receipt_invalid",
                    f"evaluation.evaluations[{index}].independence_receipt",
                    _audit_exception_message(exc),
                )
            )

    checks["evidence_refs_content_addressed"] = refs_valid and bool(all_refs)
    checks["evidence_refs_resolved"] = refs_resolved
    bound_executor_hashes = executor_hashes.intersection(all_refs)
    checks["executor_artifact_bound"] = bool(bound_executor_hashes)
    outcome = str(rollout.get("outcome") or "")
    if outcome == "completed" and not bound_executor_hashes:
        blockers.append(
            _audit_issue(
                "completed_executor_artifact_not_evaluated",
                "evaluation.evaluations[].evidence_refs",
                "at least one successful executor output hash must be evaluated",
            )
        )
    checks["independent_evaluator_receipt_valid"] = valid_independent > 0
    checks["independent_evaluator_attestation_valid"] = valid_independent > 0
    if outcome == "completed":
        if not successful_executors:
            blockers.append(
                _audit_issue(
                    "completed_without_successful_executor",
                    "tool_delegations",
                    "completed rollout requires a successful coding executor with output hash",
                )
            )
        if not evaluation_rows:
            blockers.append(
                _audit_issue(
                    "completed_without_evaluation",
                    "evaluation",
                    "completed rollout requires at least one evaluator observation",
                )
            )
        if not refs_valid or not all_refs:
            blockers.append(
                _audit_issue(
                    "completed_without_content_addressed_evidence",
                    "evaluation",
                    "completed rollout requires content-addressed evidence references",
                )
            )
        if evidence_hashes is not None and not refs_resolved:
            # The resolver is an explicit claim-boundary check.  If supplied,
            # unresolved artifacts must block rather than degrade to a warning.
            blockers.append(
                _audit_issue(
                    "completed_with_unresolved_evidence",
                    "evaluation",
                    "completed rollout has evidence refs absent from the resolver",
                )
            )
        if evidence_hashes is None:
            blockers.append(
                _audit_issue(
                    "completed_without_evidence_resolver",
                    "evidence_hashes",
                    "completed rollout requires an explicit evidence hash resolver",
                )
            )
        if valid_independent == 0:
            blockers.append(
                _audit_issue(
                    "completed_without_independent_evaluator",
                    "evaluation",
                    "completed rollout requires a trusted, signed actor-disjoint evaluator receipt",
                )
            )
    else:
        warnings.append(
            _audit_issue(
                "rollout_not_completed",
                "outcome",
                "non-completed rollouts remain exploratory and cannot be verification-eligible",
            )
        )
    if evidence_hashes is None:
        warnings.append(
            _audit_issue(
                "evidence_resolver_not_supplied",
                "evidence_hashes",
                "artifact existence was not checked; only hash syntax was observed",
            )
        )

    # A single judge (even when structurally independent) is a weak signal;
    # expose this as a warning so a project-specific gate can choose a stricter
    # minimum without silently treating repeated samples as independent people.
    evaluator_ids = {
        str(row.get("evaluator_id") or "")
        for row in evaluation_rows
        if isinstance(row, Mapping)
    }
    if valid_independent == 1 and len(evaluator_ids) <= 1:
        warnings.append(
            _audit_issue(
                "single_independent_evaluator",
                "evaluation",
                "only one independent evaluator identity is recorded",
            )
        )
    return _rollout_audit_result(rollout, checks, blockers, warnings)


def _rollout_audit_result(
    rollout: Mapping[str, Any] | Any,
    checks: Mapping[str, bool],
    blockers: Sequence[Mapping[str, str]],
    warnings: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Build a stable, payload-free audit report."""

    def normalize_issues(
        items: Sequence[Mapping[str, Any]], *, category: str
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, Mapping):
                rows.append(
                    _audit_issue("malformed_issue", "audit", "issue is not an object")
                )
                continue
            rows.append(
                _audit_issue(
                    item.get("code", "unknown"),
                    item.get("field", "audit"),
                    item.get("message", "unspecified audit issue"),
                )
            )
        unique: dict[str, dict[str, str]] = {}
        for item in rows:
            try:
                key = canonical_content_hash(item)
            except Exception:
                key = canonical_content_hash(
                    {"code": item["code"], "field": item["field"]}
                )
            unique[key] = item
        ordered = sorted(
            unique.values(),
            key=lambda item: (item.get("code", ""), item.get("field", "")),
        )
        if len(ordered) <= 64:
            return ordered
        omitted = len(ordered) - 63
        return [
            _audit_issue(
                f"{category}_truncated",
                "audit",
                f"{omitted} additional {category} issues were omitted by the report limit",
            ),
            *ordered[:63],
        ]

    normalized_blockers = normalize_issues(blockers, category="blockers")
    normalized_warnings = normalize_issues(warnings, category="warnings")
    raw_task_id = (
        str(rollout.get("task_id") or "") if isinstance(rollout, Mapping) else ""
    )
    task_id = raw_task_id if _NAME_RE.fullmatch(raw_task_id) else ""
    raw_task_hash = (
        str(rollout.get("task_hash") or "") if isinstance(rollout, Mapping) else ""
    )
    task_hash = raw_task_hash if _HASH_RE.fullmatch(raw_task_hash) else ""
    raw_outcome = (
        str(rollout.get("outcome") or "") if isinstance(rollout, Mapping) else ""
    )
    outcome = raw_outcome if raw_outcome in ROLLOUT_OUTCOMES else ""
    base = {
        "schema_version": "xscientist.research-rollout-audit.v1",
        "task_id": task_id,
        "task_hash": task_hash,
        "outcome": outcome,
        "status": (
            "blocked"
            if normalized_blockers
            else ("ready" if outcome == "completed" else "not_ready")
        ),
        "complete": not normalized_blockers,
        "verification_allowed": bool(
            outcome == "completed" and not normalized_blockers
        ),
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "checks": dict(checks),
        "blockers": normalized_blockers,
        "warnings": normalized_warnings,
        "payloads_disclosed": False,
    }
    try:
        audit_hash = canonical_content_hash(base)
    except Exception:
        # The report is deliberately still useful for a malformed input: the
        # fallback hash covers only bounded status metadata and never echoes the
        # offending payload.
        audit_hash = canonical_content_hash(
            {
                "schema_version": base["schema_version"],
                "status": "blocked",
                "blocker_count": len(normalized_blockers),
            }
        )
    return {**base, "audit_hash": audit_hash}


def assess_tool_swap_compatibility(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    strict: bool = False,
    audit_evidence_hashes: Sequence[str] | None = None,
    audit_trust_store: Mapping[str, Mapping[str, Any]] | None = None,
    max_attestation_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Check whether two rollouts share a comparison boundary.

    This only reports eligibility.  It never turns a tool swap into a quality
    claim, because evaluator and execution stochasticity still need a separate
    benchmark protocol.  ``strict=True`` additionally requires both rollouts
    to pass :func:`audit_research_rollout` using the supplied union evidence
    resolver and local trust store, and rejects comparing the same rollout or
    the same tool signature.  The default remains the historical boundary-only
    check for backwards compatibility; strict mode fails closed when either
    audit input is absent.
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
    left_boundary = reference.get("comparison_boundary")
    right_boundary = candidate.get("comparison_boundary")
    if (left_boundary is None) != (right_boundary is None):
        reasons.append("comparison_boundary_missing")
    elif isinstance(left_boundary, Mapping) and isinstance(right_boundary, Mapping):
        try:
            left_boundary = build_comparison_boundary(
                {
                    key: left_boundary.get(key)
                    for key in (
                        "harness_id",
                        "resource_fingerprint",
                        "evaluator_protocol_hash",
                        "starting_artifact_hash",
                        "network_policy",
                        "seed_policy",
                    )
                }
            )
            right_boundary = build_comparison_boundary(
                {
                    key: right_boundary.get(key)
                    for key in (
                        "harness_id",
                        "resource_fingerprint",
                        "evaluator_protocol_hash",
                        "starting_artifact_hash",
                        "network_policy",
                        "seed_policy",
                    )
                }
            )
            if left_boundary != right_boundary:
                reasons.append("comparison_boundary_mismatch")
        except ResearchRolloutError:
            reasons.append("comparison_boundary_invalid")
    elif left_boundary is not None or right_boundary is not None:
        reasons.append("comparison_boundary_invalid")
    if strict:
        for label, rollout in (("reference", reference), ("candidate", candidate)):
            audit = audit_research_rollout(
                rollout,
                evidence_hashes=audit_evidence_hashes,
                trust_store=audit_trust_store,
                max_attestation_age_seconds=max_attestation_age_seconds,
            )
            if not audit["verification_allowed"]:
                reasons.append(f"{label}_rollout_not_verification_ready")
        if reference.get("rollout_hash") == candidate.get("rollout_hash"):
            reasons.append("same_rollout")

        def _tool_signature(rollout: Mapping[str, Any]) -> tuple[str, ...]:
            rows = rollout.get("tool_delegations") or []
            return tuple(
                sorted(
                    {
                        ":".join(
                            str(row.get(key) or "")
                            for key in ("role", "tool", "provider", "model")
                        )
                        for row in rows
                        if isinstance(row, Mapping)
                        and row.get("role") == "coding_executor"
                    }
                )
            )

        if _tool_signature(reference) == _tool_signature(candidate):
            reasons.append("tool_signature_not_changed")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "official_comparable": False,
        "strict": bool(strict),
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "comparison_scope": (
            "harness_resource_evaluator_bound"
            if reference.get("comparison_boundary") is not None
            and candidate.get("comparison_boundary") is not None
            else "boundary_only"
        ),
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
    finished = _finish(
        repository,
        result,
        stage="experiment",
        subject=message or f"record research rollout {payload['task_id']}",
        status=state,
        commit=commit,
    )
    return {**finished, "rollout": payload}


__all__ = [
    "COMPARISON_BOUNDARY_POLICY",
    "EVALUATOR_AUTHORITIES",
    "INDEPENDENCE_ASSURANCE",
    "INDEPENDENCE_ATTESTATION_PURPOSE",
    "INDEPENDENCE_POLICY",
    "ResearchRolloutError",
    "ROLLOUT_PROFILE_URI",
    "ROLLOUT_SCHEMA_VERSION",
    "RUBRIC_DIMENSIONS",
    "assess_tool_swap_compatibility",
    "audit_research_rollout",
    "build_comparison_boundary",
    "build_independence_attestation_payload",
    "build_independence_receipt",
    "build_replication_rubric",
    "build_research_rollout",
    "build_strategy_budget_summary",
    "build_tool_delegation_trace",
    "build_turn_credit_summary",
    "evaluate_replication_rollout",
    "rollout_producer_actor_ids",
    "save_research_rollout",
]
