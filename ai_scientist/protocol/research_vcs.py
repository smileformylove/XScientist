"""Canonical Research VCS object construction and validation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from .hashing import content_hash
from .schemas import load_schema

RESEARCH_OBJECT_SCHEMA = "xscientist.research-object.v1"
RESEARCH_OBJECT_KINDS = (
    "question",
    "hypothesis",
    "preregistration",
    "research_plan",
    "experiment_attempt",
    "metric",
    "evidence",
    "claim",
    "review",
    "gate_decision",
    "manuscript",
    "reproduction",
    "agent_candidate",
    "agent_evaluation",
)
RESEARCH_OBJECT_STATES = (
    "draft",
    "locked",
    "running",
    "completed",
    "failed",
    "timed_out",
    "cancelled",
    "rejected",
    "superseded",
    "verified",
    "promoted",
)
RESEARCH_RELATION_TYPES = (
    "depends_on",
    "supports",
    "refutes",
    "supersedes",
    "reproduces",
    "contradicts",
    "derived_from",
    "evaluates",
    "promotes",
)
RESEARCH_AUTHORITIES = (
    "research_agent",
    "recorder",
    "independent_evaluator",
    "deterministic_gate",
    "human",
)


class ResearchObjectError(ValueError):
    """A Research VCS object is malformed or has lost content integrity."""


_PAYLOAD_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "question": ("text", "question"),
    "hypothesis": ("statement", "core_hypothesis", "title"),
    "preregistration": ("status", "registration_id", "hypothesis_id"),
    "research_plan": ("plan_id", "tasks", "summary", "hypothesis"),
    "experiment_attempt": ("status",),
    "metric": ("name", "metric", "value"),
    "evidence": (
        "result",
        "summary",
        "measurement",
        "metrics",
        "metric",
        "effect",
        "ara_manifest_hash",
    ),
    "claim": ("statement", "text", "claim", "claim_hash"),
    "review": ("summary", "status", "decision", "report_hash"),
    "gate_decision": ("decision", "claim_promotion_allowed"),
    "manuscript": ("title", "status", "final", "idea_idx"),
    "reproduction": (
        "checkpoint_hash",
        "receipt_hash",
        "reproduction_level",
        "verdict",
    ),
    "agent_candidate": (
        "candidate_id",
        "candidate_hash",
        "candidate",
        "promotion",
        "summary",
        "version",
    ),
    "agent_evaluation": (
        "candidate_id",
        "candidate",
        "summary",
        "status",
        "verdict",
        "decision",
        "gate_hash",
    ),
}


def research_payload_issues(kind: str, payload: Mapping[str, Any]) -> list[str]:
    """Return semantic payload gaps relevant to traceability.

    The outer Research Object schema deliberately remains forwards-compatible.
    This second layer distinguishes a syntactically storable legacy object from
    one that is sufficiently typed for closure auditing.
    """

    normalized_kind = str(kind or "").strip()
    if normalized_kind not in RESEARCH_OBJECT_KINDS:
        return [f"unsupported research object kind: {normalized_kind}"]
    if not isinstance(payload, Mapping):
        return ["payload must be a mapping"]
    if not payload:
        return ["payload must not be empty"]
    identity_fields = _PAYLOAD_IDENTITY_FIELDS.get(normalized_kind, ())
    if identity_fields and not any(
        field in payload and payload.get(field) not in (None, "", [], {})
        for field in identity_fields
    ):
        return [
            f"{normalized_kind} payload requires one of: " + ", ".join(identity_fields)
        ]
    return []


def validate_research_payload(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate kind-specific minimum semantics and return a detached payload."""

    issues = research_payload_issues(kind, payload)
    if issues:
        raise ResearchObjectError("invalid research payload: " + "; ".join(issues))
    return deepcopy(dict(payload))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResearchObjectError(f"{label} must be a mapping")
    return deepcopy(dict(value))


def _normalise_relations(
    relations: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in relations or ():
        if not isinstance(raw, Mapping):
            raise ResearchObjectError("each relation must be a mapping")
        relation_type = str(raw.get("type") or "").strip()
        target = str(raw.get("target") or "").strip()
        role = str(raw.get("role") or "").strip()
        row = {"type": relation_type, "target": target}
        if role:
            row["role"] = role
        rows.append(row)
    return sorted(
        {tuple(sorted(row.items())): row for row in rows}.values(),
        key=lambda row: (row["type"], row["target"], row.get("role", "")),
    )


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"object_id", "created_at", "content_hash"}
    }


def build_research_object(
    *,
    kind: str,
    payload: Mapping[str, Any],
    state: str = "draft",
    relations: Sequence[Mapping[str, Any]] | None = None,
    actor: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, schema-valid Research VCS object."""

    normalized_kind = str(kind or "").strip()
    normalized_state = str(state or "").strip()
    semantic_payload = validate_research_payload(normalized_kind, payload)
    actor_payload = _mapping(actor, label="actor") or {
        "actor_id": "xscientist",
        "authority": "research_agent",
    }
    core = {
        "schema_version": RESEARCH_OBJECT_SCHEMA,
        "protocol_kind": "research_object",
        "kind": normalized_kind,
        "state": normalized_state,
        "payload": semantic_payload,
        "relations": _normalise_relations(relations),
        "actor": actor_payload,
        "provenance": _mapping(provenance, label="provenance"),
    }
    object_hash = content_hash(core)
    result = {
        **core,
        "object_id": f"rso-{object_hash.split(':', 1)[1][:16]}",
        "created_at": str(created_at or _now_iso()),
        "content_hash": object_hash,
    }
    try:
        validate_json(result, load_schema("research_object"))
    except ValidationError as exc:
        raise ResearchObjectError(f"invalid research object: {exc.message}") from exc
    return result


def validate_research_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, canonical identity, and object identifier."""

    if not isinstance(payload, Mapping):
        raise ResearchObjectError("research object must be a mapping")
    result = deepcopy(dict(payload))
    try:
        validate_json(result, load_schema("research_object"))
    except ValidationError as exc:
        raise ResearchObjectError(f"invalid research object: {exc.message}") from exc
    validate_research_payload(
        str(result.get("kind") or ""), result.get("payload") or {}
    )
    expected = content_hash(_identity_payload(result))
    if result.get("content_hash") != expected:
        raise ResearchObjectError("research object content hash mismatch")
    expected_id = f"rso-{expected.split(':', 1)[1][:16]}"
    if result.get("object_id") != expected_id:
        raise ResearchObjectError("research object identifier mismatch")
    return result


__all__ = [
    "RESEARCH_AUTHORITIES",
    "RESEARCH_OBJECT_KINDS",
    "RESEARCH_OBJECT_SCHEMA",
    "RESEARCH_OBJECT_STATES",
    "RESEARCH_RELATION_TYPES",
    "ResearchObjectError",
    "build_research_object",
    "research_payload_issues",
    "validate_research_payload",
    "validate_research_object",
]
