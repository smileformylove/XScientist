"""Canonical Research VCS object construction and validation."""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from .canonical_json import canonical_content_hash
from .hashing import content_hash
from .schemas import load_schema

RESEARCH_OBJECT_SCHEMA = "xscientist.research-object.v1"
RESEARCH_OBJECT_IDENTITY_PROFILE = "xscientist.research-object-identity.v1"
RESEARCH_SEMANTIC_PROFILE_SCHEMA = "xscientist.research-semantic-profile.v1"

CORE_RESEARCH_OBJECT_KINDS = (
    "question",
    "search_plan",
    "search_receipt",
    "source_snapshot",
    "passage_evidence",
    "hypothesis",
    "preregistration",
    "research_plan",
    "experiment_attempt",
    "observation",
    "metric",
    "evidence",
    "claim",
    "review",
    "gate_decision",
    "manuscript",
    "reproduction",
    "agent_candidate",
    "agent_evaluation",
    "context_snapshot",
)
EPISTEMIC_RESEARCH_OBJECT_KINDS = (
    "inference",
    "warrant",
    "assumption",
    "method",
    "estimand",
    "effect_estimate",
    "protocol_deviation",
    "sensitivity_analysis",
    "risk_of_bias",
    "evidence_synthesis",
    "challenge",
    "source_update",
    "context_robustness",
)
AUTONOMOUS_RESEARCH_OBJECT_KINDS = (
    "research_goal",
    "action_proposal",
    "experiment_design",
    "resource_budget",
    "stopping_decision",
    "novelty_check",
    "evaluation_blinding",
    "human_escalation",
)
RESEARCH_OBJECT_KINDS = (
    *CORE_RESEARCH_OBJECT_KINDS,
    *EPISTEMIC_RESEARCH_OBJECT_KINDS,
    *AUTONOMOUS_RESEARCH_OBJECT_KINDS,
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
    "retrieves",
    "cites",
    "quotes",
    "observes",
    "generated_by",
    "qualified_supports",
    "qualified_refutes",
    "attests",
    "uses_context",
    "has_premise",
    "uses_method",
    "under_assumption",
    "addresses_estimand",
    "has_effect_estimate",
    "challenges_inference",
    "derived_by",
    "qualifies",
    "updates",
    "invalidates",
    "selects",
    "rejects",
    "consumes",
    "produces",
)
RESEARCH_AUTHORITIES = (
    "research_agent",
    "recorder",
    "independent_evaluator",
    "deterministic_gate",
    "human",
)


def _profile_descriptor(
    uri: str, version: str, kinds: Sequence[str], relations: Sequence[str]
) -> dict[str, Any]:
    core = {
        "schema": RESEARCH_SEMANTIC_PROFILE_SCHEMA,
        "uri": uri,
        "version": version,
        "kinds": sorted(set(kinds)),
        "relations": sorted(set(relations)),
    }
    return {**core, "schema_digest": canonical_content_hash(core)}


_CORE_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/research-core/v1",
    "1.0.0",
    CORE_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
_EPISTEMIC_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/epistemic-argument/v1",
    "1.0.0",
    EPISTEMIC_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
_AUTONOMOUS_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/autonomous-research/v1",
    "1.0.0",
    AUTONOMOUS_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
BUILTIN_RESEARCH_PROFILES = {
    profile["uri"]: profile
    for profile in (_CORE_PROFILE, _EPISTEMIC_PROFILE, _AUTONOMOUS_PROFILE)
}


def _default_profile_for_kind(kind: str) -> dict[str, Any] | None:
    if kind in CORE_RESEARCH_OBJECT_KINDS:
        return deepcopy(_CORE_PROFILE)
    if kind in EPISTEMIC_RESEARCH_OBJECT_KINDS:
        return deepcopy(_EPISTEMIC_PROFILE)
    if kind in AUTONOMOUS_RESEARCH_OBJECT_KINDS:
        return deepcopy(_AUTONOMOUS_PROFILE)
    return None


def _normalise_semantic_profile(
    kind: str, profile: Mapping[str, Any] | None
) -> dict[str, Any]:
    default = _default_profile_for_kind(kind)
    if profile is None:
        if default is None:
            raise ResearchObjectError(
                "extension research object kinds require semantic_profile metadata"
            )
        return default
    if not isinstance(profile, Mapping):
        raise ResearchObjectError("semantic_profile must be a mapping")
    row = deepcopy(dict(profile))
    uri = str(row.get("uri") or "").strip()
    version = str(row.get("version") or "").strip()
    schema_digest = str(row.get("schema_digest") or "").strip()
    if not uri.startswith(("https://", "http://", "urn:")):
        raise ResearchObjectError("semantic_profile uri must be an absolute URI")
    if not version:
        raise ResearchObjectError("semantic_profile version is required")
    if not _is_sha256(schema_digest):
        raise ResearchObjectError(
            "semantic_profile schema_digest must use sha256:<64 hex>"
        )
    if set(row) != {"schema", "uri", "version", "kinds", "relations", "schema_digest"}:
        raise ResearchObjectError("semantic_profile has unsupported fields")
    if row.get("schema") != RESEARCH_SEMANTIC_PROFILE_SCHEMA:
        raise ResearchObjectError("semantic_profile schema is invalid")
    if not isinstance(row.get("kinds"), list) or kind not in row["kinds"]:
        raise ResearchObjectError("semantic_profile does not declare the object kind")
    if not isinstance(row.get("relations"), list):
        raise ResearchObjectError("semantic_profile relations must be an array")
    expected_digest = canonical_content_hash(
        {key: value for key, value in row.items() if key != "schema_digest"}
    )
    if schema_digest != expected_digest:
        raise ResearchObjectError("semantic_profile schema_digest mismatch")
    builtin = BUILTIN_RESEARCH_PROFILES.get(uri)
    if builtin is not None and row != builtin:
        raise ResearchObjectError("built-in semantic_profile metadata mismatch")
    return row


def research_profile_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Report whether an object's semantic profile has a local validator."""

    profile = payload.get("semantic_profile")
    if not isinstance(profile, Mapping):
        return {"declared": False, "validator_available": False, "builtin": False}
    uri = str(profile.get("uri") or "")
    builtin = BUILTIN_RESEARCH_PROFILES.get(uri)
    return {
        "declared": True,
        "uri": uri,
        "version": str(profile.get("version") or ""),
        "schema_digest": str(profile.get("schema_digest") or ""),
        "validator_available": builtin is not None,
        "builtin": builtin is not None,
    }


class ResearchObjectError(ValueError):
    """A Research VCS object is malformed or has lost content integrity."""


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


_PAYLOAD_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "question": ("text", "question"),
    "search_plan": ("queries", "question", "search_plan_hash"),
    "search_receipt": ("receipt_hash", "provider", "candidates"),
    "source_snapshot": (
        "source_hash",
        "content_hash",
        "metadata_hash",
        "doi",
        "pmid",
        "arxiv_id",
        "title",
    ),
    "passage_evidence": ("passage_hash", "quote_hash", "locator"),
    "hypothesis": ("statement", "core_hypothesis", "title"),
    "preregistration": ("status", "registration_id", "hypothesis_id"),
    "research_plan": ("plan_id", "tasks", "summary", "hypothesis"),
    "experiment_attempt": ("status",),
    "observation": ("measurement", "result", "output_hash", "metrics"),
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
    "context_snapshot": (
        "context_hash",
        "source_closure_hash",
        "memory_snapshot_hash",
    ),
    "inference": ("statement", "conclusion", "inference_hash"),
    "warrant": ("statement", "rule", "warrant_hash"),
    "assumption": ("statement", "assumption", "assumption_hash"),
    "method": ("name", "description", "method_hash"),
    "estimand": ("outcome", "summary", "estimand_hash"),
    "effect_estimate": ("estimate", "value", "effect_hash"),
    "protocol_deviation": ("description", "reason", "deviation_hash"),
    "sensitivity_analysis": ("summary", "result", "analysis_hash"),
    "risk_of_bias": ("assessment", "domain", "assessment_hash"),
    "evidence_synthesis": ("summary", "conclusion", "synthesis_hash"),
    "challenge": ("statement", "reason", "challenge_hash"),
    "source_update": ("status", "update_type", "update_hash"),
    "context_robustness": ("status", "result", "robustness_hash"),
    "research_goal": ("question", "objective", "goal_hash"),
    "action_proposal": ("action", "summary", "proposal_hash"),
    "experiment_design": ("summary", "design", "design_hash"),
    "resource_budget": ("budget", "limits", "budget_hash"),
    "stopping_decision": ("decision", "reason", "decision_hash"),
    "novelty_check": ("verdict", "summary", "check_hash"),
    "evaluation_blinding": ("policy", "summary", "blinding_hash"),
    "human_escalation": ("reason", "question", "escalation_hash"),
}


def research_payload_issues(
    kind: str,
    payload: Mapping[str, Any],
    semantic_profile: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return semantic payload gaps relevant to traceability.

    The outer Research Object schema deliberately remains forwards-compatible.
    This second layer distinguishes a syntactically storable legacy object from
    one that is sufficiently typed for closure auditing.
    """

    normalized_kind = str(kind or "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", normalized_kind):
        return [f"invalid research object kind: {normalized_kind}"]
    if normalized_kind not in RESEARCH_OBJECT_KINDS:
        try:
            _normalise_semantic_profile(normalized_kind, semantic_profile)
        except ResearchObjectError as exc:
            return [str(exc)]
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
    issues: list[str] = []
    required_fields: dict[str, tuple[str, ...]] = {
        "search_plan": ("question", "queries", "search_plan_hash"),
        "search_receipt": (
            "profile",
            "provider",
            "query",
            "retrieved_at",
            "candidates",
            "receipt_hash",
        ),
        "source_snapshot": ("title", "content_hash", "source_hash"),
        "passage_evidence": (
            "source_id",
            "locator",
            "quote",
            "quote_hash",
            "passage_hash",
        ),
        "source_update": (
            "source_id",
            "status",
            "provider",
            "checked_at",
            "update_type",
            "update_hash",
        ),
    }
    for field in required_fields.get(normalized_kind, ()):
        if field not in payload or payload.get(field) in (None, ""):
            issues.append(f"{normalized_kind} payload requires {field}")
    if normalized_kind == "search_plan" and not isinstance(
        payload.get("queries"), list
    ):
        issues.append("search_plan queries must be an array")
    if normalized_kind == "search_receipt" and not isinstance(
        payload.get("candidates"), list
    ):
        issues.append("search_receipt candidates must be an array")
    if (
        normalized_kind == "search_receipt"
        and payload.get("profile") == "xscientist.retrieval-receipt.v2"
    ):
        candidates = payload.get("candidates") or []
        request = payload.get("request")
        if not isinstance(request, Mapping):
            issues.append("search_receipt v2 requires a request object")
        elif payload.get("request_hash") != canonical_content_hash(dict(request)):
            issues.append("search_receipt request_hash mismatch")
        if payload.get("candidate_set_hash") != canonical_content_hash(candidates):
            issues.append("search_receipt candidate_set_hash mismatch")
        completeness = payload.get("completeness")
        if not isinstance(completeness, Mapping):
            issues.append("search_receipt v2 requires completeness metadata")
        elif completeness.get("candidate_count") != len(candidates):
            issues.append("search_receipt candidate count mismatch")
    commitment_fields = {
        "search_plan": "search_plan_hash",
        "search_receipt": "receipt_hash",
        "source_snapshot": "source_hash",
        "passage_evidence": "passage_hash",
        "source_update": "update_hash",
    }
    commitment_field = commitment_fields.get(normalized_kind)
    if commitment_field and payload.get(commitment_field):
        expected = canonical_content_hash(
            {key: value for key, value in payload.items() if key != commitment_field}
        )
        if payload.get(commitment_field) != expected:
            issues.append(f"{normalized_kind} {commitment_field} mismatch")
    if normalized_kind == "passage_evidence" and payload.get("quote"):
        expected_quote = canonical_content_hash(str(payload["quote"]))
        if payload.get("quote_hash") != expected_quote:
            issues.append("passage_evidence quote_hash mismatch")
        selector = payload.get("selector")
        if selector is not None:
            if not isinstance(selector, Mapping):
                issues.append("passage_evidence selector must be an object")
            else:
                selector_core = {
                    key: value
                    for key, value in selector.items()
                    if key != "selector_hash"
                }
                expected_selector_hash = canonical_content_hash(selector_core)
                if selector.get("selector_hash") != expected_selector_hash:
                    issues.append("passage_evidence selector hash mismatch")
                if payload.get("selector_hash") != expected_selector_hash:
                    issues.append("passage_evidence selector binding mismatch")
                selectors = selector.get("selectors") or []
                exacts = [
                    item.get("exact")
                    for item in selectors
                    if isinstance(item, Mapping)
                    and item.get("type") == "TextQuoteSelector"
                ]
                if str(payload.get("quote")) not in exacts:
                    issues.append("passage_evidence selector exact quote mismatch")
    if isinstance(payload.get("scope"), Mapping):
        expected_scope = canonical_content_hash(dict(payload["scope"]))
        if payload.get("scope_hash") != expected_scope:
            issues.append(f"{normalized_kind} structured scope_hash mismatch")
    for field in (
        "search_plan_hash",
        "receipt_hash",
        "source_hash",
        "content_hash",
        "metadata_hash",
        "passage_hash",
        "quote_hash",
        "scope_hash",
        "request_hash",
        "candidate_set_hash",
        "selector_hash",
        "update_hash",
    ):
        value = payload.get(field)
        if value not in (None, "") and not _is_sha256(value):
            issues.append(f"{normalized_kind} {field} must use sha256:<64 hex>")
    return issues


def validate_research_payload(
    kind: str,
    payload: Mapping[str, Any],
    semantic_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate kind-specific minimum semantics and return a detached payload."""

    issues = research_payload_issues(kind, payload, semantic_profile)
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
        if (
            relation_type not in RESEARCH_RELATION_TYPES
            and not relation_type.startswith(("https://", "http://", "urn:"))
        ):
            raise ResearchObjectError(
                "extension relation types must use an absolute URI"
            )
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
        if key
        not in {
            "object_id",
            "qualified_id",
            "identity_profile",
            "created_at",
            "content_hash",
        }
    }


def build_research_object(
    *,
    kind: str,
    payload: Mapping[str, Any],
    state: str = "draft",
    relations: Sequence[Mapping[str, Any]] | None = None,
    actor: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    semantic_profile: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, schema-valid Research VCS object."""

    normalized_kind = str(kind or "").strip()
    normalized_state = str(state or "").strip()
    normalized_profile = _normalise_semantic_profile(normalized_kind, semantic_profile)
    semantic_payload = validate_research_payload(
        normalized_kind, payload, normalized_profile
    )
    actor_payload = _mapping(actor, label="actor") or {
        "actor_id": "xscientist",
        "authority": "research_agent",
    }
    normalized_relations = _normalise_relations(relations)
    declared_relations = set(normalized_profile.get("relations") or [])
    undeclared_relations = sorted(
        {
            row["type"]
            for row in normalized_relations
            if row["type"] not in declared_relations
        }
    )
    if undeclared_relations:
        raise ResearchObjectError(
            "semantic_profile does not declare relation types: "
            + ", ".join(undeclared_relations)
        )
    core = {
        "schema_version": RESEARCH_OBJECT_SCHEMA,
        "protocol_kind": "research_object",
        "kind": normalized_kind,
        "semantic_profile": normalized_profile,
        "state": normalized_state,
        "payload": semantic_payload,
        "relations": normalized_relations,
        "actor": actor_payload,
        "provenance": _mapping(provenance, label="provenance"),
    }
    object_hash = content_hash(core)
    result = {
        **core,
        "object_id": f"rso-{object_hash.split(':', 1)[1][:16]}",
        "qualified_id": (
            "urn:xscientist:research-object:sha256:" + object_hash.split(":", 1)[1]
        ),
        "identity_profile": RESEARCH_OBJECT_IDENTITY_PROFILE,
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
    semantic_profile = result.get("semantic_profile")
    if semantic_profile is None:
        # Objects produced before semantic profiles were introduced remain
        # valid; new builders always emit an explicit, content-bound profile.
        if str(result.get("kind") or "") not in CORE_RESEARCH_OBJECT_KINDS:
            raise ResearchObjectError("research object semantic_profile is missing")
        legacy_relations = _normalise_relations(result.get("relations") or [])
        if any(
            relation["type"] not in RESEARCH_RELATION_TYPES
            for relation in legacy_relations
        ):
            raise ResearchObjectError(
                "legacy research objects cannot use extension relation types"
            )
    else:
        normalized_profile = _normalise_semantic_profile(
            str(result.get("kind") or ""), semantic_profile
        )
        undeclared = sorted(
            {
                str(relation.get("type") or "")
                for relation in result.get("relations") or []
                if str(relation.get("type") or "")
                not in set(normalized_profile.get("relations") or [])
            }
        )
        if undeclared:
            raise ResearchObjectError(
                "semantic_profile does not declare relation types: "
                + ", ".join(undeclared)
            )
    validate_research_payload(
        str(result.get("kind") or ""),
        result.get("payload") or {},
        semantic_profile,
    )
    expected = content_hash(_identity_payload(result))
    if result.get("content_hash") != expected:
        raise ResearchObjectError("research object content hash mismatch")
    expected_id = f"rso-{expected.split(':', 1)[1][:16]}"
    if result.get("object_id") != expected_id:
        raise ResearchObjectError("research object identifier mismatch")
    qualified_id = result.get("qualified_id")
    identity_profile = result.get("identity_profile")
    if qualified_id is not None or identity_profile is not None:
        expected_qualified = (
            "urn:xscientist:research-object:sha256:" + expected.split(":", 1)[1]
        )
        if identity_profile != RESEARCH_OBJECT_IDENTITY_PROFILE:
            raise ResearchObjectError("research object identity profile mismatch")
        if qualified_id != expected_qualified:
            raise ResearchObjectError("research object qualified identifier mismatch")
    return result


__all__ = [
    "AUTONOMOUS_RESEARCH_OBJECT_KINDS",
    "BUILTIN_RESEARCH_PROFILES",
    "CORE_RESEARCH_OBJECT_KINDS",
    "EPISTEMIC_RESEARCH_OBJECT_KINDS",
    "RESEARCH_AUTHORITIES",
    "RESEARCH_OBJECT_KINDS",
    "RESEARCH_OBJECT_IDENTITY_PROFILE",
    "RESEARCH_OBJECT_SCHEMA",
    "RESEARCH_OBJECT_STATES",
    "RESEARCH_RELATION_TYPES",
    "RESEARCH_SEMANTIC_PROFILE_SCHEMA",
    "ResearchObjectError",
    "build_research_object",
    "research_payload_issues",
    "research_profile_status",
    "validate_research_payload",
    "validate_research_object",
]
