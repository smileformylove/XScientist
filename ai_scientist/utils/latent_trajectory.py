"""Bounded, read-only latent dynamics and belief-state projections.

The event ledger is authoritative for admitted events, but it does not imply
that a latent state exists.  This module only projects explicitly declared
``attributes.latent_state``, ``belief_state``, ``action``, ``observation`` and
``transition`` fields.  It never infers a latent state from a plan, metric,
text excerpt, or DAG edge, and every returned projection is presentation-only.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from jsonschema import validate as validate_json

from ai_scientist.protocol.hashing import content_hash
from ai_scientist.protocol.schemas import load_schema

LATENT_TRAJECTORY_SCHEMA_VERSION = "xscientist.latent-trajectory.v1"
LATENT_TRAJECTORY_POLICY = "xscientist.latent-dynamics-presentation.v1"
LATENT_TRAJECTORY_AUDIT_POLICY = "xscientist.latent-trajectory-audit.v1"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_STEPS = 512
_MAX_TEXT = 240
_COMPONENTS = ("latent_state", "belief_state", "action", "observation", "transition")


def _text(value: Any, *, max_length: int = _MAX_TEXT) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _id(value: Any) -> str:
    return value if isinstance(value, str) and 0 < len(value) <= 256 else ""


def _hash_or_redacted(value: Any) -> str | None:
    text = _text(value, max_length=80)
    if text == "redacted" or _HASH_RE.fullmatch(text):
        return text
    return None


def _subject(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {"type": "", "id": ""}
    return {"type": _text(value.get("type")), "id": _id(value.get("id"))}


def _latent_state(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    state_id = _id(value.get("state_id"))
    if not state_id:
        return None
    result: dict[str, Any] = {"state_id": state_id}
    ref = _hash_or_redacted(value.get("representation_ref"))
    if ref:
        result["representation_ref"] = ref
    for key in ("encoder_id", "space", "summary"):
        text = _text(value.get(key))
        if text:
            result[key] = text
    dimension = value.get("dimension")
    if (
        isinstance(dimension, int)
        and not isinstance(dimension, bool)
        and 0 <= dimension <= 1_000_000_000
    ):
        result["dimension"] = dimension
    return result


def _belief_state(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    belief_id = _id(value.get("belief_id"))
    semantics = _text(value.get("semantics"), max_length=32)
    if not belief_id or semantics not in {
        "producer_declared",
        "ordinal",
        "distribution",
    }:
        return None
    result: dict[str, Any] = {"belief_id": belief_id, "semantics": semantics}
    ref = _hash_or_redacted(value.get("support_ref"))
    if ref:
        result["support_ref"] = ref
    summary = _text(value.get("summary"))
    if summary:
        result["summary"] = summary
    return result


def _action(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    action_id = _id(value.get("action_id"))
    if not action_id:
        return None
    result: dict[str, Any] = {"action_id": action_id}
    label = _text(value.get("label"))
    if label:
        result["label"] = label
    ref = _hash_or_redacted(value.get("policy_ref"))
    if ref:
        result["policy_ref"] = ref
    return result


def _observation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    ref = _hash_or_redacted(value.get("observation_ref"))
    if not ref:
        return None
    result: dict[str, Any] = {"observation_ref": ref}
    for key in ("kind", "summary"):
        text = _text(value.get(key))
        if text:
            result[key] = text
    return result


def _transition(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    from_state = _id(value.get("from_state_id"))
    to_state = _id(value.get("to_state_id"))
    if not from_state or not to_state:
        return None
    result: dict[str, Any] = {"from_state_id": from_state, "to_state_id": to_state}
    operator = _text(value.get("operator"))
    if operator:
        result["operator"] = operator
    ref = _hash_or_redacted(value.get("delta_ref"))
    if ref:
        result["delta_ref"] = ref
    return result


def _project_step(
    event: Mapping[str, Any], sequence: int
) -> tuple[dict[str, Any] | None, int]:
    attributes = event.get("attributes")
    if not isinstance(attributes, Mapping):
        return None, 0
    builders = {
        "latent_state": _latent_state,
        "belief_state": _belief_state,
        "action": _action,
        "observation": _observation,
        "transition": _transition,
    }
    components: list[str] = []
    values: dict[str, Any] = {}
    invalid = 0
    for key, builder in builders.items():
        if key not in attributes:
            continue
        value = builder(attributes.get(key))
        if value is None:
            invalid += 1
            continue
        components.append(key)
        values[key] = value
    if not components:
        return None, invalid
    event_id = _id(event.get("event_id"))
    event_type = _text(event.get("event_type"))
    subject = _subject(event.get("subject"))
    if not event_id or not event_type or not subject["type"] or not subject["id"]:
        return None, invalid + 1
    step = {
        "sequence": sequence,
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": _text(event.get("timestamp")),
        "subject": subject,
        "components": components,
    }
    step.update({key: values.get(key) for key in _COMPONENTS})
    return step, invalid


def build_latent_trajectory_projection(
    events: Iterable[Mapping[str, Any]] | None,
    *,
    max_steps: int = _MAX_STEPS,
) -> dict[str, Any]:
    """Build a deterministic, bounded projection from explicit event fields."""

    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps < 1
        or max_steps > _MAX_STEPS
    ):
        raise ValueError(f"max_steps must be between 1 and {_MAX_STEPS}")
    source_count = 0
    window: deque[Any] = deque(maxlen=max_steps)
    # Consume the source exactly once without retaining an unbounded ledger.
    for event in (() if events is None else events):
        source_count += 1
        window.append(event)
    window_truncated = source_count > max_steps
    steps: list[dict[str, Any]] = []
    ignored = 0
    invalid_components = 0
    seen_event_ids: set[str] = set()
    duplicate_event_count = 0
    for event in window:
        if not isinstance(event, Mapping):
            ignored += 1
            continue
        step, invalid = _project_step(event, len(steps))
        invalid_components += invalid
        if step is None:
            ignored += 1
            continue
        if step["event_id"] in seen_event_ids:
            duplicate_event_count += 1
            ignored += 1
            continue
        seen_event_ids.add(step["event_id"])
        steps.append(step)
    status, blockers, warnings = _diagnostics(
        steps, ignored, invalid_components, duplicate_event_count, window_truncated
    )
    core = {
        "schema_version": LATENT_TRAJECTORY_SCHEMA_VERSION,
        "policy": LATENT_TRAJECTORY_POLICY,
        "source": {
            "kind": "events/research_events.jsonl",
            "event_count": source_count,
            "window_event_count": len(window),
            "max_events": max_steps,
            "observed_event_count": len(steps),
            "ignored_event_count": ignored,
            "ledger_window_truncated": window_truncated,
        },
        "status": status,
        "steps": steps,
        "integrity": {
            "ordered": True,
            "duplicate_event_count": duplicate_event_count,
            "invalid_component_count": invalid_components,
        },
        "complete": False,
        "presentation_only": True,
        "scientific_promotion_allowed": False,
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
    }
    return {**core, "projection_hash": content_hash(core)}


def _diagnostics(
    steps: list[dict[str, Any]],
    ignored: int,
    invalid: int,
    duplicates: int,
    truncated: bool,
) -> tuple[str, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    stateful = sum(
        bool(step.get("latent_state") or step.get("belief_state")) for step in steps
    )
    if invalid:
        warnings.append("invalid_explicit_component_ignored")
    if ignored:
        warnings.append("non_trajectory_events_ignored")
    if truncated:
        warnings.append("ledger_window_truncated")
    if duplicates:
        warnings.append("duplicate_event_ignored")
    if stateful != len(steps):
        warnings.append("trajectory_step_without_latent_or_belief")
    if not stateful:
        blockers.append("latent_state_not_declared")
    status = (
        "unavailable"
        if not steps
        else ("partial" if blockers or warnings else "observed")
    )
    if status != "observed":
        blockers.append("trajectory_is_not_complete")
    return status, sorted(blockers), sorted(warnings)


def latent_trajectory_issues(payload: Mapping[str, Any]) -> list[str]:
    """Return stable, payload-free issues for a trajectory projection."""

    if not isinstance(payload, Mapping):
        return ["projection_not_object"]
    issues: list[str] = []
    try:
        validate_json(payload, load_schema("latent_trajectory"))
    except Exception:
        issues.append("schema_invalid")
    if payload.get("presentation_only") is not True:
        issues.append("presentation_authority_invalid")
    if any(
        payload.get(key) is not False
        for key in (
            "complete",
            "scientific_promotion_allowed",
            "quality_claim_allowed",
            "causal_claim_allowed",
        )
    ):
        issues.append("promotion_authority_escalated")
    steps = payload.get("steps")
    if not isinstance(steps, list) or [
        item.get("sequence") for item in steps if isinstance(item, Mapping)
    ] != list(range(len(steps))):
        issues.append("sequence_not_canonical")
    if "schema_invalid" not in issues:
        source = payload["source"]
        integrity = payload["integrity"]
        window_count = min(source["event_count"], source["max_events"])
        truncated = source["event_count"] > source["max_events"]
        if (
            source["window_event_count"] != window_count
            or source["observed_event_count"] != len(steps)
            or source["ignored_event_count"] + len(steps) != window_count
            or source["ledger_window_truncated"] is not truncated
        ):
            issues.append("source_counts_inconsistent")
        if (
            integrity["ordered"] is not True
            or integrity["duplicate_event_count"] > source["ignored_event_count"]
            or integrity["invalid_component_count"] > window_count * 6
        ):
            issues.append("integrity_counts_inconsistent")
        event_ids = [step["event_id"] for step in steps]
        if len(set(event_ids)) != len(event_ids):
            issues.append("duplicate_projected_event")
        for step in steps:
            if step["components"] != [
                key for key in _COMPONENTS if step.get(key) is not None
            ]:
                issues.append("component_presence_inconsistent")
        status, blockers, warnings = _diagnostics(
            steps,
            source["ignored_event_count"],
            integrity["invalid_component_count"],
            integrity["duplicate_event_count"],
            truncated,
        )
        if payload["status"] != status:
            issues.append("status_inconsistent")
        if payload["blockers"] != blockers or payload["warnings"] != warnings:
            issues.append("diagnostics_inconsistent")
    core = {key: value for key, value in payload.items() if key != "projection_hash"}
    try:
        if payload.get("projection_hash") != content_hash(core):
            issues.append("projection_hash_mismatch")
    except Exception:
        issues.append("projection_not_hashable")
    return sorted(set(issues))


def audit_latent_trajectory_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed and disclose that this is not scientific authority."""

    try:
        issues = latent_trajectory_issues(payload)
        contract_valid = not issues
        status = (
            str(payload.get("status") or "") if isinstance(payload, Mapping) else ""
        )
        if status == "unavailable":
            issues.append("trajectory_unavailable")
        if status != "observed":
            issues.append("trajectory_incomplete")
        projection_hash = str(payload.get("projection_hash") or "")
    except Exception:
        issues = ["projection_not_verifiable"]
        contract_valid = False
        projection_hash = ""
    core = {
        "policy": LATENT_TRAJECTORY_AUDIT_POLICY,
        "projection_hash": (
            projection_hash if _HASH_RE.fullmatch(projection_hash) else None
        ),
        "presentation_contract_valid": contract_valid,
        "verification_allowed": False,
        "issues": sorted(set(issues))[:64],
        "presentation_only": True,
        "scientific_promotion_allowed": False,
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
    }
    return {**core, "audit_hash": content_hash(core)}


__all__ = [
    "LATENT_TRAJECTORY_AUDIT_POLICY",
    "LATENT_TRAJECTORY_POLICY",
    "LATENT_TRAJECTORY_SCHEMA_VERSION",
    "audit_latent_trajectory_projection",
    "build_latent_trajectory_projection",
    "latent_trajectory_issues",
]
