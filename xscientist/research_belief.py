"""Bounded belief-context projections over immutable Research VCS objects.

The Research VCS remains the source of truth.  This module does not create a
second mutable memory database and it never turns an agent-provided confidence
number into scientific authority.  Instead it derives a compact, deterministic
view of support, challenge, source independence, temporal validity, and missing
evidence for a declared set of target objects.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from jsonschema import validate as validate_json

from ai_scientist.protocol.hashing import content_hash
from ai_scientist.protocol.schemas import load_schema

BELIEF_CONTEXT_POLICY = "xscientist.belief-context-projection.v1"
BELIEF_CONTEXT_SEMANTICS = (
    "deterministic_ordinal_evidence_state_not_calibrated_probability"
)

SUPPORT_RELATIONS = {
    "supports",
    "qualified_supports",
    "replicates",
    "reproduces",
    "depends_on_evidence",
}
CHALLENGE_RELATIONS = {"refutes", "qualified_refutes", "contradicts"}
LINEAGE_RELATIONS = {
    "depends_on",
    "derived_from",
    "quotes",
    "observes",
    "tested_by",
}
INVALIDATION_RELATIONS = {"invalidates", "supersedes"}
CLAIM_EVIDENCE_KINDS = {
    "evidence",
    "passage_evidence",
    "inference",
    "evidence_synthesis",
}

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OBJECT_ID_RE = re.compile(r"^rso-[0-9a-f]{16}$")
_MAX_NODES = 1024
_MAX_RELATIONS = 8192
_MAX_LINEAGE_DEPTH = 8
_INVALID_SOURCE_STATES = {"retracted", "withdrawn", "invalid"}
_TERMINAL_NEGATIVE_STATES = {
    "failed",
    "rejected",
    "refuted",
    "superseded",
}
_INDEPENDENT_AUTHORITIES = {"human", "independent_evaluator"}
_ACTIVE_SIGNAL_STATES = {"completed", "verified", "promoted"}


class BeliefContextError(ValueError):
    """Raised when a belief-context request violates its public contract."""


def _positive_int(value: Any, *, label: str, hard_max: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BeliefContextError(f"{label} must be a positive integer")
    if value > hard_max:
        raise BeliefContextError(f"{label} exceeds the hard maximum {hard_max}")
    return value


def _iso_datetime(value: Any, *, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise BeliefContextError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BeliefContextError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise BeliefContextError(f"{label} must include a timezone")
    return parsed


def _logical_as_of(
    objects: Sequence[Mapping[str, Any]], explicit: str | None
) -> tuple[str | None, str]:
    if explicit is not None:
        return _iso_datetime(explicit, label="as_of").isoformat(), "explicit"
    parsed: list[datetime] = []
    for item in objects:
        value = item.get("created_at")
        if value in (None, ""):
            continue
        try:
            parsed.append(_iso_datetime(value, label="created_at"))
        except BeliefContextError:
            continue
    if not parsed:
        return None, "unavailable"
    return max(parsed).isoformat(), "latest_source_timestamp"


def _canonical_source_identity(item: Mapping[str, Any]) -> str | None:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    kind = str(item.get("kind") or "")
    if kind == "source_snapshot":
        for key in ("doi", "pmid", "arxiv_id", "url", "content_hash"):
            value = str(payload.get(key) or "").strip().lower()
            if value:
                return content_hash({"source_kind": key, "source_identity": value})
    actor = item.get("actor")
    actor = actor if isinstance(actor, Mapping) else {}
    actor_id = str(actor.get("actor_id") or "").strip()
    if actor_id:
        # Authority is a role assertion, not a stable producer identity. Including
        # it here would let one actor manufacture independent corroboration merely
        # by changing its declared authority between evidence objects.
        return content_hash({"source_kind": "producer", "actor_id": actor_id})
    return None


def _source_roots(
    object_id: str,
    *,
    objects: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Sequence[Mapping[str, str]]],
    max_depth: int,
    as_of: datetime | None,
) -> tuple[list[str], bool, bool]:
    """Resolve source-family fingerprints without double-counting descendants."""

    roots: set[str] = set()
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(object_id, 0)])
    truncated = False
    future_lineage_excluded = False
    while queue:
        current_id, depth = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)
        item = objects.get(current_id)
        if item is None:
            continue
        if not _exists_by(item, as_of=as_of):
            future_lineage_excluded = True
            continue
        identity = _canonical_source_identity(item)
        if str(item.get("kind") or "") == "source_snapshot" and identity:
            roots.add(identity)
            continue
        raw_lineage = [
            row["target"]
            for row in relations.get(current_id, ())
            if row["type"] in LINEAGE_RELATIONS and row["target"] in objects
        ]
        lineage = [
            target for target in raw_lineage if _exists_by(objects[target], as_of=as_of)
        ]
        if len(lineage) != len(raw_lineage):
            future_lineage_excluded = True
        if lineage and depth >= max_depth:
            truncated = True
            continue
        for target in sorted(set(lineage)):
            queue.append((target, depth + 1))
        # An explicit but not-yet-observed lineage cannot be replaced with the
        # descendant actor identity; doing so would manufacture an independent
        # source family in a historical projection.
        if not raw_lineage and identity:
            roots.add(identity)
    ordered_roots = sorted(roots)
    if len(ordered_roots) > 128:
        truncated = True
        ordered_roots = ordered_roots[:128]
    return ordered_roots, truncated, future_lineage_excluded


def _valid_until(item: Mapping[str, Any]) -> str | None:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    value = payload.get("valid_until")
    return str(value).strip() if value not in (None, "") else None


def _source_invalidated(
    object_id: str,
    *,
    objects: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Sequence[Mapping[str, str]]],
    invalidated_ids: set[str],
    as_of: datetime | None,
) -> bool:
    if object_id in invalidated_ids:
        return True
    item = objects.get(object_id) or {}
    if str(item.get("state") or "").strip().lower() in _INVALID_SOURCE_STATES:
        return True
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if (
        str(payload.get("retraction_status") or "").strip().lower()
        in _INVALID_SOURCE_STATES
    ):
        return True
    queue = deque([(object_id, 0)])
    visited: set[str] = set()
    while queue:
        current_id, depth = queue.popleft()
        if current_id in visited or depth > _MAX_LINEAGE_DEPTH:
            continue
        visited.add(current_id)
        if current_id in objects and not _exists_by(objects[current_id], as_of=as_of):
            continue
        if current_id in invalidated_ids:
            return True
        current = objects.get(current_id) or {}
        if str(current.get("state") or "").strip().lower() in _INVALID_SOURCE_STATES:
            return True
        current_payload = current.get("payload")
        current_payload = (
            current_payload if isinstance(current_payload, Mapping) else {}
        )
        if (
            str(current_payload.get("retraction_status") or "").strip().lower()
            in _INVALID_SOURCE_STATES
        ):
            return True
        for row in relations.get(current_id, ()):
            if (
                row["type"] in LINEAGE_RELATIONS
                and row["target"] in objects
                and _exists_by(objects[row["target"]], as_of=as_of)
            ):
                queue.append((row["target"], depth + 1))
    return False


def _temporal_state(
    item: Mapping[str, Any],
    *,
    as_of: datetime | None,
) -> tuple[str, str | None]:
    created_at = item.get("created_at")
    if created_at not in (None, ""):
        try:
            observed_at = _iso_datetime(created_at, label="created_at")
        except BeliefContextError:
            return "invalid", None
        if as_of is not None and observed_at > as_of:
            return "not_yet_observed", None
    valid_until = _valid_until(item)
    if valid_until is None:
        return "not_declared", None
    try:
        expires = _iso_datetime(valid_until, label="valid_until")
    except BeliefContextError:
        return "invalid", None
    if as_of is None:
        return "unassessed", expires.isoformat()
    return ("expired" if expires < as_of else "current"), expires.isoformat()


def _exists_by(item: Mapping[str, Any], *, as_of: datetime | None) -> bool:
    """Return whether an object's creation metadata places it in the snapshot."""

    if as_of is None or item.get("created_at") in (None, ""):
        return True
    try:
        return _iso_datetime(item.get("created_at"), label="created_at") <= as_of
    except BeliefContextError:
        return False


def _source_update_time(item: Mapping[str, Any]) -> datetime | None:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    try:
        return _iso_datetime(payload.get("checked_at"), label="checked_at")
    except BeliefContextError:
        return None


def _is_retraction_update(item: Mapping[str, Any]) -> bool:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    return str(
        payload.get("status") or ""
    ).strip().lower() in _INVALID_SOURCE_STATES or str(
        payload.get("update_type") or ""
    ).strip().lower() in {
        "retraction",
        "withdrawal",
    }


def _valid_source_update_supersessions(
    objects: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    as_of: datetime | None,
) -> set[str]:
    """Resolve retraction supersession without letting status checks erase it."""

    valid: set[str] = set()
    for successor_id, successor in objects.items():
        if (
            successor.get("kind") != "source_update"
            or str(successor.get("state") or "").strip().lower()
            not in _ACTIVE_SIGNAL_STATES
            or not _exists_by(successor, as_of=as_of)
        ):
            continue
        payload = successor.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if (
            str(payload.get("update_type") or "").strip().lower() != "reinstatement"
            or str(payload.get("status") or "").strip().lower()
            in _INVALID_SOURCE_STATES
            or not str(payload.get("notice_id") or "").strip()
        ):
            continue
        successor_sources = sorted(
            {
                row["target"]
                for row in relations.get(successor_id, ())
                if row["type"] == "updates"
            }
        )
        if successor_sources != [str(payload.get("source_id") or "")]:
            continue
        successor_time = _source_update_time(successor)
        if successor_time is None:
            continue
        superseded_rows = [
            row
            for row in relations.get(successor_id, ())
            if row["type"] == "supersedes"
        ]
        if len(superseded_rows) != 1:
            continue
        for row in superseded_rows:
            target = objects.get(row["target"])
            if target is None or not _exists_by(target, as_of=as_of):
                continue
            target_payload = target.get("payload")
            target_payload = (
                target_payload if isinstance(target_payload, Mapping) else {}
            )
            target_sources = sorted(
                {
                    target_row["target"]
                    for target_row in relations.get(row["target"], ())
                    if target_row["type"] == "updates"
                }
            )
            target_invalidates = {
                target_row["target"]
                for target_row in relations.get(row["target"], ())
                if target_row["type"] == "invalidates"
            }
            target_time = _source_update_time(target)
            if (
                target.get("kind") == "source_update"
                and str(target.get("state") or "").strip().lower()
                in _ACTIVE_SIGNAL_STATES
                and _is_retraction_update(target)
                and target_sources == successor_sources
                and target_sources
                and target_sources == [str(target_payload.get("source_id") or "")]
                and target_invalidates == set(target_sources)
                and str(payload.get("provider") or "")
                == str(target_payload.get("provider") or "")
                and target_time is not None
                and successor_time > target_time
            ):
                valid.add(row["target"])
    return valid


def _selector_bound(item: Mapping[str, Any]) -> bool:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    return bool(
        str(item.get("kind") or "") == "passage_evidence"
        and _HASH_RE.fullmatch(str(payload.get("selector_hash") or ""))
    )


def _dependency_cycle(
    objects: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Sequence[Mapping[str, str]]],
) -> tuple[bool, int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    observed = 0
    for object_id in sorted(objects):
        for row in relations.get(object_id, ()):
            if row["type"] not in LINEAGE_RELATIONS or row["target"] not in objects:
                continue
            adjacency[object_id].add(row["target"])
            observed += 1
    # Kahn's algorithm avoids recursion depth becoming an input-controlled
    # failure mode at the public node limit.
    indegree = {object_id: 0 for object_id in objects}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    frontier = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited_count = 0
    while frontier:
        node = frontier.popleft()
        visited_count += 1
        for target in sorted(adjacency.get(node, set())):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    cycle_detected = visited_count != len(indegree)
    return cycle_detected, observed


def _signal_row(
    item: Mapping[str, Any],
    relation_type: str,
    *,
    objects: Mapping[str, Mapping[str, Any]],
    relations: Mapping[str, Sequence[Mapping[str, str]]],
    as_of: datetime | None,
    invalidated_ids: set[str],
) -> dict[str, Any]:
    object_id = str(item.get("object_id") or "")
    actor = item.get("actor")
    actor = actor if isinstance(actor, Mapping) else {}
    authority = str(actor.get("authority") or "unknown").strip() or "unknown"
    roots, lineage_truncated, future_lineage_excluded = _source_roots(
        object_id,
        objects=objects,
        relations=relations,
        max_depth=_MAX_LINEAGE_DEPTH,
        as_of=as_of,
    )
    temporal_status, valid_until = _temporal_state(item, as_of=as_of)
    invalidated = _source_invalidated(
        object_id,
        objects=objects,
        relations=relations,
        invalidated_ids=invalidated_ids,
        as_of=as_of,
    )
    # Malformed temporal metadata is not evidence of freshness.  Keep the
    # signal visible for audit, but fail closed so it cannot support a belief.
    state = str(item.get("state") or "").strip().lower()[:128]
    active = (
        temporal_status in {"not_declared", "current"}
        and not invalidated
        and not future_lineage_excluded
        # Only terminal positive lifecycle states may influence the projection.
        # Draft/locked/running and unknown states remain visible but inactive.
        and state in _ACTIVE_SIGNAL_STATES
    )
    return {
        "object_id": object_id,
        "content_hash": str(item.get("content_hash") or ""),
        "relation_type": relation_type,
        "state": state,
        "authority": authority[:128],
        "source_identity_hashes": roots,
        "independence_observed": bool(roots),
        "selector_bound": _selector_bound(item),
        "temporal_status": temporal_status,
        "valid_until": valid_until,
        "invalidated": invalidated,
        "active": active,
        "lineage_truncated": lineage_truncated,
        "lineage_not_observed_as_of": future_lineage_excluded,
    }


def build_belief_context_projection(
    objects: Sequence[Mapping[str, Any]],
    *,
    target_ids: Sequence[str],
    as_of: str | None = None,
    max_nodes: int = 512,
    max_relations: int = 4096,
) -> dict[str, Any]:
    """Build a bounded, read-only epistemic projection for decision context.

    The returned ordinal states are deterministic audit metadata, not calibrated
    probabilities.  ``scientific_promotion_allowed`` is always false because
    Research VCS closure and independent evaluation remain the promotion
    authority.
    """

    node_limit = _positive_int(max_nodes, label="max_nodes", hard_max=_MAX_NODES)
    relation_limit = _positive_int(
        max_relations, label="max_relations", hard_max=_MAX_RELATIONS
    )
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise BeliefContextError("objects must be a sequence")
    if not isinstance(target_ids, Sequence) or isinstance(target_ids, (str, bytes)):
        raise BeliefContextError("target_ids must be a sequence")

    structural_blockers: list[str] = []
    warnings: list[str] = []
    all_rows = [item for item in objects if isinstance(item, Mapping)]
    if len(all_rows) != len(objects):
        structural_blockers.append("non_object_source_row")
    canonical_rows = sorted(
        all_rows,
        key=lambda item: str(item.get("object_id") or ""),
    )
    truncated_nodes = len(canonical_rows) > node_limit
    if truncated_nodes:
        structural_blockers.append("node_limit_exceeded")
        canonical_rows = canonical_rows[:node_limit]
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in canonical_rows:
        object_id = str(item.get("object_id") or "")
        if not _OBJECT_ID_RE.fullmatch(object_id):
            structural_blockers.append("invalid_object_id")
            continue
        if object_id in by_id:
            structural_blockers.append("duplicate_object_id")
            continue
        if not _HASH_RE.fullmatch(str(item.get("content_hash") or "")):
            structural_blockers.append("invalid_content_hash")
            continue
        by_id[object_id] = item

    if len(target_ids) > node_limit:
        raise BeliefContextError("target_ids exceeds max_nodes")
    requested_targets = [str(value or "").strip() for value in target_ids]
    if any(not _OBJECT_ID_RE.fullmatch(value) for value in requested_targets):
        raise BeliefContextError(
            "target_ids must contain canonical Research Object ids"
        )
    normalized_targets = sorted(set(requested_targets))
    if not normalized_targets:
        raise BeliefContextError("target_ids must contain at least one object id")
    missing_targets = sorted(set(normalized_targets) - set(by_id))
    if missing_targets:
        structural_blockers.append("target_outside_observed_graph")

    as_of_text, as_of_source = _logical_as_of(canonical_rows, as_of)
    as_of_datetime = (
        _iso_datetime(as_of_text, label="as_of") if as_of_text is not None else None
    )

    # Normalize the entire observed relation set exactly once.  Counting raw
    # rows before parsing prevents a single object from materializing an
    # unbounded relation list ahead of the public graph budget.
    relation_rows_by_id: dict[str, list[dict[str, str]]] = {
        object_id: [] for object_id in by_id
    }
    observed_relations = 0
    relation_truncated = False
    for object_id in sorted(by_id):
        raw_relations = by_id[object_id].get("relations") or ()
        if not isinstance(raw_relations, Sequence) or isinstance(
            raw_relations, (str, bytes)
        ):
            structural_blockers.append("invalid_relation_collection")
            continue
        for raw in raw_relations:
            if observed_relations >= relation_limit:
                relation_truncated = True
                break
            observed_relations += 1
            if not isinstance(raw, Mapping):
                structural_blockers.append("invalid_relation_row")
                continue
            relation_type = str(raw.get("type") or "").strip()
            target = str(raw.get("target") or "").strip()
            if (
                not relation_type
                or not target
                or len(relation_type) > 128
                or len(target) > 128
            ):
                structural_blockers.append("invalid_relation_row")
                continue
            relation_rows_by_id[object_id].append(
                {"type": relation_type, "target": target}
            )
        if relation_truncated:
            break
    if relation_truncated:
        structural_blockers.append("relation_limit_exceeded")

    valid_source_update_supersessions = _valid_source_update_supersessions(
        by_id,
        relation_rows_by_id,
        as_of=as_of_datetime,
    )
    invalidated_ids = {
        row["target"]
        for object_id in by_id
        for row in relation_rows_by_id[object_id]
        if row["type"] in INVALIDATION_RELATIONS
        and row["target"] in by_id
        and object_id not in valid_source_update_supersessions
        and _exists_by(by_id[object_id], as_of=as_of_datetime)
        and str(by_id[object_id].get("state") or "").strip().lower()
        in _ACTIVE_SIGNAL_STATES
    }
    cycle_detected, lineage_relation_count = _dependency_cycle(
        by_id, relation_rows_by_id
    )
    if cycle_detected:
        structural_blockers.append("lineage_cycle_detected")

    incoming: dict[str, list[tuple[Mapping[str, Any], str]]] = defaultdict(list)
    invalid_endpoint_count = 0
    seen_signal_relations: set[tuple[str, str, str]] = set()
    for object_id in sorted(by_id):
        for row in relation_rows_by_id[object_id]:
            if row["target"] not in by_id:
                invalid_endpoint_count += 1
                continue
            if row["type"] in SUPPORT_RELATIONS | CHALLENGE_RELATIONS:
                relation_key = (object_id, row["type"], row["target"])
                if relation_key not in seen_signal_relations:
                    seen_signal_relations.add(relation_key)
                    incoming[row["target"]].append((by_id[object_id], row["type"]))
            if (
                str(by_id[object_id].get("kind") or "") == "claim"
                and row["type"] == "depends_on"
                and str(by_id[row["target"]].get("kind") or "") in CLAIM_EVIDENCE_KINDS
            ):
                relation_key = (row["target"], "depends_on_evidence", object_id)
                if relation_key not in seen_signal_relations:
                    seen_signal_relations.add(relation_key)
                    incoming[object_id].append(
                        (by_id[row["target"]], "depends_on_evidence")
                    )
    if invalid_endpoint_count:
        structural_blockers.append("relation_endpoint_outside_observed_graph")
    if relation_truncated and "relation_limit_exceeded" not in structural_blockers:
        structural_blockers.append("relation_limit_exceeded")

    assessments: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for target_id in normalized_targets:
        target = by_id.get(target_id)
        if target is None:
            continue
        support_rows: list[dict[str, Any]] = []
        challenge_rows: list[dict[str, Any]] = []
        for source, relation_type in incoming.get(target_id, []):
            signal = _signal_row(
                source,
                relation_type,
                objects=by_id,
                relations=relation_rows_by_id,
                as_of=as_of_datetime,
                invalidated_ids=invalidated_ids,
            )
            (
                support_rows if relation_type in SUPPORT_RELATIONS else challenge_rows
            ).append(signal)
        support_rows.sort(key=lambda row: (row["object_id"], row["relation_type"]))
        challenge_rows.sort(key=lambda row: (row["object_id"], row["relation_type"]))
        active_support = [row for row in support_rows if row["active"]]
        active_challenge = [row for row in challenge_rows if row["active"]]
        support_roots = sorted(
            {
                identity
                for row in active_support
                for identity in row["source_identity_hashes"]
            }
        )
        challenge_roots = sorted(
            {
                identity
                for row in active_challenge
                for identity in row["source_identity_hashes"]
            }
        )
        independent_authority_observed = any(
            row["authority"] in _INDEPENDENT_AUTHORITIES for row in active_support
        )
        target_state = str(target.get("state") or "").strip().lower()[:128]
        target_temporal, target_valid_until = _temporal_state(
            target, as_of=as_of_datetime
        )
        if target_temporal == "not_yet_observed":
            structural_blockers.append("target_not_yet_observed")
        is_superseded = target_id in invalidated_ids or target_state == "superseded"
        if is_superseded:
            belief_state = "superseded"
        elif target_state in _TERMINAL_NEGATIVE_STATES:
            belief_state = "challenged"
        elif active_support and active_challenge:
            belief_state = "contested"
        elif active_challenge:
            belief_state = "challenged"
        elif len(support_roots) >= 2:
            belief_state = "corroborated"
        elif active_support:
            belief_state = "supported"
        elif (
            support_rows
            and not active_support
            and any(
                row["temporal_status"] != "not_yet_observed" for row in support_rows
            )
        ):
            belief_state = "stale"
        elif target_temporal in {"expired", "invalid", "unassessed"}:
            belief_state = "stale"
        else:
            belief_state = "unassessed"

        action_blockers: list[str] = []
        if belief_state in {
            "superseded",
            "contested",
            "challenged",
            "stale",
            "unassessed",
        }:
            action_blockers.append(f"belief_state_{belief_state}")
        if len(support_roots) < 2:
            action_blockers.append("independent_support_below_two")
        if not independent_authority_observed:
            action_blockers.append("independent_evaluator_not_observed")
        if any(row["lineage_truncated"] for row in support_rows + challenge_rows):
            action_blockers.append("source_lineage_truncated")
        if any(
            row.get("lineage_not_observed_as_of")
            for row in support_rows + challenge_rows
        ):
            action_blockers.append("source_lineage_not_observed_as_of")
        if structural_blockers:
            action_blockers.append("projection_incomplete")
        if belief_state == "contested":
            next_action = "investigate_conflict"
        elif belief_state == "stale":
            next_action = "refresh_evidence"
        elif belief_state in {"unassessed", "challenged"}:
            next_action = "collect_discriminating_evidence"
        elif not independent_authority_observed:
            next_action = "seek_independent_review"
        else:
            next_action = "review_with_scientific_gate"

        assessment = {
            "target_id": target_id,
            "target_content_hash": str(target.get("content_hash") or ""),
            "target_state": target_state,
            "belief_state": belief_state,
            "supporting_signals": support_rows,
            "challenging_signals": challenge_rows,
            "active_support_count": len(active_support),
            "active_challenge_count": len(active_challenge),
            "independent_support_source_count": len(support_roots),
            "independent_challenge_source_count": len(challenge_roots),
            "independent_authority_observed": independent_authority_observed,
            "temporal_status": target_temporal,
            "valid_until": target_valid_until,
            "decision_posture": next_action,
            "action_blockers": sorted(set(action_blockers)),
            "scientific_promotion_allowed": False,
        }
        assessments.append(assessment)
        if active_support and active_challenge:
            conflict_core = {
                "target_id": target_id,
                "supporting_ids": sorted({row["object_id"] for row in active_support}),
                "challenging_ids": sorted(
                    {row["object_id"] for row in active_challenge}
                ),
                "status": "unresolved",
            }
            conflicts.append(
                {
                    **conflict_core,
                    "conflict_id": "conflict-"
                    + content_hash(conflict_core).split(":", 1)[1][:16],
                }
            )

    structural_blockers = sorted(set(structural_blockers))
    if structural_blockers:
        for assessment in assessments:
            assessment["action_blockers"] = sorted(
                set(assessment["action_blockers"]) | {"projection_incomplete"}
            )
    if any(item["belief_state"] == "contested" for item in assessments):
        warnings.append("unresolved_target_conflict")
    if any(item["belief_state"] == "stale" for item in assessments):
        warnings.append("stale_target_evidence")
    if any(item["independent_support_source_count"] < 2 for item in assessments):
        warnings.append("limited_independent_support")
    if any(
        row.get("lineage_not_observed_as_of")
        for assessment in assessments
        for row in (
            assessment["supporting_signals"] + assessment["challenging_signals"]
        )
    ):
        warnings.append("future_lineage_excluded")
    core = {
        "policy": BELIEF_CONTEXT_POLICY,
        "confidence_semantics": BELIEF_CONTEXT_SEMANTICS,
        "as_of": as_of_text,
        "as_of_source": as_of_source,
        "target_ids": normalized_targets,
        "source_object_ids": sorted(by_id),
        "source_closure_hash": content_hash(
            [
                {
                    "object_id": object_id,
                    "content_hash": str(by_id[object_id].get("content_hash") or ""),
                }
                for object_id in sorted(by_id)
            ]
        ),
        "target_assessments": assessments,
        "conflict_sets": conflicts,
        "graph_audit": {
            "observed_node_count": len(by_id),
            "input_node_count": len(objects),
            "observed_relation_count": observed_relations,
            "lineage_relation_count": lineage_relation_count,
            "invalid_endpoint_count": invalid_endpoint_count,
            "lineage_cycle_detected": cycle_detected,
            "truncated": truncated_nodes or relation_truncated,
        },
        "limits": {
            "max_nodes": node_limit,
            "max_relations": relation_limit,
            "max_lineage_depth": _MAX_LINEAGE_DEPTH,
        },
        "complete": not structural_blockers,
        "decision_context_usable": not structural_blockers,
        "blockers": structural_blockers,
        "warnings": sorted(set(warnings)),
        "scientific_promotion_allowed": False,
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
    }
    return {**core, "projection_hash": content_hash(core)}


def _visible_projection_issues(payload: Mapping[str, Any]) -> list[str]:
    """Verify derivations that can be recomputed from the public projection.

    The source graph is intentionally not embedded in this compact view, but an
    auditor can still reject internally impossible counts, states, and conflict
    sets even when an attacker has recomputed the outer projection hash.
    """

    issues: set[str] = set()
    source_ids = payload.get("source_object_ids")
    source_id_set = set(source_ids) if isinstance(source_ids, list) else set()
    assessments = payload.get("target_assessments")
    if not isinstance(assessments, list):
        return ["target_assessments_invalid"]
    graph_audit = payload.get("graph_audit")
    limits = payload.get("limits")
    blocker_rows = payload.get("blockers")
    blocker_set = set(blocker_rows) if isinstance(blocker_rows, list) else set()
    if not isinstance(graph_audit, Mapping) or not isinstance(limits, Mapping):
        issues.add("graph_audit_inconsistent")
    else:
        observed_nodes = graph_audit.get("observed_node_count")
        input_nodes = graph_audit.get("input_node_count")
        observed_relations = graph_audit.get("observed_relation_count")
        lineage_relations = graph_audit.get("lineage_relation_count")
        invalid_endpoints = graph_audit.get("invalid_endpoint_count")
        max_nodes = limits.get("max_nodes")
        max_relations = limits.get("max_relations")
        integer_values = (
            observed_nodes,
            input_nodes,
            observed_relations,
            lineage_relations,
            invalid_endpoints,
            max_nodes,
            max_relations,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_values
        ):
            issues.add("graph_audit_inconsistent")
        else:
            if (
                observed_nodes != len(source_id_set)
                or input_nodes < observed_nodes
                or observed_nodes > max_nodes
                or observed_relations > max_relations
                or lineage_relations > observed_relations
                or invalid_endpoints > observed_relations
            ):
                issues.add("graph_audit_inconsistent")
            if payload.get("complete") is True and input_nodes != observed_nodes:
                issues.add("graph_audit_inconsistent")
        cycle_detected = graph_audit.get("lineage_cycle_detected") is True
        truncated = graph_audit.get("truncated") is True
        if cycle_detected != ("lineage_cycle_detected" in blocker_set):
            issues.add("graph_audit_inconsistent")
        if (isinstance(invalid_endpoints, int) and invalid_endpoints > 0) != (
            "relation_endpoint_outside_observed_graph" in blocker_set
        ):
            issues.add("graph_audit_inconsistent")
        limit_blocked = bool(
            {"node_limit_exceeded", "relation_limit_exceeded"} & blocker_set
        )
        if truncated != limit_blocked:
            issues.add("graph_audit_inconsistent")
        if payload.get("complete") is True and (
            cycle_detected or truncated or bool(invalid_endpoints)
        ):
            issues.add("graph_audit_inconsistent")
    try:
        projection_as_of = (
            _iso_datetime(payload.get("as_of"), label="as_of")
            if payload.get("as_of") is not None
            else None
        )
    except BeliefContextError:
        projection_as_of = None

    def temporal_metadata_consistent(row: Mapping[str, Any]) -> bool:
        status = row.get("temporal_status")
        valid_until = row.get("valid_until")
        if status in {"not_declared", "not_yet_observed", "invalid"}:
            return valid_until is None
        if status not in {"current", "expired", "unassessed"}:
            return False
        if valid_until is None:
            return False
        try:
            expires = _iso_datetime(valid_until, label="valid_until")
        except BeliefContextError:
            return False
        if status == "unassessed":
            return projection_as_of is None
        if projection_as_of is None:
            return False
        if status == "expired":
            return expires < projection_as_of
        return expires >= projection_as_of

    expected_conflicts: list[dict[str, Any]] = []
    for assessment in assessments:
        if not isinstance(assessment, Mapping):
            issues.add("target_assessments_invalid")
            continue
        target_id = str(assessment.get("target_id") or "")
        if target_id not in source_id_set:
            issues.add("assessment_target_outside_source_closure")
        signal_groups = (
            ("supporting_signals", SUPPORT_RELATIONS),
            ("challenging_signals", CHALLENGE_RELATIONS),
        )
        normalized_groups: dict[str, list[Mapping[str, Any]]] = {}
        for field, allowed_relations in signal_groups:
            raw_signals = assessment.get(field)
            if not isinstance(raw_signals, list) or any(
                not isinstance(row, Mapping) for row in raw_signals
            ):
                issues.add("assessment_signals_invalid")
                normalized_groups[field] = []
                continue
            signals = [row for row in raw_signals if isinstance(row, Mapping)]
            normalized_groups[field] = signals
            signal_keys = [
                (
                    str(row.get("object_id") or ""),
                    str(row.get("relation_type") or ""),
                )
                for row in signals
            ]
            if len(signal_keys) != len(set(signal_keys)):
                issues.add("duplicate_assessment_signal")
            canonical_order = sorted(
                signals,
                key=lambda row: (
                    str(row.get("object_id") or ""),
                    str(row.get("relation_type") or ""),
                ),
            )
            if signals != canonical_order:
                issues.add("assessment_signals_not_canonical")
            for signal in signals:
                if signal.get("relation_type") not in allowed_relations:
                    issues.add("assessment_signal_relation_inconsistent")
                if str(signal.get("object_id") or "") not in source_id_set:
                    issues.add("assessment_signal_outside_source_closure")
                identities = signal.get("source_identity_hashes")
                identity_rows = identities if isinstance(identities, list) else []
                if identity_rows != sorted(set(identity_rows)):
                    issues.add("signal_source_identities_not_canonical")
                if signal.get("independence_observed") is not bool(identity_rows):
                    issues.add("signal_independence_inconsistent")
                expected_active = (
                    signal.get("state") in _ACTIVE_SIGNAL_STATES
                    and signal.get("temporal_status") in {"not_declared", "current"}
                    and signal.get("invalidated") is False
                    and signal.get("lineage_not_observed_as_of") is not True
                )
                if signal.get("active") is not expected_active:
                    issues.add("signal_activity_inconsistent")
                if not temporal_metadata_consistent(signal):
                    issues.add("temporal_metadata_inconsistent")

        support_rows = normalized_groups.get("supporting_signals", [])
        challenge_rows = normalized_groups.get("challenging_signals", [])
        active_support = [row for row in support_rows if row.get("active") is True]
        active_challenge = [row for row in challenge_rows if row.get("active") is True]
        support_roots = {
            identity
            for row in active_support
            for identity in (
                row.get("source_identity_hashes")
                if isinstance(row.get("source_identity_hashes"), list)
                else []
            )
        }
        challenge_roots = {
            identity
            for row in active_challenge
            for identity in (
                row.get("source_identity_hashes")
                if isinstance(row.get("source_identity_hashes"), list)
                else []
            )
        }
        expected_counts = {
            "active_support_count": len(active_support),
            "active_challenge_count": len(active_challenge),
            "independent_support_source_count": len(support_roots),
            "independent_challenge_source_count": len(challenge_roots),
        }
        if any(
            assessment.get(field) != value for field, value in expected_counts.items()
        ):
            issues.add("assessment_counts_inconsistent")
        independent_authority = any(
            row.get("authority") in _INDEPENDENT_AUTHORITIES for row in active_support
        )
        if (
            assessment.get("independent_authority_observed")
            is not independent_authority
        ):
            issues.add("assessment_authority_inconsistent")

        target_state = str(assessment.get("target_state") or "")
        target_temporal = assessment.get("temporal_status")
        if not temporal_metadata_consistent(assessment):
            issues.add("temporal_metadata_inconsistent")
        if target_state == "superseded":
            expected_belief = "superseded"
        elif target_state in _TERMINAL_NEGATIVE_STATES:
            expected_belief = "challenged"
        elif active_support and active_challenge:
            expected_belief = "contested"
        elif active_challenge:
            expected_belief = "challenged"
        elif len(support_roots) >= 2:
            expected_belief = "corroborated"
        elif active_support:
            expected_belief = "supported"
        elif support_rows and any(
            row.get("temporal_status") != "not_yet_observed" for row in support_rows
        ):
            expected_belief = "stale"
        elif target_temporal in {"expired", "invalid", "unassessed"}:
            expected_belief = "stale"
        else:
            expected_belief = "unassessed"
        observed_belief = assessment.get("belief_state")
        # A completed target can be superseded by an external relation that is
        # not repeated in the compact assessment. Accept that conservative state;
        # every less restrictive state must remain derivable from visible fields.
        if observed_belief != "superseded" and observed_belief != expected_belief:
            issues.add("belief_state_inconsistent")
        if target_state == "superseded" and observed_belief != "superseded":
            issues.add("belief_state_inconsistent")

        if observed_belief == "contested":
            expected_posture = "investigate_conflict"
        elif observed_belief == "stale":
            expected_posture = "refresh_evidence"
        elif observed_belief in {"unassessed", "challenged"}:
            expected_posture = "collect_discriminating_evidence"
        elif not independent_authority:
            expected_posture = "seek_independent_review"
        else:
            expected_posture = "review_with_scientific_gate"
        if assessment.get("decision_posture") != expected_posture:
            issues.add("decision_posture_inconsistent")

        expected_blockers: set[str] = set()
        if observed_belief in {
            "superseded",
            "contested",
            "challenged",
            "stale",
            "unassessed",
        }:
            expected_blockers.add(f"belief_state_{observed_belief}")
        if len(support_roots) < 2:
            expected_blockers.add("independent_support_below_two")
        if not independent_authority:
            expected_blockers.add("independent_evaluator_not_observed")
        if any(
            row.get("lineage_truncated") is True
            for row in support_rows + challenge_rows
        ):
            expected_blockers.add("source_lineage_truncated")
        if any(
            row.get("lineage_not_observed_as_of") is True
            for row in support_rows + challenge_rows
        ):
            expected_blockers.add("source_lineage_not_observed_as_of")
        if payload.get("complete") is not True:
            expected_blockers.add("projection_incomplete")
        if assessment.get("action_blockers") != sorted(expected_blockers):
            issues.add("assessment_blockers_inconsistent")

        if active_support and active_challenge:
            conflict_core = {
                "target_id": target_id,
                "supporting_ids": sorted(
                    {str(row.get("object_id") or "") for row in active_support}
                ),
                "challenging_ids": sorted(
                    {str(row.get("object_id") or "") for row in active_challenge}
                ),
                "status": "unresolved",
            }
            expected_conflicts.append(
                {
                    **conflict_core,
                    "conflict_id": "conflict-"
                    + content_hash(conflict_core).split(":", 1)[1][:16],
                }
            )

    if payload.get("conflict_sets") != expected_conflicts:
        issues.add("conflict_sets_inconsistent")
    expected_warnings: set[str] = set()
    if any(
        isinstance(item, Mapping) and item.get("belief_state") == "contested"
        for item in assessments
    ):
        expected_warnings.add("unresolved_target_conflict")
    if any(
        isinstance(item, Mapping) and item.get("belief_state") == "stale"
        for item in assessments
    ):
        expected_warnings.add("stale_target_evidence")
    if any(
        isinstance(item, Mapping)
        and isinstance(item.get("independent_support_source_count"), int)
        and item.get("independent_support_source_count") < 2
        for item in assessments
    ):
        expected_warnings.add("limited_independent_support")
    if any(
        isinstance(row, Mapping) and row.get("lineage_not_observed_as_of") is True
        for assessment in assessments
        if isinstance(assessment, Mapping)
        for field in ("supporting_signals", "challenging_signals")
        for row in (
            assessment.get(field) if isinstance(assessment.get(field), list) else []
        )
    ):
        expected_warnings.add("future_lineage_excluded")
    if payload.get("warnings") != sorted(expected_warnings):
        issues.add("projection_warnings_inconsistent")
    return sorted(issues)


def belief_context_issues(payload: Mapping[str, Any]) -> list[str]:
    """Return stable, payload-free contract issues for one projection."""

    if not isinstance(payload, Mapping):
        return ["projection_not_object"]
    issues: list[str] = []
    if payload.get("policy") != BELIEF_CONTEXT_POLICY:
        issues.append("policy_invalid")
    if payload.get("confidence_semantics") != BELIEF_CONTEXT_SEMANTICS:
        issues.append("confidence_semantics_invalid")
    if payload.get("scientific_promotion_allowed") is not False:
        issues.append("promotion_authority_escalated")
    if payload.get("quality_claim_allowed") is not False:
        issues.append("quality_claim_escalated")
    if payload.get("causal_claim_allowed") is not False:
        issues.append("causal_claim_escalated")
    as_of = payload.get("as_of")
    as_of_source = payload.get("as_of_source")
    if as_of is not None:
        try:
            _iso_datetime(as_of, label="as_of")
        except BeliefContextError:
            issues.append("as_of_invalid")
    if (as_of_source == "unavailable") != (as_of is None):
        issues.append("as_of_source_inconsistent")
    if as_of_source in {"explicit", "latest_source_timestamp"} and as_of is None:
        issues.append("as_of_source_inconsistent")
    try:
        validate_json(payload, load_schema("belief_context"))
    except Exception:
        issues.append("schema_invalid")
    target_ids = payload.get("target_ids")
    if not isinstance(target_ids, list) or target_ids != sorted(set(target_ids)):
        issues.append("target_ids_not_canonical")
    source_ids = payload.get("source_object_ids")
    if not isinstance(source_ids, list) or source_ids != sorted(set(source_ids)):
        issues.append("source_object_ids_not_canonical")
    assessments = payload.get("target_assessments")
    if not isinstance(assessments, list):
        issues.append("target_assessments_invalid")
    elif any(not isinstance(item, Mapping) for item in assessments):
        issues.append("target_assessments_invalid")
    else:
        assessed_ids = [str(item.get("target_id") or "") for item in assessments]
        expected_targets = target_ids or []
        assessment_ids_valid = assessed_ids == sorted(set(assessed_ids)) and set(
            assessed_ids
        ) <= set(expected_targets)
        if payload.get("complete") is True:
            assessment_ids_valid = assessment_ids_valid and (
                assessed_ids == expected_targets
            )
        if not assessment_ids_valid:
            issues.append("target_assessments_not_canonical")
    if bool(payload.get("complete")) == bool(payload.get("blockers")):
        issues.append("completeness_inconsistent")
    if payload.get("decision_context_usable") is not payload.get("complete"):
        issues.append("decision_context_usability_inconsistent")
    try:
        issues.extend(_visible_projection_issues(payload))
    except (TypeError, ValueError, OverflowError, RecursionError, MemoryError):
        issues.append("projection_derivation_not_verifiable")
    core = {key: value for key, value in payload.items() if key != "projection_hash"}
    try:
        expected_hash = content_hash(core)
    except (TypeError, ValueError, OverflowError, RecursionError):
        issues.append("projection_not_hashable")
    else:
        if payload.get("projection_hash") != expected_hash:
            issues.append("projection_hash_mismatch")
    return sorted(set(issues))


def audit_belief_context_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed without returning source statements or raw evidence payloads."""

    try:
        issues = belief_context_issues(payload)
        projection_complete = (
            isinstance(payload, Mapping) and payload.get("complete") is True
        )
        decision_context_usable = (
            isinstance(payload, Mapping)
            and payload.get("decision_context_usable") is True
        )
        if not projection_complete:
            issues.append("projection_incomplete")
        if not decision_context_usable:
            issues.append("decision_context_not_usable")
        issues = sorted(set(issues))
        projection_hash = (
            str(payload.get("projection_hash") or "")
            if isinstance(payload, Mapping)
            else ""
        )
    except (TypeError, ValueError, OverflowError, RecursionError, MemoryError):
        issues = ["projection_not_verifiable"]
        projection_hash = ""
        projection_complete = False
        decision_context_usable = False
    core = {
        "policy": "xscientist.belief-context-audit.v1",
        "projection_hash": (
            projection_hash if _HASH_RE.fullmatch(projection_hash) else None
        ),
        "verification_allowed": not issues,
        "projection_complete": projection_complete,
        "decision_context_usable": decision_context_usable,
        "issues": issues[:128],
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "scientific_promotion_allowed": False,
        "payloads_disclosed": False,
    }
    return {**core, "audit_hash": content_hash(core)}


__all__ = [
    "BELIEF_CONTEXT_POLICY",
    "BELIEF_CONTEXT_SEMANTICS",
    "BeliefContextError",
    "audit_belief_context_projection",
    "belief_context_issues",
    "build_belief_context_projection",
]
