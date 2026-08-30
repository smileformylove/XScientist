"""Shared authority and independence checks for Research VCS.

An ``independent_evaluator`` label is not evidence of independence by itself.
This module derives the producer set from the immutable Research DAG and emits
the deterministic receipt that explains why an evaluator was accepted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.principal_identity import canonical_principal

from .research_git import ResearchGitError
from .research_vcs import ResearchRepository

INDEPENDENCE_POLICY = "xscientist.provenance-actor-disjoint.v1"
PRODUCER_LINEAGE_RELATIONS = frozenset(
    {
        "consumes",
        "depends_on",
        "derived_from",
        "generated_by",
        "produces",
        "reproduces",
        "retrieves",
        "uses_context",
        "uses_method",
    }
)


def _actor_id(item: dict[str, Any]) -> str:
    return " ".join(str((item.get("actor") or {}).get("actor_id") or "").split())


def _payload_producer_id(item: dict[str, Any]) -> str:
    payload = item.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    return " ".join(str(payload.get("producer_id") or "").split())


def producer_actor_ids(
    repository: ResearchRepository,
    target_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return producer actors and traversed objects for a provenance closure."""

    pending = [str(value) for value in target_ids]
    visited: set[str] = set()
    actors: set[str] = set()
    while pending:
        selector = pending.pop()
        item = repository.get(selector)
        object_id = str(item["object_id"])
        if object_id in visited:
            continue
        visited.add(object_id)
        actor = _actor_id(item)
        payload_producer = _payload_producer_id(item)
        if actor and payload_producer:
            try:
                same_principal = canonical_principal(
                    actor, label="research object actor"
                ) == canonical_principal(
                    payload_producer, label="research object producer_id"
                )
            except ValueError as exc:
                raise ResearchGitError(
                    "research object producer identity is invalid"
                ) from exc
            if not same_principal:
                raise ResearchGitError(
                    "research object actor does not match its payload producer_id"
                )
        if actor:
            actors.add(actor)
        if payload_producer:
            actors.add(payload_producer)
        for relation in item.get("relations") or []:
            if relation.get("type") in PRODUCER_LINEAGE_RELATIONS:
                target = str(relation.get("target") or "")
                if target and target not in visited:
                    pending.append(target)
    return sorted(actors), sorted(visited)


def require_independent_evaluator(
    repository: ResearchRepository,
    *,
    evaluator_id: str,
    target_ids: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Fail closed on a provenance conflict and return an auditable receipt."""

    evaluator = " ".join(str(evaluator_id or "").split())
    try:
        evaluator_principal = canonical_principal(
            evaluator, label=f"{label} evaluator id"
        )
    except ValueError as exc:
        raise ResearchGitError(f"{label} requires a valid evaluator id") from exc
    if not target_ids:
        raise ResearchGitError(f"{label} requires at least one evaluated object")
    resolved_target_ids = sorted(
        {str(repository.get(value)["object_id"]) for value in target_ids}
    )
    producer_ids, lineage_ids = producer_actor_ids(repository, resolved_target_ids)
    try:
        producer_principals = sorted(
            {
                canonical_principal(value, label=f"{label} producer id")
                for value in producer_ids
            }
        )
    except ValueError as exc:
        raise ResearchGitError(
            f"{label} producer provenance contains an invalid identity"
        ) from exc
    if evaluator_principal in producer_principals:
        raise ResearchGitError(
            f"{label} requires an evaluator independent of the complete producer "
            "provenance"
        )
    core = {
        "policy": INDEPENDENCE_POLICY,
        "assurance": "declared_actor_disjointness",
        "identity_verified": False,
        "evaluator_id": evaluator,
        "evaluator_principal": evaluator_principal,
        "target_ids": resolved_target_ids,
        "producer_actor_ids": producer_ids,
        "producer_principals": producer_principals,
        "lineage_object_ids": lineage_ids,
    }
    return {**core, "receipt_hash": canonical_content_hash(core)}


__all__ = [
    "INDEPENDENCE_POLICY",
    "PRODUCER_LINEAGE_RELATIONS",
    "producer_actor_ids",
    "require_independent_evaluator",
]
