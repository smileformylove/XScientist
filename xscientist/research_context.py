"""Immutable decision-context and scientific-memory snapshots for Research VCS."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_scientist.protocol.hashing import content_hash
from ai_scientist.utils.semantic_memory import (
    SEMANTIC_MEMORY_POLICY_VERSION,
    estimate_text_tokens,
    semantic_overlap,
    truncate_text_to_tokens,
)

from .research_git import (
    ResearchGitError,
    ResearchObjectResult,
    list_research_objects_at_ref,
    show_checkpoint,
)
from .research_vcs import ResearchRepository

RESEARCH_CONTEXT_POLICY_VERSION = "3.0"
RESEARCH_CONTEXT_RECEIPT_PROFILE = "xscientist.context-retrieval-receipt.v3"
RESEARCH_CONTEXT_INTENTS = ("decide", "continue", "write", "audit", "reproduce")

_DECISION_KINDS = {"review", "gate_decision", "agent_evaluation"}
_MEMORY_KINDS = {
    "claim",
    "evidence",
    "experiment_attempt",
    "review",
    "gate_decision",
    "agent_candidate",
    "agent_evaluation",
    "context_snapshot",
    "reproduction",
}
_ACTIVE_NEGATIVE_STATES = {"failed", "timed_out", "cancelled", "rejected"}
_EVIDENCE_KINDS = {"claim", "evidence", "experiment_attempt", "reproduction"}
_ROLE_BASE_SCORES = {
    "target": 120,
    "active_contradiction": 105,
    "verified_evidence": 95,
    "active_evidence": 82,
    "prior_decision": 72,
    "prior_context": 58,
    "lineage": 45,
    "archived_history": 12,
}


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _summary(item: Mapping[str, Any]) -> str:
    payload = item.get("payload") or {}
    for key in (
        "statement",
        "result",
        "summary",
        "decision",
        "status",
        "title",
        "context_hash",
    ):
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if value not in (None, "", [], {}):
            text = " ".join(str(value).split())
            return text[:237] + ("..." if len(text) > 237 else "")
    return f"{item.get('kind')} {item.get('object_id')}"


def _targets(item: Mapping[str, Any]) -> set[str]:
    return {
        str(relation.get("target") or "")
        for relation in item.get("relations") or []
        if str(relation.get("target") or "")
    }


def _relation_targets(item: Mapping[str, Any], relation_types: set[str]) -> set[str]:
    return {
        str(relation.get("target") or "")
        for relation in item.get("relations") or []
        if str(relation.get("type") or "") in relation_types
        and str(relation.get("target") or "")
    }


def _effective_superseded_ids(
    objects: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    superseded = {
        str(object_id)
        for object_id, item in objects.items()
        if item.get("state") == "superseded"
    }
    superseded.update(
        target
        for item in objects.values()
        for target in _relation_targets(item, {"supersedes"})
        if target in objects
    )
    return superseded


def _context_query(
    objects: Mapping[str, Mapping[str, Any]],
    target_ids: Sequence[str],
    *,
    intent: str,
    decision_kind: str,
    selected: str,
    rationale: Sequence[str],
    constraints: Sequence[str],
) -> str:
    return " ".join(
        [
            intent,
            decision_kind,
            selected,
            *rationale,
            *constraints,
            *(_summary(objects[object_id]) for object_id in target_ids),
        ]
    )


def _context_role(
    item: Mapping[str, Any],
    *,
    object_id: str,
    targets: set[str],
    negatives: set[str],
    decisions: set[str],
    prior_contexts: set[str],
    superseded: set[str],
) -> str:
    if object_id in targets:
        return "target"
    if object_id in superseded:
        return "archived_history"
    if object_id in negatives:
        return "active_contradiction"
    if object_id in decisions:
        return "prior_decision"
    if object_id in prior_contexts:
        return "prior_context"
    authority = str((item.get("actor") or {}).get("authority") or "")
    if item.get("state") in {"verified", "promoted"} or authority in {
        "independent_evaluator",
        "deterministic_gate",
    }:
        return "verified_evidence"
    if item.get("kind") in _EVIDENCE_KINDS:
        return "active_evidence"
    return "lineage"


def _fit_source_views(
    *,
    objects: Mapping[str, Mapping[str, Any]],
    candidates: list[dict[str, Any]],
    required_ids: Sequence[str],
    budget_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project ranked audit candidates into a bounded semantic working set."""

    required = [object_id for object_id in required_ids if object_id in objects]
    required_set = set(required)
    by_id = {str(item["object_id"]): item for item in candidates}
    ordered_ids = list(
        dict.fromkeys(required + [str(row["object_id"]) for row in candidates])
    )
    # Leave room for intent, options, applicability constraints, and compact
    # audit hashes used by the prompt view.
    wrapper_reserve = 160
    available = max(32, int(budget_tokens) - wrapper_reserve)
    views: list[dict[str, Any]] = []
    used = 0
    for object_id in ordered_ids:
        candidate = by_id[object_id]
        item = objects[object_id]
        summary_budget = 42 if object_id in required_set else 64
        view = {
            "object_id": object_id,
            "kind": str(item.get("kind") or ""),
            "state": str(item.get("state") or ""),
            "role": str(candidate.get("role") or "lineage"),
            "summary": truncate_text_to_tokens(_summary(item), summary_budget),
        }
        cost = estimate_text_tokens(view)
        if object_id not in required_set and used + cost > available:
            continue
        if object_id in required_set and used + cost > available:
            view["summary"] = truncate_text_to_tokens(_summary(item), 12)
            cost = estimate_text_tokens(view)
        views.append(view)
        used += cost

    selected_ids = {str(item["object_id"]) for item in views}
    missing_required = sorted(required_set - selected_ids)
    working_core = {
        "policy_version": SEMANTIC_MEMORY_POLICY_VERSION,
        "source_view_ids": [str(item["object_id"]) for item in views],
        "required_view_ids": sorted(required_set),
        "missing_required_ids": missing_required,
        "estimated_tokens": used + wrapper_reserve,
        "budget_tokens": int(budget_tokens),
        "decision_usable": (
            not missing_required and used + wrapper_reserve <= int(budget_tokens)
        ),
    }
    return views, {**working_core, "working_set_hash": content_hash(working_core)}


def _research_prompt_visible_payload(
    payload: Mapping[str, Any], *, context_hash_placeholder: bool = False
) -> dict[str, Any]:
    working_set = payload.get("working_set") or {}
    context_hash = payload.get("context_hash")
    if context_hash_placeholder and not context_hash:
        context_hash = "sha256:" + ("0" * 64)
    return {
        "intent": payload.get("intent"),
        "decision_kind": payload.get("decision_kind"),
        "target_ids": payload.get("target_ids") or [],
        "source_views": payload.get("source_views") or [],
        "options_considered": payload.get("options_considered") or [],
        "selected": payload.get("selected") or "",
        "rationale": payload.get("rationale") or [],
        "constraints": payload.get("constraints") or [],
        "working_memory": {
            "decision_usable": working_set.get("decision_usable"),
            "omitted_source_views": (payload.get("omitted") or {}).get(
                "source_views", 0
            ),
        },
        "audit": {
            "source_closure_hash": payload.get("source_closure_hash"),
            "memory_snapshot_hash": payload.get("memory_snapshot_hash"),
            "context_hash": context_hash,
        },
    }


def _research_prompt_token_estimate(payload: Mapping[str, Any]) -> int:
    return estimate_text_tokens(
        "## Research decision context (bounded, source-bound)\n"
        + json.dumps(
            _research_prompt_visible_payload(payload, context_hash_placeholder=True),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    )


def render_research_context_for_prompt(
    payload: Mapping[str, Any], *, allow_incomplete: bool = False
) -> str:
    """Render only the bounded working memory, rejecting unsafe input by default."""

    issues = research_context_issues(payload)
    if issues:
        raise ResearchGitError("invalid research context: " + "; ".join(issues))
    if not allow_incomplete and (
        payload.get("complete") is not True
        or (payload.get("working_set") or {}).get("decision_usable") is not True
    ):
        raise ResearchGitError(
            "research context is incomplete or not decision-usable; inspect JSON instead"
        )

    return "## Research decision context (bounded, source-bound)\n" + json.dumps(
        _research_prompt_visible_payload(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _normalize_options(
    values: Sequence[Mapping[str, Any] | str],
    *,
    selected: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_text = str(selected or "").strip()
    for index, raw in enumerate(values):
        if isinstance(raw, Mapping):
            option = str(
                raw.get("option") or raw.get("name") or raw.get("value") or ""
            ).strip()
            reason = str(raw.get("reason") or raw.get("rejected_because") or "").strip()
        else:
            option = str(raw or "").strip()
            reason = ""
        if not option:
            raise ResearchGitError(f"context option {index} has no name")
        is_selected = option == selected_text
        if selected_text and not is_selected and not reason:
            raise ResearchGitError(
                f"context option {option!r} requires a rejection reason"
            )
        rows.append(
            {
                "option": option,
                "selected": is_selected,
                "status": "selected" if is_selected else "rejected",
                "rejected_because": "" if is_selected else reason,
            }
        )
    if selected_text and selected_text not in {row["option"] for row in rows}:
        raise ResearchGitError("selected context option is not in options_considered")
    return sorted(rows, key=lambda row: (not row["selected"], row["option"]))


def _repository_rows(
    repository: ResearchRepository,
    ref: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if ref and ref not in {"WORKTREE", "worktree"}:
        checkpoint = show_checkpoint(repository.path, ref)
        commit = str(checkpoint["commit"])
        return list_research_objects_at_ref(repository.path, commit), {
            "ref": ref,
            "commit": commit,
            "branch": None,
            "worktree": False,
        }
    status = repository.status()
    return repository.objects(), {
        "ref": "WORKTREE",
        "commit": status.get("head"),
        "branch": status.get("branch"),
        "worktree": True,
    }


def _resolve_at_snapshot(
    selector: str,
    *,
    objects: Mapping[str, Mapping[str, Any]],
) -> str:
    """Resolve the same friendly selectors without consulting the worktree."""

    normalized = str(selector or "").strip()
    if normalized.startswith("@latest"):
        prefix, separator, selected_kind = normalized.partition(":")
        if prefix != "@latest" or not separator or not selected_kind:
            raise ResearchGitError("historical selector must use @latest:<kind>")
        candidates = [
            item for item in objects.values() if item.get("kind") == selected_kind
        ]
        if not candidates:
            raise ResearchGitError(
                f"no research objects found for historical selector: {normalized}"
            )
        return str(
            max(
                candidates,
                key=lambda item: (str(item.get("created_at") or ""), item["object_id"]),
            )["object_id"]
        )
    if re.fullmatch(r"rso-[0-9a-f]{16}", normalized):
        return normalized
    if re.fullmatch(r"rso-[0-9a-f]{6,15}", normalized):
        matches = sorted(
            object_id for object_id in objects if object_id.startswith(normalized)
        )
        if not matches:
            raise ResearchGitError(f"research object not found: {normalized}")
        if len(matches) != 1:
            raise ResearchGitError(f"ambiguous research object prefix: {normalized}")
        return matches[0]
    raise ResearchGitError("invalid research object identifier or selector")


def build_research_context_snapshot(
    repo: ResearchRepository | str | Path,
    *,
    target_ids: Sequence[str],
    intent: str = "decide",
    decision_kind: str = "research_decision",
    selected: str = "",
    options_considered: Sequence[Mapping[str, Any] | str] = (),
    rationale: Sequence[str] = (),
    constraints: Sequence[str] = (),
    memory_refs: Sequence[str] = (),
    ref: str | None = None,
    budget_tokens: int = 4000,
) -> dict[str, Any]:
    """Compile the exact evidence and memory visible to one decision.

    IDs and hashes form the untrimmed hard closure. Only human-readable source
    summaries are budgeted, so a small model context cannot silently erase an
    inconvenient failure, rejected gate, or prior decision.
    """

    repository = (
        repo if isinstance(repo, ResearchRepository) else ResearchRepository(repo)
    )
    normalized_intent = str(intent or "").strip().lower()
    if normalized_intent not in RESEARCH_CONTEXT_INTENTS:
        raise ResearchGitError(f"unsupported research context intent: {intent}")
    if int(budget_tokens) < 128:
        raise ResearchGitError("research context budget must be at least 128 tokens")
    rows, as_of = _repository_rows(repository, ref)
    objects = {str(item["object_id"]): item for item in rows}
    resolved_targets: list[str] = []
    for raw in target_ids:
        selector = str(raw or "").strip()
        if not selector:
            continue
        object_id = (
            repository.resolve(selector)
            if as_of["worktree"]
            else _resolve_at_snapshot(selector, objects=objects)
        )
        if object_id not in objects:
            raise ResearchGitError(
                f"context target is unavailable at {as_of['ref']}: {object_id}"
            )
        resolved_targets.append(object_id)
    resolved_targets = sorted(set(resolved_targets))
    if not resolved_targets:
        raise ResearchGitError("research context requires at least one target object")

    # Close over explicit provenance and related incoming memory in one indexed
    # traversal. Context snapshots are derived views and never recursively
    # expand the scientific source closure.
    incoming: dict[str, list[str]] = {}
    for object_id, item in objects.items():
        for target in _targets(item):
            if target in objects:
                incoming.setdefault(target, []).append(object_id)
    closure = set(resolved_targets)
    queue = deque(resolved_targets)
    while queue:
        current = queue.popleft()
        for target in _targets(objects[current]):
            if target in objects and target not in closure:
                closure.add(target)
                queue.append(target)
        for object_id in incoming.get(current, []):
            if object_id in closure:
                continue
            item = objects[object_id]
            if item.get("kind") == "context_snapshot":
                continue
            is_negative = item.get("state") in _ACTIVE_NEGATIVE_STATES or bool(
                _relation_targets(item, {"refutes", "contradicts"}) & closure
            )
            if item.get("kind") in _MEMORY_KINDS or is_negative:
                closure.add(object_id)
                queue.append(object_id)

    superseded_ids = _effective_superseded_ids(objects) & closure
    related_contexts = [
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "context_snapshot"
        and set((item.get("payload") or {}).get("source_object_ids") or []) & closure
    ]
    latest_prior_context_id = (
        max(
            related_contexts,
            key=lambda object_id: (
                str(objects[object_id].get("created_at") or ""),
                object_id,
            ),
        )
        if related_contexts
        else None
    )
    prior_context_ids = (
        {latest_prior_context_id} if latest_prior_context_id is not None else set()
    )
    negative_ids = {
        object_id
        for object_id in closure
        if object_id not in superseded_ids
        and (
            objects[object_id].get("state") in _ACTIVE_NEGATIVE_STATES
            or bool(_relation_targets(objects[object_id], {"refutes", "contradicts"}))
        )
    }
    prior_decision_ids = {
        object_id
        for object_id in closure
        if objects[object_id].get("kind") in _DECISION_KINDS
        and object_id not in superseded_ids
    }
    memory_object_ids = sorted(negative_ids | prior_decision_ids | prior_context_ids)
    source_object_ids = sorted(closure | prior_context_ids)

    adjacency: dict[str, set[str]] = {object_id: set() for object_id in closure}
    for object_id in closure:
        for target in _targets(objects[object_id]) & closure:
            adjacency[object_id].add(target)
            adjacency[target].add(object_id)
    distances = {object_id: 0 for object_id in resolved_targets}
    distance_queue = deque(resolved_targets)
    while distance_queue:
        current = distance_queue.popleft()
        for related in sorted(adjacency.get(current, set())):
            if related not in distances:
                distances[related] = distances[current] + 1
                distance_queue.append(related)

    query = _context_query(
        objects,
        resolved_targets,
        intent=normalized_intent,
        decision_kind=str(decision_kind or "research_decision"),
        selected=str(selected or ""),
        rationale=rationale,
        constraints=constraints,
    )
    recency_order = sorted(
        source_object_ids,
        key=lambda object_id: (
            str(objects[object_id].get("created_at") or ""),
            object_id,
        ),
    )
    recency_rank = {
        object_id: index for index, object_id in enumerate(recency_order, start=1)
    }

    source_objects = []
    for object_id in source_object_ids:
        item = objects[object_id]
        role = _context_role(
            item,
            object_id=object_id,
            targets=set(resolved_targets),
            negatives=negative_ids,
            decisions=prior_decision_ids,
            prior_contexts=prior_context_ids,
            superseded=superseded_ids,
        )
        source_objects.append(
            {
                "object_id": object_id,
                "kind": str(item["kind"]),
                "state": str(item["state"]),
                "content_hash": str(item["content_hash"]),
                "role": role,
                "frontier_status": (
                    "archived" if object_id in superseded_ids else "active"
                ),
                "created_at": str(item.get("created_at") or ""),
            }
        )
    memory_objects = [
        item for item in source_objects if item["object_id"] in memory_object_ids
    ]
    normalized_memory_refs = sorted({str(value) for value in memory_refs})
    if any(not _is_sha256(value) for value in normalized_memory_refs):
        raise ResearchGitError("memory refs must use sha256:<64 lowercase hex>")

    retrieval_candidates = []
    for source in source_objects:
        object_id = str(source["object_id"])
        item = objects[object_id]
        relevance = semantic_overlap(
            query, {"summary": _summary(item), "payload": item.get("payload") or {}}
        )
        authority = str((item.get("actor") or {}).get("authority") or "")
        authority_score = {
            "independent_evaluator": 12,
            "deterministic_gate": 10,
            "human": 8,
            "research_agent": 3,
            "recorder": 0,
        }.get(authority, 0)
        distance_score = max(0, 14 - (2 * int(distances.get(object_id, 7))))
        recency_score = round(
            10 * recency_rank.get(object_id, 0) / max(1, len(recency_order)), 3
        )
        role = str(source.get("role") or "lineage")
        components = {
            "role": _ROLE_BASE_SCORES.get(role, 0),
            "relation_distance": distance_score,
            "semantic_relevance": round(relevance * 24, 3),
            "recency": recency_score,
            "authority": authority_score,
        }
        retrieval_candidates.append(
            {
                "object_id": object_id,
                "role": role,
                "state": str(source.get("state") or ""),
                "frontier_status": str(source.get("frontier_status") or "active"),
                "score": round(sum(components.values()), 3),
                "score_components": components,
                "score_semantics": "role + graph distance + lexical relevance + recency + authority",
                "selected": False,
                "selection_reason": "ranked audit candidate",
            }
        )
    retrieval_candidates.sort(
        key=lambda row: (-float(row["score"]), str(row["object_id"]))
    )
    for rank, candidate in enumerate(retrieval_candidates, start=1):
        candidate["rank"] = rank

    required_view_ids = list(resolved_targets)
    for role in ("active_contradiction", "verified_evidence", "active_evidence"):
        matching = [
            row
            for row in retrieval_candidates
            if row.get("role") == role
            and str(row.get("object_id")) not in required_view_ids
        ]
        if matching:
            required_view_ids.append(str(matching[0]["object_id"]))
            break
    if normalized_intent == "decide":
        matching_decisions = [
            row
            for row in retrieval_candidates
            if row.get("role") == "prior_decision"
            and str(row.get("object_id")) not in required_view_ids
        ]
        if matching_decisions:
            required_view_ids.append(str(matching_decisions[0]["object_id"]))

    source_views, working_set = _fit_source_views(
        objects=objects,
        candidates=retrieval_candidates,
        required_ids=required_view_ids,
        budget_tokens=int(budget_tokens),
    )
    selected_view_ids = {str(item["object_id"]) for item in source_views}
    for candidate in retrieval_candidates:
        selected_for_view = str(candidate["object_id"]) in selected_view_ids
        candidate["selected"] = selected_for_view
        candidate["selection_reason"] = (
            "selected for bounded semantic working memory"
            if selected_for_view
            else (
                "archived or superseded history retained for audit"
                if candidate.get("frontier_status") == "archived"
                else "working-memory budget exhausted; immutable audit identity retained"
            )
        )
    omitted_count = max(0, len(source_object_ids) - len(source_views))
    retrieval_request = {
        "intent": normalized_intent,
        "decision_kind": str(decision_kind or "research_decision"),
        "target_ids": resolved_targets,
        "as_of": as_of,
        "budget_tokens": int(budget_tokens),
    }
    retrieval_algorithm = {
        "name": "frontier_aware_semantic_relation_closure",
        "version": RESEARCH_CONTEXT_POLICY_VERSION,
        "ranking": "active frontier role + graph distance + lexical relevance + recency + authority",
        "model": None,
        "index": "research_object_relations_at_ref+deterministic_lexical_features",
    }
    transform_lineage = [
        {
            "type": "deterministic_summary_projection",
            "version": "2.0",
            "input_hash": content_hash(source_objects),
            "output_hash": content_hash(source_views),
            "omitted_count": omitted_count,
        }
    ]
    retrieval_receipt_core = {
        "profile": RESEARCH_CONTEXT_RECEIPT_PROFILE,
        "request": retrieval_request,
        "request_hash": content_hash(retrieval_request),
        "algorithm": retrieval_algorithm,
        "algorithm_hash": content_hash(retrieval_algorithm),
        "candidate_set": retrieval_candidates,
        "candidate_set_hash": content_hash(retrieval_candidates),
        "transform_lineage": transform_lineage,
        "complete_candidate_set": True,
    }
    retrieval_receipt = {
        **retrieval_receipt_core,
        "receipt_hash": content_hash(retrieval_receipt_core),
    }
    options = _normalize_options(options_considered, selected=selected)
    policy = {
        "version": RESEARCH_CONTEXT_POLICY_VERSION,
        "rules": [
            "close over explicit Research Object relations",
            "rank the effective frontier before superseded history",
            "retain one current contradiction or decisive evidence in working memory",
            "budget semantic working views only; never trim audit IDs or hashes",
            "link only the latest prior context using a hash chain",
        ],
    }
    source_closure_hash = content_hash({"source_objects": source_objects})
    memory_snapshot_hash = content_hash(
        {"memory_objects": memory_objects, "memory_refs": normalized_memory_refs}
    )
    blockers: list[str] = []
    if selected and len(options) < 2:
        blockers.append("a decision context must retain selected and rejected options")

    if latest_prior_context_id is not None:
        previous_payload = objects[latest_prior_context_id].get("payload") or {}
        previous_chain = previous_payload.get("context_chain") or {}
        previous_chain_hash = str(
            previous_chain.get("chain_hash")
            or content_hash(
                {
                    "context_id": latest_prior_context_id,
                    "context_hash": previous_payload.get("context_hash"),
                }
            )
        )
        context_chain_core = {
            "previous_context_id": latest_prior_context_id,
            "previous_context_hash": previous_payload.get("context_hash"),
            "previous_chain_hash": previous_chain_hash,
            "depth": int(previous_chain.get("depth") or 0) + 1,
        }
    else:
        context_chain_core = {
            "previous_context_id": None,
            "previous_context_hash": None,
            "previous_chain_hash": None,
            "depth": 0,
        }
    context_chain = {
        **context_chain_core,
        "chain_hash": content_hash(context_chain_core),
    }
    base = {
        "protocol_kind": "research_context_snapshot",
        "intent": normalized_intent,
        "decision_kind": str(decision_kind or "research_decision"),
        "target_ids": resolved_targets,
        "source_object_ids": source_object_ids,
        "source_objects": source_objects,
        "source_views": source_views,
        "working_set": working_set,
        "retrieval_receipt": retrieval_receipt,
        "source_refs": sorted(item["content_hash"] for item in source_objects),
        "source_closure_hash": source_closure_hash,
        "memory_object_ids": memory_object_ids,
        "memory_objects": memory_objects,
        "memory_refs": normalized_memory_refs,
        "memory_snapshot_hash": memory_snapshot_hash,
        "negative_knowledge_ids": sorted(negative_ids),
        "prior_decision_ids": sorted(prior_decision_ids),
        "archived_history_ids": sorted(superseded_ids),
        "context_chain": context_chain,
        "options_considered": options,
        "selected": str(selected or ""),
        "rationale": [str(value) for value in rationale if str(value).strip()],
        "constraints": [str(value) for value in constraints if str(value).strip()],
        "selection_policy": policy,
        "selection_policy_hash": content_hash(policy),
        "as_of": as_of,
        "budget": {
            "requested_tokens": int(budget_tokens),
            "hard_closure_preserved": True,
            "working_set_estimated_tokens": working_set["estimated_tokens"],
        },
        "omitted": {"source_views": omitted_count} if omitted_count else {},
        "blockers": blockers,
        "complete": not blockers,
    }
    prompt_estimate = _research_prompt_token_estimate(base)
    working_core = {
        key: value for key, value in working_set.items() if key != "working_set_hash"
    }
    working_core["estimated_tokens"] = prompt_estimate
    working_core["decision_usable"] = not working_core.get(
        "missing_required_ids"
    ) and prompt_estimate <= int(budget_tokens)
    working_set = {
        **working_core,
        "working_set_hash": content_hash(working_core),
    }
    base["working_set"] = working_set
    base["budget"]["working_set_estimated_tokens"] = prompt_estimate
    if working_set["decision_usable"] is not True:
        blocker = "bounded working memory is missing required decision semantics"
        if blocker not in blockers:
            blockers.append(blocker)
    base["blockers"] = blockers
    base["complete"] = not blockers
    return {**base, "context_hash": content_hash(base)}


def research_context_issues(
    payload: Mapping[str, Any],
    *,
    objects: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate internal context/memory identities and optional source objects."""

    issues: list[str] = []
    if payload.get("protocol_kind") != "research_context_snapshot":
        issues.append("context protocol_kind is invalid")
    source_objects = payload.get("source_objects") or []
    memory_objects = payload.get("memory_objects") or []
    memory_refs = payload.get("memory_refs") or []
    source_ids = [str(item.get("object_id") or "") for item in source_objects]
    source_hashes = [str(item.get("content_hash") or "") for item in source_objects]
    memory_ids = [str(item.get("object_id") or "") for item in memory_objects]
    if source_ids != sorted(set(source_ids)):
        issues.append("context source objects are not canonical and unique")
    if payload.get("source_object_ids") != source_ids:
        issues.append("context source object IDs do not match source objects")
    if payload.get("source_refs") != sorted(source_hashes):
        issues.append("context source refs do not match source objects")
    if any(not _is_sha256(value) for value in source_hashes):
        issues.append("context source object has an invalid content hash")
    if memory_ids != sorted(set(memory_ids)):
        issues.append("context memory objects are not canonical and unique")
    if payload.get("memory_object_ids") != memory_ids:
        issues.append("context memory object IDs do not match memory objects")
    source_by_id = {str(item.get("object_id") or ""): item for item in source_objects}
    if any(
        source_by_id.get(object_id) != item
        for object_id, item in zip(memory_ids, memory_objects)
    ):
        issues.append("context memory objects are not an exact source subset")
    if not set(payload.get("target_ids") or []) <= set(source_ids):
        issues.append("context target IDs are not contained in source closure")
    if not set(payload.get("negative_knowledge_ids") or []) <= set(memory_ids):
        issues.append("context negative knowledge is not contained in memory")
    if not set(payload.get("prior_decision_ids") or []) <= set(memory_ids):
        issues.append("context prior decisions are not contained in memory")
    if payload.get("source_closure_hash") != content_hash(
        {"source_objects": source_objects}
    ):
        issues.append("context source closure hash mismatch")
    if payload.get("memory_snapshot_hash") != content_hash(
        {"memory_objects": memory_objects, "memory_refs": memory_refs}
    ):
        issues.append("context memory snapshot hash mismatch")
    if payload.get("selection_policy_hash") != content_hash(
        payload.get("selection_policy") or {}
    ):
        issues.append("context selection policy hash mismatch")
    retrieval_receipt = payload.get("retrieval_receipt")
    if retrieval_receipt is not None:
        if not isinstance(retrieval_receipt, Mapping):
            issues.append("context retrieval receipt must be an object")
        else:
            receipt_core = {
                key: value
                for key, value in retrieval_receipt.items()
                if key != "receipt_hash"
            }
            if retrieval_receipt.get("receipt_hash") != content_hash(receipt_core):
                issues.append("context retrieval receipt hash mismatch")
            request = retrieval_receipt.get("request") or {}
            if retrieval_receipt.get("request_hash") != content_hash(request):
                issues.append("context retrieval request hash mismatch")
            algorithm = retrieval_receipt.get("algorithm") or {}
            if retrieval_receipt.get("algorithm_hash") != content_hash(algorithm):
                issues.append("context retrieval algorithm hash mismatch")
            candidates = retrieval_receipt.get("candidate_set") or []
            if retrieval_receipt.get("candidate_set_hash") != content_hash(candidates):
                issues.append("context retrieval candidate set hash mismatch")
            candidate_ids = sorted(
                str(item.get("object_id") or "")
                for item in candidates
                if isinstance(item, Mapping)
            )
            if candidate_ids != source_ids:
                issues.append(
                    "context retrieval candidates do not cover source closure"
                )
            ranks = [
                item.get("rank") for item in candidates if isinstance(item, Mapping)
            ]
            if any(not isinstance(rank, int) for rank in ranks) or sorted(
                rank for rank in ranks if isinstance(rank, int)
            ) != list(range(1, len(candidates) + 1)):
                issues.append("context retrieval candidate ranks are invalid")
            selected_candidate_ids = {
                str(item.get("object_id") or "")
                for item in candidates
                if isinstance(item, Mapping) and item.get("selected") is True
            }
            view_ids = {
                str(item.get("object_id") or "")
                for item in payload.get("source_views") or []
                if isinstance(item, Mapping)
            }
            if selected_candidate_ids != view_ids:
                issues.append(
                    "context selected retrieval candidates do not match source views"
                )
    policy_version = str((payload.get("selection_policy") or {}).get("version") or "")
    if policy_version == RESEARCH_CONTEXT_POLICY_VERSION:
        if (
            not isinstance(retrieval_receipt, Mapping)
            or retrieval_receipt.get("profile") != RESEARCH_CONTEXT_RECEIPT_PROFILE
        ):
            issues.append("context retrieval receipt profile is invalid")
        else:
            request = retrieval_receipt.get("request") or {}
            if request.get("target_ids") != payload.get("target_ids"):
                issues.append("context retrieval request targets do not match context")
            if request.get("as_of") != payload.get("as_of"):
                issues.append("context retrieval request ref does not match context")
            transforms = retrieval_receipt.get("transform_lineage") or []
            if not transforms or not isinstance(transforms[0], Mapping):
                issues.append("context retrieval transform lineage is missing")
            else:
                transform = transforms[0]
                if transform.get("input_hash") != content_hash(source_objects):
                    issues.append("context retrieval transform input hash mismatch")
                if transform.get("output_hash") != content_hash(
                    payload.get("source_views") or []
                ):
                    issues.append("context retrieval transform output hash mismatch")
        working_set = payload.get("working_set") or {}
        working_core = {
            key: value
            for key, value in working_set.items()
            if key != "working_set_hash"
        }
        if working_set.get("working_set_hash") != content_hash(working_core):
            issues.append("context working-set hash mismatch")
        view_ids = [
            str(item.get("object_id") or "")
            for item in payload.get("source_views") or []
            if isinstance(item, Mapping)
        ]
        if view_ids != list(dict.fromkeys(view_ids)):
            issues.append("context source views are not canonical and unique")
        if not set(view_ids) <= set(source_ids):
            issues.append("context source views are outside source closure")
        if working_set.get("source_view_ids") != view_ids:
            issues.append("context working-set IDs do not match source views")
        required_ids = set(working_set.get("required_view_ids") or [])
        if not required_ids <= set(source_ids):
            issues.append("context required working-set IDs are outside source closure")
        missing_ids = sorted(required_ids - set(view_ids))
        if working_set.get("missing_required_ids") != missing_ids:
            issues.append("context working-set missing IDs are inconsistent")
        prompt_estimate = _research_prompt_token_estimate(payload)
        if working_set.get("estimated_tokens") != prompt_estimate:
            issues.append("context working-set token estimate mismatch")
        usable = not missing_ids and prompt_estimate <= int(
            working_set.get("budget_tokens") or 0
        )
        if working_set.get("decision_usable") is not usable:
            issues.append("context working-set usability verdict is inconsistent")
        if payload.get("complete") is True and not usable:
            issues.append("complete context does not contain usable working memory")
        archived_ids = set(payload.get("archived_history_ids") or [])
        expected_archived_ids = {
            str(item.get("object_id") or "")
            for item in source_objects
            if item.get("frontier_status") == "archived"
        }
        if archived_ids != expected_archived_ids:
            issues.append("context archived history does not match source frontier")
        if archived_ids & set(payload.get("negative_knowledge_ids") or []):
            issues.append(
                "context archived history is labelled as active negative memory"
            )
        chain = payload.get("context_chain") or {}
        chain_core = {key: value for key, value in chain.items() if key != "chain_hash"}
        if chain.get("chain_hash") != content_hash(chain_core):
            issues.append("context chain hash mismatch")
        previous_context_id = chain.get("previous_context_id")
        if previous_context_id is None:
            if not isinstance(chain.get("depth"), int) or chain.get("depth") != 0:
                issues.append("context chain without a predecessor has nonzero depth")
        else:
            previous_source = source_by_id.get(str(previous_context_id))
            if not previous_source or previous_source.get("kind") != "context_snapshot":
                issues.append("context chain predecessor is not in source closure")
            if not _is_sha256(chain.get("previous_context_hash")):
                issues.append("context chain predecessor hash is invalid")
            if not isinstance(chain.get("depth"), int) or int(chain["depth"]) < 1:
                issues.append("context chain depth is invalid")
    identity = {key: value for key, value in payload.items() if key != "context_hash"}
    if payload.get("context_hash") != content_hash(identity):
        issues.append("context hash mismatch")
    if bool(payload.get("complete")) == bool(payload.get("blockers")):
        issues.append("context completeness does not match blockers")
    options = payload.get("options_considered") or []
    selected = str(payload.get("selected") or "")
    if selected and sum(bool(item.get("selected")) for item in options) != 1:
        issues.append("context must contain exactly one selected option")
    if selected and not any(item.get("option") == selected for item in options):
        issues.append("context selected option is unavailable")
    if any(
        not item.get("selected") and not str(item.get("rejected_because") or "").strip()
        for item in options
    ):
        issues.append("context rejected option is missing its reason")
    if any(not _is_sha256(value) for value in payload.get("source_refs") or []):
        issues.append("context has an invalid source ref")
    if any(not _is_sha256(value) for value in memory_refs):
        issues.append("context has an invalid memory ref")
    if objects is not None:
        for source in source_objects:
            object_id = str(source.get("object_id") or "")
            actual = objects.get(object_id)
            if actual is None:
                issues.append(f"context source object is missing: {object_id}")
            elif actual.get("content_hash") != source.get("content_hash"):
                issues.append(f"context source object hash mismatch: {object_id}")
    return sorted(set(issues))


def record_research_context_snapshot(
    repo: ResearchRepository | str | Path,
    *,
    target_ids: Sequence[str],
    intent: str = "decide",
    decision_kind: str = "research_decision",
    selected: str = "",
    options_considered: Sequence[Mapping[str, Any] | str] = (),
    rationale: Sequence[str] = (),
    constraints: Sequence[str] = (),
    memory_refs: Sequence[str] = (),
    budget_tokens: int = 4000,
    actor_id: str = "research-context-recorder",
) -> ResearchObjectResult:
    repository = (
        repo if isinstance(repo, ResearchRepository) else ResearchRepository(repo)
    )
    payload = build_research_context_snapshot(
        repository,
        target_ids=target_ids,
        intent=intent,
        decision_kind=decision_kind,
        selected=selected,
        options_considered=options_considered,
        rationale=rationale,
        constraints=constraints,
        memory_refs=memory_refs,
        ref="WORKTREE",
        budget_tokens=budget_tokens,
    )
    issues = research_context_issues(
        payload,
        objects={str(item["object_id"]): item for item in repository.objects()},
    )
    if issues:
        raise ResearchGitError(
            "invalid research context snapshot: " + "; ".join(issues)
        )
    return repository.record(
        "context_snapshot",
        payload,
        state="completed" if payload["complete"] else "rejected",
        relations=[
            {"type": "depends_on", "target": object_id, "role": "context_source"}
            for object_id in payload["source_object_ids"]
        ],
        actor={"actor_id": actor_id, "authority": "recorder"},
        provenance={
            "context_hashes": [payload["context_hash"]],
            "memory_hashes": [payload["memory_snapshot_hash"]],
            "decision_input_hash": payload["source_closure_hash"],
        },
    )


__all__ = [
    "RESEARCH_CONTEXT_INTENTS",
    "RESEARCH_CONTEXT_POLICY_VERSION",
    "build_research_context_snapshot",
    "record_research_context_snapshot",
    "render_research_context_for_prompt",
    "research_context_issues",
]
