"""Immutable decision-context and scientific-memory snapshots for Research VCS."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_scientist.protocol.hashing import content_hash

from .research_git import (
    ResearchGitError,
    ResearchObjectResult,
    list_research_objects_at_ref,
    show_checkpoint,
)
from .research_vcs import ResearchRepository

RESEARCH_CONTEXT_POLICY_VERSION = "1.0"
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
_NEGATIVE_STATES = {"failed", "timed_out", "cancelled", "rejected", "superseded"}


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
    if int(budget_tokens) <= 0:
        raise ResearchGitError("research context budget must be greater than zero")
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

    # First close over explicit provenance. Then admit decision-bearing incoming
    # objects and negative results that affected the same scientific lineage.
    closure = set(resolved_targets)
    queue = deque(resolved_targets)
    while queue:
        current = queue.popleft()
        for target in _targets(objects[current]):
            if target in objects and target not in closure:
                closure.add(target)
                queue.append(target)
    changed = True
    while changed:
        changed = False
        for object_id, item in objects.items():
            if object_id in closure or item.get("kind") == "context_snapshot":
                continue
            related = bool(_targets(item) & closure)
            is_negative = item.get("state") in _NEGATIVE_STATES or any(
                relation.get("type") in {"refutes", "contradicts"}
                and str(relation.get("target") or "") in closure
                for relation in item.get("relations") or []
            )
            if related and (item.get("kind") in _MEMORY_KINDS or is_negative):
                closure.add(object_id)
                changed = True
        if changed:
            queue = deque(closure)
            while queue:
                current = queue.popleft()
                for target in _targets(objects[current]):
                    if target in objects and target not in closure:
                        closure.add(target)
                        queue.append(target)

    prior_context_ids = {
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "context_snapshot"
        and set((item.get("payload") or {}).get("source_object_ids") or []) & closure
    }
    negative_ids = {
        object_id
        for object_id in closure
        if objects[object_id].get("state") in _NEGATIVE_STATES
        or any(
            relation.get("type") in {"refutes", "contradicts"}
            for relation in objects[object_id].get("relations") or []
        )
    }
    prior_decision_ids = {
        object_id
        for object_id in closure
        if objects[object_id].get("kind") in _DECISION_KINDS
    }
    memory_object_ids = sorted(negative_ids | prior_decision_ids | prior_context_ids)
    source_object_ids = sorted(closure | prior_context_ids)

    source_objects = []
    for object_id in source_object_ids:
        item = objects[object_id]
        role = (
            "target"
            if object_id in resolved_targets
            else (
                "negative_knowledge"
                if object_id in negative_ids
                else (
                    "prior_decision"
                    if object_id in prior_decision_ids
                    else (
                        "prior_context" if object_id in prior_context_ids else "lineage"
                    )
                )
            )
        )
        source_objects.append(
            {
                "object_id": object_id,
                "kind": str(item["kind"]),
                "state": str(item["state"]),
                "content_hash": str(item["content_hash"]),
                "role": role,
            }
        )
    memory_objects = [
        item for item in source_objects if item["object_id"] in memory_object_ids
    ]
    normalized_memory_refs = sorted({str(value) for value in memory_refs})
    if any(not _is_sha256(value) for value in normalized_memory_refs):
        raise ResearchGitError("memory refs must use sha256:<64 lowercase hex>")

    max_views = max(4, int(budget_tokens) // 180)
    ranked_ids = [*resolved_targets, *sorted(negative_ids), *source_object_ids]
    seen: set[str] = set()
    source_views = []
    for object_id in ranked_ids:
        if object_id in seen or len(source_views) >= max_views:
            continue
        seen.add(object_id)
        item = objects[object_id]
        source_views.append(
            {
                "object_id": object_id,
                "kind": item["kind"],
                "state": item["state"],
                "summary": _summary(item),
            }
        )
    omitted_count = max(0, len(source_object_ids) - len(source_views))
    options = _normalize_options(options_considered, selected=selected)
    policy = {
        "version": RESEARCH_CONTEXT_POLICY_VERSION,
        "rules": [
            "close over explicit Research Object relations",
            "include related negative outcomes and prior decisions",
            "budget summaries only; never trim IDs or hashes",
        ],
    }
    source_closure_hash = content_hash({"source_objects": source_objects})
    memory_snapshot_hash = content_hash(
        {"memory_objects": memory_objects, "memory_refs": normalized_memory_refs}
    )
    blockers: list[str] = []
    if selected and len(options) < 2:
        blockers.append("a decision context must retain selected and rejected options")
    base = {
        "protocol_kind": "research_context_snapshot",
        "intent": normalized_intent,
        "decision_kind": str(decision_kind or "research_decision"),
        "target_ids": resolved_targets,
        "source_object_ids": source_object_ids,
        "source_objects": source_objects,
        "source_views": source_views,
        "source_refs": sorted(item["content_hash"] for item in source_objects),
        "source_closure_hash": source_closure_hash,
        "memory_object_ids": memory_object_ids,
        "memory_objects": memory_objects,
        "memory_refs": normalized_memory_refs,
        "memory_snapshot_hash": memory_snapshot_hash,
        "negative_knowledge_ids": sorted(negative_ids),
        "prior_decision_ids": sorted(prior_decision_ids),
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
            "summary_limit": max_views,
        },
        "omitted": {"source_views": omitted_count} if omitted_count else {},
        "blockers": blockers,
        "complete": not blockers,
    }
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
    "research_context_issues",
]
