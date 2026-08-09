"""Deterministic policy for Research VCS checkpoints, forks, and lineage views."""

from __future__ import annotations

import re
import shlex
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.protocol.hashing import content_hash

from .research_git import (
    list_research_branches,
    list_research_objects,
    list_research_objects_at_ref,
    preview_research_merge,
    repository_status,
)

POLICY_VERSION = "1.0"
RESEARCH_EVENTS = (
    "observation",
    "hypothesis",
    "preregistration",
    "experiment-started",
    "experiment-completed",
    "experiment-failed",
    "evidence",
    "review",
    "gate",
    "manuscript",
    "release",
    "method-change",
    "contradiction",
    "replication",
    "agent-candidate",
    "merge-candidate",
)

_CHECKPOINT_EVENTS = {
    "hypothesis",
    "preregistration",
    "experiment-started",
    "experiment-completed",
    "experiment-failed",
    "evidence",
    "review",
    "gate",
    "manuscript",
    "release",
}
_FORK_EVENT_PREFIX = {
    "hypothesis": "hypothesis",
    "method-change": "method",
    "contradiction": "interpretation",
    "replication": "replication",
    "agent-candidate": "evolve",
}
_LANDMARK_STATES = {"completed", "verified", "promoted"}


def _slug(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return (normalized or fallback)[:48]


def _checkpoint_action(event: str) -> dict[str, Any]:
    stage = {
        "hypothesis": "ideation",
        "experiment-started": "experiment",
        "experiment-completed": "experiment",
        "experiment-failed": "experiment",
        "manuscript": "paper",
        "agent-candidate": "evolve",
    }.get(event, event)
    subject = f"record {event.replace('-', ' ')}"
    return {
        "action": "checkpoint",
        "reason": "material research state changed and must remain reproducible",
        "commands": [
            "xscientist research stage --all",
            "xscientist research commit "
            f"--stage {shlex.quote(stage)} -m {shlex.quote(subject)}",
        ],
    }


def decide_research_transition(
    repo: str | Path,
    *,
    event: str,
    name: str = "",
    state: str = "",
    source_branch: str | None = None,
    competing_hypothesis: bool = False,
    contradictory_evidence: bool = False,
    protocol_change: bool = False,
    independent_replication: bool = False,
) -> dict[str, Any]:
    """Return an explainable, read-only version-control decision.

    The policy never mutates a repository.  It emits explicit commands so an
    agent or human can inspect the reasons before applying a transition.
    """

    normalized_event = str(event or "").strip().lower()
    if normalized_event not in RESEARCH_EVENTS:
        raise ValueError(f"unsupported research event: {event!r}")
    status = repository_status(repo)
    dirty_paths = sorted(
        set(status.get("eligible_changes") or [])
        | set((status.get("research_stage") or {}).get("paths") or [])
    )
    signals = {
        "competing_hypothesis": bool(competing_hypothesis),
        "contradictory_evidence": bool(contradictory_evidence),
        "protocol_change": bool(protocol_change),
        "independent_replication": bool(independent_replication),
    }
    fork_requested = (
        normalized_event
        in {"method-change", "contradiction", "replication", "agent-candidate"}
        or competing_hypothesis
        or contradictory_evidence
        or protocol_change
        or independent_replication
    )
    actions: list[dict[str, Any]] = []
    rationale: list[str] = []
    merge_preview: dict[str, Any] | None = None

    if normalized_event == "merge-candidate":
        if not source_branch:
            actions.append(
                {
                    "action": "hold",
                    "reason": "a merge decision requires --source-branch",
                    "commands": [],
                }
            )
        elif dirty_paths:
            actions.append(
                {
                    "action": "hold",
                    "reason": "merge preflight requires a clean research working state",
                    "commands": [
                        "xscientist research status",
                        "xscientist research stage --all",
                    ],
                }
            )
        else:
            merge_preview = preview_research_merge(repo, source_branch)
            if merge_preview["clean"]:
                actions.append(
                    {
                        "action": "merge",
                        "reason": "backend and scientific conflict preflight are clean",
                        "commands": [
                            "xscientist research merge " + shlex.quote(source_branch)
                        ],
                    }
                )
            else:
                conflict_types = {
                    str(item.get("type") or "") for item in merge_preview["conflicts"]
                }
                commands = [
                    "xscientist research merge "
                    + shlex.quote(source_branch)
                    + " --preview --json"
                ]
                if conflict_types == {"opposed_evidence"}:
                    commands.append(
                        "xscientist research merge "
                        + shlex.quote(source_branch)
                        + " --preserve-conflicts"
                    )
                actions.append(
                    {
                        "action": "reconcile",
                        "reason": "scientific conflicts must be represented, not overwritten",
                        "commands": commands,
                        "conflict_ids": [
                            item.get("conflict_id")
                            for item in merge_preview["conflicts"]
                        ],
                    }
                )
    else:
        if dirty_paths and (normalized_event in _CHECKPOINT_EVENTS or fork_requested):
            actions.append(_checkpoint_action(normalized_event))
            rationale.append("uncommitted material state requires a durable boundary")

        if fork_requested:
            prefix = _FORK_EVENT_PREFIX.get(normalized_event)
            if competing_hypothesis:
                prefix = "hypothesis"
            elif protocol_change:
                prefix = "method"
            elif independent_replication:
                prefix = "replication"
            elif contradictory_evidence:
                prefix = "interpretation"
            prefix = prefix or "exploration"
            branch = f"{prefix}/{_slug(name, fallback=normalized_event)}"
            actions.append(
                {
                    "action": "fork",
                    "reason": (
                        "the new line changes an independent scientific assumption, "
                        "method, interpretation, replication, or agent candidate"
                    ),
                    "branch": branch,
                    "commands": [
                        "xscientist research branch "
                        + shlex.quote(branch)
                        + " --switch"
                    ],
                }
            )
            rationale.append("independent claims must not rewrite the stable line")

        if normalized_event == "release":
            if str(state or "").strip().lower() not in _LANDMARK_STATES:
                actions.append(
                    {
                        "action": "hold",
                        "reason": "release tags require completed, verified, or promoted state",
                        "commands": [],
                    }
                )
            elif not dirty_paths:
                tag = "result/" + _slug(name, fallback="release")
                actions.append(
                    {
                        "action": "tag",
                        "reason": "a verified stable landmark needs an immutable name",
                        "tag": tag,
                        "commands": ["xscientist research tag " + shlex.quote(tag)],
                    }
                )

    if not actions:
        actions.append(
            {
                "action": "none",
                "reason": "no material, divergent, or stable-landmark transition was detected",
                "commands": [],
            }
        )
    context_snapshot: dict[str, Any] | None = None
    context_candidates = [
        item
        for item in list_research_objects(repo)
        if item.get("kind")
        in {
            "hypothesis",
            "research_plan",
            "preregistration",
            "experiment_attempt",
            "evidence",
            "claim",
            "review",
            "gate_decision",
            "agent_candidate",
            "agent_evaluation",
        }
    ]
    if context_candidates:
        from .research_context import build_research_context_snapshot

        chosen = "+".join(str(item["action"]) for item in actions)
        alternative = "hold" if chosen != "hold" else "apply_transition"
        latest_targets = [
            str(item["object_id"])
            for item in sorted(
                context_candidates,
                key=lambda item: (
                    str(item.get("created_at") or ""),
                    str(item["object_id"]),
                ),
                reverse=True,
            )[:8]
        ]
        context_snapshot = build_research_context_snapshot(
            repo,
            target_ids=latest_targets,
            decision_kind="version_control_transition",
            selected=chosen,
            options_considered=[
                {"option": chosen, "rejected_because": ""},
                {
                    "option": alternative,
                    "rejected_because": "deterministic transition policy selected "
                    + chosen,
                },
            ],
            rationale=[
                str(item.get("reason") or "") for item in actions if item.get("reason")
            ],
            constraints=[
                "decision trace is read-only",
                "material divergent state must remain on an explicit research line",
            ],
            ref="WORKTREE",
            budget_tokens=3000,
        )
    decision_base = {
        "policy_version": POLICY_VERSION,
        "event": normalized_event,
        "name": str(name or ""),
        "state": str(state or ""),
        "branch": status.get("branch"),
        "head": status.get("head"),
        "dirty_path_count": len(dirty_paths),
        "signals": signals,
        "actions": actions,
        "merge_preview": merge_preview,
        "context_hash": (
            context_snapshot.get("context_hash") if context_snapshot else None
        ),
    }
    return {
        "schema": "xscientist.research-vcs-decision.v1",
        "decision_id": "rvd-" + content_hash(decision_base).split(":", 1)[-1][:16],
        **decision_base,
        "rationale": rationale,
        "mutates_repository": False,
        "trace_required": True,
        "context": context_snapshot,
        "host_paths_disclosed": False,
    }


def _topological_order(
    node_ids: Iterable[str], edges: list[dict[str, str]]
) -> tuple[list[str], list[str]]:
    nodes = sorted(set(node_ids))
    indegree = {node: 0 for node in nodes}
    outgoing: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        # The referenced target precedes the object that declares the relation.
        before, after = edge["target"], edge["source"]
        if before not in indegree or after not in indegree or after in outgoing[before]:
            continue
        outgoing[before].add(after)
        indegree[after] += 1
    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for child in sorted(outgoing[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready = deque(sorted(ready))
    cycles = sorted(node for node, degree in indegree.items() if degree > 0)
    return order, cycles


def build_research_technology_tree(repo: str | Path) -> dict[str, Any]:
    """Build a payload-free semantic view of accumulated research lines."""

    branches_raw = list_research_branches(repo)
    by_id: dict[str, dict[str, Any]] = {}
    object_lines: dict[str, set[str]] = defaultdict(set)
    for branch in branches_raw:
        branch_name = str(branch["name"])
        for item in list_research_objects_at_ref(repo, str(branch["commit"])):
            object_id = str(item["object_id"])
            by_id[object_id] = item
            object_lines[object_id].add(branch_name)
    current_branch = next(
        (str(item["name"]) for item in branches_raw if item["current"]),
        "",
    )
    for item in list_research_objects(repo):
        object_id = str(item["object_id"])
        by_id[object_id] = item
        if current_branch:
            object_lines[object_id].add(current_branch)
    nodes = [
        {
            "object_id": object_id,
            "kind": str(item["kind"]),
            "state": str(item["state"]),
            "content_hash": str(item["content_hash"]),
            "research_lines": sorted(object_lines[object_id]),
        }
        for object_id, item in sorted(by_id.items())
    ]
    edges = sorted(
        (
            {
                "source": object_id,
                "target": str(relation.get("target") or ""),
                "type": str(relation.get("type") or ""),
                "role": str(relation.get("role") or ""),
            }
            for object_id, item in sorted(by_id.items())
            for relation in item.get("relations") or []
        ),
        key=lambda item: (item["source"], item["target"], item["type"], item["role"]),
    )
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        incoming[edge["target"]].add(edge["type"])
    frontier: list[dict[str, str]] = []
    for node in nodes:
        if node["kind"] not in {"hypothesis", "claim", "agent_candidate"}:
            continue
        relations = incoming[node["object_id"]]
        if node["state"] == "superseded" or "supersedes" in relations:
            classification = "superseded"
        elif {"supports", "refutes"}.issubset(relations) or "contradicts" in relations:
            classification = "contested"
        elif "refutes" in relations:
            classification = "refuted"
        elif node["state"] in {"verified", "promoted"} or "supports" in relations:
            classification = "supported"
        else:
            classification = "open"
        if classification in {"open", "contested"}:
            frontier.append(
                {
                    "object_id": node["object_id"],
                    "kind": node["kind"],
                    "classification": classification,
                }
            )
    order, cycles = _topological_order(by_id, edges)
    missing_targets = sorted(
        {edge["target"] for edge in edges if edge["target"] not in by_id}
    )
    branches = [
        {
            "name": str(item["name"]),
            "current": bool(item["current"]),
            "checkpoint_id": item.get("checkpoint_id"),
        }
        for item in branches_raw
    ]
    return {
        "schema": "xscientist.research-technology-tree.v1",
        "nodes": nodes,
        "edges": edges,
        "topological_order": order,
        "frontier": frontier,
        "research_lines": branches,
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "branches": len(branches),
            "by_kind": dict(sorted(Counter(node["kind"] for node in nodes).items())),
            "by_state": dict(sorted(Counter(node["state"] for node in nodes).items())),
        },
        "integrity": {
            "ok": not missing_targets and not cycles,
            "missing_relation_targets": missing_targets,
            "cycle_nodes": cycles,
        },
        "payloads_disclosed": False,
        "host_paths_disclosed": False,
    }


__all__ = [
    "POLICY_VERSION",
    "RESEARCH_EVENTS",
    "build_research_technology_tree",
    "decide_research_transition",
]
