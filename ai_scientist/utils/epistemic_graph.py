from __future__ import annotations

"""Append-only epistemic graph for cumulative questions, claims, and evidence."""

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.pipeline_contracts import (
    load_contract_artifact,
    save_contract_artifact,
)
from ai_scientist.utils.science_constitution import (
    assert_science_constitution_intact,
)

SCHEMA_VERSION = 1
NODE_TYPES = {
    "question",
    "concept",
    "theory",
    "hypothesis",
    "prediction",
    "protocol",
    "observation",
    "claim",
    "theorem",
    "refutation",
    "replication",
    "artifact",
}
EDGE_TYPES = {
    "addresses",
    "supports",
    "refutes",
    "depends_on",
    "generalizes",
    "specializes",
    "equivalent_to",
    "contradicts",
    "derived_from",
    "replicates",
    "observes",
    "implements",
    "tested_by",
    "formalizes",
    "produces",
}
EPISTEMIC_STATES = {
    "speculative",
    "grounded",
    "preregistered",
    "tested",
    "replicated",
    "robust",
    "canonical",
    "contested",
    "failed",
    "refuted",
    "superseded",
}
ALLOWED_TRANSITIONS = {
    "speculative": {"grounded", "contested", "failed", "refuted"},
    "grounded": {"preregistered", "contested", "failed", "refuted"},
    "preregistered": {"tested", "contested", "failed", "refuted"},
    "tested": {"replicated", "contested", "failed", "refuted"},
    "replicated": {"robust", "contested", "refuted"},
    "robust": {"canonical", "contested", "refuted", "superseded"},
    "canonical": {"contested", "refuted", "superseded"},
    "contested": {"tested", "replicated", "robust", "refuted", "superseded"},
    "failed": set(),
    "refuted": set(),
    "superseded": set(),
}
MINIMUM_EVIDENCE_REFS = {
    "grounded": 1,
    "preregistered": 1,
    "tested": 1,
    "replicated": 1,
    "robust": 2,
    "canonical": 2,
    "contested": 1,
    "failed": 1,
    "refuted": 1,
    "superseded": 1,
}
FALSIFIER_REQUIRED_TYPES = {"theory", "hypothesis", "prediction", "claim"}


class EpistemicGraphError(ValueError):
    """Raised when an epistemic artifact violates append-only rules."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _strings(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(item).strip() for item in (values or []) if str(item).strip()})


def _confidence(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EpistemicGraphError("confidence must be a finite number in [0, 1]")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise EpistemicGraphError("confidence must be a finite number in [0, 1]")
    return round(parsed, 6)


def _node_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"node_id", "content_hash", "created_at"}
    }


def build_epistemic_node(
    *,
    node_type: str,
    title: str,
    statement: str,
    created_by: str,
    parent_ids: Iterable[str] | None = None,
    falsifiers: Iterable[str] | None = None,
    applicability: Iterable[str] | None = None,
    confidence: float = 0.0,
    uncertainty_basis: str = "unassessed",
    initial_status: str = "speculative",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kind = str(node_type or "").strip()
    status = str(initial_status or "").strip()
    if kind not in NODE_TYPES:
        raise EpistemicGraphError(f"unsupported node_type={kind!r}")
    if status not in EPISTEMIC_STATES:
        raise EpistemicGraphError(f"unsupported initial_status={status!r}")
    if status != "speculative":
        raise EpistemicGraphError(
            "new nodes must start speculative; use hash-chained transitions to advance"
        )
    required = {"title": title, "statement": statement, "created_by": created_by}
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise EpistemicGraphError("missing node fields: " + ", ".join(missing))
    falsifier_rows = _strings(falsifiers)
    if kind in FALSIFIER_REQUIRED_TYPES and not falsifier_rows:
        raise EpistemicGraphError(f"{kind} nodes require at least one falsifier")
    core = {
        "schema_version": SCHEMA_VERSION,
        "node_type": kind,
        "title": str(title).strip(),
        "statement": str(statement).strip(),
        "created_by": str(created_by).strip(),
        "parent_ids": _strings(parent_ids),
        "falsifiers": falsifier_rows,
        "applicability": _strings(applicability),
        "uncertainty": {
            "confidence": _confidence(confidence),
            "basis": str(uncertainty_basis or "unassessed").strip(),
        },
        "initial_status": status,
        "metadata": deepcopy(metadata or {}),
    }
    content_hash = _canonical_hash(core)
    return {
        "node_id": f"{kind}:" + content_hash.split(":", 1)[1][:16],
        "content_hash": content_hash,
        "created_at": _now_iso(),
        **core,
    }


def build_epistemic_edge(
    *, source: str, target: str, edge_type: str, created_by: str, rationale: str
) -> dict[str, Any]:
    relation = str(edge_type or "").strip()
    required = {
        "source": source,
        "target": target,
        "created_by": created_by,
        "rationale": rationale,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise EpistemicGraphError("missing edge fields: " + ", ".join(missing))
    if relation not in EDGE_TYPES:
        raise EpistemicGraphError(f"unsupported edge_type={relation!r}")
    if str(source).strip() == str(target).strip():
        raise EpistemicGraphError("self-referential epistemic edges are forbidden")
    core = {
        "source": str(source).strip(),
        "target": str(target).strip(),
        "edge_type": relation,
        "created_by": str(created_by).strip(),
        "rationale": str(rationale).strip(),
    }
    edge_hash = _canonical_hash(core)
    return {
        "edge_id": "edge:" + edge_hash.split(":", 1)[1][:16],
        "edge_hash": edge_hash,
        "created_at": _now_iso(),
        **core,
    }


def _graph_hash_payload(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": graph.get("schema_version"),
        "constitution_hash": graph.get("constitution_hash"),
        "nodes": graph.get("nodes"),
        "edges": graph.get("edges"),
        "transitions": graph.get("transitions"),
    }


def _refresh_graph(graph: dict[str, Any]) -> dict[str, Any]:
    graph["updated_at"] = _now_iso()
    graph["graph_hash"] = _canonical_hash(_graph_hash_payload(graph))
    return graph


def build_epistemic_graph(
    idea_cards: Iterable[dict[str, Any]],
    *,
    constitution: dict[str, Any],
    producer: str,
) -> dict[str, Any]:
    assert_science_constitution_intact(constitution)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for index, card in enumerate(item for item in idea_cards if isinstance(item, dict)):
        source = (
            card.get("source_idea") if isinstance(card.get("source_idea"), dict) else {}
        )
        title = str(card.get("title") or card.get("name") or f"Idea {index}").strip()
        question_text = str(
            card.get("research_question")
            or source.get("Research Question")
            or f"Under what conditions is the proposed mechanism for {title} valid?"
        ).strip()
        question = build_epistemic_node(
            node_type="question",
            title=f"Research question: {title}",
            statement=question_text,
            created_by=producer,
            confidence=0.0,
            uncertainty_basis="Open research question",
            metadata={"source_idea_id": card.get("idea_id")},
        )
        hypothesis_text = str(
            card.get("core_hypothesis") or source.get("Short Hypothesis") or title
        ).strip()
        falsifiers = _strings(card.get("failure_criteria")) or [
            "A preregistered discriminating test contradicts the predicted outcome."
        ]
        hypothesis = build_epistemic_node(
            node_type="hypothesis",
            title=title,
            statement=hypothesis_text,
            created_by=producer,
            parent_ids=[question["node_id"]],
            falsifiers=falsifiers,
            confidence=0.0,
            uncertainty_basis="Seed hypothesis; no confirmatory evidence yet",
            metadata={
                "source_idea_id": card.get("idea_id"),
                "mechanism": card.get("mechanism"),
                "literature_queries": list(card.get("literature_queries") or []),
            },
        )
        nodes.extend([question, hypothesis])
        edges.append(
            build_epistemic_edge(
                source=hypothesis["node_id"],
                target=question["node_id"],
                edge_type="addresses",
                created_by=producer,
                rationale="Seed hypothesis is an attempted answer to the research question.",
            )
        )
    graph = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "updated_at": _now_iso(),
        "constitution_hash": constitution["constitution_hash"],
        "nodes": nodes,
        "edges": edges,
        "transitions": [],
    }
    return _refresh_graph(graph)


def _current_statuses(graph: dict[str, Any]) -> dict[str, str]:
    statuses = {
        str(node.get("node_id")): str(node.get("initial_status"))
        for node in graph.get("nodes") or []
        if isinstance(node, dict)
    }
    for event in graph.get("transitions") or []:
        if isinstance(event, dict) and event.get("node_id") in statuses:
            statuses[str(event["node_id"])] = str(event.get("to_status"))
    return statuses


def current_epistemic_status(graph: dict[str, Any], node_id: str) -> str:
    status = _current_statuses(graph).get(str(node_id))
    if status is None:
        raise EpistemicGraphError(f"unknown node_id={node_id!r}")
    return status


def validate_epistemic_graph(graph: dict[str, Any] | None) -> dict[str, Any]:
    payload = graph if isinstance(graph, dict) else {}
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if not str(payload.get("constitution_hash") or "").startswith("sha256:"):
        errors.append("constitution_hash_invalid")
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    node_ids: set[str] = set()
    statuses: dict[str, str] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"node_invalid:{index}")
            continue
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in node_ids:
            errors.append(f"node_id_invalid_or_duplicate:{index}")
        node_ids.add(node_id)
        if node.get("node_type") not in NODE_TYPES:
            errors.append(f"node_type_invalid:{node_id}")
        if node.get("initial_status") not in EPISTEMIC_STATES:
            errors.append(f"node_initial_status_invalid:{node_id}")
        elif node.get("initial_status") != "speculative":
            errors.append(f"node_initial_status_bypasses_history:{node_id}")
        statuses[node_id] = str(node.get("initial_status"))
        if node.get("content_hash") != _canonical_hash(_node_core(node)):
            errors.append(f"node_content_hash_mismatch:{node_id}")
        if node.get("node_type") in FALSIFIER_REQUIRED_TYPES and not _strings(
            node.get("falsifiers")
        ):
            errors.append(f"node_falsifier_missing:{node_id}")
    edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge_invalid:{index}")
            continue
        edge_id = str(edge.get("edge_id") or "")
        if not edge_id or edge_id in edge_ids:
            errors.append(f"edge_id_invalid_or_duplicate:{index}")
        edge_ids.add(edge_id)
        core = {
            key: edge.get(key)
            for key in ("source", "target", "edge_type", "created_by", "rationale")
        }
        if edge.get("edge_hash") != _canonical_hash(core):
            errors.append(f"edge_hash_mismatch:{edge_id}")
        if edge.get("edge_type") not in EDGE_TYPES:
            errors.append(f"edge_type_invalid:{edge_id}")
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            errors.append(f"edge_endpoint_missing:{edge_id}")
    transitions = (
        payload.get("transitions")
        if isinstance(payload.get("transitions"), list)
        else []
    )
    previous_hash: str | None = None
    for index, event in enumerate(transitions):
        if not isinstance(event, dict):
            errors.append(f"transition_invalid:{index}")
            continue
        node_id = str(event.get("node_id") or "")
        core = {
            key: event.get(key)
            for key in (
                "event_id",
                "node_id",
                "from_status",
                "to_status",
                "actor_id",
                "reason",
                "evidence_refs",
                "previous_event_hash",
            )
        }
        if event.get("event_hash") != _canonical_hash(core):
            errors.append(f"transition_hash_mismatch:{index}")
        if event.get("previous_event_hash") != previous_hash:
            errors.append(f"transition_chain_mismatch:{index}")
        previous_hash = event.get("event_hash")
        if node_id not in statuses:
            errors.append(f"transition_node_missing:{index}")
            continue
        from_status = statuses[node_id]
        to_status = str(event.get("to_status") or "")
        if event.get("from_status") != from_status:
            errors.append(f"transition_from_status_mismatch:{index}")
        if to_status not in ALLOWED_TRANSITIONS.get(from_status, set()):
            errors.append(f"transition_not_allowed:{from_status}->{to_status}")
        if len(_strings(event.get("evidence_refs"))) < MINIMUM_EVIDENCE_REFS.get(
            to_status, 1
        ):
            errors.append(f"transition_evidence_insufficient:{index}")
        statuses[node_id] = to_status
    if payload.get("graph_hash") != _canonical_hash(_graph_hash_payload(payload)):
        errors.append("graph_hash_mismatch")
    return {"passed": not errors, "errors": errors, "current_statuses": statuses}


def assert_epistemic_graph_valid(graph: dict[str, Any] | None) -> None:
    validation = validate_epistemic_graph(graph)
    if not validation["passed"]:
        raise EpistemicGraphError(
            "epistemic graph invalid: " + ", ".join(validation["errors"])
        )


def advance_epistemic_node(
    graph: dict[str, Any],
    *,
    node_id: str,
    to_status: str,
    actor_id: str,
    reason: str,
    evidence_refs: Iterable[str],
    evaluation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a hash-chained state transition without mutating the node."""

    assert_epistemic_graph_valid(graph)
    destination = str(to_status or "").strip()
    actor = str(actor_id or "").strip()
    explanation = str(reason or "").strip()
    if not actor or not explanation:
        raise EpistemicGraphError("actor_id and reason are required")
    source = current_epistemic_status(graph, node_id)
    if destination not in ALLOWED_TRANSITIONS.get(source, set()):
        raise EpistemicGraphError(f"transition not allowed: {source}->{destination}")
    refs = _strings(evidence_refs)
    minimum = MINIMUM_EVIDENCE_REFS.get(destination, 1)
    if len(refs) < minimum:
        raise EpistemicGraphError(
            f"transition to {destination} requires at least {minimum} evidence refs"
        )
    if destination in {"robust", "canonical"}:
        from ai_scientist.utils.evaluation_governance import (
            assert_scientific_promotion_allowed,
        )

        assert_scientific_promotion_allowed(evaluation_report, node_id=node_id)
    updated = deepcopy(graph)
    previous_hash = (
        updated["transitions"][-1]["event_hash"] if updated["transitions"] else None
    )
    event_id = f"transition:{len(updated['transitions']) + 1}"
    core = {
        "event_id": event_id,
        "node_id": str(node_id),
        "from_status": source,
        "to_status": destination,
        "actor_id": actor,
        "reason": explanation,
        "evidence_refs": refs,
        "previous_event_hash": previous_hash,
    }
    updated["transitions"].append(
        {**core, "created_at": _now_iso(), "event_hash": _canonical_hash(core)}
    )
    return _refresh_graph(updated)


def save_epistemic_graph(
    project_root: str | Path,
    graph: dict[str, Any],
    *,
    producer: str,
) -> str:
    assert_epistemic_graph_valid(graph)
    constitution = load_contract_artifact(
        project_root,
        "science_constitution",
        default={},
    )
    assert_science_constitution_intact(constitution)
    if graph.get("constitution_hash") != constitution.get("constitution_hash"):
        raise EpistemicGraphError(
            "epistemic graph is not bound to the project's locked constitution"
        )
    return save_contract_artifact(
        project_root,
        "epistemic_graph",
        graph,
        producer=producer,
        depends_on=["science_constitution", "idea_cards"],
        notes="Append-only nodes and hash-chained epistemic transitions.",
    )


def seed_scientific_foundation(
    project_root: str | Path,
    idea_cards: Iterable[dict[str, Any]],
    *,
    producer: str,
    additional_constraints: Iterable[str] | None = None,
) -> dict[str, str]:
    from ai_scientist.utils.science_constitution import (
        build_science_constitution,
        save_science_constitution,
    )

    root = Path(project_root).expanduser().resolve()
    constitution = build_science_constitution(
        project_name=root.name,
        additional_constraints=additional_constraints,
    )
    constitution_path = save_science_constitution(root, constitution, producer=producer)
    graph = build_epistemic_graph(
        idea_cards,
        constitution=constitution,
        producer=producer,
    )
    graph_path = save_epistemic_graph(root, graph, producer=producer)
    return {
        "science_constitution": constitution_path,
        "epistemic_graph": graph_path,
    }


__all__ = [
    "ALLOWED_TRANSITIONS",
    "EDGE_TYPES",
    "EPISTEMIC_STATES",
    "EpistemicGraphError",
    "NODE_TYPES",
    "advance_epistemic_node",
    "assert_epistemic_graph_valid",
    "build_epistemic_edge",
    "build_epistemic_graph",
    "build_epistemic_node",
    "current_epistemic_status",
    "save_epistemic_graph",
    "seed_scientific_foundation",
    "validate_epistemic_graph",
]
