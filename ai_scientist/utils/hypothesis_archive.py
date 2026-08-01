from __future__ import annotations

"""Append-only, quality-diverse hypothesis archive for open-ended discovery."""

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.pipeline_contracts import save_contract_artifact

SCHEMA_VERSION = 1
SCORE_DIMENSIONS = (
    "novelty",
    "feasibility",
    "falsifiability",
    "information_gain",
    "impact",
    "evidence_grounding",
    "safety",
)
DEFAULT_WEIGHTS = {
    "novelty": 1.2,
    "feasibility": 0.8,
    "falsifiability": 1.2,
    "information_gain": 1.1,
    "impact": 1.0,
    "evidence_grounding": 1.2,
    "safety": 0.7,
}
GENERATION_OPERATORS = {
    "initial",
    "analogy",
    "combination",
    "contradiction",
    "boundary_condition",
    "simplification",
    "failure_driven",
    "mechanism",
    "out_of_distribution",
}


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


def _text_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    return {
        token
        for token in re.findall(r"[a-z0-9_\-]{3,}|[\u4e00-\u9fff]", text)
        if token not in {"the", "and", "with", "from", "that", "this", "using"}
    }


def hypothesis_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Dependency-free lexical proximity used for dedupe and diversity niches."""

    left_tokens = _text_tokens(
        " ".join(
            str(left.get(key) or "")
            for key in ("title", "hypothesis", "mechanism", "research_question")
        )
    )
    right_tokens = _text_tokens(
        " ".join(
            str(right.get(key) or "")
            for key in ("title", "hypothesis", "mechanism", "research_question")
        )
    )
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 6)


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return round(max(0.0, min(5.0, score)), 4)


def _score_payload(card: dict[str, Any]) -> dict[str, float]:
    ranking = card.get("ranking") if isinstance(card.get("ranking"), dict) else {}
    source = (
        card.get("source_idea") if isinstance(card.get("source_idea"), dict) else {}
    )
    return {
        name: _safe_score(card.get(name, ranking.get(name, source.get(name, 0.0))))
        for name in SCORE_DIMENSIONS
    }


def build_hypothesis_node(
    card: dict[str, Any],
    *,
    parent_ids: Iterable[str] | None = None,
    generation_operator: str | None = None,
) -> dict[str, Any]:
    source = (
        card.get("source_idea") if isinstance(card.get("source_idea"), dict) else {}
    )
    hypothesis = str(
        card.get("core_hypothesis")
        or source.get("Short Hypothesis")
        or card.get("title")
        or source.get("Title")
        or ""
    ).strip()
    title = str(card.get("title") or source.get("Title") or hypothesis).strip()
    operator = str(
        generation_operator
        or card.get("generation_operator")
        or source.get("Generation Operator")
        or "initial"
    ).strip()
    if operator not in GENERATION_OPERATORS:
        operator = "initial"
    literature = (
        source.get("Literature Search")
        if isinstance(source.get("Literature Search"), dict)
        else {}
    )
    content = {
        "title": title,
        "hypothesis": hypothesis,
        "mechanism": str(
            card.get("mechanism") or source.get("Mechanism") or ""
        ).strip(),
        "research_question": str(card.get("research_question") or title).strip(),
        "falsifiers": [
            str(item).strip()
            for item in card.get("failure_criteria") or []
            if str(item).strip()
        ],
        "candidate_datasets": list(card.get("candidate_datasets") or []),
        "candidate_metrics": list(card.get("candidate_metrics") or []),
        "candidate_baselines": list(card.get("candidate_baselines") or []),
        "literature_queries": list(
            literature.get("queries") or card.get("literature_queries") or []
        ),
        "supporting_papers": list(
            literature.get("papers") or card.get("supporting_papers") or []
        ),
    }
    content_hash = _canonical_hash(content)
    return {
        "hypothesis_id": "hyp_" + content_hash.split(":", 1)[1][:16],
        "content_hash": content_hash,
        "created_at": _now_iso(),
        "source_idea_id": card.get("idea_id"),
        "parent_ids": sorted(
            {
                str(item).strip()
                for item in (parent_ids or card.get("parent_ids") or [])
                if str(item).strip()
            }
        ),
        "generation_operator": operator,
        "status": str(card.get("status") or "proposed"),
        "title": content["title"],
        "hypothesis": content["hypothesis"],
        "mechanism": content["mechanism"],
        "research_question": content["research_question"],
        "falsifiers": content["falsifiers"],
        "candidate_datasets": content["candidate_datasets"],
        "candidate_metrics": content["candidate_metrics"],
        "candidate_baselines": content["candidate_baselines"],
        "literature_queries": content["literature_queries"],
        "supporting_papers": content["supporting_papers"],
        "scores": _score_payload(card),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_scores = left.get("scores") or {}
    right_scores = right.get("scores") or {}
    comparisons = [
        (_safe_score(left_scores.get(name)), _safe_score(right_scores.get(name)))
        for name in SCORE_DIMENSIONS
    ]
    return all(a >= b for a, b in comparisons) and any(a > b for a, b in comparisons)


def pareto_front(nodes: Iterable[dict[str, Any]]) -> list[str]:
    rows = [item for item in nodes if isinstance(item, dict)]
    return sorted(
        str(node.get("hypothesis_id"))
        for node in rows
        if not any(other is not node and _dominates(other, node) for other in rows)
    )


def _weighted_score(
    node: dict[str, Any], weights: dict[str, float] | None = None
) -> float:
    resolved = dict(DEFAULT_WEIGHTS)
    resolved.update(weights or {})
    scores = node.get("scores") or {}
    denominator = sum(max(float(value), 0.0) for value in resolved.values()) or 1.0
    weighted = sum(
        _safe_score(scores.get(name)) * max(float(weight), 0.0)
        for name, weight in resolved.items()
    )
    return weighted / denominator


def _proximity_graph(
    nodes: list[dict[str, Any]], *, threshold: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    parent = {str(node["hypothesis_id"]): str(node["hypothesis_id"]) for node in nodes}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges: list[dict[str, Any]] = []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            similarity = hypothesis_similarity(left, right)
            if similarity < threshold:
                continue
            left_id = str(left["hypothesis_id"])
            right_id = str(right["hypothesis_id"])
            union(left_id, right_id)
            edges.append(
                {
                    "source": left_id,
                    "target": right_id,
                    "type": "proximate",
                    "similarity": similarity,
                }
            )

    roots = sorted({find(item) for item in parent})
    root_index = {root: index for index, root in enumerate(roots)}
    clusters = {item: root_index[find(item)] for item in sorted(parent)}
    return edges, clusters


def _lineage_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_ids = {str(item.get("hypothesis_id")) for item in nodes}
    edges: list[dict[str, Any]] = []
    for node in nodes:
        child = str(node.get("hypothesis_id"))
        for parent in node.get("parent_ids") or []:
            parent_id = str(parent)
            edges.append(
                {
                    "source": parent_id,
                    "target": child,
                    "type": "evolves_to",
                    "resolved": parent_id in node_ids,
                }
            )
    return edges


def select_quality_diverse(
    archive: dict[str, Any],
    *,
    limit: int,
    weights: dict[str, float] | None = None,
) -> list[str]:
    """Select strong hypotheses while preserving at least one per niche."""

    if limit <= 0:
        return []
    nodes = [item for item in archive.get("nodes") or [] if isinstance(item, dict)]
    ratings = archive.get("ratings") or {}
    cluster_by_id = archive.get("clusters") or {}
    front = set(archive.get("pareto_front") or pareto_front(nodes))
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        node_id = str(node.get("hypothesis_id"))
        cluster = int(cluster_by_id.get(node_id, len(by_cluster)))
        by_cluster.setdefault(cluster, []).append(node)
    selected: list[str] = []
    representatives = []
    for cluster_nodes in by_cluster.values():
        representatives.append(
            max(
                cluster_nodes,
                key=lambda item: (
                    str(item.get("hypothesis_id")) in front,
                    _weighted_score(item, weights),
                    float(ratings.get(str(item.get("hypothesis_id")), 1000.0)),
                ),
            )
        )
    representatives.sort(
        key=lambda item: (
            str(item.get("hypothesis_id")) in front,
            _weighted_score(item, weights),
            float(ratings.get(str(item.get("hypothesis_id")), 1000.0)),
        ),
        reverse=True,
    )
    selected.extend(str(item.get("hypothesis_id")) for item in representatives[:limit])
    if len(selected) < limit:
        remaining = [
            item for item in nodes if str(item.get("hypothesis_id")) not in selected
        ]
        remaining.sort(
            key=lambda item: (
                str(item.get("hypothesis_id")) in front,
                _weighted_score(item, weights),
                float(ratings.get(str(item.get("hypothesis_id")), 1000.0)),
            ),
            reverse=True,
        )
        selected.extend(
            str(item.get("hypothesis_id"))
            for item in remaining[: limit - len(selected)]
        )
    return selected


def build_hypothesis_archive(
    idea_cards: Iterable[dict[str, Any]],
    *,
    proximity_threshold: float = 0.72,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for card in idea_cards:
        if not isinstance(card, dict):
            continue
        node = build_hypothesis_node(card)
        base_id = str(node["hypothesis_id"])
        occurrence = seen_ids.get(base_id, 0)
        seen_ids[base_id] = occurrence + 1
        if occurrence:
            node["hypothesis_id"] = f"{base_id}_{occurrence + 1}"
            node["duplicate_of"] = base_id
        nodes.append(node)
    proximity_edges, clusters = _proximity_graph(
        nodes, threshold=max(0.0, min(float(proximity_threshold), 1.0))
    )
    archive = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "append_only": True,
        "generation_operators": sorted(GENERATION_OPERATORS),
        "proximity_threshold": proximity_threshold,
        "nodes": nodes,
        "lineage_edges": _lineage_edges(nodes),
        "proximity_edges": proximity_edges,
        "clusters": clusters,
        "pareto_front": pareto_front(nodes),
        "ratings": {str(item["hypothesis_id"]): 1000.0 for item in nodes},
        "tournament_history": [],
    }
    archive["quality_diverse_selection"] = select_quality_diverse(
        archive, limit=min(8, len(nodes))
    )
    archive["summary"] = {
        "node_count": len(nodes),
        "cluster_count": len(set(clusters.values())),
        "pareto_count": len(archive["pareto_front"]),
        "duplicate_count": sum(bool(item.get("duplicate_of")) for item in nodes),
        "operator_counts": {
            operator: sum(item.get("generation_operator") == operator for item in nodes)
            for operator in sorted(GENERATION_OPERATORS)
            if any(item.get("generation_operator") == operator for item in nodes)
        },
    }
    return archive


def record_pairwise_comparison(
    archive: dict[str, Any],
    *,
    left_id: str,
    right_id: str,
    winner_id: str | None,
    judge_id: str,
    rationale: str,
    k_factor: float = 24.0,
) -> dict[str, Any]:
    """Return a new archive snapshot with an auditable Elo comparison."""

    updated = deepcopy(archive)
    nodes = {
        str(item.get("hypothesis_id")): item
        for item in updated.get("nodes") or []
        if isinstance(item, dict)
    }
    if left_id not in nodes or right_id not in nodes or left_id == right_id:
        raise ValueError("comparison requires two distinct archived hypotheses")
    if winner_id not in {None, left_id, right_id}:
        raise ValueError("winner_id must be left_id, right_id, or None for a tie")
    ratings = updated.setdefault("ratings", {})
    left_rating = float(ratings.get(left_id, 1000.0))
    right_rating = float(ratings.get(right_id, 1000.0))
    expected_left = 1.0 / (1.0 + 10.0 ** ((right_rating - left_rating) / 400.0))
    actual_left = 0.5 if winner_id is None else 1.0 if winner_id == left_id else 0.0
    ratings[left_id] = round(left_rating + k_factor * (actual_left - expected_left), 4)
    ratings[right_id] = round(
        right_rating + k_factor * ((1.0 - actual_left) - (1.0 - expected_left)),
        4,
    )
    updated.setdefault("tournament_history", []).append(
        {
            "compared_at": _now_iso(),
            "left_id": left_id,
            "right_id": right_id,
            "winner_id": winner_id,
            "judge_id": str(judge_id or "").strip(),
            "rationale": str(rationale or "").strip(),
            "left_rating_after": ratings[left_id],
            "right_rating_after": ratings[right_id],
        }
    )
    updated["generated_at"] = _now_iso()
    updated["quality_diverse_selection"] = select_quality_diverse(
        updated, limit=min(8, len(nodes))
    )
    return updated


def save_hypothesis_archive(
    project_root: str | Path,
    archive: dict[str, Any],
    *,
    producer: str,
) -> str:
    return save_contract_artifact(
        project_root,
        "hypothesis_archive",
        archive,
        producer=producer,
        depends_on=["idea_cards"],
    )


__all__ = [
    "GENERATION_OPERATORS",
    "SCORE_DIMENSIONS",
    "build_hypothesis_archive",
    "build_hypothesis_node",
    "hypothesis_similarity",
    "pareto_front",
    "record_pairwise_comparison",
    "save_hypothesis_archive",
    "select_quality_diverse",
]
