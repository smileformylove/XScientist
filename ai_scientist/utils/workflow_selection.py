from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ai_scientist.config.paths import resolve_output_path
from ai_scientist.utils.high_quality_pipeline import (
    is_paper_type_fit_for_venue,
    recommend_paper_type_for_venue,
)
from ai_scientist.utils.idea_ranking import rank_ideas


def _normalize_indices(
    indices: Iterable[int] | None,
    *,
    idea_count: int,
) -> list[int]:
    if indices is None:
        return []

    normalized: list[int] = []
    seen: set[int] = set()
    for value in indices:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= idea_count or index in seen:
            continue
        normalized.append(index)
        seen.add(index)
    return normalized


def _selection_penalty(item: dict) -> float:
    penalty = 0.0
    if item.get("fallback_used"):
        penalty += 0.4
        stage = str(item.get("fallback_stage") or "")
        if stage == "client_creation":
            penalty += 0.35
        elif stage == "response_parsing":
            penalty += 0.15
        else:
            penalty += 0.2
    return round(penalty, 3)


def _prepare_rankings_for_selection(
    rankings: Sequence[dict],
) -> list[dict]:
    prepared: list[dict] = []
    for item in rankings:
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        base_score = enriched.get("ranking_score", enriched.get("total_score", 0.0))
        try:
            numeric_base_score = float(base_score)
        except (TypeError, ValueError):
            numeric_base_score = 0.0
        penalty = _selection_penalty(enriched)
        enriched["selection_penalty"] = penalty
        enriched["selection_score"] = round(numeric_base_score - penalty, 3)
        prepared.append(enriched)
    prepared.sort(
        key=lambda item: (
            item.get("selection_score", 0.0),
            item.get("ranking_score", item.get("total_score", 0.0)),
            item.get("total_score", 0.0),
            item.get("breakthrough_potential", 0.0),
            item.get("fallback_used") is not True,
        ),
        reverse=True,
    )
    for idx, item in enumerate(prepared, start=1):
        item["selection_rank"] = idx
    return prepared


def select_ranked_idea_candidates(
    ideas: Sequence[dict],
    *,
    ranking_enabled: bool,
    ranking_model: str | None,
    target_venue: str | None = None,
    prioritize_breakthrough: bool = False,
    research_root: str | Path | None = None,
    ranking_output_path: str | Path | None = None,
    requested_indices: Sequence[int] | None = None,
    default_indices: Sequence[int] | None = None,
    fallback_to_ranked: bool = False,
    use_ranked_all: bool = False,
    limit: int | None = None,
    ranker: Callable[..., list[dict]] = rank_ideas,
) -> tuple[list[int], list[dict]]:
    active_research_root = research_root if research_root is not None else resolve_output_path()
    idea_count = len(ideas)
    requested = _normalize_indices(requested_indices, idea_count=idea_count)
    defaults = _normalize_indices(default_indices, idea_count=idea_count)
    max_candidates = None if limit is None else max(0, int(limit))

    rankings: list[dict] = []
    if ranking_enabled:
        rankings = ranker(
            list(ideas),
            model=ranking_model,
            target_venue=target_venue,
            prioritize_breakthrough=prioritize_breakthrough,
            research_root=active_research_root,
            output_path=ranking_output_path,
        )
        rankings = _prepare_rankings_for_selection(rankings)
        if ranking_output_path is not None:
            output_path = Path(ranking_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rankings, f, indent=2, ensure_ascii=False)
        ranked_indices = _normalize_indices(
            [item.get("idea_idx") for item in rankings],
            idea_count=idea_count,
        )
        if max_candidates is not None:
            ranked_indices = ranked_indices[:max_candidates]
        if requested_indices is not None:
            requested_set = set(requested)
            selected = [idx for idx in ranked_indices if idx in requested_set]
        elif use_ranked_all or fallback_to_ranked:
            selected = ranked_indices
        else:
            selected = ranked_indices[:1]
        return selected, rankings

    if requested_indices is not None:
        selected = requested
    elif defaults:
        selected = defaults
    elif idea_count:
        selected = [0]
    else:
        selected = []

    if max_candidates is not None:
        selected = selected[:max_candidates]
    return selected, rankings


def resolve_paper_type_for_venue(
    paper_type: str,
    target_venue: str | None,
    *,
    auto_adjust: bool = False,
    logger: Callable[[str], None] = print,
    warning_template: str | None = None,
    adjusted_template: str | None = None,
) -> str:
    if not target_venue or is_paper_type_fit_for_venue(paper_type, target_venue):
        return paper_type

    warning = warning_template or (
        "Warning: paper_type '{paper_type}' may be a weak fit for target venue "
        "'{target_venue}'"
    )
    logger(warning.format(paper_type=paper_type, target_venue=target_venue))
    if not auto_adjust:
        return paper_type

    adjusted = recommend_paper_type_for_venue(target_venue)
    adjusted_message = adjusted_template or (
        "Auto-adjusted paper_type '{paper_type}' -> '{adjusted}'"
    )
    logger(
        adjusted_message.format(
            paper_type=paper_type,
            target_venue=target_venue,
            adjusted=adjusted,
        )
    )
    return adjusted


def resolve_paper_types_for_venue(
    paper_types: Sequence[str],
    target_venue: str | None,
    *,
    auto_adjust: bool = False,
    logger: Callable[[str], None] = print,
    warning_template: str | None = None,
    adjusted_template: str | None = None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for paper_type in paper_types:
        adjusted = resolve_paper_type_for_venue(
            paper_type,
            target_venue,
            auto_adjust=auto_adjust,
            logger=logger,
            warning_template=warning_template,
            adjusted_template=adjusted_template,
        )
        if adjusted not in seen:
            resolved.append(adjusted)
            seen.add(adjusted)
    return resolved
