from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai_scientist.review_strategies import (
    ReviewStrategy,
    ReviewStrategyManager,
    generate_review_instruction,
)
from ai_scientist.utils.high_quality_pipeline import VENUE_PRESETS, resolve_target_venue
from ai_scientist.utils.submission_history import (
    recommend_submission_strategy_adjustments,
)


DEFAULT_REVIEW_STRATEGY_BY_TYPE = {
    "normal": ReviewStrategy.NEURIPS,
    "icbinb": ReviewStrategy.ICLR,
    "journal": ReviewStrategy.JOURNAL,
    "extended": ReviewStrategy.FAST,
}

DEFAULT_REVIEW_STRATEGY_BY_VENUE = {
    "neurips": ReviewStrategy.NEURIPS,
    "iclr": ReviewStrategy.ICLR,
    "cvpr": ReviewStrategy.CVPR,
    "journal": ReviewStrategy.JOURNAL,
    "nature": ReviewStrategy.NATURE,
}


def resolve_review_strategy(
    paper_type: str,
    *,
    target_venue: Optional[str] = None,
    review_strategy: Optional[str] = None,
    high_quality_mode: bool = False,
    default_quality_requirement: str = "standard",
) -> ReviewStrategy:
    if review_strategy is not None:
        return ReviewStrategy(review_strategy)

    resolved_target_venue = resolve_target_venue(paper_type, target_venue)
    if high_quality_mode:
        return DEFAULT_REVIEW_STRATEGY_BY_VENUE.get(
            resolved_target_venue,
            DEFAULT_REVIEW_STRATEGY_BY_TYPE.get(
                paper_type,
                ReviewStrategyManager.recommend_strategy(
                    paper_type=paper_type,
                    quality_requirement="high",
                ),
            ),
        )

    return ReviewStrategyManager.recommend_strategy(
        paper_type=paper_type,
        quality_requirement=default_quality_requirement,
    )


def build_review_execution_plan(
    paper_type: str,
    *,
    target_venue: Optional[str] = None,
    review_reflections: int = 1,
    review_ensemble: int = 1,
    review_fewshot: int = 1,
    review_temperature: float = 0.75,
    review_strategy: Optional[str] = None,
    high_quality_mode: bool = False,
    research_root: str | Path | None = None,
    default_quality_requirement: str = "standard",
) -> dict:
    resolved_target_venue = resolve_target_venue(paper_type, target_venue)
    strategy = resolve_review_strategy(
        paper_type,
        target_venue=resolved_target_venue,
        review_strategy=review_strategy,
        high_quality_mode=high_quality_mode,
        default_quality_requirement=default_quality_requirement,
    )

    effective_review_reflections = review_reflections
    effective_review_ensemble = review_ensemble
    effective_review_fewshot = review_fewshot
    effective_review_temperature = review_temperature
    strategy_feedback = {}

    if high_quality_mode:
        venue_config = VENUE_PRESETS.get(
            resolved_target_venue,
            VENUE_PRESETS["neurips"],
        )
        strategy_feedback = recommend_submission_strategy_adjustments(
            resolved_target_venue,
            research_root=research_root,
        )
        effective_review_reflections = max(
            review_reflections,
            venue_config.get("min_review_reflections", 2),
        ) + strategy_feedback.get("review_reflection_boost", 0)
        effective_review_ensemble = max(
            review_ensemble,
            venue_config.get("min_review_ensemble", 3),
        ) + strategy_feedback.get("review_ensemble_boost", 0)
        effective_review_fewshot = max(
            review_fewshot,
            venue_config.get("min_review_fewshot", 2),
        ) + strategy_feedback.get("review_fewshot_boost", 0)
        effective_review_temperature = min(review_temperature, 0.65)

    return {
        "target_venue": resolved_target_venue,
        "strategy": strategy,
        "review_instruction": generate_review_instruction(strategy),
        "review_reflections": effective_review_reflections,
        "review_ensemble": effective_review_ensemble,
        "review_fewshot": effective_review_fewshot,
        "review_temperature": effective_review_temperature,
        "strategy_feedback": strategy_feedback,
    }
