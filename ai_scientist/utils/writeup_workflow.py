from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai_scientist.utils.high_quality_pipeline import VENUE_PRESETS, resolve_target_venue
from ai_scientist.utils.submission_history import (
    recommend_submission_strategy_adjustments,
)


PAGE_LIMITS = {
    "normal": 8,
    "icbinb": 4,
    "journal": 12,
    "extended": 2,
}

MANUSCRIPT_POLICIES = {
    "neurips": {
        "required_sections": [
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "results",
            "limitations",
            "conclusion",
        ],
        "mandatory_evidence_density": "high",
        "frontmatter_numerics_required": True,
        "limitation_policy": "state at least one real failure mode or scope caveat in the main paper",
        "rebuttal_style": "defensive but evidence-first",
    },
    "iclr": {
        "required_sections": [
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "discussion",
            "limitations",
            "conclusion",
        ],
        "mandatory_evidence_density": "high",
        "frontmatter_numerics_required": True,
        "limitation_policy": "pair every strong claim with its evaluated scope",
        "rebuttal_style": "clear, scoped, and mechanistic",
    },
    "cvpr": {
        "required_sections": [
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "results",
            "visual_analysis",
            "conclusion",
        ],
        "mandatory_evidence_density": "high",
        "frontmatter_numerics_required": True,
        "limitation_policy": "call out visual failure cases and qualitative boundaries",
        "rebuttal_style": "visual-evidence heavy",
    },
    "journal": {
        "required_sections": [
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "results",
            "discussion",
            "limitations",
            "reproducibility",
            "conclusion",
        ],
        "mandatory_evidence_density": "very_high",
        "frontmatter_numerics_required": True,
        "limitation_policy": "document stability, robustness, and clear caveats in the main narrative",
        "rebuttal_style": "comprehensive and conservative",
    },
    "nature": {
        "required_sections": [
            "title",
            "abstract",
            "introduction",
            "results",
            "discussion",
            "limitations",
            "conclusion",
        ],
        "mandatory_evidence_density": "extreme",
        "frontmatter_numerics_required": True,
        "limitation_policy": "keep claims narrow enough that every major statement is visibly evidenced",
        "rebuttal_style": "high-scrutiny, significance-aware, and heavily caveated",
    },
}


def resolve_writeup_engine(writeup_type: str) -> str:
    return "icbinb" if writeup_type in {"icbinb", "extended"} else "normal"


def resolve_page_limit(writeup_type: str) -> int:
    return PAGE_LIMITS.get(writeup_type, PAGE_LIMITS["icbinb"])


def resolve_manuscript_policy(target_venue: str) -> dict:
    return dict(MANUSCRIPT_POLICIES.get(target_venue, MANUSCRIPT_POLICIES["neurips"]))


def build_writeup_execution_plan(
    writeup_type: str,
    *,
    num_cite_rounds: int,
    writeup_retries: int,
    target_venue: Optional[str] = None,
    high_quality_mode: bool = False,
    research_root: str | Path | None = None,
) -> dict:
    resolved_target_venue = resolve_target_venue(writeup_type, target_venue)
    page_limit = resolve_page_limit(writeup_type)
    strategy_feedback = {}
    effective_num_cite_rounds = num_cite_rounds
    effective_writeup_retries = writeup_retries

    if high_quality_mode:
        venue_config = VENUE_PRESETS.get(
            resolved_target_venue,
            VENUE_PRESETS["neurips"],
        )
        strategy_feedback = recommend_submission_strategy_adjustments(
            resolved_target_venue,
            research_root=research_root,
        )
        effective_num_cite_rounds = max(
            num_cite_rounds,
            venue_config.get("min_cite_rounds", 20),
        ) + strategy_feedback.get("cite_round_boost", 0)
        effective_writeup_retries = max(
            writeup_retries,
            venue_config.get("min_writeup_retries", 4),
        ) + strategy_feedback.get("writeup_retry_boost", 0)

    return {
        "target_venue": resolved_target_venue,
        "page_limit": page_limit,
        "writeup_engine": resolve_writeup_engine(writeup_type),
        "num_cite_rounds": effective_num_cite_rounds,
        "writeup_retries": effective_writeup_retries,
        "manuscript_policy": resolve_manuscript_policy(resolved_target_venue),
        "strategy_feedback": strategy_feedback,
    }
