from __future__ import annotations

from argparse import Namespace
from typing import Callable, Sequence

from ai_scientist.utils.high_quality_pipeline import (
    recommend_target_venue_from_idea,
    resolve_submission_acceptance_settings,
)
from ai_scientist.utils.workflow_modes import apply_workflow_mode_defaults
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    normalize_writing_profile,
)


def normalize_writing_workflow_args(
    args: Namespace,
    *,
    invalid_profile_logger: Callable[[ValueError], None] | None = None,
) -> Namespace:
    try:
        args.writing_profile = normalize_writing_profile(args.writing_profile)
    except ValueError as exc:
        if invalid_profile_logger is not None:
            invalid_profile_logger(exc)
        args.writing_profile = DEFAULT_WRITING_PROFILE

    args.writing_audit_rounds = max(0, int(args.writing_audit_rounds))
    args.guardrail_repair_rounds = max(0, int(args.guardrail_repair_rounds))
    if not hasattr(args, "autonomous_quality_followup_rounds"):
        args.autonomous_quality_followup_rounds = 0
    args.autonomous_quality_followup_rounds = max(
        0, int(args.autonomous_quality_followup_rounds)
    )
    return args


def recommend_default_target_venue(
    *,
    writeup_type: str | None = None,
    paper_types: Sequence[str] | None = None,
    all_types: bool = False,
) -> str:
    if writeup_type is not None:
        base_paper_type = writeup_type
    elif all_types:
        base_paper_type = "journal"
    else:
        normalized_paper_types = [str(paper_type) for paper_type in (paper_types or [])]
        base_paper_type = normalized_paper_types[0] if normalized_paper_types else "normal"
    return recommend_target_venue_from_idea(None, base_paper_type)


def _apply_ranked_high_quality_defaults(
    args: Namespace,
    *,
    rank_flag_attr: str,
    candidate_limit_attr: str,
    default_candidate_count: int,
    fallback_flag_attr: str | None = None,
) -> None:
    setattr(args, rank_flag_attr, True)
    if fallback_flag_attr is not None and hasattr(args, fallback_flag_attr):
        setattr(args, fallback_flag_attr, True)

    args.high_quality_mode = True
    args.require_quality_gate = True
    args.auto_adjust_paper_type = True
    args.quality_preset = "publishable"
    args.writing_audit_rounds = max(args.writing_audit_rounds, 1)
    args.strict_writing_guardrails = True
    args.guardrail_repair_rounds = max(args.guardrail_repair_rounds, 1)
    args.autonomous_quality_followup_rounds = max(
        args.autonomous_quality_followup_rounds, 1
    )
    (
        args.min_submission_priority,
        args.max_submission_blockers,
    ) = resolve_submission_acceptance_settings(
        args.target_venue,
        min_submission_priority=args.min_submission_priority,
        max_submission_blockers=args.max_submission_blockers,
    )
    if getattr(args, candidate_limit_attr) is None:
        setattr(args, candidate_limit_attr, default_candidate_count)


def normalize_launcher_workflow_args(
    args: Namespace,
    *,
    invalid_profile_logger: Callable[[ValueError], None] | None = None,
) -> Namespace:
    normalize_writing_workflow_args(
        args,
        invalid_profile_logger=invalid_profile_logger,
    )

    if args.submission_mode:
        if args.target_venue is None:
            args.target_venue = recommend_default_target_venue(
                writeup_type=args.writeup_type,
            )
        default_max_ranked_candidates = (
            5 if args.target_venue in {"nature", "journal"} else 3
        )
        _apply_ranked_high_quality_defaults(
            args,
            rank_flag_attr="auto_best_idea",
            candidate_limit_attr="max_ranked_candidates",
            fallback_flag_attr="fallback_ranked_ideas",
            default_candidate_count=default_max_ranked_candidates,
        )

    if args.breakthrough_mode:
        args.target_venue = args.target_venue or "nature"
        _apply_ranked_high_quality_defaults(
            args,
            rank_flag_attr="auto_best_idea",
            candidate_limit_attr="max_ranked_candidates",
            fallback_flag_attr="fallback_ranked_ideas",
            default_candidate_count=7,
        )

    apply_workflow_mode_defaults(
        args,
        rank_flag_attr="auto_best_idea",
        candidate_limit_attr="max_ranked_candidates",
        fallback_flag_attr="fallback_ranked_ideas",
    )
    return args


def normalize_project_workflow_args(
    args: Namespace,
    *,
    invalid_profile_logger: Callable[[ValueError], None] | None = None,
) -> Namespace:
    normalize_writing_workflow_args(
        args,
        invalid_profile_logger=invalid_profile_logger,
    )

    if args.submission_mode:
        if args.target_venue is None:
            args.target_venue = recommend_default_target_venue(
                writeup_type=args.writeup_type,
            )
        default_top_k_ideas = 5 if args.target_venue in {"nature", "journal"} else 3
        _apply_ranked_high_quality_defaults(
            args,
            rank_flag_attr="rank_ideas",
            candidate_limit_attr="top_k_ideas",
            fallback_flag_attr="fallback_ranked_ideas",
            default_candidate_count=default_top_k_ideas,
        )

    if args.breakthrough_mode:
        args.target_venue = args.target_venue or "nature"
        _apply_ranked_high_quality_defaults(
            args,
            rank_flag_attr="rank_ideas",
            candidate_limit_attr="top_k_ideas",
            fallback_flag_attr="fallback_ranked_ideas",
            default_candidate_count=7,
        )

    apply_workflow_mode_defaults(
        args,
        rank_flag_attr="rank_ideas",
        candidate_limit_attr="top_k_ideas",
        fallback_flag_attr="fallback_ranked_ideas",
    )
    return args


def normalize_batch_workflow_args(
    args: Namespace,
    *,
    invalid_profile_logger: Callable[[ValueError], None] | None = None,
) -> Namespace:
    normalize_writing_workflow_args(
        args,
        invalid_profile_logger=invalid_profile_logger,
    )

    if args.submission_mode:
        if args.target_venue is None:
            args.target_venue = recommend_default_target_venue(
                paper_types=args.paper_types,
                all_types=args.all_types,
            )
        default_top_k_ideas = 5 if args.target_venue in {"nature", "journal"} else 3
        _apply_ranked_high_quality_defaults(
            args,
            rank_flag_attr="rank_ideas",
            candidate_limit_attr="top_k_ideas",
            default_candidate_count=default_top_k_ideas,
        )

    if args.breakthrough_mode:
        args.target_venue = args.target_venue or "nature"
        _apply_ranked_high_quality_defaults(
            args,
            rank_flag_attr="rank_ideas",
            candidate_limit_attr="top_k_ideas",
            default_candidate_count=7,
        )

    apply_workflow_mode_defaults(
        args,
        rank_flag_attr="rank_ideas",
        candidate_limit_attr="top_k_ideas",
    )
    return args
