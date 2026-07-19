"""Pure ranking, filtering, and rewrite recommendation rules for ResearchManager."""

from __future__ import annotations

from typing import Dict


def _submission_priority_sort_key(paper: Dict):
    unresolved_critical = (
        paper.get("self_review_unresolved_critical")
        if isinstance(paper.get("self_review_unresolved_critical"), int)
        else 999
    )
    blocked_stage_count = (
        int(paper.get("blocked_stage_count"))
        if isinstance(paper.get("blocked_stage_count"), int)
        else 999
    )
    missing_stage_count = (
        int(paper.get("missing_stage_count"))
        if isinstance(paper.get("missing_stage_count"), int)
        else 999
    )
    attention_stage_count = (
        int(paper.get("needs_attention_stage_count"))
        if isinstance(paper.get("needs_attention_stage_count"), int)
        else 999
    )
    stage_overall_score = (
        float(paper.get("stage_overall_score"))
        if isinstance(paper.get("stage_overall_score"), (int, float))
        else -1.0
    )
    fallback_count = (
        int(paper.get("fallback_count"))
        if isinstance(paper.get("fallback_count"), int)
        else 999
    )
    strict_fallback_count = (
        int(paper.get("strict_fallback_count"))
        if isinstance(paper.get("strict_fallback_count"), int)
        else 999
    )
    gate_score = (
        paper.get("self_review_round_gate_score")
        if isinstance(paper.get("self_review_round_gate_score"), (int, float))
        else -1
    )
    review_resolution_rate = (
        float(paper.get("review_resolution_rate"))
        if isinstance(paper.get("review_resolution_rate"), (int, float))
        else -1.0
    )
    review_active_issue_count = (
        int(paper.get("review_active_issue_count"))
        if isinstance(paper.get("review_active_issue_count"), int)
        else 999
    )
    review_persistent_issue_count = (
        int(paper.get("review_persistent_issue_count"))
        if isinstance(paper.get("review_persistent_issue_count"), int)
        else 999
    )
    review_unbound_issue_count = (
        int(paper.get("review_unbound_issue_count"))
        if isinstance(paper.get("review_unbound_issue_count"), int)
        else 999
    )
    review_target_binding_coverage = (
        float(paper.get("review_target_binding_coverage"))
        if isinstance(paper.get("review_target_binding_coverage"), (int, float))
        else -1.0
    )
    review_repair_ready_coverage = (
        float(paper.get("review_repair_ready_coverage"))
        if isinstance(paper.get("review_repair_ready_coverage"), (int, float))
        else -1.0
    )
    self_evolution_status = str(paper.get("self_evolution_status") or "").strip()
    self_evolution_score = (
        float(paper.get("self_evolution_score"))
        if isinstance(paper.get("self_evolution_score"), (int, float))
        else -1.0
    )
    self_evolution_required_failure_count = (
        int(paper.get("self_evolution_required_failure_count"))
        if isinstance(paper.get("self_evolution_required_failure_count"), int)
        else 999
    )
    process_alignment_score = (
        float(paper.get("process_alignment_overall_score"))
        if isinstance(paper.get("process_alignment_overall_score"), (int, float))
        else -1.0
    )
    process_alignment_blocked_count = (
        int(paper.get("process_alignment_blocked_process_count"))
        if isinstance(paper.get("process_alignment_blocked_process_count"), int)
        else 999
    )
    return (
        paper.get("submission_status") == "ready",
        paper.get("quality_gate_passed") is True,
        paper.get("self_review_round_gate_ready") is True,
        blocked_stage_count == 0,
        missing_stage_count == 0,
        attention_stage_count == 0,
        process_alignment_blocked_count == 0,
        self_evolution_status != "blocked",
        gate_score,
        stage_overall_score,
        process_alignment_score,
        self_evolution_score,
        review_resolution_rate,
        review_repair_ready_coverage,
        review_target_binding_coverage,
        -process_alignment_blocked_count,
        -self_evolution_required_failure_count,
        -review_active_issue_count,
        -review_persistent_issue_count,
        -review_unbound_issue_count,
        -blocked_stage_count,
        -missing_stage_count,
        -attention_stage_count,
        -strict_fallback_count,
        -fallback_count,
        -unresolved_critical,
        (
            paper.get("submission_priority_score")
            if isinstance(paper.get("submission_priority_score"), (int, float))
            else -1
        ),
        (
            paper.get("rewrite_priority_gain_total")
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float))
            else -999
        ),
        -(
            paper.get("blocker_count")
            if isinstance(paper.get("blocker_count"), int)
            else 999
        ),
        -(
            paper.get("critical_revision_actions_count")
            if isinstance(paper.get("critical_revision_actions_count"), int)
            else 999
        ),
        (
            paper.get("quality_score")
            if isinstance(paper.get("quality_score"), (int, float))
            else -1
        ),
        (
            paper.get("breakthrough_score")
            if isinstance(paper.get("breakthrough_score"), (int, float))
            else -1
        ),
        (
            paper.get("rigor_score")
            if isinstance(paper.get("rigor_score"), (int, float))
            else -1
        ),
        (
            paper.get("claim_support_score")
            if isinstance(paper.get("claim_support_score"), (int, float))
            else -1
        ),
        (
            paper.get("numeric_coverage_score")
            if isinstance(paper.get("numeric_coverage_score"), (int, float))
            else -1
        ),
        (
            paper.get("evidence_density_score")
            if isinstance(paper.get("evidence_density_score"), (int, float))
            else -1
        ),
        (
            paper.get("contribution_count")
            if isinstance(paper.get("contribution_count"), int)
            else -1
        ),
        -(
            paper.get("unsupported_claims_count")
            if isinstance(paper.get("unsupported_claims_count"), int)
            else 999
        ),
        paper.get("modified_at", ""),
    )


def _passes_submission_filters(
    paper: Dict,
    *,
    target_venue: str = None,
    require_gate: bool = False,
    require_ready: bool = False,
    min_breakthrough: float = None,
    min_priority: float = None,
    max_blockers: int = None,
    min_rewrite_gain: float = None,
    max_fallbacks: int = None,
    max_strict_fallbacks: int = None,
    max_blocked_stages: int = None,
    max_missing_stages: int = None,
    max_attention_stages: int = None,
    min_stage_score: float = None,
    max_self_evolution_required_failures: int | None = 0,
    min_self_evolution_score: float | None = None,
    allow_blocked_self_evolution: bool = False,
    max_blocked_processes: int | None = 0,
    min_process_alignment_score: float | None = None,
) -> bool:
    if target_venue and paper.get("target_venue") != target_venue:
        return False
    if require_gate and paper.get("quality_gate_passed") is not True:
        return False
    if require_ready and paper.get("submission_status") != "ready":
        return False
    if (
        min_breakthrough is not None
        and (paper.get("breakthrough_score") or 0) < min_breakthrough
    ):
        return False
    if (
        min_priority is not None
        and (paper.get("submission_priority_score") or 0) < min_priority
    ):
        return False
    if (
        max_blockers is not None
        and isinstance(paper.get("blocker_count"), int)
        and paper.get("blocker_count") > max_blockers
    ):
        return False
    if (
        min_rewrite_gain is not None
        and (paper.get("rewrite_priority_gain_total") or 0) < min_rewrite_gain
    ):
        return False
    if (
        max_fallbacks is not None
        and isinstance(paper.get("fallback_count"), int)
        and paper.get("fallback_count") > max_fallbacks
    ):
        return False
    if (
        max_strict_fallbacks is not None
        and isinstance(paper.get("strict_fallback_count"), int)
        and paper.get("strict_fallback_count") > max_strict_fallbacks
    ):
        return False
    if (
        max_blocked_stages is not None
        and isinstance(paper.get("blocked_stage_count"), int)
        and paper.get("blocked_stage_count") > max_blocked_stages
    ):
        return False
    if (
        max_missing_stages is not None
        and isinstance(paper.get("missing_stage_count"), int)
        and paper.get("missing_stage_count") > max_missing_stages
    ):
        return False
    if (
        max_attention_stages is not None
        and isinstance(paper.get("needs_attention_stage_count"), int)
        and paper.get("needs_attention_stage_count") > max_attention_stages
    ):
        return False
    if (
        min_stage_score is not None
        and isinstance(paper.get("stage_overall_score"), (int, float))
        and float(paper.get("stage_overall_score")) < min_stage_score
    ):
        return False
    if (
        not allow_blocked_self_evolution
        and str(paper.get("self_evolution_status") or "").strip() == "blocked"
    ):
        return False
    if (
        max_self_evolution_required_failures is not None
        and isinstance(paper.get("self_evolution_required_failure_count"), int)
        and paper.get("self_evolution_required_failure_count")
        > max_self_evolution_required_failures
    ):
        return False
    if (
        min_self_evolution_score is not None
        and isinstance(paper.get("self_evolution_score"), (int, float))
        and float(paper.get("self_evolution_score")) < min_self_evolution_score
    ):
        return False
    if (
        max_blocked_processes is not None
        and isinstance(paper.get("process_alignment_blocked_process_count"), int)
        and paper.get("process_alignment_blocked_process_count") > max_blocked_processes
    ):
        return False
    if (
        min_process_alignment_score is not None
        and isinstance(paper.get("process_alignment_overall_score"), (int, float))
        and float(paper.get("process_alignment_overall_score"))
        < min_process_alignment_score
    ):
        return False
    return True


def _rewrite_board_sort_key(paper: Dict):
    gate_ready = paper.get("self_review_round_gate_ready")
    gate_score = (
        float(paper.get("self_review_round_gate_score"))
        if isinstance(paper.get("self_review_round_gate_score"), (int, float))
        else (100.0 if gate_ready is True else 0.0 if gate_ready is False else 50.0)
    )
    gate_deficit = max(0.0, 100.0 - gate_score)
    unresolved_critical = (
        int(paper.get("self_review_unresolved_critical"))
        if isinstance(paper.get("self_review_unresolved_critical"), int)
        else 0
    )
    high_value_coverage = (
        float(paper.get("self_review_high_value_coverage"))
        if isinstance(paper.get("self_review_high_value_coverage"), (int, float))
        else 1.0
    )
    high_value_gap = max(0.0, 1.0 - high_value_coverage)
    focus_issue_count = (
        int(paper.get("self_review_focus_issue_count"))
        if isinstance(paper.get("self_review_focus_issue_count"), int)
        else 0
    )
    review_active_issue_count = (
        int(paper.get("review_active_issue_count"))
        if isinstance(paper.get("review_active_issue_count"), int)
        else 0
    )
    review_unbound_issue_count = (
        int(paper.get("review_unbound_issue_count"))
        if isinstance(paper.get("review_unbound_issue_count"), int)
        else 0
    )
    review_persistent_issue_count = (
        int(paper.get("review_persistent_issue_count"))
        if isinstance(paper.get("review_persistent_issue_count"), int)
        else 0
    )
    review_active_binding_coverage = (
        float(paper.get("review_active_binding_coverage"))
        if isinstance(paper.get("review_active_binding_coverage"), (int, float))
        else 1.0
    )
    review_repair_ready_coverage = (
        float(paper.get("review_repair_ready_coverage"))
        if isinstance(paper.get("review_repair_ready_coverage"), (int, float))
        else 0.0
    )
    review_resolution_rate = (
        float(paper.get("review_resolution_rate"))
        if isinstance(paper.get("review_resolution_rate"), (int, float))
        else 0.0
    )
    experiment_todo_count = (
        int(paper.get("experiment_todo_count"))
        if isinstance(paper.get("experiment_todo_count"), int)
        else 0
    )
    experiment_todo_p0_count = (
        int(paper.get("experiment_todo_p0_count"))
        if isinstance(paper.get("experiment_todo_p0_count"), int)
        else 0
    )
    experiment_todo_closure_rate = (
        float(paper.get("experiment_todo_closure_rate"))
        if isinstance(paper.get("experiment_todo_closure_rate"), (int, float))
        else 1.0
    )
    experiment_todo_closure_gap = max(0.0, 1.0 - experiment_todo_closure_rate)
    return (
        paper.get("submission_status") != "ready",
        gate_ready is False,
        gate_deficit,
        unresolved_critical,
        high_value_gap,
        review_persistent_issue_count,
        review_active_issue_count,
        review_unbound_issue_count,
        max(0.0, 1.0 - review_repair_ready_coverage),
        max(0.0, 1.0 - review_active_binding_coverage),
        max(0.0, 1.0 - review_resolution_rate),
        focus_issue_count,
        experiment_todo_p0_count,
        experiment_todo_count,
        experiment_todo_closure_gap,
        (
            paper.get("rewrite_priority_gain_total")
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float))
            else -999
        ),
        (
            paper.get("rewrite_best_round_priority_delta")
            if isinstance(paper.get("rewrite_best_round_priority_delta"), (int, float))
            else -999
        ),
        (
            paper.get("submission_priority_score")
            if isinstance(paper.get("submission_priority_score"), (int, float))
            else -1
        ),
        -(
            paper.get("blocker_count")
            if isinstance(paper.get("blocker_count"), int)
            else 999
        ),
        (
            paper.get("quality_score")
            if isinstance(paper.get("quality_score"), (int, float))
            else -1
        ),
        paper.get("modified_at", ""),
    )


def _suggest_rewrite_next_step(paper: Dict) -> str:
    gate_ready = paper.get("self_review_round_gate_ready")
    gate_reasons = paper.get("self_review_round_gate_reasons") or []
    next_focus = paper.get("self_review_next_focus") or []
    unresolved_critical = paper.get("self_review_unresolved_critical")
    experiment_todo_count = (
        int(paper.get("experiment_todo_count"))
        if isinstance(paper.get("experiment_todo_count"), int)
        else 0
    )
    experiment_todo_p0_count = (
        int(paper.get("experiment_todo_p0_count"))
        if isinstance(paper.get("experiment_todo_p0_count"), int)
        else 0
    )
    experiment_todo_top_action = str(
        paper.get("experiment_todo_top_action") or ""
    ).strip()
    review_persistent_issue_count = (
        int(paper.get("review_persistent_issue_count"))
        if isinstance(paper.get("review_persistent_issue_count"), int)
        else 0
    )
    review_active_issue_count = (
        int(paper.get("review_active_issue_count"))
        if isinstance(paper.get("review_active_issue_count"), int)
        else 0
    )
    review_unbound_issue_count = (
        int(paper.get("review_unbound_issue_count"))
        if isinstance(paper.get("review_unbound_issue_count"), int)
        else 0
    )
    review_resolution_rate = (
        float(paper.get("review_resolution_rate"))
        if isinstance(paper.get("review_resolution_rate"), (int, float))
        else None
    )
    review_repair_ready_coverage = (
        float(paper.get("review_repair_ready_coverage"))
        if isinstance(paper.get("review_repair_ready_coverage"), (int, float))
        else None
    )
    experiment_todo_closure_rate = (
        float(paper.get("experiment_todo_closure_rate"))
        if isinstance(paper.get("experiment_todo_closure_rate"), (int, float))
        else None
    )
    if gate_ready is False:
        if isinstance(unresolved_critical, int) and unresolved_critical > 0:
            return "Round gate still reports unresolved critical issues; fix critical soundness/evidence gaps before polish."
        if "high_value_coverage_low" in gate_reasons:
            if next_focus:
                return f"Prioritize unresolved high-value issues first: {next_focus[0]}"
            return "High-value issue coverage is still low; target P0/P1 issues before broad rewrites."
        if "rewrite_coverage_low" in gate_reasons:
            return "Increase issue-linked rewrite coverage and ensure addressed_issue_ids match recommended targets."
        if "persistent_issues_high" in gate_reasons and next_focus:
            return f"Persistent issues remain; continue with focused repair on: {next_focus[0]}"
        if next_focus:
            return f"Continue round-gate focus: {next_focus[0]}"
    if review_persistent_issue_count > 0:
        return "Reviewer issues are persisting across rounds; do focused issue-by-issue repairs with explicit verification before broader rewrites."
    if (
        review_active_issue_count > 0
        and review_repair_ready_coverage is not None
        and review_repair_ready_coverage < 1.0
    ):
        return "Some active reviewer issues still lack fully executable repair tasks; expand the repair queue with concrete actions and target-specific verification."
    if review_active_issue_count > 0 and review_unbound_issue_count > 0:
        return "Some reviewer issues are not yet mapped to a claim, figure, or section; bind them first so the next repair round is targeted."
    if (
        review_active_issue_count > 0
        and review_resolution_rate is not None
        and review_resolution_rate < 0.35
    ):
        return "Reviewer debt is still high relative to resolved issues; prioritize concrete fixes and evidence checks over stylistic polish."
    if experiment_todo_p0_count > 0 and experiment_todo_top_action:
        return (
            f"Execute the highest-priority experiment TODO first: "
            f"{experiment_todo_top_action}"
        )
    if experiment_todo_p0_count > 0:
        return "Resolve P0 experiment TODO items before broad stylistic rewrites."
    if experiment_todo_count > 0 and experiment_todo_top_action:
        return f"Start with open experiment TODO item: " f"{experiment_todo_top_action}"
    if (
        experiment_todo_count > 0
        and isinstance(experiment_todo_closure_rate, float)
        and experiment_todo_closure_rate < 0.5
    ):
        return "Experiment TODO closure rate remains low; prioritize one measurable evidence task this round."
    if paper.get("submission_status") == "ready":
        return "Ready or near-ready; do final polish and package review."
    if paper.get("rewrite_top_section") and paper.get("rewrite_top_frontmatter_style"):
        return (
            f"Continue with {paper.get('rewrite_top_section')} using the "
            f"'{paper.get('rewrite_top_frontmatter_style')}' frontmatter framing style."
        )
    if paper.get("rewrite_top_section"):
        return f"Continue targeted rewriting on {paper.get('rewrite_top_section')}."
    if isinstance(paper.get("blocker_count"), int) and paper.get("blocker_count") > 4:
        return "Too many blockers remain; reduce blockers before spending more rewrite budget."
    if (
        isinstance(paper.get("rewrite_priority_gain_total"), (int, float))
        and paper.get("rewrite_priority_gain_total") > 1.0
    ):
        return "One more rewrite pass looks worthwhile; recent rewrites are still improving submission priority."
    if gate_ready is False:
        return "Round gate not yet ready; prioritize unresolved high-value self-review issues."
    return "Review the risk-language and claim-softening plans before the next rewrite pass."
