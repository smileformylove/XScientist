"""Pure Markdown rendering for ResearchManager boards and shortlists."""

from __future__ import annotations

from typing import Dict, List


def render_submission_board_markdown(board: Dict[str, List[Dict]]) -> str:
    lines = ["# Submission Board", ""]
    for venue, papers in sorted(board.items()):
        lines.append(f"## {venue}")
        for paper in papers:
            lines.append(
                f"- {paper['name']} | priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                f"rewrite_gain={paper.get('rewrite_priority_gain_total')} | blockers={paper.get('blocker_count')} | "
                f"stage_score={paper.get('stage_overall_score')} blocked_stages={paper.get('blocked_stage_count')} "
                f"attention_stages={paper.get('needs_attention_stage_count')} missing_stages={paper.get('missing_stage_count')} | "
                f"self_evolution={paper.get('self_evolution_status')} score={paper.get('self_evolution_score')} "
                f"required_failures={paper.get('self_evolution_required_failure_count')} | "
                f"process_alignment={paper.get('process_alignment_overall_score')} blocked_processes={paper.get('process_alignment_blocked_process_count')} | "
                f"review_resolution={paper.get('review_resolution_rate')} review_binding={paper.get('review_target_binding_coverage')} "
                f"active_review_issues={paper.get('review_active_issue_count')} persistent_review_issues={paper.get('review_persistent_issue_count')} | "
                f"fallbacks={paper.get('fallback_count')} strict={paper.get('strict_fallback_count')} | quality={paper.get('quality_score')} | "
                f"rigor={paper.get('rigor_score')} | claim={paper.get('claim_support_score')} | "
                f"package={paper.get('submission_package_file')}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_rewrite_board_markdown(papers: List[Dict]) -> str:
    lines = ["# Rewrite Board", ""]
    for paper in papers:
        lines.extend(
            [
                f"## {paper.get('name')}",
                f"- Venue: {paper.get('target_venue')}",
                f"- Submission Priority: {paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')})",
                f"- Rewrite Gain: {paper.get('rewrite_priority_gain_total')}",
                f"- Best Round Delta: {paper.get('rewrite_best_round_priority_delta')}",
                f"- Rewrite Rounds: {paper.get('rewrite_round_count')}",
                f"- Self Review Rounds: {paper.get('self_review_rounds_completed')}",
                f"- Round Gate: ready={paper.get('self_review_round_gate_ready')} score={paper.get('self_review_round_gate_score')} unresolved_critical={paper.get('self_review_unresolved_critical')}",
                f"- Reviewer Repair: resolution={paper.get('review_resolution_rate')} active={paper.get('review_active_issue_count')} persistent={paper.get('review_persistent_issue_count')} checks={paper.get('review_verification_count')}",
                f"- Reviewer Repair Queue: queue={paper.get('review_repair_queue_count')} ready={paper.get('review_repair_ready_count')} ready_coverage={paper.get('review_repair_ready_coverage')} verification_ready={paper.get('review_repair_verification_ready_count')}",
                f"- Reviewer Target Binding: coverage={paper.get('review_target_binding_coverage')} active_coverage={paper.get('review_active_binding_coverage')} unbound={paper.get('review_unbound_issue_count')}",
                f"- Experiment TODO: total={paper.get('experiment_todo_count')} p0={paper.get('experiment_todo_p0_count')}",
                f"- Experiment TODO Progress: closed={paper.get('experiment_todo_closed_count')} unresolved={paper.get('experiment_todo_unresolved_count')} closure_rate={paper.get('experiment_todo_closure_rate')} p0_closure_rate={paper.get('experiment_todo_p0_closure_rate')}",
                f"- Experiment TODO Top Action: {paper.get('experiment_todo_top_action')}",
                f"- High-Value Coverage: {paper.get('self_review_high_value_coverage')}",
                f"- Top Section: {paper.get('rewrite_top_section')}",
                f"- Top Section Style: {paper.get('rewrite_top_section_style')}",
                f"- Top Frontmatter Style: {paper.get('rewrite_top_frontmatter_style')}",
                f"- Blockers: {paper.get('blocker_count')}",
                f"- Next Step: {paper.get('suggested_next_step')}",
                f"- Experiment TODO File: {paper.get('experiment_todo_file')}",
                f"- Experiment TODO Progress File: {paper.get('experiment_todo_progress_file')}",
                f"- Rewrite Effectiveness: {paper.get('rewrite_effectiveness_file')}",
                f"- Path: {paper.get('path')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def render_repair_board_markdown(rows: List[Dict]) -> str:
    lines = ["# Repair Board", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('name')} :: {row.get('repair_id')}",
                f"- Venue: {row.get('target_venue')}",
                f"- Project: {row.get('project')}",
                f"- Reviewer role: {row.get('role')}",
                f"- Priority: {row.get('priority_tier')} ({row.get('priority_score')})",
                f"- Status: {row.get('status')}",
                f"- Issue: {row.get('issue_text')}",
                f"- Target: {row.get('primary_target_type')} {row.get('primary_target_id')} ({row.get('primary_target_label')})",
                f"- Lane: {row.get('lane')} ({row.get('lane_label')})",
                f"- Blocking reasons: {', '.join(row.get('blocking_reasons') or []) or 'none'}",
                f"- Repair actions: {' | '.join(row.get('repair_actions') or []) or 'none'}",
                f"- Verification checks: {' | '.join(row.get('verification_checks') or []) or 'none'}",
                f"- Path: {row.get('project_root')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def render_shortlist_markdown(papers: List[Dict]) -> str:
    lines = ["# Submission Shortlist", ""]
    for paper in papers:
        lines.extend(
            [
                f"## {paper.get('name')}",
                f"- Type: {paper.get('type')}",
                f"- Venue: {paper.get('target_venue')}",
                f"- Submission Priority: {paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')})",
                f"- Rewrite Priority Gain: {paper.get('rewrite_priority_gain_total')}",
                f"- Blockers: {paper.get('blocker_count')}",
                f"- Stage Standards: score={paper.get('stage_overall_score')} blocked={paper.get('blocked_stage_count')} attention={paper.get('needs_attention_stage_count')} missing={paper.get('missing_stage_count')}",
                f"- Top Standard Risks: {', '.join(paper.get('top_standard_risks') or [])}",
                f"- Process Alignment: score={paper.get('process_alignment_overall_score')} blocked={paper.get('process_alignment_blocked_process_count')} attention={paper.get('process_alignment_attention_process_count')} missing={paper.get('process_alignment_missing_process_count')}",
                f"- Process Risks: {', '.join(paper.get('process_alignment_top_risks') or [])}",
                f"- Reviewer Repair: resolution={paper.get('review_resolution_rate')} active={paper.get('review_active_issue_count')} persistent={paper.get('review_persistent_issue_count')} checks={paper.get('review_verification_count')}",
                f"- Reviewer Repair Queue: queue={paper.get('review_repair_queue_count')} ready={paper.get('review_repair_ready_count')} ready_coverage={paper.get('review_repair_ready_coverage')} verification_ready={paper.get('review_repair_verification_ready_count')}",
                f"- Reviewer Target Binding: coverage={paper.get('review_target_binding_coverage')} active_coverage={paper.get('review_active_binding_coverage')} unbound={paper.get('review_unbound_issue_count')}",
                f"- Self-Evolution: status={paper.get('self_evolution_status')} score={paper.get('self_evolution_score')} required_failures={paper.get('self_evolution_required_failure_count')} lane={paper.get('self_evolution_dominant_lane')} role={paper.get('self_evolution_dominant_role')}",
                f"- Self-Evolution Risks: {', '.join(paper.get('self_evolution_top_risks') or [])}",
                f"- Quality: {paper.get('quality_score')}",
                f"- Breakthrough: {paper.get('breakthrough_score')}",
                f"- Rigor: {paper.get('rigor_score')}",
                f"- Claim Support: {paper.get('claim_support_score')}",
                f"- Numeric Coverage: {paper.get('numeric_coverage_score')}",
                f"- Contributions: {paper.get('contribution_count')}",
                f"- Gate Passed: {paper.get('quality_gate_passed')}",
                f"- Submission Status: {paper.get('submission_status')}",
                f"- Submission Package: {paper.get('submission_package_file')}",
                f"- Narrative Map: {paper.get('narrative_map_file')}",
                f"- Contribution Map: {paper.get('contribution_map_file')}",
                f"- Editor Pitch: {paper.get('editor_pitch_file')}",
                f"- Submission Dashboard: {paper.get('submission_dashboard_file')}",
                f"- Risk Language Plan: {paper.get('risk_language_plan_file')}",
                f"- Claim Softening Plan: {paper.get('claim_softening_plan_file')}",
                f"- Rewrite Effectiveness: {paper.get('rewrite_effectiveness_file')}",
                f"- Rewrite Best Round Delta: {paper.get('rewrite_best_round_priority_delta')}",
                f"- Rewrite Top Section: {paper.get('rewrite_top_section')}",
                f"- Risk Register: {paper.get('risk_register_file')}",
                f"- Path: {paper.get('path')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"
