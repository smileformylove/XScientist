"""Markdown formatters for daemon operational artifacts."""

from __future__ import annotations

from typing import Any


def _build_source_runtime_board_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Source Runtime Board", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('name')}",
                f"- Availability: {row.get('availability_state')} ({row.get('availability_reason') or 'ok'})",
                f"- Priority: {row.get('priority')}",
                f"- Current Daypart: {row.get('current_daypart')}",
                f"- Time Preference: {row.get('time_of_day_preference')}",
                f"- Target Venue: {row.get('target_venue')}",
                f"- Paper Types: {', '.join(row.get('paper_types') or [])}",
                f"- Source Archetype: {row.get('source_archetype_label')} ({row.get('source_archetype')})",
                f"- Batch Profile: {row.get('batch_profile_label')} ({row.get('batch_profile')})",
                f"- Resolved Workflow Mode: {row.get('resolved_workflow_mode')}",
                f"- Compatible Workflow Modes: {', '.join(row.get('compatible_workflow_modes') or [])}",
                f"- Preferred Policy: {row.get('preferred_execution_policy') or 'n/a'}",
                f"- Workflow Alignment: {row.get('workflow_alignment_score')} ({row.get('workflow_alignment_reason')})",
                f"- Batch Goal: {row.get('batch_goal')}",
                f"- Recommended Defaults: {', '.join(row.get('recommended_generator_preview') or []) or 'n/a'}",
                f"- Alignment Tags: {', '.join(row.get('alignment_tags') or []) or 'n/a'}",
                f"- Inspirations: {', '.join(row.get('archetype_inspirations') or []) or 'n/a'}",
                f"- Planning Notes: {row.get('planning_notes') or 'n/a'}",
                f"- Num Ideas: {row.get('num_ideas')}",
                f"- Cycles Today: {row.get('cycles_today')}",
                f"- Successes Today: {row.get('successes_today')}",
                f"- Cooldown Until Cycle: {row.get('cooldown_until_cycle')}",
                f"- Last Selected: {row.get('last_selected_at')}",
                f"- Last Finished: {row.get('last_finished_at')}",
                f"- Suggested Action: {row.get('suggested_action')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_source_health_board_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Source Health Board", ""]
    for row in sorted(rows, key=lambda item: item.get("health_score", 0), reverse=True):
        lines.append(
            f"- {row.get('name')} | health={row.get('health_score')} | align={row.get('workflow_alignment_score')} | state={row.get('availability_state')} | "
            f"workflow={row.get('resolved_workflow_mode')} | profile={row.get('batch_profile')} | "
            f"priority={row.get('priority')} | successes_today={row.get('successes_today')} | cycles_today={row.get('cycles_today')} | action={row.get('suggested_action')}"
        )
    return "\n".join(lines) + "\n"


def _build_source_batch_plan_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Source Batch Plan", ""]
    if not rows:
        lines.extend(["- No batch plan items available yet.", ""])
        return "\n".join(lines)
    for row in rows:
        lines.extend(
            [
                f"## {row.get('source')} [{row.get('tier')}]",
                f"- Availability: {row.get('availability_state')}",
                f"- Workflow Mode: {row.get('resolved_workflow_mode')}",
                f"- Source Archetype: {row.get('source_archetype_label')} ({row.get('source_archetype')})",
                f"- Batch Profile: {row.get('batch_profile_label')} ({row.get('batch_profile')})",
                f"- Batch Goal: {row.get('batch_goal')}",
                f"- Workflow Alignment: {row.get('workflow_alignment_score')}",
                f"- Health Score: {row.get('health_score')}",
                f"- Recommended Defaults: {', '.join(row.get('recommended_generator_preview') or []) or 'n/a'}",
                f"- Alignment Tags: {', '.join(row.get('alignment_tags') or []) or 'n/a'}",
                f"- Inspirations: {', '.join(row.get('archetype_inspirations') or []) or 'n/a'}",
                f"- Planning Notes: {row.get('planning_notes') or 'n/a'}",
                f"- Recommendation: {row.get('recommendation')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_source_next_batch_markdown(advisory: dict[str, Any]) -> str:
    lines = ["# Next Batch Source Mix", ""]
    summary = advisory.get("summary") or {}
    cadence = advisory.get("cadence") or {}
    lines.append(f"- Desired Policy: {advisory.get('desired_policy') or 'n/a'}")
    lines.append(f"- Source Count: {summary.get('source_count')}")
    lines.append(f"- Dominant Archetype: {summary.get('dominant_archetype')}")
    lines.append(f"- Dominant Workflow: {summary.get('dominant_workflow_mode')}")
    lines.append(
        f"- Cadence: {cadence.get('label') or 'n/a'} | {cadence.get('reason') or 'n/a'}"
    )
    lines.append("")
    if advisory.get("slots"):
        for slot in advisory.get("slots", []):
            lines.extend(
                [
                    f"## {slot.get('source')} [{slot.get('lane')}]",
                    f"- Share: {slot.get('share')}",
                    f"- Availability: {slot.get('availability_state') or 'n/a'}",
                    f"- Health Score: {slot.get('health_score')}",
                    f"- Workflow Alignment: {slot.get('workflow_alignment_score')}",
                    f"- Workflow Mode: {slot.get('source_workflow_mode')}",
                    f"- Source Archetype: {slot.get('source_archetype')}",
                    f"- Batch Profile: {slot.get('source_batch_profile')}",
                    f"- Target Venue: {slot.get('target_venue')}",
                    f"- Ready / Gate: {slot.get('ready_count')} / {slot.get('gate_pass_count')}",
                    f"- Avg Submission Priority: {slot.get('avg_submission_priority')}",
                    f"- Recommended Defaults: {', '.join(slot.get('recommended_generator_preview') or []) or 'n/a'}",
                    f"- Alignment Tags: {', '.join(slot.get('alignment_tags') or []) or 'n/a'}",
                    f"- Focus: {slot.get('focus')}",
                    f"- Rationale: {slot.get('rationale')}",
                    f"- Planning Notes: {slot.get('planning_notes') or 'n/a'}",
                    "",
                ]
            )
    else:
        lines.extend(["- No next-batch slots available yet.", ""])
    if advisory.get("recommendations"):
        lines.append("## Mix Recommendations")
        for item in advisory.get("recommendations", [])[:5]:
            lines.append(f"- [{item.get('tier')}] {item.get('recommendation')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_cycle_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Daemon Cycle Summary",
        "",
        f"- Generated at: {summary.get('generated_at')}",
        f"- Health: {summary.get('health', {}).get('score')} ({summary.get('health', {}).get('state')})",
        f"- Guardrail phase: {summary.get('guardrail_phase')} ({summary.get('guardrail_phase_reason')})",
        f"- Guardrail mode: {summary.get('guardrail_mode')} ({summary.get('guardrail_reason')})",
        f"- Success count: {summary.get('success_count')}",
        f"- Failure count: {summary.get('failure_count')}",
        f"- Submission board items: {summary.get('submission_board_items')}",
        f"- Rewrite board items: {summary.get('rewrite_board_items')}",
        f"- Shortlist items: {summary.get('shortlist_items')}",
        f"- Follow-up count: {summary.get('followup_count')}",
        f"- Follow-up avg priority delta: {summary.get('followup_avg_priority_delta')}",
        f"- Follow-up improved count: {summary.get('followup_improved_count')}",
        f"- Active source TODO closure: {summary.get('active_source_todo_closure_rate')}",
        f"- Active source TODO backlog: {summary.get('active_source_todo_backlog')}",
        f"- Active source TODO P0 backlog: {summary.get('active_source_todo_p0_backlog')}",
        f"- Health recommendation: {summary.get('health', {}).get('recommendation')}",
        "",
        "## Health Signals",
    ]
    for item in summary.get("health", {}).get("reasons", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Latest Artifacts",
            f"- Submission board: {summary.get('last_views', {}).get('submission_board')}",
            f"- Rewrite board: {summary.get('last_views', {}).get('rewrite_board')}",
            f"- Shortlist: {summary.get('last_views', {}).get('shortlist')}",
            f"- Source runtime board: {summary.get('source_runtime_board')}",
            f"- Source health board: {summary.get('source_health_board')}",
            f"- Source batch plan: {summary.get('source_batch_plan')}",
            f"- Source next batch: {summary.get('source_next_batch')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_autonomy_program_markdown(program: dict[str, Any]) -> str:
    lines = [
        "# Autonomous Research Program",
        "",
        f"- Generated at: {program.get('generated_at')}",
        f"- Goal: {program.get('goal')}",
        f"- Primary target venue: {program.get('primary_target_venue')}",
        "",
        "## Fixed Evaluation Harness",
    ]
    for item in program.get("fixed_evaluation_harness", []):
        lines.append(f"- {item}")

    active_source = program.get("current_active_source") or {}
    lines.extend(["", "## Active Source"])
    lines.append(f"- Name: {active_source.get('name')}")
    lines.append(f"- Type: {active_source.get('type')}")
    lines.append(f"- Value: {active_source.get('value')}")
    lines.append(f"- Target venue: {active_source.get('target_venue')}")
    lines.append(
        f"- Paper types: {', '.join(active_source.get('paper_types') or []) or 'n/a'}"
    )

    lines.extend(["", "## Automation Stack"])
    for key, value in (program.get("automation_stack") or {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Current Strategies"])
    strategies = program.get("current_strategies") or {}
    lines.append(f"- Guardrail phase: {strategies.get('guardrail_phase')}")
    lines.append(f"- Guardrail mode: {strategies.get('guardrail_mode')}")
    lines.append(
        f"- Quality strategy: {(strategies.get('quality_strategy') or {}).get('mode')} ({(strategies.get('quality_strategy') or {}).get('reason')})"
    )
    lines.append(
        f"- Evidence strategy: {(strategies.get('evidence_strategy') or {}).get('mode')} ({(strategies.get('evidence_strategy') or {}).get('reason')})"
    )
    lines.append(
        f"- Quality governor: {(strategies.get('quality_governor') or {}).get('mode')} ({(strategies.get('quality_governor') or {}).get('reason')})"
    )

    feedback = program.get("source_feedback_snapshot") or {}
    lines.extend(["", "## Source Feedback Snapshot"])
    if feedback:
        for key in [
            "count",
            "avg_priority",
            "gate_pass_rate",
            "ready_rate",
            "avg_claim_support",
            "avg_claim_alignment",
            "avg_numeric_coverage",
            "avg_evidence_density",
            "avg_unsupported_claims",
            "avg_experiment_todo",
            "avg_experiment_todo_p0",
            "avg_experiment_todo_closure_rate",
            "priority_bonus",
            "dominant_venue",
            "dominant_paper_type",
        ]:
            lines.append(f"- {key}: {feedback.get(key)}")
    else:
        lines.append("- No source feedback available yet.")

    lines.extend(["", "## Keep Criteria"])
    for item in program.get("keep_criteria", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Discard Criteria"])
    for item in program.get("discard_criteria", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Adjustable Levers"])
    for item in program.get("adjustable_levers", []):
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def _build_primary_action_queue_markdown(queue: list[dict[str, Any]]) -> str:
    lines = ["# Primary Action Queue", ""]
    if queue:
        for item in queue[:10]:
            source = f" | source={item.get('source')}" if item.get("source") else ""
            lines.append(
                f"- [{item.get('priority')}] {item.get('label')} | category={item.get('category')}{source} | {item.get('reason')} | command=`{item.get('command')}`"
            )
    else:
        lines.append("- No prioritized commands available.")
    return "\n".join(lines) + "\n"


def _build_operator_brief_markdown(brief: dict[str, Any]) -> str:
    lines = [
        "# Operator Brief",
        "",
        f"- Generated at: {brief.get('generated_at')}",
        f"- Health: {brief.get('health', {}).get('score')} ({brief.get('health', {}).get('state')})",
        f"- Active source: {brief.get('active_source')}",
        f"- Daypart: {brief.get('current_daypart')}",
        f"- Phase: {brief.get('guardrail_phase')}",
        f"- Mode: {brief.get('guardrail_mode')} ({brief.get('guardrail_reason')})",
        f"- Success count: {brief.get('success_count')}",
        f"- Failure count: {brief.get('failure_count')}",
        "",
        "## Priorities",
    ]
    for item in brief.get("priorities", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Do Now"])
    if brief.get("do_now_actions"):
        for item in brief.get("do_now_actions", [])[:3]:
            command = item.get("command")
            if command:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')} | command=`{command}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No immediate source actions right now.")

    lines.extend(["", "## Primary Commands"])
    if brief.get("primary_action_queue"):
        for item in brief.get("primary_action_queue", [])[:5]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('label')} | {item.get('reason')} | command=`{item.get('command')}`"
            )
    else:
        lines.append("- No prioritized commands available.")

    lines.extend(["", "## Health Signals"])
    for item in brief.get("health", {}).get("reasons", []):
        lines.append(f"- {item}")
    lines.append(f"- Recommendation: {brief.get('health', {}).get('recommendation')}")
    lines.extend(["", "## Immediate Actions"])
    for item in brief.get("actions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Source Health Snapshot"])
    for row in (brief.get("source_runtime_rows") or [])[:5]:
        lines.append(
            f"- {row.get('name')} | health={row.get('health_score')} | state={row.get('availability_state')} | venue={row.get('target_venue')} | paper_types={','.join(row.get('paper_types') or [])}"
        )

    lines.extend(["", "## Source Actions"])
    for row in (brief.get("source_actions") or [])[:5]:
        lines.append(
            f"- {row.get('name')} | state={row.get('availability_state')} | health={row.get('health_score')} | action={row.get('suggested_action')}"
        )

    lines.extend(["", "## Source Plan"])
    if brief.get("source_advisory"):
        for item in brief.get("source_advisory", [])[:6]:
            command = item.get("command")
            if command:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | state={item.get('state')} | health={item.get('health_score')} | {item.get('recommendation')} | command=`{command}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | state={item.get('state')} | health={item.get('health_score')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No source advisory recommendations yet.")

    lines.extend(["", "## Source Mix"])
    source_mix = brief.get("source_mix_advisory") or {}
    mix_summary = source_mix.get("summary") or {}
    lines.append(
        f"- desired_policy={source_mix.get('desired_policy') or 'n/a'} | dominant_archetype={mix_summary.get('dominant_archetype')} | dominant_workflow={mix_summary.get('dominant_workflow_mode')}"
    )
    lines.append(
        f"- archetype_counts={mix_summary.get('archetype_counts')} | workflow_counts={mix_summary.get('workflow_mode_counts')}"
    )
    if source_mix.get("recommendations"):
        for item in source_mix.get("recommendations", [])[:5]:
            lines.append(f"- [{item.get('tier')}] {item.get('recommendation')}")
    else:
        lines.append("- No source mix recommendations yet.")

    lines.extend(["", "## Next Batch Recipe"])
    next_batch = brief.get("source_next_batch_advisory") or {}
    next_batch_cadence = next_batch.get("cadence") or {}
    lines.append(
        f"- cadence={next_batch_cadence.get('label') or 'n/a'} | {next_batch_cadence.get('reason') or 'n/a'}"
    )
    if next_batch.get("slots"):
        for slot in next_batch.get("slots", [])[:5]:
            lines.append(
                f"- [{slot.get('lane')}] {slot.get('source')} | share={slot.get('share')} | state={slot.get('availability_state') or 'n/a'} | health={slot.get('health_score')} | workflow={slot.get('source_workflow_mode')} | {slot.get('rationale')}"
            )
    else:
        lines.append("- No next-batch source slots yet.")

    lines.extend(["", "## Evidence Strategy"])
    evidence_strategy = brief.get("evidence_strategy") or {}
    if evidence_strategy.get("enabled"):
        lines.append(
            f"- mode={evidence_strategy.get('mode')} | reason={evidence_strategy.get('reason')}"
        )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Pipeline Contract Strategy"])
    pipeline_strategy = brief.get("pipeline_contract_strategy") or {}
    if pipeline_strategy.get("enabled"):
        lines.append(
            f"- mode={pipeline_strategy.get('mode')} | reason={pipeline_strategy.get('reason')}"
        )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Quality Governor"])
    quality_governor = brief.get("quality_governor") or {}
    if quality_governor.get("enabled"):
        lines.append(
            f"- mode={quality_governor.get('mode')} | rewrite_top_k={quality_governor.get('rewrite_followup_top_k_effective')} | dossier_top_k={quality_governor.get('auto_submission_dossier_top_k_effective')} | source_plan_max_actions={quality_governor.get('auto_source_plan_max_actions_effective')}"
        )
        lines.append(f"- reason={quality_governor.get('reason')}")
    else:
        lines.append("- disabled")

    lines.extend(["", "## Auto Source Plan"])
    auto_source_plan = brief.get("auto_source_plan") or {}
    if auto_source_plan.get("enabled"):
        lines.append(
            f"- enabled | min_health={auto_source_plan.get('min_health')} | max_actions={auto_source_plan.get('max_actions')} | expires_after_cycles={auto_source_plan.get('expires_after_cycles')}"
        )
        if auto_source_plan.get("applied"):
            for item in auto_source_plan.get("applied", [])[:5]:
                lines.append(
                    f"- applied {item.get('operation')} to {item.get('source')} | health={item.get('health_score')} | tier={item.get('tier')}"
                )
        else:
            lines.append(
                f"- no action applied | reason={auto_source_plan.get('skipped_reason')}"
            )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Submission Autopilot"])
    submission_autopilot = brief.get("submission_autopilot") or {}
    if submission_autopilot.get("enabled"):
        lines.append(
            f"- enabled | top_k={submission_autopilot.get('top_k')} | require_ready={submission_autopilot.get('require_ready')} | require_gate={submission_autopilot.get('require_gate')}"
        )
        if submission_autopilot.get("exported"):
            for item in submission_autopilot.get("exported", [])[:5]:
                lines.append(
                    f"- exported {item.get('folder')} | priority={item.get('priority')} | dossier={item.get('output_dir')}"
                )
        elif submission_autopilot.get("reused"):
            for item in submission_autopilot.get("reused", [])[:5]:
                lines.append(
                    f"- reused {item.get('folder')} | priority={item.get('priority')} | dossier={item.get('output_dir')}"
                )
        else:
            lines.append(
                f"- no dossier exported | reason={submission_autopilot.get('skipped_reason')}"
            )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Failure Guard"])
    failure_guard = brief.get("failure_guard") or {}
    if failure_guard.get("enabled"):
        lines.append(
            f"- source={failure_guard.get('source')} | consecutive_failures={failure_guard.get('consecutive_failures')} | threshold={failure_guard.get('threshold')} | cooldown={failure_guard.get('cooldown_cycles')} | applied={failure_guard.get('applied')}"
        )
        if failure_guard.get("reason"):
            lines.append(f"- reason={failure_guard.get('reason')}")
    else:
        lines.append("- disabled")

    lines.extend(["", "## Control Summary"])
    for item in brief.get("control_summary") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Recent Control Events"])
    if brief.get("recent_control_events"):
        for item in brief.get("recent_control_events", []):
            lines.append(
                f"- {item.get('type')} | source={item.get('matched_key') or item.get('active_source')} | at={item.get('timestamp')}"
            )
    else:
        lines.append("- No recent control events recorded.")

    lines.extend(["", "## Recent Failure Hotspots"])
    if brief.get("failure_hotspots"):
        for item in brief.get("failure_hotspots", []):
            lines.append(f"- {item.get('reason')}: {item.get('count')}")
    else:
        lines.append("- No recent failure hotspots detected.")

    lines.extend(["", "## Rewrite Style Hotspots"])
    if brief.get("rewrite_style_hotspots"):
        for item in brief.get("rewrite_style_hotspots", []):
            lines.append(f"- {item.get('style')}: {item.get('score')}")
    else:
        lines.append("- No rewrite style hotspot signals yet.")

    lines.extend(["", "## Pipeline Contracts"])
    pipeline_contracts = brief.get("pipeline_contracts") or {}
    if pipeline_contracts.get("enabled"):
        lines.append(
            f"- project_count={pipeline_contracts.get('project_count')} | blocked_projects={pipeline_contracts.get('blocked_project_count')} | stage_blocked_projects={pipeline_contracts.get('stage_blocked_project_count')} | stage_missing_projects={pipeline_contracts.get('stage_missing_project_count')} | review_low_resolution_projects={pipeline_contracts.get('review_low_resolution_project_count')} | review_low_binding_projects={pipeline_contracts.get('review_low_binding_project_count')} | review_low_repair_ready_projects={pipeline_contracts.get('review_low_repair_ready_project_count')} | failed_projects={pipeline_contracts.get('failed_project_count')} | blocked_figures={pipeline_contracts.get('blocked_figure_count')} | failed_experiments={pipeline_contracts.get('failed_experiment_count')} | budget_exhausted={pipeline_contracts.get('budget_exhausted_experiment_count')}"
        )
        if pipeline_contracts.get("dominant_execution_policy"):
            lines.append(
                f"- dominant execution policy: {pipeline_contracts.get('dominant_execution_policy')}"
            )
        if pipeline_contracts.get("avg_stage_overall_score") is not None:
            lines.append(
                f"- average stage-standard score: {pipeline_contracts.get('avg_stage_overall_score')}"
            )
        if pipeline_contracts.get("avg_review_resolution_rate") is not None:
            lines.append(
                f"- average reviewer-resolution rate: {pipeline_contracts.get('avg_review_resolution_rate')}"
            )
        if pipeline_contracts.get("avg_review_binding_coverage") is not None:
            lines.append(
                f"- average reviewer target-binding coverage: {pipeline_contracts.get('avg_review_binding_coverage')}"
            )
        if pipeline_contracts.get("avg_review_repair_ready_coverage") is not None:
            lines.append(
                f"- average reviewer repair-ready coverage: {pipeline_contracts.get('avg_review_repair_ready_coverage')}"
            )
        artifact_blockers = pipeline_contracts.get("artifact_blockers") or {}
        if artifact_blockers:
            lines.append(
                "- artifact blockers: "
                + ", ".join(
                    f"{name}={count}" for name, count in artifact_blockers.items()
                )
            )
        stage_risks = pipeline_contracts.get("top_stage_standard_risks") or {}
        if stage_risks:
            lines.append(
                "- stage-standard risks: "
                + ", ".join(f"{name}={count}" for name, count in stage_risks.items())
            )
        execution_policy_counts = (
            pipeline_contracts.get("execution_policy_counts") or {}
        )
        if execution_policy_counts:
            lines.append(
                "- execution policies: "
                + ", ".join(
                    f"{name}={count}" for name, count in execution_policy_counts.items()
                )
            )
        budget_status_counts = pipeline_contracts.get("budget_status_counts") or {}
        if budget_status_counts:
            lines.append(
                "- budget status: "
                + ", ".join(
                    f"{name}={count}" for name, count in budget_status_counts.items()
                )
            )
        for item in pipeline_contracts.get("top_blocked_projects") or []:
            lines.append(
                f"- {item.get('project')} | blocked={item.get('blocked_artifacts')} | failed={item.get('failed_artifacts')} | missing={item.get('missing_artifacts')} | "
                f"stage_score={item.get('stage_overall_score')} blocked_stages={item.get('blocked_stage_count')} attention_stages={item.get('needs_attention_stage_count')} missing_stages={item.get('missing_stage_count')} | "
                f"review_resolution={item.get('review_resolution_rate')} review_binding={item.get('review_target_binding_coverage')} repair_ready={item.get('review_repair_ready_coverage')} persistent_review_issues={item.get('review_persistent_issue_count')}"
            )
    else:
        lines.append("- No contract-enabled pipeline roots detected yet.")

    lines.extend(["", "## Active Blockers"])
    if brief.get("blockers"):
        for item in brief.get("blockers", []):
            lines.append(f"- {item}")
    else:
        lines.append("- No major blockers detected.")

    lines.extend(["", "## Top Submission Targets"])
    for item in brief.get("top_submission_targets", []):
        lines.append(
            f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | rewrite_gain={item.get('rewrite_gain')}"
        )
    if not brief.get("top_submission_targets"):
        lines.append("- No strong submission targets yet.")

    lines.extend(["", "## Top Rewrite Targets"])
    for item in brief.get("top_rewrite_targets", []):
        lines.append(
            f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | rewrite_gain={item.get('rewrite_gain')} | next={item.get('next_step')}"
        )
    if not brief.get("top_rewrite_targets"):
        lines.append("- No rewrite targets currently meet the bar.")

    lines.extend(["", "## Primary Commands"])
    if brief.get("primary_action_queue"):
        for item in brief.get("primary_action_queue", [])[:5]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('label')} | {item.get('reason')} | command=`{item.get('command')}`"
            )
    else:
        lines.append("- No prioritized commands available.")

    lines.extend(["", "## Recommended Commands"])
    for key, value in (brief.get("recommended_commands") or {}).items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Recent Follow-up Wins"])
    for item in brief.get("recent_followup_wins", []):
        lines.append(
            f"- {item.get('paper')} | priority_delta={item.get('priority_delta')} | priority_after={item.get('priority_after')}"
        )
    if not brief.get("recent_followup_wins"):
        lines.append("- No recent follow-up wins recorded.")

    return "\n".join(lines) + "\n"


def _build_handoff_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Handoff Report",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Health: {report.get('health_score')} ({report.get('health_state')})",
        f"- Phase: {report.get('phase')}",
        f"- Mode: {report.get('mode')}",
        f"- Active source: {report.get('active_source')}",
        f"- Daypart: {report.get('daypart')}",
        f"- Success count: {report.get('success_count')}",
        f"- Failure count: {report.get('failure_count')}",
        f"- Attention Label: {report.get('attention_label')}",
        "",
        "## Recovery",
        f"- Reason: {report.get('recovery_reason')}",
        (
            f"- Command: `{report.get('recovery_command')}`"
            if report.get("recovery_command")
            else "- Command: n/a"
        ),
        "",
        "## Do Now",
    ]
    if report.get("do_now_actions"):
        for item in report.get("do_now_actions", [])[:3]:
            if item.get("command"):
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')} | command=`{item.get('command')}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No immediate actions available.")
    lines.extend(["", "## Top Submission Targets"])
    if report.get("top_submission_targets"):
        for item in report.get("top_submission_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | rewrite_gain={item.get('rewrite_gain')}"
            )
    else:
        lines.append("- No strong submission targets yet.")
    lines.extend(["", "## Top Rewrite Targets"])
    if report.get("top_rewrite_targets"):
        for item in report.get("top_rewrite_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | next={item.get('next_step')}"
            )
    else:
        lines.append("- No active rewrite targets.")
    lines.extend(["", "## Blockers"])
    if report.get("blockers"):
        for item in report.get("blockers"):
            lines.append(f"- {item}")
    else:
        lines.append("- No major blockers detected.")
    lines.extend(["", "## Recommended Commands"])
    for key, value in (report.get("recommended_commands") or {}).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def _build_daily_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Daily Report",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Report date: {report.get('report_date')}",
        f"- Health: {report.get('health_score')} ({report.get('health_state')})",
        f"- Phase: {report.get('phase')}",
        f"- Mode: {report.get('mode')}",
        f"- Success count: {report.get('success_count')}",
        f"- Failure count: {report.get('failure_count')}",
        "",
        "## Do Now",
    ]
    if report.get("do_now_actions"):
        for item in report.get("do_now_actions", [])[:3]:
            if item.get("command"):
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')} | command=`{item.get('command')}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No immediate actions recorded.")
    lines.extend(["", "## Top Submission Targets"])
    if report.get("top_submission_targets"):
        for item in report.get("top_submission_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')}"
            )
    else:
        lines.append("- No strong submission targets yet.")
    lines.extend(["", "## Top Rewrite Targets"])
    if report.get("top_rewrite_targets"):
        for item in report.get("top_rewrite_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | next={item.get('next_step')}"
            )
    else:
        lines.append("- No active rewrite targets.")
    lines.extend(["", "## Blockers"])
    if report.get("blockers"):
        for item in report.get("blockers"):
            lines.append(f"- {item}")
    else:
        lines.append("- No major blockers detected.")
    lines.extend(["", "## Recovery"])
    lines.append(f"- Reason: {report.get('recovery_reason')}")
    lines.append(
        f"- Command: `{report.get('recovery_command')}`"
        if report.get("recovery_command")
        else "- Command: n/a"
    )
    return "\n".join(lines) + "\n"


def _build_report_archive_index_markdown(index: dict[str, Any]) -> str:
    lines = [
        "# Report Archive Index",
        "",
        f"- Generated at: {index.get('generated_at')}",
        f"- Daily reports: {(index.get('counts') or {}).get('daily')}",
        f"- Handoff reports: {(index.get('counts') or {}).get('handoff')}",
        "",
        "## Recent Entries",
    ]
    if index.get("entries"):
        for item in index.get("entries", [])[:20]:
            lines.append(
                f"- [{item.get('kind')}] {item.get('name')} | report_date={item.get('report_date')} | health={item.get('health_state')} | path={item.get('path')}"
            )
    else:
        lines.append("- No archived reports found.")
    return "\n".join(lines) + "\n"


def _build_report_archive_trends_markdown(trends: dict[str, Any]) -> str:
    lines = [
        "# Report Trends",
        "",
        f"- Generated at: {trends.get('generated_at')}",
        f"- Daily reports considered: {trends.get('daily_reports_considered')}",
        f"- Handoff reports considered: {trends.get('handoff_reports_considered')}",
        f"- Average daily health: {trends.get('average_daily_health_score')}",
        f"- Daily health delta: {trends.get('daily_health_delta')}",
        f"- Latest daily health: {trends.get('latest_daily_health_score')}",
        f"- Average TODO closure rate: {trends.get('average_todo_closure_rate')}",
        f"- Latest TODO closure rate: {trends.get('latest_todo_closure_rate')}",
        f"- TODO closure delta: {trends.get('todo_closure_delta')}",
        f"- Average TODO backlog: {trends.get('average_todo_backlog')}",
        f"- Latest TODO backlog: {trends.get('latest_todo_backlog')}",
        f"- Trend action: {trends.get('trend_action_label')}",
        "",
        "## Trend Action",
        f"- Reason: {trends.get('trend_action_reason')}",
        (
            f"- Command: `{trends.get('trend_action_command')}`"
            if trends.get("trend_action_command")
            else "- Command: n/a"
        ),
        "",
        "## Attention Labels",
    ]
    if trends.get("attention_label_counts"):
        for item in trends.get("attention_label_counts") or []:
            lines.append(f"- {item.get('label')}: {item.get('count')}")
    else:
        lines.append("- No handoff labels yet.")
    lines.extend(["", "## Recovery Reason Hotspots"])
    if trends.get("recovery_reason_hotspots"):
        for item in trends.get("recovery_reason_hotspots") or []:
            lines.append(f"- {item.get('reason')}: {item.get('count')}")
    else:
        lines.append("- No recovery hotspots yet.")
    lines.extend(["", "## Do-Now Source Hotspots"])
    if trends.get("do_now_source_hotspots"):
        for item in trends.get("do_now_source_hotspots") or []:
            lines.append(f"- {item.get('source')}: {item.get('count')}")
    else:
        lines.append("- No do-now source hotspots yet.")
    return "\n".join(lines) + "\n"


def _build_submission_autopilot_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Submission Autopilot", ""]
    lines.append(f"- Enabled: {summary.get('enabled')}")
    lines.append(f"- Top K: {summary.get('top_k')}")
    lines.append(f"- Require Gate: {summary.get('require_gate')}")
    lines.append(f"- Require Ready: {summary.get('require_ready')}")
    lines.append(f"- Min Priority: {summary.get('min_priority')}")
    lines.append(f"- Max Blockers: {summary.get('max_blockers')}")
    lines.append(f"- Min Rewrite Gain: {summary.get('min_rewrite_gain')}")
    if summary.get("skipped_reason"):
        lines.append(f"- Skipped: {summary.get('skipped_reason')}")
    lines.extend(["", "## Exported"])
    if summary.get("exported"):
        for item in summary.get("exported", [])[:10]:
            lines.append(
                f"- {item.get('folder')} | priority={item.get('priority')} | blockers={item.get('blockers')} | dossier={item.get('output_dir')}"
            )
    else:
        lines.append("- No dossiers exported in this cycle.")
    lines.extend(["", "## Reused"])
    if summary.get("reused"):
        for item in summary.get("reused", [])[:10]:
            lines.append(
                f"- {item.get('folder')} | priority={item.get('priority')} | dossier={item.get('output_dir')}"
            )
    else:
        lines.append("- No existing dossiers reused.")
    return "\n".join(lines) + "\n"


def _build_failure_guard_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Failure Guard", ""]
    lines.append(f"- Enabled: {summary.get('enabled')}")
    lines.append(f"- Source: {summary.get('source')}")
    lines.append(f"- Consecutive Failures: {summary.get('consecutive_failures')}")
    lines.append(f"- Threshold: {summary.get('threshold')}")
    lines.append(f"- Cooldown Cycles: {summary.get('cooldown_cycles')}")
    lines.append(f"- Applied: {summary.get('applied')}")
    if summary.get("reason"):
        lines.append(f"- Reason: {summary.get('reason')}")
    return "\n".join(lines) + "\n"
