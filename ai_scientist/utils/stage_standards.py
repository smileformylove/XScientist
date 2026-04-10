from __future__ import annotations

"""Structured stage-level evaluation standards for autonomous research runs."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.figure_spec import summarize_figure_spec
from ai_scientist.utils.pipeline_contracts import (
    artifact_path,
    load_contract_artifact,
    load_jsonl_artifact,
    save_contract_artifact,
)
from ai_scientist.utils.review_jobs import compute_review_repair_metrics

STAGE_ORDER = (
    "ideation",
    "planning",
    "experiment",
    "figure",
    "manuscript",
    "review",
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _criterion(
    criterion_id: str,
    label: str,
    *,
    passed: bool,
    required: bool = True,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "id": criterion_id,
        "label": label,
        "passed": bool(passed),
        "required": bool(required),
        "detail": str(detail or "").strip() or None,
    }


def _finalize_stage(
    stage: str,
    artifact: str,
    criteria: list[dict[str, Any]],
    *,
    signals: dict[str, Any] | None = None,
    missing_reason: str | None = None,
) -> dict[str, Any]:
    if missing_reason:
        return {
            "stage": stage,
            "artifact": artifact,
            "status": "missing",
            "score": 0.0,
            "criteria": [],
            "required_failures": [],
            "signals": signals or {},
            "missing_reason": missing_reason,
        }
    total = max(len(criteria), 1)
    passed = sum(1 for item in criteria if item.get("passed"))
    required_failures = [
        item["id"] for item in criteria if item.get("required") and not item.get("passed")
    ]
    score = round((passed / total) * 100.0, 1)
    if required_failures:
        status = "blocked"
    elif score < 80.0:
        status = "needs_attention"
    else:
        status = "ready"
    return {
        "stage": stage,
        "artifact": artifact,
        "status": status,
        "score": score,
        "criteria": criteria,
        "required_failures": required_failures,
        "signals": signals or {},
        "missing_reason": None,
    }


def _evaluate_ideation(idea_cards: Any) -> dict[str, Any]:
    cards = idea_cards if isinstance(idea_cards, list) else []
    if not cards:
        return _finalize_stage(
            "ideation",
            "idea_cards",
            [],
            missing_reason="idea_cards artifact missing or empty",
        )
    lead = cards[0] if isinstance(cards[0], dict) else {}
    criteria = [
        _criterion("idea_count", "At least one idea card exists", passed=len(cards) > 0, detail=f"count={len(cards)}"),
        _criterion(
            "core_hypothesis",
            "Lead idea defines a core hypothesis",
            passed=bool(str(lead.get("core_hypothesis") or "").strip()),
        ),
        _criterion(
            "novelty_claim",
            "Lead idea has novelty or related-work positioning",
            passed=bool(
                str(lead.get("novelty_claim") or "").strip()
                or str(lead.get("related_work_notes") or "").strip()
            ),
        ),
        _criterion(
            "minimum_experiment",
            "Lead idea specifies a minimum viable experiment",
            passed=bool(str(lead.get("minimum_viable_experiment") or "").strip()),
        ),
        _criterion(
            "datasets_metrics_baselines",
            "Lead idea names candidate datasets, metrics, and baselines",
            passed=bool(lead.get("candidate_datasets"))
            and bool(lead.get("candidate_metrics"))
            and bool(lead.get("candidate_baselines")),
        ),
        _criterion(
            "failure_criteria",
            "Lead idea includes explicit failure criteria",
            passed=bool(lead.get("failure_criteria")),
        ),
        _criterion(
            "literature_queries",
            "Lead idea includes literature queries or support probes",
            passed=bool(lead.get("literature_queries")),
            required=False,
        ),
    ]
    return _finalize_stage(
        "ideation",
        "idea_cards",
        criteria,
        signals={
            "idea_count": len(cards),
            "lead_idea_id": lead.get("idea_id"),
            "target_venue": lead.get("target_venue"),
        },
    )


def _evaluate_planning(research_plan: Any, claim_graph: Any) -> dict[str, Any]:
    plan = research_plan if isinstance(research_plan, dict) else {}
    graph = claim_graph if isinstance(claim_graph, dict) else {}
    tasks = [item for item in (plan.get("tasks") or []) if isinstance(item, dict)]
    nodes = [item for item in (graph.get("nodes") or []) if isinstance(item, dict)]
    edges = [item for item in (graph.get("edges") or []) if isinstance(item, dict)]
    claim_nodes = [item for item in nodes if item.get("type") == "claim"]
    if not plan:
        return _finalize_stage(
            "planning",
            "research_plan",
            [],
            missing_reason="research_plan artifact missing",
        )
    budget = plan.get("budget") if isinstance(plan.get("budget"), dict) else {}
    workflow_mode = str(plan.get("workflow_mode") or "").strip().lower()
    agent_plan = plan.get("agent_plan") if isinstance(plan.get("agent_plan"), dict) else {}
    agent_lanes = [item for item in (agent_plan.get("lanes") or []) if isinstance(item, dict)]
    criteria = [
        _criterion("task_count", "Research plan contains at least one task", passed=len(tasks) > 0, detail=f"count={len(tasks)}"),
        _criterion(
            "budget_fields",
            "Research plan sets max steps, wallclock, and retry budget",
            passed=all(
                key in budget and isinstance(budget.get(key), int)
                for key in ("max_steps", "max_wallclock_minutes", "max_retry_per_task")
            ),
        ),
        _criterion(
            "task_success_criteria",
            "Each task defines success and stop conditions",
            passed=all(
                str(task.get("success_criterion") or "").strip()
                and str(task.get("stop_condition") or "").strip()
                for task in tasks
            ),
        ),
        _criterion(
            "task_acceptance_checks",
            "Each task carries explicit acceptance checks",
            passed=all(bool(task.get("acceptance_checks")) for task in tasks),
        ),
        _criterion(
            "task_claim_targets",
            "Each task targets at least one claim",
            passed=all(bool(task.get("claim_targets")) for task in tasks),
        ),
        _criterion(
            "claim_graph_nodes",
            "Claim-evidence graph contains hypothesis and claim nodes",
            passed=any(node.get("type") == "hypothesis" for node in nodes)
            and bool(claim_nodes),
        ),
        _criterion(
            "claim_graph_edges",
            "Claim-evidence graph contains explicit edges",
            passed=bool(edges),
        ),
        _criterion(
            "agent_plan",
            "Planning artifact includes agent ownership and lane metadata for execution",
            passed=workflow_mode != "multi_agent_board"
            or (
                bool(agent_plan)
                and bool(agent_lanes)
                and all(bool(task.get("owner")) for task in tasks)
            ),
            required=workflow_mode == "multi_agent_board",
        ),
    ]
    return _finalize_stage(
        "planning",
        "research_plan",
        criteria,
        signals={
            "task_count": len(tasks),
            "claim_count": len(claim_nodes),
            "edge_count": len(edges),
            "workflow_mode": plan.get("workflow_mode"),
            "policy_name": (plan.get("execution_policy") or {}).get("policy_name"),
            "agent_lane_count": len(agent_lanes),
        },
    )


def _evaluate_experiment(project_root: str | Path) -> dict[str, Any]:
    records = load_jsonl_artifact(artifact_path(project_root, "experiment_registry"))
    if not records:
        return _finalize_stage(
            "experiment",
            "experiment_registry",
            [],
            missing_reason="experiment_registry artifact missing or empty",
        )
    completed_or_failed = [
        item
        for item in records
        if str(item.get("status") or "").lower() in {"completed", "failed"}
    ]
    criteria = [
        _criterion(
            "record_count",
            "Experiment registry contains records",
            passed=len(records) > 0,
            detail=f"count={len(records)}",
        ),
        _criterion(
            "task_linkage",
            "Every record keeps task, dataset, metric, and baseline linkage",
            passed=all(
                str(item.get("task_id") or "").strip()
                and str(item.get("dataset") or "").strip()
                and str(item.get("metric") or "").strip()
                and str(item.get("baseline_ref") or "").strip()
                for item in records
            ),
        ),
        _criterion(
            "budget_status",
            "Every record reports a budget status",
            passed=all(str(item.get("budget_status") or "").strip() for item in records),
        ),
        _criterion(
            "acceptance_checks",
            "Every record carries explicit acceptance checks",
            passed=all(bool(item.get("acceptance_checks")) for item in records),
        ),
        _criterion(
            "outcome_logged",
            "Completed or failed records log either results or errors",
            passed=all(
                bool(item.get("result_summary"))
                or bool(str(item.get("error_message") or "").strip())
                for item in completed_or_failed
            ),
        ),
        _criterion(
            "storyline_or_completion",
            "Registry contains at least one storyline-ready or completed run",
            passed=any(item.get("entered_storyline") for item in records)
            or any(str(item.get("status") or "").lower() == "completed" for item in records),
        ),
    ]
    return _finalize_stage(
        "experiment",
        "experiment_registry",
        criteria,
        signals={
            "record_count": len(records),
            "completed_count": sum(
                str(item.get("status") or "").lower() == "completed" for item in records
            ),
            "storyline_count": sum(bool(item.get("entered_storyline")) for item in records),
            "budget_exhausted_count": sum(
                str(item.get("budget_status") or "").lower() == "budget_exhausted"
                for item in records
            ),
        },
    )


def _evaluate_figure(
    figure_spec: Any,
    claim_graph: Any,
    research_plan: Any,
) -> dict[str, Any]:
    spec = figure_spec if isinstance(figure_spec, dict) else {}
    graph = claim_graph if isinstance(claim_graph, dict) else {}
    plan = research_plan if isinstance(research_plan, dict) else {}
    figures = [item for item in (spec.get("figures") or []) if isinstance(item, dict)]
    if not figures:
        return _finalize_stage(
            "figure",
            "figure_spec",
            [],
            missing_reason="figure_spec artifact missing or empty",
        )
    figure_summary = summarize_figure_spec(spec, claim_evidence_graph=graph)
    ready_figures = [item for item in figures if item.get("status") == "ready"]
    dedupe_signatures = [
        str(item.get("dedupe_signature") or "").strip() for item in figures
    ]
    claim_ids = {
        str(node.get("id"))
        for node in (graph.get("nodes") or [])
        if isinstance(node, dict) and node.get("type") == "claim"
    }
    ready_claim_ids = {
        str(item.get("claim_id") or "").strip() for item in ready_figures if item.get("claim_id")
    }
    workflow_mode = str(plan.get("workflow_mode") or "").strip().lower()
    strict_visual_lane = workflow_mode in {
        "program_driven",
        "writing_studio",
        "review_board",
        "multi_agent_board",
    }
    expected_main_ready = min(max(len(claim_ids), 1), 4) if claim_ids else min(
        max(len(ready_figures), 1),
        4,
    )
    criteria = [
        _criterion(
            "figure_count",
            "Figure spec contains at least one figure",
            passed=len(figures) > 0,
            detail=f"count={len(figures)}",
        ),
        _criterion(
            "ready_figure_count",
            "At least one figure is ready for evidence packaging",
            passed=len(ready_figures) > 0,
            detail=f"ready={len(ready_figures)}",
        ),
        _criterion(
            "ready_data_files",
            "Ready figures include traceable data files",
            passed=all(bool(item.get("data_files")) for item in ready_figures),
        ),
        _criterion(
            "dedupe_signatures",
            "Figure spec avoids duplicate dedupe signatures",
            passed=len([item for item in dedupe_signatures if item])
            == len(set(item for item in dedupe_signatures if item)),
        ),
        _criterion(
            "claim_coverage",
            "Ready figures cover all planned claims",
            passed=(not claim_ids) or claim_ids.issubset(ready_claim_ids),
            required=bool(claim_ids),
            detail=(
                f"covered={figure_summary.get('covered_claim_count', len(ready_claim_ids))}"
                f"/{figure_summary.get('claim_count', len(claim_ids))}"
            ),
        ),
        _criterion(
            "main_paper_figures",
            "Visual workflow keeps enough main-paper figures ready for the narrative spine",
            passed=expected_main_ready == 0
            or int(figure_summary.get("main_ready_count") or 0) >= expected_main_ready,
            required=strict_visual_lane or bool(claim_ids),
            detail=(
                f"main_ready={int(figure_summary.get('main_ready_count') or 0)}"
                f"/{expected_main_ready}"
            ),
        ),
        _criterion(
            "primary_figure_blockers",
            "Main-paper visual slots do not remain blocked",
            passed=int(figure_summary.get("main_blocked_count") or 0) == 0,
            required=strict_visual_lane,
            detail=f"blocked_main={int(figure_summary.get('main_blocked_count') or 0)}",
        ),
        _criterion(
            "data_file_availability",
            "Ready figures only reference data files that exist when availability is checked",
            passed=(
                not figure_summary.get("checked_data_file_availability")
                or int(figure_summary.get("ready_missing_data_file_count") or 0) == 0
            ),
            required=bool(figure_summary.get("checked_data_file_availability")),
            detail=(
                f"ready_missing_files={int(figure_summary.get('ready_missing_data_file_count') or 0)}"
            ),
        ),
    ]
    return _finalize_stage(
        "figure",
        "figure_spec",
        criteria,
        signals={
            "figure_count": len(figures),
            "ready_figure_count": len(ready_figures),
            "blocked_figure_count": sum(item.get("status") == "blocked" for item in figures),
            "claim_coverage_ratio": figure_summary.get("claim_coverage_ratio"),
            "main_ready_count": figure_summary.get("main_ready_count"),
            "main_blocked_count": figure_summary.get("main_blocked_count"),
            "missing_data_file_count": figure_summary.get("missing_data_file_count"),
            "strict_visual_lane": strict_visual_lane,
        },
    )


def _evaluate_manuscript(manuscript_state: Any) -> dict[str, Any]:
    state = manuscript_state if isinstance(manuscript_state, dict) else {}
    if not state:
        return _finalize_stage(
            "manuscript",
            "manuscript_state",
            [],
            missing_reason="manuscript_state artifact missing",
        )
    claim_bindings = list(state.get("claim_bindings") or [])
    figure_bindings = state.get("figure_bindings") or {}
    missing_evidence = list(state.get("missing_evidence") or [])
    criteria = [
        _criterion(
            "outline",
            "Manuscript state defines an outline",
            passed=bool(state.get("outline")),
        ),
        _criterion(
            "claim_bindings",
            "Manuscript state binds explicit claims",
            passed=bool(claim_bindings),
        ),
        _criterion(
            "figure_bindings",
            "Manuscript state binds claims to figures",
            passed=bool(figure_bindings),
        ),
        _criterion(
            "guardrail_status",
            "Manuscript guardrails are ready",
            passed=str(state.get("guardrail_status") or "").lower() == "ready",
            detail=str(state.get("guardrail_status") or ""),
        ),
        _criterion(
            "missing_evidence",
            "Manuscript has no unresolved missing evidence",
            passed=not missing_evidence,
            detail=f"missing={len(missing_evidence)}",
        ),
        _criterion(
            "skill_pack",
            "Writeup skill pack is explicit",
            passed=bool(state.get("skill_pack")),
            required=False,
        ),
    ]
    return _finalize_stage(
        "manuscript",
        "manuscript_state",
        criteria,
        signals={
            "claim_count": len(claim_bindings),
            "figure_binding_count": len(figure_bindings),
            "missing_evidence_count": len(missing_evidence),
            "guardrail_status": state.get("guardrail_status"),
        },
    )


def _evaluate_review(review_state: Any, repair_plan: Any, self_evolution: Any) -> dict[str, Any]:
    state = review_state if isinstance(review_state, dict) else {}
    repair_plan_payload = repair_plan if isinstance(repair_plan, dict) else {}
    self_evolution_payload = self_evolution if isinstance(self_evolution, dict) else {}
    if not state:
        return _finalize_stage(
            "review",
            "review_state",
            [],
            missing_reason="review_state artifact missing",
        )
    rounds = [item for item in (state.get("rounds") or []) if isinstance(item, dict)]
    role_summaries = state.get("role_summaries") or {}
    lane_summaries = state.get("lane_summaries") or {}
    usage = state.get("usage_accounting") or {}
    repair_metrics = compute_review_repair_metrics(state)
    active_issue_count = int(repair_metrics.get("active_issue_count") or 0)
    resolved_issue_count = int(repair_metrics.get("resolved_issue_count") or 0)
    persistent_issue_count = int(repair_metrics.get("persistent_issue_count") or 0)
    repair_action_count = int(repair_metrics.get("repair_action_count") or 0)
    verification_count = int(repair_metrics.get("verification_count") or 0)
    bound_issue_count = int(repair_metrics.get("bound_issue_count") or 0)
    unbound_issue_count = int(repair_metrics.get("unbound_issue_count") or 0)
    bound_active_issue_count = int(repair_metrics.get("bound_active_issue_count") or 0)
    target_binding_coverage = float(repair_metrics.get("target_binding_coverage") or 0.0)
    active_binding_coverage = float(repair_metrics.get("active_binding_coverage") or 0.0)
    repair_queue_count = int(repair_metrics.get("repair_queue_count") or 0)
    repair_ready_count = int(repair_metrics.get("repair_ready_count") or 0)
    repair_verification_ready_count = int(
        repair_metrics.get("repair_verification_ready_count") or 0
    )
    repair_targeted_count = int(repair_metrics.get("repair_targeted_count") or 0)
    repair_queue_coverage = float(repair_metrics.get("repair_queue_coverage") or 0.0)
    repair_ready_coverage = float(repair_metrics.get("repair_ready_coverage") or 0.0)
    repair_verification_ready_coverage = float(
        repair_metrics.get("repair_verification_ready_coverage") or 0.0
    )
    repair_targeted_coverage = float(
        repair_metrics.get("repair_targeted_coverage") or 0.0
    )
    repair_plan_summary = (
        repair_plan_payload.get("summary")
        if isinstance(repair_plan_payload.get("summary"), dict)
        else {}
    )
    repair_plan_task_count = int(repair_plan_summary.get("task_count") or 0)
    repair_plan_lane_count = int(repair_plan_summary.get("lane_count") or 0)
    repair_plan_ready_rate = float(repair_plan_summary.get("ready_rate") or 0.0)
    repair_plan_verification_ready_rate = float(
        repair_plan_summary.get("verification_ready_rate") or 0.0
    )
    evolution_summary = (
        self_evolution_payload.get("summary")
        if isinstance(self_evolution_payload.get("summary"), dict)
        else {}
    )
    evolution_self_check = (
        self_evolution_payload.get("self_check")
        if isinstance(self_evolution_payload.get("self_check"), dict)
        else {}
    )
    evolution_status = str(evolution_summary.get("status") or "").strip()
    evolution_score = float(evolution_summary.get("score") or 0.0)
    evolution_lesson_count = len(self_evolution_payload.get("lessons") or [])
    evolution_required_failure_count = len(
        evolution_self_check.get("required_failures") or []
    )
    role_coverage_ratio = float(repair_metrics.get("role_coverage_ratio") or 0.0)
    resolution_rate = float(repair_metrics.get("resolution_rate") or 0.0)
    hostile_summary = (
        lane_summaries.get("hostile_critic")
        if isinstance(lane_summaries.get("hostile_critic"), dict)
        else {}
    )
    hostile_active_issue_count = int(hostile_summary.get("active_issue_count") or 0)
    hostile_blocking_issue_count = int(hostile_summary.get("blocking_issue_count") or 0)
    criteria = [
        _criterion(
            "review_rounds",
            "Review state contains at least one review round",
            passed=len(rounds) > 0,
            detail=f"count={len(rounds)}",
        ),
        _criterion(
            "role_summaries",
            "Review state stores role summaries",
            passed=bool(role_summaries),
        ),
        _criterion(
            "usage_accounting",
            "Review state records usage accounting",
            passed=bool(usage),
        ),
        _criterion(
            "issue_or_repair_trace",
            "Review state records either issues or repair actions",
            passed=bool(state.get("active_issues"))
            or bool(state.get("repair_actions"))
            or bool(state.get("resolved_issues")),
        ),
        _criterion(
            "repair_metrics",
            "Review state exposes explicit repair metrics",
            passed=bool(state.get("repair_metrics"))
            or bool(state.get("issue_ledger"))
            or active_issue_count > 0
            or resolved_issue_count > 0
            or verification_count > 0
            or repair_action_count > 0
            or (len(rounds) > 0 and not bool(state.get("active_issues"))),
        ),
        _criterion(
            "repair_plan_for_active_issues",
            "Active review issues have a concrete repair plan or verification path",
            passed=active_issue_count == 0
            or repair_action_count > 0
            or verification_count > 0,
            detail=f"active={active_issue_count} repairs={repair_action_count} checks={verification_count}",
        ),
        _criterion(
            "repair_queue",
            "Active review issues are converted into structured repair tasks",
            passed=active_issue_count == 0 or repair_queue_coverage >= 1.0,
            detail=f"queue={repair_queue_count}/{active_issue_count}",
        ),
        _criterion(
            "repair_queue_ready",
            "Structured repair tasks are ready for execution and verification",
            passed=active_issue_count == 0 or repair_ready_coverage >= 0.5,
            detail=(
                f"ready={repair_ready_count}/{active_issue_count} "
                f"verification_ready={repair_verification_ready_count}/{active_issue_count}"
            ),
        ),
        _criterion(
            "issue_target_bindings",
            "Review issues are bound to concrete claims, figures, or sections",
            passed=active_issue_count == 0 or active_binding_coverage >= 0.5,
            detail=f"bound_active={bound_active_issue_count}/{active_issue_count}",
        ),
        _criterion(
            "repair_plan_artifact",
            "Reviewer repair queue is elevated into an agentic repair plan artifact",
            passed=active_issue_count == 0
            or (
                bool(repair_plan_payload)
                and repair_plan_task_count >= active_issue_count
                and repair_plan_lane_count >= 1
            ),
            detail=(
                f"tasks={repair_plan_task_count} lanes={repair_plan_lane_count} "
                f"ready_rate={repair_plan_ready_rate}"
            ),
        ),
        _criterion(
            "repair_plan_verification",
            "Agentic repair plan carries explicit verification-ready tasks",
            passed=active_issue_count == 0
            or repair_plan_verification_ready_rate >= 0.5,
            required=False,
            detail=(
                f"verification_ready_rate={repair_plan_verification_ready_rate:.2f}"
            ),
        ),
        _criterion(
            "verification_checks",
            "Review state exposes verification checks or issue-resolution tracking",
            passed=verification_count > 0 or resolved_issue_count > 0 or persistent_issue_count > 0,
            required=False,
        ),
        _criterion(
            "role_coverage",
            "Review coverage includes at least one explicit reviewer role",
            passed=role_coverage_ratio >= 0.25,
            required=False,
            detail=f"coverage={role_coverage_ratio:.2f}",
        ),
        _criterion(
            "hostile_critic_lane",
            "Independent hostile critic lane is either clear or absent",
            passed=not hostile_summary or hostile_active_issue_count == 0,
            required=False,
            detail=(
                f"active={hostile_active_issue_count} blocking={hostile_blocking_issue_count}"
            )
            if hostile_summary
            else None,
        ),
        _criterion(
            "persistent_issue_budget",
            "Persistent reviewer debt stays within a manageable budget",
            passed=persistent_issue_count <= max(1, active_issue_count),
            required=False,
            detail=f"persistent={persistent_issue_count}",
        ),
        _criterion(
            "self_evolution_artifact",
            "Review loop produces a self-evolution artifact for the next cycle",
            passed=active_issue_count == 0
            or (
                bool(self_evolution_payload)
                and evolution_lesson_count >= 1
                and evolution_score >= 50.0
            ),
            detail=(
                f"status={evolution_status or 'missing'} "
                f"score={evolution_score:.1f} "
                f"lessons={evolution_lesson_count}"
            ),
        ),
        _criterion(
            "self_evolution_self_check",
            "Self-evolution artifact carries an explicit self-check with manageable failures",
            passed=active_issue_count == 0 or evolution_required_failure_count <= 1,
            required=False,
            detail=(
                f"status={evolution_status or 'missing'} "
                f"required_failures={evolution_required_failure_count}"
            ),
        ),
    ]
    return _finalize_stage(
        "review",
        "review_state",
        criteria,
        signals={
            "round_count": len(rounds),
            "active_issue_count": active_issue_count,
            "resolved_issue_count": resolved_issue_count,
            "persistent_issue_count": persistent_issue_count,
            "repair_action_count": repair_action_count,
            "verification_count": verification_count,
            "bound_issue_count": bound_issue_count,
            "unbound_issue_count": unbound_issue_count,
            "target_binding_coverage": target_binding_coverage,
            "active_binding_coverage": active_binding_coverage,
            "repair_queue_count": repair_queue_count,
            "repair_ready_count": repair_ready_count,
            "repair_verification_ready_count": repair_verification_ready_count,
            "repair_targeted_count": repair_targeted_count,
            "repair_queue_coverage": repair_queue_coverage,
            "repair_ready_coverage": repair_ready_coverage,
            "repair_verification_ready_coverage": repair_verification_ready_coverage,
            "repair_targeted_coverage": repair_targeted_coverage,
            "repair_plan_task_count": repair_plan_task_count,
            "repair_plan_lane_count": repair_plan_lane_count,
            "repair_plan_ready_rate": repair_plan_ready_rate,
            "repair_plan_verification_ready_rate": repair_plan_verification_ready_rate,
            "self_evolution_status": evolution_status,
            "self_evolution_score": evolution_score,
            "self_evolution_lesson_count": evolution_lesson_count,
            "self_evolution_required_failure_count": evolution_required_failure_count,
            "resolution_rate": resolution_rate,
            "role_coverage_ratio": role_coverage_ratio,
            "role_count": len(role_summaries),
            "lane_count": len(lane_summaries),
            "hostile_critic_active_issue_count": hostile_active_issue_count,
            "hostile_critic_blocking_issue_count": hostile_blocking_issue_count,
        },
    )


def build_stage_standards(project_root: str | Path) -> dict[str, Any]:
    resolved_root = Path(project_root).expanduser().resolve()
    idea_cards = load_contract_artifact(resolved_root, "idea_cards", default=[])
    research_plan = load_contract_artifact(resolved_root, "research_plan", default={})
    claim_graph = load_contract_artifact(
        resolved_root, "claim_evidence_graph", default={}
    )
    figure_spec = load_contract_artifact(resolved_root, "figure_spec", default={})
    manuscript_state = load_contract_artifact(
        resolved_root, "manuscript_state", default={}
    )
    review_state = load_contract_artifact(resolved_root, "review_state", default={})
    repair_plan = load_contract_artifact(resolved_root, "repair_plan", default={})
    self_evolution = load_contract_artifact(
        resolved_root, "self_evolution", default={}
    )

    stage_results = [
        _evaluate_ideation(idea_cards),
        _evaluate_planning(research_plan, claim_graph),
        _evaluate_experiment(resolved_root),
        _evaluate_figure(figure_spec, claim_graph, research_plan),
        _evaluate_manuscript(manuscript_state),
        _evaluate_review(review_state, repair_plan, self_evolution),
    ]
    ready = [item for item in stage_results if item.get("status") == "ready"]
    blocked = [item for item in stage_results if item.get("status") == "blocked"]
    attention = [
        item for item in stage_results if item.get("status") == "needs_attention"
    ]
    missing = [item for item in stage_results if item.get("status") == "missing"]
    overall_score = round(
        sum(float(item.get("score") or 0.0) for item in stage_results)
        / max(len(stage_results), 1),
        1,
    )
    top_risks: list[str] = []
    for item in blocked + attention + missing:
        if item.get("required_failures"):
            top_risks.append(
                f"{item.get('stage')}: {', '.join(item.get('required_failures') or [])}"
            )
        elif item.get("missing_reason"):
            top_risks.append(f"{item.get('stage')}: {item.get('missing_reason')}")
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "project_root": str(resolved_root),
        "overall_score": overall_score,
        "ready_stage_count": len(ready),
        "blocked_stage_count": len(blocked),
        "needs_attention_stage_count": len(attention),
        "missing_stage_count": len(missing),
        "stage_results": stage_results,
        "summary": {
            "blocked_stages": [item.get("stage") for item in blocked],
            "attention_stages": [item.get("stage") for item in attention],
            "missing_stages": [item.get("stage") for item in missing],
            "top_risks": top_risks[:6],
        },
    }


def save_stage_standards(project_root: str | Path) -> str:
    payload = build_stage_standards(project_root)
    artifact = save_contract_artifact(
        project_root,
        "stage_standards",
        payload,
        producer="stage_standards",
        depends_on=[
            "idea_cards",
            "research_plan",
            "claim_evidence_graph",
            "experiment_registry",
            "figure_spec",
            "manuscript_state",
            "review_state",
        ],
    )
    # Keep the cross-reference process audit fresh whenever stage standards change.
    from ai_scientist.utils.process_alignment import save_process_alignment

    save_process_alignment(project_root, producer="stage_standards")
    return artifact
