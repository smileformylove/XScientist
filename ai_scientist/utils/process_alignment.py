from __future__ import annotations

"""Process-level alignment checks against key open-source autonomous research references."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.figure_spec import summarize_figure_spec
from ai_scientist.utils.pipeline_contracts import (
    artifact_path,
    load_contract_artifact,
    load_json_artifact,
    load_jsonl_artifact,
    save_contract_artifact,
)
from ai_scientist.utils.review_jobs import compute_review_repair_metrics


PROCESS_ALIGNMENT_SCHEMA_VERSION = 1

REFERENCE_URLS = {
    "SakanaAI/AI-Scientist": "https://github.com/SakanaAI/AI-Scientist",
    "SakanaAI/AI-Scientist-v2": "https://github.com/SakanaAI/AI-Scientist-v2",
    "karpathy/autoresearch": "https://github.com/karpathy/autoresearch",
    "Leey21/awesome-ai-research-writing": "https://github.com/Leey21/awesome-ai-research-writing",
    "ResearAI/DeepReviewer-v2": "https://github.com/ResearAI/DeepReviewer-v2",
}

PROCESS_BLUEPRINTS = (
    {
        "process": "ideation",
        "label": "Idea Formation",
        "references": ["SakanaAI/AI-Scientist", "karpathy/autoresearch"],
        "artifacts": ["idea_cards"],
        "focus": "Generate grounded ideas with explicit novelty, risks, and minimum viable experiments.",
    },
    {
        "process": "program",
        "label": "Research Program",
        "references": ["karpathy/autoresearch"],
        "artifacts": ["research_program", "research_plan"],
        "focus": "Define the research operating program, acceptance rules, and budget discipline.",
    },
    {
        "process": "exploration",
        "label": "Exploration Graph",
        "references": ["SakanaAI/AI-Scientist-v2", "karpathy/autoresearch"],
        "artifacts": ["research_plan", "claim_evidence_graph"],
        "focus": "Maintain explicit exploratory tasks, claim targets, and graph-structured reasoning.",
    },
    {
        "process": "experiment",
        "label": "Experiment Execution",
        "references": [
            "SakanaAI/AI-Scientist",
            "SakanaAI/AI-Scientist-v2",
            "karpathy/autoresearch",
        ],
        "artifacts": ["experiment_registry"],
        "focus": "Run budgeted experiments with registry discipline, acceptance checks, and recoverable records.",
    },
    {
        "process": "figure",
        "label": "Figure Packaging",
        "references": ["SakanaAI/AI-Scientist", "Leey21/awesome-ai-research-writing"],
        "artifacts": ["figure_spec"],
        "focus": "Bind every figure to a claim, data file, and narrative slot with traceable provenance.",
    },
    {
        "process": "writing",
        "label": "Writing Studio",
        "references": ["Leey21/awesome-ai-research-writing"],
        "artifacts": ["manuscript_state"],
        "focus": "Convert evidence into sectioned writing with explicit bindings, missing-evidence checks, and polishing skills.",
    },
    {
        "process": "review",
        "label": "Reviewer Board",
        "references": ["ResearAI/DeepReviewer-v2"],
        "artifacts": ["review_state", "critic_findings", "repair_plan"],
        "focus": "Run structured, tool-grounded review with binding coverage, hostile-critic findings, repair queue, and verification paths.",
    },
    {
        "process": "evolution",
        "label": "Self Evolution",
        "references": ["ResearAI/DeepReviewer-v2", "SakanaAI/AI-Scientist-v2"],
        "artifacts": ["self_evolution"],
        "focus": "Turn reviewer feedback into reusable agentic defaults, lessons, and next-cycle repair pressure.",
    },
    {
        "process": "packaging",
        "label": "Submission Packaging",
        "references": [
            "SakanaAI/AI-Scientist",
            "Leey21/awesome-ai-research-writing",
            "ResearAI/DeepReviewer-v2",
        ],
        "artifacts": ["manuscript_state", "review_state"],
        "focus": "Produce submission-facing quality signals, packaging assets, and evidence-safe submission posture.",
    },
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _coerce_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _coerce_str_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in _coerce_list(value):
        text = str(item).strip()
        if text:
            result.append(text)
    return result


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


def _finalize_process(
    blueprint: dict[str, Any],
    criteria: list[dict[str, Any]],
    *,
    signals: dict[str, Any] | None = None,
    risks: list[str] | None = None,
    missing_reason: str | None = None,
) -> dict[str, Any]:
    if missing_reason:
        return {
            "process": blueprint["process"],
            "label": blueprint["label"],
            "focus": blueprint["focus"],
            "references": [
                {
                    "name": name,
                    "url": REFERENCE_URLS.get(name),
                }
                for name in blueprint["references"]
            ],
            "artifacts": blueprint["artifacts"],
            "status": "missing",
            "score": 0.0,
            "criteria": [],
            "required_failures": [],
            "signals": signals or {},
            "risks": list(risks or []),
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
        "process": blueprint["process"],
        "label": blueprint["label"],
        "focus": blueprint["focus"],
        "references": [
            {
                "name": name,
                "url": REFERENCE_URLS.get(name),
            }
            for name in blueprint["references"]
        ],
        "artifacts": blueprint["artifacts"],
        "status": status,
        "score": score,
        "criteria": criteria,
        "required_failures": required_failures,
        "signals": signals or {},
        "risks": list(risks or []),
        "missing_reason": None,
    }


def _safe_quality_result(project_root: Path) -> dict[str, Any]:
    payload = load_json_artifact(project_root / "quality" / "high_quality_result.json", default={})
    return payload if isinstance(payload, dict) else {}


def build_process_alignment(project_root: str | Path) -> dict[str, Any]:
    resolved_root = Path(project_root).expanduser().resolve()
    idea_cards = load_contract_artifact(resolved_root, "idea_cards", default=[]) or []
    research_plan = load_contract_artifact(resolved_root, "research_plan", default={}) or {}
    claim_graph = load_contract_artifact(resolved_root, "claim_evidence_graph", default={}) or {}
    experiment_records = load_jsonl_artifact(artifact_path(resolved_root, "experiment_registry"))
    figure_spec = load_contract_artifact(resolved_root, "figure_spec", default={}) or {}
    manuscript_state = load_contract_artifact(resolved_root, "manuscript_state", default={}) or {}
    review_state = load_contract_artifact(resolved_root, "review_state", default={}) or {}
    repair_plan = load_contract_artifact(resolved_root, "repair_plan", default={}) or {}
    self_evolution = load_contract_artifact(resolved_root, "self_evolution", default={}) or {}
    quality_result = _safe_quality_result(resolved_root)
    research_program_path = artifact_path(resolved_root, "research_program")
    research_program_text = (
        research_program_path.read_text(encoding="utf-8")
        if research_program_path.exists()
        else ""
    )
    review_metrics = compute_review_repair_metrics(review_state)
    self_evolution_summary = (
        self_evolution.get("summary")
        if isinstance(self_evolution.get("summary"), dict)
        else {}
    )
    self_evolution_self_check = (
        self_evolution.get("self_check")
        if isinstance(self_evolution.get("self_check"), dict)
        else {}
    )
    repair_plan_summary = (
        repair_plan.get("summary")
        if isinstance(repair_plan.get("summary"), dict)
        else {}
    )

    cards = [item for item in idea_cards if isinstance(item, dict)]
    lead_idea = cards[0] if cards else {}
    tasks = [item for item in (research_plan.get("tasks") or []) if isinstance(item, dict)]
    graph_nodes = [item for item in (claim_graph.get("nodes") or []) if isinstance(item, dict)]
    graph_edges = [item for item in (claim_graph.get("edges") or []) if isinstance(item, dict)]
    figure_rows = [
        item for item in (figure_spec.get("figures") or []) if isinstance(item, dict)
    ]
    ready_figures = [
        item for item in figure_rows if str(item.get("status") or "").strip() == "ready"
    ]
    blocked_figures = [
        item for item in figure_rows if str(item.get("status") or "").strip() == "blocked"
    ]
    figure_summary = summarize_figure_spec(
        figure_spec,
        claim_evidence_graph=claim_graph,
    )
    completed_records = [
        item
        for item in experiment_records
        if str(item.get("status") or "").strip() == "completed"
    ]
    storyline_records = [
        item for item in experiment_records if item.get("entered_storyline") is True
    ]
    submission_readiness = (
        quality_result.get("submission_readiness")
        if isinstance(quality_result.get("submission_readiness"), dict)
        else {}
    )

    by_process: dict[str, dict[str, Any]] = {}

    for blueprint in PROCESS_BLUEPRINTS:
        process = blueprint["process"]
        if process == "ideation":
            if not cards:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="idea_cards artifact missing or empty",
                )
                continue
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("idea_count", "At least one idea card exists", passed=bool(cards), detail=f"count={len(cards)}"),
                    _criterion("core_hypothesis", "Lead idea defines a core hypothesis", passed=bool(str(lead_idea.get("core_hypothesis") or "").strip())),
                    _criterion("novelty_claim", "Lead idea declares novelty or related-work positioning", passed=bool(str(lead_idea.get("novelty_claim") or "").strip() or str(lead_idea.get("related_work_notes") or "").strip())),
                    _criterion("minimum_experiment", "Lead idea specifies a minimum viable experiment", passed=bool(str(lead_idea.get("minimum_viable_experiment") or "").strip())),
                    _criterion("datasets_metrics_baselines", "Lead idea lists datasets, metrics, and baselines", passed=bool(lead_idea.get("candidate_datasets")) and bool(lead_idea.get("candidate_metrics")) and bool(lead_idea.get("candidate_baselines"))),
                    _criterion("failure_criteria", "Lead idea includes explicit failure criteria", passed=bool(lead_idea.get("failure_criteria"))),
                ],
                signals={
                    "idea_count": len(cards),
                    "lead_idea_id": lead_idea.get("idea_id"),
                },
                risks=[] if lead_idea.get("failure_criteria") else ["idea_failure_criteria_gap"],
            )
        elif process == "program":
            budget = research_plan.get("budget") if isinstance(research_plan.get("budget"), dict) else {}
            execution_policy = (
                research_plan.get("execution_policy")
                if isinstance(research_plan.get("execution_policy"), dict)
                else {}
            )
            if not research_plan:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="research_plan artifact missing",
                )
                continue
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("research_program", "Research program markdown exists", passed=bool(research_program_text.strip())),
                    _criterion("budget_fields", "Research plan declares max steps, wallclock, and retry budget", passed=all(isinstance(budget.get(key), int) for key in ("max_steps", "max_wallclock_minutes", "max_retry_per_task"))),
                    _criterion("acceptance_rules", "Execution policy declares acceptance rules", passed=bool(execution_policy.get("acceptance_rules"))),
                    _criterion("registry_expectations", "Execution policy declares registry discipline", passed=bool(execution_policy.get("registry_expectations")), required=False),
                    _criterion("success_criteria", "Research program names success criteria", passed="## Success Criteria" in research_program_text),
                    _criterion("failure_handling_rules", "Research program names failure handling rules", passed="## Failure Handling" in research_program_text),
                ],
                signals={
                    "workflow_mode": research_plan.get("workflow_mode"),
                    "policy_name": execution_policy.get("policy_name"),
                },
                risks=[] if execution_policy.get("acceptance_rules") else ["program_acceptance_rule_gap"],
            )
        elif process == "exploration":
            if not research_plan:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="research_plan artifact missing",
                )
                continue
            task_kinds = {
                str(item.get("task_kind") or "").strip()
                for item in tasks
                if str(item.get("task_kind") or "").strip()
            }
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("task_count", "Research plan includes explicit tasks", passed=bool(tasks), detail=f"count={len(tasks)}"),
                    _criterion("claim_targets", "Tasks target downstream claims", passed=all(bool(item.get("claim_targets")) for item in tasks) if tasks else False),
                    _criterion("graph_nodes", "Claim graph contains explicit nodes", passed=bool(graph_nodes)),
                    _criterion("graph_edges", "Claim graph contains explicit edges", passed=bool(graph_edges)),
                    _criterion("expected_outputs", "Tasks declare expected outputs", passed=all(bool(item.get("expected_outputs")) for item in tasks) if tasks else False),
                    _criterion("task_kinds", "Exploration tracks task kinds for agentic branching", passed=bool(task_kinds), required=False),
                ],
                signals={
                    "task_count": len(tasks),
                    "graph_node_count": len(graph_nodes),
                    "graph_edge_count": len(graph_edges),
                    "task_kinds": sorted(task_kinds),
                },
                risks=[] if graph_edges else ["exploration_graph_gap"],
            )
        elif process == "experiment":
            if not experiment_records:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="experiment_registry artifact missing or empty",
                )
                continue
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("record_count", "Experiment registry contains records", passed=bool(experiment_records), detail=f"count={len(experiment_records)}"),
                    _criterion("budget_status", "Every experiment record reports budget status", passed=all(str(item.get("budget_status") or "").strip() for item in experiment_records)),
                    _criterion("acceptance_checks", "Every experiment record carries acceptance checks", passed=all(bool(item.get("acceptance_checks")) for item in experiment_records)),
                    _criterion("completed_or_failed", "Experiment registry preserves terminal records", passed=bool(completed_records) or any(str(item.get("status") or "").strip() == "failed" for item in experiment_records)),
                    _criterion("result_summary", "Completed records keep structured result summaries", passed=all(bool(item.get("result_summary")) for item in completed_records) if completed_records else False),
                    _criterion("storyline_trace", "At least one record is explicitly promoted into the storyline", passed=bool(storyline_records), required=False),
                ],
                signals={
                    "record_count": len(experiment_records),
                    "completed_count": len(completed_records),
                    "storyline_count": len(storyline_records),
                },
                risks=[] if completed_records else ["experiment_completion_gap"],
            )
        elif process == "figure":
            if not figure_rows:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="figure_spec artifact missing or empty",
                )
                continue
            workflow_mode = str(research_plan.get("workflow_mode") or "").strip().lower()
            strict_visual_lane = workflow_mode in {
                "program_driven",
                "writing_studio",
                "review_board",
                "multi_agent_board",
            }
            expected_main_ready = min(
                max(int(figure_summary.get("claim_count") or 0), 1),
                4,
            ) if int(figure_summary.get("claim_count") or 0) else min(
                max(len(ready_figures), 1),
                4,
            )
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("figure_count", "Figure spec contains figures", passed=bool(figure_rows), detail=f"count={len(figure_rows)}"),
                    _criterion("claim_binding", "Each figure is bound to a claim", passed=all(str(item.get("claim_id") or "").strip() for item in figure_rows)),
                    _criterion("data_traceability", "Each figure cites data files and source records", passed=all(bool(item.get("data_files")) and bool(item.get("source_records")) for item in figure_rows)),
                    _criterion("caption_intent", "Each figure defines caption intent", passed=all(str(item.get("caption_intent") or "").strip() for item in figure_rows)),
                    _criterion(
                        "claim_coverage",
                        "Ready figures cover the planned claims",
                        passed=float(figure_summary.get("claim_coverage_ratio") or 0.0) >= 1.0,
                        required=bool(figure_summary.get("claim_count")),
                        detail=(
                            f"covered={int(figure_summary.get('covered_claim_count') or 0)}"
                            f"/{int(figure_summary.get('claim_count') or 0)}"
                        ),
                    ),
                    _criterion(
                        "main_ready_figures",
                        "Main-paper figures are ready for the narrative spine",
                        passed=expected_main_ready == 0
                        or int(figure_summary.get("main_ready_count") or 0) >= expected_main_ready,
                        required=strict_visual_lane or bool(figure_summary.get("claim_count")),
                        detail=(
                            f"main_ready={int(figure_summary.get('main_ready_count') or 0)}"
                            f"/{expected_main_ready}"
                        ),
                    ),
                    _criterion(
                        "blocked_figures",
                        "Main-paper figures do not remain blocked",
                        passed=int(figure_summary.get("main_blocked_count") or 0) == 0,
                        required=strict_visual_lane,
                        detail=f"blocked_main={int(figure_summary.get('main_blocked_count') or 0)}",
                    ),
                    _criterion(
                        "data_file_availability",
                        "Figure spec records no missing ready data files when availability is checked",
                        passed=(
                            not figure_summary.get("checked_data_file_availability")
                            or int(figure_summary.get("ready_missing_data_file_count") or 0) == 0
                        ),
                        required=bool(figure_summary.get("checked_data_file_availability")),
                        detail=(
                            f"ready_missing_files={int(figure_summary.get('ready_missing_data_file_count') or 0)}"
                        ),
                    ),
                    _criterion("ready_figures", "At least one figure is ready for use", passed=bool(ready_figures)),
                ],
                signals={
                    "figure_count": len(figure_rows),
                    "ready_figure_count": len(ready_figures),
                    "blocked_figure_count": len(blocked_figures),
                    "claim_coverage_ratio": figure_summary.get("claim_coverage_ratio"),
                    "main_ready_count": figure_summary.get("main_ready_count"),
                    "main_blocked_count": figure_summary.get("main_blocked_count"),
                },
                risks=["figure_blocked"] if blocked_figures else [],
            )
        elif process == "writing":
            if not manuscript_state:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="manuscript_state artifact missing",
                )
                continue
            section_briefs = manuscript_state.get("section_briefs") if isinstance(manuscript_state.get("section_briefs"), dict) else {}
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("outline", "Manuscript state includes an outline", passed=bool(manuscript_state.get("outline"))),
                    _criterion("section_briefs", "Manuscript state includes section briefs", passed=bool(section_briefs)),
                    _criterion("claim_bindings", "Manuscript state binds claims", passed=bool(manuscript_state.get("claim_bindings"))),
                    _criterion("figure_bindings", "Manuscript state binds figures or tables", passed=bool(manuscript_state.get("figure_bindings")) or bool(manuscript_state.get("table_bindings"))),
                    _criterion("guardrail_status", "Writing guardrail status is not blocked", passed=str(manuscript_state.get("guardrail_status") or "").strip() not in {"", "blocked"}),
                    _criterion("missing_evidence", "Writing state is not missing evidence", passed=not manuscript_state.get("missing_evidence"), required=False),
                ],
                signals={
                    "guardrail_status": manuscript_state.get("guardrail_status"),
                    "missing_evidence_count": len(_coerce_list(manuscript_state.get("missing_evidence"))),
                },
                risks=["writing_missing_evidence"] if manuscript_state.get("missing_evidence") else [],
            )
        elif process == "review":
            if not review_state:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="review_state artifact missing",
                )
                continue
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("rounds", "Review state contains at least one review round", passed=bool(review_state.get("rounds"))),
                    _criterion("role_coverage", "Review covers multiple reviewer roles", passed=float(review_metrics.get("role_coverage_ratio") or 0.0) >= 0.5, required=False, detail=f"role_coverage_ratio={float(review_metrics.get('role_coverage_ratio') or 0.0):.2f}"),
                    _criterion("binding_coverage", "Review issues are bound to claim / figure / section targets", passed=float(review_metrics.get("target_binding_coverage") or 0.0) >= 1.0),
                    _criterion("repair_queue", "Active issues are converted into repair queue items", passed=float(review_metrics.get("repair_queue_coverage") or 0.0) >= 1.0),
                    _criterion("repair_ready", "Repair queue is verification-ready", passed=float(review_metrics.get("repair_ready_coverage") or 0.0) >= 1.0),
                    _criterion("repair_plan", "Repair plan exists and tracks lane-oriented execution", passed=bool(repair_plan_summary.get("task_count"))),
                ],
                signals={
                    "active_issue_count": int(review_metrics.get("active_issue_count") or 0),
                    "repair_queue_count": int(review_metrics.get("repair_queue_count") or 0),
                    "repair_ready_coverage": float(review_metrics.get("repair_ready_coverage") or 0.0),
                },
                risks=["review_binding_gap"] if float(review_metrics.get("target_binding_coverage") or 0.0) < 1.0 else [],
            )
        elif process == "evolution":
            if not self_evolution:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="self_evolution artifact missing",
                )
                continue
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("self_check", "Self-evolution self-check exists", passed=bool(self_evolution_self_check)),
                    _criterion("required_failures", "Self-evolution has no required failures", passed=not self_evolution_self_check.get("required_failures")),
                    _criterion("lessons", "Self-evolution records lessons", passed=bool(self_evolution.get("lessons"))),
                    _criterion("next_cycle_defaults", "Self-evolution proposes next-cycle defaults", passed=bool(self_evolution.get("next_cycle_defaults"))),
                    _criterion("score", "Self-evolution score clears 80", passed=float(self_evolution_summary.get("score") or 0.0) >= 80.0),
                    _criterion("stage_risks", "Self-evolution surfaces stage risks", passed=bool(self_evolution.get("stage_risks")), required=False),
                ],
                signals={
                    "status": self_evolution_summary.get("status"),
                    "score": float(self_evolution_summary.get("score") or 0.0),
                    "lesson_count": int(self_evolution_summary.get("lesson_count") or 0),
                },
                risks=_coerce_str_list(self_evolution.get("stage_risks")),
            )
        elif process == "packaging":
            if not quality_result:
                by_process[process] = _finalize_process(
                    blueprint,
                    [],
                    missing_reason="quality/high_quality_result.json missing",
                )
                continue
            packaging_files = [
                quality_result.get("submission_package_file"),
                quality_result.get("submission_dashboard_file"),
                quality_result.get("editor_pitch_file"),
                quality_result.get("risk_register_file"),
                quality_result.get("cover_letter_file"),
            ]
            by_process[process] = _finalize_process(
                blueprint,
                [
                    _criterion("quality_result", "Quality pipeline produced a successful result", passed=str(quality_result.get("status") or "").strip() == "success"),
                    _criterion("quality_gate", "Submission-quality gate passed", passed=quality_result.get("quality_gate_passed") is True),
                    _criterion("submission_priority", "Submission priority is present", passed=isinstance(quality_result.get("submission_priority_score"), (int, float))),
                    _criterion("submission_readiness", "Submission readiness payload exists", passed=bool(submission_readiness)),
                    _criterion("ready_or_rewrite", "Run is either ready or has rewrite trace evidence", passed=bool(submission_readiness) and (bool(submission_readiness.get("ready")) or bool(quality_result.get("rewrite_trace")))),
                    _criterion("packaging_assets", "At least one submission-facing packaging asset exists", passed=any(str(item or "").strip() for item in packaging_files), required=False),
                ],
                signals={
                    "quality_gate_passed": quality_result.get("quality_gate_passed"),
                    "submission_status": submission_readiness.get("status"),
                    "submission_priority_score": quality_result.get("submission_priority_score"),
                },
                risks=[] if quality_result.get("quality_gate_passed") else ["submission_packaging_not_ready"],
            )

    process_results = [by_process[item["process"]] for item in PROCESS_BLUEPRINTS]
    ready_count = sum(item["status"] == "ready" for item in process_results)
    blocked_count = sum(item["status"] == "blocked" for item in process_results)
    attention_count = sum(item["status"] == "needs_attention" for item in process_results)
    missing_count = sum(item["status"] == "missing" for item in process_results)
    overall_score = round(
        sum(float(item.get("score") or 0.0) for item in process_results)
        / max(len(process_results), 1),
        2,
    )
    risk_counts: dict[str, int] = {}
    reference_summary: dict[str, dict[str, Any]] = {}
    for item in process_results:
        for risk in item.get("risks") or []:
            label = str(risk).strip()
            if label:
                risk_counts[label] = risk_counts.get(label, 0) + 1
        for ref in item.get("references") or []:
            name = str((ref or {}).get("name") or "").strip()
            if not name:
                continue
            summary = reference_summary.setdefault(
                name,
                {"url": REFERENCE_URLS.get(name), "process_count": 0, "ready_count": 0, "blocked_count": 0, "processes": []},
            )
            summary["process_count"] += 1
            if item["status"] == "ready":
                summary["ready_count"] += 1
            if item["status"] == "blocked":
                summary["blocked_count"] += 1
            summary["processes"].append(item["process"])

    return {
        "schema_version": PROCESS_ALIGNMENT_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "project_root": str(resolved_root),
        "workflow_mode": research_plan.get("workflow_mode"),
        "summary": {
            "overall_score": overall_score,
            "ready_process_count": ready_count,
            "blocked_process_count": blocked_count,
            "needs_attention_process_count": attention_count,
            "missing_process_count": missing_count,
            "top_process_risks": dict(
                sorted(risk_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
            ),
        },
        "reference_summary": reference_summary,
        "process_results": process_results,
    }


def save_process_alignment(
    project_root: str | Path,
    *,
    producer: str,
) -> str:
    payload = build_process_alignment(project_root)
    return save_contract_artifact(
        project_root,
        "process_alignment",
        payload,
        producer=producer,
        depends_on=[
            "idea_cards",
            "research_plan",
            "claim_evidence_graph",
            "experiment_registry",
            "figure_spec",
            "manuscript_state",
            "review_state",
            "repair_plan",
            "self_evolution",
        ],
    )
