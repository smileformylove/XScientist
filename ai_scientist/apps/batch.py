#!/usr/bin/env python3
"""
XScientist continuous paper generator
支持连续产生论文，不仅限于workshop，所有文件输出到指定目录
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

from ai_scientist.utils.guardrail_artifacts import (
    load_guardrail_artifacts,
    result_passed_writeup_guardrails,
)
from ai_scientist.utils.pipeline_helpers import (
    find_latest_bfts_run_dir,
    find_latest_pdf_path,
    save_review_artifacts,
)
from ai_scientist.utils.integrity_workflow import (
    integrity_forensics_result_fields,
    run_integrity_forensics_for_manuscript,
    summarize_integrity_forensics_result,
)
from ai_scientist.utils.experiment_registry import (
    build_experiment_record,
    save_experiment_registry,
)
from ai_scientist.utils.experiment_report import write_experiment_report
from ai_scientist.utils.deferred_imports import load_module_attr
from ai_scientist.utils.auth_session import require_login
from ai_scientist.utils.ara_pipeline import (
    finalize_ara_for_idea,
    summarise_finalize,
)
from ai_scientist.utils.ara_minimal import export_minimal_ara
from ai_scientist.utils.manuscript_state import (
    build_manuscript_state,
    save_manuscript_state,
)
from ai_scientist.utils.fallback_audit import (
    format_strict_fallback_error,
    record_quality_fallback_if_needed,
    record_ranking_fallbacks,
    should_enforce_strict_fallbacks,
    StrictFallbackViolation,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    render_research_program_markdown,
    save_contract_artifact,
)

from ai_scientist.review_strategies import (
    SmartIterationController,
    PRESET_STRATEGIES,
)
from ai_scientist.improvement_reporter import (
    ImprovementReporter,
    print_improvement_summary,
)
from ai_scientist.utils.high_quality_pipeline import run_high_quality_pass
from ai_scientist.utils.workflow_cli import normalize_batch_workflow_args
from ai_scientist.utils.quality_workflow import (
    execute_quality_workflow_with_followups,
)
from ai_scientist.utils.review_workflow import build_review_execution_plan
from ai_scientist.utils.writeup_workflow import build_writeup_execution_plan
from ai_scientist.utils.workflow_modes import (
    build_workflow_manifest_metadata,
    list_workflow_modes,
    resolve_template_mode_for_workflow,
    resolve_workflow_mode,
)
from ai_scientist.utils.workflow_selection import (
    resolve_paper_types_for_venue,
    select_ranked_idea_candidates,
)
from ai_scientist.utils.research_planning import (
    build_claim_evidence_graph,
    build_idea_cards,
    build_research_plan,
)
from ai_scientist.utils.research_integrity import (
    build_preregistration,
    save_preregistration,
)
from ai_scientist.utils.hypothesis_archive import (
    build_hypothesis_archive,
    save_hypothesis_archive,
)
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    list_writing_profiles,
    normalize_writing_profile,
)
from ai_scientist.writing_skill_pack import list_writing_skills
from ai_scientist.utils.runtime_bootstrap import (
    initialize_runtime,
    require_model_credentials,
    resolve_writing_profile_env,
)
from ai_scientist.utils.workflow_runtime import (
    build_workflow_runtime_plan,
    execute_review_suite,
)
from ai_scientist.utils.critic_workflow import run_independent_critic_pass
from ai_scientist.utils.stage_standards import save_stage_standards
from ai_scientist.utils.truth_contracts import save_truth_contract_bundle
from ai_scientist.utils.sample_gate import (
    evaluate_and_save_sample_gate,
    save_sample_gate_plan,
)
from ai_scientist.utils.decision_log import (
    build_sample_gate_decision_options,
    build_workflow_strategy_decision_options,
    record_decision,
    record_model_provider_decisions,
)
from ai_scientist.utils.gate_preconditions import (
    GatePreconditionContext,
    UnsatisfiableGateError,
    assert_gate_preconditions_satisfiable,
)
from ai_scientist.apps.batch_experiment_artifacts import (
    _annotate_report_with_experiment_todo,
    _build_batch_experiment_agenda,
    _build_batch_experiment_agenda_markdown,
    _build_batch_experiment_ledger_rows,
    _build_batch_experiment_ledger_tsv,
    _build_batch_experiment_todo,
    _build_batch_experiment_todo_markdown,
    _build_paper_experiment_todo_tasks,
    _classify_batch_experiment_outcome,
    _coerce_str_list,
    _extract_self_review_gate_context,
    _normalize_priority,
    _priority_rank,
    _safe_read_json_dict,
    _slugify_token,
    _write_per_paper_experiment_todo_artifacts,
)

PROFESSIONAL_TEMPLATE_BY_PAPER_TYPE = {
    "normal": "neurips",
    "icbinb": "icbinb",
    "journal": "journal",
    "extended": "icbinb",
}

# 导入路径配置
from ai_scientist.config.paths import (
    PAPER_TYPES,
    create_paper_structure,
    get_batch_dir,
    get_paper_dir,
    get_paper_type_config,
    ensure_output_dirs,
    resolve_output_path,
)

_RUNTIME_IMPORT_ATTRS = {
    "create_client": ("ai_scientist.llm", "create_client"),
    "get_response_from_llm": ("ai_scientist.llm", "get_response_from_llm"),
    "generate_temp_free_idea": (
        "ai_scientist.perform_ideation_temp_free",
        "generate_temp_free_idea",
    ),
    "perform_experiments_bfts": (
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
        "perform_experiments_bfts",
    ),
    "idea_to_markdown": ("ai_scientist.treesearch.bfts_utils", "idea_to_markdown"),
    "edit_bfts_config_file": (
        "ai_scientist.treesearch.bfts_utils",
        "edit_bfts_config_file",
    ),
    "aggregate_plots": ("ai_scientist.perform_plotting", "aggregate_plots"),
    "perform_icbinb_writeup": (
        "ai_scientist.perform_icbinb_writeup",
        "perform_writeup",
    ),
    "gather_citations": (
        "ai_scientist.perform_icbinb_writeup",
        "gather_citations",
    ),
    "perform_writeup": ("ai_scientist.perform_writeup", "perform_writeup"),
    "perform_review": ("ai_scientist.perform_llm_review", "perform_review"),
    "load_paper": ("ai_scientist.perform_llm_review", "load_paper"),
    "perform_imgs_cap_ref_review": (
        "ai_scientist.perform_vlm_review",
        "perform_imgs_cap_ref_review",
    ),
    "improve_paper_with_review": (
        "ai_scientist.perform_auto_improvement",
        "improve_paper_with_review",
    ),
    "ExpertSectionWriter": (
        "ai_scientist.professional_writing_system",
        "ExpertSectionWriter",
    ),
    "ProfessionalPaperEvaluator": (
        "ai_scientist.professional_writing_system",
        "ProfessionalPaperEvaluator",
    ),
    "PAPER_TEMPLATES": (
        "ai_scientist.professional_writing_system",
        "PAPER_TEMPLATES",
    ),
    "ACADEMIC_WRITING_STANDARDS": (
        "ai_scientist.professional_writing_system",
        "ACADEMIC_WRITING_STANDARDS",
    ),
    "recommend_template": (
        "ai_scientist.professional_writing_system",
        "recommend_template",
    ),
    "list_templates": ("ai_scientist.professional_writing_system", "list_templates"),
    "SelfLearningKnowledgeBase": (
        "ai_scientist.self_learning_knowledge_base",
        "SelfLearningKnowledgeBase",
    ),
    "PatternAnalyzer": (
        "ai_scientist.self_learning_knowledge_base",
        "PatternAnalyzer",
    ),
    "AdaptiveLearningEngine": (
        "ai_scientist.adaptive_learning_engine",
        "AdaptiveLearningEngine",
    ),
    "AdaptiveWriter": ("ai_scientist.adaptive_learning_engine", "AdaptiveWriter"),
    "AutonomousEvolutionEngine": (
        "ai_scientist.autonomous_evolution",
        "AutonomousEvolutionEngine",
    ),
    "FeedbackSource": ("ai_scientist.autonomous_evolution", "FeedbackSource"),
    "BaseAgent": ("ai_scientist.agent_interface", "BaseAgent"),
    "AgentOrchestrator": ("ai_scientist.agent_interface", "AgentOrchestrator"),
    "register_agent_with_evolution": (
        "ai_scientist.agent_interface",
        "register_agent_with_evolution",
    ),
    "PaperMetadata": ("ai_scientist.paper_metadata", "PaperMetadata"),
    "MetadataRegistry": ("ai_scientist.paper_metadata", "MetadataRegistry"),
    "PaperStatus": ("ai_scientist.paper_metadata", "PaperStatus"),
    "create_paper_metadata": (
        "ai_scientist.paper_metadata",
        "create_paper_metadata",
    ),
    "AgentGuidanceAPI": (
        "ai_scientist.agent_guidance_coordinator",
        "AgentGuidanceAPI",
    ),
    "create_standardized_markers": (
        "ai_scientist.agent_guidance_coordinator",
        "create_standardized_markers",
    ),
    "AgentGuidanceCoordinator": (
        "ai_scientist.agent_guidance_coordinator",
        "AgentGuidanceCoordinator",
    ),
}


def _ensure_runtime_imports() -> None:
    runtime_globals = globals()
    if runtime_globals.get("_RUNTIME_IMPORTS_LOADED"):
        return

    for name, (module_name, attr_name) in _RUNTIME_IMPORT_ATTRS.items():
        runtime_globals[name] = load_module_attr(module_name, attr_name)
    runtime_globals["_RUNTIME_IMPORTS_LOADED"] = True


def _try_minimal_ara(
    *,
    paper_dir: Any,
    idea: dict[str, Any] | None,
    writing_profile: str | None,
    producer: str,
) -> dict[str, Any]:
    """Best-effort minimal ARA export for the non-BFTS writeup paths.

    Returns extra keys to merge into the caller's return dict. Any failure
    downgrades to an empty dict so the paper generator never dies on ARA.
    """
    try:
        paper_dir_path = Path(paper_dir)
        pdf_candidates = (
            list(paper_dir_path.glob("*.pdf")) if paper_dir_path.exists() else []
        )
        pdf = pdf_candidates[0] if pdf_candidates else None
        export = export_minimal_ara(
            project_dir=paper_dir_path,
            manuscript_pdf=pdf,
            idea=idea or {},
            writing_profile=writing_profile,
            producer=producer,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  minimal ARA export failed ({producer}): {exc}")
        return {}
    return {
        "ara_dir": str(export.root),
        "ara_manifest": str(export.manifest_path),
        "ara_node_count": export.node_count,
    }


def _write_json_artifact(
    path: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = True,
) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=ensure_ascii)
    return artifact_path


def _write_text_artifact(
    path: str | Path, content: str, *, encoding: str = "utf-8"
) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding=encoding) as f:
        f.write(content)
    return artifact_path


def _safe_load_json(path: str | Path, *, default=None):
    artifact_path = Path(path)
    if not artifact_path.exists():
        return default
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _resolve_workflow_strategy(
    *,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
    target_venue: str | None = None,
):
    workflow_spec = resolve_workflow_mode(
        workflow_mode,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
        target_venue=target_venue,
    )
    template_profile, template_capability = resolve_template_mode_for_workflow(
        workflow_spec,
        submission_mode=submission_mode,
    )
    workflow_metadata = build_workflow_manifest_metadata(workflow_spec)
    return workflow_spec, workflow_metadata, template_profile, template_capability


def _collect_requested_models(args: argparse.Namespace) -> list[str]:
    candidates = [
        args.model_ideation,
        args.model_agg_plots,
        args.model_writeup,
        args.model_writeup_small,
        args.model_citation,
        args.model_review,
        args.idea_rank_model,
        args.quality_model,
    ]
    models: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        for model in [item.strip() for item in str(value or "").split(",")]:
            if not model or model in seen:
                continue
            models.append(model)
            seen.add(model)
    return models


def _save_batch_pipeline_seed_artifacts(
    *,
    batch_dir: str | Path,
    ideas: list[dict[str, Any]],
    target_venue: str | None,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
) -> list[dict[str, Any]]:
    batch_root = Path(batch_dir).expanduser().resolve()
    workflow_spec, workflow_metadata, template_profile, template_capability = (
        _resolve_workflow_strategy(
            workflow_mode=workflow_mode,
            submission_mode=submission_mode,
            breakthrough_mode=breakthrough_mode,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
    )
    initialize_pipeline_contracts(
        batch_root,
        project_name=batch_root.name,
        template_profile=template_profile,
        template_capability=template_capability,
        pipeline_goal=workflow_spec.pipeline_goal,
        workflow_mode=workflow_metadata["workflow_mode"],
        workflow_label=workflow_metadata["workflow_label"],
        workflow_summary=workflow_metadata["workflow_summary"],
        workflow_inspirations=workflow_metadata["workflow_inspirations"],
        workflow_sequence=workflow_metadata["workflow_sequence"],
    )
    idea_cards = build_idea_cards(
        ideas,
        target_venue=target_venue,
        template_profile=template_profile,
        workflow_mode=workflow_spec.name,
    )
    save_contract_artifact(
        batch_root,
        "idea_cards",
        idea_cards,
        producer="continuous_paper_generator.batch_seed",
    )
    save_hypothesis_archive(
        batch_root,
        build_hypothesis_archive(idea_cards),
        producer="continuous_paper_generator.batch_seed",
    )
    lead_idea = idea_cards[0] if idea_cards else {}
    lead_plan = (
        build_research_plan(
            lead_idea,
            target_venue=target_venue,
            submission_mode=submission_mode,
            breakthrough_mode=breakthrough_mode,
            high_quality_mode=high_quality_mode,
        )
        if lead_idea
        else {}
    )
    research_program = render_research_program_markdown(
        project_name=batch_root.name,
        target_venue=target_venue,
        template_profile=template_profile,
        idea_name=str(lead_idea.get("name") or batch_root.name),
        hypothesis=str(lead_idea.get("core_hypothesis") or ""),
        workflow_mode=workflow_spec.name,
        workflow_summary=str(workflow_metadata["workflow_summary"]),
        workflow_inspirations=list(workflow_metadata["workflow_inspirations"]),
        workflow_sequence=list(workflow_metadata["workflow_sequence"]),
        budget=lead_plan.get("budget"),
        execution_policy=lead_plan.get("execution_policy"),
        success_criteria=[
            str(task.get("success_criterion") or "").strip()
            for task in (lead_plan.get("tasks") or [])[:4]
            if str(task.get("success_criterion") or "").strip()
        ],
        failure_handling_rules=[
            "Batch coordination records failures in progress.json and pipeline_manifest.json.",
            "Only ready artifacts should be consumed by downstream writing or review stages.",
        ],
    )
    save_contract_artifact(
        batch_root,
        "research_program",
        research_program,
        producer="continuous_paper_generator.batch_seed",
        depends_on=["idea_cards"],
    )
    save_stage_standards(batch_root)
    return idea_cards


def _save_paper_pipeline_seed_artifacts(
    *,
    paper_root: str | Path,
    idea: dict[str, Any],
    idea_idx: int,
    target_venue: str | None,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    paper_root_path = Path(paper_root).expanduser().resolve()
    workflow_spec, workflow_metadata, template_profile, template_capability = (
        _resolve_workflow_strategy(
            workflow_mode=workflow_mode,
            submission_mode=submission_mode,
            breakthrough_mode=breakthrough_mode,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
    )
    initialize_pipeline_contracts(
        paper_root_path,
        project_name=paper_root_path.name,
        template_profile=template_profile,
        template_capability=template_capability,
        pipeline_goal=workflow_spec.pipeline_goal,
        workflow_mode=workflow_metadata["workflow_mode"],
        workflow_label=workflow_metadata["workflow_label"],
        workflow_summary=workflow_metadata["workflow_summary"],
        workflow_inspirations=workflow_metadata["workflow_inspirations"],
        workflow_sequence=workflow_metadata["workflow_sequence"],
    )
    idea_cards = build_idea_cards(
        [idea],
        target_venue=target_venue,
        template_profile=template_profile,
        workflow_mode=workflow_spec.name,
    )
    idea_card = dict(idea_cards[0] if idea_cards else {})
    idea_card["idea_id"] = f"idea_{idea_idx}"
    save_contract_artifact(
        paper_root_path,
        "idea_cards",
        [idea_card],
        producer="continuous_paper_generator.paper_seed",
    )
    save_hypothesis_archive(
        paper_root_path,
        build_hypothesis_archive([idea_card]),
        producer="continuous_paper_generator.paper_seed",
    )
    research_plan = build_research_plan(
        idea_card,
        target_venue=target_venue,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
    )
    save_contract_artifact(
        paper_root_path,
        "research_plan",
        research_plan,
        producer="continuous_paper_generator.planning",
        depends_on=["idea_cards"],
    )
    save_preregistration(
        paper_root_path,
        build_preregistration(idea_card, research_plan),
        producer="continuous_paper_generator.planning",
    )
    save_truth_contract_bundle(
        paper_root_path,
        idea_card=idea_card,
        research_plan=research_plan,
        producer="continuous_paper_generator.planning",
    )
    save_sample_gate_plan(
        paper_root_path,
        research_plan=research_plan,
        producer="continuous_paper_generator.planning",
    )
    claim_evidence_graph = build_claim_evidence_graph(idea_card, research_plan)
    save_contract_artifact(
        paper_root_path,
        "claim_evidence_graph",
        claim_evidence_graph,
        producer="continuous_paper_generator.planning",
        depends_on=["research_plan"],
    )
    research_program = render_research_program_markdown(
        project_name=paper_root_path.name,
        target_venue=target_venue,
        template_profile=template_profile,
        idea_name=str(idea_card.get("name") or f"idea_{idea_idx}"),
        hypothesis=str(idea_card.get("core_hypothesis") or ""),
        workflow_mode=workflow_spec.name,
        workflow_summary=str(workflow_metadata["workflow_summary"]),
        workflow_inspirations=list(workflow_metadata["workflow_inspirations"]),
        workflow_sequence=list(workflow_metadata["workflow_sequence"]),
        budget=research_plan.get("budget"),
        execution_policy=research_plan.get("execution_policy"),
        success_criteria=[
            str(task.get("success_criterion") or "").strip()
            for task in (research_plan.get("tasks") or [])[:6]
            if str(task.get("success_criterion") or "").strip()
        ],
        failure_handling_rules=[
            "Persist failed experiment attempts in experiment_registry.jsonl.",
            "Keep claims blocked until figures and manuscript evidence are ready.",
        ],
    )
    save_contract_artifact(
        paper_root_path,
        "research_program",
        research_program,
        producer="continuous_paper_generator.planning",
        depends_on=["research_plan"],
    )
    save_stage_standards(paper_root_path)
    return (
        idea_card,
        research_plan,
        claim_evidence_graph,
        template_profile,
        template_capability,
    )


def _build_experiment_registry_rows(
    *,
    paper_root: str | Path,
    research_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    paper_root_path = Path(paper_root).expanduser().resolve()
    experiment_report = (
        _safe_load_json(paper_root_path / "experiment_report.json", default={}) or {}
    )
    stages = list(experiment_report.get("stages") or [])
    warnings = list(experiment_report.get("warnings") or [])
    latest_run_dir = experiment_report.get("latest_run_dir")
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(research_plan.get("tasks") or []):
        stage = stages[min(idx, len(stages) - 1)] if stages else {}
        best = stage.get("best") or {}
        if best:
            status = "completed"
            result_summary = {
                "metric_name": best.get("metric_name"),
                "metric_mean": best.get("metric_mean"),
                "metric_objective": best.get("metric_objective"),
                "dataset_names": best.get("dataset_names") or [task.get("dataset")],
                "seed_eval": best.get("seed_eval"),
                "delta_objective_vs_prev_stage": stage.get(
                    "delta_objective_vs_prev_stage"
                ),
                "warnings": warnings,
            }
            error_type = None
            error_message = None
            entered_storyline = idx == 0
        else:
            status = "failed" if stages or warnings else "planned"
            result_summary = {
                "warnings": warnings,
                "node_counts": stage.get("node_counts"),
            }
            error_type = (
                "missing_best_result" if stages else "missing_experiment_report"
            )
            error_message = (
                "; ".join(warnings[:3]) or "No valid experiment summary was detected."
            )
            entered_storyline = False
        rows.append(
            build_experiment_record(
                task_id=str(task.get("task_id") or f"task_{idx}"),
                dataset=str(task.get("dataset") or "dataset_to_be_selected"),
                metric=str(task.get("metric") or "primary_task_metric"),
                baseline_ref=str(task.get("baseline") or "strong_existing_baseline"),
                config={
                    "goal": task.get("goal"),
                    "priority": task.get("priority"),
                },
                status=status,
                result_summary=result_summary,
                artifacts={
                    "paper_root": str(paper_root_path),
                    "experiment_report_json": str(
                        paper_root_path / "experiment_report.json"
                    ),
                    "experiment_report_md": str(
                        paper_root_path / "experiment_report.md"
                    ),
                    "latest_run_dir": latest_run_dir,
                    "stage_dir": stage.get("stage_dir"),
                    "journal_path": stage.get("journal_path"),
                },
                error_type=error_type,
                error_message=error_message,
                entered_storyline=entered_storyline,
                budget=task.get("budget"),
                workflow_mode=research_plan.get("workflow_mode"),
                policy_name=(research_plan.get("execution_policy") or {}).get(
                    "policy_name"
                ),
                acceptance_checks=task.get("acceptance_checks"),
                acceptance_results=[
                    {
                        "check": str(check),
                        "passed": status == "completed",
                        "source": "experiment_report_best_result",
                    }
                    for check in (task.get("acceptance_checks") or [])
                ],
                budget_audit={
                    "audited": status == "completed",
                    "within_budget": status == "completed",
                    "source": "experiment_registry_builder",
                },
                budget_status="within_budget" if status == "completed" else None,
            )
        )
    return rows


def _save_manuscript_contract_state(
    *,
    paper_root: str | Path,
    paper_type: str,
    target_venue: str | None,
    writing_profile: str,
) -> dict[str, Any]:
    paper_root_path = Path(paper_root).expanduser().resolve()
    claim_evidence_graph = load_contract_artifact(
        paper_root_path,
        "claim_evidence_graph",
        default={},
    )
    figure_spec = load_contract_artifact(
        paper_root_path,
        "figure_spec",
        default={},
    )
    manuscript_state = build_manuscript_state(
        writeup_type=paper_type,
        target_venue=target_venue,
        writing_profile=writing_profile,
        skill_pack=list_writing_skills(),
        claim_evidence_graph=(
            claim_evidence_graph if isinstance(claim_evidence_graph, dict) else {}
        ),
        figure_spec=figure_spec if isinstance(figure_spec, dict) else {},
        latex_path=str(paper_root_path / "latex" / "template.tex"),
    )
    save_manuscript_state(str(paper_root_path), manuscript_state)
    return manuscript_state


def _create_paper_workspace(
    *,
    idea: Dict[str, Any],
    paper_type: str,
    default_idea_name: str | None = None,
    timestamp: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = idea.get(
        "Name",
        default_idea_name or f"idea_{resolved_timestamp}",
    )
    paper_dir = get_paper_dir(
        idea_name,
        paper_type,
        resolved_timestamp,
        output_root=output_root,
    )
    paper_structure = create_paper_structure(paper_dir)
    idea_path = _write_json_artifact(paper_structure["root"] / "idea.json", idea)
    return {
        "idea_name": idea_name,
        "timestamp": resolved_timestamp,
        "paper_dir": paper_dir,
        "paper_structure": paper_structure,
        "idea_path": idea_path,
    }


def _save_generated_manuscript(
    *,
    paper_structure: dict[str, Path],
    paper_type: str,
    outline: Any,
    full_paper: str,
    logger: Callable[[str], None] = print,
) -> Path:
    _write_json_artifact(paper_structure["latex"] / "outline.json", outline)
    latex_path = _write_text_artifact(
        paper_structure["latex"] / f"{paper_type}_paper.tex",
        full_paper,
    )
    _write_text_artifact(paper_structure["latex"] / "template.tex", full_paper)
    logger(f"✅ LaTeX文件已保存: {latex_path}")
    return latex_path


def _run_optional_quality_pass(
    *,
    enabled: bool,
    run_dir: str | Path,
    paper_type: str,
    rewrite_model: str,
    quality_model: str | None,
    target_venue: str | None,
    quality_preset: str,
    logger: Callable[[str], None],
    quality_threshold: float | None = None,
    rigor_threshold: float | None = None,
    max_quality_rewrites: int | None = None,
    autonomous_quality_followup_rounds: int = 0,
    require_quality_gate: bool = False,
    min_submission_priority: float | None = None,
    max_submission_blockers: int | None = None,
    allow_auto_improvement_fallback: bool | None = None,
    reject_on_auto_improvement_fallback: bool = False,
    resume: bool = False,
    strict_fallbacks: bool = False,
    workflow_mode: str | None = None,
) -> dict[str, Any] | None:
    if not enabled:
        return None

    quality_pass = execute_quality_workflow_with_followups(
        run_high_quality_pass_fn=run_high_quality_pass,
        run_dir=str(run_dir),
        paper_type=paper_type,
        rewrite_model=rewrite_model,
        quality_model=quality_model or rewrite_model,
        target_venue=target_venue,
        quality_preset=quality_preset,
        quality_threshold=quality_threshold,
        rigor_threshold=rigor_threshold,
        max_quality_rewrites=max_quality_rewrites,
        require_quality_gate=require_quality_gate,
        min_submission_priority=min_submission_priority,
        max_submission_blockers=max_submission_blockers,
        autonomous_followup_rounds=autonomous_quality_followup_rounds,
        allow_auto_improvement_fallback=allow_auto_improvement_fallback,
        reject_on_auto_improvement_fallback=reject_on_auto_improvement_fallback,
        resume=resume,
        logger=logger,
    )
    logger(quality_pass["summary"].replace("High-quality pass: ", "高质量模式: "))
    fallback_event = record_quality_fallback_if_needed(
        run_dir,
        quality_pass.get("quality_result"),
        producer="continuous_paper_generator.high_quality",
        strict=strict_fallbacks,
    )
    if strict_fallbacks and fallback_event:
        raise StrictFallbackViolation(
            format_strict_fallback_error(
                fallback_event,
                workflow_mode=workflow_mode,
                stage_hint="quality_review",
            )
        )
    return quality_pass


class ContinuousPaperGenerator:
    """连续论文生成器 - 支持批量生成不同类型的论文，集成自适应学习"""

    def __init__(
        self,
        research_dir: str | Path | None = None,
        batch_name: str = None,
        paper_types: Sequence[str] | None = None,
        enable_learning: bool = True,
        strict_fallbacks: bool = False,
    ):
        """
        初始化连续论文生成器

        Args:
            research_dir: 研究输出目录
            batch_name: 批次名称，为None时自动生成
            paper_types: 支持的论文类型列表
            enable_learning: 是否启用自适应学习
        """
        _ensure_runtime_imports()

        if research_dir is None:
            research_dir = resolve_output_path()
        self.research_dir = Path(research_dir).expanduser()
        self.batch_name = batch_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_output_dirs(output_root=self.research_dir)
        self.batch_dir = get_batch_dir(self.batch_name, output_root=self.research_dir)
        self.paper_types = list(paper_types or ["icbinb", "normal", "journal"])
        self.enable_learning = enable_learning
        self.strict_fallbacks = bool(strict_fallbacks)

        # 创建批次目录结构
        self._create_batch_structure()

        # 进度跟踪
        self.progress_file = self.batch_dir / "progress.json"
        self.progress = self._load_progress()

        # 初始化自适应学习系统
        if self.enable_learning:
            self.knowledge_base = SelfLearningKnowledgeBase(str(self.research_dir))
            self.learning_engine = AdaptiveLearningEngine(self.knowledge_base)
            self.adaptive_writer = AdaptiveWriter(self.learning_engine)

            # 初始化自主进化引擎
            self.evolution_engine = AutonomousEvolutionEngine(str(self.research_dir))
            self.agent_orchestrator = AgentOrchestrator()

            print(f"✅ 自适应学习系统已启用")
            print(f"✅ 自主进化系统已启用")
        else:
            self.knowledge_base = None
            self.learning_engine = None
            self.adaptive_writer = None
            self.evolution_engine = None
            self.agent_orchestrator = None

    def _create_batch_structure(self):
        """创建批次目录结构"""
        dirs = {
            "root": self.batch_dir,
            "ideas": self.batch_dir / "ideas",
            "logs": self.batch_dir / "logs",
        }

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ 创建批次目录结构: {self.batch_dir}")

    def _load_progress(self) -> dict:
        """加载进度"""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {
            "batch_name": self.batch_name,
            "started_at": datetime.now().isoformat(),
            "source_provenance": _current_generation_provenance(self.batch_dir),
            "papers_generated": [],
            "papers_completed": [],
            "papers_failed": [],
            "current_stage": "initialized",
        }

    def _save_progress(self):
        """保存进度"""
        self.progress["last_updated"] = datetime.now().isoformat()
        self.progress["source_provenance"] = _current_generation_provenance(
            self.batch_dir
        )
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=4)

    def generate_ideas(
        self,
        topic_file: str,
        num_ideas: int = 5,
        model: str = "glm-4-flash",
        num_reflections: int = 5,
    ) -> str:
        """
        生成研究想法

        Args:
            topic_file: 主题描述文件
            num_ideas: 生成想法数量
            model: 使用的模型
            num_reflections: 反思次数

        Returns:
            想法JSON文件路径
        """
        print("\n" + "=" * 80)
        print("🔬 阶段 1: 生成研究想法")
        print("=" * 80)

        self.progress["current_stage"] = "generating_ideas"
        self._save_progress()

        # 读取主题描述
        with open(topic_file, "r") as f:
            workshop_description = f.read()

        # 输出文件路径
        idea_json = str(self.batch_dir / "ideas" / "generated_ideas.json")

        # 创建客户端
        client, client_model = create_client(model)

        # 生成想法
        ideas = generate_temp_free_idea(
            idea_fname=idea_json,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            max_num_generations=num_ideas,
            num_reflections=num_reflections,
        )

        print(f"✅ 生成了 {len(ideas)} 个想法，保存到 {idea_json}")
        self.progress["num_ideas"] = len(ideas)
        self.progress["ideas_file"] = idea_json
        self._save_progress()

        return idea_json

    def load_existing_ideas(self, ideas_file: str) -> str:
        """加载已有想法"""
        print("\n" + "=" * 80)
        print("📂 加载已有想法")
        print("=" * 80)

        # 复制到批次目录
        target_ideas_file = self.batch_dir / "ideas" / "loaded_ideas.json"
        shutil.copy(ideas_file, target_ideas_file)

        with open(target_ideas_file, "r") as f:
            ideas = json.load(f)

        print(f"✅ 加载了 {len(ideas)} 个想法")
        self.progress["num_ideas"] = len(ideas)
        self.progress["ideas_file"] = str(target_ideas_file)
        self._save_progress()

        return str(target_ideas_file)

    def rank_ideas(
        self,
        ideas_json: str,
        model: str,
        top_k: int = None,
        target_venue: str = None,
    ) -> List[Dict]:
        """对想法进行排序，优先选择更适合高质量论文的想法。"""
        with open(ideas_json, "r") as f:
            ideas = json.load(f)

        ranking_file = self.batch_dir / "ideas" / "idea_rankings.json"
        selected_indices, rankings = select_ranked_idea_candidates(
            ideas,
            ranking_enabled=True,
            ranking_model=model,
            target_venue=target_venue,
            prioritize_breakthrough=(target_venue == "nature"),
            research_root=self.research_dir,
            ranking_output_path=ranking_file,
            default_indices=[0],
            use_ranked_all=True,
            limit=top_k,
        )

        if rankings:
            event = record_ranking_fallbacks(
                self.batch_dir,
                rankings,
                producer="continuous_paper_generator.idea_ranking",
                strict=self.strict_fallbacks,
            )
            if self.strict_fallbacks and event:
                raise StrictFallbackViolation(
                    format_strict_fallback_error(
                        event,
                        workflow_mode="batch_ranking",
                        stage_hint="idea_ranking",
                    )
                )
            print(
                f"🏅 Top ranked idea: #{rankings[0].get('idea_idx')} score={rankings[0].get('ranking_score')} total={rankings[0].get('total_score')}"
            )
        self.progress["idea_rankings_file"] = str(ranking_file)
        if top_k is not None:
            self.progress["selected_idea_indices"] = selected_indices
        self._save_progress()
        return rankings

    def generate_paper_batch(
        self,
        ideas_json: str,
        paper_type: str = "icbinb",
        idea_indices: List[int] = None,
        submission_mode: bool = False,
        bfts_config: str = "bfts_config.yaml",
        model_writeup: str = "glm-4-plus",
        model_citation: str = "glm-4-air",
        model_review: str = "glm-4-plus",
        model_agg_plots: str = "glm-4-flash",
        model_writeup_small: str = "glm-4-air",
        num_cite_rounds: int = 15,
        writeup_retries: int = 3,
        improvement_rounds: int = 1,
        num_workers: int = 1,
        high_quality_mode: bool = False,
        quality_preset: str = "balanced",
        quality_model: str = None,
        target_venue: str = None,
        quality_threshold: float = None,
        rigor_threshold: float = None,
        quality_rewrite_rounds: int = None,
        autonomous_quality_followup_rounds: int = 0,
        require_quality_gate: bool = False,
        min_submission_priority: float = None,
        max_submission_blockers: int = None,
        review_reflections: int = 1,
        review_ensemble: int = 1,
        review_fewshot: int = 1,
        review_temperature: float = 0.75,
        review_strategy: str = None,
        writing_profile: str = DEFAULT_WRITING_PROFILE,
        writing_audit_rounds: int = 0,
        strict_writing_guardrails: bool = False,
        guardrail_repair_rounds: int = 1,
        workflow_mode: str = "classic_pipeline",
        integrity_forensics_enabled: bool | None = None,
        requested_workflow_mode: str | None = None,
    ) -> List[dict]:
        """
        批量生成指定类型的论文

        Args:
            ideas_json: 想法JSON文件路径
            paper_type: 论文类型
            idea_indices: 要处理的想法索引列表
            bfts_config: BFTS实验配置文件路径 (影响搜索深度/seed/并行度/超时等)
            model_writeup: 写作模型
            model_citation: 引用检索模型
            model_review: 审查模型
            model_agg_plots: 图表聚合模型
            num_cite_rounds: 引用检索轮数
            writeup_retries: 写作重试次数
            improvement_rounds: 改进轮数
            num_workers: 并行worker数量

        Returns:
            结果列表
        """
        print("\n" + "=" * 80)
        print(f"📝 生成 {paper_type.upper()} 类型论文")
        print("=" * 80)

        # 获取论文类型配置
        paper_config = get_paper_type_config(paper_type)
        print(f"论文类型: {paper_config['name']}")
        print(f"页数限制: {paper_config['page_limit']}")
        print(f"描述: {paper_config['description']}")
        print(f"编排模式: {workflow_mode}")
        resolved_integrity_forensics_enabled = (
            bool(integrity_forensics_enabled)
            if integrity_forensics_enabled is not None
            else bool(submission_mode or high_quality_mode)
        )

        self.progress["current_stage"] = f"generating_{paper_type}_papers"
        self._save_progress()

        # 加载想法
        with open(ideas_json, "r") as f:
            ideas = json.load(f)

        # 确定要处理的想法索引
        if idea_indices is None:
            idea_indices = list(range(len(ideas)))

        # 准备参数
        args_list = []
        for idx in idea_indices:
            if idx >= len(ideas):
                continue
            args_list.append(
                (
                    str(self.batch_dir),
                    str(self.research_dir),
                    idx,
                    ideas[idx],
                    paper_type,
                    bfts_config,
                    model_writeup,
                    model_citation,
                    model_review,
                    model_agg_plots,
                    model_writeup_small,
                    num_cite_rounds,
                    writeup_retries,
                    improvement_rounds,
                    high_quality_mode,
                    quality_preset,
                    quality_model,
                    target_venue,
                    quality_threshold,
                    rigor_threshold,
                    quality_rewrite_rounds,
                    autonomous_quality_followup_rounds,
                    require_quality_gate,
                    min_submission_priority,
                    max_submission_blockers,
                    review_reflections,
                    review_ensemble,
                    review_fewshot,
                    review_temperature,
                    review_strategy,
                    writing_profile,
                    writing_audit_rounds,
                    strict_writing_guardrails,
                    guardrail_repair_rounds,
                    workflow_mode,
                    submission_mode,
                    self.strict_fallbacks,
                    resolved_integrity_forensics_enabled,
                    requested_workflow_mode or workflow_mode,
                )
            )

        # 执行论文生成
        results = []
        if num_workers > 1 and len(args_list) > 1:
            # 并行处理
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_process_single_paper, args): args[2]
                    for args in args_list
                }

                for future in as_completed(futures):
                    idea_idx = futures[future]
                    try:
                        result = _attach_source_provenance_to_result(
                            future.result(),
                            self.progress.get("source_provenance")
                            or _current_generation_provenance(self.batch_dir),
                        )
                        results.append(result)

                        # 更新进度
                        if result["status"] == "success":
                            self.progress["papers_completed"].append(result)
                        else:
                            self.progress["papers_failed"].append(result)

                        self.progress["papers_generated"].append(result)
                        self._save_progress()

                    except Exception as e:
                        print(f"❌ 想法 #{idea_idx} 执行出错: {e}")
                        traceback.print_exc()

        else:
            # 串行处理
            for args in args_list:
                result = _attach_source_provenance_to_result(
                    _process_single_paper(args),
                    self.progress.get("source_provenance")
                    or _current_generation_provenance(self.batch_dir),
                )
                results.append(result)

                if result["status"] == "success":
                    self.progress["papers_completed"].append(result)
                else:
                    self.progress["papers_failed"].append(result)

                self.progress["papers_generated"].append(result)
                self._save_progress()

        # 打印结果摘要
        print("\n" + "=" * 80)
        print(f"📊 {paper_type.upper()} 论文生成完成")
        print("=" * 80)
        for r in results:
            status_icon = "✅" if r["status"] == "success" else "❌"
            print(f"{status_icon} 想法 #{r['idea_idx']}: {r['status']}")
            if r["status"] == "success" and "pdf_path" in r:
                print(f"   📄 {r['pdf_path']}")

        return results

    def generate_all_paper_types(
        self,
        ideas_json: str,
        idea_indices: List[int] = None,
        **kwargs,
    ) -> Dict[str, List[dict]]:
        """
        为所有支持的论文类型生成论文

        Args:
            ideas_json: 想法JSON文件路径
            idea_indices: 要处理的想法索引列表
            **kwargs: 传递给generate_paper_batch的其他参数

        Returns:
            各类型论文的结果字典
        """
        all_results = {}

        for paper_type in self.paper_types:
            print("\n" + "=" * 80)
            print(f"🔄 开始生成 {paper_type.upper()} 类型论文")
            print("=" * 80)

            results = self.generate_paper_batch(
                ideas_json=ideas_json,
                paper_type=paper_type,
                idea_indices=idea_indices,
                **kwargs,
            )

            all_results[paper_type] = results

            # 打印阶段性总结
            success_count = sum(1 for r in results if r["status"] == "success")
            print(f"\n✅ {paper_type.upper()}: {success_count}/{len(results)} 成功")

        return all_results

    def generate_summary_report(self) -> str:
        """生成总结报告"""
        print("\n" + "=" * 80)
        print("📋 生成总结报告")
        print("=" * 80)

        completed = self.progress.get("papers_completed", [])
        failed = self.progress.get("papers_failed", [])

        report = {
            "batch_name": self.batch_name,
            "research_dir": str(self.research_dir),
            "batch_dir": str(self.batch_dir),
            "generated_at": datetime.now().isoformat(),
            "source_provenance": self.progress.get("source_provenance")
            or _current_generation_provenance(self.batch_dir),
            "idea_rankings_file": self.progress.get("idea_rankings_file"),
            "selected_idea_indices": self.progress.get("selected_idea_indices"),
            "pipeline_manifest_file": str(self.batch_dir / "pipeline_manifest.json"),
            "idea_cards_file": self.progress.get("idea_cards_file"),
            "research_program_file": self.progress.get("research_program_file"),
            "statistics": {
                "total_papers": len(completed) + len(failed),
                "completed": len(completed),
                "failed": len(failed),
                "by_type": {},
            },
            "completed_papers": completed,
            "failed_papers": failed,
            "quality_summary": {
                "avg_quality_score": None,
                "avg_rigor_score": None,
                "avg_claim_support_score": None,
                "gate_passed": 0,
                "gate_failed": 0,
                "submission_ready": 0,
                "guardrail_passed": 0,
                "guardrail_blocked": 0,
                "guardrail_blocking_reasons": {},
                "integrity_forensics_enabled": 0,
                "integrity_forensics_completed": 0,
                "integrity_forensics_hard_flags": 0,
                "integrity_forensics_soft_flags": 0,
                "integrity_forensics_status_counts": {},
                "integrity_forensics_verdict_counts": {},
                "by_venue": {},
                "top_papers": [],
            },
            "experiment_ledger_file": None,
            "experiment_agenda_file": None,
            "experiment_todo_file": None,
            "experiment_todo_markdown_file": None,
            "experiment_todo_count": 0,
            "experiment_todo_p0_count": 0,
            "pipeline_summary": {
                "workflow_mode_counts": {},
                "source_workflow_mode_counts": {},
                "source_archetype_counts": {},
                "source_batch_profile_counts": {},
                "template_profile_counts": {},
                "template_capability_counts": {},
                "artifacts_with_research_plan": 0,
                "artifacts_with_claim_graph": 0,
                "artifacts_with_experiment_registry": 0,
                "artifacts_with_figure_spec": 0,
                "artifacts_with_manuscript_state": 0,
                "artifacts_with_review_state": 0,
                "artifacts_with_repair_plan": 0,
                "artifacts_with_self_evolution": 0,
                "artifacts_with_stage_standards": 0,
            },
        }

        # 按类型统计
        for paper in completed:
            paper_type = paper.get("paper_type", "unknown")
            if paper_type not in report["statistics"]["by_type"]:
                report["statistics"]["by_type"][paper_type] = {
                    "completed": 0,
                    "failed": 0,
                }
            report["statistics"]["by_type"][paper_type]["completed"] += 1

        for paper in failed:
            paper_type = paper.get("paper_type", "unknown")
            if paper_type not in report["statistics"]["by_type"]:
                report["statistics"]["by_type"][paper_type] = {
                    "completed": 0,
                    "failed": 0,
                }
            report["statistics"]["by_type"][paper_type]["failed"] += 1

        quality_scores = [
            paper.get("quality_score")
            for paper in completed
            if isinstance(paper.get("quality_score"), (int, float))
        ]
        rigor_scores = [
            paper.get("rigor_score")
            for paper in completed
            if isinstance(paper.get("rigor_score"), (int, float))
        ]
        claim_support_scores = [
            paper.get("claim_support_score")
            for paper in completed
            if isinstance(paper.get("claim_support_score"), (int, float))
        ]
        priority_scores = [
            paper.get("submission_priority_score")
            for paper in completed
            if isinstance(paper.get("submission_priority_score"), (int, float))
        ]
        followup_rounds = [
            paper.get("autonomous_followup_rounds_run")
            for paper in completed
            if isinstance(paper.get("autonomous_followup_rounds_run"), int)
        ]
        gate_passed = sum(
            1 for paper in completed if paper.get("quality_gate_passed") is True
        )
        gate_failed = sum(
            1 for paper in completed if paper.get("quality_gate_passed") is False
        )
        submission_ready = sum(
            1 for paper in completed if paper.get("quality_status") == "pass"
        )
        submission_accepted = sum(
            1
            for paper in completed
            if paper.get("submission_acceptance_passed") is True
        )
        guardrail_passed = sum(
            1
            for paper in (completed + failed)
            if result_passed_writeup_guardrails(paper)
        )
        guardrail_blocked = sum(
            1 for paper in failed if paper.get("guardrail_blocking") is True
        )
        guardrail_reason_counts: Dict[str, int] = {}
        integrity_status_counts: Dict[str, int] = {}
        integrity_verdict_counts: Dict[str, int] = {}
        workflow_mode_counts: Dict[str, int] = {}
        source_workflow_mode_counts: Dict[str, int] = {}
        source_archetype_counts: Dict[str, int] = {}
        source_batch_profile_counts: Dict[str, int] = {}
        template_profile_counts: Dict[str, int] = {}
        template_capability_counts: Dict[str, int] = {}
        execution_policy_counts: Dict[str, int] = {}
        review_role_counts: Dict[str, int] = {}
        artifacts_with_research_plan = 0
        artifacts_with_claim_graph = 0
        artifacts_with_experiment_registry = 0
        artifacts_with_figure_spec = 0
        artifacts_with_manuscript_state = 0
        artifacts_with_review_state = 0
        artifacts_with_repair_plan = 0
        artifacts_with_self_evolution = 0
        artifacts_with_stage_standards = 0
        for paper in failed:
            reasons = paper.get("guardrail_blocking_reasons", [])
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                key = str(reason).strip()
                if not key:
                    continue
                guardrail_reason_counts[key] = guardrail_reason_counts.get(key, 0) + 1
        for paper in completed + failed:
            workflow_mode = str(paper.get("workflow_mode") or "unknown")
            template_profile = str(paper.get("template_profile") or "unknown")
            template_capability = str(paper.get("template_capability") or "unknown")
            workflow_mode_counts[workflow_mode] = (
                workflow_mode_counts.get(workflow_mode, 0) + 1
            )
            template_profile_counts[template_profile] = (
                template_profile_counts.get(template_profile, 0) + 1
            )
            template_capability_counts[template_capability] = (
                template_capability_counts.get(template_capability, 0) + 1
            )
            integrity = summarize_integrity_forensics_result(paper)
            if integrity.get("enabled") is True:
                status = str(integrity.get("status") or "unknown")
                integrity_status_counts[status] = (
                    integrity_status_counts.get(status, 0) + 1
                )
                verdict = str(integrity.get("verdict") or "unknown")
                integrity_verdict_counts[verdict] = (
                    integrity_verdict_counts.get(verdict, 0) + 1
                )
            source_workflow_mode = str(paper.get("source_workflow_mode") or "unknown")
            source_archetype = str(paper.get("source_archetype") or "unknown")
            source_batch_profile = str(paper.get("source_batch_profile") or "unknown")
            source_workflow_mode_counts[source_workflow_mode] = (
                source_workflow_mode_counts.get(source_workflow_mode, 0) + 1
            )
            source_archetype_counts[source_archetype] = (
                source_archetype_counts.get(source_archetype, 0) + 1
            )
            source_batch_profile_counts[source_batch_profile] = (
                source_batch_profile_counts.get(source_batch_profile, 0) + 1
            )
            execution_policy = str(paper.get("execution_policy") or "unknown")
            execution_policy_counts[execution_policy] = (
                execution_policy_counts.get(execution_policy, 0) + 1
            )
            review_roles = paper.get("review_roles_used") or []
            if isinstance(review_roles, list):
                for role in review_roles:
                    key = str(role).strip()
                    if not key:
                        continue
                    review_role_counts[key] = review_role_counts.get(key, 0) + 1
            if paper.get("research_plan_file"):
                artifacts_with_research_plan += 1
            if paper.get("claim_evidence_graph_file"):
                artifacts_with_claim_graph += 1
            if paper.get("experiment_registry_file"):
                artifacts_with_experiment_registry += 1
            if paper.get("figure_spec_file"):
                artifacts_with_figure_spec += 1
            if paper.get("manuscript_state_file"):
                artifacts_with_manuscript_state += 1
            if paper.get("review_state_file"):
                artifacts_with_review_state += 1
            if paper.get("repair_plan_file"):
                artifacts_with_repair_plan += 1
            if paper.get("self_evolution_file"):
                artifacts_with_self_evolution += 1
            if paper.get("stage_standards_file"):
                artifacts_with_stage_standards += 1
        by_venue = {}
        for paper in completed:
            venue = paper.get("target_venue") or "unknown"
            by_venue.setdefault(
                venue, {"count": 0, "gate_passed": 0, "submission_accepted": 0}
            )
            by_venue[venue]["count"] += 1
            if paper.get("quality_gate_passed") is True:
                by_venue[venue]["gate_passed"] += 1
            if paper.get("submission_acceptance_passed") is True:
                by_venue[venue]["submission_accepted"] += 1
        ranked_completed = sorted(
            completed,
            key=lambda paper: (
                paper.get("submission_acceptance_passed") is True,
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
                paper.get("quality_gate_passed") is True,
                (
                    paper.get("quality_score")
                    if isinstance(paper.get("quality_score"), (int, float))
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
                -(
                    paper.get("unsupported_claims_count")
                    if isinstance(paper.get("unsupported_claims_count"), int)
                    else 999
                ),
            ),
            reverse=True,
        )

        report["quality_summary"] = {
            "avg_quality_score": (
                (sum(quality_scores) / len(quality_scores)) if quality_scores else None
            ),
            "avg_rigor_score": (
                (sum(rigor_scores) / len(rigor_scores)) if rigor_scores else None
            ),
            "avg_claim_support_score": (
                (sum(claim_support_scores) / len(claim_support_scores))
                if claim_support_scores
                else None
            ),
            "avg_submission_priority_score": (
                (sum(priority_scores) / len(priority_scores))
                if priority_scores
                else None
            ),
            "avg_autonomous_followup_rounds": (
                (sum(followup_rounds) / len(followup_rounds))
                if followup_rounds
                else None
            ),
            "gate_passed": gate_passed,
            "gate_failed": gate_failed,
            "submission_ready": submission_ready,
            "submission_accepted": submission_accepted,
            "guardrail_passed": guardrail_passed,
            "guardrail_blocked": guardrail_blocked,
            "guardrail_blocking_reasons": guardrail_reason_counts,
            "integrity_forensics_enabled": sum(
                1
                for paper in completed + failed
                if summarize_integrity_forensics_result(paper).get("enabled") is True
            ),
            "integrity_forensics_completed": integrity_status_counts.get(
                "completed", 0
            ),
            "integrity_forensics_hard_flags": integrity_verdict_counts.get(
                "HARD_FLAGS", 0
            ),
            "integrity_forensics_soft_flags": integrity_verdict_counts.get(
                "SOFT_FLAGS", 0
            ),
            "integrity_forensics_status_counts": integrity_status_counts,
            "integrity_forensics_verdict_counts": integrity_verdict_counts,
            "by_venue": by_venue,
            "top_papers": ranked_completed[:5],
        }
        report["pipeline_summary"] = {
            "workflow_mode_counts": workflow_mode_counts,
            "source_workflow_mode_counts": source_workflow_mode_counts,
            "source_archetype_counts": source_archetype_counts,
            "source_batch_profile_counts": source_batch_profile_counts,
            "template_profile_counts": template_profile_counts,
            "template_capability_counts": template_capability_counts,
            "execution_policy_counts": execution_policy_counts,
            "review_role_counts": review_role_counts,
            "artifacts_with_research_plan": artifacts_with_research_plan,
            "artifacts_with_claim_graph": artifacts_with_claim_graph,
            "artifacts_with_experiment_registry": artifacts_with_experiment_registry,
            "artifacts_with_figure_spec": artifacts_with_figure_spec,
            "artifacts_with_manuscript_state": artifacts_with_manuscript_state,
            "artifacts_with_review_state": artifacts_with_review_state,
            "artifacts_with_repair_plan": artifacts_with_repair_plan,
            "artifacts_with_self_evolution": artifacts_with_self_evolution,
            "artifacts_with_stage_standards": artifacts_with_stage_standards,
        }

        # 保存报告
        report_file = self.batch_dir / "final_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)

        shortlist_file = self.batch_dir / "submission_shortlist.md"
        shortlist_lines = ["# Submission Shortlist", ""]
        for paper in report["quality_summary"]["top_papers"]:
            integrity = summarize_integrity_forensics_result(paper)
            shortlist_lines.extend(
                [
                    f"## idea #{paper.get('idea_idx')} — {paper.get('idea_name', paper.get('paper_type'))}",
                    f"- Paper type: {paper.get('paper_type')}",
                    f"- Target venue: {paper.get('target_venue')}",
                    f"- Submission priority: {paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')})",
                    f"- Blockers: {paper.get('blocker_count')}",
                    f"- Quality: {paper.get('quality_score')}",
                    f"- Breakthrough: {paper.get('breakthrough_score')}",
                    f"- Rigor: {paper.get('rigor_score')}",
                    f"- Claim support: {paper.get('claim_support_score')}",
                    f"- Gate passed: {paper.get('quality_gate_passed')}",
                    f"- Accepted: {paper.get('submission_acceptance_passed')}",
                    f"- Integrity forensics: {integrity.get('verdict')} "
                    f"({integrity.get('status')}, findings={integrity.get('findings')})",
                    f"- Integrity report: {integrity.get('report_file')}",
                    f"- Autonomous followups: {paper.get('autonomous_followup_rounds_run')}",
                    f"- Submission package: {paper.get('submission_package_file')}",
                    f"- Submission dashboard: {paper.get('submission_dashboard_file')}",
                    f"- PDF: {paper.get('pdf_path')}",
                    "",
                ]
            )
        shortlist_file.write_text("\n".join(shortlist_lines) + "\n", encoding="utf-8")

        experiment_ledger_rows = _build_batch_experiment_ledger_rows(report)
        experiment_ledger_file = self.batch_dir / "experiment_ledger.tsv"
        experiment_ledger_file.write_text(
            _build_batch_experiment_ledger_tsv(experiment_ledger_rows),
            encoding="utf-8",
        )
        experiment_agenda = _build_batch_experiment_agenda(report)
        experiment_agenda_file = self.batch_dir / "experiment_agenda.md"
        experiment_agenda_file.write_text(
            _build_batch_experiment_agenda_markdown(experiment_agenda),
            encoding="utf-8",
        )
        experiment_todo = _build_batch_experiment_todo(report)
        experiment_todo_file = self.batch_dir / "experiment_todo.json"
        experiment_todo_markdown_file = self.batch_dir / "experiment_todo.md"
        experiment_todo_file.write_text(
            json.dumps(experiment_todo, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        experiment_todo_markdown_file.write_text(
            _build_batch_experiment_todo_markdown(experiment_todo),
            encoding="utf-8",
        )
        _write_per_paper_experiment_todo_artifacts(experiment_todo)
        _annotate_report_with_experiment_todo(report, experiment_todo)
        report["experiment_ledger_file"] = str(experiment_ledger_file)
        report["experiment_agenda_file"] = str(experiment_agenda_file)
        report["experiment_todo_file"] = str(experiment_todo_file)
        report["experiment_todo_markdown_file"] = str(experiment_todo_markdown_file)
        report["experiment_todo_count"] = int(
            (experiment_todo.get("counts") or {}).get("total_tasks") or 0
        )
        report["experiment_todo_p0_count"] = int(
            (experiment_todo.get("counts") or {}).get("p0_tasks") or 0
        )

        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)

        # 打印摘要
        print("\n" + "=" * 80)
        print("🎉 批次完成!")
        print("=" * 80)
        print(f"批次名称: {self.batch_name}")
        print(f"输出目录: {self.batch_dir}")
        print(f"实验账本: {experiment_ledger_file}")
        print(f"实验 agenda: {experiment_agenda_file}")
        print(f"实验 TODO: {experiment_todo_file}")
        if report.get("selected_idea_indices"):
            print(f"选择的 idea: {report['selected_idea_indices']}")
        print(f"总计论文: {report['statistics']['total_papers']}")
        print(f"成功完成: {report['statistics']['completed']}")
        print(f"失败: {report['statistics']['failed']}")
        if report["quality_summary"]["avg_quality_score"] is not None:
            print(f"平均质量分: {report['quality_summary']['avg_quality_score']:.2f}")
        if report["quality_summary"]["avg_rigor_score"] is not None:
            print(f"平均严谨性: {report['quality_summary']['avg_rigor_score']:.2f}")
        if report["quality_summary"]["avg_claim_support_score"] is not None:
            print(
                f"平均论证支撑: {report['quality_summary']['avg_claim_support_score']:.2f}"
            )
        if report["quality_summary"]["avg_submission_priority_score"] is not None:
            print(
                f"平均投稿优先级: {report['quality_summary']['avg_submission_priority_score']:.2f}"
            )
        if report["quality_summary"]["avg_autonomous_followup_rounds"] is not None:
            print(
                f"平均自动质量补跑轮数: {report['quality_summary']['avg_autonomous_followup_rounds']:.2f}"
            )
        print(f"质量门槛通过: {report['quality_summary']['gate_passed']}")
        print(f"质量门槛未过: {report['quality_summary']['gate_failed']}")
        print(f"投稿准备完成: {report['quality_summary']['submission_ready']}")
        print(f"投稿接受标准达标: {report['quality_summary']['submission_accepted']}")
        print(f"写作守护通过: {report['quality_summary'].get('guardrail_passed', 0)}")
        print(f"写作守护拦截: {report['quality_summary'].get('guardrail_blocked', 0)}")
        print(
            "Integrity forensics: "
            f"completed={report['quality_summary'].get('integrity_forensics_completed', 0)} "
            f"hard={report['quality_summary'].get('integrity_forensics_hard_flags', 0)} "
            f"soft={report['quality_summary'].get('integrity_forensics_soft_flags', 0)}"
        )
        if report["quality_summary"].get("guardrail_blocking_reasons"):
            print("写作守护拦截原因:")
            for reason, count in sorted(
                report["quality_summary"]["guardrail_blocking_reasons"].items(),
                key=lambda item: (-item[1], item[0]),
            ):
                print(f"  - {reason}: {count}")
        print("\n按类型统计:")
        for paper_type, stats in report["statistics"]["by_type"].items():
            print(
                f"  {paper_type}: {stats['completed']}/{stats['completed'] + stats['failed']} 成功"
            )

        if report["quality_summary"]["by_venue"]:
            print("\n按 venue 统计:")
            for venue, stats in sorted(report["quality_summary"]["by_venue"].items()):
                print(
                    f"  {venue}: {stats['gate_passed']}/{stats['count']} 通过质量门槛, {stats['submission_accepted']}/{stats['count']} 达到投稿接受标准"
                )

        if report["quality_summary"]["top_papers"]:
            print("\n推荐优先查看:")
            for paper in report["quality_summary"]["top_papers"][:3]:
                print(
                    f"  - idea #{paper.get('idea_idx')} [{paper.get('paper_type')}] "
                    f"priority={paper.get('submission_priority_score')} blockers={paper.get('blocker_count')} "
                    f"quality={paper.get('quality_score')} rigor={paper.get('rigor_score')} claim={paper.get('claim_support_score')} "
                    f"gate={paper.get('quality_gate_passed')} accepted={paper.get('submission_acceptance_passed')}"
                )

        print(f"\n📄 报告已保存到: {report_file}")
        print(f"📝 投稿 shortlist: {shortlist_file}")

        return str(report_file)

    def generate_paper_with_professional_writing(
        self,
        idea: Dict,
        paper_type: str = "neurips",
        experiment_results: Dict = None,
        model: str = "claude-3-5-sonnet",
        enable_evaluation: bool = True,
        target_venue: str = None,
        high_quality_mode: bool = False,
        quality_preset: str = "balanced",
        writing_profile: str = DEFAULT_WRITING_PROFILE,
    ) -> Dict:
        """
        使用专业写作系统生成高质量论文

        Args:
            idea: 研究想法
            paper_type: 论文类型 (neurips, iclr, cvpr, icbinb, journal)
            experiment_results: 实验结果
            model: 使用的模型
            enable_evaluation: 是否启用质量评估

        Returns:
            生成结果
        """
        print("\n" + "=" * 80)
        print(f"✍️  使用专业写作系统生成 {paper_type.upper()} 论文")
        print("=" * 80)

        workspace = _create_paper_workspace(
            idea=idea,
            paper_type=paper_type,
            output_root=self.research_dir,
        )
        idea_name = workspace["idea_name"]
        paper_dir = workspace["paper_dir"]
        paper_structure = workspace["paper_structure"]

        try:
            # 创建专业写作器
            writer = ExpertSectionWriter(
                template=target_venue
                or PROFESSIONAL_TEMPLATE_BY_PAPER_TYPE.get(paper_type, paper_type),
                model=model,
                writing_profile=writing_profile,
            )

            print(f"\n📝 步骤 1: 生成详细大纲")
            outline = writer.generate_detailed_outline(idea, experiment_results)

            print(f"\n📝 步骤 2: 编写完整论文")
            full_paper = writer.write_full_paper(
                idea=idea,
                experiment_results=experiment_results,
                iterative=True,
            )

            latex_path = _save_generated_manuscript(
                paper_structure=paper_structure,
                paper_type=paper_type,
                outline=outline,
                full_paper=full_paper,
            )

            quality_pass = _run_optional_quality_pass(
                enabled=high_quality_mode,
                run_dir=paper_structure["root"],
                paper_type=paper_type,
                rewrite_model=model,
                quality_model=model,
                target_venue=target_venue,
                quality_preset=quality_preset,
                logger=print,
                strict_fallbacks=self.strict_fallbacks,
                workflow_mode="professional_writing",
            )
            quality_result = (
                quality_pass["quality_result"] if quality_pass is not None else None
            )

            # 质量评估
            if enable_evaluation:
                print(f"\n📊 步骤 3: 评估论文质量")

                evaluator = ProfessionalPaperEvaluator(
                    template=target_venue
                    or PROFESSIONAL_TEMPLATE_BY_PAPER_TYPE.get(paper_type, paper_type),
                    model="gpt-4o",
                )

                evaluation = evaluator.evaluate_paper_quality(
                    paper_content=full_paper,
                    idea=idea,
                )

                # 保存评估结果
                _write_json_artifact(
                    paper_structure["reviews"] / "quality_evaluation.json",
                    evaluation,
                    ensure_ascii=False,
                )

                print(f"\n📊 质量评估结果:")
                print(f"   总分: {evaluation['overall']['score']:.1f}/5")
                print(f"   等级: {evaluation['overall']['level']}")

                if evaluation["overall"].get("strengths"):
                    print(f"\n   优点:")
                    for strength in evaluation["overall"]["strengths"][:3]:
                        print(f"      • {strength}")

                if evaluation["overall"].get("weaknesses"):
                    print(f"\n   需改进:")
                    for weakness in evaluation["overall"]["weaknesses"][:3]:
                        print(f"      • {weakness}")

                return {
                    "status": "success",
                    "paper_dir": str(paper_dir),
                    "latex_path": str(latex_path),
                    "evaluation": evaluation,
                    "idea_name": idea_name,
                    "paper_type": paper_type,
                    "writing_profile": writing_profile,
                    "quality_result": quality_result,
                    **_try_minimal_ara(
                        paper_dir=paper_dir,
                        idea=idea,
                        writing_profile=writing_profile,
                        producer="continuous_paper_generator.professional_writing",
                    ),
                }
            else:
                return {
                    "status": "success",
                    "paper_dir": str(paper_dir),
                    "latex_path": str(latex_path),
                    "idea_name": idea_name,
                    "paper_type": paper_type,
                    "writing_profile": writing_profile,
                    "quality_result": quality_result,
                    **_try_minimal_ara(
                        paper_dir=paper_dir,
                        idea=idea,
                        writing_profile=writing_profile,
                        producer="continuous_paper_generator.professional_writing",
                    ),
                }

        except StrictFallbackViolation as exc:
            return {
                "status": "failed",
                "stage": "quality_fallback_blocked",
                "reason": str(exc),
                "idea_name": idea_name,
                "paper_type": paper_type,
            }
        except Exception as e:
            print(f"❌ 专业写作失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "idea_name": idea_name,
                "paper_type": paper_type,
                "writing_profile": writing_profile,
            }

    def generate_paper_with_adaptive_learning(
        self,
        idea: Dict,
        paper_type: str = "neurips",
        experiment_results: Dict = None,
        enable_evaluation: bool = True,
        learn_from_result: bool = True,
        target_venue: str = None,
        high_quality_mode: bool = False,
        quality_preset: str = "balanced",
        writing_profile: str = DEFAULT_WRITING_PROFILE,
    ) -> Dict:
        """
        使用自适应学习系统生成高质量论文

        Args:
            idea: 研究想法
            paper_type: 论文类型
            experiment_results: 实验结果
            enable_evaluation: 是否启用质量评估
            learn_from_result: 是否从结果中学习

        Returns:
            生成结果
        """
        if not self.enable_learning:
            print("⚠️  自适应学习未启用，使用标准写作系统")
            return self.generate_paper_with_professional_writing(
                idea,
                paper_type,
                experiment_results,
                enable_evaluation=enable_evaluation,
                target_venue=target_venue,
                high_quality_mode=high_quality_mode,
                quality_preset=quality_preset,
                writing_profile=writing_profile,
            )

        print("\n" + "=" * 80)
        print(f"🧠 使用自适应学习系统生成 {paper_type.upper()} 论文")
        print("=" * 80)

        workspace = _create_paper_workspace(
            idea=idea,
            paper_type=paper_type,
            output_root=self.research_dir,
        )
        idea_name = workspace["idea_name"]
        timestamp = workspace["timestamp"]
        paper_dir = workspace["paper_dir"]
        paper_structure = workspace["paper_structure"]

        try:
            # 获取自适应推荐
            print(f"\n🧠 步骤 1: 分析历史模式并生成推荐")
            recommendation = self.learning_engine.recommend_strategy(
                idea=idea,
                paper_type=paper_type,
                context={"experiment_results": experiment_results},
            )

            # 保存推荐
            _write_json_artifact(
                paper_structure["root"] / "recommendation.json",
                recommendation,
                ensure_ascii=False,
            )

            print(f"   成功概率预测: {recommendation['success_probability']:.1%}")
            print(f"   推荐置信度: {recommendation['confidence']:.1%}")

            if recommendation.get("similar_papers"):
                print(
                    f"   找到 {len(recommendation['similar_papers'])} 篇相似的成功论文"
                )
            self_evolution_guidance = (
                recommendation.get("self_evolution_guidance") or {}
            )
            if self_evolution_guidance.get("top_agentic_defaults"):
                print(
                    "   Self-Evolution Defaults: "
                    + " | ".join(self_evolution_guidance["top_agentic_defaults"][:3])
                )

            # 使用自适应写作器
            print(f"\n📝 步骤 2: 使用自适应写作生成论文")
            writer = ExpertSectionWriter(
                template=target_venue
                or PROFESSIONAL_TEMPLATE_BY_PAPER_TYPE.get(paper_type, paper_type),
                writing_profile=writing_profile,
            )

            outline = writer.generate_detailed_outline(idea, experiment_results)

            full_paper = writer.write_full_paper(idea, experiment_results)
            latex_path = _save_generated_manuscript(
                paper_structure=paper_structure,
                paper_type=paper_type,
                outline=outline,
                full_paper=full_paper,
            )

            rewrite_model = getattr(self.adaptive_writer, "model", "claude-3-5-sonnet")
            quality_pass = _run_optional_quality_pass(
                enabled=high_quality_mode,
                run_dir=paper_structure["root"],
                paper_type=paper_type,
                rewrite_model=rewrite_model,
                quality_model=rewrite_model,
                target_venue=target_venue,
                quality_preset=quality_preset,
                logger=print,
                strict_fallbacks=self.strict_fallbacks,
                workflow_mode="adaptive_writing",
            )
            quality_result = (
                quality_pass["quality_result"] if quality_pass is not None else None
            )

            # 质量评估
            outcome = "unknown"
            final_scores = {}
            reviews_collected = []
            improvements_collected = []

            if enable_evaluation:
                print(f"\n📊 步骤 3: 评估论文质量")

                evaluator = ProfessionalPaperEvaluator(
                    template=target_venue
                    or PROFESSIONAL_TEMPLATE_BY_PAPER_TYPE.get(paper_type, paper_type)
                )
                evaluation = evaluator.evaluate_paper_quality(full_paper, idea)

                # 保存评估结果
                _write_json_artifact(
                    paper_structure["reviews"] / "quality_evaluation.json",
                    evaluation,
                    ensure_ascii=False,
                )

                final_scores = {
                    dim: result.get("score", 0)
                    for dim, result in evaluation.items()
                    if isinstance(result, dict) and "score" in result
                }

                print(f"\n📊 质量评估结果:")
                print(f"   总分: {evaluation['overall']['score']:.1f}/5")
                print(f"   等级: {evaluation['overall']['level']}")

                # 确定结果
                if evaluation["overall"]["score"] >= 4.0:
                    outcome = "accepted"
                elif evaluation["overall"]["score"] >= 3.5:
                    outcome = "minor_revision"
                elif evaluation["overall"]["score"] >= 3.0:
                    outcome = "major_revision"
                else:
                    outcome = "rejected"

                print(f"   预测结果: {outcome}")

                # 模拟收集审查（实际应用中会有真实审查）
                reviews_collected.append(
                    {
                        "review": evaluation,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # 从结果中学习
            if learn_from_result:
                print(f"\n📚 步骤 4: 从本次生成中学习")

                learning_insights = self.learning_engine.learn_from_generation(
                    idea=idea,
                    paper_data={
                        "paper_id": f"{paper_type}_{idea_name}_{timestamp}",
                        "idea_name": idea_name,
                        "paper_type": paper_type,
                    },
                    outcome=outcome,
                    reviews=reviews_collected,
                    improvements=improvements_collected,
                    final_scores=final_scores,
                )

                # 保存学习洞察
                _write_json_artifact(
                    paper_structure["root"] / "learning_insights.json",
                    learning_insights,
                    ensure_ascii=False,
                )

                print(f"✅ 学习完成并已更新知识库")

                # 显示学习摘要
                summary = self.knowledge_base.generate_learning_summary()
                print(f"\n📊 知识库摘要:")
                print(f"   总论文数: {summary['total_papers']}")
                print(f"   成功率: {summary['success_rate']:.1%}")
                print(f"   成功模式: {summary['success_count']}")
                print(f"   失败模式: {summary['failure_count']}")

            return {
                "status": "success",
                "paper_dir": str(paper_dir),
                "latex_path": str(latex_path),
                "recommendation": recommendation,
                "evaluation": final_scores if enable_evaluation else None,
                "outcome": outcome,
                "quality_result": quality_result,
                "idea_name": idea_name,
                "paper_type": paper_type,
                "learning_enabled": True,
                **_try_minimal_ara(
                    paper_dir=paper_dir,
                    idea=idea,
                    writing_profile=writing_profile,
                    producer="continuous_paper_generator.adaptive_learning",
                ),
            }

        except StrictFallbackViolation as exc:
            return {
                "status": "failed",
                "stage": "quality_fallback_blocked",
                "reason": str(exc),
                "idea_name": idea_name,
                "paper_type": paper_type,
                "workflow_mode": "adaptive_writing",
            }
        except Exception as e:
            print(f"❌ 自适应学习写作失败: {e}")
            import traceback

            traceback.print_exc()

            # 即使失败也记录经验
            if learn_from_result:
                self.learning_engine.learn_from_generation(
                    idea=idea,
                    paper_data={
                        "paper_id": f"{paper_type}_{idea_name}_{timestamp}",
                        "idea_name": idea_name,
                        "paper_type": paper_type,
                    },
                    outcome="failed",
                    final_scores={},
                )

            return {
                "status": "failed",
                "error": str(e),
                "idea_name": idea_name,
                "paper_type": paper_type,
            }

    def show_learning_status(self):
        """显示学习系统状态"""
        if not self.enable_learning:
            print("自适应学习未启用")
            return

        print("\n" + "=" * 80)
        print("🧠 自适应学习系统状态")
        print("=" * 80)

        summary = self.knowledge_base.generate_learning_summary()

        print(f"\n📊 知识库统计:")
        print(f"   总论文数: {summary['total_papers']}")
        print(f"   成功论文: {summary['success_count']}")
        print(f"   失败论文: {summary['failure_count']}")
        print(f"   整体成功率: {summary['success_rate']:.1%}")

        if summary.get("common_issues"):
            print(f"\n⚠️  常见问题 (Top 5):")
            for issue, count in list(summary["common_issues"].items())[:5]:
                print(f"   {issue}: {count} 次")

        if summary.get("effective_strategies"):
            print(f"\n✅ 有效策略 (Top 5):")
            for strategy in summary["effective_strategies"][:5]:
                print(
                    f"   {strategy['success_rate']:.1%} - {strategy['strategy']} "
                    f"({strategy['total_count']} 次)"
                )

        playbook = summary.get("self_evolution_playbook") or {}
        if playbook.get("top_agentic_defaults"):
            print(f"\n🧠 Self-Evolution Playbook:")
            for item in playbook["top_agentic_defaults"][:3]:
                print(
                    f"   {item.get('stage')}: {item.get('action')} "
                    f"({item.get('count')} 次)"
                )

        if summary.get("score_thresholds"):
            print(f"\n📈 分数阈值:")
            for dimension, threshold in summary["score_thresholds"].items():
                print(f"   {dimension}: {threshold}")

        print(f"\n📁 知识库位置: {self.knowledge_base.knowledge_dir}")

    # ========================================
    # 元数据管理
    # ========================================

    def create_paper_metadata_markers(
        self,
        paper_dir: str,
        idea: Dict,
        paper_type: str,
    ) -> PaperMetadata:
        """
        为论文创建标准化元数据标记

        Args:
            paper_dir: 论文目录
            idea: 研究想法
            paper_type: 论文类型

        Returns:
            PaperMetadata实例
        """
        print(f"\n📋 创建论文元数据标记...")

        # 创建元数据
        metadata = create_paper_metadata(paper_dir, idea, paper_type)
        metadata.metadata["batch_name"] = provenance.get("batch_name")
        metadata.metadata["batch_dir"] = provenance.get("batch_dir")
        metadata.metadata["daemon_name"] = provenance.get("daemon_name")
        metadata.metadata["source_name"] = provenance.get("source_name")
        metadata.metadata["source_key"] = provenance.get("source_key")
        metadata.metadata["source_type"] = provenance.get("source_type")
        metadata.metadata["source_value"] = provenance.get("source_value")
        metadata.metadata["source_target_venue"] = provenance.get("source_target_venue")
        metadata.metadata["source_paper_types"] = (
            provenance.get("source_paper_types") or []
        )
        metadata.metadata["source_workflow_mode"] = provenance.get(
            "source_workflow_mode"
        )
        metadata.metadata["source_archetype"] = provenance.get("source_archetype")
        metadata.metadata["source_batch_profile"] = provenance.get(
            "source_batch_profile"
        )
        metadata.save()

        # 设置状态
        metadata.set_status(PaperStatus.GENERATING, "论文生成中")

        # 创建标准化标记文件
        create_standardized_markers(paper_dir)

        # 注册到注册表
        if self.enable_learning:
            registry = MetadataRegistry(str(self.research_dir))
            registry.register_paper(paper_dir)

        print(f"✅ 元数据标记已创建")
        print(f"   论文ID: {metadata.metadata['paper_id']}")
        print(f"   标记文件: {paper_dir}/README.md")

        return metadata

    def update_paper_metadata_from_evaluation(
        self,
        paper_dir: str,
        evaluation: Dict,
    ):
        """
        从评估结果更新元数据

        Args:
            paper_dir: 论文目录
            evaluation: 评估结果
        """
        metadata = PaperMetadata(paper_dir)
        metadata.update_from_evaluation(evaluation)

        # 更新注册表
        if self.enable_learning:
            registry = MetadataRegistry(str(self.research_dir))
            registry.register_paper(paper_dir)

        print(f"✅ 元数据已更新")

    def get_papers_for_agent_guidance(
        self,
        agent_name: str = None,
        agent_capabilities: List[str] = None,
    ) -> List[Dict]:
        """
        获取需要Agent指导的论文列表

        Args:
            agent_name: Agent名称
            agent_capabilities: Agent能力

        Returns:
            论文列表
        """
        if not self.enable_learning:
            return []

        api = AgentGuidanceAPI(str(self.research_dir))

        if agent_name and agent_capabilities:
            return api.discover_papers(
                agent_name=agent_name,
                agent_capabilities=agent_capabilities,
                max_papers=10,
            )
        else:
            # 返回所有需要改进的论文
            registry = MetadataRegistry(str(self.research_dir))
            papers = registry.get_papers_needing_improvement()

            # 添加详细信息
            result = []
            for paper in papers:
                metadata = registry.get_paper_metadata(paper["paper_id"])
                if metadata:
                    result.append(
                        {
                            **paper,
                            "info": metadata.get_info_for_agents(),
                            "actionable_items": metadata.get_actionable_items(),
                        }
                    )

            return result

    def submit_agent_guidance(
        self,
        paper_id: str,
        agent_name: str,
        comment: str,
        score: float = None,
        issues: List[str] = None,
        suggestions: List[str] = None,
        priority: str = "medium",
    ) -> Dict:
        """
        提交Agent指导

        Args:
            paper_id: 论文ID
            agent_name: Agent名称
            comment: 评论
            score: 评分
            issues: 问题列表
            suggestions: 建议列表
            priority: 优先级

        Returns:
            提交结果
        """
        if not self.enable_learning:
            return {"error": "Learning system not enabled"}

        api = AgentGuidanceAPI(str(self.research_dir))

        result = api.submit_guidance(
            agent_name=agent_name,
            paper_id=paper_id,
            comment=comment,
            score=score,
            issues=issues,
            suggestions=suggestions,
            priority=priority,
        )

        print(f"✅ Agent指导已提交: {agent_name} → {paper_id}")

        return result

    def get_guidance_report(self) -> Dict:
        """获取指导报告"""
        if not self.enable_learning:
            return {"error": "Learning system not enabled"}

        coordinator = AgentGuidanceCoordinator(str(self.research_dir))
        return coordinator.generate_guidance_report()

    def register_external_agent(self, agent: BaseAgent, group: str = "default") -> bool:
        """
        注册外部Agent

        Args:
            agent: Agent实例（继承自BaseAgent）
            group: 分组名称

        Returns:
            是否成功注册
        """
        if not self.enable_learning:
            print("⚠️  自适应学习未启用，无法注册Agent")
            return False

        # 注册到编排器
        success = self.agent_orchestrator.register_agent(agent, group)

        if success:
            # 注册到进化引擎
            register_agent_with_evolution(self.evolution_engine, agent, group)

        return success

    async def generate_paper_with_evolution(
        self,
        idea: Dict,
        paper_type: str = "neurips",
        experiment_results: Dict = None,
        enable_external_agents: bool = True,
        agent_filter: List[str] = None,
        evolution_rounds: int = 1,
    ) -> Dict:
        """
        使用自主进化系统生成论文

        Args:
            idea: 研究想法
            paper_type: 论文类型
            experiment_results: 实验结果
            enable_external_agents: 是否启用外部agent
            agent_filter: 要咨询的agent列表
            evolution_rounds: 进化轮数

        Returns:
            生成结果
        """
        if not self.enable_learning:
            print("⚠️  自适应学习未启用，使用标准写作系统")
            return await self._generate_with_fallback(
                idea, paper_type, experiment_results
            )

        print("\n" + "=" * 80)
        print(f"🧬 使用自主进化系统生成 {paper_type.upper()} 论文")
        print("=" * 80)

        workspace = _create_paper_workspace(
            idea=idea,
            paper_type=paper_type,
            output_root=self.research_dir,
        )
        idea_name = workspace["idea_name"]
        timestamp = workspace["timestamp"]
        paper_dir = workspace["paper_dir"]
        paper_structure = workspace["paper_structure"]

        try:
            # 初始生成
            print(f"\n📝 步骤 1: 初始论文生成")
            writer = ExpertSectionWriter(template=paper_type)

            outline = writer.generate_detailed_outline(idea, experiment_results)
            full_paper = writer.write_full_paper(idea, experiment_results)
            latex_path = _save_generated_manuscript(
                paper_structure=paper_structure,
                paper_type=paper_type,
                outline=outline,
                full_paper=full_paper,
            )

            print(f"✅ 初始论文已生成")

            # 进化循环
            current_state = {
                "paper_content": full_paper,
                "outline": outline,
                "generation_method": "professional_writing",
            }

            paper_data = {
                "paper_id": f"{paper_type}_{idea_name}_{timestamp}",
                "idea": idea,
                "idea_name": idea_name,
                "paper_type": paper_type,
                "Title": idea.get("Title", ""),
                "Abstract": idea.get("Abstract", ""),
                "Method": idea.get("Method", ""),
            }

            for round_num in range(evolution_rounds):
                print(f"\n{'='*80}")
                print(f"🧪 进化轮次 {round_num + 1}/{evolution_rounds}")
                print(f"{'='*80}")

                # 收集外部agent反馈
                external_feedback = []
                if enable_external_agents:
                    print(f"\n🤖 步骤 2.{round_num + 1}.1: 咨询外部Agent")
                    external_feedback = await self.agent_orchestrator.consult_agents(
                        paper_data=paper_data,
                        current_state=current_state,
                        agent_names=agent_filter,
                    )

                    if external_feedback:
                        print(f"   收到 {len(external_feedback)} 条Agent反馈")
                        for fb in external_feedback:
                            score = fb["feedback"].get("score", 0)
                            print(f"   - {fb['agent_name']}: {score:.1f}/5")
                    else:
                        print(f"   没有收到外部Agent反馈")

                # 执行自主进化
                print(f"\n🧬 步骤 2.{round_num + 1}.2: 执行自主进化")
                evolution_result = await self.evolution_engine.evolve(
                    paper_data=paper_data,
                    current_state=current_state,
                    external_feedback=external_feedback,
                )

                # 更新状态
                if evolution_result.get("validation_score", 0) >= 4.0:
                    print(
                        f"✅ 进化成功，分数: {evolution_result['validation_score']:.1f}/5"
                    )
                    break
                else:
                    print(f"⚠️  进化分数: {evolution_result['validation_score']:.1f}/5")

                # 更新当前状态
                current_state["evolution_history"] = evolution_result.get(
                    "evolution_record", []
                )

            # 最终评估
            print(f"\n📊 步骤 3: 最终质量评估")
            evaluator = ProfessionalPaperEvaluator(template=paper_type)
            final_evaluation = evaluator.evaluate_paper_quality(full_paper, idea)

            _write_json_artifact(
                paper_structure["reviews"] / "final_evaluation.json",
                final_evaluation,
                ensure_ascii=False,
            )

            # 保存进化报告
            evolution_report = self.evolution_engine.get_evolution_report()
            _write_json_artifact(
                paper_structure["root"] / "evolution_report.json",
                evolution_report,
                ensure_ascii=False,
            )

            print(f"\n📊 最终评估:")
            print(f"   总分: {final_evaluation['overall']['score']:.1f}/5")
            print(f"   等级: {final_evaluation['overall']['level']}")

            return {
                "status": "success",
                "paper_dir": str(paper_dir),
                "latex_path": str(latex_path),
                "evaluation": final_evaluation,
                "evolution_report": evolution_report,
                "idea_name": idea_name,
                "paper_type": paper_type,
                "evolution_enabled": True,
            }

        except Exception as e:
            print(f"❌ 自主进化生成失败: {e}")
            import traceback

            traceback.print_exc()

            return {
                "status": "failed",
                "error": str(e),
                "idea_name": idea_name,
                "paper_type": paper_type,
            }

    async def _generate_with_fallback(
        self,
        idea: Dict,
        paper_type: str,
        experiment_results: Dict,
    ) -> Dict:
        """降级到标准生成"""
        return self.generate_paper_with_professional_writing(
            idea, paper_type, experiment_results, enable_evaluation=True
        )

    def submit_feedback(
        self,
        source: str,
        feedback: Dict,
        metadata: Dict = None,
    ):
        """
        提交反馈到进化系统

        Args:
            source: 反馈来源 (self, external_agent, human, peer_review, metrics)
            feedback: 反馈内容
            metadata: 元数据
        """
        if not self.enable_learning:
            print("⚠️  自适应学习未启用")
            return

        self.evolution_engine.submit_feedback(
            FeedbackSource(source),
            feedback,
            metadata,
        )

        print(f"✅ 反馈已提交 (来源: {source})")

    def get_evolution_status(self) -> Dict:
        """获取进化系统状态"""
        if not self.enable_learning:
            return {"enabled": False}

        evolution_report = self.evolution_engine.get_evolution_report()
        agent_stats = self.agent_orchestrator.get_agent_statistics()

        return {
            "enabled": True,
            "evolution": evolution_report,
            "agents": agent_stats,
            "knowledge_base": self.knowledge_base.generate_learning_summary(),
        }


def _current_generation_provenance(batch_dir: str | Path) -> dict[str, Any]:
    batch_path = Path(batch_dir)
    return {
        "batch_name": batch_path.name,
        "batch_dir": str(batch_path),
        "daemon_name": os.environ.get("AI_SCIENTIST_DAEMON_NAME"),
        "source_name": os.environ.get("AI_SCIENTIST_SOURCE_NAME"),
        "source_key": os.environ.get("AI_SCIENTIST_SOURCE_KEY"),
        "source_type": os.environ.get("AI_SCIENTIST_SOURCE_TYPE"),
        "source_value": os.environ.get("AI_SCIENTIST_SOURCE_VALUE"),
        "source_target_venue": os.environ.get("AI_SCIENTIST_SOURCE_TARGET_VENUE"),
        "source_paper_types": [
            item
            for item in (os.environ.get("AI_SCIENTIST_SOURCE_PAPER_TYPES") or "").split(
                ","
            )
            if item
        ],
        "source_workflow_mode": os.environ.get("AI_SCIENTIST_SOURCE_WORKFLOW_MODE"),
        "source_archetype": os.environ.get("AI_SCIENTIST_SOURCE_ARCHETYPE"),
        "source_batch_profile": os.environ.get("AI_SCIENTIST_SOURCE_BATCH_PROFILE"),
    }


def _attach_source_provenance_to_result(
    result: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    enriched = dict(result)
    enriched.setdefault("batch_name", provenance.get("batch_name"))
    enriched.setdefault("batch_dir", provenance.get("batch_dir"))
    enriched.setdefault("daemon_name", provenance.get("daemon_name"))
    enriched.setdefault("source_name", provenance.get("source_name"))
    enriched.setdefault("source_key", provenance.get("source_key"))
    enriched.setdefault("source_type", provenance.get("source_type"))
    enriched.setdefault("source_value", provenance.get("source_value"))
    enriched.setdefault("source_target_venue", provenance.get("source_target_venue"))
    enriched.setdefault(
        "source_paper_types", provenance.get("source_paper_types") or []
    )
    enriched.setdefault("source_workflow_mode", provenance.get("source_workflow_mode"))
    enriched.setdefault("source_archetype", provenance.get("source_archetype"))
    enriched.setdefault("source_batch_profile", provenance.get("source_batch_profile"))
    return enriched


def _process_single_paper(args):
    """
    处理单篇论文的完整流程（可在子进程中运行）
    每篇论文都有独立的文件夹

    Args:
        args: 参数元组

    Returns:
        结果字典
    """
    args = list(args)
    requested_workflow_mode = None
    if len(args) == 39:
        requested_workflow_mode = args.pop()
    elif len(args) == 38 and isinstance(args[-1], str) and isinstance(args[-2], bool):
        requested_workflow_mode = args.pop()
        args.append(False)
    elif len(args) == 37:
        args.append(False)
    (
        batch_dir,
        research_dir,
        idea_idx,
        idea,
        paper_type,
        bfts_config_path,
        model_writeup,
        model_citation,
        model_review,
        model_agg_plots,
        model_writeup_small,
        num_cite_rounds,
        writeup_retries,
        improvement_rounds,
        high_quality_mode,
        quality_preset,
        quality_model,
        target_venue,
        quality_threshold,
        rigor_threshold,
        quality_rewrite_rounds,
        autonomous_quality_followup_rounds,
        require_quality_gate,
        min_submission_priority,
        max_submission_blockers,
        review_reflections,
        review_ensemble,
        review_fewshot,
        review_temperature,
        review_strategy,
        writing_profile,
        writing_audit_rounds,
        strict_writing_guardrails,
        guardrail_repair_rounds,
        workflow_mode,
        submission_mode,
        strict_fallbacks,
        integrity_forensics_enabled,
    ) = args
    requested_workflow_mode = requested_workflow_mode or workflow_mode
    try:
        writing_profile = normalize_writing_profile(writing_profile)
    except ValueError as exc:
        print(
            f"[想法 #{idea_idx}] ⚠️  Invalid writing profile '{writing_profile}': {exc}; "
            f"falling back to {DEFAULT_WRITING_PROFILE}"
        )
        writing_profile = DEFAULT_WRITING_PROFILE
    writing_audit_rounds = max(0, int(writing_audit_rounds))
    strict_writing_guardrails = bool(strict_writing_guardrails or high_quality_mode)
    guardrail_repair_rounds = max(0, int(guardrail_repair_rounds))
    integrity_forensics_result: Dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
    }

    try:
        current_stage = "prepare"
        idea_name = idea.get("Name", f"idea_{idea_idx}")

        print(f"\n{'='*80}")
        print(f"🚀 开始处理想法 #{idea_idx}: {idea_name} ({paper_type})")
        print(f"{'='*80}")
        print(f"🧭 写作 profile: {writing_profile}")
        print(f"🧪 写作审计轮数: {writing_audit_rounds}")
        print(f"🛡️ 严格写作守护: {strict_writing_guardrails}")
        print(f"🔧 守护自动修复轮数: {guardrail_repair_rounds}")

        workspace = _create_paper_workspace(
            idea=idea,
            paper_type=paper_type,
            default_idea_name=f"idea_{idea_idx}",
            output_root=research_dir,
        )
        paper_dir = workspace["paper_dir"]
        paper_structure = workspace["paper_structure"]
        (
            idea_card,
            research_plan,
            claim_evidence_graph,
            template_profile,
            template_capability,
        ) = _save_paper_pipeline_seed_artifacts(
            paper_root=paper_structure["root"],
            idea=idea,
            idea_idx=idea_idx,
            target_venue=target_venue,
            workflow_mode=workflow_mode,
            submission_mode=bool(submission_mode),
            high_quality_mode=bool(high_quality_mode),
        )
        record_decision(
            paper_structure["root"],
            category="workflow_strategy",
            selected=str(research_plan.get("workflow_mode") or workflow_mode),
            options_considered=build_workflow_strategy_decision_options(
                selected=str(research_plan.get("workflow_mode") or workflow_mode),
                requested_workflow_mode=requested_workflow_mode,
                submission_mode=bool(submission_mode),
                high_quality_mode=bool(high_quality_mode),
                target_venue=target_venue,
            ),
            producer="continuous_paper_generator.workflow_strategy",
            metadata={
                "requested_workflow_mode": requested_workflow_mode,
                "resolved_workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
                "target_venue": target_venue,
                "submission_mode": bool(submission_mode),
                "high_quality_mode": bool(high_quality_mode),
            },
        )
        workflow_runtime_plan = build_workflow_runtime_plan(
            workflow_mode,
            submission_mode=bool(submission_mode),
            high_quality_mode=bool(high_quality_mode),
            target_venue=target_venue,
        )
        assert_gate_preconditions_satisfiable(
            GatePreconditionContext(
                research_plan=research_plan,
                improvement_rounds=improvement_rounds,
                require_quality_gate=require_quality_gate,
                high_quality_mode=high_quality_mode,
                target_venue=target_venue,
                require_sample_gate=False,
            )
        )

        print(f"📁 论文目录: {paper_dir}")
        print(
            f"🧪 审稿编排: 改进轮={list(workflow_runtime_plan.improvement_review_roles)} "
            f"终审={list(workflow_runtime_plan.final_review_roles)}"
        )

        idea_path_json = workspace["idea_path"]
        idea_path_md = paper_structure["root"] / "idea.md"

        provenance = _current_generation_provenance(batch_dir)
        (paper_structure["root"] / "source_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # ========== 步骤1: 运行实验 ==========
        current_stage = "experiment"
        print(f"\n📊 [想法 #{idea_idx}] 步骤 1/4: 运行实验")
        idea_to_markdown(idea, str(idea_path_md), None)

        # 使用论文目录作为实验目录
        exp_dir = str(paper_structure["root"])

        config_path = str(bfts_config_path or "bfts_config.yaml")
        idea_config_path = edit_bfts_config_file(
            config_path, exp_dir, str(idea_path_json)
        )

        experiment_result = perform_experiments_bfts(idea_config_path)
        if (
            isinstance(experiment_result, dict)
            and experiment_result.get("status") != "completed"
        ):
            return {
                "idea_idx": idea_idx,
                "status": experiment_result.get("status", "failed"),
                "stage": "experiment",
                "error": (
                    (experiment_result.get("failure_error") or {}).get("message")
                    or (experiment_result.get("budget_error") or {}).get("message")
                ),
                "resumable": bool(experiment_result.get("resumable")),
                "checkpoint_path": experiment_result.get("checkpoint_path"),
                "run_status_path": experiment_result.get("run_status_path"),
                "manager_state_path": experiment_result.get("manager_state_path"),
                "initialization_status_path": experiment_result.get(
                    "initialization_status_path"
                ),
                "initialization_phase": experiment_result.get("initialization_phase"),
                "lock_path": experiment_result.get("lock_path"),
                "lock_owner": experiment_result.get("lock_owner"),
                "paper_type": paper_type,
                "workflow_mode": workflow_mode,
            }

        try:
            write_experiment_report(exp_dir)
        except Exception as exc:
            print(f"[想法 #{idea_idx}] ⚠️  实验报告生成失败: {exc}")
        registry_rows = _build_experiment_registry_rows(
            paper_root=paper_structure["root"],
            research_plan=research_plan,
        )
        save_experiment_registry(str(paper_structure["root"]), registry_rows)
        sample_gate = evaluate_and_save_sample_gate(
            paper_structure["root"],
            producer="continuous_paper_generator.experiment_registry",
        )
        sample_gate_decision = (
            "continue_full_generation"
            if sample_gate.get("full_generation_allowed")
            else "block_full_generation"
        )
        record_decision(
            paper_structure["root"],
            category="sample_gate_full_generation",
            selected=sample_gate_decision,
            options_considered=build_sample_gate_decision_options(
                bool(sample_gate.get("full_generation_allowed"))
            ),
            producer="continuous_paper_generator.sample_gate",
            metadata={
                "sample_gate_reasons": sample_gate.get("result", {}).get("reasons", []),
                "sample_task_count": sample_gate.get("sample_task_count"),
            },
        )
        if not sample_gate.get("full_generation_allowed"):
            print(
                f"[想法 #{idea_idx}] ⚠️  Sample gate 未通过: "
                f"{sample_gate.get('result', {}).get('reasons', [])}"
            )
        assert_gate_preconditions_satisfiable(
            GatePreconditionContext(
                research_plan=research_plan,
                sample_gate=sample_gate,
                improvement_rounds=improvement_rounds,
                require_quality_gate=require_quality_gate,
                high_quality_mode=high_quality_mode,
                target_venue=target_venue,
            )
        )

        # 复制实验结果到experiment子目录
        latest_run_dir = find_latest_bfts_run_dir(exp_dir, logs_subdir="logs")
        experiment_results_dir = (
            osp.join(str(latest_run_dir), "experiment_results")
            if latest_run_dir is not None
            else osp.join(exp_dir, "logs/0-run/experiment_results")
        )
        if osp.exists(experiment_results_dir):
            shutil.copytree(
                experiment_results_dir,
                paper_structure["experiment"],
                dirs_exist_ok=True,
            )

        # ========== 步骤2: 生成论文 ==========
        current_stage = "writeup"
        print(f"\n📝 [想法 #{idea_idx}] 步骤 2/4: 生成论文")
        aggregate_plots(base_folder=exp_dir, model=model_agg_plots)
        _save_manuscript_contract_state(
            paper_root=paper_structure["root"],
            paper_type=paper_type,
            target_venue=target_venue,
            writing_profile=writing_profile,
        )

        # 清理实验结果
        experiment_results = osp.join(exp_dir, "experiment_results")
        if osp.exists(experiment_results):
            shutil.rmtree(experiment_results)

        writeup_plan = build_writeup_execution_plan(
            paper_type,
            num_cite_rounds=num_cite_rounds,
            writeup_retries=writeup_retries,
            target_venue=target_venue,
            high_quality_mode=high_quality_mode,
            research_root=research_dir,
        )
        selected_venue = writeup_plan["target_venue"]
        page_limit = writeup_plan["page_limit"]
        writeup_func = (
            perform_writeup
            if writeup_plan["writeup_engine"] == "normal"
            else perform_icbinb_writeup
        )
        strategy_feedback = writeup_plan["strategy_feedback"]
        _save_manuscript_contract_state(
            paper_root=paper_structure["root"],
            paper_type=paper_type,
            target_venue=selected_venue,
            writing_profile=writing_profile,
        )

        citations_text = gather_citations(
            exp_dir,
            num_cite_rounds=writeup_plan["num_cite_rounds"],
            small_model=model_citation,
        )

        writeup_success = False
        effective_writeup_retries = writeup_plan["writeup_retries"]
        if high_quality_mode:
            print(
                f"[想法 #{idea_idx}] 历史预算反馈: {strategy_feedback.get('rationale', [])}"
            )
        for attempt in range(effective_writeup_retries):
            print(
                f"[想法 #{idea_idx}] 论文写作尝试 {attempt + 1}/{effective_writeup_retries}"
            )

            writeup_success = writeup_func(
                base_folder=exp_dir,
                small_model=model_writeup_small,
                big_model=model_writeup,
                page_limit=page_limit,
                citations_text=citations_text,
                writing_profile=writing_profile,
                writing_audit_rounds=writing_audit_rounds,
                target_venue=selected_venue,
                strict_guardrails=strict_writing_guardrails,
                guardrail_repair_rounds=guardrail_repair_rounds,
            )

            if writeup_success:
                print(f"[想法 #{idea_idx}] ✅ 论文写作成功!")
                break
            else:
                print(f"[想法 #{idea_idx}] ⚠️  尝试 {attempt + 1} 失败")

        if not writeup_success:
            print(f"[想法 #{idea_idx}] ❌ 论文写作失败")
            guardrail_findings, guardrail_reasons = load_guardrail_artifacts(exp_dir)
            guardrail_blocking = bool(guardrail_reasons)
            return {
                "idea_idx": idea_idx,
                "status": "failed",
                "stage": "writeup",
                "paper_type": paper_type,
                "writing_profile": writing_profile,
                "writing_audit_rounds": writing_audit_rounds,
                "strict_writing_guardrails": strict_writing_guardrails,
                "guardrail_repair_rounds": guardrail_repair_rounds,
                "guardrail_blocking": guardrail_blocking,
                "guardrail_blocking_reasons": guardrail_reasons,
                "guardrail_findings": guardrail_findings,
                "workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
            }

        quality_pass = _run_optional_quality_pass(
            enabled=high_quality_mode,
            run_dir=exp_dir,
            paper_type=paper_type,
            rewrite_model=model_writeup,
            quality_model=quality_model or model_review,
            target_venue=selected_venue,
            quality_preset=quality_preset,
            quality_threshold=quality_threshold,
            rigor_threshold=rigor_threshold,
            max_quality_rewrites=quality_rewrite_rounds,
            autonomous_quality_followup_rounds=autonomous_quality_followup_rounds,
            require_quality_gate=require_quality_gate,
            min_submission_priority=min_submission_priority,
            max_submission_blockers=max_submission_blockers,
            allow_auto_improvement_fallback=(
                (research_plan.get("execution_policy") or {}).get(
                    "allow_auto_improvement_fallback"
                )
            ),
            reject_on_auto_improvement_fallback=bool(
                (research_plan.get("execution_policy") or {}).get(
                    "reject_on_auto_improvement_fallback"
                )
            ),
            resume=False,
            logger=lambda msg: print(f"[想法 #{idea_idx}] {msg}"),
            strict_fallbacks=strict_fallbacks,
            workflow_mode=workflow_mode,
        )
        if quality_pass is not None:
            quality_result = quality_pass["quality_result"]
            acceptance = quality_pass["acceptance"]
            if not acceptance.get("accepted"):
                return {
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "stage": "submission_bar",
                    "paper_type": paper_type,
                    "writing_profile": writing_profile,
                    "writing_audit_rounds": writing_audit_rounds,
                    "strict_writing_guardrails": strict_writing_guardrails,
                    "guardrail_repair_rounds": guardrail_repair_rounds,
                    "quality_score": quality_result.get("quality_score_after"),
                    "rigor_score": quality_result.get("rigor_score_after"),
                    "claim_support_score": quality_result.get("claim_support_after"),
                    "submission_priority_score": quality_result.get(
                        "submission_priority_score"
                    ),
                    "submission_priority_tier": quality_result.get(
                        "submission_priority_tier"
                    ),
                    "blocker_count": quality_result.get("blocker_count"),
                    "autonomous_followup_rounds_run": quality_result.get(
                        "autonomous_followup_rounds_run"
                    ),
                    "acceptance_reasons": acceptance.get("reasons", []),
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                }
        else:
            quality_result = {}

        # ========== 步骤3: 智能审查和自动改进 ==========
        current_stage = "review_improvement"
        review_plan = build_review_execution_plan(
            paper_type,
            target_venue=selected_venue,
            review_reflections=review_reflections,
            review_ensemble=review_ensemble,
            review_fewshot=review_fewshot,
            review_temperature=review_temperature,
            review_strategy=review_strategy,
            high_quality_mode=high_quality_mode,
            research_root=research_dir,
        )
        if high_quality_mode:
            print(
                f"[想法 #{idea_idx}] 历史审稿反馈: "
                f"{review_plan['strategy_feedback'].get('rationale', [])}"
            )
        if improvement_rounds > 0:
            print(f"\n🔍 [想法 #{idea_idx}] 步骤 3/4: 智能审查和自动改进")
            initial_review_dir = paper_structure["reviews"] / "initial"

            initial_review_pass = execute_review_suite(
                review_roles=workflow_runtime_plan.improvement_review_roles,
                paper_dir=exp_dir,
                model_review=model_review,
                review_plan=review_plan,
                create_client_fn=create_client,
                load_paper_fn=load_paper,
                perform_review_fn=perform_review,
                perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
                pdf_path_resolver=find_latest_pdf_path,
                save_dir=initial_review_dir,
                project_root=paper_structure["root"],
                persist_job=True,
                evidence_refs=[
                    "claim_evidence_graph.json",
                    "experiment_registry.jsonl",
                    "figure_spec.json",
                    "manuscript_state.json",
                ],
                suite_name="initial_review",
                lane_name="review_board",
                strictness_profile="standard",
            )
            if not initial_review_pass["found"]:
                print(f"[想法 #{idea_idx}] ⚠️  未找到PDF，跳过改进")
            else:
                pdf_path = initial_review_pass["pdf_path"]
                review_text = initial_review_pass["review_text"]
                review_img = initial_review_pass["review_img"]
                initial_review_text = review_text

                initial_pdf = paper_structure["root"] / "paper_initial.pdf"
                shutil.copy(pdf_path, initial_pdf)

                from ai_scientist.perform_auto_improvement import AutoImprovementEngine

                iteration_controller = SmartIterationController(
                    min_rounds=1,
                    max_rounds=improvement_rounds,
                    improvement_threshold=0.5,
                    convergence_rounds=2,
                )
                improvement_engine = AutoImprovementEngine(model=model_review)

                for iter_round in range(improvement_rounds):
                    print(f"\n{'='*60}")
                    print(
                        f"[想法 #{idea_idx}] 改进轮次 {iter_round + 1}/{improvement_rounds}"
                    )
                    print(f"{'='*60}")

                    improvement_result = improve_paper_with_review(
                        paper_dir=exp_dir,
                        text_review=review_text,
                        img_review=review_img,
                        model=model_writeup,
                        max_rounds=1,
                        target_venue=selected_venue,
                    )
                    if improvement_result["status"] != "success":
                        print(
                            f"[想法 #{idea_idx}] ⚠️  改进失败: {improvement_result.get('reason', 'Unknown')}"
                        )
                        break

                    print(f"[想法 #{idea_idx}] 重新审查改进后的论文...")
                    round_review_dir = (
                        paper_structure["reviews"] / f"round_{iter_round + 1}"
                    )
                    round_review_pass = execute_review_suite(
                        review_roles=workflow_runtime_plan.improvement_review_roles,
                        paper_dir=exp_dir,
                        model_review=model_review,
                        review_plan=review_plan,
                        create_client_fn=create_client,
                        load_paper_fn=load_paper,
                        perform_review_fn=perform_review,
                        perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
                        pdf_path_resolver=find_latest_pdf_path,
                        save_dir=round_review_dir,
                        project_root=paper_structure["root"],
                        persist_job=True,
                        evidence_refs=[
                            "claim_evidence_graph.json",
                            "experiment_registry.jsonl",
                            "figure_spec.json",
                            "manuscript_state.json",
                        ],
                        suite_name=f"round_{iter_round + 1}",
                        lane_name="review_board",
                        strictness_profile="standard",
                    )
                    if not round_review_pass["found"]:
                        print(f"[想法 #{idea_idx}] ⚠️  PDF未找到，停止改进")
                        break

                    new_review_text = round_review_pass["review_text"]
                    new_review_img = round_review_pass["review_img"]
                    improvement_eval = improvement_engine.evaluate_improvement(
                        review_text, new_review_text
                    )
                    current_improvement = improvement_eval["overall_improvement"]
                    review_scores = new_review_text.get("review", {}).get("scores", {})
                    should_continue, reason = iteration_controller.should_continue(
                        current_improvement, review_scores
                    )

                    print(f"[想法 #{idea_idx}] {reason}")
                    print(f"[想法 #{idea_idx}] 总体改进: {current_improvement:+.1f}")
                    _write_json_artifact(
                        round_review_dir / "improvement_eval.json",
                        improvement_eval,
                    )

                    review_text = new_review_text
                    review_img = new_review_img
                    if not should_continue:
                        print(f"[想法 #{idea_idx}] ✅ 改进完成，达到停止条件")
                        break

                save_review_artifacts(
                    paper_structure["reviews"],
                    text_review=review_text,
                    image_review=review_img,
                    text_filename="final_review.json",
                    image_filename="final_review_img.json",
                )

                try:
                    improvement_record_file = osp.join(
                        exp_dir, "improvement_record.json"
                    )
                    if osp.exists(improvement_record_file):
                        with open(improvement_record_file, "r", encoding="utf-8") as f:
                            improvement_record = json.load(f)
                        reporter = ImprovementReporter(exp_dir)
                        reporter.generate_improvement_report(
                            paper_name=idea_name,
                            improvement_record=improvement_record,
                            original_review=initial_review_text,
                            final_review=review_text,
                        )
                        print(f"[想法 #{idea_idx}] 📊 改进报告已生成")
                        print_improvement_summary(improvement_record)
                except Exception as e:
                    print(f"[想法 #{idea_idx}] ⚠️  生成报告失败: {e}")

        else:
            print(f"[想法 #{idea_idx}] 跳过改进步骤")

        # ========== 步骤4: 最终处理 ==========
        current_stage = "finalize"
        print(f"\n🎯 [想法 #{idea_idx}] 步骤 4/4: 最终处理")
        final_review_dir = paper_structure["reviews"] / "final"
        final_review_pass = execute_review_suite(
            review_roles=workflow_runtime_plan.final_review_roles,
            paper_dir=exp_dir,
            model_review=model_review,
            review_plan=review_plan,
            create_client_fn=create_client,
            load_paper_fn=load_paper,
            perform_review_fn=perform_review,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
            pdf_path_resolver=find_latest_pdf_path,
            save_dir=final_review_dir,
            text_filename="final_review.json",
            image_filename="final_review_img.json",
            project_root=paper_structure["root"],
            persist_job=True,
            evidence_refs=[
                "claim_evidence_graph.json",
                "experiment_registry.jsonl",
                "figure_spec.json",
                "manuscript_state.json",
            ],
            suite_name="final_review",
            lane_name="review_board",
            strictness_profile="standard",
        )
        critic_pass = run_independent_critic_pass(
            workflow_runtime_plan=workflow_runtime_plan,
            paper_dir=exp_dir,
            model_review=model_review,
            review_plan=review_plan,
            create_client_fn=create_client,
            load_paper_fn=load_paper,
            perform_review_fn=perform_review,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
            pdf_path_resolver=find_latest_pdf_path,
            save_dir=paper_structure["reviews"] / "hostile_critic",
            project_root=paper_structure["root"],
            evidence_refs=[
                "claim_evidence_graph.json",
                "experiment_registry.jsonl",
                "figure_spec.json",
                "manuscript_state.json",
            ],
        )
        if critic_pass.get("blocking_issue_count", 0) > 0:
            return {
                "idea_idx": idea_idx,
                "status": "failed",
                "stage": "hostile_critic",
                "paper_type": paper_type,
                "workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
                "review_roles_improvement": list(
                    workflow_runtime_plan.improvement_review_roles
                ),
                "review_roles_final": list(workflow_runtime_plan.final_review_roles),
                "review_roles_used": sorted(
                    set(workflow_runtime_plan.improvement_review_roles)
                    | set(workflow_runtime_plan.final_review_roles)
                    | set(workflow_runtime_plan.critic_review_roles)
                ),
                "critic_roles_used": list(workflow_runtime_plan.critic_review_roles),
                "critic_active_issue_count": critic_pass.get("active_issue_count"),
                "critic_blocking_issue_count": critic_pass.get("blocking_issue_count"),
                "critic_findings_file": critic_pass.get("critic_findings_file"),
            }
        if not final_review_pass["found"]:
            print(f"[想法 #{idea_idx}] ⚠️  最终 review 未找到 PDF，将继续保留当前产物")

        # 查找最终PDF并重命名为paper.pdf
        pdf_files = [f for f in os.listdir(exp_dir) if f.endswith(".pdf")]
        if pdf_files:
            pdf_files.sort()
            source_pdf = osp.join(exp_dir, pdf_files[-1])
            final_pdf = paper_structure["root"] / "paper.pdf"

            # 复制并重命名
            shutil.copy(source_pdf, final_pdf)

            print(f"\n✅ [想法 #{idea_idx}] 完成!")
            print(f"   论文目录: {paper_dir}")
            print(f"   最终PDF: {final_pdf}")

            integrity_forensics_result = run_integrity_forensics_for_manuscript(
                root=paper_structure["root"],
                paper_id=f"{idea_idx}_{idea_name}",
                enabled=bool(integrity_forensics_enabled),
            )
            if integrity_forensics_result.get("status") == "completed":
                print(
                    f"[想法 #{idea_idx}] Integrity forensics: "
                    f"verdict={integrity_forensics_result.get('overall_verdict')} "
                    f"findings={integrity_forensics_result.get('finding_count')} "
                    f"dir={integrity_forensics_result.get('output_dir')}"
                )
            elif integrity_forensics_result.get("enabled"):
                print(
                    f"[想法 #{idea_idx}] ⚠️  Integrity forensics "
                    f"{integrity_forensics_result.get('status')}: "
                    f"{integrity_forensics_result.get('reason') or integrity_forensics_result.get('error')}"
                )
            if (
                bool(submission_mode or high_quality_mode)
                and integrity_forensics_result.get("overall_verdict") == "HARD_FLAGS"
            ):
                return {
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "stage": "integrity_forensics",
                    "paper_type": paper_type,
                    "paper_dir": str(paper_dir),
                    "pdf_path": str(final_pdf),
                    "acceptance_reasons": ["integrity forensics reported HARD_FLAGS"],
                    "integrity_forensics": integrity_forensics_result,
                    **integrity_forensics_result_fields(integrity_forensics_result),
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                }

            # ========== ARA finalisation ==========
            # continuous_paper_generator drives BFTS end-to-end just like
            # run_project.py does — so the same finalize path applies. ARA
            # lands under <paper_dir>/ara/ (paper_dir is both the project
            # anchor and the experiment root in this codepath).
            try:
                ara_result = finalize_ara_for_idea(
                    project_dir=str(paper_dir),
                    exp_dir=str(paper_dir),
                    idea=idea,
                    timestamp=None,
                    bfts_config_path=config_path,
                    model_spec={
                        "writeup": model_writeup,
                        "writeup_small": model_writeup_small,
                        "citation": model_citation,
                        "review": model_review,
                        "agg_plots": model_agg_plots,
                        "quality": quality_model,
                    },
                    writing_profile=writing_profile,
                )
                for line in summarise_finalize(idea_idx, ara_result):
                    print(line)
                ara_export = ara_result.export
            except (
                Exception
            ) as ara_exc:  # noqa: BLE001 — never break the paper on ARA fail
                print(f"[想法 #{idea_idx}] ⚠️  ARA finalize 失败: {ara_exc}")
                ara_result = None
                ara_export = None

            return {
                "idea_idx": idea_idx,
                "status": "success",
                "paper_dir": str(paper_dir),
                "pdf_path": str(final_pdf),
                "ara_dir": str(ara_export.root) if ara_export is not None else None,
                "ara_manifest": (
                    str(ara_export.manifest_path) if ara_export is not None else None
                ),
                "ara_node_count": (
                    ara_export.node_count if ara_export is not None else None
                ),
                "ara_claim_coverage_score": (
                    ara_result.claim_coverage.coverage_score
                    if ara_result is not None and ara_result.claim_coverage is not None
                    else None
                ),
                "ara_claim_coverage_severity": (
                    ara_result.claim_coverage.severity
                    if ara_result is not None and ara_result.claim_coverage is not None
                    else None
                ),
                "paper_type": paper_type,
                "idea_name": idea_name,
                "workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
                "execution_policy": (research_plan.get("execution_policy") or {}).get(
                    "policy_name"
                ),
                "evidence_pressure": (research_plan.get("execution_policy") or {}).get(
                    "evidence_pressure"
                ),
                "quality_fallback_policy": (
                    research_plan.get("execution_policy") or {}
                ).get("quality_fallback_policy"),
                "review_roles_improvement": list(
                    workflow_runtime_plan.improvement_review_roles
                ),
                "review_roles_final": list(workflow_runtime_plan.final_review_roles),
                "review_roles_used": sorted(
                    set(workflow_runtime_plan.improvement_review_roles)
                    | set(workflow_runtime_plan.final_review_roles)
                    | set(workflow_runtime_plan.critic_review_roles)
                ),
                "critic_roles_used": list(workflow_runtime_plan.critic_review_roles),
                "critic_active_issue_count": critic_pass.get("active_issue_count"),
                "critic_blocking_issue_count": critic_pass.get("blocking_issue_count"),
                "critic_findings_file": critic_pass.get("critic_findings_file"),
                "pipeline_manifest": str(
                    Path(paper_structure["root"]) / "pipeline_manifest.json"
                ),
                "idea_card_id": idea_card.get("idea_id"),
                "research_plan_file": str(
                    Path(paper_structure["root"]) / "research_plan.json"
                ),
                "claim_evidence_graph_file": str(
                    Path(paper_structure["root"]) / "claim_evidence_graph.json"
                ),
                "experiment_registry_file": str(
                    Path(paper_structure["root"]) / "experiment_registry.jsonl"
                ),
                "figure_spec_file": str(
                    Path(paper_structure["root"]) / "figure_spec.json"
                ),
                "manuscript_state_file": str(
                    Path(paper_structure["root"]) / "manuscript_state.json"
                ),
                "review_state_file": str(
                    Path(paper_structure["root"]) / "review_state.json"
                ),
                "repair_plan_file": str(
                    Path(paper_structure["root"]) / "repair_plan.json"
                ),
                "self_evolution_file": str(
                    Path(paper_structure["root"]) / "self_evolution.json"
                ),
                "stage_standards_file": str(
                    Path(paper_structure["root"]) / "stage_standards.json"
                ),
                "writing_profile": writing_profile,
                "writing_audit_rounds": writing_audit_rounds,
                "strict_writing_guardrails": strict_writing_guardrails,
                "guardrail_repair_rounds": guardrail_repair_rounds,
                "target_venue": quality_result.get("target_venue", selected_venue),
                "quality_score": quality_result.get("quality_score_after"),
                "breakthrough_score": quality_result.get("breakthrough_score"),
                "rigor_score": quality_result.get("rigor_score_after"),
                "claim_support_score": quality_result.get("claim_support_after"),
                "claim_alignment_score": quality_result.get("claim_alignment_after"),
                "numeric_coverage_score": quality_result.get("numeric_coverage_after"),
                "evidence_density_score": quality_result.get("evidence_density_score"),
                "unsupported_claims_count": quality_result.get(
                    "unsupported_claims_count"
                ),
                "quality_gate_passed": quality_result.get("quality_gate_passed"),
                "quality_status": quality_result.get("quality_status"),
                "submission_priority_score": quality_result.get(
                    "submission_priority_score"
                ),
                "submission_priority_tier": quality_result.get(
                    "submission_priority_tier"
                ),
                "blocker_count": quality_result.get("blocker_count"),
                "autonomous_followup_rounds_run": quality_result.get(
                    "autonomous_followup_rounds_run"
                ),
                "auto_improvement_fallback_used": quality_result.get(
                    "auto_improvement_fallback_used"
                ),
                "submission_acceptance_passed": (
                    acceptance.get("accepted") if high_quality_mode else None
                ),
                "submission_package_file": quality_result.get(
                    "submission_package_file"
                ),
                "submission_dashboard_file": quality_result.get(
                    "submission_dashboard_file"
                ),
                "revision_actions": quality_result.get("revision_actions", []),
                "logic_check_file": quality_result.get("logic_check_file"),
                "reviewer_gate_report_file": quality_result.get(
                    "reviewer_gate_report_file"
                ),
                **integrity_forensics_result_fields(integrity_forensics_result),
                "experiment_analysis_file": quality_result.get(
                    "experiment_analysis_file"
                ),
                "experiment_visualization_brief_file": quality_result.get(
                    "experiment_visualization_brief_file"
                ),
                "figure_caption_guidance_file": quality_result.get(
                    "figure_caption_guidance_file"
                ),
                "table_caption_guidance_file": quality_result.get(
                    "table_caption_guidance_file"
                ),
                "architecture_figure_brief_file": quality_result.get(
                    "architecture_figure_brief_file"
                ),
                "humanizer_style_notes_file": quality_result.get(
                    "humanizer_style_notes_file"
                ),
                "writing_skill_pack_file": quality_result.get(
                    "writing_skill_pack_file"
                ),
            }
        else:
            return {
                "idea_idx": idea_idx,
                "status": "failed",
                "stage": "final_pdf_not_found",
                "paper_type": paper_type,
                "writing_profile": writing_profile,
                "writing_audit_rounds": writing_audit_rounds,
                "strict_writing_guardrails": strict_writing_guardrails,
                "guardrail_repair_rounds": guardrail_repair_rounds,
                "workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
                "execution_policy": (research_plan.get("execution_policy") or {}).get(
                    "policy_name"
                ),
                "review_roles_improvement": list(
                    workflow_runtime_plan.improvement_review_roles
                ),
                "review_roles_final": list(workflow_runtime_plan.final_review_roles),
                "review_roles_used": sorted(
                    set(workflow_runtime_plan.improvement_review_roles)
                    | set(workflow_runtime_plan.final_review_roles)
                    | set(workflow_runtime_plan.critic_review_roles)
                ),
                "critic_roles_used": list(workflow_runtime_plan.critic_review_roles),
            }

    except StrictFallbackViolation as exc:
        print(f"\n🛑 [想法 #{idea_idx}] 严格兜底策略阻断: {exc}")
        return {
            "idea_idx": idea_idx,
            "status": "failed",
            "stage": "quality_fallback_blocked",
            "reason": str(exc),
            "workflow_mode": workflow_mode,
            "paper_type": paper_type,
            "writing_profile": writing_profile,
            "high_quality_mode": high_quality_mode,
        }
    except Exception as e:
        print(f"\n❌ [想法 #{idea_idx}] 处理失败: {e}")
        traceback.print_exc()
        failure_runtime_plan = (
            workflow_runtime_plan
            if "workflow_runtime_plan" in locals()
            else build_workflow_runtime_plan(
                workflow_mode,
                submission_mode=bool(submission_mode),
                high_quality_mode=bool(high_quality_mode),
                target_venue=target_venue,
            )
        )
        return {
            "idea_idx": idea_idx,
            "status": "failed",
            "stage": current_stage,
            "error": str(e),
            "pause_reason": (
                "unsatisfiable_gate" if isinstance(e, UnsatisfiableGateError) else None
            ),
            "gate_reasons": (
                str(e).replace("Unsatisfiable gate preconditions: ", "").split(", ")
                if isinstance(e, UnsatisfiableGateError)
                else []
            ),
            "traceback": traceback.format_exc(),
            "paper_type": paper_type,
            "writing_profile": writing_profile,
            "writing_audit_rounds": writing_audit_rounds,
            "strict_writing_guardrails": strict_writing_guardrails,
            "guardrail_repair_rounds": guardrail_repair_rounds,
            "workflow_mode": workflow_mode,
            "template_profile": (
                template_profile if "template_profile" in locals() else None
            ),
            "template_capability": (
                template_capability if "template_capability" in locals() else None
            ),
            "execution_policy": (
                (research_plan.get("execution_policy") or {}).get("policy_name")
                if "research_plan" in locals()
                else None
            ),
            "review_roles_improvement": list(
                failure_runtime_plan.improvement_review_roles
            ),
            "review_roles_final": list(failure_runtime_plan.final_review_roles),
            "review_roles_used": sorted(
                set(failure_runtime_plan.improvement_review_roles)
                | set(failure_runtime_plan.final_review_roles)
                | set(failure_runtime_plan.critic_review_roles)
            ),
            "critic_roles_used": list(failure_runtime_plan.critic_review_roles),
        }


def main(argv=None):
    from ai_scientist.apps.batch_cli import build_parser

    default_writing_profile = resolve_writing_profile_env(
        invalid_profile_logger=lambda exc, raw: print(
            "⚠️  忽略无效的 AI_SCIENTIST_WRITING_PROFILE="
            f"{raw!r}，回退为 {DEFAULT_WRITING_PROFILE}"
        )
    )

    parser = build_parser(
        default_research_dir=str(resolve_output_path()),
        default_writing_profile=default_writing_profile,
        writing_profiles=list_writing_profiles(),
        workflow_modes=list_workflow_modes(),
    )
    args = parser.parse_args(argv)
    require_login("连续论文生成(continuous_paper_generator)")
    requested_workflow_mode = args.workflow_mode
    normalize_batch_workflow_args(args)
    strict_fallbacks = should_enforce_strict_fallbacks(
        args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        high_quality_mode=bool(args.high_quality_mode),
        target_venue=args.target_venue,
    )
    if args.override_strict_fallbacks and strict_fallbacks:
        print(
            "⚠️  override-strict-fallbacks: 严格兜底拦截已禁用，本批次可能继续在 fallback 状态下运行。"
        )
        strict_fallbacks = False
    elif strict_fallbacks:
        print(
            "🛡️  Strict fallback policy active: 任何 fallback 将直接终止本批次（使用 --override-strict-fallbacks 可临时放宽）。"
        )
    integrity_forensics_enabled = (
        bool(args.integrity_forensics)
        if args.integrity_forensics is not None
        else bool(args.submission_mode or args.high_quality_mode)
    )
    if integrity_forensics_enabled:
        print("🔎 启用 integrity forensics：每篇最终稿将生成可追溯一致性检查报告。")
    runtime = initialize_runtime(
        source_file=__file__,
        project_root=PROJECT_ROOT,
        output_root=args.research_dir,
        ensure_dirs=True,
        apply_cache=True,
    )
    args.research_dir = str(runtime.research_root)

    # 按实际模型检查 provider 凭证
    requested_models = _collect_requested_models(args)
    require_model_credentials(requested_models)

    # 确定论文类型
    paper_types = list(PAPER_TYPES.keys()) if args.all_types else args.paper_types

    paper_types = resolve_paper_types_for_venue(
        paper_types,
        args.target_venue,
        auto_adjust=args.auto_adjust_paper_type,
        warning_template="⚠️  警告: paper_type '{paper_type}' 与目标 venue '{target_venue}' 可能不够匹配",
        adjusted_template="✅ 自动调整 paper_type '{paper_type}' -> '{adjusted}'",
    )

    # 创建生成器
    generator = ContinuousPaperGenerator(
        research_dir=args.research_dir,
        batch_name=args.batch_name,
        paper_types=paper_types,
        strict_fallbacks=strict_fallbacks,
    )
    record_model_provider_decisions(
        generator.batch_dir,
        requested_models,
        producer="continuous_paper_generator.model_provider",
    )

    # 确定想法文件
    ideas_json = None

    if args.topic:
        ideas_json = generator.generate_ideas(
            topic_file=args.topic,
            num_ideas=args.num_ideas,
            model=args.model_ideation,
            num_reflections=args.num_reflections,
        )
    elif args.ideas:
        ideas_json = generator.load_existing_ideas(args.ideas)
    else:
        print("❌ 错误: 需要指定 --topic 或 --ideas")
        sys.exit(1)

    # 确定要处理的想法索引
    requested_indices = (
        [int(x.strip()) for x in args.idea_indices.split(",")]
        if args.idea_indices
        else None
    )
    with open(ideas_json, "r") as f:
        ideas = json.load(f)
    batch_idea_cards = _save_batch_pipeline_seed_artifacts(
        batch_dir=generator.batch_dir,
        ideas=ideas,
        target_venue=args.target_venue,
        workflow_mode=args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        breakthrough_mode=bool(args.breakthrough_mode),
        high_quality_mode=bool(args.high_quality_mode),
    )
    if batch_idea_cards:
        generator.progress["idea_cards_file"] = str(
            Path(generator.batch_dir) / "idea_cards.json"
        )
        generator.progress["research_program_file"] = str(
            Path(generator.batch_dir) / "research_program.md"
        )
        generator._save_progress()
        print("🧭 批次级 idea cards / research_program 已生成")

    idea_indices, rankings = select_ranked_idea_candidates(
        ideas,
        ranking_enabled=args.rank_ideas,
        ranking_model=args.idea_rank_model or args.model_writeup,
        target_venue=args.target_venue,
        prioritize_breakthrough=(args.target_venue == "nature"),
        research_root=generator.research_dir,
        ranking_output_path=generator.batch_dir / "ideas" / "idea_rankings.json",
        requested_indices=requested_indices,
        default_indices=list(range(len(ideas))),
        use_ranked_all=True,
        limit=args.top_k_ideas if args.rank_ideas else None,
    )
    if rankings:
        ranking_event = record_ranking_fallbacks(
            generator.batch_dir,
            rankings,
            producer="continuous_paper_generator.idea_ranking",
            strict=strict_fallbacks,
        )
        if strict_fallbacks and ranking_event:
            print(
                "❌ "
                + format_strict_fallback_error(
                    ranking_event,
                    workflow_mode=args.workflow_mode,
                    stage_hint="idea_ranking",
                )
            )
            sys.exit(1)
        generator.progress["idea_rankings_file"] = str(
            generator.batch_dir / "ideas" / "idea_rankings.json"
        )
        if args.top_k_ideas is not None:
            generator.progress["selected_idea_indices"] = idea_indices
        generator._save_progress()
        print("\n🏅 Idea ranking complete")
        for item in rankings[: min(5, len(rankings))]:
            print(
                f"  - idea #{item['idea_idx']} score={item.get('total_score')} "
                f"name={item.get('idea_name')}"
            )

    # 生成论文
    print("\n" + "=" * 80)
    print("🚀 开始批量论文生成")
    print("=" * 80)

    kwargs = {
        "submission_mode": args.submission_mode,
        "bfts_config": args.bfts_config,
        "model_writeup": args.model_writeup,
        "model_citation": args.model_citation,
        "model_review": args.model_review,
        "model_agg_plots": args.model_agg_plots,
        "model_writeup_small": args.model_writeup_small,
        "num_cite_rounds": args.num_cite_rounds,
        "writeup_retries": args.writeup_retries,
        "improvement_rounds": args.improvement_rounds,
        "num_workers": args.num_workers,
        "high_quality_mode": args.high_quality_mode,
        "quality_preset": args.quality_preset,
        "quality_model": args.quality_model,
        "target_venue": args.target_venue,
        "quality_threshold": args.quality_threshold,
        "rigor_threshold": args.rigor_threshold,
        "quality_rewrite_rounds": args.quality_rewrite_rounds,
        "autonomous_quality_followup_rounds": args.autonomous_quality_followup_rounds,
        "require_quality_gate": args.require_quality_gate,
        "min_submission_priority": args.min_submission_priority,
        "max_submission_blockers": args.max_submission_blockers,
        "review_reflections": args.review_reflections,
        "review_ensemble": args.review_ensemble,
        "review_fewshot": args.review_fewshot,
        "review_temperature": args.review_temperature,
        "review_strategy": args.review_strategy,
        "writing_profile": args.writing_profile,
        "writing_audit_rounds": args.writing_audit_rounds,
        "strict_writing_guardrails": args.strict_writing_guardrails,
        "guardrail_repair_rounds": args.guardrail_repair_rounds,
        "workflow_mode": args.workflow_mode,
        "integrity_forensics_enabled": integrity_forensics_enabled,
        "requested_workflow_mode": requested_workflow_mode,
    }

    if args.all_types:
        results = generator.generate_all_paper_types(
            ideas_json=ideas_json,
            idea_indices=idea_indices,
            **kwargs,
        )
    else:
        for paper_type in paper_types:
            generator.generate_paper_batch(
                ideas_json=ideas_json,
                paper_type=paper_type,
                idea_indices=idea_indices,
                **kwargs,
            )

    # 生成总结报告
    report_path = generator.generate_summary_report()

    constraints_active = (
        args.require_quality_gate
        or args.min_submission_priority is not None
        or args.max_submission_blockers is not None
    )
    if constraints_active:
        try:
            with open(report_path, "r") as f:
                report = json.load(f)
            accepted = report.get("quality_summary", {}).get("submission_accepted", 0)
            if accepted == 0:
                print("❌ 没有任何论文达到当前投稿接受标准，批量任务视为失败")
                sys.exit(1)
        except Exception as exc:
            print(f"⚠️  读取总结报告失败，无法验证投稿接受标准: {exc}")
    if args.strict_writing_guardrails:
        try:
            with open(report_path, "r") as f:
                report = json.load(f)
            guardrail_passed = int(
                report.get("quality_summary", {}).get("guardrail_passed", 0)
            )
            if guardrail_passed == 0:
                print(
                    "❌ 严格写作守护开启，但没有任何论文通过写作守护检查，批量任务视为失败"
                )
                sys.exit(1)
        except Exception as exc:
            print(f"⚠️  读取总结报告失败，无法验证严格写作守护结果: {exc}")


if __name__ == "__main__":
    main()
