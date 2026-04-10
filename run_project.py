#!/usr/bin/env python3
"""
AI Scientist 项目管理器
在一个独立文件夹中运行完整的科研流程：想法生成 -> 实验 -> 论文
支持并行处理多个论文和自动反思改进
不影响 ai_scientist 原有代码结构
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 添加项目根目录到路径
PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ai_scientist.utils.deferred_imports import load_module_attr
from ai_scientist.utils.high_quality_pipeline import run_high_quality_pass
from ai_scientist.utils.workflow_cli import normalize_project_workflow_args
from ai_scientist.utils.fallback_audit import (
    format_strict_fallback_error,
    record_quality_fallback_if_needed,
    record_ranking_fallbacks,
    should_enforce_strict_fallbacks,
)
from ai_scientist.utils.pipeline_helpers import (
    find_latest_bfts_run_dir,
    find_latest_pdf_path,
)
from ai_scientist.utils.quality_workflow import (
    evaluate_final_submission_readiness,
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
    resolve_paper_type_for_venue,
    select_ranked_idea_candidates,
)
from ai_scientist.utils.experiment_report import write_experiment_report
from ai_scientist.utils.experiment_registry import (
    build_experiment_record,
    save_experiment_registry,
)
from ai_scientist.utils.guardrail_artifacts import (
    load_guardrail_artifacts,
    result_passed_writeup_guardrails,
)
from ai_scientist.utils.manuscript_state import (
    build_manuscript_state,
    save_manuscript_state,
)
from ai_scientist.utils.auth_session import require_login
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    render_research_program_markdown,
    save_contract_artifact,
    update_pipeline_artifact,
)
from ai_scientist.utils.research_planning import (
    build_claim_evidence_graph,
    build_idea_cards,
    build_research_plan,
)
from ai_scientist.utils.self_review_optimizer import (
    apply_issue_driven_rewrite,
    assess_self_review_gate,
    build_issue_ledger,
    evaluate_issue_progress,
    save_self_review_artifacts,
)
from ai_scientist.utils.experiment_todo_progress import (
    bootstrap_todo_tasks_from_round_gate,
    build_todo_progress_payload,
    evaluate_todo_progress_snapshot,
    load_experiment_todo_payload,
    save_todo_progress_artifacts,
)
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    list_writing_profiles,
)
from ai_scientist.writing_skill_pack import list_writing_skills
from ai_scientist.utils.runtime_bootstrap import (
    format_project_relative_path,
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

# 导入路径配置
from ai_scientist.config.paths import (
    get_project_dir,
    resolve_output_path,
)

_HOSTILE_CRITIC_ABLATION_ENV = "AI_SCIENTIST_ABLATE_HOSTILE_CRITIC"
_OWNER_AWARE_REPAIR_ABLATION_ENV = "AI_SCIENTIST_DISABLE_OWNER_AWARE_REPAIR"


def create_client(*args, **kwargs):
    return load_module_attr("ai_scientist.llm", "create_client")(*args, **kwargs)


def generate_temp_free_idea(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.perform_ideation_temp_free",
        "generate_temp_free_idea",
    )(*args, **kwargs)


def configure_benchmark_ablation_env(
    *,
    disable_hostile_critic: bool = False,
    disable_owner_aware_repair: bool = False,
) -> dict[str, bool]:
    """Make benchmark ablation flags explicit and reproducible for this run."""

    os.environ.pop(_HOSTILE_CRITIC_ABLATION_ENV, None)
    os.environ.pop(_OWNER_AWARE_REPAIR_ABLATION_ENV, None)
    if disable_hostile_critic:
        os.environ[_HOSTILE_CRITIC_ABLATION_ENV] = "1"
    if disable_owner_aware_repair:
        os.environ[_OWNER_AWARE_REPAIR_ABLATION_ENV] = "1"
    return {
        "disable_hostile_critic": bool(disable_hostile_critic),
        "disable_owner_aware_repair": bool(disable_owner_aware_repair),
    }


def perform_experiments_bfts(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
        "perform_experiments_bfts",
    )(*args, **kwargs)


def idea_to_markdown(*args, **kwargs):
    return load_module_attr("ai_scientist.treesearch.bfts_utils", "idea_to_markdown")(
        *args, **kwargs
    )


def edit_bfts_config_file(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.treesearch.bfts_utils",
        "edit_bfts_config_file",
    )(*args, **kwargs)


def aggregate_plots(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_plotting", "aggregate_plots")(
        *args, **kwargs
    )


def perform_icbinb_writeup(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_icbinb_writeup", "perform_writeup")(
        *args, **kwargs
    )


def gather_citations(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_icbinb_writeup", "gather_citations")(
        *args, **kwargs
    )


def perform_normal_writeup(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_writeup", "perform_writeup")(
        *args, **kwargs
    )


def perform_review(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "perform_review")(
        *args, **kwargs
    )


def load_paper(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "load_paper")(
        *args, **kwargs
    )


def perform_imgs_cap_ref_review(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.perform_vlm_review",
        "perform_imgs_cap_ref_review",
    )(*args, **kwargs)


def resolve_project_path(
    project_dir: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    """解析项目目录路径（相对路径会落到统一输出根目录下）。"""
    project_path = Path(project_dir).expanduser()
    if project_path.is_absolute():
        return project_path
    return get_project_dir(project_dir, output_root=output_root)


def create_project_structure(
    project_dir: str,
    *,
    output_root: str | Path | None = None,
) -> dict:
    """创建项目目录结构。"""
    project_path = resolve_project_path(project_dir, output_root=output_root)

    # 创建子目录
    dirs = {
        "root": project_path,
        "ideas": project_path / "01_ideas",
        "experiments": project_path / "02_experiments",
        "papers": project_path / "03_papers",
        "logs": project_path / "04_logs",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"✅ 创建项目目录结构: {project_path}")
    print(
        "   (相对于项目根目录: "
        f"{format_project_relative_path(project_path, project_root=PROJECT_ROOT)})"
    )
    return dirs


def _safe_load_json(path: str | Path, *, default=None):
    path_obj = Path(path)
    if not path_obj.exists():
        return default
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
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
        model = str(value or "").strip()
        if not model or model in seen:
            continue
        models.append(model)
        seen.add(model)
    return models


def _write_project_pipeline_seed_artifacts(
    *,
    project_dir: str | Path,
    ideas: list[dict],
    target_venue: str | None,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
) -> list[dict]:
    project_root = Path(project_dir).expanduser().resolve()
    workflow_spec, workflow_metadata, template_profile, template_capability = _resolve_workflow_strategy(
        workflow_mode=workflow_mode,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
        target_venue=target_venue,
    )
    initialize_pipeline_contracts(
        project_root,
        project_name=project_root.name,
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
        project_root,
        "idea_cards",
        idea_cards,
        producer="run_project.project_seed",
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
        project_name=project_root.name,
        target_venue=target_venue,
        template_profile=template_profile,
        idea_name=str(lead_idea.get("name") or project_root.name),
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
            "Preserve partial outputs and mark failed stages in pipeline_manifest.json.",
            "Do not let downstream stages consume artifacts unless their contract status is ready.",
        ],
    )
    save_contract_artifact(
        project_root,
        "research_program",
        research_program,
        producer="run_project.project_seed",
        depends_on=["idea_cards"],
    )
    save_stage_standards(project_root)
    return idea_cards


def _write_experiment_pipeline_seed_artifacts(
    *,
    exp_dir: str | Path,
    idea: dict,
    idea_idx: int,
    target_venue: str | None,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool,
    high_quality_mode: bool,
    template_profile: str,
    template_capability: str,
) -> tuple[dict, dict, dict]:
    exp_root = Path(exp_dir).expanduser().resolve()
    workflow_spec, workflow_metadata, _, _ = _resolve_workflow_strategy(
        workflow_mode=workflow_mode,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
        target_venue=target_venue,
    )
    initialize_pipeline_contracts(
        exp_root,
        project_name=exp_root.name,
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
        exp_root,
        "idea_cards",
        [idea_card],
        producer="run_project.idea_seed",
    )
    research_plan = build_research_plan(
        idea_card,
        target_venue=target_venue,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
    )
    save_contract_artifact(
        exp_root,
        "research_plan",
        research_plan,
        producer="run_project.planning",
        depends_on=["idea_cards"],
    )
    claim_evidence_graph = build_claim_evidence_graph(idea_card, research_plan)
    save_contract_artifact(
        exp_root,
        "claim_evidence_graph",
        claim_evidence_graph,
        producer="run_project.planning",
        depends_on=["research_plan"],
    )
    research_program = render_research_program_markdown(
        project_name=exp_root.name,
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
            "Failed experiments must still be recorded in experiment_registry.jsonl.",
            "Claims and figures remain blocked until their supporting evidence is ready.",
        ],
    )
    save_contract_artifact(
        exp_root,
        "research_program",
        research_program,
        producer="run_project.planning",
        depends_on=["research_plan"],
    )
    save_stage_standards(exp_root)
    return idea_card, research_plan, claim_evidence_graph


def _build_experiment_registry_rows(
    *,
    exp_dir: str | Path,
    research_plan: dict,
) -> list[dict]:
    exp_root = Path(exp_dir).expanduser().resolve()
    experiment_report = _safe_load_json(exp_root / "experiment_report.json", default={}) or {}
    stages = list(experiment_report.get("stages") or [])
    warnings = list(experiment_report.get("warnings") or [])
    latest_run_dir = experiment_report.get("latest_run_dir")
    rows: list[dict] = []
    for idx, task in enumerate(research_plan.get("tasks") or []):
        stage = stages[min(idx, len(stages) - 1)] if stages else {}
        best = stage.get("best") or {}
        artifacts = {
            "exp_dir": str(exp_root),
            "experiment_report_json": str(exp_root / "experiment_report.json"),
            "experiment_report_md": str(exp_root / "experiment_report.md"),
            "latest_run_dir": latest_run_dir,
            "stage_dir": stage.get("stage_dir"),
            "journal_path": stage.get("journal_path"),
        }
        if best:
            status = "completed"
            result_summary = {
                "metric_name": best.get("metric_name"),
                "metric_mean": best.get("metric_mean"),
                "metric_objective": best.get("metric_objective"),
                "dataset_names": best.get("dataset_names") or [task.get("dataset")],
                "seed_eval": best.get("seed_eval"),
                "delta_objective_vs_prev_stage": stage.get("delta_objective_vs_prev_stage"),
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
            error_type = "missing_best_result" if stages else "missing_experiment_report"
            error_message = "; ".join(warnings[:3]) or "No valid experiment summary was detected."
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
                artifacts=artifacts,
                error_type=error_type,
                error_message=error_message,
                entered_storyline=entered_storyline,
                budget=task.get("budget"),
                workflow_mode=research_plan.get("workflow_mode"),
                policy_name=(research_plan.get("execution_policy") or {}).get("policy_name"),
                acceptance_checks=task.get("acceptance_checks"),
            )
        )
    return rows


def _save_manuscript_contract_state(
    *,
    exp_dir: str | Path,
    writeup_type: str,
    target_venue: str | None,
    writing_profile: str,
) -> dict:
    exp_root = Path(exp_dir).expanduser().resolve()
    claim_evidence_graph = load_contract_artifact(
        exp_root,
        "claim_evidence_graph",
        default={},
    )
    figure_spec = load_contract_artifact(
        exp_root,
        "figure_spec",
        default={},
    )
    manuscript_state = build_manuscript_state(
        writeup_type=writeup_type,
        target_venue=target_venue,
        writing_profile=writing_profile,
        skill_pack=list_writing_skills(),
        claim_evidence_graph=(
            claim_evidence_graph if isinstance(claim_evidence_graph, dict) else {}
        ),
        figure_spec=figure_spec if isinstance(figure_spec, dict) else {},
        latex_path=str(exp_root / "template.tex"),
    )
    save_manuscript_state(str(exp_root), manuscript_state)
    return manuscript_state


def generate_ideas(project_dir: str, topic_file: str, model: str, num_ideas: int, reflections: int):
    """步骤1: 生成研究想法"""
    print("\n" + "=" * 80)
    print("🔬 步骤 1: 生成研究想法")
    print("=" * 80)

    ideas_dir = osp.join(project_dir, "01_ideas")

    # 读取主题描述
    with open(topic_file, "r") as f:
        workshop_description = f.read()

    # 输出文件路径
    idea_json = osp.join(ideas_dir, "generated_ideas.json")
    stable_idea_json = osp.join(ideas_dir, "ideas.json")

    # 创建客户端
    client, client_model = create_client(model)

    # 生成想法
    ideas = generate_temp_free_idea(
        idea_fname=idea_json,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=num_ideas,
        num_reflections=reflections,
    )

    print(f"✅ 生成了 {len(ideas)} 个想法，保存到 {idea_json}")
    # Benchmarks (and downstream workflows) expect a stable `ideas.json` fixture.
    # Keep the historical `generated_ideas.json` name but also mirror it to `ideas.json`.
    try:
        shutil.copy(idea_json, stable_idea_json)
        print(f"✅ 已同步 ideas fixture: {stable_idea_json}")
    except OSError as exc:
        print(f"⚠️  未能同步 ideas.json fixture: {exc}")
    return idea_json


def find_latest_pdf(exp_dir: str):
    """查找最新的PDF文件"""
    return find_latest_pdf_path(exp_dir)


def improve_paper_with_review(
    exp_dir: str,
    review_text: dict,
    review_img: dict,
    model: str,
    *,
    round_index: int,
    review_dir: str,
    previous_issue_ledger=None,
    target_venue: str | None = None,
):
    """根据审查意见进行 issue-driven 论文改进。"""
    print("\n" + "=" * 80)
    print(f"🔧 根据审查意见改进论文 (issue-driven, round {round_index})")
    print("=" * 80)

    issue_ledger = build_issue_ledger(
        text_review=review_text,
        img_review=review_img,
        max_issues=14,
        previous_ledger=previous_issue_ledger,
        target_venue=target_venue,
    )
    print(
        "🧭 本轮问题台账: "
        f"{issue_ledger.get('issue_count', 0)} "
        f"(critical={issue_ledger.get('critical_count', 0)}, "
        f"major={issue_ledger.get('major_count', 0)}, minor={issue_ledger.get('minor_count', 0)})"
    )

    progress = None
    if isinstance(previous_issue_ledger, dict):
        progress = evaluate_issue_progress(
            previous_issues=list(previous_issue_ledger.get("issues") or []),
            current_issues=list(issue_ledger.get("issues") or []),
        )
        print(
            "📈 跨轮问题进展: "
            f"resolved={progress.get('resolved_issue_count', 0)}, "
            f"persistent={progress.get('persistent_issue_count', 0)}, "
            f"new={progress.get('new_issue_count', 0)}, "
            f"unresolved_critical={progress.get('unresolved_critical_count', 0)}"
        )

    rewrite_result = apply_issue_driven_rewrite(
        paper_dir=exp_dir,
        model=model,
        ledger=issue_ledger,
        round_index=round_index,
        artifact_dir=review_dir,
        target_venue=target_venue,
        temperature=0.35,
    )
    round_gate = assess_self_review_gate(
        ledger=issue_ledger,
        progress=progress,
        rewrite_result=rewrite_result,
        round_index=round_index,
        target_venue=target_venue,
    )
    artifact_files = save_self_review_artifacts(
        review_dir=review_dir,
        ledger=issue_ledger,
        progress=progress,
        gate=round_gate,
    )
    if rewrite_result.get("status") == "success":
        print(
            "✅ Issue-driven 改稿完成: "
            f"coverage={rewrite_result.get('coverage_ratio', 0):.2f}, "
            f"high_value_coverage={rewrite_result.get('high_value_coverage_ratio', 0):.2f}, "
            f"compile_ok={rewrite_result.get('compile_ok')}"
        )
    else:
        print(
            "⚠️  Issue-driven 改稿未完全成功: "
            f"status={rewrite_result.get('status')}, "
            f"reason={rewrite_result.get('reason')}"
        )
    print(
        "🚦 Round gate: "
        f"ready={round_gate.get('ready')}, "
        f"score={round_gate.get('score')}, "
        f"reasons={round_gate.get('reasons', [])}"
    )

    return {
        "status": "success"
        if rewrite_result.get("status") == "success"
        else "failed",
        "issue_ledger": issue_ledger,
        "issue_progress": progress,
        "round_gate": round_gate,
        "artifact_files": artifact_files,
        "rewrite_result": rewrite_result,
    }


def process_single_idea(args):
    """处理单个想法的完整流程（可在子进程中运行）"""
    (
        project_dir,
        research_root,
        idea_idx,
        idea,
        bfts_config_path,
        model_writeup,
        model_citation,
        model_review,
        model_agg_plots,
        model_writeup_small,
        num_cite_rounds,
        writeup_retries,
        writeup_type,
        improvement_rounds,
        review_reflections,
        review_ensemble,
        review_fewshot,
        review_temperature,
        review_strategy,
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
        writing_profile,
        writing_audit_rounds,
        strict_writing_guardrails,
        guardrail_repair_rounds,
        workflow_mode,
        template_profile,
        template_capability,
        strict_fallbacks,
    ) = args

    try:
        strict_writing_guardrails = bool(strict_writing_guardrails or high_quality_mode)
        guardrail_repair_rounds = max(0, int(guardrail_repair_rounds))
        research_root = Path(research_root).expanduser()
        idea_name = idea.get("Name", f"idea_{idea_idx}")
        workflow_spec, workflow_metadata, _, _ = _resolve_workflow_strategy(
            workflow_mode=workflow_mode,
            submission_mode=(template_profile == "template_first"),
            breakthrough_mode=False,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )

        print(f"\n{'='*80}")
        print(f"🚀 开始处理想法 #{idea_idx}: {idea_name}")
        print(f"{'='*80}")

        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = osp.join(project_dir, "02_experiments", f"{timestamp}_{idea_name}")
        os.makedirs(exp_dir, exist_ok=True)
        initialize_pipeline_contracts(
            exp_dir,
            project_name=Path(exp_dir).name,
            template_profile=template_profile,
            template_capability=template_capability,
            pipeline_goal=workflow_spec.pipeline_goal,
            workflow_mode=workflow_metadata["workflow_mode"],
            workflow_label=workflow_metadata["workflow_label"],
            workflow_summary=workflow_metadata["workflow_summary"],
            workflow_inspirations=workflow_metadata["workflow_inspirations"],
            workflow_sequence=workflow_metadata["workflow_sequence"],
        )

        # 保存想法
        idea_path_json = osp.join(exp_dir, "idea.json")
        with open(idea_path_json, "w") as f:
            json.dump(idea, f, indent=4)
        idea_card, research_plan, claim_evidence_graph = (
            _write_experiment_pipeline_seed_artifacts(
                exp_dir=exp_dir,
                idea=idea,
                idea_idx=idea_idx,
                target_venue=target_venue,
                workflow_mode=workflow_mode,
                submission_mode=(template_profile == "template_first"),
                breakthrough_mode=False,
                high_quality_mode=high_quality_mode,
                template_profile=template_profile,
                template_capability=template_capability,
            )
        )

        # ========== 步骤1: 运行实验 ==========
        print(f"\n📊 [想法 #{idea_idx}] 步骤 1/4: 运行实验")
        idea_path_md = osp.join(exp_dir, "idea.md")
        idea_to_markdown(idea, idea_path_md, None)

        config_path = str(bfts_config_path or "bfts_config.yaml")
        if not osp.isabs(config_path):
            config_path = osp.join(PROJECT_ROOT, config_path)
        idea_config_path = edit_bfts_config_file(config_path, exp_dir, idea_path_json)

        perform_experiments_bfts(idea_config_path)

        # 复制实验结果
        latest_run_dir = find_latest_bfts_run_dir(exp_dir, logs_subdir="logs")
        experiment_results_dir = (
            osp.join(str(latest_run_dir), "experiment_results")
            if latest_run_dir is not None
            else osp.join(exp_dir, "logs/0-run/experiment_results")
        )
        if osp.exists(experiment_results_dir):
            shutil.copytree(
                experiment_results_dir,
                osp.join(exp_dir, "experiment_results"),
                dirs_exist_ok=True,
            )
        # Persist a deterministic experiment evaluation report for iterative debugging.
        try:
            write_experiment_report(exp_dir)
        except Exception as exc:
            print(f"[想法 #{idea_idx}] ⚠️  实验报告生成失败: {exc}")
        registry_rows = _build_experiment_registry_rows(
            exp_dir=exp_dir,
            research_plan=research_plan,
        )
        save_experiment_registry(exp_dir, registry_rows)

        # ========== 步骤2: 生成论文 ==========
        print(f"\n📝 [想法 #{idea_idx}] 步骤 2/4: 生成论文")
        aggregate_plots(base_folder=exp_dir, model=model_agg_plots)
        _save_manuscript_contract_state(
            exp_dir=exp_dir,
            writeup_type=writeup_type,
            target_venue=target_venue,
            writing_profile=writing_profile,
        )

        experiment_results = osp.join(exp_dir, "experiment_results")
        if osp.exists(experiment_results):
            shutil.rmtree(experiment_results)

        writeup_plan = build_writeup_execution_plan(
            writeup_type,
            num_cite_rounds=num_cite_rounds,
            writeup_retries=writeup_retries,
            target_venue=target_venue,
            high_quality_mode=high_quality_mode,
            research_root=research_root,
        )
        selected_venue = writeup_plan["target_venue"]
        strategy_feedback = writeup_plan["strategy_feedback"]
        effective_num_cite_rounds = writeup_plan["num_cite_rounds"]
        effective_writeup_retries = writeup_plan["writeup_retries"]
        _save_manuscript_contract_state(
            exp_dir=exp_dir,
            writeup_type=writeup_type,
            target_venue=selected_venue,
            writing_profile=writing_profile,
        )
        if high_quality_mode:
            print(f"[想法 #{idea_idx}] 历史预算反馈: {strategy_feedback.get('rationale', [])}")

        citations_text = gather_citations(
            exp_dir,
            num_cite_rounds=effective_num_cite_rounds,
            small_model=model_citation,
        )

        writeup_success = False
        writeup_func = (
            perform_normal_writeup
            if writeup_plan["writeup_engine"] == "normal"
            else perform_icbinb_writeup
        )
        page_limit = writeup_plan["page_limit"]
        for attempt in range(effective_writeup_retries):
            print(f"[想法 #{idea_idx}] 论文写作尝试 {attempt + 1}/{effective_writeup_retries}")
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
                "exp_dir": exp_dir,
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

        quality_policy = research_plan.get("execution_policy") or {}
        quality_pass = {}
        quality_result = {}
        acceptance = {}
        final_submission_gate = {}
        if high_quality_mode:
            print(
                f"[想法 #{idea_idx}] 质量兜底策略: "
                f"{quality_policy.get('quality_fallback_policy', 'allowed')}"
            )
            quality_pass = execute_quality_workflow_with_followups(
                run_high_quality_pass_fn=run_high_quality_pass,
                run_dir=exp_dir,
                paper_type=writeup_type,
                rewrite_model=model_writeup,
                quality_model=quality_model or model_review,
                target_venue=selected_venue,
                quality_preset=quality_preset,
                quality_threshold=quality_threshold,
                rigor_threshold=rigor_threshold,
                max_quality_rewrites=quality_rewrite_rounds,
                require_quality_gate=require_quality_gate,
                min_submission_priority=min_submission_priority,
                max_submission_blockers=max_submission_blockers,
                autonomous_followup_rounds=autonomous_quality_followup_rounds,
                allow_auto_improvement_fallback=quality_policy.get(
                    "allow_auto_improvement_fallback"
                ),
                reject_on_auto_improvement_fallback=bool(
                    quality_policy.get("reject_on_auto_improvement_fallback")
                ),
                resume=False,
                logger=lambda msg: print(f"[想法 #{idea_idx}] {msg}"),
            )
            quality_result = quality_pass["quality_result"]
            quality_fallback_event = record_quality_fallback_if_needed(
                exp_dir,
                quality_result,
                producer="run_project.high_quality",
                strict=strict_fallbacks,
            )
            if strict_fallbacks and quality_fallback_event:
                return {
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "stage": "quality_fallback_blocked",
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                    "quality_fallback_policy": quality_policy.get(
                        "quality_fallback_policy"
                    ),
                    "reason": format_strict_fallback_error(
                        quality_fallback_event,
                        workflow_mode=workflow_mode,
                        stage_hint="quality_review",
                    ),
                    "quality_score": quality_result.get("quality_score_after"),
                    "rigor_score": quality_result.get("rigor_score_after"),
                    "submission_priority_score": quality_result.get(
                        "submission_priority_score"
                    ),
                    "blocker_count": quality_result.get("blocker_count"),
                }
            acceptance = quality_pass["acceptance"]
            print(
                f"[想法 #{idea_idx}] {quality_pass['summary'].replace('High-quality pass: ', '高质量模式: ')}"
            )
            if not acceptance.get("accepted"):
                print(
                    f"[想法 #{idea_idx}] ⚠️  预审质量门槛暂未满足，将继续进入严格 review/self-improvement："
                    f"{acceptance.get('reasons', [])}"
                )

        # ========== 步骤3: 反思和改进 ==========
        review_plan = build_review_execution_plan(
            writeup_type,
            target_venue=selected_venue,
            review_reflections=review_reflections,
            review_ensemble=review_ensemble,
            review_fewshot=review_fewshot,
            review_temperature=review_temperature,
            review_strategy=review_strategy,
            high_quality_mode=high_quality_mode,
            research_root=research_root,
            default_quality_requirement=(
                "high" if improvement_rounds > 0 else "standard"
            ),
        )
        workflow_runtime_plan = build_workflow_runtime_plan(
            workflow_spec,
            submission_mode=(template_profile == "template_first"),
            high_quality_mode=high_quality_mode,
            target_venue=selected_venue,
        )
        print(
            f"[想法 #{idea_idx}] 审稿编排: 改进轮={list(workflow_runtime_plan.improvement_review_roles)} "
            f"终审={list(workflow_runtime_plan.final_review_roles)}"
        )
        if high_quality_mode:
            print(
                f"[想法 #{idea_idx}] 历史审稿反馈: "
                f"{review_plan['strategy_feedback'].get('rationale', [])}"
            )

        previous_issue_ledger = None
        self_review_round_records = []
        self_review_summary_file = None
        self_review_summary_payload = None
        experiment_todo_payload = load_experiment_todo_payload(exp_dir)
        experiment_todo_round_snapshots = []
        experiment_todo_progress_payload = None
        experiment_todo_progress_files = {}
        final_experiment_todo_snapshot = None

        for round_num in range(improvement_rounds):
            print(f"\n🔍 [想法 #{idea_idx}] 步骤 3/4: 反思改进轮次 {round_num + 1}/{improvement_rounds}")

            review_dir = osp.join(exp_dir, f"reviews_round_{round_num + 1}")
            review_pass = execute_review_suite(
                review_roles=workflow_runtime_plan.improvement_review_roles,
                paper_dir=exp_dir,
                model_review=model_review,
                review_plan=review_plan,
                create_client_fn=create_client,
                load_paper_fn=load_paper,
                perform_review_fn=perform_review,
                perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
                pdf_path_resolver=find_latest_pdf,
                save_dir=review_dir,
                project_root=exp_dir,
                persist_job=True,
                evidence_refs=[
                    "claim_evidence_graph.json",
                    "experiment_registry.jsonl",
                    "figure_spec.json",
                    "manuscript_state.json",
                ],
                suite_name=f"improvement_round_{round_num + 1}",
                lane_name="review_board",
                strictness_profile="standard",
            )
            if not review_pass["found"]:
                print(f"[想法 #{idea_idx}] ⚠️  未找到PDF，跳过改进")
                break

            review_text = review_pass["review_text"]
            review_img = review_pass["review_img"]

            # 基于 issue ledger 的改进论文
            improvement_result = improve_paper_with_review(
                exp_dir,
                review_text,
                review_img,
                model_writeup,
                round_index=round_num + 1,
                review_dir=review_dir,
                previous_issue_ledger=previous_issue_ledger,
                target_venue=selected_venue,
            )
            round_gate = (
                improvement_result.get("round_gate")
                if isinstance(improvement_result.get("round_gate"), dict)
                else {}
            )
            issue_progress = (
                improvement_result.get("issue_progress")
                if isinstance(improvement_result.get("issue_progress"), dict)
                else None
            )

            if not (experiment_todo_payload.get("tasks") or []):
                bootstrap_tasks = bootstrap_todo_tasks_from_round_gate(
                    round_gate,
                    prefix=f"idea{idea_idx}",
                    max_tasks=8,
                )
                if bootstrap_tasks:
                    experiment_todo_payload["tasks"] = bootstrap_tasks
                    experiment_todo_payload["generated_at"] = datetime.now().isoformat()
                    experiment_todo_payload["bootstrap"] = True
                    todo_file = Path(exp_dir) / "experiment_todo.json"
                    todo_file.write_text(
                        json.dumps(experiment_todo_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    experiment_todo_payload["file"] = str(todo_file)

            round_todo_snapshot = evaluate_todo_progress_snapshot(
                experiment_todo_payload,
                round_gate=round_gate,
                issue_progress=issue_progress,
                round_index=round_num + 1,
            )
            if int((round_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0:
                experiment_todo_round_snapshots.append(round_todo_snapshot)
                print(
                    f"[想法 #{idea_idx}] TODO进展: closure={round_todo_snapshot.get('closure_rate')} "
                    f"p0_closure={round_todo_snapshot.get('p0_closure_rate')} "
                    f"unresolved={round_todo_snapshot.get('counts', {}).get('unresolved_tasks')}"
                )

            self_review_round_records.append({
                "round": round_num + 1,
                "issue_count": (
                    improvement_result.get("issue_ledger", {}).get("issue_count")
                    if isinstance(improvement_result.get("issue_ledger"), dict)
                    else None
                ),
                "critical_count": (
                    improvement_result.get("issue_ledger", {}).get("critical_count")
                    if isinstance(improvement_result.get("issue_ledger"), dict)
                    else None
                ),
                "progress": issue_progress,
                "round_gate": round_gate,
                "rewrite": improvement_result.get("rewrite_result"),
                "artifacts": improvement_result.get("artifact_files"),
                "experiment_todo_progress": (
                    round_todo_snapshot
                    if int((round_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0
                    else None
                ),
            })

            if isinstance(improvement_result.get("issue_ledger"), dict):
                previous_issue_ledger = improvement_result["issue_ledger"]

            rewrite_payload = improvement_result.get("rewrite_result", {})
            if rewrite_payload.get("status") != "success":
                print(
                    f"[想法 #{idea_idx}] ⚠️  本轮改稿未成功，提前结束改进循环: "
                    f"{rewrite_payload.get('status')}"
                )
                break

            if isinstance(round_gate, dict) and round_gate.get("ready"):
                print(
                    f"[想法 #{idea_idx}] ✅ Round gate 已达标，提前结束改进循环 "
                    f"(score={round_gate.get('score')})"
                )
                break

            if isinstance(issue_progress, dict):
                if (
                    issue_progress.get("unresolved_critical_count", 0) == 0
                    and issue_progress.get("persistent_issue_count", 0) == 0
                    and (round_num + 1) >= 2
                ):
                    print(
                        f"[想法 #{idea_idx}] ✅ 问题台账已收敛（无未解决关键问题且无持续问题），提前结束改进循环"
                    )
                    break

            if (
                isinstance(improvement_result.get("issue_ledger"), dict)
                and improvement_result["issue_ledger"].get("issue_count", 0) == 0
            ):
                print(f"[想法 #{idea_idx}] ✅ 本轮未发现可执行问题，提前结束改进循环")
                break

        if self_review_round_records:
            self_review_summary_file = osp.join(exp_dir, "self_review_iteration_summary.json")
            latest_round_gate = self_review_round_records[-1].get("round_gate")
            self_review_summary_payload = {
                "generated_at": datetime.now().isoformat(),
                "rounds": self_review_round_records,
                "rounds_completed": len(self_review_round_records),
                "latest_issue_ledger": previous_issue_ledger,
                "latest_round_gate": latest_round_gate,
                "round_gate_ready": bool(
                    isinstance(latest_round_gate, dict)
                    and latest_round_gate.get("ready")
                ),
                "experiment_todo_round_snapshots": experiment_todo_round_snapshots,
            }
            with open(self_review_summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    self_review_summary_payload,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        if high_quality_mode:
            print(f"\n🧪 [想法 #{idea_idx}] 最终投稿级质量复核")
            final_quality_pass = execute_quality_workflow_with_followups(
                run_high_quality_pass_fn=run_high_quality_pass,
                run_dir=exp_dir,
                paper_type=writeup_type,
                rewrite_model=model_writeup,
                quality_model=quality_model or model_review,
                target_venue=selected_venue,
                quality_preset=quality_preset,
                quality_threshold=quality_threshold,
                rigor_threshold=rigor_threshold,
                max_quality_rewrites=quality_rewrite_rounds,
                require_quality_gate=require_quality_gate,
                min_submission_priority=min_submission_priority,
                max_submission_blockers=max_submission_blockers,
                autonomous_followup_rounds=autonomous_quality_followup_rounds,
                allow_auto_improvement_fallback=quality_policy.get(
                    "allow_auto_improvement_fallback"
                ),
                reject_on_auto_improvement_fallback=bool(
                    quality_policy.get("reject_on_auto_improvement_fallback")
                ),
                resume=False,
                logger=lambda msg: print(f"[想法 #{idea_idx}] {msg}"),
            )
            quality_pass = final_quality_pass
            quality_result = final_quality_pass["quality_result"]
            acceptance = final_quality_pass["acceptance"]
            quality_fallback_event = record_quality_fallback_if_needed(
                exp_dir,
                quality_result,
                producer="run_project.final_quality",
                strict=strict_fallbacks,
            )
            if strict_fallbacks and quality_fallback_event:
                return {
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "stage": "quality_fallback_blocked",
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                    "quality_fallback_policy": quality_policy.get(
                        "quality_fallback_policy"
                    ),
                    "reason": format_strict_fallback_error(
                        quality_fallback_event,
                        workflow_mode=workflow_mode,
                        stage_hint="quality_review",
                    ),
                    "quality_score": quality_result.get("quality_score_after"),
                    "rigor_score": quality_result.get("rigor_score_after"),
                    "submission_priority_score": quality_result.get(
                        "submission_priority_score"
                    ),
                    "blocker_count": quality_result.get("blocker_count"),
                }
            print(
                f"[想法 #{idea_idx}] {final_quality_pass['summary'].replace('High-quality pass: ', '最终质量复核: ')}"
            )

        # ========== 步骤4: 最终审查 ==========
        print(f"\n🎯 [想法 #{idea_idx}] 步骤 4/4: 最终审查")

        final_review_pass = execute_review_suite(
            review_roles=workflow_runtime_plan.final_review_roles,
            paper_dir=exp_dir,
            model_review=model_review,
            review_plan=review_plan,
            create_client_fn=create_client,
            load_paper_fn=load_paper,
            perform_review_fn=perform_review,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
            pdf_path_resolver=find_latest_pdf,
            save_dir=exp_dir,
            text_filename="final_review.json",
            image_filename="final_review_img.json",
            project_root=exp_dir,
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
        pdf_path = final_review_pass["pdf_path"]
        critic_pass = run_independent_critic_pass(
            workflow_runtime_plan=workflow_runtime_plan,
            paper_dir=exp_dir,
            model_review=model_review,
            review_plan=review_plan,
            create_client_fn=create_client,
            load_paper_fn=load_paper,
            perform_review_fn=perform_review,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
            pdf_path_resolver=find_latest_pdf,
            save_dir=Path(exp_dir) / "hostile_critic",
            project_root=exp_dir,
            evidence_refs=[
                "claim_evidence_graph.json",
                "experiment_registry.jsonl",
                "figure_spec.json",
                "manuscript_state.json",
            ],
        )
        if critic_pass.get("ran"):
            print(
                f"[想法 #{idea_idx}] 尖锐 reviewer: roles={critic_pass.get('review_roles_used', [])} "
                f"active={critic_pass.get('active_issue_count', 0)} "
                f"blocking={critic_pass.get('blocking_issue_count', 0)}"
            )
        if critic_pass.get("blocking_issue_count", 0) > 0:
            return {
                "idea_idx": idea_idx,
                "status": "failed",
                "stage": "hostile_critic",
                "workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
                "review_roles_used": sorted(
                    set(workflow_runtime_plan.improvement_review_roles)
                    | set(workflow_runtime_plan.final_review_roles)
                    | set(workflow_runtime_plan.critic_review_roles)
                ),
                "critic_roles_used": list(workflow_runtime_plan.critic_review_roles),
                "critic_active_issue_count": critic_pass.get("active_issue_count"),
                "critic_blocking_issue_count": critic_pass.get("blocking_issue_count"),
                "critic_findings_file": critic_pass.get("critic_findings_file"),
                "reason": "Independent hostile critic reported blocking issues.",
            }
        final_self_review_progress = None
        final_self_review_progress_file = None
        if final_review_pass["found"]:
            if isinstance(previous_issue_ledger, dict):
                final_issue_ledger = build_issue_ledger(
                    text_review=final_review_pass.get("review_text"),
                    img_review=final_review_pass.get("review_img"),
                    max_issues=14,
                    previous_ledger=previous_issue_ledger,
                    target_venue=selected_venue,
                )
                final_self_review_progress = evaluate_issue_progress(
                    previous_issues=list(previous_issue_ledger.get("issues") or []),
                    current_issues=list(final_issue_ledger.get("issues") or []),
                )
                final_self_review_progress_file = osp.join(
                    exp_dir, "self_review_final_progress.json"
                )
                with open(final_self_review_progress_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "generated_at": datetime.now().isoformat(),
                            "final_progress": final_self_review_progress,
                            "final_issue_ledger": final_issue_ledger,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

            # 复制最终PDF到papers目录
            final_pdf_name = f"{idea_name}_final.pdf"
            final_pdf_dst = osp.join(project_dir, "03_papers", final_pdf_name)
            if osp.exists(pdf_path):
                shutil.copy(pdf_path, final_pdf_dst)

        final_round_gate_for_todo = (
            self_review_round_records[-1].get("round_gate")
            if self_review_round_records
            and isinstance(self_review_round_records[-1].get("round_gate"), dict)
            else {}
        )
        final_experiment_todo_snapshot = evaluate_todo_progress_snapshot(
            experiment_todo_payload,
            round_gate=final_round_gate_for_todo,
            issue_progress=(
                final_self_review_progress
                if isinstance(final_self_review_progress, dict)
                else None
            ),
            round_index=(len(self_review_round_records) or None),
        )
        if int((final_experiment_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0:
            experiment_todo_progress_payload = build_todo_progress_payload(
                experiment_todo_payload,
                round_snapshots=experiment_todo_round_snapshots,
                final_snapshot=final_experiment_todo_snapshot,
            )
            experiment_todo_progress_files = save_todo_progress_artifacts(
                exp_dir,
                experiment_todo_progress_payload,
            )
            print(
                f"[想法 #{idea_idx}] TODO闭环: closure={final_experiment_todo_snapshot.get('closure_rate')} "
                f"p0_closure={final_experiment_todo_snapshot.get('p0_closure_rate')} "
                f"unresolved={final_experiment_todo_snapshot.get('counts', {}).get('unresolved_tasks')}"
            )
        if self_review_summary_file and self_review_summary_payload is not None:
            self_review_summary_payload["experiment_todo_progress_file"] = (
                experiment_todo_progress_files.get("json")
            )
            self_review_summary_payload["experiment_todo_progress_markdown_file"] = (
                experiment_todo_progress_files.get("markdown")
            )
            self_review_summary_payload["experiment_todo_final_snapshot"] = (
                final_experiment_todo_snapshot
                if int((final_experiment_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0
                else None
            )
            with open(self_review_summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    self_review_summary_payload,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        if high_quality_mode:
            final_submission_gate = evaluate_final_submission_readiness(
                run_dir=exp_dir,
                quality_result=quality_result,
                require_quality_gate=require_quality_gate,
                min_submission_priority=quality_pass.get("effective_priority_bar"),
                max_submission_blockers=quality_pass.get("effective_blocker_bar"),
                reject_on_auto_improvement_fallback=bool(
                    quality_policy.get("reject_on_auto_improvement_fallback")
                ),
                final_issue_progress=(
                    final_self_review_progress
                    if isinstance(final_self_review_progress, dict)
                    else None
                ),
                final_todo_snapshot=(
                    final_experiment_todo_snapshot
                    if int(
                        (final_experiment_todo_snapshot.get("counts") or {}).get(
                            "total_tasks"
                        )
                        or 0
                    )
                    > 0
                    else None
                ),
            )
            if not final_submission_gate.get("accepted"):
                return {
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "stage": "final_submission_bar",
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
                    "quality_fallback_policy": quality_policy.get(
                        "quality_fallback_policy"
                    ),
                    "acceptance_reasons": final_submission_gate.get("reasons", []),
                    "acceptance_signals": final_submission_gate.get("signals", {}),
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                }

        print(f"\n✅ [想法 #{idea_idx}] 完成! 结果保存在: {exp_dir}")

        return {
            "idea_idx": idea_idx,
            "status": "success",
            "exp_dir": exp_dir,
            "pdf_path": pdf_path if pdf_path and osp.exists(pdf_path) else None,
            "workflow_mode": workflow_mode,
            "template_profile": template_profile,
            "template_capability": template_capability,
            "execution_policy": (research_plan.get("execution_policy") or {}).get("policy_name"),
            "evidence_pressure": (research_plan.get("execution_policy") or {}).get("evidence_pressure"),
            "quality_fallback_policy": (research_plan.get("execution_policy") or {}).get(
                "quality_fallback_policy"
            ),
            "review_roles_improvement": list(workflow_runtime_plan.improvement_review_roles),
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
            "pipeline_manifest": str(Path(exp_dir) / "pipeline_manifest.json"),
            "idea_card_id": idea_card.get("idea_id"),
            "research_plan_file": str(Path(exp_dir) / "research_plan.json"),
            "claim_evidence_graph_file": str(Path(exp_dir) / "claim_evidence_graph.json"),
            "experiment_registry_file": str(Path(exp_dir) / "experiment_registry.jsonl"),
            "figure_spec_file": str(Path(exp_dir) / "figure_spec.json"),
            "manuscript_state_file": str(Path(exp_dir) / "manuscript_state.json"),
            "review_state_file": str(Path(exp_dir) / "review_state.json"),
            "repair_plan_file": str(Path(exp_dir) / "repair_plan.json"),
            "self_evolution_file": str(Path(exp_dir) / "self_evolution.json"),
            "stage_standards_file": str(Path(exp_dir) / "stage_standards.json"),
            "quality_score": quality_result.get("quality_score_after") if high_quality_mode else None,
            "rigor_score": quality_result.get("rigor_score_after") if high_quality_mode else None,
            "claim_support_score": quality_result.get("claim_support_after") if high_quality_mode else None,
            "quality_gate_passed": quality_result.get("quality_gate_passed") if high_quality_mode else None,
            "submission_priority_score": quality_result.get("submission_priority_score") if high_quality_mode else None,
            "submission_priority_tier": quality_result.get("submission_priority_tier") if high_quality_mode else None,
            "blocker_count": quality_result.get("blocker_count") if high_quality_mode else None,
            "submission_acceptance_passed": (
                final_submission_gate.get("accepted") if high_quality_mode else None
            ),
            "submission_acceptance_reasons": (
                final_submission_gate.get("reasons", []) if high_quality_mode else []
            ),
            "quality_acceptance_passed": (
                acceptance.get("accepted") if high_quality_mode else None
            ),
            "auto_improvement_fallback_used": quality_result.get("auto_improvement_fallback_used") if high_quality_mode else None,
            "submission_package_file": quality_result.get("submission_package_file") if high_quality_mode else None,
            "submission_dashboard_file": quality_result.get("submission_dashboard_file") if high_quality_mode else None,
            "autonomous_followup_rounds_run": quality_result.get("autonomous_followup_rounds_run") if high_quality_mode else None,
            "writing_profile": writing_profile,
            "writing_audit_rounds": writing_audit_rounds,
            "strict_writing_guardrails": strict_writing_guardrails,
            "guardrail_repair_rounds": guardrail_repair_rounds,
            "self_review_rounds_completed": len(self_review_round_records),
            "self_review_summary_file": self_review_summary_file,
            "self_review_final_progress_file": final_self_review_progress_file,
            "self_review_unresolved_critical": (
                final_self_review_progress.get("unresolved_critical_count")
                if isinstance(final_self_review_progress, dict)
                else None
            ),
            "self_review_round_gate_ready": (
                bool(
                    isinstance(self_review_round_records[-1].get("round_gate"), dict)
                    and self_review_round_records[-1]["round_gate"].get("ready")
                )
                if self_review_round_records
                else None
            ),
            "experiment_todo_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("total_tasks")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_p0_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("p0_total")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_closed_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("closed_tasks")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_unresolved_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("unresolved_tasks")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_closure_rate": (
                final_experiment_todo_snapshot.get("closure_rate")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_p0_closure_rate": (
                final_experiment_todo_snapshot.get("p0_closure_rate")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_progress_file": experiment_todo_progress_files.get("json"),
            "experiment_todo_progress_markdown_file": experiment_todo_progress_files.get(
                "markdown"
            ),
        }

    except Exception as e:
        print(f"\n❌ [想法 #{idea_idx}] 处理失败: {e}")
        traceback.print_exc()
        failure_runtime_plan = build_workflow_runtime_plan(
            workflow_mode,
            submission_mode=(template_profile == "template_first"),
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
        return {
            "idea_idx": idea_idx,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "writing_profile": writing_profile,
            "writing_audit_rounds": writing_audit_rounds,
            "strict_writing_guardrails": strict_writing_guardrails,
            "guardrail_repair_rounds": guardrail_repair_rounds,
            "workflow_mode": workflow_mode,
            "template_profile": template_profile,
            "template_capability": template_capability,
            "execution_policy": (research_plan.get("execution_policy") or {}).get("policy_name")
            if "research_plan" in locals()
            else None,
            "review_roles_improvement": list(failure_runtime_plan.improvement_review_roles),
            "review_roles_final": list(failure_runtime_plan.final_review_roles),
            "review_roles_used": sorted(
                set(failure_runtime_plan.improvement_review_roles)
                | set(failure_runtime_plan.final_review_roles)
                | set(failure_runtime_plan.critic_review_roles)
            ),
            "critic_roles_used": list(failure_runtime_plan.critic_review_roles),
        }


def run_parallel_experiments(project_dir, idea_json, num_workers, idea_indices, **kwargs):
    """并行运行多个实验"""
    print("\n" + "=" * 80)
    print(f"🚀 并行运行 {len(idea_indices)} 个实验 (使用 {num_workers} 个worker)")
    print("=" * 80)

    # 加载想法
    with open(idea_json, "r") as f:
        ideas = json.load(f)

    # 准备参数
    args_list = []
    for idx in idea_indices:
        if idx >= len(ideas):
            print(f"⚠️  想法索引 {idx} 超出范围，跳过")
            continue
        args_list.append((
            project_dir,
            kwargs["output_root"],
            idx,
            ideas[idx],
            kwargs["bfts_config"],
            kwargs["model_writeup"],
            kwargs["model_citation"],
            kwargs["model_review"],
            kwargs["model_agg_plots"],
            kwargs["model_writeup_small"],
            kwargs["num_cite_rounds"],
            kwargs["writeup_retries"],
            kwargs["writeup_type"],
            kwargs["improvement_rounds"],
            kwargs["review_reflections"],
            kwargs["review_ensemble"],
            kwargs["review_fewshot"],
            kwargs["review_temperature"],
            kwargs["review_strategy"],
            kwargs["high_quality_mode"],
            kwargs["quality_preset"],
            kwargs["quality_model"],
            kwargs["target_venue"],
            kwargs["quality_threshold"],
            kwargs["rigor_threshold"],
            kwargs["quality_rewrite_rounds"],
            kwargs["autonomous_quality_followup_rounds"],
            kwargs["require_quality_gate"],
            kwargs["min_submission_priority"],
            kwargs["max_submission_blockers"],
            kwargs["writing_profile"],
            kwargs["writing_audit_rounds"],
            kwargs["strict_writing_guardrails"],
            kwargs["guardrail_repair_rounds"],
            kwargs["workflow_mode"],
            kwargs["template_profile"],
            kwargs["template_capability"],
            kwargs["strict_fallbacks"],
        ))

    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_idea, args): args[2] for args in args_list}

        for future in as_completed(futures):
            idea_idx = futures[future]
            try:
                result = future.result()
                results.append(result)

                # 保存进度
                progress_file = osp.join(project_dir, "04_logs", "progress.json")
                with open(progress_file, "w") as f:
                    json.dump({
                        "completed": len(results),
                        "total": len(args_list),
                        "results": results,
                    }, f, indent=4)

            except Exception as e:
                print(f"❌ 想法 #{idea_idx} 执行出错: {e}")
                traceback.print_exc()
                results.append({
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "error": str(e),
                })

    return results


def save_project_summary(project_dir: str, results: list[dict]) -> tuple[str, str, dict]:
    logs_dir = Path(project_dir) / "04_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    quality_scores = [result.get("quality_score") for result in results if isinstance(result.get("quality_score"), (int, float))]
    rigor_scores = [result.get("rigor_score") for result in results if isinstance(result.get("rigor_score"), (int, float))]
    priority_scores = [result.get("submission_priority_score") for result in results if isinstance(result.get("submission_priority_score"), (int, float))]
    followup_rounds = [
        result.get("autonomous_followup_rounds_run")
        for result in results
        if isinstance(result.get("autonomous_followup_rounds_run"), int)
    ]
    gate_passed = sum(1 for result in results if result.get("quality_gate_passed") is True)
    submission_accepted = sum(1 for result in results if result.get("submission_acceptance_passed") is True)
    guardrail_passed = sum(
        1 for result in results if result_passed_writeup_guardrails(result)
    )
    failed = [result for result in results if result.get("status") != "success"]
    guardrail_blocked = sum(
        1 for result in failed if result.get("guardrail_blocking") is True
    )
    guardrail_reason_counts: dict[str, int] = {}
    workflow_mode_counts: dict[str, int] = {}
    template_profile_counts: dict[str, int] = {}
    template_capability_counts: dict[str, int] = {}
    execution_policy_counts: dict[str, int] = {}
    review_role_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    for result in failed:
        reasons = result.get("guardrail_blocking_reasons", [])
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            key = str(reason).strip()
            if not key:
                continue
            guardrail_reason_counts[key] = guardrail_reason_counts.get(key, 0) + 1
    for result in results:
        workflow_mode = str(result.get("workflow_mode") or "unknown")
        template_profile = str(result.get("template_profile") or "unknown")
        template_capability = str(result.get("template_capability") or "unknown")
        workflow_mode_counts[workflow_mode] = (
            workflow_mode_counts.get(workflow_mode, 0) + 1
        )
        template_profile_counts[template_profile] = (
            template_profile_counts.get(template_profile, 0) + 1
        )
        template_capability_counts[template_capability] = (
            template_capability_counts.get(template_capability, 0) + 1
        )
        execution_policy = str(result.get("execution_policy") or "unknown")
        execution_policy_counts[execution_policy] = (
            execution_policy_counts.get(execution_policy, 0) + 1
        )
        review_roles = result.get("review_roles_used") or []
        if isinstance(review_roles, list):
            for role in review_roles:
                key = str(role).strip()
                if not key:
                    continue
                review_role_counts[key] = review_role_counts.get(key, 0) + 1
        if result.get("status") != "success":
            stage = str(result.get("stage") or "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    summary = {
        "project_dir": project_dir,
        "generated_at": datetime.now().isoformat(),
        "total": len(results),
        "success": sum(1 for result in results if result.get("status") == "success"),
        "failed": sum(1 for result in results if result.get("status") != "success"),
        "quality_summary": {
            "avg_quality_score": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
            "avg_rigor_score": (sum(rigor_scores) / len(rigor_scores)) if rigor_scores else None,
            "avg_submission_priority_score": (sum(priority_scores) / len(priority_scores)) if priority_scores else None,
            "avg_autonomous_followup_rounds": (sum(followup_rounds) / len(followup_rounds)) if followup_rounds else None,
            "gate_passed": gate_passed,
            "submission_accepted": submission_accepted,
            "guardrail_passed": guardrail_passed,
            "guardrail_blocked": guardrail_blocked,
            "guardrail_blocking_reasons": guardrail_reason_counts,
        },
        "pipeline_summary": {
            "workflow_mode_counts": workflow_mode_counts,
            "template_profile_counts": template_profile_counts,
            "template_capability_counts": template_capability_counts,
            "execution_policy_counts": execution_policy_counts,
            "review_role_counts": review_role_counts,
            "failed_stage_counts": stage_counts,
        },
        "results": results,
    }

    summary_file = logs_dir / "project_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    shortlist_pool = [r for r in results if r.get("status") == "success"] or results
    ranked = sorted(
        shortlist_pool,
        key=lambda result: (
            result.get("submission_acceptance_passed") is True,
            result.get("submission_priority_score") if isinstance(result.get("submission_priority_score"), (int, float)) else -1,
            -(result.get("blocker_count") if isinstance(result.get("blocker_count"), int) else 999),
            result.get("quality_gate_passed") is True,
            result.get("quality_score") if isinstance(result.get("quality_score"), (int, float)) else -1,
            result.get("rigor_score") if isinstance(result.get("rigor_score"), (int, float)) else -1,
        ),
        reverse=True,
    )
    shortlist_file = logs_dir / "submission_shortlist.md"
    lines = ["# Project Submission Shortlist", ""]
    for result in ranked[:5]:
        lines.extend([
            f"## idea #{result.get('idea_idx')}",
            f"- Status: {result.get('status')}",
            f"- PDF: {result.get('pdf_path')}",
            f"- Submission Priority: {result.get('submission_priority_score')} ({result.get('submission_priority_tier')})",
            f"- Blockers: {result.get('blocker_count')}",
            f"- Quality: {result.get('quality_score')}",
            f"- Rigor: {result.get('rigor_score')}",
            f"- Gate Passed: {result.get('quality_gate_passed')}",
            f"- Accepted: {result.get('submission_acceptance_passed')}",
            f"- Autonomous Followups: {result.get('autonomous_followup_rounds_run')}",
            f"- Submission Package: {result.get('submission_package_file')}",
            "",
        ])
    shortlist_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(summary_file), str(shortlist_file), summary


def main():
    require_login("项目管理操作(run_project)")

    parser = argparse.ArgumentParser(
        description="AI Scientist 项目管理器 - 并行处理多个论文",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 生成3个想法并并行写成论文:
   python run_project.py my_project --topic topic.md --num-ideas 3 --parallel

2. 并行处理已有想法的前2个:
   python run_project.py my_project --ideas ideas.json --idea-indices 0,1 --parallel

3. 自动改进2轮:
   python run_project.py my_project --topic topic.md --improvement-rounds 2
        """,
    )

    # 项目设置
    parser.add_argument(
        "project_dir",
        type=str,
        help="项目目录路径（相对路径会解析到 --output-root/projects 下）",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(resolve_output_path()),
        help="当 project_dir 为相对路径时，作为统一输出根目录",
    )

    # 想法生成
    parser.add_argument("--topic", type=str, help="主题描述文件")
    parser.add_argument("--ideas", type=str, help="已有想法JSON文件")
    parser.add_argument("--model-ideation", type=str, default="glm-4-flash")
    parser.add_argument("--num-ideas", type=int, default=3, help="生成的想法数量")
    parser.add_argument("--num-reflections", type=int, default=5)

    # 并行处理
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行处理多个想法",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="并行worker数量",
    )
    parser.add_argument(
        "--idea-indices",
        type=str,
        help="要处理的想法索引 (逗号分隔)，如: 0,1,2",
    )
    parser.add_argument("--rank-ideas", action="store_true", help="先对 idea 排序再选择")
    parser.add_argument("--top-k-ideas", type=int, default=None, help="只处理评分最高的前 K 个 idea")
    parser.add_argument("--idea-rank-model", type=str, default=None, help="用于 idea 排序的模型")
    parser.add_argument("--submission-mode", action="store_true")
    parser.add_argument("--fallback-ranked-ideas", action="store_true")
    parser.add_argument("--breakthrough-mode", action="store_true")
    parser.add_argument(
        "--workflow-mode",
        type=str,
        choices=list_workflow_modes(),
        default="adaptive",
        help="研究编排模式：兼容经典模板流、agentic tree、program-driven、writing-studio、review-board。",
    )
    parser.add_argument(
        "--override-strict-fallbacks",
        action="store_true",
        help="禁用严格兜底拦截（默认投稿/高质量/程序驱动模式会在出现 fallback 时终止）。",
    )

    # 改进设置
    parser.add_argument(
        "--improvement-rounds",
        type=int,
        default=1,
        help="每篇论文的反思改进轮数",
    )

    # 跳过步骤
    parser.add_argument("--skip-ideation", action="store_true")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument(
        "--bfts-config",
        type=str,
        default="bfts_config.yaml",
        help="BFTS实验配置文件路径 (控制搜索深度、seed、并行度、超时等)",
    )

    # 模型配置
    parser.add_argument("--model-agg-plots", type=str, default="glm-4-flash")
    parser.add_argument("--model-writeup", type=str, default="glm-4-plus")
    parser.add_argument("--model-writeup-small", type=str, default="glm-4-air")
    parser.add_argument("--model-citation", type=str, default="glm-4-air")
    parser.add_argument("--model-review", type=str, default="glm-4-plus")

    # 其他参数
    parser.add_argument("--num-cite-rounds", type=int, default=15)
    parser.add_argument("--writeup-retries", type=int, default=3)
    parser.add_argument("--writeup-type", type=str, default="icbinb", choices=["normal", "icbinb", "journal", "extended"])
    parser.add_argument("--review-reflections", type=int, default=1)
    parser.add_argument("--review-ensemble", type=int, default=1)
    parser.add_argument("--review-fewshot", type=int, default=1)
    parser.add_argument("--review-temperature", type=float, default=0.75)
    parser.add_argument(
        "--review-strategy",
        type=str,
        choices=["standard", "fast", "depth", "neurips", "iclr", "cvpr", "journal", "nature"],
        default=None,
    )
    parser.add_argument("--high-quality-mode", action="store_true")
    parser.add_argument("--quality-preset", choices=["balanced", "high", "publishable"], default="balanced")
    parser.add_argument("--quality-model", type=str, default=None)
    parser.add_argument("--target-venue", type=str, choices=["neurips", "iclr", "cvpr", "journal", "nature"], default=None)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--rigor-threshold", type=float, default=None)
    parser.add_argument("--quality-rewrite-rounds", type=int, default=None)
    parser.add_argument("--autonomous-quality-followup-rounds", type=int, default=0)
    parser.add_argument("--min-submission-priority", type=float, default=None)
    parser.add_argument("--max-submission-blockers", type=int, default=None)
    parser.add_argument("--require-quality-gate", action="store_true")
    parser.add_argument("--auto-adjust-paper-type", action="store_true")
    parser.add_argument(
        "--writing-profile",
        type=str,
        choices=list_writing_profiles(),
        default=resolve_writing_profile_env(
            invalid_profile_logger=lambda exc, raw: print(
                "⚠️  忽略无效的 AI_SCIENTIST_WRITING_PROFILE="
                f"{raw!r}，回退为 {DEFAULT_WRITING_PROFILE}"
            )
        ),
        help="写作提示词 profile（影响写作约束与反思自检）",
    )
    parser.add_argument(
        "--writing-audit-rounds",
        type=int,
        default=0,
        help="写作反思阶段追加的结构化写作审计轮数",
    )
    parser.add_argument(
        "--strict-writing-guardrails",
        action="store_true",
        help="启用严格写作守护：最终稿若存在关键引用/章节缺口则判定失败",
    )
    parser.add_argument(
        "--guardrail-repair-rounds",
        type=int,
        default=1,
        help="严格写作守护失败前自动尝试的修复轮数",
    )
    parser.add_argument(
        "--disable-hostile-critic",
        action="store_true",
        help="基准 ablation: 关闭独立 hostile critic 通道。",
    )
    parser.add_argument(
        "--disable-owner-aware-repair",
        action="store_true",
        help="基准 ablation: 关闭 owner-aware reviewer repair routing。",
    )

    args = parser.parse_args()
    normalize_project_workflow_args(
        args,
        invalid_profile_logger=lambda exc: print(
            f"⚠️  无效 writing profile: {exc}，回退为 {DEFAULT_WRITING_PROFILE}"
        ),
    )
    ablation_config = configure_benchmark_ablation_env(
        disable_hostile_critic=bool(args.disable_hostile_critic),
        disable_owner_aware_repair=bool(args.disable_owner_aware_repair),
    )
    if ablation_config["disable_hostile_critic"]:
        print("🧪 Benchmark ablation active: hostile critic disabled.")
    if ablation_config["disable_owner_aware_repair"]:
        print("🧪 Benchmark ablation active: owner-aware repair routing disabled.")
    runtime = initialize_runtime(
        source_file=__file__,
        output_root=args.output_root,
        ensure_dirs=True,
        apply_cache=True,
    )
    args.output_root = str(runtime.research_root)

    strict_fallbacks = should_enforce_strict_fallbacks(
        args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        high_quality_mode=bool(args.high_quality_mode),
        target_venue=args.target_venue,
    )
    if args.override_strict_fallbacks and strict_fallbacks:
        print("⚠️  override-strict-fallbacks: 严格兜底拦截已禁用，本次运行将继续记录但不阻断 fallback。")
        strict_fallbacks = False
    elif strict_fallbacks:
        print("🛡️  启用严格兜底策略：出现 fallback 将直接阻断流程（可通过 --override-strict-fallbacks 临时放宽）。")

    # 按实际模型检查 provider 凭证
    require_model_credentials(_collect_requested_models(args))

    # 创建项目结构
    dirs = create_project_structure(args.project_dir, output_root=args.output_root)
    args.project_dir = str(dirs["root"])

    args.writeup_type = resolve_paper_type_for_venue(
        args.writeup_type,
        args.target_venue,
        auto_adjust=args.auto_adjust_paper_type,
        warning_template="⚠️  警告: writeup_type '{paper_type}' 与目标 venue '{target_venue}' 可能不够匹配",
        adjusted_template="✅ 自动调整 writeup_type -> {adjusted}",
    )

    # 确定想法文件
    idea_json = None

    if not args.skip_ideation:
        if args.topic:
            idea_json = generate_ideas(
                args.project_dir,
                args.topic,
                args.model_ideation,
                args.num_ideas,
                args.num_reflections,
            )
        elif args.ideas:
            idea_json = osp.join(args.project_dir, "01_ideas", "ideas.json")
            shutil.copy(args.ideas, idea_json)
        else:
            print("❌ 错误: 需要指定 --topic 或 --ideas")
            sys.exit(1)
    else:
        idea_json = osp.join(args.project_dir, "01_ideas", "generated_ideas.json")
        if not osp.exists(idea_json):
            idea_json = osp.join(args.project_dir, "01_ideas", "ideas.json")
        if not osp.exists(idea_json):
            print(f"❌ 错误: 未找到想法文件")
            sys.exit(1)

    # 加载想法
    with open(idea_json, "r") as f:
        ideas = json.load(f)
    project_idea_cards = _write_project_pipeline_seed_artifacts(
        project_dir=args.project_dir,
        ideas=ideas,
        target_venue=args.target_venue,
        workflow_mode=args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        breakthrough_mode=bool(args.breakthrough_mode),
        high_quality_mode=bool(args.high_quality_mode),
    )
    if project_idea_cards:
        print(
            "🧭 已生成结构化 idea cards / research_program，用于后续 planning、实验追踪和看板消费"
        )

    requested_indices = (
        [int(x.strip()) for x in args.idea_indices.split(",")]
        if args.idea_indices
        else None
    )
    default_indices = list(range(len(ideas))) if args.parallel else [0]
    idea_indices, rankings = select_ranked_idea_candidates(
        ideas,
        ranking_enabled=args.rank_ideas,
        ranking_model=args.idea_rank_model or args.model_writeup,
        target_venue=args.target_venue,
        prioritize_breakthrough=args.breakthrough_mode,
        research_root=Path(args.output_root).expanduser(),
        ranking_output_path=osp.join(args.project_dir, "04_logs", "idea_rankings.json"),
        requested_indices=requested_indices,
        default_indices=default_indices,
        fallback_to_ranked=args.fallback_ranked_ideas,
        use_ranked_all=args.parallel,
        limit=args.top_k_ideas if args.rank_ideas else None,
    )
    if rankings:
        ranking_event = record_ranking_fallbacks(
            args.project_dir,
            rankings,
            producer="run_project.idea_ranking",
            strict=strict_fallbacks,
        )
        if strict_fallbacks and ranking_event:
            raise RuntimeError(
                format_strict_fallback_error(
                    ranking_event,
                    workflow_mode=args.workflow_mode,
                    stage_hint="idea_ranking",
                )
            )
        print("\n🏅 Idea ranking complete")
        for item in rankings[: min(5, len(rankings))]:
            print(
                f"  - idea #{item['idea_idx']} ranking_score={item.get('ranking_score')} total={item.get('total_score')} "
                f"name={item.get('idea_name')}"
            )

    # 运行实验
    if not args.skip_experiment:
        workflow_spec, _, template_profile, template_capability = _resolve_workflow_strategy(
            workflow_mode=args.workflow_mode,
            submission_mode=bool(args.submission_mode),
            breakthrough_mode=bool(args.breakthrough_mode),
            high_quality_mode=bool(args.high_quality_mode),
            target_venue=args.target_venue,
        )
        kwargs = {
            "output_root": args.output_root,
            "bfts_config": args.bfts_config,
            "model_writeup": args.model_writeup,
            "model_citation": args.model_citation,
            "model_review": args.model_review,
            "model_agg_plots": args.model_agg_plots,
            "model_writeup_small": args.model_writeup_small,
            "num_cite_rounds": args.num_cite_rounds,
            "writeup_retries": args.writeup_retries,
            "writeup_type": args.writeup_type,
            "improvement_rounds": args.improvement_rounds,
            "review_reflections": args.review_reflections,
            "review_ensemble": args.review_ensemble,
            "review_fewshot": args.review_fewshot,
            "review_temperature": args.review_temperature,
            "review_strategy": args.review_strategy,
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
            "writing_profile": args.writing_profile,
            "writing_audit_rounds": args.writing_audit_rounds,
            "strict_writing_guardrails": args.strict_writing_guardrails,
            "guardrail_repair_rounds": args.guardrail_repair_rounds,
            "workflow_mode": workflow_spec.name,
            "template_profile": template_profile,
            "template_capability": template_capability,
            "strict_fallbacks": strict_fallbacks,
        }
        results = []

        if args.parallel and len(idea_indices) > 1:
            # 并行处理
            results = run_parallel_experiments(
                args.project_dir,
                idea_json,
                args.num_workers,
                idea_indices,
                **kwargs,
            )

            # 打印结果摘要
            print("\n" + "=" * 80)
            print("📊 执行结果摘要")
            print("=" * 80)
            for r in results:
                status_icon = "✅" if r["status"] == "success" else "❌"
                print(f"{status_icon} 想法 #{r['idea_idx']}: {r['status']}")
                if r["status"] == "success" and "pdf_path" in r:
                    print(f"   📄 {r['pdf_path']}")

        else:
            # 串行处理
            for idx in idea_indices:
                if idx >= len(ideas):
                    continue

                # 构建参数元组
                process_args = (
                    args.project_dir,
                    args.output_root,
                    idx,
                    ideas[idx],
                    kwargs["bfts_config"],
                    kwargs["model_writeup"],
                    kwargs["model_citation"],
                    kwargs["model_review"],
                    kwargs["model_agg_plots"],
                    kwargs["model_writeup_small"],
                    kwargs["num_cite_rounds"],
                    kwargs["writeup_retries"],
                    kwargs["writeup_type"],
                    kwargs["improvement_rounds"],
                    kwargs["review_reflections"],
                    kwargs["review_ensemble"],
                    kwargs["review_fewshot"],
                    kwargs["review_temperature"],
                    kwargs["review_strategy"],
                    kwargs["high_quality_mode"],
                    kwargs["quality_preset"],
                    kwargs["quality_model"],
                    kwargs["target_venue"],
                    kwargs["quality_threshold"],
                    kwargs["rigor_threshold"],
                    kwargs["quality_rewrite_rounds"],
                    kwargs["autonomous_quality_followup_rounds"],
                    kwargs["require_quality_gate"],
                    kwargs["min_submission_priority"],
                    kwargs["max_submission_blockers"],
                    kwargs["writing_profile"],
                    kwargs["writing_audit_rounds"],
                    kwargs["strict_writing_guardrails"],
                    kwargs["guardrail_repair_rounds"],
                    kwargs["workflow_mode"],
                    kwargs["template_profile"],
                    kwargs["template_capability"],
                    kwargs["strict_fallbacks"],
                )

                result = process_single_idea(process_args)
                results.append(result)

                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"\n{status_icon} 想法 #{idx}: {result['status']}")

        summary_file, shortlist_file, summary_payload = save_project_summary(
            args.project_dir, results
        )
        save_contract_artifact(
            args.project_dir,
            "pipeline_manifest",
            load_contract_artifact(args.project_dir, "pipeline_manifest", default={}),
            producer="run_project.summary",
            warnings=[],
        )
        print(f"\n📋 项目总结: {summary_file}")
        print(f"📝 投稿 shortlist: {shortlist_file}")
        summary_quality = summary_payload.get("quality_summary", {})
        print(f"🛡️ 写作守护通过: {summary_quality.get('guardrail_passed', 0)}")
        print(f"🛡️ 写作守护拦截: {summary_quality.get('guardrail_blocked', 0)}")
        reason_counts = summary_quality.get("guardrail_blocking_reasons", {}) or {}
        if reason_counts:
            print("🧾 写作守护拦截原因:")
            for reason, count in sorted(
                reason_counts.items(), key=lambda item: (-item[1], item[0])
            ):
                print(f"   - {reason}: {count}")

        constraints_active = (
            args.require_quality_gate
            or args.min_submission_priority is not None
            or args.max_submission_blockers is not None
        )
        if constraints_active:
            passed = any(result.get("submission_acceptance_passed") is True for result in results)
            if not passed:
                print("❌ 没有任何项目论文达到当前投稿接受标准，任务视为失败")
                sys.exit(1)
        if args.strict_writing_guardrails or args.high_quality_mode:
            any_guardrail_pass = any(
                result_passed_writeup_guardrails(result) for result in results
            )
            if not any_guardrail_pass:
                print("❌ 严格写作守护开启，但没有任何论文通过写作守护检查，任务视为失败")
                sys.exit(1)

    print("\n" + "=" * 80)
    print("🎉 项目完成!")
    print(f"📁 项目目录: {args.project_dir}")
    print(f"📄 论文位置: {osp.join(args.project_dir, '03_papers')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
