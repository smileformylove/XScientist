from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from ai_scientist.utils.deferred_imports import load_module_attr
from ai_scientist.utils.high_quality_pipeline import (
    resolve_submission_acceptance_settings,
    run_high_quality_pass,
)
from ai_scientist.utils.pipeline_helpers import (
    find_best_pdf_path,
    find_latest_bfts_run_dir,
    save_token_tracker,
)
from ai_scientist.utils.quality_workflow import (
    evaluate_final_submission_readiness,
    execute_quality_workflow_with_followups,
)
from ai_scientist.utils.review_workflow import build_review_execution_plan
from ai_scientist.utils.writeup_workflow import build_writeup_execution_plan
from ai_scientist.utils.fallback_audit import (
    format_strict_fallback_error,
    record_quality_fallback_if_needed,
)
from ai_scientist.utils.run_index import is_stage_complete, mark_stage_complete
from ai_scientist.utils.experiment_report import write_experiment_report
from ai_scientist.utils.workflow_runtime import (
    build_workflow_runtime_plan,
    execute_review_suite,
)
from ai_scientist.utils.critic_workflow import run_independent_critic_pass
from ai_scientist.utils.workflow_execution_policy import (
    build_workflow_execution_policy,
)


def create_client(*args, **kwargs):
    return load_module_attr("ai_scientist.llm", "create_client")(*args, **kwargs)


def gather_citations(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_icbinb_writeup", "gather_citations")(
        *args, **kwargs
    )


def perform_icbinb_writeup(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_icbinb_writeup", "perform_writeup")(
        *args, **kwargs
    )


def load_paper(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "load_paper")(
        *args, **kwargs
    )


def perform_review(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "perform_review")(
        *args, **kwargs
    )


def aggregate_plots(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_plotting", "aggregate_plots")(
        *args, **kwargs
    )


def perform_imgs_cap_ref_review(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.perform_vlm_review",
        "perform_imgs_cap_ref_review",
    )(*args, **kwargs)


def perform_writeup(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_writeup", "perform_writeup")(
        *args, **kwargs
    )


def edit_bfts_config_file(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.treesearch.bfts_utils",
        "edit_bfts_config_file",
    )(*args, **kwargs)


def idea_to_markdown(*args, **kwargs):
    return load_module_attr("ai_scientist.treesearch.bfts_utils", "idea_to_markdown")(
        *args, **kwargs
    )


def perform_experiments_bfts(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
        "perform_experiments_bfts",
    )(*args, **kwargs)


def prepare_idea_artifacts(
    ideas: list[dict],
    idea_idx: int,
    load_ideas_path: str,
    idea_dir: str | Path,
    *,
    load_code: bool = False,
    add_dataset_ref: bool = False,
    logger: Callable[[str], None] = print,
    echo_added_code: bool = False,
) -> tuple[dict, str]:
    idea_dir = Path(idea_dir)
    idea = ideas[idea_idx]
    idea_path_md = idea_dir / "idea.md"

    code = None
    code_path = None
    if load_code:
        code_path = load_ideas_path.rsplit(".", 1)[0] + ".py"
        if os.path.exists(code_path):
            with open(code_path, "r") as f:
                code = f.read()
        else:
            logger(f"Warning: Code file {code_path} not found")
            code_path = None

    idea_to_markdown(idea, str(idea_path_md), code_path)

    dataset_ref_code = None
    if add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()
        else:
            logger(f"Warning: Dataset reference file {dataset_ref_path} not found")

    if dataset_ref_code is not None and code is not None:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code is not None:
        added_code = dataset_ref_code
    else:
        added_code = code

    if echo_added_code and added_code is not None:
        logger(added_code)

    if added_code is not None:
        idea["Code"] = added_code

    idea_path_json = idea_dir / "idea.json"
    with open(idea_path_json, "w") as f:
        json.dump(idea, f, indent=4)

    mark_stage_complete(
        idea_dir,
        "prepare",
        artifacts={
            "idea_json": str(idea_path_json),
            "idea_markdown": str(idea_path_md),
        },
    )

    return idea, str(idea_path_json)


def run_experiment_phase(
    idea_dir: str | Path,
    idea_path_json: str | Path,
    model_agg_plots: str,
    *,
    config_path: str = "bfts_config.yaml",
    resume: bool = True,
    logger: Callable[[str], None] = print,
) -> str:
    idea_dir = Path(idea_dir)
    if resume and is_stage_complete(idea_dir, "experiment") and (idea_dir / "logs").exists():
        logger(f"Resume: skipping experiment phase for {idea_dir}")
        report_json = idea_dir / "experiment_report.json"
        report_md = idea_dir / "experiment_report.md"
        if (not report_json.exists()) or (not report_md.exists()):
            try:
                write_experiment_report(idea_dir)
            except Exception:
                pass
        return str(idea_dir / "logs")

    idea_config_path = edit_bfts_config_file(
        config_path,
        str(idea_dir),
        str(idea_path_json),
    )

    perform_experiments_bfts(idea_config_path)

    latest_run_dir = find_latest_bfts_run_dir(idea_dir, logs_subdir="logs")
    experiment_results_dir = (
        latest_run_dir / "experiment_results"
        if latest_run_dir is not None
        else idea_dir / "logs/0-run/experiment_results"
    )
    local_results_dir = idea_dir / "experiment_results"
    if experiment_results_dir.exists():
        shutil.copytree(experiment_results_dir, local_results_dir, dirs_exist_ok=True)

    aggregate_plots(base_folder=str(idea_dir), model=model_agg_plots)

    if local_results_dir.exists():
        shutil.rmtree(local_results_dir)

    report_json, report_md = write_experiment_report(idea_dir)
    save_token_tracker(idea_dir)
    mark_stage_complete(
        idea_dir,
        "experiment",
        artifacts={
            "logs_dir": str(idea_dir / "logs"),
            "config_path": str(idea_config_path),
            "experiment_report_json": str(report_json),
            "experiment_report_md": str(report_md),
        },
    )
    return idea_config_path


def run_writeup_phase(
    idea_dir: str | Path,
    *,
    writeup_type: str,
    writeup_retries: int,
    num_cite_rounds: int,
    model_citation: str,
    model_writeup_small: str,
    model_writeup: str,
    high_quality_mode: bool = False,
    quality_preset: str = "balanced",
    quality_model: Optional[str] = None,
    target_venue: Optional[str] = None,
    quality_threshold: Optional[float] = None,
    rigor_threshold: Optional[float] = None,
    max_quality_rewrites: Optional[int] = None,
    autonomous_quality_followup_rounds: int = 0,
    require_quality_gate: bool = False,
    min_submission_priority: Optional[float] = None,
    max_submission_blockers: Optional[int] = None,
    writing_profile: str = "default",
    writing_audit_rounds: int = 0,
    strict_guardrails: bool = False,
    guardrail_repair_rounds: int = 1,
    workflow_mode: Optional[str] = None,
    submission_mode: bool = False,
    strict_fallbacks: bool = False,
    research_root: str | Path | None = None,
    resume: bool = True,
    logger: Callable[[str], None] = print,
) -> dict[str, Any]:
    existing_pdf = find_best_pdf_path(idea_dir, prefer_reflections=True)
    quality_result: dict[str, Any] = {}
    acceptance: dict[str, Any] = {}
    strategy_feedback: dict[str, Any] = {}
    reused_writeup = bool(
        resume and is_stage_complete(idea_dir, "writeup") and existing_pdf is not None
    )
    if reused_writeup:
        logger(f"Resume: reusing writeup artifacts for {idea_dir}")

    writeup_plan = build_writeup_execution_plan(
        writeup_type,
        num_cite_rounds=num_cite_rounds,
        writeup_retries=writeup_retries,
        target_venue=target_venue,
        high_quality_mode=high_quality_mode,
        research_root=research_root,
    )
    resolved_target_venue = writeup_plan["target_venue"]
    if high_quality_mode:
        venue = resolved_target_venue
        strategy_feedback = writeup_plan["strategy_feedback"]
        num_cite_rounds = writeup_plan["num_cite_rounds"]
        writeup_retries = writeup_plan["writeup_retries"]
        effective_priority_bar, effective_blocker_bar = resolve_submission_acceptance_settings(
            venue,
            min_submission_priority=min_submission_priority,
            max_submission_blockers=max_submission_blockers,
        )
        logger(
            f"High-quality mode venue: {venue}; budgets: num_cite_rounds={num_cite_rounds}, "
            f"writeup_retries={writeup_retries}; submission_priority>={effective_priority_bar}, "
            f"max_blockers={effective_blocker_bar}; strategy_feedback={strategy_feedback.get('rationale', [])}"
        )

    page_limit = writeup_plan["page_limit"]
    writing_audit_rounds = max(0, int(writing_audit_rounds))
    strict_guardrails = bool(strict_guardrails)
    guardrail_repair_rounds = max(0, int(guardrail_repair_rounds))
    if high_quality_mode:
        strict_guardrails = True
    logger(
        f"Writeup profile={writing_profile}; writing_audit_rounds={writing_audit_rounds}; "
        f"strict_guardrails={strict_guardrails}; guardrail_repair_rounds={guardrail_repair_rounds}"
    )
    writeup_success = reused_writeup
    if not reused_writeup:
        citations_text = gather_citations(
            str(idea_dir),
            num_cite_rounds=num_cite_rounds,
            small_model=model_citation,
        )

        for attempt in range(writeup_retries):
            logger(f"Writeup attempt {attempt + 1} of {writeup_retries}")
            if writeup_plan["writeup_engine"] == "normal":
                writeup_success = perform_writeup(
                    base_folder=str(idea_dir),
                    small_model=model_writeup_small,
                    big_model=model_writeup,
                    page_limit=page_limit,
                    citations_text=citations_text,
                    writing_profile=writing_profile,
                    writing_audit_rounds=writing_audit_rounds,
                    target_venue=resolved_target_venue,
                    strict_guardrails=strict_guardrails,
                    guardrail_repair_rounds=guardrail_repair_rounds,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=str(idea_dir),
                    small_model=model_writeup_small,
                    big_model=model_writeup,
                    page_limit=page_limit,
                    citations_text=citations_text,
                    writing_profile=writing_profile,
                    writing_audit_rounds=writing_audit_rounds,
                    target_venue=resolved_target_venue,
                    strict_guardrails=strict_guardrails,
                    guardrail_repair_rounds=guardrail_repair_rounds,
                )
            if writeup_success:
                break

    save_token_tracker(idea_dir)
    if not writeup_success:
        return {
            "success": False,
            "failure_stage": "writeup",
            "failure_reason": "writeup retries exhausted",
            "quality_result": quality_result,
            "acceptance": acceptance,
        }

    if high_quality_mode:
        execution_policy = build_workflow_execution_policy(
            workflow_mode,
            submission_mode=submission_mode,
            high_quality_mode=high_quality_mode,
            target_venue=resolved_target_venue,
        )
        quality_pass = execute_quality_workflow_with_followups(
            run_high_quality_pass_fn=run_high_quality_pass,
            run_dir=str(idea_dir),
            paper_type=writeup_type,
            rewrite_model=model_writeup,
            quality_model=quality_model or model_writeup,
            target_venue=resolved_target_venue,
            quality_preset=quality_preset,
            quality_threshold=quality_threshold,
            rigor_threshold=rigor_threshold,
            max_quality_rewrites=max_quality_rewrites,
            require_quality_gate=require_quality_gate,
            min_submission_priority=effective_priority_bar,
            max_submission_blockers=effective_blocker_bar,
            autonomous_followup_rounds=autonomous_quality_followup_rounds,
            allow_auto_improvement_fallback=(
                execution_policy.allow_auto_improvement_fallback
            ),
            reject_on_auto_improvement_fallback=(
                execution_policy.reject_on_auto_improvement_fallback
            ),
            resume=resume,
            logger=logger,
        )
        quality_result = quality_pass["quality_result"]
        quality_fallback_event = record_quality_fallback_if_needed(
            idea_dir,
            quality_result,
            producer="launcher_workflow.high_quality",
            strict=strict_fallbacks,
        )
        acceptance = quality_pass["acceptance"]
        logger(quality_pass["summary"])
        if strict_fallbacks and quality_fallback_event:
            reason = format_strict_fallback_error(
                quality_fallback_event,
                workflow_mode=workflow_mode,
                stage_hint="quality_review",
            )
            logger(reason)
            return {
                "success": False,
                "failure_stage": "quality_fallback_blocked",
                "failure_reason": reason,
                "quality_result": quality_result,
                "acceptance": acceptance,
            }
        if not acceptance.get("accepted"):
            logger(
                "High-quality pre-review gate not yet met; continuing into the final review loop: "
                + "; ".join(acceptance.get("reasons", []))
            )

    mark_stage_complete(
        idea_dir,
        "writeup",
        artifacts={"pdf_path": find_best_pdf_path(idea_dir, prefer_reflections=True)},
        metadata={
            "writeup_type": writeup_type,
            "page_limit": page_limit,
            "high_quality_mode": high_quality_mode,
            "writing_profile": writing_profile,
            "writing_audit_rounds": writing_audit_rounds,
            "strict_guardrails": strict_guardrails,
            "guardrail_repair_rounds": guardrail_repair_rounds,
            "target_venue": resolved_target_venue,
            "quality_gate_passed": quality_result.get("quality_gate_passed") if high_quality_mode else None,
            "submission_priority_score": quality_result.get("submission_priority_score") if high_quality_mode else None,
            "blocker_count": quality_result.get("blocker_count") if high_quality_mode else None,
            "autonomous_followup_rounds_run": quality_result.get("autonomous_followup_rounds_run") if high_quality_mode else None,
            "pre_review_submission_acceptance_passed": acceptance.get("accepted") if high_quality_mode else None,
            "pre_review_submission_acceptance_reasons": acceptance.get("reasons", []) if high_quality_mode else [],
            "strategy_feedback": strategy_feedback.get("rationale") if high_quality_mode else None,
        },
    )
    return {
        "success": True,
        "failure_stage": None,
        "failure_reason": None,
        "quality_result": quality_result,
        "acceptance": acceptance,
        "reused_writeup": reused_writeup,
    }


def run_review_phase(
    idea_dir: str | Path,
    *,
    model_review: str,
    paper_type: str = "icbinb",
    target_venue: Optional[str] = None,
    text_filename: str = "review_text.txt",
    image_filename: str = "review_img_cap_ref.json",
    text_mode: str = "text_json",
    review_reflections: int = 1,
    review_fewshot: int = 1,
    review_ensemble: int = 1,
    review_temperature: float = 0.75,
    review_strategy: Optional[str] = None,
    high_quality_mode: bool = False,
    research_root: str | Path | None = None,
    workflow_mode: Optional[str] = None,
    submission_mode: bool = False,
    require_quality_gate: bool = False,
    min_submission_priority: Optional[float] = None,
    max_submission_blockers: Optional[int] = None,
    reject_on_auto_improvement_fallback: bool | None = None,
    resume: bool = True,
) -> dict:
    text_path = Path(idea_dir) / text_filename
    image_path = Path(idea_dir) / image_filename
    pdf_path = find_best_pdf_path(idea_dir, prefer_reflections=True)
    execution_policy = None
    if high_quality_mode:
        execution_policy = build_workflow_execution_policy(
            workflow_mode,
            submission_mode=submission_mode,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
        if reject_on_auto_improvement_fallback is None:
            reject_on_auto_improvement_fallback = (
                execution_policy.reject_on_auto_improvement_fallback
            )
    elif reject_on_auto_improvement_fallback is None:
        reject_on_auto_improvement_fallback = False

    if resume and is_stage_complete(idea_dir, "review") and text_path.exists() and image_path.exists():
        submission_acceptance: dict[str, Any] = {}
        if high_quality_mode:
            submission_acceptance = evaluate_final_submission_readiness(
                run_dir=idea_dir,
                require_quality_gate=require_quality_gate,
                min_submission_priority=min_submission_priority,
                max_submission_blockers=max_submission_blockers,
                reject_on_auto_improvement_fallback=bool(
                    reject_on_auto_improvement_fallback
                ),
            )
        return {
            "found": pdf_path is not None,
            "pdf_path": pdf_path,
            "resumed": True,
            "submission_acceptance": submission_acceptance,
        }

    if pdf_path is None or not os.path.exists(pdf_path):
        return {"found": False, "pdf_path": pdf_path}

    review_plan = build_review_execution_plan(
        paper_type,
        target_venue=target_venue,
        review_reflections=review_reflections,
        review_ensemble=review_ensemble,
        review_fewshot=review_fewshot,
        review_temperature=review_temperature,
        review_strategy=review_strategy,
        high_quality_mode=high_quality_mode,
        research_root=research_root,
    )

    if high_quality_mode:
        venue = review_plan["target_venue"]
        strategy_feedback = review_plan["strategy_feedback"]
        logger = print
        logger(
            f"High-quality review venue: {venue}; reflections={review_plan['review_reflections']}, "
            f"ensemble={review_plan['review_ensemble']}, fewshot={review_plan['review_fewshot']}; "
            f"strategy_feedback={strategy_feedback.get('rationale', [])}"
        )
    workflow_runtime_plan = build_workflow_runtime_plan(
        workflow_mode,
        submission_mode=submission_mode,
        high_quality_mode=high_quality_mode,
        target_venue=target_venue,
    )
    review_pass = execute_review_suite(
        review_roles=workflow_runtime_plan.final_review_roles,
        paper_dir=idea_dir,
        model_review=model_review,
        review_plan=review_plan,
        create_client_fn=create_client,
        load_paper_fn=load_paper,
        perform_review_fn=perform_review,
        perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
        pdf_path_resolver=lambda folder: find_best_pdf_path(
            folder, prefer_reflections=True
        ),
        save_dir=idea_dir,
        text_filename=text_filename,
        image_filename=image_filename,
        text_mode=text_mode,
        project_root=idea_dir,
        persist_job=True,
        evidence_refs=[
            "claim_evidence_graph.json",
            "experiment_registry.jsonl",
            "figure_spec.json",
            "manuscript_state.json",
        ],
        suite_name="launcher_review",
        lane_name="review_board",
        strictness_profile="standard",
    )
    critic_pass = run_independent_critic_pass(
        workflow_runtime_plan=workflow_runtime_plan,
        paper_dir=idea_dir,
        model_review=model_review,
        review_plan=review_plan,
        create_client_fn=create_client,
        load_paper_fn=load_paper,
        perform_review_fn=perform_review,
        perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
        pdf_path_resolver=lambda folder: find_best_pdf_path(
            folder, prefer_reflections=True
        ),
        save_dir=Path(idea_dir) / "hostile_critic",
        project_root=idea_dir,
        evidence_refs=[
            "claim_evidence_graph.json",
            "experiment_registry.jsonl",
            "figure_spec.json",
            "manuscript_state.json",
        ],
        suite_name="launcher_hostile_critic",
    )
    review_text = review_pass["review_text"]
    review_img = review_pass["review_img"]
    submission_acceptance: dict[str, Any] = {}
    if high_quality_mode:
        submission_acceptance = evaluate_final_submission_readiness(
            run_dir=idea_dir,
            require_quality_gate=require_quality_gate,
            min_submission_priority=min_submission_priority,
            max_submission_blockers=max_submission_blockers,
            reject_on_auto_improvement_fallback=bool(
                reject_on_auto_improvement_fallback
            ),
        )

    mark_stage_complete(
        idea_dir,
        "review",
        artifacts={
            "pdf_path": pdf_path,
            "review_text": str(text_path),
            "review_image": str(image_path),
        },
        metadata={
            "strategy": review_plan["strategy"].value,
            "review_reflections": review_plan["review_reflections"],
            "review_fewshot": review_plan["review_fewshot"],
            "review_ensemble": review_plan["review_ensemble"],
            "workflow_mode": workflow_runtime_plan.workflow_mode,
            "review_roles": list(workflow_runtime_plan.final_review_roles),
            "critic_roles": list(workflow_runtime_plan.critic_review_roles),
            "critic_blocking_issue_count": critic_pass.get("blocking_issue_count"),
            "submission_acceptance_passed": submission_acceptance.get("accepted")
            if high_quality_mode
            else None,
            "submission_acceptance_reasons": submission_acceptance.get("reasons", [])
            if high_quality_mode
            else [],
            "strategy_feedback": review_plan["strategy_feedback"].get("rationale")
            if high_quality_mode
            else None,
        },
    )
    return {
        "found": True,
        "pdf_path": pdf_path,
        "review_text": review_text,
        "review_img": review_img,
        "strategy": review_plan["strategy"].value,
        "review_roles_used": list(workflow_runtime_plan.final_review_roles),
        "critic_roles_used": list(workflow_runtime_plan.critic_review_roles),
        "critic_active_issue_count": critic_pass.get("active_issue_count"),
        "critic_blocking_issue_count": critic_pass.get("blocking_issue_count"),
        "critic_findings_file": critic_pass.get("critic_findings_file"),
        "submission_acceptance": submission_acceptance,
    }
