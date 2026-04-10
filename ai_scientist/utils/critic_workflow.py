from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ai_scientist.utils.pipeline_contracts import load_contract_artifact
from ai_scientist.utils.workflow_runtime import WorkflowRuntimePlan, execute_review_suite


def _hostile_critic_ablation_enabled() -> bool:
    return str(os.environ.get("AI_SCIENTIST_ABLATE_HOSTILE_CRITIC") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def run_independent_critic_pass(
    *,
    workflow_runtime_plan: WorkflowRuntimePlan,
    paper_dir: str | Path,
    model_review: str,
    review_plan: dict[str, Any],
    create_client_fn: Callable[[str], tuple[Any, Any]],
    load_paper_fn: Callable[[str], Any],
    perform_review_fn: Callable[..., Any],
    perform_imgs_cap_ref_review_fn: Callable[..., Any],
    pdf_path_resolver: Callable[[str | Path], str | None],
    save_dir: str | Path,
    project_root: str | Path,
    evidence_refs: list[str] | None = None,
    text_filename: str = "critic_review.json",
    image_filename: str = "critic_review_img.json",
    suite_name: str = "hostile_critic",
) -> dict[str, Any]:
    if (
        _hostile_critic_ablation_enabled()
        or
        not workflow_runtime_plan.requires_independent_critic
        or not workflow_runtime_plan.critic_review_roles
    ):
        return {
            "ran": False,
            "found": False,
            "review_roles_used": [],
            "active_issue_count": 0,
            "blocking_issue_count": 0,
            "critic_findings_file": None,
        }

    review_pass = execute_review_suite(
        review_roles=workflow_runtime_plan.critic_review_roles,
        paper_dir=paper_dir,
        model_review=model_review,
        review_plan=review_plan,
        create_client_fn=create_client_fn,
        load_paper_fn=load_paper_fn,
        perform_review_fn=perform_review_fn,
        perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review_fn,
        pdf_path_resolver=pdf_path_resolver,
        save_dir=save_dir,
        text_filename=text_filename,
        image_filename=image_filename,
        project_root=project_root,
        persist_job=True,
        evidence_refs=evidence_refs,
        suite_name=suite_name,
        lane_name="hostile_critic",
        strictness_profile=workflow_runtime_plan.critic_strictness_profile,
    )
    review_state = load_contract_artifact(project_root, "review_state", default={}) or {}
    lane_summaries = (
        review_state.get("lane_summaries")
        if isinstance(review_state.get("lane_summaries"), dict)
        else {}
    )
    hostile_summary = (
        lane_summaries.get("hostile_critic")
        if isinstance(lane_summaries.get("hostile_critic"), dict)
        else {}
    )
    critic_findings_path = Path(project_root).expanduser().resolve() / "critic_findings.json"
    return {
        **review_pass,
        "ran": True,
        "active_issue_count": int(hostile_summary.get("active_issue_count") or 0),
        "blocking_issue_count": int(hostile_summary.get("blocking_issue_count") or 0),
        "critic_findings_file": (
            str(critic_findings_path) if critic_findings_path.exists() else None
        ),
    }
