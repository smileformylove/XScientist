from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ai_scientist.utils.pipeline_helpers import save_review_artifacts
from ai_scientist.utils.review_jobs import (
    begin_review_job,
    build_review_role_instruction,
    finalize_review_job,
)
from ai_scientist.utils.token_tracker import token_tracker


def execute_review_pass(
    *,
    paper_dir: str | Path,
    model_review: str,
    review_plan: dict[str, Any],
    create_client_fn: Callable[[str], tuple[Any, Any]],
    load_paper_fn: Callable[[str], Any],
    perform_review_fn: Callable[..., Any],
    perform_imgs_cap_ref_review_fn: Callable[..., Any],
    pdf_path_resolver: Callable[[str | Path], str | None],
    save_dir: str | Path | None = None,
    text_filename: str = "review_text.json",
    image_filename: str = "review_img.json",
    text_mode: str = "json",
    save_review_artifacts_fn: Callable[..., None] = save_review_artifacts,
    project_root: str | Path | None = None,
    review_role: str = "general",
    persist_job: bool = False,
    evidence_refs: list[str] | None = None,
    lane_name: str = "review",
    suite_name: str | None = None,
    strictness_profile: str = "standard",
) -> dict[str, Any]:
    pdf_path = pdf_path_resolver(paper_dir)
    job = None
    effective_review_plan = dict(review_plan)
    if review_role and review_role != "general":
        effective_review_plan["review_instruction"] = (
            f"{effective_review_plan['review_instruction']}\n\n"
            f"Reviewer role focus: {build_review_role_instruction(review_role)}"
        )
    if persist_job and project_root is not None:
        job = begin_review_job(
            project_root,
            role=review_role,
            model_review=model_review,
            review_plan=effective_review_plan,
            lane_name=lane_name,
            suite_name=suite_name,
            strictness_profile=strictness_profile,
        )
        if save_dir is None:
            save_dir = Path(job["job_dir"])
    if pdf_path is None or not os.path.exists(pdf_path):
        if job is not None:
            finalize_review_job(
                project_root,
                job=job,
                review_text=None,
                review_img=None,
                pdf_path=pdf_path,
                usage_summary=token_tracker.get_summary(),
                evidence_refs=evidence_refs,
            )
        return {
            "found": False,
            "pdf_path": pdf_path,
            "review_text": None,
            "review_img": None,
            "job": job,
        }

    paper_content = load_paper_fn(pdf_path)
    client, client_model = create_client_fn(model_review)
    review_text = perform_review_fn(
        paper_content,
        client_model,
        client,
        num_reflections=effective_review_plan["review_reflections"],
        num_fs_examples=effective_review_plan["review_fewshot"],
        num_reviews_ensemble=effective_review_plan["review_ensemble"],
        temperature=effective_review_plan["review_temperature"],
        review_instruction_form=effective_review_plan["review_instruction"],
    )
    review_img = perform_imgs_cap_ref_review_fn(client, client_model, pdf_path)

    if save_dir is not None:
        save_review_artifacts_fn(
            save_dir,
            text_review=review_text,
            image_review=review_img,
            text_filename=text_filename,
            image_filename=image_filename,
            text_mode=text_mode,
        )

    if job is not None:
        finalize_review_job(
            project_root,
            job=job,
            review_text=review_text,
            review_img=review_img,
            pdf_path=pdf_path,
            usage_summary=token_tracker.get_summary(),
            evidence_refs=evidence_refs,
        )

    return {
        "found": True,
        "pdf_path": pdf_path,
        "review_text": review_text,
        "review_img": review_img,
        "job": job,
    }
