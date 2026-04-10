from __future__ import annotations

"""Runtime helpers that turn workflow modes into concrete review behavior."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ai_scientist.utils.pipeline_helpers import save_review_artifacts
from ai_scientist.utils.review_execution import execute_review_pass
from ai_scientist.utils.review_jobs import REVIEW_ROLES
from ai_scientist.utils.workflow_modes import (
    WorkflowModeSpec,
    resolve_workflow_mode,
    stage_sequence_for_mode,
)


@dataclass(frozen=True)
class WorkflowRuntimePlan:
    workflow_mode: str
    workflow_label: str
    stage_sequence: tuple[str, ...]
    inspirations: tuple[str, ...]
    agent_lanes: tuple[str, ...]
    improvement_review_roles: tuple[str, ...]
    final_review_roles: tuple[str, ...]
    critic_review_roles: tuple[str, ...]
    requires_independent_critic: bool
    critic_strictness_profile: str


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _review_root(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("review"), dict):
        return dict(payload["review"])
    return dict(payload)


def _collect_scores(root: dict[str, Any]) -> dict[str, float]:
    collected: dict[str, float] = {}
    raw_scores = root.get("scores")
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            try:
                collected[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return collected


def build_workflow_runtime_plan(
    workflow_mode: str | WorkflowModeSpec | None,
    *,
    submission_mode: bool = False,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
    target_venue: str | None = None,
) -> WorkflowRuntimePlan:
    spec = (
        workflow_mode
        if isinstance(workflow_mode, WorkflowModeSpec)
        else resolve_workflow_mode(
            str(workflow_mode or "adaptive"),
            submission_mode=submission_mode,
            breakthrough_mode=breakthrough_mode,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
    )
    runtime_map: dict[str, dict[str, Any]] = {
        "classic_pipeline": {
            "agent_lanes": ("planner", "experiment", "writer", "reviewer"),
            "improvement_roles": ("rigor",),
            "final_roles": ("clarity",),
            "critic_roles": (),
            "requires_independent_critic": False,
            "critic_strictness_profile": "standard",
        },
        "agentic_tree": {
            "agent_lanes": (
                "planner",
                "experiment_manager",
                "experiment_worker",
                "writer",
                "reviewer",
            ),
            "improvement_roles": ("novelty", "rigor"),
            "final_roles": ("clarity", "reproducibility"),
            "critic_roles": (),
            "requires_independent_critic": False,
            "critic_strictness_profile": "exploratory",
        },
        "program_driven": {
            "agent_lanes": ("planner", "experiment", "writer", "reviewer", "repair"),
            "improvement_roles": ("rigor", "reproducibility"),
            "final_roles": ("clarity", "reproducibility"),
            "critic_roles": (),
            "requires_independent_critic": False,
            "critic_strictness_profile": "programmatic",
        },
        "writing_studio": {
            "agent_lanes": (
                "planner",
                "results_analyst",
                "storyline_editor",
                "latex_hygiene_editor",
                "humanizer",
                "reviewer",
            ),
            "improvement_roles": ("clarity", "rigor"),
            "final_roles": ("clarity",),
            "critic_roles": (),
            "requires_independent_critic": False,
            "critic_strictness_profile": "writing_heavy",
        },
        "review_board": {
            "agent_lanes": (
                "planner",
                "experiment",
                "writer",
                "reviewer_board",
                "repair",
            ),
            "improvement_roles": (
                "novelty",
                "rigor",
                "clarity",
                "reproducibility",
                "claim_cross_examiner",
            ),
            "final_roles": (
                "novelty",
                "rigor",
                "clarity",
                "reproducibility",
                "skeptical_pc_member",
            ),
            "critic_roles": (),
            "requires_independent_critic": False,
            "critic_strictness_profile": "review_board",
        },
        "multi_agent_board": {
            "agent_lanes": (
                "planner",
                "experiment_manager",
                "experiment_worker",
                "results_analyst",
                "storyline_editor",
                "latex_hygiene_editor",
                "humanizer",
                "quality_gate",
                "reviewer_board",
                "hostile_critic",
                "repair",
            ),
            "improvement_roles": (
                "novelty",
                "rigor",
                "clarity",
                "reproducibility",
                "claim_cross_examiner",
            ),
            "final_roles": (
                "novelty",
                "rigor",
                "clarity",
                "reproducibility",
                "skeptical_pc_member",
                "meta_reviewer",
            ),
            "critic_roles": (
                "skeptical_pc_member",
                "claim_cross_examiner",
                "reproducibility_assassin",
                "novelty_executioner",
                "stats_sniper",
                "related_work_skeptic",
                "meta_reviewer",
                "desk_reject_editor",
            ),
            "requires_independent_critic": True,
            "critic_strictness_profile": "adversarial",
        },
    }
    runtime_config = runtime_map.get(
        spec.name,
        runtime_map["classic_pipeline"],
    )
    improvement_roles = tuple(runtime_config["improvement_roles"])
    final_roles = tuple(runtime_config["final_roles"])
    critic_roles = tuple(runtime_config["critic_roles"])
    if high_quality_mode and "reproducibility" not in final_roles:
        final_roles = final_roles + ("reproducibility",)
    return WorkflowRuntimePlan(
        workflow_mode=spec.name,
        workflow_label=spec.label,
        stage_sequence=tuple(stage_sequence_for_mode(spec)),
        inspirations=tuple(spec.inspirations),
        agent_lanes=tuple(runtime_config["agent_lanes"]),
        improvement_review_roles=tuple(_dedupe_preserve(list(improvement_roles))),
        final_review_roles=tuple(_dedupe_preserve(list(final_roles))),
        critic_review_roles=tuple(_dedupe_preserve(list(critic_roles))),
        requires_independent_critic=bool(
            runtime_config["requires_independent_critic"]
        ),
        critic_strictness_profile=str(runtime_config["critic_strictness_profile"]),
    )


def merge_text_reviews(
    review_text_by_role: dict[str, Any],
    *,
    primary_role: str | None = None,
) -> dict[str, Any] | None:
    if not review_text_by_role:
        return None

    summaries: list[str] = []
    strengths: list[str] = []
    weaknesses: list[str] = []
    questions: list[str] = []
    limitations: list[str] = []
    decisions: list[str] = []
    score_buckets: dict[str, list[float]] = {}
    role_reviews: dict[str, Any] = {}

    for role, payload in review_text_by_role.items():
        root = _review_root(payload)
        if not root:
            continue
        role_reviews[role] = root
        summary = str(root.get("Summary") or "").strip()
        if summary:
            summaries.append(f"{role}: {summary}")
        strengths.extend(_coerce_list(root.get("Strengths")))
        weaknesses.extend(_coerce_list(root.get("Weaknesses")))
        questions.extend(_coerce_list(root.get("Questions")))
        limitations.extend(_coerce_list(root.get("Limitations")))
        decision = str(root.get("Decision") or "").strip()
        if decision:
            decisions.append(decision)
        for score_key, score_value in _collect_scores(root).items():
            score_buckets.setdefault(score_key, []).append(score_value)

    ordered_roles = list(review_text_by_role.keys())
    merged_scores = {
        key: round(sum(values) / len(values), 3)
        for key, values in score_buckets.items()
        if values
    }
    decision_values = _dedupe_preserve(decisions)
    effective_primary_role = str(primary_role or (ordered_roles[0] if ordered_roles else "general"))
    merged_review = {
        "Summary": (
            "\n".join(summaries[:6])
            if summaries
            else f"Multi-role review synthesized across {', '.join(ordered_roles)}."
        ),
        "Strengths": _dedupe_preserve(strengths),
        "Weaknesses": _dedupe_preserve(weaknesses),
        "Questions": _dedupe_preserve(questions),
        "Limitations": _dedupe_preserve(limitations),
        "Decision": "; ".join(decision_values[:3]) if decision_values else "",
        "scores": merged_scores,
        "role_reviews": role_reviews,
    }
    return {
        "review": merged_review,
        "workflow_review_roles": ordered_roles,
        "primary_role": effective_primary_role,
    }


def merge_image_reviews(
    review_img_by_role: dict[str, Any],
    *,
    primary_role: str | None = None,
) -> dict[str, Any] | None:
    if not review_img_by_role:
        return None

    figure_reviews: list[dict[str, Any]] = []
    role_reviews: dict[str, Any] = {}
    ordered_roles = list(review_img_by_role.keys())
    for role, payload in review_img_by_role.items():
        if not isinstance(payload, dict):
            continue
        role_reviews[role] = payload
        if isinstance(payload.get("figure_reviews"), list):
            for row in payload["figure_reviews"]:
                if not isinstance(row, dict):
                    continue
                annotated = dict(row)
                annotated.setdefault("review_role", role)
                figure_reviews.append(annotated)
            continue
        for figure_id, row in payload.items():
            if not isinstance(row, dict):
                continue
            annotated = dict(row)
            annotated.setdefault("figure_id", str(row.get("figure_id") or figure_id))
            annotated.setdefault("review_role", role)
            figure_reviews.append(annotated)

    return {
        "figure_reviews": figure_reviews,
        "role_reviews": role_reviews,
        "workflow_review_roles": ordered_roles,
        "primary_role": str(primary_role or (ordered_roles[0] if ordered_roles else "general")),
    }


def execute_review_suite(
    *,
    review_roles: list[str] | tuple[str, ...],
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
    persist_job: bool = False,
    evidence_refs: list[str] | None = None,
    suite_name: str | None = None,
    lane_name: str = "review",
    strictness_profile: str = "standard",
) -> dict[str, Any]:
    normalized_roles = [
        role
        for role in _dedupe_preserve([str(role or "").strip().lower() for role in review_roles])
        if role == "general" or role in REVIEW_ROLES
    ]
    if not normalized_roles:
        normalized_roles = ["general"]

    suite_dir = Path(save_dir) if save_dir is not None else None
    per_role_artifacts = len(normalized_roles) > 1 and suite_dir is not None
    review_text_by_role: dict[str, Any] = {}
    review_img_by_role: dict[str, Any] = {}
    passes_by_role: dict[str, dict[str, Any]] = {}
    pdf_path: str | None = None

    for role in normalized_roles:
        role_save_dir = suite_dir / role if per_role_artifacts and suite_dir is not None else suite_dir
        review_pass = execute_review_pass(
            paper_dir=paper_dir,
            model_review=model_review,
            review_plan=review_plan,
            create_client_fn=create_client_fn,
            load_paper_fn=load_paper_fn,
            perform_review_fn=perform_review_fn,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review_fn,
            pdf_path_resolver=pdf_path_resolver,
            save_dir=role_save_dir,
            text_filename=text_filename,
            image_filename=image_filename,
            text_mode=text_mode,
            save_review_artifacts_fn=save_review_artifacts_fn,
            project_root=project_root,
            review_role=role,
            persist_job=persist_job,
            evidence_refs=evidence_refs,
            lane_name=lane_name,
            suite_name=suite_name,
            strictness_profile=strictness_profile,
        )
        passes_by_role[role] = review_pass
        if not review_pass.get("found"):
            return {
                "found": False,
                "pdf_path": review_pass.get("pdf_path"),
                "review_text": None,
                "review_img": None,
                "passes_by_role": passes_by_role,
                "review_roles_used": normalized_roles,
                "primary_role": normalized_roles[0],
            }
        pdf_path = str(review_pass.get("pdf_path") or pdf_path or "")
        review_text_by_role[role] = review_pass.get("review_text")
        review_img_by_role[role] = review_pass.get("review_img")

    primary_role = normalized_roles[0]
    merged_review_text = merge_text_reviews(
        review_text_by_role,
        primary_role=primary_role,
    )
    merged_review_img = merge_image_reviews(
        review_img_by_role,
        primary_role=primary_role,
    )
    if suite_dir is not None:
        save_review_artifacts_fn(
            suite_dir,
            text_review=merged_review_text,
            image_review=merged_review_img,
            text_filename=text_filename,
            image_filename=image_filename,
            text_mode=text_mode,
        )
        suite_summary = {
            "suite_name": str(suite_name or "review_suite"),
            "lane_name": str(lane_name or "review"),
            "strictness_profile": str(strictness_profile or "standard"),
            "review_roles": normalized_roles,
            "primary_role": primary_role,
            "pdf_path": pdf_path,
            "job_ids": {
                role: ((payload.get("job") or {}).get("job_id"))
                for role, payload in passes_by_role.items()
            },
        }
        (suite_dir / "review_suite.json").write_text(
            json.dumps(suite_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "found": True,
        "pdf_path": pdf_path,
        "review_text": merged_review_text,
        "review_img": merged_review_img,
        "passes_by_role": passes_by_role,
        "review_roles_used": normalized_roles,
        "primary_role": primary_role,
    }
