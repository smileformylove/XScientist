from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ai_scientist.utils.decision_log import record_decision
from ai_scientist.utils.pipeline_contracts import load_contract_artifact
from ai_scientist.utils.workflow_runtime import WorkflowRuntimePlan, execute_review_suite


def _hostile_critic_ablation_enabled() -> bool:
    return str(os.environ.get("AI_SCIENTIST_ABLATE_HOSTILE_CRITIC") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _lane_summary(project_root: str | Path, lane_name: str) -> dict[str, Any]:
    review_state = load_contract_artifact(project_root, "review_state", default={}) or {}
    lane_summaries = (
        review_state.get("lane_summaries")
        if isinstance(review_state.get("lane_summaries"), dict)
        else {}
    )
    summary = lane_summaries.get(lane_name)
    return dict(summary) if isinstance(summary, dict) else {}


def _summary_count(summary: dict[str, Any], key: str) -> int:
    try:
        return int(summary.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _review_root(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("review"), dict):
        return dict(payload["review"])
    return dict(payload)


def _review_issue_counts(review_pass: dict[str, Any]) -> dict[str, Any]:
    """Compute role-level issue counts without relying on mutable lane summaries."""

    role_payloads: list[tuple[str, Any]] = []
    passes_by_role = review_pass.get("passes_by_role")
    if isinstance(passes_by_role, dict) and passes_by_role:
        for role, payload in passes_by_role.items():
            if not isinstance(payload, dict):
                continue
            role_payloads.append((str(role or "general"), payload.get("review_text")))
    elif review_pass.get("review_text") is not None:
        role_payloads.append((
            str(review_pass.get("primary_role") or "general"),
            review_pass.get("review_text"),
        ))

    active_count = 0
    blocking_count = 0
    parsed_role_count = 0
    malformed_roles: list[str] = []
    role_counts: dict[str, dict[str, int]] = {}
    for role, payload in role_payloads:
        root = _review_root(payload)
        if root is None:
            malformed_roles.append(role)
            continue
        parsed_role_count += 1
        weakness_count = len(_coerce_list(root.get("Weaknesses")))
        limitation_count = len(_coerce_list(root.get("Limitations")))
        concern_count = len(_coerce_list(root.get("Concerns")))
        risk_count = len(_coerce_list(root.get("Risks")))
        question_count = len(_coerce_list(root.get("Questions")))
        role_blocking = weakness_count + limitation_count + concern_count + risk_count
        role_active = role_blocking + question_count
        role_counts[role] = {
            "active_issue_count": role_active,
            "blocking_issue_count": role_blocking,
        }
        active_count += role_active
        blocking_count += role_blocking
    parse_complete = bool(role_payloads) and parsed_role_count == len(role_payloads)
    return {
        "active_issue_count": active_count,
        "blocking_issue_count": blocking_count,
        "parsed_role_count": parsed_role_count,
        "expected_role_count": len(role_payloads),
        "parse_complete": parse_complete,
        "malformed_roles": malformed_roles,
        "role_counts": role_counts,
    }


def _effective_issue_counts(
    *,
    review_pass: dict[str, Any],
    project_root: str | Path,
    lane_name: str,
    use_lane_summary: bool = True,
) -> dict[str, Any]:
    computed = _review_issue_counts(review_pass)
    lane_summary = _lane_summary(project_root, lane_name) if use_lane_summary else {}
    role_payload_materialized = bool(computed.get("expected_role_count"))
    lane_materialized = role_payload_materialized or (use_lane_summary and bool(lane_summary))
    if not computed["parse_complete"]:
        summary_blocking = _summary_count(lane_summary, "blocking_issue_count")
        summary_active = _summary_count(lane_summary, "active_issue_count")
        parse_failure_penalty = 1
        computed["blocking_issue_count"] = max(
            int(computed["blocking_issue_count"]),
            summary_blocking,
            parse_failure_penalty,
        )
        computed["active_issue_count"] = max(
            int(computed["active_issue_count"]),
            summary_active,
            int(computed["blocking_issue_count"]),
        )
        computed["parse_failure_penalty"] = parse_failure_penalty
    else:
        computed["blocking_issue_count"] = max(
            int(computed["blocking_issue_count"]),
            _summary_count(lane_summary, "blocking_issue_count"),
        )
        computed["active_issue_count"] = max(
            int(computed["active_issue_count"]),
            _summary_count(lane_summary, "active_issue_count"),
        )
        computed["parse_failure_penalty"] = 0
    computed["lane_materialized"] = lane_materialized
    return computed


def _confirmation_review_plan(review_plan: dict[str, Any]) -> dict[str, Any]:
    confirmation_plan = dict(review_plan)
    instruction = str(confirmation_plan.get("review_instruction") or "").strip()
    confirmation_plan["review_instruction"] = (
        f"{instruction}\n\n"
        "Independent confirmation pass: evaluate the artifact directly against the "
        "same acceptance criteria. Do not rely on or mention any prior critic "
        "verdict, self-assessment, score, or pass/fail result. Surface concrete "
        "blocking issues if the artifact would not independently pass."
    ).strip()
    return confirmation_plan


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
    ablation_enabled = _hostile_critic_ablation_enabled()
    skip_reason = None
    if ablation_enabled:
        skip_reason = "hostile critic ablation is enabled"
    elif not workflow_runtime_plan.requires_independent_critic:
        skip_reason = "workflow runtime plan does not require independent critic"
    elif not workflow_runtime_plan.critic_review_roles:
        skip_reason = "workflow runtime plan has no critic review roles"
    if skip_reason:
        record_decision(
            project_root,
            category="hostile_critic_confirmation",
            selected="skip_independent_confirmation",
            options_considered=[
                "run_independent_confirmation",
                "skip_independent_confirmation",
            ],
            rejected_because={
                "run_independent_confirmation": skip_reason,
            },
            producer="critic_workflow.independent_confirmation",
            metadata={
                "suite_name": suite_name,
                "workflow_mode": workflow_runtime_plan.workflow_mode,
                "ablation_enabled": bool(ablation_enabled),
                "requires_independent_critic": bool(
                    workflow_runtime_plan.requires_independent_critic
                ),
                "critic_review_roles": list(workflow_runtime_plan.critic_review_roles),
            },
        )
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
    primary_counts = _effective_issue_counts(
        review_pass=review_pass,
        project_root=project_root,
        lane_name="hostile_critic",
    )
    primary_active_count = int(primary_counts["active_issue_count"])
    primary_blocking_count = int(primary_counts["blocking_issue_count"])
    confirmation = {
        "ran": False,
        "found": False,
        "confirmed": False,
        "active_issue_count": 0,
        "blocking_issue_count": 0,
        "source": "ep_independent_eval",
        "review_roles_used": [],
        "reason": "primary_critic_not_clear",
    }
    if review_pass.get("found") and primary_blocking_count == 0:
        record_decision(
            project_root,
            category="hostile_critic_confirmation",
            selected="run_independent_confirmation",
            options_considered=[
                "run_independent_confirmation",
                "skip_independent_confirmation",
            ],
            rejected_because={
                "skip_independent_confirmation": "primary hostile critic produced no blockers"
            },
            producer="critic_workflow.independent_confirmation",
            metadata={
                "suite_name": suite_name,
                "primary_active_issue_count": primary_active_count,
                "primary_blocking_issue_count": primary_blocking_count,
            },
        )
        confirmation_pass = execute_review_suite(
            review_roles=workflow_runtime_plan.critic_review_roles,
            paper_dir=paper_dir,
            model_review=model_review,
            review_plan=_confirmation_review_plan(review_plan),
            create_client_fn=create_client_fn,
            load_paper_fn=load_paper_fn,
            perform_review_fn=perform_review_fn,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review_fn,
            pdf_path_resolver=pdf_path_resolver,
            save_dir=Path(save_dir) / "independent_confirmation",
            text_filename=text_filename,
            image_filename=image_filename,
            project_root=project_root,
            persist_job=True,
            evidence_refs=evidence_refs,
            suite_name=f"{suite_name}_independent_confirmation",
            lane_name="hostile_critic_confirmation",
            strictness_profile=workflow_runtime_plan.critic_strictness_profile,
        )
        confirmation_counts = _effective_issue_counts(
            review_pass=confirmation_pass,
            project_root=project_root,
            lane_name="hostile_critic_confirmation",
            use_lane_summary=False,
        )
        confirmation_active_count = int(confirmation_counts["active_issue_count"])
        confirmation_blocking_count = int(confirmation_counts["blocking_issue_count"])
        confirmation_found = bool(confirmation_pass.get("found")) and bool(
            confirmation_counts.get("lane_materialized")
        )
        confirmation = {
            "ran": True,
            "found": confirmation_found,
            "confirmed": confirmation_found and confirmation_blocking_count == 0,
            "active_issue_count": confirmation_active_count,
            "blocking_issue_count": confirmation_blocking_count,
            "parse_complete": bool(confirmation_counts.get("parse_complete")),
            "lane_materialized": bool(confirmation_counts.get("lane_materialized")),
            "role_counts": confirmation_counts.get("role_counts") or {},
            "malformed_roles": confirmation_counts.get("malformed_roles") or [],
            "source": "ep_independent_eval",
            "review_roles_used": list(confirmation_pass.get("review_roles_used") or []),
            "reason": (
                "confirmed"
                if confirmation_found and confirmation_blocking_count == 0
                else (
                    "confirmation_blockers"
                    if confirmation_found
                    else "confirmation_missing_artifact"
                )
            ),
        }
    else:
        record_decision(
            project_root,
            category="hostile_critic_confirmation",
            selected="skip_independent_confirmation",
            options_considered=[
                "run_independent_confirmation",
                "skip_independent_confirmation",
            ],
            rejected_because={
                "run_independent_confirmation": (
                    "primary hostile critic did not produce a clear pass"
                )
            },
            producer="critic_workflow.independent_confirmation",
            metadata={
                "suite_name": suite_name,
                "primary_found": bool(review_pass.get("found")),
                "primary_active_issue_count": primary_active_count,
                "primary_blocking_issue_count": primary_blocking_count,
            },
        )
    confirmation_blocker_penalty = (
        1
        if confirmation.get("ran") and not confirmation.get("found")
        else int(confirmation.get("blocking_issue_count") or 0)
    )
    aggregate_blocking_count = primary_blocking_count + confirmation_blocker_penalty
    aggregate_active_count = primary_active_count + int(
        confirmation.get("active_issue_count") or 0
    ) + (1 if confirmation.get("ran") and not confirmation.get("found") else 0)
    critic_findings_path = Path(project_root).expanduser().resolve() / "critic_findings.json"
    return {
        **review_pass,
        "ran": True,
        "active_issue_count": aggregate_active_count,
        "blocking_issue_count": aggregate_blocking_count,
        "primary_active_issue_count": primary_active_count,
        "primary_blocking_issue_count": primary_blocking_count,
        "primary_issue_counts": primary_counts,
        "critic_confirmed": bool(
            review_pass.get("found")
            and primary_blocking_count == 0
            and confirmation.get("confirmed")
        ),
        "critic_confirmation": confirmation,
        "blocking_source": (
            "ep_independent_eval"
            if confirmation_blocker_penalty
            else (
                "hostile_critic"
                if primary_blocking_count
                else None
            )
        ),
        "critic_findings_file": (
            str(critic_findings_path) if critic_findings_path.exists() else None
        ),
    }
