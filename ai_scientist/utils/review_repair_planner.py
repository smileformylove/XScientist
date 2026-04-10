from __future__ import annotations

"""Agentic reviewer-repair planning built from structured review queues."""

from datetime import datetime
import os
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import (
    load_contract_artifact,
    load_pipeline_manifest,
    save_contract_artifact,
)

LANE_LABELS = {
    "figure_repair": "Figure Repair Lane",
    "claim_repair": "Claim Repair Lane",
    "section_rewrite": "Section Rewrite Lane",
    "evidence_followup": "Evidence Follow-up Lane",
    "method_repair": "Method Repair Lane",
    "generic_rewrite": "Generic Rewrite Lane",
    "triage": "Issue Triage Lane",
}

_EXPERIMENT_KEYWORDS = (
    "baseline",
    "ablation",
    "experiment",
    "benchmark",
    "control",
    "variance",
    "stability",
    "significance",
    "robustness",
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize(text: Any) -> str:
    return str(text or "").strip().lower()


def _owner_aware_repair_disabled() -> bool:
    return str(os.environ.get("AI_SCIENTIST_DISABLE_OWNER_AWARE_REPAIR") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _infer_lane(task: dict[str, Any]) -> str:
    if _owner_aware_repair_disabled():
        return "generic_rewrite"
    issue_text = _normalize(task.get("issue_text"))
    role = _normalize(task.get("role"))
    if _coerce_list(task.get("figure_ids")):
        return "figure_repair"
    if role == "reproducibility":
        return "method_repair"
    if any(token in issue_text for token in _EXPERIMENT_KEYWORDS):
        return "evidence_followup"
    if _coerce_list(task.get("claim_ids")):
        return "claim_repair"
    if _coerce_list(task.get("section_ids")):
        return "section_rewrite"
    return "triage"


def _infer_owner(task: dict[str, Any], lane: str) -> tuple[str, str]:
    if _owner_aware_repair_disabled() or lane == "generic_rewrite":
        return ("repair_agent", "Owner-aware routing is disabled for this ablation run.")
    blocker_class = str(task.get("blocker_class") or "").strip().lower()
    role = str(task.get("role") or "").strip().lower()
    primary_target_type = str(task.get("primary_target_type") or "").strip().lower()
    if lane in {"evidence_followup", "method_repair"} or blocker_class in {
        "evidence_hole",
        "reproducibility_gap",
        "statistical_gap",
    }:
        return ("experiment_agent", "Requires stronger evidence or protocol detail.")
    if lane == "figure_repair" or primary_target_type == "figure" or blocker_class == "figure_gap":
        return ("figure_agent", "Issue is centered on figures, captions, or visual packaging.")
    if blocker_class in {"oversell", "novelty_risk", "citation_gap", "positioning_gap"}:
        return ("storyline_editor", "Issue is about claim scope, novelty framing, or overstatement.")
    if role in {"clarity", "style_snob", "desk_reject_editor"} or lane == "section_rewrite":
        return ("writing_agent", "Issue is primarily narrative clarity or exposition.")
    if lane == "triage":
        return ("planner_agent", "Issue needs routing, scoping, or dependency clarification.")
    return ("repair_agent", "Generic repair fallback.")


def _build_execution_steps(task: dict[str, Any], lane: str) -> list[str]:
    actions = _coerce_list(task.get("repair_actions"))
    if actions:
        return actions

    issue_text = str(task.get("issue_text") or "").strip()
    target_label = str(task.get("primary_target_label") or "target").strip()
    if lane == "generic_rewrite":
        return [f"Handle the reviewer issue for {target_label} through a generic rewrite lane: {issue_text}"]
    if lane == "figure_repair":
        return [f"Revise the figure package for {target_label} to resolve: {issue_text}"]
    if lane == "claim_repair":
        return [f"Rewrite and tighten the claim framing for {target_label}: {issue_text}"]
    if lane == "evidence_followup":
        return [f"Run or surface a stronger evidence check for: {issue_text}"]
    if lane == "method_repair":
        return [f"Clarify the reproducibility and setup details for {target_label}: {issue_text}"]
    if lane == "section_rewrite":
        return [f"Rewrite the section for {target_label}: {issue_text}"]
    return [f"Triage reviewer issue and define a repair path: {issue_text}"]


def _build_success_criteria(task: dict[str, Any], lane: str) -> list[str]:
    target_label = str(task.get("primary_target_label") or "target").strip()
    issue_text = str(task.get("issue_text") or "").strip()
    verification_checks = _coerce_list(task.get("verification_checks"))
    criteria = list(verification_checks)
    if lane == "generic_rewrite":
        criteria.append(
            f"The generic repair lane closes the reviewer issue for {target_label}: {issue_text}"
        )
    elif lane == "figure_repair":
        criteria.append(
            f"{target_label} is traceable to data and directly addresses the reviewer concern."
        )
    elif lane == "claim_repair":
        criteria.append(
            f"The revised claim for {target_label} is supported by explicitly linked evidence."
        )
    elif lane == "evidence_followup":
        criteria.append(
            f"New or clarified experiment evidence resolves: {issue_text}"
        )
    elif lane == "method_repair":
        criteria.append(
            f"Another team could reproduce the updated method/setup without guessing hidden details."
        )
    elif lane == "section_rewrite":
        criteria.append(
            f"The updated section narrative explicitly closes the reviewer concern: {issue_text}"
        )
    else:
        criteria.append("The issue has a concrete owner, target, and verification path.")
    return _dedupe(criteria)


def _build_required_inputs(task: dict[str, Any], lane: str) -> list[str]:
    inputs = ["review_state", "repair_queue"]
    if lane == "generic_rewrite":
        inputs.extend(["manuscript_state", "claim_evidence_graph"])
    if lane in {"evidence_followup", "method_repair"}:
        inputs.extend(["experiment_registry", "claim_evidence_graph"])
    if lane == "figure_repair":
        inputs.extend(["figure_spec", "manuscript_state"])
    if lane in {"claim_repair", "section_rewrite"}:
        inputs.extend(["manuscript_state", "claim_evidence_graph"])
    if str(task.get("review_lane") or "").strip() == "hostile_critic":
        inputs.append("critic_findings")
    inputs.extend(f"claim:{item}" for item in _coerce_list(task.get("claim_ids")))
    inputs.extend(f"figure:{item}" for item in _coerce_list(task.get("figure_ids")))
    inputs.extend(f"section:{item}" for item in _coerce_list(task.get("section_ids")))
    return _dedupe(inputs)


def _build_produced_artifacts(task: dict[str, Any], lane: str) -> list[str]:
    target_label = str(task.get("primary_target_label") or "target").strip()
    if lane == "generic_rewrite":
        return [f"generic_rewrite:{target_label}", f"verification_note:{target_label}"]
    if lane == "figure_repair":
        return [f"figure_revision:{target_label}", f"caption_revision:{target_label}"]
    if lane == "claim_repair":
        return [f"claim_rewrite:{target_label}", f"claim_binding_update:{target_label}"]
    if lane == "section_rewrite":
        return [f"section_rewrite:{target_label}", f"narrative_diff:{target_label}"]
    if lane in {"evidence_followup", "method_repair"}:
        return [f"evidence_update:{target_label}", f"verification_note:{target_label}"]
    return [f"triage_note:{target_label}", f"routing_decision:{target_label}"]


def _build_verifier(task: dict[str, Any], lane: str) -> str:
    if str(task.get("review_lane") or "").strip() == "hostile_critic":
        return "hostile_critic_recheck"
    if lane == "generic_rewrite":
        return "reviewer_board_recheck"
    if lane in {"evidence_followup", "method_repair"}:
        return "experiment_validation"
    if lane == "figure_repair":
        return "figure_alignment_check"
    if lane in {"claim_repair", "section_rewrite"}:
        return "reviewer_board_recheck"
    return "planner_triage_recheck"


def _build_close_condition(task: dict[str, Any], lane: str) -> str:
    issue_text = str(task.get("issue_text") or "").strip()
    if lane == "generic_rewrite":
        return f"Close only when the generic repair lane resolves: {issue_text}"
    if lane in {"evidence_followup", "method_repair"}:
        return f"Close only when new evidence or protocol detail directly resolves: {issue_text}"
    if lane == "figure_repair":
        return f"Close only when the updated figure package is evidence-aligned and no longer triggers: {issue_text}"
    if lane in {"claim_repair", "section_rewrite"}:
        return f"Close only when the rewritten manuscript text explicitly removes the blocker: {issue_text}"
    return "Close only when the issue has an owner, target, and verification path."


def _build_closure_evidence_refs(task: dict[str, Any]) -> list[str]:
    refs = [str(task.get("issue_id") or "").strip()]
    refs.extend(f"claim:{item}" for item in _coerce_list(task.get("claim_ids")))
    refs.extend(f"figure:{item}" for item in _coerce_list(task.get("figure_ids")))
    refs.extend(f"section:{item}" for item in _coerce_list(task.get("section_ids")))
    target_type = str(task.get("primary_target_type") or "").strip()
    target_id = str(task.get("primary_target_id") or "").strip()
    if target_type and target_id:
        refs.append(f"{target_type}:{target_id}")
    return _dedupe(refs)


def _build_retry_budget(task: dict[str, Any], lane: str) -> int:
    priority_tier = str(task.get("priority_tier") or "p2").strip().lower()
    if lane in {"evidence_followup", "method_repair"}:
        return 1
    if priority_tier == "p0":
        return 1
    if priority_tier == "p1":
        return 2
    return 2


def _build_escalation_lane(task: dict[str, Any], lane: str) -> str:
    if str(task.get("review_lane") or "").strip() == "hostile_critic":
        return "hostile_critic"
    if lane == "generic_rewrite":
        return "reviewer_board"
    if lane in {"evidence_followup", "method_repair"}:
        return "reviewer_board"
    if lane in {"claim_repair", "section_rewrite", "figure_repair"}:
        return "quality_gate"
    return "planner"


def build_repair_plan(
    project_root: str | Path,
    *,
    review_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = Path(project_root).expanduser().resolve()
    state = review_state
    if not isinstance(state, dict):
        state = load_contract_artifact(resolved_root, "review_state", default={}) or {}
    manifest = load_pipeline_manifest(resolved_root)
    repair_queue = [
        item for item in (state.get("repair_queue") or []) if isinstance(item, dict)
    ]
    role_summaries = (
        state.get("role_summaries") if isinstance(state.get("role_summaries"), dict) else {}
    )
    tasks: list[dict[str, Any]] = []
    lane_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for idx, queue_item in enumerate(repair_queue):
        lane = _infer_lane(queue_item)
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        status = str(queue_item.get("status") or "ready").strip() or "ready"
        status_counts[status] = status_counts.get(status, 0) + 1
        priority_tier = str(queue_item.get("priority_tier") or "p2").strip() or "p2"
        priority_counts[priority_tier] = priority_counts.get(priority_tier, 0) + 1
        issue_id = str(queue_item.get("issue_id") or f"issue_{idx}").strip()
        execution_steps = _build_execution_steps(queue_item, lane)
        verification_checks = _coerce_list(queue_item.get("verification_checks"))
        success_criteria = _build_success_criteria(queue_item, lane)
        blocking_reasons = _coerce_list(queue_item.get("blocking_reasons"))
        owner, owner_reason = _infer_owner(queue_item, lane)
        required_inputs = _build_required_inputs(queue_item, lane)
        produced_artifacts = _build_produced_artifacts(queue_item, lane)
        verifier = _build_verifier(queue_item, lane)
        close_condition = _build_close_condition(queue_item, lane)
        closure_evidence_refs = _build_closure_evidence_refs(queue_item)
        retry_budget = _build_retry_budget(queue_item, lane)
        escalation_lane = _build_escalation_lane(queue_item, lane)
        depends_on: list[str] = []
        if "missing_target_binding" in blocking_reasons:
            depends_on.append("target_binding")
        if "missing_repair_actions" in blocking_reasons:
            depends_on.append("repair_actions")
        if "missing_verification_checks" in blocking_reasons:
            depends_on.append("verification_checks")
        tasks.append(
            {
                "task_id": f"repair_task_{idx}",
                "repair_id": queue_item.get("repair_id") or f"RPR-{issue_id}",
                "issue_id": issue_id,
                "lane": lane,
                "lane_label": LANE_LABELS.get(lane, lane.replace("_", " ").title()),
                "status": status,
                "priority_tier": priority_tier,
                "priority_score": int(queue_item.get("priority_score") or 0),
                "role": queue_item.get("role"),
                "severity": queue_item.get("severity"),
                "review_lane": queue_item.get("review_lane"),
                "strictness_profile": queue_item.get("strictness_profile"),
                "blocker_class": queue_item.get("blocker_class"),
                "primary_target_type": queue_item.get("primary_target_type"),
                "primary_target_id": queue_item.get("primary_target_id"),
                "primary_target_label": queue_item.get("primary_target_label"),
                "owner": owner,
                "owner_reason": owner_reason,
                "claim_ids": _coerce_list(queue_item.get("claim_ids")),
                "figure_ids": _coerce_list(queue_item.get("figure_ids")),
                "section_ids": _coerce_list(queue_item.get("section_ids")),
                "depends_on": depends_on,
                "execution_steps": execution_steps,
                "success_criteria": success_criteria,
                "verification_checks": verification_checks,
                "required_inputs": required_inputs,
                "produced_artifacts": produced_artifacts,
                "verifier": verifier,
                "close_condition": close_condition,
                "closure_evidence_refs": closure_evidence_refs,
                "retry_budget": retry_budget,
                "escalation_lane": escalation_lane,
                "blocking_reasons": blocking_reasons,
            }
        )

    lane_rows: list[dict[str, Any]] = []
    for lane_name, count in sorted(lane_counts.items()):
        lane_tasks = [item for item in tasks if item.get("lane") == lane_name]
        lane_rows.append(
            {
                "lane": lane_name,
                "label": LANE_LABELS.get(lane_name, lane_name.replace("_", " ").title()),
                "task_count": count,
                "ready_count": sum(
                    str(item.get("status") or "") == "ready" for item in lane_tasks
                ),
                "priority_counts": {
                    tier: sum(
                        str(item.get("priority_tier") or "") == tier
                        for item in lane_tasks
                    )
                    for tier in ("p0", "p1", "p2")
                },
            }
        )

    task_count = len(tasks)
    ready_task_count = sum(str(item.get("status") or "") == "ready" for item in tasks)
    verification_ready_count = sum(
        bool(_coerce_list(item.get("verification_checks"))) for item in tasks
    )
    executable_ready_count = sum(
        bool(item.get("required_inputs"))
        and bool(item.get("produced_artifacts"))
        and bool(str(item.get("verifier") or "").strip())
        and bool(str(item.get("close_condition") or "").strip())
        for item in tasks
    )
    verifier_ready_count = sum(bool(str(item.get("verifier") or "").strip()) for item in tasks)
    targeted_count = sum(
        bool(str(item.get("primary_target_id") or "").strip())
        or str(item.get("primary_target_type") or "").strip() == "section"
        for item in tasks
    )
    escalation_counts = {
        lane_name: sum(str(item.get("escalation_lane") or "") == lane_name for item in tasks)
        for lane_name in sorted(
            {
                str(item.get("escalation_lane") or "").strip()
                for item in tasks
                if str(item.get("escalation_lane") or "").strip()
            }
        )
    }
    execution_board = [
        {
            "lane": lane_name,
            "label": LANE_LABELS.get(lane_name, lane_name.replace("_", " ").title()),
            "task_ids": [
                str(item.get("task_id"))
                for item in tasks
                if str(item.get("lane") or "") == lane_name
            ],
            "escalates_to": sorted(
                {
                    str(item.get("escalation_lane") or "").strip()
                    for item in tasks
                    if str(item.get("lane") or "") == lane_name
                    and str(item.get("escalation_lane") or "").strip()
                }
            ),
        }
        for lane_name in [row.get("lane") for row in lane_rows]
    ]
    summary = {
        "task_count": task_count,
        "ready_task_count": ready_task_count,
        "blocked_task_count": sum(
            str(item.get("status") or "") != "ready" for item in tasks
        ),
        "verification_ready_count": verification_ready_count,
        "verifier_ready_count": verifier_ready_count,
        "executable_ready_count": executable_ready_count,
        "targeted_task_count": targeted_count,
        "lane_count": len(lane_rows),
        "lane_counts": lane_counts,
        "status_counts": status_counts,
        "priority_counts": priority_counts,
        "escalation_counts": escalation_counts,
        "owner_counts": {
            owner: sum(str(item.get("owner") or "") == owner for item in tasks)
            for owner in sorted({str(item.get("owner") or "") for item in tasks if str(item.get("owner") or "")})
        },
        "ready_rate": round(ready_task_count / max(task_count, 1), 3)
        if task_count
        else 1.0,
        "verification_ready_rate": round(
            verification_ready_count / max(task_count, 1), 3
        )
        if task_count
        else 1.0,
        "verifier_ready_rate": round(verifier_ready_count / max(task_count, 1), 3)
        if task_count
        else 1.0,
        "executable_ready_rate": round(executable_ready_count / max(task_count, 1), 3)
        if task_count
        else 1.0,
        "targeted_rate": round(targeted_count / max(task_count, 1), 3)
        if task_count
        else 1.0,
    }
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "project_root": str(resolved_root),
        "workflow_mode": manifest.get("workflow_mode"),
        "workflow_label": manifest.get("workflow_label"),
        "owner_aware_routing_disabled": _owner_aware_repair_disabled(),
        "review_round_count": len(
            [item for item in (state.get("rounds") or []) if isinstance(item, dict)]
        ),
        "active_issue_count": len(
            [item for item in (state.get("active_issue_records") or []) if isinstance(item, dict)]
        ),
        "role_count": len(role_summaries),
        "lanes": lane_rows,
        "execution_board": execution_board,
        "tasks": tasks,
        "summary": summary,
    }


def save_repair_plan(
    project_root: str | Path,
    *,
    review_state: dict[str, Any] | None = None,
    producer: str = "review_repair_planner",
) -> str:
    payload = build_repair_plan(project_root, review_state=review_state)
    return save_contract_artifact(
        project_root,
        "repair_plan",
        payload,
        producer=producer,
        depends_on=[
            "review_state",
            "manuscript_state",
            "figure_spec",
            "claim_evidence_graph",
        ],
    )
