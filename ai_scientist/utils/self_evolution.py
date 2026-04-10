from __future__ import annotations

"""Structured self-evolution artifacts and playbooks for reviewer-driven repair."""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    load_contract_artifact,
    load_json_artifact,
    load_jsonl_artifact,
    load_pipeline_manifest,
    save_contract_artifact,
    save_json_artifact,
)
from ai_scientist.utils.review_jobs import compute_review_repair_metrics


LANE_DEFAULTS = {
    "figure_repair": {
        "stage": "figure",
        "risk": "figure_traceability_gap",
        "action": "Bind every figure to a claim, source record, and verifiable data file before rewrite.",
    },
    "claim_repair": {
        "stage": "planning",
        "risk": "claim_evidence_gap",
        "action": "Tighten claim scope and explicitly map each claim to evidence before drafting the final narrative.",
    },
    "section_rewrite": {
        "stage": "manuscript",
        "risk": "section_clarity_gap",
        "action": "Rewrite the affected section with reviewer-targeted closure language and explicit evidence references.",
    },
    "evidence_followup": {
        "stage": "experiment",
        "risk": "evidence_depth_gap",
        "action": "Front-load baseline, ablation, robustness, or significance checks before another writing pass.",
    },
    "method_repair": {
        "stage": "manuscript",
        "risk": "reproducibility_gap",
        "action": "Document setup, data, and implementation details so another team can reproduce the result without hidden assumptions.",
    },
    "triage": {
        "stage": "review",
        "risk": "repair_ownership_gap",
        "action": "Convert unbound reviewer feedback into targeted repair tasks with an owner, target, and verification path.",
    },
}

ROLE_DEFAULTS = {
    "novelty": {
        "stage": "planning",
        "risk": "novelty_positioning_gap",
        "action": "Strengthen novelty framing against related work before expanding downstream execution.",
    },
    "rigor": {
        "stage": "experiment",
        "risk": "rigor_validation_gap",
        "action": "Raise experiment rigor with stronger controls, broader baselines, and clearer acceptance checks.",
    },
    "clarity": {
        "stage": "manuscript",
        "risk": "clarity_gap",
        "action": "Improve framing, organization, and section-level storytelling before another review round.",
    },
    "reproducibility": {
        "stage": "manuscript",
        "risk": "reproducibility_gap",
        "action": "Make the method, setup, and artifacts reproducible enough for an independent team to rerun them.",
    },
}


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


def _finalize_self_check(criteria: list[dict[str, Any]]) -> dict[str, Any]:
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
        "status": status,
        "score": score,
        "criteria": criteria,
        "required_failures": required_failures,
    }


def _priority_for_count(count: int) -> str:
    if count >= 3:
        return "p0"
    if count >= 2:
        return "p1"
    return "p2"


def _infer_queue_lane(task: dict[str, Any]) -> str:
    primary_target_type = str(task.get("primary_target_type") or "").strip()
    role = str(task.get("role") or "").strip()
    issue_text = str(task.get("issue_text") or task.get("text") or "").lower()
    if primary_target_type == "figure":
        return "figure_repair"
    if primary_target_type == "claim":
        return "claim_repair"
    if primary_target_type == "section":
        return "section_rewrite"
    if role == "reproducibility":
        return "method_repair"
    if any(
        token in issue_text
        for token in (
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
    ):
        return "evidence_followup"
    return "triage"


def _resolve_knowledge_dir(project_root: str | Path) -> Path:
    resolved_root = Path(project_root).expanduser().resolve()
    parent = resolved_root.parent
    if parent.name in {"projects", "papers", "batches"}:
        research_root = parent.parent
    else:
        research_root = parent
    return research_root / "knowledge_base"


def _build_lane_lessons(
    lane_counts: Counter[str],
    *,
    lane_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lessons: list[dict[str, Any]] = []
    lane_by_name = {
        str(item.get("lane") or "").strip(): item
        for item in lane_rows
        if isinstance(item, dict) and str(item.get("lane") or "").strip()
    }
    for lane, count in lane_counts.most_common():
        defaults = LANE_DEFAULTS.get(lane, LANE_DEFAULTS["triage"])
        lane_row = lane_by_name.get(lane, {})
        lessons.append(
            {
                "lesson_id": f"lane::{lane}",
                "source": "repair_plan",
                "priority_tier": _priority_for_count(count),
                "stage": defaults["stage"],
                "focus": lane,
                "risk": defaults["risk"],
                "signal": f"{count} repair tasks routed to {lane}",
                "recommended_action": defaults["action"],
                "agentic_default_update": defaults["action"],
                "ready_rate": float(lane_row.get("ready_count") or 0) / max(
                    int(lane_row.get("task_count") or count), 1
                ),
            }
        )
    return lessons


def _build_role_lessons(role_counts: Counter[str]) -> list[dict[str, Any]]:
    lessons: list[dict[str, Any]] = []
    for role, count in role_counts.most_common():
        defaults = ROLE_DEFAULTS.get(role)
        if not defaults:
            continue
        lessons.append(
            {
                "lesson_id": f"role::{role}",
                "source": "review_state",
                "priority_tier": _priority_for_count(count),
                "stage": defaults["stage"],
                "focus": role,
                "risk": defaults["risk"],
                "signal": f"{count} reviewer issues came from the {role} role",
                "recommended_action": defaults["action"],
                "agentic_default_update": defaults["action"],
            }
        )
    return lessons


def _build_metric_lessons(
    review_metrics: dict[str, Any],
    *,
    blocked_stage_count: int,
    top_stage_risks: list[str],
) -> list[dict[str, Any]]:
    lessons: list[dict[str, Any]] = []
    active_binding = float(review_metrics.get("active_binding_coverage") or 0.0)
    verification_coverage = float(review_metrics.get("verification_coverage") or 0.0)
    resolution_rate = float(review_metrics.get("resolution_rate") or 0.0)
    persistent_issue_count = int(review_metrics.get("persistent_issue_count") or 0)

    if active_binding < 1.0:
        lessons.append(
            {
                "lesson_id": "metric::binding",
                "source": "review_metrics",
                "priority_tier": "p0" if active_binding < 0.75 else "p1",
                "stage": "review",
                "focus": "binding",
                "risk": "issue_binding_gap",
                "signal": f"active reviewer binding coverage is {active_binding:.2f}",
                "recommended_action": "Bind every active reviewer issue to a concrete claim, figure, or section before another repair round.",
                "agentic_default_update": "Require issue-to-target binding before queuing repair execution.",
            }
        )
    if verification_coverage < 1.0:
        lessons.append(
            {
                "lesson_id": "metric::verification",
                "source": "review_metrics",
                "priority_tier": "p0" if verification_coverage < 0.6 else "p1",
                "stage": "review",
                "focus": "verification",
                "risk": "verification_path_gap",
                "signal": f"review verification coverage is {verification_coverage:.2f}",
                "recommended_action": "Require an explicit verification check for each repair lane before treating reviewer debt as actionable.",
                "agentic_default_update": "Block ready-to-run repairs that do not define a verification path.",
            }
        )
    if persistent_issue_count > 0 and resolution_rate < 1.0:
        lessons.append(
            {
                "lesson_id": "metric::persistence",
                "source": "review_metrics",
                "priority_tier": "p0" if persistent_issue_count >= 2 else "p1",
                "stage": "review",
                "focus": "persistence",
                "risk": "persistent_reviewer_debt",
                "signal": (
                    f"{persistent_issue_count} persistent reviewer issues remain; "
                    f"resolution_rate={resolution_rate:.2f}"
                ),
                "recommended_action": "Escalate persistent reviewer issues into explicit repair lanes and keep them open until a verification check passes.",
                "agentic_default_update": "Raise reviewer debt as a scheduling pressure rather than treating it as passive commentary.",
            }
        )
    if blocked_stage_count > 0:
        lessons.append(
            {
                "lesson_id": "metric::stage_blockers",
                "source": "stage_standards",
                "priority_tier": "p0",
                "stage": "planning",
                "focus": "stage_blockers",
                "risk": "stage_standard_blocker",
                "signal": f"{blocked_stage_count} blocked stage standards: {', '.join(top_stage_risks[:3]) or 'unspecified'}",
                "recommended_action": "Route resources to the blocked stages before investing in more open-ended generation.",
                "agentic_default_update": "Use stage-standard blockers as a hard gate for the next batch plan.",
            }
        )
    return lessons


def _build_next_cycle_defaults(lessons: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for lesson in lessons:
        stage = str(lesson.get("stage") or "").strip()
        action = str(lesson.get("agentic_default_update") or "").strip()
        if not stage or not action:
            continue
        grouped.setdefault(stage, [])
        grouped[stage].append(action)
    return {
        stage: _dedupe(actions)
        for stage, actions in grouped.items()
        if actions
    }


def build_self_evolution(
    project_root: str | Path,
    *,
    review_state: dict[str, Any] | None = None,
    repair_plan: dict[str, Any] | None = None,
    stage_standards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = Path(project_root).expanduser().resolve()
    manifest = load_pipeline_manifest(resolved_root)
    review_state_payload = (
        review_state
        if isinstance(review_state, dict)
        else load_contract_artifact(resolved_root, "review_state", default={}) or {}
    )
    repair_plan_payload = (
        repair_plan
        if isinstance(repair_plan, dict)
        else load_contract_artifact(resolved_root, "repair_plan", default={}) or {}
    )
    standards_payload = (
        stage_standards
        if isinstance(stage_standards, dict)
        else load_contract_artifact(resolved_root, "stage_standards", default={}) or {}
    )
    review_metrics = review_state_payload.get("repair_metrics")
    if not isinstance(review_metrics, dict):
        review_metrics = compute_review_repair_metrics(review_state_payload)
    repair_plan_summary = (
        repair_plan_payload.get("summary")
        if isinstance(repair_plan_payload.get("summary"), dict)
        else {}
    )
    standards_summary = (
        standards_payload.get("summary")
        if isinstance(standards_payload.get("summary"), dict)
        else {}
    )
    issue_ledger = [
        item
        for item in (review_state_payload.get("issue_ledger") or [])
        if isinstance(item, dict)
    ]
    if not issue_ledger:
        for field in (
            "active_issue_records",
            "resolved_issue_records",
            "persistent_issue_records",
        ):
            issue_ledger.extend(
                item
                for item in (review_state_payload.get(field) or [])
                if isinstance(item, dict)
            )
    role_counts = Counter(
        str(item.get("role") or "").strip()
        for item in issue_ledger
        if str(item.get("role") or "").strip()
    )
    lane_rows = [
        item for item in (repair_plan_payload.get("lanes") or []) if isinstance(item, dict)
    ]
    lane_counts = Counter(
        str(item.get("lane") or "").strip()
        for item in lane_rows
        if str(item.get("lane") or "").strip()
    )
    if not lane_counts:
        lane_counts = Counter(
            str(item.get("lane") or "").strip()
            for item in (repair_plan_payload.get("tasks") or [])
            if isinstance(item, dict) and str(item.get("lane") or "").strip()
        )
    if not lane_counts:
        lane_counts = Counter(
            _infer_queue_lane(item)
            for item in (review_state_payload.get("repair_queue") or [])
            if isinstance(item, dict)
        )

    active_issue_count = int(review_metrics.get("active_issue_count") or 0)
    repair_ready_coverage = float(review_metrics.get("repair_ready_coverage") or 0.0)
    active_binding_coverage = float(
        review_metrics.get("active_binding_coverage") or 0.0
    )
    repair_targeted_coverage = float(
        review_metrics.get("repair_targeted_coverage") or 0.0
    )
    verification_ready_rate = float(
        repair_plan_summary.get("verification_ready_rate")
        or review_metrics.get("repair_verification_ready_coverage")
        or 0.0
    )
    persistent_issue_count = int(review_metrics.get("persistent_issue_count") or 0)
    blocked_stage_count = int(standards_payload.get("blocked_stage_count") or 0)
    criteria = [
        _criterion(
            "repair_ready_coverage",
            "Active reviewer issues are translated into ready repair tasks",
            passed=active_issue_count == 0 or repair_ready_coverage >= 1.0,
            detail=f"ready_coverage={repair_ready_coverage:.2f}",
        ),
        _criterion(
            "repair_targeting",
            "Active reviewer issues are bound to concrete repair targets",
            passed=active_issue_count == 0 or repair_targeted_coverage >= 1.0,
            detail=f"targeted_coverage={repair_targeted_coverage:.2f}",
        ),
        _criterion(
            "active_binding",
            "Bound reviewer issues maintain explicit claim/figure/section linkage",
            passed=active_issue_count == 0 or active_binding_coverage >= 1.0,
            detail=f"active_binding_coverage={active_binding_coverage:.2f}",
        ),
        _criterion(
            "verification_path",
            "Repair tasks expose verification checks",
            passed=active_issue_count == 0 or verification_ready_rate >= 0.5,
            detail=f"verification_ready_rate={verification_ready_rate:.2f}",
        ),
        _criterion(
            "persistent_issue_control",
            "Persistent reviewer debt is under control",
            passed=persistent_issue_count == 0
            or float(review_metrics.get("resolution_rate") or 0.0) >= 0.5,
            detail=(
                f"persistent={persistent_issue_count} "
                f"resolution_rate={float(review_metrics.get('resolution_rate') or 0.0):.2f}"
            ),
            required=False,
        ),
        _criterion(
            "stage_blockers",
            "No blocked stage standards remain after repair planning",
            passed=blocked_stage_count == 0,
            detail=f"blocked_stage_count={blocked_stage_count}",
            required=bool(standards_payload),
        ),
    ]
    self_check = _finalize_self_check(criteria)
    lessons = _build_lane_lessons(lane_counts, lane_rows=lane_rows)
    lessons.extend(_build_role_lessons(role_counts))
    lessons.extend(
        _build_metric_lessons(
            review_metrics,
            blocked_stage_count=blocked_stage_count,
            top_stage_risks=_coerce_list(standards_summary.get("top_risks")),
        )
    )
    lessons.sort(
        key=lambda item: (
            {"p0": 0, "p1": 1, "p2": 2}.get(str(item.get("priority_tier") or "p2"), 9),
            str(item.get("stage") or ""),
            str(item.get("lesson_id") or ""),
        )
    )
    next_cycle_defaults = _build_next_cycle_defaults(lessons)
    dominant_lane = lane_counts.most_common(1)[0][0] if lane_counts else None
    dominant_role = role_counts.most_common(1)[0][0] if role_counts else None
    summary = {
        "status": self_check["status"],
        "score": self_check["score"],
        "criterion_count": len(criteria),
        "required_failure_count": len(self_check["required_failures"]),
        "active_issue_count": active_issue_count,
        "resolved_issue_count": int(review_metrics.get("resolved_issue_count") or 0),
        "persistent_issue_count": persistent_issue_count,
        "repair_plan_task_count": int(repair_plan_summary.get("task_count") or 0),
        "repair_lane_count": len(lane_counts),
        "dominant_lane": dominant_lane,
        "dominant_role": dominant_role,
        "resolution_rate": float(review_metrics.get("resolution_rate") or 0.0),
        "verification_coverage": float(
            review_metrics.get("verification_coverage") or 0.0
        ),
        "active_binding_coverage": active_binding_coverage,
        "repair_ready_coverage": repair_ready_coverage,
        "repair_targeted_coverage": repair_targeted_coverage,
        "verification_ready_rate": verification_ready_rate,
        "blocked_stage_count": blocked_stage_count,
        "lesson_count": len(lessons),
    }
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "project_root": str(resolved_root),
        "workflow_mode": manifest.get("workflow_mode"),
        "workflow_label": manifest.get("workflow_label"),
        "summary": summary,
        "self_check": self_check,
        "lane_counts": dict(lane_counts),
        "role_counts": dict(role_counts),
        "next_cycle_defaults": next_cycle_defaults,
        "lessons": lessons,
        "stage_risks": _coerce_list(standards_summary.get("top_risks")),
        "blocked_stages": _coerce_list(standards_summary.get("blocked_stages")),
    }


def _snapshot_for_history(evolution_payload: dict[str, Any]) -> dict[str, Any]:
    summary = evolution_payload.get("summary") if isinstance(evolution_payload.get("summary"), dict) else {}
    lessons = [
        item for item in (evolution_payload.get("lessons") or []) if isinstance(item, dict)
    ]
    return {
        "generated_at": evolution_payload.get("generated_at"),
        "project_root": evolution_payload.get("project_root"),
        "workflow_mode": evolution_payload.get("workflow_mode"),
        "workflow_label": evolution_payload.get("workflow_label"),
        "status": summary.get("status"),
        "score": summary.get("score"),
        "dominant_lane": summary.get("dominant_lane"),
        "dominant_role": summary.get("dominant_role"),
        "blocked_stage_count": summary.get("blocked_stage_count"),
        "active_issue_count": summary.get("active_issue_count"),
        "persistent_issue_count": summary.get("persistent_issue_count"),
        "resolution_rate": summary.get("resolution_rate"),
        "lane_counts": evolution_payload.get("lane_counts") or {},
        "role_counts": evolution_payload.get("role_counts") or {},
        "stage_risks": evolution_payload.get("stage_risks") or [],
        "next_cycle_defaults": evolution_payload.get("next_cycle_defaults") or {},
        "top_lessons": lessons[:5],
    }


def _latest_snapshots(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_project: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        project_root = str(entry.get("project_root") or "").strip()
        if not project_root:
            continue
        previous = latest_by_project.get(project_root)
        timestamp = str(entry.get("generated_at") or "")
        if not previous or timestamp >= str(previous.get("generated_at") or ""):
            latest_by_project[project_root] = entry
    return list(latest_by_project.values())


def build_self_evolution_playbook(entries: list[dict[str, Any]]) -> dict[str, Any]:
    latest_entries = _latest_snapshots(entries)
    status_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    stage_focus_counts: Counter[str] = Counter()
    action_counts: Counter[tuple[str, str]] = Counter()
    for entry in latest_entries:
        status = str(entry.get("status") or "").strip()
        if status:
            status_counts[status] += 1
        for name, count in (entry.get("lane_counts") or {}).items():
            lane_counts[str(name)] += int(count or 0)
        for name, count in (entry.get("role_counts") or {}).items():
            role_counts[str(name)] += int(count or 0)
        for risk in _coerce_list(entry.get("stage_risks")):
            risk_counts[risk] += 1
        for stage, actions in (entry.get("next_cycle_defaults") or {}).items():
            stage_focus_counts[str(stage)] += len(_coerce_list(actions))
            for action in _coerce_list(actions):
                action_counts[(str(stage), action)] += 1
        for lesson in entry.get("top_lessons") or []:
            if not isinstance(lesson, dict):
                continue
            risk = str(lesson.get("risk") or "").strip()
            stage = str(lesson.get("stage") or "").strip()
            action = str(lesson.get("agentic_default_update") or "").strip()
            if risk:
                risk_counts[risk] += 1
            if stage and action:
                action_counts[(stage, action)] += 1
                stage_focus_counts[stage] += 1
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "project_count": len(latest_entries),
        "history_entry_count": len(entries),
        "status_counts": dict(status_counts),
        "lane_counts": dict(lane_counts),
        "role_counts": dict(role_counts),
        "stage_focus_counts": dict(stage_focus_counts),
        "top_recurring_risks": [
            {"risk": name, "count": count}
            for name, count in risk_counts.most_common(8)
        ],
        "top_agentic_defaults": [
            {"stage": stage, "action": action, "count": count}
            for (stage, action), count in action_counts.most_common(8)
        ],
        "latest_projects": [
            {
                "project_root": item.get("project_root"),
                "workflow_mode": item.get("workflow_mode"),
                "status": item.get("status"),
                "score": item.get("score"),
                "dominant_lane": item.get("dominant_lane"),
            }
            for item in sorted(
                latest_entries,
                key=lambda row: (str(row.get("generated_at") or ""), str(row.get("project_root") or "")),
                reverse=True,
            )[:10]
        ],
    }


def save_self_evolution(
    project_root: str | Path,
    *,
    review_state: dict[str, Any] | None = None,
    repair_plan: dict[str, Any] | None = None,
    stage_standards: dict[str, Any] | None = None,
    producer: str = "self_evolution",
) -> str:
    payload = build_self_evolution(
        project_root,
        review_state=review_state,
        repair_plan=repair_plan,
        stage_standards=stage_standards,
    )
    output_path = save_contract_artifact(
        project_root,
        "self_evolution",
        payload,
        producer=producer,
        depends_on=["review_state", "repair_plan"],
    )
    knowledge_dir = _resolve_knowledge_dir(project_root)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    history_path = knowledge_dir / "self_evolution_history.jsonl"
    append_jsonl_artifact(history_path, _snapshot_for_history(payload))
    history_entries = load_jsonl_artifact(history_path)
    playbook = build_self_evolution_playbook(history_entries)
    save_json_artifact(knowledge_dir / "self_evolution_playbook.json", playbook)
    return output_path


def load_self_evolution_playbook(project_root: str | Path) -> dict[str, Any]:
    knowledge_dir = _resolve_knowledge_dir(project_root)
    return load_json_artifact(knowledge_dir / "self_evolution_playbook.json", default={}) or {}
