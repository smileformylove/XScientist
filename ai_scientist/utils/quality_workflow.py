from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Callable, Optional

from ai_scientist.utils.high_quality_pipeline import (
    QUALITY_PRESETS,
    evaluate_submission_acceptance,
    resolve_submission_acceptance_settings,
)
from ai_scientist.utils.pipeline_contracts import load_contract_artifact

QUALITY_PRESET_ORDER = ("balanced", "high", "publishable")
AUTONOMOUS_STYLE_ORDER = ("conservative", "professional", "assertive")
FOLLOWUP_FOCUS_PROFILES = {
    "narrative_quality": {
        "preferred_sections": ["title", "abstract", "introduction", "conclusion"],
        "frontmatter_required": True,
        "candidate_boost": 1,
        "target_section_limit": 5,
        "frontmatter_style_order": ["professional", "conservative", "assertive"],
        "section_style_order": ["professional", "conservative", "assertive"],
        "notes": [
            "Tighten the paper story around a small set of concrete contributions.",
            "Reduce diffuse framing and make the main value proposition legible in frontmatter.",
        ],
    },
    "rigor": {
        "preferred_sections": ["method", "results", "analysis", "discussion"],
        "frontmatter_required": False,
        "candidate_boost": 1,
        "target_section_limit": 5,
        "frontmatter_style_order": ["professional", "conservative", "assertive"],
        "section_style_order": ["professional", "conservative", "assertive"],
        "rebuttal_focus": [
            "Preempt reviewer concerns about baselines, ablations, statistics, and reproducibility.",
        ],
        "anticipated_objections": [
            "Reviewers may question whether the evaluation protocol and ablations are strong enough.",
        ],
        "notes": [
            "Bias the rewrite toward methodology clarity, controls, and evidence-backed rigor claims.",
        ],
    },
    "claim_support": {
        "preferred_sections": ["abstract", "results", "discussion", "conclusion"],
        "frontmatter_required": True,
        "candidate_boost": 2,
        "target_section_limit": 6,
        "frontmatter_style_order": ["conservative", "professional", "assertive"],
        "section_style_order": ["professional", "conservative", "assertive"],
        "claim_softening_advice": [
            "Anchor every major claim to one figure/table or citation and avoid unsupported superiority language.",
        ],
        "limitation_emphasis": [
            "Add an explicit scope caveat whenever the evidence does not justify broad generalization.",
        ],
        "rebuttal_focus": [
            "Connect each strong claim to a concrete experimental artifact before broadening the statement.",
        ],
        "anticipated_objections": [
            "Reviewers may flag claims that are stronger than the cited evidence or numeric support.",
        ],
        "notes": [
            "Treat unsupported claims as first-class blockers and rewrite them into evidence-backed statements.",
        ],
    },
    "numeric_coverage": {
        "preferred_sections": ["abstract", "results", "conclusion"],
        "frontmatter_required": True,
        "candidate_boost": 1,
        "target_section_limit": 5,
        "frontmatter_style_order": ["professional", "conservative", "assertive"],
        "section_style_order": ["professional", "conservative", "assertive"],
        "rebuttal_focus": [
            "Inject one or two key numerical comparisons into every major claim path.",
        ],
        "notes": [
            "Make the strongest quantitative deltas explicit in frontmatter and takeaway sections.",
        ],
    },
    "evidence_packaging": {
        "preferred_sections": ["results", "analysis", "discussion", "conclusion"],
        "frontmatter_required": True,
        "candidate_boost": 1,
        "target_section_limit": 6,
        "frontmatter_style_order": ["professional", "conservative", "assertive"],
        "section_style_order": ["professional", "conservative", "assertive"],
        "rebuttal_focus": [
            "Surface the lead figure/table earlier and explicitly reference it in the surrounding text.",
        ],
        "anticipated_objections": [
            "Reviewers may see the manuscript as under-evidenced if visuals and tables are not integrated into the narrative.",
        ],
        "notes": [
            "Prioritize sections that should cite figures/tables and strongest results more aggressively.",
        ],
    },
    "contribution_framing": {
        "preferred_sections": ["title", "abstract", "introduction", "conclusion"],
        "frontmatter_required": True,
        "candidate_boost": 1,
        "target_section_limit": 5,
        "frontmatter_style_order": ["assertive", "professional", "conservative"],
        "section_style_order": ["professional", "assertive", "conservative"],
        "notes": [
            "Make each contribution explicit, traceable to evidence, and appropriately scoped.",
        ],
    },
    "breakthrough_significance": {
        "preferred_sections": ["abstract", "introduction", "discussion", "conclusion"],
        "frontmatter_required": True,
        "candidate_boost": 1,
        "target_section_limit": 5,
        "frontmatter_style_order": ["assertive", "professional", "conservative"],
        "section_style_order": ["assertive", "professional", "conservative"],
        "claim_softening_advice": [
            "Push the significance story only where the evidence can sustain it; avoid overselling breadth.",
        ],
        "limitation_emphasis": [
            "Pair any broad-significance framing with a precise statement of the evaluated scope.",
        ],
        "notes": [
            "Raise the significance framing while keeping claims evidence-backed and venue-appropriate.",
        ],
    },
    "venue_fit": {
        "preferred_sections": ["title", "abstract", "introduction", "discussion"],
        "frontmatter_required": True,
        "candidate_boost": 1,
        "target_section_limit": 4,
        "frontmatter_style_order": ["conservative", "professional", "assertive"],
        "section_style_order": ["conservative", "professional", "assertive"],
        "claim_softening_advice": [
            "Narrow the scope and framing so the manuscript reads like a good fit for the target venue.",
        ],
        "notes": [
            "Favor venue-fit repairs over broader ambition when fit itself is the blocker.",
        ],
    },
}


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _safe_load_json_dict(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dedupe_strings(values: list[Any], *, limit: Optional[int] = None) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in deduped:
            continue
        deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _merge_style_order(*orders: list[Any]) -> list[str]:
    merged: list[str] = []
    for order in orders:
        for item in order or []:
            text = str(item).strip().lower()
            if text not in AUTONOMOUS_STYLE_ORDER or text in merged:
                continue
            merged.append(text)
    for item in AUTONOMOUS_STYLE_ORDER:
        if item not in merged:
            merged.append(item)
    return merged


def _dedupe_reason_list(values: list[Any]) -> list[str]:
    return _dedupe_strings(values, limit=16)


def _load_saved_high_quality_result(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    return _safe_load_json_dict(root / "quality" / "high_quality_result.json")


def _load_final_submission_artifacts(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    return {
        "stage_standards": load_contract_artifact(root, "stage_standards", default={})
        or {},
        "review_state": load_contract_artifact(root, "review_state", default={}) or {},
        "repair_plan": load_contract_artifact(root, "repair_plan", default={}) or {},
        "self_evolution": load_contract_artifact(root, "self_evolution", default={})
        or {},
    }


def evaluate_final_submission_readiness(
    *,
    run_dir: str | Path,
    quality_result: Optional[dict[str, Any]] = None,
    require_quality_gate: bool,
    min_submission_priority: Optional[float],
    max_submission_blockers: Optional[int],
    reject_on_auto_improvement_fallback: bool = False,
    min_stage_overall_score: float = 80.0,
    max_blocked_stages: int = 0,
    min_self_evolution_score: float = 80.0,
    max_self_evolution_required_failures: int = 0,
    require_review_artifacts: bool = True,
    final_issue_progress: Optional[dict[str, Any]] = None,
    final_todo_snapshot: Optional[dict[str, Any]] = None,
    max_open_p0_todos: Optional[int] = 0,
) -> dict[str, Any]:
    artifacts = _load_final_submission_artifacts(run_dir)
    resolved_quality_result = (
        dict(quality_result)
        if isinstance(quality_result, dict) and quality_result
        else _load_saved_high_quality_result(run_dir)
    )

    reasons: list[str] = []
    accepted = True
    quality_acceptance = {
        "accepted": True,
        "reasons": [],
        "submission_priority_score": None,
        "submission_priority_tier": None,
        "blocker_count": None,
    }
    if resolved_quality_result:
        quality_acceptance = evaluate_submission_acceptance(
            resolved_quality_result,
            require_quality_gate=require_quality_gate,
            min_submission_priority=min_submission_priority,
            max_submission_blockers=max_submission_blockers,
            reject_on_auto_improvement_fallback=reject_on_auto_improvement_fallback,
        )
        if not quality_acceptance["accepted"]:
            accepted = False
            reasons.extend(quality_acceptance["reasons"])
    elif (
        require_quality_gate
        or min_submission_priority is not None
        or max_submission_blockers is not None
        or reject_on_auto_improvement_fallback
    ):
        accepted = False
        reasons.append("final high-quality assessment artifact is missing")

    missing_artifacts = [
        name for name, payload in artifacts.items() if not isinstance(payload, dict) or not payload
    ]
    if require_review_artifacts and missing_artifacts:
        accepted = False
        reasons.append(
            "final review/self-optimization artifacts are missing: "
            + ", ".join(sorted(missing_artifacts))
        )

    stage_standards = (
        artifacts.get("stage_standards")
        if isinstance(artifacts.get("stage_standards"), dict)
        else {}
    )
    stage_summary = (
        stage_standards.get("summary")
        if isinstance(stage_standards.get("summary"), dict)
        else {}
    )
    stage_overall_score = _coerce_float(stage_standards.get("overall_score"))
    blocked_stage_count = _coerce_int(stage_standards.get("blocked_stage_count"))
    if stage_overall_score is not None and stage_overall_score < float(min_stage_overall_score):
        accepted = False
        reasons.append(
            f"stage standards overall score below target ({stage_overall_score:.1f} < {float(min_stage_overall_score):.1f})"
        )
    if (
        blocked_stage_count is not None
        and max_blocked_stages is not None
        and blocked_stage_count > max_blocked_stages
    ):
        top_risks = _dedupe_reason_list(stage_summary.get("top_risks") or [])
        detail = f"; top_risks={top_risks[:3]}" if top_risks else ""
        accepted = False
        reasons.append(
            f"blocked stage standards remain ({blocked_stage_count} > {max_blocked_stages}){detail}"
        )

    self_evolution = (
        artifacts.get("self_evolution")
        if isinstance(artifacts.get("self_evolution"), dict)
        else {}
    )
    evolution_summary = (
        self_evolution.get("summary")
        if isinstance(self_evolution.get("summary"), dict)
        else {}
    )
    evolution_self_check = (
        self_evolution.get("self_check")
        if isinstance(self_evolution.get("self_check"), dict)
        else {}
    )
    self_evolution_score = _coerce_float(
        evolution_summary.get("score")
        if evolution_summary.get("score") is not None
        else evolution_self_check.get("score")
    )
    self_evolution_required_failures = _coerce_int(
        evolution_summary.get("required_failure_count")
    )
    if self_evolution_required_failures is None:
        self_evolution_required_failures = len(
            evolution_self_check.get("required_failures") or []
        )
    self_evolution_status = str(
        evolution_summary.get("status") or evolution_self_check.get("status") or ""
    ).strip()
    if (
        self_evolution_score is not None
        and self_evolution_score < float(min_self_evolution_score)
    ):
        accepted = False
        reasons.append(
            f"self-evolution score below target ({self_evolution_score:.1f} < {float(min_self_evolution_score):.1f})"
        )
    if (
        max_self_evolution_required_failures is not None
        and self_evolution_required_failures is not None
        and self_evolution_required_failures > max_self_evolution_required_failures
    ):
        accepted = False
        reasons.append(
            "self-evolution still has required failures "
            f"({self_evolution_required_failures} > {max_self_evolution_required_failures})"
        )
    if self_evolution_status and self_evolution_status != "ready":
        accepted = False
        reasons.append(
            f"self-evolution status is not ready ({self_evolution_status})"
        )

    review_state = (
        artifacts.get("review_state")
        if isinstance(artifacts.get("review_state"), dict)
        else {}
    )
    review_metrics = (
        review_state.get("repair_metrics")
        if isinstance(review_state.get("repair_metrics"), dict)
        else {}
    )
    active_issue_count = _coerce_int(review_metrics.get("active_issue_count"))
    if active_issue_count is None:
        active_issue_count = len(review_state.get("active_issue_records") or [])
    lane_summaries = (
        review_state.get("lane_summaries")
        if isinstance(review_state.get("lane_summaries"), dict)
        else {}
    )
    hostile_critic_summary = (
        lane_summaries.get("hostile_critic")
        if isinstance(lane_summaries.get("hostile_critic"), dict)
        else {}
    )
    hostile_critic_active_issue_count = _coerce_int(
        hostile_critic_summary.get("active_issue_count")
    )
    hostile_critic_blocking_issue_count = _coerce_int(
        hostile_critic_summary.get("blocking_issue_count")
    )
    if hostile_critic_active_issue_count is not None and hostile_critic_active_issue_count > 0:
        accepted = False
        reasons.append(
            "independent hostile critic still reports active blockers "
            f"({hostile_critic_active_issue_count})"
        )

    repair_plan = (
        artifacts.get("repair_plan")
        if isinstance(artifacts.get("repair_plan"), dict)
        else {}
    )
    repair_plan_summary = (
        repair_plan.get("summary")
        if isinstance(repair_plan.get("summary"), dict)
        else {}
    )
    repair_execution_focus = _derive_repair_execution_focus(repair_plan, review_state)
    verification_ready_rate = _coerce_float(
        repair_plan_summary.get("verification_ready_rate")
    )
    if (
        active_issue_count is not None
        and active_issue_count > 0
        and verification_ready_rate is not None
        and verification_ready_rate < 1.0
    ):
        accepted = False
        reasons.append(
            "active reviewer issues remain without full verification-ready repair coverage "
            f"({verification_ready_rate:.2f} < 1.00)"
        )

    unresolved_critical_count = None
    persistent_issue_count = None
    if isinstance(final_issue_progress, dict):
        unresolved_critical_count = _coerce_int(
            final_issue_progress.get("unresolved_critical_count")
        )
        persistent_issue_count = _coerce_int(
            final_issue_progress.get("persistent_issue_count")
        )
        if unresolved_critical_count is not None and unresolved_critical_count > 0:
            accepted = False
            reasons.append(
                f"final self-review still has unresolved critical issues ({unresolved_critical_count})"
            )
        if persistent_issue_count is not None and persistent_issue_count > 0:
            accepted = False
            reasons.append(
                f"final self-review still has persistent issues ({persistent_issue_count})"
            )

    p0_unresolved = None
    if isinstance(final_todo_snapshot, dict):
        counts = (
            final_todo_snapshot.get("counts")
            if isinstance(final_todo_snapshot.get("counts"), dict)
            else {}
        )
        p0_unresolved = _coerce_int(counts.get("p0_unresolved"))
        if (
            p0_unresolved is not None
            and max_open_p0_todos is not None
            and p0_unresolved > max_open_p0_todos
        ):
            accepted = False
            reasons.append(
                f"experiment TODO still has unresolved P0 items ({p0_unresolved} > {max_open_p0_todos})"
            )

    return {
        "accepted": accepted,
        "reasons": _dedupe_reason_list(reasons),
        "quality_acceptance": quality_acceptance,
        "signals": {
            "stage_overall_score": stage_overall_score,
            "blocked_stage_count": blocked_stage_count,
            "active_issue_count": active_issue_count,
            "hostile_critic_active_issue_count": hostile_critic_active_issue_count,
            "hostile_critic_blocking_issue_count": hostile_critic_blocking_issue_count,
            "verification_ready_rate": verification_ready_rate,
            "repair_lane_order": repair_execution_focus.get("lane_order") or [],
            "repair_top_tasks": repair_execution_focus.get("top_tasks") or [],
            "hostile_recheck_required": repair_execution_focus.get(
                "hostile_recheck_required"
            ),
            "self_evolution_status": self_evolution_status or None,
            "self_evolution_score": self_evolution_score,
            "self_evolution_required_failure_count": self_evolution_required_failures,
            "unresolved_critical_count": unresolved_critical_count,
            "persistent_issue_count": persistent_issue_count,
            "p0_unresolved_todos": p0_unresolved,
        },
        "artifacts_present": {
            name: bool(isinstance(payload, dict) and payload)
            for name, payload in artifacts.items()
        },
    }


def _normalize_revision_actions(actions: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in actions or []:
        if not isinstance(item, dict):
            continue
        action = {
            "priority": str(item.get("priority") or "").strip() or "P2",
            "focus": str(item.get("focus") or "").strip(),
            "action": str(item.get("action") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
        }
        signature = (action["focus"], action["action"])
        if not action["focus"] and not action["action"]:
            continue
        if signature in {
            (entry.get("focus", ""), entry.get("action", "")) for entry in normalized
        }:
            continue
        normalized.append(action)
    return normalized


def _focus_area_from_text(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    if not lowered:
        return None
    if "claim" in lowered or "unsupported" in lowered:
        return "claim_support"
    if any(token in lowered for token in ["rigor", "baseline", "ablation", "methodology", "reproduc"]):
        return "rigor"
    if any(token in lowered for token in ["numeric", "number", "quantitative", "quantitative", "comparison"]):
        return "numeric_coverage"
    if any(token in lowered for token in ["evidence", "figure", "table", "visual", "caption"]):
        return "evidence_packaging"
    if "contribution" in lowered:
        return "contribution_framing"
    if any(token in lowered for token in ["breakthrough", "significance", "broad impact", "nature-style"]):
        return "breakthrough_significance"
    if any(token in lowered for token in ["venue", "paper_type", "fit for target"]):
        return "venue_fit"
    if any(token in lowered for token in ["narrative", "abstract", "introduction", "title", "framing", "quality"]):
        return "narrative_quality"
    return None


def _focus_area_from_scorecard_metric(metric_name: str) -> Optional[str]:
    metric = (metric_name or "").strip().lower()
    return {
        "quality": "narrative_quality",
        "rigor": "rigor",
        "claim": "claim_support",
        "claim_support": "claim_support",
        "claim_alignment": "claim_support",
        "numeric": "numeric_coverage",
        "numeric_coverage": "numeric_coverage",
        "evidence": "evidence_packaging",
        "evidence_density": "evidence_packaging",
        "contribution": "contribution_framing",
        "contributions": "contribution_framing",
        "breakthrough": "breakthrough_significance",
        "submission": "narrative_quality",
        "venue_fit": "venue_fit",
    }.get(metric)


def _collect_scorecard_focus_areas(quality_result: dict[str, Any]) -> list[str]:
    scorecard = quality_result.get("submission_scorecard")
    if not isinstance(scorecard, dict):
        return []
    gap_items: list[tuple[float, str]] = []
    for name, metric in scorecard.items():
        if not isinstance(metric, dict):
            continue
        gap = _coerce_float(metric.get("gap"))
        focus_area = _focus_area_from_scorecard_metric(name)
        if gap is None or gap <= 0 or focus_area is None:
            continue
        gap_items.append((gap, focus_area))
    gap_items.sort(reverse=True)
    return _dedupe_strings([item[1] for item in gap_items], limit=3)


def _derive_repair_execution_focus(
    repair_plan: dict[str, Any],
    review_state: dict[str, Any],
) -> dict[str, Any]:
    tasks = [item for item in (repair_plan.get("tasks") or []) if isinstance(item, dict)]
    ready_tasks = [item for item in tasks if str(item.get("status") or "").strip() == "ready"]
    lane_priority = {
        "evidence_followup": 0,
        "method_repair": 1,
        "claim_repair": 2,
        "figure_repair": 3,
        "section_rewrite": 4,
        "triage": 5,
    }
    sorted_ready_tasks = sorted(
        ready_tasks,
        key=lambda item: (
            {"p0": 0, "p1": 1, "p2": 2}.get(
                str(item.get("priority_tier") or "p2").strip().lower(),
                3,
            ),
            lane_priority.get(str(item.get("lane") or "").strip(), 9),
            -int(item.get("priority_score") or 0),
        ),
    )
    top_tasks = sorted_ready_tasks[:3]
    lane_order = _dedupe_strings(
        [str(item.get("lane") or "").strip() for item in sorted_ready_tasks if str(item.get("lane") or "").strip()],
        limit=6,
    )
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
    hostile_recheck_required = bool(int(hostile_summary.get("active_issue_count") or 0) > 0) or any(
        str(item.get("escalation_lane") or "").strip() == "hostile_critic" for item in top_tasks
    )
    return {
        "lane_order": lane_order,
        "top_tasks": [
            {
                "task_id": str(item.get("task_id") or "").strip(),
                "lane": str(item.get("lane") or "").strip(),
                "priority_tier": str(item.get("priority_tier") or "").strip() or "p2",
                "owner": str(item.get("owner") or "").strip() or None,
                "verifier": str(item.get("verifier") or "").strip() or None,
                "close_condition": str(item.get("close_condition") or "").strip() or None,
            }
            for item in top_tasks
        ],
        "hostile_recheck_required": hostile_recheck_required,
    }


def derive_autonomous_followup_focus(
    *,
    quality_result: dict[str, Any],
    acceptance: dict[str, Any],
    repair_plan: dict[str, Any] | None = None,
    review_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = quality_result.get("submission_readiness") or {}
    categories = readiness.get("categories") if isinstance(readiness, dict) else {}
    revision_actions = _normalize_revision_actions(
        quality_result.get("revision_actions") or []
    )
    acceptance_reasons = _dedupe_strings(
        list(acceptance.get("reasons") or [])
        + list(quality_result.get("submission_priority_reasons") or []),
        limit=8,
    )

    focus_areas: list[str] = []
    for action in revision_actions:
        for candidate_text in (
            action.get("focus"),
            action.get("action"),
            action.get("reason"),
        ):
            focus_area = _focus_area_from_text(candidate_text or "")
            if focus_area and focus_area not in focus_areas:
                focus_areas.append(focus_area)
                break

    if isinstance(categories, dict):
        for category, value in categories.items():
            if not value:
                continue
            focus_area = _focus_area_from_scorecard_metric(category)
            if focus_area and focus_area not in focus_areas:
                focus_areas.append(focus_area)

    for reason in acceptance_reasons:
        focus_area = _focus_area_from_text(reason)
        if focus_area and focus_area not in focus_areas:
            focus_areas.append(focus_area)

    for focus_area in _collect_scorecard_focus_areas(quality_result):
        if focus_area not in focus_areas:
            focus_areas.append(focus_area)

    gate_passed = quality_result.get("quality_gate_passed")
    blockers = _coerce_int(quality_result.get("blocker_count")) or 0
    if not focus_areas:
        if gate_passed is False or blockers > 0:
            focus_areas.extend(["claim_support", "rigor"])
        else:
            focus_areas.append("narrative_quality")

    focus = {
        "focus_areas": focus_areas,
        "preferred_sections": [],
        "frontmatter_required": False,
        "candidate_boost": 0,
        "target_section_limit": 4,
        "frontmatter_style_order": [],
        "section_style_order": [],
        "claim_softening_advice": [],
        "limitation_emphasis": [],
        "rebuttal_focus": [],
        "anticipated_objections": [],
        "notes": [],
        "required_actions": revision_actions[:4],
    }

    for area in focus_areas:
        profile = FOLLOWUP_FOCUS_PROFILES.get(area, {})
        focus["preferred_sections"] = _dedupe_strings(
            focus["preferred_sections"] + list(profile.get("preferred_sections") or []),
            limit=8,
        )
        focus["frontmatter_required"] = bool(
            focus["frontmatter_required"] or profile.get("frontmatter_required")
        )
        focus["candidate_boost"] = max(
            int(focus["candidate_boost"]),
            int(profile.get("candidate_boost") or 0),
        )
        focus["target_section_limit"] = max(
            int(focus["target_section_limit"]),
            int(profile.get("target_section_limit") or 0),
        )
        focus["frontmatter_style_order"] = _merge_style_order(
            list(profile.get("frontmatter_style_order") or []),
            focus["frontmatter_style_order"],
        )
        focus["section_style_order"] = _merge_style_order(
            list(profile.get("section_style_order") or []),
            focus["section_style_order"],
        )
        for key in [
            "claim_softening_advice",
            "limitation_emphasis",
            "rebuttal_focus",
            "anticipated_objections",
            "notes",
        ]:
            focus[key] = _dedupe_strings(
                list(focus.get(key) or []) + list(profile.get(key) or []),
                limit=8,
            )

    p0_actions = sum(item.get("priority") == "P0" for item in revision_actions)
    if p0_actions >= 2 or blockers >= 4:
        focus["candidate_boost"] = max(int(focus["candidate_boost"]), 2)
        focus["target_section_limit"] = max(int(focus["target_section_limit"]), 6)
    if any(
        area in focus_areas
        for area in ("claim_support", "numeric_coverage", "contribution_framing")
    ):
        focus["frontmatter_required"] = True

    if acceptance_reasons:
        focus["notes"] = _dedupe_strings(
            list(focus["notes"]) + acceptance_reasons[:4],
            limit=8,
        )
    if isinstance(repair_plan, dict) and repair_plan:
        repair_focus = _derive_repair_execution_focus(
            repair_plan,
            review_state if isinstance(review_state, dict) else {},
        )
        focus["repair_lane_order"] = repair_focus["lane_order"]
        focus["hostile_recheck_required"] = repair_focus["hostile_recheck_required"]
        top_task_notes = []
        for item in repair_focus["top_tasks"]:
            lane = str(item.get("lane") or "").replace("_", " ")
            verifier = str(item.get("verifier") or "").replace("_", " ")
            close_condition = str(item.get("close_condition") or "").strip()
            if lane:
                top_task_notes.append(f"repair lane {lane} should clear via {verifier}")
            if close_condition:
                top_task_notes.append(close_condition)
        focus["notes"] = _dedupe_strings(
            list(focus["notes"]) + top_task_notes,
            limit=10,
        )
    return focus


def _invoke_high_quality_pass(
    *,
    run_high_quality_pass_fn: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(run_high_quality_pass_fn)
    except (TypeError, ValueError):
        return run_high_quality_pass_fn(**kwargs)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return run_high_quality_pass_fn(**kwargs)

    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return run_high_quality_pass_fn(**filtered_kwargs)


def _effective_quality_settings(
    *,
    quality_preset: str,
    quality_threshold: Optional[float],
    rigor_threshold: Optional[float],
    max_quality_rewrites: Optional[int],
) -> dict[str, Any]:
    preset = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["balanced"])
    return {
        "quality_preset": quality_preset,
        "quality_threshold": quality_threshold or preset["quality_threshold"],
        "rigor_threshold": rigor_threshold or preset["rigor_threshold"],
        "max_quality_rewrites": max_quality_rewrites or preset["max_rewrite_rounds"],
    }


def _stronger_followup_settings(
    *,
    current_settings: dict[str, Any],
    target_preset: str,
) -> dict[str, Any]:
    target_preset_settings = _effective_quality_settings(
        quality_preset=target_preset,
        quality_threshold=None,
        rigor_threshold=None,
        max_quality_rewrites=None,
    )
    return {
        "quality_preset": target_preset,
        "quality_threshold": max(
            float(current_settings["quality_threshold"]),
            float(target_preset_settings["quality_threshold"]),
        ),
        "rigor_threshold": max(
            float(current_settings["rigor_threshold"]),
            float(target_preset_settings["rigor_threshold"]),
        ),
        "max_quality_rewrites": max(
            int(current_settings["max_quality_rewrites"]),
            int(target_preset_settings["max_quality_rewrites"]),
        ),
    }


def _next_quality_preset(current_preset: str) -> Optional[str]:
    if current_preset not in QUALITY_PRESET_ORDER:
        return "high"
    current_idx = QUALITY_PRESET_ORDER.index(current_preset)
    if current_idx >= len(QUALITY_PRESET_ORDER) - 1:
        return None
    return QUALITY_PRESET_ORDER[current_idx + 1]


def format_quality_pass_summary(quality_result: dict[str, Any]) -> str:
    return (
        "High-quality pass: "
        f"before={quality_result.get('quality_score_before', 0):.2f}, "
        f"after={quality_result.get('quality_score_after', 0):.2f}, "
        f"rewrite_applied={quality_result.get('rewrite_applied')}, "
        f"gate_passed={quality_result.get('quality_gate_passed')}, "
        f"priority={quality_result.get('submission_priority_score')}, "
        f"blockers={quality_result.get('blocker_count')}"
    )


def execute_quality_workflow(
    *,
    run_high_quality_pass_fn: Callable[..., dict[str, Any]],
    run_dir: str,
    paper_type: str,
    rewrite_model: str,
    quality_model: str,
    target_venue: str,
    quality_preset: str,
    quality_threshold: Optional[float],
    rigor_threshold: Optional[float],
    max_quality_rewrites: Optional[int],
    require_quality_gate: bool,
    min_submission_priority: Optional[float],
    max_submission_blockers: Optional[int],
    allow_auto_improvement_fallback: Optional[bool] = None,
    reject_on_auto_improvement_fallback: bool = False,
    autonomous_followup_focus: Optional[dict[str, Any]] = None,
    resume: bool,
    logger: Callable[[str], None],
) -> dict[str, Any]:
    preset = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["balanced"])
    quality_result = _invoke_high_quality_pass(
        run_high_quality_pass_fn=run_high_quality_pass_fn,
        kwargs={
            "base_folder": run_dir,
            "run_dir": run_dir,
            "paper_type": paper_type,
            "rewrite_model": rewrite_model,
            "quality_model": quality_model,
            "target_venue": target_venue,
            "quality_threshold": quality_threshold or preset["quality_threshold"],
            "rigor_threshold": rigor_threshold or preset["rigor_threshold"],
            "max_rewrite_rounds": max_quality_rewrites or preset["max_rewrite_rounds"],
            "auto_improvement_fallback": (
                allow_auto_improvement_fallback
                if allow_auto_improvement_fallback is not None
                else preset.get("auto_improvement_fallback", True)
            ),
            "autonomous_followup_focus": autonomous_followup_focus,
            "resume": resume,
            "logger": logger,
        },
    )
    effective_priority_bar, effective_blocker_bar = (
        resolve_submission_acceptance_settings(
            quality_result.get("target_venue", target_venue),
            min_submission_priority=min_submission_priority,
            max_submission_blockers=max_submission_blockers,
        )
    )
    acceptance = evaluate_submission_acceptance(
        quality_result,
        require_quality_gate=require_quality_gate,
        min_submission_priority=effective_priority_bar,
        max_submission_blockers=effective_blocker_bar,
        reject_on_auto_improvement_fallback=reject_on_auto_improvement_fallback,
    )
    return {
        "quality_result": quality_result,
        "acceptance": acceptance,
        "effective_priority_bar": effective_priority_bar,
        "effective_blocker_bar": effective_blocker_bar,
        "summary": format_quality_pass_summary(quality_result),
    }


def derive_quality_followup_plan(
    *,
    quality_result: dict[str, Any],
    acceptance: dict[str, Any],
    quality_preset: str,
    quality_threshold: Optional[float],
    rigor_threshold: Optional[float],
    max_quality_rewrites: Optional[int],
) -> dict[str, Any]:
    settings = _effective_quality_settings(
        quality_preset=quality_preset,
        quality_threshold=quality_threshold,
        rigor_threshold=rigor_threshold,
        max_quality_rewrites=max_quality_rewrites,
    )
    blockers = _coerce_int(quality_result.get("blocker_count"))
    priority = _coerce_float(quality_result.get("submission_priority_score"))
    readiness = str(
        (quality_result.get("submission_readiness") or {}).get("status") or ""
    ).lower()
    rewrite_gain = _coerce_float(
        (quality_result.get("rewrite_effectiveness_summary") or {}).get(
            "priority_gain_total"
        )
    )
    gate_passed = quality_result.get("quality_gate_passed")
    acceptance_reasons = [str(item) for item in (acceptance.get("reasons") or [])]
    acceptance_reasons_lower = [item.lower() for item in acceptance_reasons]
    next_preset = _next_quality_preset(quality_preset)

    plan = {
        **settings,
        "mode": "standard",
        "skip": False,
        "reason": "retry quality workflow with current settings",
        "autonomous_followup_focus": derive_autonomous_followup_focus(
            quality_result=quality_result,
            acceptance=acceptance,
        ),
    }

    if gate_passed is False or any(
        "quality gate" in reason or "blocker" in reason
        for reason in acceptance_reasons_lower
    ):
        target_preset = next_preset or quality_preset
        target_settings = _stronger_followup_settings(
            current_settings=settings,
            target_preset=target_preset,
        )
        plan.update(
            {
                **target_settings,
                "mode": "blocker_reduction",
                "reason": "gate/blocker feedback triggered a stronger autonomous rewrite follow-up",
            }
        )
    elif readiness == "ready":
        plan.update(
            {
                "mode": "final_polish",
                "max_quality_rewrites": max(1, min(int(settings["max_quality_rewrites"]), 1)),
                "reason": "paper is close to ready; run a short autonomous polish pass",
            }
        )
    elif (
        priority is not None
        and priority >= 82.0
    ) or (
        rewrite_gain is not None and rewrite_gain >= 1.0
    ):
        target_settings = _stronger_followup_settings(
            current_settings=settings,
            target_preset="publishable",
        )
        plan.update(
            {
                **target_settings,
                "mode": "submission_push",
                "reason": "submission priority trend justifies a stronger publishable follow-up",
            }
        )
    elif next_preset is not None:
        target_settings = _stronger_followup_settings(
            current_settings=settings,
            target_preset=next_preset,
        )
        plan.update(
            {
                **target_settings,
                "mode": "preset_escalation",
                "reason": "escalate to the next stronger preset for an autonomous retry",
            }
        )
    else:
        plan.update(
            {
                "mode": "extra_rewrite_round",
                "max_quality_rewrites": int(settings["max_quality_rewrites"]) + 1,
                "reason": "no stronger preset remains; extend rewrite budget by one autonomous round",
            }
        )

    if (
        plan["quality_preset"] == settings["quality_preset"]
        and plan["quality_threshold"] == settings["quality_threshold"]
        and plan["rigor_threshold"] == settings["rigor_threshold"]
        and int(plan["max_quality_rewrites"]) <= int(settings["max_quality_rewrites"])
    ):
        plan["max_quality_rewrites"] = int(settings["max_quality_rewrites"]) + 1
        plan["reason"] += "; forced rewrite budget increase to avoid a no-op retry"

    if blockers is not None and blockers >= 8 and plan["mode"] != "blocker_reduction":
        plan["mode"] = "blocker_reduction"
        plan["reason"] += "; severe blocker count detected"

    return plan


def execute_quality_workflow_with_followups(
    *,
    run_high_quality_pass_fn: Callable[..., dict[str, Any]],
    run_dir: str,
    paper_type: str,
    rewrite_model: str,
    quality_model: str,
    target_venue: str,
    quality_preset: str,
    quality_threshold: Optional[float],
    rigor_threshold: Optional[float],
    max_quality_rewrites: Optional[int],
    require_quality_gate: bool,
    min_submission_priority: Optional[float],
    max_submission_blockers: Optional[int],
    autonomous_followup_rounds: int,
    allow_auto_improvement_fallback: Optional[bool] = None,
    reject_on_auto_improvement_fallback: bool = False,
    resume: bool,
    logger: Callable[[str], None],
) -> dict[str, Any]:
    current_settings = {
        "quality_preset": quality_preset,
        "quality_threshold": quality_threshold,
        "rigor_threshold": rigor_threshold,
        "max_quality_rewrites": max_quality_rewrites,
    }
    quality_pass = execute_quality_workflow(
        run_high_quality_pass_fn=run_high_quality_pass_fn,
        run_dir=run_dir,
        paper_type=paper_type,
        rewrite_model=rewrite_model,
        quality_model=quality_model,
        target_venue=target_venue,
        quality_preset=current_settings["quality_preset"],
        quality_threshold=current_settings["quality_threshold"],
        rigor_threshold=current_settings["rigor_threshold"],
        max_quality_rewrites=current_settings["max_quality_rewrites"],
        require_quality_gate=require_quality_gate,
        min_submission_priority=min_submission_priority,
        max_submission_blockers=max_submission_blockers,
        allow_auto_improvement_fallback=allow_auto_improvement_fallback,
        reject_on_auto_improvement_fallback=reject_on_auto_improvement_fallback,
        resume=resume,
        logger=logger,
    )

    followup_history: list[dict[str, Any]] = []
    effective_rounds = max(0, int(autonomous_followup_rounds))
    for round_idx in range(effective_rounds):
        if quality_pass["acceptance"].get("accepted"):
            break

        plan = derive_quality_followup_plan(
            quality_result=quality_pass["quality_result"],
            acceptance=quality_pass["acceptance"],
            quality_preset=current_settings["quality_preset"],
            quality_threshold=current_settings["quality_threshold"],
            rigor_threshold=current_settings["rigor_threshold"],
            max_quality_rewrites=current_settings["max_quality_rewrites"],
        )
        followup_item = {
            "round": round_idx + 1,
            "executed": not plan.get("skip", False),
            "plan": plan,
            "accepted_before": quality_pass["acceptance"].get("accepted"),
            "reasons_before": quality_pass["acceptance"].get("reasons", []),
            "summary_before": quality_pass["summary"],
        }
        followup_history.append(followup_item)
        if plan.get("skip", False):
            logger(
                f"Autonomous quality follow-up {round_idx + 1}/{effective_rounds} skipped: {plan.get('reason')}"
            )
            break

        logger(
            f"Autonomous quality follow-up {round_idx + 1}/{effective_rounds}: "
            f"mode={plan['mode']}, preset={plan['quality_preset']}, "
            f"quality_threshold={plan['quality_threshold']}, "
            f"rigor_threshold={plan['rigor_threshold']}, "
            f"max_rewrite_rounds={plan['max_quality_rewrites']}; "
            f"reason={plan['reason']}"
        )
        current_settings = {
            "quality_preset": plan["quality_preset"],
            "quality_threshold": plan["quality_threshold"],
            "rigor_threshold": plan["rigor_threshold"],
            "max_quality_rewrites": plan["max_quality_rewrites"],
        }
        quality_pass = execute_quality_workflow(
            run_high_quality_pass_fn=run_high_quality_pass_fn,
            run_dir=run_dir,
            paper_type=paper_type,
            rewrite_model=rewrite_model,
            quality_model=quality_model,
            target_venue=target_venue,
            quality_preset=current_settings["quality_preset"],
            quality_threshold=current_settings["quality_threshold"],
            rigor_threshold=current_settings["rigor_threshold"],
            max_quality_rewrites=current_settings["max_quality_rewrites"],
            require_quality_gate=require_quality_gate,
            min_submission_priority=min_submission_priority,
            max_submission_blockers=max_submission_blockers,
            allow_auto_improvement_fallback=allow_auto_improvement_fallback,
            reject_on_auto_improvement_fallback=reject_on_auto_improvement_fallback,
            autonomous_followup_focus=plan.get("autonomous_followup_focus"),
            resume=False,
            logger=logger,
        )
        followup_item["summary_after"] = quality_pass["summary"]
        followup_item["accepted_after"] = quality_pass["acceptance"].get("accepted")
        followup_item["reasons_after"] = quality_pass["acceptance"].get("reasons", [])

    quality_result = dict(quality_pass["quality_result"])
    quality_result["autonomous_followup_rounds_run"] = sum(
        1 for item in followup_history if item.get("executed")
    )
    quality_result["autonomous_followup_applied"] = bool(followup_history)
    quality_result["autonomous_followup_history"] = followup_history

    summary = quality_pass["summary"]
    if followup_history:
        summary = (
            f"{summary}, autonomous_followups={quality_result['autonomous_followup_rounds_run']}"
        )

    return {
        **quality_pass,
        "quality_result": quality_result,
        "summary": summary,
        "autonomous_followup_history": followup_history,
    }
