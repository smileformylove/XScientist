from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import record_fallback_event


class StrictFallbackViolation(RuntimeError):
    """Raised when strict fallback enforcement blocks progression."""

    pass

STRICT_FALLBACK_WORKFLOWS = {
    "program_driven",
    "writing_studio",
    "review_board",
    "multi_agent_board",
}


def should_enforce_strict_fallbacks(
    workflow_mode: str | None,
    *,
    submission_mode: bool,
    high_quality_mode: bool,
    target_venue: str | None,
) -> bool:
    normalized = str(workflow_mode or "").strip().lower()
    if normalized in STRICT_FALLBACK_WORKFLOWS:
        return True
    if submission_mode or high_quality_mode:
        return True
    if str(target_venue or "").lower() in {"journal", "nature"}:
        return True
    return False


def format_strict_fallback_error(
    event: dict[str, Any] | None,
    *,
    workflow_mode: str | None = None,
    stage_hint: str | None = None,
) -> str:
    if not isinstance(event, dict):
        return (
            "Strict fallback policy triggered; rerun after fixing the upstream "
            "stage or pass --override-strict-fallbacks if you intentionally "
            "accept degraded rigor."
        )
    stage = stage_hint or event.get("stage") or "pipeline"
    fallback_kind = event.get("fallback_kind") or "unspecified"
    reason = event.get("reason") or ""
    workflow = str(workflow_mode or "adaptive")
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    idea_hint = metadata.get("idea_indices") or metadata.get("idea_idx")
    message = (
        f"Strict fallback policy blocked {stage}: fallback_kind={fallback_kind}"
    )
    if reason:
        message += f"; reason={reason}"
    if idea_hint:
        message += f"; idea_indices={idea_hint}"
    if workflow:
        message += f"; workflow={workflow}"
    message += ". Use --override-strict-fallbacks only if you intentionally accept fallback behavior."
    return message


def collect_ranking_fallback_summary(
    rankings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    fallback_rows = [
        item for item in rankings if isinstance(item, dict) and item.get("fallback_used")
    ]
    if not fallback_rows:
        return None
    stage_counts = Counter(
        str(item.get("fallback_stage") or "unknown") for item in fallback_rows
    )
    reason_counts = Counter(
        str(item.get("fallback_reason") or "unknown") for item in fallback_rows
    )
    return {
        "count": len(fallback_rows),
        "idea_count": len(rankings),
        "idea_indices": [item.get("idea_idx") for item in fallback_rows],
        "idea_names": [item.get("idea_name") for item in fallback_rows],
        "stage_counts": dict(stage_counts),
        "reason_counts": dict(reason_counts),
    }


def record_ranking_fallbacks(
    project_root: str | Path,
    rankings: list[dict[str, Any]],
    *,
    producer: str,
    strict: bool = False,
) -> dict[str, Any] | None:
    summary = collect_ranking_fallback_summary(rankings)
    if not summary:
        return None
    reasons = summary.get("reason_counts") or {}
    reason = (
        f"{summary['count']}/{summary['idea_count']} ranked ideas used fallback scoring; "
        f"reasons={reasons}"
    )
    return record_fallback_event(
        project_root,
        stage="idea_ranking",
        producer=producer,
        fallback_kind="heuristic_ranking",
        reason=reason,
        strict=strict,
        metadata=summary,
    )


def record_quality_fallback_if_needed(
    project_root: str | Path,
    quality_result: dict[str, Any] | None,
    *,
    producer: str,
    strict: bool = True,
) -> dict[str, Any] | None:
    if not isinstance(quality_result, dict):
        return None
    if quality_result.get("auto_improvement_fallback_used") is not True:
        return None
    return record_fallback_event(
        project_root,
        stage="quality_review",
        producer=producer,
        fallback_kind="auto_improvement_rewrite",
        reason="High-quality gate did not pass, so auto-improvement fallback rewrites were used.",
        strict=bool(strict),
        metadata={
            "target_venue": quality_result.get("target_venue"),
            "quality_gate_passed": quality_result.get("quality_gate_passed"),
            "submission_priority_score": quality_result.get(
                "submission_priority_score"
            ),
            "blocker_count": quality_result.get("blocker_count"),
            "fallback_reason": quality_result.get("auto_improvement_fallback_reason"),
        },
    )
