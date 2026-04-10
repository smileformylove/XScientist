from __future__ import annotations

"""Benchmark historical high-quality runs against submission-readiness bars."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ai_scientist.config.paths import resolve_output_path
from ai_scientist.utils.high_quality_pipeline import VENUE_PRESETS
from ai_scientist.utils.process_alignment import build_process_alignment
from ai_scientist.utils.self_evolution import build_self_evolution
from ai_scientist.utils.stage_standards import build_stage_standards
from ai_scientist.utils.submission_history import iter_historical_quality_results


__all__ = [
    "build_readiness_benchmark",
    "export_readiness_benchmark_markdown",
]


def _resolve_research_root(research_root: str | Path | None = None) -> Path:
    if research_root is None:
        return resolve_output_path()
    return Path(research_root).expanduser()


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_manifest_fallback_summary(run_dir: Path) -> dict[str, Any]:
    manifest = _safe_load_json(run_dir / "pipeline_manifest.json")
    summary = (
        manifest.get("fallback_summary")
        if isinstance(manifest.get("fallback_summary"), dict)
        else {}
    )
    return summary if isinstance(summary, dict) else {}


def _safe_stage_standards(run_dir: Path) -> dict[str, Any]:
    standards = _safe_load_json(run_dir / "stage_standards.json")
    if isinstance(standards, dict) and (
        "stage_results" in standards or "overall_score" in standards
    ):
        return standards
    computed = build_stage_standards(run_dir)
    return computed if isinstance(computed, dict) else {}


def _safe_self_evolution(run_dir: Path) -> dict[str, Any]:
    evolution = _safe_load_json(run_dir / "self_evolution.json")
    if isinstance(evolution, dict) and (
        "summary" in evolution or "self_check" in evolution
    ):
        return evolution
    computed = build_self_evolution(run_dir)
    return computed if isinstance(computed, dict) else {}


def _safe_process_alignment(run_dir: Path) -> dict[str, Any]:
    alignment = _safe_load_json(run_dir / "process_alignment.json")
    if isinstance(alignment, dict) and (
        "summary" in alignment
        or "process_results" in alignment
        or "reference_summary" in alignment
    ):
        return alignment
    computed = build_process_alignment(run_dir)
    return computed if isinstance(computed, dict) else {}


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_benchmark_thresholds(target_venue: str) -> dict[str, float]:
    venue = str(target_venue or "nature")
    preset = VENUE_PRESETS.get(venue, VENUE_PRESETS["nature"])
    return {
        "quality": float(preset["quality_threshold"]),
        "rigor": float(preset["rigor_threshold"]),
        "claim_support": float(preset["claim_support_threshold"]),
        "claim_alignment": 3.5,
        "numeric_coverage": 3.5,
        "evidence_density": 3.0,
        "breakthrough": 4.0 if venue == "nature" else 3.5,
        "contributions": 2.0,
    }


def _build_metric_payload(
    scorecard: dict[str, Any],
    name: str,
    *,
    fallback_score: Any,
    threshold: float,
) -> dict[str, Any]:
    item = scorecard.get(name)
    score = _coerce_float((item or {}).get("score")) if isinstance(item, dict) else None
    gap = _coerce_float((item or {}).get("gap")) if isinstance(item, dict) else None
    passed = (item or {}).get("pass") if isinstance(item, dict) else None
    if score is None:
        score = _coerce_float(fallback_score)
    if score is None:
        score = 0.0
    if gap is None:
        gap = max(0.0, threshold - score)
    if passed is None:
        passed = gap <= 0.0
    return {
        "score": round(score, 2),
        "threshold": round(threshold, 2),
        "gap": round(max(0.0, gap), 2),
        "pass": bool(passed),
    }


def _build_benchmark_score(
    *,
    metrics: dict[str, dict[str, Any]],
    quality_gate_passed: bool,
    ready: bool,
    blocker_count: int,
    unsupported_claims_count: int,
    fallback_count: int,
    strict_fallback_count: int,
    stage_overall_score: float,
    blocked_stage_count: int,
    needs_attention_stage_count: int,
    missing_stage_count: int,
    self_evolution_score: float,
    self_evolution_status: str,
    self_evolution_required_failure_count: int,
    process_alignment_score: float,
    process_alignment_blocked_count: int,
    process_alignment_missing_count: int,
    target_venue: str,
) -> float:
    weights = {
        "quality": 18,
        "rigor": 16,
        "claim_support": 14,
        "claim_alignment": 10,
        "numeric_coverage": 10,
        "evidence_density": 8,
        "breakthrough": 14 if target_venue == "nature" else 10,
        "contributions": 6,
    }
    weighted = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        metric = metrics.get(name, {})
        threshold = float(metric.get("threshold") or 1.0)
        score = float(metric.get("score") or 0.0)
        weighted += weight * max(0.0, min(score / threshold, 1.0))
        total_weight += weight

    # Leave room above the metric-only score so readiness bonuses and debt penalties
    # can still differentiate otherwise-strong papers instead of saturating at 100.
    base = (weighted / max(total_weight, 1.0)) * 90.0
    bonus = (
        (4.0 if quality_gate_passed else 0.0)
        + (6.0 if ready else 0.0)
        + max(0.0, min(stage_overall_score, 100.0)) * 0.04
        + max(0.0, min(self_evolution_score, 100.0)) * 0.03
        + max(0.0, min(process_alignment_score, 100.0)) * 0.03
    )
    penalty = (
        blocker_count * 1.8
        + min(unsupported_claims_count, 8) * 0.7
        + min(fallback_count, 8) * 0.45
        + min(strict_fallback_count, 4) * 1.15
        + blocked_stage_count * 2.6
        + missing_stage_count * 0.9
        + needs_attention_stage_count * 0.55
        + self_evolution_required_failure_count * 1.5
        + process_alignment_blocked_count * 2.2
        + process_alignment_missing_count * 0.8
        + (
            4.2
            if self_evolution_status == "blocked"
            else 1.6 if self_evolution_status == "needs_attention" else 0.0
        )
    )
    return round(max(0.0, min(100.0, base + bonus - penalty)), 2)


def _build_recommendation(
    *,
    failing_metrics: list[dict[str, Any]],
    blockers: list[str],
    self_evolution_status: str,
    self_evolution_required_failure_count: int,
    top_self_evolution_risks: list[str],
    process_alignment_blocked_count: int,
    top_process_alignment_risks: list[str],
    target_venue: str,
    venue_match: bool,
) -> str:
    if not venue_match:
        return (
            f"Current assessment targets another venue; rerun quality evaluation with "
            f"`--target-venue {target_venue}` for a stricter benchmark."
        )
    if self_evolution_status == "blocked":
        top_risk = top_self_evolution_risks[0] if top_self_evolution_risks else None
        if top_risk:
            return (
                "Clear self-evolution required failures before another submission push; "
                f"highest-risk repair lane is `{top_risk}`."
            )
        return (
            "Clear self-evolution required failures before another submission push."
        )
    if self_evolution_required_failure_count > 0:
        return (
            "Close self-evolution required failures and verify the repair plan before "
            "another submission attempt."
        )
    if process_alignment_blocked_count > 0:
        top_risk = top_process_alignment_risks[0] if top_process_alignment_risks else None
        if top_risk:
            return (
                "Repair blocked process alignment before another submission push; "
                f"highest-risk process gap is `{top_risk}`."
            )
        return "Repair blocked process alignment before another submission push."
    if blockers:
        return blockers[0]
    if failing_metrics:
        top = failing_metrics[0]
        return (
            f"Close the {top['name']} gap first "
            f"({top['score']:.2f} < {top['threshold']:.2f})."
        )
    return "Meets the current benchmark bar; focus on polish and submission packaging."


def _build_benchmark_entry(
    quality_path: Path,
    *,
    research_root: Path,
    target_venue: str,
) -> Optional[dict[str, Any]]:
    result = _safe_load_json(quality_path)
    if result.get("status") != "success":
        return None

    run_dir = quality_path.parent.parent
    idea = _safe_load_json(run_dir / "idea.json")
    scorecard = (
        result.get("submission_scorecard")
        if isinstance(result.get("submission_scorecard"), dict)
        else {}
    )
    readiness = (
        result.get("submission_readiness")
        if isinstance(result.get("submission_readiness"), dict)
        else {}
    )
    thresholds = _resolve_benchmark_thresholds(target_venue)
    paper_target_venue = str(result.get("target_venue") or "unknown")
    venue_match = paper_target_venue == target_venue
    fallback_summary = _safe_manifest_fallback_summary(run_dir)
    fallback_count = _coerce_int(fallback_summary.get("count")) or 0
    strict_fallback_count = _coerce_int(fallback_summary.get("strict_count")) or 0
    stage_standards = _safe_stage_standards(run_dir)
    self_evolution = _safe_self_evolution(run_dir)
    process_alignment = _safe_process_alignment(run_dir)
    stage_summary = (
        stage_standards.get("summary")
        if isinstance(stage_standards.get("summary"), dict)
        else {}
    )
    self_evolution_summary = (
        self_evolution.get("summary")
        if isinstance(self_evolution.get("summary"), dict)
        else {}
    )
    process_alignment_summary = (
        process_alignment.get("summary")
        if isinstance(process_alignment.get("summary"), dict)
        else {}
    )
    self_evolution_self_check = (
        self_evolution.get("self_check")
        if isinstance(self_evolution.get("self_check"), dict)
        else {}
    )
    stage_overall_score = _coerce_float(stage_standards.get("overall_score")) or 0.0
    blocked_stage_count = _coerce_int(stage_standards.get("blocked_stage_count")) or 0
    needs_attention_stage_count = (
        _coerce_int(stage_standards.get("needs_attention_stage_count")) or 0
    )
    missing_stage_count = _coerce_int(stage_standards.get("missing_stage_count")) or 0
    self_evolution_score = (
        _coerce_float(self_evolution_summary.get("score"))
        or _coerce_float(self_evolution_self_check.get("score"))
        or 0.0
    )
    self_evolution_status = str(
        self_evolution_summary.get("status")
        or self_evolution_self_check.get("status")
        or ""
    ).strip()
    self_evolution_required_failure_count = len(
        self_evolution_self_check.get("required_failures") or []
    )
    top_self_evolution_risks = [
        str(item).strip()
        for item in (self_evolution.get("stage_risks") or [])
        if str(item).strip()
    ]
    process_alignment_score = (
        _coerce_float(process_alignment_summary.get("overall_score")) or 0.0
    )
    process_alignment_blocked_count = (
        _coerce_int(process_alignment_summary.get("blocked_process_count")) or 0
    )
    process_alignment_missing_count = (
        _coerce_int(process_alignment_summary.get("missing_process_count")) or 0
    )
    top_process_alignment_risks = [
        str(item).strip()
        for item in (process_alignment_summary.get("top_process_risks") or {}).keys()
        if str(item).strip()
    ]

    metrics = {
        "quality": _build_metric_payload(
            scorecard,
            "quality",
            fallback_score=result.get("quality_score_after"),
            threshold=thresholds["quality"],
        ),
        "rigor": _build_metric_payload(
            scorecard,
            "rigor",
            fallback_score=result.get("rigor_score_after"),
            threshold=thresholds["rigor"],
        ),
        "claim_support": _build_metric_payload(
            scorecard,
            "claim_support",
            fallback_score=result.get("claim_support_after"),
            threshold=thresholds["claim_support"],
        ),
        "claim_alignment": _build_metric_payload(
            scorecard,
            "claim_alignment",
            fallback_score=result.get("claim_alignment_after"),
            threshold=thresholds["claim_alignment"],
        ),
        "numeric_coverage": _build_metric_payload(
            scorecard,
            "numeric_coverage",
            fallback_score=result.get("numeric_coverage_after"),
            threshold=thresholds["numeric_coverage"],
        ),
        "evidence_density": _build_metric_payload(
            scorecard,
            "evidence_density",
            fallback_score=result.get("evidence_density_score"),
            threshold=thresholds["evidence_density"],
        ),
        "breakthrough": _build_metric_payload(
            scorecard,
            "breakthrough",
            fallback_score=result.get("breakthrough_score"),
            threshold=thresholds["breakthrough"],
        ),
        "contributions": _build_metric_payload(
            scorecard,
            "contributions",
            fallback_score=result.get("contribution_count"),
            threshold=thresholds["contributions"],
        ),
    }

    blocker_count = _coerce_int(result.get("blocker_count")) or len(
        readiness.get("blockers") or []
    )
    unsupported_claims_count = _coerce_int(result.get("unsupported_claims_count")) or 0
    quality_gate_passed = bool(result.get("quality_gate_passed"))
    ready = (
        bool(readiness.get("ready"))
        if "ready" in readiness
        else str(readiness.get("status") or "").strip().lower() == "ready"
    )
    failing_metrics = [
        {
            "name": name,
            "score": metric["score"],
            "threshold": metric["threshold"],
            "gap": metric["gap"],
        }
        for name, metric in metrics.items()
        if not metric["pass"]
    ]
    failing_metrics.sort(key=lambda item: (item["gap"], item["threshold"]), reverse=True)
    blockers = [str(item).strip() for item in (readiness.get("blockers") or []) if str(item).strip()]

    benchmark_score = _build_benchmark_score(
        metrics=metrics,
        quality_gate_passed=quality_gate_passed,
        ready=ready,
        blocker_count=blocker_count,
        unsupported_claims_count=unsupported_claims_count,
        fallback_count=fallback_count,
        strict_fallback_count=strict_fallback_count,
        stage_overall_score=stage_overall_score,
        blocked_stage_count=blocked_stage_count,
        needs_attention_stage_count=needs_attention_stage_count,
        missing_stage_count=missing_stage_count,
        self_evolution_score=self_evolution_score,
        self_evolution_status=self_evolution_status,
        self_evolution_required_failure_count=self_evolution_required_failure_count,
        process_alignment_score=process_alignment_score,
        process_alignment_blocked_count=process_alignment_blocked_count,
        process_alignment_missing_count=process_alignment_missing_count,
        target_venue=target_venue,
    )

    try:
        relative_run_dir = str(run_dir.resolve().relative_to(research_root.resolve()))
    except ValueError:
        relative_run_dir = str(run_dir.resolve())

    return {
        "name": idea.get("Name") or run_dir.name,
        "title": idea.get("Title") or "",
        "run_dir": str(run_dir),
        "relative_run_dir": relative_run_dir,
        "quality_result_file": str(quality_path),
        "paper_target_venue": paper_target_venue,
        "benchmark_target_venue": target_venue,
        "venue_match": venue_match,
        "submission_status": readiness.get("status") or ("ready" if ready else "needs_work"),
        "quality_gate_passed": quality_gate_passed,
        "submission_priority_score": _coerce_float(result.get("submission_priority_score")),
        "submission_priority_tier": result.get("submission_priority_tier"),
        "blocker_count": blocker_count,
        "fallback_count": fallback_count,
        "strict_fallback_count": strict_fallback_count,
        "fallback_stage_counts": fallback_summary.get("stage_counts") or {},
        "fallback_kind_counts": fallback_summary.get("kind_counts") or {},
        "stage_overall_score": round(stage_overall_score, 2),
        "blocked_stage_count": blocked_stage_count,
        "needs_attention_stage_count": needs_attention_stage_count,
        "missing_stage_count": missing_stage_count,
        "blocked_standard_stages": stage_summary.get("blocked_stages") or [],
        "attention_standard_stages": stage_summary.get("attention_stages") or [],
        "missing_standard_stages": stage_summary.get("missing_stages") or [],
        "top_standard_risks": stage_summary.get("top_risks") or [],
        "self_evolution_status": self_evolution_status,
        "self_evolution_score": round(self_evolution_score, 2),
        "self_evolution_required_failure_count": self_evolution_required_failure_count,
        "self_evolution_dominant_lane": self_evolution_summary.get("dominant_lane"),
        "self_evolution_dominant_role": self_evolution_summary.get("dominant_role"),
        "top_self_evolution_risks": top_self_evolution_risks[:3],
        "process_alignment_overall_score": round(process_alignment_score, 2),
        "process_alignment_blocked_process_count": process_alignment_blocked_count,
        "process_alignment_missing_process_count": process_alignment_missing_count,
        "top_process_alignment_risks": top_process_alignment_risks[:3],
        "unsupported_claims_count": unsupported_claims_count,
        "critical_revision_actions_count": _coerce_int(
            result.get("critical_revision_actions_count")
        ),
        "rewrite_round_count": len(result.get("rewrite_trace") or []),
        "rewrite_applied": bool(result.get("rewrite_applied")),
        "metrics": metrics,
        "failing_metrics": failing_metrics,
        "blocker_categories": readiness.get("categories") or {},
        "top_blockers": blockers[:3],
        "benchmark_score": benchmark_score,
        "recommendation": _build_recommendation(
            failing_metrics=failing_metrics,
            blockers=blockers,
            self_evolution_status=self_evolution_status,
            self_evolution_required_failure_count=self_evolution_required_failure_count,
            top_self_evolution_risks=top_self_evolution_risks,
            process_alignment_blocked_count=process_alignment_blocked_count,
            top_process_alignment_risks=top_process_alignment_risks,
            target_venue=target_venue,
            venue_match=venue_match,
        ),
        "modified_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
    }


def build_readiness_benchmark(
    research_root: str | Path | None = None,
    *,
    target_venue: str = "nature",
    max_entries: int = 200,
    top_n: int = 10,
    include_other_venues: bool = False,
) -> dict[str, Any]:
    resolved_root = _resolve_research_root(research_root)
    rows: list[dict[str, Any]] = []

    for quality_path in iter_historical_quality_results(resolved_root):
        entry = _build_benchmark_entry(
            quality_path,
            research_root=resolved_root,
            target_venue=target_venue,
        )
        if entry is None:
            continue
        if not include_other_venues and not entry["venue_match"]:
            continue
        rows.append(entry)
        if len(rows) >= max_entries:
            break

    rows.sort(
        key=lambda item: (
            item.get("submission_status") == "ready",
            item.get("quality_gate_passed") is True,
            item.get("venue_match") is True,
            -(item.get("process_alignment_blocked_process_count") or 0),
            item.get("self_evolution_status") != "blocked",
            item.get("benchmark_score") or 0.0,
            item.get("stage_overall_score") or 0.0,
            item.get("process_alignment_overall_score") or 0.0,
            item.get("self_evolution_score") or 0.0,
            -(item.get("self_evolution_required_failure_count") or 0),
            -(item.get("blocked_stage_count") or 0),
            -(item.get("missing_stage_count") or 0),
            -(item.get("needs_attention_stage_count") or 0),
            item.get("submission_priority_score") or 0.0,
            -(item.get("strict_fallback_count") or 0),
            -(item.get("fallback_count") or 0),
            -(item.get("blocker_count") or 0),
            item.get("modified_at") or "",
        ),
        reverse=True,
    )

    blocker_categories = Counter()
    top_gap_dimensions = Counter()
    ready_count = 0
    gate_pass_count = 0
    venue_match_count = 0
    benchmark_scores: list[float] = []
    priority_scores: list[float] = []
    blocker_counts: list[int] = []
    fallback_counts: list[int] = []
    strict_fallback_counts: list[int] = []
    stage_overall_scores: list[float] = []
    blocked_stage_counts: list[int] = []
    attention_stage_counts: list[int] = []
    missing_stage_counts: list[int] = []
    process_alignment_scores: list[float] = []
    process_alignment_blocked_counts: list[int] = []
    process_alignment_missing_counts: list[int] = []
    self_evolution_scores: list[float] = []
    self_evolution_required_failure_counts: list[int] = []
    blocked_self_evolution_count = 0
    attention_self_evolution_count = 0
    top_fallback_kinds = Counter()
    top_stage_standard_risks = Counter()
    top_process_alignment_risks = Counter()
    top_self_evolution_risks = Counter()

    for row in rows:
        if row.get("submission_status") == "ready":
            ready_count += 1
        if row.get("quality_gate_passed") is True:
            gate_pass_count += 1
        if row.get("venue_match"):
            venue_match_count += 1
        if isinstance(row.get("benchmark_score"), (int, float)):
            benchmark_scores.append(float(row["benchmark_score"]))
        if isinstance(row.get("submission_priority_score"), (int, float)):
            priority_scores.append(float(row["submission_priority_score"]))
        if isinstance(row.get("blocker_count"), int):
            blocker_counts.append(int(row["blocker_count"]))
        if isinstance(row.get("fallback_count"), int):
            fallback_counts.append(int(row["fallback_count"]))
        if isinstance(row.get("strict_fallback_count"), int):
            strict_fallback_counts.append(int(row["strict_fallback_count"]))
        if isinstance(row.get("stage_overall_score"), (int, float)):
            stage_overall_scores.append(float(row["stage_overall_score"]))
        if isinstance(row.get("blocked_stage_count"), int):
            blocked_stage_counts.append(int(row["blocked_stage_count"]))
        if isinstance(row.get("needs_attention_stage_count"), int):
            attention_stage_counts.append(int(row["needs_attention_stage_count"]))
        if isinstance(row.get("missing_stage_count"), int):
            missing_stage_counts.append(int(row["missing_stage_count"]))
        if isinstance(row.get("process_alignment_overall_score"), (int, float)):
            process_alignment_scores.append(float(row["process_alignment_overall_score"]))
        if isinstance(row.get("process_alignment_blocked_process_count"), int):
            process_alignment_blocked_counts.append(
                int(row["process_alignment_blocked_process_count"])
            )
        if isinstance(row.get("process_alignment_missing_process_count"), int):
            process_alignment_missing_counts.append(
                int(row["process_alignment_missing_process_count"])
            )
        if isinstance(row.get("self_evolution_score"), (int, float)):
            self_evolution_scores.append(float(row["self_evolution_score"]))
        if isinstance(row.get("self_evolution_required_failure_count"), int):
            self_evolution_required_failure_counts.append(
                int(row["self_evolution_required_failure_count"])
            )
        if str(row.get("self_evolution_status") or "") == "blocked":
            blocked_self_evolution_count += 1
        if str(row.get("self_evolution_status") or "") == "needs_attention":
            attention_self_evolution_count += 1
        for key, value in (row.get("blocker_categories") or {}).items():
            blocker_categories[str(key)] += int(value or 0)
        for key, value in (row.get("fallback_kind_counts") or {}).items():
            top_fallback_kinds[str(key)] += int(value or 0)
        for risk in row.get("top_standard_risks") or []:
            top_stage_standard_risks[str(risk)] += 1
        for risk in row.get("top_process_alignment_risks") or []:
            top_process_alignment_risks[str(risk)] += 1
        for risk in row.get("top_self_evolution_risks") or []:
            top_self_evolution_risks[str(risk)] += 1
        for item in row.get("failing_metrics", [])[:2]:
            top_gap_dimensions[str(item.get("name"))] += 1

    thresholds = _resolve_benchmark_thresholds(target_venue)
    return {
        "generated_at": datetime.now().isoformat(),
        "research_root": str(resolved_root),
        "target_venue": target_venue,
        "include_other_venues": include_other_venues,
        "thresholds": thresholds,
        "summary": {
            "entries": len(rows),
            "venue_match_count": venue_match_count,
            "ready_count": ready_count,
            "gate_pass_count": gate_pass_count,
            "avg_benchmark_score": (
                round(sum(benchmark_scores) / len(benchmark_scores), 2)
                if benchmark_scores
                else None
            ),
            "avg_submission_priority": (
                round(sum(priority_scores) / len(priority_scores), 2)
                if priority_scores
                else None
            ),
            "avg_blocker_count": (
                round(sum(blocker_counts) / len(blocker_counts), 2)
                if blocker_counts
                else None
            ),
            "avg_fallback_count": (
                round(sum(fallback_counts) / len(fallback_counts), 2)
                if fallback_counts
                else None
            ),
            "avg_strict_fallback_count": (
                round(sum(strict_fallback_counts) / len(strict_fallback_counts), 2)
                if strict_fallback_counts
                else None
            ),
            "avg_stage_overall_score": (
                round(sum(stage_overall_scores) / len(stage_overall_scores), 2)
                if stage_overall_scores
                else None
            ),
            "avg_blocked_stage_count": (
                round(sum(blocked_stage_counts) / len(blocked_stage_counts), 2)
                if blocked_stage_counts
                else None
            ),
            "avg_attention_stage_count": (
                round(sum(attention_stage_counts) / len(attention_stage_counts), 2)
                if attention_stage_counts
                else None
            ),
            "avg_missing_stage_count": (
                round(sum(missing_stage_counts) / len(missing_stage_counts), 2)
                if missing_stage_counts
                else None
            ),
            "avg_process_alignment_score": (
                round(sum(process_alignment_scores) / len(process_alignment_scores), 2)
                if process_alignment_scores
                else None
            ),
            "avg_process_alignment_blocked_count": (
                round(
                    sum(process_alignment_blocked_counts)
                    / len(process_alignment_blocked_counts),
                    2,
                )
                if process_alignment_blocked_counts
                else None
            ),
            "avg_process_alignment_missing_count": (
                round(
                    sum(process_alignment_missing_counts)
                    / len(process_alignment_missing_counts),
                    2,
                )
                if process_alignment_missing_counts
                else None
            ),
            "avg_self_evolution_score": (
                round(sum(self_evolution_scores) / len(self_evolution_scores), 2)
                if self_evolution_scores
                else None
            ),
            "avg_self_evolution_required_failure_count": (
                round(
                    sum(self_evolution_required_failure_counts)
                    / len(self_evolution_required_failure_counts),
                    2,
                )
                if self_evolution_required_failure_counts
                else None
            ),
            "blocked_self_evolution_count": blocked_self_evolution_count,
            "needs_attention_self_evolution_count": attention_self_evolution_count,
            "top_blocker_categories": dict(blocker_categories.most_common(6)),
            "top_fallback_kinds": dict(top_fallback_kinds.most_common(6)),
            "top_stage_standard_risks": dict(top_stage_standard_risks.most_common(6)),
            "top_process_alignment_risks": dict(
                top_process_alignment_risks.most_common(6)
            ),
            "top_self_evolution_risks": dict(top_self_evolution_risks.most_common(6)),
            "top_gap_dimensions": dict(top_gap_dimensions.most_common(6)),
        },
        "ranked_papers": rows[:top_n],
        "all_papers": rows,
    }


def export_readiness_benchmark_markdown(
    benchmark: dict[str, Any],
    output_path: str | Path,
) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = benchmark.get("summary") or {}
    lines = [
        "# Readiness Benchmark",
        "",
        f"- Generated at: {benchmark.get('generated_at')}",
        f"- Research root: {benchmark.get('research_root')}",
        f"- Target venue: {benchmark.get('target_venue')}",
        f"- Entries: {summary.get('entries')}",
        f"- Venue-match count: {summary.get('venue_match_count')}",
        f"- Ready count: {summary.get('ready_count')}",
        f"- Gate-pass count: {summary.get('gate_pass_count')}",
        f"- Average benchmark score: {summary.get('avg_benchmark_score')}",
        f"- Average submission priority: {summary.get('avg_submission_priority')}",
        f"- Average blocker count: {summary.get('avg_blocker_count')}",
        f"- Average fallback count: {summary.get('avg_fallback_count')}",
        f"- Average strict fallback count: {summary.get('avg_strict_fallback_count')}",
        f"- Average stage overall score: {summary.get('avg_stage_overall_score')}",
        f"- Average blocked stage count: {summary.get('avg_blocked_stage_count')}",
        f"- Average attention stage count: {summary.get('avg_attention_stage_count')}",
        f"- Average missing stage count: {summary.get('avg_missing_stage_count')}",
        f"- Average process alignment score: {summary.get('avg_process_alignment_score')}",
        f"- Average blocked process alignment count: {summary.get('avg_process_alignment_blocked_count')}",
        f"- Average missing process alignment count: {summary.get('avg_process_alignment_missing_count')}",
        f"- Average self-evolution score: {summary.get('avg_self_evolution_score')}",
        f"- Average self-evolution required failures: {summary.get('avg_self_evolution_required_failure_count')}",
        f"- Blocked self-evolution count: {summary.get('blocked_self_evolution_count')}",
        f"- Needs-attention self-evolution count: {summary.get('needs_attention_self_evolution_count')}",
        "",
        "## Top Blocker Categories",
    ]
    for name, count in (summary.get("top_blocker_categories") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Top Fallback Kinds"])
    for name, count in (summary.get("top_fallback_kinds") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Top Stage Standard Risks"])
    for name, count in (summary.get("top_stage_standard_risks") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Top Process Alignment Risks"])
    for name, count in (summary.get("top_process_alignment_risks") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Top Self-Evolution Risks"])
    for name, count in (summary.get("top_self_evolution_risks") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Top Gap Dimensions"])
    for name, count in (summary.get("top_gap_dimensions") or {}).items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Top Papers")
    for row in benchmark.get("ranked_papers") or []:
        lines.extend(
            [
                "",
                f"### {row.get('name')}",
                f"- Benchmark score: {row.get('benchmark_score')}",
                f"- Submission status: {row.get('submission_status')}",
                f"- Gate passed: {row.get('quality_gate_passed')}",
                f"- Paper target venue: {row.get('paper_target_venue')}",
                f"- Venue match: {row.get('venue_match')}",
                f"- Submission priority: {row.get('submission_priority_score')} ({row.get('submission_priority_tier')})",
                f"- Blockers: {row.get('blocker_count')}",
                f"- Fallbacks: {row.get('fallback_count')} (strict={row.get('strict_fallback_count')})",
                f"- Stage standards: score={row.get('stage_overall_score')} blocked={row.get('blocked_stage_count')} attention={row.get('needs_attention_stage_count')} missing={row.get('missing_stage_count')}",
                f"- Process alignment: score={row.get('process_alignment_overall_score')} blocked={row.get('process_alignment_blocked_process_count')} missing={row.get('process_alignment_missing_process_count')}",
                f"- Self-evolution: status={row.get('self_evolution_status')} score={row.get('self_evolution_score')} required_failures={row.get('self_evolution_required_failure_count')}",
                f"- Recommendation: {row.get('recommendation')}",
                f"- Run dir: {row.get('relative_run_dir')}",
            ]
        )
        if row.get("failing_metrics"):
            top_gaps = ", ".join(
                f"{item.get('name')} gap={item.get('gap')}"
                for item in row["failing_metrics"][:3]
            )
            lines.append(f"- Top gaps: {top_gaps}")
        if row.get("top_blockers"):
            lines.append("- Top blockers:")
            for blocker in row["top_blockers"]:
                lines.append(f"  - {blocker}")
        if row.get("top_standard_risks"):
            lines.append(
                "- Top stage risks: "
                + ", ".join(str(item) for item in row.get("top_standard_risks") or [])
            )
        if row.get("top_process_alignment_risks"):
            lines.append(
                "- Top process-alignment risks: "
                + ", ".join(
                    str(item) for item in row.get("top_process_alignment_risks") or []
                )
            )
        if row.get("top_self_evolution_risks"):
            lines.append(
                "- Top self-evolution risks: "
                + ", ".join(
                    str(item) for item in row.get("top_self_evolution_risks") or []
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
