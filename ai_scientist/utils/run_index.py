from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ai_scientist.config.paths import resolve_output_path
from ai_scientist.utils.process_alignment import build_process_alignment
from ai_scientist.utils.review_jobs import compute_review_repair_metrics
from ai_scientist.utils.self_evolution import build_self_evolution
from ai_scientist.utils.stage_standards import build_stage_standards

WORKFLOW_STATE_FILE = ".workflow_state.json"
RUN_INDEX_FILE = ".run_index.json"
PIPELINE_MANIFEST_FILE = "pipeline_manifest.json"
STAGE_STANDARDS_FILE = "stage_standards.json"
SELF_EVOLUTION_FILE = "self_evolution.json"
PROCESS_ALIGNMENT_FILE = "process_alignment.json"
STAGE_ORDER = ["prepare", "experiment", "writeup", "review"]


def _now() -> str:
    return datetime.now().isoformat()


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return []


def _coerce_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _coerce_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _coerce_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _resolve_text_field(*values: Any) -> Optional[str]:
    for value in values:
        text = _coerce_text(value)
        if text:
            return text
    return None


def _resolve_stage_sequence(
    run_dir: Path,
    state: dict[str, Any],
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    manifest = _safe_read_json(run_dir / PIPELINE_MANIFEST_FILE)
    declared_stage_sequence = _coerce_str_list(manifest.get("workflow_sequence"))
    observed_stage_sequence = _coerce_str_list(list((state.get("stages") or {}).keys()))
    combined: list[str] = []
    seen: set[str] = set()
    for stage in observed_stage_sequence + declared_stage_sequence + STAGE_ORDER:
        name = str(stage).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        combined.append(name)
    return combined, declared_stage_sequence, observed_stage_sequence, manifest


def _resolve_run_index_root(output_root: str | Path | None = None) -> Path:
    if output_root is None:
        return resolve_output_path().resolve()
    return Path(output_root).expanduser().resolve()


def _load_stage_standards(run_dir: Path) -> dict[str, Any]:
    standards = _safe_read_json(run_dir / STAGE_STANDARDS_FILE)
    if isinstance(standards, dict) and (
        "stage_results" in standards or "overall_score" in standards
    ):
        return standards
    try:
        computed = build_stage_standards(run_dir)
    except Exception:
        return {}
    return computed if isinstance(computed, dict) else {}


def _load_self_evolution(run_dir: Path) -> dict[str, Any]:
    evolution = _safe_read_json(run_dir / SELF_EVOLUTION_FILE)
    if isinstance(evolution, dict) and (
        "summary" in evolution or "self_check" in evolution
    ):
        return evolution
    try:
        computed = build_self_evolution(run_dir)
    except Exception:
        return {}
    return computed if isinstance(computed, dict) else {}


def _load_process_alignment(run_dir: Path) -> dict[str, Any]:
    alignment = _safe_read_json(run_dir / PROCESS_ALIGNMENT_FILE)
    if isinstance(alignment, dict) and (
        "summary" in alignment
        or "process_results" in alignment
        or "reference_summary" in alignment
    ):
        return alignment
    try:
        computed = build_process_alignment(run_dir)
    except Exception:
        return {}
    return computed if isinstance(computed, dict) else {}


def workflow_state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / WORKFLOW_STATE_FILE


def run_index_path(output_root: str | Path | None = None) -> Path:
    return _resolve_run_index_root(output_root) / RUN_INDEX_FILE


def load_workflow_state(run_dir: str | Path) -> dict:
    path = workflow_state_path(run_dir)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"stages": {}, "artifacts": {}, "updated_at": None}


def save_workflow_state(run_dir: str | Path, state: dict) -> dict:
    state["updated_at"] = _now()
    path = workflow_state_path(run_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return state


def _stage_artifact_defaults(run_dir: Path, stage: str) -> bool:
    if stage == "prepare":
        return (run_dir / "idea.json").exists() and (run_dir / "idea.md").exists()
    if stage == "experiment":
        return (run_dir / "logs").exists()
    if stage == "writeup":
        return any(run_dir.glob("*.pdf")) or (run_dir / "latex").exists()
    if stage == "review":
        review_markers = [
            "review_text.txt",
            "review_text.json",
            "review_img.json",
            "review_img_cap_ref.json",
            "final_review.json",
            "final_review_img.json",
        ]
        return (
            any((run_dir / marker).exists() for marker in review_markers)
            or (run_dir / "reviews").exists()
        )
    return False


def is_stage_complete(run_dir: str | Path, stage: str) -> bool:
    run_dir = Path(run_dir)
    state = load_workflow_state(run_dir)
    stage_state = state.get("stages", {}).get(stage, {})
    if stage_state.get("status") == "completed":
        return True
    return _stage_artifact_defaults(run_dir, stage)


def mark_stage_complete(
    run_dir: str | Path,
    stage: str,
    *,
    artifacts: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
    run_dir = Path(run_dir)
    state = load_workflow_state(run_dir)
    state.setdefault("stages", {})[stage] = {
        "status": "completed",
        "updated_at": _now(),
        "artifacts": artifacts or {},
        "metadata": metadata or {},
    }
    if artifacts:
        state.setdefault("artifacts", {}).update(artifacts)
    save_workflow_state(run_dir, state)
    update_run_index_entry(run_dir, state=state)
    return state


def infer_run_entry(
    run_dir: str | Path,
    state: Optional[dict] = None,
    *,
    output_root: str | Path | None = None,
) -> dict:
    run_dir = Path(run_dir)
    output_root = _resolve_run_index_root(output_root)
    if state is None:
        state = load_workflow_state(run_dir)

    try:
        relative_path = str(run_dir.resolve().relative_to(output_root))
        category = relative_path.split("/", 1)[0]
    except ValueError:
        relative_path = str(run_dir.resolve())
        category = "external"

    stage_sequence, declared_stage_sequence, observed_stage_sequence, manifest = (
        _resolve_stage_sequence(run_dir, state)
    )
    completed_stages = []
    for stage in stage_sequence:
        if is_stage_complete(run_dir, stage):
            completed_stages.append(stage)

    latest_stage = completed_stages[-1] if completed_stages else "initialized"
    pdf_files = sorted(path.name for path in run_dir.glob("*.pdf"))
    quality_result_path = run_dir / "quality" / "high_quality_result.json"
    quality_result = _safe_read_json(quality_result_path)
    source_provenance_path = run_dir / "source_provenance.json"
    source_provenance = _safe_read_json(source_provenance_path)
    benchmark_suite = _resolve_text_field(
        source_provenance.get("benchmark_suite"),
        manifest.get("benchmark_suite"),
    )
    benchmark_topic = _resolve_text_field(
        source_provenance.get("benchmark_topic"),
        manifest.get("benchmark_topic"),
        source_provenance.get("source_key"),
        source_provenance.get("source_name"),
        source_provenance.get("source_value"),
        run_dir.name,
    )
    variant_label = _resolve_text_field(
        source_provenance.get("variant_label"),
        manifest.get("variant_label"),
        manifest.get("workflow_label"),
        manifest.get("workflow_mode"),
    )
    ablation_group = _resolve_text_field(
        source_provenance.get("ablation_group"),
        manifest.get("ablation_group"),
    )
    ablation_label = _resolve_text_field(
        source_provenance.get("ablation_label"),
        manifest.get("ablation_label"),
        variant_label if ablation_group else None,
    )
    fallback_summary = (
        manifest.get("fallback_summary")
        if isinstance(manifest.get("fallback_summary"), dict)
        else {}
    )
    stage_standards = _load_stage_standards(run_dir)
    stage_summary = (
        stage_standards.get("summary")
        if isinstance(stage_standards.get("summary"), dict)
        else {}
    )
    self_review_summary = _safe_read_json(run_dir / "self_review_iteration_summary.json")
    self_review_final_progress = _safe_read_json(run_dir / "self_review_final_progress.json")
    improvement_record = _safe_read_json(run_dir / "improvement_record.json")
    review_state = _safe_read_json(run_dir / "review_state.json")
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
    repair_plan = _safe_read_json(run_dir / "repair_plan.json")
    self_evolution = _load_self_evolution(run_dir)
    process_alignment = _load_process_alignment(run_dir)
    review_repair_metrics = compute_review_repair_metrics(review_state)
    repair_plan_summary = (
        repair_plan.get("summary")
        if isinstance(repair_plan.get("summary"), dict)
        else {}
    )
    self_evolution_summary = (
        self_evolution.get("summary")
        if isinstance(self_evolution.get("summary"), dict)
        else {}
    )
    self_evolution_self_check = (
        self_evolution.get("self_check")
        if isinstance(self_evolution.get("self_check"), dict)
        else {}
    )
    self_evolution_defaults = (
        self_evolution.get("next_cycle_defaults")
        if isinstance(self_evolution.get("next_cycle_defaults"), dict)
        else {}
    )
    process_alignment_summary = (
        process_alignment.get("summary")
        if isinstance(process_alignment.get("summary"), dict)
        else {}
    )

    rounds = (
        self_review_summary.get("rounds")
        if isinstance(self_review_summary.get("rounds"), list)
        else []
    )
    latest_round = rounds[-1] if rounds else {}
    latest_round_gate = self_review_summary.get("latest_round_gate")
    if not isinstance(latest_round_gate, dict):
        latest_round_gate = (
            dict(latest_round.get("round_gate") or {})
            if isinstance(latest_round, dict)
            else {}
        )
    if not latest_round_gate:
        latest_round_gate = (
            dict(improvement_record.get("final_round_gate") or {})
            if isinstance(improvement_record.get("final_round_gate"), dict)
            else {}
        )
    latest_rewrite = (
        dict(latest_round.get("rewrite") or {})
        if isinstance(latest_round, dict)
        else {}
    )
    if not latest_rewrite:
        latest_rewrite = (
            dict(improvement_record.get("rounds", [{}])[-1].get("rewrite_result") or {})
            if isinstance(improvement_record.get("rounds"), list)
            and improvement_record.get("rounds")
            and isinstance(improvement_record.get("rounds")[-1], dict)
            else {}
        )

    final_progress = self_review_final_progress.get("final_progress")
    if not isinstance(final_progress, dict):
        final_progress = {}
    gate_metrics = latest_round_gate.get("metrics")
    if not isinstance(gate_metrics, dict):
        gate_metrics = {}

    self_review_unresolved_critical = (
        _coerce_int(final_progress.get("unresolved_critical_count"))
        if final_progress
        else _coerce_int(gate_metrics.get("unresolved_critical_count"))
    )
    self_review_persistent_issues = (
        _coerce_int(final_progress.get("persistent_issue_count"))
        if final_progress
        else _coerce_int(gate_metrics.get("persistent_issue_count"))
    )
    self_review_round_gate_ready = (
        bool(latest_round_gate.get("ready"))
        if latest_round_gate
        else (
            True
            if self_review_summary.get("round_gate_ready") is True
            else False
            if self_review_summary.get("round_gate_ready") is False
            else None
        )
    )
    self_review_round_gate_score = _coerce_float(latest_round_gate.get("score"))
    self_review_round_gate_reasons = _coerce_str_list(
        latest_round_gate.get("reasons")
    )
    self_review_next_focus = _coerce_str_list(
        latest_round_gate.get("next_focus_summaries")
        or latest_round_gate.get("next_focus_issue_ids")
    )
    self_review_rounds_completed = _coerce_int(
        self_review_summary.get("rounds_completed")
    )
    if self_review_rounds_completed is None and isinstance(improvement_record.get("total_rounds"), int):
        self_review_rounds_completed = int(improvement_record.get("total_rounds"))
    self_review_high_value_coverage = _coerce_float(
        latest_rewrite.get("high_value_coverage_ratio")
    )
    self_review_coverage = _coerce_float(latest_rewrite.get("coverage_ratio"))
    self_review_round_gate_file = ""
    if isinstance(latest_round, dict):
        self_review_round_gate_file = str(
            (latest_round.get("artifacts") or {}).get("round_gate") or ""
        ).strip()
    experiment_todo_path = run_dir / "experiment_todo.json"
    experiment_todo_payload = _safe_read_json(experiment_todo_path)
    experiment_todo_tasks = (
        experiment_todo_payload.get("tasks")
        if isinstance(experiment_todo_payload.get("tasks"), list)
        else []
    )
    experiment_todo_count = len(experiment_todo_tasks)
    experiment_todo_p0_count = sum(
        str(item.get("priority") or "").upper() == "P0"
        for item in experiment_todo_tasks
        if isinstance(item, dict)
    )
    if (
        experiment_todo_p0_count == 0
        and isinstance(experiment_todo_payload.get("counts"), dict)
    ):
        experiment_todo_p0_count = _coerce_int(
            experiment_todo_payload.get("counts", {}).get("p0_tasks")
        ) or 0
    experiment_todo_top_action = ""
    if experiment_todo_tasks:
        ranked_tasks = sorted(
            [item for item in experiment_todo_tasks if isinstance(item, dict)],
            key=lambda item: (
                {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(
                    str(item.get("priority") or "").upper(), 9
                ),
                str(item.get("task_id") or ""),
            ),
        )
        if ranked_tasks:
            experiment_todo_top_action = str(
                ranked_tasks[0].get("action") or ""
            ).strip()
    experiment_todo_progress_path = run_dir / "experiment_todo_progress.json"
    experiment_todo_progress_payload = _safe_read_json(experiment_todo_progress_path)
    final_todo_snapshot = (
        experiment_todo_progress_payload.get("final_snapshot")
        if isinstance(experiment_todo_progress_payload.get("final_snapshot"), dict)
        else {}
    )
    final_todo_counts = (
        final_todo_snapshot.get("counts")
        if isinstance(final_todo_snapshot.get("counts"), dict)
        else {}
    )
    experiment_todo_closed_count = _coerce_int(final_todo_counts.get("closed_tasks"))
    experiment_todo_unresolved_count = _coerce_int(
        final_todo_counts.get("unresolved_tasks")
    )
    experiment_todo_closure_rate = _coerce_float(final_todo_snapshot.get("closure_rate"))
    experiment_todo_p0_closure_rate = _coerce_float(
        final_todo_snapshot.get("p0_closure_rate")
    )

    return {
        "path": str(run_dir),
        "relative_path": relative_path,
        "category": category,
        "workflow_mode": manifest.get("workflow_mode"),
        "workflow_label": manifest.get("workflow_label"),
        "stage_sequence": stage_sequence,
        "declared_stage_sequence": declared_stage_sequence,
        "observed_stage_sequence": observed_stage_sequence,
        "updated_at": state.get("updated_at")
        or datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
        "completed_stages": completed_stages,
        "latest_stage": latest_stage,
        "pdf_files": pdf_files,
        "has_reviews": is_stage_complete(run_dir, "review"),
        "has_latex": (run_dir / "latex").exists(),
        "batch_name": source_provenance.get("batch_name"),
        "batch_dir": source_provenance.get("batch_dir"),
        "daemon_name": source_provenance.get("daemon_name"),
        "source_name": source_provenance.get("source_name"),
        "source_key": source_provenance.get("source_key"),
        "source_type": source_provenance.get("source_type"),
        "source_value": source_provenance.get("source_value"),
        "source_target_venue": source_provenance.get("source_target_venue"),
        "source_paper_types": source_provenance.get("source_paper_types") or [],
        "source_workflow_mode": source_provenance.get("source_workflow_mode"),
        "source_archetype": source_provenance.get("source_archetype"),
        "source_batch_profile": source_provenance.get("source_batch_profile"),
        "benchmark_suite": benchmark_suite,
        "benchmark_topic": benchmark_topic,
        "variant_label": variant_label,
        "ablation_group": ablation_group,
        "ablation_label": ablation_label,
        "fallback_count": _coerce_int(fallback_summary.get("count")) or 0,
        "strict_fallback_count": _coerce_int(fallback_summary.get("strict_count"))
        or 0,
        "fallback_stage_counts": fallback_summary.get("stage_counts") or {},
        "fallback_kind_counts": fallback_summary.get("kind_counts") or {},
        "latest_fallback_event": fallback_summary.get("latest_event") or {},
        "stage_standards_file": (
            str(run_dir / STAGE_STANDARDS_FILE)
            if (run_dir / STAGE_STANDARDS_FILE).exists()
            else None
        ),
        "repair_plan_file": (
            str(run_dir / "repair_plan.json")
            if (run_dir / "repair_plan.json").exists()
            else None
        ),
        "self_evolution_file": (
            str(run_dir / SELF_EVOLUTION_FILE)
            if (run_dir / SELF_EVOLUTION_FILE).exists()
            else None
        ),
        "process_alignment_file": (
            str(run_dir / PROCESS_ALIGNMENT_FILE)
            if (run_dir / PROCESS_ALIGNMENT_FILE).exists()
            else None
        ),
        "stage_overall_score": _coerce_float(stage_standards.get("overall_score")),
        "ready_stage_count": _coerce_int(stage_standards.get("ready_stage_count")),
        "blocked_stage_count": _coerce_int(
            stage_standards.get("blocked_stage_count")
        ),
        "needs_attention_stage_count": _coerce_int(
            stage_standards.get("needs_attention_stage_count")
        ),
        "missing_stage_count": _coerce_int(stage_standards.get("missing_stage_count")),
        "blocked_standard_stages": stage_summary.get("blocked_stages") or [],
        "attention_standard_stages": stage_summary.get("attention_stages") or [],
        "missing_standard_stages": stage_summary.get("missing_stages") or [],
        "top_standard_risks": stage_summary.get("top_risks") or [],
        "self_review_rounds_completed": self_review_rounds_completed,
        "self_review_round_gate_ready": self_review_round_gate_ready,
        "self_review_round_gate_score": self_review_round_gate_score,
        "self_review_round_gate_reasons": self_review_round_gate_reasons,
        "self_review_round_gate_file": self_review_round_gate_file or None,
        "self_review_unresolved_critical": self_review_unresolved_critical,
        "self_review_persistent_issues": self_review_persistent_issues,
        "review_active_issue_count": review_repair_metrics.get("active_issue_count"),
        "review_resolved_issue_count": review_repair_metrics.get("resolved_issue_count"),
        "review_persistent_issue_count": review_repair_metrics.get(
            "persistent_issue_count"
        ),
        "review_repair_action_count": review_repair_metrics.get("repair_action_count"),
        "review_verification_count": review_repair_metrics.get("verification_count"),
        "review_bound_issue_count": review_repair_metrics.get("bound_issue_count"),
        "review_unbound_issue_count": review_repair_metrics.get("unbound_issue_count"),
        "review_bound_active_issue_count": review_repair_metrics.get(
            "bound_active_issue_count"
        ),
        "review_target_binding_coverage": review_repair_metrics.get(
            "target_binding_coverage"
        ),
        "review_active_binding_coverage": review_repair_metrics.get(
            "active_binding_coverage"
        ),
        "review_role_count": review_repair_metrics.get("role_count"),
        "review_role_coverage_ratio": review_repair_metrics.get(
            "role_coverage_ratio"
        ),
        "review_resolution_rate": review_repair_metrics.get("resolution_rate"),
        "review_verification_coverage": review_repair_metrics.get(
            "verification_coverage"
        ),
        "review_repair_queue_count": review_repair_metrics.get("repair_queue_count"),
        "review_repair_ready_count": review_repair_metrics.get("repair_ready_count"),
        "review_repair_verification_ready_count": review_repair_metrics.get(
            "repair_verification_ready_count"
        ),
        "review_repair_targeted_count": review_repair_metrics.get(
            "repair_targeted_count"
        ),
        "review_repair_queue_coverage": review_repair_metrics.get(
            "repair_queue_coverage"
        ),
        "review_repair_ready_coverage": review_repair_metrics.get(
            "repair_ready_coverage"
        ),
        "review_repair_verification_ready_coverage": review_repair_metrics.get(
            "repair_verification_ready_coverage"
        ),
        "review_repair_targeted_coverage": review_repair_metrics.get(
            "repair_targeted_coverage"
        ),
        "critic_findings_file": (
            str(run_dir / "critic_findings.json")
            if (run_dir / "critic_findings.json").exists()
            else None
        ),
        "critic_active_issue_count": hostile_critic_summary.get("active_issue_count"),
        "critic_blocking_issue_count": hostile_critic_summary.get(
            "blocking_issue_count"
        ),
        "critic_role_count": len(hostile_critic_summary.get("roles") or []),
        "critic_strictness_profile": hostile_critic_summary.get(
            "strictness_profile"
        ),
        "repair_plan_task_count": _coerce_int(repair_plan_summary.get("task_count")),
        "repair_plan_ready_task_count": _coerce_int(
            repair_plan_summary.get("ready_task_count")
        ),
        "repair_plan_blocked_task_count": _coerce_int(
            repair_plan_summary.get("blocked_task_count")
        ),
        "repair_plan_verification_ready_count": _coerce_int(
            repair_plan_summary.get("verification_ready_count")
        ),
        "repair_plan_lane_count": _coerce_int(repair_plan_summary.get("lane_count")),
        "repair_plan_ready_rate": _coerce_float(repair_plan_summary.get("ready_rate")),
        "repair_plan_verification_ready_rate": _coerce_float(
            repair_plan_summary.get("verification_ready_rate")
        ),
        "repair_plan_targeted_rate": _coerce_float(
            repair_plan_summary.get("targeted_rate")
        ),
        "self_evolution_status": str(self_evolution_summary.get("status") or "").strip()
        or None,
        "self_evolution_score": _coerce_float(self_evolution_summary.get("score")),
        "self_evolution_lesson_count": _coerce_int(
            self_evolution_summary.get("lesson_count")
        ),
        "self_evolution_required_failure_count": _coerce_int(
            self_evolution_summary.get("required_failure_count")
        )
        or len(_coerce_str_list(self_evolution_self_check.get("required_failures"))),
        "self_evolution_dominant_lane": str(
            self_evolution_summary.get("dominant_lane") or ""
        ).strip()
        or None,
        "self_evolution_dominant_role": str(
            self_evolution_summary.get("dominant_role") or ""
        ).strip()
        or None,
        "self_evolution_next_cycle_stages": sorted(
            str(name).strip()
            for name in self_evolution_defaults.keys()
            if str(name).strip()
        ),
        "self_evolution_required_failures": _coerce_str_list(
            self_evolution_self_check.get("required_failures")
        ),
        "self_evolution_top_risks": _coerce_str_list(
            self_evolution.get("stage_risks")
        ),
        "process_alignment_overall_score": _coerce_float(
            process_alignment_summary.get("overall_score")
        ),
        "process_alignment_ready_process_count": _coerce_int(
            process_alignment_summary.get("ready_process_count")
        ),
        "process_alignment_blocked_process_count": _coerce_int(
            process_alignment_summary.get("blocked_process_count")
        ),
        "process_alignment_attention_process_count": _coerce_int(
            process_alignment_summary.get("needs_attention_process_count")
        ),
        "process_alignment_missing_process_count": _coerce_int(
            process_alignment_summary.get("missing_process_count")
        ),
        "process_alignment_top_risks": _coerce_str_list(
            list((process_alignment_summary.get("top_process_risks") or {}).keys())
        ),
        "self_review_high_value_coverage": self_review_high_value_coverage,
        "self_review_coverage": self_review_coverage,
        "self_review_focus_issue_count": len(self_review_next_focus),
        "self_review_next_focus": self_review_next_focus,
        "experiment_todo_count": experiment_todo_count,
        "experiment_todo_p0_count": experiment_todo_p0_count,
        "experiment_todo_top_action": experiment_todo_top_action or None,
        "experiment_todo_file": (
            str(experiment_todo_path) if experiment_todo_path.exists() else None
        ),
        "experiment_todo_closed_count": experiment_todo_closed_count,
        "experiment_todo_unresolved_count": experiment_todo_unresolved_count,
        "experiment_todo_closure_rate": experiment_todo_closure_rate,
        "experiment_todo_p0_closure_rate": experiment_todo_p0_closure_rate,
        "experiment_todo_progress_file": (
            str(experiment_todo_progress_path)
            if experiment_todo_progress_path.exists()
            else None
        ),
        "quality_score": quality_result.get("quality_score_after"),
        "rigor_score": quality_result.get("rigor_score_after"),
        "claim_support_score": quality_result.get("claim_support_after"),
        "claim_alignment_score": quality_result.get("claim_alignment_after"),
        "numeric_coverage_score": quality_result.get("numeric_coverage_after"),
        "breakthrough_score": quality_result.get("breakthrough_score"),
        "claims_detected": quality_result.get("claims_detected"),
        "unsupported_claims_count": quality_result.get("unsupported_claims_count"),
        "suggested_claim_rewrites_count": quality_result.get(
            "suggested_claim_rewrites_count"
        ),
        "num_figures": quality_result.get("num_figures"),
        "num_tables": quality_result.get("num_tables"),
        "evidence_density_score": quality_result.get("evidence_density_score"),
        "key_results_count": quality_result.get("key_results_count"),
        "structured_results_count": quality_result.get("structured_results_count"),
        "contribution_count": quality_result.get("contribution_count"),
        "target_venue": quality_result.get("target_venue"),
        "submission_status": quality_result.get("submission_readiness", {}).get(
            "status"
        ),
        "submission_package_file": quality_result.get("submission_package_file"),
        "claim_alignment_file": quality_result.get("claim_alignment_file"),
        "narrative_map_file": quality_result.get("narrative_map_file"),
        "result_story_file": quality_result.get("result_story_file"),
        "contribution_map_file": quality_result.get("contribution_map_file"),
        "editor_pitch_file": quality_result.get("editor_pitch_file"),
        "rebuttal_package_file": quality_result.get("rebuttal_package_file"),
        "risk_register_file": quality_result.get("risk_register_file"),
        "cover_letter_file": quality_result.get("cover_letter_file"),
        "abstract_polish_file": quality_result.get("abstract_polish_file"),
        "impact_brief_file": quality_result.get("impact_brief_file"),
        "contribution_bullets_file": quality_result.get("contribution_bullets_file"),
        "strongest_claims_file": quality_result.get("strongest_claims_file"),
        "submission_manifest_file": quality_result.get("submission_manifest_file"),
        "submission_dashboard_file": quality_result.get("submission_dashboard_file"),
        "risk_language_plan_file": quality_result.get("risk_language_plan_file"),
        "claim_softening_plan_file": quality_result.get("claim_softening_plan_file"),
        "rewrite_effectiveness_file": quality_result.get("rewrite_effectiveness_file"),
        "rewrite_trace_summary_file": quality_result.get("rewrite_trace_summary_file"),
        "rewrite_round_count": quality_result.get(
            "rewrite_effectiveness_summary", {}
        ).get("round_count"),
        "rewrite_priority_gain_total": quality_result.get(
            "rewrite_effectiveness_summary", {}
        ).get("priority_gain_total"),
        "rewrite_quality_gain_total": quality_result.get(
            "rewrite_effectiveness_summary", {}
        ).get("quality_gain_total"),
        "rewrite_best_round_priority_delta": (
            quality_result.get("rewrite_effectiveness_summary", {}).get("best_round")
            or {}
        ).get("priority_delta"),
        "rewrite_top_frontmatter_style": quality_result.get(
            "rewrite_effectiveness_summary", {}
        ).get("top_frontmatter_style"),
        "rewrite_top_section_style": quality_result.get(
            "rewrite_effectiveness_summary", {}
        ).get("top_section_style"),
        "rewrite_top_section": quality_result.get(
            "rewrite_effectiveness_summary", {}
        ).get("top_section"),
        "submission_priority_score": quality_result.get("submission_priority_score"),
        "submission_priority_tier": quality_result.get("submission_priority_tier"),
        "blocker_count": quality_result.get("blocker_count"),
        "critical_revision_actions_count": quality_result.get(
            "critical_revision_actions_count"
        ),
        "quality_rewrite_applied": quality_result.get("rewrite_applied"),
        "quality_gate_passed": quality_result.get("quality_gate_passed"),
        "quality_status": quality_result.get("quality_status"),
        "auto_improvement_fallback_used": quality_result.get(
            "auto_improvement_fallback_used"
        ),
        "auto_improvement_fallback_reason": quality_result.get(
            "auto_improvement_fallback_reason"
        ),
    }


def load_run_index(output_root: str | Path | None = None) -> dict:
    path = run_index_path(output_root)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"generated_at": None, "entries": {}}


def save_run_index(index: dict, output_root: str | Path | None = None) -> dict:
    index["generated_at"] = _now()
    path = run_index_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index


def update_run_index_entry(
    run_dir: str | Path,
    state: Optional[dict] = None,
    *,
    output_root: str | Path | None = None,
) -> dict:
    index = load_run_index(output_root)
    entry = infer_run_entry(run_dir, state=state, output_root=output_root)
    index.setdefault("entries", {})[entry["relative_path"]] = entry
    save_run_index(index, output_root=output_root)
    return entry


def _infer_batch_entry(
    batch_dir: Path, *, output_root: str | Path | None = None
) -> dict:
    output_root = _resolve_run_index_root(output_root)
    progress = {}
    progress_file = batch_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            progress = json.load(f)

    return {
        "path": str(batch_dir),
        "relative_path": str(batch_dir.resolve().relative_to(output_root)),
        "category": "batches",
        "updated_at": progress.get("last_updated")
        or datetime.fromtimestamp(batch_dir.stat().st_mtime).isoformat(),
        "latest_stage": progress.get("current_stage", "unknown"),
        "completed": len(progress.get("papers_completed", [])),
        "failed": len(progress.get("papers_failed", [])),
    }


def rebuild_run_index(output_root: str | Path | None = None) -> dict:
    output_root = _resolve_run_index_root(output_root)
    entries = {}

    for base_dir in [
        output_root / "experiments",
        output_root / "papers",
        output_root / "projects",
    ]:
        if not base_dir.exists():
            continue
        for run_dir in sorted(base_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            entry = infer_run_entry(run_dir, output_root=output_root)
            entries[entry["relative_path"]] = entry

    batches_dir = output_root / "batches"
    if batches_dir.exists():
        for batch_dir in sorted(batches_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            entry = _infer_batch_entry(batch_dir, output_root=output_root)
            entries[entry["relative_path"]] = entry

    index = {"generated_at": _now(), "entries": entries}
    save_run_index(index, output_root=output_root)
    return index
