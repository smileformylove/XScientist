from __future__ import annotations

from typing import Any


def get_active_source_feedback(status: dict[str, Any]) -> dict[str, Any]:
    active_source = status.get("active_source") or {}
    feedback_map = status.get("source_quality_feedback") or {}
    by_name = feedback_map.get(str(active_source.get("name")))
    by_key = feedback_map.get(str(status.get("active_source_key")))
    return by_name or by_key or {}


def build_active_source_feedback_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    feedback = get_active_source_feedback(status)
    return {
        "source_name": feedback.get("source_name"),
        "avg_experiment_todo": feedback.get("avg_experiment_todo"),
        "avg_experiment_todo_p0": feedback.get("avg_experiment_todo_p0"),
        "avg_experiment_todo_closure_rate": feedback.get(
            "avg_experiment_todo_closure_rate"
        ),
        "experiment_todo_pressure_rate": feedback.get("experiment_todo_pressure_rate"),
        "self_review_gate_ready_rate": feedback.get("self_review_gate_ready_rate"),
    }
