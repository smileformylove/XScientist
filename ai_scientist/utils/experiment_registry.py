from __future__ import annotations

"""Experiment registry helpers inspired by explicit experiment managers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    artifact_path,
    load_jsonl_artifact,
    update_pipeline_artifact,
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def build_experiment_record(
    *,
    task_id: str,
    dataset: str,
    metric: str,
    baseline_ref: str,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    status: str = "planned",
    result_summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    entered_storyline: bool = False,
    budget: dict[str, Any] | None = None,
    workflow_mode: str | None = None,
    policy_name: str | None = None,
    acceptance_checks: list[str] | None = None,
    budget_status: str | None = None,
) -> dict[str, Any]:
    error_tokens = " ".join(
        [
            str(error_type or "").strip().lower(),
            str(error_message or "").strip().lower(),
        ]
    )
    if budget_status is None:
        if status in {"planned", "running"}:
            budget_status = "not_started"
        elif "budget" in error_tokens or "timeout" in error_tokens:
            budget_status = "budget_exhausted"
        else:
            budget_status = "within_budget"
    return {
        "record_id": f"{task_id}_{_now_iso()}",
        "task_id": task_id,
        "dataset": dataset,
        "metric": metric,
        "baseline_ref": baseline_ref,
        "config": config or {},
        "seed": seed,
        "status": status,
        "result_summary": result_summary or {},
        "artifacts": artifacts or {},
        "error_type": error_type,
        "error_message": error_message,
        "entered_storyline": bool(entered_storyline),
        "budget": budget or {},
        "budget_status": budget_status,
        "workflow_mode": workflow_mode,
        "policy_name": policy_name or workflow_mode,
        "acceptance_checks": list(acceptance_checks or []),
        "started_at": _now_iso(),
        "finished_at": None if status in {"planned", "running"} else _now_iso(),
    }


def append_experiment_record(project_root: str | Path, record: dict[str, Any]) -> str:
    output_path = artifact_path(project_root, "experiment_registry")
    append_jsonl_artifact(output_path, record)
    update_pipeline_artifact(
        project_root,
        "experiment_registry",
        status="ready",
        producer="experiment_registry",
        depends_on=["research_plan"],
    )
    from ai_scientist.utils.stage_standards import save_stage_standards

    save_stage_standards(project_root)
    return str(output_path)


def load_experiment_records(project_root: str | Path) -> list[dict[str, Any]]:
    return load_jsonl_artifact(artifact_path(project_root, "experiment_registry"))


def summarize_experiment_registry(project_root: str | Path) -> dict[str, Any]:
    records = load_experiment_records(project_root)
    summary = {
        "total": len(records),
        "by_status": {},
        "by_budget_status": {},
        "policy_names": {},
        "storyline_count": 0,
        "datasets": {},
    }
    for record in records:
        status = str(record.get("status") or "unknown")
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        budget_status = str(record.get("budget_status") or "unknown")
        summary["by_budget_status"][budget_status] = (
            summary["by_budget_status"].get(budget_status, 0) + 1
        )
        policy_name = str(record.get("policy_name") or "unknown")
        summary["policy_names"][policy_name] = (
            summary["policy_names"].get(policy_name, 0) + 1
        )
        if record.get("entered_storyline"):
            summary["storyline_count"] += 1
        dataset = str(record.get("dataset") or "unknown")
        summary["datasets"][dataset] = summary["datasets"].get(dataset, 0) + 1
    return summary


def save_experiment_registry(project_root: str | Path, rows: list[dict[str, Any]]) -> str:
    output_path = Path(artifact_path(project_root, "experiment_registry"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    update_pipeline_artifact(
        project_root,
        "experiment_registry",
        status="ready",
        producer="experiment_registry",
        depends_on=["research_plan"],
    )
    from ai_scientist.utils.stage_standards import save_stage_standards

    save_stage_standards(project_root)
    return str(output_path)
