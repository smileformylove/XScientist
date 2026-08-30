from __future__ import annotations

"""Sample-gate helpers for low-cost validation before full execution."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.experiment_registry import load_verified_experiment_records
from ai_scientist.utils.pipeline_contracts import (
    load_contract_artifact,
    save_contract_artifact,
    save_json_artifact,
    update_pipeline_artifact,
)
from ai_scientist.utils.research_integrity import ResearchIntegrityError

DEFAULT_SAMPLE_TASK_COUNT = 1


class SampleGateBlocked(RuntimeError):
    """Raised when full execution is attempted before the sample gate passes."""


def _now_iso() -> str:
    return datetime.now().isoformat()


def _coerce_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _acceptance_results_passed(record: dict[str, Any]) -> bool:
    results = record.get("acceptance_results")
    if not isinstance(results, list) or not results:
        return False
    for item in results:
        if not isinstance(item, dict) or item.get("passed") is not True:
            return False
    return True


def _budget_audit_passed(record: dict[str, Any]) -> bool:
    budget_status = str(record.get("budget_status") or "").strip()
    if budget_status != "within_budget":
        return False
    audit = record.get("budget_audit")
    if not isinstance(audit, dict):
        return False
    return audit.get("audited") is True and audit.get("within_budget") is True


def build_sample_gate_plan(
    research_plan: dict[str, Any],
    *,
    sample_task_count: int = DEFAULT_SAMPLE_TASK_COUNT,
    mode: str = "sample_then_full",
) -> dict[str, Any]:
    tasks = [
        item for item in (research_plan.get("tasks") or []) if isinstance(item, dict)
    ]
    sample_count = min(
        _coerce_positive_int(sample_task_count, DEFAULT_SAMPLE_TASK_COUNT), len(tasks)
    )
    sample_tasks = []
    for idx, task in enumerate(tasks[:sample_count]):
        task_id = str(task.get("task_id") or f"task_{idx}").strip()
        sample_tasks.append(
            {
                "task_id": task_id,
                "goal": task.get("goal"),
                "dataset": task.get("dataset"),
                "metric": task.get("metric"),
                "baseline": task.get("baseline"),
                "acceptance_checks": list(task.get("acceptance_checks") or []),
                "required_outputs": list(task.get("expected_outputs") or []),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "mode": str(mode or "sample_then_full"),
        "workflow_mode": research_plan.get("workflow_mode"),
        "status": "planned" if sample_tasks else "blocked",
        "full_generation_allowed": False,
        "sample_task_count": len(sample_tasks),
        "sample_tasks": sample_tasks,
        "checks": [
            "sample_task_has_completed_experiment_record",
            "sample_record_within_budget",
            "sample_record_has_passing_acceptance_results",
            "sample_record_has_result_summary",
        ],
        "result": {
            "passed": False,
            "reasons": ["sample_not_run"] if sample_tasks else ["no_sample_tasks"],
        },
    }


def _latest_record_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            continue
        latest[task_id] = record
    return latest


def evaluate_sample_gate(
    sample_gate: dict[str, Any],
    *,
    experiment_records: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_tasks = [
        item
        for item in (sample_gate.get("sample_tasks") or [])
        if isinstance(item, dict)
    ]
    latest = _latest_record_by_task(experiment_records)
    reasons: list[str] = []
    task_results: list[dict[str, Any]] = []
    if not sample_tasks:
        reasons.append("no_sample_tasks")
    for task in sample_tasks:
        task_id = str(task.get("task_id") or "").strip()
        record = latest.get(task_id)
        task_reasons: list[str] = []
        if not record:
            task_reasons.append("missing_sample_record")
        else:
            status = str(record.get("status") or "").strip()
            if status != "completed":
                task_reasons.append(f"sample_status:{status or 'missing'}")
            budget_status = str(record.get("budget_status") or "").strip()
            if budget_status != "within_budget":
                task_reasons.append(f"sample_budget:{budget_status or 'missing'}")
            if not _budget_audit_passed(record):
                task_reasons.append("missing_budget_audit")
            if not _acceptance_results_passed(record):
                task_reasons.append("acceptance_results_not_passed")
            result_summary = record.get("result_summary")
            if not isinstance(result_summary, dict) or not result_summary:
                task_reasons.append("missing_result_summary")
        reasons.extend(task_reasons)
        task_results.append(
            {
                "task_id": task_id,
                "passed": not task_reasons,
                "reasons": task_reasons,
                "record_id": (
                    record.get("record_id") if isinstance(record, dict) else None
                ),
            }
        )
    passed = bool(sample_tasks) and not reasons
    updated = dict(sample_gate)
    updated["evaluated_at"] = _now_iso()
    updated["status"] = "passed" if passed else "blocked"
    updated["full_generation_allowed"] = passed
    updated["result"] = {
        "passed": passed,
        "reasons": reasons,
        "task_results": task_results,
    }
    return updated


def _save_sample_gate_artifact(
    project_root: str | Path,
    sample_gate: dict[str, Any],
    *,
    producer: str,
    depends_on: list[str],
) -> str:
    status = "ready" if sample_gate.get("full_generation_allowed") else "blocked"
    output_path = save_json_artifact(
        Path(project_root).expanduser().resolve() / "sample_gate.json",
        sample_gate,
    )
    update_pipeline_artifact(
        project_root,
        "sample_gate",
        status=status,
        producer=producer,
        depends_on=depends_on,
        recovery_hint=(
            None
            if status == "ready"
            else "Run and verify the sample task before full generation."
        ),
        notes="Full generation is allowed only when sample_gate.full_generation_allowed is true.",
    )
    return output_path


def save_sample_gate_plan(
    project_root: str | Path,
    *,
    research_plan: dict[str, Any] | None = None,
    sample_task_count: int = DEFAULT_SAMPLE_TASK_COUNT,
    producer: str = "sample_gate",
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    plan = research_plan
    if not isinstance(plan, dict):
        plan = load_contract_artifact(root, "research_plan", default={}) or {}
    sample_gate = build_sample_gate_plan(plan, sample_task_count=sample_task_count)
    _save_sample_gate_artifact(
        root,
        sample_gate,
        producer=producer,
        depends_on=["research_plan"],
    )
    return sample_gate


def evaluate_and_save_sample_gate(
    project_root: str | Path,
    *,
    sample_gate: dict[str, Any] | None = None,
    producer: str = "sample_gate",
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    gate = sample_gate
    if not isinstance(gate, dict):
        gate = load_contract_artifact(root, "sample_gate", default={}) or {}
    try:
        records = load_verified_experiment_records(root)
    except ResearchIntegrityError as exc:
        # An execution gate must fail closed.  In particular, do not reuse an
        # older valid row when a later row is malformed or when either integrity
        # sidecar is absent/tampered.
        evaluated = evaluate_sample_gate(gate, experiment_records=[])
        evaluated["status"] = "blocked"
        evaluated["full_generation_allowed"] = False
        evaluated["result"] = {
            "passed": False,
            "reasons": ["experiment_registry_integrity_failed"],
            "task_results": [
                {
                    **item,
                    "passed": False,
                    "reasons": ["experiment_registry_integrity_failed"],
                    "record_id": None,
                }
                for item in evaluated.get("result", {}).get("task_results", [])
                if isinstance(item, dict)
            ],
            "registry_integrity": {
                "ok": False,
                "detail": str(exc),
            },
        }
    else:
        evaluated = evaluate_sample_gate(gate, experiment_records=records)
    _save_sample_gate_artifact(
        root,
        evaluated,
        producer=producer,
        depends_on=["sample_gate", "experiment_registry"],
    )
    return evaluated


def assert_sample_gate_allows_full_generation(sample_gate: dict[str, Any]) -> None:
    if sample_gate.get("full_generation_allowed") is True:
        return
    reasons = sample_gate.get("result", {}).get("reasons", [])
    raise SampleGateBlocked(
        "Sample gate blocked full generation: "
        + ", ".join(str(item) for item in reasons)
    )
