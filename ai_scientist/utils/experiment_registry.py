from __future__ import annotations

"""Experiment registry helpers inspired by explicit experiment managers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.atomic_io import atomic_write_json
from ai_scientist.utils.evidence_snapshot import (
    REGISTRY_HISTORY_FILENAME,
    REGISTRY_INTEGRITY_FILENAME,
    build_registry_integrity,
    canonical_hash,
    verify_registry_integrity,
)
from ai_scientist.utils.research_integrity import ResearchIntegrityError
from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    artifact_path,
    load_jsonl_artifact,
    save_jsonl_artifact,
    update_pipeline_artifact,
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def build_experiment_record(
    *,
    task_id: str,
    record_id: str | None = None,
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
    acceptance_results: list[dict[str, Any]] | None = None,
    budget_audit: dict[str, Any] | None = None,
    budget_status: str | None = None,
    study_phase: str = "exploratory",
    preregistration_id: str | None = None,
    protocol_fidelity_hash: str | None = None,
    dataset_split_hash: str | None = None,
    metric_provenance: str | None = None,
    evaluator_input_hash: str | None = None,
    evaluator_result_hash: str | None = None,
    holdout_access: str | None = None,
    producer_id: str | None = None,
    independent_reproduction: bool = False,
    replicates_record_id: str | None = None,
    verifier_id: str | None = None,
    clean_room: bool = False,
    verification_recomputed: bool = False,
    verification_metric_hash: str | None = None,
    verification_output_hash: str | None = None,
    verification_command: str | None = None,
    verification_method: str | None = None,
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
        "record_id": str(record_id or f"{task_id}_{_now_iso()}").strip(),
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
        "acceptance_results": list(acceptance_results or []),
        "budget_audit": dict(budget_audit or {}),
        "started_at": _now_iso(),
        "finished_at": None if status in {"planned", "running"} else _now_iso(),
        "study_phase": str(study_phase or "exploratory").strip().lower(),
        "preregistration_id": preregistration_id,
        "protocol_fidelity_hash": protocol_fidelity_hash,
        "dataset_split_hash": dataset_split_hash,
        "metric_provenance": metric_provenance,
        "evaluator_input_hash": evaluator_input_hash,
        "evaluator_result_hash": evaluator_result_hash,
        "holdout_access": holdout_access,
        "producer_id": producer_id,
        "independent_reproduction": bool(independent_reproduction),
        "replicates_record_id": replicates_record_id,
        "verifier_id": verifier_id,
        "clean_room": bool(clean_room),
        "verification_recomputed": bool(verification_recomputed),
        "verification_metric_hash": verification_metric_hash,
        "verification_output_hash": verification_output_hash,
        "verification_command": verification_command,
        "verification_method": verification_method,
    }


def append_experiment_record(project_root: str | Path, record: dict[str, Any]) -> str:
    output_path = artifact_path(project_root, "experiment_registry")
    append_jsonl_artifact(output_path, record)
    _write_registry_integrity(project_root)
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


def save_experiment_registry(
    project_root: str | Path, rows: list[dict[str, Any]]
) -> str:
    output_path = Path(artifact_path(project_root, "experiment_registry"))
    integrity_path = output_path.parent / REGISTRY_INTEGRITY_FILENAME
    if output_path.exists() and integrity_path.exists():
        existing_rows, existing_raw = _load_registry_rows_and_bytes(output_path.parent)
        current_integrity = verify_registry_integrity(
            output_path.parent, existing_rows, raw_bytes=existing_raw
        )
        if not current_integrity.get("ok"):
            raise ResearchIntegrityError(
                "Refusing to rewrite a tampered experiment registry: "
                + ", ".join(current_integrity.get("errors") or [])
            )
        if existing_rows != rows:
            if len(rows) < len(existing_rows) or rows[: len(existing_rows)] != existing_rows:
                raise ResearchIntegrityError(
                    "Experiment registry is append-only; deletion, reordering, and replacement are blocked."
                )
            for row in rows[len(existing_rows) :]:
                append_jsonl_artifact(output_path, row)
        _write_registry_integrity(project_root)
    else:
        save_jsonl_artifact(output_path, rows)
        _write_registry_integrity(project_root)
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


def _load_registry_rows_and_bytes(project_root: str | Path) -> tuple[list[dict[str, Any]], bytes]:
    output_path = Path(artifact_path(project_root, "experiment_registry"))
    try:
        raw = output_path.read_bytes()
    except OSError:
        return [], b""
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows, raw


def _write_registry_integrity(project_root: str | Path) -> None:
    """Persist the registry's ordered chain and append an audit history row."""

    root = Path(project_root).expanduser().resolve()
    rows, raw = _load_registry_rows_and_bytes(root)
    integrity = build_registry_integrity(rows, raw_bytes=raw)
    atomic_write_json(root / REGISTRY_INTEGRITY_FILENAME, integrity, indent=2)
    history_path = root / REGISTRY_HISTORY_FILENAME
    history_row = {
        "version": 1,
        "record_count": len(rows),
        "records_hash": integrity["records_hash"],
        "raw_hash": integrity["raw_hash"],
        "chain_tip": integrity["chain_tip"],
        "audit_hash": canonical_hash(
            {
                "record_count": len(rows),
                "records_hash": integrity["records_hash"],
                "raw_hash": integrity["raw_hash"],
                "chain_tip": integrity["chain_tip"],
            }
        ),
    }
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_row, ensure_ascii=False) + "\n")


def check_experiment_registry_integrity(project_root: str | Path) -> dict[str, Any]:
    """Verify the registry against its sidecar without silently repairing it."""

    root = Path(project_root).expanduser().resolve()
    output_path = Path(artifact_path(root, "experiment_registry"))
    try:
        rows, raw = _load_registry_rows_and_bytes(root)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "errors": ["registry_read_failed"],
            "detail": str(exc),
            "manifest_path": str(root / REGISTRY_INTEGRITY_FILENAME),
        }
    report = verify_registry_integrity(root, rows, raw_bytes=raw)
    history_path = root / REGISTRY_HISTORY_FILENAME
    try:
        history = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        history = []
    if not history:
        report["errors"].append("registry_history_missing")
    elif history[-1].get("audit_hash") != canonical_hash(
        {
            "record_count": report["current"]["row_count"],
            "records_hash": report["current"]["records_hash"],
            "raw_hash": report["current"]["raw_hash"],
            "chain_tip": report["current"]["chain_tip"],
        }
    ):
        report["errors"].append("registry_history_tip_mismatch")
    report["errors"] = sorted(set(report["errors"]))
    report["ok"] = not report["errors"]
    report["history_path"] = str(history_path)
    return report
