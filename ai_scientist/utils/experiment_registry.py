from __future__ import annotations

"""Experiment registry helpers inspired by explicit experiment managers."""

import json
import hashlib
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - platform-specific import
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from ai_scientist.utils.atomic_io import atomic_write_bytes
from ai_scientist.utils.decision_log import record_decision
from ai_scientist.utils.evidence_snapshot import (
    REGISTRY_HISTORY_FILENAME,
    REGISTRY_INTEGRITY_FILENAME,
    build_registry_integrity,
    canonical_hash,
)
from ai_scientist.utils.research_integrity import ResearchIntegrityError
from ai_scientist.utils.safe_files import BoundedFileError, read_bounded_regular_file
from ai_scientist.utils.pipeline_contracts import (
    artifact_path,
    load_jsonl_artifact,
    update_pipeline_artifact,
)
from ai_scientist.utils.privacy import (
    REDACTED_PATH,
    portable_path,
    redact_sensitive_text,
)

_REGISTRY_LOCKS: dict[str, threading.RLock] = {}
_REGISTRY_LOCKS_GUARD = threading.Lock()
_MAX_REGISTRY_BYTES = 64 * 1024 * 1024
_MAX_INTEGRITY_BYTES = 32 * 1024 * 1024
_MAX_HISTORY_BYTES = 64 * 1024 * 1024


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
    evidence_role: str | None = None,
    paired_control_task_id: str | None = None,
    intervention_variant: str | None = None,
    stress_condition: str | None = None,
    transformation_manifest: dict[str, Any] | None = None,
    study_phase: str = "exploratory",
    preregistration_id: str | None = None,
    protocol_fidelity_hash: str | None = None,
    adaptive_state_hash: str | None = None,
    research_state_hash: str | None = None,
    post_freeze_adaptation: bool | None = None,
    dataset_split_hash: str | None = None,
    data_manifest_hash: str | None = None,
    data_snapshot_id: str | None = None,
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
    resolved_config = dict(config or {})
    resolved_transformation_manifest = dict(transformation_manifest or {})
    return {
        "record_id": str(record_id or f"{task_id}_{_now_iso()}").strip(),
        "task_id": task_id,
        "dataset": dataset,
        "metric": metric,
        "baseline_ref": baseline_ref,
        "config": resolved_config,
        "configuration_hash": canonical_hash(resolved_config),
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
        "evidence_role": str(evidence_role or "").strip().lower() or None,
        "paired_control_task_id": (str(paired_control_task_id or "").strip() or None),
        "intervention_variant": str(intervention_variant or "").strip() or None,
        "stress_condition": str(stress_condition or "").strip() or None,
        "transformation_manifest": resolved_transformation_manifest,
        "transformation_manifest_hash": (
            canonical_hash(resolved_transformation_manifest)
            if resolved_transformation_manifest
            else None
        ),
        "started_at": _now_iso(),
        "finished_at": None if status in {"planned", "running"} else _now_iso(),
        "study_phase": str(study_phase or "exploratory").strip().lower(),
        "preregistration_id": preregistration_id,
        "protocol_fidelity_hash": protocol_fidelity_hash,
        "adaptive_state_hash": adaptive_state_hash,
        "research_state_hash": research_state_hash,
        "post_freeze_adaptation": post_freeze_adaptation,
        "dataset_split_hash": dataset_split_hash,
        "data_manifest_hash": data_manifest_hash,
        "data_snapshot_id": data_snapshot_id,
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


def _registry_paths(project_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(project_root).expanduser().resolve()
    registry = Path(artifact_path(root, "experiment_registry"))
    return (
        registry,
        root / REGISTRY_INTEGRITY_FILENAME,
        root / REGISTRY_HISTORY_FILENAME,
    )


@contextmanager
def _registry_transaction_lock(project_root: str | Path):
    """Serialize cooperating writers without adding mutable files to the tree."""

    root = Path(project_root).expanduser().resolve()
    if fcntl is None:
        raise ResearchIntegrityError(
            "cross-process registry locking is unavailable on this platform"
        )
    lock_key = str(root)
    with _REGISTRY_LOCKS_GUARD:
        thread_lock = _REGISTRY_LOCKS.setdefault(lock_key, threading.RLock())
    with thread_lock:
        user_id = os.getuid() if hasattr(os, "getuid") else os.getpid()
        lock_root = Path(tempfile.gettempdir()) / f"xscientist-registry-locks-{user_id}"
        try:
            lock_root.mkdir(mode=0o700, parents=False, exist_ok=True)
            lock_metadata = lock_root.lstat()
        except OSError as exc:
            raise ResearchIntegrityError(
                "cannot establish the private registry lock directory"
            ) from exc
        if (
            not stat.S_ISDIR(lock_metadata.st_mode)
            or stat.S_ISLNK(lock_metadata.st_mode)
            or (hasattr(os, "getuid") and lock_metadata.st_uid != os.getuid())
            or stat.S_IMODE(lock_metadata.st_mode) & 0o077
        ):
            raise ResearchIntegrityError("registry lock directory is not private")
        digest = hashlib.sha256(lock_key.encode("utf-8")).hexdigest()
        lock_path = lock_root / f"{digest}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            lock_file_metadata = os.fstat(descriptor)
        except OSError as exc:
            raise ResearchIntegrityError(
                "cannot open the private registry lock"
            ) from exc
        if (
            not stat.S_ISREG(lock_file_metadata.st_mode)
            or lock_file_metadata.st_nlink != 1
            or (hasattr(os, "getuid") and lock_file_metadata.st_uid != os.getuid())
            or stat.S_IMODE(lock_file_metadata.st_mode) & 0o077
        ):
            os.close(descriptor)
            raise ResearchIntegrityError("registry lock file is not private")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _read_optional_bytes(path: Path) -> bytes | None:
    try:
        maximum = {
            REGISTRY_INTEGRITY_FILENAME: _MAX_INTEGRITY_BYTES,
            REGISTRY_HISTORY_FILENAME: _MAX_HISTORY_BYTES,
        }.get(path.name, _MAX_REGISTRY_BYTES)
        return read_bounded_regular_file(
            path,
            maximum=maximum,
            label="experiment_registry_artifact",
        )
    except BoundedFileError as exc:
        if exc.reason == "missing":
            return None
        raise ResearchIntegrityError(exc.code) from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def _encode_registry_rows(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    ).encode("utf-8")


def _same_registry_record(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """Treat builder-generated wall-clock fields as idempotent metadata only."""

    volatile = {"started_at", "finished_at"}
    return {key: value for key, value in existing.items() if key not in volatile} == {
        key: value for key, value in incoming.items() if key not in volatile
    }


def _strict_history_entries(
    raw: bytes,
) -> list[tuple[dict[str, Any], bytes]]:
    """Return strict rows together with their exact preceding byte prefix."""

    entries: list[tuple[dict[str, Any], bytes]] = []
    offset = 0
    for line_number, raw_line in enumerate(raw.splitlines(keepends=True), start=1):
        prefix = raw[:offset]
        offset += len(raw_line)
        line = raw_line.decode("utf-8")
        if not line.strip():
            continue
        payload = _strict_json_loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"registry history row {line_number} must be a JSON object"
            )
        entries.append((payload, prefix))
    return entries


def _strict_history_rows(raw: bytes) -> list[dict[str, Any]]:
    return [row for row, _prefix in _strict_history_entries(raw)]


def _raw_history_hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _history_core(
    integrity: dict[str, Any],
    *,
    sequence: int,
    previous_audit_hash: str,
) -> dict[str, Any]:
    return {
        "version": 2,
        "sequence": sequence,
        "previous_audit_hash": previous_audit_hash,
        "record_count": int(integrity["row_count"]),
        "records_hash": integrity["records_hash"],
        "raw_hash": integrity["raw_hash"],
        "chain_tip": integrity["chain_tip"],
    }


def _next_history_bytes(
    previous_raw: bytes,
    *,
    integrity: dict[str, Any],
) -> bytes:
    history = _strict_history_rows(previous_raw) if previous_raw else []
    previous_hash = str(history[-1].get("audit_hash") or "") if history else "GENESIS"
    prefix = previous_raw
    if prefix and not prefix.endswith(b"\n"):
        prefix += b"\n"
    has_unanchored_legacy = any(row.get("version") == 1 for row in history) and not any(
        row.get("version") == 3 for row in history
    )
    if has_unanchored_legacy:
        core = {
            **_history_core(
                integrity,
                sequence=len(history) + 1,
                previous_audit_hash=previous_hash,
            ),
            "version": 3,
            "legacy_history_raw_hash": _raw_history_hash(prefix),
            "legacy_history_row_count": len(history),
            "legacy_history_tip": previous_hash,
        }
    else:
        core = _history_core(
            integrity,
            sequence=len(history) + 1,
            previous_audit_hash=previous_hash,
        )
    row = {**core, "audit_hash": canonical_hash(core)}
    return prefix + (
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _restore_registry_transaction(
    originals: dict[Path, bytes | None],
    intended: dict[Path, bytes],
) -> None:
    """Rollback only when every path still has our expected transaction state."""

    current = {path: _read_optional_bytes(path) for path in originals}
    if any(
        current[path] not in {originals[path], intended[path]} for path in originals
    ):
        raise ResearchIntegrityError(
            "registry transaction failed after an external concurrent mutation; "
            "automatic rollback was refused"
        )
    for path, raw in originals.items():
        if raw is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(path, raw)


def _write_registry_transaction(
    project_root: str | Path,
    *,
    registry_raw: bytes,
    require_existing_valid: bool,
    expected_originals: dict[Path, bytes | None],
) -> None:
    registry_path, integrity_path, history_path = _registry_paths(project_root)
    paths = (registry_path, integrity_path, history_path)
    if set(expected_originals) != set(paths):
        raise ResearchIntegrityError(
            "experiment registry transaction received an invalid observed snapshot"
        )
    originals = {path: expected_originals[path] for path in paths}
    # The caller must merge against the same registry/integrity/history triple
    # that this transaction is about to replace. Re-reading the baseline here
    # would allow a stale A+B calculation to treat a later coherent A+C commit
    # as its original state and silently discard C.
    if any(_read_optional_bytes(path) != originals[path] for path in paths):
        raise ResearchIntegrityError(
            "experiment registry changed concurrently after its observed snapshot"
        )
    present = {path: raw is not None for path, raw in originals.items()}
    if any(present.values()) and not all(present.values()):
        raise ResearchIntegrityError(
            "experiment registry, integrity sidecar, and audit history must all "
            "exist before an append/save; refusing to rebuild missing evidence"
        )
    if require_existing_valid and all(present.values()):
        report = _check_experiment_registry_snapshot(
            project_root,
            snapshot=originals,
            allow_unanchored_legacy=True,
        )
        if not report.get("ok"):
            raise ResearchIntegrityError(
                "Refusing to append/rewrite a tampered experiment registry: "
                + ", ".join(report.get("errors") or [])
            )
    elif require_existing_valid and any(present.values()):
        raise ResearchIntegrityError("experiment registry evidence is incomplete")

    try:
        rows = _parse_registry_bytes(registry_raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResearchIntegrityError(
            "experiment registry contains a malformed or non-object JSONL row"
        ) from exc
    record_ids = [str(row.get("record_id") or "").strip() for row in rows]
    if not rows or any(not record_id for record_id in record_ids):
        raise ResearchIntegrityError(
            "experiment registry requires at least one non-empty record_id"
        )
    if len(record_ids) != len(set(record_ids)):
        raise ResearchIntegrityError(
            "experiment registry record_id values must be unique"
        )
    integrity = build_registry_integrity(rows, raw_bytes=registry_raw)
    integrity_raw = json.dumps(
        integrity, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    history_raw = _next_history_bytes(
        originals[history_path] or b"",
        integrity=integrity,
    )
    intended = {
        registry_path: registry_raw,
        integrity_path: integrity_raw,
        history_path: history_raw,
    }
    # Compare-and-swap immediately before the first replacement. The lock
    # coordinates compliant writers; this check also catches non-compliant ones.
    if any(_read_optional_bytes(path) != originals[path] for path in paths):
        raise ResearchIntegrityError(
            "experiment registry changed concurrently before transaction commit"
        )
    try:
        atomic_write_bytes(registry_path, registry_raw)
        if _read_optional_bytes(registry_path) != registry_raw:
            raise ResearchIntegrityError("experiment registry CAS verification failed")
        atomic_write_bytes(integrity_path, integrity_raw)
        if (
            _read_optional_bytes(registry_path) != registry_raw
            or _read_optional_bytes(integrity_path) != integrity_raw
        ):
            raise ResearchIntegrityError(
                "experiment registry changed during integrity commit"
            )
        atomic_write_bytes(history_path, history_raw)
        if any(_read_optional_bytes(path) != intended[path] for path in paths):
            raise ResearchIntegrityError(
                "experiment registry transaction failed final CAS verification"
            )
    except BaseException:
        _restore_registry_transaction(originals, intended)
        raise


def append_experiment_record(project_root: str | Path, record: dict[str, Any]) -> str:
    output_path, integrity_path, history_path = _registry_paths(project_root)
    with _registry_transaction_lock(project_root):
        paths = (output_path, integrity_path, history_path)
        observed = {path: _read_optional_bytes(path) for path in paths}
        current = observed[output_path]
        if current is None:
            new_raw = _encode_registry_rows([record])
            require_existing_valid = False
        else:
            # Parsing is deliberately strict: a non-object or malformed row is
            # evidence of corruption, never an entry to silently discard.
            try:
                _parse_registry_bytes(current)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ResearchIntegrityError(
                    "experiment registry contains a malformed or non-object JSONL row"
                ) from exc
            prefix = current + (b"" if current.endswith(b"\n") else b"\n")
            new_raw = prefix + _encode_registry_rows([record])
            require_existing_valid = True
        _write_registry_transaction(
            project_root,
            registry_raw=new_raw,
            require_existing_valid=require_existing_valid,
            expected_originals=observed,
        )
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
        "by_evidence_role": {},
        "paired_control_count": 0,
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
        evidence_role = str(record.get("evidence_role") or "unassigned")
        summary["by_evidence_role"][evidence_role] = (
            summary["by_evidence_role"].get(evidence_role, 0) + 1
        )
        if str(record.get("paired_control_task_id") or "").strip():
            summary["paired_control_count"] += 1
    return summary


def _save_experiment_registry_rows(
    project_root: str | Path, rows: list[dict[str, Any]]
) -> tuple[str, bool]:
    """Merge rows transactionally and report whether durable state changed."""

    output_path, integrity_path, history_path = _registry_paths(project_root)
    changed = False
    with _registry_transaction_lock(project_root):
        paths = (output_path, integrity_path, history_path)
        observed = {path: _read_optional_bytes(path) for path in paths}
        current_raw = observed[output_path]
        if current_raw is None:
            require_existing_valid = False
            desired_raw = _encode_registry_rows(rows)
            should_write = True
        else:
            try:
                existing_rows = _parse_registry_bytes(current_raw)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ResearchIntegrityError(
                    "experiment registry contains a malformed or non-object JSONL row"
                ) from exc
            report = _check_experiment_registry_snapshot(
                project_root,
                snapshot=observed,
                allow_unanchored_legacy=True,
            )
            if not report.get("ok"):
                raise ResearchIntegrityError(
                    "Refusing to rewrite a tampered experiment registry: "
                    + ", ".join(report.get("errors") or [])
                )
            existing_by_id = {
                str(row.get("record_id") or ""): row for row in existing_rows
            }
            incoming_ids: set[str] = set()
            appended: list[dict[str, Any]] = []
            for row in rows:
                record_id = str(row.get("record_id") or "").strip()
                if not record_id or record_id in incoming_ids:
                    raise ResearchIntegrityError(
                        "incoming experiment registry rows require unique record_id "
                        "values"
                    )
                incoming_ids.add(record_id)
                existing = existing_by_id.get(record_id)
                if existing is not None:
                    if not _same_registry_record(existing, row):
                        raise ResearchIntegrityError(
                            f"experiment registry record_id is immutable: {record_id}"
                        )
                    continue
                appended.append(row)
            prefix = current_raw + (b"" if current_raw.endswith(b"\n") else b"\n")
            desired_raw = prefix + _encode_registry_rows(appended)
            require_existing_valid = True
            # An unchanged legacy registry still needs a one-time migration row
            # before it can be trusted by execution or publication gates.
            should_write = bool(appended) or bool(
                report.get("legacy_history_unanchored")
            )
        if should_write:
            _write_registry_transaction(
                project_root,
                registry_raw=desired_raw,
                require_existing_valid=require_existing_valid,
                expected_originals=observed,
            )
            changed = True
    if changed:
        update_pipeline_artifact(
            project_root,
            "experiment_registry",
            status="ready",
            producer="experiment_registry",
            depends_on=["research_plan"],
        )
        from ai_scientist.utils.stage_standards import save_stage_standards

        save_stage_standards(project_root)
    return str(output_path), changed


def save_experiment_registry(
    project_root: str | Path, rows: list[dict[str, Any]]
) -> str:
    output_path, _ = _save_experiment_registry_rows(project_root, rows)
    return output_path


def normalize_terminal_experiment_status(raw_status: Any) -> str:
    normalized = str(raw_status or "failed").strip().lower()
    if normalized in {"timeout", "timed_out"}:
        return "timed_out"
    if normalized in {"cancelled", "canceled", "interrupted"}:
        return "cancelled"
    return "failed"


def _terminal_failure_receipt_hash(
    root: Path, experiment_result: dict[str, Any]
) -> str:
    """Identify one runtime receipt without persisting host paths or wall-clock time."""

    error = experiment_result.get("failure_error") or experiment_result.get(
        "budget_error"
    )
    error = error if isinstance(error, dict) else {}
    explicit_identity = {
        key: str(value).strip()
        for key, value in {
            "receipt_hash": experiment_result.get("receipt_hash"),
            "run_id": experiment_result.get("run_id"),
            "attempt_id": experiment_result.get("attempt_id"),
            "failure_ref": error.get("failure_ref"),
        }.items()
        if str(value or "").strip()
    }
    artifact_hashes: dict[str, str] = {}
    for field in (
        "run_status_path",
        "initialization_status_path",
        "checkpoint_path",
        "manager_state_path",
    ):
        raw_path = str(experiment_result.get(field) or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
            encoded = read_bounded_regular_file(
                resolved,
                maximum=16 * 1024 * 1024,
                label="terminal_experiment_receipt",
            )
        except (BoundedFileError, OSError, ValueError):
            continue
        artifact_hashes[field] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    identity: dict[str, Any] = {
        "status": str(experiment_result.get("status") or "failed").strip().lower(),
        "explicit_identity": explicit_identity,
        "artifact_hashes": artifact_hashes,
    }
    if not explicit_identity and not artifact_hashes:
        identity["fallback"] = {
            "error_type": str(error.get("type") or "").strip(),
            "error_code": str(error.get("error_code") or "").strip(),
            "error_message": redact_sensitive_text(str(error.get("message") or "")),
            "initialization_phase": experiment_result.get("initialization_phase"),
            "resumable": bool(experiment_result.get("resumable")),
            "lock_owner": experiment_result.get("lock_owner") or {},
        }
    return canonical_hash(identity)


def record_terminal_experiment_failure(
    project_root: str | Path,
    *,
    research_plan: dict[str, Any],
    experiment_result: dict[str, Any],
    producer: str,
) -> dict[str, Any]:
    """Persist a terminal runtime failure before manuscript work is skipped."""

    root = Path(project_root).expanduser().resolve()
    raw_status = str(experiment_result.get("status") or "failed").strip().lower()
    status = normalize_terminal_experiment_status(raw_status)
    receipt_hash = _terminal_failure_receipt_hash(root, experiment_result)
    receipt_id = receipt_hash.removeprefix("sha256:")[:20]
    raw_error = (
        (experiment_result.get("failure_error") or {}).get("message")
        or (experiment_result.get("budget_error") or {}).get("message")
        or experiment_result.get("error")
        or f"experiment runtime ended with status {raw_status}"
    )
    error_message = redact_sensitive_text(str(raw_error))
    artifact_fields: dict[str, str] = {}
    for field in (
        "checkpoint_path",
        "run_status_path",
        "manager_state_path",
        "initialization_status_path",
        "lock_path",
    ):
        raw_path = str(experiment_result.get(field) or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        rendered = portable_path(candidate, base=root)
        if rendered != REDACTED_PATH:
            artifact_fields[field] = rendered
    tasks = [
        task for task in research_plan.get("tasks") or [] if isinstance(task, dict)
    ] or [{"task_id": "experiment"}]
    rows = [
        build_experiment_record(
            record_id=(
                f"{str(task.get('task_id') or f'task_{index}')}:runtime:"
                f"{raw_status}:{receipt_id}"
            ),
            task_id=str(task.get("task_id") or f"task_{index}"),
            dataset=str(task.get("dataset") or "dataset_to_be_selected"),
            metric=str(task.get("metric") or "primary_task_metric"),
            baseline_ref=str(task.get("baseline") or "strong_existing_baseline"),
            config={
                "goal": task.get("goal"),
                "priority": task.get("priority"),
            },
            status=status,
            result_summary={
                "runtime_status": raw_status,
                "terminal_receipt_hash": receipt_hash,
                "resumable": bool(experiment_result.get("resumable")),
                "initialization_phase": experiment_result.get("initialization_phase"),
            },
            artifacts=artifact_fields,
            error_type=raw_status,
            error_message=error_message,
            entered_storyline=False,
            budget=task.get("budget"),
            budget_status=(
                "budget_exhausted" if raw_status == "budget_exhausted" else None
            ),
            workflow_mode=research_plan.get("workflow_mode"),
            policy_name=(research_plan.get("execution_policy") or {}).get(
                "policy_name"
            ),
            acceptance_checks=task.get("acceptance_checks"),
            acceptance_results=[
                {
                    "check": str(check),
                    "passed": False,
                    "source": "terminal_runtime_failure",
                }
                for check in task.get("acceptance_checks") or []
            ],
            budget_audit={
                "audited": True,
                "within_budget": raw_status != "budget_exhausted",
                "source": "terminal_runtime_failure",
            },
            evidence_role=task.get("evidence_role"),
            paired_control_task_id=task.get("paired_control_task_id"),
            intervention_variant=task.get("intervention_variant"),
            stress_condition=task.get("stress_condition"),
        )
        for index, task in enumerate(tasks)
    ]
    _, changed = _save_experiment_registry_rows(root, rows)
    replayed = not changed
    if not replayed:
        record_decision(
            root,
            category="experiment_terminal_outcome",
            selected=status,
            options_considered=[
                {"option": status, "selected": True},
                {
                    "option": "continue_to_writeup",
                    "rejected_because": (
                        "the experiment did not produce a completed terminal outcome"
                    ),
                },
            ],
            producer=producer,
            metadata={
                "runtime_status": raw_status,
                "terminal_receipt_hash": receipt_hash,
                "resumable": bool(experiment_result.get("resumable")),
                "registry_rows": len(rows),
            },
        )
    return {
        "status": status,
        "runtime_status": raw_status,
        "terminal_receipt_hash": receipt_hash,
        "replayed": replayed,
        "error": error_message,
        "registry_rows": len(rows),
    }


def _load_registry_rows_and_bytes(
    project_root: str | Path,
) -> tuple[list[dict[str, Any]], bytes]:
    output_path = Path(artifact_path(project_root, "experiment_registry"))
    raw = _read_optional_bytes(output_path)
    if raw is None:
        return [], b""
    return _parse_registry_bytes(raw), raw


def _parse_registry_bytes(raw: bytes) -> list[dict[str, Any]]:
    text = raw.decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = _strict_json_loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                f"experiment registry row {line_number} must be a JSON object"
            )
        rows.append(payload)
    return rows


def _write_registry_integrity(project_root: str | Path) -> None:
    """Validate and append one integrity snapshot under the registry lock."""

    output_path, integrity_path, history_path = _registry_paths(project_root)
    with _registry_transaction_lock(project_root):
        paths = (output_path, integrity_path, history_path)
        observed = {path: _read_optional_bytes(path) for path in paths}
        raw = observed[output_path]
        if raw is None:
            raise ResearchIntegrityError("experiment registry is missing")
        _write_registry_transaction(
            project_root,
            registry_raw=raw,
            require_existing_valid=True,
            expected_originals=observed,
        )


def _check_experiment_registry_snapshot(
    project_root: str | Path,
    *,
    snapshot: dict[Path, bytes | None] | None = None,
    allow_unanchored_legacy: bool = False,
) -> dict[str, Any]:
    """Verify one caller-observed registry triple without silently repairing it."""

    root = Path(project_root).expanduser().resolve()
    output_path, integrity_path, history_path = _registry_paths(root)
    existence_errors: list[str] = []
    try:
        paths = (output_path, integrity_path, history_path)
        if snapshot is None:
            observed = {path: _read_optional_bytes(path) for path in paths}
        else:
            if set(snapshot) != set(paths):
                raise ResearchIntegrityError(
                    "experiment registry integrity check received an invalid snapshot"
                )
            observed = {path: snapshot[path] for path in paths}
        raw = observed[output_path]
        integrity_raw = observed[integrity_path]
        history_raw = observed[history_path]
        if raw is None:
            existence_errors.append("registry_missing")
            raw = b""
        if integrity_raw is None:
            existence_errors.append("registry_integrity_manifest_missing")
            integrity_raw = b""
        if history_raw is None:
            existence_errors.append("registry_history_missing")
            history_raw = b""
        rows = _parse_registry_bytes(raw)
    except (
        ResearchIntegrityError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "ok": False,
            "errors": sorted(set([*existence_errors, "registry_read_failed"])),
            "detail": str(exc),
            "manifest_path": str(integrity_path),
            "history_path": str(history_path),
        }
    current = build_registry_integrity(rows, raw_bytes=raw)
    errors = list(existence_errors)
    expected: dict[str, Any] = {}
    try:
        parsed_integrity = _strict_json_loads(integrity_raw.decode("utf-8"))
        if not isinstance(parsed_integrity, dict):
            raise ValueError("registry integrity sidecar must be a JSON object")
        allowed_integrity_keys = set(current)
        if set(parsed_integrity) != allowed_integrity_keys:
            errors.append("registry_integrity_fields_invalid")
        expected = parsed_integrity
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append("registry_integrity_read_failed")
        integrity_detail = str(exc)
    for key in current:
        if expected.get(key) != current.get(key):
            errors.append(f"registry_{key}_mismatch")
    record_ids = current["record_ids"]
    if not record_ids or any(not value for value in record_ids):
        errors.append("registry_record_id_missing")
    if len(record_ids) != len(set(record_ids)):
        errors.append("registry_record_id_duplicate")
    report: dict[str, Any] = {
        "ok": False,
        "errors": errors,
        "expected": expected,
        "current": current,
        "manifest_path": str(integrity_path),
    }
    if "integrity_detail" in locals():
        report["integrity_detail"] = integrity_detail
    try:
        history_entries = _strict_history_entries(history_raw)
        history = [row for row, _prefix in history_entries]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        history = []
        report["errors"].append("registry_history_read_failed")
        report["history_detail"] = str(exc)
    if not history:
        report["errors"].append("registry_history_missing")
    else:
        previous_audit_hash = "GENESIS"
        previous_record_count = -1
        saw_modern = False
        saw_legacy = False
        legacy_anchor_valid = False
        for index, (row, raw_prefix) in enumerate(history_entries, start=1):
            version = row.get("version")
            if version == 1:
                allowed_history_keys = {
                    "version",
                    "record_count",
                    "records_hash",
                    "raw_hash",
                    "chain_tip",
                    "audit_hash",
                }
            elif version == 3:
                allowed_history_keys = {
                    "version",
                    "sequence",
                    "previous_audit_hash",
                    "record_count",
                    "records_hash",
                    "raw_hash",
                    "chain_tip",
                    "legacy_history_raw_hash",
                    "legacy_history_row_count",
                    "legacy_history_tip",
                    "audit_hash",
                }
            else:
                allowed_history_keys = {
                    "version",
                    "sequence",
                    "previous_audit_hash",
                    "record_count",
                    "records_hash",
                    "raw_hash",
                    "chain_tip",
                    "audit_hash",
                }
            if set(row) != allowed_history_keys:
                report["errors"].append("registry_history_fields_invalid")
            record_count = row.get("record_count")
            if (
                isinstance(record_count, bool)
                or not isinstance(record_count, int)
                or record_count < 0
            ):
                report["errors"].append("registry_history_record_count_invalid")
                record_count = -1
            if record_count < previous_record_count:
                report["errors"].append("registry_history_record_count_regressed")
            previous_record_count = max(previous_record_count, record_count)
            if version == 1:
                saw_legacy = True
                if saw_modern:
                    report["errors"].append("registry_history_legacy_row_after_v2")
                core = {
                    "record_count": row.get("record_count"),
                    "records_hash": row.get("records_hash"),
                    "raw_hash": row.get("raw_hash"),
                    "chain_tip": row.get("chain_tip"),
                }
            elif version == 2:
                saw_modern = True
                if row.get("sequence") != index:
                    report["errors"].append("registry_history_sequence_mismatch")
                if row.get("previous_audit_hash") != previous_audit_hash:
                    report["errors"].append("registry_history_previous_hash_mismatch")
                core = {
                    "version": 2,
                    "sequence": row.get("sequence"),
                    "previous_audit_hash": row.get("previous_audit_hash"),
                    "record_count": row.get("record_count"),
                    "records_hash": row.get("records_hash"),
                    "raw_hash": row.get("raw_hash"),
                    "chain_tip": row.get("chain_tip"),
                }
            elif version == 3:
                saw_modern = True
                if legacy_anchor_valid:
                    report["errors"].append("registry_history_legacy_anchor_duplicate")
                if not saw_legacy:
                    report["errors"].append(
                        "registry_history_legacy_anchor_without_legacy"
                    )
                if row.get("sequence") != index:
                    report["errors"].append("registry_history_sequence_mismatch")
                if row.get("previous_audit_hash") != previous_audit_hash:
                    report["errors"].append("registry_history_previous_hash_mismatch")
                raw_hash_matches = row.get(
                    "legacy_history_raw_hash"
                ) == _raw_history_hash(raw_prefix)
                row_count_matches = row.get("legacy_history_row_count") == index - 1
                tip_matches = row.get("legacy_history_tip") == previous_audit_hash
                if not raw_hash_matches:
                    report["errors"].append("registry_history_legacy_raw_hash_mismatch")
                if not row_count_matches:
                    report["errors"].append(
                        "registry_history_legacy_row_count_mismatch"
                    )
                if not tip_matches:
                    report["errors"].append("registry_history_legacy_tip_mismatch")
                legacy_anchor_valid = bool(
                    saw_legacy
                    and not legacy_anchor_valid
                    and raw_hash_matches
                    and row_count_matches
                    and tip_matches
                )
                core = {
                    "version": 3,
                    "sequence": row.get("sequence"),
                    "previous_audit_hash": row.get("previous_audit_hash"),
                    "record_count": row.get("record_count"),
                    "records_hash": row.get("records_hash"),
                    "raw_hash": row.get("raw_hash"),
                    "chain_tip": row.get("chain_tip"),
                    "legacy_history_raw_hash": row.get("legacy_history_raw_hash"),
                    "legacy_history_row_count": row.get("legacy_history_row_count"),
                    "legacy_history_tip": row.get("legacy_history_tip"),
                }
            else:
                report["errors"].append("registry_history_version_invalid")
                core = {}
            expected_audit_hash = canonical_hash(core) if core else ""
            if row.get("audit_hash") != expected_audit_hash:
                report["errors"].append("registry_history_audit_hash_mismatch")
            previous_audit_hash = str(row.get("audit_hash") or "")

        report["legacy_history_unanchored"] = bool(
            saw_legacy and not legacy_anchor_valid
        )
        if report["legacy_history_unanchored"] and not allow_unanchored_legacy:
            report["errors"].append("legacy_history_unanchored")

        latest = history[-1]
        latest_fields = {
            "record_count": report["current"]["row_count"],
            "records_hash": report["current"]["records_hash"],
            "raw_hash": report["current"]["raw_hash"],
            "chain_tip": report["current"]["chain_tip"],
        }
        if any(latest.get(key) != value for key, value in latest_fields.items()):
            report["errors"].append("registry_history_tip_mismatch")
    report["errors"] = sorted(set(report["errors"]))
    report["ok"] = not report["errors"]
    report["history_path"] = str(history_path)
    return report


def check_experiment_registry_integrity(project_root: str | Path) -> dict[str, Any]:
    """Verify the on-disk registry against its sidecar and audit history."""

    return _check_experiment_registry_snapshot(project_root)


def load_verified_experiment_records(
    project_root: str | Path,
) -> list[dict[str, Any]]:
    """Load one bounded registry snapshot only after all evidence verifies.

    The registry, integrity sidecar, and append-only history are read once and
    verified as the same snapshot.  Callers making execution or publication
    decisions must not use the permissive legacy JSONL loader, which can skip a
    malformed row and accidentally authorize work from stale evidence.
    """

    paths = _registry_paths(project_root)
    observed = {path: _read_optional_bytes(path) for path in paths}
    report = _check_experiment_registry_snapshot(
        project_root,
        snapshot=observed,
    )
    if not report.get("ok"):
        errors = ", ".join(str(item) for item in report.get("errors") or [])
        raise ResearchIntegrityError(
            "experiment registry integrity verification failed"
            + (f": {errors}" if errors else "")
        )
    raw = observed[paths[0]]
    if raw is None:  # Defensive: a verified snapshot must contain the registry.
        raise ResearchIntegrityError("experiment registry is missing")
    try:
        return _parse_registry_bytes(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ResearchIntegrityError(
            "experiment registry strict parsing failed after verification"
        ) from exc
