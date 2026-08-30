"""Bind publication registry evidence to the native structured Research VCS trail."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from ai_scientist.protocol import content_hash
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.principal_identity import canonical_principal
from ai_scientist.utils.safe_files import BoundedFileError, read_bounded_regular_file

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
TRAJECTORY_BINDING_PROTOCOL = "xscientist.trajectory-binding.v1"
ATTEMPT_DISPOSITION_PROTOCOL = "xscientist.attempt-disposition.v1"
ATTEMPT_DISPOSITIONS = frozenset(
    {
        "terminal_negative",
        "technical_failure_retried",
        "approved_deviation",
        "excluded_with_reason",
    }
)
_UNSUCCESSFUL_STATES = frozenset(
    {"failed", "error", "timed_out", "timeout", "cancelled", "canceled"}
)
_MAX_NEGATIVE_RESULT_ARTIFACT_BYTES = 64 * 1024 * 1024


def registry_row_hash(row: Mapping[str, Any]) -> str:
    """Return the content identity of one immutable registry/disposition row."""

    return canonical_content_hash(dict(row))


def _relation_targets(
    research_object: Mapping[str, Any], relation_type: str, role: str | None = None
) -> set[str]:
    return {
        str(item.get("target") or "").strip()
        for item in research_object.get("relations") or []
        if isinstance(item, Mapping)
        and item.get("type") == relation_type
        and (role is None or item.get("role") == role)
        and str(item.get("target") or "").strip()
    }


def _content_hashes(value: Any) -> set[str]:
    hashes: set[str] = set()
    stack = [value]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            if _HASH_RE.fullmatch(current):
                hashes.add(current)
            continue
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        marker = id(current)
        if marker in visited:
            continue
        visited.add(marker)
        if isinstance(current, Mapping):
            stack.extend(current.values())
        else:
            stack.extend(current)
    return hashes


def _normalized_state(value: Any) -> str:
    return {
        "success": "completed",
        "completed": "completed",
        "verified": "completed",
        "failed": "failed",
        "error": "failed",
        "timeout": "timed_out",
        "timed_out": "timed_out",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "running": "running",
        "planned": "running",
    }.get(str(value or "").strip().lower(), "invalid")


def build_terminal_negative_artifact_receipt(
    repository: Any,
    artifact_selector: str | Path,
) -> dict[str, Any]:
    """Hash one durable in-repository result without trusting caller metadata."""

    repository_root = Path(repository.path).expanduser().resolve()
    selected = Path(artifact_selector).expanduser()
    selected = selected if selected.is_absolute() else repository_root / selected
    try:
        target = selected.resolve(strict=True)
        relative = target.relative_to(repository_root).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError(
            "terminal negative artifact must be a file inside the research repository"
        ) from exc
    try:
        encoded = read_bounded_regular_file(
            target,
            maximum=_MAX_NEGATIVE_RESULT_ARTIFACT_BYTES,
            label="terminal_negative_artifact",
        )
    except BoundedFileError as exc:
        raise ValueError("terminal negative artifact is missing or unsafe") from exc
    return {
        "path": relative,
        "content_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def terminal_negative_contract_errors(
    repository: Any,
    disposition_object: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    registry_row: Mapping[str, Any] | None = None,
) -> list[str]:
    """Recompute the local scientific-negative contract from immutable objects."""

    errors: list[str] = []
    disposition = disposition_object.get("payload")
    disposition = disposition if isinstance(disposition, Mapping) else {}
    attempt_payload = attempt.get("payload")
    attempt_payload = attempt_payload if isinstance(attempt_payload, Mapping) else {}
    attempt_id = str(attempt.get("object_id") or "").strip()
    if (
        attempt.get("state") != "failed"
        or str(attempt_payload.get("failure_class") or "").strip().lower()
        != "scientific_negative_result"
    ):
        errors.append("terminal_negative_failure_class_invalid")
    if registry_row is not None:
        if (
            _normalized_state(registry_row.get("status")) != "failed"
            or str(registry_row.get("error_type") or "").strip().lower()
            != "scientific_negative_result"
        ):
            errors.append("terminal_negative_registry_class_invalid")

    artifact = disposition.get("negative_result_artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    relative_text = str(artifact.get("path") or "").strip()
    relative = PurePosixPath(relative_text)
    if (
        not relative_text
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        errors.append("terminal_negative_artifact_path_invalid")
        encoded = None
    else:
        try:
            repository_root = Path(repository.path).resolve()
            target = repository_root.joinpath(*relative.parts).resolve(strict=True)
            target.relative_to(repository_root)
            encoded = read_bounded_regular_file(
                target,
                maximum=_MAX_NEGATIVE_RESULT_ARTIFACT_BYTES,
                label="terminal_negative_artifact",
            )
        except (BoundedFileError, OSError, ValueError):
            encoded = None
            errors.append("terminal_negative_artifact_unreadable")
    if encoded is not None:
        actual_hash = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if artifact.get("content_hash") != actual_hash or artifact.get(
            "size_bytes"
        ) != len(encoded):
            errors.append("terminal_negative_artifact_receipt_invalid")
        attempt_hashes = _content_hashes(attempt_payload.get("result_artifact_hashes"))
        row_artifacts = (
            registry_row.get("artifacts") if registry_row is not None else {}
        )
        row_artifacts = row_artifacts if isinstance(row_artifacts, Mapping) else {}
        row_hashes = (
            _content_hashes(row_artifacts.get("artifact_hashes"))
            if registry_row is not None
            else attempt_hashes
        )
        if (
            not attempt_hashes
            or attempt_hashes != row_hashes
            or actual_hash not in attempt_hashes
        ):
            errors.append("terminal_negative_artifact_binding_invalid")

    evidence_id = str(disposition.get("negative_result_evidence_id") or "").strip()
    try:
        evidence = repository.get(evidence_id)
    except Exception:
        evidence = {}
        errors.append("terminal_negative_evidence_missing")
    evidence_payload = evidence.get("payload")
    evidence_payload = evidence_payload if isinstance(evidence_payload, Mapping) else {}
    expected_measurement_hash = content_hash(
        {
            "result": evidence_payload.get("result"),
            "metrics": evidence_payload.get("metrics") or {},
        }
    )
    if (
        evidence.get("kind") != "evidence"
        or evidence.get("state") not in {"completed", "verified"}
        or _relation_targets(evidence, "derived_from") != {attempt_id}
        or not str(evidence_payload.get("result") or "").strip()
        or not isinstance(evidence_payload.get("metrics"), Mapping)
        or not evidence_payload.get("metrics")
        or evidence_payload.get("measurement_hash") != expected_measurement_hash
        or disposition.get("negative_result_evidence_hash")
        != evidence.get("content_hash")
        or disposition.get("negative_result_measurement_hash")
        != expected_measurement_hash
        or _relation_targets(
            disposition_object, "depends_on", "negative_result_evidence"
        )
        != {evidence_id}
    ):
        errors.append("terminal_negative_evidence_assessment_invalid")
    if disposition.get("negative_result_preserved") is not True:
        errors.append("terminal_negative_preservation_not_host_verified")
    return sorted(set(errors))


def _registration_object(repository: Any, preregistration: Mapping[str, Any]) -> dict:
    matches = [
        item
        for item in repository.objects(kind="preregistration", state="locked")
        if (item.get("payload") or {}).get("registration_hash")
        == preregistration.get("registration_hash")
        and (item.get("payload") or {}).get("preregistration_id")
        == preregistration.get("preregistration_id")
    ]
    if len(matches) != 1:
        raise ValueError("trajectory_registration_object_not_unique")
    return matches[0]


def attempt_registry_contract_errors(
    row: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    registration_object_id: str,
) -> list[str]:
    errors: list[str] = []
    payload = attempt.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if attempt.get("kind") != "experiment_attempt":
        errors.append("trajectory_attempt_kind_invalid")
    if _relation_targets(attempt, "depends_on", "protocol") != {registration_object_id}:
        errors.append("trajectory_attempt_protocol_relation_invalid")
    comparisons = {
        "task_id": str(row.get("task_id") or "").strip(),
        "study_phase": "confirmatory",
        "preregistration_id": str(row.get("preregistration_id") or "").strip(),
        "adaptive_state_hash": row.get("adaptive_state_hash"),
        "research_state_hash": row.get("research_state_hash"),
        "post_freeze_adaptation": False,
        "data_manifest_hash": row.get("data_manifest_hash"),
        "data_snapshot_id": row.get("data_snapshot_id"),
        "dataset_split_hash": row.get("dataset_split_hash"),
        "protocol_fidelity_hash": row.get("protocol_fidelity_hash"),
        "evidence_role": row.get("evidence_role"),
        "paired_control_task_id": row.get("paired_control_task_id"),
        "intervention_variant": row.get("intervention_variant"),
        "stress_condition": row.get("stress_condition"),
    }
    mismatches = [
        field
        for field, expected in comparisons.items()
        if payload.get(field) != expected
    ]
    if mismatches:
        errors.append("trajectory_attempt_contract_mismatch:" + ",".join(mismatches))
    if _normalized_state(row.get("status")) != str(attempt.get("state") or ""):
        errors.append("trajectory_attempt_status_mismatch")
    if (
        str(payload.get("producer_id") or "").strip()
        != str(row.get("producer_id") or "").strip()
    ):
        errors.append("trajectory_attempt_producer_mismatch")
    try:
        payload_producer = canonical_principal(
            payload.get("producer_id"), label="trajectory attempt producer_id"
        )
        actor_producer = canonical_principal(
            (attempt.get("actor") or {}).get("actor_id"),
            label="trajectory attempt actor",
        )
        if payload_producer != actor_producer:
            errors.append("trajectory_attempt_actor_producer_mismatch")
    except ValueError:
        errors.append("trajectory_attempt_actor_producer_invalid")
    if str(attempt.get("state") or "") == "completed":
        if payload.get("configuration_hash") != row.get("configuration_hash"):
            errors.append("trajectory_attempt_configuration_mismatch")
        row_hashes = _content_hashes(
            (row.get("artifacts") or {}).get("artifact_hashes")
        )
        attempt_hashes = _content_hashes(payload.get("result_artifact_hashes"))
        if not row_hashes or row_hashes != attempt_hashes:
            errors.append("trajectory_attempt_artifact_mismatch")
    elif str(attempt.get("state") or "") in {"failed", "timed_out", "cancelled"}:
        if not str(payload.get("failure_class") or "").strip():
            errors.append("trajectory_attempt_failure_class_missing")
    return errors


def attest_structured_trajectory(
    paper_root: str | Path,
    preregistration: Mapping[str, Any],
    experiment_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute registry↔object↔checkpoint closure from the live repository."""

    errors: list[str] = []
    try:
        from xscientist.research_git import repository_status
        from xscientist.research_lifecycle import ResearchLifecycle
        from xscientist.research_vcs import ResearchRepository

        status = repository_status(Path(paper_root).expanduser().resolve())
        repository = ResearchRepository(str(status["repository"]))
        if repository.status().get("worktree_clean") is not True:
            raise ValueError("trajectory_research_worktree_not_clean")
        registration = _registration_object(repository, preregistration)
        registration_object_id = str(registration.get("object_id") or "")
        adaptive_freeze = preregistration.get("adaptive_state_freeze")
        if not isinstance(adaptive_freeze, Mapping):
            raise ValueError("trajectory_adaptive_freeze_missing")
        lineage = ResearchLifecycle(repository)._confirmatory_lineage_attestation(
            registration=registration,
            preregistration_id=registration_object_id,
            preregistration=preregistration,
            adaptive_freeze=adaptive_freeze,
        )
        lineage_head = str(lineage.get("lineage_head") or "")
        frozen_head = str(lineage.get("frozen_head") or "")
        try:
            canonical_trajectory = repository.trajectory(
                ref=lineage_head,
                require_complete=True,
            )
        except Exception as exc:
            raise ValueError(
                "trajectory_projection_invalid:"
                + (str(exc) or "complete_trajectory_validation_failed")
            ) from exc
        trajectory_projection_receipt = {
            "schema_version": canonical_trajectory.get("schema_version"),
            "projection_hash": canonical_trajectory.get("projection_hash"),
            "resolved_head": canonical_trajectory.get("resolved_head"),
            "complete": canonical_trajectory.get("complete") is True,
            "truncated": canonical_trajectory.get("truncated") is True,
            "checkpoint_count": canonical_trajectory.get("checkpoint_count"),
            "boundary_parent_edges": [
                dict(item)
                for item in canonical_trajectory.get("boundary_parent_edges") or []
                if isinstance(item, Mapping)
            ],
            "boundary_rollback_edges": [
                dict(item)
                for item in canonical_trajectory.get("boundary_rollback_edges") or []
                if isinstance(item, Mapping)
            ],
        }
        paper_relative = (
            Path(paper_root).expanduser().resolve().relative_to(repository.path)
        )
        registry_repository_path = (
            paper_relative / "experiment_registry.jsonl"
        ).as_posix()
    except Exception as exc:
        return {
            "ok": False,
            "errors": [str(exc) or "trajectory_repository_attestation_failed"],
            "trajectory_hash": None,
            "trajectory_projection": None,
            "failed_record_ids": [],
            "incomplete_record_ids": [],
            "disposed_attempt_record_ids": [],
            "publication_blocking_attempt_record_ids": [],
            "publication_ready": False,
        }

    records = [dict(item) for item in experiment_records if isinstance(item, Mapping)]
    evidence_rows = [
        row
        for row in records
        if row.get("record_type") != "attempt_disposition"
        and (
            str(row.get("study_phase") or "").strip().lower() == "confirmatory"
            or row.get("independent_reproduction") is True
        )
    ]
    evidence_by_id = {
        str(row.get("record_id") or "").strip(): row for row in evidence_rows
    }
    if (
        not evidence_rows
        or any(not record_id for record_id in evidence_by_id)
        or len(evidence_by_id) != len(evidence_rows)
    ):
        errors.append("trajectory_registry_record_ids_invalid")

    trajectory_objects = [
        item
        for item in repository.objects(kind="gate_decision")
        if (item.get("payload") or {}).get("protocol_kind")
        == TRAJECTORY_BINDING_PROTOCOL
    ]
    disposition_objects = [
        item
        for item in repository.objects(kind="gate_decision")
        if (item.get("payload") or {}).get("protocol_kind")
        == ATTEMPT_DISPOSITION_PROTOCOL
    ]
    bindings_by_record: dict[str, list[dict[str, Any]]] = {}
    for item in trajectory_objects:
        record_id = str((item.get("payload") or {}).get("record_id") or "").strip()
        bindings_by_record.setdefault(record_id, []).append(item)

    binding_summaries: list[dict[str, Any]] = []
    bound_attempt_sequence: list[str] = []
    bound_attempt_ids: set[str] = set()
    attempt_checkpoint_hashes: set[str] = set()
    binding_checkpoint_hashes: set[str] = set()
    for record_id, row in sorted(evidence_by_id.items()):
        candidates = bindings_by_record.get(record_id, [])
        if len(candidates) != 1:
            errors.append(f"trajectory_binding_count_invalid:{record_id}")
            continue
        binding = candidates[0]
        payload = binding.get("payload") or {}
        attempt_id = str(payload.get("attempt_id") or "").strip()
        binding_id = str(binding.get("object_id") or "").strip()
        try:
            attempt = repository.get(attempt_id)
            origin = (
                repository.blame(attempt_id, commit=lineage_head).get("origin") or {}
            )
            origin_commit = str(origin.get("commit") or "").strip()
            shown = repository.show(origin_commit)
            checkpoint = shown.get("checkpoint") or {}
            checkpoint_hash = str(checkpoint.get("content_hash") or "").strip()
            binding_origin = (
                repository.blame(binding_id, commit=lineage_head).get("origin") or {}
            )
            binding_origin_commit = str(binding_origin.get("commit") or "").strip()
            binding_shown = repository.show(binding_origin_commit)
            binding_checkpoint = binding_shown.get("checkpoint") or {}
            binding_checkpoint_hash = str(
                binding_checkpoint.get("content_hash") or ""
            ).strip()
        except Exception:
            errors.append(f"trajectory_attempt_resolution_failed:{record_id}")
            continue
        attempt_path = f".xscientist/objects/experiment_attempt/{attempt_id}.json"
        binding_path = f".xscientist/objects/gate_decision/{binding_id}.json"
        attempt_changed_paths = {
            str(path) for path in checkpoint.get("changed_paths") or []
        }
        binding_changed_paths = {
            str(path) for path in binding_checkpoint.get("changed_paths") or []
        }
        attempt_object_paths = {
            path
            for path in attempt_changed_paths
            if path.startswith(".xscientist/objects/")
        }
        binding_object_paths = {
            path
            for path in binding_changed_paths
            if path.startswith(".xscientist/objects/")
        }
        binding_core = {
            key: value for key, value in payload.items() if key != "binding_hash"
        }
        if (
            payload.get("binding_hash") != canonical_content_hash(binding_core)
            or payload.get("registry_row_hash") != registry_row_hash(row)
            or payload.get("attempt_content_hash") != attempt.get("content_hash")
            or payload.get("attempt_origin_commit") != origin_commit
            or payload.get("attempt_checkpoint_hash") != checkpoint_hash
            or shown.get("checkpoint_hash_valid") is not True
            or checkpoint.get("stage")
            != (
                "experiment"
                if str(attempt.get("state") or "") == "completed"
                else "failed"
            )
            or attempt_object_paths != {attempt_path}
            or binding_shown.get("checkpoint_hash_valid") is not True
            or binding_checkpoint.get("stage") != "review"
            or binding_object_paths != {binding_path}
            or registry_repository_path not in binding_changed_paths
            or (binding.get("actor") or {}).get("authority") != "deterministic_gate"
            or _relation_targets(binding, "attests") != {attempt_id}
            or _relation_targets(binding, "depends_on", "protocol")
            != {registration_object_id}
        ):
            errors.append(f"trajectory_binding_payload_invalid:{record_id}")
            continue
        errors.extend(
            f"{error}:{record_id}"
            for error in attempt_registry_contract_errors(
                row, attempt, registration_object_id=registration_object_id
            )
        )
        bound_attempt_sequence.append(attempt_id)
        bound_attempt_ids.add(attempt_id)
        attempt_checkpoint_hashes.add(checkpoint_hash)
        binding_checkpoint_hashes.add(binding_checkpoint_hash)
        binding_summaries.append(
            {
                "record_id": record_id,
                "registry_row_hash": registry_row_hash(row),
                "binding_object_id": binding_id,
                "binding_object_hash": binding.get("content_hash"),
                "binding_origin_commit": binding_origin_commit,
                "binding_checkpoint_hash": binding_checkpoint_hash,
                "attempt_object_id": attempt_id,
                "attempt_object_hash": attempt.get("content_hash"),
                "attempt_origin_commit": origin_commit,
                "attempt_checkpoint_hash": checkpoint_hash,
            }
        )

    lineage_attempt_ids = {
        str(item) for item in lineage.get("prior_confirmatory_attempt_ids") or []
    }
    if len(bound_attempt_sequence) != len(bound_attempt_ids):
        errors.append("trajectory_attempt_binding_not_one_to_one")
    if bound_attempt_ids != lineage_attempt_ids:
        errors.append("trajectory_attempt_registry_bijection_invalid")
    if set(bindings_by_record) - set(evidence_by_id):
        errors.append("trajectory_orphan_bindings_present")
    if len(binding_checkpoint_hashes) != len(binding_summaries):
        errors.append("trajectory_binding_checkpoint_not_one_to_one")

    disposition_by_attempt: dict[str, list[dict[str, Any]]] = {}
    for item in disposition_objects:
        disposition_by_attempt.setdefault(
            str((item.get("payload") or {}).get("attempt_record_id") or "").strip(),
            [],
        ).append(item)
    disposition_summaries: list[dict[str, Any]] = []
    disposition_checkpoint_hashes: set[str] = set()
    publication_resolved_attempt_ids: set[str] = set()
    for record_id, row in sorted(evidence_by_id.items()):
        if _normalized_state(row.get("status")) not in {
            "failed",
            "timed_out",
            "cancelled",
        }:
            continue
        rows = disposition_by_attempt.get(record_id, [])
        if len(rows) != 1:
            errors.append(f"trajectory_disposition_count_invalid:{record_id}")
            continue
        disposition_object = rows[0]
        disposition = disposition_object.get("payload") or {}
        disposition_type = str(disposition.get("disposition") or "").strip()
        core = {
            key: value
            for key, value in disposition.items()
            if key != "disposition_hash"
        }
        binding = bindings_by_record.get(record_id, [{}])[0].get("payload") or {}
        disposition_object_id = str(disposition_object.get("object_id") or "").strip()
        try:
            attempt = repository.get(str(binding.get("attempt_id") or ""))
            disposition_origin = (
                repository.blame(
                    disposition_object_id,
                    commit=lineage_head,
                ).get("origin")
                or {}
            )
            disposition_origin_commit = str(
                disposition_origin.get("commit") or ""
            ).strip()
            disposition_shown = repository.show(disposition_origin_commit)
            disposition_checkpoint = disposition_shown.get("checkpoint") or {}
            disposition_checkpoint_hash = str(
                disposition_checkpoint.get("content_hash") or ""
            ).strip()
        except Exception:
            errors.append(f"trajectory_disposition_resolution_failed:{record_id}")
            continue
        disposition_path = (
            f".xscientist/objects/gate_decision/{disposition_object_id}.json"
        )
        disposition_object_paths = {
            str(path)
            for path in disposition_checkpoint.get("changed_paths") or []
            if str(path).startswith(".xscientist/objects/")
        }
        if (
            disposition_type not in ATTEMPT_DISPOSITIONS
            or not str(disposition.get("reason") or "").strip()
            or disposition.get("attempt_record_hash") != registry_row_hash(row)
            or disposition.get("disposition_hash") != canonical_content_hash(core)
            or disposition.get("attempt_id") != binding.get("attempt_id")
            or disposition.get("attempt_content_hash")
            != binding.get("attempt_content_hash")
            or (disposition_object.get("actor") or {}).get("authority") != "recorder"
            or disposition_shown.get("checkpoint_hash_valid") is not True
            or disposition_checkpoint.get("stage") != "review"
            or disposition_object_paths != {disposition_path}
            or _relation_targets(disposition_object, "attests")
            != {str(binding.get("attempt_id") or "")}
            or _relation_targets(disposition_object, "depends_on", "protocol")
            != {registration_object_id}
        ):
            errors.append(f"trajectory_disposition_invalid:{record_id}")
            continue
        if disposition_type == "technical_failure_retried":
            retry_id = str(disposition.get("retry_record_id") or "").strip()
            retry = evidence_by_id.get(retry_id)
            retry_bindings = bindings_by_record.get(retry_id, [])
            retry_binding = (
                retry_bindings[0].get("payload") or {}
                if len(retry_bindings) == 1
                else {}
            )
            retry_attempt_id = str(retry_binding.get("attempt_id") or "").strip()
            if (
                not retry
                or retry_id == record_id
                or retry.get("task_id") != row.get("task_id")
                or retry.get("preregistration_id") != row.get("preregistration_id")
                or _normalized_state(retry.get("status")) != "completed"
                or len(retry_bindings) != 1
                or not retry_attempt_id
                or disposition.get("retry_attempt_id") != retry_attempt_id
                or _relation_targets(disposition_object, "depends_on", "retry")
                != {retry_attempt_id}
            ):
                errors.append(f"trajectory_disposition_retry_invalid:{record_id}")
                continue
        elif (
            str(disposition.get("retry_record_id") or "").strip()
            or str(disposition.get("retry_attempt_id") or "").strip()
            or _relation_targets(disposition_object, "depends_on", "retry")
        ):
            errors.append(f"trajectory_disposition_retry_invalid:{record_id}")
            continue
        if (
            disposition_type == "approved_deviation"
            and disposition.get("approved_before_unblinding") is not True
        ):
            errors.append(f"trajectory_disposition_approval_invalid:{record_id}")
            continue
        if disposition_type == "terminal_negative":
            terminal_errors = terminal_negative_contract_errors(
                repository,
                disposition_object,
                attempt,
                registry_row=row,
            )
            if terminal_errors:
                errors.extend(f"{error}:{record_id}" for error in terminal_errors)
                continue
        resolves_publication_block = disposition_type in {
            "terminal_negative",
            "technical_failure_retried",
        }
        if resolves_publication_block:
            publication_resolved_attempt_ids.add(record_id)
        disposition_checkpoint_hashes.add(disposition_checkpoint_hash)
        disposition_summaries.append(
            {
                "attempt_record_id": record_id,
                "disposition": disposition_type,
                "resolves_publication_block": resolves_publication_block,
                "disposition_object_id": disposition_object_id,
                "disposition_object_hash": disposition_object.get("content_hash"),
                "disposition_origin_commit": disposition_origin_commit,
                "disposition_checkpoint_hash": disposition_checkpoint_hash,
            }
        )

    unused_dispositions = set(disposition_by_attempt) - set(evidence_by_id)
    if unused_dispositions:
        errors.append("trajectory_orphan_dispositions_present")
    if len(disposition_checkpoint_hashes) != len(disposition_summaries):
        errors.append("trajectory_disposition_checkpoint_not_one_to_one")
    core = {
        "schema": "xscientist.structured-trajectory-attestation.v2",
        "frozen_head": frozen_head,
        "lineage_head": lineage_head,
        "trajectory_projection": trajectory_projection_receipt,
        "registration_object_id": registration_object_id,
        "registration_object_hash": registration.get("content_hash"),
        "bindings": binding_summaries,
        "dispositions": disposition_summaries,
        "checkpoint_hashes": sorted(
            attempt_checkpoint_hashes
            | binding_checkpoint_hashes
            | disposition_checkpoint_hashes
        ),
    }
    trajectory_hash = canonical_content_hash(core) if not errors else None
    failed_record_ids = sorted(
        record_id
        for record_id, row in evidence_by_id.items()
        if _normalized_state(row.get("status")) in {"failed", "timed_out", "cancelled"}
    )
    incomplete_record_ids = sorted(
        record_id
        for record_id, row in evidence_by_id.items()
        if _normalized_state(row.get("status")) != "completed"
    )
    publication_blocking_attempt_ids = sorted(
        set(incomplete_record_ids) - publication_resolved_attempt_ids
    )
    return {
        **core,
        "ok": not errors,
        "errors": sorted(set(errors)),
        "trajectory_hash": trajectory_hash,
        "attempt_object_ids": sorted(bound_attempt_ids),
        "binding_object_ids": sorted(
            str(item.get("binding_object_id") or "") for item in binding_summaries
        ),
        "failed_record_ids": failed_record_ids,
        "incomplete_record_ids": incomplete_record_ids,
        "disposition_object_ids": sorted(
            str(item.get("disposition_object_id") or "")
            for item in disposition_summaries
        ),
        "recorded_disposition_attempt_record_ids": sorted(
            str(item.get("attempt_record_id") or "") for item in disposition_summaries
        ),
        "disposed_attempt_record_ids": sorted(publication_resolved_attempt_ids),
        "publication_blocking_attempt_record_ids": publication_blocking_attempt_ids,
        "publication_ready": not errors and not publication_blocking_attempt_ids,
    }


__all__ = [
    "ATTEMPT_DISPOSITION_PROTOCOL",
    "ATTEMPT_DISPOSITIONS",
    "TRAJECTORY_BINDING_PROTOCOL",
    "attempt_registry_contract_errors",
    "attest_structured_trajectory",
    "build_terminal_negative_artifact_receipt",
    "registry_row_hash",
    "terminal_negative_contract_errors",
]
