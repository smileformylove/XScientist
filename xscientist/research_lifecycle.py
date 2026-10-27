"""Scientific lifecycle operations built on native Research VCS objects."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from copy import deepcopy
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ai_scientist.protocol import (
    ResearchObjectError,
    build_research_object,
    content_hash,
)
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.safe_files import BoundedFileError, read_bounded_regular_file
from ai_scientist.utils.principal_identity import canonical_principal
from ai_scientist.utils.research_integrity import (
    _protocol_fidelity_hash,
    validate_preregistration,
)
from ai_scientist.utils.trajectory_binding import terminal_negative_contract_errors

from .experiment_admission import (
    EXPERIMENT_ADMISSION_ACTOR,
    EXPERIMENT_ADMISSION_AUTHORITY_PROTOCOL,
    EXPERIMENT_ADMISSION_PROTOCOL,
    EXPERIMENT_ADMISSION_SCOPE_FIELDS,
    XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE,
    build_experiment_admission_payload,
    normalize_experiment_admission_payload,
    validate_experiment_admission_scope,
    verify_experiment_admission_attestation,
    verify_persisted_experiment_admission_gate,
)
from .experiment_admission_state import (
    begin_experiment_admission_transition,
    clear_experiment_admission_transition,
    experiment_admission_subject_hash,
    load_all_experiment_admission_high_water,
    load_all_experiment_admission_transitions,
    load_experiment_admission_high_water,
    recover_experiment_admission_staging,
    record_experiment_admission_high_water,
    require_current_experiment_admission_decision,
)
from .research_authority import (
    PRODUCER_LINEAGE_RELATIONS,
    require_independent_evaluator,
)
from .research_git import (
    CheckpointResult,
    ObjectPointerResult,
    ResearchGitError,
    ResearchObjectResult,
    _load_experiment_admission_terminal_completion_ref,
    _load_experiment_admission_execution_claim,
    _load_experiment_admission_terminal_reservation,
    _experiment_admission_terminal_reserved,
    _fsync_directory,
    _pointer_records_at_commit,
    _publish_experiment_admission_terminal_completion_ref,
    _repository_admission_forbidden_roots,
    _repository_read_lock,
    _repository_lock,
    _resolve_configured_path,
    _reserve_experiment_admission_terminal,
    _run_git,
    _validate_pointer_payload,
    _verify_pointer_object,
    create_checkpoint,
    list_research_objects_at_ref,
    load_repository_config_at_ref,
    show_checkpoint,
)
from .research_vcs import ResearchRepository

_MAX_CONFIRMATORY_LINEAGE_COMMITS = 4096
_MAX_CONFIRMATORY_CAMPAIGN_BYTES = 16 * 1024 * 1024
_MAX_RESULT_ARTIFACT_POINTER_BYTES = 1024 * 1024
_RESEARCH_OBJECT_PATH = re.compile(
    r"^\.xscientist/objects/(?P<kind>[a-z][a-z0-9_-]*)/"
    r"(?P<object_id>rso-[0-9a-f]{16})\.json$"
)

_EXPERIMENT_ADMISSION_PROTOCOL = EXPERIMENT_ADMISSION_PROTOCOL
_EXPERIMENT_ADMISSION_PREFLIGHT_PROTOCOL = (
    "xscientist.experiment-admission-preflight.v1"
)
_EXPERIMENT_ADMISSION_FINAL_PROTOCOL = "xscientist.experiment-admission-preflight.v2"
_EXPERIMENT_ADMISSION_START_PROTOCOL = "xscientist.experiment-admission-start.v1"
_EXPERIMENT_ADMISSION_STATES = {"verified", "promoted"}
_EXPERIMENT_ADMISSION_SCOPE_FIELDS = EXPERIMENT_ADMISSION_SCOPE_FIELDS
_EXPERIMENT_ADMISSION_TERMINAL_POLICY_SNAPSHOT_PROTOCOL = (
    "xscientist.experiment-admission-terminal-policy-snapshot.v1"
)
_EXPERIMENT_ADMISSION_TERMINAL_POLICY_FIELDS = (
    "admitted_at_start",
    "current_policy_status",
    "quarantined",
    "claim_promotion_allowed",
)
_CONTENT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXPERIMENT_TRANSITION_PROTOCOL = "xscientist.experiment-transition.v1"
_EXPERIMENT_TRANSITION_KINDS = {
    "follow_up",
    "replication",
    "extension",
    "recovery",
}
_EXPERIMENT_TRANSITION_REUSE_KINDS = {
    "experiment_attempt",
    "evidence",
    "observation",
    "source_snapshot",
    "passage_evidence",
    "context_snapshot",
}
_EXPERIMENT_TRANSITION_TERMINAL_STATES = {
    "completed",
    "failed",
    "timed_out",
    "cancelled",
    "rejected",
}
_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER = object()
_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER = object()


def _validate_admission_runtime_scope(scope: Mapping[str, Any]) -> None:
    """Validate live scope values while preserving legacy tuple seed inputs.

    Signed payloads remain strict JSON lists. A few in-process callers pass a
    tuple for runtime seeds; validation inspects a list copy while the original
    tuple remains available to existing subset/matching logic.
    """

    normalized = dict(scope) if isinstance(scope, Mapping) else scope
    if isinstance(normalized, dict):
        for field in ("allowed_seeds", "seeds"):
            if isinstance(normalized.get(field), tuple):
                normalized[field] = list(normalized[field])
    validate_experiment_admission_scope(normalized)


def _result_artifact_pointer_identity(path: Path) -> tuple[int, ...]:
    metadata = path.lstat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class _ResultArtifactPointerTransaction:
    """Opaque pre-write snapshot for one result-artifact pointer transaction."""

    __slots__ = (
        "__head",
        "__planned_bindings",
        "__preexisting_paths",
        "__repository_path",
    )

    def __init__(
        self,
        *,
        repository_path: Path,
        head: str,
        planned_bindings: Mapping[str, tuple[str, str]],
        preexisting_paths: set[str],
        issuer: object,
    ) -> None:
        if issuer is not _RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER:
            raise TypeError("result artifact pointer transaction issuer is invalid")
        object.__setattr__(self, "_ResultArtifactPointerTransaction__head", head)
        object.__setattr__(
            self,
            "_ResultArtifactPointerTransaction__planned_bindings",
            MappingProxyType(dict(planned_bindings)),
        )
        object.__setattr__(
            self,
            "_ResultArtifactPointerTransaction__preexisting_paths",
            frozenset(preexisting_paths),
        )
        object.__setattr__(
            self,
            "_ResultArtifactPointerTransaction__repository_path",
            str(repository_path.resolve()),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("result artifact pointer transactions are immutable")

    def _snapshot(
        self,
        *,
        repository_path: Path,
        issuer: object,
    ) -> tuple[str, dict[str, tuple[str, str]], set[str]]:
        if issuer is not _RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER:
            raise ResearchGitError(
                "result artifact pointer transaction issuer is invalid"
            )
        if self.__repository_path != str(repository_path.resolve()):
            raise ResearchGitError(
                "result artifact pointer transaction belongs to another repository"
            )
        return (
            self.__head,
            dict(self.__planned_bindings),
            set(self.__preexisting_paths),
        )


class _ResultArtifactPointerRollbackReceipt:
    """Verified pointer identities that a failed transaction may safely unlink."""

    __slots__ = ("__head", "__records", "__repository_path")

    def __init__(
        self,
        *,
        repository_path: Path,
        head: str,
        records: Mapping[str, tuple[tuple[int, ...], str]],
        issuer: object,
    ) -> None:
        if issuer is not _RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER:
            raise TypeError("result artifact rollback receipt issuer is invalid")
        object.__setattr__(self, "_ResultArtifactPointerRollbackReceipt__head", head)
        object.__setattr__(
            self,
            "_ResultArtifactPointerRollbackReceipt__records",
            MappingProxyType(dict(records)),
        )
        object.__setattr__(
            self,
            "_ResultArtifactPointerRollbackReceipt__repository_path",
            str(repository_path.resolve()),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("result artifact rollback receipts are immutable")

    def _snapshot(
        self,
        *,
        repository_path: Path,
        issuer: object,
    ) -> tuple[str, dict[str, tuple[tuple[int, ...], str]]]:
        if issuer is not _RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER:
            raise ResearchGitError("result artifact rollback receipt issuer is invalid")
        if self.__repository_path != str(repository_path.resolve()):
            raise ResearchGitError(
                "result artifact rollback receipt belongs to another repository"
            )
        return self.__head, dict(self.__records)


def _freeze_validated_attempt_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_validated_attempt_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_validated_attempt_value(item) for item in value)
    return value


def _thaw_validated_attempt_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_validated_attempt_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_validated_attempt_value(item) for item in value]
    return value


class _ValidatedExperimentAttempt(Mapping[str, Any]):
    """Repository-bound, deeply immutable output of attempt validation."""

    __slots__ = (
        "__contract",
        "__contract_hash",
        "__pending_result_paths",
        "__repository_path",
    )

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        repository_path: Path,
        issuer: object,
        pending_result_paths: Sequence[str] = (),
    ) -> None:
        if issuer is not _VALIDATED_EXPERIMENT_ATTEMPT_ISSUER:
            raise TypeError(
                "validated experiment attempts can only be issued by "
                "ResearchLifecycle.validate_experiment_attempt"
            )
        detached = deepcopy(dict(contract))
        try:
            contract_hash = content_hash(detached)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ResearchGitError(
                "validated experiment attempt contract must contain only JSON values"
            ) from exc
        object.__setattr__(
            self,
            "_ValidatedExperimentAttempt__contract",
            _freeze_validated_attempt_value(detached),
        )
        object.__setattr__(
            self,
            "_ValidatedExperimentAttempt__contract_hash",
            contract_hash,
        )
        object.__setattr__(
            self,
            "_ValidatedExperimentAttempt__repository_path",
            str(repository_path.resolve()),
        )
        object.__setattr__(
            self,
            "_ValidatedExperimentAttempt__pending_result_paths",
            tuple(sorted({str(path) for path in pending_result_paths})),
        )

    def __getitem__(self, key: str) -> Any:
        return self.__contract[key]

    def __iter__(self):
        return iter(self.__contract)

    def __len__(self) -> int:
        return len(self.__contract)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("validated experiment attempt contracts are immutable")

    def __delattr__(self, name: str) -> None:
        raise TypeError("validated experiment attempt contracts are immutable")

    def _materialize(self, *, repository_path: Path, issuer: object) -> dict[str, Any]:
        if issuer is not _VALIDATED_EXPERIMENT_ATTEMPT_ISSUER:
            raise ResearchGitError("validated experiment attempt issuer is invalid")
        if self.__repository_path != str(repository_path.resolve()):
            raise ResearchGitError(
                "validated experiment attempt belongs to a different repository"
            )
        contract = _thaw_validated_attempt_value(self.__contract)
        try:
            contract_hash = content_hash(contract)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ResearchGitError(
                "validated experiment attempt contract is no longer valid JSON"
            ) from exc
        if contract_hash != self.__contract_hash:
            raise ResearchGitError(
                "validated experiment attempt contract failed its integrity check"
            )
        return contract

    def _pending_paths(self, *, repository_path: Path, issuer: object) -> set[str]:
        if issuer is not _VALIDATED_EXPERIMENT_ATTEMPT_ISSUER:
            raise ResearchGitError("validated experiment attempt issuer is invalid")
        if self.__repository_path != str(repository_path.resolve()):
            raise ResearchGitError(
                "validated experiment attempt belongs to a different repository"
            )
        return set(self.__pending_result_paths)


def _canonical_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""

    try:
        return canonical_content_hash(left) == canonical_content_hash(right)
    except (TypeError, ValueError):
        return False


def _relation_targets(
    research_object: Mapping[str, Any],
    *,
    relation_type: str,
    role: str | None = None,
) -> set[str]:
    return {
        str(item.get("target") or "").strip()
        for item in research_object.get("relations") or []
        if isinstance(item, Mapping)
        and item.get("type") == relation_type
        and (role is None or item.get("role") == role)
        and str(item.get("target") or "").strip()
    }


def _registered_task_ids(preregistration: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("task_id") or "").strip()
        for item in preregistration.get("outcomes") or []
        if isinstance(item, Mapping) and str(item.get("task_id") or "").strip()
    }


def _safe_campaign_path(value: Any) -> str:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResearchGitError("confirmatory campaign artifact path is unsafe")
    return path.as_posix()


class ResearchLifecycle:
    """Record an evidence-gated research lifecycle without backend commands."""

    def __init__(
        self,
        repository: ResearchRepository | str | Path,
        *,
        admission_trust_store: str | Path | None = None,
    ) -> None:
        self.repository = (
            repository
            if isinstance(repository, ResearchRepository)
            else ResearchRepository(repository)
        )
        self._admission_trust_store = (
            str(admission_trust_store) if admission_trust_store is not None else None
        )

    def planning(
        self,
        *,
        hypothesis: Mapping[str, Any],
        plan: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record hypothesis -> plan -> preregistration provenance."""

        # Validate the whole multi-object transition before the first immutable
        # object is written.  A malformed locked registration must not leave a
        # hypothesis and plan behind for a later checkpoint to absorb.
        registration_payload: dict[str, Any] | None = None
        locked = False
        validation: dict[str, Any] | None = None
        if preregistration is not None:
            registration_payload = dict(preregistration)
            locked = registration_payload.get("status") == "locked"
            validation = validate_preregistration(
                registration_payload,
                require_locked=locked,
            )
            if locked and not validation["ok"]:
                raise ResearchGitError(
                    "locked preregistration failed integrity validation: "
                    + ", ".join(validation["errors"])
                )

        hypothesis_result = self.repository.record(
            "hypothesis",
            hypothesis,
            state="draft",
        )
        plan_result = self.repository.record(
            "research_plan",
            plan,
            state="draft",
            relations=[{"type": "depends_on", "target": hypothesis_result.object_id}],
        )
        preregistration_result: ResearchObjectResult | None = None
        if registration_payload is not None:
            preregistration_result = self.repository.record(
                "preregistration",
                registration_payload,
                state="locked" if locked else "draft",
                relations=[{"type": "depends_on", "target": plan_result.object_id}],
                actor={
                    "actor_id": str(
                        registration_payload.get("registered_by") or "research-planner"
                    ),
                    "authority": "research_agent",
                },
            )
        checkpoint: CheckpointResult | None = None
        if commit:
            checkpoint = self.repository.commit(
                stage="preregister" if locked else "ideation",
                subject=(
                    "lock confirmatory research plan"
                    if locked
                    else "record exploratory research plan"
                ),
                status="completed" if locked else "draft",
            )
        return {
            "hypothesis": hypothesis_result,
            "plan": plan_result,
            "preregistration": preregistration_result,
            "preregistration_validation": validation,
            "checkpoint": checkpoint,
        }

    def _validate_confirmatory_attempt_path(
        self,
        path: str,
        *,
        commit: str,
        current_head: str,
        preregistration_id: str,
        preregistration: Mapping[str, Any],
        adaptive_freeze: Mapping[str, Any],
    ) -> str:
        match = _RESEARCH_OBJECT_PATH.fullmatch(path)
        if match is None or match.group("kind") != "experiment_attempt":
            raise ResearchGitError(
                "confirmatory experiment lineage changed a frozen research path: "
                + path
            )
        object_id = match.group("object_id")
        attempt = self.repository.get(object_id)
        if attempt.get("kind") != "experiment_attempt":
            raise ResearchGitError(
                "confirmatory experiment lineage contains a non-attempt object"
            )
        try:
            origin = (
                self.repository.blame(object_id, commit=current_head).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory experiment lineage contains an untraceable attempt"
            ) from exc
        if str(origin.get("commit") or "") != commit:
            raise ResearchGitError(
                "confirmatory experiment lineage modified an existing attempt"
            )
        payload = attempt.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if str(payload.get("study_phase") or "").strip().lower() != "confirmatory":
            raise ResearchGitError(
                "confirmatory experiment lineage contains an exploratory attempt"
            )
        if _relation_targets(
            attempt,
            relation_type="depends_on",
            role="protocol",
        ) != {preregistration_id}:
            raise ResearchGitError(
                "confirmatory experiment lineage contains an attempt bound to a "
                "different protocol"
            )
        expected_bindings = {
            "adaptive_state_hash": adaptive_freeze.get("state_hash"),
            "research_state_hash": adaptive_freeze.get("research_state_hash"),
            "post_freeze_adaptation": False,
            "preregistration_id": preregistration.get("preregistration_id"),
        }
        task_id = str(payload.get("task_id") or "").strip()
        if task_id and task_id not in _registered_task_ids(preregistration):
            raise ResearchGitError(
                "confirmatory experiment lineage contains an unregistered task"
            )
        data_policy = preregistration.get("data_policy")
        data_policy = data_policy if isinstance(data_policy, Mapping) else {}
        for field in ("data_manifest_hash", "data_snapshot_id"):
            expected_bindings[field] = data_policy.get(field)
        if task_id:
            outcome = next(
                (
                    item
                    for item in preregistration.get("outcomes") or []
                    if isinstance(item, Mapping)
                    and str(item.get("task_id") or "").strip() == task_id
                ),
                {},
            )
            for field in (
                "evidence_role",
                "paired_control_task_id",
                "intervention_variant",
                "stress_condition",
                "transformation_contract",
                "transformation_contract_hash",
            ):
                expected_bindings[field] = outcome.get(field)
            expected_bindings["protocol_fidelity_hash"] = _protocol_fidelity_hash(
                preregistration, task_id
            )
            expected_bindings["dataset_split_hash"] = (
                data_policy.get("split_hashes") or {}
            ).get(task_id)
        if any(
            payload.get(field) != expected
            for field, expected in expected_bindings.items()
        ):
            raise ResearchGitError(
                "confirmatory experiment lineage contains an attempt that changed "
                "the frozen adaptive state, data, or task contract"
            )
        provenance = attempt.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        code_commit = str(provenance.get("code_commit") or "").strip()
        if code_commit != str(adaptive_freeze.get("research_vcs_head") or "").strip():
            raise ResearchGitError(
                "confirmatory experiment lineage is missing or changed the frozen "
                "code state"
            )
        return object_id

    def _validate_confirmatory_gate_path(
        self,
        path: str,
        *,
        commit: str,
        current_head: str,
        preregistration_id: str,
    ) -> tuple[str, str]:
        """Validate a post-freeze trajectory binding/disposition decision."""

        match = _RESEARCH_OBJECT_PATH.fullmatch(path)
        if match is None or match.group("kind") != "gate_decision":
            raise ResearchGitError(
                "confirmatory lineage contains a non-attempt/non-gate object"
            )
        object_id = match.group("object_id")
        decision = self.repository.get(object_id)
        if decision.get("kind") != "gate_decision" or decision.get("state") not in {
            "completed",
            "verified",
        }:
            raise ResearchGitError(
                "confirmatory lineage contains an invalid gate decision"
            )
        try:
            origin = (
                self.repository.blame(object_id, commit=current_head).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory lineage contains an untraceable gate decision"
            ) from exc
        if str(origin.get("commit") or "") != commit:
            raise ResearchGitError(
                "confirmatory lineage modified an existing gate decision"
            )
        payload = decision.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        protocol_kind = str(payload.get("protocol_kind") or "").strip()
        if protocol_kind not in {
            "xscientist.trajectory-binding.v1",
            "xscientist.attempt-disposition.v1",
        }:
            raise ResearchGitError(
                "confirmatory lineage contains an unrecognized gate decision"
            )
        actor = decision.get("actor")
        actor = actor if isinstance(actor, Mapping) else {}
        expected_authority = (
            "deterministic_gate"
            if protocol_kind == "xscientist.trajectory-binding.v1"
            else "recorder"
        )
        if (
            actor.get("authority") != expected_authority
            or not str(actor.get("actor_id") or "").strip()
        ):
            raise ResearchGitError(
                "confirmatory gate decision actor authority is invalid"
            )
        hash_field = (
            "binding_hash"
            if protocol_kind == "xscientist.trajectory-binding.v1"
            else "disposition_hash"
        )
        payload_core = {
            key: value for key, value in payload.items() if key != hash_field
        }
        if payload.get(hash_field) != canonical_content_hash(payload_core):
            raise ResearchGitError(
                "confirmatory gate decision integrity hash is invalid"
            )
        expected_decision = (
            "bind_registry_record_to_research_trajectory"
            if protocol_kind == "xscientist.trajectory-binding.v1"
            else "record_attempt_disposition"
        )
        if payload.get("decision") != expected_decision:
            raise ResearchGitError("confirmatory gate decision action is invalid")
        if _relation_targets(decision, relation_type="depends_on", role="protocol") != {
            preregistration_id
        }:
            raise ResearchGitError(
                "confirmatory gate decision is bound to a different protocol"
            )
        attempt_targets = _relation_targets(decision, relation_type="attests")
        attempt_id = str(payload.get("attempt_id") or "").strip()
        if attempt_targets != {attempt_id}:
            raise ResearchGitError(
                "confirmatory gate decision does not attest its declared attempt"
            )
        attempt = self.repository.get(attempt_id)
        if (
            attempt.get("kind") != "experiment_attempt"
            or (attempt.get("payload") or {}).get("study_phase") != "confirmatory"
            or _relation_targets(attempt, relation_type="depends_on", role="protocol")
            != {preregistration_id}
            or payload.get("attempt_content_hash") != attempt.get("content_hash")
        ):
            raise ResearchGitError(
                "confirmatory gate decision attempt binding is invalid"
            )
        if protocol_kind == "xscientist.trajectory-binding.v1":
            required_text_fields = (
                "record_id",
                "registry_row_hash",
                "attempt_origin_commit",
                "attempt_checkpoint_hash",
            )
            if any(
                not str(payload.get(field) or "").strip()
                for field in required_text_fields
            ) or any(
                _CONTENT_HASH_RE.fullmatch(str(payload.get(field) or "")) is None
                for field in (
                    "registry_row_hash",
                    "attempt_content_hash",
                    "attempt_checkpoint_hash",
                    "binding_hash",
                )
            ):
                raise ResearchGitError(
                    "confirmatory trajectory binding payload is incomplete"
                )
            if _relation_targets(
                decision,
                relation_type="depends_on",
                role="retry",
            ):
                raise ResearchGitError(
                    "confirmatory trajectory binding has an invalid retry relation"
                )
            attempt_origin = (
                self.repository.blame(attempt_id, commit=current_head).get("origin")
                or {}
            )
            attempt_origin_commit = str(attempt_origin.get("commit") or "").strip()
            try:
                attempt_checkpoint = self.repository.show(attempt_origin_commit)
            except ResearchGitError as exc:
                raise ResearchGitError(
                    "confirmatory trajectory binding attempt checkpoint is invalid"
                ) from exc
            if (
                payload.get("attempt_origin_commit") != attempt_origin_commit
                or attempt_checkpoint.get("checkpoint_hash_valid") is not True
                or payload.get("attempt_checkpoint_hash")
                != (attempt_checkpoint.get("checkpoint") or {}).get("content_hash")
            ):
                raise ResearchGitError(
                    "confirmatory trajectory binding attempt checkpoint is invalid"
                )
        else:
            disposition = str(payload.get("disposition") or "").strip()
            if (
                disposition
                not in {
                    "terminal_negative",
                    "technical_failure_retried",
                    "approved_deviation",
                    "excluded_with_reason",
                }
                or not str(payload.get("attempt_record_id") or "").strip()
                or not str(payload.get("reason") or "").strip()
                or _CONTENT_HASH_RE.fullmatch(
                    str(payload.get("attempt_record_hash") or "")
                )
                is None
                or _CONTENT_HASH_RE.fullmatch(
                    str(payload.get("attempt_content_hash") or "")
                )
                is None
                or not isinstance(payload.get("approved_before_unblinding"), bool)
                or not isinstance(payload.get("negative_result_preserved"), bool)
            ):
                raise ResearchGitError(
                    "confirmatory attempt disposition payload is incomplete"
                )
            retry_targets = _relation_targets(
                decision,
                relation_type="depends_on",
                role="retry",
            )
            retry_attempt_id = str(payload.get("retry_attempt_id") or "").strip()
            retry_record_id = str(payload.get("retry_record_id") or "").strip()
            if disposition == "technical_failure_retried":
                if (
                    not retry_record_id
                    or not retry_attempt_id
                    or retry_targets != {retry_attempt_id}
                ):
                    raise ResearchGitError(
                        "confirmatory attempt disposition retry binding is invalid"
                    )
                retry_attempt = self.repository.get(retry_attempt_id)
                retry_payload = retry_attempt.get("payload") or {}
                attempt_payload = attempt.get("payload") or {}
                if (
                    retry_attempt.get("kind") != "experiment_attempt"
                    or retry_attempt.get("state") != "completed"
                    or retry_payload.get("study_phase") != "confirmatory"
                    or _relation_targets(
                        retry_attempt,
                        relation_type="depends_on",
                        role="protocol",
                    )
                    != {preregistration_id}
                    or retry_payload.get("task_id") != attempt_payload.get("task_id")
                    or retry_payload.get("preregistration_id")
                    != attempt_payload.get("preregistration_id")
                ):
                    raise ResearchGitError(
                        "confirmatory attempt disposition retry attempt is invalid"
                    )
            elif retry_record_id or retry_attempt_id or retry_targets:
                raise ResearchGitError(
                    "confirmatory attempt disposition has an unexpected retry binding"
                )
            if (
                disposition == "approved_deviation"
                and payload.get("approved_before_unblinding") is not True
            ):
                raise ResearchGitError(
                    "confirmatory attempt disposition approval is invalid"
                )
            if disposition == "terminal_negative":
                terminal_errors = terminal_negative_contract_errors(
                    self.repository,
                    decision,
                    attempt,
                )
                if terminal_errors:
                    raise ResearchGitError(
                        "confirmatory attempt disposition negative result is not "
                        "host-verifiable: " + ", ".join(terminal_errors)
                    )
        return object_id, protocol_kind

    def _validate_confirmatory_negative_evidence_path(
        self,
        path: str,
        *,
        commit: str,
        current_head: str,
        preregistration_id: str,
    ) -> str:
        """Admit only a metric-bearing assessment of one scientific failure."""

        match = _RESEARCH_OBJECT_PATH.fullmatch(path)
        if match is None or match.group("kind") != "evidence":
            raise ResearchGitError(
                "confirmatory lineage contains an unsupported evidence object"
            )
        object_id = match.group("object_id")
        evidence = self.repository.get(object_id)
        try:
            origin = (
                self.repository.blame(object_id, commit=current_head).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory lineage contains untraceable negative evidence"
            ) from exc
        payload = evidence.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        attempt_ids = _relation_targets(
            evidence,
            relation_type="derived_from",
        )
        if len(attempt_ids) != 1:
            raise ResearchGitError(
                "confirmatory negative evidence must derive from exactly one attempt"
            )
        attempt = self.repository.get(next(iter(attempt_ids)))
        attempt_payload = attempt.get("payload")
        attempt_payload = (
            attempt_payload if isinstance(attempt_payload, Mapping) else {}
        )
        expected_measurement_hash = content_hash(
            {
                "result": payload.get("result"),
                "metrics": payload.get("metrics") or {},
            }
        )
        if (
            evidence.get("kind") != "evidence"
            or evidence.get("state") not in {"completed", "verified"}
            or str(origin.get("commit") or "") != commit
            or attempt.get("kind") != "experiment_attempt"
            or attempt.get("state") != "failed"
            or attempt_payload.get("study_phase") != "confirmatory"
            or str(attempt_payload.get("failure_class") or "").strip().lower()
            != "scientific_negative_result"
            or _relation_targets(
                attempt,
                relation_type="depends_on",
                role="protocol",
            )
            != {preregistration_id}
            or not str(payload.get("result") or "").strip()
            or not isinstance(payload.get("metrics"), Mapping)
            or not payload.get("metrics")
            or payload.get("measurement_hash") != expected_measurement_hash
        ):
            raise ResearchGitError(
                "confirmatory negative evidence is not bound to one host-verifiable "
                "scientific negative attempt"
            )
        return object_id

    def _confirmatory_lineage_attestation(
        self,
        *,
        registration: Mapping[str, Any],
        preregistration_id: str,
        preregistration: Mapping[str, Any],
        adaptive_freeze: Mapping[str, Any],
        allowed_pending_paths: set[str] | None = None,
    ) -> dict[str, Any]:
        """Prove every post-freeze path is a protocol-bound immutable attempt.

        Merely checking that ``HEAD`` descends from the frozen commit is unsafe:
        a descendant could change the hypothesis, method, code, memory, protocol,
        or evaluator.  This verifier walks the first-parent checkpoint chain and
        accepts only the exact preregistration transition followed by newly added
        confirmatory-attempt objects bound to this registration.
        """

        current_state = self.repository.status()
        backend_staged = set(current_state.get("staged_paths") or [])
        research_stage = current_state.get("research_stage")
        research_stage = research_stage if isinstance(research_stage, Mapping) else {}
        native_staged = set(research_stage.get("paths") or [])
        if backend_staged or native_staged:
            raise ResearchGitError(
                "confirmatory experiment requires the exact clean Research VCS "
                "lineage frozen at preregistration"
            )
        pending_paths = {
            *list(current_state.get("tracked_changes") or []),
            *list(current_state.get("eligible_changes") or []),
        }
        if allowed_pending_paths is not None:
            allowed_pending = {str(path) for path in allowed_pending_paths}
            if allowed_pending:
                explicitly_allowed_untracked = _run_git(
                    self.repository.path,
                    [
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                        "-z",
                        "--",
                        *sorted(allowed_pending),
                    ],
                ).stdout.split("\0")
                pending_paths.update(
                    path for path in explicitly_allowed_untracked if path
                )
            if pending_paths != allowed_pending:
                raise ResearchGitError(
                    "confirmatory experiment requires the exact clean Research VCS "
                    "lineage frozen at preregistration"
                )
        elif current_state.get("worktree_clean") is not True:
            raise ResearchGitError(
                "confirmatory experiment requires the exact clean Research VCS "
                "lineage frozen at preregistration"
            )
        frozen_head = str(adaptive_freeze.get("research_vcs_head") or "").strip()
        current_head = str(current_state.get("head") or "").strip()
        if not frozen_head or not current_head:
            raise ResearchGitError(
                "confirmatory experiment requires a committed Research VCS freeze"
            )

        registration_object_id = str(
            registration.get("object_id") or preregistration_id or ""
        ).strip()
        plan_ids = []
        for target in sorted(
            _relation_targets(registration, relation_type="depends_on")
        ):
            try:
                dependency = self.repository.get(target)
            except ResearchGitError:
                continue
            if dependency.get("kind") == "research_plan":
                plan_ids.append(str(dependency.get("object_id") or ""))
        if not registration_object_id or len(plan_ids) != 1:
            raise ResearchGitError(
                "confirmatory experiment requires one committed locked research plan"
            )
        plan_object_id = plan_ids[0]
        try:
            registration_origin = (
                self.repository.blame(
                    registration_object_id,
                    commit=current_head,
                ).get("origin")
                or {}
            )
            plan_origin = (
                self.repository.blame(
                    plan_object_id,
                    commit=current_head,
                ).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory experiment requires a committed preregistration "
                "transition"
            ) from exc
        registration_commit = str(registration_origin.get("commit") or "").strip()
        if (
            not registration_commit
            or str(plan_origin.get("commit") or "").strip() != registration_commit
        ):
            raise ResearchGitError(
                "confirmatory experiment requires the plan and preregistration in "
                "one committed transition"
            )

        expected_registration_paths = {
            ".xscientist/objects/preregistration/" f"{registration_object_id}.json",
            ".xscientist/objects/research_plan/" f"{plan_object_id}.json",
        }
        campaign = preregistration.get("confirmatory_campaign")
        campaign = campaign if isinstance(campaign, Mapping) else {}
        expected_campaign_paths: set[str] = set()
        allowed_registry_paths: set[str] = set()
        if campaign:
            if campaign.get("schema") != "xscientist.confirmatory-campaign.v1":
                raise ResearchGitError("confirmatory campaign schema is invalid")
            queue_contract = campaign.get("queue_contract")
            queue_contract = (
                queue_contract if isinstance(queue_contract, Mapping) else {}
            )
            queue_contract_core = {
                key: value
                for key, value in queue_contract.items()
                if key != "queue_contract_hash"
            }
            queue_contract_hash = canonical_content_hash(queue_contract_core)
            if (
                queue_contract.get("schema")
                != "xscientist.confirmatory-queue-contract.v1"
                or queue_contract.get("queue_contract_hash") != queue_contract_hash
                or campaign.get("queue_contract_hash") != queue_contract_hash
            ):
                raise ResearchGitError(
                    "confirmatory campaign queue contract is missing or hash-invalid"
                )
            expected_campaign_paths = {
                _safe_campaign_path(campaign.get("preregistration_path")),
                _safe_campaign_path(campaign.get("queue_path")),
            }
            if len(expected_campaign_paths) != 2:
                raise ResearchGitError(
                    "confirmatory campaign requires distinct registration and queue "
                    "artifacts"
                )
            paper_prefix = PurePosixPath(
                _safe_campaign_path(campaign.get("preregistration_path"))
            ).parent
            allowed_registry_paths = {
                (paper_prefix / filename).as_posix()
                for filename in (
                    "experiment_registry.jsonl",
                    "experiment_registry.integrity.json",
                    "experiment_registry.history.jsonl",
                    "pipeline_manifest.json",
                    "stage_standards.json",
                    "process_alignment.json",
                )
            }
        cursor = current_head
        visited: set[str] = set()
        checked_paths: list[dict[str, str]] = []
        prior_attempt_ids: list[str] = []
        prior_gate_decision_ids: list[str] = []
        registration_seen = False
        campaign_seen = not expected_campaign_paths
        for _ in range(_MAX_CONFIRMATORY_LINEAGE_COMMITS):
            if cursor == frozen_head:
                break
            if not cursor or cursor in visited:
                raise ResearchGitError(
                    "confirmatory experiment Research VCS lineage is cyclic or broken"
                )
            visited.add(cursor)
            try:
                shown = self.repository.show(cursor)
            except ResearchGitError as exc:
                raise ResearchGitError(
                    "confirmatory experiment Research VCS lineage is not fully "
                    "checkpoint-attested"
                ) from exc
            if shown.get("checkpoint_hash_valid") is not True:
                raise ResearchGitError(
                    "confirmatory experiment Research VCS lineage has an invalid "
                    "checkpoint hash"
                )
            checkpoint = shown.get("checkpoint")
            checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
            changed_paths = {
                str(path) for path in checkpoint.get("changed_paths") or []
            }
            parent = str(checkpoint.get("parent_commit") or "").strip()
            if cursor == registration_commit:
                if (
                    registration_seen
                    or checkpoint.get("stage") != "preregister"
                    or parent != frozen_head
                    or changed_paths != expected_registration_paths
                ):
                    raise ResearchGitError(
                        "confirmatory experiment requires the exact clean Research "
                        "VCS preregistration transition"
                    )
                registration_seen = True
                checked_paths.extend(
                    {
                        "commit": cursor,
                        "path": path,
                        "classification": "frozen_contract",
                    }
                    for path in sorted(changed_paths)
                )
            else:
                if expected_campaign_paths and checkpoint.get("stage") == "preregister":
                    if (
                        campaign_seen
                        or parent != registration_commit
                        or changed_paths != expected_campaign_paths
                    ):
                        raise ResearchGitError(
                            "confirmatory campaign artifacts are not the exact "
                            "post-registration materialization transition"
                        )
                    preregistration_path = self.repository.path / _safe_campaign_path(
                        campaign.get("preregistration_path")
                    )
                    queue_path = self.repository.path / _safe_campaign_path(
                        campaign.get("queue_path")
                    )
                    try:
                        if (
                            preregistration_path.is_symlink()
                            or queue_path.is_symlink()
                            or not preregistration_path.is_file()
                            or not queue_path.is_file()
                            or preregistration_path.stat().st_size
                            > _MAX_CONFIRMATORY_CAMPAIGN_BYTES
                            or queue_path.stat().st_size
                            > _MAX_CONFIRMATORY_CAMPAIGN_BYTES
                        ):
                            raise ValueError("campaign artifact boundary violation")
                        saved_registration = json.loads(
                            preregistration_path.read_text(encoding="utf-8")
                        )
                        saved_queue = json.loads(queue_path.read_text(encoding="utf-8"))
                    except (
                        OSError,
                        UnicodeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise ResearchGitError(
                            "confirmatory campaign artifacts are unreadable or unsafe"
                        ) from exc
                    if saved_registration != preregistration:
                        raise ResearchGitError(
                            "confirmatory campaign registration artifact differs from "
                            "the locked Research VCS object"
                        )
                    if not isinstance(saved_queue, Mapping):
                        raise ResearchGitError(
                            "confirmatory campaign queue must be a JSON object"
                        )
                    queue_core = {
                        key: value
                        for key, value in saved_queue.items()
                        if key != "queue_hash"
                    }
                    queue_valid = bool(
                        saved_queue.get("schema") == "xscientist.confirmatory-queue.v1"
                        and saved_queue.get("preregistration_object_id")
                        == registration_object_id
                        and saved_queue.get("plan_object_id") == plan_object_id
                        and saved_queue.get("registration_hash")
                        == preregistration.get("registration_hash")
                        and saved_queue.get("frozen_state_hash")
                        == adaptive_freeze.get("state_hash")
                        and saved_queue.get("data_manifest_hash")
                        == (preregistration.get("data_policy") or {}).get(
                            "data_manifest_hash"
                        )
                        and saved_queue.get("data_snapshot_id")
                        == (preregistration.get("data_policy") or {}).get(
                            "data_snapshot_id"
                        )
                        and saved_queue.get("queue_hash")
                        == canonical_content_hash(queue_core)
                    )
                    queued_task_ids = {
                        str(item.get("task_id") or "").strip()
                        for item in saved_queue.get("tasks") or []
                        if isinstance(item, Mapping)
                    }
                    queue_tasks = {
                        str(item.get("task_id") or "").strip(): item
                        for item in saved_queue.get("tasks") or []
                        if isinstance(item, Mapping)
                        and str(item.get("task_id") or "").strip()
                    }
                    data_policy = preregistration.get("data_policy")
                    data_policy = (
                        data_policy if isinstance(data_policy, Mapping) else {}
                    )
                    outcome_contracts_valid = True
                    for outcome in preregistration.get("outcomes") or []:
                        if not isinstance(outcome, Mapping):
                            outcome_contracts_valid = False
                            continue
                        task_id = str(outcome.get("task_id") or "").strip()
                        queued = queue_tasks.get(task_id) or {}
                        expected_fields = {
                            "dataset": outcome.get("dataset"),
                            "metric": outcome.get("metric"),
                            "baseline": outcome.get("baseline"),
                            "evidence_role": outcome.get("evidence_role"),
                            "paired_control_task_id": outcome.get(
                                "paired_control_task_id"
                            ),
                            "intervention_variant": outcome.get("intervention_variant"),
                            "stress_condition": outcome.get("stress_condition"),
                            "transformation_contract": outcome.get(
                                "transformation_contract"
                            ),
                            "transformation_contract_hash": outcome.get(
                                "transformation_contract_hash"
                            ),
                            "split_hash": (data_policy.get("split_hashes") or {}).get(
                                task_id
                            ),
                            "data_manifest_hash": data_policy.get("data_manifest_hash"),
                            "data_snapshot_id": data_policy.get("data_snapshot_id"),
                        }
                        if any(
                            queued.get(field) != expected
                            for field, expected in expected_fields.items()
                        ):
                            outcome_contracts_valid = False
                    contract_tasks = {
                        str(item.get("task_id") or "").strip(): item
                        for item in queue_contract.get("tasks") or []
                        if isinstance(item, Mapping)
                        and str(item.get("task_id") or "").strip()
                    }
                    materialized_contract_valid = set(contract_tasks) == set(
                        queue_tasks
                    )
                    for task_id, contract_task in contract_tasks.items():
                        queued = queue_tasks.get(task_id) or {}
                        contract_core = {
                            key: value
                            for key, value in contract_task.items()
                            if key != "task_contract_hash"
                        }
                        expected_prefix = [
                            "xscientist",
                            "research",
                            "experiment",
                            "{RESULT_SUMMARY}",
                            "--study-phase",
                            "confirmatory",
                            "--task",
                            task_id,
                            "--plan",
                            plan_object_id,
                            "--preregistration",
                            registration_object_id,
                            "--repo",
                            ".",
                        ]
                        if (
                            contract_task.get("task_contract_hash")
                            != canonical_content_hash(contract_core)
                            or any(
                                queued.get(field) != value
                                for field, value in contract_task.items()
                            )
                            or queued.get("record_command_prefix") != expected_prefix
                            or queued.get("bound_object_ids")
                            != {
                                "plan": plan_object_id,
                                "preregistration": registration_object_id,
                            }
                            or queued.get("status") != "queued"
                            or queued.get("study_phase") != "confirmatory"
                            or queued.get("adaptive_state_hash")
                            != adaptive_freeze.get("state_hash")
                            or queued.get("research_state_hash")
                            != adaptive_freeze.get("research_state_hash")
                            or queued.get("post_freeze_adaptation") is not False
                        ):
                            materialized_contract_valid = False
                    if (
                        not queue_valid
                        or saved_queue.get("queue_contract_hash") != queue_contract_hash
                        or queued_task_ids != _registered_task_ids(preregistration)
                        or len(queue_tasks)
                        != len(_registered_task_ids(preregistration))
                        or not outcome_contracts_valid
                        or not materialized_contract_valid
                    ):
                        raise ResearchGitError(
                            "confirmatory campaign queue is not bound to the locked "
                            "plan, data, and adaptive state"
                        )
                    campaign_seen = True
                    checked_paths.extend(
                        {
                            "commit": cursor,
                            "path": path,
                            "classification": "derived_campaign_artifact",
                        }
                        for path in sorted(changed_paths)
                    )
                    cursor = parent
                    continue
                if registration_seen or checkpoint.get("stage") not in {
                    "experiment",
                    "failed",
                    "evidence",
                    "review",
                }:
                    raise ResearchGitError(
                        "confirmatory experiment requires the exact clean Research "
                        "VCS lineage; found a non-confirmatory post-freeze transition"
                    )
                if not changed_paths:
                    raise ResearchGitError(
                        "confirmatory experiment lineage contains an empty transition"
                    )
                registry_artifact_paths = changed_paths & allowed_registry_paths
                pointer_paths = {
                    path
                    for path in changed_paths
                    if path.startswith("research-objects/") and path.endswith(".json")
                }
                object_paths = changed_paths - registry_artifact_paths - pointer_paths
                if registry_artifact_paths and checkpoint.get("stage") != "review":
                    raise ResearchGitError(
                        "confirmatory registry artifacts must be admitted with a "
                        "trajectory-binding review transition"
                    )
                if registry_artifact_paths and not object_paths:
                    raise ResearchGitError(
                        "confirmatory registry artifact transition has no binding object"
                    )
                declared_pointer_bindings: dict[str, str] = {}
                for path in sorted(object_paths):
                    match = _RESEARCH_OBJECT_PATH.fullmatch(path)
                    if (
                        match is not None
                        and match.group("kind") == "experiment_attempt"
                    ):
                        object_id = self._validate_confirmatory_attempt_path(
                            path,
                            commit=cursor,
                            current_head=current_head,
                            preregistration_id=registration_object_id,
                            preregistration=preregistration,
                            adaptive_freeze=adaptive_freeze,
                        )
                        prior_attempt_ids.append(object_id)
                        classification = "confirmatory_attempt"
                        attempt = self.repository.get(object_id)
                        attempt_payload = attempt.get("payload") or {}
                        result_artifacts = attempt_payload.get("result_artifacts") or {}
                        if isinstance(result_artifacts, Mapping):
                            for item in result_artifacts.values():
                                if not isinstance(item, Mapping):
                                    continue
                                pointer_path = str(item.get("pointer_path") or "")
                                object_hash = str(item.get("content_hash") or "")
                                if pointer_path and object_hash:
                                    declared_pointer_bindings[pointer_path] = (
                                        object_hash
                                    )
                    elif match is not None and match.group("kind") == "evidence":
                        object_id = self._validate_confirmatory_negative_evidence_path(
                            path,
                            commit=cursor,
                            current_head=current_head,
                            preregistration_id=registration_object_id,
                        )
                        classification = "terminal_negative_evidence"
                    else:
                        object_id, protocol_kind = (
                            self._validate_confirmatory_gate_path(
                                path,
                                commit=cursor,
                                current_head=current_head,
                                preregistration_id=registration_object_id,
                            )
                        )
                        prior_gate_decision_ids.append(object_id)
                        classification = protocol_kind
                    checked_paths.append(
                        {
                            "commit": cursor,
                            "path": path,
                            "classification": classification,
                        }
                    )
                unexpected_pointer_paths = pointer_paths - set(
                    declared_pointer_bindings
                )
                if unexpected_pointer_paths:
                    raise ResearchGitError(
                        "confirmatory attempt transition contains an unbound research "
                        "object pointer: " + ", ".join(sorted(unexpected_pointer_paths))
                    )
                committed_pointers = _pointer_records_at_commit(
                    self.repository.path,
                    cursor,
                    strict=True,
                )
                committed_pointer_hashes = {
                    str(pointer["pointer_path"]): object_hash
                    for object_hash, pointers in committed_pointers.items()
                    for pointer in pointers
                }
                mismatched_pointer_paths = {
                    path
                    for path, declared_hash in declared_pointer_bindings.items()
                    if committed_pointer_hashes.get(path) != declared_hash
                }
                if mismatched_pointer_paths:
                    raise ResearchGitError(
                        "confirmatory result pointer does not match its attempt: "
                        + ", ".join(sorted(mismatched_pointer_paths))
                    )
                checked_paths.extend(
                    {
                        "commit": cursor,
                        "path": path,
                        "classification": "confirmatory_result_pointer",
                    }
                    for path in sorted(pointer_paths)
                )
                checked_paths.extend(
                    {
                        "commit": cursor,
                        "path": path,
                        "classification": "derived_registry_artifact",
                    }
                    for path in sorted(registry_artifact_paths)
                )
            cursor = parent
        else:
            raise ResearchGitError(
                "confirmatory experiment lineage exceeds the bounded audit limit"
            )
        if cursor != frozen_head or not registration_seen or not campaign_seen:
            raise ResearchGitError(
                "confirmatory experiment requires the exact clean Research VCS "
                "lineage frozen at preregistration"
            )

        component_fields = {
            "hypothesis": "hypothesis_hash",
            "method": "method_hash",
            "code": "code_state_hash",
            "memory": "memory_state_hash",
            "protocol": "protocol_hash",
            "evaluator": "evaluator_spec_hash",
        }
        return {
            "schema": "xscientist.confirmatory-lineage-attestation.v1",
            "status": "verified",
            "frozen_head": frozen_head,
            "lineage_head": current_head,
            "registration_commit": registration_commit,
            "registration_object_id": registration_object_id,
            "plan_object_id": plan_object_id,
            "prior_confirmatory_attempt_ids": list(reversed(prior_attempt_ids)),
            "prior_gate_decision_ids": list(reversed(prior_gate_decision_ids)),
            "checked_paths": list(reversed(checked_paths)),
            "path_policy": (
                "exact_preregistration_transition_then_new_protocol_bound_"
                "confirmatory_attempt_objects_only"
            ),
            "frozen_components": {
                component: {
                    "hash": adaptive_freeze.get(field),
                    "unchanged": True,
                    "proof": "no_post_freeze_path_can_mutate_frozen_tree",
                }
                for component, field in component_fields.items()
            },
        }

    def _experiment_admission_required(self) -> bool:
        return self.repository.experiment_admission_mode() == "required"

    @staticmethod
    def _require_committed_admission_object(
        research_object: Mapping[str, Any],
        *,
        committed_objects: Mapping[str, Mapping[str, Any]],
        label: str,
    ) -> Mapping[str, Any]:
        object_id = str(research_object.get("object_id") or "")
        committed = committed_objects.get(object_id)
        if committed is None:
            raise ResearchGitError(
                f"experiment admission {label} must be committed at HEAD before use"
            )
        if committed.get("kind") != research_object.get("kind") or committed.get(
            "content_hash"
        ) != research_object.get("content_hash"):
            raise ResearchGitError(
                f"experiment admission {label} does not match its committed HEAD object"
            )
        return committed

    def _plan_hypothesis(
        self,
        plan: Mapping[str, Any],
        *,
        committed_objects: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        hypothesis_ids: set[str] = set()
        for target in _relation_targets(plan, relation_type="depends_on"):
            dependency = committed_objects.get(target)
            if dependency is None:
                continue
            if dependency.get("kind") == "hypothesis":
                hypothesis_ids.add(str(dependency["object_id"]))
        if len(hypothesis_ids) != 1:
            raise ResearchGitError(
                "experiment admission requires a committed plan bound to exactly "
                "one committed hypothesis"
            )
        return committed_objects[next(iter(hypothesis_ids))]

    def _experiment_admission_trust_store_path(self) -> str:
        selected = self._admission_trust_store or os.environ.get(
            XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE
        )
        if not isinstance(selected, str) or not selected.strip():
            raise ResearchGitError(
                "experiment admission consumption requires the current external "
                f"trust store via {XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE} "
                "or ResearchLifecycle(admission_trust_store=...)"
            )
        return selected

    def _experiment_admission_introduction_checkpoint(
        self,
        gate: Mapping[str, Any],
        *,
        ref: str = "HEAD",
    ) -> str:
        object_id = str(gate.get("object_id") or "")
        path = f".xscientist/objects/gate_decision/{object_id}.json"
        history = _run_git(
            self.repository.path,
            ["rev-list", "--first-parent", "--reverse", ref, "--", path],
        ).stdout.splitlines()
        if len(history) != 1 or re.fullmatch(r"[0-9a-f]{40,64}", history[0]) is None:
            raise ResearchGitError(
                "experiment admission gate must enter through exactly one dedicated "
                "checkpoint"
            )
        introduction = history[0]
        checkpoint = show_checkpoint(self.repository.path, introduction).get(
            "checkpoint"
        )
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        payload = gate.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if (
            checkpoint.get("stage") != "experiment_admission"
            or checkpoint.get("status") != "verified"
            or checkpoint.get("parent_commit") != payload.get("repository_head")
            or checkpoint.get("changed_paths") != [path]
        ):
            raise ResearchGitError(
                "experiment admission gate introduction checkpoint is invalid"
            )
        return introduction

    def _require_experiment_admission_high_water_visible(
        self,
        high_water: Mapping[str, Any] | None,
        *,
        committed_objects: Mapping[str, Mapping[str, Any]],
        signed_head: str,
    ) -> Mapping[str, Any] | None:
        """Require an earlier host decision to be part of the signed history."""

        if high_water is None:
            return None
        gate_id = str(high_water.get("gate_object_id") or "")
        gate = committed_objects.get(gate_id)
        if (
            gate is None
            or gate.get("content_hash") != high_water.get("gate_object_hash")
            or not self._admission_gate_payload_valid(gate)
        ):
            raise ResearchGitError(
                "experiment admission signed HEAD does not include the latest host "
                "high-water decision"
            )
        current = load_experiment_admission_high_water(
            self.repository.path,
            gate.get("payload") or {},
        )
        if not _canonical_values_equal(current, high_water):
            raise ResearchGitError(
                "experiment admission high-water decision changed during validation"
            )
        introduction = self._experiment_admission_introduction_checkpoint(
            gate,
            ref=signed_head,
        )
        if introduction != high_water.get("checkpoint_commit"):
            raise ResearchGitError(
                "experiment admission high-water checkpoint does not match its "
                "dedicated introduction"
            )
        ancestry = _run_git(
            self.repository.path,
            ["merge-base", "--is-ancestor", introduction, signed_head],
            check=False,
        )
        if ancestry.returncode != 0:
            raise ResearchGitError(
                "experiment admission signed HEAD does not descend from the latest "
                "host high-water decision"
            )
        return gate

    def _verify_experiment_admission_gate_authority(
        self,
        gate: Mapping[str, Any],
        *,
        committed_objects: Mapping[str, Mapping[str, Any]],
        plan: Mapping[str, Any],
        hypothesis: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None,
        trust_store_path: str | Path | None = None,
        require_high_water: bool = True,
        ref: str = "HEAD",
    ) -> dict[str, Any]:
        actor = gate.get("actor")
        if actor != EXPERIMENT_ADMISSION_ACTOR:
            raise ResearchGitError(
                "experiment admission gate actor is not the host-owned verifier"
            )
        target_ids = [str(plan["object_id"]), str(hypothesis["object_id"])]
        if preregistration is not None:
            target_ids.append(str(preregistration["object_id"]))
        closure = self._experiment_admission_producer_closure(
            committed_objects,
            target_ids=target_ids,
        )
        verification = verify_persisted_experiment_admission_gate(
            gate.get("payload") or {},
            trust_store_path=(
                trust_store_path
                if trust_store_path is not None
                else self._experiment_admission_trust_store_path()
            ),
            forbidden_roots=_repository_admission_forbidden_roots(self.repository.path),
            expected_producer_actor_ids=closure["producer_actor_ids"],
            expected_producer_principals=closure["producer_principals"],
            expected_lineage_object_ids=closure["lineage_object_ids"],
        )
        introduction = self._experiment_admission_introduction_checkpoint(
            gate,
            ref=ref,
        )
        high_water = (
            require_current_experiment_admission_decision(
                self.repository.path,
                gate,
            )
            if require_high_water
            else None
        )
        if (
            high_water is not None
            and high_water.get("checkpoint_commit") != introduction
        ):
            raise ResearchGitError(
                "experiment admission high-water checkpoint does not match its "
                "dedicated introduction"
            )
        return {
            **verification,
            "introduction_checkpoint": introduction,
            "high_water": high_water,
        }

    @staticmethod
    def _admission_gate_payload_valid(gate: Mapping[str, Any]) -> bool:
        payload = gate.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        actor = gate.get("actor")
        actor = actor if isinstance(actor, Mapping) else {}
        decision = payload.get("decision")
        authorized = payload.get("experiment_authorized")
        scope = payload.get("scope")
        if not isinstance(scope, Mapping):
            return False
        try:
            validate_experiment_admission_scope(scope)
        except ResearchGitError:
            return False
        expected_fields = {
            "protocol_kind",
            "gate_id",
            "decision",
            "experiment_authorized",
            "claim_promotion_allowed",
            "repository_id",
            "repository_head",
            "input_bindings",
            "scope",
            "scope_hash",
            "admission_request_hash",
            "authority_receipt",
            "gate_hash",
        }
        gate_core = {key: value for key, value in payload.items() if key != "gate_hash"}
        return bool(
            gate.get("kind") == "gate_decision"
            and gate.get("state") in _EXPERIMENT_ADMISSION_STATES
            and actor == EXPERIMENT_ADMISSION_ACTOR
            and set(payload) == expected_fields
            and payload.get("protocol_kind") == _EXPERIMENT_ADMISSION_PROTOCOL
            and decision in {"authorize", "deny"}
            and isinstance(authorized, bool)
            and authorized is (decision == "authorize")
            and payload.get("claim_promotion_allowed") is False
            and payload.get("scope_hash") == canonical_content_hash(dict(scope))
            and payload.get("gate_hash") == canonical_content_hash(gate_core)
        )

    @staticmethod
    def _admission_binding_matches(
        binding: Any,
        research_object: Mapping[str, Any],
    ) -> bool:
        return bool(
            isinstance(binding, Mapping)
            and binding.get("object_id") == research_object.get("object_id")
            and binding.get("content_hash") == research_object.get("content_hash")
        )

    @staticmethod
    def _admission_scope_matches(
        scope_value: Any,
        payload: Mapping[str, Any],
    ) -> bool:
        scope = dict(scope_value) if isinstance(scope_value, Mapping) else {}
        actual_scope = {
            field: payload[field]
            for field in _EXPERIMENT_ADMISSION_SCOPE_FIELDS
            if field in payload
        }
        # Runtime callers historically pass concrete seed tuples, while the
        # signed JSON scope is intentionally list-only. Normalize that legacy
        # runtime representation before applying the strict scope validator.
        if isinstance(actual_scope.get("seeds"), tuple):
            actual_scope["seeds"] = list(actual_scope["seeds"])
        try:
            validate_experiment_admission_scope(scope)
            _validate_admission_runtime_scope(actual_scope)
        except ResearchGitError:
            return False
        for candidate, signed in ((scope, True), (actual_scope, False)):
            if "idea_index" in candidate and (
                type(candidate.get("idea_index")) is not int
                or candidate["idea_index"] < 0
            ):
                return False
            for field in ("allowed_seeds", "seeds"):
                if field not in candidate:
                    continue
                seeds = candidate.get(field)
                expected_container = list if signed else (list, tuple)
                if (
                    not isinstance(seeds, expected_container)
                    or not seeds
                    or any(
                        isinstance(seed, bool) or not isinstance(seed, int)
                        for seed in seeds
                    )
                    or len(set(seeds)) != len(seeds)
                ):
                    return False
        allowed_seeds = scope.pop("allowed_seeds", None)
        if allowed_seeds is None:
            return _canonical_values_equal(scope, actual_scope)
        if (
            "allowed_seeds" in actual_scope
            or "seeds" in scope
            or not isinstance(allowed_seeds, list)
            or not allowed_seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in allowed_seeds
            )
        ):
            return False
        supplied_seeds = actual_scope.pop("seeds", None)
        return bool(
            isinstance(supplied_seeds, (list, tuple))
            and supplied_seeds
            and all(
                not isinstance(seed, bool) and isinstance(seed, int)
                for seed in supplied_seeds
            )
            and set(supplied_seeds) <= set(allowed_seeds)
            and _canonical_values_equal(scope, actual_scope)
        )

    @staticmethod
    def _admission_scope_overlaps(
        scope_value: Any,
        payload: Mapping[str, Any],
    ) -> bool:
        """Return whether a signed denial intersects one concrete execution scope."""

        scope = dict(scope_value) if isinstance(scope_value, Mapping) else {}
        actual_scope = {
            field: payload[field]
            for field in _EXPERIMENT_ADMISSION_SCOPE_FIELDS
            if field in payload
        }
        if isinstance(actual_scope.get("seeds"), tuple):
            actual_scope["seeds"] = list(actual_scope["seeds"])
        try:
            validate_experiment_admission_scope(scope)
            _validate_admission_runtime_scope(actual_scope)
        except ResearchGitError:
            return False
        for candidate, signed in ((scope, True), (actual_scope, False)):
            if "idea_index" in candidate and (
                type(candidate.get("idea_index")) is not int
                or candidate["idea_index"] < 0
            ):
                return False
            for field in ("allowed_seeds", "seeds"):
                if field not in candidate:
                    continue
                seeds = candidate.get(field)
                expected_container = list if signed else (list, tuple)
                if (
                    not isinstance(seeds, expected_container)
                    or not seeds
                    or any(
                        isinstance(seed, bool) or not isinstance(seed, int)
                        for seed in seeds
                    )
                    or len(set(seeds)) != len(seeds)
                ):
                    return False
        if "allowed_seeds" in actual_scope:
            return False
        denied_seeds = scope.pop("allowed_seeds", None)
        if denied_seeds is None:
            denied_seeds = scope.pop("seeds", None)
        if denied_seeds is None:
            return _canonical_values_equal(scope, actual_scope)
        actual_seeds = actual_scope.pop("seeds", None)
        return bool(
            isinstance(actual_seeds, (list, tuple))
            and set(denied_seeds).intersection(actual_seeds)
            and _canonical_values_equal(scope, actual_scope)
        )

    def _admission_gate_bindings_match(
        self,
        gate: Mapping[str, Any],
        *,
        plan: Mapping[str, Any],
        hypothesis: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None,
    ) -> bool:
        if not self._admission_gate_payload_valid(gate):
            return False
        gate_payload = gate.get("payload") or {}
        bindings = gate_payload.get("input_bindings")
        if not isinstance(bindings, Mapping):
            return False
        if not self._admission_binding_matches(bindings.get("plan"), plan):
            return False
        if not self._admission_binding_matches(bindings.get("hypothesis"), hypothesis):
            return False
        if _relation_targets(gate, relation_type="depends_on", role="plan") != {
            str(plan["object_id"])
        }:
            return False
        if _relation_targets(gate, relation_type="evaluates") != {
            str(hypothesis["object_id"])
        }:
            return False
        if preregistration is not None:
            if not self._admission_binding_matches(
                bindings.get("preregistration"), preregistration
            ):
                return False
            if _relation_targets(gate, relation_type="depends_on", role="protocol") != {
                str(preregistration["object_id"])
            }:
                return False
        elif bindings.get("preregistration") is not None or _relation_targets(
            gate, relation_type="depends_on", role="protocol"
        ):
            return False
        return True

    def _admission_gate_matches(
        self,
        gate: Mapping[str, Any],
        *,
        plan: Mapping[str, Any],
        hypothesis: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None,
        payload: Mapping[str, Any],
    ) -> bool:
        if not self._admission_gate_bindings_match(
            gate,
            plan=plan,
            hypothesis=hypothesis,
            preregistration=preregistration,
        ):
            return False
        gate_payload = gate.get("payload") or {}
        return self._admission_scope_matches(gate_payload.get("scope"), payload)

    def _admission_denial_overlaps(
        self,
        gate: Mapping[str, Any],
        *,
        plan: Mapping[str, Any],
        hypothesis: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None,
        payload: Mapping[str, Any],
    ) -> bool:
        if not self._admission_gate_bindings_match(
            gate,
            plan=plan,
            hypothesis=hypothesis,
            preregistration=preregistration,
        ):
            return False
        gate_payload = gate.get("payload") or {}
        return self._admission_scope_overlaps(gate_payload.get("scope"), payload)

    def _experiment_admission_context_at_head(
        self,
        *,
        head: str,
        plan_id: str,
        preregistration_id: str | None,
    ) -> dict[str, Any]:
        committed_objects = {
            str(item["object_id"]): item
            for item in list_research_objects_at_ref(self.repository.path, head)
        }

        def require_object(
            object_id: str, *, label: str, kinds: set[str]
        ) -> Mapping[str, Any]:
            normalized_id = str(object_id or "").strip()
            item = committed_objects.get(normalized_id)
            if item is None:
                raise ResearchGitError(
                    f"experiment admission {label} must be committed at the signed HEAD"
                )
            if item.get("kind") not in kinds:
                raise ResearchGitError(
                    f"experiment admission {label} reference has wrong kind"
                )
            return item

        plan = require_object(
            plan_id,
            label="plan",
            kinds={"research_plan", "experiment_design"},
        )
        hypothesis = self._plan_hypothesis(plan, committed_objects=committed_objects)
        preregistration = (
            require_object(
                preregistration_id,
                label="preregistration",
                kinds={"preregistration"},
            )
            if preregistration_id
            else None
        )
        repository_config = load_repository_config_at_ref(self.repository.path, head)
        return {
            "repository_id": str(repository_config.get("repository_id") or ""),
            "objects": committed_objects,
            "plan": plan,
            "hypothesis": hypothesis,
            "preregistration": preregistration,
        }

    @staticmethod
    def _experiment_admission_producer_closure(
        committed_objects: Mapping[str, Mapping[str, Any]],
        *,
        target_ids: Sequence[str],
    ) -> dict[str, list[str]]:
        pending = [str(value) for value in target_ids]
        visited: set[str] = set()
        actor_ids: set[str] = set()
        while pending:
            object_id = pending.pop()
            if object_id in visited:
                continue
            item = committed_objects.get(object_id)
            if item is None:
                raise ResearchGitError(
                    "experiment admission producer lineage is incomplete at the signed HEAD"
                )
            visited.add(object_id)
            actor = item.get("actor")
            actor = actor if isinstance(actor, Mapping) else {}
            payload = item.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            actor_id = " ".join(str(actor.get("actor_id") or "").split())
            producer_id = " ".join(str(payload.get("producer_id") or "").split())
            try:
                if (
                    actor_id
                    and producer_id
                    and canonical_principal(
                        actor_id, label="experiment admission producer actor"
                    )
                    != canonical_principal(
                        producer_id, label="experiment admission payload producer"
                    )
                ):
                    raise ResearchGitError(
                        "experiment admission producer actor does not match payload producer_id"
                    )
                if actor_id:
                    canonical_principal(
                        actor_id, label="experiment admission producer actor"
                    )
                if producer_id:
                    canonical_principal(
                        producer_id, label="experiment admission payload producer"
                    )
            except ValueError as exc:
                raise ResearchGitError(
                    "experiment admission producer provenance contains an invalid identity"
                ) from exc
            if actor_id:
                actor_ids.add(actor_id)
            if producer_id:
                actor_ids.add(producer_id)
            for relation in item.get("relations") or []:
                if not isinstance(relation, Mapping):
                    continue
                if relation.get("type") not in PRODUCER_LINEAGE_RELATIONS:
                    continue
                target = str(relation.get("target") or "").strip()
                if target and target not in visited:
                    pending.append(target)
        try:
            principals = sorted(
                {
                    canonical_principal(
                        value, label="experiment admission producer identity"
                    )
                    for value in actor_ids
                }
            )
        except ValueError as exc:  # pragma: no cover - checked while traversing
            raise ResearchGitError(
                "experiment admission producer provenance contains an invalid identity"
            ) from exc
        return {
            "producer_actor_ids": sorted(actor_ids),
            "producer_principals": principals,
            "lineage_object_ids": sorted(visited),
        }

    def _exact_experiment_admission_gates(
        self,
        committed_objects: Mapping[str, Mapping[str, Any]],
        *,
        plan: Mapping[str, Any],
        hypothesis: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None,
        scope: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        exact: list[Mapping[str, Any]] = []
        expected_bindings = {
            "hypothesis": {
                "object_id": hypothesis["object_id"],
                "content_hash": hypothesis["content_hash"],
            },
            "plan": {
                "object_id": plan["object_id"],
                "content_hash": plan["content_hash"],
            },
            "preregistration": (
                {
                    "object_id": preregistration["object_id"],
                    "content_hash": preregistration["content_hash"],
                }
                if preregistration is not None
                else None
            ),
        }
        for item in committed_objects.values():
            if not self._admission_gate_payload_valid(item):
                continue
            payload = item.get("payload") or {}
            if not _canonical_values_equal(
                payload.get("input_bindings"), expected_bindings
            ) or not _canonical_values_equal(payload.get("scope"), dict(scope)):
                continue
            if _relation_targets(item, relation_type="evaluates") != {
                str(hypothesis["object_id"])
            }:
                continue
            if _relation_targets(item, relation_type="depends_on", role="plan") != {
                str(plan["object_id"])
            }:
                continue
            protocol_targets = _relation_targets(
                item, relation_type="depends_on", role="protocol"
            )
            expected_protocol_targets = (
                {str(preregistration["object_id"])}
                if preregistration is not None
                else set()
            )
            if protocol_targets != expected_protocol_targets:
                continue
            exact.append(item)
        return exact

    def _latest_committed_experiment_admission_gate(
        self,
        committed_objects: Mapping[str, Mapping[str, Any]],
        *,
        plan: Mapping[str, Any],
        hypothesis: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None,
        scope: Mapping[str, Any],
        ref: str = "HEAD",
    ) -> tuple[Mapping[str, Any], str]:
        """Return the newest dedicated gate for one exact signed subject."""

        exact = self._exact_experiment_admission_gates(
            committed_objects,
            plan=plan,
            hypothesis=hypothesis,
            preregistration=preregistration,
            scope=scope,
        )
        if not exact:
            raise ResearchGitError("experiment admission subject has no committed gate")
        history = _run_git(
            self.repository.path,
            ["rev-list", "--first-parent", "--reverse", ref],
        ).stdout.splitlines()
        positions = {commit: index for index, commit in enumerate(history)}
        ranked: list[tuple[int, str, Mapping[str, Any]]] = []
        for gate in exact:
            introduction = self._experiment_admission_introduction_checkpoint(
                gate,
                ref=ref,
            )
            if introduction not in positions:
                raise ResearchGitError(
                    "experiment admission gate introduction is outside first-parent history"
                )
            ranked.append((positions[introduction], introduction, gate))
        _, introduction, latest = max(ranked, key=lambda item: item[0])
        return latest, introduction

    def _repository_wide_experiment_admission_gates(
        self,
    ) -> list[tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]], str]]:
        """Rebuild every host-current decision without trusting the checkout HEAD."""

        pending = load_all_experiment_admission_transitions(self.repository.path)
        if pending:
            raise ResearchGitError(
                "experiment admission high-water transition is pending; retry the signed "
                "verdict ingestion before executing"
            )
        object_cache: dict[str, dict[str, Mapping[str, Any]]] = {}
        current: list[
            tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]], str]
        ] = []
        for event in load_all_experiment_admission_high_water(self.repository.path):
            checkpoint = str(event.get("checkpoint_commit") or "")
            committed_objects = object_cache.get(checkpoint)
            if committed_objects is None:
                committed_objects = {
                    str(item["object_id"]): item
                    for item in list_research_objects_at_ref(
                        self.repository.path,
                        checkpoint,
                    )
                }
                object_cache[checkpoint] = committed_objects
            gate_id = str(event.get("gate_object_id") or "")
            gate = committed_objects.get(gate_id)
            if (
                gate is None
                or gate.get("content_hash") != event.get("gate_object_hash")
                or not self._admission_gate_payload_valid(gate)
            ):
                raise ResearchGitError(
                    "experiment admission host high-water gate is missing or invalid"
                )
            gate_payload = gate.get("payload") or {}
            authority = gate_payload.get("authority_receipt") or {}
            if (
                experiment_admission_subject_hash(gate_payload)
                != event.get("subject_hash")
                or gate_payload.get("decision") != event.get("decision")
                or gate_payload.get("admission_request_hash")
                != event.get("admission_request_hash")
                or authority.get("receipt_hash") != event.get("authority_receipt_hash")
                or self._experiment_admission_introduction_checkpoint(
                    gate,
                    ref=checkpoint,
                )
                != checkpoint
            ):
                raise ResearchGitError(
                    "experiment admission host high-water gate does not match its "
                    "dedicated checkpoint"
                )
            current.append((gate, committed_objects, checkpoint))
        return current

    def prepare_experiment_admission_payload(
        self,
        *,
        plan_id: str,
        preregistration_id: str | None = None,
        execution_scope: Mapping[str, Any] | None = None,
        decision: str,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Build a non-authoritative verdict bound to the committed repository."""

        with _repository_read_lock(self.repository.path):
            initial_head = str(self.repository.status().get("head") or "")
            if not initial_head:
                raise ResearchGitError(
                    "experiment admission requires a committed Research VCS HEAD"
                )
            if expected_head is not None and initial_head != str(expected_head):
                raise ResearchGitError(
                    "Research VCS HEAD changed while experiment admission was prepared"
                )
            context = self._experiment_admission_context_at_head(
                head=initial_head,
                plan_id=plan_id,
                preregistration_id=preregistration_id,
            )
            payload = build_experiment_admission_payload(
                repository_id=context["repository_id"],
                repository_head=initial_head,
                decision=decision,
                hypothesis=context["hypothesis"],
                plan=context["plan"],
                preregistration=context["preregistration"],
                execution_scope=execution_scope,
            )
            if str(self.repository.status().get("head") or "") != initial_head:
                raise ResearchGitError(
                    "Research VCS HEAD changed during experiment admission preparation"
                )
            return payload

    def ingest_experiment_admission(
        self,
        signed_payload: Mapping[str, Any],
        *,
        attestation: Mapping[str, Any],
        trust_store: str | Path,
        commit: bool = True,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Verify and record one externally signed experiment-admission verdict."""

        if commit is not True:
            raise ResearchGitError(
                "experiment admission ingestion requires a dedicated committed checkpoint"
            )
        payload = normalize_experiment_admission_payload(signed_payload)
        with _repository_lock(self.repository.path):
            current_head = str(self.repository.status().get("head") or "")
            if not current_head:
                raise ResearchGitError(
                    "experiment admission requires a committed Research VCS HEAD"
                )
            expected_head_mismatch = bool(
                expected_head is not None and current_head != str(expected_head)
            )
            signed_head = str(payload["repository_head"])
            bindings = payload["input_bindings"]
            preregistration_binding = bindings.get("preregistration")
            try:
                context = self._experiment_admission_context_at_head(
                    head=signed_head,
                    plan_id=str(bindings["plan"]["object_id"]),
                    preregistration_id=(
                        str(preregistration_binding["object_id"])
                        if isinstance(preregistration_binding, Mapping)
                        else None
                    ),
                )
            except (KeyError, TypeError) as exc:  # pragma: no cover - schema defense
                raise ResearchGitError(
                    "experiment admission input bindings are invalid"
                ) from exc
            current_config = load_repository_config_at_ref(
                self.repository.path, current_head
            )
            if payload.get("repository_id") != context["repository_id"] or payload.get(
                "repository_id"
            ) != str(current_config.get("repository_id") or ""):
                raise ResearchGitError(
                    "experiment admission repository identity does not match"
                )
            expected_payload = build_experiment_admission_payload(
                repository_id=context["repository_id"],
                repository_head=signed_head,
                decision=str(payload["decision"]),
                hypothesis=context["hypothesis"],
                plan=context["plan"],
                preregistration=context["preregistration"],
                execution_scope=payload["scope"],
            )
            if not _canonical_values_equal(payload, expected_payload):
                raise ResearchGitError(
                    "experiment admission payload is not bound to the exact committed inputs"
                )
            authority = verify_experiment_admission_attestation(
                payload,
                attestation,
                trust_store_path=trust_store,
                forbidden_roots=_repository_admission_forbidden_roots(
                    self.repository.path
                ),
            )
            target_ids = [
                str(context["plan"]["object_id"]),
                str(context["hypothesis"]["object_id"]),
            ]
            if context["preregistration"] is not None:
                target_ids.append(str(context["preregistration"]["object_id"]))
            producer_closure = self._experiment_admission_producer_closure(
                context["objects"], target_ids=target_ids
            )
            if authority["verifier_principal"] in set(
                producer_closure["producer_principals"]
            ):
                raise ResearchGitError(
                    "experiment admission requires a verifier independent of the "
                    "complete producer provenance"
                )
            scope_producer = payload["scope"].get("producer_id")
            if scope_producer is not None:
                if not isinstance(scope_producer, str) or not scope_producer.strip():
                    raise ResearchGitError(
                        "experiment admission scope producer_id is invalid"
                    )
                try:
                    scope_producer_principal = canonical_principal(
                        scope_producer,
                        label="experiment admission scope producer_id",
                    )
                except ValueError as exc:
                    raise ResearchGitError(
                        "experiment admission scope producer_id is invalid"
                    ) from exc
                if authority["verifier_principal"] == scope_producer_principal:
                    raise ResearchGitError(
                        "experiment admission requires a verifier independent of the "
                        "scoped producer"
                    )
            request_hash = canonical_content_hash(payload)
            authority_core = {
                "protocol_kind": EXPERIMENT_ADMISSION_AUTHORITY_PROTOCOL,
                "verifier_identity": authority["verifier_identity"],
                "verifier_principal": authority["verifier_principal"],
                "key_id": authority["key_id"],
                "algorithm": authority["algorithm"],
                "attestation_hash": authority["attestation_hash"],
                "trust_store_hash": authority["trust_store_hash"],
                "signed_payload_hash": request_hash,
                "producer_actor_ids": producer_closure["producer_actor_ids"],
                "producer_principals": producer_closure["producer_principals"],
                "lineage_object_ids": producer_closure["lineage_object_ids"],
                "signed_payload": deepcopy(payload),
                "attestation_record": deepcopy(authority["attestation_record"]),
            }
            authority_receipt = {
                **authority_core,
                "receipt_hash": canonical_content_hash(authority_core),
            }
            gate_core = {
                "protocol_kind": _EXPERIMENT_ADMISSION_PROTOCOL,
                "gate_id": (
                    "admission-"
                    + request_hash.removeprefix("sha256:")[:16]
                    + "-"
                    + authority["attestation_hash"].removeprefix("sha256:")[:16]
                ),
                "decision": payload["decision"],
                "experiment_authorized": payload["decision"] == "authorize",
                "claim_promotion_allowed": False,
                "repository_id": payload["repository_id"],
                "repository_head": signed_head,
                "input_bindings": deepcopy(payload["input_bindings"]),
                "scope": deepcopy(payload["scope"]),
                "scope_hash": payload["scope_hash"],
                "admission_request_hash": request_hash,
                "authority_receipt": authority_receipt,
            }
            gate_payload = {
                **gate_core,
                "gate_hash": canonical_content_hash(gate_core),
            }
            recover_experiment_admission_staging(
                self.repository.path,
                gate_payload,
            )
            prior_high_water = load_experiment_admission_high_water(
                self.repository.path,
                gate_payload,
            )
            same_verdict = bool(
                prior_high_water is not None
                and prior_high_water.get("admission_request_hash") == request_hash
                and prior_high_water.get("authority_receipt_hash")
                == authority_receipt["receipt_hash"]
            )
            exact_gates = self._exact_experiment_admission_gates(
                context["objects"],
                plan=context["plan"],
                hypothesis=context["hypothesis"],
                preregistration=context["preregistration"],
                scope=payload["scope"],
            )
            exact_ids = {str(item["object_id"]) for item in exact_gates}
            already_superseded = {
                target
                for item in exact_gates
                for target in _relation_targets(item, relation_type="supersedes")
                if target in exact_ids
            }
            active_exact_ids = sorted(exact_ids - already_superseded)
            relations: list[dict[str, str]] = [
                {
                    "type": "evaluates",
                    "target": str(context["hypothesis"]["object_id"]),
                },
                {
                    "type": "depends_on",
                    "target": str(context["plan"]["object_id"]),
                    "role": "plan",
                },
            ]
            if context["preregistration"] is not None:
                relations.append(
                    {
                        "type": "depends_on",
                        "target": str(context["preregistration"]["object_id"]),
                        "role": "protocol",
                    }
                )
            relations.extend(
                {"type": "supersedes", "target": object_id}
                for object_id in active_exact_ids
            )
            try:
                candidate = build_research_object(
                    kind="gate_decision",
                    payload=gate_payload,
                    state="verified",
                    relations=relations,
                    actor=EXPERIMENT_ADMISSION_ACTOR,
                )
            except ResearchObjectError as exc:
                raise ResearchGitError(str(exc)) from exc

            current_objects = {
                str(item["object_id"]): item
                for item in list_research_objects_at_ref(
                    self.repository.path, current_head
                )
            }
            existing = current_objects.get(str(candidate["object_id"]))
            existing_matches = bool(
                existing is not None
                and existing.get("content_hash") == candidate.get("content_hash")
                and self._admission_gate_payload_valid(existing)
            )

            def recorded_result(gate: Mapping[str, Any]) -> ResearchObjectResult:
                return ResearchObjectResult(
                    created=False,
                    path=(
                        self.repository.path
                        / ".xscientist"
                        / "objects"
                        / "gate_decision"
                        / f"{gate['object_id']}.json"
                    ),
                    object_id=str(gate["object_id"]),
                    qualified_id=(
                        str(gate["qualified_id"]) if gate.get("qualified_id") else None
                    ),
                    object_hash=str(gate["content_hash"]),
                    kind="gate_decision",
                    state=str(gate["state"]),
                )

            if current_head != signed_head:
                bindings_current = all(
                    current_objects.get(str(binding["object_id"]), {}).get(
                        "content_hash"
                    )
                    == binding["content_hash"]
                    for binding in payload["input_bindings"].values()
                    if isinstance(binding, Mapping)
                )
                if not existing_matches or not bindings_current:
                    raise ResearchGitError(
                        "Research VCS HEAD changed after the experiment admission "
                        "payload was signed"
                    )
                if expected_head_mismatch and str(expected_head) != signed_head:
                    raise ResearchGitError(
                        "Research VCS HEAD changed before experiment admission ingestion"
                    )
                assert existing is not None  # narrowed by existing_matches
                latest, latest_introduction = (
                    self._latest_committed_experiment_admission_gate(
                        current_objects,
                        plan=context["plan"],
                        hypothesis=context["hypothesis"],
                        preregistration=context["preregistration"],
                        scope=payload["scope"],
                        ref=current_head,
                    )
                )
                if latest.get("object_id") != existing.get("object_id") or latest.get(
                    "content_hash"
                ) != existing.get("content_hash"):
                    raise ResearchGitError(
                        "experiment admission replay is older than the latest committed "
                        "decision"
                    )
                if not same_verdict:
                    self._require_experiment_admission_high_water_visible(
                        prior_high_water,
                        committed_objects=context["objects"],
                        signed_head=signed_head,
                    )
                begin_experiment_admission_transition(
                    self.repository.path,
                    existing,
                )
                verification = self._verify_experiment_admission_gate_authority(
                    existing,
                    committed_objects=context["objects"],
                    plan=context["plan"],
                    hypothesis=context["hypothesis"],
                    preregistration=context["preregistration"],
                    trust_store_path=trust_store,
                    require_high_water=False,
                )
                introduction = str(verification["introduction_checkpoint"])
                if introduction != latest_introduction:
                    raise ResearchGitError(
                        "experiment admission recovery checkpoint does not match the "
                        "latest committed decision"
                    )
                record_experiment_admission_high_water(
                    self.repository.path,
                    existing,
                    checkpoint_commit=introduction,
                )
                clear_experiment_admission_transition(
                    self.repository.path,
                    existing,
                )
                self._verify_experiment_admission_gate_authority(
                    existing,
                    committed_objects=context["objects"],
                    plan=context["plan"],
                    hypothesis=context["hypothesis"],
                    preregistration=context["preregistration"],
                    trust_store_path=trust_store,
                )
                recorded = recorded_result(existing)
                return {
                    "gate": recorded,
                    "checkpoint": None,
                    "authority_receipt": authority_receipt,
                    "idempotent": True,
                    "committed": True,
                    "active": True,
                    "usable": payload["decision"] == "authorize",
                }

            if expected_head_mismatch:
                raise ResearchGitError(
                    "Research VCS HEAD changed before experiment admission ingestion"
                )

            self._require_experiment_admission_high_water_visible(
                prior_high_water,
                committed_objects=context["objects"],
                signed_head=signed_head,
            )
            begin_experiment_admission_transition(
                self.repository.path,
                candidate,
            )

            recorded: ResearchObjectResult | None = None
            checkpoint: CheckpointResult | None = None
            try:
                recorded = self.repository._record(
                    "gate_decision",
                    gate_payload,
                    state="verified",
                    relations=relations,
                    actor=EXPERIMENT_ADMISSION_ACTOR,
                )
                relative_gate = (
                    recorded.path.resolve().relative_to(self.repository.path).as_posix()
                )
                checkpoint = create_checkpoint(
                    self.repository.path,
                    stage="experiment_admission",
                    subject=f"{payload['decision']} exact experiment",
                    status="verified",
                    only_paths=[relative_gate],
                )
                if not checkpoint.committed or not checkpoint.commit:
                    raise ResearchGitError(
                        "experiment admission gate checkpoint was not committed"
                    )
            except BaseException:
                committed_at_head = False
                if recorded is not None:
                    try:
                        committed_at_head = any(
                            item.get("object_id") == recorded.object_id
                            and item.get("content_hash") == recorded.object_hash
                            for item in list_research_objects_at_ref(
                                self.repository.path, "HEAD"
                            )
                        )
                    except BaseException:
                        pass
                if recorded is not None and recorded.created and not committed_at_head:
                    recorded.path.unlink(missing_ok=True)
                    _fsync_directory(recorded.path.parent)
                if not committed_at_head:
                    try:
                        clear_experiment_admission_transition(
                            self.repository.path,
                            candidate,
                        )
                    except BaseException:
                        # A surviving marker is intentionally fail-closed and can be
                        # reconciled by retrying this exact signed verdict.
                        pass
                raise
            if recorded is None:  # pragma: no cover - transaction invariant
                raise ResearchGitError("experiment admission gate was not recorded")
            if checkpoint is None or checkpoint.commit is None:
                raise ResearchGitError(
                    "experiment admission gate checkpoint was not committed"
                )
            committed_after = {
                str(item["object_id"]): item
                for item in list_research_objects_at_ref(
                    self.repository.path,
                    checkpoint.commit,
                )
            }
            committed_gate = committed_after.get(recorded.object_id)
            if (
                committed_gate is None
                or committed_gate.get("content_hash") != recorded.object_hash
            ):
                raise ResearchGitError(
                    "experiment admission gate does not match its committed checkpoint"
                )
            latest, latest_introduction = (
                self._latest_committed_experiment_admission_gate(
                    committed_after,
                    plan=context["plan"],
                    hypothesis=context["hypothesis"],
                    preregistration=context["preregistration"],
                    scope=payload["scope"],
                    ref=checkpoint.commit,
                )
            )
            if latest.get("object_id") != committed_gate.get("object_id") or latest.get(
                "content_hash"
            ) != committed_gate.get("content_hash"):
                raise ResearchGitError(
                    "experiment admission checkpoint is not the latest committed decision"
                )
            verification = self._verify_experiment_admission_gate_authority(
                committed_gate,
                committed_objects=context["objects"],
                plan=context["plan"],
                hypothesis=context["hypothesis"],
                preregistration=context["preregistration"],
                trust_store_path=trust_store,
                require_high_water=False,
            )
            introduction = str(verification["introduction_checkpoint"])
            if checkpoint.commit != introduction or introduction != latest_introduction:
                raise ResearchGitError(
                    "experiment admission gate checkpoint is not its dedicated "
                    "introduction"
                )
            record_experiment_admission_high_water(
                self.repository.path,
                committed_gate,
                checkpoint_commit=introduction,
            )
            clear_experiment_admission_transition(
                self.repository.path,
                committed_gate,
            )
            self._verify_experiment_admission_gate_authority(
                committed_gate,
                committed_objects=context["objects"],
                plan=context["plan"],
                hypothesis=context["hypothesis"],
                preregistration=context["preregistration"],
                trust_store_path=trust_store,
            )
            return {
                "gate": recorded,
                "checkpoint": checkpoint,
                "authority_receipt": authority_receipt,
                "idempotent": not recorded.created,
                "committed": True,
                "active": True,
                "usable": payload["decision"] == "authorize",
            }

    def _validate_experiment_admission(
        self,
        payload: dict[str, Any],
        *,
        raw_status: str,
        plan: Mapping[str, Any] | None,
        preregistration: Mapping[str, Any] | None,
        admission_gate_id: str | None,
        provenance: Mapping[str, Any] | None = None,
        discover_gate: bool = False,
    ) -> tuple[dict[str, Any], str | None]:
        """Apply the repository's execution-admission policy to one receipt."""

        payload = dict(payload)
        requested_gate_id = str(
            admission_gate_id or payload.get("admission_gate_id") or ""
        ).strip()
        if admission_gate_id and payload.get("admission_gate_id") not in {
            None,
            admission_gate_id,
        }:
            raise ResearchGitError("experiment admission gate ids conflict")

        if raw_status in {"orphan", "orphaned"}:
            if requested_gate_id:
                raise ResearchGitError(
                    "orphaned experiment receipts cannot claim an admission gate"
                )
            payload["experiment_authorized"] = False
            payload["claim_promotion_allowed"] = False
            payload["admission"] = {
                "protocol_kind": "xscientist.experiment-admission-receipt.v1",
                "status": "orphaned",
                "reason": "no_valid_pre_execution_admission",
            }
            return payload, None

        if (
            payload.get("experiment_authorized") is False
            or payload.get("scientific_experiment_authorized") is False
        ):
            raise ResearchGitError(
                "unauthorized execution must be recorded with status orphaned"
            )

        required = self._experiment_admission_required()
        if not requested_gate_id:
            if required and not discover_gate:
                raise ResearchGitError(
                    "experiment admission is required; supply a verified gate bound "
                    "to the exact plan and hypothesis"
                )
            if not required:
                return payload, None
        if plan is None:
            raise ResearchGitError("experiment admission requires an exact plan")

        committed_objects = {
            str(item["object_id"]): item
            for item in list_research_objects_at_ref(self.repository.path, "HEAD")
        }
        current_repository_id = str(
            load_repository_config_at_ref(self.repository.path, "HEAD").get(
                "repository_id"
            )
            or ""
        )
        plan = self._require_committed_admission_object(
            plan,
            committed_objects=committed_objects,
            label="plan",
        )
        hypothesis = self._plan_hypothesis(plan, committed_objects=committed_objects)
        if preregistration is not None:
            preregistration = self._require_committed_admission_object(
                preregistration,
                committed_objects=committed_objects,
                label="preregistration",
            )

        execution_scope = dict(payload)
        provenance = provenance if isinstance(provenance, Mapping) else {}
        for field in ("code_commit", "dataset_hashes", "environment_hash"):
            if field in provenance:
                execution_scope[field] = provenance.get(field)
        requested_gate: Mapping[str, Any] | None = None
        if requested_gate_id:
            requested_gate = self._require_committed_admission_object(
                self.repository.get(requested_gate_id),
                committed_objects=committed_objects,
                label="gate",
            )
            if not self._admission_gate_payload_valid(requested_gate):
                raise ResearchGitError(
                    "experiment admission gate is not a verified deterministic "
                    f"{_EXPERIMENT_ADMISSION_PROTOCOL} decision"
                )
            if (requested_gate.get("payload") or {}).get(
                "repository_id"
            ) != current_repository_id:
                raise ResearchGitError(
                    "experiment admission gate repository identity does not match"
                )
            if not self._admission_gate_matches(
                requested_gate,
                plan=plan,
                hypothesis=hypothesis,
                preregistration=preregistration,
                payload=execution_scope,
            ):
                raise ResearchGitError(
                    "experiment admission gate is not bound to the exact plan, "
                    "hypothesis, protocol, or execution scope"
                )

        active_by_id: dict[str, Mapping[str, Any]] = {}
        for (
            latest,
            historical_objects,
            checkpoint,
        ) in self._repository_wide_experiment_admission_gates():
            latest_payload = latest.get("payload") or {}
            if latest_payload.get("repository_id") != current_repository_id:
                continue
            decision = latest_payload.get("decision")
            if decision == "deny":
                relevant = self._admission_denial_overlaps(
                    latest,
                    plan=plan,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    payload=execution_scope,
                )
            else:
                relevant = self._admission_gate_matches(
                    latest,
                    plan=plan,
                    hypothesis=hypothesis,
                    preregistration=preregistration,
                    payload=execution_scope,
                )
            if not relevant:
                continue
            bindings = latest_payload.get("input_bindings") or {}
            historical_plan = historical_objects.get(
                str((bindings.get("plan") or {}).get("object_id") or "")
            )
            historical_hypothesis = historical_objects.get(
                str((bindings.get("hypothesis") or {}).get("object_id") or "")
            )
            preregistration_binding = bindings.get("preregistration")
            historical_preregistration = (
                historical_objects.get(
                    str(preregistration_binding.get("object_id") or "")
                )
                if isinstance(preregistration_binding, Mapping)
                else None
            )
            if historical_plan is None or historical_hypothesis is None:
                raise ResearchGitError(
                    "experiment admission host high-water input bindings are missing"
                )
            verification = self._verify_experiment_admission_gate_authority(
                latest,
                committed_objects=historical_objects,
                plan=historical_plan,
                hypothesis=historical_hypothesis,
                preregistration=historical_preregistration,
                ref=checkpoint,
            )
            high_water = verification.get("high_water") or {}
            if high_water.get("checkpoint_commit") != checkpoint:
                raise ResearchGitError(
                    "experiment admission host high-water state lags the latest "
                    "committed decision"
                )
            active_by_id[str(latest["object_id"])] = latest

        active_gates = list(active_by_id.values())
        active_denials = [
            item
            for item in active_gates
            if (item.get("payload") or {}).get("decision") == "deny"
        ]
        if active_denials:
            raise ResearchGitError(
                "experiment admission is denied for the exact plan and hypothesis"
            )

        gate: Mapping[str, Any]
        if requested_gate is not None:
            gate = active_by_id.get(str(requested_gate["object_id"])) or {}
            if not gate:
                raise ResearchGitError(
                    "experiment admission gate is older than the latest committed "
                    "decision"
                )
        else:
            authorizations = [
                item
                for item in active_gates
                if (item.get("payload") or {}).get("decision") == "authorize"
                and committed_objects.get(str(item.get("object_id") or ""), {}).get(
                    "content_hash"
                )
                == item.get("content_hash")
            ]
            if not authorizations:
                raise ResearchGitError(
                    "experiment admission is required; no committed active "
                    "authorization matches the exact plan, hypothesis, protocol, "
                    "and available execution scope"
                )
            if len(authorizations) > 1:
                raise ResearchGitError(
                    "experiment admission is ambiguous; supply one committed active "
                    "gate explicitly"
                )
            gate = authorizations[0]

        if (gate.get("payload") or {}).get("decision") != "authorize":
            raise ResearchGitError(
                "experiment admission is denied for the exact plan and hypothesis"
            )

        payload["admission_gate_id"] = str(gate["object_id"])
        payload["admission_gate_hash"] = str(gate["content_hash"])
        payload["experiment_authorized"] = True
        payload["claim_promotion_allowed"] = False
        return payload, str(gate["object_id"])

    def preflight_experiment_admission(
        self,
        *,
        plan_id: str,
        preregistration_id: str | None = None,
        admission_gate_id: str | None = None,
        execution_scope: Mapping[str, Any] | None = None,
        code_commit: str | None = None,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Validate execution admission without writing repository or run state."""

        with _repository_read_lock(self.repository.path):
            initial_head = str(self.repository.status().get("head") or "")
            if expected_head is not None and initial_head != str(expected_head):
                raise ResearchGitError(
                    "Research VCS HEAD changed while experiment admission was prepared"
                )
            scope = dict(execution_scope or {})
            unsupported_scope_fields = sorted(
                set(scope) - _EXPERIMENT_ADMISSION_SCOPE_FIELDS
            )
            if unsupported_scope_fields:
                raise ResearchGitError(
                    "experiment admission execution scope contains unsupported fields: "
                    + ", ".join(unsupported_scope_fields)
                )
            try:
                _validate_admission_runtime_scope(
                    {**scope, **({"code_commit": code_commit} if code_commit else {})}
                )
            except ResearchGitError as exc:
                raise ResearchGitError(
                    "experiment admission execution scope is invalid: " + str(exc)
                ) from exc
            plan = self.repository.get(plan_id)
            if plan.get("kind") not in {"research_plan", "experiment_design"}:
                raise ResearchGitError("experiment plan reference has wrong kind")
            preregistration = None
            if preregistration_id:
                preregistration = self.repository.get(preregistration_id)
                if preregistration.get("kind") != "preregistration":
                    raise ResearchGitError(
                        "experiment preregistration reference has wrong kind"
                    )
            provenance = {"code_commit": code_commit} if code_commit else None
            normalized, resolved_gate_id = self._validate_experiment_admission(
                scope,
                raw_status="preflight",
                plan=plan,
                preregistration=preregistration,
                admission_gate_id=admission_gate_id,
                provenance=provenance,
                discover_gate=True,
            )
            if code_commit:
                scope["code_commit"] = code_commit
            final_head = str(self.repository.status().get("head") or "")
            if not initial_head or final_head != initial_head:
                raise ResearchGitError(
                    "Research VCS HEAD changed during experiment admission preflight"
                )
            committed_objects = {
                str(item["object_id"]): item
                for item in list_research_objects_at_ref(
                    self.repository.path, initial_head
                )
            }
            committed_plan = self._require_committed_admission_object(
                plan,
                committed_objects=committed_objects,
                label="plan",
            )
            hypothesis = self._plan_hypothesis(
                committed_plan,
                committed_objects=committed_objects,
            )
            committed_preregistration = (
                self._require_committed_admission_object(
                    preregistration,
                    committed_objects=committed_objects,
                    label="preregistration",
                )
                if preregistration is not None
                else None
            )
            repository_config = load_repository_config_at_ref(
                self.repository.path, initial_head
            )
            core = {
                "protocol_kind": _EXPERIMENT_ADMISSION_PREFLIGHT_PROTOCOL,
                "status": "authorized" if resolved_gate_id else "not_required",
                "experiment_authorized": True,
                "admission_required": self._experiment_admission_required(),
                "repository_id": str(repository_config["repository_id"]),
                "repository_head": initial_head,
                "plan_id": str(committed_plan["object_id"]),
                "plan_hash": str(committed_plan["content_hash"]),
                "hypothesis_id": str(hypothesis["object_id"]),
                "hypothesis_hash": str(hypothesis["content_hash"]),
                "preregistration_id": (
                    str(committed_preregistration["object_id"])
                    if committed_preregistration is not None
                    else None
                ),
                "preregistration_hash": (
                    str(committed_preregistration["content_hash"])
                    if committed_preregistration is not None
                    else None
                ),
                "admission_gate_id": resolved_gate_id,
                "admission_gate_hash": normalized.get("admission_gate_hash"),
                "execution_scope": scope,
                "execution_scope_hash": canonical_content_hash(scope),
            }
            return {**core, "receipt_hash": canonical_content_hash(core)}

    def issue_experiment_admission_start(
        self,
        *,
        plan_id: str,
        preregistration_id: str | None = None,
        admission_gate_id: str | None = None,
        execution_scope: Mapping[str, Any] | None = None,
        code_commit: str | None = None,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Issue one durable, single-consumption start record for a runner."""

        with _repository_lock(self.repository.path):
            preflight = self.preflight_experiment_admission(
                plan_id=plan_id,
                preregistration_id=preregistration_id,
                admission_gate_id=admission_gate_id,
                execution_scope=execution_scope,
                code_commit=code_commit,
                expected_head=expected_head,
            )
            if (
                preflight.get("status") != "authorized"
                or preflight.get("admission_required") is not True
            ):
                raise ResearchGitError(
                    "an experiment start can only be issued from required admission"
                )
            start_core = {
                "protocol_kind": _EXPERIMENT_ADMISSION_START_PROTOCOL,
                "status": "issued",
                "nonce": secrets.token_hex(16),
                "repository_id": preflight["repository_id"],
                "admission_snapshot_head": preflight["repository_head"],
                "plan_id": preflight["plan_id"],
                "plan_hash": preflight["plan_hash"],
                "hypothesis_id": preflight["hypothesis_id"],
                "hypothesis_hash": preflight["hypothesis_hash"],
                "preregistration_id": preflight["preregistration_id"],
                "preregistration_hash": preflight["preregistration_hash"],
                "admission_gate_id": preflight["admission_gate_id"],
                "admission_gate_hash": preflight["admission_gate_hash"],
                "execution_scope": preflight["execution_scope"],
                "execution_scope_hash": preflight["execution_scope_hash"],
                "preflight_receipt_hash": preflight["receipt_hash"],
            }
            relations = [
                {
                    "type": "depends_on",
                    "target": str(preflight["plan_id"]),
                    "role": "plan",
                },
                {
                    "type": "depends_on",
                    "target": str(preflight["admission_gate_id"]),
                    "role": "experiment_admission",
                },
            ]
            if preflight.get("preregistration_id"):
                relations.append(
                    {
                        "type": "depends_on",
                        "target": str(preflight["preregistration_id"]),
                        "role": "protocol",
                    }
                )
            start_hash = canonical_content_hash(start_core)
            start = self.repository._record(
                "context_snapshot",
                {
                    **start_core,
                    "context_hash": start_hash,
                    "start_hash": start_hash,
                },
                state="locked",
                relations=relations,
                actor={
                    "actor_id": "xscientist-project-runner",
                    "authority": "research_agent",
                },
            )
            receipt_core = {
                key: value for key, value in preflight.items() if key != "receipt_hash"
            }
            receipt_core.update(
                {
                    "start_object_id": start.object_id,
                    "start_object_hash": start.object_hash,
                    "start_hash": start_hash,
                }
            )
            return {
                "start": start,
                "receipt": {
                    **receipt_core,
                    "receipt_hash": canonical_content_hash(receipt_core),
                },
            }

    @staticmethod
    def _preflight_core_from_start_payload(
        start_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reconstruct the authorization receipt sealed into one start object."""

        return {
            "protocol_kind": _EXPERIMENT_ADMISSION_PREFLIGHT_PROTOCOL,
            "status": "authorized",
            "experiment_authorized": True,
            "admission_required": True,
            "repository_id": start_payload.get("repository_id"),
            "repository_head": start_payload.get("admission_snapshot_head"),
            "plan_id": start_payload.get("plan_id"),
            "plan_hash": start_payload.get("plan_hash"),
            "hypothesis_id": start_payload.get("hypothesis_id"),
            "hypothesis_hash": start_payload.get("hypothesis_hash"),
            "preregistration_id": start_payload.get("preregistration_id"),
            "preregistration_hash": start_payload.get("preregistration_hash"),
            "admission_gate_id": start_payload.get("admission_gate_id"),
            "admission_gate_hash": start_payload.get("admission_gate_hash"),
            "execution_scope": start_payload.get("execution_scope"),
            "execution_scope_hash": start_payload.get("execution_scope_hash"),
        }

    @staticmethod
    def _validate_issued_experiment_admission_receipt(
        receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        issued = dict(receipt)
        receipt_hash = str(issued.pop("receipt_hash", ""))
        scope = issued.get("execution_scope")
        try:
            scope_hash_valid = bool(
                isinstance(scope, Mapping)
                and issued.get("execution_scope_hash")
                == canonical_content_hash(dict(scope))
            )
        except (TypeError, ValueError):
            scope_hash_valid = False
        expected_fields = {
            "protocol_kind",
            "status",
            "experiment_authorized",
            "admission_required",
            "repository_id",
            "repository_head",
            "plan_id",
            "plan_hash",
            "hypothesis_id",
            "hypothesis_hash",
            "preregistration_id",
            "preregistration_hash",
            "admission_gate_id",
            "admission_gate_hash",
            "execution_scope",
            "execution_scope_hash",
            "start_object_id",
            "start_object_hash",
            "start_hash",
        }
        if set(issued) != expected_fields:
            raise ResearchGitError(
                "issued experiment admission receipt shape is invalid"
            )
        if (
            issued.get("protocol_kind") != _EXPERIMENT_ADMISSION_PREFLIGHT_PROTOCOL
            or issued.get("status") != "authorized"
            or issued.get("experiment_authorized") is not True
            or issued.get("admission_required") is not True
            or not scope_hash_valid
            or receipt_hash != canonical_content_hash(issued)
        ):
            raise ResearchGitError("issued experiment admission receipt is invalid")
        return issued, receipt_hash

    def _committed_experiment_admission_start(
        self,
        *,
        start_object_id: str,
        start_object_hash: str,
        admission_snapshot_head: str,
        start_checkpoint: str,
        allow_descendant_head: bool = False,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        current_head = str(self.repository.status().get("head") or "")
        if not start_checkpoint:
            raise ResearchGitError(
                "Research VCS HEAD changed after the experiment admission start checkpoint"
            )
        if current_head != start_checkpoint:
            first_parent_history = _run_git(
                self.repository.path,
                ["rev-list", "--first-parent", current_head],
                check=False,
            )
            if (
                not allow_descendant_head
                or first_parent_history.returncode
                or start_checkpoint not in first_parent_history.stdout.splitlines()
            ):
                raise ResearchGitError(
                    "Research VCS HEAD changed after the experiment admission start "
                    "checkpoint"
                )
        shown = show_checkpoint(self.repository.path, start_checkpoint)
        checkpoint = shown.get("checkpoint")
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        if (
            shown.get("checkpoint_hash_valid") is not True
            or checkpoint.get("stage") != "experiment_admission_start"
            or checkpoint.get("parent_commit") != admission_snapshot_head
        ):
            raise ResearchGitError(
                "experiment admission start checkpoint is not a dedicated transition"
            )

        committed_now = {
            str(item["object_id"]): item
            for item in list_research_objects_at_ref(
                self.repository.path, start_checkpoint
            )
        }
        committed_before = {
            str(item["object_id"]): item
            for item in list_research_objects_at_ref(
                self.repository.path, admission_snapshot_head
            )
        }
        start = committed_now.get(start_object_id)
        if start is None or start_object_id in committed_before:
            raise ResearchGitError(
                "experiment admission start must first appear in its start checkpoint"
            )
        start_path = f".xscientist/objects/context_snapshot/{start_object_id}.json"
        changed_paths = {str(item) for item in checkpoint.get("changed_paths") or []}
        if start_path not in changed_paths:
            raise ResearchGitError(
                "experiment admission start checkpoint does not introduce its start object"
            )
        for path in changed_paths:
            match = _RESEARCH_OBJECT_PATH.fullmatch(path)
            changed_object = (
                committed_now.get(match.group("object_id")) if match else None
            )
            if (
                match is None
                or match.group("kind") != "context_snapshot"
                or changed_object is None
                or match.group("object_id") in committed_before
                or (changed_object.get("payload") or {}).get("protocol_kind")
                != _EXPERIMENT_ADMISSION_START_PROTOCOL
            ):
                raise ResearchGitError(
                    "experiment admission start checkpoint contains non-start material"
                )

        start_payload = start.get("payload")
        start_payload = start_payload if isinstance(start_payload, Mapping) else {}
        start_actor = start.get("actor")
        start_actor = start_actor if isinstance(start_actor, Mapping) else {}
        expected_start_fields = {
            "protocol_kind",
            "status",
            "nonce",
            "repository_id",
            "admission_snapshot_head",
            "plan_id",
            "plan_hash",
            "hypothesis_id",
            "hypothesis_hash",
            "preregistration_id",
            "preregistration_hash",
            "admission_gate_id",
            "admission_gate_hash",
            "execution_scope",
            "execution_scope_hash",
            "preflight_receipt_hash",
            "context_hash",
            "start_hash",
        }
        start_core = {
            key: value
            for key, value in start_payload.items()
            if key not in {"context_hash", "start_hash"}
        }
        preflight_core = self._preflight_core_from_start_payload(start_payload)
        scope = start_payload.get("execution_scope")
        if (
            set(start_payload) != expected_start_fields
            or start.get("kind") != "context_snapshot"
            or start.get("state") != "locked"
            or start.get("content_hash") != start_object_hash
            or start_payload.get("protocol_kind")
            != _EXPERIMENT_ADMISSION_START_PROTOCOL
            or start_payload.get("status") != "issued"
            or start_actor.get("actor_id") != "xscientist-project-runner"
            or start_actor.get("authority") != "research_agent"
            or not str(start_payload.get("nonce") or "")
            or not isinstance(scope, Mapping)
            or start_payload.get("execution_scope_hash")
            != canonical_content_hash(dict(scope or {}))
            or start_payload.get("preflight_receipt_hash")
            != canonical_content_hash(preflight_core)
            or start_payload.get("start_hash") != canonical_content_hash(start_core)
            or start_payload.get("context_hash") != start_payload.get("start_hash")
            or start_payload.get("admission_snapshot_head") != admission_snapshot_head
        ):
            raise ResearchGitError("experiment admission start object is invalid")

        snapshot_config = load_repository_config_at_ref(
            self.repository.path, admission_snapshot_head
        )
        checkpoint_config = load_repository_config_at_ref(
            self.repository.path, start_checkpoint
        )
        if snapshot_config.get("repository_id") != start_payload.get(
            "repository_id"
        ) or checkpoint_config.get("repository_id") != start_payload.get(
            "repository_id"
        ):
            raise ResearchGitError("experiment admission repository identity changed")

        binding_fields = {
            "plan_id": "plan_hash",
            "hypothesis_id": "hypothesis_hash",
            "admission_gate_id": "admission_gate_hash",
        }
        if start_payload.get("preregistration_id") is not None:
            binding_fields["preregistration_id"] = "preregistration_hash"
        elif start_payload.get("preregistration_hash") is not None:
            raise ResearchGitError(
                "experiment admission start has an unpaired preregistration hash"
            )
        for id_field, hash_field in binding_fields.items():
            bound = committed_before.get(str(start_payload.get(id_field) or ""))
            if bound is None or bound.get("content_hash") != start_payload.get(
                hash_field
            ):
                raise ResearchGitError(
                    f"experiment admission start {id_field} binding is invalid"
                )
        committed_plan = committed_before[str(start_payload["plan_id"])]
        if committed_plan.get("kind") not in {"research_plan", "experiment_design"}:
            raise ResearchGitError("experiment admission start plan has wrong kind")
        committed_hypothesis = self._plan_hypothesis(
            committed_plan,
            committed_objects=committed_before,
        )
        if committed_hypothesis.get("object_id") != start_payload.get(
            "hypothesis_id"
        ) or committed_hypothesis.get("content_hash") != start_payload.get(
            "hypothesis_hash"
        ):
            raise ResearchGitError(
                "experiment admission start hypothesis binding is invalid"
            )
        committed_preregistration = (
            committed_before[str(start_payload["preregistration_id"])]
            if start_payload.get("preregistration_id")
            else None
        )
        if (
            committed_preregistration is not None
            and committed_preregistration.get("kind") != "preregistration"
        ):
            raise ResearchGitError(
                "experiment admission start preregistration has wrong kind"
            )
        committed_gate = committed_before[str(start_payload["admission_gate_id"])]
        if (committed_gate.get("payload") or {}).get(
            "decision"
        ) != "authorize" or not self._admission_gate_matches(
            committed_gate,
            plan=committed_plan,
            hypothesis=committed_hypothesis,
            preregistration=committed_preregistration,
            payload=dict(scope),
        ):
            raise ResearchGitError("experiment admission start gate binding is invalid")
        if _relation_targets(start, relation_type="depends_on", role="plan") != {
            str(start_payload["plan_id"])
        } or _relation_targets(
            start, relation_type="depends_on", role="experiment_admission"
        ) != {
            str(start_payload["admission_gate_id"])
        }:
            raise ResearchGitError(
                "experiment admission start relations do not match its bindings"
            )
        expected_protocol_targets = (
            {str(start_payload["preregistration_id"])}
            if start_payload.get("preregistration_id")
            else set()
        )
        if (
            _relation_targets(start, relation_type="depends_on", role="protocol")
            != expected_protocol_targets
        ):
            raise ResearchGitError(
                "experiment admission start protocol relation does not match"
            )
        return start, checkpoint

    def finalize_experiment_admission_start(
        self,
        receipt: Mapping[str, Any],
        *,
        start_checkpoint: str | None = None,
        allow_descendant_head: bool = False,
    ) -> dict[str, Any]:
        """Finalize an issued receipt after its start is independently committed."""

        with _repository_read_lock(self.repository.path):
            issued, issued_receipt_hash = (
                self._validate_issued_experiment_admission_receipt(receipt)
            )
            admission_snapshot_head = str(issued["repository_head"])
            checkpoint_head = str(
                start_checkpoint or self.repository.status().get("head") or ""
            )
            start, checkpoint = self._committed_experiment_admission_start(
                start_object_id=str(issued["start_object_id"]),
                start_object_hash=str(issued["start_object_hash"]),
                admission_snapshot_head=admission_snapshot_head,
                start_checkpoint=checkpoint_head,
                allow_descendant_head=allow_descendant_head,
            )
            start_payload = start.get("payload") or {}
            preflight_core = self._preflight_core_from_start_payload(start_payload)
            issued_preflight_core = {
                key: value
                for key, value in issued.items()
                if key not in {"start_object_id", "start_object_hash", "start_hash"}
            }
            if (
                not _canonical_values_equal(issued_preflight_core, preflight_core)
                or start_payload.get("preflight_receipt_hash")
                != canonical_content_hash(preflight_core)
                or issued.get("start_hash") != start_payload.get("start_hash")
            ):
                raise ResearchGitError(
                    "issued experiment admission receipt does not match its start object"
                )
            final_core = {
                "protocol_kind": _EXPERIMENT_ADMISSION_FINAL_PROTOCOL,
                "status": "finalized",
                "experiment_authorized": True,
                "admission_required": True,
                "repository_id": issued["repository_id"],
                "admission_snapshot_head": admission_snapshot_head,
                "repository_head": checkpoint_head,
                "start_checkpoint": checkpoint_head,
                "start_checkpoint_id": checkpoint.get("checkpoint_id"),
                "start_checkpoint_hash": checkpoint.get("content_hash"),
                "plan_id": issued["plan_id"],
                "plan_hash": issued["plan_hash"],
                "hypothesis_id": issued["hypothesis_id"],
                "hypothesis_hash": issued["hypothesis_hash"],
                "preregistration_id": issued["preregistration_id"],
                "preregistration_hash": issued["preregistration_hash"],
                "admission_gate_id": issued["admission_gate_id"],
                "admission_gate_hash": issued["admission_gate_hash"],
                "execution_scope": issued["execution_scope"],
                "execution_scope_hash": issued["execution_scope_hash"],
                "preflight_receipt_hash": start_payload["preflight_receipt_hash"],
                "issued_receipt_hash": issued_receipt_hash,
                "start_object_id": issued["start_object_id"],
                "start_object_hash": issued["start_object_hash"],
                "start_hash": issued["start_hash"],
                "start_nonce": start_payload["nonce"],
            }
            return {
                **final_core,
                "receipt_hash": canonical_content_hash(final_core),
            }

    def reconstruct_finalized_experiment_admission_start(
        self,
        *,
        start_object_id: str,
        start_checkpoint: str,
    ) -> dict[str, Any]:
        """Rebuild a finalized receipt solely from one committed start."""

        committed = {
            str(item.get("object_id") or ""): item
            for item in list_research_objects_at_ref(
                self.repository.path, start_checkpoint
            )
        }
        start = committed.get(str(start_object_id))
        payload = start.get("payload") if isinstance(start, Mapping) else None
        if (
            not isinstance(start, Mapping)
            or start.get("kind") != "context_snapshot"
            or not isinstance(payload, Mapping)
            or payload.get("protocol_kind") != _EXPERIMENT_ADMISSION_START_PROTOCOL
        ):
            raise ResearchGitError(
                "committed experiment admission start cannot be reconstructed"
            )
        preflight_core = self._preflight_core_from_start_payload(payload)
        issued_core = {
            **preflight_core,
            "start_object_id": start_object_id,
            "start_object_hash": start.get("content_hash"),
            "start_hash": payload.get("start_hash"),
        }
        finalized = self.finalize_experiment_admission_start(
            {
                **issued_core,
                "receipt_hash": canonical_content_hash(issued_core),
            },
            start_checkpoint=start_checkpoint,
            allow_descendant_head=True,
        )
        return self.validate_finalized_experiment_admission_start(
            finalized,
            allow_descendant_head=True,
            allow_terminal_state=True,
        )

    def reconstruct_experiment_admission_batch(
        self,
        start_checkpoint: str,
    ) -> dict[int, dict[str, Any]]:
        """Rebuild the exact indexed batch introduced by a start checkpoint."""

        shown = show_checkpoint(self.repository.path, start_checkpoint)
        checkpoint = shown.get("checkpoint")
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        changed_paths = checkpoint.get("changed_paths")
        if (
            shown.get("checkpoint_hash_valid") is not True
            or shown.get("commit") != start_checkpoint
            or checkpoint.get("stage") != "experiment_admission_start"
            or checkpoint.get("status") != "locked"
            or not isinstance(changed_paths, list)
            or not changed_paths
            or len(set(changed_paths)) != len(changed_paths)
        ):
            raise ResearchGitError(
                "experiment admission start batch checkpoint is invalid"
            )
        prefix = ".xscientist/objects/context_snapshot/"
        start_ids: list[str] = []
        for path in changed_paths:
            if (
                not isinstance(path, str)
                or not path.startswith(prefix)
                or not path.endswith(".json")
            ):
                raise ResearchGitError(
                    "experiment admission start checkpoint contains non-start material"
                )
            start_ids.append(path[len(prefix) : -len(".json")])
        receipts: dict[int, dict[str, Any]] = {}
        for start_object_id in start_ids:
            receipt = self.reconstruct_finalized_experiment_admission_start(
                start_object_id=start_object_id,
                start_checkpoint=start_checkpoint,
            )
            scope = receipt.get("execution_scope")
            idea_index = scope.get("idea_index") if isinstance(scope, Mapping) else None
            if type(idea_index) is not int or idea_index < 0 or idea_index in receipts:
                raise ResearchGitError(
                    "experiment admission start batch has an invalid idea index"
                )
            receipts[idea_index] = receipt
        return {index: receipts[index] for index in sorted(receipts)}

    def validate_finalized_experiment_admission_start(
        self,
        receipt: Mapping[str, Any],
        *,
        allow_descendant_head: bool = False,
        allow_terminal_state: bool = False,
    ) -> dict[str, Any]:
        """Validate one committed and finalized start receipt.

        ``allow_terminal_state`` is reserved for terminal recovery.  Callers still
        have to prove the durable execution claim and exact attempt identity.
        """

        with _repository_read_lock(self.repository.path):
            finalized = dict(receipt)
            receipt_hash = str(finalized.pop("receipt_hash", ""))
            scope = finalized.get("execution_scope")
            try:
                scope_hash_valid = bool(
                    isinstance(scope, Mapping)
                    and finalized.get("execution_scope_hash")
                    == canonical_content_hash(dict(scope))
                )
            except (TypeError, ValueError):
                scope_hash_valid = False
            expected_fields = {
                "protocol_kind",
                "status",
                "experiment_authorized",
                "admission_required",
                "repository_id",
                "admission_snapshot_head",
                "repository_head",
                "start_checkpoint",
                "start_checkpoint_id",
                "start_checkpoint_hash",
                "plan_id",
                "plan_hash",
                "hypothesis_id",
                "hypothesis_hash",
                "preregistration_id",
                "preregistration_hash",
                "admission_gate_id",
                "admission_gate_hash",
                "execution_scope",
                "execution_scope_hash",
                "preflight_receipt_hash",
                "issued_receipt_hash",
                "start_object_id",
                "start_object_hash",
                "start_hash",
                "start_nonce",
            }
            if set(finalized) != expected_fields:
                raise ResearchGitError(
                    "finalized experiment admission receipt shape is invalid"
                )
            if (
                finalized.get("protocol_kind") != _EXPERIMENT_ADMISSION_FINAL_PROTOCOL
                or finalized.get("status") != "finalized"
                or finalized.get("experiment_authorized") is not True
                or finalized.get("admission_required") is not True
                or finalized.get("repository_head") != finalized.get("start_checkpoint")
                or not scope_hash_valid
                or receipt_hash != canonical_content_hash(finalized)
            ):
                raise ResearchGitError(
                    "finalized experiment admission receipt is invalid"
                )
            start, checkpoint = self._committed_experiment_admission_start(
                start_object_id=str(finalized["start_object_id"]),
                start_object_hash=str(finalized["start_object_hash"]),
                admission_snapshot_head=str(finalized["admission_snapshot_head"]),
                start_checkpoint=str(finalized["start_checkpoint"]),
                allow_descendant_head=allow_descendant_head,
            )
            start_payload = start.get("payload") or {}
            preflight_core = self._preflight_core_from_start_payload(start_payload)
            issued_core = {
                **preflight_core,
                "start_object_id": finalized["start_object_id"],
                "start_object_hash": finalized["start_object_hash"],
                "start_hash": finalized["start_hash"],
            }
            matching_fields = {
                "repository_id": "repository_id",
                "plan_id": "plan_id",
                "plan_hash": "plan_hash",
                "hypothesis_id": "hypothesis_id",
                "hypothesis_hash": "hypothesis_hash",
                "preregistration_id": "preregistration_id",
                "preregistration_hash": "preregistration_hash",
                "admission_gate_id": "admission_gate_id",
                "admission_gate_hash": "admission_gate_hash",
                "execution_scope": "execution_scope",
                "execution_scope_hash": "execution_scope_hash",
                "start_hash": "start_hash",
                "start_nonce": "nonce",
            }
            if (
                finalized.get("start_checkpoint_id") != checkpoint.get("checkpoint_id")
                or finalized.get("start_checkpoint_hash")
                != checkpoint.get("content_hash")
                or finalized.get("preflight_receipt_hash")
                != canonical_content_hash(preflight_core)
                or start_payload.get("preflight_receipt_hash")
                != canonical_content_hash(preflight_core)
                or finalized.get("issued_receipt_hash")
                != canonical_content_hash(issued_core)
                or any(
                    not _canonical_values_equal(
                        finalized.get(receipt_field),
                        start_payload.get(start_field),
                    )
                    for receipt_field, start_field in matching_fields.items()
                )
            ):
                raise ResearchGitError(
                    "finalized experiment admission receipt does not match its committed start"
                )
            consumed_by = [
                str(item.get("object_id") or "")
                for item in self.repository.objects(kind="experiment_attempt")
                if str(finalized["start_object_id"])
                in _relation_targets(
                    item,
                    relation_type="consumes",
                    role="experiment_admission_start",
                )
            ]
            if consumed_by and not allow_terminal_state:
                raise ResearchGitError(
                    "experiment admission start receipt has already been consumed"
                )
            if not allow_terminal_state and _experiment_admission_terminal_reserved(
                self.repository.path,
                start_object_id=str(finalized["start_object_id"]),
            ):
                raise ResearchGitError(
                    "experiment admission start has already been reserved for terminal "
                    "recording"
                )
            return {**finalized, "receipt_hash": receipt_hash}

    def _validated_terminal_admission_receipt(
        self,
        payload: Mapping[str, Any],
        *,
        raw_status: str,
        admission_preflight: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None]:
        """Validate a terminal receipt and return its matching durable claim, if any."""

        embedded = payload.get("admission_preflight_receipt")
        if admission_preflight is not None and embedded is not None:
            if not _canonical_values_equal(dict(admission_preflight), embedded):
                raise ResearchGitError(
                    "experiment admission preflight receipts conflict"
                )
        receipt = admission_preflight if admission_preflight is not None else embedded
        if raw_status in {"orphan", "orphaned"}:
            return None, None
        if receipt is None:
            return None, None
        if not isinstance(receipt, Mapping):
            raise ResearchGitError("experiment admission preflight receipt is invalid")
        finalized = self.validate_finalized_experiment_admission_start(
            receipt,
            allow_descendant_head=True,
            allow_terminal_state=True,
        )
        claim = _load_experiment_admission_execution_claim(
            self.repository.path,
            start_object_id=str(finalized["start_object_id"]),
            receipt_hash=str(finalized["receipt_hash"]),
        )
        return finalized, claim

    @staticmethod
    def _post_claim_policy_status(error: ResearchGitError) -> str:
        message = str(error).lower()
        if "is denied" in message:
            return "denied_after_start"
        if "revoked" in message:
            return "signer_revoked_after_start"
        if "transition is pending" in message:
            return "decision_transition_after_start"
        if "trust store" in message or "authority" in message or "signature" in message:
            return "authority_policy_changed_after_start"
        return "authorization_changed_after_start"

    def _validate_claimed_terminal_admission(
        self,
        payload: dict[str, Any],
        *,
        plan: Mapping[str, Any] | None,
        preregistration: Mapping[str, Any] | None,
        admission_gate_id: str | None,
        provenance: Mapping[str, Any] | None,
        receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Preserve an executed outcome against its immutable admission snapshot."""

        if plan is None:
            raise ResearchGitError(
                "experiment admission terminal requires an exact plan"
            )
        committed_objects = {
            str(item["object_id"]): item
            for item in list_research_objects_at_ref(self.repository.path, "HEAD")
        }
        plan = self._require_committed_admission_object(
            plan,
            committed_objects=committed_objects,
            label="plan",
        )
        hypothesis = self._plan_hypothesis(plan, committed_objects=committed_objects)
        if preregistration is not None:
            preregistration = self._require_committed_admission_object(
                preregistration,
                committed_objects=committed_objects,
                label="preregistration",
            )
        requested_gate_id = str(
            admission_gate_id or payload.get("admission_gate_id") or ""
        ).strip()
        if (
            not requested_gate_id
            or requested_gate_id != receipt.get("admission_gate_id")
            or receipt.get("plan_id") != plan.get("object_id")
            or receipt.get("plan_hash") != plan.get("content_hash")
            or receipt.get("hypothesis_id") != hypothesis.get("object_id")
            or receipt.get("hypothesis_hash") != hypothesis.get("content_hash")
        ):
            raise ResearchGitError(
                "claimed experiment admission terminal bindings do not match"
            )
        expected_preregistration_id = (
            preregistration.get("object_id") if preregistration is not None else None
        )
        expected_preregistration_hash = (
            preregistration.get("content_hash") if preregistration is not None else None
        )
        if (
            receipt.get("preregistration_id") != expected_preregistration_id
            or receipt.get("preregistration_hash") != expected_preregistration_hash
        ):
            raise ResearchGitError(
                "claimed experiment admission terminal protocol does not match"
            )
        gate = committed_objects.get(requested_gate_id)
        if (
            gate is None
            or gate.get("content_hash") != receipt.get("admission_gate_hash")
            or (gate.get("payload") or {}).get("decision") != "authorize"
        ):
            raise ResearchGitError(
                "claimed experiment admission terminal gate does not match"
            )
        execution_scope = {
            field: payload[field]
            for field in _EXPERIMENT_ADMISSION_SCOPE_FIELDS
            if field in payload
        }
        provenance = provenance if isinstance(provenance, Mapping) else {}
        for field in ("code_commit", "dataset_hashes", "environment_hash"):
            if field in provenance:
                execution_scope[field] = provenance.get(field)
        if not _canonical_values_equal(
            execution_scope,
            receipt.get("execution_scope"),
        ) or not self._admission_gate_matches(
            gate,
            plan=plan,
            hypothesis=hypothesis,
            preregistration=preregistration,
            payload=execution_scope,
        ):
            raise ResearchGitError(
                "claimed experiment admission terminal execution scope changed"
            )
        payload["admission_gate_id"] = requested_gate_id
        payload["admission_gate_hash"] = str(gate["content_hash"])
        payload["experiment_authorized"] = True
        payload["claim_promotion_allowed"] = False
        return payload, requested_gate_id

    def _bind_experiment_admission_preflight(
        self,
        payload: dict[str, Any],
        *,
        raw_status: str,
        plan: Mapping[str, Any] | None,
        preregistration: Mapping[str, Any] | None,
        admission_gate_id: str | None,
        provenance: Mapping[str, Any] | None,
        admission_preflight: Mapping[str, Any] | None,
        validated_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and bind the exact start receipt consumed by an attempt."""

        payload = dict(payload)
        embedded = payload.pop("admission_preflight_receipt", None)
        if admission_preflight is not None and embedded is not None:
            if not _canonical_values_equal(dict(admission_preflight), embedded):
                raise ResearchGitError(
                    "experiment admission preflight receipts conflict"
                )
        receipt = admission_preflight if admission_preflight is not None else embedded
        if raw_status in {"orphan", "orphaned"}:
            if receipt is not None:
                raise ResearchGitError(
                    "orphaned experiment receipts cannot claim a start preflight"
                )
            return payload
        if receipt is None:
            if self._experiment_admission_required():
                raise ResearchGitError(
                    "experiment admission is required; a bound start preflight "
                    "receipt is missing"
                )
            return payload
        if not isinstance(receipt, Mapping):
            raise ResearchGitError("experiment admission preflight receipt is invalid")
        if validated_receipt is not None and not _canonical_values_equal(
            validated_receipt,
            receipt,
        ):
            raise ResearchGitError("experiment admission preflight receipts conflict")
        bound_receipt = (
            dict(validated_receipt)
            if validated_receipt is not None
            else self.validate_finalized_experiment_admission_start(
                receipt,
                allow_descendant_head=True,
                allow_terminal_state=True,
            )
        )
        receipt_hash = bound_receipt["receipt_hash"]
        if (
            plan is None
            or bound_receipt.get("plan_id") != plan.get("object_id")
            or bound_receipt.get("plan_hash") != plan.get("content_hash")
        ):
            raise ResearchGitError("experiment admission preflight plan does not match")
        expected_preregistration_id = (
            preregistration.get("object_id") if preregistration is not None else None
        )
        expected_preregistration_hash = (
            preregistration.get("content_hash") if preregistration is not None else None
        )
        if (
            bound_receipt.get("preregistration_id") != expected_preregistration_id
            or bound_receipt.get("preregistration_hash")
            != expected_preregistration_hash
        ):
            raise ResearchGitError(
                "experiment admission preflight preregistration does not match"
            )
        if bound_receipt.get(
            "admission_gate_id"
        ) != admission_gate_id or bound_receipt.get(
            "admission_gate_hash"
        ) != payload.get(
            "admission_gate_hash"
        ):
            raise ResearchGitError("experiment admission preflight gate does not match")
        scope = bound_receipt["execution_scope"]
        actual_scope = dict(payload)
        provenance = provenance if isinstance(provenance, Mapping) else {}
        for field in ("code_commit", "dataset_hashes", "environment_hash"):
            if field in provenance:
                actual_scope[field] = provenance.get(field)
        if not self._admission_scope_matches(scope, actual_scope):
            raise ResearchGitError(
                "experiment admission preflight execution scope changed"
            )
        payload["admission_preflight_receipt"] = bound_receipt
        payload["admission_preflight_hash"] = receipt_hash
        return payload

    def _begin_result_artifact_pointer_transaction(
        self,
        planned_bindings: Mapping[Path, tuple[str, str]],
    ) -> _ResultArtifactPointerTransaction:
        """Snapshot exact pointer leaves before the command publishes any of them."""

        root = self.repository.path
        current_state = self.repository.status()
        current_head = str(current_state.get("head") or "").strip()
        if not current_head:
            raise ResearchGitError(
                "result artifact pointers require a committed repository state"
            )
        config = load_repository_config_at_ref(root, current_head)
        pointer_root = _resolve_configured_path(
            root,
            (config.get("storage") or {}).get("pointer_directory")
            or "research-objects",
            label="pointer directory",
        )
        normalized_paths: dict[str, Path] = {}
        normalized_bindings: dict[str, tuple[str, str]] = {}
        for raw_path, raw_binding in planned_bindings.items():
            path = Path(raw_path).resolve()
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ResearchGitError(
                    "planned result artifact pointer escapes the repository"
                ) from exc
            if path.parent != pointer_root or relative in normalized_paths:
                raise ResearchGitError(
                    "planned result artifact pointer is outside its configured directory"
                )
            object_hash, logical_path = raw_binding
            if _CONTENT_HASH_RE.fullmatch(str(object_hash)) is None or not str(
                logical_path
            ):
                raise ResearchGitError("planned result artifact binding is invalid")
            normalized_paths[relative] = path
            normalized_bindings[relative] = (str(object_hash), str(logical_path))
        preexisting_paths = {
            relative
            for relative, path in normalized_paths.items()
            if path.exists() or path.is_symlink()
        }
        transaction = _ResultArtifactPointerTransaction(
            repository_path=root,
            head=current_head,
            planned_bindings=normalized_bindings,
            preexisting_paths=preexisting_paths,
            issuer=_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER,
        )
        return transaction

    def _capture_result_artifact_pointer_rollback_receipt(
        self,
        pointer_transaction: _ResultArtifactPointerTransaction,
        pointer_results: Sequence[ObjectPointerResult],
    ) -> _ResultArtifactPointerRollbackReceipt:
        """Capture only new pointer leaves whose on-disk identity is fully verified."""

        root = self.repository.path
        transaction_head, planned_bindings, preexisting_paths = (
            pointer_transaction._snapshot(
                repository_path=root,
                issuer=_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER,
            )
        )
        records: dict[str, tuple[tuple[int, ...], str]] = {}
        if str(self.repository.status().get("head") or "") != transaction_head:
            return _ResultArtifactPointerRollbackReceipt(
                repository_path=root,
                head=transaction_head,
                records=records,
                issuer=_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER,
            )
        for result in pointer_results:
            if type(result) is not ObjectPointerResult or result.linked:
                continue
            try:
                pointer_path = result.pointer_path.resolve()
                relative_pointer = pointer_path.relative_to(root).as_posix()
            except (OSError, RuntimeError, ValueError):
                continue
            expected = planned_bindings.get(relative_pointer)
            if expected is None or relative_pointer in preexisting_paths:
                continue
            try:
                encoded = read_bounded_regular_file(
                    pointer_path,
                    maximum=_MAX_RESULT_ARTIFACT_POINTER_BYTES,
                    label="result_artifact_rollback_pointer",
                )
                pointer = _validate_pointer_payload(
                    json.loads(encoded.decode("utf-8")),
                    pointer_path=relative_pointer,
                )
                store_path, store_error = _verify_pointer_object(
                    root,
                    expected[0],
                    pointer,
                )
                if (
                    pointer.get("object_hash") != expected[0]
                    or pointer.get("logical_path") != expected[1]
                    or result.object_hash != expected[0]
                    or result.size != pointer.get("size")
                    or store_error is not None
                    or store_path is None
                    or result.store_path.resolve() != store_path.resolve()
                ):
                    continue
                identity = _result_artifact_pointer_identity(pointer_path)
            except (
                BoundedFileError,
                OSError,
                RuntimeError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
                ResearchGitError,
            ):
                continue
            records[relative_pointer] = (
                identity,
                hashlib.sha256(encoded).hexdigest(),
            )
        return _ResultArtifactPointerRollbackReceipt(
            repository_path=root,
            head=transaction_head,
            records=records,
            issuer=_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER,
        )

    def _rollback_result_artifact_pointer_transaction(
        self,
        pointer_transaction: _ResultArtifactPointerTransaction,
        rollback_receipt: _ResultArtifactPointerRollbackReceipt,
    ) -> None:
        """Unlink only verified transaction leaves that have not changed since capture."""

        root = self.repository.path
        transaction_head, planned_bindings, preexisting_paths = (
            pointer_transaction._snapshot(
                repository_path=root,
                issuer=_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER,
            )
        )
        receipt_head, records = rollback_receipt._snapshot(
            repository_path=root,
            issuer=_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER,
        )
        if (
            receipt_head != transaction_head
            or str(self.repository.status().get("head") or "") != transaction_head
        ):
            return
        touched_parents: set[Path] = set()
        for relative_pointer, (captured_identity, captured_hash) in records.items():
            expected = planned_bindings.get(relative_pointer)
            if expected is None or relative_pointer in preexisting_paths:
                continue
            pointer_path = (root / relative_pointer).resolve()
            try:
                if pointer_path.relative_to(root).as_posix() != relative_pointer:
                    continue
                encoded = read_bounded_regular_file(
                    pointer_path,
                    maximum=_MAX_RESULT_ARTIFACT_POINTER_BYTES,
                    label="result_artifact_rollback_pointer",
                )
                if (
                    hashlib.sha256(encoded).hexdigest() != captured_hash
                    or _result_artifact_pointer_identity(pointer_path)
                    != captured_identity
                ):
                    continue
                pointer = _validate_pointer_payload(
                    json.loads(encoded.decode("utf-8")),
                    pointer_path=relative_pointer,
                )
                _store_path, store_error = _verify_pointer_object(
                    root,
                    expected[0],
                    pointer,
                )
                if (
                    pointer.get("object_hash") != expected[0]
                    or pointer.get("logical_path") != expected[1]
                    or store_error is not None
                ):
                    continue
                pointer_path.unlink()
                touched_parents.add(pointer_path.parent)
            except (
                BoundedFileError,
                FileNotFoundError,
                OSError,
                RuntimeError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
                ResearchGitError,
            ):
                continue
        for parent in touched_parents:
            _fsync_directory(parent)

    def _validate_result_artifact_bindings(
        self,
        payload: Mapping[str, Any],
        *,
        pointer_results: Sequence[ObjectPointerResult] | None = None,
        pointer_transaction: _ResultArtifactPointerTransaction | None = None,
        allowed_new_paths: set[str] | None = None,
    ) -> set[str]:
        """Verify durable result pointers from disk and return new untracked paths."""

        if (pointer_results is not None) != (pointer_transaction is not None):
            raise ResearchGitError(
                "result artifact verification requires complete transaction state"
            )
        if pointer_results is not None and allowed_new_paths is not None:
            raise ResearchGitError(
                "result artifact verification received ambiguous transaction state"
            )

        raw_bindings = payload.get("result_artifacts")
        if raw_bindings is None:
            if pointer_results:
                raise ResearchGitError(
                    "result artifact pointers are not bound to the attempt payload"
                )
            return set()
        if not isinstance(raw_bindings, Mapping) or not raw_bindings:
            raise ResearchGitError(
                "result artifact bindings must be a non-empty mapping"
            )
        raw_hashes = payload.get("result_artifact_hashes")
        if not isinstance(raw_hashes, Mapping) or set(raw_hashes) != set(raw_bindings):
            raise ResearchGitError(
                "result artifact hashes and pointer bindings must have identical labels"
            )

        root = self.repository.path
        current_state = self.repository.status()
        current_head = str(current_state.get("head") or "").strip()
        if not current_head:
            raise ResearchGitError(
                "result artifact pointers require a committed repository state"
            )
        transaction_planned_bindings: dict[str, tuple[str, str]] | None = None
        transaction_preexisting_paths: set[str] = set()
        if pointer_transaction is not None:
            (
                transaction_head,
                transaction_planned_bindings,
                transaction_preexisting_paths,
            ) = pointer_transaction._snapshot(
                repository_path=root,
                issuer=_RESULT_ARTIFACT_POINTER_TRANSACTION_ISSUER,
            )
            if transaction_head != current_head:
                raise ResearchGitError(
                    "repository HEAD changed during result artifact transaction"
                )
        config = load_repository_config_at_ref(root, current_head)
        pointer_root = _resolve_configured_path(
            root,
            (config.get("storage") or {}).get("pointer_directory")
            or "research-objects",
            label="pointer directory",
        )

        supplied_results: dict[str, ObjectPointerResult] | None = None
        if pointer_results is not None:
            supplied_results = {}
            for result in pointer_results:
                if type(result) is not ObjectPointerResult:
                    raise ResearchGitError(
                        "result artifact transaction returned an invalid pointer result"
                    )
                try:
                    result_path = (
                        result.pointer_path.resolve().relative_to(root).as_posix()
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ResearchGitError(
                        "result artifact pointer is outside the research repository"
                    ) from exc
                if result_path in supplied_results:
                    raise ResearchGitError(
                        "result artifact transaction returned a duplicate pointer"
                    )
                supplied_results[result_path] = result

        expected_fields = {
            "logical_name",
            "content_hash",
            "logical_path",
            "pointer_path",
        }
        bound_paths: set[str] = set()
        new_untracked_paths: set[str] = set()
        for raw_label, raw_binding in raw_bindings.items():
            label = str(raw_label)
            if not label or not isinstance(raw_binding, Mapping):
                raise ResearchGitError("result artifact binding is invalid")
            binding = dict(raw_binding)
            if set(binding) != expected_fields:
                raise ResearchGitError(
                    f"result artifact binding has unexpected fields: {label}"
                )
            object_hash = str(binding.get("content_hash") or "")
            if (
                binding.get("logical_name") != label
                or raw_hashes.get(raw_label) != object_hash
                or _CONTENT_HASH_RE.fullmatch(object_hash) is None
            ):
                raise ResearchGitError(
                    f"result artifact hash binding does not match its label: {label}"
                )
            pointer_path_value = str(binding.get("pointer_path") or "")
            pointer_path = (root / pointer_path_value).resolve()
            try:
                relative_pointer = pointer_path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ResearchGitError(
                    f"result artifact pointer escapes the repository: {label}"
                ) from exc
            if (
                pointer_path_value != relative_pointer
                or pointer_path.parent != pointer_root
                or relative_pointer in bound_paths
            ):
                raise ResearchGitError(
                    f"result artifact pointer is outside its configured directory: {label}"
                )
            if transaction_planned_bindings is not None and (
                transaction_planned_bindings.get(relative_pointer)
                != (object_hash, str(binding.get("logical_path") or ""))
            ):
                raise ResearchGitError(
                    f"result artifact binding differs from its transaction plan: {label}"
                )
            bound_paths.add(relative_pointer)

            try:
                encoded = read_bounded_regular_file(
                    pointer_path,
                    maximum=_MAX_RESULT_ARTIFACT_POINTER_BYTES,
                    label="result_artifact_pointer",
                )
                pointer = _validate_pointer_payload(
                    json.loads(encoded.decode("utf-8")),
                    pointer_path=relative_pointer,
                )
            except (
                BoundedFileError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ResearchGitError,
            ) as exc:
                raise ResearchGitError(
                    f"result artifact pointer is unsafe or invalid: {label}"
                ) from exc
            if pointer.get("object_hash") != object_hash or pointer.get(
                "logical_path"
            ) != binding.get("logical_path"):
                raise ResearchGitError(
                    f"result artifact pointer does not match its payload binding: {label}"
                )
            store_path, store_error = _verify_pointer_object(
                root,
                object_hash,
                pointer,
            )
            if store_error is not None or store_path is None:
                raise ResearchGitError(
                    f"result artifact CAS verification failed: {label}"
                )

            result = (
                supplied_results.get(relative_pointer)
                if supplied_results is not None
                else None
            )
            if supplied_results is not None and result is None:
                raise ResearchGitError(
                    f"result artifact pointer has no transaction result: {label}"
                )
            tracked_in_index = relative_pointer in {
                path
                for path in _run_git(
                    root,
                    ["ls-files", "-z", "--", relative_pointer],
                ).stdout.split("\0")
                if path
            }
            untracked = {
                path
                for path in _run_git(
                    root,
                    [
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                        "-z",
                        "--",
                        relative_pointer,
                    ],
                ).stdout.split("\0")
                if path
            }
            exists_at_head = (
                _run_git(
                    root,
                    ["cat-file", "-e", f"{current_head}:{relative_pointer}"],
                    check=False,
                ).returncode
                == 0
            )
            path_is_allowed_new = (
                allowed_new_paths is not None and relative_pointer in allowed_new_paths
            )
            if result is None and path_is_allowed_new:
                if (
                    tracked_in_index
                    or exists_at_head
                    or untracked != {relative_pointer}
                ):
                    raise ResearchGitError(
                        f"new result artifact pointer is not transaction-local: {label}"
                    )
                new_untracked_paths.add(relative_pointer)
                continue
            if result is None:
                if not tracked_in_index or not exists_at_head or untracked:
                    raise ResearchGitError(
                        f"result artifact pointer is not committed at HEAD: {label}"
                    )
                if (
                    _run_git(
                        root,
                        ["diff", "--quiet", current_head, "--", relative_pointer],
                        check=False,
                    ).returncode
                    != 0
                ):
                    raise ResearchGitError(
                        f"result artifact pointer differs from committed HEAD: {label}"
                    )
                continue

            if (
                result.object_hash != object_hash
                or result.size != pointer.get("size")
                or result.store_path.resolve() != store_path.resolve()
            ):
                raise ResearchGitError(
                    f"result artifact transaction result does not match disk: {label}"
                )
            if result.linked:
                if (
                    relative_pointer not in transaction_preexisting_paths
                    or not tracked_in_index
                    or not exists_at_head
                    or untracked
                ):
                    raise ResearchGitError(
                        f"linked result artifact pointer is not committed at HEAD: {label}"
                    )
                if (
                    _run_git(
                        root,
                        ["diff", "--quiet", current_head, "--", relative_pointer],
                        check=False,
                    ).returncode
                    != 0
                ):
                    raise ResearchGitError(
                        f"linked result artifact pointer differs from HEAD: {label}"
                    )
            else:
                if (
                    relative_pointer in transaction_preexisting_paths
                    or tracked_in_index
                    or exists_at_head
                    or untracked != {relative_pointer}
                ):
                    raise ResearchGitError(
                        f"new result artifact pointer is not transaction-local: {label}"
                    )
                new_untracked_paths.add(relative_pointer)

        if supplied_results is not None and set(supplied_results) != bound_paths:
            raise ResearchGitError(
                "result artifact transaction and payload bindings are not identical"
            )
        if (
            transaction_planned_bindings is not None
            and set(transaction_planned_bindings) != bound_paths
        ):
            raise ResearchGitError(
                "planned result artifact pointers and payload bindings are not identical"
            )
        if allowed_new_paths is not None and allowed_new_paths != new_untracked_paths:
            raise ResearchGitError(
                "sealed result artifact pending paths no longer match the repository"
            )
        return new_untracked_paths

    def _validate_experiment_attempt_with_result_artifacts(
        self,
        prepared: Mapping[str, Any],
        result_artifacts: Mapping[str, Any],
        *,
        pointer_results: Sequence[ObjectPointerResult],
        pointer_transaction: _ResultArtifactPointerTransaction,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        priority_id: str | None = None,
        admission_gate_id: str | None = None,
        admission_preflight: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Privately extend one sealed attempt with verified transaction pointers."""

        if type(prepared) is not _ValidatedExperimentAttempt:
            raise ResearchGitError(
                "result artifact revalidation requires a sealed attempt contract"
            )
        contract = prepared._materialize(
            repository_path=self.repository.path,
            issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
        )
        if "result_artifacts" in contract["payload"]:
            raise ResearchGitError(
                "sealed attempt contract already contains result artifact bindings"
            )
        candidate_payload = deepcopy(contract["payload"])
        candidate_payload["result_artifacts"] = deepcopy(dict(result_artifacts))
        allowed_pending_paths = self._validate_result_artifact_bindings(
            candidate_payload,
            pointer_results=pointer_results,
            pointer_transaction=pointer_transaction,
        )
        validation_kwargs = {
            "preregistration_id": preregistration_id,
            "plan_id": plan_id,
            "priority_id": priority_id,
            "admission_gate_id": admission_gate_id,
            "admission_preflight": admission_preflight,
            "provenance": contract["provenance"],
        }
        if candidate_payload.get("study_phase") == "confirmatory":
            candidate = self._prepare_experiment_attempt(
                candidate_payload,
                **validation_kwargs,
                allowed_pending_paths=allowed_pending_paths,
            )
        else:
            publicly_validated = self.validate_experiment_attempt(
                candidate_payload,
                **validation_kwargs,
            )
            candidate = publicly_validated._materialize(
                repository_path=self.repository.path,
                issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
            )
        return _ValidatedExperimentAttempt(
            candidate,
            repository_path=self.repository.path,
            issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
            pending_result_paths=allowed_pending_paths,
        )

    def _revalidate_sealed_result_artifacts(
        self,
        prepared: Mapping[str, Any],
    ) -> None:
        """Recheck pointer bytes immediately before the checkpoint boundary."""

        if type(prepared) is not _ValidatedExperimentAttempt:
            raise ResearchGitError(
                "result artifact revalidation requires a sealed attempt contract"
            )
        contract = prepared._materialize(
            repository_path=self.repository.path,
            issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
        )
        pending_result_paths = prepared._pending_paths(
            repository_path=self.repository.path,
            issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
        )
        self._validate_result_artifact_bindings(
            contract["payload"],
            allowed_new_paths=pending_result_paths,
        )

    def _validate_experiment_transition(
        self,
        payload: dict[str, Any],
        *,
        plan_id: str | None,
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Validate the explicit handoff from prior experiments.

        A transition is optional for legacy/first experiments. Once supplied,
        it becomes a strict contract: every parent must be terminal, remain in
        the same plan lineage when both sides name a plan, and be covered by a
        reused attempt/evidence/asset or one of its recorded artifact hashes.
        """

        raw_transition = payload.get("transition")
        if raw_transition is None:
            return [], []
        if not isinstance(raw_transition, Mapping):
            raise ResearchGitError("experiment transition must be an object")

        protocol = str(raw_transition.get("protocol") or "").strip()
        if protocol != _EXPERIMENT_TRANSITION_PROTOCOL:
            raise ResearchGitError("experiment transition protocol is invalid")
        transition_kind = str(raw_transition.get("kind") or "").strip().lower()
        if transition_kind not in _EXPERIMENT_TRANSITION_KINDS:
            raise ResearchGitError(
                "experiment transition kind must be one of: "
                + ", ".join(sorted(_EXPERIMENT_TRANSITION_KINDS))
            )
        rationale = str(raw_transition.get("rationale") or "").strip()
        if not rationale or len(rationale) > 2000:
            raise ResearchGitError(
                "experiment transition requires a rationale of 1-2000 characters"
            )

        def normalize_ids(
            value: Any,
            *,
            label: str,
            maximum: int,
            allow_empty: bool = False,
        ) -> list[str]:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ResearchGitError(f"experiment transition {label} must be an array")
            result: list[str] = []
            seen: set[str] = set()
            for raw in value:
                selector = str(raw or "").strip()
                if not selector:
                    raise ResearchGitError(
                        f"experiment transition {label} contains an empty selector"
                    )
                object_id = self.repository.resolve(selector)
                if object_id in seen:
                    raise ResearchGitError(
                        f"experiment transition {label} contains duplicates"
                    )
                seen.add(object_id)
                result.append(object_id)
            if (not allow_empty and not result) or len(result) > maximum:
                raise ResearchGitError(
                    f"experiment transition {label} must contain 1-{maximum} items"
                )
            return result

        parent_ids = normalize_ids(
            raw_transition.get("continues_from"),
            label="continues_from",
            maximum=8,
        )
        reuse_ids = normalize_ids(
            raw_transition.get("reused_object_ids"),
            label="reused_object_ids",
            maximum=32,
            allow_empty=True,
        )
        raw_hashes = raw_transition.get("reused_artifact_hashes") or []
        if isinstance(raw_hashes, (str, bytes)) or not isinstance(raw_hashes, Sequence):
            raise ResearchGitError(
                "experiment transition reused_artifact_hashes must be an array"
            )
        artifact_hashes: list[str] = []
        for raw_hash in raw_hashes:
            digest = str(raw_hash or "").strip()
            if _CONTENT_HASH_RE.fullmatch(digest) is None:
                raise ResearchGitError(
                    "experiment transition artifact hashes must use sha256:<64 lowercase hex>"
                )
            if digest in artifact_hashes:
                raise ResearchGitError(
                    "experiment transition reused_artifact_hashes contains duplicates"
                )
            artifact_hashes.append(digest)
        if len(artifact_hashes) > 64:
            raise ResearchGitError(
                "experiment transition may reuse at most 64 artifact hashes"
            )
        if not reuse_ids and not artifact_hashes:
            raise ResearchGitError(
                "experiment transition must declare reused objects or artifact hashes"
            )

        parents: dict[str, Mapping[str, Any]] = {}
        for parent_id in parent_ids:
            parent = self.repository.get(parent_id)
            if parent.get("kind") != "experiment_attempt":
                raise ResearchGitError(
                    "experiment transition parent must be an experiment attempt"
                )
            if parent.get("state") not in _EXPERIMENT_TRANSITION_TERMINAL_STATES:
                raise ResearchGitError(
                    "experiment transition parent must be a terminal experiment attempt"
                )
            parents[parent_id] = parent
            if plan_id:
                parent_plan_ids = {
                    str(relation.get("target") or "")
                    for relation in parent.get("relations") or []
                    if relation.get("type") == "depends_on"
                    and relation.get("role") in {"plan", "design"}
                }
                if plan_id not in parent_plan_ids:
                    raise ResearchGitError(
                        "experiment transition parent must belong to the same plan/design"
                    )

        reused_objects: dict[str, Mapping[str, Any]] = {}
        for object_id in reuse_ids:
            item = self.repository.get(object_id)
            if item.get("kind") not in _EXPERIMENT_TRANSITION_REUSE_KINDS:
                raise ResearchGitError(
                    "experiment transition can reuse only attempts, evidence, "
                    "observations, sources, passages, or context snapshots"
                )
            reused_objects[object_id] = item

        parent_artifacts: dict[str, set[str]] = {}
        for parent_id, parent in parents.items():
            parent_payload = parent.get("payload") or {}
            hashes = {
                str(value)
                for value in (parent_payload.get("result_artifact_hashes") or {}).values()
                if _CONTENT_HASH_RE.fullmatch(str(value))
            }
            bindings = parent_payload.get("result_artifacts") or {}
            if isinstance(bindings, Mapping):
                hashes.update(
                    str(item.get("content_hash"))
                    for item in bindings.values()
                    if isinstance(item, Mapping)
                    and _CONTENT_HASH_RE.fullmatch(str(item.get("content_hash")))
                )
            parent_artifacts[parent_id] = hashes

        def derived_from(item: Mapping[str, Any]) -> set[str]:
            return {
                str(relation.get("target") or "")
                for relation in item.get("relations") or []
                if relation.get("type") == "derived_from"
            }

        uncovered: list[str] = []
        for parent_id in parent_ids:
            covered = parent_id in reused_objects
            covered = covered or bool(parent_artifacts[parent_id] & set(artifact_hashes))
            covered = covered or any(
                parent_id in derived_from(item) for item in reused_objects.values()
            )
            if not covered:
                uncovered.append(parent_id)
        if uncovered:
            raise ResearchGitError(
                "experiment transition must reuse a result, evidence, or artifact "
                "from every parent attempt: " + ", ".join(uncovered)
            )

        transition_core = {
            "protocol": _EXPERIMENT_TRANSITION_PROTOCOL,
            "kind": transition_kind,
            "continues_from": sorted(parent_ids),
            "reused_object_ids": sorted(reuse_ids),
            "reused_artifact_hashes": sorted(artifact_hashes),
            "rationale": rationale,
        }
        supplied_hash = str(raw_transition.get("transition_hash") or "")
        expected_hash = canonical_content_hash(transition_core)
        if supplied_hash and supplied_hash != expected_hash:
            raise ResearchGitError("experiment transition integrity hash is invalid")
        payload["transition"] = {
            **transition_core,
            "transition_hash": expected_hash,
        }
        relations = [
            {"type": "depends_on", "target": parent_id, "role": "continues_from"}
            for parent_id in sorted(parent_ids)
        ]
        relations.extend(
            {"type": "consumes", "target": object_id, "role": "prior_result"}
            for object_id in sorted(reuse_ids)
        )
        return relations, parent_ids

    def _prepare_experiment_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        priority_id: str | None = None,
        admission_gate_id: str | None = None,
        admission_preflight: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        allowed_pending_paths: set[str] | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize an attempt without mutating the repository."""

        payload = dict(attempt)
        study_phase = str(payload.get("study_phase") or "").strip().lower()
        if study_phase:
            if study_phase not in {"exploratory", "confirmatory"}:
                raise ResearchGitError(
                    f"unsupported experiment study phase: {study_phase}"
                )
            payload["study_phase"] = study_phase
        raw_status = str(payload.get("status") or "completed").strip().lower()
        payload["status"] = raw_status
        state = {
            "success": "completed",
            "completed": "completed",
            "failed": "failed",
            "error": "failed",
            "timeout": "timed_out",
            "timed_out": "timed_out",
            "cancelled": "cancelled",
            "canceled": "cancelled",
            "rejected": "rejected",
            "orphan": "failed",
            "orphaned": "failed",
        }.get(raw_status)
        if state is None:
            if raw_status == "running":
                raise ResearchGitError(
                    "running attempts are mutable execution state and cannot be "
                    "persisted as immutable Research VCS objects; record a terminal "
                    "completed, failed, timed_out, cancelled, rejected, or orphaned "
                    "receipt"
                )
            raise ResearchGitError(
                f"unsupported experiment attempt status: {raw_status}"
            )
        registration_object: Mapping[str, Any] | None = None
        plan_object: Mapping[str, Any] | None = None
        if payload.get("study_phase") == "confirmatory" and not preregistration_id:
            raise ResearchGitError(
                "confirmatory experiment requires a locked preregistration"
            )
        relations = []
        if preregistration_id:
            registration = self.repository.get(preregistration_id)
            registration_object = registration
            preregistration_id = str(registration.get("object_id") or "").strip()
            if registration["kind"] != "preregistration":
                raise ResearchGitError(
                    "experiment preregistration reference has wrong kind"
                )
            if (
                payload.get("study_phase") == "confirmatory"
                and registration["state"] != "locked"
            ):
                raise ResearchGitError(
                    "confirmatory experiment requires a locked preregistration"
                )
            if payload.get("study_phase") == "confirmatory":
                registration_payload = registration.get("payload") or {}
                registration_validation = validate_preregistration(
                    registration_payload,
                    require_locked=True,
                )
                if not registration_validation["ok"]:
                    raise ResearchGitError(
                        "confirmatory experiment preregistration failed integrity "
                        "validation: " + ", ".join(registration_validation["errors"])
                    )
                adaptive_freeze = registration_payload.get("adaptive_state_freeze")
                freeze_validation = registration_validation.get("adaptive_state_freeze")
                if isinstance(adaptive_freeze, dict) and isinstance(
                    freeze_validation, dict
                ):
                    lineage_attestation = self._confirmatory_lineage_attestation(
                        registration=registration,
                        preregistration_id=preregistration_id,
                        preregistration=registration_payload,
                        adaptive_freeze=adaptive_freeze,
                        allowed_pending_paths=allowed_pending_paths,
                    )
                    task_id = str(payload.get("task_id") or "").strip()
                    registered_task_ids = _registered_task_ids(registration_payload)
                    if len(registered_task_ids) > 1 and not task_id:
                        raise ResearchGitError(
                            "multi-task confirmatory experiment requires a locked "
                            "task id"
                        )
                    if task_id and task_id not in _registered_task_ids(
                        registration_payload
                    ):
                        raise ResearchGitError(
                            "confirmatory experiment task is not in the locked "
                            "preregistration"
                        )
                    expected_bindings = {
                        "adaptive_state_hash": adaptive_freeze.get("state_hash"),
                        "research_state_hash": adaptive_freeze.get(
                            "research_state_hash"
                        ),
                        "post_freeze_adaptation": False,
                        "preregistration_id": registration_payload.get(
                            "preregistration_id"
                        ),
                    }
                    data_policy = registration_payload.get("data_policy")
                    data_policy = (
                        data_policy if isinstance(data_policy, Mapping) else {}
                    )
                    for field in ("data_manifest_hash", "data_snapshot_id"):
                        expected_bindings[field] = data_policy.get(field)
                    if task_id:
                        outcome = next(
                            (
                                item
                                for item in registration_payload.get("outcomes") or []
                                if isinstance(item, Mapping)
                                and str(item.get("task_id") or "").strip() == task_id
                            ),
                            {},
                        )
                        for field in (
                            "evidence_role",
                            "paired_control_task_id",
                            "intervention_variant",
                            "stress_condition",
                            "transformation_contract",
                            "transformation_contract_hash",
                        ):
                            expected_bindings[field] = outcome.get(field)
                        expected_bindings["protocol_fidelity_hash"] = (
                            _protocol_fidelity_hash(registration_payload, task_id)
                        )
                        expected_bindings["dataset_split_hash"] = (
                            data_policy.get("split_hashes") or {}
                        ).get(task_id)
                    campaign = registration_payload.get("confirmatory_campaign")
                    campaign = campaign if isinstance(campaign, Mapping) else {}
                    if campaign.get("schema") == "xscientist.confirmatory-campaign.v1":
                        if not str(payload.get("producer_id") or "").strip():
                            raise ResearchGitError(
                                "campaign experiment requires an explicit producer id"
                            )
                        if state == "completed":
                            if not isinstance(payload.get("configuration"), Mapping):
                                raise ResearchGitError(
                                    "completed campaign experiment requires its exact "
                                    "configuration"
                                )
                            if payload.get(
                                "configuration_hash"
                            ) != canonical_content_hash(dict(payload["configuration"])):
                                raise ResearchGitError(
                                    "completed campaign experiment configuration hash "
                                    "is invalid"
                                )
                            artifact_hashes = payload.get("result_artifact_hashes")
                            if (
                                not isinstance(artifact_hashes, Mapping)
                                or not artifact_hashes
                            ):
                                raise ResearchGitError(
                                    "completed campaign experiment requires at least one "
                                    "content-addressed result artifact"
                                )
                        elif (
                            state in {"failed", "timed_out", "cancelled", "rejected"}
                            and not str(payload.get("failure_class") or "").strip()
                        ):
                            raise ResearchGitError(
                                "unsuccessful campaign experiment requires a failure class"
                            )
                    conflicts = [
                        field
                        for field, expected in expected_bindings.items()
                        if field in payload and payload.get(field) != expected
                    ]
                    if conflicts:
                        raise ResearchGitError(
                            "confirmatory experiment conflicts with the frozen "
                            "research state: " + ", ".join(sorted(conflicts))
                        )
                    payload.update(expected_bindings)
                    payload["frozen_path_attestation"] = lineage_attestation
                    provenance_payload = dict(provenance or {})
                    frozen_code_commit = str(
                        adaptive_freeze.get("research_vcs_head") or ""
                    ).strip()
                    supplied_code_commit = str(
                        provenance_payload.get("code_commit") or ""
                    ).strip()
                    if (
                        supplied_code_commit
                        and supplied_code_commit != frozen_code_commit
                    ):
                        raise ResearchGitError(
                            "confirmatory experiment conflicts with the frozen code "
                            "state: code_commit"
                        )
                    provenance_payload["code_commit"] = frozen_code_commit
                    provenance = provenance_payload
                else:
                    raise ResearchGitError(
                        "confirmatory experiment requires a host-attested adaptive "
                        "state freeze"
                    )
            relations.append(
                {"type": "depends_on", "target": preregistration_id, "role": "protocol"}
            )
        if plan_id:
            plan = self.repository.get(plan_id)
            plan_object = plan
            if plan["kind"] not in {"research_plan", "experiment_design"}:
                raise ResearchGitError("experiment plan reference has wrong kind")
            plan_id = str(plan["object_id"])
            plan_role = "design" if plan["kind"] == "experiment_design" else "plan"
            relations.append(
                {"type": "depends_on", "target": plan_id, "role": plan_role}
            )
            if (
                plan["kind"] == "experiment_design"
                and (plan.get("payload") or {}).get("protocol_kind")
                == "competitive_experiment_candidate"
            ):
                if not priority_id:
                    raise ResearchGitError(
                        "competitive experiment design requires its locked priority"
                    )
                priority = self.repository.get(priority_id)
                priority_payload = priority.get("payload") or {}
                if (
                    priority.get("kind") != "experiment_priority"
                    or priority.get("state") != "locked"
                    or priority_payload.get("selected_design_id") != plan_id
                ):
                    raise ResearchGitError(
                        "experiment priority does not select the supplied design"
                    )
                relations.append(
                    {
                        "type": "consumes",
                        "target": str(priority["object_id"]),
                        "role": "selected_priority",
                    }
                )
                payload["priority_id"] = str(priority["object_id"])
                payload["design_id"] = plan_id
        elif priority_id:
            raise ResearchGitError(
                "experiment priority requires a selected plan/design"
            )
        transition_relations, _ = self._validate_experiment_transition(
            payload,
            plan_id=plan_id,
        )
        relations.extend(transition_relations)
        validated_receipt, execution_claim = self._validated_terminal_admission_receipt(
            payload,
            raw_status=raw_status,
            admission_preflight=admission_preflight,
        )
        current_policy_error: ResearchGitError | None = None
        try:
            payload, resolved_admission_gate_id = self._validate_experiment_admission(
                payload,
                raw_status=raw_status,
                plan=plan_object,
                preregistration=registration_object,
                admission_gate_id=admission_gate_id,
                provenance=provenance,
            )
        except ResearchGitError as exc:
            if validated_receipt is None or execution_claim is None:
                raise
            current_policy_error = exc
            payload, resolved_admission_gate_id = (
                self._validate_claimed_terminal_admission(
                    payload,
                    plan=plan_object,
                    preregistration=registration_object,
                    admission_gate_id=admission_gate_id,
                    provenance=provenance,
                    receipt=validated_receipt,
                )
            )
        payload = self._bind_experiment_admission_preflight(
            payload,
            raw_status=raw_status,
            plan=plan_object,
            preregistration=registration_object,
            admission_gate_id=resolved_admission_gate_id,
            provenance=provenance,
            admission_preflight=admission_preflight,
            validated_receipt=validated_receipt,
        )
        if execution_claim is not None:
            payload["admitted_at_start"] = True
            payload["current_policy_status"] = (
                "authorized"
                if current_policy_error is None
                else self._post_claim_policy_status(current_policy_error)
            )
            payload["quarantined"] = current_policy_error is not None
        if resolved_admission_gate_id is not None:
            relations.append(
                {
                    "type": "depends_on",
                    "target": resolved_admission_gate_id,
                    "role": "experiment_admission",
                }
            )
        bound_start_id = str(
            (payload.get("admission_preflight_receipt") or {}).get("start_object_id")
            or ""
        )
        if bound_start_id:
            relations.append(
                {
                    "type": "consumes",
                    "target": bound_start_id,
                    "role": "experiment_admission_start",
                }
            )
        producer_id = str(payload.get("producer_id") or "").strip()
        producer_actor = None
        if producer_id:
            try:
                canonical_principal(producer_id, label="experiment producer_id")
            except ValueError as exc:
                raise ResearchGitError(
                    "experiment producer_id is not a valid principal"
                ) from exc
            # Bind the immutable object's actor to the producer declared in the
            # scientific payload.  Role labels remain metadata; independence is
            # evaluated on the canonical principal.
            producer_actor = {
                "actor_id": producer_id,
                "authority": "research_agent",
            }
        return {
            "payload": payload,
            "state": state,
            "relations": relations,
            "actor": producer_actor,
            "provenance": provenance,
        }

    def validate_experiment_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        priority_id: str | None = None,
        admission_gate_id: str | None = None,
        admission_preflight: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return a sealed attempt contract without recording any object."""

        return _ValidatedExperimentAttempt(
            self._prepare_experiment_attempt(
                attempt,
                preregistration_id=preregistration_id,
                plan_id=plan_id,
                priority_id=priority_id,
                admission_gate_id=admission_gate_id,
                admission_preflight=admission_preflight,
                provenance=provenance,
            ),
            repository_path=self.repository.path,
            issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
        )

    def experiment_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        priority_id: str | None = None,
        admission_gate_id: str | None = None,
        admission_preflight: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record every attempt, including failure, timeout, and cancellation."""

        prepared = self.validate_experiment_attempt(
            attempt,
            preregistration_id=preregistration_id,
            plan_id=plan_id,
            priority_id=priority_id,
            admission_gate_id=admission_gate_id,
            admission_preflight=admission_preflight,
            provenance=provenance,
        )
        return self.record_validated_experiment_attempt(prepared, commit=commit)

    def _committed_experiment_admission_terminal(
        self,
        result: ResearchObjectResult,
    ) -> CheckpointResult | None:
        """Return the exact terminal checkpoint for an idempotent retry."""

        relative_attempt = (
            result.path.resolve().relative_to(self.repository.path).as_posix()
        )
        retained = _load_experiment_admission_terminal_completion_ref(
            self.repository.path,
            start_object_id=next(
                (
                    str(item.get("target") or "")
                    for item in self.repository.get(result.object_id).get("relations")
                    or []
                    if item.get("type") == "consumes"
                    and item.get("role") == "experiment_admission_start"
                ),
                "",
            ),
        )
        history = _run_git(
            self.repository.path,
            [
                "log",
                "--all",
                "--full-history",
                "--topo-order",
                "--no-renames",
                "--max-count=2",
                "--format=%H",
                "--",
                relative_attempt,
            ],
        ).stdout.splitlines()
        if not history:
            return None
        if len(history) != 1 or re.fullmatch(r"[0-9a-f]{40,64}", history[0]) is None:
            raise ResearchGitError(
                "experiment admission terminal object has ambiguous Git history"
            )
        introduction = history[0]
        if retained is not None and retained != introduction:
            raise ResearchGitError(
                "experiment admission terminal completion ref does not match its "
                "unique terminal checkpoint"
            )
        shown = show_checkpoint(self.repository.path, introduction)
        checkpoint = shown.get("checkpoint")
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        changed_paths = checkpoint.get("changed_paths")
        committed_objects = {
            str(item.get("object_id") or ""): item
            for item in list_research_objects_at_ref(
                self.repository.path,
                introduction,
            )
        }
        committed_attempt = committed_objects.get(result.object_id)
        if (
            shown.get("checkpoint_hash_valid") is not True
            or checkpoint.get("stage") != "experiment_admission_terminal"
            or checkpoint.get("status") != result.state
            or not isinstance(changed_paths, list)
            or relative_attempt not in changed_paths
            or len(set(changed_paths)) != len(changed_paths)
            or any(
                (match := _RESEARCH_OBJECT_PATH.fullmatch(str(path))) is None
                or match.group("kind") != "experiment_attempt"
                for path in changed_paths
            )
            or not isinstance(committed_attempt, Mapping)
            or committed_attempt.get("content_hash") != result.object_hash
        ):
            raise ResearchGitError(
                "experiment admission terminal object was not introduced by a "
                "valid terminal-only checkpoint"
            )
        checkpoint_path = shown.get("path")
        return CheckpointResult(
            created=False,
            committed=True,
            checkpoint_path=(
                self.repository.path / str(checkpoint_path)
                if isinstance(checkpoint_path, str) and checkpoint_path
                else None
            ),
            commit=introduction,
            checkpoint_id=str(checkpoint.get("checkpoint_id") or "") or None,
            content_hash=str(checkpoint.get("content_hash") or "") or None,
            staged_paths=tuple(str(path) for path in changed_paths),
            reason="experiment admission terminal checkpoint already committed",
        )

    def record_validated_experiment_attempt(
        self,
        prepared: Mapping[str, Any],
        *,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record a contract returned by ``validate_experiment_attempt`` once."""

        if type(prepared) is not _ValidatedExperimentAttempt:
            raise ResearchGitError(
                "validated experiment attempt must be returned by "
                "validate_experiment_attempt"
            )
        contract = prepared._materialize(
            repository_path=self.repository.path,
            issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
        )
        pending_result_paths = prepared._pending_paths(
            repository_path=self.repository.path,
            issuer=_VALIDATED_EXPERIMENT_ATTEMPT_ISSUER,
        )
        checkpoint: CheckpointResult | None = None
        with _repository_lock(self.repository.path):
            admission_required = self._experiment_admission_required()
            payload = contract["payload"]
            self._validate_result_artifact_bindings(
                payload,
                allowed_new_paths=pending_result_paths,
            )
            relations = contract["relations"]
            plan_targets = {
                str(item.get("target") or "")
                for item in relations
                if item.get("type") == "depends_on"
                and item.get("role") in {"plan", "design"}
            }
            protocol_targets = {
                str(item.get("target") or "")
                for item in relations
                if item.get("type") == "depends_on" and item.get("role") == "protocol"
            }
            admission_targets = {
                str(item.get("target") or "")
                for item in relations
                if item.get("type") == "depends_on"
                and item.get("role") == "experiment_admission"
            }
            admission_start_targets = {
                str(item.get("target") or "")
                for item in relations
                if item.get("type") == "consumes"
                and item.get("role") == "experiment_admission_start"
            }
            if len(plan_targets) > 1 or len(protocol_targets) > 1:
                raise ResearchGitError(
                    "validated experiment attempt has ambiguous protocol bindings"
                )
            if len(admission_targets) > 1:
                raise ResearchGitError(
                    "validated experiment attempt has ambiguous admission bindings"
                )
            if len(admission_start_targets) > 1:
                raise ResearchGitError(
                    "validated experiment attempt has ambiguous admission starts"
                )
            plan = (
                self.repository.get(next(iter(plan_targets))) if plan_targets else None
            )
            preregistration = (
                self.repository.get(next(iter(protocol_targets)))
                if protocol_targets
                else None
            )
            raw_status = str(payload.get("status") or "").strip().lower()
            validated_receipt, execution_claim = (
                self._validated_terminal_admission_receipt(
                    payload,
                    raw_status=raw_status,
                    admission_preflight=None,
                )
            )
            requested_admission_gate_id = (
                next(iter(admission_targets)) if admission_targets else None
            )
            current_policy_error: ResearchGitError | None = None
            try:
                payload, resolved_admission_gate_id = (
                    self._validate_experiment_admission(
                        payload,
                        raw_status=raw_status,
                        plan=plan,
                        preregistration=preregistration,
                        admission_gate_id=requested_admission_gate_id,
                        provenance=contract["provenance"],
                    )
                )
            except ResearchGitError as exc:
                if validated_receipt is None or execution_claim is None:
                    raise
                current_policy_error = exc
                payload, resolved_admission_gate_id = (
                    self._validate_claimed_terminal_admission(
                        payload,
                        plan=plan,
                        preregistration=preregistration,
                        admission_gate_id=requested_admission_gate_id,
                        provenance=contract["provenance"],
                        receipt=validated_receipt,
                    )
                )
            payload = self._bind_experiment_admission_preflight(
                payload,
                raw_status=raw_status,
                plan=plan,
                preregistration=preregistration,
                admission_gate_id=resolved_admission_gate_id,
                provenance=contract["provenance"],
                admission_preflight=None,
                validated_receipt=validated_receipt,
            )
            if execution_claim is not None:
                payload["admitted_at_start"] = True
                payload["current_policy_status"] = (
                    "authorized"
                    if current_policy_error is None
                    else self._post_claim_policy_status(current_policy_error)
                )
                payload["quarantined"] = current_policy_error is not None
            if resolved_admission_gate_id is not None and admission_targets != {
                resolved_admission_gate_id
            }:
                raise ResearchGitError(
                    "validated experiment attempt is missing its admission relation"
                )
            if resolved_admission_gate_id is None and admission_targets:
                raise ResearchGitError(
                    "validated experiment attempt has an invalid admission relation"
                )
            bound_start_id = str(
                (payload.get("admission_preflight_receipt") or {}).get(
                    "start_object_id"
                )
                or ""
            )
            terminal_policy_snapshot: dict[str, Any] | None = None
            reservation_preexisted = False
            if bound_start_id:
                existing_reservation = _load_experiment_admission_terminal_reservation(
                    self.repository.path,
                    start_object_id=bound_start_id,
                )
                reservation_preexisted = existing_reservation is not None
                reserved_policy = (
                    existing_reservation.get("terminal_policy_snapshot")
                    if existing_reservation is not None
                    else None
                )
                if isinstance(reserved_policy, Mapping):
                    terminal_policy_snapshot = dict(reserved_policy)
                    for field in _EXPERIMENT_ADMISSION_TERMINAL_POLICY_FIELDS:
                        payload[field] = terminal_policy_snapshot[field]
                elif execution_claim is not None:
                    terminal_policy_snapshot = {
                        "schema": (
                            _EXPERIMENT_ADMISSION_TERMINAL_POLICY_SNAPSHOT_PROTOCOL
                        ),
                        **{
                            field: payload[field]
                            for field in _EXPERIMENT_ADMISSION_TERMINAL_POLICY_FIELDS
                        },
                    }
            if admission_required and bound_start_id and execution_claim is None:
                raise ResearchGitError(
                    "experiment admission start has no durable execution claim"
                )
            if admission_start_targets != (
                {bound_start_id} if bound_start_id else set()
            ):
                raise ResearchGitError(
                    "validated experiment attempt is missing its admission start relation"
                )
            state = str(contract["state"])
            reservation_path: Path | None = None
            result: ResearchObjectResult | None = None
            if bound_start_id:
                try:
                    attempt_identity = build_research_object(
                        kind="experiment_attempt",
                        payload=payload,
                        state=state,
                        relations=relations,
                        actor=contract["actor"],
                        provenance=contract["provenance"],
                    )
                except ResearchObjectError as exc:
                    raise ResearchGitError(str(exc)) from exc
                bound_receipt = payload.get("admission_preflight_receipt") or {}
                reservation_path = _reserve_experiment_admission_terminal(
                    self.repository.path,
                    start_object_id=bound_start_id,
                    receipt_hash=str(bound_receipt.get("receipt_hash") or ""),
                    attempt_object_id=str(attempt_identity["object_id"]),
                    attempt_object_hash=str(attempt_identity["content_hash"]),
                    terminal_policy_snapshot=terminal_policy_snapshot,
                )
            try:
                result = self.repository._record(
                    "experiment_attempt",
                    payload,
                    state=state,
                    relations=relations,
                    actor=contract["actor"],
                    provenance=contract["provenance"],
                    _issuer=self.repository._validated_attempt_issuer,
                )
                if bound_start_id and (
                    result.object_id != attempt_identity["object_id"]
                    or result.object_hash != attempt_identity["content_hash"]
                ):
                    raise ResearchGitError(
                        "reserved experiment attempt identity changed before "
                        "persistence"
                    )
                if commit:
                    checkpoint_stage = (
                        "experiment_admission_terminal"
                        if admission_required
                        else ("experiment" if state == "completed" else "failed")
                    )
                    relative_attempt = (
                        result.path.resolve()
                        .relative_to(self.repository.path)
                        .as_posix()
                    )
                    checkpoint = (
                        self._committed_experiment_admission_terminal(result)
                        if admission_required and bound_start_id
                        else None
                    )
                    if checkpoint is not None and result.created:
                        result.path.unlink(missing_ok=True)
                        _fsync_directory(result.path.parent)
                        result = ResearchObjectResult(
                            created=False,
                            path=result.path,
                            object_id=result.object_id,
                            qualified_id=result.qualified_id,
                            object_hash=result.object_hash,
                            kind=result.kind,
                            state=result.state,
                        )
                    if checkpoint is None:
                        checkpoint = create_checkpoint(
                            self.repository.path,
                            stage=checkpoint_stage,
                            subject=f"record {state} experiment attempt",
                            status=state,
                            only_paths=[relative_attempt],
                        )
                    if not checkpoint.committed or not checkpoint.commit:
                        raise ResearchGitError(
                            "experiment attempt checkpoint was not committed"
                        )
                    if admission_required and bound_start_id:
                        _publish_experiment_admission_terminal_completion_ref(
                            self.repository.path,
                            start_object_id=bound_start_id,
                            commit=checkpoint.commit,
                        )
            except BaseException:
                committed_at_head = False
                if commit and result is not None:
                    committed_at_head = True
                    try:
                        committed_at_head = any(
                            item.get("object_id") == result.object_id
                            and item.get("content_hash") == result.object_hash
                            for item in list_research_objects_at_ref(
                                self.repository.path, "HEAD"
                            )
                        )
                    except BaseException:
                        # Preserve both durable records unless HEAD positively proves
                        # that the failed checkpoint did not commit the attempt.
                        pass
                if not committed_at_head:
                    if result is not None and result.created:
                        result.path.unlink(missing_ok=True)
                        _fsync_directory(result.path.parent)
                    if reservation_path is not None and not reservation_preexisted:
                        reservation_path.unlink(missing_ok=True)
                        _fsync_directory(reservation_path.parent)
                raise
        if result is None:  # pragma: no cover - defensive transaction invariant
            raise ResearchGitError("experiment attempt was not recorded")
        return {
            "attempt": result,
            "checkpoint": checkpoint,
            "terminal_reservation": reservation_path,
        }

    def evidence(
        self,
        payload: Mapping[str, Any],
        *,
        attempt_ids: Sequence[str],
        supports: Sequence[str] = (),
        refutes: Sequence[str] = (),
        verified: bool = False,
        verifier_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not attempt_ids:
            raise ResearchGitError("evidence requires at least one experiment attempt")
        if verified and not str(verifier_id or "").strip():
            raise ResearchGitError("verified evidence requires verifier_id")
        relations: list[dict[str, str]] = []
        for object_id in attempt_ids:
            attempt = self.repository.get(object_id)
            if attempt["kind"] != "experiment_attempt":
                raise ResearchGitError("evidence attempt reference has wrong kind")
            relations.append(
                {"type": "derived_from", "target": str(attempt["object_id"])}
            )
        independence = None
        if verified:
            independence = require_independent_evaluator(
                self.repository,
                evaluator_id=str(verifier_id or ""),
                target_ids=attempt_ids,
                label="verified evidence",
            )
        for object_id in supports:
            self.repository.get(object_id)
            relations.append({"type": "supports", "target": object_id})
        for object_id in refutes:
            self.repository.get(object_id)
            relations.append({"type": "refutes", "target": object_id})
        evidence_payload = dict(payload)
        if independence is not None:
            evidence_payload["independence"] = independence
        result = self.repository.record(
            "evidence",
            evidence_payload,
            state="verified" if verified else "completed",
            relations=relations,
            actor=(
                {
                    "actor_id": str(verifier_id),
                    "authority": "independent_evaluator",
                }
                if verified
                else None
            ),
        )
        checkpoint = (
            self.repository.commit(
                stage="evidence",
                subject="bind experiment evidence",
                status=result.state,
            )
            if commit
            else None
        )
        return {"evidence": result, "checkpoint": checkpoint}

    def evaluation(
        self,
        report: Mapping[str, Any],
        *,
        evaluates: Sequence[str],
        verifier_id: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not verifier_id.strip():
            raise ResearchGitError("independent verifier_id is required")
        report_payload = dict(report)
        requested_verified = (
            report_payload.get("status") == "verified"
            and report_payload.get("claim_promotion_allowed") is True
            and not report_payload.get("required_failures")
        )
        relations = []
        for object_id in evaluates:
            evaluated = self.repository.get(object_id)
            relations.append(
                {"type": "evaluates", "target": str(evaluated["object_id"])}
            )
        independence = require_independent_evaluator(
            self.repository,
            evaluator_id=verifier_id,
            target_ids=evaluates,
            label="independent review",
        )
        from .research_context import record_research_context_snapshot

        # This API records an in-workspace, declared-identity review.  It is
        # useful scientific criticism, but it is not an externally signed
        # authority receipt and therefore cannot authorize claim promotion.
        # Preserve what the reviewer requested for audit while forcing the
        # effective decision to hold.
        report_payload["requested_status"] = report_payload.get("status")
        report_payload["requested_claim_promotion_allowed"] = (
            report_payload.get("claim_promotion_allowed") is True
        )
        report_payload["status"] = "completed"
        report_payload["claim_promotion_allowed"] = False
        report_payload["authority_scope"] = "local_advisory"
        report_payload["external_authority_required"] = True
        advisory_failures = list(report_payload.get("required_failures") or [])
        if requested_verified and "external_authority_missing" not in advisory_failures:
            advisory_failures.append("external_authority_missing")
        report_payload["required_failures"] = advisory_failures
        selected_option = "hold"
        context = record_research_context_snapshot(
            self.repository,
            target_ids=list(evaluates),
            decision_kind="independent_evidence_review",
            selected=selected_option,
            options_considered=[
                {
                    "option": "promote",
                    "rejected_because": (
                        "local declared-identity review has advisory scope only"
                    ),
                },
                {
                    "option": "hold",
                    "rejected_because": (""),
                },
            ],
            rationale=[
                str(report_payload.get("summary") or "independent review decision")
            ],
            constraints=[
                str(value) for value in report_payload.get("required_failures") or []
            ],
            actor_id="research-review-context-recorder",
        )
        context_payload = self.repository.get(context.object_id)["payload"]
        report_payload["context_required"] = True
        report_payload["context_hash"] = context_payload["context_hash"]
        report_payload["independence"] = independence
        review = self.repository.record(
            "review",
            report_payload,
            state="completed",
            relations=[
                *relations,
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
            actor={
                "actor_id": verifier_id,
                "authority": "recorder",
            },
        )
        gate = self.repository.record(
            "gate_decision",
            {
                "decision": "hold",
                "claim_promotion_allowed": False,
                "required_failures": list(
                    report_payload.get("required_failures") or []
                ),
                "report_hash": report_payload.get("report_hash"),
                "context_required": True,
                "context_hash": context_payload["context_hash"],
            },
            state="rejected",
            relations=[
                {"type": "evaluates", "target": review.object_id},
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
            actor={
                "actor_id": "research-integrity-gate",
                "authority": "deterministic_gate",
            },
        )
        checkpoint = (
            self.repository.commit(
                stage="review",
                subject="record independent evidence gate",
                status=gate.state,
            )
            if commit
            else None
        )
        return {
            "context": context,
            "review": review,
            "gate": gate,
            "checkpoint": checkpoint,
        }

    def claim(
        self,
        payload: Mapping[str, Any],
        *,
        evidence_ids: Sequence[str],
        qualification_ids: Sequence[str] = (),
        gate_id: str | None = None,
        verified: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not evidence_ids:
            raise ResearchGitError("claim requires evidence")
        relations: list[dict[str, str]] = []
        resolved_evidence_ids: list[str] = []
        for object_id in evidence_ids:
            evidence = self.repository.get(object_id)
            if evidence["kind"] not in {
                "evidence",
                "passage_evidence",
                "inference",
                "evidence_synthesis",
            }:
                raise ResearchGitError("claim evidence reference has wrong kind")
            resolved_id = str(evidence["object_id"])
            resolved_evidence_ids.append(resolved_id)
            relations.append({"type": "depends_on", "target": resolved_id})
        qualifications: dict[str, list[dict[str, Any]]] = {
            "mechanism_model": [],
            "evidence_quality": [],
            "transfer_matrix": [],
        }
        role_by_kind = {
            "mechanism_model": "mechanism",
            "evidence_quality": "quality",
            "transfer_matrix": "transfer",
        }
        for object_id in qualification_ids:
            qualification = self.repository.get(object_id)
            kind = str(qualification.get("kind") or "")
            if kind not in qualifications:
                raise ResearchGitError("claim qualification reference has wrong kind")
            qualifications[kind].append(qualification)
            relations.append(
                {
                    "type": "depends_on",
                    "target": str(qualification["object_id"]),
                    "role": role_by_kind[kind],
                }
            )
        depth_level = str(payload.get("depth_level") or "descriptive")
        if depth_level not in {"descriptive", "causal", "transferable"}:
            raise ResearchGitError("claim depth_level is invalid")
        if verified and depth_level in {"causal", "transferable"}:
            valid_mechanisms = [
                item
                for item in qualifications["mechanism_model"]
                if item.get("state") == "verified"
                and (item.get("payload") or {}).get("status") == "validated"
                and (item.get("payload") or {}).get("validation")
                and set(
                    (item.get("payload") or {}).get("evidence_ids") or []
                ).intersection(resolved_evidence_ids)
            ]
            valid_quality = [
                item
                for item in qualifications["evidence_quality"]
                if item.get("state") == "verified"
                and (item.get("payload") or {}).get("independent") is True
                and (item.get("payload") or {}).get("independence_receipt")
                and (item.get("payload") or {}).get("overall_grade")
                in {"strong", "moderate"}
                and (item.get("payload") or {}).get("evidence_id")
                in resolved_evidence_ids
            ]
            if not valid_mechanisms:
                raise ResearchGitError(
                    "verified causal claim requires a validated intervention-tested "
                    "mechanism bound to its evidence"
                )
            if not valid_quality:
                raise ResearchGitError(
                    "verified causal claim requires an independent strong/moderate "
                    "quality assessment of its evidence"
                )
            if depth_level == "transferable":
                valid_transfer = []
                claim_scope_hash = payload.get("scope_hash")
                claim_statement = " ".join(str(payload.get("statement") or "").split())
                for item in qualifications["transfer_matrix"]:
                    matrix_payload = item.get("payload") or {}
                    if (
                        item.get("state") != "verified"
                        or matrix_payload.get("transfer_ready") is not True
                    ):
                        continue
                    matrix_claim = self.repository.get(
                        str(matrix_payload.get("claim_id") or "")
                    )
                    matrix_claim_payload = matrix_claim.get("payload") or {}
                    matrix_statement = " ".join(
                        str(matrix_claim_payload.get("statement") or "").split()
                    )
                    if (
                        matrix_statement != claim_statement
                        or matrix_claim_payload.get("scope_hash") != claim_scope_hash
                    ):
                        continue
                    valid_transfer.append(item)
                if not valid_transfer:
                    raise ResearchGitError(
                        "verified transferable claim requires a passing transfer matrix "
                        "for the same statement and scope"
                    )
        if payload.get("contribution_level") == "method_discovery" and verified:
            supported = any(
                item["kind"] == "evidence_synthesis"
                and (item.get("payload") or {}).get("protocol_kind")
                == "generalization_assessment"
                and (item.get("payload") or {}).get("verdict")
                == "method_discovery_supported"
                for item in (
                    self.repository.get(object_id)
                    for object_id in resolved_evidence_ids
                )
            )
            if not supported:
                raise ResearchGitError(
                    "verified method-discovery claim requires a passing "
                    "generalization assessment"
                )
        if verified:
            if not gate_id:
                raise ResearchGitError("verified claim requires a gate decision")
            gate = self.repository.get(gate_id)
            if gate["kind"] != "gate_decision" or gate["state"] != "verified":
                raise ResearchGitError(
                    "verified claim requires a passing gate decision"
                )
            evaluated = {
                str(item.get("target") or "")
                for item in gate.get("relations") or []
                if item.get("type") == "evaluates"
            }
            review_target_sets: list[set[str]] = []
            for target in list(evaluated):
                try:
                    linked = self.repository.get(target)
                except ResearchGitError:
                    continue
                if linked.get("kind") == "review":
                    review_targets = {
                        str(item.get("target") or "")
                        for item in linked.get("relations") or []
                        if item.get("type") == "evaluates"
                    }
                    review_target_sets.append(review_targets)
                    evaluated.update(review_targets)
            required_closure_ids = set(resolved_evidence_ids)
            required_closure_ids.update(
                str(item["object_id"])
                for rows in qualifications.values()
                for item in rows
            )
            if not required_closure_ids.issubset(evaluated):
                raise ResearchGitError(
                    "verified claim gate does not evaluate the complete selected "
                    "scientific closure"
                )
            if not any(
                required_closure_ids.issubset(review_targets)
                for review_targets in review_target_sets
            ):
                raise ResearchGitError(
                    "verified claim requires one review covering the complete "
                    "selected scientific closure"
                )
            relations.append({"type": "depends_on", "target": gate_id, "role": "gate"})
        result = self.repository.record(
            "claim",
            payload,
            state="verified" if verified else "draft",
            relations=relations,
        )
        checkpoint = (
            self.repository.commit(
                stage="evidence",
                subject="record evidence-bound claim",
                status=result.state,
            )
            if commit
            else None
        )
        return {"claim": result, "checkpoint": checkpoint}

    def manuscript(
        self,
        payload: Mapping[str, Any],
        *,
        claim_ids: Sequence[str],
        gate_id: str | None = None,
        final: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        relations: list[dict[str, str]] = []
        for object_id in claim_ids:
            claim = self.repository.get(object_id)
            if claim["kind"] != "claim":
                raise ResearchGitError("manuscript claim reference has wrong kind")
            if final and claim["state"] != "verified":
                raise ResearchGitError(
                    "final manuscript cannot include an unverified claim"
                )
            relations.append({"type": "depends_on", "target": object_id})
        if final and not gate_id:
            raise ResearchGitError("final manuscript requires a passing gate decision")
        if gate_id:
            gate = self.repository.get(gate_id)
            if gate["kind"] != "gate_decision":
                raise ResearchGitError("manuscript gate reference has wrong kind")
            if final and gate["state"] != "verified":
                raise ResearchGitError(
                    "final manuscript requires a passing gate decision"
                )
            relations.append({"type": "depends_on", "target": gate_id, "role": "gate"})
        result = self.repository.record(
            "manuscript",
            payload,
            state="completed" if final else "draft",
            relations=relations,
        )
        checkpoint = (
            self.repository.commit(
                stage="paper",
                subject=(
                    "freeze evidence-bound manuscript"
                    if final
                    else "record manuscript draft"
                ),
                status=result.state,
            )
            if commit
            else None
        )
        return {"manuscript": result, "checkpoint": checkpoint}

    def reproduction(
        self,
        receipt: Mapping[str, Any],
        *,
        reproduces: Sequence[str],
        verifier_id: str | None = None,
        verified: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Bind a compact reproduction receipt to the objects it checked."""

        from jsonschema import ValidationError, validate as validate_json

        from ai_scientist.protocol.schemas import load_schema

        if not reproduces:
            raise ResearchGitError("reproduction requires at least one target object")
        receipt_payload = dict(receipt)
        try:
            validate_json(receipt_payload, load_schema("reproduction_receipt"))
        except ValidationError as exc:
            raise ResearchGitError(
                f"invalid reproduction receipt: {exc.message}"
            ) from exc
        from ai_scientist.protocol import content_hash

        receipt_base = {
            key: value
            for key, value in receipt_payload.items()
            if key not in {"receipt_id", "content_hash"}
        }
        expected_hash = content_hash(receipt_base)
        expected_id = f"rr-{expected_hash.split(':', 1)[1][:16]}"
        if receipt_payload.get("content_hash") != expected_hash:
            raise ResearchGitError("reproduction receipt content hash mismatch")
        if receipt_payload.get("receipt_id") != expected_id:
            raise ResearchGitError("reproduction receipt identifier mismatch")
        reproduced_objects = [
            self.repository.get(object_id) for object_id in reproduces
        ]
        resolved_target_ids = sorted(
            {str(item["object_id"]) for item in reproduced_objects}
        )
        if (
            verified
            and receipt_payload.get("schema_version")
            == "xscientist.reproduction-receipt.v1"
        ):
            # Existing clients may still hand the lifecycle a locally generated
            # v1 execution receipt. Upgrade it before recording, while resolving
            # the checkpoint independently instead of trusting its copied fields.
            source_record = show_checkpoint(
                self.repository.path, str(receipt_payload.get("commit") or "")
            )
            source_checkpoint = source_record.get("checkpoint") or {}
            checkpoint_core = {
                "commit": str(source_record.get("commit") or ""),
                "checkpoint_id": str(source_checkpoint.get("checkpoint_id") or ""),
                "checkpoint_content_hash": str(
                    source_checkpoint.get("content_hash") or ""
                ),
            }
            if (
                receipt_payload.get("commit") != checkpoint_core["commit"]
                or receipt_payload.get("checkpoint_id")
                != checkpoint_core["checkpoint_id"]
                or receipt_payload.get("checkpoint_hash")
                != checkpoint_core["checkpoint_content_hash"]
            ):
                raise ResearchGitError(
                    "legacy reproduction receipt disagrees with its resolved checkpoint"
                )
            receipt_payload.update(
                {
                    "stdout_truncated": None,
                    "stderr_truncated": None,
                    "output_capture": "legacy_unknown",
                    "max_output_chars": None,
                }
            )
            execution_core = {
                key: receipt_payload.get(key)
                for key in (
                    "command_hash",
                    "reproduction_level",
                    "verdict",
                    "objects_complete",
                    "executed",
                    "returncode",
                    "timed_out",
                    "stdout_hash",
                    "stderr_hash",
                    "stdout_truncated",
                    "stderr_truncated",
                    "output_capture",
                    "max_output_chars",
                )
            }
            receipt_payload["schema_version"] = "xscientist.reproduction-receipt.v2"
            receipt_payload["checkpoint_binding"] = {
                **checkpoint_core,
                "binding_hash": content_hash(checkpoint_core),
            }
            receipt_payload["execution_result"] = {
                **execution_core,
                "result_hash": content_hash(execution_core),
            }
            # Historical v1 receipts did not persist execution-boundary
            # metadata. Preserve that uncertainty explicitly instead of
            # inventing a platform or claiming that the old run was sanitized.
            receipt_payload["execution_isolation"] = {
                "isolated": False,
                "security_boundary": False,
                "environment": "legacy_unknown",
                "environment_scope": "legacy_unknown",
                "process_tree": "legacy_unknown_no_tree_guarantee",
                "process_control": "legacy_unknown",
                "process_tree_termination_guaranteed": False,
                "filesystem": "host_visible",
                "network": "host_unrestricted",
            }
        if (
            receipt_payload.get("schema_version")
            == "xscientist.reproduction-receipt.v2"
        ):
            from .research_closure import build_reproduction_target_binding

            source_record = show_checkpoint(
                self.repository.path, str(receipt_payload.get("commit") or "")
            )
            source_checkpoint = source_record.get("checkpoint") or {}
            checkpoint_core = {
                "commit": str(source_record.get("commit") or ""),
                "checkpoint_id": str(source_checkpoint.get("checkpoint_id") or ""),
                "checkpoint_content_hash": str(
                    source_checkpoint.get("content_hash") or ""
                ),
            }
            expected_checkpoint_binding = {
                **checkpoint_core,
                "binding_hash": content_hash(checkpoint_core),
            }
            if (
                receipt_payload.get("commit") != checkpoint_core["commit"]
                or receipt_payload.get("checkpoint_id")
                != checkpoint_core["checkpoint_id"]
                or receipt_payload.get("checkpoint_hash")
                != checkpoint_core["checkpoint_content_hash"]
                or receipt_payload.get("checkpoint_binding")
                != expected_checkpoint_binding
            ):
                raise ResearchGitError(
                    "reproduction receipt checkpoint binding is inconsistent"
                )
            execution_core = {
                key: receipt_payload.get(key)
                for key in (
                    "command_hash",
                    "reproduction_level",
                    "verdict",
                    "objects_complete",
                    "executed",
                    "returncode",
                    "timed_out",
                    "stdout_hash",
                    "stderr_hash",
                    "stdout_truncated",
                    "stderr_truncated",
                    "output_capture",
                    "max_output_chars",
                )
            }
            expected_execution_result = {
                **execution_core,
                "result_hash": content_hash(execution_core),
            }
            if receipt_payload.get("execution_result") != expected_execution_result:
                raise ResearchGitError(
                    "reproduction receipt execution-result binding is inconsistent"
                )

            receipt_payload["target_binding"] = build_reproduction_target_binding(
                self.repository.path,
                resolved_target_ids,
                ref=str(receipt_payload.get("commit") or ""),
            )
            receipt_base = {
                key: value
                for key, value in receipt_payload.items()
                if key not in {"receipt_id", "content_hash"}
            }
            expected_hash = content_hash(receipt_base)
            receipt_payload["receipt_id"] = f"rr-{expected_hash.split(':', 1)[1][:16]}"
            receipt_payload["content_hash"] = expected_hash
            try:
                validate_json(receipt_payload, load_schema("reproduction_receipt"))
            except ValidationError as exc:  # pragma: no cover - contract guard
                raise ResearchGitError(
                    f"bound reproduction receipt is invalid: {exc.message}"
                ) from exc
        elif verified:  # pragma: no cover - schema currently permits only v1/v2
            raise ResearchGitError("verified reproduction requires a v2 receipt")
        if verified and not str(verifier_id or "").strip():
            raise ResearchGitError("verified reproduction requires verifier_id")
        if verified and (
            receipt_payload.get("verdict") != "passed"
            or receipt_payload.get("reproduction_level") != "computational_rerun"
            or receipt_payload.get("executed") is not True
            or receipt_payload.get("returncode") != 0
            or receipt_payload.get("timed_out") is True
            or receipt_payload.get("objects_complete") is not True
        ):
            raise ResearchGitError(
                "verified reproduction requires a successful computational rerun"
            )
        if verified and not any(
            item.get("kind") == "claim" for item in reproduced_objects
        ):
            raise ResearchGitError(
                "verified reproduction requires an exact reproduced claim target"
            )
        relations = [
            {"type": "reproduces", "target": object_id}
            for object_id in resolved_target_ids
        ]
        independence = None
        if verified:
            independence = require_independent_evaluator(
                self.repository,
                evaluator_id=str(verifier_id or ""),
                target_ids=resolved_target_ids,
                label="verified reproduction",
            )
        payload = {
            key: receipt_payload[key]
            for key in (
                "receipt_id",
                "content_hash",
                "checkpoint_hash",
                "commit",
                "reproduction_level",
                "verdict",
                "objects_complete",
                "executed",
                "returncode",
                "timed_out",
            )
        }
        payload["receipt_hash"] = payload.pop("content_hash")
        payload["receipt"] = receipt_payload
        if independence is not None:
            payload["independence"] = independence
        result = self.repository.record(
            "reproduction",
            payload,
            state=(
                "verified"
                if verified
                else (
                    "completed"
                    if receipt_payload.get("verdict") != "failed"
                    else "failed"
                )
            ),
            relations=relations,
            actor={
                "actor_id": str(verifier_id or "reproduction-recorder"),
                "authority": "independent_evaluator" if verified else "recorder",
            },
            provenance=(
                {
                    "environment_hash": str(
                        (receipt_payload.get("environment") or {}).get(
                            "recorded_content_hash"
                        )
                        or ""
                    )
                }
                if (receipt_payload.get("environment") or {}).get(
                    "recorded_content_hash"
                )
                else None
            ),
        )
        checkpoint = (
            self.repository.commit(
                stage="review",
                subject="record reproduction receipt",
                status=result.state,
            )
            if commit
            else None
        )
        return {
            "reproduction": result,
            "checkpoint": checkpoint,
            "receipt": receipt_payload,
        }


__all__ = ["ResearchLifecycle"]
