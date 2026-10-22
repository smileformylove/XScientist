from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist import ResearchRepository
from xscientist import research_git
from xscientist.research_git import (
    _EXPERIMENT_ADMISSION_INTERNAL,
    _claim_experiment_admission_start,
    _load_experiment_admission_terminal_reservation,
    _publish_experiment_admission_terminal_completion_ref,
    _repository_common_git_dir,
    _reserve_experiment_admission_terminal,
    create_checkpoint,
    load_repository_config,
    record_research_object,
    verify_research_repository,
)


def _git(repository: ResearchRepository, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository.path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> ResearchRepository:
    return ResearchRepository.init(
        tmp_path / "research",
        question="Can repository fsck close admission runtime state?",
        git_user_name="Admission Fsck Test",
        git_user_email="admission-fsck@example.invalid",
    )


def _state_root(repository: ResearchRepository) -> Path:
    return _repository_common_git_dir(repository.path) / "xscientist"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _valid_claim(
    repository: ResearchRepository,
    *,
    seed: int = 7,
    nonce: str = "0" * 32,
) -> tuple[dict[str, Any], str, str]:
    snapshot = _git(repository, "rev-parse", "HEAD")
    repository_id = str(load_repository_config(repository.path)["repository_id"])
    scope = {"seeds": [seed], "max_steps": 1}
    preflight_core = {
        "protocol_kind": "xscientist.experiment-admission-preflight.v1",
        "status": "authorized",
        "experiment_authorized": True,
        "admission_required": True,
        "repository_id": repository_id,
        "repository_head": snapshot,
        "plan_id": "rso-1111111111111111",
        "plan_hash": "sha256:" + "1" * 64,
        "hypothesis_id": "rso-2222222222222222",
        "hypothesis_hash": "sha256:" + "2" * 64,
        "preregistration_id": None,
        "preregistration_hash": None,
        "admission_gate_id": "rso-3333333333333333",
        "admission_gate_hash": "sha256:" + "3" * 64,
        "execution_scope": scope,
        "execution_scope_hash": canonical_content_hash(scope),
    }
    start_core = {
        "protocol_kind": "xscientist.experiment-admission-start.v1",
        "status": "issued",
        "nonce": nonce,
        "repository_id": repository_id,
        "admission_snapshot_head": snapshot,
        "plan_id": preflight_core["plan_id"],
        "plan_hash": preflight_core["plan_hash"],
        "hypothesis_id": preflight_core["hypothesis_id"],
        "hypothesis_hash": preflight_core["hypothesis_hash"],
        "preregistration_id": None,
        "preregistration_hash": None,
        "admission_gate_id": preflight_core["admission_gate_id"],
        "admission_gate_hash": preflight_core["admission_gate_hash"],
        "execution_scope": scope,
        "execution_scope_hash": canonical_content_hash(scope),
        "preflight_receipt_hash": canonical_content_hash(preflight_core),
    }
    start_hash = canonical_content_hash(start_core)
    start = record_research_object(
        repository.path,
        kind="context_snapshot",
        payload={
            **start_core,
            "context_hash": start_hash,
            "start_hash": start_hash,
        },
        state="locked",
        _admission_capability=_EXPERIMENT_ADMISSION_INTERNAL,
    )
    relative_start = start.path.relative_to(repository.path).as_posix()
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_start",
        subject="record admission start",
        only_paths=[relative_start],
    )
    assert checkpoint.commit
    issued_core = {
        **preflight_core,
        "start_object_id": start.object_id,
        "start_object_hash": start.object_hash,
        "start_hash": start_hash,
    }
    final_core = {
        "protocol_kind": "xscientist.experiment-admission-preflight.v2",
        "status": "finalized",
        "experiment_authorized": True,
        "admission_required": True,
        "repository_id": repository_id,
        "admission_snapshot_head": snapshot,
        "repository_head": checkpoint.commit,
        "start_checkpoint": checkpoint.commit,
        "start_checkpoint_id": checkpoint.checkpoint_id,
        "start_checkpoint_hash": checkpoint.content_hash,
        "plan_id": preflight_core["plan_id"],
        "plan_hash": preflight_core["plan_hash"],
        "hypothesis_id": preflight_core["hypothesis_id"],
        "hypothesis_hash": preflight_core["hypothesis_hash"],
        "preregistration_id": None,
        "preregistration_hash": None,
        "admission_gate_id": preflight_core["admission_gate_id"],
        "admission_gate_hash": preflight_core["admission_gate_hash"],
        "execution_scope": scope,
        "execution_scope_hash": canonical_content_hash(scope),
        "preflight_receipt_hash": canonical_content_hash(preflight_core),
        "issued_receipt_hash": canonical_content_hash(issued_core),
        "start_object_id": start.object_id,
        "start_object_hash": start.object_hash,
        "start_hash": start_hash,
        "start_nonce": start_core["nonce"],
    }
    receipt = {**final_core, "receipt_hash": canonical_content_hash(final_core)}
    _claim_experiment_admission_start(
        repository.path,
        start_object_id=start.object_id,
        receipt_hash=receipt["receipt_hash"],
    )
    return receipt, start.object_id, start.object_hash


def _attempt(
    repository: ResearchRepository,
    *,
    receipt: dict[str, Any],
    start_object_id: str,
    consume_start: bool = True,
    state: str = "failed",
    status: str = "failed",
    payload_overrides: dict[str, Any] | None = None,
):
    return record_research_object(
        repository.path,
        kind="experiment_attempt",
        payload={
            "status": status,
            "attempt_hash": "sha256:" + "4" * 64,
            "admission_preflight_receipt": receipt,
            **(payload_overrides or {}),
        },
        state=state,
        relations=(
            [
                {
                    "type": "consumes",
                    "target": start_object_id,
                    "role": "experiment_admission_start",
                }
            ]
            if consume_start
            else []
        ),
    )


def _upgrade_reservation_to_v2(path: Path, snapshot: dict[str, Any]) -> None:
    reservation = json.loads(path.read_text(encoding="utf-8"))
    reservation.update(
        {
            "schema": "xscientist.experiment-admission-terminal-reservation.v2",
            "terminal_policy_snapshot": snapshot,
            "terminal_policy_snapshot_hash": canonical_content_hash(snapshot),
        }
    )
    _write_json(path, reservation)


def test_v2_reservation_writer_binds_policy_snapshot(
    repository: ResearchRepository,
) -> None:
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    values = {
        "start_object_id": "rso-1111111111111111",
        "receipt_hash": "sha256:" + "2" * 64,
        "attempt_object_id": "rso-3333333333333333",
        "attempt_object_hash": "sha256:" + "3" * 64,
        "terminal_policy_snapshot": snapshot,
    }

    first = _reserve_experiment_admission_terminal(repository.path, **values)
    second = _reserve_experiment_admission_terminal(repository.path, **values)
    reservation = _load_experiment_admission_terminal_reservation(
        repository.path,
        start_object_id=values["start_object_id"],
    )

    assert second == first
    assert reservation is not None
    assert reservation["schema"].endswith(".v2")
    assert reservation["terminal_policy_snapshot"] == snapshot
    assert reservation["terminal_policy_snapshot_hash"] == canonical_content_hash(
        snapshot
    )
    with pytest.raises(research_git.ResearchGitError, match="conflicting attempt"):
        _reserve_experiment_admission_terminal(
            repository.path,
            **{
                key: value
                for key, value in values.items()
                if key != "terminal_policy_snapshot"
            },
        )


def test_fsck_accepts_valid_claim_as_in_flight(repository: ResearchRepository) -> None:
    _receipt, start_object_id, _start_object_hash = _valid_claim(repository)

    report = verify_research_repository(repository.path)

    assert report["ok"], report["errors"]
    assert report["runtime_audited"] is True
    assert report["checked"]["admission_execution_claims"] == 1
    assert any(start_object_id in warning for warning in report["warnings"])


def test_historical_fsck_is_snapshot_only_and_never_scans_all_refs(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _git(repository, "rev-parse", "HEAD")
    _receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    claim_path = (
        _state_root(repository) / "admission-claims" / f"{start_object_id}.json"
    )
    claim_path.write_text('{"schema":', encoding="utf-8")
    original_run_git_bounded = research_git._run_git_bounded

    def reject_all_refs(repo, args, **kwargs):
        assert "--all" not in args
        return original_run_git_bounded(repo, args, **kwargs)

    monkeypatch.setattr(research_git, "_run_git_bounded", reject_all_refs)

    report = verify_research_repository(repository.path, commit=historical)

    assert report["ok"], report["errors"]
    assert report["commit"] == historical
    assert report["runtime_audited"] is False
    assert report["checked"]["admission_execution_claims"] == 0
    assert not any("experiment admission" in item for item in report["warnings"])


def test_runtime_fsck_can_be_disabled_but_not_applied_to_an_old_revision(
    repository: ResearchRepository,
) -> None:
    historical = _git(repository, "rev-parse", "HEAD")
    _valid_claim(repository)

    snapshot = verify_research_repository(
        repository.path,
        commit="HEAD",
        audit_runtime=False,
    )

    assert snapshot["ok"], snapshot["errors"]
    assert snapshot["runtime_audited"] is False
    assert snapshot["checked"]["admission_execution_claims"] == 0
    with pytest.raises(
        research_git.ResearchGitError,
        match="only valid for the current HEAD",
    ):
        verify_research_repository(
            repository.path,
            commit=historical,
            audit_runtime=True,
        )


def test_fsck_marks_objectless_v1_reservation_for_manual_recovery(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    attempt_hash = "sha256:" + "a" * 64
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id="rso-" + "a" * 16,
        attempt_object_hash=attempt_hash,
    )

    report = verify_research_repository(repository.path)

    assert report["ok"], report["errors"]
    assert report["checked"]["admission_terminal_reservations"] == 1
    assert report["checked"]["admission_terminal_objects"] == 0
    warning = next(item for item in report["warnings"] if "legacy v1" in item)
    assert "legacy v1" in warning
    assert "manual recovery is required" in warning
    assert "exact retry" not in warning


def test_fsck_keeps_objectless_v2_exact_retry_diagnostic(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    attempt_hash = "sha256:" + "a" * 64
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id="rso-" + "a" * 16,
        attempt_object_hash=attempt_hash,
        terminal_policy_snapshot=snapshot,
    )

    report = verify_research_repository(repository.path)

    assert report["ok"], report["errors"]
    warning = next(
        item for item in report["warnings"] if "exact retry can recover" in item
    )
    assert "exact retry can recover" in warning


def test_v1_to_v2_retry_requires_an_existing_policy_bound_attempt(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    missing_hash = "sha256:" + "a" * 64
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id="rso-" + "a" * 16,
        attempt_object_hash=missing_hash,
    )
    with pytest.raises(research_git.ResearchGitError, match="manual recovery"):
        _reserve_experiment_admission_terminal(
            repository.path,
            start_object_id=start_object_id,
            receipt_hash=receipt["receipt_hash"],
            attempt_object_id="rso-" + "a" * 16,
            attempt_object_hash=missing_hash,
            terminal_policy_snapshot=snapshot,
        )


def test_v1_to_v2_retry_accepts_only_a_validated_existing_attempt(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    attempt = _attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
        payload_overrides={
            key: value for key, value in snapshot.items() if key != "schema"
        },
    )
    reservation_path = _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
    )

    retried = _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
        terminal_policy_snapshot=snapshot,
    )
    report = verify_research_repository(repository.path)

    assert retried == reservation_path
    assert report["ok"], report["errors"]
    assert any("validated v2 migration" in item for item in report["warnings"])


def test_fsck_validates_uncommitted_and_committed_terminal_attempts(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    attempt = _attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
        payload_overrides={
            key: value for key, value in snapshot.items() if key != "schema"
        },
    )
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
        terminal_policy_snapshot=snapshot,
    )

    uncommitted = verify_research_repository(repository.path)
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_terminal",
        subject="record failed experiment attempt",
        status="failed",
        only_paths=[attempt.path.relative_to(repository.path).as_posix()],
    )
    assert checkpoint.commit
    _publish_experiment_admission_terminal_completion_ref(
        repository.path,
        start_object_id=start_object_id,
        commit=checkpoint.commit,
    )
    committed = verify_research_repository(repository.path)

    assert uncommitted["ok"], uncommitted["errors"]
    assert any("uncommitted recoverable" in item for item in uncommitted["warnings"])
    assert committed["ok"], committed["errors"]
    assert committed["checked"]["admission_terminal_objects"] == 1


def test_fsck_rejects_direct_terminal_checkpoint_status_mismatch(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    attempt = _attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
    )
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
    )
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_terminal",
        subject="record failed attempt with wrong checkpoint status",
        status="completed",
        only_paths=[attempt.path.relative_to(repository.path).as_posix()],
    )
    assert checkpoint.commit
    _publish_experiment_admission_terminal_completion_ref(
        repository.path,
        start_object_id=start_object_id,
        commit=checkpoint.commit,
    )

    report = verify_research_repository(repository.path, commit=checkpoint.commit)

    assert not report["ok"]
    assert report["runtime_audited"] is False
    assert any(
        "status does not match the direct attempt state" in error
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("checkpoint_status", "expected_ok"),
    [("completed", True), ("failed", False)],
)
def test_fsck_enforces_terminal_batch_aggregate_status(
    repository: ResearchRepository,
    checkpoint_status: str,
    expected_ok: bool,
) -> None:
    first_receipt, first_start, _first_hash = _valid_claim(
        repository,
        seed=7,
        nonce="1" * 32,
    )
    second_receipt, second_start, _second_hash = _valid_claim(
        repository,
        seed=11,
        nonce="2" * 32,
    )
    first_attempt = _attempt(
        repository,
        receipt=first_receipt,
        start_object_id=first_start,
        state="completed",
        status="completed",
    )
    second_attempt = _attempt(
        repository,
        receipt=second_receipt,
        start_object_id=second_start,
    )
    for receipt, start_object_id, attempt in (
        (first_receipt, first_start, first_attempt),
        (second_receipt, second_start, second_attempt),
    ):
        _reserve_experiment_admission_terminal(
            repository.path,
            start_object_id=start_object_id,
            receipt_hash=receipt["receipt_hash"],
            attempt_object_id=attempt.object_id,
            attempt_object_hash=attempt.object_hash,
        )
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_terminal",
        subject="record mixed terminal attempt batch",
        status=checkpoint_status,
        only_paths=[
            first_attempt.path.relative_to(repository.path).as_posix(),
            second_attempt.path.relative_to(repository.path).as_posix(),
        ],
    )
    assert checkpoint.commit
    for start_object_id in (first_start, second_start):
        _publish_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
            commit=checkpoint.commit,
        )

    report = verify_research_repository(repository.path, commit=checkpoint.commit)

    assert report["ok"] is expected_ok, report["errors"]
    assert report["runtime_audited"] is False
    if not expected_ok:
        assert any(
            "status does not match the batch aggregate rule" in error
            for error in report["errors"]
        )


def test_fsck_validates_v2_terminal_policy_snapshot(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    attempt = _attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
        payload_overrides={
            key: value for key, value in snapshot.items() if key != "schema"
        },
    )
    reservation_path = _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
    )
    _upgrade_reservation_to_v2(reservation_path, snapshot)
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_terminal",
        subject="record v2 reserved attempt",
        status="failed",
        only_paths=[attempt.path.relative_to(repository.path).as_posix()],
    )
    assert checkpoint.commit
    _publish_experiment_admission_terminal_completion_ref(
        repository.path,
        start_object_id=start_object_id,
        commit=checkpoint.commit,
    )

    report = verify_research_repository(repository.path)

    assert report["ok"], report["errors"]
    assert report["checked"]["admission_terminal_objects"] == 1


def test_fsck_rejects_v2_policy_snapshot_that_does_not_bind_attempt(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    attempt = _attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
        payload_overrides={
            "admitted_at_start": True,
            "current_policy_status": "authorized",
            "quarantined": False,
            "claim_promotion_allowed": False,
        },
    )
    reservation_path = _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
    )
    mismatched_snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "denied_after_start",
        "quarantined": True,
        "claim_promotion_allowed": False,
    }
    _upgrade_reservation_to_v2(reservation_path, mismatched_snapshot)

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any(
        "policy snapshot does not bind its attempt" in error
        for error in report["errors"]
    )


def test_fsck_rejects_v2_policy_snapshot_hash_mismatch(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    attempt_hash = "sha256:" + "a" * 64
    reservation_path = _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id="rso-" + "a" * 16,
        attempt_object_hash=attempt_hash,
    )
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    _upgrade_reservation_to_v2(reservation_path, snapshot)
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation["terminal_policy_snapshot_hash"] = "sha256:" + "f" * 64
    _write_json(reservation_path, reservation)

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any("policy snapshot is invalid" in error for error in report["errors"])


def test_fsck_rejects_terminal_attempt_introduced_by_wrong_checkpoint(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id, _start_object_hash = _valid_claim(repository)
    attempt = _attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
    )
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
    )
    create_checkpoint(
        repository.path,
        stage="failed",
        subject="record attempt under wrong checkpoint",
        status="failed",
        only_paths=[attempt.path.relative_to(repository.path).as_posix()],
    )

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any("experiment_admission_terminal" in error for error in report["errors"])


def test_fsck_rejects_reservation_without_matching_claim(
    repository: ResearchRepository,
) -> None:
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id="rso-1111111111111111",
        receipt_hash="sha256:" + "2" * 64,
        attempt_object_id="rso-3333333333333333",
        attempt_object_hash="sha256:" + "3" * 64,
    )

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any("no matching execution claim" in error for error in report["errors"])


@pytest.mark.parametrize("marker_kind", ["symlink", "directory", "malformed"])
def test_fsck_rejects_unsafe_or_malformed_claim_marker(
    repository: ResearchRepository,
    tmp_path: Path,
    marker_kind: str,
) -> None:
    claims = _state_root(repository) / "admission-claims"
    claims.mkdir(parents=True)
    marker = claims / "rso-1111111111111111.json"
    if marker_kind == "symlink":
        external = tmp_path / "outside-claim.json"
        external.write_text("{}\n", encoding="utf-8")
        marker.symlink_to(external)
    elif marker_kind == "directory":
        marker.mkdir()
    else:
        marker.write_text('{"schema":', encoding="utf-8")

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any(
        "execution claim" in error or "claims" in error for error in report["errors"]
    )


def test_fsck_enforces_runtime_marker_scan_cap(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claims = _state_root(repository) / "admission-claims"
    claims.mkdir(parents=True)
    _write_json(claims / "rso-1111111111111111.json", {})
    _write_json(claims / "rso-2222222222222222.json", {})
    monkeypatch.setattr(research_git, "_MAX_EXPERIMENT_ADMISSION_RUNTIME_MARKERS", 1)

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any("too many entries" in error for error in report["errors"])


def _pending_path(repository: ResearchRepository) -> tuple[Path, dict[str, Any]]:
    subject = "sha256:" + "1" * 64
    core = {
        "schema_version": "xscientist.experiment-admission-transition.v1",
        "subject_hash": subject,
        "gate_object_id": "rso-2222222222222222",
        "gate_object_hash": "sha256:" + "2" * 64,
        "decision": "deny",
        "admission_request_hash": "sha256:" + "3" * 64,
        "authority_receipt_hash": "sha256:" + "4" * 64,
    }
    pending = {**core, "transition_hash": canonical_content_hash(core)}
    subject_root = (
        _state_root(repository)
        / "admission-decisions"
        / subject.removeprefix("sha256:")
    )
    (subject_root / "events").mkdir(parents=True)
    return subject_root / "pending.json", pending


def test_fsck_reports_valid_pending_transition_as_warning(
    repository: ResearchRepository,
) -> None:
    pending_path, pending = _pending_path(repository)
    _write_json(pending_path, pending)

    report = verify_research_repository(repository.path)

    assert report["ok"], report["errors"]
    assert report["checked"]["admission_pending_transitions"] == 1
    assert any("transition is pending" in warning for warning in report["warnings"])


@pytest.mark.parametrize("marker_kind", ["symlink", "directory", "malformed"])
def test_fsck_rejects_unsafe_or_malformed_pending_transition(
    repository: ResearchRepository,
    tmp_path: Path,
    marker_kind: str,
) -> None:
    pending_path, pending = _pending_path(repository)
    if marker_kind == "symlink":
        external = tmp_path / "outside-pending.json"
        _write_json(external, pending)
        pending_path.symlink_to(external)
    elif marker_kind == "directory":
        pending_path.mkdir()
    else:
        pending["transition_hash"] = "sha256:" + "f" * 64
        _write_json(pending_path, pending)

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any(
        "pending" in error or "high-water event" in error for error in report["errors"]
    )


def test_fsck_rejects_orphan_and_malformed_retention_refs(
    repository: ResearchRepository,
) -> None:
    orphan = "refs/xscientist/admission/" + "a" * 64
    malformed = "refs/xscientist/admission/not-a-subject"
    _git(repository, "update-ref", orphan, "HEAD")
    _git(repository, "update-ref", malformed, "HEAD")

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any("retention ref is invalid" in error for error in report["errors"])


def test_fsck_enforces_retention_ref_scan_cap(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git(
        repository,
        "update-ref",
        "refs/xscientist/admission/" + "a" * 64,
        "HEAD",
    )
    _git(
        repository,
        "update-ref",
        "refs/xscientist/admission/" + "b" * 64,
        "HEAD",
    )
    monkeypatch.setattr(research_git, "_MAX_EXPERIMENT_ADMISSION_RETENTION_REFS", 1)

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any("exceed the audit cap" in error for error in report["errors"])
