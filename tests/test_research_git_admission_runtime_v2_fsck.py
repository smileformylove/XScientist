from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ai_scientist.protocol.canonical_json import (
    canonical_content_hash,
    canonical_json_bytes,
)
from xscientist import ResearchRepository
from xscientist import experiment_admission_runtime as runtime
from xscientist.research_git import (
    ResearchGitError,
    _EXPERIMENT_ADMISSION_INTERNAL,
    _repository_common_git_dir,
    create_checkpoint,
    load_repository_config,
    record_research_object,
    verify_research_repository,
)
from xscientist.research_lifecycle import ResearchLifecycle


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
        question="Can fsck distinguish runtime evidence from committed truth?",
        git_user_name="Runtime Fsck Test",
        git_user_email="runtime-fsck@example.invalid",
    )


def _claim_v2(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    snapshot = _git(repository, "rev-parse", "HEAD")
    repository_id = str(load_repository_config(repository.path)["repository_id"])
    scope = {
        "idea_index": 2,
        "ideas_hash": "sha256:" + "6" * 64,
        "configuration_hash": "sha256:" + "7" * 64,
        "budget_hash": "sha256:" + "8" * 64,
        "seeds": [7],
        "max_steps": 1,
    }
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
        "nonce": "0" * 32,
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
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_start",
        subject="record admission start for runtime fsck",
        only_paths=[start.path.relative_to(repository.path).as_posix()],
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
    experiment_dir = repository.path / "02_experiments" / "run-2"
    experiment_dir.mkdir(parents=True)
    monkeypatch.setattr(
        ResearchLifecycle,
        "validate_finalized_experiment_admission_start",
        lambda self, value: dict(value),
    )
    monkeypatch.setattr(runtime, "_boot_id", lambda: "a" * 64)
    monkeypatch.setattr(runtime, "_process_identity", lambda pid: "b" * 64)
    monkeypatch.setattr(
        runtime, "_process_identity_matches", lambda pid, expected: True
    )
    return runtime.claim_experiment_admission_start_v2(
        repository.path,
        receipt,
        experiment_dir=experiment_dir,
        idea_key="idea-2",
        run_nonce="c" * 32,
        worker_nonce="d" * 32,
        bfts_run_nonce="e" * 32,
        owner_nonce="f" * 32,
    )


def _runtime_marker(
    repository: ResearchRepository,
    directory: str,
    start_object_id: str,
) -> Path:
    return (
        _repository_common_git_dir(repository.path)
        / "xscientist"
        / directory
        / f"{start_object_id}.json"
    )


def _rewrite_canonical(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def test_live_fsck_reports_alive_dead_and_unknown_v2_owners(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim_v2(repository, monkeypatch)

    alive = verify_research_repository(repository.path)
    assert alive["ok"], alive["errors"]
    assert alive["runtime_audited"] is True
    assert alive["checked"]["admission_execution_claims"] == 1
    assert alive["checked"]["admission_execution_claims_v2"] == 1
    assert alive["checked"]["admission_runtime_owners_alive"] == 1
    assert any(
        "remains in flight with a live owner" in row for row in alive["warnings"]
    )

    monkeypatch.setattr(
        runtime, "_process_identity_matches", lambda pid, expected: False
    )
    dead = verify_research_repository(repository.path)
    assert dead["ok"], dead["errors"]
    assert dead["checked"]["admission_runtime_owners_dead"] == 1
    assert any(
        "conservative interrupted recovery is required" in row
        for row in dead["warnings"]
    )

    monkeypatch.setattr(
        runtime, "_process_identity_matches", lambda pid, expected: None
    )
    unknown = verify_research_repository(repository.path)
    assert unknown["ok"], unknown["errors"]
    assert unknown["checked"]["admission_runtime_owners_unknown"] == 1
    assert any(
        "execution remains conservatively unresolved" in row
        for row in unknown["warnings"]
    )
    assert claim["start_object_id"] in "\n".join(unknown["warnings"])


def test_live_fsck_counts_outboxes_and_detects_changed_artifacts(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim_v2(repository, monkeypatch)
    artifact = repository.path / claim["experiment_dir"] / "manager_state.json"
    artifact.write_text('{"step":1}\n', encoding="utf-8")
    runtime.publish_bfts_terminal(
        repository.path,
        claim,
        status="completed",
        stage="bfts",
        resumable=False,
        artifact_paths=[artifact],
    )
    runtime.publish_worker_terminal(
        repository.path,
        claim,
        {
            "idea_idx": claim["idea_index"],
            "status": "success",
            "registry_status": "completed",
            "research_status": "recording",
            "stage": "evidence",
            "manager_state_path": str(artifact),
        },
    )

    valid = verify_research_repository(repository.path)
    assert valid["ok"], valid["errors"]
    assert valid["checked"]["admission_bfts_terminals"] == 1
    assert valid["checked"]["admission_worker_terminals"] == 1
    assert valid["checked"]["admission_runtime_artifacts"] == 2
    assert any(
        "awaits durable Research VCS recording" in row for row in valid["warnings"]
    )

    artifact.write_text('{"step":2}\n', encoding="utf-8")
    changed = verify_research_repository(repository.path)
    assert not changed["ok"]
    assert any("terminal artifact changed" in error for error in changed["errors"])


@pytest.mark.parametrize(
    ("terminal_kind", "directory"),
    [
        ("bfts", "admission-bfts-terminals"),
        ("worker", "admission-worker-terminals"),
    ],
)
def test_live_fsck_rejects_claim_outbox_identity_changes(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
    terminal_kind: str,
    directory: str,
) -> None:
    claim = _claim_v2(repository, monkeypatch)
    if terminal_kind == "bfts":
        terminal = runtime.publish_bfts_terminal(
            repository.path,
            claim,
            status="failed",
            stage="bfts",
            resumable=False,
        )
    else:
        terminal = runtime.publish_worker_terminal(
            repository.path,
            claim,
            {"idea_idx": claim["idea_index"], "status": "failed"},
        )
    forged = dict(terminal)
    forged["claim_hash"] = "sha256:" + "9" * 64
    core = {key: value for key, value in forged.items() if key != "terminal_hash"}
    forged["terminal_hash"] = canonical_content_hash(core)
    _rewrite_canonical(
        _runtime_marker(repository, directory, claim["start_object_id"]),
        forged,
    )

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any("terminal identity conflicts" in error for error in report["errors"])


def test_live_fsck_rejects_orphan_outbox_and_noncanonical_v2_claim(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim_v2(repository, monkeypatch)
    runtime.publish_bfts_terminal(
        repository.path,
        claim,
        status="failed",
        stage="bfts",
        resumable=False,
    )
    claim_path = _runtime_marker(
        repository,
        "admission-claims",
        claim["start_object_id"],
    )
    claim_path.unlink()

    orphan = verify_research_repository(repository.path)
    assert not orphan["ok"]
    assert any(
        "terminal outbox has no matching v2 execution claim" in error
        for error in orphan["errors"]
    )

    claim_path.write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")
    noncanonical = verify_research_repository(repository.path)
    assert not noncanonical["ok"]
    assert any("not canonical JSON" in error for error in noncanonical["errors"])


def test_live_fsck_rejects_worker_reference_to_missing_bfts_terminal(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim_v2(repository, monkeypatch)
    runtime.publish_bfts_terminal(
        repository.path,
        claim,
        status="completed",
        stage="bfts",
        resumable=False,
    )
    runtime.publish_worker_terminal(
        repository.path,
        claim,
        {"idea_idx": claim["idea_index"], "status": "success"},
    )
    _runtime_marker(
        repository,
        "admission-bfts-terminals",
        claim["start_object_id"],
    ).unlink()

    report = verify_research_repository(repository.path)

    assert not report["ok"]
    assert any(
        "references a missing BFTS terminal" in error for error in report["errors"]
    )


def test_historical_fsck_never_uses_v2_runtime_markers_as_commit_truth(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim_v2(repository, monkeypatch)
    historical = str(claim["start_checkpoint_commit"])
    observation = repository.record(
        "observation",
        {"summary": "Advance beyond the admitted start.", "measurement": 1.0},
    )
    later = create_checkpoint(
        repository.path,
        stage="evidence",
        subject="advance live HEAD",
        only_paths=[observation.path.relative_to(repository.path).as_posix()],
    )
    assert later.commit and later.commit != historical
    claim_path = _runtime_marker(
        repository,
        "admission-claims",
        claim["start_object_id"],
    )
    claim_path.write_text('{"schema":', encoding="utf-8")

    live = verify_research_repository(repository.path)
    snapshot = verify_research_repository(repository.path, commit=historical)

    assert not live["ok"]
    assert snapshot["ok"], snapshot["errors"]
    assert snapshot["runtime_audited"] is False
    assert snapshot["checked"]["admission_execution_claims_v2"] == 0
    assert snapshot["checked"]["admission_bfts_terminals"] == 0
    assert not any("runtime" in row for row in snapshot["warnings"])
    with pytest.raises(ResearchGitError, match="only valid for the current HEAD"):
        verify_research_repository(
            repository.path,
            commit=historical,
            audit_runtime=True,
        )
