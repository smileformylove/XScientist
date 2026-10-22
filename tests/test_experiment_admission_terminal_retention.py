from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist import ResearchRepository
from xscientist import research_git
from xscientist.research_git import (
    ResearchGitError,
    _EXPERIMENT_ADMISSION_INTERNAL,
    _claim_experiment_admission_start,
    _experiment_admission_terminal_completion_ref_name,
    _experiment_admission_terminal_completion_refs,
    _load_experiment_admission_terminal_completion_ref,
    _publish_experiment_admission_terminal_completion_ref,
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
        question="Can terminal admission commits survive repository cleanup?",
        git_user_name="Terminal Retention Test",
        git_user_email="terminal-retention@example.invalid",
    )


def _valid_claim(
    repository: ResearchRepository,
) -> tuple[dict[str, Any], str]:
    snapshot = _git(repository, "rev-parse", "HEAD")
    repository_id = str(load_repository_config(repository.path)["repository_id"])
    scope = {"seeds": [7], "max_steps": 1}
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
        subject="record admission start",
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
    _claim_experiment_admission_start(
        repository.path,
        start_object_id=start.object_id,
        receipt_hash=receipt["receipt_hash"],
    )
    return receipt, start.object_id


def _reserve_attempt(
    repository: ResearchRepository,
    *,
    receipt: dict[str, Any],
    start_object_id: str,
):
    snapshot = {
        "schema": "xscientist.experiment-admission-terminal-policy-snapshot.v1",
        "admitted_at_start": True,
        "current_policy_status": "authorized",
        "quarantined": False,
        "claim_promotion_allowed": False,
    }
    attempt = record_research_object(
        repository.path,
        kind="experiment_attempt",
        payload={
            "status": "failed",
            "attempt_hash": "sha256:" + "4" * 64,
            "admission_preflight_receipt": receipt,
            **{key: value for key, value in snapshot.items() if key != "schema"},
        },
        state="failed",
        relations=[
            {
                "type": "consumes",
                "target": start_object_id,
                "role": "experiment_admission_start",
            }
        ],
    )
    _reserve_experiment_admission_terminal(
        repository.path,
        start_object_id=start_object_id,
        receipt_hash=receipt["receipt_hash"],
        attempt_object_id=attempt.object_id,
        attempt_object_hash=attempt.object_hash,
        terminal_policy_snapshot=snapshot,
    )
    return attempt


def _commit_terminal(
    repository: ResearchRepository,
    *,
    receipt: dict[str, Any],
    start_object_id: str,
) -> str:
    attempt = _reserve_attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
    )
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_terminal",
        subject="record failed experiment attempt",
        status="failed",
        only_paths=[attempt.path.relative_to(repository.path).as_posix()],
    )
    assert checkpoint.commit
    return checkpoint.commit


def test_terminal_completion_ref_publish_is_exact_and_idempotent(
    repository: ResearchRepository,
) -> None:
    start_object_id = "rso-aaaaaaaaaaaaaaaa"
    first_commit = _git(repository, "rev-parse", "HEAD")
    expected_digest = hashlib.sha256(start_object_id.encode("utf-8")).hexdigest()

    assert _experiment_admission_terminal_completion_ref_name(start_object_id) == (
        f"refs/xscientist/admission-terminal/{expected_digest}"
    )
    assert (
        _load_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
        )
        is None
    )
    assert (
        _publish_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
            commit=first_commit,
        )
        == first_commit
    )
    assert (
        _publish_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
            commit=first_commit,
        )
        == first_commit
    )
    assert _experiment_admission_terminal_completion_refs(repository.path) == {
        expected_digest: first_commit
    }

    observation = repository.record(
        "observation",
        {"summary": "Create a second valid commit.", "measurement": 1.0},
    )
    second = create_checkpoint(
        repository.path,
        stage="evidence",
        subject="record second commit",
        only_paths=[observation.path.relative_to(repository.path).as_posix()],
    )
    assert second.commit
    with pytest.raises(ResearchGitError, match="conflicts with its existing commit"):
        _publish_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
            commit=second.commit,
        )


def test_terminal_completion_commit_probe_tolerates_platform_diagnostic(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_object_id = "rso-abababababababab"
    commit = _git(repository, "rev-parse", "HEAD")
    diagnostic = b"platform diagnostic: " + (b"x" * 126) + b"\n"
    assert len(diagnostic) >= 147

    real_popen = subprocess.Popen
    real_run_git_bounded = research_git._run_git_bounded
    observed_limits: list[int] = []

    def run_git_bounded(repo: Path, args: list[str], **kwargs: Any):
        if list(args[:2]) != ["cat-file", "-t"]:
            return real_run_git_bounded(repo, args, **kwargs)

        observed_limits.append(int(kwargs["max_output_bytes"]))
        child_script = (
            "import os; "
            f"os.write(1, {b'commit\n'!r}); "
            f"os.write(2, {diagnostic!r})"
        )

        def launch(_command: list[str], **popen_kwargs: Any):
            return real_popen([sys.executable, "-c", child_script], **popen_kwargs)

        with mock.patch.object(
            research_git.subprocess,
            "Popen",
            side_effect=launch,
        ):
            return real_run_git_bounded(repo, args, **kwargs)

    monkeypatch.setattr(research_git, "_run_git_bounded", run_git_bounded)

    assert (
        _publish_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
            commit=commit,
        )
        == commit
    )
    assert observed_limits == [4096]
    assert observed_limits[0] <= research_git._HARD_MAX_BOUNDED_LOG_OUTPUT
    with pytest.raises(ResearchGitError, match="output limit is invalid"):
        real_run_git_bounded(
            repository.path,
            ["cat-file", "-t", commit],
            max_output_bytes=research_git._HARD_MAX_BOUNDED_LOG_OUTPUT + 1,
        )


@pytest.mark.parametrize("target_kind", ["symbolic", "tree"])
def test_terminal_completion_ref_loader_rejects_non_direct_commit_refs(
    repository: ResearchRepository,
    target_kind: str,
) -> None:
    start_object_id = "rso-bbbbbbbbbbbbbbbb"
    ref_name = _experiment_admission_terminal_completion_ref_name(start_object_id)
    if target_kind == "symbolic":
        branch = _git(repository, "symbolic-ref", "--short", "HEAD")
        _git(repository, "symbolic-ref", ref_name, f"refs/heads/{branch}")
    else:
        tree = _git(repository, "rev-parse", "HEAD^{tree}")
        _git(repository, "update-ref", ref_name, tree)

    with pytest.raises(ResearchGitError, match="completion ref is invalid"):
        _load_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
        )


def test_terminal_completion_ref_enumeration_is_bounded_and_strict(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _git(repository, "rev-parse", "HEAD")
    for value in ("c", "d"):
        start_object_id = f"rso-{value * 16}"
        _git(
            repository,
            "update-ref",
            _experiment_admission_terminal_completion_ref_name(start_object_id),
            head,
        )
    monkeypatch.setattr(
        research_git,
        "_MAX_EXPERIMENT_ADMISSION_TERMINAL_RETENTION_REFS",
        1,
    )

    with pytest.raises(ResearchGitError, match="exceed the audit cap"):
        _experiment_admission_terminal_completion_refs(repository.path)


def test_fsck_accepts_uncommitted_reservation_without_completion_ref(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id = _valid_claim(repository)
    _reserve_attempt(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
    )

    report = verify_research_repository(repository.path)

    assert report["ok"], report["errors"]
    assert report["checked"]["admission_terminal_retention_refs"] == 0
    assert any("uncommitted recoverable" in item for item in report["warnings"])


def test_fsck_requires_and_validates_terminal_completion_ref(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id = _valid_claim(repository)
    terminal_commit = _commit_terminal(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
    )

    missing = verify_research_repository(repository.path)
    assert not missing["ok"]
    assert any("has no completion ref" in error for error in missing["errors"])

    _publish_experiment_admission_terminal_completion_ref(
        repository.path,
        start_object_id=start_object_id,
        commit=terminal_commit,
    )
    valid = verify_research_repository(repository.path)
    assert valid["ok"], valid["errors"]
    assert valid["checked"]["admission_terminal_retention_refs"] == 1

    observation = repository.record(
        "observation",
        {"summary": "Move HEAD beyond the terminal introduction.", "measurement": 2.0},
    )
    later = create_checkpoint(
        repository.path,
        stage="evidence",
        subject="record later evidence",
        only_paths=[observation.path.relative_to(repository.path).as_posix()],
    )
    assert later.commit
    _git(
        repository,
        "update-ref",
        _experiment_admission_terminal_completion_ref_name(start_object_id),
        later.commit,
    )

    divergent = verify_research_repository(repository.path)
    assert not divergent["ok"]
    assert any(
        "does not match the unique terminal-only introduction" in error
        for error in divergent["errors"]
    )


def test_fsck_rejects_orphan_and_malformed_terminal_completion_refs(
    repository: ResearchRepository,
) -> None:
    start_object_id = "rso-eeeeeeeeeeeeeeee"
    head = _git(repository, "rev-parse", "HEAD")
    _publish_experiment_admission_terminal_completion_ref(
        repository.path,
        start_object_id=start_object_id,
        commit=head,
    )

    orphan = verify_research_repository(repository.path)
    assert not orphan["ok"]
    assert any("completion ref is orphaned" in error for error in orphan["errors"])

    malformed_name = "refs/xscientist/admission-terminal/not-a-start-digest"
    _git(repository, "update-ref", malformed_name, head)
    malformed = verify_research_repository(repository.path)
    assert not malformed["ok"]
    assert any("completion ref is invalid" in error for error in malformed["errors"])


def test_terminal_completion_ref_survives_branch_deletion_and_gc(
    repository: ResearchRepository,
) -> None:
    receipt, start_object_id = _valid_claim(repository)
    base_branch = _git(repository, "symbolic-ref", "--short", "HEAD")
    _git(repository, "checkout", "-b", "terminal-retention-test")
    terminal_commit = _commit_terminal(
        repository,
        receipt=receipt,
        start_object_id=start_object_id,
    )
    _publish_experiment_admission_terminal_completion_ref(
        repository.path,
        start_object_id=start_object_id,
        commit=terminal_commit,
    )

    _git(repository, "checkout", base_branch)
    _git(repository, "branch", "-D", "terminal-retention-test")
    _git(repository, "reflog", "expire", "--expire=now", "--all")
    _git(repository, "gc", "--prune=now")

    assert _git(repository, "cat-file", "-t", terminal_commit) == "commit"
    assert (
        _load_experiment_admission_terminal_completion_ref(
            repository.path,
            start_object_id=start_object_id,
        )
        == terminal_commit
    )
    report = verify_research_repository(repository.path)
    assert report["ok"], report["errors"]
    assert report["checked"]["admission_terminal_objects"] == 1
    assert report["checked"]["admission_terminal_retention_refs"] == 1
