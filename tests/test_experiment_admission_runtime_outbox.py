from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import pytest

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist import ResearchRepository
from xscientist import experiment_admission_runtime as runtime
import xscientist.run_control as run_control
from xscientist.research_git import (
    ResearchGitError,
    _claim_experiment_admission_start,
    _repository_common_git_dir,
    load_repository_config_at_ref,
)
from xscientist.research_lifecycle import ResearchLifecycle

START_ID = "rso-1111111111111111"
START_HASH = "sha256:" + "2" * 64
SCOPE_HASH = "sha256:" + "3" * 64
CHECKPOINT_HASH = "sha256:" + "4" * 64
CHECKPOINT_COMMIT = "5" * 40


class ProcessIdentityTests(unittest.TestCase):
    @staticmethod
    def _fake_libproc(
        *, received: int | None = None, **overrides: int | bytes
    ) -> tuple[mock.Mock, mock.Mock]:
        def populate(
            pid: int, flavor: int, argument: int, buffer: object, size: int
        ) -> int:
            if flavor != run_control._DARWIN_PROC_PIDTBSDINFO or argument != 0:
                raise AssertionError("unexpected proc_pidinfo request")
            info = run_control.ctypes.cast(
                buffer, run_control.ctypes.POINTER(run_control._DarwinProcBSDInfo)
            ).contents
            values = {
                "pbi_pid": pid,
                "pbi_status": 2,
                "pbi_start_tvsec": 1_700_000_000,
                "pbi_start_tvusec": 123_456,
                "pbi_comm": b"python",
                "pbi_name": b"Python",
                **overrides,
            }
            for field, value in values.items():
                setattr(info, field, value)
            return size if received is None else received

        function = mock.Mock(side_effect=populate)
        library = mock.Mock()
        library.proc_pidinfo = function
        return library, function

    def test_ps_identity_preserves_legacy_digest_and_uses_absolute_path(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="start command\n", stderr=""
        )
        with (
            mock.patch.dict(
                os.environ,
                {"PATH": "/definitely-not-a-real-directory"},
                clear=False,
            ),
            mock.patch.object(run_control.sys, "platform", "darwin"),
            mock.patch.object(
                run_control.subprocess, "run", return_value=completed
            ) as run,
            mock.patch.object(run_control.ctypes, "CDLL") as loader,
        ):
            identity = run_control._process_identity(321)

        self.assertEqual(identity, hashlib.sha256(b"start command").hexdigest())
        self.assertEqual(
            run.call_args.args[0],
            ["/bin/ps", "-p", "321", "-o", "lstart=", "-o", "command="],
        )
        loader.assert_not_called()

    def test_darwin_fallback_uses_libproc_after_both_ps_paths_fail(self) -> None:
        library, proc_pidinfo = self._fake_libproc()
        with (
            mock.patch.object(run_control.sys, "platform", "darwin"),
            mock.patch.object(
                run_control.ctypes, "CDLL", return_value=library
            ) as loader,
            mock.patch.object(
                run_control.subprocess,
                "run",
                side_effect=[
                    PermissionError(1, "denied"),
                    FileNotFoundError(2, "missing"),
                ],
            ) as run,
        ):
            identity = run_control._process_identity(321)

        self.assertRegex(str(identity), r"\A[0-9a-f]{64}\Z")
        self.assertEqual(
            [call.args[0][0] for call in run.call_args_list],
            ["/bin/ps", "/usr/bin/ps"],
        )
        self.assertEqual(proc_pidinfo.call_count, 1)
        loader.assert_called_once_with("/usr/lib/libproc.dylib", use_errno=True)

    def test_darwin_fallback_is_stable_and_pid_reuse_sensitive(self) -> None:
        with (
            mock.patch.object(run_control.sys, "platform", "darwin"),
            mock.patch.object(run_control, "_ps_process_identity", return_value=None),
            mock.patch.object(
                run_control,
                "_darwin_process_metadata",
                side_effect=[
                    (1_700_000_000, 1, b"python", b"Python"),
                    (1_700_000_000, 1, b"python", b"Python"),
                    (1_700_000_000, 2, b"python", b"Python"),
                    (1_700_000_000, 1, b"python", b"renamed"),
                ],
            ),
        ):
            first = run_control._process_identity(321)
            same = run_control._process_identity(321)
            reused = run_control._process_identity(321)
            renamed = run_control._process_identity(321)

        self.assertEqual(first, same)
        self.assertNotEqual(first, reused)
        self.assertNotEqual(first, renamed)

    def test_darwin_match_accepts_a_native_identity_after_provider_switch(
        self,
    ) -> None:
        with (
            mock.patch.object(run_control.os, "name", "posix"),
            mock.patch.object(run_control.sys, "platform", "darwin"),
            mock.patch.object(
                run_control, "_process_identity", return_value="current-ps"
            ),
            mock.patch.object(
                run_control, "_ps_process_identity", return_value="current-ps"
            ),
            mock.patch.object(
                run_control,
                "_darwin_process_identity",
                return_value="stored-native",
            ),
        ):
            self.assertIs(
                run_control._process_identity_matches(321, "stored-native"),
                True,
            )

    def test_darwin_match_is_unknown_when_a_provider_is_missing(self) -> None:
        with (
            mock.patch.object(run_control.os, "name", "posix"),
            mock.patch.object(run_control.sys, "platform", "darwin"),
            mock.patch.object(
                run_control, "_process_identity", return_value="current-native"
            ),
            mock.patch.object(run_control, "_ps_process_identity", return_value=None),
            mock.patch.object(
                run_control,
                "_darwin_process_identity",
                return_value="current-native",
            ),
        ):
            self.assertIsNone(run_control._process_identity_matches(321, "stored-ps"))

    def test_darwin_match_rejects_only_when_both_providers_disagree(self) -> None:
        with (
            mock.patch.object(run_control.os, "name", "posix"),
            mock.patch.object(run_control.sys, "platform", "darwin"),
            mock.patch.object(
                run_control, "_process_identity", return_value="current-ps"
            ),
            mock.patch.object(
                run_control, "_ps_process_identity", return_value="current-ps"
            ),
            mock.patch.object(
                run_control,
                "_darwin_process_identity",
                return_value="current-native",
            ),
        ):
            self.assertIs(
                run_control._process_identity_matches(321, "stale-owner"),
                False,
            )

    def test_darwin_provider_rejects_invalid_and_incomplete_kernel_records(
        self,
    ) -> None:
        cases = {
            "short-read": {"received": run_control._DARWIN_PROC_BSDINFO_SIZE - 1},
            "wrong-pid": {"pbi_pid": 322},
            "zombie": {"pbi_status": run_control._DARWIN_ZOMBIE_STATUS},
            "zero-start": {"pbi_start_tvsec": 0},
            "invalid-microseconds": {"pbi_start_tvusec": 1_000_000},
        }
        with mock.patch.object(run_control.sys, "platform", "darwin"):
            for label, values in cases.items():
                with self.subTest(label=label):
                    received = values.pop("received", None)
                    library, _ = self._fake_libproc(received=received, **values)
                    with mock.patch.object(
                        run_control.ctypes, "CDLL", return_value=library
                    ):
                        self.assertIsNone(run_control._darwin_process_metadata(321))

            for error in (PermissionError(1, "denied"), OSError("unavailable")):
                with self.subTest(error=type(error).__name__):
                    with mock.patch.object(
                        run_control.ctypes, "CDLL", side_effect=error
                    ):
                        self.assertIsNone(run_control._darwin_process_metadata(321))

            with mock.patch.object(run_control.ctypes, "CDLL", return_value=object()):
                self.assertIsNone(run_control._darwin_process_metadata(321))

    def test_invalid_pids_never_reach_the_native_provider(self) -> None:
        with (
            mock.patch.object(run_control.sys, "platform", "darwin"),
            mock.patch.object(run_control.subprocess, "run") as run,
            mock.patch.object(run_control.ctypes, "CDLL") as loader,
        ):
            for pid in (True, 0, -1, run_control._MAX_SIGNED_PID + 1):
                with self.subTest(pid=pid):
                    self.assertIsNone(run_control._process_identity(pid))
        run.assert_not_called()
        loader.assert_not_called()

    def test_non_darwin_ps_failures_return_none_without_native_fallback(self) -> None:
        with (
            mock.patch.object(run_control.sys, "platform", "linux"),
            mock.patch.object(
                run_control.subprocess,
                "run",
                side_effect=[
                    PermissionError(1, "denied"),
                    FileNotFoundError(2, "missing"),
                ],
            ) as run,
            mock.patch.object(run_control.ctypes, "CDLL") as loader,
        ):
            self.assertIsNone(run_control._process_identity(321))

        self.assertEqual(run.call_count, 2)
        loader.assert_not_called()

    def test_linux_identity_prefers_kernel_start_tick(self) -> None:
        with (
            mock.patch.object(run_control.sys, "platform", "linux"),
            mock.patch.object(
                run_control, "_linux_process_identity", return_value="linux-start"
            ) as linux_identity,
            mock.patch.object(run_control, "_ps_process_identity") as ps_identity,
        ):
            self.assertEqual(run_control._process_identity(321), "linux-start")

        linux_identity.assert_called_once_with(321)
        ps_identity.assert_not_called()

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin libproc")
    def test_darwin_live_and_reaped_processes_have_conservative_identities(
        self,
    ) -> None:
        self.assertEqual(
            run_control.ctypes.sizeof(run_control._DarwinProcBSDInfo),
            run_control._DARWIN_PROC_BSDINFO_SIZE,
        )
        metadata = run_control._darwin_process_metadata(os.getpid())
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertGreater(metadata[0], 0)
        self.assertGreaterEqual(metadata[1], 0)
        self.assertTrue(metadata[2] or metadata[3])
        with mock.patch.object(run_control, "_ps_process_identity", return_value=None):
            current = run_control._process_identity(os.getpid())
            self.assertRegex(str(current), r"\A[0-9a-f]{64}\Z")
            self.assertEqual(current, run_control._process_identity(os.getpid()))

            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                child_identity = run_control._process_identity(child.pid)
                self.assertRegex(str(child_identity), r"\A[0-9a-f]{64}\Z")
                self.assertNotEqual(child_identity, current)
            finally:
                child.terminate()
                child.wait(timeout=10)
            self.assertIsNone(run_control._process_identity(child.pid))


def test_run_control_keeps_live_unknown_identity_and_refuses_cancel(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "04_logs" / "runs"
    runs.mkdir(parents=True)
    state_path = runs / "a1.json"
    state_path.write_text(
        json.dumps(
            {
                "schema": run_control.RUN_SCHEMA,
                "id": "a1",
                "status": "running",
                "hostname": run_control.socket.gethostname(),
                "pid": 12345,
                "process_identity": "stored-owner",
                "created_at": "now",
                "resume_argv": ["start", str(tmp_path), "--question", "q"],
            }
        ),
        encoding="utf-8",
    )

    with (
        mock.patch.object(run_control, "_pid_is_alive", return_value=True),
        mock.patch.object(
            run_control, "_process_identity_matches", return_value=None
        ) as identity_matches,
        mock.patch.object(run_control.os, "killpg") as killpg,
    ):
        observed = run_control.get_run(tmp_path, "a1")
        with pytest.raises(
            run_control.RunControlError,
            match="identity cannot be verified",
        ):
            run_control.cancel_run(tmp_path, "a1")

    assert observed["status"] == "running"
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "running"
    assert identity_matches.call_count == 3
    killpg.assert_not_called()


@pytest.mark.skipif(runtime.os.name != "posix", reason="boot identity is POSIX-only")
def test_boot_id_uses_trusted_sysctl_candidates_with_poisoned_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_LINUX_BOOT_ID_PATH", tmp_path / "missing")
    monkeypatch.setenv("PATH", str(tmp_path))
    boot_time = "{ sec = 1234, usec = 5678 } Thu Jan  1 00:20:34 1970"
    commands: list[list[str]] = []

    def run(
        command: list[str], **kwargs: object
    ) -> runtime.subprocess.CompletedProcess:
        commands.append(command)
        if command[0] == "/usr/sbin/sysctl":
            raise FileNotFoundError(command[0])
        return runtime.subprocess.CompletedProcess(command, 0, boot_time, "")

    monkeypatch.setattr(runtime.subprocess, "run", run)

    assert runtime._boot_id() == hashlib.sha256(boot_time.encode("utf-8")).hexdigest()
    assert commands == [
        ["/usr/sbin/sysctl", "-n", "kern.boottime"],
        ["/sbin/sysctl", "-n", "kern.boottime"],
    ]


@pytest.mark.skipif(runtime.os.name != "posix", reason="boot identity is POSIX-only")
def test_boot_id_returns_none_when_trusted_sysctl_candidates_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_LINUX_BOOT_ID_PATH", tmp_path / "missing")
    commands: list[list[str]] = []

    def missing(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(runtime.subprocess, "run", missing)

    assert runtime._boot_id() is None
    assert [command[0] for command in commands] == [
        "/usr/sbin/sysctl",
        "/sbin/sysctl",
    ]


@pytest.mark.skipif(runtime.os.name != "posix", reason="boot identity is POSIX-only")
def test_boot_id_prefers_linux_kernel_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    linux_boot_id = "12345678-1234-1234-1234-123456789abc"
    boot_id_path = tmp_path / "boot-id"
    boot_id_path.write_text(linux_boot_id, encoding="ascii")
    monkeypatch.setattr(runtime, "_LINUX_BOOT_ID_PATH", boot_id_path)

    def unexpected_sysctl(*args: object, **kwargs: object) -> None:
        raise AssertionError("sysctl must not run when the Linux boot ID is available")

    monkeypatch.setattr(runtime.subprocess, "run", unexpected_sysctl)

    assert (
        runtime._boot_id() == hashlib.sha256(linux_boot_id.encode("ascii")).hexdigest()
    )


@pytest.fixture
def repository(tmp_path: Path) -> ResearchRepository:
    return ResearchRepository.init(
        tmp_path / "research",
        question="Can runtime outcomes survive an abrupt worker exit?",
        git_user_name="Runtime Test",
        git_user_email="runtime@example.invalid",
    )


def _receipt(repository: ResearchRepository) -> dict:
    scope = {
        "idea_index": 2,
        "ideas_hash": "sha256:" + "6" * 64,
        "configuration_hash": "sha256:" + "7" * 64,
        "budget_hash": "sha256:" + "8" * 64,
    }
    core = {
        "protocol_kind": "xscientist.experiment-admission-preflight.v2",
        "status": "finalized",
        "experiment_authorized": True,
        "admission_required": True,
        "repository_id": load_repository_config_at_ref(repository.path, "HEAD")[
            "repository_id"
        ],
        "admission_snapshot_head": "9" * 40,
        "repository_head": CHECKPOINT_COMMIT,
        "start_checkpoint": CHECKPOINT_COMMIT,
        "start_checkpoint_id": "rcp-1234567890abcdef",
        "start_checkpoint_hash": CHECKPOINT_HASH,
        "plan_id": "rso-2222222222222222",
        "plan_hash": "sha256:" + "a" * 64,
        "hypothesis_id": "rso-3333333333333333",
        "hypothesis_hash": "sha256:" + "b" * 64,
        "preregistration_id": None,
        "preregistration_hash": None,
        "admission_gate_id": "rso-4444444444444444",
        "admission_gate_hash": "sha256:" + "c" * 64,
        "execution_scope": scope,
        "execution_scope_hash": canonical_content_hash(scope),
        "preflight_receipt_hash": "sha256:" + "d" * 64,
        "issued_receipt_hash": "sha256:" + "e" * 64,
        "start_object_id": START_ID,
        "start_object_hash": START_HASH,
        "start_hash": "sha256:" + "f" * 64,
        "start_nonce": "0" * 32,
    }
    return {**core, "receipt_hash": canonical_content_hash(core)}


@pytest.fixture
def claimed(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path]:
    receipt = _receipt(repository)
    monkeypatch.setattr(
        ResearchLifecycle,
        "validate_finalized_experiment_admission_start",
        lambda self, value: dict(value),
    )
    monkeypatch.setattr(runtime, "_boot_id", lambda: "1" * 64)
    monkeypatch.setattr(runtime, "_process_identity", lambda pid: "2" * 64)
    monkeypatch.setattr(
        runtime, "_process_identity_matches", lambda pid, expected: True
    )
    experiment_dir = repository.path / "02_experiments" / "run-2"
    experiment_dir.mkdir(parents=True)
    claim = runtime.claim_experiment_admission_start_v2(
        repository.path,
        receipt,
        experiment_dir=experiment_dir,
        idea_key="idea-2",
        run_nonce="3" * 32,
        worker_nonce="4" * 32,
        bfts_run_nonce="5" * 32,
        owner_nonce="6" * 32,
    )
    return claim, experiment_dir


def test_claim_v2_binds_process_instance_and_exact_execution_scope(
    repository: ResearchRepository,
    claimed: tuple[dict, Path],
) -> None:
    claim, experiment_dir = claimed
    loaded = runtime.load_execution_claim_v2(
        repository.path,
        start_object_id=START_ID,
        receipt_hash=claim["receipt_hash"],
    )

    assert loaded == claim
    assert (
        claim["experiment_dir"]
        == experiment_dir.relative_to(repository.path).as_posix()
    )
    assert claim["owner"]["boot_id"] == "1" * 64
    assert claim["owner"]["process_identity"] == "2" * 64
    assert claim["ideas_hash"] == "sha256:" + "6" * 64
    assert claim["configuration_hash"] == "sha256:" + "7" * 64
    assert claim["budget_hash"] == "sha256:" + "8" * 64
    assert runtime.execution_claim_owner_state(claim) == "alive"
    with pytest.raises(ResearchGitError, match="already been claimed"):
        runtime.claim_experiment_admission_start_v2(
            repository.path,
            _receipt(repository),
            experiment_dir=experiment_dir,
            idea_key="idea-2",
        )


def test_bfts_terminal_is_immutable_idempotent_and_detects_artifact_change(
    repository: ResearchRepository,
    claimed: tuple[dict, Path],
) -> None:
    claim, experiment_dir = claimed
    checkpoint = experiment_dir / "checkpoint.json"
    checkpoint.write_text('{"step":1}\n', encoding="utf-8")

    first = runtime.publish_bfts_terminal(
        repository.path,
        claim,
        status="interrupted",
        stage="bfts",
        resumable=True,
        failure={"type": "KeyboardInterrupt", "message": "secret-token"},
        artifact_paths=[checkpoint],
    )
    second = runtime.publish_bfts_terminal(
        repository.path,
        claim,
        status="interrupted",
        stage="bfts",
        resumable=True,
        failure={"type": "KeyboardInterrupt", "message": "different secret"},
        artifact_paths=[checkpoint],
    )

    assert second == first
    encoded = json.dumps(first, sort_keys=True)
    assert "secret-token" not in encoded
    assert first["failure"] == {
        "code": "bfts_failed",
        "exception_type": "KeyboardInterrupt",
    }
    runtime.verify_terminal_artifacts(repository.path, first)
    checkpoint.write_text('{"step":2}\n', encoding="utf-8")
    with pytest.raises(ResearchGitError, match="artifact changed"):
        runtime.verify_terminal_artifacts(repository.path, first)


def test_worker_terminal_replays_only_bounded_result_and_provenance(
    repository: ResearchRepository,
    claimed: tuple[dict, Path],
) -> None:
    claim, experiment_dir = claimed
    manager_state = experiment_dir / "manager_state.json"
    manager_state.write_text("{}\n", encoding="utf-8")
    runtime.publish_bfts_terminal(
        repository.path,
        claim,
        status="completed",
        stage="bfts",
        resumable=False,
        artifact_paths=[manager_state],
    )
    result = {
        "idea_idx": 2,
        "status": "success",
        "registry_status": "completed",
        "research_status": "manuscript_draft",
        "stage": "paper",
        "exp_dir": str(experiment_dir),
        "manager_state_path": str(manager_state),
        "error": "must not enter the outbox",
        "traceback": "must not enter the outbox",
    }

    terminal = runtime.publish_worker_terminal(
        repository.path,
        claim,
        result,
        provenance={"environment_hash": "sha256:" + "a" * 64},
        failure={"type": "SecretFailure", "message": "api-key"},
    )
    replayed = runtime.result_from_runtime_record(
        repository.path,
        {
            "claim": claim,
            "worker_terminal": terminal,
            "bfts_terminal": runtime.load_bfts_terminal(
                repository.path, start_object_id=START_ID, claim=claim
            ),
            "owner_state": "dead",
        },
    )

    assert replayed is not None
    recovered, provenance = replayed
    assert recovered["status"] == "success"
    assert recovered["registry_status"] == "completed"
    assert recovered["manager_state_path"] == str(manager_state)
    assert "error" not in terminal["result"]
    assert "traceback" not in terminal["result"]
    assert "api-key" not in json.dumps(terminal)
    assert provenance == {"environment_hash": "sha256:" + "a" * 64}

    durable, durable_provenance = runtime.validate_result_against_runtime_record(
        repository.path,
        {
            "claim": claim,
            "worker_terminal": terminal,
            "bfts_terminal": runtime.load_bfts_terminal(
                repository.path, start_object_id=START_ID, claim=claim
            ),
            "owner_state": "dead",
        },
        result,
    )
    assert durable == recovered
    assert durable_provenance == provenance

    conflicting = dict(result)
    conflicting["research_status"] = "submission_ready"
    with pytest.raises(ResearchGitError, match="conflicts with its terminal outbox"):
        runtime.validate_result_against_runtime_record(
            repository.path,
            {
                "claim": claim,
                "worker_terminal": terminal,
                "bfts_terminal": runtime.load_bfts_terminal(
                    repository.path, start_object_id=START_ID, claim=claim
                ),
                "owner_state": "dead",
            },
            conflicting,
        )


def test_bfts_only_recovery_preserves_experiment_truth_but_fails_pipeline(
    repository: ResearchRepository,
    claimed: tuple[dict, Path],
) -> None:
    claim, experiment_dir = claimed
    bfts = runtime.publish_bfts_terminal(
        repository.path,
        claim,
        status="completed",
        stage="bfts",
        resumable=False,
    )
    replayed = runtime.result_from_runtime_record(
        repository.path,
        {
            "claim": claim,
            "worker_terminal": None,
            "bfts_terminal": bfts,
            "owner_state": "dead",
        },
    )

    assert replayed is not None
    result, provenance = replayed
    assert result["status"] == "failed"
    assert result["registry_status"] == "completed"
    assert result["stage"] == "worker_lost_after_bfts"
    assert result["exp_dir"] == str(experiment_dir)
    assert provenance == {}


def test_dead_owner_without_outbox_yields_conservative_interruption(
    repository: ResearchRepository,
    claimed: tuple[dict, Path],
) -> None:
    claim, _ = claimed
    replayed = runtime.result_from_runtime_record(
        repository.path,
        {
            "claim": claim,
            "worker_terminal": None,
            "bfts_terminal": None,
            "owner_state": "dead",
        },
    )
    assert replayed is not None
    result, _ = replayed
    assert result["runtime_status"] == "interrupted"
    assert result["registry_status"] == "failed"
    assert result["resumable"] is False


def test_unknown_or_alive_owner_without_terminal_blocks_recovery(
    repository: ResearchRepository,
    claimed: tuple[dict, Path],
) -> None:
    claim, _ = claimed
    for owner_state in ("alive", "unknown"):
        assert (
            runtime.result_from_runtime_record(
                repository.path,
                {
                    "claim": claim,
                    "worker_terminal": None,
                    "bfts_terminal": None,
                    "owner_state": owner_state,
                },
            )
            is None
        )


def test_outbox_reader_rejects_symlink_and_duplicate_json_keys(
    repository: ResearchRepository,
) -> None:
    common = _repository_common_git_dir(repository.path) / "xscientist"
    bfts_dir = common / "admission-bfts-terminals"
    bfts_dir.mkdir(parents=True)
    external = repository.path.parent / "external.json"
    external.write_text("{}\n", encoding="utf-8")
    marker = bfts_dir / f"{START_ID}.json"
    marker.symlink_to(external)
    with pytest.raises(ResearchGitError, match="unsafe"):
        runtime.load_bfts_terminal(repository.path, start_object_id=START_ID)

    marker.unlink()
    marker.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
    with pytest.raises(ResearchGitError, match="invalid JSON"):
        runtime.load_bfts_terminal(repository.path, start_object_id=START_ID)


def test_claim_rejects_experiment_path_outside_project(
    repository: ResearchRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ResearchLifecycle,
        "validate_finalized_experiment_admission_start",
        lambda self, value: dict(value),
    )
    monkeypatch.setattr(runtime, "_boot_id", lambda: "1" * 64)
    monkeypatch.setattr(runtime, "_process_identity", lambda pid: "2" * 64)
    outside = repository.path.parent / "outside"
    outside.mkdir()
    with pytest.raises(ResearchGitError, match="leaves the project"):
        runtime.claim_experiment_admission_start_v2(
            repository.path,
            _receipt(repository),
            experiment_dir=outside,
            idea_key="idea-2",
        )


def test_pid_reuse_and_identity_uncertainty_are_not_reported_alive(
    claimed: tuple[dict, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, _ = claimed
    monkeypatch.setattr(
        runtime, "_process_identity_matches", lambda pid, expected: False
    )
    assert runtime.execution_claim_owner_state(claim) == "dead"
    monkeypatch.setattr(
        runtime, "_process_identity_matches", lambda pid, expected: None
    )
    assert runtime.execution_claim_owner_state(claim) == "unknown"


def test_legacy_claim_is_reported_in_doubt_without_synthesizing_an_outcome(
    repository: ResearchRepository,
) -> None:
    _claim_experiment_admission_start(
        repository.path,
        start_object_id=START_ID,
        receipt_hash="sha256:" + "d" * 64,
    )

    records = runtime.admission_runtime_records(repository.path)

    assert len(records) == 1
    assert records[0]["owner_state"] == "legacy_in_doubt"
    assert records[0]["claim"]["schema"].endswith("claim.v1")
    assert runtime.result_from_runtime_record(repository.path, records[0]) is None


def test_runtime_inventory_rejects_outbox_bound_to_legacy_claim(
    repository: ResearchRepository,
) -> None:
    _claim_experiment_admission_start(
        repository.path,
        start_object_id=START_ID,
        receipt_hash="sha256:" + "d" * 64,
    )
    common = _repository_common_git_dir(repository.path) / "xscientist"
    terminal_dir = common / "admission-bfts-terminals"
    terminal_dir.mkdir(parents=True)
    (terminal_dir / f"{START_ID}.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ResearchGitError, match="bound to a legacy claim"):
        runtime.admission_runtime_records(repository.path)


def test_runtime_inventory_rejects_orphan_outbox(
    repository: ResearchRepository,
) -> None:
    common = _repository_common_git_dir(repository.path) / "xscientist"
    terminal_dir = common / "admission-worker-terminals"
    terminal_dir.mkdir(parents=True)
    (terminal_dir / f"{START_ID}.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ResearchGitError, match="has no v2 claim"):
        runtime.admission_runtime_records(repository.path)
