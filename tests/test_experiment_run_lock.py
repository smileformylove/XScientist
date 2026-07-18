from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.utils.experiment_run_lock import (
    LOCK_DIR_NAME,
    OWNER_FILE_NAME,
    ExperimentRunLock,
    ExperimentRunLocked,
    _sanitized_command,
    experiment_lock_root,
)
from ai_scientist.utils.launcher_workflow import (
    experiment_stop_exit_code,
    run_experiment_phase,
)


class ExperimentRunLockTests(unittest.TestCase):
    def _write_owner(self, root: Path, owner: dict) -> Path:
        lock_dir = root / LOCK_DIR_NAME
        lock_dir.mkdir()
        (lock_dir / OWNER_FILE_NAME).write_text(json.dumps(owner), encoding="utf-8")
        return lock_dir

    def test_second_acquisition_reports_current_owner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.yaml"
            with ExperimentRunLock(root, config_path=config_path):
                with self.assertRaises(ExperimentRunLocked) as ctx:
                    ExperimentRunLock(root).acquire()

            self.assertEqual(ctx.exception.lock_path, root.resolve() / LOCK_DIR_NAME)
            self.assertEqual(ctx.exception.owner["pid"], os.getpid())
            self.assertEqual(ctx.exception.owner["hostname"], socket.gethostname())
            self.assertEqual(ctx.exception.owner["config_path"], str(config_path))
            self.assertIn("heartbeat_at", ctx.exception.owner)
            self.assertIn("lease_timeout_seconds", ctx.exception.owner)

    def test_context_exit_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with ExperimentRunLock(root):
                self.assertTrue((root / LOCK_DIR_NAME).is_dir())

            self.assertFalse((root / LOCK_DIR_NAME).exists())
            with ExperimentRunLock(root):
                self.assertTrue((root / LOCK_DIR_NAME).is_dir())

    def test_heartbeat_refreshes_owner_and_stops_on_release(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock = ExperimentRunLock(root, heartbeat_interval_seconds=60)
            lock.acquire()
            heartbeat_thread = lock.heartbeat_thread
            owner_path = lock.lock_dir / OWNER_FILE_NAME
            before = json.loads(owner_path.read_text(encoding="utf-8"))["heartbeat_at"]

            with mock.patch(
                "ai_scientist.utils.experiment_run_lock.time.time",
                return_value=before + 10,
            ):
                self.assertTrue(lock._refresh_heartbeat())

            after = json.loads(owner_path.read_text(encoding="utf-8"))["heartbeat_at"]
            self.assertEqual(after, before + 10)
            self.assertTrue(heartbeat_thread.is_alive())
            lock.release()
            self.assertFalse(heartbeat_thread.is_alive())
            self.assertFalse(lock.lock_dir.exists())

    def test_heartbeat_keeps_lease_after_transient_write_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock = ExperimentRunLock(root, heartbeat_interval_seconds=60)
            lock.acquire()
            with mock.patch.object(
                lock, "_write_owner_atomic", side_effect=OSError("shared disk busy")
            ):
                self.assertIsNone(lock._refresh_heartbeat())

            self.assertTrue(lock.acquired)
            self.assertTrue(lock.lock_dir.is_dir())
            lock.release()

    def test_heartbeat_start_failure_removes_acquired_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock = ExperimentRunLock(root)
            with (
                mock.patch.object(
                    lock, "_start_heartbeat", side_effect=RuntimeError("thread denied")
                ),
                self.assertRaisesRegex(RuntimeError, "thread denied"),
            ):
                lock.acquire()

            self.assertFalse(lock.acquired)
            self.assertFalse(lock.lock_dir.exists())

    def test_owner_write_failure_removes_partial_lock(self) -> None:
        failures = (OSError("disk unavailable"), KeyboardInterrupt())
        for failure in failures:
            with (
                self.subTest(failure=type(failure).__name__),
                tempfile.TemporaryDirectory() as td,
            ):
                root = Path(td)
                with (
                    mock.patch.object(Path, "write_text", side_effect=failure),
                    self.assertRaises(type(failure)),
                ):
                    ExperimentRunLock(root).acquire()

                self.assertFalse((root / LOCK_DIR_NAME).exists())

    def test_dead_local_pid_lock_is_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_owner(
                root,
                {
                    "token": "dead-owner",
                    "pid": 999999,
                    "hostname": socket.gethostname(),
                },
            )

            with mock.patch(
                "ai_scientist.utils.experiment_run_lock._pid_is_alive",
                return_value=False,
            ):
                with ExperimentRunLock(root) as acquired:
                    owner = json.loads(
                        (acquired.lock_dir / OWNER_FILE_NAME).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(owner["token"], acquired.token)

            self.assertFalse((root / LOCK_DIR_NAME).exists())

    def test_live_local_and_remote_locks_are_not_stolen(self) -> None:
        owners = (
            {"token": "live", "pid": os.getpid(), "hostname": socket.gethostname()},
            {"token": "remote", "pid": 1234, "hostname": "remote.example"},
        )
        for owner in owners:
            with (
                self.subTest(owner=owner["token"]),
                tempfile.TemporaryDirectory() as td,
            ):
                root = Path(td)
                lock_dir = self._write_owner(root, owner)

                with self.assertRaises(ExperimentRunLocked):
                    ExperimentRunLock(root).acquire()

                persisted = json.loads(
                    (lock_dir / OWNER_FILE_NAME).read_text(encoding="utf-8")
                )
                self.assertEqual(persisted["token"], owner["token"])

    def test_expired_remote_lock_is_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock_dir = self._write_owner(
                root,
                {
                    "token": "remote-dead",
                    "pid": 1234,
                    "hostname": "remote.example",
                    "lease_timeout_seconds": 1,
                },
            )
            owner_path = lock_dir / OWNER_FILE_NAME
            stale_time = time.time() - 10
            os.utime(owner_path, (stale_time, stale_time))

            with ExperimentRunLock(
                root,
                heartbeat_interval_seconds=60,
                lease_timeout_seconds=1,
            ) as acquired:
                owner = json.loads(
                    (acquired.lock_dir / OWNER_FILE_NAME).read_text(encoding="utf-8")
                )
                self.assertEqual(owner["token"], acquired.token)

    def test_remote_heartbeat_refresh_during_reclaim_prevents_steal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original_owner = {
                "token": "remote-live",
                "pid": 1234,
                "hostname": "remote.example",
                "lease_timeout_seconds": 1,
                "heartbeat_at": time.time() - 10,
            }
            lock_dir = self._write_owner(root, original_owner)
            owner_path = lock_dir / OWNER_FILE_NAME
            stale_time = time.time() - 10
            os.utime(owner_path, (stale_time, stale_time))
            refreshed_owner = {**original_owner, "heartbeat_at": time.time()}

            read_count = 0

            def read_owner_with_refresh(_lock_dir: Path) -> dict:
                nonlocal read_count
                read_count += 1
                if read_count == 1:
                    return original_owner
                if read_count == 2:
                    owner_path.write_text(
                        json.dumps(refreshed_owner), encoding="utf-8"
                    )
                return refreshed_owner

            with (
                mock.patch(
                    "ai_scientist.utils.experiment_run_lock._read_owner",
                    side_effect=read_owner_with_refresh,
                ),
                self.assertRaises(ExperimentRunLocked),
            ):
                ExperimentRunLock(
                    root,
                    heartbeat_interval_seconds=0.1,
                    lease_timeout_seconds=1,
                ).acquire()

            self.assertTrue(lock_dir.is_dir())

    def test_live_local_pid_is_not_reclaimed_when_heartbeat_is_old(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock_dir = self._write_owner(
                root,
                {
                    "token": "local-live",
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "lease_timeout_seconds": 1,
                },
            )
            owner_path = lock_dir / OWNER_FILE_NAME
            stale_time = time.time() - 10
            os.utime(owner_path, (stale_time, stale_time))

            with self.assertRaises(ExperimentRunLocked):
                ExperimentRunLock(root, lease_timeout_seconds=1).acquire()

    def test_lock_root_uses_generated_layout_config_and_safe_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            idea_dir = root / "idea"
            config_dir = idea_dir / ".xscientist" / "configs"
            config_dir.mkdir(parents=True)
            generated = config_dir / "run.yaml"
            generated.write_text("workspace_dir: /different/path\n", encoding="utf-8")
            self.assertEqual(experiment_lock_root(generated), idea_dir.resolve())

            configured = root / "standalone.yaml"
            workspace = root / "workspace"
            configured.write_text(
                yaml.safe_dump({"workspace_dir": str(workspace)}), encoding="utf-8"
            )
            self.assertEqual(experiment_lock_root(configured), workspace.resolve())

            relative = root / "relative.yaml"
            relative.write_text("workspace_dir: nested/workspace\n", encoding="utf-8")
            self.assertEqual(
                experiment_lock_root(relative), (root / "nested/workspace").resolve()
            )

            fallback = root / "missing.yaml"
            self.assertEqual(experiment_lock_root(fallback), root.resolve())

    def test_owner_command_redacts_sensitive_cli_values(self) -> None:
        self.assertEqual(
            _sanitized_command(
                [
                    "runner.py",
                    "--api-key",
                    "top-secret",
                    "--auth-file=/tmp/session.json",
                    "--model",
                    "demo",
                ]
            ),
            [
                "runner.py",
                "--api-key",
                "[REDACTED]",
                "--auth-file=[REDACTED]",
                "--model",
                "demo",
            ],
        )

    def test_generated_run_configs_are_unique_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template = root / "template.yaml"
            template.write_text("goal: demo\n", encoding="utf-8")
            idea_dir = root / "idea"
            idea_dir.mkdir()
            idea_path = idea_dir / "idea.json"
            idea_path.write_text("{}", encoding="utf-8")

            first = Path(
                edit_bfts_config_file(str(template), str(idea_dir), str(idea_path))
            )
            second = Path(
                edit_bfts_config_file(str(template), str(idea_dir), str(idea_path))
            )

            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            self.assertEqual(first.parent, idea_dir / ".xscientist" / "configs")
            first_payload = yaml.safe_load(first.read_text(encoding="utf-8"))
            self.assertEqual(first_payload["workspace_dir"], str(idea_dir))
            self.assertEqual(first_payload["desc_file"], str(idea_path))

    @mock.patch(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager._perform_experiments_bfts_locked"
    )
    def test_perform_returns_locked_owner_without_starting_run(
        self, perform_locked_mock: mock.Mock
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            idea_dir = Path(td) / "idea"
            config_dir = idea_dir / ".xscientist" / "configs"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "run.yaml"
            config_path.write_text(
                yaml.safe_dump({"workspace_dir": str(idea_dir)}), encoding="utf-8"
            )

            with ExperimentRunLock(idea_dir, config_path=config_path) as owner_lock:
                result = perform_experiments_bfts(config_path)

            self.assertEqual(result["status"], "locked")
            self.assertEqual(
                result["lock_path"], str(idea_dir.resolve() / LOCK_DIR_NAME)
            )
            self.assertEqual(result["lock_owner"]["token"], owner_lock.token)
            self.assertEqual(result["failure_error"]["type"], "ExperimentRunLocked")
            perform_locked_mock.assert_not_called()

    @mock.patch("ai_scientist.utils.launcher_workflow.mark_stage_complete")
    @mock.patch("ai_scientist.utils.launcher_workflow.aggregate_plots")
    @mock.patch("ai_scientist.utils.launcher_workflow.write_experiment_report")
    @mock.patch("ai_scientist.utils.launcher_workflow.perform_experiments_bfts")
    @mock.patch("ai_scientist.utils.launcher_workflow.edit_bfts_config_file")
    @mock.patch(
        "ai_scientist.utils.launcher_workflow.is_stage_complete", return_value=False
    )
    def test_launcher_returns_locked_and_skips_later_phases(
        self,
        _stage_complete_mock: mock.Mock,
        edit_config_mock: mock.Mock,
        perform_mock: mock.Mock,
        write_report_mock: mock.Mock,
        aggregate_plots_mock: mock.Mock,
        mark_complete_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            idea_path = root / "idea.json"
            idea_path.write_text("{}", encoding="utf-8")
            edit_config_mock.return_value = str(root / "config.yaml")
            perform_mock.return_value = {
                "status": "locked",
                "lock_path": str(root / LOCK_DIR_NAME),
                "lock_owner": {"pid": 123},
                "resumable": False,
            }

            result = run_experiment_phase(
                root,
                idea_path,
                "plot-model",
                resume=False,
                logger=lambda _message: None,
            )

        self.assertEqual(result["status"], "locked")
        self.assertEqual(experiment_stop_exit_code(result), 75)
        write_report_mock.assert_not_called()
        aggregate_plots_mock.assert_not_called()
        mark_complete_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
