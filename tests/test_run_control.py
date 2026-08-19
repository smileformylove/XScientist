from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist.cli import main as cli_main
from xscientist.run_control import (
    RUN_SCHEMA,
    RunControlError,
    cancel_run,
    get_run,
    launch_detached_run,
    list_runs,
    read_run_logs,
    resume_run,
)


class LocalRunControlTests(unittest.TestCase):
    def test_launch_persists_private_resumable_state_and_public_redaction(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("xscientist.run_control.subprocess.Popen") as popen,
            mock.patch(
                "xscientist.run_control._process_identity",
                return_value="launched-process",
            ),
        ):
            popen.return_value.pid = 12345
            workspace = Path(td) / "study"
            payload = launch_detached_run(
                workspace,
                [
                    "start",
                    str(workspace),
                    "--question",
                    "private question",
                    "--allow-synthetic-data",
                    "--detach",
                ],
            )

            self.assertEqual(payload["status"], "running")
            self.assertNotIn("resume_argv", payload)
            self.assertNotIn("process_identity", payload)
            self.assertIn("<stored-in-workspace>", payload["command"])
            state_path = workspace / "04_logs" / "runs" / f"{payload['id']}.json"
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], RUN_SCHEMA)
            self.assertIn("private question", saved["resume_argv"])
            self.assertNotIn("--detach", saved["resume_argv"])

    def test_cancel_signals_the_detached_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            runs = workspace / "04_logs" / "runs"
            runs.mkdir(parents=True)
            state = {
                "schema": RUN_SCHEMA,
                "id": "a1",
                "status": "running",
                "hostname": __import__("socket").gethostname(),
                "pid": 12345,
                "process_identity": "owned-process",
                "created_at": "now",
                "resume_argv": ["start", str(workspace), "--question", "q"],
            }
            (runs / "a1.json").write_text(json.dumps(state), encoding="utf-8")
            with (
                mock.patch("xscientist.run_control._pid_is_alive", return_value=True),
                mock.patch(
                    "xscientist.run_control._process_identity",
                    return_value="owned-process",
                ),
                mock.patch("xscientist.run_control.os.getpgid", return_value=12345),
                mock.patch("xscientist.run_control.os.killpg") as killpg,
            ):
                cancelled = cancel_run(workspace, "a1")

            self.assertEqual(cancelled["status"], "cancelling")
            self.assertTrue(cancelled["cancel_requested_at"])
            killpg.assert_called_once()

    def test_cancel_refuses_a_reused_pid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            runs = workspace / "04_logs" / "runs"
            runs.mkdir(parents=True)
            state = {
                "schema": RUN_SCHEMA,
                "id": "a2",
                "status": "running",
                "hostname": __import__("socket").gethostname(),
                "pid": 12345,
                "process_identity": "original-process",
                "created_at": "now",
                "resume_argv": ["start", str(workspace), "--question", "q"],
            }
            (runs / "a2.json").write_text(json.dumps(state), encoding="utf-8")
            with (
                mock.patch("xscientist.run_control._pid_is_alive", return_value=True),
                mock.patch(
                    "xscientist.run_control._process_identity",
                    return_value="different-process",
                ),
                mock.patch("xscientist.run_control.os.killpg") as killpg,
                self.assertRaisesRegex(RunControlError, "interrupted|identity"),
            ):
                cancel_run(workspace, "a2")
            killpg.assert_not_called()

    def test_dead_cancelled_run_reconciles_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            runs = workspace / "04_logs" / "runs"
            runs.mkdir(parents=True)
            state = {
                "schema": RUN_SCHEMA,
                "id": "b2",
                "status": "cancelling",
                "hostname": __import__("socket").gethostname(),
                "pid": 12345,
                "created_at": "now",
                "cancel_requested_at": "then",
                "resume_argv": [
                    "start",
                    str(workspace),
                    "--question",
                    "q",
                    "--allow-synthetic-data",
                ],
            }
            (runs / "b2.json").write_text(json.dumps(state), encoding="utf-8")
            with mock.patch("xscientist.run_control._pid_is_alive", return_value=False):
                reconciled = get_run(workspace, "b2")
            self.assertEqual(reconciled["status"], "cancelled")

            with (
                mock.patch("xscientist.run_control.subprocess.Popen") as popen,
                mock.patch(
                    "xscientist.run_control._process_identity",
                    return_value="resumed-process",
                ),
            ):
                popen.return_value.pid = 54321
                resumed = resume_run(workspace, "b2", force=True)
            self.assertEqual(resumed["status"], "running")
            self.assertEqual(resumed["resume_of"], "b2")

    def test_resume_rechecks_prerequisites_and_reports_the_repair(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            runs = workspace / "04_logs" / "runs"
            runs.mkdir(parents=True)
            state = {
                "schema": RUN_SCHEMA,
                "id": "b3",
                "status": "failed",
                "hostname": __import__("socket").gethostname(),
                "pid": 12345,
                "created_at": "now",
                "resume_argv": ["start", str(workspace), "--question", "q"],
            }
            (runs / "b3.json").write_text(json.dumps(state), encoding="utf-8")
            report = {
                "ok": False,
                "next_actions": ["xscientist executor prepare --workspace ."],
            }
            with mock.patch("xscientist.diagnostics.diagnose", return_value=report):
                with self.assertRaises(RunControlError) as raised:
                    resume_run(workspace, "b3")

            self.assertIn("executor prepare", str(raised.exception))
            self.assertIn(str(workspace), str(raised.exception))

    def test_logs_are_tailed_and_cli_lists_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            runs = workspace / "04_logs" / "runs"
            runs.mkdir(parents=True)
            state = {
                "schema": RUN_SCHEMA,
                "id": "c3",
                "status": "succeeded",
                "created_at": "2026-01-01T00:00:00+00:00",
                "stdout": "c3.out.log",
                "stderr": "c3.err.log",
                "resume_argv": ["start"],
            }
            (runs / "c3.json").write_text(json.dumps(state), encoding="utf-8")
            (runs / "c3.out.log").write_text("one\ntwo\nthree\n", encoding="utf-8")
            (runs / "c3.err.log").write_text("warning\n", encoding="utf-8")

            logs = read_run_logs(workspace, "c3", tail=2)
            self.assertEqual(logs["stdout"], ["two", "three"])
            self.assertEqual(logs["stderr"], ["warning"])
            self.assertEqual(len(list_runs(workspace)), 1)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["runs", "list", "--workspace", str(workspace)])
            self.assertEqual(exit_code, 0)
            self.assertIn("c3", output.getvalue())
            self.assertIn("succeeded", output.getvalue())
            self.assertIn("2026-01-01", output.getvalue())

    def test_failed_json_run_has_actionable_summary_and_status_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            runs = workspace / "04_logs" / "runs"
            runs.mkdir(parents=True)
            state = {
                "schema": RUN_SCHEMA,
                "id": "d4",
                "status": "failed",
                "created_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:01+00:00",
                "returncode": 1,
                "stdout": "d4.out.log",
                "stderr": "d4.err.log",
                "provider": "ollama",
                "model": "ollama/qwen2.5:7b",
                "profile": "balanced",
                "task": "research",
                "resume_argv": ["start"],
            }
            (runs / "d4.json").write_text(json.dumps(state), encoding="utf-8")
            (runs / "d4.out.log").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "next_actions": [
                            'python -m pip install "xscientist[research]"'
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (runs / "d4.err.log").write_text("", encoding="utf-8")

            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                self.assertEqual(
                    cli_main(["runs", "show", "d4", "--workspace", str(workspace)]),
                    1,
                )
            self.assertIn("Prerequisite check failed", shown.getvalue())
            self.assertIn("xscientist[research]", shown.getvalue())
            self.assertNotIn("Failure: }", shown.getvalue())

            status = io.StringIO()
            with contextlib.redirect_stdout(status):
                self.assertEqual(cli_main(["status", str(workspace)]), 1)
            self.assertIn("State: needs attention", status.getvalue())
            self.assertNotIn("Latest background run:", status.getvalue())
            self.assertIn("xscientist runs show d4", status.getvalue())
            self.assertIn(
                "Inspect and repair the latest failed background run",
                status.getvalue(),
            )
            self.assertNotIn("--repo", status.getvalue().split("Run:", 1)[-1])

            verbose_status = io.StringIO()
            with contextlib.redirect_stdout(verbose_status):
                self.assertEqual(cli_main(["status", str(workspace), "--verbose"]), 1)
            self.assertIn(
                "Latest background run: d4 / failed", verbose_status.getvalue()
            )

            logs = io.StringIO()
            with contextlib.redirect_stdout(logs):
                self.assertEqual(
                    cli_main(["runs", "logs", "d4", "--workspace", str(workspace)]),
                    0,
                )
            self.assertIn("--- stdout ---", logs.getvalue())

            watched = io.StringIO()
            with contextlib.redirect_stdout(watched):
                self.assertEqual(
                    cli_main(
                        [
                            "runs",
                            "watch",
                            "d4",
                            "--workspace",
                            str(workspace),
                            "--interval",
                            "0.1",
                        ]
                    ),
                    1,
                )
            self.assertIn("failed", watched.getvalue())
            self.assertIn("Prerequisite check failed", watched.getvalue())

    def test_plaintext_failure_logs_choose_the_first_repair_without_crashing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            runs = workspace / "04_logs" / "runs"
            runs.mkdir(parents=True)
            state = {
                "schema": RUN_SCHEMA,
                "id": "e5",
                "status": "failed",
                "created_at": "2026-01-01T00:00:00+00:00",
                "stdout": "e5.out.log",
                "stderr": "e5.err.log",
                "resume_argv": ["start"],
            }
            (runs / "e5.json").write_text(json.dumps(state), encoding="utf-8")
            (runs / "e5.out.log").write_text(
                "XScientist is configured, but the automated run is not ready.\n"
                "Resolve these items in order:\n"
                '  python -m pip install "xscientist[research]"\n'
                "  xscientist auth login\n"
                "Retry: press Up and rerun.\n",
                encoding="utf-8",
            )
            (runs / "e5.err.log").write_text("", encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(
                    ["runs", "logs", "e5", "--workspace", str(workspace)]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn(
                'Summary: Prerequisite check failed; next: python -m pip install "xscientist[research]"',
                output.getvalue(),
            )

    def test_immediate_detached_exit_is_reported_as_startup_failure(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch("xscientist.run_control.subprocess.Popen") as popen,
            mock.patch(
                "xscientist.run_control._process_identity",
                return_value="short-lived-process",
            ),
        ):
            popen.return_value.pid = 12345
            popen.return_value.poll.return_value = 1
            workspace = Path(td) / "study"

            payload = launch_detached_run(
                workspace,
                [
                    "start",
                    str(workspace),
                    "--question",
                    "q",
                    "--provider",
                    "ollama",
                    "--model",
                    "qwen2.5:7b",
                    "--allow-synthetic-data",
                ],
                startup_grace_seconds=0,
            )

            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["returncode"], 1)
            self.assertEqual(payload["model"], "ollama/qwen2.5:7b")

    def test_detached_cli_does_not_announce_a_terminal_run_as_started(self) -> None:
        payload = {
            "schema": RUN_SCHEMA,
            "id": "f6",
            "status": "failed",
            "workspace": "study",
        }
        stderr = io.StringIO()
        with (
            mock.patch(
                "xscientist.run_control.launch_detached_run",
                return_value=payload,
            ),
            mock.patch(
                "xscientist.run_control.read_run_logs",
                return_value={"stdout": ["Problem: missing dependency"], "stderr": []},
            ),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = cli_main(
                [
                    "start",
                    "study",
                    "--question",
                    "q",
                    "--allow-synthetic-data",
                    "--detach",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("stopped during startup", stderr.getvalue())
        self.assertNotIn("run started", stderr.getvalue())

    def test_detached_launch_error_is_one_json_document(self) -> None:
        output = io.StringIO()
        with (
            mock.patch(
                "xscientist.run_control.launch_detached_run",
                side_effect=RunControlError("cannot create run state"),
            ),
            contextlib.redirect_stdout(output),
        ):
            exit_code = cli_main(
                [
                    "start",
                    "study",
                    "--question",
                    "q",
                    "--allow-synthetic-data",
                    "--detach",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 2)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["phase"], "launch")
        self.assertEqual(payload["error"], "cannot create run state")

    def test_logs_warn_when_structured_tail_starts_mid_document(self) -> None:
        payload = {
            "schema": "xscientist.local-run-logs.v1",
            "ok": True,
            "id": "run-1",
            "stdout": ['"error_codes": [', '  "docker_unavailable"', "]"],
            "stderr": [],
        }
        output = io.StringIO()
        with (
            mock.patch("xscientist.run_control.read_run_logs", return_value=payload),
            contextlib.redirect_stdout(output),
        ):
            exit_code = cli_main(
                ["runs", "logs", "run-1", "--workspace", ".", "--tail", "3"]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("starts mid-document", output.getvalue())

    def test_invalid_run_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RunControlError):
                get_run(td, "../escape")


if __name__ == "__main__":
    unittest.main()
