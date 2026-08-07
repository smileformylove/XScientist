from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ai_scientist.apps import preflight as preflight_check


def _write_session(path: Path, *, username: str = "smoke-user") -> None:
    payload = {
        "username": username,
        "session_id": "test-session",
        "issued_at": "2026-03-22T00:00:00+00:00",
        "expires_at": "2099-03-22T00:00:00+00:00",
        "last_seen_at": "2026-03-22T00:00:00+00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PreflightCheckTests(unittest.TestCase):
    def test_bfts_config_checks_selected_model_and_exact_docker_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bfts.yaml"
            config_path.write_text(
                json.dumps(
                    {
                        "report": {"model": "glm-4-flash"},
                        "agent": {
                            "code": {"model": "glm-4-flash"},
                            "feedback": {"model": "glm-4-flash"},
                            "vlm_feedback": {"model": "glm-4-flash"},
                        },
                        "exec": {
                            "backend": "docker",
                            "require_isolation": True,
                            "docker_image": "xscientist-exec:test",
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = mock.Mock(returncode=0)
            with (
                mock.patch.dict(
                    "os.environ", {"ZHIPU_API_KEY": "configured"}, clear=False
                ),
                mock.patch.object(
                    preflight_check.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
                mock.patch.object(
                    preflight_check.shutil, "which", return_value="/usr/bin/docker"
                ),
                mock.patch.object(
                    preflight_check.subprocess, "run", return_value=completed
                ) as run,
            ):
                results = preflight_check.check_bfts_config(str(config_path))

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(
            [result.label for result in results],
            [
                "BFTS configuration",
                "Configured model `glm-4-flash`",
                "Experiment isolation",
            ],
        )
        self.assertEqual(run.call_count, 2)
        self.assertIn(
            "xscientist-exec:test",
            run.call_args_list[1].args[0],
        )

    def test_bfts_config_fails_closed_when_docker_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bfts.yaml"
            config_path.write_text(
                json.dumps(
                    {
                        "report": {"model": "glm-4-flash"},
                        "agent": {"code": {"model": "glm-4-flash"}},
                        "exec": {
                            "backend": "docker",
                            "require_isolation": True,
                            "docker_image": "xscientist-exec:test",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.dict(
                    "os.environ", {"ZHIPU_API_KEY": "configured"}, clear=False
                ),
                mock.patch.object(
                    preflight_check.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
                mock.patch.object(preflight_check.shutil, "which", return_value=None),
            ):
                results = preflight_check.check_bfts_config(str(config_path))

        isolation = next(
            result for result in results if result.label == "Experiment isolation"
        )
        self.assertFalse(isolation.ok)
        self.assertEqual(isolation.severity, "error")
        self.assertIn("docker executable not found", isolation.detail)

    def test_preflight_should_respect_auth_file_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "auth" / "session.json"
            _write_session(session_path)
            stdout = io.StringIO()
            with (
                mock.patch.object(preflight_check, "CORE_PACKAGES", {}),
                mock.patch.object(preflight_check, "PIPELINE_PACKAGES", {}),
                mock.patch.object(preflight_check, "COMMANDS", {}),
                mock.patch(
                    "sys.argv",
                    ["preflight_check.py", "--auth-file", str(session_path)],
                ),
                mock.patch.dict("os.environ", {}, clear=False),
                redirect_stdout(stdout),
            ):
                exit_code = preflight_check.main()

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn(str(session_path), output)
        self.assertIn("[OK] Login session", output)

    def test_preflight_strict_should_fail_for_missing_login_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "auth" / "missing.json"
            stdout = io.StringIO()
            with (
                mock.patch.object(preflight_check, "CORE_PACKAGES", {}),
                mock.patch.object(preflight_check, "PIPELINE_PACKAGES", {}),
                mock.patch.object(preflight_check, "COMMANDS", {}),
                mock.patch(
                    "sys.argv",
                    [
                        "preflight_check.py",
                        "--strict",
                        "--auth-file",
                        str(session_path),
                    ],
                ),
                mock.patch.dict("os.environ", {}, clear=False),
                redirect_stdout(stdout),
            ):
                exit_code = preflight_check.main()

        self.assertEqual(exit_code, 1)
        output = stdout.getvalue()
        self.assertIn(str(session_path), output)
        self.assertIn("[ERROR] Login session", output)
        self.assertIn("未检测到登录会话", output)


if __name__ == "__main__":
    unittest.main()
