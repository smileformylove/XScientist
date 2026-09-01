from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ai_scientist.apps import preflight as preflight_check
from ai_scientist.utils.privacy import REDACTED_PATH


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
    def test_generic_docker_probe_errors_do_not_disclose_host_paths(self) -> None:
        private_docker = "/" + "Users" + "/alice/private-tools/docker"
        with (
            mock.patch.object(
                preflight_check.shutil, "which", return_value=private_docker
            ),
            mock.patch.object(
                preflight_check.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(private_docker, 5),
            ),
        ):
            ok, detail = preflight_check._check_docker_image("xscientist-exec:test")

        self.assertFalse(ok)
        self.assertEqual(detail, "docker daemon check timed out")
        self.assertNotIn(private_docker, detail)

    def test_configured_models_include_existing_optional_agent_routes(self) -> None:
        payload = {
            "report": {"model": "openai_compat/report-model"},
            "agent": {
                "code": {"model": "openai_compat/code-model"},
                "feedback": {"model": "openai_compat/feedback-model"},
                "vlm_feedback": {"model": "openai_compat/vlm-model"},
                "summary": {"model": "openai_compat/summary-model"},
                "select_node": {"model": "openai_compat/planner-model"},
            },
        }

        self.assertEqual(
            preflight_check._configured_models(payload),
            [
                "openai_compat/report-model",
                "openai_compat/code-model",
                "openai_compat/feedback-model",
                "openai_compat/vlm-model",
                "openai_compat/summary-model",
                "openai_compat/planner-model",
            ],
        )

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

    def test_workspace_preflight_rejects_a_stale_executor_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "study"
            workspace.mkdir()
            config_path = workspace / "bfts_config.yaml"
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
            (workspace / "Dockerfile.executor").write_text(
                "FROM python:3.11-slim\n", encoding="utf-8"
            )
            stale = {
                "ok": False,
                "image": "xscientist-exec:test",
                "error": (
                    "executor build recipe does not match at "
                    f"{workspace / 'private' / 'inspect.log'}"
                ),
            }
            with (
                mock.patch.dict(
                    "os.environ", {"ZHIPU_API_KEY": "configured"}, clear=False
                ),
                mock.patch.object(
                    preflight_check.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    return_value=stale,
                ) as inspect,
            ):
                results = preflight_check.check_bfts_config(
                    str(config_path), workspace=workspace
                )

        isolation = next(
            result for result in results if result.label == "Experiment isolation"
        )
        self.assertFalse(isolation.ok)
        self.assertIn("identity check failed", isolation.detail)
        self.assertIn("build recipe", isolation.detail)
        self.assertNotIn(str(workspace), isolation.detail)
        inspect.assert_called_once_with(workspace.resolve())

    def test_explicit_workspace_cannot_be_shadowed_by_external_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trusted = Path(tmpdir) / "trusted"
            external = Path(tmpdir) / "external-project"
            for root in (trusted, external):
                root.mkdir()
                (root / "bfts_config.yaml").write_text(
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
                (root / "Dockerfile.executor").write_text(
                    "FROM python:3.11-slim\n", encoding="utf-8"
                )

            inspected: list[Path] = []

            def inspect(root):
                resolved = Path(root).resolve()
                inspected.append(resolved)
                if resolved == external.resolve():
                    return {
                        "ok": True,
                        "image": "xscientist-exec:test",
                        "error": None,
                    }
                return {
                    "ok": False,
                    "image": "xscientist-exec:test",
                    "error": "trusted executor source revision is stale",
                }

            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "XSCIENTIST_WORKSPACE": str(trusted),
                        "ZHIPU_API_KEY": "configured",
                    },
                    clear=False,
                ),
                mock.patch.object(
                    preflight_check.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    side_effect=inspect,
                ),
            ):
                results = preflight_check.check_bfts_config(
                    str(trusted / "bfts_config.yaml"),
                    workspace=external,
                )

        isolation = next(
            result for result in results if result.label == "Experiment isolation"
        )
        self.assertFalse(isolation.ok)
        self.assertIn("trusted executor source revision is stale", isolation.detail)
        self.assertEqual(inspected, [trusted.resolve()])

    def test_initialized_workspace_missing_executor_recipe_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "study"
            workspace.mkdir()
            config_path = workspace / "bfts_config.yaml"
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
            provider_state = workspace / ".xscientist" / "providers.json"
            provider_state.parent.mkdir()
            provider_state.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ,
                    {"ZHIPU_API_KEY": "configured"},
                    clear=False,
                ),
                mock.patch.object(
                    preflight_check.importlib.util,
                    "find_spec",
                    return_value=object(),
                ),
            ):
                results = preflight_check.check_bfts_config(
                    str(config_path), workspace=workspace
                )

        isolation = next(
            result for result in results if result.label == "Experiment isolation"
        )
        self.assertFalse(isolation.ok)
        self.assertIn("Dockerfile.executor", isolation.detail)

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
        self.assertNotIn(str(session_path), output)
        self.assertIn(REDACTED_PATH, output)
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
        self.assertNotIn(str(session_path), output)
        self.assertIn(REDACTED_PATH, output)
        self.assertIn("[ERROR] Login session", output)
        self.assertIn("未检测到登录会话", output)


if __name__ == "__main__":
    unittest.main()
