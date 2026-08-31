from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from xscientist._version import __version__
from xscientist.cli import main as cli_main
from xscientist.executor_manager import (
    DOCKER_INSTALL_URL,
    build_executor,
    inspect_executor,
    prepare_executor,
)


class ExecutorManagerTests(unittest.TestCase):
    def _workspace(self, root: str) -> Path:
        workspace = Path(root) / "study"
        workspace.mkdir()
        (workspace / "bfts_config.yaml").write_text(
            yaml.safe_dump(
                {"exec": {"docker_image": f"xscientist-exec:{__version__}"}}
            ),
            encoding="utf-8",
        )
        (workspace / "Dockerfile.executor").write_text(
            "FROM python:3.11-slim\nCOPY . /tmp/xscientist-build-context\n",
            encoding="utf-8",
        )
        return workspace

    def test_installed_build_uses_dockerfile_only_temporary_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            sentinel = workspace / "private-research-sentinel.txt"
            sentinel.write_text("must never reach the Docker daemon", encoding="utf-8")
            installed_module = (
                Path(td) / "site-packages" / "xscientist" / "executor_manager.py"
            )
            observed: dict[str, object] = {}

            def run(command, **kwargs):
                build_context = Path(command[-1])
                selected_dockerfile = Path(command[command.index("-f") + 1])
                observed.update(
                    {
                        "command": list(command),
                        "cwd": Path(kwargs["cwd"]),
                        "context": build_context,
                        "dockerfile": selected_dockerfile,
                        "dockerfile_text": selected_dockerfile.read_text(
                            encoding="utf-8"
                        ),
                        "context_files": sorted(
                            path.name for path in build_context.iterdir()
                        ),
                        "sentinel_present": (build_context / sentinel.name).exists(),
                    }
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            ready = {"ok": True, "image": f"xscientist-exec:{__version__}"}
            with (
                mock.patch(
                    "xscientist.executor_manager.shutil.which",
                    return_value="/usr/bin/docker",
                ),
                mock.patch(
                    "xscientist.executor_manager.__file__",
                    str(installed_module),
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    return_value=ready,
                ),
            ):
                payload = build_executor(workspace, pull_base=True, run=run)

        command = observed["command"]
        context = observed["context"]
        dockerfile = observed["dockerfile"]
        self.assertEqual(payload, ready)
        self.assertNotEqual(context, workspace)
        self.assertEqual(observed["cwd"], context)
        self.assertEqual(Path(dockerfile).parent, context)
        self.assertEqual(observed["context_files"], ["Dockerfile.executor"])
        self.assertEqual(
            observed["dockerfile_text"],
            "FROM python:3.11-slim\nCOPY . /tmp/xscientist-build-context\n",
        )
        self.assertFalse(observed["sentinel_present"])
        self.assertIn("--pull", command)
        self.assertNotIn("XSCIENTIST_INSTALL_MODE=local", command)
        self.assertFalse(Path(context).exists())

    def test_source_checkout_build_still_passes_source_context_and_arguments(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            source_context = Path(td) / "source-checkout"
            (source_context / "xscientist").mkdir(parents=True)
            (source_context / "pyproject.toml").write_text(
                "[project]\nname = 'xscientist'\n", encoding="utf-8"
            )
            (source_context / "source-sentinel.txt").write_text(
                "source is intentionally available", encoding="utf-8"
            )
            source_args = [
                "--build-arg",
                "XSCIENTIST_INSTALL_MODE=local",
                "--build-arg",
                "XSCIENTIST_SOURCE_REVISION=abc123",
                "--build-arg",
                "XSCIENTIST_INSTALL_SOURCE=local-source",
            ]
            observed: dict[str, object] = {}

            def run(command, **kwargs):
                observed["command"] = list(command)
                observed["cwd"] = Path(kwargs["cwd"])
                return subprocess.CompletedProcess(command, 0, "", "")

            ready = {"ok": True, "image": f"xscientist-exec:{__version__}"}
            with (
                mock.patch(
                    "xscientist.executor_manager.shutil.which",
                    return_value="/usr/bin/docker",
                ),
                mock.patch(
                    "xscientist.executor_manager.__file__",
                    str(source_context / "xscientist" / "executor_manager.py"),
                ),
                mock.patch(
                    "xscientist.executor_manager.subprocess.run",
                    side_effect=(
                        subprocess.CompletedProcess(
                            ["git", "rev-parse", "HEAD"], 0, "abc123\n", ""
                        ),
                        subprocess.CompletedProcess(
                            ["git", "status", "--porcelain"], 0, "", ""
                        ),
                    ),
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    return_value=ready,
                ),
            ):
                payload = build_executor(workspace, run=run)

        command = observed["command"]
        self.assertEqual(payload, ready)
        self.assertEqual(observed["cwd"], source_context.resolve())
        self.assertEqual(Path(command[-1]), source_context.resolve())
        self.assertEqual(
            Path(command[command.index("-f") + 1]),
            (workspace / "Dockerfile.executor").resolve(),
        )
        for argument in source_args:
            self.assertIn(argument, command)

    def test_check_requires_exact_version_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)

            def run(command, **_kwargs):
                if command[1] == "info":
                    return subprocess.CompletedProcess(command, 0, '"27.0"\n', "")
                labels = {
                    "org.opencontainers.image.version": __version__,
                    "org.opencontainers.image.revision": "release",
                    "org.xscientist.install-source": "pypi-release",
                }
                return subprocess.CompletedProcess(command, 0, json.dumps(labels), "")

            with mock.patch(
                "xscientist.executor_manager.shutil.which",
                return_value="/usr/bin/docker",
            ):
                status = inspect_executor(workspace, run=run)

        self.assertTrue(status["ok"])
        self.assertTrue(status["version_match"])
        self.assertEqual(status["install_source"], "pypi-release")
        self.assertFalse(status["host_paths_disclosed"])

    def test_prepare_reuses_a_valid_cached_image(self) -> None:
        ready = {
            "schema": "xscientist.executor-status.v1",
            "ok": True,
            "image": f"xscientist-exec:{__version__}",
        }
        with (
            mock.patch(
                "xscientist.executor_manager.inspect_executor", return_value=ready
            ),
            mock.patch("xscientist.executor_manager.build_executor") as build,
        ):
            payload = prepare_executor(".")

        self.assertTrue(payload["cache_hit"])
        self.assertFalse(payload["built"])
        build.assert_not_called()

    def test_missing_docker_gives_install_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = self._workspace(td)
            with mock.patch(
                "xscientist.executor_manager.shutil.which", return_value=None
            ):
                status = inspect_executor(workspace)

        self.assertFalse(status["ok"])
        self.assertFalse(status["docker_available"])
        self.assertIn(DOCKER_INSTALL_URL, status["error"])
        self.assertEqual(status["next_action"], DOCKER_INSTALL_URL)

    def test_cli_check_explains_an_unavailable_executor(self) -> None:
        unavailable = {
            "schema": "xscientist.executor-status.v1",
            "ok": False,
            "image": f"xscientist-exec:{__version__}",
            "daemon_ready": False,
            "image_available": False,
            "version_match": False,
            "install_source": None,
            "next_action": "xscientist executor prepare --workspace study",
        }
        output = io.StringIO()
        with (
            mock.patch(
                "xscientist.executor_manager.inspect_executor", return_value=unavailable
            ),
            contextlib.redirect_stdout(output),
        ):
            exit_code = cli_main(["executor", "check", "--workspace", "study"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Image available: False", output.getvalue())
        self.assertIn("executor prepare", output.getvalue())


if __name__ == "__main__":
    unittest.main()
