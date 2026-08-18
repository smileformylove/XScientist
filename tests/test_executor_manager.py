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
        return workspace

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
