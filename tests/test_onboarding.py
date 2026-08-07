from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from xscientist._version import __version__
from xscientist.cli import main as cli_main
from xscientist.onboarding import WORKSPACE_FILES, create_workspace


class OnboardingTests(unittest.TestCase):
    def test_init_creates_safe_installed_package_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["init", str(workspace), "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["files"], list(WORKSPACE_FILES))
            self.assertFalse(payload["secrets_written"])
            for relative in WORKSPACE_FILES:
                self.assertTrue((workspace / relative).is_file(), relative)

            env_text = (workspace / ".env.example").read_text(encoding="utf-8")
            self.assertIn("ZHIPU_API_KEY=replace-me", env_text)
            self.assertNotIn("sk-", env_text)
            self.assertIn(".env\n", (workspace / ".gitignore").read_text())
            self.assertIn(".env\n", (workspace / ".dockerignore").read_text())

            provider_config_text = (
                workspace / ".xscientist" / "providers.json"
            ).read_text(encoding="utf-8")
            provider_config = json.loads(provider_config_text)
            self.assertEqual(provider_config["active_provider"], "zhipu")
            self.assertEqual(
                provider_config["providers"]["zhipu"]["model"], "glm-4-flash"
            )
            self.assertNotIn("replace-me", provider_config_text)

            config = yaml.safe_load(
                (workspace / "bfts_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(config["exec"]["backend"], "docker")
            self.assertTrue(config["exec"]["require_isolation"])
            self.assertEqual(
                config["exec"]["docker_image"], f"xscientist-exec:{__version__}"
            )
            self.assertEqual(config["agent"]["code"]["model"], "glm-4-flash")

            dockerfile = (workspace / "Dockerfile.executor").read_text()
            self.assertIn(f"ARG XSCIENTIST_VERSION={__version__}", dockerfile)
            self.assertIn("xscientist[full]==${XSCIENTIST_VERSION}", dockerfile)
            readme = (workspace / "README.md").read_text()
            self.assertIn("BFTS `default` configuration", readme)
            self.assertIn("xscientist provider add zhipu", readme)
            self.assertNotIn("\n+  --model", readme)

    def test_non_default_provider_requires_matching_explicit_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                missing_model = cli_main(
                    ["init", str(workspace), "--provider", "openai"]
                )
            self.assertEqual(missing_model, 2)
            self.assertIn("--model is required", stderr.getvalue())
            self.assertFalse(workspace.exists())

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                wrong_provider = cli_main(
                    [
                        "init",
                        str(workspace),
                        "--provider",
                        "openai",
                        "--model",
                        "glm-4-flash",
                    ]
                )
            self.assertEqual(wrong_provider, 2)
            self.assertIn("resolves to provider 'zhipu'", stderr.getvalue())
            self.assertFalse(workspace.exists())

    def test_init_refuses_overwrite_and_force_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            readme = workspace / "README.md"
            readme.write_text("user content\n", encoding="utf-8")
            unrelated = workspace / "notes.md"
            unrelated.write_text("keep me\n", encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                refused = cli_main(["init", str(workspace)])
            self.assertEqual(refused, 2)
            self.assertEqual(readme.read_text(), "user content\n")
            self.assertEqual(unrelated.read_text(), "keep me\n")

            replaced = cli_main(["init", str(workspace), "--force"])
            self.assertEqual(replaced, 0)
            self.assertIn("# XScientist workspace", readme.read_text())
            self.assertEqual(unrelated.read_text(), "keep me\n")


if __name__ == "__main__":
    unittest.main()
