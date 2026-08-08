from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from xscientist._version import __version__
from xscientist.cli import main as cli_main
from xscientist.diagnostics import diagnose
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
            self.assertIn(".env.*\n", (workspace / ".gitignore").read_text())
            self.assertIn("!.env.example\n", (workspace / ".gitignore").read_text())

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
            self.assertEqual(config["llm_budget"]["max_total_tokens"], 500_000)
            self.assertEqual(config["llm_budget"]["max_wall_time_seconds"], 21_600)

            dockerfile = (workspace / "Dockerfile.executor").read_text()
            self.assertIn(f"ARG XSCIENTIST_VERSION={__version__}", dockerfile)
            self.assertIn(
                "xscientist[research,ml,pdf-layout,zhipu]==${XSCIENTIST_VERSION}",
                dockerfile,
            )
            readme = (workspace / "README.md").read_text()
            self.assertIn("BFTS `default` configuration", readme)
            self.assertIn(
                f"xscientist[research,zhipu]=={__version__}",
                readme,
            )
            self.assertIn("xscientist provider add zhipu", readme)
            self.assertIn("xscientist git doctor", readme)
            self.assertIn("--output-root ./outputs", readme)
            self.assertIn(
                "xscientist manager --research-dir ./outputs list-papers", readme
            )
            self.assertIn("xscientist research objects", readme)
            self.assertNotIn("--research-git", readme)
            self.assertNotIn("\n+  --model", readme)
            self.assertEqual(
                payload["next_steps"][1],
                f'python -m pip install "xscientist[research,zhipu]=={__version__}"',
            )
            self.assertEqual(payload["next_steps"][2], "xscientist git doctor")

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

    def test_setup_creates_and_diagnoses_without_requiring_a_secret(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["workspace_created"])
            self.assertEqual(payload["task"], "protocol")
            self.assertEqual(payload["doctor"]["schema"], "xscientist.doctor.v1")
            self.assertIsNone(payload["doctor"]["checks"]["capabilities"]["provider"])
            self.assertTrue(payload["doctor"]["checks"]["capabilities"]["ready"])
            self.assertFalse(payload["host_paths_disclosed"])
            self.assertFalse((workspace / ".env").exists())
            self.assertTrue((workspace / "research.yaml").is_file())
            self.assertTrue(payload["research_vcs"]["initialized"])
            self.assertIsNotNone(payload["research_vcs"]["checkpoint_id"])
            dockerfile = (workspace / "Dockerfile.executor").read_text()
            self.assertIn(f"xscientist==${{XSCIENTIST_VERSION}}", dockerfile)
            self.assertNotIn("torch --index-url", dockerfile)
            self.assertNotIn("xscientist[research", dockerfile)
            generated_readme = (workspace / "README.md").read_text()
            self.assertIn("provider-neutral", generated_readme)
            self.assertNotIn("xscientist provider add zhipu", generated_readme)
            serialized = json.dumps(payload)
            self.assertNotIn(str(Path(td).resolve()), serialized)

    def test_deep_doctor_redacts_paths_from_runtime_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            leaked_path = str(workspace / "private" / "executor.log")
            with (
                mock.patch(
                    "xscientist.provider_config.load_workspace_environment",
                    return_value={"loaded": True},
                ),
                mock.patch(
                    "ai_scientist.apps.preflight.check_bfts_config",
                    return_value=[
                        SimpleNamespace(
                            label="Executor image",
                            ok=False,
                            severity="error",
                            detail=f"runtime failed at {leaked_path}",
                        )
                    ],
                ),
            ):
                payload = diagnose(
                    workspace,
                    task="research",
                    provider="zhipu",
                    deep=True,
                )

            serialized = json.dumps(payload)
            self.assertNotIn(str(Path(td).resolve()), serialized)
            self.assertIn("[REDACTED", serialized)
            self.assertFalse(payload["host_paths_disclosed"])


if __name__ == "__main__":
    unittest.main()
