from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from xscientist._version import __version__
from xscientist.cli import main as cli_main
from xscientist.cli import _interactive_start_inputs, _prompt_provider_model
from xscientist.diagnostics import diagnose
from xscientist.onboarding import (
    WORKSPACE_FILES,
    _installed_vcs_source,
    _render_dockerfile,
    _render_readme,
    create_workspace,
)
from xscientist.provider_config import (
    discover_provider_models,
    probe_provider_model,
    validate_provider_model,
)


class OnboardingTests(unittest.TestCase):
    def test_ollama_model_discovery_and_bare_name_normalization(self) -> None:
        response = io.StringIO(
            json.dumps({"models": [{"name": "qwen2.5:7b"}, {"name": "qwen3:1.7b"}]})
        )
        with mock.patch("urllib.request.OpenerDirector.open", return_value=response):
            models = discover_provider_models("ollama")

        self.assertEqual(models[0], "ollama/qwen2.5:7b")
        self.assertEqual(
            validate_provider_model("ollama", "qwen2.5:7b"),
            ("ollama", "ollama/qwen2.5:7b"),
        )
        with self.assertRaisesRegex(ValueError, "not 'zhipu'"):
            validate_provider_model("zhipu", "openai/wrong-provider")

    def test_ollama_discovery_accepts_a_host_without_a_url_scheme(self) -> None:
        response = io.StringIO(json.dumps({"models": [{"name": "qwen2.5:7b"}]}))
        with mock.patch(
            "urllib.request.OpenerDirector.open", return_value=response
        ) as opened:
            models = discover_provider_models(
                "ollama", environ={"OLLAMA_HOST": "127.0.0.1:11434"}
            )

        self.assertEqual(models, ["ollama/qwen2.5:7b"])
        self.assertEqual(
            opened.call_args.args[0].full_url, "http://127.0.0.1:11434/api/tags"
        )

    def test_numbered_local_model_selection_selects_the_displayed_model(self) -> None:
        with (
            mock.patch(
                "xscientist.provider_config.discover_provider_models",
                return_value=["ollama/first", "ollama/second"],
            ),
            mock.patch("builtins.input", return_value="2"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            selected = _prompt_provider_model("ollama")

        self.assertEqual(selected, "ollama/second")

    def test_ollama_probe_requires_reachable_service_and_installed_model(self) -> None:
        response = io.StringIO(json.dumps({"models": [{"name": "qwen2.5:7b"}]}))
        with mock.patch("urllib.request.OpenerDirector.open", return_value=response):
            ready = probe_provider_model("ollama", "ollama/qwen2.5:7b")

        self.assertTrue(ready["ok"])
        self.assertTrue(ready["service_reachable"])
        self.assertTrue(ready["model_available"])

        with mock.patch(
            "urllib.request.OpenerDirector.open",
            side_effect=OSError("not reachable"),
        ):
            unavailable = probe_provider_model("ollama", "ollama/qwen2.5:7b")

        self.assertFalse(unavailable["ok"])
        self.assertIn("ollama serve", unavailable["error"])

    def test_interactive_start_fills_only_missing_first_run_choices(self) -> None:
        parsed = argparse.Namespace(
            non_interactive=False,
            question=None,
            provider=None,
            model=None,
            prepare_only=False,
            data_dir=None,
            allow_synthetic_data=False,
            max_cost_usd=None,
            max_project_tokens=None,
        )
        answers = iter(
            [
                "Does the intervention improve the target?",
                "1",
                "",
                "2",
                "",
            ]
        )
        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", side_effect=lambda _prompt: next(answers)),
            mock.patch(
                "xscientist.dependency_profiles.missing_provider_modules",
                return_value=[],
            ),
            mock.patch(
                "xscientist.provider_config.configured_field_value",
                return_value="configured",
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            _interactive_start_inputs(parsed, new_workspace=True)

        self.assertEqual(parsed.question, "Does the intervention improve the target?")
        self.assertEqual(parsed.provider, "zhipu")
        self.assertEqual(parsed.model, "glm-4-flash")
        self.assertTrue(parsed.allow_synthetic_data)

    def test_root_help_is_progressive_and_advanced_help_is_available(self) -> None:
        concise = io.StringIO()
        advanced = io.StringIO()
        with contextlib.redirect_stdout(concise):
            self.assertEqual(cli_main(["--help"]), 0)
        with contextlib.redirect_stdout(advanced):
            self.assertEqual(cli_main(["help", "--all"]), 0)

        self.assertIn("Start here:", concise.getvalue())
        self.assertIn("runs", concise.getvalue())
        self.assertNotIn("evolution-gate", concise.getvalue())
        self.assertIn("evolution-gate", advanced.getvalue())

    def test_start_requires_an_explicit_data_mode_before_creating_workspace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli_main(
                    ["start", str(workspace), "--question", "Does X affect Y?"]
                )

            self.assertEqual(exit_code, 2)
            self.assertFalse(workspace.exists())
            self.assertIn("--data-dir", stderr.getvalue())

    def test_start_refuses_to_silently_override_the_active_research_actor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            stderr = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(
                        True,
                        "ok",
                        {"username": "ExistingUser"},
                    ),
                ),
                mock.patch(
                    "ai_scientist.utils.auth_session.create_session"
                ) as create_session,
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X affect Y?",
                        "--allow-synthetic-data",
                        "--user",
                        "RequestedUser",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertFalse(workspace.exists())
            self.assertIn(
                "conflicts with the active research identity", stderr.getvalue()
            )
            self.assertIn(
                "xscientist auth login --user RequestedUser", stderr.getvalue()
            )
            create_session.assert_not_called()

    def test_start_reuses_one_workspace_and_forwards_guarded_autopilot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            ready = {
                "schema": "xscientist.doctor.v1",
                "ok": True,
                "configuration_ready": True,
                "runtime_ready": True,
                "task": "research",
                "workspace": ".",
                "checks": {},
                "next_actions": [],
                "host_paths_disclosed": False,
            }
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "tester"}),
                ),
                mock.patch(
                    "xscientist.provider_config.load_workspace_environment",
                    return_value={"loaded": True},
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                mock.patch("xscientist.cli.project_main", return_value=0) as project,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X affect Y?",
                        "--allow-synthetic-data",
                        "--skip-credentials",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((workspace / "research.yaml").is_file())
            argv = project.call_args.args[0]
            self.assertEqual(argv[0], str(workspace.resolve()))
            self.assertIn("--autopilot", argv)
            self.assertIn("--allow-synthetic-data", argv)
            self.assertIn("--research-vcs-strict", argv)

    def test_start_fails_before_provider_use_when_cost_price_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "tester"}),
                ),
                mock.patch(
                    "xscientist.provider_config.load_workspace_environment",
                    return_value={"loaded": True},
                ),
                mock.patch("xscientist.diagnostics.diagnose") as diagnose_mock,
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X affect Y?",
                        "--prepare-only",
                        "--skip-credentials",
                        "--non-interactive",
                        "--max-cost-usd",
                        "1",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["phase"], "budget")
            self.assertEqual(payload["error_code"], "unknown_model_price")
            self.assertFalse(payload["phases"]["budget"]["price_configured"])
            diagnose_mock.assert_not_called()

    def test_start_accepts_an_explicit_price_for_cost_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            ready = {
                "schema": "xscientist.doctor.v1",
                "ok": True,
                "configuration_ready": True,
                "runtime_ready": True,
                "task": "research",
                "workspace": ".",
                "checks": {},
                "next_actions": [],
            }
            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "tester"}),
                ),
                mock.patch(
                    "xscientist.provider_config.load_workspace_environment",
                    return_value={"loaded": True},
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X affect Y?",
                        "--prepare-only",
                        "--skip-credentials",
                        "--non-interactive",
                        "--max-cost-usd",
                        "1",
                        "--price-input-per-million",
                        "0.1",
                        "--price-output-per-million",
                        "0.3",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["phases"]["budget"]["price_configured"])
            self.assertEqual(payload["phases"]["budget"]["price_source"], "workspace")
            config = yaml.safe_load(
                (workspace / "bfts_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                config["llm_budget"]["prices_per_million"]["glm-4-flash"],
                {"input": 0.1, "output": 0.3},
            )

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
                "xscientist[research,zhipu]==${XSCIENTIST_VERSION}",
                dockerfile,
            )
            self.assertIn("ARG XSCIENTIST_INSTALL_MODE=pypi", dockerfile)
            self.assertIn(
                "/tmp/xscientist-build-context[research,zhipu]",
                dockerfile,
            )
            self.assertNotIn("torch --index-url", dockerfile)
            self.assertIn("org.opencontainers.image.revision", dockerfile)
            readme = (workspace / "README.md").read_text()
            self.assertIn("BFTS `default` configuration", readme)
            self.assertIn(
                f"xscientist[research,zhipu]=={__version__}",
                readme,
            )
            self.assertIn("xscientist provider check zhipu", readme)
            self.assertIn("xscientist auth status", readme)
            self.assertIn("xscientist provider add zhipu", readme)
            self.assertIn("xscientist git doctor", readme)
            self.assertIn("--output-root ./outputs", readme)
            self.assertIn(
                "xscientist manager --research-dir ./outputs list-papers", readme
            )
            self.assertIn("xscientist research objects", readme)
            self.assertIn("xscientist research guide", readme)
            self.assertIn("xscientist research dag", readme)
            self.assertNotIn("--research-git", readme)
            self.assertNotIn("\n+  --model", readme)
            self.assertEqual(
                payload["next_steps"][1],
                f'python -m pip install "xscientist[research,zhipu]=={__version__}"',
            )
            self.assertEqual(payload["next_steps"][2], "xscientist git doctor")

    def test_vcs_installs_pin_executor_to_the_exact_safe_commit(self) -> None:
        direct_url = json.dumps(
            {
                "url": "https://github.com/example/XScientist.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "a" * 40,
                },
            }
        )
        distribution = SimpleNamespace(read_text=lambda _name: direct_url)
        with mock.patch(
            "xscientist.onboarding.importlib_metadata.distribution",
            return_value=distribution,
        ):
            self.assertEqual(
                _installed_vcs_source(),
                ("https://github.com/example/XScientist.git", "a" * 40),
            )
            dockerfile = _render_dockerfile("zhipu")
            readme = _render_readme(
                profile="default",
                provider="zhipu",
                model="glm-4-flash",
            )

        self.assertIn(
            "xscientist[research,zhipu] @ "
            "git+https://github.com/example/XScientist.git@" + "a" * 40,
            dockerfile,
        )
        self.assertIn("ARG XSCIENTIST_INSTALL_SOURCE=vcs-commit", dockerfile)
        self.assertIn(
            'org.xscientist.install-source="$XSCIENTIST_INSTALL_SOURCE"',
            dockerfile,
        )
        self.assertNotIn("==${XSCIENTIST_VERSION}", dockerfile)
        self.assertIn(
            "xscientist[research,zhipu] @ "
            "git+https://github.com/example/XScientist.git@" + "a" * 40,
            readme,
        )

    def test_provider_refresh_does_not_create_formatting_only_research_changes(
        self,
    ) -> None:
        from xscientist.research_git import init_repository, repository_status

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            init_repository(
                workspace,
                name="canonical-workspace",
                question="# Canonical setup?\n",
                git_user_name="Setup Test",
                git_user_email="setup@example.invalid",
            )
            with mock.patch.dict("os.environ", {"ZHIPU_API_KEY": "test-key"}):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = cli_main(
                        [
                            "provider",
                            "add",
                            "zhipu",
                            "--workspace",
                            str(workspace),
                            "--non-interactive",
                            "--json",
                        ]
                    )

            self.assertEqual(exit_code, 0, output.getvalue())
            status = repository_status(workspace)
            self.assertEqual(status["staged_paths"], [])
            self.assertEqual(status["eligible_changes"], [])
            self.assertTrue(
                (workspace / "bfts_config.yaml")
                .read_text(encoding="utf-8")
                .startswith("# Generated by xscientist init")
            )

    def test_vcs_executor_source_rejects_urls_that_can_embed_credentials(self) -> None:
        direct_url = json.dumps(
            {
                "url": "https://token@github.com/example/XScientist.git",
                "vcs_info": {"vcs": "git", "commit_id": "b" * 40},
            }
        )
        distribution = SimpleNamespace(read_text=lambda _name: direct_url)
        with mock.patch(
            "xscientist.onboarding.importlib_metadata.distribution",
            return_value=distribution,
        ):
            self.assertIsNone(_installed_vcs_source())

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

    def test_unready_setup_returns_nonzero_for_automation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(False, "missing", None),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--task",
                        "research",
                        "--skip-credentials",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 1)

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

    def test_deep_doctor_only_blocks_on_pdflatex_for_paper_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            with (
                mock.patch(
                    "xscientist.provider_config.load_workspace_environment",
                    return_value={"loaded": True},
                ),
                mock.patch(
                    "ai_scientist.apps.preflight.check_bfts_config",
                    return_value=[],
                ),
                mock.patch("shutil.which", return_value=None),
            ):
                research = diagnose(workspace, task="research", deep=True)
                paper = diagnose(workspace, task="paper", deep=True)

            research_pdflatex = next(
                item
                for item in research["checks"]["runtime"]["results"]
                if item["label"] == "pdflatex"
            )
            paper_pdflatex = next(
                item
                for item in paper["checks"]["runtime"]["results"]
                if item["label"] == "pdflatex"
            )
            self.assertEqual(research_pdflatex["severity"], "warning")
            self.assertIn("optional", research_pdflatex["detail"])
            self.assertTrue(research["checks"]["runtime"]["ok"])
            self.assertEqual(paper_pdflatex["severity"], "error")
            self.assertFalse(paper["checks"]["runtime"]["ok"])
            self.assertIn("paper_compiler_missing", paper["error_codes"])

    def test_human_doctor_renders_every_check_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(
                    ["doctor", "--workspace", str(workspace), "--task", "research"]
                )

            self.assertEqual(exit_code, 1)
            rendered = output.getvalue()
            self.assertIn("Dependencies", rendered)
            self.assertIn("Isolated runtime", rendered)
            self.assertIn("not checked", rendered)
            self.assertIn("Next actions:", rendered)

    def test_human_doctor_distinguishes_configured_from_live_verified(self) -> None:
        report = {
            "schema": "xscientist.doctor.v1",
            "ok": True,
            "configuration_ready": True,
            "runtime_ready": None,
            "task": "research",
            "checks": {
                "provider": {
                    "ok": True,
                    "required": True,
                    "credentials_available": True,
                    "local_probe": {"checked": False},
                },
                "runtime": {"ok": None, "required": True, "checked": False},
            },
            "next_actions": [],
        }
        output = io.StringIO()
        with (
            mock.patch("xscientist.diagnostics.diagnose", return_value=report),
            contextlib.redirect_stdout(output),
        ):
            exit_code = cli_main(["doctor", "--workspace", "."])

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "configured (credentials present; not live-verified)",
            output.getvalue(),
        )
        self.assertIn("Deep runtime: not checked", output.getvalue())


if __name__ == "__main__":
    unittest.main()
