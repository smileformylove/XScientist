from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
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
    WorkspaceInitError,
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
    def test_glm53_workspace_keeps_custom_route_and_optional_sections(self) -> None:
        self.assertEqual(
            validate_provider_model("custom", "glm-5.3"),
            ("openai_compat", "openai_compat/glm-5.3"),
        )
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "glm53-study"
            create_workspace(
                workspace,
                profile="glm53",
                provider="custom",
                model="glm-5.3",
            )

            provider_payload = json.loads(
                (workspace / ".xscientist" / "providers.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(provider_payload["active_provider"], "openai_compat")
            self.assertEqual(
                provider_payload["providers"]["openai_compat"]["model"],
                "openai_compat/glm-5.3",
            )

            config_text = (workspace / "bfts_config.yaml").read_text(encoding="utf-8")
            config = yaml.safe_load(config_text)
            for role in ("code", "feedback", "vlm_feedback", "summary"):
                self.assertEqual(
                    config["agent"][role]["model"], "openai_compat/glm-5.3"
                )
            self.assertEqual(config["report"]["model"], "openai_compat/glm-5.3")
            self.assertNotIn("select_node", config["agent"])
            self.assertNotIn("OPENAI_COMPAT_API_KEY", config_text)
            self.assertNotIn("OPENAI_COMPAT_BASE_URL", config_text)

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
            mock.patch(
                "xscientist.provider_config.discover_provider_models",
                return_value=[],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            _interactive_start_inputs(parsed, new_workspace=True)

        self.assertEqual(parsed.question, "Does the intervention improve the target?")
        self.assertEqual(parsed.provider, "zhipu")
        self.assertEqual(parsed.model, "glm-4-flash")
        self.assertTrue(parsed.allow_synthetic_data)

    def test_start_json_implies_noninteractive_even_in_a_tty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sentinel = "SENTINEL_PRIVATE_WORKSPACE_INPUT"
            workspace = Path(td) / sentinel
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(sys.stdin, "isatty", return_value=True),
                mock.patch(
                    "builtins.input",
                    side_effect=AssertionError("JSON mode must not prompt"),
                ) as prompted,
                mock.patch(
                    "getpass.getpass",
                    side_effect=AssertionError("JSON mode must not request secrets"),
                ) as secret_prompted,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = cli_main(["start", str(workspace), "--json"])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["phase"], "input")
            self.assertEqual(payload["workspace"], ".")
            self.assertIn("--question", payload["error"])
            self.assertNotIn(sentinel, stdout.getvalue())
            self.assertNotIn(str(Path(td).resolve()), stdout.getvalue())
            prompted.assert_not_called()
            secret_prompted.assert_not_called()
            self.assertFalse(workspace.exists())

    def test_start_json_redacts_host_paths_from_preparation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "test-researcher"}),
                ),
                mock.patch(
                    "xscientist.onboarding.create_workspace",
                    side_effect=OSError(f"cannot write {workspace / 'private.txt'}"),
                ),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X change Y?",
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 2)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["phase"], "prepare")
            self.assertNotIn(str(Path(td).resolve()), output.getvalue())
            self.assertIn("[REDACTED_PATH]", payload["error"])
            self.assertFalse(workspace.exists())

    def test_init_and_setup_json_errors_are_portable_and_path_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sentinel = "SENTINEL_PRIVATE_INIT_SETUP"
            for command in ("init", "setup"):
                with self.subTest(command=command):
                    workspace = root / f"{sentinel}-{command}"
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    argv = [command, str(workspace), "--json"]
                    if command == "setup":
                        argv.extend(["--task", "protocol", "--skip-credentials"])
                    with (
                        mock.patch(
                            "xscientist.onboarding.create_workspace",
                            side_effect=OSError(
                                f"cannot write {workspace / 'private-output'}"
                            ),
                        ),
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        code = cli_main(argv)

                    self.assertEqual(code, 2)
                    self.assertEqual(stderr.getvalue(), "")
                    rendered = stdout.getvalue()
                    payload = json.loads(rendered)
                    self.assertEqual(payload["schema"], f"xscientist.{command}.v1")
                    self.assertEqual(payload["workspace"], ".")
                    self.assertIn("[REDACTED_PATH]", payload["error"])
                    self.assertNotIn(sentinel, rendered)
                    self.assertNotIn(str(root.resolve()), rendered)

    def test_start_json_redacts_the_final_execution_error_and_workspace_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            sentinel = "SENTINEL_PRIVATE_WORKSPACE_EXECUTION"
            workspace = Path(td) / sentinel
            ready = {
                "schema": "xscientist.doctor.v1",
                "ok": True,
                "configuration_ready": True,
                "runtime_ready": True,
                "checks": {},
                "next_actions": [],
            }

            def failed_project(_args):
                print(
                    f"execution failed under {workspace / 'private-output'}",
                    file=sys.stderr,
                )
                return 1

            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "test-researcher"}),
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                mock.patch("xscientist.cli.project_main", side_effect=failed_project),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X change Y?",
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--allow-synthetic-data",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["phase"], "research")
            self.assertEqual(payload["workspace"], ".")
            self.assertEqual(payload["project"], ".")
            self.assertIn("{workspace}", payload["research_dag"])
            self.assertNotIn(sentinel, output.getvalue())
            self.assertNotIn(str(Path(td).resolve()), output.getvalue())
            self.assertIn("[REDACTED_PATH]", payload["error"][0])

    def test_root_help_is_progressive_and_advanced_help_is_available(self) -> None:
        concise = io.StringIO()
        advanced = io.StringIO()
        with contextlib.redirect_stdout(concise):
            self.assertEqual(cli_main(["--help"]), 0)
        with contextlib.redirect_stdout(advanced):
            self.assertEqual(cli_main(["help", "--all"]), 0)

        self.assertIn("Start here:", concise.getvalue())
        self.assertIn("explore", concise.getvalue())
        self.assertIn("runs", concise.getvalue())
        self.assertIn("audit", concise.getvalue())
        self.assertIn("history", concise.getvalue())
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
                        "--provider",
                        "zhipu",
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
            for flag in (
                "--model-ideation",
                "--model-agg-plots",
                "--model-writeup",
                "--model-writeup-small",
                "--model-citation",
                "--model-review",
                "--idea-rank-model",
                "--quality-model",
            ):
                self.assertEqual(argv[argv.index(flag) + 1], "glm-4-flash")

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
                        "--provider",
                        "zhipu",
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
                        "--provider",
                        "zhipu",
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
            self.assertEqual(payload["schema"], "xscientist.start.v1")
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
            self.assertEqual(payload["schema"], "xscientist.init.v1")
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

    def test_create_workspace_rolls_back_late_writes_with_bytes_and_mode(self) -> None:
        import xscientist.onboarding as onboarding_module

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fresh = root / "fresh"
            original_writer = onboarding_module.atomic_write_text
            calls = 0

            def fail_late(path: Path, content: str, **kwargs: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise PermissionError("late injected write failure")
                original_writer(path, content, **kwargs)

            with mock.patch(
                "xscientist.onboarding.atomic_write_text",
                side_effect=fail_late,
            ):
                with self.assertRaises(PermissionError):
                    create_workspace(fresh)
            self.assertFalse(fresh.exists())

            existing = root / "existing"
            create_workspace(existing)
            readme = existing / "README.md"
            readme.write_bytes(b"custom\r\nbytes\xff\n")
            readme.chmod(0o640)
            before = {
                relative: (
                    (existing / relative).read_bytes(),
                    (existing / relative).stat().st_mode & 0o7777,
                )
                for relative in WORKSPACE_FILES
            }
            calls = 0
            with mock.patch(
                "xscientist.onboarding.atomic_write_text",
                side_effect=fail_late,
            ):
                with self.assertRaises(PermissionError):
                    create_workspace(existing, force=True, profile="deep")
            after = {
                relative: (
                    (existing / relative).read_bytes(),
                    (existing / relative).stat().st_mode & 0o7777,
                )
                for relative in WORKSPACE_FILES
            }
            self.assertEqual(after, before)

    def test_create_workspace_rollback_preserves_concurrent_managed_write(
        self,
    ) -> None:
        import xscientist.onboarding as onboarding_module

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            original_writer = onboarding_module.atomic_write_text
            calls = 0

            def mutate_then_fail(path: Path, content: str, **kwargs: object) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    (workspace / ".dockerignore").write_text(
                        "concurrent user data\n",
                        encoding="utf-8",
                    )
                    raise PermissionError("late injected write failure")
                original_writer(path, content, **kwargs)

            with mock.patch(
                "xscientist.onboarding.atomic_write_text",
                side_effect=mutate_then_fail,
            ):
                with self.assertRaisesRegex(
                    WorkspaceInitError,
                    "rollback was incomplete",
                ):
                    create_workspace(workspace)

            self.assertEqual(
                (workspace / ".dockerignore").read_text(encoding="utf-8"),
                "concurrent user data\n",
            )

    def test_create_workspace_refuses_concurrent_leaf_before_replace(self) -> None:
        import xscientist.onboarding as onboarding_module

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original_writer = onboarding_module.atomic_write_text

            for existing in (False, True):
                with self.subTest(existing=existing):
                    workspace = root / ("existing" if existing else "fresh")
                    if existing:
                        create_workspace(workspace)
                    calls = 0

                    def change_next_leaf(
                        path: Path, content: str, **kwargs: object
                    ) -> None:
                        nonlocal calls
                        calls += 1
                        original_writer(path, content, **kwargs)
                        if calls == 1:
                            (workspace / ".env.example").write_text(
                                "concurrent user data\n",
                                encoding="utf-8",
                            )

                    with mock.patch(
                        "xscientist.onboarding.atomic_write_text",
                        side_effect=change_next_leaf,
                    ):
                        with self.assertRaisesRegex(
                            WorkspaceInitError,
                            "changed concurrently: \\.env\\.example",
                        ):
                            create_workspace(
                                workspace,
                                force=existing,
                                profile="deep",
                            )

                    self.assertEqual(
                        (workspace / ".env.example").read_text(encoding="utf-8"),
                        "concurrent user data\n",
                    )

    def test_generated_gitignore_block_is_complete_idempotent_and_repaired(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            gitignore = workspace / ".gitignore"
            initial = gitignore.read_text(encoding="utf-8")
            self.assertIn("# BEGIN XScientist managed workspace ignores v1\n", initial)
            self.assertIn("# END XScientist managed workspace ignores v1\n", initial)
            self.assertIn(".secrets/\n", initial)

            create_workspace(workspace, force=True)
            self.assertEqual(gitignore.read_text(encoding="utf-8"), initial)

            damaged = initial.replace(".secrets/\n", "", 1)
            gitignore.write_text(damaged, encoding="utf-8")
            create_workspace(workspace, force=True)
            repaired = gitignore.read_text(encoding="utf-8")
            self.assertTrue(repaired.startswith(damaged))
            self.assertGreaterEqual(repaired.count(".secrets/\n"), 1)
            complete_block = repaired[repaired.rfind("# BEGIN XScientist") :]
            self.assertIn(".env.*\n", complete_block)
            self.assertIn("!.env.example\n", complete_block)
            self.assertIn(".secrets/\n", complete_block)
            self.assertIn("outputs/\n", complete_block)

    def test_workspace_creation_refuses_a_symlinked_metadata_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "study"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            try:
                (workspace / ".xscientist").symlink_to(
                    outside,
                    target_is_directory=True,
                )
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")

            with self.assertRaisesRegex(
                WorkspaceInitError,
                "symlinked .xscientist",
            ):
                create_workspace(workspace)

            self.assertEqual(list(outside.iterdir()), [])

    def test_workspace_creation_refuses_existing_and_dangling_root_symlinks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for target_exists in (True, False):
                with self.subTest(target_exists=target_exists):
                    target = root / f"outside-{target_exists}"
                    sentinel = target / "bfts_config.yaml"
                    if target_exists:
                        target.mkdir()
                        sentinel.write_text(
                            "SENTINEL_EXTERNAL_CONFIG\n",
                            encoding="utf-8",
                        )
                    workspace = root / f"study-{target_exists}"
                    try:
                        workspace.symlink_to(target, target_is_directory=True)
                    except OSError as exc:
                        self.skipTest(f"directory symlinks unavailable: {exc}")

                    with self.assertRaisesRegex(
                        WorkspaceInitError,
                        "symlinked workspace path",
                    ):
                        create_workspace(workspace, force=True)

                    self.assertTrue(workspace.is_symlink())
                    if target_exists:
                        self.assertEqual(
                            sentinel.read_text(encoding="utf-8"),
                            "SENTINEL_EXTERNAL_CONFIG\n",
                        )
                        self.assertEqual(list(target.iterdir()), [sentinel])
                    else:
                        self.assertFalse(target.exists())

    def test_workspace_commands_reject_root_symlinks_with_stable_json(self) -> None:
        commands = (
            (
                "init",
                "xscientist.init.v1",
                lambda workspace: [
                    "init",
                    str(workspace),
                    "--force",
                    "--json",
                ],
            ),
            (
                "setup",
                "xscientist.setup.v1",
                lambda workspace: [
                    "setup",
                    str(workspace),
                    "--task",
                    "protocol",
                    "--skip-credentials",
                    "--force",
                    "--json",
                ],
            ),
            (
                "start",
                "xscientist.start.v1",
                lambda workspace: [
                    "start",
                    str(workspace),
                    "--question",
                    "Does X change Y?",
                    "--provider",
                    "ollama",
                    "--model",
                    "ollama/qwen2.5:7b",
                    "--prepare-only",
                    "--skip-credentials",
                    "--force",
                    "--json",
                ],
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for target_exists in (True, False):
                for command, schema, argv in commands:
                    with self.subTest(
                        command=command,
                        target_exists=target_exists,
                    ):
                        suffix = f"{command}-{target_exists}"
                        target = root / f"outside-{suffix}"
                        sentinel = target / "bfts_config.yaml"
                        if target_exists:
                            target.mkdir()
                            sentinel.write_text(
                                "SENTINEL_EXTERNAL_CONFIG\n",
                                encoding="utf-8",
                            )
                        workspace = root / f"study-{suffix}"
                        try:
                            workspace.symlink_to(target, target_is_directory=True)
                        except OSError as exc:
                            self.skipTest(f"directory symlinks unavailable: {exc}")

                        stdout, stderr = io.StringIO(), io.StringIO()
                        with (
                            contextlib.redirect_stdout(stdout),
                            contextlib.redirect_stderr(stderr),
                        ):
                            code = cli_main(argv(workspace))

                        self.assertEqual(code, 2)
                        payload = json.loads(stdout.getvalue())
                        self.assertEqual(payload["schema"], schema)
                        self.assertFalse(payload["ok"])
                        self.assertIn("symlinked workspace path", payload["error"])
                        self.assertEqual(stderr.getvalue(), "")
                        self.assertTrue(workspace.is_symlink())
                        if target_exists:
                            self.assertEqual(
                                sentinel.read_text(encoding="utf-8"),
                                "SENTINEL_EXTERNAL_CONFIG\n",
                            )
                            self.assertEqual(list(target.iterdir()), [sentinel])
                        else:
                            self.assertFalse(target.exists())

    def test_workspace_creation_refuses_a_symlinked_parent_component(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for target_exists in (True, False):
                with self.subTest(target_exists=target_exists):
                    target = root / f"parent-target-{target_exists}"
                    if target_exists:
                        target.mkdir()
                        (target / "sentinel.txt").write_text(
                            "EXTERNAL SENTINEL\n",
                            encoding="utf-8",
                        )
                    linked_parent = root / f"linked-parent-{target_exists}"
                    try:
                        linked_parent.symlink_to(target, target_is_directory=True)
                    except OSError as exc:
                        self.skipTest(f"directory symlinks unavailable: {exc}")

                    with self.assertRaisesRegex(
                        WorkspaceInitError,
                        "symlinked workspace path",
                    ):
                        create_workspace(linked_parent / "study", force=True)

                    self.assertFalse((target / "study").exists())
                    if target_exists:
                        self.assertEqual(
                            (target / "sentinel.txt").read_text(encoding="utf-8"),
                            "EXTERNAL SENTINEL\n",
                        )

    def test_workspace_commands_reject_parent_symlinks_with_stable_json(self) -> None:
        commands = (
            (
                "init",
                "xscientist.init.v1",
                lambda workspace: ["init", str(workspace), "--force", "--json"],
            ),
            (
                "setup",
                "xscientist.setup.v1",
                lambda workspace: [
                    "setup",
                    str(workspace),
                    "--task",
                    "protocol",
                    "--skip-credentials",
                    "--force",
                    "--json",
                ],
            ),
            (
                "start",
                "xscientist.start.v1",
                lambda workspace: [
                    "start",
                    str(workspace),
                    "--question",
                    "Does X change Y?",
                    "--provider",
                    "ollama",
                    "--model",
                    "ollama/qwen2.5:7b",
                    "--prepare-only",
                    "--skip-credentials",
                    "--force",
                    "--json",
                ],
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for command, schema, argv in commands:
                with self.subTest(command=command):
                    target = root / f"outside-parent-{command}"
                    target.mkdir()
                    sentinel = target / "sentinel.txt"
                    sentinel.write_text("EXTERNAL SENTINEL\n", encoding="utf-8")
                    linked_parent = root / f"linked-parent-{command}"
                    try:
                        linked_parent.symlink_to(target, target_is_directory=True)
                    except OSError as exc:
                        self.skipTest(f"directory symlinks unavailable: {exc}")
                    workspace = linked_parent / "study"

                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        code = cli_main(argv(workspace))

                    self.assertEqual(code, 2)
                    payload = json.loads(stdout.getvalue())
                    self.assertEqual(payload["schema"], schema)
                    self.assertFalse(payload["ok"])
                    self.assertIn("symlinked workspace path", payload["error"])
                    self.assertEqual(stderr.getvalue(), "")
                    self.assertNotIn(str(target), stdout.getvalue())
                    self.assertNotIn(str(linked_parent), stdout.getvalue())
                    self.assertFalse(workspace.exists())
                    self.assertEqual(
                        sentinel.read_text(encoding="utf-8"),
                        "EXTERNAL SENTINEL\n",
                    )

    def test_workspace_creation_refuses_a_symlinked_managed_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "study"
            outside = root / "outside.json"
            outside.write_text("SENTINEL_EXTERNAL_PROVIDER\n", encoding="utf-8")
            (workspace / ".xscientist").mkdir(parents=True)
            try:
                (workspace / ".xscientist" / "providers.json").symlink_to(outside)
            except OSError as exc:
                self.skipTest(f"file symlinks unavailable: {exc}")

            with self.assertRaisesRegex(
                WorkspaceInitError,
                "symlinked managed workspace",
            ):
                create_workspace(workspace, preserve_existing=True)

            self.assertEqual(
                outside.read_text(encoding="utf-8"),
                "SENTINEL_EXTERNAL_PROVIDER\n",
            )

    def test_start_refuses_symlinked_managed_files_before_reading(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for index, relative in enumerate(
                (
                    ".xscientist/providers.json",
                    "bfts_config.yaml",
                    "research.yaml",
                )
            ):
                with self.subTest(relative=relative):
                    workspace = root / f"study-{index}"
                    create_workspace(workspace)
                    external = root / f"external-{index}.txt"
                    marker = f"SENTINEL_EXTERNAL_{index}\n"
                    external.write_text(marker, encoding="utf-8")
                    target = workspace / relative
                    target.unlink(missing_ok=True)
                    try:
                        target.symlink_to(external)
                    except OSError as exc:
                        self.skipTest(f"file symlinks unavailable: {exc}")
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = cli_main(
                            [
                                "start",
                                str(workspace),
                                "--question",
                                "Does X change Y?",
                                "--prepare-only",
                                "--json",
                            ]
                        )

                    self.assertEqual(code, 2)
                    payload = json.loads(output.getvalue())
                    self.assertIn("must not be a symlink", payload["error"])
                    self.assertNotIn(marker.strip(), output.getvalue())
                    self.assertEqual(external.read_text(encoding="utf-8"), marker)

    def test_init_json_rejects_research_yaml_symlink_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "study"
            workspace.mkdir()
            external = root / "external-research.yaml"
            external.write_text("sentinel: keep\n", encoding="utf-8")
            research_config = workspace / "research.yaml"
            try:
                research_config.symlink_to(external)
            except OSError as exc:
                self.skipTest(f"file symlinks unavailable: {exc}")

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli_main(["init", str(workspace), "--json"])

            self.assertEqual(code, 2)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "xscientist.init.v1")
            self.assertFalse(payload["ok"])
            self.assertIn("must not be a symlink", payload["error"])
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(external.read_text(encoding="utf-8"), "sentinel: keep\n")
            self.assertTrue(research_config.is_symlink())

    def test_start_preserves_non_research_topic_and_rolls_back_failed_prepare(
        self,
    ) -> None:
        from xscientist.onboarding import _render_topic

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            conflicting = root / "conflicting"
            conflicting.mkdir()
            topic = conflicting / "topic.md"
            topic.write_text("# Research question\n\nUser-owned question.\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "start",
                        str(conflicting),
                        "--question",
                        "Different question?",
                        "--provider",
                        "zhipu",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("refusing to overwrite", output.getvalue())
            self.assertEqual(
                topic.read_text(), "# Research question\n\nUser-owned question.\n"
            )
            self.assertEqual(list(conflicting.iterdir()), [topic])

            packaged = root / "packaged"
            packaged.mkdir()
            packaged_topic = packaged / "topic.md"
            packaged_topic.write_text(_render_topic(), encoding="utf-8")
            unready = {
                "schema": "xscientist.doctor.v1",
                "ok": False,
                "configuration_ready": False,
                "runtime_ready": False,
                "checks": {},
                "next_actions": ["repair runtime"],
            }
            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "tester"}),
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=unready),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "start",
                        str(packaged),
                        "--question",
                        "Does X change Y?",
                        "--provider",
                        "zhipu",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 1, output.getvalue())
            self.assertEqual(packaged_topic.read_text(), _render_topic())
            self.assertEqual(list(packaged.iterdir()), [packaged_topic])

    def test_start_rejects_private_question_before_any_workspace_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            private_question = f"Does {workspace / 'secret.csv'} change Y?"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        private_question,
                        "--provider",
                        "zhipu",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertFalse(workspace.exists())
            self.assertNotIn(str(workspace), output.getvalue())
            self.assertIn("host-local paths", output.getvalue())

    def test_start_does_not_commit_preexisting_unprovenanced_research_drafts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            draft = workspace / "claims" / "user-draft.md"
            draft.parent.mkdir(parents=True)
            draft.write_text("user-owned draft\n", encoding="utf-8")
            ready = {
                "schema": "xscientist.doctor.v1",
                "ok": True,
                "configuration_ready": True,
                "runtime_ready": True,
                "checks": {},
                "next_actions": [],
            }
            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "tester"}),
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X change Y?",
                        "--provider",
                        "zhipu",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, output.getvalue())
            tracked = subprocess.run(
                ["git", "ls-files"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            self.assertNotIn("claims/user-draft.md", tracked)
            self.assertEqual(draft.read_text(), "user-owned draft\n")

    def test_scientific_path_privacy_gate_avoids_runtime_and_url_false_positives(
        self,
    ) -> None:
        from ai_scientist.utils.privacy import scan_file
        from xscientist.research_git import ResearchGitError
        from xscientist.research_journey import start_guided_research

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            hypothesis = workspace / "hypotheses" / "candidate.md"
            hypothesis.parent.mkdir(parents=True)
            hypothesis.write_text(
                "Compare https://example.org/a/b and ratio 1/2.\n",
                encoding="utf-8",
            )
            self.assertNotIn(
                "absolute_path",
                {finding.rule for finding in scan_file(hypothesis, root=workspace)},
            )
            runtime = workspace / "Dockerfile.executor"
            runtime.write_text(
                "#!/usr/bin/env sh\nRUN rm -rf /var/lib/apt/lists/* /tmp/build\n",
                encoding="utf-8",
            )
            self.assertNotIn(
                "absolute_path",
                {finding.rule for finding in scan_file(runtime, root=workspace)},
            )
            hypothesis.write_text(
                "Load /private/var/folders/private.csv before testing.\n",
                encoding="utf-8",
            )
            self.assertIn(
                "absolute_path",
                {finding.rule for finding in scan_file(hypothesis, root=workspace)},
            )

            guided = Path(td) / "guided"
            with self.assertRaisesRegex(ResearchGitError, "private literals"):
                start_guided_research(
                    guided,
                    question="Does X change Y?",
                    hypothesis=f"Use {guided / 'secret.csv'} to predict Y",
                    falsifier="No improvement over baseline",
                )
            self.assertFalse(guided.exists())

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
            self.assertEqual(payload["schema"], "xscientist.setup.v1")
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
            provider_config = json.loads(
                (workspace / ".xscientist" / "providers.json").read_text()
            )
            self.assertIsNone(provider_config["active_provider"])
            self.assertEqual(provider_config["providers"], {})
            dockerfile = (workspace / "Dockerfile.executor").read_text()
            self.assertIn(f"xscientist==${{XSCIENTIST_VERSION}}", dockerfile)
            self.assertNotIn("torch --index-url", dockerfile)
            self.assertNotIn("xscientist[research", dockerfile)
            generated_readme = (workspace / "README.md").read_text()
            self.assertIn("provider-neutral", generated_readme)
            self.assertNotIn("xscientist provider add zhipu", generated_readme)
            serialized = json.dumps(payload)
            self.assertNotIn(str(Path(td).resolve()), serialized)

    def test_setup_placeholder_can_be_established_once_by_start(self) -> None:
        from xscientist.research_vcs import ResearchRepository

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "setup",
                            str(workspace),
                            "--task",
                            "protocol",
                            "--skip-credentials",
                            "--json",
                        ]
                    ),
                    0,
                )
            ready = {
                "schema": "xscientist.doctor.v1",
                "ok": True,
                "configuration_ready": True,
                "runtime_ready": True,
                "checks": {},
                "next_actions": [],
            }
            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "test-researcher"}),
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X change Y?",
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["phases"]["research_vcs"]["question_established"])
            expected = "# Research question\n\nDoes X change Y?\n"
            self.assertEqual((workspace / "question.md").read_text(), expected)
            self.assertEqual((workspace / "topic.md").read_text(), expected)
            repository = ResearchRepository(workspace)
            self.assertEqual(repository.status()["eligible_changes"], [])
            changed_paths = set(repository.status()["last_checkpoint"]["changed_paths"])
            self.assertEqual(changed_paths, {"question.md", "topic.md"})

    def test_force_refresh_preserves_scientific_sources_and_init_refuses_them(
        self,
    ) -> None:
        from xscientist.research_vcs import ResearchRepository

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "setup",
                            str(workspace),
                            "--task",
                            "protocol",
                            "--skip-credentials",
                            "--json",
                        ]
                    ),
                    0,
                )
            ready = {
                "schema": "xscientist.doctor.v1",
                "ok": True,
                "configuration_ready": True,
                "runtime_ready": True,
                "checks": {},
                "next_actions": [],
            }
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "test-researcher"}),
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    cli_main(
                        [
                            "start",
                            str(workspace),
                            "--question",
                            "Does X change Y?",
                            "--provider",
                            "ollama",
                            "--model",
                            "ollama/qwen2.5:7b",
                            "--prepare-only",
                            "--skip-credentials",
                            "--json",
                        ]
                    ),
                    0,
                )

            question_before = (workspace / "question.md").read_bytes()
            topic_before = (workspace / "topic.md").read_bytes()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--force",
                        "--profile",
                        "deep",
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, output.getvalue())
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["workspace_action"], "refreshed")
            self.assertFalse(payload["workspace_created"])
            self.assertEqual((workspace / "question.md").read_bytes(), question_before)
            self.assertEqual((workspace / "topic.md").read_bytes(), topic_before)
            repository = ResearchRepository(workspace)
            status = repository.status()
            self.assertEqual(status["eligible_changes"], [])
            self.assertEqual(status["staged_paths"], [])
            changed = set(status["last_checkpoint"]["changed_paths"])
            self.assertIn("bfts_config.yaml", changed)
            self.assertNotIn("question.md", changed)
            self.assertNotIn("topic.md", changed)

            output = io.StringIO()
            errors = io.StringIO()
            with (
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(errors),
            ):
                init_code = cli_main(["init", str(workspace), "--force", "--json"])
            self.assertEqual(init_code, 2)
            self.assertEqual(errors.getvalue(), "")
            self.assertIn("existing Research VCS", output.getvalue())
            self.assertEqual((workspace / "question.md").read_bytes(), question_before)
            self.assertEqual((workspace / "topic.md").read_bytes(), topic_before)

    def test_placeholder_privacy_failure_restores_both_sources_and_index(self) -> None:
        from xscientist.research_vcs import ResearchRepository

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "setup",
                            str(workspace),
                            "--task",
                            "protocol",
                            "--skip-credentials",
                            "--json",
                        ]
                    ),
                    0,
                )
            repository = ResearchRepository(workspace)
            checkpoint_before = repository.status()["last_checkpoint"]["checkpoint_id"]
            question_path = workspace / "question.md"
            topic_path = workspace / "topic.md"
            question_before = question_path.read_bytes()
            topic_before = topic_path.read_bytes()
            private_question = f"Does the private file {workspace / 'secret.csv'} help?"
            output = io.StringIO()
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "test-researcher"}),
                ),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        private_question,
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertNotIn(str(workspace), output.getvalue())
            self.assertEqual(question_path.read_bytes(), question_before)
            self.assertEqual(topic_path.read_bytes(), topic_before)
            status = repository.status()
            self.assertEqual(
                status["last_checkpoint"]["checkpoint_id"], checkpoint_before
            )
            self.assertEqual(status["eligible_changes"], [])
            self.assertEqual(status["staged_paths"], [])

    def test_existing_git_setup_checkpoints_only_generated_runtime_before_start(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            workspace.mkdir()

            def git(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", *args],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            git("init", "-b", "main")
            git("config", "user.name", "Test Researcher")
            git("config", "user.email", "test@example.invalid")
            note = workspace / "user-notes.txt"
            note.write_text("baseline\n", encoding="utf-8")
            git("add", "user-notes.txt")
            git("commit", "-m", "user baseline")
            note.write_text("baseline\nprivate draft\n", encoding="utf-8")
            scratch = workspace / "user-scratch.txt"
            scratch.write_text("do not checkpoint\n", encoding="utf-8")

            setup_output = io.StringIO()
            with contextlib.redirect_stdout(setup_output):
                setup_code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(setup_code, 0, setup_output.getvalue())
            setup_payload = json.loads(setup_output.getvalue())
            self.assertTrue(setup_payload["research_vcs"].get("setup_checkpoint_id"))
            tracked = set(git("ls-files").stdout.splitlines())
            generated_runtime = set(WORKSPACE_FILES) - {".gitignore"}
            self.assertTrue(generated_runtime.issubset(tracked))
            self.assertNotIn("user-scratch.txt", tracked)
            self.assertIn("private draft", note.read_text(encoding="utf-8"))

            ready = {
                "schema": "xscientist.doctor.v1",
                "ok": True,
                "configuration_ready": True,
                "runtime_ready": True,
                "checks": {},
                "next_actions": [],
            }
            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=(True, "ok", {"username": "test-researcher"}),
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                start_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X change Y?",
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(start_code, 0)
            expected = "# Research question\n\nDoes X change Y?\n"
            self.assertEqual((workspace / "question.md").read_text(), expected)
            self.assertEqual((workspace / "topic.md").read_text(), expected)
            self.assertIn("private draft", note.read_text(encoding="utf-8"))
            self.assertEqual(scratch.read_text(encoding="utf-8"), "do not checkpoint\n")
            self.assertNotIn(
                "user-scratch.txt", set(git("ls-files").stdout.splitlines())
            )
            self.assertIn(
                "user-notes.txt", git("diff", "--name-only").stdout.splitlines()
            )

    def test_setup_preflight_preserves_staged_and_unknown_tracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            workspace.mkdir()

            def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", *args],
                    cwd=workspace,
                    check=check,
                    capture_output=True,
                    text=True,
                )

            git("init", "-b", "main")
            readme = workspace / "README.md"
            gitignore = workspace / ".gitignore"
            readme.write_text("custom project readme\n", encoding="utf-8")
            gitignore.write_text("custom-cache/\n", encoding="utf-8")
            git("add", "README.md", ".gitignore")
            git(
                "-c",
                "user.name=Test Researcher",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "baseline",
            )
            pending = workspace / "user.txt"
            pending.write_text("pending staged work\n", encoding="utf-8")
            git("add", "user.txt")
            gitignore_before = gitignore.read_bytes()
            readme_before = readme.read_bytes()

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--force",
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("staged work", json.loads(output.getvalue())["error"])
            self.assertEqual(readme.read_bytes(), readme_before)
            self.assertEqual(gitignore.read_bytes(), gitignore_before)
            self.assertEqual(
                git("diff", "--cached", "--name-only").stdout, "user.txt\n"
            )
            self.assertFalse((workspace / "research.yaml").exists())
            self.assertFalse((workspace / "topic.md").exists())

            git("reset")
            pending.unlink()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--force",
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, output.getvalue())
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["workspace_action"], "initialized")
            self.assertFalse(payload["workspace_created"])
            self.assertEqual(readme.read_bytes(), readme_before)
            self.assertTrue(gitignore.read_text().startswith("custom-cache/\n"))
            self.assertIn(
                "# BEGIN XScientist managed workspace ignores v1",
                gitignore.read_text(),
            )
            self.assertEqual(git("status", "--porcelain").stdout, "")

    def test_setup_failure_is_atomic_for_non_git_privacy_and_question_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            privacy_workspace = root / "privacy"
            unsafe = privacy_workspace / "hypotheses" / "unsafe.json"
            unsafe.parent.mkdir(parents=True)
            unsafe_payload = json.dumps(
                {"private_path": str(privacy_workspace / "private.csv")}
            )
            unsafe.write_text(unsafe_payload, encoding="utf-8")

            output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.diagnostics.diagnose",
                    side_effect=OSError("late diagnose failure"),
                ),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "setup",
                        str(privacy_workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2, output.getvalue())
            self.assertEqual(unsafe.read_text(encoding="utf-8"), unsafe_payload)
            self.assertEqual(
                {
                    path.relative_to(privacy_workspace).as_posix()
                    for path in privacy_workspace.rglob("*")
                },
                {"hypotheses", "hypotheses/unsafe.json"},
            )
            self.assertFalse((privacy_workspace / ".git").exists())

            question_workspace = root / "question"
            question_workspace.mkdir()
            question = question_workspace / "question.md"
            question.write_bytes(b"# Research question\n\nReal question?\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(question_workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("without Research VCS provenance", output.getvalue())
            self.assertEqual(
                list(question_workspace.iterdir()),
                [question],
            )

    def test_setup_does_not_commit_preexisting_unprovenanced_research_drafts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            draft = workspace / "hypotheses" / "user-draft.json"
            draft.parent.mkdir(parents=True)
            draft.write_text('{"draft": "user owned"}\n', encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, output.getvalue())
            tracked = subprocess.run(
                ["git", "ls-files"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            self.assertNotIn("hypotheses/user-draft.json", tracked)
            self.assertEqual(draft.read_text(), '{"draft": "user owned"}\n')

    def test_setup_and_start_preserve_a_pending_native_research_stage(self) -> None:
        from xscientist.research_vcs import ResearchRepository

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "setup",
                            str(workspace),
                            "--task",
                            "protocol",
                            "--skip-credentials",
                            "--json",
                        ]
                    ),
                    0,
                )
            repository = ResearchRepository(workspace)
            pending = workspace / "hypotheses" / "pending.json"
            pending.parent.mkdir(exist_ok=True)
            pending.write_text('{"pending": true}\n', encoding="utf-8")
            repository.stage(["hypotheses/pending.json"])
            before = repository.status()

            for argv in (
                [
                    "setup",
                    str(workspace),
                    "--force",
                    "--task",
                    "protocol",
                    "--skip-credentials",
                    "--json",
                ],
                [
                    "start",
                    str(workspace),
                    "--question",
                    "Does X change Y?",
                    "--prepare-only",
                    "--json",
                ],
            ):
                with self.subTest(command=argv[0]):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = cli_main(argv)
                    self.assertEqual(code, 2, output.getvalue())
                    self.assertIn("native Research VCS stage", output.getvalue())
                    current = repository.status()
                    self.assertEqual(current["head"], before["head"])
                    self.assertEqual(
                        current["research_stage"]["paths"],
                        ["hypotheses/pending.json"],
                    )
                    self.assertEqual(pending.read_text(), '{"pending": true}\n')

    def test_failed_setup_preserves_gitfile_identity_checkpoints_and_environment(
        self,
    ) -> None:
        import os

        from xscientist import provider_config as provider_config_module

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            main_repo = root / "main"
            worktree = root / "worktree"
            main_repo.mkdir()

            def main_git(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", *args],
                    cwd=main_repo,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            main_git("init", "-b", "main")
            (main_repo / "baseline.txt").write_text("baseline\n", encoding="utf-8")
            main_git("add", "baseline.txt")
            main_git(
                "-c",
                "user.name=Test Researcher",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "baseline",
            )
            main_git("worktree", "add", "-b", "audit-worktree", str(worktree))
            gitfile = worktree / ".git"
            self.assertTrue(gitfile.is_file())
            gitfile_before = (
                gitfile.read_bytes(),
                gitfile.stat().st_mode & 0o7777,
            )
            unsafe = worktree / "hypotheses" / "unsafe.json"
            unsafe.parent.mkdir()
            unsafe.write_text(
                json.dumps({"path": str(worktree / "private.csv")}),
                encoding="utf-8",
            )
            head_before = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=worktree,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.diagnostics.diagnose",
                    side_effect=OSError("late diagnose failure"),
                ),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "setup",
                        str(worktree),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2, output.getvalue())
            self.assertEqual(
                (gitfile.read_bytes(), gitfile.stat().st_mode & 0o7777),
                gitfile_before,
            )
            self.assertEqual(
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=worktree,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                head_before,
            )
            self.assertTrue(unsafe.is_file())
            self.assertFalse((worktree / "research.yaml").exists())

            identity_repo = root / "identity"
            identity_repo.mkdir()

            def identity_git(
                *args: str, check: bool = True
            ) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", *args],
                    cwd=identity_repo,
                    check=check,
                    capture_output=True,
                    text=True,
                )

            identity_git("init", "-b", "main")
            baseline = identity_repo / "baseline.txt"
            baseline.write_text("baseline\n", encoding="utf-8")
            identity_git("add", "baseline.txt")
            identity_git(
                "-c",
                "user.name=One-shot",
                "-c",
                "user.email=one-shot@example.invalid",
                "commit",
                "-m",
                "baseline",
            )
            (identity_repo / "checkpoints").mkdir()
            identity_head = identity_git("rev-parse", "HEAD").stdout.strip()
            output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.diagnostics.diagnose",
                    side_effect=OSError("late diagnose failure"),
                ),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "setup",
                        str(identity_repo),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2, output.getvalue())
            self.assertEqual(
                identity_git("rev-parse", "HEAD").stdout.strip(), identity_head
            )
            self.assertNotEqual(
                identity_git(
                    "config", "--local", "--get", "user.name", check=False
                ).returncode,
                0,
            )
            self.assertNotEqual(
                identity_git(
                    "config", "--local", "--get", "user.email", check=False
                ).returncode,
                0,
            )
            self.assertEqual(list((identity_repo / "checkpoints").iterdir()), [])
            self.assertEqual(identity_git("status", "--porcelain").stdout, "")

            environment_names = {
                "ZHIPU_API_KEY",
                "GOOGLE_API_KEY",
                "AI_SCIENTIST_ACTIVE_PROVIDER",
                "AI_SCIENTIST_DEFAULT_MODEL",
                "ZHIPU_DEFAULT_MODEL",
            }
            environment_before = {
                name: os.environ.get(name) for name in environment_names
            }
            managed_before = dict(provider_config_module._MANAGED_ENV_VALUES)
            provider_workspace = root / "provider"
            with (
                mock.patch.object(sys.stdin, "isatty", return_value=True),
                mock.patch("getpass.getpass", return_value="sk-" + "C" * 32),
                mock.patch(
                    "xscientist.diagnostics.diagnose",
                    side_effect=OSError("late diagnose failure"),
                ),
                contextlib.redirect_stderr(io.StringIO()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = cli_main(
                    [
                        "setup",
                        str(provider_workspace),
                        "--task",
                        "research",
                        "--provider",
                        "zhipu",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertFalse(provider_workspace.exists())
            self.assertEqual(
                {name: os.environ.get(name) for name in environment_names},
                environment_before,
            )
            self.assertEqual(
                provider_config_module._MANAGED_ENV_VALUES,
                managed_before,
            )

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
                        "--provider",
                        "zhipu",
                        "--skip-credentials",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 1)

    def test_json_parser_errors_redact_raw_choices_without_stderr(self) -> None:
        secret = "sk-" + "P" * 32
        cases = (
            (["init", "study", "--provider", secret, "--json"], "init"),
            (["setup", "study", "--task", secret, "--json"], "setup"),
            (
                [
                    "start",
                    "study",
                    "--question",
                    "q",
                    "--allow-synthetic-data",
                    "--provider",
                    secret,
                    "--json",
                ],
                "start",
            ),
        )
        for argv, command in cases:
            with self.subTest(command=command):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    code = cli_main(argv)
                self.assertEqual(code, 2)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["schema"], f"xscientist.{command}.v1")
                self.assertFalse(payload["ok"])
                self.assertNotIn(secret, stdout.getvalue())
                self.assertEqual(stderr.getvalue(), "")

    def test_secret_model_is_rejected_before_init_or_no_vcs_setup_writes(self) -> None:
        secret = "sk-" + "M" * 32
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for command, workspace in (
                (
                    [
                        "init",
                        str(root / "init"),
                        "--provider",
                        "ollama",
                        "--model",
                        f"ollama/{secret}",
                        "--json",
                    ],
                    root / "init",
                ),
                (
                    [
                        "setup",
                        str(root / "service"),
                        "--task",
                        "service",
                        "--no-research-vcs",
                        "--model",
                        secret,
                        "--skip-credentials",
                        "--json",
                    ],
                    root / "service",
                ),
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = cli_main(command)
                self.assertEqual(code, 2, output.getvalue())
                self.assertNotIn(secret, output.getvalue())
                self.assertFalse(workspace.exists())

    def test_setup_rejects_special_managed_leaf_and_unowned_git_directory(
        self,
    ) -> None:
        import os

        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fifo_workspace = root / "fifo"
            fifo_workspace.mkdir()
            os.mkfifo(fifo_workspace / "README.md")
            with self.assertRaisesRegex(WorkspaceInitError, "non-regular"):
                create_workspace(fifo_workspace, force=True)
            self.assertTrue((fifo_workspace / "README.md").exists())

            fake_git_workspace = root / "fake-git"
            fake_git = fake_git_workspace / ".git"
            fake_git.mkdir(parents=True)
            sentinel = fake_git / "IRREPLACEABLE"
            sentinel.write_text("keep\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(fake_git_workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("not an exact-root Git worktree", output.getvalue())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(list(fake_git_workspace.iterdir()), [fake_git])

    def test_setup_detects_intent_to_add_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            workspace.mkdir()

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", *args],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout

            git("init", "-b", "main")
            pending = workspace / "user.txt"
            pending.write_text("pending\n", encoding="utf-8")
            git("add", "-N", "user.txt")
            before = git("status", "--porcelain=v2")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("staged work", output.getvalue())
            self.assertEqual(git("status", "--porcelain=v2"), before)
            self.assertEqual(set(workspace.iterdir()), {workspace / ".git", pending})

    def test_setup_runtime_failure_preserves_concurrent_scientific_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"

            def diagnose_then_commit(target: Path, **_kwargs: object) -> object:
                result = Path(target) / "hypotheses" / "concurrent-user-result.json"
                result.parent.mkdir(exist_ok=True)
                result.write_text('{"owner": "user"}\n', encoding="utf-8")
                subprocess.run(
                    ["git", "add", result.relative_to(target).as_posix()],
                    cwd=target,
                    check=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-c",
                        "user.name=Concurrent User",
                        "-c",
                        "user.email=concurrent@example.invalid",
                        "commit",
                        "-m",
                        "concurrent result",
                    ],
                    cwd=target,
                    check=True,
                    capture_output=True,
                )
                raise RuntimeError("unexpected diagnose failure")

            output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.diagnostics.diagnose",
                    side_effect=diagnose_then_commit,
                ),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertTrue((workspace / ".git").is_dir())
            self.assertEqual(
                (workspace / "hypotheses" / "concurrent-user-result.json").read_text(),
                '{"owner": "user"}\n',
            )
            subject = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(subject, "concurrent result")

    def test_setup_linked_worktree_failure_preserves_common_git_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            primary = root / "primary"
            linked = root / "linked"
            primary.mkdir()
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=primary,
                check=True,
                capture_output=True,
            )
            (primary / "seed.txt").write_text("seed\n", encoding="utf-8")
            subprocess.run(["git", "add", "seed.txt"], cwd=primary, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Initial User",
                    "-c",
                    "user.email=initial@example.invalid",
                    "commit",
                    "-m",
                    "seed",
                ],
                cwd=primary,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "worktree", "add", "-b", "linked", str(linked)],
                cwd=primary,
                check=True,
                capture_output=True,
            )
            original_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=linked,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            def change_common_config(target: Path, **_kwargs: object) -> object:
                subprocess.run(
                    ["git", "config", "--local", "user.name", "Concurrent User"],
                    cwd=target,
                    check=True,
                )
                raise RuntimeError("unexpected diagnose failure")

            output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.diagnostics.diagnose",
                    side_effect=change_common_config,
                ),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "setup",
                        str(linked),
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2, output.getvalue())
            self.assertTrue((linked / ".git").is_file())
            self.assertEqual(
                subprocess.run(
                    ["git", "config", "--local", "--get", "user.name"],
                    cwd=linked,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "Concurrent User",
            )
            self.assertEqual(
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=linked,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                original_head,
            )

    def test_setup_and_start_rollback_preserve_concurrent_process_environment(
        self,
    ) -> None:
        import os

        import xscientist.provider_config as provider_config_module

        key = "GOOGLE_API_KEY"
        missing = object()
        original_environment = os.environ.get(key, missing)
        original_managed = provider_config_module._MANAGED_ENV_VALUES.get(key, missing)
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                commands = (
                    (
                        "setup",
                        [
                            "setup",
                            str(root / "setup"),
                            "--task",
                            "protocol",
                            "--skip-credentials",
                            "--json",
                        ],
                        root / "setup",
                    ),
                    (
                        "start",
                        [
                            "start",
                            str(root / "start"),
                            "--question",
                            "Does X change Y?",
                            "--provider",
                            "zhipu",
                            "--prepare-only",
                            "--skip-credentials",
                            "--json",
                        ],
                        root / "start",
                    ),
                )
                for command_name, argv, workspace in commands:
                    with self.subTest(command=command_name):
                        os.environ.pop(key, None)
                        provider_config_module._MANAGED_ENV_VALUES.pop(key, None)
                        concurrent_value = f"concurrent-{command_name}-value"
                        concurrent_owner = f"concurrent-{command_name}-owner"

                        def fail_with_concurrent_environment(
                            *_args: object, **_kwargs: object
                        ) -> object:
                            os.environ[key] = concurrent_value
                            provider_config_module._MANAGED_ENV_VALUES[key] = (
                                concurrent_owner,
                                concurrent_value,
                            )
                            raise RuntimeError("unexpected diagnose failure")

                        output = io.StringIO()
                        with (
                            mock.patch(
                                "ai_scientist.utils.auth_session.validate_session",
                                return_value=(True, "ok", {"username": "tester"}),
                            ),
                            mock.patch(
                                "xscientist.diagnostics.diagnose",
                                side_effect=fail_with_concurrent_environment,
                            ),
                            contextlib.redirect_stdout(output),
                        ):
                            code = cli_main(argv)

                        self.assertEqual(code, 2, output.getvalue())
                        self.assertEqual(os.environ.get(key), concurrent_value)
                        self.assertEqual(
                            provider_config_module._MANAGED_ENV_VALUES.get(key),
                            (concurrent_owner, concurrent_value),
                        )
                        self.assertNotIn(concurrent_value, output.getvalue())
                        self.assertFalse(workspace.exists())
        finally:
            if original_environment is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(original_environment)
            if original_managed is missing:
                provider_config_module._MANAGED_ENV_VALUES.pop(key, None)
            else:
                provider_config_module._MANAGED_ENV_VALUES[key] = original_managed

    def test_existing_research_force_preserves_custom_runtime_configuration(
        self,
    ) -> None:
        from xscientist.provider_config import save_provider
        from xscientist.research_vcs import ResearchRepository

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "setup",
                            str(workspace),
                            "--task",
                            "protocol",
                            "--skip-credentials",
                            "--json",
                        ]
                    ),
                    0,
                )
            save_provider(
                workspace,
                provider="ollama",
                model="ollama/qwen2.5:7b",
                env_file=".env.local",
                activate=True,
            )
            budget_path = workspace / "bfts_config.yaml"
            budget = yaml.safe_load(budget_path.read_text(encoding="utf-8"))
            budget["llm_budget"]["prices_per_million"]["custom/model"] = {
                "input": 1.25,
                "output": 3.5,
            }
            budget["agent"]["user_extension"] = {"keep": True}
            header = budget_path.read_text(encoding="utf-8").splitlines()[0]
            budget_path.write_text(
                header
                + "\n"
                + yaml.safe_dump(budget, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            readme = workspace / "README.md"
            dockerfile = workspace / "Dockerfile.executor"
            readme.write_text("CUSTOM README\n", encoding="utf-8")
            dockerfile.write_text("FROM custom/image\n", encoding="utf-8")
            repository = ResearchRepository(workspace)
            custom_paths = [
                ".xscientist/providers.json",
                "bfts_config.yaml",
                "README.md",
                "Dockerfile.executor",
            ]
            repository.stage(custom_paths)
            checkpoint = repository.commit(
                stage="setup",
                subject="record custom runtime configuration",
                status="completed",
                staged_only=True,
            )
            self.assertTrue(checkpoint.committed)
            before = {path: (workspace / path).read_bytes() for path in custom_paths}

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--force",
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 0, output.getvalue())
            self.assertEqual(
                {path: (workspace / path).read_bytes() for path in custom_paths},
                before,
            )
            self.assertEqual(repository.status()["eligible_changes"], [])

    def test_untracked_research_yaml_does_not_grant_refresh_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "setup",
                            str(source),
                            "--task",
                            "protocol",
                            "--skip-credentials",
                            "--json",
                        ]
                    ),
                    0,
                )
            spoof = root / "spoof"
            spoof.mkdir()
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=spoof,
                check=True,
                capture_output=True,
            )
            (spoof / "research.yaml").write_bytes(
                (source / "research.yaml").read_bytes()
            )
            readme = spoof / "README.md"
            readme.write_text("IRREPLACEABLE\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "setup",
                        str(spoof),
                        "--force",
                        "--task",
                        "protocol",
                        "--skip-credentials",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("not owned by the exact-root Git HEAD", output.getvalue())
            self.assertEqual(readme.read_text(encoding="utf-8"), "IRREPLACEABLE\n")

    def test_setup_json_is_noninteractive_and_emits_one_stdout_document(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(sys.stdin, "isatty", return_value=True),
                mock.patch(
                    "builtins.input",
                    side_effect=AssertionError("JSON mode must not prompt"),
                ) as prompted,
                mock.patch(
                    "getpass.getpass",
                    side_effect=AssertionError("JSON mode must not request secrets"),
                ) as secret_prompted,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = cli_main(
                    [
                        "setup",
                        str(workspace),
                        "--task",
                        "research",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertFalse(workspace.exists())
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(stdout.getvalue().count("\n"), 1)
            self.assertFalse(payload["ok"])
            self.assertIn("--provider is required", payload["error"])
            prompted.assert_not_called()
            secret_prompted.assert_not_called()

    def test_start_json_input_error_is_one_machine_readable_document(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X affect Y?",
                        "--allow-synthetic-data",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["phase"], "input")
            self.assertIn("--provider is required", payload["error"])

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
