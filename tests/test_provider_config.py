from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from xscientist.cli import main as cli_main
from xscientist.cli import _PROVIDER_CHOICES
from xscientist.entrypoints import project_main
from xscientist.dependency_profiles import (
    installation_command,
    provider_client_modules,
    provider_extra,
)
from xscientist.onboarding import create_workspace
from xscientist.provider_config import (
    ProviderConfigError,
    PROVIDER_NAMES,
    discover_workspace_root,
    load_provider_config,
    load_workspace_environment,
    provider_statuses,
    resolve_env_file,
)


class ProviderConfigTests(unittest.TestCase):
    def test_cli_provider_choices_match_registry(self) -> None:
        self.assertEqual(tuple(_PROVIDER_CHOICES), PROVIDER_NAMES)

    def test_each_provider_has_a_path_free_install_profile(self) -> None:
        for provider in PROVIDER_NAMES:
            self.assertTrue(provider_extra(provider))
            self.assertTrue(provider_client_modules(provider))
            command = installation_command(provider)
            self.assertEqual(
                command,
                f'python -m pip install "xscientist[research,{provider_extra(provider)}]"',
            )
            self.assertNotIn(str(Path.home()), command)

    def test_provider_readiness_requires_credentials_and_client_packages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/research-model",
            )
            with mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "process-only-secret"},
                clear=True,
            ):
                missing_row = next(
                    row
                    for row in provider_statuses(
                        workspace, find_spec=lambda _name: None
                    )
                    if row["provider"] == "openai"
                )
                ready_row = next(
                    row
                    for row in provider_statuses(
                        workspace, find_spec=lambda _name: object()
                    )
                    if row["provider"] == "openai"
                )

            self.assertTrue(missing_row["credentials_available"])
            self.assertFalse(missing_row["client_available"])
            self.assertFalse(missing_row["ready"])
            self.assertEqual(missing_row["missing_client_modules"], ["openai"])
            self.assertEqual(
                missing_row["install_command"],
                'python -m pip install "xscientist[research,openai]"',
            )
            self.assertTrue(ready_row["client_available"])
            self.assertTrue(ready_row["ready"])

    def test_provider_check_discloses_presence_only_validation_and_price(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="gpt-4o-mini",
            )
            output = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "process-only-secret"},
                    clear=True,
                ),
                mock.patch(
                    "xscientist.provider_config.missing_provider_modules",
                    return_value=[],
                ),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "check",
                        "--workspace",
                        str(workspace),
                        "--max-cost-usd",
                        "1",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["schema"], "xscientist.provider-check.v1")
            self.assertTrue(payload["ok"])
            self.assertEqual(
                payload["checks"]["credential_validation"], "presence_only"
            )
            self.assertFalse(payload["checks"]["live_api_verified"])
            self.assertTrue(payload["checks"]["model_price_known"])
            self.assertNotIn("process-only-secret", output.getvalue())

    def test_provider_check_fails_closed_for_an_unknown_cost(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/research-model",
            )
            output = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "process-only-secret"},
                    clear=True,
                ),
                mock.patch(
                    "xscientist.provider_config.missing_provider_modules",
                    return_value=[],
                ),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "check",
                        "--workspace",
                        str(workspace),
                        "--max-cost-usd",
                        "1",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertIn("unknown_model_price", payload["error_codes"])
            self.assertFalse(payload["checks"]["model_price_known"])

    def test_ollama_check_verifies_service_model_and_zero_cost(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="ollama",
                model="qwen2.5:7b",
            )
            output = io.StringIO()
            response = io.StringIO(json.dumps({"models": [{"name": "qwen2.5:7b"}]}))
            with (
                mock.patch(
                    "xscientist.provider_config.missing_provider_modules",
                    return_value=[],
                ),
                mock.patch("urllib.request.OpenerDirector.open", return_value=response),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "check",
                        "--workspace",
                        str(workspace),
                        "--max-cost-usd",
                        "1",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["checks"]["live_api_verified"])
            self.assertFalse(payload["checks"]["credentials_required"])
            self.assertEqual(payload["checks"]["credential_validation"], "not_required")
            self.assertEqual(payload["price_per_million"]["input"], 0.0)

    def test_ollama_check_fails_when_the_local_service_is_unreachable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="ollama",
                model="qwen2.5:7b",
            )
            output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.provider_config.missing_provider_modules",
                    return_value=[],
                ),
                mock.patch(
                    "urllib.request.OpenerDirector.open",
                    side_effect=OSError("not reachable"),
                ),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    ["provider", "check", "--workspace", str(workspace), "--json"]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertIn("local_provider_unreachable", payload["error_codes"])
            self.assertFalse(payload["checks"]["live_api_verified"])

    def test_workspace_is_discovered_from_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            nested = workspace / "notes" / "drafts"
            create_workspace(workspace)
            nested.mkdir(parents=True)
            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("xscientist.provider_config.Path.cwd", return_value=nested),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(discover_workspace_root(), workspace.resolve())
                self.assertEqual(cli_main(["provider", "list", "--json"]), 0)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["workspace"], "study")
            self.assertEqual(len(payload["providers"]), len(PROVIDER_NAMES))
            self.assertNotIn(str(workspace), stdout.getvalue())

    def test_workflow_entrypoint_loads_parent_workspace_environment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            nested = workspace / "experiments" / "trial-a"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/research-model",
            )
            nested.mkdir(parents=True)
            env_file = workspace / ".env"
            env_file.write_text("OPENAI_API_KEY=parent-secret\n", encoding="utf-8")
            env_file.chmod(0o600)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("xscientist.entrypoints.Path.cwd", return_value=nested),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(project_main(["--help"]), 0)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "parent-secret")
                self.assertEqual(
                    os.environ["AI_SCIENTIST_DEFAULT_MODEL"],
                    "openai/research-model",
                )

    def test_provider_add_hides_secret_and_updates_active_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/research-model",
            )
            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("xscientist.cli.sys.stdin.isatty", return_value=True),
                mock.patch("getpass.getpass", return_value="test-secret-value"),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "add",
                        "openai",
                        "--workspace",
                        str(workspace),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertNotIn("test-secret-value", stdout.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["credentials_written"], ["OPENAI_API_KEY"])
            self.assertEqual(payload["workspace"], ".")
            self.assertEqual(payload["env_file"], ".env")
            self.assertNotIn(str(workspace), stdout.getvalue())
            env_file = workspace / ".env"
            self.assertEqual(
                env_file.read_text(encoding="utf-8"),
                "OPENAI_API_KEY=test-secret-value\n",
            )
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(env_file.stat().st_mode), 0o600)

            metadata_text = (workspace / ".xscientist" / "providers.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("test-secret-value", metadata_text)
            config = yaml.safe_load(
                (workspace / "bfts_config.yaml").read_text(encoding="utf-8")
            )
            for role in ("code", "feedback", "vlm_feedback"):
                self.assertEqual(
                    config["agent"][role]["model"], "openai/research-model"
                )
            self.assertEqual(config["report"]["model"], "openai/research-model")

    def test_workspace_loads_private_env_and_direct_help_uses_default_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/research-model",
            )
            env_file = workspace / ".env"
            env_file.write_text("OPENAI_API_KEY=test-secret\n", encoding="utf-8")
            env_file.chmod(0o600)
            with mock.patch.dict(
                os.environ,
                {"XSCIENTIST_WORKSPACE": str(workspace)},
                clear=True,
            ):
                state = load_workspace_environment()
                self.assertTrue(state["loaded"])
                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-secret")
                self.assertEqual(
                    os.environ["AI_SCIENTIST_DEFAULT_MODEL"],
                    "openai/research-model",
                )
                self.assertEqual(
                    os.environ["ZHIPU_DEFAULT_MODEL"], "openai/research-model"
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(project_main(["--help"]), 0)

    def test_process_environment_is_not_copied_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/research-model",
            )
            stdout = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "process-only-secret"},
                    clear=True,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "add",
                        "openai",
                        "--workspace",
                        str(workspace),
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse((workspace / ".env").exists())
            self.assertNotIn("process-only-secret", stdout.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())["credentials_written"], [])

    def test_openai_compatible_alias_environment_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai_compat",
                model="openai_compat/research-model",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "process-only-secret",
                    "OPENAI_BASE_URL": "https://example.invalid/v1",
                },
                clear=True,
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "add",
                        "openai_compat",
                        "--workspace",
                        str(workspace),
                        "--non-interactive",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse((workspace / ".env").exists())

    def test_provider_remove_keeps_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            env_file = workspace / ".env"
            env_file.write_text("ZHIPU_API_KEY=keep-this-secret\n", encoding="utf-8")
            env_file.chmod(0o600)
            with mock.patch.dict(os.environ, {}, clear=True):
                exit_code = cli_main(
                    ["provider", "remove", "zhipu", "--workspace", str(workspace)]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("keep-this-secret", env_file.read_text(encoding="utf-8"))
            self.assertIsNone(load_provider_config(workspace)["active_provider"])

    def test_add_without_activation_preserves_active_provider_and_bfts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            with mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "process-only-secret"},
                clear=True,
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "add",
                        "openai",
                        "--workspace",
                        str(workspace),
                        "--model",
                        "openai/research-model",
                        "--no-activate",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(exit_code, 0)
            metadata = load_provider_config(workspace)
            self.assertEqual(metadata["active_provider"], "zhipu")
            self.assertIn("openai", metadata["providers"])
            bfts = yaml.safe_load(
                (workspace / "bfts_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(bfts["agent"]["code"]["model"], "glm-4-flash")

    def test_broad_env_permissions_fail_closed_without_loading_secret(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX permission semantics are unavailable")
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            env_file = workspace / ".env"
            env_file.write_text("ZHIPU_API_KEY=must-not-load\n", encoding="utf-8")
            env_file.chmod(0o644)
            with mock.patch.dict(os.environ, {}, clear=True):
                state = load_workspace_environment(workspace)
                self.assertFalse(state["loaded"])
                self.assertIn("broad permissions", state["error"])
                self.assertNotIn("ZHIPU_API_KEY", os.environ)
                active = next(
                    row for row in provider_statuses(workspace) if row["active"]
                )
                self.assertTrue(active["configured"])
                self.assertFalse(active["ready"])
                self.assertIn("broad permissions", active["error"])

    def test_env_file_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            workspace.mkdir()
            with self.assertRaisesRegex(ProviderConfigError, "escape"):
                resolve_env_file(workspace, "../outside.env")
            with self.assertRaisesRegex(ProviderConfigError, "relative"):
                resolve_env_file(workspace, str(Path(td) / "outside.env"))

    def test_malformed_provider_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            metadata_path = workspace / ".xscientist" / "providers.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["providers"]["zhipu"]["model"] = "openai/wrong-provider"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ProviderConfigError, "invalid model"):
                load_provider_config(workspace)

            metadata["providers"]["zhipu"]["model"] = "glm-4-flash"
            metadata["providers"]["zhipu"]["api_key"] = "must-not-be-accepted"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ProviderConfigError, "unknown fields"):
                load_provider_config(workspace)

    def test_noninteractive_add_reports_missing_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/research-model",
            )
            stderr = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "add",
                        "openai",
                        "--workspace",
                        str(workspace),
                        "--non-interactive",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("missing OPENAI_API_KEY", stderr.getvalue())
            self.assertFalse((workspace / ".env").exists())

    def test_add_requires_initialized_workspace_before_prompting_or_writing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            workspace.mkdir()
            stderr = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("getpass.getpass") as secret_prompt,
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = cli_main(
                    ["provider", "add", "zhipu", "--workspace", str(workspace)]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("provider configuration not found", stderr.getvalue())
            secret_prompt.assert_not_called()
            self.assertFalse((workspace / ".env").exists())

    def test_explicit_provider_workspace_does_not_load_another_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first"
            second = Path(td) / "second"
            for workspace in (first, second):
                create_workspace(
                    workspace,
                    provider="openai",
                    model="openai/research-model",
                )
            first_env = first / ".env"
            first_env.write_text("OPENAI_API_KEY=first-secret\n", encoding="utf-8")
            first_env.chmod(0o600)
            stdout = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ,
                    {"XSCIENTIST_WORKSPACE": str(first)},
                    clear=True,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = cli_main(
                    ["provider", "list", "--workspace", str(second), "--json"]
                )
                self.assertNotIn("OPENAI_API_KEY", os.environ)

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            openai = next(
                row for row in payload["providers"] if row["provider"] == "openai"
            )
            self.assertFalse(openai["ready"])

    def test_provider_list_never_prints_the_absolute_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "private-workspace"
            create_workspace(workspace)
            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(
                    cli_main(
                        ["provider", "list", "--workspace", str(workspace), "--json"]
                    ),
                    0,
                )
            self.assertNotIn(str(workspace), stdout.getvalue())
            self.assertEqual(
                json.loads(stdout.getvalue())["workspace"], "private-workspace"
            )

    def test_human_provider_list_focuses_on_configured_routes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "focused-workspace"
            create_workspace(workspace)
            stdout = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch(
                    "xscientist.provider_config.discover_provider_models",
                    return_value=[],
                ),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(
                    cli_main(["provider", "list", "--workspace", str(workspace)]),
                    0,
                )

            rendered = stdout.getvalue()
            self.assertIn("Workspace: focused-workspace", rendered)
            self.assertIn("zhipu:", rendered)
            self.assertIn("Other providers hidden:", rendered)
            self.assertNotIn("openrouter:", rendered)


if __name__ == "__main__":
    unittest.main()
