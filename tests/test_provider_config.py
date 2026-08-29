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
    update_bfts_models,
    validate_custom_base_url,
    workspace_environment,
)


class ProviderConfigTests(unittest.TestCase):
    def test_bfts_model_update_preserves_optional_agent_section_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with_optional = Path(td) / "with-optional.yaml"
            with_optional.write_text(
                yaml.safe_dump(
                    {
                        "report": {"model": "old/report"},
                        "agent": {
                            "code": {"model": "old/code"},
                            "feedback": {"model": "old/feedback"},
                            "vlm_feedback": {"model": "old/vlm"},
                            "summary": {"model": "old/summary"},
                            "select_node": {"model": "old/planner"},
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            without_optional = Path(td) / "without-optional.yaml"
            without_optional.write_text(
                yaml.safe_dump(
                    {
                        "report": {"model": "old/report"},
                        "agent": {"code": {"model": "old/code"}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            model = "openai_compat/glm-5.3"
            self.assertTrue(update_bfts_models(with_optional, model))
            self.assertTrue(update_bfts_models(without_optional, model))

            updated = yaml.safe_load(with_optional.read_text(encoding="utf-8"))
            self.assertEqual(updated["report"]["model"], model)
            for role in (
                "code",
                "feedback",
                "vlm_feedback",
                "summary",
                "select_node",
            ):
                self.assertEqual(updated["agent"][role]["model"], model)

            minimal = yaml.safe_load(without_optional.read_text(encoding="utf-8"))
            self.assertNotIn("summary", minimal["agent"])
            self.assertNotIn("select_node", minimal["agent"])

    def test_loading_two_workspaces_does_not_reuse_managed_provider_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first"
            second = Path(td) / "second"
            create_workspace(
                first,
                provider="openai_compat",
                model="openai_compat/first-model",
            )
            create_workspace(
                second,
                provider="openai_compat",
                model="openai_compat/second-model",
            )
            for workspace, suffix in ((first, "first"), (second, "second")):
                env_file = workspace / ".env"
                env_file.write_text(
                    "OPENAI_COMPAT_API_KEY={0}-key\n"
                    "OPENAI_COMPAT_BASE_URL=https://{0}.example/v1\n".format(suffix),
                    encoding="utf-8",
                )
                env_file.chmod(0o600)

            with mock.patch.dict(os.environ, {}, clear=True):
                load_workspace_environment(first)
                self.assertEqual(
                    os.environ["OPENAI_COMPAT_BASE_URL"],
                    "https://first.example/v1",
                )
                load_workspace_environment(second)
                self.assertEqual(
                    os.environ["OPENAI_COMPAT_BASE_URL"],
                    "https://second.example/v1",
                )
                self.assertEqual(
                    workspace_environment(second)["OPENAI_COMPAT_API_KEY"],
                    "second-key",
                )

    def test_cli_provider_choices_match_registry(self) -> None:
        self.assertEqual(tuple(_PROVIDER_CHOICES[:-1]), PROVIDER_NAMES)
        self.assertEqual(_PROVIDER_CHOICES[-1], "custom")

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

    def test_provider_list_discovers_local_models_before_workspace_setup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stdout = io.StringIO()
            human_stdout = io.StringIO()
            local_probe = {
                "checked": True,
                "ok": False,
                "service_reachable": True,
                "model_available": False,
                "models": ["ollama/qwen2.5:7b"],
                "error": "No Ollama model is selected",
            }
            with (
                mock.patch(
                    "xscientist.cli.discover_workspace_root",
                    create=True,
                ),
                mock.patch(
                    "xscientist.provider_config.discover_workspace_root",
                    return_value=None,
                ),
                mock.patch("xscientist.cli.Path.cwd", return_value=Path(td)),
                mock.patch(
                    "xscientist.provider_config.probe_provider_model",
                    side_effect=lambda provider, _model, **_kwargs: (
                        local_probe
                        if provider == "ollama"
                        else {
                            "checked": False,
                            "ok": True,
                            "service_reachable": None,
                            "model_available": None,
                            "models": [],
                            "error": None,
                        }
                    ),
                ),
                mock.patch(
                    "xscientist.provider_config.missing_provider_modules",
                    side_effect=lambda provider: (
                        ["openai"] if provider == "ollama" else []
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(cli_main(["provider", "list", "--json"]), 0)

            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["workspace_initialized"])
            self.assertTrue(payload["discovery_only"])
            ollama = next(
                row for row in payload["providers"] if row["provider"] == "ollama"
            )
            self.assertTrue(ollama["local_detected"])
            self.assertEqual(ollama["suggested_model"], "ollama/qwen2.5:7b")
            with (
                mock.patch(
                    "xscientist.provider_config.probe_provider_model",
                    side_effect=lambda provider, _model, **_kwargs: (
                        local_probe
                        if provider == "ollama"
                        else {
                            "checked": False,
                            "ok": True,
                            "service_reachable": None,
                            "model_available": None,
                            "models": [],
                            "error": None,
                        }
                    ),
                ),
                mock.patch(
                    "xscientist.provider_config.missing_provider_modules",
                    side_effect=lambda provider: (
                        ["openai"] if provider == "ollama" else []
                    ),
                ),
                mock.patch(
                    "xscientist.provider_config.discover_workspace_root",
                    return_value=None,
                ),
                mock.patch("xscientist.cli.Path.cwd", return_value=Path(td)),
                contextlib.redirect_stdout(human_stdout),
            ):
                self.assertEqual(cli_main(["provider", "list"]), 0)
            self.assertIn(
                'Install: python -m pip install "xscientist[research,openai-compatible]"',
                human_stdout.getvalue(),
            )

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

    def test_provider_check_live_is_explicit_and_reports_model_identity(self) -> None:
        """A configuration check must only make a request when --live is given."""

        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="gpt-4o-mini",
            )
            output = io.StringIO()
            live_result = {
                "ok": True,
                "supported": True,
                "transport_ok": True,
                "provider": "openai",
                "requested_model": "gpt-4o-mini",
                "client_model": "gpt-4o-mini",
                "reported_model": "gpt-4o-mini",
                "identity_status": "exact",
                "model_identity_verified": True,
                "exact_model_match": True,
                "response_content_recorded": False,
            }
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
                mock.patch(
                    "ai_scientist.utils.provider_registry.probe_live_model",
                    return_value=live_result,
                ) as probe,
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "check",
                        "--workspace",
                        str(workspace),
                        "--live",
                        "--timeout",
                        "4",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["checks"]["live_probe_requested"])
            self.assertTrue(payload["checks"]["live_request_attempted"])
            self.assertTrue(payload["checks"]["live_api_verified"])
            self.assertEqual(payload["checks"]["verification_scope"], "live_request")
            self.assertEqual(payload["live_probe"]["identity_status"], "exact")
            probe.assert_called_once()
            self.assertEqual(probe.call_args.kwargs["timeout"], 4.0)
            self.assertNotIn("process-only-secret", output.getvalue())

    def test_provider_check_without_live_never_calls_the_provider_probe(self) -> None:
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
                mock.patch(
                    "ai_scientist.utils.provider_registry.probe_live_model"
                ) as probe,
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "check",
                        "--workspace",
                        str(workspace),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["checks"]["live_probe_requested"])
            self.assertFalse(payload["checks"]["live_api_verified"])
            self.assertIsNone(payload["live_probe"])
            probe.assert_not_called()

    def test_provider_check_live_respects_unknown_cost_guard_before_network(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai",
                model="openai/unpriced-research-model",
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
                mock.patch(
                    "ai_scientist.utils.provider_registry.probe_live_model"
                ) as probe,
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
                        "--live",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertIn("unknown_model_price", payload["error_codes"])
            self.assertIn("live_probe_blocked_by_unknown_cost", payload["error_codes"])
            self.assertFalse(payload["checks"]["live_api_verified"])
            self.assertFalse(payload["checks"]["live_request_attempted"])
            self.assertEqual(
                payload["checks"]["verification_scope"], "live_request_blocked"
            )
            probe.assert_not_called()

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

    def test_workspace_model_replaces_stale_process_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="custom",
                model="gpt-5.6-luna",
            )
            with mock.patch.dict(
                os.environ,
                {"AI_SCIENTIST_DEFAULT_MODEL": "gpt-5.4-mini"},
                clear=True,
            ):
                state = load_workspace_environment(workspace)

                self.assertEqual(state["model"], "openai_compat/gpt-5.6-luna")
                self.assertEqual(
                    os.environ["AI_SCIENTIST_DEFAULT_MODEL"],
                    "openai_compat/gpt-5.6-luna",
                )

    def test_custom_provider_add_accepts_base_url_without_serializing_secret(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace, provider="zhipu")
            output = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("getpass.getpass", return_value="custom-secret"),
                mock.patch("xscientist.cli.sys.stdin.isatty", return_value=True),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "add",
                        "custom",
                        "--workspace",
                        str(workspace),
                        "--model",
                        "gpt-5.6-luna",
                        "--base-url",
                        "https://gateway.example/v1",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            rendered = output.getvalue()
            self.assertNotIn("custom-secret", rendered)
            payload = json.loads(rendered)
            self.assertEqual(payload["provider"], "openai_compat")
            self.assertEqual(payload["model"], "openai_compat/gpt-5.6-luna")
            self.assertEqual(payload["settings_written"], ["OPENAI_COMPAT_BASE_URL"])
            self.assertEqual(payload["credentials_written"], ["OPENAI_COMPAT_API_KEY"])
            metadata = (workspace / ".xscientist" / "providers.json").read_text()
            self.assertNotIn("gateway.example", metadata)
            self.assertNotIn("custom-secret", metadata)
            env_text = (workspace / ".env").read_text()
            self.assertIn("OPENAI_COMPAT_BASE_URL=https://gateway.example/v1", env_text)
            self.assertIn("OPENAI_COMPAT_API_KEY=custom-secret", env_text)

    def test_custom_base_url_validation_requires_a_safe_absolute_endpoint(self) -> None:
        self.assertEqual(
            validate_custom_base_url("https://gateway.example/v1/"),
            "https://gateway.example/v1",
        )
        self.assertEqual(
            validate_custom_base_url("http://127.0.0.1:8080/v1"),
            "http://127.0.0.1:8080/v1",
        )
        invalid_urls = {
            "": "cannot be empty",
            "gateway.example/v1": "absolute HTTP",
            "http://gateway.example/v1": "require HTTPS",
            "https://user:password@gateway.example/v1": "embedded credentials",
            "https://gateway.example/v1?token=secret": "query",
            "https://gateway.example:bad/v1": "invalid port",
        }
        for value, message in invalid_urls.items():
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ProviderConfigError, message),
            ):
                validate_custom_base_url(value)

    def test_provider_test_uses_workspace_endpoint_without_printing_credentials(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace, provider="custom", model="gpt-5.6-luna")
            env_file = workspace / ".env"
            env_file.write_text(
                "OPENAI_COMPAT_API_KEY=probe-secret\n"
                "OPENAI_COMPAT_BASE_URL=https://gateway.example/v1\n",
                encoding="utf-8",
            )
            env_file.chmod(0o600)
            probe_result = {
                "ok": True,
                "provider": "openai_compat",
                "requested_model": "openai_compat/gpt-5.6-luna",
                "client_model": "gpt-5.6-luna",
                "reported_model": "gpt-5.6-luna",
                "model_identity_verified": True,
                "exact_model_match": True,
                "identity_status": "exact",
                "capability": "forced_function_call",
                "tool_call_valid": True,
                "usage_valid": True,
                "response_content_recorded": False,
                "request_content_recorded": False,
                "tool_arguments_recorded": False,
            }
            output = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch(
                    "ai_scientist.utils.provider_registry."
                    "probe_openai_compatible_tool_call",
                    return_value=probe_result,
                ) as probe,
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli_main(
                    [
                        "provider",
                        "test",
                        "custom",
                        "--workspace",
                        str(workspace),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertNotIn("probe-secret", output.getvalue())
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["capability"], "forced_function_call")
            self.assertTrue(payload["probe"]["tool_call_valid"])
            self.assertFalse(payload["response_content_recorded"])
            probe.assert_called_once()
            self.assertEqual(
                probe.call_args.kwargs["env"]["OPENAI_COMPAT_BASE_URL"],
                "https://gateway.example/v1",
            )
            self.assertEqual(
                probe.call_args.kwargs["env"]["OPENAI_COMPAT_API_KEY"],
                "probe-secret",
            )

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

    def test_openai_compatible_generic_alias_environment_is_rejected(self) -> None:
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

            self.assertEqual(exit_code, 2)
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

    def test_env_file_cannot_overlap_scientific_or_managed_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(workspace)
            metadata_path = workspace / ".xscientist" / "providers.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["env_file"] = "question.md"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            question = workspace / "question.md"
            question.write_text("IRREPLACEABLE SCIENTIFIC SOURCE\n", encoding="utf-8")

            with self.assertRaisesRegex(ProviderConfigError, r"private \.env"):
                load_provider_config(workspace)
            self.assertEqual(
                question.read_text(encoding="utf-8"),
                "IRREPLACEABLE SCIENTIFIC SOURCE\n",
            )

    def test_provider_add_rejects_secret_shaped_model_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="ollama",
                model="ollama/qwen2.5:7b",
            )
            metadata_path = workspace / ".xscientist" / "providers.json"
            budget_path = workspace / "bfts_config.yaml"
            before = (metadata_path.read_bytes(), budget_path.read_bytes())
            secret = "sk-" + "Q" * 32
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli_main(
                    [
                        "provider",
                        "add",
                        "ollama",
                        "--workspace",
                        str(workspace),
                        "--model",
                        f"ollama/{secret}",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertNotIn(secret, stdout.getvalue() + stderr.getvalue())
            self.assertEqual(
                (metadata_path.read_bytes(), budget_path.read_bytes()),
                before,
            )

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
