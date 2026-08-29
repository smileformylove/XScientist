from __future__ import annotations

import contextlib
import io
import os
import tempfile
import threading
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import xscientist.cli as cli_module
import xscientist.provider_config as provider_config
from xscientist.cli import main as cli_main
from xscientist.onboarding import create_workspace


@contextlib.contextmanager
def _isolated_managed_environment(*names: str) -> Iterator[None]:
    missing = object()
    with provider_config._ENVIRONMENT_LOCK:
        original_process = {name: os.environ.get(name, missing) for name in names}
        original_managed = {
            name: provider_config._MANAGED_ENV_VALUES.get(name, missing)
            for name in names
        }
        original_generations = {
            name: provider_config._ENVIRONMENT_GENERATIONS.get(name, missing)
            for name in names
        }
        for name in names:
            os.environ.pop(name, None)
            provider_config._MANAGED_ENV_VALUES.pop(name, None)
            provider_config._ENVIRONMENT_GENERATIONS.pop(name, None)
    try:
        yield
    finally:
        with provider_config._ENVIRONMENT_LOCK:
            for name in names:
                process_value = original_process[name]
                if process_value is missing:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = str(process_value)
                managed_value = original_managed[name]
                if managed_value is missing:
                    provider_config._MANAGED_ENV_VALUES.pop(name, None)
                else:
                    provider_config._MANAGED_ENV_VALUES[name] = managed_value
                generation = original_generations[name]
                if generation is missing:
                    provider_config._ENVIRONMENT_GENERATIONS.pop(name, None)
                else:
                    provider_config._ENVIRONMENT_GENERATIONS[name] = generation


class ManagedEnvironmentTransactionTests(unittest.TestCase):
    def test_transaction_rolls_back_multiple_owned_mutations(self) -> None:
        key = "GOOGLE_API_KEY"
        with _isolated_managed_environment(key):
            os.environ[key] = "baseline-value"
            provider_config._MANAGED_ENV_VALUES[key] = (
                "baseline-owner",
                "baseline-value",
            )

            transaction = provider_config.begin_managed_environment_transaction()
            provider_config.mark_managed_environment("first", key, "first-value")
            provider_config.mark_managed_environment("second", key, "second-value")

            self.assertEqual(transaction.rollback(), ())
            self.assertEqual(os.environ[key], "baseline-value")
            self.assertEqual(
                provider_config._MANAGED_ENV_VALUES[key],
                ("baseline-owner", "baseline-value"),
            )

    def test_transaction_preserves_same_value_written_by_another_thread(self) -> None:
        key = "GOOGLE_API_KEY"
        with _isolated_managed_environment(key), tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "first"
            second = root / "second"
            shared_value = "same-visible-value"
            transaction = provider_config.begin_managed_environment_transaction()
            provider_config.mark_managed_environment(first, key, shared_value)

            writer = threading.Thread(
                target=provider_config.mark_managed_environment,
                args=(second, key, shared_value),
            )
            writer.start()
            writer.join(timeout=2)
            self.assertFalse(writer.is_alive())

            self.assertEqual(transaction.rollback(), (key,))
            self.assertEqual(os.environ[key], shared_value)
            self.assertEqual(
                provider_config._MANAGED_ENV_VALUES[key],
                (str(second.resolve()), shared_value),
            )

    def test_setup_and_start_do_not_absorb_a_write_before_post_state_capture(
        self,
    ) -> None:
        key = "GOOGLE_API_KEY"
        with _isolated_managed_environment(key), tempfile.TemporaryDirectory() as td:
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
                    provider_config._MANAGED_ENV_VALUES.pop(key, None)
                    provider_config._ENVIRONMENT_GENERATIONS.pop(key, None)
                    calls = 0
                    original_record = cli_module._record_setup_post_state
                    concurrent_workspace = root / f"concurrent-{command_name}"
                    concurrent_value = f"concurrent-{command_name}-value"

                    def record_after_concurrent_write(
                        target: Path,
                        snapshot: dict[str, object],
                    ) -> None:
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            writer = threading.Thread(
                                target=provider_config.mark_managed_environment,
                                args=(concurrent_workspace, key, concurrent_value),
                            )
                            writer.start()
                            writer.join(timeout=2)
                            self.assertFalse(writer.is_alive())
                        original_record(target, snapshot)

                    output = io.StringIO()
                    with (
                        mock.patch(
                            "ai_scientist.utils.auth_session.validate_session",
                            return_value=(True, "ok", {"username": "tester"}),
                        ),
                        mock.patch(
                            "xscientist.cli._record_setup_post_state",
                            side_effect=record_after_concurrent_write,
                        ),
                        mock.patch(
                            "xscientist.diagnostics.diagnose",
                            side_effect=RuntimeError("forced failure"),
                        ),
                        contextlib.redirect_stdout(output),
                    ):
                        code = cli_main(argv)

                    self.assertEqual(code, 2, output.getvalue())
                    self.assertEqual(os.environ.get(key), concurrent_value)
                    self.assertEqual(
                        provider_config._MANAGED_ENV_VALUES.get(key),
                        (str(concurrent_workspace.resolve()), concurrent_value),
                    )
                    self.assertFalse(workspace.exists())

    def test_provider_add_failure_rolls_back_process_environment(self) -> None:
        keys = ("OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_BASE_URL")
        with (
            _isolated_managed_environment(*keys),
            tempfile.TemporaryDirectory() as td,
        ):
            workspace = Path(td) / "study"
            create_workspace(workspace, provider="zhipu")
            provider_path = workspace / ".xscientist" / "providers.json"
            bfts_path = workspace / "bfts_config.yaml"
            provider_before = provider_path.read_bytes()
            bfts_before = bfts_path.read_bytes()
            provider_mode_before = provider_path.stat().st_mode & 0o7777
            bfts_mode_before = bfts_path.stat().st_mode & 0o7777
            output = io.StringIO()
            with (
                mock.patch("xscientist.cli.sys.stdin.isatty", return_value=True),
                mock.patch("getpass.getpass", return_value="provider-test-value"),
                mock.patch(
                    "xscientist.provider_config.update_bfts_models",
                    side_effect=OSError("forced BFTS failure"),
                ),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = cli_main(
                    [
                        "provider",
                        "add",
                        "custom",
                        "--workspace",
                        str(workspace),
                        "--model",
                        "glm-5.3",
                        "--base-url",
                        "https://gateway.example/v1",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2, output.getvalue())
            for key in keys:
                self.assertNotIn(key, os.environ)
                self.assertNotIn(key, provider_config._MANAGED_ENV_VALUES)
            self.assertFalse((workspace / ".env").exists())
            self.assertEqual(provider_path.read_bytes(), provider_before)
            self.assertEqual(bfts_path.read_bytes(), bfts_before)
            self.assertEqual(
                provider_path.stat().st_mode & 0o7777, provider_mode_before
            )
            self.assertEqual(bfts_path.stat().st_mode & 0o7777, bfts_mode_before)

    def test_provider_add_rollback_preserves_a_concurrent_file_change(self) -> None:
        keys = ("OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_BASE_URL")
        with (
            _isolated_managed_environment(*keys),
            tempfile.TemporaryDirectory() as td,
        ):
            workspace = Path(td) / "study"
            create_workspace(workspace, provider="zhipu")
            provider_path = workspace / ".xscientist" / "providers.json"
            bfts_path = workspace / "bfts_config.yaml"
            bfts_before = bfts_path.read_bytes()
            concurrent_content = b"concurrent provider metadata\n"

            def change_provider_file_then_fail(
                *_args: object, **_kwargs: object
            ) -> bool:
                provider_path.write_bytes(concurrent_content)
                raise OSError("forced BFTS failure")

            with (
                mock.patch("xscientist.cli.sys.stdin.isatty", return_value=True),
                mock.patch("getpass.getpass", return_value="provider-test-value"),
                mock.patch(
                    "xscientist.provider_config.update_bfts_models",
                    side_effect=change_provider_file_then_fail,
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = cli_main(
                    [
                        "provider",
                        "add",
                        "custom",
                        "--workspace",
                        str(workspace),
                        "--model",
                        "glm-5.3",
                        "--base-url",
                        "https://gateway.example/v1",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(provider_path.read_bytes(), concurrent_content)
            self.assertEqual(bfts_path.read_bytes(), bfts_before)
            self.assertFalse((workspace / ".env").exists())
            for key in keys:
                self.assertNotIn(key, os.environ)
                self.assertNotIn(key, provider_config._MANAGED_ENV_VALUES)

    def test_post_replace_inspection_failure_keeps_a_rollback_journal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td).resolve() / "providers.json"
            baseline = b'{"state": "baseline"}\n'
            replacement = b'{"state": "replacement"}\n'
            target.write_bytes(baseline)
            original_capture = provider_config._capture_provider_file_state

            def fail_only_after_replacement(path: Path) -> object:
                state = original_capture(path)
                if Path(path) == target and state.content == replacement:
                    raise OSError("forced post-replace lstat/read failure")
                return state

            transaction = provider_config.begin_provider_file_transaction()
            try:
                with (
                    mock.patch(
                        "xscientist.provider_config._capture_provider_file_state",
                        side_effect=fail_only_after_replacement,
                    ),
                    self.assertRaisesRegex(OSError, "post-replace"),
                ):
                    provider_config._tracked_provider_file_write(
                        target,
                        replacement.decode("utf-8"),
                    )

                self.assertEqual(target.read_bytes(), replacement)
                self.assertEqual(transaction.rollback(), ())
            finally:
                transaction.rollback()

            self.assertEqual(target.read_bytes(), baseline)

    def test_pre_replace_recheck_preserves_an_external_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td).resolve() / "providers.json"
            target.write_text("baseline\n", encoding="utf-8")
            external = "external concurrent value\n"
            original_capture = provider_config._capture_provider_file_state
            calls = 0

            def change_before_rename(path: Path) -> object:
                nonlocal calls
                calls += 1
                if calls == 2:
                    target.write_text(external, encoding="utf-8")
                return original_capture(path)

            with (
                mock.patch(
                    "xscientist.provider_config._capture_provider_file_state",
                    side_effect=change_before_rename,
                ),
                self.assertRaisesRegex(
                    provider_config.ProviderConfigError,
                    "before atomic replacement",
                ),
            ):
                provider_config._tracked_provider_file_write(target, "ours\n")

            self.assertEqual(target.read_text(encoding="utf-8"), external)

    def test_provider_test_uses_a_pure_workspace_mapping(self) -> None:
        keys = ("OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_BASE_URL")
        with (
            _isolated_managed_environment(*keys),
            tempfile.TemporaryDirectory() as td,
        ):
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai_compat",
                model="openai_compat/glm-5.3",
            )
            env_path = workspace / ".env"
            env_path.write_text(
                "OPENAI_COMPAT_API_KEY=workspace-test-value\n"
                "OPENAI_COMPAT_BASE_URL=https://gateway.example/v1\n",
                encoding="utf-8",
            )
            env_path.chmod(0o600)
            observed_environment: dict[str, str] = {}

            def probe(_model: str, *, env: dict[str, str], **_kwargs: object) -> dict:
                observed_environment.update(env)
                return {
                    "ok": True,
                    "supported": True,
                    "identity_status": "exact",
                    "client_model": "glm-5.3",
                    "reported_model": "glm-5.3",
                    "tool_call_valid": True,
                    "capability": "forced_function_call",
                }

            with (
                mock.patch(
                    "ai_scientist.utils.provider_registry."
                    "probe_openai_compatible_tool_call",
                    side_effect=probe,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = cli_main(
                    [
                        "provider",
                        "test",
                        "custom",
                        "--workspace",
                        str(workspace),
                        "--json",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(
                observed_environment["OPENAI_COMPAT_BASE_URL"],
                "https://gateway.example/v1",
            )
            for key in keys:
                self.assertNotIn(key, os.environ)
                self.assertNotIn(key, provider_config._MANAGED_ENV_VALUES)

    def test_pure_workspace_mapping_rejects_broad_credential_permissions(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX permission semantics are unavailable")
        key = "OPENAI_COMPAT_API_KEY"
        with _isolated_managed_environment(key), tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            create_workspace(
                workspace,
                provider="openai_compat",
                model="openai_compat/glm-5.3",
            )
            env_path = workspace / ".env"
            env_path.write_text(
                "OPENAI_COMPAT_API_KEY=workspace-test-value\n",
                encoding="utf-8",
            )
            env_path.chmod(0o644)

            with self.assertRaisesRegex(
                provider_config.ProviderConfigError,
                "broad permissions",
            ):
                provider_config.workspace_environment(workspace)

            self.assertNotIn(key, os.environ)
            self.assertNotIn(key, provider_config._MANAGED_ENV_VALUES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
