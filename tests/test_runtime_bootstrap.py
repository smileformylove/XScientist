from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.utils.runtime_bootstrap import (
    PROJECT_ROOT_ENV_VAR,
    RuntimeContext,
    format_project_relative_path,
    initialize_runtime,
    require_env_var,
    require_model_credentials,
    resolve_writing_profile_env,
)


class RuntimeBootstrapTests(unittest.TestCase):
    def test_initialize_runtime_should_sync_output_env_and_create_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "research-root"
            with mock.patch.dict("os.environ", {}, clear=False):
                context = initialize_runtime(
                    source_file=Path(td) / "entry.py",
                    output_root=output_root,
                    ensure_dirs=True,
                    apply_cache=True,
                )
                self.assertEqual(
                    str(output_root.resolve()),
                    os.environ["RESEARCH_OUTPUT_DIR"],
                )

            self.assertIsInstance(context, RuntimeContext)
            self.assertEqual(context.project_root, Path(td).resolve())
            self.assertEqual(context.research_root, output_root.resolve())
            self.assertEqual(
                context.applied_cache_env["HF_HOME"],
                str(output_root.resolve() / "cache" / "huggingface"),
            )
            self.assertTrue((output_root / "projects").exists())

    def test_format_project_relative_path_should_prefer_relative_display(self) -> None:
        project_root = Path("/tmp/project")
        self.assertEqual(
            format_project_relative_path(
                project_root / "outputs" / "run1",
                project_root=project_root,
            ),
            "outputs/run1",
        )

    def test_resolve_writing_profile_env_should_fallback_on_invalid_value(self) -> None:
        observed: list[str] = []
        with mock.patch.dict(
            "os.environ",
            {"AI_SCIENTIST_WRITING_PROFILE": "does-not-exist"},
            clear=False,
        ):
            profile = resolve_writing_profile_env(
                invalid_profile_logger=lambda exc, raw: observed.append(
                    f"{raw}:{exc}"
                )
            )
        self.assertEqual(profile, "default")
        self.assertEqual(len(observed), 1)

    def test_require_env_var_should_raise_when_missing(self) -> None:
        messages: list[str] = []
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                require_env_var(
                    "ZHIPU_API_KEY",
                    missing_message="missing key",
                    hint="set the key first",
                    logger=messages.append,
                )
        self.assertEqual(messages, ["missing key", "set the key first"])

    def test_initialize_runtime_should_set_project_root_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source_file = Path(td) / "bin" / "entry.py"
            with mock.patch.dict("os.environ", {}, clear=False):
                context = initialize_runtime(
                    source_file=source_file,
                    ensure_dirs=False,
                    apply_cache=False,
                )
                self.assertEqual(
                    os.environ[PROJECT_ROOT_ENV_VAR],
                    str(source_file.resolve().parent),
                )

            self.assertEqual(
                context.project_root,
                source_file.resolve().parent,
            )

    def test_require_model_credentials_should_accept_multi_provider_models(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "router-key",
                "OPENAI_COMPAT_API_KEY": "compat-key",
                "OPENAI_COMPAT_BASE_URL": "https://compat.example/v1",
            },
            clear=False,
        ):
            require_model_credentials(
                [
                    "openrouter/meta-llama/llama-3.1-405b-instruct",
                    "openai_compat/qwen2.5-72b-instruct",
                ],
                logger=lambda _: None,
            )

    def test_require_model_credentials_should_raise_with_missing_provider_envs(self) -> None:
        messages: list[str] = []
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                require_model_credentials(
                    ["gemini/gemini-2.5-pro-preview-03-25"],
                    logger=messages.append,
                )
        self.assertIn("缺少所需服务商凭证", messages[0])
        self.assertIn("GEMINI_API_KEY | GOOGLE_API_KEY", "\n".join(messages))


if __name__ == "__main__":
    unittest.main()
