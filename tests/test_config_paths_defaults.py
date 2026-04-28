from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.config import paths as config_paths
from ai_scientist.config.paths import _resolve_default_research_dir


class ConfigPathsDefaultTests(unittest.TestCase):
    def test_should_preserve_legacy_documents_path_in_compat_mode(self) -> None:
        home_dir = Path("/tmp/home")
        result = _resolve_default_research_dir(
            home_dir=home_dir,
            platform_name="posix",
            xdg_data_home="/tmp/xdg-data",
            legacy_documents_dir_exists=True,
            prefer_project_sibling=False,
        )
        self.assertEqual(result, home_dir / "Documents" / "research")

    def test_should_prefer_repo_sibling_output_when_legacy_missing(self) -> None:
        home_dir = Path("/tmp/home")
        project_root = Path("/tmp/work/XScientist")
        result = _resolve_default_research_dir(
            home_dir=home_dir,
            platform_name="posix",
            xdg_data_home="/tmp/xdg-data",
            legacy_documents_dir_exists=False,
            project_root=project_root,
            project_parent_writable=True,
        )
        self.assertEqual(result, project_root.resolve().parent / "XScientist_outputs")

    def test_should_support_windows_local_appdata_fallback_when_requested(self) -> None:
        home_dir = Path("C:/Users/tester")
        result = _resolve_default_research_dir(
            home_dir=home_dir,
            platform_name="nt",
            xdg_data_home=None,
            legacy_documents_dir_exists=False,
            prefer_project_sibling=False,
        )
        self.assertEqual(result, home_dir / "AppData" / "Local" / "ai_scientist" / "research")

    def test_should_use_xdg_data_home_when_provided_in_compat_mode(self) -> None:
        home_dir = Path("/tmp/home")
        result = _resolve_default_research_dir(
            home_dir=home_dir,
            platform_name="posix",
            xdg_data_home="/tmp/xdg-data",
            legacy_documents_dir_exists=False,
            prefer_project_sibling=False,
        )
        self.assertEqual(result, Path("/tmp/xdg-data/ai_scientist/research"))

    def test_should_fallback_when_project_parent_not_writable(self) -> None:
        home_dir = Path("/tmp/home")
        result = _resolve_default_research_dir(
            home_dir=home_dir,
            platform_name="posix",
            xdg_data_home="/tmp/xdg-data",
            legacy_documents_dir_exists=False,
            project_root=Path("/tmp/work/ai_scientist"),
            project_parent_writable=False,
        )
        self.assertEqual(result, Path("/tmp/xdg-data/ai_scientist/research"))

    def test_should_fallback_to_local_share_when_xdg_missing_in_compat_mode(self) -> None:
        home_dir = Path("/tmp/home")
        result = _resolve_default_research_dir(
            home_dir=home_dir,
            platform_name="posix",
            xdg_data_home=None,
            legacy_documents_dir_exists=False,
            prefer_project_sibling=False,
        )
        self.assertEqual(result, home_dir / ".local" / "share" / "ai_scientist" / "research")

    def test_should_resolve_output_path_from_runtime_env_override(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"RESEARCH_OUTPUT_DIR": "/tmp/runtime-output"},
            clear=False,
        ):
            self.assertEqual(
                config_paths.resolve_output_path(),
                Path("/tmp/runtime-output"),
            )

    def test_cache_env_vars_should_follow_explicit_output_root(self) -> None:
        cache_env = config_paths.get_cache_env_vars(output_root="/tmp/research-root")
        self.assertEqual(cache_env["HF_HOME"], "/tmp/research-root/cache/huggingface")
        self.assertEqual(cache_env["TORCH_HOME"], "/tmp/research-root/cache/torch")
        self.assertEqual(cache_env["WANDB_DIR"], "/tmp/research-root/cache/wandb")


if __name__ == "__main__":
    unittest.main()
