from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.performance_regression import (
    DEFAULT_PROFILES,
    PerformanceRegressionError,
    compare_results,
    record_profiles,
)


def _result(*, seconds: float, rss: float) -> dict:
    return {
        "schema_version": 1,
        "environment": {"python": "test", "platform": "test"},
        "profiles": {
            "profile": {
                "median_seconds": seconds,
                "median_max_rss_native": rss,
            }
        },
    }


class PerformanceRegressionTests(unittest.TestCase):
    def test_default_profiles_cover_public_and_long_running_entrypoints(self) -> None:
        self.assertEqual(
            set(DEFAULT_PROFILES),
            {
                "public_cli_import",
                "project_import",
                "batch_import",
                "daemon_import",
                "quality_pipeline_import",
            },
        )

    def test_comparison_allows_noise_but_rejects_material_regression(self) -> None:
        baseline = _result(seconds=0.100, rss=10_000)
        small_change = _result(seconds=0.108, rss=10_700)
        regression = _result(seconds=0.130, rss=13_000)
        self.assertTrue(compare_results(baseline, small_change)["passed"])
        check = compare_results(baseline, regression)
        self.assertFalse(check["passed"])
        self.assertEqual(check["failures"], ["profile:time", "profile:rss"])

    def test_comparison_rejects_different_environments(self) -> None:
        baseline = _result(seconds=0.1, rss=1)
        candidate = _result(seconds=0.1, rss=1)
        candidate["environment"]["python"] = "different"
        with self.assertRaisesRegex(PerformanceRegressionError, "same Python"):
            compare_results(baseline, candidate)

    def test_recording_uses_one_isolated_prewarmed_bytecode_cache(self) -> None:
        cache_paths: list[Path] = []

        def fake_measure(
            snippet: str,
            *,
            cwd: Path,
            env_overrides: dict[str, str],
        ) -> dict[str, float]:
            self.assertEqual(snippet, "pass")
            self.assertTrue(cwd.is_absolute())
            cache_paths.append(Path(env_overrides["PYTHONPYCACHEPREFIX"]))
            return {"seconds": 0.01, "max_rss_native": 100.0}

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch(
                "tools.performance_regression._measure_once",
                side_effect=fake_measure,
            ),
        ):
            result = record_profiles(
                cwd=td,
                repeats=3,
                profiles={"profile": "pass"},
            )

        self.assertEqual(len(cache_paths), 4)
        self.assertEqual(len(set(cache_paths)), 1)
        self.assertFalse(cache_paths[0].exists())
        self.assertEqual(
            result["environment"]["bytecode_cache_mode"],
            "isolated-prewarmed",
        )

    def test_public_cli_import_stays_free_of_heavy_runtime_dependencies(self) -> None:
        forbidden = {
            "anthropic",
            "datasets",
            "matplotlib",
            "numpy",
            "openai",
            "pandas",
            "pymupdf",
            "seaborn",
            "sklearn",
            "transformers",
            "zhipuai",
        }
        code = (
            "import sys; import xscientist.cli; "
            f"forbidden={forbidden!r}; "
            "loaded=forbidden.intersection(sys.modules); "
            "assert not loaded, sorted(loaded)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_public_package_defers_sdk_imports_until_attribute_access(self) -> None:
        code = """
import sys
import xscientist

assert "xscientist.client" not in sys.modules
assert "xscientist.models" not in sys.modules
assert {"XScientist", "ProjectRequest", "CommandResult", "ServiceSettings"} <= set(dir(xscientist))

from xscientist import CommandResult, ProjectRequest, ServiceSettings, XScientist

assert xscientist.XScientist is XScientist
assert xscientist.ProjectRequest is ProjectRequest
assert xscientist.CommandResult is CommandResult
assert xscientist.ServiceSettings is ServiceSettings
assert "xscientist.client" in sys.modules
assert "xscientist.models" in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
