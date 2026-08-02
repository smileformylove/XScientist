from __future__ import annotations

import subprocess
import sys
import unittest

from tools.performance_regression import (
    DEFAULT_PROFILES,
    PerformanceRegressionError,
    compare_results,
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


if __name__ == "__main__":
    unittest.main()
