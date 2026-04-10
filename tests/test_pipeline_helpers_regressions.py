from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.utils.pipeline_helpers import compile_latex


class PipelineHelpersRegressionTests(unittest.TestCase):
    def test_compile_latex_should_fail_on_nonzero_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            with mock.patch(
                "ai_scientist.utils.pipeline_helpers.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["pdflatex", "-interaction=nonstopmode", "template.tex"],
                    returncode=1,
                    stdout="",
                    stderr="latex error",
                ),
            ) as run_mock:
                result = compile_latex(cwd, verbose=False)

        self.assertFalse(result)
        self.assertEqual(run_mock.call_count, 1)

    def test_compile_latex_should_fail_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cwd = Path(td)
            with mock.patch(
                "ai_scientist.utils.pipeline_helpers.subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["pdflatex", "-interaction=nonstopmode", "template.tex"],
                    timeout=5,
                ),
            ) as run_mock:
                result = compile_latex(cwd, timeout=5, verbose=False)

        self.assertFalse(result)
        self.assertEqual(run_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
