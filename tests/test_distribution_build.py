from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class DistributionBuildTests(unittest.TestCase):
    def test_wheel_contains_public_api_entrypoints_and_runtime_resources(self) -> None:
        if shutil.which(sys.executable) is None:
            self.skipTest("python executable unavailable")
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(out_dir),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if (
                completed.returncode != 0
                and "No module named build" in completed.stderr
            ):
                self.skipTest("build frontend not installed")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            wheel = next(out_dir.glob("xscientist-*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())

            required = {
                "xscientist/__init__.py",
                "xscientist/cli.py",
                "xscientist/service.py",
                "run_project.py",
                "ai_scientist/apps/project.py",
                "ai_scientist/resources/configs/bfts_default.yaml",
                "ai_scientist/blank_icbinb_latex/template.tex",
                "ai_scientist/treesearch/utils/viz_templates/template.html",
                "ai_scientist/protocol/schemas/manifest.schema.json",
            }
            self.assertFalse(
                required - names, f"missing wheel files: {required - names}"
            )
            self.assertFalse(
                any(name.startswith("tests/") for name in names),
                "wheel should not include the repository test suite",
            )


if __name__ == "__main__":
    unittest.main()
