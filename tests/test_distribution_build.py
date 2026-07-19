from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
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
                    "--sdist",
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
            sdist = next(out_dir.glob("xscientist-*.tar.gz"))
            with tarfile.open(sdist) as archive:
                sdist_names = set(archive.getnames())
                archive.extractall(out_dir)
            sdist_root = next(iter(sdist_names)).split("/", 1)[0]
            extracted_root = out_dir / sdist_root
            wheel_build = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(out_dir),
                ],
                cwd=extracted_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(wheel_build.returncode, 0, wheel_build.stderr)
            wheel = next(out_dir.glob("xscientist-*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())

            required = {
                "xscientist/__init__.py",
                "xscientist/cli.py",
                "xscientist/service.py",
                "run_project.py",
                "ai_scientist/apps/project.py",
                "continuous_paper_generator.py",
                "ai_scientist/apps/batch.py",
                "ai_scientist/apps/batch_experiment_artifacts.py",
                "research_manager.py",
                "ai_scientist/apps/manager.py",
                "continuous_research_daemon.py",
                "ai_scientist/apps/daemon.py",
                "ai_scientist/apps/daemon_dashboard.py",
                "ai_scientist/apps/daemon_reports.py",
                "auth_cli.py",
                "ai_scientist/apps/auth.py",
                "feedback_cli.py",
                "ai_scientist/apps/feedback.py",
                "run_ara_fork.py",
                "ai_scientist/apps/ara.py",
                "validate_repo.py",
                "ai_scientist/apps/validate.py",
                "launch_scientist_bfts.py",
                "ai_scientist/apps/bfts.py",
                "launch_scientist_zhipu.py",
                "ai_scientist/apps/zhipu.py",
                "preflight_check.py",
                "ai_scientist/apps/preflight.py",
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
            source_only = {"run_daemon_profile.py", "run_daemon_rehearsal.py"}
            self.assertFalse(
                source_only & names,
                f"wheel should not include source-only ops: {source_only & names}",
            )
            missing_source_ops = {
                path
                for path in source_only
                if f"{sdist_root}/{path}" not in sdist_names
            }
            self.assertFalse(
                missing_source_ops,
                f"sdist is missing source-only ops: {missing_source_ops}",
            )
            self.assertFalse(
                any(
                    name.startswith(("tools/", "docs/", "examples/")) for name in names
                ),
                "wheel should not include source-only maintenance assets",
            )
            self.assertIn(
                f"{sdist_root}/tools/repository_validation.py",
                sdist_names,
                "sdist should retain repository-only validation",
            )
            for source_asset in (
                "docs/CONFIG_REFERENCE.md",
                "docs/OPERATIONS_CHECKLIST.md",
                "examples/example_topic.md",
            ):
                self.assertIn(
                    f"{sdist_root}/{source_asset}",
                    sdist_names,
                    f"sdist should retain source operation asset: {source_asset}",
                )


if __name__ == "__main__":
    unittest.main()
