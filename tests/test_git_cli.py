from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from xscientist.entrypoints import git_main
from xscientist.git_support import inspect_git_backend
from xscientist.research_vcs import ResearchRepository
from xscientist.cli import main as cli_main


class GitDoctorTests(unittest.TestCase):
    def test_missing_backend_is_actionable_and_path_safe(self) -> None:
        with mock.patch("xscientist.git_support.shutil.which", return_value=None):
            report = inspect_git_backend()

        self.assertFalse(report["ok"])
        self.assertFalse(report["available"])
        self.assertIn("Git is required", report["errors"][0])
        self.assertTrue(report["install_hint"])
        self.assertFalse(report["host_paths_disclosed"])

    def test_installed_backend_supports_native_merge_preflight(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("Git is not installed")
        report = inspect_git_backend()

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["capabilities"]["switch"])
        self.assertTrue(report["capabilities"]["merge_tree_write"])
        self.assertNotIn(str(Path.home()), str(report))


class GitStyleCliTests(unittest.TestCase):
    def test_unified_cli_exposes_git_doctor(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("Git is not installed")
        with redirect_stdout(io.StringIO()) as output:
            result = cli_main(["git", "doctor", "--json"])

        self.assertEqual(result, 0)
        self.assertIn('"xscientist.git-doctor.v1"', output.getvalue())

    def test_git_add_and_commit_drive_native_research_stage(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("Git is not installed")
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            root = Path(td) / "study"
            self.assertEqual(
                git_main(["init", str(root), "--question", "Falsifiable question"]),
                0,
            )
            repository = ResearchRepository(root)
            hypothesis = repository.record(
                "hypothesis",
                {"statement": "H1", "falsifier": "delta <= 0"},
            )
            self.assertEqual(git_main(["add", "-A", "--repo", str(root)]), 0)
            self.assertEqual(
                git_main(
                    [
                        "commit",
                        "--repo",
                        str(root),
                        "--stage",
                        "ideation",
                        "-m",
                        "record H1",
                    ]
                ),
                0,
            )
            blame = repository.blame(hypothesis.object_id)

        self.assertEqual(blame["object"]["kind"], "hypothesis")


if __name__ == "__main__":
    unittest.main()
