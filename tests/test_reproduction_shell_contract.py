from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist import research_git


class ReproductionShellContractTests(unittest.TestCase):
    def test_unquoted_shell_operators_are_rejected(self) -> None:
        for operator in ("&&", "||", ";", "|", "&", ">", ">>", "<", "2>"):
            for spacing in ("", " "):
                command = f"python first.py{spacing}{operator}{spacing}second.py"
                with self.subTest(command=command):
                    with self.assertRaisesRegex(research_git.ResearchGitError, "shell-free"):
                        research_git._parse_reproduction_command(command)

    def test_quoted_and_escaped_operators_remain_literal_arguments(self) -> None:
        for command, expected in (
            ("python check.py 'a&&b'", "a&&b"),
            ('python check.py "a;b"', "a;b"),
            (r"python check.py a\&\&b", "a&&b"),
            (r"python check.py a\;b", "a;b"),
            (r'''python check.py escaped\"\&value''', 'escaped"&value'),
            ('''python -c 'print("a>b"); print("ok")' ''', 'print("a>b"); print("ok")'),
        ):
            with self.subTest(command=command):
                argv = research_git._parse_reproduction_command(command, platform_name="posix")
                self.assertEqual(argv[-1], expected)

    def test_invalid_command_does_not_create_checkpoint_or_change_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "research"
            research_git.init_repository(root, question="Synthetic product test only.")
            artifact = root / "hypotheses" / "h1.json"
            artifact.write_text('{"statement":"synthetic"}\n', encoding="utf-8")
            research_git.research_stage(root, ["hypotheses/h1.json"])
            before = research_git.repository_status(root)
            with self.assertRaisesRegex(research_git.ResearchGitError, "shell-free"):
                research_git.commit_research_stage(
                    root, stage="review", subject="invalid replay command",
                    reproduce_command="python first.py && python second.py",
                )
            after = research_git.repository_status(root)
            self.assertEqual(after["head"], before["head"])
            self.assertEqual(after["research_stage"], before["research_stage"])
            self.assertEqual(after["staged_paths"], [])

    def test_legacy_invalid_command_is_rejected_before_worktree_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "research"
            research_git.init_repository(root, question="Synthetic product test only.")
            artifact = root / "hypotheses" / "h1.json"
            artifact.write_text('{"statement":"synthetic"}\n', encoding="utf-8")
            with mock.patch.object(research_git, "_parse_reproduction_command", return_value=[]):
                research_git.create_checkpoint(
                    root, stage="review", subject="legacy replay fixture",
                    reproduce_command="python first.py && python second.py",
                )
            destination = Path(temporary) / "replay"
            with self.assertRaisesRegex(research_git.ResearchGitError, "shell-free"):
                research_git.reproduce_checkpoint(root, destination=destination, execute=True)
            self.assertFalse(destination.exists())

    def test_single_python_entry_point_executes_all_checks(self) -> None:
        argv = research_git._parse_reproduction_command(
            '''python -c 'import sys; print("first"); sys.exit(7)' '''
        )
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout.strip(), "first")


if __name__ == "__main__":
    unittest.main()
