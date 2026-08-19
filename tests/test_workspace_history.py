from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from xscientist.cli import main as cli_main
from xscientist.research_journey import explore_research_idea
from xscientist.workspace_history import (
    inspect_workspace_history,
    preview_workspace_rollback,
    rollback_workspace_checkpoint,
    save_workspace_checkpoint,
)


class WorkspaceHistoryTests(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        workspace = root / "study"
        explore_research_idea(
            workspace,
            idea="Does a short walk improve sleep?",
            expectation="The preregistered sleep score improves.",
            disconfirming_result="The score is unchanged or worse.",
            first_test="Compare walking and usual-activity periods.",
            git_user_name="History Tester",
            git_user_email="history@example.test",
        )
        return workspace

    def test_save_and_append_only_rollback_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._workspace(Path(tmp))
            question = workspace / "question.md"
            original = question.read_text(encoding="utf-8")
            question.write_text(
                original + "\nA deliberately bounded note.\n", encoding="utf-8"
            )

            saved = save_workspace_checkpoint(
                workspace,
                message="record bounded note",
            )
            self.assertTrue(saved["checkpoint"]["committed"])
            saved_commit = saved["checkpoint"]["commit"]
            preview = preview_workspace_rollback(workspace, commit=saved_commit)
            self.assertTrue(preview["ready_to_apply"])
            self.assertFalse(preview["history_rewritten"])
            self.assertIn("question.md", "\n".join(preview["impact"]["changes"]))
            self.assertIn("--apply", preview["apply_command"])
            self.assertIn("A deliberately bounded note", question.read_text())

            rolled_back = rollback_workspace_checkpoint(
                workspace,
                commit=saved_commit,
            )
            self.assertFalse(rolled_back["history_rewritten"])
            self.assertEqual(question.read_text(encoding="utf-8"), original)
            history = inspect_workspace_history(workspace)
            self.assertEqual(
                history["entries"][0]["trailers"]["Research-Stage"], ["revert"]
            )
            commits = {entry["commit"] for entry in history["entries"]}
            self.assertIn(saved_commit, commits)
            self.assertIn(rolled_back["result"]["revert_commit"], commits)

    def test_rollback_preview_blocks_unsaved_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._workspace(Path(tmp))
            (workspace / "question.md").write_text("unsaved\n", encoding="utf-8")
            (workspace / "scratch.tmp").write_text("excluded\n", encoding="utf-8")

            preview = preview_workspace_rollback(workspace)

            self.assertFalse(preview["ready_to_apply"])
            self.assertEqual(
                {item["code"] for item in preview["blockers"]},
                {
                    "unsaved_research_changes",
                    "excluded_worktree_changes",
                    "initial_checkpoint",
                },
            )

    def test_cli_history_and_audit_facades_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._workspace(Path(tmp))
            history_output = io.StringIO()
            with contextlib.redirect_stdout(history_output):
                code = cli_main(
                    ["history", "list", str(workspace), "--limit", "2", "--json"]
                )
            self.assertEqual(code, 0)
            history = json.loads(history_output.getvalue())
            self.assertEqual(
                history["schema_version"], "xscientist.workspace-history.v1"
            )
            self.assertEqual(len(history["entries"]), 1)
            self.assertFalse(history["auto_push"])

            audit_output = io.StringIO()
            with contextlib.redirect_stdout(audit_output):
                audit_code = cli_main(
                    ["audit", str(workspace), "--level", "trace", "--json"]
                )
            audit = json.loads(audit_output.getvalue())
            self.assertEqual(audit_code, 1)
            self.assertEqual(audit["target_level"], "trace")
            self.assertFalse(audit["complete"])

    def test_cli_rollback_requires_explicit_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._workspace(Path(tmp))
            question = workspace / "question.md"
            original = question.read_text(encoding="utf-8")
            question.write_text(original + "\nchanged\n", encoding="utf-8")
            saved = save_workspace_checkpoint(workspace, message="change question")

            preview_output = io.StringIO()
            with contextlib.redirect_stdout(preview_output):
                code = cli_main(
                    [
                        "history",
                        "rollback",
                        str(workspace),
                        "--commit",
                        saved["checkpoint"]["commit"],
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("preview only", preview_output.getvalue())
            self.assertTrue(question.read_text(encoding="utf-8").endswith("changed\n"))

            apply_output = io.StringIO()
            with contextlib.redirect_stdout(apply_output):
                code = cli_main(
                    [
                        "history",
                        "rollback",
                        str(workspace),
                        "--commit",
                        saved["checkpoint"]["commit"],
                        "--apply",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("history was not rewritten", apply_output.getvalue())
            self.assertEqual(question.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
