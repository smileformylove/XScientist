from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist.cli import main as cli_main
from xscientist.research_git import ResearchGitError
from xscientist.research_journey import explore_research_idea
from xscientist.workspace_history import (
    compare_workspace_history,
    inspect_workspace_checkpoint,
    inspect_workspace_history,
    preview_workspace_rollback,
    public_workspace_history_payload,
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
            generated = workspace / "research-dag" / "research-dag.html"
            generated.parent.mkdir()
            generated.write_text("<html>rebuildable</html>\n", encoding="utf-8")
            preview = preview_workspace_rollback(workspace, commit=saved_commit)
            self.assertTrue(preview["ready_to_apply"])
            self.assertFalse(preview["history_rewritten"])
            self.assertIn("question.md", "\n".join(preview["impact"]["changes"]))
            self.assertIn("--apply", preview["apply_command"])
            self.assertEqual(preview["preserved"]["count"], 1)
            self.assertIn("A deliberately bounded note", question.read_text())

            rolled_back = rollback_workspace_checkpoint(
                workspace,
                commit=saved_commit,
            )
            self.assertFalse(rolled_back["history_rewritten"])
            self.assertEqual(question.read_text(encoding="utf-8"), original)
            self.assertTrue(generated.is_file())
            self.assertIn("research dag", rolled_back["next_actions"][1])
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
                    "tracked_worktree_changes",
                    "initial_checkpoint",
                },
            )
            self.assertEqual(preview["preserved"]["count"], 1)
            self.assertIn("scratch.tmp", preview["preserved"]["policy_excluded"][0])

    def test_rollback_removes_files_added_by_checkpoint_and_preserves_views(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._workspace(Path(tmp))
            added = workspace / "04_logs" / "progress.json"
            added.parent.mkdir()
            added.write_text('{"current_stage":"planned"}\n', encoding="utf-8")
            saved = save_workspace_checkpoint(workspace, message="record progress")
            view = workspace / "research-dag" / "research-dag.html"
            view.parent.mkdir()
            view.write_text("<html>rebuildable</html>\n", encoding="utf-8")

            preview = preview_workspace_rollback(
                workspace,
                commit=saved["checkpoint"]["commit"],
            )
            self.assertTrue(preview["ready_to_apply"])
            rolled_back = rollback_workspace_checkpoint(
                workspace,
                commit=saved["checkpoint"]["commit"],
            )

            self.assertFalse(added.exists())
            self.assertTrue(view.is_file())
            self.assertTrue(inspect_workspace_history(workspace)["clean"])
            self.assertTrue(rolled_back["result"]["checkpoint"]["committed"])

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

    def test_history_json_uses_portable_workspace_for_list_save_and_rollback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            private_root = Path(tmp) / "private workspace parent"
            workspace = self._workspace(private_root)
            private_path = str(workspace)

            list_output = io.StringIO()
            with contextlib.redirect_stdout(list_output):
                self.assertEqual(
                    cli_main(["history", "list", private_path, "--json"]),
                    0,
                )
            listed = json.loads(list_output.getvalue())
            self.assertEqual(listed["workspace"], ".")
            self.assertNotIn(private_path, list_output.getvalue())

            question = workspace / "question.md"
            original = question.read_text(encoding="utf-8")
            question.write_text(
                original + "\nPortable output check.\n", encoding="utf-8"
            )
            save_output = io.StringIO()
            with contextlib.redirect_stdout(save_output):
                self.assertEqual(
                    cli_main(
                        [
                            "history",
                            "save",
                            private_path,
                            "--message",
                            "portable history output",
                            "--json",
                        ]
                    ),
                    0,
                )
            saved = json.loads(save_output.getvalue())
            self.assertEqual(saved["workspace"], ".")
            self.assertNotIn(private_path, save_output.getvalue())

            saved_commit = saved["checkpoint"]["commit"]
            preview_output = io.StringIO()
            with contextlib.redirect_stdout(preview_output):
                self.assertEqual(
                    cli_main(
                        [
                            "history",
                            "rollback",
                            private_path,
                            "--commit",
                            saved_commit,
                            "--json",
                        ]
                    ),
                    0,
                )
            preview = json.loads(preview_output.getvalue())
            self.assertEqual(preview["workspace"], ".")
            self.assertIn("history rollback .", preview["apply_command"])
            self.assertNotIn(private_path, preview_output.getvalue())

            apply_output = io.StringIO()
            with contextlib.redirect_stdout(apply_output):
                self.assertEqual(
                    cli_main(
                        [
                            "history",
                            "rollback",
                            private_path,
                            "--commit",
                            saved_commit,
                            "--apply",
                            "--json",
                        ]
                    ),
                    0,
                )
            applied = json.loads(apply_output.getvalue())
            self.assertEqual(applied["workspace"], ".")
            self.assertEqual(applied["preview"]["workspace"], ".")
            self.assertIn("xscientist status .", applied["next_actions"])
            self.assertNotIn(private_path, apply_output.getvalue())

    def test_history_public_payload_recursively_redacts_actions_details_and_errors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "private" / "study"
            outside = Path(tmp) / "private" / "trace.log"
            public = public_workspace_history_payload(
                {
                    "workspace": str(workspace),
                    "nested": {
                        "action": f"xscientist status {workspace}",
                        "detail": f"repository {workspace} is unavailable",
                        "error": [f"trace written to {outside}"],
                    },
                },
                workspace=workspace,
            )

            rendered = json.dumps(public)
            self.assertEqual(public["workspace"], ".")
            self.assertEqual(public["nested"]["action"], "xscientist status .")
            self.assertNotIn(str(workspace), rendered)
            self.assertNotIn(str(outside), rendered)

    def test_history_errors_are_redacted_in_json_and_human_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "private-study"
            private_trace = Path(tmp) / "elsewhere" / "trace.log"
            error = ResearchGitError(
                f"cannot inspect {workspace}; details at {private_trace}"
            )

            json_output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.workspace_history.inspect_workspace_history",
                    side_effect=error,
                ),
                contextlib.redirect_stdout(json_output),
            ):
                self.assertEqual(
                    cli_main(["history", "list", str(workspace), "--json"]),
                    2,
                )
            rendered_json = json_output.getvalue()
            self.assertNotIn(str(workspace), rendered_json)
            self.assertNotIn(str(private_trace), rendered_json)

            error_output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.workspace_history.inspect_workspace_history",
                    side_effect=error,
                ),
                contextlib.redirect_stderr(error_output),
            ):
                self.assertEqual(
                    cli_main(["history", "list", str(workspace)]),
                    2,
                )
            rendered_error = error_output.getvalue()
            self.assertNotIn(str(workspace), rendered_error)
            self.assertNotIn(str(private_trace), rendered_error)

    def test_checkpoint_show_and_semantic_diff_are_compact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self._workspace(Path(tmp))
            question = workspace / "question.md"
            question.write_text(
                question.read_text(encoding="utf-8") + "\nBounded note.\n",
                encoding="utf-8",
            )
            saved = save_workspace_checkpoint(workspace, message="record bounded note")

            shown = inspect_workspace_checkpoint(
                workspace,
                commit=saved["checkpoint"]["commit"],
            )
            compared = compare_workspace_history(workspace)

            self.assertTrue(shown["checkpoint_hash_valid"])
            self.assertFalse(shown["payloads_disclosed"])
            self.assertEqual(shown["checkpoint"]["subject"], "record bounded note")
            self.assertIn("question.md", shown["checkpoint"]["changed_paths"])
            self.assertFalse(compared["payloads_disclosed"])
            self.assertIn("M\tquestion.md", compared["changes"])

            show_output = io.StringIO()
            diff_output = io.StringIO()
            with contextlib.redirect_stdout(show_output):
                self.assertEqual(
                    cli_main(["history", "show", str(workspace)]),
                    0,
                )
            with contextlib.redirect_stdout(diff_output):
                self.assertEqual(
                    cli_main(["history", "diff", str(workspace)]),
                    0,
                )
            self.assertIn("Hash:        valid", show_output.getvalue())
            self.assertIn("Environment changed:", diff_output.getvalue())

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
            self.assertNotIn(str(workspace), preview_output.getvalue())
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
            self.assertNotIn(str(workspace), apply_output.getvalue())
            self.assertEqual(question.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
