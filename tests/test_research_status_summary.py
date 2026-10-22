"""Compact status presentation must not change repository scans or full output."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from unittest import TestCase, main
from unittest.mock import patch

from xscientist.research_cli import main as research_main, _status_summary


def fixture():
    return {
        "repository": "/private/status-fixture", "name": "status fixture",
        "branch": "main", "head": "a" * 40, "checkpoint_policy": "milestone",
        "auto_commit": True, "auto_push": False, "worktree_clean": False,
        "staged_paths": ["backend.txt"], "tracked_changes": ["tracked.txt"],
        "research_stage": {"head": "b" * 40, "paths": ["research.txt", "other.txt"]},
        "eligible_changes": [f"eligible-{i}.txt" for i in range(5000)],
        "excluded_changes": [f"excluded-{i}.txt (policy)" for i in range(3000)],
        "last_checkpoint": {"checkpoint_id": "rcp-example", "sequence": 50,
                            "stage": "theory", "status": "completed", "created_at": "2026-09-19",
                            "changed_paths": ["hidden-checkpoint-path.txt"],
                            "reproduce": {"command": "hidden-command"}},
        "object_store": {"path": "/private/status-fixture/.ara-store", "objects": 8, "bytes": 128},
    }


class SummaryTests(TestCase):
    def invoke(self, data, *args):
        out = io.StringIO()
        with patch("xscientist.research_cli.repository_status", return_value=data) as status:
            with redirect_stdout(out):
                self.assertEqual(research_main(["status", "--repo", "fixture", *args]), 0)
            status.assert_called_once_with("fixture")
        return out.getvalue()

    def test_exact_counts_do_not_mutate_full_status(self):
        data = fixture()
        before = deepcopy(data)
        result = _status_summary(data)
        self.assertEqual(result["change_counts"], {
            "git_staged": 1, "research_staged": 2, "tracked": 1, "eligible": 5000, "excluded": 3000,
        })
        self.assertEqual(result["research_stage_head"], "b" * 40)
        self.assertEqual(data, before)
        self.assertNotIn("changed_paths", result["last_checkpoint"])
        self.assertLess(len(json.dumps(result)), 2000)

    def test_json_summary_has_identity_not_paths_or_payloads(self):
        output = self.invoke(fixture(), "--summary", "--json")
        result = json.loads(output)
        self.assertEqual(result["view"], "summary")
        self.assertFalse(result["worktree_clean"])
        self.assertEqual(result["last_checkpoint"]["sequence"], 50)
        for omitted in ("eligible-", "excluded-", "hidden-command", "hidden-checkpoint-path", "/private/"):
            self.assertNotIn(omitted, output)
        self.assertIn("[REDACTED_PATH]", output)

    def test_human_summary_is_compact_and_redacted(self):
        output = self.invoke(fixture(), "--summary")
        self.assertIn("Eligible changes:  5000", output)
        self.assertIn("Excluded changes:  3000", output)
        self.assertIn("omitted (--summary)", output)
        self.assertNotIn("eligible-", output)
        self.assertNotIn("/private/", output)
        self.assertLess(len(output.splitlines()), 20)

    def test_default_json_remains_full(self):
        result = json.loads(self.invoke(fixture(), "--json"))
        self.assertNotIn("view", result)
        self.assertEqual(len(result["eligible_changes"]), 5000)
        self.assertEqual(result["last_checkpoint"]["reproduce"]["command"], "hidden-command")

    def test_default_human_keeps_paths(self):
        output = self.invoke(fixture())
        self.assertIn("eligible-4999.txt", output)
        self.assertIn("excluded-2999.txt", output)

    def test_no_checkpoint_or_changes(self):
        data = fixture()
        data.update(last_checkpoint=None, worktree_clean=True, eligible_changes=[], excluded_changes=[])
        result = json.loads(self.invoke(data, "--summary", "--json"))
        self.assertIsNone(result["last_checkpoint"])
        self.assertTrue(result["worktree_clean"])
        self.assertEqual(result["change_counts"]["eligible"], 0)


if __name__ == "__main__":
    main()
