from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.research_actions import ResearchActionLedger
from xscientist.research_git import ResearchGitError
from xscientist.research_vcs import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required")
class ResearchActionLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = ResearchRepository.init(
            Path(self.tmp.name) / "study",
            question="Can action receipts improve continuity?",
            git_user_name="Action Test",
            git_user_email="action@example.invalid",
        )
        self.ledger = ResearchActionLedger(self.repo)

    def test_begin_is_idempotent_and_does_not_store_private_input(self) -> None:
        started = self.ledger.begin(
            "act-video-inspect-1",
            "video.inspect",
            {"prompt": "private user note", "asset": "rso-aaaaaaaaaaaaaaaa"},
        )
        repeated = self.ledger.begin(
            "act-video-inspect-1",
            "video.inspect",
            {"prompt": "private user note", "asset": "rso-aaaaaaaaaaaaaaaa"},
        )
        self.assertEqual(started["object_id"], repeated["object_id"])
        self.assertNotIn("private user note", str(started))
        self.assertEqual(started["payload"]["sequence"], 0)

    def test_completion_binds_consumed_and_produced_objects(self) -> None:
        source = self.repo.record("hypothesis", {"statement": "Video signal is useful."})
        evidence = self.repo.record("evidence", {"summary": "Observed signal"})
        started = self.ledger.begin(
            "act-video-search-1",
            "video.search",
            {"query": "signal"},
            consumes=[source.object_id],
            budget_seconds=30,
        )
        output = {"matches": [evidence.object_id]}
        finished = self.ledger.finish(
            "act-video-search-1",
            "completed",
            output,
            produces=[evidence.object_id],
            consumed_seconds=3.25,
        )
        self.assertEqual(finished["payload"]["status"], "completed")
        self.assertEqual(finished["payload"]["output_hash"], canonical_content_hash(output))
        self.assertIn(
            {"type": "produces", "target": evidence.object_id}, finished["relations"]
        )
        self.assertEqual(len(self.ledger.history("act-video-search-1")), 2)
        self.assertEqual(
            self.ledger.finish("act-video-search-1", "completed", output)["object_id"],
            finished["object_id"],
        )

    def test_interruption_requires_explicit_retry_and_preserves_lineage(self) -> None:
        self.ledger.begin("act-recover-1", "experiment.run", {"seed": 7})
        interrupted = self.ledger.finish(
            "act-recover-1", "interrupted", error_code="worker.crash"
        )
        self.assertEqual(interrupted["payload"]["status"], "interrupted")
        with self.assertRaisesRegex(ResearchGitError, "requires retry or recovery"):
            self.ledger.begin("act-recover-1", "experiment.run", {"seed": 7})
        resumed = self.ledger.begin(
            "act-recover-1", "experiment.run", {"seed": 7}, recovery=True
        )
        self.assertEqual(resumed["payload"]["attempt"], 1)
        self.assertEqual(resumed["payload"]["recovery_of"], "act-recover-1")
        self.assertEqual(len(self.ledger.history("act-recover-1")), 3)

    def test_input_change_cannot_reuse_action_id(self) -> None:
        self.ledger.begin("act-bound-1", "search", {"query": "one"})
        with self.assertRaisesRegex(ResearchGitError, "input_hash"):
            self.ledger.begin("act-bound-1", "search", {"query": "two"})

    def test_parent_action_and_cumulative_budget_are_bound(self) -> None:
        parent = self.ledger.begin("act-parent", "search", {"query": "one"})
        child = self.ledger.begin(
            "act-child", "video.inspect", {"query": "one"}, parent_action_id="act-parent"
        )
        self.assertIn(
            {"type": "depends_on", "target": parent["object_id"]}, child["relations"]
        )
        self.ledger.begin("act-budget", "video.search", {"query": "one"}, budget_seconds=5)
        self.ledger.finish("act-budget", "interrupted", consumed_seconds=2)
        resumed = self.ledger.begin(
            "act-budget", "video.search", {"query": "one"}, recovery=True
        )
        self.assertEqual(resumed["payload"]["cumulative_consumed_seconds"], 2.0)
        self.assertEqual(resumed["payload"]["remaining_seconds"], 3.0)
        with self.assertRaisesRegex(ResearchGitError, "budget exceeded"):
            self.ledger.finish("act-budget", "completed", {"ok": True}, consumed_seconds=4)


if __name__ == "__main__":
    unittest.main()
