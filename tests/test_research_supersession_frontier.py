from __future__ import annotations

import unittest
from unittest import mock

from ai_scientist.protocol.research_vcs import RESEARCH_OBJECT_STATES
from xscientist.research_context import _effective_superseded_ids
from xscientist.research_dag import _strategy_projection
from xscientist.research_strategy import _effective_objects


class ResearchSupersessionFrontierTests(unittest.TestCase):
    def _objects(self, successor_state: str) -> dict[str, dict]:
        return {
            "hypothesis": {
                "object_id": "hypothesis",
                "kind": "hypothesis",
                "state": "draft",
                "payload": {"statement": "Test hypothesis"},
                "relations": [],
            },
            "negative": {
                "object_id": "negative",
                "kind": "evidence",
                "state": "completed",
                "payload": {"result": "Recorded negative evidence"},
                "relations": [{"type": "refutes", "target": "hypothesis"}],
            },
            "correction": {
                "object_id": "correction",
                "kind": "evidence",
                "state": successor_state,
                "payload": {"result": "Replacement evidence"},
                "relations": [{"type": "supersedes", "target": "negative"}],
            },
        }

    def _frontiers(self, objects: dict[str, dict]) -> tuple[dict, dict]:
        repository = mock.Mock()
        repository.objects.return_value = list(objects.values())
        effective = _effective_objects(repository)
        order = {object_id: index for index, object_id in enumerate(objects)}
        frontier, _insights, _summary = _strategy_projection(
            objects, order, disclose=True
        )
        return effective, frontier

    def test_only_completed_successors_retire_current_negative_evidence(self) -> None:
        accepted_states = {"completed", "verified", "promoted"}
        for state in RESEARCH_OBJECT_STATES:
            with self.subTest(successor_state=state):
                objects = self._objects(state)
                effective, frontier = self._frontiers(objects)
                replaced = state in accepted_states
                self.assertEqual("negative" not in effective, replaced)
                self.assertEqual(
                    "negative" in _effective_superseded_ids(objects), replaced
                )
                self.assertEqual(
                    frontier["contested_hypothesis_ids"],
                    [] if replaced else ["hypothesis"],
                )
                self.assertEqual(
                    frontier["open_theory_hypothesis_ids"],
                    ["hypothesis"] if replaced else [],
                )
                self.assertIn("negative", objects)
                self.assertIn("correction", objects)

    def test_draft_successor_does_not_undo_an_accepted_supersession_chain(self) -> None:
        objects = self._objects("completed")
        objects["correction"]["relations"].append(
            {"type": "refutes", "target": "hypothesis"}
        )
        objects["future"] = {
            "object_id": "future",
            "kind": "evidence",
            "state": "draft",
            "payload": {"result": "Unfinished follow-up"},
            "relations": [{"type": "supersedes", "target": "correction"}],
        }
        effective, frontier = self._frontiers(objects)
        self.assertNotIn("negative", effective)
        self.assertIn("correction", effective)
        self.assertEqual(frontier["contested_hypothesis_ids"], ["hypothesis"])
        self.assertEqual(_effective_superseded_ids(objects), {"negative"})

    def test_explicitly_superseded_records_remain_archived(self) -> None:
        objects = self._objects("draft")
        objects["negative"]["state"] = "superseded"
        effective, frontier = self._frontiers(objects)
        self.assertNotIn("negative", effective)
        self.assertEqual(frontier["contested_hypothesis_ids"], [])
        self.assertEqual(_effective_superseded_ids(objects), {"negative"})


if __name__ == "__main__":
    unittest.main()
