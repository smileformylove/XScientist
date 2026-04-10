from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.workflow_selection import select_ranked_idea_candidates


class WorkflowSelectionTests(unittest.TestCase):
    def test_select_ranked_idea_candidates_should_penalize_fallback_rankings(
        self,
    ) -> None:
        ideas = [
            {"Name": "Fallback idea"},
            {"Name": "Clean idea"},
        ]

        def fake_ranker(*args, **kwargs):
            return [
                {
                    "idea_idx": 0,
                    "idea_name": "Fallback idea",
                    "ranking_score": 4.9,
                    "total_score": 4.9,
                    "fallback_used": True,
                    "fallback_stage": "client_creation",
                    "fallback_reason": "client_creation_failed",
                },
                {
                    "idea_idx": 1,
                    "idea_name": "Clean idea",
                    "ranking_score": 4.6,
                    "total_score": 4.6,
                    "fallback_used": False,
                },
            ]

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "rankings.json"
            selected, rankings = select_ranked_idea_candidates(
                ideas,
                ranking_enabled=True,
                ranking_model="demo-model",
                ranking_output_path=output_path,
                ranker=fake_ranker,
            )

            stored = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(selected, [1])
        self.assertEqual(rankings[0]["idea_idx"], 1)
        self.assertGreater(rankings[0]["selection_score"], rankings[1]["selection_score"])
        self.assertEqual(stored[0]["idea_idx"], 1)
        self.assertIn("selection_score", stored[0])
        self.assertGreater(stored[1]["selection_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
