from __future__ import annotations

import unittest
from unittest import mock

from ai_scientist.utils.idea_ranking import score_idea_for_venue


SAMPLE_IDEA = {
    "Name": "Adaptive Discovery",
    "Title": "A Novel Adaptive System for Real-World Scientific Discovery",
    "Short Hypothesis": (
        "This novel adaptive method addresses a fundamental cross-domain "
        "scientific discovery challenge with real-world impact, strong "
        "experiments, baseline comparisons, dataset transfer studies, and "
        "clear analysis."
    ),
    "Related Work": (
        "We compare against each baseline, include ablation analysis, study "
        "dataset transfer behavior, and evaluate medical and climate variants."
    ),
}


class IdeaRankingFallbackTests(unittest.TestCase):
    @mock.patch("ai_scientist.llm.create_client", side_effect=RuntimeError("boom"))
    def test_score_idea_for_venue_should_mark_client_creation_fallback(
        self,
        _mock_create_client: mock.Mock,
    ) -> None:
        result = score_idea_for_venue(
            SAMPLE_IDEA,
            model="demo-model",
            target_venue="nature",
        )

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_stage"], "client_creation")
        self.assertEqual(result["fallback_reason"], "client_creation_failed")
        self.assertIn("boom", result["fallback_detail"])
        self.assertGreater(result["total_score"], 3.0)

    @mock.patch("ai_scientist.llm.get_response_from_llm", return_value=("not-json", None))
    @mock.patch("ai_scientist.llm.create_client", return_value=(object(), "demo-model"))
    def test_score_idea_for_venue_should_use_heuristic_parse_fallback(
        self,
        _mock_create_client: mock.Mock,
        _mock_get_response: mock.Mock,
    ) -> None:
        result = score_idea_for_venue(
            SAMPLE_IDEA,
            model="demo-model",
            target_venue="neurips",
        )

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["fallback_stage"], "response_parsing")
        self.assertEqual(result["fallback_reason"], "response_parse_failed")
        self.assertEqual(
            result["rationale"],
            "heuristic fallback ranking due to parse failure",
        )
        self.assertGreater(result["total_score"], 3.0)


if __name__ == "__main__":
    unittest.main()
