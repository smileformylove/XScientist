from __future__ import annotations

import unittest
import json
from unittest import mock

from ai_scientist.utils.idea_ranking import (
    IDEA_SCORE_DIMENSIONS,
    score_idea_for_venue,
    score_idea_with_judges,
)

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
        self.assertEqual(result["total_score"], 0.0)
        self.assertFalse(result["ranking_eligible"])
        self.assertEqual(result["trust_tier"], "untrusted_fallback")

    @mock.patch(
        "ai_scientist.llm.get_response_from_llm", return_value=("not-json", None)
    )
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
        self.assertEqual(result["fallback_stage"], "response_validation")
        self.assertEqual(result["fallback_reason"], "response_invalid")
        self.assertFalse(result["ranking_eligible"])
        self.assertEqual(result["total_score"], 0.0)

    @mock.patch("ai_scientist.llm.create_client", return_value=(object(), "demo-model"))
    @mock.patch("ai_scientist.llm.get_response_from_llm")
    def test_score_idea_for_venue_should_validate_all_scientific_dimensions(
        self,
        mock_get_response: mock.Mock,
        _mock_create_client: mock.Mock,
    ) -> None:
        payload = {
            "novelty": 4.5,
            "feasibility": 3.5,
            "rigor_potential": 4.0,
            "falsifiability": 4.5,
            "information_gain": 4.0,
            "evidence_grounding": 3.5,
            "impact": 4.0,
            "writing_potential": 3.5,
            "breakthrough_potential": 4.0,
            "total_score": 5.0,
            "rationale": "Grounded and testable.",
        }
        mock_get_response.return_value = (json.dumps(payload), None)

        result = score_idea_for_venue(
            SAMPLE_IDEA,
            model="demo-model",
            target_venue="neurips",
        )

        self.assertTrue(result["ranking_eligible"])
        self.assertEqual(result["trust_tier"], "llm_judged")
        self.assertFalse(result["fallback_used"])
        self.assertNotEqual(result["total_score"], payload["total_score"])

    @mock.patch("ai_scientist.utils.idea_ranking.score_idea_for_venue")
    def test_multi_judge_consensus_uses_median_and_reports_disagreement(
        self, mock_score: mock.Mock
    ) -> None:
        low = {name: 3.0 for name in IDEA_SCORE_DIMENSIONS}
        high = {name: 5.0 for name in IDEA_SCORE_DIMENSIONS}
        low.update(
            {
                "total_score": 3.0,
                "ranking_eligible": True,
                "rationale": "conservative",
            }
        )
        high.update(
            {
                "total_score": 5.0,
                "ranking_eligible": True,
                "rationale": "optimistic",
            }
        )
        mock_score.side_effect = [low, high]

        result = score_idea_with_judges(
            SAMPLE_IDEA,
            models=["judge-a", "judge-b"],
            target_venue="neurips",
        )

        self.assertTrue(result["ranking_eligible"])
        self.assertEqual(result["trust_tier"], "multi_judge_consensus")
        self.assertEqual(result["novelty"], 4.0)
        self.assertEqual(result["judge_disagreement"], 2.0)


if __name__ == "__main__":
    unittest.main()
