from __future__ import annotations

import unittest

from ai_scientist.utils.hypothesis_archive import (
    build_hypothesis_archive,
    pareto_front,
    record_pairwise_comparison,
    select_quality_diverse,
)


def _card(
    idea_id: str,
    title: str,
    hypothesis: str,
    *,
    scores: dict[str, float],
    operator: str = "initial",
) -> dict:
    return {
        "idea_id": idea_id,
        "title": title,
        "core_hypothesis": hypothesis,
        "mechanism": hypothesis,
        "generation_operator": operator,
        "failure_criteria": ["The primary metric does not change."],
        "candidate_datasets": ["benchmark-v1"],
        "candidate_metrics": ["accuracy"],
        "candidate_baselines": ["baseline-a"],
        **scores,
    }


class HypothesisArchiveTests(unittest.TestCase):
    def test_archive_preserves_lineage_proximity_and_quality_diversity(self) -> None:
        strong = {
            "novelty": 4.0,
            "feasibility": 4.0,
            "falsifiability": 4.0,
            "information_gain": 4.0,
            "impact": 4.0,
            "evidence_grounding": 4.0,
            "safety": 4.0,
        }
        weak = {name: 2.0 for name in strong}
        tradeoff = dict(strong, novelty=5.0, feasibility=3.0)
        archive = build_hypothesis_archive(
            [
                _card(
                    "idea_0",
                    "Adaptive graph pruning for discovery",
                    "Adaptive graph pruning improves scientific discovery accuracy",
                    scores=strong,
                ),
                _card(
                    "idea_1",
                    "Adaptive graph pruning in discovery",
                    "Adaptive graph pruning improves discovery accuracy",
                    scores=weak,
                ),
                _card(
                    "idea_2",
                    "Contradiction-guided causal experiments",
                    "Contradictory papers identify high-information causal experiments",
                    scores=tradeoff,
                    operator="contradiction",
                ),
            ],
            proximity_threshold=0.45,
        )

        self.assertTrue(archive["append_only"])
        self.assertEqual(archive["summary"]["node_count"], 3)
        self.assertGreaterEqual(archive["summary"]["cluster_count"], 2)
        self.assertEqual(len(pareto_front(archive["nodes"])), 2)
        selected = select_quality_diverse(archive, limit=2)
        self.assertEqual(len(selected), 2)
        selected_clusters = {archive["clusters"][item] for item in selected}
        self.assertEqual(len(selected_clusters), 2)

    def test_pairwise_tournament_updates_ratings_without_mutating_input(self) -> None:
        scores = {
            "novelty": 4.0,
            "feasibility": 4.0,
            "falsifiability": 4.0,
            "information_gain": 4.0,
            "impact": 4.0,
            "evidence_grounding": 4.0,
            "safety": 4.0,
        }
        archive = build_hypothesis_archive(
            [
                _card("idea_0", "A", "Mechanism alpha", scores=scores),
                _card("idea_1", "B", "Mechanism beta", scores=scores),
            ]
        )
        left, right = [item["hypothesis_id"] for item in archive["nodes"]]

        updated = record_pairwise_comparison(
            archive,
            left_id=left,
            right_id=right,
            winner_id=left,
            judge_id="judge-a",
            rationale="More falsifiable.",
        )

        self.assertEqual(archive["ratings"][left], 1000.0)
        self.assertGreater(updated["ratings"][left], 1000.0)
        self.assertLess(updated["ratings"][right], 1000.0)
        self.assertEqual(updated["nodes"], archive["nodes"])
        self.assertEqual(len(updated["tournament_history"]), 1)


if __name__ == "__main__":
    unittest.main()
