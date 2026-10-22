from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from ai_scientist.protocol.research_vcs import RESEARCH_OBJECT_STATES
from xscientist import ResearchRepository

CHALLENGE_RELATIONS = (
    "refutes",
    "qualified_refutes",
    "contradicts",
    "challenges_inference",
    "invalidates",
)


@unittest.skipUnless(shutil.which("git"), "Git is required for challenge DAG tests")
class ResearchCurrentChallengeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = ResearchRepository.init(
            Path(temporary.name) / "research",
            name="current-challenge-fixture",
            question="Synthetic current versus historical challenge test",
            git_user_name="Challenge Test",
            git_user_email="challenge@example.invalid",
        )

    def _hypothesis(self, label: str) -> str:
        return self.repository.record(
            "hypothesis", {"statement": label, "falsifier": "Synthetic falsifier"}
        ).object_id

    def _evidence(self, label: str, relations: list[dict], state="completed") -> str:
        return self.repository.record(
            "evidence",
            {"result": label, "measurement_hash": "sha256:" + "a" * 64},
            state=state,
            relations=relations,
        ).object_id

    def _assert_status(self, graph: dict, hypothesis: str, contested: bool) -> None:
        node = next(node for node in graph["nodes"] if node["id"] == hypothesis)
        self.assertEqual(node["proof"]["contested"], contested)
        self.assertEqual(
            node["proof"]["level"], "contested" if contested else "traceable"
        )
        frontier = graph["theory_frontier"]
        self.assertEqual(hypothesis in frontier["contested_hypothesis_ids"], contested)
        self.assertEqual(
            hypothesis in frontier["open_theory_hypothesis_ids"], not contested
        )

    def test_only_accepted_successors_retire_current_challenges(self) -> None:
        cases = []
        for state in RESEARCH_OBJECT_STATES:
            hypothesis = self._hypothesis(state)
            inference = self.repository.record(
                "inference", {"statement": "Synthetic inference " + state}
            ).object_id
            relations = [
                {
                    "type": kind,
                    "target": (
                        inference if kind == "challenges_inference" else hypothesis
                    ),
                }
                for kind in CHALLENGE_RELATIONS
            ]
            negative = self._evidence(
                "Original " + state,
                relations,
            )
            replacement = self._evidence(
                "Replacement " + state,
                [{"type": "supersedes", "target": negative}],
                state=state,
            )
            cases.append((state, hypothesis, inference, negative, replacement))
        self.repository.commit(stage="review", subject="successor state controls")
        for disclose in (True, False):
            graph = self.repository.dag(disclose_summaries=disclose)
            for state, hypothesis, inference, negative, replacement in cases:
                with self.subTest(state=state, disclose=disclose):
                    contested = state not in {"completed", "verified", "promoted"}
                    self._assert_status(graph, hypothesis, contested)
                    inference_proof = next(
                        node["proof"]
                        for node in graph["nodes"]
                        if node["id"] == inference
                    )
                    self.assertEqual(inference_proof["contested"], contested)
                    self.assertEqual(inference_proof["level"] == "contested", contested)
                    self.assertNotEqual(inference_proof["level"], "verified")
                    self.assertTrue(
                        {negative, replacement}.issubset(
                            {node["id"] for node in graph["nodes"]}
                        )
                    )
                    self.assertEqual(
                        {
                            edge["type"]
                            for edge in graph["edges"]
                            if edge["target"] == negative
                        },
                        set(CHALLENGE_RELATIONS),
                    )

    def test_completed_replacement_preserves_historical_ref_and_edges(self) -> None:
        hypothesis = self._hypothesis("Historical ref control")
        negative = self._evidence(
            "Original", [{"type": "refutes", "target": hypothesis}]
        )
        self.repository.commit(stage="review", subject="original refutation")
        baseline = self.repository.dag()
        baseline_ref = next(
            node["source_ref"] for node in baseline["nodes"] if node["id"] == hypothesis
        )
        baseline = self.repository.dag(ref=baseline_ref)
        self._assert_status(baseline, hypothesis, True)
        draft = self._evidence(
            "Draft replacement", [{"type": "supersedes", "target": negative}], "draft"
        )
        self.repository.commit(stage="review", subject="draft replacement")
        self._assert_status(self.repository.dag(), hypothesis, True)
        self._evidence(
            "Completed replacement",
            [
                {"type": "supersedes", "target": negative},
                {"type": "supersedes", "target": draft},
            ],
        )
        self.repository.commit(stage="review", subject="completed replacement")
        graph = self.repository.dag()
        self._assert_status(graph, hypothesis, False)
        self.assertTrue(all(edge in graph["edges"] for edge in baseline["edges"]))
        replay = self.repository.dag(ref=baseline_ref)
        self.assertEqual(
            {key: value for key, value in replay.items() if key != "generated_at"},
            {key: value for key, value in baseline.items() if key != "generated_at"},
        )
        metadata = self.repository.dag(disclose_summaries=False)
        self._assert_status(metadata, hypothesis, False)

    def test_archived_sources_do_not_hide_other_active_challenges(self) -> None:
        archived_target = self._hypothesis("Explicitly archived source")
        self._evidence(
            "Archived refutation",
            [{"type": "refutes", "target": archived_target}],
            "superseded",
        )
        active_target = self._hypothesis("Replacement has its own refutation")
        original = self._evidence(
            "Old refutation", [{"type": "refutes", "target": active_target}]
        )
        replacement = self._evidence(
            "Corrected but still negative",
            [
                {"type": "supersedes", "target": original},
                {"type": "contradicts", "target": active_target},
            ],
        )
        self._evidence(
            "Unfinished follow-up",
            [{"type": "supersedes", "target": replacement}],
            "draft",
        )
        unrelated_target = self._hypothesis("Another independent challenge survives")
        retired = self._evidence(
            "First negative", [{"type": "refutes", "target": unrelated_target}]
        )
        self._evidence("Replacement", [{"type": "supersedes", "target": retired}])
        self._evidence(
            "Independent negative",
            [{"type": "invalidates", "target": unrelated_target}],
        )
        self.repository.commit(stage="review", subject="active and archived controls")
        for disclose in (True, False):
            graph = self.repository.dag(disclose_summaries=disclose)
            self._assert_status(graph, archived_target, False)
            self._assert_status(graph, active_target, True)
            self._assert_status(graph, unrelated_target, True)


if __name__ == "__main__":
    unittest.main()
