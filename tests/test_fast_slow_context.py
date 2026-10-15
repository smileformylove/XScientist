from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from xscientist import ResearchRepository
from xscientist.research_context import (
    build_research_context_snapshot,
    research_context_issues,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for context tests")
class FastSlowContextTests(unittest.TestCase):
    def test_localization_prioritizes_objective_relevant_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = ResearchRepository.init(
                Path(td) / "research",
                name="fast-slow",
                question="# Fast/slow routing\n",
                git_user_name="Fast Slow Test",
                git_user_email="fast-slow@example.invalid",
            )
            hypothesis = repository.record(
                "hypothesis",
                {
                    "statement": "Calibration improves transfer accuracy",
                    "falsifier": "transfer accuracy does not improve",
                },
            )
            relevant = repository.record(
                "evidence",
                {
                    "result": "Calibration improves transfer accuracy on held-out data.",
                    "measurement_hash": "sha256:" + "a" * 64,
                },
                state="completed",
                relations=[
                    {"type": "qualified_supports", "target": hypothesis.object_id}
                ],
            )
            distractor = repository.record(
                "evidence",
                {
                    "result": "The training throughput increased on the development set.",
                    "measurement_hash": "sha256:" + "b" * 64,
                },
                state="completed",
                relations=[
                    {"type": "qualified_supports", "target": hypothesis.object_id}
                ],
            )

            payload = build_research_context_snapshot(
                repository,
                target_ids=[hypothesis.object_id],
                evidence_objectives=["transfer accuracy"],
                budget_tokens=900,
            )

            self.assertEqual(
                payload["retrieval_receipt"]["request"]["evidence_objectives"],
                ["transfer accuracy"],
            )
            self.assertEqual(
                payload["retrieval_receipt"]["algorithm"]["routing_policy"],
                "xscientist.fast-slow-evidence.v1",
            )
            candidates = {
                item["object_id"]: item
                for item in payload["retrieval_receipt"]["candidate_set"]
            }
            self.assertIn(relevant.object_id, candidates)
            self.assertIn(distractor.object_id, candidates)
            self.assertGreater(
                candidates[relevant.object_id]["localization"]["score"],
                candidates[distractor.object_id]["localization"]["score"],
            )
            relevant_view = next(
                item
                for item in payload["source_views"]
                if item["object_id"] == relevant.object_id
            )
            self.assertIn("evidence_slice", relevant_view)
            self.assertIn("transfer accuracy", relevant_view["evidence_slice"])
            self.assertIn(relevant.object_id, payload["source_object_ids"])
            self.assertFalse(research_context_issues(payload))

    def test_repository_context_api_exposes_explicit_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = ResearchRepository.init(
                Path(td) / "research",
                name="fast-slow-api",
                question="# Fast/slow API\n",
                git_user_name="Fast Slow Test",
                git_user_email="fast-slow-api@example.invalid",
            )
            hypothesis = repository.record(
                "hypothesis",
                {"statement": "A signal transfers", "falsifier": "no transfer"},
            )
            context = repository.context(
                target_ids=[hypothesis.object_id],
                evidence_objectives=["held-out transfer signal"],
            )
            prompt = repository.context_prompt(
                target_ids=[hypothesis.object_id],
                evidence_objectives=["held-out transfer signal"],
            )
            self.assertEqual(
                context["evidence_objectives"], ["held-out transfer signal"]
            )
            self.assertIn("source_views", prompt)
            self.assertNotIn("source_objects", prompt)


if __name__ == "__main__":
    unittest.main()
