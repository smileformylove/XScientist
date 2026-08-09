from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from xscientist.research_cli import main as research_main
from xscientist.research_closure import audit_research_closure
from xscientist.research_dag import build_research_dag
from xscientist.research_vcs import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required for epistemic DAG tests")
class ResearchEpistemicJourneyTests(unittest.TestCase):
    def _run_json(self, argv: list[str]) -> dict:
        output = StringIO()
        with redirect_stdout(output):
            code = research_main([*argv, "--json"])
        self.assertEqual(code, 0, output.getvalue())
        return json.loads(output.getvalue())

    def test_plain_language_effect_and_inference_form_a_traceable_argument(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "study"
            ResearchRepository.init(
                repo,
                name="epistemic-study",
                git_user_name="XScientist Tests",
                git_user_email="tests@example.invalid",
            )
            common = ["--repo", str(repo)]
            hypothesis = self._run_json(
                [
                    "hypothesis",
                    "Treatment X improves outcome Y.",
                    "--falsifier",
                    "The estimated effect is non-positive.",
                    *common,
                ]
            )["object"]
            plan = self._run_json(
                [
                    "plan",
                    hypothesis["object_id"],
                    "Estimate the treatment effect.",
                    *common,
                ]
            )["object"]
            attempt = self._run_json(
                [
                    "experiment",
                    "Run the prespecified comparison.",
                    "--status",
                    "completed",
                    "--plan",
                    plan["object_id"],
                    *common,
                ]
            )["object"]
            evidence = self._run_json(
                [
                    "evidence",
                    "Treatment X improved the measured outcome.",
                    "--attempt",
                    attempt["object_id"],
                    *common,
                ]
            )["object"]
            estimand = self._run_json(
                [
                    "estimand",
                    "Outcome Y at 30 days",
                    "--population",
                    "eligible adults",
                    "--intervention",
                    "Treatment X",
                    "--comparator",
                    "control",
                    "--summary-measure",
                    "risk difference",
                    *common,
                ]
            )["object"]
            effect = self._run_json(
                [
                    "effect",
                    estimand["object_id"],
                    "0.25",
                    "--metric",
                    "risk_difference",
                    "--lower",
                    "0.10",
                    "--upper",
                    "0.40",
                    "--from",
                    evidence["object_id"],
                    *common,
                ]
            )["object"]
            inference = self._run_json(
                [
                    "infer",
                    "The estimated effect is positive for the target population.",
                    "--premise",
                    effect["object_id"],
                    "--warrant",
                    "The interval excludes a non-positive effect under the recorded design.",
                    *common,
                ]
            )["object"]
            claim = self._run_json(
                [
                    "claim",
                    "Treatment X improves outcome Y in eligible adults.",
                    "--evidence",
                    inference["object_id"],
                    "--population",
                    "eligible adults",
                    "--intervention",
                    "Treatment X",
                    "--outcome",
                    "Outcome Y at 30 days",
                    *common,
                ]
            )["object"]

            audit = audit_research_closure(repo, level="trace")
            self.assertTrue(audit["complete"], audit["blockers"])
            row = next(
                item
                for item in audit["claims"]
                if item["claim_id"] == claim["object_id"]
            )
            self.assertIn(inference["object_id"], row["argument_ids"])
            self.assertIn(effect["object_id"], row["argument_ids"])
            self.assertIn(evidence["object_id"], row["evidence_ids"])

            dag = build_research_dag(repo)
            self.assertIn(
                "argument",
                {
                    edge["category"]
                    for edge in dag["edges"]
                    if edge["source"] in row["argument_ids"]
                },
            )


if __name__ == "__main__":
    unittest.main()
