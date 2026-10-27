from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from xscientist.research_commands import (
    save_experiment,
    save_hypothesis,
    save_research_plan,
)
from xscientist.research_git import ResearchGitError
from xscientist.research_journey import start_guided_research
from xscientist.research_vcs import ResearchRepository


class ExperimentTransitionTests(unittest.TestCase):
    def _plan(self, workspace: Path, suffix: str = "") -> str:
        if not workspace.exists():
            start_guided_research(
                workspace,
                question="Does the intervention change the measured outcome?",
                hypothesis="The intervention changes the measured outcome.",
                falsifier="The measured outcome is unchanged.",
            )
        hypothesis = save_hypothesis(
            str(workspace),
            statement=f"A measurable effect exists {suffix}".strip(),
            falsifier="The measured effect is absent.",
            commit=True,
        )
        plan = save_research_plan(
            str(workspace),
            hypothesis_id=hypothesis["object"].object_id,
            summary=f"Run a bounded test {suffix}".strip(),
            commit=True,
        )
        return plan["object"].object_id

    def test_first_experiment_has_no_transition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            plan_id = self._plan(workspace)
            result = save_experiment(
                str(workspace),
                summary="First run",
                status="completed",
                plan_id=plan_id,
                commit=True,
            )
            attempt = ResearchRepository(workspace).get(result["object"].object_id)
            self.assertNotIn("transition", attempt["payload"])

    def test_follow_up_requires_reused_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            plan_id = self._plan(workspace)
            first = save_experiment(
                str(workspace),
                summary="First run",
                status="completed",
                plan_id=plan_id,
                commit=True,
            )
            with self.assertRaisesRegex(
                ResearchGitError, "reused objects or artifact hashes"
            ):
                save_experiment(
                    str(workspace),
                    summary="Follow-up without reuse",
                    status="completed",
                    plan_id=plan_id,
                    continues_from=[first["object"].object_id],
                    transition_rationale="The next run should continue the first run.",
                    commit=True,
                )

    def test_follow_up_records_lineage_and_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            plan_id = self._plan(workspace)
            first = save_experiment(
                str(workspace),
                summary="First run",
                status="completed",
                plan_id=plan_id,
                commit=True,
            )
            first_id = first["object"].object_id
            second = save_experiment(
                str(workspace),
                summary="Follow-up using the first result",
                status="completed",
                plan_id=plan_id,
                continues_from=[first_id],
                reused_object_ids=[first_id],
                transition_rationale="Reuse the prior terminal result as the baseline.",
                commit=True,
            )
            repository = ResearchRepository(workspace)
            attempt = repository.get(second["object"].object_id)
            transition = attempt["payload"]["transition"]
            self.assertEqual(transition["continues_from"], [first_id])
            self.assertTrue(transition["transition_hash"].startswith("sha256:"))
            self.assertIn(
                {"type": "depends_on", "target": first_id, "role": "continues_from"},
                attempt["relations"],
            )
            self.assertIn(
                {"type": "consumes", "target": first_id, "role": "prior_result"},
                attempt["relations"],
            )

    def test_follow_up_rejects_cross_plan_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            first_plan = self._plan(workspace, "one")
            second_plan = self._plan(workspace, "two")
            first = save_experiment(
                str(workspace),
                summary="First plan run",
                status="completed",
                plan_id=first_plan,
                commit=True,
            )
            with self.assertRaisesRegex(ResearchGitError, "same plan/design"):
                save_experiment(
                    str(workspace),
                    summary="Cross-plan follow-up",
                    status="completed",
                    plan_id=second_plan,
                    continues_from=[first["object"].object_id],
                    reused_object_ids=[first["object"].object_id],
                    transition_rationale="Try to carry the result into another plan.",
                    commit=True,
                )

    def test_follow_up_can_reuse_prior_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            plan_id = self._plan(workspace)
            artifact_hash = "sha256:" + "a" * 64
            first = save_experiment(
                str(workspace),
                summary="First run with artifact",
                status="completed",
                plan_id=plan_id,
                result_artifact_hashes={"metrics": artifact_hash},
                commit=True,
            )
            second = save_experiment(
                str(workspace),
                summary="Follow-up reusing artifact",
                status="completed",
                plan_id=plan_id,
                continues_from=[first["object"].object_id],
                reused_artifact_hashes=[artifact_hash],
                transition_rationale="Reuse the prior metrics artifact for comparison.",
                commit=True,
            )
            repository = ResearchRepository(workspace)
            transition = repository.get(second["object"].object_id)["payload"][
                "transition"
            ]
            self.assertEqual(transition["reused_artifact_hashes"], [artifact_hash])


if __name__ == "__main__":
    unittest.main()
