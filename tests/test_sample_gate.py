from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.experiment_registry import build_experiment_record, save_experiment_registry
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    load_pipeline_manifest,
)
from ai_scientist.utils.research_planning import build_idea_cards, build_research_plan
from ai_scientist.utils.sample_gate import (
    SampleGateBlocked,
    assert_sample_gate_allows_full_generation,
    build_sample_gate_plan,
    evaluate_and_save_sample_gate,
    evaluate_sample_gate,
    save_sample_gate_plan,
)


class SampleGateTests(unittest.TestCase):
    def _research_plan(self) -> dict:
        idea_card = build_idea_cards(
            [
                {
                    "Name": "Sample Gate Idea",
                    "Short Hypothesis": "A sample run catches failed plans early.",
                    "Experiments": [
                        "Compare against baseline: base on dataset: demo with metric: accuracy.",
                        "Run an ablation on the verifier.",
                    ],
                }
            ],
            target_venue="neurips",
            workflow_mode="multi_agent_board",
        )[0]
        return build_research_plan(idea_card, target_venue="neurips")

    def test_plan_should_select_first_task_and_block_full_generation_until_evaluated(self) -> None:
        plan = self._research_plan()
        gate = build_sample_gate_plan(plan)

        self.assertEqual(gate["sample_task_count"], 1)
        self.assertEqual(gate["sample_tasks"][0]["task_id"], "task_0")
        self.assertFalse(gate["full_generation_allowed"])
        self.assertEqual(gate["result"]["reasons"], ["sample_not_run"])

    def test_evaluate_sample_gate_should_pass_completed_within_budget_record(self) -> None:
        plan = self._research_plan()
        gate = build_sample_gate_plan(plan)
        record = build_experiment_record(
            task_id="task_0",
            dataset="demo",
            metric="accuracy",
            baseline_ref="base",
            status="completed",
            result_summary={"metric_name": "accuracy", "metric_mean": 0.8},
            acceptance_checks=["accuracy improves"],
            acceptance_results=[
                {"check": "accuracy improves", "passed": True, "source": "unit"}
            ],
            budget_audit={"audited": True, "within_budget": True, "source": "unit"},
            budget_status="within_budget",
        )

        evaluated = evaluate_sample_gate(gate, experiment_records=[record])

        self.assertTrue(evaluated["full_generation_allowed"])
        self.assertEqual(evaluated["status"], "passed")
        self.assertEqual(evaluated["result"]["reasons"], [])

    def test_evaluate_sample_gate_should_block_missing_or_budget_exhausted_sample(self) -> None:
        plan = self._research_plan()
        gate = build_sample_gate_plan(plan)
        missing = evaluate_sample_gate(gate, experiment_records=[])
        self.assertFalse(missing["full_generation_allowed"])
        self.assertIn("missing_sample_record", missing["result"]["reasons"])

        exhausted = build_experiment_record(
            task_id="task_0",
            dataset="demo",
            metric="accuracy",
            baseline_ref="base",
            status="completed",
            result_summary={"metric_name": "accuracy"},
            acceptance_checks=["accuracy improves"],
            acceptance_results=[
                {"check": "accuracy improves", "passed": True, "source": "unit"}
            ],
            budget_audit={"audited": True, "within_budget": False, "source": "unit"},
            budget_status="budget_exhausted",
        )
        failed = evaluate_sample_gate(gate, experiment_records=[exhausted])
        self.assertFalse(failed["full_generation_allowed"])
        self.assertIn("sample_budget:budget_exhausted", failed["result"]["reasons"])

    def test_evaluate_sample_gate_should_require_explicit_acceptance_and_budget_audit(self) -> None:
        plan = self._research_plan()
        gate = build_sample_gate_plan(plan)
        weak_record = build_experiment_record(
            task_id="task_0",
            dataset="demo",
            metric="accuracy",
            baseline_ref="base",
            status="completed",
            result_summary={"metric_name": "accuracy", "metric_mean": 0.8},
            acceptance_checks=["planned check only"],
            budget_status="within_budget",
        )

        evaluated = evaluate_sample_gate(gate, experiment_records=[weak_record])

        self.assertFalse(evaluated["full_generation_allowed"])
        self.assertIn("acceptance_results_not_passed", evaluated["result"]["reasons"])
        self.assertIn("missing_budget_audit", evaluated["result"]["reasons"])

    def test_save_and_evaluate_sample_gate_should_update_manifest_artifact(self) -> None:
        plan = self._research_plan()
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="multi_agent_board")
            gate = save_sample_gate_plan(
                project_root,
                research_plan=plan,
                producer="test_sample_gate",
            )
            save_experiment_registry(
                project_root,
                [
                    build_experiment_record(
                        task_id=gate["sample_tasks"][0]["task_id"],
                        dataset="demo",
                        metric="accuracy",
                        baseline_ref="base",
                        status="completed",
                        result_summary={"metric_name": "accuracy", "metric_mean": 0.8},
                        acceptance_checks=["accuracy improves"],
                        acceptance_results=[
                            {
                                "check": "accuracy improves",
                                "passed": True,
                                "source": "unit",
                            }
                        ],
                        budget_audit={
                            "audited": True,
                            "within_budget": True,
                            "source": "unit",
                        },
                        budget_status="within_budget",
                    )
                ],
            )
            evaluated = evaluate_and_save_sample_gate(
                project_root,
                producer="test_sample_gate",
            )

            self.assertTrue(evaluated["full_generation_allowed"])
            saved = load_contract_artifact(project_root, "sample_gate", default={})
            self.assertTrue(saved["result"]["passed"])
            manifest = load_pipeline_manifest(project_root)
            self.assertEqual(manifest["artifacts"]["sample_gate"]["status"], "ready")

    def test_blocked_sample_gate_should_mark_manifest_blocked_and_raise(self) -> None:
        plan = self._research_plan()
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="multi_agent_board")
            save_sample_gate_plan(
                project_root,
                research_plan=plan,
                producer="test_sample_gate",
            )
            evaluated = evaluate_and_save_sample_gate(
                project_root,
                producer="test_sample_gate",
            )

            self.assertFalse(evaluated["full_generation_allowed"])
            with self.assertRaises(SampleGateBlocked):
                assert_sample_gate_allows_full_generation(evaluated)
            manifest = load_pipeline_manifest(project_root)
            self.assertEqual(manifest["artifacts"]["sample_gate"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
