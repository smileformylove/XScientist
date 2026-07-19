from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ai_scientist.utils.gate_preconditions import (
    GatePreconditionContext,
    UnsatisfiableGateError,
    assert_gate_preconditions_satisfiable,
    evaluate_gate_preconditions,
)
from ai_scientist.utils.decision_log import load_decision_log
from ai_scientist.utils.research_planning import build_idea_cards, build_research_plan
from ai_scientist.apps.batch import _process_single_paper
from ai_scientist.apps.project import process_single_idea


class GatePreconditionTests(unittest.TestCase):
    def _plan(self, workflow_mode: str = "multi_agent_board") -> dict:
        idea_card = build_idea_cards(
            [
                {
                    "Name": "Gate Preconditions",
                    "Short Hypothesis": "Hard gate conflicts should fail fast.",
                    "Experiments": [
                        "Compare against baseline: base on dataset: demo with metric: accuracy."
                    ],
                }
            ],
            target_venue="neurips",
            workflow_mode=workflow_mode,
        )[0]
        return build_research_plan(idea_card, target_venue="neurips")

    def test_satisfiable_when_contracts_and_sample_gate_are_clear(self) -> None:
        result = evaluate_gate_preconditions(
            GatePreconditionContext(
                research_plan=self._plan(),
                sample_gate={"full_generation_allowed": True},
                improvement_rounds=1,
                require_quality_gate=True,
                high_quality_mode=True,
                target_venue="neurips",
            )
        )

        self.assertTrue(result["satisfiable"])
        self.assertIsNone(result["pause_reason"])

    def test_blocks_quality_gate_without_high_quality_mode(self) -> None:
        result = evaluate_gate_preconditions(
            GatePreconditionContext(
                research_plan=self._plan("classic_pipeline"),
                sample_gate={"full_generation_allowed": True},
                improvement_rounds=1,
                require_quality_gate=True,
                high_quality_mode=False,
            )
        )

        self.assertFalse(result["satisfiable"])
        self.assertIn("quality_gate_requires_high_quality_mode", result["reasons"])

    def test_blocks_sample_gate_and_raises(self) -> None:
        context = GatePreconditionContext(
            research_plan=self._plan(),
            sample_gate={
                "full_generation_allowed": False,
                "result": {"reasons": ["missing_sample_record"]},
            },
            improvement_rounds=1,
            high_quality_mode=True,
        )

        result = evaluate_gate_preconditions(context)

        self.assertFalse(result["satisfiable"])
        self.assertEqual(result["pause_reason"], "unsatisfiable_gate")
        self.assertIn("sample_gate_blocked", result["reasons"])
        with self.assertRaises(UnsatisfiableGateError):
            assert_gate_preconditions_satisfiable(context)

    def test_blocks_round_budget_below_venue_minimum(self) -> None:
        result = evaluate_gate_preconditions(
            GatePreconditionContext(
                research_plan=self._plan(),
                sample_gate={"full_generation_allowed": True},
                improvement_rounds=1,
                high_quality_mode=True,
                target_venue="nature",
            )
        )

        self.assertFalse(result["satisfiable"])
        self.assertIn("improvement_round_budget_below_gate_minimum", result["reasons"])

    def test_blocks_invalid_truth_contracts_for_high_risk_workflow(self) -> None:
        plan = self._plan()
        plan["truth_contract"] = {"categories": {}}
        plan["hallucination_checks"] = {"checks": []}

        result = evaluate_gate_preconditions(
            GatePreconditionContext(
                research_plan=plan,
                sample_gate={"full_generation_allowed": True},
                improvement_rounds=1,
                high_quality_mode=True,
            )
        )

        self.assertFalse(result["satisfiable"])
        self.assertIn("truth_contract_invalid", result["reasons"])
        self.assertIn("hallucination_checks_invalid", result["reasons"])

    @mock.patch("ai_scientist.apps.project.perform_experiments_bfts")
    def test_process_single_idea_should_fail_fast_before_bfts_on_static_gate_conflict(
        self,
        perform_bfts_mock: mock.Mock,
    ) -> None:
        with TemporaryDirectory() as td:
            project_dir = Path(td) / "project"
            project_dir.mkdir()
            result = process_single_idea(
                (
                    str(project_dir),
                    str(project_dir),
                    1,
                    {
                        "Name": "Static Gate Conflict",
                        "Short Hypothesis": "A quality gate without high-quality mode is invalid.",
                        "Experiments": ["Run a small baseline comparison."],
                    },
                    None,
                    "model-writeup",
                    "model-citation",
                    "model-review",
                    "model-plots",
                    "model-small",
                    1,
                    1,
                    "normal",
                    1,
                    0,
                    1,
                    0,
                    0.0,
                    "depth",
                    False,
                    "publishable",
                    "model-quality",
                    "neurips",
                    8.0,
                    8.0,
                    0,
                    0,
                    True,
                    "P1",
                    0,
                    "default",
                    0,
                    False,
                    0,
                    "classic_pipeline",
                    "open_ended",
                    "adaptive",
                    False,
                    "adaptive",
                )
            )
            exp_roots = list((project_dir / "02_experiments").iterdir())
            decisions = load_decision_log(exp_roots[0])

        perform_bfts_mock.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["pause_reason"], "unsatisfiable_gate")
        self.assertIn("quality_gate_requires_high_quality_mode", result["gate_reasons"])
        self.assertEqual(decisions[0]["category"], "workflow_strategy")
        self.assertEqual(decisions[0]["selected"], "classic_pipeline")
        self.assertEqual(
            [item["option"] for item in decisions[0]["options_considered"]],
            ["adaptive", "classic_pipeline"],
        )
        self.assertEqual(decisions[0]["metadata"]["requested_workflow_mode"], "adaptive")
        self.assertEqual(decisions[0]["metadata"]["resolved_workflow_mode"], "classic_pipeline")

    @mock.patch("ai_scientist.apps.project.gather_citations")
    @mock.patch("ai_scientist.apps.project.write_experiment_report", side_effect=RuntimeError("no report"))
    @mock.patch("ai_scientist.apps.project.perform_experiments_bfts")
    @mock.patch("ai_scientist.apps.project.edit_bfts_config_file", return_value="/tmp/demo_config.yaml")
    @mock.patch("ai_scientist.apps.project.idea_to_markdown")
    def test_process_single_idea_should_log_sample_gate_block_decision(
        self,
        _idea_to_markdown_mock: mock.Mock,
        _edit_config_mock: mock.Mock,
        perform_bfts_mock: mock.Mock,
        _write_report_mock: mock.Mock,
        gather_citations_mock: mock.Mock,
    ) -> None:
        with TemporaryDirectory() as td:
            project_dir = Path(td) / "project"
            project_dir.mkdir()
            result = process_single_idea(
                (
                    str(project_dir),
                    str(project_dir),
                    1,
                    {
                        "Name": "Sample Gate Block",
                        "Short Hypothesis": "A missing sample report should block writeup.",
                        "Experiments": ["Run a small baseline comparison."],
                    },
                    None,
                    "model-writeup",
                    "model-citation",
                    "model-review",
                    "model-plots",
                    "model-small",
                    1,
                    1,
                    "normal",
                    1,
                    0,
                    1,
                    0,
                    0.0,
                    "depth",
                    False,
                    "publishable",
                    "model-quality",
                    "neurips",
                    8.0,
                    8.0,
                    0,
                    0,
                    False,
                    "P1",
                    0,
                    "default",
                    0,
                    False,
                    0,
                    "classic_pipeline",
                    "open_ended",
                    "adaptive",
                    False,
                    "adaptive",
                )
            )
            exp_roots = list((project_dir / "02_experiments").iterdir())
            decisions = load_decision_log(exp_roots[0])

        perform_bfts_mock.assert_called_once()
        gather_citations_mock.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["pause_reason"], "unsatisfiable_gate")
        self.assertIn("sample_gate_blocked", result["gate_reasons"])
        sample_decisions = [
            decision
            for decision in decisions
            if decision["category"] == "sample_gate_full_generation"
        ]
        self.assertEqual(sample_decisions[-1]["selected"], "block_full_generation")
        rejected_continue = next(
            item
            for item in sample_decisions[-1]["options_considered"]
            if item["option"] == "continue_full_generation"
        )
        self.assertEqual(
            rejected_continue["rejected_because"],
            "sample gate did not pass, so full generation remains blocked",
        )

    @mock.patch("ai_scientist.apps.project.gather_citations")
    @mock.patch("ai_scientist.apps.project.write_experiment_report")
    @mock.patch("ai_scientist.apps.project.perform_experiments_bfts")
    @mock.patch(
        "ai_scientist.apps.project.edit_bfts_config_file", return_value="/tmp/demo_config.yaml"
    )
    @mock.patch("ai_scientist.apps.project.idea_to_markdown")
    def test_process_single_idea_should_stop_when_experiment_is_locked(
        self,
        _idea_to_markdown_mock: mock.Mock,
        _edit_config_mock: mock.Mock,
        perform_bfts_mock: mock.Mock,
        write_report_mock: mock.Mock,
        gather_citations_mock: mock.Mock,
    ) -> None:
        perform_bfts_mock.return_value = {
            "status": "locked",
            "resumable": False,
            "lock_path": "/tmp/idea/.xscientist-experiment.lock",
            "lock_owner": {"pid": 123},
            "failure_error": {
                "type": "ExperimentRunLocked",
                "message": "Experiment directory is already locked",
            },
        }
        with TemporaryDirectory() as td:
            project_dir = Path(td) / "project"
            project_dir.mkdir()
            result = process_single_idea(
                (
                    str(project_dir), str(project_dir), 1,
                    {"Name": "Locked", "Experiments": ["Run a baseline."]},
                    None, "model-writeup", "model-citation", "model-review",
                    "model-plots", "model-small", 1, 1, "normal", 1, 0, 1,
                    0, 0.0, "depth", False, "publishable", "model-quality",
                    "neurips", 8.0, 8.0, 0, 0, False, "P1", 0, "default",
                    0, False, 0, "classic_pipeline", "open_ended", "adaptive",
                    False, "adaptive",
                )
            )

        self.assertEqual(result["status"], "locked")
        self.assertEqual(result["stage"], "experiment")
        self.assertEqual(result["lock_owner"]["pid"], 123)
        write_report_mock.assert_not_called()
        gather_citations_mock.assert_not_called()

    @mock.patch("ai_scientist.apps.batch.perform_experiments_bfts", create=True)
    def test_process_single_paper_should_fail_fast_before_bfts_on_static_gate_conflict(
        self,
        perform_bfts_mock: mock.Mock,
    ) -> None:
        with TemporaryDirectory() as td:
            batch_dir = Path(td) / "batch"
            research_dir = Path(td) / "research"
            batch_dir.mkdir()
            research_dir.mkdir()
            result = _process_single_paper(
                (
                    str(batch_dir),
                    str(research_dir),
                    1,
                    {
                        "Name": "Static Gate Conflict",
                        "Short Hypothesis": "A quality gate without high-quality mode is invalid.",
                        "Experiments": ["Run a small baseline comparison."],
                    },
                    "normal",
                    None,
                    "model-writeup",
                    "model-citation",
                    "model-review",
                    "model-plots",
                    "model-small",
                    1,
                    1,
                    1,
                    False,
                    "publishable",
                    "model-quality",
                    "neurips",
                    8.0,
                    8.0,
                    0,
                    0,
                    True,
                    "P1",
                    0,
                    0,
                    1,
                    0,
                    0.0,
                    "depth",
                    "default",
                    0,
                    False,
                    0,
                    "classic_pipeline",
                    False,
                    False,
                    "adaptive",
                )
            )
            paper_roots = list((research_dir / "papers").iterdir())
            decisions = load_decision_log(paper_roots[0])

        perform_bfts_mock.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["pause_reason"], "unsatisfiable_gate")
        self.assertIn("quality_gate_requires_high_quality_mode", result["gate_reasons"])
        self.assertEqual(decisions[0]["category"], "workflow_strategy")
        self.assertEqual(decisions[0]["selected"], "classic_pipeline")
        self.assertEqual(
            [item["option"] for item in decisions[0]["options_considered"]],
            ["adaptive", "classic_pipeline"],
        )
        self.assertEqual(decisions[0]["metadata"]["requested_workflow_mode"], "adaptive")
        self.assertEqual(decisions[0]["metadata"]["resolved_workflow_mode"], "classic_pipeline")

    @mock.patch("ai_scientist.apps.batch.gather_citations", create=True)
    @mock.patch("ai_scientist.apps.batch.write_experiment_report", side_effect=RuntimeError("no report"))
    @mock.patch("ai_scientist.apps.batch.perform_experiments_bfts", create=True)
    @mock.patch("ai_scientist.apps.batch.edit_bfts_config_file", create=True, return_value="/tmp/demo_config.yaml")
    @mock.patch("ai_scientist.apps.batch.idea_to_markdown", create=True)
    def test_process_single_paper_should_log_sample_gate_block_decision(
        self,
        _idea_to_markdown_mock: mock.Mock,
        _edit_config_mock: mock.Mock,
        perform_bfts_mock: mock.Mock,
        _write_report_mock: mock.Mock,
        gather_citations_mock: mock.Mock,
    ) -> None:
        with TemporaryDirectory() as td:
            batch_dir = Path(td) / "batch"
            research_dir = Path(td) / "research"
            batch_dir.mkdir()
            research_dir.mkdir()
            result = _process_single_paper(
                (
                    str(batch_dir),
                    str(research_dir),
                    1,
                    {
                        "Name": "Sample Gate Block",
                        "Short Hypothesis": "A missing sample report should block writeup.",
                        "Experiments": ["Run a small baseline comparison."],
                    },
                    "normal",
                    None,
                    "model-writeup",
                    "model-citation",
                    "model-review",
                    "model-plots",
                    "model-small",
                    1,
                    1,
                    1,
                    False,
                    "publishable",
                    "model-quality",
                    "neurips",
                    8.0,
                    8.0,
                    0,
                    0,
                    False,
                    "P1",
                    0,
                    0,
                    1,
                    0,
                    0.0,
                    "depth",
                    "default",
                    0,
                    False,
                    0,
                    "classic_pipeline",
                    False,
                    False,
                    "adaptive",
                )
            )
            paper_roots = list((research_dir / "papers").iterdir())
            decisions = load_decision_log(paper_roots[0])

        perform_bfts_mock.assert_called_once()
        gather_citations_mock.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["pause_reason"], "unsatisfiable_gate")
        self.assertIn("sample_gate_blocked", result["gate_reasons"])
        sample_decisions = [
            decision
            for decision in decisions
            if decision["category"] == "sample_gate_full_generation"
        ]
        self.assertEqual(sample_decisions[-1]["selected"], "block_full_generation")
        rejected_continue = next(
            item
            for item in sample_decisions[-1]["options_considered"]
            if item["option"] == "continue_full_generation"
        )
        self.assertEqual(
            rejected_continue["rejected_because"],
            "sample gate did not pass, so full generation remains blocked",
        )

    @mock.patch("ai_scientist.apps.batch.gather_citations", create=True)
    @mock.patch("ai_scientist.apps.batch.write_experiment_report")
    @mock.patch("ai_scientist.apps.batch.perform_experiments_bfts", create=True)
    @mock.patch(
        "ai_scientist.apps.batch.edit_bfts_config_file",
        create=True,
        return_value="/tmp/demo_config.yaml",
    )
    @mock.patch("ai_scientist.apps.batch.idea_to_markdown", create=True)
    def test_process_single_paper_should_stop_when_experiment_is_locked(
        self,
        _idea_to_markdown_mock: mock.Mock,
        _edit_config_mock: mock.Mock,
        perform_bfts_mock: mock.Mock,
        write_report_mock: mock.Mock,
        gather_citations_mock: mock.Mock,
    ) -> None:
        perform_bfts_mock.return_value = {
            "status": "locked",
            "resumable": False,
            "lock_path": "/tmp/paper/.xscientist-experiment.lock",
            "lock_owner": {"pid": 456},
            "failure_error": {
                "type": "ExperimentRunLocked",
                "message": "Experiment directory is already locked",
            },
        }
        with TemporaryDirectory() as td:
            batch_dir = Path(td) / "batch"
            research_dir = Path(td) / "research"
            batch_dir.mkdir()
            research_dir.mkdir()
            result = _process_single_paper(
                (
                    str(batch_dir), str(research_dir), 1,
                    {"Name": "Locked", "Experiments": ["Run a baseline."]},
                    "normal", None, "model-writeup", "model-citation",
                    "model-review", "model-plots", "model-small", 1, 1, 1,
                    False, "publishable", "model-quality", "neurips", 8.0,
                    8.0, 0, 0, False, "P1", 0, 0, 1, 0, 0.0, "depth",
                    "default", 0, False, 0, "classic_pipeline", False, False,
                    "adaptive",
                )
            )

        self.assertEqual(result["status"], "locked")
        self.assertEqual(result["stage"], "experiment")
        self.assertEqual(result["lock_owner"]["pid"], 456)
        write_report_mock.assert_not_called()
        gather_citations_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
