from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.repair_verifier import (
    maybe_run_repair_plan_verifiers,
    run_task_verifier,
    verify_enabled,
    verify_experiment_validation,
    verify_figure_alignment_check,
    verify_hostile_critic_recheck,
    verify_planner_triage_recheck,
    verify_reviewer_board_recheck,
)


def _save_review_state(project_root: Path, issues: list[dict]) -> None:
    save_contract_artifact(
        project_root,
        "review_state",
        {"issues_log": issues, "role_summaries": {}},
        producer="test",
    )


class VerifierBoardRecheckTests(unittest.TestCase):
    def test_pass_when_issue_absent_from_latest_round(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            _save_review_state(
                project_root,
                [
                    {"issue_id": "RVW-1", "round_index": 1},
                    {"issue_id": "RVW-OTHER", "round_index": 2},
                ],
            )
            passed, evidence = verify_reviewer_board_recheck(
                {"issue_id": "RVW-1"}, project_root
            )
            self.assertTrue(passed)
            self.assertFalse(evidence["still_present_in_latest_round"])

    def test_fail_when_issue_resurfaces(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            _save_review_state(
                project_root,
                [
                    {"issue_id": "RVW-1", "round_index": 2},
                ],
            )
            passed, evidence = verify_reviewer_board_recheck(
                {"issue_id": "RVW-1"}, project_root
            )
            self.assertFalse(passed)
            self.assertTrue(evidence["still_present_in_latest_round"])


class HostileCriticRecheckTests(unittest.TestCase):
    def test_pass_when_no_hostile_role_surfaces_issue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            _save_review_state(
                project_root,
                [{"issue_id": "RVW-1", "round_index": 1, "role": "reviewer_a"}],
            )
            passed, _ = verify_hostile_critic_recheck({"issue_id": "RVW-1"}, project_root)
            self.assertTrue(passed)

    def test_fail_when_hostile_role_re_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            _save_review_state(
                project_root,
                [
                    {
                        "issue_id": "RVW-1",
                        "round_index": 1,
                        "review_lane": "hostile_critic",
                    }
                ],
            )
            passed, evidence = verify_hostile_critic_recheck(
                {"issue_id": "RVW-1"}, project_root
            )
            self.assertFalse(passed)
            self.assertTrue(evidence["hostile_role_resurfaced"])


class FigureAlignmentTests(unittest.TestCase):
    def test_pass_when_no_figure_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            save_contract_artifact(project_root, "figure_spec", {"figures": []}, producer="t")
            passed, _ = verify_figure_alignment_check({"figure_ids": []}, project_root)
            self.assertTrue(passed)

    def test_fail_when_bound_figure_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            save_contract_artifact(
                project_root,
                "figure_spec",
                {"figures": [{"figure_id": "F1"}]},
                producer="t",
            )
            passed, evidence = verify_figure_alignment_check(
                {"figure_ids": ["F1", "F-missing"]}, project_root
            )
            self.assertFalse(passed)
            self.assertEqual(evidence["missing_in_spec"], ["F-missing"])


class ExperimentValidationTests(unittest.TestCase):
    def test_pass_when_claim_status_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            save_contract_artifact(
                project_root,
                "claim_evidence_graph",
                {
                    "nodes": [{"id": "C1", "type": "claim", "status": "supported"}],
                    "edges": [],
                },
                producer="t",
            )
            passed, _ = verify_experiment_validation({"claim_ids": ["C1"]}, project_root)
            self.assertTrue(passed)

    def test_fail_when_claim_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            save_contract_artifact(
                project_root,
                "claim_evidence_graph",
                {
                    "nodes": [{"id": "C1", "type": "claim", "status": "unsupported"}],
                    "edges": [],
                },
                producer="t",
            )
            passed, evidence = verify_experiment_validation(
                {"claim_ids": ["C1"]}, project_root
            )
            self.assertFalse(passed)
            self.assertEqual(evidence["unsupported"], ["C1"])

    def test_pass_via_supporting_edge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            save_contract_artifact(
                project_root,
                "claim_evidence_graph",
                {
                    "nodes": [{"id": "C1", "type": "claim", "status": "pending"}],
                    "edges": [
                        {"source": "E1", "target": "C1", "relation": "supports"}
                    ],
                },
                producer="t",
            )
            passed, _ = verify_experiment_validation({"claim_ids": ["C1"]}, project_root)
            self.assertTrue(passed)


class PlannerTriageRecheckTests(unittest.TestCase):
    def test_pass_with_full_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            passed, _ = verify_planner_triage_recheck(
                {
                    "issue_id": "RVW-1",
                    "primary_target_id": "C1",
                    "primary_target_type": "claim",
                    "owner": "reviewer_board",
                    "close_condition": "Close when bound claim supported.",
                },
                project_root,
            )
            self.assertTrue(passed)

    def test_fail_when_missing_owner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            passed, evidence = verify_planner_triage_recheck(
                {
                    "issue_id": "RVW-1",
                    "primary_target_id": "C1",
                    "primary_target_type": "claim",
                    "owner": "",
                    "close_condition": "Close when bound claim supported.",
                },
                project_root,
            )
            self.assertFalse(passed)
            self.assertFalse(evidence["has_owner"])


class DispatchTests(unittest.TestCase):
    def test_unknown_verifier_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            outcome = run_task_verifier(
                {"issue_id": "RVW-1", "verifier": "unknown_one"}, project_root
            )
            self.assertTrue(outcome["skipped"])
            self.assertFalse(outcome["passed"])

    def test_known_verifier_dispatched(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            _save_review_state(project_root, [{"issue_id": "RVW-OTHER", "round_index": 1}])
            outcome = run_task_verifier(
                {"issue_id": "RVW-1", "verifier": "reviewer_board_recheck"},
                project_root,
            )
            self.assertFalse(outcome["skipped"])
            self.assertTrue(outcome["passed"])


class MaybeRunRepairPlanVerifiersTests(unittest.TestCase):
    def test_returns_none_when_flag_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            os.environ.pop("AI_SCIENTIST_REPAIR_VERIFY", None)
            self.assertFalse(verify_enabled())
            self.assertIsNone(maybe_run_repair_plan_verifiers(project_root))

    def test_aggregates_results_when_flag_on(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            _save_review_state(
                project_root,
                [{"issue_id": "RVW-STILL", "round_index": 1}],
            )
            save_contract_artifact(
                project_root,
                "repair_plan",
                {
                    "tasks": [
                        {
                            "task_id": "T-1",
                            "issue_id": "RVW-CLEAR",
                            "verifier": "reviewer_board_recheck",
                        },
                        {
                            "task_id": "T-2",
                            "issue_id": "RVW-STILL",
                            "verifier": "reviewer_board_recheck",
                        },
                        {
                            "task_id": "T-3",
                            "issue_id": "RVW-UNK",
                            "verifier": "bogus",
                        },
                    ]
                },
                producer="test",
            )
            with patch.dict(os.environ, {"AI_SCIENTIST_REPAIR_VERIFY": "1"}):
                report = maybe_run_repair_plan_verifiers(project_root)
            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report["task_count"], 3)
            self.assertEqual(report["passed"], 1)
            self.assertEqual(report["skipped"], 1)
            self.assertEqual(report["failed"], 1)


if __name__ == "__main__":
    unittest.main()
