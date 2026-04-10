from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.pipeline_contracts import initialize_pipeline_contracts, save_contract_artifact
from ai_scientist.utils.review_repair_planner import build_repair_plan, save_repair_plan


class ReviewRepairPlannerTests(unittest.TestCase):
    def test_build_repair_plan_should_group_tasks_into_agentic_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "repair_demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "schema_version": 1,
                    "rounds": [{"job_id": "rigor_0", "role": "rigor"}],
                    "repair_queue": [
                        {
                            "repair_id": "RPR-1",
                            "issue_id": "RVW-1",
                            "issue_text": "The main figure still needs a clearer baseline comparison.",
                            "role": "rigor",
                            "severity": "major",
                            "status": "ready",
                            "priority_tier": "p0",
                            "priority_score": 48,
                            "primary_target_type": "figure",
                            "primary_target_id": "figure_0",
                            "primary_target_label": "Main Result Figure",
                            "figure_ids": ["figure_0"],
                            "repair_actions": ["Revise figure_0 caption and panel notes."],
                            "verification_checks": ["Verify figure_0 now supports the main claim."],
                        },
                        {
                            "repair_id": "RPR-2",
                            "issue_id": "RVW-2",
                            "issue_text": "Ablation evidence is still too thin for the core robustness claim.",
                            "role": "rigor",
                            "severity": "major",
                            "status": "ready",
                            "priority_tier": "p1",
                            "priority_score": 36,
                            "primary_target_type": "claim",
                            "primary_target_id": "claim_0",
                            "primary_target_label": "Robustness Claim",
                            "claim_ids": ["claim_0"],
                            "repair_actions": ["Run one stronger ablation and add the result."],
                            "verification_checks": ["Verify the new ablation closes the reviewer concern."],
                        },
                    ],
                },
                producer="test_review_repair_planner",
            )

            repair_plan = build_repair_plan(project_root)

            self.assertEqual(repair_plan["summary"]["task_count"], 2)
            self.assertEqual(repair_plan["summary"]["ready_task_count"], 2)
            self.assertGreaterEqual(repair_plan["summary"]["lane_count"], 2)
            lanes = {task["issue_id"]: task["lane"] for task in repair_plan["tasks"]}
            self.assertEqual(lanes["RVW-1"], "figure_repair")
            self.assertEqual(lanes["RVW-2"], "evidence_followup")
            self.assertTrue(repair_plan["tasks"][0]["success_criteria"])
            self.assertEqual(repair_plan["tasks"][0]["verifier"], "figure_alignment_check")
            self.assertEqual(repair_plan["tasks"][1]["verifier"], "experiment_validation")
            self.assertTrue(repair_plan["tasks"][0]["required_inputs"])
            self.assertTrue(repair_plan["tasks"][1]["produced_artifacts"])
            self.assertGreaterEqual(
                repair_plan["summary"]["executable_ready_rate"],
                1.0,
            )
            self.assertTrue(repair_plan["execution_board"])

    def test_save_repair_plan_should_persist_contract_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "repair_demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            output_path = save_repair_plan(
                project_root,
                review_state={
                    "schema_version": 1,
                    "rounds": [{"job_id": "clarity_0", "role": "clarity"}],
                    "repair_queue": [],
                },
                producer="test_review_repair_planner",
            )

            self.assertTrue(Path(output_path).exists())
            persisted = Path(output_path).read_text(encoding="utf-8")
            self.assertIn('"summary"', persisted)

    def test_build_repair_plan_should_support_generic_repair_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "repair_ablation_demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="multi_agent_board")
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "schema_version": 1,
                    "rounds": [{"job_id": "critic_0", "role": "rigor"}],
                    "repair_queue": [
                        {
                            "repair_id": "RPR-critic",
                            "issue_id": "RVW-critic",
                            "issue_text": "The evidence is too weak for the main robustness claim.",
                            "role": "rigor",
                            "severity": "major",
                            "status": "ready",
                            "priority_tier": "p0",
                            "priority_score": 52,
                            "primary_target_type": "claim",
                            "primary_target_id": "claim_0",
                            "primary_target_label": "Main Robustness Claim",
                            "claim_ids": ["claim_0"],
                            "review_lane": "hostile_critic",
                            "repair_actions": ["Tighten or repair the affected claim."],
                            "verification_checks": ["Verify the revised claim no longer overstates support."],
                        }
                    ],
                },
                producer="test_review_repair_planner",
            )

            with patch.dict(os.environ, {"AI_SCIENTIST_DISABLE_OWNER_AWARE_REPAIR": "1"}):
                repair_plan = build_repair_plan(project_root)

            self.assertTrue(repair_plan["owner_aware_routing_disabled"])
            self.assertEqual(repair_plan["summary"]["task_count"], 1)
            self.assertEqual(repair_plan["tasks"][0]["lane"], "generic_rewrite")
            self.assertEqual(repair_plan["tasks"][0]["owner"], "repair_agent")
            self.assertEqual(repair_plan["tasks"][0]["verifier"], "hostile_critic_recheck")


if __name__ == "__main__":
    unittest.main()
