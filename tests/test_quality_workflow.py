from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.high_quality_pipeline import (
    evaluate_submission_acceptance,
    should_resume_high_quality_result,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.quality_workflow import (
    derive_autonomous_followup_focus,
    evaluate_final_submission_readiness,
    execute_quality_workflow,
)


class QualityWorkflowTests(unittest.TestCase):
    def test_evaluate_submission_acceptance_should_reject_strict_quality_fallback(
        self,
    ) -> None:
        acceptance = evaluate_submission_acceptance(
            {
                "quality_gate_passed": True,
                "submission_priority_score": 88.0,
                "blocker_count": 0,
                "auto_improvement_fallback_used": True,
            },
            require_quality_gate=True,
            min_submission_priority=80.0,
            max_submission_blockers=1,
            reject_on_auto_improvement_fallback=True,
        )

        self.assertFalse(acceptance["accepted"])
        self.assertIn("strict submission discipline", " ".join(acceptance["reasons"]))

    def test_execute_quality_workflow_should_pass_fallback_policy_to_runner(self) -> None:
        seen_kwargs = {}

        def fake_run_high_quality_pass(**kwargs):
            seen_kwargs.update(kwargs)
            return {
                "status": "success",
                "target_venue": "nature",
                "quality_gate_passed": True,
                "submission_priority_score": 91.0,
                "submission_priority_tier": "high",
                "blocker_count": 0,
                "auto_improvement_fallback_used": True,
            }

        result = execute_quality_workflow(
            run_high_quality_pass_fn=fake_run_high_quality_pass,
            run_dir="/tmp/demo",
            paper_type="normal",
            rewrite_model="model-writeup",
            quality_model="model-quality",
            target_venue="nature",
            quality_preset="publishable",
            quality_threshold=4.4,
            rigor_threshold=4.0,
            max_quality_rewrites=2,
            require_quality_gate=True,
            min_submission_priority=85.0,
            max_submission_blockers=0,
            allow_auto_improvement_fallback=False,
            reject_on_auto_improvement_fallback=True,
            resume=False,
            logger=lambda _msg: None,
        )

        self.assertFalse(seen_kwargs["auto_improvement_fallback"])
        self.assertFalse(result["acceptance"]["accepted"])
        self.assertIn(
            "strict submission discipline",
            " ".join(result["acceptance"]["reasons"]),
        )

    def test_should_resume_high_quality_result_should_require_matching_fallback_discipline(
        self,
    ) -> None:
        reusable = should_resume_high_quality_result(
            {
                "status": "success",
                "target_venue": "nature",
                "quality_threshold": 4.5,
                "rigor_threshold": 4.1,
                "auto_improvement_fallback_enabled": False,
            },
            auto_improvement_fallback=False,
            target_venue="nature",
            quality_threshold=4.5,
            rigor_threshold=4.1,
        )
        mismatched = should_resume_high_quality_result(
            {
                "status": "success",
                "target_venue": "nature",
                "quality_threshold": 4.5,
                "rigor_threshold": 4.1,
                "auto_improvement_fallback_enabled": True,
            },
            auto_improvement_fallback=False,
            target_venue="nature",
            quality_threshold=4.5,
            rigor_threshold=4.1,
        )

        self.assertTrue(reusable)
        self.assertFalse(mismatched)

    def test_evaluate_final_submission_readiness_should_reject_open_process_debt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "stage_standards",
                {
                    "overall_score": 76.0,
                    "blocked_stage_count": 1,
                    "summary": {"top_risks": ["review: repair_queue"]},
                },
                producer="test_quality_workflow",
            )
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "repair_metrics": {"active_issue_count": 1},
                    "lane_summaries": {
                        "hostile_critic": {"active_issue_count": 1, "blocking_issue_count": 1}
                    },
                },
                producer="test_quality_workflow",
            )
            save_contract_artifact(
                project_root,
                "repair_plan",
                {
                    "summary": {"verification_ready_rate": 0.5},
                    "tasks": [
                        {
                            "task_id": "repair_task_0",
                            "lane": "evidence_followup",
                            "status": "ready",
                            "priority_tier": "p0",
                            "priority_score": 42,
                            "owner": "experiment_agent",
                            "verifier": "experiment_validation",
                            "close_condition": "run a stronger ablation",
                            "escalation_lane": "hostile_critic",
                        }
                    ],
                },
                producer="test_quality_workflow",
            )
            save_contract_artifact(
                project_root,
                "self_evolution",
                {
                    "summary": {
                        "status": "blocked",
                        "score": 72.0,
                        "required_failure_count": 1,
                    },
                    "self_check": {"required_failures": ["stage_blockers"]},
                },
                producer="test_quality_workflow",
            )

            acceptance = evaluate_final_submission_readiness(
                run_dir=project_root,
                quality_result={
                    "quality_gate_passed": True,
                    "submission_priority_score": 91.0,
                    "submission_priority_tier": "high",
                    "blocker_count": 0,
                },
                require_quality_gate=True,
                min_submission_priority=85.0,
                max_submission_blockers=0,
                final_issue_progress={
                    "unresolved_critical_count": 1,
                    "persistent_issue_count": 2,
                },
                final_todo_snapshot={
                    "counts": {"total_tasks": 2, "p0_unresolved": 1}
                },
            )

            self.assertFalse(acceptance["accepted"])
            joined = " | ".join(acceptance["reasons"])
            self.assertIn("stage standards overall score below target", joined)
            self.assertIn("blocked stage standards remain", joined)
            self.assertIn("self-evolution score below target", joined)
            self.assertIn("final self-review still has unresolved critical issues", joined)
            self.assertIn("experiment TODO still has unresolved P0 items", joined)
            self.assertTrue(acceptance["signals"]["repair_lane_order"])
            self.assertTrue(acceptance["signals"]["hostile_recheck_required"])

    def test_evaluate_final_submission_readiness_should_accept_clean_submission_bundle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            quality_dir = project_root / "quality"
            quality_dir.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "stage_standards",
                {
                    "overall_score": 93.0,
                    "blocked_stage_count": 0,
                    "summary": {"top_risks": []},
                },
                producer="test_quality_workflow",
            )
            save_contract_artifact(
                project_root,
                "review_state",
                {"repair_metrics": {"active_issue_count": 0}},
                producer="test_quality_workflow",
            )
            save_contract_artifact(
                project_root,
                "repair_plan",
                {"summary": {"verification_ready_rate": 1.0}},
                producer="test_quality_workflow",
            )
            save_contract_artifact(
                project_root,
                "self_evolution",
                {
                    "summary": {
                        "status": "ready",
                        "score": 91.0,
                        "required_failure_count": 0,
                    },
                    "self_check": {"required_failures": []},
                },
                producer="test_quality_workflow",
            )
            (quality_dir / "high_quality_result.json").write_text(
                json.dumps(
                    {
                        "quality_gate_passed": True,
                        "submission_priority_score": 92.0,
                        "submission_priority_tier": "high",
                        "blocker_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            acceptance = evaluate_final_submission_readiness(
                run_dir=project_root,
                require_quality_gate=True,
                min_submission_priority=85.0,
                max_submission_blockers=0,
                final_issue_progress={
                    "unresolved_critical_count": 0,
                    "persistent_issue_count": 0,
                },
                final_todo_snapshot={
                    "counts": {"total_tasks": 1, "p0_unresolved": 0}
                },
            )

            self.assertTrue(acceptance["accepted"])
            self.assertEqual(acceptance["reasons"], [])

    def test_derive_autonomous_followup_focus_should_absorb_repair_lane_priorities(self) -> None:
        focus = derive_autonomous_followup_focus(
            quality_result={
                "quality_gate_passed": False,
                "blocker_count": 2,
                "revision_actions": [
                    {
                        "priority": "P0",
                        "focus": "claim support",
                        "action": "tighten the main claim",
                        "reason": "unsupported evidence path",
                    }
                ],
            },
            acceptance={"reasons": ["blocker remains"]},
            review_state={
                "lane_summaries": {
                    "hostile_critic": {"active_issue_count": 1},
                }
            },
            repair_plan={
                "tasks": [
                    {
                        "task_id": "repair_task_0",
                        "lane": "evidence_followup",
                        "status": "ready",
                        "priority_tier": "p0",
                        "priority_score": 50,
                        "owner": "experiment_agent",
                        "verifier": "experiment_validation",
                        "close_condition": "add a stronger ablation",
                        "escalation_lane": "hostile_critic",
                    }
                ]
            },
        )

        self.assertIn("claim_support", focus["focus_areas"])
        self.assertEqual(focus["repair_lane_order"], ["evidence_followup"])
        self.assertTrue(focus["hostile_recheck_required"])
        self.assertTrue(any("experiment validation" in note for note in focus["notes"]))


if __name__ == "__main__":
    unittest.main()
