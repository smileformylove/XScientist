from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.experiment_registry import save_experiment_registry
from ai_scientist.utils.figure_spec import save_figure_spec
from ai_scientist.utils.manuscript_state import save_manuscript_state
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    save_contract_artifact,
)
from ai_scientist.utils.stage_standards import (
    build_stage_standards,
    save_stage_standards,
)


class StageStandardsTests(unittest.TestCase):
    def test_build_stage_standards_should_score_pipeline_with_explicit_checks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "idea_cards",
                [
                    {
                        "idea_id": "idea_0",
                        "name": "Idea Alpha",
                        "core_hypothesis": "Structured evaluation improves pipeline quality.",
                        "novelty_claim": "Adds explicit stage standards.",
                        "related_work_notes": "Program-style disciplined execution.",
                        "minimum_viable_experiment": "Compare with a baseline.",
                        "candidate_datasets": ["demo-ds"],
                        "candidate_metrics": ["accuracy"],
                        "candidate_baselines": ["baseline_a"],
                        "failure_criteria": ["No measurable improvement."],
                        "literature_queries": ["structured pipeline evaluation"],
                        "target_venue": "nature",
                    }
                ],
                producer="test_stage_standards",
            )
            save_contract_artifact(
                project_root,
                "research_plan",
                {
                    "workflow_mode": "program_driven",
                    "budget": {
                        "max_steps": 8,
                        "max_wallclock_minutes": 75,
                        "max_retry_per_task": 1,
                    },
                    "execution_policy": {"policy_name": "program_driven"},
                    "tasks": [
                        {
                            "task_id": "task_0",
                            "success_criterion": "Improve accuracy.",
                            "stop_condition": "Budget exhausted or claim resolved.",
                            "claim_targets": ["claim_0"],
                            "acceptance_checks": ["Keep evidence traceable."],
                        }
                    ],
                },
                producer="test_stage_standards",
            )
            save_contract_artifact(
                project_root,
                "claim_evidence_graph",
                {
                    "nodes": [
                        {"id": "hypothesis_0", "type": "hypothesis"},
                        {"id": "claim_0", "type": "claim"},
                    ],
                    "edges": [{"source": "hypothesis_0", "target": "claim_0"}],
                },
                producer="test_stage_standards",
            )
            save_experiment_registry(
                project_root,
                [
                    {
                        "record_id": "task_0_1",
                        "task_id": "task_0",
                        "dataset": "demo-ds",
                        "metric": "accuracy",
                        "baseline_ref": "baseline_a",
                        "status": "completed",
                        "entered_storyline": True,
                        "budget_status": "within_budget",
                        "acceptance_checks": ["Keep evidence traceable."],
                        "result_summary": {"metric_mean": 0.9},
                    }
                ],
            )
            save_figure_spec(
                project_root,
                {
                    "schema_version": 1,
                    "figure_count": 1,
                    "figures": [
                        {
                            "figure_id": "figure_0",
                            "claim_id": "claim_0",
                            "status": "ready",
                            "data_files": ["results/demo.npy"],
                            "dedupe_signature": "abc123",
                        }
                    ],
                },
            )
            save_manuscript_state(
                str(project_root),
                {
                    "schema_version": 1,
                    "outline": ["abstract", "results"],
                    "claim_bindings": ["claim_0"],
                    "figure_bindings": {"claim_0": "figure_0"},
                    "missing_evidence": [],
                    "guardrail_status": "ready",
                    "skill_pack": ["results_narrative"],
                },
            )
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "schema_version": 1,
                    "rounds": [{"job_id": "rigor_0", "role": "rigor"}],
                    "active_issues": ["Need one stronger baseline."],
                    "active_issue_records": [
                        {
                            "issue_id": "RVW-demo0001",
                            "text": "Need one stronger baseline.",
                            "status": "active",
                            "section_ids": ["results"],
                        }
                    ],
                    "repair_actions": ["Add a stronger baseline comparison."],
                    "repair_queue": [
                        {
                            "repair_id": "RPR-demo0001",
                            "issue_id": "RVW-demo0001",
                            "issue_text": "Need one stronger baseline.",
                            "status": "ready",
                            "priority_tier": "p1",
                            "priority_score": 31,
                            "primary_target_type": "section",
                            "primary_target_id": "results",
                            "primary_target_label": "Results",
                            "section_ids": ["results"],
                            "repair_actions": ["Add a stronger baseline comparison."],
                            "verification_checks": ["baseline comparison verified"],
                        }
                    ],
                    "role_summaries": {"rigor": {"scores": {"overall": 4}}},
                    "usage_accounting": {"rigor_0": {"tokens": 128}},
                    "verification_checks": ["baseline comparison verified"],
                    "repair_metrics": {
                        "total_issue_count": 1,
                        "active_issue_count": 1,
                        "resolved_issue_count": 0,
                        "persistent_issue_count": 0,
                        "repair_action_count": 1,
                        "verification_count": 1,
                        "bound_issue_count": 1,
                        "unbound_issue_count": 0,
                        "bound_active_issue_count": 1,
                        "target_binding_coverage": 1.0,
                        "active_binding_coverage": 1.0,
                        "role_count": 1,
                        "role_coverage_ratio": 0.25,
                        "resolution_rate": 0.0,
                        "verification_coverage": 1.0,
                        "repair_queue_count": 1,
                        "repair_ready_count": 1,
                        "repair_verification_ready_count": 1,
                        "repair_targeted_count": 1,
                        "repair_queue_coverage": 1.0,
                        "repair_ready_coverage": 1.0,
                        "repair_verification_ready_coverage": 1.0,
                        "repair_targeted_coverage": 1.0,
                    },
                },
                producer="test_stage_standards",
            )
            save_contract_artifact(
                project_root,
                "repair_plan",
                {
                    "schema_version": 1,
                    "lanes": [{"lane": "section_rewrite", "task_count": 1}],
                    "tasks": [
                        {
                            "task_id": "repair_task_0",
                            "issue_id": "RVW-demo0001",
                            "lane": "section_rewrite",
                            "status": "ready",
                            "verification_checks": ["baseline comparison verified"],
                        }
                    ],
                    "summary": {
                        "task_count": 1,
                        "ready_task_count": 1,
                        "blocked_task_count": 0,
                        "verification_ready_count": 1,
                        "lane_count": 1,
                        "ready_rate": 1.0,
                        "verification_ready_rate": 1.0,
                        "targeted_rate": 1.0,
                    },
                },
                producer="test_stage_standards",
            )
            save_contract_artifact(
                project_root,
                "self_evolution",
                {
                    "schema_version": 1,
                    "summary": {
                        "status": "ready",
                        "score": 100.0,
                        "lesson_count": 2,
                        "required_failure_count": 0,
                    },
                    "self_check": {
                        "status": "ready",
                        "score": 100.0,
                        "criteria": [],
                        "required_failures": [],
                    },
                    "lessons": [
                        {
                            "lesson_id": "lane::section_rewrite",
                            "recommended_action": "Rewrite the results section with stronger baseline evidence.",
                        }
                    ],
                },
                producer="test_stage_standards",
            )

            standards = build_stage_standards(project_root)

            by_stage = {row["stage"]: row for row in standards["stage_results"]}
            self.assertGreaterEqual(standards["overall_score"], 80.0)
            self.assertEqual(by_stage["ideation"]["status"], "ready")
            self.assertEqual(by_stage["planning"]["status"], "ready")
            self.assertEqual(by_stage["experiment"]["status"], "ready")
            self.assertEqual(by_stage["figure"]["status"], "ready")
            self.assertEqual(by_stage["manuscript"]["status"], "ready")
            self.assertEqual(by_stage["review"]["status"], "ready")

    def test_save_stage_standards_should_persist_missing_stage_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "partial_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "idea_cards",
                [{"idea_id": "idea_0", "core_hypothesis": "A", "minimum_viable_experiment": "B"}],
                producer="test_stage_standards",
            )

            output_path = save_stage_standards(project_root)
            persisted = json.loads(Path(output_path).read_text(encoding="utf-8"))
            manifest_stage = load_contract_artifact(
                project_root,
                "stage_standards",
                default={},
            )

            self.assertTrue(Path(output_path).exists())
            self.assertEqual(manifest_stage["summary"]["missing_stages"][0], "planning")
            self.assertGreaterEqual(persisted["blocked_stage_count"], 1)
            self.assertGreaterEqual(persisted["missing_stage_count"], 1)

    def test_build_stage_standards_should_block_review_without_repair_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "review_debt_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "schema_version": 1,
                    "rounds": [{"job_id": "clarity_0", "role": "clarity"}],
                    "active_issues": ["The core claim is still too vague."],
                    "active_issue_records": [
                        {
                            "issue_id": "RVW-demo0002",
                            "text": "The core claim is still too vague.",
                            "status": "active",
                        }
                    ],
                    "repair_actions": [],
                    "verification_checks": [],
                    "role_summaries": {"clarity": {"scores": {"overall": 3}}},
                    "usage_accounting": {"clarity_0": {"tokens": 64}},
                    "repair_metrics": {
                        "total_issue_count": 1,
                        "active_issue_count": 1,
                        "resolved_issue_count": 0,
                        "persistent_issue_count": 0,
                        "repair_action_count": 0,
                        "verification_count": 0,
                        "role_count": 1,
                        "role_coverage_ratio": 0.25,
                        "resolution_rate": 0.0,
                        "verification_coverage": 0.0,
                    },
                },
                producer="test_stage_standards",
            )

            standards = build_stage_standards(project_root)

            by_stage = {row["stage"]: row for row in standards["stage_results"]}
            self.assertEqual(by_stage["review"]["status"], "blocked")
            self.assertIn(
                "repair_plan_for_active_issues",
                by_stage["review"]["required_failures"],
            )
            self.assertIn(
                "repair_queue",
                by_stage["review"]["required_failures"],
            )
            self.assertIn(
                "repair_plan_artifact",
                by_stage["review"]["required_failures"],
            )
            self.assertIn(
                "self_evolution_artifact",
                by_stage["review"]["required_failures"],
            )

    def test_build_stage_standards_should_block_strict_visual_workflow_without_full_claim_coverage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "visual_gap_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "research_plan",
                {
                    "workflow_mode": "review_board",
                    "tasks": [
                        {"task_id": "task_0", "claim_targets": ["claim_0"]},
                        {"task_id": "task_1", "claim_targets": ["claim_1"]},
                    ],
                },
                producer="test_stage_standards",
            )
            save_contract_artifact(
                project_root,
                "claim_evidence_graph",
                {
                    "nodes": [
                        {"id": "claim_0", "type": "claim"},
                        {"id": "claim_1", "type": "claim"},
                    ],
                    "edges": [],
                },
                producer="test_stage_standards",
            )
            save_figure_spec(
                project_root,
                {
                    "schema_version": 1,
                    "figure_count": 1,
                    "figures": [
                        {
                            "figure_id": "figure_0",
                            "claim_id": "claim_0",
                            "status": "ready",
                            "paper_slot": "main",
                            "data_files": ["results/demo.npy"],
                            "available_data_files": ["results/demo.npy"],
                            "missing_data_files": [],
                            "dedupe_signature": "sig_0",
                        }
                    ],
                },
            )

            standards = build_stage_standards(project_root)

        figure_stage = next(
            item for item in standards["stage_results"] if item["stage"] == "figure"
        )
        self.assertEqual(figure_stage["status"], "blocked")
        self.assertIn("claim_coverage", figure_stage["required_failures"])
        self.assertIn("main_paper_figures", figure_stage["required_failures"])


if __name__ == "__main__":
    unittest.main()
