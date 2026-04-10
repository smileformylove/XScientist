from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.run_index import infer_run_entry, save_workflow_state


class RunIndexTests(unittest.TestCase):
    def test_infer_run_entry_should_track_dynamic_stage_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            run_dir = output_root / "projects" / "demo_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(
                run_dir,
                workflow_mode="review_board",
                workflow_label="Review Board",
                workflow_sequence=["program", "planning", "multi_role_review"],
            )
            save_workflow_state(
                run_dir,
                {
                    "stages": {
                        "program": {"status": "completed"},
                        "planning": {"status": "completed"},
                        "multi_role_review": {"status": "completed"},
                    },
                    "artifacts": {},
                    "updated_at": "2026-03-15T12:00:00",
                },
            )
            (run_dir / "source_provenance.json").write_text(
                json.dumps(
                    {
                        "source_name": "day_source",
                        "source_key": "topic::day_source::topic.md",
                        "source_workflow_mode": "program_driven",
                        "source_archetype": "program_guarded",
                        "source_batch_profile": "submission_push",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            save_contract_artifact(
                run_dir,
                "stage_standards",
                {
                    "schema_version": 1,
                    "overall_score": 88.5,
                    "ready_stage_count": 4,
                    "blocked_stage_count": 1,
                    "needs_attention_stage_count": 1,
                    "missing_stage_count": 0,
                    "summary": {
                        "blocked_stages": ["review"],
                        "attention_stages": ["figure"],
                        "missing_stages": [],
                        "top_risks": ["review_not_ready", "figure_support_thin"],
                    },
                    "stage_results": [],
                },
                producer="test_run_index",
            )
            save_contract_artifact(
                run_dir,
                "review_state",
                {
                    "schema_version": 1,
                    "rounds": [{"job_id": "rigor_0", "role": "rigor"}],
                    "active_issue_records": [
                        {
                            "issue_id": "RVW-demo1001",
                            "text": "Need a stronger baseline.",
                            "status": "active",
                            "claim_ids": ["claim_0"],
                            "section_ids": ["results"],
                        }
                    ],
                    "resolved_issue_records": [
                        {
                            "issue_id": "RVW-demo1002",
                            "text": "Clarify novelty positioning.",
                            "status": "resolved",
                            "section_ids": ["introduction"],
                        }
                    ],
                    "persistent_issue_records": [
                        {
                            "issue_id": "RVW-demo1001",
                            "text": "Need a stronger baseline.",
                            "status": "active",
                            "claim_ids": ["claim_0"],
                            "section_ids": ["results"],
                        }
                    ],
                    "repair_actions": ["Add a stronger baseline."],
                    "repair_queue": [
                        {
                            "repair_id": "RPR-demo1001",
                            "issue_id": "RVW-demo1001",
                            "issue_text": "Need a stronger baseline.",
                            "status": "ready",
                            "priority_tier": "p0",
                            "priority_score": 44,
                            "primary_target_type": "claim",
                            "primary_target_id": "claim_0",
                            "primary_target_label": "claim 0",
                            "claim_ids": ["claim_0"],
                            "section_ids": ["results"],
                            "repair_actions": ["Add a stronger baseline."],
                            "verification_checks": [
                                "Verified against experiment_registry.jsonl"
                            ],
                        }
                    ],
                    "verification_checks": ["Verified against experiment_registry.jsonl"],
                    "role_summaries": {"rigor": {"scores": {"overall": 4}}},
                    "usage_accounting": {"rigor_0": {"tokens": 100}},
                },
                producer="test_run_index",
            )
            save_contract_artifact(
                run_dir,
                "repair_plan",
                {
                    "schema_version": 1,
                    "lanes": [{"lane": "claim_repair", "task_count": 1}],
                    "tasks": [
                        {
                            "task_id": "repair_task_0",
                            "issue_id": "RVW-demo1001",
                            "lane": "claim_repair",
                            "status": "ready",
                            "verification_checks": [
                                "Verified against experiment_registry.jsonl"
                            ],
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
                producer="test_run_index",
            )
            save_contract_artifact(
                run_dir,
                "self_evolution",
                {
                    "schema_version": 1,
                    "summary": {
                        "status": "needs_attention",
                        "score": 83.0,
                        "lesson_count": 3,
                        "required_failure_count": 0,
                        "dominant_lane": "claim_repair",
                        "dominant_role": "rigor",
                    },
                    "self_check": {
                        "status": "needs_attention",
                        "score": 83.0,
                        "criteria": [],
                        "required_failures": [],
                    },
                    "stage_risks": ["review_not_ready"],
                    "next_cycle_defaults": {
                        "experiment": ["Front-load stronger baseline evidence."]
                    },
                },
                producer="test_run_index",
            )

            entry = infer_run_entry(run_dir, output_root=output_root)

            self.assertEqual(entry["workflow_mode"], "review_board")
            self.assertEqual(
                entry["declared_stage_sequence"],
                ["program", "planning", "multi_role_review"],
            )
            self.assertEqual(
                entry["observed_stage_sequence"],
                ["program", "planning", "multi_role_review"],
            )
            self.assertEqual(
                entry["completed_stages"][:3],
                ["program", "planning", "multi_role_review"],
            )
            self.assertEqual(entry["latest_stage"], "multi_role_review")
            self.assertEqual(entry["stage_sequence"][:3], entry["declared_stage_sequence"])
            self.assertEqual(entry["source_workflow_mode"], "program_driven")
            self.assertEqual(entry["source_archetype"], "program_guarded")
            self.assertEqual(entry["source_batch_profile"], "submission_push")
            self.assertEqual(entry["stage_overall_score"], 88.5)
            self.assertEqual(entry["blocked_stage_count"], 1)
            self.assertEqual(entry["needs_attention_stage_count"], 1)
            self.assertEqual(entry["missing_stage_count"], 0)
            self.assertTrue(entry["repair_plan_file"])
            self.assertEqual(entry["blocked_standard_stages"], ["review"])
            self.assertEqual(
                entry["top_standard_risks"],
                ["review_not_ready", "figure_support_thin"],
            )
            self.assertEqual(entry["review_active_issue_count"], 1)
            self.assertEqual(entry["review_resolved_issue_count"], 1)
            self.assertEqual(entry["review_persistent_issue_count"], 1)
            self.assertEqual(entry["review_repair_action_count"], 1)
            self.assertEqual(entry["review_verification_count"], 1)
            self.assertEqual(entry["review_bound_issue_count"], 2)
            self.assertEqual(entry["review_unbound_issue_count"], 0)
            self.assertEqual(entry["review_target_binding_coverage"], 1.0)
            self.assertEqual(entry["review_repair_queue_count"], 1)
            self.assertEqual(entry["review_repair_ready_count"], 1)
            self.assertEqual(entry["review_repair_ready_coverage"], 1.0)
            self.assertEqual(entry["repair_plan_task_count"], 1)
            self.assertEqual(entry["repair_plan_ready_task_count"], 1)
            self.assertEqual(entry["repair_plan_ready_rate"], 1.0)
            self.assertEqual(entry["self_evolution_status"], "needs_attention")
            self.assertEqual(entry["self_evolution_score"], 83.0)
            self.assertEqual(entry["self_evolution_lesson_count"], 3)
            self.assertEqual(entry["self_evolution_dominant_lane"], "claim_repair")
            self.assertEqual(entry["self_evolution_next_cycle_stages"], ["experiment"])


if __name__ == "__main__":
    unittest.main()
