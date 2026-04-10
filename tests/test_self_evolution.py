from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.self_evolution import (
    build_self_evolution,
    load_self_evolution_playbook,
    save_self_evolution,
)


class SelfEvolutionTests(unittest.TestCase):
    def test_save_self_evolution_should_persist_project_artifact_and_playbook(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            project_root = research_root / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "schema_version": 1,
                    "rounds": [{"job_id": "rigor_0", "role": "rigor"}],
                    "issue_ledger": [
                        {
                            "issue_id": "RVW-demo3001",
                            "text": "Need a stronger baseline comparison.",
                            "status": "active",
                            "role": "rigor",
                            "claim_ids": ["claim_0"],
                        }
                    ],
                    "active_issue_records": [
                        {
                            "issue_id": "RVW-demo3001",
                            "text": "Need a stronger baseline comparison.",
                            "status": "active",
                            "role": "rigor",
                            "claim_ids": ["claim_0"],
                        }
                    ],
                    "repair_metrics": {
                        "active_issue_count": 1,
                        "resolved_issue_count": 0,
                        "persistent_issue_count": 0,
                        "resolution_rate": 0.0,
                        "verification_coverage": 1.0,
                        "active_binding_coverage": 1.0,
                        "repair_ready_coverage": 1.0,
                        "repair_targeted_coverage": 1.0,
                    },
                },
                producer="test_self_evolution",
            )
            save_contract_artifact(
                project_root,
                "repair_plan",
                {
                    "schema_version": 1,
                    "lanes": [
                        {
                            "lane": "evidence_followup",
                            "task_count": 1,
                            "ready_count": 1,
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
                producer="test_self_evolution",
            )

            output_path = save_self_evolution(project_root, producer="test_self_evolution")

            self.assertTrue(Path(output_path).exists())
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["dominant_lane"], "evidence_followup")
            self.assertGreaterEqual(payload["summary"]["lesson_count"], 1)

            playbook = load_self_evolution_playbook(project_root)
            self.assertEqual(playbook["project_count"], 1)
            self.assertTrue(playbook["top_agentic_defaults"])
            self.assertTrue(playbook["top_recurring_risks"])

    def test_build_self_evolution_should_surface_stage_blockers_as_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "blocked_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            payload = build_self_evolution(
                project_root,
                review_state={
                    "schema_version": 1,
                    "rounds": [{"job_id": "clarity_0", "role": "clarity"}],
                    "issue_ledger": [],
                    "repair_metrics": {
                        "active_issue_count": 0,
                        "resolved_issue_count": 1,
                        "persistent_issue_count": 0,
                        "resolution_rate": 1.0,
                        "verification_coverage": 1.0,
                        "active_binding_coverage": 1.0,
                        "repair_ready_coverage": 1.0,
                        "repair_targeted_coverage": 1.0,
                    },
                },
                repair_plan={
                    "schema_version": 1,
                    "lanes": [],
                    "summary": {
                        "task_count": 0,
                        "ready_task_count": 0,
                        "blocked_task_count": 0,
                        "verification_ready_count": 0,
                        "lane_count": 0,
                        "ready_rate": 1.0,
                        "verification_ready_rate": 1.0,
                        "targeted_rate": 1.0,
                    },
                },
                stage_standards={
                    "schema_version": 1,
                    "blocked_stage_count": 1,
                    "summary": {
                        "blocked_stages": ["figure"],
                        "top_risks": ["figure_support_thin"],
                    },
                },
            )

            self.assertEqual(payload["summary"]["blocked_stage_count"], 1)
            self.assertIn("figure_support_thin", payload["stage_risks"])
            self.assertTrue(
                any(
                    lesson.get("risk") == "stage_standard_blocker"
                    for lesson in payload["lessons"]
                )
            )


if __name__ == "__main__":
    unittest.main()
