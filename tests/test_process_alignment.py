from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.process_alignment import build_process_alignment, save_process_alignment
from ai_scientist.utils.stage_standards import save_stage_standards


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class ProcessAlignmentTests(unittest.TestCase):
    def test_build_process_alignment_should_map_processes_to_reference_repos(self) -> None:
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
                        "core_hypothesis": "A strong hypothesis",
                        "novelty_claim": "Novel idea",
                        "minimum_viable_experiment": "Run one baseline.",
                        "candidate_datasets": ["demo"],
                        "candidate_metrics": ["accuracy"],
                        "candidate_baselines": ["base"],
                        "failure_criteria": ["no gain"],
                    }
                ],
                producer="test_process_alignment",
            )
            save_contract_artifact(
                project_root,
                "research_plan",
                {
                    "workflow_mode": "program_driven",
                    "budget": {
                        "max_steps": 6,
                        "max_wallclock_minutes": 20,
                        "max_retry_per_task": 1,
                    },
                    "execution_policy": {
                        "policy_name": "program_driven",
                        "acceptance_rules": ["Every task must define success criteria."],
                        "registry_expectations": ["Preserve every failure in experiment_registry.jsonl."],
                    },
                    "tasks": [
                        {
                            "task_id": "task_0",
                            "task_kind": "ablation",
                            "claim_targets": ["claim_0"],
                            "expected_outputs": ["figure_0"],
                        }
                    ],
                },
                producer="test_process_alignment",
            )
            save_contract_artifact(
                project_root,
                "claim_evidence_graph",
                {
                    "nodes": [
                        {"id": "hyp_0", "type": "hypothesis"},
                        {"id": "claim_0", "type": "claim"},
                    ],
                    "edges": [{"source": "hyp_0", "target": "claim_0", "type": "supports"}],
                },
                producer="test_process_alignment",
            )
            (project_root / "experiment_registry.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "task_0",
                        "dataset": "demo",
                        "metric": "accuracy",
                        "baseline_ref": "base",
                        "status": "completed",
                        "budget_status": "within_budget",
                        "acceptance_checks": ["accuracy improves"],
                        "result_summary": {"metric_name": "accuracy", "metric_mean": 0.9},
                        "entered_storyline": True,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            save_contract_artifact(
                project_root,
                "figure_spec",
                {
                    "figures": [
                        {
                            "figure_id": "figure_0",
                            "claim_id": "claim_0",
                            "status": "ready",
                            "data_files": ["results/demo.json"],
                            "source_records": ["task_0"],
                            "caption_intent": "Show accuracy gain.",
                        }
                    ]
                },
                producer="test_process_alignment",
            )
            save_contract_artifact(
                project_root,
                "manuscript_state",
                {
                    "outline": ["intro", "results"],
                    "section_briefs": {"results": "Show the core result."},
                    "claim_bindings": ["claim_0"],
                    "figure_bindings": {"claim_0": "figure_0"},
                    "guardrail_status": "ready",
                    "missing_evidence": [],
                },
                producer="test_process_alignment",
            )
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "rounds": [{"job_id": "review_0", "role": "rigor"}],
                    "repair_queue": [{"repair_id": "RPR-0", "status": "ready"}],
                    "repair_metrics": {
                        "role_coverage_ratio": 0.5,
                        "target_binding_coverage": 1.0,
                        "repair_queue_coverage": 1.0,
                        "repair_ready_coverage": 1.0,
                    },
                },
                producer="test_process_alignment",
            )
            save_contract_artifact(
                project_root,
                "repair_plan",
                {
                    "summary": {
                        "task_count": 1,
                    }
                },
                producer="test_process_alignment",
            )
            save_contract_artifact(
                project_root,
                "self_evolution",
                {
                    "summary": {
                        "status": "ready",
                        "score": 91.0,
                        "lesson_count": 2,
                    },
                    "self_check": {
                        "required_failures": [],
                    },
                    "lessons": [{"lesson_id": "L0"}],
                    "next_cycle_defaults": {"review": "tighten"},
                    "stage_risks": ["clarity_gap"],
                },
                producer="test_process_alignment",
            )
            _write_json(
                project_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "quality_gate_passed": True,
                    "submission_priority_score": 92.0,
                    "submission_readiness": {"ready": True, "status": "ready"},
                    "rewrite_trace": [{"round": 1}],
                    "submission_package_file": "quality/submission_package.md",
                },
            )
            (project_root / "research_program.md").write_text(
                "# Research Program\n\n## Success Criteria\n- pass\n\n## Failure Handling\n- repair\n",
                encoding="utf-8",
            )

            alignment = build_process_alignment(project_root)

        self.assertEqual(alignment["summary"]["blocked_process_count"], 0)
        self.assertGreaterEqual(alignment["summary"]["overall_score"], 80.0)
        by_process = {
            item["process"]: item for item in alignment.get("process_results") or []
        }
        self.assertIn("ideation", by_process)
        self.assertIn("program", by_process)
        self.assertIn("review", by_process)
        self.assertIn("evolution", by_process)
        self.assertIn("SakanaAI/AI-Scientist", [item["name"] for item in by_process["ideation"]["references"]])
        self.assertIn("karpathy/autoresearch", [item["name"] for item in by_process["program"]["references"]])
        self.assertIn("ResearAI/DeepReviewer-v2", [item["name"] for item in by_process["review"]["references"]])
        self.assertEqual(by_process["evolution"]["status"], "ready")

    def test_save_process_alignment_should_persist_contract_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "idea_cards",
                [{"idea_id": "idea_0", "core_hypothesis": "x"}],
                producer="test_process_alignment",
            )

            path = save_process_alignment(project_root, producer="test_process_alignment")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertTrue(path.endswith("process_alignment.json"))
        self.assertEqual(payload["schema_version"], 1)

    def test_save_stage_standards_should_refresh_process_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "idea_cards",
                [{"idea_id": "idea_0", "core_hypothesis": "x"}],
                producer="test_process_alignment",
            )

            save_stage_standards(project_root)
            process_alignment_path = project_root / "process_alignment.json"
            exists = process_alignment_path.exists()
            payload = json.loads(process_alignment_path.read_text(encoding="utf-8"))

        self.assertTrue(exists)
        self.assertIn("process_results", payload)


if __name__ == "__main__":
    unittest.main()
