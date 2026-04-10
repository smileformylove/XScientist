from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    artifact_path,
    initialize_pipeline_contracts,
    iter_project_roots,
    load_contract_artifact,
    load_pipeline_manifest,
    record_fallback_event,
    render_research_program_markdown,
    save_contract_artifact,
)


class PipelineContractsTests(unittest.TestCase):
    def test_initialize_and_save_contract_artifacts_should_persist_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)

            manifest = initialize_pipeline_contracts(
                project_root,
                project_name="demo_project",
                template_profile="template_first",
                template_capability="high_success_templates",
                workflow_mode="program_driven",
                workflow_label="Program Driven",
                workflow_summary="Program-first orchestration.",
                workflow_inspirations=["karpathy/autoresearch"],
                workflow_sequence=["program", "experiment", "review"],
            )
            self.assertEqual(manifest["project_name"], "demo_project")
            self.assertEqual(manifest["template_profile"], "template_first")
            self.assertEqual(manifest["workflow_mode"], "program_driven")

            save_contract_artifact(
                project_root,
                "idea_cards",
                [{"idea_id": "idea_0", "status": "proposed"}],
                producer="test_pipeline_contracts",
            )
            save_contract_artifact(
                project_root,
                "research_program",
                render_research_program_markdown(
                    project_name="demo_project",
                    target_venue="neurips",
                    template_profile="template_first",
                    idea_name="Idea Alpha",
                    hypothesis="A focused hypothesis.",
                    workflow_mode="program_driven",
                    workflow_summary="Program-first orchestration.",
                    workflow_inspirations=["karpathy/autoresearch"],
                    workflow_sequence=["program", "experiment", "review"],
                    budget={"max_steps": 4, "max_wallclock_minutes": 30, "max_retry_per_task": 1},
                    execution_policy={
                        "execution_style": "budgeted_program_execution",
                        "evidence_pressure": "disciplined",
                        "quality_fallback_policy": "disallowed",
                        "allow_auto_improvement_fallback": False,
                        "reject_on_auto_improvement_fallback": True,
                        "acceptance_rules": [
                            "Every task must declare success_criterion and stop_condition before execution."
                        ],
                        "registry_expectations": [
                            "Persist budget_status and acceptance_checks for each task."
                        ],
                    },
                ),
                producer="test_pipeline_contracts",
            )

            reloaded = load_pipeline_manifest(project_root)
            self.assertEqual(
                reloaded["artifacts"]["idea_cards"]["status"],
                "ready",
            )
            self.assertIn("repair_plan", reloaded["artifacts"])
            self.assertIn("self_evolution", reloaded["artifacts"])
            self.assertIn("process_alignment", reloaded["artifacts"])
            self.assertIn("critic_findings", reloaded["artifacts"])
            self.assertEqual(
                reloaded["artifacts"]["repair_plan"]["status"],
                "missing",
            )
            self.assertEqual(
                reloaded["artifacts"]["critic_findings"]["status"],
                "missing",
            )
            self.assertEqual(
                reloaded["artifacts"]["self_evolution"]["status"],
                "missing",
            )
            self.assertEqual(
                reloaded["artifacts"]["process_alignment"]["status"],
                "missing",
            )
            self.assertEqual(
                load_contract_artifact(project_root, "idea_cards", default=[])[0]["idea_id"],
                "idea_0",
            )
            self.assertTrue(artifact_path(project_root, "research_program").exists())
            research_program = artifact_path(project_root, "research_program").read_text(
                encoding="utf-8"
            )
            self.assertIn("## Operating Policy", research_program)
            self.assertIn("### Registry Discipline", research_program)
            self.assertIn("Quality fallback policy: disallowed", research_program)
            self.assertIn("Reject submission after quality fallback: yes", research_program)

    def test_iter_project_roots_should_only_return_manifest_backed_projects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            valid_project = research_root / "projects" / "valid_project"
            valid_paper = research_root / "papers" / "paper_valid"
            valid_batch = research_root / "batches" / "batch_valid"
            invalid_project = research_root / "projects" / "invalid_project"
            valid_project.mkdir(parents=True, exist_ok=True)
            valid_paper.mkdir(parents=True, exist_ok=True)
            valid_batch.mkdir(parents=True, exist_ok=True)
            invalid_project.mkdir(parents=True, exist_ok=True)

            initialize_pipeline_contracts(valid_project)
            initialize_pipeline_contracts(valid_paper)
            initialize_pipeline_contracts(valid_batch)

            roots = iter_project_roots(research_root)
            self.assertEqual(
                roots,
                [
                    valid_batch.resolve(),
                    valid_paper.resolve(),
                    valid_project.resolve(),
                ],
            )

    def test_record_fallback_event_should_update_manifest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)

            record_fallback_event(
                project_root,
                stage="idea_ranking",
                producer="test_pipeline_contracts",
                fallback_kind="heuristic_ranking",
                reason="ranking parse failed",
                metadata={"reason_counts": {"response_parse_failed": 1}},
            )
            record_fallback_event(
                project_root,
                stage="quality_review",
                producer="test_pipeline_contracts",
                fallback_kind="auto_improvement_rewrite",
                reason="quality gate fallback",
                strict=True,
            )

            manifest = load_pipeline_manifest(project_root)
            summary = manifest["fallback_summary"]
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["strict_count"], 1)
            self.assertEqual(summary["stage_counts"]["idea_ranking"], 1)
            self.assertEqual(summary["stage_counts"]["quality_review"], 1)
            self.assertEqual(summary["kind_counts"]["heuristic_ranking"], 1)
            self.assertEqual(
                summary["kind_counts"]["auto_improvement_rewrite"],
                1,
            )
            self.assertEqual(
                summary["latest_event"]["fallback_kind"],
                "auto_improvement_rewrite",
            )


if __name__ == "__main__":
    unittest.main()
