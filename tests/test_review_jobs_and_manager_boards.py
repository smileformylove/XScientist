from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    record_fallback_event,
    save_contract_artifact,
)
from ai_scientist.utils.review_jobs import begin_review_job, finalize_review_job
from research_manager import ResearchManager


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class ReviewJobsAndManagerBoardTests(unittest.TestCase):
    def _write_source_project(
        self,
        research_root: Path,
        *,
        project_name: str,
        source_name: str,
        source_key: str,
        source_workflow_mode: str,
        source_archetype: str,
        source_batch_profile: str,
        target_venue: str,
        quality_score_after: float,
        submission_priority_score: float,
        quality_gate_passed: bool,
        submission_status: str,
        fallback_count: int = 0,
        strict_fallback_count: int = 0,
        self_evolution: dict | None = None,
    ) -> None:
        project_root = research_root / "projects" / project_name
        project_root.mkdir(parents=True, exist_ok=True)
        initialize_pipeline_contracts(project_root)
        _write_json(
            project_root / "idea.json",
            {"Name": project_name, "Title": f"{project_name} Title"},
        )
        _write_json(
            project_root / "source_provenance.json",
            {
                "source_name": source_name,
                "source_key": source_key,
                "source_type": "topic",
                "source_value": f"{source_name}.md",
                "source_target_venue": target_venue,
                "source_paper_types": ["journal"],
                "source_workflow_mode": source_workflow_mode,
                "source_archetype": source_archetype,
                "source_batch_profile": source_batch_profile,
            },
        )
        for idx in range(fallback_count):
            record_fallback_event(
                project_root,
                stage="idea_ranking",
                producer="test_source_project",
                fallback_kind="heuristic_ranking",
                reason=f"fallback event {idx}",
            )
        for idx in range(strict_fallback_count):
            record_fallback_event(
                project_root,
                stage="quality_review",
                producer="test_source_project",
                fallback_kind="auto_improvement_rewrite",
                reason=f"strict fallback event {idx}",
                strict=True,
            )
        _write_json(
            project_root / "quality" / "high_quality_result.json",
            {
                "status": "success",
                "target_venue": target_venue,
                "quality_score_after": quality_score_after,
                "rigor_score_after": 4.0,
                "claim_support_after": 4.0,
                "claim_alignment_after": 4.0,
                "numeric_coverage_after": 4.0,
                "breakthrough_score": 4.1,
                "evidence_density_score": 3.2,
                "contribution_count": 3,
                "quality_gate_passed": quality_gate_passed,
                "submission_priority_score": submission_priority_score,
                "submission_priority_tier": "submit_now",
                "blocker_count": 0,
                "unsupported_claims_count": 0,
                "critical_revision_actions_count": 0,
                "rewrite_trace": [{"round": 1}],
                "rewrite_applied": True,
                "submission_readiness": {
                    "status": submission_status,
                    "ready": submission_status == "ready",
                    "blockers": [],
                    "categories": {},
                },
            },
        )
        if self_evolution is not None:
            save_contract_artifact(
                project_root,
                "self_evolution",
                self_evolution,
                producer="test_source_project",
            )

    def test_review_job_should_persist_review_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "manuscript_state",
                {"schema_version": 1, "guardrail_status": "ready"},
                producer="test_review_job",
            )

            job = begin_review_job(
                project_root,
                role="rigor",
                model_review="demo-model",
                review_plan={"strategy": "depth"},
            )
            finalize_review_job(
                project_root,
                job=job,
                review_text={
                    "review": {
                        "Weaknesses": ["Need a stronger baseline."],
                        "Questions": ["Why is the improvement stable?"],
                        "Limitations": ["Ablation coverage is thin."],
                    }
                },
                review_img={"review": {}},
                pdf_path=str(project_root / "paper.pdf"),
                usage_summary={"tokens": 128},
                evidence_refs=["experiment_registry.jsonl"],
            )

            review_state = json.loads((project_root / "review_state.json").read_text(encoding="utf-8"))
            repair_plan = json.loads((project_root / "repair_plan.json").read_text(encoding="utf-8"))
            self_evolution = json.loads(
                (project_root / "self_evolution.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(review_state["rounds"]), 1)
            self.assertIn("Need a stronger baseline.", review_state["active_issues"])
            self.assertEqual(review_state["usage_accounting"][job["job_id"]]["tokens"], 128)
            self.assertEqual(review_state["repair_metrics"]["active_issue_count"], 3)
            self.assertEqual(review_state["repair_metrics"]["repair_action_count"], 1)
            self.assertEqual(review_state["repair_metrics"]["verification_count"], 1)
            self.assertEqual(len(review_state["repair_queue"]), 3)
            self.assertEqual(review_state["repair_metrics"]["repair_queue_count"], 3)
            self.assertGreaterEqual(
                review_state["repair_metrics"]["repair_ready_coverage"],
                1.0,
            )
            self.assertEqual(repair_plan["summary"]["task_count"], 3)
            self.assertGreaterEqual(repair_plan["summary"]["lane_count"], 1)
            self.assertGreaterEqual(self_evolution["summary"]["lesson_count"], 1)
            self.assertEqual(
                self_evolution["summary"]["dominant_lane"],
                "evidence_followup",
            )

    def test_review_job_should_track_resolution_metrics_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "manuscript_state",
                {"schema_version": 1, "guardrail_status": "ready"},
                producer="test_review_job",
            )

            first_job = begin_review_job(
                project_root,
                role="rigor",
                model_review="demo-model",
                review_plan={"strategy": "depth"},
            )
            finalize_review_job(
                project_root,
                job=first_job,
                review_text={
                    "review": {
                        "Weaknesses": ["Need a stronger baseline."],
                        "Questions": [],
                        "Limitations": ["Add one controlled comparison."],
                    }
                },
                review_img={"review": {}},
                pdf_path=str(project_root / "paper.pdf"),
                usage_summary={"tokens": 64},
                evidence_refs=["experiment_registry.jsonl"],
            )

            second_job = begin_review_job(
                project_root,
                role="rigor",
                model_review="demo-model",
                review_plan={"strategy": "depth"},
            )
            finalize_review_job(
                project_root,
                job=second_job,
                review_text={
                    "review": {
                        "Weaknesses": [],
                        "Questions": [],
                        "Limitations": [],
                        "Verification Checks": ["Baseline comparison now included."],
                    }
                },
                review_img={"review": {}},
                pdf_path=str(project_root / "paper.pdf"),
                usage_summary={"tokens": 72},
                evidence_refs=["quality/high_quality_result.json"],
            )

            review_state = json.loads(
                (project_root / "review_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(review_state["rounds"]), 2)
            self.assertEqual(review_state["active_issues"], [])
            self.assertIn("Need a stronger baseline.", review_state["resolved_issues"])
            self.assertEqual(
                review_state["repair_metrics"]["resolved_issue_count"],
                2,
            )
            self.assertEqual(
                review_state["repair_metrics"]["active_issue_count"],
                0,
            )
            self.assertGreaterEqual(
                review_state["repair_metrics"]["resolution_rate"],
                1.0,
            )

    def test_hostile_critic_job_should_persist_lane_summary_and_findings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "critic_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "manuscript_state",
                {"schema_version": 1, "guardrail_status": "ready"},
                producer="test_review_job",
            )

            job = begin_review_job(
                project_root,
                role="claim_cross_examiner",
                model_review="demo-model",
                review_plan={"strategy": "depth"},
                lane_name="hostile_critic",
                suite_name="hostile_critic",
                strictness_profile="adversarial",
            )
            finalize_review_job(
                project_root,
                job=job,
                review_text={
                    "review": {
                        "Weaknesses": ["The abstract overclaims generalization beyond the evaluated setting."],
                        "Questions": ["Which figure actually supports the broad claim?"],
                        "Limitations": [],
                    }
                },
                review_img={"review": {}},
                pdf_path=str(project_root / "paper.pdf"),
                usage_summary={"tokens": 80},
                evidence_refs=["claim_evidence_graph.json", "experiment_registry.jsonl"],
            )

            review_state = json.loads((project_root / "review_state.json").read_text(encoding="utf-8"))
            critic_findings = json.loads(
                (project_root / "critic_findings.json").read_text(encoding="utf-8")
            )

            self.assertIn("hostile_critic", review_state["lane_summaries"])
            self.assertEqual(
                review_state["lane_summaries"]["hostile_critic"]["strictness_profile"],
                "adversarial",
            )
            self.assertEqual(critic_findings["lane_name"], "hostile_critic")
            self.assertEqual(critic_findings["active_issue_count"], 2)
            self.assertTrue(critic_findings["findings"])
            self.assertEqual(
                critic_findings["findings"][0]["role"],
                "claim_cross_examiner",
            )

    def test_review_job_should_bind_issue_targets_to_claim_figure_and_section(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "binding_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            save_contract_artifact(
                project_root,
                "claim_evidence_graph",
                {
                    "nodes": [
                        {"id": "claim_0", "type": "claim", "label": "Baseline robustness claim"},
                    ],
                    "edges": [],
                },
                producer="test_review_job",
            )
            save_contract_artifact(
                project_root,
                "figure_spec",
                {
                    "schema_version": 1,
                    "figure_count": 1,
                    "figures": [
                        {
                            "figure_id": "figure_0",
                            "claim_id": "claim_0",
                            "status": "ready",
                            "suggested_title": "Baseline robustness figure",
                            "caption_intent": "Show baseline robustness in the results section.",
                            "figure_type": "comparison_plot",
                        }
                    ],
                },
                producer="test_review_job",
            )
            save_contract_artifact(
                project_root,
                "manuscript_state",
                {
                    "schema_version": 1,
                    "guardrail_status": "ready",
                    "outline": ["introduction", "results", "discussion"],
                    "section_briefs": {
                        "results": "Present baseline robustness evidence.",
                        "discussion": "Interpret robustness and limitations.",
                    },
                    "claim_bindings": ["claim_0"],
                    "figure_bindings": {"claim_0": "figure_0"},
                    "section_claim_bindings": {
                        "results": ["claim_0"],
                        "discussion": ["claim_0"],
                    },
                    "section_figure_bindings": {"results": ["figure_0"]},
                },
                producer="test_review_job",
            )

            job = begin_review_job(
                project_root,
                role="rigor",
                model_review="demo-model",
                review_plan={"strategy": "depth"},
            )
            finalize_review_job(
                project_root,
                job=job,
                review_text={
                    "review": {
                        "Weaknesses": [
                            "The baseline robustness figure in the results section is unclear."
                        ],
                        "Limitations": ["Clarify the baseline robustness figure legend."],
                    }
                },
                review_img={"review": {}},
                pdf_path=str(project_root / "paper.pdf"),
                usage_summary={"tokens": 96},
                evidence_refs=["figure_spec.json"],
            )

            review_state = json.loads(
                (project_root / "review_state.json").read_text(encoding="utf-8")
            )
            repair_plan = json.loads(
                (project_root / "repair_plan.json").read_text(encoding="utf-8")
            )
            issue_record = review_state["active_issue_records"][0]
            self.assertEqual(issue_record["claim_ids"], ["claim_0"])
            self.assertEqual(issue_record["figure_ids"], ["figure_0"])
            self.assertIn("results", issue_record["section_ids"])
            self.assertEqual(review_state["issue_to_claim"][issue_record["issue_id"]], ["claim_0"])
            self.assertEqual(review_state["issue_to_figure"][issue_record["issue_id"]], ["figure_0"])
            self.assertIn("results", review_state["issue_to_section"][issue_record["issue_id"]])
            self.assertEqual(
                review_state["repair_metrics"]["target_binding_coverage"],
                1.0,
            )
            repair_task = review_state["repair_queue"][0]
            self.assertEqual(repair_task["primary_target_type"], "figure")
            self.assertEqual(repair_task["primary_target_id"], "figure_0")
            self.assertEqual(repair_task["status"], "ready")
            self.assertTrue(repair_task["verification_checks"])
            self.assertEqual(repair_plan["tasks"][0]["lane"], "figure_repair")
            self.assertEqual(repair_plan["summary"]["ready_rate"], 1.0)

    def test_manager_boards_should_surface_pipeline_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            project_root = research_root / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(
                project_root,
                template_profile="template_first",
                template_capability="high_success_templates",
                workflow_mode="review_board",
                workflow_label="Review Board",
                workflow_summary="Review-first hardening.",
                workflow_inspirations=["ResearAI/DeepReviewer-v2"],
                workflow_sequence=["writeup", "multi_role_review", "repair"],
            )
            save_contract_artifact(
                project_root,
                "idea_cards",
                [
                    {
                        "idea_id": "idea_0",
                        "name": "Idea Alpha",
                        "title": "Idea Alpha Title",
                        "status": "proposed",
                        "target_venue": "neurips",
                        "candidate_datasets": ["demo-ds"],
                        "candidate_metrics": ["accuracy"],
                        "candidate_baselines": ["baseline_a"],
                        "compute_risk": "low",
                        "minimum_viable_experiment": "Run a baseline comparison.",
                    }
                ],
                producer="test_manager",
            )
            save_contract_artifact(
                project_root,
                "figure_spec",
                {
                    "schema_version": 1,
                    "figure_count": 1,
                    "figures": [
                        {
                            "figure_id": "figure_0",
                            "claim_id": "claim_0",
                            "status": "ready",
                            "figure_type": "comparison_plot",
                            "paper_slot": "main",
                            "data_files": ["results/demo.npy"],
                            "source_records": ["baseline_summary"],
                            "blocking_reasons": [],
                        }
                    ],
                },
                producer="test_manager",
            )
            (project_root / "experiment_registry.jsonl").write_text(
                json.dumps(
                    {
                        "record_id": "task_0_record",
                        "task_id": "task_0",
                        "dataset": "demo-ds",
                        "metric": "accuracy",
                        "baseline_ref": "baseline_a",
                        "status": "completed",
                        "entered_storyline": True,
                        "workflow_mode": "review_board",
                        "policy_name": "review_board",
                        "budget": {
                            "max_steps": 9,
                            "max_wallclock_minutes": 70,
                            "max_retry_per_task": 1,
                        },
                        "budget_status": "within_budget",
                        "acceptance_checks": [
                            "Only storyline tasks with reproducibility-ready evidence can enter the final narrative."
                        ],
                        "result_summary": {"metric_name": "accuracy", "metric_mean": 0.9},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            save_contract_artifact(
                project_root,
                "research_plan",
                {
                    "plan_id": "idea_0_plan",
                    "idea_id": "idea_0",
                    "workflow_mode": "review_board",
                    "budget": {
                        "max_steps": 9,
                        "max_wallclock_minutes": 70,
                        "max_retry_per_task": 1,
                    },
                    "execution_policy": {
                        "policy_name": "review_board",
                        "execution_style": "review_hardened_evidence_flow",
                        "evidence_pressure": "review_hardened",
                        "acceptance_rules": [
                            "Only storyline tasks with reproducibility-ready evidence can enter the final narrative."
                        ],
                    },
                    "tasks": [],
                },
                producer="test_manager",
            )
            manifest_path = project_root / "pipeline_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["experiment_registry"]["status"] = "ready"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            record_fallback_event(
                project_root,
                stage="idea_ranking",
                producer="test_manager",
                fallback_kind="heuristic_ranking",
                reason="ranking parser fallback",
                metadata={"reason_counts": {"response_parse_failed": 1}},
            )
            record_fallback_event(
                project_root,
                stage="quality_review",
                producer="test_manager",
                fallback_kind="auto_improvement_rewrite",
                reason="quality auto rewrite fallback",
                strict=True,
            )
            _write_json(
                project_root / "source_provenance.json",
                {
                    "source_name": "demo_source",
                    "source_key": "topic::demo_source::topic.md",
                    "source_type": "topic",
                    "source_value": "topic.md",
                    "source_target_venue": "nature",
                    "source_paper_types": ["journal"],
                    "source_workflow_mode": "review_board",
                    "source_archetype": "review_hardening",
                    "source_batch_profile": "review_hardening",
                },
            )

            _write_json(
                research_root / "papers" / "paper_demo" / "idea.json",
                {"Name": "Paper Demo", "Title": "Paper Demo Title"},
            )
            _write_json(
                research_root / "papers" / "paper_demo" / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.6,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.1,
                    "claim_alignment_after": 4.0,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.2,
                    "evidence_density_score": 3.5,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 93.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {
                        "status": "ready",
                        "ready": True,
                        "blockers": [],
                        "categories": {},
                    },
                },
            )
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "schema_version": 1,
                    "rounds": [{"job_id": "rigor_0", "role": "rigor"}],
                    "active_issues": ["The main figure still needs a stronger baseline caption."],
                    "active_issue_records": [
                        {
                            "issue_id": "RVW-demo2001",
                            "text": "The main figure still needs a stronger baseline caption.",
                            "status": "active",
                            "role": "rigor",
                            "severity": "major",
                            "figure_ids": ["figure_0"],
                            "section_ids": ["results"],
                            "is_bound": True,
                            "is_strongly_bound": True,
                        }
                    ],
                    "repair_actions": ["Revise figure_0 caption with the stronger baseline comparison."],
                    "verification_checks": ["Verify figure_0 now supports the main result claim."],
                    "repair_queue": [
                        {
                            "repair_id": "RPR-demo2001",
                            "issue_id": "RVW-demo2001",
                            "issue_text": "The main figure still needs a stronger baseline caption.",
                            "role": "rigor",
                            "severity": "major",
                            "status": "ready",
                            "priority_tier": "p0",
                            "priority_score": 48,
                            "primary_target_type": "figure",
                            "primary_target_id": "figure_0",
                            "primary_target_label": "figure 0",
                            "figure_ids": ["figure_0"],
                            "section_ids": ["results"],
                            "repair_actions": [
                                "Revise figure_0 caption with the stronger baseline comparison."
                            ],
                            "verification_checks": [
                                "Verify figure_0 now supports the main result claim."
                            ],
                            "blocking_reasons": [],
                        }
                    ],
                    "role_summaries": {"rigor": {"scores": {"overall": 4}}},
                    "usage_accounting": {"rigor_0": {"tokens": 88}},
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
                producer="test_manager",
            )

            manager = ResearchManager(str(research_root))
            manager.rebuild_index()
            pipeline_rows = manager.pipeline_status(top_n=5)
            fallback_rows = manager.fallback_board(top_n=5)
            standards_rows = manager.stage_standards_board(top_n=10)
            idea_rows = manager.idea_board(top_n=5)
            experiment_rows = manager.experiment_board(top_n=5)
            figure_rows = manager.figure_board(top_n=5)
            repair_rows = manager.repair_board(top_n=5)
            evolution_rows = manager.evolution_board(top_n=5)
            process_rows = manager.process_board(top_n=20)
            source_rows = manager.source_board(top_n=5)
            source_mix = manager.source_mix_advisory(
                desired_policy="review_board",
                top_n=5,
            )
            source_mix_mismatch = manager.source_mix_advisory(
                desired_policy="program_driven",
                top_n=5,
            )
            trends = manager.benchmark_trends(target_venue="nature", max_entries=10)

            self.assertEqual(pipeline_rows[0]["project"], "demo_project")
            self.assertEqual(pipeline_rows[0]["workflow_mode"], "review_board")
            self.assertEqual(pipeline_rows[0]["execution_policy"], "review_board")
            self.assertEqual(pipeline_rows[0]["fallback_count"], 2)
            self.assertEqual(pipeline_rows[0]["strict_fallback_count"], 1)
            self.assertEqual(
                pipeline_rows[0]["budget"]["max_retry_per_task"],
                1,
            )
            self.assertEqual(fallback_rows[0]["project"], "demo_project")
            self.assertEqual(fallback_rows[0]["fallback_count"], 2)
            self.assertEqual(
                fallback_rows[0]["kind_counts"]["heuristic_ranking"],
                1,
            )
            self.assertEqual(
                fallback_rows[0]["kind_counts"]["auto_improvement_rewrite"],
                1,
            )
            self.assertEqual(standards_rows[0]["project"], "demo_project")
            self.assertIn(standards_rows[0]["status"], {"blocked", "missing"})
            self.assertEqual(idea_rows[0]["idea_id"], "idea_0")
            self.assertEqual(idea_rows[0]["workflow_mode"], "review_board")
            self.assertEqual(experiment_rows[0]["task_id"], "task_0")
            self.assertEqual(experiment_rows[0]["policy_name"], "review_board")
            self.assertEqual(experiment_rows[0]["budget_status"], "within_budget")
            self.assertTrue(experiment_rows[0]["acceptance_checks"])
            self.assertEqual(figure_rows[0]["figure_id"], "figure_0")
            self.assertEqual(repair_rows[0]["primary_target_type"], "figure")
            self.assertEqual(repair_rows[0]["status"], "ready")
            self.assertEqual(repair_rows[0]["lane"], "figure_repair")
            self.assertEqual(evolution_rows[0]["project"], "demo_project")
            self.assertIn(evolution_rows[0]["status"], {"ready", "needs_attention"})
            self.assertGreaterEqual(evolution_rows[0]["lesson_count"], 1)
            self.assertTrue(process_rows)
            review_process_rows = [
                row for row in process_rows if row.get("process") == "review"
            ]
            self.assertTrue(review_process_rows)
            self.assertEqual(review_process_rows[0]["project"], "demo_project")
            self.assertIn(
                "ResearAI/DeepReviewer-v2",
                review_process_rows[0]["references"],
            )
            self.assertEqual(source_rows[0]["source_name"], "demo_source")
            self.assertEqual(source_rows[0]["source_archetype"], "review_hardening")
            self.assertEqual(source_rows[0]["source_batch_profile"], "review_hardening")
            self.assertEqual(source_rows[0]["source_workflow_mode"], "review_board")
            self.assertEqual(source_rows[0]["avg_fallback_count"], 2.0)
            self.assertEqual(source_rows[0]["avg_strict_fallback_count"], 1.0)
            self.assertEqual(source_rows[0]["fallback_run_count"], 1)
            self.assertIsInstance(
                source_rows[0]["avg_self_evolution_score"],
                float,
            )
            self.assertEqual(
                source_mix["summary"]["dominant_archetype"],
                "review_hardening",
            )
            self.assertEqual(
                source_mix["summary"]["dominant_workflow_mode"],
                "review_board",
            )
            self.assertTrue(source_mix["recommendations"])
            self.assertEqual(
                source_mix_mismatch["recommendations"][0]["label"],
                "missing_desired_policy_source",
            )
            self.assertEqual(trends["summary"]["ready_count"], 1)
            self.assertTrue(trends["timeline"])

    def test_submission_board_should_exclude_strict_fallback_runs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            clean_root = research_root / "papers" / "paper_20260321_000001_clean_journal"
            clean_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(clean_root)
            _write_json(
                clean_root / "idea.json",
                {"Name": "Clean Nature", "Title": "Clean Nature Title"},
            )
            (clean_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                clean_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.6,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.3,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 92.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {
                        "status": "ready",
                        "ready": True,
                        "blockers": [],
                        "categories": {},
                    },
                },
            )

            fallback_root = research_root / "papers" / "paper_20260321_000002_fallback_journal"
            fallback_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(fallback_root)
            record_fallback_event(
                fallback_root,
                stage="quality_review",
                producer="test_submission_board",
                fallback_kind="auto_improvement_rewrite",
                reason="strict fallback",
                strict=True,
            )
            _write_json(
                fallback_root / "idea.json",
                {"Name": "Fallback Nature", "Title": "Fallback Nature Title"},
            )
            (fallback_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                fallback_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.7,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.4,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 94.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {
                        "status": "ready",
                        "ready": True,
                        "blockers": [],
                        "categories": {},
                    },
                },
            )

            manager = ResearchManager(str(research_root))
            manager.rebuild_index()
            papers = manager.list_papers(sort_by="quality")
            board = manager.submission_board(top_n_per_venue=5, require_gate=True)
            shortlist = manager.shortlist_papers(require_gate=True, top_n=5)

            by_name = {paper["name"]: paper for paper in papers}
            self.assertEqual(by_name["Fallback Nature"]["strict_fallback_count"], 1)
            self.assertEqual(by_name["Clean Nature"]["strict_fallback_count"], 0)
            self.assertEqual(
                [paper["name"] for paper in board["nature"]],
                ["Clean Nature"],
            )
            self.assertEqual(
                [paper["name"] for paper in shortlist],
                ["Clean Nature"],
            )

    def test_submission_board_should_exclude_blocked_stage_standard_runs_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            clean_root = research_root / "papers" / "paper_20260321_000003_clean_stage_journal"
            clean_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(clean_root)
            _write_json(
                clean_root / "idea.json",
                {"Name": "Clean Stage Nature", "Title": "Clean Stage Nature Title"},
            )
            (clean_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                clean_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.6,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.3,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 92.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {
                        "status": "ready",
                        "ready": True,
                        "blockers": [],
                        "categories": {},
                    },
                },
            )
            save_contract_artifact(
                clean_root,
                "stage_standards",
                {
                    "schema_version": 1,
                    "overall_score": 94.0,
                    "ready_stage_count": 6,
                    "blocked_stage_count": 0,
                    "needs_attention_stage_count": 0,
                    "missing_stage_count": 0,
                    "summary": {
                        "blocked_stages": [],
                        "attention_stages": [],
                        "missing_stages": [],
                        "top_risks": [],
                    },
                    "stage_results": [],
                },
                producer="test_submission_board",
            )

            blocked_root = research_root / "papers" / "paper_20260321_000004_blocked_stage_journal"
            blocked_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(blocked_root)
            _write_json(
                blocked_root / "idea.json",
                {"Name": "Blocked Stage Nature", "Title": "Blocked Stage Nature Title"},
            )
            (blocked_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                blocked_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.7,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.4,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 94.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {
                        "status": "ready",
                        "ready": True,
                        "blockers": [],
                        "categories": {},
                    },
                },
            )
            save_contract_artifact(
                blocked_root,
                "stage_standards",
                {
                    "schema_version": 1,
                    "overall_score": 61.0,
                    "ready_stage_count": 4,
                    "blocked_stage_count": 1,
                    "needs_attention_stage_count": 1,
                    "missing_stage_count": 0,
                    "summary": {
                        "blocked_stages": ["review"],
                        "attention_stages": ["manuscript"],
                        "missing_stages": [],
                        "top_risks": ["review_evidence_gap"],
                    },
                    "stage_results": [],
                },
                producer="test_submission_board",
            )

            manager = ResearchManager(str(research_root))
            manager.rebuild_index()
            papers = manager.list_papers(sort_by="quality")
            board = manager.submission_board(top_n_per_venue=5, require_gate=True)
            shortlist = manager.shortlist_papers(require_gate=True, top_n=5)

            by_name = {paper["name"]: paper for paper in papers}
            self.assertEqual(by_name["Blocked Stage Nature"]["blocked_stage_count"], 1)
            self.assertEqual(by_name["Clean Stage Nature"]["blocked_stage_count"], 0)
            self.assertEqual(
                [paper["name"] for paper in board["nature"]],
                ["Clean Stage Nature"],
            )
            self.assertEqual(
                [paper["name"] for paper in shortlist],
                ["Clean Stage Nature"],
            )

    def test_submission_board_should_exclude_blocked_self_evolution_runs_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            clean_root = research_root / "papers" / "paper_20260321_000005_clean_evolution_journal"
            clean_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(clean_root)
            _write_json(
                clean_root / "idea.json",
                {"Name": "Clean Evolution Nature", "Title": "Clean Evolution Nature Title"},
            )
            (clean_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                clean_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.6,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.3,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 92.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {
                        "status": "ready",
                        "ready": True,
                        "blockers": [],
                        "categories": {},
                    },
                },
            )
            save_contract_artifact(
                clean_root,
                "self_evolution",
                {
                    "summary": {
                        "status": "ready",
                        "score": 94.0,
                        "dominant_lane": "section_rewrite",
                        "dominant_role": "clarity",
                    },
                    "self_check": {
                        "status": "ready",
                        "score": 94.0,
                        "required_failures": [],
                    },
                    "stage_risks": [],
                },
                producer="test_submission_board",
            )
            save_contract_artifact(
                clean_root,
                "process_alignment",
                {
                    "summary": {
                        "overall_score": 92.0,
                        "ready_process_count": 8,
                        "blocked_process_count": 0,
                        "needs_attention_process_count": 1,
                        "missing_process_count": 0,
                        "top_process_risks": {},
                    },
                    "process_results": [],
                },
                producer="test_submission_board",
            )

            blocked_root = research_root / "papers" / "paper_20260321_000006_blocked_evolution_journal"
            blocked_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(blocked_root)
            _write_json(
                blocked_root / "idea.json",
                {"Name": "Blocked Evolution Nature", "Title": "Blocked Evolution Nature Title"},
            )
            (blocked_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                blocked_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.7,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.4,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 94.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {
                        "status": "ready",
                        "ready": True,
                        "blockers": [],
                        "categories": {},
                    },
                },
            )
            save_contract_artifact(
                blocked_root,
                "self_evolution",
                {
                    "summary": {
                        "status": "blocked",
                        "score": 61.0,
                        "dominant_lane": "triage",
                        "dominant_role": "rigor",
                    },
                    "self_check": {
                        "status": "blocked",
                        "score": 61.0,
                        "required_failures": ["repair_targeting"],
                    },
                    "stage_risks": ["repair_ownership_gap"],
                },
                producer="test_submission_board",
            )
            save_contract_artifact(
                blocked_root,
                "process_alignment",
                {
                    "summary": {
                        "overall_score": 92.0,
                        "ready_process_count": 8,
                        "blocked_process_count": 0,
                        "needs_attention_process_count": 1,
                        "missing_process_count": 0,
                        "top_process_risks": {},
                    },
                    "process_results": [],
                },
                producer="test_submission_board",
            )

            manager = ResearchManager(str(research_root))
            manager.rebuild_index()
            papers = manager.list_papers(sort_by="quality")
            board = manager.submission_board(top_n_per_venue=5, require_gate=True)
            shortlist = manager.shortlist_papers(require_gate=True, top_n=5)

            by_name = {paper["name"]: paper for paper in papers}
            self.assertEqual(by_name["Blocked Evolution Nature"]["self_evolution_status"], "blocked")
            self.assertEqual(
                by_name["Blocked Evolution Nature"]["self_evolution_required_failure_count"],
                1,
            )
            self.assertEqual(
                [paper["name"] for paper in board["nature"]],
                ["Clean Evolution Nature"],
            )
            self.assertEqual(
                [paper["name"] for paper in shortlist],
                ["Clean Evolution Nature"],
            )

    def test_submission_board_should_exclude_blocked_process_alignment_runs_by_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            clean_root = research_root / "papers" / "paper_20260321_000007_clean_process_journal"
            clean_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(clean_root)
            _write_json(clean_root / "idea.json", {"Name": "Clean Process Nature"})
            (clean_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                clean_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.6,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.3,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 92.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {"status": "ready", "ready": True, "blockers": [], "categories": {}},
                },
            )
            save_contract_artifact(
                clean_root,
                "process_alignment",
                {
                    "summary": {
                        "overall_score": 92.0,
                        "ready_process_count": 8,
                        "blocked_process_count": 0,
                        "needs_attention_process_count": 1,
                        "missing_process_count": 0,
                        "top_process_risks": {},
                    },
                    "process_results": [],
                },
                producer="test_submission_board",
            )

            blocked_root = research_root / "papers" / "paper_20260321_000008_blocked_process_journal"
            blocked_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(blocked_root)
            _write_json(blocked_root / "idea.json", {"Name": "Blocked Process Nature"})
            (blocked_root / "paper.pdf").write_text("pdf", encoding="utf-8")
            _write_json(
                blocked_root / "quality" / "high_quality_result.json",
                {
                    "status": "success",
                    "target_venue": "nature",
                    "quality_score_after": 4.7,
                    "rigor_score_after": 4.2,
                    "claim_support_after": 4.2,
                    "claim_alignment_after": 4.1,
                    "numeric_coverage_after": 4.0,
                    "breakthrough_score": 4.4,
                    "evidence_density_score": 3.8,
                    "contribution_count": 3,
                    "quality_gate_passed": True,
                    "submission_priority_score": 94.0,
                    "submission_priority_tier": "submit_now",
                    "blocker_count": 0,
                    "unsupported_claims_count": 0,
                    "critical_revision_actions_count": 0,
                    "rewrite_trace": [{"round": 1}],
                    "rewrite_applied": True,
                    "submission_readiness": {"status": "ready", "ready": True, "blockers": [], "categories": {}},
                },
            )
            save_contract_artifact(
                blocked_root,
                "process_alignment",
                {
                    "summary": {
                        "overall_score": 58.0,
                        "ready_process_count": 4,
                        "blocked_process_count": 1,
                        "needs_attention_process_count": 2,
                        "missing_process_count": 1,
                        "top_process_risks": {"exploration_graph_gap": 1},
                    },
                    "process_results": [],
                },
                producer="test_submission_board",
            )

            manager = ResearchManager(str(research_root))
            manager.rebuild_index()
            papers = manager.list_papers(sort_by="quality")
            board = manager.submission_board(top_n_per_venue=5, require_gate=True)
            shortlist = manager.shortlist_papers(require_gate=True, top_n=5)

            by_name = {paper["name"]: paper for paper in papers}
            self.assertEqual(
                by_name["Blocked Process Nature"]["process_alignment_blocked_process_count"],
                1,
            )
            self.assertEqual(
                [paper["name"] for paper in board["nature"]],
                ["Clean Process Nature"],
            )
            self.assertEqual(
                [paper["name"] for paper in shortlist],
                ["Clean Process Nature"],
            )

    def test_source_next_batch_advisory_should_build_multi_lane_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            self._write_source_project(
                research_root,
                project_name="program_project",
                source_name="program_source",
                source_key="topic::program_source::program.md",
                source_workflow_mode="program_driven",
                source_archetype="program_guarded",
                source_batch_profile="submission_push",
                target_venue="nature",
                quality_score_after=4.7,
                submission_priority_score=95.0,
                quality_gate_passed=True,
                submission_status="ready",
                fallback_count=3,
                strict_fallback_count=1,
            )
            self._write_source_project(
                research_root,
                project_name="frontier_project",
                source_name="frontier_source",
                source_key="topic::frontier_source::frontier.md",
                source_workflow_mode="agentic_tree",
                source_archetype="frontier_exploration",
                source_batch_profile="exploration_sprint",
                target_venue="neurips",
                quality_score_after=4.2,
                submission_priority_score=86.0,
                quality_gate_passed=False,
                submission_status="drafting",
            )
            self._write_source_project(
                research_root,
                project_name="review_project",
                source_name="review_source",
                source_key="topic::review_source::review.md",
                source_workflow_mode="review_board",
                source_archetype="review_hardening",
                source_batch_profile="review_hardening",
                target_venue="nature",
                quality_score_after=4.5,
                submission_priority_score=91.0,
                quality_gate_passed=True,
                submission_status="ready",
            )

            manager = ResearchManager(str(research_root))
            manager.rebuild_index()
            advisory = manager.source_next_batch_advisory(
                desired_policy="program_driven",
                top_n=10,
                max_slots=3,
            )

            self.assertEqual(
                advisory["cadence"]["label"],
                "submission_hardening_loop",
            )
            self.assertEqual(len(advisory["slots"]), 3)
            slots_by_lane = {
                str(item.get("lane")): item for item in advisory.get("slots") or []
            }
            self.assertEqual(
                slots_by_lane["primary_lane"]["source"],
                "program_source",
            )
            self.assertEqual(
                slots_by_lane["primary_lane"]["source_workflow_mode"],
                "program_driven",
            )
            self.assertEqual(
                slots_by_lane["diversification_lane"]["source"],
                "frontier_source",
            )
            self.assertEqual(
                slots_by_lane["hardening_lane"]["source"],
                "review_source",
            )
            self.assertTrue(advisory["recommendations"])
            self.assertTrue(
                any(
                    item.get("label") == "reduce_fallback_debt"
                    for item in advisory["recommendations"]
                )
            )


if __name__ == "__main__":
    unittest.main()
