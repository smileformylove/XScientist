from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.apps.project import (
    _experiment_checkpoint_status,
    _finalize_local_research_handoff,
    _initialize_local_research_git,
    _manuscript_artifact_bindings,
    _paper_result_artifact_fields,
    _record_local_research_attempt_objects,
    _record_local_research_checkpoint,
    _record_local_research_handoff_objects,
    _record_local_research_planning_objects,
    process_single_idea,
)
from ai_scientist.protocol.hashing import content_hash, hash_manifest
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.research_integrity import (
    build_preregistration,
    save_preregistration,
)
from xscientist.research_git import (
    list_research_objects,
    research_log,
    repository_status,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for research history tests")
class ProjectResearchGitIntegrationTests(unittest.TestCase):
    def test_manuscript_artifact_roles_ignore_symlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            paper_root = root / "03_papers"
            private = root / "private.txt"
            paper_root.mkdir(parents=True)
            private.write_text("not a manuscript", encoding="utf-8")
            alias = paper_root / "paper.pdf"
            linked_directory = paper_root / "linked"
            try:
                alias.symlink_to(private)
                linked_directory.symlink_to(root, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks are unavailable: {exc}")

            bindings = _manuscript_artifact_bindings(
                root,
                result={
                    "pdf_path": str(alias),
                    "figure_spec_file": str(linked_directory / "private.txt"),
                },
            )

            self.assertEqual(bindings, [])
            self.assertFalse((root / "research-objects").exists())

    def test_experiment_checkpoint_status_ignores_prior_successes(self) -> None:
        prior_results = [{"status": "success"}]
        executed_results = [{"status": "failed"}]

        self.assertEqual(
            _experiment_checkpoint_status(executed_results),
            "failed",
            prior_results,
        )
        self.assertEqual(
            _experiment_checkpoint_status([*executed_results, {"status": "success"}]),
            "completed",
        )

    def test_terminal_runtime_status_records_typed_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            topic = Path(td) / "topic.md"
            topic.write_text("# Question\n", encoding="utf-8")
            args = self._args(root, topic, policy="stage")
            args._research_git_active = _initialize_local_research_git(args)

            _record_local_research_attempt_objects(
                args,
                results=[
                    {
                        "idea_idx": 0,
                        "status": "locked",
                        "runtime_status": "locked",
                        "registry_status": "failed",
                        "terminal_receipt_hash": "sha256:" + "1" * 64,
                        "stage": "experiment",
                        "error": "Experiment directory is already locked",
                        "resumable": False,
                    }
                ],
            )

            attempt = next(
                item
                for item in list_research_objects(root)
                if item["kind"] == "experiment_attempt"
            )
            self.assertEqual(attempt["state"], "failed")
            self.assertEqual(attempt["payload"]["status"], "failed")
            self.assertEqual(attempt["payload"]["runtime_status"], "locked")
            self.assertEqual(
                attempt["payload"]["terminal_receipt_hash"],
                "sha256:" + "1" * 64,
            )

    def test_distinct_terminal_receipts_create_distinct_manual_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            topic = Path(td) / "topic.md"
            topic.write_text("# Question\n", encoding="utf-8")
            args = self._args(root, topic, policy="manual")
            args._research_git_active = _initialize_local_research_git(args)
            shared = {
                "idea_idx": 0,
                "status": "locked",
                "runtime_status": "locked",
                "registry_status": "failed",
                "stage": "experiment",
                "error": "Experiment directory is already locked",
                "resumable": False,
            }

            _record_local_research_attempt_objects(
                args,
                results=[
                    {**shared, "terminal_receipt_hash": "sha256:" + "1" * 64},
                    {**shared, "terminal_receipt_hash": "sha256:" + "2" * 64},
                ],
            )

            attempts = [
                item
                for item in list_research_objects(root)
                if item["kind"] == "experiment_attempt"
            ]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(
                {item["payload"]["terminal_receipt_hash"] for item in attempts},
                {"sha256:" + "1" * 64, "sha256:" + "2" * 64},
            )

    def test_terminal_paper_failure_binds_generated_pdf_to_cas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            topic = Path(td) / "topic.md"
            topic.write_text("# Question\n", encoding="utf-8")
            args = self._args(root, topic, policy="stage")
            self.assertTrue(_initialize_local_research_git(args))
            exp_dir = root / "02_experiments" / "failed-review"
            exp_dir.mkdir(parents=True)
            pdf = exp_dir / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4\nrejected manuscript\n")
            critic = exp_dir / "hostile_critic" / "findings.json"
            critic.parent.mkdir()
            critic.write_text('{"blocking": true}\n', encoding="utf-8")
            result = {
                "status": "failed",
                **_paper_result_artifact_fields(
                    exp_dir,
                    pdf_path=pdf,
                    critic_findings_file=critic,
                ),
            }

            bindings = _manuscript_artifact_bindings(root, result=result)

            pdf_binding = next(
                item for item in bindings if item["role"] == "manuscript_pdf"
            )
            pointer = json.loads(
                (root / pdf_binding["pointer_path"]).read_text(encoding="utf-8")
            )
            pdf.unlink()
            self.assertTrue((root / pointer["store_relpath"]).is_file())
            self.assertIn("critic_findings", {item["role"] for item in bindings})

    def test_generic_pipeline_failure_preserves_generated_paper_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()

            def fail_after_pdf(*, base_folder, **_kwargs):
                Path(base_folder, "paper.pdf").write_bytes(
                    b"%PDF-1.4\npartial manuscript\n"
                )
                raise RuntimeError("simulated post-generation failure")

            with (
                mock.patch("ai_scientist.apps.project.idea_to_markdown"),
                mock.patch(
                    "ai_scientist.apps.project.edit_bfts_config_file",
                    return_value="/tmp/demo_config.yaml",
                ),
                mock.patch(
                    "ai_scientist.apps.project.perform_experiments_bfts",
                    return_value={"status": "completed"},
                ),
                mock.patch("ai_scientist.apps.project.write_experiment_report"),
                mock.patch(
                    "ai_scientist.apps.project.evaluate_and_save_sample_gate",
                    return_value={"full_generation_allowed": True, "result": {}},
                ),
                mock.patch(
                    "ai_scientist.apps.project.aggregate_plots",
                    side_effect=fail_after_pdf,
                ),
            ):
                result = process_single_idea(
                    (
                        str(root),
                        str(root),
                        0,
                        {
                            "Name": "Partial paper",
                            "Short Hypothesis": "Preserve generated artifacts.",
                            "Experiments": ["Run a baseline."],
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

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "plot_aggregation")
            self.assertTrue(Path(result["pdf_path"]).is_file())
            self.assertEqual(Path(result["exp_dir"]), Path(result["pdf_path"]).parent)
            self.assertEqual(
                Path(result["manuscript_state_file"]).parent,
                Path(result["exp_dir"]),
            )

    def _args(self, root: Path, topic: Path, *, policy: str) -> argparse.Namespace:
        return argparse.Namespace(
            project_dir=str(root),
            topic=str(topic),
            research_git="local",
            git_checkpoint_policy=policy,
            research_git_strict=True,
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )

    def test_failed_publication_gate_is_checkpointed_before_handoff_exits(self) -> None:
        args = argparse.Namespace(
            require_quality_gate=True,
            min_submission_priority=None,
            max_submission_blockers=None,
            strict_writing_guardrails=False,
            high_quality_mode=False,
        )
        events: list[str] = []

        with (
            mock.patch(
                "ai_scientist.apps.project._record_local_research_checkpoint",
                side_effect=lambda *args, **kwargs: events.append(
                    f"checkpoint:{kwargs['status']}"
                ),
            ) as checkpoint_mock,
            mock.patch(
                "ai_scientist.apps.project._export_project_research_dag",
                side_effect=lambda *args, **kwargs: events.append("dag"),
            ) as dag_mock,
        ):
            failure = _finalize_local_research_handoff(
                args,
                results=[{"submission_acceptance_passed": False}],
                ara_paths=["ara/run-0/manifest.json"],
            )

        self.assertIsNotNone(failure)
        self.assertEqual(events, ["checkpoint:failed", "dag"])
        checkpoint_mock.assert_called_once()
        self.assertEqual(checkpoint_mock.call_args.kwargs["stage"], "paper")
        self.assertEqual(
            checkpoint_mock.call_args.kwargs["ara_paths"],
            ["ara/run-0/manifest.json"],
        )
        dag_mock.assert_called_once()

    def test_successful_publication_handoff_uses_completed_checkpoint(self) -> None:
        args = argparse.Namespace(
            require_quality_gate=True,
            min_submission_priority=None,
            max_submission_blockers=None,
            strict_writing_guardrails=False,
            high_quality_mode=False,
        )
        with (
            mock.patch(
                "ai_scientist.apps.project._record_local_research_checkpoint"
            ) as checkpoint_mock,
            mock.patch("ai_scientist.apps.project._export_project_research_dag"),
        ):
            failure = _finalize_local_research_handoff(
                args,
                results=[{"submission_acceptance_passed": True}],
                ara_paths=[],
            )

        self.assertIsNone(failure)
        self.assertEqual(checkpoint_mock.call_args.kwargs["status"], "completed")

    def test_milestone_policy_records_ideation_and_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "project"
            root.mkdir()
            topic = base / "topic.md"
            topic.write_text("# Question\n\nDoes H1 hold?\n", encoding="utf-8")
            args = self._args(root, topic, policy="milestone")
            args._research_git_active = _initialize_local_research_git(args)
            ideas = root / "01_ideas" / "ideas.json"
            ideas.parent.mkdir()
            ideas.write_text("[]\n", encoding="utf-8")

            _record_local_research_checkpoint(
                args,
                stage="ideation",
                subject="record candidates",
                summary="Idea state.",
            )
            metrics = root / "02_experiments" / "run-1" / "metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text('{"metric":{"value":1.0}}\n', encoding="utf-8")
            _record_local_research_checkpoint(
                args,
                stage="experiment",
                subject="complete run-1",
                summary="Experiment state.",
            )

            stages = [
                (entry["trailers"].get("Research-Stage") or [None])[0]
                for entry in research_log(root)
            ]
            self.assertEqual(stages, ["experiment", "ideation", "init"])
            self.assertTrue(
                ideas.exists(), "checkpointed ideation output must be preserved"
            )

    def test_stage_policy_records_ideation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "project"
            root.mkdir()
            topic = base / "topic.md"
            topic.write_text("# Question\n", encoding="utf-8")
            args = self._args(root, topic, policy="stage")
            args._research_git_active = _initialize_local_research_git(args)
            ideas = root / "01_ideas" / "ideas.json"
            ideas.parent.mkdir()
            ideas.write_text("[]\n", encoding="utf-8")

            _record_local_research_checkpoint(
                args,
                stage="ideation",
                subject="record candidates",
                summary="Idea state.",
            )

            latest = research_log(root)[0]
            self.assertEqual(latest["trailers"]["Research-Stage"], ["ideation"])

    def test_project_milestones_persist_typed_lifecycle_and_hold_unverified_paper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "project"
            root.mkdir()
            topic = base / "topic.md"
            topic.write_text("# Question\n", encoding="utf-8")
            args = self._args(root, topic, policy="stage")
            args._research_git_active = _initialize_local_research_git(args)
            idea = {
                "idea_id": "idea_0",
                "title": "H1 study",
                "core_hypothesis": "H1 improves accuracy.",
                "failure_criteria": ["H1 does not improve accuracy."],
            }
            plan = {
                "plan_id": "plan_0",
                "tasks": [
                    {
                        "task_id": "task_0",
                        "dataset": "benchmark-v1",
                        "metric": "accuracy",
                        "baseline": "baseline-a",
                    }
                ],
                "socratic_challenge": {
                    "primary_hypothesis": "H1 improves accuracy.",
                    "proposed_mechanism": "feature M",
                    "rival_hypotheses": [
                        {
                            "rival_id": "rival_null",
                            "class": "null_effect",
                            "statement": "H1 has no reliable effect.",
                            "discriminating_prediction": "accuracy does not improve",
                            "source": "protocol_default",
                        }
                    ],
                    "discriminating_tests": [
                        {
                            "test_id": "paired-test",
                            "targets": ["rival_null"],
                            "design": "run a paired test on the locked split",
                        }
                    ],
                },
            }
            initialize_pipeline_contracts(root)
            save_contract_artifact(root, "research_plan", plan, producer="test")
            save_preregistration(
                root,
                build_preregistration(idea, plan),
                producer="test",
            )

            _record_local_research_planning_objects(args, idea_cards=[idea])
            _record_local_research_checkpoint(
                args,
                stage="ideation",
                subject="record planning",
                summary="planning",
            )
            ara_dir = root / "ara" / "run-0"
            ara_dir.mkdir(parents=True)
            ara_manifest = {
                "schema_version": "ara.v1",
                "protocol_kind": "manifest",
                "counts": {"nodes": 1},
            }
            ara_graph = {
                "nodes": [
                    {
                        "id": "n0",
                        "content_hash": "sha256:" + "a" * 64,
                        "execution_isolation": {"isolated": True},
                        "context_pack_refs": ["sha256:" + "b" * 64],
                    }
                ],
                "edges": [],
            }
            ara_manifest_path = ara_dir / "manifest.json"
            ara_manifest_path.write_text(json.dumps(ara_manifest), encoding="utf-8")
            (ara_dir / "exploration_graph.json").write_text(
                json.dumps(ara_graph), encoding="utf-8"
            )
            paper_root = root / "02_experiments" / "run-0"
            paper_root.mkdir(parents=True)
            latex_path = paper_root / "template.tex"
            latex_path.write_text("\\documentclass{article}\n", encoding="utf-8")
            pdf_path = paper_root / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\nauditable paper\n")
            artifact_payloads = {
                "pipeline_manifest.json": {"artifacts": {}},
                "claim_evidence_graph.json": {"nodes": [], "edges": []},
                "figure_spec.json": {"figures": []},
                "manuscript_state.json": {
                    "latex_path": "template.tex",
                    "guardrail_status": "ready",
                },
                "review_state.json": {"rounds": [], "active_issue_records": []},
                "repair_plan.json": {"summary": {"task_count": 0}},
                "stage_standards.json": {"stages": {}},
            }
            for filename, payload in artifact_payloads.items():
                (paper_root / filename).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            (paper_root / "experiment_registry.jsonl").write_text(
                '{"status":"completed"}\n', encoding="utf-8"
            )
            results = [
                {
                    "idea_idx": 0,
                    "status": "success",
                    "quality_score": 8.0,
                    "rigor_score": 7.5,
                    "quality_gate_passed": True,
                    "submission_acceptance_passed": True,
                    "ara_manifest": str(ara_manifest_path),
                    "pdf_path": str(pdf_path),
                    "pipeline_manifest": str(paper_root / "pipeline_manifest.json"),
                    "claim_evidence_graph_file": str(
                        paper_root / "claim_evidence_graph.json"
                    ),
                    "experiment_registry_file": str(
                        paper_root / "experiment_registry.jsonl"
                    ),
                    "figure_spec_file": str(paper_root / "figure_spec.json"),
                    "manuscript_state_file": str(paper_root / "manuscript_state.json"),
                    "review_state_file": str(paper_root / "review_state.json"),
                    "repair_plan_file": str(paper_root / "repair_plan.json"),
                    "stage_standards_file": str(paper_root / "stage_standards.json"),
                }
            ]
            _record_local_research_attempt_objects(args, results=results)
            _record_local_research_checkpoint(
                args,
                stage="experiment",
                subject="record experiment",
                summary="experiment",
            )
            _record_local_research_handoff_objects(args, results=results)
            _record_local_research_checkpoint(
                args,
                stage="paper",
                subject="record handoff",
                summary="handoff",
            )

            objects = list_research_objects(root)
            gate = next(item for item in objects if item["kind"] == "gate_decision")
            manuscript = next(item for item in objects if item["kind"] == "manuscript")
            attempt = next(
                item for item in objects if item["kind"] == "experiment_attempt"
            )
            evidence = next(item for item in objects if item["kind"] == "evidence")
            context = next(
                item for item in objects if item["kind"] == "context_snapshot"
            )
            self.assertEqual(gate["state"], "rejected")
            self.assertFalse(gate["payload"]["claim_promotion_allowed"])
            self.assertEqual(manuscript["state"], "draft")
            bindings = manuscript["payload"]["artifact_bindings"]
            binding_roles = {item["role"] for item in bindings}
            self.assertTrue(
                {
                    "manuscript_pdf",
                    "manuscript_source",
                    "figure_spec",
                    "manuscript_state",
                    "review_state",
                    "repair_plan",
                }
                <= binding_roles
            )
            self.assertTrue(
                manuscript["payload"]["revision_hash"].startswith("sha256:")
            )
            self.assertEqual(
                set(manuscript["payload"]["artifact_hashes"]),
                {item["content_hash"] for item in bindings},
            )
            checkpoint_refs = set(
                (repository_status(root).get("last_checkpoint") or {}).get(
                    "object_refs"
                )
                or []
            )
            self.assertTrue(
                set(manuscript["payload"]["artifact_hashes"]) <= checkpoint_refs
            )
            pdf_binding = next(
                item for item in bindings if item["role"] == "manuscript_pdf"
            )
            pointer_payload = json.loads(
                (root / pdf_binding["pointer_path"]).read_text(encoding="utf-8")
            )
            pdf_path.unlink()
            self.assertTrue((root / pointer_payload["store_relpath"]).is_file())
            self.assertIn("experiment_attempt", {item["kind"] for item in objects})
            self.assertTrue(
                {
                    "hypothesis_portfolio",
                    "discriminating_prediction",
                    "experiment_design",
                    "experiment_priority",
                    "research_review",
                }
                <= {item["kind"] for item in objects}
            )
            self.assertEqual(
                args._research_vcs_ids["strategy"]["status"],
                "proposed_not_executed",
            )
            self.assertEqual(
                attempt["provenance"]["ara_manifest_hash"],
                hash_manifest(ara_manifest),
            )
            self.assertEqual(
                attempt["provenance"]["ara_exploration_graph_hash"],
                content_hash(ara_graph),
            )
            self.assertEqual(
                evidence["payload"]["ara_exploration_graph_hash"],
                content_hash(ara_graph),
            )
            self.assertEqual(
                attempt["provenance"]["context_hashes"],
                ["sha256:" + "b" * 64],
            )
            self.assertEqual(
                evidence["payload"]["context_pack_refs"],
                ["sha256:" + "b" * 64],
            )
            self.assertTrue(context["payload"]["complete"])
            self.assertEqual(
                gate["payload"]["context_hash"], context["payload"]["context_hash"]
            )
            strategy_followups = json.loads(
                (root / "04_logs" / "research_strategy_followups.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(strategy_followups["active"], [])


if __name__ == "__main__":
    unittest.main()
