from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ai_scientist.apps.project import (
    _initialize_local_research_git,
    _record_local_research_attempt_objects,
    _record_local_research_checkpoint,
    _record_local_research_handoff_objects,
    _record_local_research_planning_objects,
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
from xscientist.research_git import list_research_objects, research_log


@unittest.skipUnless(shutil.which("git"), "Git is required for research history tests")
class ProjectResearchGitIntegrationTests(unittest.TestCase):
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
            results = [
                {
                    "idea_idx": 0,
                    "status": "success",
                    "quality_score": 8.0,
                    "rigor_score": 7.5,
                    "quality_gate_passed": True,
                    "submission_acceptance_passed": True,
                    "ara_manifest": str(ara_manifest_path),
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


if __name__ == "__main__":
    unittest.main()
