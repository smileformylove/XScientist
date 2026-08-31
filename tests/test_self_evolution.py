from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_harness import (
    bind_version_evidence,
    build_evolution_harness_audit,
    build_harness_policy_hash,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.self_evolution import (
    build_self_evolution,
    load_self_evolution_playbook,
    save_self_evolution,
)
from ai_scientist.utils.science_constitution import (
    build_science_constitution,
    save_science_constitution,
)


def _harness_version(
    version_id: str,
    parent_version_id: str | None,
    *,
    score: float,
    loop_rate: float,
    test_rate: float,
) -> dict:
    digest = canonical_content_hash({"fixture": "self-evolution-harness"})
    return bind_version_evidence(
        {
            "epoch_id": "epoch:self-evolution-test",
            "version_id": version_id,
            "parent_version_id": parent_version_id,
            "scores": {"objective": score},
            "reward_groups": [[0.0, 0.4, 1.0], [0.1, 0.5, 0.8]],
            "behavior": {"loop_rate": loop_rate, "test_rate": test_rate},
            "behavior_thresholds": {
                "loop_rate": {
                    "direction": "lower",
                    "healthy_bound": 0.25,
                    "max_regression": 0.05,
                },
                "test_rate": {
                    "direction": "higher",
                    "healthy_bound": 0.5,
                    "max_regression": 0.05,
                },
            },
            "integrity_checks": {
                "evidence_bound": True,
                "environment_isolated": True,
                "evaluation_frozen": True,
                "git_leakage_absent": True,
            },
            "comparison_hashes": {
                "harness_hash": digest,
                "policy_hash": build_harness_policy_hash(),
                "evaluator_hash": digest,
                "task_hash": digest,
                "resource_hash": digest,
                "seed_policy_hash": digest,
            },
            "cost": {"observed": 4.0, "budget": 10.0, "unit": "tokens"},
        }
    )


class SelfEvolutionTests(unittest.TestCase):
    def test_existing_invalid_constitution_stops_evolution_without_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "tampered_project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            constitution = build_science_constitution(project_name="tampered")
            save_science_constitution(project_root, constitution, producer="test")
            constitution_path = project_root / "science_constitution.json"
            tampered = json.loads(constitution_path.read_text(encoding="utf-8"))
            tampered["core_policy"]["principles"][0][
                "rule"
            ] = "Optimize appearance only."
            constitution_path.write_text(
                json.dumps(tampered, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "refusing self-evolution"):
                save_self_evolution(project_root, producer="test")

            persisted = json.loads(constitution_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["core_policy"]["principles"][0]["rule"],
                "Optimize appearance only.",
            )
            self.assertFalse((project_root / "self_evolution.json").exists())

    def test_save_self_evolution_should_persist_project_artifact_and_playbook(
        self,
    ) -> None:
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

            output_path = save_self_evolution(
                project_root, producer="test_self_evolution"
            )

            self.assertTrue(Path(output_path).exists())
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["dominant_lane"], "evidence_followup")
            self.assertGreaterEqual(payload["summary"]["lesson_count"], 1)
            self.assertFalse(
                payload["promotion_policy"]["automatic_production_mutation_allowed"]
            )
            self.assertTrue(payload["promotion_policy"]["requires_evolution_gate"])
            self.assertTrue(payload["promotion_policy"]["requires_evolution_program"])
            self.assertTrue((project_root / "evolution_program.json").exists())
            self.assertTrue((project_root / "science_constitution.json").exists())

            playbook = load_self_evolution_playbook(project_root)
            self.assertEqual(playbook["project_count"], 1)
            self.assertTrue(playbook["top_agentic_defaults"])
            self.assertTrue(playbook["top_recurring_risks"])

    def test_build_self_evolution_should_surface_stage_blockers_as_lessons(
        self,
    ) -> None:
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

    def test_harness_score_behavior_divergence_blocks_self_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "harness_project"
            project_root.mkdir(parents=True)
            initialize_pipeline_contracts(project_root)
            audit = build_evolution_harness_audit(
                [
                    _harness_version(
                        "v1", None, score=0.5, loop_rate=0.1, test_rate=0.8
                    ),
                    _harness_version(
                        "v2", "v1", score=0.8, loop_rate=0.5, test_rate=0.2
                    ),
                ]
            )

            payload = build_self_evolution(
                project_root,
                review_state={
                    "repair_metrics": {
                        "active_issue_count": 0,
                        "persistent_issue_count": 0,
                        "resolution_rate": 1.0,
                    }
                },
                repair_plan={"summary": {}},
                harness_audit=audit,
            )

            self.assertEqual(payload["self_check"]["status"], "blocked")
            self.assertEqual(payload["summary"]["harness_decision"], "hold")
            self.assertGreater(payload["summary"]["harness_blocking_risk_count"], 0)
            self.assertIn(
                "BEHAVIOR.SCORE_DIVERGENCE",
                payload["harness_snapshot"]["blocking_risk_codes"],
            )
            self.assertTrue(
                any(
                    lesson.get("risk") == "harness_behavior_score_divergence"
                    for lesson in payload["lessons"]
                )
            )

    def test_explicit_invalid_harness_argument_fails_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "invalid_harness"
            project_root.mkdir(parents=True)
            initialize_pipeline_contracts(project_root)

            with self.assertRaisesRegex(TypeError, "harness_audit"):
                build_self_evolution(project_root, harness_audit=[])  # type: ignore[arg-type]
            with self.assertRaisesRegex(TypeError, "harness_audit"):
                save_self_evolution(project_root, harness_audit=[])  # type: ignore[arg-type]

            self.assertFalse((project_root / "evolution_harness.json").exists())
            self.assertFalse((project_root / "self_evolution.json").exists())
            self.assertFalse((project_root / "science_constitution.json").exists())

    def test_saving_harness_evidence_feeds_next_epoch_challenges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "harness_program"
            project_root.mkdir(parents=True)
            initialize_pipeline_contracts(project_root)
            audit = build_evolution_harness_audit(
                [
                    _harness_version(
                        "v1", None, score=0.5, loop_rate=0.1, test_rate=0.8
                    ),
                    _harness_version(
                        "v2", "v1", score=0.8, loop_rate=0.5, test_rate=0.2
                    ),
                ]
            )

            save_self_evolution(
                project_root,
                review_state={
                    "repair_metrics": {
                        "active_issue_count": 0,
                        "persistent_issue_count": 0,
                        "resolution_rate": 1.0,
                    }
                },
                repair_plan={"summary": {}},
                harness_audit=audit,
                producer="test_harness_integration",
            )

            self.assertTrue((project_root / "evolution_harness.json").exists())
            program = json.loads(
                (project_root / "evolution_program.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                any(
                    str(item.get("risk") or "").startswith("harness_")
                    for item in program["evaluation_challenges"]
                )
            )
            self.assertTrue(
                all(
                    item["automatic_application_allowed"] is False
                    for item in program["evaluation_challenges"]
                )
            )


if __name__ == "__main__":
    unittest.main()
