from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ai_scientist.utils.evolution_gate import (
    EvolutionGateError,
    approve_production_promotion,
    build_ablation_report,
    build_canary_report,
    build_evolution_candidate,
    build_evolution_gate,
    build_rollback_receipt,
    save_evolution_gate,
    validate_ablation_report,
    validate_evolution_candidate,
    validate_evolution_gate,
    validate_production_promotion,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_pipeline_manifest,
)
from ai_scientist.utils.science_constitution import (
    build_science_constitution,
    save_science_constitution,
)
from xscientist import ResearchEvolution, ResearchRepository
from xscientist.research_git import ResearchGitError


def _digest(char: str) -> str:
    return "sha256:" + hashlib.sha256(char.encode("utf-8")).hexdigest()


def _constitution() -> dict:
    return build_science_constitution(project_name="evolution-test")


def _candidate(*, risk_tier: str = "moderate") -> dict:
    return build_evolution_candidate(
        constitution=_constitution(),
        candidate_id="search-policy-v2",
        component_type="search_policy",
        base_version="1.0.0",
        candidate_version="1.1.0",
        base_artifact_hash=_digest("a"),
        candidate_artifact_hash=_digest("b"),
        rollback_ref="git:base-commit",
        proposed_by="agent:evolution",
        change_summary="Allocate more search budget to falsification branches.",
        change_scope=["search/falsification-budget"],
        applicability_domains=["general"],
        failure_taxonomy_refs=["failure:premature-convergence"],
        ablation_dimensions=["falsification-budget"],
        provenance_hashes=[_digest("candidate-provenance")],
        risk_tier=risk_tier,
    )


def _sample(index: int, *, hidden: bool = True, regressed: bool = False) -> dict:
    prospective = index == 0
    payload = {
        "task_id": f"hidden-task-{index}",
        "task_hash": _digest(f"hidden-task-{index}"),
        "split": "hidden" if hidden else "public",
        "evaluation_layer": "prospective" if prospective else "sealed",
        "frozen_before_candidate": True,
        "benchmark_frozen_at": "2020-01-01T00:00:00Z",
        "domain": "general",
        "producer_stack_id": "stack:research-producer",
        "evaluator_stack_id": f"stack:independent-{index % 2}",
        "producer_stack_hash": _digest("research-producer-stack"),
        "evaluator_stack_hash": _digest(f"independent-stack-{index % 2}"),
        "baseline": {
            "objective_score": 0.60,
            "reproducibility_rate": 0.90,
            "false_discovery_rate": 0.10,
            "cost_per_task": 1.00,
            "latency_seconds": 10.0,
        },
        "candidate": {
            "objective_score": 0.50 if regressed else 0.66,
            "reproducibility_rate": 0.80 if regressed else 0.95,
            "false_discovery_rate": 0.20 if regressed else 0.08,
            "cost_per_task": 1.05,
            "latency_seconds": 11.0,
        },
        "safety_pass": not regressed,
        "integrity_pass": not regressed,
        "reproducibility_pass": not regressed,
    }
    if prospective:
        payload.update(
            {
                "prospective_resolved": True,
                "prospective_protocol_hash": _digest("prospective-protocol"),
                "resolution_attestation_hash": _digest("prospective-resolution"),
                "resolution_not_before": "2020-01-02T00:00:00Z",
                "resolved_at": "2020-01-03T00:00:00Z",
            }
        )
    else:
        payload["custodian_attestation_hash"] = _digest(f"sealed-custody-{index}")
    return payload


def _ablation(candidate: dict | None = None) -> dict:
    resolved = candidate or _candidate()
    return build_ablation_report(
        resolved,
        [
            {
                "task_id": f"ablation-{index}",
                "dimension": "falsification-budget",
                "full_candidate_score": 0.66,
                "ablated_score": 0.60,
                "full_run_hash": _digest(f"full-ablation-{index}"),
                "ablated_run_hash": _digest(f"removed-ablation-{index}"),
            }
            for index in range(3)
        ],
    )


def _gate(
    *,
    candidate: dict | None = None,
    samples: list[dict] | None = None,
    policy: dict | None = None,
) -> dict:
    resolved_candidate = candidate or _candidate()
    return build_evolution_gate(
        resolved_candidate,
        samples if samples is not None else [_sample(index) for index in range(5)],
        constitution=_constitution(),
        ablation_report=_ablation(resolved_candidate),
        policy=policy,
    )


def _canary(candidate: dict) -> dict:
    rollback = build_rollback_receipt(
        candidate,
        restored_artifact_hash=candidate["base_artifact_hash"],
        execution_log_hash=_digest("rollback-log"),
        executed_by="service:release-canary",
    )
    return build_canary_report(
        candidate,
        rollback,
        executed_by="service:release-canary",
        observation_count=25,
        error_rate_delta=-0.01,
        quality_delta=0.02,
        real_research_project_ids=["project:a", "project:b", "project:c"],
        project_run_hashes={
            project_id: _digest(project_id)
            for project_id in ["project:a", "project:b", "project:c"]
        },
        incidents=[],
        long_tail_pass=True,
        common_mode_failure_pass=True,
        out_of_distribution_pass=True,
    )


class EvolutionGateTests(unittest.TestCase):
    def test_candidate_requires_distinct_content_addressed_artifacts(self) -> None:
        with self.assertRaises(EvolutionGateError):
            build_evolution_candidate(
                constitution=_constitution(),
                candidate_id="same",
                component_type="prompt",
                base_version="1",
                candidate_version="2",
                base_artifact_hash=_digest("a"),
                candidate_artifact_hash=_digest("a"),
                rollback_ref="git:base",
                proposed_by="agent:test",
                change_summary="No real change.",
                change_scope=["prompts/demo"],
                applicability_domains=["general"],
                failure_taxonomy_refs=["failure:no-change"],
                ablation_dimensions=["prompt-change"],
                provenance_hashes=[_digest("provenance")],
            )

    def test_candidate_cannot_mutate_constitution_protected_components(self) -> None:
        with self.assertRaises(EvolutionGateError):
            build_evolution_candidate(
                constitution=_constitution(),
                candidate_id="weaken-verification",
                component_type="verification_policy",
                base_version="1",
                candidate_version="2",
                base_artifact_hash=_digest("base"),
                candidate_artifact_hash=_digest("candidate"),
                rollback_ref="git:base",
                proposed_by="agent:evolution",
                change_summary="Weaken a protected gate.",
                change_scope=["verification_policy/gates"],
                applicability_domains=["all"],
                failure_taxonomy_refs=["failure:false-positive"],
                ablation_dimensions=["gate"],
                provenance_hashes=[_digest("provenance")],
            )

    def test_candidate_hash_detects_shadow_mutation(self) -> None:
        candidate = _candidate()
        self.assertTrue(validate_evolution_candidate(candidate)["ok"])
        candidate["change_summary"] = "Post-hoc mutation"
        self.assertIn(
            "candidate_hash_mismatch",
            validate_evolution_candidate(candidate)["errors"],
        )

    def test_policy_cannot_disable_integrity_hard_gates(self) -> None:
        with self.assertRaises(EvolutionGateError):
            _gate(policy={"require_integrity_pass": False})
        with self.assertRaises(EvolutionGateError):
            _gate(policy={"minimum_hidden_tasks": 1})
        with self.assertRaises(EvolutionGateError):
            _gate(
                policy={
                    "metric_specs": {"objective_score": {"minimum_improvement": -1.0}}
                },
            )

    def test_ablation_must_attribute_each_declared_change(self) -> None:
        candidate = _candidate()
        failed = build_ablation_report(
            candidate,
            [
                {
                    "task_id": "ablation-0",
                    "dimension": "falsification-budget",
                    "full_candidate_score": 0.60,
                    "ablated_score": 0.61,
                    "full_run_hash": _digest("failed-full-run"),
                    "ablated_run_hash": _digest("failed-ablated-run"),
                }
            ],
        )
        self.assertFalse(failed["passed"])
        self.assertTrue(validate_ablation_report(failed, candidate=candidate)["passed"])

    def test_hidden_benchmark_can_promote_only_to_canary(self) -> None:
        report = _gate()

        self.assertEqual(report["decision"], "promote_to_canary")
        self.assertFalse(report["production_promotion_allowed"])
        self.assertFalse(report["required_failures"])
        self.assertTrue(
            report["metric_results"]["objective_score"]["promotion_signal_passed"]
        )
        self.assertTrue(
            validate_evolution_gate(report, constitution=_constitution())["passed"]
        )

    def test_gate_requires_prospective_work_and_stack_diversity(self) -> None:
        sealed_only = [_sample(index + 1) for index in range(5)]
        for sample in sealed_only:
            sample["evaluator_stack_id"] = "stack:research-producer"
        report = _gate(samples=sealed_only)
        self.assertEqual(report["decision"], "hold")
        self.assertIn("prospective_tasks", report["required_failures"])
        self.assertIn("independent_evaluator_stacks", report["required_failures"])

    def test_public_or_regressed_benchmark_is_held(self) -> None:
        samples = [_sample(index) for index in range(5)]
        samples[0] = _sample(0, hidden=False, regressed=True)
        report = _gate(samples=samples)

        self.assertEqual(report["decision"], "hold")
        self.assertIn("hidden_benchmark", report["required_failures"])
        self.assertIn("safety_regression", report["required_failures"])
        self.assertIn("metric_regression", report["required_failures"])

    def test_production_requires_canary_rollback_and_independent_approval(
        self,
    ) -> None:
        gate = _gate()
        canary = _canary(gate["candidate"])

        blocked = approve_production_promotion(
            gate,
            canary,
            constitution=_constitution(),
            approver_id="agent:evolution",
        )
        self.assertEqual(blocked["decision"], "blocked")
        self.assertIn("independent_approval", blocked["required_failures"])

        approved = approve_production_promotion(
            gate,
            canary,
            constitution=_constitution(),
            approver_id="human:release-controller",
        )
        self.assertEqual(approved["decision"], "approved")
        self.assertTrue(approved["production_promotion_allowed"])
        self.assertEqual(approved["rollback_ref"], "git:base-commit")
        self.assertTrue(
            validate_production_promotion(approved, constitution=_constitution())[
                "passed"
            ]
        )

        gate["metric_results"]["objective_score"]["confidence_lower_bound"] = 9.0
        tampered = approve_production_promotion(
            gate,
            canary,
            constitution=_constitution(),
            approver_id="human:release-controller",
        )
        self.assertEqual(tampered["decision"], "blocked")
        self.assertIn("shadow_gate_integrity", tampered["required_failures"])

    def test_high_risk_promotion_requires_two_independent_humans(self) -> None:
        candidate = _candidate(risk_tier="high")
        gate = _gate(candidate=candidate)
        canary = _canary(candidate)
        one_approver = approve_production_promotion(
            gate,
            canary,
            constitution=_constitution(),
            approver_ids=["human:release-controller"],
        )
        self.assertIn("independent_approval", one_approver["required_failures"])

        two_approvers = approve_production_promotion(
            gate,
            canary,
            constitution=_constitution(),
            approver_ids=["human:release-controller", "human:safety-owner"],
        )
        self.assertEqual(two_approvers["decision"], "approved")

    def test_canary_rejects_non_finite_metrics(self) -> None:
        gate = _gate()
        candidate = gate["candidate"]
        rollback = build_rollback_receipt(
            candidate,
            restored_artifact_hash=candidate["base_artifact_hash"],
            execution_log_hash=_digest("rollback-log"),
            executed_by="service:release-canary",
        )
        canary = build_canary_report(
            candidate,
            rollback,
            executed_by="service:release-canary",
            observation_count=25,
            error_rate_delta=float("-inf"),
            quality_delta=float("inf"),
            real_research_project_ids=["project:a", "project:b", "project:c"],
            project_run_hashes={
                project_id: _digest(project_id)
                for project_id in ["project:a", "project:b", "project:c"]
            },
            incidents=[],
            long_tail_pass=True,
            common_mode_failure_pass=True,
            out_of_distribution_pass=True,
        )

        report = approve_production_promotion(
            gate,
            canary,
            constitution=_constitution(),
            approver_id="human:release-controller",
        )
        self.assertEqual(report["decision"], "blocked")
        self.assertIn("canary_error_rate", report["required_failures"])
        self.assertIn("canary_quality", report["required_failures"])

    def test_gate_save_persists_latest_decision_and_append_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            constitution = _constitution()
            save_science_constitution(root, constitution, producer="test")
            report = _gate()
            output = save_evolution_gate(
                root, report, constitution=constitution, producer="test"
            )

            self.assertTrue(Path(output).exists())
            manifest = load_pipeline_manifest(root)
            self.assertEqual(manifest["artifacts"]["evolution_gate"]["status"], "ready")
            history = root / "evolution_gate_history.jsonl"
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual(rows[0]["decision"], "promote_to_canary")

    @unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
    def test_research_vcs_blocks_shadow_candidate_from_stable_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = ResearchRepository.init(
                Path(td) / "research",
                question="Can the agent improve without weakening integrity?",
                git_user_name="Evolution Test",
                git_user_email="evolution@example.invalid",
            )
            evolution = ResearchEvolution(repository)
            evolution.candidate_line("search-policy-v2")
            evolution.candidate(_candidate(), constitution=_constitution())
            repository.switch("main")

            preview = repository.merge_preview("evolve/search-policy-v2")

            self.assertFalse(preview["clean"])
            self.assertIn(
                "ungated_agent_candidate",
                {item["type"] for item in preview["conflicts"]},
            )

    @unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
    def test_candidate_api_refuses_stable_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = ResearchRepository.init(
                Path(td) / "research",
                git_user_name="Evolution Test",
                git_user_email="evolution@example.invalid",
            )

            with self.assertRaisesRegex(ResearchGitError, r"evolve/\*"):
                ResearchEvolution(repository).candidate(
                    _candidate(),
                    constitution=_constitution(),
                )

    @unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
    def test_approved_candidate_can_merge_and_rollback_is_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = ResearchRepository.init(
                Path(td) / "research",
                question="Can the agent improve without weakening integrity?",
                git_user_name="Evolution Test",
                git_user_email="evolution@example.invalid",
            )
            evolution = ResearchEvolution(repository)
            evolution.candidate_line("search-policy-v2")
            candidate_payload = _candidate()
            candidate = evolution.candidate(
                candidate_payload,
                constitution=_constitution(),
            )["candidate"]
            gate_payload = _gate(candidate=candidate_payload)
            evaluation = evolution.evaluate(
                gate_payload,
                constitution=_constitution(),
                candidate_id=candidate.object_id,
                evaluator_id="service:independent-evaluator",
            )["evaluation"]
            canary = _canary(candidate_payload)
            promotion_payload = approve_production_promotion(
                gate_payload,
                canary,
                constitution=_constitution(),
                approver_ids=["human:release-owner"],
            )
            promoted = evolution.promote(
                promotion_payload,
                constitution=_constitution(),
                candidate_id=candidate.object_id,
                evaluation_id=evaluation.object_id,
            )["promoted_candidate"]
            repository.switch("main")

            preview = repository.merge_preview("evolve/search-policy-v2")
            merged = repository.merge("evolve/search-policy-v2")
            rollback = ResearchEvolution(repository).rollback(
                canary["rollback_receipt"],
                candidate_id=candidate.object_id,
                promoted_id=promoted.object_id,
                trigger="canary_quality_regression",
            )

            self.assertTrue(preview["clean"], preview["conflicts"])
            self.assertTrue(merged.commit)
            self.assertEqual(rollback["decision"].state, "superseded")
            self.assertTrue(repository.fsck()["ok"])

    @unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
    def test_held_evolution_evaluation_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = ResearchRepository.init(
                Path(td) / "research",
                git_user_name="Evolution Test",
                git_user_email="evolution@example.invalid",
            )
            evolution = ResearchEvolution(repository)
            evolution.candidate_line("held-search-policy")
            candidate_payload = _candidate()
            candidate = evolution.candidate(
                candidate_payload,
                constitution=_constitution(),
            )["candidate"]
            held_gate = _gate(
                candidate=candidate_payload,
                samples=[_sample(index, regressed=True) for index in range(5)],
            )
            evaluation = evolution.evaluate(
                held_gate,
                constitution=_constitution(),
                candidate_id=candidate.object_id,
                evaluator_id="service:independent-evaluator",
            )["evaluation"]

            with self.assertRaisesRegex(ResearchGitError, "verified independent"):
                evolution.promote(
                    {},
                    constitution=_constitution(),
                    candidate_id=candidate.object_id,
                    evaluation_id=evaluation.object_id,
                )

    def test_public_cli_runs_shadow_gate_and_persists_report(self) -> None:
        from xscientist.cli import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            save_science_constitution(root, _constitution(), producer="test")
            candidate_path = Path(td) / "candidate.json"
            benchmark_path = Path(td) / "benchmark.json"
            ablation_path = Path(td) / "ablation.json"
            candidate_path.write_text(
                json.dumps(
                    {
                        "candidate_id": "search-policy-v2",
                        "component_type": "search_policy",
                        "base_version": "1.0.0",
                        "candidate_version": "1.1.0",
                        "base_artifact_hash": _digest("a"),
                        "candidate_artifact_hash": _digest("b"),
                        "rollback_ref": "git:base-commit",
                        "proposed_by": "agent:evolution",
                        "change_summary": "Improve falsification search.",
                        "change_scope": ["search/falsification-budget"],
                        "applicability_domains": ["general"],
                        "failure_taxonomy_refs": ["failure:premature-convergence"],
                        "ablation_dimensions": ["falsification-budget"],
                        "provenance_hashes": [_digest("candidate-provenance")],
                    }
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(
                json.dumps([_sample(index) for index in range(5)]),
                encoding="utf-8",
            )
            ablation_path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": f"ablation-{index}",
                            "dimension": "falsification-budget",
                            "full_candidate_score": 0.66,
                            "ablated_score": 0.60,
                            "full_run_hash": _digest(f"full-ablation-{index}"),
                            "ablated_run_hash": _digest(f"removed-ablation-{index}"),
                        }
                        for index in range(3)
                    ]
                ),
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "evolution-gate",
                        "--project-root",
                        str(root),
                        "--candidate",
                        str(candidate_path),
                        "--benchmark",
                        str(benchmark_path),
                        "--ablation",
                        str(ablation_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads((root / "evolution_gate.json").read_text())
            self.assertEqual(report["decision"], "promote_to_canary")


if __name__ == "__main__":
    unittest.main()
