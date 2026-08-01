from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ai_scientist.utils.evolution_gate import (
    EvolutionGateError,
    approve_production_promotion,
    build_evolution_candidate,
    build_evolution_gate,
    save_evolution_gate,
    validate_evolution_candidate,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_pipeline_manifest,
)


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _candidate() -> dict:
    return build_evolution_candidate(
        candidate_id="search-policy-v2",
        component_type="search_policy",
        base_version="1.0.0",
        candidate_version="1.1.0",
        base_artifact_hash=_digest("a"),
        candidate_artifact_hash=_digest("b"),
        rollback_ref="git:base-commit",
        proposed_by="evolution-agent",
        change_summary="Allocate more search budget to falsification branches.",
    )


def _sample(index: int, *, hidden: bool = True, regressed: bool = False) -> dict:
    return {
        "task_id": f"hidden-task-{index}",
        "split": "hidden" if hidden else "public",
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


class EvolutionGateTests(unittest.TestCase):
    def test_candidate_requires_distinct_content_addressed_artifacts(self) -> None:
        with self.assertRaises(EvolutionGateError):
            build_evolution_candidate(
                candidate_id="same",
                component_type="prompt",
                base_version="1",
                candidate_version="2",
                base_artifact_hash=_digest("a"),
                candidate_artifact_hash=_digest("a"),
                rollback_ref="git:base",
                proposed_by="agent",
                change_summary="No real change.",
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
            build_evolution_gate(
                _candidate(),
                [_sample(index) for index in range(5)],
                policy={"require_integrity_pass": False},
            )
        with self.assertRaises(EvolutionGateError):
            build_evolution_gate(
                _candidate(),
                [_sample(index) for index in range(5)],
                policy={"minimum_hidden_tasks": 1},
            )
        with self.assertRaises(EvolutionGateError):
            build_evolution_gate(
                _candidate(),
                [_sample(index) for index in range(5)],
                policy={
                    "metric_specs": {"objective_score": {"minimum_improvement": -1.0}}
                },
            )

    def test_hidden_benchmark_can_promote_only_to_canary(self) -> None:
        report = build_evolution_gate(
            _candidate(),
            [_sample(index) for index in range(5)],
        )

        self.assertEqual(report["decision"], "promote_to_canary")
        self.assertFalse(report["production_promotion_allowed"])
        self.assertFalse(report["required_failures"])
        self.assertTrue(
            report["metric_results"]["objective_score"]["promotion_signal_passed"]
        )

    def test_public_or_regressed_benchmark_is_held(self) -> None:
        samples = [_sample(index) for index in range(5)]
        samples[0] = _sample(0, hidden=False, regressed=True)
        report = build_evolution_gate(_candidate(), samples)

        self.assertEqual(report["decision"], "hold")
        self.assertIn("hidden_benchmark", report["required_failures"])
        self.assertIn("safety_regression", report["required_failures"])
        self.assertIn("metric_regression", report["required_failures"])

    def test_production_requires_canary_rollback_and_independent_approval(
        self,
    ) -> None:
        gate = build_evolution_gate(
            _candidate(),
            [_sample(index) for index in range(5)],
        )
        canary = {
            "candidate_hash": gate["candidate"]["candidate_hash"],
            "status": "passed",
            "observation_count": 25,
            "error_rate_delta": -0.01,
            "quality_delta": 0.02,
            "incidents": [],
            "rollback_tested": True,
        }

        blocked = approve_production_promotion(
            gate, canary, approver_id="evolution-agent"
        )
        self.assertEqual(blocked["decision"], "blocked")
        self.assertIn("independent_approval", blocked["required_failures"])

        approved = approve_production_promotion(
            gate, canary, approver_id="release-controller"
        )
        self.assertEqual(approved["decision"], "approved")
        self.assertTrue(approved["production_promotion_allowed"])
        self.assertEqual(approved["rollback_ref"], "git:base-commit")

        gate["metric_results"]["objective_score"]["confidence_lower_bound"] = 9.0
        tampered = approve_production_promotion(
            gate, canary, approver_id="release-controller"
        )
        self.assertEqual(tampered["decision"], "blocked")
        self.assertIn("shadow_gate_integrity", tampered["required_failures"])

    def test_canary_rejects_non_finite_metrics(self) -> None:
        gate = build_evolution_gate(
            _candidate(),
            [_sample(index) for index in range(5)],
        )
        canary = {
            "candidate_hash": gate["candidate"]["candidate_hash"],
            "status": "passed",
            "observation_count": 25,
            "error_rate_delta": float("-inf"),
            "quality_delta": float("inf"),
            "incidents": [],
            "rollback_tested": True,
        }

        report = approve_production_promotion(
            gate, canary, approver_id="release-controller"
        )
        self.assertEqual(report["decision"], "blocked")
        self.assertIn("canary_error_rate", report["required_failures"])
        self.assertIn("canary_quality", report["required_failures"])

    def test_gate_save_persists_latest_decision_and_append_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            report = build_evolution_gate(
                _candidate(),
                [_sample(index) for index in range(5)],
            )
            output = save_evolution_gate(root, report, producer="test")

            self.assertTrue(Path(output).exists())
            manifest = load_pipeline_manifest(root)
            self.assertEqual(manifest["artifacts"]["evolution_gate"]["status"], "ready")
            history = root / "evolution_gate_history.jsonl"
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual(rows[0]["decision"], "promote_to_canary")

    def test_public_cli_runs_shadow_gate_and_persists_report(self) -> None:
        from xscientist.cli import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            candidate_path = Path(td) / "candidate.json"
            benchmark_path = Path(td) / "benchmark.json"
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
                        "proposed_by": "evolution-agent",
                        "change_summary": "Improve falsification search.",
                    }
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(
                json.dumps([_sample(index) for index in range(5)]),
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
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads((root / "evolution_gate.json").read_text())
            self.assertEqual(report["decision"], "promote_to_canary")


if __name__ == "__main__":
    unittest.main()
