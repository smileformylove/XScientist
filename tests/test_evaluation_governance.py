from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.epistemic_graph import (
    advance_epistemic_node,
    build_epistemic_graph,
)
from ai_scientist.utils.evaluation_governance import (
    EvaluationGovernanceError,
    build_benchmark_manifest,
    build_evaluation_charter,
    build_evaluation_decision,
    build_evaluation_run,
    save_benchmark_manifest,
    save_evaluation_charter,
    save_evaluation_report,
    validate_benchmark_manifest,
    validate_evaluation_decision,
    validate_evaluation_run,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_jsonl_artifact,
    load_pipeline_manifest,
)
from ai_scientist.utils.science_constitution import (
    build_science_constitution,
    save_science_constitution,
)
from ai_scientist.utils.stage_standards import _evaluate_ideation


def _digest(char: str) -> str:
    return "sha256:" + hashlib.sha256(char.encode("utf-8")).hexdigest()


def _constitution() -> dict:
    return build_science_constitution(project_name="grand-discovery")


def _charter() -> dict:
    return build_evaluation_charter(
        _constitution(),
        assignments={
            "researcher": ["agent:researcher"],
            "verifier": ["service:internal-verifier", "lab:external-verifier"],
            "benchmark_custodian": ["service:benchmark-custodian"],
            "approver": ["human:principal-investigator"],
        },
    )


def _manifest(layer: str, *, candidate_hash: str = _digest("a")) -> dict:
    task_hash_character = {
        "public": "a",
        "sealed": "b",
        "prospective": "c",
        "external": "d",
    }[layer]
    kwargs = {}
    if layer == "sealed":
        kwargs["answer_key_hash"] = _digest("e")
    elif layer == "prospective":
        kwargs.update(
            {
                "resolution_condition": "Outcome is published by the instrument.",
                "resolution_not_before": "2020-01-01T00:00:00Z",
            }
        )
    elif layer == "external":
        kwargs.update(
            {
                "external_organization_id": "lab:independent-institute",
                "external_protocol_hash": _digest("f"),
            }
        )
    return build_benchmark_manifest(
        _charter(),
        benchmark_id=f"benchmark-{layer}",
        layer=layer,
        candidate_hash=candidate_hash,
        task_hashes=[_digest(task_hash_character)],
        custodian_id="service:benchmark-custodian",
        **kwargs,
    )


def _passing_results(*, prospective: bool = False) -> dict:
    return {
        "integrity_pass": True,
        "safety_pass": True,
        "reproducibility_pass": True,
        "prospective_resolved": prospective,
        "metrics": {
            "objective_quality": 0.80,
            "worst_domain_quality": 0.70,
            "reproducibility_rate": 0.95,
            "false_discovery_rate": 0.02,
            "calibration_error": 0.05,
            "information_gain": 0.20,
        },
    }


def _run(layer: str) -> dict:
    verifier = (
        "lab:external-verifier" if layer == "external" else "service:internal-verifier"
    )
    kwargs = {}
    if layer == "sealed":
        kwargs["custodian_attestation_hash"] = _digest("c")
    elif layer == "prospective":
        kwargs["resolution_attestation_hash"] = _digest("r")
    elif layer == "external":
        kwargs["external_attestation_hash"] = _digest("e")
    return build_evaluation_run(
        _charter(),
        _manifest(layer),
        evaluator_id=verifier,
        results=_passing_results(prospective=layer == "prospective"),
        evaluator_input_hash=_digest("i"),
        evaluator_result_hash=_digest("o"),
        **kwargs,
    )


def _approved_decision(*, scope_node_ids: list[str]) -> dict:
    return build_evaluation_decision(
        _charter(),
        [_run("sealed"), _run("prospective"), _run("external")],
        candidate_hash=_digest("a"),
        candidate_producer_id="agent:researcher",
        approver_id="human:principal-investigator",
        scope_node_ids=scope_node_ids,
    )


class EvaluationGovernanceTests(unittest.TestCase):
    def test_charter_rejects_role_overlap_and_non_human_approver(self) -> None:
        assignments = {
            "researcher": ["agent:same"],
            "verifier": ["agent:same", "lab:external"],
            "benchmark_custodian": ["service:custodian"],
            "approver": ["agent:approver"],
        }
        with self.assertRaises(EvaluationGovernanceError):
            build_evaluation_charter(_constitution(), assignments=assignments)

    def test_benchmark_manifest_keeps_tasks_and_answers_opaque(self) -> None:
        manifest = _manifest("sealed")
        self.assertFalse(manifest["raw_tasks_in_manifest"])
        self.assertFalse(manifest["raw_answers_in_manifest"])
        self.assertTrue(manifest["answer_key_hash"].startswith("sha256:"))
        self.assertNotIn("answers", manifest)
        manifest["raw_answers"] = ["forbidden"]
        check = validate_benchmark_manifest(manifest, charter=_charter())
        self.assertIn("benchmark_fields_invalid", check["errors"])

    def test_prospective_and_external_manifests_require_custody_metadata(self) -> None:
        with self.assertRaises(EvaluationGovernanceError):
            build_benchmark_manifest(
                _charter(),
                benchmark_id="future",
                layer="prospective",
                candidate_hash=_digest("a"),
                task_hashes=[_digest("t")],
                custodian_id="service:benchmark-custodian",
            )
        with self.assertRaises(EvaluationGovernanceError):
            build_benchmark_manifest(
                _charter(),
                benchmark_id="external",
                layer="external",
                candidate_hash=_digest("a"),
                task_hashes=[_digest("t")],
                custodian_id="service:benchmark-custodian",
                external_organization_id="lab:unknown",
            )

    def test_run_rejects_metric_regression_and_detects_semantic_tampering(self) -> None:
        results = _passing_results()
        results["metrics"]["calibration_error"] = 0.90
        run = build_evaluation_run(
            _charter(),
            _manifest("sealed"),
            evaluator_id="service:internal-verifier",
            results=results,
            evaluator_input_hash=_digest("i"),
            evaluator_result_hash=_digest("o"),
            custodian_attestation_hash=_digest("c"),
        )
        self.assertEqual(run["decision"], "blocked")

        passing = _run("sealed")
        self.assertTrue(validate_evaluation_run(passing, charter=_charter())["passed"])
        passing["metric_results"]["calibration_error"]["value"] = 0.99
        self.assertFalse(validate_evaluation_run(passing, charter=_charter())["passed"])

    def test_prospective_run_cannot_resolve_before_frozen_time(self) -> None:
        manifest = build_benchmark_manifest(
            _charter(),
            benchmark_id="future",
            layer="prospective",
            candidate_hash=_digest("a"),
            task_hashes=[_digest("future-task")],
            custodian_id="service:benchmark-custodian",
            resolution_condition="A future instrument publishes the outcome.",
            resolution_not_before="2999-01-01T00:00:00Z",
        )
        run = build_evaluation_run(
            _charter(),
            manifest,
            evaluator_id="service:internal-verifier",
            results=_passing_results(prospective=True),
            evaluator_input_hash=_digest("future-input"),
            evaluator_result_hash=_digest("future-result"),
            resolution_attestation_hash=_digest("future-attestation"),
        )
        self.assertEqual(run["decision"], "blocked")
        self.assertIn("prospective_embargo_elapsed", run["required_failures"])

    def test_decision_requires_all_layers_and_external_verifier(self) -> None:
        blocked = build_evaluation_decision(
            _charter(),
            [_run("sealed"), _run("prospective")],
            candidate_hash=_digest("a"),
            candidate_producer_id="agent:researcher",
            approver_id="human:principal-investigator",
            scope_node_ids=["hypothesis:demo"],
        )
        self.assertEqual(blocked["decision"], "blocked")
        self.assertIn("layer:external", blocked["required_failures"])

        approved = _approved_decision(scope_node_ids=["hypothesis:demo"])
        self.assertEqual(approved["decision"], "approved")
        self.assertTrue(approved["claim_promotion_allowed"])
        self.assertTrue(validate_evaluation_decision(approved)["passed"])

    def test_decision_detects_post_hoc_criterion_rewrite(self) -> None:
        report = _approved_decision(scope_node_ids=["hypothesis:demo"])
        report["criteria"][0]["detail"] = "rewritten after approval"
        self.assertFalse(validate_evaluation_decision(report)["passed"])

    def test_robust_epistemic_transition_requires_scoped_approval(self) -> None:
        constitution = _constitution()
        graph = build_epistemic_graph(
            [
                {
                    "idea_id": "idea_0",
                    "title": "A hypothesis",
                    "core_hypothesis": "A causes B.",
                    "failure_criteria": ["B is absent under A."],
                }
            ],
            constitution=constitution,
            producer="agent:researcher",
        )
        node_id = next(
            node["node_id"]
            for node in graph["nodes"]
            if node["node_type"] == "hypothesis"
        )
        transitions = [
            ("grounded", ["literature:review"]),
            ("preregistered", ["protocol:locked"]),
            ("tested", ["experiment:confirmatory"]),
            ("replicated", ["replication:independent"]),
        ]
        for status, refs in transitions:
            graph = advance_epistemic_node(
                graph,
                node_id=node_id,
                to_status=status,
                actor_id="service:internal-verifier",
                reason=f"Advance to {status}.",
                evidence_refs=refs,
            )
        with self.assertRaises(EvaluationGovernanceError):
            advance_epistemic_node(
                graph,
                node_id=node_id,
                to_status="robust",
                actor_id="human:principal-investigator",
                reason="Attempt without governed evaluation.",
                evidence_refs=["evaluation:sealed", "evaluation:external"],
            )

        approved = _approved_decision(scope_node_ids=[node_id])
        robust = advance_epistemic_node(
            graph,
            node_id=node_id,
            to_status="robust",
            actor_id="human:principal-investigator",
            reason="All governed layers passed.",
            evidence_refs=["evaluation:prospective", "evaluation:external"],
            evaluation_report=approved,
        )
        self.assertEqual(robust["transitions"][-1]["to_status"], "robust")

        idea_cards = [
            {
                "idea_id": "idea_0",
                "core_hypothesis": "A causes B.",
                "novelty_claim": "Tests a new causal mechanism.",
                "minimum_viable_experiment": "Intervene on A and measure B.",
                "candidate_datasets": ["dataset:a-b"],
                "candidate_metrics": ["effect_size"],
                "candidate_baselines": ["no_intervention"],
                "failure_criteria": ["B is absent under A."],
            }
        ]
        blocked_stage = _evaluate_ideation(idea_cards, {}, constitution, robust, {}, {})
        blocked_criteria = {item["id"]: item for item in blocked_stage["criteria"]}
        self.assertFalse(blocked_criteria["independent_evaluation"]["passed"])

        governed_stage = _evaluate_ideation(
            idea_cards, {}, constitution, robust, _charter(), approved
        )
        governed_criteria = {item["id"]: item for item in governed_stage["criteria"]}
        self.assertTrue(governed_criteria["evaluation_charter"]["passed"])
        self.assertTrue(governed_criteria["independent_evaluation"]["passed"])
        self.assertTrue(governed_criteria["evaluation_scope"]["passed"])

    def test_governance_artifacts_persist_with_append_only_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            constitution = _constitution()
            save_science_constitution(root, constitution, producer="test")
            charter = _charter()
            save_evaluation_charter(root, charter, producer="test")
            save_benchmark_manifest(
                root,
                _manifest("sealed"),
                charter=charter,
                producer="service:benchmark-custodian",
            )
            save_benchmark_manifest(
                root,
                _manifest("prospective"),
                charter=charter,
                producer="service:benchmark-custodian",
            )
            report = _approved_decision(scope_node_ids=["hypothesis:demo"])
            save_evaluation_report(
                root, report, producer="human:principal-investigator"
            )

            rows = load_jsonl_artifact(root / "evaluation_benchmarks.jsonl")
            self.assertEqual(len(rows), 2)
            manifest = load_pipeline_manifest(root)
            self.assertEqual(
                manifest["artifacts"]["evaluation_report"]["status"],
                "ready",
            )


if __name__ == "__main__":
    unittest.main()
