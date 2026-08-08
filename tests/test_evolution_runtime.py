from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ai_scientist.protocol.attestation import (
    sign_attestation,
    verify_attestation,
    verify_authorization_bundle,
)
from ai_scientist.protocol.canonical_json import (
    CanonicalJSONError,
    canonical_content_hash,
    canonical_json,
)
from ai_scientist.protocol.schemas import schema_validator
from ai_scientist.utils.evolution_artifacts import (
    EvolutionArtifactError,
    build_evolution_candidate_from_sources,
    verify_evolution_artifact,
)
from ai_scientist.utils.evolution_deployment import (
    LocalEvolutionDeployment,
    deploy_approved_candidate,
    validate_deployment_receipt,
)
from ai_scientist.utils.evolution_gate import (
    approve_production_promotion,
    build_ablation_report,
    build_evolution_gate,
)
from ai_scientist.utils.evolution_runtime import (
    BENCHMARK_SUITE_SCHEMA,
    CANARY_SPEC_SCHEMA,
    EvolutionRuntimeError,
    run_canary_suite,
    run_shadow_benchmark,
)
from ai_scientist.utils.science_constitution import build_science_constitution


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _candidate_fixture(root: Path) -> tuple[dict, dict, Path]:
    base = root / "base"
    changed = root / "changed"
    (base / "search").mkdir(parents=True)
    (changed / "search").mkdir(parents=True)
    (base / "search" / "policy.json").write_text(
        json.dumps({"score": 0.60}), encoding="utf-8"
    )
    (changed / "search" / "policy.json").write_text(
        json.dumps({"score": 0.66}), encoding="utf-8"
    )
    store = root / "artifact-store"
    constitution = build_science_constitution(project_name="runtime-test")
    result = build_evolution_candidate_from_sources(
        constitution=constitution,
        base_root=base,
        candidate_root=changed,
        store_root=store,
        candidate_id="search-v2",
        component_type="search_policy",
        base_version="1.0.0",
        candidate_version="1.1.0",
        proposed_by="agent:evolution",
        change_summary="Increase falsification search score.",
        change_scope=["search/policy.json"],
        applicability_domains=["general"],
        failure_taxonomy_refs=["failure:premature-convergence"],
        ablation_dimensions=["policy-score"],
        provenance_hashes=[_digest("intent")],
    )
    return result, constitution, store


def _evaluator(path: Path) -> Path:
    script = path / "evaluator.py"
    script.write_text(
        """
import json
import os
from pathlib import Path

artifact = Path(os.environ["XSCIENTIST_ARTIFACT_ROOT"])
score = json.loads((artifact / "policy.json").read_text())["score"]
payload = {
    "metrics": {
        "objective_score": score,
        "reproducibility_rate": 0.95,
        "false_discovery_rate": 0.05,
        "cost_per_task": 1.0,
        "latency_seconds": 1.0
    },
    "safety_pass": True,
    "integrity_pass": True,
    "reproducibility_pass": True
}
print(json.dumps(payload))
""".strip() + "\n",
        encoding="utf-8",
    )
    return script


def _canary_evaluator(path: Path) -> Path:
    script = path / "canary.py"
    script.write_text(
        """
import json
import os
from pathlib import Path

artifact = Path(os.environ["XSCIENTIST_ARTIFACT_ROOT"])
score = json.loads((artifact / "policy.json").read_text())["score"]
print(json.dumps({
    "metrics": {"error_rate": 0.01, "quality": score},
    "safety_pass": True,
    "integrity_pass": True,
    "reproducibility_pass": True,
    "observations": 10,
    "incidents": [],
    "long_tail_pass": True,
    "common_mode_failure_pass": True,
    "out_of_distribution_pass": True
}))
""".strip() + "\n",
        encoding="utf-8",
    )
    return script


def _failing_evaluator(path: Path) -> Path:
    script = path / "failing_evaluator.py"
    script.write_text(
        "import sys\nprint('controlled failure', file=sys.stderr)\nraise SystemExit(4)\n",
        encoding="utf-8",
    )
    return script


class CanonicalJSONTests(unittest.TestCase):
    def test_cross_language_number_and_key_profile(self) -> None:
        payload = {"z": 1.0, "😀": 1e-7, "\ue000": 1e20, "a": -0.0}
        self.assertEqual(
            canonical_json(payload),
            '{"a":0,"z":1,"😀":1e-7,"\ue000":100000000000000000000}',
        )
        self.assertTrue(canonical_content_hash(payload).startswith("sha256:"))
        with self.assertRaises(CanonicalJSONError):
            canonical_json({"unsafe": 2**60})
        with self.assertRaises(CanonicalJSONError):
            canonical_json({"nan": float("nan")})

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_non_python_conformance_consumer(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                "node",
                str(
                    root
                    / "ai_scientist"
                    / "protocol"
                    / "conformance"
                    / "verify-canonical-json.mjs"
                ),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(json.loads(completed.stdout)["ok"])


class AttestationTests(unittest.TestCase):
    def test_hmac_attestation_binds_identity_purpose_and_payload(self) -> None:
        payload = {"candidate_hash": _digest("candidate")}
        trust = {
            "key:evaluator": {
                "identity": "service:evaluator",
                "algorithm": "hmac-sha256",
                "key": b"test-secret",
            }
        }
        envelope = sign_attestation(
            payload,
            purpose="independent_benchmark",
            identity="service:evaluator",
            key_id="key:evaluator",
            key=b"test-secret",
            issued_at="2026-08-08T00:00:00+00:00",
        )
        schema_validator("attestation").validate(envelope)
        verified = verify_attestation(
            envelope,
            payload,
            trust_store=trust,
            purpose="independent_benchmark",
            now=datetime(2026, 8, 8, 0, 1, tzinfo=timezone.utc),
            max_age_seconds=120,
        )
        self.assertTrue(verified["ok"], verified)
        tampered = verify_attestation(
            envelope,
            {"candidate_hash": _digest("tampered")},
            trust_store=trust,
        )
        self.assertIn("payload_hash_mismatch", tampered["errors"])

    def test_authorization_requires_independent_evidence_and_human(self) -> None:
        candidate = {
            "candidate_hash": _digest("candidate"),
            "proposed_by": "agent:evolution",
        }
        promotion = {
            "promotion_hash": _digest("promotion"),
            "decision": "approved",
            "production_promotion_allowed": True,
            "gate_report": {"gate_hash": _digest("gate")},
            "canary_report": {"canary_hash": _digest("canary")},
        }
        payloads = {
            "candidate_artifact": candidate,
            "independent_benchmark": promotion["gate_report"],
            "canary_execution": promotion["canary_report"],
            "production_approval": {
                "candidate_hash": candidate["candidate_hash"],
                "promotion_hash": promotion["promotion_hash"],
                "decision": "approved",
            },
        }
        identities = {
            "candidate_artifact": "agent:evolution",
            "independent_benchmark": "service:evaluator",
            "canary_execution": "service:canary",
            "production_approval": "human:release",
        }
        trust = {}
        attestations = []
        for purpose, payload in payloads.items():
            identity = identities[purpose]
            key_id = "key:" + purpose
            trust[key_id] = {
                "identity": identity,
                "algorithm": "hmac-sha256",
                "key": purpose.encode(),
            }
            attestations.append(
                sign_attestation(
                    payload,
                    purpose=purpose,
                    identity=identity,
                    key_id=key_id,
                    key=purpose.encode(),
                )
            )
        result = verify_authorization_bundle(
            {"attestations": attestations},
            candidate=candidate,
            promotion=promotion,
            trust_store=trust,
        )
        self.assertTrue(result["ok"], result)
        without_human = verify_authorization_bundle(
            {"attestations": attestations[:-1]},
            candidate=candidate,
            promotion=promotion,
            trust_store=trust,
        )
        self.assertFalse(without_human["ok"])


class EvolutionExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = tempfile.mkdtemp(prefix="xscientist-evolution-runtime-")
        self.root = Path(self.raw)
        self.build, self.constitution, self.store = _candidate_fixture(self.root)
        self.candidate = self.build["candidate"]

    def tearDown(self) -> None:
        shutil.rmtree(self.raw)

    def test_candidate_artifacts_are_immutable_and_diff_bound(self) -> None:
        for field in ("base_artifact", "candidate_artifact"):
            artifact = self.build[field]
            check = verify_evolution_artifact(self.store, artifact["artifact_hash"])
            self.assertTrue(check["ok"], check)
            schema_validator("evolution_artifact").validate(check["manifest"])
        self.assertEqual(self.build["change_set"]["modified"], ["search/policy.json"])
        candidate_file = (
            self.store
            / "objects"
            / self.candidate["candidate_artifact_hash"].split(":", 1)[1]
            / "files"
            / "search"
            / "policy.json"
        )
        candidate_file.write_text("tampered", encoding="utf-8")
        check = verify_evolution_artifact(
            self.store, self.candidate["candidate_artifact_hash"]
        )
        self.assertFalse(check["ok"])
        self.assertTrue(
            any(error.startswith("entry_hash_mismatch") for error in check["errors"])
        )

    def test_candidate_builder_rejects_undeclared_changes(self) -> None:
        changed = self.root / "other-change"
        shutil.copytree(self.root / "base", changed)
        (changed / "search" / "extra.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(EvolutionArtifactError):
            build_evolution_candidate_from_sources(
                constitution=self.constitution,
                base_root=self.root / "base",
                candidate_root=changed,
                store_root=self.root / "other-store",
                candidate_id="bad",
                component_type="search_policy",
                base_version="1",
                candidate_version="2",
                proposed_by="agent:evolution",
                change_summary="Undeclared file.",
                change_scope=["search/policy.json"],
                applicability_domains=["general"],
                failure_taxonomy_refs=["failure:scope"],
                ablation_dimensions=["policy-score"],
            )

    def _benchmark_suite(self, evaluator: Path) -> dict:
        tasks = []
        for index in range(5):
            prospective = index == 0
            task = {
                "task_id": f"task-{index}",
                "task_hash": _digest(f"task-{index}"),
                "split": "hidden",
                "evaluation_layer": "prospective" if prospective else "sealed",
                "frozen_before_candidate": True,
                "benchmark_frozen_at": "2020-01-01T00:00:00Z",
                "domain": "general",
                "evaluator_stack_id": f"stack:evaluator-{index % 2}",
                "evaluator_stack_hash": _digest(f"evaluator-{index % 2}"),
                "command": [sys.executable, str(evaluator)],
            }
            if prospective:
                task.update(
                    {
                        "prospective_resolved": True,
                        "prospective_protocol_hash": _digest("protocol"),
                        "resolution_attestation_hash": _digest("resolution"),
                        "resolution_not_before": "2020-01-02T00:00:00Z",
                        "resolved_at": "2020-01-03T00:00:00Z",
                    }
                )
            else:
                task["custodian_attestation_hash"] = _digest(f"custody-{index}")
            tasks.append(task)
        return {
            "schema_version": BENCHMARK_SUITE_SCHEMA,
            "benchmark_id": "sealed-v1",
            "producer_stack_id": "stack:producer",
            "producer_stack_hash": _digest("producer"),
            "evaluator_stack_id": "stack:evaluator-0",
            "evaluator_stack_hash": _digest("evaluator-0"),
            "evaluator_id": "service:benchmark",
            "tasks": tasks,
        }

    def test_shadow_runner_produces_gate_consumable_evidence(self) -> None:
        evaluator = _evaluator(self.root)
        suite = self._benchmark_suite(evaluator)
        with self.assertRaises(EvolutionRuntimeError):
            run_shadow_benchmark(
                suite,
                self.candidate,
                artifact_store=self.store,
            )
        report = run_shadow_benchmark(
            suite,
            self.candidate,
            artifact_store=self.store,
            allow_execution=True,
        )
        schema_validator("benchmark_run").validate(report)
        self.assertEqual(report["task_count"], 5)
        self.assertEqual(len(report["run_receipts"]), 10)
        ablation = build_ablation_report(
            self.candidate,
            [
                {
                    "task_id": f"ablation-{index}",
                    "dimension": "policy-score",
                    "full_candidate_score": 0.66,
                    "ablated_score": 0.60,
                    "full_run_hash": _digest(f"full-{index}"),
                    "ablated_run_hash": _digest(f"ablated-{index}"),
                }
                for index in range(3)
            ],
        )
        gate = build_evolution_gate(
            self.candidate,
            report["samples"],
            constitution=self.constitution,
            ablation_report=ablation,
        )
        self.assertEqual(gate["decision"], "promote_to_canary", gate)

    def test_shadow_failure_is_preserved_as_negative_run_receipt(self) -> None:
        report = run_shadow_benchmark(
            self._benchmark_suite(_failing_evaluator(self.root)),
            self.candidate,
            artifact_store=self.store,
            allow_execution=True,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["task_count"], 0)
        self.assertEqual(report["run_receipts"][0]["status"], "failed")
        self.assertTrue(report["run_receipts"][0]["run_hash"].startswith("sha256:"))
        schema_validator("benchmark_run").validate(report)

    def test_canary_runs_real_commands_and_restores_baseline(self) -> None:
        evaluator = _canary_evaluator(self.root)
        suite = {
            "schema_version": CANARY_SPEC_SCHEMA,
            "target": "search-canary",
            "approval_id": "human:canary-owner",
            "projects": [
                {
                    "project_id": f"project:{index}",
                    "baseline": {"error_rate": 0.02, "quality": 0.60},
                    "command": [sys.executable, str(evaluator)],
                }
                for index in range(3)
            ],
        }
        result = run_canary_suite(
            suite,
            self.candidate,
            artifact_store=self.store,
            deployment_root=self.root / "deployments",
            executed_by="service:canary",
            allow_execution=True,
        )
        schema_validator("canary_run").validate(result)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["canary_report"]["observation_count"], 30)
        restored = json.loads(
            (self.root / "deployments" / "search-canary" / "policy.json").read_text()
        )
        self.assertEqual(restored["score"], 0.60)
        self.assertTrue(result["canary_report"]["rollback_receipt"]["exercise_only"])

    def test_canary_failure_is_recorded_and_still_restores_baseline(self) -> None:
        result = run_canary_suite(
            {
                "schema_version": CANARY_SPEC_SCHEMA,
                "target": "failed-canary",
                "approval_id": "human:canary-owner",
                "projects": [
                    {
                        "project_id": "project:failed",
                        "baseline": {"error_rate": 0.02, "quality": 0.60},
                        "command": [
                            sys.executable,
                            str(_failing_evaluator(self.root)),
                        ],
                    }
                ],
            },
            self.candidate,
            artifact_store=self.store,
            deployment_root=self.root / "failed-canary-root",
            executed_by="service:canary",
            allow_execution=True,
        )
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["errors"])
        self.assertEqual(result["project_receipts"][0]["status"], "failed")
        restored = json.loads(
            (
                self.root / "failed-canary-root" / "failed-canary" / "policy.json"
            ).read_text()
        )
        self.assertEqual(restored["score"], 0.60)
        schema_validator("canary_run").validate(result)

    def test_local_deployment_defaults_to_plan_and_requires_human_apply(self) -> None:
        adapter = LocalEvolutionDeployment(
            artifact_store=self.store,
            deployment_root=self.root / "deploy",
            executed_by="service:release",
        )
        plan = adapter.deploy(
            target="search",
            artifact_hash=self.candidate["candidate_artifact_hash"],
        )
        self.assertFalse(plan["production_mutated"])
        self.assertFalse((self.root / "deploy" / "search").exists())
        with self.assertRaises(Exception):
            adapter.deploy(
                target="search",
                artifact_hash=self.candidate["candidate_artifact_hash"],
                apply=True,
                approval_id="service:not-human",
            )

    def test_signed_promotion_deploys_exact_candidate_and_rolls_back(self) -> None:
        benchmark = run_shadow_benchmark(
            self._benchmark_suite(_evaluator(self.root)),
            self.candidate,
            artifact_store=self.store,
            allow_execution=True,
        )
        ablation = build_ablation_report(
            self.candidate,
            [
                {
                    "task_id": f"signed-ablation-{index}",
                    "dimension": "policy-score",
                    "full_candidate_score": 0.66,
                    "ablated_score": 0.60,
                    "full_run_hash": _digest(f"signed-full-{index}"),
                    "ablated_run_hash": _digest(f"signed-ablated-{index}"),
                }
                for index in range(3)
            ],
        )
        gate = build_evolution_gate(
            self.candidate,
            benchmark["samples"],
            constitution=self.constitution,
            ablation_report=ablation,
        )
        canary = run_canary_suite(
            {
                "schema_version": CANARY_SPEC_SCHEMA,
                "target": "signed-canary",
                "approval_id": "human:canary-owner",
                "projects": [
                    {
                        "project_id": f"signed-project:{index}",
                        "baseline": {"error_rate": 0.02, "quality": 0.60},
                        "command": [
                            sys.executable,
                            str(_canary_evaluator(self.root)),
                        ],
                    }
                    for index in range(3)
                ],
            },
            self.candidate,
            artifact_store=self.store,
            deployment_root=self.root / "signed-canary-root",
            executed_by="service:canary",
            allow_execution=True,
        )["canary_report"]
        promotion = approve_production_promotion(
            gate,
            canary,
            constitution=self.constitution,
            approver_ids=["human:release"],
        )
        self.assertEqual(promotion["decision"], "approved")
        approval_payload = {
            "candidate_hash": self.candidate["candidate_hash"],
            "promotion_hash": promotion["promotion_hash"],
            "decision": "approved",
        }
        artifacts = [
            (
                "candidate_artifact",
                "agent:evolution",
                self.candidate,
            ),
            (
                "independent_benchmark",
                "service:evaluator",
                gate,
            ),
            ("canary_execution", "service:canary", canary),
            ("production_approval", "human:release", approval_payload),
        ]
        attestations = []
        trust = {}
        for purpose, identity, payload in artifacts:
            key_id = "key:" + purpose
            key = ("secret-" + purpose).encode()
            trust[key_id] = {
                "identity": identity,
                "algorithm": "hmac-sha256",
                "key": key,
            }
            attestations.append(
                sign_attestation(
                    payload,
                    purpose=purpose,
                    identity=identity,
                    key_id=key_id,
                    key=key,
                )
            )
        adapter = LocalEvolutionDeployment(
            artifact_store=self.store,
            deployment_root=self.root / "production",
            executed_by="service:release-adapter",
        )
        kwargs = {
            "constitution": self.constitution,
            "authorization_bundle": {"attestations": attestations},
            "trust_store": trust,
            "deployment": adapter,
            "target": "search",
            "approval_id": "human:release",
        }
        plan = deploy_approved_candidate(
            self.candidate, promotion, apply=False, **kwargs
        )
        self.assertFalse(plan["production_mutated"])
        receipt = deploy_approved_candidate(
            self.candidate, promotion, apply=True, **kwargs
        )
        self.assertTrue(receipt["production_mutated"])
        self.assertTrue(
            validate_deployment_receipt(
                receipt,
                candidate_artifact_hash=self.candidate["candidate_artifact_hash"],
            )["ok"]
        )
        deployed = json.loads(
            (self.root / "production" / "search" / "policy.json").read_text()
        )
        self.assertEqual(deployed["score"], 0.66)
        rollback_payload = {
            "candidate_hash": self.candidate["candidate_hash"],
            "baseline_artifact_hash": self.candidate["base_artifact_hash"],
            "target": "search",
            "trigger": "quality_regression",
        }
        rollback_attestation = sign_attestation(
            rollback_payload,
            purpose="production_rollback",
            identity="human:release",
            key_id="key:rollback",
            key=b"rollback-secret",
        )
        rollback_authorization = verify_attestation(
            rollback_attestation,
            rollback_payload,
            trust_store={
                "key:rollback": {
                    "identity": "human:release",
                    "algorithm": "hmac-sha256",
                    "key": b"rollback-secret",
                }
            },
            purpose="production_rollback",
            identity="human:release",
        )
        rollback = adapter.rollback(
            self.candidate,
            target="search",
            apply=True,
            approval_id="human:release",
            production=True,
            trigger="quality_regression",
            authorization=rollback_authorization,
        )
        self.assertFalse(rollback["rollback_receipt"]["exercise_only"])
        restored = json.loads(
            (self.root / "production" / "search" / "policy.json").read_text()
        )
        self.assertEqual(restored["score"], 0.60)
