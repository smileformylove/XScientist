from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_harness import build_harness_policy_hash
from ai_scientist.utils.pipeline_contracts import load_pipeline_manifest
from ai_scientist.utils.science_constitution import build_science_constitution
from xscientist.cli import main as xscientist_main


class EvolutionCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = tempfile.mkdtemp(prefix="xscientist-evolution-cli-")
        self.root = Path(self.raw)
        self.base = self.root / "base"
        self.changed = self.root / "changed"
        (self.base / "search").mkdir(parents=True)
        (self.changed / "search").mkdir(parents=True)
        (self.base / "search" / "policy.json").write_text(
            '{"score": 0.5}\n', encoding="utf-8"
        )
        (self.changed / "search" / "policy.json").write_text(
            '{"score": 0.6}\n', encoding="utf-8"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.raw)

    def _harness_version(
        self,
        version_id: str,
        parent_version_id: str | None,
        score: float,
        *,
        epoch_id: str = "epoch:cli-history",
    ) -> dict:
        def digest(label: str) -> str:
            return canonical_content_hash({"label": label})

        return {
            "epoch_id": epoch_id,
            "version_id": version_id,
            "parent_version_id": parent_version_id,
            "scores": {"objective": score},
            "reward_groups": [[0.0, 0.4, 1.0], [0.1, 0.5, 0.8]],
            "behavior": {"loop_rate": 0.1, "test_rate": 0.8},
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
                "harness_hash": digest("harness"),
                "policy_hash": build_harness_policy_hash(),
                "evaluator_hash": digest("evaluator"),
                "task_hash": digest("tasks"),
                "resource_hash": digest("resources"),
                "seed_policy_hash": digest("seeds"),
            },
            "cost": {"observed": 4.0, "budget": 10.0, "unit": "tokens"},
        }

    def _write_harness_evidence(self, filename: str, versions: list[dict]) -> Path:
        path = self.root / filename
        path.write_text(json.dumps({"versions": versions}), encoding="utf-8")
        return path

    def _run_project_harness(
        self,
        evidence: Path,
        project_root: Path,
        *,
        supersede: bool = False,
    ) -> tuple[int, str]:
        argv = [
            "evolution",
            "harness-audit",
            "--evidence",
            str(evidence),
            "--project-root",
            str(project_root),
        ]
        if supersede:
            argv.append("--supersede")
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = xscientist_main(argv)
        return code, stderr.getvalue()

    @staticmethod
    def _harness_history(project_root: Path) -> list[dict]:
        path = project_root / "knowledge" / "evolution_harness_history.jsonl"
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_top_level_candidate_command_builds_real_artifacts(self) -> None:
        constitution_path = self.root / "constitution.json"
        constitution_path.write_text(
            json.dumps(build_science_constitution(project_name="cli-test")),
            encoding="utf-8",
        )
        spec = {
            "base_root": str(self.base),
            "candidate_root": str(self.changed),
            "candidate_id": "cli-candidate",
            "component_type": "search_policy",
            "base_version": "1",
            "candidate_version": "2",
            "proposed_by": "agent:cli",
            "change_summary": "Change search score.",
            "change_scope": ["search/policy.json"],
            "applicability_domains": ["general"],
            "failure_taxonomy_refs": ["failure:search"],
            "ablation_dimensions": ["score"],
        }
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        output = self.root / "candidate.json"
        code = xscientist_main(
            [
                "evolution",
                "candidate",
                "--spec",
                str(spec_path),
                "--constitution",
                str(constitution_path),
                "--store",
                str(self.root / "store"),
                "--out",
                str(output),
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(output.read_text())
        self.assertEqual(payload["candidate"]["candidate_id"], "cli-candidate")
        self.assertEqual(payload["change_set"]["modified"], ["search/policy.json"])

    def test_attestation_cli_keeps_secret_in_environment(self) -> None:
        payload_path = self.root / "payload.json"
        payload_path.write_text('{"result": "verified"}\n', encoding="utf-8")
        attestation_path = self.root / "attestation.json"
        trust_path = self.root / "trust.json"
        trust_path.write_text(
            json.dumps(
                {
                    "keys": {
                        "key:evaluator": {
                            "identity": "service:evaluator",
                            "algorithm": "hmac-sha256",
                            "key_env": "XSCIENTIST_TEST_SIGNING_KEY",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        with patch.dict(
            os.environ, {"XSCIENTIST_TEST_SIGNING_KEY": "local-test-secret"}
        ):
            code = xscientist_main(
                [
                    "evolution",
                    "attest",
                    "sign",
                    "--payload",
                    str(payload_path),
                    "--purpose",
                    "independent_benchmark",
                    "--identity",
                    "service:evaluator",
                    "--key-id",
                    "key:evaluator",
                    "--key-env",
                    "XSCIENTIST_TEST_SIGNING_KEY",
                    "--out",
                    str(attestation_path),
                ]
            )
            stdout = StringIO()
            with redirect_stdout(stdout):
                verify_code = xscientist_main(
                    [
                        "evolution",
                        "attest",
                        "verify",
                        "--payload",
                        str(payload_path),
                        "--attestation",
                        str(attestation_path),
                        "--trust-store",
                        str(trust_path),
                        "--purpose",
                        "independent_benchmark",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(verify_code, 0)
        self.assertTrue(json.loads(stdout.getvalue())["ok"])
        self.assertNotIn("local-test-secret", attestation_path.read_text())

    def test_execution_is_refused_without_explicit_execute(self) -> None:
        candidate_path = self.root / "candidate.json"
        candidate_path.write_text("{}", encoding="utf-8")
        suite_path = self.root / "suite.json"
        suite_path.write_text("{}", encoding="utf-8")
        stderr = StringIO()
        with redirect_stderr(stderr):
            code = xscientist_main(
                [
                    "evolution",
                    "benchmark",
                    "--suite",
                    str(suite_path),
                    "--candidate",
                    str(candidate_path),
                    "--store",
                    str(self.root / "store"),
                ]
            )
        self.assertEqual(code, 2)
        error = json.loads(stderr.getvalue())
        self.assertFalse(error["ok"])
        self.assertIn("execution is disabled", error["error"]["message"])

    def test_harness_audit_persists_a_healthy_project_artifact(self) -> None:
        def digest(label: str) -> str:
            return canonical_content_hash({"label": label})

        comparison_hashes = {
            "harness_hash": digest("harness"),
            "policy_hash": build_harness_policy_hash(),
            "evaluator_hash": digest("evaluator"),
            "task_hash": digest("tasks"),
            "resource_hash": digest("resources"),
            "seed_policy_hash": digest("seeds"),
        }
        thresholds = {
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
        }

        def version(
            version_id: str,
            parent_version_id: str | None,
            score: float,
            loop_rate: float,
            test_rate: float,
        ) -> dict:
            return {
                "epoch_id": "epoch:cli-test",
                "version_id": version_id,
                "parent_version_id": parent_version_id,
                "scores": {"objective": score},
                "reward_groups": [[0.0, 0.4, 1.0], [0.1, 0.5, 0.8]],
                "behavior": {
                    "loop_rate": loop_rate,
                    "test_rate": test_rate,
                },
                "behavior_thresholds": thresholds,
                "integrity_checks": {
                    "evidence_bound": True,
                    "environment_isolated": True,
                    "evaluation_frozen": True,
                    "git_leakage_absent": True,
                },
                "comparison_hashes": comparison_hashes,
                "cost": {"observed": 4.0, "budget": 10.0, "unit": "tokens"},
            }

        evidence = self.root / "harness-evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "versions": [
                        version("v1", None, 0.5, 0.2, 0.6),
                        version("v2", "v1", 0.7, 0.1, 0.8),
                    ]
                }
            ),
            encoding="utf-8",
        )
        project_root = self.root / "project"
        project_root.mkdir()
        output = self.root / "harness-report.json"

        code = xscientist_main(
            [
                "evolution",
                "harness-audit",
                "--evidence",
                str(evidence),
                "--project-root",
                str(project_root),
                "--out",
                str(output),
            ]
        )

        self.assertEqual(code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["progression"]["decision"], "eligible_for_human_review")
        self.assertFalse(report["progression"]["automatic_progression_allowed"])
        self.assertEqual(
            json.loads((project_root / "evolution_harness.json").read_text()),
            report,
        )
        manifest = load_pipeline_manifest(project_root)
        self.assertEqual(manifest["artifacts"]["evolution_harness"]["status"], "ready")

        held_spec = json.loads(evidence.read_text(encoding="utf-8"))
        held_spec["versions"][1]["reward_groups"] = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
        evidence.write_text(json.dumps(held_spec), encoding="utf-8")
        held_output = self.root / "held-harness-report.json"
        held_code = xscientist_main(
            [
                "evolution",
                "harness-audit",
                "--evidence",
                str(evidence),
                "--out",
                str(held_output),
            ]
        )
        self.assertEqual(held_code, 3)
        held_report = json.loads(held_output.read_text(encoding="utf-8"))
        self.assertEqual(held_report["progression"]["decision"], "hold")
        self.assertIn(
            "SIGNAL.ALL_TIED",
            held_report["progression"]["blocking_risk_codes"],
        )

    def test_harness_same_epoch_replacement_requires_supersede_without_mutation(
        self,
    ) -> None:
        held = self._write_harness_evidence(
            "held-history.json",
            [
                self._harness_version("v1", None, 0.5),
                self._harness_version("v2", "v1", 0.4),
            ],
        )
        clean = self._write_harness_evidence(
            "clean-history.json",
            [
                self._harness_version("v1", None, 0.5),
                self._harness_version("v2", "v1", 0.7),
            ],
        )
        project_root = self.root / "history-refusal-project"
        project_root.mkdir()

        held_code, held_error = self._run_project_harness(held, project_root)
        self.assertEqual(held_code, 3)
        self.assertEqual(held_error, "")
        current_path = project_root / "evolution_harness.json"
        history_path = project_root / "knowledge" / "evolution_harness_history.jsonl"
        current_before = current_path.read_bytes()
        history_before = history_path.read_bytes()

        clean_code, clean_error = self._run_project_harness(clean, project_root)

        self.assertEqual(clean_code, 2)
        error = json.loads(clean_error)
        self.assertIn("not an exact prefix extension", error["error"]["message"])
        self.assertIn("--supersede", error["error"]["message"])
        self.assertEqual(current_path.read_bytes(), current_before)
        self.assertEqual(history_path.read_bytes(), history_before)

    def test_harness_supersede_requires_project_root(self) -> None:
        evidence = self._write_harness_evidence(
            "supersede-without-project.json",
            [
                self._harness_version("v1", None, 0.5),
                self._harness_version("v2", "v1", 0.7),
            ],
        )
        stderr = StringIO()
        with redirect_stderr(stderr):
            code = xscientist_main(
                [
                    "evolution",
                    "harness-audit",
                    "--evidence",
                    str(evidence),
                    "--supersede",
                ]
            )

        self.assertEqual(code, 2)
        error = json.loads(stderr.getvalue())
        self.assertEqual(
            error["error"]["message"], "--supersede requires --project-root"
        )

    def test_harness_supersede_retains_both_hash_chained_audits(self) -> None:
        held = self._write_harness_evidence(
            "held-supersede.json",
            [
                self._harness_version("v1", None, 0.5),
                self._harness_version("v2", "v1", 0.4),
            ],
        )
        clean = self._write_harness_evidence(
            "clean-supersede.json",
            [
                self._harness_version("v1", None, 0.5),
                self._harness_version("v2", "v1", 0.7),
            ],
        )
        project_root = self.root / "history-supersede-project"
        project_root.mkdir()

        held_code, _ = self._run_project_harness(held, project_root)
        clean_code, clean_error = self._run_project_harness(
            clean, project_root, supersede=True
        )

        self.assertEqual(held_code, 3)
        self.assertEqual(clean_code, 0)
        self.assertEqual(clean_error, "")
        rows = self._harness_history(project_root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"previous_audit_hash", "epoch_id", "audit"})
        self.assertIsNone(rows[0]["previous_audit_hash"])
        self.assertEqual(rows[1]["previous_audit_hash"], rows[0]["audit"]["audit_hash"])
        self.assertNotEqual(
            rows[0]["audit"]["audit_hash"], rows[1]["audit"]["audit_hash"]
        )
        current = json.loads(
            (project_root / "evolution_harness.json").read_text(encoding="utf-8")
        )
        self.assertEqual(current["audit_hash"], rows[1]["audit"]["audit_hash"])
        self.assertEqual(
            current["progression"]["decision"], "eligible_for_human_review"
        )

    def test_harness_exact_prefix_extension_updates_current_idempotently(self) -> None:
        initial_versions = [
            self._harness_version("v1", None, 0.5),
            self._harness_version("v2", "v1", 0.6),
        ]
        initial = self._write_harness_evidence("prefix-initial.json", initial_versions)
        extended = self._write_harness_evidence(
            "prefix-extended.json",
            [
                *initial_versions,
                self._harness_version("v3", "v2", 0.7),
            ],
        )
        project_root = self.root / "history-prefix-project"
        project_root.mkdir()

        initial_code, _ = self._run_project_harness(initial, project_root)
        extended_code, extended_error = self._run_project_harness(
            extended, project_root
        )
        repeat_code, repeat_error = self._run_project_harness(extended, project_root)

        self.assertEqual(initial_code, 0)
        self.assertEqual(extended_code, 0)
        self.assertEqual(repeat_code, 0)
        self.assertEqual(extended_error, "")
        self.assertEqual(repeat_error, "")
        rows = self._harness_history(project_root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["previous_audit_hash"], rows[0]["audit"]["audit_hash"])
        current = json.loads(
            (project_root / "evolution_harness.json").read_text(encoding="utf-8")
        )
        self.assertEqual(current["audit_hash"], rows[1]["audit"]["audit_hash"])
        self.assertEqual(current["evidence"]["version_ids"], ["v1", "v2", "v3"])
