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
