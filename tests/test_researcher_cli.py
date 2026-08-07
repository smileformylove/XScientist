from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from xscientist.entrypoints import research_main
from xscientist.research_vcs import ResearchRepository


class ResearcherCliTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("Git is not installed")

    def _run_json(self, args: list[str]) -> tuple[int, dict]:
        with redirect_stdout(io.StringIO()) as output:
            returncode = research_main([*args, "--json"])
        return returncode, json.loads(output.getvalue())

    def test_one_command_research_lifecycle_records_negative_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")

            code, hypothesis = self._run_json(
                [
                    "hypothesis",
                    "Method A improves the fixed baseline",
                    "--falsifier",
                    "delta <= 0",
                    "--prediction",
                    "delta > 0",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            hypothesis_id = hypothesis["object"]["object_id"]
            self.assertTrue(hypothesis["checkpoint"]["committed"])

            code, duplicate = self._run_json(
                [
                    "hypothesis",
                    "Method A improves the fixed baseline",
                    "--falsifier",
                    "delta <= 0",
                    "--prediction",
                    "delta > 0",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            self.assertFalse(duplicate["object"]["created"])
            self.assertFalse(duplicate["checkpoint"]["committed"])
            self.assertEqual(len(repository.log()), 2)

            code, experiment = self._run_json(
                [
                    "experiment",
                    "Seed 7 exceeded the wall-clock budget",
                    "--status",
                    "timeout",
                    "--failure-class",
                    "budget_exhausted",
                    "--metric",
                    "elapsed_seconds=60.0",
                    "--seed",
                    "7",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            attempt_id = experiment["object"]["object_id"]
            self.assertEqual(experiment["object"]["state"], "timed_out")

            code, evidence = self._run_json(
                [
                    "evidence",
                    "The timeout is reproducible under the fixed budget",
                    "--attempt",
                    attempt_id,
                    "--supports",
                    hypothesis_id,
                    "--metric",
                    "reproduced=true",
                    "--verified",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            evidence_id = evidence["object"]["object_id"]
            self.assertEqual(evidence["object"]["state"], "verified")

            code, claim = self._run_json(
                [
                    "claim",
                    "Method A is not evaluable within the fixed budget",
                    "--evidence",
                    evidence_id,
                    "--scope",
                    "seed 7 and the sealed environment",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(claim["object"]["state"], "draft")
            self.assertEqual(len(repository.log()), 5)
            self.assertTrue(repository.fsck()["ok"])

    def test_verified_claim_without_gate_fails_before_recording(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            attempt = repository.record(
                "experiment_attempt", {"status": "completed"}, state="completed"
            )
            repository.commit(stage="experiment", subject="record attempt")
            evidence = repository.record(
                "evidence",
                {"result": "fixed"},
                state="verified",
                relations=[{"type": "derived_from", "target": attempt.object_id}],
            )
            repository.commit(stage="evidence", subject="record evidence")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = research_main(
                    [
                        "claim",
                        "Unsupported promotion",
                        "--evidence",
                        evidence.object_id,
                        "--verified",
                        "--repo",
                        str(root),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(repository.objects(kind="claim"), [])

    def test_confirmatory_experiment_requires_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = research_main(
                    [
                        "experiment",
                        "Confirmatory run without a sealed protocol",
                        "--status",
                        "success",
                        "--study-phase",
                        "confirmatory",
                        "--repo",
                        str(root),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_confirmatory_lifecycle_has_one_command_integrity_steps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")

            code, hypothesis = self._run_json(
                [
                    "hypothesis",
                    "Method A improves accuracy over baseline A",
                    "--falsifier",
                    "accuracy improvement <= 0.01",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            hypothesis_id = hypothesis["object"]["object_id"]

            code, preregistration = self._run_json(
                [
                    "preregister",
                    hypothesis_id,
                    "--dataset",
                    "benchmark-v1",
                    "--metric",
                    "accuracy",
                    "--baseline",
                    "baseline-a",
                    "--split-hash",
                    "sha256:" + "a" * 64,
                    "--registered-by",
                    "lead-researcher",
                    "--minimum-effect",
                    "0.01",
                    "--minimum-seeds",
                    "3",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            preregistration_id = preregistration["object"]["object_id"]
            self.assertEqual(preregistration["object"]["state"], "locked")
            self.assertEqual(len(preregistration["related_objects"]), 1)
            plan_id = preregistration["related_objects"][0]["object_id"]
            self.assertEqual(
                preregistration["related_objects"][0]["kind"], "research_plan"
            )

            code, experiment = self._run_json(
                [
                    "experiment",
                    "Three sealed seeds completed",
                    "--status",
                    "success",
                    "--study-phase",
                    "confirmatory",
                    "--preregistration",
                    preregistration_id,
                    "--plan",
                    plan_id,
                    "--metric",
                    "accuracy=0.91",
                    "--seed",
                    "1",
                    "--seed",
                    "2",
                    "--seed",
                    "3",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            attempt_id = experiment["object"]["object_id"]

            code, evidence = self._run_json(
                [
                    "evidence",
                    "The preregistered accuracy threshold passed",
                    "--attempt",
                    attempt_id,
                    "--supports",
                    hypothesis_id,
                    "--metric",
                    "accuracy=0.91",
                    "--verified",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            evidence_id = evidence["object"]["object_id"]

            code, review = self._run_json(
                [
                    "review",
                    "Independent checks found no required failure",
                    "--evaluates",
                    evidence_id,
                    "--verifier",
                    "independent-reviewer",
                    "--decision",
                    "pass",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            gate_id = review["object"]["object_id"]
            self.assertEqual(review["object"]["kind"], "gate_decision")
            self.assertEqual(review["object"]["state"], "verified")
            self.assertEqual(review["related_objects"][0]["kind"], "review")

            code, claim = self._run_json(
                [
                    "claim",
                    "Method A improves accuracy over baseline A on benchmark-v1",
                    "--evidence",
                    evidence_id,
                    "--gate",
                    gate_id,
                    "--verified",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(claim["object"]["state"], "verified")
            self.assertEqual(len(repository.log()), 7)
            self.assertTrue(repository.fsck()["ok"])

    def test_passing_review_rejects_declared_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            attempt = repository.record(
                "experiment_attempt", {"status": "completed"}, state="completed"
            )
            repository.commit(stage="experiment", subject="record attempt")

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = research_main(
                    [
                        "review",
                        "Conflicting review",
                        "--evaluates",
                        attempt.object_id,
                        "--verifier",
                        "independent-reviewer",
                        "--decision",
                        "pass",
                        "--failure",
                        "missing replication",
                        "--repo",
                        str(root),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(repository.objects(kind="review"), [])

    def test_preregister_hashes_split_file_without_persisting_its_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            private_split = Path(td) / "private-split.json"
            private_split.write_text('{"train":[1,2],"test":[3]}', encoding="utf-8")
            ResearchRepository.init(root, question="Fixed question")
            code, hypothesis = self._run_json(
                [
                    "hypothesis",
                    "Method A improves accuracy",
                    "--falsifier",
                    "accuracy does not improve",
                    "--repo",
                    str(root),
                ]
            )
            self.assertEqual(code, 0)

            code, registration = self._run_json(
                [
                    "preregister",
                    hypothesis["object"]["object_id"],
                    "--dataset",
                    "private-benchmark",
                    "--metric",
                    "accuracy",
                    "--baseline",
                    "baseline-a",
                    "--split-file",
                    str(private_split),
                    "--registered-by",
                    "lead-researcher",
                    "--repo",
                    str(root),
                ]
            )

            self.assertEqual(code, 0)
            saved = ResearchRepository(root).get(registration["object"]["object_id"])
            serialized = json.dumps(saved, sort_keys=True)
            self.assertNotIn(str(private_split), serialized)
            split_hash = saved["payload"]["data_policy"]["split_hashes"]["primary"]
            self.assertRegex(split_hash, r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
