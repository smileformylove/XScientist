from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol import content_hash
from ai_scientist.protocol.schemas import load_schema
from xscientist import ResearchRepository
from xscientist.research_authority import require_independent_evaluator
from xscientist.research_cli import main as research_main


@unittest.skipUnless(shutil.which("git"), "Git is required for closure tests")
class ResearchClosureTests(unittest.TestCase):
    def _repository(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            question="Does H1 improve the metric?",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )

    def _record_lineage(
        self, repository: ResearchRepository, *, replay_ready: bool = False
    ) -> tuple[str, str, str]:
        hypothesis = repository.record("hypothesis", {"statement": "H1"})
        plan = repository.record(
            "research_plan",
            {"summary": "Evaluate H1"},
            relations=[{"type": "depends_on", "target": hypothesis.object_id}],
        )
        provenance = {}
        evidence_payload = {"result": "positive"}
        if replay_ready:
            provenance = {
                "environment_hash": "sha256:" + "1" * 64,
                "code_commit": "2" * 40,
                "dataset_hashes": ["sha256:" + "3" * 64],
                "dependency_lock_hashes": ["sha256:" + "6" * 64],
                "seeds": [7],
            }
            evidence_payload["measurement_hash"] = "sha256:" + "4" * 64
        attempt = repository.record(
            "experiment_attempt",
            {"status": "completed", "study_phase": "exploratory"},
            state="completed",
            relations=[
                {"type": "depends_on", "target": plan.object_id, "role": "plan"}
            ],
            provenance=provenance,
        )
        evidence = repository.record(
            "evidence",
            evidence_payload,
            state="completed",
            relations=[{"type": "derived_from", "target": attempt.object_id}],
        )
        claim = repository.record(
            "claim",
            {"statement": "H1 improves the metric."},
            relations=[{"type": "depends_on", "target": evidence.object_id}],
        )
        return claim.object_id, evidence.object_id, attempt.object_id

    def test_trace_audit_is_payload_free_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            claim_id, evidence_id, attempt_id = self._record_lineage(repository)
            repository.commit(stage="evidence", subject="bind claim lineage")

            audit = repository.audit(level="trace")

            self.assertTrue(audit["complete"], audit["blockers"])
            self.assertFalse(audit["payloads_disclosed"])
            self.assertEqual(audit["claims"][0]["claim_id"], claim_id)
            self.assertEqual(audit["claims"][0]["evidence_ids"], [evidence_id])
            self.assertEqual(audit["claims"][0]["attempt_ids"], [attempt_id])
            self.assertNotIn("payload", audit["claims"][0])
            self.assertEqual(
                set(audit["closure_levels"]), {"trace", "replay", "verify"}
            )
            self.assertTrue(audit["closure_levels"]["trace"]["complete"])
            self.assertEqual(audit["closure_levels"]["trace"]["blocker_count"], 0)
            self.assertGreaterEqual(
                audit["closure_levels"]["verify"]["blocker_count"], 1
            )
            validate(audit, load_schema("research_closure"))

    def test_replay_audit_distinguishes_saved_from_replay_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            self._record_lineage(repository)
            repository.commit(stage="evidence", subject="record incomplete provenance")

            audit = repository.audit(level="replay")

            self.assertFalse(audit["complete"])
            self.assertIn(
                "missing_environment_identity",
                {item["code"] for item in audit["blockers"]},
            )

    def test_verified_closure_requires_gate_and_reproduction_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            claim_id, evidence_id, attempt_id = self._record_lineage(
                repository, replay_ready=True
            )
            review_independence = require_independent_evaluator(
                repository,
                evaluator_id="independent-reviewer",
                target_ids=[evidence_id],
                label="test review",
            )
            self.assertEqual(
                review_independence["assurance"], "declared_actor_disjointness"
            )
            self.assertFalse(review_independence["identity_verified"])
            review = repository.record(
                "review",
                {
                    "summary": "Independent verification passed",
                    "status": "verified",
                    "report_hash": "sha256:" + "9" * 64,
                    "independence": review_independence,
                },
                state="verified",
                relations=[{"type": "evaluates", "target": evidence_id}],
                actor={
                    "actor_id": "independent-reviewer",
                    "authority": "independent_evaluator",
                },
            )
            gate = repository.record(
                "gate_decision",
                {"decision": "promote", "claim_promotion_allowed": True},
                state="verified",
                relations=[{"type": "evaluates", "target": review.object_id}],
                actor={
                    "actor_id": "integrity-gate",
                    "authority": "deterministic_gate",
                },
            )
            # A verified claim is a new immutable object bound to the same evidence.
            draft = repository.get(claim_id)
            verified_claim = repository.record(
                "claim",
                draft["payload"],
                state="verified",
                relations=[
                    *draft["relations"],
                    {"type": "depends_on", "target": gate.object_id, "role": "gate"},
                ],
            )
            receipt_base = {
                "schema_version": "xscientist.reproduction-receipt.v1",
                "created_at": "2026-08-08T00:00:00+00:00",
                "commit": "a" * 40,
                "checkpoint_id": "rcp-test",
                "checkpoint_hash": "sha256:" + "b" * 64,
                "reproduction_level": "computational_rerun",
                "verdict": "passed",
                "objects_complete": True,
                "environment": {
                    "policy": "strict",
                    "recorded": True,
                    "matches": True,
                    "recorded_content_hash": "sha256:" + "c" * 64,
                    "mismatch_fields": [],
                },
                "command_hash": "sha256:" + "d" * 64,
                "executed": True,
                "returncode": 0,
                "timed_out": False,
                "stdout_hash": "sha256:" + "e" * 64,
                "stderr_hash": "sha256:" + "f" * 64,
            }
            receipt_hash = content_hash(receipt_base)
            receipt = {
                **receipt_base,
                "receipt_id": f"rr-{receipt_hash.split(':', 1)[1][:16]}",
                "content_hash": receipt_hash,
            }
            reproduction_independence = require_independent_evaluator(
                repository,
                evaluator_id="independent-reproducer",
                target_ids=[attempt_id, verified_claim.object_id],
                label="test reproduction",
            )
            repository.record(
                "reproduction",
                {
                    "reproduction_level": "computational_rerun",
                    "verdict": "passed",
                    "receipt_hash": receipt_hash,
                    "receipt": receipt,
                    "independence": reproduction_independence,
                },
                state="verified",
                relations=[
                    {"type": "reproduces", "target": attempt_id},
                    {"type": "reproduces", "target": verified_claim.object_id},
                ],
                actor={
                    "actor_id": "independent-reproducer",
                    "authority": "independent_evaluator",
                },
            )
            repository.commit(stage="review", subject="verify reproduced claim")

            audit = repository.audit(level="verify")
            verified_row = next(
                item
                for item in audit["claims"]
                if item["claim_id"] == verified_claim.object_id
            )

            self.assertTrue(verified_row["complete"], audit["blockers"])
            self.assertTrue(audit["complete"], audit["blockers"])
            self.assertEqual(audit["superseded_claim_ids"], [claim_id])

    def test_cli_audit_returns_success_for_trace_complete_ref(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            self._record_lineage(repository)
            repository.commit(stage="evidence", subject="bind claim lineage")
            output = io.StringIO()

            with redirect_stdout(output):
                status = research_main(
                    ["audit", "--repo", str(repository.path), "--level", "trace"]
                )

            self.assertEqual(status, 0)
            rendered = output.getvalue()
            self.assertIn("Traceability closure: complete", rendered)
            self.assertIn("replay=blocked", rendered)
            self.assertIn("verification=blocked", rendered)
            self.assertIn("Overall scientific closure: pending", rendered)
            self.assertNotIn("Scientific closure: complete", rendered)

    def test_complete_high_level_cli_journey_reaches_verified_closure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"

            def run_json(*args: str, expected: int = 0) -> dict:
                output = io.StringIO()
                with redirect_stdout(output):
                    status = research_main([*args, "--json"])
                self.assertEqual(status, expected, output.getvalue())
                return json.loads(output.getvalue())

            run_json(
                "init",
                str(root),
                "--question",
                "Does H1 improve accuracy?",
                "--git-user-name",
                "Research Test",
                "--git-user-email",
                "research@example.invalid",
            )
            (root / "requirements.txt").write_text(
                "jsonschema==4.23.0\n", encoding="utf-8"
            )
            verifier_script = root / ".xscientist" / "verify.py"
            verifier_script.write_text("print('verified')\n", encoding="utf-8")
            hypothesis = run_json(
                "hypothesis",
                "H1 improves accuracy",
                "--falsifier",
                "Accuracy does not improve",
                "--repo",
                str(root),
            )["object"]
            run_json(
                "preregister",
                "@latest:hypothesis",
                "--dataset",
                "benchmark-v1",
                "--metric",
                "accuracy",
                "--baseline",
                "baseline-a",
                "--split-hash",
                "sha256:" + "a" * 64,
                "--registered-by",
                "principal-investigator",
                "--repo",
                str(root),
            )
            experiment_result = run_json(
                "experiment",
                "confirmatory run",
                "--status",
                "success",
                "--study-phase",
                "confirmatory",
                "--plan",
                "@latest:research_plan",
                "--preregistration",
                "@latest:preregistration",
                "--seed",
                "42",
                "--reproduce-command",
                "python .xscientist/verify.py",
                "--repo",
                str(root),
            )
            attempt = experiment_result["object"]
            experiment_commit = experiment_result["checkpoint"]["commit"]
            evidence = run_json(
                "evidence",
                "accuracy improved",
                "--attempt",
                "@latest:experiment_attempt",
                "--supports",
                hypothesis["object_id"],
                "--metric",
                "accuracy=0.91",
                "--repo",
                str(root),
            )["object"]
            run_json(
                "review",
                "independent review passed",
                "--evaluates",
                "@latest:evidence",
                "--verifier",
                "external-reviewer",
                "--decision",
                "pass",
                "--repo",
                str(root),
            )
            claim = run_json(
                "claim",
                "H1 improves accuracy",
                "--evidence",
                "@latest:evidence",
                "--gate",
                "@latest:gate_decision",
                "--verified",
                "--repo",
                str(root),
            )["object"]

            replay = run_json("audit", "--repo", str(root), "--level", "replay")
            self.assertTrue(replay["complete"], replay["blockers"])
            run_json(
                "reproduce",
                experiment_commit,
                "--repo",
                str(root),
                "--dest",
                str(base / "reproduction"),
                "--execute",
                "--record",
                "--reproduces",
                attempt["object_id"],
                "--reproduces",
                claim["object_id"],
                "--verifier",
                "independent-reproducer",
                "--verified",
            )
            verified = run_json("audit", "--repo", str(root), "--level", "verify")
            self.assertTrue(verified["complete"], verified["blockers"])
            self.assertEqual(verified["claims"][0]["claim_id"], claim["object_id"])
            self.assertEqual(evidence["kind"], "evidence")


if __name__ == "__main__":
    unittest.main()
