from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol.schemas import load_schema
from xscientist import ResearchRepository
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
            claim_id, _evidence_id, attempt_id = self._record_lineage(
                repository, replay_ready=True
            )
            gate = repository.record(
                "gate_decision",
                {"decision": "promote", "claim_promotion_allowed": True},
                state="verified",
                relations=[{"type": "evaluates", "target": claim_id}],
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
            repository.record(
                "reproduction",
                {
                    "reproduction_level": "computational_rerun",
                    "verdict": "passed",
                    "receipt_hash": "sha256:" + "5" * 64,
                },
                state="verified",
                relations=[
                    {"type": "reproduces", "target": attempt_id},
                    {"type": "reproduces", "target": verified_claim.object_id},
                ],
            )
            repository.commit(stage="review", subject="verify reproduced claim")

            audit = repository.audit(level="verify")
            verified_row = next(
                item
                for item in audit["claims"]
                if item["claim_id"] == verified_claim.object_id
            )

            self.assertTrue(verified_row["complete"], audit["blockers"])
            # The historical draft remains visible and correctly prevents the
            # whole ref from being called fully verified.
            self.assertFalse(audit["complete"])

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
            self.assertIn("Scientific closure: complete", output.getvalue())


if __name__ == "__main__":
    unittest.main()
