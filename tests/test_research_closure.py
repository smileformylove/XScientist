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
from xscientist import ResearchLifecycle, ResearchRepository
from xscientist.research_authority import require_independent_evaluator
from xscientist.research_cli import main as research_main
from xscientist.research_closure import build_reproduction_target_binding
from xscientist.research_git import (
    CheckpointResult,
    ResearchGitError,
    ResearchObjectResult,
    create_checkpoint,
)


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

    def _record_verified_closure(self, repository: ResearchRepository) -> dict:
        draft_claim_id, evidence_id, attempt_id = self._record_lineage(
            repository, replay_ready=True
        )
        review_independence = require_independent_evaluator(
            repository,
            evaluator_id="independent-reviewer",
            target_ids=[evidence_id],
            label="test review",
        )
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
        draft = repository.get(draft_claim_id)
        verified_claim = repository.record(
            "claim",
            draft["payload"],
            state="verified",
            relations=[
                *draft["relations"],
                {"type": "depends_on", "target": gate.object_id, "role": "gate"},
            ],
        )
        source_checkpoint = create_checkpoint(
            repository.path,
            stage="review",
            subject="bind claim before independent reproduction",
            reproduce_command="python verify.py",
        )
        reproduction = self._record_verified_reproduction(
            repository,
            attempt_id=attempt_id,
            claim_id=verified_claim.object_id,
            source_checkpoint=source_checkpoint,
        )
        return {
            "draft_claim_id": draft_claim_id,
            "verified_claim_id": verified_claim.object_id,
            "evidence_id": evidence_id,
            "attempt_id": attempt_id,
            "gate_id": gate.object_id,
            "review_independence": review_independence,
            "reproduction_id": reproduction.object_id,
        }

    def _record_verified_reproduction(
        self,
        repository: ResearchRepository,
        *,
        attempt_id: str,
        claim_id: str,
        source_checkpoint: CheckpointResult,
        verifier_id: str = "independent-reproducer",
    ) -> ResearchObjectResult:
        source_commit = str(source_checkpoint.commit or "")
        target_binding = build_reproduction_target_binding(
            repository.path,
            [attempt_id, claim_id],
            ref=source_commit,
        )
        checkpoint_binding_core = {
            "commit": source_commit,
            "checkpoint_id": str(source_checkpoint.checkpoint_id or ""),
            "checkpoint_content_hash": str(source_checkpoint.content_hash or ""),
        }
        execution_result_core = {
            "command_hash": content_hash("python verify.py"),
            "reproduction_level": "computational_rerun",
            "verdict": "passed",
            "objects_complete": True,
            "executed": True,
            "returncode": 0,
            "timed_out": False,
            "stdout_hash": "sha256:" + "e" * 64,
            "stderr_hash": "sha256:" + "f" * 64,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "output_capture": "bounded_tail",
            "max_output_chars": 20_000,
        }
        receipt_base = {
            "schema_version": "xscientist.reproduction-receipt.v2",
            "created_at": "2026-08-08T00:00:00+00:00",
            "commit": source_commit,
            "checkpoint_id": str(source_checkpoint.checkpoint_id or ""),
            "checkpoint_hash": str(source_checkpoint.content_hash or ""),
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
            "command_hash": execution_result_core["command_hash"],
            "executed": True,
            "returncode": 0,
            "timed_out": False,
            "stdout_hash": "sha256:" + "e" * 64,
            "stderr_hash": "sha256:" + "f" * 64,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "output_capture": "bounded_tail",
            "max_output_chars": 20_000,
            "checkpoint_binding": {
                **checkpoint_binding_core,
                "binding_hash": content_hash(checkpoint_binding_core),
            },
            "target_binding": target_binding,
            "execution_result": {
                **execution_result_core,
                "result_hash": content_hash(execution_result_core),
            },
            "execution_isolation": {
                "isolated": False,
                "security_boundary": False,
                "environment": "sanitized",
                "environment_scope": "variables_only",
                "process_tree": "best_effort_process_group",
                "process_control": "posix_process_group_best_effort",
                "process_tree_termination_guaranteed": False,
                "filesystem": "host_visible",
                "network": "host_unrestricted",
            },
        }
        receipt_hash = content_hash(receipt_base)
        receipt = {
            **receipt_base,
            "receipt_id": f"rr-{receipt_hash.split(':', 1)[1][:16]}",
            "content_hash": receipt_hash,
        }
        reproduction_independence = require_independent_evaluator(
            repository,
            evaluator_id=verifier_id,
            target_ids=[attempt_id, claim_id],
            label="test reproduction",
        )
        reproduction = repository.record(
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
                {"type": "reproduces", "target": claim_id},
            ],
            actor={
                "actor_id": verifier_id,
                "authority": "independent_evaluator",
            },
        )
        return reproduction

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
            closure = self._record_verified_closure(repository)
            review_independence = closure["review_independence"]
            self.assertEqual(
                review_independence["assurance"], "declared_actor_disjointness"
            )
            self.assertFalse(review_independence["identity_verified"])
            repository.commit(stage="review", subject="verify reproduced claim")

            audit = repository.audit(level="verify")
            verified_row = next(
                item
                for item in audit["claims"]
                if item["claim_id"] == closure["verified_claim_id"]
            )

            self.assertTrue(verified_row["complete"], audit["blockers"])
            self.assertTrue(audit["complete"], audit["blockers"])
            self.assertEqual(audit["superseded_claim_ids"], [closure["draft_claim_id"]])

    def test_verified_lifecycle_atomically_upgrades_legacy_receipt_to_v2(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            original = repository.get(str(closure["reproduction_id"]))["payload"][
                "receipt"
            ]
            legacy_base = {
                key: value
                for key, value in original.items()
                if key
                not in {
                    "receipt_id",
                    "content_hash",
                    "checkpoint_binding",
                    "target_binding",
                    "execution_result",
                    "execution_isolation",
                    "stdout_truncated",
                    "stderr_truncated",
                    "output_capture",
                    "max_output_chars",
                }
            }
            legacy_base["schema_version"] = "xscientist.reproduction-receipt.v1"
            legacy_hash = content_hash(legacy_base)
            legacy = {
                **legacy_base,
                "receipt_id": f"rr-{legacy_hash.split(':', 1)[1][:16]}",
                "content_hash": legacy_hash,
            }

            recorded = ResearchLifecycle(repository).reproduction(
                legacy,
                reproduces=[
                    str(closure["attempt_id"]),
                    str(closure["verified_claim_id"]),
                ],
                verifier_id="legacy-client-reproducer",
                verified=True,
            )

            bound = recorded["receipt"]
            self.assertEqual(
                bound["schema_version"], "xscientist.reproduction-receipt.v2"
            )
            self.assertEqual(
                bound["execution_isolation"]["environment_scope"],
                "legacy_unknown",
            )
            self.assertFalse(bound["execution_isolation"]["security_boundary"])
            self.assertEqual(bound["output_capture"], "legacy_unknown")
            self.assertIsNone(bound["stdout_truncated"])
            self.assertEqual(
                [
                    item["object_id"]
                    for item in bound["target_binding"]["target_objects"]
                ],
                sorted([str(closure["attempt_id"]), str(closure["verified_claim_id"])]),
            )
            self.assertTrue(repository.audit(level="verify")["complete"])

    def test_legacy_receipt_upgrade_rejects_an_unrelated_source_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            repository.record(
                "hypothesis", {"statement": "Unrelated historical hypothesis"}
            )
            stale_checkpoint = create_checkpoint(
                repository.path,
                stage="ideation",
                subject="record unrelated historical checkpoint",
                reproduce_command="python verify.py",
            )
            closure = self._record_verified_closure(repository)
            original = repository.get(str(closure["reproduction_id"]))["payload"][
                "receipt"
            ]
            legacy_base = {
                key: value
                for key, value in original.items()
                if key
                not in {
                    "receipt_id",
                    "content_hash",
                    "checkpoint_binding",
                    "target_binding",
                    "execution_result",
                    "execution_isolation",
                    "stdout_truncated",
                    "stderr_truncated",
                    "output_capture",
                    "max_output_chars",
                }
            }
            legacy_base.update(
                {
                    "schema_version": "xscientist.reproduction-receipt.v1",
                    "commit": str(stale_checkpoint.commit or ""),
                    "checkpoint_id": str(stale_checkpoint.checkpoint_id or ""),
                    "checkpoint_hash": str(stale_checkpoint.content_hash or ""),
                }
            )
            legacy_hash = content_hash(legacy_base)
            legacy = {
                **legacy_base,
                "receipt_id": f"rr-{legacy_hash.split(':', 1)[1][:16]}",
                "content_hash": legacy_hash,
            }

            with self.assertRaises(ResearchGitError):
                ResearchLifecycle(repository).reproduction(
                    legacy,
                    reproduces=[
                        str(closure["attempt_id"]),
                        str(closure["verified_claim_id"]),
                    ],
                    verifier_id="stale-legacy-reproducer",
                    verified=True,
                )

    def test_split_source_and_target_checkpoint_v2_receipt_cannot_verify(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            original = repository.get(str(closure["reproduction_id"]))["payload"][
                "receipt"
            ]
            repository.commit(
                stage="review", subject="commit original verified reproduction"
            )
            target_ids = sorted(
                [str(closure["attempt_id"]), str(closure["verified_claim_id"])]
            )
            later_target_binding = build_reproduction_target_binding(
                repository.path,
                target_ids,
                ref="HEAD",
            )
            self.assertNotEqual(
                original["commit"], later_target_binding["audit_commit"]
            )
            split_base = {
                key: value
                for key, value in original.items()
                if key not in {"receipt_id", "content_hash"}
            }
            split_base["target_binding"] = later_target_binding
            split_hash = content_hash(split_base)
            split_receipt = {
                **split_base,
                "receipt_id": f"rr-{split_hash.split(':', 1)[1][:16]}",
                "content_hash": split_hash,
            }
            independence = require_independent_evaluator(
                repository,
                evaluator_id="split-checkpoint-reproducer",
                target_ids=target_ids,
                label="split checkpoint reproduction",
            )
            split_reproduction = repository.record(
                "reproduction",
                {
                    "reproduction_level": "computational_rerun",
                    "verdict": "passed",
                    "receipt_hash": split_hash,
                    "receipt": split_receipt,
                    "independence": independence,
                },
                state="verified",
                relations=[
                    {"type": "reproduces", "target": target_id}
                    for target_id in target_ids
                ],
                actor={
                    "actor_id": "split-checkpoint-reproducer",
                    "authority": "independent_evaluator",
                },
            )
            repository.commit(
                stage="review", subject="record split-checkpoint reproduction"
            )

            audit = repository.audit(level="verify")
            split_blockers = [
                item
                for item in audit["blockers"]
                if item["code"] == "invalid_reproduction_receipt"
                and item["object_id"] == split_reproduction.object_id
            ]
            self.assertTrue(split_blockers, audit["blockers"])
            self.assertFalse(audit["complete"])

    def test_rehashed_receipt_cannot_claim_a_stronger_execution_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            original = repository.get(str(closure["reproduction_id"]))["payload"][
                "receipt"
            ]
            forged_base = {
                key: value
                for key, value in json.loads(json.dumps(original)).items()
                if key not in {"receipt_id", "content_hash"}
            }
            forged_base["execution_isolation"]["isolated"] = True
            forged_hash = content_hash(forged_base)
            forged_receipt = {
                **forged_base,
                "receipt_id": f"rr-{forged_hash.split(':', 1)[1][:16]}",
                "content_hash": forged_hash,
            }
            target_ids = sorted(
                [str(closure["attempt_id"]), str(closure["verified_claim_id"])]
            )
            independence = require_independent_evaluator(
                repository,
                evaluator_id="false-isolation-reproducer",
                target_ids=target_ids,
                label="false isolation reproduction",
            )
            forged = repository.record(
                "reproduction",
                {
                    "reproduction_level": "computational_rerun",
                    "verdict": "passed",
                    "receipt_hash": forged_hash,
                    "receipt": forged_receipt,
                    "independence": independence,
                },
                state="verified",
                relations=[
                    {"type": "reproduces", "target": target_id}
                    for target_id in target_ids
                ],
                actor={
                    "actor_id": "false-isolation-reproducer",
                    "authority": "independent_evaluator",
                },
            )
            repository.commit(stage="review", subject="record forged isolation")

            audit = repository.audit(level="verify")
            blockers = [
                item
                for item in audit["blockers"]
                if item["code"] == "invalid_reproduction_receipt"
                and item["object_id"] == forged.object_id
            ]
            self.assertTrue(blockers, audit["blockers"])
            self.assertFalse(audit["complete"])

    def test_lifecycle_rejects_rehashed_inconsistent_v2_inner_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            original = repository.get(str(closure["reproduction_id"]))["payload"][
                "receipt"
            ]
            targets = [
                str(closure["attempt_id"]),
                str(closure["verified_claim_id"]),
            ]

            for binding_kind in ("checkpoint", "execution"):
                with self.subTest(binding_kind=binding_kind):
                    forged_base = {
                        key: value
                        for key, value in json.loads(json.dumps(original)).items()
                        if key not in {"receipt_id", "content_hash"}
                    }
                    if binding_kind == "checkpoint":
                        binding = forged_base["checkpoint_binding"]
                        binding["checkpoint_id"] = "rcp-forged-binding"
                        binding_core = {
                            key: value
                            for key, value in binding.items()
                            if key != "binding_hash"
                        }
                        binding["binding_hash"] = content_hash(binding_core)
                    else:
                        binding = forged_base["execution_result"]
                        binding["verdict"] = "failed"
                        binding_core = {
                            key: value
                            for key, value in binding.items()
                            if key != "result_hash"
                        }
                        binding["result_hash"] = content_hash(binding_core)
                    forged_hash = content_hash(forged_base)
                    forged = {
                        **forged_base,
                        "receipt_id": f"rr-{forged_hash.split(':', 1)[1][:16]}",
                        "content_hash": forged_hash,
                    }

                    with self.assertRaisesRegex(
                        ResearchGitError, "binding is inconsistent"
                    ):
                        ResearchLifecycle(repository).reproduction(
                            forged,
                            reproduces=targets,
                            verifier_id=f"forged-{binding_kind}-reproducer",
                            verified=True,
                        )

    def test_raw_legacy_receipt_is_retained_but_cannot_verify(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            original = repository.get(str(closure["reproduction_id"]))["payload"][
                "receipt"
            ]
            legacy_base = {
                key: value
                for key, value in original.items()
                if key
                not in {
                    "receipt_id",
                    "content_hash",
                    "checkpoint_binding",
                    "target_binding",
                    "execution_result",
                    "execution_isolation",
                    "stdout_truncated",
                    "stderr_truncated",
                    "output_capture",
                    "max_output_chars",
                }
            }
            legacy_base["schema_version"] = "xscientist.reproduction-receipt.v1"
            legacy_hash = content_hash(legacy_base)
            legacy = {
                **legacy_base,
                "receipt_id": f"rr-{legacy_hash.split(':', 1)[1][:16]}",
                "content_hash": legacy_hash,
            }
            repository.record(
                "challenge",
                {"reason": "replace the bound reproduction with a legacy receipt"},
                state="completed",
                relations=[
                    {
                        "type": "supersedes",
                        "target": str(closure["reproduction_id"]),
                    }
                ],
            )
            target_ids = [
                str(closure["attempt_id"]),
                str(closure["verified_claim_id"]),
            ]
            independence = require_independent_evaluator(
                repository,
                evaluator_id="legacy-raw-reproducer",
                target_ids=target_ids,
                label="legacy raw reproduction",
            )
            repository.record(
                "reproduction",
                {
                    "reproduction_level": "computational_rerun",
                    "verdict": "passed",
                    "receipt_hash": legacy_hash,
                    "receipt": legacy,
                    "independence": independence,
                },
                state="verified",
                relations=[
                    {"type": "reproduces", "target": target_id}
                    for target_id in target_ids
                ],
                actor={
                    "actor_id": "legacy-raw-reproducer",
                    "authority": "independent_evaluator",
                },
            )
            repository.commit(stage="review", subject="record legacy reproduction")

            audit = repository.audit(level="verify")

            self.assertFalse(audit["complete"])
            self.assertIn(
                "legacy_reproduction_receipt_unbound",
                {item["code"] for item in audit["warnings"]},
            )
            self.assertIn(
                "missing_verified_reproduction",
                {item["code"] for item in audit["blockers"]},
            )

    def test_active_claim_challenges_block_only_verification_closure(self) -> None:
        for relation_type in (
            "refutes",
            "qualified_refutes",
            "contradicts",
            "challenges_inference",
        ):
            with (
                self.subTest(relation_type=relation_type),
                tempfile.TemporaryDirectory() as td,
            ):
                repository = self._repository(Path(td) / "research")
                closure = self._record_verified_closure(repository)
                challenge_target = str(closure["draft_claim_id"])
                if relation_type == "challenges_inference":
                    inference = repository.record(
                        "inference",
                        {"statement": "The original evidence supports the claim."},
                        state="completed",
                        relations=[
                            {
                                "type": "has_premise",
                                "target": str(closure["evidence_id"]),
                            },
                            {
                                "type": "supports",
                                "target": str(closure["draft_claim_id"]),
                            },
                        ],
                    )
                    challenge_target = inference.object_id
                challenge = repository.record(
                    "evidence",
                    {
                        "result": f"negative result via {relation_type}",
                        "measurement_hash": "sha256:" + "7" * 64,
                    },
                    state="completed",
                    relations=[
                        {
                            "type": "derived_from",
                            "target": str(closure["attempt_id"]),
                        },
                        {
                            "type": relation_type,
                            # Draft claim versions and their inference closure
                            # remain visible after semantic promotion.
                            "target": challenge_target,
                        },
                    ],
                )
                repository.commit(stage="review", subject="record active challenge")

                trace = repository.audit(level="trace")
                replay = repository.audit(level="replay")
                verified = repository.audit(level="verify")

                self.assertTrue(trace["complete"], trace["blockers"])
                self.assertTrue(replay["complete"], replay["blockers"])
                self.assertFalse(verified["complete"])
                active = [
                    item
                    for item in verified["blockers"]
                    if item["code"] == "active_claim_challenge"
                ]
                self.assertEqual(
                    [item["object_id"] for item in active], [challenge.object_id]
                )

    def test_gate_and_one_review_must_cover_every_active_claim_signal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            unreviewed_support = repository.record(
                "evidence",
                {
                    "result": "independent additional support",
                    "measurement_hash": "sha256:" + "8" * 64,
                },
                state="completed",
                relations=[
                    {
                        "type": "derived_from",
                        "target": str(closure["attempt_id"]),
                    },
                    {
                        "type": "supports",
                        "target": str(closure["verified_claim_id"]),
                    },
                ],
            )
            repository.commit(stage="review", subject="add unreviewed support")

            audit = repository.audit(level="verify")
            codes = {item["code"] for item in audit["blockers"]}

            self.assertFalse(audit["complete"])
            self.assertIn("incomplete_gate_closure", codes)
            self.assertIn("incomplete_review_closure", codes)
            self.assertIn("invalid_reproduction_receipt", codes)
            self.assertNotIn("active_claim_challenge", codes)
            self.assertIn(
                unreviewed_support.object_id,
                next(
                    item["message"]
                    for item in audit["blockers"]
                    if item["code"] == "incomplete_gate_closure"
                ),
            )

    def test_challenge_to_experimental_lineage_blocks_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            challenge = repository.record(
                "evidence",
                {
                    "result": "The underlying attempt is not scientifically valid.",
                    "measurement_hash": "sha256:" + "7" * 64,
                },
                state="completed",
                relations=[
                    {
                        "type": "derived_from",
                        "target": str(closure["attempt_id"]),
                    },
                    {
                        "type": "contradicts",
                        "target": str(closure["attempt_id"]),
                    },
                ],
            )
            repository.commit(stage="review", subject="challenge attempt lineage")

            audit = repository.audit(level="verify")

            self.assertFalse(audit["complete"])
            self.assertIn(
                challenge.object_id,
                {
                    item["object_id"]
                    for item in audit["blockers"]
                    if item["code"] == "active_claim_challenge"
                },
            )
            gate_blocker = next(
                item
                for item in audit["blockers"]
                if item["code"] == "incomplete_gate_closure"
            )
            self.assertIn(str(closure["attempt_id"]), gate_blocker["message"])

    def test_bare_supersession_cannot_hide_challenge_or_reuse_old_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            challenge = repository.record(
                "evidence",
                {
                    "result": "negative result",
                    "measurement_hash": "sha256:" + "7" * 64,
                },
                state="completed",
                relations=[
                    {
                        "type": "derived_from",
                        "target": str(closure["attempt_id"]),
                    },
                    {
                        "type": "refutes",
                        "target": str(closure["draft_claim_id"]),
                    },
                ],
            )
            repository.record(
                "challenge",
                {"reason": "The negative measurement was explicitly resolved."},
                state="completed",
                relations=[{"type": "supersedes", "target": challenge.object_id}],
            )
            repository.commit(stage="review", subject="resolve old challenge")

            audit = repository.audit(level="verify")

            self.assertFalse(audit["complete"])
            self.assertIn(
                challenge.object_id,
                audit["claims"][0]["evidence_ids"],
            )
            self.assertIn(
                "unreviewed_challenge_resolution",
                {item["code"] for item in audit["blockers"]},
            )

    def test_reviewed_challenge_resolution_requires_and_accepts_fresh_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            challenge = repository.record(
                "evidence",
                {
                    "result": "negative result",
                    "measurement_hash": "sha256:" + "7" * 64,
                },
                state="completed",
                relations=[
                    {
                        "type": "derived_from",
                        "target": str(closure["attempt_id"]),
                    },
                    {
                        "type": "refutes",
                        "target": str(closure["draft_claim_id"]),
                    },
                ],
            )
            resolution = repository.record(
                "challenge",
                {"reason": "A follow-up measurement resolved the contradiction."},
                state="completed",
                relations=[{"type": "supersedes", "target": challenge.object_id}],
            )
            review_targets = [
                str(closure["evidence_id"]),
                challenge.object_id,
                resolution.object_id,
            ]
            independence = require_independent_evaluator(
                repository,
                evaluator_id="resolution-reviewer",
                target_ids=review_targets,
                label="challenge resolution review",
            )
            review = repository.record(
                "review",
                {
                    "summary": "Independent review accepts the recorded resolution.",
                    "status": "verified",
                    "independence": independence,
                },
                state="verified",
                relations=[
                    {"type": "evaluates", "target": target} for target in review_targets
                ],
                actor={
                    "actor_id": "resolution-reviewer",
                    "authority": "independent_evaluator",
                },
            )
            gate = repository.record(
                "gate_decision",
                {"decision": "promote", "claim_promotion_allowed": True},
                state="verified",
                relations=[{"type": "evaluates", "target": review.object_id}],
                actor={
                    "actor_id": "resolution-gate",
                    "authority": "deterministic_gate",
                },
            )
            old_claim = repository.get(str(closure["verified_claim_id"]))
            revised_claim = repository.record(
                "claim",
                old_claim["payload"],
                state="verified",
                relations=[
                    {
                        "type": "depends_on",
                        "target": str(closure["evidence_id"]),
                    },
                    {"type": "depends_on", "target": gate.object_id, "role": "gate"},
                    {
                        "type": "supersedes",
                        "target": str(closure["verified_claim_id"]),
                    },
                ],
            )
            source_checkpoint = create_checkpoint(
                repository.path,
                stage="review",
                subject="review challenge resolution",
                reproduce_command="python verify.py",
            )
            self._record_verified_reproduction(
                repository,
                attempt_id=str(closure["attempt_id"]),
                claim_id=revised_claim.object_id,
                source_checkpoint=source_checkpoint,
                verifier_id="resolution-reproducer",
            )
            repository.commit(
                stage="review", subject="record fresh challenge reproduction"
            )

            audit = repository.audit(level="verify")

            self.assertTrue(audit["complete"], audit["blockers"])

    def test_superseded_gate_review_and_reproduction_cannot_be_reused(self) -> None:
        for target_kind, expected_code in (
            ("gate", "inactive_gate"),
            ("review", "inactive_review"),
            ("reproduction", "inactive_reproduction"),
        ):
            with (
                self.subTest(target_kind=target_kind),
                tempfile.TemporaryDirectory() as td,
            ):
                repository = self._repository(Path(td) / "research")
                closure = self._record_verified_closure(repository)
                if target_kind == "review":
                    gate = repository.get(str(closure["gate_id"]))
                    target_id = next(
                        relation["target"]
                        for relation in gate["relations"]
                        if relation["type"] == "evaluates"
                    )
                else:
                    target_id = str(closure[f"{target_kind}_id"])
                repository.record(
                    "challenge",
                    {"reason": f"The prior {target_kind} is no longer authoritative."},
                    state="completed",
                    relations=[{"type": "supersedes", "target": target_id}],
                )
                repository.commit(
                    stage="review", subject=f"supersede prior {target_kind}"
                )

                audit = repository.audit(level="verify")

                self.assertFalse(audit["complete"])
                self.assertIn(
                    expected_code,
                    {item["code"] for item in audit["blockers"]},
                )

    def test_unreviewed_warrant_is_part_of_full_gate_closure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            warrant = repository.record(
                "warrant",
                {"statement": "A newly recorded assumption links result to claim."},
                state="completed",
                relations=[
                    {
                        "type": "supports",
                        "target": str(closure["verified_claim_id"]),
                    }
                ],
            )
            repository.commit(stage="review", subject="add unreviewed warrant")

            audit = repository.audit(level="verify")

            self.assertFalse(audit["complete"])
            self.assertIn(warrant.object_id, audit["claims"][0]["argument_ids"])
            self.assertIn(
                "incomplete_gate_closure",
                {item["code"] for item in audit["blockers"]},
            )

    def test_claim_hash_change_cannot_shed_ancestor_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            closure = self._record_verified_closure(repository)
            challenge = repository.record(
                "evidence",
                {
                    "result": "negative result on the draft claim",
                    "measurement_hash": "sha256:" + "7" * 64,
                },
                state="completed",
                relations=[
                    {
                        "type": "derived_from",
                        "target": str(closure["attempt_id"]),
                    },
                    {
                        "type": "refutes",
                        "target": str(closure["draft_claim_id"]),
                    },
                ],
            )
            repository.record(
                "claim",
                {
                    "statement": "H1 improves the metric.",
                    "claim_hash": "sha256:" + "8" * 64,
                },
                state="verified",
                relations=[
                    {
                        "type": "depends_on",
                        "target": str(closure["evidence_id"]),
                    },
                    {
                        "type": "depends_on",
                        "target": str(closure["gate_id"]),
                        "role": "gate",
                    },
                    {
                        "type": "supersedes",
                        "target": str(closure["verified_claim_id"]),
                    },
                ],
            )
            repository.commit(stage="review", subject="revise immutable claim")

            audit = repository.audit(level="verify")

            self.assertFalse(audit["complete"])
            active = [
                item
                for item in audit["blockers"]
                if item["code"] == "active_claim_challenge"
            ]
            self.assertEqual(
                [item["object_id"] for item in active], [challenge.object_id]
            )

    def test_lifecycle_rejects_split_reviews_of_selected_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            lifecycle = ResearchLifecycle(repository)
            evidence_ids = [
                repository.record(
                    "evidence",
                    {"result": result, "measurement_hash": "sha256:" + digit * 64},
                    state="completed",
                ).object_id
                for result, digit in (("first", "1"), ("second", "2"))
            ]
            reviews = [
                repository.record(
                    "review",
                    {"status": "verified", "claim_promotion_allowed": True},
                    state="verified",
                    relations=[{"type": "evaluates", "target": evidence_id}],
                )
                for evidence_id in evidence_ids
            ]
            gate = repository.record(
                "gate_decision",
                {"decision": "promote", "claim_promotion_allowed": True},
                state="verified",
                relations=[
                    {"type": "evaluates", "target": review.object_id}
                    for review in reviews
                ],
            )

            with self.assertRaisesRegex(ResearchGitError, "one review covering"):
                lifecycle.claim(
                    {"statement": "Both measurements support the claim."},
                    evidence_ids=evidence_ids,
                    gate_id=gate.object_id,
                    verified=True,
                    commit=False,
                )

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
            run_json(
                "stage",
                "requirements.txt",
                ".xscientist/verify.py",
                "--repo",
                str(root),
            )
            run_json(
                "checkpoint",
                "--repo",
                str(root),
                "--stage",
                "environment",
                "--subject",
                "bind the reproduction environment",
                "--staged",
            )
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
            reproduction_checkpoint = run_json(
                "checkpoint",
                "--repo",
                str(root),
                "--stage",
                "review",
                "--subject",
                "bind the verified claim reproduction",
                "--reproduce",
                "python .xscientist/verify.py",
                "--allow-checkpoint-only",
            )["commit"]

            replay = run_json("audit", "--repo", str(root), "--level", "replay")
            self.assertTrue(replay["complete"], replay["blockers"])
            run_json(
                "reproduce",
                reproduction_checkpoint,
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
