from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ai_scientist.protocol.attestation import sign_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.experiment_admission import (
    EXPERIMENT_ADMISSION_AUTHORITY_PROTOCOL,
    EXPERIMENT_ADMISSION_PROTOCOL,
    XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE,
    build_experiment_admission_payload,
    verify_experiment_admission_attestation,
    verify_persisted_experiment_admission_gate,
)
from xscientist.research_git import ResearchGitError


class PersistedExperimentAdmissionReceiptValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            self.skipTest("cryptography is required for Ed25519 admission tests")
        self.serialization = serialization
        self.Ed25519PrivateKey = Ed25519PrivateKey

    @staticmethod
    def _binding(object_id: str, digit: str) -> dict[str, str]:
        return {
            "object_id": object_id,
            "content_hash": f"sha256:{digit * 64}",
        }

    def _write_key(
        self,
        authority_root: Path,
        *,
        filename: str,
    ) -> tuple[bytes, str]:
        private_key = self.Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(
            encoding=self.serialization.Encoding.PEM,
            format=self.serialization.PrivateFormat.PKCS8,
            encryption_algorithm=self.serialization.NoEncryption(),
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=self.serialization.Encoding.PEM,
            format=self.serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        (authority_root / filename).write_bytes(public_pem)
        return private_pem, filename

    @staticmethod
    def _write_trust_store(
        trust_store: Path,
        *,
        key_id: str,
        identity: str,
        public_key_file: str,
        revoked: bool = False,
        extras: dict | None = None,
    ) -> None:
        keys = {
            key_id: {
                "identity": identity,
                "algorithm": "ed25519",
                "public_key_file": public_key_file,
                "revoked": revoked,
            }
        }
        keys.update(extras or {})
        trust_store.write_text(json.dumps({"keys": keys}), encoding="utf-8")

    @staticmethod
    def _rehash(gate: dict) -> None:
        authority = gate.get("authority_receipt")
        if isinstance(authority, dict):
            authority["receipt_hash"] = canonical_content_hash(
                {
                    key: value
                    for key, value in authority.items()
                    if key != "receipt_hash"
                }
            )
        gate["gate_hash"] = canonical_content_hash(
            {key: value for key, value in gate.items() if key != "gate_hash"}
        )

    def _fixture(
        self,
        root: Path,
        *,
        signer_identity: str = "human:independent-reviewer",
        scope: dict | None = None,
        producer_actor_ids: list[str] | None = None,
        producer_principals: list[str] | None = None,
    ) -> tuple[dict, Path, Path, str, str]:
        boundary = root / "research"
        boundary.mkdir(parents=True)
        authority_root = root / "external-authority"
        authority_root.mkdir()
        private_pem, public_key_file = self._write_key(
            authority_root, filename="reviewer.pub"
        )
        key_id = "key:independent-reviewer"
        trust_store = authority_root / "trust.json"
        self._write_trust_store(
            trust_store,
            key_id=key_id,
            identity=signer_identity,
            public_key_file=public_key_file,
        )
        payload = build_experiment_admission_payload(
            repository_id="ws-0123456789abcdef",
            repository_head="a" * 40,
            decision="authorize",
            hypothesis=self._binding("rso-1111111111111111", "1"),
            plan=self._binding("rso-2222222222222222", "2"),
            preregistration=None,
            execution_scope=scope or {"task_id": "task-1"},
        )
        attestation = sign_attestation(
            payload,
            purpose=EXPERIMENT_ADMISSION_PROTOCOL,
            identity=signer_identity,
            key_id=key_id,
            algorithm="ed25519",
            key=private_pem,
            issued_at="2026-09-04T00:00:00+00:00",
        )
        verified = verify_experiment_admission_attestation(
            payload,
            attestation,
            trust_store_path=trust_store,
            forbidden_roots=(boundary,),
        )
        actor_ids = producer_actor_ids or ["agent:planner"]
        principals = producer_principals or ["planner"]
        lineage_ids = ["rso-1111111111111111", "rso-2222222222222222"]
        request_hash = canonical_content_hash(payload)
        authority_core = {
            "protocol_kind": EXPERIMENT_ADMISSION_AUTHORITY_PROTOCOL,
            "verifier_identity": verified["verifier_identity"],
            "verifier_principal": verified["verifier_principal"],
            "key_id": verified["key_id"],
            "algorithm": verified["algorithm"],
            "attestation_hash": verified["attestation_hash"],
            "trust_store_hash": verified["trust_store_hash"],
            "signed_payload_hash": request_hash,
            "producer_actor_ids": actor_ids,
            "producer_principals": principals,
            "lineage_object_ids": lineage_ids,
            "signed_payload": deepcopy(payload),
            "attestation_record": deepcopy(verified["attestation_record"]),
        }
        authority_receipt = {
            **authority_core,
            "receipt_hash": canonical_content_hash(authority_core),
        }
        gate_core = {
            "protocol_kind": EXPERIMENT_ADMISSION_PROTOCOL,
            "gate_id": (
                "admission-"
                + request_hash.removeprefix("sha256:")[:16]
                + "-"
                + verified["attestation_hash"].removeprefix("sha256:")[:16]
            ),
            "decision": payload["decision"],
            "experiment_authorized": True,
            "claim_promotion_allowed": False,
            "repository_id": payload["repository_id"],
            "repository_head": payload["repository_head"],
            "input_bindings": deepcopy(payload["input_bindings"]),
            "scope": deepcopy(payload["scope"]),
            "scope_hash": payload["scope_hash"],
            "admission_request_hash": request_hash,
            "authority_receipt": authority_receipt,
        }
        gate = {**gate_core, "gate_hash": canonical_content_hash(gate_core)}
        return gate, trust_store, boundary, key_id, public_key_file

    def _verify(self, gate: dict, trust_store: Path, boundary: Path) -> dict:
        return verify_persisted_experiment_admission_gate(
            gate,
            trust_store_path=trust_store,
            forbidden_roots=(boundary,),
            expected_producer_actor_ids=["agent:planner"],
            expected_producer_principals=["reviewer:planner"],
            expected_lineage_object_ids=[
                "rso-2222222222222222",
                "rso-1111111111111111",
            ],
        )

    def test_real_signature_round_trip_reconstructs_persisted_authority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate, trust_store, boundary, _, _ = self._fixture(Path(td))

            result = self._verify(gate, trust_store, boundary)

            self.assertTrue(result["experiment_authorized"])
            self.assertEqual(result["verifier_principal"], "independent-reviewer")
            self.assertEqual(result["producer_principals"], ["planner"])
            self.assertEqual(
                XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE,
                "XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE",
            )

    def test_missing_receipt_and_extra_signed_field_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate, trust_store, boundary, _, _ = self._fixture(Path(td))
            missing = deepcopy(gate)
            del missing["authority_receipt"]
            with self.assertRaisesRegex(ResearchGitError, "gate payload fields"):
                self._verify(missing, trust_store, boundary)

            extra = deepcopy(gate)
            extra["authority_receipt"]["signed_payload"]["unsigned_extra"] = True
            self._rehash(extra)
            with self.assertRaisesRegex(ResearchGitError, "payload fields"):
                self._verify(extra, trust_store, boundary)

    def test_forged_hashes_and_repository_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate, trust_store, boundary, _, _ = self._fixture(Path(td))
            forged = deepcopy(gate)
            forged["authority_receipt"]["receipt_hash"] = f"sha256:{'0' * 64}"
            forged["gate_hash"] = canonical_content_hash(
                {key: value for key, value in forged.items() if key != "gate_hash"}
            )
            with self.assertRaisesRegex(ResearchGitError, "receipt hash"):
                self._verify(forged, trust_store, boundary)

            rebound = deepcopy(gate)
            rebound["repository_id"] = "ws-fedcba9876543210"
            reconstructed = deepcopy(rebound["authority_receipt"]["signed_payload"])
            reconstructed["repository_id"] = rebound["repository_id"]
            rebound["admission_request_hash"] = canonical_content_hash(reconstructed)
            rebound["gate_hash"] = canonical_content_hash(
                {key: value for key, value in rebound.items() if key != "gate_hash"}
            )
            with self.assertRaisesRegex(ResearchGitError, "signed payload.*gate"):
                self._verify(rebound, trust_store, boundary)

    def test_wrong_signature_fails_even_when_outer_hashes_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate, trust_store, boundary, _, _ = self._fixture(Path(td))
            tampered = deepcopy(gate)
            record = tampered["authority_receipt"]["attestation_record"]
            record["signature_hex"] = "0" * 128
            self._rehash(tampered)

            with self.assertRaisesRegex(ResearchGitError, "verification failed"):
                self._verify(tampered, trust_store, boundary)

    def test_revoked_key_fails_under_current_trust_policy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate, trust_store, boundary, key_id, public_key_file = self._fixture(root)
            self._write_trust_store(
                trust_store,
                key_id=key_id,
                identity="human:independent-reviewer",
                public_key_file=public_key_file,
                revoked=True,
            )

            with self.assertRaisesRegex(ResearchGitError, "key_revoked"):
                self._verify(gate, trust_store, boundary)

    def test_trust_policy_change_invalidates_persisted_policy_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate, trust_store, boundary, key_id, public_key_file = self._fixture(root)
            _, second_public_key = self._write_key(
                trust_store.parent, filename="second-reviewer.pub"
            )
            self._write_trust_store(
                trust_store,
                key_id=key_id,
                identity="human:independent-reviewer",
                public_key_file=public_key_file,
                extras={
                    "key:second-reviewer": {
                        "identity": "human:second-reviewer",
                        "algorithm": "ed25519",
                        "public_key_file": second_public_key,
                    }
                },
            )

            with self.assertRaisesRegex(ResearchGitError, "current policy"):
                self._verify(gate, trust_store, boundary)

    def test_signer_aliases_cannot_overlap_persisted_or_scope_producer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            persisted, trust_store, boundary, _, _ = self._fixture(
                root / "persisted",
                signer_identity="human:reviewer",
                producer_actor_ids=["executor:reviewer"],
                producer_principals=["reviewer"],
            )
            with self.assertRaisesRegex(ResearchGitError, "independent of producers"):
                verify_persisted_experiment_admission_gate(
                    persisted,
                    trust_store_path=trust_store,
                    forbidden_roots=(boundary,),
                )

            scoped, trust_store, boundary, _, _ = self._fixture(
                root / "scoped",
                signer_identity="human:reviewer",
                scope={
                    "task_id": "task-1",
                    "producer_id": "executor:reviewer",
                },
            )
            with self.assertRaisesRegex(
                ResearchGitError, "independent of scope producer"
            ):
                verify_persisted_experiment_admission_gate(
                    scoped,
                    trust_store_path=trust_store,
                    forbidden_roots=(boundary,),
                )

    def test_recomputed_producer_closure_must_match_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate, trust_store, boundary, _, _ = self._fixture(Path(td))

            with self.assertRaisesRegex(ResearchGitError, "closure must be complete"):
                verify_persisted_experiment_admission_gate(
                    gate,
                    trust_store_path=trust_store,
                    forbidden_roots=(boundary,),
                    expected_producer_principals=["planner"],
                )
            with self.assertRaisesRegex(ResearchGitError, "lineage closure"):
                verify_persisted_experiment_admission_gate(
                    gate,
                    trust_store_path=trust_store,
                    forbidden_roots=(boundary,),
                    expected_producer_actor_ids=["agent:planner"],
                    expected_producer_principals=["planner"],
                    expected_lineage_object_ids=["rso-3333333333333333"],
                )

            unrelated = deepcopy(gate)
            unrelated["authority_receipt"]["lineage_object_ids"] = [
                "rso-3333333333333333"
            ]
            self._rehash(unrelated)
            with self.assertRaisesRegex(ResearchGitError, "cover signed inputs"):
                verify_persisted_experiment_admission_gate(
                    unrelated,
                    trust_store_path=trust_store,
                    forbidden_roots=(boundary,),
                )

    def test_trust_store_path_and_forbidden_roots_are_both_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate, trust_store, boundary, _, _ = self._fixture(Path(td))

            with self.assertRaisesRegex(ResearchGitError, "explicit external"):
                verify_persisted_experiment_admission_gate(
                    gate,
                    trust_store_path="",
                    forbidden_roots=(boundary,),
                )
            with self.assertRaisesRegex(ResearchGitError, "non-empty iterable"):
                verify_persisted_experiment_admission_gate(
                    gate,
                    trust_store_path=trust_store,
                    forbidden_roots=(),
                )


if __name__ == "__main__":
    unittest.main()
