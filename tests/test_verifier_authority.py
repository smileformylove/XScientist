from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai_scientist.protocol.attestation import sign_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import schema_validator
from ai_scientist.utils.research_integrity import (
    VERIFICATION_REQUIRED_CRITERIA,
    _canonical_hash,
    _verification_report_hash_payload,
)
from ai_scientist.utils.verifier_authority import (
    VERIFIER_AUTHORITY_PURPOSE,
    build_verifier_authority_payload,
    build_verifier_authority_receipt,
    verify_verifier_authority_receipt,
)

PRODUCER_IDS = ["agent:experiment-executor", "service:training-runtime"]
VERIFIER_IDENTITY = "human:independent-reviewer"
KEY_ID = "key:independent-reviewer"
DATA_MANIFEST_HASH = "sha256:" + "6" * 64
DATA_SNAPSHOT_ID = "sha256:" + "7" * 64
DATA_BINDING = {
    "data_manifest_hash": DATA_MANIFEST_HASH,
    "data_snapshot_id": DATA_SNAPSHOT_ID,
}


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _verification_report() -> dict:
    report = {
        "schema_version": 1,
        "generated_at": "2026-08-30T00:00:00+00:00",
        "preregistration_id": "prereg-1",
        "registration_hash": _digest("1"),
        "verifier_id": "independent-reviewer",
        "clean_room": True,
        "status": "verified",
        "claim_promotion_allowed": True,
        "confirmatory_record_ids": ["confirm-2", "confirm-1"],
        "reproduction_record_ids": ["reproduce-2", "reproduce-1"],
        "reproduction_task_counts": {"task-1": 2},
        "conclusion_interpretation": {"status": "supports_effect"},
        "analysis_plan_contract": {"ok": True},
        "records_hash": _digest("2"),
        "registry_hash": _digest("3"),
        "evidence_snapshot_hash": _digest("4"),
        "manuscript_hash": _digest("5"),
        "data_manifest_hash": DATA_MANIFEST_HASH,
        "data_snapshot_id": DATA_SNAPSHOT_ID,
        "trajectory_binding_hash": _digest("8"),
        "research_vcs_frozen_head": "a" * 40,
        "research_vcs_lineage_head": "b" * 40,
        "research_vcs_attempt_object_ids": ["rso-1111111111111111"],
        "research_vcs_binding_object_ids": ["rso-2222222222222222"],
        "research_vcs_checkpoint_hashes": [_digest("9")],
        "failed_record_ids": ["confirm-2"],
        "research_vcs_disposition_object_ids": ["rso-3333333333333333"],
        "criteria": [
            {
                "id": criterion_id,
                "passed": True,
                "required": True,
                "detail": "verified",
            }
            for criterion_id in sorted(
                {
                    *VERIFICATION_REQUIRED_CRITERIA,
                    "data_contract_binding",
                    "trajectory_binding",
                }
            )
        ],
        "required_failures": [],
    }
    report["report_hash"] = _canonical_hash(_verification_report_hash_payload(report))
    return report


def _ed25519_keypair() -> tuple[bytes, bytes]:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    private_key = Ed25519PrivateKey.generate()
    return (
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )


def _signed_receipt(report: dict, *, private_key: bytes, key_id: str = KEY_ID) -> dict:
    payload = build_verifier_authority_payload(
        report, PRODUCER_IDS, VERIFIER_IDENTITY, **DATA_BINDING
    )
    attestation = sign_attestation(
        payload,
        purpose=VERIFIER_AUTHORITY_PURPOSE,
        identity=VERIFIER_IDENTITY,
        key_id=key_id,
        algorithm="ed25519",
        key=private_key,
        issued_at="2026-08-30T00:00:00+00:00",
    )
    return build_verifier_authority_receipt(
        report, PRODUCER_IDS, VERIFIER_IDENTITY, attestation, **DATA_BINDING
    )


def _write_trust_store(
    root: Path,
    public_key: bytes,
    *,
    key_id: str = KEY_ID,
    identity: str = VERIFIER_IDENTITY,
) -> Path:
    public_path = root / "independent-reviewer.pub"
    public_path.write_bytes(public_key)
    trust_path = root / "verifier-trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "keys": {
                    key_id: {
                        "identity": identity,
                        "algorithm": "ed25519",
                        "public_key_file": public_path.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return trust_path


def test_executor_alias_and_clean_room_claim_cannot_replace_signature(
    tmp_path: Path,
) -> None:
    report = _verification_report()
    report["verifier_id"] = "experiment-executor-alias"
    report["report_hash"] = _canonical_hash(_verification_report_hash_payload(report))
    payload = build_verifier_authority_payload(
        report,
        PRODUCER_IDS,
        "agent:experiment-executor-alias",
        **DATA_BINDING,
    )
    core = {
        "schema_version": "xscientist.verifier-authority.v1",
        "purpose": VERIFIER_AUTHORITY_PURPOSE,
        "payload": payload,
        "attestation": {},
    }
    receipt = {**core, "receipt_hash": canonical_content_hash(core)}

    result = verify_verifier_authority_receipt(
        receipt,
        report,
        PRODUCER_IDS,
        tmp_path / "missing-trust.json",
        **DATA_BINDING,
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert "attestation_algorithm_not_ed25519" in result["errors"]


def test_signer_identity_must_match_report_verifier() -> None:
    report = _verification_report()
    with pytest.raises(ValueError, match="must match"):
        build_verifier_authority_payload(
            report,
            PRODUCER_IDS,
            "human:different-reviewer",
            **DATA_BINDING,
        )


def test_self_signed_verifier_not_in_local_trust_store_is_rejected(
    tmp_path: Path,
) -> None:
    signing_private, _ = _ed25519_keypair()
    _, unrelated_public = _ed25519_keypair()
    report = _verification_report()
    receipt = _signed_receipt(report, private_key=signing_private)
    trust_path = _write_trust_store(
        tmp_path,
        unrelated_public,
        key_id="key:unrelated-reviewer",
        identity="human:unrelated-reviewer",
    )

    result = verify_verifier_authority_receipt(
        receipt, report, PRODUCER_IDS, trust_path, **DATA_BINDING
    )

    assert result["ok"] is False
    assert "attestation:key_untrusted" in result["errors"]


def test_tampered_report_binding_is_rejected(tmp_path: Path) -> None:
    private_key, public_key = _ed25519_keypair()
    report = _verification_report()
    receipt = _signed_receipt(report, private_key=private_key)
    trust_path = _write_trust_store(tmp_path, public_key)
    tampered = deepcopy(report)
    tampered["manuscript_hash"] = _digest("9")

    result = verify_verifier_authority_receipt(
        receipt, tampered, PRODUCER_IDS, trust_path, **DATA_BINDING
    )

    assert result["ok"] is False
    assert "report_binding_invalid" in result["errors"]


def test_valid_external_ed25519_authority_receipt_passes(tmp_path: Path) -> None:
    private_key, public_key = _ed25519_keypair()
    report = _verification_report()
    receipt = _signed_receipt(report, private_key=private_key)
    trust_path = _write_trust_store(tmp_path, public_key)

    result = verify_verifier_authority_receipt(
        receipt, report, PRODUCER_IDS, trust_path, **DATA_BINDING
    )

    schema_validator("verifier_authority").validate(receipt)
    schema_validator("verification_report").validate(report)
    assert result["ok"] is True, result
    assert result["status"] == "verified"
    assert result["errors"] == []
    assert result["verifier_identity"] == VERIFIER_IDENTITY
    assert result["report_hash"] == report["report_hash"]
    assert receipt["payload"]["research_vcs_disposition_object_ids"] == [
        "rso-3333333333333333"
    ]
    serialized = json.dumps(receipt)
    assert "PRIVATE KEY" not in serialized
    assert "PUBLIC KEY" not in serialized


def test_hmac_cannot_grant_independent_verifier_authority(tmp_path: Path) -> None:
    report = _verification_report()
    payload = build_verifier_authority_payload(
        report, PRODUCER_IDS, VERIFIER_IDENTITY, **DATA_BINDING
    )
    attestation = sign_attestation(
        payload,
        purpose=VERIFIER_AUTHORITY_PURPOSE,
        identity=VERIFIER_IDENTITY,
        key_id=KEY_ID,
        algorithm="hmac-sha256",
        key=b"test-only-shared-secret",
        issued_at="2026-08-30T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="requires Ed25519"):
        build_verifier_authority_receipt(
            report, PRODUCER_IDS, VERIFIER_IDENTITY, attestation, **DATA_BINDING
        )

    core = {
        "schema_version": "xscientist.verifier-authority.v1",
        "purpose": VERIFIER_AUTHORITY_PURPOSE,
        "payload": payload,
        "attestation": attestation,
    }
    receipt = {**core, "receipt_hash": canonical_content_hash(core)}
    trust_path = tmp_path / "hmac-trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "keys": {
                    KEY_ID: {
                        "identity": VERIFIER_IDENTITY,
                        "algorithm": "hmac-sha256",
                        "key_env": "IGNORED_SHARED_SECRET",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = verify_verifier_authority_receipt(
        receipt, report, PRODUCER_IDS, trust_path, **DATA_BINDING
    )

    assert result["ok"] is False
    assert "attestation_algorithm_not_ed25519" in result["errors"]
    assert "trust_store_algorithm_not_ed25519" in result["errors"]


def test_receipt_cannot_choose_workspace_trust_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, public_key = _ed25519_keypair()
    report = _verification_report()
    receipt = _signed_receipt(report, private_key=private_key)
    trust_path = _write_trust_store(tmp_path, public_key)
    receipt["trust_store_path"] = str(trust_path)
    monkeypatch.delenv("XSCIENTIST_VERIFIER_TRUST_STORE", raising=False)

    result = verify_verifier_authority_receipt(
        receipt, report, PRODUCER_IDS, **DATA_BINDING
    )

    assert result["ok"] is False
    assert "receipt_fields_invalid" in result["errors"]
    assert "trust_store_missing" in result["errors"]


def test_symlinked_public_key_is_rejected(tmp_path: Path) -> None:
    private_key, public_key = _ed25519_keypair()
    report = _verification_report()
    receipt = _signed_receipt(report, private_key=private_key)
    real_key = tmp_path / "real.pub"
    real_key.write_bytes(public_key)
    linked_key = tmp_path / "linked.pub"
    linked_key.symlink_to(real_key.name)
    trust_path = tmp_path / "trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "keys": {
                    KEY_ID: {
                        "identity": VERIFIER_IDENTITY,
                        "algorithm": "ed25519",
                        "public_key_file": linked_key.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = verify_verifier_authority_receipt(
        receipt, report, PRODUCER_IDS, trust_path, **DATA_BINDING
    )

    assert result["ok"] is False
    assert "public_key_symlink_rejected" in result["errors"]


def test_cross_prefix_principal_alias_is_not_independent() -> None:
    report = _verification_report()
    report["verifier_id"] = "executor"
    report["report_hash"] = _canonical_hash(_verification_report_hash_payload(report))

    with pytest.raises(ValueError, match="disjoint"):
        build_verifier_authority_payload(
            report,
            ["agent:executor"],
            "human:executor",
            **DATA_BINDING,
        )


def test_external_signer_cannot_use_a_workflow_role_namespace() -> None:
    report = _verification_report()
    report["verifier_id"] = "independent-reviewer"
    report["report_hash"] = _canonical_hash(_verification_report_hash_payload(report))

    with pytest.raises(
        ValueError,
        match="verifier_identity must use agent:, service:, or human:",
    ):
        build_verifier_authority_payload(
            report,
            PRODUCER_IDS,
            "verifier:independent-reviewer",
            **DATA_BINDING,
        )


def test_signed_receipt_cannot_be_reused_for_another_valid_data_snapshot(
    tmp_path: Path,
) -> None:
    private_key, public_key = _ed25519_keypair()
    report = _verification_report()
    receipt = _signed_receipt(report, private_key=private_key)
    trust_path = _write_trust_store(tmp_path, public_key)

    result = verify_verifier_authority_receipt(
        receipt,
        report,
        PRODUCER_IDS,
        trust_path,
        data_manifest_hash=_digest("8"),
        data_snapshot_id=_digest("9"),
    )

    assert result["ok"] is False
    assert "report_binding_invalid" in result["errors"]


def test_research_boundary_cannot_supply_top_venue_trust_root(
    tmp_path: Path,
) -> None:
    private_key, public_key = _ed25519_keypair()
    report = _verification_report()
    receipt = _signed_receipt(report, private_key=private_key)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trust_path = _write_trust_store(workspace, public_key)

    result = verify_verifier_authority_receipt(
        receipt,
        report,
        PRODUCER_IDS,
        trust_path,
        forbidden_trust_roots=[workspace],
        **DATA_BINDING,
    )

    assert result["ok"] is False
    assert "trust_store_inside_research_boundary" in result["errors"]


def test_malformed_verification_report_is_not_signable() -> None:
    report = _verification_report()
    report["unexpected_self_attestation"] = True
    report["report_hash"] = _canonical_hash(_verification_report_hash_payload(report))

    with pytest.raises(ValueError, match="schema_invalid"):
        build_verifier_authority_payload(
            report,
            PRODUCER_IDS,
            VERIFIER_IDENTITY,
            **DATA_BINDING,
        )
