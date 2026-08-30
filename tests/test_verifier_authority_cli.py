from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from ai_scientist.protocol.attestation import sign_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.high_quality_pipeline import build_scientific_evidence_gate
from ai_scientist.utils.research_integrity import (
    VERIFICATION_REQUIRED_CRITERIA,
    _canonical_hash,
    _verification_report_hash_payload,
)
from ai_scientist.utils.verifier_authority import VERIFIER_AUTHORITY_PURPOSE
from xscientist.entrypoints import research_main


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _write_publication_inputs(
    root: Path,
    *,
    producer_id: str = "experimenter",
    reproduction_producer_id: str | None = None,
    verifier_id: str = "independent-reviewer",
) -> None:
    root.mkdir()
    data = b"fixed empirical observations\n"
    files = [
        {
            "path": "observations.bin",
            "size_bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        }
    ]
    data_snapshot_id = canonical_content_hash({"files": files})
    snapshot_root = (
        root / ".ara-store" / "datasets" / data_snapshot_id.removeprefix("sha256:")
    )
    snapshot_root.mkdir(parents=True)
    data_path = snapshot_root / "observations.bin"
    data_path.write_bytes(data)
    data_path.chmod(0o444)
    snapshot_root.chmod(0o555)
    manifest_core = {
        "schema_version": "xscientist.data-contract.v1",
        "mode": "content_addressed_snapshot_read_only",
        "ready": True,
        "source_path_disclosed": False,
        "snapshot_id": data_snapshot_id,
        "file_count": 1,
        "total_bytes": len(data),
        "files": files,
        "scientific_boundary": "fixed empirical test data",
    }
    manifest = {
        **manifest_core,
        "manifest_hash": canonical_content_hash(manifest_core),
    }
    config_dir = root / "00_config"
    config_dir.mkdir()
    (config_dir / "data_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "generated_at": "2026-08-30T00:00:00+00:00",
        "preregistration_id": "prereg-1",
        "registration_hash": _digest("1"),
        "verifier_id": verifier_id,
        "clean_room": True,
        "status": "verified",
        "claim_promotion_allowed": True,
        "confirmatory_record_ids": ["confirm-1"],
        "reproduction_record_ids": ["reproduce-1"],
        "reproduction_task_counts": {"task-1": 1},
        "conclusion_interpretation": {"status": "supports_effect"},
        "analysis_plan_contract": {"ok": True},
        "records_hash": _digest("2"),
        "registry_hash": _digest("3"),
        "evidence_snapshot_hash": _digest("4"),
        "manuscript_hash": _digest("5"),
        "data_manifest_hash": manifest["manifest_hash"],
        "data_snapshot_id": data_snapshot_id,
        "trajectory_binding_hash": _digest("8"),
        "research_vcs_frozen_head": "a" * 40,
        "research_vcs_lineage_head": "b" * 40,
        "research_vcs_attempt_object_ids": ["rso-1111111111111111"],
        "research_vcs_binding_object_ids": ["rso-2222222222222222"],
        "research_vcs_checkpoint_hashes": [_digest("9")],
        "failed_record_ids": [],
        "research_vcs_disposition_object_ids": [],
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
    (root / "verification_report.json").write_text(json.dumps(report), encoding="utf-8")
    registry_rows = [
        {
            "record_id": "confirm-1",
            "study_phase": "confirmatory",
            "status": "completed",
            "producer_id": producer_id,
            "data_manifest_hash": manifest["manifest_hash"],
            "data_snapshot_id": data_snapshot_id,
        }
    ]
    if reproduction_producer_id is not None:
        registry_rows.append(
            {
                "record_id": "reproduce-1",
                "study_phase": "confirmatory",
                "status": "verified",
                "producer_id": reproduction_producer_id,
                "independent_reproduction": True,
                "data_manifest_hash": manifest["manifest_hash"],
                "data_snapshot_id": data_snapshot_id,
            }
        )
    (root / "experiment_registry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in registry_rows),
        encoding="utf-8",
    )


def _keypair() -> tuple[bytes, bytes]:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )


def _replace_data_contract(root: Path, data: bytes) -> dict:
    files = [
        {
            "path": "observations.bin",
            "size_bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        }
    ]
    snapshot_id = canonical_content_hash({"files": files})
    snapshot_root = (
        root / ".ara-store" / "datasets" / snapshot_id.removeprefix("sha256:")
    )
    snapshot_root.mkdir(parents=True)
    data_path = snapshot_root / "observations.bin"
    data_path.write_bytes(data)
    data_path.chmod(0o444)
    snapshot_root.chmod(0o555)
    core = {
        "schema_version": "xscientist.data-contract.v1",
        "mode": "content_addressed_snapshot_read_only",
        "ready": True,
        "source_path_disclosed": False,
        "snapshot_id": snapshot_id,
        "file_count": 1,
        "total_bytes": len(data),
        "files": files,
        "scientific_boundary": "replacement empirical test data",
    }
    manifest = {**core, "manifest_hash": canonical_content_hash(core)}
    (root / "00_config" / "data_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def test_verifier_authority_cli_prepare_finalize_and_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paper = tmp_path / "paper"
    _write_publication_inputs(paper)
    payload_path = tmp_path / "authority-payload.json"
    identity = "human:independent-reviewer"

    assert (
        research_main(
            [
                "verifier-authority",
                "prepare",
                "--paper-dir",
                str(paper),
                "--identity",
                identity,
                "--output",
                str(payload_path),
                "--json",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert prepared["status"] == "prepared"
    assert prepared["payload_hash"] == canonical_content_hash(payload)
    assert payload["producer_ids"] == ["experimenter"]
    assert payload["verifier_identity"] == identity

    private_key, public_key = _keypair()
    attestation = sign_attestation(
        payload,
        purpose=VERIFIER_AUTHORITY_PURPOSE,
        identity=identity,
        key_id="key:independent-reviewer",
        algorithm="ed25519",
        key=private_key,
        issued_at="2026-08-30T00:00:00+00:00",
    )
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    assert (
        research_main(
            [
                "verifier-authority",
                "finalize",
                "--paper-dir",
                str(paper),
                "--identity",
                identity,
                "--attestation",
                str(attestation_path),
                "--json",
            ]
        )
        == 0
    )
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["status"] == "finalized"
    assert (paper / "verifier_authority_receipt.json").is_file()

    public_key_path = tmp_path / "reviewer.pub"
    public_key_path.write_bytes(public_key)
    trust_store = tmp_path / "trust.json"
    trust_store.write_text(
        json.dumps(
            {
                "keys": {
                    "key:independent-reviewer": {
                        "identity": identity,
                        "algorithm": "ed25519",
                        "public_key_file": public_key_path.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        research_main(
            [
                "verifier-authority",
                "verify",
                "--paper-dir",
                str(paper),
                "--trust-store",
                str(trust_store),
                "--json",
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["ok"] is True
    assert verified["status"] == "signature_binding_verified"
    assert verified["signature_binding_verified"] is True
    assert verified["submission_ready"] is False
    assert verified["submission_readiness"] == "unknown"

    local_public_key = paper / "self-approved-reviewer.pub"
    local_public_key.write_bytes(public_key)
    local_trust_store = paper / "self-approved-trust.json"
    local_trust_store.write_text(
        json.dumps(
            {
                "keys": {
                    "key:independent-reviewer": {
                        "identity": identity,
                        "algorithm": "ed25519",
                        "public_key_file": local_public_key.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        research_main(
            [
                "verifier-authority",
                "verify",
                "--paper-dir",
                str(paper),
                "--trust-store",
                str(local_trust_store),
                "--json",
            ]
        )
        == 1
    )
    locally_trusted = json.loads(capsys.readouterr().out)
    assert locally_trusted["submission_ready"] is False
    assert "trust_store_inside_research_boundary" in locally_trusted["errors"]
    local_gate = build_scientific_evidence_gate(
        paper,
        target_venue="neurips",
        verifier_trust_store=local_trust_store,
    )
    local_checks = {item["id"]: item for item in local_gate["checks"]}
    assert local_checks["independent_verifier_authority"]["passed"] is False
    assert (
        "trust_store_inside_research_boundary"
        in local_gate["verifier_authority"]["errors"]
    )

    gate = build_scientific_evidence_gate(
        paper,
        target_venue="neurips",
        verifier_trust_store=trust_store,
    )
    checks = {item["id"]: item for item in gate["checks"]}
    assert checks["independent_verifier_authority"]["passed"] is True
    assert "independent_verifier_authority" not in gate["hard_failures"]

    _replace_data_contract(paper, b"another legitimate empirical snapshot\n")
    assert (
        research_main(
            [
                "verifier-authority",
                "verify",
                "--paper-dir",
                str(paper),
                "--trust-store",
                str(trust_store),
                "--json",
            ]
        )
        == 1
    )
    swapped = json.loads(capsys.readouterr().out)
    assert swapped["signature_binding_verified"] is False
    assert swapped["submission_ready"] is False
    assert "report_binding_invalid" in swapped["errors"]


def test_verifier_authority_cli_rejects_producer_alias_as_verifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paper = tmp_path / "paper"
    _write_publication_inputs(
        paper,
        producer_id="agent:executor",
        verifier_id="executor",
    )

    assert (
        research_main(
            [
                "verifier-authority",
                "prepare",
                "--paper-dir",
                str(paper),
                "--identity",
                "agent:executor",
                "--output",
                str(tmp_path / "payload.json"),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert "disjoint" in error["error"]["message"]


def test_verifier_authority_cli_rejects_reproduction_producer_as_verifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paper = tmp_path / "paper"
    _write_publication_inputs(
        paper,
        producer_id="primary-experimenter",
        reproduction_producer_id="agent:reproducer",
        verifier_id="reproducer",
    )

    assert (
        research_main(
            [
                "verifier-authority",
                "prepare",
                "--paper-dir",
                str(paper),
                "--identity",
                "human:reproducer",
                "--output",
                str(tmp_path / "payload.json"),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert "disjoint" in error["error"]["message"]


def test_verifier_authority_cli_includes_failed_reproduction_producer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paper = tmp_path / "paper"
    _write_publication_inputs(
        paper,
        producer_id="primary-experimenter",
        verifier_id="failed-reproducer",
    )
    with (paper / "experiment_registry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "record_id": "reproduce-failed",
                    "study_phase": "exploratory",
                    "independent_reproduction": True,
                    "status": "failed",
                    "producer_id": "executor:failed-reproducer",
                }
            )
            + "\n"
        )

    assert (
        research_main(
            [
                "verifier-authority",
                "prepare",
                "--paper-dir",
                str(paper),
                "--identity",
                "human:failed-reproducer",
                "--output",
                str(tmp_path / "payload.json"),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert "disjoint" in error["error"]["message"]
