"""Signed identity envelopes for research and self-evolution artifacts."""

from __future__ import annotations

import base64
import hashlib
import hmac
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from .canonical_json import (
    CANONICAL_JSON_PROFILE,
    canonical_content_hash,
    canonical_json_bytes,
)

ATTESTATION_SCHEMA = "xscientist.attestation.v1"
SUPPORTED_ALGORITHMS = {"hmac-sha256", "ed25519"}


class AttestationError(ValueError):
    """Raised when an attestation cannot be signed or trusted."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _signature_payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in envelope.items()
        if key not in {"signature", "attestation_hash"}
    }


def _load_ed25519_private_key(value: bytes):
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AttestationError(
            "ed25519 requires the optional 'trust' dependency"
        ) from exc
    try:
        key = serialization.load_pem_private_key(value, password=None)
    except (TypeError, ValueError) as exc:
        raise AttestationError("invalid Ed25519 private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AttestationError("private key is not Ed25519")
    return key


def _load_ed25519_public_key(value: bytes):
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AttestationError(
            "ed25519 requires the optional 'trust' dependency"
        ) from exc
    try:
        key = serialization.load_pem_public_key(value)
    except (TypeError, ValueError) as exc:
        raise AttestationError("invalid Ed25519 public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise AttestationError("public key is not Ed25519")
    return key


def sign_attestation(
    payload: Any,
    *,
    purpose: str,
    identity: str,
    key_id: str,
    algorithm: str = "hmac-sha256",
    key: bytes,
    issued_at: str | None = None,
) -> dict[str, Any]:
    """Sign a payload without embedding the payload or private key."""

    normalized_purpose = str(purpose or "").strip()
    normalized_identity = str(identity or "").strip()
    normalized_key_id = str(key_id or "").strip()
    normalized_algorithm = str(algorithm or "").strip().lower()
    if not normalized_purpose or not normalized_identity or not normalized_key_id:
        raise AttestationError("purpose, identity, and key_id are required")
    if not normalized_identity.startswith(("agent:", "service:", "human:")):
        raise AttestationError("identity must use agent:, service:, or human:")
    if normalized_algorithm not in SUPPORTED_ALGORITHMS:
        raise AttestationError(f"unsupported signature algorithm: {algorithm}")
    if not isinstance(key, bytes) or not key:
        raise AttestationError("non-empty signing key bytes are required")
    timestamp = issued_at or _now_iso()
    if _timestamp(timestamp) is None:
        raise AttestationError("issued_at must be timezone-aware ISO-8601")
    envelope = {
        "schema_version": ATTESTATION_SCHEMA,
        "canonicalization": CANONICAL_JSON_PROFILE,
        "purpose": normalized_purpose,
        "identity": normalized_identity,
        "key_id": normalized_key_id,
        "algorithm": normalized_algorithm,
        "issued_at": timestamp,
        "payload_hash": canonical_content_hash(payload),
    }
    message = canonical_json_bytes(envelope)
    if normalized_algorithm == "hmac-sha256":
        raw_signature = hmac.new(key, message, hashlib.sha256).digest()
    else:
        raw_signature = _load_ed25519_private_key(key).sign(message)
    envelope["signature"] = base64.b64encode(raw_signature).decode("ascii")
    envelope["attestation_hash"] = canonical_content_hash(envelope)
    return envelope


def verify_attestation(
    envelope: Mapping[str, Any],
    payload: Any,
    *,
    trust_store: Mapping[str, Mapping[str, Any]],
    purpose: str | None = None,
    identity: str | None = None,
    max_age_seconds: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify content binding, signer identity, policy, freshness, and signature."""

    errors: list[str] = []
    row = dict(envelope or {})
    expected_fields = {
        "schema_version",
        "canonicalization",
        "purpose",
        "identity",
        "key_id",
        "algorithm",
        "issued_at",
        "payload_hash",
        "signature",
        "attestation_hash",
    }
    if set(row) != expected_fields:
        errors.append("attestation_fields_invalid")
    if row.get("schema_version") != ATTESTATION_SCHEMA:
        errors.append("schema_version_invalid")
    if row.get("canonicalization") != CANONICAL_JSON_PROFILE:
        errors.append("canonicalization_invalid")
    if purpose is not None and row.get("purpose") != purpose:
        errors.append("purpose_mismatch")
    if identity is not None and row.get("identity") != identity:
        errors.append("identity_mismatch")
    try:
        expected_payload_hash = canonical_content_hash(payload)
    except (TypeError, ValueError):
        expected_payload_hash = None
        errors.append("payload_not_canonicalizable")
    if row.get("payload_hash") != expected_payload_hash:
        errors.append("payload_hash_mismatch")
    attestation_hash = row.pop("attestation_hash", None)
    try:
        if attestation_hash != canonical_content_hash(row):
            errors.append("attestation_hash_mismatch")
    except (TypeError, ValueError):
        errors.append("attestation_hash_invalid")
    row["attestation_hash"] = attestation_hash
    issued_at = _timestamp(row.get("issued_at"))
    if issued_at is None:
        errors.append("issued_at_invalid")
    elif max_age_seconds is not None:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (current - issued_at).total_seconds()
        if age < -300 or age > max_age_seconds:
            errors.append("attestation_expired")
    key_id = str(row.get("key_id") or "")
    trusted = trust_store.get(key_id)
    if not isinstance(trusted, Mapping):
        errors.append("key_untrusted")
        trusted = {}
    if trusted.get("revoked") is True:
        errors.append("key_revoked")
    if trusted.get("identity") != row.get("identity"):
        errors.append("trusted_identity_mismatch")
    if trusted.get("algorithm") != row.get("algorithm"):
        errors.append("trusted_algorithm_mismatch")
    signature_text = row.get("signature")
    try:
        signature = base64.b64decode(str(signature_text), validate=True)
    except (ValueError, TypeError):
        signature = b""
        errors.append("signature_encoding_invalid")
    unsigned = _signature_payload(row)
    try:
        message = canonical_json_bytes(unsigned)
        algorithm = row.get("algorithm")
        if algorithm == "hmac-sha256":
            secret = trusted.get("key")
            if isinstance(secret, str):
                secret = secret.encode("utf-8")
            if not isinstance(secret, bytes) or not secret:
                errors.append("trusted_key_missing")
            elif not hmac.compare_digest(
                signature, hmac.new(secret, message, hashlib.sha256).digest()
            ):
                errors.append("signature_invalid")
        elif algorithm == "ed25519":
            public_key = trusted.get("public_key")
            if isinstance(public_key, str):
                public_key = public_key.encode("utf-8")
            if not isinstance(public_key, bytes) or not public_key:
                errors.append("trusted_key_missing")
            else:
                try:
                    _load_ed25519_public_key(public_key).verify(signature, message)
                except Exception:  # cryptography uses InvalidSignature
                    errors.append("signature_invalid")
        else:
            errors.append("algorithm_invalid")
    except (AttestationError, TypeError, ValueError):
        errors.append("signature_verification_failed")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "identity": row.get("identity"),
        "purpose": row.get("purpose"),
        "payload_hash": row.get("payload_hash"),
        "attestation_hash": attestation_hash,
    }


def verify_authorization_bundle(
    bundle: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    promotion: Mapping[str, Any],
    trust_store: Mapping[str, Mapping[str, Any]],
    minimum_human_approvers: int = 1,
) -> dict[str, Any]:
    """Require independent signed evidence before a deployment adapter mutates files."""

    artifacts = {
        "candidate": candidate,
        "shadow_gate": promotion.get("gate_report") or {},
        "canary": promotion.get("canary_report") or {},
        "promotion": promotion,
    }
    required_purposes = {
        "candidate_artifact": "candidate",
        "independent_benchmark": "shadow_gate",
        "canary_execution": "canary",
    }
    checks: list[dict[str, Any]] = []
    envelopes = list(bundle.get("attestations") or [])
    for purpose, artifact_name in required_purposes.items():
        matching = [item for item in envelopes if item.get("purpose") == purpose]
        if len(matching) != 1:
            checks.append({"ok": False, "purpose": purpose, "errors": ["missing"]})
            continue
        checks.append(
            verify_attestation(
                matching[0],
                artifacts[artifact_name],
                trust_store=trust_store,
                purpose=purpose,
            )
        )
    approval_payload = {
        "candidate_hash": candidate.get("candidate_hash"),
        "promotion_hash": promotion.get("promotion_hash"),
        "decision": promotion.get("decision"),
    }
    human_checks = [
        verify_attestation(
            item,
            approval_payload,
            trust_store=trust_store,
            purpose="production_approval",
        )
        for item in envelopes
        if item.get("purpose") == "production_approval"
    ]
    valid_humans = {
        str(item.get("identity"))
        for item in human_checks
        if item.get("ok") and str(item.get("identity")).startswith("human:")
    }
    errors = [
        f"{item.get('purpose')}:{','.join(item.get('errors') or [])}"
        for item in checks
        if not item.get("ok")
    ]
    if len(valid_humans) < minimum_human_approvers:
        errors.append(
            f"production_approval:humans={len(valid_humans)}/{minimum_human_approvers}"
        )
    producer = str(candidate.get("proposed_by") or "")
    candidate_signers = {
        str(item.get("identity"))
        for item in checks
        if item.get("purpose") == "candidate_artifact" and item.get("ok")
    }
    evaluator_identities = {
        str(item.get("identity"))
        for item in checks
        if item.get("purpose") == "independent_benchmark" and item.get("ok")
    }
    canary_identities = {
        str(item.get("identity"))
        for item in checks
        if item.get("purpose") == "canary_execution" and item.get("ok")
    }
    if producer and candidate_signers != {producer}:
        errors.append("candidate_signer_mismatch")
    if producer in evaluator_identities or producer in valid_humans:
        errors.append("authority_separation_failed")
    canary_executor = str(
        (promotion.get("canary_report") or {}).get("executed_by") or ""
    )
    if canary_executor and canary_identities != {canary_executor}:
        errors.append("canary_signer_mismatch")
    recorded_approvers = {
        str(item) for item in (promotion.get("approver_ids") or []) if str(item)
    }
    if recorded_approvers and not recorded_approvers.issubset(valid_humans):
        errors.append("recorded_approver_signature_missing")
    if (
        promotion.get("decision") != "approved"
        or promotion.get("production_promotion_allowed") is not True
    ):
        errors.append("promotion_not_approved")
    return {
        "ok": not errors,
        "errors": errors,
        "checks": checks,
        "human_approvers": sorted(valid_humans),
        "candidate_hash": candidate.get("candidate_hash"),
        "promotion_hash": promotion.get("promotion_hash"),
    }


__all__ = [
    "ATTESTATION_SCHEMA",
    "SUPPORTED_ALGORITHMS",
    "AttestationError",
    "sign_attestation",
    "verify_attestation",
    "verify_authorization_bundle",
]
