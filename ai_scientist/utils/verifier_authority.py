"""External authority receipts for independent scientific verification.

An executor cannot establish its own independence by changing a role label or
setting ``clean_room`` to true.  This module binds a verification report to a
principal whose Ed25519 key is trusted outside the research workspace.  Trust
is supplied explicitly by the caller (or one process-level environment
variable); receipts can never select their own trust root.
"""

from __future__ import annotations

import json
import os
import re
import stat
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from ai_scientist.protocol.attestation import verify_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import schema_validator
from ai_scientist.utils.principal_identity import canonical_principal

from .research_integrity import (
    SCHEMA_VERSION,
    VERIFICATION_REQUIRED_CRITERIA,
    _canonical_hash,
    _verification_report_hash_payload,
)

VERIFIER_AUTHORITY_SCHEMA = "xscientist.verifier-authority.v1"
VERIFIER_AUTHORITY_PURPOSE = "xscientist.independent-verification.v1"
VERIFIER_TRUST_STORE_ENV = "XSCIENTIST_VERIFIER_TRUST_STORE"

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_TRUST_STORE_BYTES = 1 << 20
_MAX_PUBLIC_KEY_BYTES = 1 << 16
_EXTERNAL_IDENTITY_PREFIXES = ("agent:", "service:", "human:")

_PAYLOAD_FIELDS = {
    "schema_version",
    "purpose",
    "verifier_id",
    "verifier_identity",
    "producer_ids",
    "data_manifest_hash",
    "data_snapshot_id",
    "trajectory_binding_hash",
    "research_vcs_frozen_head",
    "research_vcs_lineage_head",
    "research_vcs_attempt_object_ids",
    "research_vcs_binding_object_ids",
    "research_vcs_checkpoint_hashes",
    "failed_record_ids",
    "research_vcs_disposition_object_ids",
    "registration_hash",
    "records_hash",
    "registry_hash",
    "evidence_snapshot_hash",
    "manuscript_hash",
    "report_hash",
    "confirmatory_record_ids",
    "reproduction_record_ids",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "purpose",
    "payload",
    "attestation",
    "receipt_hash",
}
_ATTESTATION_FIELDS = {
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


def _required_text(value: Any, *, label: str, maximum: int = 512) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(ord(char) < 32 for char in text):
        raise ValueError(f"{label} must be a non-empty bounded string")
    return text


def _content_hash(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not _HASH_RE.fullmatch(text):
        raise ValueError(f"{label} must use sha256:<64 lowercase hex>")
    return text


def _identity(value: Any) -> str:
    identity = _required_text(value, label="verifier_identity")
    normalized = unicodedata.normalize("NFKC", identity).casefold()
    # The canonical principal parser deliberately understands additional
    # workflow-role prefixes so aliases cannot manufacture independence.  The
    # externally trusted signer namespace is narrower and fixed by the public
    # receipt schema; do not let the broader alias parser widen this boundary.
    if not normalized.startswith(_EXTERNAL_IDENTITY_PREFIXES):
        raise ValueError("verifier_identity must use agent:, service:, or human:")
    prefix, _separator, _principal = normalized.partition(":")
    principal = canonical_principal(normalized, label="verifier_identity")
    if not principal:
        raise ValueError("verifier_identity principal is empty")
    return f"{prefix}:{principal}"


def _sorted_unique_texts(
    values: Any,
    *,
    label: str,
    required: bool,
    maximum: int,
) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable of strings")
    try:
        rows = list(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of strings") from exc
    if len(rows) > maximum:
        raise ValueError(f"{label} exceeds the maximum of {maximum} entries")
    normalized = sorted(
        {_required_text(item, label=label, maximum=512) for item in rows}
    )
    if required and not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _report_hash_is_valid(report: Mapping[str, Any]) -> bool:
    try:
        expected = _canonical_hash(_verification_report_hash_payload(dict(report)))
    except (TypeError, ValueError):
        return False
    return report.get("report_hash") == expected


def _validate_final_verification_report(report: Mapping[str, Any]) -> None:
    """Validate the final report shape and authority-signing semantics."""

    if not schema_validator("verification_report").is_valid(dict(report)):
        raise ValueError("verification_report_schema_invalid")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("verification_report_schema_version_invalid")
    if (
        report.get("status") != "verified"
        or report.get("claim_promotion_allowed") is not True
        or report.get("clean_room") is not True
        or report.get("required_failures") != []
    ):
        raise ValueError("verification_report_not_final_verified")

    criteria = report.get("criteria")
    criterion_ids = [
        str(item.get("id") or "").strip()
        for item in criteria
        if isinstance(item, Mapping)
    ]
    criteria_by_id = {
        str(item.get("id") or "").strip(): item
        for item in criteria
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    }
    if (
        len(criterion_ids) != len(criteria)
        or len(criterion_ids) != len(set(criterion_ids))
        or not VERIFICATION_REQUIRED_CRITERIA.issubset(criteria_by_id)
        or any(
            criteria_by_id[item].get("passed") is not True
            or criteria_by_id[item].get("required") is not True
            for item in VERIFICATION_REQUIRED_CRITERIA
        )
    ):
        raise ValueError("verification_report_criteria_invalid")
    data_criterion = criteria_by_id.get("data_contract_binding")
    if (
        not isinstance(data_criterion, Mapping)
        or data_criterion.get("passed") is not True
        or data_criterion.get("required") is not True
    ):
        raise ValueError("verification_report_data_binding_invalid")
    trajectory_criterion = criteria_by_id.get("trajectory_binding")
    if (
        not isinstance(trajectory_criterion, Mapping)
        or trajectory_criterion.get("passed") is not True
        or trajectory_criterion.get("required") is not True
        or not _HASH_RE.fullmatch(str(report.get("trajectory_binding_hash") or ""))
        or not re.fullmatch(
            r"[0-9a-f]{40,64}", str(report.get("research_vcs_frozen_head") or "")
        )
        or not re.fullmatch(
            r"[0-9a-f]{40,64}", str(report.get("research_vcs_lineage_head") or "")
        )
        or not report.get("research_vcs_attempt_object_ids")
        or not report.get("research_vcs_binding_object_ids")
        or not report.get("research_vcs_checkpoint_hashes")
    ):
        raise ValueError("verification_report_trajectory_binding_invalid")
    if not report.get("confirmatory_record_ids"):
        raise ValueError("verification_report_confirmatory_records_missing")
    if not report.get("reproduction_record_ids"):
        raise ValueError("verification_report_reproduction_records_missing")
    if not _report_hash_is_valid(report):
        raise ValueError("verification_report.report_hash is invalid")


def build_verifier_authority_payload(
    verification_report: Mapping[str, Any],
    producer_ids: Iterable[str],
    verifier_identity: str,
    *,
    data_manifest_hash: str,
    data_snapshot_id: str,
) -> dict[str, Any]:
    """Build the exact report-and-principal payload an authority signs.

    All scientific content bindings are required.  A partial report can still
    be retained as an audit artifact, but it cannot obtain this top-venue
    independent-verifier authority receipt.
    """

    if not isinstance(verification_report, Mapping):
        raise ValueError("verification_report must be an object")
    report = dict(verification_report)
    _validate_final_verification_report(report)
    report_manifest_hash = _content_hash(
        report.get("data_manifest_hash"), label="verification_report.data_manifest_hash"
    )
    report_snapshot_id = _content_hash(
        report.get("data_snapshot_id"), label="verification_report.data_snapshot_id"
    )
    current_manifest_hash = _content_hash(
        data_manifest_hash, label="data_manifest_hash"
    )
    current_snapshot_id = _content_hash(data_snapshot_id, label="data_snapshot_id")
    if (
        report_manifest_hash != current_manifest_hash
        or report_snapshot_id != current_snapshot_id
    ):
        raise ValueError("verification_report_data_binding_mismatch")
    identity = _identity(verifier_identity)
    verifier_principal = canonical_principal(
        report.get("verifier_id"), label="verification_report.verifier_id"
    )
    raw_producers = _sorted_unique_texts(
        producer_ids,
        label="producer_ids",
        required=True,
        maximum=1024,
    )
    producers = sorted(
        {canonical_principal(item, label="producer_ids") for item in raw_producers}
    )
    identity_principal = canonical_principal(identity, label="verifier_identity")
    if verifier_principal != identity_principal:
        raise ValueError(
            "verifier_identity principal must match verification_report.verifier_id"
        )
    if identity_principal in producers:
        raise ValueError("verifier identity must be disjoint from producers")

    return {
        "schema_version": VERIFIER_AUTHORITY_SCHEMA,
        "purpose": VERIFIER_AUTHORITY_PURPOSE,
        "verifier_id": verifier_principal,
        "verifier_identity": identity,
        "producer_ids": producers,
        "data_manifest_hash": current_manifest_hash,
        "data_snapshot_id": current_snapshot_id,
        "trajectory_binding_hash": _content_hash(
            report.get("trajectory_binding_hash"), label="trajectory_binding_hash"
        ),
        "research_vcs_frozen_head": _required_text(
            report.get("research_vcs_frozen_head"), label="research_vcs_frozen_head"
        ),
        "research_vcs_lineage_head": _required_text(
            report.get("research_vcs_lineage_head"), label="research_vcs_lineage_head"
        ),
        "research_vcs_attempt_object_ids": _sorted_unique_texts(
            report.get("research_vcs_attempt_object_ids") or [],
            label="research_vcs_attempt_object_ids",
            required=True,
            maximum=100_000,
        ),
        "research_vcs_binding_object_ids": _sorted_unique_texts(
            report.get("research_vcs_binding_object_ids") or [],
            label="research_vcs_binding_object_ids",
            required=True,
            maximum=100_000,
        ),
        "research_vcs_checkpoint_hashes": sorted(
            {
                _content_hash(item, label="research_vcs_checkpoint_hashes")
                for item in _sorted_unique_texts(
                    report.get("research_vcs_checkpoint_hashes") or [],
                    label="research_vcs_checkpoint_hashes",
                    required=True,
                    maximum=100_000,
                )
            }
        ),
        "failed_record_ids": _sorted_unique_texts(
            report.get("failed_record_ids") or [],
            label="failed_record_ids",
            required=False,
            maximum=100_000,
        ),
        "research_vcs_disposition_object_ids": _sorted_unique_texts(
            report.get("research_vcs_disposition_object_ids") or [],
            label="research_vcs_disposition_object_ids",
            required=False,
            maximum=100_000,
        ),
        "registration_hash": _content_hash(
            report.get("registration_hash"), label="registration_hash"
        ),
        "records_hash": _content_hash(report.get("records_hash"), label="records_hash"),
        "registry_hash": _content_hash(
            report.get("registry_hash"), label="registry_hash"
        ),
        "evidence_snapshot_hash": _content_hash(
            report.get("evidence_snapshot_hash"), label="evidence_snapshot_hash"
        ),
        "manuscript_hash": _content_hash(
            report.get("manuscript_hash"), label="manuscript_hash"
        ),
        "report_hash": _content_hash(report.get("report_hash"), label="report_hash"),
        "confirmatory_record_ids": _sorted_unique_texts(
            report.get("confirmatory_record_ids") or [],
            label="confirmatory_record_ids",
            required=False,
            maximum=100_000,
        ),
        "reproduction_record_ids": _sorted_unique_texts(
            report.get("reproduction_record_ids") or [],
            label="reproduction_record_ids",
            required=False,
            maximum=100_000,
        ),
    }


def build_verifier_authority_receipt(
    verification_report: Mapping[str, Any],
    producer_ids: Iterable[str],
    verifier_identity: str,
    attestation: Mapping[str, Any],
    *,
    data_manifest_hash: str,
    data_snapshot_id: str,
) -> dict[str, Any]:
    """Wrap a correctly bound attestation without claiming that it is trusted."""

    payload = build_verifier_authority_payload(
        verification_report,
        producer_ids,
        verifier_identity,
        data_manifest_hash=data_manifest_hash,
        data_snapshot_id=data_snapshot_id,
    )
    if not isinstance(attestation, Mapping):
        raise ValueError("attestation must be an object")
    envelope = deepcopy(dict(attestation))
    if set(envelope) != _ATTESTATION_FIELDS:
        raise ValueError("attestation fields are invalid")
    if envelope.get("payload_hash") != canonical_content_hash(payload):
        raise ValueError("attestation payload_hash does not match authority payload")
    if envelope.get("identity") != payload["verifier_identity"]:
        raise ValueError("attestation identity does not match verifier identity")
    if envelope.get("purpose") != VERIFIER_AUTHORITY_PURPOSE:
        raise ValueError("attestation purpose is invalid")
    if envelope.get("algorithm") != "ed25519":
        raise ValueError("independent verifier authority requires Ed25519")
    core = {
        "schema_version": VERIFIER_AUTHORITY_SCHEMA,
        "purpose": VERIFIER_AUTHORITY_PURPOSE,
        "payload": payload,
        "attestation": envelope,
    }
    return {**core, "receipt_hash": canonical_content_hash(core)}


def _read_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    """Read one bounded regular file without following a final symlink."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label}_unreadable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label}_symlink_rejected")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label}_not_regular")
    if metadata.st_size > maximum:
        raise ValueError(f"{label}_too_large")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label}_unreadable") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label}_not_regular")
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError(f"{label}_changed_during_read")
        if opened.st_size > maximum:
            raise ValueError(f"{label}_too_large")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum:
            raise ValueError(f"{label}_too_large")
        return data
    finally:
        os.close(descriptor)


def _normalized_boundary_roots(
    values: Iterable[str | os.PathLike[str]] | None,
) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in values or ():
        try:
            root = Path(value).expanduser().resolve(strict=True)
        except (OSError, TypeError, ValueError):
            continue
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _inside_boundary(path: Path, roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        resolved = path.resolve(strict=False)
    for root in roots:
        if path.is_relative_to(root) or resolved.is_relative_to(root):
            return True
    return False


def _load_ed25519_trust_store(
    path_value: str | os.PathLike[str],
    *,
    forbidden_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    raw_path = Path(os.path.abspath(os.path.expanduser(str(path_value))))
    boundaries = _normalized_boundary_roots(forbidden_roots)
    if _inside_boundary(raw_path, boundaries):
        raise ValueError("trust_store_inside_research_boundary")
    encoded = _read_regular_file(
        raw_path,
        maximum=_MAX_TRUST_STORE_BYTES,
        label="trust_store",
    )
    try:
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("trust_store_json_invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError("trust_store_not_object")
    entries = decoded.get("keys", decoded)
    if not isinstance(entries, dict) or not entries:
        raise ValueError("trust_store_keys_invalid")

    try:
        trust_directory = raw_path.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("trust_store_directory_unreadable") from exc
    loaded: dict[str, dict[str, Any]] = {}
    for raw_key_id, raw_entry in entries.items():
        key_id = _required_text(raw_key_id, label="trust_store.key_id")
        if not isinstance(raw_entry, Mapping):
            raise ValueError("trust_store_entry_invalid")
        entry = dict(raw_entry)
        if entry.get("algorithm") != "ed25519":
            raise ValueError("trust_store_algorithm_not_ed25519")
        trusted_identity = _identity(entry.get("identity"))
        public_key_file = _required_text(
            entry.get("public_key_file"), label="public_key_file", maximum=4096
        )
        relative_key = Path(public_key_file)
        if relative_key.is_absolute():
            raise ValueError("public_key_outside_trust_store")
        lexical_key = trust_directory / relative_key
        try:
            resolved_key = lexical_key.resolve(strict=True)
        except OSError as exc:
            raise ValueError("public_key_unreadable") from exc
        if not resolved_key.is_relative_to(trust_directory):
            raise ValueError("public_key_outside_trust_store")
        if _inside_boundary(resolved_key, boundaries):
            raise ValueError("public_key_inside_research_boundary")
        public_key = _read_regular_file(
            lexical_key,
            maximum=_MAX_PUBLIC_KEY_BYTES,
            label="public_key",
        )
        loaded[key_id] = {
            "identity": trusted_identity,
            "algorithm": "ed25519",
            "public_key": public_key,
            "revoked": entry.get("revoked") is True,
        }
    return loaded


def _blocked(errors: Iterable[str], **metadata: Any) -> dict[str, Any]:
    unique = sorted({str(error) for error in errors if str(error)})
    return {
        "ok": False,
        "status": "blocked",
        "errors": unique or ["verifier_authority_invalid"],
        **metadata,
    }


def _verify_verifier_authority_receipt(
    receipt: Mapping[str, Any],
    verification_report: Mapping[str, Any],
    producer_ids: Iterable[str],
    trust_store_path: str | os.PathLike[str] | None = None,
    *,
    data_manifest_hash: str | None = None,
    data_snapshot_id: str | None = None,
    forbidden_trust_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Verify a receipt against an external Ed25519 authority.

    This function is deliberately fail-closed and non-throwing.  The trust
    store can only come from ``trust_store_path`` or
    :data:`VERIFIER_TRUST_STORE_ENV`; no path embedded in the receipt or found
    in the workspace is consulted.
    """

    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return _blocked(["receipt_not_object"])
    row = dict(receipt)
    if set(row) != _RECEIPT_FIELDS:
        errors.append("receipt_fields_invalid")
    if row.get("schema_version") != VERIFIER_AUTHORITY_SCHEMA:
        errors.append("schema_version_invalid")
    if row.get("purpose") != VERIFIER_AUTHORITY_PURPOSE:
        errors.append("purpose_invalid")

    payload = row.get("payload")
    payload_row = dict(payload) if isinstance(payload, Mapping) else {}
    if not isinstance(payload, Mapping):
        errors.append("payload_not_object")
    elif set(payload_row) != _PAYLOAD_FIELDS:
        errors.append("payload_fields_invalid")

    identity = payload_row.get("verifier_identity")
    try:
        expected_payload = build_verifier_authority_payload(
            verification_report,
            producer_ids,
            str(identity or ""),
            data_manifest_hash=str(data_manifest_hash or ""),
            data_snapshot_id=str(data_snapshot_id or ""),
        )
    except (TypeError, ValueError):
        expected_payload = None
        errors.append("report_binding_invalid")
    if expected_payload is not None and payload_row != expected_payload:
        errors.append("report_binding_mismatch")
    try:
        if not schema_validator("verifier_authority").is_valid(row):
            errors.append("receipt_schema_invalid")
    except (TypeError, ValueError):
        errors.append("receipt_schema_invalid")

    core = {
        "schema_version": row.get("schema_version"),
        "purpose": row.get("purpose"),
        "payload": deepcopy(payload),
        "attestation": deepcopy(row.get("attestation")),
    }
    try:
        if row.get("receipt_hash") != canonical_content_hash(core):
            errors.append("receipt_hash_mismatch")
    except (TypeError, ValueError):
        errors.append("receipt_hash_invalid")

    attestation = row.get("attestation")
    if not isinstance(attestation, Mapping):
        errors.append("attestation_not_object")
        attestation_row: dict[str, Any] = {}
    else:
        attestation_row = dict(attestation)
        if set(attestation_row) != _ATTESTATION_FIELDS:
            errors.append("attestation_fields_invalid")
    if attestation_row.get("algorithm") != "ed25519":
        errors.append("attestation_algorithm_not_ed25519")
    if attestation_row.get("purpose") != VERIFIER_AUTHORITY_PURPOSE:
        errors.append("attestation_purpose_mismatch")
    if attestation_row.get("identity") != identity:
        errors.append("attestation_identity_mismatch")
    try:
        expected_payload_hash = canonical_content_hash(payload_row)
    except (TypeError, ValueError):
        expected_payload_hash = None
        errors.append("payload_hash_invalid")
    if attestation_row.get("payload_hash") != expected_payload_hash:
        errors.append("attestation_payload_hash_mismatch")

    selected_path = trust_store_path
    trust_source = "explicit"
    if selected_path is None:
        selected_path = os.environ.get(VERIFIER_TRUST_STORE_ENV)
        trust_source = "environment"
    if not selected_path:
        errors.append("trust_store_missing")
        trust_store: dict[str, Any] | None = None
    else:
        try:
            trust_store = _load_ed25519_trust_store(
                selected_path,
                forbidden_roots=forbidden_trust_roots,
            )
        except (OSError, TypeError, ValueError) as exc:
            trust_store = None
            errors.append(str(exc) or "trust_store_invalid")

    attestation_result: dict[str, Any] | None = None
    if trust_store is not None and expected_payload is not None:
        try:
            attestation_result = verify_attestation(
                attestation_row,
                expected_payload,
                trust_store=trust_store,
                purpose=VERIFIER_AUTHORITY_PURPOSE,
                identity=str(identity or ""),
            )
        except Exception:  # fail closed across optional crypto backends
            errors.append("attestation_verification_failed")
        else:
            if attestation_result.get("ok") is not True:
                errors.extend(
                    f"attestation:{item}"
                    for item in attestation_result.get("errors") or []
                )

    metadata = {
        "verifier_id": payload_row.get("verifier_id"),
        "verifier_identity": identity,
        "report_hash": payload_row.get("report_hash"),
        "data_manifest_hash": payload_row.get("data_manifest_hash"),
        "data_snapshot_id": payload_row.get("data_snapshot_id"),
        "receipt_hash": row.get("receipt_hash"),
        "trust_source": trust_source,
        "attestation_hash": attestation_row.get("attestation_hash"),
    }
    if errors:
        return _blocked(errors, **metadata)
    return {
        "ok": True,
        "status": "verified",
        "errors": [],
        **metadata,
    }


def verify_verifier_authority_receipt(
    receipt: Mapping[str, Any],
    verification_report: Mapping[str, Any],
    producer_ids: Iterable[str],
    trust_store_path: str | os.PathLike[str] | None = None,
    *,
    data_manifest_hash: str | None = None,
    data_snapshot_id: str | None = None,
    forbidden_trust_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Non-throwing wrapper around the fail-closed authority verifier."""

    try:
        return _verify_verifier_authority_receipt(
            receipt,
            verification_report,
            producer_ids,
            trust_store_path,
            data_manifest_hash=data_manifest_hash,
            data_snapshot_id=data_snapshot_id,
            forbidden_trust_roots=forbidden_trust_roots,
        )
    except Exception:
        # Receipts and report mappings are untrusted inputs.  Do not leak their
        # values through exception text, and never turn parser failure into an
        # authority decision.
        return _blocked(["verifier_authority_verification_failed_closed"])


__all__ = [
    "VERIFIER_AUTHORITY_PURPOSE",
    "VERIFIER_AUTHORITY_SCHEMA",
    "VERIFIER_TRUST_STORE_ENV",
    "build_verifier_authority_payload",
    "build_verifier_authority_receipt",
    "canonical_principal",
    "verify_verifier_authority_receipt",
]
