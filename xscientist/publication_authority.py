"""User-facing preparation and verification for publication authority receipts."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from ai_scientist.utils.atomic_io import atomic_write_json
from ai_scientist.utils.research_integrity import validate_empirical_data_manifest
from ai_scientist.utils.verifier_authority import (
    build_verifier_authority_payload,
    build_verifier_authority_receipt,
    verify_verifier_authority_receipt,
)

_MAX_REPORT_BYTES = 16 * 1024 * 1024
_MAX_REGISTRY_BYTES = 64 * 1024 * 1024
_MAX_REGISTRY_ROWS = 100_000


class PublicationAuthorityError(ValueError):
    """Raised when publication authority inputs are unsafe or incomplete."""


def _regular_file_bytes(path: Path, *, label: str, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PublicationAuthorityError(
            f"{label} was not found or is unreadable"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PublicationAuthorityError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > maximum:
        raise PublicationAuthorityError(f"{label} exceeds the {maximum}-byte limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationAuthorityError(f"{label} is unreadable") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PublicationAuthorityError(f"{label} must be a regular file")
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise PublicationAuthorityError(f"{label} changed during validation")
        if opened.st_size > maximum:
            raise PublicationAuthorityError(f"{label} exceeds the {maximum}-byte limit")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum:
            raise PublicationAuthorityError(f"{label} exceeds the {maximum}-byte limit")
        return payload
    finally:
        os.close(descriptor)


def _json_mapping(path: Path, *, label: str, maximum: int) -> dict[str, Any]:
    try:
        payload = json.loads(
            _regular_file_bytes(path, label=label, maximum=maximum).decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationAuthorityError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise PublicationAuthorityError(f"{label} must contain one JSON object")
    return payload


def _publication_inputs(
    paper_dir: str | Path,
) -> tuple[Path, dict[str, Any], list[str], dict[str, Any]]:
    root = Path(paper_dir).expanduser().resolve()
    if not root.is_dir():
        raise PublicationAuthorityError("paper directory was not found")
    report = _json_mapping(
        root / "verification_report.json",
        label="verification report",
        maximum=_MAX_REPORT_BYTES,
    )
    registry_path = root / "experiment_registry.jsonl"
    encoded = _regular_file_bytes(
        registry_path,
        label="experiment registry",
        maximum=_MAX_REGISTRY_BYTES,
    )
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise PublicationAuthorityError("experiment registry is not UTF-8") from exc
    producers: set[str] = set()
    row_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row_count += 1
        if row_count > _MAX_REGISTRY_ROWS:
            raise PublicationAuthorityError(
                f"experiment registry exceeds {_MAX_REGISTRY_ROWS} rows"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicationAuthorityError(
                f"experiment registry line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise PublicationAuthorityError(
                f"experiment registry line {line_number} is not an object"
            )
        is_confirmatory = (
            str(row.get("study_phase") or "").strip().lower() == "confirmatory"
        )
        is_reproduction = row.get("independent_reproduction") is True
        if row.get("record_type") != "attempt_disposition" and (
            is_confirmatory or is_reproduction
        ):
            producer = str(row.get("producer_id") or "").strip()
            if not producer:
                raise PublicationAuthorityError(
                    "publication-facing confirmatory/reproduction attempt has no "
                    "producer_id"
                )
            producers.add(producer)
    if not producers:
        raise PublicationAuthorityError(
            "no publication-facing confirmatory/reproduction producer identities "
            "were found"
        )
    try:
        data_attestation = validate_empirical_data_manifest(root)
    except Exception as exc:
        raise PublicationAuthorityError(
            "empirical data manifest validation failed closed"
        ) from exc
    if data_attestation.get("ok") is not True:
        errors = ", ".join(data_attestation.get("errors") or [])
        raise PublicationAuthorityError(
            "empirical data snapshot is not authority-signable: "
            + (errors or "validation failed")
        )
    return root, report, sorted(producers), data_attestation


def _research_boundary_roots(
    paper_root: Path,
    data_attestation: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Find local roots that cannot establish their own external authority."""

    boundaries = [paper_root]
    data_root = str(data_attestation.get("project_root") or "").strip()
    if data_root:
        boundaries.append(Path(data_root).expanduser().resolve())
    for candidate in (paper_root, *paper_root.parents):
        if (candidate / ".git").exists() or (candidate / ".xscientist").is_dir():
            boundaries.append(candidate.resolve())
            break
    return tuple(dict.fromkeys(boundaries))


def prepare_verifier_authority_payload(
    paper_dir: str | Path,
    *,
    verifier_identity: str,
) -> dict[str, Any]:
    """Prepare the exact hash-only payload an external verifier must sign."""

    _root, report, producers, data_attestation = _publication_inputs(paper_dir)
    try:
        return build_verifier_authority_payload(
            report,
            producers,
            verifier_identity,
            data_manifest_hash=str(data_attestation.get("manifest_hash") or ""),
            data_snapshot_id=str(data_attestation.get("snapshot_id") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise PublicationAuthorityError(str(exc)) from exc


def finalize_verifier_authority_receipt(
    paper_dir: str | Path,
    *,
    verifier_identity: str,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an external signature to the current report and producer set."""

    _root, report, producers, data_attestation = _publication_inputs(paper_dir)
    try:
        return build_verifier_authority_receipt(
            report,
            producers,
            verifier_identity,
            attestation,
            data_manifest_hash=str(data_attestation.get("manifest_hash") or ""),
            data_snapshot_id=str(data_attestation.get("snapshot_id") or ""),
        )
    except (TypeError, ValueError) as exc:
        raise PublicationAuthorityError(str(exc)) from exc


def verify_publication_authority(
    paper_dir: str | Path,
    *,
    trust_store: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the external receipt against the live publication artifacts."""

    root, report, producers, data_attestation = _publication_inputs(paper_dir)
    receipt = _json_mapping(
        (
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else root / "verifier_authority_receipt.json"
        ),
        label="verifier authority receipt",
        maximum=_MAX_REPORT_BYTES,
    )
    verified = verify_verifier_authority_receipt(
        receipt,
        report,
        producers,
        trust_store,
        data_manifest_hash=str(data_attestation.get("manifest_hash") or ""),
        data_snapshot_id=str(data_attestation.get("snapshot_id") or ""),
        forbidden_trust_roots=_research_boundary_roots(root, data_attestation),
    )
    signature_binding_verified = verified.get("ok") is True
    return {
        **verified,
        "status": (
            "signature_binding_verified" if signature_binding_verified else "blocked"
        ),
        "signature_binding_verified": signature_binding_verified,
        # This command verifies one cryptographic binding. Only the complete
        # scientific evidence gate can assess submission readiness.
        "submission_ready": False,
        "submission_readiness": (
            "unknown" if signature_binding_verified else "blocked"
        ),
    }


def write_publication_authority_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Atomically write a non-secret payload without following a target symlink."""

    target = Path(path).expanduser().resolve(strict=False)
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        raise PublicationAuthorityError("output path must not be a symlink")
    if target.exists() and not force:
        raise PublicationAuthorityError(
            "output already exists; pass --force to replace it"
        )
    atomic_write_json(
        target, dict(payload), indent=2, ensure_ascii=False, allow_nan=False
    )
    return target


__all__ = [
    "PublicationAuthorityError",
    "finalize_verifier_authority_receipt",
    "prepare_verifier_authority_payload",
    "verify_publication_authority",
    "write_publication_authority_json",
]
