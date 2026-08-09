"""Integrity helpers for ARA ContextPack compilation/consumption receipts."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ai_scientist.protocol.hashing import content_hash

CONTEXT_RECEIPT_SCHEMA = "ara.context.receipt.v1"


class ContextReceiptError(ValueError):
    """Raised when a ContextPack receipt cannot prove its own identity."""


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _pack_hash(receipt: Mapping[str, Any]) -> str:
    raw = receipt.get("pack_ref")
    if isinstance(raw, Mapping):
        raw = raw.get("hash")
    return str(raw or "")


def seal_context_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached receipt carrying a timestamp-independent identity."""

    sealed = dict(receipt)
    identity = {
        key: value
        for key, value in sealed.items()
        if key not in {"recorded_at", "receipt_hash"}
    }
    sealed["receipt_hash"] = content_hash(identity)
    return sealed


def validate_context_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate receipt structure and its content-addressed bindings."""

    if not isinstance(receipt, Mapping):
        raise ContextReceiptError("ContextPack receipt must be an object")
    detached = dict(receipt)
    if detached.get("schema") != CONTEXT_RECEIPT_SCHEMA:
        raise ContextReceiptError("ContextPack receipt schema is invalid")
    receipt_type = str(detached.get("type") or "")
    if receipt_type not in {"compiled", "consumed"}:
        raise ContextReceiptError("ContextPack receipt type is invalid")
    if not _is_sha256(_pack_hash(detached)):
        raise ContextReceiptError("ContextPack receipt pack_ref is invalid")
    for key in ("pack_hash", "source_closure_hash"):
        if not _is_sha256(detached.get(key)):
            raise ContextReceiptError(f"ContextPack receipt {key} is invalid")
    memory_hash = detached.get("memory_snapshot_hash")
    if memory_hash is not None and not _is_sha256(memory_hash):
        raise ContextReceiptError("ContextPack receipt memory_snapshot_hash is invalid")
    if receipt_type == "consumed":
        output = detached.get("output")
        if not isinstance(output, Mapping) or not str(output.get("id") or ""):
            raise ContextReceiptError("consumed ContextPack receipt has no output")
        if not _is_sha256(memory_hash):
            raise ContextReceiptError(
                "consumed ContextPack receipt has no memory snapshot"
            )
    expected = seal_context_receipt(detached)["receipt_hash"]
    if detached.get("receipt_hash") != expected:
        raise ContextReceiptError("ContextPack receipt hash mismatch")
    return detached


__all__ = [
    "CONTEXT_RECEIPT_SCHEMA",
    "ContextReceiptError",
    "seal_context_receipt",
    "validate_context_receipt",
]
