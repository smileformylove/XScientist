"""Canonical actor identity for scientific independence checks."""

from __future__ import annotations

import unicodedata
from typing import Any

# These are declared roles/namespaces, not identity.  Repeated prefixes are
# removed so ``human:reviewer:alice`` and ``executor:alice`` cannot pretend to
# be distinct principals merely by choosing different workflow labels.
IDENTITY_PREFIXES = (
    "agent:",
    "service:",
    "human:",
    "executor:",
    "verifier:",
    "reviewer:",
    "recorder:",
    "model:",
    "tool:",
)


def canonical_principal(value: Any, *, label: str = "principal") -> str:
    """Normalize one actor independently of its declared role namespace."""

    text = str(value or "").strip()
    if not text or len(text) > 512 or any(ord(char) < 32 for char in text):
        raise ValueError(f"{label} must be a non-empty bounded string")
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    while normalized.startswith(IDENTITY_PREFIXES):
        normalized = normalized.split(":", 1)[1].strip()
    if (
        not normalized
        or any(char.isspace() or ord(char) < 32 for char in normalized)
        or len(normalized) > 512
    ):
        raise ValueError(f"{label} is not a valid canonical principal")
    return normalized


__all__ = ["IDENTITY_PREFIXES", "canonical_principal"]
