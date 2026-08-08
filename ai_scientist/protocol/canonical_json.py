"""Cross-language canonical JSON for signatures and protocol commitments.

The legacy ARA hashes intentionally remain stable.  New trust-boundary
artifacts use this stricter profile: object keys are ordered as UTF-16 code
units, non-finite and unsafe numbers are rejected, and IEEE-754 numbers use
the ECMAScript/JCS spelling rules.  A small JavaScript consumer under
``conformance/`` exercises the same profile.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Any

CANONICAL_JSON_PROFILE = "xscientist.canonical-json.v1"
_MAX_SAFE_INTEGER = (1 << 53) - 1


class CanonicalJSONError(ValueError):
    """Raised when a value cannot have one portable JSON identity."""


def _validate_string(value: str) -> str:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise CanonicalJSONError("lone UTF-16 surrogate is not portable JSON")
    return value


def _string(value: str) -> str:
    return json.dumps(
        _validate_string(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _utf16_sort_key(value: str) -> bytes:
    return _validate_string(value).encode("utf-16-be")


def _exponent_from_decimal(value: Decimal) -> str:
    sign, digits, exponent = value.as_tuple()
    raw = "".join(str(digit) for digit in digits).lstrip("0") or "0"
    scientific_exponent = len(raw) + exponent - 1
    mantissa = raw[0]
    tail = raw[1:].rstrip("0")
    if tail:
        mantissa += "." + tail
    prefix = "-" if sign else ""
    exponent_sign = "+" if scientific_exponent >= 0 else "-"
    return f"{prefix}{mantissa}e{exponent_sign}{abs(scientific_exponent)}"


def _number(value: int | float) -> str:
    if isinstance(value, bool):
        raise CanonicalJSONError("booleans are not numbers in canonical JSON")
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise CanonicalJSONError(
                "integer exceeds the interoperable IEEE-754 safe range"
            )
        return str(value)
    if not math.isfinite(value):
        raise CanonicalJSONError("non-finite numbers are not valid JSON")
    if value == 0:
        return "0"
    shortest = repr(value)
    decimal = Decimal(shortest)
    magnitude = abs(value)
    if 1e-6 <= magnitude < 1e21:
        fixed = format(decimal, "f")
        if "." in fixed:
            fixed = fixed.rstrip("0").rstrip(".")
        return fixed
    return _exponent_from_decimal(decimal)


def canonical_json(value: Any) -> str:
    """Return deterministic JSON shared by Python and non-Python consumers."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CanonicalJSONError("canonical JSON object keys must be strings")
        keys = sorted(value, key=_utf16_sort_key)
        return (
            "{"
            + ",".join(_string(key) + ":" + canonical_json(value[key]) for key in keys)
            + "}"
        )
    raise CanonicalJSONError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_content_hash(value: Any) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return "sha256:" + digest


__all__ = [
    "CANONICAL_JSON_PROFILE",
    "CanonicalJSONError",
    "canonical_content_hash",
    "canonical_json",
    "canonical_json_bytes",
]
