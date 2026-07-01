"""Content-addressable hashing for ARA payloads.

Rationale
---------
Path names are unstable — two producers can call the same experiment ``n1`` or
``node_root`` without meaning the same thing. A **content hash** derived from
the node's code + its normalised metric gives a stable identifier that lets
downstream tools de-dupe, verify provenance, and compare "the same experiment"
across ARA instances.

We deliberately hash a *minimal* payload: code + a normalised metric summary.
Full stdout would drift on every run (timestamps, memory addresses); the code
+ headline metric is the thing that actually defines a scientific claim.

Hash format: ``sha256:<64 hex chars>``. The prefix is part of the identifier
so consumers can detect algorithm mismatches instead of silently comparing
raw digests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .constants import CONTENT_HASH_ALGO


def _normalise_metric(metric: Any) -> dict[str, Any]:
    """Reduce a metric dict to its stable, hash-worthy subset.

    We keep ``name`` / ``value`` / ``maximize`` — everything else (description,
    metadata about *how* it was measured) can drift without changing the
    scientific claim.
    """
    if not isinstance(metric, dict):
        return {}
    keep = {}
    if "name" in metric and metric["name"] is not None:
        keep["name"] = str(metric["name"])
    if "value" in metric and metric["value"] is not None:
        try:
            keep["value"] = float(metric["value"])
        except (TypeError, ValueError):
            keep["value"] = metric["value"]
    if "maximize" in metric and metric["maximize"] is not None:
        keep["maximize"] = bool(metric["maximize"])
    return keep


def _canonical(payload: Any) -> str:
    """Deterministic JSON serialisation (sorted keys, no whitespace)."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def content_hash(payload: dict[str, Any]) -> str:
    """Content-address a serialisable payload.

    ``payload`` should be a dict of primitive-ish values. Anything unhashable
    is coerced through JSON's ``default=str`` — the hash still stabilises on
    identical inputs across processes.
    """
    canonical = _canonical(payload).encode("utf-8")
    digest = hashlib.new(CONTENT_HASH_ALGO, canonical).hexdigest()
    return f"{CONTENT_HASH_ALGO}:{digest}"


def hash_node_payload(*, code: str, metric: Any, extras: dict[str, Any] | None = None) -> str:
    """Compute the canonical hash for one exploration node.

    ``extras`` gives producers a hook to bind additional stable inputs (e.g.
    ``{"dataset": "cifar10", "seed": 42}``). Keep the payload lean — adding
    unstable fields to ``extras`` will make the hash drift for cosmetic reasons.
    """
    payload: dict[str, Any] = {
        "code": (code or "").strip(),
        "metric": _normalise_metric(metric),
    }
    if extras:
        payload["extras"] = dict(extras)
    return content_hash(payload)


def hash_matches(expected: str, actual: str) -> bool:
    """Compare hashes, ensuring both used the same algorithm prefix."""
    if ":" not in expected or ":" not in actual:
        return False
    exp_algo, exp_digest = expected.split(":", 1)
    act_algo, act_digest = actual.split(":", 1)
    if exp_algo != act_algo:
        return False
    return exp_digest == act_digest
