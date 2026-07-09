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

# Cap the code payload we feed into the hash. Pathologically large nodes (some
# ARA producers embed data blobs inline) would otherwise turn every export
# into a 500-node × N-MB SHA loop. When we truncate we still record the
# original length in the payload so two different long files hash differently.
_MAX_CODE_BYTES = 256 * 1024  # 256 KiB — well above any realistic node code


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


def _prep_code_for_hash(code: str) -> tuple[str, int, bool]:
    """Return ``(payload_code, original_len_bytes, truncated)``.

    Truncation kicks in above ``_MAX_CODE_BYTES``; the returned tuple lets
    callers include the pre-truncation length in the hash payload so two
    long-but-different sources don't collide.
    """
    code = (code or "").strip()
    original_len = len(code.encode("utf-8", errors="replace"))
    if original_len <= _MAX_CODE_BYTES:
        return code, original_len, False
    truncated = code.encode("utf-8", errors="replace")[:_MAX_CODE_BYTES].decode(
        "utf-8", errors="replace"
    )
    return truncated, original_len, True


def hash_node_payload(
    *,
    code: str,
    metric: Any,
    extras: dict[str, Any] | None = None,
    llm_call_hashes: list[str] | None = None,
) -> str:
    """Compute the canonical hash for one exploration node.

    ``extras`` gives producers a hook to bind additional stable inputs (e.g.
    ``{"dataset": "cifar10", "seed": 42}``). Keep the payload lean — adding
    unstable fields to ``extras`` will make the hash drift for cosmetic reasons.

    ``llm_call_hashes`` optionally binds the hashes of the LLM message-blobs
    that produced this node's code (``messages_ref.hash`` from
    ``<ara>/llm/calls.jsonl``). When supplied, two nodes with identical code
    and metric but generated from different prompts hash differently — this
    is what closes the "code same, prompt different" hole. Order-insensitive:
    the list is sorted before hashing, so a caller can pass the raw call
    sequence without worrying about interleaving.

    Nodes that don't opt in to LLM binding are unchanged from earlier
    protocol revisions — hash-compatible with ARAs exported before this
    field existed.

    Large ``code`` inputs (>256 KiB) are truncated for hashing speed, but the
    original length is captured in the payload so distinct long files still
    hash differently.
    """
    prepped, original_len, truncated = _prep_code_for_hash(code or "")
    payload: dict[str, Any] = {
        "code": prepped,
        "code_len_bytes": original_len,
        "metric": _normalise_metric(metric),
    }
    if truncated:
        payload["code_truncated"] = True
    if extras:
        payload["extras"] = dict(extras)
    if llm_call_hashes:
        # sort so ordering of concurrent calls doesn't churn the hash.
        # duplicates and empty strings collapse to nothing — a list of only
        # empty strings must be treated the same as "no LLM calls bound".
        cleaned = sorted({str(h) for h in llm_call_hashes if h})
        if cleaned:
            payload["llm_calls"] = cleaned
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


def hash_manifest(manifest: dict[str, Any]) -> str:
    """Content hash for a manifest payload.

    Excludes fields that describe the *lock itself* — signatures, and any
    previously-computed self-reference — because including them would create
    a chicken-and-egg loop (the manifest's hash would depend on the hash it
    was about to declare). Everything else, including counts / provenance /
    references, feeds the hash so a change to any of them is visible.

    Callers that want to detect tampering only need to re-run this over the
    on-disk manifest.json and compare against manifest.lock.
    """
    # Copy so we don't mutate the caller's dict.
    scrubbed = {k: v for k, v in manifest.items() if k not in _LOCK_EXCLUDED_KEYS}
    return content_hash(scrubbed)


# Fields that must NOT feed hash_manifest — see hash_manifest() docstring.
# Keep this set in sync with manifest.schema.json.
_LOCK_EXCLUDED_KEYS = frozenset({
    "signatures",       # signatures cover the hash, so can't be part of it
    "manifest_hash",    # hypothetical self-reference; not currently written
})


def build_provenance(
    *,
    parent_ara_root: str | None = None,
    parent_node_id: str | None = None,
    parent_content_hash: str | None = None,
    parents: list[dict] | None = None,
) -> dict:
    """Assemble a schema-conformant ``provenance`` block.

    Two shapes are supported and preserved together for backward compat:

      1. Single-parent — the three top-level ``parent_*`` fields.
      2. Multi-parent — a ``parents: []`` array where each entry names a
         ``role`` (``code`` / ``env`` / ``data`` / ...) so consumers can
         reason about *what* was inherited from each ancestor.

    When both are supplied, we ALSO echo the first ``role="code"`` parent
    (or the first entry when no ``code`` role exists) into the single-parent
    fields — this keeps existing consumers that only look at the top-level
    slots working without needing to teach them the array shape.
    """
    payload: dict = {}
    if parent_ara_root is not None:
        payload["parent_ara_root"] = parent_ara_root
    if parent_node_id is not None:
        payload["parent_node_id"] = parent_node_id
    if parent_content_hash is not None:
        payload["parent_content_hash"] = parent_content_hash

    if parents:
        cleaned: list[dict] = []
        for p in parents:
            if not isinstance(p, dict):
                continue
            entry = {k: v for k, v in p.items() if v is not None}
            if entry:
                cleaned.append(entry)
        if cleaned:
            payload["parents"] = cleaned
            # Back-compat: if nothing was supplied for the single-parent
            # slots, elect the first `code` role (or the first entry) so
            # existing single-parent consumers still function.
            if "parent_content_hash" not in payload:
                elected = next(
                    (p for p in cleaned if p.get("role") == "code"),
                    cleaned[0],
                )
                for key in ("parent_ara_root", "parent_node_id", "parent_content_hash"):
                    if key in elected and key not in payload:
                        payload[key] = elected[key]

    return payload
