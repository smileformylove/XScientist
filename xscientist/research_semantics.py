"""Portable semantic identities for research claims and literature evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import re
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.privacy import redact_sensitive_payload

CLAIM_SCOPE_PROFILE = "xscientist.claim-scope.v1"
LITERATURE_RECEIPT_PROFILE = "xscientist.retrieval-receipt.v2"
WEB_ANNOTATION_SELECTOR_PROFILE = "http://www.w3.org/ns/oa#"

_SCALAR_SCOPE_FIELDS = (
    "population",
    "intervention",
    "comparator",
    "outcome",
    "metric",
    "unit",
    "time_window",
    "estimand",
)
_SET_SCOPE_FIELDS = ("datasets", "dataset_slices", "conditions")
_FORBIDDEN_RECEIPT_FIELDS = {
    "api_key",
    "authorization",
    "credentials",
    "password",
    "raw_request",
    "raw_response",
    "refresh_token",
    "request_headers",
    "response_body",
    "secret",
    "token",
}


def _safe_receipt_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_receipt_value(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in _FORBIDDEN_RECEIPT_FIELDS
        }
    if isinstance(value, list):
        return [_safe_receipt_value(item) for item in value]
    return redact_sensitive_payload(deepcopy(value))


def normalize_claim_scope(
    scope: Mapping[str, Any] | None = None,
    *,
    legacy_text: str = "",
) -> dict[str, Any]:
    """Return a deterministic, portable Claim applicability envelope."""

    raw = dict(scope or {})
    result: dict[str, Any] = {"profile": CLAIM_SCOPE_PROFILE}
    for field in _SCALAR_SCOPE_FIELDS:
        value = " ".join(str(raw.get(field) or "").split())
        if value:
            result[field] = value
    for field in _SET_SCOPE_FIELDS:
        value = raw.get(field) or []
        if isinstance(value, str):
            value = [value]
        rows = sorted(
            {" ".join(str(item).split()) for item in value if str(item).strip()}
        )
        if rows:
            result[field] = rows
    normalized_legacy = " ".join(
        str(legacy_text or raw.get("description") or "").split()
    )
    if normalized_legacy:
        result["description"] = normalized_legacy
    if len(result) == 1:
        return {}
    return result


def claim_scope_hash(scope: Mapping[str, Any]) -> str:
    normalized = normalize_claim_scope(scope)
    if not normalized:
        raise ValueError("claim scope cannot be empty")
    return canonical_content_hash(normalized)


def scopes_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Conservatively decide whether two scope envelopes can overlap.

    Missing fields remain compatible.  A conflict is ruled out only when both
    sides explicitly bind the same dimension to disjoint values.
    """

    a = normalize_claim_scope(left)
    b = normalize_claim_scope(right)
    if not a or not b:
        return True
    for field in _SCALAR_SCOPE_FIELDS:
        if a.get(field) and b.get(field) and a[field] != b[field]:
            return False
    for field in _SET_SCOPE_FIELDS:
        a_values = set(a.get(field) or [])
        b_values = set(b.get(field) or [])
        if a_values and b_values and a_values.isdisjoint(b_values):
            return False
    return True


def build_search_receipt_payload(
    *,
    provider: str,
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    retrieved_at: str,
    corpus_version: str = "",
    errors: Sequence[str] = (),
    query_rewrites: Sequence[str] = (),
    filters: Mapping[str, Any] | None = None,
    retrieval_system: Mapping[str, Any] | None = None,
    corpus_snapshot_hash: str = "",
    pagination: Mapping[str, Any] | None = None,
    transform_lineage: Sequence[Mapping[str, Any]] = (),
    complete: bool = True,
) -> dict[str, Any]:
    """Build a replay-oriented retrieval receipt without secrets or raw bodies."""

    normalized_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        row = _safe_receipt_value(candidate)
        row.setdefault("rank", index)
        if not any(
            row.get(key) for key in ("id", "doi", "pmid", "arxiv_id", "url", "title")
        ):
            raise ValueError(
                "each literature candidate requires an id, persistent identifier, URL, or title"
            )
        selected = row.get("selected")
        if selected not in {True, False, None}:
            raise ValueError(
                "literature candidate selected must be true, false, or null"
            )
        row["selection_status"] = (
            "selected"
            if selected is True
            else "rejected" if selected is False else "pending"
        )
        if selected is False and not str(row.get("selection_reason") or "").strip():
            raise ValueError("rejected literature candidate requires selection_reason")
        normalized_candidates.append(row)
    ranks = [int(row.get("rank") or 0) for row in normalized_candidates]
    if any(rank <= 0 for rank in ranks) or len(ranks) != len(set(ranks)):
        raise ValueError("literature candidate ranks must be unique positive integers")
    try:
        parsed_retrieved_at = datetime.fromisoformat(
            str(retrieved_at).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("retrieved_at must be ISO-8601") from exc
    if parsed_retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must include a timezone")
    normalized_query = redact_sensitive_payload(" ".join(str(query).split()))
    normalized_rewrites = list(
        dict.fromkeys(
            redact_sensitive_payload(" ".join(str(value).split()))
            for value in query_rewrites
            if str(value).strip()
        )
    )
    request = {
        "query": normalized_query,
        "query_rewrites": normalized_rewrites,
        "filters": _safe_receipt_value(dict(filters or {})),
    }
    normalized_retrieval_system = _safe_receipt_value(dict(retrieval_system or {}))
    normalized_pagination = _safe_receipt_value(dict(pagination or {}))
    normalized_transforms = [
        _safe_receipt_value(dict(item)) for item in transform_lineage
    ]
    corpus: dict[str, Any] = {}
    if corpus_version.strip():
        corpus["version"] = corpus_version.strip()
    if corpus_snapshot_hash.strip():
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", corpus_snapshot_hash.strip()):
            raise ValueError("corpus snapshot hash must use sha256:<64 hex>")
        corpus["snapshot_hash"] = corpus_snapshot_hash.strip()
    core: dict[str, Any] = {
        "profile": LITERATURE_RECEIPT_PROFILE,
        "provider": redact_sensitive_payload(" ".join(str(provider).split())),
        "query": normalized_query,
        "retrieved_at": str(retrieved_at),
        "candidates": normalized_candidates,
        "request": request,
        "request_hash": canonical_content_hash(request),
        "candidate_set_hash": canonical_content_hash(normalized_candidates),
        "retrieval_system": normalized_retrieval_system,
        "pagination": normalized_pagination,
        "transform_lineage": normalized_transforms,
        "completeness": {
            "complete_candidate_set": bool(complete),
            "candidate_count": len(normalized_candidates),
            "selected_count": sum(
                row["selection_status"] == "selected" for row in normalized_candidates
            ),
            "rejected_count": sum(
                row["selection_status"] == "rejected" for row in normalized_candidates
            ),
            "pending_count": sum(
                row["selection_status"] == "pending" for row in normalized_candidates
            ),
        },
    }
    if corpus:
        core["corpus"] = corpus
    # Retain the v1 convenience field for older readers.
    if corpus_version.strip():
        core["corpus_version"] = corpus_version.strip()
    if errors:
        core["errors"] = [
            redact_sensitive_payload(str(item).strip())
            for item in errors
            if str(item).strip()
        ]
    return {**core, "receipt_hash": canonical_content_hash(core)}


def build_text_quote_selector(
    quote: str,
    *,
    prefix: str = "",
    suffix: str = "",
    start: int | None = None,
    end: int | None = None,
) -> dict[str, Any]:
    """Return deterministic W3C Web Annotation selectors for exact evidence."""

    exact = str(quote or "").strip()
    if not exact:
        raise ValueError("text selector requires an exact quote")
    selectors: list[dict[str, Any]] = [
        {
            "type": "TextQuoteSelector",
            "exact": exact,
            **({"prefix": str(prefix)} if prefix else {}),
            **({"suffix": str(suffix)} if suffix else {}),
        }
    ]
    if start is not None or end is not None:
        if start is None or end is None or start < 0 or end <= start:
            raise ValueError("text position selector requires 0 <= start < end")
        selectors.append(
            {"type": "TextPositionSelector", "start": int(start), "end": int(end)}
        )
    core = {
        "profile": WEB_ANNOTATION_SELECTOR_PROFILE,
        "selectors": selectors,
    }
    return {**core, "selector_hash": canonical_content_hash(core)}


__all__ = [
    "CLAIM_SCOPE_PROFILE",
    "LITERATURE_RECEIPT_PROFILE",
    "build_search_receipt_payload",
    "build_text_quote_selector",
    "claim_scope_hash",
    "normalize_claim_scope",
    "scopes_compatible",
]
