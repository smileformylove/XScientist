"""Deterministic helpers for bounded, task-relevant scientific memory.

The durable research graph is intentionally complete and can grow without a
prompt-sized bound.  These helpers operate on the *working memory* projected
from that graph.  They are dependency-free so Research VCS, ARA, and the
cross-project learning layer can share the same conservative semantics.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

SEMANTIC_MEMORY_POLICY_VERSION = "1.0"

_STOP_WORDS = {
    "and",
    "for",
    "from",
    "that",
    "the",
    "this",
    "using",
    "with",
}


def semantic_tokens(value: Any) -> set[str]:
    """Return stable lexical features for lightweight relevance scoring.

    CJK characters are retained individually while latin words, numbers, and
    identifiers are normalized as tokens.  This is not presented as an
    embedding model; it is an auditable fallback that also works offline.
    """

    if isinstance(value, (dict, list, tuple, set)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = str(value or "")
    return {
        token
        for token in re.findall(r"[a-z0-9_\-]{2,}|[\u3400-\u9fff]", text.lower())
        if token not in _STOP_WORDS
    }


def semantic_overlap(query: Any, candidate: Any) -> float:
    """Return a deterministic containment-aware overlap in ``[0, 1]``."""

    query_tokens = semantic_tokens(query)
    candidate_tokens = semantic_tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    common = len(query_tokens & candidate_tokens)
    containment = common / len(query_tokens)
    jaccard = common / len(query_tokens | candidate_tokens)
    return round((0.7 * containment) + (0.3 * jaccard), 6)


def estimate_text_tokens(value: Any) -> int:
    """Conservatively estimate prompt tokens without a model dependency.

    ASCII text is estimated at four characters per token.  Non-ASCII code
    points are charged one token each, which is deliberately conservative for
    CJK prose and safe for budget gating.
    """

    if not isinstance(value, str):
        value = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    ascii_count = sum(ord(char) < 128 for char in value)
    non_ascii_count = len(value) - ascii_count
    return max(1, int(math.ceil(ascii_count / 4.0)) + non_ascii_count)


def truncate_text_to_tokens(value: Any, max_tokens: int) -> str:
    """Return a stable, visibly truncated string within ``max_tokens``."""

    text = " ".join(str(value or "").split())
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text) <= max_tokens:
        return text
    suffix = "..."
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(text[:middle] + suffix) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + suffix


def bounded_semantic_view(
    value: Any,
    *,
    query: Any,
    budget_tokens: int,
    max_items: int = 24,
) -> list[dict[str, Any]]:
    """Select a bounded, path-addressed semantic projection of nested data."""

    leaves: list[tuple[str, Any]] = []

    def walk(current: Any, path: str) -> None:
        if len(leaves) >= 256:
            return
        if isinstance(current, dict):
            for key in sorted(current):
                walk(current[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(current, (list, tuple)):
            for index, item in enumerate(current[:32]):
                walk(item, f"{path}[{index}]")
            return
        if current not in (None, ""):
            leaves.append((path or "value", current))

    walk(value, "")
    priority_tokens = {
        "active",
        "blocker",
        "claim",
        "constraint",
        "current",
        "evidence",
        "failure",
        "frontier",
        "hash",
        "id",
        "memory",
        "metric",
        "risk",
        "scope",
        "target",
    }
    ranked = []
    for path, leaf in leaves:
        path_tokens = semantic_tokens(path)
        priority = min(0.3, 0.04 * len(path_tokens & priority_tokens))
        ranked.append(
            (
                semantic_overlap(query, {"path": path, "value": leaf}) + priority,
                path,
                leaf,
            )
        )
    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected: list[dict[str, Any]] = []
    for score, path, leaf in ranked:
        if len(selected) >= max_items:
            break
        item = {
            "path": path,
            "value": truncate_text_to_tokens(leaf, 48),
            "relevance": round(score, 6),
        }
        if estimate_text_tokens([*selected, item]) <= int(budget_tokens):
            selected.append(item)
    return selected


__all__ = [
    "SEMANTIC_MEMORY_POLICY_VERSION",
    "bounded_semantic_view",
    "estimate_text_tokens",
    "semantic_overlap",
    "semantic_tokens",
    "truncate_text_to_tokens",
]
