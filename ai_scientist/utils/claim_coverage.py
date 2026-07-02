r"""Claim coverage — how much of the manuscript's claim vocabulary is grounded?

Motivation
----------
`claim_registry.py` writes one JSON file per `\claimref{node_id}` marker it
finds in the manuscript, flagging each as ``resolved: true/false``. Until
now that count only went into a print statement and the manifest's ``counts``
block. This module turns it into a **score dimension**: a normalised
``coverage_score`` in ``[0, 1]`` plus a per-severity breakdown, ready to be
fed into a quality gate, a submission ranker, or the review board's own
scoring vector.

Deliberately kept small — one file, no LLM calls, no writes into review
artefacts. Callers decide where to spend the score.

The point (from *The Second Half of AI for Science*): if we don't let claim
coverage influence *what counts as a good paper*, the writing prompt has no
pressure to actually tie its numbers to exploration nodes, and `\claimref`
degenerates into cargo-cult syntax.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Suggested defaults. Tunable by callers via `evaluate_claim_coverage(...,
# minimum_expected=..., critical_threshold=...)`. See docstrings.
_DEFAULT_MIN_EXPECTED = 3        # papers with <3 quant claims aren't shortcuts we should fail
_DEFAULT_CRITICAL_THRESHOLD = 0.5  # below this we call it "insufficient"


@dataclass
class ClaimCoverageReport:
    """Score-ready summary of a manuscript's `\\claimref` grounding."""

    claim_count: int
    resolved_count: int
    unresolved_node_ids: list[str] = field(default_factory=list)
    coverage_score: float = 0.0     # resolved / max(claim_count, minimum_expected)
    coverage_ratio: float = 0.0     # resolved / max(claim_count, 1)
    severity: str = "unknown"       # "ok" | "sparse" | "unresolved" | "insufficient" | "none"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_claim_index(ara_dir: Path) -> dict[str, Any] | None:
    """Read ``claims/_index.json`` — the pre-built summary from claim_registry."""
    idx = ara_dir / "claims" / "_index.json"
    if not idx.exists():
        return None
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _severity_from(
    claim_count: int,
    resolved_count: int,
    coverage_ratio: float,
    minimum_expected: int,
    critical_threshold: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if claim_count == 0:
        return "none", ["no \\claimref markers found — writeup did not ground any claim"]
    if claim_count < minimum_expected:
        reasons.append(
            f"only {claim_count} claim marker(s); minimum expected is {minimum_expected}"
        )
    if coverage_ratio < critical_threshold:
        reasons.append(
            f"only {coverage_ratio:.0%} of claims resolved to real nodes "
            f"(<{critical_threshold:.0%} critical threshold)"
        )
    if resolved_count == claim_count and claim_count >= minimum_expected:
        return "ok", reasons
    if coverage_ratio < critical_threshold:
        return "insufficient", reasons
    if resolved_count < claim_count:
        return "unresolved", reasons or ["some claim node_ids do not resolve"]
    return "sparse", reasons


def evaluate_claim_coverage(
    ara_dir: str | Path,
    *,
    minimum_expected: int = _DEFAULT_MIN_EXPECTED,
    critical_threshold: float = _DEFAULT_CRITICAL_THRESHOLD,
) -> ClaimCoverageReport:
    r"""Read ``ara_dir/claims/_index.json`` and build a coverage report.

    Two failure modes we watch for:
      1. ``claim_count == 0`` — the writeup ignored the ``\claimref`` convention.
      2. ``resolved / total < critical_threshold`` — the writeup emitted
         fabricated node_ids that don't exist in exploration_graph.

    Both are cheap to detect and useful signal for the review board.
    """
    ara_dir = Path(ara_dir)
    index = _load_claim_index(ara_dir)
    if index is None:
        # No claim scan happened. Different from "scanned and found 0" —
        # we shouldn't punish a producer that skipped the whole subsystem.
        return ClaimCoverageReport(
            claim_count=0,
            resolved_count=0,
            severity="unknown",
            reasons=["no claims/_index.json — claim scan did not run"],
        )

    claim_count = int(index.get("claim_count") or 0)
    resolved_count = int(index.get("resolved_count") or 0)
    unresolved = list(index.get("unresolved_node_ids") or [])
    coverage_ratio = (resolved_count / claim_count) if claim_count > 0 else 0.0
    # Normalise against `minimum_expected` so an under-annotated but 100%
    # resolved paper still scores below one with plenty of grounded claims.
    denominator = max(claim_count, minimum_expected, 1)
    coverage_score = resolved_count / denominator

    severity, reasons = _severity_from(
        claim_count, resolved_count, coverage_ratio, minimum_expected, critical_threshold
    )

    return ClaimCoverageReport(
        claim_count=claim_count,
        resolved_count=resolved_count,
        unresolved_node_ids=unresolved,
        coverage_score=coverage_score,
        coverage_ratio=coverage_ratio,
        severity=severity,
        reasons=reasons,
    )


def write_coverage_into_ara(ara_dir: str | Path, report: ClaimCoverageReport) -> Path:
    """Persist the coverage report under ``<ara_dir>/claims/coverage.json``.

    A separate file (not tacked onto ``_index.json``) so we can bump the
    scoring heuristic later without invalidating the raw claim index.
    """
    ara_dir = Path(ara_dir)
    claims_dir = ara_dir / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    dest = claims_dir / "coverage.json"
    dest.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return dest
