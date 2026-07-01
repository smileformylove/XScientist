"""ARA A/B experiment package.

See `harness.py` for the runner. Kept out of the main pipeline import graph
so importing ``ai_scientist`` doesn't pull the harness in.
"""

from __future__ import annotations

from .harness import (
    ABReport,
    ArmResult,
    build_verdict,
    compute_hash_overlap,
    run_ab_real,
    run_ab_stub,
    write_report,
)

__all__ = [
    "ABReport",
    "ArmResult",
    "build_verdict",
    "compute_hash_overlap",
    "run_ab_real",
    "run_ab_stub",
    "write_report",
]
