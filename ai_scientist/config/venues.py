"""Stable target-venue identifiers shared by CLI and workflow layers.

This module intentionally has no runtime or workflow imports so parsers,
validators, and policy code can depend on one venue contract without creating
import cycles.
"""

from __future__ import annotations

DEFAULT_TARGET_VENUE = "neurips"

# Keep the default first for deterministic help text and serialized contracts.
TARGET_VENUES = (
    DEFAULT_TARGET_VENUE,
    "icml",
    "iclr",
    "cvpr",
    "journal",
    "nature",
)
TARGET_VENUE_SET = frozenset(TARGET_VENUES)
CONFERENCE_TARGET_VENUES = frozenset({"neurips", "icml", "iclr", "cvpr"})

__all__ = [
    "CONFERENCE_TARGET_VENUES",
    "DEFAULT_TARGET_VENUE",
    "TARGET_VENUES",
    "TARGET_VENUE_SET",
]
