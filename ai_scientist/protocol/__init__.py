"""ARA (Agent-Native Research Artifact) protocol.

Public surface:

    from ai_scientist.protocol import (
        PROTOCOL_VERSION,     # canonical version string
        Kind,                 # enumerated artifact kinds
        content_hash,         # content-addressable id for a node payload
        validate_ara,         # conformance validator returning a ValidationReport
        load_schema,          # fetch a JSON Schema by kind
    )

The protocol lives *next to* the exporter so producers and consumers can share
one source of truth. Every schema is a plain JSON file under `schemas/`,
loadable at runtime without a Python dependency.
"""

from __future__ import annotations

from .constants import PROTOCOL_VERSION, Kind
from .hashing import content_hash, hash_node_payload
from .schemas import load_schema, available_schemas
from .validator import ValidationReport, validate_ara, validate_manifest

__all__ = [
    "PROTOCOL_VERSION",
    "Kind",
    "ValidationReport",
    "available_schemas",
    "content_hash",
    "hash_node_payload",
    "load_schema",
    "validate_ara",
    "validate_manifest",
]
