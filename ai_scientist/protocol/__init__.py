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
from .graph import analyze_exploration_graph, graph_with_dag_metadata
from .hashing import build_provenance, content_hash, hash_manifest, hash_node_payload
from .llm_trace import active_ara_root, capture_llm_calls, record_llm_call
from .objects import ObjectRef, ObjectStore
from .research_vcs import (
    RESEARCH_AUTHORITIES,
    RESEARCH_OBJECT_KINDS,
    RESEARCH_OBJECT_STATES,
    RESEARCH_RELATION_TYPES,
    ResearchObjectError,
    build_research_object,
    validate_research_object,
)
from .schemas import available_schemas, load_schema
from .validator import ValidationReport, validate_ara, validate_manifest

__all__ = [
    "PROTOCOL_VERSION",
    "RESEARCH_AUTHORITIES",
    "RESEARCH_OBJECT_KINDS",
    "RESEARCH_OBJECT_STATES",
    "RESEARCH_RELATION_TYPES",
    "Kind",
    "ObjectRef",
    "ObjectStore",
    "ResearchObjectError",
    "ValidationReport",
    "active_ara_root",
    "available_schemas",
    "analyze_exploration_graph",
    "build_provenance",
    "build_research_object",
    "capture_llm_calls",
    "content_hash",
    "graph_with_dag_metadata",
    "hash_manifest",
    "hash_node_payload",
    "load_schema",
    "record_llm_call",
    "validate_ara",
    "validate_manifest",
    "validate_research_object",
]
