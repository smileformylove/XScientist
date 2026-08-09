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
and can be consumed by any JSON Schema 2020-12 implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import PROTOCOL_VERSION, Kind
from .canonical_json import (
    CANONICAL_JSON_PROFILE,
    CanonicalJSONError,
    canonical_content_hash,
    canonical_json,
    canonical_json_bytes,
)
from .attestation import (
    ATTESTATION_SCHEMA,
    DSSE_PAYLOAD_TYPE,
    IN_TOTO_STATEMENT_TYPE,
    AttestationError,
    build_in_toto_statement,
    sign_attestation,
    sign_dsse_statement,
    verify_attestation,
    verify_authorization_bundle,
    verify_dsse_statement,
)
from .graph import analyze_exploration_graph, graph_with_dag_metadata
from .hashing import (
    LEGACY_NODE_IDENTITY_PROFILE,
    NODE_IDENTITY_PROFILE,
    SUPPORTED_NODE_IDENTITY_PROFILES,
    build_provenance,
    content_hash,
    hash_manifest,
    hash_node_payload,
)
from .llm_trace import active_ara_root, capture_llm_calls, record_llm_call
from .objects import ObjectRef, ObjectStore
from .schemas import available_schemas, load_schema
from .validator import (
    ARA_CONFORMANCE_LEVELS,
    PORTABILITY_PROFILE,
    ValidationReport,
    validate_ara,
    validate_manifest,
)

if TYPE_CHECKING:
    from .research_vcs import (
        AUTONOMOUS_RESEARCH_OBJECT_KINDS,
        BUILTIN_RESEARCH_PROFILES,
        CORE_RESEARCH_OBJECT_KINDS,
        EPISTEMIC_RESEARCH_OBJECT_KINDS,
        RESEARCH_AUTHORITIES,
        RESEARCH_OBJECT_KINDS,
        RESEARCH_OBJECT_IDENTITY_PROFILE,
        RESEARCH_OBJECT_STATES,
        RESEARCH_RELATION_TYPES,
        RESEARCH_SEMANTIC_PROFILE_SCHEMA,
        ResearchObjectError,
        build_research_object,
        research_payload_issues,
        research_profile_status,
        validate_research_payload,
        validate_research_object,
    )


_RESEARCH_VCS_EXPORTS = {
    "AUTONOMOUS_RESEARCH_OBJECT_KINDS",
    "BUILTIN_RESEARCH_PROFILES",
    "CORE_RESEARCH_OBJECT_KINDS",
    "EPISTEMIC_RESEARCH_OBJECT_KINDS",
    "RESEARCH_AUTHORITIES",
    "RESEARCH_OBJECT_IDENTITY_PROFILE",
    "RESEARCH_OBJECT_KINDS",
    "RESEARCH_OBJECT_IDENTITY_PROFILE",
    "RESEARCH_OBJECT_STATES",
    "RESEARCH_RELATION_TYPES",
    "RESEARCH_SEMANTIC_PROFILE_SCHEMA",
    "ResearchObjectError",
    "build_research_object",
    "research_payload_issues",
    "research_profile_status",
    "validate_research_payload",
    "validate_research_object",
}


def __getattr__(name: str):
    """Load Research VCS protocol helpers only when they are used."""

    if name not in _RESEARCH_VCS_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import research_vcs

    value = getattr(research_vcs, name)
    globals()[name] = value
    return value


__all__ = [
    "ATTESTATION_SCHEMA",
    "DSSE_PAYLOAD_TYPE",
    "IN_TOTO_STATEMENT_TYPE",
    "ARA_CONFORMANCE_LEVELS",
    "CANONICAL_JSON_PROFILE",
    "AttestationError",
    "build_in_toto_statement",
    "CanonicalJSONError",
    "PROTOCOL_VERSION",
    "PORTABILITY_PROFILE",
    "AUTONOMOUS_RESEARCH_OBJECT_KINDS",
    "BUILTIN_RESEARCH_PROFILES",
    "CORE_RESEARCH_OBJECT_KINDS",
    "EPISTEMIC_RESEARCH_OBJECT_KINDS",
    "RESEARCH_AUTHORITIES",
    "RESEARCH_OBJECT_KINDS",
    "RESEARCH_OBJECT_STATES",
    "RESEARCH_RELATION_TYPES",
    "RESEARCH_SEMANTIC_PROFILE_SCHEMA",
    "Kind",
    "LEGACY_NODE_IDENTITY_PROFILE",
    "NODE_IDENTITY_PROFILE",
    "ObjectRef",
    "ObjectStore",
    "ResearchObjectError",
    "SUPPORTED_NODE_IDENTITY_PROFILES",
    "ValidationReport",
    "active_ara_root",
    "available_schemas",
    "analyze_exploration_graph",
    "build_provenance",
    "build_research_object",
    "capture_llm_calls",
    "canonical_content_hash",
    "canonical_json",
    "canonical_json_bytes",
    "content_hash",
    "graph_with_dag_metadata",
    "hash_manifest",
    "hash_node_payload",
    "load_schema",
    "record_llm_call",
    "research_payload_issues",
    "research_profile_status",
    "validate_ara",
    "validate_manifest",
    "validate_research_payload",
    "validate_research_object",
    "sign_attestation",
    "sign_dsse_statement",
    "verify_attestation",
    "verify_authorization_bundle",
    "verify_dsse_statement",
]
