"""Constants + enums for the ARA protocol.

The version string is *the* source of truth. Bump it whenever the protocol
gains a required field or changes the meaning of an existing one; add a note
to SPEC.md when you do.
"""

from __future__ import annotations

from enum import Enum

# semver-ish; the leading digit is the "breaking" tier the protocol lives on,
# the minor bumps signal additive changes.
PROTOCOL_VERSION = "ara.v1"


class Kind(str, Enum):
    """Enumerated artifact kinds recognised by validators.

    New kinds must land in this enum *and* have a matching JSON Schema. If
    you're just adding an optional field to an existing kind, bump
    PROTOCOL_VERSION's minor tier — do NOT invent a new Kind string.
    """

    MANIFEST = "manifest"
    EXPLORATION_GRAPH = "exploration_graph"
    NODE = "node"
    CLAIM = "claim"
    VERIFY_REPORT = "verify_report"
    REEXEC_BATCH = "reexec_batch"
    METRIC_MARKER = "metric_marker"
    LLM_CALL = "llm_call"
    REFERENCE_MANIFEST = "reference_manifest"
    MANIFEST_LOCK = "manifest_lock"
    MANIFEST_REVISION = "manifest_revision"
    RESEARCH_CHECKPOINT = "research_checkpoint"
    RESEARCH_OBJECT_POINTER = "research_object_pointer"
    RESEARCH_BUNDLE = "research_bundle"
    RESEARCH_ENVIRONMENT = "research_environment"
    RESEARCH_REPOSITORY = "research_repository"
    RESEARCH_OBJECT = "research_object"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(m.value for m in cls)


# Content-hash algorithm. Kept explicit so downstream producers/consumers can
# check they agree on this before comparing hashes.
CONTENT_HASH_ALGO = "sha256"

# Filenames the protocol *requires* to exist at the ARA root. See SPEC.md.
REQUIRED_TOP_LEVEL = ("manifest.json", "exploration_graph.json")

# Reserved subdirectories. Producers may add others, but a validator will
# treat unknown top-level directories as informational, not failing.
RESERVED_SUBDIRS = ("nodes", "claims", "verify", "env")
