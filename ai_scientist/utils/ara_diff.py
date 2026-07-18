"""Structural diff between two ARA directories.

The point of ARA's content_hash is that we can compare two experiments
by payload, not by directory path. This module walks two ARAs, hashes
what needs hashing, and returns a structured diff so the CLI (or a
programmatic caller) can render it however it likes.

We deliberately return **data**, not formatted text. A ``diff`` verb in
a CLI is thin: it calls one of these functions and pretty-prints. That
lets tests assert against the data shape without parsing terminal
output, and lets other consumers (a future web UI, a scripted audit)
reuse the same engine.

What "diff" covers
------------------
Manifest — hash equality first (short-circuit), then structural diff of:
- counts (nodes / edges / claims / …)
- provenance (parent_content_hash, seed_hash, parents[])
- references.pipeline_artifacts[*].content_hash — per-kind
- references.seed.content_hash
- schema_version

Nodes — as a set of (id, content_hash):
- added / removed / hash_changed
- for hash_changed nodes, we peek at content_hash_inputs on both sides
  and, if we can read the raw code + prompt refs, tell the caller which
  category (code, metric, llm_calls) flipped

Prompts — LLM messages_ref hashes touched by either side (from
llm/calls.jsonl):
- only_in_a, only_in_b, shared_count

We do NOT re-hash blobs on disk here — the manifest / graph already
carry the authoritative hashes. If those disagree with what's on disk,
that's a tampering finding for ``verify_manifest_lock``, not for diff.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public data shapes
# ---------------------------------------------------------------------------


@dataclass
class NodeDelta:
    """One node's story across two ARAs."""
    id: str
    kind: str  # "added" | "removed" | "hash_changed" | "same"
    hash_a: str | None = None
    hash_b: str | None = None
    inputs_a: list[str] = field(default_factory=list)
    inputs_b: list[str] = field(default_factory=list)
    # Which input categories differ (only populated for hash_changed).
    # Empty when we can't tell (e.g. one side is missing input metadata).
    changed_categories: list[str] = field(default_factory=list)


@dataclass
class ManifestDelta:
    hash_equal: bool
    hash_a: str | None
    hash_b: str | None
    field_changes: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ReferencesDelta:
    seed_hash_a: str | None = None
    seed_hash_b: str | None = None
    seed_changed: bool = False
    pipeline_added: list[str] = field(default_factory=list)
    pipeline_removed: list[str] = field(default_factory=list)
    pipeline_hash_changed: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptsDelta:
    total_a: int = 0
    total_b: int = 0
    only_in_a: int = 0
    only_in_b: int = 0
    shared: int = 0


@dataclass
class ARADiff:
    ara_a: str
    ara_b: str
    manifest: ManifestDelta
    references: ReferencesDelta
    nodes_added: list[NodeDelta] = field(default_factory=list)
    nodes_removed: list[NodeDelta] = field(default_factory=list)
    nodes_hash_changed: list[NodeDelta] = field(default_factory=list)
    nodes_unchanged: int = 0
    prompts: PromptsDelta = field(default_factory=PromptsDelta)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def diff_ara(ara_a: str | Path, ara_b: str | Path) -> ARADiff:
    """Compute the full structured diff between two ARA directories."""
    a = Path(ara_a)
    b = Path(ara_b)

    manifest_a = _load_json(a / "manifest.json") or {}
    manifest_b = _load_json(b / "manifest.json") or {}

    return ARADiff(
        ara_a=str(a),
        ara_b=str(b),
        manifest=_diff_manifest(manifest_a, manifest_b),
        references=_diff_references(manifest_a, manifest_b),
        **_diff_nodes(a, b),
        prompts=_diff_prompts(a, b),
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _diff_manifest(a: dict, b: dict) -> ManifestDelta:
    # Use hash_manifest so signatures / manifest_hash don't perturb equality.
    from ai_scientist.protocol import hash_manifest
    ha = hash_manifest(a) if a else None
    hb = hash_manifest(b) if b else None

    changes: dict[str, dict[str, Any]] = {}
    # Structural comparison of a small set of "interesting" top-level slices.
    for key in ("schema_version", "counts", "provenance"):
        if a.get(key) != b.get(key):
            changes[key] = {"a": a.get(key), "b": b.get(key)}
    return ManifestDelta(
        hash_equal=(ha is not None and ha == hb),
        hash_a=ha,
        hash_b=hb,
        field_changes=changes,
    )


# ---------------------------------------------------------------------------
# References (pipeline_artifacts + seed)
# ---------------------------------------------------------------------------


def _diff_references(a: dict, b: dict) -> ReferencesDelta:
    refs_a = (a.get("references") or {})
    refs_b = (b.get("references") or {})

    seed_a = (refs_a.get("seed") or {}).get("content_hash")
    seed_b = (refs_b.get("seed") or {}).get("content_hash")

    pipe_a = _index_pipeline(refs_a.get("pipeline_artifacts") or [])
    pipe_b = _index_pipeline(refs_b.get("pipeline_artifacts") or [])
    added = sorted(pipe_b.keys() - pipe_a.keys())
    removed = sorted(pipe_a.keys() - pipe_b.keys())
    changed: list[dict[str, Any]] = []
    for kind in sorted(pipe_a.keys() & pipe_b.keys()):
        if pipe_a[kind] != pipe_b[kind]:
            changed.append({"kind": kind, "hash_a": pipe_a[kind], "hash_b": pipe_b[kind]})

    return ReferencesDelta(
        seed_hash_a=seed_a,
        seed_hash_b=seed_b,
        seed_changed=(seed_a != seed_b),
        pipeline_added=added,
        pipeline_removed=removed,
        pipeline_hash_changed=changed,
    )


def _index_pipeline(entries: list) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        kind = e.get("kind")
        if isinstance(kind, str):
            out[kind] = e.get("content_hash")
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _diff_nodes(a: Path, b: Path) -> dict[str, Any]:
    ga = _load_json(a / "exploration_graph.json") or {}
    gb = _load_json(b / "exploration_graph.json") or {}
    idx_a = _index_nodes(ga)
    idx_b = _index_nodes(gb)

    added: list[NodeDelta] = []
    removed: list[NodeDelta] = []
    changed: list[NodeDelta] = []
    unchanged = 0
    for nid in sorted(idx_b.keys() - idx_a.keys()):
        added.append(NodeDelta(id=nid, kind="added", hash_b=idx_b[nid]["hash"],
                               inputs_b=idx_b[nid]["inputs"]))
    for nid in sorted(idx_a.keys() - idx_b.keys()):
        removed.append(NodeDelta(id=nid, kind="removed", hash_a=idx_a[nid]["hash"],
                                 inputs_a=idx_a[nid]["inputs"]))
    for nid in sorted(idx_a.keys() & idx_b.keys()):
        ha, hb = idx_a[nid]["hash"], idx_b[nid]["hash"]
        if ha == hb:
            unchanged += 1
            continue
        cats = _which_categories_flipped(
            a / "nodes" / nid,
            b / "nodes" / nid,
            idx_a[nid],
            idx_b[nid],
        )
        changed.append(NodeDelta(
            id=nid, kind="hash_changed",
            hash_a=ha, hash_b=hb,
            inputs_a=idx_a[nid]["inputs"], inputs_b=idx_b[nid]["inputs"],
            changed_categories=cats,
        ))
    return {
        "nodes_added": added,
        "nodes_removed": removed,
        "nodes_hash_changed": changed,
        "nodes_unchanged": unchanged,
    }


def _index_nodes(graph: dict) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for n in (graph.get("nodes") or []):
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if not isinstance(nid, str):
            continue
        out[nid] = {
            "hash": n.get("content_hash"),
            "inputs": list(n.get("content_hash_inputs") or []),
            "llm_call_refs": list(n.get("llm_call_refs") or []),
            "is_seed_node": bool(n.get("is_seed_node")),
        }
    return out


def _which_categories_flipped(
    node_dir_a: Path,
    node_dir_b: Path,
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
) -> list[str]:
    """Best-effort: read the node's on-disk code + metric to tell which
    hashed category actually changed. Falls back to inputs-set delta when
    the underlying files aren't present."""
    cats: list[str] = []
    code_a = _read_text(node_dir_a / "code.py")
    code_b = _read_text(node_dir_b / "code.py")
    if code_a != code_b:
        cats.append("code")

    m_a = _load_json(node_dir_a / "metrics.json") or {}
    m_b = _load_json(node_dir_b / "metrics.json") or {}
    if m_a.get("metric") != m_b.get("metric"):
        cats.append("metric")

    if m_a.get("evaluation_report") != m_b.get("evaluation_report"):
        cats.append("evaluation")

    if set(entry_a["llm_call_refs"]) != set(entry_b["llm_call_refs"]):
        cats.append("llm_calls")

    # Seed-role toggle is bound into content_hash (SPEC §11.1 / c1d51d5) but
    # doesn't live in a file on disk — probe the graph entry directly rather
    # than leaning on content_hash_inputs, which some producers omit.
    if bool(entry_a.get("is_seed_node")) != bool(entry_b.get("is_seed_node")):
        cats.append("seed")

    # If nothing flipped locally but the hashes differ, the difference is in
    # some other input category that fed the hash — surface that explicitly.
    if not cats:
        input_delta = set(entry_a["inputs"]) ^ set(entry_b["inputs"])
        if input_delta:
            cats.extend(sorted(input_delta))
        else:
            cats.append("unknown")
    return cats


# ---------------------------------------------------------------------------
# Prompts (llm/calls.jsonl)
# ---------------------------------------------------------------------------


def _diff_prompts(a: Path, b: Path) -> PromptsDelta:
    refs_a = _messages_refs(a / "llm" / "calls.jsonl")
    refs_b = _messages_refs(b / "llm" / "calls.jsonl")
    return PromptsDelta(
        total_a=len(refs_a),
        total_b=len(refs_b),
        only_in_a=len(refs_a - refs_b),
        only_in_b=len(refs_b - refs_a),
        shared=len(refs_a & refs_b),
    )


def _messages_refs(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    out: set[str] = set()
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            h = ((row.get("messages_ref") or {}).get("hash"))
            if isinstance(h, str):
                out.add(h)
    except (OSError, json.JSONDecodeError):
        return set()
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


__all__ = [
    "ARADiff",
    "ManifestDelta",
    "NodeDelta",
    "PromptsDelta",
    "ReferencesDelta",
    "diff_ara",
]
