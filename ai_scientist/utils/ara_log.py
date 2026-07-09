"""Commit-log view of an ARA.

Two orthogonal histories converge on any given ARA:

1. The manifest revision chain, entirely local to this ARA (from
   :mod:`ai_scientist.utils.ara_manifest_lock`). One row per legal
   post-export edit; rev 0 lives in manifest.lock, rev >= 1 in
   manifest.history.jsonl. This is the "git log for THIS commit"
   view.

2. The provenance ancestry — how this ARA came to be. Starts with
   manifest.provenance and walks parent_ara_root outward. Each step
   verifies content_hash against the parent's own manifest / node so
   the trail is authoritative when hashes match and merely
   informational when paths dead-end.

:func:`ara_log` returns both, structured. The CLI renders; other
consumers can walk the returned graph programmatically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ara_manifest_lock import (
    MANIFEST_HISTORY_NAME,
    MANIFEST_LOCK_NAME,
    verify_manifest_lock,
)


# Max ancestors we'll walk before giving up. Deep chains are legal but
# a runaway loop (broken symlinks, adversarial parent_ara_root) shouldn't
# hang the CLI.
_MAX_ANCESTORS = 32


@dataclass
class RevisionEntry:
    revision: int
    ts: str | None
    base_hash: str | None
    new_hash: str | None
    changed_fields: list[str] = field(default_factory=list)
    reason: str | None = None
    producer: str | None = None


@dataclass
class AncestorEntry:
    """One step up the provenance chain."""
    depth: int
    ara_root: str | None
    node_id: str | None
    content_hash: str | None
    seed_hash: str | None = None
    # Reachable == we could actually read the parent ARA at ara_root.
    # Not reachable is legal (paths break when directories move); the
    # content_hash still identifies the parent for anyone else who has it.
    reachable: bool = False
    # When reachable, this is hash_manifest(parent.manifest.json).
    resolved_manifest_hash: str | None = None
    # When both hashes are set, whether they agree.
    hash_verified: bool | None = None
    detail: str | None = None


@dataclass
class ARALog:
    ara_root: str
    lock: dict[str, Any] | None
    verify: dict[str, Any]
    revisions: list[RevisionEntry] = field(default_factory=list)
    ancestors: list[AncestorEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ara_log(ara_root: str | Path) -> ARALog:
    """Return the full log (revision + ancestry) for one ARA."""
    root = Path(ara_root)
    lock = _load_json(root / MANIFEST_LOCK_NAME)
    verify = verify_manifest_lock(root)

    revisions = _read_revisions(root)
    ancestors = _walk_ancestors(root)

    return ARALog(
        ara_root=str(root),
        lock=lock if isinstance(lock, dict) else None,
        verify=verify,
        revisions=revisions,
        ancestors=ancestors,
    )


# ---------------------------------------------------------------------------
# Revisions
# ---------------------------------------------------------------------------


def _read_revisions(root: Path) -> list[RevisionEntry]:
    p = root / MANIFEST_HISTORY_NAME
    if not p.exists():
        return []
    entries: list[RevisionEntry] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            entries.append(RevisionEntry(
                revision=int(row.get("revision") or 0),
                ts=row.get("ts"),
                base_hash=row.get("base_hash"),
                new_hash=row.get("new_hash"),
                changed_fields=list(row.get("changed_fields") or []),
                reason=row.get("reason"),
                producer=row.get("producer"),
            ))
    except (OSError, json.JSONDecodeError):
        return entries
    return entries


# ---------------------------------------------------------------------------
# Ancestry
# ---------------------------------------------------------------------------


def _walk_ancestors(root: Path) -> list[AncestorEntry]:
    """Walk parent_ara_root outward, verifying content_hash at each step."""
    manifest = _load_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        return []
    ancestors: list[AncestorEntry] = []
    seen_hashes: set[str] = set()
    current_prov = manifest.get("provenance") or {}
    for depth in range(1, _MAX_ANCESTORS + 1):
        if not isinstance(current_prov, dict) or not current_prov:
            break
        parent_root = current_prov.get("parent_ara_root")
        parent_node = current_prov.get("parent_node_id")
        parent_hash = current_prov.get("parent_content_hash")
        seed_hash = current_prov.get("seed_hash")

        # Nothing to trace? stop.
        if not (parent_root or parent_hash or seed_hash):
            break

        entry = AncestorEntry(
            depth=depth,
            ara_root=parent_root,
            node_id=parent_node,
            content_hash=parent_hash,
            seed_hash=seed_hash,
        )
        parent_manifest = None
        if parent_root:
            parent_manifest_path = Path(parent_root) / "manifest.json"
            parent_manifest = _load_json(parent_manifest_path)
            if isinstance(parent_manifest, dict):
                entry.reachable = True
                try:
                    from ai_scientist.protocol import hash_manifest
                    entry.resolved_manifest_hash = hash_manifest(parent_manifest)
                except Exception:  # pragma: no cover - defensive
                    entry.resolved_manifest_hash = None
                # Verify the recorded parent_content_hash against the parent
                # node's own content_hash if we can resolve it.
                node_hash = _lookup_node_hash(Path(parent_root), parent_node)
                if parent_hash and node_hash:
                    entry.hash_verified = (parent_hash == node_hash)
                    entry.detail = (
                        "parent node hash matches provenance"
                        if entry.hash_verified else
                        "parent node hash differs from provenance — path may have drifted"
                    )
            else:
                entry.detail = "parent_ara_root not readable — using content_hash only"
        else:
            entry.detail = "no parent_ara_root path — content-hash-only reference"

        ancestors.append(entry)

        # Advance. Use the parent's provenance for the next hop when we
        # could read it; otherwise stop (we can't crawl what we can't reach).
        if not (isinstance(parent_manifest, dict) and parent_manifest):
            break
        current_prov = parent_manifest.get("provenance") or {}
        # Cycle guard: if we've seen this manifest hash before, bail.
        if entry.resolved_manifest_hash:
            if entry.resolved_manifest_hash in seen_hashes:
                break
            seen_hashes.add(entry.resolved_manifest_hash)
    return ancestors


def _lookup_node_hash(ara_root: Path, node_id: str | None) -> str | None:
    if not node_id:
        return None
    graph = _load_json(ara_root / "exploration_graph.json")
    if not isinstance(graph, dict):
        return None
    for n in (graph.get("nodes") or []):
        if isinstance(n, dict) and str(n.get("id")) == node_id:
            h = n.get("content_hash")
            return h if isinstance(h, str) else None
    return None


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


__all__ = ["ARALog", "AncestorEntry", "RevisionEntry", "ara_log"]
