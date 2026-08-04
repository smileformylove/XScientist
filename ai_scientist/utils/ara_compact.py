"""Non-destructive compaction of legacy ARA directories.

Compaction never rewrites the source artifact.  It creates a new conformant
ARA, preserves the old lock/history under ``legacy/``, removes redundant
claim/node topology snapshots, and externalises verify log tails into CAS.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_scientist.protocol import content_hash, hash_manifest, validate_ara
from ai_scientist.utils.ara_manifest_lock import write_manifest_lock
from ai_scientist.utils.ara_reexec import persist_reexec_verdict
from ai_scientist.utils.ara_storage import ARAStorageError, storage_report


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _compact_graph(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "exploration_graph.json"
    graph = _load_json(path)
    if not isinstance(graph, dict):
        raise ARAStorageError("compaction requires exploration_graph.json")
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges: list[dict[str, Any]] = [
        dict(edge) for edge in graph.get("edges") or [] if isinstance(edge, dict)
    ]
    for node in nodes:
        node_id = str(node.get("id") or "").strip()
        parent = str(node.get("parent_id") or "").strip()
        if parent and node_id:
            edges.append(
                {"parent": parent, "child": node_id, "stage": node.get("stage")}
            )
        for child_value in node.get("children") or []:
            child = str(child_value or "").strip()
            if node_id and child:
                edges.append(
                    {"parent": node_id, "child": child, "stage": node.get("stage")}
                )
        node.pop("parent_id", None)
        node.pop("parent_id_conflicts", None)
        node.pop("children", None)
    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for edge in edges:
        parent = str(edge.get("parent") or "").strip()
        child = str(edge.get("child") or "").strip()
        if not parent or not child or (parent, child) in seen:
            continue
        seen.add((parent, child))
        normalized.append(
            {"parent": parent, "child": child, "stage": edge.get("stage")}
        )
    graph["nodes"] = nodes
    graph["edges"] = normalized
    graph["topology_encoding"] = "edges"
    graph["counts"] = {
        **(graph.get("counts") or {}),
        "nodes": len(nodes),
        "edges": len(normalized),
        "buggy": sum(1 for node in nodes if node.get("is_buggy")),
    }
    graph.pop("dag", None)
    _write_json(path, graph)
    return {str(node.get("id")): node for node in nodes if node.get("id") is not None}


def _compact_claims(root: Path, node_index: dict[str, dict[str, Any]]) -> int:
    changed = 0
    claims = root / "claims"
    if not claims.is_dir():
        return changed
    for path in claims.glob("*.json"):
        if path.name in {"_index.json", "coverage.json"}:
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        node_id = str(payload.get("node_id") or "")
        node = node_index.get(node_id)
        if "claim_hash" not in payload:
            payload["claim_hash"] = content_hash(
                {
                    "assertion": " ".join(str(payload.get("context") or "").split()),
                    "node_id": node_id,
                    "options": payload.get("options") or {},
                }
            )
        if "evidence_refs" not in payload:
            node_hash = node.get("content_hash") if isinstance(node, dict) else None
            payload["evidence_refs"] = [node_hash] if isinstance(node_hash, str) else []
        if "source" not in payload:
            payload["source"] = {
                "document_hash": None,
                "selector": {"type": "line", "value": payload.get("line")},
            }
        if "node" in payload:
            payload.pop("node", None)
            changed += 1
        _write_json(path, payload)
    return changed


def _compact_verify(root: Path) -> int:
    changed = 0
    verify = root / "verify"
    if not verify.is_dir():
        return changed
    for path in verify.glob("*.json"):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") == "ara.reexec.batch.v1":
            verdicts = []
            for verdict in payload.get("verdicts") or []:
                if not isinstance(verdict, dict):
                    continue
                if "stdout_tail" in verdict or "stderr_tail" in verdict:
                    verdict = persist_reexec_verdict(root, verdict)
                    changed += 1
                verdicts.append(verdict)
            payload["verdicts"] = verdicts
            _write_json(path, payload)
        elif payload.get("schema") == "ara.verify.v1":
            if "stdout_tail" in payload or "stderr_tail" in payload:
                _write_json(path, persist_reexec_verdict(root, payload))
                changed += 1
    return changed


def _relocate_legacy_history(root: Path) -> None:
    legacy = root / "legacy"
    legacy.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.lock", "manifest.history.jsonl"):
        source = root / name
        if source.exists():
            os.replace(source, legacy / name)
    history = root / "history"
    if history.exists():
        os.replace(history, legacy / "history")


def compact_ara(
    source: str | Path,
    destination: str | Path,
    *,
    drop_derived: bool = True,
) -> dict[str, Any]:
    """Create a compacted successor ARA and return a migration report."""

    src = Path(source).expanduser().resolve()
    dest = Path(destination).expanduser().resolve()
    if not (src / "manifest.json").exists():
        raise ARAStorageError(f"source is not an ARA: {src}")
    if dest.exists():
        raise ARAStorageError(f"compaction destination already exists: {dest}")
    try:
        dest.relative_to(src)
    except ValueError:
        pass
    else:
        raise ARAStorageError(
            "compaction destination must not be inside the source ARA"
        )

    before = storage_report(src)
    shutil.copytree(src, dest, symlinks=True)
    node_index = _compact_graph(dest)
    compacted_claims = _compact_claims(dest, node_index)
    compacted_verdicts = _compact_verify(dest)
    if drop_derived:
        for name in ("exploration_graph.html", "exploration_graph.summary.json"):
            (dest / name).unlink(missing_ok=True)

    source_manifest = _load_json(src / "manifest.json")
    if not isinstance(source_manifest, dict):
        raise ARAStorageError("source manifest is unreadable")
    source_hash = hash_manifest(source_manifest)
    _relocate_legacy_history(dest)
    manifest = dict(source_manifest)
    manifest["compaction"] = {
        "schema": "ara.compaction.v1",
        "compacted_at": _now_iso(),
        "source_manifest_hash": source_hash,
        "source_ara": str(src),
        "transformations": [
            "canonical_edges",
            "claim_refs_without_node_snapshots",
            "content_addressed_verify_output",
            *(["drop_derived_graph_views"] if drop_derived else []),
        ],
    }
    provenance = dict(manifest.get("provenance") or {})
    provenance["supersedes_manifest_hash"] = source_hash
    manifest["provenance"] = provenance
    references = dict(manifest.get("references") or {})
    if drop_derived:
        references.pop("exploration_graph_visualization", None)
    references["legacy_history"] = "legacy/"
    manifest["references"] = references
    _write_json(dest / "manifest.json", manifest)
    write_manifest_lock(dest, manifest)

    # The SQLite catalog is a derived view and must describe the compacted
    # successor, not the copied source. Rebuilding also bootstraps any semantic
    # events missing from older ARAs without duplicating existing rows.
    try:
        from ai_scientist.utils.ara_catalog import rebuild_semantic_catalog

        rebuild_semantic_catalog(dest)
    except Exception as exc:
        raise ARAStorageError(f"compacted ARA catalog rebuild failed: {exc}") from exc

    validation = validate_ara(dest)
    if not validation.ok:
        raise ARAStorageError(
            "compacted ARA failed validation: "
            + "; ".join(issue.message for issue in validation.errors)
        )
    after = storage_report(dest)
    return {
        "schema": "ara.compaction.report.v1",
        "source": str(src),
        "destination": str(dest),
        "source_manifest_hash": source_hash,
        "compacted_claim_snapshots": compacted_claims,
        "compacted_verify_records": compacted_verdicts,
        "before_physical_bytes": before["physical_bytes"],
        "after_physical_bytes": after["physical_bytes"],
        "validation": validation.to_dict(),
    }


__all__ = ["compact_ara"]
