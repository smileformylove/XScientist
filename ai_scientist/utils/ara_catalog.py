"""Rebuildable semantic catalog over an ARA's durable records.

The SQLite file is a disposable read index, never protocol truth.  It joins
nodes, claims, relations, and admitted research events so consumers can query
by meaning without scanning every stored file.  ``source_fingerprint`` makes
staleness explicit and lets callers rebuild automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.ara_events import (
    EVENTS_RELPATH,
    bootstrap_event_ledger,
    iter_events,
)

CATALOG_RELPATH = Path("catalog") / "semantic.sqlite"
CATALOG_SCHEMA = "ara.semantic.catalog.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _source_paths(root: Path) -> list[Path]:
    paths = [
        root / "manifest.json",
        root / "exploration_graph.json",
        root / EVENTS_RELPATH,
    ]
    for directory in (root / "claims", root / "verify"):
        if directory.is_dir():
            paths.extend(sorted(directory.glob("*.json")))
    return [path for path in paths if path.is_file()]


def catalog_source_fingerprint(ara_root: str | Path) -> str:
    root = Path(ara_root).expanduser().resolve()
    digest = hashlib.sha256()
    for path in _source_paths(root):
        relative = str(path.relative_to(root)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return f"sha256:{digest.hexdigest()}"


def _metric_value(node: dict[str, Any]) -> float | None:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_name(node: dict[str, Any]) -> str | None:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return None
    value = metric.get("name")
    return str(value) if value is not None else None


def _insert_relations(
    connection: sqlite3.Connection,
    rows: Iterable[tuple[str, str, str, str, str, str | None]],
) -> None:
    connection.executemany(
        """
        INSERT OR IGNORE INTO relations
        (source_type, source_id, relation, target_type, target_id, source_event_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        list(rows),
    )


def rebuild_semantic_catalog(
    ara_root: str | Path,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """Build a fresh catalog and atomically replace the previous index."""

    root = Path(ara_root).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"ARA manifest not found: {root / 'manifest.json'}")
    bootstrap_summary = bootstrap_event_ledger(root) if bootstrap else None
    fingerprint = catalog_source_fingerprint(root)
    destination = root / CATALOG_RELPATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    connection = sqlite3.connect(temp)
    try:
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE nodes (
                node_id TEXT PRIMARY KEY,
                content_hash TEXT,
                stage TEXT,
                step INTEGER,
                is_buggy INTEGER,
                metric_value REAL,
                metric_name TEXT,
                plan_excerpt TEXT,
                artifacts_dir TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY,
                node_id TEXT,
                claim_hash TEXT,
                assertion TEXT,
                resolved INTEGER NOT NULL,
                source_file TEXT,
                source_line INTEGER,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT,
                actor TEXT,
                subject_type TEXT,
                subject_id TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE relations (
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                source_event_id TEXT,
                UNIQUE(source_type, source_id, relation, target_type, target_id, source_event_id)
            );
            CREATE TABLE object_links (
                owner_type TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                object_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                UNIQUE(owner_type, owner_id, object_hash, role)
            );
            CREATE INDEX idx_nodes_stage ON nodes(stage, step);
            CREATE INDEX idx_claims_node ON claims(node_id);
            CREATE INDEX idx_events_subject ON events(subject_type, subject_id);
            CREATE INDEX idx_rel_source ON relations(source_type, source_id);
            CREATE INDEX idx_rel_target ON relations(target_type, target_id);
            CREATE INDEX idx_object_owner ON object_links(owner_type, owner_id);
            """)
        metadata = {
            "schema": CATALOG_SCHEMA,
            "source_fingerprint": fingerprint,
            "generated_at": _now_iso(),
            "ara_root": str(root),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )

        graph = _read_json(root / "exploration_graph.json") or {}
        nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
        for node in nodes:
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            connection.execute(
                """
                INSERT OR REPLACE INTO nodes
                (node_id, content_hash, stage, step, is_buggy, metric_value,
                 metric_name, plan_excerpt, artifacts_dir, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    node.get("content_hash"),
                    node.get("stage"),
                    node.get("step"),
                    (
                        None
                        if node.get("is_buggy") is None
                        else int(bool(node.get("is_buggy")))
                    ),
                    _metric_value(node),
                    _metric_name(node),
                    node.get("plan_excerpt") or "",
                    node.get("artifacts_dir"),
                    json.dumps(
                        node, ensure_ascii=False, separators=(",", ":"), default=str
                    ),
                ),
            )
            refs = []
            if isinstance(node.get("content_hash"), str):
                refs.append(("node", node_id, node["content_hash"], "content"))
            refs.extend(
                ("node", node_id, ref, "llm_call")
                for ref in node.get("llm_call_refs") or []
                if isinstance(ref, str)
            )
            refs.extend(
                ("node", node_id, ref, "context_pack")
                for ref in node.get("context_pack_refs") or []
                if isinstance(ref, str)
            )
            connection.executemany(
                "INSERT OR IGNORE INTO object_links VALUES (?, ?, ?, ?)", refs
            )

        graph_relations = []
        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            parent = str(edge.get("parent") or "").strip()
            child = str(edge.get("child") or "").strip()
            if parent and child:
                graph_relations.append(
                    ("node", child, "derived_from", "node", parent, None)
                )
                graph_relations.append(
                    ("node", parent, "has_child", "node", child, None)
                )
        _insert_relations(connection, graph_relations)

        claims_root = root / "claims"
        if claims_root.is_dir():
            for path in sorted(claims_root.glob("*.json")):
                if path.name in {"_index.json", "coverage.json"}:
                    continue
                claim = _read_json(path)
                if not isinstance(claim, dict):
                    continue
                claim_id = str(claim.get("claim_id") or path.stem)
                node_id = str(claim.get("node_id") or "").strip() or None
                connection.execute(
                    """
                    INSERT OR REPLACE INTO claims
                    (claim_id, node_id, claim_hash, assertion, resolved,
                     source_file, source_line, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim_id,
                        node_id,
                        claim.get("claim_hash"),
                        claim.get("context") or "",
                        int(bool(claim.get("resolved"))),
                        claim.get("tex_file"),
                        claim.get("line"),
                        json.dumps(
                            claim,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    ),
                )
                if node_id:
                    relation = (
                        "supported_by"
                        if claim.get("resolved")
                        else "references_unresolved"
                    )
                    _insert_relations(
                        connection,
                        [("claim", claim_id, relation, "node", node_id, None)],
                    )
                connection.executemany(
                    "INSERT OR IGNORE INTO object_links VALUES (?, ?, ?, ?)",
                    [
                        ("claim", claim_id, ref, "evidence")
                        for ref in claim.get("evidence_refs") or []
                        if isinstance(ref, str)
                    ],
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO object_links VALUES (?, ?, ?, ?)",
                    [
                        ("claim", claim_id, ref, "context_pack")
                        for ref in claim.get("context_pack_refs") or []
                        if isinstance(ref, str)
                    ],
                )

        for event in iter_events(root):
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            subject = event.get("subject") or {}
            subject_type = str(subject.get("type") or "")
            subject_id = str(subject.get("id") or "")
            event.pop("_line", None)
            connection.execute(
                """
                INSERT OR REPLACE INTO events
                (event_id, event_type, timestamp, actor, subject_type,
                 subject_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event.get("event_type"),
                    event.get("timestamp"),
                    event.get("actor"),
                    subject_type,
                    subject_id,
                    json.dumps(
                        event, ensure_ascii=False, separators=(",", ":"), default=str
                    ),
                ),
            )
            _insert_relations(
                connection,
                [
                    (
                        subject_type,
                        subject_id,
                        str(relation.get("type") or "related_to"),
                        str(relation.get("target_type") or "unknown"),
                        str(relation.get("target_id") or ""),
                        event_id,
                    )
                    for relation in event.get("relations") or []
                    if isinstance(relation, dict)
                    and subject_type
                    and subject_id
                    and relation.get("target_id")
                ],
            )
            connection.executemany(
                "INSERT OR IGNORE INTO object_links VALUES (?, ?, ?, ?)",
                [
                    ("event", event_id, ref, "event_object")
                    for ref in event.get("object_refs") or []
                    if isinstance(ref, str)
                ],
            )

        try:
            connection.execute(
                "CREATE VIRTUAL TABLE search USING fts5(entity_type, entity_id, text)"
            )
            connection.executemany(
                "INSERT INTO search(entity_type, entity_id, text) VALUES (?, ?, ?)",
                [
                    (
                        "node",
                        str(node.get("id") or ""),
                        str(node.get("plan_excerpt") or ""),
                    )
                    for node in nodes
                    if node.get("id")
                ],
            )
            claim_rows = connection.execute(
                "SELECT claim_id, assertion FROM claims"
            ).fetchall()
            connection.executemany(
                "INSERT INTO search(entity_type, entity_id, text) VALUES (?, ?, ?)",
                [("claim", row[0], row[1] or "") for row in claim_rows],
            )
        except sqlite3.OperationalError:
            pass  # Some minimal SQLite builds omit FTS5; graph queries still work.

        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        connection.close()
    os.replace(temp, destination)

    check = sqlite3.connect(destination)
    try:
        counts = {
            table: int(check.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("nodes", "claims", "events", "relations", "object_links")
        }
    finally:
        check.close()
    return {
        "schema": CATALOG_SCHEMA,
        "catalog_path": str(destination),
        "source_fingerprint": fingerprint,
        "counts": counts,
        "bootstrap": bootstrap_summary,
    }


def catalog_status(ara_root: str | Path) -> dict[str, Any]:
    root = Path(ara_root).expanduser().resolve()
    path = root / CATALOG_RELPATH
    expected = catalog_source_fingerprint(root)
    if not path.is_file():
        return {
            "schema": CATALOG_SCHEMA,
            "catalog_path": str(path),
            "exists": False,
            "fresh": False,
            "expected_source_fingerprint": expected,
        }
    try:
        connection = sqlite3.connect(path)
        try:
            rows = dict(
                connection.execute("SELECT key, value FROM metadata").fetchall()
            )
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in ("nodes", "claims", "events", "relations", "object_links")
            }
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return {
            "schema": CATALOG_SCHEMA,
            "catalog_path": str(path),
            "exists": True,
            "fresh": False,
            "error": str(exc),
            "expected_source_fingerprint": expected,
        }
    return {
        "schema": CATALOG_SCHEMA,
        "catalog_path": str(path),
        "exists": True,
        "fresh": (
            rows.get("source_fingerprint") == expected
            and Path(rows.get("ara_root") or "").expanduser().resolve() == root
        ),
        "source_fingerprint": rows.get("source_fingerprint"),
        "expected_source_fingerprint": expected,
        "generated_at": rows.get("generated_at"),
        "counts": counts,
    }


def ensure_semantic_catalog(ara_root: str | Path) -> dict[str, Any]:
    status = catalog_status(ara_root)
    if status.get("fresh"):
        return status
    return rebuild_semantic_catalog(ara_root)


__all__ = [
    "CATALOG_RELPATH",
    "CATALOG_SCHEMA",
    "catalog_source_fingerprint",
    "catalog_status",
    "ensure_semantic_catalog",
    "rebuild_semantic_catalog",
]
