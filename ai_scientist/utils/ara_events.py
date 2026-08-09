"""Compact semantic event ledger for Agent-Native Research Artifacts.

ARA keeps raw payloads and detailed execution logs, but those files are not a
useful default reading order.  The event ledger is a small, append-only set of
*admitted* research events: node outcomes, claim bindings, and verification
outcomes.  It deliberately does not mirror every tool call or log line.

Rows are idempotent when ``source_key`` is supplied.  This lets an old ARA be
bootstrapped repeatedly without duplicating events while still allowing live
producers to append genuinely new events.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ai_scientist.utils.atomic_io import durable_append_text
from ai_scientist.utils.context_receipts import (
    ContextReceiptError,
    validate_context_receipt,
)

EVENTS_RELPATH = Path("events") / "research_events.jsonl"
_HASH_PREFIX = "sha256:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _stable_event_id(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"evt-{digest}"


def _hash_refs(values: Iterable[Any] | None) -> list[str]:
    refs = {
        str(value)
        for value in (values or [])
        if isinstance(value, str)
        and value.startswith(_HASH_PREFIX)
        and len(value) == len(_HASH_PREFIX) + 64
    }
    return sorted(refs)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _walk_hashes(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value.startswith(_HASH_PREFIX) and len(value) == len(_HASH_PREFIX) + 64:
            yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_hashes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_hashes(nested)


@contextlib.contextmanager
def _ledger_lock(root: Path) -> Iterator[None]:
    """Serialize scan-and-append so idempotence survives concurrent writers."""

    lock_path = root / "events" / ".ledger.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        handle.close()


def iter_events(ara_root: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(ara_root).expanduser().resolve() / EVENTS_RELPATH
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row.setdefault("_line", line_number)
            yield row


def append_event(
    ara_root: str | Path,
    *,
    event_type: str,
    actor: str,
    subject: dict[str, Any] | None = None,
    object_refs: Iterable[str] | None = None,
    relations: Iterable[dict[str, Any]] | None = None,
    attributes: dict[str, Any] | None = None,
    source_key: str | None = None,
    timestamp: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Append one compact event and return ``(row, created)``.

    ``source_key`` should identify the source fact (for example a node id plus
    content hash).  Repeating the call then returns the existing row instead
    of growing the ledger.
    """

    root = Path(ara_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "event_type": str(event_type),
        "source_key": source_key,
        "subject": subject or {},
    }
    event_id = _stable_event_id(
        identity if source_key else {**identity, "nonce": time.time_ns()}
    )
    row = {
        "schema_version": "ara.v1",
        "protocol_kind": "research_event",
        "event_id": event_id,
        "event_type": str(event_type),
        "timestamp": timestamp or _now_iso(),
        "actor": str(actor),
        "subject": subject or {},
        "object_refs": _hash_refs(object_refs),
        "relations": [
            dict(item) for item in (relations or []) if isinstance(item, dict)
        ],
        "attributes": attributes or {},
    }
    if source_key:
        row["source_key"] = source_key

    path = root / EVENTS_RELPATH
    with _ledger_lock(root):
        for existing in iter_events(root):
            if existing.get("event_id") == event_id:
                existing.pop("_line", None)
                return existing, False
        durable_append_text(path, _canonical_json(row) + "\n")
    return row, True


def _metric_summary(node: dict[str, Any]) -> dict[str, Any]:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return {}
    return {
        key: metric.get(key)
        for key in ("name", "value", "maximize", "description")
        if metric.get(key) is not None
    }


def bootstrap_event_ledger(ara_root: str | Path) -> dict[str, Any]:
    """Admit semantic events already present in an ARA.

    The operation is additive and idempotent.  Raw LLM calls and raw logs stay
    in their native stores; only state-changing outcomes and evidence bindings
    are promoted into this ledger.
    """

    root = Path(ara_root).expanduser().resolve()
    created = 0
    existing = 0
    invalid_context_receipts = 0

    graph = _read_json(root / "exploration_graph.json")
    if isinstance(graph, dict):
        parents: dict[str, list[str]] = {}
        for edge in graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            parent = str(edge.get("parent") or "").strip()
            child = str(edge.get("child") or "").strip()
            if parent and child:
                parents.setdefault(child, []).append(parent)
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            content_hash = node.get("content_hash")
            refs = [content_hash] if isinstance(content_hash, str) else []
            refs.extend(node.get("llm_call_refs") or [])
            context_refs = node.get("context_pack_refs") or []
            refs.extend(context_refs)
            relations = [
                {"type": "derived_from", "target_type": "node", "target_id": parent}
                for parent in sorted(set(parents.get(node_id, [])))
            ]
            is_buggy = node.get("is_buggy")
            row, was_created = append_event(
                root,
                event_type="node_completed",
                actor="experiment_agent",
                subject={"type": "node", "id": node_id},
                object_refs=refs,
                relations=relations,
                attributes={
                    "stage": node.get("stage"),
                    "step": node.get("step"),
                    "status": "failed" if is_buggy else "succeeded",
                    "decision": "discard" if is_buggy else "keep_candidate",
                    "metric": _metric_summary(node),
                    "plan_excerpt": node.get("plan_excerpt") or "",
                    "artifacts_dir": node.get("artifacts_dir"),
                    "context_pack_refs": _hash_refs(context_refs),
                },
                source_key=f"graph:node:{node_id}:{content_hash or 'legacy'}",
            )
            del row
            created += int(was_created)
            existing += int(not was_created)

    claims_root = root / "claims"
    if claims_root.is_dir():
        for path in sorted(claims_root.glob("*.json")):
            if path.name in {"_index.json", "coverage.json"}:
                continue
            claim = _read_json(path)
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id") or path.stem)
            node_id = str(claim.get("node_id") or "").strip()
            relation_type = (
                "supported_by" if claim.get("resolved") else "references_unresolved"
            )
            relations = (
                [{"type": relation_type, "target_type": "node", "target_id": node_id}]
                if node_id
                else []
            )
            row, was_created = append_event(
                root,
                event_type=(
                    "claim_bound" if claim.get("resolved") else "claim_unresolved"
                ),
                actor="writing_agent",
                subject={"type": "claim", "id": claim_id},
                object_refs=[
                    *(claim.get("evidence_refs") or []),
                    *(claim.get("context_pack_refs") or []),
                ],
                relations=relations,
                attributes={
                    "assertion": claim.get("context") or "",
                    "resolved": bool(claim.get("resolved")),
                    "source": claim.get("source") or {},
                    "context_pack_refs": _hash_refs(
                        claim.get("context_pack_refs") or []
                    ),
                },
                source_key=f"claim:{claim_id}:{claim.get('claim_hash') or 'legacy'}",
                timestamp=claim.get("recorded_at") or None,
            )
            del row
            created += int(was_created)
            existing += int(not was_created)

    verify_root = root / "verify"
    if verify_root.is_dir():
        for path in sorted(verify_root.glob("*.json")):
            report = _read_json(path)
            if not isinstance(report, dict):
                continue
            report_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            node_id = str(report.get("node_id") or "").strip()
            status = str(report.get("status") or report.get("verdict") or "unknown")
            row, was_created = append_event(
                root,
                event_type="verification_completed",
                actor="reproduce_agent",
                subject=(
                    {"type": "node", "id": node_id}
                    if node_id
                    else {"type": "report", "id": path.stem}
                ),
                object_refs=list(_walk_hashes(report)),
                relations=(
                    [{"type": "verifies", "target_type": "node", "target_id": node_id}]
                    if node_id
                    else []
                ),
                attributes={
                    "status": status,
                    "report_path": str(path.relative_to(root)),
                },
                source_key=f"verify:{path.name}:{report_hash}",
            )
            del row
            created += int(was_created)
            existing += int(not was_created)

    context_receipts = root / "context" / "receipts.jsonl"
    if context_receipts.is_file():
        try:
            receipt_lines = context_receipts.read_text(encoding="utf-8").splitlines()
        except OSError:
            receipt_lines = []
        for line_number, line in enumerate(receipt_lines, start=1):
            try:
                receipt = json.loads(line)
            except json.JSONDecodeError:
                invalid_context_receipts += 1
                continue
            if not isinstance(receipt, dict):
                invalid_context_receipts += 1
                continue
            try:
                receipt = validate_context_receipt(receipt)
            except ContextReceiptError:
                invalid_context_receipts += 1
                continue
            pack_ref = receipt.get("pack_ref")
            if isinstance(pack_ref, dict):
                pack_ref = pack_ref.get("hash")
            output = receipt.get("output") or {}
            receipt_hash = str(receipt.get("receipt_hash") or "")
            row, was_created = append_event(
                root,
                event_type=(
                    "context_consumed"
                    if receipt.get("type") == "consumed"
                    else "context_compiled"
                ),
                actor=str(receipt.get("consumer") or "context_compiler"),
                subject={
                    "type": str(output.get("type") or "context_pack"),
                    "id": str(output.get("id") or pack_ref or line_number),
                },
                object_refs=[
                    pack_ref,
                    receipt.get("pack_hash"),
                    receipt.get("source_closure_hash"),
                    receipt.get("memory_snapshot_hash"),
                ],
                attributes={
                    "intent": receipt.get("intent"),
                    "receipt_hash": receipt_hash or None,
                    "output": output,
                },
                source_key=f"context:{receipt_hash or line_number}",
                timestamp=receipt.get("recorded_at") or None,
            )
            del row
            created += int(was_created)
            existing += int(not was_created)

    return {
        "schema": "ara.event.bootstrap.v1",
        "ledger": str(root / EVENTS_RELPATH),
        "created": created,
        "existing": existing,
        "invalid_context_receipts": invalid_context_receipts,
        "total": sum(1 for _ in iter_events(root)),
    }


__all__ = [
    "EVENTS_RELPATH",
    "append_event",
    "bootstrap_event_ledger",
    "iter_events",
]
