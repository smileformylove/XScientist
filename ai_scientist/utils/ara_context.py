"""Task-oriented context compilation for stored ARA information.

Storage and consumption are intentionally separate.  This module converts the
complete archive into a bounded ContextPack for one concrete consumer:

* ``continue``  — experiment agent choosing the next branch;
* ``write``     — writing agent grounding manuscript claims;
* ``audit``     — reviewer checking support, contradictions, and omissions;
* ``reproduce`` — executor reconstructing a node's minimal run closure.

Hard graph dependencies are selected deterministically.  The token budget only
trims optional lists; it never removes the target, execution hook, or evidence
references required by the requested operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol import ObjectStore, content_hash
from ai_scientist.protocol.llm_trace import active_ara_root
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.ara_catalog import CATALOG_RELPATH, ensure_semantic_catalog
from ai_scientist.utils.atomic_io import durable_append_text
from ai_scientist.utils.context_receipts import seal_context_receipt

CONTEXT_INTENTS = ("continue", "write", "audit", "reproduce", "decide")
CONTEXT_RECEIPTS_RELPATH = Path("context") / "receipts.jsonl"


class ARAContextError(RuntimeError):
    """Raised when a requested context target cannot be resolved."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _json_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    )


def _sha256_file(path: Path) -> str | None:
    try:
        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    except OSError:
        return None


def _plain_metric(node: dict[str, Any]) -> dict[str, Any]:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return {}
    return {
        key: metric.get(key)
        for key in ("name", "value", "maximize", "description")
        if metric.get(key) is not None
    }


def _numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("objective", "final_value", "best_value", "value", "mean"):
            if key in value:
                parsed = _numeric_metric(value.get(key))
                if parsed is not None:
                    return parsed
        for nested in value.values():
            parsed = _numeric_metric(nested)
            if parsed is not None:
                return parsed
    if isinstance(value, list):
        for nested in value:
            parsed = _numeric_metric(nested)
            if parsed is not None:
                return parsed
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _node_view(node: dict[str, Any], root: Path) -> dict[str, Any]:
    node_id = str(node.get("id") or node.get("node_id") or "")
    node_dir = root / "nodes" / node_id
    return {
        "node_id": node_id,
        "content_hash": node.get("content_hash"),
        "stage": node.get("stage"),
        "step": node.get("step"),
        "status": "failed" if node.get("is_buggy") else "succeeded",
        "metric": _plain_metric(node),
        "plan_excerpt": str(node.get("plan_excerpt") or node.get("plan") or "")[:400],
        "code_path": (
            str((node_dir / "code.py").relative_to(root))
            if (node_dir / "code.py").is_file()
            else None
        ),
        "run_hook": (
            str((node_dir / "run.sh").relative_to(root))
            if (node_dir / "run.sh").is_file()
            else None
        ),
        "context_pack_refs": list(node.get("context_pack_refs") or []),
    }


def _load_catalog(
    root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[tuple[str, str, str, str, str]],
]:
    ensure_semantic_catalog(root)
    path = root / CATALOG_RELPATH
    connection = sqlite3.connect(path)
    try:
        node_rows = connection.execute(
            "SELECT node_id, payload_json FROM nodes"
        ).fetchall()
        claim_rows = connection.execute(
            "SELECT claim_id, payload_json FROM claims"
        ).fetchall()
        relation_rows = connection.execute(
            "SELECT source_type, source_id, relation, target_type, target_id FROM relations"
        ).fetchall()
    finally:
        connection.close()
    nodes = {row[0]: json.loads(row[1]) for row in node_rows}
    claims = {row[0]: json.loads(row[1]) for row in claim_rows}
    return nodes, claims, [tuple(str(value) for value in row) for row in relation_rows]


def _parent_maps(
    relations: Iterable[tuple[str, str, str, str, str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}
    for source_type, source_id, relation, target_type, target_id in relations:
        if source_type != "node" or target_type != "node":
            continue
        if relation == "derived_from":
            parents.setdefault(source_id, []).append(target_id)
            children.setdefault(target_id, []).append(source_id)
        elif relation == "has_child":
            children.setdefault(source_id, []).append(target_id)
            parents.setdefault(target_id, []).append(source_id)
    return (
        {key: sorted(set(values)) for key, values in parents.items()},
        {key: sorted(set(values)) for key, values in children.items()},
    )


def _ancestry(node_id: str, parents: dict[str, list[str]]) -> list[str]:
    ordered: list[str] = []
    queue = list(parents.get(node_id, []))
    seen = {node_id}
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        queue.extend(parents.get(current, []))
    return ordered


def _best_node_id(nodes: dict[str, dict[str, Any]]) -> str | None:
    candidates = [node for node in nodes.values() if not node.get("is_buggy")]
    if not candidates:
        candidates = list(nodes.values())
    if not candidates:
        return None

    def score(node: dict[str, Any]) -> tuple[int, float, int]:
        metric = _plain_metric(node)
        parsed = _numeric_metric(metric.get("value"))
        value = parsed if parsed is not None else 0.0
        has_metric = int(parsed is not None)
        if metric.get("maximize") is False:
            value = -value
        return has_metric, value, int(node.get("step") or 0)

    return str(max(candidates, key=score).get("id") or "") or None


def _trim_optional_lists(
    pack: dict[str, Any],
    *,
    budget_tokens: int,
) -> None:
    """Apply a deterministic budget while preserving hard closure fields."""

    budget_chars = max(1000, int(budget_tokens) * 4)
    mandatory_keys = {
        "schema_version",
        "protocol_kind",
        "intent",
        "target",
        "generated_at",
        "must_read",
        "execution",
        "verification_rules",
        "source_refs",
        "source_closure_hash",
        "memory_refs",
        "memory_snapshot_hash",
        "complete",
        "blockers",
    }
    mandatory_size = sum(
        _json_size({key: pack[key]}) for key in mandatory_keys if key in pack
    )
    remaining = max(0, budget_chars - mandatory_size)
    omitted = dict(pack.get("omitted") or {})
    for key in (
        "decisive_evidence",
        "failed_attempts",
        "do_not_repeat",
        "open_questions",
        "constraints",
        "related_claims",
    ):
        values = pack.get(key)
        if not isinstance(values, list):
            continue
        kept = []
        for item in values:
            size = _json_size(item)
            if size <= remaining:
                kept.append(item)
                remaining -= size
            else:
                omitted[key] = int(omitted.get(key) or 0) + 1
        pack[key] = kept
    pack["omitted"] = omitted
    pack["budget"] = {
        "requested_tokens": int(budget_tokens),
        "estimated_chars": _json_size(pack),
        "hard_closure_preserved": True,
    }


def _manifest_constraints(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = _read_json(root / "manifest.json") or {}
    constraints: list[dict[str, Any]] = []
    blockers: list[str] = []
    for omission in manifest.get("omissions") or []:
        if not isinstance(omission, dict):
            continue
        item = {
            "kind": omission.get("kind"),
            "reason": omission.get("reason"),
            "affects": omission.get("affects") or [],
        }
        constraints.append(item)
    return constraints, blockers


def _finalize_pack(
    pack: dict[str, Any], source_refs: Iterable[str], budget_tokens: int
) -> dict[str, Any]:
    refs = sorted({str(ref) for ref in source_refs if ref})
    memory_refs = sorted(
        {
            str(ref)
            for ref in pack.get("memory_refs") or []
            if isinstance(ref, str) and ref
        }
    )
    pack["source_refs"] = refs
    pack["source_closure_hash"] = content_hash({"source_refs": refs})
    pack["memory_refs"] = memory_refs
    pack["memory_snapshot_hash"] = content_hash({"memory_refs": memory_refs})
    pack["complete"] = not bool(pack.get("blockers"))
    _trim_optional_lists(pack, budget_tokens=budget_tokens)
    identity = {
        key: value
        for key, value in pack.items()
        if key not in {"generated_at", "pack_hash", "persisted_ref"}
    }
    pack["pack_hash"] = content_hash(identity)
    validate_context_pack(pack)
    return pack


def validate_context_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Validate schema and every internal ContextPack identity."""

    if not isinstance(pack, dict):
        raise ARAContextError("ContextPack must be an object")
    try:
        validate_json(pack, load_schema("context_pack"))
    except ValidationError as exc:
        raise ARAContextError(f"ContextPack schema is invalid: {exc.message}") from exc
    refs = sorted({str(ref) for ref in pack.get("source_refs") or []})
    if pack.get("source_refs") != refs:
        raise ARAContextError("ContextPack source refs are not canonical and unique")
    if pack.get("source_closure_hash") != content_hash({"source_refs": refs}):
        raise ARAContextError("ContextPack source closure hash mismatch")
    memory_refs = sorted({str(ref) for ref in pack.get("memory_refs") or []})
    if pack.get("memory_refs") != memory_refs:
        raise ARAContextError("ContextPack memory refs are not canonical and unique")
    if pack.get("memory_snapshot_hash") != content_hash({"memory_refs": memory_refs}):
        raise ARAContextError("ContextPack memory snapshot hash mismatch")
    identity = {
        key: value
        for key, value in pack.items()
        if key not in {"generated_at", "pack_hash", "persisted_ref"}
    }
    if pack.get("pack_hash") != content_hash(identity):
        raise ARAContextError("ContextPack identity hash mismatch")
    if bool(pack.get("complete")) == bool(pack.get("blockers")):
        raise ARAContextError("ContextPack completeness does not match blockers")
    return pack


def compile_context_pack(
    ara_root: str | Path,
    *,
    intent: str,
    node_id: str | None = None,
    claim_id: str | None = None,
    budget_tokens: int = 12000,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a bounded ContextPack from the semantic catalog."""

    intent = str(intent or "").lower()
    if intent not in CONTEXT_INTENTS:
        raise ARAContextError(f"unknown context intent: {intent}")
    root = Path(ara_root).expanduser().resolve()
    nodes, claims, relations = _load_catalog(root)
    parents, children = _parent_maps(relations)

    selected_claim = claims.get(str(claim_id)) if claim_id else None
    if claim_id and selected_claim is None:
        raise ARAContextError(f"claim not found: {claim_id}")
    if node_id is None and selected_claim is not None:
        node_id = str(selected_claim.get("node_id") or "") or None
    if node_id is None and intent in {"continue", "reproduce", "decide"}:
        node_id = _best_node_id(nodes)
    selected_node = nodes.get(str(node_id)) if node_id else None
    if node_id and selected_node is None:
        raise ARAContextError(f"node not found: {node_id}")

    constraints, blockers = _manifest_constraints(root)
    source_refs: set[str] = set()
    for path in (root / "manifest.json", root / "exploration_graph.json"):
        ref = _sha256_file(path)
        if ref:
            source_refs.add(ref)

    pack: dict[str, Any] = {
        "schema_version": "ara.v1",
        "protocol_kind": "context_pack",
        "intent": intent,
        "target": {"node_id": node_id, "claim_id": claim_id},
        "generated_at": _now_iso(),
        "consumer": {
            "continue": "experiment_agent",
            "write": "writing_agent",
            "audit": "reviewer_agent",
            "reproduce": "reproduce_agent",
            "decide": "decision_agent",
        }[intent],
        "must_read": [],
        "decisive_evidence": [],
        "failed_attempts": [],
        "do_not_repeat": [],
        "open_questions": [],
        "constraints": constraints,
        "related_claims": [],
        "blockers": blockers,
        "omitted": {},
        "memory_refs": [],
    }

    if selected_node is not None:
        selected_view = _node_view(selected_node, root)
        pack["must_read"].append(selected_view)
        if selected_node.get("content_hash"):
            source_refs.add(str(selected_node["content_hash"]))
        pack["memory_refs"].extend(selected_node.get("context_pack_refs") or [])
        for ancestor_id in _ancestry(str(node_id), parents):
            ancestor = nodes.get(ancestor_id)
            if ancestor is None:
                continue
            pack["must_read"].append(_node_view(ancestor, root))
            if ancestor.get("content_hash"):
                source_refs.add(str(ancestor["content_hash"]))
            pack["memory_refs"].extend(ancestor.get("context_pack_refs") or [])

    if intent == "continue":
        related_failed: list[dict[str, Any]] = []
        preferred_ids = set(children.get(str(node_id), [])) if node_id else set()
        if node_id:
            for parent in parents.get(str(node_id), []):
                preferred_ids.update(children.get(parent, []))
        ordered = sorted(
            (node for node in nodes.values() if node.get("is_buggy")),
            key=lambda node: (
                str(node.get("id") or "") not in preferred_ids,
                str(node.get("stage") or "")
                != str((selected_node or {}).get("stage") or ""),
                -(int(node.get("step") or 0)),
            ),
        )
        for node in ordered:
            view = _node_view(node, root)
            related_failed.append(view)
            pack["do_not_repeat"].append(
                {
                    "node_id": view["node_id"],
                    "attempt": view["plan_excerpt"],
                    "reason": "recorded as failed/buggy",
                }
            )
        pack["failed_attempts"] = related_failed
        unresolved = [claim for claim in claims.values() if not claim.get("resolved")]
        source_refs.update(
            str(claim.get("claim_hash") or content_hash(claim)) for claim in unresolved
        )
        pack["open_questions"].extend(
            {
                "claim_id": claim.get("claim_id"),
                "question": str(claim.get("context") or "")[:400],
                "reason": "claim has no resolved evidence node",
            }
            for claim in unresolved
        )
        if selected_node is not None and _plain_metric(selected_node):
            pack["decisive_evidence"].append(
                {"node_id": node_id, "metric": _plain_metric(selected_node)}
            )

    if intent in {"write", "audit", "decide"}:
        selected_claims = (
            [selected_claim] if selected_claim is not None else list(claims.values())
        )
        selected_claims = [
            claim for claim in selected_claims if isinstance(claim, dict)
        ]
        for claim in sorted(
            selected_claims, key=lambda item: str(item.get("claim_id") or "")
        ):
            claim_view = {
                "claim_id": claim.get("claim_id"),
                "assertion": claim.get("context") or "",
                "resolved": bool(claim.get("resolved")),
                "node_id": claim.get("node_id"),
                "evidence_refs": list(claim.get("evidence_refs") or []),
                "source": claim.get("source") or {},
            }
            pack["related_claims"].append(claim_view)
            source_refs.add(str(claim.get("claim_hash") or content_hash(claim)))
            source_refs.update(str(ref) for ref in claim_view["evidence_refs"] if ref)
            pack["memory_refs"].extend(claim.get("context_pack_refs") or [])
            if claim_view["resolved"]:
                pack["decisive_evidence"].append(claim_view)
            else:
                pack["open_questions"].append(
                    {
                        **claim_view,
                        "reason": "must remain a hypothesis until evidence is bound",
                    }
                )
        pack["constraints"].append(
            {
                "kind": "claim_grounding",
                "rule": "Do not state an unresolved or unanchored claim as an established result.",
            }
        )
        if intent in {"audit", "decide"}:
            pack["failed_attempts"] = [
                _node_view(node, root)
                for node in nodes.values()
                if node.get("is_buggy")
            ]
            verify_root = root / "verify"
            verify_reports = []
            if verify_root.is_dir():
                for path in sorted(verify_root.glob("*.json")):
                    payload = _read_json(path)
                    if isinstance(payload, dict):
                        verify_reports.append(
                            {
                                "path": str(path.relative_to(root)),
                                "node_id": payload.get("node_id"),
                                "status": payload.get("status")
                                or payload.get("verdict"),
                            }
                        )
                        ref = _sha256_file(path)
                        if ref:
                            source_refs.add(ref)
            pack["verification_reports"] = verify_reports

    if intent == "decide":
        decision_payload = dict(decision or {})
        pack["decision"] = decision_payload
        if decision_payload:
            source_refs.add(content_hash({"decision_inputs": decision_payload}))
        if selected_node is not None:
            selected_view = _node_view(selected_node, root)
            if selected_view not in pack["decisive_evidence"]:
                pack["decisive_evidence"].append(selected_view)
        pack["constraints"].append(
            {
                "kind": "decision_memory",
                "rule": "Consider retained failures, contradictions, omissions, and prior context before selecting an action.",
            }
        )
        if not pack["decisive_evidence"]:
            pack["blockers"].append(
                "decision has no decisive evidence in its hard closure"
            )

    if intent == "reproduce":
        if selected_node is None:
            pack["blockers"].append("no reproducible node could be resolved")
            pack["execution"] = {}
            pack["verification_rules"] = []
        else:
            node_dir = root / "nodes" / str(node_id)
            env_payload = _read_json(node_dir / "env.json") or {}
            execution = {
                "node_id": node_id,
                "code_path": str((node_dir / "code.py").relative_to(root)),
                "run_hook": str((node_dir / "run.sh").relative_to(root)),
                "node_env_path": str((node_dir / "env.json").relative_to(root)),
                "expected_cwd": env_payload.get("expected_cwd"),
                "python_version": env_payload.get("python_version"),
                "environment_paths": (
                    sorted(
                        str(path.relative_to(root))
                        for path in (root / "env").glob("*")
                        if path.is_file()
                    )
                    if (root / "env").is_dir()
                    else []
                ),
            }
            for required in (
                node_dir / "code.py",
                node_dir / "run.sh",
                node_dir / "env.json",
            ):
                if not required.is_file():
                    pack["blockers"].append(
                        f"missing required reproduction file: {required.relative_to(root)}"
                    )
                ref = _sha256_file(required)
                if ref:
                    source_refs.add(ref)
            expected_metric = _plain_metric(selected_node)
            pack["execution"] = execution
            pack["verification_rules"] = [
                {
                    "type": "metric_tolerance",
                    "expected": expected_metric,
                    "default_absolute_tolerance": 1e-3,
                },
                {"type": "process_exit", "expected_returncode": 0},
            ]
            pack["decisive_evidence"].append(
                {
                    "node_id": node_id,
                    "metric": expected_metric,
                    "content_hash": selected_node.get("content_hash"),
                }
            )

    return _finalize_pack(pack, source_refs, budget_tokens)


def _live_base(intent: str, consumer: str, target: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ara.v1",
        "protocol_kind": "context_pack",
        "intent": intent,
        "target": target,
        "generated_at": _now_iso(),
        "consumer": consumer,
        "must_read": [],
        "decisive_evidence": [],
        "failed_attempts": [],
        "do_not_repeat": [],
        "open_questions": [],
        "constraints": [],
        "related_claims": [],
        "blockers": [],
        "omitted": {},
        "memory_refs": [],
        "live": True,
    }


def compile_live_continue_context(
    nodes: Iterable[dict[str, Any]],
    *,
    target_node_id: str | None,
    stage: str | None,
    budget_tokens: int = 3000,
) -> dict[str, Any]:
    """Compile the experiment context before the final ARA graph exists."""

    plain_nodes = [dict(node) for node in nodes if isinstance(node, dict)]
    by_id = {str(node.get("id") or ""): node for node in plain_nodes if node.get("id")}
    pack = _live_base(
        "continue", "experiment_agent", {"node_id": target_node_id, "stage": stage}
    )
    target = by_id.get(str(target_node_id)) if target_node_id else None
    if target is not None:
        pack["must_read"].append(
            {
                "node_id": target_node_id,
                "status": "failed" if target.get("is_buggy") else "succeeded",
                "metric": _plain_metric(target),
                "plan_excerpt": str(target.get("plan") or "")[:400],
            }
        )
    good = [node for node in plain_nodes if not node.get("is_buggy")]
    if good:

        def live_score(node: dict[str, Any]) -> tuple[int, float]:
            metric = _plain_metric(node)
            parsed = _numeric_metric(metric.get("value"))
            value = parsed if parsed is not None else 0.0
            if metric.get("maximize") is False:
                value = -value
            return int(parsed is not None), value

        best = max(
            good,
            key=live_score,
        )
        pack["decisive_evidence"].append(
            {
                "node_id": best.get("id"),
                "metric": _plain_metric(best),
                "plan": str(best.get("plan") or "")[:300],
            }
        )
    for node in plain_nodes:
        if not node.get("is_buggy"):
            continue
        item = {
            "node_id": node.get("id"),
            "attempt": str(node.get("plan") or "")[:300],
            "reason": str(node.get("exc_type") or "recorded as failed/buggy"),
        }
        pack["failed_attempts"].append(item)
        pack["do_not_repeat"].append(item)
    pack["constraints"].append(
        {
            "kind": "stage",
            "rule": f"Respect the current experimental stage: {stage or 'unspecified'}",
        }
    )
    source_refs = [
        str(node.get("content_hash"))
        for node in plain_nodes
        if isinstance(node.get("content_hash"), str)
    ]
    if not source_refs:
        source_refs = [content_hash({"nodes": plain_nodes})]
    pack["memory_refs"] = sorted(
        {
            str(ref)
            for node in plain_nodes
            for ref in node.get("context_pack_refs") or []
            if isinstance(ref, str)
        }
    )
    return _finalize_pack(pack, source_refs, budget_tokens)


def compile_live_write_context(
    summaries: dict[str, Any],
    manuscript_state: dict[str, Any],
    *,
    budget_tokens: int = 4000,
) -> dict[str, Any]:
    pack = _live_base("write", "writing_agent", {"manuscript": "current"})

    evidence_keys = (
        "metric",
        "score",
        "accuracy",
        "loss",
        "result",
        "finding",
        "dataset",
        "best",
        "claim",
        "limitation",
    )

    def extract(value: Any, path: str, out: list[dict[str, Any]]) -> None:
        if len(out) >= 40:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                extract(nested, f"{path}.{key}" if path else str(key), out)
            return
        if isinstance(value, list):
            if value and all(not isinstance(item, (dict, list)) for item in value):
                if any(token in path.lower() for token in evidence_keys):
                    out.append({"path": path, "value": value[:20]})
                return
            for index, nested in enumerate(value[:20]):
                extract(nested, f"{path}[{index}]", out)
            return
        if value not in (None, "") and any(
            token in path.lower() for token in evidence_keys
        ):
            out.append({"path": path, "value": str(value)[:500]})

    for name, summary in summaries.items():
        if not summary:
            continue
        extracted: list[dict[str, Any]] = []
        extract(summary, name, extracted)
        pack["decisive_evidence"].extend(extracted)
        pack["must_read"].append(
            {
                "summary": name,
                "source_hash": content_hash({"summary": name, "payload": summary}),
                "extracted_evidence_count": len(extracted),
            }
        )
    claim_bindings = manuscript_state.get("claim_bindings") or []
    pack["related_claims"] = (
        list(claim_bindings) if isinstance(claim_bindings, list) else []
    )
    pack["must_read"].append(
        {
            "manuscript_state": {
                key: manuscript_state.get(key)
                for key in (
                    "claim_bindings",
                    "claim_figure_bindings",
                    "unresolved_questions",
                    "limitations",
                )
                if key in manuscript_state
            }
        }
    )
    if not pack["decisive_evidence"]:
        pack["blockers"].append(
            "no experiment summaries are available for result claims"
        )
    pack["constraints"].extend(
        [
            {
                "kind": "claim_grounding",
                "rule": "Only state measured results present in the evidence summaries.",
            },
            {
                "kind": "uncertainty",
                "rule": "Label unsupported statements as hypotheses or limitations.",
            },
        ]
    )
    refs = [
        content_hash({"summary": name, "payload": payload})
        for name, payload in sorted(summaries.items())
    ]
    refs.append(content_hash({"manuscript_state": manuscript_state}))
    return _finalize_pack(pack, refs, budget_tokens)


def compile_live_audit_context(
    *,
    evidence_refs: Iterable[str] | None,
    review_plan: dict[str, Any],
    budget_tokens: int = 3000,
) -> dict[str, Any]:
    pack = _live_base("audit", "reviewer_agent", {"paper": "current"})
    refs = sorted({str(ref) for ref in (evidence_refs or []) if ref})
    pack["must_read"] = [{"evidence_ref": ref} for ref in refs]
    pack["constraints"].append(
        {
            "kind": "review_scope",
            "rule": "Separate claims supported by supplied evidence from claims that require additional verification.",
        }
    )
    if not refs:
        pack["open_questions"].append(
            {"reason": "review received no explicit evidence references"}
        )
    source_refs = refs or [content_hash({"review_plan": review_plan})]
    return _finalize_pack(pack, source_refs, budget_tokens)


def render_context_pack_for_prompt(pack: dict[str, Any]) -> str:
    """Stable, compact prompt block. Raw logs are deliberately absent."""

    visible = {
        key: pack.get(key)
        for key in (
            "intent",
            "target",
            "must_read",
            "decisive_evidence",
            "failed_attempts",
            "do_not_repeat",
            "open_questions",
            "constraints",
            "related_claims",
            "blockers",
            "omitted",
            "source_closure_hash",
            "memory_refs",
            "memory_snapshot_hash",
            "pack_hash",
        )
        if pack.get(key) not in (None, [], {}, "")
    }
    return "## ARA task context (source-bound)\n" + json.dumps(
        visible, indent=2, ensure_ascii=False, default=str
    )


def _shared_root_for(root: Path) -> Path | None:
    from ai_scientist.utils.privacy import resolve_portable_path

    manifest = _read_json(root / "manifest.json")
    if isinstance(manifest, dict) and manifest.get("project_dir"):
        project_dir = resolve_portable_path(manifest["project_dir"], base=root)
        if project_dir is not None:
            return project_dir / ".ara-store"
    if root.parent.name == "ara":
        return root.parent.parent / ".ara-store"
    return None


def persist_context_pack(
    ara_root: str | Path,
    pack: dict[str, Any],
    *,
    consumer: str | None = None,
) -> str:
    """Store a pack once and append a compact compilation receipt."""

    root = Path(ara_root).expanduser().resolve()
    validate_context_pack(pack)
    store = ObjectStore(root, shared_root=_shared_root_for(root))
    ref = store.put_json(pack)
    receipt = {
        "schema": "ara.context.receipt.v1",
        "type": "compiled",
        "recorded_at": _now_iso(),
        "consumer": consumer or pack.get("consumer"),
        "intent": pack.get("intent"),
        "target": pack.get("target") or {},
        "pack_ref": ref.to_json(),
        "pack_hash": pack.get("pack_hash"),
        "source_closure_hash": pack.get("source_closure_hash"),
        "memory_snapshot_hash": pack.get("memory_snapshot_hash"),
        "omitted": pack.get("omitted") or {},
    }
    receipt = seal_context_receipt(receipt)
    durable_append_text(
        root / CONTEXT_RECEIPTS_RELPATH,
        json.dumps(receipt, ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n",
    )
    return ref.hash


def persist_active_context_pack(
    pack: dict[str, Any],
    *,
    consumer: str | None = None,
) -> str | None:
    root = active_ara_root()
    if not root:
        return None
    try:
        return persist_context_pack(root, pack, consumer=consumer)
    except Exception:
        return None


def record_context_consumption(
    ara_root: str | Path,
    *,
    pack_ref: str,
    consumer: str,
    output_type: str,
    output_id: str,
) -> None:
    root = Path(ara_root).expanduser().resolve()
    stored_pack = ObjectStore(root, shared_root=_shared_root_for(root)).get_json(
        pack_ref
    )
    validate_context_pack(stored_pack)
    receipt = {
        "schema": "ara.context.receipt.v1",
        "type": "consumed",
        "recorded_at": _now_iso(),
        "consumer": consumer,
        "pack_ref": pack_ref,
        "output": {"type": output_type, "id": output_id},
        "pack_hash": stored_pack.get("pack_hash"),
        "source_closure_hash": stored_pack.get("source_closure_hash"),
        "memory_snapshot_hash": stored_pack.get("memory_snapshot_hash"),
    }
    receipt = seal_context_receipt(receipt)
    durable_append_text(
        root / CONTEXT_RECEIPTS_RELPATH,
        json.dumps(receipt, ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n",
    )


def record_active_context_consumption(
    *,
    pack_ref: str,
    consumer: str,
    output_type: str,
    output_id: str,
) -> None:
    root = active_ara_root()
    if not root:
        return
    try:
        record_context_consumption(
            root,
            pack_ref=pack_ref,
            consumer=consumer,
            output_type=output_type,
            output_id=output_id,
        )
    except Exception:
        return


__all__ = [
    "ARAContextError",
    "CONTEXT_INTENTS",
    "CONTEXT_RECEIPTS_RELPATH",
    "compile_context_pack",
    "compile_live_audit_context",
    "compile_live_continue_context",
    "compile_live_write_context",
    "persist_active_context_pack",
    "persist_context_pack",
    "record_active_context_consumption",
    "record_context_consumption",
    "render_context_pack_for_prompt",
    "validate_context_pack",
]
