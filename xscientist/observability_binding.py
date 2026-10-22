"""Read-only projection of existing ARA / Research VCS source bindings."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value))


def _commit(value: Any) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value)
    )


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_observability_binding_projection(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Expose existing hashes and anchors without mutating the canonical DAG."""

    commit = graph.get("commit")
    graph_hash = graph.get("graph_hash")
    global_issues: list[str] = []
    if not _commit(commit):
        global_issues.append("invalid_research_commit")
    if not _sha256(graph_hash):
        global_issues.append("missing_research_graph_hash")
    else:
        try:
            actual = canonical_content_hash(
                {
                    key: value
                    for key, value in graph.items()
                    if key not in {"generated_at", "graph_hash"}
                }
            )
        except (TypeError, ValueError, OverflowError):
            actual = None
        if actual != graph_hash:
            global_issues.append("research_graph_hash_mismatch")

    nodes = [_object(node) for node in _rows(graph.get("nodes"))]
    ids = [node.get("id") for node in nodes]
    if any(not isinstance(value, str) for value in ids):
        global_issues.append("invalid_node_identity")
    known_nodes = {
        node["id"]: node for node in nodes if isinstance(node.get("id"), str)
    }
    if len(known_nodes) != len(nodes):
        global_issues.append("duplicate_or_invalid_node_identity")
    source_rows = [_object(source) for source in _rows(graph.get("sources"))]
    names = Counter(
        source.get("name")
        for source in source_rows
        if isinstance(source.get("name"), str)
    )
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        name = source.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"ara:\d+", name):
            continue
        issues = list(global_issues)
        unbound_reasons: list[str] = []
        if names[name] != 1:
            issues.append("duplicate_source_identity")
        manifest = _object(source.get("manifest_integrity"))
        binding = _object(source.get("graph_binding"))
        integrity = _object(source.get("integrity"))
        manifest_hash = source.get("manifest_hash")
        ara_graph_hash = binding.get("graph_hash")
        snapshot_ref = source.get("snapshot_ref")
        if not _sha256(manifest_hash):
            issues.append("missing_or_invalid_manifest_hash")
        if not _sha256(ara_graph_hash):
            issues.append("missing_or_invalid_ara_graph_hash")
        if (
            manifest.get("state") not in {"clean", "revised"}
            or manifest.get("ok") is not True
        ):
            unbound_reasons.append("manifest_not_verified")
        if manifest.get("current_hash") != manifest_hash:
            issues.append("manifest_hash_mismatch")
        if (
            manifest.get("expected_hash") is not None
            and manifest.get("expected_hash") != manifest_hash
        ):
            issues.append("expected_manifest_hash_mismatch")
        if (
            binding.get("state") not in {"verified", "commit_bound"}
            or binding.get("ok") is not True
        ):
            unbound_reasons.append("graph_not_checkpoint_bound")
        if binding.get("state") == "verified" and not _sha256(
            binding.get("expected_graph_hash")
        ):
            issues.append("missing_expected_graph_hash")
        if (
            binding.get("expected_graph_hash") is not None
            and binding.get("expected_graph_hash") != ara_graph_hash
        ):
            issues.append("ara_graph_hash_mismatch")
        if not _commit(snapshot_ref):
            unbound_reasons.append("snapshot_not_immutable_commit")
        if (
            integrity.get("is_dag") is not True
            or type(integrity.get("error_count")) is not int
            or integrity.get("error_count") != 0
        ):
            issues.append("source_dag_integrity_invalid")

        declared_ids = source.get("bound_object_ids")
        valid_ids = isinstance(declared_ids, list) and all(
            isinstance(item, str) and item for item in declared_ids
        )
        declared = set(declared_ids) if valid_ids else set()
        count = source.get("binding_count")
        if (
            not valid_ids
            or type(count) is not int
            or count < 0
            or count != len(declared)
            or len(declared) != len(declared_ids)
        ):
            issues.append("invalid_declared_anchor_set")
        actual_targets: set[str] = set()
        for edge in _rows(graph.get("edges")):
            edge = _object(edge)
            parent_id = edge.get("source")
            if (
                edge.get("type") != "anchors"
                or not isinstance(parent_id, str)
                or not parent_id.startswith(name + ":")
            ):
                continue
            target_id = edge.get("target")
            parent = known_nodes.get(parent_id, {})
            target = (
                known_nodes.get(target_id, {}) if isinstance(target_id, str) else {}
            )
            if parent.get("source") != "ara" or target.get("source") != "research_vcs":
                issues.append("anchor_endpoint_invalid")
                continue
            actual_targets.add(target_id)
            if edge.get("category") != "verification":
                unbound_reasons.append("anchor_is_not_verified")
        if actual_targets != declared:
            issues.append("anchor_set_mismatch")
        if not actual_targets:
            unbound_reasons.append("missing_exact_research_object_anchor")
        reasons = sorted(set(issues + unbound_reasons))
        state = "invalid" if issues else ("unbound" if unbound_reasons else "verified")
        binding_core = {
            "vcs_commit": commit,
            "ara_snapshot_ref": snapshot_ref,
            "ara_source": name,
            "manifest_hash": manifest_hash,
            "graph_hash": ara_graph_hash,
            "expected_graph_hash": binding.get("expected_graph_hash"),
            "anchor_object_ids": sorted(actual_targets),
        }
        rows.append(
            {
                "source": name,
                "state": state,
                "vcs_commit": commit if _commit(commit) else None,
                "snapshot_ref": snapshot_ref if isinstance(snapshot_ref, str) else None,
                "manifest_hash": manifest_hash if _sha256(manifest_hash) else None,
                "graph_hash": ara_graph_hash if _sha256(ara_graph_hash) else None,
                "expected_graph_hash": (
                    binding.get("expected_graph_hash")
                    if _sha256(binding.get("expected_graph_hash"))
                    else None
                ),
                "graph_binding_state": (
                    binding.get("state")
                    if isinstance(binding.get("state"), str)
                    else None
                ),
                "anchor_count": len(actual_targets),
                "anchor_object_ids": sorted(actual_targets),
                "reasons": reasons,
                "binding_hash": (
                    canonical_content_hash(binding_core)
                    if state == "verified"
                    else None
                ),
            }
        )
    counts = Counter(row["state"] for row in rows)
    state = (
        "invalid"
        if counts["invalid"] or global_issues
        else (
            "verified"
            if rows and counts["verified"] == len(rows)
            else ("partial" if counts["verified"] else "unbound")
        )
    )
    return {
        "state": state,
        "rows": rows,
        "verified_count": counts["verified"],
        "invalid_count": counts["invalid"],
        "research_graph_hash": graph_hash if _sha256(graph_hash) else None,
        "issues": global_issues,
        "presentation_only": True,
        "authority": "none",
        "binding_hash_scope": "Existing Research DAG hash; not a signature, admission, claim-role, or trajectory binding.",
    }


__all__ = ["build_observability_binding_projection"]
