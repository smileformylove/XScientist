"""Dependency-free DAG analysis for ARA exploration graphs."""

from __future__ import annotations

from collections import deque
from typing import Any


def _node_sort_key(node: dict[str, Any]) -> tuple[float, int, str]:
    ctime = node.get("ctime")
    ctime_value = float(ctime) if isinstance(ctime, (int, float)) else float("inf")
    step = node.get("step")
    step_value = int(step) if isinstance(step, int) and not isinstance(step, bool) else 10**9
    return (ctime_value, step_value, str(node.get("id") or ""))


def _graph_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in (graph.get("nodes") or []) if isinstance(node, dict)]


def _record_issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    code: str,
    message: str,
    path: str,
) -> None:
    issues.append({
        "severity": severity,
        "code": code,
        "message": message,
        "path": path,
    })


def analyze_exploration_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Return DAG diagnostics for an ARA ``exploration_graph.json`` payload."""

    nodes = _graph_nodes(graph)
    issues: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()

    for index, node in enumerate(nodes):
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            _record_issue(
                issues,
                severity="error",
                code="missing_node_id",
                message="node entry is missing a non-empty id",
                path=f"nodes[{index}].id",
            )
            continue
        if node_id in node_by_id:
            duplicate_ids.add(node_id)
            _record_issue(
                issues,
                severity="error",
                code="duplicate_node_id",
                message=f"duplicate node id {node_id!r}",
                path=f"nodes[{index}].id",
            )
            continue
        node_by_id[node_id] = node

    edge_seen: set[tuple[str, str]] = set()
    edge_entries: list[dict[str, Any]] = []

    def add_edge(parent: Any, child: Any, *, source: str, path: str, stage: Any = None) -> None:
        parent_id = str(parent or "").strip()
        child_id = str(child or "").strip()
        if not parent_id or not child_id:
            _record_issue(
                issues,
                severity="error",
                code="invalid_edge",
                message="edge is missing parent or child",
                path=path,
            )
            return
        if parent_id == child_id:
            _record_issue(
                issues,
                severity="error",
                code="self_loop",
                message=f"node {parent_id!r} links to itself",
                path=path,
            )
        if parent_id not in node_by_id:
            _record_issue(
                issues,
                severity="error",
                code="missing_edge_parent",
                message=f"edge parent {parent_id!r} is not present in nodes",
                path=path,
            )
        if child_id not in node_by_id:
            _record_issue(
                issues,
                severity="error",
                code="missing_edge_child",
                message=f"edge child {child_id!r} is not present in nodes",
                path=path,
            )
        edge_key = (parent_id, child_id)
        if edge_key in edge_seen:
            return
        edge_seen.add(edge_key)
        edge_entries.append({
            "parent": parent_id,
            "child": child_id,
            "stage": stage,
            "source": source,
        })

    for index, edge in enumerate(graph.get("edges") or []):
        if not isinstance(edge, dict):
            _record_issue(
                issues,
                severity="error",
                code="invalid_edge",
                message="edge entry is not an object",
                path=f"edges[{index}]",
            )
            continue
        add_edge(
            edge.get("parent"),
            edge.get("child"),
            source="edges",
            path=f"edges[{index}]",
            stage=edge.get("stage"),
        )

    for index, node in enumerate(nodes):
        node_id = str(node.get("id") or "").strip()
        if not node_id or node_id in duplicate_ids:
            continue
        parent_id = node.get("parent_id")
        if parent_id:
            add_edge(
                parent_id,
                node_id,
                source="parent_id",
                path=f"nodes[{index}].parent_id",
                stage=node.get("stage"),
            )
        children = node.get("children") or []
        if isinstance(children, list):
            for child_index, child_id in enumerate(children):
                add_edge(
                    node_id,
                    child_id,
                    source="children",
                    path=f"nodes[{index}].children[{child_index}]",
                    stage=node.get("stage"),
                )
        elif children:
            _record_issue(
                issues,
                severity="warning",
                code="invalid_children",
                message="children field is not an array",
                path=f"nodes[{index}].children",
            )

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_by_id}
    for edge in edge_entries:
        parent = edge["parent"]
        child = edge["child"]
        if parent not in node_by_id or child not in node_by_id:
            continue
        if child not in adjacency[parent]:
            adjacency[parent].add(child)
            indegree[child] += 1

    ordered_ids = [
        str(node.get("id"))
        for node in sorted(node_by_id.values(), key=_node_sort_key)
    ]
    order_index = {node_id: index for index, node_id in enumerate(ordered_ids)}
    queue = deque([node_id for node_id in ordered_ids if indegree.get(node_id, 0) == 0])
    indegree_work = dict(indegree)
    topological_order: list[str] = []
    depth: dict[str, int] = {node_id: 0 for node_id in node_by_id}

    while queue:
        node_id = queue.popleft()
        topological_order.append(node_id)
        for child_id in sorted(
            adjacency.get(node_id, []),
            key=lambda cid: order_index.get(cid, len(ordered_ids)),
        ):
            depth[child_id] = max(depth.get(child_id, 0), depth.get(node_id, 0) + 1)
            indegree_work[child_id] -= 1
            if indegree_work[child_id] == 0:
                queue.append(child_id)

    cycle_nodes = [node_id for node_id, value in indegree_work.items() if value > 0]
    if cycle_nodes:
        _record_issue(
            issues,
            severity="error",
            code="cycle_detected",
            message="exploration graph contains a directed cycle",
            path="edges",
        )

    root_ids = [node_id for node_id in ordered_ids if indegree.get(node_id, 0) == 0]
    leaf_ids = [node_id for node_id in ordered_ids if not adjacency.get(node_id)]
    error_count = sum(1 for issue in issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")

    return {
        "is_dag": error_count == 0 and not cycle_nodes,
        "node_count": len(node_by_id),
        "edge_count": len(
            [
                edge
                for edge in edge_entries
                if edge["parent"] in node_by_id and edge["child"] in node_by_id
            ]
        ),
        "root_ids": root_ids,
        "leaf_ids": leaf_ids,
        "topological_order": topological_order if not cycle_nodes else [],
        "max_depth": max(depth.values()) if depth and not cycle_nodes else None,
        "cycle_nodes": sorted(cycle_nodes),
        "issues": issues,
        "error_count": error_count,
        "warning_count": warning_count,
    }


def graph_with_dag_metadata(graph: dict[str, Any]) -> dict[str, Any]:
    payload = dict(graph)
    payload["dag"] = analyze_exploration_graph(payload)
    return payload
