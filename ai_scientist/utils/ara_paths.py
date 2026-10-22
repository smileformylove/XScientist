"""Safe path helpers for ARA node bundles.

ARA v1 node identifiers are portable ASCII names, not filesystem paths.  The
contract is intentionally narrow so every producer and consumer maps an ID to
exactly one directory below ``<ara_root>/nodes`` on every supported platform.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

NODE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
NODE_ID_RE = re.compile(NODE_ID_PATTERN)


def validate_node_id(node_id: Any) -> str:
    """Return a valid ARA v1 node ID or raise ``ValueError``.

    IDs are deliberately not stripped or coerced here.  Callers crossing a
    legacy serialization boundary must make that conversion explicitly so a
    filesystem-facing API never silently reinterprets attacker-controlled data.
    """

    if not isinstance(node_id, str) or NODE_ID_RE.fullmatch(node_id) is None:
        raise ValueError(f"unsafe ARA node id: {node_id!r}")
    return node_id


def resolve_nodes_root(ara_root: str | Path) -> Path:
    """Resolve ``<ara_root>/nodes`` and require it to remain a real child."""

    resolved_root = Path(ara_root).expanduser().resolve(strict=False)
    candidate = resolved_root / "nodes"
    if candidate.is_symlink():
        raise ValueError("unsafe ARA nodes directory: symlinks are not allowed")
    nodes_root = candidate.resolve(strict=False)
    if nodes_root == resolved_root or not nodes_root.is_relative_to(resolved_root):
        raise ValueError("unsafe ARA nodes directory: resolves outside ARA root")
    return nodes_root


def resolve_node_directory(ara_root: str | Path, node_id: Any) -> Path:
    """Resolve one node directory and prove it is contained by ``nodes/``."""

    safe_node_id = validate_node_id(node_id)
    nodes_root = resolve_nodes_root(ara_root)
    candidate = nodes_root / safe_node_id
    if candidate.is_symlink():
        raise ValueError(
            f"unsafe ARA node directory for {safe_node_id!r}: symlinks are not allowed"
        )
    node_dir = candidate.resolve(strict=False)
    if node_dir == nodes_root or not node_dir.is_relative_to(nodes_root):
        raise ValueError(f"unsafe ARA node directory for {safe_node_id!r}")
    return node_dir


def resolve_node_file(node_dir: str | Path, filename: str) -> Path:
    """Resolve a fixed bundle file without following a symlink out of its node."""

    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError(f"unsafe ARA node filename: {filename!r}")

    resolved_node_dir = Path(node_dir).expanduser().resolve(strict=False)
    candidate = resolved_node_dir / filename
    if candidate.is_symlink():
        raise ValueError(f"unsafe ARA node file {filename!r}: symlinks are not allowed")
    resolved = candidate.resolve(strict=False)
    if resolved == resolved_node_dir or not resolved.is_relative_to(resolved_node_dir):
        raise ValueError(
            f"unsafe ARA node file {filename!r}: resolves outside node directory"
        )
    return resolved


def resolve_node_paths(ara_root: str | Path, node_id: Any) -> tuple[Path, Path, Path]:
    """Resolve the node directory and its two verifier inputs."""

    node_dir = resolve_node_directory(ara_root, node_id)
    return (
        node_dir,
        resolve_node_file(node_dir, "code.py"),
        resolve_node_file(node_dir, "metrics.json"),
    )
