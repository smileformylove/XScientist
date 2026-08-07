"""Seed a fresh tree search from an ARA fork.

Motivation
----------
An ARA node is only useful if the *next* agent can start from it. This module
wires the "seed" side of that story:

1. A caller (``run_project.py``) points us at a fork (or an ARA + node_id).
2. We build a *seed manifest* — a small JSON with the exact ``code`` + ``plan``
   the BFTS ``_draft`` step should use, plus provenance metadata.
3. We stage that manifest to disk and hand its path back through an env var.
4. When BFTS's ``_draft`` fires, ``parallel_agent.py`` checks for the env var
   and, if present, returns the seeded ``Node`` *without* calling the LLM.

The env var boundary is deliberate — it keeps BFTS unaware of the ARA layout,
and it makes seeding trivially opt-in (unset var → default behaviour).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_scientist.protocol import PROTOCOL_VERSION
from ai_scientist.utils.privacy import relativize_path_fields, resolve_portable_path

logger = logging.getLogger(__name__)

SEED_ENV_VAR = "AI_SCIENTIST_ARA_SEED_PATH"
SEED_MANIFEST_NAME = "ara_seed.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_seed_manifest_from_fork(fork_dir: str | Path) -> dict[str, Any]:
    """Read a fork directory (as produced by ``run_ara_fork.py fork``) and
    build a seed manifest.

    Layouts we accept:

      New (protocol-conformant ARA; see SPEC.md §7.1):
          <fork_dir>/
              manifest.json                    # own manifest, has provenance
              exploration_graph.json           # single-node graph
              nodes/<node_id>/code.py          # the code to seed with
              nodes/<node_id>/metrics.json
              fork.json                        # legacy marker

      Legacy (pre-O6):
          <fork_dir>/
              node/code.py
              node/metrics.json
              manifest.json                    # copy of parent's manifest
              fork.json
    """
    fork_dir = Path(fork_dir).expanduser().resolve()

    fork_meta = _load_json(fork_dir / "fork.json") or {}
    fork_manifest = _load_json(fork_dir / "manifest.json") or {}
    node_id = fork_meta.get("source_node_id")

    # Prefer the new layout (nodes/<id>/), then fall back to legacy "node/".
    candidate_dirs: list[Path] = []
    if node_id:
        candidate_dirs.append(fork_dir / "nodes" / str(node_id))
    candidate_dirs.append(fork_dir / "node")
    code_path: Path | None = None
    node_dir_used: Path | None = None
    for candidate in candidate_dirs:
        if (candidate / "code.py").exists():
            code_path = candidate / "code.py"
            node_dir_used = candidate
            break
    if code_path is None:
        raise FileNotFoundError(
            f"Fork {fork_dir} exposes no readable code.py "
            f"(checked {', '.join(str(c) for c in candidate_dirs)})"
        )
    code = code_path.read_text(encoding="utf-8")
    metrics = _load_json(node_dir_used / "metrics.json") if node_dir_used else {}
    if not isinstance(metrics, dict):
        metrics = {}

    parent_ara = resolve_portable_path(fork_meta.get("source_ara"), base=fork_dir)
    provenance = {
        "parent_ara_root": str(parent_ara) if parent_ara is not None else None,
        "parent_node_id": fork_meta.get("source_node_id"),
        "parent_content_hash": (
            fork_meta.get("source_content_hash") or metrics.get("content_hash")
        ),
    }
    plan = (
        f"Continuing from ARA node {provenance['parent_node_id']} "
        f"({provenance['parent_content_hash'] or 'no-hash'}). "
        "Baseline code is copied verbatim; refine and explore from here."
    )
    return {
        "schema_version": PROTOCOL_VERSION,
        "protocol_kind": "seed",
        "created_at": _now_iso(),
        "source_fork_dir": str(fork_dir),
        "provenance": provenance,
        "parent_manifest_idea": fork_manifest.get("idea"),
        "plan": plan,
        "code": code,
    }


def build_seed_manifest_from_ara_node(
    *, ara_root: str | Path, node_id: str, applies_to_idea_name: str | None = None
) -> dict[str, Any]:
    """Seed directly from an ARA node without going through a fork step.

    ``applies_to_idea_name`` — when set, ``load_active_seed`` will refuse to
    hand the manifest back to a differently-named idea. This is the O8 knob
    for parallel-idea projects: without it, one seed would silently short-
    circuit every idea's draft.
    """
    ara_root = Path(ara_root).expanduser().resolve()
    node_dir = ara_root / "nodes" / node_id
    code_path = node_dir / "code.py"
    if not code_path.exists():
        raise FileNotFoundError(f"ARA node has no code.py: {node_dir}")
    metrics = _load_json(node_dir / "metrics.json") or {}
    manifest = _load_json(ara_root / "manifest.json") or {}
    provenance = {
        "parent_ara_root": str(ara_root),
        "parent_node_id": node_id,
        "parent_content_hash": metrics.get("content_hash"),
    }
    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_kind": "seed",
        "created_at": _now_iso(),
        "source_fork_dir": None,
        "provenance": provenance,
        "parent_manifest_idea": manifest.get("idea"),
        "plan": (
            f"Continuing from ARA node {node_id} "
            f"({provenance['parent_content_hash'] or 'no-hash'})."
        ),
        "code": code_path.read_text(encoding="utf-8"),
    }
    if applies_to_idea_name:
        payload["applies_to_idea_name"] = str(applies_to_idea_name)
    return payload


def stage_seed_manifest(manifest: dict[str, Any], *, workspace_dir: str | Path) -> Path:
    """Write the seed manifest to disk and return its path.

    Callers set ``AI_SCIENTIST_ARA_SEED_PATH=<returned path>`` before invoking
    the BFTS pipeline. BFTS reads the manifest via ``load_active_seed``.
    """
    workspace = Path(workspace_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    dest = workspace / SEED_MANIFEST_NAME
    portable_manifest = relativize_path_fields(manifest, base=workspace)
    dest.write_text(
        json.dumps(portable_manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return dest


def load_active_seed(*, current_idea_name: str | None = None) -> dict[str, Any] | None:
    """Return the currently-active seed manifest (if any).

    Called by ``parallel_agent._draft`` to decide whether to bypass the LLM.
    Returns ``None`` when:

      - the env var is unset / empty / dangles at a missing file;
      - the manifest lacks a usable ``code`` field;
      - a ``.consumed`` sidecar exists (consume-once, see O8);
      - the manifest declares ``applies_to_idea_name`` but the caller's
        ``current_idea_name`` doesn't match.

    On successful load we drop a ``<manifest>.consumed`` marker so subsequent
    ``_draft`` calls (e.g. sibling ideas or worker restarts) fall back to
    normal LLM drafting. Callers who want reusable seeds can delete the
    marker between runs.
    """
    raw = os.environ.get(SEED_ENV_VAR)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        logger.warning("ARA seed manifest not found at %s", path)
        return None
    consumed_marker = path.with_suffix(path.suffix + ".consumed")
    if consumed_marker.exists():
        logger.info(
            "ARA seed at %s already consumed (marker: %s); falling back to normal drafting",
            path,
            consumed_marker,
        )
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ARA seed manifest unreadable at %s: %s", path, exc)
        return None
    if not isinstance(payload, dict) or "code" not in payload:
        logger.warning("ARA seed manifest at %s is missing 'code'", path)
        return None

    applies_to = payload.get("applies_to_idea_name")
    if applies_to and current_idea_name and str(applies_to) != str(current_idea_name):
        logger.info(
            "ARA seed at %s is bound to idea %r; current idea is %r — skipping",
            path,
            applies_to,
            current_idea_name,
        )
        return None

    # Consume-once: write the marker atomically before returning so a racing
    # sibling draft won't also see the seed as active.
    try:
        consumed_marker.write_text(
            json.dumps({"consumed_at": _now_iso(), "by_idea": current_idea_name}),
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Could not write consumed marker %s: %s", consumed_marker, exc)
    return payload


def clear_active_seed_env() -> None:
    """Remove the seed env var so subsequent runs don't accidentally re-seed."""
    os.environ.pop(SEED_ENV_VAR, None)


def resolve_seed_manifest_from_source(source: str | Path) -> dict[str, Any]:
    """Convenience: accept either a fork dir or an ARA-root path.

    Detection is by presence of ``fork.json`` — that file is unique to
    directories produced by ``run_ara_fork.py fork``.
    """
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"seed source does not exist: {path}")
    if (path / "fork.json").exists():
        return build_seed_manifest_from_fork(path)
    if (path / "manifest.json").exists():
        raise ValueError(
            f"{path} looks like an ARA root, not a fork. "
            "Pass an explicit node_id via build_seed_manifest_from_ara_node()."
        )
    raise ValueError(f"Cannot determine seed source type at: {path}")
