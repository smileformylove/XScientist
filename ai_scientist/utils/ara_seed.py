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

    The fork layout is:

        <fork_dir>/
            node/code.py           # the code to seed with
            node/metrics.json      # optional; used to populate provenance
            fork.json              # provenance record
            manifest.json          # parent ARA manifest (read-only lineage)
    """
    fork_dir = Path(fork_dir).expanduser().resolve()
    code_path = fork_dir / "node" / "code.py"
    if not code_path.exists():
        raise FileNotFoundError(f"Fork missing node/code.py: {fork_dir}")
    code = code_path.read_text(encoding="utf-8")

    fork_meta = _load_json(fork_dir / "fork.json") or {}
    parent_manifest = _load_json(fork_dir / "manifest.json") or {}
    metrics = _load_json(fork_dir / "node" / "metrics.json") or {}

    provenance = {
        "parent_ara_root": fork_meta.get("source_ara"),
        "parent_node_id": fork_meta.get("source_node_id"),
        "parent_content_hash": (
            fork_meta.get("source_content_hash")
            or metrics.get("content_hash")
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
        "parent_manifest_idea": parent_manifest.get("idea"),
        "plan": plan,
        "code": code,
    }


def build_seed_manifest_from_ara_node(
    *, ara_root: str | Path, node_id: str
) -> dict[str, Any]:
    """Seed directly from an ARA node without going through a fork step."""
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
    return {
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


def stage_seed_manifest(
    manifest: dict[str, Any], *, workspace_dir: str | Path
) -> Path:
    """Write the seed manifest to disk and return its path.

    Callers set ``AI_SCIENTIST_ARA_SEED_PATH=<returned path>`` before invoking
    the BFTS pipeline. BFTS reads the manifest via ``load_active_seed``.
    """
    workspace = Path(workspace_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    dest = workspace / SEED_MANIFEST_NAME
    dest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return dest


def load_active_seed() -> dict[str, Any] | None:
    """Return the currently-active seed manifest (if any).

    Called by ``parallel_agent._draft`` to decide whether to bypass the LLM.
    Returns ``None`` when the env var is unset, empty, or points to a missing
    / malformed file — the caller falls back to normal drafting.
    """
    raw = os.environ.get(SEED_ENV_VAR)
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        logger.warning("ARA seed manifest not found at %s", path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ARA seed manifest unreadable at %s: %s", path, exc)
        return None
    if not isinstance(payload, dict) or "code" not in payload:
        logger.warning("ARA seed manifest at %s is missing 'code'", path)
        return None
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
