"""Minimal ARA export for producers that never ran a tree search.

Rationale
---------
``continuous_paper_generator.py`` has two writeup paths (professional /
adaptive) that jump straight from *idea → manuscript* — no BFTS, no
``journal.json``. Full ``export_ara`` would produce an ARA whose
``exploration_graph.json`` is empty, which is fine but misleading (it looks
like "we ran a search and got nothing").

This module writes an *explicitly manuscript-only* ARA:

- ``exploration_graph.json`` is a single synthetic node whose id encodes the
  manuscript hash, ``is_seed_node: true`` so downstream tools understand it
  was not the product of a search.
- ``manifest.json`` sets ``missing`` to record the tree-less nature.
- No ``nodes/<id>/code.py`` (there wasn't any code) — but a
  ``nodes/<id>/manuscript.pdf`` symlink/copy if the caller provides one.

The whole point is that even the "论文工厂" flows produce an artifact that:
  1. Passes ``validate_ara``.
  2. Carries ``provenance`` if forked from something else.
  3. Can be re-fork-continued by another producer.

Kept small and additive — ``run_project.py`` and BFTS-driven flows still use
``export_ara`` from ``ara_artifact.py``.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_scientist.protocol import PROTOCOL_VERSION, hash_node_payload
from ai_scientist.utils.ara_artifact import (
    ARAExportResult,
    ara_dir_for_idea,
    ara_root_for_project,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def export_minimal_ara(
    *,
    project_dir: str | Path,
    manuscript_pdf: str | Path | None,
    idea: dict[str, Any] | None = None,
    timestamp: str | None = None,
    writing_profile: str | None = None,
    provenance: dict[str, Any] | None = None,
    producer: str = "continuous_paper_generator",
) -> ARAExportResult:
    """Emit a conformant single-node ARA for a manuscript-only run.

    ``manuscript_pdf`` — when provided, is copied into ``nodes/<id>/``. When
    absent we still write everything else so the ARA validates; the caller
    can fill it in later.

    Returns the same ``ARAExportResult`` shape as ``export_ara`` so callers
    can treat both paths uniformly.
    """
    project_dir_path = Path(project_dir).expanduser().resolve()
    idea = idea or {}
    idea_name = str(idea.get("Name") or idea.get("name") or project_dir_path.name)

    ara_dir = ara_dir_for_idea(project_dir_path, idea_name, timestamp=timestamp)
    ara_dir.mkdir(parents=True, exist_ok=True)

    manuscript_bytes = b""
    if manuscript_pdf is not None:
        try:
            pdf_path = Path(manuscript_pdf)
            if pdf_path.exists():
                manuscript_bytes = pdf_path.read_bytes()
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Cannot read manuscript PDF %s: %s", manuscript_pdf, exc)

    # Synthetic node keyed by the manuscript content — gives us a
    # content-addressable id for the "code-less" case.
    node_id = hash_node_payload(
        code="",
        metric=None,
        extras={
            "producer": producer,
            "manuscript_bytes_len": len(manuscript_bytes),
            "idea_name": idea_name,
        },
    ).split(":", 1)[-1][:12] or "manuscript"
    node_id = f"manuscript-{node_id}"

    nodes_dir = ara_dir / "nodes" / node_id
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort: copy the PDF alongside the synthetic node.
    if manuscript_pdf is not None:
        src = Path(manuscript_pdf)
        if src.exists():
            try:
                shutil.copy2(src, nodes_dir / src.name)
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Could not copy manuscript to ARA: %s", exc)

    _write_json(
        nodes_dir / "metrics.json",
        {
            "metric": None,
            "analysis": (
                f"Manuscript-only ARA produced by {producer!r}; no exploration "
                "tree was run for this idea."
            ),
            "is_buggy": False,
            "is_buggy_plots": None,
            "exc_type": None,
            "exc_info": None,
            "exec_time": None,
            "content_hash": hash_node_payload(code="", metric=None, is_seed=True),
        },
    )

    now = _now_iso()
    graph = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_kind": "exploration_graph",
        "generated_at": now,
        "nodes": [
            {
                "id": node_id,
                "content_hash": hash_node_payload(code="", metric=None, is_seed=True),
                "stage": "manuscript_only",
                "step": 0,
                "parent_id": None,
                "children": [],
                "is_buggy": False,
                "is_seed_node": True,
                "is_seed_agg_node": False,
                "metric": None,
                "plan_excerpt": (
                    "Manuscript-only export: producer skipped BFTS and wrote "
                    "the manuscript directly."
                ),
                "exp_results_dir": None,
                "ctime": None,
                "artifacts_dir": f"nodes/{node_id}",
            }
        ],
        "edges": [],
        "source_journals": [],
        "counts": {"nodes": 1, "edges": 0, "buggy": 0},
    }
    _write_json(ara_dir / "exploration_graph.json", graph)

    manifest_payload = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_kind": "manifest",
        "created_at": now,
        "source_exp_dir": str(project_dir_path),
        "project_dir": str(project_dir_path),
        "idea": {
            "name": idea_name,
            "title": idea.get("Title") or idea.get("title"),
            "raw": idea,
        },
        "counts": {"nodes": 1, "edges": 0, "buggy_nodes": 0, "journals": 0, "claims": 0},
        "references": {
            "producer": producer,
            "writing_profile": writing_profile,
        },
        "missing": [
            "no journal.json (this producer skips BFTS)",
            "no env/ snapshot (call `run_ara_fork.py freeze` if you need one)",
        ],
    }
    if provenance:
        manifest_payload["provenance"] = dict(provenance)

    manifest_path = ara_dir / "manifest.json"
    _write_json(manifest_path, manifest_payload)

    # Tiny agent-facing README so downstream tools don't wonder what happened.
    (ara_dir / "README.md").write_text(
        (
            "# ARA (manuscript-only)\n\n"
            f"Producer: `{producer}`\n"
            f"Idea: `{idea_name}`\n\n"
            "This ARA has no exploration tree — the producer wrote the "
            "manuscript directly from the idea. `nodes/<id>/` contains the "
            "PDF (when available); `manifest.provenance` links back to a "
            "parent ARA if this run was seeded.\n"
        ),
        encoding="utf-8",
    )

    return ARAExportResult(
        root=ara_dir,
        manifest_path=manifest_path,
        missing=list(manifest_payload["missing"]),
        node_count=1,
        claim_count=0,
    )


def project_ara_root(project_dir: str | Path) -> Path:
    """Convenience passthrough so callers don't need to import ara_artifact."""
    return ara_root_for_project(project_dir)
