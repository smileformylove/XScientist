"""Re-execution verifier for ARA nodes.

Existing `repair_verifier.py` does *structural* checks against manifest
artifacts — it never runs code. This module is the mechanical counterpart:
given an ARA directory, re-execute one or more nodes and diff the fresh
metric against the recorded one.

Kept as an opt-in helper (env flag ``AI_SCIENTIST_ARA_REEXEC``) so the main
pipeline is unchanged unless the user explicitly turns this on.

Public API
----------
- ``reexec_node(ara_root, node_id, ...)`` — run one node, return a verdict dict.
- ``reexec_ara(ara_root, ...)`` — pick a handful of nodes and verdict each.
- ``reexec_enabled()`` — cheap env-flag helper for callers to gate on.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.ara_artifact import ara_root_for_project

logger = logging.getLogger(__name__)

REEXEC_ENV_FLAG = "AI_SCIENTIST_ARA_REEXEC"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def reexec_enabled() -> bool:
    return str(os.environ.get(REEXEC_ENV_FLAG) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _import_fork_utils():
    """Re-use the parsing helpers from run_ara_fork without pulling its argparse."""
    from importlib import import_module, util

    # run_ara_fork lives at the repo root, not inside a package. Import by path.
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "run_ara_fork.py"
    if not module_path.exists():
        raise RuntimeError(f"run_ara_fork.py not found at {module_path}")
    spec = util.spec_from_file_location("_ara_fork_helpers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_ara_fork.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_candidate_nodes(ara_root: Path, *, limit: int, include_buggy: bool) -> list[str]:
    """Pick which nodes to re-execute.

    Heuristic: the *best-metric* good nodes first (they usually carry the
    claims the paper leans on), plus a small tail of buggy nodes when the
    caller asks — knowing "what didn't work" is half the value of ARA.
    """
    graph = _load_json(ara_root / "exploration_graph.json") or {}
    entries: list[dict[str, Any]] = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    if not entries:
        return []

    def metric_value(node: dict[str, Any]) -> float:
        metric = node.get("metric") or {}
        if isinstance(metric, dict):
            try:
                return float(metric.get("value"))
            except (TypeError, ValueError):
                return float("-inf")
        return float("-inf")

    good = sorted(
        (n for n in entries if not n.get("is_buggy") and (ara_root / "nodes" / str(n.get("id") or "") / "code.py").exists()),
        key=metric_value,
        reverse=True,
    )
    picks: list[str] = [str(n.get("id")) for n in good[:limit]]
    if include_buggy:
        buggy = [n for n in entries if n.get("is_buggy") and (ara_root / "nodes" / str(n.get("id") or "") / "code.py").exists()]
        for n in buggy[: max(0, limit // 2)]:
            nid = str(n.get("id"))
            if nid not in picks:
                picks.append(nid)
    return picks


def reexec_node(
    ara_root: str | Path,
    node_id: str,
    *,
    python: str | None = None,
    cwd: str | Path | None = None,
    timeout: int = 900,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Run one node's ``code.py`` in a subprocess and produce a verdict dict.

    The return shape mirrors the JSON written by ``run_ara_fork.py exec`` so
    callers can persist either form interchangeably.
    """
    ara_root = Path(ara_root).expanduser().resolve()
    node_dir = ara_root / "nodes" / node_id
    code_path = node_dir / "code.py"
    if not code_path.exists():
        return {"node_id": node_id, "status": "no_code", "error": "code.py missing"}

    helpers = _import_fork_utils()
    manifest = _load_json(ara_root / "manifest.json") or {}
    default_cwd = Path(manifest.get("source_exp_dir") or "").expanduser()
    resolved_cwd = Path(cwd).expanduser() if cwd else default_cwd
    if not resolved_cwd.exists():
        resolved_cwd = node_dir

    python_bin = python or sys.executable
    started = _now_iso()
    try:
        proc = subprocess.run(
            [python_bin, str(code_path)],
            cwd=str(resolved_cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        timed_out = False
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout, stderr = (exc.stdout or ""), (exc.stderr or "")
        returncode = -1

    fresh = helpers._parse_metric_from_stdout(stdout + "\n" + stderr)
    recorded = (_load_json(node_dir / "metrics.json") or {}).get("metric")
    comparison = helpers._compare_metrics(recorded, fresh, tolerance)

    verdict = {
        "schema": "ara.reexec.v1",
        "node_id": node_id,
        "ara_root": str(ara_root),
        "python": python_bin,
        "cwd": str(resolved_cwd),
        "started_at": started,
        "finished_at": _now_iso(),
        "returncode": returncode,
        "timed_out": timed_out,
        "recorded_metric": recorded,
        "fresh_metric": fresh,
        "comparison": comparison,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    return verdict


def _write_reexec_report(ara_root: Path, verdicts: list[dict[str, Any]]) -> Path:
    verify_dir = ara_root / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_iso().replace(":", "").replace("-", "")[:15]
    path = verify_dir / f"reexec_batch_{stamp}.json"
    passed = sum(
        1
        for v in verdicts
        if v.get("comparison", {}).get("within_tolerance") is True and not v.get("timed_out")
    )
    payload = {
        "schema": "ara.reexec.batch.v1",
        "generated_at": _now_iso(),
        "ara_root": str(ara_root),
        "verdict_count": len(verdicts),
        "passed": passed,
        "verdicts": verdicts,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def reexec_ara(
    ara_root: str | Path,
    *,
    node_ids: Iterable[str] | None = None,
    limit: int = 3,
    include_buggy: bool = False,
    python: str | None = None,
    timeout: int = 900,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Re-execute a small set of nodes and write a batch report into ``verify/``."""
    ara_root = Path(ara_root).expanduser().resolve()
    if node_ids is None:
        picks = _select_candidate_nodes(ara_root, limit=limit, include_buggy=include_buggy)
    else:
        picks = [str(nid) for nid in node_ids]
    if not picks:
        return {"status": "no_candidates", "ara_root": str(ara_root)}

    verdicts: list[dict[str, Any]] = []
    for nid in picks:
        verdict = reexec_node(
            ara_root, nid, python=python, timeout=timeout, tolerance=tolerance
        )
        verdicts.append(verdict)

    report_path = _write_reexec_report(ara_root, verdicts)
    return {
        "status": "ok",
        "ara_root": str(ara_root),
        "report_path": str(report_path),
        "node_ids": picks,
        "verdict_count": len(verdicts),
    }


def reexec_project(
    project_dir: str | Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Convenience: re-execute the newest export under each ``<project>/ara/<idea>``."""
    root = ara_root_for_project(project_dir)
    if not root.exists():
        return []
    reports: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        reports.append(reexec_ara(manifest_path.parent, **kwargs))
    return reports
