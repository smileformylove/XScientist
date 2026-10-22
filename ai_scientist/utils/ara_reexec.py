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

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.protocol import ObjectStore
from ai_scientist.utils.ara_artifact import ara_root_for_project
from ai_scientist.utils.ara_metric_parser import (
    compare_metrics,
    numeric_metric_fraction,
    parse_metric_from_stdout,
    validate_metric_tolerance,
)
from ai_scientist.utils.ara_paths import resolve_node_paths, validate_node_id
from ai_scientist.utils.privacy import (
    redact_sensitive_payload,
    redact_sensitive_text,
    relative_path_reference,
    resolve_portable_path,
)

logger = logging.getLogger(__name__)

REEXEC_ENV_FLAG = "AI_SCIENTIST_ARA_REEXEC"
_READ_RECORDED_METRIC = object()


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


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _subprocess_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def validate_reexec_options(
    *, timeout: int | float, tolerance: float, limit: int | None = None
) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a finite positive number")
    try:
        finite_timeout = math.isfinite(float(timeout))
    except OverflowError:
        finite_timeout = False
    if not finite_timeout or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    validate_metric_tolerance(tolerance)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")


def write_unique_json_report(directory: Path, stem: str, payload: Any) -> Path:
    """Persist one report under a collision-resistant, exclusive filename."""
    directory.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    for _ in range(16):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = directory / f"{stem}_{stamp}_{uuid.uuid4().hex[:12]}.json"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate a unique report path under {directory}")


def require_indexed_node(ara_root: Path, node_id: str) -> None:
    """Reject materialized node bundles that are absent from the ARA graph."""
    graph = _load_json(ara_root / "exploration_graph.json")
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("ARA exploration_graph.json is missing or invalid")
    indexed: set[str] = set()
    for index, node in enumerate(graph["nodes"]):
        if not isinstance(node, dict):
            continue
        try:
            indexed.add(validate_node_id(node.get("id")))
        except ValueError as exc:
            raise ValueError(
                f"ARA exploration graph node[{index}] has an invalid id: {exc}"
            ) from exc
    if node_id not in indexed:
        raise ValueError(
            f"ARA node {node_id!r} is not indexed by exploration_graph.json"
        )


def _snapshot_reexec_inputs(
    ara_root: Path,
    node_ids: Iterable[str],
) -> tuple[dict[str, Any], dict[Path, str | None]]:
    recorded_metrics: dict[str, Any] = {}
    protected_inputs = [
        ara_root / "manifest.json",
        ara_root / "manifest.lock",
        ara_root / "exploration_graph.json",
    ]
    for node_id in node_ids:
        safe_node_id = validate_node_id(node_id)
        _, code_path, metrics_path = resolve_node_paths(ara_root, safe_node_id)
        recorded_metrics[safe_node_id] = (_load_json(metrics_path) or {}).get("metric")
        protected_inputs.extend((code_path, metrics_path))
    return recorded_metrics, {path: _file_sha256(path) for path in protected_inputs}


def _select_candidate_nodes(
    ara_root: Path, *, limit: int, include_buggy: bool
) -> list[str]:
    """Pick which nodes to re-execute.

    Heuristic: the *best-metric* good nodes first (they usually carry the
    claims the paper leans on), plus a small tail of buggy nodes when the
    caller asks — knowing "what didn't work" is half the value of ARA.
    """
    graph = _load_json(ara_root / "exploration_graph.json") or {}
    entries: list[dict[str, Any]] = [
        n for n in (graph.get("nodes") or []) if isinstance(n, dict)
    ]
    if not entries:
        return []

    def metric_value(node: dict[str, Any]) -> tuple[int, Any]:
        metric = node.get("metric") or {}
        if isinstance(metric, dict):
            value, error = numeric_metric_fraction(metric.get("value"))
            if error is None and value is not None:
                return (1, value)
        return (0, 0)

    resolved_entries: list[tuple[dict[str, Any], str, Path]] = []
    for node in entries:
        node_id = validate_node_id(node.get("id"))
        _, code_path, _ = resolve_node_paths(ara_root, node_id)
        resolved_entries.append((node, node_id, code_path))

    good = sorted(
        (
            (node, node_id)
            for node, node_id, code_path in resolved_entries
            if not node.get("is_buggy") and code_path.exists()
        ),
        key=lambda item: metric_value(item[0]),
        reverse=True,
    )
    picks: list[str] = [node_id for _, node_id in good[:limit]]
    if include_buggy:
        buggy = [
            (node, node_id)
            for node, node_id, code_path in resolved_entries
            if node.get("is_buggy") and code_path.exists()
        ]
        for _, nid in buggy[: max(0, limit // 2)]:
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
    recorded_metric: Any = _READ_RECORDED_METRIC,
    protected_input_hashes: dict[Path, str | None] | None = None,
) -> dict[str, Any]:
    """Run one node's ``code.py`` in a subprocess and produce a verdict dict.

    The return shape mirrors the JSON written by ``run_ara_fork.py exec`` so
    callers can persist either form interchangeably.
    """
    validate_reexec_options(timeout=timeout, tolerance=tolerance)
    ara_root = Path(ara_root).expanduser().resolve()
    node_dir, code_path, _ = resolve_node_paths(ara_root, node_id)
    require_indexed_node(ara_root, node_id)
    if not code_path.exists():
        return {"node_id": node_id, "status": "no_code", "error": "code.py missing"}

    manifest = _load_json(ara_root / "manifest.json") or {}
    default_cwd = resolve_portable_path(manifest.get("source_exp_dir"), base=ara_root)
    if cwd is not None:
        resolved_cwd = Path(cwd).expanduser().resolve()
        if not resolved_cwd.is_dir():
            raise ValueError("cwd must resolve to an existing directory")
    else:
        resolved_cwd = default_cwd
    if resolved_cwd is None or not resolved_cwd.is_dir():
        resolved_cwd = node_dir

    python_bin = python or sys.executable
    if recorded_metric is _READ_RECORDED_METRIC or protected_input_hashes is None:
        local_metrics, local_hashes = _snapshot_reexec_inputs(ara_root, [node_id])
        if recorded_metric is _READ_RECORDED_METRIC:
            recorded_metric = local_metrics[node_id]
        if protected_input_hashes is None:
            protected_input_hashes = local_hashes
    started = _now_iso()
    execution_error = None
    try:
        proc = subprocess.run(
            [python_bin, str(code_path)],
            cwd=str(resolved_cwd),
            capture_output=True,
            timeout=timeout,
        )
        timed_out = False
        stdout = _subprocess_text(proc.stdout)
        stderr = _subprocess_text(proc.stderr)
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _subprocess_text(exc.stdout)
        stderr = _subprocess_text(exc.stderr)
        returncode = -1
    except OSError as exc:
        timed_out = False
        stdout = ""
        stderr = f"failed to start subprocess: {exc}"
        returncode = -1
        execution_error = exc.__class__.__name__

    # stderr is diagnostic-only. Accepting metric markers from it lets an error
    # stream override the experiment's actual stdout result.
    fresh = parse_metric_from_stdout(stdout)
    comparison = compare_metrics(recorded_metric, fresh, tolerance)
    changed_inputs = [
        path.relative_to(ara_root).as_posix()
        for path, before in protected_input_hashes.items()
        if _file_sha256(path) != before
    ]

    verdict = {
        "schema": "ara.reexec.v1",
        "node_id": node_id,
        "ara_root": ".",
        "python": Path(python_bin).name,
        "cwd": relative_path_reference(resolved_cwd, base=ara_root),
        "started_at": started,
        "finished_at": _now_iso(),
        "returncode": returncode,
        "timed_out": timed_out,
        "execution_error": execution_error,
        "recorded_metric": recorded_metric,
        "fresh_metric": fresh,
        "comparison": comparison,
        "input_integrity": {
            "status": "clean" if not changed_inputs else "mutated",
            "changed_paths": changed_inputs,
        },
        "stdout_tail": redact_sensitive_text(stdout[-4000:]),
        "stderr_tail": redact_sensitive_text(stderr[-4000:]),
    }
    return redact_sensitive_payload(verdict)


def persist_reexec_verdict(
    ara_root: str | Path,
    verdict: dict[str, Any],
) -> dict[str, Any]:
    """Externalise output tails and return a compact, addressable verdict.

    Older reports inlined both tails in every JSON file.  New reports retain
    the same scalar/comparison fields but point at deduplicated CAS objects.
    The compact verdict itself is also content-addressed so batch reports can
    reference it without copying the complete payload.
    """

    root = Path(ara_root).expanduser().resolve()
    store = ObjectStore(root)
    compact = redact_sensitive_payload(dict(verdict))
    stdout_tail = str(compact.pop("stdout_tail", "") or "")
    stderr_tail = str(compact.pop("stderr_tail", "") or "")
    if stdout_tail:
        compact["stdout_ref"] = store.put_text(stdout_tail).to_json()
    if stderr_tail:
        compact["stderr_ref"] = store.put_text(stderr_tail).to_json()
    verdict_ref = store.put_json(compact)
    return {**compact, "verdict_ref": verdict_ref.to_json()}


def reexec_verdict_compared(verdict: Any) -> bool:
    """Return whether one re-execution completed with a comparable metric."""
    if not isinstance(verdict, dict):
        return False
    comparison = verdict.get("comparison")
    return (
        verdict.get("returncode") == 0
        and verdict.get("timed_out") is False
        and (verdict.get("input_integrity") or {}).get("status") == "clean"
        and isinstance(comparison, dict)
        and comparison.get("status") == "compared"
    )


def reexec_verdict_passed(verdict: Any) -> bool:
    """Return whether one re-execution completed and matched its metric."""
    return reexec_verdict_compared(verdict) and (
        verdict["comparison"].get("within_tolerance") is True
    )


def _write_reexec_report(ara_root: Path, verdicts: list[dict[str, Any]]) -> Path:
    verify_dir = ara_root / "verify"
    passed = sum(1 for verdict in verdicts if reexec_verdict_passed(verdict))
    compact_verdicts = [
        persist_reexec_verdict(ara_root, verdict) for verdict in verdicts
    ]
    payload = {
        "schema": "ara.reexec.batch.v1",
        "generated_at": _now_iso(),
        "ara_root": ".",
        "verdict_count": len(verdicts),
        "passed": passed,
        "verdicts": compact_verdicts,
    }
    return write_unique_json_report(
        verify_dir,
        "reexec_batch",
        redact_sensitive_payload(payload),
    )


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
    if limit is None:
        raise ValueError("limit must be a positive integer")
    validate_reexec_options(timeout=timeout, tolerance=tolerance, limit=limit)
    ara_root = Path(ara_root).expanduser().resolve()
    if node_ids is None:
        picks = _select_candidate_nodes(
            ara_root, limit=limit, include_buggy=include_buggy
        )
    else:
        picks = list(node_ids)
    if not picks:
        return {"status": "no_candidates", "ara_root": "."}

    # Validate the complete batch before compiling context or starting any node.
    for node_id in picks:
        resolve_node_paths(ara_root, node_id)
        require_indexed_node(ara_root, node_id)

    recorded_metrics, protected_input_hashes = _snapshot_reexec_inputs(ara_root, picks)
    verdicts: list[dict[str, Any]] = []
    for nid in picks:
        context_pack_ref = None
        context_error = None
        try:
            from ai_scientist.utils.ara_context import (
                compile_context_pack,
                persist_context_pack,
            )

            context_pack = compile_context_pack(
                ara_root,
                intent="reproduce",
                node_id=nid,
                budget_tokens=4000,
            )
            context_pack_ref = persist_context_pack(
                ara_root,
                context_pack,
                consumer="reproduce_agent",
            )
            if context_pack.get("blockers"):
                context_error = "; ".join(
                    str(item) for item in context_pack["blockers"]
                )
        except Exception as exc:  # Legacy/minimal ARAs can still execute.
            context_error = redact_sensitive_text(str(exc))
        verdict = reexec_node(
            ara_root,
            nid,
            python=python,
            timeout=timeout,
            tolerance=tolerance,
            recorded_metric=recorded_metrics[nid],
            protected_input_hashes=protected_input_hashes,
        )
        verdict["context_pack_ref"] = context_pack_ref
        verdict["context_error"] = context_error
        verdicts.append(verdict)

    report_path = _write_reexec_report(ara_root, verdicts)
    try:
        from ai_scientist.utils.ara_catalog import rebuild_semantic_catalog

        rebuild_semantic_catalog(ara_root)
    except Exception:
        pass
    return {
        "status": "ok",
        "ara_root": ".",
        "report_path": report_path.relative_to(ara_root).as_posix(),
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
