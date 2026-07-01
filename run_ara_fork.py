"""CLI: fork or re-execute a node from an ARA (Agent-Native Research Artifact).

Why this exists
---------------
The whole point of ARA (see `ai_scientist/utils/ara_artifact.py`) is that a
downstream agent — human or AI — can pick any node from an exploration graph
and continue from there without decoding the PDF. This script is the concrete
"fork" and "verify" verbs.

Sub-commands
------------
- ``inspect``  — pretty-print a node's summary (metrics, plots, buggy flag).
- ``exec``     — re-execute the node's ``code.py`` in a subprocess and emit a
                 verify report comparing the fresh metric to the recorded one.
- ``fork``     — copy the node's bundle into a fresh directory that a new
                 tree search can seed from. Optionally snapshot the current
                 interpreter's package list into ``env/requirements.freeze``.
- ``freeze``   — snapshot ``pip freeze`` into the ARA's ``env/`` directory.

None of these commands mutate the source ARA — they only read from it. Fork
outputs go to a caller-supplied ``--dest``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_ara_root(raw: str) -> Path:
    """Accept either the ARA directory or its manifest.json."""
    path = Path(raw).expanduser().resolve()
    if path.is_file() and path.name == "manifest.json":
        return path.parent
    if path.is_dir() and (path / "manifest.json").exists():
        return path
    raise SystemExit(f"ARA not found or missing manifest.json at: {path}")


def _load_node(ara_root: Path, node_id: str) -> tuple[dict[str, Any], Path]:
    node_dir = ara_root / "nodes" / node_id
    if not node_dir.exists():
        raise SystemExit(f"Node {node_id} not found under {ara_root / 'nodes'}")
    graph = _load_json(ara_root / "exploration_graph.json") or {}
    meta: dict[str, Any] = {}
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id")) == node_id:
            meta = node
            break
    return meta, node_dir


def _parse_metric_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract a machine-readable metric from a re-executed run.

    We try two strategies, in order:
      1. Any line matching ``ARA_METRIC={"name": "...", "value": 0.42}`` (JSON
         after the equals sign). This is the format we recommend downstream
         agents emit inside their code.
      2. A trailing line of the form ``metric: <float>`` (case-insensitive).

    If neither matches, we return ``{"available": False}`` so verify can
    downgrade gracefully to a soft comparison.
    """
    marker_re = re.compile(r"ARA_METRIC\s*=\s*(\{.*\})\s*$", re.MULTILINE)
    match = None
    for candidate in marker_re.finditer(stdout):
        match = candidate  # keep the last one — freshest wins
    if match:
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict) and "value" in payload:
                return {"available": True, **payload}
        except json.JSONDecodeError:
            pass

    tail_re = re.compile(r"^\s*metric\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", re.MULTILINE)
    tail = None
    for candidate in tail_re.finditer(stdout):
        tail = candidate
    if tail:
        try:
            return {"available": True, "value": float(tail.group(1)), "source": "text_tail"}
        except ValueError:
            pass

    return {"available": False}


def _compare_metrics(recorded: Any, fresh: dict[str, Any], tolerance: float) -> dict[str, Any]:
    if not fresh.get("available"):
        return {"status": "no_metric_parsed", "delta": None, "within_tolerance": None}
    recorded_value = None
    if isinstance(recorded, dict):
        recorded_value = recorded.get("value")
    elif isinstance(recorded, (int, float)):
        recorded_value = recorded
    fresh_value = fresh.get("value")
    if recorded_value is None or fresh_value is None:
        return {"status": "missing_recorded_value", "delta": None, "within_tolerance": None}
    try:
        delta = float(fresh_value) - float(recorded_value)
    except (TypeError, ValueError):
        return {"status": "non_numeric_metric", "delta": None, "within_tolerance": None}
    return {
        "status": "compared",
        "recorded_value": recorded_value,
        "fresh_value": fresh_value,
        "delta": delta,
        "within_tolerance": abs(delta) <= tolerance,
    }


def _write_verify_report(ara_root: Path, node_id: str, report: dict[str, Any]) -> Path:
    verify_dir = ara_root / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9T]", "", _now_iso())[:14] or "run"
    path = verify_dir / f"{node_id}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Sub-commands
# ----------------------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    ara_root = _resolve_ara_root(args.ara)
    meta, node_dir = _load_node(ara_root, args.node_id)
    metrics = _load_json(node_dir / "metrics.json") or {}
    print(f"# Node {args.node_id}")
    print(f"ARA root:      {ara_root}")
    print(f"Node dir:      {node_dir}")
    print(f"Step:          {meta.get('step')}")
    print(f"Stage:         {meta.get('stage')}")
    print(f"Buggy:         {meta.get('is_buggy')}")
    print(f"Parent:        {meta.get('parent_id')}")
    print(f"Metric:        {metrics.get('metric')}")
    print(f"Analysis:      {(metrics.get('analysis') or '').strip()[:400]}")
    if (node_dir / "code.py").exists():
        print(f"Code size:     {(node_dir / 'code.py').stat().st_size} bytes")
    if (node_dir / "term_out.log").exists():
        print(f"Term out size: {(node_dir / 'term_out.log').stat().st_size} bytes")
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    ara_root = _resolve_ara_root(args.ara)
    meta, node_dir = _load_node(ara_root, args.node_id)
    code_path = node_dir / "code.py"
    if not code_path.exists():
        print(f"Node {args.node_id} has no code.py — nothing to execute.", file=sys.stderr)
        return 3

    manifest = _load_json(ara_root / "manifest.json") or {}
    default_cwd = Path(manifest.get("source_exp_dir") or "").expanduser()
    cwd = Path(args.cwd).expanduser() if args.cwd else default_cwd
    if not cwd.exists():
        cwd = node_dir  # last-resort fallback so the subprocess starts somewhere valid

    python_bin = args.python or sys.executable
    print(f"[ara-fork] exec node={args.node_id} cwd={cwd} python={python_bin}")

    started_at = _now_iso()
    try:
        completed = subprocess.run(
            [python_bin, str(code_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = None
        timed_out = True
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        returncode = -1

    if not timed_out:
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        returncode = completed.returncode

    fresh_metric = _parse_metric_from_stdout(stdout_text + "\n" + stderr_text)
    metrics_recorded = (_load_json(node_dir / "metrics.json") or {}).get("metric")
    comparison = _compare_metrics(metrics_recorded, fresh_metric, args.tolerance)

    report = {
        "schema": "ara.verify.v1",
        "node_id": args.node_id,
        "ara_root": str(ara_root),
        "started_at": started_at,
        "finished_at": _now_iso(),
        "python": python_bin,
        "cwd": str(cwd),
        "timeout_seconds": args.timeout,
        "returncode": returncode,
        "timed_out": timed_out,
        "recorded_metric": metrics_recorded,
        "fresh_metric": fresh_metric,
        "comparison": comparison,
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
    }
    report_path = _write_verify_report(ara_root, args.node_id, report)
    print(f"[ara-fork] verify report: {report_path}")
    print(f"[ara-fork] returncode={returncode} timed_out={timed_out} comparison={comparison}")
    if timed_out:
        return 4
    if returncode != 0 and not args.allow_nonzero:
        return 5
    if comparison.get("within_tolerance") is False and not args.allow_metric_drift:
        return 6
    return 0


def cmd_fork(args: argparse.Namespace) -> int:
    ara_root = _resolve_ara_root(args.ara)
    _, node_dir = _load_node(ara_root, args.node_id)
    dest = Path(args.dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"Destination {dest} not empty (pass --force to overwrite)", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copytree(node_dir, dest / "node", dirs_exist_ok=True)

    for name in ("manifest.json", "exploration_graph.json"):
        src = ara_root / name
        if src.exists():
            shutil.copy2(src, dest / name)

    fork_meta = {
        "schema": "ara.fork.v1",
        "created_at": _now_iso(),
        "source_ara": str(ara_root),
        "source_node_id": args.node_id,
        "notes": (
            "Seed a fresh tree search from `node/code.py`. `manifest.json` and "
            "`exploration_graph.json` are the origin context — keep them as a "
            "read-only lineage record."
        ),
    }
    (dest / "fork.json").write_text(
        json.dumps(fork_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ara-fork] forked node {args.node_id} into {dest}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    ara_root = _resolve_ara_root(args.ara)
    env_dir = ara_root / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    dest = env_dir / "requirements.freeze"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"pip freeze failed: {exc}", file=sys.stderr)
        return 7
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        return completed.returncode
    dest.write_text(completed.stdout, encoding="utf-8")
    print(f"[ara-fork] wrote {dest} ({len(completed.stdout.splitlines())} packages)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fork or re-execute a node from an Agent-Native Research Artifact.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_p = sub.add_parser("inspect", help="Print a node's recorded metadata.")
    inspect_p.add_argument("--ara", required=True, help="Path to the ARA directory or manifest.json")
    inspect_p.add_argument("--node-id", required=True)
    inspect_p.set_defaults(func=cmd_inspect)

    exec_p = sub.add_parser("exec", help="Re-execute a node and compare metrics.")
    exec_p.add_argument("--ara", required=True)
    exec_p.add_argument("--node-id", required=True)
    exec_p.add_argument("--python", default=None, help="Python interpreter (default: current)")
    exec_p.add_argument("--cwd", default=None, help="Override the working directory")
    exec_p.add_argument("--timeout", type=int, default=1800)
    exec_p.add_argument("--tolerance", type=float, default=1e-3)
    exec_p.add_argument("--allow-nonzero", action="store_true", help="Don't fail on non-zero exit")
    exec_p.add_argument(
        "--allow-metric-drift", action="store_true",
        help="Don't fail when the fresh metric drifts beyond tolerance",
    )
    exec_p.set_defaults(func=cmd_exec)

    fork_p = sub.add_parser("fork", help="Copy a node bundle to a new directory to seed further work.")
    fork_p.add_argument("--ara", required=True)
    fork_p.add_argument("--node-id", required=True)
    fork_p.add_argument("--dest", required=True)
    fork_p.add_argument("--force", action="store_true")
    fork_p.set_defaults(func=cmd_fork)

    freeze_p = sub.add_parser("freeze", help="Snapshot the current interpreter's pip freeze into env/.")
    freeze_p.add_argument("--ara", required=True)
    freeze_p.set_defaults(func=cmd_freeze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
