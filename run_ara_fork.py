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
    """CLI-side shim — the real implementation lives in
    ``ai_scientist.utils.ara_metric_parser``.
    """
    from ai_scientist.utils.ara_metric_parser import parse_metric_from_stdout
    return parse_metric_from_stdout(stdout)


def _compare_metrics(recorded: Any, fresh: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """CLI-side shim — see ``ai_scientist.utils.ara_metric_parser``."""
    from ai_scientist.utils.ara_metric_parser import compare_metrics
    return compare_metrics(recorded, fresh, tolerance)


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
    """Fork a node into a *new, valid ARA* rooted at ``--dest``.

    Design (see ai_scientist/protocol/SPEC.md §7.1): the fork target is
    itself a conformant ARA — one node, its own manifest, its own
    single-entry exploration_graph, and provenance pointing back at the
    parent's ``content_hash``. This means:

      1. `validate_ara(dest)` passes (unlike the older 'copy parent + node/'
         layout that used to fail conformance).
      2. `run_ara_fork.py fork` can be called on the *fork* to make a
         grand-fork — the provenance chain is recursive without special code.
      3. Downstream tools that only speak ARA don't need to learn a second
         "fork layout".
    """
    from ai_scientist.protocol import PROTOCOL_VERSION, hash_node_payload

    ara_root = _resolve_ara_root(args.ara)
    meta, node_dir = _load_node(ara_root, args.node_id)
    dest = Path(args.dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"Destination {dest} not empty (pass --force to overwrite)", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Copy the node bundle into <dest>/nodes/<node_id>/ (canonical ARA layout).
    dest_nodes_dir = dest / "nodes" / args.node_id
    shutil.copytree(node_dir, dest_nodes_dir, dirs_exist_ok=True)

    # 2. Compute / re-use content hash. If the parent had one, keep it; else
    #    reconstruct from code+metric so the fork is still content-addressable.
    parent_metrics = _load_json(node_dir / "metrics.json") or {}
    parent_hash = parent_metrics.get("content_hash") if isinstance(parent_metrics, dict) else None
    if not parent_hash:
        code = (node_dir / "code.py").read_text(encoding="utf-8") if (node_dir / "code.py").exists() else ""
        recorded_metric = parent_metrics.get("metric") if isinstance(parent_metrics, dict) else None
        try:
            parent_hash = hash_node_payload(code=code, metric=recorded_metric)
        except Exception:  # pragma: no cover - defensive
            parent_hash = None

    parent_manifest = _load_json(ara_root / "manifest.json") or {}
    parent_graph = _load_json(ara_root / "exploration_graph.json") or {}
    parent_node_entry: dict[str, Any] = {}
    for node in parent_graph.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id")) == args.node_id:
            parent_node_entry = node
            break

    # 3. Emit a single-node exploration_graph for the fork.
    now_iso = _now_iso()
    graph_payload = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_kind": "exploration_graph",
        "generated_at": now_iso,
        "nodes": [
            {
                "id": args.node_id,
                "content_hash": parent_hash,
                "stage": parent_node_entry.get("stage") or "forked",
                "step": 0,
                "parent_id": None,
                "children": [],
                "is_buggy": parent_node_entry.get("is_buggy"),
                "is_seed_node": True,
                "is_seed_agg_node": False,
                "metric": parent_node_entry.get("metric") or parent_metrics.get("metric"),
                "plan_excerpt": (parent_node_entry.get("plan_excerpt") or "")[:400],
                "exp_results_dir": None,
                "ctime": parent_node_entry.get("ctime"),
                "artifacts_dir": f"nodes/{args.node_id}",
            }
        ],
        "edges": [],
        "source_journals": [],
        "counts": {"nodes": 1, "edges": 0, "buggy": 1 if parent_node_entry.get("is_buggy") else 0},
    }
    (dest / "exploration_graph.json").write_text(
        json.dumps(graph_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 4. Emit the fork's own manifest — a fully conformant ARA manifest.
    provenance = {
        "parent_ara_root": str(ara_root),
        "parent_node_id": args.node_id,
        "parent_content_hash": parent_hash,
    }
    parent_idea = parent_manifest.get("idea") or {}
    fork_manifest = {
        "schema_version": PROTOCOL_VERSION,
        "protocol_kind": "manifest",
        "created_at": now_iso,
        "source_exp_dir": str(node_dir),  # informational only
        "project_dir": str(dest.parent),
        "idea": {
            "name": f"fork_of_{parent_idea.get('name') or 'unknown'}_at_{args.node_id}",
            "title": parent_idea.get("title"),
            "raw": {"forked_from": provenance},
        },
        "counts": {"nodes": 1, "edges": 0, "buggy_nodes": 0, "journals": 0, "claims": 0},
        "references": {},
        "missing": [
            "no source journals (fork is a synthetic single-node ARA)",
            "no env/ snapshot (call `run_ara_fork.py freeze` after seeding)",
        ],
        "provenance": provenance,
    }
    (dest / "manifest.json").write_text(
        json.dumps(fork_manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 5. Small compat file — `ara_seed.py` still looks for `fork.json` to
    #    disambiguate fork dirs from arbitrary ARAs. Keep the schema tag but
    #    make its payload minimal (real data lives in `manifest.provenance`).
    fork_meta = {
        "schema": "ara.fork.v1",
        "created_at": now_iso,
        "source_ara": str(ara_root),
        "source_node_id": args.node_id,
        "source_content_hash": parent_hash,
        "provenance_hint": provenance,
        "notes": (
            "This directory is itself a conformant ARA (see manifest.json). "
            "`fork.json` is a legacy marker so `ara_seed.py` can tell forks "
            "apart from other ARAs. All authoritative data lives in "
            "`manifest.json` / `exploration_graph.json`."
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


def cmd_validate(args: argparse.Namespace) -> int:
    """Run ARA conformance validation and print a report."""
    # Local import so callers who never validate don't pay the cost.
    from ai_scientist.protocol import validate_ara

    ara_root = _resolve_ara_root(args.ara)
    report = validate_ara(ara_root, strict=args.strict)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.ok else 8


def cmd_verify(args: argparse.Namespace) -> int:
    """Batch re-execution: pick a handful of nodes and diff fresh vs recorded metrics.

    Delegates to ``ai_scientist.utils.ara_reexec.reexec_ara`` — the CLI's job
    is to expose the knobs and translate exit codes.
    """
    from ai_scientist.utils.ara_reexec import reexec_ara

    ara_root = _resolve_ara_root(args.ara)
    node_ids = args.node_ids if args.node_ids else None
    summary = reexec_ara(
        ara_root,
        node_ids=node_ids,
        limit=args.limit,
        include_buggy=args.include_buggy,
        python=args.python,
        timeout=args.timeout,
        tolerance=args.tolerance,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("status") != "ok":
        return 9

    # Non-zero exit when *no* verdict landed within tolerance — makes CI usage
    # trivial. Callers who don't care about drift can pass --allow-drift.
    report_path = summary.get("report_path")
    if report_path:
        report = _load_json(Path(report_path)) or {}
        verdicts = report.get("verdicts") or []
        within = sum(
            1
            for v in verdicts
            if isinstance(v, dict)
            and (v.get("comparison") or {}).get("within_tolerance") is True
        )
        if within == 0 and verdicts and not args.allow_drift:
            return 10
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

    validate_p = sub.add_parser("validate", help="Check an ARA against the protocol schema.")
    validate_p.add_argument("--ara", required=True)
    validate_p.add_argument("--strict", action="store_true", help="Promote warnings to errors")
    validate_p.set_defaults(func=cmd_validate)

    verify_p = sub.add_parser(
        "verify",
        help="Batch re-execute selected nodes and diff fresh vs recorded metrics.",
    )
    verify_p.add_argument("--ara", required=True)
    verify_p.add_argument(
        "--node-ids", nargs="*", default=None,
        help="Specific node ids to verify. Omit for the top-metric picks.",
    )
    verify_p.add_argument("--limit", type=int, default=3, help="Max nodes to pick automatically")
    verify_p.add_argument("--include-buggy", action="store_true")
    verify_p.add_argument("--python", default=None)
    verify_p.add_argument("--timeout", type=int, default=900)
    verify_p.add_argument("--tolerance", type=float, default=1e-3)
    verify_p.add_argument(
        "--allow-drift", action="store_true",
        help="Exit zero even when no fresh metric matches the recorded one within tolerance.",
    )
    verify_p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
