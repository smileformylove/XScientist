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
- ``diff``     — structural diff of two ARAs: manifest hashes, node set
                 delta, per-node hash-change categorisation, pipeline
                 artifacts, prompt overlap.
- ``log``      — commit-log view: manifest revision chain (from
                 manifest.history.jsonl) plus provenance ancestry walk.
- ``refs``     — git-style local refs stored under ``<ara>/refs/``.

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
    parent_code = (node_dir / "code.py").read_text(encoding="utf-8") if (node_dir / "code.py").exists() else ""
    parent_metric = parent_metrics.get("metric") if isinstance(parent_metrics, dict) else None
    if not parent_hash:
        try:
            parent_hash = hash_node_payload(code=parent_code, metric=parent_metric)
        except Exception:  # pragma: no cover - defensive
            parent_hash = None

    # The fork node carries is_seed_node=True, so its OWN content_hash must be
    # computed with is_seed=True to agree with the declared role. The parent's
    # hash is still recorded in provenance (source_content_hash /
    # parent_content_hash) as a REFERENCE to the parent — those stay pointing
    # at the parent's original hash.
    try:
        fork_hash = hash_node_payload(code=parent_code, metric=parent_metric, is_seed=True)
    except Exception:  # pragma: no cover - defensive
        fork_hash = parent_hash

    # Rewrite the copied metrics.json so its content_hash matches the fork's
    # declared seed role (the copytree above left the parent's hash in place).
    # We must also refresh content_hash_inputs — the fork_hash was computed
    # with is_seed=True, so the declared inputs need to advertise 'seed' or
    # anyone re-hashing per those inputs gets a different digest (drift).
    if isinstance(parent_metrics, dict) and fork_hash:
        fork_metrics = dict(parent_metrics)
        fork_metrics["content_hash"] = fork_hash
        fork_metrics["content_hash_inputs"] = ["code", "metric", "seed"]
        (dest_nodes_dir / "metrics.json").write_text(
            json.dumps(fork_metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

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
                "content_hash": fork_hash,
                # Match the metrics.json declaration above so ara_diff's
                # category-flip logic can index the fork's inputs, and so the
                # fork's node entry matches the schema shape enforced by
                # export_ara. SEED nodes typically bypass the LLM entirely,
                # so llm_call_refs defaults to [] — the parent's refs are
                # part of the *parent's* identity, not the fork's. If a
                # future fork variant wants to inherit refs, it should add
                # 'llm_calls' back into content_hash_inputs at the same time.
                "content_hash_inputs": ["code", "metric", "seed"],
                "llm_call_refs": [],
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
        "counts": {
            "nodes": 1,
            "edges": 0,
            "buggy_nodes": 1 if parent_node_entry.get("is_buggy") else 0,
            "journals": 0,
            "claims": 0,
        },
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

    # Anchor the fork's manifest under the immutability layer (SPEC §7.2) —
    # forks are commit-like, so they need the same lock export_ara writes.
    from ai_scientist.utils.ara_manifest_lock import write_manifest_lock
    write_manifest_lock(dest, fork_manifest)

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


# ---------------------------------------------------------------------------
# diff / log / refs — added in Phase 2 (git-style verbs over ARAs).
#
# These commands are all read-only w.r.t. the source ARAs (refs writes to
# <ara>/refs/, which is caller-local and not part of the content-addressed
# state of the ARA). The heavy lifting lives in ai_scientist.utils.ara_diff,
# ara_log, ara_refs — this file is thin CLI glue.
# ---------------------------------------------------------------------------


def cmd_diff(args: argparse.Namespace) -> int:
    from ai_scientist.utils.ara_diff import diff_ara

    ara_a = _resolve_ara_root(args.ara)
    ara_b = _resolve_ara_root(args.other)
    result = diff_ara(ara_a, ara_b)
    lists = ("nodes_added", "nodes_removed", "nodes_hash_changed")

    # --only-node narrows the per-node lists; manifest/references stay aggregate.
    if args.only_node:
        matched = False
        for name in lists:
            kept = [n for n in getattr(result, name) if n.id == args.only_node]
            matched = matched or bool(kept)
            setattr(result, name, kept)
        if not matched:
            print(f"node {args.only_node} not present in either side", file=sys.stderr)
            return 3

    # Snapshot pre-truncation counts so the summary line stays honest.
    summary_counts = {k: len(getattr(result, n))
                      for k, n in zip(("added", "removed", "changed"), lists)}
    truncated = {"added": 0, "removed": 0, "changed": 0}
    if args.limit_nodes is not None and args.limit_nodes >= 0:
        n = args.limit_nodes
        for key, name in zip(("added", "removed", "changed"), lists):
            lst = getattr(result, name)
            if len(lst) > n:
                truncated[key] = len(lst) - n
                setattr(result, name, lst[:n])

    if args.json:
        payload = result.to_dict()
        if any(truncated.values()):
            payload["truncated"] = truncated
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    _render_diff(result, summary_counts=summary_counts, truncated=truncated)
    # Non-zero when there's any material difference — makes it usable in CI.
    if result.manifest.hash_equal and not (
        summary_counts["added"] or summary_counts["removed"] or summary_counts["changed"]
        or result.references.seed_changed
        or result.references.pipeline_added
        or result.references.pipeline_removed
        or result.references.pipeline_hash_changed
    ):
        return 0
    return 1 if args.exit_code_on_diff else 0


def cmd_log(args: argparse.Namespace) -> int:
    from ai_scientist.utils.ara_log import ara_log

    ara_root = _resolve_ara_root(args.ara)
    log = ara_log(ara_root)
    if args.json:
        print(json.dumps(log.to_dict(), indent=2, ensure_ascii=False, default=str))
        return 0
    _render_log(log)
    return 0


def cmd_verify_lock(args: argparse.Namespace) -> int:
    """Check whether manifest.json still matches its manifest.lock chain.

    This is the headline promise of the immutability layer
    (:mod:`ai_scientist.utils.ara_manifest_lock`): downstream agents who
    trust the ARA's top-level pointer can prove it hasn't been silently
    edited outside the append-only revision API. Exit codes are shaped so
    CI can gate on tampering: ``rc=0`` for ``clean``/``revised``,
    ``rc=2`` for ``tampered`` (real integrity breach), ``rc=3`` for
    ``unlocked`` (missing lock — can't judge). The report itself is
    always printed so the caller can diagnose.
    """
    from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock

    ara_root = _resolve_ara_root(args.ara)
    report = verify_manifest_lock(ara_root)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"# verify-lock  {ara_root}")
        print(f"  state:          {report.get('state')}")
        print(f"  base_hash:      {report.get('base_hash')}")
        print(f"  current_hash:   {report.get('current_hash')}")
        print(f"  revision_count: {report.get('revision_count')}")
        print(f"  detail:         {report.get('detail')}")

    state = report.get("state")
    if state in ("clean", "revised"):
        return 0
    if state == "tampered":
        return 2
    return 3  # unlocked / anything else — mirror `refs --get` missing semantics


def cmd_history(args: argparse.Namespace) -> int:
    """Render the manifest revision chain (lock + manifest.history.jsonl).

    Row 0 is synthesised from ``manifest.lock`` — the immutability
    anchor written at export time. Subsequent rows are streamed straight
    from ``manifest.history.jsonl`` in order. Exit codes mirror
    ``verify-lock``: rc=0 when the lock exists (even with zero
    revisions), rc=3 when it doesn't.
    """
    from ai_scientist.utils.ara_manifest_lock import (
        MANIFEST_HISTORY_NAME, MANIFEST_LOCK_NAME,
    )

    ara_root = _resolve_ara_root(args.ara)
    lock_path = ara_root / MANIFEST_LOCK_NAME
    if not lock_path.exists():
        print(f"[ara-history] {ara_root} is unlocked (no {MANIFEST_LOCK_NAME})",
              file=sys.stderr)
        return 3

    lock = _load_json(lock_path) or {}
    history_path = ara_root / MANIFEST_HISTORY_NAME
    revisions: list[dict[str, Any]] = []
    if history_path.exists():
        try:
            for line in history_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    revisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            revisions = []

    if args.limit is not None and args.limit >= 0:
        revisions = revisions[-args.limit:] if args.limit else []

    rows: list[dict[str, Any]] = [{
        "revision": 0,
        "ts": lock.get("created_at"),
        "base_hash": None,
        "new_hash": lock.get("manifest_hash"),
        "producer": None,
        "reason": "(initial export)",
        "changed_fields": [],
    }]
    for r in revisions:
        rows.append({
            "revision": r.get("revision"),
            "ts": r.get("ts"),
            "base_hash": r.get("base_hash"),
            "new_hash": r.get("new_hash"),
            "producer": r.get("producer"),
            "reason": r.get("reason"),
            "changed_fields": r.get("changed_fields") or [],
        })

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
        return 0

    _render_history(rows)
    return 0


def cmd_refs(args: argparse.Namespace) -> int:
    from ai_scientist.utils.ara_refs import (
        RefError, delete_ref, get_ref, list_refs, set_ref,
    )

    ara_root = _resolve_ara_root(args.ara)

    if args.set:
        try:
            path = set_ref(ara_root, args.set[0], args.set[1])
        except RefError as exc:
            print(f"[ara-refs] refused: {exc}", file=sys.stderr)
            return 2
        print(f"[ara-refs] set {args.set[0]} -> {args.set[1]}  ({path})")
        return 0

    if args.get:
        target = get_ref(ara_root, args.get)
        if target is None:
            print(f"[ara-refs] {args.get} not set", file=sys.stderr)
            return 3
        print(target)
        return 0

    if args.delete:
        removed = delete_ref(ara_root, args.delete)
        if not removed:
            print(f"[ara-refs] {args.delete} not set", file=sys.stderr)
            return 3
        print(f"[ara-refs] deleted {args.delete}")
        return 0

    refs = list_refs(ara_root)
    if args.json:
        print(json.dumps([{"name": r.name, "target": r.target} for r in refs],
                         indent=2, ensure_ascii=False))
        return 0
    if not refs:
        print("(no refs set)")
        return 0
    width = max(len(r.name) for r in refs)
    for r in refs:
        print(f"{r.name.ljust(width)}  {r.target}")
    return 0


# ---------------------------------------------------------------------------
# Rendering helpers (kept in this file so the CLI stays visually consistent).
# ---------------------------------------------------------------------------


def _render_diff(
    result,
    *,
    summary_counts: dict[str, int] | None = None,
    truncated: dict[str, int] | None = None,
) -> None:  # ARADiff
    # Fall back to counts from result when caller didn't pre-snapshot them.
    counts = summary_counts or {
        "added": len(result.nodes_added),
        "removed": len(result.nodes_removed),
        "changed": len(result.nodes_hash_changed),
    }
    trunc = truncated or {"added": 0, "removed": 0, "changed": 0}
    footer = "  ... {n} more (use --only-node <id> to inspect)"

    print(f"# diff  {result.ara_a}  <->  {result.ara_b}")
    print()
    m = result.manifest
    print("## manifest")
    if m.hash_equal:
        print(f"  hashes match: {m.hash_a}")
    else:
        print(f"  a: {m.hash_a}")
        print(f"  b: {m.hash_b}")
        for field_name, delta in m.field_changes.items():
            print(f"  - {field_name}")
            print(f"      a: {delta['a']}")
            print(f"      b: {delta['b']}")

    r = result.references
    if (r.seed_changed or r.pipeline_added or r.pipeline_removed
            or r.pipeline_hash_changed):
        print()
        print("## references")
        if r.seed_changed:
            print(f"  seed: {r.seed_hash_a}  ->  {r.seed_hash_b}")
        for kind in r.pipeline_added:
            print(f"  + pipeline/{kind}")
        for kind in r.pipeline_removed:
            print(f"  - pipeline/{kind}")
        for entry in r.pipeline_hash_changed:
            print(f"  ~ pipeline/{entry['kind']}: {entry['hash_a']}  ->  {entry['hash_b']}")

    print()
    print(f"## nodes  (+{counts['added']} "
          f"-{counts['removed']} "
          f"~{counts['changed']} "
          f"={result.nodes_unchanged})")
    for n in result.nodes_added:
        print(f"  + {n.id}  {n.hash_b}")
    if trunc["added"]:
        print(footer.format(n=trunc["added"]))
    for n in result.nodes_removed:
        print(f"  - {n.id}  {n.hash_a}")
    if trunc["removed"]:
        print(footer.format(n=trunc["removed"]))
    for n in result.nodes_hash_changed:
        cats = ",".join(n.changed_categories) or "unknown"
        print(f"  ~ {n.id}  [{cats}]  {n.hash_a}  ->  {n.hash_b}")
    if trunc["changed"]:
        print(footer.format(n=trunc["changed"]))

    p = result.prompts
    print()
    print(f"## prompts  a={p.total_a} b={p.total_b} "
          f"shared={p.shared} only_a={p.only_in_a} only_b={p.only_in_b}")


def _render_log(log) -> None:  # ARALog
    print(f"# log  {log.ara_root}")
    print(f"  verify: {log.verify.get('state')}  "
          f"({log.verify.get('revision_count')} revisions)")
    print()
    print("## revisions")
    if log.lock:
        print(f"  rev 0 (lock)  {log.lock.get('manifest_hash')}  "
              f"{log.lock.get('created_at')}")
    else:
        print("  (no manifest.lock — this ARA predates the immutability layer)")
    for r in log.revisions:
        fields = ",".join(r.changed_fields) or "-"
        reason = f"  # {r.reason}" if r.reason else ""
        print(f"  rev {r.revision}       {r.new_hash}  "
              f"{r.ts}  [{fields}] {r.producer or ''}{reason}")

    print()
    print("## ancestry")
    if not log.ancestors:
        print("  (root ARA — no provenance)")
        return
    for a in log.ancestors:
        marker = "reachable" if a.reachable else "unreachable"
        verify = ""
        if a.hash_verified is True:
            verify = "  ✓ hash matches"
        elif a.hash_verified is False:
            verify = "  ✗ hash mismatch"
        print(f"  ^{a.depth}  {a.ara_root or '(no path)'}  "
              f"node={a.node_id}  hash={a.content_hash}  ({marker}){verify}")
        if a.seed_hash:
            print(f"       seed_hash={a.seed_hash}")
        if a.detail:
            print(f"       note: {a.detail}")


def _short_hash(h: str | None) -> str:
    if not h:
        return "-"
    # Full form is sha256:<64 hex>. Keep the prefix + first 16 hex + ellipsis.
    if ":" in h:
        prefix, digest = h.split(":", 1)
        return f"{prefix}:{digest[:16]}…" if len(digest) > 16 else h
    return f"{h[:16]}…" if len(h) > 16 else h


def _render_history(rows: list[dict[str, Any]]) -> None:
    header = ("rev", "ts", "base_hash", "new_hash", "producer", "reason")
    widths = {
        "rev": max(3, max(len(str(r["revision"])) for r in rows)),
        "ts": 20,
        "base_hash": 24,
        "new_hash": 24,
        "producer": 30,
    }
    line = (
        f"{header[0]:<{widths['rev']}}  "
        f"{header[1]:<{widths['ts']}}  "
        f"{header[2]:<{widths['base_hash']}}  "
        f"{header[3]:<{widths['new_hash']}}  "
        f"{header[4]:<{widths['producer']}}  "
        f"{header[5]}"
    )
    print(line)
    for r in rows:
        base = "(base)" if r["revision"] == 0 else _short_hash(r["base_hash"])
        print(
            f"{r['revision']:<{widths['rev']}}  "
            f"{(r['ts'] or '-'):<{widths['ts']}}  "
            f"{base:<{widths['base_hash']}}  "
            f"{_short_hash(r['new_hash']):<{widths['new_hash']}}  "
            f"{(r['producer'] or '-'):<{widths['producer']}}  "
            f"{r['reason'] or '-'}"
        )


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

    diff_p = sub.add_parser(
        "diff",
        help="Structural diff of two ARAs (manifest, nodes, prompts, pipeline).",
    )
    diff_p.add_argument("--ara", required=True, help="First ARA (a)")
    diff_p.add_argument("--other", required=True, help="Second ARA (b)")
    diff_p.add_argument("--json", action="store_true",
                        help="Machine-readable output")
    diff_p.add_argument(
        "--exit-code-on-diff", action="store_true",
        help="Exit non-zero when any material difference is found (useful in CI)",
    )
    diff_p.add_argument(
        "--only-node", metavar="ID", default=None,
        help="Filter node lists to entries with this id (exits rc=3 if absent).",
    )
    diff_p.add_argument(
        "--limit-nodes", type=int, default=None, metavar="N",
        help="Truncate each of added/removed/changed to N entries. Summary "
             "counts remain accurate; JSON output gains a `truncated` field.",
    )
    diff_p.set_defaults(func=cmd_diff)

    log_p = sub.add_parser(
        "log",
        help="Commit-log view: manifest revisions + provenance ancestry.",
    )
    log_p.add_argument("--ara", required=True)
    log_p.add_argument("--json", action="store_true")
    log_p.set_defaults(func=cmd_log)

    verify_lock_p = sub.add_parser(
        "verify-lock",
        help="Check that manifest.json still matches manifest.lock (immutability audit).",
    )
    verify_lock_p.add_argument("--ara", required=True)
    verify_lock_p.add_argument("--json", action="store_true",
                               help="Emit the raw report dict as JSON.")
    verify_lock_p.set_defaults(func=cmd_verify_lock)

    history_p = sub.add_parser(
        "history",
        help="Render the manifest revision chain (lock + manifest.history.jsonl).",
    )
    history_p.add_argument("--ara", required=True)
    history_p.add_argument("--json", action="store_true",
                           help="Emit rows as a JSON array (full hashes preserved).")
    history_p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Show only the last N revisions (row 0 / base is always shown).",
    )
    history_p.set_defaults(func=cmd_history)

    refs_p = sub.add_parser(
        "refs",
        help="git-style local refs stored under <ara>/refs/.",
    )
    refs_p.add_argument("--ara", required=True)
    refs_p.add_argument("--set", nargs=2, metavar=("NAME", "TARGET"),
                        help="Create or update a ref pointing at TARGET (a content hash).")
    refs_p.add_argument("--get", metavar="NAME",
                        help="Print a ref's target and exit.")
    refs_p.add_argument("--delete", metavar="NAME",
                        help="Remove a ref.")
    refs_p.add_argument("--json", action="store_true",
                        help="Machine-readable listing.")
    refs_p.set_defaults(func=cmd_refs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
