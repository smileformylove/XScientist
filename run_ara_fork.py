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
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
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


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce ``value`` to int; return ``default`` on TypeError/ValueError.

    Manifests can be hand-edited or produced by older codepaths where
    ``counts.nodes`` / ``counts.edges`` may end up as strings or other
    non-numeric junk. ``int("foo")`` would crash describe/list; this
    helper keeps the CLI resilient without silently masking real bugs
    (a wrong-but-numeric value still round-trips faithfully).
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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

    # --stat is a whole-ARA one-liner (or single JSON object). It intentionally
    # bypasses --only-node / --limit-nodes — those slice per-node output that
    # --stat doesn't emit. Bailing out here also means the pre-truncation
    # counts we display are the raw diff counts, exactly what a CI dashboard
    # cell wants to render.
    if args.stat:
        return _emit_diff_stat(result, args)

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


def _emit_diff_stat(result, args: argparse.Namespace) -> int:  # ARADiff
    """One-line (or one JSON object) whole-ARA summary for scripts / CI.

    Text grammar (all lowercase, single spaces):
      ``nodes: +A -R ~C prompts: shared=S only_a=X only_b=Y \
seed_ref_changed=(yes|no) pipeline_changed=N``

    Exit code mirrors the long-form diff: rc=1 when ``--exit-code-on-diff``
    is set AND any node/prompt/reference movement is present; else rc=0.
    """
    added = len(result.nodes_added)
    removed = len(result.nodes_removed)
    changed = len(result.nodes_hash_changed)
    r = result.references
    pipeline_changed = (
        len(r.pipeline_added) + len(r.pipeline_removed) + len(r.pipeline_hash_changed)
    )
    seed_ref_changed = bool(r.seed_changed)
    p = result.prompts

    if args.json:
        print(json.dumps({
            "added": added, "removed": removed, "changed": changed,
            "prompts_shared": p.shared,
            "prompts_only_a": p.only_in_a, "prompts_only_b": p.only_in_b,
            "seed_ref_changed": seed_ref_changed,
            "pipeline_changed": pipeline_changed,
        }, ensure_ascii=False))
    else:
        print(
            f"nodes: +{added} -{removed} ~{changed} "
            f"prompts: shared={p.shared} only_a={p.only_in_a} only_b={p.only_in_b} "
            f"seed_ref_changed={'yes' if seed_ref_changed else 'no'} "
            f"pipeline_changed={pipeline_changed}"
        )

    # Manifest-only drift (timestamps etc.) is intentionally NOT surfaced by
    # the one-liner, so it must not trip --exit-code-on-diff either.
    material = bool(added or removed or changed or seed_ref_changed or pipeline_changed)
    return 1 if args.exit_code_on_diff and material else 0


def cmd_log(args: argparse.Namespace) -> int:
    from ai_scientist.utils.ara_log import ara_log, walk_node_ancestry

    ara_root = _resolve_ara_root(args.ara)

    # --node narrows to a single node's in-ARA ancestry (leaf → root).
    # This is the "git log for THIS node" verb, complementing
    # `diff --only-node` and `inspect --node-id`.
    if args.node:
        try:
            chain = walk_node_ancestry(ara_root, args.node)
        except KeyError:
            print(f"node {args.node} not present in exploration_graph.json",
                  file=sys.stderr)
            return 3
        if args.json:
            print(json.dumps(chain, indent=2, ensure_ascii=False, default=str))
            return 0
        _render_node_ancestry(ara_root, args.node, chain)
        return 0

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

    ``--all --project <path>`` sweeps every ARA under ``<path>/ara/`` and
    aggregates state — tampered wins over unlocked in the exit code.
    """
    from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock

    if args.project:
        # --project without --all is rejected loudly so sweep intent stays explicit.
        if not args.all:
            print("[verify-lock] --project requires --all", file=sys.stderr)
            return 2
        return _verify_lock_all(Path(args.project).expanduser().resolve(), args)

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


def _verify_lock_all(project_dir: Path, args: argparse.Namespace) -> int:
    """Sweep <project_dir>/ara/ and aggregate lock state.

    rc rule: tampered (2) > unlocked (3) > pass (0). Empty projects pass.
    """
    from ai_scientist.utils.ara_artifact import ara_root_for_project
    from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock

    ara_base = ara_root_for_project(str(project_dir))
    entries: list[dict[str, Any]] = []
    if ara_base.exists():
        for sub in sorted(ara_base.iterdir()):
            if sub.is_dir() and (sub / "manifest.json").exists():
                r = verify_manifest_lock(sub)
                entries.append({
                    "ara_root": str(sub),
                    "state": r.get("state"),
                    "revision_count": r.get("revision_count"),
                    "manifest_hash": r.get("base_hash"),
                    "detail": r.get("detail"),
                })
    if not entries:
        if args.json:
            print("[]")
        print(f"(no ARAs found under {ara_base})", file=sys.stderr)
        return 0
    if args.json:
        print(json.dumps(entries, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"{'STATE':<10} {'REVS':<5} {'HASH_PREFIX':<24} PATH")
        for e in entries:
            locked = e["state"] in ("clean", "revised")
            revs = str(e["revision_count"]) if locked else "-"
            h = _short_hash(e["manifest_hash"]) if locked else "-"
            print(f"{e['state'] or '?':<10} {revs:<5} {h:<24} {Path(e['ara_root']).name}")
    if any(e["state"] == "tampered" for e in entries):
        return 2
    if any(e["state"] == "unlocked" for e in entries):
        return 3
    return 0


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


def cmd_show(args: argparse.Namespace) -> int:
    """Dump a single node's full metadata as JSON to stdout.

    Machine-readable complement to ``inspect`` — every field a
    downstream script might want (hash inputs, llm refs, code, tail of
    term_out) in one shot, so pipelines don't have to xargs six files.

    Exit codes mirror ``log --node`` / ``diff --only-node`` / ``refs
    --get``: rc=3 when the requested node id is not present.
    """
    ara_root = _resolve_ara_root(args.ara)
    graph = _load_json(ara_root / "exploration_graph.json") or {}
    meta: dict[str, Any] = {}
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id")) == args.node:
            meta = node
            break
    node_dir = ara_root / "nodes" / args.node
    if not meta and not node_dir.exists():
        print(f"[ara-show] node {args.node} not found in {ara_root}",
              file=sys.stderr)
        return 3

    metrics = _load_json(node_dir / "metrics.json") or {}
    plots = _load_json(node_dir / "plots.json") or {}

    code_path = node_dir / "code.py"
    code_text: str | None = None
    if code_path.exists():
        try:
            code_text = code_path.read_text(encoding="utf-8")
        except OSError:
            code_text = None

    term_path = node_dir / "term_out.log"
    term_tail: str | None = None
    term_size = 0
    if term_path.exists():
        try:
            term_size = term_path.stat().st_size
            raw = term_path.read_bytes()
            tail = raw if args.term_tail is None else raw[-args.term_tail:] if args.term_tail > 0 else b""
            term_tail = tail.decode("utf-8", errors="replace")
        except OSError:
            term_tail = None
            term_size = 0

    payload = {
        "id": args.node,
        "content_hash": meta.get("content_hash") or metrics.get("content_hash"),
        "content_hash_inputs": (
            meta.get("content_hash_inputs")
            or metrics.get("content_hash_inputs")
            or []
        ),
        "llm_call_refs": meta.get("llm_call_refs") or [],
        "is_buggy": meta.get("is_buggy"),
        "is_seed_node": meta.get("is_seed_node"),
        "step": meta.get("step"),
        "parent_id": meta.get("parent_id"),
        "children": meta.get("children") or [],
        "metric": metrics.get("metric") if metrics.get("metric") is not None else meta.get("metric"),
        "analysis": metrics.get("analysis"),
        "exec_time": metrics.get("exec_time"),
        "exc_type": metrics.get("exc_type"),
        "code": code_text,
        "term_out_tail": term_tail,
        "term_out_size": term_size,
        "plots": plots.get("plots") or plots.get("plot_paths") or [],
        "plots_generated": plots.get("plots_generated"),
        "vlm_feedback_summary": plots.get("vlm_feedback_summary") or [],
    }
    if getattr(args, "terse", False):
        # Shape-only view — drop the two large text blobs while keeping every
        # other field (including term_out_size) so callers can still see how
        # big term_out was without paging through the bytes.
        payload.pop("code", None)
        payload.pop("term_out_tail", None)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_hash_check(args: argparse.Namespace) -> int:
    """Node-level integrity check: recompute each node's content_hash from disk.

    Complements ``verify-lock`` (manifest-level) with per-node coverage — the
    manifest lock protects the pointer to the graph, but individual node
    payloads (``nodes/<id>/code.py`` and ``metrics.json``) live outside the
    hash chain. This verb rehashes each node from its on-disk inputs using
    the exact same binding rule ``export_ara`` used at write time
    (``ai_scientist/utils/ara_artifact.py::_export_nodes_from_journal``) and
    reports drift.

    States:
      * ``clean``         — stored hash matches recompute.
      * ``drift``         — stored hash present but differs from recompute
                            (tampering / silent edit signal).
      * ``missing_code``  — stored hash present but ``code.py`` absent
                            (data loss — cannot recompute).
      * ``unhashed``      — no stored hash on the graph entry (legacy /
                            partial export; not a failure).

    Exit codes (drift beats missing_code so tampering wins in CI):
      * rc=0 all clean (unhashed nodes allowed)
      * rc=1 any drift
      * rc=2 any missing_code (and no drift)

    ``--all --project <path>`` sweeps every ARA under ``<path>/ara/`` and
    aggregates counts + state per ARA — drift beats missing_code in the rc.
    """
    if args.project:
        # --project without --all is rejected loudly so sweep intent stays explicit.
        if not args.all:
            print("[hash-check] --project requires --all", file=sys.stderr)
            return 2
        return _hash_check_all(Path(args.project).expanduser().resolve(), args)

    ara_root = _resolve_ara_root(args.ara)
    entries = _hash_check_ara(ara_root)
    if args.json:
        print(json.dumps(entries, indent=2, ensure_ascii=False, default=str))
    else:
        _render_hash_check(entries)
    if any(e["state"] == "drift" for e in entries):
        return 1
    if any(e["state"] == "missing_code" for e in entries):
        return 2
    return 0


def _hash_check_ara(ara_root: Path) -> list[dict[str, Any]]:
    """Recompute each node's content_hash from disk and return per-node states.

    Shared between single-ARA ``hash-check --ara`` and the ``--all --project``
    sweep — keeps the classification rules in exactly one place.
    """
    from ai_scientist.protocol import hash_node_payload

    graph = _load_json(ara_root / "exploration_graph.json") or {}
    entries: list[dict[str, Any]] = []

    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue

        stored_hash = node.get("content_hash")
        node_dir = ara_root / "nodes" / node_id
        code_path = node_dir / "code.py"
        metrics = _load_json(node_dir / "metrics.json") or {}

        # Mirror _export_nodes_from_journal's binding logic exactly.
        llm_refs_raw = node.get("llm_call_refs") or []
        llm_refs = [str(r) for r in llm_refs_raw if r]
        is_seed = bool(node.get("is_seed_node"))
        metric = metrics.get("metric") if isinstance(metrics, dict) else None

        # Export skips writing code.py when the journal node's code is
        # empty/whitespace, but still stamps a content_hash computed with
        # ``code=""``. So an absent code.py is only data loss if the stored
        # hash does NOT match the empty-code recompute; otherwise it's a
        # legitimate empty-code node and we treat it as clean.
        code_missing = not code_path.exists()
        code_text = ""
        if not code_missing:
            try:
                code_text = code_path.read_text(encoding="utf-8")
            except OSError as exc:
                entries.append({
                    "node_id": node_id, "state": "missing_code",
                    "stored_hash": stored_hash, "computed_hash": None,
                    "notes": f"code.py unreadable: {exc}",
                })
                continue

        try:
            computed = hash_node_payload(
                code=code_text,
                metric=metric,
                llm_call_hashes=llm_refs or None,
                is_seed=is_seed,
            )
        except Exception as exc:  # pragma: no cover - defensive
            entries.append({
                "node_id": node_id, "state": "drift",
                "stored_hash": stored_hash, "computed_hash": None,
                "notes": f"hash recompute failed: {exc}",
            })
            continue

        if not stored_hash:
            # No stored hash to compare against. Preserve the legacy
            # ``no_code`` label for the code-absent case so the caller can
            # distinguish "legacy graph entry" from "legacy AND code gone".
            state = "no_code" if code_missing else "unhashed"
            note = "code.py absent (no stored hash either)" if code_missing else None
            entry = {
                "node_id": node_id, "state": state,
                "stored_hash": None,
                "computed_hash": None if code_missing else computed,
            }
            if note:
                entry["notes"] = note
            entries.append(entry)
            continue

        if stored_hash == computed:
            entries.append({
                "node_id": node_id, "state": "clean",
                "stored_hash": stored_hash, "computed_hash": computed,
            })
            continue

        # Recompute differs from stored. If code.py was absent and the
        # empty-code recompute did NOT match, the stored hash implies
        # non-empty code that's now gone from disk — genuine data loss.
        if code_missing:
            entries.append({
                "node_id": node_id, "state": "missing_code",
                "stored_hash": stored_hash, "computed_hash": None,
                "notes": "code.py absent; stored hash implies non-empty code",
            })
            continue

        # Drift — try to explain which category flipped. The graph entry
        # carries an independent copy of the pre-export metric alongside
        # metrics.json's copy (see _export_nodes_from_journal — both are
        # written from the same `raw.get("metric")`). If they've drifted
        # apart, someone edited metrics.json (the primary source we hashed
        # from). Otherwise the drift is on the code side — since metric
        # matches its export-time twin, the only remaining hash input that
        # can have changed is code (llm_call_refs / is_seed_node live on the
        # graph entry and were used verbatim in the recompute).
        graph_metric = node.get("metric")
        if metric != graph_metric:
            note = "metric differs"
        else:
            note = "code differs"
        entries.append({
            "node_id": node_id, "state": "drift",
            "stored_hash": stored_hash, "computed_hash": computed,
            "notes": note,
        })

    return entries


def _render_hash_check(entries: list[dict[str, Any]]) -> None:
    header = ("NODE", "STATE", "STORED", "COMPUTED", "NOTES")
    rows: list[tuple[str, str, str, str, str]] = []
    for e in entries:
        rows.append((
            e["node_id"],
            e["state"],
            _short_hash(e.get("stored_hash")),
            _short_hash(e.get("computed_hash")),
            e.get("notes") or "",
        ))
    if not rows:
        print("(no nodes to check)")
        return
    widths = [max(len(header[i]), max(len(r[i]) for r in rows))
              for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))


def _hash_check_all(project_dir: Path, args: argparse.Namespace) -> int:
    """Sweep <project_dir>/ara/ and aggregate per-node hash-check state.

    rc rule mirrors single-ARA hash-check: drift (1) beats missing_code (2).
    Empty projects pass with a stderr note (never silent — iter-13 invariant).
    """
    from ai_scientist.utils.ara_artifact import ara_root_for_project

    ara_base = ara_root_for_project(str(project_dir))
    aras: list[dict[str, Any]] = []
    if ara_base.exists():
        for sub in sorted(ara_base.iterdir()):
            if not (sub.is_dir() and (sub / "manifest.json").exists()):
                continue
            entries = _hash_check_ara(sub)
            counts = {"clean": 0, "drift": 0, "missing_code": 0,
                      "unhashed": 0, "no_code": 0}
            for e in entries:
                counts[e["state"]] = counts.get(e["state"], 0) + 1
            state = ("drift" if counts["drift"] else
                     "missing_code" if counts["missing_code"] else "clean")
            aras.append({"ara_root": str(sub), "nodes": len(entries),
                         "counts": counts, "state": state})
    totals = {
        "aras": len(aras),
        "nodes": sum(a["nodes"] for a in aras),
        "drift": sum(a["counts"]["drift"] for a in aras),
        "missing_code": sum(a["counts"]["missing_code"] for a in aras),
    }
    if not aras:
        if args.json:
            print(json.dumps({"aras": [], "totals": totals}, indent=2))
        print(f"(no ARAs found under {ara_base})", file=sys.stderr)
        return 0
    if args.json:
        print(json.dumps({"aras": aras, "totals": totals},
                         indent=2, ensure_ascii=False, default=str))
    else:
        header = ("ARA", "NODES", "CLEAN", "DRIFT", "MISSING", "UNHASHED", "STATE")
        rows = [(Path(a["ara_root"]).name, str(a["nodes"]),
                 str(a["counts"]["clean"]), str(a["counts"]["drift"]),
                 str(a["counts"]["missing_code"]),
                 str(a["counts"]["unhashed"] + a["counts"]["no_code"]),
                 a["state"]) for a in aras]
        widths = [max(len(header[i]), max(len(r[i]) for r in rows))
                  for i in range(len(header))]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        for r in rows:
            print(fmt.format(*r))
        print(f"Total: {totals['aras']} ARAs, {totals['nodes']} nodes, "
              f"{totals['drift']} drift, {totals['missing_code']} missing_code")
    if totals["drift"]:
        return 1
    if totals["missing_code"]:
        return 2
    return 0


def _pick_top_metric_node(nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best-scored non-buggy node.

    Respects each node's ``metric.maximize`` flag. Nodes with ``is_buggy=True``
    or without a scalar ``metric.value`` are skipped. Returns None when nothing
    qualifies.
    """
    best: dict[str, Any] | None = None
    best_value: float | None = None
    best_maximize: bool = True
    for node in nodes or []:
        if not isinstance(node, dict) or node.get("is_buggy"):
            continue
        metric = node.get("metric")
        if not isinstance(metric, dict):
            continue
        value = metric.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        maximize = bool(metric.get("maximize", True))
        # Compare against the current best using its own maximize direction —
        # a node with maximize=false beats the best iff its value is lower.
        if best is None or (
            float(value) > best_value if best_maximize else float(value) < best_value
        ):
            best, best_value, best_maximize = node, float(value), maximize
    return best


def cmd_describe(args: argparse.Namespace) -> int:
    """Emit a one-shot overview of an ARA.

    Fills the "at-a-glance" gap between ``inspect`` (one node) and
    ``log`` / ``history`` (chains). Prints a compact human-readable
    block by default; ``--json`` returns a machine-readable dict.
    """
    from ai_scientist.protocol import hash_manifest
    from ai_scientist.utils.ara_log import ara_log
    from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock

    ara_root = _resolve_ara_root(args.ara)
    manifest = _load_json(ara_root / "manifest.json")
    if not isinstance(manifest, dict):
        print(f"[ara-describe] {ara_root} has no readable manifest.json",
              file=sys.stderr)
        return 3
    graph = _load_json(ara_root / "exploration_graph.json") or {}
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    counts = manifest.get("counts") or {}
    if not isinstance(counts, dict):
        counts = {}
    buggy = _safe_int(
        counts.get("buggy_nodes"),
        default=sum(1 for n in nodes if n.get("is_buggy")),
    )
    node_count = _safe_int(counts.get("nodes"), default=len(nodes))
    edge_count = _safe_int(counts.get("edges"), default=len(graph.get("edges") or []))
    seed_count = sum(1 for n in nodes if n.get("is_seed_node"))

    top = _pick_top_metric_node(nodes)
    top_payload = None if top is None else {
        "id": str(top.get("id")),
        "content_hash": top.get("content_hash"),
        "metric": top.get("metric"),
        "is_buggy": bool(top.get("is_buggy")),
    }

    verify = verify_manifest_lock(ara_root)
    lock_payload = {
        "manifest_hash": verify.get("base_hash"),
        "revision": int(verify.get("revision_count") or 0),
        "state": verify.get("state"),
        "current_hash": verify.get("current_hash"),
    }
    try:
        manifest_hash_current = hash_manifest(manifest)
    except Exception:  # pragma: no cover - defensive
        manifest_hash_current = None

    provenance = manifest.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    seed_hash = provenance.get("seed_hash")
    seed_payload = None
    if seed_hash or provenance.get("parent_ara_root"):
        seed_payload = {
            "hash": seed_hash,
            "parent_ara_root": provenance.get("parent_ara_root"),
            "parent_node_id": provenance.get("parent_node_id"),
            "parent_content_hash": provenance.get("parent_content_hash"),
        }

    log = ara_log(ara_root)
    # all_verified: treat None (couldn't check) as pass — same convention as
    # cmd_log's render — so an unverified but reachable ancestor doesn't get
    # flagged as a failure.
    ancestors_payload = {
        "count": len(log.ancestors),
        "all_reachable": all(a.reachable for a in log.ancestors) if log.ancestors else True,
        "all_verified": all(a.hash_verified is not False for a in log.ancestors) if log.ancestors else True,
    }
    idea = manifest.get("idea") or {}
    if not isinstance(idea, dict):
        idea = {}
    idea_payload = {"name": idea.get("name"), "title": idea.get("title")}

    if args.json:
        print(json.dumps({
            "ara_root": str(ara_root),
            "idea": idea_payload,
            "counts": {
                "nodes": node_count, "buggy": buggy, "seeds": seed_count,
                "edges": edge_count, "revisions": lock_payload["revision"],
            },
            "top_metric_node": top_payload,
            "seed": seed_payload,
            "lock": lock_payload,
            "verify_state": verify.get("state"),
            "manifest_hash_current": manifest_hash_current,
            "ancestors": ancestors_payload,
        }, indent=2, ensure_ascii=False, default=str))
        return 0

    # Human view.
    print(f"# ARA overview: {ara_root}")
    print(f"Idea:            {idea_payload['name'] or '(unknown)'}")
    if idea_payload.get("title"):
        print(f"Title:           {idea_payload['title']}")
    print(f"Nodes:           {node_count}  (buggy: {buggy}, seeds: {seed_count}, edges: {edge_count})")
    if top_payload is None:
        print("Top metric:      (no scored nodes)")
    else:
        m = top_payload["metric"] or {}
        name = m.get("name") if isinstance(m, dict) else None
        value = m.get("value") if isinstance(m, dict) else m
        maximize = m.get("maximize") if isinstance(m, dict) else None
        suffix = f"  (name={name}, maximize={maximize})" if name is not None else ""
        print(f"Top metric:      {top_payload['id']}  metric={value}{suffix}")
    if seed_payload:
        parent_ara = seed_payload.get("parent_ara_root") or "(unknown)"
        parent_node = seed_payload.get("parent_node_id") or "(unknown)"
        print(f"Seed:            {_short_hash(seed_payload.get('hash'))}  (parent: {parent_ara} / {parent_node})")
    else:
        print("Seed:            (none — root ARA)")
    print(f"Latest lock:     {_short_hash(lock_payload['manifest_hash'])}  (rev {lock_payload['revision']}, {lock_payload['state']})")
    print(f"Manifest hash:   {_short_hash(manifest_hash_current)}")
    print(f"Verify state:    {verify.get('state')}")
    if ancestors_payload["count"] == 0:
        print("Ancestors:       0 (root ARA — no provenance)")
    else:
        reach = "all reachable" if ancestors_payload["all_reachable"] else "some unreachable"
        verify_word = "hash verified" if ancestors_payload["all_verified"] else "hash mismatch"
        print(f"Ancestors:       {ancestors_payload['count']} ({reach}, {verify_word})")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Enumerate every ARA under ``<project>/ara/`` with a one-line summary.

    Complements ``verify-lock --all --project <path>`` (lock state only)
    with a richer per-ARA overview: idea, node counts, seed presence, and
    lock state. See ``describe`` for the deeper single-ARA view (this
    intentionally omits the top-metric field to stay one line per ARA).

    Exit rule: always rc=0 on success — this is a list, not a checker.
    Empty projects print a stderr note and emit ``[]`` (JSON) or nothing
    (human), mirroring ``verify-lock --all --project`` (iter-13 fix).
    Broken manifests appear with idea=``?`` and state=``error`` rather
    than failing the whole sweep.
    """
    from ai_scientist.utils.ara_artifact import ara_root_for_project
    from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock

    project_dir = Path(args.project).expanduser().resolve()
    ara_base = ara_root_for_project(str(project_dir))
    entries: list[dict[str, Any]] = []
    if ara_base.exists():
        for sub in sorted(ara_base.iterdir()):
            if not sub.is_dir() or not (sub / "manifest.json").exists():
                continue  # not an ARA — skip _scratch/ / __pycache__/ / .hidden/ / etc.
            manifest = _load_json(sub / "manifest.json")
            if not isinstance(manifest, dict):
                entries.append({
                    "ara_root": str(sub), "idea": "?",
                    "nodes": None, "buggy_nodes": None, "seed_present": None,
                    "state": "error", "manifest_hash": None, "path": sub.name,
                })
                continue
            counts = manifest.get("counts") or {}
            if not isinstance(counts, dict):
                counts = {}
            provenance = manifest.get("provenance") or {}
            if not isinstance(provenance, dict):
                provenance = {}
            # Match cmd_describe's convention: parent_ara_root also counts
            # as a seed reference (forks set that instead of seed_hash).
            seed_present = bool(
                provenance.get("seed_hash") or provenance.get("parent_ara_root")
            )
            idea = manifest.get("idea") or {}
            if not isinstance(idea, dict):
                idea = {}
            r = verify_manifest_lock(sub)
            entries.append({
                "ara_root": str(sub),
                "idea": idea.get("name") or "?",
                "nodes": counts.get("nodes"),
                "buggy_nodes": counts.get("buggy_nodes"),
                "seed_present": seed_present,
                "state": r.get("state"),
                "manifest_hash": r.get("base_hash"),
                "path": sub.name,
            })
    if not entries:
        if args.json:
            print("[]")
        print(f"(no ARAs found under {ara_base})", file=sys.stderr)
        return 0
    if args.json:
        print(json.dumps(entries, indent=2, ensure_ascii=False, default=str))
        return 0
    _render_list(entries)
    return 0


def _render_list(entries: list[dict[str, Any]]) -> None:
    header = ("IDEA", "NODES", "BUGGY", "SEED", "STATE", "LOCK", "PATH")

    def _fmt_count(v: Any) -> str:
        return "-" if v is None else str(v)

    def _fmt_seed(v: Any) -> str:
        return "-" if v is None else ("yes" if v else "no")

    def _fmt_lock(state: Any, h: Any) -> str:
        return _short_hash(h) if state in ("clean", "revised") else "-"

    rows = [
        (
            e["idea"] or "?", _fmt_count(e["nodes"]), _fmt_count(e["buggy_nodes"]),
            _fmt_seed(e["seed_present"]), e["state"] or "?",
            _fmt_lock(e["state"], e["manifest_hash"]), e["path"],
        )
        for e in entries
    ]
    widths = [max(len(header[i]), max(len(r[i]) for r in rows))
              for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))


def cmd_claims(args: argparse.Namespace) -> int:
    """Enumerate `\\claimref`-derived claim files under ``<ara>/claims/``.

    Fills the "what does this ARA actually assert?" gap between
    ``describe`` (whole-ARA) and ``show`` (one node). Each claim file
    written by :mod:`ai_scientist.utils.claim_registry` carries a link
    to an exploration-graph node; this verb joins that link so a
    downstream consumer can see, per claim, which node's hash/metric
    the assertion is anchored to.

    On-disk shape (per ``claim_registry.write_claims_into_ara``): one
    ``<ara>/claims/<claim_id>.json`` per marker, plus ``_index.json``
    and (optionally) ``coverage.json`` — the latter two are metadata,
    not claims, so they're excluded from enumeration.

    Exit codes mirror ``log --node`` / ``diff --only-node`` / ``refs
    --get``: rc=3 when ``--node`` filters to zero results. Empty
    claims dir (or absent) → rc=0 with a stderr note (matches the
    iter-13 empty-result invariant used by ``list`` and
    ``verify-lock --all``).
    """
    ara_root = _resolve_ara_root(args.ara)
    claims_dir = ara_root / "claims"

    # Build id -> graph node lookup once so we can annotate each claim
    # with the linked node's hash/metric/is_buggy without re-parsing the
    # graph per file. Missing graph = every claim's node fields are null.
    graph = _load_json(ara_root / "exploration_graph.json") or {}
    node_index: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes") or []:
        if isinstance(node, dict):
            nid = node.get("id")
            if isinstance(nid, str) and nid:
                node_index[nid] = node

    # Coverage report is whole-manuscript severity (see claim_coverage.py) —
    # not per-claim, but downstream consumers benefit from the tag alongside
    # each row, so we copy the single value onto every record when present.
    coverage = _load_json(claims_dir / "coverage.json") or {}
    severity = coverage.get("severity") if isinstance(coverage, dict) else None

    # `_index.json` and `coverage.json` are the registry's own bookkeeping;
    # only user-facing per-claim files (created by claim_registry) count.
    claim_files: list[Path] = []
    if claims_dir.is_dir():
        for p in sorted(claims_dir.glob("*.json")):
            if p.name in {"_index.json", "coverage.json"}:
                continue
            claim_files.append(p)

    if not claim_files:
        if args.json:
            print("[]")
        print("(no claims recorded)", file=sys.stderr)
        return 0

    rows: list[dict[str, Any]] = []
    for path in claim_files:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        claim_id = str(payload.get("claim_id") or path.stem)
        node_id = payload.get("node_id")
        node_id = str(node_id) if isinstance(node_id, str) and node_id else None
        # Prefer the graph as source of truth (its content_hash is the one
        # downstream tools reference) but fall back to the claim file's
        # embedded copy — write_claims_into_ara snapshots the node under
        # `node`, so an ARA whose graph has since been pruned still shows
        # something meaningful.
        node_meta = node_index.get(node_id) if node_id else None
        if node_meta is None and isinstance(payload.get("node"), dict):
            node_meta = payload["node"]
        if node_meta is not None:
            content_hash = node_meta.get("content_hash")
            metric = node_meta.get("metric")
            is_buggy = node_meta.get("is_buggy")
        else:
            content_hash = metric = is_buggy = None

        rows.append({
            "claim_id": claim_id,
            "node_id": node_id,
            "node_content_hash": content_hash,
            "node_metric": metric,
            "node_is_buggy": is_buggy,
            "assertion": payload.get("context") or "",
            "tex_file": payload.get("tex_file"),
            "line": payload.get("line"),
            "severity": severity,
        })

    if args.node:
        rows = [r for r in rows if r["node_id"] == args.node]
        if not rows:
            print(f"[ara-claims] no claims link to node {args.node}",
                  file=sys.stderr)
            return 3

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
        return 0

    _render_claims(rows)
    return 0


def _render_claims(rows: list[dict[str, Any]]) -> None:
    header = ("CLAIM_ID", "NODE", "HASH_PREFIX", "METRIC", "BUGGY", "ASSERTION")

    def _fmt_metric(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, dict):
            val = v.get("value")
            return "-" if val is None else str(val)
        return str(v)

    def _fmt_buggy(v: Any) -> str:
        if v is None:
            return "-"
        return "yes" if v else "no"

    def _fmt_assertion(s: str) -> str:
        s = " ".join((s or "").split())
        return s[:120] + "…" if len(s) > 120 else s

    body: list[tuple[str, str, str, str, str, str]] = []
    for r in rows:
        # An unresolved node_id (present on the claim but absent from the
        # graph) is functionally unlinked for the reader — show `-` so the
        # column matches the null hash/metric/buggy neighbours. JSON keeps
        # the raw reference so a caller can still cross-check.
        linked = r["node_content_hash"] is not None
        body.append((
            r["claim_id"],
            r["node_id"] if linked and r["node_id"] else "-",
            _short_hash(r["node_content_hash"]),
            _fmt_metric(r["node_metric"]),
            _fmt_buggy(r["node_is_buggy"]),
            _fmt_assertion(r["assertion"]),
        ))
    if not body:
        return
    widths = [max(len(header[i]), max(len(row[i]) for row in body))
              for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    for row in body:
        print(fmt.format(*row))


def cmd_provenance(args: argparse.Namespace) -> int:
    """Reverse-lookup: find every ARA under a project that references a hash.

    Complements the immutability/inspection verbs (verify-lock, list, hash-check,
    describe) with a "who has this hash?" query. Given a full ``sha256:<hex>``
    string, this sweeps ``<project>/ara/*/`` and reports every place the hash
    appears: as a manifest_hash (lock), a node content_hash, an entry in a
    node's llm_call_refs, the provenance seed_hash / parent_content_hash, the
    references.seed / references.pipeline_artifacts content_hash, or the target
    of any local ref under ``<ara>/refs/``.

    Always exits rc=0 — this is a lookup, not a check. Callers distinguish
    empty vs match by parsing output (``[]`` for --json).
    """
    from ai_scientist.utils.ara_artifact import ara_root_for_project
    from ai_scientist.utils.ara_refs import list_refs

    target = args.hash
    project_dir = Path(args.project).expanduser().resolve()
    ara_base = ara_root_for_project(str(project_dir))
    hits: list[dict[str, Any]] = []

    if ara_base.exists():
        for sub in sorted(ara_base.iterdir()):
            if not (sub.is_dir() and (sub / "manifest.json").exists()):
                continue
            hits.extend(_provenance_scan_ara(sub, target, list_refs))

    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False, default=str))
    if not hits:
        print(f"(no ARAs reference {target})", file=sys.stderr)
        return 0
    if not args.json:
        _render_provenance(hits)
    return 0


def _provenance_scan_ara(ara_root: Path, target: str, list_refs) -> list[dict[str, Any]]:
    """Scan one ARA and return every hit dict where ``target`` is referenced."""
    out: list[dict[str, Any]] = []

    lock = _load_json(ara_root / "manifest.lock") or {}
    if isinstance(lock, dict) and lock.get("manifest_hash") == target:
        out.append({"ara_root": str(ara_root), "kind": "manifest",
                    "detail": {"revision": lock.get("revision")}})

    manifest = _load_json(ara_root / "manifest.json") or {}
    provenance = manifest.get("provenance") if isinstance(manifest, dict) else None
    if isinstance(provenance, dict):
        for key in ("seed_hash", "parent_content_hash"):
            if provenance.get(key) == target:
                out.append({"ara_root": str(ara_root),
                            "kind": "seed" if key == "seed_hash" else "provenance",
                            "detail": {"field": key}})
        # Multi-parent shape: provenance.parents[] carries one entry per role
        # (code/env/data/...). Only the code-role parent is echoed into the
        # top-level slots by build_provenance, so we must also scan the array
        # to honor the verb contract "find every ARA that references a hash."
        parents = provenance.get("parents")
        if isinstance(parents, list):
            for i, p in enumerate(parents):
                if not isinstance(p, dict):
                    continue
                if p.get("parent_content_hash") == target:
                    out.append({"ara_root": str(ara_root), "kind": "provenance",
                                "detail": {"field": "parents[]", "index": i,
                                           "role": p.get("role"),
                                           "parent_node_id": p.get("parent_node_id")}})
    references = manifest.get("references") if isinstance(manifest, dict) else None
    if isinstance(references, dict):
        seed_ref = references.get("seed")
        if isinstance(seed_ref, dict) and seed_ref.get("content_hash") == target:
            out.append({"ara_root": str(ara_root), "kind": "seed",
                        "detail": {"field": "references.seed"}})
        for entry in references.get("pipeline_artifacts") or []:
            if isinstance(entry, dict) and entry.get("content_hash") == target:
                out.append({"ara_root": str(ara_root), "kind": "pipeline_artifact",
                            "detail": {"kind": entry.get("kind"),
                                       "path": entry.get("path")}})

    graph = _load_json(ara_root / "exploration_graph.json") or {}
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node.get("content_hash") == target:
            out.append({"ara_root": str(ara_root), "kind": "node",
                        "detail": {"node_id": node_id}})
        for ref in node.get("llm_call_refs") or []:
            if ref == target:
                out.append({"ara_root": str(ara_root), "kind": "llm_call",
                            "detail": {"node_id": node_id}})

    try:
        for r in list_refs(ara_root):
            if r.target == target:
                out.append({"ara_root": str(ara_root), "kind": "ref",
                            "detail": {"ref_name": r.name}})
    except Exception:  # pragma: no cover - defensive
        pass

    return out


def _render_provenance(hits: list[dict[str, Any]]) -> None:
    header = ("KIND", "DETAIL", "ARA")

    def _fmt_detail(d: dict[str, Any] | None) -> str:
        if not isinstance(d, dict) or not d:
            return "-"
        return ", ".join(f"{k}={v}" for k, v in d.items() if v is not None) or "-"

    rows = [(h["kind"], _fmt_detail(h.get("detail")),
             Path(h["ara_root"]).name) for h in hits]
    widths = [max(len(header[i]), max(len(r[i]) for r in rows))
              for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))


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
    if args.prefix is not None:
        refs = [r for r in refs if r.name.startswith(args.prefix)]
    if args.json:
        print(json.dumps([{"name": r.name, "target": r.target} for r in refs],
                         indent=2, ensure_ascii=False))
        return 0
    if not refs:
        # When a prefix filter is in effect, stay silent so callers can pipe
        # the output to xargs without a spurious "(no refs set)" line.
        if args.prefix is None:
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


def _render_node_ancestry(ara_root: Path, node_id: str, chain: list[dict[str, Any]]) -> None:
    print(f"# log --node {node_id}  {ara_root}")
    if not chain:
        print("  (empty ancestry)")
        return
    last_ix = len(chain) - 1
    for ix, entry in enumerate(chain):
        metric = entry.get("metric")
        # Metric may be a dict (canonical journal form) or a bare number —
        # collapse to a compact scalar for the log-style line; --json keeps full shape.
        if isinstance(metric, dict):
            metric_out = metric.get("value", metric)
        else:
            metric_out = metric
        print(
            f"* {entry['id']}  "
            f"{_short_hash(entry.get('content_hash'))}  "
            f"is_buggy={entry.get('is_buggy')}  "
            f"is_seed={entry.get('is_seed_node')}  "
            f"metric={metric_out}"
        )
        if entry.get("note"):
            print(f"    note: {entry['note']}")
        if ix != last_ix:
            print("|")


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


def _hash_file_bytes(path: Path) -> str | None:
    """sha256 of file bytes as ``sha256:<hex>``; None if unreadable."""
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def cmd_env(args: argparse.Namespace) -> int:
    """Dump a compact JSON summary of ``<ara>/env/*`` — the reproducibility
    fingerprint (bfts_config, model_fingerprint, requirements.freeze).

    Complements ``describe`` (whole-ARA), ``show`` (one node), and
    ``hash-check`` (per-node hash integrity) with a focused view on
    what a downstream agent needs to re-create the run environment.
    Always exits rc=0 — this is a dump, not a check.
    """
    ara_root = _resolve_ara_root(args.ara)
    env_dir = ara_root / "env"

    payload: dict[str, Any] = {
        "ara_root": str(ara_root),
        "env_dir": None,
        "bfts_config": {"present": False, "path": "env/bfts_config.yaml"},
        "model_fingerprint": {"present": False, "path": "env/model_fingerprint.json"},
        "requirements_freeze": {
            "present": False,
            "path": "env/requirements.freeze",
            "note": "run `run_ara_fork.py freeze` to snapshot",
        },
        "other_env_files": [],
    }

    if not env_dir.is_dir():
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    payload["env_dir"] = str(env_dir)
    known_names: set[str] = set()

    # bfts_config: prefer .yaml (per iter-1 SPEC / build_env_snapshot) but
    # fall back to .json for older/hand-authored exports.
    bfts_yaml = env_dir / "bfts_config.yaml"
    bfts_json = env_dir / "bfts_config.json"
    bfts_path = bfts_yaml if bfts_yaml.exists() else (bfts_json if bfts_json.exists() else None)
    if bfts_path is not None:
        try:
            size = bfts_path.stat().st_size
        except OSError:
            size = None
        payload["bfts_config"] = {
            "present": True,
            "path": f"env/{bfts_path.name}",
            "content_hash": _hash_file_bytes(bfts_path),
            "size_bytes": size,
        }
        known_names.add(bfts_path.name)

    # model_fingerprint.json — parsed to lift `models` + `writing_profile`.
    fp_path = env_dir / "model_fingerprint.json"
    if fp_path.exists():
        parsed = _load_json(fp_path) or {}
        spec = parsed.get("spec") if isinstance(parsed, dict) else None
        spec = spec if isinstance(spec, dict) else {}
        payload["model_fingerprint"] = {
            "present": True,
            "path": "env/model_fingerprint.json",
            "content_hash": _hash_file_bytes(fp_path),
            "fingerprint": parsed.get("fingerprint") if isinstance(parsed, dict) else None,
            "models": spec.get("models") or {},
            "writing_profile": spec.get("writing_profile"),
        }
        known_names.add("model_fingerprint.json")

    # requirements.freeze — presence + line_count when small (< 4 KiB).
    freeze_path = env_dir / "requirements.freeze"
    if freeze_path.exists():
        try:
            size = freeze_path.stat().st_size
        except OSError:
            size = None
        entry: dict[str, Any] = {
            "present": True,
            "path": "env/requirements.freeze",
            "content_hash": _hash_file_bytes(freeze_path),
            "size_bytes": size,
        }
        if size is not None and size < 4096:
            try:
                entry["line_count"] = sum(
                    1 for _ in freeze_path.read_text(encoding="utf-8").splitlines()
                )
            except OSError:
                pass
            except UnicodeDecodeError:
                entry["line_count"] = None
                entry["note"] = "file is not valid UTF-8"
        payload["requirements_freeze"] = entry
        known_names.add("requirements.freeze")

    # Anything else in env/ — listed relative to ara_root so the paths match
    # the shape used by the three known-file entries above.
    try:
        for child in sorted(env_dir.iterdir()):
            if child.is_file() and child.name not in known_names:
                payload["other_env_files"].append(f"env/{child.name}")
    except OSError:
        pass

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
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


def cmd_bundle(args: argparse.Namespace) -> int:
    """Package an ARA into a portable gzip tarball — the ``git bundle`` analog.

    Pre-flight: unless ``--no-verify`` is set, refuse when
    ``verify_manifest_lock`` reports the ARA is ``tampered`` or ``unlocked``
    — a bundled artifact must be provably intact end-to-end. On success the
    tarball is written atomically (``<dest>.tmp`` → ``os.replace``) so an
    interrupted run never leaves a half-written archive at ``dest``.
    """
    from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock

    ara_root = _resolve_ara_root(args.ara)
    dest = Path(args.dest).expanduser().resolve()

    # Refuse writing back into the ARA itself — that would recurse the tarball
    # into its own input on the next `bundle` call.
    try:
        dest.parent.resolve().relative_to(ara_root)
    except ValueError:
        pass
    else:
        print(f"[ara-bundle] refusing: dest {dest} is inside ARA {ara_root}",
              file=sys.stderr)
        return 2

    if not args.no_verify:
        report = verify_manifest_lock(ara_root)
        state = report.get("state")
        if state in ("tampered", "unlocked"):
            print(
                f"[ara-bundle] refusing: state={state}, "
                f"detail={report.get('detail')}. Pass --no-verify to override.",
                file=sys.stderr,
            )
            return 2

    if dest.exists() and not args.force:
        print(f"[ara-bundle] refusing: dest already exists at {dest} "
              f"(pass --force to overwrite)", file=sys.stderr)
        return 2

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    file_count = 0
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            # arcname = ara_root.name so extraction produces a folder, not
            # scattered files. dereference=False keeps symlinks as symlinks.
            tar.add(str(ara_root), arcname=ara_root.name, recursive=True)
            # Count members after the fact so the log matches the archive.
            file_count = len(tar.getmembers())
    except OSError as exc:
        # Clean up partial temp file so retries start clean.
        try:
            tmp.unlink()
        except OSError:
            pass
        print(f"[ara-bundle] tarball write failed: {exc}", file=sys.stderr)
        return 2

    os.replace(tmp, dest)
    try:
        size_bytes = dest.stat().st_size
    except OSError:
        size_bytes = 0
    print(f"[ara-bundle] wrote {dest} ({size_bytes} bytes, {file_count} files)")
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

    env_p = sub.add_parser(
        "env",
        help="Dump a JSON summary of env/ (bfts_config, model_fingerprint, requirements.freeze).",
    )
    env_p.add_argument("--ara", required=True)
    env_p.set_defaults(func=cmd_env)

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
    diff_p.add_argument(
        "--stat", action="store_true",
        help="Emit a one-line whole-ARA summary for scripts/CI dashboards "
             "instead of the multi-section report. With --json, emit a single "
             "JSON object of the same counts. --only-node and --limit-nodes "
             "are silently ignored (this flag is a whole-ARA summary).",
    )
    diff_p.set_defaults(func=cmd_diff)

    log_p = sub.add_parser(
        "log",
        help="Commit-log view: manifest revisions + provenance ancestry.",
    )
    log_p.add_argument("--ara", required=True)
    log_p.add_argument("--json", action="store_true")
    log_p.add_argument(
        "--node", metavar="ID", default=None,
        help="Focus on one node's in-ARA ancestry (parent_id chain, leaf → root). "
             "Exits rc=3 when the id isn't in exploration_graph.json.",
    )
    log_p.set_defaults(func=cmd_log)

    verify_lock_p = sub.add_parser(
        "verify-lock",
        help="Check that manifest.json still matches manifest.lock (immutability audit).",
    )
    # --ara (single) and --project (sweep) are mutually exclusive; argparse
    # itself emits a predictable SystemExit(2) when neither/both are given.
    verify_lock_target = verify_lock_p.add_mutually_exclusive_group(required=True)
    verify_lock_target.add_argument("--ara",
                                    help="Path to a single ARA directory or manifest.json.")
    verify_lock_target.add_argument(
        "--project",
        help="Project directory whose ara/ subtree will be swept (requires --all).",
    )
    verify_lock_p.add_argument(
        "--all", action="store_true",
        help="With --project, emit a per-ARA state summary across <project>/ara/.",
    )
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

    show_p = sub.add_parser(
        "show",
        help="Dump a node's full metadata as JSON (machine-readable inspect).",
    )
    show_p.add_argument("--ara", required=True)
    show_p.add_argument("--node", required=True, help="Node id to dump.")
    show_p.add_argument(
        "--term-tail", type=int, default=4000, metavar="N",
        help="Limit term_out.log tail to N bytes (default 4000; 0 = empty).",
    )
    show_p.add_argument(
        "--terse", action="store_true",
        help="Omit large text blobs (code, term_out_tail) — shape-only view.",
    )
    show_p.set_defaults(func=cmd_show)

    describe_p = sub.add_parser(
        "describe",
        help="Compact human overview of an ARA (idea, counts, top metric, lock, ancestors).",
    )
    describe_p.add_argument("--ara", required=True)
    describe_p.add_argument("--json", action="store_true",
                            help="Emit the overview as a JSON object.")
    describe_p.set_defaults(func=cmd_describe)

    list_p = sub.add_parser(
        "list",
        help="Enumerate every ARA under <project>/ara/ with a one-line summary.",
    )
    list_p.add_argument("--project", required=True,
                        help="Project directory whose ara/ subtree will be enumerated.")
    list_p.add_argument("--json", action="store_true",
                        help="Emit entries as a JSON array (full hashes preserved).")
    list_p.set_defaults(func=cmd_list)

    hash_check_p = sub.add_parser(
        "hash-check",
        help="Recompute each node's content_hash from disk and report drift.",
    )
    hash_check_target = hash_check_p.add_mutually_exclusive_group(required=True)
    hash_check_target.add_argument("--ara")
    hash_check_target.add_argument(
        "--project",
        help="Project directory whose ara/ subtree will be swept (requires --all).",
    )
    hash_check_p.add_argument(
        "--all", action="store_true",
        help="With --project, walk every ARA and aggregate per-node states.",
    )
    hash_check_p.add_argument(
        "--json", action="store_true",
        help="Emit entries as a JSON array (full hashes preserved).",
    )
    hash_check_p.set_defaults(func=cmd_hash_check)

    claims_p = sub.add_parser(
        "claims",
        help="Enumerate `\\claimref`-derived claim files with linked node metadata.",
    )
    claims_p.add_argument("--ara", required=True)
    claims_p.add_argument(
        "--json", action="store_true",
        help="Emit rows as a JSON array (full hashes + full assertion text).",
    )
    claims_p.add_argument(
        "--node", metavar="ID", default=None,
        help="Filter to claims linked to this node id (rc=3 when none match).",
    )
    claims_p.set_defaults(func=cmd_claims)

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
    refs_p.add_argument("--prefix", metavar="PATH", default=None,
                        help="Filter list mode to refs whose name starts with PATH "
                             "(git-style; use trailing slash to scope a namespace).")
    refs_p.set_defaults(func=cmd_refs)

    provenance_p = sub.add_parser(
        "provenance",
        help="Reverse-lookup: find every ARA under <project>/ara/ that references a hash.",
    )
    provenance_p.add_argument("--hash", required=True,
                              help="Full content hash string, e.g. sha256:<hex>.")
    provenance_p.add_argument("--project", required=True,
                              help="Project directory whose ara/ subtree will be swept.")
    provenance_p.add_argument("--json", action="store_true",
                              help="Emit hits as a JSON array (full hashes preserved).")
    provenance_p.set_defaults(func=cmd_provenance)

    bundle_p = sub.add_parser(
        "bundle",
        help="Package an ARA into a portable gzip tarball (git-bundle analog).",
    )
    bundle_p.add_argument("--ara", required=True,
                          help="Path to the ARA directory or manifest.json.")
    bundle_p.add_argument("--dest", required=True,
                          help="Output tarball path (e.g. out.tar.gz).")
    bundle_p.add_argument("--force", action="store_true",
                          help="Overwrite --dest if it already exists.")
    bundle_p.add_argument(
        "--no-verify", action="store_true",
        help="Skip the pre-flight verify_manifest_lock check.",
    )
    bundle_p.set_defaults(func=cmd_bundle)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
