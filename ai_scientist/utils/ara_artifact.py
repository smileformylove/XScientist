"""Agent-Native Research Artifact (ARA) exporter.

Motivation
----------
Inspired by "The Second Half of AI for Science" / "The Last Human-Written Paper"
(arXiv:2604.24658). A PDF is a lossy view of a research run — the exploration
tree, failed branches, and per-node execution logs are what a *downstream* AI
scientist needs in order to fork, re-execute, or verify prior work.

Rather than replace the existing PDF pipeline, this module writes a parallel
artifact directory next to the manuscript. Every field is a *reference* to the
underlying run data (journal.json, code, term_out, plots, pareto pool, repair
history, ...) so we neither duplicate storage nor lose fidelity.

Layout under `<project_dir>/ara/<idea_slug>/` (project_dir is the user-supplied
`--project-dir`, so the ARA export is co-located with `02_experiments/` and
`03_papers/`):

    manifest.json                Top-level pointer file (schema below).
    exploration_graph.json       Nodes + edges + status, one entry per journal node.
    nodes/<node_id>/
        code.py                  The exact code the node executed (if any).
        term_out.log             Untrimmed stdout/stderr (best-effort).
        metrics.json             Metric snapshot + analysis + is_buggy.
        plots.json               Plot paths + VLM analyses.
    claims/                      Populated later by claim_registry.
    repair_history.jsonl         Copied from repair_attempts / repair_reflection.
    pareto_pool.json             Copied from the manuscript pareto pool if present.
    env/
        bfts_config.yaml         Snapshot of the config used for this run.
        model_fingerprint.json   Model ids + temps + writing profile digest.
    README.md                    Agent-facing entry point.

The exporter is intentionally *tolerant*: if a piece of source data is missing
we skip it and record the omission in `manifest.json["missing"]`. A partial ARA
export is better than none — downstream agents can still fork whatever is
present.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.protocol import (
    NODE_IDENTITY_PROFILE,
    PROTOCOL_VERSION,
    hash_node_payload,
)
from ai_scientist.utils.ara_graph import (
    graph_with_dag_metadata,
    write_exploration_graph_visualization,
)
from ai_scientist.utils.evaluation_binding import evaluation_hash_binding
from ai_scientist.utils.ara_manifest_lock import (
    append_manifest_revision,
    write_manifest_lock,
)
from ai_scientist.utils.privacy import (
    redact_sensitive_payload,
    relativize_path_fields,
    relative_path_reference,
)

logger = logging.getLogger(__name__)


_EXECUTION_IDENTITY_FIELDS = (
    "command_hash",
    "container_digest",
    "environment_hash",
    "observed_dependency_hash",
    "runner_hash",
)
_EXECUTION_IDENTITY_LIST_FIELDS = (
    "dataset_hashes",
    "dependency_lock_hashes",
    "observed_dependency_hashes",
    "seeds",
    "tool_hashes",
)


def _node_execution_identity(raw: dict[str, Any]) -> dict[str, Any]:
    """Return portable, stable execution inputs explicitly supplied by a node.

    Paths, timestamps, host names, and raw environment variables are excluded.
    Producers may provide a complete ``execution_identity`` mapping, or the
    common hash/seed fields directly on the journal node.
    """

    supplied = raw.get("execution_identity")
    identity = dict(supplied) if isinstance(supplied, dict) else {}
    for field in _EXECUTION_IDENTITY_FIELDS:
        value = raw.get(field)
        if value not in (None, ""):
            identity.setdefault(field, str(value))
    for field in _EXECUTION_IDENTITY_LIST_FIELDS:
        values = raw.get(field)
        if isinstance(values, (list, tuple, set)):
            cleaned = sorted({value for value in values if value not in (None, "")})
            if cleaned:
                identity.setdefault(field, cleaned)
    backend = str(raw.get("execution_backend") or "").strip()
    if backend:
        identity.setdefault("backend", backend)
    isolation = raw.get("execution_isolation")
    if isinstance(isolation, dict) and isolation:
        identity.setdefault("isolation", dict(isolation))
    return identity


ARA_SCHEMA_VERSION = PROTOCOL_VERSION
ARA_ROOT_NAME = "ara"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str, fallback: str = "idea") -> str:
    text = (text or "").strip()
    if not text:
        return fallback
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    slug = slug[:80] or fallback
    return slug


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    safe_payload = redact_sensitive_payload(payload)
    tmp.write_text(
        json.dumps(safe_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def _sha256_of(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


@dataclass
class ARAExportResult:
    root: Path
    manifest_path: Path
    missing: list[str]
    node_count: int
    claim_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "missing": list(self.missing),
            "node_count": self.node_count,
            "claim_count": self.claim_count,
        }


def ara_root_for_project(project_dir: str | os.PathLike[str]) -> Path:
    """Return `<project_dir>/ara`. The user's project directory is the anchor."""
    return Path(project_dir).expanduser().resolve() / ARA_ROOT_NAME


def ara_dir_for_idea(
    project_dir: str | os.PathLike[str],
    idea_name: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Deterministic per-idea ARA directory. `timestamp` is a hint so multiple
    runs of the same idea don't collide. When omitted we let the caller decide.
    """
    root = ara_root_for_project(project_dir)
    slug = _slugify(idea_name)
    if timestamp:
        return root / f"{timestamp}_{slug}"
    return root / slug


def _find_journal_files(exp_dir: Path) -> list[Path]:
    """Locate all `journal.json` files under `<exp_dir>/logs/*/`."""
    logs_dir = exp_dir / "logs"
    if not logs_dir.exists():
        return []
    return sorted(logs_dir.rglob("journal.json"))


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ARA export: failed to load %s: %s", path, exc)
        return None


def _write_node_run_bundle(
    node_dir: Path,
    *,
    node_id: str,
    exp_dir: Path,
    ara_dir: Path,
) -> None:
    """Emit a minimal, self-describing `run.sh` + `env.json` per node.

    Design choice: we do NOT capture a full pip freeze here (too slow, and
    reproducibility to bit-level is not the goal). We just record enough for a
    downstream agent to (a) know which Python + which cwd is expected, and
    (b) invoke the node's code with one command. The env freeze is deferred to
    the fork CLI so users only pay that cost when they actually want to fork.
    """
    import sys

    exp_reference = relative_path_reference(exp_dir, base=ara_dir)

    env_payload = {
        "node_id": node_id,
        "python_executable": Path(sys.executable).name,
        "python_version": sys.version.split()[0],
        "expected_cwd": exp_reference,
        "path_base": "ara_root",
        "code_file": "code.py",
        "term_out_file": "term_out.log",
        "metrics_file": "metrics.json",
        "plots_file": "plots.json",
        "ara_root": ".",
        "notes": (
            "This is a lightweight bundle. Full env freeze is optional — call "
            "`python run_ara_fork.py --freeze` from the ARA root to snapshot "
            "the current interpreter's package versions."
        ),
    }
    _safe_write_json(node_dir / "env.json", env_payload)

    # Portable POSIX runner: cd into the original exp_dir so imports resolve,
    # execute the node's code with the current interpreter. Downstream tools
    # can override via env vars.
    run_sh = (
        "#!/usr/bin/env bash\n"
        "# ARA per-node runner. Regenerated by ara_artifact.py.\n"
        "# Override PYTHON / EXP_DIR via env if you fork this bundle.\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"\n'
        'NODE_ID="$(basename -- "$SCRIPT_DIR")"\n'
        'ARA_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"\n'
        f"EXP_REFERENCE={shlex.quote(exp_reference)}\n"
        'DEFAULT_EXP_DIR="$ARA_DIR/$EXP_REFERENCE"\n'
        'EXP_DIR="${EXP_DIR:-$DEFAULT_EXP_DIR}"\n'
        'PYTHON="${PYTHON:-python3}"\n'
        'CODE_FILE="$(dirname "$0")/code.py"\n'
        'if [ ! -f "$CODE_FILE" ]; then\n'
        '  echo "ARA: no code.py for node $NODE_ID" >&2\n'
        "  exit 2\n"
        "fi\n"
        'if [ ! -d "$EXP_DIR" ]; then\n'
        '  echo "ARA: expected cwd $EXP_DIR is missing; falling back to node dir" >&2\n'
        '  EXP_DIR="$(dirname "$0")"\n'
        "fi\n"
        'cd "$EXP_DIR"\n'
        'exec "$PYTHON" "$CODE_FILE" "$@"\n'
    )
    run_sh_path = node_dir / "run.sh"
    run_sh_path.write_text(run_sh, encoding="utf-8")
    try:
        run_sh_path.chmod(0o755)
    except OSError:  # pragma: no cover - platform-specific
        pass


def _export_nodes_from_journal(
    journal_path: Path,
    stage_label: str,
    nodes_root: Path,
    *,
    exp_dir: Path,
    ara_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read one journal.json and dump per-node artifacts.

    Returns (node_entries, edges). node_entries is the ARA graph payload for
    each node; edges is (parent_id, child_id) pairs.
    """
    payload = _load_json(journal_path)
    if not isinstance(payload, dict):
        return [], []

    raw_nodes = payload.get("nodes") or []
    node2parent = (
        payload.get("node2parent")
        if isinstance(payload.get("node2parent"), dict)
        else {}
    )
    child_map: dict[str, list[str]] = {}
    for child_id, parent_id in node2parent.items():
        if child_id and parent_id:
            child_map.setdefault(str(parent_id), []).append(str(child_id))
    node_entries: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id") or "").strip()
        if not node_id:
            continue

        node_dir = nodes_root / node_id
        node_dir.mkdir(parents=True, exist_ok=True)

        # 1. Exact code the node executed.
        code = raw.get("code")
        if isinstance(code, str) and code.strip():
            (node_dir / "code.py").write_text(code, encoding="utf-8")

        plot_code = raw.get("plot_code")
        if isinstance(plot_code, str) and plot_code.strip():
            (node_dir / "plot_code.py").write_text(plot_code, encoding="utf-8")

        # 2. Untrimmed terminal output — Node.to_dict() keeps _term_out raw.
        term_out = raw.get("_term_out")
        if isinstance(term_out, list):
            (node_dir / "term_out.log").write_text("".join(term_out), encoding="utf-8")
        elif isinstance(term_out, str) and term_out:
            (node_dir / "term_out.log").write_text(term_out, encoding="utf-8")

        plot_term_out = raw.get("plot_term_out")
        if isinstance(plot_term_out, list) and plot_term_out:
            (node_dir / "plot_term_out.log").write_text(
                "".join(plot_term_out), encoding="utf-8"
            )

        # 3. Metric + analysis snapshot.
        metrics_payload = {
            "metric": raw.get("metric"),
            "metric_provenance": raw.get("metric_provenance") or "unavailable",
            "evaluation_report": raw.get("evaluation_report"),
            "analysis": raw.get("analysis"),
            "is_buggy": raw.get("is_buggy"),
            "is_buggy_plots": raw.get("is_buggy_plots"),
            "exc_type": raw.get("exc_type"),
            "exc_info": raw.get("exc_info"),
            "exec_time": raw.get("exec_time"),
            "execution_backend": raw.get("execution_backend"),
            "execution_isolation": raw.get("execution_isolation"),
        }
        # Content hash lets downstream consumers match "the same experiment"
        # across ARA instances by payload, not path. Documented in SPEC.md.
        # We also bind Node.llm_call_refs (messages_ref hashes from
        # <ara>/llm/calls.jsonl) so two nodes with identical code+metric but
        # different prompts hash differently — closes the "same code, different
        # prompt" hole. content_hash_inputs records which categories fed the
        # hash so cross-version diff can detect binding scheme changes.
        # Seed-derived nodes also bind an "is_seed" marker so a seed vs.
        # non-seed variant of the same code hashes distinctly (SPEC §11.1).
        node_content_hash: str | None = None
        llm_refs_raw = raw.get("llm_call_refs") or []
        # Defensive: journal is JSON, but this list may arrive as any iterable
        # of strings (or nothing at all on legacy runs / seed nodes).
        llm_refs = [str(r) for r in llm_refs_raw if r]
        context_refs_raw = raw.get("context_pack_refs") or []
        context_refs = [str(r) for r in context_refs_raw if r]
        execution_identity = _node_execution_identity(raw)
        is_seed = bool(raw.get("is_seed_node"))
        evaluation_binding = evaluation_hash_binding(raw.get("evaluation_report"))
        hash_inputs = ["code", "metric"]
        if evaluation_binding:
            hash_inputs.append("evaluation")
        if llm_refs:
            hash_inputs.append("llm_calls")
        if context_refs:
            hash_inputs.append("context")
        if execution_identity:
            hash_inputs.append("execution")
        if is_seed:
            hash_inputs.append("seed")
        try:
            node_content_hash = hash_node_payload(
                code=code if isinstance(code, str) else "",
                metric=raw.get("metric"),
                extras=(
                    {"evaluation": evaluation_binding} if evaluation_binding else None
                ),
                llm_call_hashes=llm_refs or None,
                context_hashes=context_refs or None,
                execution_identity=execution_identity or None,
                is_seed=is_seed,
            )
        except Exception as exc:  # pragma: no cover - defensive; hashing is best-effort
            logger.warning("ARA export: hashing node %s failed: %s", node_id, exc)
        if node_content_hash:
            metrics_payload["content_hash"] = node_content_hash
            metrics_payload["content_hash_inputs"] = hash_inputs
            metrics_payload["identity_profile"] = NODE_IDENTITY_PROFILE
            metrics_payload["execution_identity"] = execution_identity
        _safe_write_json(node_dir / "metrics.json", metrics_payload)
        if isinstance(raw.get("evaluation_report"), dict):
            _safe_write_json(node_dir / "evaluation.json", raw["evaluation_report"])

        # 4. Plot references.
        plots_payload = {
            "plots": raw.get("plots") or [],
            "plot_paths": raw.get("plot_paths") or [],
            "plot_analyses": raw.get("plot_analyses") or [],
            "vlm_feedback_summary": raw.get("vlm_feedback_summary") or [],
            "plots_generated": raw.get("plots_generated"),
        }
        _safe_write_json(node_dir / "plots.json", plots_payload)

        # 5. Per-node run bundle (see `_write_node_run_bundle` for rationale).
        # Only emit if the node has code — otherwise nothing to re-execute.
        if isinstance(code, str) and code.strip():
            _write_node_run_bundle(
                node_dir, node_id=node_id, exp_dir=exp_dir, ara_dir=ara_dir
            )

        parent_id = raw.get("parent_id")
        if not parent_id:
            parent_id = node2parent.get(node_id)
        children = raw.get("children") or child_map.get(node_id, [])
        if not isinstance(children, list):
            children = []

        node_entries.append(
            {
                "id": node_id,
                "content_hash": node_content_hash,
                "content_hash_inputs": hash_inputs,
                "identity_profile": NODE_IDENTITY_PROFILE,
                "execution_identity": execution_identity,
                "llm_call_refs": llm_refs,
                "context_pack_refs": context_refs,
                "stage": stage_label,
                "step": raw.get("step"),
                "parent_id": str(parent_id) if parent_id else None,
                "children": [str(child_id) for child_id in children],
                "is_buggy": raw.get("is_buggy"),
                "is_seed_node": raw.get("is_seed_node"),
                "is_seed_agg_node": raw.get("is_seed_agg_node"),
                "metric": raw.get("metric"),
                "metric_provenance": raw.get("metric_provenance") or "unavailable",
                "evaluation_report": raw.get("evaluation_report"),
                "plan_excerpt": (raw.get("plan") or "")[:400],
                "exp_results_dir": raw.get("exp_results_dir"),
                "ctime": raw.get("ctime"),
                "execution_backend": raw.get("execution_backend"),
                "execution_isolation": raw.get("execution_isolation"),
                "artifacts_dir": str(node_dir.relative_to(nodes_root.parent.parent)),
            }
        )

        if parent_id:
            edges.append(
                {"parent": str(parent_id), "child": node_id, "stage": stage_label}
            )
        for child_id in children:
            child = str(child_id).strip()
            if child:
                edges.append({"parent": node_id, "child": child, "stage": stage_label})

    return node_entries, edges


def _normalize_graph_entries(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge same-content carry-over nodes and dedupe edges.

    Multi-stage BFTS journals can append a deepcopy of the previous stage's
    best node into the next stage while preserving its id. That is one logical
    exploration node, not a second DAG vertex. If the duplicate carries the same
    content_hash (or either side is legacy/unhashed), merge stage/children
    metadata. If the same id has a different content_hash, keep it duplicated so
    the DAG validator surfaces the id collision instead of hiding data loss.
    """

    merged: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids_with_distinct_hash: set[str] = set()

    for node in nodes:
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        existing = by_id.get(node_id)
        if existing is None or node_id in duplicate_ids_with_distinct_hash:
            current = dict(node)
            stage = current.get("stage")
            current["stages"] = [stage] if stage else []
            merged.append(current)
            by_id.setdefault(node_id, current)
            continue

        existing_hash = existing.get("content_hash")
        current_hash = node.get("content_hash")
        if existing_hash and current_hash and existing_hash != current_hash:
            duplicate_ids_with_distinct_hash.add(node_id)
            current = dict(node)
            stage = current.get("stage")
            current["stages"] = [stage] if stage else []
            merged.append(current)
            continue

        stage = node.get("stage")
        stages = existing.setdefault("stages", [])
        if stage and stage not in stages:
            stages.append(stage)

        if not existing.get("parent_id") and node.get("parent_id"):
            existing["parent_id"] = node.get("parent_id")
        elif (
            existing.get("parent_id")
            and node.get("parent_id")
            and existing.get("parent_id") != node.get("parent_id")
        ):
            conflicts = existing.setdefault("parent_id_conflicts", [])
            conflict = str(node.get("parent_id"))
            if conflict not in conflicts:
                conflicts.append(conflict)

        children = existing.setdefault("children", [])
        for child_id in node.get("children") or []:
            child = str(child_id)
            if child not in children:
                children.append(child)

    edge_seen: set[tuple[str, str]] = set()
    normalized_edges: list[dict[str, Any]] = []
    for edge in edges:
        parent = str(edge.get("parent") or "").strip()
        child = str(edge.get("child") or "").strip()
        if not parent or not child:
            continue
        key = (parent, child)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        normalized_edges.append(
            {
                "parent": parent,
                "child": child,
                "stage": edge.get("stage"),
            }
        )

    # ``edges`` is the canonical topology for new exports.  Keeping parent_id
    # and children alongside it made every relation appear up to three times
    # and allowed the representations to drift.  Readers remain backwards
    # compatible with older graphs that only carry node-local links.
    for node in merged:
        node.pop("parent_id", None)
        node.pop("parent_id_conflicts", None)
        node.pop("children", None)

    return merged, normalized_edges


def _copy_optional(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return True
    except OSError as exc:
        logger.warning("ARA export: failed to copy %s -> %s: %s", src, dst, exc)
        return False


def _writing_profile_content_hash(name: str | None) -> str | None:
    """Return ``sha256:<hex>`` over the resolved writing-profile dict, or None.

    Two runs with the same profile name but an edited profile body would
    otherwise share a fingerprint — silently breaking A/B comparability across
    tweaks to ``writing_prompt_profiles.py``. Hashing the resolved dict body
    closes that hole. Unknown / unresolvable names return None so the export
    stays best-effort.
    """
    if not name:
        return None
    try:
        from ai_scientist.writing_prompt_profiles import WRITING_PROFILE_SPECS
    except Exception:
        return None
    try:
        profile = WRITING_PROFILE_SPECS.get(str(name))
    except Exception:
        return None
    if profile is None:
        return None
    try:
        from ai_scientist.protocol import content_hash

        return content_hash(dict(profile))
    except Exception:
        return None


def _writing_profile_slot(name: str | None) -> dict[str, Any]:
    """Structured `writing_profile` slot for the model fingerprint.

    Emits ``{"name": <str|null>, "content_hash": <"sha256:..."|null>}`` so
    downstream consumers see a stable shape regardless of whether the name
    resolves to a known profile body.
    """
    return {
        "name": name if name else None,
        "content_hash": _writing_profile_content_hash(name),
    }


def _digest_model_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    """Derive a small, deterministic fingerprint of the models/profiles used.

    We do not attempt to record every prompt — just what a downstream agent
    needs to know to compare runs.
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"fingerprint": digest, "spec": payload}


def build_env_snapshot(
    exp_dir: Path,
    *,
    bfts_config_path: str | None,
    model_spec: dict[str, Any] | None,
    writing_profile: str | None,
) -> tuple[dict[str, str], list[str]]:
    """Write env/*.{yaml,json}; return (relative-file-map, missing-notes)."""
    env_dir_files: dict[str, str] = {}
    missing: list[str] = []

    env_dir = exp_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    if bfts_config_path and Path(bfts_config_path).exists():
        dst = env_dir / "bfts_config.yaml"
        if _copy_optional(Path(bfts_config_path), dst):
            env_dir_files["bfts_config"] = str(dst.relative_to(env_dir.parent))
    else:
        missing.append("env/bfts_config.yaml (source missing)")

    fingerprint_payload = _digest_model_fingerprint(
        {
            "models": model_spec or {},
            "writing_profile": _writing_profile_slot(writing_profile),
            "ara_schema_version": ARA_SCHEMA_VERSION,
        }
    )
    fp_path = env_dir / "model_fingerprint.json"
    _safe_write_json(fp_path, fingerprint_payload)
    env_dir_files["model_fingerprint"] = str(fp_path.relative_to(env_dir.parent))

    return env_dir_files, missing


def _pdf_reference(exp_dir_source: Path, *, ara_dir: Path) -> dict[str, Any] | None:
    """Best-effort pointer to the final PDF sitting in the source exp_dir."""
    try:
        from ai_scientist.utils.pipeline_helpers import find_latest_pdf_path
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        pdf_path = find_latest_pdf_path(exp_dir_source)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ARA export: locating PDF failed: %s", exc)
        return None
    if not pdf_path:
        return None
    return {
        "path": relative_path_reference(pdf_path, base=ara_dir),
        "sha256": _sha256_of(Path(pdf_path)),
    }


def _snapshot_pipeline_artifacts(
    *,
    ara_dir: Path,
    exp_dir: Path,
    project_dir: Path,
    missing: list[str],
) -> list[dict[str, Any]]:
    """Copy pipeline_contracts artifacts into the ARA and content-address them.

    For each of the 16 kinds registered in
    :data:`ai_scientist.utils.pipeline_contracts.ARTIFACT_FILENAMES`, this
    helper looks in a small set of well-known locations (experiment dir,
    hidden ``.pipeline`` subdir, project dir, project-level ``.pipeline``
    subdir), copies the first file it finds into ``<ara>/pipeline/`` for
    human browsing, and *also* stores the same bytes under ``<ara>/objects/``
    so cross-ARA diff can go straight to a hash comparison.

    Silently absent artifacts are recorded once in ``missing`` — we don't want
    to spam the manifest with 16 lines when a project only produces two.

    Never raises. If ``pipeline_contracts`` isn't importable (unlikely, but
    defends against install-time drift) we skip snapshotting entirely.
    """
    try:
        from ai_scientist.utils.pipeline_contracts import ARTIFACT_FILENAMES
    except Exception:  # pragma: no cover - defensive
        return []

    try:
        from ai_scientist.protocol import ObjectStore
    except Exception:  # pragma: no cover - defensive
        return []

    pipeline_dir = ara_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    try:
        store = ObjectStore(
            ara_dir,
            shared_root=project_dir / ".ara-store",
        )
    except Exception:  # pragma: no cover - defensive
        return []

    entries: list[dict[str, Any]] = []
    absent: list[str] = []
    for kind, filename in ARTIFACT_FILENAMES.items():
        src = _find_pipeline_artifact(exp_dir, project_dir, filename)
        if src is None:
            absent.append(f"pipeline_artifacts/{kind} (source missing)")
            continue
        try:
            raw = src.read_bytes()
        except OSError as exc:
            logger.warning("ARA export: pipeline artifact %s unreadable: %s", src, exc)
            continue
        try:
            ref = store.put_bytes(raw)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ARA export: CAS put failed for %s: %s", src, exc)
            continue
        # Mirror to pipeline/ for human browsing. Overwrites are fine —
        # bytes on disk equal bytes in CAS by construction of put_bytes.
        dst = pipeline_dir / filename
        try:
            dst.write_bytes(raw)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning(
                "ARA export: pipeline mirror write failed for %s: %s", dst, exc
            )
            continue
        entries.append(
            {
                "kind": kind,
                "path": str(dst.relative_to(ara_dir)),
                "content_hash": ref.hash,
                "size": ref.size,
                "generated_at": _iso_mtime(src),
                "source": relative_path_reference(src, base=ara_dir),
            }
        )

    if absent:
        # Roll the individually-absent kinds into a single note so the
        # manifest doesn't get 16 lines of noise.
        missing.append(
            f"pipeline_artifacts absent: {len(absent)}/{len(ARTIFACT_FILENAMES)}"
        )
    return entries


def _find_pipeline_artifact(
    exp_dir: Path, project_dir: Path, filename: str
) -> Path | None:
    """Return the first existing path for ``filename`` across known locations.

    Order matters — the experiment-dir hits win over project-level fallbacks
    so a per-run version overrides a leftover project-level file.
    """
    candidates = (
        exp_dir / filename,
        exp_dir / ".pipeline" / filename,
        project_dir / filename,
        project_dir / ".pipeline" / filename,
    )
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def _iso_mtime(path: Path) -> str | None:
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:  # pragma: no cover - defensive
        return None
    return ts.isoformat().replace("+00:00", "Z")


def _snapshot_seed_manifest(
    *,
    ara_dir: Path,
    project_dir: Path,
) -> dict[str, Any] | None:
    """Copy the consumed seed manifest into the child ARA.

    Rationale
    ---------
    Before this change, seed manifests lived only under
    ``<project>/.ara_seed/ara_seed.json`` and never made it into the child's
    on-disk ARA (SPEC §7.1 explicitly said so). Downstream consumers could
    see ``manifest.provenance.parent_*`` but had no way to recover *which
    exact code was seeded* without going back to the parent — and if the
    parent ARA had moved or been pruned, the trail dead-ended.

    Now every child ARA carries its own ``<ara>/seed/ara_seed.json``, and
    the manifest's ``provenance.seed_hash`` field is a content_hash over
    that file. Two effects:

    1. Consumers can inspect the seed without touching the parent ARA.
    2. Diffing two children of the same parent can compare their
       ``seed_hash`` values directly — if identical, the divergence is
       entirely post-seed (LLM stochasticity, environment); if different,
       the fork itself pivoted.

    We only snapshot when a ``.consumed`` sidecar proves the seed was
    actually used by this run. A staged-but-unused seed (rare — happens
    when the pipeline aborts before ``_draft``) is intentionally not
    snapshotted; it isn't part of this run's history.

    Returns a dict suitable for ``manifest.references['seed']`` plus a
    ``seed_hash`` string for injection into ``manifest.provenance``.
    """
    try:
        from ai_scientist.utils.ara_seed import SEED_MANIFEST_NAME
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        from ai_scientist.protocol import ObjectStore
    except Exception:  # pragma: no cover - defensive
        return None

    src = project_dir / ".ara_seed" / SEED_MANIFEST_NAME
    consumed = src.with_suffix(src.suffix + ".consumed")
    if not src.exists() or not consumed.exists():
        return None

    try:
        raw = src.read_bytes()
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("ARA export: seed manifest unreadable at %s: %s", src, exc)
        return None

    try:
        store = ObjectStore(
            ara_dir,
            shared_root=project_dir / ".ara-store",
        )
        ref = store.put_bytes(raw)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ARA export: CAS put failed for seed: %s", exc)
        return None

    seed_dir = ara_dir / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    dst = seed_dir / SEED_MANIFEST_NAME
    try:
        dst.write_bytes(raw)
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("ARA export: seed mirror write failed for %s: %s", dst, exc)
        return None

    # Preserve the .consumed sidecar too — it's the provenance receipt
    # showing which idea consumed the seed and when.
    try:
        consumed_raw = consumed.read_bytes()
        (seed_dir / consumed.name).write_bytes(consumed_raw)
    except OSError:
        pass  # sidecar is informational; missing it doesn't invalidate the seed copy

    return {
        "kind": "seed_manifest",
        "path": str(dst.relative_to(ara_dir)),
        "content_hash": ref.hash,
        "size": ref.size,
        "generated_at": _iso_mtime(src),
        "source": relative_path_reference(src, base=ara_dir),
    }


def _write_agent_readme(root: Path, manifest: dict[str, Any]) -> None:
    """A short, structured entry point for downstream agents (not humans)."""
    idea = manifest.get("idea", {})
    lines = [
        "# ARA Entry Point",
        "",
        f"- Schema: {manifest.get('schema_version')}",
        f"- Created: {manifest.get('created_at')}",
        f"- Idea: {idea.get('name') or 'unnamed'}",
        f"- Origin exp_dir: {manifest.get('source_exp_dir')}",
        "",
        "## What lives here",
        "",
        "- `exploration_graph.json` — every node from the tree search (including buggy ones).",
        "- `exploration_graph.html` — browser visualization of the exploration DAG.",
        "- `nodes/<id>/` — code, term_out, metrics, plots for each node. Fork from any node id.",
        "- `claims/` — machine-readable claims linking each manuscript assertion to its evidence node.",
        "- `events/research_events.jsonl` — compact admitted state/evidence events (not raw chatter).",
        "- `catalog/semantic.sqlite` — disposable, rebuildable query index over the complete record.",
        "- `repair_history.jsonl` — full self-review / repair trajectory.",
        "- `pareto_pool.json` — non-dominated manuscript candidates.",
        "- `env/` — config + model fingerprint needed to reproduce.",
        "- `README.md` — this file (agent-facing).",
        "",
        "## How to fork",
        "",
        "1. Pick a node id from `exploration_graph.json`.",
        "2. Read `nodes/<id>/code.py` and `metrics.json`.",
        "3. Continue the tree search with that node as the seed.",
        "4. Before acting, compile a bounded view with `xscientist ara context --intent continue --node <id> --ara <root>`.",
        "",
        "Failed branches are first-class citizens: consult them to learn what NOT to try.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_ara(
    *,
    project_dir: str | os.PathLike[str],
    exp_dir: str | os.PathLike[str],
    idea: dict[str, Any] | None = None,
    timestamp: str | None = None,
    bfts_config_path: str | None = None,
    model_spec: dict[str, Any] | None = None,
    writing_profile: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> ARAExportResult:
    """Build the ARA export for one idea run.

    Parameters
    ----------
    project_dir:
        User-supplied project root (e.g. `args.project_dir`). ARA lands at
        `{project_dir}/ara/...` per the user request.
    exp_dir:
        Per-idea experiment directory (`02_experiments/<ts>_<idea>`).
    idea:
        Idea dict (typically loaded from `exp_dir/idea.json`). Used for the
        manifest's `idea` block; falls back to the exp_dir basename.
    timestamp:
        Optional label appended to the ARA subdir to distinguish repeat runs.
    """
    project_dir_path = Path(project_dir).expanduser().resolve()
    exp_dir_path = Path(exp_dir).expanduser().resolve()

    idea = idea or {}
    idea_name = str(idea.get("Name") or idea.get("name") or exp_dir_path.name)

    # Prefer explicit caller-provided timestamp; else derive from exp_dir name so
    # exports coming from the same experiment collapse deterministically.
    if timestamp is None:
        basename = exp_dir_path.name
        ts_match = re.match(
            r"^(\d{4}-\d{2}-\d{2}[_T-]?\d{2}[:_-]?\d{2}(?:[:_-]?\d{2})?)", basename
        )
        timestamp = ts_match.group(1) if ts_match else None

    ara_dir = ara_dir_for_idea(project_dir_path, idea_name, timestamp=timestamp)
    ara_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    nodes_root = ara_dir / "nodes"
    nodes_root.mkdir(parents=True, exist_ok=True)

    # 1. Exploration graph: iterate every journal.json under logs/*.
    journals = _find_journal_files(exp_dir_path)
    all_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    if not journals:
        missing.append("exploration_graph: no journal.json found under logs/")
    for journal_path in journals:
        stage_label = journal_path.parent.name
        nodes, edges = _export_nodes_from_journal(
            journal_path,
            stage_label,
            nodes_root,
            exp_dir=exp_dir_path,
            ara_dir=ara_dir,
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)
    all_nodes, all_edges = _normalize_graph_entries(all_nodes, all_edges)

    exploration_graph_path = ara_dir / "exploration_graph.json"
    graph_payload = redact_sensitive_payload(
        graph_with_dag_metadata(
            {
                "schema_version": ARA_SCHEMA_VERSION,
                "protocol_kind": "exploration_graph",
                "topology_encoding": "edges",
                "generated_at": _now_iso(),
                "nodes": all_nodes,
                "edges": all_edges,
                "source_journals": [
                    relative_path_reference(path, base=ara_dir) for path in journals
                ],
                "counts": {
                    "nodes": len(all_nodes),
                    "edges": len(all_edges),
                    "buggy": sum(1 for n in all_nodes if n.get("is_buggy")),
                },
            },
        )
    )
    _safe_write_json(exploration_graph_path, graph_payload)
    graph_visualization_refs = write_exploration_graph_visualization(
        ara_dir,
        graph_payload,
    )

    # 2. Optional companions: pareto pool, repair history, experiment registry.
    references: dict[str, Any] = {
        "exploration_graph_visualization": graph_visualization_refs,
    }

    pareto_source = exp_dir_path / "pareto_pool" / "manuscript_candidate_pool.json"
    if not pareto_source.exists():
        # Some pipeline_contracts variants write into `.pipeline/*`.
        alt = exp_dir_path / ".pipeline" / "manuscript_candidate_pool.json"
        if alt.exists():
            pareto_source = alt
    if pareto_source.exists():
        dst = ara_dir / "pareto_pool.json"
        if _copy_optional(pareto_source, dst):
            references["pareto_pool"] = str(dst.relative_to(ara_dir))
    else:
        missing.append("pareto_pool.json (source missing)")

    # Repair history: attempt several known filenames; append into one jsonl.
    repair_dst = ara_dir / "repair_history.jsonl"
    repair_written = 0
    for candidate in [
        exp_dir_path / "repair_attempts.jsonl",
        exp_dir_path / ".pipeline" / "repair_attempts.jsonl",
        exp_dir_path / "repair_reflection.jsonl",
        exp_dir_path / ".pipeline" / "repair_reflection.jsonl",
        exp_dir_path / "repair_verifier.jsonl",
        exp_dir_path / ".pipeline" / "repair_verifier.jsonl",
    ]:
        if not candidate.exists():
            continue
        try:
            data = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        if not data:
            continue
        header = json.dumps({"__source__": candidate.name}) + "\n"
        with repair_dst.open("a", encoding="utf-8") as fh:
            fh.write(header)
            if not data.endswith("\n"):
                data = data + "\n"
            fh.write(data)
        repair_written += 1
    if repair_written == 0:
        missing.append("repair_history.jsonl (no source files found)")
    else:
        references["repair_history"] = str(repair_dst.relative_to(ara_dir))

    # Experiment registry — small, keep a copy.
    registry_src = exp_dir_path / "experiment_registry.jsonl"
    if registry_src.exists():
        dst = ara_dir / "experiment_registry.jsonl"
        if _copy_optional(registry_src, dst):
            references["experiment_registry"] = str(dst.relative_to(ara_dir))
    else:
        missing.append("experiment_registry.jsonl (source missing)")

    # 3. Env snapshot.
    env_files, env_missing = build_env_snapshot(
        ara_dir,
        bfts_config_path=bfts_config_path,
        model_spec=model_spec,
        writing_profile=writing_profile,
    )
    missing.extend(env_missing)
    references["env"] = env_files

    # 4. Claims: reserve the directory, populated later by claim_registry.
    (ara_dir / "claims").mkdir(parents=True, exist_ok=True)

    # 5. PDF reference.
    pdf_ref = _pdf_reference(exp_dir_path, ara_dir=ara_dir)
    if pdf_ref:
        references["pdf"] = pdf_ref

    # 5b. Pipeline artifacts snapshot.
    # The 16 kinds tracked in pipeline_contracts (review_state, critic_findings,
    # claim_evidence_graph, manuscript_state, figure_spec, …) live inside
    # <exp_dir>/<project layout> today and never made it into ARA. Snapshot
    # them into <ara>/pipeline/ for human browsing and into <ara>/objects/
    # for content-addressed diffing.
    references["pipeline_artifacts"] = _snapshot_pipeline_artifacts(
        ara_dir=ara_dir,
        exp_dir=exp_dir_path,
        project_dir=project_dir_path,
        missing=missing,
    )

    # 5c. Consumed-seed snapshot.
    # When this run was seeded (via `--seed-from-ara`), the seed manifest was
    # staged under <project>/.ara_seed/ but never made it into the child ARA.
    # Now we copy it in and stamp its content_hash into `provenance.seed_hash`
    # so downstream tools can inspect the exact seed without touching the
    # parent ARA (SPEC.md §7 already prefers content-hash references over
    # path-based ones).
    seed_entry = _snapshot_seed_manifest(
        ara_dir=ara_dir,
        project_dir=project_dir_path,
    )
    seed_hash: str | None = None
    if seed_entry is not None:
        references["seed"] = seed_entry
        seed_hash = seed_entry.get("content_hash")

    # 6. Manifest.
    manifest = {
        "schema_version": ARA_SCHEMA_VERSION,
        "protocol_kind": "manifest",
        "portability_profile": "ara.portable.v1",
        "conformance_profile": "ara.conformance.v1",
        "created_at": _now_iso(),
        "source_exp_dir": relative_path_reference(exp_dir_path, base=ara_dir),
        "project_dir": relative_path_reference(project_dir_path, base=ara_dir),
        "idea": {
            "name": idea_name,
            "title": idea.get("Title") or idea.get("title"),
            "raw": idea,
        },
        "counts": {
            "nodes": len(all_nodes),
            "edges": len(all_edges),
            "buggy_nodes": sum(1 for n in all_nodes if n.get("is_buggy")),
            "journals": len(journals),
        },
        "references": references,
        "missing": missing,
        "capabilities": {
            "inspect": "complete",
            "fork": (
                "complete"
                if any(
                    (nodes_root / str(node.get("id") or "") / "code.py").exists()
                    for node in all_nodes
                )
                else "unavailable"
            ),
            "reproduce": "partial",
            "audit": "partial" if missing else "complete",
            "context": "complete" if all_nodes else "partial",
        },
        "omissions": [
            {
                "kind": note.split(" ", 1)[0].rstrip(":"),
                "reason": note,
                "affects": ["reproduce", "audit"],
            }
            for note in missing
        ],
    }
    if provenance:
        manifest["provenance"] = dict(provenance)
    if seed_hash:
        # Attach the seed hash to provenance even if the caller supplied no
        # provenance dict — the seed itself IS a form of parentage.
        provenance_block = dict(manifest.get("provenance") or {})
        provenance_block["seed_hash"] = seed_hash
        manifest["provenance"] = provenance_block
    manifest = relativize_path_fields(manifest, base=ara_dir)
    manifest = redact_sensitive_payload(manifest)
    manifest_path = ara_dir / "manifest.json"
    _safe_write_json(manifest_path, manifest)

    # 6b. Freeze the manifest. write_manifest_lock stamps the base hash into
    # <ara>/manifest.lock; any post-export edit must go through
    # append_manifest_revision so the audit chain stays intact.
    try:
        write_manifest_lock(ara_dir, manifest)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ARA export: failed to write manifest.lock: %s", exc)

    _write_agent_readme(ara_dir, manifest)

    logger.info(
        "ARA export written: %s (nodes=%d, edges=%d, missing=%d)",
        ara_dir,
        len(all_nodes),
        len(all_edges),
        len(missing),
    )

    # `claim_count` is 0 at this stage; claim_registry updates the manifest
    # afterwards when it finishes scanning the manuscript.
    return ARAExportResult(
        root=ara_dir,
        manifest_path=manifest_path,
        missing=missing,
        node_count=len(all_nodes),
        claim_count=0,
    )


def update_manifest_claim_count(
    manifest_path: str | os.PathLike[str], claim_count: int
) -> None:
    """Callback used by claim_registry once claims/*.json are populated.

    Historically this rewrote manifest.json in place, silently mutating the
    "commit-like" top-level pointer. It now goes through the append-only
    revision API — the pre-mutation manifest is archived under history/,
    a manifest_revision row appears in manifest.history.jsonl, and
    manifest.lock still anchors revision 0.
    """

    def _apply(payload: dict) -> list[str]:
        payload.setdefault("counts", {})["claims"] = int(claim_count)
        payload["counts_updated_at"] = _now_iso()
        return ["counts.claims", "counts_updated_at"]

    append_manifest_revision(
        manifest_path,
        _apply,
        reason=f"claim count set to {int(claim_count)}",
        producer="update_manifest_claim_count",
    )


def iter_ara_exports(project_dir: str | os.PathLike[str]) -> Iterable[Path]:
    """Yield each `manifest.json` under `<project_dir>/ara/`."""
    root = ara_root_for_project(project_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("manifest.json"))
