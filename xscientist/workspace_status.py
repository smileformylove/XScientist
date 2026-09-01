"""One compact, read-only status view for a research workspace."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from copy import deepcopy
from itertools import islice
from pathlib import Path
from typing import Any

import yaml

from ai_scientist.utils.privacy import portable_path, redact_sensitive_payload

from .provider_config import discover_workspace_root
from .research_closure import (
    audit_research_closure,
    closure_level_summary,
    summarize_closure_levels,
)
from .research_git import ResearchGitError, repository_status, show_checkpoint
from .research_journey import (
    build_research_guide,
    public_workspace_action,
    workspace_action_context,
)

STATUS_SCHEMA = "xscientist.workspace-status.v1"
RUN_SCHEMA = "xscientist.local-run.v1"

_DELIVERABLE_CONTRACT_FILES = (
    "pipeline_manifest.json",
    "claim_evidence_graph.json",
    "experiment_registry.jsonl",
    "figure_spec.json",
    "manuscript_state.json",
    "review_state.json",
    "critic_findings.json",
    "repair_plan.json",
    "repair_attempts.jsonl",
    "stage_standards.json",
    "process_alignment.json",
    "verification_report.json",
)
_MAX_DELIVERABLE_ROOTS = 512
_MAX_POINTER_RECORDS = 4096
_MAX_CHECKPOINT_BINDING_COMMITS = 512
_SUCCESSFUL_EXPERIMENT_STATES = {"success", "succeeded", "completed"}
_FAILED_EXPERIMENT_STATES = {
    "failed",
    "error",
    "timeout",
    "timed_out",
    "cancelled",
    "canceled",
    "interrupted",
    "rejected",
    "orphan",
    "orphaned",
    "budget_exhausted",
}


def _state_path_error(path: Path, *, root: Path | None) -> str | None:
    """Reject state paths that leave the workspace or cross a symlink."""

    if root is None:
        return "state path must not be a symlink" if path.is_symlink() else None
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError:
        return "state path escapes the workspace"
    cursor = lexical_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            return "state path must not use symlinks"
    try:
        lexical_path.resolve(strict=False).relative_to(lexical_root.resolve())
    except (OSError, ValueError):
        return "state path escapes the workspace"
    return None


def _workspace_state_label(path: Path, *, root: Path) -> str:
    """Name a managed state file without resolving a rejected symlink target."""

    try:
        return (
            Path(os.path.abspath(path))
            .relative_to(Path(os.path.abspath(root)))
            .as_posix()
        )
    except ValueError:
        return "[REDACTED_PATH]"


def _read_json(
    path: Path,
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    path_error = _state_path_error(path, root=root)
    if path_error:
        return {}, path_error
    if not path.exists():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {}, f"cannot read state: {type(exc).__name__}"
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON at line {exc.lineno}, column {exc.colno}"
    if not isinstance(value, dict):
        return {}, "state root must be a JSON object"
    return value, None


def _state_shape_error(
    payload: dict[str, Any],
    *,
    label: str,
    sequence_fields: tuple[str, ...] = (),
    mapping_fields: tuple[str, ...] = (),
    text_fields: tuple[str, ...] = (),
    mapping_item_fields: tuple[str, ...] = (),
) -> str | None:
    """Validate only fields consumed by the compact status view."""

    for field in sequence_fields:
        if (
            field in payload
            and payload[field] is not None
            and not isinstance(payload[field], list)
        ):
            return f"{label}.{field} must be a JSON array"
    for field in mapping_fields:
        if (
            field in payload
            and payload[field] is not None
            and not isinstance(payload[field], dict)
        ):
            return f"{label}.{field} must be a JSON object"
    for field in text_fields:
        if (
            field in payload
            and payload[field] is not None
            and not isinstance(payload[field], str)
        ):
            return f"{label}.{field} must be a string"
    for field in mapping_item_fields:
        values = payload.get(field)
        if isinstance(values, list) and any(
            not isinstance(item, dict) for item in values
        ):
            return f"{label}.{field} entries must be JSON objects"
    return None


def _validated_state_json(
    path: Path,
    *,
    root: Path,
    label: str,
    sequence_fields: tuple[str, ...] = (),
    mapping_fields: tuple[str, ...] = (),
    text_fields: tuple[str, ...] = (),
    mapping_item_fields: tuple[str, ...] = (),
) -> tuple[dict[str, Any], str | None]:
    payload, error = _read_json(path, root=root)
    if error:
        return {}, error
    shape_error = _state_shape_error(
        payload,
        label=label,
        sequence_fields=sequence_fields,
        mapping_fields=mapping_fields,
        text_fields=text_fields,
        mapping_item_fields=mapping_item_fields,
    )
    return ({}, shape_error) if shape_error else (payload, None)


def _workspace_identity(root: Path) -> tuple[str, str]:
    """Return a useful name and stable, path-free workspace identity."""

    name = root.name or "workspace"
    identity_source = name.encode("utf-8")
    research_config = root / "research.yaml"
    if research_config.is_file():
        try:
            raw = research_config.read_text(encoding="utf-8")
            config = yaml.safe_load(raw)
            if isinstance(config, dict):
                repository_id = str(config.get("repository_id") or "")
                if repository_id.startswith("ws-") and len(repository_id) == 19:
                    return name, repository_id
                identity_source = (
                    str(config.get("schema_version") or "")
                    + ":"
                    + str(config.get("name") or name)
                ).encode("utf-8")
            else:
                identity_source = raw.encode("utf-8")
        except (OSError, yaml.YAMLError):
            pass
    return name, "ws-" + hashlib.sha256(identity_source).hexdigest()[:12]


def _first_existing(paths: list[Path], *, root: Path) -> Path | None:
    return next(
        (
            path
            for path in paths
            if _state_path_error(path, root=root) is None and path.is_file()
        ),
        None,
    )


def _count(value: Any, *, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def _workspace_file(
    root: Path,
    value: Any,
    *,
    base: Path | None = None,
) -> Path | None:
    """Resolve one recorded path only when it is a regular workspace file."""

    rendered = str(value or "").strip()
    if not rendered or "://" in rendered:
        return None
    candidate = Path(rendered).expanduser()
    if not candidate.is_absolute():
        candidate = (base or root) / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() and not resolved.is_symlink() else None


def _latest_experiment_selection(
    root: Path,
    progress: dict[str, Any],
) -> tuple[Path | None, dict[str, Any] | None]:
    """Select one experiment and the progress result that actually names it."""

    experiments_root = root / "02_experiments"
    candidates: dict[Path, dict[str, Any] | None] = {}
    for result in progress.get("results") or []:
        if not isinstance(result, dict):
            continue
        recorded = _workspace_file(root, result.get("pipeline_manifest"))
        if recorded is not None:
            candidates[recorded.parent] = result
            continue
        raw_directory = str(result.get("exp_dir") or "").strip()
        if not raw_directory:
            continue
        candidate = Path(raw_directory).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(experiments_root.resolve())
        except (OSError, ValueError):
            continue
        if resolved.is_dir() and not resolved.is_symlink():
            candidates[resolved] = result
    if experiments_root.is_dir() and not experiments_root.is_symlink():
        try:
            entries = sorted(experiments_root.iterdir(), key=lambda path: path.name)
        except OSError:
            entries = []
        for candidate in entries[-_MAX_DELIVERABLE_ROOTS:]:
            if candidate.is_dir() and not candidate.is_symlink():
                candidates.setdefault(candidate.resolve(), None)
    if not candidates:
        return None, None

    def recency_key(path: Path) -> tuple[int, int, str]:
        """Prefer the experiment most recently touched by the researcher."""

        try:
            latest_mtime = path.stat().st_mtime_ns
        except OSError:
            latest_mtime = 0
        contract_count = 0
        for filename in _DELIVERABLE_CONTRACT_FILES:
            contract = path / filename
            try:
                if contract.is_file() and not contract.is_symlink():
                    contract_count += 1
                    latest_mtime = max(latest_mtime, contract.stat().st_mtime_ns)
            except OSError:
                continue
        return latest_mtime, contract_count, path.name

    selected = max(candidates, key=recency_key)
    return selected, candidates[selected]


def _latest_experiment_root(root: Path, progress: dict[str, Any]) -> Path | None:
    """Return the selected experiment root for compatibility with local callers."""

    return _latest_experiment_selection(root, progress)[0]


def _latest_project_pdf(
    root: Path,
    *,
    experiment_root: Path | None,
    selected_result: dict[str, Any] | None,
) -> Path | None:
    """Return only a PDF attributable to the selected experiment/result."""

    if selected_result is not None:
        return _workspace_file(root, selected_result.get("pdf_path"))
    if experiment_root is not None:
        return None
    papers = root / "03_papers"
    if not papers.is_dir() or papers.is_symlink():
        return None
    try:
        candidates = sorted(
            (
                path
                for path in papers.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: path.name,
        )
    except OSError:
        return None
    return candidates[-1] if candidates else None


def _head_is_exact_research_checkpoint(
    root: Path,
    repo_status: dict[str, Any],
) -> bool:
    """Return whether HEAD itself is a hash-valid Research Git checkpoint."""

    head = str(repo_status.get("head") or "").strip()
    if not head:
        return False
    try:
        shown = show_checkpoint(root, head)
    except (OSError, ResearchGitError, ValueError):
        return False
    return bool(
        shown.get("checkpoint_hash_valid") is True
        and str(shown.get("commit") or "") == head
    )


def _git_tree_blobs(root: Path, commit: str, paths: list[str]) -> dict[str, str]:
    """Return Git blob identities for a bounded set of exact paths."""

    if not paths:
        return {}
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-z", commit, "--", *paths],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode:
        return {}
    blobs: dict[str, str] = {}
    for record in completed.stdout.split(b"\0"):
        if not record or b"\t" not in record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        fields = metadata.split()
        if len(fields) != 3 or fields[1] != b"blob":
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        blobs[path] = fields[2].decode("ascii", errors="ignore")
    return blobs


def _research_checkpoint_bindings(
    root: Path,
    *,
    repo_status: dict[str, Any],
    paths: list[str],
) -> dict[str, set[str]]:
    """Prove current path contents were recorded by an exact Research checkpoint.

    Being tracked by Git is insufficient: a raw Git commit followed by a semantic
    Research checkpoint must not retroactively bind unrelated artifacts.  The
    current blob therefore has to match a version named in ``changed_paths`` by
    a hash-valid checkpoint on HEAD's first-parent history.
    """

    head = str(repo_status.get("head") or "").strip()
    requested = sorted(
        {
            path
            for path in paths
            if path and not Path(path).is_absolute() and ".." not in Path(path).parts
        }
    )
    if (
        not head
        or not requested
        or not _head_is_exact_research_checkpoint(root, repo_status)
    ):
        return {}
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--first-parent",
                "--format=%H",
                f"--max-count={_MAX_CHECKPOINT_BINDING_COMMITS}",
                head,
                "--",
                *requested,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode:
        return {}

    current_blobs = _git_tree_blobs(root, head, requested)
    bindings: dict[str, set[str]] = {}
    requested_set = set(requested)
    for commit in completed.stdout.splitlines():
        commit = commit.strip()
        if not commit:
            continue
        try:
            shown = show_checkpoint(root, commit)
        except (OSError, ResearchGitError, ValueError):
            continue
        if (
            shown.get("checkpoint_hash_valid") is not True
            or str(shown.get("commit") or "") != commit
        ):
            continue
        checkpoint = shown.get("checkpoint")
        if not isinstance(checkpoint, dict):
            continue
        declared = requested_set & {
            str(path) for path in checkpoint.get("changed_paths") or []
        }
        checkpoint_blobs = _git_tree_blobs(root, commit, sorted(declared))
        object_refs = {
            str(item)
            for item in checkpoint.get("object_refs") or []
            if str(item).startswith("sha256:")
        }
        for path in declared:
            if current_blobs.get(path) and current_blobs[path] == checkpoint_blobs.get(
                path
            ):
                bindings.setdefault(path, set()).update(object_refs)
    return bindings


def _cas_logical_bindings(
    root: Path,
    *,
    repo_status: dict[str, Any] | None,
    pending_paths: set[str],
) -> dict[str, str]:
    """Map logical artifact paths to checkpointed or pending CAS pointers."""

    if repo_status is None:
        return {}
    pointer_root = root / "research-objects"
    if not pointer_root.is_dir() or pointer_root.is_symlink():
        return {}
    pointers: list[Path] = []
    try:
        with os.scandir(pointer_root) as entries:
            for entry in islice(entries, _MAX_POINTER_RECORDS):
                try:
                    if not entry.name.endswith(".json") or not entry.is_file(
                        follow_symlinks=False
                    ):
                        continue
                except OSError:
                    continue
                pointers.append(Path(entry.path))
    except OSError:
        return {}
    bindings: dict[str, str] = {}
    pointer_relatives: list[str] = []
    rows: list[tuple[str, str, str]] = []
    for pointer in pointers:
        payload, error = _read_json(pointer, root=root)
        object_hash = str(payload.get("object_hash") or "")
        logical_path = str(payload.get("logical_path") or "").strip()
        if (
            error
            or not logical_path
            or Path(logical_path).is_absolute()
            or ".." in Path(logical_path).parts
            or not object_hash.startswith("sha256:")
        ):
            continue
        relative = pointer.relative_to(root).as_posix()
        pointer_relatives.append(relative)
        rows.append((logical_path, object_hash, relative))
    checkpoint_bindings = _research_checkpoint_bindings(
        root,
        repo_status=repo_status,
        paths=pointer_relatives,
    )
    for logical_path, object_hash, pointer_relative in rows:
        if pointer_relative in pending_paths:
            bindings[logical_path] = "pending"
        elif object_hash in checkpoint_bindings.get(pointer_relative, set()):
            bindings[logical_path] = "checkpointed"
        else:
            bindings.setdefault(logical_path, "unbound")
    return bindings


def _deliverable_audit(
    root: Path,
    *,
    paths: list[Path],
    repo_status: dict[str, Any] | None,
) -> dict[str, Any]:
    relative_paths = sorted(
        {
            path.relative_to(root).as_posix()
            for path in paths
            if path.is_file() and root in path.parents
        }
    )
    if repo_status is None:
        return {
            "available": False,
            "research_checkpoint_head": None,
            "total": len(relative_paths),
            "checkpointed": 0,
            "pending": 0,
            "unbound": len(relative_paths),
            "checkpointed_paths": [],
            "pending_paths": [],
            "unbound_paths": relative_paths,
        }
    checkpoint_head_valid = _head_is_exact_research_checkpoint(root, repo_status)
    pending_set = {
        str(path)
        for path in (
            list(repo_status.get("staged_paths") or [])
            + list((repo_status.get("research_stage") or {}).get("paths") or [])
            + list(repo_status.get("tracked_changes") or [])
            + list(repo_status.get("eligible_changes") or [])
        )
    }
    checkpoint_bindings = _research_checkpoint_bindings(
        root,
        repo_status=repo_status,
        paths=relative_paths,
    )
    cas_bindings = _cas_logical_bindings(
        root,
        repo_status=repo_status,
        pending_paths=pending_set,
    )
    checkpointed: list[str] = []
    pending: list[str] = []
    unbound: list[str] = []
    for relative in relative_paths:
        if relative in pending_set or cas_bindings.get(relative) == "pending":
            pending.append(relative)
        elif checkpoint_head_valid and (
            relative in checkpoint_bindings
            or cas_bindings.get(relative) == "checkpointed"
        ):
            checkpointed.append(relative)
        else:
            unbound.append(relative)
    return {
        "available": True,
        "research_checkpoint_head": checkpoint_head_valid,
        "total": len(relative_paths),
        "checkpointed": len(checkpointed),
        "pending": len(pending),
        "unbound": len(unbound),
        "checkpointed_paths": checkpointed,
        "pending_paths": pending,
        "unbound_paths": unbound,
    }


def _deliverables_summary(
    root: Path,
    *,
    progress: dict[str, Any],
    research: dict[str, Any],
    review: dict[str, Any],
    repo_status: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Summarize experiments and paper readiness from existing local contracts."""

    experiment_root, selected_result = _latest_experiment_selection(root, progress)
    contract_root = experiment_root or root
    contract_paths = [
        contract_root / filename
        for filename in _DELIVERABLE_CONTRACT_FILES
        if (contract_root / filename).is_file()
        and not (contract_root / filename).is_symlink()
    ]
    manifest, manifest_error = _validated_state_json(
        contract_root / "pipeline_manifest.json",
        root=root,
        label="pipeline_manifest",
        mapping_fields=("artifacts",),
    )
    manuscript, manuscript_error = _validated_state_json(
        contract_root / "manuscript_state.json",
        root=root,
        label="manuscript_state",
        mapping_fields=("evidence_summary",),
    )
    figure_spec, figure_error = _validated_state_json(
        contract_root / "figure_spec.json",
        root=root,
        label="figure_spec",
        sequence_fields=("figures",),
        mapping_fields=("summary",),
        mapping_item_fields=("figures",),
    )
    review_state, review_error = _validated_state_json(
        contract_root / "review_state.json",
        root=root,
        label="review_state",
        sequence_fields=("active_issue_records", "active_issues"),
        mapping_fields=("repair_metrics", "lane_summaries"),
        mapping_item_fields=("active_issue_records", "active_issues"),
    )
    repair_plan, repair_error = _validated_state_json(
        contract_root / "repair_plan.json",
        root=root,
        label="repair_plan",
        mapping_fields=("summary",),
    )
    warnings: list[dict[str, str]] = []
    for path, error in (
        (contract_root / "pipeline_manifest.json", manifest_error),
        (contract_root / "manuscript_state.json", manuscript_error),
        (contract_root / "figure_spec.json", figure_error),
        (contract_root / "review_state.json", review_error),
        (contract_root / "repair_plan.json", repair_error),
    ):
        if error:
            warnings.append(
                {
                    "code": "deliverable_contract_unreadable",
                    "detail": f"{_workspace_state_label(path, root=root)}: {error}",
                    "remediation": "restore the contract from Research Git or rerun its producing stage",
                }
            )

    progress_results = [
        item for item in progress.get("results") or [] if isinstance(item, dict)
    ]
    run_states = [
        str(item.get("status") or "").strip().lower() for item in progress_results
    ]
    successful_runs = sum(
        state in _SUCCESSFUL_EXPERIMENT_STATES for state in run_states
    )
    failed_runs = sum(state in _FAILED_EXPERIMENT_STATES for state in run_states)
    counts = review.get("object_counts") or {}
    attempt_states = (research.get("guide") or {}).get("experiment_states") or {}
    recorded_attempts = max(
        _count(counts.get("experiment_attempt")),
        sum(_count(value) for value in attempt_states.values()),
    )
    recorded_successes = sum(
        _count(attempt_states.get(state)) for state in _SUCCESSFUL_EXPERIMENT_STATES
    )
    recorded_failures = sum(
        _count(attempt_states.get(state)) for state in _FAILED_EXPERIMENT_STATES
    )
    if not progress_results:
        successful_runs = recorded_successes
        failed_runs = recorded_failures
    else:
        # Project progress and Research Git can overlap, so do not add their
        # counts.  Never let a coarse successful project row hide a recorded
        # terminal failure, though: preserve at least the larger failure count.
        failed_runs = max(failed_runs, recorded_failures)
    manuscript_revisions = _count(counts.get("manuscript"))

    figure_summary = (
        figure_spec.get("summary")
        if isinstance(figure_spec.get("summary"), dict)
        else {}
    )
    figures = [
        item for item in figure_spec.get("figures") or [] if isinstance(item, dict)
    ]
    figure_count = _count(figure_summary.get("figure_count"), default=len(figures))
    ready_figure_count = _count(
        figure_summary.get("ready_figure_count")
        or sum(item.get("status") == "ready" for item in figures),
    )

    manuscript_evidence = (
        manuscript.get("evidence_summary")
        if isinstance(manuscript.get("evidence_summary"), dict)
        else {}
    )
    review_metrics = (
        review_state.get("repair_metrics")
        if isinstance(review_state.get("repair_metrics"), dict)
        else {}
    )
    active_issue_count = _count(
        review_metrics.get("active_issue_count")
        or len(review_state.get("active_issue_records") or [])
    )
    lane_summaries = (
        review_state.get("lane_summaries")
        if isinstance(review_state.get("lane_summaries"), dict)
        else {}
    )
    blocking_issue_count = sum(
        _count(item.get("blocking_issue_count"))
        for item in lane_summaries.values()
        if isinstance(item, dict)
    )
    repair_summary = (
        repair_plan.get("summary")
        if isinstance(repair_plan.get("summary"), dict)
        else {}
    )
    repair_task_count = _count(repair_summary.get("task_count"))
    repair_ready_count = _count(repair_summary.get("ready_task_count"))

    explicit_research_status = (
        str((selected_result or {}).get("research_status") or "").strip().lower()
    )
    guardrail_status = str(manuscript.get("guardrail_status") or "") or None
    if blocking_issue_count or active_issue_count or repair_task_count:
        paper_state = "revision_needed"
    elif guardrail_status == "blocked":
        paper_state = "evidence_blocked"
    elif explicit_research_status == "submission_ready":
        paper_state = "submission_ready"
    elif manuscript or manuscript_revisions:
        paper_state = "draft"
    else:
        paper_state = "not_started"

    source_path = _workspace_file(
        root,
        manuscript.get("latex_path"),
        base=contract_root,
    )
    pdf_path = _latest_project_pdf(
        root,
        experiment_root=experiment_root,
        selected_result=selected_result,
    )
    material_paths = list(contract_paths)
    if source_path is not None:
        material_paths.append(source_path)
    if pdf_path is not None:
        material_paths.append(pdf_path)
    audit = _deliverable_audit(
        root,
        paths=material_paths,
        repo_status=repo_status,
    )
    if audit["unbound"]:
        warnings.append(
            {
                "code": "deliverable_artifacts_unbound",
                "detail": (
                    f"{audit['unbound']} experiment or paper artifact(s) are not "
                    "bound to the current Research Git history"
                ),
                "remediation": (
                    "initialize Research Git before binding existing outputs"
                    if repo_status is None
                    else "bind large outputs through `xscientist research object add`, then save one checkpoint"
                ),
            }
        )
    artifact_entries = (
        manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    )
    return (
        {
            "available": bool(
                progress_results
                or recorded_attempts
                or manuscript_revisions
                or contract_paths
                or pdf_path
            ),
            "experiment_root": (
                portable_path(experiment_root, base=root)
                if experiment_root is not None
                else None
            ),
            "experiment": {
                "runs": max(
                    len(progress_results),
                    recorded_attempts,
                    successful_runs + failed_runs,
                ),
                "successful": successful_runs,
                "failed": failed_runs,
                "recorded_attempts": recorded_attempts,
                "terminal_states": dict(sorted(attempt_states.items())),
                "registry_status": (
                    (artifact_entries.get("experiment_registry") or {}).get("status")
                    if isinstance(artifact_entries.get("experiment_registry"), dict)
                    else None
                ),
            },
            "paper": {
                "state": paper_state,
                "manuscript_revisions": manuscript_revisions,
                "guardrail_status": guardrail_status,
                "claim_count": _count(manuscript_evidence.get("claim_count")),
                "supported_claim_count": _count(
                    manuscript_evidence.get("supported_claim_count")
                ),
                "figures": {
                    "total": figure_count,
                    "ready": ready_figure_count,
                },
                "review": {
                    "active_issues": active_issue_count,
                    "blocking_issues": blocking_issue_count,
                },
                "repair": {
                    "tasks": repair_task_count,
                    "ready": repair_ready_count,
                },
                "source": (
                    portable_path(source_path, base=root)
                    if source_path is not None
                    else None
                ),
                "pdf": (
                    portable_path(pdf_path, base=root) if pdf_path is not None else None
                ),
            },
            "audit": audit,
        },
        warnings,
    )


def _empty_closure_levels(status: str = "unavailable") -> dict[str, dict[str, Any]]:
    """Return a shape-stable closure ladder when no repository audit is available."""

    return {
        level: {
            "complete": False,
            "status": status,
            "claim_count": 0,
            "complete_claim_count": 0,
            "blocker_count": 0,
            "warning_count": 0,
            "blocker_codes": [],
            "warning_codes": [],
        }
        for level in ("trace", "replay", "verify")
    }


def _selected_background_run(root: Path) -> dict[str, Any] | None:
    candidates: list[tuple[str, dict[str, Any]]] = []
    runs_root = root / "04_logs" / "runs"
    if _state_path_error(runs_root, root=root) or not runs_root.is_dir():
        return None
    for path in runs_root.glob("*.json"):
        payload, error = _read_json(path, root=root)
        if error or payload.get("schema") != RUN_SCHEMA:
            continue
        candidates.append((path.name, payload))
    if not candidates:
        return None
    active_states = {"queued", "running", "cancelling"}
    _selected_name, selected = max(
        candidates,
        key=lambda item: (
            str(item[1].get("status") or "") in active_states,
            item[0],
        ),
    )
    return {
        "id": selected.get("id"),
        "status": selected.get("status"),
        "created_at": selected.get("created_at"),
        "finished_at": selected.get("finished_at"),
        "provider": selected.get("provider"),
        "model": selected.get("model"),
        "profile": selected.get("profile"),
        "returncode": selected.get("returncode"),
        "selection_basis": "active_state_then_stable_run_id",
    }


def _review_summary(root: Path, status: dict[str, Any]) -> dict[str, Any]:
    """Build one compact, GitHub-check-like view over existing scientific gates."""

    tracked = list(status.get("tracked_changes") or [])
    tracked_set = set(tracked)
    eligible = [
        path for path in status.get("eligible_changes") or [] if path not in tracked_set
    ]
    last_checkpoint = status.get("last_checkpoint") or {}
    base: dict[str, Any] = {
        "available": False,
        "head": status.get("head"),
        "clean": bool(status.get("worktree_clean")),
        "pending": {
            "backend_staged": list(status.get("staged_paths") or []),
            "selected": list((status.get("research_stage") or {}).get("paths") or []),
            "tracked": tracked,
            "eligible": eligible,
            "preserved": list(status.get("excluded_changes") or []),
        },
        "checks": {
            "trace": "unavailable",
            "replay": "unavailable",
            "verify": "unavailable",
        },
        "closure_levels": _empty_closure_levels(),
        "target_level": None,
        "commit": None,
        "blocker_count": 0,
        "warning_count": 0,
        "promotion_ready": False,
        "blocker_codes": [],
        "object_counts": {},
        "evolution": {"candidates": 0, "evaluations": 0, "gate_decisions": 0},
        "commands": {
            "audit": "xscientist audit . --level verify",
            "diff": (
                "xscientist history diff ."
                if last_checkpoint.get("parent_commit")
                else None
            ),
        },
        "error": None,
    }
    try:
        closure = audit_research_closure(
            root,
            ref=str(status.get("head") or "HEAD"),
            level="verify",
            verify_objects=False,
        )
    except (OSError, ResearchGitError, ValueError) as exc:
        base["error"] = str(exc)
        return base

    levels = summarize_closure_levels(closure)
    level_summary = closure_level_summary(closure)
    claim_count = len(closure.get("claims") or [])
    integrity_errors = list((closure.get("integrity") or {}).get("errors") or [])
    checks = {
        name: (
            "pass"
            if complete
            else "not_ready" if not claim_count and not integrity_errors else "pending"
        )
        for name, complete in levels.items()
    }
    counts = dict(closure.get("counts") or {})
    base.update(
        {
            "available": True,
            "checks": checks,
            "closure_levels": level_summary,
            "target_level": closure.get("target_level"),
            "commit": closure.get("commit"),
            "blocker_count": len(closure.get("blockers") or []),
            "warning_count": len(closure.get("warnings") or []),
            "promotion_ready": levels["verify"],
            "blocker_codes": sorted(
                {str(item.get("code") or "") for item in closure.get("blockers") or []}
                - {""}
            ),
            "object_counts": counts,
            "evolution": {
                "candidates": int(counts.get("agent_candidate") or 0),
                "evaluations": int(counts.get("agent_evaluation") or 0),
                "gate_decisions": int(counts.get("gate_decision") or 0),
            },
        }
    )
    return base


def _dag_view_status(
    root: Path,
    dag_path: Path | None,
    *,
    head: str | None,
) -> dict[str, Any]:
    if dag_path is None:
        return {
            "html": None,
            "json": None,
            "commit": None,
            "current": None,
            "warning": None,
            "refresh_command": None,
        }
    metadata_path = dag_path.with_name("research-dag.json")
    metadata, error = _read_json(metadata_path, root=root)
    recorded_commit = str(metadata.get("commit") or "") or None
    current = bool(recorded_commit and head and recorded_commit == head)
    warning = None
    if error:
        warning = f"generated DAG metadata is unreadable: {error}"
        current = False
    elif not recorded_commit:
        warning = "generated DAG does not record the checkpoint it represents"
    elif head and not current:
        warning = "generated DAG represents an older checkpoint"
    return {
        "html": portable_path(dag_path, base=root),
        "json": (
            portable_path(metadata_path, base=root) if metadata_path.is_file() else None
        ),
        "commit": recorded_commit,
        "current": current,
        "warning": warning,
        "refresh_command": "xscientist research dag --repo . --output research-dag",
    }


def build_workspace_status(
    workspace: str | Path | None = None,
    *,
    language: str = "auto",
) -> dict[str, Any]:
    root = (
        Path(workspace).expanduser().resolve()
        if workspace is not None
        else discover_workspace_root() or Path.cwd().resolve()
    )
    research_enabled = (root / "research.yaml").is_file()
    research: dict[str, Any] = {
        "initialized": research_enabled,
        "branch": None,
        "head": None,
        "staged": 0,
        "worktree_clean": None,
        "last_checkpoint": None,
        "guide": None,
    }
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    repo_status: dict[str, Any] | None = None
    if not root.exists():
        errors.append(
            {
                "code": "workspace_not_found",
                "detail": "the requested workspace path does not exist",
                "remediation": "check the workspace path and run the command again",
            }
        )
    elif not root.is_dir():
        errors.append(
            {
                "code": "workspace_not_directory",
                "detail": "the requested workspace path is not a directory",
                "remediation": "select a workspace directory and run the command again",
            }
        )
    if research_enabled and not errors:
        try:
            repo_status = repository_status(root)
            guide = build_research_guide(root, language=language, command_repo=".")
            research.update(
                {
                    "branch": repo_status.get("branch"),
                    "head": repo_status.get("head"),
                    "staged": len(repo_status.get("staged_paths") or [])
                    + len((repo_status.get("research_stage") or {}).get("paths") or []),
                    "worktree_clean": repo_status.get("worktree_clean"),
                    "last_checkpoint": repo_status.get("last_checkpoint"),
                    "guide": {
                        "progress": guide.get("progress"),
                        "counts": guide.get("counts"),
                        "experiment_states": guide.get("experiment_states"),
                        "next_steps": guide.get("next_steps"),
                        "warnings": guide.get("warnings"),
                        "program_review": guide.get("program_review"),
                    },
                }
            )
        except (OSError, ResearchGitError, ValueError) as exc:
            errors.append({"code": "research_status_unavailable", "detail": str(exc)})

    if (
        repo_status is not None
        and repo_status.get("head")
        and not _head_is_exact_research_checkpoint(root, repo_status)
    ):
        errors.append(
            {
                "code": "research_head_not_checkpointed",
                "detail": (
                    "Git HEAD is not an exact, hash-valid Research Git checkpoint; "
                    "ordinary Git commits cannot certify scientific artifacts"
                ),
                "remediation": (
                    "restore or switch to a valid Research Git checkpoint, or use "
                    "the explicit Research Git migration workflow"
                ),
            }
        )

    progress_path = root / "04_logs" / "progress.json"
    budget_path = root / "04_logs" / "llm_budget.json"
    insight_path = root / "04_logs" / "insight_report.json"
    readiness_path = root / ".xscientist" / "readiness.json"
    strategy_followups_path = root / "04_logs" / "research_strategy_followups.json"
    progress, progress_error = _validated_state_json(
        progress_path,
        root=root,
        label="progress",
        sequence_fields=("results", "selected_indices"),
        text_fields=("current_stage",),
        mapping_item_fields=("results",),
    )
    budget, budget_error = _validated_state_json(
        budget_path,
        root=root,
        label="llm_budget",
        mapping_fields=("limits", "used", "reserved"),
    )
    insight, insight_error = _validated_state_json(
        insight_path,
        root=root,
        label="insight_report",
        text_fields=("epistemic_status",),
    )
    readiness, readiness_error = _validated_state_json(
        readiness_path,
        root=root,
        label="readiness",
        sequence_fields=("remediations",),
        mapping_item_fields=("remediations",),
    )
    strategy_followups, strategy_followups_error = _validated_state_json(
        strategy_followups_path,
        root=root,
        label="research_strategy_followups",
        sequence_fields=("active", "queued"),
        mapping_item_fields=("active", "queued"),
    )
    for path, detail in (
        (progress_path, progress_error),
        (budget_path, budget_error),
        (insight_path, insight_error),
        (readiness_path, readiness_error),
        (strategy_followups_path, strategy_followups_error),
    ):
        if detail:
            errors.append(
                {
                    "code": "workspace_state_corrupted",
                    "file": _workspace_state_label(path, root=root),
                    "detail": detail,
                    "remediation": (
                        "restore this file from a known-good checkpoint or move it "
                        "aside, then run `xscientist status` again"
                    ),
                }
            )
    dag_path = _first_existing(
        [
            root / "research-dag" / "research-dag.html",
            root
            / "outputs"
            / "views"
            / root.name
            / "research-dag"
            / "research-dag.html",
        ],
        root=root,
    )
    review = (
        _review_summary(root, repo_status)
        if repo_status is not None
        else {
            "available": False,
            "head": None,
            "clean": None,
            "pending": {},
            "checks": {
                "trace": "unavailable",
                "replay": "unavailable",
                "verify": "unavailable",
            },
            "promotion_ready": False,
            "blocker_codes": [],
            "closure_levels": _empty_closure_levels(),
            "target_level": "unavailable",
            "commit": None,
            "blocker_count": 0,
            "warning_count": 0,
            "object_counts": {},
            "evolution": {
                "candidates": 0,
                "evaluations": 0,
                "gate_decisions": 0,
            },
            "commands": {"audit": None, "diff": None},
            "error": None,
        }
    )
    if review.get("error"):
        warnings.append(
            {
                "code": "review_status_unavailable",
                "detail": str(review["error"]),
                "remediation": "run `xscientist audit . --level trace` for details",
            }
        )
    deliverables, deliverable_warnings = _deliverables_summary(
        root,
        progress=progress,
        research=research,
        review=review,
        repo_status=repo_status,
    )
    warnings.extend(deliverable_warnings)
    deliverable_audit = deliverables["audit"]
    if repo_status is not None and (
        deliverable_audit.get("pending") or deliverable_audit.get("unbound")
    ):
        review["clean"] = False
    dag_view = _dag_view_status(root, dag_path, head=research.get("head"))
    if dag_view.get("warning"):
        warnings.append(
            {
                "code": "generated_view_stale",
                "detail": str(dag_view["warning"]),
                "remediation": str(dag_view["refresh_command"]),
            }
        )
    background_run = _selected_background_run(root)
    next_steps = list((research.get("guide") or {}).get("next_steps") or [])
    deliverable_step: dict[str, str] | None = None
    if not research_enabled and deliverables.get("available"):
        deliverable_step = {
            "code": "initialize_research_history",
            "title": "Initialize Research Git before binding existing outputs",
            "command": "xscientist research init .",
        }
    elif repo_status is not None and deliverable_audit.get("unbound_paths"):
        unbound_path = str(deliverable_audit["unbound_paths"][0])
        deliverable_step = {
            "code": "bind_research_output",
            "title": "Bind the latest experiment or manuscript output to Research Git",
            "command": (
                "xscientist research object add "
                f"{shlex.quote(unbound_path)} --repo . "
                f"--logical-path {shlex.quote(unbound_path)}"
            ),
        }
    elif repo_status is not None and deliverable_audit.get("pending"):
        deliverable_step = {
            "code": "checkpoint_research_outputs",
            "title": "Save the latest experiment and paper state as one checkpoint",
            "command": (
                "xscientist history save . "
                '-m "checkpoint experiment and manuscript state"'
            ),
        }
    readiness_blockers = [
        item
        for item in readiness.get("remediations") or []
        if isinstance(item, dict) and item.get("severity") == "error"
    ]
    current_program_review = (research.get("guide") or {}).get("program_review")
    open_strategy_gaps = {
        str(item.get("code") or "")
        for item in ((current_program_review or {}).get("gaps") or [])
        if isinstance(item, dict) and item.get("code")
    }
    queued_followups = [
        item
        for item in (
            strategy_followups.get("active") or strategy_followups.get("queued") or []
        )
        if isinstance(item, dict)
        and item.get("object_id")
        and (
            not item.get("gap")
            or not isinstance(current_program_review, dict)
            or item.get("gap") in open_strategy_gaps
        )
    ]
    if queued_followups:
        followup = queued_followups[0]
        next_steps.insert(
            0,
            {
                "code": "inspect_scientific_strategy_followup",
                "title": str(
                    followup.get("action")
                    or "Inspect the next bounded scientific strategy follow-up"
                ),
                "command": (
                    "xscientist research objects "
                    + str(followup["object_id"])
                    + " --repo ."
                ),
            },
        )
    if deliverable_step is not None:
        next_steps.insert(0, deliverable_step)
    if readiness_blockers:
        blocker = readiness_blockers[0]
        next_steps.insert(
            0,
            {
                "code": str(blocker.get("code") or "resolve_readiness_blocker"),
                "title": str(
                    blocker.get("detail") or "Resolve the latest readiness blocker"
                ),
                "command": str(blocker.get("command") or "xscientist doctor --deep"),
            },
        )
    if isinstance(background_run, dict) and background_run.get("status") in {
        "failed",
        "cancelled",
        "interrupted",
    }:
        run_id = str(background_run.get("id") or "").strip()
        if run_id:
            next_steps.insert(
                0,
                {
                    "code": "repair_failed_background_run",
                    "title": "Inspect and repair the latest failed background run",
                    "command": (f"xscientist runs show {run_id} --workspace ."),
                },
            )
    elif isinstance(background_run, dict) and background_run.get("status") in {
        "queued",
        "running",
        "cancelling",
    }:
        run_id = str(background_run.get("id") or "").strip()
        if run_id:
            next_steps.insert(
                0,
                {
                    "code": "watch_background_run",
                    "title": "Watch the active background run",
                    "command": (f"xscientist runs watch {run_id} --workspace ."),
                },
            )
    if not next_steps and not research_enabled and not errors:
        next_steps = [
            {
                "code": "start_research",
                "title": "Create an offline research history",
                "command": "xscientist demo ./xscientist-demo",
            }
        ]
    workspace_name, workspace_id = _workspace_identity(root)
    background_status = str((background_run or {}).get("status") or "")
    guide_counts = (research.get("guide") or {}).get("counts") or {}
    scientific_work_exists = any(
        _count(guide_counts.get(kind)) > 0
        for kind in (
            "experiment_attempt",
            "evidence",
            "passage_evidence",
            "inference",
            "review",
            "claim",
            "reproduction",
        )
    ) or bool(deliverables.get("available"))
    contract_attention = any(
        str(item.get("code") or "") == "deliverable_contract_unreadable"
        for item in warnings
        if isinstance(item, dict)
    )
    attention_required = bool(
        errors
        or readiness_blockers
        or contract_attention
        or background_status in {"failed", "cancelled", "interrupted"}
    )
    if errors:
        operational_state = "invalid"
    elif attention_required:
        operational_state = "needs_attention"
    elif background_status in {"queued", "running", "cancelling"}:
        operational_state = "running"
    elif progress.get("current_stage") == "complete":
        operational_state = (
            "complete"
            if insight.get("epistemic_status") == "verified"
            else "scientific_followup"
        )
    elif progress:
        operational_state = "running"
    elif scientific_work_exists:
        operational_state = (
            "complete"
            if insight.get("epistemic_status") == "verified"
            else "scientific_followup"
        )
    elif research_enabled:
        operational_state = "ready"
    else:
        operational_state = "not_started"
    payload = {
        "schema": STATUS_SCHEMA,
        "ok": not errors,
        "operational_state": operational_state,
        "attention_required": attention_required,
        "workspace": workspace_name,
        "workspace_id": workspace_id,
        "research": research,
        "run": {
            "started": bool(progress),
            "current_stage": progress.get("current_stage"),
            "completed": len(progress.get("results") or []),
            "selected": len(progress.get("selected_indices") or []),
            "progress_file": (
                portable_path(progress_path, base=root)
                if progress_path.is_file()
                else None
            ),
        },
        "background_run": background_run,
        "readiness": {
            "available": bool(readiness),
            "configuration_ready": readiness.get("configuration_ready"),
            "runtime_ready": readiness.get("runtime_ready"),
            "blockers": readiness_blockers,
            "file": (
                portable_path(readiness_path, base=root)
                if readiness_path.is_file()
                else None
            ),
        },
        "strategy_followups": {
            "available": bool(strategy_followups),
            "queued": queued_followups,
            "file": (
                portable_path(strategy_followups_path, base=root)
                if strategy_followups_path.is_file()
                else None
            ),
        },
        "deliverables": deliverables,
        "review": review,
        "budget": {
            "available": bool(budget),
            "limits": budget.get("limits"),
            "used": budget.get("used"),
            "reserved": budget.get("reserved"),
            "file": (
                portable_path(budget_path, base=root) if budget_path.is_file() else None
            ),
        },
        "result": {
            "insight_available": bool(insight),
            "epistemic_status": insight.get("epistemic_status"),
            "insight_file": (
                portable_path(insight_path, base=root)
                if insight_path.is_file()
                else None
            ),
            "dag_html": dag_view["html"],
            "dag_json": dag_view["json"],
            "dag_commit": dag_view["commit"],
            "dag_current": dag_view["current"],
            "dag_refresh_command": dag_view["refresh_command"],
        },
        "next_steps": next_steps,
        "errors": errors,
        "warnings": warnings,
        "host_paths_disclosed": False,
    }
    return redact_sensitive_payload(payload)


def public_workspace_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return status JSON with explicit, host-path-free next-action bindings."""

    safe = deepcopy(payload)
    safe["workspace_context"] = workspace_action_context()
    safe["next_steps"] = [
        public_workspace_action(step)
        for step in safe.get("next_steps") or []
        if isinstance(step, dict)
    ]
    research = safe.get("research")
    if isinstance(research, dict):
        guide = research.get("guide")
        if isinstance(guide, dict):
            guide["workspace_context"] = workspace_action_context()
            guide["next_steps"] = [
                public_workspace_action(step)
                for step in guide.get("next_steps") or []
                if isinstance(step, dict)
            ]
    return redact_sensitive_payload(safe)


__all__ = [
    "STATUS_SCHEMA",
    "build_workspace_status",
    "public_workspace_status_payload",
]
