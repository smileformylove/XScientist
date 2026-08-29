"""One compact, read-only status view for a research workspace."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
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
from .research_git import ResearchGitError, repository_status
from .research_journey import (
    build_research_guide,
    public_workspace_action,
    workspace_action_context,
)

STATUS_SCHEMA = "xscientist.workspace-status.v1"
RUN_SCHEMA = "xscientist.local-run.v1"


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
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


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


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


def _latest_background_run(root: Path) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for path in (root / "04_logs" / "runs").glob("*.json"):
        payload, error = _read_json(path)
        if error or payload.get("schema") != RUN_SCHEMA:
            continue
        candidates.append(payload)
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: str(item.get("created_at") or ""))
    return {
        "id": latest.get("id"),
        "status": latest.get("status"),
        "created_at": latest.get("created_at"),
        "finished_at": latest.get("finished_at"),
        "provider": latest.get("provider"),
        "model": latest.get("model"),
        "profile": latest.get("profile"),
        "returncode": latest.get("returncode"),
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
    metadata, error = _read_json(metadata_path)
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
                        "next_steps": guide.get("next_steps"),
                        "warnings": guide.get("warnings"),
                        "program_review": guide.get("program_review"),
                    },
                }
            )
        except (OSError, ResearchGitError, ValueError) as exc:
            errors.append({"code": "research_status_unavailable", "detail": str(exc)})

    progress_path = root / "04_logs" / "progress.json"
    budget_path = root / "04_logs" / "llm_budget.json"
    insight_path = root / "04_logs" / "insight_report.json"
    readiness_path = root / ".xscientist" / "readiness.json"
    strategy_followups_path = root / "04_logs" / "research_strategy_followups.json"
    progress, progress_error = _read_json(progress_path)
    budget, budget_error = _read_json(budget_path)
    insight, insight_error = _read_json(insight_path)
    readiness, readiness_error = _read_json(readiness_path)
    strategy_followups, strategy_followups_error = _read_json(strategy_followups_path)
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
                    "file": portable_path(path, base=root),
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
        ]
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
    dag_view = _dag_view_status(root, dag_path, head=research.get("head"))
    if dag_view.get("warning"):
        warnings.append(
            {
                "code": "generated_view_stale",
                "detail": str(dag_view["warning"]),
                "remediation": str(dag_view["refresh_command"]),
            }
        )
    background_run = _latest_background_run(root)
    next_steps = list((research.get("guide") or {}).get("next_steps") or [])
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
    guide_progress = (research.get("guide") or {}).get("progress") or {}
    scientific_work_exists = int(guide_progress.get("completed_stages") or 0) > 0
    attention_required = bool(
        errors
        or readiness_blockers
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
