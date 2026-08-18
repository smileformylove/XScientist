"""One compact, read-only status view for a research workspace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ai_scientist.utils.privacy import portable_path, redact_sensitive_payload

from .provider_config import discover_workspace_root
from .research_git import ResearchGitError, repository_status
from .research_journey import build_research_guide

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
        "last_checkpoint": None,
        "guide": None,
    }
    errors: list[dict[str, str]] = []
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
            guide = build_research_guide(root, language=language)
            research.update(
                {
                    "branch": repo_status.get("branch"),
                    "head": repo_status.get("head"),
                    "staged": repo_status.get("staged_count", 0),
                    "last_checkpoint": repo_status.get("last_checkpoint"),
                    "guide": {
                        "progress": guide.get("progress"),
                        "next_steps": guide.get("next_steps"),
                        "warnings": guide.get("warnings"),
                    },
                }
            )
        except (OSError, ResearchGitError, ValueError) as exc:
            errors.append({"code": "research_status_unavailable", "detail": str(exc)})

    progress_path = root / "04_logs" / "progress.json"
    budget_path = root / "04_logs" / "llm_budget.json"
    insight_path = root / "04_logs" / "insight_report.json"
    progress, progress_error = _read_json(progress_path)
    budget, budget_error = _read_json(budget_path)
    insight, insight_error = _read_json(insight_path)
    for path, detail in (
        (progress_path, progress_error),
        (budget_path, budget_error),
        (insight_path, insight_error),
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
    next_steps = list((research.get("guide") or {}).get("next_steps") or [])
    if not next_steps and not research_enabled and not errors:
        next_steps = [
            {
                "code": "start_research",
                "title": "Create an offline research history",
                "command": "xscientist demo ./xscientist-demo",
            }
        ]
    workspace_name, workspace_id = _workspace_identity(root)
    payload = {
        "schema": STATUS_SCHEMA,
        "ok": not errors,
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
        "background_run": _latest_background_run(root),
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
            "dag_html": (
                portable_path(dag_path, base=root) if dag_path is not None else None
            ),
        },
        "next_steps": next_steps,
        "errors": errors,
        "host_paths_disclosed": False,
    }
    return redact_sensitive_payload(payload)


__all__ = ["STATUS_SCHEMA", "build_workspace_status"]
