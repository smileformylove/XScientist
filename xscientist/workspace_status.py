"""One compact, read-only status view for a research workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_scientist.utils.privacy import portable_path, redact_sensitive_payload

from .provider_config import discover_workspace_root
from .research_git import ResearchGitError, repository_status
from .research_journey import build_research_guide

STATUS_SCHEMA = "xscientist.workspace-status.v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


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
    if research_enabled:
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
    progress = _read_json(progress_path)
    budget = _read_json(budget_path)
    insight = _read_json(insight_path)
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
    if not next_steps and not research_enabled:
        next_steps = [
            {
                "code": "start_research",
                "title": "Create an offline research history",
                "command": "xscientist demo ./xscientist-demo",
            }
        ]
    payload = {
        "schema": STATUS_SCHEMA,
        "ok": not errors,
        "workspace": ".",
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
