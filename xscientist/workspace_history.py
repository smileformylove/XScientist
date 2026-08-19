"""Beginner-facing, append-only controls for local scientific history.

The lower-level Research VCS API remains the source of truth.  This module is
only a small workspace facade: it makes the safe defaults obvious without
duplicating Git or scientific-checkpoint logic.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from .research_git import (
    ResearchGitError,
    commit_research_stage,
    create_checkpoint,
    research_diff,
    research_log,
    repository_status,
    revert_research_checkpoint,
    show_checkpoint,
)

HISTORY_SCHEMA = "xscientist.workspace-history.v1"
ROLLBACK_PREVIEW_SCHEMA = "xscientist.rollback-preview.v1"


def inspect_workspace_history(
    workspace: str | Path,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Return a compact, payload-free view of a workspace's checkpoints."""

    status = repository_status(workspace)
    entries = research_log(workspace, limit=limit)
    return {
        "schema_version": HISTORY_SCHEMA,
        "workspace": status["repository"],
        "branch": status["branch"],
        "head": status["head"],
        "checkpoint_policy": status["checkpoint_policy"],
        "auto_push": status["auto_push"],
        "pending": {
            "backend_staged": list(status["staged_paths"]),
            "selected": list(status["research_stage"]["paths"]),
            "eligible": list(status["eligible_changes"]),
            "excluded": list(status["excluded_changes"]),
        },
        "entries": entries,
    }


def save_workspace_checkpoint(
    workspace: str | Path,
    *,
    message: str = "save current research state",
    summary: str = "",
    actor: str | None = None,
) -> dict[str, Any]:
    """Checkpoint selected changes, or all eligible changes when none are selected."""

    subject = message.strip()
    if not subject:
        raise ResearchGitError("checkpoint message must not be empty")
    status = repository_status(workspace)
    selected = list(status["research_stage"]["paths"])
    operation = commit_research_stage if selected else create_checkpoint
    result = operation(
        workspace,
        stage="manual",
        subject=subject,
        summary=summary.strip(),
        actor=actor,
    )
    return {
        "schema_version": "xscientist.workspace-checkpoint.v1",
        "workspace": status["repository"],
        "selection": "selected" if selected else "eligible",
        "checkpoint": result.to_dict(),
    }


def preview_workspace_rollback(
    workspace: str | Path,
    *,
    commit: str = "HEAD",
) -> dict[str, Any]:
    """Describe one append-only reversal without changing the workspace."""

    status = repository_status(workspace)
    shown = show_checkpoint(workspace, commit)
    checkpoint = shown["checkpoint"]
    ancestry = research_log(workspace, limit=2, ref=shown["commit"])
    is_initial_commit = len(ancestry) == 1
    backend_staged = list(status["staged_paths"])
    selected = list(status["research_stage"]["paths"])
    eligible = list(status["eligible_changes"])
    excluded = list(status["excluded_changes"])
    blockers: list[dict[str, str]] = []
    if backend_staged:
        blockers.append(
            {
                "code": "backend_stage_not_empty",
                "detail": "clear the Git index before changing scientific history",
            }
        )
    if selected:
        blockers.append(
            {
                "code": "research_stage_not_empty",
                "detail": "save or unstage the selected research changes first",
            }
        )
    if eligible:
        blockers.append(
            {
                "code": "unsaved_research_changes",
                "detail": "save or discard eligible research changes first",
            }
        )
    if excluded:
        blockers.append(
            {
                "code": "excluded_worktree_changes",
                "detail": (
                    "move or resolve untracked or policy-excluded files before "
                    "changing scientific history"
                ),
            }
        )
    if is_initial_commit:
        blockers.append(
            {
                "code": "initial_checkpoint",
                "detail": "the repository's initial checkpoint cannot be rolled back",
            }
        )
    if not shown["checkpoint_hash_valid"]:
        blockers.append(
            {
                "code": "invalid_checkpoint_hash",
                "detail": "repair or restore the damaged checkpoint before rollback",
            }
        )

    impact: dict[str, Any] | None = None
    if not is_initial_commit:
        impact = research_diff(
            workspace,
            f"{shown['commit']}^",
            shown["commit"],
            deep=False,
        )
    quoted_workspace = shlex.quote(str(Path(workspace).expanduser()))
    quoted_commit = shlex.quote(shown["commit"])
    return {
        "schema_version": ROLLBACK_PREVIEW_SCHEMA,
        "workspace": status["repository"],
        "mode": "append_only_revert",
        "history_rewritten": False,
        "target": {
            "commit": shown["commit"],
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "stage": checkpoint.get("stage"),
            "status": checkpoint.get("status"),
            "subject": checkpoint.get("subject"),
            "hash_valid": shown["checkpoint_hash_valid"],
        },
        "impact": (
            {
                "changes": impact["changes"],
                "stat": impact["stat"],
                "claims": impact["semantic"]["claims"],
                "research_objects": impact["semantic"]["research_objects"],
            }
            if impact is not None
            else None
        ),
        "ready_to_apply": not blockers,
        "blockers": blockers,
        "limitations": [
            "the preview validates local preconditions but cannot promise that an "
            "older reversal will be conflict-free; apply reports conflicts without "
            "discarding current history"
        ],
        "apply_command": (
            "xscientist history rollback "
            f"{quoted_workspace} --commit {quoted_commit} --apply"
        ),
    }


def rollback_workspace_checkpoint(
    workspace: str | Path,
    *,
    commit: str = "HEAD",
    message: str | None = None,
) -> dict[str, Any]:
    """Append a reversal checkpoint after enforcing the preview's blockers."""

    preview = preview_workspace_rollback(workspace, commit=commit)
    if not preview["ready_to_apply"]:
        details = "; ".join(item["detail"] for item in preview["blockers"])
        raise ResearchGitError(f"rollback is blocked: {details}")
    result = revert_research_checkpoint(
        workspace,
        preview["target"]["commit"],
        subject=message,
    )
    return {
        "schema_version": "xscientist.workspace-rollback.v1",
        "workspace": preview["workspace"],
        "mode": "append_only_revert",
        "history_rewritten": False,
        "preview": preview,
        "result": result,
    }


def run_history_cli(parsed: argparse.Namespace) -> int:
    """Render the top-level history facade without growing the root CLI."""

    try:
        if parsed.history_command == "list":
            payload = inspect_workspace_history(parsed.workspace, limit=parsed.limit)
        elif parsed.history_command == "save":
            payload = save_workspace_checkpoint(
                parsed.workspace,
                message=parsed.message,
                summary=parsed.summary,
                actor=parsed.actor,
            )
        elif parsed.apply:
            payload = rollback_workspace_checkpoint(
                parsed.workspace,
                commit=parsed.commit,
                message=parsed.message,
            )
        else:
            payload = preview_workspace_rollback(
                parsed.workspace,
                commit=parsed.commit,
            )
    except (OSError, ResearchGitError, ValueError) as exc:
        if parsed.as_json:
            print(
                json.dumps(
                    {
                        "schema_version": "xscientist.error.v1",
                        "ok": False,
                        "error": {
                            "command": f"history {parsed.history_command}",
                            "message": str(exc),
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(f"xscientist history: {exc}", file=sys.stderr)
        return 2
    if parsed.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if parsed.history_command == "list":
        print(f"Workspace: {payload['workspace']}")
        print(f"Branch:    {payload['branch']}")
        pending = payload["pending"]
        print(
            "Unsaved:   "
            f"{len(pending['selected'])} selected / "
            f"{len(pending['eligible'])} eligible / "
            f"{len(pending['excluded'])} excluded / "
            f"{len(pending['backend_staged'])} backend-staged"
        )
        for entry in payload["entries"]:
            stage = (entry["trailers"].get("Research-Stage") or ["-"])[0]
            print(
                f"{entry['short_commit']} {entry['authored_at']} "
                f"[{stage}] {entry['subject']}"
            )
    elif parsed.history_command == "save":
        checkpoint = payload["checkpoint"]
        if checkpoint["committed"]:
            print(
                f"Saved checkpoint: {checkpoint['checkpoint_id']} "
                f"({checkpoint['commit'][:12]})"
            )
            print(f"Selection:        {payload['selection']}")
            print(f"Files:            {len(checkpoint['staged_paths'])}")
        else:
            print(f"Nothing saved: {checkpoint['reason'] or 'no eligible changes'}")
    elif parsed.apply:
        result = payload["result"]
        print("Rollback recorded; existing history was not rewritten.")
        print(f"Reversed commit:  {result['reverted']}")
        print(f"New checkpoint:   {result['checkpoint']['checkpoint_id']}")
    else:
        target = payload["target"]
        print("Rollback preview only; no files were changed.")
        print(f"Commit:     {target['commit']}")
        print(f"Checkpoint: {target['checkpoint_id']}")
        print(f"Stage:      {target['stage']}")
        print(f"Subject:    {target['subject']}")
        impact = payload.get("impact") or {}
        print(f"Files:      {len(impact.get('changes') or [])}")
        if payload["blockers"]:
            print("Blocked:")
            for blocker in payload["blockers"]:
                print(f"  {blocker['code']}: {blocker['detail']}")
        elif payload["ready_to_apply"]:
            print(f"Apply:      {payload['apply_command']}")
    return 0


__all__ = [
    "inspect_workspace_history",
    "preview_workspace_rollback",
    "rollback_workspace_checkpoint",
    "run_history_cli",
    "save_workspace_checkpoint",
]
