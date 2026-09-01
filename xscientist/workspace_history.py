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
from copy import deepcopy
from pathlib import Path
from typing import Any

from ai_scientist.utils.privacy import redact_sensitive_payload, redact_sensitive_text

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
CHECKPOINT_SCHEMA = "xscientist.workspace-checkpoint-inspection.v1"
DIFF_SCHEMA = "xscientist.workspace-history-diff.v1"
ROLLBACK_PREVIEW_SCHEMA = "xscientist.rollback-preview.v1"


def _workspace_literals(
    workspace: str | Path,
    payload: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return concrete workspace spellings that must not cross the CLI boundary."""

    values = {str(Path(workspace).expanduser())}
    try:
        values.add(str(Path(workspace).expanduser().resolve()))
    except OSError:
        pass
    if isinstance(payload, dict):
        published = payload.get("workspace")
        if isinstance(published, (str, Path)):
            values.add(str(published))
    return tuple(
        sorted(
            (value for value in values if Path(value).is_absolute()),
            key=len,
            reverse=True,
        )
    )


def _portable_history_text(value: Any, *, workspace_literals: tuple[str, ...]) -> str:
    """Replace this invocation's workspace before applying general redaction."""

    safe = str(value)
    for literal in workspace_literals:
        # Commands may contain a shell-quoted path.  Replacing the quoted form
        # first avoids publishing awkward ``'.'`` action arguments.
        quoted = shlex.quote(literal)
        if quoted != literal:
            safe = safe.replace(quoted, ".")
        safe = safe.replace(f'"{literal}"', ".")
        safe = safe.replace(f"'{literal}'", ".")
        safe = safe.replace(literal, ".")
    return redact_sensitive_text(safe)


def public_workspace_history_payload(
    payload: dict[str, Any],
    *,
    workspace: str | Path,
) -> dict[str, Any]:
    """Return a portable, recursively redacted history CLI payload."""

    workspace_literals = _workspace_literals(workspace, payload)

    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                _portable_history_text(
                    key, workspace_literals=workspace_literals
                ): transform(nested)
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [transform(nested) for nested in value]
        if isinstance(value, tuple):
            return tuple(transform(nested) for nested in value)
        if isinstance(value, (str, Path)):
            return _portable_history_text(value, workspace_literals=workspace_literals)
        return value

    safe = transform(deepcopy(payload))
    if "workspace" in safe:
        safe["workspace"] = "."
    return redact_sensitive_payload(safe)


def inspect_workspace_history(
    workspace: str | Path,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """Return a compact, payload-free view of a workspace's checkpoints."""

    status = repository_status(workspace)
    entries = research_log(workspace, limit=limit)
    tracked = list(status.get("tracked_changes") or [])
    eligible = [path for path in status["eligible_changes"] if path not in set(tracked)]
    return {
        "schema_version": HISTORY_SCHEMA,
        "workspace": status["repository"],
        "branch": status["branch"],
        "head": status["head"],
        "checkpoint_policy": status["checkpoint_policy"],
        "auto_push": status["auto_push"],
        "clean": bool(status.get("worktree_clean")),
        "pending": {
            "backend_staged": list(status["staged_paths"]),
            "selected": list(status["research_stage"]["paths"]),
            "tracked": tracked,
            "eligible": eligible,
            "preserved": list(status["excluded_changes"]),
            # Compatibility alias for clients of the 0.1 facade.
            "excluded": list(status["excluded_changes"]),
        },
        "entries": entries,
    }


def inspect_workspace_checkpoint(
    workspace: str | Path,
    *,
    commit: str = "HEAD",
) -> dict[str, Any]:
    """Return one hash-checked checkpoint without exposing research payloads."""

    shown = show_checkpoint(workspace, commit)
    checkpoint = shown["checkpoint"]
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "commit": shown["commit"],
        "checkpoint_hash_valid": shown["checkpoint_hash_valid"],
        "checkpoint": {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "sequence": checkpoint.get("sequence"),
            "stage": checkpoint.get("stage"),
            "status": checkpoint.get("status"),
            "subject": checkpoint.get("subject"),
            "summary": checkpoint.get("summary"),
            "actor": checkpoint.get("actor"),
            "branch": checkpoint.get("branch"),
            "created_at": checkpoint.get("created_at"),
            "parent_commit": checkpoint.get("parent_commit"),
            "changed_paths": list(checkpoint.get("changed_paths") or []),
            "object_refs": list(checkpoint.get("object_refs") or []),
            "claims": list(checkpoint.get("claims") or []),
            "reproduce": checkpoint.get("reproduce"),
            "content_hash": checkpoint.get("content_hash"),
        },
        "payloads_disclosed": False,
    }


def compare_workspace_history(
    workspace: str | Path,
    *,
    before: str = "HEAD^",
    after: str = "HEAD",
    deep: bool = False,
) -> dict[str, Any]:
    """Return a compact scientific diff suitable for review before promotion."""

    diff = research_diff(workspace, before, after, deep=deep)
    semantic = diff["semantic"]
    return {
        "schema_version": DIFF_SCHEMA,
        "before": diff["before"],
        "after": diff["after"],
        "changes": list(diff["changes"]),
        "stat": diff["stat"],
        "checkpoint": {
            "before": semantic.get("before_checkpoint"),
            "after": semantic.get("after_checkpoint"),
            "fields": semantic.get("fields") or {},
        },
        "claims": semantic.get("claims") or {},
        "research_objects": semantic.get("research_objects") or {},
        "ara_manifests": semantic.get("ara_manifests") or {},
        "environment_changed": bool(semantic.get("environment_changed")),
        "structured_changes": list(semantic.get("structured_changes") or []),
        "warnings": list(semantic.get("warnings") or []),
        "payloads_disclosed": False,
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
    tracked = list(status.get("tracked_changes") or [])
    eligible = [path for path in status["eligible_changes"] if path not in set(tracked)]
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
    if tracked:
        blockers.append(
            {
                "code": "tracked_worktree_changes",
                "detail": "save or restore tracked research changes first",
            }
        )
    if eligible:
        blockers.append(
            {
                "code": "unsaved_research_changes",
                "detail": "save or discard eligible research changes first",
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
        "preserved": {
            "policy_excluded": excluded,
            "count": len(excluded),
        },
        "limitations": [
            "the preview validates local preconditions but cannot promise that an "
            "older reversal will be conflict-free; apply reports conflicts without "
            "discarding current history",
            "policy-excluded local files are preserved and generated views may need "
            "to be refreshed after rollback",
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
    workspace_path = Path(workspace).expanduser()
    quoted_workspace = shlex.quote(str(workspace_path))
    quoted_view = shlex.quote(str(workspace_path / "research-dag"))
    return {
        "schema_version": "xscientist.workspace-rollback.v1",
        "workspace": preview["workspace"],
        "mode": "append_only_revert",
        "history_rewritten": False,
        "preview": preview,
        "result": result,
        "next_actions": [
            f"xscientist status {quoted_workspace}",
            (
                f"xscientist research dag --repo {quoted_workspace} "
                f"--output {quoted_view}"
            ),
        ],
    }


def run_history_cli(parsed: argparse.Namespace) -> int:
    """Render the top-level history facade without growing the root CLI."""

    try:
        if parsed.history_command == "list":
            payload = inspect_workspace_history(parsed.workspace, limit=parsed.limit)
        elif parsed.history_command == "show":
            payload = inspect_workspace_checkpoint(
                parsed.workspace,
                commit=parsed.commit,
            )
        elif parsed.history_command == "diff":
            payload = compare_workspace_history(
                parsed.workspace,
                before=parsed.before,
                after=parsed.after,
                deep=parsed.deep,
            )
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
        workspace_literals = _workspace_literals(parsed.workspace)
        safe_error = _portable_history_text(
            exc,
            workspace_literals=workspace_literals,
        )
        if parsed.as_json:
            print(
                json.dumps(
                    {
                        "schema_version": "xscientist.error.v1",
                        "ok": False,
                        "error": {
                            "command": f"history {parsed.history_command}",
                            "message": safe_error,
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(f"xscientist history: {safe_error}", file=sys.stderr)
        return 2
    payload = public_workspace_history_payload(payload, workspace=parsed.workspace)
    if parsed.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if parsed.history_command == "list":
        print(f"Workspace: {payload['workspace']}")
        print(f"Branch:    {payload['branch']}")
        pending = payload["pending"]
        blocking_count = sum(
            len(pending[name])
            for name in ("selected", "tracked", "eligible", "backend_staged")
        )
        print(
            "Research changes: "
            + (
                "clean"
                if not blocking_count
                else (
                    f"{len(pending['selected'])} selected / "
                    f"{len(pending['tracked'])} tracked / "
                    f"{len(pending['eligible'])} eligible / "
                    f"{len(pending['backend_staged'])} backend-staged"
                )
            )
        )
        if pending["preserved"]:
            print(f"Preserved local views/files: {len(pending['preserved'])}")
        for entry in payload["entries"]:
            stage = (entry["trailers"].get("Research-Stage") or ["-"])[0]
            print(
                f"{entry['short_commit']} {entry['authored_at']} "
                f"[{stage}] {entry['subject']}"
            )
    elif parsed.history_command == "show":
        checkpoint = payload["checkpoint"]
        print(
            f"Checkpoint: {checkpoint['checkpoint_id']} " f"({payload['commit'][:12]})"
        )
        print(f"Stage/state: {checkpoint['stage']} / {checkpoint['status']}")
        print(f"Subject:     {checkpoint['subject']}")
        print(f"Actor:       {checkpoint['actor']}")
        print(f"Files:       {len(checkpoint['changed_paths'])}")
        print(
            "Hash:        "
            + ("valid" if payload["checkpoint_hash_valid"] else "INVALID")
        )
        reproduce = checkpoint.get("reproduce") or {}
        if reproduce.get("command"):
            print(f"Reproduce:   {reproduce['command']}")
    elif parsed.history_command == "diff":
        print(f"Compare: {payload['before'][:12]}..{payload['after'][:12]}")
        print(f"Files:   {len(payload['changes'])}")
        fields = payload["checkpoint"]["fields"]
        for name in ("stage", "status", "subject"):
            field = fields.get(name) or {}
            if field.get("changed"):
                print(f"{name.title()}: {field.get('before')} -> {field.get('after')}")
        for label, key in (("Claims", "claims"), ("Objects", "research_objects")):
            section = payload[key]
            print(
                f"{label}:  +{len(section.get('added') or [])} "
                f"-{len(section.get('removed') or [])}"
            )
        print(f"Environment changed: {payload['environment_changed']}")
        for change in payload["changes"][:10]:
            print(f"  {change}")
        if len(payload["changes"]) > 10:
            print(f"  ... {len(payload['changes']) - 10} more")
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
        print(f"Inspect:          {payload['next_actions'][0]}")
        print(f"Refresh view:     {payload['next_actions'][1]}")
    else:
        target = payload["target"]
        print("Rollback preview only; no files were changed.")
        print(f"Commit:     {target['commit']}")
        print(f"Checkpoint: {target['checkpoint_id']}")
        print(f"Stage:      {target['stage']}")
        print(f"Subject:    {target['subject']}")
        impact = payload.get("impact") or {}
        print(f"Files:      {len(impact.get('changes') or [])}")
        preserved = payload.get("preserved") or {}
        if preserved.get("count"):
            print(
                "Preserved:  "
                f"{preserved['count']} policy-excluded local view/file(s)"
            )
        if payload["blockers"]:
            print("Blocked:")
            for blocker in payload["blockers"]:
                print(f"  {blocker['code']}: {blocker['detail']}")
        elif payload["ready_to_apply"]:
            print(f"Apply:      {payload['apply_command']}")
    return 0


__all__ = [
    "compare_workspace_history",
    "inspect_workspace_checkpoint",
    "inspect_workspace_history",
    "preview_workspace_rollback",
    "public_workspace_history_payload",
    "rollback_workspace_checkpoint",
    "run_history_cli",
    "save_workspace_checkpoint",
]
