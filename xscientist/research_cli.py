"""Command-line interface for local-first scientific Git history."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .research_git import (
    ResearchGitError,
    add_research_object,
    create_checkpoint,
    create_research_bundle,
    init_repository,
    repository_status,
    reproduce_checkpoint,
    restore_research_bundle,
    research_diff,
    research_log,
    show_checkpoint,
    verify_research_bundle,
    verify_research_repository,
)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _read_question(args: argparse.Namespace) -> str | None:
    if args.question and args.question_file:
        raise ResearchGitError("use only one of --question or --question-file")
    if args.question_file:
        path = Path(args.question_file).expanduser()
        if not path.is_file():
            raise ResearchGitError(f"question file not found: {path}")
        return path.read_text(encoding="utf-8")
    if args.question:
        return f"# Research question\n\n{args.question.strip()}\n"
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist research",
        description=(
            "Record scientific progress in a local Git repository. "
            "No server or remote is required and no command pushes automatically."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Initialize a local research repository."
    )
    init_parser.add_argument("path")
    init_parser.add_argument("--name")
    init_parser.add_argument("--question")
    init_parser.add_argument("--question-file")
    init_parser.add_argument(
        "--policy",
        choices=["manual", "stage", "milestone"],
        default="milestone",
    )
    init_parser.add_argument("--actor", default="xscientist")
    init_parser.add_argument("--git-user-name")
    init_parser.add_argument("--git-user-email")
    init_parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    init_parser.add_argument("--no-commit", action="store_true")
    init_parser.add_argument("--json", action="store_true", dest="as_json")

    status_parser = subparsers.add_parser(
        "status", help="Show research and storage status."
    )
    status_parser.add_argument("--repo", default=".")
    status_parser.add_argument("--json", action="store_true", dest="as_json")

    fsck_parser = subparsers.add_parser(
        "fsck", help="Verify checkpoints, ARA bindings, pointers, and CAS objects."
    )
    fsck_parser.add_argument("--repo", default=".")
    fsck_parser.add_argument("--commit", default="HEAD")
    fsck_parser.add_argument("--no-objects", action="store_true")
    fsck_parser.add_argument("--json", action="store_true", dest="as_json")

    checkpoint_parser = subparsers.add_parser(
        "checkpoint", help="Create a safe scientific checkpoint and local Git commit."
    )
    checkpoint_parser.add_argument("--repo", default=".")
    checkpoint_parser.add_argument("--stage", required=True)
    checkpoint_parser.add_argument("--subject", required=True)
    checkpoint_parser.add_argument("--summary", default="")
    checkpoint_parser.add_argument("--status", default="completed")
    checkpoint_parser.add_argument("--actor")
    checkpoint_parser.add_argument("--node", action="append", default=[])
    checkpoint_parser.add_argument("--claim", action="append", default=[])
    checkpoint_parser.add_argument("--ara", action="append", default=[])
    checkpoint_parser.add_argument("--object-ref", action="append", default=[])
    checkpoint_parser.add_argument("--reproduce")
    checkpoint_parser.add_argument("--include", action="append", default=[])
    checkpoint_parser.add_argument("--no-commit", action="store_true")
    checkpoint_parser.add_argument("--allow-checkpoint-only", action="store_true")
    checkpoint_parser.add_argument("--json", action="store_true", dest="as_json")

    log_parser = subparsers.add_parser("log", help="Show scientific Git history.")
    log_parser.add_argument("--repo", default=".")
    log_parser.add_argument("--limit", type=int, default=20)
    log_parser.add_argument("--json", action="store_true", dest="as_json")

    show_parser = subparsers.add_parser("show", help="Show a checkpoint at a commit.")
    show_parser.add_argument("commit", nargs="?", default="HEAD")
    show_parser.add_argument("--repo", default=".")
    show_parser.add_argument("--json", action="store_true", dest="as_json")

    diff_parser = subparsers.add_parser("diff", help="Compare two research commits.")
    diff_parser.add_argument("before", nargs="?", default="HEAD~1")
    diff_parser.add_argument("after", nargs="?", default="HEAD")
    diff_parser.add_argument("--repo", default=".")
    diff_parser.add_argument(
        "--deep",
        action="store_true",
        help="Compare structured scientific JSON fields up to the safety limit.",
    )
    diff_parser.add_argument("--json", action="store_true", dest="as_json")

    object_parser = subparsers.add_parser(
        "object", help="Register large evidence in the local content-addressed store."
    )
    object_subparsers = object_parser.add_subparsers(
        dest="object_command", required=True
    )
    object_add = object_subparsers.add_parser(
        "add", help="Add a file and write a Git-safe pointer."
    )
    object_add.add_argument("source")
    object_add.add_argument("--repo", default=".")
    object_add.add_argument("--logical-path")
    object_add.add_argument("--media-type")
    object_add.add_argument("--json", action="store_true", dest="as_json")

    bundle_parser = subparsers.add_parser(
        "bundle", help="Create, verify, or restore an offline research bundle."
    )
    bundle_parser.add_argument(
        "action",
        nargs="?",
        choices=["create", "verify", "restore"],
        default="create",
    )
    bundle_parser.add_argument("bundle_path", nargs="?")
    bundle_parser.add_argument("--repo", default=".")
    bundle_parser.add_argument("--dest")
    bundle_parser.add_argument(
        "--profile", choices=["index", "reproduce", "audit"], default="reproduce"
    )
    bundle_parser.add_argument("--allow-incomplete", action="store_true")
    bundle_parser.add_argument("--json", action="store_true", dest="as_json")

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Inspect or materialize a commit's reproduction closure."
    )
    reproduce_parser.add_argument("commit", nargs="?", default="HEAD")
    reproduce_parser.add_argument("--repo", default=".")
    reproduce_parser.add_argument("--dest")
    reproduce_parser.add_argument("--execute", action="store_true")
    reproduce_parser.add_argument("--timeout", type=int, default=600)
    reproduce_parser.add_argument(
        "--environment-policy",
        choices=["ignore", "warn", "strict"],
        default="warn",
    )
    reproduce_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _human_status(payload: dict[str, Any]) -> None:
    print(f"Repository:        {payload['repository']}")
    print(f"Branch:            {payload['branch']}")
    print(f"HEAD:              {payload['head'] or '(no commit)'}")
    print(f"Checkpoint policy: {payload['checkpoint_policy']}")
    print(f"Auto push:         {payload['auto_push']}")
    print(f"Eligible changes:  {len(payload['eligible_changes'])}")
    print(f"Excluded changes:  {len(payload['excluded_changes'])}")
    store = payload["object_store"]
    print(f"Local CAS:         {store['objects']} objects / {store['bytes']} bytes")
    previous = payload.get("last_checkpoint") or {}
    if previous:
        print(
            f"Last checkpoint:   {previous.get('checkpoint_id')} "
            f"({previous.get('stage')} / {previous.get('status')})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = init_repository(
                args.path,
                name=args.name,
                question=_read_question(args),
                policy=args.policy,
                actor=args.actor,
                git_user_name=args.git_user_name,
                git_user_email=args.git_user_email,
                max_file_bytes=args.max_file_bytes,
                commit=not args.no_commit,
            )
            payload = result.to_dict()
            if args.as_json:
                _print_json(payload)
            else:
                print(
                    f"Initialized local research repository: {Path(args.path).expanduser().resolve()}"
                )
                if result.commit:
                    print(
                        f"Initial checkpoint: {result.commit[:12]} {result.checkpoint_id}"
                    )
                print("Remote: none (XScientist never pushes automatically)")
            return 0

        if args.command == "status":
            payload = repository_status(args.repo)
            _print_json(payload) if args.as_json else _human_status(payload)
            return 0

        if args.command == "fsck":
            payload = verify_research_repository(
                args.repo,
                commit=args.commit,
                verify_objects=not args.no_objects,
            )
            if args.as_json:
                _print_json(payload)
            else:
                verdict = "ok" if payload["ok"] else "failed"
                print(f"Research repository: {verdict}")
                print(f"Commit:              {payload['commit']}")
                checked = payload["checked"]
                print(
                    "Checked:             "
                    f"{checked['checkpoints']} checkpoints, "
                    f"{checked['pointers']} pointers, "
                    f"{checked['objects']} objects, "
                    f"{checked['ara_manifests']} ARA manifests"
                )
                for warning in payload["warnings"]:
                    print(f"warning: {warning}", file=sys.stderr)
                for error in payload["errors"]:
                    print(f"error: {error}", file=sys.stderr)
            return 0 if payload["ok"] else 1

        if args.command == "checkpoint":
            result = create_checkpoint(
                args.repo,
                stage=args.stage,
                subject=args.subject,
                summary=args.summary,
                status=args.status,
                actor=args.actor,
                nodes=args.node,
                claims=args.claim,
                ara_paths=args.ara,
                object_refs=args.object_ref,
                reproduce_command=args.reproduce,
                include=args.include,
                commit=not args.no_commit,
                allow_checkpoint_only=args.allow_checkpoint_only,
            )
            payload = result.to_dict()
            if args.as_json:
                _print_json(payload)
            elif result.committed:
                print(
                    f"Checkpoint committed: {result.commit[:12]} {result.checkpoint_id}"
                )
                print(f"Stage: {args.stage}; files: {len(result.staged_paths)}")
            elif result.created:
                print(f"Checkpoint written: {result.checkpoint_path}")
            else:
                print(f"Checkpoint skipped: {result.reason}")
            if result.excluded_paths and not args.as_json:
                print("Excluded by safety policy:")
                for item in result.excluded_paths:
                    print(f"  - {item}")
            return 0

        if args.command == "log":
            payload = research_log(args.repo, limit=args.limit)
            if args.as_json:
                _print_json(payload)
            else:
                for entry in payload:
                    stage = (entry["trailers"].get("Research-Stage") or ["-"])[0]
                    print(
                        f"{entry['short_commit']} {entry['authored_at']} "
                        f"[{stage}] {entry['subject']}"
                    )
            return 0

        if args.command == "show":
            payload = show_checkpoint(args.repo, args.commit)
            if args.as_json:
                _print_json(payload)
            else:
                checkpoint = payload["checkpoint"]
                print(f"Commit:     {payload['commit']}")
                print(f"Checkpoint: {checkpoint.get('checkpoint_id')}")
                print(f"Stage:      {checkpoint.get('stage')}")
                print(f"State:      {checkpoint.get('status')}")
                print(f"Subject:    {checkpoint.get('subject')}")
                print(f"Summary:    {checkpoint.get('summary')}")
                if (checkpoint.get("reproduce") or {}).get("command"):
                    print(f"Reproduce:  {checkpoint['reproduce']['command']}")
            return 0

        if args.command == "diff":
            payload = research_diff(
                args.repo,
                args.before,
                args.after,
                deep=args.deep,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"{payload['before']}..{payload['after']}")
                for line in payload["changes"]:
                    print(line)
                if payload["stat"]:
                    print(payload["stat"])
                semantic = payload["semantic"]
                print(
                    "Scientific checkpoints: "
                    f"{semantic['before_checkpoint']['checkpoint_id']} -> "
                    f"{semantic['after_checkpoint']['checkpoint_id']}"
                )
                print(
                    "Claims: "
                    f"+{len(semantic['claims']['added'])} "
                    f"-{len(semantic['claims']['removed'])}; "
                    "nodes: "
                    f"+{len(semantic['nodes']['added'])} "
                    f"-{len(semantic['nodes']['removed'])}; "
                    "objects: "
                    f"+{len(semantic['objects']['added'])} "
                    f"-{len(semantic['objects']['removed'])}"
                )
                if args.deep:
                    print(
                        "Structured field changes: "
                        f"{len(semantic['structured_changes'])}"
                    )
            return 0

        if args.command == "object" and args.object_command == "add":
            result = add_research_object(
                args.repo,
                args.source,
                logical_path=args.logical_path,
                media_type=args.media_type,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                print(f"Object:  {result.object_hash}")
                print(f"Pointer: {result.pointer_path}")
                print(f"Store:   {result.store_path}")
                print("Run `xscientist research checkpoint` to commit the pointer.")
            return 0

        if args.command == "bundle":
            if args.action == "create":
                if not args.dest:
                    raise ResearchGitError("bundle create requires --dest")
                payload = create_research_bundle(
                    args.repo,
                    args.dest,
                    profile=args.profile,
                    allow_incomplete=args.allow_incomplete,
                )
            elif args.action == "verify":
                if not args.bundle_path:
                    raise ResearchGitError("bundle verify requires a bundle path")
                payload = verify_research_bundle(args.bundle_path)
            else:
                if not args.bundle_path or not args.dest:
                    raise ResearchGitError(
                        "bundle restore requires a bundle path and --dest"
                    )
                payload = restore_research_bundle(args.bundle_path, args.dest)
            if args.as_json:
                _print_json(payload)
            elif args.action == "verify":
                print(f"Bundle:   {payload['bundle']}")
                print(f"Valid:    {payload['ok']}")
                for warning in payload["warnings"]:
                    print(f"warning: {warning}", file=sys.stderr)
                for error in payload["errors"]:
                    print(f"error: {error}", file=sys.stderr)
            elif args.action == "restore":
                print(f"Bundle:     {payload['bundle']}")
                print(f"Repository: {payload['repository']}")
                print(f"HEAD:       {payload['commit']}")
                print(f"Objects:    {payload['objects_restored']}")
            else:
                print(f"Bundle:   {payload['destination']}")
                print(f"Profile:  {payload['profile']}")
                print(f"Complete: {payload['complete']}")
                print(f"HEAD:     {payload['repository_head']}")
            if args.action == "verify":
                return 0 if payload["ok"] else 1
            return 0

        if args.command == "reproduce":
            payload = reproduce_checkpoint(
                args.repo,
                commit=args.commit,
                destination=args.dest,
                execute=args.execute,
                timeout_seconds=args.timeout,
                environment_policy=args.environment_policy,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Commit:           {payload['commit']}")
                print(f"Checkpoint:       {payload['checkpoint']['checkpoint_id']}")
                print(f"Objects complete: {payload['objects_complete']}")
                print(f"Environment:      {payload['environment']['matches']}")
                print(f"Command:          {payload['command'] or '(not declared)'}")
                if (
                    args.environment_policy == "warn"
                    and payload["environment"]["mismatches"]
                ):
                    for mismatch in payload["environment"]["mismatches"]:
                        print(
                            f"warning: environment mismatch: {mismatch['field']}",
                            file=sys.stderr,
                        )
                if payload.get("worktree"):
                    print(f"Worktree:         {payload['worktree']}")
                if payload.get("executed"):
                    print(f"Return code:      {payload['returncode']}")
            return int(payload.get("returncode") or 0) if args.execute else 0
    except ResearchGitError as exc:
        print(f"research git error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
