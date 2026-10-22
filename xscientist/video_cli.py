"""Video asset/evidence commands registered by the native research CLI."""

from __future__ import annotations

import argparse
from typing import Any


def add_video_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "video", help="Manage CAS video assets and timestamped evidence."
    )
    commands = parser.add_subparsers(dest="video_command", required=True)
    add = commands.add_parser("add", help="Register an immutable local video asset.")
    add.add_argument("source")
    add.add_argument("--title", default="")
    listing = commands.add_parser("list", help="List video asset indexes.")
    observe = commands.add_parser(
        "observe", help="Bind an observation to sampled frames in a video interval."
    )
    observe.add_argument("--asset", required=True)
    observe.add_argument("--start", required=True, type=float)
    observe.add_argument("--end", required=True, type=float)
    observe.add_argument("--result", required=True)
    observe.add_argument("--max-frames", type=int, default=4)
    observe.add_argument(
        "--refresh",
        action="store_true",
        help="Decode instead of reusing verified frames.",
    )
    observe.add_argument("--supports", action="append", default=[])
    observe.add_argument("--refutes", action="append", default=[])
    context = commands.add_parser(
        "context", help="Compile bounded, timestamped video evidence references."
    )
    context.add_argument("--evidence", action="append", required=True)
    context.add_argument("--max-frames", type=int, default=8)
    context.add_argument("--budget-tokens", type=int, default=2000)
    text = commands.add_parser(
        "index-text",
        help="Index normalized timed transcript, OCR, caption or shot text.",
    )
    text.add_argument("source")
    text.add_argument("--asset", required=True)
    text.add_argument(
        "--kind", required=True, choices=("transcript", "ocr", "caption", "shot")
    )
    text.add_argument("--producer", required=True)
    text.add_argument("--producer-version", required=True)
    search = commands.add_parser(
        "search", help="Retrieve content-bound video intervals offline."
    )
    search.add_argument("query")
    search.add_argument("--asset", action="append", default=[])
    search.add_argument("--target")
    search.add_argument("--start", type=float)
    search.add_argument("--end", type=float)
    search.add_argument("--limit", type=int, default=8)
    plan = commands.add_parser(
        "plan", help="Plan bounded video inspection without executing it."
    )
    plan.add_argument("query")
    plan.add_argument("--asset", required=True)
    plan.add_argument("--start", type=float, default=0)
    plan.add_argument("--end", type=float)
    plan.add_argument("--target")
    plan.add_argument("--max-frames", type=int, default=8)
    plan.add_argument("--max-decodes", type=int, default=4)
    inspect = commands.add_parser(
        "inspect",
        help="Call a configured VLM on evidence and record its actual frame inputs.",
    )
    inspect.add_argument("--evidence", action="append", required=True)
    inspect.add_argument("--message", required=True)
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--max-frames", type=int, default=8)
    inspect.add_argument("--budget-tokens", type=int, default=4000)
    for command in (context, inspect):
        command.add_argument(
            "--frames-file",
            help="JSON list of exact evidence/timestamp/hash references.",
        )
    feedback = commands.add_parser(
        "feedback", help="Summarize host-recorded video inspection events."
    )
    feedback.add_argument("--asset", action="append", default=[])
    feedback.add_argument("--limit", type=int, default=128)
    run = commands.add_parser(
        "run",
        help="Retrieve, prepare bounded visual inputs, call a VLM and record execution feedback.",
    )
    run.add_argument("query")
    run.add_argument("--asset", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--start", type=float, default=0)
    run.add_argument("--end", type=float)
    run.add_argument("--target")
    run.add_argument("--max-frames", type=int, default=8)
    run.add_argument("--max-decodes", type=int, default=4)
    for command in (
        add,
        listing,
        observe,
        context,
        text,
        search,
        plan,
        inspect,
        feedback,
        run,
    ):
        command.add_argument("--repo", default=".")
        command.add_argument("--json", action="store_true", dest="as_json")
    for command in (add, observe, text, inspect, run):
        command.add_argument("--no-commit", action="store_true")


def run_video_command(args: argparse.Namespace) -> dict[str, Any]:
    from .video_evidence import (
        build_video_evidence_pack,
        list_video_assets,
        record_video_evidence,
        register_video_asset,
    )

    if args.video_command == "add":
        return register_video_asset(
            args.repo, args.source, title=args.title, commit=not args.no_commit
        )
    if args.video_command == "observe":
        return record_video_evidence(
            args.repo,
            args.asset,
            start_seconds=args.start,
            end_seconds=args.end,
            result=args.result,
            max_frames=args.max_frames,
            supports=args.supports,
            refutes=args.refutes,
            reuse_frames=not args.refresh,
            commit=not args.no_commit,
        )
    if args.video_command == "context":
        return build_video_evidence_pack(
            args.repo,
            args.evidence,
            max_frames=args.max_frames,
            budget_tokens=args.budget_tokens,
            frame_refs=_read_frame_refs(args.frames_file),
        )
    if args.video_command == "index-text":
        from .video_retrieval import register_video_text_index

        return register_video_text_index(
            args.repo,
            args.asset,
            args.source,
            text_kind=args.kind,
            producer=args.producer,
            producer_version=args.producer_version,
            commit=not args.no_commit,
        )
    if args.video_command == "search":
        from .video_retrieval import search_video_evidence

        return search_video_evidence(
            args.repo,
            args.query,
            asset_ids=args.asset,
            target_id=args.target,
            start_seconds=args.start,
            end_seconds=args.end,
            limit=args.limit,
        )
    if args.video_command == "plan":
        from .video_planning import plan_video_inspection

        return plan_video_inspection(
            args.repo,
            args.asset,
            args.query,
            start_seconds=args.start,
            end_seconds=args.end,
            max_frames=args.max_frames,
            max_decodes=args.max_decodes,
            target_id=args.target,
        )
    if args.video_command in {"inspect", "run"}:
        import contextlib
        import sys

        from .research_git import ResearchGitError
        from .video_inspection import inspect_video_evidence

        try:
            from ai_scientist.vlm import create_client
        except ImportError as exc:
            raise ResearchGitError(
                "video inspection requires the configured VLM provider dependencies"
            ) from exc
        frame_refs = (
            _read_frame_refs(args.frames_file)
            if args.video_command == "inspect"
            else None
        )
        with contextlib.redirect_stdout(sys.stderr):
            client, model = create_client(args.model)
        if args.video_command == "run":
            from .video_planning import run_video_inspection

            return run_video_inspection(
                args.repo,
                args.asset,
                args.query,
                client=client,
                model=model,
                start_seconds=args.start,
                end_seconds=args.end,
                max_frames=args.max_frames,
                max_decodes=args.max_decodes,
                target_id=args.target,
                commit=not args.no_commit,
            )
        return inspect_video_evidence(
            args.repo,
            args.evidence,
            message=args.message,
            client=client,
            model=model,
            max_frames=args.max_frames,
            frame_refs=frame_refs,
            budget_tokens=args.budget_tokens,
            commit=not args.no_commit,
        )
    if args.video_command == "feedback":
        from .video_inspection import video_execution_feedback

        return video_execution_feedback(
            args.repo, asset_ids=args.asset, limit=args.limit
        )
    return {"assets": list_video_assets(args.repo)}


def _read_frame_refs(path: str | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    import json

    from ai_scientist.utils.safe_files import (
        BoundedFileError,
        read_bounded_regular_file,
    )
    from .research_git import ResearchGitError

    try:
        value = json.loads(
            read_bounded_regular_file(path, maximum=65536, label="frame_refs")
        )
    except (BoundedFileError, ValueError, UnicodeDecodeError) as exc:
        raise ResearchGitError(
            "frame references must be a bounded regular JSON file"
        ) from exc
    if not isinstance(value, list):
        raise ResearchGitError("frame references must be a JSON list")
    return value
