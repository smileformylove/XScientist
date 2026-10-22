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
    for command in (add, listing, observe, context):
        command.add_argument("--repo", default=".")
        command.add_argument("--json", action="store_true", dest="as_json")
    for command in (add, observe):
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
        )
    return {"assets": list_video_assets(args.repo)}
