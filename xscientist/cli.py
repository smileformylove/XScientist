from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ._version import __version__
from .entrypoints import (
    ara_main,
    auth_main,
    batch_main,
    daemon_main,
    feedback_main,
    manager_main,
    project_main,
    validate_main,
)

_DELEGATES = {
    "project": project_main,
    "batch": batch_main,
    "daemon": daemon_main,
    "manager": manager_main,
    "ara": ara_main,
    "auth": auth_main,
    "feedback": feedback_main,
    "validate": validate_main,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist",
        description="XScientist SDK, workflow CLI, and API service.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("project", "Run one end-to-end research project."),
        ("batch", "Run continuous/batch paper generation."),
        ("daemon", "Run the long-lived research daemon."),
        ("manager", "Inspect and manage research outputs."),
        ("ara", "Inspect, validate, re-execute, or fork an ARA."),
        ("auth", "Manage local login sessions."),
        ("feedback", "Inspect feedback and improvement signals."),
        ("validate", "Run repository/package validation."),
    ):
        subparser = subparsers.add_parser(
            name,
            help=help_text,
            add_help=False,
        )
        subparser.add_argument("args", nargs=argparse.REMAINDER)

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP API service.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--work-dir", default=None)
    serve_parser.add_argument("--output-root", default=None)
    serve_parser.add_argument("--max-workers", type=int, default=2)
    serve_parser.add_argument("--max-output-chars", type=int, default=200_000)
    serve_parser.add_argument("--state-dir", default=None)
    serve_parser.add_argument("--reload", action="store_true")

    info_parser = subparsers.add_parser("info", help="Print installation metadata.")
    info_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(argv)
    if parsed.command in _DELEGATES:
        return _DELEGATES[parsed.command](parsed.args)
    if parsed.command == "serve":
        from .service import run_server

        run_server(
            host=parsed.host,
            port=parsed.port,
            work_dir=parsed.work_dir,
            output_root=parsed.output_root,
            max_workers=parsed.max_workers,
            max_output_chars=parsed.max_output_chars,
            state_dir=parsed.state_dir,
            reload=parsed.reload,
        )
        return 0
    if parsed.command == "info":
        payload = {
            "name": "xscientist",
            "version": __version__,
            "python_api": "from xscientist import XScientist, ProjectRequest",
            "http_factory": "from xscientist import create_app",
        }
        if parsed.as_json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0
    parser.error(f"Unsupported command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
