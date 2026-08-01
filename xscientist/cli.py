from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ._version import __version__
from .entrypoints import (
    ara_main,
    auth_main,
    batch_main,
    bfts_main,
    daemon_main,
    feedback_main,
    manager_main,
    preflight_main,
    project_main,
    validate_main,
    zhipu_main,
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
    "bfts": bfts_main,
    "zhipu": zhipu_main,
    "preflight": preflight_main,
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
        ("bfts", "Run the low-level BFTS experiment launcher."),
        ("zhipu", "Run the Zhipu-oriented experiment launcher."),
        ("preflight", "Check runtime dependencies and credentials."),
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
    evolution_parser = subparsers.add_parser(
        "evolution-gate",
        help="Evaluate a shadow self-evolution candidate against hidden benchmarks.",
    )
    evolution_parser.add_argument("--project-root", required=True)
    evolution_parser.add_argument("--candidate", required=True)
    evolution_parser.add_argument("--benchmark", required=True)
    evolution_parser.add_argument("--ablation", required=True)
    evolution_parser.add_argument("--policy", default=None)
    evolution_parser.add_argument("--canary", default=None)
    evolution_parser.add_argument("--approver", action="append", default=[])
    return parser


def _load_json_file(path: str) -> object:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_evolution_gate(parsed: argparse.Namespace) -> int:
    from ai_scientist.utils.evolution_gate import (
        approve_production_promotion,
        build_ablation_report,
        build_evolution_candidate,
        build_evolution_gate,
        save_evolution_gate,
    )
    from ai_scientist.utils.pipeline_contracts import load_contract_artifact

    constitution = load_contract_artifact(
        parsed.project_root, "science_constitution", default={}
    )

    candidate_payload = _load_json_file(parsed.candidate)
    if not isinstance(candidate_payload, dict):
        raise ValueError("candidate JSON must be an object")
    candidate = (
        candidate_payload
        if candidate_payload.get("candidate_hash")
        else build_evolution_candidate(constitution=constitution, **candidate_payload)
    )
    benchmark_payload = _load_json_file(parsed.benchmark)
    samples = (
        benchmark_payload.get("samples")
        if isinstance(benchmark_payload, dict)
        else benchmark_payload
    )
    if not isinstance(samples, list):
        raise ValueError("benchmark JSON must be a list or an object with samples")
    policy = _load_json_file(parsed.policy) if parsed.policy else None
    if policy is not None and not isinstance(policy, dict):
        raise ValueError("policy JSON must be an object")
    ablation_payload = _load_json_file(parsed.ablation)
    if isinstance(ablation_payload, dict) and ablation_payload.get("report_hash"):
        ablation_report = ablation_payload
    else:
        ablation_samples = (
            ablation_payload.get("samples")
            if isinstance(ablation_payload, dict)
            else ablation_payload
        )
        if not isinstance(ablation_samples, list):
            raise ValueError("ablation JSON must be a list or an object with samples")
        ablation_report = build_ablation_report(candidate, ablation_samples)
    report = build_evolution_gate(
        candidate,
        samples,
        constitution=constitution,
        ablation_report=ablation_report,
        policy=policy,
    )
    if parsed.canary:
        canary = _load_json_file(parsed.canary)
        if not isinstance(canary, dict):
            raise ValueError("canary JSON must be an object")
        report = approve_production_promotion(
            report,
            canary,
            constitution=constitution,
            approver_ids=parsed.approver,
        )
    save_evolution_gate(
        parsed.project_root,
        report,
        constitution=constitution,
        producer="xscientist.evolution_gate",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("decision") in {"promote_to_canary", "approved"} else 3


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in _DELEGATES:
        return _DELEGATES[raw_argv[0]](raw_argv[1:])

    parser = _build_parser()
    parsed = parser.parse_args(raw_argv)
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
    if parsed.command == "evolution-gate":
        return _run_evolution_gate(parsed)
    parser.error(f"Unsupported command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
