from __future__ import annotations

import argparse
import json
import re
import shlex
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
    evolution_main,
    feedback_main,
    git_main,
    manager_main,
    preflight_main,
    project_main,
    research_main,
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
    "research": research_main,
    "evolution": evolution_main,
    "git": git_main,
}

_PROVIDER_CHOICES = [
    "zhipu",
    "openai",
    "anthropic",
    "deepseek",
    "gemini",
    "openrouter",
    "huggingface",
    "ollama",
    "openai_compat",
    "bedrock",
    "vertex_ai",
]

_TASK_CHOICES = [
    "protocol",
    "manage",
    "research",
    "paper",
    "pdf-review",
    "ml-study",
    "service",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist",
        description="XScientist SDK, workflow CLI, and API service.",
        epilog=(
            "Start simple: `xscientist demo ./demo-study`, then use "
            "`xscientist start --help` for model-backed research."
        ),
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
        ("research", "Use native version control for scientific progress."),
        ("evolution", "Build, evaluate, sign, deploy, or roll back agent candidates."),
        ("git", "Check and use XScientist native research version control."),
    ):
        subparser = subparsers.add_parser(
            name,
            help=help_text,
            add_help=False,
        )
        subparser.add_argument("args", nargs=argparse.REMAINDER)

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP API service.")
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: loopback only)"
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="TCP port")
    serve_parser.add_argument(
        "--work-dir",
        default=None,
        help="existing directory containing service-readable inputs",
    )
    serve_parser.add_argument(
        "--output-root", default=None, help="directory for projects and API state"
    )
    serve_parser.add_argument(
        "--max-workers", type=int, default=2, help="maximum concurrent jobs"
    )
    serve_parser.add_argument(
        "--max-output-chars",
        type=int,
        default=200_000,
        help="retained stdout/stderr characters per stream",
    )
    serve_parser.add_argument(
        "--max-workspace-bytes",
        type=int,
        default=10 * 1024**3,
        help="per-project filesystem byte limit",
    )
    serve_parser.add_argument(
        "--max-workspace-files",
        type=int,
        default=100_000,
        help="per-project filesystem file limit",
    )
    serve_parser.add_argument(
        "--state-dir", default=None, help="persistent job metadata directory"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="development reload"
    )
    serve_parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="explicitly allow an unauthenticated non-loopback binding",
    )

    info_parser = subparsers.add_parser("info", help="Print installation metadata.")
    info_parser.add_argument("--json", action="store_true", dest="as_json")
    demo_parser = subparsers.add_parser(
        "demo",
        help="Create a complete provider-free evidence demo and offline DAG.",
    )
    demo_parser.add_argument("directory", nargs="?", default="./xscientist-demo")
    demo_parser.add_argument("--lang", choices=["auto", "en", "zh"], default="auto")
    demo_parser.add_argument("--open", action="store_true", dest="open_browser")
    demo_parser.add_argument(
        "--autopilot",
        action="store_true",
        help="also create zero-cost deterministic Autopilot runtime artifacts",
    )
    demo_parser.add_argument(
        "--autopilot-profile",
        choices=["balanced", "discovery", "publication"],
        default="balanced",
    )
    demo_parser.add_argument("--git-user-name")
    demo_parser.add_argument("--git-user-email")
    demo_parser.add_argument("--json", action="store_true", dest="as_json")
    status_parser = subparsers.add_parser(
        "status",
        help="Show research progress, budget use, outputs, and the next action.",
    )
    status_parser.add_argument("workspace", nargs="?", default=None)
    status_parser.add_argument("--lang", choices=["auto", "en", "zh"], default="auto")
    status_parser.add_argument(
        "--verbose",
        action="store_true",
        help="also show branch, pipeline, token, and background-run details",
    )
    status_parser.add_argument("--json", action="store_true", dest="as_json")
    runs_parser = subparsers.add_parser(
        "runs",
        help="Start, inspect, watch, cancel, and resume detached research runs.",
    )
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_subparsers.add_parser("list", help="List local detached runs.")
    runs_list.add_argument("--workspace", default=".")
    runs_list.add_argument("--json", action="store_true", dest="as_json")
    for command, help_text in (
        ("show", "Show one detached run."),
        ("watch", "Follow one detached run until it finishes."),
        ("logs", "Show bounded stdout/stderr tails for one run."),
        ("cancel", "Request safe cancellation for one run."),
        ("resume", "Resume a failed, cancelled, or interrupted run."),
    ):
        run_parser = runs_subparsers.add_parser(command, help=help_text)
        run_parser.add_argument("run_id")
        run_parser.add_argument(
            "--workspace", default=".", help="workspace that owns the run"
        )
        run_parser.add_argument("--json", action="store_true", dest="as_json")
        if command == "watch":
            run_parser.add_argument("--interval", type=float, default=2.0)
        if command == "logs":
            run_parser.add_argument(
                "--stream", choices=["stdout", "stderr", "both"], default="both"
            )
            run_parser.add_argument("--tail", type=int, default=200)
        if command == "resume":
            run_parser.add_argument(
                "--force",
                action="store_true",
                help="resume even when local prerequisite checks still fail",
            )
    executor_parser = subparsers.add_parser(
        "executor",
        help="Inspect, build, cache, or update the version-matched Docker executor.",
    )
    executor_subparsers = executor_parser.add_subparsers(
        dest="executor_command", required=True
    )
    for command, help_text in (
        ("check", "Inspect Docker and the exact configured image."),
        ("build", "Build the configured executor image."),
        ("prepare", "Reuse a valid cached image or build it."),
        ("update", "Refresh the base image and rebuild the executor."),
    ):
        command_parser = executor_subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--workspace", default=".")
        command_parser.add_argument("--json", action="store_true", dest="as_json")
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        help="Check package and workspace compatibility without changing files.",
    )
    upgrade_subparsers = upgrade_parser.add_subparsers(
        dest="upgrade_command", required=True
    )
    upgrade_check = upgrade_subparsers.add_parser(
        "check", help="Check installed, workspace, and optional PyPI versions."
    )
    upgrade_check.add_argument("--workspace", default=".")
    upgrade_check.add_argument(
        "--online",
        action="store_true",
        help="explicitly query PyPI for the latest published version",
    )
    upgrade_check.add_argument("--timeout", type=float, default=3.0)
    upgrade_check.add_argument("--json", action="store_true", dest="as_json")
    completion_parser = subparsers.add_parser(
        "completion",
        help="Print a shell completion script; never edit shell configuration.",
    )
    completion_parser.add_argument("shell", choices=["bash", "zsh", "fish"])
    conformance_parser = subparsers.add_parser(
        "conformance",
        help="Create or verify an offline protocol producer conformance kit.",
    )
    conformance_subparsers = conformance_parser.add_subparsers(
        dest="conformance_command", required=True
    )
    conformance_init = conformance_subparsers.add_parser(
        "init", help="Create versioned known-good and known-bad fixtures."
    )
    conformance_init.add_argument("directory")
    conformance_init.add_argument("--json", action="store_true", dest="as_json")
    conformance_check = conformance_subparsers.add_parser(
        "check", help="Check a kit directory or one JSON protocol artifact."
    )
    conformance_check.add_argument("target")
    conformance_check.add_argument("--schema", default="research_object")
    conformance_check.add_argument("--json", action="store_true", dest="as_json")
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run reproducible provider-free usability benchmarks.",
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(
        dest="benchmark_command", required=True
    )
    first_run = benchmark_subparsers.add_parser(
        "first-run", help="Measure empty-directory to inspectable research status."
    )
    first_run.add_argument("--workspace", default=None)
    first_run.add_argument(
        "--profile",
        choices=["balanced", "discovery", "publication"],
        default="balanced",
    )
    first_run.add_argument("--max-seconds", type=float, default=None)
    first_run.add_argument("--json", action="store_true", dest="as_json")
    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Control explicit opt-in, local-only, payload-free usage counters.",
    )
    metrics_subparsers = metrics_parser.add_subparsers(
        dest="metrics_command", required=True
    )
    for command, help_text in (
        ("status", "Show the local collection boundary and event count."),
        ("enable", "Enable local-only usage counters."),
        ("disable", "Disable future local usage counters."),
        ("export", "Print every locally stored event for inspection."),
    ):
        command_parser = metrics_subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--json", action="store_true", dest="as_json")
    init_parser = subparsers.add_parser(
        "init",
        help="Create a ready-to-configure research workspace.",
    )
    init_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="workspace directory (default: current directory)",
    )
    init_parser.add_argument(
        "--profile",
        choices=["default", "deep"],
        default="default",
        help="packaged BFTS profile to copy",
    )
    init_parser.add_argument(
        "--provider",
        choices=_PROVIDER_CHOICES,
        default="zhipu",
        help="credential template and model provider",
    )
    init_parser.add_argument(
        "--model",
        default=None,
        help="provider-compatible model ID (required except for the Zhipu default)",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="replace only files managed by this command",
    )
    init_parser.add_argument("--json", action="store_true", dest="as_json")

    for name, help_text in (
        (
            "start",
            "Prepare and run a safe automated research project from one question.",
        ),
        ("setup", "Create, configure, and diagnose a first research workspace."),
        (
            "doctor",
            "Check workspace, Research VCS, provider, auth, and task capabilities.",
        ),
        ("capability", "Resolve optional modules for a concrete research task."),
    ):
        subparser = subparsers.add_parser(name, help=help_text, add_help=False)
        subparser.add_argument("args", nargs=argparse.REMAINDER)

    provider_parser = subparsers.add_parser(
        "provider",
        help="Add, inspect, and switch model providers without exposing API keys.",
    )
    provider_subparsers = provider_parser.add_subparsers(
        dest="provider_command", required=True
    )
    provider_list = provider_subparsers.add_parser(
        "list", help="Show provider readiness."
    )
    provider_list.add_argument(
        "--workspace",
        default=None,
        help=(
            "workspace root (default: discover from the current directory; "
            "without one, show safe provider discovery only)"
        ),
    )
    provider_list.add_argument(
        "--all",
        action="store_true",
        help="also show providers that are neither configured nor locally detected",
    )
    provider_list.add_argument("--json", action="store_true", dest="as_json")
    provider_check = provider_subparsers.add_parser(
        "check",
        help=(
            "Check one provider's local credentials, client, model, and optional "
            "cost enforcement without making a paid API call."
        ),
    )
    provider_check.add_argument("name", nargs="?", choices=_PROVIDER_CHOICES)
    provider_check.add_argument(
        "--workspace",
        default=None,
        help="workspace root (default: discover from the current directory)",
    )
    provider_check.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="also require a known non-negative price for the configured model",
    )
    provider_check.add_argument("--json", action="store_true", dest="as_json")
    provider_add = provider_subparsers.add_parser(
        "add",
        help="Configure a provider; secrets are prompted with hidden input.",
    )
    provider_add.add_argument("name", choices=_PROVIDER_CHOICES)
    provider_add.add_argument(
        "--workspace",
        default=None,
        help="workspace root (default: discover from the current directory)",
    )
    provider_add.add_argument("--model", default=None)
    provider_add.add_argument("--no-activate", action="store_true")
    provider_add.add_argument("--no-update-bfts", action="store_true")
    provider_add.add_argument("--json", action="store_true", dest="as_json")
    provider_add.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; require credentials to already exist in env or the env file",
    )
    provider_activate = provider_subparsers.add_parser(
        "activate",
        help="Switch the active provider and default model.",
    )
    provider_activate.add_argument("name", choices=_PROVIDER_CHOICES)
    provider_activate.add_argument(
        "--workspace",
        default=None,
        help="workspace root (default: discover from the current directory)",
    )
    provider_activate.add_argument("--no-update-bfts", action="store_true")
    provider_activate.add_argument("--json", action="store_true", dest="as_json")
    provider_remove = provider_subparsers.add_parser(
        "remove",
        help="Remove provider metadata; stored credentials are left untouched.",
    )
    provider_remove.add_argument("name", choices=_PROVIDER_CHOICES)
    provider_remove.add_argument(
        "--workspace",
        default=None,
        help="workspace root (default: discover from the current directory)",
    )
    provider_remove.add_argument("--json", action="store_true", dest="as_json")
    privacy_parser = subparsers.add_parser(
        "privacy",
        help="Audit files and Git history without displaying matched private values.",
    )
    privacy_subparsers = privacy_parser.add_subparsers(
        dest="privacy_command", required=True
    )
    privacy_audit = privacy_subparsers.add_parser(
        "audit", help="Scan publishable text for credentials and machine-local paths."
    )
    privacy_audit.add_argument("path", nargs="?", default=".")
    privacy_audit.add_argument("--include-untracked", action="store_true")
    privacy_audit.add_argument("--history", action="store_true")
    privacy_audit.add_argument("--json", action="store_true", dest="as_json")
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


def _build_setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist setup",
        description="Create, configure, and diagnose a first research workspace.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--task",
        choices=_TASK_CHOICES,
        default="research",
        help="research intent used to resolve optional capabilities",
    )
    parser.add_argument("--profile", choices=["default", "deep"], default="default")
    parser.add_argument(
        "--provider",
        choices=_PROVIDER_CHOICES,
        default=None,
        help=(
            "model provider; required for provider-backed non-interactive setup, "
            "prompted interactively when omitted"
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-research-vcs",
        action="store_true",
        help="create files without initializing local Research VCS",
    )
    parser.add_argument(
        "--skip-credentials",
        action="store_true",
        help="create metadata but do not prompt for or write provider credentials",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; reuse existing environment or local env values only",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also probe the Docker daemon and exact experiment image",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _build_start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist start",
        usage="xscientist start DIRECTORY [study options] [safety options]",
        description=(
            "Create or reuse one workspace, validate every prerequisite, and run "
            "a traceable automated research project."
        ),
    )
    parser.add_argument("directory", help="workspace and Research VCS root")
    study = parser.add_argument_group("study")
    study.add_argument(
        "--question",
        required=False,
        help="one concrete research question; prompted interactively when omitted",
    )
    study.add_argument(
        "--autopilot",
        choices=["balanced", "discovery", "publication"],
        default="balanced",
        help="research behavior (default: balanced)",
    )
    study.add_argument(
        "--task",
        choices=["research", "paper", "pdf-review", "ml-study"],
        default=None,
        help=(
            "runtime capability profile; defaults to research, or paper for "
            "publication autopilot"
        ),
    )
    study.add_argument(
        "--provider",
        choices=_PROVIDER_CHOICES,
        default=None,
        help="provider for a new workspace; existing workspaces reuse their active provider",
    )
    study.add_argument(
        "--model",
        default=None,
        help="provider-compatible model ID; prompted when needed",
    )

    evidence = parser.add_argument_group("evidence and accountability")
    evidence.add_argument(
        "--user",
        default=None,
        help=(
            "accountable research actor; must match the active local login, or "
            "creates one when no login exists"
        ),
    )
    evidence.add_argument(
        "--data-dir", default=None, help="read-only empirical input directory"
    )
    evidence.add_argument(
        "--allow-synthetic-data",
        action="store_true",
        help="explicitly permit exploratory synthetic/computational evidence",
    )
    limits = parser.add_argument_group("limits")
    limits.add_argument(
        "--max-project-tokens",
        type=int,
        default=None,
        help="hard project token limit",
    )
    limits.add_argument(
        "--max-project-hours",
        type=float,
        default=None,
        help="hard wall-clock hour limit",
    )
    limits.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="hard model-cost limit in USD",
    )
    limits.add_argument(
        "--price-input-per-million",
        type=float,
        default=None,
        help="explicit USD input-token price for models without a bundled price",
    )
    limits.add_argument(
        "--price-output-per-million",
        type=float,
        default=None,
        help="explicit USD output-token price for models without a bundled price",
    )
    limits.add_argument(
        "--price-cached-input-per-million",
        type=float,
        default=None,
        help="optional USD cached-input price; defaults to the input price",
    )
    safety = parser.add_argument_group("safety and automation")
    safety.add_argument(
        "--profile",
        choices=["default", "deep"],
        default="default",
        help="experiment-search configuration (default: default)",
    )
    safety.add_argument(
        "--build-executor",
        action="store_true",
        help="explicitly build the generated isolated Docker image before diagnosis",
    )
    safety.add_argument(
        "--skip-credentials",
        action="store_true",
        help="prepare configuration without requiring credential values",
    )
    safety.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; require every missing choice explicitly",
    )
    safety.add_argument(
        "--force",
        action="store_true",
        help="refresh only XScientist-managed workspace files",
    )
    safety.add_argument(
        "--detach",
        action="store_true",
        help="run in the background and manage it with `xscientist runs`",
    )
    safety.add_argument(
        "--prepare-only",
        action="store_true",
        help="stop after workspace, login, provider, and deep runtime validation",
    )
    safety.add_argument(
        "--json", action="store_true", dest="as_json", help="emit structured JSON"
    )
    return parser


def _print_curated_help(*, include_advanced: bool = False) -> None:
    print("usage: xscientist COMMAND [options]")
    print()
    print("Start here:")
    print("  demo       Create a free, offline contested-evidence example")
    print("  start      Prepare and run one guarded autonomous study")
    print("  status     Show progress, budget, outputs, and the next action")
    print("  runs       Watch, inspect, cancel, or resume detached runs")
    print("  doctor     Diagnose setup and print copyable repairs")
    print()
    print("Configure:")
    print("  setup      Create and diagnose a workspace")
    print("  provider   Add, check, and switch model providers")
    print("  capability Explain task-specific optional dependencies")
    print("  executor   Check, cache, build, or update the isolated executor")
    print("  upgrade    Check package and workspace compatibility (read-only)")
    print("  completion Generate bash, zsh, or fish completion")
    print()
    print("Scientific history:")
    print("  research   Plan, record, review, branch, diff, and reproduce")
    if include_advanced:
        print()
        print("Advanced and compatibility commands:")
        print("  project batch daemon manager ara feedback evolution evolution-gate")
        print("  serve privacy conformance metrics auth git preflight validate")
        print("  benchmark bfts zhipu init info")
    else:
        print()
        print("Run `xscientist help --all` for advanced and compatibility commands.")
    print("Run `xscientist COMMAND --help` for command-specific options.")


def _selected_language(value: str) -> str:
    if value in {"en", "zh"}:
        return value
    import locale

    detected = (locale.getlocale()[0] or "en").lower()
    return "zh" if detected.startswith("zh") else "en"


def _command_with_workspace(
    command: str,
    workspace: str | Path | None,
    *,
    flag: str = "--workspace",
) -> str:
    """Render one command that remains copyable from the caller's directory."""

    if workspace is None or flag in command:
        return command
    return f"{command} {flag} {shlex.quote(str(workspace))}"


def _contextual_action(command: str, workspace: str | Path | None) -> str:
    if workspace is None:
        return command
    quoted = shlex.quote(str(workspace))
    if "--workspace ." in command:
        return command.replace("--workspace .", f"--workspace {quoted}")
    if "--repo ." in command:
        return command.replace("--repo .", f"--repo {quoted}")
    if command == "xscientist research init .":
        return f"xscientist research init {quoted}"
    if command.startswith("xscientist provider add "):
        return _command_with_workspace(command, workspace)
    if command.startswith("xscientist doctor "):
        return _command_with_workspace(command, workspace)
    if command.startswith("xscientist executor "):
        return _command_with_workspace(command, workspace)
    if command.startswith("xscientist preflight "):
        return f"cd {quoted} && {command}"
    if command.startswith("xscientist research "):
        return _command_with_workspace(command, workspace, flag="--repo")
    return command


def _parse_structured_log(lines: Sequence[str]) -> dict[str, object] | None:
    text = "\n".join(lines).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _structured_log_tail_is_partial(lines: Sequence[str]) -> bool:
    text = "\n".join(lines).strip()
    if not text or _parse_structured_log(lines) is not None:
        return False
    first = text.splitlines()[0].strip()
    return first.startswith(("{", "[", '"')) or bool(
        re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", first)
    )


def _nested_first(payload: object, key: str) -> object | None:
    if isinstance(payload, dict):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return value
        for child in payload.values():
            found = _nested_first(child, key)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(payload, list):
        for child in payload:
            found = _nested_first(child, key)
            if found not in (None, "", [], {}):
                return found
    return None


def _run_failure_summary(logs: dict[str, object]) -> str | None:
    streams = [
        [str(line) for line in logs.get(stream, [])] for stream in ("stderr", "stdout")
    ]

    # Structured prerequisite reports are the strongest signal, regardless of
    # which stream also happens to contain a trailing warning.
    for lines in streams:
        payload = _parse_structured_log(lines)
        if payload is None:
            continue
        error = _nested_first(payload, "error")
        if isinstance(error, str):
            return error
        actions = _nested_first(payload, "next_actions")
        if isinstance(actions, list) and actions:
            return f"Prerequisite check failed; next: {actions[0]}"
        codes = _nested_first(payload, "error_codes")
        if isinstance(codes, list) and codes:
            return "Run failed: " + ", ".join(str(code) for code in codes[:3])

    # Human start output lists repairs in priority order. Select the first
    # repair instead of the trailing preflight command.
    action_markers = ("resolve these items", "next actions", "next steps")
    for lines in streams:
        for index, line in enumerate(lines):
            candidate = line.strip()
            lowered = candidate.lower()
            if lowered.startswith("next:"):
                action = candidate.split(":", 1)[1].strip()
                if action:
                    return f"Prerequisite check failed; next: {action}"
            if not any(marker in lowered for marker in action_markers):
                continue
            for following in lines[index + 1 :]:
                action = following.strip().lstrip("-*").strip()
                if action:
                    return f"Prerequisite check failed; next: {action}"

    for lines in streams:
        for line in lines:
            candidate = line.strip()
            lowered = candidate.lower()
            if lowered.startswith(("problem:", "error:", "xscientist start:")):
                return candidate

    fallbacks: list[str] = []
    for lines in streams:
        for line in reversed(lines):
            candidate = line.strip()
            if candidate and candidate not in {"{", "}", "[", "]", "},", "],"}:
                fallbacks.append(candidate)
                break
    return fallbacks[0] if fallbacks else None


def _readiness_state(value: object) -> str:
    if value is None:
        return "not checked"
    return "ready" if bool(value) else "needs attention"


def _doctor_check_state(name: str, check: dict[str, object]) -> str:
    required = check.get("required")
    if required is False and name in {"provider", "auth", "runtime"}:
        return "not required"
    value = check.get("ok", check.get("ready"))
    if name == "runtime" and not check.get("checked"):
        return "not checked"
    if name == "provider" and value:
        probe = check.get("local_probe")
        if isinstance(probe, dict) and probe.get("checked"):
            return "ready (local service and model verified)"
        if check.get("credentials_available"):
            return "configured (credentials present; not live-verified)"
    return _readiness_state(value)


_DOCTOR_CHECK_LABELS = {
    "workspace": "Workspace",
    "git": "Git",
    "research_vcs": "Research history",
    "capabilities": "Dependencies",
    "provider": "Provider",
    "auth": "Research identity",
    "runtime": "Isolated runtime",
}


def _format_run_timestamp(value: object) -> str:
    from datetime import datetime

    raw = str(value or "").strip()
    if not raw:
        return "-"
    try:
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return raw
    return timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")


def _prompt_provider_model(provider: str, *, default: str | None = None) -> str:
    from .provider_config import discover_provider_models, normalize_provider_model

    discovered = discover_provider_models(provider)
    if discovered:
        print(f"Detected {provider} models:")
        for index, model in enumerate(discovered[:8], start=1):
            print(f"  {index}. {model}")
    suggested = default or (discovered[0] if discovered else None)
    example = suggested or f"{provider}/<model>"
    prompt = (
        f"Model [1, Enter={example}]: "
        if discovered and suggested == discovered[0]
        else (f"Model ID [{example}]: " if suggested else f"Model ID ({example}): ")
    )
    entered = input(prompt).strip()
    if discovered and entered.isdigit():
        selected_index = int(entered)
        if not 1 <= selected_index <= len(discovered):
            raise ValueError("model selection is not in the displayed list")
        entered = discovered[selected_index - 1]
    if not entered and not suggested:
        raise ValueError(f"a model is required; use an ID such as {provider}/<model>")
    return normalize_provider_model(provider, entered or suggested)


def _prompt_provider_choice() -> str:
    """Choose from a short readiness-led list without silently picking a vendor."""

    from .dependency_profiles import missing_provider_modules
    from .provider_config import (
        PROVIDER_FIELDS,
        PROVIDER_NAMES,
        configured_field_value,
        discover_provider_models,
    )

    rows = []
    for name in PROVIDER_NAMES:
        credentials = all(
            not field.required or bool(configured_field_value(field, {}))
            for field in PROVIDER_FIELDS[name]
        )
        client = not missing_provider_modules(name)
        local_models = discover_provider_models(name)
        rows.append((name, credentials, client, local_models))
    ready_rows = [row for row in rows if row[3] or (row[1] and row[2])]
    if len(ready_rows) == 1:
        selected = str(ready_rows[0][0])
        print(f"Using detected provider: {selected}")
        return selected
    visible = ready_rows or [
        row for row in rows if row[0] in {"ollama", "openai", "anthropic", "zhipu"}
    ]
    print("Choose a model provider:")
    for index, (name, credentials, client, local_models) in enumerate(visible, start=1):
        state = (
            f"local service ({len(local_models)} model(s))"
            if local_models
            else (
                "locally configured"
                if credentials and client
                else ("client installed" if client else "client not installed")
            )
        )
        print(f"  {index}. {name:<14} {state}")
    print("  More providers are available through `xscientist start --help`.")
    answer = input("Provider [1]: ").strip()
    if not answer:
        selected_index = 1
    elif answer.isdigit() and 1 <= int(answer) <= len(visible):
        selected_index = int(answer)
    else:
        matches = [index for index, row in enumerate(rows, start=1) if row[0] == answer]
        if not matches:
            raise ValueError("provider selection is not in the displayed list")
        return str(rows[matches[0] - 1][0])
    return str(visible[selected_index - 1][0])


def _interactive_start_inputs(
    parsed: argparse.Namespace, *, new_workspace: bool
) -> None:
    """Fill only missing first-run choices without weakening automation."""

    if parsed.non_interactive or not sys.stdin.isatty():
        return
    if not str(parsed.question or "").strip():
        parsed.question = input("Research question: ").strip()
    if new_workspace and not parsed.provider:
        from .provider_config import DEFAULT_MODELS

        parsed.provider = _prompt_provider_choice()
        if not parsed.model:
            parsed.model = _prompt_provider_model(
                parsed.provider,
                default=DEFAULT_MODELS.get(parsed.provider),
            )
    if (
        not parsed.prepare_only
        and not parsed.data_dir
        and not parsed.allow_synthetic_data
    ):
        print("Evidence input:")
        print("  1. Existing empirical data directory (read-only snapshot)")
        print(
            "  2. Synthetic/computational exploration (cannot become independent proof)"
        )
        print("  3. Prepare the workspace only")
        choice = input("Choose [1]: ").strip() or "1"
        if choice == "1":
            parsed.data_dir = input("Data directory: ").strip()
        elif choice == "2":
            parsed.allow_synthetic_data = True
        elif choice == "3":
            parsed.prepare_only = True
        else:
            raise ValueError("evidence input choice must be 1, 2, or 3")
    if (
        not parsed.prepare_only
        and parsed.max_cost_usd is None
        and parsed.max_project_tokens is None
    ):
        answer = input(
            "Optional maximum model cost in USD "
            "[blank keeps the workspace token/time limits]: "
        ).strip()
        if answer:
            parsed.max_cost_usd = float(answer)


def _build_doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist doctor",
        description=(
            "Check workspace, Research VCS, provider, auth, and task capabilities."
        ),
    )
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--task", choices=_TASK_CHOICES, default="research")
    parser.add_argument("--provider", choices=_PROVIDER_CHOICES, default=None)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also probe the configured models, Docker daemon, and exact image",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _build_capability_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist capability",
        description="Resolve optional modules for a concrete research task.",
    )
    subparsers = parser.add_subparsers(dest="capability_command", required=True)
    list_parser = subparsers.add_parser(
        "list", help="List supported tasks and composable capabilities."
    )
    list_parser.add_argument("--json", action="store_true", dest="as_json")
    check_parser = subparsers.add_parser(
        "check", help="Probe requirements for one research task."
    )
    check_parser.add_argument("task", choices=_TASK_CHOICES)
    check_parser.add_argument("--provider", choices=_PROVIDER_CHOICES, default=None)
    check_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _load_json_file(path: str) -> object:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_local_metric(event: str, *, ok: bool) -> None:
    """Best-effort local counter; metrics must never affect command behavior."""

    try:
        from .usage_metrics import record_event

        record_event(event, status="ok" if ok else "error")
    except (OSError, ValueError):
        pass


def _persist_readiness_report(
    workspace: str | Path | None,
    report: dict[str, object],
) -> None:
    """Remember the latest readiness result without entering Research VCS."""

    try:
        from ai_scientist.utils.atomic_io import atomic_write_json
        from .provider_config import discover_workspace_root

        root = (
            Path(workspace).expanduser().resolve()
            if workspace is not None
            else discover_workspace_root()
        )
        if root is None or not (root / ".xscientist").is_dir():
            return
        atomic_write_json(root / ".xscientist" / "readiness.json", report)
    except (OSError, ValueError):
        # Readiness caching is advisory and must never change command behavior.
        return


def _installation_info() -> dict[str, object]:
    import importlib.util
    import os

    from ai_scientist.utils.auth_session import validate_session
    from .dependency_profiles import (
        installation_command,
        missing_provider_modules,
        resolve_task_capabilities,
    )
    from .provider_config import discover_provider_models

    service_modules = ["fastapi", "pydantic", "uvicorn"]
    runtime_resolution = resolve_task_capabilities("research")
    missing_runtime = list(runtime_resolution["missing_modules"])
    missing_service = [
        name for name in service_modules if importlib.util.find_spec(name) is None
    ]
    runtime_ready = not missing_runtime
    service_ready = not missing_service
    if runtime_ready and service_ready:
        profile = "research+service"
    elif runtime_ready:
        profile = "research"
    elif service_ready:
        profile = "core+service"
    else:
        profile = "core"
    authenticated, auth_status, _session = validate_session()
    auth_status = {
        "未检测到登录会话": "no login session was found",
        "登录会话缺少用户名": "login session has no username",
        "登录会话缺少过期时间": "login session has no expiration time",
        "登录会话已过期": "login session has expired",
    }.get(auth_status, auth_status)
    active_provider = str(os.environ.get("AI_SCIENTIST_ACTIVE_PROVIDER") or "").strip()
    local_models = (
        [] if active_provider else discover_provider_models("ollama", timeout=0.2)
    )
    missing_provider_clients = (
        missing_provider_modules(active_provider) if active_provider else None
    )
    provider_client_ready = not missing_provider_clients if active_provider else None
    suggested_provider = active_provider or ("ollama" if local_models else None)
    suggested_missing_clients = (
        missing_provider_modules(suggested_provider) if suggested_provider else []
    )
    install_needed = bool(missing_runtime or suggested_missing_clients)
    recommended_install = (
        installation_command(suggested_provider)
        if install_needed and suggested_provider
        else (
            'python -m pip install "xscientist[research,openai]"'
            if install_needed
            else None
        )
    )
    recommended_setup = None
    if not active_provider and suggested_provider == "ollama" and local_models:
        recommended_setup = (
            "xscientist setup my-research --provider ollama --model "
            + shlex.quote(local_models[0])
        )
    return {
        "name": "xscientist",
        "version": __version__,
        "installation_profile": profile,
        "research_runtime_ready": runtime_ready,
        "service_ready": service_ready,
        "missing_research_packages": missing_runtime,
        "missing_service_packages": missing_service,
        "provider_client_ready": provider_client_ready,
        "provider_configured": bool(active_provider),
        "provider_client_status": (
            "ready"
            if provider_client_ready is True
            else ("missing" if provider_client_ready is False else "not_configured")
        ),
        "missing_provider_clients": missing_provider_clients,
        "suggested_provider": suggested_provider,
        "discovered_local_models": local_models,
        "recommended_install": recommended_install,
        "recommended_setup": recommended_setup,
        "python_version": sys.version.split()[0],
        "python_executable": Path(sys.executable).name,
        "output_root": (
            "<configured>" if os.environ.get("RESEARCH_OUTPUT_DIR") else "<default>"
        ),
        "host_paths_disclosed": False,
        "authenticated": authenticated,
        "auth_status": auth_status,
        "python_api": "from xscientist import XScientist, ProjectRequest",
        "http_factory": "from xscientist import create_app",
        "quickstart": "xscientist demo ./xscientist-demo --autopilot --open",
        "active_provider": active_provider or None,
        "default_model": os.environ.get("AI_SCIENTIST_DEFAULT_MODEL"),
    }


def _bootstrap_workspace_environment() -> dict[str, object]:
    import os

    explicit = str(os.environ.get("XSCIENTIST_WORKSPACE") or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
    else:
        current = Path.cwd().resolve()
        candidate = next(
            (
                directory
                for directory in (current, *current.parents)
                if (directory / ".xscientist" / "providers.json").is_file()
            ),
            None,
        )
    if (
        candidate is None
        or not (candidate / ".xscientist" / "providers.json").is_file()
    ):
        return {"loaded": False}
    from .provider_config import ProviderConfigError, load_workspace_environment

    try:
        return load_workspace_environment(candidate)
    except ProviderConfigError as exc:
        return {"loaded": False, "error": str(exc)}


def _configure_provider(
    workspace: Path,
    *,
    name: str,
    model_value: str | None = None,
    activate: bool = True,
    update_bfts: bool = True,
    non_interactive: bool = False,
) -> dict[str, object]:
    import getpass
    import os

    from .provider_config import (
        DEFAULT_ENV_FILE,
        DEFAULT_MODELS,
        PROVIDER_FIELDS,
        ProviderConfigError,
        configured_field_value,
        load_provider_config,
        provider_statuses,
        read_env_file,
        resolve_env_file,
        save_provider,
        update_bfts_models,
        update_env_file,
        validate_provider_model,
    )

    config = load_provider_config(workspace, missing_ok=False)
    existing = config.get("providers", {}).get(name, {})
    existing_model = (
        str(existing.get("model") or "") if isinstance(existing, dict) else ""
    )
    model = model_value or existing_model or DEFAULT_MODELS.get(name)
    if not model:
        if non_interactive or not sys.stdin.isatty():
            raise ProviderConfigError(
                f"--model is required for provider {name!r} in non-interactive mode"
            )
        model = _prompt_provider_model(name)
    provider, model = validate_provider_model(name, model)
    env_name = str(config.get("env_file") or DEFAULT_ENV_FILE)
    env_path = resolve_env_file(workspace, env_name)
    stored = read_env_file(env_path)
    updates: dict[str, str] = {}
    for field in PROVIDER_FIELDS[provider]:
        current = configured_field_value(field, stored, os.environ)
        if current:
            continue
        if field.default is not None:
            updates[field.name] = field.default
            continue
        if not field.required:
            continue
        if non_interactive or not sys.stdin.isatty():
            raise ProviderConfigError(
                f"missing {field.name}; set it in the environment or rerun interactively"
            )
        prompt = f"{field.name}: "
        value = getpass.getpass(prompt) if field.secret else input(prompt)
        value = value.strip()
        if not value:
            raise ProviderConfigError(f"{field.name} is required")
        updates[field.name] = value
    if updates or env_path.is_file():
        update_env_file(env_path, updates)
    saved = save_provider(
        workspace,
        provider=provider,
        model=model,
        env_file=env_name,
        activate=activate,
    )
    bfts_updated = False
    if update_bfts and saved.get("active_provider") == provider:
        bfts_updated = update_bfts_models(workspace / "bfts_config.yaml", model)
    payload: dict[str, object] = {
        "ok": True,
        "workspace": ".",
        "provider": provider,
        "model": model,
        "active": saved.get("active_provider") == provider,
        "env_file": env_path.relative_to(workspace).as_posix(),
        "credentials_written": sorted(updates),
        "bfts_updated": bfts_updated,
    }
    status = next(
        row for row in provider_statuses(workspace) if row["provider"] == provider
    )
    payload.update(
        {
            "client_available": status["client_available"],
            "missing_client_modules": status["missing_client_modules"],
            "ready": status["ready"],
            "install_command": status["install_command"],
        }
    )
    return payload


def _run_provider(parsed: argparse.Namespace) -> int:
    import yaml

    from ai_scientist.utils.llm_budget import resolve_model_price

    from .provider_config import (
        PROVIDER_FIELDS,
        ProviderConfigError,
        activate_provider,
        discover_workspace_root,
        load_provider_config,
        provider_statuses,
        remove_provider,
        update_bfts_models,
        workspace_config_path,
    )

    try:
        discovery_only = False
        if parsed.workspace is None:
            workspace = discover_workspace_root()
            if workspace is None:
                if parsed.provider_command == "list":
                    workspace = Path.cwd().resolve()
                    discovery_only = True
                else:
                    raise ProviderConfigError(
                        "provider configuration not found in the current directory or its parents; "
                        "run `xscientist setup WORKSPACE` first or pass --workspace"
                    )
        else:
            workspace = Path(parsed.workspace).expanduser().resolve()
        if parsed.provider_command == "list":
            initialized = workspace_config_path(workspace).is_file()
            rows = provider_statuses(
                workspace,
                probe_local=True,
                allow_uninitialized=discovery_only,
            )
            rows.sort(
                key=lambda row: (
                    not bool(row["active"]),
                    not bool(row["configured"]),
                    not bool(row["local_detected"]),
                    str(row["provider"]),
                )
            )
            workspace_label = (workspace.name or "workspace") if initialized else None
            payload = {
                "workspace": workspace_label,
                "workspace_initialized": initialized,
                "discovery_only": discovery_only,
                "providers": rows,
            }
            if parsed.as_json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                if initialized:
                    print(f"Workspace: {workspace_label}")
                else:
                    print("Workspace: not initialized (provider discovery only)")
                visible_rows = (
                    rows
                    if parsed.all
                    else [
                        row
                        for row in rows
                        if row["configured"] or row["active"] or row["local_detected"]
                    ]
                )
                if not visible_rows:
                    print("No provider is configured or locally detected.")
                for row in visible_rows:
                    marker = "*" if row["active"] else " "
                    if row["ready"]:
                        state = "ready"
                    elif row["configured"]:
                        state = "configured, not ready"
                    elif row["local_detected"]:
                        state = "locally detected, not configured"
                    else:
                        state = "not configured"
                    model = row["model"] or row["suggested_model"] or "-"
                    missing = ", ".join(row["missing"])
                    suffix = f"; missing: {missing}" if missing else ""
                    missing_clients = ", ".join(row["missing_client_modules"])
                    if missing_clients:
                        suffix += f"; missing clients: {missing_clients}"
                    if row["error"]:
                        suffix += f"; error: {row['error']}"
                    if row["local_probe"].get("error"):
                        suffix += f"; local check: {row['local_probe']['error']}"
                    print(
                        f"{marker} {row['provider']}: {state}; model: {model}{suffix}"
                    )
                    if (row["configured"] or row["local_detected"]) and missing_clients:
                        print(f"    Install: {row['install_command']}")
                    if row["local_detected"] and not row["configured"]:
                        print(
                            "    Setup: xscientist setup my-research "
                            f"--provider {row['provider']} --model "
                            f"{shlex.quote(str(row['suggested_model']))}"
                        )
                hidden_count = len(rows) - len(visible_rows)
                if hidden_count:
                    print(f"Other providers hidden: {hidden_count} (use --all)")
            return 0

        if parsed.provider_command == "check":
            if parsed.max_cost_usd is not None and parsed.max_cost_usd <= 0:
                raise ProviderConfigError("--max-cost-usd must be greater than zero")
            config = load_provider_config(workspace, missing_ok=False)
            provider = str(parsed.name or config.get("active_provider") or "")
            if not provider:
                raise ProviderConfigError(
                    "no active provider is configured; pass a provider name or run "
                    "`xscientist provider activate NAME`"
                )
            row = next(
                item
                for item in provider_statuses(workspace, probe_local=True)
                if item["provider"] == provider
            )
            custom_prices: dict[str, object] = {}
            bfts_path = workspace / "bfts_config.yaml"
            if bfts_path.is_file():
                try:
                    bfts = yaml.safe_load(bfts_path.read_text(encoding="utf-8"))
                except yaml.YAMLError as exc:
                    raise ProviderConfigError(
                        f"cannot read BFTS config: {exc}"
                    ) from exc
                if isinstance(bfts, dict):
                    budget = bfts.get("llm_budget")
                    prices = (
                        budget.get("prices_per_million")
                        if isinstance(budget, dict)
                        else None
                    )
                    if isinstance(prices, dict):
                        custom_prices = prices
            model = str(row.get("model") or "")
            price = (
                resolve_model_price(model, prices_per_million=custom_prices)
                if model
                else None
            )
            error_codes: list[str] = []
            remediations: list[dict[str, str]] = []
            if not row["configured"]:
                error_codes.append("provider_not_configured")
                remediations.append(
                    {
                        "code": "configure_provider",
                        "command": f"xscientist provider add {provider}",
                    }
                )
            if row["missing"]:
                error_codes.append("provider_credentials_missing")
                remediations.append(
                    {
                        "code": "configure_credentials",
                        "command": f"xscientist provider add {provider}",
                    }
                )
            if row["missing_client_modules"]:
                error_codes.append("provider_client_missing")
                remediations.append(
                    {
                        "code": "install_provider_client",
                        "command": str(row["install_command"]),
                    }
                )
            if row["error"]:
                error_codes.append("provider_credential_file_unsafe")
                remediations.append(
                    {
                        "code": "restrict_credential_file",
                        "command": f"chmod 600 {config.get('env_file') or '.env'}",
                    }
                )
            local_probe = row["local_probe"]
            if local_probe["checked"] and not local_probe["service_reachable"]:
                error_codes.append("local_provider_unreachable")
                remediations.append(
                    {
                        "code": "start_local_provider",
                        "command": "ollama serve",
                    }
                )
            elif local_probe["checked"] and not local_probe["model_available"]:
                error_codes.append("local_model_missing")
                model_name = model.split("/", 1)[-1] if model else "<model>"
                remediations.append(
                    {
                        "code": "install_local_model",
                        "command": f"ollama pull {model_name}",
                    }
                )
            cost_ready = parsed.max_cost_usd is None or price is not None
            if not cost_ready:
                error_codes.append("unknown_model_price")
                remediations.append(
                    {
                        "code": "configure_model_price",
                        "command": (
                            "edit llm_budget.prices_per_million in bfts_config.yaml"
                        ),
                    }
                )
            payload = {
                "schema": "xscientist.provider-check.v1",
                "ok": bool(row["ready"] and cost_ready),
                "workspace": ".",
                "provider": provider,
                "model": model or None,
                "checks": {
                    "metadata_configured": bool(row["configured"]),
                    "credentials_present": bool(row["credentials_available"]),
                    "credentials_required": any(
                        field.required for field in PROVIDER_FIELDS[provider]
                    ),
                    "credential_validation": (
                        "presence_only"
                        if any(field.required for field in PROVIDER_FIELDS[provider])
                        else "not_required"
                    ),
                    "client_available": bool(row["client_available"]),
                    "model_price_known": price is not None,
                    "cost_limit_requested": parsed.max_cost_usd is not None,
                    "local_service_reachable": local_probe["service_reachable"],
                    "model_available": local_probe["model_available"],
                    "live_api_verified": bool(
                        local_probe["checked"] and local_probe["ok"]
                    ),
                },
                "price_per_million": price,
                "error_codes": error_codes,
                "remediations": remediations,
            }
            if parsed.as_json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                state = (
                    "local service and model verified"
                    if local_probe["checked"] and payload["ok"]
                    else (
                        "configuration checks passed; credentials not live-verified"
                        if payload["ok"]
                        and any(field.required for field in PROVIDER_FIELDS[provider])
                        else (
                            "configuration checks passed"
                            if payload["ok"]
                            else "not ready"
                        )
                    )
                )
                print(f"Provider {provider}: {state}")
                print(f"Model: {model or '-'}")
                if any(field.required for field in PROVIDER_FIELDS[provider]):
                    print(
                        "Credentials: "
                        + ("present" if row["credentials_available"] else "missing")
                        + " (presence check only; no paid API request was made)"
                    )
                else:
                    print("Credentials: not required for this local provider")
                print(
                    "Model pricing: "
                    + (
                        "$0.00 for local Ollama"
                        if provider == "ollama"
                        else (
                            "price known"
                            if price is not None
                            else "not configured (required only for --max-cost-usd)"
                        )
                    )
                )
                if local_probe.get("error"):
                    print(f"Local check: {local_probe['error']}")
                for remediation in remediations:
                    print(
                        "Next: "
                        + _contextual_action(
                            remediation["command"],
                            parsed.workspace,
                        )
                    )
            return 0 if payload["ok"] else 1

        if parsed.provider_command == "remove":
            config = remove_provider(workspace, parsed.name)
            active_provider = config.get("active_provider")
            bfts_updated = False
            if active_provider:
                active_entry = config["providers"][active_provider]
                bfts_updated = update_bfts_models(
                    workspace / "bfts_config.yaml", str(active_entry["model"])
                )
            payload = {
                "ok": True,
                "removed": parsed.name,
                "active_provider": active_provider,
                "credentials_removed": False,
                "bfts_updated": bfts_updated,
            }
            if parsed.as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Removed provider metadata: {parsed.name}")
                print("Stored credentials were left untouched.")
            return 0

        if parsed.provider_command == "activate":
            config = activate_provider(workspace, parsed.name)
            entry = config["providers"][parsed.name]
            model = str(entry["model"])
            bfts_updated = False
            if not parsed.no_update_bfts:
                bfts_updated = update_bfts_models(workspace / "bfts_config.yaml", model)
            payload = {
                "ok": True,
                "active_provider": parsed.name,
                "model": model,
                "bfts_updated": bfts_updated,
            }
            if parsed.as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Active provider: {parsed.name}")
                print(f"Default model: {model}")
                print(f"BFTS config updated: {bfts_updated}")
            return 0

        payload = _configure_provider(
            workspace,
            name=parsed.name,
            model_value=parsed.model,
            activate=not parsed.no_activate,
            update_bfts=not parsed.no_update_bfts,
            non_interactive=parsed.non_interactive,
        )
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Configured provider: {payload['provider']}")
            print(f"Default model: {payload['model']}")
            if payload["credentials_written"]:
                print("Credentials saved securely to: " f"{payload['env_file']}")
            else:
                print("Credentials: using existing environment or local env file")
            print(f"Active: {payload['active']}")
            print(f"BFTS config updated: {payload['bfts_updated']}")
            if not payload["client_available"]:
                print(
                    "Provider client missing: "
                    + ", ".join(payload["missing_client_modules"])
                )
                print(f"Install: {payload['install_command']}")
        return 0
    except (OSError, ProviderConfigError) as exc:
        if getattr(parsed, "as_json", False):
            print(
                json.dumps(
                    {
                        "schema": "xscientist.provider-error.v1",
                        "ok": False,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        else:
            print(f"xscientist provider: {exc}", file=sys.stderr)
        return 2


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


def _start_input_error(
    parsed: argparse.Namespace,
    message: str,
    *,
    returncode: int = 2,
    next_actions: Sequence[str] = (),
    phase: str = "input",
) -> int:
    """Keep semantic input failures useful for both people and automation."""

    if parsed.as_json:
        print(
            json.dumps(
                {
                    "schema": "xscientist.start.v1",
                    "ok": False,
                    "phase": phase,
                    "workspace": Path(parsed.directory).name or ".",
                    "returncode": returncode,
                    "error": str(message),
                    "next_actions": list(next_actions),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"xscientist start: {message}", file=sys.stderr)
    return returncode


def _run_start(parsed: argparse.Namespace) -> int:
    """Orchestrate the safe first-run path without hiding scientific gates."""

    import contextlib
    import io
    import os

    import yaml

    from ai_scientist.utils.atomic_io import atomic_write_text
    from ai_scientist.utils.auth_session import create_session, validate_session
    from ai_scientist.utils.llm_budget import resolve_model_price
    from .diagnostics import diagnose
    from .onboarding import WorkspaceInitError, create_workspace
    from .dependency_profiles import TASK_PROFILES
    from .provider_config import (
        DEFAULT_MODELS,
        ProviderConfigError,
        load_provider_config,
        load_workspace_environment,
        validate_provider_model,
    )
    from .research_git import ResearchGitError, repository_status
    from .research_vcs import ResearchRepository

    workspace = Path(parsed.directory).expanduser().resolve()
    new_workspace = not (workspace / ".xscientist" / "providers.json").is_file()
    try:
        _interactive_start_inputs(
            parsed,
            new_workspace=new_workspace,
        )
    except KeyboardInterrupt:
        return _start_input_error(parsed, "cancelled by user", returncode=130)
    except (EOFError, ValueError) as exc:
        return _start_input_error(parsed, str(exc))
    question = str(parsed.question or "").strip()
    if not question:
        return _start_input_error(parsed, "--question cannot be empty")
    if parsed.data_dir and parsed.allow_synthetic_data:
        return _start_input_error(
            parsed,
            "--data-dir and --allow-synthetic-data are mutually exclusive",
        )
    if (
        not parsed.prepare_only
        and not parsed.data_dir
        and not parsed.allow_synthetic_data
    ):
        return _start_input_error(
            parsed,
            "choose --data-dir PATH for empirical evidence or "
            "--allow-synthetic-data for an explicitly exploratory study",
        )
    for label, value in (
        ("--max-project-tokens", parsed.max_project_tokens),
        ("--max-project-hours", parsed.max_project_hours),
        ("--max-cost-usd", parsed.max_cost_usd),
    ):
        if value is not None and float(value) <= 0:
            return _start_input_error(parsed, f"{label} must be greater than zero")
    price_values = (
        parsed.price_input_per_million,
        parsed.price_output_per_million,
        parsed.price_cached_input_per_million,
    )
    if any(value is not None and float(value) < 0 for value in price_values):
        return _start_input_error(parsed, "explicit model prices must be non-negative")
    if (parsed.price_input_per_million is None) != (
        parsed.price_output_per_million is None
    ):
        return _start_input_error(
            parsed, "input and output prices must be supplied together"
        )

    requested_user = str(parsed.user or "").strip()
    authenticated, auth_status, session = validate_session()
    active_user = (
        str((session or {}).get("username") or "").strip() if authenticated else ""
    )
    if requested_user and active_user and requested_user != active_user:
        switch_command = "xscientist auth login --user " + shlex.quote(requested_user)
        return _start_input_error(
            parsed,
            f"--user {requested_user!r} conflicts with the "
            f"active research identity {active_user!r}; switch explicitly with "
            f"`{switch_command}`",
            next_actions=[switch_command],
        )
    if new_workspace and not parsed.provider:
        return _start_input_error(
            parsed,
            "--provider is required for a new workspace in non-interactive mode; "
            "inspect safe local choices with `xscientist provider list`",
            next_actions=["xscientist provider list", "xscientist start --help"],
        )

    phases: dict[str, object] = {}
    try:
        config_exists = (workspace / ".xscientist" / "providers.json").is_file()
        existing_config = (
            load_provider_config(workspace, missing_ok=False) if config_exists else {}
        )
        selected_provider = str(
            parsed.provider or existing_config.get("active_provider") or ""
        )
        if not selected_provider:
            raise ProviderConfigError(
                "no active provider is configured; pass --provider and --model"
            )
        provider_entry = (existing_config.get("providers") or {}).get(
            selected_provider, {}
        )
        existing_model = (
            str(provider_entry.get("model") or "")
            if isinstance(provider_entry, dict)
            else ""
        )
        selected_model = (
            parsed.model or existing_model or DEFAULT_MODELS.get(selected_provider)
        )
        selected_task = str(
            parsed.task
            or ("paper" if parsed.autopilot == "publication" else "research")
        )
        selected_capabilities = tuple(
            str(item) for item in TASK_PROFILES[selected_task]["capabilities"]
        )
        if not selected_model:
            if parsed.non_interactive or not sys.stdin.isatty():
                raise ProviderConfigError(
                    f"--model is required for provider {selected_provider!r}; "
                    f"use {selected_provider}/<model>"
                )
            selected_model = input(
                f"Model ID for {selected_provider} "
                f"(for example {selected_provider}/<model>): "
            ).strip()
        selected_provider, selected_model = validate_provider_model(
            selected_provider, selected_model
        )
        if not config_exists:
            create_workspace(
                workspace,
                profile=parsed.profile,
                provider=selected_provider,
                model=selected_model,
                force=parsed.force,
                task=selected_task,
                capabilities=selected_capabilities,
                provider_required=True,
            )
        phases["workspace"] = {
            "ok": True,
            "created": not config_exists,
            "task": selected_task,
            "capabilities": list(selected_capabilities),
        }

        if not authenticated:
            username = requested_user
            if not username and not parsed.non_interactive and sys.stdin.isatty():
                username = input("Local research actor name: ").strip()
            if username:
                session = create_session(username=username)
                authenticated, auth_status = True, "ok"
        resolved_actor = (
            str((session or {}).get("username") or "").strip() if authenticated else ""
        )
        phases["auth"] = {
            "ok": authenticated,
            "status": auth_status,
            "user": resolved_actor or None,
            "session_created": bool(authenticated and not active_user),
        }

        topic = f"# Research question\n\n{question}\n"
        established_topic = workspace / "00_config" / "topic.md"
        if (
            established_topic.is_file()
            and established_topic.read_text(encoding="utf-8") != topic
        ):
            raise WorkspaceInitError(
                "this workspace already contains a different research question; "
                "reuse the original question to resume or choose a new directory"
            )
        atomic_write_text(workspace / "topic.md", topic)
        if not (workspace / "research.yaml").is_file():
            repository = ResearchRepository.init(
                workspace,
                name=workspace.name,
                question=topic,
                policy="milestone",
                actor=resolved_actor or "xscientist",
            )
            status = repository.status()
            vcs_created = True
        else:
            atomic_write_text(workspace / "question.md", topic)
            status = repository_status(workspace)
            vcs_created = False
        phases["research_vcs"] = {
            "ok": True,
            "created": vcs_created,
            "branch": status.get("branch"),
            "checkpoint_id": (status.get("last_checkpoint") or {}).get("checkpoint_id"),
        }

        if parsed.skip_credentials:
            provider_result: dict[str, object] = {
                "ok": False,
                "reason": "credentials explicitly skipped",
            }
        else:
            provider_result = _configure_provider(
                workspace,
                name=selected_provider,
                model_value=selected_model,
                non_interactive=parsed.non_interactive,
            )
        phases["provider"] = {
            "ok": bool(provider_result.get("ok")),
            "ready": bool(provider_result.get("ready")),
            "provider": selected_provider,
            "model": selected_model,
            "reason": provider_result.get("reason"),
        }

        load_result = load_workspace_environment(workspace)
        if load_result.get("error"):
            raise ProviderConfigError(str(load_result["error"]))

        budget_path = workspace / "bfts_config.yaml"
        budget_original = budget_path.read_text(encoding="utf-8")
        budget_payload = yaml.safe_load(budget_original)
        if not isinstance(budget_payload, dict):
            raise ProviderConfigError("BFTS configuration must be a mapping")
        budget_section = budget_payload.setdefault("llm_budget", {})
        if not isinstance(budget_section, dict):
            raise ProviderConfigError("BFTS llm_budget must be a mapping")
        custom_prices = budget_section.setdefault("prices_per_million", {})
        if not isinstance(custom_prices, dict):
            raise ProviderConfigError("llm_budget.prices_per_million must be a mapping")
        if parsed.price_input_per_million is not None:
            explicit_price = {
                "input": float(parsed.price_input_per_million),
                "output": float(parsed.price_output_per_million),
            }
            if parsed.price_cached_input_per_million is not None:
                explicit_price["cached_input"] = float(
                    parsed.price_cached_input_per_million
                )
            custom_prices[selected_model] = explicit_price
            generated_header = (
                budget_original.splitlines()[0] + "\n"
                if budget_original.startswith("# Generated by xscientist init ")
                else ""
            )
            atomic_write_text(
                budget_path,
                generated_header
                + yaml.safe_dump(budget_payload, sort_keys=False, allow_unicode=True),
            )
        price = resolve_model_price(
            selected_model,
            prices_per_million=custom_prices,
        )
        phases["budget"] = {
            "ok": parsed.max_cost_usd is None or price is not None,
            "cost_limit_enabled": parsed.max_cost_usd is not None,
            "model": selected_model,
            "price_configured": price is not None,
            "price_source": (
                "workspace"
                if selected_model in custom_prices
                else ("bundled" if price is not None else None)
            ),
        }
        if parsed.max_cost_usd is not None and price is None:
            payload = {
                "schema": "xscientist.start.v1",
                "ok": False,
                "phase": "budget",
                "error_code": "unknown_model_price",
                "error": (
                    f"no price is configured for model {selected_model!r}; "
                    "XScientist will not guess or treat it as free"
                ),
                "workspace": ".",
                "phases": phases,
                "next_actions": [
                    "rerun with --price-input-per-million PRICE "
                    "--price-output-per-million PRICE",
                    "or edit llm_budget.prices_per_million in bfts_config.yaml",
                ],
            }
            if parsed.as_json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print("XScientist cannot enforce the requested cost limit yet.")
                print(payload["error"])
                for action in payload["next_actions"]:
                    print(f"  {action}")
            return 1

        if parsed.build_executor:
            from .executor_manager import prepare_executor

            executor_status = prepare_executor(workspace)
            phases["executor_build"] = executor_status

        report = diagnose(
            workspace,
            task=selected_task,
            provider=selected_provider,
            deep=True,
        )
        _persist_readiness_report(workspace, report)
        phases["doctor"] = report
    except (
        OSError,
        ResearchGitError,
        ProviderConfigError,
        WorkspaceInitError,
        ValueError,
    ) as exc:
        payload = {
            "schema": "xscientist.start.v1",
            "ok": False,
            "phase": "prepare",
            "error": str(exc),
            "workspace": ".",
            "phases": phases,
        }
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(
                f"xscientist start stopped during preparation: {exc}", file=sys.stderr
            )
        return 2

    if not report["ok"]:
        payload = {
            "schema": "xscientist.start.v1",
            "ok": False,
            "phase": "doctor",
            "workspace": ".",
            "phases": phases,
            "next_actions": report["next_actions"],
        }
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("XScientist is configured, but the automated run is not ready.")
            if resolved_actor:
                print(f"Research actor: {resolved_actor}")
            print("Resolve these items in order:")
            for action in report["next_actions"]:
                print(f"  {_contextual_action(action, parsed.directory)}")
            print(
                "Retry: after fixing the first blocker, press Up in this shell "
                "and rerun the same start command."
            )
        return 1

    if parsed.prepare_only:
        payload = {
            "schema": "xscientist.start.v1",
            "ok": True,
            "phase": "ready",
            "workspace": ".",
            "phases": phases,
        }
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("Workspace is ready for automated research.")
            if resolved_actor:
                print(f"Research actor: {resolved_actor}")
        return 0

    project_args = [
        str(workspace),
        "--output-root",
        str(workspace / "outputs"),
        "--question",
        question,
        "--autopilot",
        parsed.autopilot,
        "--bfts-config",
        str(workspace / "bfts_config.yaml"),
        "--research-vcs-strict",
    ]
    if parsed.data_dir:
        project_args.extend(["--data-dir", str(Path(parsed.data_dir).expanduser())])
    else:
        project_args.append("--allow-synthetic-data")
    for flag, value in (
        ("--max-project-tokens", parsed.max_project_tokens),
        ("--max-project-hours", parsed.max_project_hours),
        ("--max-cost-usd", parsed.max_cost_usd),
    ):
        if value is not None:
            project_args.extend([flag, str(value)])

    previous_workspace = os.environ.get("XSCIENTIST_WORKSPACE")
    os.environ["XSCIENTIST_WORKSPACE"] = str(workspace)
    try:
        if parsed.as_json:
            captured_out, captured_err = io.StringIO(), io.StringIO()
            with (
                contextlib.redirect_stdout(captured_out),
                contextlib.redirect_stderr(captured_err),
            ):
                returncode = project_main(project_args)
        else:
            returncode = project_main(project_args)
    finally:
        if previous_workspace is None:
            os.environ.pop("XSCIENTIST_WORKSPACE", None)
        else:
            os.environ["XSCIENTIST_WORKSPACE"] = previous_workspace

    payload = {
        "schema": "xscientist.start.v1",
        "ok": returncode == 0,
        "phase": "complete" if returncode == 0 else "research",
        "workspace": ".",
        "project": ".",
        "returncode": returncode,
        "research_dag": "outputs/views/"
        + workspace.name
        + "/research-dag/research-dag.html",
        "phases": phases,
    }
    if parsed.as_json:
        if returncode:
            payload["error"] = captured_err.getvalue().strip().splitlines()[-1:]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif returncode == 0:
        print("Automated research completed with a local Research VCS history.")
        print(f"Open the research DAG under: {payload['research_dag']}")
    return returncode


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv in (["--help"], ["-h"]):
        _print_curated_help()
        return 0
    if raw_argv and raw_argv[0] == "help":
        if raw_argv[1:] in ([], ["--all"]):
            _print_curated_help(include_advanced=raw_argv[1:] == ["--all"])
            return 0
        print("xscientist help: use `xscientist COMMAND --help`", file=sys.stderr)
        return 2
    if raw_argv and raw_argv[0] in _DELEGATES:
        return _DELEGATES[raw_argv[0]](raw_argv[1:])

    workspace_state = (
        {"loaded": False}
        if raw_argv
        and raw_argv[0]
        in {
            "provider",
            "init",
            "start",
            "runs",
            "executor",
            "upgrade",
            "completion",
            "conformance",
            "benchmark",
            "metrics",
            "setup",
            "doctor",
            "capability",
        }
        else _bootstrap_workspace_environment()
    )
    if workspace_state.get("error") and (
        not raw_argv
        or raw_argv[0]
        not in {"provider", "info", "init", "start", "setup", "doctor", "capability"}
    ):
        print(
            f"XScientist workspace configuration error: {workspace_state['error']}",
            file=sys.stderr,
        )
        return 2

    lazy_parser_builders = {
        "start": _build_start_parser,
        "setup": _build_setup_parser,
        "doctor": _build_doctor_parser,
        "capability": _build_capability_parser,
    }
    lazy_builder = lazy_parser_builders.get(raw_argv[0]) if raw_argv else None
    if lazy_builder is not None:
        parsed = lazy_builder().parse_args(raw_argv[1:])
        parsed.command = raw_argv[0]
    else:
        parser = _build_parser()
        parsed = parser.parse_args(raw_argv)
    if parsed.command == "serve":
        from .service import run_server

        try:
            run_server(
                host=parsed.host,
                port=parsed.port,
                work_dir=parsed.work_dir,
                output_root=parsed.output_root,
                max_workers=parsed.max_workers,
                max_output_chars=parsed.max_output_chars,
                max_workspace_bytes=parsed.max_workspace_bytes,
                max_workspace_files=parsed.max_workspace_files,
                state_dir=parsed.state_dir,
                reload=parsed.reload,
                allow_unauthenticated=parsed.allow_unauthenticated,
            )
        except (ModuleNotFoundError, OSError, ValueError) as exc:
            print(f"xscientist serve: {exc}", file=sys.stderr)
            return 2
        return 0
    if parsed.command == "start":
        from .run_control import (
            RunControlError,
            begin_active_run,
            finish_active_run,
            launch_detached_run,
            read_run_logs,
        )

        if parsed.detach:
            try:
                payload = launch_detached_run(parsed.directory, raw_argv)
            except (OSError, RunControlError, ValueError) as exc:
                return _start_input_error(parsed, str(exc), phase="launch")
            detached_status = str(payload.get("status") or "unknown")
            startup_failed = detached_status in {
                "failed",
                "cancelled",
                "interrupted",
            }
            if startup_failed:
                try:
                    logs = read_run_logs(
                        parsed.directory,
                        str(payload["id"]),
                        stream="both",
                        tail=500,
                    )
                except (OSError, RunControlError, ValueError):
                    logs = {}
                payload["failure_summary"] = _run_failure_summary(logs)
            if parsed.as_json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            elif startup_failed:
                print(
                    f"Detached research run stopped during startup: {payload['id']} "
                    f"({detached_status})",
                    file=sys.stderr,
                )
                if payload.get("failure_summary"):
                    print(f"Failure: {payload['failure_summary']}", file=sys.stderr)
                print(
                    "Inspect: xscientist runs show "
                    f"{payload['id']} --workspace "
                    f"{shlex.quote(str(parsed.directory))}",
                    file=sys.stderr,
                )
                print(
                    "Retry after repair: xscientist runs resume "
                    f"{payload['id']} --workspace "
                    f"{shlex.quote(str(parsed.directory))}",
                    file=sys.stderr,
                )
            elif detached_status == "succeeded":
                print(f"Detached research run completed: {payload['id']}")
                print(f"Workspace: {payload['workspace']}")
            else:
                print(f"Detached research run started: {payload['id']}")
                print(f"Workspace: {payload['workspace']}")
                print(
                    "Watch: xscientist runs watch "
                    f"{payload['id']} --workspace "
                    f"{shlex.quote(str(parsed.directory))}"
                )
            return 1 if startup_failed else 0
        active_run = begin_active_run()
        returncode = 1
        try:
            returncode = _run_start(parsed)
            return returncode
        finally:
            finish_active_run(active_run, returncode)
    if parsed.command == "runs":
        from .run_control import (
            RunControlError,
            cancel_run,
            get_run,
            list_runs,
            read_run_logs,
            resume_run,
        )

        try:
            if parsed.runs_command == "list":
                rows = list_runs(parsed.workspace)
                payload = {"schema": "xscientist.local-runs.v1", "items": rows}
            elif parsed.runs_command == "show":
                payload = get_run(parsed.workspace, parsed.run_id)
                if payload.get("status") in {
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    logs = read_run_logs(
                        parsed.workspace,
                        parsed.run_id,
                        stream="both",
                        tail=500,
                    )
                    payload["failure_summary"] = _run_failure_summary(logs)
            elif parsed.runs_command == "logs":
                payload = read_run_logs(
                    parsed.workspace,
                    parsed.run_id,
                    stream=parsed.stream,
                    tail=parsed.tail,
                )
            elif parsed.runs_command == "cancel":
                payload = cancel_run(parsed.workspace, parsed.run_id)
            elif parsed.runs_command == "resume":
                payload = resume_run(
                    parsed.workspace,
                    parsed.run_id,
                    force=parsed.force,
                )
            else:
                import time

                previous = None
                while True:
                    payload = get_run(parsed.workspace, parsed.run_id)
                    status = payload.get("status")
                    if not parsed.as_json and status != previous:
                        print(f"{payload['id']}  {status}")
                    previous = status
                    if status not in {"queued", "running", "cancelling"}:
                        break
                    time.sleep(parsed.interval)
                if payload.get("status") in {
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    logs = read_run_logs(
                        parsed.workspace,
                        parsed.run_id,
                        stream="both",
                        tail=500,
                    )
                    payload["failure_summary"] = _run_failure_summary(logs)
        except (OSError, RunControlError, ValueError) as exc:
            if getattr(parsed, "as_json", False):
                print(
                    json.dumps(
                        {
                            "schema": "xscientist.run-error.v1",
                            "ok": False,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            else:
                print(f"xscientist runs: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif parsed.runs_command == "list":
            if not payload["items"]:
                print("No detached runs in this workspace.")
            for item in payload["items"]:
                duration = item.get("duration_seconds")
                duration_text = (
                    f"{duration:.1f}s" if isinstance(duration, float) else "-"
                )
                route = str(
                    item.get("model") or item.get("provider") or "provider pending"
                )
                print(
                    f"{item['id']}  {item['status']:<11} "
                    f"{duration_text:>8}  {item.get('profile') or 'balanced':<11} "
                    f"{route}  {_format_run_timestamp(item.get('created_at'))}"
                )
        elif parsed.runs_command == "logs":
            summary = _run_failure_summary(payload)
            if summary:
                print(f"Summary: {summary}")
            partial_streams = [
                stream
                for stream in ("stdout", "stderr")
                if _structured_log_tail_is_partial(payload.get(stream, []))
            ]
            if partial_streams:
                print(
                    "Note: this structured log tail starts mid-document; "
                    "increase --tail or use --json for machine-readable output."
                )
            if payload["stdout"]:
                print("--- stdout ---")
                print("\n".join(payload["stdout"]))
            if payload["stderr"]:
                print("--- stderr ---", file=sys.stderr)
                print("\n".join(payload["stderr"]), file=sys.stderr)
        elif parsed.runs_command == "watch":
            if payload.get("failure_summary"):
                print(f"Failure: {payload['failure_summary']}")
            if payload.get("status") in {"failed", "cancelled", "interrupted"}:
                print(
                    f"Logs: xscientist runs logs {payload['id']} --workspace "
                    f"{shlex.quote(str(parsed.workspace))}"
                )
        else:
            print(f"Run {payload['id']}: {payload['status']}")
            if parsed.runs_command == "show":
                print(f"Profile: {payload.get('profile') or 'balanced'}")
                print(f"Task: {payload.get('task') or 'research'}")
                print(f"Provider: {payload.get('provider') or 'pending'}")
                print(f"Model: {payload.get('model') or 'pending'}")
                if payload.get("duration_seconds") is not None:
                    print(f"Duration: {payload['duration_seconds']}s")
                if payload.get("returncode") is not None:
                    print(f"Exit code: {payload['returncode']}")
                if payload.get("failure_summary"):
                    print(f"Failure: {payload['failure_summary']}")
                    print(
                        "Logs: xscientist runs logs "
                        f"{payload['id']} --workspace "
                        f"{shlex.quote(str(parsed.workspace))}"
                    )
            if parsed.runs_command == "resume":
                print(
                    f"Watch: xscientist runs watch {payload['id']} "
                    f"--workspace {shlex.quote(str(parsed.workspace))}"
                )
        if parsed.runs_command in {"show", "watch"} and payload.get("status") in {
            "failed",
            "cancelled",
            "interrupted",
        }:
            return 1
        return 0
    if parsed.command == "executor":
        from .executor_manager import (
            ExecutorManagerError,
            build_executor,
            inspect_executor,
            prepare_executor,
        )

        try:
            if parsed.executor_command == "check":
                payload = inspect_executor(parsed.workspace)
            elif parsed.executor_command == "build":
                payload = build_executor(parsed.workspace)
            else:
                payload = prepare_executor(
                    parsed.workspace,
                    update=parsed.executor_command == "update",
                )
        except (OSError, ExecutorManagerError, ValueError) as exc:
            if parsed.as_json:
                print(
                    json.dumps(
                        {
                            "schema": "xscientist.executor-error.v1",
                            "ok": False,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            else:
                print(f"xscientist executor: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Executor image: {payload['image']}")
            print(f"Docker daemon: {payload['daemon_ready']}")
            print(f"Image available: {payload['image_available']}")
            print(f"Version match: {payload['version_match']}")
            if payload.get("install_source"):
                print(f"Install source: {payload['install_source']}")
            if payload.get("error"):
                print(f"Problem: {payload['error']}")
            if payload.get("next_action"):
                next_action = str(payload["next_action"])
                if next_action.startswith("xscientist executor prepare"):
                    next_action = "xscientist executor prepare --workspace ."
                print(
                    "Next: "
                    + _contextual_action(
                        next_action,
                        parsed.workspace,
                    )
                )
        return 0 if payload["ok"] else 1
    if parsed.command == "upgrade":
        from .upgrade_check import check_upgrade

        if parsed.timeout <= 0:
            print(
                "xscientist upgrade: --timeout must be greater than zero",
                file=sys.stderr,
            )
            return 2
        payload = check_upgrade(
            parsed.workspace,
            online=parsed.online,
            timeout=parsed.timeout,
        )
        _record_local_metric("upgrade_check", ok=bool(payload["ok"]))
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            package = payload["package"]
            print(f"Installed version: {package['installed_version']}")
            if package["online_checked"]:
                print(f"Latest version: {package['latest_version'] or 'unavailable'}")
                if package.get("index_relation") == "newer_than_index":
                    print(
                        "Release status: this installation is newer than PyPI "
                        "and is likely a development or unreleased build."
                    )
                elif package.get("index_relation") == "current":
                    print("Release status: current with PyPI")
                elif package.get("index_relation") == "update_available":
                    print("Release status: an update is available")
            print(f"Workspace compatible: {payload['compatible']}")
            for name, check in payload["checks"].items():
                state = "compatible" if check["compatible"] else "incompatible"
                suffix = " (not present)" if not check["present"] else ""
                print(f"{name}: {state}{suffix}")
            if package["online_error"]:
                print(
                    f"Online check failed: {package['online_error']}", file=sys.stderr
                )
            for remediation in payload["remediations"]:
                print(f"Next: {remediation}")
        return 0 if payload["ok"] else 1
    if parsed.command == "completion":
        from .completion import completion_script

        print(completion_script(parsed.shell), end="")
        return 0
    if parsed.command == "conformance":
        from .conformance import check_conformance, init_conformance_kit

        try:
            if parsed.conformance_command == "init":
                payload = init_conformance_kit(parsed.directory)
            else:
                payload = check_conformance(parsed.target, schema_name=parsed.schema)
                _record_local_metric("conformance_check", ok=bool(payload["ok"]))
        except (OSError, ValueError) as exc:
            print(f"xscientist conformance: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif parsed.conformance_command == "init":
            print(f"Conformance kit ready: {payload['directory']}")
            print(f"Cases: {payload['cases']}")
            print(
                "Run: xscientist conformance check "
                + shlex.quote(str(parsed.directory))
            )
        else:
            print(
                f"Protocol conformance: {payload['passed']}/{payload['total']} passed"
            )
            for case in payload["cases"]:
                state = "PASS" if case["passed"] else "FAIL"
                expectation = "valid" if case["expected_valid"] else "invalid"
                actual = "valid" if case["actual_valid"] else "invalid"
                print(
                    f"{state}  {case['file']} ({case['schema_name']}; "
                    f"expected {expectation}, got {actual})"
                )
        return 0 if payload["ok"] else 1
    if parsed.command == "benchmark":
        from .benchmark import benchmark_first_run

        try:
            payload = benchmark_first_run(
                parsed.workspace,
                profile=parsed.profile,
                max_seconds=parsed.max_seconds,
            )
        except (OSError, ValueError) as exc:
            print(f"xscientist benchmark: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"First-run benchmark: {payload['duration_seconds']}s")
            print(
                "Evidence DAG: "
                f"{payload['research']['dag_nodes']} nodes / "
                f"{payload['research']['dag_relations']} relations"
            )
            print(f"Scientific closure: {payload['research']['closure']}")
            print("Cost: $0.00; network and model providers were not used.")
            if parsed.max_seconds is not None:
                print(f"Threshold passed: {payload['threshold_passed']}")
        return 0 if payload["ok"] else 1
    if parsed.command == "metrics":
        from .usage_metrics import export_metrics, metrics_status, set_metrics_enabled

        if parsed.metrics_command == "enable":
            payload = set_metrics_enabled(True)
        elif parsed.metrics_command == "disable":
            payload = set_metrics_enabled(False)
        elif parsed.metrics_command == "export":
            payload = export_metrics()
        else:
            payload = metrics_status()
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Local usage metrics enabled: {payload['enabled']}")
            print(f"Stored events: {payload['event_count']}")
            print("Network transmission: disabled")
            print("Excluded: " + ", ".join(payload["excluded_fields"]))
            if parsed.metrics_command == "export":
                for event in payload["events"]:
                    print(json.dumps(event, sort_keys=True))
        return 0
    if parsed.command == "demo":
        import webbrowser

        from .demo import create_autopilot_demo, create_demo
        from .research_git import ResearchGitError

        try:
            if parsed.autopilot:
                payload = create_autopilot_demo(
                    parsed.directory,
                    profile=parsed.autopilot_profile,
                    language=parsed.lang,
                    git_user_name=parsed.git_user_name,
                    git_user_email=parsed.git_user_email,
                )
            else:
                payload = create_demo(
                    parsed.directory,
                    language=parsed.lang,
                    git_user_name=parsed.git_user_name,
                    git_user_email=parsed.git_user_email,
                )
        except (OSError, ResearchGitError, ValueError) as exc:
            if parsed.as_json:
                print(
                    json.dumps(
                        {
                            "schema": "xscientist.demo.v1",
                            "ok": False,
                            "error_code": "demo_creation_failed",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            else:
                print(f"xscientist demo: {exc}", file=sys.stderr)
            return 2
        if parsed.open_browser:
            payload["browser_opened"] = bool(
                webbrowser.open(Path(payload["dag"]["html"]).as_uri())
            )
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            language = _selected_language(parsed.lang)
            if language == "zh":
                closure = {
                    "blocked": "已完成（结论存在待解决争议）",
                    "complete": "已完成",
                    "verified": "已验证",
                }.get(payload["dag"]["closure"], payload["dag"]["closure"])
                print("XScientist 零 Provider 演示已就绪。")
                print(
                    f"证据 DAG：{payload['dag']['nodes']} 个节点，"
                    f"{payload['dag']['relations']} 条关系"
                )
                print(f"科学闭环：{closure}")
                if payload["dag"]["closure"] == "blocked":
                    print(
                        "含义：演示成功；反驳证据缩小了结论范围，需要继续做边界检验。"
                    )
                print(f"打开：{payload['dag']['html']}")
                print("费用：$0.00；未使用 Provider 或网络。")
            else:
                print("XScientist provider-free demo is ready.")
                print(
                    f"Evidence DAG: {payload['dag']['nodes']} nodes, "
                    f"{payload['dag']['relations']} relations"
                )
                print(f"Scientific closure: {payload['dag']['closure']}")
                if payload["dag"]["closure"] == "blocked":
                    print(
                        "Meaning: expected scientific result—refuting evidence "
                        "narrowed the claim; the demo itself succeeded."
                    )
                print(f"Open: {payload['dag']['html']}")
                print("Cost: $0.00; no provider or network was used.")
            if language == "zh":
                print("下一步：xscientist status " + shlex.quote(str(parsed.directory)))
            else:
                print("Next: xscientist status " + shlex.quote(str(parsed.directory)))
            if payload.get("autopilot_fixture"):
                profile = payload["autopilot_fixture"]["profile"]
                if language == "zh":
                    print(f"Autopilot 样例：完整 / 确定性 / 可恢复（{profile}）")
                else:
                    print(
                        "Autopilot fixture: complete / deterministic / resumable "
                        f"({profile})"
                    )
        _record_local_metric("demo", ok=True)
        return 0
    if parsed.command == "status":
        from .workspace_status import build_workspace_status

        payload = build_workspace_status(parsed.workspace, language=parsed.lang)
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            language = _selected_language(parsed.lang)
            research = payload["research"]
            run = payload["run"]
            result = payload["result"]
            background_run = payload.get("background_run")
            state_labels = {
                "en": {
                    "invalid": "invalid workspace state",
                    "needs_attention": "needs attention",
                    "running": "running",
                    "scientific_followup": "run complete; more evidence needed",
                    "complete": "complete and independently verified",
                    "ready": "ready",
                    "not_started": "not started",
                },
                "zh": {
                    "invalid": "工作区状态无效",
                    "needs_attention": "需要处理",
                    "running": "运行中",
                    "scientific_followup": "运行完成，仍需补充证据",
                    "complete": "已完成并通过独立验证",
                    "ready": "已就绪",
                    "not_started": "尚未开始",
                },
            }
            if language == "zh":
                print(f"工作区：{payload['workspace']}")
                print(f"状态：{state_labels['zh'][payload['operational_state']]}")
            else:
                print(f"Workspace: {payload['workspace']}")
                print(f"State: {state_labels['en'][payload['operational_state']]}")
            if parsed.verbose or research["staged"]:
                print(
                    ("科研历史：" if language == "zh" else "Research: ")
                    + (
                        (
                            f"{research['branch']} / 暂存={research['staged']}"
                            if language == "zh"
                            else f"{research['branch']} / staged={research['staged']}"
                        )
                        if research["initialized"]
                        else ("未初始化" if language == "zh" else "not initialized")
                    )
                )
            if (research.get("guide") or {}).get("progress"):
                progress = research["guide"]["progress"]
                label = "科学进度" if language == "zh" else "Scientific progress"
                print(
                    f"{label}: {progress['completed_stages']}/"
                    f"{progress['total_stages']} ({progress['percent']}%)"
                )
            if run["started"]:
                run_state = run["current_stage"] or (
                    "已启动" if language == "zh" else "started"
                )
                progress = (research.get("guide") or {}).get("progress") or {}
                closure_pending = str(run_state) == "complete" and (
                    float(progress.get("percent") or 0) < 100
                    or result.get("epistemic_status") != "verified"
                )
                if closure_pending:
                    run_state = (
                        "运行已完成；科学闭环待完成"
                        if language == "zh"
                        else "run complete; scientific closure pending"
                    )
                elif language == "zh":
                    run_state = {
                        "complete": "已完成",
                        "failed": "失败",
                        "cancelled": "已取消",
                        "running": "运行中",
                    }.get(str(run_state), run_state)
            elif result["dag_html"]:
                run_state = (
                    "未启动（已有离线科研历史）"
                    if language == "zh"
                    else "not started (offline research history is available)"
                )
            else:
                run_state = "未启动" if language == "zh" else "not started"
            if parsed.verbose:
                print(
                    f"{'科研流水线' if language == 'zh' else 'Research pipeline'}: {run_state}"
                )
            background_inspect_command = None
            if isinstance(background_run, dict) and parsed.verbose:
                background_status = str(background_run.get("status") or "unknown")
                if language == "zh":
                    background_status = {
                        "queued": "排队中",
                        "running": "运行中",
                        "cancelling": "取消中",
                        "cancelled": "已取消",
                        "failed": "失败",
                        "succeeded": "成功",
                        "interrupted": "已中断",
                    }.get(background_status, background_status)
                    print(
                        f"最近后台任务：{background_run.get('id')} / "
                        f"{background_status}"
                    )
                else:
                    print(
                        f"Latest background run: {background_run.get('id')} / "
                        f"{background_status}"
                    )
                if background_run.get("id"):
                    command = (
                        f"xscientist runs show {background_run['id']} --workspace "
                        f"{shlex.quote(str(parsed.workspace or '.'))}"
                    )
                    background_inspect_command = command
                    print(f"{'查看' if language == 'zh' else 'Inspect'}: {command}")
            if payload["budget"]["available"]:
                used = payload["budget"]["used"] or {}
                cost = float(used.get("cost_usd") or 0)
                input_tokens = int(used.get("input_tokens") or 0)
                output_tokens = int(used.get("output_tokens") or 0)
                if language == "zh" and parsed.verbose:
                    print(
                        f"预算已用：${cost:.2f}；输入令牌 {input_tokens:,}，"
                        f"输出令牌 {output_tokens:,}"
                    )
                elif language == "en" and parsed.verbose:
                    print(
                        f"Budget used: ${cost:.2f}; {input_tokens:,} input tokens; "
                        f"{output_tokens:,} output tokens"
                    )
                else:
                    print(f"{'成本' if language == 'zh' else 'Cost'}: ${cost:.2f}")
            if result["dag_html"]:
                label = "证据 DAG" if language == "zh" else "Evidence DAG"
                print(f"{label}: {result['dag_html']}")
            if result.get("epistemic_status"):
                label = "结论状态" if language == "zh" else "Result status"
                status_value = result["epistemic_status"]
                if language == "zh":
                    status_value = {
                        "machine_synthesized_unverified": "机器综合，尚未独立验证",
                        "verified": "已验证",
                        "contested": "存在争议",
                    }.get(status_value, status_value)
                else:
                    status_value = {
                        "machine_synthesized_unverified": (
                            "machine-synthesized; not independently verified"
                        ),
                        "verified": "verified",
                        "contested": "contested",
                    }.get(status_value, status_value)
                print(f"{label}: {status_value}")
            for error in payload["errors"]:
                target = f" ({error.get('file')})" if error.get("file") else ""
                problem_label = "问题" if language == "zh" else "Problem"
                print(
                    f"{problem_label}: {error.get('code', 'status_error')}{target}: "
                    f"{error.get('detail', '')}",
                    file=sys.stderr,
                )
                if error.get("remediation"):
                    repair_label = "修复" if language == "zh" else "Repair"
                    print(f"{repair_label}: {error['remediation']}", file=sys.stderr)
            if payload["next_steps"]:
                step = payload["next_steps"][0]
                step_title = step.get("title") or step.get("code")
                if (
                    language == "zh"
                    and step.get("code") == "repair_failed_background_run"
                ):
                    step_title = "检查并修复最近失败的后台任务"
                contextual_command = None
                if step.get("command"):
                    command = step["command"]
                    if " research discovery template " not in f" {command} ":
                        command = _contextual_action(command, parsed.workspace)
                    contextual_command = command
                print(f"{'下一步' if language == 'zh' else 'Next'}: " f"{step_title}")
                if (
                    contextual_command
                    and contextual_command != background_inspect_command
                ):
                    print(
                        f"{'运行' if language == 'zh' else 'Run'}:  "
                        f"{contextual_command}"
                    )
        status_ok = bool(payload["ok"] and not payload["attention_required"])
        _record_local_metric("status", ok=status_ok)
        return 0 if status_ok else 1
    if parsed.command == "info":
        payload = _installation_info()
        if parsed.as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"XScientist {payload['version']}")
            print(f"Installation: {payload['installation_profile']}")
            if payload["research_runtime_ready"]:
                print("Research runtime: ready")
            else:
                missing = ", ".join(payload["missing_research_packages"])
                print(f"Research runtime: not ready (missing: {missing})")
            if payload["provider_configured"]:
                print(
                    f"Provider: {payload['active_provider']} "
                    f"({payload['provider_client_status']})"
                )
            elif payload["suggested_provider"] == "ollama":
                models = payload["discovered_local_models"]
                print(
                    f"Provider: not configured; detected {len(models)} Ollama model(s)"
                )
                print(f"Suggested model: {models[0]}")
            else:
                print("Provider: not configured")
            if payload.get("recommended_install"):
                print(f"Recommended install: {payload['recommended_install']}")
            elif payload.get("recommended_setup"):
                print(f"Next setup: {payload['recommended_setup']}")
            else:
                print("Runtime dependencies: already installed")
            print(f"Quick start: {payload['quickstart']}")
        return 0
    if parsed.command == "init":
        from .onboarding import WorkspaceInitError, create_workspace

        try:
            payload = create_workspace(
                parsed.directory,
                profile=parsed.profile,
                provider=parsed.provider,
                model=parsed.model,
                force=parsed.force,
            )
        except (OSError, WorkspaceInitError) as exc:
            if parsed.as_json:
                print(
                    json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
                    file=sys.stderr,
                )
            else:
                print(f"xscientist init: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Created XScientist workspace: {payload['workspace']}")
            print("No secrets were written.")
            print("Next steps:")
            for index, step in enumerate(payload["next_steps"], start=1):
                print(f"  {index}. {_contextual_action(step, parsed.directory)}")
        return 0
    if parsed.command == "setup":
        from .diagnostics import diagnose
        from .dependency_profiles import TASK_PROFILES
        from .onboarding import WorkspaceInitError, create_workspace
        from .provider_config import DEFAULT_MODELS, ProviderConfigError

        try:
            task_profile = TASK_PROFILES[parsed.task]
            selected_provider = parsed.provider
            if task_profile["provider_required"] and not selected_provider:
                if parsed.non_interactive or not sys.stdin.isatty():
                    raise ProviderConfigError(
                        "--provider is required for provider-backed non-interactive "
                        "setup; inspect safe local choices with `xscientist provider list`"
                    )
                selected_provider = _prompt_provider_choice()
            # Provider-neutral workspaces keep a dormant compatibility template,
            # but no provider is presented as configured or required to the user.
            workspace_template_provider = selected_provider or "zhipu"
            selected_model = parsed.model
            if not selected_model and not DEFAULT_MODELS.get(
                workspace_template_provider
            ):
                if parsed.non_interactive or not sys.stdin.isatty():
                    raise ProviderConfigError(
                        f"--model is required for provider {workspace_template_provider!r} "
                        "in non-interactive mode"
                    )
                selected_model = _prompt_provider_model(workspace_template_provider)
            onboarding = create_workspace(
                parsed.directory,
                profile=parsed.profile,
                provider=workspace_template_provider,
                model=selected_model,
                force=parsed.force,
                task=parsed.task,
                capabilities=task_profile["capabilities"],
                provider_required=bool(task_profile["provider_required"]),
            )
            workspace = Path(parsed.directory).expanduser().resolve()
            research_vcs: dict[str, object] = {
                "required": parsed.task != "service",
                "initialized": False,
                "checkpoint_id": None,
                "reason": "disabled for service-only workspace",
            }
            if parsed.task != "service" and not parsed.no_research_vcs:
                from .research_git import ResearchGitError, repository_status
                from .research_vcs import ResearchRepository

                try:
                    if (workspace / "research.yaml").is_file():
                        status = repository_status(workspace)
                        research_vcs = {
                            "required": True,
                            "initialized": True,
                            "checkpoint_id": (
                                (status.get("last_checkpoint") or {}).get(
                                    "checkpoint_id"
                                )
                            ),
                            "reason": "existing local research repository reused",
                        }
                    else:
                        question = (workspace / "topic.md").read_text(encoding="utf-8")
                        repository = ResearchRepository.init(
                            workspace,
                            name=workspace.name,
                            question=question,
                            policy="milestone",
                        )
                        status = repository.status()
                        research_vcs = {
                            "required": True,
                            "initialized": True,
                            "checkpoint_id": (
                                (status.get("last_checkpoint") or {}).get(
                                    "checkpoint_id"
                                )
                            ),
                            "reason": "local research repository initialized",
                        }
                except (OSError, ResearchGitError, ValueError) as exc:
                    raise WorkspaceInitError(
                        f"local Research VCS initialization failed: {exc}"
                    ) from exc
            elif parsed.no_research_vcs:
                research_vcs = {
                    "required": parsed.task != "service",
                    "initialized": False,
                    "checkpoint_id": None,
                    "reason": "disabled by --no-research-vcs",
                }
            provider_setup: dict[str, object] = {
                "ok": False,
                "skipped": True,
                "reason": "credentials explicitly skipped",
            }
            if not parsed.skip_credentials and task_profile["provider_required"]:
                try:
                    provider_setup = _configure_provider(
                        workspace,
                        name=workspace_template_provider,
                        model_value=selected_model,
                        non_interactive=parsed.non_interactive,
                    )
                except ProviderConfigError as exc:
                    provider_setup = {
                        "ok": False,
                        "skipped": True,
                        "reason": str(exc),
                    }
            elif not task_profile["provider_required"]:
                provider_setup = {
                    "ok": True,
                    "skipped": True,
                    "reason": "provider is not required for the selected task",
                }
            report = diagnose(
                workspace,
                task=parsed.task,
                provider=(
                    workspace_template_provider
                    if task_profile["provider_required"]
                    else None
                ),
                deep=parsed.deep,
            )
            _persist_readiness_report(workspace, report)
        except (OSError, WorkspaceInitError, ProviderConfigError, ValueError) as exc:
            if parsed.as_json:
                print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            else:
                print(f"xscientist setup: {exc}", file=sys.stderr)
            return 2
        payload = {
            "ok": report["ok"],
            "workspace_created": True,
            "workspace": onboarding["workspace"],
            "task": parsed.task,
            "provider_configuration": provider_setup,
            "research_vcs": research_vcs,
            "doctor": report,
            "next_actions": report["next_actions"],
            "host_paths_disclosed": False,
        }
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Created XScientist workspace: {payload['workspace']}")
            print(f"Task profile: {parsed.task}")
            print(
                "Configuration: " f"{_readiness_state(report['configuration_ready'])}"
            )
            print(f"Isolated runtime: {_readiness_state(report['runtime_ready'])}")
            print(
                "Research history: " f"{_readiness_state(research_vcs['initialized'])}"
            )
            if provider_setup.get("reason"):
                print(f"Provider setup: {provider_setup['reason']}")
            if payload["next_actions"]:
                print("Next actions:")
                for index, action in enumerate(payload["next_actions"], start=1):
                    print(
                        f"  {index}. " f"{_contextual_action(action, parsed.directory)}"
                    )
        return 0 if payload["ok"] else 1
    if parsed.command == "doctor":
        from .diagnostics import diagnose
        from .provider_config import ProviderConfigError

        try:
            payload = diagnose(
                parsed.workspace,
                task=parsed.task,
                provider=parsed.provider,
                deep=parsed.deep,
            )
            _persist_readiness_report(parsed.workspace, payload)
        except (OSError, ProviderConfigError, ValueError) as exc:
            print(f"xscientist doctor: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(
                f"XScientist configuration for {payload['task']}: "
                f"{_readiness_state(payload['configuration_ready'])}"
            )
            print("Deep runtime: " f"{_readiness_state(payload['runtime_ready'])}")
            for name, check in payload["checks"].items():
                label = _DOCTOR_CHECK_LABELS.get(name, name.replace("_", " ").title())
                rendered = _doctor_check_state(name, check)
                print(f"{label:<20} {rendered}")
            if payload["next_actions"]:
                print("Next actions:")
                for action in payload["next_actions"]:
                    print(f"  {_contextual_action(action, parsed.workspace)}")
        _record_local_metric("doctor", ok=bool(payload["ok"]))
        return 0 if payload["ok"] else 1
    if parsed.command == "capability":
        from .dependency_profiles import (
            CAPABILITY_MODULES,
            TASK_PROFILES,
            resolve_task_capabilities,
        )

        if parsed.capability_command == "list":
            payload = {
                "schema": "xscientist.capability-catalog.v1",
                "tasks": {
                    name: {
                        "description": profile["description"],
                        "capabilities": list(profile["capabilities"]),
                        "provider_required": profile["provider_required"],
                        "auth_required": profile["auth_required"],
                        "runtime_preflight": profile["runtime_preflight"],
                    }
                    for name, profile in TASK_PROFILES.items()
                },
                "capabilities": {
                    name: {"extra": name, "modules": list(modules)}
                    for name, modules in CAPABILITY_MODULES.items()
                },
            }
        else:
            payload = resolve_task_capabilities(
                parsed.task,
                provider=parsed.provider,
            )
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif parsed.capability_command == "list":
            for name, profile in payload["tasks"].items():
                capabilities = ", ".join(profile["capabilities"]) or "core"
                provider = " + provider" if profile["provider_required"] else ""
                print(f"{name:<12} {capabilities}{provider}")
        else:
            print(f"Task: {payload['task']}")
            print(f"Ready: {payload['ready']}")
            if payload["missing_modules"]:
                print("Missing modules: " + ", ".join(payload["missing_modules"]))
                print("Install: " + payload["install_command"])
            elif payload["provider_required"] and not payload["provider"]:
                print("Select a provider with --provider.")
        return 0 if parsed.capability_command == "list" or payload["ready"] else 1
    if parsed.command == "provider":
        return _run_provider(parsed)
    if parsed.command == "privacy":
        from ai_scientist.utils.privacy import (
            PrivacyFinding,
            format_privacy_findings,
            privacy_report,
        )

        report = privacy_report(
            parsed.path,
            include_untracked=parsed.include_untracked,
            history=parsed.history,
        )
        if parsed.as_json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        elif report["ok"]:
            print("Privacy audit: clean (matched values were never displayed)")
        else:
            print(
                format_privacy_findings(
                    PrivacyFinding(**finding) for finding in report["findings"]
                ),
                file=sys.stderr,
            )
            print(
                f"Privacy audit: {report['finding_count']} finding(s); "
                "matched values were not displayed",
                file=sys.stderr,
            )
        return 0 if report["ok"] else 1
    if parsed.command == "evolution-gate":
        return _run_evolution_gate(parsed)
    parser.error(f"Unsupported command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
