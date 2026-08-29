from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
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
    "custom",
]

_RESEARCH_MODEL_FLAGS = (
    ("ideation", "--model-ideation"),
    ("agg_plots", "--model-agg-plots"),
    ("writeup", "--model-writeup"),
    ("writeup_small", "--model-writeup-small"),
    ("citation", "--model-citation"),
    ("review", "--model-review"),
    ("idea_ranking", "--idea-rank-model"),
    ("quality", "--quality-model"),
)

_TASK_CHOICES = [
    "protocol",
    "manage",
    "research",
    "paper",
    "pdf-review",
    "ml-study",
    "service",
]

_SPECIAL_COMMAND_OPTIONS = {"help": ("--all",)}


class _ArgumentParseFailure(ValueError):
    """A parser error that the CLI can render without leaking raw argv."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentParseFailure(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="xscientist",
        description="XScientist SDK, workflow CLI, and API service.",
        epilog=(
            "Start with your own idea: `xscientist explore`, then use "
            "`xscientist start --help` only when you want model-backed research."
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
    explore_parser = subparsers.add_parser(
        "explore",
        help="Turn your own idea into a testable offline research start.",
    )
    explore_parser.add_argument("directory", nargs="?", default="./my-study")
    explore_parser.add_argument(
        "--idea", help="the idea or question you want to investigate"
    )
    explore_parser.add_argument(
        "--expect",
        "--hypothesis",
        dest="expectation",
        help="the observable result you expect if the idea is right",
    )
    explore_parser.add_argument(
        "--disprove",
        "--falsifier",
        dest="disconfirming_result",
        help="the result that would make you doubt or reject that expectation",
    )
    explore_parser.add_argument(
        "--test", dest="first_test", help="the first fair comparison or test to run"
    )
    explore_parser.add_argument(
        "--success-rule",
        help="the rule for deciding what the first test means",
    )
    explore_parser.add_argument("--name")
    explore_parser.add_argument("--actor", default="human:researcher")
    explore_parser.add_argument("--lang", choices=["auto", "en", "zh"], default="auto")
    explore_parser.add_argument("--git-user-name")
    explore_parser.add_argument("--git-user-email")
    explore_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; saving only --idea is valid and honestly incomplete",
    )
    explore_parser.add_argument("--json", action="store_true", dest="as_json")
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
    audit_parser = subparsers.add_parser(
        "audit",
        help="Check traceability, replayability, or independent verification.",
    )
    audit_parser.add_argument("workspace", nargs="?", default=".")
    audit_parser.add_argument("--ref", default="HEAD")
    audit_parser.add_argument(
        "--level", choices=["trace", "replay", "verify"], default="trace"
    )
    audit_parser.add_argument("--no-objects", action="store_true")
    audit_parser.add_argument("--json", action="store_true", dest="as_json")
    history_parser = subparsers.add_parser(
        "history",
        help="Review, save, or safely roll back local scientific history.",
    )
    history_subparsers = history_parser.add_subparsers(
        dest="history_command", required=True
    )
    history_list = history_subparsers.add_parser(
        "list", help="Show checkpoints and unsaved research changes."
    )
    history_list.add_argument("workspace", nargs="?", default=".")
    history_list.add_argument("--limit", type=int, default=20)
    history_list.add_argument("--json", action="store_true", dest="as_json")
    history_show = history_subparsers.add_parser(
        "show", help="Inspect one hash-checked scientific checkpoint."
    )
    history_show.add_argument("workspace", nargs="?", default=".")
    history_show.add_argument("--commit", default="HEAD")
    history_show.add_argument("--json", action="store_true", dest="as_json")
    history_diff = history_subparsers.add_parser(
        "diff", help="Review file and semantic changes between two checkpoints."
    )
    history_diff.add_argument("workspace", nargs="?", default=".")
    history_diff.add_argument("--from", dest="before", default="HEAD^")
    history_diff.add_argument("--to", dest="after", default="HEAD")
    history_diff.add_argument("--deep", action="store_true")
    history_diff.add_argument("--json", action="store_true", dest="as_json")
    history_save = history_subparsers.add_parser(
        "save", help="Save selected or eligible research changes as one checkpoint."
    )
    history_save.add_argument("workspace", nargs="?", default=".")
    history_save.add_argument("-m", "--message", default="save current research state")
    history_save.add_argument("--summary", default="")
    history_save.add_argument("--actor")
    history_save.add_argument("--json", action="store_true", dest="as_json")
    history_rollback = history_subparsers.add_parser(
        "rollback",
        help="Preview or append a reversal checkpoint without rewriting history.",
    )
    history_rollback.add_argument("workspace", nargs="?", default=".")
    history_rollback.add_argument("--commit", default="HEAD")
    history_rollback.add_argument("-m", "--message")
    history_rollback.add_argument(
        "--apply",
        action="store_true",
        help="apply the previewed reversal as a new checkpoint",
    )
    history_rollback.add_argument("--json", action="store_true", dest="as_json")
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
    first_run.add_argument(
        "--output",
        default=None,
        help="optional JSON file for the redacted report (atomic write)",
    )
    first_run.add_argument("--json", action="store_true", dest="as_json")
    autoresearch = benchmark_subparsers.add_parser(
        "autoresearch",
        help=(
            "Run an offline AutoResearchEval-inspired artifact conformance pilot; "
            "this does not produce an official score."
        ),
    )
    autoresearch.add_argument(
        "--tasks",
        required=True,
        help="local JSONL/JSON task file; no network download is attempted",
    )
    autoresearch.add_argument(
        "--workspace",
        default=None,
        help="optional existing XScientist workspace to inspect read-only",
    )
    autoresearch.add_argument("--limit", type=int, default=20)
    autoresearch.add_argument(
        "--kind",
        choices=["all", "open-ended", "optimization"],
        default="all",
        help="task subset to inspect (default: all)",
    )
    autoresearch.add_argument(
        "--show-process",
        action="store_true",
        help=(
            "show the bounded git-like commit, branch, artifact, and fairness "
            "timeline in human output; JSON always includes it"
        ),
    )
    autoresearch.add_argument(
        "--output",
        default=None,
        help="optional JSON file for the redacted report (atomic write)",
    )
    autoresearch.add_argument("--json", action="store_true", dest="as_json")
    verify_report = benchmark_subparsers.add_parser(
        "verify",
        help=(
            "Validate a saved benchmark report and its fail-closed comparison "
            "boundary without network or providers."
        ),
    )
    verify_report.add_argument(
        "--report", required=True, help="path to a saved benchmark JSON report"
    )
    verify_report.add_argument("--json", action="store_true", dest="as_json")
    systems = benchmark_subparsers.add_parser(
        "systems",
        help=(
            "Show a source-audited capability matrix for XScientist and related "
            "research systems; this never produces a cross-system score."
        ),
    )
    systems.add_argument(
        "--workspace",
        default=None,
        help="optional workspace whose redacted Git-like process summary is included",
    )
    systems.add_argument(
        "--show-process",
        action="store_true",
        help="print branch/intermediate process counts for the supplied workspace",
    )
    systems.add_argument(
        "--output",
        default=None,
        help="optional JSON file for the redacted report (atomic write)",
    )
    systems.add_argument("--json", action="store_true", dest="as_json")
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
            "cost enforcement; no request is made unless --live is supplied."
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
    provider_check.add_argument(
        "--live",
        action="store_true",
        help=(
            "make one explicit minimal provider request; this may incur cost "
            "and requires a live-capable provider"
        ),
    )
    provider_check.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="timeout in seconds for --live (default: 30)",
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
    provider_add.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API root for the custom provider",
    )
    provider_add.add_argument("--no-activate", action="store_true")
    provider_add.add_argument("--no-update-bfts", action="store_true")
    provider_add.add_argument("--json", action="store_true", dest="as_json")
    provider_add.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; require credentials to already exist in env or the env file",
    )
    provider_test = provider_subparsers.add_parser(
        "test",
        help=(
            "Make one explicit minimal API call and verify the provider-reported model."
        ),
    )
    provider_test.add_argument("name", nargs="?", choices=_PROVIDER_CHOICES)
    provider_test.add_argument(
        "--workspace",
        default=None,
        help="workspace root (default: discover from the current directory)",
    )
    provider_test.add_argument("--timeout", type=float, default=30.0)
    provider_test.add_argument("--json", action="store_true", dest="as_json")
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
    parser = _SafeArgumentParser(
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
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API root when --provider custom is selected",
    )
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
    parser = _SafeArgumentParser(
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
    study.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API root when --provider custom is selected",
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
    print("  explore    Make your own idea testable; no API key needed")
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
    print("Trust and recovery:")
    print("  audit      Check traceability, replayability, and independent review")
    print("  history    Review, save, or safely roll back scientific checkpoints")
    print("  research   Use advanced planning, review, branching, and reproduction")
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


def _interactive_explore_inputs(
    parsed: argparse.Namespace,
    *,
    existing: dict[str, object] | None,
) -> None:
    """Ask only plain-language questions that are still scientifically missing."""

    interactive = (
        not parsed.non_interactive and not parsed.as_json and sys.stdin.isatty()
    )
    language = _selected_language(parsed.lang)
    if existing is not None and not parsed.idea:
        parsed.idea = str(existing["idea"])
    if not parsed.idea and interactive:
        parsed.idea = input(
            "你想研究什么想法或问题？ "
            if language == "zh"
            else "What idea or question do you want to investigate? "
        ).strip()
    if not parsed.idea:
        raise ValueError(
            "--idea is required when input is not interactive"
            if language == "en"
            else "非交互模式必须提供 --idea"
        )

    has_hypothesis = bool(existing and existing.get("hypothesis_id"))
    has_plan = bool(existing and existing.get("plan_id"))
    if not has_hypothesis and not parsed.expectation and interactive:
        parsed.expectation = input(
            ("如果这个想法成立，你预计能观察到什么变化？" "（暂时不知道可直接回车） ")
            if language == "zh"
            else (
                "If the idea is right, what observable change do you expect? "
                "(Press Enter to decide later) "
            )
        ).strip()
    if not has_hypothesis and parsed.expectation and not parsed.disconfirming_result:
        if interactive:
            parsed.disconfirming_result = input(
                ("什么结果会让你怀疑或放弃这个预期？" "（暂时不知道可直接回车） ")
                if language == "zh"
                else (
                    "What result would make you doubt or reject that expectation? "
                    "(Press Enter to decide later) "
                )
            ).strip()
            if not parsed.disconfirming_result:
                # A one-sided prediction is not persisted as if it were already
                # falsifiable. The original idea is still saved below.
                parsed.expectation = None
        else:
            raise ValueError("--expect and --disprove must be provided together")

    will_have_hypothesis = has_hypothesis or bool(
        parsed.expectation and parsed.disconfirming_result
    )
    if will_have_hypothesis and not has_plan and not parsed.first_test and interactive:
        parsed.first_test = input(
            (
                "你能先做哪一个公平比较或检验？（暂时不知道可直接回车） "
                if language == "zh"
                else (
                    "What fair comparison or test could you run first? "
                    "(Press Enter to decide later) "
                )
            )
        ).strip()
    if parsed.first_test and not parsed.success_rule and interactive:
        parsed.success_rule = input(
            (
                "看到什么结果时，你会认为这次检验支持预期？（可选） "
                if language == "zh"
                else ("What result would count as support in this test? (Optional) ")
            )
        ).strip()


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
    # ``status`` is commonly invoked from a parent directory (or a CI job)
    # while its guide emits the intentionally short ``explore .`` command.
    # Keep the next action bound to the workspace the user just inspected;
    # otherwise a copy/paste would silently create/update a different repo in
    # the caller's current directory.
    if command in {"xscientist explore .", "xscientist explore . --lang zh"}:
        suffix = " --lang zh" if command.endswith("--lang zh") else ""
        return f"xscientist explore {quoted}{suffix}"
    if "--workspace ." in command:
        return command.replace("--workspace .", f"--workspace {quoted}")
    if "--repo ." in command:
        contextual = command.replace("--repo .", f"--repo {quoted}")
        if "--output research-dag" in contextual:
            output = shlex.quote(str(Path(workspace).expanduser() / "research-dag"))
            contextual = contextual.replace(
                "--output research-dag", f"--output {output}"
            )
        return contextual
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

    if (
        parsed.non_interactive
        or bool(getattr(parsed, "as_json", False))
        or not sys.stdin.isatty()
    ):
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


def _research_model_contract(model: str, *, source: str) -> dict[str, object]:
    """Describe the exact model contract shared by an autonomous study."""

    from ai_scientist.utils.provider_registry import resolve_model_provider

    spec = resolve_model_provider(model)
    roles = {role: model for role, _flag in _RESEARCH_MODEL_FLAGS}
    return {
        "selected_model": model,
        "provider": spec.provider,
        "provider_display_name": spec.display_name,
        "client_family": spec.client_family,
        "client_model": spec.client_model,
        "request_style": spec.request_style,
        "selection_source": source,
        "roles": roles,
        "all_roles_explicit": True,
    }


def _research_model_arguments(model: str) -> list[str]:
    """Pass the selected model explicitly through every project role."""

    arguments: list[str] = []
    for _role, flag in _RESEARCH_MODEL_FLAGS:
        arguments.extend([flag, model])
    return arguments


def _build_doctor_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
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
    parser = _SafeArgumentParser(
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
    from .research_adapters import (
        ADAPTER_API_VERSION,
        ADAPTER_ENTRYPOINT_GROUP,
        available_research_adapters,
    )

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
    adapters = available_research_adapters()
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
        "extensions": {
            "research_adapters": {
                "api_version": ADAPTER_API_VERSION,
                "entry_point_group": ADAPTER_ENTRYPOINT_GROUP,
                "available": [item["name"] for item in adapters],
                "discovery_command": "xscientist research adapter list --json",
            }
        },
        "quickstart": "xscientist explore ./my-study",
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
    base_url_value: str | None = None,
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
        mark_managed_environment,
        normalize_provider_name,
        provider_statuses,
        read_env_file,
        resolve_env_file,
        save_provider,
        update_bfts_models,
        update_env_file,
        validate_custom_base_url,
        validate_provider_model,
        workspace_environment,
    )

    requested_name = str(name or "").strip().lower()
    name = normalize_provider_name(requested_name)
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
    effective_environment = workspace_environment(workspace)
    updates: dict[str, str] = {}
    if base_url_value is not None:
        if provider != "openai_compat":
            raise ProviderConfigError(
                "--base-url is supported only with provider custom/openai_compat"
            )
        updates["OPENAI_COMPAT_BASE_URL"] = validate_custom_base_url(base_url_value)
    for field in PROVIDER_FIELDS[provider]:
        if field.name in updates:
            continue
        current = configured_field_value(field, stored, effective_environment)
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
    # Explicitly entered values must also win for the rest of this process;
    # otherwise a stale desktop/shell environment could still route this run
    # to another endpoint or credential.
    for env_field, value in updates.items():
        mark_managed_environment(workspace, env_field, value)
    saved = save_provider(
        workspace,
        provider=provider,
        model=model,
        env_file=env_name,
        activate=activate,
    )
    from ai_scientist.utils.provider_registry import model_provenance

    provenance = model_provenance(model, env=workspace_environment(workspace))
    bfts_updated = False
    if update_bfts and saved.get("active_provider") == provider:
        bfts_updated = update_bfts_models(workspace / "bfts_config.yaml", model)
    payload: dict[str, object] = {
        "ok": True,
        "workspace": ".",
        "provider": provider,
        "requested_provider": requested_name,
        "model": model,
        "resolved_client_model": provenance["client_model"],
        "endpoint_fingerprint": provenance["endpoint_fingerprint"],
        "configuration_fingerprint": provenance["configuration_fingerprint"],
        "active": saved.get("active_provider") == provider,
        "env_file": env_path.relative_to(workspace).as_posix(),
        "credentials_written": sorted(
            field.name
            for field in PROVIDER_FIELDS[provider]
            if field.secret and field.name in updates
        ),
        "settings_written": sorted(
            field.name
            for field in PROVIDER_FIELDS[provider]
            if not field.secret and field.name in updates
        ),
        "endpoint_configured": (
            bool(
                configured_field_value(
                    next(
                        (
                            field
                            for field in PROVIDER_FIELDS[provider]
                            if field.name.endswith("BASE_URL")
                        ),
                        PROVIDER_FIELDS[provider][0],
                    ),
                    {**stored, **updates},
                    workspace_environment(workspace),
                )
            )
            if provider in {"openai_compat", "ollama"}
            else None
        ),
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
        begin_managed_environment_transaction,
        begin_provider_file_transaction,
        discover_workspace_root,
        load_provider_config,
        normalize_provider_name,
        provider_statuses,
        remove_provider,
        resolve_provider_workspace_root,
        update_bfts_models,
        workspace_config_path,
        workspace_environment,
    )

    try:
        discovery_only = False
        if parsed.workspace is None:
            workspace = discover_workspace_root()
            if workspace is None:
                if parsed.provider_command == "list":
                    workspace = resolve_provider_workspace_root(Path.cwd())
                    discovery_only = True
                else:
                    raise ProviderConfigError(
                        "provider configuration not found in the current directory or its parents; "
                        "run `xscientist setup WORKSPACE` first or pass --workspace"
                    )
        else:
            workspace = resolve_provider_workspace_root(parsed.workspace)
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
                print(
                    json.dumps(
                        _safe_public_json_payload(payload),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
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

        if parsed.provider_command == "test":
            from ai_scientist.utils.provider_registry import (
                probe_live_model,
                probe_openai_compatible_tool_call,
            )

            if parsed.timeout <= 0:
                raise ProviderConfigError("--timeout must be greater than zero")
            config = load_provider_config(workspace, missing_ok=False)
            requested_provider = str(parsed.name or config.get("active_provider") or "")
            provider = normalize_provider_name(requested_provider)
            if not provider:
                raise ProviderConfigError(
                    "no active provider is configured; pass a provider name"
                )
            entry = (config.get("providers") or {}).get(provider, {})
            if not isinstance(entry, dict) or not entry.get("model"):
                raise ProviderConfigError(f"provider {provider!r} is not configured")
            model = str(entry["model"])
            provider_environment = workspace_environment(workspace)
            try:
                if provider == "openai_compat":
                    probe = probe_openai_compatible_tool_call(
                        model,
                        timeout=float(parsed.timeout),
                        env=provider_environment,
                    )
                else:
                    probe = probe_live_model(
                        model,
                        timeout=float(parsed.timeout),
                        env=provider_environment,
                    )
            except ValueError as exc:
                raise ProviderConfigError(str(exc)) from exc
            payload = {
                "schema": "xscientist.provider-test.v1",
                "ok": bool(probe.get("ok")),
                "workspace": ".",
                "provider": provider,
                "model": model,
                "supported": probe.get("supported", True),
                "identity_status": probe.get("identity_status", "unavailable"),
                "billing": "one explicit minimal model request",
                "capability": probe.get("capability", "text_completion"),
                "probe": probe,
                "response_content_recorded": False,
            }
            if parsed.as_json:
                print(
                    json.dumps(
                        _safe_public_json_payload(payload),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(f"Provider test: {provider}")
                print(f"Requested model: {probe.get('client_model') or model}")
                print(f"Reported model: {probe.get('reported_model') or '-'}")
                if provider == "openai_compat":
                    state = "verified" if probe.get("tool_call_valid") else "failed"
                    print(f"Forced function call: {state}")
                if probe.get("error_code") == "live_probe_not_supported":
                    print(
                        "Live probe: not supported for this provider family; "
                        "no API request was made"
                    )
                elif probe.get("identity_status") == "exact":
                    print("Model identity: exact match")
                elif probe.get("identity_status") == "alias":
                    print(
                        "Model identity: route alias (request succeeded, "
                        "but the gateway returned a prefixed alias)"
                    )
                elif probe.get("error_code") == "live_request_failed":
                    print(
                        "Model identity: unavailable "
                        f"({probe.get('error_type') or 'request failed'})"
                    )
                else:
                    print("Model identity: mismatch; treat this run as unverified")
            return 0 if payload["ok"] else 1

        if parsed.provider_command == "check":
            if parsed.max_cost_usd is not None and parsed.max_cost_usd <= 0:
                raise ProviderConfigError("--max-cost-usd must be greater than zero")
            config = load_provider_config(workspace, missing_ok=False)
            provider = normalize_provider_name(
                str(parsed.name or config.get("active_provider") or "")
            )
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
            from ai_scientist.utils.provider_registry import (
                model_provenance,
                probe_live_model,
            )

            provenance = (
                model_provenance(model, env=workspace_environment(workspace))
                if model
                else None
            )
            price = (
                resolve_model_price(model, prices_per_million=custom_prices)
                if model
                else None
            )
            cost_ready = parsed.max_cost_usd is None or price is not None
            live_probe: dict[str, object] | None = None
            if parsed.live:
                if parsed.timeout <= 0:
                    raise ProviderConfigError("--timeout must be greater than zero")
                # A live request must never bypass an explicit cost guard.  If
                # the configured model has no price, fail closed before opening
                # a network connection and tell the user how to supply one.
                if not cost_ready:
                    live_probe = {
                        "ok": False,
                        "supported": None,
                        "transport_ok": None,
                        "error_code": "live_probe_blocked_by_unknown_cost",
                        "response_content_recorded": False,
                    }
                elif not row["ready"]:
                    live_probe = {
                        "ok": False,
                        "supported": None,
                        "transport_ok": None,
                        "error_code": "live_probe_blocked_by_configuration",
                        "response_content_recorded": False,
                    }
                else:
                    try:
                        live_probe = probe_live_model(
                            model,
                            timeout=float(parsed.timeout),
                            env=workspace_environment(workspace),
                        )
                    except ValueError as exc:
                        live_probe = {
                            "ok": False,
                            "supported": None,
                            "transport_ok": False,
                            "error_code": "live_probe_configuration_error",
                            "error_message": str(exc),
                            "response_content_recorded": False,
                        }
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
            if parsed.live and live_probe is not None and not live_probe.get("ok"):
                probe_error = str(live_probe.get("error_code") or "live_probe_failed")
                error_codes.append(probe_error)
                remediations.append(
                    {
                        "code": "inspect_live_provider_probe",
                        # Keep the repair command path-free.  Provider commands
                        # already discover the nearest workspace, so a copied
                        # command remains useful without disclosing the local
                        # absolute path in JSON output.
                        "command": f"xscientist provider test {provider} --json",
                    }
                )
            ready_with_live = bool(
                row["ready"]
                and cost_ready
                and (not parsed.live or (live_probe and live_probe.get("ok")))
            )
            live_request_attempted = bool(
                parsed.live
                and live_probe is not None
                and live_probe.get("transport_ok") is not None
            )
            payload = {
                "schema": "xscientist.provider-check.v1",
                "ok": ready_with_live,
                "workspace": ".",
                "provider": provider,
                "model": model or None,
                "provenance": provenance,
                "live_probe": live_probe,
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
                        (local_probe["checked"] and local_probe["ok"])
                        or (live_probe is not None and live_probe.get("ok"))
                    ),
                    "live_probe_requested": bool(parsed.live),
                    "live_request_attempted": live_request_attempted,
                    "live_probe_supported": (
                        live_probe.get("supported") if live_probe is not None else None
                    ),
                    "verification_scope": (
                        "live_request"
                        if live_request_attempted
                        else (
                            "live_request_blocked"
                            if parsed.live
                            else (
                                "local_service"
                                if local_probe["checked"]
                                else "configuration_only"
                            )
                        )
                    ),
                },
                "price_per_million": price,
                "error_codes": error_codes,
                "remediations": remediations,
            }
            if parsed.as_json:
                print(
                    json.dumps(
                        _safe_public_json_payload(payload),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                state = (
                    "live provider request verified"
                    if parsed.live and payload["ok"]
                    else (
                        (
                            "live provider request failed"
                            if live_request_attempted
                            else "live provider request not made (blocked)"
                        )
                        if parsed.live
                        else (
                            "local service and model verified"
                            if local_probe["checked"] and payload["ok"]
                            else (
                                "configuration checks passed; credentials not live-verified"
                                if payload["ok"]
                                and any(
                                    field.required
                                    for field in PROVIDER_FIELDS[provider]
                                )
                                else (
                                    "configuration checks passed"
                                    if payload["ok"]
                                    else "not ready"
                                )
                            )
                        )
                    )
                )
                print(f"Provider {provider}: {state}")
                print(f"Model: {model or '-'}")
                if any(field.required for field in PROVIDER_FIELDS[provider]):
                    if not parsed.live:
                        credential_note = (
                            "presence check only; no provider request was made"
                        )
                    elif live_probe and live_probe.get("transport_ok") is not None:
                        credential_note = "presence check plus explicit live probe; see request outcome"
                    else:
                        credential_note = (
                            "presence check; live probe was blocked before a request"
                        )
                    print(
                        "Credentials: "
                        + ("present" if row["credentials_available"] else "missing")
                        + f" ({credential_note})"
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
            config = remove_provider(workspace, normalize_provider_name(parsed.name))
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
                print(json.dumps(_safe_public_json_payload(payload), indent=2))
            else:
                print(f"Removed provider metadata: {parsed.name}")
                print("Stored credentials were left untouched.")
            return 0

        if parsed.provider_command == "activate":
            provider_name = normalize_provider_name(parsed.name)
            config = activate_provider(workspace, provider_name)
            entry = config["providers"][provider_name]
            model = str(entry["model"])
            bfts_updated = False
            if not parsed.no_update_bfts:
                bfts_updated = update_bfts_models(workspace / "bfts_config.yaml", model)
            payload = {
                "ok": True,
                "active_provider": provider_name,
                "model": model,
                "bfts_updated": bfts_updated,
            }
            if parsed.as_json:
                print(json.dumps(_safe_public_json_payload(payload), indent=2))
            else:
                print(f"Active provider: {parsed.name}")
                print(f"Default model: {model}")
                print(f"BFTS config updated: {bfts_updated}")
            return 0

        environment_transaction = begin_managed_environment_transaction()
        try:
            file_transaction = begin_provider_file_transaction()
        except BaseException:
            environment_transaction.rollback()
            raise
        try:
            payload = _configure_provider(
                workspace,
                name=parsed.name,
                model_value=parsed.model,
                base_url_value=parsed.base_url,
                activate=not parsed.no_activate,
                update_bfts=not parsed.no_update_bfts,
                non_interactive=parsed.non_interactive,
            )
        except BaseException as exc:
            try:
                file_conflicts = file_transaction.rollback()
            finally:
                environment_conflicts = environment_transaction.rollback()
            if file_conflicts or environment_conflicts:
                raise ProviderConfigError(
                    "provider configuration failed; rollback preserved "
                    "concurrently changed state"
                ) from exc
            raise
        else:
            file_transaction.commit()
            environment_transaction.commit()
        if parsed.as_json:
            print(
                json.dumps(
                    _safe_public_json_payload(payload),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(f"Configured provider: {payload['provider']}")
            print(f"Default model: {payload['model']}")
            if payload["credentials_written"]:
                print("Credentials saved securely to: " f"{payload['env_file']}")
            else:
                print("Credentials: using existing environment or local env file")
            print(f"Active: {payload['active']}")
            print(f"BFTS config updated: {payload['bfts_updated']}")
            if payload.get("settings_written"):
                print("Provider settings saved securely to: " f"{payload['env_file']}")
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


def _research_question_body(value: object) -> str:
    """Normalize the first Markdown section that carries a research question."""

    lines = str(value or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines:
        heading = re.sub(r"^#{1,6}\s*", "", lines[0].strip()).casefold()
        if heading in {"question", "research question", "topic", "research topic"}:
            lines.pop(0)
    body: list[str] = []
    for line in lines:
        stripped = line.strip()
        if body and re.match(r"^#{1,6}\s+", stripped):
            break
        if stripped:
            body.append(stripped)
    return re.sub(r"\s+", " ", " ".join(body)).strip()


def _validated_existing_research_question(
    workspace: Path,
    requested_question: object,
) -> tuple[str, bool]:
    """Read one immutable question identity without rewriting scientific files."""

    from .onboarding import _render_topic
    from .research_vcs import ResearchRepository

    repository = ResearchRepository(workspace)
    question_objects = repository.objects(kind="question")
    latest_question = (
        max(question_objects, key=lambda item: str(item.get("created_at") or ""))
        if question_objects
        else None
    )
    object_question = ""
    if latest_question is not None:
        payload = latest_question.get("payload") or {}
        if not isinstance(payload, dict):
            raise ValueError("the recorded Research VCS question payload is invalid")
        object_question = _research_question_body(payload.get("question"))

    source_paths = {
        "question.md": workspace / "question.md",
        "topic.md": workspace / "topic.md",
        "00_config/topic.md": workspace / "00_config" / "topic.md",
    }
    source_questions: dict[str, str] = {}
    source_texts: dict[str, str] = {}
    for label, path in source_paths.items():
        if path.is_symlink():
            raise ValueError(
                f"existing Research VCS question source {label} must not be a symlink"
            )
        if not path.exists():
            if label == "question.md":
                raise ValueError("existing Research VCS is missing question.md")
            continue
        if not path.is_file():
            raise ValueError(
                f"existing Research VCS question source {label} is not a file"
            )
        source_texts[label] = path.read_text(encoding="utf-8")
        source_questions[label] = _research_question_body(source_texts[label])

    status = repository.status()
    scientific_dirt = {
        "question.md",
        "topic.md",
        "00_config/topic.md",
    } & (
        set(status.get("eligible_changes") or [])
        | set(status.get("staged_paths") or [])
        | set((status.get("research_stage") or {}).get("paths") or [])
    )
    packaged_placeholder = _render_topic()
    pristine_packaged_placeholder = bool(
        not object_question
        and not scientific_dirt
        and source_texts.get("question.md") == packaged_placeholder
        and source_texts.get("topic.md") == packaged_placeholder
        and "00_config/topic.md" not in source_texts
    )
    if pristine_packaged_placeholder:
        requested = _research_question_body(requested_question)
        if requested.casefold() == source_questions["question.md"].casefold():
            requested = ""
        return requested, True

    canonical_question = object_question or source_questions.get("question.md", "")
    canonical_key = canonical_question.casefold()
    if not canonical_key:
        raise ValueError("existing Research VCS has no usable research question")

    conflicts = sorted(
        label
        for label, question in source_questions.items()
        if question.casefold() != canonical_key
    )
    if conflicts:
        raise ValueError(
            "existing Research VCS question sources disagree ("
            + ", ".join(conflicts)
            + "); refusing to rewrite question.md or topic.md"
        )

    requested = _research_question_body(requested_question)
    if requested and requested.casefold() != canonical_key:
        raise ValueError(
            "--question conflicts with the existing Research VCS question; "
            "reuse the original question or choose a new directory"
        )
    return canonical_question, False


def _safe_json_error_payload(error: object, **fields: object) -> dict[str, object]:
    """Build one portable machine error without host paths or secret values."""

    from ai_scientist.utils.privacy import redact_sensitive_payload

    payload: dict[str, object] = {
        **fields,
        "ok": False,
        "workspace": ".",
        "error": str(error),
    }
    redacted = redact_sensitive_payload(payload)
    return dict(redacted) if isinstance(redacted, dict) else payload


def _safe_public_json_payload(payload: object) -> object:
    """Apply the public JSON privacy boundary to success and failure payloads."""

    from ai_scientist.utils.privacy import redact_sensitive_payload

    return redact_sensitive_payload(payload)


def _validated_workspace_file(workspace: Path, relative: str) -> Path | None:
    """Resolve a managed file without following parent or leaf symlinks."""

    root = workspace.resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("managed workspace path must stay below the workspace")
    cursor = root
    for part in relative_path.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(
                f"managed workspace parent for {relative} must not be a symlink"
            )
        if cursor.exists() and not cursor.is_dir():
            raise ValueError(
                f"managed workspace parent for {relative} is not a directory"
            )
    target = root / relative_path
    if target.is_symlink():
        raise ValueError(f"managed workspace file {relative} must not be a symlink")
    if target.exists() and not target.is_file():
        raise ValueError(f"managed workspace file {relative} is not a regular file")
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"managed workspace file {relative} escapes the workspace"
        ) from exc
    return target if target.is_file() else None


def _workspace_git_changes(
    workspace: Path,
    *,
    required: bool = False,
) -> tuple[bool, set[str], set[str]]:
    """Return exact-root Git state without mutating or discovering parent repos."""

    if not workspace.is_dir():
        return False, set(), set()
    if shutil.which("git") is None:
        if required:
            raise OSError("Git is required for local Research VCS setup")
        return False, set(), set()
    probe = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode or not probe.stdout.strip():
        return False, set(), set()
    try:
        if Path(probe.stdout.strip()).resolve() != workspace.resolve():
            return False, set(), set()
    except OSError:
        return False, set(), set()

    def paths(*arguments: str) -> set[str]:
        completed = subprocess.run(
            ["git", "-C", str(workspace), *arguments],
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            raise OSError("could not inspect the existing Git worktree")
        return {
            item.decode("utf-8", errors="surrogateescape")
            for item in completed.stdout.split(b"\0")
            if item
        }

    staged = paths(
        "diff",
        "--cached",
        "--ita-visible-in-index",
        "--name-only",
        "--no-renames",
        "-z",
    )
    tracked_changes = paths("diff", "--name-only", "--no-renames", "-z")
    return True, staged, tracked_changes


def _workspace_git_head_contains(workspace: Path, relative: str) -> bool:
    """Return whether an exact-root repository HEAD owns a managed path."""

    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        return False
    had_git, _staged, _tracked = _workspace_git_changes(workspace)
    if not had_git:
        return False
    completed = subprocess.run(
        ["git", "-C", str(workspace), "cat-file", "-e", f"HEAD:{relative}"],
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


_SETUP_TRANSACTION_FILES = (
    ".dockerignore",
    ".env.example",
    ".gitignore",
    ".xscientist/providers.json",
    ".xscientist/README.md",
    ".xscientist/readiness.json",
    "Dockerfile.executor",
    "README.md",
    "bfts_config.yaml",
    "question.md",
    "research.yaml",
    "topic.md",
)
_SETUP_TRANSACTION_DIRS = (
    ".git",
    ".xscientist",
    ".ara-store",
    "ara",
    "checkpoints",
    "claims",
    "hypotheses",
    "manuscript",
    "research-objects",
)


def _setup_workspace_snapshot(
    workspace: Path,
    *,
    extra_files: Sequence[str] = (),
) -> dict[str, object]:
    """Capture only files and repository state setup is authorized to mutate."""

    from . import provider_config as provider_config_module

    workspace_existed = workspace.exists()
    files: dict[str, dict[str, object] | None] = {}
    for relative in _SETUP_TRANSACTION_FILES:
        target = _validated_workspace_file(workspace, relative)
        files[relative] = (
            {
                "content": target.read_bytes(),
                "mode": target.stat().st_mode & 0o7777,
            }
            if target is not None
            else None
        )
    directories: dict[str, dict[str, object]] = {}
    for relative in _SETUP_TRANSACTION_DIRS:
        target = workspace / relative
        if target.is_symlink():
            directories[relative] = {"kind": "symlink"}
        elif target.is_dir():
            directories[relative] = {"kind": "directory"}
        elif target.is_file():
            directories[relative] = {
                "kind": "file",
                "content": target.read_bytes(),
                "mode": target.stat().st_mode & 0o7777,
            }
        else:
            directories[relative] = {"kind": "absent"}
    checkpoint_entries = (
        {path.name for path in (workspace / "checkpoints").iterdir()}
        if directories["checkpoints"]["kind"] == "directory"
        else set()
    )
    extra_file_state: dict[str, dict[str, object] | None] = {}
    extra_parent_state: dict[str, bool] = {}
    for relative in extra_files:
        target = _validated_workspace_file(workspace, relative)
        extra_file_state[relative] = (
            {
                "content": target.read_bytes(),
                "mode": target.stat().st_mode & 0o7777,
            }
            if target is not None
            else None
        )
        parent = Path(relative).parent
        while parent != Path("."):
            extra_parent_state[parent.as_posix()] = (workspace / parent).is_dir()
            parent = parent.parent
    had_git, _staged, _tracked = _workspace_git_changes(workspace)
    head = None
    symbolic_ref = None
    git_identity: dict[str, dict[str, object]] = {}
    if had_git:
        completed = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        head = completed.stdout.strip() if completed.returncode == 0 else None
        completed = subprocess.run(
            ["git", "-C", str(workspace), "symbolic-ref", "-q", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        symbolic_ref = completed.stdout.strip() if completed.returncode == 0 else None
        for key in ("user.name", "user.email"):
            completed = subprocess.run(
                ["git", "-C", str(workspace), "config", "--local", "--get", key],
                check=False,
                capture_output=True,
                text=True,
            )
            git_identity[key] = {
                "present": completed.returncode == 0,
                "value": (
                    completed.stdout.rstrip("\n") if completed.returncode == 0 else ""
                ),
            }
    snapshot = {
        "workspace_existed": workspace_existed,
        "files": files,
        "directories": directories,
        "checkpoint_entries": checkpoint_entries,
        "extra_files": extra_file_state,
        "extra_parents": extra_parent_state,
        "had_git": had_git,
        "head": head,
        "symbolic_ref": symbolic_ref,
        "git_identity": git_identity,
        "post_files": {},
    }
    # Begin attribution only after every fallible snapshot read completed.  All
    # deeper calls in this thread (including diagnostics) inherit the context,
    # while another thread receives an independent journal.
    snapshot["environment_transaction"] = (
        provider_config_module.begin_managed_environment_transaction()
    )
    return snapshot


def _record_setup_post_state(workspace: Path, snapshot: dict[str, object]) -> None:
    """Record exact transaction-owned leaf states before calling external checks."""

    selected = set(_SETUP_TRANSACTION_FILES)
    extra_files = snapshot.get("extra_files")
    if isinstance(extra_files, dict):
        selected.update(str(path) for path in extra_files)
    post_files: dict[str, dict[str, object]] = {}
    for relative in sorted(selected):
        try:
            target = _validated_workspace_file(workspace, relative)
        except ValueError as exc:
            raise OSError("managed transaction path changed identity") from exc
        if target is None:
            post_files[relative] = {"kind": "absent"}
        else:
            post_files[relative] = {
                "kind": "file",
                "content": target.read_bytes(),
                "mode": target.stat().st_mode & 0o7777,
            }
    snapshot["post_files"] = post_files
    checkpoint_root = workspace / "checkpoints"
    snapshot["post_checkpoint_entries"] = (
        {path.name for path in checkpoint_root.iterdir()}
        if checkpoint_root.is_dir() and not checkpoint_root.is_symlink()
        else set()
    )
    snapshot["post_git"] = _setup_git_control_state(
        workspace,
        complete=not bool(snapshot.get("had_git")),
    )


def _commit_setup_environment(snapshot: dict[str, object]) -> None:
    transaction = snapshot.get("environment_transaction")
    commit = getattr(transaction, "commit", None)
    if not callable(commit):  # pragma: no cover - internal invariant guard
        raise OSError("managed environment transaction snapshot is invalid")
    commit()


def _setup_git_control_state(
    workspace: Path,
    *,
    complete: bool = False,
) -> dict[str, object]:
    """Fingerprint ref/index/reflog state without following a .git symlink."""

    control = workspace / ".git"
    if control.is_symlink():
        return {"kind": "symlink"}
    if not control.exists():
        return {"kind": "absent"}
    had_git, _staged, _tracked = _workspace_git_changes(workspace)
    if not had_git:
        return {"kind": "unrecognized"}

    def command(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(workspace), *arguments],
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            return b""
        return completed.stdout

    git_dir_raw = (
        command("rev-parse", "--absolute-git-dir")
        .decode("utf-8", errors="surrogateescape")
        .strip()
    )
    git_dir = Path(git_dir_raw)
    common_dir_raw = (
        command("rev-parse", "--git-common-dir")
        .decode("utf-8", errors="surrogateescape")
        .strip()
    )
    common_dir = Path(common_dir_raw)
    if not common_dir.is_absolute():
        common_dir = (workspace / common_dir).resolve()

    def digest_file(path: Path) -> str | None:
        if path.is_symlink() or not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    control_tree: dict[str, tuple[str, object, int]] | None = None
    if complete:
        import stat

        control_tree = {}
        for path in sorted(git_dir.rglob("*")):
            relative = path.relative_to(git_dir).as_posix()
            metadata = path.lstat()
            mode = metadata.st_mode & 0o7777
            if stat.S_ISREG(metadata.st_mode):
                control_tree[relative] = (
                    "file",
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    mode,
                )
            elif stat.S_ISDIR(metadata.st_mode):
                control_tree[relative] = ("directory", "", mode)
            elif stat.S_ISLNK(metadata.st_mode):
                control_tree[relative] = ("symlink", path.readlink().as_posix(), mode)
            else:
                control_tree[relative] = ("special", metadata.st_rdev, mode)

    log_digests: dict[str, str] = {}
    logs_root = git_dir / "logs"
    if logs_root.is_dir() and not logs_root.is_symlink():
        for path in sorted(logs_root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                log_digests[path.relative_to(logs_root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return {
        "kind": "repository",
        "head": command("rev-parse", "--verify", "HEAD"),
        "symbolic_head": command("symbolic-ref", "-q", "HEAD"),
        "refs": command(
            "for-each-ref",
            "--format=%(refname)%00%(objectname)",
            "refs",
        ),
        "index": digest_file(git_dir / "index"),
        "config": digest_file(git_dir / "config"),
        "common_config": digest_file(common_dir / "config"),
        "worktree_config": digest_file(git_dir / "config.worktree"),
        "logs": log_digests,
        "control_tree": control_tree,
    }


def _setup_leaf_matches_post_state(
    workspace: Path,
    relative: str,
    post_files: object,
) -> bool:
    if not isinstance(post_files, dict) or relative not in post_files:
        return True
    expected = post_files[relative]
    if not isinstance(expected, dict):
        return False
    try:
        target = _validated_workspace_file(workspace, relative)
    except ValueError:
        return False
    if expected.get("kind") == "absent":
        return target is None
    return bool(
        target is not None
        and expected.get("kind") == "file"
        and isinstance(expected.get("content"), bytes)
        and target.read_bytes() == expected["content"]
        and target.stat().st_mode & 0o7777 == int(expected.get("mode") or 0)
    )


def _rollback_setup_workspace(workspace: Path, snapshot: dict[str, object]) -> None:
    """Undo only paths created or changed by one failed setup invocation."""

    rollback_failed = False

    environment_transaction = snapshot.get("environment_transaction")
    rollback_environment = getattr(environment_transaction, "rollback", None)
    if not callable(rollback_environment):
        rollback_failed = True
    elif rollback_environment():
        rollback_failed = True

    post_git = snapshot.get("post_git")
    git_control_owned = bool(
        isinstance(post_git, dict)
        and _setup_git_control_state(
            workspace,
            complete=not bool(snapshot.get("had_git")),
        )
        == post_git
    )
    if bool(snapshot["had_git"]) and not git_control_owned:
        rollback_failed = True
    elif bool(snapshot["had_git"]):
        head = snapshot.get("head")
        current = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        current_head = current.stdout.strip() if current.returncode == 0 else None
        if head and current_head != head:
            subprocess.run(
                ["git", "-C", str(workspace), "reset", "--mixed", str(head)],
                check=True,
                capture_output=True,
            )
        elif not head and current_head:
            symbolic_ref = snapshot.get("symbolic_ref")
            if symbolic_ref:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(workspace),
                        "update-ref",
                        "-d",
                        str(symbolic_ref),
                    ],
                    check=True,
                    capture_output=True,
                )
            subprocess.run(
                ["git", "-C", str(workspace), "read-tree", "--empty"],
                check=True,
                capture_output=True,
            )
        git_identity = snapshot.get("git_identity")
        current_git_state = _setup_git_control_state(workspace)
        config_identity_unchanged = bool(
            isinstance(post_git, dict)
            and all(
                current_git_state.get(key) == post_git.get(key)
                for key in ("config", "common_config", "worktree_config")
            )
        )
        if isinstance(git_identity, dict) and config_identity_unchanged:
            for key, state in git_identity.items():
                if not isinstance(state, dict):
                    continue
                if state.get("present"):
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(workspace),
                            "config",
                            "--local",
                            str(key),
                            str(state.get("value") or ""),
                        ],
                        check=True,
                        capture_output=True,
                    )
                else:
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(workspace),
                            "config",
                            "--local",
                            "--unset-all",
                            str(key),
                        ],
                        check=False,
                        capture_output=True,
                    )
        elif isinstance(git_identity, dict):
            rollback_failed = True

    files = snapshot["files"]
    assert isinstance(files, dict)
    post_files = snapshot.get("post_files")
    for relative, original in files.items():
        target = workspace / relative
        if not _setup_leaf_matches_post_state(workspace, relative, post_files):
            rollback_failed = True
            continue
        if original is None:
            if target.is_file() and not target.is_symlink():
                target.unlink(missing_ok=True)
            continue
        if not isinstance(original, dict) or not isinstance(
            original.get("content"), bytes
        ):  # pragma: no cover - invariant guard
            raise OSError("invalid setup rollback snapshot")
        if target.is_symlink():
            rollback_failed = True
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original["content"])
        target.chmod(int(original.get("mode") or 0o600))

    extra_files = snapshot.get("extra_files")
    if isinstance(extra_files, dict):
        for relative, original in extra_files.items():
            target = workspace / relative
            if not _setup_leaf_matches_post_state(workspace, relative, post_files):
                rollback_failed = True
                continue
            if original is None:
                if target.is_file() and not target.is_symlink():
                    target.unlink(missing_ok=True)
                continue
            if not isinstance(original, dict) or not isinstance(
                original.get("content"), bytes
            ):
                raise OSError("invalid setup environment rollback snapshot")
            if target.is_symlink():
                rollback_failed = True
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original["content"])
            target.chmod(int(original.get("mode") or 0o600))
    extra_parents = snapshot.get("extra_parents")
    if isinstance(extra_parents, dict):
        for relative, existed in sorted(
            extra_parents.items(), key=lambda item: len(item[0]), reverse=True
        ):
            target = workspace / relative
            if (
                not existed
                and target.is_dir()
                and not target.is_symlink()
                and not any(target.iterdir())
            ):
                target.rmdir()

    checkpoint_entries = snapshot["checkpoint_entries"]
    assert isinstance(checkpoint_entries, set)
    post_checkpoint_entries = snapshot.get("post_checkpoint_entries")
    owned_checkpoint_entries = (
        set(post_checkpoint_entries) - checkpoint_entries
        if isinstance(post_checkpoint_entries, set)
        else set()
    )
    checkpoint_root = workspace / "checkpoints"
    if checkpoint_root.is_dir() and not checkpoint_root.is_symlink():
        for path in checkpoint_root.iterdir():
            if path.name in checkpoint_entries:
                continue
            if path.name not in owned_checkpoint_entries:
                rollback_failed = True
                continue
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
            else:
                rollback_failed = True

    directories = snapshot["directories"]
    assert isinstance(directories, dict)
    for relative in sorted(_SETUP_TRANSACTION_DIRS, key=len, reverse=True):
        target = workspace / relative
        state = directories[relative]
        if not isinstance(state, dict):  # pragma: no cover - invariant guard
            raise OSError("invalid setup directory rollback snapshot")
        kind = state.get("kind")
        if kind == "absent" and (target.exists() or target.is_symlink()):
            if (
                relative == ".git"
                and git_control_owned
                and target.is_dir()
                and not target.is_symlink()
            ):
                shutil.rmtree(target)
            elif (
                target.is_dir()
                and not target.is_symlink()
                and not any(target.iterdir())
            ):
                target.rmdir()
            else:
                rollback_failed = True
        elif kind == "file":
            content = state.get("content")
            if not isinstance(content, bytes):
                raise OSError("invalid setup gitfile rollback snapshot")
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                raise OSError(
                    "setup replaced a pre-existing managed file with a directory"
                )
            target.write_bytes(content)
            target.chmod(int(state.get("mode") or 0o600))

    if not bool(snapshot["workspace_existed"]):
        if (
            workspace.is_dir()
            and not workspace.is_symlink()
            and not any(workspace.iterdir())
        ):
            workspace.rmdir()
        elif workspace.exists() or workspace.is_symlink():
            rollback_failed = True
    if rollback_failed:
        raise OSError(
            "setup rollback preserved paths whose identity changed concurrently"
        )


def _setup_env_file_relative(workspace: Path) -> str:
    """Resolve the provider env file as a safe workspace-relative regular path."""

    from .provider_config import resolve_env_file

    config_path = _validated_workspace_file(
        workspace,
        ".xscientist/providers.json",
    )
    if config_path is None:
        relative = Path(".env")
    else:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing provider configuration is unreadable") from exc
        if not isinstance(payload, dict):
            raise ValueError("existing provider configuration must be an object")
        raw_env_file = payload.get("env_file") or ".env"
        if not isinstance(raw_env_file, str) or not raw_env_file.strip():
            raise ValueError("existing provider configuration has invalid env_file")
        relative = Path(raw_env_file.strip())
    resolved = resolve_env_file(workspace, relative.as_posix())
    return resolved.relative_to(workspace.resolve()).as_posix()


def _initialize_cli_research_repository(
    workspace: Path,
    *,
    name: str,
    question: str,
    actor: str = "xscientist",
    commit: bool = True,
) -> object:
    """Initialize Research VCS while committing only CLI-confirmed source files."""

    from .research_git import ResearchGitError, init_repository
    from .research_vcs import ResearchRepository

    init_repository(
        workspace,
        name=name,
        question=question,
        policy="milestone",
        actor=actor,
        commit=False,
    )
    repository = ResearchRepository(workspace)
    if not commit:
        return repository
    initial_candidates = {
        ".gitignore",
        "research.yaml",
        "question.md",
        ".xscientist/README.md",
    }
    initial_paths = sorted(
        initial_candidates & set(repository.status().get("eligible_changes") or [])
    )
    if not initial_paths:
        raise ResearchGitError("Research VCS initialization produced no source changes")
    repository.stage(initial_paths)
    checkpoint = repository.commit(
        stage="init",
        subject=f"initialize {name}",
        summary="Initialize only the confirmed local research source files.",
        status="completed",
        actor=actor,
        staged_only=True,
    )
    if not checkpoint.committed:
        raise ResearchGitError("Research VCS initialization was not checkpointed")
    return repository


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
                _safe_json_error_payload(
                    message,
                    schema="xscientist.start.v1",
                    phase=phase,
                    returncode=returncode,
                    next_actions=list(next_actions),
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"xscientist start: {message}", file=sys.stderr)
    return returncode


def _run_start(parsed: argparse.Namespace) -> int:
    """Orchestrate the safe first-run path without hiding scientific gates."""

    # Machine-readable mode is also machine-input mode. This single assignment
    # covers question/model selection, local identity, and hidden credential
    # prompts throughout the shared preparation path.
    parsed.non_interactive = bool(parsed.non_interactive or parsed.as_json)

    import contextlib
    import io
    import os

    import yaml

    from ai_scientist.utils.atomic_io import atomic_write_text
    from ai_scientist.utils.auth_session import create_session, validate_session
    from ai_scientist.utils.llm_budget import resolve_model_price
    from .diagnostics import diagnose
    from .onboarding import (
        WORKSPACE_FILES,
        WorkspaceInitError,
        _render_topic,
        create_workspace,
        ensure_workspace_gitignore,
        resolve_workspace_root,
    )
    from .dependency_profiles import TASK_PROFILES
    from .provider_config import (
        DEFAULT_MODELS,
        ProviderConfigError,
        load_provider_config,
        load_workspace_environment,
        normalize_provider_name,
        validate_provider_model,
    )
    from .research_git import ResearchGitError, create_checkpoint, repository_status
    from .research_vcs import ResearchRepository

    try:
        workspace = resolve_workspace_root(parsed.directory)
        provider_config_path = _validated_workspace_file(
            workspace,
            ".xscientist/providers.json",
        )
        _validated_workspace_file(workspace, "bfts_config.yaml")
        research_config_path = _validated_workspace_file(workspace, "research.yaml")
        existing_topic_path = _validated_workspace_file(workspace, "topic.md")
        existing_question_path = _validated_workspace_file(workspace, "question.md")
    except ValueError as exc:
        return _start_input_error(parsed, str(exc))
    new_workspace = provider_config_path is None
    existing_research = research_config_path is not None
    existing_topic_was_present = existing_topic_path is not None
    establish_packaged_question = False
    if existing_research:
        try:
            had_research_git, _git_stage, _git_changes = _workspace_git_changes(
                workspace,
                required=True,
            )
            if not had_research_git or not _workspace_git_head_contains(
                workspace, "research.yaml"
            ):
                raise ValueError(
                    "research.yaml is not owned by the exact-root Git HEAD; "
                    "refusing to treat it as Research VCS provenance"
                )
            existing_status = repository_status(workspace)
            if not existing_status.get("head") or not existing_status.get(
                "last_checkpoint"
            ):
                raise ValueError(
                    "research.yaml has no verifiable Research VCS checkpoint lineage"
                )
            pending_native_stage = (existing_status.get("research_stage") or {}).get(
                "paths"
            ) or []
            if pending_native_stage:
                raise ValueError(
                    "start refused because the native Research VCS stage contains "
                    "pending work: "
                    + ", ".join(sorted(str(path) for path in pending_native_stage))
                )
            if existing_status.get("staged_paths"):
                raise ValueError(
                    "start refused because the Git index contains staged work"
                )
            parsed.question, establish_packaged_question = (
                _validated_existing_research_question(
                    workspace,
                    parsed.question,
                )
            )
        except (OSError, ResearchGitError, UnicodeError, ValueError) as exc:
            return _start_input_error(parsed, str(exc))
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

    target_topic = f"# Research question\n\n{question}\n"
    from ai_scientist.utils.privacy import redact_sensitive_text

    if redact_sensitive_text(question) != question:
        return _start_input_error(
            parsed,
            "research questions must not contain credentials, host-local paths, "
            "email addresses, or other private literals",
        )
    start_protected_paths: set[str] = set()
    if not existing_research:
        if existing_topic_path is not None:
            existing_topic = existing_topic_path.read_text(encoding="utf-8")
            if existing_topic not in {target_topic, _render_topic()}:
                return _start_input_error(
                    parsed,
                    "existing topic.md is neither the requested research question "
                    "nor the pristine packaged placeholder; refusing to overwrite it",
                )
        if existing_question_path is not None:
            existing_question = existing_question_path.read_text(encoding="utf-8")
            if existing_question != target_topic:
                return _start_input_error(
                    parsed,
                    "question.md exists without Research VCS provenance and does "
                    "not exactly match --question; refusing to overwrite it",
                )
        try:
            start_protected_paths = {
                relative
                for relative in WORKSPACE_FILES
                if _validated_workspace_file(workspace, relative) is not None
            }
        except ValueError as exc:
            return _start_input_error(parsed, str(exc))
    start_checkpoint_protected_paths = start_protected_paths - {".gitignore"}

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

    try:
        if (workspace / ".git").is_symlink():
            raise WorkspaceInitError("start refuses a symlinked .git control path")
        unsafe_managed_directories = [
            relative
            for relative in _SETUP_TRANSACTION_DIRS
            if relative != ".git"
            and (
                (workspace / relative).is_symlink()
                or (
                    (workspace / relative).exists()
                    and not (workspace / relative).is_dir()
                )
            )
        ]
        if unsafe_managed_directories:
            raise WorkspaceInitError(
                "start refused unsafe managed directories: "
                + ", ".join(sorted(unsafe_managed_directories))
            )
        _had_git, start_staged, _tracked = _workspace_git_changes(
            workspace,
            required=True,
        )
        git_control_path = workspace / ".git"
        if (
            git_control_path.exists() or git_control_path.is_symlink()
        ) and not _had_git:
            raise WorkspaceInitError(
                "start refused an existing .git control path that is not an "
                "exact-root Git worktree"
            )
        if start_staged:
            raise WorkspaceInitError(
                "start refused because the existing Git index contains staged work"
            )
        start_env_file = _setup_env_file_relative(workspace)
        start_snapshot = _setup_workspace_snapshot(
            workspace,
            extra_files=tuple(dict.fromkeys((start_env_file, ".env"))),
        )
    except (OSError, WorkspaceInitError, ValueError) as exc:
        return _start_input_error(parsed, str(exc))

    phases: dict[str, object] = {}
    start_mutated = True

    def rollback_preparation() -> None:
        nonlocal start_mutated
        if start_mutated:
            _rollback_setup_workspace(workspace, start_snapshot)
            start_mutated = False

    try:
        config_exists = (
            _validated_workspace_file(
                workspace,
                ".xscientist/providers.json",
            )
            is not None
        )
        existing_config = (
            load_provider_config(workspace, missing_ok=False) if config_exists else {}
        )
        requested_provider = str(
            parsed.provider or existing_config.get("active_provider") or ""
        )
        selected_provider = normalize_provider_name(requested_provider)
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
        model_source = (
            "start_argument"
            if parsed.model
            else ("workspace_provider_config" if existing_model else "provider_default")
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
        for label, value in (
            ("provider", selected_provider),
            ("model", selected_model),
            ("task", selected_task),
        ):
            if redact_sensitive_text(value) != value:
                raise ProviderConfigError(
                    f"{label} contains a credential or private literal; refusing "
                    "to persist it"
                )
        model_contract = _research_model_contract(
            selected_model,
            source=model_source,
        )
        workspace_creation: dict[str, object] | None = None
        if not config_exists:
            workspace_creation = create_workspace(
                workspace,
                profile=parsed.profile,
                provider=selected_provider,
                model=selected_model,
                task=selected_task,
                capabilities=selected_capabilities,
                provider_required=True,
                preserve_existing=existing_research,
                preserve_paths=start_protected_paths,
                force=bool(parsed.force or start_protected_paths),
            )
            if existing_research and not existing_topic_was_present:
                # ``topic.md`` is an onboarding template, not permission to add a
                # second scientific source to an existing Research VCS. Remove
                # only the file created by this call and keep the original absence.
                generated_topic = workspace / "topic.md"
                if generated_topic.is_symlink() or (
                    generated_topic.exists() and not generated_topic.is_file()
                ):
                    raise WorkspaceInitError(
                        "runtime setup created an unsafe topic.md in the existing "
                        "Research VCS workspace"
                    )
                if generated_topic.is_file():
                    generated_topic.unlink()
                workspace_creation["files"] = [
                    path
                    for path in workspace_creation.get("files", [])
                    if path != "topic.md"
                ]
        _record_setup_post_state(workspace, start_snapshot)
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

        question_checkpoint_id: str | None = None
        question_checkpoint_pending = False
        repository_created = False
        topic = target_topic
        if establish_packaged_question:
            repository_before = repository_status(workspace)
            pending_stage = set(repository_before.get("staged_paths") or []) | set(
                (repository_before.get("research_stage") or {}).get("paths") or []
            )
            if pending_stage:
                raise WorkspaceInitError(
                    "cannot establish the packaged research question while the "
                    "repository has staged work"
                )
            question_path = workspace / "question.md"
            topic_path = workspace / "topic.md"
            atomic_write_text(question_path, topic)
            atomic_write_text(topic_path, topic)
            question_checkpoint_pending = True
            status = repository_status(workspace)
            vcs_created = False
        elif not existing_research:
            atomic_write_text(workspace / "topic.md", topic)
            repository = _initialize_cli_research_repository(
                workspace,
                name=workspace.name,
                question=topic,
                actor=resolved_actor or "xscientist",
                commit=False,
            )
            status = repository.status()
            repository_created = True
            vcs_created = True
        else:
            status = repository_status(workspace)
            vcs_created = False
        phases["research_vcs"] = {
            "ok": True,
            "created": vcs_created,
            "question_established": question_checkpoint_pending,
            "branch": status.get("branch"),
            "checkpoint_id": (status.get("last_checkpoint") or {}).get("checkpoint_id"),
        }
        if workspace_creation is not None:
            ensure_workspace_gitignore(workspace)

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
                base_url_value=parsed.base_url,
                non_interactive=parsed.non_interactive,
            )
        phases["provider"] = {
            "ok": bool(provider_result.get("ok")),
            "ready": bool(provider_result.get("ready")),
            "provider": selected_provider,
            "model": selected_model,
            "model_contract": model_contract,
            "reason": provider_result.get("reason"),
        }
        _record_setup_post_state(workspace, start_snapshot)

        load_result = load_workspace_environment(workspace)
        if load_result.get("error"):
            raise ProviderConfigError(str(load_result["error"]))
        _record_setup_post_state(workspace, start_snapshot)

        budget_path = _validated_workspace_file(workspace, "bfts_config.yaml")
        if budget_path is None:
            raise ProviderConfigError("workspace is missing bfts_config.yaml")
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
        pending_runtime_paths: list[str] = []
        if workspace_creation is not None:
            repository = ResearchRepository(workspace)
            generated_paths = (
                set(workspace_creation["files"]) - start_checkpoint_protected_paths
            )
            pending_runtime_paths = [
                path
                for path in repository.status()["eligible_changes"]
                if path in generated_paths
            ]
        _record_setup_post_state(workspace, start_snapshot)
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
            rollback_preparation()
            if parsed.as_json:
                print(
                    json.dumps(
                        _safe_public_json_payload(payload),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
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

        _record_setup_post_state(workspace, start_snapshot)
        report = diagnose(
            workspace,
            task=selected_task,
            provider=selected_provider,
            deep=True,
        )
        _persist_readiness_report(workspace, report)
        _record_setup_post_state(workspace, start_snapshot)
        phases["doctor"] = report
        committed_checkpoint = None
        readiness_status = "completed" if report["ok"] else "blocked"
        persist_preparation = bool(report["ok"] or existing_research)
        if persist_preparation:
            repository = ResearchRepository(workspace)
        if persist_preparation and repository_created:
            initial_candidates = {
                ".gitignore",
                ".xscientist/README.md",
                "question.md",
                "research.yaml",
            }
            runtime_candidates = initial_candidates | set(pending_runtime_paths)
            eligible_paths = set(repository.status().get("eligible_changes") or [])
            initial_paths = sorted(runtime_candidates & eligible_paths)
            if not initial_paths:
                raise ResearchGitError(
                    "Research VCS initialization produced no source changes"
                )
            repository.stage(initial_paths)
            committed_checkpoint = repository.commit(
                stage="init",
                subject=f"initialize {workspace.name}",
                summary=(
                    "Initialize only confirmed local research sources and runtime "
                    "files after readiness checks completed."
                ),
                status=readiness_status,
                actor=resolved_actor or None,
                staged_only=True,
            )
            phases["research_vcs"]["question_established"] = True
        elif persist_preparation:
            runtime_paths = sorted(
                set(pending_runtime_paths)
                & set(repository.status().get("eligible_changes") or [])
            )
            if question_checkpoint_pending:
                committed_checkpoint = create_checkpoint(
                    workspace,
                    stage="ideation",
                    subject="establish the research question",
                    summary=(
                        "Replace the pristine packaged placeholder and checkpoint "
                        "only runtime files prepared by this start invocation."
                    ),
                    status=readiness_status,
                    actor=resolved_actor or None,
                    only_paths=["question.md", "topic.md", *runtime_paths],
                )
                phases["research_vcs"]["question_established"] = True
            elif runtime_paths:
                repository.stage(runtime_paths)
                committed_checkpoint = repository.commit(
                    stage="setup",
                    subject="refresh the generated research runtime",
                    summary=(
                        "Checkpoint only runtime files prepared by this start "
                        "invocation after readiness checks completed."
                    ),
                    status=readiness_status,
                    actor=resolved_actor or None,
                    staged_only=True,
                )
        if committed_checkpoint is not None:
            if not committed_checkpoint.committed:
                raise ResearchGitError(
                    "prepared research sources were not checkpointed"
                )
            question_checkpoint_id = committed_checkpoint.checkpoint_id
            phases["research_vcs"]["checkpoint_id"] = question_checkpoint_id
            research_check = (report.get("checks") or {}).get("research_vcs")
            if isinstance(research_check, dict):
                research_check["head"] = committed_checkpoint.commit
                research_check["checkpoint_id"] = question_checkpoint_id
    except BaseException as exc:
        try:
            rollback_preparation()
        except Exception:
            exc = WorkspaceInitError(
                "start preparation failed and managed rollback was incomplete"
            )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        payload = _safe_json_error_payload(
            exc,
            schema="xscientist.start.v1",
            phase="prepare",
            phases=phases,
        )
        if parsed.as_json:
            print(
                json.dumps(
                    _safe_public_json_payload(payload),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"xscientist start stopped during preparation: {exc}", file=sys.stderr
            )
        return 2

    if not report["ok"]:
        if existing_research:
            _commit_setup_environment(start_snapshot)
            start_mutated = False
        else:
            try:
                rollback_preparation()
            except (OSError, subprocess.SubprocessError):
                return _start_input_error(
                    parsed,
                    "start preparation failed and managed rollback was incomplete",
                    phase="prepare",
                )
        payload = {
            "schema": "xscientist.start.v1",
            "ok": False,
            "phase": "doctor",
            "workspace": ".",
            "phases": phases,
            "next_actions": report["next_actions"],
        }
        if parsed.as_json:
            print(
                json.dumps(
                    _safe_public_json_payload(payload),
                    indent=2,
                    ensure_ascii=False,
                )
            )
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

    _commit_setup_environment(start_snapshot)
    start_mutated = False
    if parsed.prepare_only:
        payload = {
            "schema": "xscientist.start.v1",
            "ok": True,
            "phase": "ready",
            "workspace": ".",
            "phases": phases,
        }
        if parsed.as_json:
            print(
                json.dumps(
                    _safe_public_json_payload(payload),
                    indent=2,
                    ensure_ascii=False,
                )
            )
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
    project_args.extend(_research_model_arguments(selected_model))
    if not parsed.as_json:
        print(
            "Research model contract: "
            f"{model_contract['provider']}/{model_contract['client_model']} "
            f"(source: {model_contract['selection_source']}; "
            "all research roles explicit)"
        )
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
        "model_contract": model_contract,
        "research_dag": ("outputs/views/{workspace}/research-dag/research-dag.html"),
        "phases": phases,
    }
    if parsed.as_json:
        if returncode:
            payload["error"] = captured_err.getvalue().strip().splitlines()[-1:]
        from ai_scientist.utils.privacy import redact_sensitive_payload

        payload = redact_sensitive_payload(payload)
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
        advanced_help = list(_SPECIAL_COMMAND_OPTIONS["help"])
        if raw_argv[1:] in ([], advanced_help):
            _print_curated_help(include_advanced=raw_argv[1:] == advanced_help)
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
            "explore",
            "audit",
            "history",
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
        not in {
            "provider",
            "info",
            "init",
            "explore",
            "start",
            "setup",
            "doctor",
            "capability",
        }
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
    try:
        if lazy_builder is not None:
            parsed = lazy_builder().parse_args(raw_argv[1:])
            parsed.command = raw_argv[0]
        else:
            parser = _build_parser()
            parsed = parser.parse_args(raw_argv)
    except _ArgumentParseFailure as exc:
        from ai_scientist.utils.privacy import redact_sensitive_text

        command = raw_argv[0] if raw_argv else "command"
        safe_error = redact_sensitive_text(str(exc))
        if "--json" in raw_argv and command in {"init", "setup", "start"}:
            print(
                json.dumps(
                    _safe_json_error_payload(
                        safe_error,
                        schema=f"xscientist.{command}.v1",
                        phase="input",
                    ),
                    ensure_ascii=False,
                )
            )
        else:
            print(f"xscientist {command}: {safe_error}", file=sys.stderr)
        return 2
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
            public_run_view,
            read_run_logs,
        )

        if parsed.detach:
            try:
                payload = launch_detached_run(parsed.directory, raw_argv)
            except (OSError, RunControlError, ValueError) as exc:
                return _start_input_error(parsed, str(exc), phase="launch")
            payload = public_run_view(payload)
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
                from ai_scientist.utils.privacy import redact_sensitive_payload

                safe_payload = redact_sensitive_payload(payload)
                print(json.dumps(safe_payload, indent=2, ensure_ascii=False))
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
        from .benchmark import (
            benchmark_autoresearch_pilot,
            benchmark_first_run,
            persist_benchmark_report,
            verify_benchmark_report,
        )
        from .research_git import ResearchGitError

        try:
            if parsed.benchmark_command == "verify":
                payload = verify_benchmark_report(parsed.report)
                if parsed.as_json:
                    print(json.dumps(payload, indent=2, ensure_ascii=False))
                else:
                    state = "PASS" if payload.get("ok") else "FAIL"
                    print(f"Benchmark report verification: {state}")
                    for name, result in (payload.get("checks") or {}).items():
                        print(f"  {name}: {result}")
                    for error in payload.get("errors") or []:
                        print(f"  error: {error}")
                return 0 if payload.get("ok") else 1
            if parsed.benchmark_command == "systems":
                from .system_comparison import build_system_comparison

                payload = build_system_comparison(parsed.workspace)
            elif parsed.benchmark_command == "autoresearch":
                payload = benchmark_autoresearch_pilot(
                    parsed.tasks,
                    workspace=parsed.workspace,
                    limit=parsed.limit,
                    task_kind=parsed.kind,
                )
            else:
                payload = benchmark_first_run(
                    parsed.workspace,
                    profile=parsed.profile,
                    max_seconds=parsed.max_seconds,
                )
        except (OSError, ValueError, ResearchGitError, RuntimeError) as exc:
            print(f"xscientist benchmark: {exc}", file=sys.stderr)
            return 2
        output_path = getattr(parsed, "output", None)
        persisted_path = None
        if output_path:
            destination_digest = hashlib.sha256(
                str(Path(output_path).expanduser().resolve()).encode("utf-8")
            ).hexdigest()[:16]
            payload = dict(payload)
            payload["report_persistence"] = {
                "requested": True,
                "format": "json",
                "destination_digest": f"sha256:{destination_digest}",
                "raw_payloads_included": False,
            }
            try:
                persisted_path = persist_benchmark_report(payload, output_path)
            except (OSError, ValueError) as exc:
                print(
                    f"xscientist benchmark: cannot write report: {exc}", file=sys.stderr
                )
                return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            if persisted_path is not None:
                print(
                    f"Benchmark report written to {persisted_path}",
                    file=sys.stderr,
                )
        elif parsed.benchmark_command == "systems":
            print(
                "System comparison: qualitative source audit only; "
                "no cross-system score, provider, network, or external rollout."
            )
            print(f"Systems covered: {len(payload['systems'])}")
            local = payload.get("xscientist_local") or {}
            print(
                "XScientist local observation: "
                f"{local.get('status', 'unknown')} "
                f"(audit rollouts {local.get('rollouts', 0)}, "
                f"audit cost ${local.get('model_cost_usd', 0.0):.2f}; "
                "historical trajectory cost unobserved)"
            )
            if parsed.show_process:
                process = local.get("process") or {}
                if not process:
                    print("Process: not requested (pass --workspace)")
                elif not process.get("available", False):
                    print(
                        "Process: unavailable " f"({process.get('reason', 'unknown')})"
                    )
                else:
                    topology = process.get("branch_topology") or {}
                    branches = process.get("branches") or []
                    commits = process.get("commits") or []
                    intermediate = process.get("intermediate") or {}
                    source_branches = topology.get("source_branch_count", len(branches))
                    print(
                        "Process: "
                        f"{len(commits)} visible commits, "
                        f"{len(branches)}/{source_branches} branches, "
                        f"{intermediate.get('object_count', 0)} typed artifacts"
                    )
                    print(
                        "  Branch topology: "
                        f"branching={'yes' if topology.get('branching_observed') else 'no'}, "
                        f"merge={'yes' if topology.get('merge_observed') else 'no'}"
                    )
                    print(
                        "Fairness: "
                        + (
                            "ELIGIBLE"
                            if topology.get("fair_branch_comparison", {}).get(
                                "eligible"
                            )
                            else "NOT VERIFIED"
                        )
                    )
                    print(
                        "Artifact scope: "
                        f"{topology.get('artifact_scope', 'unavailable')}"
                    )
                    print(
                        "  Attempts: "
                        f"{intermediate.get('attempt_count', 0)} total, "
                        f"{intermediate.get('failed_attempts', 0)} failed, "
                        f"{intermediate.get('completed_attempts', 0)} completed"
                        + (
                            " (header totals; artifact rows truncated)"
                            if intermediate.get("attempts_truncated")
                            else ""
                        )
                    )
                    print(
                        "  Decisions: "
                        f"{len(intermediate.get('decision_events') or [])} visible / "
                        f"{(process.get('limits') or {}).get('source_totals', {}).get('decision_events', 0)} source"
                    )
                    fair = topology.get("fair_branch_comparison") or {}
                    checks = fair.get("checks") or {}
                    unverified = [
                        name
                        for name in fair.get("requirements") or []
                        if checks.get(name) is not True
                    ]
                    print(
                        "  Fair branch comparison: "
                        + ("ELIGIBLE" if fair.get("eligible") else "NOT VERIFIED")
                        + (
                            " (unverified: " + ", ".join(unverified) + ")"
                            if unverified
                            else ""
                        )
                    )
                    truncated = (process.get("limits") or {}).get("truncated") or {}
                    if any(truncated.values()):
                        print(
                            "  Bounds: truncated "
                            + ", ".join(
                                name for name, value in truncated.items() if value
                            )
                        )
                    if parsed.show_process:
                        for branch in branches:
                            print(
                                f"  branch {branch.get('name', '?')} "
                                f"({branch.get('relation', 'line')}, "
                                f"{branch.get('commit_count', 0)} commits)"
                            )
                        for event in commits:
                            memberships = ",".join(event.get("branches") or [])
                            suffix = f" [{memberships}]" if memberships else ""
                            print(
                                f"  commit {event.get('short_commit', '?')} "
                                f"{event.get('stage', 'unknown')}/"
                                f"{event.get('status', 'unknown')}: "
                                f"{event.get('subject', 'checkpoint:unknown')}"
                                f"{suffix}"
                            )
                        for artifact in (intermediate.get("artifacts") or [])[:12]:
                            print(
                                "  artifact "
                                f"{artifact.get('object_id', '?')} "
                                f"{artifact.get('kind', 'extension')}/"
                                f"{artifact.get('stage', 'X')}/"
                                f"{artifact.get('state', 'other')}"
                            )
                        for decision in (intermediate.get("decision_events") or [])[:8]:
                            print(
                                "  decision "
                                f"{decision.get('kind', 'extension')}:"
                                f"{decision.get('decision', 'observed')} "
                                f"issues={decision.get('issue_count', 0)}"
                            )
            print(
                "Use --json for the full source links, dimensions, and redacted process view."
            )
            if persisted_path is not None:
                print(f"Report written: {persisted_path}")
        elif parsed.benchmark_command == "autoresearch":
            tasks = payload["tasks"]
            print(
                "AutoResearchEval-inspired conformance pilot: "
                f"{tasks['valid_task_contracts']}/{tasks['count']} task contracts valid"
            )
            print(
                "Official score: not applicable; no model/provider/network or rollouts used."
            )
            comparison_context = payload.get("comparison_context") or {}
            if comparison_context:
                print(
                    "System comparison: qualitative matrix only; "
                    "external scores were not injected."
                )
            workspace_report = payload.get("workspace")
            if workspace_report:
                print(
                    "Six-stage artifact coverage: "
                    f"{workspace_report['stage_coverage']:.0%} "
                    f"(structural criteria only; "
                    f"{workspace_report.get('stage_score', 0.0):.1f}/100, "
                    "not a quality score)"
                )
                process = workspace_report.get("process") or {}
                if process:
                    repository = process.get("repository") or {}
                    intermediate = process.get("intermediate") or {}
                    if not process.get("available", False):
                        print(
                            "Process: unavailable "
                            f"({process.get('reason', 'unknown')}); "
                            "no local Research VCS evidence was counted"
                        )
                    else:
                        topology = process.get("branch_topology") or {}
                        source_branches = topology.get(
                            "source_branch_count", len(process.get("branches") or [])
                        )
                        print(
                            "Process: "
                            f"{len(process.get('commits') or [])} visible commits, "
                            f"{len(process.get('branches') or [])}/{source_branches} branches, "
                            f"{intermediate.get('object_count', 0)} typed artifacts"
                        )
                        if parsed.show_process:
                            print(
                                "  Branch topology: "
                                f"branching={'yes' if topology.get('branching_observed') else 'no'}, "
                                f"merge={'yes' if topology.get('merge_observed') else 'no'}"
                            )
                            print(
                                "  Reasoning trail: artifact-backed signals only; "
                                "hidden chain-of-thought omitted"
                            )
                            print(
                                "  Artifact scope: "
                                f"{topology.get('artifact_scope', 'current_checkout_only')}"
                            )
                            for branch in process.get("branches") or []:
                                print(
                                    f"  branch {branch.get('name', '?')} "
                                    f"({branch.get('relation', 'line')}, "
                                    f"{branch.get('commit_count', 0)} commits)"
                                )
                            for event in process.get("commits") or []:
                                memberships = ",".join(event.get("branches") or [])
                                suffix = f" [{memberships}]" if memberships else ""
                                print(
                                    f"  commit {event.get('short_commit', '?')} "
                                    f"{event.get('stage', 'unknown')} "
                                    f"{event.get('status', 'unknown')}: "
                                    f"{event.get('subject', 'checkpoint:unknown')}"
                                    f"{suffix}"
                                )
                            coverage = intermediate.get("coverage") or {}
                            print(
                                "  Artifacts: "
                                f"{len(intermediate.get('artifacts') or [])} visible / "
                                f"{intermediate.get('object_count', 0)} source; "
                                f"stages={coverage.get('stage_count', 0)}/"
                                f"{coverage.get('stage_total', 6)}"
                            )
                            for artifact in (intermediate.get("artifacts") or [])[:12]:
                                print(
                                    "  artifact "
                                    f"{artifact.get('object_id', '?')} "
                                    f"{artifact.get('kind', 'extension')}/"
                                    f"{artifact.get('stage', 'X')}/"
                                    f"{artifact.get('state', 'other')}"
                                )
                            print(
                                "  Attempts: "
                                f"{intermediate.get('attempt_count', 0)} total, "
                                f"{intermediate.get('failed_attempts', 0)} failed, "
                                f"{intermediate.get('completed_attempts', 0)} completed"
                                + (
                                    " (sample truncated)"
                                    if intermediate.get("attempts_truncated")
                                    else ""
                                )
                            )
                            for artifact in (
                                intermediate.get("failed_or_blocked_artifacts") or []
                            )[:8]:
                                print(
                                    "  failure "
                                    f"{artifact.get('object_id', '?')} "
                                    f"{artifact.get('kind', 'extension')}/"
                                    f"{artifact.get('state', 'other')} "
                                    f"codes={','.join(artifact.get('failure_codes') or [])}"
                                )
                            for decision in (intermediate.get("decision_events") or [])[
                                :8
                            ]:
                                print(
                                    "  decision "
                                    f"{decision.get('kind', 'extension')}:"
                                    f"{decision.get('decision', 'observed')} "
                                    f"issues={decision.get('issue_count', 0)}"
                                )
                            fair = topology.get("fair_branch_comparison") or {}
                            checks = fair.get("checks") or {}
                            unverified = fair.get("unverified_reasons") or [
                                name
                                for name in fair.get("requirements") or []
                                if checks.get(name) is not True
                            ]
                            print(
                                "  Fair branch comparison: "
                                + (
                                    "ELIGIBLE"
                                    if fair.get("eligible")
                                    else "NOT VERIFIED"
                                )
                                + (
                                    " (unverified: " + ", ".join(unverified) + ")"
                                    if unverified
                                    else ""
                                )
                            )
                            truncated = process.get("limits", {}).get("truncated") or {}
                            if any(truncated.values()):
                                print(
                                    "  Bounds: truncated "
                                    + ", ".join(
                                        name
                                        for name, value in truncated.items()
                                        if value
                                    )
                                )
                            evidence_index = (
                                workspace_report.get("evidence_index") or {}
                            )
                            categories = evidence_index.get("categories") or {}
                            present = [
                                name
                                for name, row in categories.items()
                                if isinstance(row, dict) and row.get("present")
                            ]
                            print(
                                "  Evidence index: "
                                + (", ".join(present) if present else "none")
                                + (
                                    " (bounded/truncated)"
                                    if evidence_index.get("truncated")
                                    else ""
                                )
                            )
                            exploration = workspace_report.get("exploration") or {}
                            print(
                                "  Exploration: "
                                f"{exploration.get('status', 'unavailable')}"
                                + (
                                    f" ({exploration.get('attempted')} attempted, "
                                    f"{exploration.get('unattempted')} unattempted)"
                                    if exploration.get("status")
                                    in {"observed", "partially_observed"}
                                    else ""
                                )
                            )
                            if exploration.get("counts_are_nonexclusive"):
                                print("  Exploration counters: non-exclusive")
                        if repository.get("worktree_clean") is False:
                            print("  WARN  worktree has uncheckpointed changes")
                arft_summary = workspace_report.get("arft_coverage", {}).get(
                    "summary", {}
                )
                if "coverage_score" in arft_summary:
                    print(
                        "ARFT evidence-channel coverage: "
                        f"{float(arft_summary['coverage_score']):.1f}% "
                        "(structural only; not a quality score)"
                    )
                for stage, result in workspace_report["stages"].items():
                    state = (
                        "PASS"
                        if result.get("complete")
                        else "MIN" if result.get("covered") else "MISS"
                    )
                    print(
                        f"{state}  "
                        f"{result.get('label', stage)} ({result.get('score', 0.0):.1f})"
                    )
                metacognition = workspace_report.get("metacognition") or {}
                print(
                    "Metacognitive loop: "
                    f"{metacognition.get('status', 'unavailable')} "
                    f"(issues {metacognition.get('issue_count', 0)}, "
                    f"repaired {metacognition.get('repaired_issue_count', 0)})"
                )
                closure = workspace_report.get("closure") or {}
                if closure.get("available"):
                    levels = closure.get("levels") or {}
                    print(
                        "Closure: "
                        + ", ".join(
                            f"{name}={'PASS' if row.get('complete') else 'BLOCKED'}"
                            for name, row in levels.items()
                        )
                    )
                for signal in workspace_report.get("metacognitive_signals") or []:
                    print(f"WARN  {signal['code']}: {signal['detail']}")
                diagnostics = payload.get("diagnostics") or {}
                if diagnostics:
                    counts = diagnostics.get("priority_counts") or {}
                    print(
                        "Current gap register: "
                        + ", ".join(
                            f"{priority}={counts[priority]}"
                            for priority in ("P0", "P1", "P2")
                            if counts.get(priority)
                        )
                    )
                    for item in (diagnostics.get("items") or [])[:5]:
                        print(
                            f"  {item.get('priority', 'P2')} "
                            f"{item.get('id', 'UNKNOWN')}: "
                            f"{item.get('recommendation', 'inspect evidence')}"
                        )
            else:
                print("Pass --workspace to inspect a local research artifact set.")
                diagnostics = payload.get("diagnostics") or {}
                if diagnostics.get("next_required"):
                    print(
                        "Current required evidence: "
                        f"{diagnostics['next_required']} (structural audit only)"
                    )
            if persisted_path is not None:
                print(f"Report written: {persisted_path}")
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
            if persisted_path is not None:
                print(f"Report written: {persisted_path}")
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
    if parsed.command == "explore":
        from .research_git import ResearchGitError
        from .research_journey import (
            explore_research_idea,
            inspect_idea_research,
            public_exploration_payload,
        )

        try:
            destination = Path(parsed.directory).expanduser()
            existing = (
                inspect_idea_research(destination)
                if (destination / "research.yaml").is_file()
                else None
            )
            _interactive_explore_inputs(parsed, existing=existing)
            payload = explore_research_idea(
                destination,
                idea=parsed.idea,
                expectation=parsed.expectation,
                disconfirming_result=parsed.disconfirming_result,
                first_test=parsed.first_test,
                success_rule=parsed.success_rule,
                name=parsed.name,
                actor=parsed.actor,
                language=parsed.lang,
                git_user_name=parsed.git_user_name,
                git_user_email=parsed.git_user_email,
            )
        except (OSError, ResearchGitError, ValueError) as exc:
            if parsed.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": "xscientist.idea-exploration.v1",
                            "ok": False,
                            "error_code": "idea_exploration_failed",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            else:
                print(f"xscientist explore: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(
                json.dumps(
                    public_exploration_payload(payload, workspace=destination),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            language = payload["language"]
            framing = payload["framing"]
            complete = {
                "question": True,
                "expectation": bool(framing["hypothesis_id"]),
                "disproof": bool(framing["hypothesis_id"]),
                "test": bool(framing["plan_id"]),
            }
            if language == "zh":
                print(f"研究工作区：{payload['repository']}")
                print(f"你的想法：{payload['idea']}")
                print("严谨性检查：")
                print("  ✓ 已原样保存研究问题")
                print(
                    "  "
                    + ("✓" if complete["expectation"] else "○")
                    + " 已说明预期观察到的变化"
                )
                print(
                    "  "
                    + ("✓" if complete["disproof"] else "○")
                    + " 已说明什么结果会推翻预期"
                )
                print(
                    "  "
                    + ("✓" if complete["test"] else "○")
                    + " 已记录第一个公平比较或检验"
                )
                print(
                    "边界：未调用 API、模型或外部网络，未执行代码，也未生成证据或结论。"
                )
                if framing["status"] == "planned":
                    print(
                        "提醒：这只是第一版探索计划；运行前仍要检查替代解释、"
                        "真实数据、指标和独立复核。"
                    )
                    print("下一步：先准备真实数据或执行已记录的比较，再记录结果。")
                    print(f"查看：{payload['status_command']}")
                else:
                    print("下一步：继续回答当前缺少的一个问题。")
                    print(f"继续：{payload['continue_command']}")
                print(
                    "可选：需要 AI 自主推进时再运行 `xscientist provider list`；"
                    "它会优先发现免 Key 的本地 Ollama。"
                )
            else:
                print(f"Research workspace: {payload['repository']}")
                print(f"Your idea: {payload['idea']}")
                print("Rigor check:")
                print("  ✓ Research question saved exactly as supplied")
                print(
                    "  "
                    + ("✓" if complete["expectation"] else "○")
                    + " Expected observable change stated"
                )
                print(
                    "  "
                    + ("✓" if complete["disproof"] else "○")
                    + " Result that would disprove it stated"
                )
                print(
                    "  "
                    + ("✓" if complete["test"] else "○")
                    + " First fair comparison or test recorded"
                )
                print(
                    "Boundary: no API, model, external network, or generated code "
                    "was used; no evidence or conclusion was generated."
                )
                if framing["status"] == "planned":
                    print(
                        "Reminder: this is a first exploratory plan; check rival "
                        "explanations, real data, metrics, and independent review "
                        "before drawing conclusions."
                    )
                    print(
                        "Next: prepare real data or run the recorded comparison, "
                        "then record the result."
                    )
                    print(f"Inspect: {payload['status_command']}")
                else:
                    print("Next: answer the next missing framing question.")
                    print(f"Continue: {payload['continue_command']}")
                print(
                    "Optional: run `xscientist provider list` only when you want AI "
                    "help; it detects key-free local Ollama first."
                )
        _record_local_metric("explore", ok=True)
        return 0
    if parsed.command == "demo":
        import webbrowser

        from .demo import create_autopilot_demo, create_demo, public_demo_payload
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
        public_payload = public_demo_payload(payload, workspace=parsed.directory)
        if parsed.as_json:
            print(json.dumps(public_payload, indent=2, ensure_ascii=False))
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
    if parsed.command == "audit":
        forwarded = [
            "audit",
            parsed.ref,
            "--repo",
            parsed.workspace,
            "--level",
            parsed.level,
        ]
        if parsed.no_objects:
            forwarded.append("--no-objects")
        if parsed.as_json:
            forwarded.append("--json")
        return research_main(forwarded)
    if parsed.command == "history":
        from .workspace_history import run_history_cli

        return run_history_cli(parsed)
    if parsed.command == "status":
        from .workspace_status import (
            build_workspace_status,
            public_workspace_status_payload,
        )

        payload = build_workspace_status(parsed.workspace, language=parsed.lang)
        if parsed.as_json:
            print(
                json.dumps(
                    public_workspace_status_payload(payload),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            language = _selected_language(parsed.lang)
            research = payload["research"]
            run = payload["run"]
            result = payload["result"]
            review = payload["review"]
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
            if research["initialized"]:
                head = str(research.get("head") or "-")[:12]
                clean = bool(review.get("clean"))
                pending = review.get("pending") or {}
                pending_count = sum(
                    len(pending.get(name) or [])
                    for name in ("backend_staged", "selected", "tracked", "eligible")
                )
                if language == "zh":
                    history_state = "干净" if clean else f"待保存={pending_count}"
                    print(f"科研历史：{research['branch']}@{head} / {history_state}")
                else:
                    history_state = "clean" if clean else f"pending={pending_count}"
                    print(f"History: {research['branch']}@{head} / {history_state}")
                if review.get("available"):
                    check_labels = {
                        "en": {
                            "pass": "pass",
                            "pending": "pending",
                            "not_ready": "not ready",
                            "unavailable": "unavailable",
                        },
                        "zh": {
                            "pass": "通过",
                            "pending": "待补充",
                            "not_ready": "尚未产生结论",
                            "unavailable": "不可用",
                        },
                    }[language]
                    checks = review["checks"]
                    label = "科研门禁" if language == "zh" else "Checks"
                    print(
                        f"{label}: "
                        + ", ".join(
                            f"{name}={check_labels.get(checks[name], checks[name])}"
                            for name in ("trace", "replay", "verify")
                        )
                    )
                if parsed.verbose and pending.get("preserved"):
                    label = (
                        "保留的可重建/策略排除文件"
                        if language == "zh"
                        else "Preserved local views/files"
                    )
                    print(f"{label}: {len(pending['preserved'])}")
                evolution = review.get("evolution") or {}
                if parsed.verbose and any(evolution.values()):
                    label = "自主进化记录" if language == "zh" else "Evolution records"
                    print(
                        f"{label}: candidates={evolution.get('candidates', 0)}, "
                        f"evaluations={evolution.get('evaluations', 0)}, "
                        f"gates={evolution.get('gate_decisions', 0)}"
                    )
            elif parsed.verbose:
                print(
                    "科研历史：未初始化"
                    if language == "zh"
                    else "History: not initialized"
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
                freshness = ""
                if result.get("dag_current") is False:
                    freshness = "（需要刷新）" if language == "zh" else " (stale)"
                print(f"{label}: {result['dag_html']}{freshness}")
                if result.get("dag_current") is False:
                    refresh_label = "刷新" if language == "zh" else "Refresh"
                    refresh_command = _contextual_action(
                        result["dag_refresh_command"], parsed.workspace
                    )
                    print(f"{refresh_label}: {refresh_command}")
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
            for warning in payload.get("warnings") or []:
                if warning.get("code") == "generated_view_stale":
                    continue
                warning_label = "提示" if language == "zh" else "Warning"
                print(
                    f"{warning_label}: {warning.get('detail', '')}",
                    file=sys.stderr,
                )
                if warning.get("remediation"):
                    repair_label = "处理" if language == "zh" else "Action"
                    print(
                        f"{repair_label}: {warning['remediation']}",
                        file=sys.stderr,
                    )
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
            adapters = (payload.get("extensions") or {}).get("research_adapters", {})
            if adapters:
                print(
                    "Research adapters: "
                    + ", ".join(adapters.get("available") or ["none"])
                    + f" (API {adapters.get('api_version')})"
                )
            print(f"Quick start: {payload['quickstart']}")
        return 0
    if parsed.command == "init":
        from .onboarding import (
            WorkspaceInitError,
            create_workspace,
            resolve_workspace_root,
        )

        try:
            workspace = resolve_workspace_root(parsed.directory)
            if _validated_workspace_file(workspace, "research.yaml") is not None:
                raise WorkspaceInitError(
                    "refusing to refresh an existing Research VCS workspace; "
                    "use `xscientist setup --force` to refresh managed runtime files"
                )
            payload = create_workspace(
                parsed.directory,
                profile=parsed.profile,
                provider=parsed.provider,
                model=parsed.model,
                force=parsed.force,
            )
        except (OSError, ValueError, WorkspaceInitError) as exc:
            if parsed.as_json:
                print(
                    json.dumps(
                        _safe_json_error_payload(
                            exc,
                            schema="xscientist.init.v1",
                        ),
                        ensure_ascii=False,
                    )
                )
            else:
                print(f"xscientist init: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(
                json.dumps(
                    _safe_public_json_payload(
                        {"schema": "xscientist.init.v1", **payload}
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
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
        from .onboarding import (
            WORKSPACE_FILES,
            WorkspaceInitError,
            create_workspace,
            ensure_workspace_gitignore,
            refresh_generated_bfts_profile,
            resolve_workspace_root,
        )
        from .provider_config import (
            DEFAULT_MODELS,
            ProviderConfigError,
            normalize_provider_name,
        )

        # JSON is both the output and input contract for automation. It must not
        # depend on TTY state or fall through to visible/hidden prompts.
        parsed.non_interactive = bool(parsed.non_interactive or parsed.as_json)
        setup_snapshot: dict[str, object] | None = None
        setup_mutated = False
        try:
            task_profile = TASK_PROFILES[parsed.task]
            selected_provider = (
                normalize_provider_name(parsed.provider) if parsed.provider else None
            )
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
            workspace = resolve_workspace_root(parsed.directory)
            workspace_was_present = workspace.exists()
            if (workspace / ".git").is_symlink():
                raise WorkspaceInitError("setup refuses a symlinked .git control path")
            unsafe_managed_directories = [
                relative
                for relative in _SETUP_TRANSACTION_DIRS
                if relative != ".git"
                and (
                    (workspace / relative).is_symlink()
                    or (
                        (workspace / relative).exists()
                        and not (workspace / relative).is_dir()
                    )
                )
            ]
            if unsafe_managed_directories:
                raise WorkspaceInitError(
                    "setup refused unsafe managed directories: "
                    + ", ".join(sorted(unsafe_managed_directories))
                )
            research_config_present = (
                _validated_workspace_file(workspace, "research.yaml") is not None
            )
            existing_research_config = False
            existing_status: dict[str, object] | None = None
            existing_managed_paths = {
                relative
                for relative in WORKSPACE_FILES
                if (workspace / relative).exists()
                or (workspace / relative).is_symlink()
            }
            vcs_enabled = parsed.task != "service" and not parsed.no_research_vcs
            _had_existing_git, staged_paths, tracked_changes = _workspace_git_changes(
                workspace,
                required=vcs_enabled,
            )
            git_control_path = workspace / ".git"
            if (
                git_control_path.exists() or git_control_path.is_symlink()
            ) and not _had_existing_git:
                raise WorkspaceInitError(
                    "setup refused an existing .git control path that is not an "
                    "exact-root Git worktree"
                )
            if staged_paths:
                raise WorkspaceInitError(
                    "setup refused because the existing Git index contains staged "
                    "work: " + ", ".join(sorted(staged_paths))
                )
            tracked_managed_changes = sorted(
                tracked_changes
                & (
                    set(WORKSPACE_FILES)
                    | {"question.md", "research.yaml", ".xscientist/README.md"}
                )
            )
            if tracked_managed_changes:
                raise WorkspaceInitError(
                    "setup refused to absorb existing edits to managed files: "
                    + ", ".join(tracked_managed_changes)
                )
            if research_config_present:
                from .research_git import ResearchGitError, repository_status

                if not _had_existing_git or not _workspace_git_head_contains(
                    workspace, "research.yaml"
                ):
                    raise WorkspaceInitError(
                        "research.yaml is not owned by the exact-root Git HEAD; "
                        "refusing Research VCS refresh privileges"
                    )
                try:
                    existing_status = repository_status(workspace)
                except (OSError, ResearchGitError, ValueError) as exc:
                    raise WorkspaceInitError(
                        f"existing Research VCS preflight failed: {exc}"
                    ) from exc
                if not existing_status.get("head") or not existing_status.get(
                    "last_checkpoint"
                ):
                    raise WorkspaceInitError(
                        "research.yaml has no verifiable Research VCS checkpoint lineage"
                    )
                existing_research_config = True
            if existing_research_config and not vcs_enabled:
                raise WorkspaceInitError(
                    "setup cannot refresh an existing Research VCS workspace while "
                    "local research checkpoints are disabled"
                )
            if existing_research_config:
                assert existing_status is not None
                native_stage = (existing_status.get("research_stage") or {}).get(
                    "paths"
                ) or []
                if native_stage:
                    raise WorkspaceInitError(
                        "setup refused because the native Research VCS stage "
                        "contains pending work: "
                        + ", ".join(sorted(str(path) for path in native_stage))
                    )
            existing_question = _validated_workspace_file(workspace, "question.md")
            if existing_question is not None and not existing_research_config:
                raise WorkspaceInitError(
                    "setup found question.md without Research VCS provenance; "
                    "initialize or move that scientific source explicitly before "
                    "retrying"
                )
            # A Research VCS config is provenance for generated runtime files.
            # Before that exists, --force must preserve every pre-existing name:
            # it is unknown project data, not an XScientist template to replace.
            protected_paths = set(existing_managed_paths)
            if existing_research_config:
                protected_paths.add("topic.md")
            checkpoint_protected_paths = protected_paths - {".gitignore"}
            if (
                not existing_research_config
                and not parsed.skip_credentials
                and task_profile["provider_required"]
                and protected_paths & {".xscientist/providers.json", "bfts_config.yaml"}
            ):
                raise WorkspaceInitError(
                    "setup refuses to reconfigure pre-existing provider runtime "
                    "files without Research VCS provenance"
                )
            env_file_relative = _setup_env_file_relative(workspace)
            setup_snapshot = _setup_workspace_snapshot(
                workspace,
                extra_files=tuple(dict.fromkeys((env_file_relative, ".env"))),
            )
            setup_mutated = True
            onboarding = create_workspace(
                parsed.directory,
                profile=parsed.profile,
                provider=workspace_template_provider,
                model=selected_model,
                force=parsed.force,
                task=parsed.task,
                capabilities=task_profile["capabilities"],
                provider_required=bool(task_profile["provider_required"]),
                preserve_paths=protected_paths,
            )
            bfts_profile_refreshed = False
            profile_explicit = any(
                argument == "--profile" or argument.startswith("--profile=")
                for argument in raw_argv
            )
            if existing_research_config and profile_explicit:
                bfts_profile_refreshed = refresh_generated_bfts_profile(
                    workspace,
                    parsed.profile,
                )
                if bfts_profile_refreshed:
                    checkpoint_protected_paths.discard("bfts_config.yaml")
            _record_setup_post_state(workspace, setup_snapshot)
            repository = None
            repository_created = False
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
                        repository = ResearchRepository(workspace)
                        research_vcs = {
                            "required": True,
                            "initialized": True,
                            "checkpoint_id": (
                                (existing_status or {}).get("last_checkpoint") or {}
                            ).get("checkpoint_id"),
                            "reason": "existing local research repository reused",
                        }
                    else:
                        question = (workspace / "topic.md").read_text(encoding="utf-8")
                        repository = _initialize_cli_research_repository(
                            workspace,
                            name=workspace.name,
                            question=question,
                            commit=False,
                        )
                        repository_created = True
                        research_vcs = {
                            "required": True,
                            "initialized": True,
                            "checkpoint_id": None,
                            "reason": "local research repository prepared",
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
            ensure_workspace_gitignore(workspace)
            _record_setup_post_state(workspace, setup_snapshot)
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
                        base_url_value=parsed.base_url,
                        non_interactive=parsed.non_interactive,
                    )
                except ProviderConfigError as exc:
                    provider_setup = {
                        "ok": False,
                        "skipped": True,
                        "reason": str(exc),
                    }
                if provider_setup.get("ok"):
                    checkpoint_protected_paths.discard(".xscientist/providers.json")
                    checkpoint_protected_paths.discard("bfts_config.yaml")
            elif not task_profile["provider_required"]:
                provider_setup = {
                    "ok": True,
                    "skipped": True,
                    "reason": "provider is not required for the selected task",
                }
            _record_setup_post_state(workspace, setup_snapshot)
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
            _record_setup_post_state(workspace, setup_snapshot)
            if repository is not None:
                from .research_git import ResearchGitError

                try:
                    runtime_candidates = (
                        set(str(path) for path in onboarding.get("files", []))
                        - checkpoint_protected_paths
                    )
                    if repository_created:
                        runtime_candidates.update(
                            {
                                ".gitignore",
                                ".xscientist/README.md",
                                "question.md",
                                "research.yaml",
                            }
                        )
                    eligible_paths = set(
                        repository.status().get("eligible_changes") or []
                    )
                    generated_runtime_paths = sorted(
                        runtime_candidates & eligible_paths
                    )
                    if not generated_runtime_paths and repository_created:
                        raise ResearchGitError(
                            "Research VCS initialization produced no source changes"
                        )
                    if generated_runtime_paths:
                        repository.stage(generated_runtime_paths)
                        setup_checkpoint = repository.commit(
                            stage="init" if repository_created else "setup",
                            subject=(
                                f"initialize {workspace.name}"
                                if repository_created
                                else "refresh the generated research runtime"
                            ),
                            summary=(
                                "Initialize only confirmed local research sources and "
                                "runtime files."
                                if repository_created
                                else "Checkpoint only runtime files generated or "
                                "refreshed by this setup invocation."
                            ),
                            status="completed",
                            staged_only=True,
                        )
                        if not setup_checkpoint.committed:
                            raise ResearchGitError(
                                "generated runtime files were not checkpointed"
                            )
                        research_vcs["checkpoint_id"] = setup_checkpoint.checkpoint_id
                        research_vcs["setup_checkpoint_id"] = (
                            setup_checkpoint.checkpoint_id
                        )
                        research_vcs["reason"] = (
                            "local research repository initialized"
                            if repository_created
                            else "existing local research repository refreshed"
                        )
                        research_check = (report.get("checks") or {}).get(
                            "research_vcs"
                        )
                        if isinstance(research_check, dict):
                            research_check["head"] = setup_checkpoint.commit
                            research_check["checkpoint_id"] = (
                                setup_checkpoint.checkpoint_id
                            )
                except (OSError, ResearchGitError, ValueError) as exc:
                    raise WorkspaceInitError(
                        f"managed runtime checkpoint failed: {exc}"
                    ) from exc
            _commit_setup_environment(setup_snapshot)
            setup_mutated = False
        except BaseException as exc:
            if setup_mutated and setup_snapshot is not None:
                try:
                    _rollback_setup_workspace(workspace, setup_snapshot)
                except Exception:
                    exc = WorkspaceInitError(
                        "setup failed and could not fully restore its managed paths"
                    )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if parsed.as_json:
                print(
                    json.dumps(
                        _safe_json_error_payload(
                            exc,
                            schema="xscientist.setup.v1",
                        )
                    )
                )
            else:
                print(f"xscientist setup: {exc}", file=sys.stderr)
            return 2
        payload = {
            "schema": "xscientist.setup.v1",
            "ok": report["ok"],
            "workspace_created": not workspace_was_present,
            "workspace_action": (
                "refreshed"
                if existing_research_config
                else ("created" if not workspace_was_present else "initialized")
            ),
            "workspace": onboarding["workspace"],
            "task": parsed.task,
            "provider_configuration": provider_setup,
            "research_vcs": research_vcs,
            "doctor": report,
            "next_actions": report["next_actions"],
            "host_paths_disclosed": False,
        }
        if parsed.as_json:
            print(
                json.dumps(
                    _safe_public_json_payload(payload),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            action_label = {
                "created": "Created",
                "initialized": "Initialized",
                "refreshed": "Refreshed",
            }[str(payload["workspace_action"])]
            print(f"{action_label} XScientist workspace: {payload['workspace']}")
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
