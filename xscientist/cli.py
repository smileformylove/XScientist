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
        help="workspace root (default: discover from the current directory)",
    )
    provider_list.add_argument("--json", action="store_true", dest="as_json")
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
        default="zhipu",
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
        description=(
            "Create or reuse one workspace, validate every prerequisite, and run "
            "a traceable automated research project."
        ),
    )
    parser.add_argument("directory", help="workspace and Research VCS root")
    parser.add_argument(
        "--question", required=True, help="one concrete research question"
    )
    parser.add_argument(
        "--autopilot",
        choices=["balanced", "discovery", "publication"],
        default="balanced",
    )
    parser.add_argument(
        "--provider",
        choices=_PROVIDER_CHOICES,
        default=None,
        help="provider for a new workspace; existing workspaces reuse their active provider",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--profile", choices=["default", "deep"], default="default")
    parser.add_argument("--user", default=None, help="local research actor name")
    parser.add_argument(
        "--data-dir", default=None, help="read-only empirical input directory"
    )
    parser.add_argument(
        "--allow-synthetic-data",
        action="store_true",
        help="explicitly permit exploratory synthetic/computational evidence",
    )
    parser.add_argument("--max-project-tokens", type=int, default=None)
    parser.add_argument("--max-project-hours", type=float, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument(
        "--build-executor",
        action="store_true",
        help="explicitly build the generated isolated Docker image before diagnosis",
    )
    parser.add_argument("--skip-credentials", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="stop after workspace, login, provider, and deep runtime validation",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


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


def _installation_info() -> dict[str, object]:
    import importlib.util
    import os

    from ai_scientist.apps.preflight import CORE_PACKAGES
    from ai_scientist.utils.auth_session import validate_session
    from .dependency_profiles import installation_command, missing_provider_modules

    runtime_modules = sorted(CORE_PACKAGES)
    service_modules = ["fastapi", "pydantic", "uvicorn"]
    missing_runtime = [
        name for name in runtime_modules if importlib.util.find_spec(name) is None
    ]
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
    active_provider = str(os.environ.get("AI_SCIENTIST_ACTIVE_PROVIDER") or "").strip()
    missing_provider_clients = (
        missing_provider_modules(active_provider) if active_provider else []
    )
    provider_client_ready = bool(active_provider) and not missing_provider_clients
    recommended_install = (
        installation_command(active_provider)
        if active_provider
        else 'python -m pip install "xscientist[research,<provider-extra>]"'
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
        "missing_provider_clients": missing_provider_clients,
        "recommended_install": recommended_install,
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
        "quickstart": "xscientist setup my-research --task research",
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
        model = input(f"Model ID for {name}: ").strip()
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
    from .provider_config import (
        ProviderConfigError,
        activate_provider,
        discover_workspace_root,
        load_provider_config,
        provider_statuses,
        remove_provider,
        update_bfts_models,
    )

    try:
        if parsed.workspace is None:
            workspace = discover_workspace_root()
            if workspace is None:
                raise ProviderConfigError(
                    "provider configuration not found in the current directory or its parents; "
                    "run `xscientist init` first or pass --workspace"
                )
        else:
            workspace = Path(parsed.workspace).expanduser().resolve()
        if parsed.provider_command == "list":
            rows = provider_statuses(workspace)
            payload = {
                "workspace": ".",
                "providers": rows,
            }
            if parsed.as_json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print("Workspace: .")
                for row in rows:
                    marker = "*" if row["active"] else " "
                    if row["ready"]:
                        state = "ready"
                    elif row["configured"]:
                        state = "configured, not ready"
                    else:
                        state = "not configured"
                    model = row["model"] or "-"
                    missing = ", ".join(row["missing"])
                    suffix = f"; missing: {missing}" if missing else ""
                    missing_clients = ", ".join(row["missing_client_modules"])
                    if missing_clients:
                        suffix += f"; missing clients: {missing_clients}"
                    if row["error"]:
                        suffix += f"; error: {row['error']}"
                    print(
                        f"{marker} {row['provider']}: {state}; model: {model}{suffix}"
                    )
                    if row["configured"] and missing_clients:
                        print(f"    Install: {row['install_command']}")
            return 0

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


def _run_start(parsed: argparse.Namespace) -> int:
    """Orchestrate the safe first-run path without hiding scientific gates."""

    import contextlib
    import io
    import os
    import subprocess

    from ai_scientist.utils.atomic_io import atomic_write_text
    from ai_scientist.utils.auth_session import create_session, validate_session
    from .diagnostics import diagnose
    from .onboarding import WorkspaceInitError, create_workspace
    from .provider_config import (
        DEFAULT_MODELS,
        ProviderConfigError,
        load_provider_config,
        load_workspace_environment,
    )
    from .research_git import ResearchGitError, repository_status
    from .research_vcs import ResearchRepository

    question = str(parsed.question or "").strip()
    if not question:
        print("xscientist start: --question cannot be empty", file=sys.stderr)
        return 2
    if parsed.data_dir and parsed.allow_synthetic_data:
        print(
            "xscientist start: --data-dir and --allow-synthetic-data are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if (
        not parsed.prepare_only
        and not parsed.data_dir
        and not parsed.allow_synthetic_data
    ):
        print(
            "xscientist start: choose --data-dir PATH for empirical evidence or "
            "--allow-synthetic-data for an explicitly exploratory study",
            file=sys.stderr,
        )
        return 2
    for label, value in (
        ("--max-project-tokens", parsed.max_project_tokens),
        ("--max-project-hours", parsed.max_project_hours),
        ("--max-cost-usd", parsed.max_cost_usd),
    ):
        if value is not None and float(value) <= 0:
            print(
                f"xscientist start: {label} must be greater than zero", file=sys.stderr
            )
            return 2

    workspace = Path(parsed.directory).expanduser().resolve()
    phases: dict[str, object] = {}
    try:
        config_exists = (workspace / ".xscientist" / "providers.json").is_file()
        existing_config = (
            load_provider_config(workspace, missing_ok=False) if config_exists else {}
        )
        selected_provider = str(
            parsed.provider or existing_config.get("active_provider") or "zhipu"
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
        if not selected_model:
            if parsed.non_interactive or not sys.stdin.isatty():
                raise ProviderConfigError(
                    f"--model is required for provider {selected_provider!r}"
                )
            selected_model = input(f"Model ID for {selected_provider}: ").strip()
        if not config_exists:
            create_workspace(
                workspace,
                profile=parsed.profile,
                provider=selected_provider,
                model=selected_model,
                force=parsed.force,
                task="research",
                capabilities=("research", "ml", "pdf-layout"),
                provider_required=True,
            )
        phases["workspace"] = {"ok": True, "created": not config_exists}

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
                actor=parsed.user or "xscientist",
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

        authenticated, auth_status, session = validate_session()
        if not authenticated:
            username = str(parsed.user or "").strip()
            if not username and not parsed.non_interactive and sys.stdin.isatty():
                username = input("Local research actor name: ").strip()
            if username:
                session = create_session(username=username)
                authenticated, auth_status = True, "ok"
        phases["auth"] = {
            "ok": authenticated,
            "status": auth_status,
            "user": (session or {}).get("username") if authenticated else None,
        }

        load_result = load_workspace_environment(workspace)
        if load_result.get("error"):
            raise ProviderConfigError(str(load_result["error"]))

        if parsed.build_executor:
            source_root = Path(__file__).resolve().parents[1]
            local_source = (source_root / "pyproject.toml").is_file() and (
                source_root / "xscientist"
            ).is_dir()
            build_context = source_root if local_source else workspace
            command = [
                "docker",
                "build",
                "-f",
                str(workspace / "Dockerfile.executor"),
                "-t",
                f"xscientist-exec:{__version__}",
            ]
            if local_source:
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source_root,
                    text=True,
                    capture_output=True,
                    check=False,
                ).stdout.strip()
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=source_root,
                    text=True,
                    capture_output=True,
                    check=False,
                ).stdout.strip()
                if revision and dirty:
                    revision += "-dirty"
                command.extend(
                    [
                        "--build-arg",
                        "XSCIENTIST_INSTALL_MODE=local",
                        "--build-arg",
                        "XSCIENTIST_SOURCE_REVISION=" + (revision or "local-source"),
                    ]
                )
            command.append(str(build_context))
            completed = subprocess.run(
                command,
                cwd=build_context,
                text=True,
                capture_output=bool(parsed.as_json),
                check=False,
            )
            phases["executor_build"] = {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
            }

        report = diagnose(
            workspace,
            task="research",
            provider=selected_provider,
            deep=True,
        )
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
            print("Resolve these items, then rerun the same command:")
            for action in report["next_actions"]:
                print(f"  {action}")
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
    if raw_argv and raw_argv[0] in _DELEGATES:
        return _DELEGATES[raw_argv[0]](raw_argv[1:])

    workspace_state = (
        {"loaded": False}
        if raw_argv
        and raw_argv[0]
        in {"provider", "init", "start", "setup", "doctor", "capability"}
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
    if parsed.command == "start":
        return _run_start(parsed)
    if parsed.command == "info":
        payload = _installation_info()
        if parsed.as_json:
            print(json.dumps(payload, indent=2))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
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
                print(f"  {index}. {step}")
        return 0
    if parsed.command == "setup":
        from .diagnostics import diagnose
        from .dependency_profiles import TASK_PROFILES
        from .onboarding import WorkspaceInitError, create_workspace
        from .provider_config import DEFAULT_MODELS, ProviderConfigError

        try:
            selected_model = parsed.model
            task_profile = TASK_PROFILES[parsed.task]
            if not selected_model and not DEFAULT_MODELS.get(parsed.provider):
                if parsed.non_interactive or not sys.stdin.isatty():
                    raise ProviderConfigError(
                        f"--model is required for provider {parsed.provider!r} "
                        "in non-interactive mode"
                    )
                selected_model = input(f"Model ID for {parsed.provider}: ").strip()
                if not selected_model:
                    raise ProviderConfigError("model ID is required")
            onboarding = create_workspace(
                parsed.directory,
                profile=parsed.profile,
                provider=parsed.provider,
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
                        name=parsed.provider,
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
                    parsed.provider if task_profile["provider_required"] else None
                ),
                deep=parsed.deep,
            )
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
            print(f"Configuration ready: {report['configuration_ready']}")
            print(f"Runtime ready: {report['runtime_ready']}")
            print(f"Research VCS: {research_vcs['initialized']}")
            if provider_setup.get("reason"):
                print(f"Provider setup: {provider_setup['reason']}")
            if payload["next_actions"]:
                print("Next actions:")
                for index, action in enumerate(payload["next_actions"], start=1):
                    print(f"  {index}. {action}")
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
        except (OSError, ProviderConfigError, ValueError) as exc:
            print(f"xscientist doctor: {exc}", file=sys.stderr)
            return 2
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(
                f"XScientist configuration ready for {payload['task']}: "
                f"{payload['configuration_ready']}"
            )
            print(f"Deep runtime ready: {payload['runtime_ready']}")
            for name, check in payload["checks"].items():
                print(f"{name:<14} {check['ok']}")
            if payload["next_actions"]:
                print("Next actions:")
                for action in payload["next_actions"]:
                    print(f"  {action}")
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
