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
        ("research", "Record scientific progress in a local Git repository."),
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
    provider_list.add_argument("--workspace", default=".")
    provider_list.add_argument("--json", action="store_true", dest="as_json")
    provider_add = provider_subparsers.add_parser(
        "add",
        help="Configure a provider; secrets are prompted with hidden input.",
    )
    provider_add.add_argument("name", choices=_PROVIDER_CHOICES)
    provider_add.add_argument("--workspace", default=".")
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
    provider_activate.add_argument("--workspace", default=".")
    provider_activate.add_argument("--no-update-bfts", action="store_true")
    provider_activate.add_argument("--json", action="store_true", dest="as_json")
    provider_remove = provider_subparsers.add_parser(
        "remove",
        help="Remove provider metadata; stored credentials are left untouched.",
    )
    provider_remove.add_argument("name", choices=_PROVIDER_CHOICES)
    provider_remove.add_argument("--workspace", default=".")
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


def _load_json_file(path: str) -> object:
    with open(Path(path).expanduser().resolve(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _installation_info() -> dict[str, object]:
    import importlib.util
    import os

    from ai_scientist.apps.preflight import CORE_PACKAGES
    from ai_scientist.utils.auth_session import validate_session

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
        profile = "full+service"
    elif runtime_ready:
        profile = "full"
    elif service_ready:
        profile = "core+service"
    else:
        profile = "core"
    authenticated, auth_status, _session = validate_session()
    return {
        "name": "xscientist",
        "version": __version__,
        "installation_profile": profile,
        "research_runtime_ready": runtime_ready,
        "service_ready": service_ready,
        "missing_research_packages": missing_runtime,
        "missing_service_packages": missing_service,
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
        "quickstart": "xscientist init my-research",
        "active_provider": os.environ.get("AI_SCIENTIST_ACTIVE_PROVIDER"),
        "default_model": os.environ.get("AI_SCIENTIST_DEFAULT_MODEL"),
    }


def _bootstrap_workspace_environment() -> dict[str, object]:
    import os

    explicit = str(os.environ.get("XSCIENTIST_WORKSPACE") or "").strip()
    candidate = Path(explicit).expanduser() if explicit else Path.cwd()
    if not (candidate / ".xscientist" / "providers.json").is_file():
        return {"loaded": False}
    from .provider_config import ProviderConfigError, load_workspace_environment

    try:
        return load_workspace_environment(candidate)
    except ProviderConfigError as exc:
        return {"loaded": False, "error": str(exc)}


def _run_provider(parsed: argparse.Namespace) -> int:
    import getpass
    import os

    from .provider_config import (
        DEFAULT_ENV_FILE,
        DEFAULT_MODELS,
        PROVIDER_FIELDS,
        ProviderConfigError,
        activate_provider,
        configured_field_value,
        load_provider_config,
        provider_statuses,
        read_env_file,
        remove_provider,
        resolve_env_file,
        save_provider,
        update_bfts_models,
        update_env_file,
        validate_provider_model,
    )

    workspace = Path(parsed.workspace).expanduser().resolve()
    try:
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
                    if row["error"]:
                        suffix += f"; error: {row['error']}"
                    print(
                        f"{marker} {row['provider']}: {state}; model: {model}{suffix}"
                    )
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

        config = load_provider_config(workspace, missing_ok=False)
        existing = config.get("providers", {}).get(parsed.name, {})
        existing_model = (
            str(existing.get("model") or "") if isinstance(existing, dict) else ""
        )
        model = parsed.model or existing_model or DEFAULT_MODELS.get(parsed.name)
        if not model:
            if parsed.non_interactive or not sys.stdin.isatty():
                raise ProviderConfigError(
                    f"--model is required for provider {parsed.name!r} in non-interactive mode"
                )
            model = input(f"Model ID for {parsed.name}: ").strip()
        provider, model = validate_provider_model(parsed.name, model)
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
            if parsed.non_interactive or not sys.stdin.isatty():
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
            activate=not parsed.no_activate,
        )
        bfts_updated = False
        if not parsed.no_update_bfts and saved.get("active_provider") == provider:
            bfts_updated = update_bfts_models(workspace / "bfts_config.yaml", model)
        payload = {
            "ok": True,
            "workspace": ".",
            "provider": provider,
            "model": model,
            "active": saved.get("active_provider") == provider,
            "env_file": env_path.relative_to(workspace).as_posix(),
            "credentials_written": sorted(updates),
            "bfts_updated": bfts_updated,
        }
        if parsed.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Configured provider: {provider}")
            print(f"Default model: {model}")
            if updates:
                print(
                    "Credentials saved securely to: "
                    f"{env_path.relative_to(workspace).as_posix()}"
                )
            else:
                print("Credentials: using existing environment or local env file")
            print(f"Active: {payload['active']}")
            print(f"BFTS config updated: {bfts_updated}")
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


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in _DELEGATES:
        return _DELEGATES[raw_argv[0]](raw_argv[1:])

    workspace_state = (
        {"loaded": False}
        if raw_argv and raw_argv[0] in {"provider", "init"}
        else _bootstrap_workspace_environment()
    )
    if workspace_state.get("error") and (
        not raw_argv or raw_argv[0] not in {"provider", "info", "init"}
    ):
        print(
            f"XScientist workspace configuration error: {workspace_state['error']}",
            file=sys.stderr,
        )
        return 2

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
