"""Unified, path-safe first-run diagnostics."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .dependency_profiles import TASK_PROFILES, resolve_task_capabilities
from .git_support import inspect_git_backend
from .provider_config import (
    PROVIDER_NAMES,
    ProviderConfigError,
    discover_provider_models,
    discover_workspace_root,
    load_provider_config,
    normalize_provider_name,
    provider_statuses,
)


def diagnose(
    workspace: str | Path | None = None,
    *,
    task: str = "research",
    provider: str | None = None,
    deep: bool = False,
    find_spec: Callable[[str], object | None] | None = None,
) -> dict[str, Any]:
    """Check one requested workflow without exposing credentials or host paths."""

    from ai_scientist.utils.auth_session import validate_session
    from ai_scientist.utils.privacy import redact_sensitive_payload

    root = (
        Path(workspace).expanduser().resolve()
        if workspace is not None
        else discover_workspace_root()
    )
    workspace_error = ""
    config: dict[str, Any] | None = None
    if root is not None:
        try:
            config = load_provider_config(root, missing_ok=False)
        except (OSError, ProviderConfigError) as exc:
            workspace_error = str(exc)
    selected_provider = (
        normalize_provider_name(provider) if str(provider or "").strip() else None
    )
    if selected_provider is not None and selected_provider not in PROVIDER_NAMES:
        raise ProviderConfigError(f"unknown provider {selected_provider!r}")
    task_profile = TASK_PROFILES.get(str(task or "").strip().lower())
    if (
        selected_provider is None
        and config is not None
        and bool(task_profile and task_profile["provider_required"])
    ):
        selected_provider = str(config.get("active_provider") or "").strip() or None

    detected_local_models = (
        []
        if selected_provider
        or not bool(task_profile and task_profile["provider_required"])
        else discover_provider_models("ollama", timeout=0.2)
    )
    suggested_provider = "ollama" if detected_local_models else None
    capability_provider = (
        selected_provider or suggested_provider
        if bool(task_profile and task_profile["provider_required"])
        else None
    )
    resolver_kwargs = {"provider": capability_provider}
    if find_spec is not None:
        resolver_kwargs["find_spec"] = find_spec
    capabilities = resolve_task_capabilities(task, **resolver_kwargs)
    git = inspect_git_backend()
    authenticated, auth_status, _session = validate_session()

    provider_row: dict[str, Any] | None = None
    if root is not None and config is not None and selected_provider:
        status_kwargs = {}
        if find_spec is not None:
            status_kwargs["find_spec"] = find_spec
        provider_row = next(
            (
                row
                for row in provider_statuses(
                    root,
                    probe_local=True,
                    **status_kwargs,
                )
                if row["provider"] == selected_provider
            ),
            None,
        )
    provider_required = bool(capabilities["provider_required"])
    auth_required = bool(capabilities["auth_required"])
    runtime_preflight = bool(capabilities["runtime_preflight"])
    provider_ready = (
        bool(provider_row and provider_row["ready"]) if provider_required else True
    )
    workspace_ready = root is not None and config is not None and not workspace_error
    research_vcs_required = str(task or "").strip().lower() != "service"
    research_vcs_ready = False
    research_vcs_error = ""
    research_vcs_summary: dict[str, Any] = {
        "branch": None,
        "head": None,
        "checkpoint_id": None,
    }
    if root is not None and (root / "research.yaml").is_file():
        try:
            from .research_git import ResearchGitError, repository_status

            status = repository_status(root)
            research_vcs_ready = True
            research_vcs_summary = {
                "branch": status.get("branch"),
                "head": status.get("head"),
                "checkpoint_id": (status.get("last_checkpoint") or {}).get(
                    "checkpoint_id"
                ),
            }
        except (OSError, ResearchGitError, ValueError) as exc:
            research_vcs_error = str(exc)
    elif research_vcs_required:
        research_vcs_error = "local Research VCS is not initialized"
    else:
        research_vcs_ready = True

    actions: list[str] = []
    remediations: list[dict[str, Any]] = []

    def add_remediation(
        code: str,
        command: str,
        detail: str,
        *,
        severity: str = "error",
    ) -> None:
        if command not in actions:
            actions.append(command)
        if not any(item["code"] == code for item in remediations):
            remediations.append(
                {
                    "code": code,
                    "command": command,
                    "detail": detail,
                    "severity": severity,
                }
            )

    if not workspace_ready:
        setup_command = "xscientist setup my-research"
        if suggested_provider and detected_local_models:
            setup_command += " --provider ollama --model " + shlex.quote(
                detected_local_models[0]
            )
        add_remediation(
            "workspace_not_configured",
            setup_command,
            "Create a workspace before running this task.",
        )
    if capabilities["missing_modules"]:
        add_remediation(
            "capability_modules_missing",
            str(capabilities["install_command"]),
            "Install the exact optional modules required by the selected task.",
        )
    if provider_required and not selected_provider and workspace_ready:
        add_remediation(
            "provider_not_selected",
            (
                "xscientist provider add ollama"
                if suggested_provider == "ollama"
                else "xscientist provider add --help"
            ),
            (
                "Configure the detected local Ollama provider."
                if suggested_provider == "ollama"
                else "Choose and configure one model provider."
            ),
        )
    elif provider_required and not provider_ready and selected_provider:
        local_probe = (provider_row or {}).get("local_probe") or {}
        if local_probe.get("checked") and not local_probe.get("service_reachable"):
            add_remediation(
                "local_provider_unreachable",
                "ollama serve",
                "Start the local Ollama service, then run Doctor again.",
            )
        elif local_probe.get("checked") and not local_probe.get("model_available"):
            model = str((provider_row or {}).get("model") or "<model>")
            model_name = model.split("/", 1)[-1]
            add_remediation(
                "local_model_missing",
                f"ollama pull {model_name}",
                "Install the model selected by this workspace.",
            )
        else:
            add_remediation(
                "provider_not_ready",
                f"xscientist provider add {selected_provider}",
                "Configure credentials and the client for the selected provider.",
            )
    if auth_required and not authenticated:
        add_remediation(
            "research_identity_missing",
            "xscientist auth login",
            "Create the local actor identity used for accountable research history.",
        )
    if not git["ok"] and git.get("install_hint"):
        add_remediation(
            "git_backend_unavailable",
            str(git["install_hint"]),
            "Install a compatible local Git backend.",
        )
    if research_vcs_required and not research_vcs_ready and root is not None:
        add_remediation(
            "research_vcs_not_initialized",
            "xscientist research init .",
            "Initialize local scientific history in this workspace.",
        )

    runtime_results: list[dict[str, Any]] = []
    runtime_ok: bool | None = None
    if deep and runtime_preflight and root is not None and config is not None:
        import shutil

        from ai_scientist.apps.preflight import check_bfts_config
        from .provider_config import load_workspace_environment

        paper_tools_required = str(task).strip().lower() in {
            "paper",
            "ml-study",
        }
        environment = load_workspace_environment(root)
        if environment.get("error"):
            runtime_results.append(
                {
                    "code": "runtime_workspace_environment",
                    "label": "Workspace environment",
                    "ok": False,
                    "severity": "error",
                    "detail": str(environment["error"]),
                }
            )
        else:
            for result in check_bfts_config(
                str(root / "bfts_config.yaml"), workspace=root
            ):
                code = "runtime_" + re.sub(
                    r"[^a-z0-9]+", "_", str(result.label).lower()
                ).strip("_")
                runtime_results.append(
                    {
                        "code": code,
                        "label": result.label,
                        "ok": result.ok,
                        "severity": result.severity,
                        "detail": result.detail,
                    }
                )
            for command, severity, purpose in (
                (
                    "pdflatex",
                    "error" if paper_tools_required else "warning",
                    "paper compilation",
                ),
                ("chktex", "warning", "LaTeX linting"),
            ):
                available = shutil.which(command) is not None
                runtime_results.append(
                    {
                        "code": "runtime_" + command,
                        "label": command,
                        "ok": available,
                        "severity": severity,
                        "detail": (
                            f"available for {purpose}"
                            if available
                            else (
                                f"required for {purpose} but not found on PATH"
                                if severity == "error"
                                else f"optional for {purpose} but not found on PATH"
                            )
                        ),
                    }
                )
        runtime_ok = not any(
            not item["ok"] and item["severity"] == "error" for item in runtime_results
        )
        if not runtime_ok:
            failed_labels = {
                str(item["label"])
                for item in runtime_results
                if not item["ok"] and item["severity"] == "error"
            }
            if "Experiment isolation" in failed_labels:
                isolation = next(
                    (
                        item
                        for item in runtime_results
                        if item["label"] == "Experiment isolation" and not item["ok"]
                    ),
                    {},
                )
                if (
                    "docker executable not found"
                    in str(isolation.get("detail") or "").lower()
                ):
                    add_remediation(
                        "docker_cli_missing",
                        "https://docs.docker.com/get-started/get-docker/",
                        "Install and start Docker before preparing the executor.",
                    )
                else:
                    add_remediation(
                        "executor_image_unavailable",
                        "xscientist executor prepare --workspace .",
                        "Prepare the exact isolated executor selected by the workspace.",
                    )
            add_remediation(
                "runtime_preflight_failed",
                "xscientist preflight --strict --bfts-config bfts_config.yaml",
                "Re-run the detailed runtime probe after resolving blocking checks.",
            )
            if paper_tools_required and not any(
                item["ok"] for item in runtime_results if item["label"] == "pdflatex"
            ):
                add_remediation(
                    "paper_compiler_missing",
                    "install a TeX distribution that provides pdflatex",
                    "Publication output requires a local LaTeX compiler.",
                )
    elif runtime_preflight:
        add_remediation(
            "runtime_not_checked",
            f"xscientist doctor --task {capabilities['task']} --deep",
            "Probe the selected model and isolated executor before paid work.",
            severity="info",
        )

    checks = {
        "workspace": {
            "code": "workspace",
            "ok": workspace_ready,
            "configured": config is not None,
            "error": workspace_error or None,
        },
        "git": {
            "code": "git_backend",
            "ok": bool(git["ok"]),
            "backend": git["backend"],
            "version": git["version"],
            "capabilities": git["capabilities"],
            "errors": git["errors"],
        },
        "research_vcs": {
            "code": "research_vcs",
            "ok": research_vcs_ready,
            "required": research_vcs_required,
            "initialized": bool(
                root is not None and (root / "research.yaml").is_file()
            ),
            "error": research_vcs_error or None,
            **research_vcs_summary,
        },
        "capabilities": {
            "code": "capabilities",
            # Keep every doctor row renderable through the same public
            # ``ok`` field.  ``ready`` remains the richer capability-contract
            # spelling for existing JSON consumers.
            "ok": bool(capabilities["ready"]),
            **capabilities,
        },
        "provider": {
            "code": "provider",
            "ok": provider_ready,
            "required": provider_required,
            "name": selected_provider,
            "suggested": suggested_provider,
            "detected_local_models": detected_local_models,
            "configured": bool(provider_row and provider_row["configured"]),
            "credentials_available": bool(
                provider_row and provider_row["credentials_available"]
            ),
            "client_available": bool(provider_row and provider_row["client_available"]),
            "missing": list(provider_row["missing"] if provider_row else []),
            "missing_client_modules": list(
                provider_row["missing_client_modules"] if provider_row else []
            ),
            "error": str(provider_row["error"] if provider_row else "") or None,
            "local_probe": dict(
                provider_row.get("local_probe") if provider_row else {}
            ),
        },
        "auth": {
            "code": "research_identity",
            "ok": authenticated or not auth_required,
            "authenticated": authenticated,
            "required": auth_required,
            "status": auth_status,
        },
        "runtime": {
            "code": "runtime",
            "ok": runtime_ok,
            "required": runtime_preflight,
            "checked": bool(deep and runtime_preflight),
            "paper_tools_required": str(task).strip().lower() in {"paper", "ml-study"},
            "results": runtime_results,
        },
    }
    configuration_ready = all(
        (
            checks["workspace"]["ok"],
            checks["git"]["ok"],
            checks["research_vcs"]["ok"],
            checks["capabilities"]["ready"],
            checks["provider"]["ok"],
            checks["auth"]["ok"],
        )
    )
    ready = configuration_ready and (runtime_ok is not False)
    report = {
        "schema": "xscientist.doctor.v1",
        "ok": ready,
        "configuration_ready": configuration_ready,
        "runtime_ready": runtime_ok,
        "task": str(capabilities["task"]),
        "workspace": "." if root is not None else None,
        "checks": checks,
        "next_actions": actions,
        "remediations": remediations,
        "issue_codes": [item["code"] for item in remediations],
        "error_codes": [
            item["code"] for item in remediations if item["severity"] == "error"
        ],
        "host_paths_disclosed": False,
    }
    # Preflight providers and OS errors can embed host-local paths.  Apply the
    # same recursive privacy filter used by persisted research artifacts before
    # any structured diagnostic reaches stdout or an API caller.
    return redact_sensitive_payload(report)


__all__ = ["diagnose"]
