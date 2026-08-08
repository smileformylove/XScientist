"""Unified, path-safe first-run diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._version import __version__
from .dependency_profiles import TASK_PROFILES, resolve_task_capabilities
from .git_support import inspect_git_backend
from .provider_config import (
    PROVIDER_NAMES,
    ProviderConfigError,
    discover_workspace_root,
    load_provider_config,
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
    selected_provider = str(provider or "").strip().lower() or None
    if selected_provider is not None and selected_provider not in PROVIDER_NAMES:
        raise ProviderConfigError(f"unknown provider {selected_provider!r}")
    task_profile = TASK_PROFILES.get(str(task or "").strip().lower())
    if (
        selected_provider is None
        and config is not None
        and bool(task_profile and task_profile["provider_required"])
    ):
        selected_provider = str(config.get("active_provider") or "").strip() or None

    resolver_kwargs = {"provider": selected_provider}
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
                for row in provider_statuses(root, **status_kwargs)
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
    if not workspace_ready:
        actions.append("xscientist setup my-research")
    if capabilities["missing_modules"]:
        actions.append(str(capabilities["install_command"]))
    if provider_required and not selected_provider:
        actions.append("xscientist provider add <provider>")
    elif provider_required and not provider_ready and selected_provider:
        actions.append(f"xscientist provider add {selected_provider}")
    if auth_required and not authenticated:
        actions.append("xscientist auth login --user <your-name>")
    if not git["ok"] and git.get("install_hint"):
        actions.append(str(git["install_hint"]))
    if research_vcs_required and not research_vcs_ready and root is not None:
        actions.append("xscientist research init .")

    runtime_results: list[dict[str, Any]] = []
    runtime_ok: bool | None = None
    if deep and runtime_preflight and root is not None and config is not None:
        import shutil

        from ai_scientist.apps.preflight import check_bfts_config
        from .provider_config import load_workspace_environment

        environment = load_workspace_environment(root)
        if environment.get("error"):
            runtime_results.append(
                {
                    "label": "Workspace environment",
                    "ok": False,
                    "severity": "error",
                    "detail": str(environment["error"]),
                }
            )
        else:
            for result in check_bfts_config(str(root / "bfts_config.yaml")):
                runtime_results.append(
                    {
                        "label": result.label,
                        "ok": result.ok,
                        "severity": result.severity,
                        "detail": result.detail,
                    }
                )
            for command, severity, purpose in (
                ("pdflatex", "error", "paper compilation"),
                ("chktex", "warning", "LaTeX linting"),
            ):
                available = shutil.which(command) is not None
                runtime_results.append(
                    {
                        "label": command,
                        "ok": available,
                        "severity": severity,
                        "detail": (
                            f"available for {purpose}"
                            if available
                            else f"required for {purpose} but not found on PATH"
                        ),
                    }
                )
        runtime_ok = not any(
            not item["ok"] and item["severity"] == "error" for item in runtime_results
        )
        if not runtime_ok:
            actions.extend(
                [
                    "docker build -f Dockerfile.executor "
                    f"-t xscientist-exec:{__version__} .",
                    "xscientist preflight --strict --bfts-config bfts_config.yaml",
                ]
            )
            if not any(
                item["ok"] for item in runtime_results if item["label"] == "pdflatex"
            ):
                actions.append("install a TeX distribution that provides pdflatex")
    elif runtime_preflight:
        actions.append(f"xscientist doctor --task {capabilities['task']} --deep")
    actions = list(dict.fromkeys(actions))

    checks = {
        "workspace": {
            "ok": workspace_ready,
            "configured": config is not None,
            "error": workspace_error or None,
        },
        "git": {
            "ok": bool(git["ok"]),
            "backend": git["backend"],
            "version": git["version"],
            "capabilities": git["capabilities"],
            "errors": git["errors"],
        },
        "research_vcs": {
            "ok": research_vcs_ready,
            "required": research_vcs_required,
            "initialized": bool(
                root is not None and (root / "research.yaml").is_file()
            ),
            "error": research_vcs_error or None,
            **research_vcs_summary,
        },
        "capabilities": capabilities,
        "provider": {
            "ok": provider_ready,
            "required": provider_required,
            "name": selected_provider,
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
        },
        "auth": {
            "ok": authenticated or not auth_required,
            "authenticated": authenticated,
            "required": auth_required,
            "status": auth_status,
        },
        "runtime": {
            "ok": runtime_ok,
            "required": runtime_preflight,
            "checked": bool(deep and runtime_preflight),
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
        "host_paths_disclosed": False,
    }
    # Preflight providers and OS errors can embed host-local paths.  Apply the
    # same recursive privacy filter used by persisted research artifacts before
    # any structured diagnostic reaches stdout or an API caller.
    return redact_sensitive_payload(report)


__all__ = ["diagnose"]
