from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

_BOOTSTRAPPED_EXECUTOR_WORKSPACE: str | None = None
_PROJECT_BFTS_CONFIG_EXPLICIT = False


def _executor_workspace_was_bootstrapped() -> bool:
    current = str(os.environ.get("XSCIENTIST_WORKSPACE") or "").strip()
    return bool(
        _BOOTSTRAPPED_EXECUTOR_WORKSPACE and current == _BOOTSTRAPPED_EXECUTOR_WORKSPACE
    )


def _project_bfts_config_was_explicit() -> bool:
    return _PROJECT_BFTS_CONFIG_EXPLICIT


def _rebind_bootstrapped_executor_workspace(workspace: str | Path | None) -> None:
    global _BOOTSTRAPPED_EXECUTOR_WORKSPACE

    if not _executor_workspace_was_bootstrapped():
        return
    if workspace is None:
        os.environ.pop("XSCIENTIST_WORKSPACE", None)
        _BOOTSTRAPPED_EXECUTOR_WORKSPACE = None
        return
    resolved = str(Path(workspace).expanduser().resolve())
    os.environ["XSCIENTIST_WORKSPACE"] = resolved
    _BOOTSTRAPPED_EXECUTOR_WORKSPACE = resolved


def _bootstrap_workspace_environment() -> str | None:
    global _BOOTSTRAPPED_EXECUTOR_WORKSPACE

    current = str(os.environ.get("XSCIENTIST_WORKSPACE") or "").strip()
    previously_bootstrapped = bool(
        _BOOTSTRAPPED_EXECUTOR_WORKSPACE and current == _BOOTSTRAPPED_EXECUTOR_WORKSPACE
    )
    if current and not previously_bootstrapped:
        _BOOTSTRAPPED_EXECUTOR_WORKSPACE = None
    explicit = current if current and not previously_bootstrapped else ""
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
        if previously_bootstrapped:
            os.environ.pop("XSCIENTIST_WORKSPACE", None)
            _BOOTSTRAPPED_EXECUTOR_WORKSPACE = None
        return None
    from .provider_config import ProviderConfigError, load_workspace_environment

    try:
        state = load_workspace_environment(candidate)
    except ProviderConfigError as exc:
        return str(exc)
    # Provider environment and executor identity must come from the same
    # discovered workspace, including when project outputs live elsewhere.
    # This mirrors the process-wide provider variables loaded just above.
    if not explicit:
        resolved = str(candidate.resolve())
        os.environ["XSCIENTIST_WORKSPACE"] = resolved
        _BOOTSTRAPPED_EXECUTOR_WORKSPACE = resolved
    return str(state.get("error") or "") or None


def _workflow_help_main(
    workflow: str,
    argv: Sequence[str] | None,
) -> int | None:
    """Render workflow help without importing the optional full runtime."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg in {"-h", "--help"} for arg in args):
        return None

    from ai_scientist.config.paths import resolve_output_path
    from ai_scientist.utils.runtime_bootstrap import resolve_writing_profile_env
    from ai_scientist.utils.workflow_modes import list_workflow_modes
    from ai_scientist.writing_prompt_profiles import (
        DEFAULT_WRITING_PROFILE,
        list_writing_profiles,
    )

    default_writing_profile = resolve_writing_profile_env(
        invalid_profile_logger=lambda exc, raw: print(
            "⚠️  忽略无效的 AI_SCIENTIST_WRITING_PROFILE="
            f"{raw!r}，回退为 {DEFAULT_WRITING_PROFILE}"
        )
    )
    parser_module = importlib.import_module(f"ai_scientist.apps.{workflow}_cli")
    parser_factory: Callable[..., object] = getattr(parser_module, "build_parser")
    common = {
        "default_writing_profile": default_writing_profile,
        "writing_profiles": list_writing_profiles(),
        "workflow_modes": list_workflow_modes(),
    }
    if workflow == "project":
        parser = parser_factory(
            default_output_root=str(resolve_output_path()),
            **common,
        )
    else:
        parser = parser_factory(
            default_research_dir=str(resolve_output_path()),
            **common,
        )
    parser.prog = f"xscientist {workflow}"
    try:
        parser.parse_args(args)  # type: ignore[attr-defined]
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _call_main(
    module_name: str,
    argv: Sequence[str] | None = None,
    *,
    bootstrap_workspace: bool = True,
) -> int:
    if bootstrap_workspace:
        workspace_error = _bootstrap_workspace_environment()
        if workspace_error:
            print(
                f"XScientist workspace configuration error: {workspace_error}",
                file=sys.stderr,
            )
            return 2
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional runtime dependency"
        base_install_hint = 'python -m pip install "xscientist[research]"'
        active_provider = str(
            os.environ.get("AI_SCIENTIST_ACTIVE_PROVIDER") or ""
        ).strip()
        provider_install_hint: str | None = None
        if active_provider:
            from .dependency_profiles import installation_command

            try:
                provider_install_hint = installation_command(active_provider)
            except ValueError:
                provider_install_hint = None
        provider_hint = (
            " For the selected provider client, the combined installation is "
            f"`{provider_install_hint}`."
            if provider_install_hint and provider_install_hint != base_install_hint
            else ""
        )
        print(
            f"XScientist workflow dependency {missing!r} is not installed. "
            f"Install the base research runtime with `{base_install_hint}`."
            f"{provider_hint}",
            file=sys.stderr,
        )
        return 2
    main_fn: Callable[..., object] = getattr(module, "main")
    try:
        if argv is None:
            result = main_fn()
        else:
            original = sys.argv
            command = module_name.rsplit(".", 1)[-1].removesuffix("_cli")
            sys.argv = [f"xscientist {command}", *argv]
            try:
                result = main_fn()
            finally:
                sys.argv = original
    except (OSError, RuntimeError, ValueError) as exc:
        command = module_name.rsplit(".", 1)[-1].removesuffix("_cli")
        try:
            sys.stdout.flush()
        except OSError:
            pass
        print(
            f"XScientist {command} stopped: {exc}",
            file=sys.stderr,
        )
        return 2
    return int(result or 0)


def project_main(argv: Sequence[str] | None = None) -> int:
    global _PROJECT_BFTS_CONFIG_EXPLICIT

    invocation_args = list(sys.argv[1:] if argv is None else argv)
    previous_explicit = _PROJECT_BFTS_CONFIG_EXPLICIT
    _PROJECT_BFTS_CONFIG_EXPLICIT = any(
        item == "--bfts-config" or item.startswith("--bfts-config=")
        for item in invocation_args
    )
    try:
        workspace_error = _bootstrap_workspace_environment()
        if workspace_error:
            print(
                f"XScientist workspace configuration error: {workspace_error}",
                file=sys.stderr,
            )
            return 2
        help_result = _workflow_help_main("project", argv)
        if help_result is not None:
            return help_result
        return _call_main("ai_scientist.apps.project", argv, bootstrap_workspace=False)
    finally:
        _PROJECT_BFTS_CONFIG_EXPLICIT = previous_explicit


def batch_main(argv: Sequence[str] | None = None) -> int:
    workspace_error = _bootstrap_workspace_environment()
    if workspace_error:
        print(
            f"XScientist workspace configuration error: {workspace_error}",
            file=sys.stderr,
        )
        return 2
    help_result = _workflow_help_main("batch", argv)
    if help_result is not None:
        return help_result
    return _call_main("ai_scientist.apps.batch", argv, bootstrap_workspace=False)


def daemon_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.daemon", argv)


def manager_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.manager", argv)


def ara_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.ara", argv)


def auth_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.auth", argv)


def feedback_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.feedback", argv)


def validate_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.validate", argv)


def bfts_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.bfts", argv)


def zhipu_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.zhipu", argv)


def preflight_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("ai_scientist.apps.preflight", argv)


def research_main(argv: Sequence[str] | None = None) -> int:
    from .research_cli import main

    return main(argv, prog="xscientist research")


def evolution_main(argv: Sequence[str] | None = None) -> int:
    from .evolution_cli import main

    return main(argv)


def git_main(argv: Sequence[str] | None = None) -> int:
    """Expose Research VCS with familiar Git-like command names."""

    from .research_cli import main

    return main(argv, prog="xscientist git")
