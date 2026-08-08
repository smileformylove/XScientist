from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


def _bootstrap_workspace_environment() -> str | None:
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
        return None
    from .provider_config import ProviderConfigError, load_workspace_environment

    try:
        state = load_workspace_environment(candidate)
    except ProviderConfigError as exc:
        return str(exc)
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
        active_provider = str(
            os.environ.get("AI_SCIENTIST_ACTIVE_PROVIDER") or ""
        ).strip()
        if active_provider:
            from .dependency_profiles import installation_command

            install_hint = installation_command(active_provider)
        else:
            install_hint = 'python -m pip install "xscientist[research]"'
        print(
            f"XScientist workflow dependency {missing!r} is not installed. "
            f"Install the selected runtime with `{install_hint}`.",
            file=sys.stderr,
        )
        return 2
    main_fn: Callable[..., object] = getattr(module, "main")
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
    return int(result or 0)


def project_main(argv: Sequence[str] | None = None) -> int:
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
