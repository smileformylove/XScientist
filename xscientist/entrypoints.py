from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence


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
    try:
        parser.parse_args(args)  # type: ignore[attr-defined]
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _call_main(module_name: str, argv: Sequence[str] | None = None) -> int:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional runtime dependency"
        print(
            f"XScientist workflow dependency {missing!r} is not installed. "
            'Install the full runtime with `pip install "xscientist[full]"`.',
            file=sys.stderr,
        )
        return 2
    main_fn: Callable[..., object] = getattr(module, "main")
    if argv is None:
        result = main_fn()
    else:
        original = sys.argv
        sys.argv = [module_name, *argv]
        try:
            result = main_fn()
        finally:
            sys.argv = original
    return int(result or 0)


def project_main(argv: Sequence[str] | None = None) -> int:
    help_result = _workflow_help_main("project", argv)
    if help_result is not None:
        return help_result
    return _call_main("ai_scientist.apps.project", argv)


def batch_main(argv: Sequence[str] | None = None) -> int:
    help_result = _workflow_help_main("batch", argv)
    if help_result is not None:
        return help_result
    return _call_main("ai_scientist.apps.batch", argv)


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
    return _call_main("xscientist.research_cli", argv)
