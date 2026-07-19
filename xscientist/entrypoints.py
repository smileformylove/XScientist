from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence


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
    return _call_main("ai_scientist.apps.project", argv)


def batch_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("continuous_paper_generator", argv)


def daemon_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("continuous_research_daemon", argv)


def manager_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("research_manager", argv)


def ara_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("run_ara_fork", argv)


def auth_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("auth_cli", argv)


def feedback_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("feedback_cli", argv)


def validate_main(argv: Sequence[str] | None = None) -> int:
    return _call_main("validate_repo", argv)
