from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .client import XScientist
    from .models import CommandResult, ProjectRequest, ServiceSettings
    from .research_lifecycle import ResearchLifecycle
    from .research_vcs import ResearchRepository

_MODEL_EXPORTS = {"CommandResult", "ProjectRequest", "ServiceSettings"}


def __getattr__(name: str) -> Any:
    """Load SDK exports only when callers access them."""

    if name == "XScientist":
        from .client import XScientist

        value = XScientist
    elif name in _MODEL_EXPORTS:
        from . import models

        value = getattr(models, name)
    elif name == "ResearchRepository":
        from .research_vcs import ResearchRepository

        value = ResearchRepository
    elif name == "ResearchLifecycle":
        from .research_lifecycle import ResearchLifecycle

        value = ResearchLifecycle
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to introspection without importing them."""

    return sorted(set(globals()) | set(__all__))


def create_app(*args, **kwargs):
    """Create the optional FastAPI application without importing it eagerly."""

    from .service import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "CommandResult",
    "ProjectRequest",
    "ResearchLifecycle",
    "ResearchRepository",
    "ServiceSettings",
    "XScientist",
    "__version__",
    "create_app",
]
