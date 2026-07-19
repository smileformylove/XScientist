from __future__ import annotations

from ._version import __version__
from .client import XScientist
from .models import CommandResult, ProjectRequest, ServiceSettings


def create_app(*args, **kwargs):
    """Create the optional FastAPI application without importing it eagerly."""

    from .service import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "CommandResult",
    "ProjectRequest",
    "ServiceSettings",
    "XScientist",
    "__version__",
    "create_app",
]
