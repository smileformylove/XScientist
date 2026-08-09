from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .client import XScientist
    from .models import CommandResult, ProjectRequest, ServiceSettings
    from .research_evolution import ResearchEvolution
    from .research_lifecycle import ResearchLifecycle
    from .research_closure import audit_research_closure
    from .research_dag import build_research_dag, export_research_dag
    from .research_discovery import (
        assess_generalization,
        build_discovery_contract,
        discovery_contract_template,
        save_discovery_contract,
        save_generalization_assessment,
    )
    from .research_journey import build_research_guide, start_guided_research
    from .research_tools import ingest_tool_evidence
    from .research_vcs import ResearchRepository

_MODEL_EXPORTS = {"CommandResult", "ProjectRequest", "ServiceSettings"}
_DISCOVERY_EXPORTS = {
    "assess_generalization",
    "build_discovery_contract",
    "discovery_contract_template",
    "save_discovery_contract",
    "save_generalization_assessment",
}


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
    elif name == "ResearchEvolution":
        from .research_evolution import ResearchEvolution

        value = ResearchEvolution
    elif name == "audit_research_closure":
        from .research_closure import audit_research_closure

        value = audit_research_closure
    elif name in {"build_research_dag", "export_research_dag"}:
        from . import research_dag

        value = getattr(research_dag, name)
    elif name in _DISCOVERY_EXPORTS:
        from . import research_discovery

        value = getattr(research_discovery, name)
    elif name in {"build_research_guide", "start_guided_research"}:
        from . import research_journey

        value = getattr(research_journey, name)
    elif name == "ingest_tool_evidence":
        from .research_tools import ingest_tool_evidence

        value = ingest_tool_evidence
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
    "ResearchEvolution",
    "ResearchLifecycle",
    "ResearchRepository",
    "ServiceSettings",
    "XScientist",
    "audit_research_closure",
    "assess_generalization",
    "build_discovery_contract",
    "build_research_dag",
    "build_research_guide",
    "discovery_contract_template",
    "export_research_dag",
    "start_guided_research",
    "ingest_tool_evidence",
    "save_discovery_contract",
    "save_generalization_assessment",
    "__version__",
    "create_app",
]
