from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .client import XScientist
    from .models import CommandResult, ProjectRequest, ServiceSettings
    from .research_evolution import ResearchEvolution
    from .research_lifecycle import ResearchLifecycle
    from .research_closure import audit_research_closure, closure_level_summary
    from .process_audit import build_process_summary
    from .evidence_index import build_evidence_index
    from .system_comparison import build_system_comparison
    from .benchmark import (
        benchmark_autoresearch_pilot,
        benchmark_first_run,
        persist_benchmark_report,
        verify_benchmark_report,
    )
    from ai_scientist.utils.arft_coverage import build_arft_coverage, save_arft_coverage
    from .research_dag import build_research_dag, export_research_dag
    from .research_discovery import (
        assess_generalization,
        build_discovery_contract,
        discovery_contract_template,
        save_discovery_contract,
        save_generalization_assessment,
    )
    from .research_journey import build_research_guide, start_guided_research
    from .research_strategy import (
        inspect_claim_depth,
        rank_experiment_candidates,
        research_strategy_template,
        review_research_program,
        save_discriminating_prediction,
        save_evidence_quality_assessment,
        save_hypothesis_portfolio,
        save_mechanism_model,
        save_posterior_update,
        save_transfer_matrix,
        scan_research_anomalies,
    )
    from .opportunity_funnel import (
        build_opportunity_attempt,
        build_opportunity_funnel_summary,
        build_opportunity_grade,
        build_opportunity_judgment,
        build_opportunity_pool,
        build_research_direction,
        inspect_opportunity_funnel,
        normalize_opportunity_candidates,
        rank_opportunity_candidates,
        save_opportunity_allocation,
        save_opportunity_attempt,
        save_opportunity_grade,
        save_opportunity_judgment,
        save_opportunity_pool,
        save_research_direction,
    )
    from .research_rollout import (
        COMPARISON_BOUNDARY_POLICY,
        INDEPENDENCE_ASSURANCE,
        INDEPENDENCE_ATTESTATION_PURPOSE,
        INDEPENDENCE_POLICY,
        ResearchRolloutError,
        assess_tool_swap_compatibility,
        audit_research_rollout,
        build_replication_rubric,
        build_research_rollout,
        build_comparison_boundary,
        build_independence_attestation_payload,
        build_independence_receipt,
        build_strategy_budget_summary,
        build_tool_delegation_trace,
        build_turn_credit_summary,
        evaluate_replication_rollout,
        rollout_producer_actor_ids,
        save_research_rollout,
    )
    from .research_context import (
        build_research_context_snapshot,
        render_research_context_for_prompt,
    )
    from .research_belief import (
        BELIEF_CONTEXT_POLICY,
        BELIEF_CONTEXT_SEMANTICS,
        BeliefContextError,
        audit_belief_context_projection,
        belief_context_issues,
        build_belief_context_projection,
    )
    from .research_tools import ingest_tool_evidence
    from .research_vcs import ResearchRepository
    from .workspace_history import (
        compare_workspace_history,
        inspect_workspace_checkpoint,
        inspect_workspace_history,
        preview_workspace_rollback,
        rollback_workspace_checkpoint,
        save_workspace_checkpoint,
    )

_MODEL_EXPORTS = {"CommandResult", "ProjectRequest", "ServiceSettings"}
_DISCOVERY_EXPORTS = {
    "assess_generalization",
    "build_discovery_contract",
    "discovery_contract_template",
    "save_discovery_contract",
    "save_generalization_assessment",
}
_STRATEGY_EXPORTS = {
    "inspect_claim_depth",
    "rank_experiment_candidates",
    "research_strategy_template",
    "review_research_program",
    "save_discriminating_prediction",
    "save_evidence_quality_assessment",
    "save_hypothesis_portfolio",
    "save_mechanism_model",
    "save_posterior_update",
    "save_transfer_matrix",
    "scan_research_anomalies",
}

_OPPORTUNITY_EXPORTS = {
    "build_opportunity_attempt",
    "build_opportunity_funnel_summary",
    "build_opportunity_grade",
    "build_opportunity_judgment",
    "build_opportunity_pool",
    "build_research_direction",
    "inspect_opportunity_funnel",
    "normalize_opportunity_candidates",
    "rank_opportunity_candidates",
    "save_opportunity_allocation",
    "save_opportunity_attempt",
    "save_opportunity_grade",
    "save_opportunity_judgment",
    "save_opportunity_pool",
    "save_research_direction",
}

_ROLLOUT_EXPORTS = {
    "COMPARISON_BOUNDARY_POLICY",
    "INDEPENDENCE_ASSURANCE",
    "INDEPENDENCE_ATTESTATION_PURPOSE",
    "INDEPENDENCE_POLICY",
    "ResearchRolloutError",
    "assess_tool_swap_compatibility",
    "audit_research_rollout",
    "build_replication_rubric",
    "build_research_rollout",
    "build_comparison_boundary",
    "build_independence_attestation_payload",
    "build_independence_receipt",
    "build_strategy_budget_summary",
    "build_tool_delegation_trace",
    "build_turn_credit_summary",
    "evaluate_replication_rollout",
    "rollout_producer_actor_ids",
    "save_research_rollout",
}

_BELIEF_CONTEXT_EXPORTS = {
    "BELIEF_CONTEXT_POLICY",
    "BELIEF_CONTEXT_SEMANTICS",
    "BeliefContextError",
    "audit_belief_context_projection",
    "belief_context_issues",
    "build_belief_context_projection",
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
    elif name == "closure_level_summary":
        from .research_closure import closure_level_summary

        value = closure_level_summary
    elif name in {"build_arft_coverage", "save_arft_coverage"}:
        from ai_scientist.utils import arft_coverage

        value = getattr(arft_coverage, name)
    elif name == "build_process_summary":
        from .process_audit import build_process_summary

        value = build_process_summary
    elif name == "build_evidence_index":
        from .evidence_index import build_evidence_index

        value = build_evidence_index
    elif name == "build_system_comparison":
        from .system_comparison import build_system_comparison

        value = build_system_comparison
    elif name in {
        "benchmark_autoresearch_pilot",
        "benchmark_first_run",
        "persist_benchmark_report",
        "verify_benchmark_report",
    }:
        from . import benchmark

        value = getattr(benchmark, name)
    elif name in {"build_research_dag", "export_research_dag"}:
        from . import research_dag

        value = getattr(research_dag, name)
    elif name in _DISCOVERY_EXPORTS:
        from . import research_discovery

        value = getattr(research_discovery, name)
    elif name in _STRATEGY_EXPORTS:
        from . import research_strategy

        value = getattr(research_strategy, name)
    elif name in _OPPORTUNITY_EXPORTS:
        from . import opportunity_funnel

        value = getattr(opportunity_funnel, name)
    elif name in _ROLLOUT_EXPORTS:
        from . import research_rollout

        value = getattr(research_rollout, name)
    elif name in {"build_research_guide", "start_guided_research"}:
        from . import research_journey

        value = getattr(research_journey, name)
    elif name in {
        "build_research_context_snapshot",
        "render_research_context_for_prompt",
    }:
        from . import research_context

        value = getattr(research_context, name)
    elif name in _BELIEF_CONTEXT_EXPORTS:
        from . import research_belief

        value = getattr(research_belief, name)
    elif name == "ingest_tool_evidence":
        from .research_tools import ingest_tool_evidence

        value = ingest_tool_evidence
    elif name in {
        "compare_workspace_history",
        "inspect_workspace_checkpoint",
        "inspect_workspace_history",
        "preview_workspace_rollback",
        "rollback_workspace_checkpoint",
        "save_workspace_checkpoint",
    }:
        from . import workspace_history

        value = getattr(workspace_history, name)
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
    "build_arft_coverage",
    "build_process_summary",
    "build_evidence_index",
    "build_system_comparison",
    "benchmark_autoresearch_pilot",
    "benchmark_first_run",
    "closure_level_summary",
    "assess_generalization",
    "build_discovery_contract",
    "build_research_dag",
    "build_research_context_snapshot",
    "build_research_guide",
    "compare_workspace_history",
    "discovery_contract_template",
    "export_research_dag",
    "render_research_context_for_prompt",
    "BELIEF_CONTEXT_POLICY",
    "BELIEF_CONTEXT_SEMANTICS",
    "BeliefContextError",
    "audit_belief_context_projection",
    "belief_context_issues",
    "build_belief_context_projection",
    "start_guided_research",
    "ingest_tool_evidence",
    "inspect_workspace_history",
    "inspect_workspace_checkpoint",
    "inspect_claim_depth",
    "preview_workspace_rollback",
    "rank_experiment_candidates",
    "research_strategy_template",
    "review_research_program",
    "save_discovery_contract",
    "save_discriminating_prediction",
    "save_evidence_quality_assessment",
    "save_generalization_assessment",
    "save_hypothesis_portfolio",
    "save_mechanism_model",
    "save_posterior_update",
    "save_transfer_matrix",
    "save_workspace_checkpoint",
    "save_arft_coverage",
    "persist_benchmark_report",
    "verify_benchmark_report",
    "scan_research_anomalies",
    "build_opportunity_attempt",
    "build_opportunity_funnel_summary",
    "build_opportunity_grade",
    "build_opportunity_judgment",
    "build_opportunity_pool",
    "build_research_direction",
    "inspect_opportunity_funnel",
    "normalize_opportunity_candidates",
    "rank_opportunity_candidates",
    "save_opportunity_allocation",
    "save_opportunity_attempt",
    "save_opportunity_grade",
    "save_opportunity_judgment",
    "save_opportunity_pool",
    "save_research_direction",
    "ResearchRolloutError",
    "COMPARISON_BOUNDARY_POLICY",
    "INDEPENDENCE_ASSURANCE",
    "INDEPENDENCE_ATTESTATION_PURPOSE",
    "INDEPENDENCE_POLICY",
    "assess_tool_swap_compatibility",
    "audit_research_rollout",
    "build_replication_rubric",
    "build_research_rollout",
    "build_comparison_boundary",
    "build_independence_attestation_payload",
    "build_independence_receipt",
    "build_strategy_budget_summary",
    "build_tool_delegation_trace",
    "build_turn_credit_summary",
    "evaluate_replication_rollout",
    "rollout_producer_actor_ids",
    "save_research_rollout",
    "rollback_workspace_checkpoint",
    "__version__",
    "create_app",
]
