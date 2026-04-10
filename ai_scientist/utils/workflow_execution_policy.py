from __future__ import annotations

"""Execution-policy helpers for workflow-specific planning and budget discipline."""

from dataclasses import dataclass, replace
from typing import Any

from ai_scientist.utils.workflow_modes import (
    WorkflowModeSpec,
    resolve_workflow_mode,
)


@dataclass(frozen=True)
class WorkflowExecutionPolicy:
    workflow_mode: str
    workflow_label: str
    execution_style: str
    evidence_pressure: str
    quality_fallback_policy: str
    allow_auto_improvement_fallback: bool
    reject_on_auto_improvement_fallback: bool
    budget: dict[str, int]
    acceptance_rules: tuple[str, ...]
    logging_requirements: tuple[str, ...]
    registry_expectations: tuple[str, ...]


def build_workflow_execution_policy(
    workflow_mode: str | WorkflowModeSpec | None,
    *,
    submission_mode: bool = False,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
    target_venue: str | None = None,
) -> WorkflowExecutionPolicy:
    spec = (
        workflow_mode
        if isinstance(workflow_mode, WorkflowModeSpec)
        else resolve_workflow_mode(
            workflow_mode,
            submission_mode=submission_mode,
            breakthrough_mode=breakthrough_mode,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
    )
    policies: dict[str, WorkflowExecutionPolicy] = {
        "classic_pipeline": WorkflowExecutionPolicy(
            workflow_mode="classic_pipeline",
            workflow_label=spec.label,
            execution_style="stable_template_flow",
            evidence_pressure="standard",
            quality_fallback_policy="allowed",
            allow_auto_improvement_fallback=True,
            reject_on_auto_improvement_fallback=False,
            budget={
                "max_steps": 12,
                "max_wallclock_minutes": 90,
                "max_retry_per_task": 2,
            },
            acceptance_rules=(
                "At least one task should produce a claim-linked result artifact.",
                "Keep one strong baseline and one primary metric traceable through the final narrative.",
            ),
            logging_requirements=(
                "Record baseline, dataset, and primary metric for each task.",
                "Preserve the first successful storyline candidate for later writeup.",
            ),
            registry_expectations=(
                "Mark entered_storyline only for claim-linked runs.",
                "Do not drop failed tasks; keep error_type and error_message populated.",
            ),
        ),
        "agentic_tree": WorkflowExecutionPolicy(
            workflow_mode="agentic_tree",
            workflow_label=spec.label,
            execution_style="exploratory_branch_search",
            evidence_pressure="exploratory",
            quality_fallback_policy="allowed",
            allow_auto_improvement_fallback=True,
            reject_on_auto_improvement_fallback=False,
            budget={
                "max_steps": 18,
                "max_wallclock_minutes": 150,
                "max_retry_per_task": 3,
            },
            acceptance_rules=(
                "Preserve exploratory branches, including negative or redirected outcomes.",
                "At least one branch should explicitly probe novelty beyond the baseline template path.",
            ),
            logging_requirements=(
                "Capture branch intent or search rationale in the experiment config.",
                "Keep redirected and dead-end branches in the registry instead of collapsing them away.",
            ),
            registry_expectations=(
                "Use config.branch_mode or config.goal to describe branch purpose.",
                "Record whether a branch strengthened, weakened, or redirected the hypothesis.",
            ),
        ),
        "program_driven": WorkflowExecutionPolicy(
            workflow_mode="program_driven",
            workflow_label=spec.label,
            execution_style="budgeted_program_execution",
            evidence_pressure="disciplined",
            quality_fallback_policy="disallowed",
            allow_auto_improvement_fallback=False,
            reject_on_auto_improvement_fallback=True,
            budget={
                "max_steps": 8,
                "max_wallclock_minutes": 75,
                "max_retry_per_task": 1,
            },
            acceptance_rules=(
                "Every task must declare success_criterion and stop_condition before execution.",
                "Do not exceed retry budget without explicitly revising the research program.",
            ),
            logging_requirements=(
                "Record budget exhaustion explicitly instead of treating it as a generic failure.",
                "Keep acceptance_checks and revision requirements visible to downstream stages.",
            ),
            registry_expectations=(
                "Persist budget_status and acceptance_checks for each task.",
                "Surface whether a failure consumed the planned retry budget or can still be resumed.",
            ),
        ),
        "writing_studio": WorkflowExecutionPolicy(
            workflow_mode="writing_studio",
            workflow_label=spec.label,
            execution_style="evidence_packaging_first",
            evidence_pressure="writing_heavy",
            quality_fallback_policy="disallowed",
            allow_auto_improvement_fallback=False,
            reject_on_auto_improvement_fallback=True,
            budget={
                "max_steps": 10,
                "max_wallclock_minutes": 80,
                "max_retry_per_task": 2,
            },
            acceptance_rules=(
                "Each surviving claim needs a figure or table path before final polish.",
                "Prefer evidence packaging and caption readiness over extra exploratory runs.",
            ),
            logging_requirements=(
                "Track which runs are intended for figures, tables, or narrative analysis.",
                "Persist caption and experiment-analysis expectations alongside the task goal.",
            ),
            registry_expectations=(
                "Expose whether a run is writeup-ready or still evidence-only.",
                "Keep figure/table dependencies obvious for downstream writing audits.",
            ),
        ),
        "review_board": WorkflowExecutionPolicy(
            workflow_mode="review_board",
            workflow_label=spec.label,
            execution_style="review_hardened_evidence_flow",
            evidence_pressure="review_hardened",
            quality_fallback_policy="disallowed",
            allow_auto_improvement_fallback=False,
            reject_on_auto_improvement_fallback=True,
            budget={
                "max_steps": 9,
                "max_wallclock_minutes": 70,
                "max_retry_per_task": 1,
            },
            acceptance_rules=(
                "Only storyline tasks with reproducibility-ready evidence can enter the final narrative.",
                "Multi-role review blockers must map back to a claim, figure, or section owner.",
            ),
            logging_requirements=(
                "Track which experiment artifacts are intended to survive review-board scrutiny.",
                "Record blocker-facing evidence gaps so repair loops know what to fix next.",
            ),
            registry_expectations=(
                "Persist review-facing acceptance_checks for each task.",
                "Mark blocked evidence clearly when the run cannot support reproducibility or reviewer scrutiny.",
            ),
        ),
        "multi_agent_board": WorkflowExecutionPolicy(
            workflow_mode="multi_agent_board",
            workflow_label=spec.label,
            execution_style="multi_agent_submission_board",
            evidence_pressure="adversarial_submission_grade",
            quality_fallback_policy="disallowed",
            allow_auto_improvement_fallback=False,
            reject_on_auto_improvement_fallback=True,
            budget={
                "max_steps": 10,
                "max_wallclock_minutes": 90,
                "max_retry_per_task": 1,
            },
            acceptance_rules=(
                "Every lead claim must bind to a baseline delta, experiment record, figure/table path, and citation plan before submission.",
                "Only keep branches whose evidence survives both review-board and hostile-critic scrutiny.",
                "A hostile critic blocker must either trigger repair or block submission readiness.",
            ),
            logging_requirements=(
                "Record owner, dependency, and kill criteria for each task in the research plan.",
                "Persist keep/discard/crash outcomes and baseline-comparability notes for reviewer-facing runs.",
                "Keep hostile-critic findings as immutable read-only artifacts with evidence anchors.",
            ),
            registry_expectations=(
                "Persist branch outcome, normalized budget envelope, and baseline linkage for each run.",
                "Mark which artifacts are intended for claim survival, figure packaging, or reviewer rebuttal.",
                "Keep critic blockers and repair ownership visible to downstream manager boards.",
            ),
        ),
    }
    policy = policies[spec.name]
    if high_quality_mode and policy.allow_auto_improvement_fallback:
        policy = replace(policy, quality_fallback_policy="discouraged")
    if submission_mode or target_venue in {"journal", "nature"}:
        policy = replace(
            policy,
            quality_fallback_policy="disallowed",
            allow_auto_improvement_fallback=False,
            reject_on_auto_improvement_fallback=True,
        )
    return policy


def policy_snapshot(policy: WorkflowExecutionPolicy) -> dict[str, Any]:
    return {
        "policy_name": policy.workflow_mode,
        "policy_label": policy.workflow_label,
        "execution_style": policy.execution_style,
        "evidence_pressure": policy.evidence_pressure,
        "quality_fallback_policy": policy.quality_fallback_policy,
        "allow_auto_improvement_fallback": policy.allow_auto_improvement_fallback,
        "reject_on_auto_improvement_fallback": (
            policy.reject_on_auto_improvement_fallback
        ),
        "budget": dict(policy.budget),
        "acceptance_rules": list(policy.acceptance_rules),
        "logging_requirements": list(policy.logging_requirements),
        "registry_expectations": list(policy.registry_expectations),
    }
