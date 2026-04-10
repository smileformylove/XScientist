from __future__ import annotations

"""Workflow mode registry for multi-style autonomous research orchestration."""

from argparse import Namespace
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowModeSpec:
    name: str
    label: str
    summary: str
    inspirations: tuple[str, ...]
    stage_sequence: tuple[str, ...]
    pipeline_goal: str


WORKFLOW_MODE_SPECS = {
    "classic_pipeline": WorkflowModeSpec(
        name="classic_pipeline",
        label="Classic Pipeline",
        summary="Stable end-to-end template flow inspired by AI-Scientist style runs.",
        inspirations=("SakanaAI/AI-Scientist",),
        stage_sequence=("ideation", "experiment", "figure", "writeup", "review"),
        pipeline_goal="stable_submission_pipeline",
    ),
    "agentic_tree": WorkflowModeSpec(
        name="agentic_tree",
        label="Agentic Tree Search",
        summary="Open-ended planning and experiment search with stronger exploration pressure.",
        inspirations=("SakanaAI/AI-Scientist-v2",),
        stage_sequence=(
            "ideation",
            "planning",
            "agentic_search",
            "experiment",
            "analysis",
            "writeup",
            "review",
        ),
        pipeline_goal="agentic_discovery_pipeline",
    ),
    "program_driven": WorkflowModeSpec(
        name="program_driven",
        label="Program Driven",
        summary="Research-program-first execution with explicit budgets and acceptance rules.",
        inspirations=("karpathy/autoresearch",),
        stage_sequence=(
            "program",
            "ideation",
            "planning",
            "experiment",
            "writeup",
            "review",
            "revision",
        ),
        pipeline_goal="program_driven_submission",
    ),
    "writing_studio": WorkflowModeSpec(
        name="writing_studio",
        label="Writing Studio",
        summary="Evidence-to-writing workflow with stronger writing audits and polish loops.",
        inspirations=("Leey21/awesome-ai-research-writing",),
        stage_sequence=(
            "ideation",
            "evidence_pack",
            "writeup",
            "writing_audit",
            "review",
        ),
        pipeline_goal="writing_heavy_submission",
    ),
    "review_board": WorkflowModeSpec(
        name="review_board",
        label="Review Board",
        summary="Multi-role review-first hardening for clarity, rigor, reproducibility, and novelty.",
        inspirations=("ResearAI/DeepReviewer-v2",),
        stage_sequence=(
            "ideation",
            "experiment",
            "writeup",
            "multi_role_review",
            "repair",
            "benchmark",
        ),
        pipeline_goal="review_hardened_submission",
    ),
    "multi_agent_board": WorkflowModeSpec(
        name="multi_agent_board",
        label="Multi-Agent Board",
        summary=(
            "Planner/experiment/writing/quality/reviewer/hostile-critic board aimed at"
            " higher-quality, submission-grade papers."
        ),
        inspirations=(
            "SakanaAI/AI-Scientist",
            "SakanaAI/AI-Scientist-v2",
            "karpathy/autoresearch",
            "Leey21/awesome-ai-research-writing",
            "ResearAI/DeepReviewer-v2",
        ),
        stage_sequence=(
            "ideation",
            "planning",
            "evidence_packaging",
            "experiment",
            "writing",
            "quality_gate",
            "review_board",
            "hostile_critic",
            "repair",
            "recheck",
            "submission",
        ),
        pipeline_goal="multi_agent_submission_board",
    ),
}

WORKFLOW_MODE_CHOICES = ("adaptive",) + tuple(WORKFLOW_MODE_SPECS.keys())


def list_workflow_modes() -> list[str]:
    return list(WORKFLOW_MODE_CHOICES)


def resolve_workflow_mode(
    raw_mode: str | None,
    *,
    submission_mode: bool = False,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
    target_venue: str | None = None,
) -> WorkflowModeSpec:
    normalized = str(raw_mode or "adaptive").strip().lower()
    if normalized and normalized != "adaptive":
        if normalized not in WORKFLOW_MODE_SPECS:
            raise ValueError(
                f"Unknown workflow mode {raw_mode!r}. Expected one of {sorted(WORKFLOW_MODE_CHOICES)}"
            )
        return WORKFLOW_MODE_SPECS[normalized]

    if breakthrough_mode:
        return WORKFLOW_MODE_SPECS["agentic_tree"]
    if submission_mode:
        return WORKFLOW_MODE_SPECS["program_driven"]
    if high_quality_mode and target_venue in {"nature", "journal"}:
        return WORKFLOW_MODE_SPECS["review_board"]
    if high_quality_mode:
        return WORKFLOW_MODE_SPECS["writing_studio"]
    return WORKFLOW_MODE_SPECS["classic_pipeline"]


def resolve_template_mode_for_workflow(
    workflow_mode: str | WorkflowModeSpec,
    *,
    submission_mode: bool = False,
) -> tuple[str, str]:
    name = workflow_mode.name if isinstance(workflow_mode, WorkflowModeSpec) else str(workflow_mode)
    if name == "agentic_tree":
        return ("open_ended", "agentic_search")
    if name == "program_driven":
        return ("program_driven", "research_program")
    if name == "writing_studio":
        return ("writing_studio", "writing_skill_pack")
    if name == "review_board":
        return ("review_board", "tool_grounded_review")
    if name == "multi_agent_board":
        return ("program_driven", "multi_agent_board")
    if submission_mode:
        return ("template_first", "high_success_templates")
    return ("open_ended", "adaptive")


def _set_max(args: Namespace, attr: str, value: int | float) -> None:
    if not hasattr(args, attr):
        return
    current = getattr(args, attr)
    if current is None:
        setattr(args, attr, value)
        return
    try:
        if isinstance(value, float):
            setattr(args, attr, max(float(current), value))
        else:
            setattr(args, attr, max(int(current), value))
    except (TypeError, ValueError):
        setattr(args, attr, value)


def _set_if_missing(args: Namespace, attr: str, value) -> None:
    if not hasattr(args, attr):
        return
    current = getattr(args, attr)
    if current in (None, False):
        setattr(args, attr, value)


def apply_workflow_mode_defaults(
    args: Namespace,
    *,
    rank_flag_attr: str,
    candidate_limit_attr: str,
    fallback_flag_attr: str | None = None,
) -> WorkflowModeSpec:
    spec = resolve_workflow_mode(
        getattr(args, "workflow_mode", None),
        submission_mode=bool(getattr(args, "submission_mode", False)),
        breakthrough_mode=bool(getattr(args, "breakthrough_mode", False)),
        high_quality_mode=bool(getattr(args, "high_quality_mode", False)),
        target_venue=getattr(args, "target_venue", None),
    )
    args.workflow_mode = spec.name

    if spec.name == "agentic_tree":
        setattr(args, "high_quality_mode", True)
        _set_max(args, "autonomous_quality_followup_rounds", 1)
        _set_if_missing(args, "review_strategy", "depth")
        _set_max(args, "quality_rewrite_rounds", 2)
    elif spec.name == "program_driven":
        setattr(args, rank_flag_attr, True)
        setattr(args, "high_quality_mode", True)
        setattr(args, "require_quality_gate", True)
        setattr(args, "strict_writing_guardrails", True)
        _set_max(args, "writing_audit_rounds", 1)
        _set_max(args, "guardrail_repair_rounds", 1)
        _set_max(args, "autonomous_quality_followup_rounds", 1)
        _set_max(args, "num_cite_rounds", 20)
        _set_max(args, "writeup_retries", 4)
        if hasattr(args, "quality_preset"):
            args.quality_preset = "publishable"
        _set_if_missing(args, "review_strategy", "depth")
        if getattr(args, candidate_limit_attr, None) is None:
            setattr(
                args,
                candidate_limit_attr,
                5 if getattr(args, "target_venue", None) in {"nature", "journal"} else 3,
            )
        if fallback_flag_attr is not None and hasattr(args, fallback_flag_attr):
            setattr(args, fallback_flag_attr, True)
    elif spec.name == "writing_studio":
        _set_max(args, "num_cite_rounds", 20)
        _set_max(args, "writeup_retries", 4)
        _set_max(args, "writing_audit_rounds", 2)
        _set_max(args, "guardrail_repair_rounds", 2)
        setattr(args, "strict_writing_guardrails", True)
    elif spec.name == "review_board":
        setattr(args, "high_quality_mode", True)
        setattr(args, "require_quality_gate", True)
        setattr(args, "strict_writing_guardrails", True)
        _set_max(args, "review_reflections", 2)
        _set_max(args, "review_ensemble", 3)
        _set_max(args, "review_fewshot", 2)
        if hasattr(args, "review_temperature"):
            try:
                args.review_temperature = min(float(args.review_temperature), 0.65)
            except (TypeError, ValueError):
                args.review_temperature = 0.65
        _set_max(args, "guardrail_repair_rounds", 2)
        _set_if_missing(
            args,
            "review_strategy",
            "nature" if getattr(args, "target_venue", None) in {"nature", "journal"} else "depth",
        )
    elif spec.name == "multi_agent_board":
        setattr(args, "high_quality_mode", True)
        setattr(args, "require_quality_gate", True)
        setattr(args, "strict_writing_guardrails", True)
        _set_max(args, "review_reflections", 2)
        _set_max(args, "review_ensemble", 4)
        _set_max(args, "review_fewshot", 2)
        _set_max(args, "quality_rewrite_rounds", 2)
        _set_max(args, "autonomous_quality_followup_rounds", 1)
        _set_max(args, "writing_audit_rounds", 2)
        _set_max(args, "guardrail_repair_rounds", 2)
        _set_max(args, "num_cite_rounds", 20)
        _set_max(args, "writeup_retries", 4)
        if hasattr(args, "quality_preset"):
            args.quality_preset = "publishable"
        if hasattr(args, "review_temperature"):
            try:
                args.review_temperature = min(float(args.review_temperature), 0.6)
            except (TypeError, ValueError):
                args.review_temperature = 0.6
        _set_if_missing(
            args,
            "review_strategy",
            "nature" if getattr(args, "target_venue", None) in {"nature", "journal"} else "depth",
        )
        if getattr(args, candidate_limit_attr, None) is None:
            setattr(args, candidate_limit_attr, 2)
        if fallback_flag_attr is not None and hasattr(args, fallback_flag_attr):
            setattr(args, fallback_flag_attr, True)
    return spec


def build_workflow_manifest_metadata(
    workflow_mode: str | WorkflowModeSpec,
) -> dict[str, object]:
    spec = (
        workflow_mode
        if isinstance(workflow_mode, WorkflowModeSpec)
        else WORKFLOW_MODE_SPECS[str(workflow_mode)]
    )
    return {
        "workflow_mode": spec.name,
        "workflow_label": spec.label,
        "workflow_summary": spec.summary,
        "workflow_inspirations": list(spec.inspirations),
        "workflow_sequence": list(spec.stage_sequence),
    }


def stage_sequence_for_mode(workflow_mode: str | WorkflowModeSpec) -> list[str]:
    spec = (
        workflow_mode
        if isinstance(workflow_mode, WorkflowModeSpec)
        else WORKFLOW_MODE_SPECS[str(workflow_mode)]
    )
    return list(spec.stage_sequence)
