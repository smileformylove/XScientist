from __future__ import annotations

"""Workflow-aware source planning helpers for daemon orchestration."""

from dataclasses import dataclass
from typing import Any, Iterable

from ai_scientist.utils.workflow_modes import WORKFLOW_MODE_CHOICES


@dataclass(frozen=True)
class SourceArchetypeSpec:
    name: str
    label: str
    summary: str
    inspirations: tuple[str, ...]
    default_workflow_mode: str
    default_batch_profile: str


@dataclass(frozen=True)
class BatchProfileSpec:
    name: str
    label: str
    summary: str
    batch_goal: str
    default_workflow_mode: str


SOURCE_ARCHETYPE_SPECS: dict[str, SourceArchetypeSpec] = {
    "adaptive": SourceArchetypeSpec(
        name="adaptive",
        label="Adaptive",
        summary="Let the daemon and execution policy decide how this source should be used.",
        inspirations=(),
        default_workflow_mode="adaptive",
        default_batch_profile="adaptive",
    ),
    "template_first": SourceArchetypeSpec(
        name="template_first",
        label="Template First",
        summary="Stable, repeatable source that favors the classic AI-Scientist template path.",
        inspirations=("SakanaAI/AI-Scientist",),
        default_workflow_mode="classic_pipeline",
        default_batch_profile="submission_push",
    ),
    "frontier_exploration": SourceArchetypeSpec(
        name="frontier_exploration",
        label="Frontier Exploration",
        summary="High-variance exploratory source tuned for broader branching and breakthrough search.",
        inspirations=("SakanaAI/AI-Scientist-v2",),
        default_workflow_mode="agentic_tree",
        default_batch_profile="exploration_sprint",
    ),
    "program_guarded": SourceArchetypeSpec(
        name="program_guarded",
        label="Program Guarded",
        summary="Research-program-first source with stronger budget discipline and stop criteria.",
        inspirations=("karpathy/autoresearch",),
        default_workflow_mode="program_driven",
        default_batch_profile="submission_push",
    ),
    "writing_polish": SourceArchetypeSpec(
        name="writing_polish",
        label="Writing Polish",
        summary="Evidence-heavy source designed to convert validated results into polished manuscripts.",
        inspirations=("Leey21/awesome-ai-research-writing",),
        default_workflow_mode="writing_studio",
        default_batch_profile="evidence_pack",
    ),
    "review_hardening": SourceArchetypeSpec(
        name="review_hardening",
        label="Review Hardening",
        summary="Reviewer-facing source focused on hardening clarity, rigor, and reproducibility.",
        inspirations=("ResearAI/DeepReviewer-v2",),
        default_workflow_mode="review_board",
        default_batch_profile="review_hardening",
    ),
    "paper_hardening_board": SourceArchetypeSpec(
        name="paper_hardening_board",
        label="Paper Hardening Board",
        summary="Full multi-agent source for program discipline, evidence packaging, and hostile-critic hardening.",
        inspirations=(
            "SakanaAI/AI-Scientist",
            "SakanaAI/AI-Scientist-v2",
            "karpathy/autoresearch",
            "Leey21/awesome-ai-research-writing",
            "ResearAI/DeepReviewer-v2",
        ),
        default_workflow_mode="multi_agent_board",
        default_batch_profile="paper_hardening",
    ),
}

SOURCE_ARCHETYPE_CHOICES = tuple(SOURCE_ARCHETYPE_SPECS.keys())


BATCH_PROFILE_SPECS: dict[str, BatchProfileSpec] = {
    "adaptive": BatchProfileSpec(
        name="adaptive",
        label="Adaptive",
        summary="Follow the currently active execution policy and source metadata.",
        batch_goal="Keep the source flexible and daemon-directed.",
        default_workflow_mode="adaptive",
    ),
    "discovery_sprint": BatchProfileSpec(
        name="discovery_sprint",
        label="Discovery Sprint",
        summary="Fast, stable discovery loop with enough structure to keep throughput high.",
        batch_goal="Generate and rank ideas quickly without overcommitting the runtime.",
        default_workflow_mode="classic_pipeline",
    ),
    "exploration_sprint": BatchProfileSpec(
        name="exploration_sprint",
        label="Exploration Sprint",
        summary="Increase branching pressure and allow broader exploration before convergence.",
        batch_goal="Search for frontier ideas and experimental branches with higher variance.",
        default_workflow_mode="agentic_tree",
    ),
    "submission_push": BatchProfileSpec(
        name="submission_push",
        label="Submission Push",
        summary="Drive toward submission-grade packaging with tighter budgets and stronger gates.",
        batch_goal="Convert a source into submission-ready experiments and writing artifacts.",
        default_workflow_mode="program_driven",
    ),
    "evidence_pack": BatchProfileSpec(
        name="evidence_pack",
        label="Evidence Pack",
        summary="Package figures, evidence, and writing support for stronger narrative cohesion.",
        batch_goal="Turn experimental evidence into a clear, defensible manuscript package.",
        default_workflow_mode="writing_studio",
    ),
    "review_hardening": BatchProfileSpec(
        name="review_hardening",
        label="Review Hardening",
        summary="Stress-test the paper with reviewer-facing audits, fixes, and reproducibility checks.",
        batch_goal="Surface and repair issues before the next submission-quality pass.",
        default_workflow_mode="review_board",
    ),
    "paper_hardening": BatchProfileSpec(
        name="paper_hardening",
        label="Paper Hardening",
        summary="Run the full multi-agent paper board including hostile critic and repair ownership.",
        batch_goal="Drive a candidate through quality gate, hostile critic, and ownership-aware repair.",
        default_workflow_mode="multi_agent_board",
    ),
}

BATCH_PROFILE_CHOICES = tuple(BATCH_PROFILE_SPECS.keys())


def normalize_source_archetype(raw_value: str | None) -> str:
    value = str(raw_value or "adaptive").strip().lower()
    if value not in SOURCE_ARCHETYPE_SPECS:
        raise ValueError(
            f"Unknown source archetype {raw_value!r}. Expected one of {sorted(SOURCE_ARCHETYPE_CHOICES)}"
        )
    return value


def normalize_batch_profile(raw_value: str | None) -> str:
    value = str(raw_value or "adaptive").strip().lower()
    if value not in BATCH_PROFILE_SPECS:
        raise ValueError(
            f"Unknown batch profile {raw_value!r}. Expected one of {sorted(BATCH_PROFILE_CHOICES)}"
        )
    return value


def normalize_workflow_mode_name(raw_value: str | None) -> str:
    value = str(raw_value or "adaptive").strip().lower()
    if value not in WORKFLOW_MODE_CHOICES:
        raise ValueError(
            f"Unknown workflow mode {raw_value!r}. Expected one of {sorted(WORKFLOW_MODE_CHOICES)}"
        )
    return value


def normalize_workflow_mode_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        raise ValueError("workflow mode list must be a string or a sequence")
    normalized: list[str] = []
    for value in values:
        candidate = normalize_workflow_mode_name(value)
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _as_string_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _resolve_with_daypart(
    source: dict[str, Any], daypart: str, base_name: str
) -> Any:
    daypart_value = source.get(f"{daypart}_{base_name}")
    if daypart_value is not None:
        return daypart_value
    return source.get(base_name)


def infer_source_archetype(source: dict[str, Any]) -> str:
    paper_types = _as_string_list(source.get("paper_types"))
    target_venue = str(source.get("target_venue") or "").strip().lower()
    if source.get("breakthrough_mode"):
        return "frontier_exploration"
    if source.get("submission_mode"):
        if target_venue in {"nature", "journal"} or any(
            paper_type in {"journal", "extended"} for paper_type in paper_types
        ):
            return "program_guarded"
        return "template_first"
    if any(paper_type in {"journal", "extended"} for paper_type in paper_types):
        return "writing_polish"
    return "adaptive"


def infer_batch_profile(
    source: dict[str, Any], *, workflow_mode: str, source_archetype: str
) -> str:
    if workflow_mode == "agentic_tree":
        return "exploration_sprint"
    if workflow_mode == "program_driven":
        return "submission_push"
    if workflow_mode == "writing_studio":
        return "evidence_pack"
    if workflow_mode == "review_board":
        return "review_hardening"
    if workflow_mode == "multi_agent_board":
        return "paper_hardening"
    if source_archetype == "template_first":
        return "submission_push"
    return "discovery_sprint"


def _build_recommended_flags(
    *, resolved_workflow_mode: str, batch_profile: str
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if resolved_workflow_mode and resolved_workflow_mode != "adaptive":
        flags.append({"flag": "--workflow-mode", "value": resolved_workflow_mode})

    if batch_profile == "exploration_sprint":
        flags.extend(
            [
                {"flag": "--breakthrough-mode", "switch": True},
                {"flag": "--quality-rewrite-rounds", "value": 2},
                {"flag": "--autonomous-quality-followup-rounds", "value": 1},
            ]
        )
    elif batch_profile == "submission_push":
        flags.extend(
            [
                {"flag": "--submission-mode", "switch": True},
                {"flag": "--writing-audit-rounds", "value": 1},
                {"flag": "--guardrail-repair-rounds", "value": 1},
            ]
        )
    elif batch_profile == "evidence_pack":
        flags.extend(
            [
                {"flag": "--strict-writing-guardrails", "switch": True},
                {"flag": "--writing-audit-rounds", "value": 2},
                {"flag": "--guardrail-repair-rounds", "value": 2},
            ]
        )
    elif batch_profile == "review_hardening":
        flags.extend(
            [
                {"flag": "--submission-mode", "switch": True},
                {"flag": "--strict-writing-guardrails", "switch": True},
                {"flag": "--review-strategy", "value": "depth"},
                {"flag": "--writing-audit-rounds", "value": 1},
                {"flag": "--guardrail-repair-rounds", "value": 2},
            ]
        )
    elif batch_profile == "paper_hardening":
        flags.extend(
            [
                {"flag": "--submission-mode", "switch": True},
                {"flag": "--strict-writing-guardrails", "switch": True},
                {"flag": "--review-strategy", "value": "depth"},
                {"flag": "--writing-audit-rounds", "value": 2},
                {"flag": "--guardrail-repair-rounds", "value": 2},
                {"flag": "--quality-rewrite-rounds", "value": 2},
            ]
        )
    return flags


def _flag_preview_items(items: Iterable[dict[str, Any]]) -> list[str]:
    preview = []
    for item in items:
        flag = str(item.get("flag") or "").strip()
        if not flag:
            continue
        if item.get("switch"):
            preview.append(flag)
        elif item.get("value") is not None:
            preview.append(f"{flag} {item.get('value')}")
    return preview


def build_source_planning_profile(
    source: dict[str, Any],
    *,
    daypart: str,
    desired_execution_policy: str | None = None,
) -> dict[str, Any]:
    daypart = str(daypart or "day").strip().lower()
    explicit_mode = normalize_workflow_mode_name(
        _resolve_with_daypart(source, daypart, "workflow_mode")
    )
    compatible_modes = normalize_workflow_mode_list(
        _resolve_with_daypart(source, daypart, "workflow_modes")
    )

    source_archetype = normalize_source_archetype(
        _resolve_with_daypart(source, daypart, "source_archetype")
        or infer_source_archetype(source)
    )
    archetype_spec = SOURCE_ARCHETYPE_SPECS[source_archetype]

    if explicit_mode != "adaptive":
        resolved_workflow_mode = explicit_mode
        workflow_reason = "Source declares an explicit workflow mode."
    elif desired_execution_policy and desired_execution_policy in compatible_modes:
        resolved_workflow_mode = desired_execution_policy
        workflow_reason = "Daemon execution policy matches the source compatibility list."
    elif compatible_modes:
        preferred = [mode for mode in compatible_modes if mode != "adaptive"]
        resolved_workflow_mode = preferred[0] if preferred else archetype_spec.default_workflow_mode
        workflow_reason = "Source compatibility list chooses the workflow mode."
    elif desired_execution_policy and desired_execution_policy != "adaptive":
        resolved_workflow_mode = desired_execution_policy
        workflow_reason = "Daemon execution policy takes priority for this adaptive source."
    else:
        resolved_workflow_mode = archetype_spec.default_workflow_mode
        workflow_reason = "Source archetype default determines the workflow mode."

    batch_profile = normalize_batch_profile(
        _resolve_with_daypart(source, daypart, "batch_profile")
        or infer_batch_profile(
            source,
            workflow_mode=resolved_workflow_mode,
            source_archetype=source_archetype,
        )
    )
    batch_spec = BATCH_PROFILE_SPECS[batch_profile]
    if resolved_workflow_mode == "adaptive":
        resolved_workflow_mode = batch_spec.default_workflow_mode
        workflow_reason = "Batch profile supplies the workflow mode for an adaptive source."
    if not compatible_modes:
        compatible_modes = [resolved_workflow_mode]

    planning_notes = str(source.get("planning_notes") or "").strip()
    alignment_tags = _as_string_list(_resolve_with_daypart(source, daypart, "alignment_tags"))
    recommended_flags = _build_recommended_flags(
        resolved_workflow_mode=resolved_workflow_mode,
        batch_profile=batch_profile,
    )

    return {
        "source_archetype": source_archetype,
        "source_archetype_label": archetype_spec.label,
        "source_archetype_summary": archetype_spec.summary,
        "archetype_inspirations": list(archetype_spec.inspirations),
        "batch_profile": batch_profile,
        "batch_profile_label": batch_spec.label,
        "batch_profile_summary": batch_spec.summary,
        "batch_goal": batch_spec.batch_goal,
        "resolved_workflow_mode": resolved_workflow_mode,
        "compatible_workflow_modes": compatible_modes,
        "workflow_reason": workflow_reason,
        "planning_notes": planning_notes,
        "alignment_tags": alignment_tags,
        "recommended_generator_defaults": recommended_flags,
        "recommended_generator_preview": _flag_preview_items(recommended_flags),
    }
