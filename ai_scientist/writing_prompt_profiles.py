"""Reusable writing profile directives for paper generation prompts.

These profiles are inspired by practical prompt libraries used in research
writing workflows. They are intentionally short, explicit, and composable.
"""

from __future__ import annotations

from typing import Dict, List

DEFAULT_WRITING_PROFILE = "default"

WRITING_PROFILE_SPECS: Dict[str, Dict[str, object]] = {
    "default": {
        "summary": "Balanced academic writing with standard rigor and readability.",
        "constraints": [
            "Report methods and results faithfully; do not invent unsupported claims.",
            "Keep structure coherent and aligned with venue conventions.",
            "Preserve valid LaTeX commands and references.",
        ],
        "self_checks": [
            "Every major claim is supported by experiment logs, tables, or figures.",
            "Section transitions are clear and logically connected.",
            "The manuscript remains compilable LaTeX end-to-end.",
        ],
    },
    "strict_latex": {
        "summary": "Prioritize clean LaTeX style and concise professional tone.",
        "constraints": [
            "Prefer continuous paragraphs; avoid itemize/enumerate unless essential.",
            "Avoid decorative formatting and avoid unnecessary rhetorical punctuation.",
            "Escape LaTeX special characters and keep equations unchanged.",
        ],
        "self_checks": [
            "No malformed commands, unmatched braces, or broken references remain.",
            "No placeholder text or unfinished TODO-style fragments remain.",
            "Captions and in-text references are consistent and precise.",
        ],
    },
    "reviewer_strict": {
        "summary": "Write as if preparing for a tough top-conference review.",
        "constraints": [
            "State assumptions, limitations, and failure modes explicitly.",
            "Do not over-claim; separate observations from conclusions.",
            "Compare against baselines with concrete evidence from provided outputs.",
        ],
        "self_checks": [
            "Key novelty points are explicit and distinguishable from prior work.",
            "Potential reviewer objections are proactively addressed in text.",
            "Any negative or inconclusive findings are reported transparently.",
        ],
    },
    "logic_first": {
        "summary": "Emphasize argument flow and paragraph-level coherence.",
        "constraints": [
            "Each paragraph should center on one core point with clear linkage.",
            "Use explicit cause-effect or problem-solution transitions when needed.",
            "Keep terminology consistent across sections to avoid ambiguity.",
        ],
        "self_checks": [
            "No abrupt topic jumps between adjacent paragraphs.",
            "Introduction, method, and results follow a consistent narrative arc.",
            "Conclusions map directly to evidence shown in experiments.",
        ],
    },
    "conference_checklist": {
        "summary": "Optimize for submission-readiness against venue checklist requirements.",
        "constraints": [
            "Treat claims, reproducibility, and limitation disclosures as hard constraints.",
            "Add concrete implementation and evaluation details before polishing language.",
            "Prefer explicit compliance wording over implied coverage for required items.",
        ],
        "self_checks": [
            "Major venue checklist items are explicitly covered or intentionally marked N/A.",
            "Compute/data/code access and uncertainty reporting are stated concretely.",
            "Limitations and risk discussion match the scope and claims of the paper.",
        ],
    },
    "citation_guard": {
        "summary": "Prioritize citation integrity and explicit uncertainty handling.",
        "constraints": [
            "Never invent citations, metadata, or claims not backed by evidence.",
            "If uncertain citation metadata exists, preserve placeholder markers for manual check.",
            "Keep citation usage precise: each claim should map to the most relevant source.",
        ],
        "self_checks": [
            "All in-text citation keys resolve to references.bib or explicit placeholders.",
            "No unsupported claim is phrased as definitively cited fact.",
            "Related work distinguishes what is verified versus pending verification.",
        ],
    },
    "humanized_academic": {
        "summary": "Keep technical rigor while reducing formulaic AI-style prose patterns.",
        "constraints": [
            "Avoid marketing adjectives and vague authority phrasing.",
            "Use direct verbs and concrete claims instead of inflated generic language.",
            "Maintain natural sentence rhythm without sacrificing precision.",
        ],
        "self_checks": [
            "The text reads like a researcher report, not promotional copy.",
            "Uncertainty and scope boundaries are stated naturally and explicitly.",
            "Paragraph style varies enough to avoid repetitive AI-like cadence.",
        ],
    },
}


def list_writing_profiles() -> List[str]:
    """Return supported writing profiles in deterministic order."""
    return sorted(WRITING_PROFILE_SPECS.keys())


def normalize_writing_profile(profile: str | None) -> str:
    """Normalize and validate profile name."""
    raw = profile if profile is not None else DEFAULT_WRITING_PROFILE
    normalized = str(raw).strip().lower().replace("-", "_")
    if not normalized:
        normalized = DEFAULT_WRITING_PROFILE
    if normalized not in WRITING_PROFILE_SPECS:
        available = ", ".join(list_writing_profiles())
        raise ValueError(f"Unknown writing profile '{profile}'. Available: {available}")
    return normalized


def render_writing_profile_system_guidance(profile: str | None) -> str:
    """Render concise profile guidance for system prompts."""
    resolved = normalize_writing_profile(profile)
    spec = WRITING_PROFILE_SPECS[resolved]
    constraints = spec.get("constraints", [])

    lines = [
        f"Writing profile: {resolved}",
        f"Profile goal: {spec.get('summary', '')}",
        "Profile constraints:",
    ]
    for idx, item in enumerate(constraints, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def render_writing_profile_self_checks(profile: str | None) -> str:
    """Render profile-specific self-check list for reflection prompts."""
    resolved = normalize_writing_profile(profile)
    spec = WRITING_PROFILE_SPECS[resolved]
    checks = spec.get("self_checks", [])

    lines = [f"Profile self-checks ({resolved}):"]
    for idx, item in enumerate(checks, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)
