"""Composable writing skills inspired by research-writing prompt libraries."""

from __future__ import annotations

from typing import Dict, List


DEFAULT_WRITING_SKILLS = [
    "abstract_framing",
    "intro_motivation",
    "related_work_positioning",
    "method_clarity",
    "results_narrative",
    "limitations",
    "reproducibility",
    "rebuttal_tone",
    "plot_title",
    "table_title",
    "experiment_analysis",
    "anti_ai_style_polish",
]

WRITING_SKILLS: Dict[str, Dict[str, object]] = {
    "abstract_framing": {
        "summary": "Open with the problem, intervention, evidence, and bounded takeaway.",
        "checks": ["Abstract should name the strongest evidence, not just the ambition."],
    },
    "intro_motivation": {
        "summary": "Move quickly from research problem to why the current gap matters.",
        "checks": ["Introduction must justify why this problem is worth a reviewer’s attention."],
    },
    "related_work_positioning": {
        "summary": "Differentiate against the closest baselines instead of broad generic contrast.",
        "checks": ["Related work should state what is truly different and what remains similar."],
    },
    "method_clarity": {
        "summary": "Describe the method as a reproducible sequence, not a vague idea sketch.",
        "checks": ["Method section should expose inputs, transformations, and controls clearly."],
    },
    "results_narrative": {
        "summary": "Tie each major result to one claim and one implication.",
        "checks": ["Results should surface the lead finding before minor observations."],
    },
    "limitations": {
        "summary": "State boundary conditions and failure modes explicitly.",
        "checks": ["Limitations should narrow scope rather than weaken trust."],
    },
    "reproducibility": {
        "summary": "Make setup choices, datasets, and metrics explicit enough to repeat.",
        "checks": ["Reproducibility details should be concrete and review-friendly."],
    },
    "rebuttal_tone": {
        "summary": "Use calm, precise, evidence-first language that anticipates reviewer objections.",
        "checks": ["Claims should sound review-ready, not defensive or promotional."],
    },
    "plot_title": {
        "summary": "Use descriptive figure titles that encode the comparison and takeaway.",
        "checks": ["Plot titles should help a reader understand the point before reading the caption."],
    },
    "table_title": {
        "summary": "Write table titles that specify data, metric, and comparison target.",
        "checks": ["Table titles should disambiguate what is being measured."],
    },
    "experiment_analysis": {
        "summary": "Interpret experiment outcomes with concrete numerical and methodological context.",
        "checks": ["Analysis should separate observed outcomes from speculative explanations."],
    },
    "anti_ai_style_polish": {
        "summary": "Remove repetitive generic phrasing and keep the prose natural but precise.",
        "checks": ["The manuscript should read like a researcher report, not polished marketing copy."],
    },
}


def list_writing_skills() -> List[str]:
    return list(DEFAULT_WRITING_SKILLS)


def render_writing_skill_pack(
    skills: List[str] | None = None,
    *,
    target_venue: str | None = None,
) -> str:
    resolved = skills or DEFAULT_WRITING_SKILLS
    lines = [
        "Writing skill pack:",
        f"- Target venue: {target_venue or 'general'}",
    ]
    for skill in resolved:
        spec = WRITING_SKILLS.get(skill)
        if not isinstance(spec, dict):
            continue
        lines.append(f"- {skill}: {spec.get('summary')}")
        for item in spec.get("checks") or []:
            lines.append(f"  check: {item}")
    return "\n".join(lines)
