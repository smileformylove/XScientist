from __future__ import annotations

import json
import os
import os.path as osp
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from ai_scientist.config.paths import resolve_output_path
from ai_scientist.utils.pipeline_helpers import (
    compile_latex,
    find_best_pdf_path,
    iter_bfts_run_dirs,
)

QUALITY_PRESETS = {
    "balanced": {
        "quality_threshold": 3.8,
        "rigor_threshold": 3.0,
        "max_rewrite_rounds": 1,
        "auto_improvement_fallback": True,
    },
    "high": {
        "quality_threshold": 4.1,
        "rigor_threshold": 3.5,
        "max_rewrite_rounds": 2,
        "auto_improvement_fallback": True,
    },
    "publishable": {
        "quality_threshold": 4.3,
        "rigor_threshold": 3.8,
        "max_rewrite_rounds": 2,
        "auto_improvement_fallback": True,
    },
}

VENUE_PRESETS = {
    "neurips": {
        "template": "neurips",
        "quality_threshold": 4.2,
        "rigor_threshold": 3.7,
        "claim_support_threshold": 3.6,
        "style_guidance": "强调方法创新、实验严谨性、baseline 对比、消融和可复现性。",
        "min_cite_rounds": 20,
        "min_writeup_retries": 4,
        "candidate_boost": 0,
        "min_review_reflections": 2,
        "min_review_ensemble": 3,
        "min_review_fewshot": 2,
        "checklist": [
            "Clear contribution list in introduction",
            "Strong baseline comparisons and ablations",
            "Reproducibility details and implementation clarity",
            "Numerical takeaways in abstract and conclusion",
        ],
        "submission_priority_threshold": 75.0,
        "max_submission_blockers": 2,
    },
    "iclr": {
        "template": "iclr",
        "quality_threshold": 4.1,
        "rigor_threshold": 3.6,
        "claim_support_threshold": 3.5,
        "style_guidance": "强调核心 insight、方法清晰度、理论或经验支撑以及完整消融。",
        "min_cite_rounds": 18,
        "min_writeup_retries": 4,
        "candidate_boost": 0,
        "min_review_reflections": 2,
        "min_review_ensemble": 3,
        "min_review_fewshot": 2,
        "checklist": [
            "Clear core insight and motivation",
            "Method and evaluation described cleanly",
            "Ablation or analytical evidence for design choices",
            "Credible limitations and scope discussion",
        ],
        "submission_priority_threshold": 74.0,
        "max_submission_blockers": 2,
    },
    "cvpr": {
        "template": "cvpr",
        "quality_threshold": 4.1,
        "rigor_threshold": 3.5,
        "claim_support_threshold": 3.5,
        "style_guidance": "强调图像表达、实验全面性、视觉结果质量与比较充分性。",
        "min_cite_rounds": 18,
        "min_writeup_retries": 4,
        "candidate_boost": 0,
        "min_review_reflections": 2,
        "min_review_ensemble": 3,
        "min_review_fewshot": 2,
        "checklist": [
            "High-quality visual evidence",
            "Competitive baselines on standard benchmarks",
            "Readable captions and well-referenced figures",
            "Quantitative and qualitative analysis both present",
        ],
        "submission_priority_threshold": 75.0,
        "max_submission_blockers": 2,
    },
    "journal": {
        "template": "journal",
        "quality_threshold": 4.3,
        "rigor_threshold": 3.9,
        "claim_support_threshold": 3.7,
        "style_guidance": "强调完整性、稳定性、理论与实验闭环、局限性与复现实验细节。",
        "min_cite_rounds": 22,
        "min_writeup_retries": 5,
        "candidate_boost": 1,
        "min_review_reflections": 3,
        "min_review_ensemble": 4,
        "min_review_fewshot": 2,
        "checklist": [
            "Comprehensive literature coverage",
            "Thorough methods and experimental details",
            "Robustness, sensitivity, and limitations discussion",
            "Submission-ready narrative coherence across sections",
        ],
        "submission_priority_threshold": 80.0,
        "max_submission_blockers": 2,
    },
    "nature": {
        "template": "nature",
        "quality_threshold": 4.5,
        "rigor_threshold": 4.1,
        "claim_support_threshold": 4.0,
        "style_guidance": "强调重大问题、跨领域意义、核心证据链和高度克制的 claim。",
        "min_cite_rounds": 25,
        "min_writeup_retries": 5,
        "candidate_boost": 1,
        "min_review_reflections": 3,
        "min_review_ensemble": 5,
        "min_review_fewshot": 3,
        "checklist": [
            "Broad scientific significance is explicit",
            "Claims are ambitious but numerically grounded",
            "Strongest results are easy to identify",
            "Limitations and caveats are clearly acknowledged",
        ],
        "submission_priority_threshold": 86.0,
        "max_submission_blockers": 1,
    },
}

TEMPLATE_MAP = {
    "normal": "neurips",
    "icbinb": "icbinb",
    "journal": "journal",
    "extended": "icbinb",
}

SECTION_PRIORITY_MAP = {
    "structure": ["introduction", "related", "method", "experiment", "conclusion"],
    "content": ["method", "experiment", "results", "discussion"],
    "innovation": ["abstract", "introduction", "discussion", "conclusion"],
    "rigor": ["method", "experiment", "results", "analysis", "discussion"],
    "clarity": ["abstract", "introduction", "results", "conclusion"],
    "professionalism": ["abstract", "introduction", "conclusion"],
}

SECTION_CANDIDATE_COUNT = {
    "title": 3,
    "abstract": 3,
    "introduction": 3,
    "conclusion": 3,
    "results": 2,
    "discussion": 2,
    "method": 2,
    "experiment": 2,
}

SECTION_SPECIFIC_PRIORITIES = {
    "abstract": [
        "突出最核心贡献与最强数字结果",
        "避免空泛表述，优先给出定量 takeaway",
        "用最少术语表达最强结论与边界",
    ],
    "introduction": [
        "先交代为什么问题重要，再讲贡献",
        "明确 contribution bullets，避免分散叙事",
        "强化 broad impact / scientific significance framing",
    ],
    "results": [
        "按 strongest-results 顺序组织结果",
        "每个核心结论都尽量绑定 figure/table 和关键数字",
        "优先比较最强 baseline 与最关键 ablation",
    ],
    "discussion": [
        "解释 strongest results 为什么重要",
        "明确局限性与外推边界",
        "强调意义但避免过度外推",
    ],
    "conclusion": [
        "凝练总结最重要贡献和 strongest evidence",
        "保留边界条件和未来方向",
        "避免没有数字支撑的强结论",
    ],
    "general": [
        "保持叙事聚焦、证据充分、风格克制",
    ],
}


def _canonical_section_family(section_name: str) -> str:
    section = (section_name or "general").lower()
    for key in [
        "abstract",
        "title",
        "introduction",
        "method",
        "results",
        "discussion",
        "conclusion",
    ]:
        if key in section:
            return key
    if "experiment" in section or "analysis" in section:
        return "results"
    return "general"


def _get_section_priority_notes(title: str) -> list[str]:
    title_lower = title.lower()
    for key, notes in SECTION_SPECIFIC_PRIORITIES.items():
        if key != "general" and key in title_lower:
            return notes
    return SECTION_SPECIFIC_PRIORITIES["general"]


def map_paper_type_to_template(paper_type: str) -> str:
    return TEMPLATE_MAP.get(paper_type, "neurips")


def recommend_target_venue_from_idea(idea: dict | None, paper_type: str) -> str:
    default_venue = {
        "normal": "neurips",
        "icbinb": "iclr",
        "journal": "nature",
        "extended": "iclr",
    }.get(paper_type, "neurips")
    if idea is None:
        return default_venue

    combined = " ".join(
        str(idea.get(key, ""))
        for key in [
            "Title",
            "Abstract",
            "Short Hypothesis",
            "Hypothesis",
            "Impact",
            "Field",
            "Task",
        ]
    ).lower()
    broad_impact_markers = [
        "real-world",
        "societal",
        "climate",
        "medical",
        "biology",
        "agriculture",
        "broad impact",
        "scientific discovery",
        "cross-domain",
        "fundamental",
        "major challenge",
        "high impact",
    ]
    vision_markers = ["vision", "image", "video", "detection", "segmentation"]

    if any(marker in combined for marker in broad_impact_markers) and paper_type in {
        "normal",
        "journal",
    }:
        return "nature"
    if any(marker in combined for marker in vision_markers):
        return "cvpr"
    return default_venue


def is_paper_type_fit_for_venue(paper_type: str, target_venue: str) -> bool:
    venue_fit = {
        "neurips": {"normal"},
        "iclr": {"normal", "icbinb", "extended"},
        "cvpr": {"normal"},
        "journal": {"journal", "normal"},
        "nature": {"journal", "normal"},
    }
    return paper_type in venue_fit.get(target_venue, {paper_type})


def recommend_paper_type_for_venue(target_venue: str) -> str:
    return {
        "neurips": "normal",
        "iclr": "normal",
        "cvpr": "normal",
        "journal": "journal",
        "nature": "journal",
    }.get(target_venue, "normal")


def resolve_target_venue(paper_type: str, target_venue: Optional[str] = None) -> str:
    if target_venue is not None:
        return target_venue
    return recommend_target_venue_from_idea(None, paper_type)


def resolve_submission_acceptance_settings(
    target_venue: str,
    *,
    min_submission_priority: Optional[float] = None,
    max_submission_blockers: Optional[int] = None,
) -> tuple[Optional[float], Optional[int]]:
    venue_config = VENUE_PRESETS.get(target_venue, VENUE_PRESETS["neurips"])
    return (
        (
            min_submission_priority
            if min_submission_priority is not None
            else venue_config.get("submission_priority_threshold")
        ),
        (
            max_submission_blockers
            if max_submission_blockers is not None
            else venue_config.get("max_submission_blockers")
        ),
    )


def evaluate_submission_acceptance(
    quality_result: dict,
    *,
    require_quality_gate: bool = False,
    min_submission_priority: Optional[float] = None,
    max_submission_blockers: Optional[int] = None,
    reject_on_auto_improvement_fallback: bool = False,
) -> dict:
    reasons = []
    accepted = True
    if require_quality_gate and quality_result.get("quality_gate_passed") is not True:
        accepted = False
        reasons.append("required quality gate not met")
    if (
        reject_on_auto_improvement_fallback
        and quality_result.get("auto_improvement_fallback_used") is True
    ):
        accepted = False
        reasons.append(
            "auto-improvement fallback rewrites were used under strict submission discipline"
        )
    if min_submission_priority is not None:
        priority_score = quality_result.get("submission_priority_score")
        if (
            not isinstance(priority_score, (int, float))
            or priority_score < min_submission_priority
        ):
            accepted = False
            reasons.append(
                f"submission priority below target ({priority_score if priority_score is not None else 'n/a'} < {min_submission_priority})"
            )
    if max_submission_blockers is not None:
        blocker_count = quality_result.get("blocker_count")
        if isinstance(blocker_count, int) and blocker_count > max_submission_blockers:
            accepted = False
            reasons.append(
                f"too many blockers ({blocker_count} > {max_submission_blockers})"
            )
    return {
        "accepted": accepted,
        "reasons": reasons,
        "submission_priority_score": quality_result.get("submission_priority_score"),
        "submission_priority_tier": quality_result.get("submission_priority_tier"),
        "blocker_count": quality_result.get("blocker_count"),
    }


def should_resume_high_quality_result(
    existing_result: dict[str, Any] | None,
    *,
    auto_improvement_fallback: bool,
    target_venue: Optional[str] = None,
    quality_threshold: Optional[float] = None,
    rigor_threshold: Optional[float] = None,
) -> bool:
    if not isinstance(existing_result, dict):
        return False
    if str(existing_result.get("status") or "").lower() != "success":
        return False
    fallback_enabled = existing_result.get("auto_improvement_fallback_enabled")
    if not isinstance(fallback_enabled, bool):
        return False
    if fallback_enabled is not bool(auto_improvement_fallback):
        return False
    if target_venue is not None:
        if str(existing_result.get("target_venue") or "").strip() != str(target_venue):
            return False
    if quality_threshold is not None:
        existing_quality_threshold = existing_result.get("quality_threshold")
        if not isinstance(existing_quality_threshold, (int, float)):
            return False
        if float(existing_quality_threshold) != float(quality_threshold):
            return False
    if rigor_threshold is not None:
        existing_rigor_threshold = existing_result.get("rigor_threshold")
        if not isinstance(existing_rigor_threshold, (int, float)):
            return False
        if float(existing_rigor_threshold) != float(rigor_threshold):
            return False
    return True


def _extract_latex_response(text: str) -> str:
    latex_match = re.search(r"```latex\s*(.*?)\s*```", text, re.DOTALL)
    if latex_match:
        return latex_match.group(1)
    generic_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1)
    return text.strip()


def _safe_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_text_dump(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _build_submission_package_text(report: dict, result: dict) -> str:
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    key_results = report.get("key_results", {}).get("values", [])
    contribution_map = report.get("contribution_map", [])
    result_story = report.get("evidence_pack", {}).get("strongest_results", [])
    structured_results = report.get("key_results", {}).get("structured_results", [])
    claim_ledger = report.get("claim_ledger", [])
    venue_checklist = VENUE_PRESETS.get(result.get("target_venue"), {}).get(
        "checklist", []
    )
    breakthrough_profile = report.get("breakthrough_profile", {})
    claim_rewrite_suggestions = [
        item.get("suggested_rewrite")
        for item in report.get("claim_ledger", [])
        if item.get("suggested_rewrite")
    ]
    readiness = result.get("submission_readiness", {})
    lines = [
        f"# Submission Package",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Quality score: {result.get('quality_score_after')}",
        f"- Rigor score: {result.get('rigor_score_after')}",
        f"- Claim support score: {result.get('claim_support_after')}",
        f"- Breakthrough score: {result.get('breakthrough_score')}",
        f"- Submission priority score: {result.get('submission_priority_score')}",
        f"- Submission priority tier: {result.get('submission_priority_tier')}",
        f"- Quality gate passed: {result.get('quality_gate_passed')}",
        f"- Submission readiness: {readiness.get('status')}",
        "",
        "## Contribution Map",
    ]
    if contribution_map:
        for item in contribution_map[:5]:
            lines.extend(
                [
                    f"### {item.get('title')}",
                    f"- Claim: {item.get('claim')}",
                    f"- Evidence labels: {', '.join(item.get('evidence_labels', [])) or 'n/a'}",
                    f"- Key results: {', '.join(item.get('key_results', [])) or 'n/a'}",
                    f"- Limitation: {item.get('limitation') or 'n/a'}",
                    "",
                ]
            )
    else:
        lines.append("- No contribution map generated.")
        lines.append("")

    lines.extend(
        [
            "## Strongest Results",
        ]
    )
    if strongest_results:
        for item in strongest_results[:5]:
            lines.append(
                f"- [{item.get('type')}] {item.get('label')} | refs={item.get('ref_count')} | {item.get('caption', '')}"
            )
    else:
        lines.append("- No figure/table evidence extracted.")

    lines.append("")
    lines.append("## Breakthrough Profile")
    lines.append(f"- Score: {breakthrough_profile.get('score')}")
    for key, value in (breakthrough_profile.get("checks") or {}).items():
        lines.append(f"- {key}: {'yes' if value else 'no'}")

    lines.append("")
    lines.append("## Venue Checklist")
    if venue_checklist:
        for item in venue_checklist:
            lines.append(f"- {item}")
    else:
        lines.append("- No venue checklist available.")

    lines.append("")
    lines.append("## Key Numerical Results")
    if key_results:
        for value in key_results[:10]:
            lines.append(f"- {value}")
    else:
        lines.append("- No key numerical results extracted.")

    lines.append("")
    lines.append("## Structured Best Results")
    if structured_results:
        for item in structured_results[:10]:
            lines.append(
                f"- {item.get('source')} :: {item.get('path')} = {item.get('value')}"
            )
    else:
        lines.append("- No structured result summaries extracted.")

    lines.append("")
    lines.append("## Claim Ledger")
    if claim_ledger:
        for item in claim_ledger[:10]:
            lines.append(
                f"- [{item.get('section')}] supported={item.get('supported')} links={item.get('linked_results')} claim={item.get('claim')}"
            )
    else:
        lines.append("- No claim ledger entries extracted.")

    lines.append("")
    lines.append("## Suggested Claim Rewrites")
    if claim_rewrite_suggestions:
        for suggestion in claim_rewrite_suggestions[:10]:
            lines.append(f"- {suggestion}")
    else:
        lines.append("- No claim rewrites suggested.")

    lines.append("")
    lines.append("## Priority Revision Actions")
    revision_actions = result.get("revision_actions", [])
    if revision_actions:
        for item in revision_actions[:6]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('focus')}: {item.get('action')} ({item.get('reason')})"
            )
    else:
        lines.append("- No revision actions generated.")

    lines.append("")
    lines.append("## Submission Blockers")
    blockers = readiness.get("blockers", [])
    categories = readiness.get("categories", {})
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None detected.")

    lines.append("")
    lines.append("## Blocker Categories")
    for key, value in categories.items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines) + "\n"


def _build_narrative_map_text(report: dict, result: dict) -> str:
    contribution_map = report.get("contribution_map", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    lines = [
        "# Narrative Map",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Submission readiness: {result.get('submission_readiness', {}).get('status')}",
        "",
    ]
    for item in contribution_map[:5]:
        lines.extend(
            [
                f"## {item.get('title')}",
                f"- Claim: {item.get('claim')}",
                f"- Evidence labels: {', '.join(item.get('evidence_labels', [])) or 'n/a'}",
                f"- Key results: {', '.join(item.get('key_results', [])) or 'n/a'}",
                f"- Limitation: {item.get('limitation') or 'n/a'}",
                "",
            ]
        )
    if strongest_results:
        lines.append("## Strongest Visual/Table Evidence")
        for item in strongest_results[:5]:
            lines.append(
                f"- [{item.get('type')}] {item.get('label')} :: {item.get('caption', '')}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_result_story_text(report: dict, result: dict) -> str:
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    contribution_map = report.get("contribution_map", [])
    lines = [
        "# Result Story",
        "",
        f"Target venue: {result.get('target_venue')}",
        "",
        "## Recommended Story Order",
    ]

    if not strongest_results:
        lines.append("- No strongest results available.")
        return "\n".join(lines) + "\n"

    for idx, item in enumerate(strongest_results[:5], start=1):
        linked_contribs = [
            contrib.get("title")
            for contrib in contribution_map
            if item.get("label") in (contrib.get("evidence_labels") or [])
        ]
        lines.extend(
            [
                f"### Step {idx}",
                f"- Evidence: [{item.get('type')}] {item.get('label')}",
                f"- Why here: ref_count={item.get('ref_count')} | caption={item.get('caption')}",
                f"- Supports: {', '.join(linked_contribs) or 'core result narrative'}",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def _build_editor_pitch_text(report: dict, result: dict) -> str:
    contribution_map = report.get("contribution_map", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    key_results = report.get("key_results", {}).get("values", [])
    readiness = result.get("submission_readiness", {})

    lines = [
        "# Editor Pitch",
        "",
        f"Target venue: {result.get('target_venue')}",
        f"Quality status: {result.get('quality_status')}",
        f"Submission readiness: {readiness.get('status')}",
        "",
        "## One-paragraph pitch",
    ]
    if contribution_map:
        lead_claim = contribution_map[0].get("claim")
        lead_results = ", ".join(
            contribution_map[0].get("key_results", [])[:2]
        ) or ", ".join(key_results[:2])
        lead_evidence = ", ".join(contribution_map[0].get("evidence_labels", [])[:2])
        lines.append(
            f"This work targets {result.get('target_venue')} with a central contribution: {lead_claim}. "
            f"The strongest quantitative evidence includes {lead_results or 'key results under extraction'}, "
            f"supported by {lead_evidence or 'core figures/tables'}."
        )
    else:
        lines.append(
            "Contribution map unavailable; summarize the core contribution manually before submission."
        )

    lines.extend(["", "## Contribution Bullets"])
    if contribution_map:
        for item in contribution_map[:5]:
            lines.append(f"- {item.get('claim')}")
    else:
        lines.append("- No contribution bullets available.")

    lines.extend(["", "## Strongest Evidence"])
    if strongest_results:
        for item in strongest_results[:5]:
            lines.append(f"- {item.get('label')}: {item.get('caption')}")
    else:
        lines.append("- No strongest evidence extracted.")

    return "\n".join(lines) + "\n"


def _build_rebuttal_package_text(report: dict, result: dict) -> str:
    readiness = result.get("submission_readiness", {})
    blockers = readiness.get("blockers", [])
    historical_reviewer_risks = result.get("historical_reviewer_risks") or report.get(
        "historical_reviewer_risks", {}
    )
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    key_results = report.get("key_results", {}).get("values", [])

    lines = [
        "# Anticipated Reviewer Questions and Rebuttal Draft",
        "",
        f"Target venue: {result.get('target_venue')}",
        "",
    ]

    anticipated_questions = list(blockers[:3])
    anticipated_questions.extend(
        historical_reviewer_risks.get("anticipated_objections", [])[:3]
    )
    if not anticipated_questions:
        anticipated_questions = [
            "What is the strongest empirical improvement over the best baseline?",
            "Why is this contribution significant beyond the immediate benchmark?",
        ]

    for idx, question in enumerate(anticipated_questions, start=1):
        lines.append(f"## Q{idx}: {question}")
        lines.append("### Draft response")
        lines.append(
            f"We address this concern by grounding our response in the strongest available evidence. "
            f"Relevant quantitative results include {', '.join(key_results[:3]) or 'key results under extraction'}, "
            f"with supporting figures/tables such as {', '.join(item.get('label', '') for item in strongest_results[:3]) or 'core evidence items'}."
        )
        lines.append("")

    lines.append("## Historical Reviewer-Risk Guidance")
    for item in historical_reviewer_risks.get("rebuttal_focus", [])[:5]:
        lines.append(f"- {item}")
    if historical_reviewer_risks.get("claim_softening_advice"):
        lines.append("")
        lines.append("## Claim Softening Advice")
        for item in historical_reviewer_risks.get("claim_softening_advice", [])[:5]:
            lines.append(f"- {item}")
    if historical_reviewer_risks.get("limitation_emphasis"):
        lines.append("")
        lines.append("## Limitation Emphasis")
        for item in historical_reviewer_risks.get("limitation_emphasis", [])[:5]:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def _build_risk_register_text(result: dict) -> str:
    readiness = result.get("submission_readiness", {})
    lines = [
        "# Risk Register",
        "",
        f"Target venue: {result.get('target_venue')}",
        f"Quality status: {result.get('quality_status')}",
        f"Submission readiness: {readiness.get('status')}",
        "",
        "## Risk Categories",
    ]
    for key, value in (readiness.get("categories") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Top Risks")
    blockers = readiness.get("blockers") or ["No major blockers detected."]
    for blocker in blockers[:10]:
        lines.append(f"- {blocker}")
    return "\n".join(lines) + "\n"


def _build_submission_dashboard_text(report: dict, result: dict) -> str:
    readiness = result.get("submission_readiness", {})
    scorecard = result.get("submission_scorecard") or report.get(
        "submission_scorecard", {}
    )
    revision_actions = result.get("revision_actions") or report.get(
        "revision_actions", []
    )
    historical_rewrite_focus = result.get("historical_rewrite_focus") or report.get(
        "historical_rewrite_focus", {}
    )
    historical_reviewer_risks = result.get("historical_reviewer_risks") or report.get(
        "historical_reviewer_risks", {}
    )
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    strongest_claims = sorted(
        report.get("claim_ledger", []),
        key=lambda item: (
            item.get("supported") is True,
            len(item.get("linked_results", [])),
        ),
        reverse=True,
    )[:3]
    contribution_map = report.get("contribution_map", [])
    lines = [
        "# Submission Dashboard",
        "",
        f"- Decision: {'submit_or_share_with_confidence' if readiness.get('ready') else 'revise_before_submission'}",
        f"- Target venue: {result.get('target_venue')}",
        f"- Submission readiness: {readiness.get('status')}",
        f"- Submission priority: {result.get('submission_priority_score')} ({result.get('submission_priority_tier')})",
        f"- Quality gate passed: {result.get('quality_gate_passed')}",
        f"- Rewrite applied: {result.get('rewrite_applied')}",
        "",
        "## Scorecard",
    ]
    if scorecard:
        for key, item in scorecard.items():
            if not isinstance(item, dict):
                continue
            label = key.replace("_", " ").title()
            lines.append(
                f"- {label}: score={item.get('score')} threshold={item.get('threshold')} pass={item.get('pass')} gap={item.get('gap')}"
            )
    else:
        lines.append("- No scorecard available.")

    lines.extend(["", "## Lead Contributions"])
    if contribution_map:
        for item in contribution_map[:3]:
            lines.append(f"- {item.get('claim')}")
    else:
        lines.append("- No contribution map generated.")

    lines.extend(["", "## Lead Evidence"])
    if strongest_results:
        for item in strongest_results[:3]:
            lines.append(
                f"- [{item.get('type')}] {item.get('label')} | refs={item.get('ref_count')} | {item.get('caption', '')}"
            )
    else:
        lines.append("- No figure/table evidence extracted.")

    lines.extend(["", "## Claims To Defend Carefully"])
    if strongest_claims:
        for item in strongest_claims:
            lines.append(
                f"- supported={item.get('supported')} | links={item.get('linked_results')} | claim={item.get('claim')}"
            )
    else:
        lines.append("- No claim ledger entries available.")

    lines.extend(["", "## Priority Revision Actions"])
    if revision_actions:
        for item in revision_actions[:6]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('focus')}: {item.get('action')} ({item.get('reason')})"
            )
    else:
        lines.append("- No revision actions generated.")

    lines.extend(["", "## Top Blockers"])
    blockers = readiness.get("blockers") or ["No major blockers detected."]
    for blocker in blockers[:8]:
        lines.append(f"- {blocker}")

    lines.extend(["", "## Risk Categories"])
    for key, value in (readiness.get("categories") or {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Ranking Reasons"])
    for reason in result.get("submission_priority_reasons", [])[:6]:
        lines.append(f"- {reason}")
    if not result.get("submission_priority_reasons"):
        lines.append("- No ranking reasons generated.")

    lines.extend(["", "## Historical Rewrite Hotspots"])
    for reason in historical_rewrite_focus.get("rationale", [])[:4]:
        lines.append(f"- {reason}")
    if historical_rewrite_focus.get("preferred_sections"):
        lines.append(
            f"- Preferred sections: {', '.join(historical_rewrite_focus.get('preferred_sections', [])[:6])}"
        )
    if not historical_rewrite_focus.get("rationale"):
        lines.append("- No historical rewrite hotspots generated.")

    lines.extend(["", "## Historical Reviewer Risks"])
    for item in historical_reviewer_risks.get("anticipated_objections", [])[:4]:
        lines.append(f"- {item}")
    if historical_reviewer_risks.get("claim_softening_advice"):
        lines.append(
            f"- Claim softening: {historical_reviewer_risks.get('claim_softening_advice', [])[0]}"
        )
    if historical_reviewer_risks.get("limitation_emphasis"):
        lines.append(
            f"- Limitation emphasis: {historical_reviewer_risks.get('limitation_emphasis', [])[0]}"
        )
    if risk_language_guidance.get("objection_preemption"):
        lines.append(
            f"- Abstract preemption: {risk_language_guidance.get('objection_preemption', [])[0]}"
        )
    if not historical_reviewer_risks.get("anticipated_objections"):
        lines.append("- No historical reviewer risks generated.")

    return "\n".join(lines) + "\n"


def _build_cover_letter_text(report: dict, result: dict) -> str:
    contribution_map = report.get("contribution_map", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    readiness = result.get("submission_readiness", {})
    venue = result.get("target_venue")

    salutation = (
        "Dear Editors"
        if venue in {"nature", "journal"}
        else "Dear Area Chairs and Reviewers"
    )
    intro = (
        f"We are pleased to submit our manuscript for consideration at {venue}. "
        f"The work is positioned around the following central contribution: "
        f"{contribution_map[0].get('claim') if contribution_map else 'a high-impact contribution under refinement'}."
    )
    evidence = (
        f"The strongest supporting evidence includes {', '.join(item.get('label') for item in strongest_results[:3]) or 'our core result figures/tables'}, "
        f"with submission readiness currently assessed as {readiness.get('status')}."
    )
    significance = "We believe the manuscript is relevant to the venue because it combines a clear contribution statement, structured evidence, and explicit limitations."
    return (
        "\n\n".join(
            [
                salutation + ",",
                intro,
                evidence,
                significance,
                "Sincerely,\nThe Authors",
            ]
        )
        + "\n"
    )


def _build_impact_brief_text(report: dict, result: dict) -> str:
    contribution_map = report.get("contribution_map", [])
    breakthrough = report.get("breakthrough_profile", {})
    lines = [
        "# Impact Brief",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Breakthrough potential: {breakthrough.get('score')}",
        f"- Assessment: {breakthrough.get('overall_assessment')}",
        "",
        "## Why this might matter broadly",
    ]
    if contribution_map:
        lines.append(f"- {contribution_map[0].get('claim')}")
    else:
        lines.append("- Core broad-impact framing still needs refinement.")

    lines.extend(["", "## Grand-Challenge Checklist"])
    for key, passed in (breakthrough.get("checks") or {}).items():
        lines.append(f"- {key}: {'yes' if passed else 'no'}")

    lines.extend(["", "## Recommendations"])
    recs = breakthrough.get("recommendations") or [
        "No additional impact recommendations."
    ]
    for rec in recs:
        lines.append(f"- {rec}")

    return "\n".join(lines) + "\n"


def _build_contribution_bullets_text(report: dict) -> str:
    contribution_map = report.get("contribution_map", [])
    lines = ["# Contribution Bullets", ""]
    if contribution_map:
        for item in contribution_map[:5]:
            lines.append(f"- {item.get('claim')}")
    else:
        lines.append("- No contribution bullets available.")
    return "\n".join(lines) + "\n"


def _build_strongest_claims_text(report: dict) -> str:
    claim_ledger = report.get("claim_ledger", [])
    strongest = sorted(
        claim_ledger,
        key=lambda item: (
            item.get("supported") is True,
            len(item.get("linked_results", [])),
        ),
        reverse=True,
    )[:5]
    lines = ["# Strongest Claims Summary", ""]
    if strongest:
        for item in strongest:
            lines.extend(
                [
                    f"- Claim: {item.get('claim')}",
                    f"  - Supported: {item.get('supported')}",
                    f"  - Linked results: {', '.join(item.get('linked_results', [])) or 'n/a'}",
                ]
            )
    else:
        lines.append("- No strongest claims extracted.")
    return "\n".join(lines) + "\n"


def _build_logic_check_text(report: dict, result: dict) -> str:
    claim_ledger = report.get("claim_ledger", [])
    unsupported_claims = report.get("claim_support", {}).get("unsupported_claims", [])
    readiness = result.get("submission_readiness", {})
    revision_actions = result.get("revision_actions") or report.get(
        "revision_actions", []
    )
    weak_claims = [
        item
        for item in claim_ledger
        if item.get("supported") is not True or not item.get("linked_results")
    ][:8]
    lines = [
        "# Logic Check Report",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Claim alignment score: {result.get('claim_alignment_after')}",
        f"- Claim support score: {result.get('claim_support_after')}",
        f"- Unsupported claims: {result.get('unsupported_claims_count')}",
        f"- Submission readiness: {readiness.get('status')}",
        "",
        "## Claim-Evidence Gaps",
    ]
    if weak_claims:
        for item in weak_claims:
            lines.append(
                f"- [{item.get('section')}] supported={item.get('supported')} | links={item.get('linked_results')} | claim={item.get('claim')}"
            )
    else:
        lines.append("- No obvious claim-evidence gaps detected.")

    lines.extend(["", "## Unsupported Claim Alerts"])
    if unsupported_claims:
        for item in unsupported_claims[:8]:
            lines.append(f"- {item}")
    else:
        lines.append("- No unsupported claims detected.")

    lines.extend(["", "## Logic Repair Priorities"])
    if revision_actions:
        for item in revision_actions[:8]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('focus')}: {item.get('action')} ({item.get('reason')})"
            )
    else:
        lines.append("- No logic repair actions generated.")

    lines.extend(["", "## Submission Blockers"])
    blockers = readiness.get("blockers") or []
    if blockers:
        for blocker in blockers[:8]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- No blocking logic issues detected.")
    return "\n".join(lines) + "\n"


def _build_reviewer_gate_report_text(report: dict, result: dict) -> str:
    scorecard = result.get("submission_scorecard") or report.get(
        "submission_scorecard", {}
    )
    reviewer_risks = result.get("historical_reviewer_risks") or report.get(
        "historical_reviewer_risks", {}
    )
    readiness = result.get("submission_readiness", {})
    lines = [
        "# Reviewer Gate Report",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Readiness: {readiness.get('status')}",
        f"- Quality gate: {result.get('quality_gate_passed')}",
        f"- Priority: {result.get('submission_priority_score')} ({result.get('submission_priority_tier')})",
        "",
        "## Scorecard",
    ]
    if scorecard:
        for key, item in scorecard.items():
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {key}: score={item.get('score')} threshold={item.get('threshold')} pass={item.get('pass')} gap={item.get('gap')}"
            )
    else:
        lines.append("- No reviewer scorecard available.")

    lines.extend(["", "## Anticipated Reviewer Objections"])
    objections = reviewer_risks.get("anticipated_objections", [])
    if objections:
        for item in objections[:8]:
            lines.append(f"- {item}")
    else:
        lines.append("- No anticipated objections extracted.")

    lines.extend(["", "## Rebuttal Focus"])
    rebuttal_focus = reviewer_risks.get("rebuttal_focus", [])
    if rebuttal_focus:
        for item in rebuttal_focus[:8]:
            lines.append(f"- {item}")
    else:
        lines.append("- No explicit rebuttal focus generated.")
    return "\n".join(lines) + "\n"


def _build_experiment_analysis_text(report: dict, result: dict) -> str:
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    structured_results = report.get("key_results", {}).get("structured_results", [])
    revision_actions = result.get("revision_actions") or report.get(
        "revision_actions", []
    )
    lines = [
        "# Experiment Analysis",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Rigor score: {result.get('rigor_score_after')}",
        f"- Numeric coverage: {result.get('numeric_coverage_after')}",
        f"- Evidence density: {result.get('evidence_density_score')}",
        "",
        "## Strongest Experimental Evidence",
    ]
    if strongest_results:
        for item in strongest_results[:8]:
            lines.append(
                f"- [{item.get('type')}] {item.get('label')} | refs={item.get('ref_count')} | {item.get('caption')}"
            )
    else:
        lines.append("- No strong figure/table evidence extracted.")

    lines.extend(["", "## Structured Numerical Results"])
    if structured_results:
        for item in structured_results[:10]:
            lines.append(
                f"- {item.get('source')} :: {item.get('path')} = {item.get('value')}"
            )
    else:
        lines.append("- No structured numerical results extracted.")

    lines.extend(["", "## Missing-or-Weak Experiment Signals"])
    experiment_actions = [
        item
        for item in revision_actions
        if str(item.get("focus") or "").lower()
        in {"experiments", "rigor", "results", "analysis"}
    ]
    if experiment_actions:
        for item in experiment_actions[:8]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('focus')}: {item.get('action')} ({item.get('reason')})"
            )
    else:
        lines.append("- No explicit experiment repair actions generated.")
    return "\n".join(lines) + "\n"


def _build_figure_caption_guidance_text(report: dict, result: dict) -> str:
    strongest_results = [
        item
        for item in report.get("evidence_pack", {}).get("strongest_results", [])
        if str(item.get("type") or "").lower() == "figure"
    ]
    lines = [
        "# Figure Caption Guidance",
        "",
        f"- Target venue: {result.get('target_venue')}",
        "",
    ]
    if strongest_results:
        for item in strongest_results[:8]:
            label = item.get("label") or "Figure"
            caption = item.get("caption") or "Main result figure"
            lines.extend(
                [
                    f"## {label}",
                    f"- Evidence-first title: {label}: Main empirical result supporting the core claim",
                    f"- Caption focus: {caption}",
                    "- Reviewer emphasis: state what changes, compared to what baseline, and why the reader should trust the effect.",
                    "",
                ]
            )
    else:
        lines.append("- No figure evidence extracted yet.")
    return "\n".join(lines) + "\n"


def _build_table_caption_guidance_text(report: dict, result: dict) -> str:
    strongest_results = [
        item
        for item in report.get("evidence_pack", {}).get("strongest_results", [])
        if str(item.get("type") or "").lower() == "table"
    ]
    lines = [
        "# Table Caption Guidance",
        "",
        f"- Target venue: {result.get('target_venue')}",
        "",
    ]
    if strongest_results:
        for item in strongest_results[:8]:
            label = item.get("label") or "Table"
            caption = item.get("caption") or "Main quantitative comparison"
            lines.extend(
                [
                    f"## {label}",
                    f"- Evidence-first title: {label}: Quantitative comparison that anchors the main performance claim",
                    f"- Caption focus: {caption}",
                    "- Reviewer emphasis: state dataset/setting, strongest comparator, and the exact metric trend the table establishes.",
                    "",
                ]
            )
    else:
        lines.append("- No table evidence extracted yet.")
    return "\n".join(lines) + "\n"


def _build_architecture_figure_brief_text(report: dict, result: dict) -> str:
    contribution_map = report.get("contribution_map", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    lines = [
        "# Architecture Figure Brief",
        "",
        f"- Target venue: {result.get('target_venue')}",
        "",
        "## Core Story To Visualize",
    ]
    if contribution_map:
        for item in contribution_map[:4]:
            lines.append(f"- {item.get('claim')}")
    else:
        lines.append(
            "- Visualize the main method pipeline and its strongest evidence-backed contribution."
        )

    lines.extend(["", "## Diagram Blocks"])
    if contribution_map:
        for idx, item in enumerate(contribution_map[:4], start=1):
            lines.append(f"- Block {idx}: {item.get('title') or item.get('claim')}")
    else:
        lines.append("- Problem / Inputs / Method / Outputs / Evidence")

    lines.extend(["", "## Reviewer-Facing Caption Goals"])
    if strongest_results:
        for item in strongest_results[:3]:
            lines.append(
                f"- Tie the architecture figure back to {item.get('label')}: {item.get('caption')}"
            )
    else:
        lines.append(
            "- Explicitly connect each stage of the architecture to the experimental evidence that validates it."
        )
    return "\n".join(lines) + "\n"


def _build_experiment_visualization_brief_text(report: dict, result: dict) -> str:
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    structured_results = report.get("key_results", {}).get("structured_results", [])
    revision_actions = result.get("revision_actions") or report.get(
        "revision_actions", []
    )
    lines = [
        "# Experiment Visualization Brief",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Submission readiness: {result.get('submission_readiness', {}).get('status')}",
        f"- Evidence density score: {result.get('evidence_density_score')}",
        f"- Numeric coverage score: {result.get('numeric_coverage_after')}",
        "",
        "## Must-Have Plot Goals",
        "- Surface the strongest result first with a figure or table that clearly anchors the central claim.",
        "- Include at least one ablation or sensitivity view when the revision actions mention rigor or experiment gaps.",
        "- Prefer compact multi-panel plots when multiple related findings tell one story.",
        "",
        "## Candidate Evidence To Visualize",
    ]
    if strongest_results:
        for item in strongest_results[:8]:
            lines.append(
                f"- [{item.get('type')}] {item.get('label')} | refs={item.get('ref_count')} | {item.get('caption')}"
            )
    else:
        lines.append(
            "- No strongest evidence extracted yet; prioritize plots that directly support the main quantitative claims."
        )

    lines.extend(["", "## Structured Numerical Highlights"])
    if structured_results:
        for item in structured_results[:10]:
            lines.append(
                f"- {item.get('source')} :: {item.get('path')} = {item.get('value')}"
            )
    else:
        lines.append("- No structured result summary available.")

    lines.extend(["", "## Plotting Priorities"])
    experiment_actions = [
        item
        for item in revision_actions
        if str(item.get("focus") or "").lower()
        in {"experiments", "rigor", "results", "analysis"}
    ]
    if experiment_actions:
        for item in experiment_actions[:8]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('focus')}: {item.get('action')} ({item.get('reason')})"
            )
    else:
        lines.append("- No explicit experiment plotting priorities extracted.")

    lines.extend(["", "## Figure / Table Title Guidance"])
    figure_items = [
        item
        for item in strongest_results
        if str(item.get("type") or "").lower() == "figure"
    ]
    table_items = [
        item
        for item in strongest_results
        if str(item.get("type") or "").lower() == "table"
    ]
    if figure_items:
        lines.append(
            f"- Figure titles should foreground evidence, e.g. '{figure_items[0].get('label')}: main empirical effect supporting the core claim'."
        )
    if table_items:
        lines.append(
            f"- Table titles should foreground comparison purpose, e.g. '{table_items[0].get('label')}: quantitative comparison against strongest baselines'."
        )
    if not figure_items and not table_items:
        lines.append(
            "- Use evidence-first titles that tell the reader exactly what each visualization proves."
        )

    lines.extend(["", "## Architecture Figure Linkage"])
    lines.append(
        "- If an architecture diagram is included, ensure at least one panel or caption sentence ties the design back to the strongest experimental result."
    )
    return "\n".join(lines) + "\n"


def _build_humanizer_style_notes_text(report: dict, result: dict) -> str:
    style_preferences = result.get("historical_style_preferences") or report.get(
        "historical_style_preferences", {}
    )
    reviewer_risks = result.get("historical_reviewer_risks") or report.get(
        "historical_reviewer_risks", {}
    )
    guidance = _derive_section_risk_language_guidance("abstract", report)
    lines = [
        "# Humanizer Style Notes",
        "",
        f"- Target venue: {result.get('target_venue')}",
        "",
        "## Tone Rules",
        "- Prefer concrete nouns and measured verbs over inflated marketing language.",
        "- Keep each paragraph centered on one evidence-backed move.",
        "- Avoid repetitive lead-ins such as 'This paper' or 'We propose' unless needed for clarity.",
        "",
        "## Historical Style Preferences",
        f"- Frontmatter order: {', '.join(style_preferences.get('frontmatter_style_order', [])) or 'n/a'}",
        f"- Section order: {', '.join(style_preferences.get('section_style_order', [])) or 'n/a'}",
        "",
        "## Claim Softening",
    ]
    softening = (
        guidance.get("claim_softening")
        or reviewer_risks.get("claim_softening_advice")
        or []
    )
    if softening:
        for item in softening[:6]:
            lines.append(f"- {item}")
    else:
        lines.append(
            "- Keep claims evidence-backed and avoid unsupported superiority language."
        )

    lines.extend(["", "## Reviewer-Facing Style Fixes"])
    objections = reviewer_risks.get("anticipated_objections", [])
    if objections:
        for item in objections[:6]:
            lines.append(f"- Address implicitly in prose: {item}")
    else:
        lines.append(
            "- Use concise, defensible phrasing that anticipates skepticism without sounding evasive."
        )
    return "\n".join(lines) + "\n"


def _build_writing_skill_pack_text(report: dict, result: dict) -> str:
    readiness = result.get("submission_readiness", {})
    reviewer_risks = result.get("historical_reviewer_risks") or report.get(
        "historical_reviewer_risks", {}
    )
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    contribution_map = report.get("contribution_map", [])
    lines = [
        "# Writing Skill Pack",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Submission readiness: {readiness.get('status')}",
        f"- Quality gate: {result.get('quality_gate_passed')}",
        "",
        "## Core Writing Skills",
        "- Logic-first drafting: every strong claim should point to explicit evidence and numbers.",
        "- Reviewer-aware framing: preempt likely objections without over-claiming.",
        "- Experiment storytelling: order results to reveal the core effect before the appendix-style extras.",
        "- Evidence-first figure/table titling: titles should say what the visual proves, not just what it contains.",
        "- Humanized scientific tone: concrete, restrained, and less repetitive phrasing.",
        "",
        "## Lead Contribution Signals",
    ]
    if contribution_map:
        for item in contribution_map[:4]:
            lines.append(f"- {item.get('claim')}")
    else:
        lines.append("- No contribution map available yet.")

    lines.extend(["", "## Strongest Evidence Signals"])
    if strongest_results:
        for item in strongest_results[:6]:
            lines.append(
                f"- [{item.get('type')}] {item.get('label')} | refs={item.get('ref_count')} | {item.get('caption')}"
            )
    else:
        lines.append("- No strongest figure/table evidence extracted yet.")

    lines.extend(["", "## Reviewer-Risk Reminders"])
    objections = reviewer_risks.get("anticipated_objections", [])
    if objections:
        for item in objections[:6]:
            lines.append(f"- {item}")
    else:
        lines.append("- No specific reviewer objections extracted.")

    lines.extend(["", "## Claim Softening Rules"])
    for item in (
        reviewer_risks.get("claim_softening_advice")
        or ["Prefer measured, evidence-backed claims."]
    )[:6]:
        lines.append(f"- {item}")

    lines.extend(["", "## Figure / Table Writing Rules"])
    lines.append(
        "- Every central figure/table should connect back to one contribution claim and one numerical takeaway."
    )
    lines.append(
        "- Captions should explain comparison target, metric movement, and why the result matters."
    )
    lines.append(
        "- Architecture visuals should explicitly connect design choices to strongest empirical evidence."
    )

    return "\n".join(lines) + "\n"


def _dedupe_string_list(values: list[Any], *, limit: Optional[int] = None) -> list[str]:
    merged: list[str] = []
    for value in values:
        text = str(value).strip() if value is not None else ""
        if not text or text in merged:
            continue
        merged.append(text)
        if limit is not None and len(merged) >= limit:
            break
    return merged


def _merge_preference_order(
    preferred: list[Any],
    existing: list[Any],
    *,
    defaults: Optional[list[str]] = None,
) -> list[str]:
    merged = _dedupe_string_list(list(preferred or []) + list(existing or []))
    for item in defaults or []:
        if item not in merged:
            merged.append(item)
    return merged


def _merge_revision_action_list(
    preferred: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in list(preferred or []) + list(existing or []):
        if not isinstance(item, dict):
            continue
        normalized = {
            "priority": str(item.get("priority") or "").strip() or "P2",
            "focus": str(item.get("focus") or "").strip(),
            "action": str(item.get("action") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
        }
        signature = (normalized["focus"], normalized["action"])
        if not any(signature) or signature in seen:
            continue
        seen.add(signature)
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _apply_autonomous_followup_focus(
    report: dict,
    autonomous_followup_focus: Optional[dict[str, Any]],
) -> dict:
    if not isinstance(autonomous_followup_focus, dict) or not autonomous_followup_focus:
        return report

    report["autonomous_followup_focus"] = autonomous_followup_focus

    historical_focus = dict(report.get("historical_rewrite_focus") or {})
    historical_focus["preferred_sections"] = _dedupe_string_list(
        list(autonomous_followup_focus.get("preferred_sections") or [])
        + list(historical_focus.get("preferred_sections") or []),
        limit=8,
    )
    historical_focus["rationale"] = _dedupe_string_list(
        list(autonomous_followup_focus.get("notes") or [])
        + list(historical_focus.get("rationale") or []),
        limit=8,
    )
    if autonomous_followup_focus.get("candidate_boost") is not None:
        historical_focus["autonomous_candidate_boost"] = int(
            autonomous_followup_focus.get("candidate_boost") or 0
        )
    if autonomous_followup_focus.get("target_section_limit") is not None:
        historical_focus["autonomous_target_section_limit"] = int(
            autonomous_followup_focus.get("target_section_limit") or 0
        )
    report["historical_rewrite_focus"] = historical_focus

    historical_risks = dict(report.get("historical_reviewer_risks") or {})
    for key in [
        "claim_softening_advice",
        "limitation_emphasis",
        "rebuttal_focus",
        "anticipated_objections",
    ]:
        historical_risks[key] = _dedupe_string_list(
            list(autonomous_followup_focus.get(key) or [])
            + list(historical_risks.get(key) or []),
            limit=6,
        )
    report["historical_reviewer_risks"] = historical_risks

    style_preferences = dict(report.get("historical_style_preferences") or {})
    style_preferences["frontmatter_style_order"] = _merge_preference_order(
        list(autonomous_followup_focus.get("frontmatter_style_order") or []),
        list(style_preferences.get("frontmatter_style_order") or []),
        defaults=["professional", "conservative", "assertive"],
    )
    style_preferences["section_style_order"] = _merge_preference_order(
        list(autonomous_followup_focus.get("section_style_order") or []),
        list(style_preferences.get("section_style_order") or []),
        defaults=["professional", "conservative", "assertive"],
    )
    report["historical_style_preferences"] = style_preferences

    report["revision_actions"] = _merge_revision_action_list(
        list(autonomous_followup_focus.get("required_actions") or []),
        list(report.get("revision_actions") or []),
    )
    return report


def _derive_section_risk_language_guidance(section_name: str, report: dict) -> dict:
    section = (section_name or "general").lower()
    risks = report.get("historical_reviewer_risks", {}) or {}
    guidance = {
        "section": section_name,
        "recommended_tone": [],
        "claim_softening": [],
        "limitation_emphasis": [],
        "objection_preemption": [],
    }

    def add(bucket: str, text: str):
        if text and text not in guidance[bucket]:
            guidance[bucket].append(text)

    claim_softening = risks.get("claim_softening_advice", [])
    limitation_emphasis = risks.get("limitation_emphasis", [])
    rebuttal_focus = risks.get("rebuttal_focus", [])
    objections = risks.get("anticipated_objections", [])

    if any(key in section for key in ["title", "abstract"]):
        add(
            "recommended_tone",
            "Prefer narrower, evidence-backed phrasing over broad or absolute framing.",
        )
        add(
            "recommended_tone",
            "Make one strongest numerical takeaway explicit when possible.",
        )
        add(
            "claim_softening",
            (
                claim_softening[0]
                if claim_softening
                else "Use restrained verbs and avoid unsupported superiority claims."
            ),
        )
        add(
            "limitation_emphasis",
            (
                limitation_emphasis[0]
                if limitation_emphasis
                else "Include a brief scope caveat when evidence is incomplete."
            ),
        )
    if "introduction" in section:
        add(
            "recommended_tone",
            "Frame novelty and significance narrowly before expanding broader implications.",
        )
        add(
            "claim_softening",
            (
                claim_softening[0]
                if claim_softening
                else "Avoid over-claiming broad novelty before evidence is presented."
            ),
        )
        add(
            "limitation_emphasis",
            (
                limitation_emphasis[0]
                if limitation_emphasis
                else "Clarify contribution scope before broad claims."
            ),
        )
        if rebuttal_focus:
            add("objection_preemption", rebuttal_focus[0])
    if any(key in section for key in ["results", "experiment", "analysis"]):
        add(
            "recommended_tone",
            "Tie every strong statement to one figure/table and one quantitative comparison.",
        )
        add(
            "claim_softening",
            (
                claim_softening[0]
                if claim_softening
                else "Avoid claiming significant gains without concrete numbers."
            ),
        )
        if rebuttal_focus:
            add("objection_preemption", rebuttal_focus[0])
    if any(key in section for key in ["discussion", "conclusion"]):
        add(
            "recommended_tone",
            "End with one explicit caveat and keep broad-significance language restrained.",
        )
        add(
            "claim_softening",
            (
                claim_softening[0]
                if claim_softening
                else "Prefer cautious language such as 'suggests' when generality is not fully established."
            ),
        )
        add(
            "limitation_emphasis",
            (
                limitation_emphasis[0]
                if limitation_emphasis
                else "Add one explicit limitation sentence and one scope boundary."
            ),
        )
        if rebuttal_focus:
            add("objection_preemption", rebuttal_focus[0])
    if "method" in section:
        add(
            "recommended_tone",
            "Describe capability precisely; avoid generality or robustness claims beyond evaluation scope.",
        )
        if claim_softening:
            add("claim_softening", claim_softening[0])

    for item in objections[:2]:
        add("objection_preemption", item)
    return {
        key: (value[:4] if isinstance(value, list) else value)
        for key, value in guidance.items()
    }


def _tokenize_guidance_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", (text or "").lower())
        if token
        not in {
            "the",
            "and",
            "with",
            "that",
            "this",
            "from",
            "into",
            "over",
            "than",
            "when",
            "where",
            "then",
        }
    }


def _score_candidate_risk_alignment(
    candidate_text: str, guidance: dict, sentence_plan: dict
) -> dict:
    candidate_lower = (candidate_text or "").lower()
    hedging_markers = [
        "suggest",
        "consistent with",
        "may",
        "might",
        "within the evaluated",
        "evaluated setting",
        "scope",
        "caveat",
        "limitation",
    ]
    limitation_markers = [
        "limitation",
        "scope",
        "future work",
        "further validation",
        "not yet",
        "generalization",
        "caution",
    ]
    evidence_markers = ["figure", "table", "fig.", "tab.", "baseline", "ablation"]
    numeric_markers = bool(re.search(r"\d", candidate_lower))

    hits = []
    score = 0.0

    guidance_texts = []
    for key in ["claim_softening", "limitation_emphasis", "objection_preemption"]:
        guidance_texts.extend(guidance.get(key) or [])
    guidance_texts.extend(
        sentence_plan.get(key, "")
        for key in [
            "softened_claim_sentence",
            "limitation_sentence",
            "objection_preemption_sentence",
        ]
        if sentence_plan.get(key)
    )
    guidance_tokens = set()
    for item in guidance_texts:
        guidance_tokens |= _tokenize_guidance_text(item)
    overlap = len(guidance_tokens & _tokenize_guidance_text(candidate_text))
    if overlap:
        score += min(2.0, overlap * 0.15)
        hits.append(f"guidance_token_overlap={overlap}")

    if any(marker in candidate_lower for marker in hedging_markers):
        score += 1.2
        hits.append("contains_hedging_language")
    if any(marker in candidate_lower for marker in limitation_markers):
        score += 1.2
        hits.append("contains_limitation_language")
    if any(marker in candidate_lower for marker in evidence_markers):
        score += 0.8
        hits.append("contains_evidence_reference")
    if numeric_markers:
        score += 0.8
        hits.append("contains_numeric_anchor")

    score = round(min(5.0, score), 2)
    return {
        "risk_alignment_score": score,
        "risk_alignment_hits": hits,
    }


def _style_key_from_variant_text(style_variant: str) -> str:
    variant = (style_variant or "").lower()
    if "清晰" in variant or "风险最小" in variant or "克制" in variant:
        return "conservative"
    if "贡献" in variant or "创新" in variant or "论证力度" in variant:
        return "assertive"
    if "专业" in variant or "成熟" in variant:
        return "professional"
    return "general"


def _ordered_style_variants(
    style_variants: list[str], preferred_order: list[str]
) -> list[str]:
    weights = {
        key: len(preferred_order) - idx for idx, key in enumerate(preferred_order or [])
    }
    return sorted(
        style_variants,
        key=lambda item: (
            weights.get(_style_key_from_variant_text(item), 0),
            style_variants.index(item),
        ),
        reverse=True,
    )


def _build_section_sentence_plan(section_name: str, report: dict) -> dict:
    section = (section_name or "general").lower()
    guidance = _derive_section_risk_language_guidance(section_name, report)
    contribution_map = report.get("contribution_map", [])
    key_results = report.get("key_results", {}).get("values", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])

    lead_claim = (
        contribution_map[0].get("claim")
        if contribution_map
        else "the method improves the target problem setting"
    )
    lead_number = key_results[0] if key_results else "the strongest quantitative gain"
    lead_evidence = (
        strongest_results[0].get("label")
        if strongest_results
        else "the strongest evidence figure/table"
    )

    softened_claim = (
        f"{lead_claim}; however, the current evidence should be interpreted as strongest support within the evaluated setting rather than a universal guarantee."
        if any(
            key in section
            for key in ["abstract", "introduction", "discussion", "conclusion"]
        )
        else f"This result should be interpreted within the evaluated setting, with claims tied directly to {lead_evidence}."
    )
    limitation_sentence = f"A current limitation is that the evidence is strongest for the evaluated setting and baselines, so broader generalization still requires further validation."
    objection_preemption = f"To preempt reviewer concern, explicitly connect the main statement to {lead_number} and reference {lead_evidence} in the same paragraph."

    if guidance.get("claim_softening"):
        softened_claim = guidance["claim_softening"][0]
    if guidance.get("limitation_emphasis"):
        limitation_sentence = guidance["limitation_emphasis"][0]
    if guidance.get("objection_preemption"):
        objection_preemption = guidance["objection_preemption"][0]

    return {
        "section": section_name,
        "softened_claim_sentence": softened_claim,
        "limitation_sentence": limitation_sentence,
        "objection_preemption_sentence": objection_preemption,
    }


def _build_claim_softening_plan_text(report: dict, result: dict) -> str:
    sections = ["abstract", "introduction", "results", "discussion", "conclusion"]
    lines = [
        "# Claim Softening and Limitation Plan",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Submission readiness: {result.get('submission_readiness', {}).get('status')}",
        "",
    ]
    for section in sections:
        plan = _build_section_sentence_plan(section, report)
        lines.extend(
            [
                f"## {section.title()}",
                f"- Softened claim: {plan.get('softened_claim_sentence')}",
                f"- Limitation sentence: {plan.get('limitation_sentence')}",
                f"- Objection preemption: {plan.get('objection_preemption_sentence')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _reviewer_risk_target_sections(report: dict) -> list[str]:
    risks = report.get("historical_reviewer_risks", {}) or {}
    sections = []
    if risks.get("claim_softening_advice"):
        sections.extend(["introduction", "conclusion"])
    if risks.get("limitation_emphasis"):
        sections.extend(["discussion", "conclusion"])
    if risks.get("anticipated_objections"):
        sections.extend(["results"])
    deduped = []
    for item in sections:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _summarize_rewrite_trace(rewrite_trace: list[dict]) -> dict:
    summary = {
        "round_count": len(rewrite_trace or []),
        "priority_gain_total": 0.0,
        "quality_gain_total": 0.0,
        "best_round": None,
        "frontmatter_styles": {},
        "section_styles": {},
        "section_hits": {},
        "top_frontmatter_style": None,
        "top_section_style": None,
        "top_section": None,
        "avg_priority_gain_per_round": 0.0,
    }
    best_round = None
    best_gain = None
    for round_item in rewrite_trace or []:
        pre_priority = round_item.get("pre_submission_priority_score")
        post_priority = round_item.get("post_submission_priority_score")
        pre_quality = round_item.get("pre_quality_score")
        post_quality = round_item.get("post_quality_score")
        priority_delta = (
            (post_priority - pre_priority)
            if isinstance(pre_priority, (int, float))
            and isinstance(post_priority, (int, float))
            else 0.0
        )
        quality_delta = (
            (post_quality - pre_quality)
            if isinstance(pre_quality, (int, float))
            and isinstance(post_quality, (int, float))
            else 0.0
        )
        summary["priority_gain_total"] += priority_delta
        summary["quality_gain_total"] += quality_delta
        if best_gain is None or priority_delta > best_gain:
            best_gain = priority_delta
            best_round = {
                "round": round_item.get("round"),
                "priority_delta": round(priority_delta, 3),
                "quality_delta": round(quality_delta, 3),
            }

        for item in round_item.get("frontmatter", []) or []:
            style = item.get("selected_style_variant") or "unknown"
            summary["frontmatter_styles"][style] = (
                summary["frontmatter_styles"].get(style, 0) + 1
            )

        for item in round_item.get("targets", []) or []:
            section = item.get("section") or "unknown"
            summary["section_hits"][section] = (
                summary["section_hits"].get(section, 0) + 1
            )
            candidate_summary = item.get("candidate_summary", {}) or {}
            style = candidate_summary.get("selected_style_variant") or "unknown"
            summary["section_styles"][style] = (
                summary["section_styles"].get(style, 0) + 1
            )

    summary["priority_gain_total"] = round(summary["priority_gain_total"], 3)
    summary["quality_gain_total"] = round(summary["quality_gain_total"], 3)
    summary["avg_priority_gain_per_round"] = (
        round(summary["priority_gain_total"] / summary["round_count"], 3)
        if summary["round_count"]
        else 0.0
    )
    summary["best_round"] = best_round
    if summary["frontmatter_styles"]:
        summary["top_frontmatter_style"] = max(
            summary["frontmatter_styles"].items(), key=lambda item: item[1]
        )[0]
    if summary["section_styles"]:
        summary["top_section_style"] = max(
            summary["section_styles"].items(), key=lambda item: item[1]
        )[0]
    if summary["section_hits"]:
        summary["top_section"] = max(
            summary["section_hits"].items(), key=lambda item: item[1]
        )[0]
    return summary


def _build_rewrite_effectiveness_text(report: dict, result: dict) -> str:
    rewrite_summary = result.get(
        "rewrite_effectiveness_summary"
    ) or _summarize_rewrite_trace(result.get("rewrite_trace", []))
    style_preferences = result.get("historical_style_preferences") or report.get(
        "historical_style_preferences", {}
    )
    efficiency = result.get("historical_rewrite_efficiency") or report.get(
        "historical_rewrite_efficiency", {}
    )
    lines = [
        "# Rewrite Effectiveness Report",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Rewrite rounds: {rewrite_summary.get('round_count')}",
        f"- Total submission-priority gain: {rewrite_summary.get('priority_gain_total')}",
        f"- Avg submission-priority gain / round: {rewrite_summary.get('avg_priority_gain_per_round')}",
        f"- Total quality gain: {rewrite_summary.get('quality_gain_total')}",
        "",
        "## Best Round",
    ]
    if rewrite_summary.get("best_round"):
        best_round = rewrite_summary["best_round"]
        lines.append(
            f"- Round {best_round.get('round')}: priority_delta={best_round.get('priority_delta')} quality_delta={best_round.get('quality_delta')}"
        )
    else:
        lines.append("- No effective rewrite rounds recorded.")

    if rewrite_summary.get("top_frontmatter_style"):
        lines.append(
            f"- Top frontmatter style: {rewrite_summary.get('top_frontmatter_style')}"
        )
    if rewrite_summary.get("top_section_style"):
        lines.append(f"- Top section style: {rewrite_summary.get('top_section_style')}")
    if rewrite_summary.get("top_section"):
        lines.append(
            f"- Most valuable section target: {rewrite_summary.get('top_section')}"
        )

    lines.extend(["", "## Selected Frontmatter Styles"])
    if rewrite_summary.get("frontmatter_styles"):
        for style, count in sorted(
            rewrite_summary["frontmatter_styles"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            lines.append(f"- {style}: {count}")
    else:
        lines.append("- No frontmatter rewrite selections recorded.")

    lines.extend(["", "## Selected Section Styles"])
    if rewrite_summary.get("section_styles"):
        for style, count in sorted(
            rewrite_summary["section_styles"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            lines.append(f"- {style}: {count}")
    else:
        lines.append("- No section rewrite selections recorded.")

    lines.extend(["", "## Most-Rewritten Sections"])
    if rewrite_summary.get("section_hits"):
        for section, count in sorted(
            rewrite_summary["section_hits"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:6]:
            lines.append(f"- {section}: {count}")
    else:
        lines.append("- No targeted section rewrites recorded.")

    lines.extend(["", "## Historical Style Preferences"])
    lines.append(
        f"- Frontmatter order: {', '.join(style_preferences.get('frontmatter_style_order', [])) or 'n/a'}"
    )
    lines.append(
        f"- Section order: {', '.join(style_preferences.get('section_style_order', [])) or 'n/a'}"
    )

    lines.extend(["", "## Historical Rewrite Efficiency"])
    for item in efficiency.get("rationale", [])[:5]:
        lines.append(f"- {item}")
    if efficiency.get("preferred_sections"):
        lines.append(
            f"- Historically effective sections: {', '.join(efficiency.get('preferred_sections', []))}"
        )
    if efficiency.get("deprioritized_sections"):
        lines.append(
            f"- Historically low-yield sections: {', '.join(efficiency.get('deprioritized_sections', []))}"
        )
    if not efficiency.get("rationale"):
        lines.append("- No historical rewrite efficiency summary available.")

    return "\n".join(lines) + "\n"


def _build_risk_language_plan_text(report: dict, result: dict) -> str:
    sections = [
        "title",
        "abstract",
        "introduction",
        "results",
        "discussion",
        "conclusion",
    ]
    lines = [
        "# Risk Language Plan",
        "",
        f"- Target venue: {result.get('target_venue')}",
        f"- Submission readiness: {result.get('submission_readiness', {}).get('status')}",
        "",
    ]
    for section in sections:
        guidance = _derive_section_risk_language_guidance(section, report)
        lines.append(f"## {section.title()}")
        for key, label in [
            ("recommended_tone", "Tone"),
            ("claim_softening", "Claim Softening"),
            ("limitation_emphasis", "Limitation Emphasis"),
            ("objection_preemption", "Objection Preemption"),
        ]:
            values = guidance.get(key) or []
            if values:
                lines.append(f"### {label}")
                for item in values:
                    lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_abstract_polish_text(report: dict) -> str:
    contribution_map = report.get("contribution_map", [])
    key_results = report.get("key_results", {}).get("values", [])
    venue = report.get("target_venue")

    lead_claim = (
        contribution_map[0].get("claim")
        if contribution_map
        else "This work introduces a new approach to the target problem."
    )
    key_numbers = ", ".join(key_results[:2]) if key_results else "quantitative gains"
    guidance = _derive_section_risk_language_guidance("abstract", report)
    softening = (
        guidance.get("claim_softening")
        or ["Keep claims evidence-backed and restrained."]
    )[0]
    limitation = (
        guidance.get("limitation_emphasis")
        or ["Explicitly state scope and limitations."]
    )[0]
    return (
        f"Target venue: {venue}\n\n"
        f"Suggested polished abstract core:\n"
        f"{lead_claim} Our strongest quantitative takeaways include {key_numbers}. "
        f"{softening} {limitation}\n"
    )


def _extract_section_title(section_text: str) -> str:
    match = re.search(r"\\section\*?\{([^}]*)\}", section_text)
    return match.group(1).strip() if match else "unknown"


def _extract_title(latex_content: str) -> Optional[str]:
    match = re.search(r"\\title\{(.*?)\}", latex_content, re.DOTALL)
    return match.group(1).strip() if match else None


def _replace_title(latex_content: str, new_title: str) -> str:
    return re.sub(
        r"\\title\{(.*?)\}",
        lambda _match: f"\\title{{{new_title}}}",
        latex_content,
        count=1,
        flags=re.DOTALL,
    )


def _extract_abstract(latex_content: str) -> Optional[str]:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        latex_content,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _replace_abstract(latex_content: str, new_abstract: str) -> str:
    return re.sub(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        lambda _match: f"\\begin{{abstract}}\n{new_abstract}\n\\end{{abstract}}",
        latex_content,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _split_latex_sections(latex_content: str) -> tuple[str, list[dict]]:
    pattern = re.compile(r"^\\section\*?\{.*?\}", re.MULTILINE)
    matches = list(pattern.finditer(latex_content))
    if not matches:
        return latex_content, []

    prefix = latex_content[: matches[0].start()]
    sections = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(latex_content)
        content = latex_content[start:end]
        sections.append(
            {
                "index": idx,
                "start": start,
                "end": end,
                "title": _extract_section_title(content),
                "content": content,
            }
        )
    return prefix, sections


def _low_scoring_dimensions(report: dict, threshold: float) -> list[str]:
    professional = report.get("professional", {})
    dims = []
    for dim in [
        "structure",
        "content",
        "innovation",
        "rigor",
        "clarity",
        "professionalism",
    ]:
        score = professional.get(dim, {}).get("score", 5)
        if isinstance(score, (int, float)) and score < threshold:
            dims.append(dim)
    if report.get("rigor", {}).get("score", 5) < threshold and "rigor" not in dims:
        dims.append("rigor")

    historical_focus = report.get("historical_rewrite_focus", {})
    for dim, boost in (historical_focus.get("dimension_boosts") or {}).items():
        score = professional.get(dim, {}).get("score", 5)
        if (
            boost
            and dim not in dims
            and isinstance(score, (int, float))
            and score < (threshold + 0.4)
        ):
            dims.append(dim)
    return dims


def _identify_target_sections(
    report: dict, latex_content: str, threshold: float
) -> list[dict]:
    _, sections = _split_latex_sections(latex_content)
    if not sections:
        return []

    low_dims = _low_scoring_dimensions(report, threshold)
    if report.get("claim_support", {}).get("score", 5) < 3.5:
        low_dims.extend(["innovation", "clarity"])
    targets = []
    seen = set()
    for dim in low_dims:
        keywords = SECTION_PRIORITY_MAP.get(dim, [])
        for keyword in keywords:
            for section in sections:
                if section["index"] in seen:
                    continue
                if keyword.lower() in section["title"].lower():
                    targets.append({"dimension": dim, **section})
                    seen.add(section["index"])
                    break

    historical_focus = report.get("historical_rewrite_focus", {})
    efficiency = report.get("historical_rewrite_efficiency", {}) or {}
    preferred_sections = list(historical_focus.get("preferred_sections") or [])
    for item in efficiency.get("preferred_sections", []):
        if item not in preferred_sections:
            preferred_sections.append(item)
    for preferred in preferred_sections:
        for section in sections:
            if section["index"] in seen:
                continue
            if preferred.lower() in section["title"].lower():
                targets.append({"dimension": "general", **section})
                seen.add(section["index"])
                break

    for preferred in _reviewer_risk_target_sections(report):
        for section in sections:
            if section["index"] in seen:
                continue
            if preferred.lower() in section["title"].lower():
                targets.append({"dimension": "general", **section})
                seen.add(section["index"])
                break

    if not targets:
        for section in sections[: min(3, len(sections))]:
            if section["index"] not in seen:
                targets.append({"dimension": "general", **section})
    deprioritized = set(efficiency.get("deprioritized_sections", []))
    if deprioritized and len(targets) > 1:
        targets = sorted(
            targets,
            key=lambda item: (
                _canonical_section_family(item.get("title")) in deprioritized,
                item.get("index", 0),
            ),
        )
    autonomous_focus = report.get("autonomous_followup_focus", {}) or {}
    max_targets = 5 if _reviewer_risk_target_sections(report) else 4
    focus_limit = autonomous_focus.get("target_section_limit")
    if isinstance(focus_limit, int) and focus_limit > 0:
        max_targets = max(max_targets, focus_limit)
    return targets[:max_targets]


def _rewrite_section(
    section: dict,
    report: dict,
    *,
    client,
    model: str,
    rank_client,
    rank_model: str,
    candidate_boost: int = 0,
    venue_guidance: str = "",
) -> tuple[str, dict]:
    from ai_scientist.llm import get_response_from_llm

    professional = report.get("professional", {})
    dimension_feedback = (
        professional.get(section["dimension"], {})
        if section["dimension"] != "general"
        else {}
    )
    weaknesses = professional.get("overall", {}).get("weaknesses", [])
    recommendations = professional.get("overall", {}).get("recommendations", [])
    rigor_recommendations = report.get("rigor", {}).get("recommendations", [])
    unsupported_claims = report.get("claim_support", {}).get("unsupported_claims", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    key_results = report.get("key_results", {}).get("values", [])
    contribution_map = report.get("contribution_map", [])
    result_story = report.get("evidence_pack", {}).get("strongest_results", [])
    breakthrough_profile = report.get("breakthrough_profile", {})
    readiness_preview = report.get("submission_readiness_preview", {})
    revision_actions = report.get("revision_actions", [])
    historical_rewrite_focus = report.get("historical_rewrite_focus", {})
    historical_reviewer_risks = report.get("historical_reviewer_risks", {})
    autonomous_followup_focus = report.get("autonomous_followup_focus", {}) or {}
    section_risk_guidance = _derive_section_risk_language_guidance(
        section["title"], report
    )
    section_sentence_plan = _build_section_sentence_plan(section["title"], report)
    style_preferences = report.get("historical_style_preferences", {}) or {}
    section_priorities = _get_section_priority_notes(section["title"])

    base_prompt = f"""
请定向改进论文中的以下章节，目标是提高 {section['dimension']} 维度质量。

章节标题: {section['title']}
当前章节:
```latex
{section['content']}
```

该维度评估:
{json.dumps(dimension_feedback, indent=2, ensure_ascii=False)}

全局弱点:
{json.dumps(weaknesses, indent=2, ensure_ascii=False)}

全局建议:
{json.dumps(recommendations + rigor_recommendations, indent=2, ensure_ascii=False)}

需特别修复的 unsupported claims:
{json.dumps(unsupported_claims, indent=2, ensure_ascii=False)}

当前 strongest results 候选:
{json.dumps(strongest_results, indent=2, ensure_ascii=False)}

关键数值结果:
{json.dumps(key_results, indent=2, ensure_ascii=False)}

Contribution map:
{json.dumps(contribution_map, indent=2, ensure_ascii=False)}

Result story order:
{json.dumps(result_story, indent=2, ensure_ascii=False)}

Breakthrough profile:
{json.dumps(breakthrough_profile, indent=2, ensure_ascii=False)}

目标 venue 风格:
{venue_guidance}

该章节的叙事优先级:
{json.dumps(section_priorities, indent=2, ensure_ascii=False)}

当前投稿 readiness 预估:
{json.dumps(readiness_preview, indent=2, ensure_ascii=False)}

优先修订动作:
{json.dumps(revision_actions, indent=2, ensure_ascii=False)}

历史高频失败热点:
{json.dumps(historical_rewrite_focus, indent=2, ensure_ascii=False)}

历史 reviewer 风险:
{json.dumps(historical_reviewer_risks, indent=2, ensure_ascii=False)}

本轮 autonomous follow-up guidance:
{json.dumps(autonomous_followup_focus, indent=2, ensure_ascii=False)}

该章节的风险语言指导:
{json.dumps(section_risk_guidance, indent=2, ensure_ascii=False)}

该章节建议句子计划:
{json.dumps(section_sentence_plan, indent=2, ensure_ascii=False)}

要求:
1. 保留章节主题和 LaTeX 结构。
2. 增强论证力度、实验严谨性或表达清晰度。
3. 如果证据不足，降低夸张表述。
4. 直接返回该章节完整 LaTeX，包含原有 section 标题。
"""

    title_lower = section["title"].lower()
    candidate_count = 2
    for key, count in SECTION_CANDIDATE_COUNT.items():
        if key in title_lower:
            candidate_count = count
            break
    if any(
        section_risk_guidance.get(key)
        for key in ["claim_softening", "limitation_emphasis"]
    ):
        candidate_count += 1
    candidate_count = min(candidate_count + candidate_boost, 4)

    style_variants = _ordered_style_variants(
        [
            "优先保证表达清晰、结构稳健、风险最小。",
            "优先强化贡献表达、论证力度和实验说服力。",
            "优先强化学术专业性、凝练度和投稿成熟度。",
        ],
        style_preferences.get("section_style_order", []),
    )
    temperatures = [0.25, 0.45, 0.65]

    candidates = []
    for candidate_idx in range(candidate_count):
        prompt = (
            base_prompt
            + "\n附加风格要求: "
            + style_variants[candidate_idx % len(style_variants)]
        )
        response, _ = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model,
            system_message="你是资深学术写作专家，擅长对单个章节做最小但高价值的定向改写。",
            temperature=temperatures[candidate_idx % len(temperatures)],
        )
        rewritten = _extract_latex_response(response) or section["content"]
        alignment = _score_candidate_risk_alignment(
            rewritten, section_risk_guidance, section_sentence_plan
        )
        candidates.append(
            {
                "index": candidate_idx,
                "content": rewritten,
                "temperature": temperatures[candidate_idx % len(temperatures)],
                "style_variant": style_variants[candidate_idx % len(style_variants)],
                **alignment,
            }
        )

    ranking_prompt = f"""
你需要从多个候选中选择最适合作为论文章节最终版本的内容。

章节标题: {section['title']}
目标维度: {section['dimension']}
原始章节:
```latex
{section['content']}
```

评估摘要:
{json.dumps(dimension_feedback, indent=2, ensure_ascii=False)}

候选列表:
{json.dumps([{k: v for k, v in c.items() if k != 'content'} | {'content': c['content'][:3000]} for c in candidates], indent=2, ensure_ascii=False)}

请返回 JSON:
{{
  "best_index": 0,
  "scores": {{"0": 4.2, "1": 3.9}},
  "reason": "..."
}}

评估标准:
1. 是否更好解决该维度弱项
2. 是否保持论证严谨且不过度夸张
3. 是否表达清晰、适合直接投稿
4. 是否保留 LaTeX 结构稳定性
5. 是否已经自然融入 claim softening / limitation / objection-preemption 语言
"""

    ranking_response, _ = get_response_from_llm(
        prompt=ranking_prompt,
        client=rank_client,
        model=rank_model,
        system_message="你是资深学术评审专家，负责从多个候选章节中选择最适合投稿的版本。",
        temperature=0.2,
    )
    ranking_match = re.search(
        r"```json\s*(.*?)\s*```", ranking_response, re.DOTALL
    ) or re.search(r"\{.*\}", ranking_response, re.DOTALL)
    ranking = {}
    if ranking_match:
        try:
            ranking = json.loads(ranking_match.group(1))
        except json.JSONDecodeError:
            ranking = {}

    best_index = ranking.get("best_index", 0)
    if not isinstance(best_index, int) or not 0 <= best_index < len(candidates):
        best_index = max(
            range(len(candidates)),
            key=lambda idx: candidates[idx].get("risk_alignment_score", 0),
            default=0,
        )
    selected = candidates[best_index]["content"]
    chosen = candidates[best_index]
    return selected, {
        "best_index": best_index,
        "selected_style_variant": chosen.get("style_variant"),
        "selected_risk_alignment_score": chosen.get("risk_alignment_score"),
        "ranking": ranking,
        "candidates": [
            {
                "index": candidate["index"],
                "temperature": candidate["temperature"],
                "style_variant": candidate["style_variant"],
                "risk_alignment_score": candidate.get("risk_alignment_score"),
                "risk_alignment_hits": candidate.get("risk_alignment_hits"),
                "changed": candidate["content"] != section["content"],
            }
            for candidate in candidates
        ],
    }


def _rewrite_frontmatter_candidates(
    kind: str,
    current_text: str,
    report: dict,
    *,
    client,
    model: str,
    rank_client,
    rank_model: str,
    candidate_boost: int = 0,
    venue_guidance: str = "",
) -> tuple[str, dict]:
    from ai_scientist.llm import get_response_from_llm

    professional = report.get("professional", {})
    weaknesses = professional.get("overall", {}).get("weaknesses", [])
    recommendations = professional.get("overall", {}).get("recommendations", [])
    rigor_recommendations = report.get("rigor", {}).get("recommendations", [])
    claim_recommendations = report.get("claim_support", {}).get("recommendations", [])
    unsupported_claims = report.get("claim_support", {}).get("unsupported_claims", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    key_results = report.get("key_results", {}).get("values", [])
    contribution_map = report.get("contribution_map", [])
    result_story = report.get("evidence_pack", {}).get("strongest_results", [])
    breakthrough_profile = report.get("breakthrough_profile", {})
    readiness_preview = report.get("submission_readiness_preview", {})
    revision_actions = report.get("revision_actions", [])
    historical_rewrite_focus = report.get("historical_rewrite_focus", {})
    historical_reviewer_risks = report.get("historical_reviewer_risks", {})
    autonomous_followup_focus = report.get("autonomous_followup_focus", {}) or {}
    section_risk_guidance = _derive_section_risk_language_guidance(kind, report)
    section_sentence_plan = _build_section_sentence_plan(kind, report)
    style_preferences = report.get("historical_style_preferences", {}) or {}
    section_priorities = _get_section_priority_notes(kind)

    candidate_count = min(
        SECTION_CANDIDATE_COUNT.get(kind, 3)
        + candidate_boost
        + (
            1
            if any(
                section_risk_guidance.get(key)
                for key in ["claim_softening", "limitation_emphasis"]
            )
            else 0
        ),
        5,
    )
    temperatures = [0.2, 0.4, 0.6]
    style_variants = _ordered_style_variants(
        [
            "优先保证清晰和克制。",
            "优先突出创新性与核心贡献。",
            "优先强化可投稿性和专业性。",
        ],
        style_preferences.get("frontmatter_style_order", []),
    )

    candidates = []
    for idx in range(candidate_count):
        prompt = f"""
请改进论文的 {kind}。

当前 {kind}:
{current_text}

全局弱点:
{json.dumps(weaknesses, indent=2, ensure_ascii=False)}

全局建议:
{json.dumps(recommendations + rigor_recommendations + claim_recommendations, indent=2, ensure_ascii=False)}

需特别修复的 unsupported claims:
{json.dumps(unsupported_claims, indent=2, ensure_ascii=False)}

当前 strongest results 候选:
{json.dumps(strongest_results, indent=2, ensure_ascii=False)}

关键数值结果:
{json.dumps(key_results, indent=2, ensure_ascii=False)}

Contribution map:
{json.dumps(contribution_map, indent=2, ensure_ascii=False)}

Result story order:
{json.dumps(result_story, indent=2, ensure_ascii=False)}

Breakthrough profile:
{json.dumps(breakthrough_profile, indent=2, ensure_ascii=False)}

目标 venue 风格:
{venue_guidance}

该部分的叙事优先级:
{json.dumps(section_priorities, indent=2, ensure_ascii=False)}

当前投稿 readiness 预估:
{json.dumps(readiness_preview, indent=2, ensure_ascii=False)}

优先修订动作:
{json.dumps(revision_actions, indent=2, ensure_ascii=False)}

历史高频失败热点:
{json.dumps(historical_rewrite_focus, indent=2, ensure_ascii=False)}

历史 reviewer 风险:
{json.dumps(historical_reviewer_risks, indent=2, ensure_ascii=False)}

本轮 autonomous follow-up guidance:
{json.dumps(autonomous_followup_focus, indent=2, ensure_ascii=False)}

该部分的风险语言指导:
{json.dumps(section_risk_guidance, indent=2, ensure_ascii=False)}

该部分建议句子计划:
{json.dumps(section_sentence_plan, indent=2, ensure_ascii=False)}

额外风格要求: {style_variants[idx % len(style_variants)]}

要求:
1. 保持准确，不夸大贡献。
2. 明确问题、方法、结果与边界。
3. 适合直接用于投稿稿。
4. 仅返回改进后的 {kind} 文本本身，不要加解释。
"""
        response, _ = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model,
            system_message="你是资深学术写作专家，擅长优化论文题目和摘要。",
            temperature=temperatures[idx % len(temperatures)],
        )
        candidate_text = _extract_latex_response(response).strip()
        alignment = _score_candidate_risk_alignment(
            candidate_text, section_risk_guidance, section_sentence_plan
        )
        candidates.append(
            {
                "index": idx,
                "content": candidate_text,
                "temperature": temperatures[idx % len(temperatures)],
                "style_variant": style_variants[idx % len(style_variants)],
                **alignment,
            }
        )

    ranking_prompt = f"""
请从多个 {kind} 候选中选出最适合投稿的一版。

当前版本:
{current_text}

候选:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

评判标准:
1. 是否清晰准确
2. 是否突出真正贡献但不过度夸张
3. 是否与高质量评估建议一致
4. 是否更像成熟投稿稿
5. 是否已自然写入 claim softening / limitation / objection-preemption 语言

返回 JSON: {{"best_index": 0, "reason": "..."}}
"""
    ranking_response, _ = get_response_from_llm(
        prompt=ranking_prompt,
        client=rank_client,
        model=rank_model,
        system_message="你是资深学术评审专家，负责从多个题目/摘要候选中选出最优版本。",
        temperature=0.2,
    )
    ranking_match = re.search(
        r"```json\s*(.*?)\s*```", ranking_response, re.DOTALL
    ) or re.search(r"\{.*\}", ranking_response, re.DOTALL)
    ranking = {}
    if ranking_match:
        try:
            ranking = json.loads(ranking_match.group(1))
        except json.JSONDecodeError:
            ranking = {}
    best_index = ranking.get("best_index", 0)
    if not isinstance(best_index, int) or not 0 <= best_index < len(candidates):
        best_index = max(
            range(len(candidates)),
            key=lambda idx: candidates[idx].get("risk_alignment_score", 0),
            default=0,
        )
    chosen = candidates[best_index]
    return candidates[best_index]["content"], {
        "kind": kind,
        "best_index": best_index,
        "selected_style_variant": chosen.get("style_variant"),
        "selected_risk_alignment_score": chosen.get("risk_alignment_score"),
        "ranking": ranking,
        "candidates": [
            {k: v for k, v in c.items() if k != "content"} for c in candidates
        ],
    }


def _apply_frontmatter_rewrites(
    latex_content: str,
    report: dict,
    *,
    client,
    model: str,
    rank_client,
    rank_model: str,
    threshold: float,
    candidate_boost: int = 0,
    force_rewrite: bool = False,
    venue_guidance: str = "",
    logger: Callable[[str], None] = print,
) -> tuple[str, list[dict]]:
    rewrite_log = []
    updated = latex_content

    low_dims = _low_scoring_dimensions(report, threshold)
    frontmatter_risk_needed = bool(
        (report.get("historical_reviewer_risks", {}) or {}).get(
            "claim_softening_advice"
        )
        or (report.get("historical_reviewer_risks", {}) or {}).get(
            "limitation_emphasis"
        )
    )
    if (
        force_rewrite
        or any(
            dim in low_dims
            for dim in ["innovation", "clarity", "professionalism", "content"]
        )
        or frontmatter_risk_needed
    ):
        title_text = _extract_title(updated)
        if title_text:
            logger("High-quality pass: generating title candidates")
            rewritten_title, title_log = _rewrite_frontmatter_candidates(
                "title",
                title_text,
                report,
                client=client,
                model=model,
                rank_client=rank_client,
                rank_model=rank_model,
                candidate_boost=candidate_boost,
                venue_guidance=venue_guidance,
            )
            if rewritten_title and rewritten_title != title_text:
                updated = _replace_title(updated, rewritten_title)
            rewrite_log.append(title_log)

        abstract_text = _extract_abstract(updated)
        if abstract_text:
            logger("High-quality pass: generating abstract candidates")
            rewritten_abstract, abstract_log = _rewrite_frontmatter_candidates(
                "abstract",
                abstract_text,
                report,
                client=client,
                model=model,
                rank_client=rank_client,
                rank_model=rank_model,
                candidate_boost=candidate_boost,
                venue_guidance=venue_guidance,
            )
            if rewritten_abstract and rewritten_abstract != abstract_text:
                updated = _replace_abstract(updated, rewritten_abstract)
            rewrite_log.append(abstract_log)

    return updated, rewrite_log


def _apply_targeted_section_rewrites(
    latex_content: str,
    report: dict,
    *,
    client,
    model: str,
    rank_client,
    rank_model: str,
    threshold: float,
    candidate_boost: int = 0,
    venue_guidance: str = "",
    logger: Callable[[str], None] = print,
) -> tuple[str, dict]:
    prefix, sections = _split_latex_sections(latex_content)
    if not sections:
        return latex_content, {"mode": "full_document_fallback", "targets": []}

    targets = _identify_target_sections(report, latex_content, threshold)
    if not targets:
        return latex_content, {"mode": "no_targets", "targets": []}

    rewrites = {}
    rewrite_log = []
    for target in targets:
        logger(
            f"High-quality pass: rewriting section '{target['title']}' for {target['dimension']}"
        )
        rewritten, candidate_summary = _rewrite_section(
            target,
            report,
            client=client,
            model=model,
            rank_client=rank_client,
            rank_model=rank_model,
            candidate_boost=candidate_boost,
            venue_guidance=venue_guidance,
        )
        rewrites[target["index"]] = rewritten
        rewrite_log.append(
            {
                "section": target["title"],
                "dimension": target["dimension"],
                "changed": rewritten != target["content"],
                "candidate_summary": candidate_summary,
            }
        )

    rebuilt = prefix
    for section in sections:
        rebuilt += rewrites.get(section["index"], section["content"])
    return rebuilt, {"mode": "targeted_sections", "targets": rewrite_log}


def _load_idea(base_folder: str | Path) -> dict:
    idea_path = Path(base_folder) / "idea.json"
    if not idea_path.exists():
        return {}
    with open(idea_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_latex(base_folder: str | Path) -> tuple[Optional[Path], str]:
    latex_path = Path(base_folder) / "latex" / "template.tex"
    if not latex_path.exists():
        return None, ""
    with open(latex_path, "r", encoding="utf-8", errors="ignore") as f:
        return latex_path, f.read()


def assess_experiment_rigor(
    base_folder: str | Path, latex_content: Optional[str] = None
) -> dict:
    base_folder = Path(base_folder)
    if latex_content is None:
        _, latex_content = _load_latex(base_folder)

    text_fragments = [latex_content]
    for rel_dir in ["experiment", "experiment_results", "logs"]:
        candidate_dir = base_folder / rel_dir
        if not candidate_dir.exists():
            continue
        for path in candidate_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".txt",
                ".md",
                ".json",
                ".py",
                ".log",
            }:
                continue
            try:
                text_fragments.append(
                    path.read_text(encoding="utf-8", errors="ignore")[:4000]
                )
            except OSError:
                continue

    corpus = "\n".join(text_fragments).lower()

    checks = {
        "baseline": ["baseline", "sota", "compared", "comparison", "vs.", "versus"],
        "ablation": ["ablation", "without ", "w/o", "component", "variant"],
        "statistics": [
            "p-value",
            "confidence interval",
            "significant",
            "std",
            "standard deviation",
            "seed",
        ],
        "reproducibility": [
            "hyperparameter",
            "implementation details",
            "seed",
            "reproduce",
            "reproducibility",
            "code",
        ],
    }

    findings = {
        name: any(keyword in corpus for keyword in keywords)
        for name, keywords in checks.items()
    }
    met = sum(findings.values())
    score = 1.0 + met

    recommendations = []
    if not findings["baseline"]:
        recommendations.append("增加更强且更贴近任务的 baseline 对比。")
    if not findings["ablation"]:
        recommendations.append("增加 ablation 分析，解释关键模块贡献。")
    if not findings["statistics"]:
        recommendations.append("补充多次运行、置信区间或显著性检验。")
    if not findings["reproducibility"]:
        recommendations.append("补充实现细节、超参数与复现实验设置。")

    return {
        "score": score,
        "checks": findings,
        "recommendations": recommendations,
        "overall_assessment": "strong" if score >= 4 else "needs_improvement",
    }


def assess_claim_support(latex_content: str) -> dict:
    ledger = build_claim_evidence_ledger(latex_content)

    claim_markers = [
        "outperform",
        "significant",
        "state-of-the-art",
        "sota",
        "novel",
        "we show",
        "we demonstrate",
        "improves",
        "improvement",
        "superior",
    ]
    claims = [
        item["claim"]
        for item in ledger
        if any(marker in item["claim"].lower() for marker in claim_markers)
    ]

    has_figure_ref = any(
        token in latex_content.lower() for token in ["\\ref{fig", "figure", "fig."]
    )
    has_table_ref = any(
        token in latex_content.lower() for token in ["\\ref{tab", "table", "tab."]
    )
    has_citation = "\\cite{" in latex_content
    has_limitations = any(
        token in latex_content.lower()
        for token in ["limitation", "future work", "threats to validity"]
    )

    unsupported_claims = [item["claim"] for item in ledger if not item.get("supported")]

    score = 5.0
    if claims and unsupported_claims:
        score -= min(2.0, 0.5 * len(unsupported_claims))
    if not has_limitations:
        score -= 0.5
    if not has_citation:
        score -= 0.5
    if not (has_figure_ref or has_table_ref):
        score -= 0.5
    score = max(1.0, min(5.0, score))

    recommendations = []
    if unsupported_claims:
        recommendations.append(
            "为核心 claim 增加图表、表格或引用支撑，避免无证据强结论。"
        )
    if not has_limitations:
        recommendations.append("补充 limitations / future work，提升可信度。")
    if not has_citation:
        recommendations.append("补充与现有工作的引用对照，避免孤立 claim。")

    return {
        "score": score,
        "claims_detected": len(claims),
        "unsupported_claims": unsupported_claims[:5],
        "has_figure_ref": has_figure_ref,
        "has_table_ref": has_table_ref,
        "has_citation": has_citation,
        "has_limitations": has_limitations,
        "recommendations": recommendations,
        "overall_assessment": "supported" if score >= 4 else "needs_improvement",
    }


def assess_numeric_coverage(latex_content: str) -> dict:
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        latex_content,
        re.DOTALL | re.IGNORECASE,
    )
    conclusion_match = re.search(
        r"\\section\*?\{[^}]*conclusion[^}]*\}(.*?)(?=\\section\*?\{|\\bibliography|\\end\{document\})",
        latex_content,
        re.DOTALL | re.IGNORECASE,
    )
    numeric_patterns = [r"\b\d+\.\d+\b", r"\b\d+%\b", r"\b\d+\.\d+%\b"]

    abstract_numbers = []
    conclusion_numbers = []
    for pattern in numeric_patterns:
        if abstract_match:
            abstract_numbers.extend(re.findall(pattern, abstract_match.group(1)))
        if conclusion_match:
            conclusion_numbers.extend(re.findall(pattern, conclusion_match.group(1)))

    score = 5.0
    recommendations = []
    if len(abstract_numbers) == 0:
        score -= 1.0
        recommendations.append("摘要中加入 1-2 个关键数值结果。")
    if len(conclusion_numbers) == 0:
        score -= 0.8
        recommendations.append("结论中加入关键数值或定量 takeaway。")
    score = max(1.0, min(5.0, score))

    return {
        "score": score,
        "abstract_numbers": abstract_numbers[:10],
        "conclusion_numbers": conclusion_numbers[:10],
        "recommendations": recommendations,
        "overall_assessment": "grounded" if score >= 4 else "needs_improvement",
    }


def assess_breakthrough_potential(idea: dict, target_venue: str) -> dict:
    combined = " ".join(
        str(idea.get(key, ""))
        for key in [
            "Title",
            "Abstract",
            "Short Hypothesis",
            "Hypothesis",
            "Impact",
            "Field",
            "Task",
        ]
    ).lower()
    markers = {
        "major_problem": [
            "grand challenge",
            "major challenge",
            "fundamental",
            "core problem",
        ],
        "broad_relevance": [
            "real-world",
            "broad impact",
            "cross-domain",
            "scientific discovery",
            "societal",
        ],
        "high_stakes": ["medical", "biology", "climate", "agriculture", "safety"],
        "novel_mechanism": ["novel", "new", "first", "adaptive", "dynamic", "unified"],
    }

    checks = {
        name: any(keyword in combined for keyword in keywords)
        for name, keywords in markers.items()
    }
    score = 1.5 + 0.8 * sum(checks.values())
    if target_venue == "nature":
        score += 0.4 * sum(
            checks[name] for name in ["major_problem", "broad_relevance", "high_stakes"]
        )
    score = max(1.0, min(5.0, score))

    recommendations = []
    if not checks["major_problem"]:
        recommendations.append("将问题 framing 提升到更重大、更基础的问题层面。")
    if not checks["broad_relevance"]:
        recommendations.append("更明确说明研究对更广泛领域或应用场景的意义。")
    if target_venue == "nature" and not checks["high_stakes"]:
        recommendations.append(
            "若目标是 Nature 风格投稿，需要更清晰强调高影响问题场景。"
        )

    return {
        "score": round(score, 2),
        "checks": checks,
        "recommendations": recommendations,
        "overall_assessment": (
            "breakthrough_oriented" if score >= 4 else "incremental_or_unclear"
        ),
    }


def build_claim_evidence_ledger(latex_content: str) -> list[dict]:
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        latex_content,
        re.DOTALL | re.IGNORECASE,
    )
    prefix, sections = _split_latex_sections(latex_content)

    source_blocks = [("abstract", abstract_match.group(1) if abstract_match else "")]
    for section in sections:
        title = section["title"].lower()
        if any(
            keyword in title
            for keyword in ["introduction", "result", "discussion", "conclusion"]
        ):
            source_blocks.append((title, section["content"]))
    claim_markers = [
        "outperform",
        "significant",
        "state-of-the-art",
        "sota",
        "novel",
        "we show",
        "we demonstrate",
        "improves",
        "improvement",
        "superior",
    ]

    ledger = []
    for section_name, block in source_blocks:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
        for sentence in sentences:
            lowered = sentence.lower()
            if not any(marker in lowered for marker in claim_markers):
                continue

            support_types = []
            if any(
                token in lowered for token in ["\\cite{", "prior work", "previous work"]
            ):
                support_types.append("citation")
            if any(token in lowered for token in ["figure", "fig.", "\\ref{fig"]):
                support_types.append("figure")
            if any(token in lowered for token in ["table", "tab.", "\\ref{tab"]):
                support_types.append("table")
            if not support_types:
                if "\\cite{" in latex_content:
                    support_types.append("citation_global")
                if "\\ref{fig" in latex_content or "figure" in latex_content.lower():
                    support_types.append("figure_global")
                if "\\ref{tab" in latex_content or "table" in latex_content.lower():
                    support_types.append("table_global")

            ledger.append(
                {
                    "section": section_name,
                    "claim": sentence,
                    "support_types": support_types,
                    "supported": len(support_types) > 0,
                    "linked_results": [],
                    "suggested_rewrite": None,
                }
            )
    return ledger


def link_claims_to_key_results(
    claim_ledger: list[dict], key_results: dict
) -> list[dict]:
    values = key_results.get("values", [])
    linked = []
    for item in claim_ledger:
        claim_text = item.get("claim", "")
        local_links = [value for value in values if value in claim_text][:5]
        updated = {**item, "linked_results": local_links}
        if not local_links and values:
            sample_values = ", ".join(values[:3])
            updated["suggested_rewrite"] = (
                f"Consider grounding this claim with one or more concrete numbers such as {sample_values}."
            )
        linked.append(updated)
    return linked


def assess_claim_alignment(claim_ledger: list[dict]) -> dict:
    if not claim_ledger:
        return {
            "score": 3.0,
            "linked_claims": 0,
            "total_claims": 0,
            "recommendations": [
                "未检测到核心 claims，建议在摘要/结果/结论中明确主要贡献。"
            ],
        }

    linked_claims = sum(1 for item in claim_ledger if item.get("linked_results"))
    total_claims = len(claim_ledger)
    ratio = linked_claims / total_claims if total_claims else 0
    score = 1.0 + 4.0 * ratio
    recommendations = []
    if ratio < 0.5:
        recommendations.append(
            "大部分核心 claims 缺少明确数值绑定，建议在摘要和结论中加入关键数字。"
        )
    elif ratio < 0.8:
        recommendations.append(
            "部分核心 claims 仍未与 strongest numerical results 对齐。"
        )

    return {
        "score": round(score, 2),
        "linked_claims": linked_claims,
        "total_claims": total_claims,
        "recommendations": recommendations,
    }


def _extract_env_blocks(latex_content: str, env_name: str) -> list[str]:
    pattern = re.compile(
        rf"\\begin\{{{env_name}\*?\}}(.*?)\\end\{{{env_name}\*?\}}",
        re.DOTALL | re.IGNORECASE,
    )
    return [match.group(0) for match in pattern.finditer(latex_content)]


def _extract_caption(block: str) -> str:
    match = re.search(r"\\caption\{(.*?)\}", block, re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _extract_label(block: str) -> str:
    match = re.search(r"\\label\{(.*?)\}", block)
    return match.group(1).strip() if match else ""


def build_evidence_pack(base_folder: str | Path, latex_content: str) -> dict:
    base_folder = Path(base_folder)
    figures = []
    for idx, block in enumerate(_extract_env_blocks(latex_content, "figure")):
        image_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(.*?)\}", block)
        label = _extract_label(block) or f"figure_{idx}"
        caption = _extract_caption(block)
        ref_count = len(re.findall(rf"\\ref\{{{re.escape(label)}\}}", latex_content))
        existing_images = [
            str((base_folder / "latex" / path).resolve())
            for path in image_paths
            if (base_folder / "latex" / path).exists()
        ]
        figures.append(
            {
                "type": "figure",
                "label": label,
                "caption": caption,
                "image_paths": image_paths,
                "existing_images": existing_images,
                "ref_count": ref_count,
            }
        )

    tables = []
    for idx, block in enumerate(_extract_env_blocks(latex_content, "table")):
        label = _extract_label(block) or f"table_{idx}"
        caption = _extract_caption(block)
        ref_count = len(re.findall(rf"\\ref\{{{re.escape(label)}\}}", latex_content))
        tables.append(
            {
                "type": "table",
                "label": label,
                "caption": caption,
                "ref_count": ref_count,
            }
        )

    strongest_results = sorted(
        figures + tables,
        key=lambda item: (item.get("ref_count", 0), len(item.get("caption", ""))),
        reverse=True,
    )[:5]

    unreferenced = [
        item["label"] for item in figures + tables if item.get("ref_count", 0) == 0
    ]
    recommendations = []
    if not strongest_results:
        recommendations.append("补充更清晰的核心 figure/table 证据来支撑主要结论。")
    if unreferenced:
        recommendations.append(
            "减少未被正文引用的 figure/table，或将其与主要论断建立明确联系。"
        )

    referenced_items = [
        item for item in figures + tables if item.get("ref_count", 0) > 0
    ]
    evidence_density_score = 1.0
    if strongest_results:
        evidence_density_score += 1.5
    if len(referenced_items) >= 2:
        evidence_density_score += 1.5
    elif len(referenced_items) == 1:
        evidence_density_score += 0.8
    if len(figures) + len(tables) >= 3:
        evidence_density_score += 1.0
    evidence_density_score = max(1.0, min(5.0, evidence_density_score))

    return {
        "figures": figures,
        "tables": tables,
        "num_figures": len(figures),
        "num_tables": len(tables),
        "evidence_density_score": evidence_density_score,
        "strongest_results": strongest_results,
        "unreferenced_labels": unreferenced,
        "recommendations": recommendations,
    }


def build_contribution_map(report: dict) -> list[dict]:
    claim_ledger = report.get("claim_ledger", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    key_results = report.get("key_results", {}).get("values", [])
    structured_results = report.get("key_results", {}).get("structured_results", [])
    limitations = report.get("claim_support", {}).get(
        "recommendations", []
    ) + report.get("rigor", {}).get("recommendations", [])

    supported_claims = [item for item in claim_ledger if item.get("supported")]
    if not supported_claims:
        supported_claims = claim_ledger[:3]

    contribution_map = []
    for idx, claim in enumerate(supported_claims[:3]):
        evidence_labels = []
        for result in strongest_results[:3]:
            label = result.get("label")
            if label:
                evidence_labels.append(label)

        contribution_map.append(
            {
                "index": idx + 1,
                "title": f"Contribution {idx + 1}",
                "claim": claim.get("claim"),
                "supported": claim.get("supported"),
                "evidence_labels": evidence_labels[:3],
                "key_results": (claim.get("linked_results") or key_results[:3])[:3],
                "structured_result_refs": structured_results[:2],
                "limitation": limitations[idx] if idx < len(limitations) else None,
            }
        )

    return contribution_map


def refine_contribution_map_with_llm(
    contribution_map: list[dict],
    report: dict,
    *,
    client,
    model: str,
) -> list[dict]:
    from ai_scientist.llm import get_response_from_llm

    if not contribution_map:
        return contribution_map

    prompt = f"""
请基于以下论文质量分析，优化 contribution map，使其更适合投稿稿叙事。

当前 contribution map:
{json.dumps(contribution_map, indent=2, ensure_ascii=False)}

Strongest results:
{json.dumps(report.get('evidence_pack', {}).get('strongest_results', []), indent=2, ensure_ascii=False)}

Key numerical results:
{json.dumps(report.get('key_results', {}).get('values', []), indent=2, ensure_ascii=False)}

Claim ledger:
{json.dumps(report.get('claim_ledger', []), indent=2, ensure_ascii=False)}

Weaknesses:
{json.dumps(report.get('professional', {}).get('overall', {}).get('weaknesses', []), indent=2, ensure_ascii=False)}

要求:
1. 产出 2-3 条最核心贡献。
2. 每条贡献都要更像投稿稿中的 contribution statement。
3. 明确关联 strongest evidence 和 key numerical results。
4. 如果证据不足，不要写太强的 claim。

返回 JSON 数组，每项格式:
{{
  "title": "Contribution 1",
  "claim": "...",
  "evidence_labels": ["fig:main"],
  "key_results": ["84.7", "5.2%"],
  "limitation": "..."
}}
"""

    response, _ = get_response_from_llm(
        prompt=prompt,
        client=client,
        model=model,
        system_message="你是资深学术写作专家，负责把 contribution map 精炼成更适合 NeurIPS / Nature 投稿稿的版本。",
        temperature=0.25,
    )
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL) or re.search(
        r"\[.*\]", response, re.DOTALL
    )
    if not match:
        return contribution_map
    try:
        parsed = json.loads(match.group(1))
        if isinstance(parsed, list) and parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return contribution_map


def extract_key_results(base_folder: str | Path, latex_content: str) -> dict:
    base_folder = Path(base_folder)
    numeric_patterns = [
        r"\b\d+\.\d+\b",
        r"\b\d+%\b",
        r"\b\d+\.\d+%\b",
    ]

    sources = []
    structured_results = []
    summary_filenames = [
        "baseline_summary.json",
        "research_summary.json",
        "ablation_summary.json",
    ]
    seen = set()
    for run_dir in iter_bfts_run_dirs(base_folder, logs_subdir="logs", descending=True):
        for filename in summary_filenames:
            if filename in seen:
                continue
            path = run_dir / filename
            if not path.exists():
                continue
            try:
                raw_text = path.read_text(encoding="utf-8", errors="ignore")
                sources.append(raw_text)
                summary_json = json.loads(raw_text)
                structured_results.extend(
                    _extract_structured_results(summary_json, source_name=path.name)
                )
                seen.add(filename)
            except OSError:
                continue
            except json.JSONDecodeError:
                continue
        if len(seen) == len(summary_filenames):
            break
    sources.append(latex_content)

    values = []
    for source in sources:
        for pattern in numeric_patterns:
            values.extend(re.findall(pattern, source))

    dedup_values = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            dedup_values.append(value)

    key_results = dedup_values[:20]
    recommendations = []
    if len(key_results) < 3:
        recommendations.append(
            "关键数值结果过少，建议在摘要、结果和结论中明确写出核心数字。"
        )

    return {
        "values": key_results,
        "count": len(key_results),
        "structured_results": structured_results[:20],
        "recommendations": recommendations,
    }


def _extract_structured_results(obj, *, source_name: str, path: str = "") -> list[dict]:
    results = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            lower_key = key.lower()
            if lower_key in {
                "key_numerical_results",
                "best_metric",
                "metric",
                "final_metrics",
            }:
                results.append(
                    {
                        "source": source_name,
                        "path": current_path,
                        "value": value,
                    }
                )
            results.extend(
                _extract_structured_results(
                    value, source_name=source_name, path=current_path
                )
            )
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            current_path = f"{path}[{idx}]"
            results.extend(
                _extract_structured_results(
                    item, source_name=source_name, path=current_path
                )
            )
    return results


def evaluate_quality_bundle(
    base_folder: str | Path,
    *,
    paper_type: str,
    quality_model: str,
    target_venue: Optional[str] = None,
) -> dict:
    from ai_scientist.professional_writing_system import ProfessionalPaperEvaluator
    from ai_scientist.writing_strategies import WritingQualityAssessor

    idea = _load_idea(base_folder)
    latex_path, latex_content = _load_latex(base_folder)
    if latex_path is None:
        return {"status": "failed", "reason": "latex/template.tex not found"}

    venue = target_venue or recommend_target_venue_from_idea(idea, paper_type)
    venue_config = VENUE_PRESETS.get(venue, VENUE_PRESETS["neurips"])
    template = venue_config["template"]
    evaluator = ProfessionalPaperEvaluator(template=template, model=quality_model)
    professional = evaluator.evaluate_paper_quality(latex_content, idea)
    structural = WritingQualityAssessor.assess_paper_quality(str(base_folder))
    rigor = assess_experiment_rigor(base_folder, latex_content)
    evidence_pack = build_evidence_pack(base_folder, latex_content)
    key_results = extract_key_results(base_folder, latex_content)
    claim_ledger = link_claims_to_key_results(
        build_claim_evidence_ledger(latex_content), key_results
    )
    claim_support = assess_claim_support(latex_content)
    numeric_coverage = assess_numeric_coverage(latex_content)
    claim_alignment = assess_claim_alignment(claim_ledger)
    breakthrough_profile = assess_breakthrough_potential(idea, venue)
    contribution_map = build_contribution_map(
        {
            "claim_ledger": claim_ledger,
            "evidence_pack": evidence_pack,
            "key_results": key_results,
            "claim_support": claim_support,
            "rigor": rigor,
        }
    )

    return {
        "status": "success",
        "template": template,
        "target_venue": venue,
        "venue_guidance": venue_config["style_guidance"],
        "professional": professional,
        "structural": structural,
        "rigor": rigor,
        "claim_support": claim_support,
        "numeric_coverage": numeric_coverage,
        "claim_alignment": claim_alignment,
        "breakthrough_profile": breakthrough_profile,
        "claim_ledger": claim_ledger,
        "evidence_pack": evidence_pack,
        "key_results": key_results,
        "contribution_map": contribution_map,
    }


def _build_rewrite_feedback(report: dict) -> dict:
    professional = report.get("professional", {})
    structural = report.get("structural", {})
    rigor = report.get("rigor", {})

    weaknesses = list(professional.get("overall", {}).get("weaknesses", []))
    recommendations = list(professional.get("overall", {}).get("recommendations", []))
    recommendations.extend(structural.get("recommendations", []))
    recommendations.extend(rigor.get("recommendations", []))
    recommendations.extend(report.get("claim_support", {}).get("recommendations", []))
    recommendations.extend(report.get("claim_alignment", {}).get("recommendations", []))
    recommendations.extend(
        report.get("numeric_coverage", {}).get("recommendations", [])
    )
    recommendations.extend(
        report.get("breakthrough_profile", {}).get("recommendations", [])
    )

    scores = {
        "Originality": round(professional.get("innovation", {}).get("score", 3)),
        "Quality": round(professional.get("content", {}).get("score", 3)),
        "Clarity": round(professional.get("clarity", {}).get("score", 3)),
        "Significance": round(professional.get("innovation", {}).get("score", 3)),
        "Soundness": round(
            min(professional.get("rigor", {}).get("score", 3), rigor.get("score", 3))
        ),
        "Presentation": round(professional.get("professionalism", {}).get("score", 3)),
        "Contribution": round(
            (
                professional.get("innovation", {}).get("score", 3)
                + professional.get("content", {}).get("score", 3)
            )
            / 2
        ),
        "Overall": round(professional.get("overall", {}).get("score", 3) * 2),
        "Confidence": 4,
    }

    if not weaknesses and structural.get("assessments"):
        for dimension, result in structural["assessments"].items():
            if result.get("issues"):
                weaknesses.extend(result["issues"][:2])

    return {
        "review": {
            "scores": scores,
            "Weaknesses": weaknesses,
            "Questions": [],
            "Limitations": recommendations[:5],
        }
    }


def _build_submission_scorecard(
    report: dict,
    readiness: dict,
    *,
    target_venue: str,
    quality_threshold: float,
    rigor_threshold: float,
    claim_support_threshold: float,
) -> dict:
    def _metric(score, threshold):
        return {
            "score": round(score, 2),
            "threshold": round(threshold, 2),
            "pass": score >= threshold,
            "gap": round(max(0.0, threshold - score), 2),
        }

    breakthrough_threshold = 4.0 if target_venue == "nature" else 3.5
    evidence_threshold = 3.0
    contribution_count = len(report.get("contribution_map", []))
    return {
        "quality": _metric(
            report.get("professional", {}).get("overall", {}).get("score", 0),
            quality_threshold,
        ),
        "rigor": _metric(report.get("rigor", {}).get("score", 0), rigor_threshold),
        "claim_support": _metric(
            report.get("claim_support", {}).get("score", 0), claim_support_threshold
        ),
        "claim_alignment": _metric(
            report.get("claim_alignment", {}).get("score", 0), 3.5
        ),
        "numeric_coverage": _metric(
            report.get("numeric_coverage", {}).get("score", 0), 3.5
        ),
        "breakthrough": _metric(
            report.get("breakthrough_profile", {}).get("score", 0),
            breakthrough_threshold,
        ),
        "evidence_density": _metric(
            report.get("evidence_pack", {}).get("evidence_density_score", 0),
            evidence_threshold,
        ),
        "contributions": {
            "score": contribution_count,
            "threshold": 2,
            "pass": contribution_count >= 2,
            "gap": max(0, 2 - contribution_count),
        },
        "submission": {
            "score": 1.0 if readiness.get("ready") else 0.0,
            "threshold": 1.0,
            "pass": readiness.get("ready", False),
            "gap": 0.0 if readiness.get("ready") else 1.0,
        },
    }


def _build_submission_priority_profile(
    scorecard: dict,
    readiness: dict,
    revision_actions: list[dict],
    *,
    target_venue: str,
    gate_passed: bool,
    unsupported_claims_count: int,
) -> dict:
    weights = {
        "quality": 18,
        "rigor": 16,
        "claim_support": 14,
        "claim_alignment": 10,
        "numeric_coverage": 10,
        "breakthrough": 12 if target_venue == "nature" else 8,
        "evidence_density": 8,
        "contributions": 8,
        "submission": 8,
    }

    weighted_score = 0.0
    max_score = float(sum(weights.values())) or 1.0
    gap_items = []
    for name, weight in weights.items():
        metric = scorecard.get(name, {}) if isinstance(scorecard, dict) else {}
        score = metric.get("score", 0) if isinstance(metric, dict) else 0
        threshold = metric.get("threshold", 1) if isinstance(metric, dict) else 1
        if threshold in (None, 0):
            threshold = 1
        ratio = max(0.0, min(float(score) / float(threshold), 1.0))
        weighted_score += weight * ratio
        gap_items.append(
            (metric.get("gap", 0) if isinstance(metric, dict) else 0, name)
        )

    blocker_count = len(readiness.get("blockers") or [])
    p0_actions = sum(item.get("priority") == "P0" for item in revision_actions)
    p1_actions = sum(item.get("priority") == "P1" for item in revision_actions)
    categories = readiness.get("categories") or {}
    category_penalty = sum(1 for value in categories.values() if value)
    penalty = (
        blocker_count * 2.5
        + p0_actions * 4.0
        + p1_actions * 1.5
        + min(unsupported_claims_count, 5) * 0.8
        + category_penalty * 0.8
    )
    bonus = (6.0 if gate_passed else 0.0) + (8.0 if readiness.get("ready") else 0.0)

    score = max(
        0.0,
        min(100.0, round((weighted_score / max_score) * 100.0 - penalty + bonus, 2)),
    )
    if readiness.get("ready") and gate_passed and score >= 88:
        tier = "submit_now"
    elif score >= 75:
        tier = "near_ready"
    elif score >= 60:
        tier = "promising_but_revise"
    else:
        tier = "hold_for_major_revision"

    reasons = []
    if readiness.get("ready"):
        reasons.append("submission readiness is already marked ready")
    elif blocker_count:
        reasons.append(
            f"{blocker_count} blocker(s) still need to be cleared before submission"
        )
    if gate_passed:
        reasons.append("quality gate is already passed")
    if p0_actions:
        reasons.append(f"{p0_actions} P0 revision action(s) remain")
    for gap, name in sorted(gap_items, reverse=True):
        if gap and len(reasons) < 6:
            reasons.append(f"largest remaining gap: {name} ({gap:.2f})")
    if unsupported_claims_count and len(reasons) < 6:
        reasons.append(f"unsupported claims still detected: {unsupported_claims_count}")

    return {
        "score": score,
        "tier": tier,
        "blocker_count": blocker_count,
        "critical_revision_actions_count": p0_actions,
        "reasons": reasons[:6],
    }


def _recommend_revision_actions(
    report: dict, readiness: dict, *, target_venue: str
) -> list[dict]:
    actions = []

    def add(priority: str, focus: str, action: str, reason: str):
        signature = (focus, action)
        if signature in {(item.get("focus"), item.get("action")) for item in actions}:
            return
        actions.append(
            {
                "priority": priority,
                "focus": focus,
                "action": action,
                "reason": reason,
            }
        )

    categories = readiness.get("categories") or {}
    overall_weaknesses = (
        report.get("professional", {}).get("overall", {}).get("weaknesses", [])
    )
    unsupported_claims = report.get("claim_support", {}).get("unsupported_claims", [])
    strongest_results = report.get("evidence_pack", {}).get("strongest_results", [])
    key_results = report.get("key_results", {}).get("values", [])
    contribution_map = report.get("contribution_map", [])

    if categories.get("quality") or overall_weaknesses:
        add(
            "P0",
            "Narrative quality",
            "Tighten title, abstract, and introduction around 2–3 concrete contributions and remove overstated framing.",
            (
                overall_weaknesses[0]
                if overall_weaknesses
                else "overall quality blockers remain active"
            ),
        )
    if categories.get("rigor") or report.get("rigor", {}).get("score", 0) < 3.8:
        add(
            "P0",
            "Experimental rigor",
            "Strengthen baseline, ablation, significance, and methodology details before claiming maturity.",
            (
                report.get("rigor", {}).get("recommendations")
                or ["rigor score is still below a comfortable submission bar"]
            )[0],
        )
    if categories.get("claim") or unsupported_claims:
        claim_example = (
            unsupported_claims[0]
            if unsupported_claims
            else "core claims are not yet tightly linked to evidence"
        )
        add(
            "P0",
            "Claim support",
            "Rewrite unsupported claims so each major statement points to a figure/table and, where possible, a concrete number.",
            claim_example,
        )
    if (
        categories.get("numeric")
        or report.get("numeric_coverage", {}).get("score", 0) < 3.5
    ):
        add(
            "P1",
            "Numerical specificity",
            "Inject the strongest numerical takeaways into the abstract, results, and conclusion with explicit comparisons.",
            key_results[0] if key_results else "numeric coverage is below target",
        )
    if (
        categories.get("evidence")
        or report.get("evidence_pack", {}).get("evidence_density_score", 0) < 3.0
    ):
        lead_evidence = (
            strongest_results[0].get("label")
            if strongest_results
            else "result figures/tables"
        )
        add(
            "P1",
            "Evidence packaging",
            "Surface the strongest figures/tables earlier in the results narrative and ensure every key visual is referenced in text.",
            f"lead evidence candidate: {lead_evidence}",
        )
    if len(contribution_map) < 2 or categories.get("contribution"):
        add(
            "P1",
            "Contribution framing",
            "Make the contribution list explicit and map each contribution to one strongest result and one limitation.",
            "contribution map is still too thin for a confident submission story",
        )
    if target_venue == "nature" and (
        categories.get("breakthrough")
        or report.get("breakthrough_profile", {}).get("score", 0) < 4.0
    ):
        add(
            "P1",
            "Broad significance",
            "Raise the problem framing and cross-domain significance while keeping claims highly evidence-backed and restrained.",
            "Nature-style submissions need a stronger breakthrough and significance story",
        )
    if categories.get("venue_fit"):
        add(
            "P2",
            "Venue fit",
            "Revisit paper type and venue fit, or narrow the contribution scope to better match the target outlet.",
            "current paper type or narrative fit remains weak for the target venue",
        )

    for blocker in readiness.get("blockers", [])[:2]:
        add(
            "P2",
            "Open blocker",
            blocker.capitalize() + ".",
            "derived directly from submission blockers",
        )

    return actions[:6]


def _attach_submission_context(
    report: dict,
    *,
    paper_type: str,
    target_venue: str,
    quality_threshold: float,
    rigor_threshold: float,
    claim_support_threshold: float,
    autonomous_followup_focus: Optional[dict[str, Any]] = None,
    research_root: str | Path | None = None,
) -> dict:
    active_research_root = (
        Path(research_root).expanduser()
        if research_root is not None
        else resolve_output_path()
    )
    readiness = _build_submission_readiness(
        report,
        paper_type=paper_type,
        target_venue=target_venue,
        quality_threshold=quality_threshold,
        rigor_threshold=rigor_threshold,
        claim_support_threshold=claim_support_threshold,
    )
    from ai_scientist.utils.submission_history import (
        recommend_reviewer_risk_mitigation,
        recommend_rewrite_efficiency_controls,
        recommend_rewrite_focus_adjustments,
        recommend_rewrite_style_preferences,
    )

    report["submission_readiness_preview"] = readiness
    report["revision_actions"] = _recommend_revision_actions(
        report, readiness, target_venue=target_venue
    )
    report["historical_rewrite_focus"] = recommend_rewrite_focus_adjustments(
        target_venue,
        research_root=active_research_root,
    )
    report["historical_reviewer_risks"] = recommend_reviewer_risk_mitigation(
        target_venue,
        research_root=active_research_root,
    )
    report["historical_style_preferences"] = recommend_rewrite_style_preferences(
        target_venue,
        research_root=active_research_root,
    )
    report["historical_rewrite_efficiency"] = recommend_rewrite_efficiency_controls(
        target_venue,
        research_root=active_research_root,
    )
    report["submission_scorecard"] = _build_submission_scorecard(
        report,
        readiness,
        target_venue=target_venue,
        quality_threshold=quality_threshold,
        rigor_threshold=rigor_threshold,
        claim_support_threshold=claim_support_threshold,
    )
    report["submission_priority_preview"] = _build_submission_priority_profile(
        report["submission_scorecard"],
        readiness,
        report["revision_actions"],
        target_venue=target_venue,
        gate_passed=False,
        unsupported_claims_count=len(
            report.get("claim_support", {}).get("unsupported_claims", [])
        ),
    )
    return _apply_autonomous_followup_focus(report, autonomous_followup_focus)


def _needs_rewrite(
    report: dict, quality_threshold: float, rigor_threshold: float
) -> bool:
    professional_score = (
        report.get("professional", {}).get("overall", {}).get("score", 0)
    )
    structural_score = report.get("structural", {}).get("overall_quality", 0)
    rigor_score = report.get("rigor", {}).get("score", 0)
    claim_support_score = report.get("claim_support", {}).get("score", 0)
    claim_alignment_score = report.get("claim_alignment", {}).get("score", 0)
    numeric_coverage_score = report.get("numeric_coverage", {}).get("score", 0)
    breakthrough_score = report.get("breakthrough_profile", {}).get("score", 0)
    return (
        professional_score < quality_threshold
        or structural_score < quality_threshold
        or rigor_score < rigor_threshold
        or claim_support_score < 3.5
        or claim_alignment_score < 3.5
        or numeric_coverage_score < 3.5
        or (breakthrough_score < 3.5 and report.get("target_venue") == "nature")
    )


def _build_submission_readiness(
    report: dict,
    *,
    paper_type: str,
    target_venue: str,
    quality_threshold: float,
    rigor_threshold: float,
    claim_support_threshold: float,
) -> dict:
    professional = report.get("professional", {})
    overall = professional.get("overall", {})
    quality_score = overall.get("score", 0)
    rigor_score = report.get("rigor", {}).get("score", 0)
    claim_support_score = report.get("claim_support", {}).get("score", 0)
    claim_alignment_score = report.get("claim_alignment", {}).get("score", 0)
    numeric_coverage_score = report.get("numeric_coverage", {}).get("score", 0)
    breakthrough_score = report.get("breakthrough_profile", {}).get("score", 0)
    evidence_pack = report.get("evidence_pack", {})
    contribution_count = len(report.get("contribution_map", []))
    evidence_density_score = evidence_pack.get("evidence_density_score", 0)

    blockers = []
    if quality_score < quality_threshold:
        blockers.append(
            f"overall quality below target ({quality_score:.2f} < {quality_threshold:.2f})"
        )
    if rigor_score < rigor_threshold:
        blockers.append(
            f"rigor below target ({rigor_score:.2f} < {rigor_threshold:.2f})"
        )
    if claim_support_score < claim_support_threshold:
        blockers.append(
            f"claim support below target ({claim_support_score:.2f} < {claim_support_threshold:.2f})"
        )
    if claim_alignment_score < 3.5:
        blockers.append(
            f"claim alignment below target ({claim_alignment_score:.2f} < 3.50)"
        )
    if numeric_coverage_score < 3.5:
        blockers.append(
            f"numeric coverage below target ({numeric_coverage_score:.2f} < 3.50)"
        )
    if contribution_count < 2:
        blockers.append(
            f"too few clearly articulated contributions ({contribution_count} < 2)"
        )
    if target_venue == "nature" and breakthrough_score < 4.0:
        blockers.append(
            f"breakthrough potential below Nature-style bar ({breakthrough_score:.2f} < 4.00)"
        )
    if evidence_pack.get("num_figures", 0) + evidence_pack.get("num_tables", 0) == 0:
        blockers.append("no figure/table evidence extracted from manuscript")
    if evidence_pack.get("unreferenced_labels"):
        blockers.append("some figures/tables are not referenced from the main text")
    if evidence_density_score < 3.0:
        blockers.append(
            f"evidence density below target ({evidence_density_score:.2f} < 3.00)"
        )

    venue_fit = {
        "neurips": {"normal"},
        "iclr": {"normal", "icbinb", "extended"},
        "cvpr": {"normal"},
        "journal": {"journal", "normal"},
        "nature": {"journal", "normal"},
    }
    if paper_type not in venue_fit.get(target_venue, {paper_type}):
        blockers.append(
            f"paper_type '{paper_type}' is a weak fit for target venue '{target_venue}'"
        )

    for weakness in overall.get("weaknesses", [])[:5]:
        blockers.append(f"review weakness: {weakness}")
    for recommendation in report.get("rigor", {}).get("recommendations", [])[:3]:
        blockers.append(f"rigor action: {recommendation}")
    for recommendation in report.get("claim_support", {}).get("recommendations", [])[
        :3
    ]:
        blockers.append(f"claim action: {recommendation}")

    blockers = blockers[:10]
    categories = {
        "quality": sum(
            "quality" in blocker or "review weakness" in blocker for blocker in blockers
        ),
        "rigor": sum("rigor" in blocker for blocker in blockers),
        "claim": sum("claim" in blocker for blocker in blockers),
        "numeric": sum("numeric" in blocker for blocker in blockers),
        "evidence": sum(
            "figure/table" in blocker or "evidence" in blocker for blocker in blockers
        ),
        "contribution": sum("contribution" in blocker for blocker in blockers),
        "breakthrough": sum(
            "Nature-style" in blocker or "breakthrough" in blocker
            for blocker in blockers
        ),
        "venue_fit": sum(
            "paper_type" in blocker or "venue" in blocker for blocker in blockers
        ),
    }
    ready = len(blockers) == 0
    return {
        "target_venue": target_venue,
        "ready": ready,
        "status": "ready" if ready else "needs_work",
        "blockers": blockers,
        "categories": categories,
    }


def run_high_quality_pass(
    base_folder: str | Path,
    *,
    paper_type: str,
    rewrite_model: str,
    quality_model: Optional[str] = None,
    target_venue: Optional[str] = None,
    quality_threshold: float = 4.0,
    rigor_threshold: float = 3.5,
    max_rewrite_rounds: int = 1,
    auto_improvement_fallback: bool = True,
    autonomous_followup_focus: Optional[dict[str, Any]] = None,
    resume: bool = True,
    logger: Callable[[str], None] = print,
) -> dict:
    base_folder = Path(base_folder)
    active_research_root = resolve_output_path()
    quality_dir = base_folder / "quality"
    result_file = quality_dir / "high_quality_result.json"
    applied_followup_focus = (
        dict(autonomous_followup_focus)
        if isinstance(autonomous_followup_focus, dict) and autonomous_followup_focus
        else None
    )
    venue = target_venue or recommend_target_venue_from_idea(
        _load_idea(base_folder), paper_type
    )
    venue_config = VENUE_PRESETS.get(venue, VENUE_PRESETS["neurips"])

    quality_threshold = max(quality_threshold, venue_config["quality_threshold"])
    rigor_threshold = max(rigor_threshold, venue_config["rigor_threshold"])
    claim_support_threshold = venue_config["claim_support_threshold"]
    if applied_followup_focus:
        _safe_json_dump(
            quality_dir / "autonomous_followup_focus.json",
            applied_followup_focus,
        )

    if resume and result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            existing_result = json.load(f)
        if should_resume_high_quality_result(
            existing_result,
            auto_improvement_fallback=auto_improvement_fallback,
            target_venue=venue,
            quality_threshold=quality_threshold,
            rigor_threshold=rigor_threshold,
        ):
            return existing_result
        logger(
            "High-quality pass: cached result settings mismatch, rerunning with current fallback discipline"
        )

    quality_model = quality_model or rewrite_model
    initial_report = evaluate_quality_bundle(
        base_folder,
        paper_type=paper_type,
        quality_model=quality_model,
        target_venue=venue,
    )
    _safe_json_dump(quality_dir / "assessment_initial.json", initial_report)
    _safe_json_dump(
        quality_dir / "claim_ledger_initial.json",
        {"venue": venue, "claims": initial_report.get("claim_ledger", [])},
    )

    if initial_report.get("status") != "success":
        result = {
            "status": "failed",
            "reason": initial_report.get("reason", "quality evaluation failed"),
        }
        _safe_json_dump(result_file, result)
        return result

    rewrite_applied = False
    current_report = initial_report
    latex_path, current_latex = _load_latex(base_folder)
    if latex_path is None:
        result = {"status": "failed", "reason": "latex/template.tex not found"}
        _safe_json_dump(result_file, result)
        return result

    from ai_scientist.llm import create_client, get_response_from_llm

    client, client_model = create_client(rewrite_model)
    rank_client, rank_client_model = create_client(quality_model)
    current_report["contribution_map"] = refine_contribution_map_with_llm(
        current_report.get("contribution_map", []),
        current_report,
        client=rank_client,
        model=rank_client_model,
    )
    current_report = _attach_submission_context(
        current_report,
        paper_type=paper_type,
        target_venue=venue,
        quality_threshold=quality_threshold,
        rigor_threshold=rigor_threshold,
        claim_support_threshold=claim_support_threshold,
        autonomous_followup_focus=applied_followup_focus,
        research_root=active_research_root,
    )
    rewrite_feedback = _build_rewrite_feedback(current_report)
    _safe_json_dump(quality_dir / "rewrite_feedback.json", rewrite_feedback)
    rewrite_trace = []
    efficiency_controls = current_report.get("historical_rewrite_efficiency", {}) or {}
    effective_max_rewrite_rounds = max(
        1,
        max_rewrite_rounds
        + int(efficiency_controls.get("rewrite_round_adjustment", 0)),
    )
    effective_candidate_boost = venue_config.get("candidate_boost", 0) + int(
        (applied_followup_focus or {}).get("candidate_boost") or 0
    )
    force_frontmatter_rewrite = bool(
        (applied_followup_focus or {}).get("frontmatter_required")
    )
    if applied_followup_focus:
        logger(
            "High-quality pass: applying autonomous follow-up focus "
            f"areas={(applied_followup_focus.get('focus_areas') or [])}, "
            f"preferred_sections={(applied_followup_focus.get('preferred_sections') or [])}, "
            f"candidate_boost={effective_candidate_boost}"
        )
    logger(
        f"High-quality pass: effective_max_rewrite_rounds={effective_max_rewrite_rounds}; rewrite_efficiency={efficiency_controls.get('rationale', [])}"
    )

    for round_idx in range(effective_max_rewrite_rounds):
        if not _needs_rewrite(current_report, quality_threshold, rigor_threshold):
            break

        logger(
            f"High-quality pass: rewrite round {round_idx + 1}/{effective_max_rewrite_rounds}"
        )
        improved_latex, frontmatter_plan = _apply_frontmatter_rewrites(
            current_latex,
            current_report,
            client=client,
            model=client_model,
            rank_client=rank_client,
            rank_model=rank_client_model,
            threshold=quality_threshold,
            candidate_boost=effective_candidate_boost,
            force_rewrite=force_frontmatter_rewrite,
            venue_guidance=venue_config["style_guidance"],
            logger=logger,
        )
        improved_latex, rewrite_plan = _apply_targeted_section_rewrites(
            improved_latex,
            current_report,
            client=client,
            model=client_model,
            rank_client=rank_client,
            rank_model=rank_client_model,
            threshold=quality_threshold,
            candidate_boost=effective_candidate_boost,
            venue_guidance=venue_config["style_guidance"],
            logger=logger,
        )
        if not improved_latex or improved_latex == current_latex:
            break

        rewrite_applied = True
        round_trace = {
            "round": round_idx + 1,
            "pre_quality_score": current_report.get("professional", {})
            .get("overall", {})
            .get("score", 0),
            "pre_submission_priority_score": current_report.get(
                "submission_priority_preview", {}
            ).get("score"),
            "frontmatter": frontmatter_plan,
            **rewrite_plan,
        }
        rewrite_trace.append(round_trace)
        backup_path = (
            latex_path.parent / f"template_quality_backup_round{round_idx + 1}.tex"
        )
        shutil.copy(latex_path, backup_path)
        latex_path.write_text(improved_latex, encoding="utf-8")
        current_latex = improved_latex

        quality_pdf = (
            base_folder
            / f"{base_folder.name}_reflection_quality_final_round{round_idx + 1}.pdf"
        )
        compile_latex(latex_path.parent, quality_pdf, timeout=60, verbose=False)
        current_report = evaluate_quality_bundle(
            base_folder,
            paper_type=paper_type,
            quality_model=quality_model,
            target_venue=venue,
        )
        current_report["contribution_map"] = refine_contribution_map_with_llm(
            current_report.get("contribution_map", []),
            current_report,
            client=rank_client,
            model=rank_client_model,
        )
        current_report = _attach_submission_context(
            current_report,
            paper_type=paper_type,
            target_venue=venue,
            quality_threshold=quality_threshold,
            rigor_threshold=rigor_threshold,
            claim_support_threshold=claim_support_threshold,
            autonomous_followup_focus=applied_followup_focus,
        )
        rewrite_trace[-1]["post_quality_score"] = (
            current_report.get("professional", {}).get("overall", {}).get("score", 0)
        )
        rewrite_trace[-1]["post_submission_priority_score"] = current_report.get(
            "submission_priority_preview", {}
        ).get("score")
        _safe_json_dump(
            quality_dir / f"assessment_round_{round_idx + 1}.json", current_report
        )
        _safe_json_dump(
            quality_dir / f"claim_ledger_round_{round_idx + 1}.json",
            {"venue": venue, "claims": current_report.get("claim_ledger", [])},
        )

    final_pdf = find_best_pdf_path(base_folder, prefer_reflections=True)
    final_quality_score = (
        current_report.get("professional", {}).get("overall", {}).get("score", 0)
    )
    final_rigor_score = current_report.get("rigor", {}).get("score", 0)
    final_claim_support_score = current_report.get("claim_support", {}).get("score", 0)
    final_claim_alignment_score = current_report.get("claim_alignment", {}).get(
        "score", 0
    )
    final_numeric_coverage_score = current_report.get("numeric_coverage", {}).get(
        "score", 0
    )
    gate_passed = (
        final_quality_score >= quality_threshold
        and final_rigor_score >= rigor_threshold
        and final_claim_support_score >= claim_support_threshold
        and final_claim_alignment_score >= 3.5
        and final_numeric_coverage_score >= 3.5
    )
    readiness = _build_submission_readiness(
        current_report,
        paper_type=paper_type,
        target_venue=venue,
        quality_threshold=quality_threshold,
        rigor_threshold=rigor_threshold,
        claim_support_threshold=claim_support_threshold,
    )
    from ai_scientist.utils.submission_history import (
        recommend_reviewer_risk_mitigation,
        recommend_rewrite_efficiency_controls,
        recommend_rewrite_focus_adjustments,
        recommend_rewrite_style_preferences,
    )

    current_report["submission_readiness_preview"] = readiness
    current_report["revision_actions"] = _recommend_revision_actions(
        current_report, readiness, target_venue=venue
    )
    current_report["historical_rewrite_focus"] = recommend_rewrite_focus_adjustments(
        venue, research_root=active_research_root
    )
    current_report["historical_reviewer_risks"] = recommend_reviewer_risk_mitigation(
        venue, research_root=active_research_root
    )
    current_report["historical_style_preferences"] = (
        recommend_rewrite_style_preferences(
            venue,
            research_root=active_research_root,
        )
    )
    current_report["historical_rewrite_efficiency"] = (
        recommend_rewrite_efficiency_controls(
            venue,
            research_root=active_research_root,
        )
    )
    current_report["submission_scorecard"] = _build_submission_scorecard(
        current_report,
        readiness,
        target_venue=venue,
        quality_threshold=quality_threshold,
        rigor_threshold=rigor_threshold,
        claim_support_threshold=claim_support_threshold,
    )
    current_report = _apply_autonomous_followup_focus(
        current_report, applied_followup_focus
    )

    auto_improvement_result = None
    auto_improvement_fallback_used = False
    if auto_improvement_fallback and not gate_passed:
        auto_improvement_fallback_used = True
        logger("High-quality pass: gate not passed, running auto-improvement fallback")
        from ai_scientist.perform_auto_improvement import improve_paper_with_review

        synthetic_review = _build_rewrite_feedback(current_report)
        auto_improvement_result = improve_paper_with_review(
            paper_dir=str(base_folder),
            text_review=synthetic_review,
            img_review={"figure_reviews": []},
            model=rewrite_model,
            max_rounds=1,
            target_venue=venue,
        )
        _safe_json_dump(
            quality_dir / "auto_improvement_result.json", auto_improvement_result
        )
        current_report = evaluate_quality_bundle(
            base_folder,
            paper_type=paper_type,
            quality_model=quality_model,
            target_venue=venue,
        )
        current_report["contribution_map"] = refine_contribution_map_with_llm(
            current_report.get("contribution_map", []),
            current_report,
            client=rank_client,
            model=rank_client_model,
        )
        current_report = _attach_submission_context(
            current_report,
            paper_type=paper_type,
            target_venue=venue,
            quality_threshold=quality_threshold,
            rigor_threshold=rigor_threshold,
            claim_support_threshold=claim_support_threshold,
            autonomous_followup_focus=applied_followup_focus,
        )
        final_quality_score = (
            current_report.get("professional", {}).get("overall", {}).get("score", 0)
        )
        final_rigor_score = current_report.get("rigor", {}).get("score", 0)
        final_claim_support_score = current_report.get("claim_support", {}).get(
            "score", 0
        )
        final_claim_alignment_score = current_report.get("claim_alignment", {}).get(
            "score", 0
        )
        final_numeric_coverage_score = current_report.get("numeric_coverage", {}).get(
            "score", 0
        )
        readiness = _build_submission_readiness(
            current_report,
            paper_type=paper_type,
            target_venue=venue,
            quality_threshold=quality_threshold,
            rigor_threshold=rigor_threshold,
            claim_support_threshold=claim_support_threshold,
        )
        current_report["submission_readiness_preview"] = readiness
        current_report["revision_actions"] = _recommend_revision_actions(
            current_report, readiness, target_venue=venue
        )
        current_report["submission_scorecard"] = _build_submission_scorecard(
            current_report,
            readiness,
            target_venue=venue,
            quality_threshold=quality_threshold,
            rigor_threshold=rigor_threshold,
            claim_support_threshold=claim_support_threshold,
        )
        current_report = _apply_autonomous_followup_focus(
            current_report, applied_followup_focus
        )
        gate_passed = (
            final_quality_score >= quality_threshold
            and final_rigor_score >= rigor_threshold
            and final_claim_support_score >= claim_support_threshold
            and final_claim_alignment_score >= 3.5
            and final_numeric_coverage_score >= 3.5
        )
    current_report["auto_improvement_fallback_used"] = auto_improvement_fallback_used
    current_report["auto_improvement_fallback_enabled"] = bool(
        auto_improvement_fallback
    )
    current_report["auto_improvement_fallback_reason"] = (
        "gate_not_passed" if auto_improvement_fallback_used else None
    )
    priority_profile = _build_submission_priority_profile(
        current_report.get("submission_scorecard", {}),
        readiness,
        current_report.get("revision_actions", []),
        target_venue=venue,
        gate_passed=gate_passed,
        unsupported_claims_count=len(
            current_report.get("claim_support", {}).get("unsupported_claims", [])
        ),
    )
    summary_text = (
        f"# High Quality Summary\n\n"
        f"- Quality score before: {initial_report.get('professional', {}).get('overall', {}).get('score', 0):.2f}\n"
        f"- Quality score after: {final_quality_score:.2f}\n"
        f"- Rigor score before: {initial_report.get('rigor', {}).get('score', 0):.2f}\n"
        f"- Rigor score after: {final_rigor_score:.2f}\n"
        f"- Claim support before: {initial_report.get('claim_support', {}).get('score', 0):.2f}\n"
        f"- Claim support after: {final_claim_support_score:.2f}\n"
        f"- Claim alignment after: {final_claim_alignment_score:.2f}\n"
        f"- Numeric coverage after: {final_numeric_coverage_score:.2f}\n"
        f"- Breakthrough potential: {current_report.get('breakthrough_profile', {}).get('score', 0):.2f}\n"
        f"- Quality threshold: {quality_threshold:.2f}\n"
        f"- Rigor threshold: {rigor_threshold:.2f}\n"
        f"- Claim support threshold: {claim_support_threshold:.2f}\n"
        f"- Target venue: {venue}\n"
        f"- Claims detected: {len(current_report.get('claim_ledger', []))}\n"
        f"- Unsupported claims: {len(current_report.get('claim_support', {}).get('unsupported_claims', []))}\n"
        f"- Figures: {current_report.get('evidence_pack', {}).get('num_figures', 0)}\n"
        f"- Tables: {current_report.get('evidence_pack', {}).get('num_tables', 0)}\n"
        f"- Strongest results items: {len(current_report.get('evidence_pack', {}).get('strongest_results', []))}\n"
        f"- Evidence density: {current_report.get('evidence_pack', {}).get('evidence_density_score', 0):.2f}\n"
        f"- Key numerical results: {current_report.get('key_results', {}).get('count', 0)}\n"
        f"- Rewrite applied: {'yes' if rewrite_applied else 'no'}\n"
        f"- Gate passed: {'yes' if gate_passed else 'no'}\n"
        f"- Submission readiness: {readiness['status']}\n"
        f"- Submission priority score: {priority_profile.get('score'):.2f}\n"
        f"- Submission priority tier: {priority_profile.get('tier')}\n"
    )

    rewrite_effectiveness_summary = _summarize_rewrite_trace(rewrite_trace)

    result = {
        "status": "success",
        "quality_status": "pass" if gate_passed else "needs_revision",
        "target_venue": venue,
        "rewrite_applied": rewrite_applied,
        "quality_score_before": initial_report.get("professional", {})
        .get("overall", {})
        .get("score", 0),
        "quality_score_after": final_quality_score,
        "rigor_score_before": initial_report.get("rigor", {}).get("score", 0),
        "rigor_score_after": final_rigor_score,
        "claim_support_before": initial_report.get("claim_support", {}).get("score", 0),
        "claim_support_after": final_claim_support_score,
        "claim_alignment_after": final_claim_alignment_score,
        "numeric_coverage_after": final_numeric_coverage_score,
        "breakthrough_score": current_report.get("breakthrough_profile", {}).get(
            "score", 0
        ),
        "claims_detected": len(current_report.get("claim_ledger", [])),
        "unsupported_claims_count": len(
            current_report.get("claim_support", {}).get("unsupported_claims", [])
        ),
        "suggested_claim_rewrites_count": len(
            [
                item
                for item in current_report.get("claim_ledger", [])
                if item.get("suggested_rewrite")
            ]
        ),
        "num_figures": current_report.get("evidence_pack", {}).get("num_figures", 0),
        "num_tables": current_report.get("evidence_pack", {}).get("num_tables", 0),
        "evidence_density_score": current_report.get("evidence_pack", {}).get(
            "evidence_density_score", 0
        ),
        "strongest_results": current_report.get("evidence_pack", {}).get(
            "strongest_results", []
        ),
        "key_results_count": current_report.get("key_results", {}).get("count", 0),
        "key_results": current_report.get("key_results", {}).get("values", []),
        "structured_results_count": len(
            current_report.get("key_results", {}).get("structured_results", [])
        ),
        "contribution_count": len(current_report.get("contribution_map", [])),
        "quality_gate_passed": gate_passed,
        "submission_priority_score": priority_profile.get("score"),
        "submission_priority_tier": priority_profile.get("tier"),
        "submission_priority_reasons": priority_profile.get("reasons", []),
        "blocker_count": priority_profile.get("blocker_count"),
        "critical_revision_actions_count": priority_profile.get(
            "critical_revision_actions_count"
        ),
        "quality_threshold": quality_threshold,
        "rigor_threshold": rigor_threshold,
        "claim_support_threshold": claim_support_threshold,
        "final_pdf": final_pdf,
        "quality_dir": str(quality_dir),
        "rewrite_trace": rewrite_trace,
        "rewrite_effectiveness_summary": rewrite_effectiveness_summary,
        "rewrite_effectiveness_file": str(quality_dir / "rewrite_effectiveness.md"),
        "rewrite_trace_summary_file": str(quality_dir / "rewrite_trace_summary.json"),
        "summary_file": str(quality_dir / "summary.md"),
        "submission_package_file": str(quality_dir / "submission_package.md"),
        "claim_alignment_file": str(quality_dir / "claim_alignment_final.json"),
        "narrative_map_file": str(quality_dir / "narrative_map.md"),
        "result_story_file": str(quality_dir / "result_story.md"),
        "contribution_map_file": str(quality_dir / "contribution_map_final.json"),
        "editor_pitch_file": str(quality_dir / "editor_pitch.md"),
        "rebuttal_package_file": str(quality_dir / "rebuttal_package.md"),
        "risk_register_file": str(quality_dir / "risk_register.md"),
        "cover_letter_file": str(quality_dir / "cover_letter.md"),
        "abstract_polish_file": str(quality_dir / "abstract_polish.md"),
        "impact_brief_file": str(quality_dir / "impact_brief.md"),
        "contribution_bullets_file": str(quality_dir / "contribution_bullets.md"),
        "strongest_claims_file": str(quality_dir / "strongest_claims.md"),
        "submission_manifest_file": str(quality_dir / "submission_manifest.json"),
        "submission_dashboard_file": str(quality_dir / "submission_dashboard.md"),
        "risk_language_plan_file": str(quality_dir / "risk_language_plan.md"),
        "claim_softening_plan_file": str(quality_dir / "claim_softening_plan.md"),
        "logic_check_file": str(quality_dir / "logic_check_report.md"),
        "reviewer_gate_report_file": str(quality_dir / "reviewer_gate_report.md"),
        "experiment_analysis_file": str(quality_dir / "experiment_analysis.md"),
        "experiment_visualization_brief_file": str(
            quality_dir / "experiment_visualization_brief.md"
        ),
        "figure_caption_guidance_file": str(quality_dir / "figure_caption_guidance.md"),
        "table_caption_guidance_file": str(quality_dir / "table_caption_guidance.md"),
        "architecture_figure_brief_file": str(
            quality_dir / "architecture_figure_brief.md"
        ),
        "humanizer_style_notes_file": str(quality_dir / "humanizer_style_notes.md"),
        "writing_skill_pack_file": str(quality_dir / "writing_skill_pack.md"),
        "submission_scorecard": current_report.get("submission_scorecard", {}),
        "revision_actions": current_report.get("revision_actions", []),
        "historical_rewrite_focus": current_report.get("historical_rewrite_focus", {}),
        "historical_reviewer_risks": current_report.get(
            "historical_reviewer_risks", {}
        ),
        "historical_style_preferences": current_report.get(
            "historical_style_preferences", {}
        ),
        "historical_rewrite_efficiency": current_report.get(
            "historical_rewrite_efficiency", {}
        ),
        "autonomous_followup_focus": applied_followup_focus,
        "autonomous_followup_focus_file": (
            str(quality_dir / "autonomous_followup_focus.json")
            if applied_followup_focus
            else None
        ),
        "auto_improvement_result": auto_improvement_result,
        "submission_readiness": readiness,
    }
    _safe_json_dump(quality_dir / "assessment_final.json", current_report)
    _safe_json_dump(
        quality_dir / "claim_ledger_final.json",
        {"venue": venue, "claims": current_report.get("claim_ledger", [])},
    )
    _safe_json_dump(
        quality_dir / "claim_alignment_final.json",
        {"venue": venue, "claims": current_report.get("claim_ledger", [])},
    )
    _safe_json_dump(
        quality_dir / "evidence_pack_final.json",
        current_report.get("evidence_pack", {}),
    )
    _safe_json_dump(
        quality_dir / "key_results_final.json", current_report.get("key_results", {})
    )
    _safe_json_dump(
        quality_dir / "contribution_map_final.json",
        {"venue": venue, "contributions": current_report.get("contribution_map", [])},
    )
    _safe_text_dump(quality_dir / "summary.md", summary_text)
    _safe_text_dump(
        quality_dir / "rewrite_effectiveness.md",
        _build_rewrite_effectiveness_text(current_report, result),
    )
    _safe_json_dump(
        quality_dir / "rewrite_trace_summary.json", rewrite_effectiveness_summary
    )
    _safe_text_dump(
        quality_dir / "submission_package.md",
        _build_submission_package_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "narrative_map.md",
        _build_narrative_map_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "result_story.md",
        _build_result_story_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "editor_pitch.md",
        _build_editor_pitch_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "rebuttal_package.md",
        _build_rebuttal_package_text(current_report, result),
    )
    _safe_text_dump(quality_dir / "risk_register.md", _build_risk_register_text(result))
    _safe_text_dump(
        quality_dir / "submission_dashboard.md",
        _build_submission_dashboard_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "risk_language_plan.md",
        _build_risk_language_plan_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "claim_softening_plan.md",
        _build_claim_softening_plan_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "logic_check_report.md",
        _build_logic_check_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "reviewer_gate_report.md",
        _build_reviewer_gate_report_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "experiment_analysis.md",
        _build_experiment_analysis_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "experiment_visualization_brief.md",
        _build_experiment_visualization_brief_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "figure_caption_guidance.md",
        _build_figure_caption_guidance_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "table_caption_guidance.md",
        _build_table_caption_guidance_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "architecture_figure_brief.md",
        _build_architecture_figure_brief_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "humanizer_style_notes.md",
        _build_humanizer_style_notes_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "writing_skill_pack.md",
        _build_writing_skill_pack_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "cover_letter.md",
        _build_cover_letter_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "abstract_polish.md", _build_abstract_polish_text(current_report)
    )
    _safe_text_dump(
        quality_dir / "impact_brief.md",
        _build_impact_brief_text(current_report, result),
    )
    _safe_text_dump(
        quality_dir / "contribution_bullets.md",
        _build_contribution_bullets_text(current_report),
    )
    _safe_text_dump(
        quality_dir / "strongest_claims.md",
        _build_strongest_claims_text(current_report),
    )
    _safe_json_dump(
        quality_dir / "submission_manifest.json",
        {
            "target_venue": venue,
            "paper_dir": str(base_folder),
            "artifacts": {k: v for k, v in result.items() if k.endswith("_file")},
        },
    )
    _safe_json_dump(result_file, result)
    return result
