"""Issue-driven self-review optimization helpers.

This module upgrades review -> rewrite loops from one-shot free-form edits into
structured, traceable iterations:
1. normalize review payloads;
2. build a prioritized issue ledger;
3. compare issue status across rounds;
4. run issue-driven rewrite with coverage checks and artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ai_scientist.utils.deferred_imports import load_module_attr
from ai_scientist.utils.pipeline_helpers import compile_latex as shared_compile_latex


_SCORE_KEYS = (
    "Originality",
    "Quality",
    "Clarity",
    "Significance",
    "Soundness",
    "Presentation",
    "Contribution",
    "Overall",
    "Confidence",
)

_SEVERITY_RANK = {
    "critical": 3,
    "major": 2,
    "minor": 1,
}

_CATEGORY_PRIORITY = {
    "Soundness": 9,
    "Quality": 8,
    "Contribution": 7,
    "Originality": 6,
    "Significance": 5,
    "Clarity": 4,
    "Presentation": 3,
    "Figure Quality": 2,
    "General": 1,
}

_CATEGORY_VALUE_WEIGHT = {
    "Soundness": 1.35,
    "Quality": 1.28,
    "Contribution": 1.22,
    "Originality": 1.18,
    "Significance": 1.15,
    "Clarity": 1.0,
    "Presentation": 0.9,
    "Figure Quality": 0.88,
    "General": 0.8,
}

_HIGH_EFFORT_HINTS = (
    "new experiment",
    "additional experiment",
    "ablation",
    "proof",
    "theorem",
    "benchmark",
    "dataset",
    "更多实验",
    "补充实验",
    "消融",
    "理论证明",
)

_MEDIUM_EFFORT_HINTS = (
    "clarify",
    "analysis",
    "explain",
    "discussion",
    "figure",
    "table",
    "描述",
    "阐释",
    "分析",
    "讨论",
)

_VENUE_GATE_PROFILES = {
    "nature": {
        "min_rounds": 2,
        "max_unresolved_critical": 0,
        "max_persistent": 1,
        "min_coverage_ratio": 0.65,
        "min_high_value_coverage": 0.7,
    },
    "journal": {
        "min_rounds": 2,
        "max_unresolved_critical": 0,
        "max_persistent": 2,
        "min_coverage_ratio": 0.55,
        "min_high_value_coverage": 0.6,
    },
    "default": {
        "min_rounds": 1,
        "max_unresolved_critical": 0,
        "max_persistent": 2,
        "min_coverage_ratio": 0.5,
        "min_high_value_coverage": 0.55,
    },
}

_NEGATIVE_IMG_HINTS = (
    "poor",
    "unclear",
    "mismatch",
    "missing",
    "inconsistent",
    "confusing",
    "hard to read",
    "not informative",
    "problem",
    "issue",
    "lack",
    "不足",
    "不清晰",
    "不一致",
    "有问题",
    "欠缺",
    "模糊",
)

_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff]+")


def _create_client(*args, **kwargs):
    return load_module_attr("ai_scientist.llm", "create_client")(*args, **kwargs)


def _get_response_from_llm(*args, **kwargs):
    return load_module_attr("ai_scientist.llm", "get_response_from_llm")(
        *args, **kwargs
    )


def _extract_json_between_markers(text: str):
    return load_module_attr("ai_scientist.llm", "extract_json_between_markers")(text)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "\n" in text:
        lines = [line.strip(" -*\t") for line in text.splitlines()]
        return [line for line in lines if line]
    if ";" in text:
        parts = [line.strip(" -*\t") for line in text.split(";")]
        return [line for line in parts if line]
    return [text]


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(value: Any) -> List[str]:
    text = _normalize_text(value)
    if not text:
        return []
    return [tok for tok in _TOKEN_SPLIT.split(text) if tok]


def _issue_id(category: str, problem: str) -> str:
    signature = f"{_normalize_text(category)}::{_normalize_text(problem)}"
    return "ISS-" + hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]


def _guess_category(text: str) -> str:
    lowered = _normalize_text(text)
    if any(token in lowered for token in ("soundness", "proof", "valid", "causal", "严谨", "有效性")):
        return "Soundness"
    if any(token in lowered for token in ("novel", "original", "创新", "新颖")):
        return "Originality"
    if any(token in lowered for token in ("quality", "baseline", "ablation", "实验", "对比")):
        return "Quality"
    if any(token in lowered for token in ("clarity", "write", "read", "表达", "清晰")):
        return "Clarity"
    if any(token in lowered for token in ("significance", "impact", "贡献", "意义")):
        return "Significance"
    if any(token in lowered for token in ("figure", "caption", "plot", "table", "图", "表")):
        return "Figure Quality"
    return "General"


def _severity_from_score(category: str, score: float) -> Optional[str]:
    if category == "Overall":
        if score <= 3:
            return "critical"
        if score <= 5:
            return "major"
        if score <= 7:
            return "minor"
        return None
    if score <= 2:
        return "critical"
    if score <= 3:
        return "major"
    return None


def _severity_from_text(text: str) -> str:
    lowered = _normalize_text(text)
    critical_hints = (
        "fatal",
        "invalid",
        "not supported",
        "unreliable",
        "major flaw",
        "critical",
        "严重",
        "致命",
        "无效",
    )
    if any(token in lowered for token in critical_hints):
        return "critical"
    major_hints = (
        "missing",
        "unclear",
        "weak",
        "lack",
        "insufficient",
        "limited",
        "问题",
        "不足",
        "薄弱",
    )
    if any(token in lowered for token in major_hints):
        return "major"
    return "minor"


def _trim_text(value: Any, *, limit: int = 480) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _base_review_payload(review_text: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(review_text, dict):
        return {}
    if isinstance(review_text.get("review"), dict):
        return dict(review_text["review"])
    return dict(review_text)


def normalize_review(review_text: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    root = _base_review_payload(review_text)
    scores: Dict[str, float] = {}
    raw_scores = root.get("scores")
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            score = _coerce_float(value)
            if score is not None:
                scores[str(key)] = score
    for key in _SCORE_KEYS:
        score = _coerce_float(root.get(key))
        if score is not None:
            scores[key] = score

    return {
        "summary": str(root.get("Summary") or "").strip(),
        "strengths": _coerce_list(root.get("Strengths")),
        "weaknesses": _coerce_list(root.get("Weaknesses")),
        "questions": _coerce_list(root.get("Questions")),
        "limitations": _coerce_list(root.get("Limitations")),
        "decision": str(root.get("Decision") or "").strip(),
        "scores": scores,
        "ethical_concerns": bool(root.get("Ethical Concerns")),
        "raw": root,
    }


def _iter_image_feedback_rows(review_img: Optional[Dict[str, Any]]) -> Iterable[Dict[str, str]]:
    if not isinstance(review_img, dict):
        return
    if isinstance(review_img.get("figure_reviews"), list):
        for idx, row in enumerate(review_img["figure_reviews"], start=1):
            if not isinstance(row, dict):
                continue
            figure_id = str(row.get("figure_id") or f"figure_{idx}")
            text_parts = [
                str(row.get("description") or "").strip(),
                str(row.get("issue") or "").strip(),
                str(row.get("overall_quality") or "").strip(),
            ]
            text = " ".join(part for part in text_parts if part).strip()
            if text:
                yield {"figure_id": figure_id, "text": text}
        return
    for fig_name, payload in review_img.items():
        if not isinstance(payload, dict):
            continue
        text_parts = [
            str(payload.get("Img_review") or "").strip(),
            str(payload.get("Caption_review") or "").strip(),
            str(payload.get("Figrefs_review") or "").strip(),
            str(payload.get("Overall_comments") or "").strip(),
            str(payload.get("Informative_review") or "").strip(),
        ]
        text = " ".join(part for part in text_parts if part).strip()
        if text:
            yield {"figure_id": str(fig_name), "text": text}


def _build_score_issues(scores: Dict[str, float]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for category, score in scores.items():
        severity = _severity_from_score(category, float(score))
        if severity is None:
            continue
        if category == "Overall":
            evidence = f"Overall score is {score:.1f}/10."
            problem = (
                "Overall review confidence indicates the manuscript is not yet submission-ready."
            )
        else:
            evidence = f"{category} score is {score:.1f}/4."
            problem = f"{category} is below reviewer expectations and needs targeted revision."
        issues.append(
            {
                "source": "score",
                "category": category,
                "severity": severity,
                "problem": problem,
                "evidence": evidence,
                "action_hint": (
                    f"Address the main {category} weaknesses with explicit evidence-backed rewrites."
                ),
            }
        )
    return issues


def _build_text_issues(review: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for weakness in review.get("weaknesses", []):
        if not weakness:
            continue
        category = _guess_category(weakness)
        issues.append(
            {
                "source": "weakness",
                "category": category,
                "severity": _severity_from_text(weakness),
                "problem": _trim_text(weakness),
                "evidence": "",
                "action_hint": "Provide concrete fixes and revise affected paragraphs with stronger evidence linkage.",
            }
        )
    for question in review.get("questions", []):
        if not question:
            continue
        category = _guess_category(question)
        issues.append(
            {
                "source": "question",
                "category": category,
                "severity": "minor",
                "problem": _trim_text(question),
                "evidence": "Reviewer requested clarification.",
                "action_hint": "Add explicit clarification in the corresponding section and align claims with available evidence.",
            }
        )
    for limitation in review.get("limitations", []):
        if not limitation:
            continue
        category = _guess_category(limitation)
        issues.append(
            {
                "source": "limitation",
                "category": category,
                "severity": "minor",
                "problem": _trim_text(limitation),
                "evidence": "Reviewer highlighted limitation risk.",
                "action_hint": "Tighten scope language and add limitation-aware explanation.",
            }
        )
    return issues


def _build_image_issues(review_img: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for row in _iter_image_feedback_rows(review_img):
        text = row["text"]
        if not text:
            continue
        lowered = _normalize_text(text)
        if not any(token in lowered for token in _NEGATIVE_IMG_HINTS):
            continue
        severity = "major" if any(token in lowered for token in ("mismatch", "missing", "poor", "not informative", "不一致")) else "minor"
        issues.append(
            {
                "source": "image_review",
                "category": "Figure Quality",
                "severity": severity,
                "problem": f"{row['figure_id']}: {_trim_text(text, limit=360)}",
                "evidence": "Figure/caption/reference alignment feedback from vision review.",
                "action_hint": "Improve caption precision, figure readability, and text-figure alignment.",
            }
        )
    return issues


def _dedupe_and_rank_issues(raw_issues: List[Dict[str, Any]], *, max_issues: int = 14) -> List[Dict[str, Any]]:
    by_signature: Dict[str, Dict[str, Any]] = {}
    for issue in raw_issues:
        category = str(issue.get("category") or "General")
        problem = str(issue.get("problem") or "").strip()
        if not problem:
            continue
        signature = f"{_normalize_text(category)}::{_normalize_text(problem)}"
        item = dict(issue)
        item["issue_id"] = _issue_id(category, problem)
        if signature not in by_signature:
            by_signature[signature] = item
            continue
        existing = by_signature[signature]
        existing_rank = _SEVERITY_RANK.get(str(existing.get("severity") or "minor"), 1)
        new_rank = _SEVERITY_RANK.get(str(item.get("severity") or "minor"), 1)
        if new_rank > existing_rank:
            by_signature[signature] = item

    issues = list(by_signature.values())
    issues.sort(
        key=lambda row: (
            -_SEVERITY_RANK.get(str(row.get("severity") or "minor"), 1),
            -_CATEGORY_PRIORITY.get(str(row.get("category") or "General"), 1),
            str(row.get("problem") or ""),
        )
    )
    return issues[: max(1, int(max_issues))]


def _estimate_effort_level(issue: Dict[str, Any]) -> int:
    text = _normalize_text(
        " ".join(
            [
                str(issue.get("problem") or ""),
                str(issue.get("action_hint") or ""),
                str(issue.get("evidence") or ""),
            ]
        )
    )
    if any(token in text for token in _HIGH_EFFORT_HINTS):
        return 3
    if any(token in text for token in _MEDIUM_EFFORT_HINTS):
        return 2
    return 1


def _find_best_previous_issue(
    issue: Dict[str, Any], previous_issues: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    issue_id = str(issue.get("issue_id") or "").strip()
    if not previous_issues:
        return None
    if issue_id:
        for row in previous_issues:
            if str(row.get("issue_id") or "").strip() == issue_id:
                return row

    best_match = None
    best_score = 0.0
    for candidate in previous_issues:
        score = _issue_similarity(issue, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    if best_match is not None and best_score >= 0.46:
        return best_match
    return None


def _compute_issue_value_profile(
    issue: Dict[str, Any],
    *,
    target_venue: Optional[str],
    persistence_count: int,
) -> Dict[str, Any]:
    severity = str(issue.get("severity") or "minor").lower()
    category = str(issue.get("category") or "General")
    severity_rank = _SEVERITY_RANK.get(severity, 1)
    category_weight = _CATEGORY_VALUE_WEIGHT.get(category, 0.8)
    effort_level = _estimate_effort_level(issue)
    evidence_bonus = 0.12 if str(issue.get("evidence") or "").strip() else 0.0
    venue = str(target_venue or "default").lower()
    venue_bonus = 0.18 if venue == "nature" and category in {"Soundness", "Contribution", "Significance", "Quality"} else 0.0
    persistence_bonus = min(0.34, max(0, persistence_count - 1) * 0.12)
    impact_score = round((severity_rank * category_weight) + evidence_bonus + venue_bonus + persistence_bonus, 4)
    value_density = round(impact_score / max(1, effort_level), 4)
    if impact_score >= 3.1:
        priority_tier = "P0"
    elif impact_score >= 2.2:
        priority_tier = "P1"
    else:
        priority_tier = "P2"
    rationale_parts = [f"severity={severity_rank}", f"category_weight={category_weight:.2f}"]
    if persistence_count > 1:
        rationale_parts.append(f"persistent×{persistence_count}")
    if venue_bonus > 0:
        rationale_parts.append("nature_bonus")
    return {
        "persistence_count": persistence_count,
        "effort_level": effort_level,
        "impact_score": impact_score,
        "value_density": value_density,
        "priority_tier": priority_tier,
        "value_rationale": ", ".join(rationale_parts),
    }


def _annotate_issues_with_memory_and_value(
    issues: List[Dict[str, Any]],
    *,
    previous_ledger: Optional[Dict[str, Any]],
    target_venue: Optional[str],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    previous_issues = []
    previous_memory = {}
    if isinstance(previous_ledger, dict):
        previous_issues = [
            dict(item)
            for item in (previous_ledger.get("issues") or [])
            if isinstance(item, dict)
        ]
        previous_memory = dict(previous_ledger.get("issue_memory") or {})

    issue_memory: Dict[str, Dict[str, Any]] = {}
    annotated: List[Dict[str, Any]] = []
    persistent_issue_ids: List[str] = []

    for issue in issues:
        row = dict(issue)
        issue_id = str(row.get("issue_id") or "")
        matched = _find_best_previous_issue(row, previous_issues)

        previous_issue_id = ""
        previous_persistence = 0
        if matched is not None:
            previous_issue_id = str(matched.get("issue_id") or "").strip()
            if previous_issue_id:
                previous_persistence = int(
                    (previous_memory.get(previous_issue_id) or {}).get(
                        "persistence_count", 1
                    )
                )
            else:
                previous_persistence = 1
        persistence_count = max(1, previous_persistence + 1) if matched is not None else 1
        if persistence_count > 1 and issue_id:
            persistent_issue_ids.append(issue_id)

        profile = _compute_issue_value_profile(
            row,
            target_venue=target_venue,
            persistence_count=persistence_count,
        )
        row.update(profile)
        annotated.append(row)

        if issue_id:
            issue_memory[issue_id] = {
                "issue_id": issue_id,
                "category": str(row.get("category") or "General"),
                "severity": str(row.get("severity") or "minor"),
                "persistence_count": persistence_count,
                "last_seen_at": _now_iso(),
                "previous_issue_id": previous_issue_id,
                "value_density": row.get("value_density", 0),
                "impact_score": row.get("impact_score", 0),
            }

    annotated.sort(
        key=lambda row: (
            -float(row.get("value_density") or 0),
            -float(row.get("impact_score") or 0),
            -_CATEGORY_PRIORITY.get(str(row.get("category") or "General"), 1),
            str(row.get("problem") or ""),
        )
    )

    return annotated, {
        "generated_at": _now_iso(),
        "issue_memory": issue_memory,
        "persistent_issue_ids": persistent_issue_ids,
    }


def _select_recommended_targets(
    issues: List[Dict[str, Any]],
    *,
    max_targets: int = 8,
) -> List[str]:
    picked: List[str] = []
    for row in issues:
        issue_id = str(row.get("issue_id") or "").strip()
        if not issue_id:
            continue
        if issue_id in picked:
            continue
        picked.append(issue_id)
        if len(picked) >= max(1, int(max_targets)):
            break
    return picked


def build_issue_ledger(
    *,
    text_review: Optional[Dict[str, Any]],
    img_review: Optional[Dict[str, Any]],
    max_issues: int = 14,
    previous_ledger: Optional[Dict[str, Any]] = None,
    target_venue: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_review = normalize_review(text_review)
    raw_issues: List[Dict[str, Any]] = []
    raw_issues.extend(_build_score_issues(normalized_review.get("scores", {})))
    raw_issues.extend(_build_text_issues(normalized_review))
    raw_issues.extend(_build_image_issues(img_review))
    issues = _dedupe_and_rank_issues(raw_issues, max_issues=max_issues)
    issues, memory_meta = _annotate_issues_with_memory_and_value(
        issues,
        previous_ledger=previous_ledger,
        target_venue=target_venue,
    )

    critical = [row for row in issues if row.get("severity") == "critical"]
    major = [row for row in issues if row.get("severity") == "major"]
    minor = [row for row in issues if row.get("severity") == "minor"]
    recommended_targets = _select_recommended_targets(issues, max_targets=8)
    high_value_targets = [
        issue_id
        for issue_id in recommended_targets
        if any(
            str(row.get("issue_id")) == issue_id and str(row.get("priority_tier")) in {"P0", "P1"}
            for row in issues
        )
    ]

    return {
        "generated_at": _now_iso(),
        "target_venue": str(target_venue or "").strip(),
        "normalized_review": normalized_review,
        "issue_count": len(issues),
        "critical_count": len(critical),
        "major_count": len(major),
        "minor_count": len(minor),
        "issues": issues,
        "recommended_targets": recommended_targets,
        "high_value_targets": high_value_targets,
        "issue_memory": memory_meta.get("issue_memory", {}),
        "persistent_issue_ids": memory_meta.get("persistent_issue_ids", []),
        "value_overview": {
            "max_impact_score": round(
                max((float(row.get("impact_score") or 0) for row in issues), default=0.0),
                4,
            ),
            "avg_impact_score": round(
                sum(float(row.get("impact_score") or 0) for row in issues)
                / max(1, len(issues)),
                4,
            ),
            "high_priority_count": sum(
                1 for row in issues if str(row.get("priority_tier")) in {"P0", "P1"}
            ),
        },
    }


def render_structured_review_markdown(ledger: Dict[str, Any]) -> str:
    review = ledger.get("normalized_review", {})
    scores = review.get("scores", {})
    issues = ledger.get("issues", [])
    recommended_targets = [
        str(item).strip()
        for item in (ledger.get("recommended_targets") or [])
        if str(item).strip()
    ]
    issue_by_id = {
        str(item.get("issue_id") or "").strip(): item
        for item in issues
        if isinstance(item, dict)
    }

    lines: List[str] = [
        "# Structured Self-Review Report",
        "",
        f"- Generated at: {ledger.get('generated_at')}",
        f"- Issues: {ledger.get('issue_count', 0)} "
        f"(critical={ledger.get('critical_count', 0)}, "
        f"major={ledger.get('major_count', 0)}, minor={ledger.get('minor_count', 0)})",
        "",
        "## Summary",
        review.get("summary") or "(no summary provided by reviewer)",
        "",
        "## Strengths",
    ]
    strengths = review.get("strengths") or []
    if strengths:
        lines.extend([f"- {item}" for item in strengths[:8]])
    else:
        lines.append("- (none)")

    lines.extend(["", "## Weaknesses"])
    weaknesses = review.get("weaknesses") or []
    if weaknesses:
        lines.extend([f"- {item}" for item in weaknesses[:10]])
    else:
        lines.append("- (none)")

    lines.extend(["", "## Key Issues"])
    if issues:
        for issue in issues:
            lines.append(
                f"- [{issue.get('severity', 'minor').upper()}] {issue.get('issue_id')} "
                f"[{issue.get('category', 'General')}] {issue.get('problem')}"
            )
            if issue.get("impact_score") is not None:
                lines.append(
                    f"  Value: tier={issue.get('priority_tier', 'P2')}, "
                    f"impact={issue.get('impact_score')}, "
                    f"density={issue.get('value_density')}, "
                    f"persistence={issue.get('persistence_count', 1)}"
                )
            if issue.get("evidence"):
                lines.append(f"  Evidence: {issue.get('evidence')}")
            if issue.get("action_hint"):
                lines.append(f"  Action: {issue.get('action_hint')}")
    else:
        lines.append("- (no actionable issue extracted)")

    lines.extend(["", "## Actionable Suggestions"])
    plan_issues: List[Dict[str, Any]]
    if recommended_targets:
        plan_issues = [
            issue_by_id[issue_id]
            for issue_id in recommended_targets
            if issue_id in issue_by_id
        ][:8]
    else:
        plan_issues = issues[:8]

    if plan_issues:
        for issue in plan_issues:
            lines.append(
                f"- {issue.get('issue_id')}: prioritize {issue.get('category')} fix; {issue.get('action_hint')}"
            )
    else:
        lines.append("- Keep the current draft stable; only do minimal polish.")

    lines.extend(["", "## Priority Revision Plan"])
    if plan_issues:
        for issue in plan_issues:
            sev = str(issue.get("severity") or "minor").lower()
            priority = "P0" if sev == "critical" else "P1" if sev == "major" else "P2"
            lines.append(
                f"- {priority} {issue.get('issue_id')} [{issue.get('category')}]: {issue.get('problem')}"
            )
    else:
        lines.append("- P2 No major blockers detected.")

    lines.extend(["", "## Recommended Targets"])
    if recommended_targets:
        for issue_id in recommended_targets:
            issue = issue_by_id.get(issue_id, {})
            lines.append(
                f"- {issue_id}: tier={issue.get('priority_tier', 'P2')}, "
                f"category={issue.get('category', 'General')}, "
                f"impact={issue.get('impact_score', 0)}"
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "## Scores"])
    if scores:
        for key in sorted(scores.keys()):
            value = scores[key]
            scale = "/10" if key == "Overall" else "/4"
            lines.append(f"- {key}: {value}{scale}")
    else:
        lines.append("- (no scores)")
    return "\n".join(lines).strip() + "\n"


def _severity_rank(value: Any) -> int:
    return _SEVERITY_RANK.get(str(value or "minor").lower(), 1)


def _issue_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_tokens = set(_tokenize(left.get("problem")))
    right_tokens = set(_tokenize(right.get("problem")))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union <= 0:
        return 0.0
    jaccard = overlap / union
    if str(left.get("category")) == str(right.get("category")):
        jaccard += 0.2
    return min(1.0, jaccard)


def evaluate_issue_progress(
    *,
    previous_issues: List[Dict[str, Any]],
    current_issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    previous = [dict(item) for item in previous_issues if isinstance(item, dict)]
    current = [dict(item) for item in current_issues if isinstance(item, dict)]

    current_by_id = {
        str(item.get("issue_id")): item
        for item in current
        if str(item.get("issue_id") or "").strip()
    }
    matched_current_ids: set[str] = set()

    resolved: List[Dict[str, Any]] = []
    persistent: List[Dict[str, Any]] = []
    severity_downgraded: List[Dict[str, Any]] = []

    for old_issue in previous:
        old_id = str(old_issue.get("issue_id") or "").strip()
        best_match = None
        if old_id and old_id in current_by_id:
            best_match = current_by_id[old_id]
            matched_current_ids.add(old_id)
        else:
            best_score = 0.0
            for new_issue in current:
                new_id = str(new_issue.get("issue_id") or "").strip()
                if new_id in matched_current_ids:
                    continue
                score = _issue_similarity(old_issue, new_issue)
                if score > best_score:
                    best_score = score
                    best_match = new_issue
            if best_match is not None and best_score >= 0.46:
                new_id = str(best_match.get("issue_id") or "").strip()
                if new_id:
                    matched_current_ids.add(new_id)
            else:
                best_match = None

        if best_match is None:
            resolved.append(old_issue)
            continue

        old_rank = _severity_rank(old_issue.get("severity"))
        new_rank = _severity_rank(best_match.get("severity"))
        row = {
            "previous_issue": old_issue,
            "current_issue": best_match,
        }
        persistent.append(row)
        if new_rank < old_rank:
            severity_downgraded.append(row)

    new_issues = [
        issue
        for issue in current
        if str(issue.get("issue_id") or "").strip() not in matched_current_ids
    ]

    unresolved_critical = 0
    for row in persistent:
        if str(row["current_issue"].get("severity") or "").lower() == "critical":
            unresolved_critical += 1
    unresolved_critical += sum(
        1 for row in new_issues if str(row.get("severity") or "").lower() == "critical"
    )

    return {
        "generated_at": _now_iso(),
        "previous_issue_count": len(previous),
        "current_issue_count": len(current),
        "resolved_issue_count": len(resolved),
        "persistent_issue_count": len(persistent),
        "new_issue_count": len(new_issues),
        "severity_downgraded_count": len(severity_downgraded),
        "unresolved_critical_count": unresolved_critical,
        "resolved_issues": resolved,
        "persistent_issues": persistent,
        "new_issues": new_issues,
        "severity_downgraded": severity_downgraded,
    }


def _resolve_gate_profile(target_venue: Optional[str]) -> Dict[str, Any]:
    venue = str(target_venue or "").strip().lower()
    if not venue:
        venue = "default"
    profile = dict(_VENUE_GATE_PROFILES.get(venue) or _VENUE_GATE_PROFILES["default"])
    profile["target_venue"] = venue
    return profile


def assess_self_review_gate(
    *,
    ledger: Dict[str, Any],
    progress: Optional[Dict[str, Any]],
    rewrite_result: Optional[Dict[str, Any]],
    round_index: int,
    target_venue: Optional[str] = None,
) -> Dict[str, Any]:
    profile = _resolve_gate_profile(target_venue or ledger.get("target_venue"))
    issues = [dict(item) for item in (ledger.get("issues") or []) if isinstance(item, dict)]
    unresolved_critical = (
        int(progress.get("unresolved_critical_count", 0))
        if isinstance(progress, dict)
        else int(ledger.get("critical_count", 0))
    )
    persistent_count = int(progress.get("persistent_issue_count", 0)) if isinstance(progress, dict) else 0

    coverage_ratio = 0.0
    high_value_coverage = 0.0
    compile_ok = False
    if isinstance(rewrite_result, dict):
        coverage_ratio = float(rewrite_result.get("coverage_ratio") or 0.0)
        high_value_coverage = float(rewrite_result.get("high_value_coverage_ratio") or 0.0)
        compile_ok = bool(rewrite_result.get("compile_ok"))

    if not isinstance(rewrite_result, dict):
        coverage_ratio = 0.0
        compile_ok = False

    high_value_candidates = [
        str(issue.get("issue_id")).strip()
        for issue in issues
        if str(issue.get("priority_tier") or "") in {"P0", "P1"}
        and str(issue.get("issue_id") or "").strip()
    ]
    recommended_targets = [
        str(item).strip()
        for item in (ledger.get("recommended_targets") or [])
        if str(item).strip()
    ]
    covered_issue_ids = set(
        str(item).strip()
        for item in ((rewrite_result or {}).get("covered_issue_ids") or [])
        if str(item).strip()
    )

    if high_value_candidates and high_value_coverage <= 0:
        high_value_coverage = round(
            len(covered_issue_ids & set(high_value_candidates))
            / max(1, len(high_value_candidates)),
            4,
        )
    elif not high_value_candidates:
        high_value_coverage = 1.0

    checks = {
        "compile_ok": compile_ok,
        "round_budget_met": int(round_index) >= int(profile["min_rounds"]),
        "critical_resolved": unresolved_critical <= int(profile["max_unresolved_critical"]),
        "persistent_within_budget": persistent_count <= int(profile["max_persistent"]),
        "coverage_sufficient": coverage_ratio >= float(profile["min_coverage_ratio"]),
        "high_value_covered": high_value_coverage >= float(profile["min_high_value_coverage"]),
    }

    score = 0
    if checks["compile_ok"]:
        score += 10
    if checks["critical_resolved"]:
        score += 25
    if checks["persistent_within_budget"]:
        score += 20
    if checks["coverage_sufficient"]:
        score += 20
    if checks["high_value_covered"]:
        score += 15
    if checks["round_budget_met"]:
        score += 10

    focus_issue_ids: List[str] = []
    if recommended_targets:
        for issue_id in recommended_targets:
            if issue_id not in covered_issue_ids:
                focus_issue_ids.append(issue_id)
            if len(focus_issue_ids) >= 5:
                break

    focus_summaries: List[str] = []
    issue_by_id = {
        str(item.get("issue_id") or "").strip(): item
        for item in issues
        if str(item.get("issue_id") or "").strip()
    }
    for issue_id in focus_issue_ids:
        issue = issue_by_id.get(issue_id, {})
        if not issue:
            continue
        focus_summaries.append(
            f"{issue_id} [{issue.get('category', 'General')}/{issue.get('priority_tier', 'P2')}]: {issue.get('problem', '')}"
        )

    ready = all(checks.values())
    reasons: List[str] = []
    if not checks["round_budget_met"]:
        reasons.append(f"round<{profile['min_rounds']}")
    if not checks["compile_ok"]:
        reasons.append("latex_compile_failed")
    if not checks["critical_resolved"]:
        reasons.append("critical_issues_unresolved")
    if not checks["persistent_within_budget"]:
        reasons.append("persistent_issues_high")
    if not checks["coverage_sufficient"]:
        reasons.append("rewrite_coverage_low")
    if not checks["high_value_covered"]:
        reasons.append("high_value_coverage_low")

    return {
        "generated_at": _now_iso(),
        "target_venue": profile["target_venue"],
        "round_index": int(round_index),
        "ready": bool(ready),
        "score": int(score),
        "checks": checks,
        "thresholds": {
            "min_rounds": int(profile["min_rounds"]),
            "max_unresolved_critical": int(profile["max_unresolved_critical"]),
            "max_persistent": int(profile["max_persistent"]),
            "min_coverage_ratio": float(profile["min_coverage_ratio"]),
            "min_high_value_coverage": float(profile["min_high_value_coverage"]),
        },
        "metrics": {
            "unresolved_critical_count": unresolved_critical,
            "persistent_issue_count": persistent_count,
            "coverage_ratio": round(coverage_ratio, 4),
            "high_value_coverage_ratio": round(high_value_coverage, 4),
            "recommended_target_count": len(recommended_targets),
            "high_value_target_count": len(high_value_candidates),
        },
        "next_focus_issue_ids": focus_issue_ids,
        "next_focus_summaries": focus_summaries,
        "reasons": reasons,
    }


def render_round_gate_markdown(gate: Dict[str, Any]) -> str:
    checks = dict(gate.get("checks") or {})
    metrics = dict(gate.get("metrics") or {})
    lines = [
        "# Self-Review Round Gate",
        "",
        f"- Generated at: {gate.get('generated_at')}",
        f"- Target venue: {gate.get('target_venue') or 'default'}",
        f"- Round: {gate.get('round_index')}",
        f"- Gate ready: {'YES' if gate.get('ready') else 'NO'}",
        f"- Gate score: {gate.get('score', 0)}/100",
        "",
        "## Checks",
    ]
    if checks:
        for key, passed in checks.items():
            lines.append(f"- {'✅' if passed else '❌'} {key}")
    else:
        lines.append("- (no checks)")
    lines.extend(["", "## Metrics"])
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- (no metrics)")
    lines.extend(["", "## Next Focus"])
    focus = gate.get("next_focus_summaries") or []
    if focus:
        lines.extend([f"- {item}" for item in focus[:5]])
    else:
        lines.append("- (none)")
    if gate.get("reasons"):
        lines.extend(["", "## Not-Ready Reasons"])
        lines.extend([f"- {item}" for item in gate.get("reasons", [])])
    return "\n".join(lines).strip() + "\n"


def _extract_latex_block(text: str) -> str:
    match = re.search(r"```latex\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_json_block(text: str) -> Dict[str, Any]:
    parsed = _extract_json_between_markers(text)
    if isinstance(parsed, dict):
        return parsed
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _build_issue_driven_prompt(
    *,
    latex_content: str,
    ledger: Dict[str, Any],
    target_venue: Optional[str] = None,
    max_latex_chars: int = 22000,
) -> str:
    issues = [dict(item) for item in (ledger.get("issues") or []) if isinstance(item, dict)]
    issue_by_id = {
        str(item.get("issue_id") or "").strip(): item
        for item in issues
        if str(item.get("issue_id") or "").strip()
    }
    recommended_targets = [
        str(item).strip()
        for item in (ledger.get("recommended_targets") or [])
        if str(item).strip()
    ]
    ordered_issues: List[Dict[str, Any]] = []
    for issue_id in recommended_targets:
        row = issue_by_id.get(issue_id)
        if row is not None:
            ordered_issues.append(row)
    for row in issues:
        issue_id = str(row.get("issue_id") or "").strip()
        if issue_id and issue_id in recommended_targets:
            continue
        ordered_issues.append(row)

    persistent_issue_ids = {
        str(item).strip()
        for item in (ledger.get("persistent_issue_ids") or [])
        if str(item).strip()
    }
    active_venue = str(target_venue or ledger.get("target_venue") or "default")
    issue_rows = []
    for issue in ordered_issues[:10]:
        issue_id = str(issue.get("issue_id") or "").strip()
        issue_rows.append(
            {
                "issue_id": issue_id,
                "severity": issue.get("severity"),
                "category": issue.get("category"),
                "problem": issue.get("problem"),
                "evidence": issue.get("evidence"),
                "action_hint": issue.get("action_hint"),
                "priority_tier": issue.get("priority_tier", "P2"),
                "impact_score": issue.get("impact_score", 0),
                "value_density": issue.get("value_density", 0),
                "persistence_count": issue.get("persistence_count", 1),
                "is_persistent": issue_id in persistent_issue_ids,
            }
        )

    high_value_targets = [
        str(item).strip()
        for item in (ledger.get("high_value_targets") or [])
        if str(item).strip()
    ]

    return f"""
你是资深科研论文改写审稿助手。请基于“问题台账”进行点对点修订，不要泛泛而谈。

必须遵守：
1. 优先处理 `priority_tier=P0/P1` 与 `is_persistent=true` 的 issue_id。
2. 只针对问题台账中的 issue_id 做修改，避免无关大改。
3. 修改要可追踪：每个 issue_id 说明改了什么、在哪个章节改，并说明引用了哪条证据。
4. 保持 LaTeX 可编译，不删除关键结构（标题、作者、摘要、参考文献环境）。
5. 不要捏造新实验结果；若证据不足，必须降调 claim 并显式限定范围。
6. 输出必须包含 JSON 变更计划 + 完整 LaTeX。

目标投稿风格: {active_venue}
高价值目标 issue_id: {high_value_targets}

问题台账（按价值优先级）:
```json
{json.dumps(issue_rows, indent=2, ensure_ascii=False)}
```

当前论文 LaTeX:
```latex
{latex_content[:max_latex_chars]}
```

按以下格式返回：
THOUGHT:
<简要说明修订策略>

IMPROVEMENT JSON:
```json
{{
  "addressed_issue_ids": ["ISS-..."],
  "unaddressed_issue_ids": ["ISS-..."],
  "change_log": [
    {{
      "issue_id": "ISS-...",
      "actions": ["做了哪些具体改动"],
      "sections": ["修改到的章节标题或位置"],
      "evidence_links": ["引用的图/表/实验编号或已有证据描述"],
      "expected_effect": "预期改善点"
    }}
  ],
  "risk_checks": ["仍需注意的风险"],
  "next_round_focus": ["下一轮建议聚焦项"],
  "confidence": "high|medium|low"
}}
```

REVISED LATEX:
```latex
<完整可编译 LaTeX>
```
""".strip()


def apply_issue_driven_rewrite(
    *,
    paper_dir: str,
    model: str,
    ledger: Dict[str, Any],
    round_index: int,
    artifact_dir: str,
    target_venue: Optional[str] = None,
    temperature: float = 0.35,
) -> Dict[str, Any]:
    latex_dir = osp.join(paper_dir, "latex")
    latex_file = osp.join(latex_dir, "template.tex")
    if not osp.exists(latex_file):
        return {
            "status": "failed",
            "reason": "missing_latex",
            "latex_file": latex_file,
        }

    with open(latex_file, "r", encoding="utf-8", errors="ignore") as f:
        original_latex = f.read()

    client, client_model = _create_client(model)
    prompt = _build_issue_driven_prompt(
        latex_content=original_latex,
        ledger=ledger,
        target_venue=target_venue,
    )
    response, _ = _get_response_from_llm(
        prompt=prompt,
        client=client,
        model=client_model,
        system_message=(
            "你是严谨的科研写作修订专家。"
            "你必须按照 issue_id 点对点修复，并保持 LaTeX 稳定可编译。"
        ),
        temperature=temperature,
    )

    rewrite_plan = _extract_json_block(response)
    revised_latex = _extract_latex_block(response)
    if not revised_latex:
        return {
            "status": "failed",
            "reason": "missing_revised_latex",
            "rewrite_plan": rewrite_plan,
            "response_excerpt": response[:1500],
        }

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    before_file = artifact_path / f"template_before_round_{round_index}.tex"
    after_file = artifact_path / f"template_after_round_{round_index}.tex"
    raw_response_file = artifact_path / f"rewrite_response_round_{round_index}.txt"
    shutil.copy(latex_file, before_file)
    with open(raw_response_file, "w", encoding="utf-8") as f:
        f.write(response)

    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(revised_latex)
    with open(after_file, "w", encoding="utf-8") as f:
        f.write(revised_latex)

    compile_ok = bool(
        shared_compile_latex(latex_dir, pdf_file=None, timeout=45, verbose=False)
    )
    if not compile_ok:
        shutil.copy(before_file, latex_file)

    targeted_issue_ids = [
        str(item).strip()
        for item in (ledger.get("recommended_targets") or [])
        if str(item).strip()
    ]
    if not targeted_issue_ids:
        targeted_issue_ids = [
            str(item.get("issue_id"))
            for item in (ledger.get("issues") or [])
            if str(item.get("issue_id") or "").strip()
        ]
    high_value_issue_ids = [
        str(item).strip()
        for item in (ledger.get("high_value_targets") or [])
        if str(item).strip()
    ]
    if not high_value_issue_ids:
        high_value_issue_ids = [
            str(item.get("issue_id")).strip()
            for item in (ledger.get("issues") or [])
            if str(item.get("priority_tier") or "") in {"P0", "P1"}
            and str(item.get("issue_id") or "").strip()
        ]
    addressed_issue_ids = [
        str(item).strip()
        for item in _coerce_list(rewrite_plan.get("addressed_issue_ids"))
        if str(item).strip()
    ]
    covered = sorted(set(targeted_issue_ids) & set(addressed_issue_ids))
    coverage_ratio = (
        len(covered) / len(targeted_issue_ids) if targeted_issue_ids else 0.0
    )
    high_value_covered = sorted(set(high_value_issue_ids) & set(addressed_issue_ids))
    high_value_coverage_ratio = (
        len(high_value_covered) / len(high_value_issue_ids) if high_value_issue_ids else 1.0
    )

    change_log = rewrite_plan.get("change_log")
    if not isinstance(change_log, list):
        change_log = []
    structured_change_count = 0
    structured_change_issue_ids: set[str] = set()
    for row in change_log:
        if not isinstance(row, dict):
            continue
        issue_id = str(row.get("issue_id") or "").strip()
        if not issue_id:
            continue
        actions = _coerce_list(row.get("actions"))
        sections = _coerce_list(row.get("sections"))
        if actions and sections:
            structured_change_count += 1
            structured_change_issue_ids.add(issue_id)

    known_issue_ids = set(targeted_issue_ids)
    unknown_addressed_issue_ids = sorted(
        issue_id
        for issue_id in addressed_issue_ids
        if issue_id and issue_id not in known_issue_ids
    )
    structured_plan_coverage_ratio = (
        len(structured_change_issue_ids & known_issue_ids) / len(known_issue_ids)
        if known_issue_ids
        else 0.0
    )

    result = {
        "status": "success" if compile_ok else "failed_compile_rollback",
        "target_venue": str(target_venue or ledger.get("target_venue") or "").strip(),
        "round_index": round_index,
        "compile_ok": compile_ok,
        "targeted_issue_ids": targeted_issue_ids,
        "addressed_issue_ids": addressed_issue_ids,
        "covered_issue_ids": covered,
        "coverage_ratio": round(coverage_ratio, 4),
        "high_value_issue_ids": high_value_issue_ids,
        "high_value_covered_issue_ids": high_value_covered,
        "high_value_coverage_ratio": round(high_value_coverage_ratio, 4),
        "structured_change_count": structured_change_count,
        "structured_plan_coverage_ratio": round(structured_plan_coverage_ratio, 4),
        "unknown_addressed_issue_ids": unknown_addressed_issue_ids,
        "rewrite_plan": rewrite_plan,
        "before_file": str(before_file),
        "after_file": str(after_file),
        "raw_response_file": str(raw_response_file),
    }
    _safe_json_dump(
        artifact_path / f"rewrite_result_round_{round_index}.json",
        result,
    )
    return result


def save_self_review_artifacts(
    *,
    review_dir: str,
    ledger: Dict[str, Any],
    progress: Optional[Dict[str, Any]] = None,
    gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    review_path = Path(review_dir)
    review_path.mkdir(parents=True, exist_ok=True)
    ledger_file = review_path / "issue_ledger.json"
    report_file = review_path / "review_structured.md"
    _safe_json_dump(ledger_file, ledger)
    report_file.write_text(render_structured_review_markdown(ledger), encoding="utf-8")

    progress_file = ""
    if progress is not None:
        progress_file = str(review_path / "issue_progress.json")
        _safe_json_dump(Path(progress_file), progress)

    round_gate_file = ""
    round_gate_report_file = ""
    if gate is not None:
        round_gate_file = str(review_path / "round_gate.json")
        round_gate_report_file = str(review_path / "round_gate_report.md")
        _safe_json_dump(Path(round_gate_file), gate)
        Path(round_gate_report_file).write_text(
            render_round_gate_markdown(gate),
            encoding="utf-8",
        )

    return {
        "issue_ledger": str(ledger_file),
        "review_structured": str(report_file),
        "issue_progress": progress_file or "",
        "round_gate": round_gate_file or "",
        "round_gate_report": round_gate_report_file or "",
    }
