from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ai_scientist.config.paths import resolve_output_path
from ai_scientist.utils.high_quality_pipeline import (
    VENUE_PRESETS,
    evaluate_submission_acceptance,
    resolve_submission_acceptance_settings,
)


def _safe_load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_research_root(research_root: str | Path | None = None) -> Path:
    if research_root is None:
        return resolve_output_path()
    return Path(research_root).expanduser()


def iter_historical_quality_results(
    research_root: str | Path | None = None,
) -> list[Path]:
    root = _resolve_research_root(research_root)
    if not root.exists():
        return []

    quality_paths: list[Path] = []
    seen: set[Path] = set()
    try:
        from ai_scientist.utils.run_index import load_run_index

        index = load_run_index(root)
        for entry in index.get("entries", {}).values():
            run_path = Path(entry.get("path", ""))
            quality_path = run_path / "quality" / "high_quality_result.json"
            if quality_path.exists() and quality_path not in seen:
                quality_paths.append(quality_path)
                seen.add(quality_path)
    except Exception:
        pass

    if not quality_paths:
        for quality_path in root.rglob("quality/high_quality_result.json"):
            if quality_path not in seen:
                quality_paths.append(quality_path)
                seen.add(quality_path)

    quality_paths.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    return quality_paths[:500]


def summarize_historical_submission_outcomes(
    research_root: str | Path | None = None,
    *,
    target_venue: Optional[str] = None,
    max_entries: int = 200,
) -> dict:
    quality_paths = iter_historical_quality_results(research_root)
    entries = []

    for quality_path in quality_paths:
        result = _safe_load_json(quality_path)
        if not result:
            continue
        venue = result.get("target_venue") or "neurips"
        if target_venue and venue != target_venue:
            continue
        min_priority, max_blockers = resolve_submission_acceptance_settings(venue)
        acceptance = evaluate_submission_acceptance(
            result,
            require_quality_gate=True,
            min_submission_priority=min_priority,
            max_submission_blockers=max_blockers,
        )
        readiness = result.get("submission_readiness") or {}
        entries.append(
            {
                "venue": venue,
                "accepted": acceptance.get("accepted", False),
                "priority": result.get("submission_priority_score"),
                "blocker_count": result.get("blocker_count"),
                "categories": readiness.get("categories") or {},
                "blockers": (readiness.get("blockers") or [])[:6],
                "quality_score": result.get("quality_score_after"),
                "rigor_score": result.get("rigor_score_after"),
            }
        )
        if len(entries) >= max_entries:
            break

    if not entries:
        return {
            "target_venue": target_venue,
            "entries": 0,
            "accepted": 0,
            "acceptance_rate": None,
            "avg_priority": None,
            "avg_blockers": None,
            "category_failure_rates": {},
        }

    accepted = sum(1 for item in entries if item.get("accepted"))
    priorities = [item.get("priority") for item in entries if isinstance(item.get("priority"), (int, float))]
    blocker_counts = [item.get("blocker_count") for item in entries if isinstance(item.get("blocker_count"), int)]

    category_totals: dict[str, int] = {}
    for item in entries:
        if item.get("accepted"):
            continue
        for category, count in (item.get("categories") or {}).items():
            category_totals[category] = category_totals.get(category, 0) + int(count or 0)

    return {
        "target_venue": target_venue,
        "entries": len(entries),
        "accepted": accepted,
        "acceptance_rate": accepted / len(entries),
        "avg_priority": (sum(priorities) / len(priorities)) if priorities else None,
        "avg_blockers": (sum(blocker_counts) / len(blocker_counts)) if blocker_counts else None,
        "category_failure_rates": category_totals,
        "recent_sample": entries[:10],
    }


def recommend_rewrite_focus_adjustments(
    target_venue: str,
    *,
    research_root: str | Path | None = None,
) -> dict:
    summary = summarize_historical_submission_outcomes(research_root, target_venue=target_venue)
    category_rates = summary.get("category_failure_rates") or {}
    focus = {
        "dimension_boosts": {
            "structure": 0,
            "content": 0,
            "innovation": 0,
            "rigor": 0,
            "clarity": 0,
            "professionalism": 0,
        },
        "preferred_sections": [],
        "section_notes": {},
        "rationale": [],
        "history_summary": summary,
    }

    def add_section(name: str, note: str):
        if name not in focus["preferred_sections"]:
            focus["preferred_sections"].append(name)
        focus["section_notes"].setdefault(name, [])
        if note not in focus["section_notes"][name]:
            focus["section_notes"][name].append(note)

    entries = summary.get("entries", 0)
    if entries < 3:
        focus["rationale"].append("insufficient historical data for section-level rewrite guidance")
        return focus

    if category_rates.get("rigor", 0) >= 2:
        focus["dimension_boosts"]["rigor"] += 2
        focus["dimension_boosts"]["content"] += 1
        add_section("method", "历史结果显示 rigor 问题频繁，优先补实验设定、baseline、消融与显著性细节。")
        add_section("results", "历史结果显示 rigor 问题频繁，优先补关键对比、误差条和统计解释。")
        focus["rationale"].append("rigor blockers frequently hurt submission outcomes")
    if category_rates.get("claim", 0) >= 2:
        focus["dimension_boosts"]["clarity"] += 1
        focus["dimension_boosts"]["innovation"] += 1
        add_section("abstract", "历史结果显示 claim 支撑不足，优先把主 claim 绑定到具体数字和图表。")
        add_section("introduction", "历史结果显示 claim 支撑不足，收紧 contribution 表述，避免过度承诺。")
        add_section("conclusion", "历史结果显示 claim 支撑不足，保留边界条件和局限性。")
        focus["rationale"].append("claim blockers frequently hurt submission outcomes")
    if category_rates.get("numeric", 0) >= 2 or category_rates.get("evidence", 0) >= 2:
        focus["dimension_boosts"]["clarity"] += 1
        focus["dimension_boosts"]["content"] += 1
        add_section("abstract", "历史结果显示数字/证据覆盖不足，优先把 strongest numbers 放入摘要。")
        add_section("results", "历史结果显示数字/证据覆盖不足，优先把 strongest figure/table 和 key results 写清。")
        add_section("discussion", "历史结果显示数字/证据覆盖不足，解释 strongest result 的意义与边界。")
        focus["rationale"].append("numeric and evidence blockers frequently hurt submission outcomes")
    if category_rates.get("quality", 0) >= 2:
        focus["dimension_boosts"]["structure"] += 1
        focus["dimension_boosts"]["professionalism"] += 1
        add_section("title", "历史结果显示整体质量问题频繁，优先让题目更聚焦、更具体。")
        add_section("abstract", "历史结果显示整体质量问题频繁，摘要需更凝练、更像成熟投稿稿。")
        add_section("introduction", "历史结果显示整体质量问题频繁，引言需更快进入问题与贡献。")
        focus["rationale"].append("overall quality blockers frequently hurt submission outcomes")
    if category_rates.get("contribution", 0) >= 1:
        focus["dimension_boosts"]["innovation"] += 1
        focus["dimension_boosts"]["content"] += 1
        add_section("introduction", "历史结果显示 contribution framing 偏弱，明确 contribution bullets 和对应证据。")
        add_section("conclusion", "历史结果显示 contribution framing 偏弱，总结最重要贡献与 strongest evidence。")
        focus["rationale"].append("contribution framing often remains too weak historically")
    if target_venue == "nature" and category_rates.get("breakthrough", 0) >= 1:
        focus["dimension_boosts"]["innovation"] += 1
        focus["dimension_boosts"]["professionalism"] += 1
        add_section("introduction", "Nature 历史失败显示 broad significance 不足，需更清楚交代重大问题与跨领域意义。")
        add_section("discussion", "Nature 历史失败显示意义表达不足，需解释 strongest result 为什么重要。")
        focus["rationale"].append("breakthrough framing remains a recurring Nature-style weakness")

    if not focus["rationale"]:
        focus["rationale"].append("historical blocker patterns are stable; no extra section emphasis needed")

    return focus


def _style_key_from_variant(style_variant: str) -> str:
    variant = (style_variant or "").lower()
    if "清晰" in variant or "风险最小" in variant or "克制" in variant:
        return "conservative"
    if "贡献" in variant or "创新" in variant or "论证力度" in variant:
        return "assertive"
    if "专业" in variant or "成熟" in variant or "professional" in variant:
        return "professional"
    return "general"



def summarize_historical_rewrite_preferences(
    research_root: str | Path | None = None,
    *,
    target_venue: Optional[str] = None,
    max_entries: int = 200,
) -> dict:
    quality_paths = iter_historical_quality_results(research_root)
    preference = {
        "frontmatter": {},
        "sections": {},
        "counts": {"frontmatter": 0, "sections": 0},
        "target_venue": target_venue,
    }

    processed = 0
    for quality_path in quality_paths:
        result = _safe_load_json(quality_path)
        if not result:
            continue
        venue = result.get("target_venue") or "neurips"
        if target_venue and venue != target_venue:
            continue
        min_priority, max_blockers = resolve_submission_acceptance_settings(venue)
        acceptance = evaluate_submission_acceptance(
            result,
            require_quality_gate=True,
            min_submission_priority=min_priority,
            max_submission_blockers=max_blockers,
        )
        weight = 2.0 if acceptance.get("accepted") else 1.0
        priority = result.get("submission_priority_score") or 0
        weight += min(1.0, priority / 100.0)
        for round_item in result.get("rewrite_trace", []):
            for item in round_item.get("frontmatter", []):
                candidates = item.get("candidates", [])
                best_index = item.get("best_index")
                chosen = next((candidate for candidate in candidates if candidate.get("index") == best_index), None)
                if not chosen:
                    continue
                style_key = _style_key_from_variant(chosen.get("style_variant"))
                preference["frontmatter"][style_key] = preference["frontmatter"].get(style_key, 0.0) + weight
                preference["counts"]["frontmatter"] += 1
            for item in round_item.get("targets", []):
                summary = item.get("candidate_summary", {})
                candidates = summary.get("candidates", [])
                best_index = summary.get("best_index")
                chosen = next((candidate for candidate in candidates if candidate.get("index") == best_index), None)
                if not chosen:
                    continue
                style_key = _style_key_from_variant(chosen.get("style_variant"))
                preference["sections"][style_key] = preference["sections"].get(style_key, 0.0) + weight
                preference["counts"]["sections"] += 1
        processed += 1
        if processed >= max_entries:
            break

    return preference



def recommend_rewrite_style_preferences(
    target_venue: str,
    *,
    research_root: str | Path | None = None,
) -> dict:
    summary = summarize_historical_rewrite_preferences(research_root, target_venue=target_venue)

    def ordered(area: str) -> list[str]:
        scores = summary.get(area, {}) or {}
        if not scores:
            return ["professional", "conservative", "assertive"] if area == "frontmatter" else ["conservative", "professional", "assertive"]
        return [item[0] for item in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]

    return {
        "target_venue": target_venue,
        "frontmatter_style_order": ordered("frontmatter"),
        "section_style_order": ordered("sections"),
        "raw": summary,
    }


def _canonical_section_family(section_name: str) -> str:
    section = (section_name or "general").lower()
    for key in ["abstract", "title", "introduction", "method", "results", "discussion", "conclusion"]:
        if key in section:
            return key
    if "experiment" in section or "analysis" in section:
        return "results"
    return "general"



def summarize_historical_rewrite_effectiveness(
    research_root: str | Path | None = None,
    *,
    target_venue: Optional[str] = None,
    max_entries: int = 200,
) -> dict:
    quality_paths = iter_historical_quality_results(research_root)
    summary = {
        "target_venue": target_venue,
        "rounds": {"count": 0, "priority_delta_total": 0.0, "quality_delta_total": 0.0},
        "sections": {},
        "frontmatter": {},
    }

    processed = 0
    for quality_path in quality_paths:
        result = _safe_load_json(quality_path)
        if not result:
            continue
        venue = result.get("target_venue") or "neurips"
        if target_venue and venue != target_venue:
            continue
        min_priority, max_blockers = resolve_submission_acceptance_settings(venue)
        acceptance = evaluate_submission_acceptance(
            result,
            require_quality_gate=True,
            min_submission_priority=min_priority,
            max_submission_blockers=max_blockers,
        )
        weight = 2.0 if acceptance.get("accepted") else 1.0
        for round_item in result.get("rewrite_trace", []):
            pre_priority = round_item.get("pre_submission_priority_score")
            post_priority = round_item.get("post_submission_priority_score")
            pre_quality = round_item.get("pre_quality_score")
            post_quality = round_item.get("post_quality_score")
            if not isinstance(pre_priority, (int, float)) or not isinstance(post_priority, (int, float)):
                continue
            priority_delta = post_priority - pre_priority
            quality_delta = (post_quality - pre_quality) if isinstance(pre_quality, (int, float)) and isinstance(post_quality, (int, float)) else 0.0
            summary["rounds"]["count"] += 1
            summary["rounds"]["priority_delta_total"] += priority_delta
            summary["rounds"]["quality_delta_total"] += quality_delta

            for item in round_item.get("frontmatter", []):
                style_key = _style_key_from_variant(item.get("selected_style_variant"))
                bucket = summary["frontmatter"].setdefault(
                    style_key,
                    {"count": 0, "priority_delta_total": 0.0, "quality_delta_total": 0.0, "weight_total": 0.0},
                )
                bucket["count"] += 1
                bucket["priority_delta_total"] += priority_delta * weight
                bucket["quality_delta_total"] += quality_delta * weight
                bucket["weight_total"] += weight

            for item in round_item.get("targets", []):
                summary_meta = item.get("candidate_summary", {})
                style_key = _style_key_from_variant(summary_meta.get("selected_style_variant"))
                family = _canonical_section_family(item.get("section"))
                bucket = summary["sections"].setdefault(
                    family,
                    {"count": 0, "priority_delta_total": 0.0, "quality_delta_total": 0.0, "weight_total": 0.0, "styles": {}},
                )
                bucket["count"] += 1
                bucket["priority_delta_total"] += priority_delta * weight
                bucket["quality_delta_total"] += quality_delta * weight
                bucket["weight_total"] += weight
                style_bucket = bucket["styles"].setdefault(
                    style_key,
                    {"count": 0, "priority_delta_total": 0.0, "quality_delta_total": 0.0, "weight_total": 0.0},
                )
                style_bucket["count"] += 1
                style_bucket["priority_delta_total"] += priority_delta * weight
                style_bucket["quality_delta_total"] += quality_delta * weight
                style_bucket["weight_total"] += weight
        processed += 1
        if processed >= max_entries:
            break

    return summary



def recommend_rewrite_efficiency_controls(
    target_venue: str,
    *,
    research_root: str | Path | None = None,
) -> dict:
    summary = summarize_historical_rewrite_effectiveness(research_root, target_venue=target_venue)
    section_scores = []
    for name, stats in (summary.get("sections") or {}).items():
        weight_total = stats.get("weight_total") or 0.0
        if weight_total <= 0:
            continue
        avg_priority_delta = stats.get("priority_delta_total", 0.0) / weight_total
        section_scores.append((name, avg_priority_delta, stats.get("count", 0)))

    preferred_sections = [name for name, delta, count in sorted(section_scores, key=lambda item: item[1], reverse=True) if count >= 2 and delta > 0.5][:4]
    deprioritized_sections = [name for name, delta, count in sorted(section_scores, key=lambda item: item[1]) if count >= 2 and delta <= 0][:3]

    rounds = summary.get("rounds") or {}
    round_count = rounds.get("count", 0)
    avg_round_priority_delta = (rounds.get("priority_delta_total", 0.0) / round_count) if round_count else 0.0
    if round_count >= 5 and avg_round_priority_delta < 0.5:
        rewrite_round_adjustment = -1
    elif round_count >= 5 and avg_round_priority_delta > 1.5:
        rewrite_round_adjustment = 1
    else:
        rewrite_round_adjustment = 0

    rationale = []
    if preferred_sections:
        rationale.append(f"historically effective sections: {', '.join(preferred_sections)}")
    if deprioritized_sections:
        rationale.append(f"historically low-yield sections: {', '.join(deprioritized_sections)}")
    if rewrite_round_adjustment > 0:
        rationale.append("historical rewrite rounds often improve submission priority")
    elif rewrite_round_adjustment < 0:
        rationale.append("historical extra rewrite rounds often show limited priority uplift")
    if not rationale:
        rationale.append("historical rewrite efficiency is stable; keep default targeting")

    return {
        "target_venue": target_venue,
        "preferred_sections": preferred_sections,
        "deprioritized_sections": deprioritized_sections,
        "rewrite_round_adjustment": rewrite_round_adjustment,
        "avg_round_priority_delta": round(avg_round_priority_delta, 3),
        "rationale": rationale,
        "raw": summary,
    }


def recommend_reviewer_risk_mitigation(
    target_venue: str,
    *,
    research_root: str | Path | None = None,
) -> dict:
    summary = summarize_historical_submission_outcomes(research_root, target_venue=target_venue)
    category_rates = summary.get("category_failure_rates") or {}
    recent_sample = summary.get("recent_sample") or []

    result = {
        "anticipated_objections": [],
        "claim_softening_advice": [],
        "limitation_emphasis": [],
        "rebuttal_focus": [],
        "rationale": [],
    }

    def add(key: str, text: str):
        bucket = result[key]
        if text not in bucket:
            bucket.append(text)

    def add_blocker_objection(prefix: str):
        for item in recent_sample:
            for blocker in item.get("blockers", []):
                normalized = blocker.split("(")[0].strip()
                if prefix in normalized.lower() and len(result["anticipated_objections"]) < 6:
                    add("anticipated_objections", normalized)

    if category_rates.get("rigor", 0) >= 2:
        add("anticipated_objections", "Are the baselines, ablations, and statistical tests strong enough to support the claimed improvement?")
        add("claim_softening_advice", "Avoid robustness or superiority claims unless they are directly backed by ablations, significance tests, or strong baseline comparisons.")
        add("limitation_emphasis", "State clearly if robustness, sensitivity, or broader baseline coverage remains incomplete.")
        add("rebuttal_focus", "Prepare a concise defense built around baselines, ablations, significance, and implementation details.")
        add("rationale", "rigor objections recur historically")
        add_blocker_objection("rigor")
    if category_rates.get("claim", 0) >= 2:
        add("anticipated_objections", "Which exact figure, table, or quantitative result supports each main claim?")
        add("claim_softening_advice", "Prefer restrained wording such as 'suggests' or 'is consistent with' when evidence is incomplete.")
        add("limitation_emphasis", "Explicitly call out where evidence is promising but not yet definitive.")
        add("rebuttal_focus", "Map each major claim to one strongest result and one quantitative takeaway.")
        add("rationale", "claim-support objections recur historically")
        add_blocker_objection("claim")
    if category_rates.get("numeric", 0) >= 2 or category_rates.get("evidence", 0) >= 2:
        add("anticipated_objections", "Can the authors quantify the gain more precisely and tie it to the strongest visual/table evidence?")
        add("claim_softening_advice", "Avoid saying 'significant improvement' without naming the strongest numbers or comparison points.")
        add("limitation_emphasis", "Acknowledge where the evidence package is still thin or where visuals are not yet comprehensive.")
        add("rebuttal_focus", "Lead with the strongest numerical result, then point to the corresponding figure/table and comparison baseline.")
        add("rationale", "numeric/evidence objections recur historically")
        add_blocker_objection("numeric")
        add_blocker_objection("evidence")
    if category_rates.get("contribution", 0) >= 1:
        add("anticipated_objections", "What is the concrete contribution beyond a narrower benchmark improvement or engineering tweak?")
        add("claim_softening_advice", "Do not overstate novelty when the contribution is mainly empirical or integrative.")
        add("limitation_emphasis", "Clarify what the method does not yet establish as a general principle.")
        add("rebuttal_focus", "State the contribution in one sentence, then connect it to one strongest result and one limitation.")
        add("rationale", "contribution-framing objections recur historically")
    if category_rates.get("quality", 0) >= 2:
        add("anticipated_objections", "Is the manuscript polished and focused enough for a competitive submission, or is the story still diffuse?")
        add("claim_softening_advice", "Prefer sharper, narrower claims over broad but weakly supported framing.")
        add("limitation_emphasis", "Acknowledge when the current draft prioritizes a narrow validated contribution over a broader speculative story.")
        add("rebuttal_focus", "Defend the narrative by identifying the one central claim and the strongest evidence chain behind it.")
        add("rationale", "overall manuscript-quality objections recur historically")
    if target_venue == "nature" and category_rates.get("breakthrough", 0) >= 1:
        add("anticipated_objections", "Why is this a broad scientific advance rather than a strong but incremental technical result?")
        add("claim_softening_advice", "Use breakthrough framing cautiously unless broad significance is strongly evidenced.")
        add("limitation_emphasis", "State clearly where broad significance remains a forward-looking interpretation rather than a demonstrated fact.")
        add("rebuttal_focus", "Prepare a broad-significance defense grounded in the problem importance and strongest evidence, while keeping caveats explicit.")
        add("rationale", "Nature-style broad-significance objections recur historically")

    if not result["anticipated_objections"]:
        add("anticipated_objections", "What is the strongest empirical or conceptual takeaway of this paper?")
        add("rebuttal_focus", "Anchor every response in one strongest result and one explicit limitation.")
        add("rationale", "historical reviewer risk pattern is currently mild")

    for key in ["anticipated_objections", "claim_softening_advice", "limitation_emphasis", "rebuttal_focus", "rationale"]:
        result[key] = result[key][:6]
    return result


def recommend_section_reviewer_language_guidance(
    target_venue: str,
    section_name: str,
    *,
    research_root: str | Path | None = None,
) -> dict:
    section = (section_name or "general").lower()
    risks = recommend_reviewer_risk_mitigation(target_venue, research_root=research_root)

    guidance = {
        "section": section_name,
        "recommended_tone": [],
        "claim_softening": [],
        "limitation_emphasis": [],
        "objection_preemption": [],
    }

    def add(bucket: str, text: str):
        values = guidance[bucket]
        if text and text not in values:
            values.append(text)

    claim_softening = risks.get("claim_softening_advice", [])
    limitation_emphasis = risks.get("limitation_emphasis", [])
    rebuttal_focus = risks.get("rebuttal_focus", [])
    objections = risks.get("anticipated_objections", [])

    if any(key in section for key in ["title", "abstract"]):
        add("recommended_tone", "Prefer narrower, evidence-backed phrasing over broad or absolute framing.")
        add("recommended_tone", "Make the strongest numerical takeaway explicit if available.")
        add("claim_softening", claim_softening[0] if claim_softening else "Use restrained verbs and avoid unsupported superiority claims.")
        add("limitation_emphasis", limitation_emphasis[0] if limitation_emphasis else "Include a brief scope caveat when evidence is incomplete.")
    if "introduction" in section:
        add("recommended_tone", "Frame novelty narrowly and connect it to a concrete contribution list.")
        add("claim_softening", claim_softening[0] if claim_softening else "Avoid over-claiming broad novelty before evidence is presented.")
        add("limitation_emphasis", limitation_emphasis[0] if limitation_emphasis else "Clarify the intended scope of the contribution early.")
        add("objection_preemption", rebuttal_focus[0] if rebuttal_focus else "Anticipate the strongest novelty/significance objection in the motivation paragraph.")
    if any(key in section for key in ["results", "experiment", "analysis"]):
        add("recommended_tone", "Tie every strong statement to one figure/table and one quantitative comparison.")
        add("claim_softening", claim_softening[0] if claim_softening else "Avoid saying 'significant improvement' without concrete numbers.")
        add("objection_preemption", rebuttal_focus[0] if rebuttal_focus else "Preempt reviewer concern by naming the exact strongest evidence item.")
    if any(key in section for key in ["discussion", "conclusion"]):
        add("recommended_tone", "End with one explicit caveat and keep broad-significance language restrained.")
        add("claim_softening", claim_softening[0] if claim_softening else "Prefer cautious language such as 'suggests' when generality is not fully established.")
        add("limitation_emphasis", limitation_emphasis[0] if limitation_emphasis else "Add one explicit limitation sentence and one scope boundary.")
        add("objection_preemption", rebuttal_focus[0] if rebuttal_focus else "Address the most likely reviewer concern before the final takeaway sentence.")
    if "method" in section:
        add("recommended_tone", "Describe capability precisely; avoid generality or robustness claims beyond evaluation scope.")
        add("claim_softening", claim_softening[0] if claim_softening else "Do not imply universality if experiments only cover a narrow setting.")

    if not any(guidance.values()):
        add("recommended_tone", "Keep the language evidence-backed, concrete, and modest.")
    if objections:
        for item in objections[:2]:
            add("objection_preemption", item)

    for key in guidance:
        if isinstance(guidance[key], list):
            guidance[key] = guidance[key][:4]
    return guidance


def recommend_submission_strategy_adjustments(
    target_venue: str,
    *,
    research_root: str | Path | None = None,
) -> dict:
    venue = target_venue or "neurips"
    venue_config = VENUE_PRESETS.get(venue, VENUE_PRESETS["neurips"])
    summary = summarize_historical_submission_outcomes(research_root, target_venue=venue)

    adjustments = {
        "cite_round_boost": 0,
        "writeup_retry_boost": 0,
        "review_reflection_boost": 0,
        "review_ensemble_boost": 0,
        "review_fewshot_boost": 0,
        "rationale": [],
        "history_summary": summary,
        "effective_defaults": {
            "num_cite_rounds": venue_config.get("min_cite_rounds", 20),
            "writeup_retries": venue_config.get("min_writeup_retries", 4),
            "review_reflections": venue_config.get("min_review_reflections", 2),
            "review_ensemble": venue_config.get("min_review_ensemble", 3),
            "review_fewshot": venue_config.get("min_review_fewshot", 2),
        },
    }

    entries = summary.get("entries", 0)
    acceptance_rate = summary.get("acceptance_rate")
    category_rates = summary.get("category_failure_rates") or {}

    if entries < 3 or acceptance_rate is None:
        adjustments["rationale"].append("insufficient historical data; using venue defaults")
        return adjustments

    if acceptance_rate < 0.25:
        adjustments["writeup_retry_boost"] += 1
        adjustments["review_reflection_boost"] += 1
        adjustments["review_ensemble_boost"] += 1
        adjustments["rationale"].append("historical acceptance is low; increasing rewrite and review depth")
    elif acceptance_rate < 0.45:
        adjustments["writeup_retry_boost"] += 1
        adjustments["rationale"].append("historical acceptance is middling; adding one extra writeup retry")

    if category_rates.get("rigor", 0) >= 2:
        adjustments["review_reflection_boost"] += 1
        adjustments["review_ensemble_boost"] += 1
        adjustments["rationale"].append("rigor blockers are common historically; increasing review depth and ensemble")
    if category_rates.get("claim", 0) >= 2:
        adjustments["review_fewshot_boost"] += 1
        adjustments["rationale"].append("claim blockers are common historically; increasing few-shot review guidance")
    if category_rates.get("numeric", 0) >= 2 or category_rates.get("evidence", 0) >= 2:
        adjustments["cite_round_boost"] += 2
        adjustments["writeup_retry_boost"] += 1
        adjustments["rationale"].append("numeric/evidence blockers are common historically; increasing citation and writeup budget")
    if category_rates.get("quality", 0) >= 2:
        adjustments["writeup_retry_boost"] += 1
        adjustments["rationale"].append("quality blockers are common historically; allowing one more polish pass")
    if venue == "nature" and acceptance_rate < 0.5:
        adjustments["review_ensemble_boost"] += 1
        adjustments["review_reflection_boost"] += 1
        adjustments["rationale"].append("Nature-style bar remains hard historically; tightening review process")

    adjustments["cite_round_boost"] = min(adjustments["cite_round_boost"], 6)
    adjustments["writeup_retry_boost"] = min(adjustments["writeup_retry_boost"], 3)
    adjustments["review_reflection_boost"] = min(adjustments["review_reflection_boost"], 2)
    adjustments["review_ensemble_boost"] = min(adjustments["review_ensemble_boost"], 2)
    adjustments["review_fewshot_boost"] = min(adjustments["review_fewshot_boost"], 1)

    if not adjustments["rationale"]:
        adjustments["rationale"].append("historical outcomes are stable; venue defaults look sufficient")

    adjustments["effective_defaults"] = {
        "num_cite_rounds": venue_config.get("min_cite_rounds", 20) + adjustments["cite_round_boost"],
        "writeup_retries": venue_config.get("min_writeup_retries", 4) + adjustments["writeup_retry_boost"],
        "review_reflections": venue_config.get("min_review_reflections", 2) + adjustments["review_reflection_boost"],
        "review_ensemble": venue_config.get("min_review_ensemble", 3) + adjustments["review_ensemble_boost"],
        "review_fewshot": venue_config.get("min_review_fewshot", 2) + adjustments["review_fewshot_boost"],
    }
    return adjustments
