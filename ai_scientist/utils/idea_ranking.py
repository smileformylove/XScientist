from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

from ai_scientist.config.paths import resolve_output_path

STOPWORDS = {
    "about", "above", "after", "again", "against", "among", "approach", "baseline",
    "between", "challenge", "compare", "comparison", "dataset", "during", "efficient",
    "experiment", "experiments", "figure", "framework", "general", "improve", "improves",
    "improving", "introduction", "large", "method", "methods", "model", "models", "novel",
    "paper", "problem", "results", "section", "show", "shows", "study", "system",
    "table", "their", "these", "this", "using", "with", "work", "works",
}


def _extract_json(text: str) -> dict:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _idea_text(idea: dict) -> str:
    return "\n".join(
        str(idea.get(key, ""))
        for key in [
            "Name",
            "Title",
            "Abstract",
            "Short Hypothesis",
            "Hypothesis",
            "Impact",
            "Field",
            "Task",
            "Problem",
            "Keywords",
        ]
        if idea.get(key)
    )


def _tokenize_idea(idea: dict) -> set[str]:
    text = _idea_text(idea).lower()
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", text)
    return {
        word
        for word in words
        if len(word) >= 4 and word not in STOPWORDS and not word.isdigit()
    }


def _similarity_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    jaccard = intersection / len(left | right)
    containment = intersection / max(1, min(len(left), len(right)))
    return round(0.6 * jaccard + 0.4 * containment, 4)


def _safe_load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_historical_quality_paths(research_root: str | Path) -> list[Path]:
    root = Path(research_root)
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
    return quality_paths[:400]


def _load_historical_acceptance_profile(
    research_root: str | Path,
    *,
    target_venue: str | None,
) -> list[dict]:
    from ai_scientist.utils.high_quality_pipeline import (
        evaluate_submission_acceptance,
        resolve_submission_acceptance_settings,
    )

    profile = []
    for quality_path in _iter_historical_quality_paths(research_root):
        run_dir = quality_path.parent.parent
        idea = _safe_load_json(run_dir / "idea.json")
        quality_result = _safe_load_json(quality_path)
        if not idea or not quality_result:
            continue

        venue = quality_result.get("target_venue") or target_venue or "neurips"
        min_priority, max_blockers = resolve_submission_acceptance_settings(venue)
        acceptance = evaluate_submission_acceptance(
            quality_result,
            require_quality_gate=True,
            min_submission_priority=min_priority,
            max_submission_blockers=max_blockers,
        )
        tokens = _tokenize_idea(idea)
        if len(tokens) < 3:
            continue

        profile.append(
            {
                "name": idea.get("Name") or run_dir.name,
                "title": idea.get("Title") or "",
                "target_venue": venue,
                "accepted": acceptance.get("accepted", False),
                "priority": quality_result.get("submission_priority_score"),
                "tier": quality_result.get("submission_priority_tier"),
                "blocker_count": quality_result.get("blocker_count"),
                "tokens": tokens,
            }
        )
    return profile


def _apply_historical_acceptance_prior(
    score: dict,
    idea: dict,
    *,
    history: list[dict],
    target_venue: str | None,
) -> dict:
    base_total = score.get("total_score", 3.0)
    if not history:
        score["historical_acceptance_adjustment"] = 0.0
        score["ranking_score"] = round(base_total, 3)
        score["historical_rationale"] = "no historical acceptance data available"
        score["historical_matches"] = []
        return score

    current_tokens = _tokenize_idea(idea)
    matches = []
    for item in history:
        similarity = _similarity_score(current_tokens, item.get("tokens", set()))
        if similarity < 0.08:
            continue
        venue_weight = 1.15 if target_venue and item.get("target_venue") == target_venue else 1.0
        weighted_similarity = similarity * venue_weight
        matches.append(
            {
                "name": item.get("name"),
                "title": item.get("title"),
                "accepted": item.get("accepted"),
                "target_venue": item.get("target_venue"),
                "priority": item.get("priority"),
                "tier": item.get("tier"),
                "blocker_count": item.get("blocker_count"),
                "similarity": round(similarity, 4),
                "weighted_similarity": round(weighted_similarity, 4),
            }
        )

    matches.sort(key=lambda item: item.get("weighted_similarity", 0), reverse=True)
    accepted_matches = [item for item in matches if item.get("accepted")][:3]
    rejected_matches = [item for item in matches if not item.get("accepted")][:3]

    positive_signal = sum(
        item.get("weighted_similarity", 0) * (0.6 + min(0.4, (item.get("priority") or 0) / 100.0))
        for item in accepted_matches
    )
    negative_signal = sum(
        item.get("weighted_similarity", 0) * (0.8 + min(0.2, ((100 - (item.get("priority") or 0)) / 100.0)))
        for item in rejected_matches
    )
    adjustment = round(max(-0.5, min(0.6, (positive_signal - negative_signal) * 1.8)), 3)
    ranking_score = round(max(1.0, min(5.0, base_total + adjustment)), 3)

    rationales = []
    if accepted_matches:
        rationales.append(
            "similar accepted ideas: " + ", ".join(item.get("name") or item.get("title") or "unknown" for item in accepted_matches)
        )
    if rejected_matches:
        rationales.append(
            "similar weak outcomes: " + ", ".join(item.get("name") or item.get("title") or "unknown" for item in rejected_matches)
        )
    if not rationales:
        rationales.append("no strong historical matches found")

    score["historical_acceptance_adjustment"] = adjustment
    score["ranking_score"] = ranking_score
    score["historical_rationale"] = " | ".join(rationales)
    score["historical_matches"] = matches[:6]
    return score


def _heuristic_score_idea(idea: dict, target_venue: str | None) -> dict:
    title = idea.get("Title", "")
    hypothesis = idea.get("Short Hypothesis", "") or idea.get("Hypothesis", "")
    related = idea.get("Related Work", "")
    experiments = json.dumps(idea, ensure_ascii=False)

    novelty = 2.5 + min(1.5, 0.3 * sum(keyword in experiments.lower() for keyword in ["novel", "new", "first", "dynamic", "adaptive", "real-world"]))
    feasibility = 2.5 + min(1.5, 0.0015 * len(hypothesis))
    rigor = 2.0 + min(2.0, 0.001 * len(related) + 0.2 * sum(keyword in experiments.lower() for keyword in ["baseline", "experiment", "compare", "analysis", "dataset"]))
    impact = 2.5 + min(1.5, 0.001 * len(title) + 0.3 * sum(keyword in experiments.lower() for keyword in ["significant", "real-world", "generalization", "impact", "efficient"]))
    writing = 2.5 + min(1.5, 0.001 * (len(title) + len(hypothesis)))

    breakthrough = 2.0 + min(2.5, 0.4 * sum(keyword in experiments.lower() for keyword in [
        "fundamental", "grand challenge", "broad impact", "real-world", "scientific discovery",
        "cross-domain", "major challenge", "medical", "biology", "climate",
    ]))

    if target_venue == "nature":
        impact += 0.3
        rigor += 0.2
        breakthrough += 0.5
    elif target_venue in {"neurips", "iclr", "cvpr"}:
        novelty += 0.2
        rigor += 0.2

    scores = [max(1.0, min(5.0, value)) for value in [novelty, feasibility, rigor, impact, writing, breakthrough]]
    total = sum(scores) / len(scores)
    return {
        "novelty": scores[0],
        "feasibility": scores[1],
        "rigor_potential": scores[2],
        "impact": scores[3],
        "writing_potential": scores[4],
        "breakthrough_potential": scores[5],
        "total_score": total,
        "rationale": "heuristic fallback ranking",
    }


def _mark_ranking_fallback(
    payload: dict,
    *,
    stage: str,
    reason: str,
    detail: str | None = None,
) -> dict:
    payload["fallback_used"] = True
    payload["fallback_stage"] = stage
    payload["fallback_reason"] = reason
    if detail:
        payload["fallback_detail"] = detail
    return payload


def score_idea(idea: dict, *, model: str, logger: Callable[[str], None] = print) -> dict:
    return score_idea_for_venue(idea, model=model, target_venue=None, logger=logger)


def score_idea_for_venue(
    idea: dict,
    *,
    model: str,
    target_venue: str | None,
    logger: Callable[[str], None] = print,
) -> dict:
    try:
        from ai_scientist.llm import create_client, get_response_from_llm

        client, client_model = create_client(model)
    except Exception as exc:
        result = _heuristic_score_idea(idea, target_venue)
        result = _mark_ranking_fallback(
            result,
            stage="client_creation",
            reason="client_creation_failed",
            detail=str(exc),
        )
        result["idea_name"] = idea.get("Name")
        result["title"] = idea.get("Title")
        result["target_venue"] = target_venue
        return result
    venue_instructions = {
        None: "平衡评估创新性、可验证性和论文叙事潜力。",
        "neurips": "优先考虑方法创新、实验严谨性、baseline 对比和清晰技术贡献。",
        "iclr": "优先考虑核心 insight、理论/经验支撑与研究叙事完整性。",
        "cvpr": "优先考虑视觉结果质量、实验覆盖度和结果呈现潜力。",
        "journal": "优先考虑完整性、稳定性、系统性实验和长期学术价值。",
        "nature": "优先考虑重大问题、广泛影响、跨领域意义和强证据链潜力。",
    }
    prompt = f"""
请评估以下研究想法是否值得进入高质量论文生成流程。

研究想法:
{json.dumps(idea, indent=2, ensure_ascii=False)}

请从以下维度打分（1-5）:
1. novelty: 创新性
2. feasibility: 可实现性
3. rigor_potential: 形成严谨实验与 baseline 的潜力
4. impact: 潜在影响力
5. writing_potential: 形成强论文叙事的潜力
6. breakthrough_potential: 是否在解决重大问题并具备突破性影响潜力

目标 venue:
{target_venue or 'generic_high_quality'}

venue 约束:
{venue_instructions.get(target_venue)}

然后给出 total_score（1-5）和简短 rationale。

请返回 JSON:
{{
  "novelty": 4.2,
  "feasibility": 3.8,
  "rigor_potential": 4.1,
  "impact": 3.9,
  "writing_potential": 4.0,
  "breakthrough_potential": 4.4,
  "total_score": 4.0,
  "rationale": "..."
}}
"""
    response, _ = get_response_from_llm(
        prompt=prompt,
        client=client,
        model=client_model,
        system_message="你是资深科研选题评审专家，负责挑选最适合产出高质量论文的研究想法。",
        temperature=0.2,
    )
    result = _extract_json(response)
    if not result:
        result = _heuristic_score_idea(idea, target_venue)
        result["rationale"] = "heuristic fallback ranking due to parse failure"
        result = _mark_ranking_fallback(
            result,
            stage="response_parsing",
            reason="response_parse_failed",
        )
    else:
        result["fallback_used"] = False
    result["idea_name"] = idea.get("Name")
    result["title"] = idea.get("Title")
    result["target_venue"] = target_venue
    return result


def rank_ideas(
    ideas: list[dict],
    *,
    model: str,
    target_venue: str | None = None,
    prioritize_breakthrough: bool = False,
    research_root: str | Path | None = None,
    output_path: str | Path | None = None,
    logger: Callable[[str], None] = print,
) -> list[dict]:
    if research_root is None:
        research_root = resolve_output_path()

    historical_profile = []
    historical_profile = _load_historical_acceptance_profile(
        research_root,
        target_venue=target_venue,
    )
    if historical_profile:
        logger(f"Loaded {len(historical_profile)} historical idea outcomes for ranking")

    rankings = []
    for idx, idea in enumerate(ideas):
        logger(f"Ranking idea {idx}: {idea.get('Name', f'idea_{idx}')}")
        score = score_idea_for_venue(idea, model=model, target_venue=target_venue, logger=logger)
        score = _apply_historical_acceptance_prior(
            score,
            idea,
            history=historical_profile,
            target_venue=target_venue,
        )
        score["idea_idx"] = idx
        rankings.append(score)

    if prioritize_breakthrough:
        rankings.sort(
            key=lambda item: (
                item.get("breakthrough_potential", 0),
                item.get("ranking_score", item.get("total_score", 0)),
                item.get("impact", 0),
                item.get("rigor_potential", 0),
                item.get("total_score", 0),
            ),
            reverse=True,
        )
    else:
        rankings.sort(
            key=lambda item: (
                item.get("ranking_score", item.get("total_score", 0)),
                item.get("total_score", 0),
                item.get("rigor_potential", 0),
                item.get("impact", 0),
            ),
            reverse=True,
        )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rankings, f, indent=2, ensure_ascii=False)
    return rankings
