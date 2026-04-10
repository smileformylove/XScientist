from __future__ import annotations

"""Review job persistence inspired by tool-grounded reviewer runtimes."""

import json
import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import load_contract_artifact, save_contract_artifact


REVIEW_ROLES = (
    "novelty",
    "rigor",
    "clarity",
    "reproducibility",
    "skeptical_pc_member",
    "claim_cross_examiner",
    "reproducibility_assassin",
    "novelty_executioner",
    "stats_sniper",
    "related_work_skeptic",
    "meta_reviewer",
    "desk_reject_editor",
    "style_snob",
)
_ISSUE_SPACE_RE = re.compile(r"\s+")
_ISSUE_TOKEN_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
_SECTION_ALIASES = {
    "abstract": ("abstract", "summary"),
    "introduction": ("introduction", "intro", "motivation"),
    "related_work": ("related work", "background", "prior work"),
    "method": ("method", "approach", "algorithm", "model"),
    "experiments": ("experiment", "evaluation", "setup", "benchmark"),
    "results": ("result", "results", "finding", "findings"),
    "discussion": ("discussion", "analysis", "interpretation"),
    "limitations": ("limitation", "limitations", "caveat", "caveats"),
    "conclusion": ("conclusion", "takeaway"),
    "lessons_learned": ("lessons learned", "lesson"),
}
_ROLE_DEFAULT_SECTIONS = {
    "novelty": ("introduction", "related_work"),
    "rigor": ("experiments", "results"),
    "clarity": ("abstract", "introduction"),
    "reproducibility": ("method", "experiments"),
    "skeptical_pc_member": ("introduction", "results"),
    "claim_cross_examiner": ("abstract", "results", "discussion"),
    "reproducibility_assassin": ("method", "experiments"),
    "novelty_executioner": ("title", "abstract", "introduction", "related_work"),
    "stats_sniper": ("experiments", "results", "discussion"),
    "related_work_skeptic": ("introduction", "related_work", "discussion"),
    "meta_reviewer": ("abstract", "introduction", "results", "discussion"),
    "desk_reject_editor": ("title", "abstract", "introduction"),
    "style_snob": ("abstract", "introduction", "discussion"),
}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_issue_text(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    normalized = _ISSUE_SPACE_RE.sub(" ", normalized)
    return normalized


def _issue_id(text: Any) -> str:
    normalized = _normalize_issue_text(text)
    if not normalized:
        normalized = "empty_issue"
    return "RVW-" + hashlib.md5(normalized.encode("utf-8")).hexdigest()[:10]


def _coerce_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _dedupe_texts(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        normalized = _normalize_issue_text(text)
        if not text or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result


def _tokenize_text(text: Any) -> set[str]:
    normalized = _normalize_issue_text(text)
    if not normalized:
        return set()
    return {
        token
        for token in _ISSUE_TOKEN_RE.split(normalized)
        if token and len(token) > 1
    }


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _build_target_entry(
    target_id: str,
    *,
    label: str,
    target_type: str,
    texts: list[str],
) -> dict[str, Any]:
    merged_texts = [str(item).strip() for item in texts if str(item).strip()]
    token_set: set[str] = set()
    for item in [target_id, label] + merged_texts:
        token_set.update(_tokenize_text(item))
    return {
        "id": str(target_id).strip(),
        "label": str(label).strip() or str(target_id).strip(),
        "type": target_type,
        "texts": merged_texts,
        "tokens": token_set,
    }


def _build_issue_binding_catalog(project_root: str | Path) -> dict[str, Any]:
    claim_graph = load_contract_artifact(
        project_root, "claim_evidence_graph", default={}
    ) or {}
    figure_spec = load_contract_artifact(project_root, "figure_spec", default={}) or {}
    manuscript_state = load_contract_artifact(
        project_root, "manuscript_state", default={}
    ) or {}

    claim_nodes = [
        node
        for node in (claim_graph.get("nodes") or [])
        if isinstance(node, dict) and node.get("type") == "claim"
    ]
    claims = [
        _build_target_entry(
            str(node.get("id") or f"claim_{idx}"),
            label=str(node.get("label") or node.get("id") or f"Claim {idx}"),
            target_type="claim",
            texts=[
                str(node.get("label") or ""),
                str(node.get("status") or ""),
            ],
        )
        for idx, node in enumerate(claim_nodes)
    ]

    figures_raw = [
        item for item in (figure_spec.get("figures") or []) if isinstance(item, dict)
    ]
    figures = [
        _build_target_entry(
            str(item.get("figure_id") or f"figure_{idx}"),
            label=str(
                item.get("suggested_title")
                or item.get("caption_intent")
                or item.get("figure_id")
                or f"Figure {idx}"
            ),
            target_type="figure",
            texts=[
                str(item.get("caption_intent") or ""),
                str(item.get("figure_type") or ""),
                str(item.get("claim_id") or ""),
                str(item.get("paper_slot") or ""),
            ],
        )
        for idx, item in enumerate(figures_raw)
    ]

    section_briefs = (
        manuscript_state.get("section_briefs")
        if isinstance(manuscript_state.get("section_briefs"), dict)
        else {}
    )
    outline = _coerce_str_list(manuscript_state.get("outline")) or list(
        _SECTION_ALIASES.keys()
    )
    sections = [
        _build_target_entry(
            section_name,
            label=section_name.replace("_", " ").title(),
            target_type="section",
            texts=[str(section_briefs.get(section_name) or "")]
            + list(_SECTION_ALIASES.get(section_name, ())),
        )
        for section_name in outline
    ]

    section_claim_bindings = (
        manuscript_state.get("section_claim_bindings")
        if isinstance(manuscript_state.get("section_claim_bindings"), dict)
        else {}
    )
    section_figure_bindings = (
        manuscript_state.get("section_figure_bindings")
        if isinstance(manuscript_state.get("section_figure_bindings"), dict)
        else {}
    )
    figure_to_claim = {
        str(item.get("figure_id") or "").strip(): str(item.get("claim_id") or "").strip()
        for item in figures_raw
        if str(item.get("figure_id") or "").strip()
    }
    claim_to_figures: dict[str, list[str]] = {}
    for figure_id, claim_id in figure_to_claim.items():
        if not claim_id:
            continue
        claim_to_figures.setdefault(claim_id, []).append(figure_id)
    return {
        "claims": claims,
        "figures": figures,
        "sections": sections,
        "claim_labels": {
            str(item.get("id") or "").strip(): str(item.get("label") or "").strip()
            for item in claims
            if str(item.get("id") or "").strip()
        },
        "figure_labels": {
            str(item.get("id") or "").strip(): str(item.get("label") or "").strip()
            for item in figures
            if str(item.get("id") or "").strip()
        },
        "section_labels": {
            str(item.get("id") or "").strip(): str(item.get("label") or "").strip()
            for item in sections
            if str(item.get("id") or "").strip()
        },
        "section_claim_bindings": {
            str(key): [str(item) for item in _coerce_str_list(value)]
            for key, value in section_claim_bindings.items()
        },
        "section_figure_bindings": {
            str(key): [str(item) for item in _coerce_str_list(value)]
            for key, value in section_figure_bindings.items()
        },
        "figure_to_claim": figure_to_claim,
        "claim_to_figures": claim_to_figures,
    }


def _match_targets(
    issue_text: str,
    targets: list[dict[str, Any]],
    *,
    limit: int = 2,
) -> list[str]:
    issue_normalized = _normalize_issue_text(issue_text)
    issue_tokens = _tokenize_text(issue_text)
    scored: list[tuple[float, str]] = []
    for target in targets:
        target_id = str(target.get("id") or "").strip()
        if not target_id:
            continue
        score = 0.0
        target_tokens = set(target.get("tokens") or set())
        label = _normalize_issue_text(target.get("label") or "")
        if target_id.lower() in issue_normalized:
            score += 3.0
        if label and label in issue_normalized:
            score += 2.5
        overlap = issue_tokens & target_tokens
        if overlap:
            score += len(overlap) / max(len(target_tokens), 1)
            if len(overlap) >= 2:
                score += 0.5
        if score > 0.6:
            scored.append((score, target_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [target_id for _, target_id in scored[:limit]]


def _format_target_label(target_id: str, *, catalog: dict[str, Any]) -> str:
    text = str(target_id or "").strip()
    if not text:
        return "unbound target"
    for key in ("claim_labels", "figure_labels", "section_labels"):
        label = str((catalog.get(key) or {}).get(text) or "").strip()
        if label:
            return label
    return text.replace("_", " ")


def _match_supporting_texts(
    issue_text: str,
    candidates: list[str],
    *,
    target_ids: list[str] | None = None,
    limit: int = 3,
) -> list[str]:
    normalized_issue = _normalize_issue_text(issue_text)
    issue_tokens = _tokenize_text(issue_text)
    target_ids = [str(item).strip() for item in target_ids or [] if str(item).strip()]
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for raw in candidates:
        text = str(raw or "").strip()
        if not text:
            continue
        normalized = _normalize_issue_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        score = 0.0
        if normalized_issue and normalized_issue in normalized:
            score += 2.0
        candidate_tokens = _tokenize_text(text)
        overlap = issue_tokens & candidate_tokens
        if overlap:
            score += len(overlap) / max(len(candidate_tokens), 1)
            if len(overlap) >= 2:
                score += 0.5
        for target_id in target_ids:
            target_norm = _normalize_issue_text(target_id)
            if target_norm and target_norm in normalized:
                score += 1.25
        if score > 0.45:
            scored.append((score, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, text in scored[:limit]]


def _default_repair_actions(
    issue_record: dict[str, Any],
    *,
    primary_target_type: str,
    primary_target_id: str | None,
    primary_target_label: str,
) -> list[str]:
    issue_text = str(issue_record.get("text") or "").strip()
    role = str(issue_record.get("role") or "clarity").strip()
    section_ids = _coerce_str_list(issue_record.get("section_ids"))
    section_text = (
        section_ids[0].replace("_", " ")
        if section_ids
        else primary_target_label
    )
    if primary_target_type == "figure" and primary_target_id:
        return [
            f"Revise {primary_target_id} ({primary_target_label}) and its caption to resolve: {issue_text}"
        ]
    if primary_target_type == "claim" and primary_target_id:
        return [
            f"Tighten the evidence and wording for {primary_target_id} ({primary_target_label}) to resolve: {issue_text}"
        ]
    if primary_target_type == "section" and section_text:
        if role in {"rigor", "reproducibility"}:
            return [
                f"Revise the {section_text} section with stronger experimental evidence to resolve: {issue_text}"
            ]
        return [
            f"Revise the {section_text} section narrative to resolve: {issue_text}"
        ]
    return [f"Create a focused repair plan for reviewer issue: {issue_text}"]


def _default_verification_checks(
    issue_record: dict[str, Any],
    *,
    primary_target_type: str,
    primary_target_id: str | None,
    primary_target_label: str,
) -> list[str]:
    issue_text = str(issue_record.get("text") or "").strip()
    if primary_target_type == "figure" and primary_target_id:
        return [
            f"Verify {primary_target_id} ({primary_target_label}) has traceable data and directly addresses the reviewer concern."
        ]
    if primary_target_type == "claim" and primary_target_id:
        return [
            f"Verify the evidence linked to {primary_target_id} ({primary_target_label}) now supports the revised claim."
        ]
    if primary_target_type == "section" and primary_target_id:
        return [
            f"Verify the {primary_target_id.replace('_', ' ')} section explicitly resolves: {issue_text}"
        ]
    return []


def _compute_repair_priority(issue_record: dict[str, Any]) -> tuple[int, str]:
    score = 0
    if str(issue_record.get("severity") or "").strip() == "major":
        score += 20
    if bool(issue_record.get("is_persistent")):
        score += 18
    appearance_count = int(issue_record.get("appearance_count") or 1)
    score += min(appearance_count, 5) * 3
    if _coerce_str_list(issue_record.get("claim_ids")):
        score += 10
    if _coerce_str_list(issue_record.get("figure_ids")):
        score += 8
    if _coerce_str_list(issue_record.get("section_ids")):
        score += 6
    if not bool(issue_record.get("is_bound")):
        score += 12
    if score >= 42:
        return score, "p0"
    if score >= 26:
        return score, "p1"
    return score, "p2"


def _build_repair_queue(
    active_issue_records: list[dict[str, Any]],
    *,
    repair_actions: list[str],
    verification_checks: list[str],
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for issue_record in active_issue_records:
        issue_id = str(issue_record.get("issue_id") or "").strip()
        if not issue_id:
            continue
        claim_ids = _coerce_str_list(issue_record.get("claim_ids"))
        figure_ids = _coerce_str_list(issue_record.get("figure_ids"))
        section_ids = _coerce_str_list(issue_record.get("section_ids"))
        primary_target_type = "unbound"
        primary_target_id: str | None = None
        if figure_ids:
            primary_target_type = "figure"
            primary_target_id = figure_ids[0]
        elif claim_ids:
            primary_target_type = "claim"
            primary_target_id = claim_ids[0]
        elif section_ids:
            primary_target_type = "section"
            primary_target_id = section_ids[0]
        label_target_id = (
            primary_target_id
            or (section_ids[0] if section_ids else "")
        )
        primary_target_label = _format_target_label(label_target_id, catalog=catalog)
        target_ids = claim_ids + figure_ids + section_ids
        matched_actions = _match_supporting_texts(
            issue_record.get("text") or "",
            repair_actions,
            target_ids=target_ids,
        )
        if not matched_actions:
            matched_actions = _default_repair_actions(
                issue_record,
                primary_target_type=primary_target_type,
                primary_target_id=primary_target_id,
                primary_target_label=primary_target_label,
            )
        matched_checks = _match_supporting_texts(
            issue_record.get("text") or "",
            verification_checks,
            target_ids=target_ids,
        )
        if not matched_checks:
            matched_checks = _default_verification_checks(
                issue_record,
                primary_target_type=primary_target_type,
                primary_target_id=primary_target_id,
                primary_target_label=primary_target_label,
            )
        priority_score, priority_tier = _compute_repair_priority(issue_record)
        blocking_reasons: list[str] = []
        if not bool(issue_record.get("is_bound")):
            blocking_reasons.append("missing_target_binding")
        if not matched_actions:
            blocking_reasons.append("missing_repair_actions")
        if not matched_checks:
            blocking_reasons.append("missing_verification_checks")
        if "missing_target_binding" in blocking_reasons:
            status = "needs_targeting"
        elif "missing_repair_actions" in blocking_reasons:
            status = "needs_actions"
        elif "missing_verification_checks" in blocking_reasons:
            status = "needs_verification"
        else:
            status = "ready"
        queue.append(
            {
                "repair_id": f"RPR-{issue_id.replace('RVW-', '')}",
                "issue_id": issue_id,
                "issue_text": str(issue_record.get("text") or "").strip(),
                "role": str(issue_record.get("role") or "").strip() or None,
                "review_lane": str(issue_record.get("review_lane") or "review"),
                "strictness_profile": str(
                    issue_record.get("strictness_profile") or "standard"
                ),
                "severity": str(issue_record.get("severity") or "").strip() or None,
                "kind": str(issue_record.get("kind") or "").strip() or None,
                "blocker_class": str(
                    issue_record.get("blocker_class") or _infer_blocker_class(issue_record)
                ),
                "appearance_count": int(issue_record.get("appearance_count") or 1),
                "is_persistent": bool(issue_record.get("is_persistent")),
                "status": status,
                "priority_score": priority_score,
                "priority_tier": priority_tier,
                "claim_ids": claim_ids,
                "figure_ids": figure_ids,
                "section_ids": section_ids,
                "primary_target_type": primary_target_type,
                "primary_target_id": primary_target_id,
                "primary_target_label": primary_target_label,
                "repair_actions": matched_actions,
                "verification_checks": matched_checks,
                "blocking_reasons": blocking_reasons,
            }
        )
    queue.sort(
        key=lambda item: (
            {"p0": 0, "p1": 1, "p2": 2}.get(str(item.get("priority_tier") or "p2"), 3),
            -int(item.get("priority_score") or 0),
            str(item.get("issue_id") or ""),
        )
    )
    return queue


def _bind_issue_targets(
    issue_text: str,
    *,
    role: str,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    claim_ids = _match_targets(issue_text, catalog.get("claims") or [])
    figure_ids = _match_targets(issue_text, catalog.get("figures") or [])
    section_ids = _match_targets(issue_text, catalog.get("sections") or [], limit=3)
    binding_reasons: list[str] = []
    if claim_ids:
        binding_reasons.append("claim_text_match")
    if figure_ids:
        binding_reasons.append("figure_text_match")
    if section_ids:
        binding_reasons.append("section_text_match")

    if claim_ids and not figure_ids:
        for claim_id in claim_ids:
            figure_ids.extend(catalog.get("claim_to_figures", {}).get(claim_id, []))
        figure_ids = _dedupe_texts(figure_ids)
        if figure_ids:
            binding_reasons.append("figure_from_claim_binding")
    if figure_ids and not claim_ids:
        for figure_id in figure_ids:
            claim_id = str(catalog.get("figure_to_claim", {}).get(figure_id) or "").strip()
            if claim_id:
                claim_ids.append(claim_id)
        claim_ids = _dedupe_texts(claim_ids)
        if claim_ids:
            binding_reasons.append("claim_from_figure_binding")
    if claim_ids:
        for section_name, bound_claims in (catalog.get("section_claim_bindings") or {}).items():
            if any(claim_id in bound_claims for claim_id in claim_ids):
                section_ids.append(section_name)
    if figure_ids:
        for section_name, bound_figures in (catalog.get("section_figure_bindings") or {}).items():
            if any(figure_id in bound_figures for figure_id in figure_ids):
                section_ids.append(section_name)
    section_ids = _dedupe_texts(section_ids)
    if not section_ids:
        defaults = _ROLE_DEFAULT_SECTIONS.get(role, ())
        available_sections = {
            str(item.get("id") or "").strip()
            for item in (catalog.get("sections") or [])
            if str(item.get("id") or "").strip()
        }
        section_ids = [section for section in defaults if section in available_sections]
        if section_ids:
            binding_reasons.append("role_default_section")

    return {
        "claim_ids": claim_ids,
        "figure_ids": figure_ids,
        "section_ids": section_ids,
        "binding_reasons": binding_reasons,
        "is_bound": bool(claim_ids or figure_ids or section_ids),
        "is_strongly_bound": any(
            str(reason) != "role_default_section" for reason in binding_reasons
        )
        or bool(claim_ids)
        or bool(figure_ids),
    }


def _extract_issue_records(review_payload: dict[str, Any], *, role: str) -> list[dict[str, Any]]:
    issue_specs = [
        ("Weaknesses", "weakness", "major"),
        ("Questions", "question", "minor"),
        ("Limitations", "limitation", "major"),
        ("Concerns", "concern", "major"),
        ("Risks", "risk", "major"),
    ]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field_name, issue_kind, severity in issue_specs:
        for text in _coerce_text_list(review_payload.get(field_name)):
            issue_id = _issue_id(text)
            if issue_id in seen:
                continue
            seen.add(issue_id)
            records.append(
                {
                    "issue_id": issue_id,
                    "text": text,
                    "normalized_text": _normalize_issue_text(text),
                    "kind": issue_kind,
                    "severity": severity,
                    "role": role,
                    "source_field": field_name,
                }
            )
    return records


def _extract_repair_actions(review_payload: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for field_name in (
        "Limitations",
        "Recommendations",
        "Suggested Fixes",
        "Action Items",
        "Experiments",
        "Follow-up Actions",
    ):
        actions.extend(_coerce_text_list(review_payload.get(field_name)))
    return _dedupe_texts(actions)


def _extract_verification_checks(
    review_payload: dict[str, Any],
    *,
    evidence_refs: list[str] | None = None,
) -> list[str]:
    checks: list[str] = []
    for field_name in (
        "Verification",
        "Verification Checks",
        "Checks",
        "Acceptance Checks",
        "Follow-up Checks",
    ):
        checks.extend(_coerce_text_list(review_payload.get(field_name)))
    for ref in evidence_refs or []:
        ref_text = str(ref or "").strip()
        if ref_text:
            checks.append(f"Verified against {ref_text}")
    return _dedupe_texts(checks)


def _coerce_issue_records_from_state(
    state: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    value = state.get(key)
    records: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("issue") or "").strip()
                if not text:
                    continue
                record = dict(item)
                record.setdefault("issue_id", _issue_id(text))
                record.setdefault("text", text)
                record.setdefault("normalized_text", _normalize_issue_text(text))
                records.append(record)
            else:
                text = str(item).strip()
                if not text:
                    continue
                records.append(
                    {
                        "issue_id": _issue_id(text),
                        "text": text,
                        "normalized_text": _normalize_issue_text(text),
                    }
                )
    if records:
        return records
    if key == "active_issue_records":
        return [
            {
                "issue_id": _issue_id(text),
                "text": text,
                "normalized_text": _normalize_issue_text(text),
            }
            for text in _coerce_text_list(state.get("active_issues"))
        ]
    if key == "resolved_issue_records":
        return [
            {
                "issue_id": _issue_id(text),
                "text": text,
                "normalized_text": _normalize_issue_text(text),
            }
            for text in _coerce_text_list(state.get("resolved_issues"))
        ]
    if key == "persistent_issue_records":
        return [
            {
                "issue_id": _issue_id(text),
                "text": text,
                "normalized_text": _normalize_issue_text(text),
            }
            for text in _coerce_text_list(state.get("persistent_issues"))
        ]
    return []


def compute_review_repair_metrics(review_state: dict[str, Any] | None) -> dict[str, Any]:
    state = review_state if isinstance(review_state, dict) else {}
    active_records = _coerce_issue_records_from_state(state, "active_issue_records")
    resolved_records = _coerce_issue_records_from_state(state, "resolved_issue_records")
    persistent_records = _coerce_issue_records_from_state(
        state, "persistent_issue_records"
    )
    issue_ledger = _coerce_issue_records_from_state(state, "issue_ledger")
    roles: set[str] = set()
    for item in state.get("rounds") or []:
        if isinstance(item, dict) and str(item.get("role") or "").strip():
            roles.add(str(item.get("role")).strip())
    for role in (state.get("role_summaries") or {}).keys():
        role_text = str(role or "").strip()
        if role_text:
            roles.add(role_text)
    total_issue_ids = {
        str(item.get("issue_id") or "")
        for item in active_records + resolved_records + persistent_records + issue_ledger
        if str(item.get("issue_id") or "").strip()
    }
    total_issue_count = len(total_issue_ids)
    active_issue_count = len(
        {
            str(item.get("issue_id") or "")
            for item in active_records
            if str(item.get("issue_id") or "").strip()
        }
    )
    resolved_issue_count = len(
        {
            str(item.get("issue_id") or "")
            for item in resolved_records
            if str(item.get("issue_id") or "").strip()
        }
    )
    persistent_issue_count = len(
        {
            str(item.get("issue_id") or "")
            for item in persistent_records
            if str(item.get("issue_id") or "").strip()
        }
    )
    repair_action_count = len(_dedupe_texts(_coerce_text_list(state.get("repair_actions"))))
    verification_count = len(
        _dedupe_texts(_coerce_text_list(state.get("verification_checks")))
    )
    repair_queue = [
        item for item in (state.get("repair_queue") or []) if isinstance(item, dict)
    ]
    bound_issue_ids: set[str] = set()
    bound_active_issue_ids: set[str] = set()
    strong_bound_issue_ids: set[str] = set()
    strong_bound_active_issue_ids: set[str] = set()
    for item in active_records + resolved_records + persistent_records + issue_ledger:
        issue_id = str(item.get("issue_id") or "").strip()
        if not issue_id:
            continue
        claim_ids = _coerce_str_list(item.get("claim_ids"))
        figure_ids = _coerce_str_list(item.get("figure_ids"))
        section_ids = _coerce_str_list(item.get("section_ids"))
        if claim_ids or figure_ids or section_ids:
            bound_issue_ids.add(issue_id)
            binding_reasons = _coerce_str_list(item.get("binding_reasons"))
            is_strongly_bound = (
                bool(item.get("is_strongly_bound"))
                if "is_strongly_bound" in item
                else (not binding_reasons or any(reason != "role_default_section" for reason in binding_reasons))
            )
            if is_strongly_bound:
                strong_bound_issue_ids.add(issue_id)
    for item in active_records:
        issue_id = str(item.get("issue_id") or "").strip()
        if not issue_id:
            continue
        if (
            _coerce_str_list(item.get("claim_ids"))
            or _coerce_str_list(item.get("figure_ids"))
            or _coerce_str_list(item.get("section_ids"))
        ):
            bound_active_issue_ids.add(issue_id)
            binding_reasons = _coerce_str_list(item.get("binding_reasons"))
            is_strongly_bound = (
                bool(item.get("is_strongly_bound"))
                if "is_strongly_bound" in item
                else (not binding_reasons or any(reason != "role_default_section" for reason in binding_reasons))
            )
            if is_strongly_bound:
                strong_bound_active_issue_ids.add(issue_id)
    bound_issue_count = len(bound_issue_ids)
    bound_active_issue_count = len(bound_active_issue_ids)
    strong_bound_issue_count = len(strong_bound_issue_ids)
    strong_bound_active_issue_count = len(strong_bound_active_issue_ids)
    unbound_issue_count = max(total_issue_count - bound_issue_count, 0)
    role_coverage_ratio = round(len(roles) / max(len(REVIEW_ROLES), 1), 3)
    resolution_rate = (
        round(resolved_issue_count / max(total_issue_count, 1), 3)
        if total_issue_count
        else 1.0
    )
    target_binding_coverage = (
        round(strong_bound_issue_count / max(total_issue_count, 1), 3)
        if total_issue_count
        else 1.0
    )
    active_binding_coverage = (
        round(strong_bound_active_issue_count / max(active_issue_count, 1), 3)
        if active_issue_count
        else 1.0
    )
    verification_coverage = (
        round(
            min(1.0, verification_count / max(active_issue_count + resolved_issue_count, 1)),
            3,
        )
        if (active_issue_count + resolved_issue_count) > 0
        else 1.0
    )
    repair_queue_issue_ids = {
        str(item.get("issue_id") or "").strip()
        for item in repair_queue
        if str(item.get("issue_id") or "").strip()
    }
    repair_ready_issue_ids = {
        str(item.get("issue_id") or "").strip()
        for item in repair_queue
        if str(item.get("status") or "").strip() == "ready"
        and str(item.get("issue_id") or "").strip()
    }
    repair_verification_issue_ids = {
        str(item.get("issue_id") or "").strip()
        for item in repair_queue
        if _coerce_str_list(item.get("verification_checks"))
        and str(item.get("issue_id") or "").strip()
    }
    repair_targeted_issue_ids = {
        str(item.get("issue_id") or "").strip()
        for item in repair_queue
        if str(item.get("primary_target_id") or "").strip()
        or str(item.get("primary_target_type") or "").strip() == "section"
    }
    repair_queue_count = min(len(repair_queue_issue_ids), active_issue_count)
    repair_ready_count = min(len(repair_ready_issue_ids), active_issue_count)
    repair_verification_ready_count = min(
        len(repair_verification_issue_ids), active_issue_count
    )
    repair_targeted_count = min(len(repair_targeted_issue_ids), active_issue_count)
    repair_queue_coverage = (
        round(repair_queue_count / max(active_issue_count, 1), 3)
        if active_issue_count
        else 1.0
    )
    repair_ready_coverage = (
        round(repair_ready_count / max(active_issue_count, 1), 3)
        if active_issue_count
        else 1.0
    )
    repair_verification_ready_coverage = (
        round(repair_verification_ready_count / max(active_issue_count, 1), 3)
        if active_issue_count
        else 1.0
    )
    repair_targeted_coverage = (
        round(repair_targeted_count / max(active_issue_count, 1), 3)
        if active_issue_count
        else 1.0
    )
    return {
        "total_issue_count": total_issue_count,
        "active_issue_count": active_issue_count,
        "resolved_issue_count": resolved_issue_count,
        "persistent_issue_count": persistent_issue_count,
        "repair_action_count": repair_action_count,
        "verification_count": verification_count,
        "bound_issue_count": min(bound_issue_count, total_issue_count),
        "unbound_issue_count": unbound_issue_count,
        "bound_active_issue_count": min(bound_active_issue_count, active_issue_count),
        "strong_bound_issue_count": min(strong_bound_issue_count, total_issue_count),
        "strong_bound_active_issue_count": min(
            strong_bound_active_issue_count, active_issue_count
        ),
        "target_binding_coverage": target_binding_coverage,
        "active_binding_coverage": active_binding_coverage,
        "role_count": len(roles),
        "role_coverage_ratio": role_coverage_ratio,
        "resolution_rate": resolution_rate,
        "verification_coverage": verification_coverage,
        "repair_queue_count": repair_queue_count,
        "repair_ready_count": repair_ready_count,
        "repair_verification_ready_count": repair_verification_ready_count,
        "repair_targeted_count": repair_targeted_count,
        "repair_queue_coverage": repair_queue_coverage,
        "repair_ready_coverage": repair_ready_coverage,
        "repair_verification_ready_coverage": repair_verification_ready_coverage,
        "repair_targeted_coverage": repair_targeted_coverage,
    }


def review_jobs_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / "reviews" / "jobs"


def _infer_blocker_class(issue_record: dict[str, Any]) -> str:
    role = str(issue_record.get("role") or "").strip().lower()
    text = _normalize_issue_text(issue_record.get("text") or issue_record.get("issue_text") or "")
    if role in {"skeptical_pc_member", "claim_cross_examiner"}:
        if any(token in text for token in ("overclaim", "oversell", "generaliz", "claim", "support")):
            return "oversell"
    if role == "desk_reject_editor" and any(
        token in text for token in ("scope", "framing", "fit", "position", "motivation", "importance")
    ):
        return "positioning_gap"
    if role in {"rigor", "skeptical_pc_member"} or any(
        token in text
        for token in ("baseline", "ablation", "significance", "control", "benchmark", "experiment")
    ):
        return "evidence_hole"
    if role == "stats_sniper" or any(
        token in text for token in ("variance", "confidence interval", "p-value", "significance test", "error bar", "sample size", "statistic")
    ):
        return "statistical_gap"
    if role in {"reproducibility", "reproducibility_assassin"} or any(
        token in text
        for token in ("reproduc", "seed", "hyperparameter", "implementation", "setup")
    ):
        return "reproducibility_gap"
    if role in {"related_work_skeptic"} or any(
        token in text for token in ("citation", "prior work", "related work", "literature", "missing reference")
    ):
        return "citation_gap"
    if role in {"novelty", "novelty_executioner", "meta_reviewer"}:
        return "novelty_risk"
    if role == "style_snob":
        return "clarity_drag"
    if _coerce_str_list(issue_record.get("figure_ids")):
        return "figure_gap"
    return "general_blocker"


def _infer_repair_owner(queue_item: dict[str, Any], lane: str) -> tuple[str, str]:
    blocker_class = str(queue_item.get("blocker_class") or "").strip().lower()
    role = str(queue_item.get("role") or "").strip().lower()
    primary_target_type = str(queue_item.get("primary_target_type") or "").strip().lower()
    if lane in {"evidence_followup", "method_repair"} or blocker_class in {
        "evidence_hole",
        "reproducibility_gap",
        "statistical_gap",
    }:
        return ("experiment_agent", "Needs stronger evidence, controls, or reproducibility detail.")
    if lane == "figure_repair" or primary_target_type == "figure" or blocker_class == "figure_gap":
        return ("figure_agent", "Issue is primarily about figure packaging or visual evidence.")
    if blocker_class in {"oversell", "novelty_risk", "citation_gap", "positioning_gap"}:
        return ("storyline_editor", "Issue is about claim scope, novelty framing, or overstatement.")
    if role in {"clarity", "style_snob", "desk_reject_editor"} or lane == "section_rewrite":
        return ("writing_agent", "Issue is primarily narrative, structure, or exposition.")
    if lane == "triage":
        return ("planner_agent", "Issue still needs planning/ownership clarification before execution.")
    return ("repair_agent", "Generic repair ownership fallback.")


def build_review_role_instruction(role: str) -> str:
    mapping = {
        "novelty": "Focus on novelty, contribution distinctness, and whether the claim is more than an incremental extension.",
        "rigor": "Focus on baselines, evaluation design, ablations, significance, and whether the evidence supports the claim.",
        "clarity": "Focus on narrative coherence, precision, structure, and whether a strong reviewer can follow the argument end-to-end.",
        "reproducibility": "Focus on dataset clarity, implementation details, experimental settings, and whether another team could reproduce the results.",
        "skeptical_pc_member": "Act like a skeptical program-committee member trying to reject the paper for weak baselines, narrow gains, or inflated significance. Prefer concrete blockers over polite suggestions.",
        "claim_cross_examiner": "Cross-examine each major claim against the available evidence. Hunt for overstatement, unsupported generalization, and claim/figure mismatch.",
        "reproducibility_assassin": "Assume the method is not reproducible until the paper proves otherwise. Target hidden protocol choices, missing hyperparameters, missing seeds, and vague implementation details.",
        "novelty_executioner": "Try to prove the paper is not novel. Compare the claimed contribution to obvious prior-work baselines, incremental framing, and missing distinction arguments.",
        "stats_sniper": "Attack the statistics and measurement discipline. Hunt for missing uncertainty estimates, weak sample sizes, absent significance tests, or metrics that do not justify the claim.",
        "related_work_skeptic": "Assume the related-work positioning is incomplete until proven otherwise. Target missing citations, weak differentiation, and prior-work comparisons that are too soft.",
        "meta_reviewer": "Synthesize the strongest reject case across novelty, rigor, clarity, and reproducibility. Focus on the few blockers most likely to drive an area-chair level rejection.",
        "desk_reject_editor": "Judge whether the title, abstract, and framing would trigger an early reject for poor fit, diffuse motivation, or unclear contribution framing.",
        "style_snob": "Only flag style or narrative issues that materially block scientific evaluation. Be strict about ambiguity, vague contribution framing, and evidence-obscuring prose.",
    }
    return mapping.get(role, mapping["clarity"])


def begin_review_job(
    project_root: str | Path,
    *,
    role: str,
    model_review: str,
    review_plan: dict[str, Any],
    lane_name: str = "review",
    suite_name: str | None = None,
    strictness_profile: str = "standard",
) -> dict[str, Any]:
    job_id = f"{role}_{uuid.uuid4().hex[:10]}"
    job_dir = review_jobs_root(project_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "job_id": job_id,
        "role": role,
        "lane_name": str(lane_name or "review"),
        "suite_name": str(suite_name or ""),
        "strictness_profile": str(strictness_profile or "standard"),
        "model_review": model_review,
        "job_dir": str(job_dir),
        "review_plan": review_plan,
        "started_at": _now_iso(),
        "finished_at": None,
        "status": "running",
    }
    (job_dir / "job.json").write_text(
        json.dumps(job, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return job


def finalize_review_job(
    project_root: str | Path,
    *,
    job: dict[str, Any],
    review_text: Any,
    review_img: Any,
    pdf_path: str | None,
    usage_summary: dict[str, Any] | None = None,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    job_dir = Path(job["job_dir"])
    job["finished_at"] = _now_iso()
    job["status"] = "completed" if pdf_path else "missing_pdf"
    job["pdf_path"] = pdf_path
    job["usage_summary"] = usage_summary or {}
    job["evidence_refs"] = list(evidence_refs or [])
    (job_dir / "job.json").write_text(
        json.dumps(job, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if review_text is not None:
        (job_dir / "review_text.json").write_text(
            json.dumps(review_text, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if review_img is not None:
        (job_dir / "review_img.json").write_text(
            json.dumps(review_img, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    markdown_lines = [
        f"# Review Job {job['job_id']}",
        "",
        f"- Role: {job['role']}",
        f"- Lane: {job.get('lane_name') or 'review'}",
        f"- Strictness: {job.get('strictness_profile') or 'standard'}",
        f"- Status: {job['status']}",
        f"- PDF: {job.get('pdf_path')}",
        f"- Evidence refs: {', '.join(job.get('evidence_refs') or []) or 'none'}",
    ]
    (job_dir / "review_summary.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    update_review_state(
        project_root,
        job=job,
        review_text=review_text,
        usage_summary=usage_summary or {},
        evidence_refs=list(evidence_refs or []),
    )
    return job


def update_review_state(
    project_root: str | Path,
    *,
    job: dict[str, Any],
    review_text: Any,
    usage_summary: dict[str, Any],
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    review_state = load_contract_artifact(project_root, "review_state", default=None)
    if not isinstance(review_state, dict):
        review_state = {
            "schema_version": 1,
            "rounds": [],
            "active_issues": [],
            "active_issue_records": [],
            "resolved_issues": [],
            "resolved_issue_records": [],
            "persistent_issues": [],
            "persistent_issue_records": [],
            "issue_ledger": [],
            "issue_to_claim": {},
            "issue_to_figure": {},
            "issue_to_section": {},
            "repair_actions": [],
            "verification_checks": [],
            "repair_queue": [],
            "repair_queue_summary": {},
            "role_summaries": {},
            "lane_summaries": {},
            "usage_accounting": {},
            "repair_metrics": {},
        }
    round_item = {
        "job_id": job.get("job_id"),
        "role": job.get("role"),
        "lane_name": job.get("lane_name"),
        "suite_name": job.get("suite_name"),
        "strictness_profile": job.get("strictness_profile"),
        "status": job.get("status"),
        "finished_at": job.get("finished_at"),
    }
    review_state["rounds"].append(round_item)
    previous_active = {
        str(item.get("issue_id") or ""): item
        for item in _coerce_issue_records_from_state(review_state, "active_issue_records")
        if str(item.get("issue_id") or "").strip()
    }
    resolved_records = {
        str(item.get("issue_id") or ""): item
        for item in _coerce_issue_records_from_state(review_state, "resolved_issue_records")
        if str(item.get("issue_id") or "").strip()
    }
    issue_ledger = {
        str(item.get("issue_id") or ""): item
        for item in _coerce_issue_records_from_state(review_state, "issue_ledger")
        if str(item.get("issue_id") or "").strip()
    }
    binding_catalog = _build_issue_binding_catalog(project_root)
    active_issue_records: list[dict[str, Any]] = []
    persistent_issue_records: list[dict[str, Any]] = []
    if isinstance(review_text, dict):
        review_payload = (
            review_text.get("review", {})
            if isinstance(review_text.get("review"), dict)
            else {}
        )
        issue_records = _extract_issue_records(review_payload, role=str(job.get("role") or "clarity"))
        current_issue_ids = {
            str(item.get("issue_id") or "") for item in issue_records if str(item.get("issue_id") or "").strip()
        }
        previous_issue_ids = set(previous_active.keys())
        persistent_issue_ids = current_issue_ids & previous_issue_ids
        resolved_issue_ids = previous_issue_ids - current_issue_ids
        for item in issue_records:
            issue_id = str(item.get("issue_id") or "").strip()
            if not issue_id:
                continue
            previous = previous_active.get(issue_id) or {}
            appearance_count = int(previous.get("appearance_count") or 1)
            if issue_id in persistent_issue_ids:
                appearance_count += 1
            bindings = _bind_issue_targets(
                item.get("text") or "",
                role=str(job.get("role") or "clarity"),
                catalog=binding_catalog,
            )
            record = {
                **item,
                "job_id": job.get("job_id"),
                "review_lane": job.get("lane_name") or "review",
                "review_suite": job.get("suite_name"),
                "strictness_profile": job.get("strictness_profile") or "standard",
                "evidence_refs": list(evidence_refs or []),
                "first_seen_job_id": previous.get("first_seen_job_id") or job.get("job_id"),
                "first_seen_at": previous.get("first_seen_at") or job.get("finished_at"),
                "last_seen_job_id": job.get("job_id"),
                "last_seen_at": job.get("finished_at"),
                "appearance_count": appearance_count,
                "is_persistent": issue_id in persistent_issue_ids,
                "status": "active",
                **bindings,
            }
            record["blocker_class"] = _infer_blocker_class(record)
            active_issue_records.append(record)
            issue_ledger[issue_id] = {
                **previous,
                **record,
            }
            if issue_id in persistent_issue_ids:
                persistent_issue_records.append(record)
        for issue_id in resolved_issue_ids:
            previous = dict(previous_active.get(issue_id) or {})
            if not previous:
                continue
            previous["status"] = "resolved"
            previous["resolved_in_job_id"] = job.get("job_id")
            previous["resolved_at"] = job.get("finished_at")
            previous["review_lane"] = previous.get("review_lane") or job.get("lane_name") or "review"
            previous.setdefault("is_bound", bool(
                _coerce_str_list(previous.get("claim_ids"))
                or _coerce_str_list(previous.get("figure_ids"))
                or _coerce_str_list(previous.get("section_ids"))
            ))
            resolved_records[issue_id] = previous
            issue_ledger[issue_id] = previous
        review_state["active_issue_records"] = active_issue_records
        review_state["active_issues"] = [item.get("text") for item in active_issue_records]
        review_state["persistent_issue_records"] = persistent_issue_records
        review_state["persistent_issues"] = [
            item.get("text") for item in persistent_issue_records
        ]
        review_state["resolved_issue_records"] = list(resolved_records.values())
        review_state["resolved_issues"] = [
            item.get("text") for item in review_state["resolved_issue_records"]
        ]
        review_state["issue_ledger"] = list(issue_ledger.values())
        review_state["issue_to_claim"] = {
            issue_id: _coerce_str_list(item.get("claim_ids"))
            for issue_id, item in issue_ledger.items()
            if _coerce_str_list(item.get("claim_ids"))
        }
        review_state["issue_to_figure"] = {
            issue_id: _coerce_str_list(item.get("figure_ids"))
            for issue_id, item in issue_ledger.items()
            if _coerce_str_list(item.get("figure_ids"))
        }
        review_state["issue_to_section"] = {
            issue_id: _coerce_str_list(item.get("section_ids"))
            for issue_id, item in issue_ledger.items()
            if _coerce_str_list(item.get("section_ids"))
        }
        review_state["repair_actions"] = _dedupe_texts(
            _coerce_text_list(review_state.get("repair_actions"))
            + _extract_repair_actions(review_payload)
        )
        review_state["verification_checks"] = _dedupe_texts(
            _coerce_text_list(review_state.get("verification_checks"))
            + _extract_verification_checks(review_payload, evidence_refs=evidence_refs)
        )
        repair_queue = _build_repair_queue(
            active_issue_records,
            repair_actions=_coerce_text_list(review_state.get("repair_actions")),
            verification_checks=_coerce_text_list(
                review_state.get("verification_checks")
            ),
            catalog=binding_catalog,
        )
        review_state["repair_queue"] = repair_queue
        review_state["repair_queue_summary"] = {
            "active_issue_count": len(active_issue_records),
            "queue_count": len(repair_queue),
            "ready_count": sum(
                str(item.get("status") or "").strip() == "ready"
                for item in repair_queue
            ),
            "needs_targeting_count": sum(
                str(item.get("status") or "").strip() == "needs_targeting"
                for item in repair_queue
            ),
            "needs_actions_count": sum(
                str(item.get("status") or "").strip() == "needs_actions"
                for item in repair_queue
            ),
            "needs_verification_count": sum(
                str(item.get("status") or "").strip() == "needs_verification"
                for item in repair_queue
            ),
            "p0_count": sum(
                str(item.get("priority_tier") or "").strip() == "p0"
                for item in repair_queue
            ),
        }
        weaknesses = _coerce_text_list(review_payload.get("Weaknesses"))
        questions = _coerce_text_list(review_payload.get("Questions"))
        limitations = _coerce_text_list(review_payload.get("Limitations"))
        lane_name = str(job.get("lane_name") or "review")
        blocker_class_counts: dict[str, int] = {}
        for item in active_issue_records:
            blocker_class = str(item.get("blocker_class") or "general_blocker")
            blocker_class_counts[blocker_class] = blocker_class_counts.get(
                blocker_class, 0
            ) + 1
        review_state.setdefault("role_summaries", {})[job["role"]] = {
            "scores": review_payload.get("scores") or {},
            "weaknesses": weaknesses,
            "questions": questions,
            "limitations": limitations,
        }
        review_state.setdefault("lane_summaries", {})[lane_name] = {
            "lane_name": lane_name,
            "suite_name": job.get("suite_name"),
            "strictness_profile": job.get("strictness_profile") or "standard",
            "roles": sorted(
                {
                    str(item.get("role") or "").strip()
                    for item in active_issue_records + persistent_issue_records
                    if str(item.get("role") or "").strip()
                }
                | {str(job.get("role") or "").strip()}
            ),
            "active_issue_count": len(active_issue_records),
            "persistent_issue_count": len(persistent_issue_records),
            "blocking_issue_count": sum(
                str(item.get("severity") or "").strip() in {"major", "critical"}
                for item in active_issue_records
            ),
            "blocker_class_counts": blocker_class_counts,
            "updated_at": job.get("finished_at"),
        }
        if lane_name == "hostile_critic":
            queue_by_issue = {
                str(item.get("issue_id") or "").strip(): item
                for item in repair_queue
                if str(item.get("issue_id") or "").strip()
            }
            critic_findings = []
            for issue_record in active_issue_records:
                issue_id = str(issue_record.get("issue_id") or "").strip()
                queue_item = queue_by_issue.get(issue_id) or {}
                target_type = str(queue_item.get("primary_target_type") or "unbound")
                target_id = str(queue_item.get("primary_target_id") or "").strip() or None
                critic_findings.append(
                    {
                        "issue_id": issue_id,
                        "severity": issue_record.get("severity"),
                        "role": issue_record.get("role"),
                        "target_type": target_type,
                        "target_id": target_id,
                        "target_label": queue_item.get("primary_target_label"),
                        "blocker_class": issue_record.get("blocker_class"),
                        "attack_angle": build_review_role_instruction(
                            str(issue_record.get("role") or "clarity")
                        ),
                        "failure_mode": issue_record.get("kind"),
                        "why_blocking": issue_record.get("text"),
                        "evidence_refs": list(issue_record.get("evidence_refs") or []),
                        "suggested_verification": queue_item.get("verification_checks")
                        or [],
                    }
                )
            save_contract_artifact(
                project_root,
                "critic_findings",
                {
                    "schema_version": 1,
                    "generated_at": _now_iso(),
                    "job_id": job.get("job_id"),
                    "lane_name": lane_name,
                    "strictness_profile": job.get("strictness_profile") or "standard",
                    "active_issue_count": len(active_issue_records),
                    "blocking_issue_count": sum(
                        str(item.get("severity") or "").strip() in {"major", "critical"}
                        for item in active_issue_records
                    ),
                    "findings": critic_findings,
                },
                producer="review_jobs",
                depends_on=["review_state"],
            )
    review_state.setdefault("usage_accounting", {})[job["job_id"]] = usage_summary
    review_state["repair_metrics"] = compute_review_repair_metrics(review_state)
    save_contract_artifact(
        project_root,
        "review_state",
        review_state,
        producer="review_jobs",
        depends_on=["manuscript_state"],
    )
    from ai_scientist.utils.review_repair_planner import save_repair_plan

    save_repair_plan(
        project_root,
        review_state=review_state,
        producer="review_jobs",
    )
    from ai_scientist.utils.self_evolution import save_self_evolution

    save_self_evolution(
        project_root,
        review_state=review_state,
        producer="review_jobs",
    )
    from ai_scientist.utils.process_alignment import save_process_alignment

    save_process_alignment(
        project_root,
        producer="review_jobs",
    )
    from ai_scientist.utils.stage_standards import save_stage_standards

    save_stage_standards(project_root)
    return review_state
