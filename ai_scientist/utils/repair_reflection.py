from __future__ import annotations

"""LLM-based reflective planner for review-repair tasks (GEPA-style).

Given one reviewer issue plus its bindings, recent reviewer trace, and prior
attempt history, asks an LLM to produce a concrete repair plan as JSON. Any
missing / malformed field falls back to the existing template builders so a
broken reflection call never breaks the pipeline.
"""

import json
import os
from pathlib import Path
from typing import Any

from ai_scientist.llm import (
    create_client,
    extract_json_between_markers,
    get_response_from_llm,
)
from ai_scientist.utils.repair_attempts import load_attempts_for_issue
from ai_scientist.utils.review_repair_planner import (
    _build_close_condition,
    _build_execution_steps,
    _build_success_criteria,
    _build_verifier,
    _coerce_list,
)


REFLECTION_ENV_FLAG = "AI_SCIENTIST_REPAIR_REFLECTION"
REFLECTION_MODEL_ENV = "AI_SCIENTIST_REPAIR_REFLECTION_MODEL"
DEFAULT_REFLECTION_MODEL = "claude-3-5-sonnet-20241022"

_VERIFIER_VOCAB = {
    "reviewer_board_recheck",
    "hostile_critic_recheck",
    "experiment_validation",
    "figure_alignment_check",
    "planner_triage_recheck",
}

_CONFIDENCE_VOCAB = {"high", "medium", "low"}

_SYSTEM_MESSAGE = (
    "You are a research-paper review-repair planner. For ONE reviewer issue, produce a concrete, "
    "executable repair plan. Stay faithful to the bindings; do not invent new claims, figures, or "
    "sections. Reason about WHY prior attempts succeeded or failed and propose targeted next steps."
)


def reflection_enabled() -> bool:
    return str(os.environ.get(REFLECTION_ENV_FLAG) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_model() -> str:
    raw = str(os.environ.get(REFLECTION_MODEL_ENV) or "").strip()
    return raw or DEFAULT_REFLECTION_MODEL


def _trim(text: Any, limit: int = 500) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _summarize_role_trace(review_state: dict[str, Any], role: str | None) -> dict[str, list[str]]:
    if not isinstance(review_state, dict) or not role:
        return {}
    role_summaries = review_state.get("role_summaries") or {}
    if not isinstance(role_summaries, dict):
        return {}
    role_block = role_summaries.get(str(role))
    if not isinstance(role_block, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key in ("weaknesses", "questions", "limitations", "strengths"):
        items = role_block.get(key)
        if isinstance(items, list):
            cleaned = [_trim(item, 240) for item in items if str(item or "").strip()]
            if cleaned:
                out[key] = cleaned[:5]
    return out


def build_reflective_dataset(
    review_state: dict[str, Any],
    task: dict[str, Any],
    attempts_history: list[dict[str, Any]],
) -> dict[str, Any]:
    role = str(task.get("role") or "").strip() or None
    return {
        "issue": {
            "issue_id": str(task.get("issue_id") or "").strip(),
            "text": _trim(task.get("issue_text"), 800),
            "role": role,
            "severity": task.get("severity"),
            "blocker_class": task.get("blocker_class"),
            "priority_tier": task.get("priority_tier"),
            "review_lane": task.get("review_lane"),
            "lane": task.get("lane"),
            "lane_label": task.get("lane_label"),
        },
        "bindings": {
            "claim_ids": _coerce_list(task.get("claim_ids")),
            "figure_ids": _coerce_list(task.get("figure_ids")),
            "section_ids": _coerce_list(task.get("section_ids")),
            "primary_target_type": task.get("primary_target_type"),
            "primary_target_id": task.get("primary_target_id"),
            "primary_target_label": task.get("primary_target_label"),
        },
        "reviewer_trace": _summarize_role_trace(review_state, role),
        "prior_attempts": [
            {
                "round_index": row.get("round_index"),
                "status": row.get("status"),
                "addressed": row.get("addressed"),
                "coverage_ratio": row.get("coverage_ratio"),
                "scores_before": row.get("scores_before"),
                "scores_delta": row.get("scores_delta"),
                "notes": _trim(row.get("notes"), 240),
            }
            for row in attempts_history[-3:]
        ],
        "template_baseline": {
            "execution_steps": _build_execution_steps(task, str(task.get("lane") or "triage")),
            "success_criteria": _build_success_criteria(task, str(task.get("lane") or "triage")),
            "verifier": _build_verifier(task, str(task.get("lane") or "triage")),
            "close_condition": _build_close_condition(task, str(task.get("lane") or "triage")),
        },
    }


def _build_prompt(dataset: dict[str, Any]) -> str:
    return (
        "Below is ONE reviewer issue with its bindings, the reviewer's recent trace, "
        "and up to three of our prior repair attempts on this same issue. A template-based baseline "
        "is also included so you can improve on it.\n\n"
        "Your job is to output a single JSON object with the fields shown. "
        "Be specific, faithful to the bindings, and target the WHY behind any prior failure.\n\n"
        "## Issue & bindings\n"
        f"{json.dumps({'issue': dataset['issue'], 'bindings': dataset['bindings']}, ensure_ascii=False, indent=2)}\n\n"
        "## Reviewer trace (most recent)\n"
        f"{json.dumps(dataset['reviewer_trace'], ensure_ascii=False, indent=2)}\n\n"
        "## Prior attempts on this issue\n"
        f"{json.dumps(dataset['prior_attempts'], ensure_ascii=False, indent=2)}\n\n"
        "## Template baseline (improve on this)\n"
        f"{json.dumps(dataset['template_baseline'], ensure_ascii=False, indent=2)}\n\n"
        "Return JSON in the form:\n"
        "```json\n"
        "{\n"
        "  \"execution_steps\": [\"...\"],\n"
        "  \"success_criteria\": [\"...\"],\n"
        "  \"verifier\": \"reviewer_board_recheck | hostile_critic_recheck | experiment_validation | figure_alignment_check | planner_triage_recheck\",\n"
        "  \"close_condition\": \"...\",\n"
        "  \"depends_on\": [\"target_binding|repair_actions|verification_checks\"],\n"
        "  \"confidence\": \"high|medium|low\",\n"
        "  \"rationale\": \"...\"\n"
        "}\n"
        "```\n"
    )


def _validate_reflection(payload: Any) -> dict[str, Any]:
    """Return the per-field accepted dict; non-conforming fields are dropped."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    steps = payload.get("execution_steps")
    if isinstance(steps, list):
        cleaned = [str(s).strip() for s in steps if str(s or "").strip()]
        if cleaned:
            out["execution_steps"] = cleaned
    criteria = payload.get("success_criteria")
    if isinstance(criteria, list):
        cleaned = [str(s).strip() for s in criteria if str(s or "").strip()]
        if cleaned:
            out["success_criteria"] = cleaned
    verifier = str(payload.get("verifier") or "").strip()
    if verifier in _VERIFIER_VOCAB:
        out["verifier"] = verifier
    close_cond = str(payload.get("close_condition") or "").strip()
    if close_cond:
        out["close_condition"] = close_cond
    depends_on = payload.get("depends_on")
    if isinstance(depends_on, list):
        cleaned = [str(s).strip() for s in depends_on if str(s or "").strip()]
        if cleaned:
            out["depends_on"] = cleaned
    confidence = str(payload.get("confidence") or "").strip().lower()
    if confidence in _CONFIDENCE_VOCAB:
        out["confidence"] = confidence
    rationale = str(payload.get("rationale") or "").strip()
    if rationale:
        out["rationale"] = rationale
    return out


def reflect_issue_repair_plan(
    project_root: str | Path,
    task: dict[str, Any],
    *,
    review_state: dict[str, Any],
    model: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Run a single LLM reflection call for one repair task.

    Returns a dict with keys:
      - accepted_fields: dict of LLM-provided fields that passed validation
      - dataset: the reflective dataset (for debugging / logging)
      - status: "ok" | "errored" | "empty"
      - warnings: list[str]
    """
    issue_id = str(task.get("issue_id") or "").strip()
    warnings: list[str] = []
    dataset_attempts: list[dict[str, Any]] = []
    if issue_id:
        try:
            dataset_attempts = load_attempts_for_issue(project_root, issue_id, limit=3)
        except Exception as exc:
            warnings.append(f"attempts_load_failed:{exc}")
    dataset = build_reflective_dataset(review_state or {}, task, dataset_attempts)
    use_model = model or _resolve_model()
    use_client = client
    if use_client is None:
        try:
            use_client, use_model = create_client(use_model)
        except Exception as exc:
            warnings.append(f"client_init_failed:{exc}")
            return {
                "accepted_fields": {},
                "dataset": dataset,
                "status": "errored",
                "warnings": warnings,
            }

    prompt = _build_prompt(dataset)
    try:
        content, _ = get_response_from_llm(
            prompt,
            use_client,
            use_model,
            system_message=_SYSTEM_MESSAGE,
            temperature=0.4,
        )
    except Exception as exc:
        warnings.append(f"llm_call_failed:{exc}")
        return {
            "accepted_fields": {},
            "dataset": dataset,
            "status": "errored",
            "warnings": warnings,
        }

    parsed = extract_json_between_markers(content or "")
    accepted = _validate_reflection(parsed)
    if not accepted:
        warnings.append("reflection_no_valid_fields")
        return {
            "accepted_fields": {},
            "dataset": dataset,
            "status": "empty",
            "warnings": warnings,
        }
    return {
        "accepted_fields": accepted,
        "dataset": dataset,
        "status": "ok",
        "warnings": warnings,
    }


def apply_reflection_to_task(
    task: dict[str, Any],
    reflection: dict[str, Any],
) -> dict[str, Any]:
    """Mutate `task` in place — replacing only the fields the LLM filled successfully.

    Always annotates `task` with `reflection_status`, `reflection_warnings`, and
    `reflection_metadata` so callers can audit what was/was not LLM-driven.
    """
    status = str(reflection.get("status") or "empty")
    warnings = list(reflection.get("warnings") or [])
    accepted = reflection.get("accepted_fields") or {}
    if not isinstance(accepted, dict):
        accepted = {}

    metadata: dict[str, Any] = {
        "fields_from_llm": sorted(accepted.keys()),
        "fields_from_template": [],
    }
    template_fields = ("execution_steps", "success_criteria", "verifier", "close_condition")
    for field in template_fields:
        if field in accepted:
            task[field] = accepted[field]
        else:
            metadata["fields_from_template"].append(field)

    if "depends_on" in accepted:
        existing = list(task.get("depends_on") or [])
        merged: list[str] = []
        seen: set[str] = set()
        for value in existing + list(accepted["depends_on"]):
            v = str(value or "").strip()
            if not v or v in seen:
                continue
            seen.add(v)
            merged.append(v)
        task["depends_on"] = merged

    if "confidence" in accepted:
        metadata["confidence"] = accepted["confidence"]
    if "rationale" in accepted:
        metadata["rationale"] = accepted["rationale"]

    task["reflection_status"] = status
    if warnings:
        task["reflection_warnings"] = warnings
    task["reflection_metadata"] = metadata
    return task
