from __future__ import annotations

"""Lightweight per-task verifiers for repair tasks.

Each repair task carries a `verifier` label (one of five vocab strings — see
[[repair-reflection]]). Until now those strings were only metadata. This module
turns them into actual checks against pipeline artifacts, so the post-rewrite
pipeline can tell *per issue* whether the targeted bindings now resolve.

Design constraints:
- No LLM calls. These are cheap structural checks meant to run every round.
- Pure-function: take a task + project_root, return `(passed, evidence)`.
- Env-gated dispatcher (`AI_SCIENTIST_REPAIR_VERIFY`) — when off, default
  behavior is byte-identical to before this module existed (no callers call us).
- Each verifier always returns a structured `evidence` dict so we can log it
  alongside repair_attempts without exposing internal artifact shapes.
"""

import os
import re
from pathlib import Path
from typing import Any, Callable

from ai_scientist.utils.pipeline_contracts import load_contract_artifact
from ai_scientist.utils.review_subject import check_review_subject

VERIFY_ENV_FLAG = "AI_SCIENTIST_REPAIR_VERIFY"

VERIFIER_VOCAB: tuple[str, ...] = (
    "reviewer_board_recheck",
    "hostile_critic_recheck",
    "experiment_validation",
    "figure_alignment_check",
    "planner_triage_recheck",
)


def verify_enabled() -> bool:
    return str(os.environ.get(VERIFY_ENV_FLAG) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _coerce_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return [str(value).strip()] if str(value).strip() else []


def _review_scope(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("role") or ""),
        str(item.get("review_lane") or item.get("lane_name") or "review"),
        str(item.get("review_suite") or item.get("suite_name") or ""),
    )


def _review_recheck(
    task: dict[str, Any], project_root: str | Path, *, hostile: bool = False
) -> tuple[bool, dict[str, Any]]:
    issue_id = str(task.get("issue_id") or "").strip()
    evidence: dict[str, Any] = {
        "issue_id": issue_id,
        "still_present_in_latest_round": False,
        "hostile_role_resurfaced": False,
        "matched_role": None,
    }

    def fail(reason: str) -> tuple[bool, dict[str, Any]]:
        return False, {**evidence, "reason": reason}

    if not issue_id:
        return fail("missing_issue_id")
    state = load_contract_artifact(project_root, "review_state", default=None)
    if not isinstance(state, dict):
        return fail("missing_review_state")
    active = state.get("active_issue_records")
    resolved = state.get("resolved_issue_records")
    if not isinstance(active, list) or not isinstance(resolved, list):
        # Legacy absence never proved that the responsible reviewer checked a repair.
        legacy = state.get("issues_log") or state.get("issues") or []
        active = legacy if isinstance(legacy, list) else []
        resolved = []
    for row in active:
        if not isinstance(row, dict) or str(row.get("issue_id") or "") != issue_id:
            continue
        evidence["still_present_in_latest_round"] = True
        scopes = row.get("review_scopes") or [row]
        for observation in scopes:
            if not isinstance(observation, dict):
                continue
            role, lane, _ = _review_scope(observation)
            if "hostile" in role.lower() or lane == "hostile_critic":
                evidence["hostile_role_resurfaced"] = True
                evidence["matched_role"] = role or lane
        return fail("issue_still_active")
    resolution = next(
        (
            row
            for row in resolved
            if isinstance(row, dict)
            and str(row.get("issue_id") or "") == issue_id
            and row.get("status") == "resolved"
        ),
        None,
    )
    if resolution is None:
        return fail("issue_not_explicitly_resolved")
    rounds = state.get("rounds")
    jobs = (
        {
            row.get("job_id"): row
            for row in rounds
            if isinstance(row, dict) and row.get("job_id")
        }
        if isinstance(rounds, list)
        else {}
    )
    checks = resolution.get("resolution_checks")
    if not isinstance(checks, list) or not checks:
        return fail("missing_resolution_review")
    hostile_checked = False
    checked_jobs: list[str] = []
    subject_hash = resolution.get("resolved_subject_hash")
    if not subject_hash or subject_hash != (
        state.get("current_review_subject") or {}
    ).get("subject_hash"):
        return fail("missing_or_stale_resolution_subject")
    for check in checks:
        if not isinstance(check, dict):
            return fail("invalid_resolution_review")
        job = jobs.get(check.get("job_id"))
        if (
            not job
            or job.get("status") != "completed"
            or job.get("review_payload_valid") is not True
            or _review_scope(job) != _review_scope(check)
        ):
            return fail("unconfirmed_resolution_review")
        subject = job.get("review_subject") or {}
        if (
            check.get("review_subject_hash") != subject_hash
            or subject.get("subject_hash") != subject_hash
        ):
            return fail("mismatched_resolution_subject")
        role, lane, _ = _review_scope(job)
        hostile_checked |= "hostile" in role.lower() or lane == "hostile_critic"
        checked_jobs.append(str(job["job_id"]))
    if resolution.get("resolved_in_job_id") not in checked_jobs:
        return fail("unconfirmed_resolution_review")
    if hostile and not hostile_checked:
        return fail("missing_hostile_recheck")
    subject_ok, subject_reason = check_review_subject(
        project_root,
        jobs[resolution["resolved_in_job_id"]].get("review_subject"),
        pdf_path=task.get("pdf_path"),
    )
    if not subject_ok:
        return fail(subject_reason)
    return True, {
        **evidence,
        "reason": "confirmed_review_resolution",
        "resolution_job_ids": checked_jobs,
        "review_subject_hash": subject_hash,
    }


def verify_reviewer_board_recheck(
    task: dict[str, Any], project_root: str | Path
) -> tuple[bool, dict[str, Any]]:
    """Require explicit closure by completed, valid reviews of the reporting scopes."""
    return _review_recheck(task, project_root)


def verify_hostile_critic_recheck(
    task: dict[str, Any], project_root: str | Path
) -> tuple[bool, dict[str, Any]]:
    """Require explicit closure, including a completed hostile-critic recheck."""
    return _review_recheck(task, project_root, hostile=True)


def verify_figure_alignment_check(
    task: dict[str, Any], project_root: str | Path
) -> tuple[bool, dict[str, Any]]:
    """Pass when every bound figure exists in the current figure_spec."""
    figure_ids = _coerce_list(task.get("figure_ids"))
    primary_type = str(task.get("primary_target_type") or "").strip()
    primary_id = str(task.get("primary_target_id") or "").strip()
    if primary_type == "figure" and primary_id and primary_id not in figure_ids:
        figure_ids.append(primary_id)
    if not figure_ids:
        return True, {"figure_ids": [], "reason": "no_figure_bindings"}
    figure_spec = load_contract_artifact(project_root, "figure_spec", default={}) or {}
    known = {
        str(item.get("figure_id") or "").strip()
        for item in (figure_spec.get("figures") or [])
        if isinstance(item, dict) and str(item.get("figure_id") or "").strip()
    }
    missing = [fid for fid in figure_ids if fid not in known]
    return (not missing), {
        "figure_ids": figure_ids,
        "missing_in_spec": missing,
    }


def verify_experiment_validation(
    task: dict[str, Any], project_root: str | Path
) -> tuple[bool, dict[str, Any]]:
    """Pass when bound claims show evidence/method support in the claim graph."""
    claim_ids = _coerce_list(task.get("claim_ids"))
    primary_type = str(task.get("primary_target_type") or "").strip()
    primary_id = str(task.get("primary_target_id") or "").strip()
    if primary_type == "claim" and primary_id and primary_id not in claim_ids:
        claim_ids.append(primary_id)
    if not claim_ids:
        return True, {"claim_ids": [], "reason": "no_claim_bindings"}
    graph = (
        load_contract_artifact(project_root, "claim_evidence_graph", default={}) or {}
    )
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]
    claim_status = {
        str(n.get("id") or "").strip(): str(n.get("status") or "").strip().lower()
        for n in nodes
        if n.get("type") == "claim"
    }
    supported_status = {"supported", "evidenced", "validated", "passed"}
    supports_edge = {"supports", "evidence_for", "validates"}
    edge_targets = {
        str(e.get("target") or "").strip()
        for e in edges
        if str(e.get("relation") or e.get("label") or "").strip().lower()
        in supports_edge
    }
    unsupported: list[str] = []
    for cid in claim_ids:
        status_ok = claim_status.get(cid) in supported_status
        edge_ok = cid in edge_targets
        if not (status_ok or edge_ok):
            unsupported.append(cid)
    return (not unsupported), {
        "claim_ids": claim_ids,
        "unsupported": unsupported,
    }


def verify_planner_triage_recheck(
    task: dict[str, Any], project_root: str | Path
) -> tuple[bool, dict[str, Any]]:
    """Pass when the task has the minimum bindings a triaged task should carry."""
    issue_id = str(task.get("issue_id") or "").strip()
    target_id = str(task.get("primary_target_id") or "").strip()
    target_type = str(task.get("primary_target_type") or "").strip()
    has_owner = bool(str(task.get("owner") or "").strip())
    close_cond = str(task.get("close_condition") or "").strip()
    bindings_ok = bool(target_id and target_type)
    return (bindings_ok and has_owner and bool(close_cond) and bool(issue_id)), {
        "issue_id": issue_id,
        "has_target": bindings_ok,
        "has_owner": has_owner,
        "has_close_condition": bool(close_cond),
    }


_DISPATCH: dict[str, Callable[[dict[str, Any], Any], tuple[bool, dict[str, Any]]]] = {
    "reviewer_board_recheck": verify_reviewer_board_recheck,
    "hostile_critic_recheck": verify_hostile_critic_recheck,
    "figure_alignment_check": verify_figure_alignment_check,
    "experiment_validation": verify_experiment_validation,
    "planner_triage_recheck": verify_planner_triage_recheck,
}


def run_task_verifier(task: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Dispatch one task to its declared verifier; safe on unknown labels."""
    verifier = str(task.get("verifier") or "").strip()
    if verifier not in _DISPATCH:
        return {
            "verifier": verifier or None,
            "passed": False,
            "skipped": True,
            "reason": "unknown_verifier",
        }
    try:
        passed, evidence = _DISPATCH[verifier](task, project_root)
    except Exception as exc:
        return {
            "verifier": verifier,
            "passed": False,
            "skipped": True,
            "reason": f"verifier_exception:{type(exc).__name__}",
        }
    return {
        "verifier": verifier,
        "passed": bool(passed),
        "skipped": False,
        "evidence": evidence,
    }


def maybe_run_repair_plan_verifiers(
    project_root: str | Path,
) -> dict[str, Any] | None:
    """Env-gated batch run over all repair_plan tasks. Returns aggregated results or None."""
    if not verify_enabled():
        return None
    plan = load_contract_artifact(project_root, "repair_plan", default={}) or {}
    tasks = plan.get("tasks") if isinstance(plan, dict) else None
    if not isinstance(tasks, list) or not tasks:
        return None
    results: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        outcome = run_task_verifier(task, project_root)
        outcome["task_id"] = task.get("task_id")
        outcome["issue_id"] = task.get("issue_id")
        results.append(outcome)
    pass_count = sum(1 for r in results if r.get("passed") and not r.get("skipped"))
    skip_count = sum(1 for r in results if r.get("skipped"))
    return {
        "results": results,
        "task_count": len(results),
        "passed": pass_count,
        "failed": len(results) - pass_count - skip_count,
        "skipped": skip_count,
    }
