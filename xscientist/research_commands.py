"""High-level, one-command research recording workflows for the public CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai_scientist.protocol import content_hash
from ai_scientist.utils.research_integrity import (
    ResearchIntegrityError,
    build_preregistration,
    lock_preregistration,
)

from .research_git import (
    CheckpointResult,
    ResearchGitError,
    ResearchObjectResult,
    create_checkpoint,
)
from .research_lifecycle import ResearchLifecycle
from .research_vcs import ResearchRepository


def _required_text(value: str, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ResearchGitError(f"{label} cannot be empty")
    return normalized


def _ensure_direct_save_is_safe(
    repository: ResearchRepository, *, commit: bool
) -> None:
    if not commit:
        return
    staged = repository.status().get("research_stage", {}).get("paths") or []
    if staged:
        raise ResearchGitError(
            "one-command research recording requires an empty native stage; "
            "commit it with `xscientist git commit` or clear it with "
            "`xscientist research unstage --all`"
        )


def _finish(
    repository: ResearchRepository,
    result: ResearchObjectResult,
    *,
    stage: str,
    subject: str,
    status: str,
    commit: bool,
    related: Sequence[ResearchObjectResult] = (),
) -> dict[str, Any]:
    checkpoint: CheckpointResult | None = None
    if commit:
        includes = [
            item.path.relative_to(repository.path).as_posix()
            for item in (result, *related)
        ]
        checkpoint = create_checkpoint(
            repository.path,
            stage=stage,
            subject=subject,
            status=status,
            include=includes,
        )
    return {"object": result, "related": list(related), "checkpoint": checkpoint}


def save_hypothesis(
    repo: str,
    *,
    statement: str,
    falsifier: str,
    rationale: str = "",
    predictions: Sequence[str] = (),
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    payload: dict[str, Any] = {
        "statement": _required_text(statement, label="hypothesis statement"),
        "falsifier": _required_text(falsifier, label="hypothesis falsifier"),
    }
    if rationale.strip():
        payload["rationale"] = rationale.strip()
    if predictions:
        payload["predictions"] = [item.strip() for item in predictions if item.strip()]
    result = repository.record("hypothesis", payload, state="draft")
    return _finish(
        repository,
        result,
        stage="ideation",
        subject=message or "record falsifiable hypothesis",
        status="draft",
        commit=commit,
    )


def save_experiment(
    repo: str,
    *,
    summary: str,
    status: str,
    study_phase: str = "exploratory",
    plan_id: str | None = None,
    preregistration_id: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (),
    failure_class: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    payload: dict[str, Any] = {
        "summary": _required_text(summary, label="experiment summary"),
        "status": status,
        "study_phase": study_phase,
    }
    if metrics:
        payload["metrics"] = dict(metrics)
    if seeds:
        payload["seeds"] = list(dict.fromkeys(seeds))
    if failure_class.strip():
        payload["failure_class"] = failure_class.strip()
    lifecycle = ResearchLifecycle(repository)
    recorded = lifecycle.experiment_attempt(
        payload,
        preregistration_id=preregistration_id,
        plan_id=plan_id,
        commit=False,
    )
    result = recorded["attempt"]
    stage = "experiment" if result.state == "completed" else "failed"
    return _finish(
        repository,
        result,
        stage=stage,
        subject=message or f"record {result.state} experiment attempt",
        status=result.state,
        commit=commit,
    )


def save_preregistration(
    repo: str,
    *,
    hypothesis_id: str,
    dataset: str,
    metric: str,
    baseline: str,
    split_hash: str,
    registered_by: str,
    minimum_effect: float | None = None,
    alpha: float = 0.05,
    minimum_seeds: int = 3,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    hypothesis = repository.get(hypothesis_id)
    if hypothesis["kind"] != "hypothesis":
        raise ResearchGitError("preregistration hypothesis reference has wrong kind")
    hypothesis_payload = hypothesis["payload"]
    statement = _required_text(
        str(hypothesis_payload.get("statement") or ""),
        label="hypothesis statement",
    )
    falsifier = _required_text(
        str(hypothesis_payload.get("falsifier") or ""),
        label="hypothesis falsifier",
    )
    task_id = "primary"
    plan_payload = {
        "plan_id": f"plan-{hypothesis_id.removeprefix('rso-')}",
        "hypothesis_id": hypothesis_id,
        "tasks": [
            {
                "task_id": task_id,
                "dataset": _required_text(dataset, label="dataset"),
                "metric": _required_text(metric, label="metric"),
                "baseline": _required_text(baseline, label="baseline"),
            }
        ],
    }
    idea_card = {
        "idea_id": hypothesis_id,
        "title": statement,
        "core_hypothesis": statement,
        "failure_criteria": [falsifier],
    }
    try:
        draft = build_preregistration(
            idea_card,
            plan_payload,
            alpha=alpha,
            minimum_independent_seeds=minimum_seeds,
        )
        if minimum_effect is not None:
            draft["outcomes"][0]["minimum_effect"] = float(minimum_effect)
        locked = lock_preregistration(
            draft,
            split_hashes={task_id: split_hash},
            registered_by=registered_by,
        )
    except ResearchIntegrityError as exc:
        raise ResearchGitError(str(exc)) from exc
    plan = repository.record(
        "research_plan",
        plan_payload,
        state="draft",
        relations=[{"type": "depends_on", "target": hypothesis_id}],
    )
    registration = repository.record(
        "preregistration",
        locked,
        state="locked",
        relations=[{"type": "depends_on", "target": plan.object_id}],
        actor={"actor_id": registered_by, "authority": "human"},
    )
    return _finish(
        repository,
        registration,
        stage="preregister",
        subject=message or "lock confirmatory research plan",
        status="completed",
        commit=commit,
        related=[plan],
    )


def save_evidence(
    repo: str,
    *,
    result_summary: str,
    attempt_ids: Sequence[str],
    supports: Sequence[str] = (),
    refutes: Sequence[str] = (),
    metrics: Mapping[str, Any] | None = None,
    verified: bool = False,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    payload: dict[str, Any] = {
        "result": _required_text(result_summary, label="evidence result")
    }
    if metrics:
        payload["metrics"] = dict(metrics)
    lifecycle = ResearchLifecycle(repository)
    recorded = lifecycle.evidence(
        payload,
        attempt_ids=attempt_ids,
        supports=supports,
        refutes=refutes,
        verified=verified,
        commit=False,
    )
    result = recorded["evidence"]
    return _finish(
        repository,
        result,
        stage="evidence",
        subject=message or "bind experiment evidence",
        status=result.state,
        commit=commit,
    )


def save_claim(
    repo: str,
    *,
    statement: str,
    evidence_ids: Sequence[str],
    scope: str = "",
    gate_id: str | None = None,
    verified: bool = False,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    payload: dict[str, Any] = {
        "statement": _required_text(statement, label="claim statement")
    }
    if scope.strip():
        payload["scope"] = scope.strip()
    lifecycle = ResearchLifecycle(repository)
    recorded = lifecycle.claim(
        payload,
        evidence_ids=evidence_ids,
        gate_id=gate_id,
        verified=verified,
        commit=False,
    )
    result = recorded["claim"]
    return _finish(
        repository,
        result,
        stage="claim",
        subject=message or "record evidence-bound claim",
        status=result.state,
        commit=commit,
    )


def save_review(
    repo: str,
    *,
    summary: str,
    evaluates: Sequence[str],
    verifier_id: str,
    decision: str,
    required_failures: Sequence[str] = (),
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    passed = decision == "pass"
    failures = [item.strip() for item in required_failures if item.strip()]
    if passed and failures:
        raise ResearchGitError("a passing review cannot declare required failures")
    report: dict[str, Any] = {
        "summary": _required_text(summary, label="review summary"),
        "status": "verified" if passed else "rejected",
        "claim_promotion_allowed": passed,
        "required_failures": failures,
    }
    report["report_hash"] = content_hash(report)
    lifecycle = ResearchLifecycle(repository)
    recorded = lifecycle.evaluation(
        report,
        evaluates=evaluates,
        verifier_id=verifier_id,
        commit=False,
    )
    gate = recorded["gate"]
    review = recorded["review"]
    return _finish(
        repository,
        gate,
        stage="review",
        subject=message or "record independent evidence gate",
        status=gate.state,
        commit=commit,
        related=[review],
    )


__all__ = [
    "save_claim",
    "save_evidence",
    "save_experiment",
    "save_hypothesis",
    "save_preregistration",
    "save_review",
]
