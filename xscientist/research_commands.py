"""High-level, one-command research recording workflows for the public CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ai_scientist.protocol import content_hash
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.research_integrity import (
    ResearchIntegrityError,
    build_preregistration,
    lock_preregistration,
)

from .research_git import (
    CheckpointResult,
    ResearchGitError,
    ResearchObjectResult,
    capture_environment_receipt,
    create_checkpoint,
)
from .research_lifecycle import ResearchLifecycle
from .research_semantics import (
    build_search_receipt_payload,
    build_text_quote_selector,
    claim_scope_hash,
    normalize_claim_scope,
)
from .research_vcs import ResearchRepository


def _required_text(value: str, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ResearchGitError(f"{label} cannot be empty")
    return normalized


def _hashes(values: Sequence[str], *, label: str) -> list[str]:
    import re

    rows = sorted({str(value).strip() for value in values if str(value).strip()})
    invalid = [
        value for value in rows if not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
    ]
    if invalid:
        raise ResearchGitError(f"{label} must use sha256:<64 lowercase hex>")
    return rows


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
    reproduce_command: str | None = None,
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
            reproduce_command=reproduce_command,
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


def save_research_plan(
    repo: str,
    *,
    hypothesis_id: str,
    summary: str,
    discriminating_tests: Sequence[str] = (),
    success_rule: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record an exploratory plan without requiring raw protocol JSON."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    hypothesis = repository.get(hypothesis_id)
    if hypothesis["kind"] != "hypothesis":
        raise ResearchGitError("research plan hypothesis reference has wrong kind")
    resolved_hypothesis = str(hypothesis["object_id"])
    payload: dict[str, Any] = {
        "summary": _required_text(summary, label="research plan summary"),
        "study_phase": "exploratory",
        "hypothesis_id": resolved_hypothesis,
    }
    tests = [
        _required_text(value, label="discriminating test")
        for value in discriminating_tests
    ]
    if tests:
        payload["discriminating_tests"] = list(dict.fromkeys(tests))
    if success_rule.strip():
        payload["success_rule"] = success_rule.strip()
    result = repository.record(
        "research_plan",
        payload,
        state="draft",
        relations=[{"type": "depends_on", "target": resolved_hypothesis}],
    )
    return _finish(
        repository,
        result,
        stage="plan",
        subject=message or "record exploratory research plan",
        status="draft",
        commit=commit,
    )


def save_search_plan(
    repo: str,
    *,
    question: str,
    queries: Sequence[str],
    providers: Sequence[str] = (),
    inclusion_criteria: Sequence[str] = (),
    exclusion_criteria: Sequence[str] = (),
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Preregister a literature search so result selection is reviewable."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    normalized_queries = [
        _required_text(item, label="search query") for item in queries
    ]
    if not normalized_queries:
        raise ResearchGitError("search plan requires at least one query")
    core: dict[str, Any] = {
        "question": _required_text(question, label="search question"),
        "queries": list(dict.fromkeys(normalized_queries)),
        "providers": sorted(
            {_required_text(item, label="search provider") for item in providers}
        ),
        "inclusion_criteria": [
            _required_text(item, label="inclusion criterion")
            for item in inclusion_criteria
        ],
        "exclusion_criteria": [
            _required_text(item, label="exclusion criterion")
            for item in exclusion_criteria
        ],
    }
    payload = {**core, "search_plan_hash": canonical_content_hash(core)}
    result = repository.record("search_plan", payload, state="locked")
    return _finish(
        repository,
        result,
        stage="plan",
        subject=message or "lock literature search plan",
        status="locked",
        commit=commit,
    )


def save_search_receipt(
    repo: str,
    *,
    plan_id: str,
    provider: str,
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    retrieved_at: str = "",
    corpus_version: str = "",
    errors: Sequence[str] = (),
    query_rewrites: Sequence[str] = (),
    filters: Mapping[str, Any] | None = None,
    retrieval_system: Mapping[str, Any] | None = None,
    corpus_snapshot_hash: str = "",
    pagination: Mapping[str, Any] | None = None,
    transform_lineage: Sequence[Mapping[str, Any]] = (),
    complete: bool = True,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record the complete ranked candidate set returned by one search call."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    plan = repository.get(plan_id)
    if plan["kind"] != "search_plan":
        raise ResearchGitError("search receipt plan reference has wrong kind")
    resolved_plan = str(plan["object_id"])
    try:
        payload = build_search_receipt_payload(
            provider=_required_text(provider, label="search provider"),
            query=_required_text(query, label="search query"),
            candidates=candidates,
            retrieved_at=(
                retrieved_at.strip()
                or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            ),
            corpus_version=corpus_version,
            errors=errors,
            query_rewrites=query_rewrites,
            filters=filters,
            retrieval_system=retrieval_system,
            corpus_snapshot_hash=corpus_snapshot_hash,
            pagination=pagination,
            transform_lineage=transform_lineage,
            complete=complete,
        )
    except ValueError as exc:
        raise ResearchGitError(str(exc)) from exc
    result = repository.record(
        "search_receipt",
        payload,
        state="completed",
        relations=[
            {"type": "depends_on", "target": resolved_plan, "role": "search_plan"}
        ],
    )
    return _finish(
        repository,
        result,
        stage="evidence",
        subject=message or "record literature search receipt",
        status="completed",
        commit=commit,
    )


def save_source_snapshot(
    repo: str,
    *,
    receipt_id: str,
    title: str,
    content_hash: str,
    metadata_hash: str | None = None,
    doi: str = "",
    pmid: str = "",
    arxiv_id: str = "",
    url: str = "",
    license_name: str = "",
    retraction_status: str = "unknown",
    status_provider: str = "",
    status_checked_at: str = "",
    status_notice_id: str = "",
    previous_source_id: str | None = None,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Freeze identifiers and hashes for one selected literature source."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    receipt = repository.get(receipt_id)
    if receipt["kind"] != "search_receipt":
        raise ResearchGitError("source receipt reference has wrong kind")
    _hashes(
        [content_hash, *([metadata_hash] if metadata_hash else [])],
        label="source hash",
    )
    core = {
        "title": _required_text(title, label="source title"),
        "content_hash": content_hash,
        "metadata_hash": metadata_hash or "",
        "doi": doi.strip(),
        "pmid": pmid.strip(),
        "arxiv_id": arxiv_id.strip(),
        "url": url.strip(),
        "license": license_name.strip(),
        "retraction_status": retraction_status.strip() or "unknown",
    }
    if status_provider.strip() or status_checked_at.strip() or status_notice_id.strip():
        checked_at = status_checked_at.strip()
        if not checked_at:
            raise ResearchGitError(
                "source status provider requires a status checked-at timestamp"
            )
        try:
            parsed_checked_at = datetime.fromisoformat(
                checked_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ResearchGitError("source status checked-at must be ISO-8601") from exc
        if parsed_checked_at.tzinfo is None:
            raise ResearchGitError("source status checked-at must include a timezone")
        core["status_check"] = {
            "provider": _required_text(status_provider, label="status provider"),
            "checked_at": checked_at,
            "notice_id": status_notice_id.strip(),
        }
    payload = {**core, "source_hash": canonical_content_hash(core)}
    relations: list[dict[str, str]] = [
        {"type": "derived_from", "target": str(receipt["object_id"])}
    ]
    if previous_source_id:
        previous = repository.get(previous_source_id)
        if previous["kind"] != "source_snapshot":
            raise ResearchGitError("superseded source reference has wrong kind")
        relations.append({"type": "supersedes", "target": str(previous["object_id"])})
    result = repository.record(
        "source_snapshot", payload, state="completed", relations=relations
    )
    return _finish(
        repository,
        result,
        stage="evidence",
        subject=message or "record immutable literature source",
        status="completed",
        commit=commit,
    )


def save_source_update(
    repo: str,
    *,
    source_id: str,
    status: str,
    provider: str,
    checked_at: str,
    update_type: str = "status_check",
    notice_id: str = "",
    detail: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Append an immutable correction/retraction/status event for a source."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    source = repository.get(source_id)
    if source["kind"] != "source_snapshot":
        raise ResearchGitError("source update reference has wrong kind")
    normalized_checked_at = _required_text(checked_at, label="checked-at timestamp")
    try:
        parsed_checked_at = datetime.fromisoformat(
            normalized_checked_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ResearchGitError("checked-at timestamp must be ISO-8601") from exc
    if parsed_checked_at.tzinfo is None:
        raise ResearchGitError("checked-at timestamp must include a timezone")
    core = {
        "source_id": str(source["object_id"]),
        "status": _required_text(status, label="source status").lower(),
        "provider": _required_text(provider, label="status provider"),
        "checked_at": normalized_checked_at,
        "update_type": _required_text(update_type, label="source update type").lower(),
        "notice_id": notice_id.strip(),
        "detail": detail.strip(),
    }
    payload = {**core, "update_hash": canonical_content_hash(core)}
    relations = [{"type": "updates", "target": str(source["object_id"])}]
    if payload["status"] in {"retracted", "withdrawn", "invalid"} or payload[
        "update_type"
    ] in {"retraction", "withdrawal"}:
        relations.append({"type": "invalidates", "target": str(source["object_id"])})
    result = repository.record(
        "source_update", payload, state="completed", relations=relations
    )
    return _finish(
        repository,
        result,
        stage="evidence",
        subject=message or "record literature source status update",
        status="completed",
        commit=commit,
    )


def save_passage_evidence(
    repo: str,
    *,
    source_id: str,
    quote: str,
    locator: str,
    prefix: str = "",
    suffix: str = "",
    start: int | None = None,
    end: int | None = None,
    supports: Sequence[str] = (),
    refutes: Sequence[str] = (),
    context_id: str | None = None,
    scope: str = "",
    structured_scope: Mapping[str, Any] | None = None,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Bind a precise passage and locator to its immutable source snapshot."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    source = repository.get(source_id)
    if source["kind"] != "source_snapshot":
        raise ResearchGitError("passage source reference has wrong kind")
    normalized_quote = _required_text(quote, label="passage quote")
    normalized_locator = _required_text(locator, label="passage locator")
    quote_hash = canonical_content_hash(normalized_quote)
    try:
        selector = build_text_quote_selector(
            normalized_quote,
            prefix=prefix,
            suffix=suffix,
            start=start,
            end=end,
        )
    except ValueError as exc:
        raise ResearchGitError(str(exc)) from exc
    core = {
        "source_id": str(source["object_id"]),
        "locator": normalized_locator,
        "quote": normalized_quote,
        "quote_hash": quote_hash,
        "selector": selector,
        "selector_hash": selector["selector_hash"],
    }
    payload = {**core, "passage_hash": canonical_content_hash(core)}
    normalized_scope = normalize_claim_scope(structured_scope, legacy_text=scope)
    if normalized_scope:
        payload["scope"] = normalized_scope
        payload["scope_hash"] = claim_scope_hash(normalized_scope)
        payload["passage_hash"] = canonical_content_hash(
            {key: value for key, value in payload.items() if key != "passage_hash"}
        )
    relations: list[dict[str, str]] = [
        {"type": "quotes", "target": str(source["object_id"])}
    ]
    for selector in supports:
        target = repository.get(selector)
        relations.append(
            {"type": "qualified_supports", "target": str(target["object_id"])}
        )
    for selector in refutes:
        target = repository.get(selector)
        relations.append(
            {"type": "qualified_refutes", "target": str(target["object_id"])}
        )
    if context_id:
        context = repository.get(context_id)
        if context["kind"] != "context_snapshot":
            raise ResearchGitError("passage context reference has wrong kind")
        relations.append({"type": "uses_context", "target": str(context["object_id"])})
    result = repository.record(
        "passage_evidence", payload, state="completed", relations=relations
    )
    return _finish(
        repository,
        result,
        stage="evidence",
        subject=message or "record source-qualified passage evidence",
        status="completed",
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
    environment_hash: str | None = None,
    dependency_lock_hashes: Sequence[str] = (),
    dataset_hashes: Sequence[str] = (),
    code_commit: str | None = None,
    failure_class: str = "",
    reproduce_command: str | None = None,
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
    environment = capture_environment_receipt(repository.path)
    provenance: dict[str, Any] = {
        "environment_hash": environment_hash or environment["content_hash"],
        "dependency_lock_hashes": _hashes(
            [
                *(
                    str(item.get("hash") or "")
                    for item in environment["dependency_locks"]
                ),
                *dependency_lock_hashes,
            ],
            label="dependency lock hash",
        ),
        "dataset_hashes": _hashes(dataset_hashes, label="dataset hash"),
        "seeds": sorted(set(seeds)),
    }
    selected_commit = str(code_commit or repository.status().get("head") or "")
    if selected_commit:
        provenance["code_commit"] = selected_commit
    provenance = {
        key: value for key, value in provenance.items() if value not in (None, "", [])
    }
    lifecycle = ResearchLifecycle(repository)
    recorded = lifecycle.experiment_attempt(
        payload,
        preregistration_id=preregistration_id,
        plan_id=plan_id,
        provenance=provenance,
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
        reproduce_command=reproduce_command,
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
    hypothesis_id = str(hypothesis["object_id"])
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
    scope: str = "",
    structured_scope: Mapping[str, Any] | None = None,
    verified: bool = False,
    verifier_id: str | None = None,
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
    normalized_scope = normalize_claim_scope(structured_scope, legacy_text=scope)
    if normalized_scope:
        payload["scope"] = normalized_scope
        payload["scope_hash"] = claim_scope_hash(normalized_scope)
    payload["measurement_hash"] = content_hash(
        {"result": payload["result"], "metrics": payload.get("metrics") or {}}
    )
    lifecycle = ResearchLifecycle(repository)
    recorded = lifecycle.evidence(
        payload,
        attempt_ids=attempt_ids,
        supports=supports,
        refutes=refutes,
        verified=verified,
        verifier_id=verifier_id,
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


def save_estimand(
    repo: str,
    *,
    outcome: str,
    population: str,
    intervention: str = "",
    comparator: str = "",
    time_window: str = "",
    summary_measure: str = "",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record the precise scientific quantity an analysis intends to estimate."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    core = {
        "outcome": _required_text(outcome, label="estimand outcome"),
        "population": _required_text(population, label="estimand population"),
        "intervention": intervention.strip(),
        "comparator": comparator.strip(),
        "time_window": time_window.strip(),
        "summary_measure": summary_measure.strip(),
    }
    payload = {**core, "estimand_hash": canonical_content_hash(core)}
    result = repository.record("estimand", payload, state="locked")
    return _finish(
        repository,
        result,
        stage="plan",
        subject=message or "record target estimand",
        status="locked",
        commit=commit,
    )


def save_effect_estimate(
    repo: str,
    *,
    estimand_id: str,
    estimate: float,
    metric: str,
    unit: str = "",
    confidence_level: float = 0.95,
    interval_lower: float | None = None,
    interval_upper: float | None = None,
    standard_error: float | None = None,
    derived_from: Sequence[str] = (),
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record an effect and its uncertainty as a first-class DAG object."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    estimand = repository.get(estimand_id)
    if estimand["kind"] != "estimand":
        raise ResearchGitError("effect estimate estimand reference has wrong kind")
    if (interval_lower is None) != (interval_upper is None):
        raise ResearchGitError("confidence interval requires both lower and upper")
    if not 0 < float(confidence_level) < 1:
        raise ResearchGitError("confidence level must be between zero and one")
    core: dict[str, Any] = {
        "estimand_id": str(estimand["object_id"]),
        "estimate": float(estimate),
        "metric": _required_text(metric, label="effect metric"),
        "unit": unit.strip(),
    }
    if interval_lower is not None and interval_upper is not None:
        if interval_lower > interval_upper:
            raise ResearchGitError(
                "confidence interval lower bound exceeds upper bound"
            )
        core["confidence_interval"] = {
            "level": float(confidence_level),
            "lower": float(interval_lower),
            "upper": float(interval_upper),
        }
    if standard_error is not None:
        if standard_error < 0:
            raise ResearchGitError("standard error cannot be negative")
        core["standard_error"] = float(standard_error)
    if "confidence_interval" not in core and "standard_error" not in core:
        raise ResearchGitError(
            "effect estimate requires a confidence interval or standard error"
        )
    relations: list[dict[str, str]] = [
        {"type": "addresses_estimand", "target": str(estimand["object_id"])}
    ]
    for selector in derived_from:
        source = repository.get(selector)
        if source["kind"] not in {
            "evidence",
            "passage_evidence",
            "experiment_attempt",
            "observation",
        }:
            raise ResearchGitError("effect estimate source reference has wrong kind")
        relations.append({"type": "derived_from", "target": str(source["object_id"])})
    payload = {**core, "effect_hash": canonical_content_hash(core)}
    result = repository.record(
        "effect_estimate", payload, state="completed", relations=relations
    )
    return _finish(
        repository,
        result,
        stage="evidence",
        subject=message or "record effect estimate and uncertainty",
        status="completed",
        commit=commit,
    )


def save_inference(
    repo: str,
    *,
    statement: str,
    premises: Sequence[str],
    warrant: str,
    method_ids: Sequence[str] = (),
    assumption_ids: Sequence[str] = (),
    context_id: str | None = None,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record the explicit reasoning step from premises to a conclusion."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    if not premises:
        raise ResearchGitError("inference requires at least one premise")
    warrant_core = {"statement": _required_text(warrant, label="inference warrant")}
    warrant_payload = {
        **warrant_core,
        "warrant_hash": canonical_content_hash(warrant_core),
    }
    warrant_result = repository.record("warrant", warrant_payload, state="locked")
    relations: list[dict[str, str]] = [
        {
            "type": "depends_on",
            "target": warrant_result.object_id,
            "role": "warrant",
        }
    ]
    for selector in premises:
        premise = repository.get(selector)
        if premise["kind"] not in {
            "evidence",
            "passage_evidence",
            "observation",
            "effect_estimate",
            "claim",
            "inference",
            "evidence_synthesis",
        }:
            raise ResearchGitError("inference premise reference has wrong kind")
        relations.append({"type": "has_premise", "target": str(premise["object_id"])})
    for selector in method_ids:
        method = repository.get(selector)
        if method["kind"] != "method":
            raise ResearchGitError("inference method reference has wrong kind")
        relations.append({"type": "uses_method", "target": str(method["object_id"])})
    for selector in assumption_ids:
        assumption = repository.get(selector)
        if assumption["kind"] != "assumption":
            raise ResearchGitError("inference assumption reference has wrong kind")
        relations.append(
            {"type": "under_assumption", "target": str(assumption["object_id"])}
        )
    if context_id:
        context = repository.get(context_id)
        if context["kind"] != "context_snapshot":
            raise ResearchGitError("inference context reference has wrong kind")
        relations.append({"type": "uses_context", "target": str(context["object_id"])})
    core = {
        "statement": _required_text(statement, label="inference statement"),
        "warrant_id": warrant_result.object_id,
    }
    payload = {**core, "inference_hash": canonical_content_hash(core)}
    result = repository.record(
        "inference", payload, state="completed", relations=relations
    )
    return _finish(
        repository,
        result,
        stage="claim",
        subject=message or "record evidence-to-claim inference",
        status="completed",
        commit=commit,
        related=[warrant_result],
    )


def save_claim(
    repo: str,
    *,
    statement: str,
    evidence_ids: Sequence[str],
    scope: str = "",
    structured_scope: Mapping[str, Any] | None = None,
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
    normalized_scope = normalize_claim_scope(structured_scope, legacy_text=scope)
    if normalized_scope:
        payload["scope"] = normalized_scope
        payload["scope_hash"] = claim_scope_hash(normalized_scope)
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
    context = recorded["context"]
    return _finish(
        repository,
        gate,
        stage="review",
        subject=message or "record independent evidence gate",
        status=gate.state,
        commit=commit,
        # Preserve the long-standing CLI contract that the review is the first
        # related object; the exact context snapshot is an additive second item.
        related=[review, context],
    )


__all__ = [
    "save_claim",
    "save_effect_estimate",
    "save_estimand",
    "save_evidence",
    "save_experiment",
    "save_hypothesis",
    "save_inference",
    "save_passage_evidence",
    "save_preregistration",
    "save_research_plan",
    "save_review",
    "save_search_plan",
    "save_search_receipt",
    "save_source_snapshot",
    "save_source_update",
]
