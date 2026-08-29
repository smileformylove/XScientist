"""High-level, one-command research recording workflows for the public CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import re
from typing import Any
import unicodedata
from urllib.parse import urlsplit, urlunsplit

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
    rows = sorted({str(value).strip() for value in values if str(value).strip()})
    invalid = [
        value for value in rows if not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
    ]
    if invalid:
        raise ResearchGitError(f"{label} must use sha256:<64 lowercase hex>")
    return rows


def _normalized_literature_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _normalized_literature_identifier(kind: str, value: Any) -> str:
    """Normalize a persistent identifier conservatively for equality checks."""

    text = _normalized_literature_text(value)
    if not text:
        return ""
    folded = text.casefold()
    if kind == "doi":
        return re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", folded)
    if kind == "pmid":
        return re.sub(r"^pmid:\s*", "", folded)
    if kind == "arxiv_id":
        return re.sub(
            r"^(?:arxiv:\s*|https?://arxiv\.org/(?:abs|pdf)/)",
            "",
            folded,
        ).removesuffix(".pdf")
    if kind == "url":
        parsed = urlsplit(text)
        if parsed.scheme and parsed.netloc:
            path = parsed.path.rstrip("/") or "/"
            return urlunsplit(
                (
                    parsed.scheme.casefold(),
                    parsed.netloc.casefold(),
                    path,
                    parsed.query,
                    "",
                )
            )
        return folded.rstrip("/")
    return folded


def _selected_candidate_matches(
    receipt_payload: Mapping[str, Any],
    *,
    title: str,
    doi: str,
    pmid: str,
    arxiv_id: str,
    url: str,
) -> list[Mapping[str, Any]]:
    candidates = receipt_payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    selected = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and (
            item.get("selection_status") == "selected"
            or ("selection_status" not in item and item.get("selected") is True)
        )
    ]
    source_identifiers = {
        key: _normalized_literature_identifier(key, value)
        for key, value in {
            "doi": doi,
            "pmid": pmid,
            "arxiv_id": arxiv_id,
            "url": url,
        }.items()
        if _normalized_literature_identifier(key, value)
    }
    if source_identifiers:
        matches: list[Mapping[str, Any]] = []
        for item in selected:
            candidate_identifiers = {
                key: normalized
                for key in source_identifiers
                if (normalized := _normalized_literature_identifier(key, item.get(key)))
            }
            shared = set(source_identifiers) & set(candidate_identifiers)
            if shared and all(
                source_identifiers[key] == candidate_identifiers[key] for key in shared
            ):
                matches.append(item)
        return matches
    normalized_title = _normalized_literature_text(title).casefold()
    return [
        item
        for item in selected
        if _normalized_literature_text(item.get("title")).casefold() == normalized_title
    ]


def _relation_targets(item: Mapping[str, Any], relation_type: str) -> set[str]:
    return {
        str(row.get("target") or "")
        for row in item.get("relations") or []
        if isinstance(row, Mapping) and row.get("type") == relation_type
    }


def _is_retraction_update(item: Mapping[str, Any]) -> bool:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    return str(payload.get("status") or "").strip().lower() in {
        "retracted",
        "withdrawn",
        "invalid",
    } or str(payload.get("update_type") or "").strip().lower() in {
        "retraction",
        "withdrawal",
    }


def _source_update_time(item: Mapping[str, Any]) -> datetime | None:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    try:
        parsed = datetime.fromisoformat(
            str(payload.get("checked_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _valid_reinstatement_pair(
    successor: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    payload = successor.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    target_payload = target.get("payload")
    target_payload = target_payload if isinstance(target_payload, Mapping) else {}
    successor_time = _source_update_time(successor)
    target_time = _source_update_time(target)
    successor_source = str(payload.get("source_id") or "")
    target_source = str(target_payload.get("source_id") or "")
    return bool(
        successor.get("kind") == "source_update"
        and target.get("kind") == "source_update"
        and str(successor.get("state") or "") in {"completed", "verified", "promoted"}
        and str(payload.get("update_type") or "").strip().lower() == "reinstatement"
        and str(payload.get("status") or "").strip().lower()
        not in {"retracted", "withdrawn", "invalid"}
        and str(payload.get("notice_id") or "").strip()
        and str(payload.get("provider") or "")
        == str(target_payload.get("provider") or "")
        and successor_source
        and successor_source == target_source
        and _relation_targets(successor, "updates") == {successor_source}
        and len(_relation_targets(successor, "supersedes")) == 1
        and _relation_targets(target, "updates") == {target_source}
        and _relation_targets(target, "invalidates") == {target_source}
        and _is_retraction_update(target)
        and successor_time is not None
        and target_time is not None
        and successor_time > target_time
    )


def _active_retraction_updates(
    objects: Sequence[Mapping[str, Any]], source_id: str
) -> list[Mapping[str, Any]]:
    by_id = {
        str(item.get("object_id") or ""): item
        for item in objects
        if item.get("kind") == "source_update"
    }
    valid_superseded: set[str] = set()
    for item in by_id.values():
        for target_id in _relation_targets(item, "supersedes"):
            target = by_id.get(target_id)
            if target is not None and _valid_reinstatement_pair(item, target):
                valid_superseded.add(target_id)
    return [
        item
        for object_id, item in by_id.items()
        if object_id not in valid_superseded
        and str(item.get("state") or "") in {"completed", "verified", "promoted"}
        and str((item.get("payload") or {}).get("source_id") or "") == source_id
        and _relation_targets(item, "updates") == {source_id}
        and _relation_targets(item, "invalidates") == {source_id}
        and _is_retraction_update(item)
    ]


def _ensure_direct_save_is_safe(
    repository: ResearchRepository, *, commit: bool
) -> None:
    if not commit:
        return
    status = repository.status()
    staged = status.get("research_stage", {}).get("paths") or []
    if staged:
        raise ResearchGitError(
            "one-command research recording requires an empty native stage; "
            "commit it with `xscientist git commit` or clear it with "
            "`xscientist research unstage --all`"
        )
    pending = sorted(
        {
            *list(status.get("staged_paths") or []),
            *list(status.get("tracked_changes") or []),
            *list(status.get("eligible_changes") or []),
        }
    )
    if pending:
        raise ResearchGitError(
            "one-command research recording requires a clean research worktree; "
            "checkpoint, stage, or restore the pending paths first: "
            + ", ".join(pending)
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
        created_results = [item for item in (result, *related) if item.created]
        if not created_results:
            return {
                "object": result,
                "related": list(related),
                "checkpoint": CheckpointResult(
                    created=False,
                    committed=False,
                    reason="research objects already exist",
                ),
            }
        includes = [
            item.path.relative_to(repository.path).as_posix()
            for item in created_results
        ]
        checkpoint = create_checkpoint(
            repository.path,
            stage=stage,
            subject=subject,
            status=status,
            only_paths=includes,
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
        _normalized_literature_text(_required_text(item, label="search query"))
        for item in queries
    ]
    if not normalized_queries:
        raise ResearchGitError("search plan requires at least one query")
    core: dict[str, Any] = {
        "question": _required_text(question, label="search question"),
        "queries": list(dict.fromkeys(normalized_queries)),
        "providers": sorted(
            {
                _normalized_literature_text(
                    _required_text(item, label="search provider")
                )
                for item in providers
            }
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
    if plan.get("state") != "locked":
        raise ResearchGitError("search receipt requires a locked search plan")
    resolved_plan = str(plan["object_id"])
    plan_payload = plan.get("payload")
    plan_payload = plan_payload if isinstance(plan_payload, Mapping) else {}
    expected_plan_hash = canonical_content_hash(
        {key: value for key, value in plan_payload.items() if key != "search_plan_hash"}
    )
    if plan_payload.get("search_plan_hash") != expected_plan_hash:
        raise ResearchGitError("search plan commitment hash is invalid")
    normalized_provider = _normalized_literature_text(
        _required_text(provider, label="search provider")
    )
    normalized_query = _normalized_literature_text(
        _required_text(query, label="search query")
    )
    locked_providers = plan_payload.get("providers") or []
    if not isinstance(locked_providers, list):
        raise ResearchGitError("locked search plan providers must be an array")
    if locked_providers and normalized_provider not in {
        _normalized_literature_text(item) for item in locked_providers
    }:
        raise ResearchGitError(
            "search receipt provider is not allowed by the locked search plan"
        )
    locked_queries = plan_payload.get("queries")
    if not isinstance(locked_queries, list) or normalized_query not in {
        _normalized_literature_text(item) for item in locked_queries
    }:
        raise ResearchGitError(
            "search receipt query must exactly match a locked search-plan query"
        )
    try:
        payload = build_search_receipt_payload(
            provider=normalized_provider,
            query=normalized_query,
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
    plan_binding = {
        "object_id": resolved_plan,
        "content_hash": str(plan.get("content_hash") or ""),
        "search_plan_hash": str(plan_payload.get("search_plan_hash") or ""),
    }
    receipt_core = {
        **{key: value for key, value in payload.items() if key != "receipt_hash"},
        "plan_binding": plan_binding,
    }
    payload = {**receipt_core, "receipt_hash": canonical_content_hash(receipt_core)}
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
    receipt_payload = receipt.get("payload")
    receipt_payload = receipt_payload if isinstance(receipt_payload, Mapping) else {}
    expected_receipt_hash = canonical_content_hash(
        {key: value for key, value in receipt_payload.items() if key != "receipt_hash"}
    )
    if receipt_payload.get("receipt_hash") != expected_receipt_hash:
        raise ResearchGitError("source receipt commitment hash is invalid")
    if receipt_payload.get(
        "profile"
    ) == "xscientist.retrieval-receipt.v2" and receipt_payload.get(
        "candidate_set_hash"
    ) != canonical_content_hash(
        receipt_payload.get("candidates") or []
    ):
        raise ResearchGitError("source receipt candidate-set hash is invalid")
    _hashes(
        [content_hash, *([metadata_hash] if metadata_hash else [])],
        label="source hash",
    )
    normalized_title = _required_text(title, label="source title")
    matching_candidates = _selected_candidate_matches(
        receipt_payload,
        title=normalized_title,
        doi=doi,
        pmid=pmid,
        arxiv_id=arxiv_id,
        url=url,
    )
    if not matching_candidates:
        raise ResearchGitError(
            "source must match a selected candidate in its search receipt"
        )
    if len(matching_candidates) != 1:
        raise ResearchGitError(
            "source matches multiple selected candidates; use an unambiguous "
            "persistent identifier"
        )
    matched_candidate = matching_candidates[0]
    core = {
        "title": normalized_title,
        "content_hash": content_hash,
        "metadata_hash": metadata_hash or "",
        "doi": doi.strip(),
        "pmid": pmid.strip(),
        "arxiv_id": arxiv_id.strip(),
        "url": url.strip(),
        "license": license_name.strip(),
        "retraction_status": retraction_status.strip() or "unknown",
        "receipt_binding": {
            "object_id": str(receipt["object_id"]),
            "content_hash": str(receipt.get("content_hash") or ""),
            "receipt_hash": str(receipt_payload.get("receipt_hash") or ""),
        },
        "candidate_binding": {
            "candidate_hash": canonical_content_hash(dict(matched_candidate)),
            "selection_status": "selected",
        },
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
    relations = [{"type": "updates", "target": str(source["object_id"])}]
    if core["update_type"] == "reinstatement":
        if core["status"] in {"retracted", "withdrawn", "invalid"}:
            raise ResearchGitError(
                "source reinstatement requires a non-invalidating status"
            )
        if not core["notice_id"]:
            raise ResearchGitError("source reinstatement requires a notice id")
        active_retractions = _active_retraction_updates(
            repository.objects(kind="source_update"), str(source["object_id"])
        )
        if not active_retractions:
            raise ResearchGitError(
                "source reinstatement requires an active retraction update"
            )
        latest_retraction = max(
            active_retractions,
            key=lambda item: (
                _source_update_time(item) or datetime.min.replace(tzinfo=timezone.utc),
                str(item.get("created_at") or ""),
                str(item.get("object_id") or ""),
            ),
        )
        latest_payload = latest_retraction.get("payload") or {}
        if core["provider"] != str(latest_payload.get("provider") or ""):
            raise ResearchGitError(
                "source reinstatement provider must match the latest active retraction"
            )
        latest_time = _source_update_time(latest_retraction)
        if latest_time is None or parsed_checked_at <= latest_time:
            raise ResearchGitError(
                "source reinstatement must be checked after the latest active "
                "retraction"
            )
        relations.append(
            {
                "type": "supersedes",
                "target": str(latest_retraction["object_id"]),
            }
        )
    payload = {**core, "update_hash": canonical_content_hash(core)}
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
    priority_id: str | None = None,
    preregistration_id: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = (),
    environment_hash: str | None = None,
    dependency_lock_hashes: Sequence[str] = (),
    dataset_hashes: Sequence[str] = (),
    code_commit: str | None = None,
    failure_class: str = "",
    interventions: Sequence[str] = (),
    boundary_condition: str = "",
    boundary_role: str = "",
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
    if interventions:
        payload["interventions"] = [
            _required_text(value, label="experiment intervention")
            for value in interventions
        ]
    if boundary_condition.strip() or boundary_role.strip():
        if not boundary_condition.strip() or boundary_role not in {
            "development",
            "transfer",
            "heldout",
            "scale",
        }:
            raise ResearchGitError(
                "experiment boundary requires a condition and valid boundary role"
            )
        payload["boundary_condition"] = _required_text(
            boundary_condition, label="experiment boundary condition"
        )
        payload["boundary_role"] = boundary_role
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
        priority_id=priority_id,
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
    contribution_level: str = "",
    depth_level: str = "descriptive",
    mechanism_ids: Sequence[str] = (),
    quality_ids: Sequence[str] = (),
    transfer_ids: Sequence[str] = (),
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
    normalized_contribution = str(contribution_level or "").strip()
    if normalized_contribution:
        if normalized_contribution not in {
            "execution",
            "engineering_optimization",
            "method_discovery",
        }:
            raise ResearchGitError("claim contribution_level is invalid")
        payload["contribution_level"] = normalized_contribution
    normalized_depth = str(depth_level or "descriptive").strip()
    if normalized_depth not in {"descriptive", "causal", "transferable"}:
        raise ResearchGitError("claim depth_level is invalid")
    payload["depth_level"] = normalized_depth
    lifecycle = ResearchLifecycle(repository)
    recorded = lifecycle.claim(
        payload,
        evidence_ids=evidence_ids,
        qualification_ids=[*mechanism_ids, *quality_ids, *transfer_ids],
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
