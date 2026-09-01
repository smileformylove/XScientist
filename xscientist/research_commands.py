"""High-level, one-command research recording workflows for the public CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from ai_scientist.protocol import content_hash
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.atomic_io import atomic_write_json
from ai_scientist.utils.research_integrity import (
    ResearchIntegrityError,
    build_preregistration,
    derive_adaptive_state_hashes,
    lock_preregistration,
    validate_empirical_data_manifest,
    validate_preregistration,
)
from ai_scientist.utils.safe_files import (
    BoundedFileError,
    read_bounded_regular_file,
)
from ai_scientist.utils.trajectory_binding import (
    ATTEMPT_DISPOSITION_PROTOCOL,
    ATTEMPT_DISPOSITIONS,
    TRAJECTORY_BINDING_PROTOCOL,
    attempt_registry_contract_errors,
    build_terminal_negative_artifact_receipt,
    registry_row_hash,
    terminal_negative_contract_errors,
)

from .research_git import (
    CheckpointResult,
    ResearchGitError,
    ResearchObjectResult,
    _add_research_objects_atomically,
    _repository_lock,
    _rollback_new_research_object_pointers_locked,
    capture_environment_receipt,
    create_checkpoint,
    research_object_introduction_order,
    repository_status,
    validate_research_logical_component,
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


def _read_bounded_project_file(
    paper_root: Path,
    filename: str,
    *,
    label: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    path = paper_root / filename
    try:
        return read_bounded_regular_file(
            path,
            maximum=max_bytes,
            label=filename.replace(".", "_"),
        )
    except BoundedFileError as exc:
        raise ResearchGitError(f"{label} is missing or unsafe") from exc


def _decode_project_json(encoded: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ResearchGitError(f"{label} is missing, unsafe, or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ResearchGitError(f"{label} must be a JSON object")
    return payload


def _load_bounded_project_json(
    paper_root: Path,
    filename: str,
    *,
    label: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    return _decode_project_json(
        _read_bounded_project_file(
            paper_root,
            filename,
            label=label,
            max_bytes=max_bytes,
        ),
        label=label,
    )


def _load_bounded_registry_rows(paper_root: Path) -> list[dict[str, Any]]:
    encoded = _read_bounded_project_file(
        paper_root,
        "experiment_registry.jsonl",
        label="experiment registry",
        max_bytes=64 * 1024 * 1024,
    )
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise ResearchGitError("experiment registry is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if len(rows) >= 100_000:
            raise ResearchGitError("experiment registry exceeds 100000 rows")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResearchGitError(
                f"experiment registry line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ResearchGitError(
                f"experiment registry line {line_number} is not an object"
            )
        rows.append(row)
    return rows


def _locked_registration_object(
    repository: ResearchRepository, preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    matches = [
        item
        for item in repository.objects(kind="preregistration", state="locked")
        if (item.get("payload") or {}).get("registration_hash")
        == preregistration.get("registration_hash")
        and (item.get("payload") or {}).get("preregistration_id")
        == preregistration.get("preregistration_id")
    ]
    if len(matches) != 1:
        raise ResearchGitError(
            "paper preregistration must resolve to exactly one locked Research VCS object"
        )
    return matches[0]


def _read_optional_project_file(
    path: Path,
    *,
    label: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> bytes | None:
    try:
        return read_bounded_regular_file(
            path,
            maximum=max_bytes,
            label=path.name.replace(".", "_"),
        )
    except BoundedFileError as exc:
        if exc.reason == "missing":
            return None
        raise ResearchGitError(f"{label} is unsafe or unreadable") from exc


def _canonical_confirmatory_tasks(
    research_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_tasks = research_plan.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks or len(raw_tasks) > 256:
        raise ResearchGitError(
            "research plan must contain between 1 and 256 confirmatory tasks"
        )
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise ResearchGitError("research plan task must be a JSON object")
        task_id = _required_text(
            str(raw.get("task_id") or ""), label=f"research task {index} id"
        )
        if task_id in seen:
            raise ResearchGitError(f"duplicate research task id: {task_id}")
        seen.add(task_id)
        task = dict(raw)
        task["task_id"] = task_id
        for field in ("dataset", "metric", "baseline"):
            task[field] = _required_text(
                str(task.get(field) or ""), label=f"{task_id} {field}"
            )
        tasks.append(task)
    return tasks


def _restore_atomic_json(path: Path, previous: bytes | None) -> None:
    """Restore one campaign mirror after a failed multi-file transition."""

    if previous is None:
        path.unlink(missing_ok=True)
        return
    from ai_scientist.utils.atomic_io import atomic_write_bytes

    atomic_write_bytes(path, previous)


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
    include_paths: Sequence[str] = (),
    object_refs: Sequence[str] = (),
) -> dict[str, Any]:
    checkpoint: CheckpointResult | None = None
    if commit:
        created_results = [item for item in (result, *related) if item.created]
        if not created_results and not include_paths:
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
        includes.extend(str(path) for path in include_paths)
        checkpoint = create_checkpoint(
            repository.path,
            stage=stage,
            subject=subject,
            status=status,
            only_paths=sorted(set(includes)),
            object_refs=object_refs,
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
        introduction_order = research_object_introduction_order(repository.path)
        latest_time = max(
            _source_update_time(item) or datetime.min.replace(tzinfo=timezone.utc)
            for item in active_retractions
        )
        time_matches = [
            item
            for item in active_retractions
            if (_source_update_time(item) or datetime.min.replace(tzinfo=timezone.utc))
            == latest_time
        ]
        latest_sequence = max(
            introduction_order.get(str(item["object_id"]), -1) for item in time_matches
        )
        latest_matches = [
            item
            for item in time_matches
            if introduction_order.get(str(item["object_id"]), -1) == latest_sequence
        ]
        if latest_sequence < 0 or len(latest_matches) != 1:
            raise ResearchGitError(
                "active retractions are ambiguous at the latest source/Git event; "
                "record an explicit superseding update"
            )
        latest_retraction = latest_matches[0]
        latest_payload = latest_retraction.get("payload") or {}
        if core["provider"] != str(latest_payload.get("provider") or ""):
            raise ResearchGitError(
                "source reinstatement provider must match the latest active retraction"
            )
        latest_retraction_time = _source_update_time(latest_retraction)
        if (
            latest_retraction_time is None
            or parsed_checked_at <= latest_retraction_time
        ):
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
    task_id: str | None = None,
    plan_id: str | None = None,
    priority_id: str | None = None,
    preregistration_id: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
    producer_id: str | None = None,
    result_artifact_hashes: Mapping[str, str] | None = None,
    result_artifact_paths: Mapping[str, str | Path] | None = None,
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
    if task_id is not None:
        payload["task_id"] = _required_text(task_id, label="experiment task id")
    if metrics:
        payload["metrics"] = dict(metrics)
    if configuration:
        normalized_configuration = dict(configuration)
        payload["configuration"] = normalized_configuration
        payload["configuration_hash"] = canonical_content_hash(normalized_configuration)
    if producer_id is not None:
        payload["producer_id"] = _required_text(
            producer_id, label="experiment producer id"
        )
    artifact_hashes = (
        {
            _required_text(str(name), label="result artifact label"): _hashes(
                [str(digest)], label="result artifact hash"
            )[0]
            for name, digest in sorted(result_artifact_hashes.items())
        }
        if result_artifact_hashes
        else {}
    )
    artifact_pointers: dict[str, Any] = {}
    pointer_paths: set[str] = set()
    pointer_refs: set[str] = set()
    artifact_sources: dict[str, Path] = {}
    if result_artifact_paths:
        for raw_name, raw_path in sorted(result_artifact_paths.items()):
            name = _required_text(str(raw_name), label="result artifact label")
            source = Path(raw_path).expanduser()
            if not source.is_file():
                raise ResearchGitError(f"result artifact is not a regular file: {name}")
            digest = hashlib.sha256()
            try:
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                raise ResearchGitError(
                    f"result artifact could not be read: {name}"
                ) from exc
            source_hash = "sha256:" + digest.hexdigest()
            declared_hash = artifact_hashes.get(name)
            if declared_hash is not None and declared_hash != source_hash:
                raise ResearchGitError(f"result artifact hash mismatch: {name}")
            artifact_hashes[name] = source_hash
            artifact_sources[name] = source
    if artifact_hashes:
        payload["result_artifact_hashes"] = artifact_hashes
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
    selected_commit = str(
        code_commit
        or (
            ""
            if str(study_phase).strip().lower() == "confirmatory"
            else repository.status().get("head") or ""
        )
    )
    if selected_commit:
        provenance["code_commit"] = selected_commit
    provenance = {
        key: value for key, value in provenance.items() if value not in (None, "", [])
    }
    artifact_requests: list[tuple[Path, str, str | None, str | None]] = []
    artifact_logical_paths: dict[str, str] = {}
    for name, source in sorted(artifact_sources.items()):
        label_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
        label_slug = (label_slug or "artifact")[:48]
        label_digest = canonical_content_hash(name).split(":", 1)[1][:12]
        filename = validate_research_logical_component(source.name)
        logical_path = f"result-artifacts/{label_slug}-{label_digest}/{filename}"
        artifact_logical_paths[name] = logical_path
        artifact_requests.append(
            (source, logical_path, None, artifact_hashes.get(name))
        )

    with _repository_lock(repository.path):
        _ensure_direct_save_is_safe(repository, commit=commit)
        transaction_head = str(repository.status().get("head") or "")
        if not code_commit and str(study_phase).strip().lower() != "confirmatory":
            provenance["code_commit"] = transaction_head
        lifecycle = ResearchLifecycle(repository)
        prepared = lifecycle.validate_experiment_attempt(
            payload,
            preregistration_id=preregistration_id,
            plan_id=plan_id,
            priority_id=priority_id,
            provenance=provenance,
        )
        pointer_results = []
        recorded_result: ResearchObjectResult | None = None
        try:
            pointer_results = _add_research_objects_atomically(
                repository.path,
                artifact_requests,
            )
            pointers_by_label = dict(
                zip(sorted(artifact_sources), pointer_results, strict=True)
            )
            for name, pointer in sorted(pointers_by_label.items()):
                logical_path = artifact_logical_paths[name]
                pointer_path = pointer.pointer_path.relative_to(
                    repository.path
                ).as_posix()
                artifact_pointers[name] = {
                    "logical_name": name,
                    "content_hash": pointer.object_hash,
                    "logical_path": logical_path,
                    "pointer_path": pointer_path,
                }
                pointer_paths.add(pointer_path)
                pointer_refs.add(pointer.object_hash)
            if artifact_pointers:
                prepared_payload = dict(prepared["payload"])
                prepared_payload["result_artifacts"] = artifact_pointers
                prepared = {**prepared, "payload": prepared_payload}
            recorded = lifecycle.record_validated_experiment_attempt(
                prepared,
                commit=False,
            )
            result = recorded["attempt"]
            recorded_result = result
            stage = "experiment" if result.state == "completed" else "failed"
            return _finish(
                repository,
                result,
                stage=stage,
                subject=message or f"record {result.state} experiment attempt",
                status=result.state,
                commit=commit,
                reproduce_command=reproduce_command,
                include_paths=sorted(pointer_paths),
                object_refs=sorted(pointer_refs),
            )
        except BaseException:
            current_head = str(repository.status().get("head") or "")
            if current_head == transaction_head:
                _rollback_new_research_object_pointers_locked(pointer_results)
                if recorded_result is not None and recorded_result.created:
                    recorded_result.path.unlink(missing_ok=True)
            raise


def _confirmatory_queue_contract(
    tasks: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    *,
    split_hashes: Mapping[str, str],
    data_manifest_hash: str,
    data_snapshot_id: str,
) -> dict[str, Any]:
    """Build the immutable executor/result-recording contract locked by preregistration."""

    contract_tasks: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        outcome = outcomes[task_id]
        row = {
            "queue_index": index,
            "task_id": task_id,
            "goal": task.get("goal"),
            "dependencies": list(task.get("dependencies") or []),
            "dataset": outcome.get("dataset"),
            "metric": outcome.get("metric"),
            "baseline": outcome.get("baseline"),
            "evidence_role": outcome.get("evidence_role"),
            "paired_control_task_id": outcome.get("paired_control_task_id"),
            "intervention_variant": outcome.get("intervention_variant"),
            "stress_condition": outcome.get("stress_condition"),
            "transformation_contract": outcome.get("transformation_contract"),
            "transformation_contract_hash": outcome.get("transformation_contract_hash"),
            "split_hash": split_hashes[task_id],
            "data_manifest_hash": data_manifest_hash,
            "data_snapshot_id": data_snapshot_id,
            "allowed_terminal_states": [
                "completed",
                "failed",
                "timed_out",
                "cancelled",
            ],
            "required_completed_fields": [
                "producer_id",
                "configuration",
                "configuration_hash",
                "result_artifact_hashes",
            ],
            "required_unsuccessful_fields": ["producer_id", "failure_class"],
            "record_command_template": [
                "xscientist",
                "research",
                "experiment",
                "{RESULT_SUMMARY}",
                "--status",
                "{TERMINAL_STATUS}",
                "--study-phase",
                "confirmatory",
                "--task",
                task_id,
                "--plan",
                "{PLAN_OBJECT_ID}",
                "--preregistration",
                "{PREREGISTRATION_OBJECT_ID}",
                "--producer-id",
                "{PRODUCER_ID}",
                "--repo",
                ".",
            ],
            "record_command_terminal_arguments": {
                "completed": [
                    "--config",
                    "{NAME=VALUE}",
                    "--result-artifact",
                    "{LABEL=PATH}",
                ],
                "unsuccessful": [
                    "--failure-class",
                    "{FAILURE_CLASS}",
                ],
            },
        }
        row["task_contract_hash"] = canonical_content_hash(row)
        contract_tasks.append(row)
    core = {
        "schema": "xscientist.confirmatory-queue-contract.v1",
        "tasks": contract_tasks,
    }
    return {**core, "queue_contract_hash": canonical_content_hash(core)}


def confirm_paper_research(
    paper_dir: str,
    *,
    registered_by: str = "recorder:xscientist-user",
    split_hashes: Mapping[str, str] | None = None,
    data_manifest_hash: str | None = None,
    data_snapshot_id: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Lock a generated multi-task paper plan and materialize its run queue.

    The planning model never grants itself confirmatory authority.  This
    host-owned transition revalidates the generated plan, empirical snapshot,
    task splits, and current Research VCS head; records the plan and locked
    registration as immutable objects; then checkpoints inspectable mirrors in
    the same paper directory.
    """

    requested_paper_root = Path(paper_dir).expanduser()
    if requested_paper_root.is_symlink():
        raise ResearchGitError("paper directory must not be a symbolic link")
    paper_root = requested_paper_root.resolve()
    if not paper_root.is_dir():
        raise ResearchGitError("paper directory was not found or is a symbolic link")
    status = repository_status(paper_root)
    repository = ResearchRepository(str(status["repository"]))
    repository_root = repository.path
    try:
        paper_relative = paper_root.relative_to(repository_root)
    except ValueError as exc:
        raise ResearchGitError(
            "paper directory must be inside the selected Research VCS repository"
        ) from exc
    _ensure_direct_save_is_safe(repository, commit=True)

    preregistration_path = paper_root / "preregistration.json"
    queue_path = paper_root / "confirmatory_queue.json"
    previous_preregistration = _read_bounded_project_file(
        paper_root,
        "preregistration.json",
        label="generated preregistration draft",
    )
    previous_queue = _read_optional_project_file(
        queue_path,
        label="existing confirmatory queue",
    )
    research_plan = _load_bounded_project_json(
        paper_root,
        "research_plan.json",
        label="generated research plan",
    )
    draft = _decode_project_json(
        previous_preregistration,
        label="generated preregistration draft",
    )
    tasks = _canonical_confirmatory_tasks(research_plan)
    plan_id = _required_text(
        str(research_plan.get("plan_id") or ""), label="research plan id"
    )
    if draft.get("status") != "draft":
        raise ResearchGitError(
            "paper preregistration must be a draft; an existing lock must be "
            "audited rather than silently replaced"
        )
    if str(draft.get("plan_id") or "").strip() != plan_id:
        raise ResearchGitError(
            "preregistration draft does not bind the generated research plan"
        )

    raw_outcomes = draft.get("outcomes")
    if not isinstance(raw_outcomes, list):
        raise ResearchGitError("preregistration outcomes must be an array")
    outcomes = {
        str(item.get("task_id") or "").strip(): item
        for item in raw_outcomes
        if isinstance(item, Mapping) and str(item.get("task_id") or "").strip()
    }
    task_ids = {str(task["task_id"]) for task in tasks}
    if len(outcomes) != len(raw_outcomes) or set(outcomes) != task_ids:
        raise ResearchGitError(
            "preregistration outcomes must match every generated research task exactly"
        )
    scientific_fields = (
        "dataset",
        "metric",
        "baseline",
        "evidence_role",
        "paired_control_task_id",
        "intervention_variant",
        "stress_condition",
    )
    for task in tasks:
        task_id = str(task["task_id"])
        outcome = outcomes[task_id]
        if any(outcome.get(field) != task.get(field) for field in scientific_fields):
            raise ResearchGitError(
                f"preregistration outcome {task_id} differs from its generated task "
                "contract"
            )
        transformation = outcome.get("transformation_contract")
        transformation_hash = outcome.get("transformation_contract_hash")
        if (
            transformation is not None
            and transformation_hash != canonical_content_hash(transformation)
        ):
            raise ResearchGitError(
                f"preregistration outcome {task_id} transformation hash is invalid"
            )

    data_report = validate_empirical_data_manifest(paper_root)
    if data_report.get("ok") is not True:
        raise ResearchGitError(
            "confirmatory research requires a verified read-only empirical data "
            "snapshot: " + ", ".join(data_report.get("errors") or [])
        )
    locked_manifest_hash = str(data_report.get("manifest_hash") or "")
    locked_snapshot_id = str(data_report.get("snapshot_id") or "")
    if data_manifest_hash and data_manifest_hash != locked_manifest_hash:
        raise ResearchGitError(
            "supplied data manifest hash differs from the host-verified manifest"
        )
    if data_snapshot_id and data_snapshot_id != locked_snapshot_id:
        raise ResearchGitError(
            "supplied data snapshot id differs from the host-verified snapshot"
        )

    existing_split_hashes = (draft.get("data_policy") or {}).get("split_hashes")
    existing_split_hashes = (
        existing_split_hashes if isinstance(existing_split_hashes, Mapping) else {}
    )
    supplied_splits = dict(split_hashes or {})
    conflicting_splits = sorted(
        str(task_id)
        for task_id in set(existing_split_hashes) & set(supplied_splits)
        if str(existing_split_hashes[task_id]).strip()
        != str(supplied_splits[task_id]).strip()
    )
    if conflicting_splits:
        raise ResearchGitError(
            "supplied split hashes conflict with the generated preregistration: "
            + ", ".join(conflicting_splits)
        )
    selected_splits = {
        str(key).strip(): str(value).strip()
        for key, value in {**existing_split_hashes, **supplied_splits}.items()
        if str(key).strip()
    }
    if set(selected_splits) != task_ids:
        missing = sorted(task_ids - set(selected_splits))
        unknown = sorted(set(selected_splits) - task_ids)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        raise ResearchGitError(
            "confirmatory split hashes must cover every task exactly"
            + (": " + "; ".join(detail) if detail else "")
        )
    _hashes(list(selected_splits.values()), label="dataset split hash")

    alternative = _required_text(
        str((draft.get("hypotheses") or {}).get("alternative") or ""),
        label="registered alternative hypothesis",
    )
    falsifiers = (draft.get("hypotheses") or {}).get("falsifiers") or []
    falsifier = next(
        (str(item).strip() for item in falsifiers if str(item).strip()),
        "",
    )
    falsifier = _required_text(falsifier, label="registered falsifier")
    hypothesis_matches = [
        item
        for item in repository.objects(kind="hypothesis")
        if str((item.get("payload") or {}).get("statement") or "").strip()
        == alternative
        and str((item.get("payload") or {}).get("falsifier") or "").strip() == falsifier
    ]
    hypothesis_checkpoint: CheckpointResult | None = None
    if len(hypothesis_matches) > 1:
        raise ResearchGitError(
            "multiple Research VCS hypotheses match the generated paper; resolve "
            "the ambiguity before confirmation"
        )
    if hypothesis_matches:
        hypothesis_id = str(hypothesis_matches[0]["object_id"])
    else:
        hypothesis = repository.record(
            "hypothesis",
            {"statement": alternative, "falsifier": falsifier},
            state="draft",
            actor={"actor_id": registered_by, "authority": "recorder"},
        )
        hypothesis_id = hypothesis.object_id
        hypothesis_checkpoint = create_checkpoint(
            repository_root,
            stage="ideation",
            subject="record generated paper hypothesis before confirmation",
            status="completed",
            only_paths=[hypothesis.path.relative_to(repository_root).as_posix()],
        )

    repository_state = repository.status()
    if repository_state.get("worktree_clean") is not True:
        raise ResearchGitError(
            "confirmatory freeze requires a clean Research VCS worktree"
        )
    research_vcs_head = _required_text(
        str(repository_state.get("head") or ""),
        label="Research VCS checkpoint before confirmatory freeze",
    )
    preregistration_relative = (paper_relative / "preregistration.json").as_posix()
    queue_relative = (paper_relative / "confirmatory_queue.json").as_posix()
    if preregistration_relative.startswith("./"):
        preregistration_relative = preregistration_relative[2:]
    if queue_relative.startswith("./"):
        queue_relative = queue_relative[2:]
    draft = dict(draft)
    draft["data_policy"] = dict(draft.get("data_policy") or {})
    draft["data_policy"].update(
        {
            "split_hashes": dict(sorted(selected_splits.items())),
            "data_manifest_hash": locked_manifest_hash,
            "data_snapshot_id": locked_snapshot_id,
        }
    )
    queue_contract = _confirmatory_queue_contract(
        tasks,
        outcomes,
        split_hashes=selected_splits,
        data_manifest_hash=locked_manifest_hash,
        data_snapshot_id=locked_snapshot_id,
    )
    draft["confirmatory_campaign"] = {
        "schema": "xscientist.confirmatory-campaign.v1",
        "preregistration_path": preregistration_relative,
        "queue_path": queue_relative,
        "queue_contract": queue_contract,
        "queue_contract_hash": queue_contract["queue_contract_hash"],
    }
    try:
        derived_state_hashes = derive_adaptive_state_hashes(
            draft,
            research_vcs_head=research_vcs_head,
        )
        locked = lock_preregistration(
            draft,
            split_hashes=dict(sorted(selected_splits.items())),
            registered_by=registered_by,
            freeze_inputs={
                "research_vcs_head": research_vcs_head,
                **derived_state_hashes,
            },
        )
    except ResearchIntegrityError as exc:
        raise ResearchGitError(str(exc)) from exc
    validation = validate_preregistration(locked, require_locked=True)
    if validation.get("ok") is not True:
        raise ResearchGitError(
            "host-generated locked preregistration failed validation: "
            + ", ".join(validation.get("errors") or [])
        )

    plan = repository.record(
        "research_plan",
        research_plan,
        state="draft",
        relations=[{"type": "depends_on", "target": hypothesis_id}],
        actor={"actor_id": registered_by, "authority": "recorder"},
    )
    registration = repository.record(
        "preregistration",
        locked,
        state="locked",
        relations=[{"type": "depends_on", "target": plan.object_id}],
        actor={"actor_id": registered_by, "authority": "recorder"},
    )
    recorded = _finish(
        repository,
        registration,
        stage="preregister",
        subject=message or "lock generated multi-task confirmatory campaign",
        status="completed",
        commit=True,
        related=[plan],
    )
    registration_checkpoint = recorded.get("checkpoint")
    if registration_checkpoint is None or not registration_checkpoint.committed:
        raise ResearchGitError(
            "confirmatory campaign lock was not committed to Research VCS"
        )

    locked_campaign = locked.get("confirmatory_campaign") or {}
    locked_queue_contract = locked_campaign.get("queue_contract") or {}
    queue_tasks: list[dict[str, Any]] = []
    for contract_task in locked_queue_contract.get("tasks") or []:
        task_id = str(contract_task.get("task_id") or "")
        queue_tasks.append(
            {
                **dict(contract_task),
                "status": "queued",
                "study_phase": "confirmatory",
                "adaptive_state_hash": locked["adaptive_state_freeze"]["state_hash"],
                "research_state_hash": locked["adaptive_state_freeze"][
                    "research_state_hash"
                ],
                "post_freeze_adaptation": False,
                "bound_object_ids": {
                    "plan": plan.object_id,
                    "preregistration": registration.object_id,
                },
                "record_command_prefix": [
                    "xscientist",
                    "research",
                    "experiment",
                    "{RESULT_SUMMARY}",
                    "--study-phase",
                    "confirmatory",
                    "--task",
                    task_id,
                    "--plan",
                    plan.object_id,
                    "--preregistration",
                    registration.object_id,
                    "--repo",
                    ".",
                ],
            }
        )
    queue: dict[str, Any] = {
        "schema": "xscientist.confirmatory-queue.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked",
        "paper_dir": paper_relative.as_posix() or ".",
        "plan_id": plan_id,
        "plan_object_id": plan.object_id,
        "preregistration_id": locked["preregistration_id"],
        "preregistration_object_id": registration.object_id,
        "registration_hash": locked["registration_hash"],
        "frozen_head": research_vcs_head,
        "frozen_state_hash": locked["adaptive_state_freeze"]["state_hash"],
        "data_manifest_hash": locked_manifest_hash,
        "data_snapshot_id": locked_snapshot_id,
        "queue_contract_hash": locked_queue_contract.get("queue_contract_hash"),
        "tasks": queue_tasks,
    }
    queue["queue_hash"] = canonical_content_hash(queue)

    try:
        atomic_write_json(preregistration_path, locked, ensure_ascii=False)
        atomic_write_json(queue_path, queue, ensure_ascii=False)
    except BaseException as exc:
        _restore_atomic_json(preregistration_path, previous_preregistration)
        _restore_atomic_json(queue_path, previous_queue)
        raise ResearchGitError(
            "INCOMPLETE_CONFIRMATORY_CAMPAIGN: the immutable preregistration "
            f"object {registration.object_id} was committed, but its paper mirrors "
            "could not be written. Do not create a second lock; recover by "
            "materializing that exact locked object and its queue, then checkpoint "
            "only the declared campaign artifact paths."
        ) from exc
    try:
        campaign_checkpoint = create_checkpoint(
            repository_root,
            stage="preregister",
            subject="materialize locked confirmatory queue",
            status="completed",
            only_paths=[preregistration_relative, queue_relative],
        )
    except BaseException as exc:
        raise ResearchGitError(
            "INCOMPLETE_CONFIRMATORY_CAMPAIGN: the immutable preregistration "
            f"object {registration.object_id} is committed and the locked mirrors "
            "are present, but their checkpoint failed. Do not create a second "
            "lock. Recovery: run `xscientist research stage "
            f"{preregistration_relative} {queue_relative} --repo .` and then "
            "`xscientist research commit --repo . --stage preregister "
            "-m 'materialize locked confirmatory queue'`."
        ) from exc

    return {
        **recorded,
        "hypothesis_checkpoint": hypothesis_checkpoint,
        "campaign_checkpoint": campaign_checkpoint,
        "preregistration_path": str(preregistration_path),
        "queue_path": str(queue_path),
        "queue": queue,
        "validation": validation,
    }


def bind_experiment_trajectory(
    paper_dir: str,
    *,
    record_id: str,
    attempt_id: str,
    message: str | None = None,
) -> dict[str, Any]:
    """Bind one immutable registry row to its Research VCS attempt/checkpoint."""

    paper_root = Path(paper_dir).expanduser().resolve()
    status = repository_status(paper_root)
    repository = ResearchRepository(str(status["repository"]))
    preregistration = _load_bounded_project_json(
        paper_root,
        "preregistration.json",
        label="locked preregistration",
    )
    registration = _locked_registration_object(repository, preregistration)
    registration_object_id = str(registration["object_id"])
    try:
        paper_relative = paper_root.relative_to(repository.path)
    except ValueError as exc:
        raise ResearchGitError("paper directory is outside Research VCS") from exc
    allowed_pending = {
        (paper_relative / filename).as_posix()
        for filename in (
            "experiment_registry.jsonl",
            "experiment_registry.integrity.json",
            "experiment_registry.history.jsonl",
            "pipeline_manifest.json",
            "stage_standards.json",
            "process_alignment.json",
        )
    }
    current_status = repository.status()
    if current_status.get("research_stage", {}).get("paths"):
        raise ResearchGitError("trajectory binding requires an empty native stage")
    pending = {
        *list(current_status.get("staged_paths") or []),
        *list(current_status.get("tracked_changes") or []),
        *list(current_status.get("eligible_changes") or []),
    }
    # Paper-root registry artifacts can be policy-excluded in a standalone
    # Research VCS repository.  They are nevertheless explicit inputs to this
    # transition and must be committed with the binding object, never consumed
    # only from an untracked working copy.
    excluded_changes = [
        str(item) for item in current_status.get("excluded_changes") or []
    ]
    pending.update(
        path
        for path in allowed_pending
        if any(
            item == path or item.startswith(path + " (") for item in excluded_changes
        )
    )
    unexpected_pending = sorted(pending - allowed_pending)
    if unexpected_pending:
        raise ResearchGitError(
            "trajectory binding found unrelated pending paths: "
            + ", ".join(unexpected_pending)
        )
    registry_repository_path = (paper_relative / "experiment_registry.jsonl").as_posix()
    if registry_repository_path not in pending:
        raise ResearchGitError(
            "trajectory binding requires the registry row to be pending in the "
            "same Research VCS transition"
        )
    rows = _load_bounded_registry_rows(paper_root)
    normalized_record_id = _required_text(record_id, label="registry record id")
    matches = [
        row
        for row in rows
        if row.get("record_type") != "attempt_disposition"
        and str(row.get("record_id") or "").strip() == normalized_record_id
    ]
    if len(matches) != 1:
        raise ResearchGitError(
            "registry record id must identify exactly one immutable evidence row"
        )
    row = matches[0]
    attempt = repository.get(attempt_id)
    contract_errors = attempt_registry_contract_errors(
        row,
        attempt,
        registration_object_id=registration_object_id,
    )
    if contract_errors:
        raise ResearchGitError(
            "registry row and Research VCS attempt disagree: "
            + ", ".join(contract_errors)
        )
    adaptive_freeze = preregistration.get("adaptive_state_freeze")
    if not isinstance(adaptive_freeze, Mapping):
        raise ResearchGitError("locked preregistration has no adaptive state freeze")
    lineage = ResearchLifecycle(repository)._confirmatory_lineage_attestation(
        registration=registration,
        preregistration_id=registration_object_id,
        preregistration=preregistration,
        adaptive_freeze=adaptive_freeze,
        allowed_pending_paths=set(pending),
    )
    normalized_attempt_id = str(attempt.get("object_id") or "")
    if normalized_attempt_id not in set(
        lineage.get("prior_confirmatory_attempt_ids") or []
    ):
        raise ResearchGitError(
            "attempt is not in the host-attested post-freeze Research VCS lineage"
        )
    all_bindings = [
        item
        for item in repository.objects(kind="gate_decision")
        if (item.get("payload") or {}).get("protocol_kind")
        == TRAJECTORY_BINDING_PROTOCOL
    ]
    existing = [
        item
        for item in all_bindings
        if (item.get("payload") or {}).get("record_id") == normalized_record_id
        or (item.get("payload") or {}).get("attempt_id") == normalized_attempt_id
    ]
    if existing:
        raise ResearchGitError(
            "registry record or attempt already has a trajectory binding"
        )
    bindable_rows = [
        row
        for row in rows
        if row.get("record_type") != "attempt_disposition"
        and (
            str(row.get("study_phase") or "").strip().lower() == "confirmatory"
            or row.get("independent_reproduction") is True
        )
    ]
    bindable_ids = [str(row.get("record_id") or "").strip() for row in bindable_rows]
    bound_record_ids = {
        str((item.get("payload") or {}).get("record_id") or "").strip()
        for item in all_bindings
    }
    unbound_record_ids = [
        record_id for record_id in bindable_ids if record_id not in bound_record_ids
    ]
    if (
        any(not record_id for record_id in bindable_ids)
        or len(bindable_ids) != len(set(bindable_ids))
        or unbound_record_ids != [normalized_record_id]
    ):
        raise ResearchGitError(
            "trajectory binding requires exactly one new, uniquely identified "
            "registry row per checkpoint"
        )
    bound_attempt_ids = {
        str((item.get("payload") or {}).get("attempt_id") or "").strip()
        for item in all_bindings
    }
    unbound_lineage_attempt_ids = (
        set(lineage.get("prior_confirmatory_attempt_ids") or []) - bound_attempt_ids
    )
    if unbound_lineage_attempt_ids != {normalized_attempt_id}:
        raise ResearchGitError(
            "trajectory binding requires exactly one unbound attempt in the "
            "host-attested lineage"
        )
    head = str(repository.status().get("head") or "")
    origin = repository.blame(normalized_attempt_id, commit=head).get("origin") or {}
    origin_commit = str(origin.get("commit") or "")
    shown = repository.show(origin_commit)
    checkpoint = shown.get("checkpoint") or {}
    if shown.get("checkpoint_hash_valid") is not True:
        raise ResearchGitError("attempt origin has no valid Research VCS checkpoint")
    core = {
        "protocol_kind": TRAJECTORY_BINDING_PROTOCOL,
        "decision": "bind_registry_record_to_research_trajectory",
        "record_id": normalized_record_id,
        "registry_row_hash": registry_row_hash(row),
        "attempt_id": normalized_attempt_id,
        "attempt_content_hash": attempt.get("content_hash"),
        "attempt_origin_commit": origin_commit,
        "attempt_checkpoint_hash": checkpoint.get("content_hash"),
    }
    payload = {**core, "binding_hash": canonical_content_hash(core)}
    result = repository.record(
        "gate_decision",
        payload,
        state="verified",
        relations=[
            {"type": "attests", "target": normalized_attempt_id},
            {
                "type": "depends_on",
                "target": registration_object_id,
                "role": "protocol",
            },
        ],
        actor={"actor_id": "xscientist-host", "authority": "deterministic_gate"},
    )
    checkpoint = create_checkpoint(
        repository.path,
        stage="review",
        subject=message or f"bind registry record {normalized_record_id} to trajectory",
        status="verified",
        only_paths=[
            result.path.relative_to(repository.path).as_posix(),
            *sorted(pending),
        ],
    )
    return {"object": result, "related": [], "checkpoint": checkpoint}


def record_attempt_disposition(
    paper_dir: str,
    *,
    record_id: str,
    disposition: str,
    reason: str,
    retry_record_id: str | None = None,
    approved_before_unblinding: bool = False,
    negative_result_artifact: str | None = None,
    negative_result_evidence_id: str | None = None,
    recorded_by: str = "recorder:xscientist-user",
    message: str | None = None,
) -> dict[str, Any]:
    """Append an auditable decision for a failed/timed-out/cancelled attempt."""

    normalized_disposition = str(disposition or "").strip().lower()
    if normalized_disposition not in ATTEMPT_DISPOSITIONS:
        raise ResearchGitError("unsupported attempt disposition")
    normalized_reason = _required_text(reason, label="attempt disposition reason")
    normalized_record_id = _required_text(record_id, label="registry record id")
    normalized_recorder = _required_text(
        recorded_by,
        label="attempt disposition recorder",
    )
    paper_root = Path(paper_dir).expanduser().resolve()
    status = repository_status(paper_root)
    repository = ResearchRepository(str(status["repository"]))
    _ensure_direct_save_is_safe(repository, commit=True)
    preregistration = _load_bounded_project_json(
        paper_root,
        "preregistration.json",
        label="locked preregistration",
    )
    registration = _locked_registration_object(repository, preregistration)
    registration_object_id = str(registration["object_id"])
    rows = _load_bounded_registry_rows(paper_root)
    evidence_row_list = [
        row for row in rows if row.get("record_type") != "attempt_disposition"
    ]
    evidence_record_ids = [
        str(row.get("record_id") or "").strip() for row in evidence_row_list
    ]
    if any(not item for item in evidence_record_ids) or len(evidence_record_ids) != len(
        set(evidence_record_ids)
    ):
        raise ResearchGitError(
            "attempt disposition requires unique non-empty registry record ids"
        )
    evidence_rows = {
        str(row.get("record_id") or "").strip(): row for row in evidence_row_list
    }
    row = evidence_rows.get(normalized_record_id)
    if row is None or str(row.get("status") or "").strip().lower() not in {
        "failed",
        "error",
        "timeout",
        "timed_out",
        "cancelled",
        "canceled",
    }:
        raise ResearchGitError(
            "attempt disposition requires one unsuccessful registry record"
        )
    bindings = [
        item
        for item in repository.objects(kind="gate_decision")
        if (item.get("payload") or {}).get("protocol_kind")
        == TRAJECTORY_BINDING_PROTOCOL
        and (item.get("payload") or {}).get("record_id") == normalized_record_id
    ]
    if len(bindings) != 1:
        raise ResearchGitError(
            "failed registry record must have exactly one trajectory binding first"
        )
    binding_payload = bindings[0].get("payload") or {}
    attempt = repository.get(str(binding_payload.get("attempt_id") or ""))
    if (
        binding_payload.get("registry_row_hash") != registry_row_hash(row)
        or binding_payload.get("attempt_content_hash") != attempt.get("content_hash")
        or attempt_registry_contract_errors(
            row,
            attempt,
            registration_object_id=registration_object_id,
        )
    ):
        raise ResearchGitError(
            "failed registry row no longer matches its immutable trajectory binding"
        )
    existing = [
        item
        for item in repository.objects(kind="gate_decision")
        if (item.get("payload") or {}).get("protocol_kind")
        == ATTEMPT_DISPOSITION_PROTOCOL
        and (item.get("payload") or {}).get("attempt_record_id") == normalized_record_id
    ]
    if existing:
        raise ResearchGitError("attempt already has an immutable disposition")
    retry_binding: dict[str, Any] | None = None
    normalized_retry = str(retry_record_id or "").strip() or None
    if normalized_disposition == "technical_failure_retried":
        retry = evidence_rows.get(str(normalized_retry or ""))
        if (
            not retry
            or retry.get("task_id") != row.get("task_id")
            or retry.get("preregistration_id") != row.get("preregistration_id")
            or str(retry.get("status") or "").strip().lower()
            not in {"completed", "verified"}
        ):
            raise ResearchGitError(
                "technical failure disposition requires a completed same-task retry"
            )
        retry_bindings = [
            item
            for item in repository.objects(kind="gate_decision")
            if (item.get("payload") or {}).get("protocol_kind")
            == TRAJECTORY_BINDING_PROTOCOL
            and (item.get("payload") or {}).get("record_id") == normalized_retry
        ]
        if len(retry_bindings) != 1:
            raise ResearchGitError("retry record must have one trajectory binding")
        retry_binding = retry_bindings[0]
        retry_binding_payload = retry_binding.get("payload") or {}
        retry_attempt = repository.get(
            str(retry_binding_payload.get("attempt_id") or "")
        )
        if (
            retry_binding_payload.get("registry_row_hash") != registry_row_hash(retry)
            or retry_binding_payload.get("attempt_content_hash")
            != retry_attempt.get("content_hash")
            or attempt_registry_contract_errors(
                retry,
                retry_attempt,
                registration_object_id=registration_object_id,
            )
        ):
            raise ResearchGitError(
                "retry registry row no longer matches its immutable trajectory binding"
            )
    elif normalized_retry:
        raise ResearchGitError("retry record is only valid for technical failure")
    if (
        normalized_disposition == "approved_deviation"
        and approved_before_unblinding is not True
    ):
        raise ResearchGitError(
            "approved_deviation requires an explicit pre-unblinding timing "
            "assertion for audit; it remains a publication blocker"
        )
    artifact_receipt: dict[str, Any] | None = None
    negative_evidence: dict[str, Any] | None = None
    normalized_negative_evidence_id = (
        str(negative_result_evidence_id or "").strip() or None
    )
    if normalized_disposition == "terminal_negative":
        if not str(negative_result_artifact or "").strip():
            raise ResearchGitError(
                "terminal_negative requires --negative-result-artifact"
            )
        if normalized_negative_evidence_id is None:
            raise ResearchGitError(
                "terminal_negative requires --negative-result-evidence"
            )
        try:
            artifact_receipt = build_terminal_negative_artifact_receipt(
                repository,
                str(negative_result_artifact),
            )
            negative_evidence = repository.get(normalized_negative_evidence_id)
        except (ValueError, ResearchGitError) as exc:
            raise ResearchGitError(str(exc)) from exc
    elif negative_result_artifact or normalized_negative_evidence_id:
        raise ResearchGitError(
            "negative result artifact/evidence selectors are only valid for "
            "terminal_negative"
        )
    core: dict[str, Any] = {
        "protocol_kind": ATTEMPT_DISPOSITION_PROTOCOL,
        "decision": "record_attempt_disposition",
        "attempt_record_id": normalized_record_id,
        "attempt_record_hash": registry_row_hash(row),
        "attempt_id": binding_payload.get("attempt_id"),
        "attempt_content_hash": binding_payload.get("attempt_content_hash"),
        "disposition": normalized_disposition,
        "reason": normalized_reason,
        "retry_record_id": normalized_retry,
        "approved_before_unblinding": approved_before_unblinding is True,
        "negative_result_preserved": normalized_disposition == "terminal_negative",
    }
    if retry_binding is not None:
        core["retry_attempt_id"] = (retry_binding.get("payload") or {}).get(
            "attempt_id"
        )
    if artifact_receipt is not None and negative_evidence is not None:
        negative_payload = negative_evidence.get("payload") or {}
        core.update(
            {
                "negative_result_artifact": artifact_receipt,
                "negative_result_evidence_id": normalized_negative_evidence_id,
                "negative_result_evidence_hash": negative_evidence.get("content_hash"),
                "negative_result_measurement_hash": negative_payload.get(
                    "measurement_hash"
                ),
            }
        )
    relations: list[dict[str, str]] = [
        {"type": "attests", "target": str(binding_payload.get("attempt_id") or "")},
        {
            "type": "depends_on",
            "target": registration_object_id,
            "role": "protocol",
        },
    ]
    if retry_binding is not None:
        relations.append(
            {
                "type": "depends_on",
                "target": str((retry_binding.get("payload") or {}).get("attempt_id")),
                "role": "retry",
            }
        )
    if normalized_negative_evidence_id is not None:
        relations.append(
            {
                "type": "depends_on",
                "target": normalized_negative_evidence_id,
                "role": "negative_result_evidence",
            }
        )
    if normalized_disposition == "terminal_negative":
        terminal_errors = terminal_negative_contract_errors(
            repository,
            {"payload": core, "relations": relations},
            attempt,
            registry_row=row,
        )
        if terminal_errors:
            raise ResearchGitError(
                "terminal negative contract is not host-verifiable: "
                + ", ".join(terminal_errors)
            )
    payload = {**core, "disposition_hash": canonical_content_hash(core)}
    result = repository.record(
        "gate_decision",
        payload,
        state="verified",
        relations=relations,
        actor={"actor_id": normalized_recorder, "authority": "recorder"},
    )
    return _finish(
        repository,
        result,
        stage="review",
        subject=message or f"record disposition for {normalized_record_id}",
        status="verified",
        commit=True,
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
        data_attestation = validate_empirical_data_manifest(repo)
        if data_attestation.get("ok") is True:
            draft["data_policy"]["data_manifest_hash"] = data_attestation.get(
                "manifest_hash"
            )
            draft["data_policy"]["data_snapshot_id"] = data_attestation.get(
                "snapshot_id"
            )
        elif data_attestation.get("errors") != ["data_manifest_not_found"]:
            raise ResearchIntegrityError(
                "cannot lock preregistration to an invalid empirical data "
                "contract: " + ", ".join(data_attestation.get("errors") or ["unknown"])
            )
        repository_state = repository.status()
        if repository_state.get("worktree_clean") is not True:
            raise ResearchIntegrityError(
                "confirmatory freeze requires a clean Research VCS worktree"
            )
        research_vcs_head = _required_text(
            str(repository_state.get("head") or ""),
            label="Research VCS checkpoint before confirmatory freeze",
        )
        # Derive the freeze from the exact preregistration contract that will
        # be locked. This shared host-owned derivation is also recomputed by
        # the top-venue publication gate, so arbitrary digest-shaped values
        # cannot masquerade as committed code, memory, or evaluator state.
        draft["data_policy"]["split_hashes"] = {task_id: split_hash}
        derived_state_hashes = derive_adaptive_state_hashes(
            draft,
            research_vcs_head=research_vcs_head,
        )
        locked = lock_preregistration(
            draft,
            split_hashes={task_id: split_hash},
            registered_by=registered_by,
            freeze_inputs={
                "research_vcs_head": research_vcs_head,
                **derived_state_hashes,
            },
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
        actor={"actor_id": registered_by, "authority": "recorder"},
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
    "confirm_paper_research",
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
