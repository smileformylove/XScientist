"""Privacy-safe, append-only indexes for one agent research run.

The run index is deliberately smaller than an action or rollout receipt.  It
records the lifecycle of a run (plan, inspect, execute, repair, ...), while
large inputs, model messages and tool output stay in their existing stores.
Each step binds only hashes and Research Object IDs, so an interrupted video
inspection or experiment can be resumed from a deterministic, attention-first
summary without copying private context into the graph.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.research_vcs import (
    BUILTIN_RESEARCH_PROFILES,
    RESEARCH_RELATION_TYPES,
)
from ai_scientist.protocol.schemas import load_schema

from .research_git import ResearchGitError
from .research_vcs import ResearchRepository

RUN_PROTOCOL = "xscientist.research-run.v1"
RUN_SUMMARY_PROTOCOL = "xscientist.research-run-summary.v1"
RUN_STEP_TYPES = ("plan", "delegate", "execute", "inspect", "repair", "judge", "stop")
RUN_STEP_STATUSES = (
    "proposed",
    "running",
    "waiting",
    "ready",
    "blocked",
    "completed",
    "failed",
    "timed_out",
    "cancelled",
    "interrupted",
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OBJECT_RE = re.compile(r"^rso-[0-9a-f]{16}$")
_MAX_RUNS = 512
_RUN_SEMANTIC_PROFILE = BUILTIN_RESEARCH_PROFILES.get(
    "https://xscientist.io/profiles/research-run/v1"
)
if _RUN_SEMANTIC_PROFILE is None:
    # Keep this extension kind usable with older protocol runtimes that do not
    # yet ship the builtin profile.  New runtimes resolve to the canonical
    # builtin descriptor above.
    _RUN_PROFILE_CORE = {
        "schema": "xscientist.research-semantic-profile.v1",
        "uri": "https://xscientist.io/profiles/research-run/v1",
        "version": "1.0.0",
        "kinds": ["research_run"],
        "relations": list(RESEARCH_RELATION_TYPES),
    }
    _RUN_SEMANTIC_PROFILE = {
        **_RUN_PROFILE_CORE,
        "schema_digest": canonical_content_hash(_RUN_PROFILE_CORE),
    }


def _name(value: Any, *, label: str) -> str:
    result = str(value or "").strip()
    if not _NAME_RE.fullmatch(result):
        raise ResearchGitError(f"{label} is invalid")
    return result


def _hash(value: Any, *, label: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, str) and _HASH_RE.fullmatch(value):
        return value
    try:
        return canonical_content_hash(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ResearchGitError(f"{label} cannot be canonically hashed") from exc


def _refs(
    repository: ResearchRepository,
    values: Sequence[str],
    *,
    label: str,
) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ResearchGitError(f"{label} must be a sequence of Research Object IDs")
    if len(values) > 64:
        raise ResearchGitError(f"{label} may contain at most 64 references")
    resolved: list[str] = []
    for value in values:
        try:
            resolved_id = repository.resolve(str(value))
        except (ResearchGitError, TypeError, ValueError) as exc:
            raise ResearchGitError(f"{label} contains an unknown Research Object") from exc
        if not _OBJECT_RE.fullmatch(resolved_id):
            raise ResearchGitError(f"{label} contains an invalid Research Object ID")
        resolved.append(resolved_id)
    return sorted(set(resolved))


def _validate_payload(payload: Mapping[str, Any]) -> None:
    try:
        validate_json(dict(payload), load_schema("research_run"))
    except ValidationError as exc:
        raise ResearchGitError(f"research run step is invalid: {exc.message}") from exc


def _object_state(status: str) -> str:
    if status in {"completed", "failed", "timed_out", "cancelled", "interrupted"}:
        return status
    return "running"


def _step_payload(
    *,
    run_id: str,
    sequence: int,
    step_id: str,
    step_type: str,
    status: str,
    input_hash: str,
    output_hash: str | None,
    action_ref: str | None,
    context_refs: Sequence[str],
    requires_human: bool,
    next_action: str | None,
    error_code: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_kind": RUN_PROTOCOL,
        "run_id": run_id,
        "sequence": sequence,
        "step_id": step_id,
        "step_type": step_type,
        "status": status,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "action_ref": action_ref,
        # These are context-only by contract.  They do not grant permission
        # to mutate or conclude anything in the referenced objects.
        "context_refs": list(context_refs),
        "requires_human": requires_human,
        "next_action": next_action,
    }
    if error_code is not None:
        payload["error_code"] = _name(error_code, label="error_code")
    payload["step_hash"] = canonical_content_hash(payload)
    return payload


class ResearchRunLedger:
    """Append-only run steps bound to one :class:`ResearchRepository`."""

    def __init__(self, repository: ResearchRepository | str):
        self.repository = (
            repository
            if isinstance(repository, ResearchRepository)
            else ResearchRepository(repository)
        )

    def history(self, run_id: str) -> list[dict[str, Any]]:
        normalized = _name(run_id, label="run_id")
        rows = [
            item
            for item in self.repository.objects(kind="research_run")
            if (item.get("payload") or {}).get("protocol_kind") == RUN_PROTOCOL
            and (item.get("payload") or {}).get("run_id") == normalized
        ]
        rows.sort(key=lambda item: int((item.get("payload") or {}).get("sequence", -1)))
        for expected, item in enumerate(rows):
            payload = item.get("payload") or {}
            if payload.get("sequence") != expected:
                raise ResearchGitError("research run sequence is not contiguous")
            _validate_payload(payload)
        return rows

    def append(
        self,
        run_id: str,
        step_id: str,
        step_type: str,
        status: str,
        input_value: Any,
        *,
        output_value: Any = None,
        action_ref: str | None = None,
        context_refs: Sequence[str] = (),
        requires_human: bool = False,
        next_action: str | None = None,
        error_code: str | None = None,
        sequence: int | None = None,
        actor_id: str = "agent:xscientist",
    ) -> dict[str, Any]:
        """Append one run step, returning the existing object for an idempotent retry."""

        normalized_run = _name(run_id, label="run_id")
        normalized_step = _name(step_id, label="step_id")
        normalized_type = str(step_type or "").strip().lower()
        if normalized_type not in RUN_STEP_TYPES:
            raise ResearchGitError("step_type is invalid")
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in RUN_STEP_STATUSES:
            raise ResearchGitError("status is invalid")
        if type(requires_human) is not bool:
            raise ResearchGitError("requires_human must be a boolean")
        if next_action is not None:
            next_action = _name(next_action, label="next_action")
        normalized_actor = _name(actor_id, label="actor_id")
        rows = self.history(normalized_run)
        expected_sequence = len(rows)
        actual_sequence = expected_sequence if sequence is None else sequence
        if type(actual_sequence) is not int or actual_sequence < 0:
            raise ResearchGitError("sequence must be a non-negative integer")
        input_hash = _hash(input_value, label="input")
        output_hash = _hash(output_value, label="output", allow_none=True)
        action = None
        if action_ref is not None:
            try:
                action = self.repository.resolve(action_ref, kind="research_action")
            except (ResearchGitError, TypeError, ValueError) as exc:
                raise ResearchGitError("action_ref must resolve to a research_action") from exc
        contexts = _refs(self.repository, context_refs, label="context_refs")
        payload = _step_payload(
            run_id=normalized_run,
            sequence=actual_sequence,
            step_id=normalized_step,
            step_type=normalized_type,
            status=normalized_status,
            input_hash=str(input_hash),
            output_hash=output_hash,
            action_ref=action,
            context_refs=contexts,
            requires_human=requires_human,
            next_action=next_action,
            error_code=error_code,
        )
        if actual_sequence != expected_sequence:
            if 0 <= actual_sequence < expected_sequence:
                existing = rows[actual_sequence]
                if (existing.get("payload") or {}).get("step_hash") == payload["step_hash"]:
                    return existing
            raise ResearchGitError("research run sequence must advance by one")
        relations: list[dict[str, str]] = []
        if rows:
            relations.append({"type": "supersedes", "target": rows[-1]["object_id"]})
        if action is not None:
            relations.append({"type": "consumes", "target": action, "role": "action"})
        relations.extend(
            {"type": "consumes", "target": ref, "role": "context_only"}
            for ref in contexts
        )
        result = self.repository.record(
            "research_run",
            payload,
            state=_object_state(normalized_status),
            relations=relations,
            actor={"actor_id": normalized_actor, "authority": "research_agent"},
            semantic_profile=_RUN_SEMANTIC_PROFILE,
        )
        return self.repository.get(result.object_id)


def list_research_run(repo: str | ResearchRepository, run_id: str) -> list[dict[str, Any]]:
    return ResearchRunLedger(repo).history(run_id)


def summarize_research_run(
    repo: str | ResearchRepository,
    run_id: str,
) -> dict[str, Any]:
    """Return an attention-first, payload-free view of one run."""

    rows = list_research_run(repo, run_id)
    if not rows:
        raise ResearchGitError(f"no research run found for {run_id}")
    latest_by_step: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = row["payload"]
        latest_by_step[str(payload["step_id"])] = row
    latest = rows[-1]["payload"]
    attention = []
    for row in latest_by_step.values():
        payload = row["payload"]
        if payload["requires_human"] or payload["status"] in {"waiting", "blocked"}:
            attention.append(
                {
                    "sequence": payload["sequence"],
                    "step_id": payload["step_id"],
                    "step_type": payload["step_type"],
                    "status": payload["status"],
                    "requires_human": payload["requires_human"],
                    "next_action": payload["next_action"],
                }
            )
    attention.sort(key=lambda item: (not item["requires_human"], item["sequence"]))
    context_refs = sorted(
        {
            ref
            for row in rows
            for ref in row["payload"].get("context_refs") or []
        }
    )
    action_refs = sorted(
        {
            str(ref)
            for row in rows
            if (ref := row["payload"].get("action_ref"))
        }
    )
    core = {
        "protocol_kind": RUN_SUMMARY_PROTOCOL,
        "run_id": latest["run_id"],
        "status": latest["status"],
        "sequence": latest["sequence"],
        "current_step": {
            "step_id": latest["step_id"],
            "step_type": latest["step_type"],
            "status": latest["status"],
        },
        "step_count": len(rows),
        "attention_required": bool(attention),
        "attention": attention,
        "context_refs": context_refs,
        "action_refs": action_refs,
        "raw_content_included": False,
    }
    return {**core, "summary_hash": canonical_content_hash(core)}


def list_research_runs(repo: str | ResearchRepository) -> list[dict[str, Any]]:
    repository = repo if isinstance(repo, ResearchRepository) else ResearchRepository(repo)
    run_ids = sorted(
        {
            str((item.get("payload") or {}).get("run_id"))
            for item in repository.objects(kind="research_run")
            if (item.get("payload") or {}).get("protocol_kind") == RUN_PROTOCOL
        }
    )
    if len(run_ids) > _MAX_RUNS:
        raise ResearchGitError("research run index exceeds its safety limit")
    return [summarize_research_run(repository, run_id) for run_id in run_ids]


__all__ = [
    "RUN_PROTOCOL",
    "RUN_STEP_STATUSES",
    "RUN_STEP_TYPES",
    "ResearchRunLedger",
    "list_research_run",
    "list_research_runs",
    "summarize_research_run",
]
