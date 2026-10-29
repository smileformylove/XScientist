"""Durable, idempotent action receipts for research agents.

The action ledger is deliberately metadata-only.  Prompts, tool output and
local paths remain outside the receipt; callers bind those values through
content hashes and Research Object references.  Every transition is an
immutable ``research_action`` object, so an interrupted run can be resumed
without rewriting its history.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ai_scientist.protocol.canonical_json import canonical_content_hash

from .research_git import ResearchGitError
from .research_vcs import ResearchRepository

ACTION_PROTOCOL = "xscientist.research-action.v1"
ACTION_STATUSES = (
    "proposed",
    "running",
    "completed",
    "failed",
    "timed_out",
    "cancelled",
    "interrupted",
)
TERMINAL_ACTION_STATUSES = frozenset(
    {"completed", "failed", "timed_out", "cancelled"}
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _name(value: Any, *, label: str) -> str:
    result = str(value or "").strip()
    if not _NAME_RE.fullmatch(result):
        raise ResearchGitError(f"{label} is invalid")
    return result


def _hash(value: Any, *, label: str) -> str:
    if isinstance(value, str) and _HASH_RE.fullmatch(value):
        return value
    try:
        return canonical_content_hash(value)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ResearchGitError(f"{label} cannot be canonically hashed") from exc


def _refs(repository: ResearchRepository, values: Sequence[str], *, label: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ResearchGitError(f"{label} must be a sequence of Research Object IDs")
    if len(values) > 64:
        raise ResearchGitError(f"{label} may contain at most 64 references")
    resolved: list[str] = []
    for value in values:
        try:
            resolved.append(repository.resolve(str(value)))
        except (ResearchGitError, TypeError, ValueError) as exc:
            raise ResearchGitError(f"{label} contains an unknown Research Object") from exc
    return sorted(set(resolved))


def _budget(value: Any, *, label: str, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchGitError(f"{label} must be a finite non-negative number")
    parsed = float(value)
    if parsed < 0 or parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ResearchGitError(f"{label} must be a finite non-negative number")
    return round(parsed, 6)


def _action_payload(
    *,
    action_id: str,
    action_type: str,
    status: str,
    attempt: int,
    sequence: int,
    input_hash: str,
    output_hash: str | None = None,
    parent_action_id: str | None = None,
    retry_of: str | None = None,
    recovery_of: str | None = None,
    consumes: Sequence[str] = (),
    produces: Sequence[str] = (),
    budget_seconds: float | None = None,
    consumed_seconds: float | None = None,
    cumulative_consumed_seconds: float | None = None,
    remaining_seconds: float | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_kind": ACTION_PROTOCOL,
        "action_id": action_id,
        "action_type": action_type,
        "status": status,
        "attempt": attempt,
        "sequence": sequence,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "parent_action_id": parent_action_id,
        "retry_of": retry_of,
        "recovery_of": recovery_of,
        "consumes": list(consumes),
        "produces": list(produces),
        "budget_seconds": budget_seconds,
        "consumed_seconds": consumed_seconds,
        "cumulative_consumed_seconds": cumulative_consumed_seconds,
        "remaining_seconds": remaining_seconds,
    }
    if error_code is not None:
        payload["error_code"] = _name(error_code, label="error_code")
    payload["action_hash"] = canonical_content_hash(payload)
    return payload


class ResearchActionLedger:
    """Append-only action state for one :class:`ResearchRepository`."""

    def __init__(self, repository: ResearchRepository | str):
        self.repository = (
            repository
            if isinstance(repository, ResearchRepository)
            else ResearchRepository(repository)
        )

    def history(self, action_id: str) -> list[dict[str, Any]]:
        normalized = _name(action_id, label="action_id")
        rows = [
            item
            for item in self.repository.objects(kind="research_action")
            if (item.get("payload") or {}).get("action_id") == normalized
        ]
        return sorted(rows, key=lambda item: int(item["payload"].get("sequence", -1)))

    def current(self, action_id: str) -> dict[str, Any] | None:
        rows = self.history(action_id)
        if not rows:
            return None
        latest_sequence = max(int(item["payload"].get("sequence", -1)) for item in rows)
        latest = [
            item for item in rows
            if int(item["payload"].get("sequence", -1)) == latest_sequence
        ]
        if len(latest) != 1:
            raise ResearchGitError("research action ledger has duplicate sequence")
        return latest[0]

    def _next_sequence(self, action_id: str) -> int:
        rows = self.history(action_id)
        return max((int(item["payload"].get("sequence", -1)) for item in rows), default=-1) + 1

    def _record(
        self,
        payload: Mapping[str, Any],
        *,
        state: str,
        relations: Sequence[Mapping[str, str]],
        actor_id: str,
    ) -> dict[str, Any]:
        result = self.repository.record(
            "research_action",
            payload,
            state=state,
            relations=relations,
            actor={"actor_id": _name(actor_id, label="actor_id"), "authority": "research_agent"},
        )
        return self.repository.get(result.object_id)

    def begin(
        self,
        action_id: str,
        action_type: str,
        input_value: Any,
        *,
        parent_action_id: str | None = None,
        consumes: Sequence[str] = (),
        budget_seconds: int | float | None = None,
        retry: bool = False,
        recovery: bool = False,
        actor_id: str = "agent:xscientist",
    ) -> dict[str, Any]:
        """Start an action, returning an existing state for idempotent retries."""

        normalized_id = _name(action_id, label="action_id")
        normalized_type = _name(action_type, label="action_type")
        input_hash = _hash(input_value, label="input")
        current = self.current(normalized_id)
        if current is not None:
            current_payload = current["payload"]
            current_status = str(current_payload.get("status") or "")
            if current_payload.get("action_type") != normalized_type:
                raise ResearchGitError("action_id is already bound to another action_type")
            if current_payload.get("input_hash") != input_hash:
                raise ResearchGitError("action retry input_hash does not match the original")
            if current_status in {"proposed", "running"} and not retry and not recovery:
                return current
            if current_status in TERMINAL_ACTION_STATUSES and not retry and not recovery:
                return current
            if current_status == "interrupted" and not (retry or recovery):
                raise ResearchGitError("interrupted action requires retry or recovery")
            attempt = int(current_payload.get("attempt", 0)) + 1
            parent = str(current["object_id"])
            retry_of = normalized_id if retry else None
            recovery_of = normalized_id if recovery else None
        else:
            attempt = 0
            parent = parent_action_id
            retry_of = None
            recovery_of = None
        if parent_action_id is not None and current is None:
            parent_action_id = _name(parent_action_id, label="parent_action_id")
            parent_receipt = self.current(parent_action_id)
            if parent_receipt is None:
                raise ResearchGitError("parent_action_id does not reference an action")
        elif parent_action_id is not None:
            parent_action_id = _name(parent_action_id, label="parent_action_id")
        if parent_action_id == normalized_id:
            raise ResearchGitError("action cannot be its own parent")
        consume_ids = _refs(self.repository, consumes, label="consumes")
        declared_budget = _budget(budget_seconds, label="budget_seconds")
        previous_cumulative = None
        if current is not None:
            previous_cumulative = current["payload"].get("cumulative_consumed_seconds")
            if previous_cumulative is None:
                previous_cumulative = current["payload"].get("consumed_seconds")
            if declared_budget is None:
                declared_budget = current["payload"].get("budget_seconds")
            elif current["payload"].get("budget_seconds") not in (None, declared_budget):
                raise ResearchGitError("action retry budget_seconds does not match the original")
        remaining = (
            round(declared_budget - float(previous_cumulative or 0), 6)
            if declared_budget is not None and previous_cumulative is not None
            else declared_budget
        )
        if remaining is not None and remaining < 0:
            raise ResearchGitError("action budget has already been exhausted")
        payload = _action_payload(
            action_id=normalized_id,
            action_type=normalized_type,
            status="running",
            attempt=attempt,
            sequence=self._next_sequence(normalized_id),
            input_hash=input_hash,
            parent_action_id=parent,
            retry_of=retry_of,
            recovery_of=recovery_of,
            consumes=consume_ids,
            budget_seconds=declared_budget,
            cumulative_consumed_seconds=previous_cumulative,
            remaining_seconds=remaining,
        )
        relations = [{"type": "consumes", "target": value} for value in consume_ids]
        if current is not None:
            relations.append({"type": "supersedes", "target": current["object_id"]})
        if parent_action_id is not None and current is None:
            relations.append({"type": "depends_on", "target": parent_receipt["object_id"]})
        return self._record(
            payload,
            state="running",
            relations=relations,
            actor_id=actor_id,
        )

    def finish(
        self,
        action_id: str,
        status: str,
        output_value: Any = None,
        *,
        produces: Sequence[str] = (),
        consumes: Sequence[str] = (),
        consumed_seconds: int | float | None = None,
        error_code: str | None = None,
        actor_id: str = "agent:xscientist",
    ) -> dict[str, Any]:
        """Append a terminal receipt; repeated identical completion is idempotent."""

        normalized_status = str(status or "").strip().lower()
        if normalized_status not in TERMINAL_ACTION_STATUSES | {"interrupted"}:
            raise ResearchGitError("finish status must be terminal or interrupted")
        current = self.current(action_id)
        if current is None:
            raise ResearchGitError("cannot finish an unknown action")
        current_payload = current["payload"]
        if str(current_payload.get("status")) in TERMINAL_ACTION_STATUSES:
            output_hash = (
                _hash(output_value, label="output") if output_value is not None else None
            )
            if normalized_status == current_payload.get("status") and (
                output_hash is None or output_hash == current_payload.get("output_hash")
            ):
                return current
            raise ResearchGitError("action is already terminal")
        produced_ids = _refs(self.repository, produces, label="produces")
        consumed_ids = _refs(self.repository, consumes, label="consumes")
        output_hash = _hash(output_value, label="output") if output_value is not None else None
        if normalized_status == "completed" and output_hash is None:
            raise ResearchGitError("completed action requires an output")
        payload = _action_payload(
            action_id=str(current_payload["action_id"]),
            action_type=str(current_payload["action_type"]),
            status=normalized_status,
            attempt=int(current_payload["attempt"]),
            sequence=self._next_sequence(action_id),
            input_hash=str(current_payload["input_hash"]),
            output_hash=output_hash,
            parent_action_id=current_payload.get("parent_action_id"),
            retry_of=current_payload.get("retry_of"),
            recovery_of=current_payload.get("recovery_of"),
            consumes=sorted(set(current_payload.get("consumes") or []) | set(consumed_ids)),
            produces=produced_ids,
            budget_seconds=current_payload.get("budget_seconds"),
            consumed_seconds=_budget(consumed_seconds, label="consumed_seconds"),
            cumulative_consumed_seconds=(
                round(
                    float(current_payload.get("cumulative_consumed_seconds") or current_payload.get("consumed_seconds") or 0)
                    + float(_budget(consumed_seconds, label="consumed_seconds") or 0),
                    6,
                )
                if _budget(consumed_seconds, label="consumed_seconds") is not None
                else current_payload.get("cumulative_consumed_seconds")
            ),
            remaining_seconds=(
                round(
                    float(current_payload["budget_seconds"])
                    - float(current_payload.get("cumulative_consumed_seconds") or current_payload.get("consumed_seconds") or 0)
                    - float(_budget(consumed_seconds, label="consumed_seconds") or 0),
                    6,
                )
                if current_payload.get("budget_seconds") is not None
                and _budget(consumed_seconds, label="consumed_seconds") is not None
                else current_payload.get("budget_seconds")
            ),
            error_code=error_code,
        )
        if (
            payload.get("remaining_seconds") is not None
            and float(payload["remaining_seconds"]) < 0
        ):
            raise ResearchGitError("action budget exceeded")
        relations = [
            {"type": "supersedes", "target": current["object_id"]},
            *({"type": "consumes", "target": value} for value in payload["consumes"]),
            *({"type": "produces", "target": value} for value in produced_ids),
        ]
        return self._record(
            payload,
            state=normalized_status,
            relations=relations,
            actor_id=actor_id,
        )

    def recover(self, action_id: str, *, actor_id: str = "agent:xscientist") -> dict[str, Any]:
        current = self.current(action_id)
        if current is None or current["payload"].get("status") != "interrupted":
            raise ResearchGitError("only an interrupted action can be recovered")
        return self.begin(
            action_id,
            str(current["payload"]["action_type"]),
            str(current["payload"]["input_hash"]),
            retry=True,
            recovery=True,
            actor_id=actor_id,
        )


__all__ = ["ACTION_PROTOCOL", "ResearchActionLedger"]
