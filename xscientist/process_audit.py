"""Bounded, artifact-backed git-like trajectory summaries.

The summary is deliberately a process *audit*, not a transcript export.  It
shows checkpoint/branch topology, typed intermediate artifacts, and structured
decision signals while never returning prompts, completions, task gold fields,
or hidden chain-of-thought.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ai_scientist.protocol.research_vcs import (
    ResearchObjectError,
    RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
    validate_research_object,
)

from .research_git import (
    ResearchGitError,
    list_research_branches,
    list_research_objects,
    repository_status,
    research_log,
    show_checkpoint,
)

PROCESS_SCHEMA = "xscientist.process-audit.v1"
_MAX_COMMITS = 32
_MAX_ARTIFACTS = 96
_MAX_DECISIONS = 32
_MAX_TEXT = 160
_HARD_MAX_COMMITS = 128
_HARD_MAX_ARTIFACTS = 512
_HARD_MAX_DECISIONS = 128
_HARD_MAX_BRANCHES = 64
_DEFAULT_MAX_BRANCHES = 32
_SAFE_TASK_FILTERS = frozenset({"all", "open-ended", "optimization"})

_STAGE_LABELS = frozenset({"A", "B", "C", "D", "E", "F"})
_FAILED_STATES = frozenset({"failed", "rejected", "blocked", "timed_out", "cancelled"})
_COMPLETED_STATES = frozenset({"completed", "verified", "accepted", "promoted"})
_SAFE_DECISIONS = frozenset(
    {
        "allow",
        "allowed",
        "accept",
        "accepted",
        "block",
        "blocked",
        "deny",
        "denied",
        "hold",
        "held",
        "pause",
        "paused",
        "promote",
        "promoted",
        "reject",
        "rejected",
        "stop",
        "stopped",
    }
)
_SAFE_STAGES = frozenset(
    {
        "init",
        "ideation",
        "planning",
        "preregister",
        "experiment",
        "evidence",
        "review",
        "claim",
        "paper",
        "evolve",
        "merge",
        "release",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    }
)
_SAFE_STATES = frozenset(
    {
        "draft",
        "locked",
        "running",
        "completed",
        "failed",
        "timed_out",
        "cancelled",
        "rejected",
        "superseded",
        "verified",
        "promoted",
        "blocked",
    }
)
_SAFE_KINDS = frozenset(str(item) for item in RESEARCH_OBJECT_KINDS)
_SAFE_RELATIONS = frozenset(str(item) for item in RESEARCH_RELATION_TYPES)
_SAFE_POLICIES = frozenset(
    {"milestone", "manual", "automatic", "strict", "none", "unknown"}
)
_REPAIR_RELATIONS = frozenset(
    {"addresses", "corrects", "repaired_by", "repairs", "resolves", "retests"}
)

_KIND_STAGE = {
    "question": "A",
    "research_goal": "A",
    "hypothesis": "A",
    "research_plan": "A",
    "search_plan": "B",
    "search_receipt": "B",
    "source_snapshot": "B",
    "source_update": "B",
    "passage_evidence": "B",
    "evidence_synthesis": "B",
    "protocol": "C",
    "dataset": "C",
    "experiment_design": "C",
    "experiment_attempt": "C",
    "tool_evidence": "C",
    "effect_estimate": "D",
    "evidence": "D",
    "warrant": "D",
    "inference": "D",
    "claim": "D",
    "manuscript": "E",
    "report": "E",
    "paper": "E",
    "review": "F",
    "research_review": "F",
    "agent_evaluation": "F",
    "challenge": "F",
    "gate_decision": "F",
    "reproduction": "F",
}

_DECISION_KINDS = {
    "context_snapshot",
    "inference",
    "claim",
    "review",
    "research_review",
    "gate_decision",
    "agent_evaluation",
}


def _short_hash(value: Any) -> str | None:
    """Return a short identifier without echoing arbitrary input text.

    Research VCS identifiers have stable, typed formats.  Anything outside
    those formats is treated as untrusted metadata and replaced by a digest;
    otherwise a crafted object/checkpoint ID could smuggle task text into a
    shareable process report.
    """

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    if re.fullmatch(r"sha256:[0-9a-fA-F]{16,}", text):
        return text[:23]
    if re.fullmatch(r"rso-[0-9a-fA-F]{6,32}", text):
        return text[:12]
    if re.fullmatch(r"rcp-[0-9a-fA-F]{6,32}", text):
        return text[:12]
    if re.fullmatch(r"[0-9a-fA-F]{16,64}", text):
        return text[:12]
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _short_commit(value: Any) -> str | None:
    """Keep Git's short commit token when it is already a hash.

    ``research_log`` returns Git's 12-character abbreviated hash.  Passing it
    through ``_short_hash`` used to digest that value again because the
    generic helper intentionally required a longer token.  The extra digest
    was safe, but it made a process timeline harder to join to ``git log``.
    Only hexadecimal commit tokens are preserved; arbitrary subjects/IDs still
    take the opaque digest path.
    """

    text = " ".join(str(value or "").split()).strip()
    if re.fullmatch(r"[0-9a-fA-F]{7,64}", text):
        return text[:12]
    return _short_hash(value)


def _safe_token(value: Any, *, default: str = "unknown", limit: int = 64) -> str:
    """Keep protocol tokens bounded without exposing arbitrary payload text."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return default
    text = re.sub(r"[^A-Za-z0-9_.:/-]+", "_", text)
    return text[:limit] or default


def _safe_task_filter(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _SAFE_TASK_FILTERS else "custom"


def _safe_manifest_digest(value: Any) -> str | None:
    """Keep a real digest (or test placeholder) and hash everything else."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    if text == "sha256:manifest":
        # Retain the stable fixture token used by callers/tests; it contains
        # no task text and is intentionally not treated as a real hash.
        return text
    if re.fullmatch(r"[0-9a-fA-F]{64}", text):
        return text.lower()
    if re.fullmatch(r"sha256:[0-9a-fA-F]{16,64}", text):
        return text[:71].lower()
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _manifest_is_usable(value: Any) -> bool:
    text = " ".join(str(value or "").split()).strip()
    return bool(
        text == "sha256:manifest"
        or re.fullmatch(r"[0-9a-fA-F]{64}", text)
        or re.fullmatch(r"sha256:[0-9a-fA-F]{16,64}", text)
    )


def _safe_ref(value: Any) -> str:
    """Return a portable branch/ref token, never an absolute or traversal path."""

    text = _safe_token(value, default="unknown", limit=80)
    if text.startswith("/") or ".." in text.split("/"):
        return "[redacted-ref]"
    return text


def _ref_digest(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _safe_subject(stage: Any) -> str:
    """Subjects are intentionally reduced to a stage category.

    A Git subject is user-controlled free text and can contain a task answer,
    prompt fragment, credential, or local path.  The process view retains the
    stage signal while refusing to echo that text.
    """

    return f"checkpoint:{_safe_token(stage, default='unknown', limit=32)}"


def _safe_text(value: Any) -> str:
    """Return a bounded redaction marker, never the supplied free text.

    This helper remains for compatibility with callers importing the private
    function; process reports should use structured booleans/enums instead.
    """

    if value in (None, ""):
        return ""
    return "[redacted]"


def _safe_decision(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _SAFE_DECISIONS else "observed"


def _safe_stage(value: Any) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized in _SAFE_STAGES else "other"


def _safe_state(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _SAFE_STATES else "other"


def _safe_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    return text[:40]


def _safe_kind(value: Any) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized in _SAFE_KINDS else "extension"


def _safe_policy(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _SAFE_POLICIES else "unknown"


def _bounded_limit(value: int | None, default: int, hard: int) -> tuple[int, bool]:
    requested = default if value is None else int(value)
    if requested <= 0:
        raise ValueError("process audit limits must be greater than zero")
    return min(requested, hard), requested > hard


def _bounded_object_scan(root: Path, limit: int) -> tuple[
    list[dict[str, Any]],
    int,
    bool,
    list[str],
    Counter[str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Retain at most ``limit`` object payloads while counting source headers.

    Directory enumeration is metadata-only.  For a small store we retain the
    existing validator path; for a large store we sample round-robin by kind so
    one giant extension cannot hide every other lifecycle stage.  Omitted
    payloads are never treated as evidence of absence.
    """

    object_root = root / ".xscientist" / "objects"
    paths = sorted(
        path
        for path in object_root.glob("*/*.json")
        if path.is_file() and not path.is_symlink()
    )
    source_count = len(paths)
    source_kind_counts: Counter[str] = Counter(
        _safe_kind(path.parent.name) for path in paths
    )

    def new_source_stats() -> dict[str, Any]:
        return {
            "valid_object_count": 0,
            "valid_kind_counts": Counter(),
            "state_counts": Counter(),
            "stage_counts": Counter(),
            "attempt_metrics": {
                "attempt_count": 0,
                "failed_attempts": 0,
                "completed_attempts": 0,
            },
            "failed_or_blocked_count": 0,
            "recovery_failed_ids": set(),
            "recovery_links": [],
        }

    def accumulate_source_stats(stats: dict[str, Any], item: Mapping[str, Any]) -> None:
        """Accumulate scalar headers without retaining the full object payload."""

        stats["valid_object_count"] += 1
        kind = _safe_kind(item.get("kind"))
        stats["valid_kind_counts"][kind] += 1
        state = _safe_state(item.get("state"))
        stats["state_counts"][state] += 1
        stage = _KIND_STAGE.get(str(item.get("kind") or ""), "X")
        stats["stage_counts"][stage] += 1

        raw_state = str(item.get("state") or "").lower()
        payload = (
            item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
        )
        payload_state = str(payload.get("status") or "").lower()
        effective_state = payload_state or raw_state
        if str(item.get("kind") or "") == "experiment_attempt":
            stats["attempt_metrics"]["attempt_count"] += 1
            if effective_state in _FAILED_STATES:
                stats["attempt_metrics"]["failed_attempts"] += 1
            if effective_state in _COMPLETED_STATES:
                stats["attempt_metrics"]["completed_attempts"] += 1

        failure = raw_state in _FAILED_STATES or bool(_failure_codes(item))
        if payload_state in _FAILED_STATES:
            failure = True
        if failure:
            stats["failed_or_blocked_count"] += 1

        object_id = str(item.get("object_id") or "")
        if effective_state in _FAILED_STATES:
            # Keep only identifiers needed for an internal relation join; they
            # are never returned in the report.  The cap prevents a malformed
            # store from turning the audit into an unbounded index.
            if len(stats["recovery_failed_ids"]) < _HARD_MAX_ARTIFACTS * 1024:
                stats["recovery_failed_ids"].add(object_id)
        if effective_state in _COMPLETED_STATES:
            targets = {
                str(relation.get("target") or "")
                for relation in item.get("relations") or []
                if isinstance(relation, Mapping)
                and str(relation.get("type") or "") in _REPAIR_RELATIONS
            }
            if targets and len(stats["recovery_links"]) < _HARD_MAX_ARTIFACTS * 1024:
                stats["recovery_links"].append(targets)

    source_stats = new_source_stats()
    if not paths:
        # Keep a small compatibility seam for callers/tests that provide a
        # repository adapter without materializing an object directory.
        try:
            rows = list_research_objects(root)
        except (OSError, ValueError, ResearchGitError) as exc:
            return (
                [],
                0,
                False,
                [type(exc).__name__],
                source_kind_counts,
                [],
                source_stats,
            )
        source_count = len(rows)
        source_kind_counts = Counter(_safe_kind(item.get("kind")) for item in rows)
        decision_rows = [
            item for item in rows if str(item.get("kind") or "") in _DECISION_KINDS
        ][-_HARD_MAX_DECISIONS:]
        for item in rows:
            accumulate_source_stats(source_stats, item)
        source_stats["recovery_candidates"] = sum(
            bool(targets.intersection(source_stats["recovery_failed_ids"]))
            for targets in source_stats["recovery_links"]
        )
        source_stats.pop("recovery_failed_ids", None)
        source_stats.pop("recovery_links", None)
        return (
            rows[-limit:] if source_count > limit else rows,
            source_count,
            source_count > limit,
            [],
            source_kind_counts,
            decision_rows,
            source_stats,
        )

    truncated = source_count > limit
    selected_paths = paths
    if truncated:
        buckets: dict[str, list[Path]] = {}
        for path in paths:
            buckets.setdefault(path.parent.name, []).append(path)
        selected_paths = []
        cursors = {name: 0 for name in sorted(buckets)}
        while len(selected_paths) < limit:
            progressed = False
            for name in sorted(buckets):
                index = cursors[name]
                if index >= len(buckets[name]):
                    continue
                selected_paths.append(buckets[name][index])
                cursors[name] = index + 1
                progressed = True
                if len(selected_paths) >= limit:
                    break
            if not progressed:
                break

    selected_set = set(selected_paths)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    decision_rows: list[dict[str, Any]] = []
    # The scalar counters are computed over every valid header, while only
    # ``selected_paths`` are retained as artifact rows.  Decision rows have a
    # separate hard window so a large hypothesis/artifact history cannot hide
    # the latest gate/review events.
    for path in paths:
        try:
            item = validate_research_object(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ResearchObjectError, ValueError) as exc:
            errors.append(type(exc).__name__)
            continue
        accumulate_source_stats(source_stats, item)
        if path in selected_set:
            rows.append(item)
        if str(item.get("kind") or "") in _DECISION_KINDS:
            decision_rows.append(item)
    decision_rows = decision_rows[-_HARD_MAX_DECISIONS:]
    source_stats["recovery_candidates"] = sum(
        bool(targets.intersection(source_stats["recovery_failed_ids"]))
        for targets in source_stats["recovery_links"]
    )
    source_stats.pop("recovery_failed_ids", None)
    source_stats.pop("recovery_links", None)
    return (
        rows,
        source_count,
        truncated,
        sorted(set(errors)),
        source_kind_counts,
        decision_rows,
        source_stats,
    )


def _git_parent_hashes(root: Path, commit: str) -> list[str]:
    """Read actual Git parents without returning command errors or paths."""

    if not commit:
        return []
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-list", "--parents", "-n", "1", commit],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    fields = result.stdout.strip().split()
    return fields[1:]


def _attempt_metrics(objects: list[Mapping[str, Any]]) -> dict[str, int]:
    attempts = [
        item for item in objects if str(item.get("kind") or "") == "experiment_attempt"
    ]
    failed = 0
    completed = 0
    for item in attempts:
        state = str(item.get("state") or "").lower()
        payload = (
            item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
        )
        payload_state = str(payload.get("status") or "").lower()
        effective_state = payload_state or state
        failed += int(effective_state in _FAILED_STATES)
        completed += int(effective_state in _COMPLETED_STATES)
    return {
        "attempt_count": len(attempts),
        "failed_attempts": failed,
        "completed_attempts": completed,
    }


def _recovery_candidate_count(objects: list[Mapping[str, Any]]) -> int:
    """Count only explicit completed repair links, never all successes."""

    failed_ids = {
        str(item.get("object_id") or "")
        for item in objects
        if str(item.get("state") or "").lower() in _FAILED_STATES
    }
    if not failed_ids:
        return 0
    count = 0
    for item in objects:
        if str(item.get("state") or "").lower() not in _COMPLETED_STATES:
            continue
        targets = {
            str(relation.get("target") or "")
            for relation in item.get("relations") or []
            if isinstance(relation, Mapping)
            and str(relation.get("type") or "") in _REPAIR_RELATIONS
        }
        if targets.intersection(failed_ids):
            count += 1
    return count


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _relation_types(item: Mapping[str, Any]) -> list[str]:
    values: set[str] = set()
    for row in item.get("relations") or []:
        if not isinstance(row, Mapping) or not row.get("type"):
            continue
        relation = str(row.get("type") or "")
        # Built-in relation names are a finite protocol vocabulary.  Extension
        # URIs are represented generically so they cannot carry free text.
        values.add(relation if relation in _SAFE_RELATIONS else "extension")
    return sorted(values)


def _failure_codes(item: Mapping[str, Any]) -> list[str]:
    payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
    codes: list[str] = []
    state = str(item.get("state") or "").lower()
    payload_state = str(payload.get("status") or "").lower()
    if state in _FAILED_STATES:
        codes.append(f"state:{state}")
    elif payload_state in _FAILED_STATES:
        codes.append(f"payload_state:{payload_state}")
    for key in (
        "required_failures",
        "blocking_issues",
        "active_issues",
        "unresolved_issues",
        "issues",
    ):
        if payload.get(key) not in (None, "", [], {}):
            # Presence of an issue channel is useful; the issue value itself
            # may be a task answer, prompt fragment, or local path.
            codes.append(f"payload:{key}")
    if payload and any(
        key in payload
        for key in ("failure_reason", "error", "exception", "negative_results")
    ):
        codes.append("payload:failure_signal")
    return sorted(set(codes))[:12]


def _artifact_row(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
    provenance = (
        item.get("provenance") if isinstance(item.get("provenance"), Mapping) else {}
    )
    kind = _safe_kind(item.get("kind"))
    state = _safe_state(item.get("state"))
    return {
        "object_id": _short_hash(item.get("object_id")),
        "kind": kind,
        "stage": _KIND_STAGE.get(str(item.get("kind") or ""), "X"),
        "state": state,
        "created_at": _safe_timestamp(item.get("created_at")),
        "content_hash": _short_hash(item.get("content_hash")),
        "relation_types": _relation_types(item),
        "signals": {
            "falsifier": bool(payload.get("falsifier")),
            "metric": bool(
                payload.get("metric")
                or payload.get("metrics")
                or payload.get("result_summary")
            ),
            "failure_reason": bool(
                payload.get("failure_reason")
                or payload.get("error")
                or payload.get("issues")
            ),
            "counterevidence": bool(
                payload.get("counterevidence")
                or payload.get("negative_results")
                or payload.get("refutes")
            ),
            "provenance": bool(provenance),
            "independence": bool(
                payload.get("independence")
                or payload.get("independent")
                or payload.get("evaluator_id")
            ),
        },
        "failure_codes": _failure_codes(item),
    }


def _decision_row(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
    kind = str(item.get("kind") or "")
    state = _safe_state(item.get("state"))
    return {
        "object_id": _short_hash(item.get("object_id")),
        "kind": _safe_kind(kind),
        "stage": _KIND_STAGE.get(kind, "X"),
        "state": state,
        "decision": _safe_decision(_first(payload, "decision", "status", "intent")),
        "relation_types": _relation_types(item),
        "issue_count": sum(
            len(value) if isinstance(value, list) else 1
            for key, value in payload.items()
            if key
            in {
                "required_failures",
                "blocking_issues",
                "active_issues",
                "unresolved_issues",
                "issues",
            }
            and value not in (None, "", [], {})
        ),
        "independence_observed": bool(
            payload.get("independence")
            or payload.get("independent")
            or payload.get("evaluator_id")
        ),
    }


def _commit_row(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    trailers = (
        entry.get("trailers") if isinstance(entry.get("trailers"), Mapping) else {}
    )

    def trailer(name: str) -> list[str]:
        values = trailers.get(name) or []
        return [str(value) for value in values]

    trailer_parents = trailer("Research-Parent")
    checkpoint_ids = trailer("Research-Checkpoint")
    stages = trailer("Research-Stage")
    states = trailer("Research-State")
    reproduce = trailer("Reproduce")
    commit = str(entry.get("commit") or "")
    git_parents = _git_parent_hashes(root, commit)
    parents = git_parents or trailer_parents
    stage = _safe_stage(stages[0] if stages else "")
    status = _safe_state(states[0] if states else "")
    row: dict[str, Any] = {
        # Full commit hashes are not needed for a shareable process report.
        "commit": _short_hash(commit),
        "short_commit": _short_commit(entry.get("short_commit") or commit),
        "authored_at": _safe_timestamp(entry.get("authored_at")),
        "stage": stage,
        "status": status,
        "subject": _safe_subject(stage),
        "subject_redacted": True,
        "checkpoint_id": _short_hash(checkpoint_ids[0]) if checkpoint_ids else None,
        "parent_count": len(parents),
        "parents": [_short_hash(parent) for parent in parents],
        "has_reproduce_command": bool(reproduce),
        "checkpoint": {
            "available": False,
            "object_count": 0,
            "claim_count": 0,
            "node_count": 0,
            "changed_path_count": 0,
        },
    }
    # A checkpoint is useful as a compact “what changed” boundary.  We expose
    # counts only; changed paths and free-form summaries stay local.
    try:
        checkpoint = show_checkpoint(root, commit)
        payload = checkpoint.get("checkpoint") or {}
        checkpoint_stage = _safe_stage(payload.get("stage"))
        checkpoint_status = _safe_state(payload.get("status"))
        row["checkpoint"] = {
            "available": True,
            "object_count": len(payload.get("object_refs") or []),
            "claim_count": len(payload.get("claims") or []),
            "node_count": len(payload.get("nodes") or []),
            "changed_path_count": len(payload.get("changed_paths") or []),
        }
        row["stage"] = checkpoint_stage
        row["status"] = checkpoint_status
        row["subject"] = _safe_subject(checkpoint_stage)
        row["checkpoint_id"] = _short_hash(payload.get("checkpoint_id"))
        checkpoint_parent = str(payload.get("parent_commit") or "")
        if not git_parents and checkpoint_parent:
            row["parents"] = [_short_hash(checkpoint_parent)]
            row["parent_count"] = 1
        row["has_reproduce_command"] = bool(
            isinstance(payload.get("reproduce"), Mapping)
            and str((payload.get("reproduce") or {}).get("command") or "").strip()
        )
    except (OSError, ValueError, KeyError, ResearchGitError):
        pass
    return row


def _order_commit_entries(
    root: Path, entries: list[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Order a bounded commit union chronologically and parent-first.

    Git log is returned newest-first for each ref.  Sorting only by author
    timestamp is not enough: checkpoint commits often share a second, and a
    lexical hash tie-breaker can put an ``ideation`` commit before ``init``.
    A small topological sort over the bounded union gives reviewers a stable
    causal timeline while retaining deterministic ordering for unrelated
    roots.  Parents outside the bounded union are treated as already
    satisfied, and no commit message is read.
    """

    if not entries:
        return []
    by_commit: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        commit = str(entry.get("commit") or "")
        if commit:
            by_commit.setdefault(commit, entry)
    if not by_commit:
        return list(entries)

    # Use the hash as the deterministic fallback for unrelated roots.  The
    # timestamp is only a tie-breaker after parent relationships are honored.
    ordered_keys = sorted(by_commit)
    input_order = {commit: index for index, commit in enumerate(ordered_keys)}
    children: dict[str, set[str]] = {commit: set() for commit in by_commit}
    indegree: dict[str, int] = {commit: 0 for commit in by_commit}
    for commit in by_commit:
        known_parents = {
            parent for parent in _git_parent_hashes(root, commit) if parent in by_commit
        }
        indegree[commit] = len(known_parents)
        for parent in known_parents:
            children.setdefault(parent, set()).add(commit)

    def key(commit: str) -> tuple[str, int, str]:
        return (
            str(by_commit[commit].get("authored_at") or ""),
            input_order.get(commit, 0),
            commit,
        )

    ready = [commit for commit, degree in indegree.items() if degree == 0]
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    # Re-sort the small ready set after each pop.  The public cap is 128, so a
    # simple list keeps the ordering rule obvious and avoids exposing a
    # dependency on Git's locale-specific log order.
    while ready:
        ready.sort(key=key)
        commit = ready.pop(0)
        if commit in seen:
            continue
        seen.add(commit)
        result.append(by_commit[commit])
        for child in sorted(children.get(commit, ())):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    # Malformed or shallow Git metadata should not make the audit fail.  Any
    # unresolved cycle/entry is appended in a deterministic order and remains
    # visible as a normal bounded commit row.
    result.extend(
        by_commit[commit] for commit in sorted(by_commit, key=key) if commit not in seen
    )
    return result


def _unavailable_summary(
    *,
    task_manifest_sha256: str | None,
    task_count: int | None,
    task_filter: str,
    task_limit: int | None,
    gold_fields_used: bool,
    errors: list[str],
    limits: Mapping[str, int],
    truncation: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Return the same machine-readable shape when no Research VCS exists."""

    truncation = dict(truncation or {})
    fairness = _fairness_report(
        task_manifest_sha256=task_manifest_sha256,
        task_count=task_count,
        task_filter=task_filter,
        task_limit=task_limit,
        gold_fields_used=gold_fields_used,
    )
    return {
        "schema": PROCESS_SCHEMA,
        "available": False,
        "reason": "research_vcs_unavailable",
        "errors": sorted(set(errors)),
        "reasoning_boundary": (
            "artifact-backed decisions only; hidden chain-of-thought is not recorded"
        ),
        "repository": {
            "branch": None,
            "branch_digest": None,
            "head": None,
            "worktree_clean": None,
            "checkpoint_policy": "unknown",
            "staged_path_count": 0,
            "eligible_change_count": 0,
        },
        "branches": [],
        "branch_topology": {
            "branch_count": 0,
            "source_branch_count": 0,
            "branching_observed": False,
            "merge_observed": False,
            "artifact_scope": "current_checkout_only",
            "per_branch_artifacts_available": False,
            "truncated": bool(truncation.get("branches")),
            "fair_branch_comparison": _fair_branch_comparison(
                branch_count=0, task_manifest_sha256=task_manifest_sha256
            ),
        },
        "commits": [],
        "intermediate": {
            "object_count": 0,
            "source_object_count": 0,
            "valid_object_count": 0,
            "statistics_scope": "no_repository",
            "attempt_statistics_scope": "no_repository",
            "attempts_truncated": False,
            "kind_counts": {},
            "state_counts": {},
            "stage_counts": {},
            "coverage": {
                "observed_stages": [],
                "stage_count": 0,
                "stage_total": len(_STAGE_LABELS),
                "ratio": 0.0,
                "truncated": bool(truncation.get("artifacts")),
                "scope": "no_repository",
                "quality_claim_allowed": False,
            },
            "attempt_count": 0,
            "failed_attempts": 0,
            "completed_attempts": 0,
            "artifacts": [],
            "decision_events": [],
            "decision_statistics_scope": "no_repository",
            "failed_or_blocked_artifacts": [],
            "failed_or_blocked_count": 0,
            "failed_or_blocked_visible_count": 0,
            "source_failed_or_blocked_count": 0,
            "failure_statistics_scope": "no_repository",
            "recovery_candidates": 0,
            "recovery_claim_allowed": False,
            "recovery_statistics_scope": "no_repository",
        },
        "fairness": fairness,
        "limits": {
            **dict(limits),
            "source_totals": {
                "branches": 0,
                "unique_commits": 0,
                "objects": 0,
                "decision_events": 0,
            },
            "source_totals_scope": "no_repository",
            "truncated": {
                "branches": bool(truncation.get("branches")),
                "commits": bool(truncation.get("commits")),
                "artifacts": bool(truncation.get("artifacts")),
                "decisions": bool(truncation.get("decisions")),
            },
        },
        "redaction": {
            "free_text": "omitted",
            "absolute_paths": "omitted",
            "gold_fields": "omitted",
            "branch_names": "alias_plus_digest",
            "subjects": "stage_only",
        },
    }


def _fairness_report(
    *,
    task_manifest_sha256: str | None,
    task_count: int | None,
    task_filter: str,
    task_limit: int | None,
    gold_fields_used: bool,
) -> dict[str, Any]:
    # Keep both verbose and compact aliases: callers from the CLI and callers
    # consuming benchmark JSON should not need a schema-specific rename.
    return {
        "comparison_unit": "task-manifest + bounded artifact trajectory",
        "task_manifest_sha256": _safe_manifest_digest(task_manifest_sha256),
        "task_count": task_count,
        "task_filter": _safe_task_filter(task_filter),
        "task_limit": task_limit,
        "filter": _safe_task_filter(task_filter),
        "limit": task_limit,
        # This audit never reads task gold fields, even if a caller passes a
        # truthy legacy flag; keeping the output fail-closed protects the
        # shareable fairness contract.
        "gold_fields_used": False,
        "network_used": False,
        "provider_used": False,
        "model_used": False,
        "network": False,
        "provider": False,
        "model": False,
        "model_cost_usd": 0.0,
        "cost": 0.0,
        "audit_command_only": True,
        "trajectory_cost": "unobserved",
        "rollouts_evaluated": 0,
        "same_budget_and_evaluator": False,
        "official_comparable": False,
    }


def _fair_branch_comparison(
    *, branch_count: int, task_manifest_sha256: str | None
) -> dict[str, Any]:
    """State fair-comparison requirements without claiming they were met."""

    checks = {
        "same_manifest": _manifest_is_usable(task_manifest_sha256),
        "same_task_slice": False,
        "gold_exclusion": True,
        "same_budget": False,
        "same_evaluator": False,
        # A common Git ancestor is not a sufficient proof that both lines
        # used the same scientific starting state, so leave this unverified.
        "same_base": False,
    }
    return {
        "eligible": bool(branch_count > 1 and all(checks.values())),
        "requirements": [
            "same_manifest",
            "same_task_slice",
            "gold_exclusion",
            "same_budget",
            "same_evaluator",
            "same_base",
        ],
        "checks": checks,
        "same_task_manifest_required": True,
        "same_budget_required": True,
        "same_evaluator_required": True,
        "verified_here": False,
    }


def build_process_summary(
    project_root: str | Path,
    *,
    task_manifest_sha256: str | None = None,
    task_count: int | None = None,
    task_filter: str = "all",
    task_limit: int | None = None,
    gold_fields_used: bool = False,
    max_branches: int | None = None,
    max_commits: int | None = None,
    max_artifacts: int | None = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    """Build a bounded process view without exporting a model transcript."""

    effective_commits, commits_clamped = _bounded_limit(
        max_commits, _MAX_COMMITS, _HARD_MAX_COMMITS
    )
    effective_artifacts, artifacts_clamped = _bounded_limit(
        max_artifacts, _MAX_ARTIFACTS, _HARD_MAX_ARTIFACTS
    )
    effective_decisions, decisions_clamped = _bounded_limit(
        max_decisions, _MAX_DECISIONS, _HARD_MAX_DECISIONS
    )
    effective_branches, branches_clamped = _bounded_limit(
        max_branches, _DEFAULT_MAX_BRANCHES, _HARD_MAX_BRANCHES
    )
    limits = {
        "max_branches": effective_branches,
        "max_commits": effective_commits,
        "max_artifacts": effective_artifacts,
        "max_decisions": effective_decisions,
    }
    initial_truncation = {
        "branches": branches_clamped,
        "commits": commits_clamped,
        "artifacts": artifacts_clamped,
        "decisions": decisions_clamped,
    }
    root = Path(project_root).expanduser().resolve()
    errors: list[str] = []
    try:
        status = repository_status(root)
        all_branches = list_research_branches(root)
        (
            all_objects,
            source_object_count,
            objects_truncated,
            object_read_errors,
            source_kind_counts,
            source_decision_objects,
            source_stats,
        ) = _bounded_object_scan(root, effective_artifacts)
        errors.extend(object_read_errors)
    except (OSError, ValueError, ResearchGitError) as exc:
        return _unavailable_summary(
            task_manifest_sha256=task_manifest_sha256,
            task_count=task_count,
            task_filter=task_filter,
            task_limit=task_limit,
            gold_fields_used=gold_fields_used,
            errors=[type(exc).__name__],
            limits=limits,
            truncation=initial_truncation,
        )

    branches_truncated = len(all_branches) > effective_branches
    current_branch = str(status.get("branch") or "detached")
    current_head = str(status.get("head") or "")
    if branches_truncated:
        current_rows = [
            row
            for row in all_branches
            if bool(row.get("current")) or str(row.get("name") or "") == current_branch
        ]
        other_rows = [row for row in all_branches if row not in current_rows]
        branches = (current_rows[:1] + other_rows)[:effective_branches]
    else:
        branches = all_branches
    objects = all_objects

    branch_rows: list[dict[str, Any]] = []
    branch_logs: dict[str, list[dict[str, Any]]] = {}
    branch_aliases: dict[str, str] = {}
    alternative_index = 0
    for branch in branches:
        name = str(branch.get("name") or "")
        if not name:
            continue
        if name == current_branch or bool(branch.get("current")):
            alias = "current"
        else:
            alternative_index += 1
            alias = f"alternative-{alternative_index}"
        branch_aliases[name] = alias
        try:
            # Ask for one sentinel entry beyond the public cap so the report
            # can distinguish an exact short history from a truncated one
            # without reading an unbounded log.
            log = research_log(root, ref=name, limit=effective_commits + 1)
        except (OSError, ValueError, ResearchGitError) as exc:
            errors.append(type(exc).__name__)
            log = []
        branch_logs[name] = log
        commit = str(branch.get("commit") or "")
        stage = _safe_stage(branch.get("stage"))
        branch_status = _safe_state(branch.get("status"))
        is_current = bool(branch.get("current")) or name == current_branch
        branch_rows.append(
            {
                "name": alias,
                "name_digest": _ref_digest(name),
                "current": is_current,
                "commit": _short_hash(commit),
                "stage": stage,
                "status": branch_status,
                "checkpoint_id": _short_hash(branch.get("checkpoint_id")),
                "commit_count": min(len(log), effective_commits),
                "commit_count_truncated": len(log) > effective_commits,
                "relation": (
                    "current"
                    if is_current
                    else (
                        "same_head"
                        if commit and commit == current_head
                        else "diverged_or_behind"
                    )
                ),
            }
        )

    unique_commits: dict[str, dict[str, Any]] = {}
    for log in branch_logs.values():
        for entry in log:
            commit = str(entry.get("commit") or "")
            if commit:
                unique_commits.setdefault(commit, entry)
    all_ordered_entries = _order_commit_entries(root, list(unique_commits.values()))
    # If branch rows were capped, omitted branch logs may contain additional
    # commits even when the visible union fits the commit limit.
    branch_log_truncated = any(
        len(log) > effective_commits for log in branch_logs.values()
    )
    commits_truncated = (
        branches_truncated
        or branch_log_truncated
        or len(all_ordered_entries) > effective_commits
    )
    ordered_entries = all_ordered_entries[-effective_commits:]
    commits = [_commit_row(root, entry) for entry in ordered_entries]
    commit_branch_aliases: dict[str, set[str]] = {}
    for branch_name, log in branch_logs.items():
        alias = branch_aliases.get(branch_name)
        if not alias:
            continue
        for entry in log:
            commit = str(entry.get("commit") or "")
            if commit:
                commit_branch_aliases.setdefault(commit, set()).add(alias)
    for entry, row in zip(ordered_entries, commits):
        row["branches"] = sorted(
            commit_branch_aliases.get(str(entry.get("commit") or ""), set())
        )

    ordered_objects = sorted(
        objects,
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("object_id") or ""),
        ),
    )
    bounded_objects = ordered_objects[-effective_artifacts:]
    artifact_rows = [_artifact_row(item) for item in bounded_objects]
    # Header statistics are calculated over every valid object, even when the
    # exported artifact rows are bounded.  This prevents a small display
    # window from turning old failed/completed attempts into apparent zeros.
    kind_counts = source_stats["valid_kind_counts"]
    source_decision_count = sum(
        value for kind, value in kind_counts.items() if kind in _DECISION_KINDS
    )
    state_counts = source_stats["state_counts"]
    stage_counts = source_stats["stage_counts"]
    failed = [
        row
        for row in artifact_rows
        if row["state"] in {"failed", "rejected", "blocked"} or row["failure_codes"]
    ]
    source_failed_count = int(source_stats["failed_or_blocked_count"])
    decision_objects = source_decision_objects
    decisions = [
        _decision_row(item) for item in decision_objects[-effective_decisions:]
    ]
    decisions_truncated = source_decision_count > effective_decisions
    merge_observed = any(int(row.get("parent_count") or 0) > 1 for row in commits)
    branch_observed = len(all_branches) > 1
    # Attempt totals are scalar header counts over the source object set;
    # detailed artifact rows remain bounded and may not contain every attempt.
    attempt_metrics = dict(source_stats["attempt_metrics"])
    observed_stages = sorted(
        stage
        for stage in stage_counts
        if stage in _STAGE_LABELS and stage_counts[stage] > 0
    )
    coverage = {
        "observed_stages": observed_stages,
        "stage_count": len(observed_stages),
        "stage_total": len(_STAGE_LABELS),
        "ratio": round(len(observed_stages) / len(_STAGE_LABELS), 3),
        "truncated": objects_truncated or commits_truncated,
        "scope": "all_valid_objects",
        "quality_claim_allowed": False,
    }
    branch_fairness = _fair_branch_comparison(
        branch_count=len(all_branches), task_manifest_sha256=task_manifest_sha256
    )
    # Branch aliases are intentionally the only branch labels exported.  Keep
    # a stable digest for local cross-report joins without exposing the ref.
    current_alias = "current" if current_branch in branch_aliases else None
    process = {
        "schema": PROCESS_SCHEMA,
        "available": True,
        "reasoning_boundary": (
            "artifact-backed decisions only; hidden chain-of-thought is not recorded"
        ),
        "repository": {
            "branch": current_alias,
            "branch_digest": _ref_digest(current_branch),
            "head": _short_hash(current_head),
            "worktree_clean": bool(status.get("worktree_clean")),
            "checkpoint_policy": _safe_policy(status.get("checkpoint_policy")),
            "staged_path_count": len(status.get("staged_paths") or []),
            "eligible_change_count": len(status.get("eligible_changes") or []),
        },
        "branches": branch_rows,
        "branch_topology": {
            "branch_count": len(branch_rows),
            "source_branch_count": len(all_branches),
            "branching_observed": branch_observed,
            "merge_observed": merge_observed,
            # Objects are read from the checked-out tree.  We expose branch
            # membership for commits, but do not pretend that each branch has
            # an independently evaluated artifact outcome.
            "artifact_scope": "current_checkout_only",
            "per_branch_artifacts_available": False,
            "truncated": branches_truncated,
            "fair_branch_comparison": branch_fairness,
        },
        "commits": commits,
        "intermediate": {
            "object_count": source_object_count,
            "source_object_count": source_object_count,
            "statistics_scope": "all_valid_objects",
            "valid_object_count": int(source_stats["valid_object_count"]),
            "attempt_statistics_scope": "all_valid_objects",
            "attempts_truncated": objects_truncated,
            "kind_counts": dict(sorted(kind_counts.items())),
            "state_counts": dict(sorted(state_counts.items())),
            "stage_counts": dict(sorted(stage_counts.items())),
            "coverage": coverage,
            **attempt_metrics,
            "artifacts": artifact_rows,
            "decision_events": decisions,
            "decision_statistics_scope": "all_decision_headers; events are bounded",
            "failed_or_blocked_artifacts": failed,
            "failed_or_blocked_count": source_failed_count,
            "failed_or_blocked_visible_count": len(failed),
            "source_failed_or_blocked_count": source_failed_count,
            "failure_statistics_scope": "all_valid_objects",
            "recovery_candidates": int(source_stats["recovery_candidates"]),
            "recovery_claim_allowed": False,
            "recovery_statistics_scope": "all_valid_objects",
        },
        "fairness": _fairness_report(
            task_manifest_sha256=task_manifest_sha256,
            task_count=task_count,
            task_filter=task_filter,
            task_limit=task_limit,
            gold_fields_used=gold_fields_used,
        ),
        "limits": {
            **limits,
            "source_totals": {
                "branches": len(all_branches),
                "unique_commits": len(all_ordered_entries),
                "objects": source_object_count,
                "decision_events": source_decision_count,
            },
            "source_totals_scope": (
                "bounded_branch_logs"
                if branch_log_truncated
                else (
                    "visible_branch_subset"
                    if branches_truncated
                    else "all_visible_branches"
                )
            ),
            "truncated": {
                "branches": branches_truncated or branches_clamped,
                "commits": commits_truncated or commits_clamped,
                "artifacts": objects_truncated or artifacts_clamped,
                "decisions": decisions_truncated or decisions_clamped,
            },
        },
        "redaction": {
            "free_text": "omitted",
            "absolute_paths": "omitted",
            "gold_fields": "omitted",
            "branch_names": "alias_plus_digest",
            "subjects": "stage_only",
        },
        "errors": sorted(set(errors)),
    }
    return process


__all__ = ["PROCESS_SCHEMA", "build_process_summary"]
