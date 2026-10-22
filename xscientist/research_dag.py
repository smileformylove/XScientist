"""Unified scientific-evidence DAG projection and offline browser."""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.graph import analyze_exploration_graph
from ai_scientist.protocol.hashing import content_hash, hash_manifest
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.ara_paths import validate_node_id
from ai_scientist.utils.safe_files import BoundedFileError, read_bounded_regular_file

from .research_closure import audit_research_closure
from .research_argument import (
    build_research_argument_view,
    validate_research_argument_view,
)
from .research_git import (
    ResearchGitError,
    _normalise_relative,
    _pointer_store_path,
    _record_pointer,
    _repository_root,
    _run_git,
    _run_git_bounded,
    _validate_pointer_payload,
    list_research_objects_at_ref,
    research_object_introduction_order,
    show_checkpoint,
)

RESEARCH_DAG_SCHEMA = "xscientist.research-dag.v1"
_MAX_LINKED_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_ARTIFACT_BINDING_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_ARTIFACT_BINDING_FILES = 4096
_MAX_ARTIFACT_POINTER_BYTES = 1024 * 1024
_MAX_ARTIFACT_POINTER_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_ARTIFACT_POINTER_FILES = 4096
_MAX_ARTIFACT_POINTER_ENUMERATION_BYTES = 1024 * 1024
_MAX_ARA_NODE_DETAIL_BYTES = 32 * 1024 * 1024
_MAX_ARA_NODE_DETAILS_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_ARA_NODE_DETAIL_FILES = 4096
_MAX_ARA_CORE_JSON_BYTES = 32 * 1024 * 1024
_MAX_ARA_MANIFEST_REVISIONS = 4096
_MAX_ARA_ARCHIVED_MANIFEST_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_COMMITTED_ARA_MANIFESTS = 64
_MAX_COMMITTED_ARA_HISTORY_COMMITS = 512
_MAX_COMMITTED_ARA_ENUMERATION_BYTES = 1024 * 1024
_MAX_COMMITTED_ARA_HISTORY_OUTPUT_BYTES = 64 * 1024
_MAX_COMMITTED_ARA_CORE_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_COMMITTED_ARA_CORE_FILES = 4096
_ARA_SOURCE_SNAPSHOT_KEY = "__xscientist_source_snapshot__"
_MAX_GATE_LINEAGE_NODES = 4096
_MAX_GATE_LINEAGE_EDGES = 8192
_MAX_GATE_LINEAGE_DEPTH = 128
_MAX_GATE_LINEAGE_DETAIL_BYTES = 64 * 1024 * 1024

_PHASES = {
    "question": 0,
    "search_plan": 1,
    "search_receipt": 2,
    "source_snapshot": 3,
    "passage_evidence": 4,
    "hypothesis": 1,
    "hypothesis_portfolio": 1,
    "discriminating_prediction": 2,
    "experiment_priority": 2,
    "anomaly": 5,
    "research_review": 6,
    "mechanism_model": 5,
    "evidence_quality": 5,
    "boundary_condition": 6,
    "transfer_matrix": 6,
    "research_plan": 2,
    "preregistration": 2,
    "agent_candidate": 2,
    "experiment_attempt": 3,
    "agent_evaluation": 3,
    "context_snapshot": 4,
    "inference": 5,
    "warrant": 4,
    "assumption": 2,
    "method": 2,
    "estimand": 2,
    "effect_estimate": 4,
    "protocol_deviation": 4,
    "sensitivity_analysis": 5,
    "risk_of_bias": 5,
    "evidence_synthesis": 6,
    "challenge": 6,
    "source_update": 4,
    "context_robustness": 5,
    "research_goal": 0,
    "action_proposal": 2,
    "experiment_design": 2,
    "resource_budget": 2,
    "stopping_decision": 5,
    "novelty_check": 5,
    "evaluation_blinding": 2,
    "human_escalation": 5,
    "experiment_node": 3,
    "metric": 4,
    "evidence": 4,
    "observation": 4,
    "review": 5,
    "gate_decision": 5,
    "claim": 6,
    "reproduction": 7,
    "manuscript": 8,
}

_LAYER_KINDS = {
    "strategy": {
        "question",
        "research_goal",
        "search_plan",
        "hypothesis",
        "hypothesis_portfolio",
        "discriminating_prediction",
        "experiment_priority",
        "action_proposal",
        "research_review",
        "stopping_decision",
        "human_escalation",
    },
    "execution": {
        "research_plan",
        "preregistration",
        "experiment_design",
        "resource_budget",
        "evaluation_blinding",
        "experiment_attempt",
        "experiment_node",
    },
    "evidence": {
        "search_receipt",
        "source_snapshot",
        "passage_evidence",
        "observation",
        "metric",
        "evidence",
        "effect_estimate",
        "reproduction",
        "evidence_quality",
        "risk_of_bias",
    },
    "theory": {
        "inference",
        "warrant",
        "assumption",
        "method",
        "estimand",
        "mechanism_model",
        "sensitivity_analysis",
        "evidence_synthesis",
        "boundary_condition",
        "transfer_matrix",
        "anomaly",
        "challenge",
        "claim",
    },
    "decision_memory": {
        "context_snapshot",
        "review",
        "gate_decision",
        "context_robustness",
        "novelty_check",
        "protocol_deviation",
        "source_update",
    },
    "evolution": {"agent_candidate", "agent_evaluation", "manuscript"},
}

_GATE_UPSTREAM_RELATIONS = frozenset(
    {
        "addresses_estimand",
        "consumes",
        "depends_on",
        "derived_by",
        "derived_from",
        "evaluates",
        "generated_by",
        "has_effect_estimate",
        "has_premise",
        "observes",
        "quotes",
        "retrieves",
        "under_assumption",
        "uses_context",
        "uses_method",
    }
)

_NEGATIVE_HYPOTHESIS_RELATIONS = frozenset(
    {
        "challenges_inference",
        "contradicts",
        "invalidates",
        "qualified_refutes",
        "refutes",
        "rejects",
    }
)
_TERMINAL_HYPOTHESIS_STATES = frozenset(
    {"cancelled", "failed", "rejected", "superseded", "timed_out"}
)
_SUBMISSION_GATE_AUTHORITIES = frozenset({"deterministic_gate", "human"})
_DECIDED_GATE_STATES = frozenset({"completed", "rejected", "verified"})
_INVALID_ARA_GRAPH_HASH = "<invalid>"

_RELATION_CATEGORIES = {
    "supports": "support",
    "qualified_supports": "support",
    "refutes": "challenge",
    "qualified_refutes": "challenge",
    "contradicts": "challenge",
    "challenges_inference": "challenge",
    "invalidates": "challenge",
    "has_premise": "argument",
    "uses_method": "argument",
    "under_assumption": "argument",
    "addresses_estimand": "argument",
    "has_effect_estimate": "argument",
    "qualifies": "argument",
    "updates": "provenance",
    "selects": "decision",
    "rejects": "decision",
    "evaluates": "verification",
    "reproduces": "verification",
    "attests": "verification",
    "promotes": "evolution",
    "supersedes": "evolution",
    "depends_on": "lineage",
    "derived_from": "lineage",
    "retrieves": "lineage",
    "cites": "lineage",
    "quotes": "lineage",
    "observes": "lineage",
    "generated_by": "lineage",
    "derived_by": "lineage",
    "consumes": "lineage",
    "produces": "lineage",
    "uses_context": "context",
}

# Unknown extension relations are retained as edges for provenance, but must
# never be presented as scientific lineage.  They are surfaced explicitly in
# the graph integrity report and receive the dedicated ``unknown`` category.
_UNKNOWN_RELATION_CATEGORY = "unknown"


def _epistemic_layer(kind: str) -> str:
    for layer, kinds in _LAYER_KINDS.items():
        if kind in kinds:
            return layer
    return "evidence"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _strict_json_loads(raw: str) -> Any:
    """Decode JSON while rejecting ambiguous keys and non-finite numbers."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {token}")

    return json.loads(
        raw,
        object_pairs_hook=reject_duplicate_keys,
        parse_float=finite_float,
        parse_constant=reject_constant,
    )


def _strict_json_bytes(encoded: bytes) -> Any:
    return _strict_json_loads(encoded.decode("utf-8"))


def _nul_terminated_git_paths(raw: str, *, label: str) -> list[str]:
    """Parse a bounded ``git -z`` pathname stream without newline ambiguity."""

    if not raw:
        return []
    if "\ufffd" in raw or not raw.endswith("\0"):
        raise ResearchGitError(f"{label} returned an invalid pathname stream")
    paths = raw[:-1].split("\0")
    if any(not path for path in paths):
        raise ResearchGitError(f"{label} returned an invalid pathname stream")
    return paths


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_snapshot_receipt(
    *,
    graph_sha256: str,
    sidecar_sha256: str,
    source_ref: str,
    source_path: str,
) -> dict[str, str]:
    core = {
        "graph_sha256": graph_sha256,
        "sidecar_sha256": sidecar_sha256,
        "source_ref": source_ref,
        "source_path": source_path,
    }
    return {**core, "receipt_hash": canonical_content_hash(core)}


def _valid_source_snapshot(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "graph_sha256",
        "sidecar_sha256",
        "source_ref",
        "source_path",
        "receipt_hash",
    }
    if set(value) != expected_keys:
        return False
    core = {key: value[key] for key in expected_keys if key != "receipt_hash"}
    try:
        return bool(
            _is_sha256(core["graph_sha256"])
            and _is_sha256(core["sidecar_sha256"])
            and isinstance(core["source_ref"], str)
            and bool(core["source_ref"])
            and isinstance(core["source_path"], str)
            and bool(core["source_path"])
            and value["receipt_hash"] == canonical_content_hash(core)
        )
    except (TypeError, ValueError):
        return False


def _json_values_identical(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _json_values_identical(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_identical(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _stat_open_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _stat_stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_local_ara_root(root: Path) -> tuple[Path, int]:
    """Open the resolved external root as the sole filesystem trust anchor."""

    try:
        resolved = root.expanduser().resolve(strict=True)
        before = resolved.lstat()
    except OSError as exc:
        raise ResearchGitError(f"ARA root is unavailable: {root}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise ResearchGitError(f"ARA root is not a directory: {resolved}")
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise ResearchGitError("secure descriptor-relative ARA reads are unavailable")
    required_flags = os.O_NOFOLLOW | os.O_DIRECTORY
    descriptor: int | None = None
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | required_flags | getattr(os, "O_CLOEXEC", 0),
        )
        after = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ResearchGitError(f"cannot securely open ARA root: {resolved}") from exc
    assert descriptor is not None
    if _stat_open_identity(before) != _stat_open_identity(after):
        os.close(descriptor)
        raise ResearchGitError(f"ARA root changed while opening: {resolved}")
    return resolved, descriptor


def _read_local_ara_file(
    root_descriptor: int,
    relative_path: str | PurePosixPath,
    *,
    maximum: int,
    label: str,
) -> bytes | None:
    """Read a stable regular file without following any descendant symlink."""

    relative = PurePosixPath(str(relative_path))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ResearchGitError(f"unsafe {label} path: {relative_path}")
    if maximum < 0:
        raise ResearchGitError(f"invalid {label} byte limit")

    directory_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in relative.parts[:-1]:
            try:
                before = os.stat(
                    component,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return None
            if not stat.S_ISDIR(before.st_mode):
                raise ResearchGitError(
                    f"{label} ancestor is not a real directory: {relative_path}"
                )
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ResearchGitError(
                    f"cannot securely open {label} ancestor: {relative_path}"
                ) from exc
            try:
                after = os.fstat(child_descriptor)
            except OSError as exc:
                os.close(child_descriptor)
                raise ResearchGitError(
                    f"cannot inspect {label} ancestor: {relative_path}"
                ) from exc
            if _stat_open_identity(before) != _stat_open_identity(after):
                os.close(child_descriptor)
                raise ResearchGitError(
                    f"{label} ancestor changed while opening: {relative_path}"
                )
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor

        leaf = relative.parts[-1]
        try:
            before = os.stat(
                leaf,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode):
            raise ResearchGitError(f"{label} is not a regular file: {relative_path}")
        if before.st_size > maximum:
            raise ResearchGitError(f"{label} exceeds {maximum} bytes: {relative_path}")
        try:
            file_descriptor = os.open(
                leaf,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise ResearchGitError(
                f"cannot securely open {label}: {relative_path}"
            ) from exc
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_open_identity(
            before
        ) != _stat_open_identity(opened):
            raise ResearchGitError(f"{label} changed while opening: {relative_path}")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ResearchGitError(
                    f"{label} exceeds {maximum} bytes: {relative_path}"
                )
        finished = os.fstat(file_descriptor)
        if _stat_stable_identity(opened) != _stat_stable_identity(finished):
            raise ResearchGitError(f"{label} changed while reading: {relative_path}")
        return b"".join(chunks)
    except ResearchGitError:
        raise
    except OSError as exc:
        raise ResearchGitError(
            f"cannot securely read {label}: {relative_path}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _summary(item: Mapping[str, Any], *, disclose: bool) -> str:
    if not disclose:
        return f"{item.get('kind')} {item.get('object_id')}"
    payload = item.get("payload") or {}
    if item.get("kind") == "gate_decision" and isinstance(payload, Mapping):
        decision = payload.get("decision") or payload.get("status")
        exact_fields = [
            f"{key}={json.dumps(payload[key], ensure_ascii=False, separators=(',', ':'))}"
            for key in (
                "candidate_count",
                "evaluated_candidate_ids",
                "experiment_authorized",
            )
            if key in payload
        ]
        if exact_fields:
            compact = " | ".join(
                ([str(decision)] if decision not in (None, "") else []) + exact_fields
            )
            return compact[:509] + ("..." if len(compact) > 509 else "")
    for key in (
        "statement",
        "question",
        "summary",
        "result",
        "title",
        "query",
        "locator",
        "decision",
        "status",
        "name",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            compact = " ".join(str(value).split())
            return compact[:157] + ("..." if len(compact) > 157 else "")
    return f"{item.get('kind')} {item.get('object_id')}"


def _strategy_projection(
    objects: Mapping[str, Mapping[str, Any]],
    introduction_order: Mapping[str, int],
    *,
    disclose: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Project current theory, claim drill-down, and research-depth coverage."""

    def targets(
        item: Mapping[str, Any], relation_types: set[str] | None = None
    ) -> set[str]:
        return {
            str(relation.get("target") or "")
            for relation in item.get("relations") or []
            if str(relation.get("target") or "")
            and (not relation_types or relation.get("type") in relation_types)
        }

    superseded = {
        target
        for item in objects.values()
        if item.get("state") in {"completed", "verified", "promoted"}
        for target in targets(item, {"supersedes"})
    }
    effective = {
        object_id: item
        for object_id, item in objects.items()
        if object_id not in superseded and item.get("state") != "superseded"
    }

    def latest(
        kind: str, *, portfolio_id: str | None = None
    ) -> Mapping[str, Any] | None:
        rows = [
            item
            for item in effective.values()
            if item.get("kind") == kind
            and (
                portfolio_id is None
                or (item.get("payload") or {}).get("portfolio_id") == portfolio_id
            )
        ]
        if not rows:
            return None
        if any(str(item["object_id"]) not in introduction_order for item in rows):
            raise ResearchGitError(
                f"DAG cannot verify Git introduction order for {kind}"
            )
        latest_sequence = max(
            introduction_order[str(item["object_id"])] for item in rows
        )
        latest_rows = [
            item
            for item in rows
            if introduction_order[str(item["object_id"])] == latest_sequence
        ]
        if len(latest_rows) != 1:
            raise ResearchGitError(
                f"DAG has ambiguous latest {kind} objects at one checkpoint"
            )
        return latest_rows[0]

    historical_hypotheses = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "hypothesis"
    )
    effective_hypotheses = {
        object_id: item
        for object_id, item in effective.items()
        if item.get("kind") == "hypothesis"
    }
    # Negative records make a theory contested; they do not silently confer
    # authoritative rejection on the actor that emitted the relation.
    negative_hypothesis_signals: dict[str, set[str]] = defaultdict(set)
    for signal_id, item in effective.items():
        for target in targets(item, set(_NEGATIVE_HYPOTHESIS_RELATIONS)):
            if target in effective_hypotheses:
                negative_hypothesis_signals[target].add(signal_id)
    contested_hypotheses = sorted(
        object_id
        for object_id, item in effective_hypotheses.items()
        if item.get("state") in _TERMINAL_HYPOTHESIS_STATES
        or negative_hypothesis_signals.get(object_id)
    )
    active_hypotheses = sorted(
        set(effective_hypotheses).difference(contested_hypotheses)
    )

    pool_gates = []
    for item in effective.values():
        payload = item.get("payload") or {}
        if (
            item.get("kind") == "gate_decision"
            and isinstance(payload, Mapping)
            and all(
                key in payload
                for key in (
                    "candidate_count",
                    "evaluated_candidate_ids",
                    "experiment_authorized",
                )
            )
        ):
            pool_gates.append(item)
    if any(str(item["object_id"]) not in introduction_order for item in pool_gates):
        raise ResearchGitError(
            "DAG cannot verify Git introduction order for submission candidate gates"
        )
    latest_pool_gates: list[Mapping[str, Any]] = []
    if pool_gates:
        latest_pool_sequence = max(
            introduction_order[str(item["object_id"])] for item in pool_gates
        )
        latest_pool_gates = sorted(
            (
                item
                for item in pool_gates
                if introduction_order[str(item["object_id"])] == latest_pool_sequence
            ),
            key=lambda item: str(item["object_id"]),
        )
    submission_candidate_pool: dict[str, Any] = {
        "status": "unreported",
        "source_gate_ids": [],
        "source_gate_id": None,
        "source_gate_state": None,
        "source_gate_actor_id": None,
        "source_gate_authority": None,
        "source_gate_authoritative": False,
        "candidate_count": None,
        "evaluated_candidate_ids": [],
        "experiment_authorized": None,
        "decision": None,
        "gate_status": None,
    }
    if len(latest_pool_gates) > 1:
        submission_candidate_pool.update(
            {
                "status": "ambiguous",
                "source_gate_ids": [
                    str(item["object_id"]) for item in latest_pool_gates
                ],
            }
        )
    elif latest_pool_gates:
        pool_gate = latest_pool_gates[0]
        pool_payload = pool_gate.get("payload") or {}
        pool_actor = pool_gate.get("actor") or {}
        authority = (
            str(pool_actor.get("authority") or "")
            if isinstance(pool_actor, Mapping)
            else ""
        )
        gate_state = str(pool_gate.get("state") or "")
        submission_candidate_pool.update(
            {
                "status": "reported",
                "source_gate_ids": [str(pool_gate["object_id"])],
                "source_gate_id": str(pool_gate["object_id"]),
                "source_gate_state": gate_state or None,
                "source_gate_actor_id": (
                    (str(pool_actor.get("actor_id") or "") or None)
                    if isinstance(pool_actor, Mapping)
                    else None
                ),
                "source_gate_authority": authority or None,
                "source_gate_authoritative": bool(
                    authority in _SUBMISSION_GATE_AUTHORITIES
                    and gate_state in _DECIDED_GATE_STATES
                ),
                "candidate_count": pool_payload.get("candidate_count"),
                "evaluated_candidate_ids": pool_payload.get(
                    "evaluated_candidate_ids", []
                ),
                "experiment_authorized": pool_payload.get("experiment_authorized"),
                "decision": pool_payload.get("decision") if disclose else None,
                "gate_status": pool_payload.get("status") if disclose else None,
            }
        )
    portfolios = {
        object_id: item
        for object_id, item in effective.items()
        if item.get("kind") == "hypothesis_portfolio"
    }
    portfolios_by_hypothesis: dict[str, set[str]] = {}
    for portfolio_id, portfolio in portfolios.items():
        for member in (portfolio.get("payload") or {}).get("members") or []:
            hypothesis_id = str(member.get("hypothesis_id") or "")
            if hypothesis_id:
                portfolios_by_hypothesis.setdefault(hypothesis_id, set()).add(
                    portfolio_id
                )
    open_anomalies = sorted(
        object_id
        for object_id, item in effective.items()
        if item.get("kind") == "anomaly"
        and (item.get("payload") or {}).get("status") == "open"
    )
    mechanisms = sorted(
        object_id
        for object_id, item in effective.items()
        if item.get("kind") == "mechanism_model"
        and (item.get("payload") or {}).get("status")
        in {"proposed", "tested", "validated"}
    )
    portfolio_frontiers: list[dict[str, Any]] = []
    next_experiments: list[dict[str, Any]] = []
    for portfolio_id in sorted(portfolios):
        priority = latest("experiment_priority", portfolio_id=portfolio_id)
        posterior = latest("posterior_update", portfolio_id=portfolio_id)
        next_experiment = None
        if priority is not None:
            priority_payload = priority.get("payload") or {}
            selected_id = priority_payload.get("selected_candidate_id")
            selected = next(
                (
                    row
                    for row in priority_payload.get("candidate_set") or []
                    if row.get("candidate_id") == selected_id
                ),
                None,
            )
            if selected:
                next_experiment = {
                    "portfolio_id": portfolio_id,
                    "priority_object_id": str(priority["object_id"]),
                    "candidate_id": selected_id,
                    "design_object_id": selected.get("design_object_id"),
                    "summary": selected.get("summary") if disclose else "",
                    "expected_information_gain": selected.get(
                        "expected_information_gain"
                    ),
                    "utility_score": selected.get("utility_score"),
                }
                next_experiments.append(next_experiment)
        portfolio_frontiers.append(
            {
                "portfolio_id": portfolio_id,
                "hypothesis_ids": sorted(
                    str(item.get("hypothesis_id") or "")
                    for item in (portfolios[portfolio_id].get("payload") or {}).get(
                        "members"
                    )
                    or []
                    if item.get("hypothesis_id")
                ),
                "posterior_update_id": (
                    str(posterior["object_id"]) if posterior is not None else None
                ),
                "next_experiment": next_experiment,
            }
        )
    next_experiment = next_experiments[0] if len(next_experiments) == 1 else None
    latest_review = latest("research_review")
    open_questions: list[Any] = []
    if disclose:
        open_questions.extend(
            (latest_review or {}).get("payload", {}).get("gaps") or []
        )
        for item in effective.values():
            if item.get("kind") != "boundary_condition":
                continue
            payload = item.get("payload") or {}
            if payload.get("status") == "untested":
                open_questions.append(
                    {
                        "code": "untested_boundary",
                        "message": (
                            f"{payload.get('dimension')}: {payload.get('condition')}"
                        ),
                        "object_id": item["object_id"],
                    }
                )
    frontier_core = {
        "historical_hypothesis_ids": historical_hypotheses,
        "contested_hypothesis_ids": contested_hypotheses,
        "open_theory_hypothesis_ids": active_hypotheses,
        # Backward-compatible alias. This is a theory status, never the
        # submission-candidate pool.
        "active_hypothesis_ids": active_hypotheses,
        "submission_candidate_pool": submission_candidate_pool,
        "portfolio_ids": sorted(portfolios),
        "prediction_ids": sorted(
            object_id
            for object_id, item in effective.items()
            if item.get("kind") == "discriminating_prediction"
        ),
        "posterior_ids": sorted(
            object_id
            for object_id, item in effective.items()
            if item.get("kind") == "posterior_update"
        ),
        "mechanism_ids": mechanisms,
        "open_anomaly_ids": open_anomalies,
        "open_questions": open_questions,
        "portfolio_frontiers": portfolio_frontiers,
        "next_experiments": next_experiments,
        "next_experiment": next_experiment,
    }
    theory_frontier = {
        **frontier_core,
        "frontier_hash": canonical_content_hash(frontier_core),
    }

    claim_insights: list[dict[str, Any]] = []
    for claim_id, claim in sorted(effective.items()):
        if claim.get("kind") != "claim":
            continue
        direct = targets(claim)
        supporting = sorted(
            {
                object_id
                for object_id in direct
                if effective.get(object_id, {}).get("kind")
                in {"evidence", "passage_evidence", "inference", "evidence_synthesis"}
            }
            | {
                object_id
                for object_id, item in effective.items()
                if claim_id in targets(item, {"supports", "qualified_supports"})
            }
        )
        refuting = sorted(
            object_id
            for object_id, item in effective.items()
            if claim_id
            in targets(
                item,
                {"refutes", "qualified_refutes", "contradicts", "challenges_inference"},
            )
        )
        mechanism_ids = sorted(
            object_id
            for object_id in direct
            if effective.get(object_id, {}).get("kind") == "mechanism_model"
        )
        quality_ids = sorted(
            object_id
            for object_id, item in effective.items()
            if item.get("kind") == "evidence_quality"
            and set(supporting).intersection(targets(item, {"evaluates"}))
        )
        boundary_ids = sorted(
            {
                object_id
                for object_id, item in effective.items()
                if item.get("kind") in {"boundary_condition", "transfer_matrix"}
                and claim_id in targets(item)
            }
            | {
                object_id
                for object_id in direct
                if effective.get(object_id, {}).get("kind")
                in {"boundary_condition", "transfer_matrix"}
            }
        )
        depth_level = str(
            (claim.get("payload") or {}).get("depth_level") or "descriptive"
        )
        supporting_set = set(supporting)
        valid_mechanisms = [
            object_id
            for object_id in mechanism_ids
            if effective[object_id].get("state") == "verified"
            and (effective[object_id].get("payload") or {}).get("status") == "validated"
            and (effective[object_id].get("payload") or {}).get("validation")
            and supporting_set.intersection(
                (effective[object_id].get("payload") or {}).get("evidence_ids") or []
            )
        ]
        valid_quality = [
            object_id
            for object_id in quality_ids
            if effective[object_id].get("state") == "verified"
            and (effective[object_id].get("payload") or {}).get("independent") is True
            and (effective[object_id].get("payload") or {}).get("independence_receipt")
            and (effective[object_id].get("payload") or {}).get("overall_grade")
            in {"strong", "moderate"}
        ]
        claim_payload = claim.get("payload") or {}
        valid_transfer = []
        for object_id in boundary_ids:
            item = effective[object_id]
            if item.get("kind") != "transfer_matrix":
                continue
            matrix_payload = item.get("payload") or {}
            matrix_claim = effective.get(str(matrix_payload.get("claim_id") or ""), {})
            matrix_claim_payload = matrix_claim.get("payload") or {}
            if (
                item.get("state") == "verified"
                and matrix_payload.get("transfer_ready") is True
                and " ".join(str(matrix_claim_payload.get("statement") or "").split())
                == " ".join(str(claim_payload.get("statement") or "").split())
                and matrix_claim_payload.get("scope_hash")
                == claim_payload.get("scope_hash")
            ):
                valid_transfer.append(object_id)
        gaps = []
        if not supporting:
            gaps.append("supporting_evidence_missing")
        if depth_level in {"causal", "transferable"} and not valid_mechanisms:
            gaps.append("validated_mechanism_missing")
        if depth_level in {"causal", "transferable"} and not valid_quality:
            gaps.append("evidence_quality_missing")
        if depth_level == "transferable" and not valid_transfer:
            gaps.append("transfer_matrix_missing")
        hypothesis_ids = {
            target
            for target in direct
            if effective.get(target, {}).get("kind") == "hypothesis"
        }
        for evidence_id in (*supporting, *refuting):
            hypothesis_ids.update(
                target
                for target in targets(effective.get(evidence_id, {}))
                if effective.get(target, {}).get("kind") == "hypothesis"
            )
        for mechanism_id in mechanism_ids:
            target = str(
                (effective[mechanism_id].get("payload") or {}).get(
                    "target_hypothesis_id"
                )
                or ""
            )
            if target:
                hypothesis_ids.add(target)
        scoped_portfolios = sorted(
            {
                portfolio_id
                for hypothesis_id in hypothesis_ids
                for portfolio_id in portfolios_by_hypothesis.get(hypothesis_id, set())
            }
        )
        claim_next = [
            item
            for item in next_experiments
            if item["portfolio_id"] in scoped_portfolios
        ]
        claim_insights.append(
            {
                "claim_id": claim_id,
                "depth_level": depth_level,
                "supporting_ids": supporting,
                "refuting_ids": refuting,
                "mechanism_ids": mechanism_ids,
                "quality_assessment_ids": quality_ids,
                "boundary_ids": boundary_ids,
                "portfolio_ids": scoped_portfolios,
                "next_experiments": claim_next,
                "next_experiment": claim_next[0] if len(claim_next) == 1 else None,
                "gaps": gaps,
                "decision_ready": not gaps and not refuting,
            }
        )
    strategy_summary = {
        "profile": "xscientist.deep-research-strategy.v1",
        "layer_counts": dict(
            sorted(
                Counter(
                    _epistemic_layer(str(item.get("kind") or ""))
                    for item in effective.values()
                ).items()
            )
        ),
        "review_due": bool(
            not latest_review
            or (latest_review.get("payload") or {}).get("review_due") is True
        ),
        "latest_review_id": (
            str(latest_review["object_id"]) if latest_review is not None else None
        ),
        "open_anomaly_count": len(open_anomalies),
        "historical_hypothesis_count": len(historical_hypotheses),
        "contested_hypothesis_count": len(contested_hypotheses),
        "open_theory_hypothesis_count": len(active_hypotheses),
        "submission_candidate_count": submission_candidate_pool["candidate_count"],
        "experiment_authorized": submission_candidate_pool["experiment_authorized"],
        "claim_depth_counts": dict(
            sorted(Counter(item["depth_level"] for item in claim_insights).items())
        ),
    }
    return theory_frontier, claim_insights, strategy_summary


def _has_hash_anchor(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key.endswith("_hash") and _is_sha256(value):
                return True
            if (
                key.endswith("_hashes")
                and isinstance(value, list)
                and bool(value)
                and all(_is_sha256(item) for item in value)
            ):
                return True
            if _has_hash_anchor(value):
                return True
    elif isinstance(payload, list):
        return any(_has_hash_anchor(item) for item in payload)
    return False


def _check(code: str, label: str, passed: bool, layer: str) -> dict[str, Any]:
    return {"code": code, "label": label, "passed": bool(passed), "layer": layer}


def _read_json_lines(raw: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    record_count = 0
    for line_number, line in enumerate((raw or "").splitlines(), start=1):
        if not line.strip():
            continue
        record_count += 1
        if record_count > _MAX_ARA_MANIFEST_REVISIONS:
            errors.append(
                "manifest history exceeds "
                f"{_MAX_ARA_MANIFEST_REVISIONS} non-empty revisions"
            )
            break
        try:
            payload = _strict_json_loads(line)
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            errors.append(f"manifest history line {line_number} is invalid: {message}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"manifest history line {line_number} is not an object")
            continue
        rows.append(payload)
    return rows, errors


def _manifest_revision_state(
    manifest: Mapping[str, Any] | None,
    lock: Mapping[str, Any] | None,
    history_rows: Sequence[Mapping[str, Any]],
    *,
    expected_hash: str | None = None,
    parse_errors: Sequence[str] = (),
    archived_manifests: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Validate an ARA manifest revision chain without depending on a worktree.

    Research DAGs may be built for an old Git ref, so local-only verification is
    insufficient.  This projection validates the committed manifest, immutable
    base lock, and append-only history in memory and exposes every legitimate
    revision hash for provenance matching.
    """

    issues = list(parse_errors)
    base_hash = str((lock or {}).get("manifest_hash") or "") or None
    current_hash = hash_manifest(dict(manifest)) if manifest is not None else None
    valid_hashes: list[str] = []
    if lock is not None:
        try:
            validate_json(dict(lock), load_schema("manifest_lock"))
        except ValidationError as exc:
            issues.append(f"manifest lock schema is invalid: {exc.message}")
        if lock.get("hasher") not in {None, "hash_manifest.v1"}:
            issues.append(f"unsupported manifest hasher: {lock.get('hasher')}")
    if _is_sha256(base_hash):
        valid_hashes.append(str(base_hash))
    elif lock is not None:
        issues.append("manifest lock has an invalid base hash")

    expected_base = base_hash
    for expected_revision, row in enumerate(history_rows, start=1):
        try:
            validate_json(dict(row), load_schema("manifest_revision"))
        except ValidationError as exc:
            issues.append(
                f"manifest revision {expected_revision} schema is invalid: {exc.message}"
            )
        revision = row.get("revision")
        row_base = str(row.get("base_hash") or "")
        new_hash = str(row.get("new_hash") or "")
        if revision != expected_revision:
            issues.append(
                f"manifest revision {expected_revision} is numbered {revision}"
            )
        if expected_base and row_base != expected_base:
            issues.append(
                f"manifest revision {expected_revision} does not extend the previous hash"
            )
        if not _is_sha256(row_base) or not _is_sha256(new_hash):
            issues.append(f"manifest revision {expected_revision} has an invalid hash")
        elif new_hash not in valid_hashes:
            valid_hashes.append(new_hash)
        if archived_manifests is not None and _is_sha256(row_base):
            archived = archived_manifests.get(row_base)
            if archived is None:
                issues.append(
                    f"manifest revision {expected_revision} is missing its base snapshot"
                )
            elif hash_manifest(dict(archived)) != row_base:
                issues.append(
                    f"manifest revision {expected_revision} base snapshot hash mismatch"
                )
        expected_base = new_hash or expected_base

    if manifest is None:
        state = "legacy_lock_only" if _is_sha256(base_hash) else "unlocked"
        if state == "unlocked":
            issues.append("manifest.json is missing")
        else:
            issues.append(
                "manifest.json is unavailable; using a legacy lock-only binding"
            )
    elif lock is None:
        state = "unlocked"
        issues.append("manifest.lock is missing")
    elif issues:
        state = "tampered"
    elif not history_rows and current_hash == base_hash:
        state = "clean"
    elif history_rows and current_hash == expected_base:
        state = "revised"
    else:
        state = "tampered"
        issues.append("current manifest hash is outside the declared revision chain")

    if (
        current_hash
        and current_hash not in valid_hashes
        and state in {"clean", "revised"}
    ):
        valid_hashes.append(current_hash)
    if expected_hash and current_hash != expected_hash:
        issues.append(
            "checkpoint manifest hash does not match the manifest stored at this ref"
        )
        state = "tampered"
    if state not in {"clean", "revised", "legacy_lock_only"}:
        valid_hashes = []
    return {
        "state": state,
        "ok": state in {"clean", "revised", "legacy_lock_only"},
        "base_hash": base_hash,
        "current_hash": current_hash,
        "expected_hash": expected_hash,
        "revision_count": len(history_rows),
        "valid_hashes": valid_hashes,
        "issues": issues,
    }


def _blob_bytes_at_commit(
    repo: Path,
    commit: str,
    path: str,
    *,
    required: bool = False,
    maximum: int = _MAX_ARA_NODE_DETAIL_BYTES,
    read_budget: dict[str, Any] | None = None,
) -> bytes | None:
    normalized = _normalise_relative(path)
    if read_budget is not None:
        if read_budget["files"] >= read_budget["maximum_files"]:
            raise ResearchGitError(
                f"{read_budget['label']} exceeds "
                f"{read_budget['maximum_files']} file reads"
            )
        read_budget["files"] += 1
    tree = _run_git(
        repo,
        ["ls-tree", "-z", commit, "--", normalized],
        check=False,
    )
    if tree.returncode or not tree.stdout:
        if required:
            raise ResearchGitError(
                f"ARA snapshot path is missing at {commit}: {normalized}"
            )
        return None

    blob_id = None
    for record in tree.stdout.split("\0"):
        if not record or "\t" not in record:
            continue
        metadata, returned_path = record.split("\t", 1)
        fields = metadata.split()
        if (
            returned_path == normalized
            and len(fields) == 3
            and fields[0] in {"100644", "100755"}
            and fields[1] == "blob"
            and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", fields[2])
        ):
            blob_id = fields[2]
            break
    if blob_id is None:
        if required:
            raise ResearchGitError(
                f"ARA snapshot path is not a regular blob at {commit}: {normalized}"
            )
        return None

    size_result = _run_git(repo, ["cat-file", "-s", blob_id], check=False)
    try:
        blob_size = int(size_result.stdout.strip())
    except (TypeError, ValueError):
        blob_size = -1
    if size_result.returncode or blob_size < 0:
        raise ResearchGitError(
            f"cannot inspect ARA snapshot path at {commit}: {normalized}"
        )
    if blob_size > maximum:
        raise ResearchGitError(
            f"ARA snapshot path exceeds {maximum} bytes: {normalized}"
        )
    if read_budget is not None:
        remaining = read_budget["maximum_bytes"] - read_budget["bytes"]
        if blob_size > remaining:
            raise ResearchGitError(
                f"{read_budget['label']} exceeds "
                f"{read_budget['maximum_bytes']} aggregate bytes"
            )
        read_budget["bytes"] += blob_size
    try:
        blob = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "blob", blob_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError as exc:
        raise ResearchGitError(
            f"cannot read ARA snapshot path at {commit}: {normalized}"
        ) from exc
    if blob.returncode or len(blob.stdout) != blob_size:
        raise ResearchGitError(
            f"cannot read ARA snapshot path at {commit}: {normalized}"
        )
    return blob.stdout


def _json_blob_at_commit(
    repo: Path,
    commit: str,
    path: str,
    *,
    required: bool = False,
    maximum: int = _MAX_ARA_NODE_DETAIL_BYTES,
    read_budget: dict[str, Any] | None = None,
) -> Any | None:
    normalized = _normalise_relative(path)
    encoded = _blob_bytes_at_commit(
        repo,
        commit,
        normalized,
        required=required,
        maximum=maximum,
        read_budget=read_budget,
    )
    if encoded is None:
        return None
    try:
        return _strict_json_bytes(encoded)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ResearchGitError(
            f"ARA snapshot JSON is invalid at {commit}:{normalized}"
        ) from exc


def _text_blob_at_commit(
    repo: Path,
    commit: str,
    path: str,
    *,
    maximum: int = _MAX_ARA_NODE_DETAIL_BYTES,
    read_budget: dict[str, Any] | None = None,
) -> str | None:
    normalized = _normalise_relative(path)
    encoded = _blob_bytes_at_commit(
        repo,
        commit,
        normalized,
        maximum=maximum,
        read_budget=read_budget,
    )
    if encoded is None:
        return None
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchGitError(
            f"ARA snapshot text is invalid at {commit}:{normalized}"
        ) from exc


def _strict_pointer_records_at_commit(
    repo: Path,
    commit: str,
) -> dict[str, list[dict[str, Any]]]:
    tree = _run_git_bounded(
        repo,
        [
            "ls-tree",
            "-r",
            "-z",
            "--name-only",
            commit,
            "--",
            "research-objects",
        ],
        max_output_bytes=_MAX_ARTIFACT_POINTER_ENUMERATION_BYTES,
    )
    tree_paths = _nul_terminated_git_paths(
        tree.stdout, label="research object pointer enumeration"
    )
    pointer_paths = sorted(
        path
        for path in tree_paths
        if path.startswith("research-objects/") and path.endswith(".json")
    )
    if len(pointer_paths) > _MAX_ARTIFACT_POINTER_FILES:
        raise ResearchGitError(
            f"research object pointers exceed {_MAX_ARTIFACT_POINTER_FILES} files"
        )
    records: dict[str, list[dict[str, Any]]] = {}
    logical_bindings: dict[tuple[str, ...], tuple[str, str]] = {}
    pointer_budget: dict[str, Any] = {
        "bytes": 0,
        "files": 0,
        "maximum_bytes": _MAX_ARTIFACT_POINTER_TOTAL_BYTES,
        "maximum_files": _MAX_ARTIFACT_POINTER_FILES,
        "label": "research object pointer metadata",
    }
    for pointer_path in pointer_paths:
        encoded = _blob_bytes_at_commit(
            repo,
            commit,
            pointer_path,
            required=True,
            maximum=_MAX_ARTIFACT_POINTER_BYTES,
            read_budget=pointer_budget,
        )
        assert encoded is not None
        try:
            payload = _strict_json_bytes(encoded)
            payload = _validate_pointer_payload(payload, pointer_path=pointer_path)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
            ResearchGitError,
        ) as exc:
            raise ResearchGitError(
                f"cannot validate research object pointer at {commit}:{pointer_path}"
            ) from exc
        object_hash = str(payload.get("object_hash") or "")
        if not _is_sha256(object_hash):
            raise ResearchGitError(
                f"research object pointer has an invalid hash: {pointer_path}"
            )
        _record_pointer(
            records,
            logical_bindings,
            object_hash=object_hash,
            record={**payload, "pointer_path": pointer_path},
        )
    return records


def _targets(
    item: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    kinds: Iterable[str] = (),
    relations: Iterable[str] = (),
) -> list[str]:
    selected_kinds = set(kinds)
    selected_relations = set(relations)
    return sorted(
        {
            str(relation.get("target"))
            for relation in item.get("relations") or []
            if (
                (not selected_relations or relation.get("type") in selected_relations)
                and str(relation.get("target")) in objects
                and (
                    not selected_kinds
                    or objects[str(relation.get("target"))].get("kind")
                    in selected_kinds
                )
            )
        }
    )


def _object_proof(
    item: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    claim_closure: Mapping[str, Mapping[str, Any]],
    challenge_targets: set[str],
    closure_blockers: Mapping[str, list[str]],
    closure_warnings: Mapping[str, list[str]],
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    kind = str(item["kind"])
    state = str(item["state"])
    payload = item.get("payload") or {}
    provenance = item.get("provenance") or {}
    actor = item.get("actor") or {}
    checks = [
        _check(
            "content_addressed", "Content hash and object ID validate", True, "trace"
        )
    ]
    from ai_scientist.protocol.research_vcs import research_profile_status

    profile_status = research_profile_status(item)

    if kind == "experiment_attempt":
        checks.extend(
            [
                _check(
                    "plan_bound",
                    "Experiment is bound to a research plan",
                    bool(_targets(item, objects, kinds={"research_plan"})),
                    "trace",
                ),
                _check(
                    "environment_bound",
                    "Environment identity is recorded",
                    bool(provenance.get("environment_hash")),
                    "replay",
                ),
                _check(
                    "dependencies_bound",
                    "Dependency lock identity is recorded",
                    bool(provenance.get("dependency_lock_hashes")),
                    "replay",
                ),
                _check(
                    "code_bound",
                    "Code identity is recorded",
                    bool(
                        provenance.get("code_hash")
                        or provenance.get("code_commit")
                        or payload.get("code_ref")
                    ),
                    "replay",
                ),
                _check(
                    "data_bound",
                    "Dataset identity is recorded",
                    bool(provenance.get("dataset_hashes")),
                    "replay",
                ),
            ]
        )
    elif kind == "search_plan":
        checks.extend(
            [
                _check(
                    "search_plan_locked",
                    "Literature queries and criteria were locked",
                    state == "locked",
                    "trace",
                ),
                _check(
                    "search_plan_bound",
                    "Search plan has an immutable commitment",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "search_receipt":
        checks.extend(
            [
                _check(
                    "search_plan_bound",
                    "Retrieval receipt points to its locked search plan",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"search_plan"},
                            relations={"depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "retrieval_bound",
                    "Retrieval candidates have an immutable receipt hash",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "source_snapshot":
        checks.extend(
            [
                _check(
                    "search_receipt_bound",
                    "Source selection points to its retrieval receipt",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"search_receipt"},
                            relations={"derived_from"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "source_content_bound",
                    "Source content has an immutable identity",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "passage_evidence":
        checks.extend(
            [
                _check(
                    "source_snapshot_bound",
                    "Passage points to an immutable source snapshot",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"source_snapshot"},
                            relations={"quotes"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "passage_bound",
                    "Passage quote and locator have immutable identities",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "inference":
        checks.extend(
            [
                _check(
                    "premise_bound",
                    "Inference names its evidence or proposition premises",
                    bool(
                        _targets(
                            item,
                            objects,
                            relations={"has_premise", "depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "inference_bound",
                    "Inference payload has an immutable semantic anchor",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind in {"warrant", "assumption", "method"}:
        checks.append(
            _check(
                "semantic_anchor",
                "Argument component has an immutable semantic anchor",
                _has_hash_anchor(payload),
                "replay",
            )
        )
    elif kind == "effect_estimate":
        checks.extend(
            [
                _check(
                    "estimand_bound",
                    "Effect estimate identifies its estimand",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"estimand"},
                            relations={"addresses_estimand", "depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "uncertainty_recorded",
                    "Effect estimate records interval, standard error, or posterior",
                    any(
                        payload.get(key) not in (None, "", [], {})
                        for key in (
                            "confidence_interval",
                            "credible_interval",
                            "standard_error",
                            "posterior",
                        )
                    ),
                    "replay",
                ),
            ]
        )
    elif kind == "source_update":
        checks.extend(
            [
                _check(
                    "source_bound",
                    "Status update points to an immutable source snapshot",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"source_snapshot"},
                            relations={"updates", "invalidates"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "source_update_bound",
                    "Status update has an immutable provider receipt",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif (
        kind == "experiment_design"
        and payload.get("protocol_kind") == "method_discovery_contract"
    ):
        conditions = payload.get("conditions") or []
        baselines = payload.get("baselines") or []
        checks.extend(
            [
                _check(
                    "discovery_contract_locked",
                    "Method-discovery scope was locked before evaluation",
                    state == "locked" and _has_hash_anchor(payload),
                    "trace",
                ),
                _check(
                    "target_variable_isolated",
                    "Allowed and protected edit scopes isolate the target variable",
                    bool((payload.get("edit_scope") or {}).get("allowed_paths"))
                    and bool((payload.get("edit_scope") or {}).get("protected_paths")),
                    "replay",
                ),
                _check(
                    "strong_comparators_bound",
                    "At least three strong comparison methods are committed",
                    len(baselines) >= 3
                    and sum(
                        isinstance(row, Mapping) and row.get("strong") is True
                        for row in baselines
                    )
                    >= 3,
                    "replay",
                ),
                _check(
                    "generalization_conditions_bound",
                    "Multiple conditions include sealed transfer, heldout, or scale evaluation",
                    len(conditions) >= 3
                    and any(
                        isinstance(row, Mapping)
                        and row.get("visibility") == "sealed"
                        and row.get("role") in {"transfer", "heldout", "scale"}
                        for row in conditions
                    ),
                    "replay",
                ),
                _check(
                    "budget_and_blinding_bound",
                    "Resource budget and feedback blinding are explicit DAG parents",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"resource_budget"},
                            relations={"depends_on"},
                        )
                    )
                    and bool(
                        _targets(
                            item,
                            objects,
                            kinds={"evaluation_blinding"},
                            relations={"depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "discovery_context_bound",
                    "Method selection is bound to the exact evidence and memory context",
                    (
                        bool(
                            _targets(
                                item,
                                objects,
                                kinds={"context_snapshot"},
                                relations={"uses_context"},
                            )
                        )
                        if payload.get("context_required") is True
                        else True
                    ),
                    "trace",
                ),
            ]
        )
    elif (
        kind == "resource_budget"
        and payload.get("protocol_kind") == "method_discovery_budget"
    ):
        checks.extend(
            [
                _check(
                    "resource_budget_locked",
                    "Resource limits were locked before candidate evaluation",
                    state == "locked" and bool(payload.get("limits")),
                    "trace",
                ),
                _check(
                    "information_value_policy",
                    "Adaptive compute must be allocated by expected information value",
                    payload.get("information_value_required") is True,
                    "replay",
                ),
            ]
        )
    elif (
        kind == "evaluation_blinding"
        and payload.get("protocol_kind") == "method_discovery_blinding"
    ):
        checks.extend(
            [
                _check(
                    "sealed_feedback_locked",
                    "Held-out feedback remains sealed until final candidate commitment",
                    state == "locked"
                    and payload.get("leakage_prohibited") is True
                    and bool(payload.get("sealed_condition_ids")),
                    "trace",
                ),
                _check(
                    "blinding_bound",
                    "Blinding policy has an immutable commitment",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif (
        kind == "evidence_synthesis"
        and payload.get("protocol_kind") == "generalization_assessment"
    ):
        checks.extend(
            [
                _check(
                    "discovery_contract_bound",
                    "Assessment points to its locked method-discovery contract",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"experiment_design"},
                            relations={"depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "condition_evidence_bound",
                    "Assessment is derived from recorded condition evidence",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={
                                "evidence",
                                "effect_estimate",
                                "passage_evidence",
                            },
                            relations={"derived_from"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "generalization_protocol_valid",
                    "Every scope, resource, baseline, condition, and proxy check passed",
                    all(
                        isinstance(row, Mapping) and row.get("passed") is True
                        for row in payload.get("checks") or []
                    ),
                    "replay",
                ),
                _check(
                    "method_generalizes",
                    "Candidate improves sealed generalization conditions, not only development",
                    payload.get("verdict") == "method_discovery_supported",
                    "replay",
                ),
            ]
        )
    elif kind in {
        "action_proposal",
        "stopping_decision",
        "novelty_check",
        "human_escalation",
        "context_robustness",
    }:
        checks.append(
            _check(
                "decision_context_bound",
                "Autonomous decision is bound to an exact context snapshot",
                bool(
                    _targets(
                        item,
                        objects,
                        kinds={"context_snapshot"},
                        relations={"uses_context", "depends_on"},
                    )
                ),
                "trace",
            )
        )
    elif kind == "evidence":
        checks.extend(
            [
                _check(
                    "attempt_bound",
                    "Evidence is derived from a recorded attempt",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"experiment_attempt"},
                            relations={"derived_from", "depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "measurement_bound",
                    "Evidence has an immutable measurement anchor",
                    _has_hash_anchor(payload),
                    "replay",
                ),
                _check(
                    "independently_verified",
                    "Independent verifier accepted the evidence",
                    state == "verified"
                    and actor.get("authority") == "independent_evaluator",
                    "verify",
                ),
            ]
        )
    elif kind == "review":
        checks.append(
            _check(
                "independent_authority",
                "Review was produced by an independent evaluator",
                state == "verified"
                and actor.get("authority") == "independent_evaluator",
                "verify",
            )
        )
    elif kind == "gate_decision":
        checks.append(
            _check(
                "deterministic_authority",
                "Gate was produced by deterministic policy",
                state in {"verified", "promoted"}
                and actor.get("authority") == "deterministic_gate",
                "verify",
            )
        )
        deployment = payload.get("deployment_receipt")
        if isinstance(deployment, Mapping):
            from ai_scientist.utils.evolution_deployment import (
                validate_deployment_receipt,
            )

            checks.append(
                _check(
                    "deployment_receipt",
                    "Production deployment receipt validates",
                    validate_deployment_receipt(deployment)["ok"],
                    "verify",
                )
            )
    elif kind == "reproduction":
        checks.extend(
            [
                _check(
                    "receipt_bound",
                    "Reproduction contains a content-addressed receipt",
                    _has_hash_anchor(payload),
                    "replay",
                ),
                _check(
                    "reproduction_verified",
                    "Independent reproduction passed",
                    state == "verified",
                    "verify",
                ),
            ]
        )
    elif kind == "agent_candidate":
        candidate = payload.get("candidate") or payload
        checks.append(
            _check(
                "candidate_artifact_bound",
                "Evolution candidate is bound to an immutable artifact",
                _is_sha256(candidate.get("candidate_artifact_hash"))
                or _is_sha256(candidate.get("candidate_hash")),
                "replay",
            )
        )
    elif kind == "agent_evaluation":
        checks.append(
            _check(
                "independent_evaluation",
                "Evolution evaluation has independent authority",
                state == "verified"
                and actor.get("authority") == "independent_evaluator",
                "verify",
            )
        )
    elif kind == "claim":
        closure = claim_closure.get(object_id)
        checks.extend(
            [
                _check(
                    "claim_trace",
                    "Claim is linked to evidence, attempt, and plan",
                    bool(closure and closure.get("trace_complete")),
                    "trace",
                ),
                _check(
                    "claim_replay",
                    "Claim lineage has sufficient replay identities",
                    bool(closure and closure.get("replay_ready")),
                    "replay",
                ),
                _check(
                    "claim_verify",
                    "Claim has independent gate and reproduction",
                    bool(closure and closure.get("verified")),
                    "verify",
                ),
            ]
        )

    layer_pass = {
        layer: all(check["passed"] for check in checks if check["layer"] == layer)
        for layer in ("trace", "replay", "verify")
    }
    contested = object_id in challenge_targets
    if contested:
        level = "contested"
    elif (
        checks
        and layer_pass["verify"]
        and any(check["layer"] == "verify" for check in checks)
    ):
        level = "verified"
    elif (
        layer_pass["trace"]
        and layer_pass["replay"]
        and any(check["layer"] == "replay" for check in checks)
    ):
        level = "replayable"
    elif layer_pass["trace"]:
        level = "traceable"
    else:
        level = "recorded"
    blockers = [check["code"] for check in checks if not check["passed"]]
    blockers.extend(closure_blockers.get(object_id, []))
    proof_warnings = list(closure_warnings.get(object_id, []))
    if profile_status.get("declared") and not profile_status.get("validator_available"):
        blockers.append("profile_validator_unavailable")
        proof_warnings.append(
            "semantic profile is preserved but has no trusted local validator"
        )
        if level == "verified":
            level = "replayable" if layer_pass["replay"] else "traceable"
    if kind == "context_snapshot":
        from .research_context import research_context_issues

        context_issues = research_context_issues(payload, objects=objects)
        context_checks = [
            _check(
                "context_identity_valid",
                "Context, source closure, policy, and memory hashes validate",
                not context_issues,
                "trace",
            ),
            _check(
                "context_complete",
                "Context snapshot retained its required decision closure",
                payload.get("complete") is True,
                "replay",
            ),
        ]
        checks.extend(context_checks)
        blockers.extend(
            check["code"] for check in context_checks if not check["passed"]
        )
        proof_warnings.extend(context_issues)
        if not context_issues and payload.get("complete") is True:
            level = "replayable"
        elif not context_issues:
            level = "traceable"
        else:
            level = "recorded"
    if kind in {"review", "gate_decision", "agent_evaluation"}:
        from .research_context import research_context_issues

        context_ids = _targets(
            item,
            objects,
            kinds={"context_snapshot"},
            relations={"depends_on"},
        )
        required_context = payload.get("context_required") is True
        valid_context = False
        for context_id in context_ids:
            context_payload = objects[context_id].get("payload") or {}
            if context_payload.get("context_hash") == payload.get(
                "context_hash"
            ) and not research_context_issues(context_payload, objects=objects):
                valid_context = True
                break
        context_check = _check(
            "decision_context_bound",
            "Decision is bound to its exact evidence and memory snapshot",
            valid_context if required_context else True,
            "trace",
        )
        checks.append(context_check)
        if not context_check["passed"]:
            blockers.append(context_check["code"])
            level = "recorded"
        elif not context_ids:
            proof_warnings.append("legacy_decision_context_unbound")
    return {
        "level": level,
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(proof_warnings)),
        "contested": contested,
    }


def _project_ara_snapshot(
    graph: Mapping[str, Any],
    *,
    index: int,
    disclose: bool,
    node_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    manifest: Mapping[str, Any] | None = None,
    lock: Mapping[str, Any] | None = None,
    history_rows: Sequence[Mapping[str, Any]] = (),
    history_errors: Sequence[str] = (),
    archived_manifests: Mapping[str, Mapping[str, Any] | None] | None = None,
    expected_manifest_hash: str | None = None,
    expected_graph_hash: str | None = None,
    repository_path: str | None = None,
    snapshot_ref: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    analysis = analyze_exploration_graph(graph)
    prefix = f"ara:{index}:"
    nodes: list[dict[str, Any]] = []
    node_ids: dict[str, str] = {}
    detail_views: dict[str, Any] = {}
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping) or not str(raw.get("id") or ""):
            continue
        raw_id = str(raw["id"])
        node_ids[raw_id] = prefix + raw_id
        summary = (
            " ".join(str(raw.get("analysis") or raw.get("stage") or raw_id).split())
            if disclose
            else "experiment node"
        )
        node_hash = raw.get("content_hash")
        valid_node_hash = _is_sha256(node_hash)
        isolated = bool((raw.get("execution_isolation") or {}).get("isolated"))
        completed = not bool(raw.get("is_buggy"))
        nodes.append(
            {
                "id": prefix + raw_id,
                "source": "ara",
                "kind": "experiment_node",
                "state": "completed" if completed else "failed",
                "phase": _PHASES["experiment_node"],
                "summary": summary[:160],
                "content_hash": str(node_hash) if valid_node_hash else None,
                "actor": None,
                "created_at": None,
                "proof": {
                    "level": (
                        "replayable"
                        if valid_node_hash and isolated
                        else "traceable" if valid_node_hash else "recorded"
                    ),
                    "checks": [
                        _check(
                            "node_content_addressed",
                            "Experiment node has a content hash",
                            valid_node_hash,
                            "trace",
                        ),
                        _check(
                            "execution_isolated",
                            "Experiment executed in an isolated backend",
                            isolated,
                            "replay",
                        ),
                    ],
                    "blockers": [
                        code
                        for code, passed in (
                            ("node_content_addressed", valid_node_hash),
                            ("execution_isolated", isolated),
                        )
                        if not passed
                    ],
                    "warnings": [],
                    "contested": False,
                },
                "source_ref": snapshot_ref or f"ara:{index}",
            }
        )
        if disclose:
            companion_record = (node_metrics or {}).get(raw_id) or {}
            if not isinstance(companion_record, Mapping):
                companion_record = {}
            source_snapshot = companion_record.get(_ARA_SOURCE_SNAPSHOT_KEY)
            snapshot_verified = _valid_source_snapshot(source_snapshot)
            companion = {
                key: value
                for key, value in companion_record.items()
                if key != _ARA_SOURCE_SNAPSHOT_KEY
            }
            declared_hash_matches = bool(
                valid_node_hash
                and companion.get("content_hash") == node_hash
                and snapshot_verified
            )
            legacy_unhashed = raw.get("content_hash") in (None, "")
            raw_metric = next(
                (
                    value
                    for value in (raw.get("metrics"), raw.get("metric"))
                    if value not in (None, "", [], {})
                ),
                None,
            )
            companion_metric = next(
                (
                    value
                    for value in (companion.get("metrics"), companion.get("metric"))
                    if value not in (None, "", [], {})
                ),
                None,
            )
            metric_matches_graph = bool(
                declared_hash_matches
                and raw_metric not in (None, "", [], {})
                and companion_metric not in (None, "", [], {})
                and _json_values_identical(companion_metric, raw_metric)
            )
            # A sidecar's self-declared content hash cannot authenticate fields
            # absent from the graph. Only legacy unhashed nodes expose such fields,
            # and those are explicitly labelled as snapshot-only below.
            field_sources = (
                (raw,)
                if not legacy_unhashed
                else ((companion, raw) if snapshot_verified else (raw,))
            )
            fields = {
                key: next(
                    value
                    for source in field_sources
                    for value in (source.get(key),)
                    if value not in (None, "", [], {})
                )
                for key in (
                    "scientific_outcome",
                    "analysis",
                    "summary",
                    "plan_excerpt",
                )
                if any(
                    source.get(key) not in (None, "", [], {})
                    for source in field_sources
                )
            }
            metric = (
                raw_metric
                if not legacy_unhashed
                else companion_metric if snapshot_verified else raw_metric
            )
            if fields or metric not in (None, "", [], {}):
                detail_views[prefix + raw_id] = {
                    "title": "ARA experiment detail",
                    "fields": fields,
                    "metrics": (
                        [
                            {
                                "source_id": prefix + raw_id,
                                "source_kind": "experiment_node",
                                "values": metric,
                            }
                        ]
                        if metric not in (None, "", [], {})
                        else []
                    ),
                    "detail_binding": {
                        "state": (
                            "metric_verified"
                            if metric_matches_graph
                            else (
                                "graph_only"
                                if not legacy_unhashed
                                else (
                                    "snapshot_only"
                                    if snapshot_verified
                                    else "graph_only"
                                )
                            )
                        ),
                        "snapshot_verified": snapshot_verified,
                        "declared_node_hash_matches": declared_hash_matches,
                        "metric_matches_graph": metric_matches_graph,
                    },
                    **(
                        {"source_snapshot": dict(source_snapshot)}
                        if snapshot_verified
                        else {}
                    ),
                }
    context_node_ids: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping):
            continue
        target = node_ids.get(str(raw.get("id") or ""))
        if not target:
            continue
        for context_ref in sorted(set(raw.get("context_pack_refs") or [])):
            if not _is_sha256(context_ref):
                continue
            context_id = context_node_ids.setdefault(
                str(context_ref),
                prefix + "context:" + str(context_ref).split(":", 1)[1][:16],
            )
            if not any(node["id"] == context_id for node in nodes):
                nodes.append(
                    {
                        "id": context_id,
                        "source": "ara",
                        "kind": "context_snapshot",
                        "state": "completed",
                        "phase": _PHASES["research_plan"],
                        "summary": "ARA context/memory pack " + str(context_ref)[7:19],
                        "content_hash": str(context_ref),
                        "actor": None,
                        "created_at": None,
                        "proof": {
                            "level": "traceable",
                            "checks": [
                                _check(
                                    "context_pack_addressed",
                                    "Consumed context pack has an immutable identity",
                                    True,
                                    "trace",
                                )
                            ],
                            "blockers": [],
                            "warnings": [
                                "context payload remains in the ARA object store"
                            ],
                            "contested": False,
                        },
                        "source_ref": snapshot_ref or f"ara:{index}",
                    }
                )
            edges.append(
                {
                    "source": context_id,
                    "target": target,
                    "type": "contextualizes",
                    "role": "consumed context and memory",
                    "category": "context",
                }
            )
    valid_ids = {node["id"] for node in nodes}
    raw_edges = graph.get("edges") or []
    seen: set[tuple[str, str]] = set()
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            continue
        source, target = prefix + str(raw.get("parent")), prefix + str(raw.get("child"))
        if source in valid_ids and target in valid_ids:
            seen.add((source, target))
    if graph.get("topology_encoding") != "edges":
        for raw in graph.get("nodes") or []:
            if not isinstance(raw, Mapping):
                continue
            parent = prefix + str(raw.get("id"))
            parent_id = raw.get("parent_id")
            if parent_id:
                mirror_parent = prefix + str(parent_id)
                if mirror_parent in valid_ids and parent in valid_ids:
                    seen.add((mirror_parent, parent))
            children = raw.get("children") or []
            if isinstance(children, list):
                for child_id in children:
                    child = prefix + str(child_id)
                    if parent in valid_ids and child in valid_ids:
                        seen.add((parent, child))
    for source, target in sorted(seen):
        edges.append(
            {
                "source": source,
                "target": target,
                "type": "evolves_to",
                "role": "experiment exploration",
                "category": "evolution",
            }
        )
    manifest_integrity = _manifest_revision_state(
        manifest,
        lock,
        history_rows,
        expected_hash=expected_manifest_hash,
        parse_errors=history_errors,
        archived_manifests=archived_manifests,
    )
    manifest_hash = manifest_integrity.get("current_hash") or manifest_integrity.get(
        "base_hash"
    )
    if not _is_sha256(manifest_hash):
        manifest_hash = None
    # Research checkpoints bind ARA graphs with protocol.hashing.content_hash.
    # Keep validation on the same identity profile: JCS normalizes integral
    # floats (for example 0.0) differently and would reject the exact bytes
    # that the checkpoint recorded.
    graph_hash = content_hash(dict(graph))
    if expected_graph_hash:
        graph_binding_state = (
            "verified" if graph_hash == expected_graph_hash else "mismatch"
        )
    elif snapshot_ref and snapshot_ref != "worktree":
        graph_binding_state = "commit_bound"
    else:
        graph_binding_state = "unbound_worktree"
    graph_binding = {
        "state": graph_binding_state,
        "ok": graph_binding_state in {"verified", "commit_bound"},
        "graph_hash": graph_hash,
        "expected_graph_hash": expected_graph_hash,
    }
    if graph_binding_state == "mismatch":
        manifest_integrity["valid_hashes"] = []
    roots = [prefix + node_id for node_id in analysis.get("root_ids") or []]
    leaves = [prefix + node_id for node_id in analysis.get("leaf_ids") or []]
    return (
        nodes,
        edges,
        {
            "name": f"ara:{index}",
            "manifest_hash": manifest_hash,
            "manifest_hashes": list(manifest_integrity["valid_hashes"]),
            "manifest_integrity": manifest_integrity,
            "graph_binding": graph_binding,
            "repository_path": repository_path,
            "snapshot_ref": snapshot_ref,
            "node_ids": node_ids,
            "node_count": len(nodes),
            "experiment_node_count": len(node_ids),
            "context_pack_count": len(context_node_ids),
            "context_node_ids": context_node_ids,
            "root_ids": roots,
            "leaf_ids": leaves,
            "integrity": analysis,
            "detail_views": detail_views,
        },
    )


def _read_local_json_at(
    root_descriptor: int,
    relative_path: str | PurePosixPath,
    *,
    maximum: int = _MAX_ARA_CORE_JSON_BYTES,
) -> tuple[Mapping[str, Any] | None, bytes | None]:
    encoded = _read_local_ara_file(
        root_descriptor,
        relative_path,
        maximum=maximum,
        label="ARA JSON",
    )
    if encoded is None:
        return None, None
    try:
        value = _strict_json_bytes(encoded)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ResearchGitError(f"ARA JSON is invalid: {relative_path}") from exc
    return (value if isinstance(value, Mapping) else None), encoded


def _read_local_json(path: Path) -> Mapping[str, Any] | None:
    resolved_parent, root_descriptor = _open_local_ara_root(path.parent)
    del resolved_parent
    try:
        value, _encoded = _read_local_json_at(root_descriptor, path.name)
        return value
    finally:
        os.close(root_descriptor)


def _local_ara_node_metrics(
    root: Path,
    graph: Mapping[str, Any],
    *,
    detail_budget: dict[str, Any] | None = None,
    root_descriptor: int | None = None,
    graph_sha256: str | None = None,
    source_ref: str = "worktree",
) -> dict[str, Mapping[str, Any]]:
    details: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    budget = (
        detail_budget
        if detail_budget is not None
        else {
            "bytes": 0,
            "files": 0,
            "cache": {},
        }
    )
    opened_here = root_descriptor is None
    if root_descriptor is None:
        _resolved_root, root_descriptor = _open_local_ara_root(root)
    assert root_descriptor is not None
    try:
        for raw in graph.get("nodes") or []:
            if not isinstance(raw, Mapping):
                continue
            try:
                node_id = validate_node_id(raw.get("id"))
                if node_id in seen:
                    continue
                seen.add(node_id)
                if budget["files"] >= _MAX_ARA_NODE_DETAIL_FILES:
                    break
                remaining = _MAX_ARA_NODE_DETAILS_TOTAL_BYTES - budget["bytes"]
                if remaining <= 0:
                    break
                budget["files"] += 1
                metrics_path = PurePosixPath("nodes") / node_id / "metrics.json"
                encoded = _read_local_ara_file(
                    root_descriptor,
                    metrics_path,
                    maximum=min(_MAX_ARA_NODE_DETAIL_BYTES, remaining),
                    label="ARA node metrics",
                )
                if encoded is None:
                    continue
                budget["bytes"] += len(encoded)
                payload = _strict_json_bytes(encoded)
            except (
                ResearchGitError,
                json.JSONDecodeError,
                OSError,
                UnicodeDecodeError,
                ValueError,
                RecursionError,
            ):
                continue
            if isinstance(payload, Mapping):
                payload = dict(payload)
                if _is_sha256(graph_sha256):
                    source_path = metrics_path.as_posix()
                    payload[_ARA_SOURCE_SNAPSHOT_KEY] = _source_snapshot_receipt(
                        graph_sha256=str(graph_sha256),
                        sidecar_sha256=_sha256_bytes(encoded),
                        source_ref=source_ref,
                        source_path=source_path,
                    )
                details[node_id] = payload
    finally:
        if opened_here:
            os.close(root_descriptor)
    return details


def _committed_ara_node_metrics(
    repo: Path,
    commit: str,
    ara_dir: str,
    graph: Mapping[str, Any],
    *,
    detail_budget: dict[str, Any] | None = None,
    graph_sha256: str | None = None,
) -> dict[str, Mapping[str, Any]]:
    details: dict[str, Mapping[str, Any]] = {}
    seen: set[str] = set()
    budget = (
        detail_budget
        if detail_budget is not None
        else {
            "bytes": 0,
            "files": 0,
            "cache": {},
        }
    )
    cache = budget["cache"]
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping):
            continue
        try:
            node_id = validate_node_id(raw.get("id"))
        except ValueError:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        metrics_path = (
            PurePosixPath(ara_dir) / "nodes" / node_id / "metrics.json"
        ).as_posix()
        cache_key = ("git", repo.resolve().as_posix(), commit, metrics_path)
        if cache_key in cache:
            payload = cache[cache_key]
            if isinstance(payload, Mapping):
                details[node_id] = payload
            continue
        if budget["files"] >= _MAX_ARA_NODE_DETAIL_FILES:
            break
        remaining = _MAX_ARA_NODE_DETAILS_TOTAL_BYTES - budget["bytes"]
        if remaining <= 0:
            break
        budget["files"] += 1
        try:
            encoded = _blob_bytes_at_commit(
                repo,
                commit,
                metrics_path,
                maximum=min(_MAX_ARA_NODE_DETAIL_BYTES, remaining),
            )
            if encoded is None:
                cache[cache_key] = None
                continue
            budget["bytes"] += len(encoded)
            payload = _strict_json_bytes(encoded)
        except (
            ResearchGitError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ):
            cache[cache_key] = None
            continue
        if isinstance(payload, Mapping):
            payload = dict(payload)
            if _is_sha256(graph_sha256):
                payload[_ARA_SOURCE_SNAPSHOT_KEY] = _source_snapshot_receipt(
                    graph_sha256=str(graph_sha256),
                    sidecar_sha256=_sha256_bytes(encoded),
                    source_ref=commit,
                    source_path=metrics_path,
                )
            details[node_id] = payload
            cache[cache_key] = payload
        else:
            cache[cache_key] = None
    return details


def _read_ara(
    root: Path,
    *,
    index: int,
    disclose: bool,
    detail_budget: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    resolved_root, root_descriptor = _open_local_ara_root(root)
    try:
        graph, graph_encoded = _read_local_json_at(
            root_descriptor, "exploration_graph.json"
        )
        if graph is None or graph_encoded is None:
            raise ResearchGitError(
                f"ARA exploration graph is invalid or missing: {resolved_root}"
            )
        history_encoded = _read_local_ara_file(
            root_descriptor,
            "manifest.history.jsonl",
            maximum=_MAX_ARA_CORE_JSON_BYTES,
            label="ARA manifest history",
        )
        history_raw = None
        if history_encoded is not None:
            try:
                history_raw = history_encoded.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ResearchGitError(
                    f"ARA manifest history is invalid: {resolved_root}"
                ) from exc
        history_rows, history_errors = _read_json_lines(history_raw)
        archived_manifests: dict[str, Mapping[str, Any] | None] = {}
        archived_bytes = 0
        for row in history_rows:
            base_hash = str(row.get("base_hash") or "")
            if not _is_sha256(base_hash) or base_hash in archived_manifests:
                continue
            remaining = _MAX_ARA_ARCHIVED_MANIFEST_TOTAL_BYTES - archived_bytes
            archived_manifests[base_hash], archive_encoded = _read_local_json_at(
                root_descriptor,
                PurePosixPath("history") / f"{base_hash.split(':', 1)[1]}.json",
                maximum=min(_MAX_ARA_CORE_JSON_BYTES, max(0, remaining)),
            )
            if archive_encoded is not None:
                archived_bytes += len(archive_encoded)
        manifest, _manifest_encoded = _read_local_json_at(
            root_descriptor, "manifest.json"
        )
        lock, _lock_encoded = _read_local_json_at(root_descriptor, "manifest.lock")
        node_metrics = (
            _local_ara_node_metrics(
                resolved_root,
                graph,
                detail_budget=detail_budget,
                root_descriptor=root_descriptor,
                graph_sha256=_sha256_bytes(graph_encoded),
                source_ref="worktree",
            )
            if disclose
            else {}
        )
    finally:
        os.close(root_descriptor)
    return _project_ara_snapshot(
        graph,
        index=index,
        disclose=disclose,
        node_metrics=node_metrics,
        manifest=manifest,
        lock=lock,
        history_rows=history_rows,
        history_errors=history_errors,
        archived_manifests=archived_manifests,
        snapshot_ref="worktree",
    )


def _object_ara_manifest_hashes(item: Mapping[str, Any]) -> set[str]:
    return {manifest_hash for manifest_hash, _graph_hash in _object_ara_bindings(item)}


def _object_ara_bindings(item: Mapping[str, Any]) -> set[tuple[str, str | None]]:
    """Return the strongest ARA revision bindings declared by one object.

    A payload may repeat a manifest-only legacy pointer while provenance carries
    the newer manifest+graph pair. In that case the graph-bound declaration
    wins. Conflicting graph-bound declarations remain separate so one valid
    pointer cannot hide a broken one.
    """

    bindings: set[tuple[str, str | None]] = set()
    for container_name in ("provenance", "payload"):
        container = item.get(container_name) or {}
        if not isinstance(container, Mapping):
            continue
        manifest_hash = str(container.get("ara_manifest_hash") or "")
        if not _is_sha256(manifest_hash):
            continue
        if "ara_exploration_graph_hash" not in container:
            graph_hash: str | None = None
        else:
            declared_graph_hash = container.get("ara_exploration_graph_hash")
            graph_hash = (
                str(declared_graph_hash)
                if _is_sha256(declared_graph_hash)
                else _INVALID_ARA_GRAPH_HASH
            )
        bindings.add((manifest_hash, graph_hash))
    graph_bound_manifests = {
        manifest_hash
        for manifest_hash, graph_hash in bindings
        if graph_hash is not None
    }
    return {
        (manifest_hash, graph_hash)
        for manifest_hash, graph_hash in bindings
        if graph_hash is not None or manifest_hash not in graph_bound_manifests
    }


def _object_ara_graph_hashes(item: Mapping[str, Any]) -> set[str]:
    return {
        graph_hash
        for _manifest_hash, graph_hash in _object_ara_bindings(item)
        if graph_hash is not None
    }


def _object_context_pack_hashes(item: Mapping[str, Any]) -> set[str]:
    hashes: set[str] = set()
    provenance = item.get("provenance") or {}
    payload = item.get("payload") or {}
    for value in (
        *(
            (provenance.get("context_hashes") or [])
            if isinstance(provenance, Mapping)
            else []
        ),
        *(
            (payload.get("context_pack_refs") or [])
            if isinstance(payload, Mapping)
            else []
        ),
        *(
            (payload.get("context_hashes") or [])
            if isinstance(payload, Mapping)
            else []
        ),
    ):
        if _is_sha256(value):
            hashes.add(str(value))
    return hashes


def _checkpoint_ara_expectation(
    repo: Path,
    commit: str,
    manifest_path: str,
) -> tuple[str | None, str | None]:
    try:
        checkpoint = show_checkpoint(repo, commit)["checkpoint"]
    except ResearchGitError:
        return None, None
    for item in checkpoint.get("ara_manifests") or []:
        if not isinstance(item, Mapping):
            continue
        if _normalise_relative(str(item.get("path") or "")) != manifest_path:
            continue
        manifest_hash = str(item.get("manifest_hash") or "")
        graph_hash = str(item.get("exploration_graph_hash") or "")
        return (
            manifest_hash if _is_sha256(manifest_hash) else None,
            graph_hash if _is_sha256(graph_hash) else None,
        )
    return None, None


def _committed_ara_projection(
    repo: Path,
    commit: str,
    manifest_path: str,
    *,
    index: int,
    disclose: bool,
    expected_manifest_hash: str | None,
    expected_graph_hash: str | None,
    required: bool,
    detail_budget: dict[str, Any] | None = None,
    core_budget: dict[str, Any] | None = None,
    load_node_metrics: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    manifest_value = _json_blob_at_commit(
        repo,
        commit,
        manifest_path,
        required=required,
        maximum=_MAX_ARA_CORE_JSON_BYTES,
        read_budget=core_budget,
    )
    if not isinstance(manifest_value, Mapping):
        if required:
            raise ResearchGitError(
                f"ARA manifest is not an object at {commit}:{manifest_path}"
            )
        return None
    ara_dir = PurePosixPath(manifest_path).parent.as_posix()
    graph_path = f"{ara_dir}/exploration_graph.json"
    lock_path = f"{ara_dir}/manifest.lock"
    history_path = f"{ara_dir}/manifest.history.jsonl"
    lock_value = _json_blob_at_commit(
        repo,
        commit,
        lock_path,
        maximum=_MAX_ARA_CORE_JSON_BYTES,
        read_budget=core_budget,
    )
    history_rows, history_errors = _read_json_lines(
        _text_blob_at_commit(
            repo,
            commit,
            history_path,
            maximum=_MAX_ARA_CORE_JSON_BYTES,
            read_budget=core_budget,
        )
    )
    archived_manifests: dict[str, Mapping[str, Any] | None] = {}
    archived_bytes = 0
    for row in history_rows:
        base_hash = str(row.get("base_hash") or "")
        if not _is_sha256(base_hash) or base_hash in archived_manifests:
            continue
        archive_path = f"{ara_dir}/history/{base_hash.split(':', 1)[1]}.json"
        remaining = _MAX_ARA_ARCHIVED_MANIFEST_TOTAL_BYTES - archived_bytes
        archive_encoded = _blob_bytes_at_commit(
            repo,
            commit,
            archive_path,
            maximum=min(_MAX_ARA_CORE_JSON_BYTES, max(0, remaining)),
            read_budget=core_budget,
        )
        if archive_encoded is None:
            archive_value = None
        else:
            archived_bytes += len(archive_encoded)
            try:
                archive_value = _strict_json_bytes(archive_encoded)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
                ValueError,
            ) as exc:
                raise ResearchGitError(
                    f"ARA snapshot JSON is invalid at {commit}:{archive_path}"
                ) from exc
        archived_manifests[base_hash] = (
            archive_value if isinstance(archive_value, Mapping) else None
        )
    graph_encoded = _blob_bytes_at_commit(
        repo,
        commit,
        graph_path,
        required=required,
        maximum=_MAX_ARA_CORE_JSON_BYTES,
        read_budget=core_budget,
    )
    if graph_encoded is None:
        graph_value = None
    else:
        try:
            graph_value = _strict_json_bytes(graph_encoded)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as exc:
            raise ResearchGitError(
                f"ARA snapshot JSON is invalid at {commit}:{graph_path}"
            ) from exc
    if not isinstance(graph_value, Mapping):
        if required:
            raise ResearchGitError(
                f"ARA exploration graph is not an object at {commit}:{graph_path}"
            )
        return None
    return _project_ara_snapshot(
        graph_value,
        index=index,
        disclose=disclose,
        node_metrics=(
            _committed_ara_node_metrics(
                repo,
                commit,
                ara_dir,
                graph_value,
                detail_budget=detail_budget,
                graph_sha256=_sha256_bytes(graph_encoded),
            )
            if disclose and load_node_metrics
            else {}
        ),
        manifest=manifest_value,
        lock=lock_value if isinstance(lock_value, Mapping) else None,
        history_rows=history_rows,
        history_errors=history_errors,
        archived_manifests=archived_manifests,
        expected_manifest_hash=expected_manifest_hash,
        expected_graph_hash=expected_graph_hash,
        repository_path=manifest_path,
        snapshot_ref=commit,
    )


def _committed_ara_sources(
    repo: Path,
    commit: str,
    *,
    checkpoint: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    disclose: bool,
    explicit_paths: Iterable[str] = (),
    detail_budget: dict[str, Any] | None = None,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """Read relevant ARA projections from the exact Git tree at ``commit``."""

    tree = _run_git_bounded(
        repo,
        ["ls-tree", "-r", "-z", "--name-only", commit, "--", "ara"],
        max_output_bytes=_MAX_COMMITTED_ARA_ENUMERATION_BYTES,
    )
    tree_paths = set(
        _nul_terminated_git_paths(tree.stdout, label="committed ARA enumeration")
    )
    tracked_manifests = {
        path
        for path in tree_paths
        if path.endswith("/manifest.json") and path.startswith("ara/")
    }
    if len(tracked_manifests) > _MAX_COMMITTED_ARA_MANIFESTS:
        raise ResearchGitError("committed ARA manifests exceed the bounded audit limit")
    expected_by_path: dict[str, str] = {}
    expected_graph_by_path: dict[str, str] = {}
    for item in checkpoint.get("ara_manifests") or []:
        if not isinstance(item, Mapping):
            continue
        path = _normalise_relative(str(item.get("path") or ""))
        if not path.endswith("/manifest.json") and path != "manifest.json":
            raise ResearchGitError(f"checkpoint ARA path is not a manifest: {path}")
        expected_by_path[path] = str(item.get("manifest_hash") or "")
        graph_hash = str(item.get("exploration_graph_hash") or "")
        if graph_hash:
            expected_graph_by_path[path] = graph_hash
        if len(expected_by_path) > _MAX_COMMITTED_ARA_MANIFESTS:
            raise ResearchGitError(
                "committed ARA manifests exceed the bounded audit limit"
            )
    forced_paths = {_normalise_relative(path) for path in explicit_paths}
    if len(forced_paths) > _MAX_COMMITTED_ARA_MANIFESTS:
        raise ResearchGitError("committed ARA manifests exceed the bounded audit limit")
    priority_paths = set(expected_by_path) | forced_paths
    candidates = sorted(
        tracked_manifests | priority_paths,
        key=lambda path: (path not in priority_paths, path),
    )
    if len(candidates) > _MAX_COMMITTED_ARA_MANIFESTS:
        raise ResearchGitError("committed ARA manifests exceed the bounded audit limit")
    referenced_hashes = (
        set().union(*(_object_ara_manifest_hashes(item) for item in objects))
        if objects
        else set()
    )

    projected: list[
        tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
    ] = []
    core_budget: dict[str, Any] = {
        "bytes": 0,
        "files": 0,
        "maximum_bytes": _MAX_COMMITTED_ARA_CORE_TOTAL_BYTES,
        "maximum_files": _MAX_COMMITTED_ARA_CORE_FILES,
        "label": "committed ARA core metadata",
    }
    for manifest_path in candidates:
        preselected = manifest_path in priority_paths
        projection = _committed_ara_projection(
            repo,
            commit,
            manifest_path,
            index=len(projected),
            disclose=disclose,
            expected_manifest_hash=expected_by_path.get(manifest_path),
            expected_graph_hash=expected_graph_by_path.get(manifest_path),
            required=(
                manifest_path in expected_by_path or manifest_path in forced_paths
            ),
            detail_budget=detail_budget,
            core_budget=core_budget,
            load_node_metrics=preselected,
        )
        if projection is None:
            continue
        manifest_integrity = projection[2]["manifest_integrity"]
        relevant = (
            manifest_path in expected_by_path
            or manifest_path in forced_paths
            or bool(referenced_hashes & set(manifest_integrity["valid_hashes"]))
        )
        if not relevant:
            continue
        if disclose and not preselected:
            detailed_projection = _committed_ara_projection(
                repo,
                commit,
                manifest_path,
                index=len(projected),
                disclose=True,
                expected_manifest_hash=expected_by_path.get(manifest_path),
                expected_graph_hash=expected_graph_by_path.get(manifest_path),
                required=False,
                detail_budget=detail_budget,
                core_budget=core_budget,
            )
            if detailed_projection is not None:
                projection = detailed_projection
        projected.append(projection)

    # A current ref can contain objects bound to several immutable versions of
    # the same ARA graph. Rehydrate those versions from reachable Git history so
    # old evidence is neither attached to the newest graph nor reported missing.
    object_bindings = {
        binding for item in objects for binding in _object_ara_bindings(item)
    }

    def resolved_by_projection(
        binding: tuple[str, str | None],
        projection: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    ) -> bool:
        manifest_hash, graph_hash = binding
        metadata = projection[2]
        return bool(
            manifest_hash == metadata.get("manifest_hash")
            and (
                graph_hash is None
                or graph_hash == metadata["graph_binding"].get("graph_hash")
            )
            and metadata["manifest_integrity"].get("ok")
            and metadata["graph_binding"].get("ok")
        )

    pending = {
        binding
        for binding in object_bindings
        if not any(resolved_by_projection(binding, item) for item in projected)
    }
    seen_versions = {
        (
            str(item[2].get("repository_path") or ""),
            str(item[2].get("manifest_hash") or ""),
            str(item[2]["graph_binding"]["graph_hash"]),
        )
        for item in projected
    }
    history_commits_seen = 0
    for manifest_path in candidates:
        if not pending:
            break
        remaining_history = _MAX_COMMITTED_ARA_HISTORY_COMMITS - history_commits_seen
        if remaining_history <= 0:
            raise ResearchGitError(
                "committed ARA history exceeds the bounded audit limit"
            )
        graph_path = (
            PurePosixPath(manifest_path).parent / "exploration_graph.json"
        ).as_posix()
        history = _run_git_bounded(
            repo,
            [
                "log",
                "--format=%H",
                f"--max-count={remaining_history + 1}",
                commit,
                "--",
                manifest_path,
                graph_path,
            ],
            max_output_bytes=_MAX_COMMITTED_ARA_HISTORY_OUTPUT_BYTES,
            check=False,
        )
        if history.returncode:
            continue
        historical_commits = history.stdout.splitlines()
        if any(
            not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", historical_commit)
            for historical_commit in historical_commits
        ):
            raise ResearchGitError(
                "committed ARA history returned an invalid commit identity"
            )
        if len(historical_commits) > remaining_history:
            raise ResearchGitError(
                "committed ARA history exceeds the bounded audit limit"
            )
        history_commits_seen += len(historical_commits)
        for historical_commit in historical_commits:
            if historical_commit == commit or not pending:
                continue
            expected_manifest_hash, expected_graph_hash = _checkpoint_ara_expectation(
                repo,
                historical_commit,
                manifest_path,
            )
            if expected_manifest_hash is None:
                continue
            projection = _committed_ara_projection(
                repo,
                historical_commit,
                manifest_path,
                index=len(projected),
                disclose=disclose,
                expected_manifest_hash=expected_manifest_hash,
                expected_graph_hash=expected_graph_hash,
                required=False,
                detail_budget=detail_budget,
                core_budget=core_budget,
                load_node_metrics=False,
            )
            if projection is None:
                continue
            version_key = (
                manifest_path,
                str(projection[2].get("manifest_hash") or ""),
                str(projection[2]["graph_binding"]["graph_hash"]),
            )
            if version_key in seen_versions:
                continue
            matched = {
                binding
                for binding in pending
                if resolved_by_projection(binding, projection)
            }
            if not matched:
                continue
            if disclose:
                detailed_projection = _committed_ara_projection(
                    repo,
                    historical_commit,
                    manifest_path,
                    index=len(projected),
                    disclose=True,
                    expected_manifest_hash=expected_manifest_hash,
                    expected_graph_hash=expected_graph_hash,
                    required=False,
                    detail_budget=detail_budget,
                    core_budget=core_budget,
                )
                if detailed_projection is not None:
                    projection = detailed_projection
            projected.append(projection)
            seen_versions.add(version_key)
            pending.difference_update(matched)
    return projected


def build_research_dag(
    repo: str | Path,
    *,
    ref: str = "HEAD",
    ara_roots: Sequence[str | Path] = (),
    disclose_summaries: bool = True,
    _object_rows_out: list[Mapping[str, Any]] | None = None,
    _ara_detail_views_out: dict[str, Any] | None = None,
    _closure_out: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project Research VCS and optional ARA execution trees into one DAG."""

    repository_root = _repository_root(repo)
    checkpoint = show_checkpoint(repo, ref)
    commit = str(checkpoint["commit"])
    rows = list_research_objects_at_ref(repo, commit)
    if _object_rows_out is not None:
        _object_rows_out.extend(rows)
    objects = {str(item["object_id"]): item for item in rows}
    introduction_order = research_object_introduction_order(repo, ref=commit)
    claims = [item for item in rows if item.get("kind") == "claim"]
    closure = (
        audit_research_closure(repo, ref=commit, level="verify") if claims else None
    )
    if _closure_out is not None and closure is not None:
        _closure_out.append(closure)
    claim_closure = {
        str(item["claim_id"]): item for item in (closure or {}).get("claims") or []
    }
    blocker_map: dict[str, list[str]] = defaultdict(list)
    warning_map: dict[str, list[str]] = defaultdict(list)
    for item in (closure or {}).get("blockers") or []:
        blocker_map[str(item.get("object_id") or "")].append(str(item["code"]))
    for item in (closure or {}).get("warnings") or []:
        warning_map[str(item.get("object_id") or "")].append(str(item["code"]))
    from .research_context import _effective_superseded_ids

    # Retired challenges remain in the graph, but do not override current proof.
    superseded_sources = _effective_superseded_ids(objects)
    challenge_targets = {
        str(relation.get("target"))
        for item in rows
        if str(item["object_id"]) not in superseded_sources
        for relation in item.get("relations") or []
        if relation.get("type")
        in {
            "refutes",
            "qualified_refutes",
            "contradicts",
            "challenges_inference",
            "invalidates",
        }
    }
    nodes = [
        {
            "id": str(item["object_id"]),
            "source": "research_vcs",
            "kind": str(item["kind"]),
            "state": str(item["state"]),
            "phase": _PHASES.get(str(item["kind"]), 4),
            "layer": _epistemic_layer(str(item["kind"])),
            "summary": _summary(item, disclose=disclose_summaries),
            "content_hash": str(item["content_hash"]),
            "actor": dict(item.get("actor") or {}),
            "created_at": str(item.get("created_at") or ""),
            "proof": _object_proof(
                item,
                objects,
                claim_closure=claim_closure,
                challenge_targets=challenge_targets,
                closure_blockers=blocker_map,
                closure_warnings=warning_map,
            ),
            "source_ref": commit,
        }
        for item in rows
    ]
    context_sources = {
        node["id"]: set(objects[node["id"]]["payload"]["source_object_ids"])
        for node in nodes
        if node["kind"] == "context_snapshot"
        and any(
            check["code"] == "context_identity_valid" and check["passed"] is True
            for check in node["proof"]["checks"]
        )
    }
    edges: list[dict[str, Any]] = []
    dangling: list[str] = []
    unknown_relations: list[str] = []
    for item in rows:
        for relation in item.get("relations") or []:
            target = str(relation.get("target") or "")
            if target not in objects:
                dangling.append(f"{item['object_id']}:{target}")
                continue
            relation_type = str(relation.get("type") or "depends_on")
            relation_role = str(relation.get("role") or "")
            relation_category = _RELATION_CATEGORIES.get(
                relation_type, _UNKNOWN_RELATION_CATEGORY
            )
            if relation_type not in _RELATION_CATEGORIES:
                unknown_relations.append(f"{item['object_id']}:{relation_type}")
            edges.append(
                {
                    "source": target,
                    "target": str(item["object_id"]),
                    "type": relation_type,
                    "role": relation_role if disclose_summaries else "",
                    # Only a validated source closure specializes depends_on;
                    # free-form roles cannot override typed relation semantics.
                    "category": (
                        "context"
                        if relation_type == "depends_on"
                        and target in context_sources.get(str(item["object_id"]), ())
                        else relation_category
                    ),
                }
            )

    sources: list[dict[str, Any]] = [
        {"name": "research_vcs", "commit": commit, "object_count": len(rows)}
    ]
    ara_metadata: list[dict[str, Any]] = []
    detail_budget: dict[str, Any] = {"bytes": 0, "files": 0, "cache": {}}
    committed_explicit_paths: set[str] = set()
    local_ara_roots: list[Path] = []
    for raw_root in ara_roots:
        requested = Path(raw_root).expanduser().resolve()
        requested_root = (
            requested.parent if requested.name == "manifest.json" else requested
        )
        try:
            relative_root = requested_root.relative_to(repository_root).as_posix()
        except ValueError:
            local_ara_roots.append(requested_root)
            continue
        committed_explicit_paths.add(
            (PurePosixPath(relative_root) / "manifest.json").as_posix()
        )

    committed_sources = _committed_ara_sources(
        repository_root,
        commit,
        checkpoint=checkpoint["checkpoint"],
        objects=rows,
        disclose=disclose_summaries,
        explicit_paths=committed_explicit_paths,
        detail_budget=detail_budget,
    )
    for ara_nodes, ara_edges, metadata_row in committed_sources:
        nodes.extend(ara_nodes)
        edges.extend(ara_edges)
        ara_metadata.append(metadata_row)

    for raw_root in local_ara_roots:
        index = len(ara_metadata)
        ara_nodes, ara_edges, metadata_row = _read_ara(
            raw_root,
            index=index,
            disclose=disclose_summaries,
            detail_budget=detail_budget,
        )
        nodes.extend(ara_nodes)
        edges.extend(ara_edges)
        ara_metadata.append(metadata_row)

    if _ara_detail_views_out is not None:
        for metadata_row in ara_metadata:
            _ara_detail_views_out.update(metadata_row.get("detail_views") or {})

    for metadata_row in ara_metadata:
        sources.append(
            {
                "name": metadata_row["name"],
                "manifest_hash": metadata_row["manifest_hash"],
                "manifest_hashes": metadata_row["manifest_hashes"],
                "manifest_integrity": {
                    key: value
                    for key, value in metadata_row["manifest_integrity"].items()
                    if key != "valid_hashes"
                },
                "graph_binding": metadata_row["graph_binding"],
                "repository_path": metadata_row.get("repository_path"),
                "snapshot_ref": metadata_row.get("snapshot_ref"),
                "object_count": metadata_row["node_count"],
                "experiment_node_count": metadata_row.get("experiment_node_count", 0),
                "context_pack_count": metadata_row.get("context_pack_count", 0),
                "integrity": {
                    "is_dag": metadata_row["integrity"]["is_dag"],
                    "error_count": metadata_row["integrity"]["error_count"],
                    "warning_count": metadata_row["integrity"]["warning_count"],
                },
            }
        )
    source_by_name = {str(item["name"]): item for item in sources}
    resolved_bindings: set[tuple[str, str, str | None]] = set()
    for ara in ara_metadata:
        manifest_hash = ara.get("manifest_hash")
        bound_object_ids: list[str] = []
        if not _is_sha256(manifest_hash):
            source_by_name[ara["name"]]["binding_count"] = 0
            source_by_name[ara["name"]]["bound_object_ids"] = []
            continue
        for object_id, item in objects.items():
            matching_bindings = {
                (manifest_hash, graph_hash)
                for manifest_hash, graph_hash in _object_ara_bindings(item)
                if manifest_hash == ara.get("manifest_hash")
                and (
                    graph_hash is None
                    or graph_hash == ara["graph_binding"]["graph_hash"]
                )
            }
            if not matching_bindings:
                continue
            bound_object_ids.append(object_id)
            payload = item.get("payload") or {}
            node_id = (
                str(payload.get("node_id") or "")
                if isinstance(payload, Mapping)
                else ""
            )
            anchors = (
                [ara["node_ids"][node_id]]
                if node_id in ara["node_ids"]
                else ara["leaf_ids"]
            )
            if anchors:
                resolved_bindings.update(
                    (object_id, manifest_hash, graph_hash)
                    for manifest_hash, graph_hash in matching_bindings
                )
            for source_node in anchors:
                verified_binding = bool(
                    ara["manifest_integrity"]["ok"]
                    and ara["graph_binding"]["ok"]
                    and ara["integrity"]["is_dag"]
                )
                edges.append(
                    {
                        "source": source_node,
                        "target": object_id,
                        "type": "anchors",
                        "role": (
                            "ARA manifest revision evidence"
                            if verified_binding
                            else "unverified external ARA lineage"
                        ),
                        "category": "verification" if verified_binding else "lineage",
                    }
                )
        source_by_name[ara["name"]]["binding_count"] = len(bound_object_ids)
        source_by_name[ara["name"]]["bound_object_ids"] = sorted(bound_object_ids)

    for ara in ara_metadata:
        context_node_ids = ara.get("context_node_ids") or {}
        context_bound_objects: set[str] = set()
        for object_id, item in objects.items():
            for context_hash in sorted(
                _object_context_pack_hashes(item) & set(context_node_ids)
            ):
                edges.append(
                    {
                        "source": context_node_ids[context_hash],
                        "target": object_id,
                        "type": "contextualizes",
                        "role": "exact ARA context pack consumed by Research Object",
                        "category": "context",
                    }
                )
                context_bound_objects.add(object_id)
        source_by_name[ara["name"]]["context_binding_count"] = len(
            context_bound_objects
        )
        source_by_name[ara["name"]]["context_bound_object_ids"] = sorted(
            context_bound_objects
        )

    unresolved_ara_bindings = sorted(
        (object_id, manifest_hash, graph_hash)
        for object_id, item in objects.items()
        for manifest_hash, graph_hash in _object_ara_bindings(item)
        if (object_id, manifest_hash, graph_hash) not in resolved_bindings
    )

    for node in nodes:
        node.setdefault("layer", _epistemic_layer(str(node.get("kind") or "")))

    graph_for_analysis = {
        "nodes": [{"id": node["id"]} for node in nodes],
        "edges": [
            {"parent": edge["source"], "child": edge["target"]} for edge in edges
        ],
    }
    analysis = analyze_exploration_graph(
        graph_for_analysis,
        portable_node_ids=False,
    )
    for ara in ara_metadata:
        source_integrity = ara["integrity"]
        for issue in source_integrity.get("issues") or []:
            copied = dict(issue)
            copied["code"] = f"{ara['name']}_{copied.get('code', 'invalid')}"
            copied["path"] = f"{ara['name']}:{copied.get('path', '')}"
            analysis["issues"].append(copied)
            if copied.get("severity") == "error":
                analysis["error_count"] += 1
            elif copied.get("severity") == "warning":
                analysis["warning_count"] += 1
        if source_integrity.get("error_count"):
            analysis["is_dag"] = False
        manifest_integrity = ara["manifest_integrity"]
        severity = "error" if manifest_integrity["state"] == "tampered" else "warning"
        for message in manifest_integrity.get("issues") or []:
            analysis["issues"].append(
                {
                    "severity": severity,
                    "code": f"{ara['name']}_manifest_{manifest_integrity['state']}",
                    "message": message,
                    "path": f"{ara['name']}:manifest.json",
                }
            )
            if severity == "error":
                analysis["error_count"] += 1
                analysis["is_dag"] = False
            else:
                analysis["warning_count"] += 1
        graph_binding = ara["graph_binding"]
        if graph_binding["state"] in {"mismatch", "unbound_worktree"}:
            graph_severity = (
                "error" if graph_binding["state"] == "mismatch" else "warning"
            )
            analysis["issues"].append(
                {
                    "severity": graph_severity,
                    "code": f"{ara['name']}_graph_{graph_binding['state']}",
                    "message": (
                        "exploration graph does not match its checkpoint binding"
                        if graph_binding["state"] == "mismatch"
                        else "external worktree exploration graph has no commit binding"
                    ),
                    "path": f"{ara['name']}:exploration_graph.json",
                }
            )
            if graph_severity == "error":
                analysis["error_count"] += 1
                analysis["is_dag"] = False
            else:
                analysis["warning_count"] += 1
    if dangling:
        analysis["is_dag"] = False
        analysis["error_count"] += len(dangling)
        analysis["issues"].extend(
            {
                "severity": "error",
                "code": "dangling_research_relation",
                "message": value,
                "path": "edges",
            }
            for value in sorted(dangling)
        )
    if unknown_relations:
        # URI/custom relation extensions remain visible, but are not trusted to
        # carry a built-in scientific meaning until they are added to the
        # allowlist.  This is a warning rather than a DAG error so legacy and
        # interoperable records remain inspectable.
        analysis["warning_count"] += len(unknown_relations)
        analysis["issues"].extend(
            {
                "severity": "warning",
                "code": "unknown_research_relation_type",
                "message": value,
                "path": "edges",
            }
            for value in sorted(unknown_relations)
        )
    if unresolved_ara_bindings:
        analysis["is_dag"] = False
        analysis["error_count"] += len(unresolved_ara_bindings)
        analysis["issues"].extend(
            {
                "severity": "error",
                "code": "unresolved_ara_manifest_binding",
                "message": (
                    f"{object_id} references unavailable ARA {manifest_hash}"
                    + (f" / graph {graph_hash}" if graph_hash is not None else "")
                ),
                "path": "sources",
            }
            for object_id, manifest_hash, graph_hash in unresolved_ara_bindings
        )
    if not disclose_summaries:
        # Roles are free-form prose on both Research VCS and synthesized ARA
        # edges. Keep the typed relation/category topology, but never place
        # role text in a metadata-only JSON export.
        edges = [{**edge, "role": ""} for edge in edges]
    proof_counts = Counter(node["proof"]["level"] for node in nodes)
    theory_frontier, claim_insights, strategy_summary = _strategy_projection(
        objects, introduction_order, disclose=disclose_summaries
    )
    base = {
        "schema_version": RESEARCH_DAG_SCHEMA,
        "ref": ref,
        "commit": commit,
        "content_disclosed": bool(disclose_summaries),
        "sources": sources,
        "nodes": sorted(nodes, key=lambda node: (node["phase"], node["id"])),
        "edges": sorted(
            edges,
            key=lambda edge: (
                edge["source"],
                edge["target"],
                edge["type"],
                edge["role"],
            ),
        ),
        "proof_summary": dict(sorted(proof_counts.items())),
        "theory_frontier": theory_frontier,
        "claim_insights": claim_insights,
        "strategy_summary": strategy_summary,
        "scientific_closure": {
            "status": (closure or {}).get("status", "not_applicable"),
            "claim_count": len(claims),
            "blocker_count": len((closure or {}).get("blockers") or []),
            "warning_count": len((closure or {}).get("warnings") or []),
            "closure_levels": dict((closure or {}).get("closure_levels") or {}),
            "content_hash": (closure or {}).get("content_hash"),
        },
        "integrity": analysis,
    }
    graph = {
        **base,
        "generated_at": _now_iso(),
        "graph_hash": canonical_content_hash(base),
    }
    try:
        validate_json(graph, load_schema("research_dag"))
    except ValidationError as exc:  # pragma: no cover - implementation contract
        raise ResearchGitError(
            f"generated research DAG is invalid: {exc.message}"
        ) from exc
    return graph


_OPAQUE_SEARCH_TOKEN_RE = re.compile(
    r"(?i)(?<![0-9a-z])(?:rso-[0-9a-f]{16}|sha256:[0-9a-f]{64}|"
    r"[0-9a-f]{12,})(?![0-9a-z])"
)
_SEARCHABLE_GATE_FIELDS = (
    "gate_id",
    "candidate_id",
    "phase_id",
    "plan_id",
    "proposal_id",
    "question_id",
    "review_id",
    "sprint_id",
    "decision",
    "status",
    "reason",
    "candidate_count",
    "candidate_count_after_gate",
    "candidate_count_after_search",
    "evaluated_candidate_ids",
    "experiment_authorized",
    "claim_promotion_allowed",
    "claim_promotion_authorized",
    "best_paper_level_theorem_survives",
    "protocol_kind",
    "reentry_requirements",
    "verdict_vector",
    "preregistered_falsifier",
    "authority",
    "ara_created",
    "verification",
    "gate_rules",
    "rules",
    "criteria",
    "pass_condition",
    "failure_rules",
)
_SEMANTIC_SEARCH_ID_KEYS = frozenset(
    {
        "candidate_id",
        "criterion_id",
        "evaluated_candidate_ids",
        "gate_id",
        "phase_id",
        "plan_id",
        "proposal_id",
        "question_id",
        "review_id",
        "sprint_id",
    }
)
_SEARCHABLE_ALIAS_FIELDS = (
    "branch",
    "candidate_id",
    "gate_id",
    "phase_id",
    "plan_id",
    "proposal_id",
    "question_id",
    "review_id",
    "sprint_id",
)
_SEARCH_OMITTED_KEYS = frozenset(
    {
        "artifact_path",
        "artifacts_dir",
        "content_hash",
        "href",
        "input_bindings",
        "logical_path",
        "pointer_path",
        "source_ref",
    }
)

_RESEARCH_DAG_SEARCH_HELPERS_JS = r"""
function foldSearch(value) {
  return String(value ?? '').normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim();
}
function isDirectMatch(node, query, index) {
  const rawQuery = String(query ?? '').trim().toLowerCase();
  const foldedQuery = foldSearch(query);
  if (!foldedQuery) return true;
  return rawQuery === String(node.id).toLowerCase()
    || foldSearch(index[node.id] || (node.ui_aggregate ? node.search_text : '')).includes(foldedQuery);
}
function oneHopIds(seedIds, edges) {
  const visible = new Set(seedIds);
  for (const edge of edges) {
    if (seedIds.has(edge.source) || seedIds.has(edge.target)) {
      visible.add(edge.source);
      visible.add(edge.target);
    }
  }
  return visible;
}
function sourceNameForNode(node) {
  if (node && node.ui_aggregate && node.source_name) return String(node.source_name);
  const match = node.source === 'ara' ? /^(ara:\d+):/.exec(String(node.id)) : null;
  return match ? match[1] : String(node.source || 'unknown');
}
function sourceDisplayName(source) {
  const fallback = String((source || {}).name || 'unknown source');
  const rawPath = String((source || {}).repository_path || '').replaceAll('\\', '/');
  const parts = rawPath.split('/').filter(Boolean);
  if (parts.length && parts[parts.length - 1] === 'manifest.json') parts.pop();
  return parts.length ? parts[parts.length - 1] : fallback;
}
function fitSvgLabel(element, maxWidth) {
  if (!element || typeof element.getComputedTextLength !== 'function') return '';
  const original = String(element.textContent || '');
  const measured = Number(element.getComputedTextLength());
  if (!Number.isFinite(measured) || measured <= maxWidth) return original;
  const suffix = '\u2026';
  const characters = Array.from(original.endsWith(suffix) ? original.slice(0, -1) : original);
  let low = 0;
  let high = characters.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    element.textContent = `${characters.slice(0, middle).join('')}${suffix}`;
    if (Number(element.getComputedTextLength()) <= maxWidth) low = middle;
    else high = middle - 1;
  }
  element.textContent = `${characters.slice(0, low).join('')}${suffix}`;
  return element.textContent;
}
function matchingAraSourceNamesForQuery(query, nodes, searchIndex) {
  if (!foldSearch(query)) return [];
  return [...new Set((nodes || []).filter(candidate =>
    candidate.source === 'ara' && isDirectMatch(candidate, query, searchIndex || {})
  ).map(sourceNameForNode))].sort();
}
function araSourceAggregate(source) {
  const sourceName = String(source.name);
  const nodeCount = Number(source.object_count || 0);
  const exactLinkCount = Number(source.binding_count || 0);
  const displayName = sourceDisplayName(source);
  const noExactLinks = exactLinkCount === 0;
  return {
    id: `ui:ara-source:${sourceName}`,
    source: 'ui_aggregate',
    source_name: sourceName,
    kind: 'ara_source_snapshot',
    state: noExactLinks ? 'no_exact_research_object_link' : 'source_snapshot',
    phase: 3,
    layer: 'execution',
    summary: `${displayName} · ${nodeCount} ARA nodes`,
    content_hash: null,
    actor: null,
    created_at: null,
    source_ref: String(source.snapshot_ref || ''),
    proof: {
      level: 'recorded', checks: [], blockers: [], warnings: [], contested: false
    },
    ui_aggregate: true,
    search_text: `${sourceName} ${displayName} ARA source snapshot ${nodeCount} nodes`,
    aggregate: {
      source_name: sourceName,
      display_name: displayName,
      node_count: nodeCount,
      exact_link_count: exactLinkCount,
      context_link_count: Number(source.context_binding_count || 0),
      no_exact_links: noExactLinks,
      selected_node_count: Number(source.selected_node_count ?? nodeCount),
      omitted_node_count: Number(source.omitted_node_count || 0),
      projection: source.projection || null,
      repository_path: source.repository_path || null,
      snapshot_ref: source.snapshot_ref || null,
      manifest_state: ((source.manifest_integrity || {}).state) || 'unreported',
      graph_binding_state: ((source.graph_binding || {}).state) || 'unreported',
      source_error_count: Number(((source.integrity || {}).error_count) || 0),
      source_warning_count: Number(((source.integrity || {}).warning_count) || 0),
      bound_object_ids: Array.isArray(source.bound_object_ids)
        ? source.bound_object_ids.map(String) : [],
      context_bound_object_ids: Array.isArray(source.context_bound_object_ids)
        ? source.context_bound_object_ids.map(String) : []
    }
  };
}
const ARA_DECISION_SPINE_MAX_NODES = 64;
const ARA_DECISION_SPINE_MAX_PER_SOURCE = 12;
const ARA_DECISION_SPINE_ANCHOR_RE = /(?:decision|select(?:ed|ion)?|kill|keep|stop|hold|pass|fail|reject|promot|admit|result|verdict|comparison|baseline|candidate|evidence|metric|outcome|conclusion)/;
function araDecisionSpineScore(node, query, searchIndex) {
  const kind = String(node && node.kind || '').toLowerCase();
  const summary = String(node && node.summary || '').toLowerCase();
  const indexed = String((searchIndex || {})[node && node.id] || '').toLowerCase();
  const text = `${kind} ${summary} ${indexed}`;
  let score = 0;
  if (foldSearch(query) && isDirectMatch(node, query, searchIndex || {})) score += 100000;
  if (kind === 'gate_decision' || kind === 'stopping_decision') score += 5000;
  if (kind === 'experiment_attempt' || kind === 'reproduction') score += 3500;
  if (kind === 'evidence_synthesis' || kind === 'claim') score += 3000;
  if (kind === 'metric' || kind === 'observation' || kind === 'evidence') score += 2200;
  if (ARA_DECISION_SPINE_ANCHOR_RE.test(text)) score += 900;
  if (/\b(?:rejected|failed|blocked|superseded)\b/.test(String(node && node.state || '').toLowerCase())) score += 200;
  if (node && node.proof && node.proof.level === 'verified') score += 80;
  if (node && node.proof && node.proof.level === 'replayable') score += 40;
  // Lower phases are earlier decisions; phase is only a deterministic tie-break
  // signal and never changes the authoritative graph or proof level.
  score += Math.max(0, 20 - Number(node && node.phase || 0));
  return score;
}
function projectAraDecisionSpine(nodes, sources, query, searchIndex, selectedSource) {
  const allNodes = Array.isArray(nodes) ? nodes : [];
  const araSources = (sources || [])
    .filter(source => /^ara:\d+$/.test(String(source.name)))
    .filter(source => !selectedSource || selectedSource === 'all' || String(source.name) === String(selectedSource))
    .sort((a, b) => String(a.name).localeCompare(String(b.name)));
  let remaining = ARA_DECISION_SPINE_MAX_NODES;
  const projected = [];
  const aggregateNodes = [];
  let omittedNodeCount = 0;
  for (const source of araSources) {
    const sourceName = String(source.name);
    const sourceNodes = allNodes.filter(node => sourceNameForNode(node) === sourceName);
    const ranked = [...sourceNodes].sort((a, b) =>
      araDecisionSpineScore(b, query, searchIndex) - araDecisionSpineScore(a, query, searchIndex)
      || Number(a.phase || 0) - Number(b.phase || 0)
      || String(a.id).localeCompare(String(b.id))
    );
    const take = Math.min(ARA_DECISION_SPINE_MAX_PER_SOURCE, remaining, ranked.length);
    const shown = ranked.slice(0, take);
    projected.push(...shown);
    remaining -= shown.length;
    const omitted = Math.max(0, sourceNodes.length - shown.length);
    omittedNodeCount += omitted;
    if (omitted > 0) {
      aggregateNodes.push(araSourceAggregate({
        ...source,
        selected_node_count: shown.length,
        omitted_node_count: omitted,
        projection: 'ara_decision_spine',
      }));
    }
  }
  return {
    nodes: [...projected, ...aggregateNodes],
    aggregateCount: aggregateNodes.length,
    collapsedNodeCount: omittedNodeCount,
    omittedNodeCount,
    expandedSourceNames: [],
    expandedSourceName: null,
    presentation: 'ara_decision_spine',
    maxNodes: ARA_DECISION_SPINE_MAX_NODES,
    maxPerSource: ARA_DECISION_SPINE_MAX_PER_SOURCE,
  };
}
function projectDagView(nodes, sources, view, sourceSelection, query, searchIndex) {
  const allNodes = Array.isArray(nodes) ? nodes : [];
  const araSources = (sources || []).filter(source => /^ara:\d+$/.test(String(source.name)));
  const araSourceNames = new Set(araSources.map(source => String(source.name)));
  const selectedSource = String(sourceSelection || 'all');
  if (view === 'ara_spine') {
    if (selectedSource === 'research_vcs') {
      return {
        nodes: allNodes.filter(node => node.source === 'research_vcs'),
        aggregateCount: 0,
        collapsedNodeCount: 0,
        omittedNodeCount: 0,
        expandedSourceNames: [],
        expandedSourceName: null,
        presentation: 'research_vcs',
      };
    }
    return projectAraDecisionSpine(
      allNodes, sources, query, searchIndex, selectedSource === 'all' ? '' : selectedSource
    );
  }
  if (selectedSource === 'research_vcs') {
    return {
      nodes: allNodes.filter(node => node.source === 'research_vcs'),
      aggregateCount: 0,
      collapsedNodeCount: 0,
      expandedSourceNames: [],
      expandedSourceName: null
    };
  }
  if (araSourceNames.has(selectedSource)) {
    return {
      nodes: allNodes.filter(node => sourceNameForNode(node) === selectedSource),
      aggregateCount: 0,
      collapsedNodeCount: 0,
      expandedSourceNames: [selectedSource],
      expandedSourceName: selectedSource
    };
  }
  if (view === 'full') {
    return {
      nodes: [...allNodes],
      aggregateCount: 0,
      collapsedNodeCount: 0,
      expandedSourceNames: [],
      expandedSourceName: null
    };
  }
  // A search must reveal matching nodes from every collapsed source, otherwise
  // Overview silently drops direct matches that remain visible in Full audit.
  const expandedSourceNames = matchingAraSourceNamesForQuery(query, allNodes, searchIndex);
  const expandedSources = new Set(expandedSourceNames);
  const rawNodes = allNodes.filter(node =>
    node.source === 'research_vcs'
      || expandedSources.has(sourceNameForNode(node))
  );
  const collapsedSources = araSources.filter(
    source => !expandedSources.has(String(source.name))
  );
  return {
    nodes: [...rawNodes, ...collapsedSources.map(araSourceAggregate)],
    aggregateCount: collapsedSources.length,
    collapsedNodeCount: collapsedSources.reduce(
      (count, source) => count + Number(source.object_count || 0), 0
    ),
    expandedSourceNames,
    expandedSourceName: expandedSourceNames.length === 1 ? expandedSourceNames[0] : null
  };
}
function authoritativeEdgesForNodes(nodes, edges) {
  const visibleIds = new Set((nodes || []).map(node => String(node.id)));
  return (edges || []).filter(
    edge => visibleIds.has(String(edge.source)) && visibleIds.has(String(edge.target))
  );
}
function sourceBindingSummary(sources) {
  const araSources = (sources || []).filter(source => /^ara:\d+$/.test(String(source.name)));
  const withoutExactLinks = araSources.filter(
    source => Number(source.binding_count || 0) === 0
  );
  return {
    araSourceCount: araSources.length,
    exactLinkCount: araSources.reduce(
      (count, source) => count + Number(source.binding_count || 0), 0
    ),
    sourcesWithExactLinks: araSources.length - withoutExactLinks.length,
    sourcesWithoutExactLinks: withoutExactLinks.length,
    nodesWithoutExactLinks: withoutExactLinks.reduce(
      (count, source) => count + Number(source.object_count || 0), 0
    )
  };
}
""".strip()


def _search_key_is_opaque(key: Any) -> bool:
    normalized = str(key).strip().lower()
    if normalized in _SEMANTIC_SEARCH_ID_KEYS:
        return False
    return (
        normalized in _SEARCH_OMITTED_KEYS
        or normalized.endswith(("_hash", "_hashes", "_path", "_paths"))
        or normalized.endswith(("_id", "_ids"))
    )


def _safe_search_text(value: Any) -> str:
    text = _OPAQUE_SEARCH_TOKEN_RE.sub(" ", str(value))
    return " ".join(text.split())


def _search_terms(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        terms: list[str] = []
        for key, nested in value.items():
            if _search_key_is_opaque(key):
                continue
            label = _safe_search_text(key)
            if label:
                terms.extend((label, label.replace("_", " ")))
            terms.extend(_search_terms(nested))
        return terms
    if isinstance(value, (list, tuple)):
        return [term for nested in value for term in _search_terms(nested)]
    if value is None:
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    text = _safe_search_text(value)
    return [text] if text else []


def _build_dag_search_index(
    nodes: Sequence[Mapping[str, Any]], node_details: Mapping[str, Any]
) -> dict[str, str]:
    """Index human-facing detail without indexing opaque provenance identifiers."""

    search_index: dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        terms: list[str] = []
        for key in ("kind", "state", "layer", "summary"):
            terms.extend(_search_terms(node.get(key)))
        actor = node.get("actor") or {}
        if isinstance(actor, Mapping):
            # Actor identity is visible in the detail pane, so it is intentionally
            # searchable even though other object identifiers are not.
            terms.extend(_search_terms(actor.get("actor_id")))
            terms.extend(_search_terms(actor.get("authority")))

        view = node_details.get(node_id) or {}
        if isinstance(view, Mapping):
            terms.extend(_search_terms(view.get("title")))
            terms.extend(_search_terms(view.get("fields") or {}))
            gate = view.get("gate") or {}
            if isinstance(gate, Mapping):
                terms.extend(
                    _search_terms(
                        {
                            key: gate[key]
                            for key in _SEARCHABLE_GATE_FIELDS
                            if key in gate
                        }
                    )
                )
            for lineage in view.get("lineage") or []:
                if isinstance(lineage, Mapping):
                    terms.extend(_search_terms(lineage.get("source_kind")))
                    terms.extend(_search_terms(lineage.get("fields") or {}))
            for metric in view.get("metrics") or []:
                if isinstance(metric, Mapping):
                    terms.extend(_search_terms(metric.get("source_kind")))
                    terms.extend(_search_terms(metric.get("values") or {}))
            for artifact in view.get("artifacts") or []:
                if isinstance(artifact, Mapping):
                    terms.extend(_search_terms(artifact.get("label")))
                    terms.extend(_search_terms(artifact.get("source_kind")))

        search_index[node_id] = " ".join(dict.fromkeys(term for term in terms if term))
    return search_index


def _html_json_value(value: Any) -> Any:
    """Return a JSON value that browsers can parse without non-standard numbers."""

    if isinstance(value, Mapping):
        return {str(key): _html_json_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_html_json_value(nested) for nested in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _html_json_payload(value: Any) -> str:
    payload = json.dumps(
        _html_json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return payload.replace("<", "\\u003c").replace(">", "\\u003e")


def _metadata_only_next_experiment(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    projected = dict(value)
    if "summary" in projected:
        projected["summary"] = ""
    return projected


def _renderer_graph_value(
    graph: Mapping[str, Any], *, disclose: bool
) -> dict[str, Any]:
    """Return the graph payload allowed to enter the self-contained HTML."""

    payload = _html_json_value(graph)
    if not isinstance(payload, dict):
        raise ResearchGitError("research DAG must be an object")
    if disclose:
        return payload

    payload["content_disclosed"] = False
    for node in payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("source") == "ara":
            node["summary"] = "experiment node"
        else:
            node["summary"] = f"{node.get('kind')} {node.get('id')}"
    for edge in payload.get("edges") or []:
        if isinstance(edge, dict):
            edge["role"] = ""

    frontier = payload.get("theory_frontier")
    if isinstance(frontier, dict):
        frontier["open_questions"] = []
        frontier["next_experiments"] = [
            _metadata_only_next_experiment(item)
            for item in frontier.get("next_experiments") or []
        ]
        frontier["next_experiment"] = _metadata_only_next_experiment(
            frontier.get("next_experiment")
        )
        for portfolio in frontier.get("portfolio_frontiers") or []:
            if isinstance(portfolio, dict):
                portfolio["next_experiment"] = _metadata_only_next_experiment(
                    portfolio.get("next_experiment")
                )
        pool = frontier.get("submission_candidate_pool")
        if isinstance(pool, dict):
            pool["decision"] = None
            pool["gate_status"] = None
        if "frontier_hash" in frontier:
            frontier["frontier_hash"] = canonical_content_hash(
                {key: value for key, value in frontier.items() if key != "frontier_hash"}
            )

    for insight in payload.get("claim_insights") or []:
        if not isinstance(insight, dict):
            continue
        insight["next_experiments"] = [
            _metadata_only_next_experiment(item)
            for item in insight.get("next_experiments") or []
        ]
        insight["next_experiment"] = _metadata_only_next_experiment(
            insight.get("next_experiment")
        )

    if "graph_hash" in payload:
        payload["graph_hash"] = canonical_content_hash(
            {
                key: value
                for key, value in payload.items()
                if key not in {"generated_at", "graph_hash"}
            }
        )
    return payload


def _sanitized_node_details(node_details: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy HTML-only details without granting artifact navigation capability."""

    payload = _html_json_value(node_details or {})
    if not isinstance(payload, dict):
        return {}
    for view in payload.values():
        if not isinstance(view, dict):
            continue
        artifacts = view.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact["href"] = None
            artifact["binding_verified"] = artifact.get("binding_verified") is True
    return payload


def _renderer_argument_payload(
    graph: Mapping[str, Any],
    argument_view: Mapping[str, Any] | None,
    *,
    disclose: bool,
    fallback_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an HTML-only fallback that never invents a zero-claim projection."""

    if argument_view is not None:
        if not isinstance(argument_view, Mapping):
            raise ResearchGitError("research argument sidecar must be an object")
        validate_research_argument_view(graph, argument_view)
        if disclose:
            return _html_json_value(argument_view)
    nodes = list(fallback_graph.get("nodes") or [])
    edges = list(fallback_graph.get("edges") or [])
    claim_count = sum(
        1
        for node in nodes
        if isinstance(node, Mapping)
        and node.get("source") == "research_vcs"
        and node.get("kind") == "claim"
    )
    categories = Counter(
        str(edge.get("category") or "") for edge in edges if isinstance(edge, Mapping)
    )
    ara_total = sum(
        1
        for node in nodes
        if isinstance(node, Mapping)
        and node.get("source") == "ara"
        and node.get("kind") == "experiment_node"
    )
    pool = (fallback_graph.get("theory_frontier") or {}).get(
        "submission_candidate_pool"
    ) or {}
    candidate_count = pool.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        candidate_count = None
    experiment_authorized = pool.get("experiment_authorized")
    if not isinstance(experiment_authorized, bool):
        experiment_authorized = None
    return {
        "schema_version": "xscientist.research-argument-render-fallback.v1",
        "projection_state": "EMPTY" if claim_count == 0 else "UNAVAILABLE",
        "registered_claims": [] if claim_count == 0 else None,
        "advisory_routes": [],
        "counters": {
            "registered_claims": claim_count,
            "unregistered_routes": 0,
            "closure": str(
                (fallback_graph.get("scientific_closure") or {}).get("status")
                or "not_applicable"
            ),
            "submission_candidates": candidate_count,
            "experiment_authorized": experiment_authorized,
            "direct_semantic_edges": {
                "count": sum(
                    categories.get(category, 0)
                    for category in ("argument", "challenge", "support")
                ),
                "total": len(edges),
            },
            "verification_edges": {
                "count": categories.get("verification", 0),
                "total": len(edges),
            },
            "provenance_context_evolution_edges": {
                "count": sum(
                    categories.get(category, 0)
                    for category in ("context", "evolution", "lineage")
                ),
                "total": len(edges),
            },
            "ara_claim_role_nodes": {"bound": 0, "total": ara_total},
            "edge_categories": dict(sorted(categories.items())),
        },
    }


def render_research_dag_html(
    graph: Mapping[str, Any],
    *,
    title: str = "XScientist Scientific Evidence DAG",
    node_details: Mapping[str, Any] | None = None,
    argument_view: Mapping[str, Any] | None = None,
    disclose_summaries: bool = True,
) -> str:
    """Render a self-contained browser with epistemic layers and claim drill-down."""

    effective_disclosure = bool(
        disclose_summaries and graph.get("content_disclosed") is True
    )
    rendered_graph = _renderer_graph_value(graph, disclose=effective_disclosure)
    # Read-only cross-plane observability. This never mutates the canonical
    # DAG and cannot create scientific authority or inferred relations.
    from .observability_binding import build_observability_binding_projection

    observability_binding = build_observability_binding_projection(graph)
    payload = _html_json_payload(rendered_graph)
    safe_details = (
        _sanitized_node_details(node_details) if effective_disclosure else {}
    )
    detail_payload = _html_json_payload(safe_details)
    search_payload = _html_json_payload(
        _build_dag_search_index(rendered_graph.get("nodes") or [], safe_details)
    )
    argument_value = _renderer_argument_payload(
        graph,
        argument_view,
        disclose=effective_disclosure,
        fallback_graph=rendered_graph,
    )
    argument_payload = _html_json_payload(argument_value)
    observability_binding_payload = _html_json_payload(observability_binding)
    registered_argument_claims = argument_value.get("registered_claims")
    advisory_argument_routes = argument_value.get("advisory_routes")
    advisory_only_class = (
        ' class="advisory-only"'
        if isinstance(registered_argument_claims, list)
        and not registered_argument_claims
        and isinstance(advisory_argument_routes, list)
        and bool(advisory_argument_routes)
        else ""
    )
    title_text = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_text}</title><style>
:root{{color-scheme:light dark;--bg:light-dark(#f7f8fb,#10131a);--panel:light-dark(#fff,#181d27);--ink:light-dark(#18202f,#eef2f8);--muted:light-dark(#667085,#9aa4b5);--line:light-dark(#c9d0dc,#3a4353);--support:#16855b;--challenge:#d14b43;--verify:#7657c8;--evolve:#2f70c9;--context:#b56a13;--theory:#0089a8;--boundary:#9a5b16;--warn:#c07818}}
	*{{box-sizing:border-box}}[hidden]{{display:none!important}}body{{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}header{{padding:18px 22px;background:var(--panel);border-bottom:1px solid var(--line)}}h1{{font-size:22px;margin:0 0 8px}}.stats,.controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}.stat{{color:var(--muted)}}.controls{{padding:12px 22px;border-bottom:1px solid var(--line);background:var(--panel)}}.control{{display:flex;gap:6px;align-items:center;min-width:0;max-width:100%}}.source-control #source{{width:457px}}label{{font-weight:500}}input,select{{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:7px 9px}}input{{min-width:220px}}#source-integrity{{padding:12px 22px;background:var(--panel);border-bottom:1px solid var(--line)}}#source-integrity h2{{font-size:15px;margin:0}}#source-integrity p{{margin:3px 0 8px;color:var(--muted)}}.source-integrity-disclosure>summary{{display:list-item;margin-top:4px;color:var(--muted);font-weight:600;cursor:pointer}}.source-table-scroll{{overflow-x:auto}}#source-integrity table{{min-width:980px}}#source-integrity th{{padding:5px 14px 5px 0;text-align:left;color:var(--muted);font-weight:500;border-bottom:1px solid var(--line)}}#source-integrity td{{padding:5px 14px 5px 0}}#source-integrity td:first-child{{width:auto;color:var(--ink)}}#source-integrity tr.source-unlinked td{{color:var(--warn)}}.source-path{{display:block;max-width:360px;color:var(--muted);overflow-wrap:anywhere}}main{{display:grid;grid-template-columns:minmax(0,1fr) 360px;min-height:520px;height:max(520px,calc(100vh - 315px))}}#canvas{{min-width:0;overflow:auto;padding:18px}}svg{{display:block;background:var(--panel);border:1px solid var(--line);border-radius:8px}}aside{{min-height:0;background:var(--panel);border-left:1px solid var(--line);padding:18px;overflow:auto;overflow-wrap:anywhere}}aside h2{{font-size:17px;margin:0 0 10px}}.edge{{fill:none;stroke:var(--line);stroke-width:1.5}}.edge.support{{stroke:var(--support)}}.edge.challenge{{stroke:var(--challenge);stroke-dasharray:5 4}}.edge.verification{{stroke:var(--verify)}}.edge.evolution{{stroke:var(--evolve)}}.edge.context{{stroke:var(--context);stroke-dasharray:2 3}}.edge.theory{{stroke:var(--theory)}}.edge.boundary{{stroke:var(--boundary);stroke-dasharray:7 3}}.edge.unknown{{stroke:var(--warn);stroke-dasharray:2 2}}.node rect{{fill:var(--panel);stroke:var(--line);stroke-width:1.5;rx:8}}.node.verified rect{{stroke:var(--support)}}.node.replayable rect{{stroke:var(--evolve)}}.node.contested rect{{stroke:var(--challenge);stroke-width:2}}.node.recorded rect{{stroke:var(--warn)}}.node.ui-aggregate rect{{stroke:var(--evolve);stroke-dasharray:5 3}}.node.source-unlinked rect{{stroke:var(--warn);stroke-dasharray:3 3}}.node.search-context{{opacity:.58}}.node.search-seed rect{{stroke-width:2.5}}.node.selected{{opacity:1}}.node.selected rect{{stroke-width:3}}.node text{{fill:var(--ink);font-size:12px;pointer-events:none}}.node .muted{{fill:var(--muted)}}.node:focus{{outline:none}}.node:focus rect{{stroke-width:3}}table{{width:100%;border-collapse:collapse}}td{{padding:7px 0;border-bottom:1px solid var(--line);vertical-align:top}}td:first-child{{width:96px;color:var(--muted)}}code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}ul{{padding-left:20px}}.node-jump{{display:inline;padding:0;border:0;background:none;color:var(--evolve);cursor:pointer;text-decoration:underline;text-align:left}}.node-jump:focus-visible{{outline:2px solid var(--evolve);outline-offset:2px}}.pass{{color:var(--support)}}.fail{{color:var(--challenge)}}.empty{{padding:28px;color:var(--muted)}}@media(max-width:900px){{.source-control{{width:100%}}.source-control #source{{width:0;min-width:0;flex:1 1 0}}.source-integrity-disclosure[open]>.source-table-scroll{{max-height:min(45vh,360px);overflow:auto}}main{{grid-template-columns:1fr;height:auto;min-height:0}}#canvas{{height:min(62vh,560px);min-height:320px}}aside{{max-height:70vh;border-left:0;border-top:1px solid var(--line)}}}}
	.node.rejected rect,.node.failed rect{{stroke:var(--challenge);stroke-width:2}}
	#argument-view{{padding:18px 22px;overflow-wrap:anywhere}}
	#observability-binding{{padding:12px 22px;background:var(--panel);border-bottom:1px solid var(--line);overflow-wrap:anywhere}}
	#observability-binding h2{{font-size:15px;margin:0}}
	#observability-binding p{{margin:3px 0 8px;color:var(--muted)}}
	#observability-binding table{{min-width:1080px;table-layout:fixed}}
	#observability-binding th{{padding:5px 12px 5px 0;text-align:left;color:var(--muted);font-weight:500;border-bottom:1px solid var(--line)}}
	#observability-binding td{{padding:5px 12px 5px 0;overflow-wrap:anywhere}}
	#observability-binding td:first-child{{width:auto}}
	.argument-counters{{display:flex;gap:8px 16px;flex-wrap:wrap;padding-bottom:14px;border-bottom:1px solid var(--line)}}
	.argument-counters strong{{font-weight:600}}
	.argument-lanes{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;margin-top:18px}}
	.argument-lane{{min-width:0}}
	#argument-view.advisory-only .argument-lanes{{grid-template-columns:minmax(0,1fr)}}
	#argument-view.advisory-only .argument-lane:first-child{{display:flex;align-items:baseline;gap:4px 16px;flex-wrap:wrap;padding-bottom:12px;border-bottom:1px solid var(--line)}}
	#argument-view.advisory-only .argument-lane:first-child>h2,#argument-view.advisory-only .argument-lane:first-child>.argument-authority{{margin:0}}
	#argument-view.advisory-only .argument-lane:first-child>#registered-claims{{flex:1 1 360px}}
	#argument-view.advisory-only .argument-lane:first-child .argument-empty{{display:flex;align-items:baseline;gap:5px 9px;flex-wrap:wrap;padding:0}}
	#argument-view.advisory-only .argument-lane:first-child .argument-empty strong{{display:inline;margin:0}}
	.argument-lane>h2{{font-size:17px;margin:0 0 4px}}
	.argument-authority{{margin:0 0 12px;color:var(--muted)}}
	.argument-authority strong{{color:var(--ink)}}
	.argument-empty{{padding:20px 0;color:var(--muted)}}
	.argument-empty strong{{display:block;color:var(--ink);margin-bottom:5px}}
	.argument-card{{display:block;width:100%;margin:0 0 12px;padding:12px;background:var(--panel);border:1px solid var(--line);border-radius:8px;color:var(--ink);overflow-wrap:anywhere}}
	.argument-card:focus-visible{{outline:2px solid var(--evolve);outline-offset:2px}}
	.argument-card h3{{font-size:15px;margin:0 0 8px}}
	.argument-card p{{margin:5px 0}}
	.argument-card table{{margin-top:8px}}
	.argument-card td:first-child{{width:132px}}
	.argument-binding-list,.argument-findings{{margin:0;padding-left:18px}}
	.argument-binding-list li,.argument-findings li{{margin:0 0 7px}}
	.argument-binding-list code{{display:block;overflow-wrap:anywhere;word-break:break-word}}
	.argument-binding-hash{{color:var(--muted)}}
	.argument-gates{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:3px 14px}}
	.argument-gate{{display:flex;justify-content:space-between;gap:8px;padding:4px 0;border-bottom:1px solid var(--line)}}
	.argument-gate strong{{font-weight:600}}
	.argument-flow{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:10px}}
	.argument-step{{min-width:0;padding-top:7px;border-top:2px solid var(--line)}}
	.argument-step h4{{font-size:12px;margin:0 0 5px;color:var(--muted)}}
	.argument-step ul{{margin:4px 0;padding-left:17px}}
	.argument-step code{{overflow-wrap:anywhere;word-break:break-word}}
	.argument-bad{{color:var(--challenge)}}
	.argument-good{{color:var(--support)}}
	.rendering-legend{{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;padding:10px 22px;color:var(--muted);font-size:12px;border-bottom:1px solid var(--line);background:var(--panel)}}
	.rendering-legend-title{{font-weight:600;color:var(--ink)}}
	.legend-item{{display:inline-flex;align-items:center;gap:5px}}
	.legend-swatch{{display:inline-block;width:22px;height:12px;border:1.5px solid var(--line);border-radius:4px;background:var(--panel)}}
	.legend-swatch.edge{{width:22px;height:0;border:0;border-top:2px solid var(--line);border-radius:0}}
	.legend-swatch.support{{border-color:var(--support)}}
	.legend-swatch.challenge{{border-color:var(--challenge);border-style:dashed}}
	.legend-swatch.verification{{border-color:var(--verify)}}
	.legend-swatch.evolution{{border-color:var(--evolve)}}
	.legend-swatch.context{{border-color:var(--context);border-style:dotted}}
	.legend-swatch.theory{{border-color:var(--theory)}}
	.legend-swatch.boundary{{border-color:var(--boundary);border-style:dashed}}
	.legend-swatch.node-verified{{border-color:var(--support)}}
	.legend-swatch.node-contested{{border-color:var(--challenge);border-width:2px}}
	.legend-swatch.node-aggregate{{border-color:var(--evolve);border-style:dashed}}
	.legend-note{{flex-basis:100%}}
	.compact-disclosure,.filter-controls{{display:contents}}
	.compact-disclosure>summary{{display:none}}
	@media(min-width:601px){{.compact-disclosure::details-content{{display:contents}}}}
	@media(max-width:600px){{.compact-disclosure{{display:block;min-width:0}}.compact-disclosure>summary{{display:list-item;min-height:44px;cursor:pointer;font-weight:500}}.filter-disclosure>summary{{padding:11px 0}}.filter-disclosure[open]{{flex-basis:100%}}.filter-controls{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding-top:4px}}.filter-controls input{{min-width:0;max-width:100%}}.legend-disclosure>summary{{padding:12px 22px;background:var(--panel);border-bottom:1px solid var(--line)}}.legend-disclosure .rendering-legend-title{{display:none}}}}
	@media(max-width:900px){{.argument-lanes{{grid-template-columns:1fr}}.argument-flow{{grid-template-columns:1fr 1fr}}}}
	@media(max-width:520px){{#argument-view{{padding:14px}}.argument-flow{{grid-template-columns:1fr}}.argument-card td:first-child{{width:105px}}.argument-gates{{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}}}}
	</style></head><body><header><h1>{title_text}</h1><div class="stats" id="stats"></div></header>
			<section class="controls" aria-label="Graph filters"><span class="control"><label for="view">View</label><select id="view"><option value="overview">Overview</option><option value="ara_spine">ARA decision spine</option><option value="full">Full audit</option><option value="argument">Argument</option></select></span><details class="compact-disclosure filter-disclosure" id="filter-disclosure" data-compact-disclosure data-graph-control open><summary>Filters</summary><div class="filter-controls"><span class="control source-control" data-graph-control><label for="source">Source</label><select id="source"><option value="all">All audit sources</option><option value="research_vcs">Research VCS</option></select></span><span class="control" data-graph-control><label for="search">Search</label><input id="search" type="search" placeholder="statement, exact ID, or kind"></span><span class="control" data-graph-control><label for="layer">Layer</label><select id="layer"><option value="">All layers</option><option>strategy</option><option>execution</option><option>evidence</option><option>theory</option><option>decision_memory</option><option>evolution</option></select></span><span class="control" data-graph-control><label for="kind">Kind</label><select id="kind"><option value="">All kinds</option></select></span><span class="control" data-graph-control><label for="proof">Verification</label><select id="proof"><option value="">All levels</option><option>verified</option><option>replayable</option><option>traceable</option><option>recorded</option><option>contested</option></select></span><span class="control" data-graph-control><label for="relation">Relation</label><select id="relation"><option value="">All relations</option></select></span></div></details></section>
			<details class="compact-disclosure legend-disclosure" id="legend-disclosure" data-compact-disclosure open><summary>Rendering legend</summary><section class="rendering-legend" data-presentation-only="true" aria-label="Rendering legend"><span class="rendering-legend-title">Rendering legend</span><span class="legend-item"><span class="legend-swatch edge"></span>relation</span><span class="legend-item"><span class="legend-swatch support"></span>support</span><span class="legend-item"><span class="legend-swatch challenge"></span>challenge / contradiction</span><span class="legend-item"><span class="legend-swatch verification"></span>verification</span><span class="legend-item"><span class="legend-swatch evolution"></span>evolution</span><span class="legend-item"><span class="legend-swatch context"></span>context</span><span class="legend-item"><span class="legend-swatch theory"></span>theory</span><span class="legend-item"><span class="legend-swatch boundary"></span>boundary</span><span class="legend-item"><span class="legend-swatch node-verified"></span>verified node</span><span class="legend-item"><span class="legend-swatch node-contested"></span>contested node</span><span class="legend-item"><span class="legend-swatch node-aggregate"></span>UI aggregate</span><span class="legend-note">Presentation only; colors and borders are derived from the bound payload and do not add scientific evidence.</span></section></details>
			<section id="argument-view"{advisory_only_class} aria-label="Claim-centric scientific argument" hidden><div class="argument-counters" id="argument-counters"></div><div class="argument-lanes"><section class="argument-lane" aria-labelledby="registered-lane-title"><h2 id="registered-lane-title">REGISTERED CLAIM CLOSURE</h2><p class="argument-authority">Research VCS authority only</p><div id="registered-claims"></div></section><section class="argument-lane" aria-labelledby="advisory-lane-title"><h2 id="advisory-lane-title">UNREGISTERED SPRINT FRONTIER</h2><p class="argument-authority"><strong>UNREGISTERED</strong> · byte-bound advisory artifacts only</p><div id="advisory-routes"></div></section></div></section>
                <section id="observability-binding" aria-labelledby="observability-binding-title"><h2 id="observability-binding-title">ARA / Research VCS binding</h2><details class="source-integrity-disclosure"><summary id="observability-binding-summary">UNBOUND</summary><p>Presentation only. Hashes identify recorded source lineage, not a signature, admission receipt, claim role, or latent trajectory binding.</p><div class="source-table-scroll"><table><thead><tr><th>State</th><th>ARA source</th><th>VCS / ARA snapshot commits</th><th>Manifest hash</th><th>Graph hash</th><th>Exact anchors</th><th>Research DAG binding hash</th><th>Reason</th></tr></thead><tbody id="observability-binding-body"></tbody></table></div></details></section>
                <section id="source-integrity" aria-labelledby="source-integrity-title"><h2 id="source-integrity-title">ARA source snapshot integrity and exact Research Object links</h2><details class="source-integrity-disclosure" id="source-integrity-disclosure"><summary id="source-integrity-toggle">Source snapshot audit</summary><p id="source-integrity-summary"></p><div class="source-table-scroll"><table><thead><tr><th>Source snapshot</th><th>ARA nodes</th><th>Checkpoint snapshot</th><th>Manifest state</th><th>Graph binding</th><th>Exact Research Object links</th><th>Context links</th><th>Source DAG</th></tr></thead><tbody id="source-integrity-body"></tbody></table></div></details></section>
			<main id="graph-main"><section id="canvas" aria-label="Scientific evidence graph"></section><aside><h2>Selected graph item</h2><div id="details" aria-live="polite">Select a node to inspect its reasoning.</div></aside></main>
<script id="dag-data" type="application/json">{payload}</script>
<script id="detail-data" type="application/json">{detail_payload}</script>
<script id="search-data" type="application/json">{search_payload}</script>
    <script id="argument-data" type="application/json">{argument_payload}</script>
    <script id="observability-binding-data" type="application/json">{observability_binding_payload}</script><script>
{_RESEARCH_DAG_SEARCH_HELPERS_JS}
    const graph=JSON.parse(document.getElementById('dag-data').textContent),detailViews=JSON.parse(document.getElementById('detail-data').textContent),searchIndex=JSON.parse(document.getElementById('search-data').textContent),argumentData=JSON.parse(document.getElementById('argument-data').textContent),observabilityBinding=JSON.parse(document.getElementById('observability-binding-data').textContent),allNodes=graph.nodes||[],allEdges=graph.edges||[],sources=graph.sources||[],araSources=sources.filter(source=>/^ara:\\d+$/.test(String(source.name))),sourceByName=new Map(sources.map(source=>[String(source.name),source])),sourceBinding=sourceBindingSummary(sources),sourceNamesWithoutExactLinks=new Set(araSources.filter(source=>Number(source.binding_count||0)===0).map(source=>String(source.name))),byId=new Map(allNodes.map(n=>[n.id,n])),claimInsights=new Map((graph.claim_insights||[]).map(x=>[x.claim_id,x])),frontier=graph.theory_frontier||{{}},submissionPool=frontier.submission_candidate_pool||{{}},historicalHypotheses=new Set(frontier.historical_hypothesis_ids||[]),contestedHypotheses=new Set(frontier.contested_hypothesis_ids||[]),openTheoryHypotheses=new Set(frontier.open_theory_hypothesis_ids||frontier.active_hypothesis_ids||[]);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
                const view=document.getElementById('view'),sourceControl=document.getElementById('source'),kind=document.getElementById('kind'),layer=document.getElementById('layer'),proof=document.getElementById('proof'),relation=document.getElementById('relation'),search=document.getElementById('search'),stats=document.getElementById('stats'),details=document.getElementById('details'),observabilityBindingSection=document.getElementById('observability-binding'),observabilityBindingSummary=document.getElementById('observability-binding-summary'),observabilityBindingBody=document.getElementById('observability-binding-body'),sourceIntegrity=document.getElementById('source-integrity'),sourceIntegrityDisclosure=document.getElementById('source-integrity-disclosure'),sourceIntegrityToggle=document.getElementById('source-integrity-toggle'),sourceIntegritySummary=document.getElementById('source-integrity-summary'),sourceIntegrityBody=document.getElementById('source-integrity-body'),graphMain=document.getElementById('graph-main'),argumentView=document.getElementById('argument-view'),argumentCounters=document.getElementById('argument-counters'),registeredClaims=document.getElementById('registered-claims'),advisoryRoutes=document.getElementById('advisory-routes'),graphControls=[...document.querySelectorAll('[data-graph-control]')];let selectedId=null,currentDisplayById=new Map(),currentProjection={{aggregateCount:0,collapsedNodeCount:0,expandedSourceName:null}};
[...new Set(allNodes.map(n=>n.kind))].sort().forEach(v=>kind.insertAdjacentHTML('beforeend',`<option>${{esc(v)}}</option>`));
			if(relation)[...new Set(allEdges.map(e=>String(e.category||'unknown')))].sort().forEach(v=>relation.insertAdjacentHTML('beforeend',`<option value="${{esc(v)}}">${{esc(v)}}</option>`));
araSources.forEach(item=>sourceControl.insertAdjacentHTML('beforeend',`<option value="${{esc(item.name)}}">${{esc(sourceDisplayName(item))}} [${{esc(item.name)}}]</option>`));
const submissionCount=Number.isInteger(submissionPool.candidate_count)?submissionPool.candidate_count:(submissionPool.status==='ambiguous'?'ambiguous':'unreported'),submissionAuthority=submissionPool.status==='reported'?(submissionPool.source_gate_authoritative?'authoritative gate':'recorded gate'):'no single gate',experimentStatus=submissionPool.experiment_authorized===true?'authorized':(submissionPool.experiment_authorized===false?'not authorized':'unreported');
    const sourceState=(source,key)=>esc(((source[key]||{{}}).state)||'unreported');
    const bindingStateClass=state=>state==='verified'?'pass':(state==='invalid'?'fail':'');
    const bindingLabel=value=>value?`<code>${{esc(value)}}</code>`:'<span class="stat">unavailable</span>';
    const bindingRows=Array.isArray(observabilityBinding.rows)?observabilityBinding.rows:[];
    observabilityBindingSummary.innerHTML=`<strong class="${{bindingStateClass(observabilityBinding.state)}}">${{esc(String(observabilityBinding.state||'unbound').toUpperCase())}}</strong> · ${{esc(observabilityBinding.verified_count||0)}} verified · ${{esc(bindingRows.filter(row=>row.state==='unbound').length)}} unbound · ${{esc(observabilityBinding.invalid_count||0)}} invalid · presentation only`;
    observabilityBindingBody.innerHTML=bindingRows.length?bindingRows.map(row=>`<tr><td><strong class="${{bindingStateClass(row.state)}}">${{esc(String(row.state||'unbound').toUpperCase())}}</strong></td><td><code>${{esc(row.source||'-')}}</code></td><td><code>${{esc(row.vcs_commit||'-')}}</code><br><code>${{esc(row.snapshot_ref||'-')}}</code></td><td>${{bindingLabel(row.manifest_hash)}}</td><td>${{bindingLabel(row.graph_hash)}}</td><td>${{esc(row.anchor_count||0)}}</td><td>${{bindingLabel(row.binding_hash)}}</td><td>${{esc((row.reasons||[]).join(', ')||'source-lineage checks passed')}}</td></tr>`).join(''):'<tr><td colspan="8">UNBOUND: no ARA source snapshot was supplied.</td></tr>';
sourceIntegritySummary.innerHTML=`<strong>${{sourceBinding.exactLinkCount}} exact Research Object links across ${{sourceBinding.araSourceCount}} ARA source snapshots</strong> · ${{sourceBinding.nodesWithoutExactLinks}} nodes in ${{sourceBinding.sourcesWithoutExactLinks}} snapshots have no exact link · historical/successor identity is not inferred`;
sourceIntegrityToggle.textContent=`${{sourceBinding.araSourceCount}} snapshots · ${{sourceBinding.exactLinkCount}} exact links · ${{sourceBinding.sourcesWithoutExactLinks}} unlinked`;
sourceIntegrityBody.innerHTML=araSources.length?araSources.map(item=>{{const integrity=item.integrity||{{}},noExactLinks=Number(item.binding_count||0)===0,snapshot=String(item.snapshot_ref||'unreported'),snapshotLabel=snapshot.length>16?snapshot.slice(0,12):snapshot,path=String(item.repository_path||'external');return`<tr class="${{noExactLinks?'source-unlinked':''}}"><td><strong>${{esc(sourceDisplayName(item))}}</strong><br><code>${{esc(item.name)}}</code><small class="source-path">${{esc(path)}}</small></td><td>${{esc(item.object_count||0)}}</td><td><code title="${{esc(snapshot)}}">${{esc(snapshotLabel)}}</code></td><td>${{sourceState(item,'manifest_integrity')}}</td><td>${{sourceState(item,'graph_binding')}}</td><td>${{esc(item.binding_count||0)}}${{noExactLinks?' (none; lineage not inferred)':''}}</td><td>${{esc(item.context_binding_count||0)}}</td><td>${{esc(integrity.error_count||0)}} errors / ${{esc(integrity.warning_count||0)}} warnings</td></tr>`}}).join(''):'<tr><td colspan="8">No ARA source snapshots in this snapshot.</td></tr>';
const sourceIntegrityHasIssues=araSources.some(item=>{{const integrity=item.integrity||{{}};return Number(integrity.error_count||0)>0||Number(integrity.warning_count||0)>0}});
sourceIntegrityDisclosure.open=sourceIntegrityHasIssues;
			const renderStats=(nodes,edges,directCount,contextCount,hasQuery)=>{{const searchStats=hasQuery?`<span class="stat">direct matches ${{directCount}}</span><span class="stat">1-hop context ${{contextCount}}</span><span class="stat">filters apply to direct matches</span>`:'',collapseStats=currentProjection.aggregateCount&&currentProjection.presentation!=='ara_decision_spine'?`<span class="stat">collapsed ${{currentProjection.collapsedNodeCount}} ARA nodes in ${{currentProjection.aggregateCount}} UI source-snapshot aggregates</span>`:'',spineStats=currentProjection.presentation==='ara_decision_spine'?`<span class="stat">presentation-only ARA spine: at most ${{currentProjection.maxNodes||64}} nodes, ${{currentProjection.omittedNodeCount||0}} omitted</span>`:'',expanded=currentProjection.expandedSourceNames||[],expansionStats=expanded.length?`<span class="stat">expanded ${{esc(expanded.join(', '))}}</span>`:'';stats.innerHTML=`<span class="stat">recorded ${{allNodes.length}} nodes</span><span class="stat">displayed ${{nodes.length}} items</span>${{collapseStats}}${{spineStats}}${{expansionStats}}<span class="stat">${{edges.length}}/${{allEdges.length}} authoritative relations</span>${{searchStats}}<span class="stat">DAG ${{graph.integrity.is_dag?'valid':'blocked'}}</span><span class="stat">closure ${{esc(graph.scientific_closure.status)}}</span><span class="stat">hypothesis history ${{historicalHypotheses.size}}</span><span class="stat">contested hypotheses ${{contestedHypotheses.size}}</span><span class="stat">open hypotheses ${{openTheoryHypotheses.size}}</span><span class="stat">submission candidates ${{esc(submissionCount)}} (${{esc(submissionAuthority)}})</span><span class="stat">experiment ${{esc(experimentStatus)}}</span><span class="stat">open anomalies ${{esc(graph.strategy_summary.open_anomaly_count||0)}}</span><span class="stat"><code>${{esc(graph.commit.slice(0,12))}}</code></span>`}};
const nodeRef=x=>byId.has(String(x))?`<button type="button" class="node-jump" data-node-id="${{esc(x)}}"><code>${{esc(x)}}</code></button>`:`<code>${{esc(x)}}</code>`;
const idList=xs=>(xs&&xs.length?`<ul>${{xs.map(x=>`<li>${{nodeRef(x)}}</li>`).join('')}}</ul>`:'None');
const fieldLabel=s=>String(s).replaceAll('_',' ');
const valueView=v=>{{if(Array.isArray(v))return v.length?`<ul>${{v.map(x=>`<li>${{valueView(x)}}</li>`).join('')}}</ul>`:'-';if(v&&typeof v==='object'){{const entries=Object.entries(v);return entries.length?`<table>${{entries.map(([k,x])=>`<tr><td>${{esc(fieldLabel(k))}}</td><td>${{valueView(x)}}</td></tr>`).join('')}}</table>`:'-'}}return esc(v??'-')}};
const argumentRatio=value=>{{const row=value||{{}};return`${{esc(row.count??0)}} / ${{esc(row.total??0)}}`}};
const argumentBindingRows=value=>{{const rows=Array.isArray(value)?value:(value&&typeof value==='object'?[value]:[]);return rows.length?`<ul class="argument-binding-list">${{rows.map(row=>`<li><code>${{esc(row.path||'-')}}</code><span class="argument-binding-hash">sha256 ${{esc(row.raw_sha256||'-')}}</span></li>`).join('')}}</ul>`:'-'}};
const argumentGateRows=value=>{{const rows=Array.isArray(value)?value:[];return rows.length?`<div class="argument-gates">${{rows.map(row=>{{const status=String(row.status||'UNREVIEWED'),statusClass=status==='PASS'?'argument-good':(status==='FAIL'?'argument-bad':'');return`<span class="argument-gate"><code>${{esc(row.gate_id||'-')}}</code><strong class="${{statusClass}}">${{esc(status)}}</strong></span>`}}).join('')}}</div>`:'-'}};
const argumentFindingRows=value=>{{const rows=Array.isArray(value)?value:[];return rows.length?`<ul class="argument-findings">${{rows.map(row=>`<li><strong>${{esc(row.severity||'-')}}</strong> ${{esc(row.statement||'-')}}</li>`).join('')}}</ul>`:'-'}};
const argumentIdList=value=>idList(Array.isArray(value)?value:[]);
const argumentAraBindings=value=>{{const rows=Array.isArray(value)?value:[];return rows.length?`<ul>${{rows.map(binding=>{{const metric=binding.metric||{{}},reproduction=binding.reproduction_id||binding.reproduction_state||'unreported';return`<li><strong>${{esc(binding.role||'unreported')}}</strong> via <code>${{esc(binding.evidence_object_id||'-')}}</code><br>${{argumentIdList(binding.projected_node_ids)}}<small>metric ${{esc(metric.name||'hidden')}} · ${{esc(metric.direction||'-')}} · threshold ${{esc(metric.threshold??'-')}} · controls ${{esc((binding.control_ids||[]).join(', ')||'none')}} · reproduction ${{esc(reproduction)}}</small></li>`}}).join('')}}</ul>`:'None'}};
const argumentReproductions=value=>{{const rows=Array.isArray(value)?value:[];return rows.length?`<ul>${{rows.map(row=>{{const bindings=Array.isArray(row.claim_closure_bindings)?row.claim_closure_bindings:[];return`<li><code>${{esc(row.reproduction_id||'-')}}</code> · ${{esc(row.state||'unreported')}}${{bindings.length?`<br>closure ${{esc(bindings.map(binding=>binding.closure_hash||'-').join(', '))}}`:' · no exact claim-closure binding'}}</li>`}}).join('')}}</ul>`:'None'}};
const registeredClaimCard=claim=>{{const statement=claim.statement||'Statement hidden by metadata-only export',scope=claim.scope===undefined?'hidden':valueView(claim.scope),ready=claim.decision_ready===true;return`<article class="argument-card" tabindex="0" aria-label="Registered claim ${{esc(claim.claim_id||'unknown')}}"><h3>${{esc(statement)}}</h3><p><code>${{esc(claim.claim_id||'-')}}</code> · ${{esc(claim.state||'unreported')}} · ${{esc(claim.depth_level||'unreported')}} · <strong class="${{ready?'argument-good':'argument-bad'}}">${{ready?'DECISION READY':'NOT DECISION READY'}}</strong></p><div class="argument-flow"><section class="argument-step"><h4>CLAIM</h4><p>scope hash<br><code>${{esc(claim.scope_hash||'-')}}</code></p><p>closure ${{esc(claim.closure_state||'absent')}}</p></section><section class="argument-step"><h4>SCOPE / ASSUMPTIONS</h4><p>${{scope}}</p><p>estimands ${{argumentIdList(claim.estimand_ids)}}</p><p>assumptions ${{argumentIdList(claim.assumption_ids)}}</p><p>boundaries ${{argumentIdList(claim.boundary_ids)}}</p></section><section class="argument-step"><h4>SUPPORT + CHALLENGE</h4><p>direct support ${{argumentIdList(claim.direct_supporting_ids)}}</p><p>direct challenge ${{argumentIdList(claim.direct_challenging_ids)}}</p><p>closure evidence ${{argumentIdList(claim.closure_evidence_ids)}}</p><p>ARA exact roles ${{argumentAraBindings(claim.ara_claim_role_bindings)}}</p></section><section class="argument-step"><h4>REVIEW / REPRODUCTION</h4><p>independent reviews ${{argumentIdList(claim.independent_review_ids)}}</p><p>${{argumentReproductions(claim.reproductions)}}</p></section><section class="argument-step"><h4>GATE</h4><p>${{argumentIdList(claim.gate_ids)}}</p><p>gaps ${{valueView(claim.gaps||[])}}</p><p>rejected ARA roles ${{valueView(claim.ara_binding_rejections||[])}}</p></section></div></article>`}};
const advisoryRouteCard=route=>{{const blocked=route.route_state==='BLOCKED';if(blocked)return`<article class="argument-card" tabindex="0" aria-label="Blocked unregistered route ${{esc(route.route_id||'unknown')}}"><h3>${{esc(route.route_id||'UNREGISTERED ROUTE')}}</h3><p><strong>UNREGISTERED</strong> · <span class="argument-bad">HASH_MISMATCH / BLOCKED</span></p>${{valueView(route.binding_failures||[])}}</article>`;const question=route.question?`<p>${{esc(route.question)}}</p>`:'<p>Substantive fields hidden by metadata-only export.</p>';return`<article class="argument-card" tabindex="0" aria-label="Unregistered advisory route ${{esc(route.route_id||'unknown')}}"><h3>${{esc(route.route_id||'UNREGISTERED ROUTE')}}</h3><p><strong>UNREGISTERED</strong> · ${{esc(route.binding_state||'unreported')}} · ${{esc(route.registration_state||'absent')}}</p>${{question}}<table><tr><td>Artifact bindings</td><td>${{argumentBindingRows(route.artifact_bindings)}}</td></tr><tr><td>Gate contract</td><td>${{argumentBindingRows(route.gate_contract_binding)}}</td></tr>${{route.gate_results?`<tr><td>Gate results</td><td>${{argumentGateRows(route.gate_results)}}</td></tr>`:''}}${{route.open_findings?`<tr><td>Open findings</td><td>${{argumentFindingRows(route.open_findings)}}</td></tr>`:''}}${{route.novelty_disposition?`<tr><td>Novelty</td><td>${{esc(route.novelty_disposition)}}</td></tr>`:''}}${{route.permitted_next_action?`<tr><td>Permitted next action</td><td>${{esc(route.permitted_next_action)}}</td></tr>`:''}}</table></article>`}};
function renderArgumentView(){{const counters=(argumentData||{{}}).counters||{{}},claims=Array.isArray(argumentData.registered_claims)?argumentData.registered_claims:null,routes=Array.isArray(argumentData.advisory_routes)?argumentData.advisory_routes:[];argumentCounters.innerHTML=`<span><strong>registered claims</strong> ${{esc(counters.registered_claims??'unavailable')}}</span><span><strong>unregistered routes</strong> ${{esc(counters.unregistered_routes??0)}}</span><span><strong>direct semantic edges</strong> ${{argumentRatio(counters.direct_semantic_edges)}}</span><span><strong>verification edges</strong> ${{argumentRatio(counters.verification_edges)}}</span><span><strong>provenance/context/evolution edges</strong> ${{argumentRatio(counters.provenance_context_evolution_edges)}}</span><span><strong>ARA exact claim-role nodes</strong> ${{argumentRatio(counters.ara_claim_role_nodes)}}</span>`;if(claims===null||(claims.length===0&&Number(counters.registered_claims||0)>0))registeredClaims.innerHTML='<div class="argument-empty"><strong>ARGUMENT PROJECTION UNAVAILABLE</strong>Registered claims exist, but no exact claim-sidecar projection was supplied.</div>';else if(!claims.length)registeredClaims.innerHTML=`<div class="argument-empty"><strong>NO REGISTERED CLAIMS</strong>claim_count=0 | closure=${{esc(counters.closure||'not_applicable')}} | candidates=${{esc(counters.submission_candidates??'unreported')}} | experiment=${{esc(counters.experiment_authorized??'unreported')}}</div>`;else registeredClaims.innerHTML=claims.map(registeredClaimCard).join('');if(routes.length)advisoryRoutes.innerHTML=routes.map(advisoryRouteCard).join('');else advisoryRoutes.innerHTML=`<div class="argument-empty"><strong>${{argumentData.advisory_index_binding?'NO UNREGISTERED ROUTES INDEXED':'NO ADVISORY INDEX SUPPLIED'}}</strong>Advisory work is never inferred by scanning repository files.</div>`}}
const displaySummary=n=>{{const fields=(detailViews[n.id]||{{}}).fields||{{}};return String(fields.analysis||fields.summary||fields.plan_excerpt||fields.scientific_outcome||n.summary||n.id)}};
const nodeView=view=>{{if(!view)return'';const detailFields=Object.entries(view.fields||{{}}).map(([k,v])=>`<tr><td>${{esc(fieldLabel(k))}}</td><td>${{valueView(v)}}</td></tr>`).join('');const gateFields=Object.entries(view.gate||{{}}).map(([k,v])=>`<tr><td>${{esc(fieldLabel(k))}}</td><td>${{valueView(v)}}</td></tr>`).join('');const lineage=(view.lineage||[]).map(x=>`<h4>${{esc(x.source_kind)}} ${{nodeRef(x.source_id)}}</h4>${{valueView(x.fields)}}`).join('');const metrics=(view.metrics||[]).map(m=>`<h4>${{esc(m.source_kind)}} ${{nodeRef(m.source_id)}}</h4>${{valueView(m.values)}}`).join('');const artifacts=(view.artifacts||[]).map(a=>{{const path=a.logical_path||a.pointer_path||'',digest=a.content_hash?`<br><code>${{esc(a.content_hash)}}</code>`:'',binding=a.binding_verified?'binding verified':'binding unverified',reason=a.binding_reason?`<br><small>binding reason: ${{esc(a.binding_reason)}}${{a.binding_checked_path?` (checked: ${{esc(a.binding_checked_path)}})`:''}}</small>`:'';return`<li>${{esc(a.label)}} <small>(${{esc(a.source_kind)}} <code>${{esc(a.source_id)}}</code>; ${{binding}})</small>${{path?`<br><code>${{esc(path)}}</code>`:''}}${{digest}}${{reason}}</li>`}}).join('');return`${{detailFields?`<h3>${{esc(view.title||'Scientific detail')}}</h3><table>${{detailFields}}</table>`:''}}${{gateFields?`<h3>Gate decision</h3><table>${{gateFields}}</table>`:''}}${{lineage?`<h3>Evidence lineage</h3>${{lineage}}`:''}}${{metrics?`<h3>Metrics</h3>${{metrics}}`:''}}${{artifacts?`<h3>Artifacts</h3><ul>${{artifacts}}</ul>`:''}}`}};
		function focusNode(id){{if(!byId.has(id))return;if(view.value==='argument')view.value='overview';selectedId=id;search.value=id;sourceControl.value='all';layer.value='';kind.value='';proof.value='';if(relation)relation.value='';render()}}
		function wireDetailJumps(container=details){{container.querySelectorAll('.node-jump').forEach(el=>el.addEventListener('click',()=>focusNode(el.dataset.nodeId)))}}
		function detail(n){{
			if(!n)return;
			selectedId=n.id;
			document.querySelectorAll('.node').forEach(el=>el.classList.toggle('selected',el.dataset.id===selectedId));
			if(n.ui_aggregate){{
				const item=n.aggregate||{{}},interpretation=item.projection==='ara_decision_spine'?'Presentation-only decision spine; omitted ARA nodes remain available in Full audit.':(item.no_exact_links?'No exact Research Object link is recorded; historical or successor identity is not inferred.':'Exact Research Object links are listed below; they do not imply that every ARA node is individually linked.');
				details.className='';
				details.innerHTML=`<p><strong>UI aggregate only.</strong> This item summarizes one checkpointed ARA source snapshot and is not a scientific object or scientific relation.</p><table><tr><td>Snapshot</td><td>${{esc(item.display_name||n.summary)}} <code>${{esc(item.source_name||'-')}}</code></td></tr><tr><td>Repository path</td><td><code>${{esc(item.repository_path||'-')}}</code></td></tr><tr><td>Checkpoint ref</td><td><code>${{esc(item.snapshot_ref||'-')}}</code></td></tr><tr><td>ARA nodes</td><td>${{esc(item.node_count||0)}}</td></tr>${{item.projection==='ara_decision_spine'?`<tr><td>Decision spine</td><td>${{esc(item.selected_node_count||0)}} shown / ${{esc(item.node_count||0)}} total; ${{esc(item.omitted_node_count||0)}} omitted</td></tr>`:''}}<tr><td>Manifest integrity</td><td>${{esc(item.manifest_state||'unreported')}}</td></tr><tr><td>Graph binding</td><td>${{esc(item.graph_binding_state||'unreported')}}</td></tr><tr><td>Exact Research Object links</td><td>${{idList(item.bound_object_ids)}}</td></tr><tr><td>Context links</td><td>${{idList(item.context_bound_object_ids)}}</td></tr><tr><td>Source integrity</td><td>${{esc(item.source_error_count||0)}} errors / ${{esc(item.source_warning_count||0)}} warnings</td></tr><tr><td>Interpretation</td><td>${{esc(interpretation)}}</td></tr></table>`;
				wireDetailJumps();
				return;
			}}
			const checks=(n.proof.checks||[]).map(c=>`<li class="${{c.passed?'pass':'fail'}}">${{c.passed?'✓':'✗'}} ${{esc(c.label)}} <small>(${{esc(c.layer)}})</small></li>`).join(''),insight=claimInsights.get(n.id),hypothesisScope=n.kind==='hypothesis'?(contestedHypotheses.has(n.id)?'contested theory history':(openTheoryHypotheses.has(n.id)?'open theory frontier':(historicalHypotheses.has(n.id)?'historical theory':'unclassified'))):null,sourceName=sourceNameForNode(n),source=sourceByName.get(sourceName)||{{}},manifestIntegrity=source.manifest_integrity||{{}},graphBinding=source.graph_binding||{{}},sourceIntegrity=source.integrity||{{}},sourceAuditRows=/^ara:\\d+$/.test(sourceName)?`<tr><td>Source snapshot</td><td>${{esc(sourceDisplayName(source))}} <code>${{esc(sourceName)}}</code></td></tr><tr><td>Repository path</td><td><code>${{esc(source.repository_path||'-')}}</code></td></tr><tr><td>Manifest integrity</td><td>${{esc(manifestIntegrity.state||'unreported')}}</td></tr><tr><td>Graph binding</td><td>${{esc(graphBinding.state||'unreported')}}</td></tr><tr><td>Exact Research Object links</td><td>${{idList(source.bound_object_ids||[])}}</td></tr><tr><td>Context links</td><td>${{idList(source.context_bound_object_ids||[])}}</td></tr><tr><td>Link scope</td><td>${{Number(source.binding_count||0)===0?'No exact link; snapshot lineage is not inferred.':'Links target specific Research Objects, not every ARA node.'}}</td></tr><tr><td>Source integrity</td><td>${{esc(sourceIntegrity.error_count||0)}} errors / ${{esc(sourceIntegrity.warning_count||0)}} warnings</td></tr>`:'';
			let drill='';if(insight){{const next=insight.next_experiment;drill=`<h3>Claim reasoning</h3><table><tr><td>Depth</td><td>${{esc(insight.depth_level)}}</td></tr><tr><td>Decision</td><td class="${{insight.decision_ready?'pass':'fail'}}">${{insight.decision_ready?'ready':'blocked'}}</td></tr><tr><td>Support</td><td>${{idList(insight.supporting_ids)}}</td></tr><tr><td>Refutation</td><td>${{idList(insight.refuting_ids)}}</td></tr><tr><td>Mechanism</td><td>${{idList(insight.mechanism_ids)}}</td></tr><tr><td>Quality</td><td>${{idList(insight.quality_assessment_ids)}}</td></tr><tr><td>Boundaries</td><td>${{idList(insight.boundary_ids)}}</td></tr><tr><td>Next experiment</td><td>${{next?esc(next.summary||next.candidate_id):'None ranked'}}</td></tr><tr><td>Open gaps</td><td>${{insight.gaps.length?esc(insight.gaps.join(', ')):'None'}}</td></tr></table>`}}
			details.className='';details.innerHTML=`<table><tr><td>Summary</td><td>${{esc(displaySummary(n))}}</td></tr><tr><td>ID</td><td><code>${{esc(n.id)}}</code></td></tr><tr><td>Type</td><td>${{esc(n.kind)}} / ${{esc(n.state)}}</td></tr>${{hypothesisScope?`<tr><td>Theory scope</td><td>${{esc(hypothesisScope)}}</td></tr>`:''}}<tr><td>Layer</td><td>${{esc(n.layer)}}</td></tr><tr><td>Source</td><td>${{esc(sourceName)}}</td></tr>${{sourceAuditRows}}<tr><td>Snapshot</td><td><code>${{esc(n.source_ref||'-')}}</code></td></tr><tr><td>Proof</td><td><strong>${{esc(n.proof.level)}}</strong></td></tr><tr><td>Hash</td><td><code>${{esc(n.content_hash||'-')}}</code></td></tr><tr><td>Actor ID</td><td>${{esc((n.actor||{{}}).actor_id||'-')}}</td></tr><tr><td>Actor authority</td><td>${{esc((n.actor||{{}}).authority||'-')}}</td></tr></table>${{nodeView(detailViews[n.id])}}${{drill}}<h3>Verification checks</h3><ul>${{checks||'<li>No specialized checks.</li>'}}</ul>`;
			wireDetailJumps();
		}}
		function render(){{
			const argumentMode=view.value==='argument';
                argumentView.hidden=!argumentMode;graphMain.hidden=argumentMode;sourceIntegrity.hidden=argumentMode;observabilityBindingSection.hidden=argumentMode;graphControls.forEach(control=>control.hidden=argumentMode);
			if(argumentMode){{stats.innerHTML='<span class="stat">Claim-centric authority separation</span>';renderArgumentView();wireDetailJumps(argumentView);return}}
			const q=search.value.trim().toLowerCase();
			currentProjection=projectDagView(allNodes,sources,view.value,sourceControl.value,q,searchIndex);
			const seedNodes=currentProjection.nodes.filter(n=>(!layer.value||n.layer===layer.value)&&(!kind.value||n.kind===kind.value)&&(!proof.value||n.proof.level===proof.value)&&(!q||isDirectMatch(n,q,searchIndex)));
			const seedIds=new Set(seedNodes.map(n=>n.id));
			const visibleIds=q?oneHopIds(seedIds,allEdges):new Set(seedIds);
			const aggregateSeeds=seedNodes.filter(n=>n.ui_aggregate),queryUniverse=currentProjection.presentation==='ara_decision_spine'?currentProjection.nodes:allNodes,rawNodes=q?queryUniverse.filter(n=>visibleIds.has(n.id)):seedNodes.filter(n=>!n.ui_aggregate),rawIds=new Set(rawNodes.map(n=>n.id)),nodes=[...rawNodes,...aggregateSeeds.filter(n=>!rawIds.has(n.id))],edges=authoritativeEdgesForNodes(nodes,allEdges).filter(e=>!relation||!relation.value||String(e.category||'unknown')===relation.value),contextCount=Math.max(0,nodes.length-seedNodes.length);
			currentDisplayById=new Map(nodes.map(n=>[n.id,n]));
			renderStats(nodes,edges,seedNodes.length,contextCount,Boolean(q));
			const order=new Map((graph.integrity.topological_order||[]).map((id,index)=>[id,index])),buckets=new Map();
			nodes.forEach(n=>{{if(!buckets.has(n.phase))buckets.set(n.phase,[]);buckets.get(n.phase).push(n)}});
			[...buckets.values()].forEach(v=>v.sort((a,b)=>(order.get(a.id)??Number.MAX_SAFE_INTEGER)-(order.get(b.id)??Number.MAX_SAFE_INTEGER)||a.id.localeCompare(b.id)));
			const phases=[...buckets.keys()].sort((a,b)=>a-b),boxW=220,boxH=82,gapX=92,gapY=28,pad=34,maxRows=Math.max(1,...[...buckets.values()].map(v=>v.length)),width=Math.max(720,pad*2+phases.length*boxW+Math.max(0,phases.length-1)*gapX),height=Math.max(220,pad*2+maxRows*boxH+Math.max(0,maxRows-1)*gapY),pos=new Map();
			phases.forEach((p,col)=>buckets.get(p).forEach((n,row)=>pos.set(n.id,{{x:pad+col*(boxW+gapX),y:pad+row*(boxH+gapY)}})));
			const edgeSvg=edges.map(e=>{{const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)return'';let path;if(a.x===b.x){{const x1=a.x+boxW/2,y1=a.y+boxH,x2=b.x+boxW/2,y2=b.y,m=y1+(y2-y1)/2;path=`M${{x1}} ${{y1}} C${{x1}} ${{m}},${{x2}} ${{m}},${{x2}} ${{y2}}`}}else{{const x1=a.x+boxW,y1=a.y+boxH/2,x2=b.x,y2=b.y+boxH/2,m=x1+(x2-x1)/2;path=`M${{x1}} ${{y1}} C${{m}} ${{y1}},${{m}} ${{y2}},${{x2}} ${{y2}}`}}return`<path class="edge ${{esc(e.category)}}" d="${{path}}" marker-end="url(#arrow)"><title>${{esc(e.type||'relation')}}${{e.role?` / ${{esc(e.role)}}`:''}}</title></path>`}}).join('');
			const nodeSvg=nodes.map(n=>{{const p=pos.get(n.id),display=displaySummary(n),summary=display.length>29?display.slice(0,29)+'…':display,compactId=n.id.length>28?n.id.slice(0,12)+'…'+n.id.slice(-12):n.id,isAggregate=Boolean(n.ui_aggregate),aggregate=n.aggregate||{{}},kindText=isAggregate?'ARA source snapshot':n.kind,statusText=isAggregate?`${{aggregate.node_count||0}} nodes · ${{aggregate.exact_link_count||0}} exact links`:`${{n.state}} · ${{n.proof.level}}`,idText=isAggregate?'UI aggregate · not an object':compactId,searchClass=q?(seedIds.has(n.id)?'search-seed':'search-context'):'',sourceClass=(isAggregate&&aggregate.no_exact_links)||sourceNamesWithoutExactLinks.has(sourceNameForNode(n))?'source-unlinked':'',aggregateClass=isAggregate?'ui-aggregate':'',searchRole=q?(seedIds.has(n.id)?'direct search match':'one-hop context'):'visible node',semanticRole=isAggregate?'UI aggregate, not a scientific object':n.kind;return`<g class="node ${{esc(n.proof.level)}} ${{esc(n.state)}} ${{searchClass}} ${{sourceClass}} ${{aggregateClass}} ${{n.id===selectedId?'selected':''}}" data-id="${{esc(n.id)}}" tabindex="0" role="button" aria-label="${{esc(searchRole+': '+semanticRole+' '+display+' '+n.id)}}" transform="translate(${{p.x}},${{p.y}})"><rect width="${{boxW}}" height="${{boxH}}"/><text x="12" y="20"><tspan font-weight="600">${{esc(kindText)}}</tspan></text><text class="muted node-summary" x="12" y="40">${{esc(summary)}}</text><text class="muted" x="12" y="60">${{esc(statusText)}}</text><text class="muted" x="12" y="76">${{esc(idText)}}</text></g>`}}).join('');
			const canvas=document.getElementById('canvas');
			canvas.innerHTML=nodes.length?`<svg width="${{width}}" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>${{edgeSvg}}${{nodeSvg}}</svg>`:'<div class="empty">No nodes match these filters.</div>';
			canvas.querySelectorAll('.node-summary').forEach(el=>fitSvgLabel(el,boxW-24));
			canvas.querySelectorAll('.node').forEach(el=>{{const show=()=>detail(currentDisplayById.get(el.dataset.id));el.addEventListener('click',show);el.addEventListener('keydown',e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();show()}}}})}});
			if(!nodes.length){{selectedId=null;details.className='empty';details.textContent='No scientific object matches the current filters.';return}}
			const selected=seedNodes.find(n=>n.id===selectedId)||seedNodes[0]||nodes.find(n=>n.id===selectedId)||nodes[0];detail(selected)
		}}
	const compactMedia=typeof matchMedia==='function'?matchMedia('(max-width:600px)'):null;
	if(compactMedia){{const disclosures=[...document.querySelectorAll('[data-compact-disclosure]')],syncDisclosures=()=>disclosures.forEach(el=>el.open=!compactMedia.matches);syncDisclosures();compactMedia.addEventListener('change',syncDisclosures)}}
	[view,sourceControl,search,layer,kind,proof,...(relation?[relation]:[])].forEach(el=>el.addEventListener(el===search?'input':'change',render));render();
</script></body></html>"""


# Gate drill-down stays outside the DAG payload so it cannot change graph identity.
_GATE_DETAIL_FIELDS = (
    "gate_id",
    "candidate_id",
    "decision",
    "status",
    "reason",
    "scope",
    "candidate_count",
    "candidate_count_after_gate",
    "candidate_count_after_search",
    "evaluated_candidate_ids",
    "experiment_authorized",
    "claim_promotion_allowed",
    "claim_promotion_authorized",
    "best_paper_level_theorem_survives",
    "authority",
    "ara_created",
    "evidence_ids",
    "hypothesis_id",
    "plan_id",
    "review_id",
    "protocol_kind",
    "reentry_requirements",
    "verdict_vector",
    "preregistered_falsifier",
    "input_bindings",
    "scope_hash",
    "authority_receipt",
    "gate_hash",
    "ara_manifest_hash",
    "ara_exploration_graph_hash",
    "node_id",
    "verification",
    "gate_rules",
    "rules",
    "criteria",
    "pass_condition",
    "failure_rules",
)
_LINEAGE_DETAIL_FIELDS = (
    "summary",
    "result",
    "status",
    "study_phase",
    "boundary_condition",
    "boundary_role",
)
_EVIDENCE_ASSESSMENT_FIELDS = (
    "status",
    "authority",
    "scope",
    "result",
    "limits",
    "boundary",
    "novelty_cleared",
    "claim_promotion_allowed",
    "experiment_authorized",
    "training_authorized",
    "candidate_count",
    "registered_claim_count",
    "best_paper_evidence",
    "external_peer_review",
    "mathematical_machine_proof",
    "portable_copy_boundary",
)
_TERMINAL_RUNTIME_DETAIL_FIELDS = (
    "admission_runtime_claim_hash",
    "admission_bfts_terminal_hash",
    "admission_worker_terminal_hash",
    "terminal_receipt_hash",
)


def _present_detail_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _safe_repository_relative(value: Any) -> str | None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    try:
        return _normalise_relative(value)
    except ResearchGitError:
        return None


def _local_artifact_binding_status(
    repository_root: Path,
    value: Any,
    *,
    snapshot_commit: str,
    expected_hash: str,
    binding_budget: dict[str, Any],
) -> tuple[bool, str]:
    relative = _safe_repository_relative(value)
    if relative is None:
        return False, "invalid_artifact_path"
    root = repository_root.resolve()
    cache_key = ("git", root.as_posix(), snapshot_commit, relative, expected_hash)
    cache = binding_budget["cache"]
    if cache_key in cache:
        cached = cache[cache_key]
        return cached if isinstance(cached, tuple) else (
            bool(cached), "verified_exact_hash" if cached else "artifact_binding_unverified"
        )
    if binding_budget["files"] >= _MAX_ARTIFACT_BINDING_FILES:
        return False, "artifact_binding_file_budget_exceeded"
    binding_budget["files"] += 1
    tree = _run_git(
        root, ["ls-tree", "-z", snapshot_commit, "--", relative], check=False
    )
    if tree.returncode:
        result = (False, "artifact_not_found_at_snapshot")
        cache[cache_key] = result
        return result
    blob_id = None
    for record in tree.stdout.split("\0"):
        if not record or "\t" not in record:
            continue
        metadata, path = record.split("\t", 1)
        fields = metadata.split()
        if len(fields) != 3 or path != relative:
            continue
        mode, object_type, candidate_id = fields
        if (
            mode in {"100644", "100755"}
            and object_type == "blob"
            and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", candidate_id)
        ):
            blob_id = candidate_id
        break
    if blob_id is None:
        result = (False, "artifact_not_found_at_snapshot")
        cache[cache_key] = result
        return result
    size_result = _run_git(root, ["cat-file", "-s", blob_id], check=False)
    try:
        blob_size = int(size_result.stdout.strip())
    except (TypeError, ValueError):
        result = (False, "artifact_size_unreadable")
        cache[cache_key] = result
        return result
    if (
        size_result.returncode
        or blob_size < 0
        or blob_size > _MAX_LINKED_ARTIFACT_BYTES
    ):
        result = (False, "artifact_size_exceeds_limit")
        cache[cache_key] = result
        return result
    remaining = _MAX_ARTIFACT_BINDING_TOTAL_BYTES - binding_budget["bytes"]
    if blob_size > remaining:
        result = (False, "artifact_binding_byte_budget_exceeded")
        cache[cache_key] = result
        return result
    binding_budget["bytes"] += blob_size
    try:
        blob = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", blob_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError:
        result = (False, "artifact_read_failed")
        cache[cache_key] = result
        return result
    if blob.returncode or len(blob.stdout) != blob_size:
        result = (False, "artifact_read_failed")
        cache[cache_key] = result
        return result
    verified = _is_sha256(expected_hash) and _sha256_bytes(blob.stdout) == expected_hash
    result = (verified, "verified_exact_hash" if verified else "content_hash_mismatch")
    cache[cache_key] = result
    return result


def _local_artifact_binding_verified(
    repository_root: Path,
    value: Any,
    *,
    snapshot_commit: str,
    expected_hash: str,
    binding_budget: dict[str, Any],
) -> bool:
    """Backward-compatible boolean wrapper for local artifact verification."""

    return _local_artifact_binding_status(
        repository_root,
        value,
        snapshot_commit=snapshot_commit,
        expected_hash=expected_hash,
        binding_budget=binding_budget,
    )[0]


def _pointer_artifact_binding_status(
    repository_root: Path,
    pointer: Mapping[str, Any],
    *,
    expected_hash: str,
    binding_budget: dict[str, Any],
) -> tuple[bool, str]:
    size = pointer.get("size")
    cache_key = (
        "pointer",
        str(pointer.get("pointer_path") or ""),
        str(pointer.get("logical_path") or ""),
        str(pointer.get("object_hash") or pointer.get("content_hash") or ""),
        expected_hash,
        size if isinstance(size, int) else repr(size),
    )
    cache = binding_budget["cache"]
    if cache_key in cache:
        cached = cache[cache_key]
        return cached if isinstance(cached, tuple) else (
            bool(cached), "verified_exact_hash" if cached else "artifact_binding_unverified"
        )
    if binding_budget["files"] >= _MAX_ARTIFACT_BINDING_FILES:
        return False, "artifact_binding_file_budget_exceeded"
    binding_budget["files"] += 1
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or size > _MAX_LINKED_ARTIFACT_BYTES
    ):
        result = (False, "pointer_size_invalid")
        cache[cache_key] = result
        return result
    remaining = _MAX_ARTIFACT_BINDING_TOTAL_BYTES - binding_budget["bytes"]
    if size > remaining:
        result = (False, "artifact_binding_byte_budget_exceeded")
        cache[cache_key] = result
        return result
    binding_budget["bytes"] += size
    try:
        store_path = _pointer_store_path(repository_root.resolve(), dict(pointer))
        payload = read_bounded_regular_file(
            store_path,
            maximum=size,
            label="research_dag_artifact",
        )
    except (BoundedFileError, OSError, ResearchGitError):
        result = (False, "pointer_read_failed")
        cache[cache_key] = result
        return result
    if len(payload) != size:
        result = (False, "pointer_size_mismatch")
        cache[cache_key] = result
        return result
    verified = _is_sha256(expected_hash) and _sha256_bytes(payload) == expected_hash
    result = (verified, "verified_exact_hash" if verified else "content_hash_mismatch")
    cache[cache_key] = result
    return result


def _pointer_artifact_binding_verified(
    repository_root: Path,
    pointer: Mapping[str, Any],
    *,
    expected_hash: str,
    binding_budget: dict[str, Any],
) -> bool:
    """Backward-compatible boolean wrapper for pointer verification."""

    return _pointer_artifact_binding_status(
        repository_root,
        pointer,
        expected_hash=expected_hash,
        binding_budget=binding_budget,
    )[0]


def _artifact_detail_rows(
    item: Mapping[str, Any],
    *,
    repository_root: Path,
    html_directory: Path,
    snapshot_commit: str,
    pointer_records: Mapping[str, Sequence[Mapping[str, Any]]],
    binding_budget: dict[str, Any],
) -> list[dict[str, Any]]:
    payload = item.get("payload") or {}
    if not isinstance(payload, Mapping):
        return []
    hashes = payload.get("result_artifact_hashes") or {}
    if not isinstance(hashes, Mapping):
        hashes = {}
    candidates: list[tuple[str, Any]] = []
    for field in ("result_artifacts", "artifacts", "artifact_links"):
        raw = payload.get(field)
        if isinstance(raw, Mapping):
            candidates.extend((str(label), value) for label, value in raw.items())
        elif isinstance(raw, list):
            candidates.extend(
                (f"{field}_{index + 1}", value) for index, value in enumerate(raw)
            )
    known_labels = {label for label, _value in candidates}
    candidates.extend(
        (str(label), {"content_hash": digest})
        for label, digest in hashes.items()
        if str(label) not in known_labels
    )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for fallback_label, raw in candidates[:_MAX_ARTIFACT_BINDING_FILES]:
        if isinstance(raw, Mapping):
            record = raw
        elif isinstance(raw, (str, Path)):
            record = {"path": raw}
        else:
            continue
        label = str(
            record.get("logical_name")
            or record.get("label")
            or record.get("name")
            or fallback_label
        )
        logical_path = _safe_repository_relative(record.get("logical_path"))
        pointer_path = _safe_repository_relative(record.get("pointer_path"))
        direct_path = _safe_repository_relative(
            record.get("path") or record.get("repository_path")
        )
        declared_paths = (
            record.get("logical_path"),
            record.get("pointer_path"),
            record.get("path") or record.get("repository_path"),
        )
        invalid_declared_path = any(
            value not in (None, "")
            and _safe_repository_relative(value) is None
            for value in declared_paths
        )
        declarations: list[Any] = []
        if "content_hash" in record:
            declarations.append(record.get("content_hash"))
        for hash_key in dict.fromkeys((fallback_label, label)):
            if hash_key in hashes:
                declarations.append(hashes[hash_key])
        valid_digests = [str(value) for value in declarations if _is_sha256(value)]
        hashes_agree = (
            bool(valid_digests)
            and len(valid_digests) == len(declarations)
            and len(set(valid_digests)) == 1
        )
        digest = valid_digests[0] if valid_digests else None
        binding_verified = False
        binding_reason = "missing_content_hash"
        binding_checked_path = None
        if declarations and not valid_digests:
            binding_reason = "invalid_content_hash"
        elif digest and not hashes_agree:
            binding_reason = "conflicting_content_hash_declarations"
        elif invalid_declared_path:
            binding_reason = "invalid_artifact_path"
        elif digest and hashes_agree and pointer_path:
            pointer = next(
                (
                    candidate
                    for candidate in pointer_records.get(digest, ())
                    if str(candidate.get("pointer_path") or "") == pointer_path
                    and (
                        not logical_path
                        or str(candidate.get("logical_path") or "") == logical_path
                    )
                ),
                None,
            )
            if pointer is None:
                binding_reason = "pointer_record_missing"
            else:
                binding_checked_path = pointer_path
                binding_verified, binding_reason = _pointer_artifact_binding_status(
                    repository_root,
                    pointer,
                    expected_hash=digest,
                    binding_budget=binding_budget,
                )
        elif digest and hashes_agree:
            paths = tuple(
                dict.fromkeys(path for path in (direct_path, logical_path) if path)
            )
            if not paths:
                binding_reason = "missing_artifact_location"
            else:
                for path in paths:
                    verified, reason = _local_artifact_binding_status(
                        repository_root,
                        path,
                        snapshot_commit=snapshot_commit,
                        expected_hash=digest,
                        binding_budget=binding_budget,
                    )
                    binding_checked_path = path
                    binding_reason = reason
                    if verified:
                        binding_verified = True
                        binding_reason = reason
                        break
        row = {
            "label": label,
            "logical_path": logical_path or direct_path,
            "pointer_path": pointer_path,
            "content_hash": digest,
            "binding_verified": binding_verified,
            "binding_reason": binding_reason,
            "binding_checked_path": binding_checked_path,
            "href": None,
            "source_id": str(item.get("object_id") or ""),
            "source_kind": str(item.get("kind") or "research_object"),
        }
        identity = tuple(row.values())
        if identity not in seen and any(
            row[key]
            for key in (
                "logical_path",
                "pointer_path",
                "content_hash",
                "binding_verified",
            )
        ):
            rows.append(row)
            seen.add(identity)
    return rows


_GATE_REFERENCE_FIELDS = (
    "evidence_id",
    "attempt_id",
    "experiment_attempt_id",
    "evidence_ids",
    "attempt_ids",
    "experiment_attempt_ids",
)


def _gate_payload_reference_targets(payload: Mapping[str, Any]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for field in _GATE_REFERENCE_FIELDS:
        raw = payload.get(field)
        if isinstance(raw, str):
            values = (raw,)
        elif isinstance(raw, (list, tuple)):
            values = raw
        else:
            values = ()
        for value in values:
            if not isinstance(value, str) or not value or value in seen:
                continue
            seen.add(value)
            targets.append(value)
    return targets


def _gate_payload_size_bytes(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _gate_detail_views(
    objects: Sequence[Mapping[str, Any]],
    *,
    repository_root: Path,
    html_directory: Path,
    snapshot_commit: str,
    disclose: bool,
) -> dict[str, Any]:
    if not disclose:
        return {}
    has_pointer_paths = False
    for item in objects:
        payload = item.get("payload") or {}
        if not isinstance(payload, Mapping):
            continue
        for field in ("result_artifacts", "artifacts", "artifact_links"):
            raw = payload.get(field)
            records = raw.values() if isinstance(raw, Mapping) else raw
            if isinstance(records, (list, tuple)) or isinstance(raw, Mapping):
                if any(
                    isinstance(record, Mapping) and record.get("pointer_path")
                    for record in records
                ):
                    has_pointer_paths = True
                    break
        if has_pointer_paths:
            break
    pointer_records = (
        _strict_pointer_records_at_commit(repository_root.resolve(), snapshot_commit)
        if has_pointer_paths
        else {}
    )
    binding_budget: dict[str, Any] = {"bytes": 0, "files": 0, "cache": {}}
    by_id = {str(item.get("object_id") or ""): item for item in objects}
    details: dict[str, Any] = {}
    for item in objects:
        object_id = str(item.get("object_id") or "")
        payload = item.get("payload") or {}
        if not object_id or not isinstance(payload, Mapping):
            continue
        aliases = {
            key: payload[key]
            for key in _SEARCHABLE_ALIAS_FIELDS
            if key in payload and _present_detail_value(payload[key])
        }
        # Payload assessments are advisory, separate from the object's lifecycle state.
        assessment = (
            {
                key: payload[key]
                for key in _EVIDENCE_ASSESSMENT_FIELDS
                if key in payload and _present_detail_value(payload[key])
            }
            if item.get("kind") == "evidence_synthesis"
            else {}
        )
        runtime_bindings = (
            {
                key: payload[key]
                for key in _TERMINAL_RUNTIME_DETAIL_FIELDS
                if key in payload and _present_detail_value(payload[key])
            }
            if item.get("kind") == "experiment_attempt"
            else {}
        )
        direct_artifacts = _artifact_detail_rows(
            item,
            repository_root=repository_root,
            html_directory=html_directory,
            snapshot_commit=snapshot_commit,
            pointer_records=pointer_records,
            binding_budget=binding_budget,
        )
        if aliases or assessment or runtime_bindings or direct_artifacts:
            details[object_id] = {
                "fields": {**aliases, **assessment, **runtime_bindings},
                "artifacts": direct_artifacts,
            }
            if assessment:
                details[object_id]["title"] = "Recorded evidence assessment (advisory)"
    for gate in objects:
        if gate.get("kind") != "gate_decision":
            continue
        gate_id = str(gate.get("object_id") or "")
        payload = gate.get("payload") or {}
        if not gate_id or not isinstance(payload, Mapping):
            continue
        gate_fields = {
            key: payload[key]
            for key in _GATE_DETAIL_FIELDS
            if key in payload and _present_detail_value(payload[key])
        }
        queue: list[tuple[str, int]] = [(gate_id, 0)]
        visited: set[str] = set()
        lineage: list[Mapping[str, Any]] = []
        lineage_edge_count = 0
        lineage_bytes = 0
        lineage_max_depth = 0
        truncation_reasons: set[str] = set()
        while queue:
            object_id, depth = queue.pop(0)
            if object_id in visited or object_id not in by_id:
                continue
            if len(lineage) >= _MAX_GATE_LINEAGE_NODES:
                truncation_reasons.add("node_budget")
                break
            if depth > _MAX_GATE_LINEAGE_DEPTH:
                truncation_reasons.add("depth_budget")
                continue
            visited.add(object_id)
            item = by_id[object_id]
            item_payload = item.get("payload") or {}
            if isinstance(item_payload, Mapping):
                try:
                    item_size = _gate_payload_size_bytes(item_payload)
                except (TypeError, ValueError):
                    truncation_reasons.add("size_unavailable")
                    break
                if lineage_bytes + item_size > _MAX_GATE_LINEAGE_DETAIL_BYTES:
                    truncation_reasons.add("byte_budget")
                    break
                lineage_bytes += item_size
            lineage.append(item)
            lineage_max_depth = max(lineage_max_depth, depth)
            relation_targets = [
                str(relation.get("target") or "")
                for relation in item.get("relations") or []
                if isinstance(relation, Mapping)
                and relation.get("type") in _GATE_UPSTREAM_RELATIONS
            ]
            if isinstance(item_payload, Mapping):
                relation_targets.extend(_gate_payload_reference_targets(item_payload))
            for target in relation_targets:
                if not target or target in visited:
                    continue
                lineage_edge_count += 1
                if lineage_edge_count > _MAX_GATE_LINEAGE_EDGES:
                    truncation_reasons.add("edge_budget")
                    queue.clear()
                    break
                if depth >= _MAX_GATE_LINEAGE_DEPTH:
                    truncation_reasons.add("depth_budget")
                    continue
                queue.append((target, depth + 1))

        lineage_details: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        for item in lineage:
            item_payload = item.get("payload") or {}
            if not isinstance(item_payload, Mapping):
                continue
            source_id = str(item.get("object_id") or "")
            source_kind = str(item.get("kind") or "research_object")
            source_fields = {
                key: item_payload[key]
                for key in _LINEAGE_DETAIL_FIELDS
                if key in item_payload and _present_detail_value(item_payload[key])
            }
            if source_id != gate_id and source_fields:
                lineage_details.append(
                    {
                        "source_id": source_id,
                        "source_kind": source_kind,
                        "fields": source_fields,
                    }
                )
            for field in ("metrics", "metric"):
                value = item_payload.get(field)
                if _present_detail_value(value):
                    metrics.append(
                        {
                            "source_id": source_id,
                            "source_kind": source_kind,
                            "values": value,
                        }
                    )
            artifacts.extend(
                _artifact_detail_rows(
                    item,
                    repository_root=repository_root,
                    html_directory=html_directory,
                    snapshot_commit=snapshot_commit,
                    pointer_records=pointer_records,
                    binding_budget=binding_budget,
                )
            )
        details[gate_id] = {
            **details.get(gate_id, {}),
            "gate": gate_fields,
            "lineage": lineage_details,
            "metrics": metrics,
            "artifacts": artifacts,
            "lineage_meta": {
                "node_count": len(lineage),
                "edge_count": lineage_edge_count,
                "max_depth": lineage_max_depth,
                "detail_bytes": lineage_bytes,
                "truncated": bool(truncation_reasons),
                "truncation_reasons": sorted(truncation_reasons),
            },
        }
    return details


def _open_dag_destination(destination: str | Path) -> tuple[Path, int]:
    """Create/open a destination without following user-controlled symlinks."""

    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise ResearchGitError("secure descriptor-relative DAG export is unavailable")
    requested = Path(os.path.abspath(os.fspath(Path(destination).expanduser())))
    parts = requested.parts
    if not parts or not requested.is_absolute():
        raise ResearchGitError(f"DAG destination is invalid: {destination}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(parts[0], flags)
        opened = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ResearchGitError(
            f"cannot securely open DAG destination anchor: {parts[0]}"
        ) from exc
    assert descriptor is not None
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise ResearchGitError(f"DAG destination anchor is not a directory: {parts[0]}")
    traversed = Path(parts[0])
    try:
        for component in parts[1:]:
            traversed /= component
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise ResearchGitError(
                        f"cannot create DAG destination: {traversed}"
                    ) from exc
                try:
                    before = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise ResearchGitError(
                        f"cannot inspect DAG destination: {traversed}"
                    ) from exc
            except OSError as exc:
                raise ResearchGitError(
                    f"cannot inspect DAG destination: {traversed}"
                ) from exc
            if (
                stat.S_ISLNK(before.st_mode)
                and traversed.parent == Path("/")
                and component in {"tmp", "var"}
                and before.st_uid == 0
            ):
                try:
                    resolved_alias = traversed.resolve(strict=True)
                    alias_state = resolved_alias.lstat()
                    alias_descriptor = os.open(resolved_alias, flags)
                    alias_opened = os.fstat(alias_descriptor)
                except OSError as exc:
                    if "alias_descriptor" in locals():
                        os.close(alias_descriptor)
                        del alias_descriptor
                    raise ResearchGitError(
                        f"cannot securely open system directory alias: {traversed}"
                    ) from exc
                if not stat.S_ISDIR(alias_state.st_mode) or _stat_open_identity(
                    alias_state
                ) != _stat_open_identity(alias_opened):
                    os.close(alias_descriptor)
                    del alias_descriptor
                    raise ResearchGitError(
                        f"system directory alias changed while opening: {traversed}"
                    )
                os.close(descriptor)
                descriptor = alias_descriptor
                del alias_descriptor
                traversed = resolved_alias
                continue
            if not stat.S_ISDIR(before.st_mode):
                raise ResearchGitError(
                    f"DAG destination component is not a real directory: {traversed}"
                )
            try:
                child_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ResearchGitError(
                    f"cannot securely open DAG destination: {traversed}"
                ) from exc
            try:
                after = os.fstat(child_descriptor)
            except OSError as exc:
                os.close(child_descriptor)
                raise ResearchGitError(
                    f"cannot inspect DAG destination: {traversed}"
                ) from exc
            if _stat_open_identity(before) != _stat_open_identity(after):
                os.close(child_descriptor)
                raise ResearchGitError(
                    f"DAG destination changed while opening: {traversed}"
                )
            os.close(descriptor)
            descriptor = child_descriptor
        return traversed, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_dag_destination(output: Path, output_descriptor: int) -> None:
    try:
        path_state = output.lstat()
        opened_state = os.fstat(output_descriptor)
    except OSError as exc:
        raise ResearchGitError(f"DAG destination became unavailable: {output}") from exc
    if not stat.S_ISDIR(path_state.st_mode) or _stat_open_identity(
        path_state
    ) != _stat_open_identity(opened_state):
        raise ResearchGitError(f"DAG destination changed during export: {output}")


def _atomic_write_dag_output(
    output_descriptor: int,
    filename: str,
    content: bytes,
) -> None:
    if not filename or "/" in filename or filename in {".", ".."}:
        raise ResearchGitError(f"invalid DAG output filename: {filename}")
    temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    temporary_exists = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o666,
            dir_fd=output_descriptor,
        )
        temporary_exists = True
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short DAG output write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.rename(
            temporary,
            filename,
            src_dir_fd=output_descriptor,
            dst_dir_fd=output_descriptor,
        )
        temporary_exists = False
        try:
            os.fsync(output_descriptor)
        except OSError:
            pass
    except OSError as exc:
        raise ResearchGitError(
            f"cannot atomically write DAG output: {filename}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=output_descriptor)
            except OSError:
                pass


def _clear_artifact_blob_namespace(output: Path, output_descriptor: int) -> None:
    """Remove legacy artifact copies without traversing a substituted symlink."""

    if not shutil.rmtree.avoids_symlink_attacks:
        raise ResearchGitError("race-resistant artifact cleanup is unavailable")
    try:
        before = os.stat(
            "artifact-blobs",
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(before.st_mode):
        raise ResearchGitError(
            f"reserved artifact namespace is not a directory: {output / 'artifact-blobs'}"
        )
    try:
        shutil.rmtree("artifact-blobs", dir_fd=output_descriptor)
    except OSError as exc:
        raise ResearchGitError(
            f"cannot securely clear artifact namespace: {output / 'artifact-blobs'}"
        ) from exc
    try:
        os.stat(
            "artifact-blobs",
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise ResearchGitError(
        f"artifact namespace reappeared during export: {output / 'artifact-blobs'}"
    )


def export_research_dag(
    repo: str | Path,
    destination: str | Path,
    *,
    ref: str = "HEAD",
    ara_roots: Sequence[str | Path] = (),
    disclose_summaries: bool = True,
    advisory_index: str | Path | None = None,
) -> dict[str, Any]:
    output, output_descriptor = _open_dag_destination(destination)
    try:
        _clear_artifact_blob_namespace(output, output_descriptor)
        object_rows: list[Mapping[str, Any]] = []
        ara_detail_views: dict[str, Any] = {}
        closure_rows: list[Mapping[str, Any]] = []
        graph = build_research_dag(
            repo,
            ref=ref,
            ara_roots=ara_roots,
            disclose_summaries=disclose_summaries,
            _object_rows_out=object_rows,
            _ara_detail_views_out=ara_detail_views,
            _closure_out=closure_rows,
        )
        json_path = output / "research-dag.json"
        html_path = output / "research-dag.html"
        argument_path = output / "research-argument.json"
        argument_view = build_research_argument_view(
            graph,
            object_rows=object_rows,
            closure=closure_rows[0] if closure_rows else None,
            repo=repo,
            advisory_index=advisory_index,
            disclose_summaries=disclose_summaries,
        )
        detail_views = {
            **ara_detail_views,
            **_gate_detail_views(
                object_rows,
                repository_root=_repository_root(repo),
                html_directory=html_path.parent,
                snapshot_commit=str(graph["commit"]),
                disclose=disclose_summaries,
            ),
        }
        _atomic_write_dag_output(
            output_descriptor,
            json_path.name,
            json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            ),
        )
        _atomic_write_dag_output(
            output_descriptor,
            argument_path.name,
            json.dumps(
                argument_view, indent=2, ensure_ascii=False, allow_nan=False
            ).encode("utf-8"),
        )
        _atomic_write_dag_output(
            output_descriptor,
            html_path.name,
            render_research_dag_html(
                graph,
                node_details=detail_views,
                argument_view=argument_view,
                disclose_summaries=disclose_summaries,
            ).encode("utf-8"),
        )
        _verify_dag_destination(output, output_descriptor)
    finally:
        os.close(output_descriptor)
    advisory_routes = list(argument_view.get("advisory_routes") or [])
    advisory_blocked_count = sum(
        1
        for route in advisory_routes
        if isinstance(route, Mapping) and route.get("route_state") == "BLOCKED"
    )
    advisory_status = (
        "BLOCKED"
        if advisory_blocked_count
        else (
            "ADVISORY"
            if argument_view.get("advisory_index_binding") is not None
            else "NOT_SUPPLIED"
        )
    )
    return {
        "graph": graph,
        "json": json_path.as_posix(),
        "html": html_path.as_posix(),
        "argument": argument_path.as_posix(),
        "advisory_status": advisory_status,
        "advisory_blocked_count": advisory_blocked_count,
        "advisory_route_count": len(advisory_routes),
    }


__all__ = [
    "RESEARCH_DAG_SCHEMA",
    "build_research_dag",
    "export_research_dag",
    "render_research_dag_html",
]
