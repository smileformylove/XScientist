"""Reproducible, provider-free first-run usability benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ai_scientist.utils.atomic_io import atomic_write_json

from ._version import __version__
from .demo import create_autopilot_demo
from .workspace_status import build_workspace_status


def benchmark_first_run(
    workspace: str | Path | None = None,
    *,
    profile: str = "balanced",
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Measure the deterministic offline journey from empty dir to status."""

    normalized_profile = str(profile or "balanced").strip().lower()
    if normalized_profile not in {"balanced", "discovery", "publication"}:
        raise ValueError("profile must be one of: balanced, discovery, publication")
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not math.isfinite(float(max_seconds))
        or max_seconds <= 0
    ):
        raise ValueError("max_seconds must be greater than zero")
    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="xscientist-first-run-")
        root = Path(temporary.name) / "study"
    else:
        root = Path(workspace).expanduser().resolve()
    started = time.perf_counter()
    try:
        demo = create_autopilot_demo(root, profile=normalized_profile, language="en")
        status = build_workspace_status(root, language="en")
        try:
            from .evidence_index import build_evidence_index

            evidence_index = build_evidence_index(root)
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            RecursionError,
            MemoryError,
        ):
            evidence_index = {
                "schema": "xscientist.evidence-index.v1",
                "available": False,
                "mode": "unavailable",
                "hash_algorithm": "sha256",
                "workspace_root_disclosed": False,
                "paths_disclosed": False,
                "raw_content_included": False,
                "workspace_mutated": False,
                "limits": {
                    "max_files_per_category": 512,
                    "max_bytes": 32 * 1024 * 1024,
                },
                "categories": {},
                "ara_contract": {
                    "manifest_count": 0,
                    "lock_count": 0,
                    "graph_count": 0,
                    "verify_report_count": 0,
                    "lock_state": "not_observed",
                    "control_digest": None,
                    "digest_scope": "observed_control_files",
                    "bytes_read": 0,
                    "fsck_run": False,
                    "bundle_created": False,
                    "truncated": False,
                    "raw_payloads_included": False,
                },
                "truncated": False,
                "read_error_count": 1,
            }
        duration = time.perf_counter() - started
        threshold_passed = max_seconds is None or duration <= max_seconds
        return {
            "schema": "xscientist.first-run-benchmark.v1",
            "ok": bool(status["ok"] and threshold_passed),
            "version": __version__,
            "runtime": {
                "python": platform.python_version(),
                "system": platform.system().lower(),
            },
            "profile": normalized_profile,
            "duration_seconds": round(duration, 3),
            "max_seconds": max_seconds,
            "threshold_passed": threshold_passed,
            "network_used": False,
            "provider_used": False,
            "model_cost_usd": 0.0,
            "research": {
                "dag_nodes": demo["dag"]["nodes"],
                "dag_relations": demo["dag"]["relations"],
                "closure": demo["dag"]["closure"],
                "run_started": status["run"]["started"],
                # Keep the status contribution to ``ok`` explicit so an
                # offline verifier can distinguish a threshold failure from
                # a workspace-status failure without re-running the demo.
                "status_ok": bool(status["ok"]),
                "budget_available": status["budget"]["available"],
                "next_step": (status["next_steps"] or [{}])[0].get("code"),
            },
            "workspace_retained": workspace is not None,
            "host_paths_disclosed": False,
            "evidence_index": evidence_index,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


# This is intentionally a small, local conformance harness rather than a
# reimplementation of AutoResearchEval's agent rollouts.  The official suite
# evaluates 800 trajectories with an artifact-aware judge; this command only
# checks whether a local XScientist workspace leaves inspectable artifacts at
# the six lifecycle stages.
_AUTORESEARCH_STAGES = {
    "ideation_planning": {
        "kinds": ("question", "research_goal", "hypothesis", "research_plan"),
        "paths": ("question.md", "research.yaml"),
    },
    "retrieval_synthesis": {
        "kinds": (
            "search_plan",
            "search_receipt",
            "source_snapshot",
            "evidence_synthesis",
        ),
        "paths": ("01_literature", "literature", "sources"),
    },
    "execution_implementation": {
        "kinds": ("experiment_attempt", "protocol", "dataset"),
        "paths": ("02_experiments", "experiments", "03_data"),
    },
    "analysis_interpretation": {
        "kinds": ("evidence", "inference", "warrant", "claim"),
        "paths": ("04_logs/insight_report.json", "claims"),
    },
    "writing_documentation": {
        "kinds": ("manuscript", "report", "paper"),
        "paths": ("manuscript", "05_outputs", "outputs"),
    },
    "self_verification_review": {
        "kinds": (
            "review",
            "challenge",
            "gate_decision",
            "reproduction",
            "research_review",
        ),
        "paths": ("review", "reviews", "06_reviews"),
    },
}

_STAGE_LABELS = {
    "ideation_planning": "A · Ideation & Planning",
    "retrieval_synthesis": "B · Retrieval & Synthesis",
    "execution_implementation": "C · Execution & Implementation",
    "analysis_interpretation": "D · Analysis & Interpretation",
    "writing_documentation": "E · Writing & Documentation",
    "self_verification_review": "F · Self-Verification & Review",
}

_ISSUE_KEYS = (
    "required_failures",
    "blocking_issues",
    "active_issues",
    "unresolved_issues",
    "findings",
    "issues",
)
_REPAIR_RELATIONS = {
    "addresses",
    "corrects",
    "repaired_by",
    "repairs",
    "resolves",
    "retests",
}
_REPAIR_KINDS = {
    "experiment_attempt",
    "experiment_design",
    "gate_decision",
    "reproduction",
    "research_plan",
    "review",
    "research_review",
}
_COMPLETED_STATES = {"completed", "verified", "accepted", "promoted"}
_CONTAINING_GATE_DECISIONS = {
    "hold",
    "held",
    "reject",
    "rejected",
    "block",
    "blocked",
    "deny",
    "denied",
    "stop",
    "stopped",
    "pause",
    "paused",
}
_MAX_TASK_FILE_BYTES = 64 * 1024 * 1024
_MAX_REPORT_FILE_BYTES = 32 * 1024 * 1024
_MAX_WORKSPACE_AUDIT_OBJECTS = 512
_MAX_EXPLORATION_GRAPH_FILES = 32
_MAX_EXPLORATION_GRAPH_BYTES = 2 * 1024 * 1024
_MAX_EXPLORATION_NODES = 4096
_CANONICAL_DOMAINS = {
    "biology",
    "chemistry",
    "geophysics",
    "material_science",
    "medicine",
    "physics",
    "scientific_computing",
}
_MAX_LEGACY_CONTRACT_BYTES = 8 * 1024 * 1024
_PUBLIC_OBJECT_KINDS = frozenset(
    {kind for spec in _AUTORESEARCH_STAGES.values() for kind in spec["kinds"]}
    | {
        "agent_evaluation",
        "effect_estimate",
        "experiment_design",
        "passage_evidence",
        "source_update",
        "tool_evidence",
        "warrant",
    }
)
_EXPLORATION_STOP_REASON_ORDER = (
    "budget_exhausted",
    "tool_error",
    "model_selection",
    "missing_data",
)
_EXPLORATION_FAILED_STATES = frozenset(
    {"failed", "error", "crashed", "cancelled", "timed_out", "rejected", "blocked"}
)
_EXPLORATION_COMPLETED_STATES = frozenset(
    {"completed", "success", "succeeded", "verified", "accepted", "promoted"}
)
_EXPLORATION_PLANNED_STATES = frozenset(
    {"planned", "draft", "queued", "pending", "candidate", "unattempted"}
)
_EXPLORATION_MAX_WALK_ENTRIES = 8192


def _diagnostic_item(
    diagnostic_id: str,
    *,
    priority: str,
    area: str,
    status: str,
    evidence: str,
    recommendation: str,
    verification: str,
) -> dict[str, Any]:
    """Build a fixed-vocabulary, actionable benchmark finding.

    The benchmark is intentionally conservative: these are observations about
    missing measurement/evidence channels, not claims that a scientific result
    is wrong.  Keeping the text here fixed (rather than copying workspace
    payloads) also preserves the report's redaction boundary.
    """

    return {
        "id": diagnostic_id,
        "priority": priority,
        "area": area,
        "status": status,
        "evidence": evidence,
        "recommendation": recommendation,
        "verification": verification,
    }


def _build_benchmark_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    """Turn conformance observations into a small optimization backlog.

    This is deliberately not a weighted score.  It answers the practical
    follow-up question after a pilot run: what must be fixed before a fair
    quality comparison, and which observability improvements are next?
    """

    items: list[dict[str, Any]] = []
    tasks = report.get("tasks") or {}
    execution = report.get("execution") or {}
    human = report.get("human_baseline") or {}
    workspace = report.get("workspace") or {}

    if int(execution.get("rollouts_evaluated") or 0) == 0:
        items.append(
            _diagnostic_item(
                "QUALITY.NO_MATCHED_ROLLOUT",
                priority="P0",
                area="scientific_quality",
                status="measurement_blocked",
                evidence="rollouts_evaluated=0",
                recommendation=(
                    "Register a task manifest, evaluator, budget, and seed policy; "
                    "then run repeated local rollouts before making a quality claim."
                ),
                verification="official_comparable=true only after all fairness checks pass",
            )
        )
    if int(tasks.get("valid_task_contracts") or 0) < int(tasks.get("count") or 0):
        items.append(
            _diagnostic_item(
                "QUALITY.INVALID_TASK_CONTRACTS",
                priority="P1",
                area="scientific_quality",
                status="observed_gap",
                evidence="invalid_task_contracts>0",
                recommendation="Repair task framing and rerun the exact manifest hash.",
                verification="valid_task_contracts == count",
            )
        )
    if human.get("status") == "not_reported":
        items.append(
            _diagnostic_item(
                "COMPARISON.NO_MATCHED_HUMAN_ARM",
                priority="P1",
                area="fairness",
                status="not_reported",
                evidence="human_baseline.status=not_reported",
                recommendation=(
                    "Keep human status null; if a comparison is needed, preregister "
                    "a matched human arm instead of importing external scores."
                ),
                verification="participants, protocol, budget, and evaluator are archived",
            )
        )

    retention = report.get("evidence_retention") or {}
    if retention.get("mode") == "read_only_bounded_index":
        items.append(
            _diagnostic_item(
                "AUDIT.REPORT_INDEX_ONLY",
                priority="P1",
                area="auditability",
                status="intentional_boundary",
                evidence="evidence_retention.mode=read_only_bounded_index",
                recommendation=(
                    "Persist the redacted report with --output and retain a separately "
                    "access-controlled ARA/VCS bundle when a full audit is required."
                ),
                verification="report_persistence.requested=true and export manifest is archived",
            )
        )

    if workspace:
        stages = workspace.get("stages") or {}
        incomplete = [
            stage
            for stage, row in stages.items()
            if not bool((row or {}).get("complete"))
        ]
        if incomplete:
            items.append(
                _diagnostic_item(
                    "QUALITY.INCOMPLETE_LIFECYCLE",
                    priority="P1",
                    area="scientific_quality",
                    status="observed_gap",
                    evidence="stage_complete=false",
                    recommendation=(
                        "Add the missing typed evidence and independent checks; "
                        "do not convert stage coverage into a quality score."
                    ),
                    verification="all six stages report complete=true",
                )
            )

        process = workspace.get("process") or {}
        if not process.get("available", False):
            items.append(
                _diagnostic_item(
                    "AUDIT.NO_RESEARCH_VCS",
                    priority="P1",
                    area="auditability",
                    status="observed_gap",
                    evidence="process.available=false",
                    recommendation=(
                        "Initialize or export a Research VCS history so intermediate "
                        "decisions and branch boundaries can be inspected."
                    ),
                    verification="process.available=true and schema validates",
                )
            )
        else:
            topology = process.get("branch_topology") or {}
            fairness = topology.get("fair_branch_comparison") or {}
            if int(topology.get("source_branch_count") or 0) > 1 and not fairness.get(
                "eligible", False
            ):
                items.append(
                    _diagnostic_item(
                        "FAIRNESS.BRANCH_CONTRACT_UNVERIFIED",
                        priority="P0",
                        area="fairness",
                        status="blocked",
                        evidence="multiple_branches_without_verified_contract",
                        recommendation=(
                            "Record the same task slice, budget, evaluator, and fork "
                            "base for every branch before comparing outcomes."
                        ),
                        verification="fair_branch_comparison.eligible=true",
                    )
                )
            elif int(topology.get("source_branch_count") or 0) <= 1:
                items.append(
                    _diagnostic_item(
                        "EXPLORATION.NO_BRANCH_DIVERSITY",
                        priority="P2",
                        area="exploration",
                        status="not_observed",
                        evidence="source_branch_count<=1",
                        recommendation=(
                            "For exploration studies, preserve at least one explicit "
                            "alternative branch and its rejection/merge decision."
                        ),
                        verification="branch topology shows a registered alternative",
                    )
                )
            limits = process.get("limits") or {}
            if any((limits.get("truncated") or {}).values()):
                items.append(
                    _diagnostic_item(
                        "AUDIT.BOUNDED_VIEW",
                        priority="P1",
                        area="auditability",
                        status="bounded",
                        evidence="process.limits.truncated=true",
                        recommendation=(
                            "Retain the full Research VCS/ARA export separately; use the "
                            "redacted index only for sharing."
                        ),
                        verification="source totals and an explicit audit bundle are retained",
                    )
                )

        exploration = workspace.get("exploration") or {}
        if exploration.get("status") in {
            "unavailable",
            "unreadable",
            "observed_empty",
            "partially_observed",
        }:
            items.append(
                _diagnostic_item(
                    "EXPLORATION.SEARCH_TRACE_UNAVAILABLE",
                    priority="P1",
                    area="exploration",
                    status=(
                        "not_observed"
                        if exploration.get("status") == "unavailable"
                        else "observed_gap"
                    ),
                    evidence=(
                        "workspace.exploration.status=" + str(exploration.get("status"))
                    ),
                    recommendation=(
                        "Keep candidate, discard, and stop decisions in a bounded ARA "
                        "exploration graph; absence is not a zero-failure result."
                    ),
                    verification="workspace.exploration.status=observed",
                )
            )

        metacognition = workspace.get("metacognition") or {}
        meta_status = str(metacognition.get("status") or "")
        if meta_status in {"open", "detected"}:
            items.append(
                _diagnostic_item(
                    "FEEDBACK.OPEN_REVIEW_DEBT",
                    priority="P0",
                    area="feedback_evolution",
                    status="blocked",
                    evidence=f"metacognition.status={meta_status}",
                    recommendation=(
                        "Create a linked corrective/retest artifact and keep the release "
                        "gate closed until it verifies."
                    ),
                    verification="status becomes repaired with an explicit repair relation",
                )
            )
        elif meta_status == "contained":
            items.append(
                _diagnostic_item(
                    "FEEDBACK.CONTAINED_REVIEW_DEBT",
                    priority="P1",
                    area="feedback_evolution",
                    status="contained_unrepaired",
                    evidence="metacognition.status=contained",
                    recommendation=(
                        "Treat the hold/reject gate as a release block, not as a repaired "
                        "result; schedule the corrective experiment."
                    ),
                    verification="unresolved_issue_count=0 or status=repaired",
                )
            )

        arft = workspace.get("arft_coverage") or {}
        arft_summary = arft.get("summary") or {}
        if arft_summary.get("status") in {
            "unavailable",
            "not_initialized",
            "not_applicable",
        }:
            items.append(
                _diagnostic_item(
                    (
                        "AUDIT.ARFT_INPUT_UNAVAILABLE"
                        if arft_summary.get("status") == "unavailable"
                        else "AUDIT.ARFT_ADAPTER_MISSING"
                    ),
                    priority="P1",
                    area="auditability",
                    status=(
                        "unavailable"
                        if arft_summary.get("status") == "unavailable"
                        else "adapter_missing"
                    ),
                    evidence="arft_coverage.status=" + str(arft_summary.get("status")),
                    recommendation=(
                        "Keep the input within the bounded legacy-contract limit and "
                        "export/validate the ARFT channels; do not infer failures from "
                        "an unavailable adapter."
                        if arft_summary.get("status") == "unavailable"
                        else "Export legacy pipeline contracts or add an explicit "
                        "typed-object adapter; do not infer ARFT failures from this gap."
                    ),
                    verification="arft_coverage has assessed stages/patterns",
                )
            )
        elif float(arft_summary.get("coverage_score") or 0.0) < 100.0:
            items.append(
                _diagnostic_item(
                    "AUDIT.ARFT_CHANNELS_PARTIAL",
                    priority="P1",
                    area="auditability",
                    status="partial",
                    evidence="arft_coverage.coverage_score<100",
                    recommendation=(
                        "Add the missing evidence channels and keep unassessed patterns "
                        "distinct from observed failures."
                    ),
                    verification="all required channels are covered or explicitly waived",
                )
            )

        closure = workspace.get("closure") or {}
        if closure.get("available") and int(closure.get("blocker_count") or 0) > 0:
            items.append(
                _diagnostic_item(
                    "QUALITY.CLOSURE_BLOCKED",
                    priority="P1",
                    area="scientific_quality",
                    status="blocked",
                    evidence="closure.blocker_count>0",
                    recommendation="Resolve closure blockers and rerun trace/replay/verify.",
                    verification="closure.blocker_count=0",
                )
            )

    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    items.sort(key=lambda row: (priority_order.get(row["priority"], 9), row["id"]))
    counts = Counter(row["priority"] for row in items)
    return {
        "scope": "offline_conformance_diagnostics",
        "quality_claim_allowed": False,
        "official_comparable": False,
        "items": items,
        "priority_counts": dict(sorted(counts.items())),
        "next_required": items[0]["id"] if items else None,
        "disclaimer": (
            "Priorities identify missing measurement or evidence channels; they do not "
            "rank XScientist against another system."
        ),
    }


def _build_reproducibility_fingerprint(
    report: Mapping[str, Any],
    *,
    dataset_sha256: str,
    task_kind: str,
    limit: int,
) -> dict[str, Any]:
    """Return stable input/provenance fields without hashing report timestamps.

    ``generated_at`` and runtime duration are observations and deliberately do
    not participate in the fingerprint.  The remaining fields are bounded,
    redacted summaries, so rerunning the same manifest and checkout yields a
    joinable core fingerprint without claiming bit-for-bit scientific output.
    """

    workspace = report.get("workspace") or {}
    process = workspace.get("process") or {}
    repository = process.get("repository") or {}
    stable = {
        "schema": report.get("schema"),
        "version": report.get("version"),
        "dataset_sha256": dataset_sha256,
        "task_kind": task_kind,
        "limit": int(limit),
        "task_count": (report.get("tasks") or {}).get("count"),
        "valid_task_contracts": (report.get("tasks") or {}).get("valid_task_contracts"),
        "invalid_task_count": len(
            (report.get("tasks") or {}).get("invalid_task_contracts") or []
        ),
        "workspace_available": bool(workspace),
        "workspace_head": repository.get("head"),
        "source_object_count": (workspace.get("object_scan") or {}).get(
            "source_object_count"
        ),
        "stage_coverage": workspace.get("stage_coverage"),
        "process_source_totals": (process.get("limits") or {}).get("source_totals"),
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "fingerprint": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "input_manifest_sha256": dataset_sha256,
        "workspace_head": repository.get("head"),
        "package_version": report.get("version"),
        "schema_versions": [str(report.get("schema") or "")],
        "network_used": False,
        "provider_used": False,
        "seed": None,
        "deterministic_fields": True,
        "excluded_observations": ["generated_at", "runtime", "duration_seconds"],
    }


def persist_benchmark_report(report: dict[str, Any], destination: str | Path) -> Path:
    """Persist only the redacted benchmark report via an atomic JSON write."""

    path = Path(destination).expanduser()
    if path.exists() and path.is_dir():
        raise ValueError("benchmark output must be a file, not a directory")
    atomic_write_json(path, report)
    return path.resolve()


def verify_benchmark_report(
    report: str | Path | dict[str, Any],
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a saved benchmark report without network, providers, or payloads.

    This is intentionally a *report schema/boundary* verifier, not a
    scientific-result or provenance verifier. It checks the published JSON
    Schema and fail-closed non-comparability invariants; it cannot establish
    that the report's observations came from an untampered source workspace.
    ``report_path`` is optional when an in-memory mapping is supplied; when a
    path is supplied it is also used to check the recorded destination digest.
    """

    checks: dict[str, str] = {
        "schema": "not_checked",
        "nested_contracts": "not_checked",
        "comparison_boundary": "not_checked",
        "digest_format": "not_checked",
        "digest_consistency": "not_checked",
        "reproducibility_fingerprint": "not_checked",
        "report_persistence": "not_checked",
    }
    errors: list[str] = []
    payload: Any
    resolved_path: Path | None = None

    def _reject_nonstandard_json_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant: {token}")

    try:
        if isinstance(report, (str, Path)):
            resolved_path = Path(report).expanduser().resolve()
            try:
                report_size = resolved_path.stat().st_size
            except OSError as exc:
                raise ValueError("cannot inspect report file") from exc
            if report_size > _MAX_REPORT_FILE_BYTES:
                raise ValueError(
                    f"report file exceeds the {_MAX_REPORT_FILE_BYTES}-byte safety limit"
                )
            payload = json.loads(
                resolved_path.read_bytes().decode("utf-8"),
                parse_constant=_reject_nonstandard_json_constant,
            )
        else:
            payload = report
        if not isinstance(payload, dict):
            raise ValueError("benchmark report must be a JSON object")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        RuntimeError,
        RecursionError,
        MemoryError,
    ) as exc:
        return {
            "schema": "xscientist.benchmark-report-verification.v1",
            "ok": False,
            "checks": {**checks, "schema": "failed"},
            "errors": ["report_input_unreadable:" + type(exc).__name__],
            "network_used": False,
            "provider_used": False,
        }

    schema_name = {
        "xscientist.autoresearch-conformance.v1": "autoresearch_conformance",
        "xscientist.first-run-benchmark.v1": "first_run_benchmark",
        "xscientist.system-comparison.v1": "system_comparison",
    }.get(str(payload.get("schema") or ""))
    if not schema_name:
        checks["schema"] = "failed"
        errors.append("unknown_schema")
    else:
        try:
            from jsonschema import validate
            from ai_scientist.protocol.schemas import load_schema

            validate(payload, load_schema(schema_name))
            checks["schema"] = "passed"
        except Exception as exc:  # jsonschema errors vary by implementation
            checks["schema"] = "failed"
            errors.append("schema_validation:" + type(exc).__name__)

    def _non_finite_number(value: Any, seen: set[int] | None = None) -> bool:
        """Reject NaN/Infinity even though JSON Schema's ``number`` accepts them.

        Python's in-memory API and ``json.loads`` can otherwise carry these
        values through a report, making threshold/cost fields non-portable
        across JSON implementations.  The seen-set also keeps a malicious
        cyclic mapping from crashing this verifier.
        """

        # Walk iteratively: reports are untrusted input and a deeply nested
        # JSON object must not raise RecursionError inside the verifier.
        visited = 0
        active = set() if seen is None else seen
        stack: list[tuple[Any, int]] = [(value, 0)]
        while stack:
            current, depth = stack.pop()
            visited += 1
            if visited > 100000 or depth > 128:
                return True
            if isinstance(current, float):
                if not math.isfinite(current):
                    return True
                continue
            if not isinstance(current, (Mapping, list, tuple, set)):
                continue
            marker = id(current)
            if marker in active:
                continue
            active.add(marker)
            children = current.values() if isinstance(current, Mapping) else current
            stack.extend((child, depth + 1) for child in children)
        return False

    if _non_finite_number(payload):
        errors.append("non_finite_number")

    is_autoresearch_report = (
        str(payload.get("schema")) == "xscientist.autoresearch-conformance.v1"
    )

    # The top-level conformance schema intentionally leaves workspace details
    # extensible.  Validate the published nested contracts as well, otherwise
    # a caller could replace an evidence/process object with an arbitrary
    # shape while the outer report still appeared valid.
    nested_contract_errors: list[str] = []
    workspace_payload = payload.get("workspace")
    nested_targets: list[tuple[str, Any]] = []
    report_schema = str(payload.get("schema") or "")
    if isinstance(workspace_payload, Mapping):
        nested_targets.extend(
            (f"workspace.{field}", workspace_payload.get(field))
            for field in ("evidence_index", "process", "exploration")
            if field in workspace_payload
            and not (
                report_schema == "xscientist.system-comparison.v1"
                and field == "process"
                and workspace_payload.get(field) is None
            )
        )
    if report_schema == "xscientist.first-run-benchmark.v1":
        if "evidence_index" in payload:
            nested_targets.append(("evidence_index", payload.get("evidence_index")))
    if report_schema == "xscientist.system-comparison.v1":
        local_payload = payload.get("xscientist_local")
        if (
            isinstance(local_payload, Mapping)
            and "process" in local_payload
            and local_payload.get("process") is not None
        ):
            nested_targets.append(
                ("xscientist_local.process", local_payload.get("process"))
            )

    if nested_targets:
        try:
            from jsonschema import validate
            from ai_scientist.protocol.schemas import load_schema

            schema_by_prefix = {
                "evidence_index": "evidence_index",
                "workspace.evidence_index": "evidence_index",
                "workspace.process": "process_audit",
                "workspace.exploration": "exploration_audit",
                "xscientist_local.process": "process_audit",
            }
            for field, nested in nested_targets:
                if nested is None:
                    raise ValueError(f"workspace.{field} must not be null")
                validate(nested, load_schema(schema_by_prefix[field]))
            checks["nested_contracts"] = "passed"
        except Exception as exc:  # jsonschema implementation-specific errors
            checks["nested_contracts"] = "failed"
            nested_contract_errors.append(
                "nested_schema_validation:" + type(exc).__name__
            )
    else:
        checks["nested_contracts"] = "not_applicable"
    errors.extend(nested_contract_errors)
    if is_autoresearch_report:
        if workspace_payload is not None and isinstance(workspace_payload, Mapping):
            required_workspace_fields = {
                "stages",
                "process",
                "evidence_index",
                "exploration",
            }
            if not required_workspace_fields.issubset(workspace_payload):
                errors.append("workspace_contract_incomplete")
        elif workspace_payload is not None:
            errors.append("workspace_contract_invalid")

    execution_raw = payload.get("execution")
    execution_shape_ok = (
        isinstance(execution_raw, Mapping)
        if is_autoresearch_report
        else "execution" not in payload or isinstance(execution_raw, Mapping)
    )
    execution = execution_raw if isinstance(execution_raw, Mapping) else {}
    semantic_ok = True
    if is_autoresearch_report:
        tasks_payload = payload.get("tasks")
        if isinstance(tasks_payload, Mapping):
            count_value = tasks_payload.get("count")
            valid_value = tasks_payload.get("valid_task_contracts")
            invalid_rows = tasks_payload.get("invalid_task_contracts")
            if (
                isinstance(count_value, int)
                and not isinstance(count_value, bool)
                and isinstance(valid_value, int)
                and not isinstance(valid_value, bool)
                and isinstance(invalid_rows, list)
            ):
                expected_ok = valid_value == count_value and not invalid_rows
                if payload.get("ok") is not expected_ok:
                    semantic_ok = False
                if valid_value + len(invalid_rows) != count_value:
                    semantic_ok = False
            else:
                semantic_ok = False
        else:
            semantic_ok = False
        if not semantic_ok:
            errors.append("report_semantics_inconsistent")
    elif report_schema == "xscientist.first-run-benchmark.v1":
        # The first-run report is descriptive rather than a scientific score,
        # but its two threshold fields still form a small integrity contract.
        # Without this check a caller could flip ``ok`` or
        # ``threshold_passed`` in a saved JSON and receive a false PASS.
        first_run_research = payload.get("research")
        status_ok = (
            first_run_research.get("status_ok", True)
            if isinstance(first_run_research, Mapping)
            else False
        )
        status_ok_present = (
            isinstance(first_run_research, Mapping)
            and "status_ok" in first_run_research
        )
        threshold_value = payload.get("threshold_passed")
        duration_value = payload.get("duration_seconds")
        max_value = payload.get("max_seconds")
        threshold_semantics_ok = (
            isinstance(threshold_value, bool)
            and status_ok_present
            and isinstance(status_ok, bool)
            and isinstance(duration_value, (int, float))
            and not isinstance(duration_value, bool)
            and duration_value >= 0
            and (
                (max_value is None and threshold_value is True)
                or (
                    isinstance(max_value, (int, float))
                    and not isinstance(max_value, bool)
                    and max_value > 0
                    # ``duration_seconds`` is rounded for display; allow a
                    # millisecond at the boundary when recomputing.
                    and threshold_value == (duration_value <= max_value + 0.001)
                )
            )
        )
        expected_first_run_ok = bool(status_ok and threshold_value)
        if not threshold_semantics_ok or payload.get("ok") is not expected_first_run_ok:
            semantic_ok = False
            errors.append("report_semantics_inconsistent")
    immutable_ok = (
        execution_shape_ok
        and semantic_ok
        and payload.get("official_comparable") is False
        and payload.get("quality_claim_allowed") is False
        and execution.get("network_used", False) is False
        and execution.get("provider_used", False) is False
    )
    # System comparison uses top-level network/provider fields; first-run has
    # no execution object.  All three public report families are fail-closed.
    if str(payload.get("schema")) == "xscientist.system-comparison.v1":
        immutable_ok = (
            payload.get("official_comparable") is False
            and payload.get("score_claim_allowed") is False
            and payload.get("quality_claim_allowed") is False
            and payload.get("network_used") is False
            and payload.get("provider_used") is False
        )
    elif str(payload.get("schema")) == "xscientist.first-run-benchmark.v1":
        immutable_ok = (
            semantic_ok
            and payload.get("host_paths_disclosed") is False
            and payload.get("network_used") is False
            and payload.get("provider_used") is False
            and payload.get("model_cost_usd") == 0.0
        )
    checks["comparison_boundary"] = "passed" if immutable_ok else "failed"
    if not immutable_ok:
        errors.append("comparison_boundary_mutated")

    def digest_field_errors(value: Any, key: str = "") -> list[str]:
        """Validate only fields whose names promise a digest/hash.

        A report also contains ordinary Git hashes (for example a 40-character
        checkout head) and opaque IDs.  A whole-document regex therefore both
        rejects valid reports and misses malformed values hidden in free text.
        Keep this check field-aware and fail closed for explicit ``sha256:``
        prefixes regardless of their containing key.
        """

        errors_local: list[str] = []
        key_lower = key.lower()
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                child_name = str(child_key)
                errors_local.extend(digest_field_errors(child_value, child_name))
            return errors_local
        if isinstance(value, list):
            for child in value:
                errors_local.extend(digest_field_errors(child, key))
            return errors_local
        if value is None:
            return errors_local
        if not isinstance(value, str):
            if any(token in key_lower for token in ("sha256", "digest", "fingerprint")):
                errors_local.append("invalid_digest_format")
            return errors_local

        if key_lower == "task_manifest_sha256" and value == "sha256:manifest":
            return errors_local
        if value.startswith("sha256:") and not re.fullmatch(
            r"sha256:[0-9a-f]{16,64}", value
        ):
            errors_local.append("invalid_digest_format")
            return errors_local
        if key_lower in {
            "hash_algorithm",
            "hashing_algorithm",
            "digest_scope",
            "hash_scope",
        }:
            return errors_local
        if key_lower == "task_manifest_sha256":
            # Process-audit fairness accepts either a full manifest hash or a
            # deliberately shortened, non-reversible digest.  The latter is
            # useful for redacted reports and is constrained to the same
            # 16--64 hex range as the generic sha256 prefix check above.
            if (
                not re.fullmatch(r"(?:[0-9a-f]{64}|sha256:[0-9a-f]{16,64})", value)
                and value != "sha256:manifest"
            ):
                errors_local.append("invalid_digest_format")
        elif key_lower == "sha256" or key_lower.endswith("_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                errors_local.append("invalid_digest_format")
        elif "fingerprint" in key_lower:
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                errors_local.append("invalid_digest_format")
        elif "digest" in key_lower or "content_hash" in key_lower:
            if not re.fullmatch(
                r"(?:[0-9a-f]{64}|(?:sha256|file|task|object|branch|ref|name):[0-9a-f]{16,64})",
                value,
            ):
                errors_local.append("invalid_digest_format")
        elif "_hash" in key_lower or key_lower == "hash":
            if not re.fullmatch(r"[0-9a-f]{64}|sha256:[0-9a-f]{16,64}", value):
                errors_local.append("invalid_digest_format")
        return errors_local

    try:
        digest_errors = digest_field_errors(payload)
        # Ensure the payload is serializable as JSON even when an in-memory
        # caller supplied a custom object.  This verifier must never throw.
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        digest_errors = ["report_not_json_serializable"]
    digest_ok = not digest_errors
    checks["digest_format"] = "passed" if digest_ok else "failed"
    if not digest_ok:
        errors.append("invalid_digest_format")

    # A manifest digest is repeated deliberately in the dataset, reproducible
    # fingerprint, and process fairness surfaces.  Check those copies agree;
    # format validation alone would allow a locally edited report to splice
    # together otherwise valid but incompatible inputs.
    digest_values: list[str] = []
    dataset_payload = payload.get("dataset")
    if isinstance(dataset_payload, Mapping) and isinstance(
        dataset_payload.get("sha256"), str
    ):
        digest_values.append(dataset_payload["sha256"])
    reproducibility_payload = payload.get("reproducibility")
    if isinstance(reproducibility_payload, Mapping) and isinstance(
        reproducibility_payload.get("input_manifest_sha256"), str
    ):
        digest_values.append(reproducibility_payload["input_manifest_sha256"])
    if isinstance(workspace_payload, Mapping):
        process_payload = workspace_payload.get("process")
        fairness_payload = (
            process_payload.get("fairness")
            if isinstance(process_payload, Mapping)
            else None
        )
        if isinstance(fairness_payload, Mapping) and isinstance(
            fairness_payload.get("task_manifest_sha256"), str
        ):
            digest_values.append(fairness_payload["task_manifest_sha256"])

    def normalize_manifest_digest(value: str) -> str:
        return value[7:] if value.startswith("sha256:") else value

    placeholder_present = "sha256:manifest" in digest_values
    real_digest_values = [
        value for value in digest_values if value != "sha256:manifest"
    ]
    normalized_digests = {
        normalize_manifest_digest(value) for value in real_digest_values
    }
    fairness_missing = False
    if is_autoresearch_report and isinstance(workspace_payload, Mapping):
        process_payload = workspace_payload.get("process")
        fairness_payload = (
            process_payload.get("fairness")
            if isinstance(process_payload, Mapping)
            else None
        )
        fairness_value = (
            fairness_payload.get("task_manifest_sha256")
            if isinstance(fairness_payload, Mapping)
            else None
        )
        fairness_missing = not isinstance(fairness_value, str)
    if fairness_missing or (placeholder_present and real_digest_values):
        checks["digest_consistency"] = "failed"
        errors.append("digest_consistency_mismatch")
    elif len(normalized_digests) <= 1:
        checks["digest_consistency"] = "passed" if digest_values else "not_applicable"
    else:
        checks["digest_consistency"] = "failed"
        errors.append("digest_consistency_mismatch")

    if is_autoresearch_report:
        reproducibility_payload = payload.get("reproducibility")
        tasks_payload = payload.get("tasks")
        dataset_payload = payload.get("dataset")
        if not (
            isinstance(reproducibility_payload, Mapping)
            and isinstance(tasks_payload, Mapping)
            and isinstance(dataset_payload, Mapping)
            and isinstance(dataset_payload.get("sha256"), str)
        ):
            checks["reproducibility_fingerprint"] = "failed"
            errors.append("reproducibility_fingerprint_missing")
        else:
            try:
                expected = _build_reproducibility_fingerprint(
                    payload,
                    dataset_sha256=str(dataset_payload["sha256"]),
                    task_kind=str(tasks_payload.get("filter") or "all"),
                    limit=int(tasks_payload.get("limit")),
                ).get("fingerprint")
                actual = reproducibility_payload.get("fingerprint")
                checks["reproducibility_fingerprint"] = (
                    "passed" if actual == expected else "failed"
                )
                if actual != expected:
                    errors.append("reproducibility_fingerprint_mismatch")
            except (TypeError, ValueError, OverflowError, RecursionError):
                checks["reproducibility_fingerprint"] = "failed"
                errors.append("reproducibility_fingerprint_invalid")

    persistence = payload.get("report_persistence")
    if persistence is None:
        checks["report_persistence"] = "not_requested"
    elif isinstance(persistence, dict) and persistence.get("requested") is True:
        try:
            target = resolved_path or (
                Path(report_path).expanduser().resolve() if report_path else None
            )
        except (OSError, RuntimeError, ValueError):
            target = None
        expected = persistence.get("destination_digest")
        actual = (
            "sha256:" + hashlib.sha256(str(target).encode("utf-8")).hexdigest()[:16]
            if target is not None
            else None
        )
        checks["report_persistence"] = (
            "passed" if actual is not None and expected == actual else "unverified"
        )
        if checks["report_persistence"] != "passed":
            errors.append("report_destination_unverified")
    else:
        checks["report_persistence"] = "failed"
        errors.append("invalid_report_persistence")

    return {
        "schema": "xscientist.benchmark-report-verification.v1",
        "ok": not errors and checks["schema"] == "passed",
        "checks": checks,
        "errors": sorted(set(errors)),
        "network_used": False,
        "provider_used": False,
    }


def _opaque_token(value: Any, *, prefix: str) -> str:
    """Return a stable non-reversible label for untrusted metadata."""

    # Issue/object identifiers are opaque by contract; cap the input before
    # hashing so a malicious payload cannot turn a redaction step into a
    # multi-megabyte string allocation.
    text = str(value or "")[:4096]
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _canonical_domain(row: dict[str, Any]) -> str:
    value = str(row.get("domain_canonical7") or row.get("domain") or "").strip()
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in _CANONICAL_DOMAINS else "other"


def _safe_public_object_kind(value: Any) -> str:
    """Expose only the built-in kind vocabulary in a shareable report."""

    kind = str(value or "").strip()
    return kind if kind in _PUBLIC_OBJECT_KINDS else "extension"


def _safe_public_object_id(value: Any) -> str | None:
    """Return a stable opaque object reference without echoing arbitrary IDs."""

    if value in (None, ""):
        return None
    return _opaque_token(value, prefix="object")


def _recursive_issue_codes(value: Any) -> list[str]:
    """Extract stable opaque issue identifiers without exporting free text."""

    found: list[str] = []
    stack: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    visited = 0
    while stack and visited < 10000:
        current, depth = stack.pop()
        if depth > 64:
            continue
        visited += 1
        if isinstance(current, dict):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            for key, item in current.items():
                if str(key) in _ISSUE_KEYS:
                    values = item if isinstance(item, list) else [item]
                    for row in values:
                        if isinstance(row, dict):
                            code = row.get("code") or row.get("id") or row.get("key")
                        else:
                            code = row
                        if code not in (None, ""):
                            found.append(_opaque_token(code, prefix="issue"))
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            stack.extend((item, depth + 1) for item in current)
    return sorted(set(code for code in found if code))


def _relation_targets(item: dict[str, Any], relation_types: set[str]) -> set[str]:
    return {
        str(relation.get("target"))
        for relation in item.get("relations") or []
        if isinstance(relation, dict)
        and str(relation.get("type") or "") in relation_types
        and relation.get("target")
    }


def _metacognitive_report(
    objects: list[dict[str, Any]], *, truncated: bool = False
) -> dict[str, Any]:
    """Detect acknowledged review issues and whether a gate contained them.

    This is a conservative structural signal inspired by ARFT F.4.  It never
    labels a trajectory from prose alone: an issue must be present in a typed
    review/gate payload.  A held/rejected gate is reported as *contained*, not
    as an uncorrected shipped failure.
    """

    issue_rows: list[dict[str, Any]] = []
    for item in objects:
        if str(item.get("kind") or "") not in {
            "review",
            "research_review",
            "gate_decision",
            "agent_evaluation",
        }:
            continue
        codes = _recursive_issue_codes(item.get("payload") or {})
        for code in codes:
            issue_rows.append(
                {
                    "issue": code,
                    # Keep the raw ID only inside this function for relation
                    # matching.  It is removed before the public report is
                    # returned.
                    "_source_object_id": str(item.get("object_id") or ""),
                    "source_id": _safe_public_object_id(item.get("object_id")),
                    "source_kind": _safe_public_object_kind(item.get("kind")),
                }
            )

    repaired_codes: set[str] = set()
    unresolved: list[dict[str, str]] = []
    for row in issue_rows:
        source = next(
            (
                item
                for item in objects
                if str(item.get("object_id") or "") == row["_source_object_id"]
            ),
            {},
        )
        source_targets = _relation_targets(source, {"evaluates", "depends_on"})
        repaired_here = False
        for candidate in objects:
            if str(candidate.get("object_id") or "") == row["_source_object_id"]:
                continue
            linked = _relation_targets(candidate, _REPAIR_RELATIONS)
            if linked.intersection({row["_source_object_id"]}) or linked.intersection(
                source_targets
            ):
                if str(candidate.get("kind") or "") in _REPAIR_KINDS and str(
                    candidate.get("state") or ""
                ) not in {"failed", "rejected", "blocked", "draft"}:
                    repaired_here = True
                    break
            if (
                row["issue"] in _recursive_issue_codes(candidate.get("payload") or {})
                and str(candidate.get("kind") or "") in _REPAIR_KINDS
                and str(candidate.get("state") or "")
                in {
                    "completed",
                    "verified",
                    "accepted",
                    "promoted",
                }
            ):
                # A repeated issue code in a completed corrective object is a
                # useful fallback for older objects that lacked repair links.
                repaired_here = True
                break
        if repaired_here:
            repaired_codes.add(row["issue"])
        else:
            unresolved.append(row)

    unique_issue_codes = {row["issue"] for row in issue_rows}
    issue_source_ids = {
        row["_source_object_id"] for row in issue_rows if row.get("_source_object_id")
    }
    repair_relation_count = 0
    retest_count = 0
    for candidate in objects:
        if str(candidate.get("state") or "") not in _COMPLETED_STATES:
            continue
        relations = [
            relation
            for relation in candidate.get("relations") or []
            if isinstance(relation, dict)
            and str(relation.get("target") or "") in issue_source_ids
            and str(relation.get("type") or "") in _REPAIR_RELATIONS
        ]
        if relations:
            repair_relation_count += len(relations)
            retest_count += sum(
                str(relation.get("type") or "") == "retests" for relation in relations
            )
    unique_unresolved: dict[str, dict[str, str]] = {}
    for row in unresolved:
        unique_unresolved.setdefault(row["issue"], row)
    unresolved = [
        {
            "issue": row["issue"],
            "source_id": row.get("source_id"),
            "source_kind": row.get("source_kind", "extension"),
        }
        for row in unique_unresolved.values()
    ]

    gates = [item for item in objects if item.get("kind") == "gate_decision"]
    containment_gate_observed = any(
        str((item.get("payload") or {}).get("decision") or "").lower()
        in _CONTAINING_GATE_DECISIONS
        or str((item.get("payload") or {}).get("status") or "").lower()
        in _CONTAINING_GATE_DECISIONS
        or str(item.get("state") or "").lower() in _CONTAINING_GATE_DECISIONS
        for item in gates
    )
    shipped = any(
        str((item.get("payload") or {}).get("decision") or "").lower()
        in {"allow", "allowed", "pass", "promote", "promoted", "accept", "accepted"}
        or str(item.get("state") or "").lower() in {"verified", "promoted", "accepted"}
        for item in gates
    ) or any(
        str(item.get("state") or "").lower() in {"verified", "promoted", "accepted"}
        for item in objects
        if item.get("kind") in {"claim", "manuscript"}
    )
    if not issue_rows:
        status = "not_observed"
    elif not unresolved:
        status = "repaired"
    elif shipped:
        status = "detected"
    elif containment_gate_observed:
        status = "contained"
    else:
        status = "open"
    return {
        "status": status,
        "issue_count": len(unique_issue_codes),
        "repaired_issue_count": len(repaired_codes),
        "unresolved_issue_count": len(unresolved),
        "repair_relation_count": repair_relation_count,
        "retest_count": retest_count,
        "repair_rate": (
            round(len(repaired_codes) / len(unique_issue_codes), 3)
            if unique_issue_codes
            else None
        ),
        "recovery_claim_allowed": False,
        "statistics_scope": (
            "bounded_object_sample" if truncated else "all_visible_objects"
        ),
        "truncated": bool(truncated),
        "shipping_gate_observed": shipped,
        "containment_gate_observed": containment_gate_observed,
        "unresolved_issues": unresolved,
        "arft_inspired_code": "F.4" if unresolved and shipped else None,
        "interpretation": {
            "not_observed": "no typed review issue was found",
            "repaired": "typed issues have a completed corrective artifact",
            "contained": "issues remain, but a hold/reject gate prevents shipping",
            "detected": "an unresolved issue coexists with an allow/verified state; inspect before delivery",
            "open": "issues remain and no explicit corrective or containing gate was observed",
        }[status],
    }


def _load_autoresearch_tasks_with_digest(
    path: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Load tasks and hash the exact bytes parsed in one bounded read."""

    if not path.is_file():
        raise ValueError("tasks file does not exist")
    try:
        raw_bytes = path.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot read tasks file") from exc
    if len(raw_bytes) > _MAX_TASK_FILE_BYTES:
        raise ValueError(
            f"tasks file exceeds the {_MAX_TASK_FILE_BYTES}-byte safety limit"
        )
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("cannot decode tasks file as UTF-8") from exc

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant: {token}")

    rows: Any
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line, parse_constant=reject_constant))
            except (
                json.JSONDecodeError,
                ValueError,
                RecursionError,
                MemoryError,
            ) as exc:
                raise ValueError(
                    "invalid JSON on line "
                    f"{line_number}: {getattr(exc, 'msg', str(exc))}"
                ) from exc
    else:
        try:
            rows = json.loads(raw, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError, RecursionError, MemoryError) as exc:
            raise ValueError(
                f"invalid JSON tasks file: {getattr(exc, 'msg', str(exc))}"
            ) from exc
        if isinstance(rows, dict):
            rows = rows.get("tasks", rows.get("data", rows))
    if not isinstance(rows, list) or not rows:
        raise ValueError("tasks file must contain a non-empty JSON list or JSONL")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every task row must be a JSON object")
    return rows, hashlib.sha256(raw_bytes).hexdigest()


def _load_autoresearch_tasks(path: Path) -> list[dict[str, Any]]:
    """Load a local JSONL/JSON task file without making network requests."""

    rows, _digest = _load_autoresearch_tasks_with_digest(path)
    return rows


def _exploration_audit(root: Path) -> dict[str, Any]:
    """Summarize ARA exploration choices without exporting node content.

    Missing graphs are reported as ``unavailable`` rather than as zero
    attempts.  Status tokens are mapped to a fixed vocabulary and all IDs,
    prompts, code, and stop-reason prose stay local.
    """

    ara_root = root / "ara"
    all_graph_paths: list[Path] = []
    graph_discovery_truncated = False
    walk_entries = 0
    discovery_errors = 0
    if ara_root.is_symlink():
        # Do not follow an externally-owned ARA tree; distinguish this safety
        # boundary from a workspace that simply has no exploration graph.
        discovery_errors = 1
    elif ara_root.is_dir():
        stack = [ara_root]
        while stack and not graph_discovery_truncated:
            current_path = stack.pop()
            try:
                entries: list[os.DirEntry[str]] = []
                with os.scandir(current_path) as iterator:
                    for entry in iterator:
                        if len(entries) >= _EXPLORATION_MAX_WALK_ENTRIES:
                            graph_discovery_truncated = True
                            break
                        entries.append(entry)
                entries.sort(key=lambda entry: entry.name)
            except OSError:
                # A discovered graph may still be readable elsewhere; preserve
                # the bounded audit and let the graph read phase report errors.
                discovery_errors += 1
                continue
            for entry in entries:
                walk_entries += 1
                if walk_entries > _EXPLORATION_MAX_WALK_ENTRIES:
                    graph_discovery_truncated = True
                    break
                try:
                    if entry.is_symlink():
                        # A skipped symlink is an observed boundary, not
                        # proof that the corresponding graph is absent.
                        discovery_errors += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue
                    if entry.name == "exploration_graph.json" and entry.is_file(
                        follow_symlinks=False
                    ):
                        all_graph_paths.append(Path(entry.path))
                        if len(all_graph_paths) >= _MAX_EXPLORATION_GRAPH_FILES:
                            graph_discovery_truncated = True
                            break
                except OSError:
                    discovery_errors += 1
                    continue
    elif ara_root.exists():
        # A regular file (or other non-directory marker) at ``ara/`` is a
        # malformed exploration store, not an uninitialized workspace.
        discovery_errors = 1
    all_graph_paths.sort()
    graph_paths = all_graph_paths[:_MAX_EXPLORATION_GRAPH_FILES]
    if not graph_paths:
        return {
            "schema": "xscientist.exploration-audit.v1",
            "status": (
                "unreadable"
                if discovery_errors
                else (
                    "partially_observed" if graph_discovery_truncated else "unavailable"
                )
            ),
            "graph_count": 0,
            "node_count": None,
            "unknown_nodes": None,
            "planned": None,
            "attempted": None,
            "completed": None,
            "failed": None,
            "discarded": None,
            "crashed": None,
            "unattempted": None,
            "stop_reasons": {},
            "statistics_scope": (
                "unreadable_graph_search"
                if discovery_errors
                else (
                    "bounded_graph_search_partial"
                    if graph_discovery_truncated
                    else "no_exploration_graph"
                )
            ),
            "coverage_claim_allowed": False,
            "node_content_included": False,
            "truncated": graph_discovery_truncated,
            "read_error_count": discovery_errors,
            "walk_entries_observed": walk_entries,
            "counts_are_nonexclusive": True,
        }

    counts = Counter(
        planned=0,
        attempted=0,
        completed=0,
        failed=0,
        discarded=0,
        crashed=0,
    )
    stop_reasons: Counter[str] = Counter()
    node_count = 0
    read_errors = discovery_errors
    unknown_nodes = 0
    truncated = (
        graph_discovery_truncated or len(all_graph_paths) > _MAX_EXPLORATION_GRAPH_FILES
    )
    node_budget = _MAX_EXPLORATION_NODES
    for graph_path in graph_paths:
        if node_budget <= 0:
            truncated = True
            break
        try:
            if graph_path.stat().st_size > _MAX_EXPLORATION_GRAPH_BYTES:
                read_errors += 1
                truncated = True
                continue
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            RecursionError,
            MemoryError,
        ):
            read_errors += 1
            continue
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        if not isinstance(nodes, list):
            read_errors += 1
            continue
        per_graph_limit = min(_MAX_WORKSPACE_AUDIT_OBJECTS, node_budget)
        if len(nodes) > per_graph_limit:
            truncated = True
        for node in nodes[:per_graph_limit]:
            node_budget -= 1
            node_count += 1
            if not isinstance(node, dict):
                unknown_nodes += 1
                read_errors += 1
                continue
            raw_state = (
                str(
                    node.get("state") or node.get("status") or node.get("outcome") or ""
                )
                .strip()
                .lower()
            )
            # ARA's canonical graph schema historically uses ``is_buggy`` and
            # ``metric``/``evaluation_report`` instead of a lifecycle state.
            # Map only these explicit fields; otherwise preserve an unknown
            # node rather than manufacturing a completed/failed result.
            if not raw_state:
                if node.get("is_buggy") is True:
                    raw_state = "failed"
                elif node.get("is_buggy") is False and (
                    node.get("metric") is not None
                    or node.get("evaluation_report") is not None
                ):
                    raw_state = "completed"
                elif (
                    node.get("metric") is not None
                    or node.get("evaluation_report") is not None
                ):
                    raw_state = "attempted"
            known_states = (
                _EXPLORATION_PLANNED_STATES
                | _EXPLORATION_COMPLETED_STATES
                | _EXPLORATION_FAILED_STATES
                | {"attempted", "running", "discarded", "dropped", "pruned"}
            )
            if raw_state and raw_state not in known_states:
                unknown_nodes += 1
            explicit_attempt = node.get("attempted")
            attempted = bool(
                explicit_attempt is True
                or node.get("execution")
                or node.get("execution_isolation")
                or raw_state
                in (
                    _EXPLORATION_COMPLETED_STATES
                    | _EXPLORATION_FAILED_STATES
                    | {"attempted", "running"}
                )
            )
            if raw_state in _EXPLORATION_PLANNED_STATES or explicit_attempt is False:
                counts["planned"] += 1
            elif not raw_state and explicit_attempt is None:
                unknown_nodes += 1
            if attempted:
                counts["attempted"] += 1
            if raw_state in _EXPLORATION_COMPLETED_STATES:
                counts["completed"] += 1
            if raw_state in _EXPLORATION_FAILED_STATES:
                counts["failed"] += 1
            if raw_state in {"discarded", "dropped", "pruned", "rejected"}:
                counts["discarded"] += 1
            if raw_state in {"crashed", "error"}:
                counts["crashed"] += 1
            reason_text = str(
                node.get("stop_reason") or node.get("reason") or ""
            ).lower()
            reason = "unknown"
            reason_tokens = re.findall(r"[a-z0-9]+", reason_text)
            for candidate in _EXPLORATION_STOP_REASON_ORDER:
                candidate_tokens = candidate.split("_")
                width = len(candidate_tokens)
                for index in range(max(0, len(reason_tokens) - width + 1)):
                    if reason_tokens[index : index + width] != candidate_tokens:
                        continue
                    if index and reason_tokens[index - 1] in {
                        "not",
                        "no",
                        "without",
                        "never",
                    }:
                        continue
                    reason = candidate
                    break
                if reason != "unknown":
                    break
            # Missing/empty prose is itself an unobserved stop reason; expose
            # it as ``unknown`` rather than silently making the decision trace
            # look complete.
            stop_reasons[reason if reason_text else "unknown"] += 1

    counts["unattempted"] = max(0, counts["planned"] - counts["attempted"])
    return {
        "schema": "xscientist.exploration-audit.v1",
        "status": (
            "unreadable"
            if not node_count and read_errors
            else (
                "observed_empty"
                if not node_count
                else (
                    "partially_observed"
                    if unknown_nodes or read_errors or truncated
                    else "observed"
                )
            )
        ),
        "graph_count": len(graph_paths),
        "node_count": node_count,
        "unknown_nodes": unknown_nodes,
        **dict(counts),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "statistics_scope": (
            "bounded_graph_nodes_partial"
            if truncated and (unknown_nodes or read_errors)
            else (
                "all_graph_nodes_partial"
                if unknown_nodes or read_errors
                else "bounded_graph_nodes" if truncated else "all_graph_nodes"
            )
        ),
        "truncated": truncated,
        "read_error_count": read_errors,
        "walk_entries_observed": walk_entries,
        "coverage_claim_allowed": False,
        "node_content_included": False,
        "counts_are_nonexclusive": True,
    }


def _workspace_stage_report(
    root: Path,
    *,
    task_manifest_sha256: str | None = None,
    task_count: int | None = None,
    task_filter: str = "all",
    task_limit: int | None = None,
) -> dict[str, Any]:
    """Summarize six-stage artifact coverage using validated local evidence."""

    root = root.resolve()

    def local_exists(relative: str) -> bool:
        """Check an allowlisted relative marker without crossing symlinks."""

        current = root
        try:
            for part in Path(relative).parts:
                current = current / part
                if current.is_symlink():
                    return False
            return current.exists()
        except (OSError, RuntimeError):
            return False

    objects: list[dict[str, Any]] = []
    read_errors: list[str] = []
    source_object_count = 0
    objects_truncated = False
    source_kind_counts: Counter[str] = Counter()
    _source_stats: dict[str, Any] = {}
    try:
        # Use the same bounded scanner as the Git-like process surface.  A
        # stage report must not read an unbounded object store merely because
        # the caller requested a benchmark summary.
        from .process_audit import _bounded_object_scan

        (
            objects,
            source_object_count,
            objects_truncated,
            object_read_errors,
            source_kind_counts,
            _source_decision_objects,
            _source_stats,
        ) = _bounded_object_scan(root, _MAX_WORKSPACE_AUDIT_OBJECTS)
        read_errors.extend(object_read_errors)
    except (
        OSError,
        ValueError,
        TypeError,
        UnicodeError,
        OverflowError,
        RuntimeError,
        RecursionError,
        MemoryError,
    ) as exc:
        # Legacy/file-only workspaces are still reportable; a damaged typed
        # store is surfaced separately instead of being mistaken for empty.
        read_errors.append(type(exc).__name__)

    # Stage claims are based only on validated payloads.  Directory names are
    # useful source totals, but a corrupt/placeholder JSON file must not count
    # as scientific evidence merely because its parent directory is named
    # ``claim`` or ``review``.
    counts: Counter[str] = Counter(str(item.get("kind") or "") for item in objects)

    def count(*kinds: str) -> int:
        return sum(counts[kind] for kind in kinds)

    def criterion(code: str, passed: bool, detail: str) -> dict[str, Any]:
        return {"code": code, "passed": bool(passed), "detail": detail}

    stages: dict[str, dict[str, Any]] = {}
    for stage, spec in _AUTORESEARCH_STAGES.items():
        kind_hits = {kind: counts[kind] for kind in spec["kinds"] if counts[kind]}
        path_hits = [path for path in spec["paths"] if local_exists(path)]
        if stage == "ideation_planning":
            criteria = [
                criterion("question", count("question") > 0, "typed question"),
                criterion(
                    "falsifiable_hypothesis",
                    any(
                        item.get("kind") == "hypothesis"
                        and bool((item.get("payload") or {}).get("falsifier"))
                        for item in objects
                    ),
                    "hypothesis with explicit falsifier",
                ),
                criterion(
                    "research_plan",
                    count("research_plan", "experiment_design") > 0,
                    "typed plan or experiment design",
                ),
            ]
        elif stage == "retrieval_synthesis":
            criteria = [
                criterion(
                    "retrieval_trace",
                    count("search_plan", "search_receipt") > 0,
                    "search plan/receipt",
                ),
                criterion(
                    "source_provenance",
                    count("source_snapshot", "source_update", "passage_evidence") > 0,
                    "source snapshot/update or passage evidence",
                ),
                criterion(
                    "synthesis",
                    count("evidence_synthesis") > 0,
                    "explicit evidence synthesis",
                ),
            ]
        elif stage == "execution_implementation":
            completed = sum(
                1
                for item in objects
                if item.get("kind") == "experiment_attempt"
                and str(item.get("state") or "")
                in {"completed", "verified", "accepted"}
            )
            criteria = [
                criterion(
                    "attempt", count("experiment_attempt") > 0, "experiment attempt"
                ),
                criterion("completed_attempt", completed > 0, "completed attempt"),
                criterion(
                    "execution_receipt",
                    count("tool_evidence", "reproduction") > 0
                    or any(
                        any(
                            key in (item.get("payload") or {})
                            for key in ("command", "run_id", "seed", "metrics")
                        )
                        for item in objects
                        if item.get("kind") == "experiment_attempt"
                    ),
                    "command/run/seed/metric anchor",
                ),
            ]
        elif stage == "analysis_interpretation":
            criteria = [
                criterion(
                    "evidence",
                    count("evidence", "effect_estimate") > 0,
                    "typed evidence",
                ),
                criterion(
                    "inference",
                    count("inference", "warrant") > 0,
                    "inference or warrant",
                ),
                criterion(
                    "evidence_binding",
                    any(
                        any(
                            str(relation.get("type") or "")
                            in {"has_premise", "derived_from", "depends_on"}
                            for relation in item.get("relations") or []
                        )
                        for item in objects
                        if item.get("kind") in {"inference", "claim"}
                    ),
                    "inference/claim relation to upstream evidence",
                ),
            ]
        elif stage == "writing_documentation":
            criteria = [
                criterion(
                    "manuscript",
                    count("manuscript", "report", "paper") > 0,
                    "manuscript/report",
                ),
                criterion("claim", count("claim") > 0, "typed claim"),
                criterion(
                    "claim_trace",
                    any(
                        any(
                            str(relation.get("type") or "")
                            in {"depends_on", "supports", "refutes"}
                            for relation in item.get("relations") or []
                        )
                        for item in objects
                        if item.get("kind") in {"claim", "manuscript"}
                    ),
                    "claim/manuscript relation",
                ),
            ]
        else:
            criteria = [
                criterion(
                    "review", count("review", "research_review") > 0, "typed review"
                ),
                criterion("gate", count("gate_decision") > 0, "typed gate decision"),
                criterion(
                    "independent_check",
                    count("reproduction", "challenge") > 0
                    or any(
                        bool((item.get("payload") or {}).get("independence"))
                        for item in objects
                        if item.get("kind")
                        in {"review", "research_review", "gate_decision"}
                    ),
                    "independent reproduction/challenge or independence receipt",
                ),
            ]
        passed = sum(bool(item["passed"]) for item in criteria)
        required_passed = bool(criteria) and all(
            item["passed"] for item in criteria[:2]
        )
        # A bounded sample can show that a signal exists, but cannot prove a
        # complete stage over the full store.  Keep the score visible while
        # refusing to mark the stage covered/complete until the scan is exact.
        evidence_bar = required_passed and not objects_truncated
        complete = passed == len(criteria) and not objects_truncated
        stages[stage] = {
            "label": _STAGE_LABELS[stage],
            "covered": evidence_bar,
            "complete": complete,
            "required_criteria": [item["code"] for item in criteria[:2]],
            "status": (
                "ready"
                if evidence_bar and complete
                else (
                    "bounded_sample"
                    if objects_truncated and passed
                    else "partial" if passed else "missing"
                )
            ),
            "score": round(100.0 * passed / max(len(criteria), 1), 1),
            "criteria": criteria,
            "object_counts": kind_hits,
            "artifact_paths": path_hits,
        }

    metacognition = _metacognitive_report(objects, truncated=objects_truncated)
    signals: list[dict[str, Any]] = []
    if metacognition["status"] == "detected":
        signals.append(
            {
                "code": "XSCIENTIST.F4_UNCORRECTED_SELF_AWARENESS",
                "severity": "blocker",
                "detail": metacognition["interpretation"],
            }
        )
    elif metacognition["status"] == "contained":
        signals.append(
            {
                "code": "XSCIENTIST.F4_CONTAINED_REVIEW_DEBT",
                "severity": "warning",
                "detail": metacognition["interpretation"],
            }
        )
    elif metacognition["status"] == "open":
        signals.append(
            {
                "code": "XSCIENTIST.F4_OPEN_REVIEW_DEBT",
                "severity": "warning",
                "detail": metacognition["interpretation"],
            }
        )

    closure: dict[str, Any]
    # Closure verification re-reads committed trees and typed payloads.  Do
    # not let it bypass the bounded object scan: a truncated or unreadable
    # source set is explicitly reported as an unavailable closure rather than
    # silently triggering an unbounded second traversal.
    closure_guarded = objects_truncated or bool(
        (_source_stats or {}).get("read_error_count")
    )
    try:
        from .research_git import ResearchGitError

        if closure_guarded:
            raise RuntimeError("bounded_object_scan_incomplete")
        from .research_closure import audit_research_closure, closure_level_summary

        closure_report = audit_research_closure(
            root, level="verify", verify_objects=False
        )
        closure = {
            "available": True,
            "commit": closure_report.get("commit"),
            "target_level": closure_report.get("target_level"),
            "levels": closure_level_summary(closure_report),
            "blocker_count": len(closure_report.get("blockers") or []),
            "warning_count": len(closure_report.get("warnings") or []),
        }
    except (
        OSError,
        ValueError,
        KeyError,
        ResearchGitError,
        UnicodeError,
        OverflowError,
        RuntimeError,
        RecursionError,
        MemoryError,
    ) as exc:
        closure = {
            "available": False,
            "commit": None,
            "target_level": None,
            "levels": {},
            "blocker_count": 0,
            "warning_count": 0,
            "reason": (
                "bounded_object_scan_incomplete"
                if closure_guarded
                else type(exc).__name__
            ),
        }

    arft_coverage: dict[str, Any]
    legacy_contract_present = any(
        local_exists(name)
        for name in (
            "pipeline_manifest.json",
            "idea_cards.json",
            "stage_standards.json",
        )
    )
    if not legacy_contract_present:
        typed_objects_root = root / ".xscientist" / "objects"
        typed_objects_present = False
        if local_exists(".xscientist/objects") and typed_objects_root.is_dir():
            try:
                for kind_entry in os.scandir(typed_objects_root):
                    if kind_entry.is_symlink() or not kind_entry.is_dir(
                        follow_symlinks=False
                    ):
                        continue
                    with os.scandir(kind_entry.path) as object_entries:
                        typed_objects_present = any(
                            entry.name.endswith(".json")
                            and not entry.is_symlink()
                            and entry.is_file(follow_symlinks=False)
                            for entry in object_entries
                        )
                    if typed_objects_present:
                        break
            except OSError:
                typed_objects_present = False
        arft_coverage = {
            "schema": "xscientist.arft-coverage.v1",
            "quality_claim_allowed": False,
            "benchmark_compatible": False,
            "summary": {
                "status": (
                    "not_applicable" if typed_objects_present else "not_initialized"
                ),
                "source": (
                    "research_vcs_typed_objects"
                    if typed_objects_present
                    else "no_local_artifacts"
                ),
                "reason": (
                    "typed-object to ARFT adapter is not inferred by this pilot; "
                    "use build_arft_coverage after exporting legacy pipeline contracts"
                    if typed_objects_present
                    else "no legacy pipeline contracts or typed objects were found"
                ),
            },
            "stages": [],
        }
    else:
        try:
            legacy_paths = [
                root / name
                for name in (
                    "pipeline_manifest.json",
                    "idea_cards.json",
                    "stage_standards.json",
                )
                if local_exists(name)
            ]
            legacy_bytes = 0
            legacy_oversized = False
            for legacy_path in legacy_paths:
                try:
                    size = legacy_path.stat().st_size
                except OSError:
                    legacy_oversized = True
                    break
                legacy_bytes += size
                if (
                    size > _MAX_LEGACY_CONTRACT_BYTES
                    or legacy_bytes > _MAX_LEGACY_CONTRACT_BYTES * 2
                ):
                    legacy_oversized = True
                    break
            if legacy_oversized:
                raise RuntimeError("legacy_contract_bounds_exceeded")
            from ai_scientist.utils.arft_coverage import build_arft_coverage

            arft_report = build_arft_coverage(root)
            arft_coverage = {
                "schema": arft_report.get("schema"),
                "quality_claim_allowed": False,
                "benchmark_compatible": False,
                "summary": arft_report.get("summary") or {},
                "stages": arft_report.get("stages") or [],
            }
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            UnicodeError,
            RecursionError,
            MemoryError,
        ):
            arft_coverage = {
                "schema": "xscientist.arft-coverage.v1",
                "quality_claim_allowed": False,
                "benchmark_compatible": False,
                "summary": {
                    "status": "unavailable",
                    "reason": "legacy_contract_bounds_or_read_error",
                },
                "stages": [],
            }

    covered = sum(bool(item["covered"]) for item in stages.values())
    public_counts: Counter[str] = Counter()
    for kind, value in counts.items():
        public_counts[kind if kind in _PUBLIC_OBJECT_KINDS else "extension"] += value
    try:
        from .process_audit import build_process_summary

        process = build_process_summary(
            root,
            task_manifest_sha256=task_manifest_sha256,
            task_count=task_count,
            task_filter=task_filter,
            task_limit=task_limit,
            gold_fields_used=False,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        UnicodeError,
        OverflowError,
        RuntimeError,
        RecursionError,
        MemoryError,
    ):
        from .process_audit import _unavailable_summary

        process = _unavailable_summary(
            task_manifest_sha256=task_manifest_sha256,
            task_count=task_count,
            task_filter=task_filter,
            task_limit=task_limit,
            gold_fields_used=False,
            errors=["process_audit_error"],
            limits={
                "max_branches": 32,
                "max_commits": 32,
                "max_artifacts": 96,
                "max_decisions": 32,
            },
        )
    try:
        from .evidence_index import build_evidence_index

        evidence_index = build_evidence_index(root)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        RecursionError,
        MemoryError,
    ):
        evidence_index = {
            "schema": "xscientist.evidence-index.v1",
            "available": False,
            "mode": "unavailable",
            "hash_algorithm": "sha256",
            "workspace_root_disclosed": False,
            "paths_disclosed": False,
            "raw_content_included": False,
            "workspace_mutated": False,
            "limits": {
                "max_files_per_category": 512,
                "max_bytes": 32 * 1024 * 1024,
            },
            "categories": {},
            "ara_contract": {
                "manifest_count": 0,
                "lock_count": 0,
                "graph_count": 0,
                "verify_report_count": 0,
                "lock_state": "not_observed",
                "control_digest": None,
                "digest_scope": "observed_control_files",
                "bytes_read": 0,
                "fsck_run": False,
                "bundle_created": False,
                "truncated": False,
                "raw_payloads_included": False,
            },
            "truncated": False,
            "read_error_count": 1,
        }
    try:
        exploration = _exploration_audit(root)
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        RuntimeError,
        RecursionError,
        MemoryError,
    ):
        # A malformed/vanishing graph must not abort the whole conformance
        # report or look like an empty successful search.
        exploration = {
            "schema": "xscientist.exploration-audit.v1",
            "status": "unreadable",
            "graph_count": 0,
            "node_count": None,
            "unknown_nodes": None,
            "planned": None,
            "attempted": None,
            "completed": None,
            "failed": None,
            "discarded": None,
            "crashed": None,
            "unattempted": None,
            "stop_reasons": {
                "budget_exhausted": 0,
                "tool_error": 0,
                "model_selection": 0,
                "missing_data": 0,
                "unknown": 0,
            },
            "statistics_scope": "unreadable_graph",
            "truncated": False,
            "read_error_count": 1,
            "walk_entries_observed": 0,
            "coverage_claim_allowed": False,
            "node_content_included": False,
            "counts_are_nonexclusive": True,
        }
    return {
        "workspace": ".",
        "quality_claim_allowed": False,
        "score_semantics": "structural_stage_coverage_only",
        "stages": stages,
        "stage_coverage": round(covered / len(stages), 3),
        "stage_score": round(
            sum(float(item["score"]) for item in stages.values()) / max(len(stages), 1),
            1,
        ),
        "object_counts": dict(sorted(public_counts.items())),
        "object_store_read_errors": read_errors,
        "object_scan": {
            "visible_object_count": len(objects),
            "source_object_count": source_object_count,
            "source_kind_counts": dict(sorted(source_kind_counts.items())),
            "truncated": objects_truncated,
            "source_scan_complete": bool(
                (_source_stats or {}).get("source_scan_complete", not objects_truncated)
            ),
            "source_scan_scope": str(
                (_source_stats or {}).get(
                    "source_scan_scope",
                    "bounded_object_prefix" if objects_truncated else "all_objects",
                )
            ),
            "source_scan_entries": int(
                (_source_stats or {}).get("source_scan_entries", source_object_count)
                or 0
            ),
            "read_error_count": int(
                (_source_stats or {}).get("read_error_count", read_errors) or 0
            ),
            "statistics_scope": (
                "bounded_object_sample" if objects_truncated else "all_objects"
            ),
        },
        "closure": closure,
        "metacognition": metacognition,
        "metacognitive_signals": signals,
        "arft_coverage": arft_coverage,
        "process": process,
        "evidence_index": evidence_index,
        "exploration": exploration,
        "review_stage_covered": bool(stages["self_verification_review"]["covered"]),
    }


def benchmark_autoresearch_pilot(
    tasks: str | Path,
    *,
    workspace: str | Path | None = None,
    limit: int = 20,
    task_kind: str = "all",
) -> dict[str, Any]:
    """Run an offline AutoResearchEval-inspired conformance/pilot report.

    The task file is supplied by the caller (JSONL or JSON); no gold answers,
    source-paper text, model, provider, or network are used.  Consequently the
    result is an artifact-conformance report, not an official benchmark score.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be greater than zero")
    if task_kind not in {"all", "open-ended", "optimization"}:
        raise ValueError("task_kind must be all, open-ended, or optimization")
    task_path = Path(tasks).expanduser().resolve()
    all_rows, dataset_sha256 = _load_autoresearch_tasks_with_digest(task_path)

    def classify(row: dict[str, Any]) -> str:
        return (
            "optimization"
            if bool(row.get("task_name") or row.get("workflow_topology"))
            else "open-ended"
        )

    filtered = [
        row for row in all_rows if task_kind == "all" or classify(row) == task_kind
    ]
    if not filtered:
        raise ValueError(f"no {task_kind} tasks found in the supplied file")
    rows = filtered[:limit]
    kind_counts: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    valid_contracts = 0
    invalid_contracts: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        optimization = classify(row) == "optimization"
        kind_counts[classify(row)] += 1
        domains[_canonical_domain(row)] += 1
        missing: list[str] = []
        if not str(row.get("task_id") or "").strip():
            missing.append("task_id")
        if (
            not str(row.get("domain") or "").strip()
            and not str(row.get("domain_canonical7") or "").strip()
        ):
            missing.append("domain")
        if not optimization:
            if not str(row.get("premise") or "").strip():
                missing.append("premise")
            if not str(row.get("tension") or "").strip():
                missing.append("tension")
        if missing:
            invalid_contracts.append(
                {
                    "row": index,
                    "task_id_digest": (
                        _opaque_token(row.get("task_id"), prefix="task")
                        if row.get("task_id")
                        else None
                    ),
                    "kind": classify(row),
                    "missing": missing,
                }
            )
        else:
            valid_contracts += 1
    report: dict[str, Any] = {
        "schema": "xscientist.autoresearch-conformance.v1",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system().lower(),
        },
        "ok": valid_contracts == len(rows),
        "benchmark_family": "AutoResearchEval-inspired",
        "comparison_context": {
            "mode": "qualitative_source_audit",
            "matrix": "docs/SYSTEM_COMPARISON.md",
            "external_scores_injected": False,
            "same_task_slice_verified": False,
            "same_budget_verified": False,
            "same_evaluator_verified": False,
            "note": (
                "Use xscientist benchmark systems for the source-audited matrix; "
                "this pilot remains an XScientist process/conformance measurement."
            ),
        },
        "reference": {
            "paper": "https://arxiv.org/abs/2608.14905",
            "repository": "https://github.com/PrentisAI/AutoResearchEval",
            "dataset": "https://huggingface.co/datasets/PrentisAI/AutoResearchEval",
        },
        "related_references": [
            {
                "name": "Beyond Final Scores",
                "paper": "https://arxiv.org/abs/2608.13417",
                "note": "different 36-task/AutoLab benchmark; not used by this pilot",
            }
        ],
        "official_comparable": False,
        "scope": "offline artifact conformance; no agent rollout evaluation",
        "quality_claim_allowed": False,
        "score_semantics": "task_contract_and_structural_observability_only",
        "evidence_retention": {
            "mode": "read_only_bounded_index",
            "report_persisted_by_api": False,
            "report_persisted_by_cli_default": False,
            "task_manifest_copied": False,
            "raw_trajectory_copied": False,
            "ara_snapshot_written": False,
            "cas_payload_copied": False,
            "process_payloads_included": False,
            "workspace_artifacts_untouched": True,
            "explicit_exports": [
                "xscientist benchmark autoresearch --output <report.json>",
                "xscientist research fsck --repo <workspace>",
                "xscientist ara bundle --ara <ara-run> --dest <audit-bundle>",
                "xscientist research export --repo <workspace> --dest <export> --include-payloads",
            ],
            "note": (
                "Use --output or redirect CLI JSON to retain the report; export raw "
                "ARA/CAS only with an explicit, potentially sensitive command."
            ),
        },
        "human_baseline": {
            "status": "not_reported",
            "evidence_class": "not_reported",
            "matched_arm": False,
            "score": None,
            "participants_n": None,
            "local_runs": 0,
            "external_scores_injected": False,
            "reason_code": "NO_LOCAL_HUMAN_RUN",
            "inventory": "docs/HUMAN_BASELINES.md",
        },
        "tasks": {
            "count": len(rows),
            "limit": limit,
            "filter": task_kind,
            "available_before_filter": len(all_rows),
            "types": dict(sorted(kind_counts.items())),
            "domains": dict(sorted(domains.items())),
            "valid_task_contracts": valid_contracts,
            "invalid_task_contracts": invalid_contracts,
            "gold_fields_used": False,
            "gold_fields_omitted": True,
        },
        "execution": {
            "network_used": False,
            "provider_used": False,
            "model_used": False,
            "model_cost_usd": 0.0,
            "rollouts_evaluated": 0,
        },
        "dataset": {
            "file_name_digest": _opaque_token(task_path.name, prefix="file"),
            "sha256": dataset_sha256,
            "paths_disclosed": False,
        },
    }
    if workspace is None:
        report["workspace"] = None
        report["next_step"] = (
            "rerun with --workspace to inspect six-stage artifact coverage"
        )
    else:
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("workspace does not exist or is not a directory")
        report["workspace"] = _workspace_stage_report(
            root,
            task_manifest_sha256=dataset_sha256,
            task_count=len(rows),
            task_filter=task_kind,
            task_limit=limit,
        )
    report["reproducibility"] = _build_reproducibility_fingerprint(
        report,
        dataset_sha256=dataset_sha256,
        task_kind=task_kind,
        limit=limit,
    )
    report["diagnostics"] = _build_benchmark_diagnostics(report)
    return report


__all__ = [
    "benchmark_first_run",
    "benchmark_autoresearch_pilot",
    "persist_benchmark_report",
    "verify_benchmark_report",
]
