"""Reproducible, provider-free first-run usability benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be greater than zero")
    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="xscientist-first-run-")
        root = Path(temporary.name) / "study"
    else:
        root = Path(workspace).expanduser().resolve()
    started = time.perf_counter()
    try:
        demo = create_autopilot_demo(root, profile=profile, language="en")
        status = build_workspace_status(root, language="en")
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
            "profile": profile,
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
                "budget_available": status["budget"]["available"],
                "next_step": (status["next_steps"] or [{}])[0].get("code"),
            },
            "workspace_retained": workspace is not None,
            "host_paths_disclosed": False,
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
_MAX_WORKSPACE_AUDIT_OBJECTS = 512
_CANONICAL_DOMAINS = {
    "biology",
    "chemistry",
    "geophysics",
    "material_science",
    "medicine",
    "physics",
    "scientific_computing",
}
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


def _opaque_token(value: Any, *, prefix: str) -> str:
    """Return a stable non-reversible label for untrusted metadata."""

    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]
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
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in _ISSUE_KEYS:
                values = item if isinstance(item, list) else [item]
                for row in values:
                    if isinstance(row, dict):
                        code = row.get("code") or row.get("id") or row.get("key")
                    else:
                        code = row
                    if code not in (None, ""):
                        found.append(_opaque_token(code, prefix="issue"))
            found.extend(_recursive_issue_codes(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_recursive_issue_codes(item))
    return sorted(set(code for code in found if code))


def _relation_targets(item: dict[str, Any], relation_types: set[str]) -> set[str]:
    return {
        str(relation.get("target"))
        for relation in item.get("relations") or []
        if isinstance(relation, dict)
        and str(relation.get("type") or "") in relation_types
        and relation.get("target")
    }


def _metacognitive_report(objects: list[dict[str, Any]]) -> dict[str, Any]:
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


def _load_autoresearch_tasks(path: Path) -> list[dict[str, Any]]:
    """Load a local JSONL/JSON task file without making network requests."""

    if not path.is_file():
        raise ValueError("tasks file does not exist")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError("cannot inspect tasks file") from exc
    if size > _MAX_TASK_FILE_BYTES:
        raise ValueError(
            f"tasks file exceeds the {_MAX_TASK_FILE_BYTES}-byte safety limit"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("cannot read tasks file") from exc
    rows: Any
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
    else:
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON tasks file: {exc.msg}") from exc
        if isinstance(rows, dict):
            rows = rows.get("tasks", rows.get("data", rows))
    if not isinstance(rows, list) or not rows:
        raise ValueError("tasks file must contain a non-empty JSON list or JSONL")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every task row must be a JSON object")
    return rows


def _workspace_stage_report(
    root: Path,
    *,
    task_manifest_sha256: str | None = None,
    task_count: int | None = None,
    task_filter: str = "all",
    task_limit: int | None = None,
) -> dict[str, Any]:
    """Summarize six-stage artifact coverage using validated local evidence."""

    objects: list[dict[str, Any]] = []
    read_errors: list[str] = []
    source_object_count = 0
    objects_truncated = False
    source_kind_counts: Counter[str] = Counter()
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
        ) = _bounded_object_scan(root, _MAX_WORKSPACE_AUDIT_OBJECTS)
        read_errors.extend(object_read_errors)
    except (OSError, ValueError, TypeError) as exc:
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
        path_hits = [path for path in spec["paths"] if (root / path).exists()]
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

    metacognition = _metacognitive_report(objects)
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
    try:
        from .research_closure import audit_research_closure, closure_level_summary
        from .research_git import ResearchGitError

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
    except (OSError, ValueError, KeyError, ResearchGitError):
        closure = {
            "available": False,
            "commit": None,
            "target_level": None,
            "levels": {},
            "blocker_count": 0,
            "warning_count": 0,
        }

    arft_coverage: dict[str, Any]
    legacy_contract_present = any(
        (root / name).exists()
        for name in (
            "pipeline_manifest.json",
            "idea_cards.json",
            "stage_standards.json",
        )
    )
    if not legacy_contract_present:
        typed_objects_present = any(
            path.is_file()
            for path in (root / ".xscientist" / "objects").glob("*/*.json")
        )
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
            from ai_scientist.utils.arft_coverage import build_arft_coverage

            arft_report = build_arft_coverage(root)
            arft_coverage = {
                "schema": arft_report.get("schema"),
                "quality_claim_allowed": False,
                "benchmark_compatible": False,
                "summary": arft_report.get("summary") or {},
                "stages": arft_report.get("stages") or [],
            }
        except (OSError, ValueError, KeyError, TypeError):
            arft_coverage = {
                "schema": "xscientist.arft-coverage.v1",
                "quality_claim_allowed": False,
                "benchmark_compatible": False,
                "summary": {"status": "unavailable"},
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
    except (OSError, ValueError, KeyError, TypeError):
        process = {
            "schema": "xscientist.process-audit.v1",
            "available": False,
            "reason": "process_audit_unavailable",
            "errors": ["process_audit_error"],
        }
    return {
        "workspace": ".",
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
            "statistics_scope": (
                "bounded_object_sample" if objects_truncated else "all_objects"
            ),
        },
        "closure": closure,
        "metacognition": metacognition,
        "metacognitive_signals": signals,
        "arft_coverage": arft_coverage,
        "process": process,
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

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if task_kind not in {"all", "open-ended", "optimization"}:
        raise ValueError("task_kind must be all, open-ended, or optimization")
    task_path = Path(tasks).expanduser().resolve()
    all_rows = _load_autoresearch_tasks(task_path)

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
    try:
        dataset_sha256 = hashlib.sha256(task_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("cannot hash tasks file") from exc
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
    return report


__all__ = ["benchmark_first_run", "benchmark_autoresearch_pilot"]
