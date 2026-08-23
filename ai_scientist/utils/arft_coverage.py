"""Offline observability coverage for the AutoResearchEval/ARFT lens.

This module intentionally does *not* score a model or claim that a trajectory
failed.  AutoResearchEval evaluates complete trajectories with an
artifact-aware judge.  XScientist does not currently retain that benchmark's
task manifest and raw tool traces, so the safe thing we can expose locally is
whether the project's own contracts contain enough signals to investigate
each ARFT failure pattern.

The result is therefore a coverage/observability report, not a scientific
quality score.  ``covered`` means that the relevant local evidence channel is
present, ``partial`` means only some of the channel is present, and
``unassessed`` means a conclusion cannot be drawn from the available
artifacts.  Missing signals are surfaced explicitly so a future trajectory
judge can fill the gap without silently treating absence as success.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ai_scientist.utils.pipeline_contracts import (
    ARTIFACT_FILENAMES,
    artifact_path,
    save_contract_artifact,
)

ARFT_SCHEMA = "xscientist.arft-coverage.v1"
ARFT_REFERENCE = "https://arxiv.org/abs/2608.14905"
ARFT_ROOT_CAUSES = (
    "grounding_faithfulness",
    "cognitive_depth_adaptability",
    "integrity_alignment",
    "engineering_robustness",
)
ARFT_STAGES = ("A", "B", "C", "D", "E", "F", "X")
_MAX_INPUT_BYTES = 8 * 1024 * 1024
_MAX_INPUT_LINES = 8192
_MAX_SIGNAL_DEPTH = 64
_MAX_SIGNAL_NODES = 10000


def _owned_path(project_root: Path, path: Path) -> bool:
    """Return whether *path* stays inside ``project_root`` without symlinks.

    ARFT coverage is an offline, read-only audit.  Both contract loading and
    the summary's existence indicators must refuse symlink components so a
    report cannot accidentally inspect an external file (or mistake one for
    a local artifact).
    """

    try:
        relative = path.relative_to(project_root)
        current = project_root
        for part in relative.parts:
            if current.is_symlink():
                return False
            current = current / part
            if current.is_symlink():
                return False
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _pattern(
    pattern_id: str,
    stage: str,
    name: str,
    root_cause: str,
    description: str,
) -> dict[str, str]:
    return {
        "id": pattern_id,
        "stage": stage,
        "name": name,
        "root_cause": root_cause,
        "description": description,
    }


# The IDs and names are kept aligned with Appendix A of AutoResearchEval.  A
# compact local catalog makes reports stable even when the external paper is
# unavailable, while the reference URL above preserves provenance.
ARFT_PATTERN_CATALOG: tuple[dict[str, str], ...] = (
    _pattern(
        "A.1",
        "A",
        "Frame-Lock",
        "cognitive_depth_adaptability",
        "Narrow hypothesis space without alternatives.",
    ),
    _pattern(
        "A.2",
        "A",
        "Unfalsifiable",
        "integrity_alignment",
        "Hypothesis or experiment cannot disprove the claim.",
    ),
    _pattern(
        "A.3",
        "A",
        "Redundant",
        "cognitive_depth_adaptability",
        "Novelty is not checked against existing work.",
    ),
    _pattern(
        "A.4",
        "A",
        "Feasibility",
        "engineering_robustness",
        "Time, compute, or technical complexity is underestimated.",
    ),
    _pattern(
        "A.5",
        "A",
        "Metric-Misalign",
        "integrity_alignment",
        "Metrics do not represent the stated objective.",
    ),
    _pattern(
        "A.6",
        "A",
        "Hyp-Exp-Mismatch",
        "grounding_faithfulness",
        "Experiments do not test the proposed hypothesis.",
    ),
    _pattern(
        "B.1",
        "B",
        "Hallucinated-Evid",
        "grounding_faithfulness",
        "Evidence or provenance cannot be checked.",
    ),
    _pattern(
        "B.2",
        "B",
        "Retrieval-Gap",
        "grounding_faithfulness",
        "Retrieved knowledge never informs the plan.",
    ),
    _pattern(
        "B.3",
        "B",
        "Unvetted-Data",
        "engineering_robustness",
        "Data quality, units, or source validation is absent.",
    ),
    _pattern(
        "B.4",
        "B",
        "Shallow-Search",
        "cognitive_depth_adaptability",
        "Search stops before critical coverage is established.",
    ),
    _pattern(
        "B.5",
        "B",
        "Citation-Decorr",
        "grounding_faithfulness",
        "Citations are topical but not logically bound to claims.",
    ),
    _pattern(
        "B.6",
        "B",
        "Low-SNR",
        "cognitive_depth_adaptability",
        "Irrelevant evidence crowds out high-signal sources.",
    ),
    _pattern(
        "C.1",
        "C",
        "Circular-Valid",
        "integrity_alignment",
        "Evaluation relies on synthetic outputs or shortcuts.",
    ),
    _pattern(
        "C.2",
        "C",
        "Grader-Fit/Leak",
        "integrity_alignment",
        "Benchmark fitting, leakage, or cherry-picking is unchecked.",
    ),
    _pattern(
        "C.3",
        "C",
        "Impl-Discrep",
        "grounding_faithfulness",
        "Implementation diverges from the claimed method.",
    ),
    _pattern(
        "C.4",
        "C",
        "Exec-Fault",
        "engineering_robustness",
        "Runtime, numerical, or reproducibility faults are unhandled.",
    ),
    _pattern(
        "C.5",
        "C",
        "Infra-Misdiag",
        "engineering_robustness",
        "Infrastructure failures are mistaken for scientific failures.",
    ),
    _pattern(
        "C.6",
        "C",
        "Local-Opt",
        "cognitive_depth_adaptability",
        "Search over-tunes a narrow local neighborhood.",
    ),
    _pattern(
        "C.7",
        "C",
        "Premature-Term",
        "cognitive_depth_adaptability",
        "Execution stops at the first friction or failed attempt.",
    ),
    _pattern(
        "C.8",
        "C",
        "Env-Interact",
        "engineering_robustness",
        "CLI, API, or filesystem interactions are not verified.",
    ),
    _pattern(
        "D.1",
        "D",
        "Artifacts-as-Insight",
        "grounding_faithfulness",
        "Bugs or noise are interpreted as discoveries.",
    ),
    _pattern(
        "D.2",
        "D",
        "Confirmation-Bias",
        "integrity_alignment",
        "Favorable results are kept while counterevidence is ignored.",
    ),
    _pattern(
        "D.3",
        "D",
        "Stat-Misuse",
        "integrity_alignment",
        "Uncertainty or statistical support is missing.",
    ),
    _pattern(
        "D.4",
        "D",
        "Method-Concl-Disc",
        "grounding_faithfulness",
        "Conclusions are disconnected from outputs.",
    ),
    _pattern(
        "D.5",
        "D",
        "Baseline-Deficit",
        "cognitive_depth_adaptability",
        "Strong baselines or ablations are absent.",
    ),
    _pattern(
        "D.6",
        "D",
        "Result-Halluc",
        "grounding_faithfulness",
        "Metrics, tables, or charts lack grounded sources.",
    ),
    _pattern(
        "D.7",
        "D",
        "Unremediated-Adv",
        "integrity_alignment",
        "Known anomalies are not reflected in the conclusion.",
    ),
    _pattern(
        "E.1",
        "E",
        "Report-Trace-Gap",
        "grounding_faithfulness",
        "Narrative claims cannot be traced to execution.",
    ),
    _pattern(
        "E.2",
        "E",
        "Overclaim",
        "integrity_alignment",
        "Negative results are hidden or claims are exaggerated.",
    ),
    _pattern(
        "E.3",
        "E",
        "Omit-Limits",
        "integrity_alignment",
        "Critical limitations are omitted.",
    ),
    _pattern(
        "E.4",
        "E",
        "Method/Cite-Fab",
        "grounding_faithfulness",
        "Methods or citations are fabricated.",
    ),
    _pattern(
        "F.1",
        "F",
        "Superficial-Review",
        "cognitive_depth_adaptability",
        "Review is a passive checklist rather than a critique.",
    ),
    _pattern(
        "F.2",
        "F",
        "Fail-Gate",
        "cognitive_depth_adaptability",
        "Fatal flaws are not blocked by the final gate.",
    ),
    _pattern(
        "F.3",
        "F",
        "No-Adversarial",
        "cognitive_depth_adaptability",
        "No hostile or independent reviewer perspective.",
    ),
    _pattern(
        "F.4",
        "F",
        "Uncorrected-SelfAware",
        "cognitive_depth_adaptability",
        "Known flaws remain in the delivered artifact.",
    ),
    _pattern(
        "F.5",
        "F",
        "Review-Hack",
        "integrity_alignment",
        "Automated review scores are optimized instead of substance.",
    ),
    _pattern(
        "F.6",
        "F",
        "Halluc-Review",
        "grounding_faithfulness",
        "Review invents errors or misdiagnoses correct work.",
    ),
    _pattern(
        "X.1",
        "X",
        "Cascade",
        "engineering_robustness",
        "Early errors compound across later stages.",
    ),
    _pattern(
        "X.2",
        "X",
        "Goal-Drift",
        "integrity_alignment",
        "The trajectory drifts from the original objective.",
    ),
    _pattern(
        "X.3",
        "X",
        "Skeptic-Deficit",
        "cognitive_depth_adaptability",
        "Tool outputs are accepted without challenge.",
    ),
    _pattern(
        "X.4",
        "X",
        "Honest-Hollow",
        "integrity_alignment",
        "Formatting is complete but scientific substance is absent.",
    ),
    _pattern(
        "X.5",
        "X",
        "Teleological",
        "integrity_alignment",
        "Design and analysis are forced toward a preset result.",
    ),
    _pattern(
        "X.6",
        "X",
        "Right-Wrong-Reason",
        "grounding_faithfulness",
        "A good metric comes from an unsound method.",
    ),
    _pattern(
        "X.7",
        "X",
        "Anchoring",
        "cognitive_depth_adaptability",
        "Dead-end path persists without replanning.",
    ),
    _pattern(
        "X.8",
        "X",
        "Eng-Delivery",
        "engineering_robustness",
        "Broken scripts, environments, or output are delivered.",
    ),
)

_PATTERN_BY_ID = {item["id"]: item for item in ARFT_PATTERN_CATALOG}

# Signal requirements are intentionally conservative.  A signal is evidence
# that a failure can be investigated, not evidence that the failure occurred.
_PATTERN_SIGNALS: dict[str, tuple[str, ...]] = {
    "A.1": ("ideation_alternatives",),
    "A.2": ("ideation_falsification",),
    "A.3": ("ideation_literature_positioning",),
    "A.4": ("planning_budget",),
    "A.5": ("ideation_metrics",),
    "A.6": ("planning_hypothesis_binding",),
    "B.1": ("retrieval_provenance",),
    "B.2": ("retrieval_plan_binding",),
    "B.3": ("data_validation",),
    "B.4": ("retrieval_coverage",),
    "B.5": ("citation_binding",),
    "B.6": ("retrieval_prioritization",),
    "C.1": ("execution_independent_check",),
    "C.2": ("execution_leakage_check",),
    "C.3": ("execution_method_binding",),
    "C.4": ("execution_health", "execution_reproducibility"),
    "C.5": ("infrastructure_diagnostics",),
    "C.6": ("execution_search_breadth",),
    "C.7": ("execution_attempt_history",),
    "C.8": ("execution_provenance",),
    "D.1": ("analysis_artifact_trace",),
    "D.2": ("analysis_counterevidence",),
    "D.3": ("analysis_uncertainty",),
    "D.4": ("analysis_claim_binding",),
    "D.5": ("analysis_baselines",),
    "D.6": ("analysis_numeric_grounding",),
    "D.7": ("analysis_remediation",),
    "E.1": ("writing_claim_trace",),
    "E.2": ("writing_negative_results",),
    "E.3": ("writing_limitations",),
    "E.4": ("writing_citations",),
    "F.1": ("review_rounds",),
    "F.2": ("review_gate",),
    "F.3": ("review_adversarial",),
    "F.4": ("review_remediation",),
    "F.5": ("review_independence",),
    "F.6": ("review_evidence_anchors",),
    "X.1": ("cross_stage_lineage",),
    "X.2": ("cross_stage_goal_binding",),
    "X.3": ("cross_stage_skepticism",),
    "X.4": ("cross_stage_substance",),
    "X.5": ("cross_stage_locked_objective",),
    "X.6": ("cross_stage_independent_evidence",),
    "X.7": ("cross_stage_replanning",),
    "X.8": ("cross_stage_delivery",),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _has(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _field(
    value: Any,
    *keys: str,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _nodes: list[int] | None = None,
) -> bool:
    """Return whether any key has a non-empty value in a shallow object tree."""

    if _depth > _MAX_SIGNAL_DEPTH:
        return False
    if _seen is None:
        _seen = set()
    if _nodes is None:
        _nodes = [0]
    _nodes[0] += 1
    if _nodes[0] > _MAX_SIGNAL_NODES:
        return False
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _seen:
            return False
        _seen.add(marker)
        for key in keys:
            if _has(value.get(key)):
                _seen.discard(marker)
                return True
        result = any(
            _field(
                item,
                *keys,
                _depth=_depth + 1,
                _seen=_seen,
                _nodes=_nodes,
            )
            for item in value.values()
        )
        _seen.discard(marker)
        return result
    if isinstance(value, list):
        marker = id(value)
        if marker in _seen:
            return False
        _seen.add(marker)
        result = any(
            _field(
                item,
                *keys,
                _depth=_depth + 1,
                _seen=_seen,
                _nodes=_nodes,
            )
            for item in value
        )
        _seen.discard(marker)
        return result
    return False


def _contains(
    value: Any,
    needles: tuple[str, ...],
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _nodes: list[int] | None = None,
) -> bool:
    if _depth > _MAX_SIGNAL_DEPTH:
        return False
    if _seen is None:
        _seen = set()
    if _nodes is None:
        _nodes = [0]
    _nodes[0] += 1
    if _nodes[0] > _MAX_SIGNAL_NODES:
        return False
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _seen:
            return False
        _seen.add(marker)
        for key, item in value.items():
            key_text = str(key).lower()
            if any(needle in key_text for needle in needles) and _has(item):
                _seen.discard(marker)
                return True
            if _contains(
                item,
                needles,
                _depth=_depth + 1,
                _seen=_seen,
                _nodes=_nodes,
            ):
                _seen.discard(marker)
                return True
        _seen.discard(marker)
    elif isinstance(value, list):
        marker = id(value)
        if marker in _seen:
            return False
        _seen.add(marker)
        result = any(
            _contains(
                item,
                needles,
                _depth=_depth + 1,
                _seen=_seen,
                _nodes=_nodes,
            )
            for item in value
        )
        _seen.discard(marker)
        return result
    elif isinstance(value, str):
        lowered = value.lower()
        return any(needle in lowered for needle in needles)
    return False


def _load_inputs(project_root: Path) -> dict[str, Any]:
    input_errors: list[dict[str, Any]] = []

    def safe_input_error(exc: BaseException) -> str:
        if isinstance(exc, UnicodeError):
            return "decode_error"
        if isinstance(exc, RecursionError):
            return "nesting_limit"
        if isinstance(exc, MemoryError):
            return "memory_limit"
        if isinstance(exc, OSError):
            return "read_error"
        if isinstance(exc, ValueError):
            token = str(exc)
            if token in {"input_size_limit", "line_limit", "row_not_object"}:
                return token
            if token.startswith("non-finite JSON constant"):
                return "non_finite_json"
            return "invalid_input"
        return "read_error"

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant: {token}")

    def contract(name: str, default: Any) -> Any:
        path = artifact_path(project_root, name)
        if not _owned_path(project_root, path):
            input_errors.append({"artifact": name, "error": "symlink_boundary"})
            return deepcopy(default)
        if not path.exists():
            return deepcopy(default)
        try:
            size = path.stat().st_size
            if size > _MAX_INPUT_BYTES:
                raise ValueError("input_size_limit")
            raw = path.read_bytes()
            if path.suffix == ".json":
                return json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
            if path.suffix == ".jsonl":
                rows: list[dict[str, Any]] = []
                for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                    if line_number > _MAX_INPUT_LINES:
                        input_errors.append({"artifact": name, "error": "line_limit"})
                        break
                    if not line.strip():
                        continue
                    value = json.loads(line, parse_constant=reject_constant)
                    if not isinstance(value, dict):
                        raise ValueError("row_not_object")
                    rows.append(value)
                return rows
            return raw.decode("utf-8")
        except json.JSONDecodeError as exc:
            input_errors.append(
                {
                    "artifact": name,
                    "error": "invalid_json",
                    "line": exc.lineno,
                    "column": exc.colno,
                }
            )
        except (OSError, UnicodeError, ValueError, RecursionError, MemoryError) as exc:
            input_errors.append({"artifact": name, "error": safe_input_error(exc)})
        return deepcopy(default)

    experiments = contract("experiment_registry", [])
    experiment_errors = [
        item for item in input_errors if item.get("artifact") == "experiment_registry"
    ]
    return {
        # Coverage computation is read-only.  ``load_pipeline_manifest``
        # bootstraps a file when absent, which would make an audit have a
        # surprising side effect on an empty workspace.
        "manifest": contract("pipeline_manifest", {}),
        "idea_cards": contract("idea_cards", []),
        "hypothesis_archive": contract("hypothesis_archive", {}),
        "research_plan": contract("research_plan", {}),
        "claim_graph": contract("claim_evidence_graph", {}),
        "experiments": experiments,
        "experiment_errors": experiment_errors,
        "figure_spec": contract("figure_spec", {}),
        "manuscript": contract("manuscript_state", {}),
        "review": contract("review_state", {}),
        "critic_findings": contract("critic_findings", {}),
        "repair_plan": contract("repair_plan", {}),
        "self_evolution": contract("self_evolution", {}),
        "preregistration": contract("preregistration", {}),
        "verification_report": contract("verification_report", {}),
        "evaluation_charter": contract("evaluation_charter", {}),
        "evaluation_report": contract("evaluation_report", {}),
        "decision_log": contract("decision_log", []),
        "input_errors": input_errors,
    }


def _collect_signals(
    data: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    ideas = _rows(data.get("idea_cards"))
    lead = ideas[0] if ideas else {}
    plan = _as_mapping(data.get("research_plan"))
    graph = _as_mapping(data.get("claim_graph"))
    graph_nodes = _rows(graph.get("nodes"))
    graph_edges = _rows(graph.get("edges"))
    experiments = _rows(data.get("experiments"))
    manuscript = _as_mapping(data.get("manuscript"))
    review = _as_mapping(data.get("review"))
    repair = _as_mapping(data.get("repair_plan"))
    prereg = _as_mapping(data.get("preregistration"))
    charter = _as_mapping(data.get("evaluation_charter"))
    evaluation = _as_mapping(data.get("evaluation_report"))
    manifest = _as_mapping(data.get("manifest"))
    archive = _as_mapping(data.get("hypothesis_archive"))
    decision_log = _rows(data.get("decision_log"))
    tasks = _rows(plan.get("tasks"))
    review_adversarial_signal = _field(
        review,
        "adversarial",
        "skeptic",
        "red_team",
        "hostile",
        "independent_review",
    ) or any(
        _contains(item, ("adversarial", "skeptic", "red_team"))
        for item in _rows(review.get("rounds"))
    )

    signals: dict[str, bool] = {
        # A — ideation/planning
        "ideation_alternatives": _field(
            lead, "alternative_framing", "alternatives", "approaches_considered"
        )
        or len(ideas) > 1,
        "ideation_falsification": _field(
            lead, "failure_criteria", "falsification_check", "falsifiable_test"
        ),
        "ideation_literature_positioning": _field(
            lead, "novelty_claim", "related_work_notes", "literature_queries"
        )
        or _field(archive, "novelty", "related_work", "literature"),
        "planning_budget": _field(
            plan, "budget", "max_steps", "max_wallclock_minutes", "max_retry_per_task"
        ),
        "ideation_metrics": _field(lead, "candidate_metrics", "metric", "metrics"),
        "planning_hypothesis_binding": bool(tasks)
        and all(
            _field(
                task,
                "claim_targets",
                "hypothesis_id",
                "hypothesis",
                "success_criterion",
            )
            for task in tasks
        ),
        # B — retrieval/synthesis
        "retrieval_provenance": _field(
            lead, "sources", "source_ids", "literature_sources", "retrieval_receipt"
        )
        or _field(graph_nodes, "source", "doi", "arxiv", "citation", "provenance"),
        "retrieval_plan_binding": _field(
            plan, "literature_bindings", "source_bindings", "retrieval_to_action"
        )
        or any(_field(task, "sources", "literature", "evidence_ids") for task in tasks),
        "data_validation": _field(
            plan, "data_validation", "unit_check", "schema_check", "quality_check"
        )
        or any(
            _field(
                row, "data_validation", "unit_check", "schema_check", "quality_check"
            )
            for row in experiments
        ),
        "retrieval_coverage": _field(
            lead, "literature_queries", "search_plan", "search_coverage"
        )
        or _field(plan, "search_plan", "search_coverage"),
        "citation_binding": _field(
            graph_edges, "citation", "source", "evidence_id", "supports"
        )
        or _field(manuscript, "citation_bindings", "claim_citations", "references"),
        "retrieval_prioritization": _field(
            lead, "source_priorities", "priority_sources", "signal_rank"
        )
        or _field(plan, "source_priorities", "priority_sources"),
        # C — execution/implementation
        "execution_independent_check": _field(
            plan,
            "holdout",
            "sealed",
            "independent",
            "external_evaluation",
            "prospective",
        )
        or any(
            _field(
                row,
                "holdout",
                "sealed",
                "independent",
                "external_evaluation",
                "prospective",
            )
            for row in experiments
        ),
        "execution_leakage_check": _field(
            plan, "leakage_check", "data_split", "test_isolation", "anti_leakage"
        )
        or any(
            _field(row, "leakage_check", "data_split", "test_isolation", "anti_leakage")
            for row in experiments
        ),
        "execution_method_binding": _field(
            plan, "method", "implementation", "code_ref", "method_spec"
        )
        or any(
            _field(row, "method", "method_ref", "code_ref", "implementation")
            for row in experiments
        ),
        "execution_health": bool(experiments)
        and not bool(data.get("experiment_errors")),
        "execution_reproducibility": any(
            _field(row, "seed", "random_seed", "environment", "dependencies")
            for row in experiments
        ),
        "infrastructure_diagnostics": any(
            _field(row, "issues", "error", "diagnostic", "environment")
            for row in experiments
        )
        or bool(data.get("experiment_errors")),
        "execution_search_breadth": len(experiments) > 1
        or len(tasks) > 1
        or _field(plan, "alternatives", "ablation_grid", "search_space"),
        "execution_attempt_history": bool(experiments)
        and (
            len(experiments) > 1
            or any(
                _field(row, "attempt", "retry", "parent_id", "status")
                for row in experiments
            )
        ),
        "execution_provenance": any(
            _field(row, "provenance", "tool", "command", "artifact", "run_id")
            for row in experiments
        ),
        # D — analysis/interpretation
        "analysis_artifact_trace": any(
            _field(row, "result_summary", "result", "output", "artifact", "metric")
            for row in experiments
        ),
        "analysis_counterevidence": any(
            _field(
                row,
                "negative",
                "failed",
                "counterevidence",
                "contradiction",
                "sanity_check",
            )
            for row in experiments
        )
        or _field(
            manuscript, "negative_results", "counterevidence", "failed_iterations"
        ),
        "analysis_uncertainty": any(
            _field(
                row,
                "uncertainty",
                "confidence_interval",
                "std",
                "variance",
                "error_bar",
                "p_value",
            )
            for row in experiments
        )
        or _field(manuscript, "uncertainty", "confidence_interval", "statistics"),
        "analysis_claim_binding": bool(graph_nodes)
        and bool(graph_edges)
        and any(
            str(node.get("type") or "") in {"claim", "hypothesis"}
            for node in graph_nodes
        ),
        "analysis_baselines": any(
            _field(row, "baseline", "baseline_ref", "ablation") for row in experiments
        )
        or _field(plan, "candidate_baselines", "baselines", "ablations"),
        "analysis_numeric_grounding": any(
            _field(row, "metric", "value", "result_summary", "figure", "table")
            for row in experiments
        ),
        "analysis_remediation": bool(_rows(review.get("active_issue_records")))
        or bool(_rows(repair.get("tasks")))
        or _field(review, "resolved_issues", "verification_checks"),
        # E — writing/documentation
        "writing_claim_trace": _field(
            manuscript, "claim_bindings", "claim_evidence", "evidence_bindings"
        )
        and bool(graph_nodes),
        "writing_negative_results": _field(
            manuscript, "negative_results", "failed_iterations", "null_results"
        )
        or any(
            _field(row, "negative", "failed", "status")
            and str(row.get("status") or "") not in {"completed", "success"}
            for row in experiments
        ),
        "writing_limitations": _field(
            manuscript, "limitations", "missing_evidence", "scope_limits"
        ),
        "writing_citations": _field(
            manuscript, "references", "citations", "citation_bindings"
        )
        or _field(lead, "related_work_notes", "literature_queries"),
        # F — self-verification/review
        "review_rounds": bool(_rows(review.get("rounds")))
        or _field(review, "review_count", "reviewed_at"),
        "review_gate": _field(
            review, "gate", "decision", "guardrail_status", "claim_promotion_allowed"
        )
        or bool(data.get("evaluation_report")),
        "review_adversarial": review_adversarial_signal,
        "review_remediation": _field(
            review, "repair_actions", "resolved_issues", "verification_checks"
        )
        or bool(_rows(repair.get("tasks"))),
        "review_independence": _field(
            review, "independent", "evaluator_id", "independence", "external"
        )
        or _field(evaluation, "independent", "evaluator_id", "charter_hash"),
        "review_evidence_anchors": _field(
            review,
            "evidence_anchors",
            "artifact_refs",
            "issue_bindings",
            "finding_bindings",
        ),
        # X — cross-stage dynamics
        "cross_stage_lineage": bool(graph_edges)
        and bool(experiments)
        and bool(_field(manuscript, "claim_bindings", "claim_evidence")),
        "cross_stage_goal_binding": _has(manifest.get("pipeline_goal"))
        and bool(ideas)
        and bool(plan),
        "cross_stage_skepticism": _field(
            plan, "socratic_challenge", "rival_hypotheses", "alternative_hypotheses"
        )
        or review_adversarial_signal,
        "cross_stage_substance": bool(experiments)
        and bool(graph_nodes)
        and _field(manuscript, "claim_bindings", "results", "sections"),
        "cross_stage_locked_objective": prereg.get("status") == "locked"
        or _field(charter, "charter_hash", "locked", "sealed"),
        "cross_stage_independent_evidence": bool(
            evaluation.get("claim_promotion_allowed") is True
        )
        or _field(evaluation, "independence", "evaluator_id")
        or any(
            _field(row, "independent", "verifier_id", "independence")
            for row in experiments
        ),
        "cross_stage_replanning": len(decision_log) > 1
        or _field(plan, "replan", "pivot", "alternatives", "fallback_strategy")
        or _field(archive, "supersedes", "revisions", "branches"),
        "cross_stage_delivery": _has(manifest.get("artifacts"))
        and bool(experiments)
        and bool(manuscript),
    }

    sources: dict[str, list[str]] = {
        "ideation_alternatives": ["idea_cards"],
        "ideation_falsification": ["idea_cards"],
        "ideation_literature_positioning": ["idea_cards", "hypothesis_archive"],
        "planning_budget": ["research_plan"],
        "ideation_metrics": ["idea_cards"],
        "planning_hypothesis_binding": ["research_plan"],
        "retrieval_provenance": ["idea_cards", "claim_evidence_graph"],
        "retrieval_plan_binding": ["research_plan", "claim_evidence_graph"],
        "data_validation": ["research_plan", "experiment_registry"],
        "retrieval_coverage": ["idea_cards", "research_plan"],
        "citation_binding": ["claim_evidence_graph", "manuscript_state"],
        "retrieval_prioritization": ["idea_cards", "research_plan"],
        "execution_independent_check": ["research_plan", "experiment_registry"],
        "execution_leakage_check": ["research_plan", "experiment_registry"],
        "execution_method_binding": ["research_plan", "experiment_registry"],
        "execution_health": ["experiment_registry"],
        "execution_reproducibility": ["experiment_registry"],
        "infrastructure_diagnostics": ["experiment_registry"],
        "execution_search_breadth": ["research_plan", "experiment_registry"],
        "execution_attempt_history": ["experiment_registry"],
        "execution_provenance": ["experiment_registry"],
        "analysis_artifact_trace": ["experiment_registry"],
        "analysis_counterevidence": ["experiment_registry", "manuscript_state"],
        "analysis_uncertainty": ["experiment_registry", "manuscript_state"],
        "analysis_claim_binding": ["claim_evidence_graph"],
        "analysis_baselines": ["research_plan", "experiment_registry"],
        "analysis_numeric_grounding": ["experiment_registry"],
        "analysis_remediation": ["review_state", "repair_plan"],
        "writing_claim_trace": ["manuscript_state", "claim_evidence_graph"],
        "writing_negative_results": ["manuscript_state", "experiment_registry"],
        "writing_limitations": ["manuscript_state"],
        "writing_citations": ["manuscript_state", "idea_cards"],
        "review_rounds": ["review_state"],
        "review_gate": ["review_state", "evaluation_report"],
        "review_adversarial": ["review_state"],
        "review_remediation": ["review_state", "repair_plan"],
        "review_independence": ["review_state", "evaluation_report"],
        "review_evidence_anchors": ["review_state"],
        "cross_stage_lineage": [
            "claim_evidence_graph",
            "experiment_registry",
            "manuscript_state",
        ],
        "cross_stage_goal_binding": [
            "pipeline_manifest",
            "idea_cards",
            "research_plan",
        ],
        "cross_stage_skepticism": ["research_plan", "review_state"],
        "cross_stage_substance": [
            "experiment_registry",
            "claim_evidence_graph",
            "manuscript_state",
        ],
        "cross_stage_locked_objective": ["preregistration", "evaluation_charter"],
        "cross_stage_independent_evidence": [
            "evaluation_report",
            "experiment_registry",
        ],
        "cross_stage_replanning": [
            "decision_log",
            "research_plan",
            "hypothesis_archive",
        ],
        "cross_stage_delivery": [
            "pipeline_manifest",
            "experiment_registry",
            "manuscript_state",
        ],
    }
    return signals, sources


def _assess_pattern(
    pattern: Mapping[str, str],
    signals: Mapping[str, bool],
    sources: Mapping[str, list[str]],
) -> dict[str, Any]:
    required = list(_PATTERN_SIGNALS.get(str(pattern["id"]), ()))
    present = [name for name in required if signals.get(name) is True]
    missing = [name for name in required if name not in present]
    ratio = len(present) / max(len(required), 1)
    if not present:
        status = "unassessed"
    elif missing:
        status = "partial"
    else:
        status = "covered"
    evidence = sorted({source for name in present for source in sources.get(name, [])})
    return {
        **dict(pattern),
        "status": status,
        "coverage_score": round(ratio * 100.0, 1),
        "required_signals": required,
        "present_signals": present,
        "missing_signals": missing,
        "evidence_channels": evidence,
        "interpretation": (
            "local artifacts expose the channel needed for targeted review"
            if status == "covered"
            else (
                "some artifact signals exist; a trajectory judge still needs the missing channels"
                if status == "partial"
                else "no local artifact signal is available; do not infer success or failure"
            )
        ),
    }


def build_arft_coverage(project_root: str | Path) -> dict[str, Any]:
    """Build an offline ARFT observability report for one project directory."""

    root = Path(project_root).expanduser().resolve()
    data = _load_inputs(root)
    signals, sources = _collect_signals(data)
    patterns = [
        _assess_pattern(item, signals, sources) for item in ARFT_PATTERN_CATALOG
    ]

    stage_results: list[dict[str, Any]] = []
    for stage in ARFT_STAGES:
        rows = [item for item in patterns if item["stage"] == stage]
        scores = [float(item["coverage_score"]) for item in rows]
        counts = Counter(item["status"] for item in rows)
        stage_results.append(
            {
                "stage": stage,
                "label": {
                    "A": "Ideation & Planning",
                    "B": "Retrieval & Synthesis",
                    "C": "Execution & Implementation",
                    "D": "Analysis & Interpretation",
                    "E": "Writing & Documentation",
                    "F": "Self-Verification & Review",
                    "X": "Cross-Stage Dynamics",
                }[stage],
                "pattern_count": len(rows),
                "covered_pattern_count": counts.get("covered", 0),
                "partial_pattern_count": counts.get("partial", 0),
                "unassessed_pattern_count": counts.get("unassessed", 0),
                "coverage_score": round(sum(scores) / max(len(scores), 1), 1),
                "patterns": [item["id"] for item in rows],
            }
        )

    root_results: list[dict[str, Any]] = []
    for root_cause in ARFT_ROOT_CAUSES:
        rows = [item for item in patterns if item["root_cause"] == root_cause]
        root_results.append(
            {
                "root_cause": root_cause,
                "pattern_count": len(rows),
                "covered_pattern_count": sum(
                    item["status"] == "covered" for item in rows
                ),
                "partial_pattern_count": sum(
                    item["status"] == "partial" for item in rows
                ),
                "unassessed_pattern_count": sum(
                    item["status"] == "unassessed" for item in rows
                ),
                "coverage_score": round(
                    sum(float(item["coverage_score"]) for item in rows)
                    / max(len(rows), 1),
                    1,
                ),
            }
        )

    coverage_score = round(
        sum(float(item["coverage_score"]) for item in patterns) / max(len(patterns), 1),
        1,
    )
    missing_channels = sorted(name for name, present in signals.items() if not present)
    return {
        "schema": ARFT_SCHEMA,
        "generated_at": _now_iso(),
        "project_root": ".",
        "reference": ARFT_REFERENCE,
        "evaluation_scope": "structural_observability_only",
        "quality_claim_allowed": False,
        "benchmark_compatible": False,
        "disclaimer": (
            "This report measures whether local artifacts expose evidence channels "
            "for ARFT review. It is not an AutoResearchEval trajectory score and "
            "does not establish that any failure pattern occurred."
        ),
        "summary": {
            "pattern_count": len(patterns),
            "covered_pattern_count": sum(
                item["status"] == "covered" for item in patterns
            ),
            "partial_pattern_count": sum(
                item["status"] == "partial" for item in patterns
            ),
            "unassessed_pattern_count": sum(
                item["status"] == "unassessed" for item in patterns
            ),
            "coverage_score": coverage_score,
            "missing_signal_count": len(missing_channels),
            "missing_signal_names": missing_channels,
        },
        "stages": stage_results,
        "root_causes": root_results,
        "patterns": patterns,
        "signals": {name: bool(value) for name, value in sorted(signals.items())},
        "input_errors": list(data.get("input_errors") or []),
        "artifact_channels": {
            name: {
                "registered": name in ARTIFACT_FILENAMES,
                "exists": (
                    _owned_path(root, artifact_path(root, name))
                    and artifact_path(root, name).exists()
                    if name in ARTIFACT_FILENAMES
                    else False
                ),
            }
            for name in sorted(
                {channel for values in sources.values() for channel in values}
            )
        },
    }


def save_arft_coverage(project_root: str | Path) -> str:
    """Persist an ARFT coverage report as a normal pipeline contract artifact."""

    payload = build_arft_coverage(project_root)
    return save_contract_artifact(
        project_root,
        "arft_coverage",
        payload,
        producer="arft_coverage",
        depends_on=[
            "pipeline_manifest",
            "idea_cards",
            "hypothesis_archive",
            "research_plan",
            "claim_evidence_graph",
            "experiment_registry",
            "manuscript_state",
            "review_state",
            "repair_plan",
            "evaluation_report",
        ],
        notes="Offline ARFT observability only; not a model-quality benchmark.",
    )


__all__ = [
    "ARFT_PATTERN_CATALOG",
    "ARFT_REFERENCE",
    "ARFT_ROOT_CAUSES",
    "ARFT_SCHEMA",
    "ARFT_STAGES",
    "build_arft_coverage",
    "save_arft_coverage",
]
