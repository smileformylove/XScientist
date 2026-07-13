from __future__ import annotations

"""Truth-contract helpers for content-grounded planning and review."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import (
    load_contract_artifact,
    save_contract_artifact,
)


TRUTH_CONTRACT_CATEGORIES: tuple[str, ...] = (
    "objective_facts",
    "physical_constraints",
    "product_geometry_rules",
    "motion_coherence_rules",
    "values_guardrails",
)

CHECK_SEVERITIES: tuple[str, ...] = ("blocker", "warning")
HIGH_RISK_WORKFLOW_MODES: tuple[str, ...] = (
    "review_board",
    "multi_agent_board",
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _is_high_risk_task(task: dict[str, Any], *, workflow_mode: str) -> bool:
    return (
        workflow_mode in HIGH_RISK_WORKFLOW_MODES
        or str(task.get("priority") or "").strip().upper() == "P0"
        or str(task.get("escalation_lane") or "").strip() == "hostile_critic"
    )


def _contract_entry(
    *,
    requirement: str,
    source: str,
    applies_to: list[str],
) -> dict[str, Any]:
    return {
        "requirement": str(requirement or "").strip(),
        "source": str(source or "").strip(),
        "applies_to": _dedupe(applies_to),
    }


def _safe_contract_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for entry in entries:
        requirement = str(entry.get("requirement") or "").strip()
        source = str(entry.get("source") or "").strip()
        if not requirement or not source:
            continue
        safe.append(
            _contract_entry(
                requirement=requirement,
                source=source,
                applies_to=_coerce_list(entry.get("applies_to")),
            )
        )
    return safe


def build_truth_contract(
    idea_card: dict[str, Any],
    research_plan: dict[str, Any],
) -> dict[str, Any]:
    """Build the five-category truth contract for a research plan.

    Research runs do not have VPB-style product geometry or scene physics, so
    each non-objective category keeps one domain-level fallback rule and adds
    task-specific rules where the plan exposes concrete datasets, artifacts, or
    state transitions.
    """

    tasks = [item for item in (research_plan.get("tasks") or []) if isinstance(item, dict)]
    task_ids = [
        str(task.get("task_id") or f"task_{idx}").strip()
        for idx, task in enumerate(tasks)
    ]
    task_ids = [item for item in task_ids if item]
    all_tasks = task_ids or ["task_0"]

    objective_entries: list[dict[str, Any]] = []
    source_idea = idea_card.get("source_idea") if isinstance(idea_card.get("source_idea"), dict) else {}
    hypothesis = str(
        idea_card.get("core_hypothesis")
        or source_idea.get("Short Hypothesis")
        or source_idea.get("Title")
        or idea_card.get("name")
        or ""
    ).strip()
    if hypothesis:
        objective_entries.append(
            _contract_entry(
                requirement=(
                    "Do not present the hypothesis as proven before task evidence "
                    f"supports it: {hypothesis}"
                ),
                source="idea_card.core_hypothesis",
                applies_to=all_tasks,
            )
        )
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            continue
        for field_name, source_name in (
            ("dataset", "research_plan.task.dataset"),
            ("metric", "research_plan.task.metric"),
            ("baseline", "research_plan.task.baseline"),
        ):
            value = str(task.get(field_name) or "").strip()
            if value:
                objective_entries.append(
                    _contract_entry(
                        requirement=(
                            f"Use the declared {field_name} when discussing "
                            f"{task_id}: {value}"
                        ),
                        source=source_name,
                        applies_to=[task_id],
                    )
                )

    physical_entries = [
        _contract_entry(
            requirement=(
                "Do not claim a baseline-comparable improvement unless the experiment "
                "record contains a completed run for the same dataset, metric, and baseline."
            ),
            source="research_plan.acceptance_rules",
            applies_to=all_tasks,
        )
    ]
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        dataset = str(task.get("dataset") or "").strip()
        metric = str(task.get("metric") or "").strip()
        baseline = str(task.get("baseline") or "").strip()
        if task_id and dataset and metric and baseline:
            physical_entries.append(
                _contract_entry(
                    requirement=(
                        f"{task_id} can only report comparable improvements on "
                        f"dataset={dataset}, metric={metric}, baseline={baseline}."
                    ),
                    source="research_plan.task.comparability_fields",
                    applies_to=[task_id],
                )
            )
    geometry_entries = [
        _contract_entry(
            requirement=(
                "Every claim-facing output must remain bound to its declared claim id, "
                "figure candidate, and evidence artifact instead of inventing untracked outputs."
            ),
            source="research_plan.produced_artifacts",
            applies_to=all_tasks,
        )
    ]
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        claim_targets = _coerce_list(task.get("claim_targets"))
        produced_artifacts = _coerce_list(task.get("produced_artifacts"))
        if task_id and (claim_targets or produced_artifacts):
            geometry_entries.append(
                _contract_entry(
                    requirement=(
                        f"{task_id} outputs must bind claims {claim_targets or ['unspecified']} "
                        f"to produced artifacts {produced_artifacts or ['unspecified']}."
                    ),
                    source="research_plan.task.claim_targets_and_artifacts",
                    applies_to=[task_id],
                )
            )
    motion_entries = [
        _contract_entry(
            requirement=(
                "Branch status must move only through planned, running, completed, failed, "
                "or discarded states, and keep/discard/crash decisions must preserve their reason."
            ),
            source="research_plan.branch_outcome_labels",
            applies_to=all_tasks,
        )
    ]
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        dependencies = _coerce_list(task.get("dependencies"))
        branch_labels = _coerce_list(task.get("branch_outcome_labels"))
        if task_id and (dependencies or branch_labels):
            motion_entries.append(
                _contract_entry(
                    requirement=(
                        f"{task_id} must respect dependencies {dependencies or ['none']} "
                        f"and use only declared branch outcomes {branch_labels or ['none']}."
                    ),
                    source="research_plan.task.dependencies_and_branch_outcomes",
                    applies_to=[task_id],
                )
            )
    values_entries = [
        _contract_entry(
            requirement=(
                "Do not overstate unsupported claims; unresolved hostile-critic or reviewer "
                "blockers must trigger repair or block readiness."
            ),
            source="research_plan.execution_policy.acceptance_rules",
            applies_to=all_tasks,
        )
    ]
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        kill_criteria = _coerce_list(task.get("kill_criteria"))
        acceptance_checks = _coerce_list(task.get("acceptance_checks"))
        if task_id and (kill_criteria or acceptance_checks):
            values_entries.append(
                _contract_entry(
                    requirement=(
                        f"{task_id} must not be presented as successful unless acceptance "
                        f"checks pass and kill criteria remain false."
                    ),
                    source="research_plan.task.acceptance_checks_and_kill_criteria",
                    applies_to=[task_id],
                )
            )

    categories = {
        "objective_facts": _safe_contract_entries(objective_entries),
        "physical_constraints": _safe_contract_entries(physical_entries),
        "product_geometry_rules": _safe_contract_entries(geometry_entries),
        "motion_coherence_rules": _safe_contract_entries(motion_entries),
        "values_guardrails": _safe_contract_entries(values_entries),
    }
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "workflow_mode": research_plan.get("workflow_mode"),
        "idea_id": research_plan.get("idea_id") or idea_card.get("idea_id"),
        "categories": categories,
        "derivation_policy": (
            "Research-domain truth contracts combine domain-level fallback rules "
            "with task-derived comparability, artifact-binding, transition, and "
            "acceptance rules when those fields exist in the research plan."
        ),
        "summary": {
            "category_count": len(TRUTH_CONTRACT_CATEGORIES),
            "entry_count": sum(
                len(categories.get(category) or [])
                for category in TRUTH_CONTRACT_CATEGORIES
            ),
        },
    }


def validate_truth_contract(contract: dict[str, Any] | None) -> dict[str, Any]:
    payload = contract if isinstance(contract, dict) else {}
    categories = payload.get("categories") if isinstance(payload.get("categories"), dict) else {}
    errors: list[str] = []
    for category in TRUTH_CONTRACT_CATEGORIES:
        entries = categories.get(category)
        if not isinstance(entries, list) or not entries:
            errors.append(f"missing_category:{category}")
            continue
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"invalid_entry:{category}[{idx}]")
                continue
            if not str(entry.get("requirement") or "").strip():
                errors.append(f"missing_requirement:{category}[{idx}]")
            if not str(entry.get("source") or "").strip():
                errors.append(f"missing_source:{category}[{idx}]")
    return {"passed": not errors, "errors": errors}


def _checks_from_contract_entry(
    *,
    task_id: str,
    category: str,
    entry: dict[str, Any],
    index: int,
) -> list[dict[str, Any]]:
    applies_to = _coerce_list(entry.get("applies_to"))
    if applies_to and task_id not in applies_to and "*" not in applies_to:
        return []
    severity = "blocker" if category in {"objective_facts", "values_guardrails"} else "warning"
    return [
        {
            "check_id": f"{task_id}:{category}:{index}",
            "task_id": task_id,
            "category": category,
            "requirement": str(entry.get("requirement") or "").strip(),
            "prohibited_failure": (
                "Artifact contradicts this requirement, omits its evidence, or "
                "asserts a stronger unsupported version."
            ),
            "severity": severity,
            "evidence_source": str(entry.get("source") or "").strip(),
        }
    ]


def derive_hallucination_checks(
    truth_contract: dict[str, Any],
    research_plan: dict[str, Any],
) -> dict[str, Any]:
    """Derive per-task hallucination checks from a truth contract."""

    categories = (
        truth_contract.get("categories")
        if isinstance(truth_contract.get("categories"), dict)
        else {}
    )
    tasks = [item for item in (research_plan.get("tasks") or []) if isinstance(item, dict)]
    workflow_mode = str(research_plan.get("workflow_mode") or "").strip()
    check_rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_id = str(task.get("task_id") or f"task_{idx}").strip()
        if not task_id:
            continue
        if not _is_high_risk_task(task, workflow_mode=workflow_mode):
            continue
        for category in TRUTH_CONTRACT_CATEGORIES:
            for entry_idx, entry in enumerate(categories.get(category) or []):
                if not isinstance(entry, dict):
                    continue
                check_rows.extend(
                    _checks_from_contract_entry(
                        task_id=task_id,
                        category=category,
                        entry=entry,
                        index=entry_idx,
                    )
                )
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "workflow_mode": workflow_mode,
        "checks": check_rows,
        "summary": {
            "check_count": len(check_rows),
            "blocker_count": sum(1 for item in check_rows if item.get("severity") == "blocker"),
            "warning_count": sum(1 for item in check_rows if item.get("severity") == "warning"),
        },
    }


def validate_hallucination_checks(
    payload: dict[str, Any] | None,
    *,
    research_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = payload.get("checks") if isinstance(payload, dict) else None
    errors: list[str] = []
    if not isinstance(checks, list) or not checks:
        errors.append("missing_checks")
        return {"passed": False, "errors": errors}
    required_fields = (
        "category",
        "requirement",
        "prohibited_failure",
        "severity",
        "evidence_source",
    )
    for idx, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"invalid_check:{idx}")
            continue
        for field in required_fields:
            if not str(check.get(field) or "").strip():
                errors.append(f"missing_{field}:{idx}")
        category = str(check.get("category") or "").strip()
        if category not in TRUTH_CONTRACT_CATEGORIES:
            errors.append(f"invalid_category:{idx}:{category}")
        severity = str(check.get("severity") or "").strip()
        if severity not in CHECK_SEVERITIES:
            errors.append(f"invalid_severity:{idx}:{severity}")
    if isinstance(research_plan, dict):
        workflow_mode = str(research_plan.get("workflow_mode") or "").strip()
        categories_by_task: dict[str, set[str]] = {}
        for check in checks:
            if not isinstance(check, dict):
                continue
            task_id = str(check.get("task_id") or "").strip()
            category = str(check.get("category") or "").strip()
            if task_id and category:
                categories_by_task.setdefault(task_id, set()).add(category)
        for idx, task in enumerate(research_plan.get("tasks") or []):
            if not isinstance(task, dict):
                continue
            if not _is_high_risk_task(task, workflow_mode=workflow_mode):
                continue
            task_id = str(task.get("task_id") or f"task_{idx}").strip()
            missing = [
                category
                for category in TRUTH_CONTRACT_CATEGORIES
                if category not in categories_by_task.get(task_id, set())
            ]
            for category in missing:
                errors.append(f"missing_task_category:{task_id}:{category}")
    return {"passed": not errors, "errors": errors}


def build_hallucination_review(
    checks_payload: dict[str, Any],
    *,
    decisions: list[dict[str, Any]] | None = None,
    keyframe_paths: dict[str, Any] | None = None,
    reviewer: str = "hallucination_review_agent",
) -> dict[str, Any]:
    """Build the review shell that later reviewer agents fill with decisions."""

    checks = [
        item for item in (checks_payload.get("checks") or [])
        if isinstance(item, dict)
    ]
    keyed_decisions = {
        str(item.get("check_id") or "").strip(): dict(item)
        for item in (decisions or [])
        if isinstance(item, dict) and str(item.get("check_id") or "").strip()
    }
    review_rows: list[dict[str, Any]] = []
    for check in checks:
        check_id = str(check.get("check_id") or "").strip()
        decision = keyed_decisions.get(check_id, {})
        review_rows.append(
            {
                "check_id": check_id,
                "task_id": check.get("task_id"),
                "category": check.get("category"),
                "severity": check.get("severity"),
                "passed": decision.get("passed"),
                "finding": str(decision.get("finding") or "").strip(),
                "evidence": str(decision.get("evidence") or "").strip(),
                "start_keyframe_path": decision.get("start_keyframe_path")
                or (keyframe_paths or {}).get("start"),
                "mid_keyframe_path": decision.get("mid_keyframe_path")
                or (keyframe_paths or {}).get("mid"),
                "end_keyframe_path": decision.get("end_keyframe_path")
                or (keyframe_paths or {}).get("end"),
            }
        )
    evidence_free_decisions = [
        row for row in review_rows
        if isinstance(row.get("passed"), bool)
        and (
            not str(row.get("finding") or "").strip()
            or not str(row.get("evidence") or "").strip()
        )
    ]
    decided = [
        row for row in review_rows
        if isinstance(row.get("passed"), bool) and row not in evidence_free_decisions
    ]
    blocker_failures = [
        row for row in decided
        if row.get("passed") is False and row.get("severity") == "blocker"
    ]
    passed = bool(review_rows) and len(decided) == len(review_rows) and not blocker_failures
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "reviewer": str(reviewer or "hallucination_review_agent"),
        "checks_reviewed": len(decided),
        "checks_total": len(review_rows),
        "passed": passed,
        "results": review_rows,
        "summary": {
            "pending_count": len(review_rows) - len(decided),
            "evidence_free_decision_count": len(evidence_free_decisions),
            "blocker_failure_count": len(blocker_failures),
        },
    }


def save_truth_contract_bundle(
    project_root: str | Path,
    *,
    idea_card: dict[str, Any],
    research_plan: dict[str, Any],
    producer: str = "truth_contracts",
) -> dict[str, Any]:
    """Persist truth contract, hallucination checks, and initial review shell."""

    root = Path(project_root).expanduser().resolve()
    truth_contract = build_truth_contract(idea_card, research_plan)
    checks = derive_hallucination_checks(truth_contract, research_plan)
    review = build_hallucination_review(checks)
    save_contract_artifact(
        root,
        "truth_contract",
        truth_contract,
        producer=producer,
        depends_on=["idea_cards", "research_plan"],
    )
    save_contract_artifact(
        root,
        "hallucination_checks",
        checks,
        producer=producer,
        depends_on=["truth_contract", "research_plan"],
    )
    save_contract_artifact(
        root,
        "hallucination_review",
        review,
        producer=producer,
        depends_on=["hallucination_checks"],
        notes="Initial shell; review agents fill decisions after artifact generation.",
    )
    return {
        "truth_contract": truth_contract,
        "hallucination_checks": checks,
        "hallucination_review": review,
        "truth_contract_validation": validate_truth_contract(truth_contract),
        "hallucination_checks_validation": validate_hallucination_checks(
            checks,
            research_plan=research_plan,
        ),
    }


def load_truth_contract_bundle(project_root: str | Path) -> dict[str, Any]:
    return {
        "truth_contract": load_contract_artifact(project_root, "truth_contract", default={}) or {},
        "hallucination_checks": load_contract_artifact(project_root, "hallucination_checks", default={}) or {},
        "hallucination_review": load_contract_artifact(project_root, "hallucination_review", default={}) or {},
    }
