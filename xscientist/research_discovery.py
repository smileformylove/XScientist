"""Generalization-gated method discovery on top of Research VCS objects.

The protocol deliberately reuses built-in Research Object kinds so repositories
created before this module remain readable.  A locked ``experiment_design`` is
the discovery contract; ``resource_budget`` and ``evaluation_blinding`` bind
the two easiest sources of benchmark leakage; an ``evidence_synthesis`` stores
the deterministic cross-condition assessment.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash

from .research_commands import _ensure_direct_save_is_safe, _finish
from .research_git import ResearchGitError
from .research_vcs import ResearchRepository

DISCOVERY_LEVELS = (
    "execution",
    "engineering_optimization",
    "method_discovery",
)
CONDITION_ROLES = ("development", "transfer", "heldout", "scale")
CONDITION_VISIBILITIES = ("visible", "sealed")


def discovery_contract_template() -> dict[str, Any]:
    """Return a complete, editable contract so beginners do not start blank."""

    return {
        "summary": "Test whether the candidate mechanism transfers",
        "contribution_level": "method_discovery",
        "target_component": "REPLACE_TARGET_COMPONENT",
        "mechanism": "REPLACE_PROPOSED_MECHANISM",
        "metric": {
            "name": "REPLACE_METRIC",
            "direction": "maximize",
            "minimum_effect": 0.0,
        },
        "edit_scope": {
            "allowed_paths": ["REPLACE_TARGET_PATH"],
            "protected_paths": [
                "REPLACE_MODEL_PATH",
                "REPLACE_DATA_PATH",
                "REPLACE_EVALUATOR_PATH",
            ],
        },
        "fixed_variables": {
            "model": "REPLACE_MODEL_ID",
            "data_version": "REPLACE_DATA_ID",
            "training_steps": 0,
        },
        "resource_limits": {
            "wall_time_seconds": 0,
            "accelerator_hours": 0,
            "parameters": 0,
        },
        "runner": {
            "entrypoint": "REPLACE_EVALUATION_COMMAND",
            "seeds": [1, 2, 3],
        },
        "baselines": [
            {
                "id": "REPLACE_STRONG_BASELINE_1",
                "method": "REPLACE_BASELINE_METHOD_1",
                "source": "REPLACE_PAPER_OR_COMMIT_1",
            },
            {
                "id": "REPLACE_STRONG_BASELINE_2",
                "method": "REPLACE_BASELINE_METHOD_2",
                "source": "REPLACE_PAPER_OR_COMMIT_2",
            },
            {
                "id": "REPLACE_STRONG_BASELINE_3",
                "method": "REPLACE_BASELINE_METHOD_3",
                "source": "REPLACE_PAPER_OR_COMMIT_3",
            },
        ],
        "conditions": [
            {
                "id": "development",
                "role": "development",
                "visibility": "visible",
                "context": "REPLACE_DEVELOPMENT_CONDITION",
                "scale": "proxy",
                "proxy_for": "target-scale",
            },
            {
                "id": "transfer",
                "role": "transfer",
                "visibility": "sealed",
                "context": "REPLACE_TRANSFER_CONDITION",
                "scale": "proxy",
            },
            {
                "id": "target-scale",
                "role": "scale",
                "visibility": "sealed",
                "context": "REPLACE_TARGET_SCALE_CONDITION",
                "scale": "target",
            },
        ],
    }


def _text(value: Any, *, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ResearchGitError(f"{label} is required")
    return result


def _sha256(value: Any, *, label: str) -> str:
    import re

    result = _text(value, label=label)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", result):
        raise ResearchGitError(f"{label} must use sha256:<64 lowercase hex>")
    return result


def _relative_pattern(value: Any, *, label: str) -> str:
    result = _text(value, label=label).replace("\\", "/")
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or result.startswith("./"):
        raise ResearchGitError(f"{label} must be a repository-relative path pattern")
    return result


def _mapping(value: Any, *, label: str, required: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchGitError(f"{label} must be a JSON object")
    result = {str(key): item for key, item in value.items()}
    if required and not result:
        raise ResearchGitError(f"{label} must not be empty")
    return dict(sorted(result.items()))


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchGitError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResearchGitError(f"{label} must be finite")
    return result


def _contains_template_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "REPLACE_" in value
    if isinstance(value, Mapping):
        return any(_contains_template_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_template_placeholder(item) for item in value)
    return False


def _normalize_baselines(value: Any, *, strict: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ResearchGitError("discovery baselines must be a JSON array")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if isinstance(raw, str):
            row: dict[str, Any] = {"id": raw, "method": raw, "strong": True}
        elif isinstance(raw, Mapping):
            row = dict(raw)
        else:
            raise ResearchGitError(f"baseline {index + 1} must be a string or object")
        baseline_id = _text(row.get("id"), label=f"baseline {index + 1} id")
        rows.append(
            {
                "id": baseline_id,
                "method": _text(
                    row.get("method") or baseline_id,
                    label=f"baseline {baseline_id} method",
                ),
                "strong": row.get("strong", True) is True,
                **(
                    {"source": str(row["source"]).strip()}
                    if str(row.get("source") or "").strip()
                    else {}
                ),
            }
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ResearchGitError("discovery baseline ids must be unique")
    if not rows:
        raise ResearchGitError("discovery requires at least one comparison method")
    if strict and (len(rows) < 3 or sum(row["strong"] for row in rows) < 3):
        raise ResearchGitError(
            "method discovery requires at least three strong comparison methods"
        )
    if strict and any(not row.get("source") for row in rows):
        raise ResearchGitError(
            "method discovery baselines require a paper, artifact, or commit source"
        )
    return sorted(rows, key=lambda row: row["id"])


def _normalize_conditions(value: Any, *, strict: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ResearchGitError("discovery conditions must be a JSON array")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ResearchGitError(f"condition {index + 1} must be a JSON object")
        row = dict(raw)
        condition_id = _text(row.get("id"), label=f"condition {index + 1} id")
        role = _text(row.get("role"), label=f"condition {condition_id} role")
        visibility = _text(
            row.get("visibility"), label=f"condition {condition_id} visibility"
        )
        if role not in CONDITION_ROLES:
            raise ResearchGitError(
                f"condition {condition_id} role must be one of: "
                + ", ".join(CONDITION_ROLES)
            )
        if visibility not in CONDITION_VISIBILITIES:
            raise ResearchGitError(
                f"condition {condition_id} visibility must be visible or sealed"
            )
        descriptors = {
            key: str(row[key]).strip()
            for key in ("dataset", "task", "scale", "context")
            if str(row.get(key) or "").strip()
        }
        if not descriptors:
            raise ResearchGitError(
                f"condition {condition_id} requires dataset, task, scale, or context"
            )
        normalized = {
            "id": condition_id,
            "role": role,
            "visibility": visibility,
            **descriptors,
        }
        if str(row.get("proxy_for") or "").strip():
            normalized["proxy_for"] = str(row["proxy_for"]).strip()
        rows.append(normalized)
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        raise ResearchGitError("discovery condition ids must be unique")
    if not rows:
        raise ResearchGitError("discovery requires at least one evaluation condition")
    for row in rows:
        if row.get("proxy_for") and row["proxy_for"] not in ids:
            raise ResearchGitError(
                f"condition {row['id']} proxy_for target does not exist"
            )
    if strict:
        roles = {row["role"] for row in rows}
        if len(rows) < 3:
            raise ResearchGitError(
                "method discovery requires at least three evaluation conditions"
            )
        if "development" not in roles:
            raise ResearchGitError("method discovery requires a development condition")
        if not roles.intersection({"transfer", "heldout", "scale"}):
            raise ResearchGitError(
                "method discovery requires a transfer, heldout, or scale condition"
            )
        if not any(row["visibility"] == "sealed" for row in rows):
            raise ResearchGitError(
                "method discovery requires at least one sealed evaluation condition"
            )
    return sorted(rows, key=lambda row: row["id"])


def build_discovery_contract(
    hypothesis_id: str, spec: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Normalize and content-bind one method-discovery planning document."""

    source = _mapping(spec, label="discovery specification")
    if _contains_template_placeholder(source):
        raise ResearchGitError(
            "discovery specification still contains REPLACE_* template values"
        )
    level = _text(
        source.get("contribution_level") or "method_discovery",
        label="contribution_level",
    )
    if level not in DISCOVERY_LEVELS:
        raise ResearchGitError(
            "contribution_level must be execution, engineering_optimization, "
            "or method_discovery"
        )
    strict = level == "method_discovery"
    metric = _mapping(source.get("metric"), label="metric")
    metric_name = _text(metric.get("name"), label="metric name")
    direction = _text(metric.get("direction"), label="metric direction")
    if direction not in {"maximize", "minimize"}:
        raise ResearchGitError("metric direction must be maximize or minimize")
    minimum_effect = _number(
        metric.get("minimum_effect", 0.0), label="metric minimum_effect"
    )
    if minimum_effect < 0:
        raise ResearchGitError("metric minimum_effect cannot be negative")
    normalized_metric: dict[str, Any] = {
        "name": metric_name,
        "direction": direction,
        "minimum_effect": minimum_effect,
    }
    if metric.get("theoretical_bound") is not None:
        normalized_metric["theoretical_bound"] = _number(
            metric["theoretical_bound"], label="metric theoretical_bound"
        )

    edit_scope = _mapping(source.get("edit_scope"), label="edit_scope")
    allowed_paths = sorted(
        {
            _relative_pattern(item, label="allowed path")
            for item in edit_scope.get("allowed_paths") or []
        }
    )
    protected_paths = sorted(
        {
            _relative_pattern(item, label="protected path")
            for item in edit_scope.get("protected_paths") or []
        }
    )
    if not allowed_paths:
        raise ResearchGitError("discovery edit_scope requires allowed_paths")
    if strict and not protected_paths:
        raise ResearchGitError(
            "method discovery requires protected_paths for non-target variables"
        )

    fixed_variables = _mapping(
        source.get("fixed_variables") or {},
        label="fixed_variables",
        required=strict,
    )
    raw_limits = _mapping(
        source.get("resource_limits") or {},
        label="resource_limits",
        required=True,
    )
    resource_limits: dict[str, float] = {}
    for name, value in raw_limits.items():
        limit = _number(value, label=f"resource limit {name}")
        if limit < 0:
            raise ResearchGitError(f"resource limit {name} cannot be negative")
        resource_limits[name] = limit

    baselines = _normalize_baselines(source.get("baselines") or [], strict=strict)
    conditions = _normalize_conditions(source.get("conditions") or [], strict=strict)
    runner = _mapping(source.get("runner"), label="runner")
    runner_hash = canonical_content_hash(runner)
    allocation_policy = _text(
        source.get("allocation_policy") or "maximize_expected_information_gain",
        label="allocation_policy",
    )
    budget_core = {
        "protocol_kind": "method_discovery_budget",
        "limits": resource_limits,
        "allocation_policy": allocation_policy,
        "information_value_required": True,
    }
    budget = {**budget_core, "budget_hash": canonical_content_hash(budget_core)}

    sealed_ids = sorted(
        row["id"] for row in conditions if row["visibility"] == "sealed"
    )
    blinding_core = {
        "protocol_kind": "method_discovery_blinding",
        "policy": "sealed_condition_results",
        "sealed_condition_ids": sealed_ids,
        "feedback_policy": "release_after_final_candidate_commitment",
        "leakage_prohibited": True,
    }
    blinding = {
        **blinding_core,
        "blinding_hash": canonical_content_hash(blinding_core),
    }

    success_source = source.get("success_rule") or {}
    success = _mapping(success_source, label="success_rule", required=False)
    required_condition_ids = sorted(
        {
            str(item).strip()
            for item in success.get(
                "required_condition_ids", [row["id"] for row in conditions]
            )
            if str(item).strip()
        }
    )
    unknown_required = sorted(
        set(required_condition_ids) - {row["id"] for row in conditions}
    )
    if unknown_required:
        raise ResearchGitError(
            "success_rule references unknown conditions: " + ", ".join(unknown_required)
        )
    success_rule = {
        "required_condition_ids": required_condition_ids,
        "minimum_effect": _number(
            success.get("minimum_effect", minimum_effect),
            label="success_rule minimum_effect",
        ),
        "allow_required_condition_regression": success.get(
            "allow_required_condition_regression", False
        )
        is True,
    }
    if success_rule["minimum_effect"] < 0:
        raise ResearchGitError("success_rule minimum_effect cannot be negative")

    contract_core = {
        "protocol_kind": "method_discovery_contract",
        "summary": _text(source.get("summary"), label="discovery summary"),
        "hypothesis_id": _text(hypothesis_id, label="hypothesis_id"),
        "contribution_level": level,
        "target_component": _text(
            source.get("target_component"), label="target_component"
        ),
        "mechanism": _text(source.get("mechanism"), label="mechanism"),
        "metric": normalized_metric,
        "edit_scope": {
            "allowed_paths": allowed_paths,
            "protected_paths": protected_paths,
        },
        "fixed_variables": fixed_variables,
        "baselines": baselines,
        "conditions": conditions,
        "runner": runner,
        "runner_hash": runner_hash,
        "resource_budget_hash": budget["budget_hash"],
        "evaluation_blinding_hash": blinding["blinding_hash"],
        "proxy_fidelity_required": any(row.get("proxy_for") for row in conditions),
        "success_rule": success_rule,
    }
    contract = {
        **contract_core,
        "design_hash": canonical_content_hash(contract_core),
    }
    return {"contract": contract, "budget": budget, "blinding": blinding}


def save_discovery_contract(
    repo: str,
    *,
    hypothesis_id: str,
    spec: Mapping[str, Any],
    context_id: str | None = None,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Lock target scope, comparisons, conditions, compute, and feedback policy."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    hypothesis = repository.get(hypothesis_id)
    if hypothesis["kind"] != "hypothesis":
        raise ResearchGitError("discovery contract hypothesis reference has wrong kind")
    payloads = build_discovery_contract(str(hypothesis["object_id"]), spec)
    context_object: Mapping[str, Any] | None = None
    if context_id:
        context_object = repository.get(context_id)
        if context_object["kind"] != "context_snapshot":
            raise ResearchGitError("discovery decision context has wrong kind")
        context_payload = context_object.get("payload") or {}
        if context_payload.get("complete") is not True:
            raise ResearchGitError("discovery decision context must be complete")
        contract_core = {
            key: value
            for key, value in payloads["contract"].items()
            if key != "design_hash"
        }
        contract_core.update(
            {
                "context_required": True,
                "context_id": str(context_object["object_id"]),
                "context_hash": context_payload.get("context_hash"),
            }
        )
        payloads["contract"] = {
            **contract_core,
            "design_hash": canonical_content_hash(contract_core),
        }
    budget = repository.record("resource_budget", payloads["budget"], state="locked")
    blinding = repository.record(
        "evaluation_blinding", payloads["blinding"], state="locked"
    )
    contract = repository.record(
        "experiment_design",
        payloads["contract"],
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": str(hypothesis["object_id"]),
                "role": "hypothesis",
            },
            {
                "type": "depends_on",
                "target": budget.object_id,
                "role": "resource_budget",
            },
            {
                "type": "depends_on",
                "target": blinding.object_id,
                "role": "evaluation_blinding",
            },
            *(
                [
                    {
                        "type": "uses_context",
                        "target": str(context_object["object_id"]),
                        "role": "decision_context",
                    }
                ]
                if context_object is not None
                else []
            ),
        ],
    )
    return _finish(
        repository,
        contract,
        stage="preregister",
        subject=message or "lock method-discovery evaluation contract",
        status="locked",
        commit=commit,
        related=[budget, blinding],
    )


def _path_allowed(path: str, patterns: Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern)
        or (pattern.endswith("/") and path.startswith(pattern))
        for pattern in patterns
    )


def _utility(value: float, direction: str) -> float:
    return value if direction == "maximize" else -value


def _normalized_score(
    candidate: float,
    baselines: Sequence[float],
    *,
    direction: str,
    theoretical_bound: float | None,
) -> float | None:
    utilities = [_utility(value, direction) for value in baselines]
    weakest, strongest = min(utilities), max(utilities)
    candidate_utility = _utility(candidate, direction)
    if strongest == weakest:
        return None
    if candidate_utility <= strongest or theoretical_bound is None:
        return 50.0 * (candidate_utility - weakest) / (strongest - weakest)
    bound_utility = _utility(theoretical_bound, direction)
    if bound_utility <= strongest:
        return 50.0
    return 50.0 + 50.0 * (candidate_utility - strongest) / (bound_utility - strongest)


def assess_generalization(
    contract: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    locked_resource_limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically distinguish local engineering gains from transfer."""

    if contract.get("protocol_kind") != "method_discovery_contract":
        raise ResearchGitError("selected design is not a method discovery contract")
    source = _mapping(results, label="discovery results")
    candidate = _mapping(source.get("candidate"), label="candidate")
    candidate_id = _text(candidate.get("id"), label="candidate id")
    changed_paths = sorted(
        {
            _relative_pattern(item, label="candidate changed path")
            for item in candidate.get("changed_paths") or []
        }
    )
    if not changed_paths:
        raise ResearchGitError("candidate changed_paths must not be empty")

    checks: list[dict[str, Any]] = []

    def check(code: str, passed: bool, detail: str) -> None:
        checks.append({"code": code, "passed": bool(passed), "detail": detail})

    expected_runner = str(contract.get("runner_hash") or "")
    observed_runner = _sha256(source.get("runner_hash"), label="results runner_hash")
    check(
        "same_runner",
        observed_runner == expected_runner,
        "candidate and baselines must use the locked evaluation pipeline",
    )

    edit_scope = contract.get("edit_scope") or {}
    allowed = list(edit_scope.get("allowed_paths") or [])
    protected = list(edit_scope.get("protected_paths") or [])
    outside = [path for path in changed_paths if not _path_allowed(path, allowed)]
    touched_protected = [
        path for path in changed_paths if _path_allowed(path, protected)
    ]
    check(
        "target_variable_isolated",
        not outside and not touched_protected,
        "changed paths must stay inside the allowed target and outside protected scope",
    )

    observed_fixed = _mapping(
        source.get("fixed_variables") or {},
        label="results fixed_variables",
        required=False,
    )
    check(
        "fixed_variables_match",
        observed_fixed == dict(contract.get("fixed_variables") or {}),
        "non-target model, data, training, and evaluator variables must match",
    )

    observed_usage = _mapping(
        source.get("resource_usage") or {},
        label="results resource_usage",
        required=False,
    )
    limit_failures: list[str] = []
    locked_limits = dict(locked_resource_limits or {})
    if not locked_limits:
        # Compatibility for early hand-authored contracts. Normal builders bind
        # limits through a separate content-addressed resource_budget object.
        locked_limits = dict(contract.get("resource_limits") or {})
    limit_failures.extend(sorted(set(observed_usage) - set(locked_limits)))
    # ``save_generalization_assessment`` injects the content-bound budget limits
    # here.  Keeping evaluation pure also makes imported contracts assessable.
    for name, limit_value in locked_limits.items():
        if name not in observed_usage:
            limit_failures.append(str(name))
        elif _number(observed_usage[name], label=f"resource usage {name}") > float(
            limit_value
        ):
            limit_failures.append(str(name))
    check(
        "resource_parity",
        not limit_failures,
        "candidate resource use must be declared and stay within the locked budget",
    )

    raw_rows = source.get("condition_results")
    if not isinstance(raw_rows, list):
        raise ResearchGitError("condition_results must be a JSON array")
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ResearchGitError("each condition result must be a JSON object")
        condition_id = _text(raw.get("condition_id"), label="condition_result id")
        if condition_id in result_by_id:
            raise ResearchGitError(f"duplicate condition result: {condition_id}")
        result_by_id[condition_id] = raw
    contract_conditions = {
        str(row["id"]): row for row in contract.get("conditions") or []
    }
    complete_conditions = set(result_by_id) == set(contract_conditions)
    check(
        "condition_set_complete",
        complete_conditions,
        "every locked condition, including sealed conditions, must be reported exactly once",
    )

    baseline_ids = [str(row["id"]) for row in contract.get("baselines") or []]
    direction = str((contract.get("metric") or {}).get("direction") or "")
    theoretical_bound = (contract.get("metric") or {}).get("theoretical_bound")
    minimum_effect = float(
        (contract.get("success_rule") or {}).get("minimum_effect") or 0.0
    )
    condition_rows: list[dict[str, Any]] = []
    baseline_complete = True
    for condition_id, condition in sorted(contract_conditions.items()):
        raw = result_by_id.get(condition_id)
        if raw is None:
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "role": condition.get("role"),
                    "visibility": condition.get("visibility"),
                    "passed": False,
                    "missing": True,
                }
            )
            continue
        candidate_value = _number(
            raw.get("candidate"), label=f"condition {condition_id} candidate"
        )
        baseline_values = raw.get("baselines")
        if not isinstance(baseline_values, Mapping):
            raise ResearchGitError(
                f"condition {condition_id} baselines must be a JSON object"
            )
        present = set(map(str, baseline_values)) == set(baseline_ids)
        baseline_complete = baseline_complete and present
        values = {
            baseline_id: _number(
                baseline_values[baseline_id],
                label=f"condition {condition_id} baseline {baseline_id}",
            )
            for baseline_id in baseline_ids
            if baseline_id in baseline_values
        }
        if len(values) != len(baseline_ids):
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "role": condition.get("role"),
                    "visibility": condition.get("visibility"),
                    "candidate": candidate_value,
                    "passed": False,
                    "missing_baselines": sorted(set(baseline_ids) - set(values)),
                }
            )
            continue
        utilities = {name: _utility(value, direction) for name, value in values.items()}
        strongest = max(utilities.values())
        candidate_utility = _utility(candidate_value, direction)
        passed = candidate_utility >= strongest + minimum_effect
        ranking = sorted(baseline_ids, key=lambda name: (-utilities[name], name))
        normalized = _normalized_score(
            candidate_value,
            list(values.values()),
            direction=direction,
            theoretical_bound=(
                float(theoretical_bound) if theoretical_bound is not None else None
            ),
        )
        condition_rows.append(
            {
                "condition_id": condition_id,
                "role": condition.get("role"),
                "visibility": condition.get("visibility"),
                "candidate": candidate_value,
                "baselines": dict(sorted(values.items())),
                "baseline_ranking": ranking,
                "normalized_score": (
                    round(normalized, 8) if normalized is not None else None
                ),
                "passed": passed,
            }
        )
    check(
        "strong_baselines_reproduced",
        baseline_complete
        and len(baseline_ids)
        >= (3 if contract.get("contribution_level") == "method_discovery" else 1),
        "all locked strong baselines must be scored under every condition",
    )

    rows_by_id = {row["condition_id"]: row for row in condition_rows}
    proxy_pairs = [
        (str(row["id"]), str(row["proxy_for"]))
        for row in contract.get("conditions") or []
        if row.get("proxy_for")
    ]
    proxy_preserved = all(
        rows_by_id.get(proxy, {}).get("baseline_ranking")
        == rows_by_id.get(target, {}).get("baseline_ranking")
        for proxy, target in proxy_pairs
    )
    check(
        "proxy_ranking_preserved",
        proxy_preserved if contract.get("proxy_fidelity_required") else True,
        "proxy scale must preserve the human-baseline ranking at target scale",
    )

    required_ids = set(
        (contract.get("success_rule") or {}).get("required_condition_ids") or []
    )
    required_pass = bool(required_ids) and all(
        rows_by_id.get(condition_id, {}).get("passed") is True
        for condition_id in required_ids
    )
    development_pass = any(
        row.get("role") == "development" and row.get("passed") is True
        for row in condition_rows
    )
    generalization_pass = any(
        row.get("role") in {"transfer", "heldout", "scale"}
        and row.get("visibility") == "sealed"
        and row.get("passed") is True
        for row in condition_rows
    )
    check(
        "required_conditions_pass",
        required_pass,
        "candidate must beat the strongest baseline by the locked minimum effect",
    )
    check(
        "sealed_generalization_pass",
        generalization_pass,
        "at least one sealed transfer, heldout, or scale condition must pass",
    )

    integrity_codes = {
        "same_runner",
        "target_variable_isolated",
        "fixed_variables_match",
        "resource_parity",
        "condition_set_complete",
        "strong_baselines_reproduced",
        "proxy_ranking_preserved",
    }
    integrity_pass = all(
        row["passed"] for row in checks if row["code"] in integrity_codes
    )
    if (
        contract.get("contribution_level") == "method_discovery"
        and integrity_pass
        and required_pass
        and generalization_pass
    ):
        verdict = "method_discovery_supported"
    elif integrity_pass and development_pass:
        verdict = "engineering_gain_only"
    elif not integrity_pass:
        verdict = "invalid_protocol_execution"
    else:
        verdict = "inconclusive"
    return {
        "candidate_id": candidate_id,
        "changed_paths": changed_paths,
        "resource_usage": observed_usage,
        "condition_assessments": condition_rows,
        "checks": checks,
        "verdict": verdict,
        "method_discovery_supported": verdict == "method_discovery_supported",
    }


def save_generalization_assessment(
    repo: str,
    *,
    contract_id: str,
    results: Mapping[str, Any],
    evidence_ids: Sequence[str],
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record the cross-condition verdict and bind it to underlying evidence."""

    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    contract_object = repository.get(contract_id)
    if contract_object["kind"] != "experiment_design":
        raise ResearchGitError("generalization contract reference has wrong kind")
    contract = contract_object.get("payload") or {}
    budget_ids = [
        str(row.get("target") or "")
        for row in contract_object.get("relations") or []
        if row.get("role") == "resource_budget"
    ]
    if not budget_ids:
        raise ResearchGitError("discovery contract is missing its resource budget")
    budget_object = repository.get(budget_ids[0])
    budget_payload = budget_object.get("payload") or {}
    assessment = assess_generalization(
        contract,
        results,
        locked_resource_limits=dict(budget_payload.get("limits") or {}),
    )
    if not evidence_ids:
        raise ResearchGitError(
            "generalization assessment requires condition evidence objects"
        )
    relations: list[dict[str, str]] = [
        {
            "type": "depends_on",
            "target": str(contract_object["object_id"]),
            "role": "discovery_contract",
        }
    ]
    resolved_evidence: list[str] = []
    for selector in evidence_ids:
        item = repository.get(selector)
        if item["kind"] not in {
            "evidence",
            "effect_estimate",
            "passage_evidence",
        }:
            raise ResearchGitError(
                "generalization assessment evidence reference has wrong kind"
            )
        object_id = str(item["object_id"])
        resolved_evidence.append(object_id)
        relations.append({"type": "derived_from", "target": object_id})
    core = {
        "protocol_kind": "generalization_assessment",
        "summary": ("Cross-condition assessment of " + assessment["candidate_id"]),
        "contract_id": str(contract_object["object_id"]),
        "contract_hash": contract.get("design_hash"),
        "evidence_ids": sorted(set(resolved_evidence)),
        **assessment,
    }
    payload = {**core, "synthesis_hash": canonical_content_hash(core)}
    result = repository.record(
        "evidence_synthesis",
        payload,
        state="completed",
        relations=relations,
    )
    saved = _finish(
        repository,
        result,
        stage="evidence",
        subject=message or "assess cross-condition method generalization",
        status="completed",
        commit=commit,
    )
    saved["assessment"] = assessment
    return saved


__all__ = [
    "CONDITION_ROLES",
    "CONDITION_VISIBILITIES",
    "DISCOVERY_LEVELS",
    "assess_generalization",
    "build_discovery_contract",
    "discovery_contract_template",
    "save_discovery_contract",
    "save_generalization_assessment",
]
