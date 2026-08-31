"""Deterministic, content-addressed diagnostics for evolution harnesses.

The contract in this module adapts EvoTrainer's score/signal/behavior/version
diagnostic split to XScientist's controlled self-evolution boundary.  It is an
offline audit, not an RL runtime: callers provide already content-addressed
version evidence and optional skill backtests, and the builder returns a
bounded diagnostic report.  It performs no model call, network access, file
write, production mutation, or evaluator mutation.

Two details are deliberate:

* a healthy report only makes a version eligible for human review; it never
  authorizes automatic progression or production deployment;
* evaluator or harness changes make the observed scores incomparable in the
  current epoch.  They are emitted as next-epoch challenges rather than being
  used to improve the candidate's apparent fitness under newly written rules.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from ai_scientist.protocol.canonical_json import (
    CANONICAL_JSON_PROFILE,
    canonical_content_hash,
)

HARNESS_AUDIT_SCHEMA = "xscientist.evolution-harness-audit.v1"
HARNESS_SKILL_SCHEMA = "xscientist.evolution-harness-skill.v1"
HARNESS_POLICY_ID = "xscientist.evolution-harness-policy.v1"

MAX_VERSIONS = 32
MAX_SCORE_METRICS = 32
MAX_BEHAVIOR_METRICS = 32
MAX_REWARD_GROUPS = 128
MAX_REWARD_GROUP_SIZE = 64
MAX_BACKTESTS = 256
MAX_SKILLS = 64
MAX_DOMAINS_PER_SKILL = 16
MAX_ABS_VALUE = 1.0e12

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_METRIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")

COMPARISON_HASH_FIELDS = (
    "harness_hash",
    "policy_hash",
    "evaluator_hash",
    "task_hash",
    "resource_hash",
    "seed_policy_hash",
)
REQUIRED_INTEGRITY_CHECKS = (
    "evidence_bound",
    "environment_isolated",
    "evaluation_frozen",
    "git_leakage_absent",
)
SKILL_TYPES = frozenset({"analyzer", "repair_strategy", "procedure_template"})
BACKTEST_SPLITS = frozenset({"historical", "holdout"})
EVALUATOR_AUTHORITIES = frozenset(
    {"independent_evaluator", "human", "deterministic_gate", "research_agent"}
)
SKILL_STATUSES = frozenset(
    {"quarantined", "domain_validated", "cross_domain_validated"}
)
SKILL_QUARANTINE_REASONS = frozenset(
    {
        "historical_validation_incomplete",
        "holdout_validation_incomplete",
        "independent_evaluator_missing",
        "metadata_conflict",
    }
)

RISK_CODES = frozenset(
    {
        "SCORE.CONTRACT_MISMATCH",
        "SCORE.NO_IMPROVEMENT",
        "SIGNAL.ALL_TIED",
        "SIGNAL.LOW_INFORMATION",
        "BEHAVIOR.CONTRACT_MISMATCH",
        "BEHAVIOR.REGRESSION",
        "BEHAVIOR.SCORE_DIVERGENCE",
        "BEHAVIOR.THRESHOLD_VIOLATION",
        "VERSION.COST_BUDGET_EXCEEDED",
        "VERSION.EVALUATOR_MISMATCH",
        "VERSION.GIT_LEAKAGE",
        "VERSION.HARNESS_MISMATCH",
        "VERSION.INTEGRITY_FAILURE",
        "VERSION.RESOURCE_MISMATCH",
        "VERSION.SEED_POLICY_MISMATCH",
        "VERSION.TASK_MISMATCH",
    }
)

DEFAULT_POLICY: dict[str, Any] = {
    "primary_score": "objective",
    "score_direction": "higher",
    "minimum_score_improvement": 0.0,
    "reward_std_threshold": 1.0e-6,
    "low_information_ratio": 0.5,
    "tie_tolerance": 1.0e-12,
}

_POLICY_OVERRIDE_KEYS = frozenset(DEFAULT_POLICY)

_RISK_CHALLENGE_TARGET = {
    "SCORE.CONTRACT_MISMATCH": "score_contract",
    "SCORE.NO_IMPROVEMENT": "diagnostic_procedure",
    "SIGNAL.ALL_TIED": "signal_metric_or_filter",
    "SIGNAL.LOW_INFORMATION": "signal_metric_or_filter",
    "BEHAVIOR.CONTRACT_MISMATCH": "behavior_contract",
    "BEHAVIOR.REGRESSION": "behavior_analyzer",
    "BEHAVIOR.SCORE_DIVERGENCE": "behavior_analyzer",
    "BEHAVIOR.THRESHOLD_VIOLATION": "behavior_analyzer",
    "VERSION.COST_BUDGET_EXCEEDED": "resource_allocation",
    "VERSION.EVALUATOR_MISMATCH": "evaluation_policy",
    "VERSION.GIT_LEAKAGE": "integrity_analyzer",
    "VERSION.HARNESS_MISMATCH": "harness_manifest",
    "VERSION.INTEGRITY_FAILURE": "integrity_analyzer",
    "VERSION.RESOURCE_MISMATCH": "comparison_boundary",
    "VERSION.SEED_POLICY_MISMATCH": "comparison_boundary",
    "VERSION.TASK_MISMATCH": "comparison_boundary",
}


class EvolutionHarnessError(ValueError):
    """Raised when unbounded or unverifiable harness evidence is supplied."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvolutionHarnessError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise EvolutionHarnessError(f"{label} keys must be strings")
    return value


def _sequence(value: Any, *, label: str, maximum: int) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvolutionHarnessError(f"{label} must be an array")
    if len(value) > maximum:
        raise EvolutionHarnessError(f"{label} exceeds {maximum} entries")
    return value


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not _ID_RE.fullmatch(text):
        raise EvolutionHarnessError(f"{label} is not a bounded identifier")
    return text


def _metric_name(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not _METRIC_RE.fullmatch(text):
        raise EvolutionHarnessError(f"{label} is not a bounded metric name")
    return text


def _content_hash(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not _HASH_RE.fullmatch(text):
        raise EvolutionHarnessError(f"{label} must use sha256:<64 lowercase hex>")
    return text


def _number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvolutionHarnessError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or abs(parsed) > MAX_ABS_VALUE:
        raise EvolutionHarnessError(f"{label} is outside the finite audit boundary")
    if minimum is not None and parsed < minimum:
        raise EvolutionHarnessError(f"{label} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise EvolutionHarnessError(f"{label} must be <= {maximum}")
    return parsed


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _normalize_metric_map(
    value: Any,
    *,
    label: str,
    maximum: int,
) -> dict[str, float]:
    row = _mapping(value, label=label)
    if not row:
        raise EvolutionHarnessError(f"{label} cannot be empty")
    if len(row) > maximum:
        raise EvolutionHarnessError(f"{label} exceeds {maximum} metrics")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in row.items():
        name = _metric_name(raw_name, label=f"{label} key")
        normalized[name] = _number(raw_value, label=f"{label}.{name}")
    return {name: normalized[name] for name in sorted(normalized)}


def _normalize_reward_groups(value: Any, *, label: str) -> list[list[float]]:
    groups = _sequence(value, label=label, maximum=MAX_REWARD_GROUPS)
    if not groups:
        raise EvolutionHarnessError(f"{label} cannot be empty")
    normalized: list[list[float]] = []
    for group_index, raw_group in enumerate(groups):
        group = _sequence(
            raw_group,
            label=f"{label}[{group_index}]",
            maximum=MAX_REWARD_GROUP_SIZE,
        )
        if len(group) < 2:
            raise EvolutionHarnessError(
                f"{label}[{group_index}] requires at least two rewards"
            )
        normalized.append(
            [
                _number(item, label=f"{label}[{group_index}][{item_index}]")
                for item_index, item in enumerate(group)
            ]
        )
    return normalized


def _normalize_behavior_thresholds(
    value: Any,
    *,
    behavior_names: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    thresholds = _mapping(value, label=label)
    if set(thresholds) != behavior_names:
        raise EvolutionHarnessError(
            f"{label} keys must exactly match the behavior metrics"
        )
    output: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in thresholds.items():
        name = _metric_name(raw_name, label=f"{label} key")
        spec = _mapping(raw_spec, label=f"{label}.{name}")
        expected = {"direction", "healthy_bound", "max_regression"}
        if set(spec) != expected:
            raise EvolutionHarnessError(
                f"{label}.{name} must contain exactly {sorted(expected)}"
            )
        direction = str(spec.get("direction") or "").strip().lower()
        if direction not in {"higher", "lower"}:
            raise EvolutionHarnessError(
                f"{label}.{name}.direction must be higher or lower"
            )
        output[name] = {
            "direction": direction,
            "healthy_bound": _number(
                spec.get("healthy_bound"),
                label=f"{label}.{name}.healthy_bound",
            ),
            "max_regression": _number(
                spec.get("max_regression"),
                label=f"{label}.{name}.max_regression",
                minimum=0.0,
            ),
        }
    return {name: output[name] for name in sorted(output)}


def _normalize_integrity_checks(value: Any, *, label: str) -> dict[str, bool]:
    checks = _mapping(value, label=label)
    if set(checks) != set(REQUIRED_INTEGRITY_CHECKS):
        raise EvolutionHarnessError(
            f"{label} must contain exactly {list(REQUIRED_INTEGRITY_CHECKS)}"
        )
    output: dict[str, bool] = {}
    for name in REQUIRED_INTEGRITY_CHECKS:
        if not isinstance(checks.get(name), bool):
            raise EvolutionHarnessError(f"{label}.{name} must be boolean")
        output[name] = bool(checks[name])
    return output


def _normalize_comparison_hashes(value: Any, *, label: str) -> dict[str, str]:
    hashes = _mapping(value, label=label)
    if set(hashes) != set(COMPARISON_HASH_FIELDS):
        raise EvolutionHarnessError(
            f"{label} must contain exactly {list(COMPARISON_HASH_FIELDS)}"
        )
    return {
        name: _content_hash(hashes[name], label=f"{label}.{name}")
        for name in COMPARISON_HASH_FIELDS
    }


def _normalize_cost(value: Any, *, label: str) -> dict[str, Any]:
    cost = _mapping(value, label=label)
    expected = {"observed", "budget", "unit"}
    if set(cost) != expected:
        raise EvolutionHarnessError(f"{label} must contain exactly {sorted(expected)}")
    budget = _number(cost.get("budget"), label=f"{label}.budget", minimum=0.0)
    if budget <= 0:
        raise EvolutionHarnessError(f"{label}.budget must be positive")
    return {
        "observed": _number(
            cost.get("observed"), label=f"{label}.observed", minimum=0.0
        ),
        "budget": budget,
        "unit": _identifier(cost.get("unit"), label=f"{label}.unit"),
    }


_VERSION_CORE_FIELDS = {
    "epoch_id",
    "version_id",
    "parent_version_id",
    "scores",
    "reward_groups",
    "behavior",
    "behavior_thresholds",
    "integrity_checks",
    "comparison_hashes",
    "cost",
}


def _normalize_version_core(value: Any) -> dict[str, Any]:
    row = _mapping(value, label="version evidence")
    if set(row) != _VERSION_CORE_FIELDS:
        unknown = sorted(set(row) - _VERSION_CORE_FIELDS)
        missing = sorted(_VERSION_CORE_FIELDS - set(row))
        raise EvolutionHarnessError(
            "version evidence fields are invalid"
            + (f"; missing={missing}" if missing else "")
            + (f"; unsupported={unknown}" if unknown else "")
        )
    epoch_id = _identifier(row.get("epoch_id"), label="epoch_id")
    version_id = _identifier(row.get("version_id"), label="version_id")
    parent_raw = row.get("parent_version_id")
    parent_id = (
        None
        if parent_raw in (None, "")
        else _identifier(parent_raw, label="parent_version_id")
    )
    scores = _normalize_metric_map(
        row.get("scores"), label="scores", maximum=MAX_SCORE_METRICS
    )
    behavior = _normalize_metric_map(
        row.get("behavior"), label="behavior", maximum=MAX_BEHAVIOR_METRICS
    )
    return {
        "epoch_id": epoch_id,
        "version_id": version_id,
        "parent_version_id": parent_id,
        "scores": scores,
        "reward_groups": _normalize_reward_groups(
            row.get("reward_groups"), label="reward_groups"
        ),
        "behavior": behavior,
        "behavior_thresholds": _normalize_behavior_thresholds(
            row.get("behavior_thresholds"),
            behavior_names=set(behavior),
            label="behavior_thresholds",
        ),
        "integrity_checks": _normalize_integrity_checks(
            row.get("integrity_checks"), label="integrity_checks"
        ),
        "comparison_hashes": _normalize_comparison_hashes(
            row.get("comparison_hashes"), label="comparison_hashes"
        ),
        "cost": _normalize_cost(row.get("cost"), label="cost"),
    }


def bind_version_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and content-address one version evidence row."""

    core = _normalize_version_core(value)
    return {**core, "evidence_hash": canonical_content_hash(core)}


def _normalize_version_evidence(value: Any) -> dict[str, Any]:
    row = _mapping(value, label="version evidence")
    expected = _VERSION_CORE_FIELDS | {"evidence_hash"}
    if set(row) != expected:
        raise EvolutionHarnessError(
            "content-addressed version evidence has invalid fields"
        )
    core = _normalize_version_core({name: row[name] for name in _VERSION_CORE_FIELDS})
    supplied_hash = _content_hash(row.get("evidence_hash"), label="evidence_hash")
    expected_hash = canonical_content_hash(core)
    if supplied_hash != expected_hash:
        raise EvolutionHarnessError("evidence_hash does not match version evidence")
    return {**core, "evidence_hash": supplied_hash}


def _normalize_versions(value: Any) -> list[dict[str, Any]]:
    rows = _sequence(value, label="versions", maximum=MAX_VERSIONS)
    if len(rows) < 2:
        raise EvolutionHarnessError("versions requires at least two evidence rows")
    normalized = [_normalize_version_evidence(row) for row in rows]
    identifiers = [row["version_id"] for row in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise EvolutionHarnessError("version_id values must be unique")
    if normalized[0]["parent_version_id"] is not None:
        raise EvolutionHarnessError("the first version must have no parent")
    epoch_ids = {row["epoch_id"] for row in normalized}
    if len(epoch_ids) != 1:
        raise EvolutionHarnessError("versions must belong to one frozen epoch_id")
    for previous, current in zip(normalized, normalized[1:]):
        if current["parent_version_id"] != previous["version_id"]:
            raise EvolutionHarnessError(
                "versions must form one explicit parent-linked lineage"
            )
    return normalized


_BACKTEST_CORE_FIELDS = {
    "skill_id",
    "skill_type",
    "skill_artifact_hash",
    "domain",
    "split",
    "passed",
    "producer_id",
    "evaluator_id",
    "evaluator_authority",
    "evaluator_protocol_hash",
}


def _normalize_backtest_core(value: Any) -> dict[str, Any]:
    row = _mapping(value, label="backtest")
    if set(row) != _BACKTEST_CORE_FIELDS:
        raise EvolutionHarnessError("backtest fields are invalid")
    skill_type = str(row.get("skill_type") or "").strip().lower()
    if skill_type not in SKILL_TYPES:
        raise EvolutionHarnessError(f"skill_type must be one of {sorted(SKILL_TYPES)}")
    split = str(row.get("split") or "").strip().lower()
    if split not in BACKTEST_SPLITS:
        raise EvolutionHarnessError(
            f"backtest split must be one of {sorted(BACKTEST_SPLITS)}"
        )
    if not isinstance(row.get("passed"), bool):
        raise EvolutionHarnessError("backtest passed must be boolean")
    authority = str(row.get("evaluator_authority") or "").strip().lower()
    if authority not in EVALUATOR_AUTHORITIES:
        raise EvolutionHarnessError("backtest evaluator_authority is invalid")
    return {
        "skill_id": _identifier(row.get("skill_id"), label="skill_id"),
        "skill_type": skill_type,
        "skill_artifact_hash": _content_hash(
            row.get("skill_artifact_hash"), label="skill_artifact_hash"
        ),
        "domain": _identifier(row.get("domain"), label="domain"),
        "split": split,
        "passed": bool(row["passed"]),
        "producer_id": _identifier(row.get("producer_id"), label="producer_id"),
        "evaluator_id": _identifier(row.get("evaluator_id"), label="evaluator_id"),
        "evaluator_authority": authority,
        "evaluator_protocol_hash": _content_hash(
            row.get("evaluator_protocol_hash"), label="evaluator_protocol_hash"
        ),
    }


def bind_skill_backtest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and content-address one historical or holdout skill backtest."""

    core = _normalize_backtest_core(value)
    return {**core, "backtest_hash": canonical_content_hash(core)}


def _normalize_backtest(value: Any) -> dict[str, Any]:
    row = _mapping(value, label="backtest")
    expected = _BACKTEST_CORE_FIELDS | {"backtest_hash"}
    if set(row) != expected:
        raise EvolutionHarnessError("content-addressed backtest has invalid fields")
    core = _normalize_backtest_core({name: row[name] for name in _BACKTEST_CORE_FIELDS})
    supplied_hash = _content_hash(row.get("backtest_hash"), label="backtest_hash")
    if supplied_hash != canonical_content_hash(core):
        raise EvolutionHarnessError("backtest_hash does not match backtest evidence")
    return {**core, "backtest_hash": supplied_hash}


def _normalize_backtests(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    rows = _sequence(value, label="backtests", maximum=MAX_BACKTESTS)
    normalized = [_normalize_backtest(row) for row in rows]
    normalized.sort(
        key=lambda row: (
            row["skill_id"],
            row["domain"],
            row["split"],
            row["backtest_hash"],
        )
    )
    if len({row["backtest_hash"] for row in normalized}) != len(normalized):
        raise EvolutionHarnessError("duplicate backtest_hash values are not allowed")
    return normalized


def _normalize_policy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    overrides = {} if value is None else dict(_mapping(value, label="policy"))
    unknown = set(overrides) - _POLICY_OVERRIDE_KEYS
    if unknown:
        raise EvolutionHarnessError(
            "policy has unsupported fields: " + ", ".join(sorted(unknown))
        )
    policy = deepcopy(DEFAULT_POLICY)
    policy.update(overrides)
    primary_score = _metric_name(policy["primary_score"], label="primary_score")
    direction = str(policy["score_direction"] or "").strip().lower()
    if direction not in {"higher", "lower"}:
        raise EvolutionHarnessError("score_direction must be higher or lower")
    return {
        "policy_id": HARNESS_POLICY_ID,
        "primary_score": primary_score,
        "score_direction": direction,
        "minimum_score_improvement": _number(
            policy["minimum_score_improvement"],
            label="minimum_score_improvement",
            minimum=0.0,
        ),
        "reward_std_threshold": _number(
            policy["reward_std_threshold"],
            label="reward_std_threshold",
            minimum=0.0,
        ),
        "low_information_ratio": _number(
            policy["low_information_ratio"],
            label="low_information_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        "tie_tolerance": _number(
            policy["tie_tolerance"], label="tie_tolerance", minimum=0.0
        ),
        "fixed_within_epoch": True,
        "evaluator_change_window": "next_epoch_only",
        "automatic_evaluator_mutation_allowed": False,
        "automatic_production_mutation_allowed": False,
    }


def build_harness_policy_hash(
    policy: Mapping[str, Any] | None = None,
) -> str:
    """Return the canonical commitment versions must declare for ``policy``."""

    return canonical_content_hash(_normalize_policy(policy))


def _risk(
    code: str,
    *,
    layer: str,
    version_id: str,
    detail: str,
) -> dict[str, str]:
    if code not in RISK_CODES:
        raise EvolutionHarnessError(f"internal unsupported risk code: {code}")
    return {
        "code": code,
        "layer": layer,
        "severity": "blocker",
        "version_id": version_id,
        "detail": str(detail)[:512],
    }


def _dedupe_risks(risks: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in risks:
        key = (row["code"], row["version_id"], row["detail"])
        unique[key] = dict(row)
    return sorted(
        unique.values(),
        key=lambda row: (row["code"], row["version_id"], row["detail"]),
    )


def _build_score_layer(
    versions: Sequence[dict[str, Any]], policy: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    primary = str(policy["primary_score"])
    for row in versions:
        if primary not in row["scores"]:
            raise EvolutionHarnessError(
                f"scores for {row['version_id']} omit primary_score={primary}"
            )
    contracts = [tuple(row["scores"]) for row in versions]
    contract_consistent = all(contract == contracts[0] for contract in contracts[1:])
    previous = versions[-2]
    current = versions[-1]
    previous_score = float(previous["scores"][primary])
    current_score = float(current["scores"][primary])
    raw_delta = current_score - previous_score
    signed_improvement = (
        raw_delta if policy["score_direction"] == "higher" else -raw_delta
    )
    improved = signed_improvement > float(policy["minimum_score_improvement"])
    risks: list[dict[str, str]] = []
    if not contract_consistent:
        risks.append(
            _risk(
                "SCORE.CONTRACT_MISMATCH",
                layer="score",
                version_id=current["version_id"],
                detail="score metric sets changed inside the audited lineage",
            )
        )
    if not improved:
        risks.append(
            _risk(
                "SCORE.NO_IMPROVEMENT",
                layer="score",
                version_id=current["version_id"],
                detail=(
                    f"signed improvement {_rounded(signed_improvement)} did not exceed "
                    f"{_rounded(float(policy['minimum_score_improvement']))}"
                ),
            )
        )
    layer = {
        "primary_metric": primary,
        "direction": policy["score_direction"],
        "points": [
            {
                "version_id": row["version_id"],
                "value": _rounded(float(row["scores"][primary])),
            }
            for row in versions
        ],
        "previous_value": _rounded(previous_score),
        "current_value": _rounded(current_score),
        "raw_delta": _rounded(raw_delta),
        "signed_improvement": _rounded(signed_improvement),
        "improved": improved,
        "score_contract_consistent": contract_consistent,
        "metric_names": sorted({name for row in versions for name in row["scores"]}),
    }
    return layer, risks


def _reward_statistics(
    groups: Sequence[Sequence[float]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    standard_deviations = [statistics.pstdev(group) for group in groups]
    low_information = [
        value <= float(policy["reward_std_threshold"]) for value in standard_deviations
    ]
    tied = [
        max(group) - min(group) <= float(policy["tie_tolerance"]) for group in groups
    ]
    flattened = [float(value) for group in groups for value in group]
    return {
        "group_count": len(groups),
        "reward_count": len(flattened),
        "unique_reward_count": len(set(flattened)),
        "all_tied": all(tied),
        "tied_group_count": sum(tied),
        "low_information_group_count": sum(low_information),
        "low_information_ratio": _rounded(sum(low_information) / len(groups)),
        "mean_group_std": _rounded(statistics.mean(standard_deviations)),
        "min_group_std": _rounded(min(standard_deviations)),
        "max_group_std": _rounded(max(standard_deviations)),
    }


def _build_signal_layer(
    versions: Sequence[dict[str, Any]], policy: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    per_version = [
        {
            "version_id": row["version_id"],
            **_reward_statistics(row["reward_groups"], policy),
        }
        for row in versions
    ]
    current = per_version[-1]
    risks: list[dict[str, str]] = []
    if current["all_tied"]:
        risks.append(
            _risk(
                "SIGNAL.ALL_TIED",
                layer="signal",
                version_id=current["version_id"],
                detail="every current reward group is tied",
            )
        )
    # Make the decision from exact counts.  The rounded ratio is presentation
    # data only; using it at a boundary can turn 1/3 into 0.333333333333 and
    # incorrectly pass a threshold that sits between those two values.
    exact_low_information_ratio = (
        current["low_information_group_count"] / current["group_count"]
    )
    if exact_low_information_ratio >= float(policy["low_information_ratio"]):
        risks.append(
            _risk(
                "SIGNAL.LOW_INFORMATION",
                layer="signal",
                version_id=current["version_id"],
                detail=(
                    f"low-information ratio {current['low_information_ratio']} is at or "
                    f"above {policy['low_information_ratio']}"
                ),
            )
        )
    return {
        "per_version": per_version,
        "current": deepcopy(current),
        "signal_eligible": not risks,
    }, risks


def _build_behavior_layer(
    versions: Sequence[dict[str, Any]], score_layer: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    previous = versions[-2]
    current = versions[-1]
    contracts = [
        canonical_content_hash(
            {
                "metric_names": sorted(row["behavior"]),
                "thresholds": row["behavior_thresholds"],
            }
        )
        for row in versions
    ]
    contract_consistent = all(value == contracts[0] for value in contracts[1:])
    risks: list[dict[str, str]] = []
    if not contract_consistent:
        risks.append(
            _risk(
                "BEHAVIOR.CONTRACT_MISMATCH",
                layer="behavior",
                version_id=current["version_id"],
                detail="behavior metric names or thresholds changed inside the epoch",
            )
        )

    shared_names = sorted(set(previous["behavior"]) & set(current["behavior"]))
    evaluations: list[dict[str, Any]] = []
    violating: list[str] = []
    regressed: list[str] = []
    for name in shared_names:
        spec = current["behavior_thresholds"][name]
        before = float(previous["behavior"][name])
        after = float(current["behavior"][name])
        if spec["direction"] == "higher":
            healthy = after >= float(spec["healthy_bound"])
            signed_improvement = after - before
        else:
            healthy = after <= float(spec["healthy_bound"])
            signed_improvement = before - after
        is_regressed = signed_improvement < -float(spec["max_regression"])
        if not healthy:
            violating.append(name)
        if is_regressed:
            regressed.append(name)
        evaluations.append(
            {
                "metric": name,
                "direction": spec["direction"],
                "previous": _rounded(before),
                "current": _rounded(after),
                "healthy_bound": _rounded(float(spec["healthy_bound"])),
                "max_regression": _rounded(float(spec["max_regression"])),
                "signed_improvement": _rounded(signed_improvement),
                "healthy": healthy,
                "regressed": is_regressed,
            }
        )
    if violating:
        risks.append(
            _risk(
                "BEHAVIOR.THRESHOLD_VIOLATION",
                layer="behavior",
                version_id=current["version_id"],
                detail="unhealthy metrics: " + ", ".join(violating),
            )
        )
    if regressed:
        risks.append(
            _risk(
                "BEHAVIOR.REGRESSION",
                layer="behavior",
                version_id=current["version_id"],
                detail="regressed metrics: " + ", ".join(regressed),
            )
        )
    divergence = bool(score_layer["improved"] and (violating or regressed))
    if divergence:
        risks.append(
            _risk(
                "BEHAVIOR.SCORE_DIVERGENCE",
                layer="behavior",
                version_id=current["version_id"],
                detail="primary score improved while process behavior became unhealthy",
            )
        )
    return {
        "contract_consistent": contract_consistent,
        "evaluations": evaluations,
        "threshold_violations": violating,
        "regressed_metrics": regressed,
        "score_behavior_divergence": divergence,
        "behavior_eligible": contract_consistent and not violating and not regressed,
    }, risks


_COMPARISON_RISK_CODES = {
    "harness_hash": "VERSION.HARNESS_MISMATCH",
    "evaluator_hash": "VERSION.EVALUATOR_MISMATCH",
    "task_hash": "VERSION.TASK_MISMATCH",
    "resource_hash": "VERSION.RESOURCE_MISMATCH",
    "seed_policy_hash": "VERSION.SEED_POLICY_MISMATCH",
}


def _build_version_layer(
    versions: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    risks: list[dict[str, str]] = []
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(versions, versions[1:]):
        matches = {
            field: previous["comparison_hashes"][field]
            == current["comparison_hashes"][field]
            for field in COMPARISON_HASH_FIELDS
        }
        mismatch_fields = [
            name for name, matches_value in matches.items() if not matches_value
        ]
        transitions.append(
            {
                "from_version_id": previous["version_id"],
                "to_version_id": current["version_id"],
                "comparison_matches": matches,
                "mismatch_fields": mismatch_fields,
                "comparable": not mismatch_fields,
            }
        )
        for field in mismatch_fields:
            risks.append(
                _risk(
                    _COMPARISON_RISK_CODES[field],
                    layer="version",
                    version_id=current["version_id"],
                    detail=f"{field} changed from the parent version",
                )
            )

    versions_summary: list[dict[str, Any]] = []
    for row in versions:
        failed_checks = [
            name for name, passed in row["integrity_checks"].items() if not passed
        ]
        if failed_checks:
            risks.append(
                _risk(
                    "VERSION.INTEGRITY_FAILURE",
                    layer="version",
                    version_id=row["version_id"],
                    detail="failed integrity checks: " + ", ".join(failed_checks),
                )
            )
        if not row["integrity_checks"]["git_leakage_absent"]:
            risks.append(
                _risk(
                    "VERSION.GIT_LEAKAGE",
                    layer="version",
                    version_id=row["version_id"],
                    detail="Git history or reference-patch leakage was not excluded",
                )
            )
        over_budget = float(row["cost"]["observed"]) > float(row["cost"]["budget"])
        if over_budget:
            risks.append(
                _risk(
                    "VERSION.COST_BUDGET_EXCEEDED",
                    layer="version",
                    version_id=row["version_id"],
                    detail=(
                        f"observed cost {row['cost']['observed']} exceeds budget "
                        f"{row['cost']['budget']} {row['cost']['unit']}"
                    ),
                )
            )
        versions_summary.append(
            {
                "version_id": row["version_id"],
                "evidence_hash": row["evidence_hash"],
                "integrity_passed": not failed_checks,
                "failed_integrity_checks": failed_checks,
                "cost": deepcopy(row["cost"]),
                "within_budget": not over_budget,
            }
        )
    return {
        "lineage_complete": True,
        "content_addressed": True,
        "versions": versions_summary,
        "transitions": transitions,
        "latest_transition_comparable": bool(
            transitions and transitions[-1]["comparable"]
        ),
    }, risks


def _split_validation(rows: Sequence[dict[str, Any]]) -> tuple[bool, bool]:
    """Return (verified, independent) for one domain/split evidence set."""

    if not rows:
        return False, False
    independent = all(
        row["evaluator_authority"] == "independent_evaluator"
        and row["evaluator_id"] != row["producer_id"]
        for row in rows
    )
    return all(row["passed"] for row in rows) and independent, independent


def _build_skills(backtests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in backtests:
        grouped.setdefault(row["skill_id"], []).append(row)
    if len(grouped) > MAX_SKILLS:
        raise EvolutionHarnessError(f"backtests describe more than {MAX_SKILLS} skills")

    skills: list[dict[str, Any]] = []
    for skill_id in sorted(grouped):
        rows = grouped[skill_id]
        types = sorted({row["skill_type"] for row in rows})
        artifacts = sorted({row["skill_artifact_hash"] for row in rows})
        domains = sorted({row["domain"] for row in rows})
        if len(domains) > MAX_DOMAINS_PER_SKILL:
            raise EvolutionHarnessError(
                f"skill {skill_id} exceeds {MAX_DOMAINS_PER_SKILL} domains"
            )
        metadata_conflict = len(types) != 1 or len(artifacts) != 1
        domain_results: list[dict[str, Any]] = []
        validated_domains: list[str] = []
        all_reasons: set[str] = set()
        if metadata_conflict:
            all_reasons.add("metadata_conflict")
        for domain in domains:
            domain_rows = [row for row in rows if row["domain"] == domain]
            historical = [row for row in domain_rows if row["split"] == "historical"]
            holdout = [row for row in domain_rows if row["split"] == "holdout"]
            historical_verified, historical_independent = _split_validation(historical)
            holdout_verified, holdout_independent = _split_validation(holdout)
            reasons: list[str] = []
            if not historical_verified:
                reasons.append("historical_validation_incomplete")
            if not holdout_verified:
                reasons.append("holdout_validation_incomplete")
            if not historical_independent or not holdout_independent:
                reasons.append("independent_evaluator_missing")
            if metadata_conflict:
                reasons.append("metadata_conflict")
            reasons = sorted(set(reasons))
            all_reasons.update(reasons)
            validated = not reasons
            if validated:
                validated_domains.append(domain)
            domain_results.append(
                {
                    "domain": domain,
                    "historical_verified": historical_verified,
                    "holdout_verified": holdout_verified,
                    "independent_evaluator_verified": (
                        historical_independent and holdout_independent
                    ),
                    "validated": validated,
                    "quarantine_reasons": reasons,
                    "backtest_hashes": sorted(
                        row["backtest_hash"] for row in domain_rows
                    ),
                }
            )
        if len(validated_domains) >= 2:
            status = "cross_domain_validated"
            validation_scope = "cross_domain"
        elif len(validated_domains) == 1:
            status = "domain_validated"
            validation_scope = "domain"
        else:
            status = "quarantined"
            validation_scope = "quarantine"
        core = {
            "schema_version": HARNESS_SKILL_SCHEMA,
            "skill_id": skill_id,
            "skill_type": types[0] if len(types) == 1 else "conflicted",
            "skill_artifact_hash": artifacts[0] if len(artifacts) == 1 else None,
            "status": status,
            "validation_scope": validation_scope,
            "observed_domains": domains,
            "validated_domains": validated_domains,
            "domain_results": domain_results,
            "quarantine_reasons": sorted(all_reasons),
            "evidence_hashes": sorted(row["backtest_hash"] for row in rows),
            "historical_and_holdout_required": True,
            "independent_evaluator_required": True,
            "automatic_application_allowed": False,
            "production_promotion_allowed": False,
            "requires_fresh_evolution_gate": True,
        }
        skills.append({**core, "skill_hash": canonical_content_hash(core)})
    return skills


def _build_challenges(risks: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    by_code: dict[str, list[str]] = {}
    for row in risks:
        by_code.setdefault(row["code"], []).append(row["version_id"])
    challenges: list[dict[str, Any]] = []
    for code in sorted(by_code):
        target = _RISK_CHALLENGE_TARGET[code]
        version_ids = sorted(set(by_code[code]))
        seed = {
            "risk_code": code,
            "target_component": target,
            "version_ids": version_ids,
        }
        challenge_id = (
            "harness-challenge:" + canonical_content_hash(seed).split(":", 1)[1][:16]
        )
        core = {
            "challenge_id": challenge_id,
            "risk_code": code,
            "target_component": target,
            "version_ids": version_ids,
            "status": "next_epoch_human_review",
            "earliest_application_epoch": "next",
            "protected_component": target == "evaluation_policy",
            "historical_backtest_required": True,
            "holdout_validation_required": True,
            "automatic_application_allowed": False,
            "same_epoch_evaluator_mutation_allowed": False,
            "production_mutation_allowed": False,
        }
        challenges.append({**core, "challenge_hash": canonical_content_hash(core)})
    return challenges


def build_evolution_harness_audit(
    versions: Sequence[Mapping[str, Any]],
    *,
    backtests: Sequence[Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a four-layer, fail-closed harness audit.

    ``versions`` must be one ordered, parent-linked lineage of at least two
    rows produced by :func:`bind_version_evidence`.  ``backtests`` are optional
    rows produced by :func:`bind_skill_backtest`.  The function is pure and
    deterministic; the same normalized evidence produces the same audit hash.
    """

    normalized_versions = _normalize_versions(versions)
    normalized_backtests = _normalize_backtests(backtests)
    normalized_policy = _normalize_policy(policy)
    expected_policy_hash = canonical_content_hash(normalized_policy)
    mismatched_policy_versions = [
        row["version_id"]
        for row in normalized_versions
        if row["comparison_hashes"]["policy_hash"] != expected_policy_hash
    ]
    if mismatched_policy_versions:
        raise EvolutionHarnessError(
            "version policy_hash does not bind the normalized frozen policy: "
            + ", ".join(mismatched_policy_versions)
        )

    score_layer, score_risks = _build_score_layer(
        normalized_versions, normalized_policy
    )
    signal_layer, signal_risks = _build_signal_layer(
        normalized_versions, normalized_policy
    )
    behavior_layer, behavior_risks = _build_behavior_layer(
        normalized_versions, score_layer
    )
    version_layer, version_risks = _build_version_layer(normalized_versions)
    risks = _dedupe_risks(
        [*score_risks, *signal_risks, *behavior_risks, *version_risks]
    )
    blocking_codes = sorted({row["code"] for row in risks})
    controlled_progression_allowed = not blocking_codes
    progression = {
        "decision": (
            "eligible_for_human_review" if controlled_progression_allowed else "hold"
        ),
        "controlled_progression_allowed": controlled_progression_allowed,
        "automatic_progression_allowed": False,
        "automatic_production_mutation_allowed": False,
        "same_epoch_evaluator_mutation_allowed": False,
        "human_confirmation_required": True,
        "blocking_risk_codes": blocking_codes,
        "next_action": (
            "review_progression"
            if controlled_progression_allowed
            else "resolve_harness_blockers"
        ),
    }
    skills = _build_skills(normalized_backtests)
    core = {
        "schema_version": HARNESS_AUDIT_SCHEMA,
        "canonicalization": CANONICAL_JSON_PROFILE,
        "policy": normalized_policy,
        "evidence": {
            "epoch_id": normalized_versions[0]["epoch_id"],
            "version_count": len(normalized_versions),
            "version_ids": [row["version_id"] for row in normalized_versions],
            "evidence_hashes": [row["evidence_hash"] for row in normalized_versions],
            "latest_version_id": normalized_versions[-1]["version_id"],
            "backtest_hashes": [row["backtest_hash"] for row in normalized_backtests],
            "versions": deepcopy(normalized_versions),
            "backtests": deepcopy(normalized_backtests),
            "normalized_diagnostics_included": True,
            "raw_artifact_payloads_disclosed": False,
        },
        "layers": {
            "score": score_layer,
            "signal": signal_layer,
            "behavior": behavior_layer,
            "version": version_layer,
        },
        "risks": risks,
        "progression": progression,
        "next_epoch_harness_challenges": _build_challenges(risks),
        "skills": skills,
        "governance": {
            "scope": "offline_observational_audit",
            "model_calls_performed": 0,
            "network_calls_performed": 0,
            "filesystem_mutations_performed": 0,
            "automatic_production_mutation_allowed": False,
            "same_epoch_evaluator_mutation_allowed": False,
            "skills_are_advisory_only": True,
        },
    }
    return {**core, "audit_hash": canonical_content_hash(core)}


def validate_evolution_harness_audit(value: Any) -> dict[str, Any]:
    """Validate the canonical hash and non-bypassable governance invariants."""

    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"ok": False, "errors": ["audit_not_object"]}
    expected_fields = {
        "schema_version",
        "canonicalization",
        "policy",
        "evidence",
        "layers",
        "risks",
        "progression",
        "next_epoch_harness_challenges",
        "skills",
        "governance",
        "audit_hash",
    }
    if set(value) != expected_fields:
        errors.append("audit_fields_invalid")
    if value.get("schema_version") != HARNESS_AUDIT_SCHEMA:
        errors.append("schema_version_invalid")
    if value.get("canonicalization") != CANONICAL_JSON_PROFILE:
        errors.append("canonicalization_invalid")
    core = {key: deepcopy(item) for key, item in value.items() if key != "audit_hash"}
    try:
        expected_hash = canonical_content_hash(core)
    except Exception:
        expected_hash = None
        errors.append("audit_not_canonical")
    if value.get("audit_hash") != expected_hash:
        errors.append("audit_hash_invalid")

    normalized_policy: dict[str, Any] | None = None
    policy = value.get("policy")
    if not isinstance(policy, Mapping):
        errors.append("policy_invalid")
    else:
        try:
            normalized_policy = _normalize_policy(
                {key: policy[key] for key in _POLICY_OVERRIDE_KEYS}
            )
        except (EvolutionHarnessError, KeyError, TypeError, ValueError):
            normalized_policy = None
        if normalized_policy != dict(policy):
            errors.append("policy_contract_invalid")
        if policy.get("policy_id") != HARNESS_POLICY_ID:
            errors.append("policy_id_invalid")
        if policy.get("fixed_within_epoch") is not True:
            errors.append("policy_not_fixed_within_epoch")
        if policy.get("evaluator_change_window") != "next_epoch_only":
            errors.append("policy_evaluator_change_window_invalid")
        if policy.get("automatic_evaluator_mutation_allowed") is not False:
            errors.append("policy_evaluator_mutation_forbidden")
        if policy.get("automatic_production_mutation_allowed") is not False:
            errors.append("policy_production_mutation_forbidden")

    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        errors.append("evidence_invalid")
    else:
        expected_evidence_fields = {
            "epoch_id",
            "version_count",
            "version_ids",
            "evidence_hashes",
            "latest_version_id",
            "backtest_hashes",
            "versions",
            "backtests",
            "normalized_diagnostics_included",
            "raw_artifact_payloads_disclosed",
        }
        if set(evidence) != expected_evidence_fields:
            errors.append("evidence_fields_invalid")
        version_ids = evidence.get("version_ids")
        evidence_hashes = evidence.get("evidence_hashes")
        version_count = evidence.get("version_count")
        if (
            isinstance(version_count, bool)
            or not isinstance(version_count, int)
            or not 2 <= version_count <= MAX_VERSIONS
            or not isinstance(version_ids, list)
            or not isinstance(evidence_hashes, list)
            or len(version_ids) != version_count
            or len(evidence_hashes) != version_count
            or evidence.get("latest_version_id")
            != (version_ids[-1] if version_ids else None)
        ):
            errors.append("evidence_lineage_summary_invalid")
        if evidence.get("normalized_diagnostics_included") is not True:
            errors.append("normalized_diagnostics_missing")
        if evidence.get("raw_artifact_payloads_disclosed") is not False:
            errors.append("evidence_payload_disclosure_invalid")

    risks = value.get("risks")
    if not isinstance(risks, list):
        errors.append("risks_invalid")
        risks = []
    else:
        for row in risks:
            if not isinstance(row, Mapping) or row.get("code") not in RISK_CODES:
                errors.append("risk_code_invalid")
                break
            if row.get("severity") != "blocker":
                errors.append("risk_severity_invalid")
                break
    observed_codes = sorted(
        {str(row.get("code")) for row in risks if isinstance(row, Mapping)}
    )

    layers = value.get("layers")
    if not isinstance(layers, Mapping):
        errors.append("layers_invalid")
    else:
        score = layers.get("score")
        signal = layers.get("signal")
        behavior = layers.get("behavior")
        version = layers.get("version")
        if not isinstance(score, Mapping):
            errors.append("score_layer_invalid")
        else:
            if (
                score.get("score_contract_consistent") is False
                and "SCORE.CONTRACT_MISMATCH" not in observed_codes
            ):
                errors.append("score_contract_risk_missing")
            if (
                score.get("improved") is False
                and "SCORE.NO_IMPROVEMENT" not in observed_codes
            ):
                errors.append("score_improvement_risk_missing")
        if not isinstance(signal, Mapping) or not isinstance(
            signal.get("current"), Mapping
        ):
            errors.append("signal_layer_invalid")
        else:
            current_signal = signal["current"]
            if (
                current_signal.get("all_tied") is True
                and "SIGNAL.ALL_TIED" not in observed_codes
            ):
                errors.append("signal_tie_risk_missing")
            low_information_groups = current_signal.get("low_information_group_count")
            group_count = current_signal.get("group_count")
            policy_ratio = (
                policy.get("low_information_ratio")
                if isinstance(policy, Mapping)
                else None
            )
            if (
                isinstance(low_information_groups, int)
                and not isinstance(low_information_groups, bool)
                and isinstance(group_count, int)
                and not isinstance(group_count, bool)
                and group_count > 0
                and isinstance(policy_ratio, (int, float))
                and not isinstance(policy_ratio, bool)
                and low_information_groups / group_count >= float(policy_ratio)
                and "SIGNAL.LOW_INFORMATION" not in observed_codes
            ):
                errors.append("signal_information_risk_missing")
        if not isinstance(behavior, Mapping):
            errors.append("behavior_layer_invalid")
        else:
            required_behavior_risks = {
                "BEHAVIOR.CONTRACT_MISMATCH": behavior.get("contract_consistent")
                is False,
                "BEHAVIOR.THRESHOLD_VIOLATION": bool(
                    behavior.get("threshold_violations")
                ),
                "BEHAVIOR.REGRESSION": bool(behavior.get("regressed_metrics")),
                "BEHAVIOR.SCORE_DIVERGENCE": behavior.get("score_behavior_divergence")
                is True,
            }
            for code, required in required_behavior_risks.items():
                if required and code not in observed_codes:
                    errors.append("behavior_risk_missing:" + code)
        if not isinstance(version, Mapping):
            errors.append("version_layer_invalid")
        else:
            for transition in version.get("transitions") or []:
                if not isinstance(transition, Mapping):
                    errors.append("version_transition_invalid")
                    continue
                for field in transition.get("mismatch_fields") or []:
                    code = _COMPARISON_RISK_CODES.get(str(field))
                    if code and code not in observed_codes:
                        errors.append("version_comparison_risk_missing:" + code)
            for version_row in version.get("versions") or []:
                if not isinstance(version_row, Mapping):
                    errors.append("version_summary_invalid")
                    continue
                failed_checks = version_row.get("failed_integrity_checks") or []
                if failed_checks and "VERSION.INTEGRITY_FAILURE" not in observed_codes:
                    errors.append("version_integrity_risk_missing")
                if (
                    "git_leakage_absent" in failed_checks
                    and "VERSION.GIT_LEAKAGE" not in observed_codes
                ):
                    errors.append("version_git_leakage_risk_missing")
                if (
                    version_row.get("within_budget") is False
                    and "VERSION.COST_BUDGET_EXCEEDED" not in observed_codes
                ):
                    errors.append("version_cost_risk_missing")

    progression = value.get("progression")
    if not isinstance(progression, Mapping):
        errors.append("progression_invalid")
    else:
        for field in (
            "automatic_progression_allowed",
            "automatic_production_mutation_allowed",
            "same_epoch_evaluator_mutation_allowed",
        ):
            if progression.get(field) is not False:
                errors.append(f"{field}_forbidden")
        if progression.get("blocking_risk_codes") != observed_codes:
            errors.append("blocking_risk_codes_mismatch")
        allowed = not observed_codes
        if progression.get("controlled_progression_allowed") is not allowed:
            errors.append("controlled_progression_decision_invalid")
        expected_decision = "eligible_for_human_review" if allowed else "hold"
        if progression.get("decision") != expected_decision:
            errors.append("progression_decision_invalid")
        if progression.get("human_confirmation_required") is not True:
            errors.append("human_confirmation_required")

    challenges = value.get("next_epoch_harness_challenges")
    if not isinstance(challenges, list):
        errors.append("challenges_invalid")
    else:
        challenge_codes: list[str] = []
        for challenge in challenges:
            if not isinstance(challenge, Mapping):
                errors.append("challenge_invalid")
                continue
            challenge_codes.append(str(challenge.get("risk_code") or ""))
            challenge_core = {
                key: deepcopy(item)
                for key, item in challenge.items()
                if key != "challenge_hash"
            }
            try:
                expected_challenge_hash = canonical_content_hash(challenge_core)
            except Exception:
                expected_challenge_hash = None
            if challenge.get("challenge_hash") != expected_challenge_hash:
                errors.append("challenge_hash_invalid")
            if (
                challenge.get("status") != "next_epoch_human_review"
                or challenge.get("earliest_application_epoch") != "next"
                or challenge.get("historical_backtest_required") is not True
                or challenge.get("holdout_validation_required") is not True
                or challenge.get("automatic_application_allowed") is not False
                or challenge.get("same_epoch_evaluator_mutation_allowed") is not False
                or challenge.get("production_mutation_allowed") is not False
            ):
                errors.append("challenge_governance_invalid")
        if sorted(challenge_codes) != observed_codes:
            errors.append("challenge_risk_coverage_invalid")

    skills = value.get("skills")
    if not isinstance(skills, list):
        errors.append("skills_invalid")
    else:
        for skill in skills:
            if not isinstance(skill, Mapping):
                errors.append("skill_invalid")
                continue
            skill_core = {
                key: deepcopy(item)
                for key, item in skill.items()
                if key != "skill_hash"
            }
            try:
                expected_skill_hash = canonical_content_hash(skill_core)
            except Exception:
                expected_skill_hash = None
            if skill.get("skill_hash") != expected_skill_hash:
                errors.append("skill_hash_invalid")
            if skill.get("status") not in SKILL_STATUSES:
                errors.append("skill_status_invalid")
            if (
                skill.get("automatic_application_allowed") is not False
                or skill.get("production_promotion_allowed") is not False
                or skill.get("requires_fresh_evolution_gate") is not True
                or skill.get("historical_and_holdout_required") is not True
                or skill.get("independent_evaluator_required") is not True
            ):
                errors.append("skill_automatic_application_forbidden")
            reasons = skill.get("quarantine_reasons") or []
            if any(reason not in SKILL_QUARANTINE_REASONS for reason in reasons):
                errors.append("skill_quarantine_reason_invalid")
            validated_domains = skill.get("validated_domains")
            if not isinstance(validated_domains, list):
                errors.append("skill_validated_domains_invalid")
                validated_domains = []
            expected_status = (
                "cross_domain_validated"
                if len(validated_domains) >= 2
                else (
                    "domain_validated" if len(validated_domains) == 1 else "quarantined"
                )
            )
            if skill.get("status") != expected_status:
                errors.append("skill_validation_status_mismatch")
            for domain_result in skill.get("domain_results") or []:
                if not isinstance(domain_result, Mapping):
                    errors.append("skill_domain_result_invalid")
                    continue
                expected_validated = bool(
                    domain_result.get("historical_verified") is True
                    and domain_result.get("holdout_verified") is True
                    and domain_result.get("independent_evaluator_verified") is True
                    and not domain_result.get("quarantine_reasons")
                )
                if domain_result.get("validated") is not expected_validated:
                    errors.append("skill_domain_validation_mismatch")

    governance = value.get("governance")
    if not isinstance(governance, Mapping):
        errors.append("governance_invalid")
    else:
        if any(
            governance.get(field) is not False
            for field in (
                "automatic_production_mutation_allowed",
                "same_epoch_evaluator_mutation_allowed",
            )
        ):
            errors.append("governance_mutation_forbidden")
        if governance.get("skills_are_advisory_only") is not True:
            errors.append("governance_skills_not_advisory")
        if any(
            governance.get(field) != 0
            for field in (
                "model_calls_performed",
                "network_calls_performed",
                "filesystem_mutations_performed",
            )
        ):
            errors.append("governance_side_effects_invalid")

    # The audit carries bounded, normalized diagnostic evidence so validation
    # can replay the pure builder.  This prevents a caller from editing a layer,
    # deleting the corresponding blocker, and merely rehashing the outer object.
    if (
        isinstance(evidence, Mapping)
        and normalized_policy is not None
        and isinstance(evidence.get("versions"), list)
        and isinstance(evidence.get("backtests"), list)
    ):
        try:
            reconstructed = build_evolution_harness_audit(
                evidence["versions"],
                backtests=evidence["backtests"],
                policy={key: normalized_policy[key] for key in _POLICY_OVERRIDE_KEYS},
            )
        except (EvolutionHarnessError, KeyError, TypeError, ValueError):
            errors.append("audit_reconstruction_failed")
        else:
            if dict(value) != reconstructed:
                errors.append("audit_semantics_mismatch")
    else:
        errors.append("audit_reconstruction_unavailable")
    return {"ok": not errors, "errors": sorted(set(errors))}


__all__ = [
    "BACKTEST_SPLITS",
    "DEFAULT_POLICY",
    "EVALUATOR_AUTHORITIES",
    "EvolutionHarnessError",
    "HARNESS_AUDIT_SCHEMA",
    "HARNESS_POLICY_ID",
    "HARNESS_SKILL_SCHEMA",
    "RISK_CODES",
    "SKILL_STATUSES",
    "SKILL_TYPES",
    "bind_skill_backtest",
    "bind_version_evidence",
    "build_evolution_harness_audit",
    "build_harness_policy_hash",
    "validate_evolution_harness_audit",
]
