"""Deep-research strategy objects and deterministic scientific review.

This layer does not ask a model to certify its own science.  It records the
competitive hypotheses, predictions, information-value policy, mechanisms,
quality assessments, anomalies, and applicability boundaries that make a
research program inspectable and falsifiable.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash

from .research_authority import require_independent_evaluator
from .research_git import ResearchGitError, ResearchObjectResult, create_checkpoint
from .research_vcs import ResearchRepository

EVIDENCE_QUALITY_DOMAINS = (
    "internal_validity",
    "measurement_reliability",
    "confounding",
    "statistical_power",
    "multiplicity",
    "preregistration_fidelity",
    "independence",
    "external_validity",
)
EXPERIMENT_PRIORITY_POLICY_VERSION = "2.0"
MECHANISM_VALIDATION_POLICY = "xscientist.intervention-lineage.v1"
TRANSFER_VALIDATION_POLICY = "xscientist.disjoint-boundary-evidence.v1"
POSTERIOR_UPDATE_POLICY = "xscientist.discrete-bayes.v1"
RESEARCH_REVIEW_INTERVAL = 5
RESEARCH_FOLLOWUP_POLICY = "xscientist.bounded-strategy-followup.v1"


def _required_text(value: Any, *, label: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ResearchGitError(f"{label} cannot be empty")
    return text


def _resolve_kind(
    repository: ResearchRepository,
    selector: str,
    *,
    kinds: set[str],
    label: str,
) -> dict[str, Any]:
    item = repository.get(selector)
    if str(item.get("kind") or "") not in kinds:
        raise ResearchGitError(f"{label} reference has wrong kind")
    return item


def _checkpoint_results(
    repository: ResearchRepository,
    primary: ResearchObjectResult,
    related: Sequence[ResearchObjectResult],
    *,
    stage: str,
    subject: str,
    status: str,
    commit: bool,
) -> dict[str, Any]:
    checkpoint = None
    if commit:
        checkpoint = create_checkpoint(
            repository.path,
            stage=stage,
            subject=subject,
            status=status,
            include=[
                item.path.relative_to(repository.path).as_posix()
                for item in (primary, *related)
            ],
        )
    return {
        "object": primary,
        "related": list(related),
        "checkpoint": checkpoint,
    }


def _ensure_stage_available(repository: ResearchRepository, *, commit: bool) -> None:
    if not commit:
        return
    staged = repository.status().get("research_stage", {}).get("paths") or []
    if staged:
        raise ResearchGitError(
            "deep-research recording requires an empty native stage; commit or "
            "unstage the existing paths first"
        )


def _targets(
    item: Mapping[str, Any], relation_types: set[str] | None = None
) -> set[str]:
    return {
        str(relation.get("target") or "")
        for relation in item.get("relations") or []
        if str(relation.get("target") or "")
        and (not relation_types or relation.get("type") in relation_types)
    }


def _effective_objects(repository: ResearchRepository) -> dict[str, dict[str, Any]]:
    objects = {str(item["object_id"]): item for item in repository.objects()}
    superseded = {
        target for item in objects.values() for target in _targets(item, {"supersedes"})
    }
    return {
        object_id: item
        for object_id, item in objects.items()
        if object_id not in superseded and item.get("state") != "superseded"
    }


def _latest_portfolio_object(
    repository: ResearchRepository,
    *,
    kind: str,
    portfolio_id: str,
) -> dict[str, Any] | None:
    rows = [
        item
        for item in _effective_objects(repository).values()
        if item.get("kind") == kind
        and (item.get("payload") or {}).get("portfolio_id") == portfolio_id
    ]
    return (
        max(
            rows,
            key=lambda item: (
                str(item.get("created_at") or ""),
                str(item.get("object_id") or ""),
            ),
        )
        if rows
        else None
    )


def _portfolio_weights(
    repository: ResearchRepository,
    portfolio: Mapping[str, Any],
) -> tuple[dict[str, float], str | None]:
    portfolio_id = str(portfolio["object_id"])
    posterior = _latest_portfolio_object(
        repository,
        kind="posterior_update",
        portfolio_id=portfolio_id,
    )
    if posterior is not None:
        weights = (posterior.get("payload") or {}).get("posterior_weights") or {}
        return {str(key): float(value) for key, value in weights.items()}, str(
            posterior["object_id"]
        )
    members = (portfolio.get("payload") or {}).get("members") or []
    return {
        str(item["hypothesis_id"]): float(item["prior_weight"]) for item in members
    }, None


def _normalised_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            _normalised_text(item) for _, item in sorted(value.items())
        ).lower()
    if isinstance(value, (list, tuple)):
        return " ".join(_normalised_text(item) for item in value).lower()
    return " ".join(str(value or "").lower().split())


def _evidence_attempt_lineage(
    repository: ResearchRepository,
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve evidence to completed attempts and their committed protocols."""

    pending = list(_targets(evidence, {"derived_from", "reproduces"}))
    seen: set[str] = set()
    attempts: dict[str, dict[str, Any]] = {}
    protocols: dict[str, dict[str, Any]] = {}
    while pending:
        selector = pending.pop()
        item = repository.get(selector)
        object_id = str(item["object_id"])
        if object_id in seen:
            continue
        seen.add(object_id)
        if item.get("kind") == "experiment_attempt":
            attempts[object_id] = item
            for target in _targets(item, {"depends_on", "consumes"}):
                linked = repository.get(target)
                if linked.get("kind") in {
                    "research_plan",
                    "experiment_design",
                    "preregistration",
                    "experiment_priority",
                }:
                    protocols[str(linked["object_id"])] = linked
        pending.extend(
            target
            for target in _targets(item, {"derived_from", "reproduces"})
            if target not in seen
        )
    return (
        [attempts[key] for key in sorted(attempts)],
        [protocols[key] for key in sorted(protocols)],
    )


def _mechanism_validation_receipt(
    repository: ResearchRepository,
    *,
    evidence: Sequence[Mapping[str, Any]],
    interventions: Sequence[str],
) -> dict[str, Any]:
    if not evidence:
        raise ResearchGitError("validated mechanism requires intervention evidence")
    if any(item.get("state") != "verified" for item in evidence):
        raise ResearchGitError("validated mechanism evidence must be verified")
    attempts: dict[str, dict[str, Any]] = {}
    protocols: dict[str, dict[str, Any]] = {}
    for item in evidence:
        linked_attempts, linked_protocols = _evidence_attempt_lineage(repository, item)
        if not linked_attempts:
            raise ResearchGitError(
                "validated mechanism evidence must derive from an experiment attempt"
            )
        attempts.update({str(row["object_id"]): row for row in linked_attempts})
        protocols.update({str(row["object_id"]): row for row in linked_protocols})
    if any(item.get("state") != "completed" for item in attempts.values()):
        raise ResearchGitError(
            "validated mechanism evidence must derive from completed attempts"
        )
    locked_protocol_ids = sorted(
        str(item["object_id"])
        for item in protocols.values()
        if item.get("state") == "locked"
        and item.get("kind")
        in {"preregistration", "experiment_design", "research_plan"}
    )
    if not locked_protocol_ids:
        raise ResearchGitError(
            "validated mechanism requires a locked preregistration or experiment design"
        )
    design_payloads = [
        item.get("payload") or {}
        for item in (*attempts.values(), *protocols.values())
        if item.get("kind")
        in {"experiment_attempt", "research_plan", "experiment_design"}
    ]
    design_text = _normalised_text(design_payloads)
    missing = [
        value for value in interventions if _normalised_text(value) not in design_text
    ]
    if missing:
        raise ResearchGitError(
            "validated mechanism interventions are not committed in the attempt/design: "
            + ", ".join(missing)
        )
    core = {
        "policy": MECHANISM_VALIDATION_POLICY,
        "evidence_ids": sorted(str(item["object_id"]) for item in evidence),
        "attempt_ids": sorted(attempts),
        "protocol_ids": sorted(protocols),
        "locked_protocol_ids": locked_protocol_ids,
        "matched_interventions": sorted(set(interventions)),
    }
    return {**core, "receipt_hash": canonical_content_hash(core)}


def _boundary_evidence_receipt(
    repository: ResearchRepository,
    *,
    evidence: Sequence[Mapping[str, Any]],
    condition: str,
    role: str,
) -> dict[str, Any]:
    if not evidence:
        raise ResearchGitError("supported boundary requires evidence")
    if any(item.get("state") != "verified" for item in evidence):
        raise ResearchGitError("supported boundary evidence must be verified")
    attempts: dict[str, dict[str, Any]] = {}
    protocols: dict[str, dict[str, Any]] = {}
    for item in evidence:
        linked_attempts, linked_protocols = _evidence_attempt_lineage(repository, item)
        if not linked_attempts:
            raise ResearchGitError(
                "supported boundary evidence must derive from an experiment attempt"
            )
        attempts.update({str(row["object_id"]): row for row in linked_attempts})
        protocols.update({str(row["object_id"]): row for row in linked_protocols})
    if any(item.get("state") != "completed" for item in attempts.values()):
        raise ResearchGitError("supported boundary attempts must be completed")
    dataset_hashes = sorted(
        {
            str(value)
            for item in attempts.values()
            for value in (item.get("provenance") or {}).get("dataset_hashes") or []
        }
    )
    if not dataset_hashes:
        raise ResearchGitError(
            "supported boundary evidence requires dataset hashes in attempt provenance"
        )
    committed = [
        item
        for item in (*attempts.values(), *protocols.values())
        if item.get("kind")
        in {"experiment_attempt", "research_plan", "experiment_design"}
    ]
    condition_matches = any(
        _normalised_text((item.get("payload") or {}).get("boundary_condition"))
        == _normalised_text(condition)
        and _normalised_text((item.get("payload") or {}).get("boundary_role"))
        == _normalised_text(role)
        for item in committed
    )
    if not condition_matches:
        raise ResearchGitError(
            "supported boundary condition and role must be committed in its attempt/design"
        )
    locked_protocols = [
        str(item["object_id"])
        for item in protocols.values()
        if item.get("state") == "locked"
        and item.get("kind")
        in {"preregistration", "experiment_design", "research_plan"}
    ]
    if not locked_protocols:
        raise ResearchGitError(
            "supported boundary evidence requires a locked preregistration or design"
        )
    core = {
        "policy": TRANSFER_VALIDATION_POLICY,
        "evidence_ids": sorted(str(item["object_id"]) for item in evidence),
        "attempt_ids": sorted(attempts),
        "locked_protocol_ids": sorted(locked_protocols),
        "dataset_hashes": dataset_hashes,
        "condition": condition,
        "role": role,
    }
    return {**core, "receipt_hash": canonical_content_hash(core)}


def research_strategy_template() -> dict[str, Any]:
    """Return an editable example for experiment, quality, and boundary files."""

    return {
        "experiment_candidates": [
            {
                "candidate_id": "counterfactual-ablation",
                "summary": "Intervene on the proposed mediator and measure the outcome.",
                "condition": "mediator M is ablated on the held-out split",
                "interventions": ["do(M=0)"],
                "predictions": {
                    "REPLACE_HYPOTHESIS_ID": "effect_disappears",
                    "REPLACE_RIVAL_HYPOTHESIS_ID": "effect_remains",
                },
                "novelty": 3,
                "impact": 4,
                "transfer_value": 3,
                "cost": 2,
                "risk": 1,
                "redundancy": 0,
            }
        ],
        "quality_assessment": {
            "domains": {name: "low_risk" for name in EVIDENCE_QUALITY_DOMAINS},
            "notes": {"internal_validity": "REPLACE_WITH_AUDIT_NOTE"},
            "independent": True,
        },
        "boundary_rows": [
            {
                "dimension": "domain",
                "condition": "held-out domain",
                "role": "transfer",
                "status": "supported",
                "evidence_ids": ["REPLACE_EVIDENCE_ID"],
            },
            {
                "dimension": "scale",
                "condition": "larger model or sample",
                "role": "scale",
                "status": "untested",
                "evidence_ids": [],
            },
        ],
    }


def save_hypothesis_portfolio(
    repo: str | Path,
    *,
    question: str,
    primary_id: str,
    alternative_ids: Sequence[str],
    null_id: str | None = None,
    prior_weights: Mapping[str, float] | None = None,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_stage_available(repository, commit=commit)
    primary = _resolve_kind(
        repository, primary_id, kinds={"hypothesis"}, label="primary hypothesis"
    )
    alternatives = [
        _resolve_kind(
            repository, selector, kinds={"hypothesis"}, label="alternative hypothesis"
        )
        for selector in alternative_ids
    ]
    null = (
        _resolve_kind(
            repository, null_id, kinds={"hypothesis"}, label="null hypothesis"
        )
        if null_id
        else None
    )
    rows = [(primary, "primary"), *[(item, "alternative") for item in alternatives]]
    if null is not None:
        rows.append((null, "null"))
    unique_ids = [str(item["object_id"]) for item, _ in rows]
    if len(set(unique_ids)) != len(unique_ids) or len(unique_ids) < 2:
        raise ResearchGitError(
            "hypothesis portfolio requires at least two distinct hypotheses"
        )
    raw_weights = dict(prior_weights or {})
    selectors = [primary_id, *alternative_ids, *([null_id] if null_id else [])]
    selector_by_id = {
        object_id: selector for object_id, selector in zip(unique_ids, selectors)
    }
    weights: list[float] = []
    for object_id in unique_ids:
        value = raw_weights.get(
            object_id,
            raw_weights.get(selector_by_id[object_id], 1.0),
        )
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ResearchGitError("hypothesis prior weights must be positive numbers")
        weights.append(float(value))
    total = sum(weights)
    members = [
        {
            "hypothesis_id": object_id,
            "role": role,
            "prior_weight": round(weight / total, 12),
        }
        for (item, role), object_id, weight in zip(rows, unique_ids, weights)
    ]
    # Absorb floating-point rounding into the final member deterministically.
    members[-1]["prior_weight"] = round(
        1.0 - sum(item["prior_weight"] for item in members[:-1]), 12
    )
    core = {
        "protocol_kind": "competitive_hypothesis_portfolio",
        "question": _required_text(question, label="portfolio question"),
        "members": members,
        "update_rule": "append immutable posterior/review objects; never rewrite priors",
    }
    payload = {**core, "portfolio_hash": canonical_content_hash(core)}
    result = repository.record(
        "hypothesis_portfolio",
        payload,
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": item["hypothesis_id"],
                "role": item["role"],
            }
            for item in members
        ],
    )
    return _checkpoint_results(
        repository,
        result,
        (),
        stage="ideation",
        subject=message or "lock competitive hypothesis portfolio",
        status="locked",
        commit=commit,
    )


def save_discriminating_prediction(
    repo: str | Path,
    *,
    portfolio_id: str,
    hypothesis_id: str,
    when: str,
    expected_outcome: str,
    distinguishes_from: Sequence[str],
    falsifier: str,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_stage_available(repository, commit=commit)
    portfolio = _resolve_kind(
        repository,
        portfolio_id,
        kinds={"hypothesis_portfolio"},
        label="prediction portfolio",
    )
    member_ids = {
        str(item.get("hypothesis_id") or "")
        for item in (portfolio.get("payload") or {}).get("members") or []
    }
    hypothesis = _resolve_kind(
        repository,
        hypothesis_id,
        kinds={"hypothesis"},
        label="prediction hypothesis",
    )
    resolved_hypothesis = str(hypothesis["object_id"])
    rivals = [
        _resolve_kind(
            repository, selector, kinds={"hypothesis"}, label="rival hypothesis"
        )
        for selector in distinguishes_from
    ]
    rival_ids = sorted({str(item["object_id"]) for item in rivals})
    if resolved_hypothesis not in member_ids or not set(rival_ids) <= member_ids:
        raise ResearchGitError("prediction hypotheses must belong to the portfolio")
    if resolved_hypothesis in rival_ids or not rival_ids:
        raise ResearchGitError("prediction requires at least one distinct rival")
    core = {
        "protocol_kind": "discriminating_prediction",
        "portfolio_id": str(portfolio["object_id"]),
        "hypothesis_id": resolved_hypothesis,
        "when": _required_text(when, label="prediction condition"),
        "expected_outcome": _required_text(expected_outcome, label="expected outcome"),
        "distinguishes_from": rival_ids,
        "falsifier": _required_text(falsifier, label="prediction falsifier"),
    }
    payload = {**core, "prediction_hash": canonical_content_hash(core)}
    result = repository.record(
        "discriminating_prediction",
        payload,
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": str(portfolio["object_id"]),
                "role": "portfolio",
            },
            {"type": "depends_on", "target": resolved_hypothesis, "role": "predictor"},
            *[
                {"type": "evaluates", "target": object_id, "role": "rival"}
                for object_id in rival_ids
            ],
        ],
    )
    return _checkpoint_results(
        repository,
        result,
        (),
        stage="plan",
        subject=message or "lock discriminating prediction",
        status="locked",
        commit=commit,
    )


def _entropy(weights: Sequence[float]) -> float:
    return -sum(value * math.log2(value) for value in weights if value > 0)


def _information_gain(
    priors: Mapping[str, float], predictions: Mapping[str, str]
) -> float:
    prior_entropy = _entropy(list(priors.values()))
    if prior_entropy <= 0:
        return 0.0
    groups: dict[str, list[float]] = {}
    for hypothesis_id, weight in priors.items():
        groups.setdefault(str(predictions[hypothesis_id]), []).append(weight)
    remaining = 0.0
    for weights in groups.values():
        probability = sum(weights)
        posterior = [weight / probability for weight in weights]
        remaining += probability * _entropy(posterior)
    return max(0.0, min(1.0, (prior_entropy - remaining) / prior_entropy))


def rank_experiment_candidates(
    repo: str | Path,
    *,
    portfolio_id: str,
    candidates: Sequence[Mapping[str, Any]],
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_stage_available(repository, commit=commit)
    portfolio = _resolve_kind(
        repository,
        portfolio_id,
        kinds={"hypothesis_portfolio"},
        label="experiment portfolio",
    )
    portfolio_object_id = str(portfolio["object_id"])
    priors, posterior_id = _portfolio_weights(repository, portfolio)
    if not candidates:
        raise ResearchGitError("experiment priority requires at least one candidate")
    rating_fields = (
        "novelty",
        "impact",
        "transfer_value",
        "cost",
        "risk",
        "redundancy",
    )
    effective = _effective_objects(repository)
    locked_predictions = [
        item
        for item in effective.values()
        if item.get("kind") == "discriminating_prediction"
        and item.get("state") == "locked"
        and (item.get("payload") or {}).get("portfolio_id") == portfolio_object_id
    ]
    rows: list[dict[str, Any]] = []
    candidate_specs: list[dict[str, Any]] = []
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise ResearchGitError("each experiment candidate must be an object")
        candidate_id = _required_text(raw.get("candidate_id"), label="candidate id")
        condition = _required_text(
            raw.get("condition"), label=f"candidate {candidate_id} condition"
        )
        predictions = raw.get("predictions")
        if not isinstance(predictions, Mapping):
            raise ResearchGitError(
                f"candidate {candidate_id} predictions must be an object"
            )
        resolved_predictions: dict[str, str] = {}
        for selector, outcome in predictions.items():
            hypothesis = _resolve_kind(
                repository,
                str(selector),
                kinds={"hypothesis"},
                label="candidate prediction hypothesis",
            )
            object_id = str(hypothesis["object_id"])
            if object_id not in priors:
                raise ResearchGitError(
                    f"candidate {candidate_id} predicts outside its portfolio"
                )
            resolved_predictions[object_id] = _required_text(
                outcome, label="candidate predicted outcome"
            )
        if set(resolved_predictions) != set(priors):
            raise ResearchGitError(
                f"candidate {candidate_id} must predict every portfolio hypothesis"
            )
        raw_prediction_ids = raw.get("prediction_ids") or {}
        if not isinstance(raw_prediction_ids, Mapping):
            raise ResearchGitError(
                f"candidate {candidate_id} prediction_ids must be an object"
            )
        bound_predictions: dict[str, str] = {}
        for hypothesis_id, outcome in resolved_predictions.items():
            selector = raw_prediction_ids.get(hypothesis_id)
            if selector is None:
                selector = next(
                    (
                        key
                        for key, value in raw_prediction_ids.items()
                        if str(repository.get(str(key))["object_id"]) == hypothesis_id
                    ),
                    None,
                )
            matches = []
            if selector:
                selected_prediction = _resolve_kind(
                    repository,
                    str(selector),
                    kinds={"discriminating_prediction"},
                    label="candidate locked prediction",
                )
                matches = [selected_prediction]
            else:
                matches = [
                    item
                    for item in locked_predictions
                    if (item.get("payload") or {}).get("hypothesis_id") == hypothesis_id
                    and _normalised_text((item.get("payload") or {}).get("when"))
                    == _normalised_text(condition)
                    and _normalised_text(
                        (item.get("payload") or {}).get("expected_outcome")
                    )
                    == _normalised_text(outcome)
                ]
            matches = [
                item
                for item in matches
                if item.get("state") == "locked"
                and (item.get("payload") or {}).get("portfolio_id")
                == portfolio_object_id
                and (item.get("payload") or {}).get("hypothesis_id") == hypothesis_id
                and _normalised_text((item.get("payload") or {}).get("when"))
                == _normalised_text(condition)
                and _normalised_text(
                    (item.get("payload") or {}).get("expected_outcome")
                )
                == _normalised_text(outcome)
            ]
            if len(matches) != 1:
                raise ResearchGitError(
                    f"candidate {candidate_id} requires exactly one locked prediction "
                    f"for portfolio hypothesis {hypothesis_id}"
                )
            bound_predictions[hypothesis_id] = str(matches[0]["object_id"])
        ratings: dict[str, int] = {}
        for field in rating_fields:
            value = raw.get(field, 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 4
            ):
                raise ResearchGitError(
                    f"candidate {candidate_id} {field} must be an integer from 0 to 4"
                )
            ratings[field] = value
        eig = _information_gain(priors, resolved_predictions)
        benefit = (
            0.5 * eig
            + 0.15 * ratings["novelty"] / 4
            + 0.15 * ratings["impact"] / 4
            + 0.1 * ratings["transfer_value"] / 4
        )
        penalty = (
            0.1 * ratings["cost"] / 4
            + 0.05 * ratings["risk"] / 4
            + 0.05 * ratings["redundancy"] / 4
        )
        score = round(max(0.0, min(1.0, benefit - penalty)), 6)
        summary = _required_text(raw.get("summary"), label="candidate summary")
        interventions = [
            _required_text(value, label="candidate intervention")
            for value in (raw.get("interventions") or [])
        ]
        row = {
            "candidate_id": candidate_id,
            "summary": summary,
            "condition": condition,
            "predictions": dict(sorted(resolved_predictions.items())),
            "prediction_ids": dict(sorted(bound_predictions.items())),
            "expected_information_gain": round(eig, 6),
            "ratings": ratings,
            "utility_score": score,
            "discriminates": len(set(resolved_predictions.values())) > 1,
        }
        rows.append(row)
        candidate_specs.append(
            {
                "candidate_id": candidate_id,
                "summary": summary,
                "condition": condition,
                "interventions": interventions,
                "predictions": dict(sorted(resolved_predictions.items())),
                "prediction_ids": dict(sorted(bound_predictions.items())),
            }
        )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ResearchGitError("experiment candidate ids must be unique")
    rows.sort(
        key=lambda item: (
            -item["utility_score"],
            -item["expected_information_gain"],
            item["candidate_id"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1
        row["decision_reason"] = (
            "highest deterministic information-value utility"
            if rank == 1
            else "lower information value or higher cost/risk/redundancy"
        )
    spec_by_id = {item["candidate_id"]: item for item in candidate_specs}
    design_results: list[ResearchObjectResult] = []
    for row in rows:
        spec = spec_by_id[row["candidate_id"]]
        design_core = {
            "protocol_kind": "competitive_experiment_candidate",
            "portfolio_id": portfolio_object_id,
            **spec,
        }
        design_payload = {
            **design_core,
            "design_hash": canonical_content_hash(design_core),
        }
        design = repository.record(
            "experiment_design",
            design_payload,
            state="locked",
            relations=[
                {
                    "type": "depends_on",
                    "target": portfolio_object_id,
                    "role": "portfolio",
                },
                *[
                    {
                        "type": "depends_on",
                        "target": prediction_id,
                        "role": "locked_prediction",
                    }
                    for prediction_id in spec["prediction_ids"].values()
                ],
            ],
        )
        design_results.append(design)
        row["design_object_id"] = design.object_id
    policy = {
        "version": EXPERIMENT_PRIORITY_POLICY_VERSION,
        "score": "0.50*normalized_EIG + 0.15*novelty + 0.15*impact + 0.10*transfer - 0.10*cost - 0.05*risk - 0.05*redundancy",
        "ratings": "auditable ordinal integers from 0 to 4",
        "eig": "entropy reduction under the locked deterministic outcome partition",
        "prediction_binding": "every outcome is bound to an immutable locked prediction",
    }
    core = {
        "protocol_kind": "information_value_experiment_priority",
        "portfolio_id": portfolio_object_id,
        "prior_source_id": posterior_id or portfolio_object_id,
        "prior_weights": dict(sorted(priors.items())),
        "policy": policy,
        "policy_hash": canonical_content_hash(policy),
        "candidate_set": rows,
        "selected_candidate_id": rows[0]["candidate_id"],
        "selected_design_id": rows[0]["design_object_id"],
    }
    payload = {**core, "priority_hash": canonical_content_hash(core)}
    result = repository.record(
        "experiment_priority",
        payload,
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": portfolio_object_id,
                "role": "portfolio",
            },
            *(
                [
                    {
                        "type": "depends_on",
                        "target": posterior_id,
                        "role": "posterior_prior",
                    }
                ]
                if posterior_id
                else []
            ),
            *[
                {
                    "type": "selects" if row["selected"] else "depends_on",
                    "target": row["design_object_id"],
                    "role": "candidate_design",
                }
                for row in rows
            ],
        ],
    )
    saved = _checkpoint_results(
        repository,
        result,
        design_results,
        stage="plan",
        subject=message or "rank experiments by expected information value",
        status="locked",
        commit=commit,
    )
    saved["ranking"] = payload
    return saved


def save_posterior_update(
    repo: str | Path,
    *,
    portfolio_id: str,
    priority_id: str,
    attempt_id: str,
    evidence_id: str,
    observed_outcome: str,
    likelihoods: Mapping[str, float],
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Append one evidence-bound Bayesian update without certifying its truth."""

    repository = ResearchRepository(repo)
    _ensure_stage_available(repository, commit=commit)
    portfolio = _resolve_kind(
        repository,
        portfolio_id,
        kinds={"hypothesis_portfolio"},
        label="posterior portfolio",
    )
    resolved_portfolio_id = str(portfolio["object_id"])
    priority = _resolve_kind(
        repository,
        priority_id,
        kinds={"experiment_priority"},
        label="posterior experiment priority",
    )
    priority_payload = priority.get("payload") or {}
    if (
        priority.get("state") != "locked"
        or priority_payload.get("portfolio_id") != resolved_portfolio_id
    ):
        raise ResearchGitError("posterior priority is not locked for this portfolio")
    attempt = _resolve_kind(
        repository,
        attempt_id,
        kinds={"experiment_attempt"},
        label="posterior experiment attempt",
    )
    if attempt.get("state") != "completed":
        raise ResearchGitError("posterior update requires a completed attempt")
    attempt_targets = _targets(attempt, {"depends_on", "consumes"})
    selected_design_id = str(priority_payload.get("selected_design_id") or "")
    if (
        str(priority["object_id"]) not in attempt_targets
        or selected_design_id not in attempt_targets
    ):
        raise ResearchGitError(
            "posterior attempt must consume the selected priority and design"
        )
    evidence = _resolve_kind(
        repository,
        evidence_id,
        kinds={"evidence", "effect_estimate", "evidence_synthesis", "reproduction"},
        label="posterior evidence",
    )
    if evidence.get("state") not in {"completed", "verified"}:
        raise ResearchGitError(
            "posterior update requires completed or verified evidence"
        )
    linked_attempts, _ = _evidence_attempt_lineage(repository, evidence)
    if str(attempt["object_id"]) not in {
        str(item["object_id"]) for item in linked_attempts
    }:
        raise ResearchGitError("posterior evidence does not derive from its attempt")
    reused = [
        item
        for item in _effective_objects(repository).values()
        if item.get("kind") == "posterior_update"
        and (item.get("payload") or {}).get("portfolio_id") == resolved_portfolio_id
        and (item.get("payload") or {}).get("evidence_id") == str(evidence["object_id"])
    ]
    if reused:
        raise ResearchGitError("posterior evidence has already updated this portfolio")

    prior_weights, prior_posterior_id = _portfolio_weights(repository, portfolio)
    resolved_likelihoods: dict[str, float] = {}
    for selector, value in likelihoods.items():
        hypothesis = _resolve_kind(
            repository,
            str(selector),
            kinds={"hypothesis"},
            label="posterior likelihood hypothesis",
        )
        object_id = str(hypothesis["object_id"])
        if object_id not in prior_weights:
            raise ResearchGitError("posterior likelihood is outside its portfolio")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 1
        ):
            raise ResearchGitError("posterior likelihoods must be numbers from 0 to 1")
        resolved_likelihoods[object_id] = float(value)
    if set(resolved_likelihoods) != set(prior_weights):
        raise ResearchGitError(
            "posterior update requires one likelihood per hypothesis"
        )
    unnormalised = {
        hypothesis_id: prior_weights[hypothesis_id]
        * resolved_likelihoods[hypothesis_id]
        for hypothesis_id in prior_weights
    }
    normalizer = sum(unnormalised.values())
    if normalizer <= 0:
        raise ResearchGitError(
            "posterior likelihoods assign zero mass to every hypothesis"
        )
    posterior_weights = {
        hypothesis_id: round(value / normalizer, 12)
        for hypothesis_id, value in sorted(unnormalised.items())
    }
    final_id = sorted(posterior_weights)[-1]
    posterior_weights[final_id] = round(
        1.0
        - sum(
            value
            for hypothesis_id, value in posterior_weights.items()
            if hypothesis_id != final_id
        ),
        12,
    )
    normalized_outcome = _required_text(observed_outcome, label="observed outcome")
    observation_core = {
        "protocol_kind": "competitive_experiment_observation",
        "measurement": normalized_outcome,
        "attempt_id": str(attempt["object_id"]),
        "evidence_id": str(evidence["object_id"]),
    }
    observation_payload = {
        **observation_core,
        "observation_hash": canonical_content_hash(observation_core),
    }
    observation = repository.record(
        "observation",
        observation_payload,
        state="completed",
        relations=[
            {"type": "derived_from", "target": str(attempt["object_id"])},
            {
                "type": "depends_on",
                "target": str(evidence["object_id"]),
                "role": "evidence_anchor",
            },
        ],
    )
    core = {
        "protocol_kind": "evidence_bound_posterior_update",
        "policy": POSTERIOR_UPDATE_POLICY,
        "portfolio_id": resolved_portfolio_id,
        "priority_id": str(priority["object_id"]),
        "selected_design_id": selected_design_id,
        "attempt_id": str(attempt["object_id"]),
        "observation_id": observation.object_id,
        "evidence_id": str(evidence["object_id"]),
        "observed_outcome": normalized_outcome,
        "prior_source_id": prior_posterior_id or resolved_portfolio_id,
        "prior_weights": dict(sorted(prior_weights.items())),
        "likelihoods": dict(sorted(resolved_likelihoods.items())),
        "posterior_weights": posterior_weights,
        "evidence_state": str(evidence.get("state") or ""),
        "epistemic_status": "agent_computed_draft",
    }
    payload = {**core, "update_hash": canonical_content_hash(core)}
    result = repository.record(
        "posterior_update",
        payload,
        state="completed",
        relations=[
            {
                "type": "depends_on",
                "target": resolved_portfolio_id,
                "role": "portfolio",
            },
            {
                "type": "consumes",
                "target": str(priority["object_id"]),
                "role": "priority",
            },
            {"type": "observes", "target": observation.object_id},
            {
                "type": "consumes",
                "target": str(evidence["object_id"]),
                "role": "evidence",
            },
            *(
                [
                    {
                        "type": "depends_on",
                        "target": prior_posterior_id,
                        "role": "prior_posterior",
                    }
                ]
                if prior_posterior_id
                else []
            ),
        ],
    )
    saved = _checkpoint_results(
        repository,
        result,
        [observation],
        stage="evidence",
        subject=message or "update hypothesis portfolio from observed evidence",
        status="completed",
        commit=commit,
    )
    saved["posterior"] = payload
    saved["observation"] = observation
    return saved


def save_mechanism_model(
    repo: str | Path,
    *,
    hypothesis_id: str,
    statement: str,
    mediators: Sequence[str],
    interventions: Sequence[str],
    rival_hypothesis_ids: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
    status: str = "proposed",
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_stage_available(repository, commit=commit)
    hypothesis = _resolve_kind(
        repository, hypothesis_id, kinds={"hypothesis"}, label="mechanism hypothesis"
    )
    rivals = [
        _resolve_kind(repository, item, kinds={"hypothesis"}, label="mechanism rival")
        for item in rival_hypothesis_ids
    ]
    evidence = [
        _resolve_kind(
            repository,
            item,
            kinds={"evidence", "effect_estimate", "evidence_synthesis", "reproduction"},
            label="mechanism evidence",
        )
        for item in evidence_ids
    ]
    normalized_mediators = [
        _required_text(value, label="mechanism mediator") for value in mediators
    ]
    normalized_interventions = [
        _required_text(value, label="mechanism intervention") for value in interventions
    ]
    rival_ids = sorted({str(item["object_id"]) for item in rivals})
    if status == "validated" and not rival_ids:
        raise ResearchGitError(
            "validated mechanism requires at least one tested rival explanation"
        )
    validation = (
        _mechanism_validation_receipt(
            repository,
            evidence=evidence,
            interventions=normalized_interventions,
        )
        if status == "validated"
        else None
    )
    core = {
        "protocol_kind": "causal_mechanism_model",
        "statement": _required_text(statement, label="mechanism statement"),
        "target_hypothesis_id": str(hypothesis["object_id"]),
        "mediators": normalized_mediators,
        "interventions": normalized_interventions,
        "rival_hypothesis_ids": rival_ids,
        "evidence_ids": sorted({str(item["object_id"]) for item in evidence}),
        "status": status,
    }
    if validation is not None:
        core["validation"] = validation
    payload = {**core, "mechanism_hash": canonical_content_hash(core)}
    result = repository.record(
        "mechanism_model",
        payload,
        state="verified" if status == "validated" else "completed",
        relations=[
            {
                "type": "depends_on",
                "target": str(hypothesis["object_id"]),
                "role": "target_hypothesis",
            },
            *[
                {"type": "evaluates", "target": str(item["object_id"]), "role": "rival"}
                for item in rivals
            ],
            *[
                {
                    "type": "derived_from",
                    "target": str(item["object_id"]),
                    "role": "mechanism_test",
                }
                for item in evidence
            ],
        ],
    )
    return _checkpoint_results(
        repository,
        result,
        (),
        stage="evidence",
        subject=message or "record causal mechanism model",
        status=result.state,
        commit=commit,
    )


def _quality_grade(domains: Mapping[str, str]) -> str:
    values = list(domains.values())
    if "high_risk" in values:
        return "critical"
    if "not_assessed" in values or values.count("some_concerns") >= 3:
        return "weak"
    if "some_concerns" in values:
        return "moderate"
    return "strong"


def save_evidence_quality_assessment(
    repo: str | Path,
    *,
    evidence_id: str,
    domains: Mapping[str, str],
    notes: Mapping[str, str] | None = None,
    independent: bool,
    assessor_id: str,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_stage_available(repository, commit=commit)
    evidence = _resolve_kind(
        repository,
        evidence_id,
        kinds={
            "evidence",
            "passage_evidence",
            "effect_estimate",
            "evidence_synthesis",
            "reproduction",
        },
        label="quality evidence",
    )
    normalized_assessor = _required_text(assessor_id, label="quality assessor")
    independence = (
        require_independent_evaluator(
            repository,
            evaluator_id=normalized_assessor,
            target_ids=[str(evidence["object_id"])],
            label="independent evidence quality assessment",
        )
        if independent
        else None
    )
    normalized_domains = {
        name: str(domains.get(name) or "not_assessed")
        for name in EVIDENCE_QUALITY_DOMAINS
    }
    core = {
        "protocol_kind": "evidence_quality_assessment",
        "evidence_id": str(evidence["object_id"]),
        "domains": normalized_domains,
        "notes": {
            str(key): _required_text(value, label="quality note")
            for key, value in sorted((notes or {}).items())
            if str(value).strip()
        },
        "overall_grade": _quality_grade(normalized_domains),
        "independent": bool(independent),
    }
    if independence is not None:
        core["independence_receipt"] = independence
    payload = {**core, "assessment_hash": canonical_content_hash(core)}
    result = repository.record(
        "evidence_quality",
        payload,
        state="verified" if independent else "completed",
        relations=[
            {
                "type": "evaluates",
                "target": str(evidence["object_id"]),
                "role": "quality",
            }
        ],
        actor={
            "actor_id": normalized_assessor,
            "authority": "independent_evaluator" if independent else "research_agent",
        },
    )
    return _checkpoint_results(
        repository,
        result,
        (),
        stage="review",
        subject=message or "assess evidence quality and bias",
        status=result.state,
        commit=commit,
    )


def save_transfer_matrix(
    repo: str | Path,
    *,
    claim_id: str,
    rows: Sequence[Mapping[str, Any]],
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    _ensure_stage_available(repository, commit=commit)
    claim = _resolve_kind(repository, claim_id, kinds={"claim"}, label="boundary claim")
    prepared_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ResearchGitError("each boundary row must be an object")
        evidence = [
            _resolve_kind(
                repository,
                str(selector),
                kinds={
                    "evidence",
                    "passage_evidence",
                    "effect_estimate",
                    "evidence_synthesis",
                    "reproduction",
                },
                label="boundary evidence",
            )
            for selector in (raw.get("evidence_ids") or [])
        ]
        role = str(raw.get("role") or "development")
        status = str(raw.get("status") or "untested")
        if role not in {"development", "transfer", "heldout", "scale"}:
            raise ResearchGitError("boundary role is invalid")
        if status not in {"supported", "refuted", "mixed", "untested"}:
            raise ResearchGitError("boundary status is invalid")
        dimension = _required_text(raw.get("dimension"), label="boundary dimension")
        condition = _required_text(raw.get("condition"), label="boundary condition")
        if status == "untested" and evidence:
            raise ResearchGitError("untested boundary cannot cite result evidence")
        validation = (
            _boundary_evidence_receipt(
                repository,
                evidence=evidence,
                condition=condition,
                role=role,
            )
            if status != "untested"
            else None
        )
        core: dict[str, Any] = {
            "protocol_kind": "claim_boundary_condition",
            "claim_id": str(claim["object_id"]),
            "dimension": dimension,
            "condition": condition,
            "role": role,
            "status": status,
            "evidence_ids": sorted({str(item["object_id"]) for item in evidence}),
        }
        if validation is not None:
            core["validation"] = validation
        prepared_rows.append((core, evidence))
    if not prepared_rows:
        raise ResearchGitError("transfer matrix requires at least one boundary row")

    tested_cores = [core for core, _ in prepared_rows if core["status"] != "untested"]
    evidence_sets = [set(core["evidence_ids"]) for core in tested_cores]
    attempt_sets = [set(core["validation"]["attempt_ids"]) for core in tested_cores]
    independent_evidence = all(
        not left.intersection(right)
        for index, left in enumerate(evidence_sets)
        for right in evidence_sets[index + 1 :]
    )
    independent_attempts = all(
        not left.intersection(right)
        for index, left in enumerate(attempt_sets)
        for right in attempt_sets[index + 1 :]
    )
    development_datasets = {
        value
        for core in tested_cores
        if core["role"] == "development"
        for value in core["validation"]["dataset_hashes"]
    }
    heldout_datasets = {
        value
        for core in tested_cores
        if core["role"] in {"transfer", "heldout"}
        for value in core["validation"]["dataset_hashes"]
    }
    heldout_dataset_disjoint = bool(
        development_datasets
        and heldout_datasets
        and not development_datasets.intersection(heldout_datasets)
    )
    independence_checks = {
        "policy": TRANSFER_VALIDATION_POLICY,
        "evidence_sets_pairwise_disjoint": independent_evidence,
        "attempt_sets_pairwise_disjoint": independent_attempts,
        "development_heldout_datasets_disjoint": heldout_dataset_disjoint,
    }

    boundary_results: list[ResearchObjectResult] = []
    normalized_rows: list[dict[str, Any]] = []
    for core, evidence in prepared_rows:
        payload = {**core, "boundary_hash": canonical_content_hash(core)}
        boundary = repository.record(
            "boundary_condition",
            payload,
            state="verified" if core["status"] == "supported" else "completed",
            relations=[
                {
                    "type": "depends_on",
                    "target": str(claim["object_id"]),
                    "role": "claim",
                },
                *[
                    {
                        "type": "derived_from",
                        "target": str(item["object_id"]),
                        "role": "boundary_evidence",
                    }
                    for item in evidence
                ],
            ],
        )
        boundary_results.append(boundary)
        normalized_rows.append(
            {
                **core,
                "boundary_id": boundary.object_id,
                "boundary_hash": payload["boundary_hash"],
            }
        )
    tested = [row for row in normalized_rows if row["status"] != "untested"]
    dimensions = sorted({row["dimension"] for row in tested})
    transfer_rows = [
        row for row in tested if row["role"] in {"transfer", "heldout", "scale"}
    ]
    transfer_ready = bool(
        len(tested) >= 3
        and len(dimensions) >= 2
        and any(row["status"] == "supported" for row in transfer_rows)
        and all(row["status"] == "supported" for row in tested)
        and independence_checks["evidence_sets_pairwise_disjoint"]
        and independence_checks["attempt_sets_pairwise_disjoint"]
        and independence_checks["development_heldout_datasets_disjoint"]
    )
    coverage = {
        "row_count": len(normalized_rows),
        "tested_count": len(tested),
        "dimension_count": len(dimensions),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in normalized_rows).items())
        ),
        "transfer_condition_count": len(transfer_rows),
    }
    core = {
        "protocol_kind": "claim_transfer_matrix",
        "claim_id": str(claim["object_id"]),
        "rows": normalized_rows,
        "coverage": coverage,
        "independence_checks": independence_checks,
        "transfer_ready": transfer_ready,
    }
    payload = {**core, "matrix_hash": canonical_content_hash(core)}
    matrix = repository.record(
        "transfer_matrix",
        payload,
        state="verified" if transfer_ready else "completed",
        relations=[
            {"type": "depends_on", "target": str(claim["object_id"]), "role": "claim"},
            *[
                {"type": "depends_on", "target": item.object_id, "role": "boundary"}
                for item in boundary_results
            ],
        ],
    )
    saved = _checkpoint_results(
        repository,
        matrix,
        boundary_results,
        stage="review",
        subject=message or "map claim boundaries and transfer conditions",
        status=matrix.state,
        commit=commit,
    )
    saved["matrix"] = payload
    return saved


def scan_research_anomalies(
    repo: str | Path,
    *,
    record: bool = False,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    objects = _effective_objects(repository)
    existing = {
        (
            str((item.get("payload") or {}).get("anomaly_type") or ""),
            *sorted(
                str(value)
                for value in (item.get("payload") or {}).get("source_ids") or []
            ),
        )
        for item in objects.values()
        if item.get("kind") == "anomaly"
        and (item.get("payload") or {}).get("status") == "open"
    }
    candidates: list[dict[str, Any]] = []
    for object_id, item in objects.items():
        if item.get("kind") == "experiment_attempt" and item.get("state") in {
            "failed",
            "timed_out",
            "cancelled",
        }:
            candidates.append(
                {
                    "anomaly_type": "failed_or_incomplete_experiment",
                    "summary": f"{item.get('state')} attempt requires scientific or execution triage",
                    "severity": "medium",
                    "source_ids": [object_id],
                    "status": "open",
                }
            )
        for relation in item.get("relations") or []:
            if relation.get("type") not in {
                "refutes",
                "contradicts",
                "qualified_refutes",
            }:
                continue
            target = str(relation.get("target") or "")
            if target in objects:
                candidates.append(
                    {
                        "anomaly_type": "conflicting_evidence",
                        "summary": "Recorded evidence conflicts with an existing scientific object",
                        "severity": "high",
                        "source_ids": sorted({object_id, target}),
                        "status": "open",
                    }
                )
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            candidate["anomaly_type"],
            *sorted(candidate["source_ids"]),
        )
        if key not in existing:
            unique[key] = candidate
    pending = [unique[key] for key in sorted(unique)]
    recorded: list[ResearchObjectResult] = []
    if record:
        for candidate in pending:
            core = {"protocol_kind": "research_anomaly", **candidate}
            payload = {**core, "anomaly_hash": canonical_content_hash(core)}
            recorded.append(
                repository.record(
                    "anomaly",
                    payload,
                    state="completed",
                    relations=[
                        {
                            "type": "observes",
                            "target": source_id,
                            "role": "anomaly_source",
                        }
                        for source_id in candidate["source_ids"]
                    ],
                )
            )
    return {
        "candidate_count": len(pending),
        "candidates": pending,
        "recorded": recorded,
    }


def review_research_program(
    repo: str | Path,
    *,
    record: bool = False,
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    if record:
        _ensure_stage_available(repository, commit=commit)
    all_objects = repository.objects()
    effective = _effective_objects(repository)
    objects = list(effective.values())
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in objects:
        by_kind.setdefault(str(item["kind"]), []).append(item)
    anomalies = scan_research_anomalies(repository.path, record=record)
    last_review_at = max(
        (
            str(item.get("created_at") or "")
            for item in by_kind.get("research_review", [])
        ),
        default="",
    )
    new_object_count = sum(
        str(item.get("created_at") or "") > last_review_at
        and item.get("kind") != "research_review"
        for item in all_objects
    )
    gaps: list[dict[str, str]] = []
    recommendations: list[dict[str, str]] = []

    def gap(code: str, message_text: str, action: str) -> None:
        gaps.append({"code": code, "message": message_text})
        recommendations.append({"gap": code, "action": action})

    if len(by_kind.get("hypothesis", [])) < 2:
        gap(
            "single_hypothesis_bias",
            "The program has fewer than two explicit hypotheses.",
            "Record a rival or null hypothesis before choosing more experiments.",
        )
    if not by_kind.get("hypothesis_portfolio"):
        gap(
            "portfolio_missing",
            "No locked competitive hypothesis portfolio exists.",
            "Create `research program portfolio` with primary and rival hypotheses.",
        )
    for portfolio in by_kind.get("hypothesis_portfolio", []):
        portfolio_id = str(portfolio["object_id"])
        member_ids = {
            str(item.get("hypothesis_id") or "")
            for item in (portfolio.get("payload") or {}).get("members") or []
        }
        predictions = [
            item
            for item in by_kind.get("discriminating_prediction", [])
            if (item.get("payload") or {}).get("portfolio_id") == portfolio_id
        ]
        predicted_ids = {
            str((item.get("payload") or {}).get("hypothesis_id") or "")
            for item in predictions
        }
        if predicted_ids != member_ids:
            gap(
                f"discriminating_prediction_missing:{portfolio_id}",
                "A portfolio lacks locked predictions for every competing hypothesis.",
                "Record one condition-matched prediction per portfolio member.",
            )
        priorities = [
            item
            for item in by_kind.get("experiment_priority", [])
            if (item.get("payload") or {}).get("portfolio_id") == portfolio_id
        ]
        if not priorities:
            gap(
                f"information_value_unranked:{portfolio_id}",
                "A portfolio has no evidence-bound experiment priority.",
                "Run `research program prioritize` before the next expensive experiment.",
            )
            continue
        executable_priorities = [
            item
            for item in priorities
            if (item.get("payload") or {}).get("selected_design_id")
            and (item.get("payload") or {}).get("prior_weights")
        ]
        if not executable_priorities:
            gap(
                f"strategy_v2_upgrade_required:{portfolio_id}",
                "The portfolio only has legacy priorities without executable designs.",
                "Append v2 locked predictions, candidate designs, and a new priority.",
            )
            continue
        selected_priority_ids = {
            str(item["object_id"]) for item in executable_priorities
        }
        completed_attempts = [
            item
            for item in by_kind.get("experiment_attempt", [])
            if item.get("state") == "completed"
            and selected_priority_ids.intersection(_targets(item, {"consumes"}))
        ]
        if completed_attempts:
            attempt_ids = {str(item["object_id"]) for item in completed_attempts}
            observed_evidence = [
                item
                for item in by_kind.get("evidence", [])
                if attempt_ids.intersection(_targets(item, {"derived_from"}))
            ]
            posterior_evidence = {
                str((item.get("payload") or {}).get("evidence_id") or "")
                for item in by_kind.get("posterior_update", [])
                if (item.get("payload") or {}).get("portfolio_id") == portfolio_id
            }
            if any(
                str(item["object_id"]) not in posterior_evidence
                for item in observed_evidence
            ):
                gap(
                    f"posterior_update_missing:{portfolio_id}",
                    "Completed competitive evidence has not updated its portfolio.",
                    "Record an observation and likelihood-bound posterior update.",
                )
    if not by_kind.get("mechanism_model"):
        gap(
            "mechanism_unmodeled",
            "No intervention-testable mechanism model is recorded.",
            "Record mediators, interventions, rival explanations, and mechanism evidence.",
        )
    if by_kind.get("evidence") and not by_kind.get("evidence_quality"):
        gap(
            "evidence_quality_unassessed",
            "Evidence exists without a structured quality and bias assessment.",
            "Assess internal validity, measurement, confounding, power, and independence.",
        )
    if by_kind.get("claim") and not by_kind.get("transfer_matrix"):
        gap(
            "claim_boundaries_unmapped",
            "Claims exist without an explicit boundary and transfer matrix.",
            "Test domain, scale, population, resource, and failure conditions.",
        )
    if anomalies["candidate_count"]:
        gap(
            "open_anomalies",
            f"{anomalies['candidate_count']} unresolved anomalies need explanation.",
            "Promote failures and contradictions into mechanism or boundary tests.",
        )
    review_due = bool(
        new_object_count >= RESEARCH_REVIEW_INTERVAL
        or anomalies["candidate_count"]
        or not by_kind.get("research_review")
    )
    summary = (
        "Research strategy review found no structural depth gaps."
        if not gaps
        else f"Research strategy review found {len(gaps)} depth gaps."
    )
    core = {
        "protocol_kind": "periodic_research_strategy_review",
        "summary": summary,
        "review_due": review_due,
        "objects_since_previous_review": new_object_count,
        "gaps": gaps,
        "recommended_actions": recommendations,
        "frontier_counts": {
            kind: len(by_kind.get(kind, []))
            for kind in (
                "hypothesis",
                "hypothesis_portfolio",
                "discriminating_prediction",
                "experiment_priority",
                "experiment_design",
                "posterior_update",
                "anomaly",
                "mechanism_model",
                "evidence_quality",
                "boundary_condition",
                "transfer_matrix",
                "claim",
            )
        },
    }
    payload = {**core, "review_hash": canonical_content_hash(core)}
    result: ResearchObjectResult | None = None
    checkpoint = None
    related = list(anomalies["recorded"])
    if record:
        result = repository.record(
            "research_review",
            payload,
            state="completed" if not gaps else "draft",
            relations=[
                {"type": "evaluates", "target": item.object_id, "role": "open_anomaly"}
                for item in related
            ],
        )
        saved = _checkpoint_results(
            repository,
            result,
            related,
            stage="review",
            subject=message or "review research depth and direction",
            status=result.state,
            commit=commit,
        )
        checkpoint = saved["checkpoint"]
    return {
        "report": payload,
        "object": result,
        "related": related,
        "checkpoint": checkpoint,
    }


def record_research_followup_queue(
    repo: str | Path,
    *,
    review: Mapping[str, Any] | None = None,
    review_id: str | None = None,
    max_actions: int = 1,
    commit: bool = False,
) -> dict[str, Any]:
    """Turn strategy gaps into a finite, inspectable next-action queue.

    The queue deliberately proposes rather than fabricates experiment outcomes.
    A later model-backed or human step must turn a proposal into a locked design;
    the one-action budget and stop conditions prevent an unbounded self-loop.
    """

    if max_actions < 0:
        raise ResearchGitError("research follow-up max_actions cannot be negative")
    repository = ResearchRepository(repo)
    resolved_review_id = None
    if review_id:
        resolved_review = _resolve_kind(
            repository,
            review_id,
            kinds={"research_review"},
            label="research follow-up review",
        )
        resolved_review_id = str(resolved_review["object_id"])
        recorded_report = dict(resolved_review.get("payload") or {})
        if review is not None and review.get("review_hash") != recorded_report.get(
            "review_hash"
        ):
            raise ResearchGitError(
                "research follow-up report does not match the bound review object"
            )
        report = recorded_report
    else:
        report = dict(review or review_research_program(repo, record=False)["report"])
    recommendations = report.get("recommended_actions") or []
    if not isinstance(recommendations, list):
        raise ResearchGitError("research review recommendations must be a list")
    review_hash = _required_text(
        report.get("review_hash"), label="research follow-up review hash"
    )
    existing_items = [
        item
        for item in _effective_objects(repository).values()
        if item.get("kind") == "action_proposal"
        and (item.get("payload") or {}).get("policy") == RESEARCH_FOLLOWUP_POLICY
        and (item.get("payload") or {}).get("execution_status") == "queued"
        and (
            resolved_review_id in _targets(item)
            if resolved_review_id
            else (item.get("payload") or {}).get("review_hash") == review_hash
        )
    ]
    existing = {
        str((item.get("payload") or {}).get("gap") or "") for item in existing_items
    }
    remaining_actions = max(0, max_actions - len(existing))
    if commit and remaining_actions:
        _ensure_stage_available(repository, commit=True)
    created: list[ResearchObjectResult] = []
    for recommendation in recommendations:
        if len(created) >= remaining_actions:
            break
        if not isinstance(recommendation, Mapping):
            continue
        gap_code = _required_text(recommendation.get("gap"), label="follow-up gap")
        if gap_code in existing:
            continue
        action = _required_text(recommendation.get("action"), label="follow-up action")
        core = {
            "protocol_kind": "bounded_research_strategy_followup",
            "policy": RESEARCH_FOLLOWUP_POLICY,
            "review_hash": review_hash,
            "gap": gap_code,
            "action": action,
            "summary": f"Resolve research strategy gap: {gap_code}",
            "execution_status": "queued",
            "budget": {"max_new_designs": 1, "max_new_experiments": 1},
            "stop_conditions": [
                "the gap is resolved by immutable research objects",
                "the selected test has no positive expected information gain",
                "the declared project budget is exhausted",
            ],
            "requires_locked_design_before_execution": True,
        }
        created.append(
            repository.record(
                "action_proposal",
                {**core, "proposal_hash": canonical_content_hash(core)},
                state="draft",
                relations=(
                    [
                        {
                            "type": "derived_from",
                            "target": resolved_review_id,
                            "role": "strategy_review",
                        }
                    ]
                    if resolved_review_id
                    else []
                ),
                actor={
                    "actor_id": "research-strategy-controller",
                    "authority": "research_agent",
                },
            )
        )
        existing.add(gap_code)
    checkpoint = None
    if created and commit:
        checkpoint = repository.commit(
            stage="plan",
            subject="queue bounded scientific strategy follow-ups",
            status="draft",
        )
    created_rows = [
        {
            "object_id": item.object_id,
            "gap": (repository.get(item.object_id).get("payload") or {}).get("gap"),
            "action": (repository.get(item.object_id).get("payload") or {}).get(
                "action"
            ),
        }
        for item in created
    ]
    existing_rows = [
        {
            "object_id": str(item["object_id"]),
            "gap": (item.get("payload") or {}).get("gap"),
            "action": (item.get("payload") or {}).get("action"),
        }
        for item in existing_items
    ]
    return {
        "policy": RESEARCH_FOLLOWUP_POLICY,
        "review_id": resolved_review_id,
        "review_hash": review_hash,
        "max_actions": max_actions,
        "queued": created_rows,
        "active": [*existing_rows, *created_rows],
        "checkpoint": checkpoint,
    }


def inspect_claim_depth(
    repo: str | Path,
    claim_id: str,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    claim = _resolve_kind(repository, claim_id, kinds={"claim"}, label="claim")
    objects = _effective_objects(repository)
    resolved_claim_id = str(claim["object_id"])
    direct = _targets(claim)
    supporting = sorted(
        {
            object_id
            for object_id, item in objects.items()
            if resolved_claim_id in _targets(item, {"supports", "qualified_supports"})
        }
        | {
            object_id
            for object_id in direct
            if objects.get(object_id, {}).get("kind")
            in {"evidence", "passage_evidence", "inference", "evidence_synthesis"}
        }
    )
    supporting_set = set(supporting)
    refuting = sorted(
        object_id
        for object_id, item in objects.items()
        if resolved_claim_id
        in _targets(
            item,
            {"refutes", "qualified_refutes", "contradicts", "challenges_inference"},
        )
    )
    quality = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "evidence_quality"
        and supporting_set.intersection(_targets(item, {"evaluates"}))
    )
    boundaries = sorted(
        {
            object_id
            for object_id, item in objects.items()
            if item.get("kind") in {"boundary_condition", "transfer_matrix"}
            and resolved_claim_id in _targets(item)
        }
        | {
            object_id
            for object_id in direct
            if objects.get(object_id, {}).get("kind")
            in {"boundary_condition", "transfer_matrix"}
        }
    )
    mechanisms = sorted(
        object_id
        for object_id in direct
        if objects.get(object_id, {}).get("kind") == "mechanism_model"
    )
    hypothesis_ids = {
        target
        for target in direct
        if objects.get(target, {}).get("kind") == "hypothesis"
    }
    for evidence_object_id in (*supporting, *refuting):
        hypothesis_ids.update(
            target
            for target in _targets(objects.get(evidence_object_id, {}))
            if objects.get(target, {}).get("kind") == "hypothesis"
        )
    for mechanism_id in mechanisms:
        hypothesis_id = str(
            (objects[mechanism_id].get("payload") or {}).get("target_hypothesis_id")
            or ""
        )
        if hypothesis_id:
            hypothesis_ids.add(hypothesis_id)
    portfolio_ids = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "hypothesis_portfolio"
        and hypothesis_ids.intersection(
            str(member.get("hypothesis_id") or "")
            for member in (item.get("payload") or {}).get("members") or []
        )
    )
    priorities = sorted(
        (
            item
            for item in objects.values()
            if item.get("kind") == "experiment_priority"
            and (item.get("payload") or {}).get("portfolio_id") in portfolio_ids
        ),
        key=lambda item: (str(item.get("created_at") or ""), str(item["object_id"])),
        reverse=True,
    )
    latest_priorities: dict[str, dict[str, Any]] = {}
    for item in priorities:
        portfolio_id = str((item.get("payload") or {}).get("portfolio_id") or "")
        latest_priorities.setdefault(portfolio_id, item)
    next_experiments = []
    for portfolio_id in portfolio_ids:
        priority = latest_priorities.get(portfolio_id)
        if priority is None:
            continue
        payload = priority.get("payload") or {}
        selected_id = payload.get("selected_candidate_id")
        selected = next(
            (
                item
                for item in payload.get("candidate_set") or []
                if item.get("candidate_id") == selected_id
            ),
            None,
        )
        if selected:
            next_experiments.append(
                {
                    "portfolio_id": portfolio_id,
                    "priority_object_id": priority["object_id"],
                    "candidate_id": selected_id,
                    "design_object_id": selected.get("design_object_id"),
                    "summary": selected.get("summary"),
                    "expected_information_gain": selected.get(
                        "expected_information_gain"
                    ),
                    "utility_score": selected.get("utility_score"),
                }
            )
    next_experiment = next_experiments[0] if len(next_experiments) == 1 else None
    depth_level = str((claim.get("payload") or {}).get("depth_level") or "descriptive")
    valid_mechanisms = [
        object_id
        for object_id in mechanisms
        if objects[object_id].get("state") == "verified"
        and (objects[object_id].get("payload") or {}).get("status") == "validated"
        and (objects[object_id].get("payload") or {}).get("validation")
        and supporting_set.intersection(
            (objects[object_id].get("payload") or {}).get("evidence_ids") or []
        )
    ]
    valid_quality = [
        object_id
        for object_id in quality
        if objects[object_id].get("state") == "verified"
        and (objects[object_id].get("payload") or {}).get("independent") is True
        and (objects[object_id].get("payload") or {}).get("independence_receipt")
        and (objects[object_id].get("payload") or {}).get("overall_grade")
        in {"strong", "moderate"}
    ]
    claim_payload = claim.get("payload") or {}
    valid_transfer = []
    for object_id in boundaries:
        item = objects[object_id]
        if item.get("kind") != "transfer_matrix":
            continue
        matrix_payload = item.get("payload") or {}
        matrix_claim = objects.get(str(matrix_payload.get("claim_id") or ""), {})
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
    return {
        "claim_id": resolved_claim_id,
        "statement": (claim.get("payload") or {}).get("statement"),
        "state": claim.get("state"),
        "depth_level": depth_level,
        "supporting_ids": supporting,
        "refuting_ids": refuting,
        "quality_assessment_ids": quality,
        "mechanism_ids": mechanisms,
        "boundary_ids": boundaries,
        "portfolio_ids": portfolio_ids,
        "next_experiments": next_experiments,
        "next_experiment": next_experiment,
        "gaps": gaps,
        "decision_ready": not gaps and not refuting,
    }


__all__ = [
    "EVIDENCE_QUALITY_DOMAINS",
    "EXPERIMENT_PRIORITY_POLICY_VERSION",
    "RESEARCH_REVIEW_INTERVAL",
    "RESEARCH_FOLLOWUP_POLICY",
    "inspect_claim_depth",
    "rank_experiment_candidates",
    "record_research_followup_queue",
    "research_strategy_template",
    "review_research_program",
    "save_discriminating_prediction",
    "save_evidence_quality_assessment",
    "save_hypothesis_portfolio",
    "save_mechanism_model",
    "save_posterior_update",
    "save_transfer_matrix",
    "scan_research_anomalies",
]
