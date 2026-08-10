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
EXPERIMENT_PRIORITY_POLICY_VERSION = "1.0"
RESEARCH_REVIEW_INTERVAL = 5


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


def research_strategy_template() -> dict[str, Any]:
    """Return an editable example for experiment, quality, and boundary files."""

    return {
        "experiment_candidates": [
            {
                "candidate_id": "counterfactual-ablation",
                "summary": "Intervene on the proposed mediator and measure the outcome.",
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
    members = (portfolio.get("payload") or {}).get("members") or []
    priors = {
        str(item["hypothesis_id"]): float(item["prior_weight"]) for item in members
    }
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
    rows: list[dict[str, Any]] = []
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise ResearchGitError("each experiment candidate must be an object")
        candidate_id = _required_text(raw.get("candidate_id"), label="candidate id")
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
        rows.append(
            {
                "candidate_id": candidate_id,
                "summary": _required_text(
                    raw.get("summary"), label="candidate summary"
                ),
                "predictions": dict(sorted(resolved_predictions.items())),
                "expected_information_gain": round(eig, 6),
                "ratings": ratings,
                "utility_score": score,
                "discriminates": len(set(resolved_predictions.values())) > 1,
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
    policy = {
        "version": EXPERIMENT_PRIORITY_POLICY_VERSION,
        "score": "0.50*normalized_EIG + 0.15*novelty + 0.15*impact + 0.10*transfer - 0.10*cost - 0.05*risk - 0.05*redundancy",
        "ratings": "auditable ordinal integers from 0 to 4",
        "eig": "entropy reduction under the locked deterministic outcome partition",
    }
    core = {
        "protocol_kind": "information_value_experiment_priority",
        "portfolio_id": str(portfolio["object_id"]),
        "policy": policy,
        "policy_hash": canonical_content_hash(policy),
        "candidate_set": rows,
        "selected_candidate_id": rows[0]["candidate_id"],
    }
    payload = {**core, "priority_hash": canonical_content_hash(core)}
    result = repository.record(
        "experiment_priority",
        payload,
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": str(portfolio["object_id"]),
                "role": "portfolio",
            }
        ],
    )
    saved = _checkpoint_results(
        repository,
        result,
        (),
        stage="plan",
        subject=message or "rank experiments by expected information value",
        status="locked",
        commit=commit,
    )
    saved["ranking"] = payload
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
    core = {
        "protocol_kind": "causal_mechanism_model",
        "statement": _required_text(statement, label="mechanism statement"),
        "target_hypothesis_id": str(hypothesis["object_id"]),
        "mediators": [
            _required_text(value, label="mechanism mediator") for value in mediators
        ],
        "interventions": [
            _required_text(value, label="mechanism intervention")
            for value in interventions
        ],
        "rival_hypothesis_ids": sorted({str(item["object_id"]) for item in rivals}),
        "evidence_ids": sorted({str(item["object_id"]) for item in evidence}),
        "status": status,
    }
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
            "actor_id": _required_text(assessor_id, label="quality assessor"),
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
    boundary_results: list[ResearchObjectResult] = []
    normalized_rows: list[dict[str, Any]] = []
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
        core = {
            "protocol_kind": "claim_boundary_condition",
            "claim_id": str(claim["object_id"]),
            "dimension": _required_text(
                raw.get("dimension"), label="boundary dimension"
            ),
            "condition": _required_text(
                raw.get("condition"), label="boundary condition"
            ),
            "role": str(raw.get("role") or "development"),
            "status": str(raw.get("status") or "untested"),
            "evidence_ids": sorted({str(item["object_id"]) for item in evidence}),
        }
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
    if not normalized_rows:
        raise ResearchGitError("transfer matrix requires at least one boundary row")
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


def _targets(
    item: Mapping[str, Any], relation_types: set[str] | None = None
) -> set[str]:
    return {
        str(relation.get("target") or "")
        for relation in item.get("relations") or []
        if str(relation.get("target") or "")
        and (not relation_types or relation.get("type") in relation_types)
    }


def scan_research_anomalies(
    repo: str | Path,
    *,
    record: bool = False,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    objects = {str(item["object_id"]): item for item in repository.objects()}
    existing = {
        tuple(
            sorted(
                str(value)
                for value in (item.get("payload") or {}).get("source_ids") or []
            )
        )
        for item in objects.values()
        if item.get("kind") == "anomaly"
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
        key = tuple(sorted(candidate["source_ids"]))
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
    objects = repository.objects()
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
        for item in objects
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
    if not by_kind.get("discriminating_prediction"):
        gap(
            "discriminating_prediction_missing",
            "No prediction distinguishes competing explanations.",
            "Record outcomes that differ across portfolio members.",
        )
    if not by_kind.get("experiment_priority"):
        gap(
            "information_value_unranked",
            "Experiment candidates were not ranked by expected information value.",
            "Run `research program prioritize` before the next expensive experiment.",
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


def inspect_claim_depth(
    repo: str | Path,
    claim_id: str,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    claim = _resolve_kind(repository, claim_id, kinds={"claim"}, label="claim")
    objects = {str(item["object_id"]): item for item in repository.objects()}
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
    priorities = sorted(
        (
            item
            for item in objects.values()
            if item.get("kind") == "experiment_priority"
        ),
        key=lambda item: (str(item.get("created_at") or ""), str(item["object_id"])),
        reverse=True,
    )
    next_experiment = None
    if priorities:
        payload = priorities[0].get("payload") or {}
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
            next_experiment = {
                "priority_object_id": priorities[0]["object_id"],
                "candidate_id": selected_id,
                "summary": selected.get("summary"),
                "expected_information_gain": selected.get("expected_information_gain"),
                "utility_score": selected.get("utility_score"),
            }
    depth_level = str((claim.get("payload") or {}).get("depth_level") or "descriptive")
    valid_mechanisms = [
        object_id
        for object_id in mechanisms
        if objects[object_id].get("state") == "verified"
        and (objects[object_id].get("payload") or {}).get("status") == "validated"
        and supporting_set.intersection(
            (objects[object_id].get("payload") or {}).get("evidence_ids") or []
        )
    ]
    valid_quality = [
        object_id
        for object_id in quality
        if objects[object_id].get("state") == "verified"
        and (objects[object_id].get("payload") or {}).get("independent") is True
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
        "next_experiment": next_experiment,
        "gaps": gaps,
        "decision_ready": not gaps and not refuting,
    }


__all__ = [
    "EVIDENCE_QUALITY_DOMAINS",
    "EXPERIMENT_PRIORITY_POLICY_VERSION",
    "RESEARCH_REVIEW_INTERVAL",
    "inspect_claim_depth",
    "rank_experiment_candidates",
    "research_strategy_template",
    "review_research_program",
    "save_discriminating_prediction",
    "save_evidence_quality_assessment",
    "save_hypothesis_portfolio",
    "save_mechanism_model",
    "save_transfer_matrix",
    "scan_research_anomalies",
]
