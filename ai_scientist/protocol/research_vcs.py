"""Canonical Research VCS object construction and validation."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from .canonical_json import canonical_content_hash
from .hashing import content_hash
from .schemas import load_schema

RESEARCH_OBJECT_SCHEMA = "xscientist.research-object.v1"
LEGACY_RESEARCH_OBJECT_IDENTITY_PROFILE = "xscientist.research-object-identity.v1"
RESEARCH_OBJECT_IDENTITY_PROFILE = "xscientist.research-object-identity.v2"
SUPPORTED_RESEARCH_OBJECT_IDENTITY_PROFILES = (
    LEGACY_RESEARCH_OBJECT_IDENTITY_PROFILE,
    RESEARCH_OBJECT_IDENTITY_PROFILE,
)
RESEARCH_SEMANTIC_PROFILE_SCHEMA = "xscientist.research-semantic-profile.v1"

CORE_RESEARCH_OBJECT_KINDS = (
    "question",
    "search_plan",
    "search_receipt",
    "source_snapshot",
    "passage_evidence",
    "hypothesis",
    "preregistration",
    "research_plan",
    "experiment_attempt",
    "observation",
    "metric",
    "evidence",
    "claim",
    "review",
    "gate_decision",
    "manuscript",
    "reproduction",
    "agent_candidate",
    "agent_evaluation",
    "context_snapshot",
)
EPISTEMIC_RESEARCH_OBJECT_KINDS = (
    "inference",
    "warrant",
    "assumption",
    "method",
    "estimand",
    "effect_estimate",
    "protocol_deviation",
    "sensitivity_analysis",
    "risk_of_bias",
    "evidence_synthesis",
    "challenge",
    "source_update",
    "context_robustness",
)
AUTONOMOUS_RESEARCH_OBJECT_KINDS = (
    "research_goal",
    "action_proposal",
    "experiment_design",
    "resource_budget",
    "stopping_decision",
    "novelty_check",
    "evaluation_blinding",
    "human_escalation",
)
STRATEGY_RESEARCH_OBJECT_KINDS_V1 = (
    "hypothesis_portfolio",
    "discriminating_prediction",
    "experiment_priority",
    "anomaly",
    "research_review",
    "mechanism_model",
    "evidence_quality",
    "boundary_condition",
    "transfer_matrix",
)
STRATEGY_RESEARCH_OBJECT_KINDS = (
    *STRATEGY_RESEARCH_OBJECT_KINDS_V1,
    "posterior_update",
)
ROLLOUT_RESEARCH_OBJECT_KINDS = ("research_rollout",)
RESEARCH_OBJECT_KINDS = (
    *CORE_RESEARCH_OBJECT_KINDS,
    *EPISTEMIC_RESEARCH_OBJECT_KINDS,
    *AUTONOMOUS_RESEARCH_OBJECT_KINDS,
    *STRATEGY_RESEARCH_OBJECT_KINDS,
    *ROLLOUT_RESEARCH_OBJECT_KINDS,
)
RESEARCH_OBJECT_STATES = (
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
)
RESEARCH_RELATION_TYPES = (
    "depends_on",
    "supports",
    "refutes",
    "supersedes",
    "reproduces",
    "contradicts",
    "derived_from",
    "evaluates",
    "promotes",
    "retrieves",
    "cites",
    "quotes",
    "observes",
    "generated_by",
    "qualified_supports",
    "qualified_refutes",
    "attests",
    "uses_context",
    "has_premise",
    "uses_method",
    "under_assumption",
    "addresses_estimand",
    "has_effect_estimate",
    "challenges_inference",
    "derived_by",
    "qualifies",
    "updates",
    "invalidates",
    "selects",
    "rejects",
    "consumes",
    "produces",
)
# Only relations whose target semantics are unambiguous are constrained here.
# Broad graph relations such as ``depends_on`` and ``derived_from`` intentionally
# remain kind-polymorphic.
RESEARCH_RELATION_TARGET_KINDS = {
    "quotes": ("source_snapshot",),
    "uses_context": ("context_snapshot",),
    "uses_method": ("method",),
    "under_assumption": ("assumption",),
    "addresses_estimand": ("estimand",),
    "has_effect_estimate": ("effect_estimate",),
    "challenges_inference": ("inference",),
}
RESEARCH_AUTHORITIES = (
    "research_agent",
    "recorder",
    "independent_evaluator",
    "deterministic_gate",
    "human",
)


def _profile_descriptor(
    uri: str, version: str, kinds: Sequence[str], relations: Sequence[str]
) -> dict[str, Any]:
    core = {
        "schema": RESEARCH_SEMANTIC_PROFILE_SCHEMA,
        "uri": uri,
        "version": version,
        "kinds": sorted(set(kinds)),
        "relations": sorted(set(relations)),
    }
    return {**core, "schema_digest": canonical_content_hash(core)}


_CORE_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/research-core/v1",
    "1.0.0",
    CORE_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
_EPISTEMIC_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/epistemic-argument/v1",
    "1.0.0",
    EPISTEMIC_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
_AUTONOMOUS_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/autonomous-research/v1",
    "1.0.0",
    AUTONOMOUS_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
_STRATEGY_PROFILE_V1 = _profile_descriptor(
    "https://xscientist.io/profiles/research-strategy/v1",
    "1.0.0",
    STRATEGY_RESEARCH_OBJECT_KINDS_V1,
    RESEARCH_RELATION_TYPES,
)
_STRATEGY_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/research-strategy/v2",
    "2.0.0",
    STRATEGY_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
_ROLLOUT_PROFILE = _profile_descriptor(
    "https://xscientist.io/profiles/research-rollout/v1",
    "1.0.0",
    ROLLOUT_RESEARCH_OBJECT_KINDS,
    RESEARCH_RELATION_TYPES,
)
BUILTIN_RESEARCH_PROFILES = {
    profile["uri"]: profile
    for profile in (
        _CORE_PROFILE,
        _EPISTEMIC_PROFILE,
        _AUTONOMOUS_PROFILE,
        _STRATEGY_PROFILE_V1,
        _STRATEGY_PROFILE,
        _ROLLOUT_PROFILE,
    )
}


def _default_profile_for_kind(kind: str) -> dict[str, Any] | None:
    if kind in CORE_RESEARCH_OBJECT_KINDS:
        return deepcopy(_CORE_PROFILE)
    if kind in EPISTEMIC_RESEARCH_OBJECT_KINDS:
        return deepcopy(_EPISTEMIC_PROFILE)
    if kind in AUTONOMOUS_RESEARCH_OBJECT_KINDS:
        return deepcopy(_AUTONOMOUS_PROFILE)
    if kind in STRATEGY_RESEARCH_OBJECT_KINDS:
        return deepcopy(_STRATEGY_PROFILE)
    if kind in ROLLOUT_RESEARCH_OBJECT_KINDS:
        return deepcopy(_ROLLOUT_PROFILE)
    return None


def _normalise_semantic_profile(
    kind: str, profile: Mapping[str, Any] | None
) -> dict[str, Any]:
    default = _default_profile_for_kind(kind)
    if profile is None:
        if default is None:
            raise ResearchObjectError(
                "extension research object kinds require semantic_profile metadata"
            )
        return default
    if not isinstance(profile, Mapping):
        raise ResearchObjectError("semantic_profile must be a mapping")
    row = deepcopy(dict(profile))
    uri = str(row.get("uri") or "").strip()
    version = str(row.get("version") or "").strip()
    schema_digest = str(row.get("schema_digest") or "").strip()
    if not uri.startswith(("https://", "http://", "urn:")):
        raise ResearchObjectError("semantic_profile uri must be an absolute URI")
    if not version:
        raise ResearchObjectError("semantic_profile version is required")
    if not _is_sha256(schema_digest):
        raise ResearchObjectError(
            "semantic_profile schema_digest must use sha256:<64 hex>"
        )
    if set(row) != {"schema", "uri", "version", "kinds", "relations", "schema_digest"}:
        raise ResearchObjectError("semantic_profile has unsupported fields")
    if row.get("schema") != RESEARCH_SEMANTIC_PROFILE_SCHEMA:
        raise ResearchObjectError("semantic_profile schema is invalid")
    if not isinstance(row.get("kinds"), list) or kind not in row["kinds"]:
        raise ResearchObjectError("semantic_profile does not declare the object kind")
    if not isinstance(row.get("relations"), list):
        raise ResearchObjectError("semantic_profile relations must be an array")
    expected_digest = canonical_content_hash(
        {key: value for key, value in row.items() if key != "schema_digest"}
    )
    if schema_digest != expected_digest:
        raise ResearchObjectError("semantic_profile schema_digest mismatch")
    builtin = BUILTIN_RESEARCH_PROFILES.get(uri)
    if builtin is not None and row != builtin:
        raise ResearchObjectError("built-in semantic_profile metadata mismatch")
    return row


def research_profile_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Report whether an object's semantic profile has a local validator."""

    profile = payload.get("semantic_profile")
    if not isinstance(profile, Mapping):
        return {"declared": False, "validator_available": False, "builtin": False}
    uri = str(profile.get("uri") or "")
    builtin = BUILTIN_RESEARCH_PROFILES.get(uri)
    return {
        "declared": True,
        "uri": uri,
        "version": str(profile.get("version") or ""),
        "schema_digest": str(profile.get("schema_digest") or ""),
        "validator_available": builtin is not None,
        "builtin": builtin is not None,
    }


class ResearchObjectError(ValueError):
    """A Research VCS object is malformed or has lost content integrity."""


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


_PAYLOAD_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "question": ("text", "question", "pool_hash"),
    "search_plan": ("queries", "question", "search_plan_hash"),
    "search_receipt": ("receipt_hash", "provider", "candidates"),
    "source_snapshot": (
        "source_hash",
        "content_hash",
        "metadata_hash",
        "doi",
        "pmid",
        "arxiv_id",
        "title",
    ),
    "passage_evidence": ("passage_hash", "quote_hash", "locator"),
    "hypothesis": ("statement", "core_hypothesis", "title"),
    "preregistration": ("status", "registration_id", "hypothesis_id"),
    "research_plan": ("plan_id", "tasks", "summary", "hypothesis"),
    "experiment_attempt": ("status", "attempt_hash"),
    "observation": ("measurement", "result", "output_hash", "metrics"),
    "metric": ("name", "metric", "value"),
    "evidence": (
        "result",
        "summary",
        "measurement",
        "metrics",
        "metric",
        "effect",
        "ara_manifest_hash",
    ),
    "claim": ("statement", "text", "claim", "claim_hash"),
    "review": (
        "summary",
        "status",
        "decision",
        "report_hash",
        "judgment_hash",
        "grade_hash",
    ),
    "gate_decision": ("decision", "claim_promotion_allowed"),
    "manuscript": ("title", "status", "final", "idea_idx"),
    "reproduction": (
        "checkpoint_hash",
        "receipt_hash",
        "reproduction_level",
        "verdict",
    ),
    "agent_candidate": (
        "candidate_id",
        "candidate_hash",
        "candidate",
        "promotion",
        "summary",
        "version",
    ),
    "agent_evaluation": (
        "candidate_id",
        "candidate",
        "summary",
        "status",
        "verdict",
        "decision",
        "gate_hash",
    ),
    "context_snapshot": (
        "context_hash",
        "source_closure_hash",
        "memory_snapshot_hash",
    ),
    "inference": ("statement", "conclusion", "inference_hash"),
    "warrant": ("statement", "rule", "warrant_hash"),
    "assumption": ("statement", "assumption", "assumption_hash"),
    "method": ("name", "description", "method_hash"),
    "estimand": ("outcome", "summary", "estimand_hash"),
    "effect_estimate": ("estimate", "value", "effect_hash"),
    "protocol_deviation": ("description", "reason", "deviation_hash"),
    "sensitivity_analysis": ("summary", "result", "analysis_hash"),
    "risk_of_bias": ("assessment", "domain", "assessment_hash"),
    "evidence_synthesis": ("summary", "conclusion", "synthesis_hash"),
    "challenge": ("statement", "reason", "challenge_hash"),
    "source_update": ("status", "update_type", "update_hash"),
    "context_robustness": ("status", "result", "robustness_hash"),
    "research_goal": ("question", "objective", "goal_hash"),
    "action_proposal": ("action", "summary", "proposal_hash"),
    "experiment_design": ("summary", "design", "design_hash"),
    "resource_budget": ("budget", "limits", "budget_hash", "allocation_hash"),
    "stopping_decision": ("decision", "reason", "decision_hash"),
    "novelty_check": ("verdict", "summary", "check_hash"),
    "evaluation_blinding": ("policy", "summary", "blinding_hash"),
    "human_escalation": ("reason", "question", "escalation_hash"),
    "hypothesis_portfolio": ("question", "portfolio_hash"),
    "discriminating_prediction": ("expected_outcome", "prediction_hash"),
    "experiment_priority": ("selected_candidate_id", "priority_hash"),
    "anomaly": ("summary", "anomaly_hash"),
    "research_review": ("summary", "review_hash"),
    "mechanism_model": ("statement", "mechanism_hash"),
    "evidence_quality": ("evidence_id", "assessment_hash"),
    "boundary_condition": ("condition", "boundary_hash"),
    "transfer_matrix": ("claim_id", "matrix_hash"),
    "posterior_update": ("portfolio_id", "update_hash"),
    "research_rollout": ("task_id", "task_hash", "rollout_hash"),
}


def _strategy_protocol_issues(
    kind: str,
    payload: Mapping[str, Any],
    semantic_profile: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the deeper-research strategy profile without model judgment."""

    issues: list[str] = []
    legacy_v1 = (
        str((semantic_profile or {}).get("uri") or "")
        == "https://xscientist.io/profiles/research-strategy/v1"
    )
    required: dict[str, tuple[str, ...]] = {
        "hypothesis_portfolio": ("question", "members", "portfolio_hash"),
        "discriminating_prediction": (
            "portfolio_id",
            "hypothesis_id",
            "when",
            "expected_outcome",
            "distinguishes_from",
            "falsifier",
            "prediction_hash",
        ),
        "experiment_priority": (
            "portfolio_id",
            "policy",
            "candidate_set",
            "selected_candidate_id",
            "priority_hash",
        ),
        "anomaly": (
            "anomaly_type",
            "summary",
            "severity",
            "source_ids",
            "status",
            "anomaly_hash",
        ),
        "research_review": (
            "summary",
            "review_due",
            "gaps",
            "recommended_actions",
            "review_hash",
        ),
        "mechanism_model": (
            "statement",
            "target_hypothesis_id",
            "mediators",
            "interventions",
            "rival_hypothesis_ids",
            "evidence_ids",
            "status",
            "mechanism_hash",
        ),
        "evidence_quality": (
            "evidence_id",
            "domains",
            "overall_grade",
            "independent",
            "assessment_hash",
        ),
        "boundary_condition": (
            "claim_id",
            "dimension",
            "condition",
            "status",
            "evidence_ids",
            "boundary_hash",
        ),
        "transfer_matrix": (
            "claim_id",
            "rows",
            "coverage",
            "transfer_ready",
            "matrix_hash",
        ),
        "posterior_update": (
            "portfolio_id",
            "priority_id",
            "selected_design_id",
            "attempt_id",
            "observation_id",
            "evidence_id",
            "observed_outcome",
            "prior_weights",
            "likelihoods",
            "posterior_weights",
            "update_hash",
        ),
    }
    if not legacy_v1:
        required["experiment_priority"] = (
            *required["experiment_priority"][:-1],
            "selected_design_id",
            "prior_weights",
            "priority_hash",
        )
        required["transfer_matrix"] = (
            *required["transfer_matrix"][:-2],
            "independence_checks",
            "transfer_ready",
            "matrix_hash",
        )
    for field in required.get(kind, ()):
        if field not in payload or payload.get(field) in (None, ""):
            issues.append(f"{kind} requires {field}")

    if kind == "hypothesis_portfolio":
        members = payload.get("members")
        if not isinstance(members, list) or len(members) < 2:
            issues.append("hypothesis portfolio requires at least two members")
        else:
            ids = [
                str(item.get("hypothesis_id") or "")
                for item in members
                if isinstance(item, Mapping)
            ]
            roles = [
                str(item.get("role") or "")
                for item in members
                if isinstance(item, Mapping)
            ]
            weights = [
                item.get("prior_weight")
                for item in members
                if isinstance(item, Mapping)
            ]
            if len(ids) != len(members) or not all(ids) or len(set(ids)) != len(ids):
                issues.append(
                    "hypothesis portfolio member ids must be present and unique"
                )
            if roles.count("primary") != 1:
                issues.append(
                    "hypothesis portfolio requires exactly one primary member"
                )
            if any(role not in {"primary", "alternative", "null"} for role in roles):
                issues.append("hypothesis portfolio member role is invalid")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
                for value in weights
            ):
                issues.append(
                    "hypothesis portfolio prior weights must be positive numbers"
                )
            elif abs(sum(float(value) for value in weights) - 1.0) > 1e-6:
                issues.append("hypothesis portfolio prior weights must sum to one")

    if kind == "discriminating_prediction":
        rivals = payload.get("distinguishes_from")
        if not isinstance(rivals, list) or not rivals:
            issues.append("discriminating prediction requires a rival hypothesis")

    if kind == "experiment_priority":
        candidates = payload.get("candidate_set")
        if not isinstance(candidates, list) or not candidates:
            issues.append("experiment priority requires candidates")
        else:
            ids = [
                str(item.get("candidate_id") or "")
                for item in candidates
                if isinstance(item, Mapping)
            ]
            ranks = [
                item.get("rank") for item in candidates if isinstance(item, Mapping)
            ]
            if len(ids) != len(candidates) or not all(ids) or len(set(ids)) != len(ids):
                issues.append("experiment candidate ids must be present and unique")
            if sorted(rank for rank in ranks if isinstance(rank, int)) != list(
                range(1, len(candidates) + 1)
            ):
                issues.append("experiment candidate ranks are invalid")
            if payload.get("selected_candidate_id") not in ids:
                issues.append("selected experiment candidate is unavailable")
            design_ids = [
                str(item.get("design_object_id") or "")
                for item in candidates
                if isinstance(item, Mapping)
            ]
            if not legacy_v1 and (
                not all(design_ids) or len(set(design_ids)) != len(design_ids)
            ):
                issues.append("experiment candidates require unique design objects")
            selected_rows = [
                item
                for item in candidates
                if isinstance(item, Mapping) and item.get("selected") is True
            ]
            if not legacy_v1 and len(selected_rows) != 1:
                issues.append("experiment priority requires exactly one selected row")
            elif not legacy_v1 and payload.get("selected_design_id") != selected_rows[
                0
            ].get("design_object_id"):
                issues.append("selected experiment design does not match selected row")
            for item in candidates:
                if not isinstance(item, Mapping):
                    continue
                predictions = item.get("predictions")
                prediction_ids = item.get("prediction_ids")
                if not legacy_v1 and (
                    not isinstance(predictions, Mapping)
                    or not isinstance(prediction_ids, Mapping)
                    or set(predictions) != set(prediction_ids)
                ):
                    issues.append(
                        "experiment candidate predictions must bind locked prediction ids"
                    )
                    break
        priors = payload.get("prior_weights")
        if not legacy_v1 and (
            not isinstance(priors, Mapping)
            or not priors
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
                for value in priors.values()
            )
            or abs(sum(float(value) for value in priors.values()) - 1.0) > 1e-6
        ):
            issues.append(
                "experiment priority requires normalized positive prior weights"
            )

    if kind == "anomaly":
        if payload.get("severity") not in {"low", "medium", "high", "critical"}:
            issues.append("anomaly severity is invalid")
        if payload.get("status") not in {"open", "explained", "resolved"}:
            issues.append("anomaly status is invalid")
        if not isinstance(payload.get("source_ids"), list) or not payload.get(
            "source_ids"
        ):
            issues.append("anomaly requires source ids")

    if kind == "mechanism_model":
        if payload.get("status") not in {"proposed", "tested", "validated", "refuted"}:
            issues.append("mechanism status is invalid")
        if payload.get("status") == "validated":
            if (
                not payload.get("mediators")
                or not payload.get("interventions")
                or not payload.get("evidence_ids")
            ):
                issues.append(
                    "validated mechanism requires mediators, interventions, and evidence"
                )
            if not payload.get("rival_hypothesis_ids"):
                issues.append("validated mechanism requires a tested rival explanation")
            validation = payload.get("validation")
            if legacy_v1:
                validation = {"legacy": True}
            if not isinstance(validation, Mapping):
                issues.append("validated mechanism requires an intervention receipt")
            elif not legacy_v1:
                if validation.get("policy") != "xscientist.intervention-lineage.v1":
                    issues.append("validated mechanism intervention policy is invalid")
                if set(validation.get("evidence_ids") or []) != set(
                    payload.get("evidence_ids") or []
                ):
                    issues.append("mechanism receipt evidence does not match the model")
                if not validation.get("attempt_ids") or not validation.get(
                    "protocol_ids"
                ):
                    issues.append(
                        "validated mechanism receipt requires attempts and protocols"
                    )
                if not validation.get("locked_protocol_ids"):
                    issues.append(
                        "validated mechanism receipt requires a locked protocol"
                    )
                expected_receipt_hash = canonical_content_hash(
                    {
                        key: value
                        for key, value in validation.items()
                        if key != "receipt_hash"
                    }
                )
                if validation.get("receipt_hash") != expected_receipt_hash:
                    issues.append("validated mechanism receipt hash mismatch")

    if kind == "evidence_quality":
        domains = payload.get("domains")
        domain_names = {
            "internal_validity",
            "measurement_reliability",
            "confounding",
            "statistical_power",
            "multiplicity",
            "preregistration_fidelity",
            "independence",
            "external_validity",
        }
        allowed = {"low_risk", "some_concerns", "high_risk", "not_assessed"}
        if not isinstance(domains, Mapping) or set(domains) != domain_names:
            issues.append("evidence quality requires every fixed assessment domain")
        elif any(value not in allowed for value in domains.values()):
            issues.append("evidence quality domain verdict is invalid")
        else:
            values = list(domains.values())
            expected_grade = (
                "critical"
                if "high_risk" in values
                else (
                    "weak"
                    if "not_assessed" in values or values.count("some_concerns") >= 3
                    else "moderate" if "some_concerns" in values else "strong"
                )
            )
            if payload.get("overall_grade") != expected_grade:
                issues.append("evidence quality grade does not match domain verdicts")
        if payload.get("overall_grade") not in {
            "strong",
            "moderate",
            "weak",
            "critical",
        }:
            issues.append("evidence quality grade is invalid")
        receipt = payload.get("independence_receipt")
        if payload.get("independent") is True and not legacy_v1:
            if not isinstance(receipt, Mapping):
                issues.append(
                    "independent evidence quality requires an authority receipt"
                )
            else:
                if receipt.get("policy") != "xscientist.provenance-actor-disjoint.v1":
                    issues.append("evidence quality independence policy is invalid")
                if payload.get("evidence_id") not in (receipt.get("target_ids") or []):
                    issues.append(
                        "evidence quality receipt does not cover its evidence"
                    )
                expected_receipt_hash = canonical_content_hash(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_hash"
                    }
                )
                if receipt.get("receipt_hash") != expected_receipt_hash:
                    issues.append("evidence quality authority receipt hash mismatch")

    if kind == "boundary_condition":
        if payload.get("status") not in {
            "supported",
            "refuted",
            "mixed",
            "untested",
        }:
            issues.append("boundary condition status is invalid")
        if payload.get("role") not in {
            "development",
            "transfer",
            "heldout",
            "scale",
        }:
            issues.append("boundary condition role is invalid")
        validation = payload.get("validation")
        if payload.get("status") != "untested" and not legacy_v1:
            if not isinstance(validation, Mapping):
                issues.append("tested boundary requires a lineage receipt")
            else:
                if (
                    validation.get("policy")
                    != "xscientist.disjoint-boundary-evidence.v1"
                ):
                    issues.append("boundary evidence policy is invalid")
                if set(validation.get("evidence_ids") or []) != set(
                    payload.get("evidence_ids") or []
                ):
                    issues.append("boundary receipt evidence mismatch")
                if not validation.get("attempt_ids") or not validation.get(
                    "dataset_hashes"
                ):
                    issues.append(
                        "tested boundary requires attempt and dataset lineage"
                    )
                expected_receipt_hash = canonical_content_hash(
                    {
                        key: value
                        for key, value in validation.items()
                        if key != "receipt_hash"
                    }
                )
                if validation.get("receipt_hash") != expected_receipt_hash:
                    issues.append("boundary lineage receipt hash mismatch")
        elif (
            payload.get("status") == "untested"
            and not legacy_v1
            and (payload.get("evidence_ids") or validation)
        ):
            issues.append("untested boundary cannot contain result evidence")

    if kind == "transfer_matrix":
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            issues.append("transfer matrix requires boundary rows")
        elif any(not isinstance(row, Mapping) for row in rows):
            issues.append("transfer matrix boundary rows must be objects")
        else:
            tested = [row for row in rows if row.get("status") != "untested"]
            dimensions = {str(row.get("dimension") or "") for row in tested}
            transfer_rows = [
                row
                for row in tested
                if row.get("role") in {"transfer", "heldout", "scale"}
            ]
            expected_ready = bool(
                len(tested) >= 3
                and len(dimensions) >= 2
                and any(row.get("status") == "supported" for row in transfer_rows)
                and all(row.get("status") == "supported" for row in tested)
                and (
                    legacy_v1
                    or (
                        isinstance(payload.get("independence_checks"), Mapping)
                        and payload["independence_checks"].get(
                            "evidence_sets_pairwise_disjoint"
                        )
                        is True
                        and payload["independence_checks"].get(
                            "attempt_sets_pairwise_disjoint"
                        )
                        is True
                        and payload["independence_checks"].get(
                            "development_heldout_datasets_disjoint"
                        )
                        is True
                    )
                )
            )
            if payload.get("transfer_ready") is not expected_ready:
                issues.append(
                    "transfer matrix readiness does not match boundary coverage"
                )
        if not isinstance(payload.get("coverage"), Mapping):
            issues.append("transfer matrix coverage must be an object")
        checks = payload.get("independence_checks")
        if not legacy_v1 and (
            not isinstance(checks, Mapping)
            or checks.get("policy") != "xscientist.disjoint-boundary-evidence.v1"
        ):
            issues.append("transfer matrix requires independence checks")

    if kind == "posterior_update":
        priors = payload.get("prior_weights")
        likelihoods = payload.get("likelihoods")
        posteriors = payload.get("posterior_weights")
        if not all(
            isinstance(value, Mapping) for value in (priors, likelihoods, posteriors)
        ):
            issues.append("posterior update weights must be objects")
        elif (
            not priors
            or set(priors) != set(likelihoods)
            or set(priors) != set(posteriors)
        ):
            issues.append("posterior update hypothesis sets do not match")
        else:
            valid_numbers = all(
                not isinstance(value, bool) and isinstance(value, (int, float))
                for mapping in (priors, likelihoods, posteriors)
                for value in mapping.values()
            )
            if not valid_numbers:
                issues.append("posterior update weights must be numeric")
            else:
                prior_total = sum(float(value) for value in priors.values())
                posterior_total = sum(float(value) for value in posteriors.values())
                if abs(prior_total - 1.0) > 1e-6 or abs(posterior_total - 1.0) > 1e-6:
                    issues.append(
                        "posterior prior and posterior weights must sum to one"
                    )
                if any(not 0 <= float(value) <= 1 for value in likelihoods.values()):
                    issues.append("posterior likelihoods must be between zero and one")
                denominator = sum(
                    float(priors[key]) * float(likelihoods[key]) for key in priors
                )
                if denominator <= 0:
                    issues.append("posterior likelihoods assign zero total probability")
                elif any(
                    abs(
                        float(posteriors[key])
                        - float(priors[key]) * float(likelihoods[key]) / denominator
                    )
                    > 1e-6
                    for key in priors
                ):
                    issues.append("posterior weights do not match Bayes' rule")

    hash_fields = {
        "hypothesis_portfolio": "portfolio_hash",
        "discriminating_prediction": "prediction_hash",
        "experiment_priority": "priority_hash",
        "anomaly": "anomaly_hash",
        "research_review": "review_hash",
        "mechanism_model": "mechanism_hash",
        "evidence_quality": "assessment_hash",
        "boundary_condition": "boundary_hash",
        "transfer_matrix": "matrix_hash",
        "posterior_update": "update_hash",
    }
    hash_field = hash_fields.get(kind)
    if hash_field and payload.get(hash_field):
        expected = canonical_content_hash(
            {key: value for key, value in payload.items() if key != hash_field}
        )
        if payload.get(hash_field) != expected:
            issues.append(f"{kind} {hash_field} mismatch")
    return issues


def _discovery_protocol_issues(kind: str, payload: Mapping[str, Any]) -> list[str]:
    """Validate the built-in subtypes that gate generalizable method claims."""

    protocol_kind = payload.get("protocol_kind")
    issues: list[str] = []

    far_kind_by_protocol = {
        "xscientist.far-research-direction.v1": "research_goal",
        "xscientist.far-opportunity-pool.v1": "question",
        "xscientist.far-opportunity-attempt.v1": "experiment_attempt",
        "xscientist.far-opportunity-judgment.v1": "review",
        "xscientist.far-opportunity-grade.v1": "review",
        "xscientist.far-allocation-plan.v1": "resource_budget",
    }
    expected_far_kind = far_kind_by_protocol.get(protocol_kind)
    if expected_far_kind is not None and kind != expected_far_kind:
        issues.append(
            f"FAR protocol {protocol_kind} must be recorded as {expected_far_kind}"
        )

    # FAR-inspired opportunity funnel subtypes.  They intentionally live on
    # existing Research VCS kinds so adding this protocol does not change any
    # previously published semantic-profile digest.
    if (
        protocol_kind == "xscientist.far-research-direction.v1"
        and kind == "research_goal"
    ):
        required = ("direction_id", "question", "objective", "goal_hash")
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"FAR research direction requires {field}")
        if payload.get("goal_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "goal_hash"}
            )
            if payload.get("goal_hash") != expected:
                issues.append("FAR research direction goal_hash mismatch")

    if protocol_kind == "xscientist.far-opportunity-pool.v1" and kind == "question":
        required = (
            "direction_id",
            "candidates",
            "candidate_count",
            "candidate_set_complete",
            "candidate_set_hash",
            "pool_hash",
        )
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"FAR opportunity pool requires {field}")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            issues.append("FAR opportunity pool candidates must be a non-empty array")
        else:
            ids = [
                str(item.get("candidate_id") or "")
                for item in candidates
                if isinstance(item, Mapping)
            ]
            if len(ids) != len(candidates) or not all(ids) or len(set(ids)) != len(ids):
                issues.append(
                    "FAR opportunity candidate ids must be present and unique"
                )
            if payload.get("candidate_count") != len(candidates):
                issues.append("FAR opportunity candidate_count mismatch")
            if payload.get("candidate_set_hash") != canonical_content_hash(candidates):
                issues.append("FAR opportunity candidate_set_hash mismatch")
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                candidate_hash = candidate.get("candidate_hash")
                expected_candidate_hash = canonical_content_hash(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "candidate_hash"
                    }
                )
                if candidate_hash != expected_candidate_hash:
                    issues.append("FAR opportunity candidate_hash mismatch")
                if candidate.get("source_status") not in {
                    "open",
                    "solved",
                    "invalid",
                    "unknown",
                }:
                    issues.append("FAR opportunity candidate source_status is invalid")
                source_object_ids = candidate.get("source_object_ids")
                if not isinstance(source_object_ids, list) or any(
                    not isinstance(value, str)
                    or re.fullmatch(r"rso-[0-9a-f]{16}", value) is None
                    for value in source_object_ids
                ):
                    issues.append("FAR opportunity source_object_ids are invalid")
                if candidate.get("lineage_bound") is not bool(source_object_ids):
                    issues.append("FAR opportunity candidate lineage_bound mismatch")
                if candidate.get("source_complete") is not bool(
                    candidate.get("source_refs") or source_object_ids
                ):
                    issues.append("FAR opportunity candidate source_complete mismatch")
                if (
                    "expected_artifact_probability_semantics" in candidate
                    and candidate.get("expected_artifact_probability_semantics")
                    not in {"conditional_on_success", "joint"}
                ):
                    issues.append(
                        "FAR opportunity expected_artifact_probability_semantics is invalid"
                    )
                for score_field in (
                    "difficulty",
                    "importance",
                    "expected_success_probability",
                    "expected_artifact_probability",
                    "expected_importance",
                ):
                    if score_field in candidate:
                        value = candidate.get(score_field)
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            or not 0 <= float(value) <= 1
                        ):
                            issues.append(
                                f"FAR opportunity {score_field} must be finite in [0, 1]"
                            )
        if not isinstance(payload.get("candidate_set_complete"), bool):
            issues.append("FAR opportunity candidate_set_complete must be boolean")
        if not isinstance(payload.get("lineage_complete"), bool):
            issues.append("FAR opportunity lineage_complete must be boolean")
        elif isinstance(candidates, list):
            expected_lineage = bool(candidates) and all(
                isinstance(item, Mapping)
                and bool(item.get("source_object_ids"))
                and item.get("lineage_bound") is True
                for item in candidates
            )
            if payload.get("lineage_complete") is not expected_lineage:
                issues.append("FAR opportunity lineage_complete mismatch")
        if payload.get("pool_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "pool_hash"}
            )
            if payload.get("pool_hash") != expected:
                issues.append("FAR opportunity pool_hash mismatch")

    if (
        protocol_kind == "xscientist.far-opportunity-attempt.v1"
        and kind == "experiment_attempt"
    ):
        required = (
            "pool_id",
            "candidate_id",
            "status",
            "outcome",
            "summary",
            "attempt_hash",
        )
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"FAR opportunity attempt requires {field}")
        if payload.get("status") not in {"completed", "failed", "timed_out"}:
            issues.append("FAR opportunity attempt status is invalid")
        if payload.get("outcome") not in {"known", "new", "fix", "none"}:
            issues.append("FAR opportunity attempt outcome is invalid")
        evidence_ids = payload.get("evidence_object_ids", [])
        if not isinstance(evidence_ids, list) or any(
            not isinstance(value, str)
            or re.fullmatch(r"rso-[0-9a-f]{16}", value) is None
            for value in evidence_ids
        ):
            issues.append("FAR opportunity attempt evidence_object_ids are invalid")
        if payload.get("attempt_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "attempt_hash"}
            )
            if payload.get("attempt_hash") != expected:
                issues.append("FAR opportunity attempt_hash mismatch")

    if (
        protocol_kind
        in {
            "xscientist.far-opportunity-judgment.v1",
            "xscientist.far-opportunity-grade.v1",
        }
        and kind == "review"
    ):
        is_judgment = protocol_kind.endswith("judgment.v1")
        hash_field = "judgment_hash" if is_judgment else "grade_hash"
        required = (
            (
                "attempt_id",
                "evaluator_id",
                "summary",
                "independence_receipt",
                hash_field,
            )
            if is_judgment
            else (
                "judgment_id",
                "evaluator_id",
                "summary",
                "independence_receipt",
                hash_field,
            )
        )
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"FAR opportunity review requires {field}")
        if payload.get("status") != "completed":
            issues.append("FAR opportunity review status must remain completed")
        if is_judgment and payload.get("verdict") not in {"pass", "fail", "known"}:
            issues.append("FAR opportunity judgment verdict is invalid")
        if not is_judgment and payload.get("grade") not in {
            "known",
            "minor",
            "substantial",
        }:
            issues.append("FAR opportunity grade is invalid")
        stage_override = payload.get("stage_gate_override")
        override_allowed = (
            isinstance(stage_override, Mapping)
            and stage_override.get("allowed") is True
            and bool(str(stage_override.get("reason") or "").strip())
        )
        if is_judgment:
            target_outcome = payload.get("target_outcome")
            if target_outcome is not None and target_outcome not in {
                "new",
                "known",
                "fix",
                "none",
            }:
                issues.append("FAR opportunity judgment target_outcome is invalid")
            if (
                target_outcome is not None
                and target_outcome != "new"
                and not override_allowed
            ):
                issues.append(
                    "FAR opportunity judgment outside NEW requires stage_gate_override"
                )
        else:
            target_verdict = payload.get("target_verdict")
            if target_verdict is not None and target_verdict not in {
                "pass",
                "fail",
                "known",
            }:
                issues.append("FAR opportunity grade target_verdict is invalid")
            if (
                target_verdict is not None
                and target_verdict not in {"pass", "known"}
                and not override_allowed
            ):
                issues.append(
                    "FAR opportunity grade outside PASS/KNOWN requires stage_gate_override"
                )
        evidence_ids = payload.get("evidence_object_ids", [])
        if not isinstance(evidence_ids, list) or any(
            not isinstance(value, str)
            or re.fullmatch(r"rso-[0-9a-f]{16}", value) is None
            for value in evidence_ids
        ):
            issues.append("FAR opportunity review evidence_object_ids are invalid")
        receipt = payload.get("independence_receipt")
        if not isinstance(receipt, Mapping):
            issues.append(
                "FAR opportunity review independence_receipt must be an object"
            )
        else:
            if receipt.get("identity_verified") is not False:
                issues.append(
                    "FAR opportunity review independence must remain declared"
                )
            if receipt.get("receipt_hash") != canonical_content_hash(
                {key: value for key, value in receipt.items() if key != "receipt_hash"}
            ):
                issues.append("FAR opportunity review receipt_hash mismatch")
        if payload.get(hash_field):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != hash_field}
            )
            if payload.get(hash_field) != expected:
                issues.append(f"FAR opportunity {hash_field} mismatch")

    if (
        protocol_kind == "xscientist.far-allocation-plan.v1"
        and kind == "resource_budget"
    ):
        required = (
            "budget",
            "limits",
            "information_value_required",
            "pool_id",
            "objective",
            "candidate_set",
            "allocation_hash",
            "budget_hash",
        )
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"FAR allocation plan requires {field}")
        if payload.get("objective") not in {
            "artifact_yield",
            "importance_yield",
            "best_artifact",
        }:
            issues.append("FAR allocation objective is invalid")
        if payload.get("probability_semantics") not in {
            "conditional_artifact_given_success",
            "joint_artifact_probability",
        }:
            issues.append("FAR allocation probability_semantics is invalid")
        if payload.get("information_value_required") is not True:
            issues.append("FAR allocation must require information-value accounting")
        if payload.get("candidate_set_complete") is not True:
            issues.append("FAR allocation must bind a complete candidate set")
        if payload.get("allocation_scope") != "complete_open_candidate_pool":
            issues.append("FAR allocation scope is invalid")
        rows = payload.get("candidate_set")
        if not isinstance(rows, list) or not rows:
            issues.append("FAR allocation candidate_set must be a non-empty array")
        else:
            ranks = [item.get("rank") for item in rows if isinstance(item, Mapping)]
            if sorted(ranks) != list(range(1, len(rows) + 1)):
                issues.append("FAR allocation candidate ranks are invalid")
            if any(
                isinstance(item, Mapping)
                and item.get("allocation_eligible") is True
                and item.get("allocation_score") is None
                for item in rows
            ):
                issues.append("FAR allocation eligible row lacks a score")
            for item in rows:
                if not isinstance(item, Mapping):
                    issues.append("FAR allocation candidate row must be an object")
                    continue
                if (
                    item.get("allocation_eligible") is True
                    and item.get("source_status") != "open"
                ):
                    issues.append("FAR allocation cannot select a non-open opportunity")
                if item.get("source_status") != "open":
                    issues.append(
                        "FAR allocation candidate_set must contain only open opportunities"
                    )
                if item.get("source_status_eligible") is not (
                    item.get("source_status") == "open"
                ):
                    issues.append("FAR allocation source_status_eligible mismatch")
        if payload.get("allocation_hash"):
            expected = canonical_content_hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {"allocation_hash", "budget_hash"}
                }
            )
            if payload.get("allocation_hash") != expected:
                issues.append("FAR allocation allocation_hash mismatch")
        if payload.get("budget_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "budget_hash"}
            )
            if payload.get("budget_hash") != expected:
                issues.append("FAR allocation budget_hash mismatch")

    if kind == "experiment_design" and protocol_kind == (
        "competitive_experiment_candidate"
    ):
        required = (
            "portfolio_id",
            "candidate_id",
            "summary",
            "condition",
            "predictions",
            "prediction_ids",
            "design_hash",
        )
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"competitive experiment design requires {field}")
        predictions = payload.get("predictions")
        prediction_ids = payload.get("prediction_ids")
        if (
            not isinstance(predictions, Mapping)
            or len(predictions) < 2
            or not isinstance(prediction_ids, Mapping)
            or set(predictions) != set(prediction_ids)
            or not all(str(value or "") for value in prediction_ids.values())
        ):
            issues.append(
                "competitive experiment design must bind predictions for every hypothesis"
            )
        if payload.get("design_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "design_hash"}
            )
            if payload.get("design_hash") != expected:
                issues.append("competitive experiment design_hash mismatch")

    if kind == "observation" and protocol_kind == (
        "competitive_experiment_observation"
    ):
        for field in ("measurement", "attempt_id", "evidence_id", "observation_hash"):
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"competitive observation requires {field}")
        if payload.get("observation_hash"):
            expected = canonical_content_hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "observation_hash"
                }
            )
            if payload.get("observation_hash") != expected:
                issues.append("competitive observation_hash mismatch")

    if kind == "experiment_design" and protocol_kind == "method_discovery_contract":
        required = (
            "summary",
            "hypothesis_id",
            "contribution_level",
            "target_component",
            "mechanism",
            "metric",
            "edit_scope",
            "fixed_variables",
            "baselines",
            "conditions",
            "runner",
            "runner_hash",
            "resource_budget_hash",
            "evaluation_blinding_hash",
            "success_rule",
            "design_hash",
        )
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"method discovery contract requires {field}")
        contribution = payload.get("contribution_level")
        if contribution not in {
            "execution",
            "engineering_optimization",
            "method_discovery",
        }:
            issues.append("method discovery contract has invalid contribution_level")
        metric = payload.get("metric")
        if not isinstance(metric, Mapping) or metric.get("direction") not in {
            "maximize",
            "minimize",
        }:
            issues.append("method discovery contract requires a directed metric")
        edit_scope = payload.get("edit_scope")
        if (
            not isinstance(edit_scope, Mapping)
            or not isinstance(edit_scope.get("allowed_paths"), list)
            or not edit_scope.get("allowed_paths")
        ):
            issues.append("method discovery contract requires allowed edit paths")
        baselines = payload.get("baselines")
        conditions = payload.get("conditions")
        if not isinstance(baselines, list):
            issues.append("method discovery contract baselines must be an array")
            baselines = []
        if not isinstance(conditions, list):
            issues.append("method discovery contract conditions must be an array")
            conditions = []
        baseline_ids = [
            str(item.get("id") or "") for item in baselines if isinstance(item, Mapping)
        ]
        condition_ids = [
            str(item.get("id") or "")
            for item in conditions
            if isinstance(item, Mapping)
        ]
        if not all(baseline_ids) or len(set(baseline_ids)) != len(baseline_ids):
            issues.append("method discovery baseline ids must be present and unique")
        if not all(condition_ids) or len(set(condition_ids)) != len(condition_ids):
            issues.append("method discovery condition ids must be present and unique")
        if contribution == "method_discovery":
            strong_count = sum(
                item.get("strong") is True
                for item in baselines
                if isinstance(item, Mapping)
            )
            roles = {
                item.get("role") for item in conditions if isinstance(item, Mapping)
            }
            if len(baselines) < 3 or strong_count < 3:
                issues.append(
                    "method discovery requires at least three strong baselines"
                )
            if any(
                not isinstance(item, Mapping) or not item.get("source")
                for item in baselines
            ):
                issues.append("method discovery baselines require source identities")
            if len(conditions) < 3:
                issues.append(
                    "method discovery requires at least three evaluation conditions"
                )
            if "development" not in roles or not roles.intersection(
                {"transfer", "heldout", "scale"}
            ):
                issues.append(
                    "method discovery requires development and generalization conditions"
                )
            if not any(
                isinstance(item, Mapping) and item.get("visibility") == "sealed"
                for item in conditions
            ):
                issues.append("method discovery requires a sealed condition")
            if not isinstance(edit_scope, Mapping) or not edit_scope.get(
                "protected_paths"
            ):
                issues.append("method discovery requires protected edit paths")
            if not isinstance(
                payload.get("fixed_variables"), Mapping
            ) or not payload.get("fixed_variables"):
                issues.append("method discovery requires fixed non-target variables")
        if payload.get("runner_hash") and isinstance(payload.get("runner"), Mapping):
            if payload.get("runner_hash") != canonical_content_hash(payload["runner"]):
                issues.append("method discovery runner_hash mismatch")
        if payload.get("context_required") is True and not (
            payload.get("context_id") and payload.get("context_hash")
        ):
            issues.append("method discovery required context binding is incomplete")
        if payload.get("design_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "design_hash"}
            )
            if payload.get("design_hash") != expected:
                issues.append("method discovery design_hash mismatch")

    if kind == "resource_budget" and protocol_kind == "method_discovery_budget":
        if not isinstance(payload.get("limits"), Mapping) or not payload.get("limits"):
            issues.append("method discovery budget requires numeric limits")
        else:
            for name, value in payload["limits"].items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    issues.append(
                        f"method discovery budget limit {name} must be numeric"
                    )
        if payload.get("information_value_required") is not True:
            issues.append("method discovery budget must prioritize information value")
        if payload.get("budget_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "budget_hash"}
            )
            if payload.get("budget_hash") != expected:
                issues.append("method discovery budget_hash mismatch")

    if kind == "evaluation_blinding" and protocol_kind == "method_discovery_blinding":
        if payload.get("leakage_prohibited") is not True:
            issues.append("method discovery blinding must prohibit feedback leakage")
        if not isinstance(payload.get("sealed_condition_ids"), list):
            issues.append(
                "method discovery blinding sealed_condition_ids must be an array"
            )
        if payload.get("blinding_hash"):
            expected = canonical_content_hash(
                {key: value for key, value in payload.items() if key != "blinding_hash"}
            )
            if payload.get("blinding_hash") != expected:
                issues.append("method discovery blinding_hash mismatch")

    if kind == "evidence_synthesis" and protocol_kind == "generalization_assessment":
        required = (
            "summary",
            "contract_id",
            "contract_hash",
            "evidence_ids",
            "candidate_id",
            "condition_assessments",
            "checks",
            "verdict",
            "method_discovery_supported",
            "synthesis_hash",
        )
        for field in required:
            if field not in payload or payload.get(field) in (None, ""):
                issues.append(f"generalization assessment requires {field}")
        verdict = payload.get("verdict")
        if verdict not in {
            "method_discovery_supported",
            "engineering_gain_only",
            "invalid_protocol_execution",
            "inconclusive",
        }:
            issues.append("generalization assessment verdict is invalid")
        if payload.get("method_discovery_supported") is not (
            verdict == "method_discovery_supported"
        ):
            issues.append(
                "generalization assessment support flag disagrees with verdict"
            )
        if not isinstance(payload.get("evidence_ids"), list) or not payload.get(
            "evidence_ids"
        ):
            issues.append("generalization assessment requires evidence ids")
        if not isinstance(payload.get("checks"), list) or not payload.get("checks"):
            issues.append("generalization assessment requires deterministic checks")
        elif verdict == "method_discovery_supported" and any(
            not isinstance(item, Mapping) or item.get("passed") is not True
            for item in payload["checks"]
        ):
            issues.append("supported method discovery has a failing protocol check")
        if payload.get("synthesis_hash"):
            expected = canonical_content_hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "synthesis_hash"
                }
            )
            if payload.get("synthesis_hash") != expected:
                issues.append("generalization assessment synthesis_hash mismatch")
    return issues


def research_payload_issues(
    kind: str,
    payload: Mapping[str, Any],
    semantic_profile: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return semantic payload gaps relevant to traceability.

    The outer Research Object schema deliberately remains forwards-compatible.
    This second layer distinguishes a syntactically storable legacy object from
    one that is sufficiently typed for closure auditing.
    """

    normalized_kind = str(kind or "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", normalized_kind):
        return [f"invalid research object kind: {normalized_kind}"]
    if normalized_kind not in RESEARCH_OBJECT_KINDS:
        try:
            _normalise_semantic_profile(normalized_kind, semantic_profile)
        except ResearchObjectError as exc:
            return [str(exc)]
    if not isinstance(payload, Mapping):
        return ["payload must be a mapping"]
    if not payload:
        return ["payload must not be empty"]
    identity_fields = _PAYLOAD_IDENTITY_FIELDS.get(normalized_kind, ())
    if identity_fields and not any(
        field in payload and payload.get(field) not in (None, "", [], {})
        for field in identity_fields
    ):
        return [
            f"{normalized_kind} payload requires one of: " + ", ".join(identity_fields)
        ]
    issues: list[str] = []
    if normalized_kind == "claim" and payload.get("depth_level") not in {
        None,
        "descriptive",
        "causal",
        "transferable",
    }:
        issues.append("claim depth_level is invalid")
    issues.extend(_discovery_protocol_issues(normalized_kind, payload))
    if normalized_kind in STRATEGY_RESEARCH_OBJECT_KINDS:
        issues.extend(
            _strategy_protocol_issues(normalized_kind, payload, semantic_profile)
        )
    if normalized_kind == "research_rollout":
        try:
            validate_json(dict(payload), load_schema("research_rollout"))
        except ValidationError as exc:
            issues.append(f"research_rollout schema invalid: {exc.message}")
    required_fields: dict[str, tuple[str, ...]] = {
        "search_plan": ("question", "queries", "search_plan_hash"),
        "search_receipt": (
            "profile",
            "provider",
            "query",
            "retrieved_at",
            "candidates",
            "receipt_hash",
        ),
        "source_snapshot": ("title", "content_hash", "source_hash"),
        "passage_evidence": (
            "source_id",
            "locator",
            "quote",
            "quote_hash",
            "passage_hash",
        ),
        "source_update": (
            "source_id",
            "status",
            "provider",
            "checked_at",
            "update_type",
            "update_hash",
        ),
    }
    for field in required_fields.get(normalized_kind, ()):
        if field not in payload or payload.get(field) in (None, ""):
            issues.append(f"{normalized_kind} payload requires {field}")
    if normalized_kind == "search_plan" and not isinstance(
        payload.get("queries"), list
    ):
        issues.append("search_plan queries must be an array")
    if normalized_kind == "search_receipt" and not isinstance(
        payload.get("candidates"), list
    ):
        issues.append("search_receipt candidates must be an array")
    if (
        normalized_kind == "search_receipt"
        and payload.get("profile") == "xscientist.retrieval-receipt.v2"
    ):
        candidates = payload.get("candidates") or []
        request = payload.get("request")
        if not isinstance(request, Mapping):
            issues.append("search_receipt v2 requires a request object")
        elif payload.get("request_hash") != canonical_content_hash(dict(request)):
            issues.append("search_receipt request_hash mismatch")
        if payload.get("candidate_set_hash") != canonical_content_hash(candidates):
            issues.append("search_receipt candidate_set_hash mismatch")
        completeness = payload.get("completeness")
        if not isinstance(completeness, Mapping):
            issues.append("search_receipt v2 requires completeness metadata")
        elif completeness.get("candidate_count") != len(candidates):
            issues.append("search_receipt candidate count mismatch")
    commitment_fields = {
        "search_plan": "search_plan_hash",
        "search_receipt": "receipt_hash",
        "source_snapshot": "source_hash",
        "passage_evidence": "passage_hash",
        "source_update": "update_hash",
    }
    commitment_field = commitment_fields.get(normalized_kind)
    if commitment_field and payload.get(commitment_field):
        expected = canonical_content_hash(
            {key: value for key, value in payload.items() if key != commitment_field}
        )
        if payload.get(commitment_field) != expected:
            issues.append(f"{normalized_kind} {commitment_field} mismatch")
    if normalized_kind == "passage_evidence" and payload.get("quote"):
        expected_quote = canonical_content_hash(str(payload["quote"]))
        if payload.get("quote_hash") != expected_quote:
            issues.append("passage_evidence quote_hash mismatch")
        selector = payload.get("selector")
        if selector is not None:
            if not isinstance(selector, Mapping):
                issues.append("passage_evidence selector must be an object")
            else:
                selector_core = {
                    key: value
                    for key, value in selector.items()
                    if key != "selector_hash"
                }
                expected_selector_hash = canonical_content_hash(selector_core)
                if selector.get("selector_hash") != expected_selector_hash:
                    issues.append("passage_evidence selector hash mismatch")
                if payload.get("selector_hash") != expected_selector_hash:
                    issues.append("passage_evidence selector binding mismatch")
                selectors = selector.get("selectors") or []
                exacts = [
                    item.get("exact")
                    for item in selectors
                    if isinstance(item, Mapping)
                    and item.get("type") == "TextQuoteSelector"
                ]
                if str(payload.get("quote")) not in exacts:
                    issues.append("passage_evidence selector exact quote mismatch")
    if isinstance(payload.get("scope"), Mapping):
        expected_scope = canonical_content_hash(dict(payload["scope"]))
        if payload.get("scope_hash") != expected_scope:
            issues.append(f"{normalized_kind} structured scope_hash mismatch")
    for field in (
        "search_plan_hash",
        "receipt_hash",
        "source_hash",
        "content_hash",
        "metadata_hash",
        "passage_hash",
        "quote_hash",
        "scope_hash",
        "request_hash",
        "candidate_set_hash",
        "selector_hash",
        "update_hash",
        "design_hash",
        "budget_hash",
        "blinding_hash",
        "synthesis_hash",
        "runner_hash",
        "resource_budget_hash",
        "evaluation_blinding_hash",
        "contract_hash",
        "context_hash",
        "goal_hash",
        "candidate_hash",
        "pool_hash",
        "attempt_hash",
        "judgment_hash",
        "grade_hash",
        "allocation_hash",
        "summary_hash",
        "rollout_hash",
        "turn_credit_hash",
        "tool_trace_hash",
        "evaluation_hash",
        "rubric_hash",
    ):
        value = payload.get(field)
        if value not in (None, "") and not _is_sha256(value):
            issues.append(f"{normalized_kind} {field} must use sha256:<64 hex>")
    return issues


def validate_research_payload(
    kind: str,
    payload: Mapping[str, Any],
    semantic_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate kind-specific minimum semantics and return a detached payload."""

    issues = research_payload_issues(kind, payload, semantic_profile)
    if issues:
        raise ResearchObjectError("invalid research payload: " + "; ".join(issues))
    return deepcopy(dict(payload))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResearchObjectError(f"{label} must be a mapping")
    return deepcopy(dict(value))


def _normalise_relations(
    relations: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in relations or ():
        if not isinstance(raw, Mapping):
            raise ResearchObjectError("each relation must be a mapping")
        relation_type = str(raw.get("type") or "").strip()
        target = str(raw.get("target") or "").strip()
        role = str(raw.get("role") or "").strip()
        if (
            relation_type not in RESEARCH_RELATION_TYPES
            and not relation_type.startswith(("https://", "http://", "urn:"))
        ):
            raise ResearchObjectError(
                "extension relation types must use an absolute URI"
            )
        if (
            relation_type in RESEARCH_RELATION_TYPES
            and re.fullmatch(r"rso-[0-9a-f]{16}", target) is None
        ):
            raise ResearchObjectError(
                "built-in relation targets must use a canonical Research Object ID "
                "(rso- followed by 16 lowercase hex characters)"
            )
        row = {"type": relation_type, "target": target}
        if role:
            row["role"] = role
        rows.append(row)
    return sorted(
        {tuple(sorted(row.items())): row for row in rows}.values(),
        key=lambda row: (row["type"], row["target"], row.get("role", "")),
    )


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key
        not in {
            "object_id",
            "qualified_id",
            "identity_profile",
            "created_at",
            "content_hash",
            "envelope_hash",
        }
    }
    if payload.get("identity_profile") == RESEARCH_OBJECT_IDENTITY_PROFILE:
        # v2 binds the profile into semantic identity, making removal of the
        # envelope/profile a detectable downgrade instead of a legacy bypass.
        result["identity_profile"] = RESEARCH_OBJECT_IDENTITY_PROFILE
    return result


def _envelope_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the host envelope whose hash authenticates ordering metadata.

    ``created_at`` intentionally remains outside the semantic object identity so
    retrying the same scientific write is idempotent.  It must nevertheless be
    authenticated before any caller may use it for ``latest``/recency
    semantics.  Keeping that second hash explicit prevents mutable wall-clock
    metadata from silently changing scientific navigation.
    """

    return {
        "object_id": payload.get("object_id"),
        "qualified_id": payload.get("qualified_id"),
        "identity_profile": payload.get("identity_profile"),
        "created_at": payload.get("created_at"),
        "content_hash": payload.get("content_hash"),
    }


def _canonical_created_at(value: Any) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 40:
        raise ResearchObjectError(
            "v2 research object created_at must be a bounded UTC RFC3339 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchObjectError(
            "v2 research object created_at must be a bounded UTC RFC3339 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResearchObjectError("v2 research object created_at must use UTC")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def research_object_order_key(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a deterministic key that never trusts unauthenticated time.

    Wall-clock values never participate in this key. Repository-aware callers
    should prefer Git commit sequence; this immutable identity fallback keeps
    uncommitted and legacy views deterministic without inventing chronology.
    """

    envelope_hash = str(payload.get("envelope_hash") or "")
    identity_profile = str(payload.get("identity_profile") or "")
    expected = content_hash(_envelope_payload(payload))
    if envelope_hash:
        if identity_profile != RESEARCH_OBJECT_IDENTITY_PROFILE:
            raise ResearchObjectError(
                "legacy research objects cannot acquire a v2 envelope"
            )
        if envelope_hash != expected:
            raise ResearchObjectError("research object envelope hash mismatch")
        _canonical_created_at(payload.get("created_at"))
    return (
        str(payload.get("content_hash") or ""),
        str(payload.get("object_id") or ""),
        envelope_hash,
    )


def build_research_object(
    *,
    kind: str,
    payload: Mapping[str, Any],
    state: str = "draft",
    relations: Sequence[Mapping[str, Any]] | None = None,
    actor: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    semantic_profile: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, schema-valid Research VCS object."""

    normalized_kind = str(kind or "").strip()
    normalized_state = str(state or "").strip()
    if normalized_kind == "experiment_attempt" and normalized_state == "running":
        raise ResearchObjectError(
            "immutable experiment attempts must be terminal; running attempts "
            "belong in the mutable execution journal"
        )
    normalized_profile = _normalise_semantic_profile(normalized_kind, semantic_profile)
    semantic_payload = validate_research_payload(
        normalized_kind, payload, normalized_profile
    )
    actor_payload = _mapping(actor, label="actor") or {
        "actor_id": "xscientist",
        "authority": "research_agent",
    }
    normalized_relations = _normalise_relations(relations)
    declared_relations = set(normalized_profile.get("relations") or [])
    undeclared_relations = sorted(
        {
            row["type"]
            for row in normalized_relations
            if row["type"] not in declared_relations
        }
    )
    if undeclared_relations:
        raise ResearchObjectError(
            "semantic_profile does not declare relation types: "
            + ", ".join(undeclared_relations)
        )
    core = {
        "schema_version": RESEARCH_OBJECT_SCHEMA,
        "protocol_kind": "research_object",
        "kind": normalized_kind,
        "semantic_profile": normalized_profile,
        "state": normalized_state,
        "payload": semantic_payload,
        "relations": normalized_relations,
        "actor": actor_payload,
        "provenance": _mapping(provenance, label="provenance"),
    }
    object_hash = content_hash(
        {**core, "identity_profile": RESEARCH_OBJECT_IDENTITY_PROFILE}
    )
    result = {
        **core,
        "object_id": f"rso-{object_hash.split(':', 1)[1][:16]}",
        "qualified_id": (
            "urn:xscientist:research-object:sha256:" + object_hash.split(":", 1)[1]
        ),
        "identity_profile": RESEARCH_OBJECT_IDENTITY_PROFILE,
        "created_at": _canonical_created_at(created_at or _now_iso()),
        "content_hash": object_hash,
    }
    result["envelope_hash"] = content_hash(_envelope_payload(result))
    try:
        validate_json(result, load_schema("research_object"))
    except ValidationError as exc:
        raise ResearchObjectError(f"invalid research object: {exc.message}") from exc
    return result


def validate_research_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, canonical identity, and object identifier."""

    if not isinstance(payload, Mapping):
        raise ResearchObjectError("research object must be a mapping")
    result = deepcopy(dict(payload))
    try:
        validate_json(result, load_schema("research_object"))
    except ValidationError as exc:
        raise ResearchObjectError(f"invalid research object: {exc.message}") from exc
    semantic_profile = result.get("semantic_profile")
    normalized_relations = _normalise_relations(result.get("relations") or [])
    if semantic_profile is None:
        # Objects produced before semantic profiles were introduced remain
        # valid; new builders always emit an explicit, content-bound profile.
        if str(result.get("kind") or "") not in CORE_RESEARCH_OBJECT_KINDS:
            raise ResearchObjectError("research object semantic_profile is missing")
        if any(
            relation["type"] not in RESEARCH_RELATION_TYPES
            for relation in normalized_relations
        ):
            raise ResearchObjectError(
                "legacy research objects cannot use extension relation types"
            )
    else:
        normalized_profile = _normalise_semantic_profile(
            str(result.get("kind") or ""), semantic_profile
        )
        undeclared = sorted(
            {
                str(relation.get("type") or "")
                for relation in normalized_relations
                if str(relation.get("type") or "")
                not in set(normalized_profile.get("relations") or [])
            }
        )
        if undeclared:
            raise ResearchObjectError(
                "semantic_profile does not declare relation types: "
                + ", ".join(undeclared)
            )
    validate_research_payload(
        str(result.get("kind") or ""),
        result.get("payload") or {},
        semantic_profile,
    )
    if (
        str(result.get("kind") or "") == "experiment_attempt"
        and str(result.get("state") or "") == "running"
    ):
        raise ResearchObjectError(
            "immutable experiment attempts must be terminal; running attempts "
            "belong in the mutable execution journal"
        )
    expected = content_hash(_identity_payload(result))
    if result.get("content_hash") != expected:
        raise ResearchObjectError("research object content hash mismatch")
    expected_id = f"rso-{expected.split(':', 1)[1][:16]}"
    if result.get("object_id") != expected_id:
        raise ResearchObjectError("research object identifier mismatch")
    qualified_id = result.get("qualified_id")
    identity_profile = result.get("identity_profile")
    if qualified_id is not None or identity_profile is not None:
        expected_qualified = (
            "urn:xscientist:research-object:sha256:" + expected.split(":", 1)[1]
        )
        if identity_profile not in SUPPORTED_RESEARCH_OBJECT_IDENTITY_PROFILES:
            raise ResearchObjectError("research object identity profile mismatch")
        if qualified_id != expected_qualified:
            raise ResearchObjectError("research object qualified identifier mismatch")
        if identity_profile == RESEARCH_OBJECT_IDENTITY_PROFILE and not result.get(
            "envelope_hash"
        ):
            raise ResearchObjectError("v2 research object envelope hash is missing")
    research_object_order_key(result)
    return result


__all__ = [
    "AUTONOMOUS_RESEARCH_OBJECT_KINDS",
    "BUILTIN_RESEARCH_PROFILES",
    "CORE_RESEARCH_OBJECT_KINDS",
    "EPISTEMIC_RESEARCH_OBJECT_KINDS",
    "LEGACY_RESEARCH_OBJECT_IDENTITY_PROFILE",
    "RESEARCH_AUTHORITIES",
    "RESEARCH_OBJECT_KINDS",
    "RESEARCH_OBJECT_IDENTITY_PROFILE",
    "RESEARCH_OBJECT_SCHEMA",
    "RESEARCH_OBJECT_STATES",
    "RESEARCH_RELATION_TYPES",
    "RESEARCH_RELATION_TARGET_KINDS",
    "RESEARCH_SEMANTIC_PROFILE_SCHEMA",
    "STRATEGY_RESEARCH_OBJECT_KINDS",
    "SUPPORTED_RESEARCH_OBJECT_IDENTITY_PROFILES",
    "ROLLOUT_RESEARCH_OBJECT_KINDS",
    "ResearchObjectError",
    "build_research_object",
    "research_object_order_key",
    "research_payload_issues",
    "research_profile_status",
    "validate_research_payload",
    "validate_research_object",
]
