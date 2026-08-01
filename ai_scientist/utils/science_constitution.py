from __future__ import annotations

"""Immutable core policy for trustworthy, cumulative scientific work."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.pipeline_contracts import save_contract_artifact

SCHEMA_VERSION = 1
CORE_POLICY_VERSION = "1.0.0"

CORE_SCIENCE_POLICY: dict[str, Any] = {
    "policy_id": "xscientist-science-constitution",
    "version": CORE_POLICY_VERSION,
    "authority": "xscientist.core-policy",
    "immutable": True,
    "priority_order": ["truth", "safety", "novelty", "impact", "throughput"],
    "principles": [
        {
            "id": "truth_before_novelty",
            "rule": "Truthfulness outranks novelty, impact, publication, and throughput.",
        },
        {
            "id": "exploration_confirmation_separation",
            "rule": "Exploratory findings may not be represented as confirmatory evidence.",
        },
        {
            "id": "traceable_claims",
            "rule": "Every promoted claim must trace to protocols, evidence, and verification artifacts.",
        },
        {
            "id": "preserve_negative_evidence",
            "rule": "Negative results, failures, contradictions, and refutations are append-only scientific assets.",
        },
        {
            "id": "independent_authority",
            "rule": "A producer may not independently verify and approve the same scientific claim.",
        },
        {
            "id": "explicit_uncertainty",
            "rule": "Claims must state uncertainty, applicability boundaries, and falsifiers when applicable.",
        },
        {
            "id": "anti_goodhart",
            "rule": "Awards, citations, publication counts, and self-review scores are not direct optimization targets.",
        },
        {
            "id": "risk_bounded_autonomy",
            "rule": "High-risk physical or dual-use actions require explicit human authorization.",
        },
        {
            "id": "immutable_audit_history",
            "rule": "Constitutional history, raw evidence, and audit events may not be autonomously overwritten.",
        },
    ],
    "protected_assets": [
        "science_constitution",
        "epistemic_graph_history",
        "raw_evidence",
        "sealed_benchmarks",
        "evaluation_hard_gates",
        "identity_and_approval_rules",
        "safety_boundaries",
    ],
    "prohibited_direct_objectives": [
        "award_probability",
        "citation_count",
        "publication_count",
        "reviewer_score",
        "self_evaluation_score",
    ],
    "amendment_policy": {
        "automatic_amendment_allowed": False,
        "minimum_independent_human_approvers": 2,
        "requires_public_rationale": True,
        "requires_impact_assessment": True,
        "requires_new_version": True,
        "requires_external_audit": True,
    },
}


class ScienceConstitutionError(ValueError):
    """Raised when the core science policy is missing, weakened, or modified."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def core_science_policy_hash() -> str:
    return _canonical_hash(CORE_SCIENCE_POLICY)


def _charter_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "constitution_id": payload.get("constitution_id"),
        "status": payload.get("status"),
        "core_policy": payload.get("core_policy"),
        "core_policy_hash": payload.get("core_policy_hash"),
        "project_context": payload.get("project_context"),
        "amendment_proposals": payload.get("amendment_proposals"),
    }


def build_science_constitution(
    *,
    project_name: str,
    additional_constraints: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create a locked project charter inheriting the code-anchored core policy."""

    name = str(project_name or "").strip()
    if not name:
        raise ScienceConstitutionError("project_name is required")
    constraints = sorted(
        {
            str(item).strip()
            for item in (additional_constraints or [])
            if str(item).strip()
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "constitution_id": "constitution:"
        + hashlib.sha256(name.encode()).hexdigest()[:16],
        "status": "locked",
        "generated_at": _now_iso(),
        "core_policy": deepcopy(CORE_SCIENCE_POLICY),
        "core_policy_hash": core_science_policy_hash(),
        "project_context": {
            "project_name": name,
            "additional_constraints": constraints,
            "inheritance_mode": "core_policy_may_only_be_tightened",
        },
        "amendment_proposals": [],
    }
    payload["constitution_hash"] = _canonical_hash(_charter_core(payload))
    return payload


def validate_science_constitution(payload: dict[str, Any] | None) -> dict[str, Any]:
    charter = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if charter.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if charter.get("status") != "locked":
        errors.append("constitution_not_locked")
    if charter.get("core_policy") != CORE_SCIENCE_POLICY:
        errors.append("core_policy_modified")
    if charter.get("core_policy_hash") != core_science_policy_hash():
        errors.append("core_policy_hash_mismatch")
    context = charter.get("project_context")
    if (
        not isinstance(context, dict)
        or not str(context.get("project_name") or "").strip()
    ):
        errors.append("project_context_invalid")
    proposals = charter.get("amendment_proposals")
    if not isinstance(proposals, list):
        errors.append("amendment_proposals_invalid")
    else:
        for index, proposal in enumerate(proposals):
            if not isinstance(proposal, dict):
                errors.append(f"amendment_proposal_invalid:{index}")
                continue
            core = {
                key: proposal.get(key)
                for key in (
                    "proposal_id",
                    "base_constitution_hash",
                    "proposed_by",
                    "rationale",
                    "impact_assessment",
                    "proposed_changes",
                    "automatic_application_allowed",
                )
            }
            if proposal.get("proposal_hash") != _canonical_hash(core):
                errors.append(f"amendment_proposal_hash_mismatch:{index}")
            if proposal.get("automatic_application_allowed") is not False:
                errors.append(f"automatic_amendment_forbidden:{index}")
    if charter.get("constitution_hash") != _canonical_hash(_charter_core(charter)):
        errors.append("constitution_hash_mismatch")
    return {"passed": not errors, "errors": errors}


def assert_science_constitution_intact(payload: dict[str, Any] | None) -> None:
    validation = validate_science_constitution(payload)
    if not validation["passed"]:
        raise ScienceConstitutionError(
            "science constitution invalid: " + ", ".join(validation["errors"])
        )


def propose_science_constitution_amendment(
    constitution: dict[str, Any],
    *,
    proposed_by: str,
    rationale: str,
    impact_assessment: str,
    proposed_changes: dict[str, Any],
) -> dict[str, Any]:
    """Append an auditable proposal without applying it to the locked policy."""

    assert_science_constitution_intact(constitution)
    required = {
        "proposed_by": proposed_by,
        "rationale": rationale,
        "impact_assessment": impact_assessment,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing or not isinstance(proposed_changes, dict) or not proposed_changes:
        raise ScienceConstitutionError(
            "amendment proposal incomplete: "
            + ", ".join(missing or ["proposed_changes"])
        )
    updated = deepcopy(constitution)
    proposal_number = len(updated["amendment_proposals"]) + 1
    core = {
        "proposal_id": f"amendment:{proposal_number}",
        "base_constitution_hash": constitution["constitution_hash"],
        "proposed_by": str(proposed_by).strip(),
        "rationale": str(rationale).strip(),
        "impact_assessment": str(impact_assessment).strip(),
        "proposed_changes": deepcopy(proposed_changes),
        "automatic_application_allowed": False,
    }
    updated["amendment_proposals"].append(
        {
            **core,
            "proposed_at": _now_iso(),
            "proposal_hash": _canonical_hash(core),
        }
    )
    updated["generated_at"] = _now_iso()
    updated["constitution_hash"] = _canonical_hash(_charter_core(updated))
    return updated


def save_science_constitution(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    producer: str,
) -> str:
    assert_science_constitution_intact(payload)
    return save_contract_artifact(
        project_root,
        "science_constitution",
        payload,
        producer=producer,
        notes="Locked core policy; autonomous amendment is forbidden.",
    )


__all__ = [
    "CORE_SCIENCE_POLICY",
    "ScienceConstitutionError",
    "assert_science_constitution_intact",
    "build_science_constitution",
    "core_science_policy_hash",
    "propose_science_constitution_amendment",
    "save_science_constitution",
    "validate_science_constitution",
]
