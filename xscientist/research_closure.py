"""Payload-free scientific closure audits for Research VCS refs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash
from ai_scientist.protocol.research_vcs import (
    research_payload_issues,
    research_profile_status,
)
from ai_scientist.protocol.schemas import load_schema

from .research_git import (
    ResearchGitError,
    list_research_objects_at_ref,
    show_checkpoint,
    verify_research_repository,
)

RESEARCH_CLOSURE_SCHEMA = "xscientist.research-closure.v1"
RESEARCH_CLOSURE_LEVELS = ("trace", "replay", "verify")
_ARGUMENT_KINDS = {
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
    "mechanism_model",
    "evidence_quality",
    "boundary_condition",
    "transfer_matrix",
}
_ARGUMENT_RELATIONS = {
    "depends_on",
    "derived_from",
    "has_premise",
    "uses_method",
    "under_assumption",
    "addresses_estimand",
    "has_effect_estimate",
    "derived_by",
    "qualifies",
}


def _targets(
    research_object: Mapping[str, Any],
    *,
    relation_types: Iterable[str] | None = None,
    role: str | None = None,
) -> list[str]:
    allowed = set(relation_types or ())
    rows: list[str] = []
    for relation in research_object.get("relations") or []:
        if allowed and relation.get("type") not in allowed:
            continue
        if role is not None and relation.get("role") != role:
            continue
        target = str(relation.get("target") or "")
        if target:
            rows.append(target)
    return sorted(set(rows))


def _kind_ids(
    object_ids: Iterable[str],
    objects: Mapping[str, Mapping[str, Any]],
    kind: str,
) -> list[str]:
    return sorted(
        object_id
        for object_id in set(object_ids)
        if objects.get(object_id, {}).get("kind") == kind
    )


def _has_hash_anchor(payload: Mapping[str, Any]) -> bool:
    for key, value in payload.items():
        if (
            key.endswith("_hash")
            and isinstance(value, str)
            and value.startswith("sha256:")
        ):
            return True
        if key.endswith("_hashes") and isinstance(value, list) and value:
            return True
        if isinstance(value, Mapping) and _has_hash_anchor(value):
            return True
        if isinstance(value, list) and any(
            isinstance(item, Mapping) and _has_hash_anchor(item) for item in value
        ):
            return True
    return False


def _blocker(code: str, object_id: str, message: str) -> dict[str, str]:
    return {"code": code, "object_id": object_id, "message": message}


def _claim_identity(
    claim: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    """Return the stable semantic identity used for implicit promotions."""

    payload = claim.get("payload") or {}
    statement = next(
        (
            str(payload.get(key)).strip()
            for key in ("claim_hash", "statement", "text", "claim")
            if payload.get(key) not in (None, "")
        ),
        "",
    )
    evidence = tuple(
        target
        for target in _targets(claim, relation_types=("depends_on",), role=None)
        if objects.get(target, {}).get("kind")
        in {"evidence", "passage_evidence", "inference", "evidence_synthesis"}
    )
    return statement, evidence


def _effective_claim_frontier(
    claims: Iterable[Mapping[str, Any]],
    objects: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Select current claims while retaining superseded objects in history.

    Explicit ``supersedes`` relations and the ``superseded`` state are
    authoritative. A verified claim also implicitly promotes an older draft
    with the exact same statement and evidence anchors; this matches the
    immutable lifecycle where promotion creates a new object.
    """

    rows = list(claims)
    superseded = {
        target
        for item in objects.values()
        for target in _targets(item, relation_types=("supersedes",))
        if objects.get(target, {}).get("kind") == "claim"
    }
    superseded.update(
        str(item["object_id"]) for item in rows if item.get("state") == "superseded"
    )
    verified_identities = {
        _claim_identity(item, objects)
        for item in rows
        if item.get("state") == "verified"
    }
    for item in rows:
        if (
            item.get("state") != "verified"
            and _claim_identity(item, objects) in verified_identities
        ):
            superseded.add(str(item["object_id"]))
    frontier = [item for item in rows if str(item["object_id"]) not in superseded]
    return frontier, sorted(superseded)


def _receipt_integrity_issues(
    reproduction: Mapping[str, Any],
    *,
    producer_ids: set[str],
) -> list[str]:
    """Validate a verified reproduction beyond its caller-supplied state."""

    payload = reproduction.get("payload") or {}
    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping):
        return ["verified reproduction does not embed its source receipt"]
    receipt_base = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "content_hash"}
    }
    expected_hash = content_hash(receipt_base)
    expected_id = f"rr-{expected_hash.split(':', 1)[1][:16]}"
    issues: list[str] = []
    if receipt.get("content_hash") != expected_hash:
        issues.append("reproduction receipt content hash mismatch")
    if receipt.get("receipt_id") != expected_id:
        issues.append("reproduction receipt identifier mismatch")
    if payload.get("receipt_hash") != expected_hash:
        issues.append("reproduction object is not bound to the embedded receipt")
    if (
        not receipt.get("executed")
        or receipt.get("reproduction_level") != "computational_rerun"
    ):
        issues.append("verified reproduction was not a computational rerun")
    if (
        receipt.get("verdict") != "passed"
        or receipt.get("returncode") != 0
        or receipt.get("timed_out") is True
        or receipt.get("objects_complete") is not True
    ):
        issues.append("verified reproduction does not contain a successful result")
    if (reproduction.get("actor") or {}).get("authority") != "independent_evaluator":
        issues.append("verified reproduction lacks independent-evaluator authority")
    if str((reproduction.get("actor") or {}).get("actor_id") or "") in producer_ids:
        issues.append("reproduction verifier is also a producer in the claim lineage")
    independence = payload.get("independence")
    if not isinstance(independence, Mapping):
        issues.append("verified reproduction lacks an independence receipt")
    else:
        expected = canonical_content_hash(
            {key: value for key, value in independence.items() if key != "receipt_hash"}
        )
        if independence.get("policy") != "xscientist.provenance-actor-disjoint.v1":
            issues.append("verified reproduction independence policy is invalid")
        if independence.get("receipt_hash") != expected:
            issues.append("verified reproduction independence receipt hash mismatch")
        if independence.get("evaluator_id") != str(
            (reproduction.get("actor") or {}).get("actor_id") or ""
        ):
            issues.append("reproduction actor disagrees with its independence receipt")
    return issues


def _decision_context_issues(
    item: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[str], bool]:
    """Return ``(valid, issues, legacy_unbound)`` for a decision object."""

    payload = item.get("payload") or {}
    context_ids = _kind_ids(
        _targets(
            item,
            relation_types=("depends_on",),
            role="decision_context",
        ),
        objects,
        "context_snapshot",
    )
    required = payload.get("context_required") is True
    if not required:
        return True, [], not context_ids
    if not context_ids:
        return False, ["required decision context snapshot is missing"], False
    from .research_context import research_context_issues

    issues: list[str] = []
    for context_id in context_ids:
        context_payload = objects[context_id].get("payload") or {}
        row_issues = research_context_issues(context_payload, objects=objects)
        if context_payload.get("context_hash") != payload.get("context_hash"):
            row_issues.append("decision context hash does not match its consumer")
        if not row_issues and context_payload.get("complete") is True:
            return True, [], False
        issues.extend(f"{context_id}: {issue}" for issue in row_issues)
        if context_payload.get("complete") is not True:
            issues.append(f"{context_id}: decision context is incomplete")
    return False, sorted(set(issues)), False


def _claim_closure(
    claim: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    target_level: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    claim_id = str(claim["object_id"])
    direct = _targets(claim, relation_types=("depends_on",))
    argument_ids = {
        target
        for target in direct
        if objects.get(target, {}).get("kind") in _ARGUMENT_KINDS
    }
    argument_ids.update(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") in {"inference", "evidence_synthesis"}
        and claim_id in _targets(item, relation_types=("supports", "derived_from"))
    )
    queue = list(argument_ids)
    while queue:
        current = queue.pop()
        for target in _targets(objects[current], relation_types=_ARGUMENT_RELATIONS):
            if (
                objects.get(target, {}).get("kind") in _ARGUMENT_KINDS
                and target not in argument_ids
            ):
                argument_ids.add(target)
                queue.append(target)
    argument_ids = set(argument_ids)
    experimental_evidence_ids = _kind_ids(direct, objects, "evidence")
    passage_ids = _kind_ids(direct, objects, "passage_evidence")
    gate_ids = _kind_ids(direct, objects, "gate_decision")
    # Evidence may also point at a claim, which supports imported/legacy graphs.
    experimental_evidence_ids = sorted(
        set(experimental_evidence_ids)
        | {
            object_id
            for object_id, item in objects.items()
            if item.get("kind") == "evidence"
            and claim_id in _targets(item, relation_types=("supports", "refutes"))
        }
    )
    passage_ids = sorted(
        set(passage_ids)
        | {
            object_id
            for object_id, item in objects.items()
            if item.get("kind") == "passage_evidence"
            and claim_id
            in _targets(
                item,
                relation_types=("qualified_supports", "qualified_refutes"),
            )
        }
    )
    for argument_id in argument_ids:
        premises = _targets(objects[argument_id], relation_types=_ARGUMENT_RELATIONS)
        experimental_evidence_ids = sorted(
            set(experimental_evidence_ids)
            | set(_kind_ids(premises, objects, "evidence"))
        )
        passage_ids = sorted(
            set(passage_ids) | set(_kind_ids(premises, objects, "passage_evidence"))
        )
    evidence_ids = sorted({*experimental_evidence_ids, *passage_ids})
    attempt_ids = sorted(
        {
            target
            for evidence_id in experimental_evidence_ids
            for target in _targets(
                objects[evidence_id], relation_types=("derived_from",)
            )
            if objects.get(target, {}).get("kind") == "experiment_attempt"
        }
    )
    plan_ids = sorted(
        {
            target
            for attempt_id in attempt_ids
            for target in _targets(objects[attempt_id], relation_types=("depends_on",))
            if objects.get(target, {}).get("kind") == "research_plan"
        }
    )
    preregistration_ids = sorted(
        {
            target
            for attempt_id in attempt_ids
            for target in _targets(objects[attempt_id], relation_types=("depends_on",))
            if objects.get(target, {}).get("kind") == "preregistration"
        }
    )
    source_ids = sorted(
        {
            target
            for passage_id in passage_ids
            for target in _targets(objects[passage_id], relation_types=("quotes",))
            if objects.get(target, {}).get("kind") == "source_snapshot"
        }
    )
    source_update_ids = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "source_update"
        and set(source_ids).intersection(
            _targets(item, relation_types=("updates", "invalidates"))
        )
    )
    effective_source_updates: dict[str, str] = {}
    for source_id in source_ids:
        candidates = [
            update_id
            for update_id in source_update_ids
            if source_id
            in _targets(objects[update_id], relation_types=("updates", "invalidates"))
        ]
        if candidates:
            effective_source_updates[source_id] = max(
                candidates,
                key=lambda update_id: (
                    str(
                        (objects[update_id].get("payload") or {}).get("checked_at")
                        or ""
                    ),
                    str(objects[update_id].get("created_at") or ""),
                    update_id,
                ),
            )
    search_receipt_ids = sorted(
        {
            target
            for source_id in source_ids
            for target in _targets(objects[source_id], relation_types=("derived_from",))
            if objects.get(target, {}).get("kind") == "search_receipt"
        }
    )
    search_plan_ids = sorted(
        {
            target
            for receipt_id in search_receipt_ids
            for target in _targets(objects[receipt_id], relation_types=("depends_on",))
            if objects.get(target, {}).get("kind") == "search_plan"
        }
    )
    lineage_ids = {
        claim_id,
        *evidence_ids,
        *attempt_ids,
        *source_ids,
        *search_receipt_ids,
        *search_plan_ids,
        *argument_ids,
        *source_update_ids,
    }
    reproduction_ids = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "reproduction"
        and lineage_ids.intersection(_targets(item, relation_types=("reproduces",)))
    )

    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    claim_payload = claim.get("payload") or {}
    qualification_ids = {
        kind: sorted(
            object_id
            for object_id in direct
            if objects.get(object_id, {}).get("kind") == kind
        )
        for kind in ("mechanism_model", "evidence_quality", "transfer_matrix")
    }
    depth_level = str(claim_payload.get("depth_level") or "descriptive")
    if depth_level in {"causal", "transferable"}:
        valid_mechanisms = [
            object_id
            for object_id in qualification_ids["mechanism_model"]
            if objects[object_id].get("state") == "verified"
            and (objects[object_id].get("payload") or {}).get("status") == "validated"
            and (objects[object_id].get("payload") or {}).get("validation")
            and set(
                (objects[object_id].get("payload") or {}).get("evidence_ids") or []
            ).intersection(evidence_ids)
        ]
        valid_quality = [
            object_id
            for object_id in qualification_ids["evidence_quality"]
            if objects[object_id].get("state") == "verified"
            and (objects[object_id].get("payload") or {}).get("independent") is True
            and (objects[object_id].get("payload") or {}).get("independence_receipt")
            and (objects[object_id].get("payload") or {}).get("overall_grade")
            in {"strong", "moderate"}
            and (objects[object_id].get("payload") or {}).get("evidence_id")
            in evidence_ids
        ]
        if not valid_mechanisms:
            blockers.append(
                _blocker(
                    "causal_claim_without_validated_mechanism",
                    claim_id,
                    "causal claim lacks a validated intervention-tested mechanism",
                )
            )
        if not valid_quality:
            blockers.append(
                _blocker(
                    "causal_claim_without_quality_assessment",
                    claim_id,
                    "causal claim lacks an independent strong/moderate evidence-quality assessment",
                )
            )
        valid_transfer = []
        for object_id in qualification_ids["transfer_matrix"]:
            matrix = objects[object_id]
            matrix_payload = matrix.get("payload") or {}
            matrix_claim = objects.get(str(matrix_payload.get("claim_id") or ""), {})
            matrix_claim_payload = matrix_claim.get("payload") or {}
            if (
                matrix.get("state") == "verified"
                and matrix_payload.get("transfer_ready") is True
                and all(
                    (matrix_payload.get("independence_checks") or {}).get(key) is True
                    for key in (
                        "evidence_sets_pairwise_disjoint",
                        "attempt_sets_pairwise_disjoint",
                        "development_heldout_datasets_disjoint",
                    )
                )
                and " ".join(str(matrix_claim_payload.get("statement") or "").split())
                == " ".join(str(claim_payload.get("statement") or "").split())
                and matrix_claim_payload.get("scope_hash")
                == claim_payload.get("scope_hash")
            ):
                valid_transfer.append(object_id)
        if depth_level == "transferable" and not valid_transfer:
            blockers.append(
                _blocker(
                    "transferable_claim_without_transfer_matrix",
                    claim_id,
                    "transferable claim lacks a passing same-scope boundary and transfer matrix",
                )
            )
    elif claim_payload.get("contribution_level") == "method_discovery":
        warnings.append(
            _blocker(
                "method_discovery_depth_undeclared",
                claim_id,
                "method-discovery claim should opt into transferable depth gates",
            )
        )
    discovery_assessment_ids = sorted(
        object_id
        for object_id in argument_ids
        if objects[object_id].get("kind") == "evidence_synthesis"
        and (objects[object_id].get("payload") or {}).get("protocol_kind")
        == "generalization_assessment"
    )
    supported_discovery_assessment_ids = sorted(
        object_id
        for object_id in discovery_assessment_ids
        if (objects[object_id].get("payload") or {}).get("verdict")
        == "method_discovery_supported"
        and (objects[object_id].get("payload") or {}).get("method_discovery_supported")
        is True
    )
    if (
        claim_payload.get("contribution_level") == "method_discovery"
        and not supported_discovery_assessment_ids
    ):
        blockers.append(
            _blocker(
                "method_discovery_without_generalization",
                claim_id,
                "method-discovery claim lacks a passing cross-condition assessment",
            )
        )
    if claim.get("state") == "verified" and not claim_payload.get("scope_hash"):
        warnings.append(
            _blocker(
                "claim_scope_unstructured",
                claim_id,
                "verified claim should bind a structured applicability scope hash",
            )
        )
    if not evidence_ids:
        blockers.append(
            _blocker("claim_without_evidence", claim_id, "claim has no evidence anchor")
        )
    for inference_id in sorted(
        object_id
        for object_id in argument_ids
        if objects[object_id].get("kind") == "inference"
    ):
        premises = _targets(
            objects[inference_id], relation_types=("has_premise", "depends_on")
        )
        if not premises:
            blockers.append(
                _blocker(
                    "inference_without_premise",
                    inference_id,
                    "inference has no explicit evidence or proposition premise",
                )
            )
    if claim.get("state") == "verified" and not argument_ids:
        warnings.append(
            _blocker(
                "claim_inference_unmodeled",
                claim_id,
                "verified claim links evidence directly without an explicit inference/warrant",
            )
        )
    if experimental_evidence_ids and not attempt_ids:
        blockers.append(
            _blocker(
                "evidence_without_attempt",
                claim_id,
                "evidence is not derived from an experiment attempt",
            )
        )
    if attempt_ids and not plan_ids:
        blockers.append(
            _blocker("attempt_without_plan", claim_id, "attempt has no research plan")
        )
    for passage_id in passage_ids:
        bound_sources = [
            target
            for target in _targets(objects[passage_id], relation_types=("quotes",))
            if objects.get(target, {}).get("kind") == "source_snapshot"
        ]
        if not bound_sources:
            blockers.append(
                _blocker(
                    "passage_without_source",
                    passage_id,
                    "passage evidence has no immutable source snapshot",
                )
            )
    if passage_ids and not source_ids:
        blockers.append(
            _blocker(
                "literature_without_source",
                claim_id,
                "literature claim has no source snapshot",
            )
        )
    for source_id in source_ids:
        bound_receipts = [
            target
            for target in _targets(objects[source_id], relation_types=("derived_from",))
            if objects.get(target, {}).get("kind") == "search_receipt"
        ]
        if not bound_receipts:
            blockers.append(
                _blocker(
                    "source_without_search_receipt",
                    source_id,
                    "source selection has no retrieval receipt",
                )
            )
        source_payload = objects[source_id].get("payload") or {}
        if source_payload.get("retraction_status") in {
            "retracted",
            "withdrawn",
            "invalid",
        }:
            blockers.append(
                _blocker(
                    "source_invalidated",
                    source_id,
                    "source snapshot is marked retracted, withdrawn, or invalid",
                )
            )
        if (
            not source_payload.get("status_check")
            and source_id not in effective_source_updates
        ):
            warnings.append(
                _blocker(
                    "source_status_unchecked",
                    source_id,
                    "source has no provider-bound correction/retraction status check",
                )
            )
    for update_id in sorted(set(effective_source_updates.values())):
        update = objects[update_id]
        update_payload = update.get("payload") or {}
        if update_payload.get("status") in {
            "retracted",
            "withdrawn",
            "invalid",
        } or _targets(update, relation_types=("invalidates",)):
            blockers.append(
                _blocker(
                    "source_invalidated",
                    update_id,
                    "a source update invalidates literature used by this claim",
                )
            )
    for receipt_id in search_receipt_ids:
        bound_plans = [
            target
            for target in _targets(objects[receipt_id], relation_types=("depends_on",))
            if objects.get(target, {}).get("kind") == "search_plan"
        ]
        if not bound_plans:
            blockers.append(
                _blocker(
                    "receipt_without_search_plan",
                    receipt_id,
                    "retrieval receipt has no preregistered search plan",
                )
            )

    semantic_ids = [
        claim_id,
        *evidence_ids,
        *attempt_ids,
        *plan_ids,
        *preregistration_ids,
        *source_ids,
        *search_receipt_ids,
        *search_plan_ids,
        *sorted(argument_ids),
        *source_update_ids,
    ]
    for object_id in semantic_ids:
        item = objects[object_id]
        for issue in research_payload_issues(
            str(item["kind"]),
            item.get("payload") or {},
            item.get("semantic_profile"),
        ):
            blockers.append(_blocker("underspecified_payload", object_id, issue))

    for attempt_id in attempt_ids:
        attempt = objects[attempt_id]
        payload = attempt.get("payload") or {}
        if payload.get("study_phase") == "confirmatory":
            locked = [
                object_id
                for object_id in preregistration_ids
                if objects[object_id].get("state") == "locked"
            ]
            if not locked:
                blockers.append(
                    _blocker(
                        "confirmatory_without_locked_preregistration",
                        attempt_id,
                        "confirmatory attempt lacks a locked preregistration",
                    )
                )

    trace_complete = not blockers
    replay_blockers: list[dict[str, str]] = []
    for attempt_id in attempt_ids:
        attempt = objects[attempt_id]
        provenance = attempt.get("provenance") or {}
        payload = attempt.get("payload") or {}
        if not provenance.get("environment_hash"):
            replay_blockers.append(
                _blocker(
                    "missing_environment_identity",
                    attempt_id,
                    "attempt has no environment hash",
                )
            )
        if not provenance.get("dependency_lock_hashes"):
            replay_blockers.append(
                _blocker(
                    "missing_dependency_lock_identity",
                    attempt_id,
                    "attempt has no dependency lock or container recipe identity",
                )
            )
        if not (
            provenance.get("code_hash")
            or provenance.get("code_commit")
            or payload.get("code_ref")
        ):
            replay_blockers.append(
                _blocker(
                    "missing_code_identity", attempt_id, "attempt has no code identity"
                )
            )
        if not (
            provenance.get("dataset_hashes")
            or payload.get("data_refs")
            or any(
                _has_hash_anchor(objects[item].get("payload") or {})
                for item in preregistration_ids
            )
        ):
            replay_blockers.append(
                _blocker(
                    "missing_data_identity",
                    attempt_id,
                    "attempt has no immutable data identity",
                )
            )
        if not (provenance.get("seeds") or payload.get("deterministic") is True):
            warnings.append(
                _blocker(
                    "seed_policy_unspecified",
                    attempt_id,
                    "record seeds or declare the attempt deterministic",
                )
            )
    for evidence_id in evidence_ids:
        if not _has_hash_anchor(objects[evidence_id].get("payload") or {}):
            replay_blockers.append(
                _blocker(
                    "missing_evidence_hash_anchor",
                    evidence_id,
                    "evidence has no immutable measurement or ARA hash",
                )
            )
    if target_level in {"replay", "verify"}:
        blockers.extend(replay_blockers)
    replay_ready = trace_complete and not replay_blockers

    verification_blockers: list[dict[str, str]] = []
    producer_ids = {
        str((objects[object_id].get("actor") or {}).get("actor_id") or "")
        for object_id in {claim_id, *evidence_ids, *attempt_ids, *argument_ids}
        if str((objects[object_id].get("actor") or {}).get("actor_id") or "")
    }
    passing_gates: list[str] = []
    for object_id in gate_ids:
        gate = objects[object_id]
        gate_context_ok, gate_context_issues, gate_context_legacy = (
            _decision_context_issues(gate, objects)
        )
        if gate_context_legacy:
            warnings.append(
                _blocker(
                    "legacy_decision_context_unbound",
                    object_id,
                    "legacy gate does not identify the exact evidence/memory context it consumed",
                )
            )
        if not gate_context_ok:
            verification_blockers.extend(
                _blocker("invalid_decision_context", object_id, issue)
                for issue in gate_context_issues
            )
        direct_evaluated = set(_targets(gate, relation_types=("evaluates",)))
        evaluated = set(direct_evaluated)
        reviews = _kind_ids(direct_evaluated, objects, "review")
        trusted_review = False
        for review_id in reviews:
            review = objects[review_id]
            review_context_ok, review_context_issues, review_context_legacy = (
                _decision_context_issues(review, objects)
            )
            if review_context_legacy:
                warnings.append(
                    _blocker(
                        "legacy_decision_context_unbound",
                        review_id,
                        "legacy review does not identify the exact evidence/memory context it consumed",
                    )
                )
            if not review_context_ok:
                verification_blockers.extend(
                    _blocker("invalid_decision_context", review_id, issue)
                    for issue in review_context_issues
                )
            review_targets = set(_targets(review, relation_types=("evaluates",)))
            evaluated.update(review_targets)
            review_actor = review.get("actor") or {}
            independence = (review.get("payload") or {}).get("independence")
            independence_ok = False
            if isinstance(independence, Mapping):
                expected_independence_hash = canonical_content_hash(
                    {
                        key: value
                        for key, value in independence.items()
                        if key != "receipt_hash"
                    }
                )
                independence_ok = bool(
                    independence.get("policy")
                    == "xscientist.provenance-actor-disjoint.v1"
                    and independence.get("receipt_hash") == expected_independence_hash
                    and independence.get("evaluator_id")
                    == str(review_actor.get("actor_id") or "")
                    and review_targets.intersection(
                        set(independence.get("target_ids") or [])
                    )
                )
            trusted_review = trusted_review or (
                review.get("state") == "verified"
                and review_actor.get("authority") == "independent_evaluator"
                and str(review_actor.get("actor_id") or "") not in producer_ids
                and bool(
                    review_targets.intersection(
                        {claim_id, *evidence_ids, *argument_ids}
                    )
                )
                and review_context_ok
                and independence_ok
            )
        bound = bool(evaluated.intersection({claim_id, *evidence_ids, *argument_ids}))
        if (
            gate.get("state") == "verified"
            and (gate.get("payload") or {}).get("claim_promotion_allowed") is True
            and (gate.get("actor") or {}).get("authority") == "deterministic_gate"
            and bound
            and trusted_review
            and gate_context_ok
        ):
            passing_gates.append(object_id)
        elif gate.get("state") == "verified":
            verification_blockers.append(
                _blocker(
                    "unbound_or_untrusted_gate",
                    object_id,
                    "passing gate must bind an independent verified review of this claim or evidence",
                )
            )
    verified_reproductions: list[str] = []
    for object_id in reproduction_ids:
        reproduction = objects[object_id]
        if reproduction.get("state") != "verified":
            continue
        receipt_issues = _receipt_integrity_issues(
            reproduction,
            producer_ids=producer_ids,
        )
        if receipt_issues:
            verification_blockers.extend(
                _blocker("invalid_reproduction_receipt", object_id, issue)
                for issue in receipt_issues
            )
        else:
            verified_reproductions.append(object_id)
    if claim.get("state") != "verified":
        verification_blockers.append(
            _blocker("claim_not_verified", claim_id, "claim is not in verified state")
        )
    if not passing_gates:
        verification_blockers.append(
            _blocker(
                "missing_passing_gate",
                claim_id,
                "claim has no passing independent gate",
            )
        )
    if not verified_reproductions:
        verification_blockers.append(
            _blocker(
                "missing_verified_reproduction",
                claim_id,
                "claim lineage has no verified reproduction receipt",
            )
        )
    for object_id in semantic_ids:
        profile_status = research_profile_status(objects[object_id])
        if profile_status.get("declared") and not profile_status.get(
            "validator_available"
        ):
            verification_blockers.append(
                _blocker(
                    "profile_validator_unavailable",
                    object_id,
                    "semantic profile is preserved but no trusted local validator is installed",
                )
            )
    if target_level == "verify":
        blockers.extend(verification_blockers)
    verified = (
        replay_ready and not verification_blockers and claim.get("state") == "verified"
    )
    complete = {
        "trace": trace_complete,
        "replay": replay_ready,
        "verify": verified,
    }[target_level]
    row = {
        "claim_id": claim_id,
        "state": str(claim.get("state") or ""),
        "evidence_ids": evidence_ids,
        "experimental_evidence_ids": experimental_evidence_ids,
        "passage_evidence_ids": passage_ids,
        "source_snapshot_ids": source_ids,
        "search_receipt_ids": search_receipt_ids,
        "search_plan_ids": search_plan_ids,
        "argument_ids": sorted(argument_ids),
        "source_update_ids": source_update_ids,
        "attempt_ids": attempt_ids,
        "plan_ids": plan_ids,
        "preregistration_ids": preregistration_ids,
        "gate_ids": gate_ids,
        "reproduction_ids": reproduction_ids,
        "trace_complete": trace_complete,
        "replay_ready": replay_ready,
        "verified": verified,
        "complete": complete,
        "missing": sorted({item["code"] for item in blockers}),
    }
    return row, blockers, warnings


def audit_research_closure(
    repo: str | Path,
    *,
    ref: str = "HEAD",
    level: str = "trace",
    verify_objects: bool = True,
) -> dict[str, Any]:
    """Audit claim-to-reproduction closure at one immutable Git ref."""

    if level not in RESEARCH_CLOSURE_LEVELS:
        raise ResearchGitError("closure level must be trace, replay, or verify")
    checkpoint = show_checkpoint(repo, ref)
    resolved = str(checkpoint["commit"])
    object_rows = list_research_objects_at_ref(repo, resolved)
    objects = {str(item["object_id"]): item for item in object_rows}
    all_claims = sorted(
        (item for item in object_rows if item.get("kind") == "claim"),
        key=lambda item: str(item["object_id"]),
    )
    claims, superseded_claim_ids = _effective_claim_frontier(all_claims, objects)
    fsck = verify_research_repository(
        repo, commit=resolved, verify_objects=verify_objects
    )
    integrity = {key: value for key, value in fsck.items() if key != "repository"}
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    claim_rows: list[dict[str, Any]] = []
    if not claims:
        blockers.append(
            _blocker("no_claims", "", "selected ref contains no typed claim objects")
        )
    for claim in claims:
        row, row_blockers, row_warnings = _claim_closure(
            claim, objects, target_level=level
        )
        claim_rows.append(row)
        blockers.extend(row_blockers)
        warnings.extend(row_warnings)
    for error in integrity["errors"]:
        blockers.append(_blocker("repository_integrity_failure", "", str(error)))
    for warning in integrity["warnings"]:
        warnings.append(_blocker("repository_integrity_warning", "", str(warning)))

    blockers = sorted(
        {content_hash(item): item for item in blockers}.values(),
        key=lambda item: (item["code"], item["object_id"], item["message"]),
    )
    warnings = sorted(
        {content_hash(item): item for item in warnings}.values(),
        key=lambda item: (item["code"], item["object_id"], item["message"]),
    )
    base: dict[str, Any] = {
        "schema_version": RESEARCH_CLOSURE_SCHEMA,
        "ref": ref,
        "commit": resolved,
        "target_level": level,
        "status": "complete" if not blockers else "blocked",
        "complete": not blockers,
        "counts": dict(
            sorted(Counter(str(item["kind"]) for item in object_rows).items())
        ),
        "superseded_claim_ids": superseded_claim_ids,
        "claims": claim_rows,
        "blockers": blockers,
        "warnings": warnings,
        "integrity": integrity,
        "payloads_disclosed": False,
    }
    result = {**base, "content_hash": content_hash(base)}
    try:
        validate_json(result, load_schema("research_closure"))
    except ValidationError as exc:  # pragma: no cover - implementation contract
        raise ResearchGitError(
            f"generated research closure is invalid: {exc.message}"
        ) from exc
    return result


__all__ = [
    "RESEARCH_CLOSURE_LEVELS",
    "RESEARCH_CLOSURE_SCHEMA",
    "audit_research_closure",
]
