"""Payload-free scientific closure audits for Research VCS refs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.hashing import content_hash
from ai_scientist.protocol.research_vcs import research_payload_issues
from ai_scientist.protocol.schemas import load_schema

from .research_git import (
    ResearchGitError,
    list_research_objects_at_ref,
    show_checkpoint,
    verify_research_repository,
)

RESEARCH_CLOSURE_SCHEMA = "xscientist.research-closure.v1"
RESEARCH_CLOSURE_LEVELS = ("trace", "replay", "verify")


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
    return False


def _blocker(code: str, object_id: str, message: str) -> dict[str, str]:
    return {"code": code, "object_id": object_id, "message": message}


def _claim_closure(
    claim: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    target_level: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    claim_id = str(claim["object_id"])
    direct = _targets(claim, relation_types=("depends_on",))
    evidence_ids = _kind_ids(direct, objects, "evidence")
    gate_ids = _kind_ids(direct, objects, "gate_decision")
    # Evidence may also point at a claim, which supports imported/legacy graphs.
    evidence_ids = sorted(
        set(evidence_ids)
        | {
            object_id
            for object_id, item in objects.items()
            if item.get("kind") == "evidence"
            and claim_id in _targets(item, relation_types=("supports", "refutes"))
        }
    )
    attempt_ids = sorted(
        {
            target
            for evidence_id in evidence_ids
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
    lineage_ids = {claim_id, *evidence_ids, *attempt_ids}
    reproduction_ids = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "reproduction"
        and lineage_ids.intersection(_targets(item, relation_types=("reproduces",)))
    )

    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not evidence_ids:
        blockers.append(
            _blocker("claim_without_evidence", claim_id, "claim has no evidence anchor")
        )
    if evidence_ids and not attempt_ids:
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

    semantic_ids = [
        claim_id,
        *evidence_ids,
        *attempt_ids,
        *plan_ids,
        *preregistration_ids,
    ]
    for object_id in semantic_ids:
        item = objects[object_id]
        for issue in research_payload_issues(
            str(item["kind"]), item.get("payload") or {}
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
    passing_gates = [
        object_id
        for object_id in gate_ids
        if objects[object_id].get("state") == "verified"
        and (objects[object_id].get("payload") or {}).get("claim_promotion_allowed")
        is True
    ]
    verified_reproductions = [
        object_id
        for object_id in reproduction_ids
        if objects[object_id].get("state") == "verified"
    ]
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
    claims = sorted(
        (item for item in object_rows if item.get("kind") == "claim"),
        key=lambda item: str(item["object_id"]),
    )
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
