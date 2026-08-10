"""Unified scientific-evidence DAG projection and offline browser."""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.graph import analyze_exploration_graph
from ai_scientist.protocol.hashing import hash_manifest
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.atomic_io import atomic_write_json, atomic_write_text

from .research_closure import audit_research_closure
from .research_git import (
    ResearchGitError,
    _normalise_relative,
    _repository_root,
    _run_git,
    list_research_objects_at_ref,
    show_checkpoint,
)

RESEARCH_DAG_SCHEMA = "xscientist.research-dag.v1"

_PHASES = {
    "question": 0,
    "search_plan": 1,
    "search_receipt": 2,
    "source_snapshot": 3,
    "passage_evidence": 4,
    "hypothesis": 1,
    "hypothesis_portfolio": 1,
    "discriminating_prediction": 2,
    "experiment_priority": 2,
    "anomaly": 5,
    "research_review": 6,
    "mechanism_model": 5,
    "evidence_quality": 5,
    "boundary_condition": 6,
    "transfer_matrix": 6,
    "research_plan": 2,
    "preregistration": 2,
    "agent_candidate": 2,
    "experiment_attempt": 3,
    "agent_evaluation": 3,
    "context_snapshot": 4,
    "inference": 5,
    "warrant": 4,
    "assumption": 2,
    "method": 2,
    "estimand": 2,
    "effect_estimate": 4,
    "protocol_deviation": 4,
    "sensitivity_analysis": 5,
    "risk_of_bias": 5,
    "evidence_synthesis": 6,
    "challenge": 6,
    "source_update": 4,
    "context_robustness": 5,
    "research_goal": 0,
    "action_proposal": 2,
    "experiment_design": 2,
    "resource_budget": 2,
    "stopping_decision": 5,
    "novelty_check": 5,
    "evaluation_blinding": 2,
    "human_escalation": 5,
    "experiment_node": 3,
    "metric": 4,
    "evidence": 4,
    "observation": 4,
    "review": 5,
    "gate_decision": 5,
    "claim": 6,
    "reproduction": 7,
    "manuscript": 8,
}

_LAYER_KINDS = {
    "strategy": {
        "question",
        "research_goal",
        "search_plan",
        "hypothesis",
        "hypothesis_portfolio",
        "discriminating_prediction",
        "experiment_priority",
        "action_proposal",
        "research_review",
        "stopping_decision",
        "human_escalation",
    },
    "execution": {
        "research_plan",
        "preregistration",
        "experiment_design",
        "resource_budget",
        "evaluation_blinding",
        "experiment_attempt",
        "experiment_node",
    },
    "evidence": {
        "search_receipt",
        "source_snapshot",
        "passage_evidence",
        "observation",
        "metric",
        "evidence",
        "effect_estimate",
        "reproduction",
        "evidence_quality",
        "risk_of_bias",
    },
    "theory": {
        "inference",
        "warrant",
        "assumption",
        "method",
        "estimand",
        "mechanism_model",
        "sensitivity_analysis",
        "evidence_synthesis",
        "boundary_condition",
        "transfer_matrix",
        "anomaly",
        "challenge",
        "claim",
    },
    "decision_memory": {
        "context_snapshot",
        "review",
        "gate_decision",
        "context_robustness",
        "novelty_check",
        "protocol_deviation",
        "source_update",
    },
    "evolution": {"agent_candidate", "agent_evaluation", "manuscript"},
}


def _epistemic_layer(kind: str) -> str:
    for layer, kinds in _LAYER_KINDS.items():
        if kind in kinds:
            return layer
    return "evidence"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _summary(item: Mapping[str, Any], *, disclose: bool) -> str:
    if not disclose:
        return f"{item.get('kind')} {item.get('object_id')}"
    payload = item.get("payload") or {}
    for key in (
        "statement",
        "question",
        "summary",
        "result",
        "title",
        "query",
        "locator",
        "decision",
        "status",
        "name",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            compact = " ".join(str(value).split())
            return compact[:157] + ("..." if len(compact) > 157 else "")
    return f"{item.get('kind')} {item.get('object_id')}"


def _strategy_projection(
    objects: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Project current theory, claim drill-down, and research-depth coverage."""

    def targets(
        item: Mapping[str, Any], relation_types: set[str] | None = None
    ) -> set[str]:
        return {
            str(relation.get("target") or "")
            for relation in item.get("relations") or []
            if str(relation.get("target") or "")
            and (not relation_types or relation.get("type") in relation_types)
        }

    def latest(kind: str) -> Mapping[str, Any] | None:
        rows = [item for item in objects.values() if item.get("kind") == kind]
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

    superseded = {
        target for item in objects.values() for target in targets(item, {"supersedes"})
    }
    active_hypotheses = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "hypothesis"
        and item.get("state") != "superseded"
        and object_id not in superseded
    )
    open_anomalies = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "anomaly"
        and (item.get("payload") or {}).get("status") == "open"
        and object_id not in superseded
    )
    mechanisms = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "mechanism_model"
        and (item.get("payload") or {}).get("status")
        in {"proposed", "tested", "validated"}
        and object_id not in superseded
    )
    latest_priority = latest("experiment_priority")
    next_experiment = None
    if latest_priority is not None:
        priority_payload = latest_priority.get("payload") or {}
        selected_id = priority_payload.get("selected_candidate_id")
        selected = next(
            (
                row
                for row in priority_payload.get("candidate_set") or []
                if row.get("candidate_id") == selected_id
            ),
            None,
        )
        if selected:
            next_experiment = {
                "priority_object_id": latest_priority["object_id"],
                "candidate_id": selected_id,
                "summary": selected.get("summary"),
                "expected_information_gain": selected.get("expected_information_gain"),
                "utility_score": selected.get("utility_score"),
            }
    latest_review = latest("research_review")
    open_questions = list((latest_review or {}).get("payload", {}).get("gaps") or [])
    for item in objects.values():
        if item.get("kind") != "boundary_condition":
            continue
        payload = item.get("payload") or {}
        if payload.get("status") == "untested":
            open_questions.append(
                {
                    "code": "untested_boundary",
                    "message": f"{payload.get('dimension')}: {payload.get('condition')}",
                    "object_id": item["object_id"],
                }
            )
    frontier_core = {
        "active_hypothesis_ids": active_hypotheses,
        "portfolio_ids": sorted(
            object_id
            for object_id, item in objects.items()
            if item.get("kind") == "hypothesis_portfolio"
            and object_id not in superseded
        ),
        "prediction_ids": sorted(
            object_id
            for object_id, item in objects.items()
            if item.get("kind") == "discriminating_prediction"
            and object_id not in superseded
        ),
        "mechanism_ids": mechanisms,
        "open_anomaly_ids": open_anomalies,
        "open_questions": open_questions,
        "next_experiment": next_experiment,
    }
    theory_frontier = {
        **frontier_core,
        "frontier_hash": canonical_content_hash(frontier_core),
    }

    claim_insights: list[dict[str, Any]] = []
    for claim_id, claim in sorted(objects.items()):
        if claim.get("kind") != "claim":
            continue
        direct = targets(claim)
        supporting = sorted(
            {
                object_id
                for object_id in direct
                if objects.get(object_id, {}).get("kind")
                in {"evidence", "passage_evidence", "inference", "evidence_synthesis"}
            }
            | {
                object_id
                for object_id, item in objects.items()
                if claim_id in targets(item, {"supports", "qualified_supports"})
            }
        )
        refuting = sorted(
            object_id
            for object_id, item in objects.items()
            if claim_id
            in targets(
                item,
                {"refutes", "qualified_refutes", "contradicts", "challenges_inference"},
            )
        )
        mechanism_ids = sorted(
            object_id
            for object_id in direct
            if objects.get(object_id, {}).get("kind") == "mechanism_model"
        )
        quality_ids = sorted(
            object_id
            for object_id, item in objects.items()
            if item.get("kind") == "evidence_quality"
            and set(supporting).intersection(targets(item, {"evaluates"}))
        )
        boundary_ids = sorted(
            {
                object_id
                for object_id, item in objects.items()
                if item.get("kind") in {"boundary_condition", "transfer_matrix"}
                and claim_id in targets(item)
            }
            | {
                object_id
                for object_id in direct
                if objects.get(object_id, {}).get("kind")
                in {"boundary_condition", "transfer_matrix"}
            }
        )
        depth_level = str(
            (claim.get("payload") or {}).get("depth_level") or "descriptive"
        )
        supporting_set = set(supporting)
        valid_mechanisms = [
            object_id
            for object_id in mechanism_ids
            if objects[object_id].get("state") == "verified"
            and (objects[object_id].get("payload") or {}).get("status") == "validated"
            and supporting_set.intersection(
                (objects[object_id].get("payload") or {}).get("evidence_ids") or []
            )
        ]
        valid_quality = [
            object_id
            for object_id in quality_ids
            if objects[object_id].get("state") == "verified"
            and (objects[object_id].get("payload") or {}).get("independent") is True
            and (objects[object_id].get("payload") or {}).get("overall_grade")
            in {"strong", "moderate"}
        ]
        claim_payload = claim.get("payload") or {}
        valid_transfer = []
        for object_id in boundary_ids:
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
        claim_insights.append(
            {
                "claim_id": claim_id,
                "depth_level": depth_level,
                "supporting_ids": supporting,
                "refuting_ids": refuting,
                "mechanism_ids": mechanism_ids,
                "quality_assessment_ids": quality_ids,
                "boundary_ids": boundary_ids,
                "next_experiment": next_experiment,
                "gaps": gaps,
                "decision_ready": not gaps and not refuting,
            }
        )
    strategy_summary = {
        "profile": "xscientist.deep-research-strategy.v1",
        "layer_counts": dict(
            sorted(
                Counter(
                    _epistemic_layer(str(item.get("kind") or ""))
                    for item in objects.values()
                ).items()
            )
        ),
        "review_due": bool(
            not latest_review
            or (latest_review.get("payload") or {}).get("review_due") is True
        ),
        "latest_review_id": (
            str(latest_review["object_id"]) if latest_review is not None else None
        ),
        "open_anomaly_count": len(open_anomalies),
        "claim_depth_counts": dict(
            sorted(Counter(item["depth_level"] for item in claim_insights).items())
        ),
    }
    return theory_frontier, claim_insights, strategy_summary


def _has_hash_anchor(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key.endswith("_hash") and _is_sha256(value):
                return True
            if (
                key.endswith("_hashes")
                and isinstance(value, list)
                and bool(value)
                and all(_is_sha256(item) for item in value)
            ):
                return True
            if _has_hash_anchor(value):
                return True
    elif isinstance(payload, list):
        return any(_has_hash_anchor(item) for item in payload)
    return False


def _check(code: str, label: str, passed: bool, layer: str) -> dict[str, Any]:
    return {"code": code, "label": label, "passed": bool(passed), "layer": layer}


def _read_json_lines(raw: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate((raw or "").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"manifest history line {line_number} is invalid: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"manifest history line {line_number} is not an object")
            continue
        rows.append(payload)
    return rows, errors


def _manifest_revision_state(
    manifest: Mapping[str, Any] | None,
    lock: Mapping[str, Any] | None,
    history_rows: Sequence[Mapping[str, Any]],
    *,
    expected_hash: str | None = None,
    parse_errors: Sequence[str] = (),
    archived_manifests: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Validate an ARA manifest revision chain without depending on a worktree.

    Research DAGs may be built for an old Git ref, so local-only verification is
    insufficient.  This projection validates the committed manifest, immutable
    base lock, and append-only history in memory and exposes every legitimate
    revision hash for provenance matching.
    """

    issues = list(parse_errors)
    base_hash = str((lock or {}).get("manifest_hash") or "") or None
    current_hash = hash_manifest(dict(manifest)) if manifest is not None else None
    valid_hashes: list[str] = []
    if lock is not None:
        try:
            validate_json(dict(lock), load_schema("manifest_lock"))
        except ValidationError as exc:
            issues.append(f"manifest lock schema is invalid: {exc.message}")
        if lock.get("hasher") not in {None, "hash_manifest.v1"}:
            issues.append(f"unsupported manifest hasher: {lock.get('hasher')}")
    if _is_sha256(base_hash):
        valid_hashes.append(str(base_hash))
    elif lock is not None:
        issues.append("manifest lock has an invalid base hash")

    expected_base = base_hash
    for expected_revision, row in enumerate(history_rows, start=1):
        try:
            validate_json(dict(row), load_schema("manifest_revision"))
        except ValidationError as exc:
            issues.append(
                f"manifest revision {expected_revision} schema is invalid: {exc.message}"
            )
        revision = row.get("revision")
        row_base = str(row.get("base_hash") or "")
        new_hash = str(row.get("new_hash") or "")
        if revision != expected_revision:
            issues.append(
                f"manifest revision {expected_revision} is numbered {revision}"
            )
        if expected_base and row_base != expected_base:
            issues.append(
                f"manifest revision {expected_revision} does not extend the previous hash"
            )
        if not _is_sha256(row_base) or not _is_sha256(new_hash):
            issues.append(f"manifest revision {expected_revision} has an invalid hash")
        elif new_hash not in valid_hashes:
            valid_hashes.append(new_hash)
        if archived_manifests is not None and _is_sha256(row_base):
            archived = archived_manifests.get(row_base)
            if archived is None:
                issues.append(
                    f"manifest revision {expected_revision} is missing its base snapshot"
                )
            elif hash_manifest(dict(archived)) != row_base:
                issues.append(
                    f"manifest revision {expected_revision} base snapshot hash mismatch"
                )
        expected_base = new_hash or expected_base

    if manifest is None:
        state = "legacy_lock_only" if _is_sha256(base_hash) else "unlocked"
        if state == "unlocked":
            issues.append("manifest.json is missing")
        else:
            issues.append(
                "manifest.json is unavailable; using a legacy lock-only binding"
            )
    elif lock is None:
        state = "unlocked"
        issues.append("manifest.lock is missing")
    elif issues:
        state = "tampered"
    elif not history_rows and current_hash == base_hash:
        state = "clean"
    elif history_rows and current_hash == expected_base:
        state = "revised"
    else:
        state = "tampered"
        issues.append("current manifest hash is outside the declared revision chain")

    if (
        current_hash
        and current_hash not in valid_hashes
        and state in {"clean", "revised"}
    ):
        valid_hashes.append(current_hash)
    if expected_hash and current_hash != expected_hash:
        issues.append(
            "checkpoint manifest hash does not match the manifest stored at this ref"
        )
        state = "tampered"
    if state not in {"clean", "revised", "legacy_lock_only"}:
        valid_hashes = []
    return {
        "state": state,
        "ok": state in {"clean", "revised", "legacy_lock_only"},
        "base_hash": base_hash,
        "current_hash": current_hash,
        "expected_hash": expected_hash,
        "revision_count": len(history_rows),
        "valid_hashes": valid_hashes,
        "issues": issues,
    }


def _json_blob_at_commit(
    repo: Path,
    commit: str,
    path: str,
    *,
    required: bool = False,
) -> Any | None:
    normalized = _normalise_relative(path)
    blob = _run_git(repo, ["show", f"{commit}:{normalized}"], check=False)
    if blob.returncode:
        if required:
            raise ResearchGitError(
                f"ARA snapshot path is missing at {commit}: {normalized}"
            )
        return None
    if len(blob.stdout.encode("utf-8")) > 32 * 1024 * 1024:
        raise ResearchGitError(f"ARA snapshot path exceeds 32 MiB: {normalized}")
    try:
        return json.loads(blob.stdout)
    except json.JSONDecodeError as exc:
        raise ResearchGitError(
            f"ARA snapshot JSON is invalid at {commit}:{normalized}"
        ) from exc


def _text_blob_at_commit(repo: Path, commit: str, path: str) -> str | None:
    normalized = _normalise_relative(path)
    blob = _run_git(repo, ["show", f"{commit}:{normalized}"], check=False)
    if blob.returncode:
        return None
    if len(blob.stdout.encode("utf-8")) > 32 * 1024 * 1024:
        raise ResearchGitError(f"ARA snapshot path exceeds 32 MiB: {normalized}")
    return blob.stdout


def _targets(
    item: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    kinds: Iterable[str] = (),
    relations: Iterable[str] = (),
) -> list[str]:
    selected_kinds = set(kinds)
    selected_relations = set(relations)
    return sorted(
        {
            str(relation.get("target"))
            for relation in item.get("relations") or []
            if (
                (not selected_relations or relation.get("type") in selected_relations)
                and str(relation.get("target")) in objects
                and (
                    not selected_kinds
                    or objects[str(relation.get("target"))].get("kind")
                    in selected_kinds
                )
            )
        }
    )


def _object_proof(
    item: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    claim_closure: Mapping[str, Mapping[str, Any]],
    challenge_targets: set[str],
    closure_blockers: Mapping[str, list[str]],
    closure_warnings: Mapping[str, list[str]],
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    kind = str(item["kind"])
    state = str(item["state"])
    payload = item.get("payload") or {}
    provenance = item.get("provenance") or {}
    actor = item.get("actor") or {}
    checks = [
        _check(
            "content_addressed", "Content hash and object ID validate", True, "trace"
        )
    ]
    from ai_scientist.protocol.research_vcs import research_profile_status

    profile_status = research_profile_status(item)

    if kind == "experiment_attempt":
        checks.extend(
            [
                _check(
                    "plan_bound",
                    "Experiment is bound to a research plan",
                    bool(_targets(item, objects, kinds={"research_plan"})),
                    "trace",
                ),
                _check(
                    "environment_bound",
                    "Environment identity is recorded",
                    bool(provenance.get("environment_hash")),
                    "replay",
                ),
                _check(
                    "dependencies_bound",
                    "Dependency lock identity is recorded",
                    bool(provenance.get("dependency_lock_hashes")),
                    "replay",
                ),
                _check(
                    "code_bound",
                    "Code identity is recorded",
                    bool(
                        provenance.get("code_hash")
                        or provenance.get("code_commit")
                        or payload.get("code_ref")
                    ),
                    "replay",
                ),
                _check(
                    "data_bound",
                    "Dataset identity is recorded",
                    bool(provenance.get("dataset_hashes")),
                    "replay",
                ),
            ]
        )
    elif kind == "search_plan":
        checks.extend(
            [
                _check(
                    "search_plan_locked",
                    "Literature queries and criteria were locked",
                    state == "locked",
                    "trace",
                ),
                _check(
                    "search_plan_bound",
                    "Search plan has an immutable commitment",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "search_receipt":
        checks.extend(
            [
                _check(
                    "search_plan_bound",
                    "Retrieval receipt points to its locked search plan",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"search_plan"},
                            relations={"depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "retrieval_bound",
                    "Retrieval candidates have an immutable receipt hash",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "source_snapshot":
        checks.extend(
            [
                _check(
                    "search_receipt_bound",
                    "Source selection points to its retrieval receipt",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"search_receipt"},
                            relations={"derived_from"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "source_content_bound",
                    "Source content has an immutable identity",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "passage_evidence":
        checks.extend(
            [
                _check(
                    "source_snapshot_bound",
                    "Passage points to an immutable source snapshot",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"source_snapshot"},
                            relations={"quotes"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "passage_bound",
                    "Passage quote and locator have immutable identities",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind == "inference":
        checks.extend(
            [
                _check(
                    "premise_bound",
                    "Inference names its evidence or proposition premises",
                    bool(
                        _targets(
                            item,
                            objects,
                            relations={"has_premise", "depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "inference_bound",
                    "Inference payload has an immutable semantic anchor",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif kind in {"warrant", "assumption", "method"}:
        checks.append(
            _check(
                "semantic_anchor",
                "Argument component has an immutable semantic anchor",
                _has_hash_anchor(payload),
                "replay",
            )
        )
    elif kind == "effect_estimate":
        checks.extend(
            [
                _check(
                    "estimand_bound",
                    "Effect estimate identifies its estimand",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"estimand"},
                            relations={"addresses_estimand", "depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "uncertainty_recorded",
                    "Effect estimate records interval, standard error, or posterior",
                    any(
                        payload.get(key) not in (None, "", [], {})
                        for key in (
                            "confidence_interval",
                            "credible_interval",
                            "standard_error",
                            "posterior",
                        )
                    ),
                    "replay",
                ),
            ]
        )
    elif kind == "source_update":
        checks.extend(
            [
                _check(
                    "source_bound",
                    "Status update points to an immutable source snapshot",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"source_snapshot"},
                            relations={"updates", "invalidates"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "source_update_bound",
                    "Status update has an immutable provider receipt",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif (
        kind == "experiment_design"
        and payload.get("protocol_kind") == "method_discovery_contract"
    ):
        conditions = payload.get("conditions") or []
        baselines = payload.get("baselines") or []
        checks.extend(
            [
                _check(
                    "discovery_contract_locked",
                    "Method-discovery scope was locked before evaluation",
                    state == "locked" and _has_hash_anchor(payload),
                    "trace",
                ),
                _check(
                    "target_variable_isolated",
                    "Allowed and protected edit scopes isolate the target variable",
                    bool((payload.get("edit_scope") or {}).get("allowed_paths"))
                    and bool((payload.get("edit_scope") or {}).get("protected_paths")),
                    "replay",
                ),
                _check(
                    "strong_comparators_bound",
                    "At least three strong comparison methods are committed",
                    len(baselines) >= 3
                    and sum(
                        isinstance(row, Mapping) and row.get("strong") is True
                        for row in baselines
                    )
                    >= 3,
                    "replay",
                ),
                _check(
                    "generalization_conditions_bound",
                    "Multiple conditions include sealed transfer, heldout, or scale evaluation",
                    len(conditions) >= 3
                    and any(
                        isinstance(row, Mapping)
                        and row.get("visibility") == "sealed"
                        and row.get("role") in {"transfer", "heldout", "scale"}
                        for row in conditions
                    ),
                    "replay",
                ),
                _check(
                    "budget_and_blinding_bound",
                    "Resource budget and feedback blinding are explicit DAG parents",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"resource_budget"},
                            relations={"depends_on"},
                        )
                    )
                    and bool(
                        _targets(
                            item,
                            objects,
                            kinds={"evaluation_blinding"},
                            relations={"depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "discovery_context_bound",
                    "Method selection is bound to the exact evidence and memory context",
                    (
                        bool(
                            _targets(
                                item,
                                objects,
                                kinds={"context_snapshot"},
                                relations={"uses_context"},
                            )
                        )
                        if payload.get("context_required") is True
                        else True
                    ),
                    "trace",
                ),
            ]
        )
    elif (
        kind == "resource_budget"
        and payload.get("protocol_kind") == "method_discovery_budget"
    ):
        checks.extend(
            [
                _check(
                    "resource_budget_locked",
                    "Resource limits were locked before candidate evaluation",
                    state == "locked" and bool(payload.get("limits")),
                    "trace",
                ),
                _check(
                    "information_value_policy",
                    "Adaptive compute must be allocated by expected information value",
                    payload.get("information_value_required") is True,
                    "replay",
                ),
            ]
        )
    elif (
        kind == "evaluation_blinding"
        and payload.get("protocol_kind") == "method_discovery_blinding"
    ):
        checks.extend(
            [
                _check(
                    "sealed_feedback_locked",
                    "Held-out feedback remains sealed until final candidate commitment",
                    state == "locked"
                    and payload.get("leakage_prohibited") is True
                    and bool(payload.get("sealed_condition_ids")),
                    "trace",
                ),
                _check(
                    "blinding_bound",
                    "Blinding policy has an immutable commitment",
                    _has_hash_anchor(payload),
                    "replay",
                ),
            ]
        )
    elif (
        kind == "evidence_synthesis"
        and payload.get("protocol_kind") == "generalization_assessment"
    ):
        checks.extend(
            [
                _check(
                    "discovery_contract_bound",
                    "Assessment points to its locked method-discovery contract",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"experiment_design"},
                            relations={"depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "condition_evidence_bound",
                    "Assessment is derived from recorded condition evidence",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={
                                "evidence",
                                "effect_estimate",
                                "passage_evidence",
                            },
                            relations={"derived_from"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "generalization_protocol_valid",
                    "Every scope, resource, baseline, condition, and proxy check passed",
                    all(
                        isinstance(row, Mapping) and row.get("passed") is True
                        for row in payload.get("checks") or []
                    ),
                    "replay",
                ),
                _check(
                    "method_generalizes",
                    "Candidate improves sealed generalization conditions, not only development",
                    payload.get("verdict") == "method_discovery_supported",
                    "replay",
                ),
            ]
        )
    elif kind in {
        "action_proposal",
        "stopping_decision",
        "novelty_check",
        "human_escalation",
        "context_robustness",
    }:
        checks.append(
            _check(
                "decision_context_bound",
                "Autonomous decision is bound to an exact context snapshot",
                bool(
                    _targets(
                        item,
                        objects,
                        kinds={"context_snapshot"},
                        relations={"uses_context", "depends_on"},
                    )
                ),
                "trace",
            )
        )
    elif kind == "evidence":
        checks.extend(
            [
                _check(
                    "attempt_bound",
                    "Evidence is derived from a recorded attempt",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"experiment_attempt"},
                            relations={"derived_from", "depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "measurement_bound",
                    "Evidence has an immutable measurement anchor",
                    _has_hash_anchor(payload),
                    "replay",
                ),
                _check(
                    "independently_verified",
                    "Independent verifier accepted the evidence",
                    state == "verified"
                    and actor.get("authority") == "independent_evaluator",
                    "verify",
                ),
            ]
        )
    elif kind == "review":
        checks.append(
            _check(
                "independent_authority",
                "Review was produced by an independent evaluator",
                state == "verified"
                and actor.get("authority") == "independent_evaluator",
                "verify",
            )
        )
    elif kind == "gate_decision":
        checks.append(
            _check(
                "deterministic_authority",
                "Gate was produced by deterministic policy",
                state in {"verified", "promoted"}
                and actor.get("authority") == "deterministic_gate",
                "verify",
            )
        )
        deployment = payload.get("deployment_receipt")
        if isinstance(deployment, Mapping):
            from ai_scientist.utils.evolution_deployment import (
                validate_deployment_receipt,
            )

            checks.append(
                _check(
                    "deployment_receipt",
                    "Production deployment receipt validates",
                    validate_deployment_receipt(deployment)["ok"],
                    "verify",
                )
            )
    elif kind == "reproduction":
        checks.extend(
            [
                _check(
                    "receipt_bound",
                    "Reproduction contains a content-addressed receipt",
                    _has_hash_anchor(payload),
                    "replay",
                ),
                _check(
                    "reproduction_verified",
                    "Independent reproduction passed",
                    state == "verified",
                    "verify",
                ),
            ]
        )
    elif kind == "agent_candidate":
        candidate = payload.get("candidate") or payload
        checks.append(
            _check(
                "candidate_artifact_bound",
                "Evolution candidate is bound to an immutable artifact",
                _is_sha256(candidate.get("candidate_artifact_hash"))
                or _is_sha256(candidate.get("candidate_hash")),
                "replay",
            )
        )
    elif kind == "agent_evaluation":
        checks.append(
            _check(
                "independent_evaluation",
                "Evolution evaluation has independent authority",
                state == "verified"
                and actor.get("authority") == "independent_evaluator",
                "verify",
            )
        )
    elif kind == "claim":
        closure = claim_closure.get(object_id)
        checks.extend(
            [
                _check(
                    "claim_trace",
                    "Claim is linked to evidence, attempt, and plan",
                    bool(closure and closure.get("trace_complete")),
                    "trace",
                ),
                _check(
                    "claim_replay",
                    "Claim lineage has sufficient replay identities",
                    bool(closure and closure.get("replay_ready")),
                    "replay",
                ),
                _check(
                    "claim_verify",
                    "Claim has independent gate and reproduction",
                    bool(closure and closure.get("verified")),
                    "verify",
                ),
            ]
        )

    layer_pass = {
        layer: all(check["passed"] for check in checks if check["layer"] == layer)
        for layer in ("trace", "replay", "verify")
    }
    contested = object_id in challenge_targets
    if contested:
        level = "contested"
    elif (
        checks
        and layer_pass["verify"]
        and any(check["layer"] == "verify" for check in checks)
    ):
        level = "verified"
    elif (
        layer_pass["trace"]
        and layer_pass["replay"]
        and any(check["layer"] == "replay" for check in checks)
    ):
        level = "replayable"
    elif layer_pass["trace"]:
        level = "traceable"
    else:
        level = "recorded"
    blockers = [check["code"] for check in checks if not check["passed"]]
    blockers.extend(closure_blockers.get(object_id, []))
    proof_warnings = list(closure_warnings.get(object_id, []))
    if profile_status.get("declared") and not profile_status.get("validator_available"):
        blockers.append("profile_validator_unavailable")
        proof_warnings.append(
            "semantic profile is preserved but has no trusted local validator"
        )
        if level == "verified":
            level = "replayable" if layer_pass["replay"] else "traceable"
    if kind == "context_snapshot":
        from .research_context import research_context_issues

        context_issues = research_context_issues(payload, objects=objects)
        context_checks = [
            _check(
                "context_identity_valid",
                "Context, source closure, policy, and memory hashes validate",
                not context_issues,
                "trace",
            ),
            _check(
                "context_complete",
                "Context snapshot retained its required decision closure",
                payload.get("complete") is True,
                "replay",
            ),
        ]
        checks.extend(context_checks)
        blockers.extend(
            check["code"] for check in context_checks if not check["passed"]
        )
        proof_warnings.extend(context_issues)
        if not context_issues and payload.get("complete") is True:
            level = "replayable"
        elif not context_issues:
            level = "traceable"
        else:
            level = "recorded"
    if kind in {"review", "gate_decision", "agent_evaluation"}:
        from .research_context import research_context_issues

        context_ids = _targets(
            item,
            objects,
            kinds={"context_snapshot"},
            relations={"depends_on"},
        )
        required_context = payload.get("context_required") is True
        valid_context = False
        for context_id in context_ids:
            context_payload = objects[context_id].get("payload") or {}
            if context_payload.get("context_hash") == payload.get(
                "context_hash"
            ) and not research_context_issues(context_payload, objects=objects):
                valid_context = True
                break
        context_check = _check(
            "decision_context_bound",
            "Decision is bound to its exact evidence and memory snapshot",
            valid_context if required_context else True,
            "trace",
        )
        checks.append(context_check)
        if not context_check["passed"]:
            blockers.append(context_check["code"])
            level = "recorded"
        elif not context_ids:
            proof_warnings.append("legacy_decision_context_unbound")
    return {
        "level": level,
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(proof_warnings)),
        "contested": contested,
    }


def _project_ara_snapshot(
    graph: Mapping[str, Any],
    *,
    index: int,
    disclose: bool,
    manifest: Mapping[str, Any] | None = None,
    lock: Mapping[str, Any] | None = None,
    history_rows: Sequence[Mapping[str, Any]] = (),
    history_errors: Sequence[str] = (),
    archived_manifests: Mapping[str, Mapping[str, Any] | None] | None = None,
    expected_manifest_hash: str | None = None,
    expected_graph_hash: str | None = None,
    repository_path: str | None = None,
    snapshot_ref: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    analysis = analyze_exploration_graph(graph)
    prefix = f"ara:{index}:"
    nodes: list[dict[str, Any]] = []
    node_ids: dict[str, str] = {}
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping) or not str(raw.get("id") or ""):
            continue
        raw_id = str(raw["id"])
        node_ids[raw_id] = prefix + raw_id
        summary = (
            " ".join(str(raw.get("analysis") or raw.get("stage") or raw_id).split())
            if disclose
            else "experiment node"
        )
        node_hash = raw.get("content_hash")
        valid_node_hash = _is_sha256(node_hash)
        isolated = bool((raw.get("execution_isolation") or {}).get("isolated"))
        completed = not bool(raw.get("is_buggy"))
        nodes.append(
            {
                "id": prefix + raw_id,
                "source": "ara",
                "kind": "experiment_node",
                "state": "completed" if completed else "failed",
                "phase": _PHASES["experiment_node"],
                "summary": summary[:160],
                "content_hash": str(node_hash) if valid_node_hash else None,
                "actor": None,
                "created_at": None,
                "proof": {
                    "level": (
                        "replayable"
                        if valid_node_hash and isolated
                        else "traceable" if valid_node_hash else "recorded"
                    ),
                    "checks": [
                        _check(
                            "node_content_addressed",
                            "Experiment node has a content hash",
                            valid_node_hash,
                            "trace",
                        ),
                        _check(
                            "execution_isolated",
                            "Experiment executed in an isolated backend",
                            isolated,
                            "replay",
                        ),
                    ],
                    "blockers": [
                        code
                        for code, passed in (
                            ("node_content_addressed", valid_node_hash),
                            ("execution_isolated", isolated),
                        )
                        if not passed
                    ],
                    "warnings": [],
                    "contested": False,
                },
                "source_ref": snapshot_ref or f"ara:{index}",
            }
        )
    context_node_ids: dict[str, str] = {}
    edges: list[dict[str, Any]] = []
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping):
            continue
        target = node_ids.get(str(raw.get("id") or ""))
        if not target:
            continue
        for context_ref in sorted(set(raw.get("context_pack_refs") or [])):
            if not _is_sha256(context_ref):
                continue
            context_id = context_node_ids.setdefault(
                str(context_ref),
                prefix + "context:" + str(context_ref).split(":", 1)[1][:16],
            )
            if not any(node["id"] == context_id for node in nodes):
                nodes.append(
                    {
                        "id": context_id,
                        "source": "ara",
                        "kind": "context_snapshot",
                        "state": "completed",
                        "phase": _PHASES["research_plan"],
                        "summary": "ARA context/memory pack " + str(context_ref)[7:19],
                        "content_hash": str(context_ref),
                        "actor": None,
                        "created_at": None,
                        "proof": {
                            "level": "traceable",
                            "checks": [
                                _check(
                                    "context_pack_addressed",
                                    "Consumed context pack has an immutable identity",
                                    True,
                                    "trace",
                                )
                            ],
                            "blockers": [],
                            "warnings": [
                                "context payload remains in the ARA object store"
                            ],
                            "contested": False,
                        },
                        "source_ref": snapshot_ref or f"ara:{index}",
                    }
                )
            edges.append(
                {
                    "source": context_id,
                    "target": target,
                    "type": "contextualizes",
                    "role": "consumed context and memory",
                    "category": "context",
                }
            )
    valid_ids = {node["id"] for node in nodes}
    raw_edges = graph.get("edges") or []
    seen: set[tuple[str, str]] = set()
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            continue
        source, target = prefix + str(raw.get("parent")), prefix + str(raw.get("child"))
        if source in valid_ids and target in valid_ids:
            seen.add((source, target))
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping):
            continue
        child = prefix + str(raw.get("id"))
        parent_id = raw.get("parent_id")
        if parent_id:
            parent = prefix + str(parent_id)
            if parent in valid_ids and child in valid_ids:
                seen.add((parent, child))
    for source, target in sorted(seen):
        edges.append(
            {
                "source": source,
                "target": target,
                "type": "evolves_to",
                "role": "experiment exploration",
                "category": "evolution",
            }
        )
    manifest_integrity = _manifest_revision_state(
        manifest,
        lock,
        history_rows,
        expected_hash=expected_manifest_hash,
        parse_errors=history_errors,
        archived_manifests=archived_manifests,
    )
    manifest_hash = manifest_integrity.get("current_hash") or manifest_integrity.get(
        "base_hash"
    )
    if not _is_sha256(manifest_hash):
        manifest_hash = None
    graph_hash = canonical_content_hash(dict(graph))
    if expected_graph_hash:
        graph_binding_state = (
            "verified" if graph_hash == expected_graph_hash else "mismatch"
        )
    elif snapshot_ref and snapshot_ref != "worktree":
        graph_binding_state = "commit_bound"
    else:
        graph_binding_state = "unbound_worktree"
    graph_binding = {
        "state": graph_binding_state,
        "ok": graph_binding_state in {"verified", "commit_bound"},
        "graph_hash": graph_hash,
        "expected_graph_hash": expected_graph_hash,
    }
    if graph_binding_state == "mismatch":
        manifest_integrity["valid_hashes"] = []
    roots = [prefix + node_id for node_id in analysis.get("root_ids") or []]
    leaves = [prefix + node_id for node_id in analysis.get("leaf_ids") or []]
    return (
        nodes,
        edges,
        {
            "name": f"ara:{index}",
            "manifest_hash": manifest_hash,
            "manifest_hashes": list(manifest_integrity["valid_hashes"]),
            "manifest_integrity": manifest_integrity,
            "graph_binding": graph_binding,
            "repository_path": repository_path,
            "snapshot_ref": snapshot_ref,
            "node_ids": node_ids,
            "node_count": len(nodes),
            "experiment_node_count": len(node_ids),
            "context_pack_count": len(context_node_ids),
            "context_node_ids": context_node_ids,
            "root_ids": roots,
            "leaf_ids": leaves,
            "integrity": analysis,
        },
    )


def _read_local_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _read_ara(
    root: Path,
    *,
    index: int,
    disclose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    graph_path = root / "exploration_graph.json"
    graph = _read_local_json(graph_path)
    if graph is None:
        raise ResearchGitError(f"ARA exploration graph is invalid or missing: {root}")
    history_raw = None
    history_path = root / "manifest.history.jsonl"
    if history_path.is_file():
        try:
            history_raw = history_path.read_text(encoding="utf-8")
        except OSError:
            history_raw = None
    history_rows, history_errors = _read_json_lines(history_raw)
    archived_manifests: dict[str, Mapping[str, Any] | None] = {}
    for row in history_rows:
        base_hash = str(row.get("base_hash") or "")
        if not _is_sha256(base_hash):
            continue
        archived_manifests[base_hash] = _read_local_json(
            root / "history" / f"{base_hash.split(':', 1)[1]}.json"
        )
    return _project_ara_snapshot(
        graph,
        index=index,
        disclose=disclose,
        manifest=_read_local_json(root / "manifest.json"),
        lock=_read_local_json(root / "manifest.lock"),
        history_rows=history_rows,
        history_errors=history_errors,
        archived_manifests=archived_manifests,
        snapshot_ref="worktree",
    )


def _object_ara_manifest_hashes(item: Mapping[str, Any]) -> set[str]:
    return {manifest_hash for manifest_hash, _graph_hash in _object_ara_bindings(item)}


def _object_ara_bindings(item: Mapping[str, Any]) -> set[tuple[str, str | None]]:
    """Return the strongest ARA revision bindings declared by one object.

    A payload may repeat a manifest-only legacy pointer while provenance carries
    the newer manifest+graph pair. In that case the graph-bound declaration
    wins. Conflicting graph-bound declarations remain separate so one valid
    pointer cannot hide a broken one.
    """

    bindings: set[tuple[str, str | None]] = set()
    for container_name in ("provenance", "payload"):
        container = item.get(container_name) or {}
        if not isinstance(container, Mapping):
            continue
        manifest_hash = str(container.get("ara_manifest_hash") or "")
        if not _is_sha256(manifest_hash):
            continue
        graph_hash = str(container.get("ara_exploration_graph_hash") or "")
        bindings.add((manifest_hash, graph_hash if _is_sha256(graph_hash) else None))
    graph_bound_manifests = {
        manifest_hash
        for manifest_hash, graph_hash in bindings
        if graph_hash is not None
    }
    return {
        (manifest_hash, graph_hash)
        for manifest_hash, graph_hash in bindings
        if graph_hash is not None or manifest_hash not in graph_bound_manifests
    }


def _object_ara_graph_hashes(item: Mapping[str, Any]) -> set[str]:
    return {
        graph_hash
        for _manifest_hash, graph_hash in _object_ara_bindings(item)
        if graph_hash is not None
    }


def _object_context_pack_hashes(item: Mapping[str, Any]) -> set[str]:
    hashes: set[str] = set()
    provenance = item.get("provenance") or {}
    payload = item.get("payload") or {}
    for value in (
        *(
            (provenance.get("context_hashes") or [])
            if isinstance(provenance, Mapping)
            else []
        ),
        *(
            (payload.get("context_pack_refs") or [])
            if isinstance(payload, Mapping)
            else []
        ),
        *(
            (payload.get("context_hashes") or [])
            if isinstance(payload, Mapping)
            else []
        ),
    ):
        if _is_sha256(value):
            hashes.add(str(value))
    return hashes


def _checkpoint_ara_expectation(
    repo: Path,
    commit: str,
    manifest_path: str,
) -> tuple[str | None, str | None]:
    try:
        checkpoint = show_checkpoint(repo, commit)["checkpoint"]
    except ResearchGitError:
        return None, None
    for item in checkpoint.get("ara_manifests") or []:
        if not isinstance(item, Mapping):
            continue
        if _normalise_relative(str(item.get("path") or "")) != manifest_path:
            continue
        manifest_hash = str(item.get("manifest_hash") or "")
        graph_hash = str(item.get("exploration_graph_hash") or "")
        return (
            manifest_hash if _is_sha256(manifest_hash) else None,
            graph_hash if _is_sha256(graph_hash) else None,
        )
    return None, None


def _committed_ara_projection(
    repo: Path,
    commit: str,
    manifest_path: str,
    *,
    index: int,
    disclose: bool,
    expected_manifest_hash: str | None,
    expected_graph_hash: str | None,
    required: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    manifest_value = _json_blob_at_commit(
        repo,
        commit,
        manifest_path,
        required=required,
    )
    if not isinstance(manifest_value, Mapping):
        if required:
            raise ResearchGitError(
                f"ARA manifest is not an object at {commit}:{manifest_path}"
            )
        return None
    ara_dir = PurePosixPath(manifest_path).parent.as_posix()
    graph_path = f"{ara_dir}/exploration_graph.json"
    lock_path = f"{ara_dir}/manifest.lock"
    history_path = f"{ara_dir}/manifest.history.jsonl"
    lock_value = _json_blob_at_commit(repo, commit, lock_path)
    history_rows, history_errors = _read_json_lines(
        _text_blob_at_commit(repo, commit, history_path)
    )
    archived_manifests: dict[str, Mapping[str, Any] | None] = {}
    for row in history_rows:
        base_hash = str(row.get("base_hash") or "")
        if not _is_sha256(base_hash):
            continue
        archive_path = f"{ara_dir}/history/{base_hash.split(':', 1)[1]}.json"
        archive_value = _json_blob_at_commit(repo, commit, archive_path)
        archived_manifests[base_hash] = (
            archive_value if isinstance(archive_value, Mapping) else None
        )
    graph_value = _json_blob_at_commit(repo, commit, graph_path, required=required)
    if not isinstance(graph_value, Mapping):
        if required:
            raise ResearchGitError(
                f"ARA exploration graph is not an object at {commit}:{graph_path}"
            )
        return None
    return _project_ara_snapshot(
        graph_value,
        index=index,
        disclose=disclose,
        manifest=manifest_value,
        lock=lock_value if isinstance(lock_value, Mapping) else None,
        history_rows=history_rows,
        history_errors=history_errors,
        archived_manifests=archived_manifests,
        expected_manifest_hash=expected_manifest_hash,
        expected_graph_hash=expected_graph_hash,
        repository_path=manifest_path,
        snapshot_ref=commit,
    )


def _committed_ara_sources(
    repo: Path,
    commit: str,
    *,
    checkpoint: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    disclose: bool,
    explicit_paths: Iterable[str] = (),
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """Read relevant ARA projections from the exact Git tree at ``commit``."""

    tree_paths = set(
        _run_git(repo, ["ls-tree", "-r", "--name-only", commit]).stdout.splitlines()
    )
    tracked_manifests = {
        path
        for path in tree_paths
        if path.endswith("/manifest.json") and path.startswith("ara/")
    }
    expected_by_path: dict[str, str] = {}
    expected_graph_by_path: dict[str, str] = {}
    for item in checkpoint.get("ara_manifests") or []:
        if not isinstance(item, Mapping):
            continue
        path = _normalise_relative(str(item.get("path") or ""))
        if not path.endswith("/manifest.json") and path != "manifest.json":
            raise ResearchGitError(f"checkpoint ARA path is not a manifest: {path}")
        expected_by_path[path] = str(item.get("manifest_hash") or "")
        graph_hash = str(item.get("exploration_graph_hash") or "")
        if graph_hash:
            expected_graph_by_path[path] = graph_hash
    forced_paths = {_normalise_relative(path) for path in explicit_paths}
    candidates = sorted(tracked_manifests | set(expected_by_path) | forced_paths)
    referenced_hashes = (
        set().union(*(_object_ara_manifest_hashes(item) for item in objects))
        if objects
        else set()
    )

    projected: list[
        tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
    ] = []
    for manifest_path in candidates:
        projection = _committed_ara_projection(
            repo,
            commit,
            manifest_path,
            index=len(projected),
            disclose=disclose,
            expected_manifest_hash=expected_by_path.get(manifest_path),
            expected_graph_hash=expected_graph_by_path.get(manifest_path),
            required=(
                manifest_path in expected_by_path or manifest_path in forced_paths
            ),
        )
        if projection is None:
            continue
        manifest_integrity = projection[2]["manifest_integrity"]
        relevant = (
            manifest_path in expected_by_path
            or manifest_path in forced_paths
            or bool(referenced_hashes & set(manifest_integrity["valid_hashes"]))
        )
        if not relevant:
            continue
        projected.append(projection)

    # A current ref can contain objects bound to several immutable versions of
    # the same ARA graph. Rehydrate those versions from reachable Git history so
    # old evidence is neither attached to the newest graph nor reported missing.
    object_bindings = {
        binding for item in objects for binding in _object_ara_bindings(item)
    }

    def resolved_by_projection(
        binding: tuple[str, str | None],
        projection: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    ) -> bool:
        manifest_hash, graph_hash = binding
        metadata = projection[2]
        return bool(
            manifest_hash in set(metadata.get("manifest_hashes") or [])
            and (
                graph_hash is None
                or graph_hash == metadata["graph_binding"].get("graph_hash")
            )
            and metadata["manifest_integrity"].get("ok")
            and metadata["graph_binding"].get("ok")
        )

    pending = {
        binding
        for binding in object_bindings
        if binding[1] is not None
        and not any(resolved_by_projection(binding, item) for item in projected)
    }
    seen_versions = {
        (
            str(item[2].get("repository_path") or ""),
            str(item[2]["graph_binding"]["graph_hash"]),
        )
        for item in projected
    }
    for manifest_path in candidates:
        if not pending:
            break
        graph_path = (
            PurePosixPath(manifest_path).parent / "exploration_graph.json"
        ).as_posix()
        history = _run_git(
            repo,
            ["log", "--format=%H", commit, "--", graph_path],
            check=False,
        )
        if history.returncode:
            continue
        for historical_commit in history.stdout.splitlines():
            if historical_commit == commit or not pending:
                continue
            expected_manifest_hash, expected_graph_hash = _checkpoint_ara_expectation(
                repo,
                historical_commit,
                manifest_path,
            )
            if expected_manifest_hash is None:
                continue
            projection = _committed_ara_projection(
                repo,
                historical_commit,
                manifest_path,
                index=len(projected),
                disclose=disclose,
                expected_manifest_hash=expected_manifest_hash,
                expected_graph_hash=expected_graph_hash,
                required=False,
            )
            if projection is None:
                continue
            version_key = (
                manifest_path,
                str(projection[2]["graph_binding"]["graph_hash"]),
            )
            if version_key in seen_versions:
                continue
            matched = {
                binding
                for binding in pending
                if resolved_by_projection(binding, projection)
            }
            if not matched:
                continue
            projected.append(projection)
            seen_versions.add(version_key)
            pending.difference_update(matched)
    return projected


def build_research_dag(
    repo: str | Path,
    *,
    ref: str = "HEAD",
    ara_roots: Sequence[str | Path] = (),
    disclose_summaries: bool = True,
) -> dict[str, Any]:
    """Project Research VCS and optional ARA execution trees into one DAG."""

    repository_root = _repository_root(repo)
    checkpoint = show_checkpoint(repo, ref)
    commit = str(checkpoint["commit"])
    rows = list_research_objects_at_ref(repo, commit)
    objects = {str(item["object_id"]): item for item in rows}
    claims = [item for item in rows if item.get("kind") == "claim"]
    closure = (
        audit_research_closure(repo, ref=commit, level="verify") if claims else None
    )
    claim_closure = {
        str(item["claim_id"]): item for item in (closure or {}).get("claims") or []
    }
    blocker_map: dict[str, list[str]] = defaultdict(list)
    warning_map: dict[str, list[str]] = defaultdict(list)
    for item in (closure or {}).get("blockers") or []:
        blocker_map[str(item.get("object_id") or "")].append(str(item["code"]))
    for item in (closure or {}).get("warnings") or []:
        warning_map[str(item.get("object_id") or "")].append(str(item["code"]))
    challenge_targets = {
        str(relation.get("target"))
        for item in rows
        for relation in item.get("relations") or []
        if relation.get("type")
        in {
            "refutes",
            "qualified_refutes",
            "contradicts",
            "challenges_inference",
            "invalidates",
        }
    }
    nodes = [
        {
            "id": str(item["object_id"]),
            "source": "research_vcs",
            "kind": str(item["kind"]),
            "state": str(item["state"]),
            "phase": _PHASES.get(str(item["kind"]), 4),
            "layer": _epistemic_layer(str(item["kind"])),
            "summary": _summary(item, disclose=disclose_summaries),
            "content_hash": str(item["content_hash"]),
            "actor": dict(item.get("actor") or {}),
            "created_at": str(item.get("created_at") or ""),
            "proof": _object_proof(
                item,
                objects,
                claim_closure=claim_closure,
                challenge_targets=challenge_targets,
                closure_blockers=blocker_map,
                closure_warnings=warning_map,
            ),
            "source_ref": commit,
        }
        for item in rows
    ]
    edges: list[dict[str, Any]] = []
    dangling: list[str] = []
    category = {
        "supports": "support",
        "qualified_supports": "support",
        "refutes": "challenge",
        "qualified_refutes": "challenge",
        "contradicts": "challenge",
        "challenges_inference": "challenge",
        "invalidates": "challenge",
        "has_premise": "argument",
        "uses_method": "argument",
        "under_assumption": "argument",
        "addresses_estimand": "argument",
        "has_effect_estimate": "argument",
        "qualifies": "argument",
        "updates": "provenance",
        "selects": "decision",
        "rejects": "decision",
        "evaluates": "verification",
        "reproduces": "verification",
        "attests": "verification",
        "promotes": "evolution",
        "supersedes": "evolution",
        "depends_on": "lineage",
        "derived_from": "lineage",
        "retrieves": "lineage",
        "cites": "lineage",
        "quotes": "lineage",
        "observes": "lineage",
        "generated_by": "lineage",
        "uses_context": "context",
    }
    for item in rows:
        for relation in item.get("relations") or []:
            target = str(relation.get("target") or "")
            if target not in objects:
                dangling.append(f"{item['object_id']}:{target}")
                continue
            relation_type = str(relation.get("type") or "depends_on")
            relation_role = str(relation.get("role") or "")
            edges.append(
                {
                    "source": target,
                    "target": str(item["object_id"]),
                    "type": relation_type,
                    "role": relation_role,
                    "category": (
                        "context"
                        if relation_role in {"context_source", "decision_context"}
                        else (
                            "argument"
                            if relation_role in {"warrant", "premise", "assumption"}
                            else (
                                "decision"
                                if relation_role in {"goal", "budget", "selection"}
                                else (
                                    "theory"
                                    if relation_role
                                    in {
                                        "primary",
                                        "alternative",
                                        "null",
                                        "predictor",
                                        "rival",
                                        "mechanism",
                                    }
                                    else (
                                        "boundary"
                                        if relation_role in {"transfer", "boundary"}
                                        else category.get(relation_type, "lineage")
                                    )
                                )
                            )
                        )
                    ),
                }
            )

    sources: list[dict[str, Any]] = [
        {"name": "research_vcs", "commit": commit, "object_count": len(rows)}
    ]
    ara_metadata: list[dict[str, Any]] = []
    committed_explicit_paths: set[str] = set()
    local_ara_roots: list[Path] = []
    for raw_root in ara_roots:
        requested = Path(raw_root).expanduser().resolve()
        requested_root = (
            requested.parent if requested.name == "manifest.json" else requested
        )
        try:
            relative_root = requested_root.relative_to(repository_root).as_posix()
        except ValueError:
            local_ara_roots.append(requested_root)
            continue
        committed_explicit_paths.add(
            (PurePosixPath(relative_root) / "manifest.json").as_posix()
        )

    committed_sources = _committed_ara_sources(
        repository_root,
        commit,
        checkpoint=checkpoint["checkpoint"],
        objects=rows,
        disclose=disclose_summaries,
        explicit_paths=committed_explicit_paths,
    )
    for ara_nodes, ara_edges, metadata_row in committed_sources:
        nodes.extend(ara_nodes)
        edges.extend(ara_edges)
        ara_metadata.append(metadata_row)

    for raw_root in local_ara_roots:
        index = len(ara_metadata)
        ara_nodes, ara_edges, metadata_row = _read_ara(
            raw_root,
            index=index,
            disclose=disclose_summaries,
        )
        nodes.extend(ara_nodes)
        edges.extend(ara_edges)
        ara_metadata.append(metadata_row)

    for metadata_row in ara_metadata:
        sources.append(
            {
                "name": metadata_row["name"],
                "manifest_hash": metadata_row["manifest_hash"],
                "manifest_hashes": metadata_row["manifest_hashes"],
                "manifest_integrity": {
                    key: value
                    for key, value in metadata_row["manifest_integrity"].items()
                    if key != "valid_hashes"
                },
                "graph_binding": metadata_row["graph_binding"],
                "repository_path": metadata_row.get("repository_path"),
                "snapshot_ref": metadata_row.get("snapshot_ref"),
                "object_count": metadata_row["node_count"],
                "experiment_node_count": metadata_row.get("experiment_node_count", 0),
                "context_pack_count": metadata_row.get("context_pack_count", 0),
                "integrity": {
                    "is_dag": metadata_row["integrity"]["is_dag"],
                    "error_count": metadata_row["integrity"]["error_count"],
                    "warning_count": metadata_row["integrity"]["warning_count"],
                },
            }
        )
    source_by_name = {str(item["name"]): item for item in sources}
    resolved_bindings: set[tuple[str, str, str | None]] = set()
    for ara in ara_metadata:
        manifest_hashes = set(ara.get("manifest_hashes") or [])
        bound_object_ids: list[str] = []
        if not manifest_hashes:
            source_by_name[ara["name"]]["binding_count"] = 0
            source_by_name[ara["name"]]["bound_object_ids"] = []
            continue
        for object_id, item in objects.items():
            matching_bindings = {
                (manifest_hash, graph_hash)
                for manifest_hash, graph_hash in _object_ara_bindings(item)
                if manifest_hash in manifest_hashes
                and (
                    graph_hash is None
                    or graph_hash == ara["graph_binding"]["graph_hash"]
                )
            }
            if not matching_bindings:
                continue
            bound_object_ids.append(object_id)
            payload = item.get("payload") or {}
            node_id = (
                str(payload.get("node_id") or "")
                if isinstance(payload, Mapping)
                else ""
            )
            anchors = (
                [ara["node_ids"][node_id]]
                if node_id in ara["node_ids"]
                else ara["leaf_ids"]
            )
            if anchors:
                resolved_bindings.update(
                    (object_id, manifest_hash, graph_hash)
                    for manifest_hash, graph_hash in matching_bindings
                )
            for source_node in anchors:
                verified_binding = bool(
                    ara["manifest_integrity"]["ok"]
                    and ara["graph_binding"]["ok"]
                    and ara["integrity"]["is_dag"]
                )
                edges.append(
                    {
                        "source": source_node,
                        "target": object_id,
                        "type": "anchors",
                        "role": (
                            "ARA manifest revision evidence"
                            if verified_binding
                            else "unverified external ARA lineage"
                        ),
                        "category": "verification" if verified_binding else "lineage",
                    }
                )
        source_by_name[ara["name"]]["binding_count"] = len(bound_object_ids)
        source_by_name[ara["name"]]["bound_object_ids"] = sorted(bound_object_ids)

    for ara in ara_metadata:
        context_node_ids = ara.get("context_node_ids") or {}
        context_bound_objects: set[str] = set()
        for object_id, item in objects.items():
            for context_hash in sorted(
                _object_context_pack_hashes(item) & set(context_node_ids)
            ):
                edges.append(
                    {
                        "source": context_node_ids[context_hash],
                        "target": object_id,
                        "type": "contextualizes",
                        "role": "exact ARA context pack consumed by Research Object",
                        "category": "context",
                    }
                )
                context_bound_objects.add(object_id)
        source_by_name[ara["name"]]["context_binding_count"] = len(
            context_bound_objects
        )
        source_by_name[ara["name"]]["context_bound_object_ids"] = sorted(
            context_bound_objects
        )

    unresolved_ara_bindings = sorted(
        (object_id, manifest_hash, graph_hash)
        for object_id, item in objects.items()
        for manifest_hash, graph_hash in _object_ara_bindings(item)
        if (object_id, manifest_hash, graph_hash) not in resolved_bindings
    )

    for node in nodes:
        node.setdefault("layer", _epistemic_layer(str(node.get("kind") or "")))

    graph_for_analysis = {
        "nodes": [{"id": node["id"]} for node in nodes],
        "edges": [
            {"parent": edge["source"], "child": edge["target"]} for edge in edges
        ],
    }
    analysis = analyze_exploration_graph(graph_for_analysis)
    for ara in ara_metadata:
        source_integrity = ara["integrity"]
        for issue in source_integrity.get("issues") or []:
            copied = dict(issue)
            copied["code"] = f"{ara['name']}_{copied.get('code', 'invalid')}"
            copied["path"] = f"{ara['name']}:{copied.get('path', '')}"
            analysis["issues"].append(copied)
            if copied.get("severity") == "error":
                analysis["error_count"] += 1
            elif copied.get("severity") == "warning":
                analysis["warning_count"] += 1
        if source_integrity.get("error_count"):
            analysis["is_dag"] = False
        manifest_integrity = ara["manifest_integrity"]
        severity = "error" if manifest_integrity["state"] == "tampered" else "warning"
        for message in manifest_integrity.get("issues") or []:
            analysis["issues"].append(
                {
                    "severity": severity,
                    "code": f"{ara['name']}_manifest_{manifest_integrity['state']}",
                    "message": message,
                    "path": f"{ara['name']}:manifest.json",
                }
            )
            if severity == "error":
                analysis["error_count"] += 1
                analysis["is_dag"] = False
            else:
                analysis["warning_count"] += 1
        graph_binding = ara["graph_binding"]
        if graph_binding["state"] in {"mismatch", "unbound_worktree"}:
            graph_severity = (
                "error" if graph_binding["state"] == "mismatch" else "warning"
            )
            analysis["issues"].append(
                {
                    "severity": graph_severity,
                    "code": f"{ara['name']}_graph_{graph_binding['state']}",
                    "message": (
                        "exploration graph does not match its checkpoint binding"
                        if graph_binding["state"] == "mismatch"
                        else "external worktree exploration graph has no commit binding"
                    ),
                    "path": f"{ara['name']}:exploration_graph.json",
                }
            )
            if graph_severity == "error":
                analysis["error_count"] += 1
                analysis["is_dag"] = False
            else:
                analysis["warning_count"] += 1
    if dangling:
        analysis["is_dag"] = False
        analysis["error_count"] += len(dangling)
        analysis["issues"].extend(
            {
                "severity": "error",
                "code": "dangling_research_relation",
                "message": value,
                "path": "edges",
            }
            for value in sorted(dangling)
        )
    if unresolved_ara_bindings:
        analysis["is_dag"] = False
        analysis["error_count"] += len(unresolved_ara_bindings)
        analysis["issues"].extend(
            {
                "severity": "error",
                "code": "unresolved_ara_manifest_binding",
                "message": (
                    f"{object_id} references unavailable ARA {manifest_hash}"
                    + (f" / graph {graph_hash}" if graph_hash is not None else "")
                ),
                "path": "sources",
            }
            for object_id, manifest_hash, graph_hash in unresolved_ara_bindings
        )
    proof_counts = Counter(node["proof"]["level"] for node in nodes)
    theory_frontier, claim_insights, strategy_summary = _strategy_projection(objects)
    base = {
        "schema_version": RESEARCH_DAG_SCHEMA,
        "ref": ref,
        "commit": commit,
        "content_disclosed": bool(disclose_summaries),
        "sources": sources,
        "nodes": sorted(nodes, key=lambda node: (node["phase"], node["id"])),
        "edges": sorted(
            edges,
            key=lambda edge: (
                edge["source"],
                edge["target"],
                edge["type"],
                edge["role"],
            ),
        ),
        "proof_summary": dict(sorted(proof_counts.items())),
        "theory_frontier": theory_frontier,
        "claim_insights": claim_insights,
        "strategy_summary": strategy_summary,
        "scientific_closure": {
            "status": (closure or {}).get("status", "not_applicable"),
            "claim_count": len(claims),
            "blocker_count": len((closure or {}).get("blockers") or []),
            "warning_count": len((closure or {}).get("warnings") or []),
            "content_hash": (closure or {}).get("content_hash"),
        },
        "integrity": analysis,
    }
    graph = {
        **base,
        "generated_at": _now_iso(),
        "graph_hash": canonical_content_hash(base),
    }
    try:
        validate_json(graph, load_schema("research_dag"))
    except ValidationError as exc:  # pragma: no cover - implementation contract
        raise ResearchGitError(
            f"generated research DAG is invalid: {exc.message}"
        ) from exc
    return graph


def render_research_dag_html(
    graph: Mapping[str, Any],
    *,
    title: str = "XScientist Scientific Evidence DAG",
) -> str:
    """Render a self-contained browser with epistemic layers and claim drill-down."""

    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"), default=str)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    title_text = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_text}</title><style>
:root{{color-scheme:light dark;--bg:light-dark(#f7f8fb,#10131a);--panel:light-dark(#fff,#181d27);--ink:light-dark(#18202f,#eef2f8);--muted:light-dark(#667085,#9aa4b5);--line:light-dark(#c9d0dc,#3a4353);--support:#16855b;--challenge:#d14b43;--verify:#7657c8;--evolve:#2f70c9;--context:#b56a13;--theory:#0089a8;--boundary:#9a5b16;--warn:#c07818}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}header{{padding:18px 22px;background:var(--panel);border-bottom:1px solid var(--line)}}h1{{font-size:22px;margin:0 0 8px}}.stats,.controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}.stat{{color:var(--muted)}}.controls{{padding:12px 22px;border-bottom:1px solid var(--line);background:var(--panel)}}label{{font-weight:500}}input,select{{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:7px 9px}}input{{min-width:220px}}main{{display:grid;grid-template-columns:minmax(0,1fr) 360px;min-height:680px}}#canvas{{overflow:auto;padding:18px}}svg{{display:block;background:var(--panel);border:1px solid var(--line);border-radius:8px}}aside{{background:var(--panel);border-left:1px solid var(--line);padding:18px;overflow-wrap:anywhere}}aside h2{{font-size:17px;margin:0 0 10px}}.edge{{fill:none;stroke:var(--line);stroke-width:1.5}}.edge.support{{stroke:var(--support)}}.edge.challenge{{stroke:var(--challenge);stroke-dasharray:5 4}}.edge.verification{{stroke:var(--verify)}}.edge.evolution{{stroke:var(--evolve)}}.edge.context{{stroke:var(--context);stroke-dasharray:2 3}}.edge.theory{{stroke:var(--theory)}}.edge.boundary{{stroke:var(--boundary);stroke-dasharray:7 3}}.node rect{{fill:var(--panel);stroke:var(--line);stroke-width:1.5;rx:8}}.node.verified rect{{stroke:var(--support)}}.node.replayable rect{{stroke:var(--evolve)}}.node.contested rect{{stroke:var(--challenge);stroke-width:2}}.node.recorded rect{{stroke:var(--warn)}}.node text{{fill:var(--ink);font-size:12px;pointer-events:none}}.node .muted{{fill:var(--muted)}}.node:focus{{outline:none}}.node:focus rect{{stroke-width:3}}table{{width:100%;border-collapse:collapse}}td{{padding:7px 0;border-bottom:1px solid var(--line);vertical-align:top}}td:first-child{{width:96px;color:var(--muted)}}code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}ul{{padding-left:20px}}.pass{{color:var(--support)}}.fail{{color:var(--challenge)}}.empty{{padding:28px;color:var(--muted)}}@media(max-width:900px){{main{{grid-template-columns:1fr}}aside{{border-left:0;border-top:1px solid var(--line)}}}}
</style></head><body><header><h1>{title_text}</h1><div class="stats" id="stats"></div></header>
<section class="controls" aria-label="Graph filters"><label for="search">Search</label><input id="search" type="search" placeholder="statement, ID, or kind"><label for="layer">Layer</label><select id="layer"><option value="">All layers</option><option>strategy</option><option>execution</option><option>evidence</option><option>theory</option><option>decision_memory</option><option>evolution</option></select><label for="kind">Kind</label><select id="kind"><option value="">All kinds</option></select><label for="proof">Verification</label><select id="proof"><option value="">All levels</option><option>verified</option><option>replayable</option><option>traceable</option><option>recorded</option><option>contested</option></select></section>
<main><section id="canvas" aria-label="Scientific evidence graph"></section><aside><h2>Selected scientific object</h2><div id="details">Select a node to inspect its reasoning.</div></aside></main>
<script id="dag-data" type="application/json">{payload}</script><script>
const graph=JSON.parse(document.getElementById('dag-data').textContent),allNodes=graph.nodes||[],allEdges=graph.edges||[],byId=new Map(allNodes.map(n=>[n.id,n])),claimInsights=new Map((graph.claim_insights||[]).map(x=>[x.claim_id,x]));
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const kind=document.getElementById('kind'),layer=document.getElementById('layer'),proof=document.getElementById('proof'),search=document.getElementById('search');
[...new Set(allNodes.map(n=>n.kind))].sort().forEach(v=>kind.insertAdjacentHTML('beforeend',`<option>${{esc(v)}}</option>`));
document.getElementById('stats').innerHTML=`<span class="stat">${{allNodes.length}} nodes</span><span class="stat">${{allEdges.length}} relations</span><span class="stat">DAG ${{graph.integrity.is_dag?'valid':'blocked'}}</span><span class="stat">closure ${{esc(graph.scientific_closure.status)}}</span><span class="stat">open anomalies ${{graph.strategy_summary.open_anomaly_count||0}}</span><span class="stat"><code>${{esc(graph.commit.slice(0,12))}}</code></span>`;
const idList=xs=>(xs&&xs.length?`<ul>${{xs.map(x=>`<li><code>${{esc(x)}}</code></li>`).join('')}}</ul>`:'None');
function detail(n){{const checks=(n.proof.checks||[]).map(c=>`<li class="${{c.passed?'pass':'fail'}}">${{c.passed?'✓':'✗'}} ${{esc(c.label)}} <small>(${{esc(c.layer)}})</small></li>`).join(''),insight=claimInsights.get(n.id);let drill='';if(insight){{const next=insight.next_experiment;drill=`<h3>Claim reasoning</h3><table><tr><td>Depth</td><td>${{esc(insight.depth_level)}}</td></tr><tr><td>Decision</td><td class="${{insight.decision_ready?'pass':'fail'}}">${{insight.decision_ready?'ready':'blocked'}}</td></tr><tr><td>Support</td><td>${{idList(insight.supporting_ids)}}</td></tr><tr><td>Refutation</td><td>${{idList(insight.refuting_ids)}}</td></tr><tr><td>Mechanism</td><td>${{idList(insight.mechanism_ids)}}</td></tr><tr><td>Quality</td><td>${{idList(insight.quality_assessment_ids)}}</td></tr><tr><td>Boundaries</td><td>${{idList(insight.boundary_ids)}}</td></tr><tr><td>Next experiment</td><td>${{next?esc(next.summary||next.candidate_id):'None ranked'}}</td></tr><tr><td>Open gaps</td><td>${{insight.gaps.length?esc(insight.gaps.join(', ')):'None'}}</td></tr></table>`}}document.getElementById('details').innerHTML=`<table><tr><td>Summary</td><td>${{esc(n.summary)}}</td></tr><tr><td>ID</td><td><code>${{esc(n.id)}}</code></td></tr><tr><td>Type</td><td>${{esc(n.kind)}} / ${{esc(n.state)}}</td></tr><tr><td>Layer</td><td>${{esc(n.layer)}}</td></tr><tr><td>Source</td><td>${{esc(n.source)}}</td></tr><tr><td>Snapshot</td><td><code>${{esc(n.source_ref||'-')}}</code></td></tr><tr><td>Proof</td><td><strong>${{esc(n.proof.level)}}</strong></td></tr><tr><td>Hash</td><td><code>${{esc(n.content_hash||'-')}}</code></td></tr><tr><td>Actor</td><td>${{esc((n.actor||{{}}).actor_id||'-')}}</td></tr></table>${{drill}}<h3>Verification checks</h3><ul>${{checks||'<li>No specialized checks.</li>'}}</ul>`}}
function render(){{const q=search.value.trim().toLowerCase();const nodes=allNodes.filter(n=>(!layer.value||n.layer===layer.value)&&(!kind.value||n.kind===kind.value)&&(!proof.value||n.proof.level===proof.value)&&(!q||`${{n.id}} ${{n.kind}} ${{n.summary}}`.toLowerCase().includes(q)));const ids=new Set(nodes.map(n=>n.id)),edges=allEdges.filter(e=>ids.has(e.source)&&ids.has(e.target));const buckets=new Map();nodes.forEach(n=>{{if(!buckets.has(n.phase))buckets.set(n.phase,[]);buckets.get(n.phase).push(n)}});[...buckets.values()].forEach(v=>v.sort((a,b)=>a.id.localeCompare(b.id)));const phases=[...buckets.keys()].sort((a,b)=>a-b),boxW=220,boxH=82,gapX=92,gapY=28,pad=34,maxRows=Math.max(1,...[...buckets.values()].map(v=>v.length)),width=Math.max(720,pad*2+phases.length*boxW+Math.max(0,phases.length-1)*gapX),height=Math.max(220,pad*2+maxRows*boxH+Math.max(0,maxRows-1)*gapY),pos=new Map();phases.forEach((p,col)=>buckets.get(p).forEach((n,row)=>pos.set(n.id,{{x:pad+col*(boxW+gapX),y:pad+row*(boxH+gapY)}})));const edgeSvg=edges.map(e=>{{const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)return'';const x1=a.x+boxW,y1=a.y+boxH/2,x2=b.x,y2=b.y+boxH/2,m=x1+(x2-x1)/2;return`<path class="edge ${{esc(e.category)}}" d="M${{x1}} ${{y1}} C${{m}} ${{y1}},${{m}} ${{y2}},${{x2}} ${{y2}}" marker-end="url(#arrow)"/>`}}).join('');const nodeSvg=nodes.map(n=>{{const p=pos.get(n.id),summary=n.summary.length>29?n.summary.slice(0,29)+'…':n.summary;return`<g class="node ${{esc(n.proof.level)}}" data-id="${{esc(n.id)}}" tabindex="0" role="button" aria-label="${{esc(n.kind+' '+n.summary)}}" transform="translate(${{p.x}},${{p.y}})"><rect width="${{boxW}}" height="${{boxH}}"/><text x="12" y="20"><tspan font-weight="600">${{esc(n.kind)}}</tspan></text><text class="muted" x="12" y="40">${{esc(summary)}}</text><text class="muted" x="12" y="60">${{esc(n.layer)}} · ${{esc(n.proof.level)}}</text><text class="muted" x="12" y="76">${{esc(n.id.slice(0,25))}}</text></g>`}}).join('');const canvas=document.getElementById('canvas');canvas.innerHTML=nodes.length?`<svg width="${{width}}" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>${{edgeSvg}}${{nodeSvg}}</svg>`:'<div class="empty">No nodes match these filters.</div>';canvas.querySelectorAll('.node').forEach(el=>{{const show=()=>detail(byId.get(el.dataset.id));el.addEventListener('click',show);el.addEventListener('keydown',e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();show()}}}})}});if(nodes[0])detail(nodes[0])}}
[search,layer,kind,proof].forEach(el=>el.addEventListener(el===search?'input':'change',render));render();
</script></body></html>"""


def export_research_dag(
    repo: str | Path,
    destination: str | Path,
    *,
    ref: str = "HEAD",
    ara_roots: Sequence[str | Path] = (),
    disclose_summaries: bool = True,
) -> dict[str, Any]:
    graph = build_research_dag(
        repo,
        ref=ref,
        ara_roots=ara_roots,
        disclose_summaries=disclose_summaries,
    )
    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "research-dag.json"
    html_path = output / "research-dag.html"
    atomic_write_json(json_path, graph)
    atomic_write_text(html_path, render_research_dag_html(graph))
    return {
        "graph": graph,
        "json": json_path.as_posix(),
        "html": html_path.as_posix(),
    }


__all__ = [
    "RESEARCH_DAG_SCHEMA",
    "build_research_dag",
    "export_research_dag",
    "render_research_dag_html",
]
