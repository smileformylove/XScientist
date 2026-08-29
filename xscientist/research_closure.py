"""Payload-free scientific closure audits for Research VCS refs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash
from ai_scientist.protocol.research_vcs import (
    research_payload_issues,
    research_profile_status,
)
from ai_scientist.protocol.schemas import load_schema

from .research_authority import PRODUCER_LINEAGE_RELATIONS
from .research_git import (
    ResearchGitError,
    list_research_objects_at_ref,
    show_checkpoint,
    verify_research_repository,
)

RESEARCH_CLOSURE_SCHEMA = "xscientist.research-closure.v1"
RESEARCH_CLOSURE_LEVELS = ("trace", "replay", "verify")
REPRODUCTION_RECEIPT_V2 = "xscientist.reproduction-receipt.v2"
REPRODUCTION_TARGET_POLICY_V2 = "xscientist.reproduction-target-binding.v2"
_REPRODUCTION_ISOLATION_COMMON = {
    "isolated": False,
    "security_boundary": False,
    "process_tree_termination_guaranteed": False,
    "filesystem": "host_visible",
    "network": "host_unrestricted",
}
_ALLOWED_REPRODUCTION_ISOLATION = (
    {
        **_REPRODUCTION_ISOLATION_COMMON,
        "environment": "sanitized",
        "environment_scope": "variables_only",
        "process_tree": "best_effort_process_group",
        "process_control": "posix_process_group_best_effort",
    },
    {
        **_REPRODUCTION_ISOLATION_COMMON,
        "environment": "sanitized",
        "environment_scope": "variables_only",
        "process_tree": "parent_only_no_tree_guarantee",
        "process_control": "parent_process_only",
    },
    {
        **_REPRODUCTION_ISOLATION_COMMON,
        "environment": "legacy_unknown",
        "environment_scope": "legacy_unknown",
        "process_tree": "legacy_unknown_no_tree_guarantee",
        "process_control": "legacy_unknown",
    },
)
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
_SCIENTIFIC_LINEAGE_RELATIONS = _ARGUMENT_RELATIONS | {
    "depends_on",
    "derived_from",
    "quotes",
    "uses_context",
}
_ACTIVE_CLOSURE_STATES = {"completed", "verified", "promoted"}
_CLAIM_SUPPORT_RELATIONS = {"supports", "qualified_supports", "derived_from"}
_CLAIM_CHALLENGE_RELATIONS = {
    "refutes",
    "qualified_refutes",
    "contradicts",
    "challenges_inference",
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


def _object_binding_rows(
    object_ids: Iterable[str],
    objects: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "object_id": object_id,
            "content_hash": str(objects[object_id].get("content_hash") or ""),
        }
        for object_id in sorted(set(object_ids))
        if object_id in objects
    ]


def _claim_reproduction_binding(
    claim_id: str,
    closure_ids: Iterable[str],
    objects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    core = {
        "claim_id": claim_id,
        "claim_content_hash": str(objects[claim_id].get("content_hash") or ""),
        "objects": _object_binding_rows({claim_id, *closure_ids}, objects),
    }
    return {**core, "closure_hash": canonical_content_hash(core)}


def _producer_provenance_at_ref(
    target_ids: Iterable[str],
    objects: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    pending = list(sorted(set(target_ids)))
    visited: set[str] = set()
    actors: set[str] = set()
    while pending:
        object_id = pending.pop()
        if object_id in visited or object_id not in objects:
            continue
        visited.add(object_id)
        actor_id = " ".join(
            str((objects[object_id].get("actor") or {}).get("actor_id") or "").split()
        )
        if actor_id:
            actors.add(actor_id)
        for target in _targets(
            objects[object_id], relation_types=PRODUCER_LINEAGE_RELATIONS
        ):
            if target not in visited:
                pending.append(target)
    return sorted(actors), sorted(visited)


def _normalized_literature_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _normalized_literature_identifier(kind: str, value: Any) -> str:
    text = _normalized_literature_text(value)
    if not text:
        return ""
    folded = text.casefold()
    if kind == "doi":
        return re.sub(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)", "", folded)
    if kind == "pmid":
        return re.sub(r"^pmid:\s*", "", folded)
    if kind == "arxiv_id":
        return re.sub(
            r"^(?:arxiv:\s*|https?://arxiv\.org/(?:abs|pdf)/)",
            "",
            folded,
        ).removesuffix(".pdf")
    if kind == "url":
        parsed = urlsplit(text)
        if parsed.scheme and parsed.netloc:
            return urlunsplit(
                (
                    parsed.scheme.casefold(),
                    parsed.netloc.casefold(),
                    parsed.path.rstrip("/") or "/",
                    parsed.query,
                    "",
                )
            )
        return folded.rstrip("/")
    return folded


def _selected_candidate_matches_source(
    receipt_payload: Mapping[str, Any], source_payload: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    candidates = receipt_payload.get("candidates")
    if not isinstance(candidates, list):
        return []
    selected = [
        item
        for item in candidates
        if isinstance(item, Mapping)
        and (
            item.get("selection_status") == "selected"
            or ("selection_status" not in item and item.get("selected") is True)
        )
    ]
    identifiers = {
        key: normalized
        for key in ("doi", "pmid", "arxiv_id", "url")
        if (
            normalized := _normalized_literature_identifier(
                key, source_payload.get(key)
            )
        )
    }
    if identifiers:
        matches: list[Mapping[str, Any]] = []
        for item in selected:
            candidate_identifiers = {
                key: normalized
                for key in identifiers
                if (normalized := _normalized_literature_identifier(key, item.get(key)))
            }
            shared = set(identifiers).intersection(candidate_identifiers)
            if shared and all(
                candidate_identifiers[key] == identifiers[key] for key in shared
            ):
                matches.append(item)
        return matches
    title = _normalized_literature_text(source_payload.get("title")).casefold()
    return [
        item
        for item in selected
        if _normalized_literature_text(item.get("title")).casefold() == title
    ]


def _source_update_time(item: Mapping[str, Any]) -> datetime | None:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    try:
        parsed = datetime.fromisoformat(
            str(payload.get("checked_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _is_retraction_update(item: Mapping[str, Any]) -> bool:
    payload = item.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    return str(payload.get("status") or "").strip().lower() in {
        "retracted",
        "withdrawn",
        "invalid",
    } or str(payload.get("update_type") or "").strip().lower() in {
        "retraction",
        "withdrawal",
    }


def _valid_source_update_supersessions(
    objects: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Return retractions cleared by a semantically valid reinstatement only."""

    valid: set[str] = set()
    for successor in objects.values():
        if (
            successor.get("kind") != "source_update"
            or successor.get("state") not in _ACTIVE_CLOSURE_STATES
        ):
            continue
        successor_payload = successor.get("payload")
        successor_payload = (
            successor_payload if isinstance(successor_payload, Mapping) else {}
        )
        if (
            str(successor_payload.get("update_type") or "").strip().lower()
            != "reinstatement"
            or str(successor_payload.get("status") or "").strip().lower()
            in {"retracted", "withdrawn", "invalid"}
            or not str(successor_payload.get("notice_id") or "").strip()
        ):
            continue
        successor_sources = _targets(successor, relation_types=("updates",))
        if successor_sources != [str(successor_payload.get("source_id") or "")]:
            continue
        successor_time = _source_update_time(successor)
        if successor_time is None:
            continue
        superseded_targets = _targets(successor, relation_types=("supersedes",))
        if len(superseded_targets) != 1:
            continue
        for target_id in superseded_targets:
            target = objects.get(target_id, {})
            target_payload = target.get("payload")
            target_payload = (
                target_payload if isinstance(target_payload, Mapping) else {}
            )
            target_sources = _targets(target, relation_types=("updates",))
            target_time = _source_update_time(target)
            if (
                target.get("kind") == "source_update"
                and target.get("state") in _ACTIVE_CLOSURE_STATES
                and _is_retraction_update(target)
                and target_sources == successor_sources
                and target_sources
                and target_sources == [str(target_payload.get("source_id") or "")]
                and _targets(target, relation_types=("invalidates",)) == target_sources
                and str(successor_payload.get("provider") or "")
                == str(target_payload.get("provider") or "")
                and target_time is not None
                and successor_time > target_time
            ):
                valid.add(target_id)
    return valid


def _claim_identity(
    claim: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """Return the stable semantic identity used for implicit promotions."""

    payload = claim.get("payload") or {}
    statement = next(
        (
            " ".join(str(payload.get(key)).split())
            for key in ("statement", "text", "claim")
            if payload.get(key) not in (None, "")
        ),
        "",
    )
    if statement:
        # Evidence and gate objects evolve while the scientific proposition stays
        # stable.  Making those mutable anchors part of claim identity allowed a
        # revised claim to shed challenges aimed at an earlier immutable version.
        return statement, str(payload.get("scope_hash") or "")
    # ``claim_hash`` is caller supplied and therefore only a last-resort identity
    # for payloads that genuinely have no textual proposition.
    return str(payload.get("claim_hash") or ""), str(payload.get("scope_hash") or "")


def _effective_superseded_ids(
    objects: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Return objects removed from the active scientific frontier.

    Immutable history remains available to trace/replay, but an explicit active
    successor (or the object's own ``superseded`` state) removes an old signal
    from the current verification decision.
    """

    superseded = {
        str(object_id)
        for object_id, item in objects.items()
        if item.get("state") == "superseded"
    }
    valid_source_update_supersessions = _valid_source_update_supersessions(objects)
    superseded.update(
        target
        for item in objects.values()
        if item.get("state") in _ACTIVE_CLOSURE_STATES
        for target in _targets(item, relation_types=("supersedes",))
        if target in objects
        and (
            objects[target].get("kind") != "source_update"
            or target in valid_source_update_supersessions
        )
    )
    return superseded


def _is_active_closure_object(
    object_id: str,
    objects: Mapping[str, Mapping[str, Any]],
    superseded_ids: set[str],
) -> bool:
    item = objects.get(object_id, {})
    return (
        object_id not in superseded_ids and item.get("state") in _ACTIVE_CLOSURE_STATES
    )


def _claim_alias_ids(
    claim: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Return immutable claim versions sharing the same semantic identity."""

    claim_id = str(claim["object_id"])
    aliases = {claim_id}
    pending = [claim_id]
    claim_rows = {
        object_id: item
        for object_id, item in objects.items()
        if item.get("kind") == "claim"
    }
    reverse_supersedes: dict[str, set[str]] = {}
    for object_id, item in claim_rows.items():
        for target in _targets(item, relation_types=("supersedes",)):
            if target in claim_rows:
                reverse_supersedes.setdefault(target, set()).add(object_id)

    while pending:
        current_id = pending.pop()
        current = claim_rows.get(current_id)
        if current is None:
            continue
        identity = _claim_identity(current, objects)
        neighbors = set(reverse_supersedes.get(current_id, set()))
        neighbors.update(
            target
            for target in _targets(current, relation_types=("supersedes",))
            if target in claim_rows
        )
        if identity[0]:
            neighbors.update(
                object_id
                for object_id, item in claim_rows.items()
                if _claim_identity(item, objects) == identity
            )
        for neighbor in sorted(neighbors - aliases):
            aliases.add(neighbor)
            pending.append(neighbor)
    return aliases


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
    repo: str | Path,
    objects: Mapping[str, Mapping[str, Any]],
    reproduction_target_ids: set[str],
    expected_claim_binding: Mapping[str, Any],
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
    try:
        validate_json(receipt, load_schema("reproduction_receipt"))
    except ValidationError as exc:
        issues.append(f"reproduction receipt schema is invalid: {exc.message}")
    if receipt.get("content_hash") != expected_hash:
        issues.append("reproduction receipt content hash mismatch")
    if receipt.get("receipt_id") != expected_id:
        issues.append("reproduction receipt identifier mismatch")
    if payload.get("receipt_hash") != expected_hash:
        issues.append("reproduction object is not bound to the embedded receipt")
    if receipt.get("schema_version") != REPRODUCTION_RECEIPT_V2:
        issues.append(
            "legacy reproduction receipt is not bound to an exact checkpoint, target closure, and execution result"
        )
        return issues

    execution_isolation = receipt.get("execution_isolation")
    if not isinstance(execution_isolation, Mapping) or not any(
        dict(execution_isolation) == allowed
        for allowed in _ALLOWED_REPRODUCTION_ISOLATION
    ):
        issues.append(
            "reproduction receipt does not persist an honest execution-isolation boundary"
        )

    source_checkpoint: Mapping[str, Any] | None = None
    try:
        source_record = show_checkpoint(repo, str(receipt.get("commit") or ""))
        source_checkpoint = source_record.get("checkpoint") or {}
    except ResearchGitError as exc:
        issues.append(f"reproduction source checkpoint cannot be verified: {exc}")
    checkpoint_binding = receipt.get("checkpoint_binding")
    if source_checkpoint is not None:
        checkpoint_core = {
            "commit": str(source_record.get("commit") or ""),
            "checkpoint_id": str(source_checkpoint.get("checkpoint_id") or ""),
            "checkpoint_content_hash": str(source_checkpoint.get("content_hash") or ""),
        }
        expected_checkpoint_binding = {
            **checkpoint_core,
            "binding_hash": canonical_content_hash(checkpoint_core),
        }
        if checkpoint_binding != expected_checkpoint_binding:
            issues.append(
                "reproduction checkpoint binding disagrees with the exact resolved Git checkpoint"
            )
        if (
            receipt.get("checkpoint_id") != checkpoint_core["checkpoint_id"]
            or receipt.get("checkpoint_hash")
            != checkpoint_core["checkpoint_content_hash"]
        ):
            issues.append("reproduction top-level checkpoint identity is inconsistent")

    target_binding = receipt.get("target_binding")
    if not isinstance(target_binding, Mapping):
        issues.append("reproduction receipt lacks a v2 target binding")
    else:
        source_commit = str(receipt.get("commit") or "")
        if str(target_binding.get("audit_commit") or "") != source_commit:
            issues.append(
                "reproduction target checkpoint differs from the executed source checkpoint"
            )
        if isinstance(checkpoint_binding, Mapping) and (
            target_binding.get("audit_commit") != checkpoint_binding.get("commit")
            or target_binding.get("audit_checkpoint_id")
            != checkpoint_binding.get("checkpoint_id")
            or target_binding.get("audit_checkpoint_hash")
            != checkpoint_binding.get("checkpoint_content_hash")
        ):
            issues.append(
                "reproduction source and target checkpoint bindings are inconsistent"
            )
        target_core = {
            key: value for key, value in target_binding.items() if key != "binding_hash"
        }
        if target_binding.get("policy") != REPRODUCTION_TARGET_POLICY_V2:
            issues.append("reproduction target-binding policy is invalid")
        if target_binding.get("binding_hash") != canonical_content_hash(target_core):
            issues.append("reproduction target-binding hash mismatch")
        expected_target_objects = _object_binding_rows(reproduction_target_ids, objects)
        if target_binding.get("target_objects") != expected_target_objects:
            issues.append(
                "reproduction target objects do not exactly match its immutable reproduces relations"
            )
        claim_targets = sorted(
            object_id
            for object_id in reproduction_target_ids
            if objects.get(object_id, {}).get("kind") == "claim"
        )
        raw_claim_closures = target_binding.get("claim_closures")
        claim_closures = (
            raw_claim_closures if isinstance(raw_claim_closures, list) else []
        )
        if [
            str(item.get("claim_id") or "") for item in claim_closures
        ] != claim_targets:
            issues.append(
                "reproduction claim-closure bindings do not exactly match reproduced claims"
            )
        for binding in claim_closures:
            if not isinstance(binding, Mapping):
                issues.append("reproduction claim-closure binding is malformed")
                continue
            binding_core = {
                key: value for key, value in binding.items() if key != "closure_hash"
            }
            if binding.get("closure_hash") != canonical_content_hash(binding_core):
                issues.append("reproduction claim-closure hash mismatch")
            bound_ids = [
                str(item.get("object_id") or "")
                for item in binding.get("objects") or []
                if isinstance(item, Mapping)
            ]
            if binding.get("objects") != _object_binding_rows(bound_ids, objects):
                issues.append(
                    "reproduction claim closure contains stale or non-canonical object bindings"
                )
        if dict(expected_claim_binding) not in claim_closures:
            issues.append(
                "reproduction receipt does not bind the exact current audited claim closure"
            )
        try:
            audit_record = show_checkpoint(
                repo, str(target_binding.get("audit_commit") or "")
            )
            audit_checkpoint = audit_record.get("checkpoint") or {}
            if (
                target_binding.get("audit_commit") != audit_record.get("commit")
                or target_binding.get("audit_checkpoint_id")
                != audit_checkpoint.get("checkpoint_id")
                or target_binding.get("audit_checkpoint_hash")
                != audit_checkpoint.get("content_hash")
            ):
                issues.append(
                    "reproduction target audit checkpoint binding is inconsistent"
                )
            audit_objects = {
                str(item["object_id"]): item
                for item in list_research_objects_at_ref(
                    repo, str(audit_record.get("commit") or "")
                )
            }
            if target_binding.get("target_objects") != _object_binding_rows(
                reproduction_target_ids, audit_objects
            ):
                issues.append(
                    "reproduction targets were not present with identical content at the bound audit checkpoint"
                )
        except ResearchGitError as exc:
            issues.append(f"reproduction target checkpoint cannot be verified: {exc}")

    execution_result = receipt.get("execution_result")
    execution_core = {
        key: receipt.get(key)
        for key in (
            "command_hash",
            "reproduction_level",
            "verdict",
            "objects_complete",
            "executed",
            "returncode",
            "timed_out",
            "stdout_hash",
            "stderr_hash",
            "stdout_truncated",
            "stderr_truncated",
            "output_capture",
            "max_output_chars",
        )
    }
    expected_execution_result = {
        **execution_core,
        "result_hash": canonical_content_hash(execution_core),
    }
    if execution_result != expected_execution_result:
        issues.append("reproduction execution-result binding is invalid")
    if source_checkpoint is not None:
        source_command = str(
            (source_checkpoint.get("reproduce") or {}).get("command") or ""
        ).strip()
        expected_command_hash = (
            canonical_content_hash(source_command) if source_command else None
        )
        if receipt.get("executed") is True and not source_command:
            issues.append(
                "computational reproduction source checkpoint declares no command"
            )
        if receipt.get("command_hash") != expected_command_hash:
            issues.append(
                "reproduction command hash disagrees with the bound source checkpoint"
            )
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
        expected_producers, expected_lineage = _producer_provenance_at_ref(
            reproduction_target_ids, objects
        )
        if independence.get("target_ids") != sorted(reproduction_target_ids):
            issues.append(
                "reproduction independence receipt does not bind the exact reproduced targets"
            )
        if independence.get("producer_actor_ids") != expected_producers:
            issues.append(
                "reproduction independence receipt producer provenance is stale"
            )
        if independence.get("lineage_object_ids") != expected_lineage:
            issues.append(
                "reproduction independence receipt lineage provenance is stale"
            )
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
    repo: str | Path,
    target_level: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    claim_id = str(claim["object_id"])
    superseded_ids = _effective_superseded_ids(objects)
    claim_alias_ids = _claim_alias_ids(claim, objects)
    active_superseders_by_target: dict[str, set[str]] = {}
    for successor_id, successor in objects.items():
        if successor.get("state") not in _ACTIVE_CLOSURE_STATES:
            continue
        for target in _targets(successor, relation_types=("supersedes",)):
            if target in objects:
                active_superseders_by_target.setdefault(target, set()).add(successor_id)
    direct = _targets(claim, relation_types=("depends_on",))
    argument_ids = {
        target
        for target in direct
        if objects.get(target, {}).get("kind") in _ARGUMENT_KINDS
    }
    experimental_evidence_id_set = set(_kind_ids(direct, objects, "evidence"))
    passage_id_set = set(_kind_ids(direct, objects, "passage_evidence"))
    gate_ids = _kind_ids(direct, objects, "gate_decision")
    claim_signal_relations = _CLAIM_SUPPORT_RELATIONS | _CLAIM_CHALLENGE_RELATIONS
    # Evidence may also point at any immutable version of a claim.  Only active
    # incoming signals enter the current closure; superseded signals remain in
    # repository history without continuing to decide verification.
    for object_id, item in objects.items():
        if not _is_active_closure_object(object_id, objects, superseded_ids):
            continue
        if not claim_alias_ids.intersection(
            _targets(item, relation_types=claim_signal_relations)
        ):
            continue
        if item.get("kind") == "evidence":
            experimental_evidence_id_set.add(object_id)
        elif item.get("kind") == "passage_evidence":
            passage_id_set.add(object_id)
        elif item.get("kind") in _ARGUMENT_KINDS:
            argument_ids.add(object_id)

    active_challenge_ids: set[str] = set()
    superseded_challenge_ids: set[str] = set()
    challenge_resolution_ids: set[str] = set()
    scientific_closure_target_ids: set[str] = set()
    while True:
        previous = (
            frozenset(argument_ids),
            frozenset(experimental_evidence_id_set),
            frozenset(passage_id_set),
            frozenset(active_challenge_ids),
            frozenset(superseded_challenge_ids),
            frozenset(challenge_resolution_ids),
        )
        queue = list(argument_ids)
        visited_arguments: set[str] = set()
        while queue:
            current = queue.pop()
            if current in visited_arguments:
                continue
            visited_arguments.add(current)
            premises = _targets(objects[current], relation_types=_ARGUMENT_RELATIONS)
            experimental_evidence_id_set.update(
                _kind_ids(premises, objects, "evidence")
            )
            passage_id_set.update(_kind_ids(premises, objects, "passage_evidence"))
            for target in premises:
                if (
                    objects.get(target, {}).get("kind") in _ARGUMENT_KINDS
                    and target not in argument_ids
                ):
                    argument_ids.add(target)
                    queue.append(target)

        closure_targets = {
            *claim_alias_ids,
            *argument_ids,
            *experimental_evidence_id_set,
            *passage_id_set,
        }
        lineage_queue = list(closure_targets - claim_alias_ids)
        visited_lineage: set[str] = set()
        while lineage_queue:
            current_id = lineage_queue.pop()
            if current_id in visited_lineage or current_id not in objects:
                continue
            visited_lineage.add(current_id)
            for target in _targets(
                objects[current_id],
                relation_types=_SCIENTIFIC_LINEAGE_RELATIONS,
            ):
                if target in objects and target not in closure_targets:
                    closure_targets.add(target)
                    lineage_queue.append(target)
        scientific_closure_target_ids = set(closure_targets)
        for object_id, item in objects.items():
            if item.get("state") not in {
                *_ACTIVE_CLOSURE_STATES,
                "superseded",
            }:
                continue
            if not closure_targets.intersection(
                _targets(item, relation_types=_CLAIM_CHALLENGE_RELATIONS)
            ):
                continue
            if object_id in superseded_ids:
                superseded_challenge_ids.add(object_id)
                challenge_resolution_ids.update(
                    active_superseders_by_target.get(object_id, set())
                )
            else:
                active_challenge_ids.add(object_id)
            if item.get("kind") == "evidence":
                experimental_evidence_id_set.add(object_id)
            elif item.get("kind") == "passage_evidence":
                passage_id_set.add(object_id)
            elif item.get("kind") in _ARGUMENT_KINDS:
                argument_ids.add(object_id)
        current = (
            frozenset(argument_ids),
            frozenset(experimental_evidence_id_set),
            frozenset(passage_id_set),
            frozenset(active_challenge_ids),
            frozenset(superseded_challenge_ids),
            frozenset(challenge_resolution_ids),
        )
        if current == previous:
            break

    experimental_evidence_ids = sorted(experimental_evidence_id_set)
    passage_ids = sorted(passage_id_set)
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
    active_source_update_ids = [
        update_id
        for update_id in source_update_ids
        if _is_active_closure_object(update_id, objects, superseded_ids)
    ]
    effective_source_updates: dict[str, str] = {}
    for source_id in source_ids:
        candidates = [
            update_id
            for update_id in active_source_update_ids
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
    reproduction_ids = sorted(
        object_id
        for object_id, item in objects.items()
        if item.get("kind") == "reproduction"
        and claim_id in _targets(item, relation_types=("reproduces",))
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
        elif len(bound_receipts) != 1:
            blockers.append(
                _blocker(
                    "source_search_receipt_ambiguous",
                    source_id,
                    "source selection must bind exactly one retrieval receipt",
                )
            )
        source_payload = objects[source_id].get("payload") or {}
        if len(bound_receipts) == 1:
            receipt_id = bound_receipts[0]
            receipt = objects[receipt_id]
            receipt_payload = receipt.get("payload") or {}
            expected_receipt_binding = {
                "object_id": receipt_id,
                "content_hash": str(receipt.get("content_hash") or ""),
                "receipt_hash": str(receipt_payload.get("receipt_hash") or ""),
            }
            receipt_binding = source_payload.get("receipt_binding")
            if receipt_binding is not None and (
                not isinstance(receipt_binding, Mapping)
                or dict(receipt_binding) != expected_receipt_binding
            ):
                blockers.append(
                    _blocker(
                        "source_receipt_binding_invalid",
                        source_id,
                        "source receipt binding disagrees with immutable retrieval "
                        "lineage",
                    )
                )
            matching_candidates = _selected_candidate_matches_source(
                receipt_payload, source_payload
            )
            if not matching_candidates:
                blockers.append(
                    _blocker(
                        "source_candidate_mismatch",
                        source_id,
                        "source does not match a selected retrieval candidate",
                    )
                )
            elif len(matching_candidates) != 1:
                blockers.append(
                    _blocker(
                        "source_candidate_ambiguous",
                        source_id,
                        "source matches multiple selected retrieval candidates",
                    )
                )
            else:
                candidate_binding = source_payload.get("candidate_binding")
                expected_candidate_binding = {
                    "candidate_hash": canonical_content_hash(
                        dict(matching_candidates[0])
                    ),
                    "selection_status": "selected",
                }
                if candidate_binding is not None and (
                    not isinstance(candidate_binding, Mapping)
                    or dict(candidate_binding) != expected_candidate_binding
                ):
                    blockers.append(
                        _blocker(
                            "source_candidate_binding_invalid",
                            source_id,
                            "source candidate binding does not match the selected "
                            "candidate",
                        )
                    )
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
    for update_id in active_source_update_ids:
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
            continue
        if len(bound_plans) != 1:
            blockers.append(
                _blocker(
                    "receipt_search_plan_ambiguous",
                    receipt_id,
                    "retrieval receipt must bind exactly one locked search plan",
                )
            )
            continue
        plan_id = bound_plans[0]
        plan = objects[plan_id]
        plan_payload = plan.get("payload") or {}
        receipt_payload = objects[receipt_id].get("payload") or {}
        if plan.get("state") != "locked":
            blockers.append(
                _blocker(
                    "receipt_search_plan_unlocked",
                    receipt_id,
                    "retrieval receipt references a search plan that is not locked",
                )
            )
        locked_providers = plan_payload.get("providers") or []
        provider = _normalized_literature_text(receipt_payload.get("provider"))
        if not isinstance(locked_providers, list) or (
            locked_providers
            and provider
            not in {_normalized_literature_text(item) for item in locked_providers}
        ):
            blockers.append(
                _blocker(
                    "receipt_provider_outside_plan",
                    receipt_id,
                    "retrieval provider is not allowed by the locked search plan",
                )
            )
        locked_queries = plan_payload.get("queries")
        query = _normalized_literature_text(receipt_payload.get("query"))
        if not isinstance(locked_queries, list) or query not in {
            _normalized_literature_text(item) for item in locked_queries
        }:
            blockers.append(
                _blocker(
                    "receipt_query_outside_plan",
                    receipt_id,
                    "retrieval query is not one of the locked search-plan queries",
                )
            )
        expected_plan_binding = {
            "object_id": plan_id,
            "content_hash": str(plan.get("content_hash") or ""),
            "search_plan_hash": str(plan_payload.get("search_plan_hash") or ""),
        }
        plan_binding = receipt_payload.get("plan_binding")
        if plan_binding is not None and (
            not isinstance(plan_binding, Mapping)
            or dict(plan_binding) != expected_plan_binding
        ):
            blockers.append(
                _blocker(
                    "receipt_plan_binding_invalid",
                    receipt_id,
                    "retrieval plan binding disagrees with immutable search-plan "
                    "lineage",
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
        *sorted(active_challenge_ids),
        *sorted(superseded_challenge_ids),
        *sorted(challenge_resolution_ids),
        *source_update_ids,
    ]
    for object_id in dict.fromkeys(semantic_ids):
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
    all_challenge_ids = {*active_challenge_ids, *superseded_challenge_ids}
    claim_evidence_reasoning_ids = {
        *scientific_closure_target_ids,
        *evidence_ids,
        *argument_ids,
    } - claim_alias_ids
    challenged_closure_ids = {
        target
        for object_id in all_challenge_ids
        for target in _targets(
            objects[object_id], relation_types=_CLAIM_CHALLENGE_RELATIONS
        )
        if target in claim_evidence_reasoning_ids
    }
    active_review_closure_ids = {
        *evidence_ids,
        *argument_ids,
        *all_challenge_ids,
        *challenge_resolution_ids,
        *challenged_closure_ids,
    }
    verification_blockers.extend(
        _blocker(
            "active_claim_challenge",
            object_id,
            "active contradictory evidence or reasoning must be explicitly superseded before claim verification",
        )
        for object_id in sorted(active_challenge_ids)
    )
    producer_ids = {
        str((objects[object_id].get("actor") or {}).get("actor_id") or "")
        for object_id in {
            claim_id,
            *evidence_ids,
            *attempt_ids,
            *argument_ids,
            *all_challenge_ids,
            *challenge_resolution_ids,
        }
        if str((objects[object_id].get("actor") or {}).get("actor_id") or "")
    }
    passing_gates: list[str] = []
    for object_id in gate_ids:
        gate = objects[object_id]
        if not _is_active_closure_object(object_id, objects, superseded_ids):
            verification_blockers.append(
                _blocker(
                    "inactive_gate",
                    object_id,
                    "claim gate has been superseded and cannot authorize verification",
                )
            )
            continue
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
        complete_review_found = False
        for review_id in reviews:
            review = objects[review_id]
            if not _is_active_closure_object(review_id, objects, superseded_ids):
                verification_blockers.append(
                    _blocker(
                        "inactive_review",
                        review_id,
                        "gate review has been superseded and cannot authorize verification",
                    )
                )
                continue
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
                independence_targets = set(independence.get("target_ids") or [])
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
                    and bool(review_targets)
                    and review_targets.issubset(independence_targets)
                )
            review_covers_closure = bool(active_review_closure_ids) and (
                active_review_closure_ids.issubset(review_targets)
            )
            complete_review_found = complete_review_found or review_covers_closure
            trusted_review = trusted_review or (
                review.get("state") == "verified"
                and review_actor.get("authority") == "independent_evaluator"
                and str(review_actor.get("actor_id") or "") not in producer_ids
                and review_covers_closure
                and review_context_ok
                and independence_ok
            )
        gate_covers_closure = bool(active_review_closure_ids) and (
            active_review_closure_ids.issubset(evaluated)
        )
        if gate.get("state") == "verified" and not gate_covers_closure:
            missing_ids = sorted(active_review_closure_ids - evaluated)
            verification_blockers.append(
                _blocker(
                    "incomplete_gate_closure",
                    object_id,
                    "gate does not evaluate the complete active claim closure: "
                    + ", ".join(missing_ids),
                )
            )
        if gate.get("state") == "verified" and reviews and not complete_review_found:
            verification_blockers.append(
                _blocker(
                    "incomplete_review_closure",
                    object_id,
                    "no independent review evaluates every active evidence, reasoning, and challenge object",
                )
            )
        if (
            gate.get("state") == "verified"
            and (gate.get("payload") or {}).get("claim_promotion_allowed") is True
            and (gate.get("actor") or {}).get("authority") == "deterministic_gate"
            and gate_covers_closure
            and trusted_review
            and gate_context_ok
        ):
            passing_gates.append(object_id)
        elif gate.get("state") == "verified":
            verification_blockers.append(
                _blocker(
                    "unbound_or_untrusted_gate",
                    object_id,
                    "passing gate must bind one independent verified review of the complete active claim closure",
                )
            )
    for object_id in sorted(superseded_challenge_ids):
        resolution_ids = sorted(active_superseders_by_target.get(object_id, set()))
        if not resolution_ids:
            verification_blockers.append(
                _blocker(
                    "invalid_challenge_resolution",
                    object_id,
                    "superseded contradictory evidence has no active immutable resolution object",
                )
            )
        elif not passing_gates:
            verification_blockers.append(
                _blocker(
                    "unreviewed_challenge_resolution",
                    object_id,
                    "a fresh independent review and gate must cover the challenge and its resolution: "
                    + ", ".join(resolution_ids),
                )
            )
    reproduction_closure_ids = sorted(
        {
            *scientific_closure_target_ids,
            *evidence_ids,
            *argument_ids,
            *attempt_ids,
            *plan_ids,
            *preregistration_ids,
            *source_ids,
            *search_receipt_ids,
            *search_plan_ids,
            *source_update_ids,
            *gate_ids,
            *all_challenge_ids,
            *challenge_resolution_ids,
        }
        - {claim_id}
    )
    expected_claim_binding = _claim_reproduction_binding(
        claim_id, reproduction_closure_ids, objects
    )
    verified_reproductions: list[str] = []
    for object_id in reproduction_ids:
        reproduction = objects[object_id]
        if reproduction.get("state") != "verified":
            continue
        if not _is_active_closure_object(object_id, objects, superseded_ids):
            verification_blockers.append(
                _blocker(
                    "inactive_reproduction",
                    object_id,
                    "verified reproduction has been superseded and cannot satisfy closure",
                )
            )
            continue
        receipt = (reproduction.get("payload") or {}).get("receipt")
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema_version") != REPRODUCTION_RECEIPT_V2
        ):
            warnings.append(
                _blocker(
                    "legacy_reproduction_receipt_unbound",
                    object_id,
                    "legacy reproduction receipt is retained for history but cannot satisfy verification",
                )
            )
        reproduction_target_ids = set(
            _targets(reproduction, relation_types=("reproduces",))
        )
        receipt_issues = _receipt_integrity_issues(
            reproduction,
            repo=repo,
            objects=objects,
            reproduction_target_ids=reproduction_target_ids,
            expected_claim_binding=expected_claim_binding,
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
        "reproduction_closure_ids": reproduction_closure_ids,
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
    # Compute all three levels from the same immutable object set.  Historically
    # an audit only reported blockers for its requested level, which forced
    # reviewers to run the command three times and made a trace-complete report
    # look deceptively close to verification-complete.  Keeping the per-level
    # rows in one payload makes the scientific ladder inspectable at a glance.
    level_rows: dict[str, list[dict[str, Any]]] = {}
    level_blockers: dict[str, list[dict[str, str]]] = {}
    level_warnings: dict[str, list[dict[str, str]]] = {}
    for closure_level in RESEARCH_CLOSURE_LEVELS:
        rows: list[dict[str, Any]] = []
        level_specific_blockers: list[dict[str, str]] = []
        level_specific_warnings: list[dict[str, str]] = []
        if not claims:
            level_specific_blockers.append(
                _blocker(
                    "no_claims", "", "selected ref contains no typed claim objects"
                )
            )
        for claim in claims:
            row, row_blockers, row_warnings = _claim_closure(
                claim, objects, repo=repo, target_level=closure_level
            )
            rows.append(row)
            level_specific_blockers.extend(row_blockers)
            level_specific_warnings.extend(row_warnings)
        for error in integrity["errors"]:
            level_specific_blockers.append(
                _blocker("repository_integrity_failure", "", str(error))
            )
        for warning in integrity["warnings"]:
            level_specific_warnings.append(
                _blocker("repository_integrity_warning", "", str(warning))
            )
        level_blockers[closure_level] = sorted(
            {content_hash(item): item for item in level_specific_blockers}.values(),
            key=lambda item: (item["code"], item["object_id"], item["message"]),
        )
        level_warnings[closure_level] = sorted(
            {content_hash(item): item for item in level_specific_warnings}.values(),
            key=lambda item: (item["code"], item["object_id"], item["message"]),
        )
        level_rows[closure_level] = rows

    claim_rows = level_rows[level]
    blockers = level_blockers[level]
    warnings = level_warnings[level]
    closure_levels = {
        closure_level: {
            "complete": not level_blockers[closure_level],
            "status": "complete" if not level_blockers[closure_level] else "blocked",
            "claim_count": len(level_rows[closure_level]),
            "complete_claim_count": sum(
                bool(row.get("complete")) for row in level_rows[closure_level]
            ),
            "blocker_count": len(level_blockers[closure_level]),
            "warning_count": len(level_warnings[closure_level]),
            "blocker_codes": sorted(
                {str(item.get("code") or "") for item in level_blockers[closure_level]}
                - {""}
            ),
            "warning_codes": sorted(
                {str(item.get("code") or "") for item in level_warnings[closure_level]}
                - {""}
            ),
        }
        for closure_level in RESEARCH_CLOSURE_LEVELS
    }
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
        "closure_levels": closure_levels,
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


def build_reproduction_target_binding(
    repo: str | Path,
    target_ids: Iterable[str],
    *,
    ref: str = "HEAD",
) -> dict[str, Any]:
    """Bind exact committed targets and current claim closures for receipt v2."""

    checkpoint_record = show_checkpoint(repo, ref)
    resolved = str(checkpoint_record["commit"])
    checkpoint = checkpoint_record["checkpoint"]
    objects = {
        str(item["object_id"]): item
        for item in list_research_objects_at_ref(repo, resolved)
    }
    normalized_targets = sorted(set(str(value) for value in target_ids))
    missing = sorted(set(normalized_targets) - set(objects))
    if missing:
        raise ResearchGitError(
            "reproduction targets are not committed at the bound audit ref: "
            + ", ".join(missing)
        )
    report = audit_research_closure(repo, ref=resolved, level="trace")
    rows = {
        str(item.get("claim_id") or ""): item for item in report.get("claims") or []
    }
    claim_closures: list[dict[str, Any]] = []
    for claim_id in sorted(
        object_id
        for object_id in normalized_targets
        if objects[object_id].get("kind") == "claim"
    ):
        row = rows.get(claim_id)
        if row is None:
            raise ResearchGitError(
                "reproduction claim target is not on the active audited frontier: "
                + claim_id
            )
        claim_closures.append(
            _claim_reproduction_binding(
                claim_id,
                row.get("reproduction_closure_ids") or [],
                objects,
            )
        )
    core = {
        "policy": REPRODUCTION_TARGET_POLICY_V2,
        "audit_commit": resolved,
        "audit_checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        "audit_checkpoint_hash": str(checkpoint.get("content_hash") or ""),
        "target_objects": _object_binding_rows(normalized_targets, objects),
        "claim_closures": claim_closures,
    }
    return {**core, "binding_hash": canonical_content_hash(core)}


def summarize_closure_levels(report: Mapping[str, Any]) -> dict[str, bool]:
    """Derive the monotonic trace → replay → verify ladder from one audit.

    Keeping this calculation beside the closure model prevents status, audit,
    and future clients from assigning different meanings to the same gates.
    """

    claims = list(report.get("claims") or [])
    integrity_ok = not (report.get("integrity") or {}).get("errors")
    trace_complete = (
        bool(claims)
        and integrity_ok
        and all(bool(item.get("trace_complete")) for item in claims)
    )
    replay_complete = trace_complete and all(
        bool(item.get("replay_ready")) for item in claims
    )
    verification_complete = replay_complete and all(
        bool(item.get("verified")) for item in claims
    )
    return {
        "trace": trace_complete,
        "replay": replay_complete,
        "verify": verification_complete,
    }


def closure_level_summary(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the machine-readable status of every closure level.

    New audits include exact blocker/warning counts and code lists for all
    levels.  The small fallback keeps callers compatible with reports created
    by older XScientist versions; in that case only the requested level's
    issue counts are known and the other levels are marked as derived.
    """

    raw = report.get("closure_levels")
    if isinstance(raw, Mapping):
        result: dict[str, dict[str, Any]] = {}
        for level in RESEARCH_CLOSURE_LEVELS:
            value = raw.get(level)
            if isinstance(value, Mapping):
                result[level] = dict(value)
        if len(result) == len(RESEARCH_CLOSURE_LEVELS):
            return result

    levels = summarize_closure_levels(report)
    target = str(report.get("target_level") or "")
    blockers = list(report.get("blockers") or [])
    warnings = list(report.get("warnings") or [])
    blocker_codes = sorted({str(item.get("code") or "") for item in blockers} - {""})
    warning_codes = sorted({str(item.get("code") or "") for item in warnings} - {""})
    result = {}
    for level in RESEARCH_CLOSURE_LEVELS:
        known = level == target
        result[level] = {
            "complete": bool(levels[level]),
            "status": "complete" if levels[level] else "blocked",
            "claim_count": len(list(report.get("claims") or [])),
            "complete_claim_count": sum(
                bool(item.get("complete")) for item in list(report.get("claims") or [])
            ),
            "blocker_count": len(blockers) if known else None,
            "warning_count": len(warnings) if known else None,
            "blocker_codes": blocker_codes if known else [],
            "warning_codes": warning_codes if known else [],
            "counts_scope": "target_level" if known else "derived",
        }
    return result


__all__ = [
    "RESEARCH_CLOSURE_LEVELS",
    "RESEARCH_CLOSURE_SCHEMA",
    "audit_research_closure",
    "build_reproduction_target_binding",
    "closure_level_summary",
    "summarize_closure_levels",
]
