"""Externally authorized experiment-admission decisions.

Preparing a decision is intentionally non-authoritative.  Authority enters a
Research VCS repository only after an Ed25519 signature is verified against a
trust store held outside that repository.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ai_scientist.protocol.attestation import verify_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import schema_validator
from ai_scientist.utils.principal_identity import canonical_principal
from ai_scientist.utils.verifier_authority import load_ed25519_trust_store

from .research_git import ResearchGitError

EXPERIMENT_ADMISSION_PROTOCOL = "xscientist.experiment-admission.v1"
EXPERIMENT_ADMISSION_PAYLOAD_SCHEMA = "xscientist.experiment-admission-request.v1"
EXPERIMENT_ADMISSION_AUTHORITY_PROTOCOL = "xscientist.experiment-admission-authority.v1"
XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE = (
    "XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE"
)
EXPERIMENT_ADMISSION_ACTOR = {
    "actor_id": "xscientist-experiment-admission-verifier",
    "authority": "deterministic_gate",
}
EXPERIMENT_ADMISSION_SCOPE_FIELDS = frozenset(
    {
        "arm_ids",
        "allowed_seeds",
        "budget_hash",
        "candidate_id",
        "code_commit",
        "command_hash",
        "configuration_hash",
        "dataset_hashes",
        "environment_hash",
        "idea_index",
        "ideas_hash",
        "interventions",
        "producer_id",
        "resource_contract_hash",
        "seeds",
        "study_phase",
        "task_id",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "repository_id",
        "repository_head",
        "decision",
        "input_bindings",
        "scope",
        "scope_hash",
    }
)
_EXTERNAL_IDENTITY_PREFIXES = ("agent:", "service:", "human:")
_GATE_PAYLOAD_FIELDS = frozenset(
    {
        "protocol_kind",
        "gate_id",
        "decision",
        "experiment_authorized",
        "claim_promotion_allowed",
        "repository_id",
        "repository_head",
        "input_bindings",
        "scope",
        "scope_hash",
        "admission_request_hash",
        "authority_receipt",
        "gate_hash",
    }
)
_AUTHORITY_RECEIPT_FIELDS = frozenset(
    {
        "protocol_kind",
        "verifier_identity",
        "verifier_principal",
        "key_id",
        "algorithm",
        "attestation_hash",
        "trust_store_hash",
        "signed_payload_hash",
        "producer_actor_ids",
        "producer_principals",
        "lineage_object_ids",
        "signed_payload",
        "attestation_record",
        "receipt_hash",
    }
)
_ATTESTATION_RECORD_FIELDS = frozenset(
    {"schema_version", "signature_encoding", "envelope", "signature_hex"}
)
_ATTESTATION_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "canonicalization",
        "purpose",
        "identity",
        "key_id",
        "algorithm",
        "issued_at",
        "payload_hash",
        "attestation_hash",
    }
)
_ATTESTATION_RECORD_SCHEMA = "xscientist.experiment-admission-attestation-record.v1"
_RESEARCH_OBJECT_ID_RE = re.compile(r"rso-[0-9a-f]{16}\Z")
_ED25519_SIGNATURE_HEX_RE = re.compile(r"[0-9a-f]{128}\Z")
_SHA256_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SCOPE_TEXT_FIELDS = frozenset(
    {"candidate_id", "code_commit", "producer_id", "study_phase", "task_id"}
)
_SCOPE_HASH_FIELDS = frozenset(
    {
        "budget_hash",
        "command_hash",
        "configuration_hash",
        "environment_hash",
        "ideas_hash",
        "resource_contract_hash",
    }
)
_SCOPE_HASH_LIST_FIELDS = frozenset({"dataset_hashes"})
_SCOPE_TEXT_LIST_FIELDS = frozenset({"interventions"})
_ARM_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _binding(research_object: Mapping[str, Any]) -> dict[str, str]:
    return {
        "object_id": str(research_object.get("object_id") or ""),
        "content_hash": str(research_object.get("content_hash") or ""),
    }


def _validate_scope_constraints(scope: Mapping[str, Any]) -> None:
    if not isinstance(scope, Mapping):
        raise ResearchGitError("experiment admission execution scope must be an object")
    if any(type(field) is not str for field in scope):
        raise ResearchGitError("experiment admission execution scope fields must be strings")
    unsupported = sorted(set(scope) - EXPERIMENT_ADMISSION_SCOPE_FIELDS)
    if unsupported:
        raise ResearchGitError(
            "experiment admission execution scope contains unsupported fields: "
            + ", ".join(unsupported)
        )
    for field in _SCOPE_HASH_FIELDS:
        if field in scope and (
            type(scope[field]) is not str or _SHA256_HASH_RE.fullmatch(scope[field]) is None
        ):
            raise ResearchGitError(
                f"experiment admission {field} must be a sha256:<64 lowercase hex> hash"
            )
    for field in _SCOPE_TEXT_FIELDS:
        if field in scope and (
            type(scope[field]) is not str
            or not scope[field]
            or scope[field] != scope[field].strip()
            or len(scope[field]) > 512
        ):
            raise ResearchGitError(
                f"experiment admission {field} must be a non-empty string"
            )
    for field in _SCOPE_HASH_LIST_FIELDS:
        if field not in scope:
            continue
        values = scope[field]
        if (
            type(values) is not list
            or any(
                type(value) is not str or _SHA256_HASH_RE.fullmatch(value) is None
                for value in values
            )
            or values != sorted(set(values))
        ):
            raise ResearchGitError(
                f"experiment admission {field} must be a sorted unique sha256 hash array"
            )
    for field in _SCOPE_TEXT_LIST_FIELDS:
        if field not in scope:
            continue
        values = scope[field]
        if (
            type(values) is not list
            or not values
            or any(
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value) > 512
                for value in values
            )
        ):
            raise ResearchGitError(
                f"experiment admission {field} must be a non-empty string array"
            )
    contract_fields = {"arm_ids", "resource_contract_hash"}.intersection(scope)
    if contract_fields and contract_fields != {"arm_ids", "resource_contract_hash"}:
        raise ResearchGitError(
            "experiment admission arm_ids and resource_contract_hash must be supplied together"
        )
    if "arm_ids" in scope:
        arm_ids = scope["arm_ids"]
        if (
            type(arm_ids) is not list
            or not arm_ids
            or any(type(arm_id) is not str or _ARM_ID_RE.fullmatch(arm_id) is None for arm_id in arm_ids)
            or arm_ids != sorted(set(arm_ids))
        ):
            raise ResearchGitError(
                "experiment admission arm_ids must be sorted, unique, non-empty identifiers"
            )
    for field in ("allowed_seeds", "seeds"):
        if field not in scope:
            continue
        seeds = scope.get(field)
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
            )
            or len(set(seeds)) != len(seeds)
        ):
            raise ResearchGitError(
                f"experiment admission {field} must be a non-empty unique integer array"
            )
    if "allowed_seeds" in scope and "seeds" in scope:
        raise ResearchGitError(
            "experiment admission allowed_seeds cannot be combined with seeds"
        )
    if "idea_index" in scope and (
        type(scope.get("idea_index")) is not int or scope["idea_index"] < 0
    ):
        raise ResearchGitError(
            "experiment admission idea_index must be a non-negative integer"
        )


def validate_experiment_admission_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a runtime scope without adding fields or upgrading old scopes."""

    if not isinstance(scope, Mapping):
        raise ResearchGitError("experiment admission execution scope must be an object")
    row = deepcopy(dict(scope))
    _validate_scope_constraints(row)
    try:
        canonical_content_hash(row)
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(
            "experiment admission execution scope is not canonical JSON"
        ) from exc
    return row


def build_experiment_admission_payload(
    *,
    repository_id: str,
    repository_head: str,
    decision: str,
    hypothesis: Mapping[str, Any],
    plan: Mapping[str, Any],
    preregistration: Mapping[str, Any] | None,
    execution_scope: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the exact, non-authoritative payload an external verifier signs."""

    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision not in {"authorize", "deny"}:
        raise ResearchGitError(
            "experiment admission decision must be authorize or deny"
        )
    scope = deepcopy(dict(execution_scope or {}))
    validate_experiment_admission_scope(scope)
    try:
        scope_hash = canonical_content_hash(scope)
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(
            "experiment admission execution scope is not canonical JSON"
        ) from exc
    payload = {
        "schema_version": EXPERIMENT_ADMISSION_PAYLOAD_SCHEMA,
        "purpose": EXPERIMENT_ADMISSION_PROTOCOL,
        "repository_id": str(repository_id or "").strip(),
        "repository_head": str(repository_head or "").strip(),
        "decision": normalized_decision,
        "input_bindings": {
            "hypothesis": _binding(hypothesis),
            "plan": _binding(plan),
            "preregistration": (
                _binding(preregistration) if preregistration is not None else None
            ),
        },
        "scope": scope,
        "scope_hash": scope_hash,
    }
    return normalize_experiment_admission_payload(payload)


def normalize_experiment_admission_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and defensively copy one externally supplied signed payload."""

    if not isinstance(payload, Mapping):
        raise ResearchGitError("experiment admission payload must be a JSON object")
    row = deepcopy(dict(payload))
    if set(row) != _PAYLOAD_FIELDS:
        raise ResearchGitError("experiment admission payload fields are invalid")
    try:
        schema_valid = schema_validator("experiment_admission").is_valid(row)
    except (TypeError, ValueError):
        schema_valid = False
    if not schema_valid:
        raise ResearchGitError("experiment admission payload schema is invalid")
    scope = row.get("scope")
    if not isinstance(scope, Mapping):  # pragma: no cover - schema defense
        raise ResearchGitError("experiment admission execution scope is invalid")
    validate_experiment_admission_scope(scope)
    try:
        expected_scope_hash = canonical_content_hash(dict(scope))
        canonical_content_hash(row)
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(
            "experiment admission payload is not canonical JSON"
        ) from exc
    if row.get("scope_hash") != expected_scope_hash:
        raise ResearchGitError("experiment admission scope hash does not match")
    return row


def verify_experiment_admission_attestation(
    payload: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    trust_store_path: str | Path,
    forbidden_roots: tuple[str | Path, ...],
) -> dict[str, Any]:
    """Verify an external Ed25519 verdict and return public authority metadata."""

    if not isinstance(attestation, Mapping):
        raise ResearchGitError("experiment admission attestation must be a JSON object")
    envelope = deepcopy(dict(attestation))
    if envelope.get("algorithm") != "ed25519":
        raise ResearchGitError(
            "experiment admission authority requires an Ed25519 attestation"
        )
    identity = str(envelope.get("identity") or "").strip()
    if not identity.startswith(_EXTERNAL_IDENTITY_PREFIXES):
        raise ResearchGitError(
            "experiment admission verifier identity must use agent:, service:, or human:"
        )
    try:
        verifier_principal = canonical_principal(
            identity, label="experiment admission verifier identity"
        )
    except ValueError as exc:
        raise ResearchGitError(
            "experiment admission verifier identity is invalid"
        ) from exc
    if not str(trust_store_path or "").strip():
        raise ResearchGitError(
            "experiment admission requires an explicit external trust store"
        )
    try:
        trust_store = load_ed25519_trust_store(
            trust_store_path,
            forbidden_roots=forbidden_roots,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ResearchGitError(
            str(exc) or "experiment admission trust store is invalid"
        ) from exc
    verification = verify_attestation(
        envelope,
        payload,
        trust_store=trust_store,
        purpose=EXPERIMENT_ADMISSION_PROTOCOL,
        identity=identity,
    )
    if verification.get("ok") is not True:
        errors = ", ".join(str(item) for item in verification.get("errors") or [])
        raise ResearchGitError(
            "experiment admission attestation verification failed"
            + (f": {errors}" if errors else "")
        )
    trust_projection = {
        key_id: {
            "identity": entry.get("identity"),
            "algorithm": entry.get("algorithm"),
            "public_key": base64.b64encode(entry["public_key"]).decode("ascii"),
            "revoked": entry.get("revoked") is True,
        }
        for key_id, entry in sorted(trust_store.items())
    }
    raw_signature = base64.b64decode(str(envelope["signature"]), validate=True)
    attestation_record = {
        "schema_version": "xscientist.experiment-admission-attestation-record.v1",
        "signature_encoding": "hex",
        "envelope": {
            key: deepcopy(value)
            for key, value in envelope.items()
            if key != "signature"
        },
        "signature_hex": raw_signature.hex(),
    }
    return {
        "verifier_identity": identity,
        "verifier_principal": verifier_principal,
        "key_id": str(envelope.get("key_id") or ""),
        "algorithm": "ed25519",
        "attestation_hash": str(verification.get("attestation_hash") or ""),
        "trust_store_hash": canonical_content_hash(trust_projection),
        "attestation_record": attestation_record,
    }


def _exact_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchGitError(f"{label} must be a JSON object")
    try:
        row = deepcopy(dict(value))
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(f"{label} is invalid") from exc
    if set(row) != fields:
        raise ResearchGitError(f"{label} fields are invalid")
    return row


def _sorted_identity_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ResearchGitError(f"{label} must be a non-empty JSON array")
    if any(
        not isinstance(item, str) or not item or len(item) > 512 or item != item.strip()
        for item in value
    ):
        raise ResearchGitError(f"{label} contains an invalid identity")
    if value != sorted(set(value)):
        raise ResearchGitError(f"{label} must be sorted and unique")
    try:
        for item in value:
            canonical_principal(item, label=label)
    except ValueError as exc:
        raise ResearchGitError(f"{label} contains an invalid identity") from exc
    return list(value)


def _sorted_principal_list(value: Any, *, label: str) -> list[str]:
    principals = _sorted_identity_list(value, label=label)
    try:
        if any(canonical_principal(item, label=label) != item for item in principals):
            raise ResearchGitError(f"{label} contains a non-canonical principal")
    except ValueError as exc:  # pragma: no cover - identity list validates first
        raise ResearchGitError(f"{label} contains an invalid principal") from exc
    return principals


def _sorted_lineage_list(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or _RESEARCH_OBJECT_ID_RE.fullmatch(item) is None
            for item in value
        )
        or value != sorted(set(value))
    ):
        raise ResearchGitError(
            f"{label} must be a non-empty sorted unique research-object ID array"
        )
    return list(value)


def _iterable_values(value: Iterable[str], *, label: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ResearchGitError(f"{label} must be an iterable of strings")
    try:
        rows = list(value)
    except TypeError as exc:
        raise ResearchGitError(f"{label} must be an iterable of strings") from exc
    if any(not isinstance(item, str) for item in rows):
        raise ResearchGitError(f"{label} must contain only strings")
    return rows


def _validated_forbidden_roots(
    values: tuple[str | Path, ...],
) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes, bytearray, Path)):
        raise ResearchGitError(
            "experiment admission forbidden roots must be a non-empty iterable"
        )
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ResearchGitError(
            "experiment admission forbidden roots must be a non-empty iterable"
        ) from exc
    if not rows:
        raise ResearchGitError(
            "experiment admission forbidden roots must be a non-empty iterable"
        )
    roots: list[Path] = []
    for value in rows:
        try:
            root = Path(value).expanduser().resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise ResearchGitError(
                "experiment admission forbidden root is invalid"
            ) from exc
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def verify_persisted_experiment_admission_gate(
    gate_payload: Mapping[str, Any],
    *,
    trust_store_path: str | Path,
    forbidden_roots: tuple[str | Path, ...],
    expected_producer_actor_ids: Iterable[str] | None = None,
    expected_producer_principals: Iterable[str] | None = None,
    expected_lineage_object_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Re-verify a persisted admission gate against current external policy.

    The signed payload and Base64 Ed25519 envelope are reconstructed solely
    from the persisted gate fields. Optional expected producer-closure values
    let a consumer bind the receipt to a freshly traversed repository lineage.
    Any malformed, stale, or internally inconsistent value raises
    :class:`ResearchGitError`.
    """

    gate = _exact_mapping(
        gate_payload,
        fields=_GATE_PAYLOAD_FIELDS,
        label="persisted experiment admission gate payload",
    )
    if gate.get("protocol_kind") != EXPERIMENT_ADMISSION_PROTOCOL:
        raise ResearchGitError("persisted experiment admission protocol is invalid")
    decision = gate.get("decision")
    if decision not in {"authorize", "deny"}:
        raise ResearchGitError("persisted experiment admission decision is invalid")
    if not isinstance(gate.get("experiment_authorized"), bool) or gate[
        "experiment_authorized"
    ] is not (decision == "authorize"):
        raise ResearchGitError(
            "persisted experiment admission authorization does not match decision"
        )
    if gate.get("claim_promotion_allowed") is not False:
        raise ResearchGitError(
            "persisted experiment admission cannot allow claim promotion"
        )

    reconstructed_payload = normalize_experiment_admission_payload(
        {
            "schema_version": EXPERIMENT_ADMISSION_PAYLOAD_SCHEMA,
            "purpose": EXPERIMENT_ADMISSION_PROTOCOL,
            "repository_id": deepcopy(gate.get("repository_id")),
            "repository_head": deepcopy(gate.get("repository_head")),
            "decision": decision,
            "input_bindings": deepcopy(gate.get("input_bindings")),
            "scope": deepcopy(gate.get("scope")),
            "scope_hash": deepcopy(gate.get("scope_hash")),
        }
    )
    request_hash = canonical_content_hash(reconstructed_payload)
    if gate.get("admission_request_hash") != request_hash:
        raise ResearchGitError(
            "persisted experiment admission request hash does not match"
        )

    authority = _exact_mapping(
        gate.get("authority_receipt"),
        fields=_AUTHORITY_RECEIPT_FIELDS,
        label="persisted experiment admission authority receipt",
    )
    if authority.get("protocol_kind") != EXPERIMENT_ADMISSION_AUTHORITY_PROTOCOL:
        raise ResearchGitError(
            "persisted experiment admission authority protocol is invalid"
        )
    signed_payload = normalize_experiment_admission_payload(
        authority.get("signed_payload")
    )
    if (
        signed_payload != reconstructed_payload
        or canonical_content_hash(signed_payload) != request_hash
    ):
        raise ResearchGitError(
            "persisted experiment admission signed payload does not match gate"
        )
    if authority.get("signed_payload_hash") != request_hash:
        raise ResearchGitError(
            "persisted experiment admission signed payload hash does not match"
        )

    producer_actor_ids = _sorted_identity_list(
        authority.get("producer_actor_ids"),
        label="persisted experiment admission producer actor IDs",
    )
    producer_principals = _sorted_principal_list(
        authority.get("producer_principals"),
        label="persisted experiment admission producer principals",
    )
    lineage_object_ids = _sorted_lineage_list(
        authority.get("lineage_object_ids"),
        label="persisted experiment admission lineage object IDs",
    )
    actor_principals = sorted(
        {
            canonical_principal(
                identity,
                label="persisted experiment admission producer actor ID",
            )
            for identity in producer_actor_ids
        }
    )
    if actor_principals != producer_principals:
        raise ResearchGitError(
            "persisted experiment admission producer closure is inconsistent"
        )

    binding_ids = {
        str(binding["object_id"])
        for binding in reconstructed_payload["input_bindings"].values()
        if isinstance(binding, Mapping)
    }
    if not binding_ids.issubset(lineage_object_ids):
        raise ResearchGitError(
            "persisted experiment admission lineage does not cover signed inputs"
        )

    expected_closure_fields = (
        expected_producer_actor_ids,
        expected_producer_principals,
        expected_lineage_object_ids,
    )
    if any(value is not None for value in expected_closure_fields) and not all(
        value is not None for value in expected_closure_fields
    ):
        raise ResearchGitError(
            "expected experiment admission producer closure must be complete"
        )

    if expected_producer_actor_ids is not None:
        expected_actor_ids = sorted(
            set(
                _iterable_values(
                    expected_producer_actor_ids,
                    label="expected experiment admission producer actor IDs",
                )
            )
        )
        if producer_actor_ids != expected_actor_ids:
            raise ResearchGitError(
                "persisted experiment admission producer actor closure does not match"
            )
    if expected_producer_principals is not None:
        try:
            expected_principals = sorted(
                {
                    canonical_principal(
                        value,
                        label="expected experiment admission producer principal",
                    )
                    for value in _iterable_values(
                        expected_producer_principals,
                        label="expected experiment admission producer principals",
                    )
                }
            )
        except ValueError as exc:
            raise ResearchGitError(
                "expected experiment admission producer principals are invalid"
            ) from exc
        if producer_principals != expected_principals:
            raise ResearchGitError(
                "persisted experiment admission producer principal closure does not match"
            )
    if expected_lineage_object_ids is not None:
        expected_lineage = sorted(
            set(
                _iterable_values(
                    expected_lineage_object_ids,
                    label="expected experiment admission lineage object IDs",
                )
            )
        )
        if lineage_object_ids != expected_lineage:
            raise ResearchGitError(
                "persisted experiment admission lineage closure does not match"
            )

    record = _exact_mapping(
        authority.get("attestation_record"),
        fields=_ATTESTATION_RECORD_FIELDS,
        label="persisted experiment admission attestation record",
    )
    if (
        record.get("schema_version") != _ATTESTATION_RECORD_SCHEMA
        or record.get("signature_encoding") != "hex"
    ):
        raise ResearchGitError(
            "persisted experiment admission attestation record format is invalid"
        )
    envelope = _exact_mapping(
        record.get("envelope"),
        fields=_ATTESTATION_ENVELOPE_FIELDS,
        label="persisted experiment admission attestation envelope",
    )
    signature_hex = record.get("signature_hex")
    if (
        not isinstance(signature_hex, str)
        or _ED25519_SIGNATURE_HEX_RE.fullmatch(signature_hex) is None
    ):
        raise ResearchGitError(
            "persisted experiment admission Ed25519 signature encoding is invalid"
        )
    attestation = {
        **envelope,
        "signature": base64.b64encode(bytes.fromhex(signature_hex)).decode("ascii"),
    }
    roots = _validated_forbidden_roots(forbidden_roots)
    current_authority = verify_experiment_admission_attestation(
        reconstructed_payload,
        attestation,
        trust_store_path=trust_store_path,
        forbidden_roots=roots,
    )
    authority_bindings = {
        "verifier_identity": current_authority["verifier_identity"],
        "verifier_principal": current_authority["verifier_principal"],
        "key_id": current_authority["key_id"],
        "algorithm": current_authority["algorithm"],
        "attestation_hash": current_authority["attestation_hash"],
        "trust_store_hash": current_authority["trust_store_hash"],
    }
    if any(authority.get(key) != value for key, value in authority_bindings.items()):
        raise ResearchGitError(
            "persisted experiment admission authority binding does not match current policy"
        )
    if authority.get("attestation_record") != current_authority.get(
        "attestation_record"
    ):
        raise ResearchGitError(
            "persisted experiment admission attestation record does not match"
        )

    signer_principal = str(current_authority["verifier_principal"])
    if signer_principal in set(producer_principals):
        raise ResearchGitError(
            "persisted experiment admission signer is not independent of producers"
        )
    scope_producer = reconstructed_payload["scope"].get("producer_id")
    if scope_producer is not None:
        try:
            scope_producer_principal = canonical_principal(
                scope_producer,
                label="persisted experiment admission scope producer_id",
            )
        except ValueError as exc:
            raise ResearchGitError(
                "persisted experiment admission scope producer_id is invalid"
            ) from exc
        if signer_principal == scope_producer_principal:
            raise ResearchGitError(
                "persisted experiment admission signer is not independent of scope producer"
            )

    authority_core = {
        key: deepcopy(value)
        for key, value in authority.items()
        if key != "receipt_hash"
    }
    if authority.get("receipt_hash") != canonical_content_hash(authority_core):
        raise ResearchGitError(
            "persisted experiment admission authority receipt hash does not match"
        )
    gate_core = {
        key: deepcopy(value) for key, value in gate.items() if key != "gate_hash"
    }
    if gate.get("gate_hash") != canonical_content_hash(gate_core):
        raise ResearchGitError(
            "persisted experiment admission gate hash does not match"
        )
    expected_gate_id = (
        "admission-"
        + request_hash.removeprefix("sha256:")[:16]
        + "-"
        + str(current_authority["attestation_hash"]).removeprefix("sha256:")[:16]
    )
    if gate.get("gate_id") != expected_gate_id:
        raise ResearchGitError("persisted experiment admission gate ID does not match")

    return {
        "gate_id": expected_gate_id,
        "decision": decision,
        "experiment_authorized": decision == "authorize",
        "repository_id": reconstructed_payload["repository_id"],
        "repository_head": reconstructed_payload["repository_head"],
        "signed_payload_hash": request_hash,
        "verifier_identity": current_authority["verifier_identity"],
        "verifier_principal": signer_principal,
        "producer_actor_ids": producer_actor_ids,
        "producer_principals": producer_principals,
        "lineage_object_ids": lineage_object_ids,
    }


def prepare_experiment_admission_payload(
    repository: Any,
    *,
    plan_id: str,
    preregistration_id: str | None = None,
    execution_scope: Mapping[str, Any] | None = None,
    decision: str,
    expected_head: str | None = None,
) -> dict[str, Any]:
    """Public convenience API for preparing a non-authoritative verdict."""

    from .research_lifecycle import ResearchLifecycle

    return ResearchLifecycle(repository).prepare_experiment_admission_payload(
        plan_id=plan_id,
        preregistration_id=preregistration_id,
        execution_scope=execution_scope,
        decision=decision,
        expected_head=expected_head,
    )


__all__ = [
    "EXPERIMENT_ADMISSION_ACTOR",
    "EXPERIMENT_ADMISSION_AUTHORITY_PROTOCOL",
    "EXPERIMENT_ADMISSION_PAYLOAD_SCHEMA",
    "EXPERIMENT_ADMISSION_PROTOCOL",
    "EXPERIMENT_ADMISSION_SCOPE_FIELDS",
    "XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE",
    "build_experiment_admission_payload",
    "normalize_experiment_admission_payload",
    "prepare_experiment_admission_payload",
    "validate_experiment_admission_scope",
    "verify_experiment_admission_attestation",
    "verify_persisted_experiment_admission_gate",
]
