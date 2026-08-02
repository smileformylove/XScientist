from __future__ import annotations

"""Budgeted, diversity-aware stopping control for one self-evolution epoch."""

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    load_jsonl_artifact,
    save_contract_artifact,
    update_pipeline_artifact,
)

SCHEMA_VERSION = 1
TRIAL_STATUSES = {"proposed", "running", "invalid", "held", "gate_eligible"}
TERMINAL_TRIAL_STATUSES = {"invalid", "held", "gate_eligible"}
EPOCH_HALTING_FAILURE_TOKENS = {
    "evaluation_policy",
    "integrity",
    "raw_evidence",
    "safety",
    "science_constitution",
}
TRIAL_FIELDS = {
    "schema_version",
    "program_id",
    "epoch_id",
    "trial_id",
    "intent_id",
    "candidate_hash",
    "mechanism_hash",
    "status",
    "utility_delta",
    "cost_units",
    "hard_gate_failures",
    "recorded_at",
    "trial_hash",
}


class EvolutionControllerError(ValueError):
    """Raised when trial history cannot safely control an evolution epoch."""


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


def _is_content_hash(value: Any) -> bool:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        return False
    try:
        int(text.split(":", 1)[1], 16)
    except ValueError:
        return False
    return True


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _trial_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value) for key, value in payload.items() if key != "trial_hash"
    }


def build_evolution_trial(
    program: dict[str, Any],
    *,
    trial_id: str,
    intent_id: str,
    candidate_hash: str,
    mechanism_hash: str,
    status: str,
    utility_delta: float | None = None,
    cost_units: float = 0.0,
    hard_gate_failures: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create one immutable trial record bound to a program epoch and intent."""

    epoch = program.get("epoch") if isinstance(program.get("epoch"), dict) else {}
    intent_ids = {
        str(item.get("intent_id") or "")
        for item in program.get("intents") or []
        if isinstance(item, dict)
    }
    required = {
        "program_id": program.get("program_id"),
        "epoch_id": epoch.get("epoch_id"),
        "trial_id": trial_id,
        "intent_id": intent_id,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        raise EvolutionControllerError("missing trial fields: " + ", ".join(missing))
    if intent_id not in intent_ids:
        raise EvolutionControllerError("trial intent is not active in this epoch")
    if not _is_content_hash(candidate_hash) or not _is_content_hash(mechanism_hash):
        raise EvolutionControllerError("candidate and mechanism require SHA-256 hashes")
    trial_status = str(status or "").strip()
    if trial_status not in TRIAL_STATUSES:
        raise EvolutionControllerError(f"unsupported trial status: {trial_status!r}")
    utility = _finite_number(utility_delta)
    if utility_delta is not None and utility is None:
        raise EvolutionControllerError("utility_delta must be finite or null")
    cost = _finite_number(cost_units)
    if cost is None or cost < 0:
        raise EvolutionControllerError("cost_units must be finite and non-negative")
    if trial_status in TERMINAL_TRIAL_STATUSES and cost <= 0:
        raise EvolutionControllerError("terminal trials require positive cost_units")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "program_id": str(program["program_id"]),
        "epoch_id": str(epoch["epoch_id"]),
        "trial_id": str(trial_id).strip(),
        "intent_id": str(intent_id).strip(),
        "candidate_hash": str(candidate_hash),
        "mechanism_hash": str(mechanism_hash),
        "status": trial_status,
        "utility_delta": utility,
        "cost_units": cost,
        "hard_gate_failures": sorted(
            {
                str(item).strip()
                for item in (hard_gate_failures or [])
                if str(item).strip()
            }
        ),
        "recorded_at": _now_iso(),
    }
    payload["trial_hash"] = _canonical_hash(_trial_core(payload))
    return payload


def validate_evolution_trial(
    payload: dict[str, Any] | None,
    *,
    program: dict[str, Any],
) -> dict[str, Any]:
    trial = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if set(trial) != TRIAL_FIELDS:
        errors.append("trial_fields_invalid")
    if trial.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    epoch = program.get("epoch") if isinstance(program.get("epoch"), dict) else {}
    if trial.get("program_id") != program.get("program_id"):
        errors.append("program_binding_mismatch")
    if trial.get("epoch_id") != epoch.get("epoch_id"):
        errors.append("epoch_binding_mismatch")
    intent_ids = {
        str(item.get("intent_id") or "")
        for item in program.get("intents") or []
        if isinstance(item, dict)
    }
    if trial.get("intent_id") not in intent_ids:
        errors.append("intent_binding_mismatch")
    if not str(trial.get("trial_id") or "").strip():
        errors.append("trial_id_missing")
    if not _is_content_hash(trial.get("candidate_hash")):
        errors.append("candidate_hash_invalid")
    if not _is_content_hash(trial.get("mechanism_hash")):
        errors.append("mechanism_hash_invalid")
    if trial.get("status") not in TRIAL_STATUSES:
        errors.append("status_invalid")
    utility = _finite_number(trial.get("utility_delta"))
    if trial.get("utility_delta") is not None and utility is None:
        errors.append("utility_delta_invalid")
    cost = _finite_number(trial.get("cost_units"))
    if cost is None or cost < 0:
        errors.append("cost_units_invalid")
    elif trial.get("status") in TERMINAL_TRIAL_STATUSES and cost <= 0:
        errors.append("terminal_cost_missing")
    failures = trial.get("hard_gate_failures")
    normalized_failures = sorted(
        {str(item).strip() for item in (failures or []) if str(item).strip()}
    )
    if not isinstance(failures, list) or failures != normalized_failures:
        errors.append("hard_gate_failures_invalid")
    if trial.get("trial_hash") != _canonical_hash(_trial_core(trial)):
        errors.append("trial_hash_mismatch")
    return {"passed": not errors, "errors": errors}


def _consecutive_non_improving(
    trials: list[dict[str, Any]],
    *,
    minimum_improvement: float,
) -> int:
    count = 0
    for trial in reversed(trials):
        if trial.get("status") not in TERMINAL_TRIAL_STATUSES:
            continue
        utility = _finite_number(trial.get("utility_delta"))
        if trial.get("status") == "gate_eligible" or (
            utility is not None and utility >= minimum_improvement
        ):
            break
        count += 1
    return count


def assess_evolution_epoch(
    program: dict[str, Any],
    trial_history: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Decide whether an epoch should continue without changing production state."""

    policy = (
        program.get("program_policy")
        if isinstance(program.get("program_policy"), dict)
        else {}
    )
    maximum_per_intent = int(policy.get("maximum_trials_per_intent") or 0)
    maximum_epoch = int(policy.get("maximum_epoch_trials") or 0)
    futility_patience = int(policy.get("maximum_consecutive_non_improving_trials") or 0)
    minimum_improvement = float(policy.get("minimum_utility_improvement") or 0.0)
    if min(maximum_per_intent, maximum_epoch, futility_patience) <= 0:
        raise EvolutionControllerError("program is missing bounded execution policy")
    intents = [item for item in program.get("intents") or [] if isinstance(item, dict)]
    trials = [deepcopy(item) for item in trial_history if isinstance(item, dict)]
    validation_errors: list[str] = []
    for index, trial in enumerate(trials):
        check = validate_evolution_trial(trial, program=program)
        validation_errors.extend(f"trial:{index}:{item}" for item in check["errors"])
    trial_ids = [str(item.get("trial_id") or "") for item in trials]
    candidate_hashes = [str(item.get("candidate_hash") or "") for item in trials]
    if len(trial_ids) != len(set(trial_ids)):
        validation_errors.append("duplicate_trial_id")
    if len(candidate_hashes) != len(set(candidate_hashes)):
        validation_errors.append("duplicate_candidate_hash")

    hard_gate_failures = sorted(
        {
            str(failure)
            for trial in trials
            for failure in trial.get("hard_gate_failures") or []
        }
    )
    epoch_halting_failures = sorted(
        failure
        for failure in hard_gate_failures
        if any(token in failure.lower() for token in EPOCH_HALTING_FAILURE_TOKENS)
    )
    intent_states: list[dict[str, Any]] = []
    unique_mechanisms: set[str] = set()
    terminal_trial_count = 0
    eligible_count = 0
    total_cost = 0.0
    for intent in intents:
        intent_id = str(intent.get("intent_id") or "")
        rows = [item for item in trials if item.get("intent_id") == intent_id]
        terminal = [
            item for item in rows if item.get("status") in TERMINAL_TRIAL_STATUSES
        ]
        terminal_trial_count += len(terminal)
        total_cost += sum(float(item.get("cost_units") or 0.0) for item in terminal)
        mechanisms = [str(item.get("mechanism_hash") or "") for item in terminal]
        unique = set(mechanisms)
        unique_mechanisms.update(unique)
        duplicate_count = len(mechanisms) - len(unique)
        gate_eligible = any(item.get("status") == "gate_eligible" for item in terminal)
        if gate_eligible:
            eligible_count += 1
        no_progress = _consecutive_non_improving(
            terminal,
            minimum_improvement=minimum_improvement,
        )
        if gate_eligible:
            status = "gate_eligible"
        elif len(terminal) >= maximum_per_intent:
            status = "budget_exhausted"
        elif no_progress >= futility_patience:
            status = "stopped_for_futility"
        elif duplicate_count:
            status = "needs_new_mechanism"
        else:
            status = "active"
        intent_states.append(
            {
                "intent_id": intent_id,
                "search_mode": intent.get("search_mode"),
                "status": status,
                "terminal_trial_count": len(terminal),
                "unique_mechanism_count": len(unique),
                "duplicate_mechanism_count": duplicate_count,
                "consecutive_non_improving_trials": no_progress,
                "remaining_trial_budget": max(maximum_per_intent - len(terminal), 0),
            }
        )
    completed_states = {
        "gate_eligible",
        "budget_exhausted",
        "stopped_for_futility",
    }
    active = [item for item in intent_states if item["status"] not in completed_states]
    if validation_errors or epoch_halting_failures:
        epoch_status = "halted"
    elif terminal_trial_count >= maximum_epoch:
        epoch_status = "budget_exhausted"
    elif intent_states and not active:
        epoch_status = "complete"
    elif not intent_states:
        epoch_status = "complete"
    else:
        epoch_status = "continue"
    next_intents = sorted(
        active,
        key=lambda item: (
            item["terminal_trial_count"] > 0,
            item.get("search_mode") != "explore",
            item["terminal_trial_count"],
            item["intent_id"],
        ),
    )
    mechanism_diversity = (
        len(unique_mechanisms) / terminal_trial_count if terminal_trial_count else 0.0
    )
    discovery_efficiency = eligible_count / total_cost if total_cost else 0.0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "program_id": program.get("program_id"),
        "epoch_id": (program.get("epoch") or {}).get("epoch_id"),
        "status": epoch_status,
        "should_stop": epoch_status != "continue",
        "automatic_production_promotion_allowed": False,
        "policy": {
            "maximum_trials_per_intent": maximum_per_intent,
            "maximum_epoch_trials": maximum_epoch,
            "maximum_consecutive_non_improving_trials": futility_patience,
            "minimum_utility_improvement": minimum_improvement,
        },
        "trial_count": len(trials),
        "terminal_trial_count": terminal_trial_count,
        "total_cost_units": round(total_cost, 6),
        "gate_eligible_intent_count": eligible_count,
        "mechanism_diversity": round(mechanism_diversity, 6),
        "discovery_efficiency": round(discovery_efficiency, 6),
        "validation_errors": validation_errors,
        "hard_gate_failures": hard_gate_failures,
        "epoch_halting_failures": epoch_halting_failures,
        "intent_states": intent_states,
        "next_intent_ids": [item["intent_id"] for item in next_intents],
    }
    payload["control_hash"] = _canonical_hash(payload)
    return payload


def record_evolution_trial(
    project_root: str | Path,
    trial: dict[str, Any],
    *,
    program: dict[str, Any],
) -> str:
    check = validate_evolution_trial(trial, program=program)
    if not check["passed"]:
        raise EvolutionControllerError("trial invalid: " + ", ".join(check["errors"]))
    history_path = Path(project_root).expanduser().resolve() / "evolution_trials.jsonl"
    append_jsonl_artifact(history_path, trial)
    return str(history_path)


def evaluate_and_save_evolution_control(
    project_root: str | Path,
    *,
    program: dict[str, Any],
    producer: str,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    trials = load_jsonl_artifact(root / "evolution_trials.jsonl")
    control = assess_evolution_epoch(program, trials)
    save_contract_artifact(
        root,
        "evolution_control",
        control,
        producer=producer,
        depends_on=["evolution_program"],
        notes="Budgeted epoch progress, diversity, and stopping decision.",
    )
    if control["status"] == "halted":
        update_pipeline_artifact(
            root,
            "evolution_control",
            status="blocked",
            producer=producer,
            depends_on=["evolution_program"],
            recovery_hint="Resolve the recorded hard-gate failure before another epoch.",
            notes="Evolution epoch halted by a non-negotiable gate.",
        )
    return control


__all__ = [
    "EvolutionControllerError",
    "assess_evolution_epoch",
    "build_evolution_trial",
    "evaluate_and_save_evolution_control",
    "record_evolution_trial",
    "validate_evolution_trial",
]
