from __future__ import annotations

"""Organize lessons into a bounded, diverse, constitution-bound evolution program."""

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.evolution_gate import (
    COMPONENT_TYPES,
    DEFAULT_POLICY,
    MUTABLE_COMPONENT_SCOPES,
    PROTECTED_COMPONENT_TYPES,
)
from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    artifact_path,
    load_contract_artifact,
    load_jsonl_artifact,
    save_contract_artifact,
)
from ai_scientist.utils.science_constitution import (
    assert_science_constitution_intact,
)

SCHEMA_VERSION = 1
EVOLUTION_LEVELS = {
    "episodic": {
        "scope": "current run only",
        "persistent": False,
        "production_mutation_allowed": False,
    },
    "playbook": {
        "scope": "cross-project advisory memory",
        "persistent": True,
        "production_mutation_allowed": False,
    },
    "system": {
        "scope": "prompt, tool, scaffold, routing, search, or recovery behavior",
        "persistent": True,
        "production_mutation_allowed": False,
        "requires_evolution_gate": True,
    },
}
CORE_PROGRAM_POLICY = {
    "policy_id": "xscientist-controlled-evolution-program",
    "version": "1.0.0",
    "maximum_active_intents": 6,
    "maximum_intents_per_component": 2,
    "maximum_scopes_per_intent": 1,
    "maximum_trials_per_intent": 4,
    "maximum_epoch_trials": 24,
    "maximum_consecutive_non_improving_trials": 2,
    "minimum_utility_improvement": 0.005,
    "minimum_exploration_fraction": 0.34,
    "fixed_utility_within_epoch": True,
    "evaluator_changes_at_epoch_boundary_only": True,
    "automatic_evaluator_mutation_allowed": False,
    "automatic_production_mutation_allowed": False,
    "negative_results_are_archived": True,
    "one_causal_mechanism_per_intent": True,
    "required_candidate_gate": "evolution_gate.v2",
}
RISK_ROUTING = {
    "figure_traceability_gap": {
        "component_type": "tool",
        "target_metric": "reproducibility_rate",
        "operator": "add deterministic figure-to-evidence binding checks",
    },
    "claim_evidence_gap": {
        "component_type": "agent_scaffold",
        "target_metric": "false_discovery_rate",
        "operator": "improve claim-to-evidence planning and counterexample search",
    },
    "section_clarity_gap": {
        "component_type": "prompt",
        "target_metric": "objective_score",
        "operator": "tighten evidence-aware section rewrite instructions",
    },
    "evidence_depth_gap": {
        "component_type": "search_policy",
        "target_metric": "objective_score",
        "operator": "allocate search to controls, ablations, and robustness branches",
    },
    "reproducibility_gap": {
        "component_type": "tool",
        "target_metric": "reproducibility_rate",
        "operator": "capture and validate reproducibility-critical run metadata",
    },
    "repair_ownership_gap": {
        "component_type": "failure_recovery",
        "target_metric": "objective_score",
        "operator": "route unowned failures into bounded repair work",
    },
    "novelty_positioning_gap": {
        "component_type": "search_policy",
        "target_metric": "objective_score",
        "operator": "increase contradiction and adjacent-field literature probes",
    },
    "rigor_validation_gap": {
        "component_type": "agent_scaffold",
        "target_metric": "false_discovery_rate",
        "operator": "strengthen independent rigor challenge before writeup",
    },
    "clarity_gap": {
        "component_type": "prompt",
        "target_metric": "objective_score",
        "operator": "make argument structure and evidence boundaries explicit",
    },
    "issue_binding_gap": {
        "component_type": "tool",
        "target_metric": "reproducibility_rate",
        "operator": "enforce issue-to-claim, figure, or section bindings",
    },
    "verification_path_gap": {
        "component_type": "agent_scaffold",
        "target_metric": "reproducibility_rate",
        "operator": "require a verifier work item for every repair action",
    },
    "persistent_reviewer_debt": {
        "component_type": "failure_recovery",
        "target_metric": "objective_score",
        "operator": "escalate repeated unresolved failures without retry loops",
    },
    "stage_standard_blocker": {
        "component_type": "resource_allocation",
        "target_metric": "objective_score",
        "operator": "shift compute from generation to the blocked research stage",
    },
}


class EvolutionProgramError(ValueError):
    """Raised when an evolution portfolio violates organization invariants."""


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


def _strings(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(item).strip() for item in (values or []) if str(item).strip()})


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return text[:48] or "general"


def _priority_weight(value: Any) -> float:
    return {"p0": 3.0, "p1": 2.0, "p2": 1.0}.get(str(value or "p2"), 0.5)


def _program_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"generated_at", "program_hash"}
    }


def _source_snapshot(
    self_evolution: dict[str, Any],
    playbook: dict[str, Any],
    gate_history: Iterable[dict[str, Any]],
    program_history: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "self_evolution": {
            "summary": deepcopy(self_evolution.get("summary") or {}),
            "self_check": deepcopy(self_evolution.get("self_check") or {}),
            "lessons": deepcopy(self_evolution.get("lessons") or []),
            "stage_risks": deepcopy(self_evolution.get("stage_risks") or []),
            "harness_snapshot": deepcopy(self_evolution.get("harness_snapshot") or {}),
        },
        "playbook": {
            "project_count": int(playbook.get("project_count") or 0),
            "top_recurring_risks": deepcopy(playbook.get("top_recurring_risks") or []),
            "top_agentic_defaults": deepcopy(
                playbook.get("top_agentic_defaults") or []
            ),
        },
        "gate_history": [
            deepcopy(item) for item in gate_history if isinstance(item, dict)
        ][-50:],
        "program_history": [
            deepcopy(item) for item in program_history if isinstance(item, dict)
        ][-50:],
    }


def _route_for_lesson(lesson: dict[str, Any]) -> dict[str, str]:
    risk = str(lesson.get("risk") or "unclassified_failure").strip()
    if risk in RISK_ROUTING:
        return deepcopy(RISK_ROUTING[risk])
    if risk.startswith("harness_"):
        if "cost_budget" in risk:
            return {
                "component_type": "resource_allocation",
                "target_metric": "cost_per_task",
                "operator": "restore the declared resource boundary without changing evaluation rules",
            }
        if any(
            token in risk
            for token in (
                "behavior_regression",
                "score_divergence",
                "threshold_violation",
            )
        ):
            return {
                "component_type": "agent_scaffold",
                "target_metric": "false_discovery_rate",
                "operator": "make process-behavior regressions block score-only promotion",
            }
        if "score_no_improvement" in risk:
            return {
                "component_type": "search_policy",
                "target_metric": "objective_score",
                "operator": "explore a distinct causal intervention under the frozen objective",
            }
        return {
            "component_type": "tool",
            "target_metric": "reproducibility_rate",
            "operator": "enforce content-addressed harness comparability and fail-closed diagnostics",
        }
    stage = str(lesson.get("stage") or "review").strip()
    component_by_stage = {
        "ideation": "search_policy",
        "planning": "search_policy",
        "experiment": "tool",
        "figure": "tool",
        "manuscript": "prompt",
        "review": "agent_scaffold",
    }
    component = component_by_stage.get(stage, "failure_recovery")
    return {
        "component_type": component,
        "target_metric": "objective_score",
        "operator": "test one bounded mechanism addressing the observed failure",
    }


def _prior_intent_index(program_history: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for snapshot in program_history:
        for item in snapshot.get("intents") or []:
            if not isinstance(item, dict):
                continue
            fingerprint = str(item.get("fingerprint") or "")
            intent_id = str(item.get("intent_id") or "")
            if fingerprint and intent_id:
                result.setdefault(fingerprint, []).append(intent_id)
    return result


def _gate_outcome_index(
    gate_history: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for entry in gate_history:
        decision = str(entry.get("decision") or "unknown")
        for reference in entry.get("failure_taxonomy_refs") or []:
            risk = str(reference).removeprefix("failure:")
            counts = result.setdefault(_slug(risk), {})
            counts[decision] = counts.get(decision, 0) + 1
    return result


def _normalize_signals(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    evolution = snapshot["self_evolution"]
    playbook = snapshot["playbook"]
    recurrence = {
        str(item.get("risk") or ""): int(item.get("count") or 0)
        for item in playbook.get("top_recurring_risks") or []
        if isinstance(item, dict)
    }
    gate_outcomes = _gate_outcome_index(snapshot["gate_history"])
    grouped: dict[str, dict[str, Any]] = {}
    for lesson in evolution.get("lessons") or []:
        if not isinstance(lesson, dict):
            continue
        risk = str(lesson.get("risk") or "unclassified_failure").strip()
        stage = str(lesson.get("stage") or "review").strip()
        focus = str(lesson.get("focus") or risk).strip()
        fingerprint = _canonical_hash({"risk": risk, "stage": stage, "focus": focus})
        current = grouped.setdefault(
            fingerprint,
            {
                "fingerprint": fingerprint,
                "risk": risk,
                "stage": stage,
                "focus": focus,
                "priority_tier": str(lesson.get("priority_tier") or "p2"),
                "signals": [],
                "source_lesson_ids": [],
                "recommended_actions": [],
            },
        )
        current["signals"].extend(_strings([lesson.get("signal")]))
        current["source_lesson_ids"].extend(_strings([lesson.get("lesson_id")]))
        current["recommended_actions"].extend(
            _strings([lesson.get("recommended_action")])
        )
        if _priority_weight(lesson.get("priority_tier")) > _priority_weight(
            current["priority_tier"]
        ):
            current["priority_tier"] = str(lesson.get("priority_tier"))
    prior_index = _prior_intent_index(snapshot["program_history"])
    output: list[dict[str, Any]] = []
    for fingerprint, item in grouped.items():
        item["signals"] = _strings(item["signals"])
        item["source_lesson_ids"] = _strings(item["source_lesson_ids"])
        item["recommended_actions"] = _strings(item["recommended_actions"])
        support_count = max(len(item["source_lesson_ids"]), 1)
        recurrence_count = recurrence.get(item["risk"], 0)
        prior_count = len(prior_index.get(fingerprint, []))
        prior_gate_outcomes = gate_outcomes.get(_slug(item["risk"]), {})
        hold_count = int(prior_gate_outcomes.get("hold") or 0)
        approved_count = int(prior_gate_outcomes.get("approved") or 0)
        novelty_bonus = 0.5 if prior_count == 0 else 0.0
        score = (
            _priority_weight(item["priority_tier"])
            + min(recurrence_count, 5) * 0.35
            + min(support_count, 3) * 0.25
            + novelty_bonus
            + min(approved_count, 2) * 0.15
            - min(hold_count, 4) * 0.25
        )
        item.update(
            {
                "support_count": support_count,
                "cross_project_recurrence": recurrence_count,
                "prior_intent_count": prior_count,
                "prior_gate_outcomes": prior_gate_outcomes,
                "search_mode": (
                    "explore"
                    if hold_count >= 2
                    else (
                        "exploit"
                        if recurrence_count >= 2 or support_count >= 2
                        else "explore"
                    )
                ),
                "priority_score": round(score, 4),
                "parent_intent_ids": prior_index.get(fingerprint, [])[-3:],
            }
        )
        output.append(item)
    return sorted(
        output,
        key=lambda item: (
            -float(item["priority_score"]),
            str(item["risk"]),
            str(item["fingerprint"]),
        ),
    )


def _select_diverse_signals(
    signals: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    maximum = int(policy["maximum_active_intents"])
    per_component = int(policy["maximum_intents_per_component"])
    candidate_count = min(maximum, len(signals))
    exploration_slots = max(
        1,
        math.ceil(candidate_count * float(policy["minimum_exploration_fraction"])),
    )
    routed = [(deepcopy(item), _route_for_lesson(item)) for item in signals]
    selected: list[dict[str, Any]] = []
    component_counts: dict[str, int] = {}

    def take(mode: str | None, limit: int) -> None:
        for signal, route in routed:
            if len(selected) >= maximum or limit <= 0:
                return
            if signal in selected or (mode and signal.get("search_mode") != mode):
                continue
            component = route["component_type"]
            if component_counts.get(component, 0) >= per_component:
                continue
            selected.append(signal)
            component_counts[component] = component_counts.get(component, 0) + 1
            limit -= 1

    take("explore", exploration_slots)
    take("exploit", maximum - len(selected))
    take(None, maximum - len(selected))
    required_exploration = math.ceil(
        len(selected) * float(policy["minimum_exploration_fraction"])
    )
    current_exploration = sum(item.get("search_mode") == "explore" for item in selected)
    for signal in reversed(selected):
        if current_exploration >= required_exploration:
            break
        if signal.get("search_mode") != "exploit":
            continue
        signal["search_mode"] = "explore"
        signal["exploration_reason"] = "portfolio_exploration_floor"
        current_exploration += 1
    return selected


def _build_intent(signal: dict[str, Any], *, epoch_id: str) -> dict[str, Any]:
    route = _route_for_lesson(signal)
    component = route["component_type"]
    if component not in COMPONENT_TYPES or component in PROTECTED_COMPONENT_TYPES:
        raise EvolutionProgramError(f"unsafe routed component={component!r}")
    risk_slug = _slug(signal["risk"])
    scope = MUTABLE_COMPONENT_SCOPES[component] + risk_slug
    intent_seed = {
        "epoch_id": epoch_id,
        "fingerprint": signal["fingerprint"],
        "component_type": component,
        "scope": scope,
    }
    intent_id = "intent:" + _canonical_hash(intent_seed).split(":", 1)[1][:16]
    target_metric = route["target_metric"]
    return {
        "intent_id": intent_id,
        "fingerprint": signal["fingerprint"],
        "parent_intent_ids": signal["parent_intent_ids"],
        "status": "awaiting_candidate_artifact",
        "evolution_level": "system",
        "search_mode": signal["search_mode"],
        "exploration_reason": signal.get("exploration_reason"),
        "priority_tier": signal["priority_tier"],
        "priority_score": signal["priority_score"],
        "failure_class": signal["risk"],
        "source_lesson_ids": signal["source_lesson_ids"],
        "evidence_signals": signal["signals"],
        "support_count": signal["support_count"],
        "cross_project_recurrence": signal["cross_project_recurrence"],
        "prior_gate_outcomes": signal["prior_gate_outcomes"],
        "component_type": component,
        "change_scope": [scope],
        "applicability_domains": [f"stage:{signal['stage']}"],
        "failure_taxonomy_refs": [f"failure:{risk_slug}"],
        "ablation_dimensions": [risk_slug],
        "mutation_operator": route["operator"],
        "mechanism_constraint": (
            "Do not reuse a parent candidate's causal mechanism."
            if signal["search_mode"] == "explore"
            else "Exploit the supported mechanism while preserving one-change attribution."
        ),
        "hypothesis": (
            f"A single {component} change that will {route['operator']} reduces "
            f"{signal['risk']} without regressing protected scientific metrics."
        ),
        "falsifier": (
            f"Hold the intent if removing only {risk_slug} does not reduce "
            f"{target_metric}, or if any integrity, safety, or reproducibility gate regresses."
        ),
        "target_metrics": [target_metric],
        "recommended_actions": signal["recommended_actions"],
        "required_evidence": [
            "content-addressed base and candidate artifacts",
            "per-dimension ablation with run hashes",
            "sealed and prospective paired benchmarks",
            "two evaluator stacks distinct from the producer",
            "real-research canary with long-tail, OOD, and common-mode checks",
            "verified rollback receipt",
            "independent human approval",
        ],
        "execution_contract": {
            "protocol": "paired_shadow_ablation",
            "maximum_trials": CORE_PROGRAM_POLICY["maximum_trials_per_intent"],
            "futility_patience": CORE_PROGRAM_POLICY[
                "maximum_consecutive_non_improving_trials"
            ],
            "minimum_utility_improvement": CORE_PROGRAM_POLICY[
                "minimum_utility_improvement"
            ],
            "duplicate_mechanisms_count_as_progress": False,
            "hard_gate_failure_stops_epoch": True,
            "automatic_production_promotion_allowed": False,
        },
        "automatic_application_allowed": False,
        "requires_evolution_gate": True,
    }


def _evaluation_challenges(
    signals: list[dict[str, Any]], epoch_id: str
) -> list[dict[str, Any]]:
    challenge_risks = {
        "verification_path_gap",
        "rigor_validation_gap",
        "persistent_reviewer_debt",
    }
    challenges = []
    for signal in signals:
        if signal["risk"] not in challenge_risks and not str(signal["risk"]).startswith(
            "harness_"
        ):
            continue
        challenge_seed = {"epoch_id": epoch_id, "fingerprint": signal["fingerprint"]}
        challenges.append(
            {
                "challenge_id": "eval-challenge:"
                + _canonical_hash(challenge_seed).split(":", 1)[1][:16],
                "risk": signal["risk"],
                "evidence_signals": signal["signals"],
                "status": "epoch_boundary_human_review",
                "automatic_application_allowed": False,
                "earliest_application_epoch": "next",
                "protected_component": (
                    "evaluation_policy"
                    if "evaluator_mismatch" in str(signal["risk"])
                    else None
                ),
            }
        )
    return challenges


def build_evolution_program(
    project_root: str | Path,
    *,
    constitution: dict[str, Any],
    self_evolution: dict[str, Any],
    playbook: dict[str, Any] | None = None,
    gate_history: Iterable[dict[str, Any]] | None = None,
    program_history: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one fixed-utility epoch and a quality-diverse candidate portfolio."""

    assert_science_constitution_intact(constitution)
    if not isinstance(self_evolution, dict):
        raise EvolutionProgramError("self_evolution artifact is required")
    snapshot = _source_snapshot(
        self_evolution,
        playbook or {},
        gate_history or [],
        program_history or [],
    )
    epoch_index = len(snapshot["program_history"]) + 1
    epoch_seed = {
        "constitution_hash": constitution["constitution_hash"],
        "epoch_index": epoch_index,
        "source_hash": _canonical_hash(snapshot),
    }
    epoch_id = (
        f"epoch:{epoch_index:04d}:" + _canonical_hash(epoch_seed).split(":", 1)[1][:12]
    )
    signals = _normalize_signals(snapshot)
    selected = _select_diverse_signals(signals, CORE_PROGRAM_POLICY)
    intents = [_build_intent(item, epoch_id=epoch_id) for item in selected]
    gate_decisions: dict[str, int] = {}
    for entry in snapshot["gate_history"]:
        decision = str(entry.get("decision") or "unknown")
        gate_decisions[decision] = gate_decisions.get(decision, 0) + 1
    program_seed = {
        "epoch_id": epoch_id,
        "source_hash": _canonical_hash(snapshot),
        "intent_ids": [item["intent_id"] for item in intents],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "program_id": "evolution-program:"
        + _canonical_hash(program_seed).split(":", 1)[1][:16],
        "project_root": ".",
        "constitution_hash": constitution["constitution_hash"],
        "program_policy": deepcopy(CORE_PROGRAM_POLICY),
        "program_policy_hash": _canonical_hash(CORE_PROGRAM_POLICY),
        "evolution_levels": deepcopy(EVOLUTION_LEVELS),
        "epoch": {
            "epoch_id": epoch_id,
            "epoch_index": epoch_index,
            "status": "planning",
            "utility_mode": "fixed_within_epoch",
            "evaluation_policy_hash": _canonical_hash(DEFAULT_POLICY),
            "evaluator_change_window": "epoch_boundary_only",
        },
        "source_snapshot": snapshot,
        "signal_count": len(signals),
        "selected_intent_count": len(intents),
        "deferred_signal_count": max(len(signals) - len(intents), 0),
        "intents": intents,
        "evaluation_challenges": _evaluation_challenges(signals, epoch_id),
        "archive_summary": {
            "prior_epoch_count": len(snapshot["program_history"]),
            "gate_decision_counts": gate_decisions,
            "negative_results_retained": True,
        },
        "next_actions": [
            "Implement each intent as one content-addressed shadow candidate.",
            "Run declared ablations before consuming sealed or prospective tasks.",
            "Admit only gate-approved candidates to a real-research canary.",
            "Review evaluator challenges only when closing the epoch.",
        ],
        "production_mutation_allowed": False,
    }
    payload["program_hash"] = _canonical_hash(_program_core(payload))
    return payload


def validate_evolution_program(
    payload: dict[str, Any] | None,
    *,
    constitution: dict[str, Any],
) -> dict[str, Any]:
    program = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    try:
        assert_science_constitution_intact(constitution)
    except ValueError:
        errors.append("constitution_invalid")
    if program.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if program.get("constitution_hash") != constitution.get("constitution_hash"):
        errors.append("constitution_binding_mismatch")
    if program.get("program_policy") != CORE_PROGRAM_POLICY:
        errors.append("program_policy_modified")
    if program.get("program_policy_hash") != _canonical_hash(CORE_PROGRAM_POLICY):
        errors.append("program_policy_hash_mismatch")
    if program.get("evolution_levels") != EVOLUTION_LEVELS:
        errors.append("evolution_levels_modified")
    if program.get("production_mutation_allowed") is not False:
        errors.append("automatic_production_mutation_forbidden")
    epoch = program.get("epoch") if isinstance(program.get("epoch"), dict) else {}
    if epoch.get("utility_mode") != "fixed_within_epoch":
        errors.append("epoch_utility_not_fixed")
    if epoch.get("evaluator_change_window") != "epoch_boundary_only":
        errors.append("evaluator_change_window_invalid")
    intents = [item for item in program.get("intents") or [] if isinstance(item, dict)]
    if len(intents) > int(CORE_PROGRAM_POLICY["maximum_active_intents"]):
        errors.append("intent_budget_exceeded")
    required_exploration = math.ceil(
        len(intents) * float(CORE_PROGRAM_POLICY["minimum_exploration_fraction"])
    )
    if sum(item.get("search_mode") == "explore" for item in intents) < (
        required_exploration
    ):
        errors.append("exploration_budget_not_met")
    component_counts: dict[str, int] = {}
    intent_ids: set[str] = set()
    fingerprints: set[str] = set()
    for index, intent in enumerate(intents):
        component = str(intent.get("component_type") or "")
        component_counts[component] = component_counts.get(component, 0) + 1
        if component not in COMPONENT_TYPES or component in PROTECTED_COMPONENT_TYPES:
            errors.append(f"intent_component_invalid:{index}")
        scopes = _strings(intent.get("change_scope"))
        prefix = MUTABLE_COMPONENT_SCOPES.get(component, "forbidden/")
        if len(scopes) != 1 or not scopes[0].startswith(prefix):
            errors.append(f"intent_scope_invalid:{index}")
        if not str(intent.get("hypothesis") or "").strip():
            errors.append(f"intent_hypothesis_missing:{index}")
        if not str(intent.get("falsifier") or "").strip():
            errors.append(f"intent_falsifier_missing:{index}")
        if not intent.get("target_metrics") or not intent.get("required_evidence"):
            errors.append(f"intent_evidence_contract_missing:{index}")
        execution_contract = (
            intent.get("execution_contract")
            if isinstance(intent.get("execution_contract"), dict)
            else {}
        )
        if execution_contract.get("protocol") != "paired_shadow_ablation":
            errors.append(f"intent_execution_protocol_invalid:{index}")
        if execution_contract.get("maximum_trials") != CORE_PROGRAM_POLICY.get(
            "maximum_trials_per_intent"
        ):
            errors.append(f"intent_trial_budget_invalid:{index}")
        if (
            execution_contract.get("duplicate_mechanisms_count_as_progress")
            is not False
        ):
            errors.append(f"intent_duplicate_mechanism_policy_invalid:{index}")
        if (
            execution_contract.get("automatic_production_promotion_allowed")
            is not False
        ):
            errors.append(f"intent_production_promotion_policy_invalid:{index}")
        if len(_strings(intent.get("ablation_dimensions"))) != 1:
            errors.append(f"intent_causal_scope_invalid:{index}")
        if not str(intent.get("mechanism_constraint") or "").strip():
            errors.append(f"intent_mechanism_constraint_missing:{index}")
        if intent.get("automatic_application_allowed") is not False:
            errors.append(f"intent_automatic_application_forbidden:{index}")
        intent_id = str(intent.get("intent_id") or "")
        fingerprint = str(intent.get("fingerprint") or "")
        if not intent_id or intent_id in intent_ids:
            errors.append(f"intent_id_duplicate:{index}")
        if not fingerprint or fingerprint in fingerprints:
            errors.append(f"intent_fingerprint_duplicate:{index}")
        intent_ids.add(intent_id)
        fingerprints.add(fingerprint)
    if any(
        count > int(CORE_PROGRAM_POLICY["maximum_intents_per_component"])
        for count in component_counts.values()
    ):
        errors.append("component_intent_budget_exceeded")
    for index, challenge in enumerate(program.get("evaluation_challenges") or []):
        if not isinstance(challenge, dict) or (
            challenge.get("automatic_application_allowed") is not False
        ):
            errors.append(f"evaluation_challenge_unsafe:{index}")
    if program.get("program_hash") != _canonical_hash(_program_core(program)):
        errors.append("program_hash_mismatch")
    snapshot = (
        program.get("source_snapshot")
        if isinstance(program.get("source_snapshot"), dict)
        else {}
    )
    try:
        expected = build_evolution_program(
            str(program.get("project_root") or "."),
            constitution=constitution,
            self_evolution=snapshot.get("self_evolution") or {},
            playbook=snapshot.get("playbook") or {},
            gate_history=snapshot.get("gate_history") or [],
            program_history=snapshot.get("program_history") or [],
        )
        if _program_core(program) != _program_core(expected):
            errors.append("program_semantics_mismatch")
    except (EvolutionProgramError, TypeError, ValueError):
        errors.append("program_reconstruction_failed")
    return {"passed": not errors, "errors": errors}


def _history_snapshot(program: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": program.get("generated_at"),
        "program_id": program.get("program_id"),
        "epoch": deepcopy(program.get("epoch") or {}),
        "signal_count": program.get("signal_count"),
        "selected_intent_count": program.get("selected_intent_count"),
        "intents": [
            {
                "intent_id": item.get("intent_id"),
                "fingerprint": item.get("fingerprint"),
                "component_type": item.get("component_type"),
                "failure_class": item.get("failure_class"),
                "search_mode": item.get("search_mode"),
                "status": item.get("status"),
            }
            for item in program.get("intents") or []
            if isinstance(item, dict)
        ],
        "program_hash": program.get("program_hash"),
    }


def save_evolution_program(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    constitution: dict[str, Any],
    producer: str,
) -> str:
    check = validate_evolution_program(payload, constitution=constitution)
    if not check["passed"]:
        raise EvolutionProgramError(
            "evolution program invalid: " + ", ".join(check["errors"])
        )
    output = save_contract_artifact(
        project_root,
        "evolution_program",
        payload,
        producer=producer,
        depends_on=["science_constitution", "self_evolution"],
        notes="Fixed-utility epoch and quality-diverse system evolution intents.",
    )
    history_path = artifact_path(project_root, "evolution_program").with_name(
        "evolution_program_history.jsonl"
    )
    append_jsonl_artifact(history_path, _history_snapshot(payload))
    from ai_scientist.utils.evolution_controller import (
        evaluate_and_save_evolution_control,
    )

    evaluate_and_save_evolution_control(
        project_root,
        program=payload,
        producer=producer,
    )
    return output


def build_and_save_evolution_program(
    project_root: str | Path,
    *,
    self_evolution: dict[str, Any] | None = None,
    producer: str = "evolution_program",
) -> str:
    root = Path(project_root).expanduser().resolve()
    constitution = load_contract_artifact(root, "science_constitution", default={})
    evolution = (
        self_evolution
        if isinstance(self_evolution, dict)
        else load_contract_artifact(root, "self_evolution", default={})
    )
    from ai_scientist.utils.self_evolution import load_self_evolution_playbook

    playbook = load_self_evolution_playbook(root)
    gate_history = load_jsonl_artifact(root / "evolution_gate_history.jsonl")
    program_history = load_jsonl_artifact(root / "evolution_program_history.jsonl")
    program = build_evolution_program(
        root,
        constitution=constitution,
        self_evolution=evolution,
        playbook=playbook,
        gate_history=gate_history,
        program_history=program_history,
    )
    return save_evolution_program(
        root,
        program,
        constitution=constitution,
        producer=producer,
    )


__all__ = [
    "CORE_PROGRAM_POLICY",
    "EVOLUTION_LEVELS",
    "EvolutionProgramError",
    "build_and_save_evolution_program",
    "build_evolution_program",
    "save_evolution_program",
    "validate_evolution_program",
]
