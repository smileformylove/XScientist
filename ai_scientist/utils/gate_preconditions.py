from __future__ import annotations

"""Fail-fast checks for gates that cannot be satisfied by retrying blindly."""

from dataclasses import dataclass
from typing import Any

from ai_scientist.utils.truth_contracts import validate_hallucination_checks, validate_truth_contract


class UnsatisfiableGateError(RuntimeError):
    """Raised when a requested gate is impossible under current hard inputs."""


@dataclass(frozen=True)
class GatePreconditionContext:
    research_plan: dict[str, Any]
    truth_contract: dict[str, Any] | None = None
    hallucination_checks: dict[str, Any] | None = None
    sample_gate: dict[str, Any] | None = None
    improvement_rounds: int = 0
    require_quality_gate: bool = False
    high_quality_mode: bool = False
    target_venue: str | None = None
    require_sample_gate: bool = True


def _min_rounds_for_venue(target_venue: str | None) -> int:
    venue = str(target_venue or "").strip().lower()
    if venue in {"journal", "nature"}:
        return 2
    return 1


def _workflow_mode(plan: dict[str, Any]) -> str:
    return str(plan.get("workflow_mode") or "").strip()


def evaluate_gate_preconditions(context: GatePreconditionContext) -> dict[str, Any]:
    reasons: list[str] = []
    checks: list[dict[str, Any]] = []
    plan = context.research_plan if isinstance(context.research_plan, dict) else {}
    workflow_mode = _workflow_mode(plan)
    high_risk_workflow = workflow_mode in {"review_board", "multi_agent_board"}

    sample_gate = context.sample_gate if isinstance(context.sample_gate, dict) else {}
    if context.require_sample_gate and sample_gate.get("full_generation_allowed") is not True:
        reasons.append("sample_gate_blocked")
        checks.append(
            {
                "id": "sample_gate",
                "passed": False,
                "detail": sample_gate.get("result", {}).get("reasons", []),
            }
        )
    else:
        checks.append(
            {
                "id": "sample_gate",
                "passed": True,
                "required": bool(context.require_sample_gate),
            }
        )

    if context.require_quality_gate and not context.high_quality_mode:
        reasons.append("quality_gate_requires_high_quality_mode")
        checks.append({"id": "quality_gate_mode", "passed": False})
    else:
        checks.append({"id": "quality_gate_mode", "passed": True})

    min_rounds = _min_rounds_for_venue(context.target_venue or plan.get("target_venue"))
    if context.improvement_rounds > 0 and context.improvement_rounds < min_rounds:
        reasons.append("improvement_round_budget_below_gate_minimum")
        checks.append(
            {
                "id": "improvement_round_budget",
                "passed": False,
                "detail": {
                    "configured": int(context.improvement_rounds),
                    "required_minimum": min_rounds,
                },
            }
        )
    else:
        checks.append({"id": "improvement_round_budget", "passed": True})

    if high_risk_workflow:
        truth_contract = (
            context.truth_contract
            if isinstance(context.truth_contract, dict)
            else plan.get("truth_contract")
        )
        truth_validation = validate_truth_contract(
            truth_contract if isinstance(truth_contract, dict) else None
        )
        if not truth_validation.get("passed"):
            reasons.append("truth_contract_invalid")
            checks.append(
                {
                    "id": "truth_contract",
                    "passed": False,
                    "detail": truth_validation.get("errors", []),
                }
            )
        else:
            checks.append({"id": "truth_contract", "passed": True})

        hallucination_checks = (
            context.hallucination_checks
            if isinstance(context.hallucination_checks, dict)
            else plan.get("hallucination_checks")
        )
        hallucination_validation = validate_hallucination_checks(
            hallucination_checks if isinstance(hallucination_checks, dict) else None,
            research_plan=plan,
        )
        if not hallucination_validation.get("passed"):
            reasons.append("hallucination_checks_invalid")
            checks.append(
                {
                    "id": "hallucination_checks",
                    "passed": False,
                    "detail": hallucination_validation.get("errors", []),
                }
            )
        else:
            checks.append({"id": "hallucination_checks", "passed": True})

    return {
        "satisfiable": not reasons,
        "pause_reason": None if not reasons else "unsatisfiable_gate",
        "reasons": reasons,
        "checks": checks,
    }


def assert_gate_preconditions_satisfiable(context: GatePreconditionContext) -> dict[str, Any]:
    result = evaluate_gate_preconditions(context)
    if result["satisfiable"]:
        return result
    raise UnsatisfiableGateError(
        "Unsatisfiable gate preconditions: " + ", ".join(result["reasons"])
    )
