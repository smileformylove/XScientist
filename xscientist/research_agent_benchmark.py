"""Bounded, auditable evaluation for a research-policy agent.

The benchmark deliberately separates three responsibilities:

* an LLM research policy chooses what to inspect and when to stop;
* deterministic local tools perform the statistical calculations; and
* a deterministic evaluator checks the resulting decision and evidence trail.

It is a synthetic process test, not evidence that a model can conduct real-world
science.  Raw prompts, completions, endpoint URLs, credentials, and exception
messages are never included in the returned episode or report.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from ai_scientist.protocol.schemas import load_schema, schema_registry, schema_validator
from ai_scientist.utils.provider_registry import (
    model_identity_status,
    model_provenance,
    resolve_model_provider,
)
from ai_scientist.utils.privacy import redact_sensitive_text

from ._version import __version__

BENCHMARK_SCHEMA = "xscientist.research-agent-benchmark.v1"
EPISODE_SCHEMA = "xscientist.research-agent-episode.v1"
SCORE_SCHEMA = "xscientist.research-agent-score.v1"
VERIFICATION_SCHEMA = "xscientist.research-agent-benchmark-verification.v1"
TASK_ID = "stratified-treatment-association-v1"
DEFAULT_MODEL = "openai_compat/glm-5.3"
RUBRIC_ID = "research-policy-confounding-v1"

_MAX_REPORT_BYTES = 2 * 1024 * 1024
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$")
_FORBIDDEN_RECORDED_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|credential|"
    r"secret|base[_-]?url|endpoint[_-]?url|raw[_-]?(?:prompt|response|output)|"
    r"exception[_-]?message|error[_-]?message)$",
    re.IGNORECASE,
)

# Aggregated synthetic observations.  Treatment is over-represented in the easy
# stratum, creating a pooled benefit even though the within-stratum association
# is negative in both strata.
_CELLS: tuple[dict[str, Any], ...] = (
    {
        "treatment": "treatment",
        "stratum": "easy",
        "successes": 72,
        "total": 90,
    },
    {
        "treatment": "treatment",
        "stratum": "hard",
        "successes": 1,
        "total": 10,
    },
    {
        "treatment": "control",
        "stratum": "easy",
        "successes": 9,
        "total": 10,
    },
    {
        "treatment": "control",
        "stratum": "hard",
        "successes": 18,
        "total": 90,
    },
)

_TASK_CONTRACT = {
    "task_id": TASK_ID,
    "task_kind": "synthetic_observational_confounding",
    "split": "public_smoke_test",
    "outcome": "binary_success",
    "treatment_variable": "treatment_assignment",
    "pre_specified_stratifier": "difficulty",
    "tool_set": [
        "inspect_design",
        "pooled_effect",
        "stratified_effect",
        "standardized_effect",
    ],
    "reference_visibility": "deterministic_evaluator_only",
    "real_world_data": False,
}

_RUBRIC_CONTRACT = {
    "rubric_id": RUBRIC_ID,
    "algorithm_version": "deterministic-confounding-rubric.v2",
    "threshold": 85,
    "numeric_tolerances": {
        "effect_grounding_absolute": 0.02,
        "stratum_grounding_absolute": 0.01,
        "reference_effect_absolute": 0.01,
    },
    "sections": {
        "research_strategy": 25,
        "evidence_grounding": 25,
        "scientific_inference": 25,
        "scientific_integrity": 20,
        "execution_discipline": 5,
    },
    "hard_gates": [
        "agent_completed",
        "model_identity_exact",
        "protocol_valid",
        "design_inspected",
        "pooled_analysis_executed",
        "stratified_analysis_executed",
        "standardized_analysis_executed",
        "correct_adjusted_direction",
        "effect_grounded",
        "all_required_evidence_cited",
        "reversal_identified",
        "decision_coherent",
        "confounding_recognized",
        "causal_boundary_respected",
        "negative_result_preserved",
        "usage_accounting_complete",
        "within_budgets",
    ],
}

_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "direction",
        "adjusted_effect",
        "analysis_basis",
        "stratum_effects",
        "pattern",
        "confounding_detected",
        "confounder",
        "pooled_result_rejected",
        "causal_status",
        "recommendation",
        "negative_result",
        "uncertainty",
        "limitations",
        "next_experiment",
        "evidence_tool_call_ids",
    ],
    "properties": {
        "direction": {"enum": ["beneficial", "harmful", "inconclusive"]},
        "adjusted_effect": {"type": "number", "minimum": -1, "maximum": 1},
        "analysis_basis": {
            "enum": ["pooled", "stratified", "standardized", "inconclusive"]
        },
        "stratum_effects": {
            "type": "object",
            "additionalProperties": False,
            "required": ["easy", "hard"],
            "properties": {
                "easy": {"type": "number", "minimum": -1, "maximum": 1},
                "hard": {"type": "number", "minimum": -1, "maximum": 1},
            },
        },
        "pattern": {"enum": ["simpson_reversal", "no_reversal", "uncertain"]},
        "confounding_detected": {"type": "boolean"},
        "confounder": {"enum": ["difficulty", "none", "unknown"]},
        "pooled_result_rejected": {"type": "boolean"},
        "causal_status": {"enum": ["not_identified", "identified", "uncertain"]},
        "recommendation": {
            "enum": [
                "reject_pooled_benefit_claim",
                "accept_pooled_benefit_claim",
                "withhold_conclusion",
            ]
        },
        "negative_result": {
            "enum": ["preserve_and_report", "discard", "not_applicable"]
        },
        "uncertainty": {"enum": ["low", "moderate", "high"]},
        "limitations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {
                "enum": [
                    "aggregated_data",
                    "observational_assignment",
                    "no_causal_identification",
                    "small_hard_treatment_stratum",
                ]
            },
        },
        "next_experiment": {
            "enum": [
                "stratified_randomized_replication",
                "collect_more_observational_data",
                "none",
            ]
        },
        "evidence_tool_call_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^tool-[0-9]{2}$"},
        },
    },
}

_ACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "tool", "arguments"],
            "properties": {
                "action": {"const": "tool"},
                "tool": {"enum": list(_TASK_CONTRACT["tool_set"])},
                "arguments": {
                    "type": "object",
                    "maxProperties": 0,
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "decision"],
            "properties": {
                "action": {"const": "final"},
                "decision": _DECISION_SCHEMA,
            },
        },
    ],
}

_ACTION_VALIDATOR = Draft202012Validator(_ACTION_SCHEMA)


class ResearchAgentBenchmarkError(ValueError):
    """Raised when the benchmark contract or transport wrapper is invalid."""


class ResearchAgentTransport(Protocol):
    """Minimal transport injected into the provider-neutral benchmark core.

    Implementations return a mapping with ``content`` and ``reported_model``.
    ``usage`` is optional and, when present, contains non-negative integer token
    counts.  Implementations must enforce the supplied per-call timeout.
    """

    def __call__(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_payload(value: Any) -> str:
    return (
        "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    )


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _strict_json_object(content: str) -> dict[str, Any]:
    value = json.loads(
        content,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError("agent action must be a JSON object")
    return value


def _bounded_int(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ResearchAgentBenchmarkError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _bounded_seconds(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchAgentBenchmarkError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 1 <= parsed <= 600:
        raise ResearchAgentBenchmarkError(f"{label} must be between 1 and 600")
    return parsed


def _normalize_model(model: Any) -> str:
    normalized = str(model or "").strip()
    if (
        not _SAFE_MODEL_RE.fullmatch(normalized)
        or redact_sensitive_text(normalized) != normalized
        or normalized.lower().startswith(("http://", "https://"))
    ):
        raise ResearchAgentBenchmarkError("model contains unsupported characters")
    return normalized


def _rate(successes: int, total: int) -> float:
    return round(successes / total, 6)


def _tool_inspect_design() -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {"treatment": {}, "control": {}}
    for cell in _CELLS:
        counts[cell["treatment"]][cell["stratum"]] = cell["total"]
    return {
        "design": "observational_aggregated_2x2",
        "stratifier": "difficulty",
        "strata": ["easy", "hard"],
        "allocation_imbalance": True,
        "sample_sizes": counts,
        "causal_identification": False,
    }


def _tool_pooled_effect() -> dict[str, Any]:
    totals: dict[str, list[int]] = {"treatment": [0, 0], "control": [0, 0]}
    for cell in _CELLS:
        bucket = totals[cell["treatment"]]
        bucket[0] += int(cell["successes"])
        bucket[1] += int(cell["total"])
    treatment_rate = _rate(*totals["treatment"])
    control_rate = _rate(*totals["control"])
    return {
        "estimand": "pooled_risk_difference",
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "rate_difference": round(treatment_rate - control_rate, 6),
        "adjusted_for": [],
    }


def _tool_stratified_effect() -> dict[str, Any]:
    effects: dict[str, dict[str, float]] = {}
    for stratum in ("easy", "hard"):
        selected = {
            cell["treatment"]: cell for cell in _CELLS if cell["stratum"] == stratum
        }
        treatment_rate = _rate(
            selected["treatment"]["successes"], selected["treatment"]["total"]
        )
        control_rate = _rate(
            selected["control"]["successes"], selected["control"]["total"]
        )
        effects[stratum] = {
            "treatment_rate": treatment_rate,
            "control_rate": control_rate,
            "rate_difference": round(treatment_rate - control_rate, 6),
        }
    return {
        "estimand": "within_stratum_risk_differences",
        "stratifier": "difficulty",
        "effects": effects,
        "direction_consistent": len(
            {math.copysign(1, row["rate_difference"]) for row in effects.values()}
        )
        == 1,
    }


def _tool_standardized_effect() -> dict[str, Any]:
    rates: dict[str, list[float]] = {"treatment": [], "control": []}
    for treatment in ("treatment", "control"):
        for stratum in ("easy", "hard"):
            cell = next(
                row
                for row in _CELLS
                if row["treatment"] == treatment and row["stratum"] == stratum
            )
            rates[treatment].append(_rate(cell["successes"], cell["total"]))
    treatment_rate = round(sum(rates["treatment"]) / 2, 6)
    control_rate = round(sum(rates["control"]) / 2, 6)
    return {
        "estimand": "equal_stratum_standardized_risk_difference",
        "standardization_weights": {"easy": 0.5, "hard": 0.5},
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "rate_difference": round(treatment_rate - control_rate, 6),
        "adjusted_for": ["difficulty"],
    }


_TOOLS: dict[str, Callable[[], dict[str, Any]]] = {
    "inspect_design": _tool_inspect_design,
    "pooled_effect": _tool_pooled_effect,
    "stratified_effect": _tool_stratified_effect,
    "standardized_effect": _tool_standardized_effect,
}


def research_agent_task_contract() -> dict[str, Any]:
    """Return the immutable, public part of the synthetic task contract."""

    return json.loads(_canonical_json(_TASK_CONTRACT))


def execute_research_benchmark_tool(name: str) -> dict[str, Any]:
    """Execute one argument-free deterministic benchmark tool."""

    normalized = str(name or "").strip()
    tool = _TOOLS.get(normalized)
    if tool is None:
        raise ResearchAgentBenchmarkError("tool is not allowed by the benchmark")
    return tool()


def _system_prompt() -> str:
    action_contract = {
        "tool_action": {
            "action": "tool",
            "tool": "<allowed tool name>",
            "arguments": {},
        },
        "final_action": {
            "action": "final",
            "decision": {
                "direction": "beneficial|harmful|inconclusive",
                "adjusted_effect": "number from -1 to 1",
                "analysis_basis": "pooled|stratified|standardized|inconclusive",
                "stratum_effects": {"easy": "number", "hard": "number"},
                "pattern": "simpson_reversal|no_reversal|uncertain",
                "confounding_detected": "boolean",
                "confounder": "difficulty|none|unknown",
                "pooled_result_rejected": "boolean",
                "causal_status": "not_identified|identified|uncertain",
                "recommendation": (
                    "reject_pooled_benefit_claim|accept_pooled_benefit_claim|"
                    "withhold_conclusion"
                ),
                "negative_result": "preserve_and_report|discard|not_applicable",
                "uncertainty": "low|moderate|high",
                "limitations": [
                    "aggregated_data|observational_assignment|"
                    "no_causal_identification|small_hard_treatment_stratum"
                ],
                "next_experiment": (
                    "stratified_randomized_replication|"
                    "collect_more_observational_data|none"
                ),
                "evidence_tool_call_ids": ["tool-01"],
            },
        },
    }
    return (
        "You are the research_policy in a controlled scientific evaluation. "
        "You choose the next analysis; a deterministic statistical executor "
        "performs calculations. Assess whether treatment is associated with a "
        "higher binary success rate in an observational aggregated study. "
        "Difficulty is a pre-specified potential confounder. Do not make a causal "
        "claim from observational data. Every reply MUST be exactly one JSON "
        "object with no Markdown or prose. Use only these argument-free tools: "
        "inspect_design, pooled_effect, stratified_effect, standardized_effect. "
        "When stopping, cite successful tool_call_ids that directly support the "
        "decision. The exact action contract is: " + _canonical_json(action_contract)
    )


def _task_prompt() -> str:
    return (
        "Task stratified-treatment-association-v1: choose the analyses needed "
        "to reach a "
        "scientifically defensible conclusion about the treatment association. "
        "The outcome is binary success; the study is observational and aggregated; "
        "difficulty has easy and hard strata. You have a small tool budget, so "
        "prioritize analyses that can distinguish a pooled association from a "
        "confounded one."
    )


def _task_fingerprint_payload() -> dict[str, Any]:
    return {
        "contract": _TASK_CONTRACT,
        "cells": _CELLS,
        "system_prompt_sha256": _hash_payload(_system_prompt()),
        "task_prompt_sha256": _hash_payload(_task_prompt()),
        "action_schema": _ACTION_SCHEMA,
    }


def _rubric_fingerprint_payload() -> dict[str, Any]:
    return {
        "rubric": _RUBRIC_CONTRACT,
        "task_sha256": _hash_payload(_task_fingerprint_payload()),
        "deterministic_tool_results": {
            name: execute_research_benchmark_tool(name)
            for name in _TASK_CONTRACT["tool_set"]
        },
    }


def _implementation_sha256() -> str:
    """Bind reports to the exact local evaluator implementation without paths."""

    try:
        content = Path(__file__).read_bytes()
    except OSError as exc:
        raise ResearchAgentBenchmarkError(
            "cannot fingerprint the local benchmark implementation"
        ) from exc
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _usage_row(
    value: Any, *, max_output_tokens: int
) -> tuple[dict[str, int | None], bool]:
    if value is not None and not isinstance(value, Mapping):
        raise ResearchAgentBenchmarkError("transport usage must be an object")
    usage = value if isinstance(value, Mapping) else {}
    normalized: dict[str, int | None] = {}
    complete = True
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = usage.get(key)
        if item is None:
            normalized[key] = None
            complete = False
        elif isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ResearchAgentBenchmarkError("transport usage is invalid")
        else:
            normalized[key] = item
    if complete:
        prompt_tokens = int(normalized["prompt_tokens"] or 0)
        completion_tokens = int(normalized["completion_tokens"] or 0)
        total_tokens = int(normalized["total_tokens"] or 0)
        if total_tokens != prompt_tokens + completion_tokens:
            raise ResearchAgentBenchmarkError("transport usage totals are inconsistent")
        if completion_tokens > max_output_tokens:
            raise ResearchAgentBenchmarkError(
                "transport completion exceeds requested output limit"
            )
    return normalized, complete


def _sum_usage(rows: Sequence[Mapping[str, int | None]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [row.get(key) for row in rows]
        result[key] = (
            sum(int(value) for value in values if value is not None)
            if values and all(value is not None for value in values)
            else None
        )
    return result


def _transport_response(
    value: Any,
    *,
    max_response_bytes: int,
    max_output_tokens: int,
) -> tuple[str, str | None, dict[str, int | None], bool]:
    if not isinstance(value, Mapping):
        raise ResearchAgentBenchmarkError("transport response must be an object")
    content = value.get("content")
    if not isinstance(content, str):
        raise ResearchAgentBenchmarkError("transport content must be text")
    encoded = content.encode("utf-8")
    if len(encoded) > max_response_bytes:
        raise ResearchAgentBenchmarkError("transport content exceeds response limit")
    reported = str(value.get("reported_model") or "").strip() or None
    if reported is not None:
        if (
            not _SAFE_MODEL_RE.fullmatch(reported)
            or redact_sensitive_text(reported) != reported
        ):
            raise ResearchAgentBenchmarkError("reported model identity is invalid")
    usage, usage_complete = _usage_row(
        value.get("usage"), max_output_tokens=max_output_tokens
    )
    return content, reported, usage, usage_complete


def _transport_error_code(exc: BaseException) -> str:
    if isinstance(exc, ResearchAgentBenchmarkError):
        return "invalid_transport_response"
    return "transport_exception"


def run_research_agent_episode(
    transport: ResearchAgentTransport,
    *,
    model: str = DEFAULT_MODEL,
    execution_mode: str = "offline_test",
    provider_environment: Mapping[str, str] | None = None,
    max_turns: int = 6,
    max_tool_calls: int = 4,
    max_seconds: float = 120.0,
    max_output_tokens: int = 512,
    max_total_tokens: int = 20_000,
    max_response_bytes: int = 16_384,
) -> dict[str, Any]:
    """Run one bounded policy episode against the fixed synthetic task.

    ``execution_mode`` is a caller-owned declaration: use ``offline_test`` for
    scripted/fake transports and ``live_provider`` only from an explicit live
    wrapper.  The transport itself cannot silently relabel network use.
    """

    if not callable(transport):
        raise ResearchAgentBenchmarkError("transport must be callable")
    normalized_model = _normalize_model(model)
    if execution_mode not in {"offline_test", "live_provider"}:
        raise ResearchAgentBenchmarkError(
            "execution_mode must be offline_test or live_provider"
        )
    turn_limit = _bounded_int(max_turns, label="max_turns", minimum=1, maximum=12)
    tool_limit = _bounded_int(
        max_tool_calls, label="max_tool_calls", minimum=1, maximum=8
    )
    output_limit = _bounded_int(
        max_output_tokens,
        label="max_output_tokens",
        minimum=64,
        maximum=4096,
    )
    total_token_limit = _bounded_int(
        max_total_tokens,
        label="max_total_tokens",
        minimum=512,
        maximum=1_000_000,
    )
    response_limit = _bounded_int(
        max_response_bytes,
        label="max_response_bytes",
        minimum=512,
        maximum=65_536,
    )
    wall_limit = _bounded_seconds(max_seconds, label="max_seconds")

    spec = resolve_model_provider(normalized_model)
    expected_reported_model = _normalize_model(spec.client_model)
    provenance = model_provenance(
        normalized_model,
        env={} if provider_environment is None else provider_environment,
    )
    started = time.monotonic()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _task_prompt()},
    ]
    trace: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    usage_rows: list[dict[str, int | None]] = []
    usage_complete = True
    violations: list[str] = []
    reported_models: list[str] = []
    decision: dict[str, Any] | None = None
    termination = "turn_budget_exhausted"
    transport_error_code: str | None = None
    budget_exceeded = False

    for turn in range(1, turn_limit + 1):
        elapsed = time.monotonic() - started
        remaining = wall_limit - elapsed
        if remaining <= 0:
            termination = "wall_time_exhausted"
            budget_exceeded = True
            break
        try:
            response = transport(
                messages=tuple(dict(message) for message in messages),
                model=normalized_model,
                max_output_tokens=output_limit,
                timeout_seconds=max(0.1, remaining),
            )
            content, reported_model, usage, response_usage_complete = (
                _transport_response(
                    response,
                    max_response_bytes=response_limit,
                    max_output_tokens=output_limit,
                )
            )
        except Exception as exc:  # transport boundary: never persist its message
            termination = "transport_failed"
            transport_error_code = _transport_error_code(exc)
            break

        usage_rows.append(usage)
        usage_complete = usage_complete and response_usage_complete
        elapsed_after = time.monotonic() - started
        if elapsed_after > wall_limit:
            termination = "wall_time_exhausted"
            budget_exceeded = True
            trace.append(
                {
                    "turn": turn,
                    "response_sha256": _hash_payload(content),
                    "response_bytes": len(content.encode("utf-8")),
                    "response_content_recorded": False,
                    "reported_model": None,
                    "reported_model_sha256": (
                        _hash_payload(reported_model)
                        if reported_model is not None
                        else None
                    ),
                    "reported_model_recorded": False,
                    "identity_status": "not_checked_after_timeout",
                    "action_kind": "discarded",
                    "action_sha256": None,
                    "tool": None,
                    "tool_call_id": None,
                    "observation_sha256": None,
                    "usage": usage,
                }
            )
            break

        identity = model_identity_status(expected_reported_model, reported_model)
        recorded_model = expected_reported_model if identity == "exact" else None
        if recorded_model is not None:
            reported_models.append(recorded_model)
        base_trace = {
            "turn": turn,
            "response_sha256": _hash_payload(content),
            "response_bytes": len(content.encode("utf-8")),
            "response_content_recorded": False,
            "reported_model": recorded_model,
            "reported_model_sha256": (
                _hash_payload(reported_model) if reported_model is not None else None
            ),
            "reported_model_recorded": recorded_model is not None,
            "identity_status": identity,
            "action_kind": "invalid",
            "action_sha256": None,
            "tool": None,
            "tool_call_id": None,
            "observation_sha256": None,
            "usage": usage,
        }
        if identity != "exact":
            violations.append("model_identity_not_exact")
            base_trace["action_kind"] = "identity_rejected"
            trace.append(base_trace)
            termination = "model_identity_unverified"
            break

        cumulative_usage = _sum_usage(usage_rows)
        if (
            cumulative_usage["total_tokens"] is not None
            and cumulative_usage["total_tokens"] > total_token_limit
        ):
            base_trace["action_kind"] = "discarded"
            trace.append(base_trace)
            termination = "token_budget_exhausted"
            budget_exceeded = True
            break

        try:
            action = _strict_json_object(content)
            errors = sorted(_ACTION_VALIDATOR.iter_errors(action), key=str)
            if errors:
                raise ValueError("action schema validation failed")
        except (
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            RecursionError,
            MemoryError,
        ):
            violations.append("invalid_action_schema")
            trace.append(base_trace)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        '{"error":"invalid_action_schema","retry_allowed":true}'
                    ),
                }
            )
            if violations.count("invalid_action_schema") >= 2:
                termination = "protocol_violation_limit"
                break
            continue

        if action["action"] == "final":
            decision = json.loads(_canonical_json(action["decision"]))
            base_trace["action_kind"] = "final"
            base_trace["action_sha256"] = _hash_payload(action)
            trace.append(base_trace)
            termination = "completed"
            break

        tool_name = str(action["tool"])
        base_trace["action_sha256"] = _hash_payload(action)
        if len(observations) >= tool_limit:
            violations.append("tool_budget_exceeded")
            base_trace["action_kind"] = "tool_rejected"
            base_trace["tool"] = tool_name
            trace.append(base_trace)
            termination = "tool_budget_exhausted"
            budget_exceeded = True
            break
        result = execute_research_benchmark_tool(tool_name)
        tool_call_id = f"tool-{len(observations) + 1:02d}"
        observation = {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "arguments_sha256": _hash_payload({}),
            "outcome": "success",
            "result": result,
            "result_sha256": _hash_payload(result),
        }
        observations.append(observation)
        base_trace.update(
            {
                "action_kind": "tool",
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "observation_sha256": observation["result_sha256"],
            }
        )
        trace.append(base_trace)
        messages.extend(
            [
                {"role": "assistant", "content": _canonical_json(action)},
                {
                    "role": "user",
                    "content": _canonical_json(
                        {
                            "tool_call_id": tool_call_id,
                            "tool": tool_name,
                            "outcome": "success",
                            "result": result,
                        }
                    ),
                },
            ]
        )

    if termination == "turn_budget_exhausted":
        budget_exceeded = True
    duration = round(time.monotonic() - started, 3)
    all_exact = bool(trace) and all(row["identity_status"] == "exact" for row in trace)
    episode = {
        "schema": EPISODE_SCHEMA,
        "task_id": TASK_ID,
        "task_sha256": _hash_payload(_task_fingerprint_payload()),
        "policy_contract": {
            "decision_owner": "research_policy",
            "execution_owner": "deterministic_statistical_executor",
            "evaluation_owner": "deterministic_rubric",
            "model_requested": normalized_model,
            "provider": spec.provider,
            "expected_reported_model": expected_reported_model,
            "configuration_observation": provenance,
            "configuration_observation_source": (
                "none"
                if provider_environment is None
                else "runner_supplied_environment"
            ),
            "configuration_verified": False,
        },
        "limits": {
            "max_turns": turn_limit,
            "max_tool_calls": tool_limit,
            "max_seconds": wall_limit,
            "max_output_tokens_per_turn": output_limit,
            "max_total_output_tokens": turn_limit * output_limit,
            "max_total_tokens": total_token_limit,
            "max_response_bytes_per_turn": response_limit,
            "max_invalid_actions": 1,
        },
        "execution": {
            "mode": execution_mode,
            "network_used_declared": execution_mode == "live_provider",
            "provider_used_declared": execution_mode == "live_provider",
            "network_use_verified": False,
            "provider_execution_verified": False,
            "agent_execution_completed": termination == "completed",
            "termination_reason": termination,
            "turns_observed": len(trace),
            "tool_calls_executed": len(observations),
            "protocol_violations": sorted(set(violations)),
            "budget_exceeded": budget_exceeded,
            "duration_seconds": duration,
            "model_identity_exact": all_exact,
            "reported_models": sorted(set(reported_models)),
            "usage": _sum_usage(usage_rows),
            "usage_complete": usage_complete and bool(usage_rows),
            "transport_error_code": transport_error_code,
        },
        "trace": trace,
        "observations": observations,
        "decision": decision,
        "retention": {
            "raw_prompt_recorded": False,
            "raw_response_recorded": False,
            "response_content_recorded": False,
            "tool_results_recorded": True,
            "credentials_recorded": False,
            "endpoint_url_recorded": False,
            "exception_messages_recorded": False,
            "structured_decision_recorded": decision is not None,
        },
    }
    return episode


def _observation_effect(observation: Mapping[str, Any]) -> float | None:
    tool = observation.get("tool")
    result = observation.get("result")
    if not isinstance(result, Mapping):
        return None
    if tool == "standardized_effect":
        value = result.get("rate_difference")
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )
    if tool == "stratified_effect":
        effects = result.get("effects")
        if not isinstance(effects, Mapping) or not effects:
            return None
        values: list[float] = []
        for row in effects.values():
            if not isinstance(row, Mapping):
                return None
            value = row.get("rate_difference")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            values.append(float(value))
        return sum(values) / len(values) if values else None
    return None


def score_research_agent_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically score an episode; never call an LLM judge."""

    if not isinstance(episode, Mapping) or episode.get("schema") != EPISODE_SCHEMA:
        raise ResearchAgentBenchmarkError("episode schema is invalid")
    benchmark_schema = load_schema("research_agent_benchmark")
    try:
        Draft202012Validator(
            benchmark_schema,
            registry=schema_registry(),
        ).evolve(
            schema=benchmark_schema["$defs"]["episode"]
        ).validate(episode)
    except Exception:
        raise ResearchAgentBenchmarkError("episode schema is invalid") from None
    required_episode_fields = {
        "schema",
        "task_id",
        "task_sha256",
        "policy_contract",
        "limits",
        "execution",
        "trace",
        "observations",
        "decision",
        "retention",
    }
    if set(episode) != required_episode_fields:
        raise ResearchAgentBenchmarkError("episode fields are invalid")
    decision_for_validation = episode.get("decision")
    if decision_for_validation is not None and list(
        Draft202012Validator(_DECISION_SCHEMA).iter_errors(decision_for_validation)
    ):
        raise ResearchAgentBenchmarkError("episode decision is invalid")
    if _verify_observations(episode) or _verify_episode_consistency(episode):
        raise ResearchAgentBenchmarkError("episode provenance is invalid")
    execution = episode.get("execution")
    observations_value = episode.get("observations")
    decision_value = episode.get("decision")
    limits = episode.get("limits")
    if not isinstance(execution, Mapping) or not isinstance(limits, Mapping):
        raise ResearchAgentBenchmarkError("episode execution contract is invalid")
    if not isinstance(observations_value, Sequence) or isinstance(
        observations_value, (str, bytes)
    ):
        raise ResearchAgentBenchmarkError("episode observations are invalid")
    observations = [row for row in observations_value if isinstance(row, Mapping)]
    decision = decision_value if isinstance(decision_value, Mapping) else {}
    by_id = {
        str(row.get("tool_call_id")): row
        for row in observations
        if str(row.get("tool_call_id") or "")
    }
    tools = [str(row.get("tool") or "") for row in observations]
    cited_ids_value = decision.get("evidence_tool_call_ids")
    cited_ids = (
        list(cited_ids_value)
        if isinstance(cited_ids_value, Sequence)
        and not isinstance(cited_ids_value, (str, bytes))
        else []
    )
    cited_rows = [by_id.get(str(item)) for item in cited_ids]
    valid_citations = bool(cited_ids) and all(row is not None for row in cited_rows)
    cited_adjusted = [
        row
        for row in cited_rows
        if isinstance(row, Mapping)
        and row.get("tool") in {"stratified_effect", "standardized_effect"}
    ]
    cited_tools = {
        str(row.get("tool")) for row in cited_rows if isinstance(row, Mapping)
    }
    cited_by_tool = {
        str(row.get("tool")): row for row in cited_rows if isinstance(row, Mapping)
    }
    decision_effect = decision.get("adjusted_effect")
    effect_number = (
        float(decision_effect)
        if isinstance(decision_effect, (int, float))
        and not isinstance(decision_effect, bool)
        and math.isfinite(float(decision_effect))
        else None
    )
    grounded_effects = [
        value
        for value in (_observation_effect(row) for row in cited_adjusted)
        if value is not None
    ]
    effect_grounded = (
        bool(grounded_effects)
        and effect_number is not None
        and any(abs(effect_number - value) <= 0.02 for value in grounded_effects)
    )
    stratified_row = cited_by_tool.get("stratified_effect")
    stratified_result = (
        stratified_row.get("result")
        if isinstance(stratified_row, Mapping)
        and isinstance(stratified_row.get("result"), Mapping)
        else {}
    )
    observed_strata = stratified_result.get("effects")
    decision_strata = decision.get("stratum_effects")
    stratum_effects_grounded = (
        bool(observed_strata)
        and isinstance(observed_strata, Mapping)
        and isinstance(decision_strata, Mapping)
        and all(
            isinstance(observed_strata.get(stratum), Mapping)
            and isinstance(
                observed_strata[stratum].get("rate_difference"), (int, float)
            )
            and not isinstance(observed_strata[stratum].get("rate_difference"), bool)
            and isinstance(decision_strata.get(stratum), (int, float))
            and not isinstance(decision_strata.get(stratum), bool)
            and abs(
                float(decision_strata[stratum])
                - float(observed_strata[stratum].get("rate_difference"))
            )
            <= 0.01
            for stratum in ("easy", "hard")
        )
    )
    pooled_row = cited_by_tool.get("pooled_effect")
    pooled_result = (
        pooled_row.get("result")
        if isinstance(pooled_row, Mapping)
        and isinstance(pooled_row.get("result"), Mapping)
        else {}
    )
    pooled_effect = pooled_result.get("rate_difference")
    standardized_row = cited_by_tool.get("standardized_effect")
    standardized_effect = _observation_effect(standardized_row or {})
    analysis_basis = decision.get("analysis_basis")
    basis_row = cited_by_tool.get(
        "standardized_effect"
        if analysis_basis == "standardized"
        else "stratified_effect" if analysis_basis == "stratified" else ""
    )
    basis_effect = _observation_effect(basis_row or {})
    basis_grounded = (
        basis_effect is not None
        and effect_number is not None
        and abs(effect_number - basis_effect) <= 0.02
    )
    reversal_coherent = (
        isinstance(pooled_effect, (int, float))
        and not isinstance(pooled_effect, bool)
        and float(pooled_effect) > 0
        and standardized_effect is not None
        and standardized_effect < 0
        and stratum_effects_grounded
        and all(float(decision_strata[stratum]) < 0 for stratum in ("easy", "hard"))
        and decision.get("pattern") == "simpson_reversal"
    )
    decision_coherent = (
        basis_grounded
        and stratum_effects_grounded
        and reversal_coherent
        and decision.get("direction") == "harmful"
        and effect_number is not None
        and effect_number < 0
    )
    limitations_value = decision.get("limitations")
    limitations = (
        set(limitations_value)
        if isinstance(limitations_value, Sequence)
        and not isinstance(limitations_value, (str, bytes))
        else set()
    )
    required_limitations = {
        "aggregated_data",
        "observational_assignment",
        "no_causal_identification",
    }
    uncertainty_calibrated = decision.get("uncertainty") in {"moderate", "high"}
    follow_up_coherent = (
        decision.get("next_experiment") == "stratified_randomized_replication"
    )
    protocol_violations = execution.get("protocol_violations")
    violations = (
        list(protocol_violations)
        if isinstance(protocol_violations, Sequence)
        and not isinstance(protocol_violations, (str, bytes))
        else ["malformed_protocol_violations"]
    )

    strategy = 0
    strategy += 6 if "inspect_design" in tools else 0
    strategy += 5 if "pooled_effect" in tools else 0
    strategy += 7 if "stratified_effect" in tools else 0
    strategy += 7 if "standardized_effect" in tools else 0

    grounding = 0
    grounding += 5 if valid_citations else 0
    grounding += 5 if "inspect_design" in cited_tools else 0
    grounding += 5 if "pooled_effect" in cited_tools else 0
    grounding += 5 if "stratified_effect" in cited_tools else 0
    grounding += 5 if "standardized_effect" in cited_tools else 0

    inference = 0
    inference += 5 if decision.get("direction") == "harmful" else 0
    if effect_number is not None:
        distance = abs(effect_number - (-0.1))
        inference += 5 if distance <= 0.01 else 2 if distance <= 0.05 else 0
    inference += 6 if stratum_effects_grounded else 0
    inference += 5 if reversal_coherent else 0
    inference += (
        4
        if (
            decision.get("confounding_detected") is True
            and decision.get("confounder") == "difficulty"
        )
        else 0
    )

    integrity = 0
    integrity += (
        4
        if (
            decision.get("pooled_result_rejected") is True
            and decision.get("recommendation") == "reject_pooled_benefit_claim"
        )
        else 0
    )
    integrity += (
        6
        if (
            decision.get("causal_status") == "not_identified"
            and "no_causal_identification" in limitations
        )
        else 0
    )
    integrity += 4 if decision.get("negative_result") == "preserve_and_report" else 0
    integrity += 1 if "observational_assignment" in limitations else 0
    integrity += 1 if "aggregated_data" in limitations else 0
    integrity += 2 if decision.get("uncertainty") in {"moderate", "high"} else 0
    integrity += (
        2
        if (decision.get("next_experiment") == "stratified_randomized_replication")
        else 0
    )

    discipline = 0
    discipline += 2 if execution.get("agent_execution_completed") is True else 0
    discipline += 2 if not violations else 0
    discipline += 1 if execution.get("model_identity_exact") is True else 0

    sections = {
        "research_strategy": strategy,
        "evidence_grounding": grounding,
        "scientific_inference": inference,
        "scientific_integrity": integrity,
        "execution_discipline": discipline,
    }
    total = sum(sections.values())
    hard_gates = {
        "agent_completed": execution.get("agent_execution_completed") is True,
        "model_identity_exact": execution.get("model_identity_exact") is True,
        "protocol_valid": not violations,
        "design_inspected": "inspect_design" in tools,
        "pooled_analysis_executed": "pooled_effect" in tools,
        "stratified_analysis_executed": "stratified_effect" in tools,
        "standardized_analysis_executed": "standardized_effect" in tools,
        "correct_adjusted_direction": (
            decision.get("direction") == "harmful"
            and effect_number is not None
            and effect_number < 0
        ),
        "effect_grounded": effect_grounded,
        "all_required_evidence_cited": set(_TASK_CONTRACT["tool_set"]).issubset(
            cited_tools
        ),
        "reversal_identified": reversal_coherent,
        "decision_coherent": decision_coherent,
        "confounding_recognized": (
            decision.get("confounding_detected") is True
            and decision.get("confounder") == "difficulty"
            and decision.get("pooled_result_rejected") is True
        ),
        "causal_boundary_respected": (
            decision.get("causal_status") == "not_identified"
            and "no_causal_identification" in limitations
        ),
        "limitations_complete": required_limitations.issubset(limitations),
        "uncertainty_calibrated": uncertainty_calibrated,
        "follow_up_experiment_identified": follow_up_coherent,
        "negative_result_preserved": (
            decision.get("negative_result") == "preserve_and_report"
            and decision.get("recommendation") == "reject_pooled_benefit_claim"
        ),
        "usage_accounting_complete": execution.get("usage_complete") is True,
        "within_budgets": (
            execution.get("budget_exceeded") is False
            and int(execution.get("turns_observed") or 0)
            <= int(limits.get("max_turns") or 0)
            and int(execution.get("tool_calls_executed") or 0)
            <= int(limits.get("max_tool_calls") or 0)
            and isinstance((execution.get("usage") or {}).get("total_tokens"), int)
            and int((execution.get("usage") or {}).get("total_tokens"))
            <= int(limits.get("max_total_tokens") or 0)
        ),
    }
    return {
        "schema": SCORE_SCHEMA,
        "rubric_id": RUBRIC_ID,
        "rubric_sha256": _hash_payload(_rubric_fingerprint_payload()),
        "score": total,
        "score_max": 100,
        "threshold": 85,
        "sections": sections,
        "hard_gates": hard_gates,
        "benchmark_contract_passed": total >= 85 and all(hard_gates.values()),
        "judge": "deterministic_local_rubric",
        "llm_judge_used": False,
    }


def _stable_report_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(report.get("episode", {}).get("execution") or {})
    execution.pop("duration_seconds", None)
    episode = dict(report.get("episode") or {})
    episode["execution"] = execution
    return {
        "schema": report.get("schema"),
        "version": report.get("version"),
        "task": report.get("task"),
        "episode": episode,
        "score": report.get("score"),
        "claims": report.get("claims"),
        "comparison_boundary": report.get("comparison_boundary"),
    }


def benchmark_research_agent(
    transport: ResearchAgentTransport,
    **episode_options: Any,
) -> dict[str, Any]:
    """Run and deterministically score one research-policy smoke-test episode."""

    episode = run_research_agent_episode(transport, **episode_options)
    score = score_research_agent_episode(episode)
    claims = {
        "agent_execution_completed": bool(
            episode["execution"]["agent_execution_completed"]
        ),
        "benchmark_contract_passed": bool(score["benchmark_contract_passed"]),
        "rollout_audit_complete": True,
        "scientific_contract_verified": False,
        "quality_claim_allowed": False,
        "causal_claim_allowed": False,
        "real_world_truth_claim_allowed": False,
        "generalization_claim_allowed": False,
        "cross_model_comparison_allowed": False,
        "cross_system_comparison_allowed": False,
        "production_promotion_allowed": False,
        "report_authenticity_verified": False,
        "live_rollout_verified": False,
        "research_taste_claim_allowed": False,
        "independent_scientific_review_completed": False,
    }
    report: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system().lower(),
        },
        "ok": bool(score["benchmark_contract_passed"]),
        "task": {
            **research_agent_task_contract(),
            "task_sha256": episode["task_sha256"],
        },
        "episode": episode,
        "score": score,
        "claims": claims,
        "comparison_boundary": {
            "scope": "single_fixed_synthetic_task",
            "official_benchmark": False,
            "external_scores_injected": False,
            "matched_cross_model_conditions": False,
            "real_scientific_result_evaluated": False,
            "human_baseline_run": False,
        },
        "reproducibility": {
            "task_sha256": episode["task_sha256"],
            "rubric_sha256": score["rubric_sha256"],
            "implementation_sha256": _implementation_sha256(),
            "trace_sha256": _hash_payload(
                {"trace": episode["trace"], "observations": episode["observations"]}
            ),
            "fingerprint": None,
            "deterministic_fields": True,
            "excluded_observations": [
                "generated_at",
                "runtime",
                "episode.execution.duration_seconds",
            ],
        },
    }
    report["reproducibility"]["fingerprint"] = _hash_payload(
        _stable_report_payload(report)
    )
    try:
        schema_validator("research_agent_benchmark").validate(report)
    except Exception as exc:
        raise ResearchAgentBenchmarkError(
            "generated benchmark report failed its published schema"
        ) from exc
    verification = verify_research_agent_benchmark(report)
    if not verification["ok"]:
        raise ResearchAgentBenchmarkError(
            "generated benchmark report failed deterministic self-audit"
        )
    return report


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _FORBIDDEN_RECORDED_KEY.search(str(key)) or _has_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_forbidden_key(item) for item in value)
    if isinstance(value, str):
        return redact_sensitive_text(value) != value
    return False


def _verify_observations(episode: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    observations = episode.get("observations")
    if not isinstance(observations, list):
        return ["observations_invalid"]
    for index, row in enumerate(observations, start=1):
        if not isinstance(row, Mapping):
            errors.append("observation_invalid")
            continue
        expected_id = f"tool-{index:02d}"
        tool = str(row.get("tool") or "")
        if row.get("tool_call_id") != expected_id:
            errors.append("tool_call_sequence_invalid")
        try:
            expected = execute_research_benchmark_tool(tool)
        except ResearchAgentBenchmarkError:
            errors.append("observation_tool_not_allowed")
            continue
        if row.get("result") != expected:
            errors.append("observation_result_mismatch")
        if row.get("result_sha256") != _hash_payload(expected):
            errors.append("observation_digest_mismatch")
        if row.get("arguments_sha256") != _hash_payload({}):
            errors.append("observation_arguments_mismatch")
    return errors


def _verify_episode_consistency(episode: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    execution = episode.get("execution")
    policy = episode.get("policy_contract")
    limits = episode.get("limits")
    trace = episode.get("trace")
    observations = episode.get("observations")
    decision = episode.get("decision")
    if not all(
        (
            isinstance(execution, Mapping),
            isinstance(policy, Mapping),
            isinstance(limits, Mapping),
            isinstance(trace, list),
            isinstance(observations, list),
        )
    ):
        return ["episode_structure_invalid"]

    expected_task_hash = _hash_payload(_task_fingerprint_payload())
    if (
        episode.get("schema") != EPISODE_SCHEMA
        or episode.get("task_id") != TASK_ID
        or episode.get("task_sha256") != expected_task_hash
    ):
        errors.append("task_binding_mismatch")
    try:
        requested_model = _normalize_model(policy.get("model_requested"))
        resolved = resolve_model_provider(requested_model)
        expected_model = _normalize_model(resolved.client_model)
    except (ResearchAgentBenchmarkError, ValueError):
        expected_model = ""
        resolved = None
        errors.append("policy_model_binding_invalid")
    if resolved is not None and (
        policy.get("provider") != resolved.provider
        or policy.get("expected_reported_model") != expected_model
        or policy.get("decision_owner") != "research_policy"
        or policy.get("execution_owner") != "deterministic_statistical_executor"
        or policy.get("evaluation_owner") != "deterministic_rubric"
    ):
        errors.append("policy_model_binding_mismatch")
    configuration = policy.get("configuration_observation")
    if not isinstance(configuration, Mapping):
        errors.append("provider_configuration_observation_invalid")
    elif resolved is not None:
        configuration_core = {
            "provider": configuration.get("provider"),
            "requested_model": configuration.get("requested_model"),
            "client_model": configuration.get("client_model"),
            "request_style": configuration.get("request_style"),
            "endpoint_fingerprint": configuration.get("endpoint_fingerprint"),
        }
        if (
            configuration_core
            != {
                "provider": resolved.provider,
                "requested_model": requested_model,
                "client_model": resolved.client_model,
                "request_style": resolved.request_style,
                "endpoint_fingerprint": configuration.get("endpoint_fingerprint"),
            }
            or configuration.get("configuration_fingerprint")
            != _hash_payload(configuration_core)
            or configuration.get("endpoint_configured")
            is not bool(configuration.get("endpoint_fingerprint"))
            or policy.get("configuration_verified") is not False
        ):
            errors.append("provider_configuration_observation_mismatch")
        if policy.get("configuration_observation_source") == "none" and (
            configuration.get("endpoint_env") is not None
            or configuration.get("api_key_env") is not None
        ):
            errors.append("unexpected_provider_configuration_observation")
        if (
            resolved.provider == "openai_compat"
            and policy.get("configuration_observation_source")
            == "runner_supplied_environment"
            and (
                configuration.get("endpoint_env") != "OPENAI_COMPAT_BASE_URL"
                or configuration.get("api_key_env") != "OPENAI_COMPAT_API_KEY"
                or configuration.get("endpoint_configured") is not True
            )
        ):
            errors.append("dedicated_provider_configuration_required")

    expected_model = str(policy.get("expected_reported_model") or "")
    if execution.get("turns_observed") != len(trace):
        errors.append("turn_count_mismatch")
    if execution.get("tool_calls_executed") != len(observations):
        errors.append("tool_count_mismatch")
    if [row.get("turn") for row in trace if isinstance(row, Mapping)] != list(
        range(1, len(trace) + 1)
    ):
        errors.append("turn_sequence_invalid")

    exact_rows: list[bool] = []
    for row in trace:
        if not isinstance(row, Mapping):
            exact_rows.append(False)
            continue
        is_exact = (
            row.get("identity_status") == "exact"
            and row.get("reported_model") == expected_model
            and row.get("reported_model_recorded") is True
            and row.get("reported_model_sha256") == _hash_payload(expected_model)
        )
        exact_rows.append(is_exact)
        if row.get("identity_status") != "exact" and (
            row.get("reported_model") is not None
            or row.get("reported_model_recorded") is not False
        ):
            errors.append("untrusted_reported_model_recorded")
    expected_identity = bool(trace) and all(exact_rows)
    if execution.get("model_identity_exact") is not expected_identity:
        errors.append("model_identity_summary_mismatch")
    reported = sorted(
        {
            str(row.get("reported_model"))
            for row in trace
            if isinstance(row, Mapping) and row.get("reported_model") is not None
        }
    )
    if execution.get("reported_models") != reported:
        errors.append("reported_models_mismatch")

    mode = execution.get("mode")
    expected_live = mode == "live_provider"
    if (
        execution.get("network_used_declared") is not expected_live
        or execution.get("provider_used_declared") is not expected_live
        or execution.get("network_use_verified") is not False
        or execution.get("provider_execution_verified") is not False
    ):
        errors.append("execution_mode_flags_mismatch")

    tool_rows = [
        row
        for row in trace
        if isinstance(row, Mapping) and row.get("action_kind") == "tool"
    ]
    if len(tool_rows) != len(observations):
        errors.append("trace_observation_count_mismatch")
    else:
        for trace_row, observation in zip(tool_rows, observations):
            if not isinstance(observation, Mapping) or (
                trace_row.get("tool") != observation.get("tool")
                or trace_row.get("tool_call_id") != observation.get("tool_call_id")
                or trace_row.get("observation_sha256")
                != observation.get("result_sha256")
            ):
                errors.append("trace_observation_binding_mismatch")
            expected_action = {
                "action": "tool",
                "tool": trace_row.get("tool"),
                "arguments": {},
            }
            if trace_row.get("action_sha256") != _hash_payload(expected_action):
                errors.append("tool_action_binding_mismatch")

    for row in trace:
        if not isinstance(row, Mapping):
            continue
        kind = row.get("action_kind")
        if kind == "tool_rejected":
            expected_action = {
                "action": "tool",
                "tool": row.get("tool"),
                "arguments": {},
            }
            if row.get("action_sha256") != _hash_payload(expected_action):
                errors.append("rejected_action_binding_mismatch")
        elif kind not in {"tool", "final"} and row.get("action_sha256") is not None:
            errors.append("unparsed_action_digest_present")

    termination = execution.get("termination_reason")
    completed = execution.get("agent_execution_completed") is True
    terminal_kinds = {"final", "identity_rejected", "tool_rejected", "discarded"}
    terminal_rows = [
        (index, row)
        for index, row in enumerate(trace)
        if isinstance(row, Mapping) and row.get("action_kind") in terminal_kinds
    ]
    if terminal_rows and (
        len(terminal_rows) != 1 or terminal_rows[0][0] != len(trace) - 1
    ):
        errors.append("terminal_trace_state_invalid")
    expected_terminal_kind = {
        "completed": "final",
        "model_identity_unverified": "identity_rejected",
        "tool_budget_exhausted": "tool_rejected",
        "token_budget_exhausted": "discarded",
    }.get(termination)
    if expected_terminal_kind is not None:
        if not trace or trace[-1].get("action_kind") != expected_terminal_kind:
            errors.append("termination_trace_binding_mismatch")
    elif terminal_rows:
        # A wall-time exhaustion may append a discarded provider response, but
        # no other non-mapped termination may contain a terminal action row.
        if not (
            termination == "wall_time_exhausted"
            and trace[-1].get("action_kind") == "discarded"
        ):
            errors.append("termination_trace_binding_mismatch")
    if any(
        isinstance(row, Mapping) and row.get("action_kind") == "discarded"
        for row in trace
    ) and termination not in {"wall_time_exhausted", "token_budget_exhausted"}:
        errors.append("discarded_trace_without_budget_termination")

    final_rows = [
        row
        for row in trace
        if isinstance(row, Mapping) and row.get("action_kind") == "final"
    ]
    if completed != (termination == "completed"):
        errors.append("completion_summary_mismatch")
    if completed:
        if len(final_rows) != 1 or final_rows[-1] is not trace[-1] or decision is None:
            errors.append("final_decision_binding_mismatch")
        elif final_rows[0].get("action_sha256") != _hash_payload(
            {"action": "final", "decision": decision}
        ):
            errors.append("final_action_binding_mismatch")
    elif decision is not None or final_rows:
        errors.append("unfinished_decision_present")

    usage_rows: list[dict[str, int | None]] = []
    usage_rows_complete = True
    for row in trace:
        if not isinstance(row, Mapping):
            usage_rows_complete = False
            continue
        try:
            usage, complete = _usage_row(
                row.get("usage"),
                max_output_tokens=int(limits.get("max_output_tokens_per_turn") or 0),
            )
        except ResearchAgentBenchmarkError:
            errors.append("usage_row_invalid")
            usage_rows_complete = False
            continue
        usage_rows.append(usage)
        usage_rows_complete = usage_rows_complete and complete
    expected_usage = _sum_usage(usage_rows)
    expected_usage_complete = bool(usage_rows) and usage_rows_complete
    if (
        execution.get("usage") != expected_usage
        or execution.get("usage_complete") is not expected_usage_complete
    ):
        errors.append("usage_summary_mismatch")

    violations = set(execution.get("protocol_violations") or [])
    trace_kinds = {row.get("action_kind") for row in trace if isinstance(row, Mapping)}
    expected_violation_markers = {
        "invalid_action_schema": "invalid" in trace_kinds,
        "model_identity_not_exact": any(
            isinstance(row, Mapping)
            and row.get("identity_status") in {"alias", "mismatch", "unavailable"}
            for row in trace
        ),
        "tool_budget_exceeded": "tool_rejected" in trace_kinds,
    }
    for code, observed in expected_violation_markers.items():
        if (code in violations) != observed:
            errors.append("protocol_violation_summary_mismatch")
            break

    budget_termination = termination in {
        "turn_budget_exhausted",
        "wall_time_exhausted",
        "tool_budget_exhausted",
        "token_budget_exhausted",
    }
    if execution.get("budget_exceeded") is not budget_termination:
        errors.append("budget_summary_mismatch")
    error_code = execution.get("transport_error_code")
    if (error_code is not None) != (termination == "transport_failed"):
        errors.append("transport_error_summary_mismatch")
    if error_code not in {None, "invalid_transport_response", "transport_exception"}:
        errors.append("transport_error_code_invalid")
    duration = execution.get("duration_seconds")
    max_seconds = limits.get("max_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0
        or (
            termination == "completed"
            and isinstance(max_seconds, (int, float))
            and float(duration) > float(max_seconds) + 0.1
        )
    ):
        errors.append("duration_observation_invalid")
    return errors


def verify_research_agent_benchmark(
    report: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a saved benchmark report offline and fail closed on tampering."""

    checks = {
        "schema": "not_checked",
        "privacy_boundary": "not_checked",
        "episode_consistency": "not_checked",
        "deterministic_tools": "not_checked",
        "score_recomputed": "not_checked",
        "claim_boundary": "not_checked",
        "reproducibility": "not_checked",
    }
    errors: list[str] = []
    payload: Any
    try:
        if isinstance(report, (str, Path)):
            path = Path(report).expanduser().resolve()
            if path.stat().st_size > _MAX_REPORT_BYTES:
                raise ValueError("report exceeds size limit")
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        else:
            payload = report
        if not isinstance(payload, Mapping):
            raise ValueError("report must be an object")
        schema_validator("research_agent_benchmark").validate(payload)
        checks["schema"] = "passed"
    except Exception:
        checks["schema"] = "failed"
        return {
            "schema": VERIFICATION_SCHEMA,
            "ok": False,
            "checks": checks,
            "errors": ["report_invalid"],
            "network_used": False,
            "provider_used": False,
        }

    episode = payload["episode"]
    if _has_forbidden_key(payload):
        errors.append("privacy_boundary_violated")
        checks["privacy_boundary"] = "failed"
    else:
        retention = episode["retention"]
        privacy_ok = (
            retention["raw_prompt_recorded"] is False
            and retention["raw_response_recorded"] is False
            and retention["response_content_recorded"] is False
            and retention["credentials_recorded"] is False
            and retention["endpoint_url_recorded"] is False
            and retention["exception_messages_recorded"] is False
            and all(
                row["response_content_recorded"] is False for row in episode["trace"]
            )
        )
        checks["privacy_boundary"] = "passed" if privacy_ok else "failed"
        if not privacy_ok:
            errors.append("privacy_boundary_mutated")

    tool_errors = _verify_observations(episode)
    if tool_errors:
        errors.extend(tool_errors)
        checks["deterministic_tools"] = "failed"
    else:
        checks["deterministic_tools"] = "passed"

    consistency_errors = _verify_episode_consistency(episode)
    if consistency_errors:
        errors.extend(consistency_errors)
        checks["episode_consistency"] = "failed"
    else:
        checks["episode_consistency"] = "passed"

    try:
        recomputed_score = score_research_agent_episode(episode)
    except ResearchAgentBenchmarkError:
        recomputed_score = None
    if recomputed_score != payload["score"]:
        errors.append("score_mismatch")
        checks["score_recomputed"] = "failed"
    else:
        checks["score_recomputed"] = "passed"

    claims = payload["claims"]
    claim_ok = (
        claims["agent_execution_completed"]
        == episode["execution"]["agent_execution_completed"]
        and claims["benchmark_contract_passed"]
        == payload["score"]["benchmark_contract_passed"]
        and claims["rollout_audit_complete"] is True
        and claims["scientific_contract_verified"] is False
        and claims["quality_claim_allowed"] is False
        and claims["causal_claim_allowed"] is False
        and claims["real_world_truth_claim_allowed"] is False
        and claims["generalization_claim_allowed"] is False
        and claims["cross_model_comparison_allowed"] is False
        and claims["cross_system_comparison_allowed"] is False
        and claims["production_promotion_allowed"] is False
        and claims["report_authenticity_verified"] is False
        and claims["live_rollout_verified"] is False
        and claims["research_taste_claim_allowed"] is False
        and claims["independent_scientific_review_completed"] is False
        and payload["ok"] == claims["benchmark_contract_passed"]
    )
    checks["claim_boundary"] = "passed" if claim_ok else "failed"
    if not claim_ok:
        errors.append("claim_boundary_mutated")

    reproducibility = payload["reproducibility"]
    expected_task_hash = _hash_payload(_task_fingerprint_payload())
    expected_trace_hash = _hash_payload(
        {"trace": episode["trace"], "observations": episode["observations"]}
    )
    reproducible = (
        episode["task_sha256"] == expected_task_hash
        and payload["task"]["task_sha256"] == expected_task_hash
        and reproducibility["task_sha256"] == expected_task_hash
        and reproducibility["rubric_sha256"]
        == _hash_payload(_rubric_fingerprint_payload())
        and reproducibility["implementation_sha256"] == _implementation_sha256()
        and reproducibility["trace_sha256"] == expected_trace_hash
        and reproducibility["fingerprint"]
        == _hash_payload(_stable_report_payload(payload))
    )
    checks["reproducibility"] = "passed" if reproducible else "failed"
    if not reproducible:
        errors.append("reproducibility_mismatch")

    return {
        "schema": VERIFICATION_SCHEMA,
        "ok": not errors,
        "checks": checks,
        "errors": sorted(set(errors)),
        "network_used": False,
        "provider_used": False,
    }


__all__ = [
    "BENCHMARK_SCHEMA",
    "DEFAULT_MODEL",
    "ResearchAgentBenchmarkError",
    "ResearchAgentTransport",
    "benchmark_research_agent",
    "execute_research_benchmark_tool",
    "research_agent_task_contract",
    "run_research_agent_episode",
    "score_research_agent_episode",
    "verify_research_agent_benchmark",
]
