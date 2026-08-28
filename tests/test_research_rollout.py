from __future__ import annotations

import json
import re

import pytest

from xscientist import (
    ResearchRepository,
    ResearchRolloutError,
    assess_tool_swap_compatibility,
    audit_research_rollout,
    build_comparison_boundary,
    build_independence_attestation_payload,
    build_independence_receipt,
    build_replication_rubric,
    build_research_rollout,
    build_strategy_budget_summary,
    build_tool_delegation_trace,
    build_turn_credit_summary,
    evaluate_replication_rollout,
    rollout_producer_actor_ids,
)
from ai_scientist.protocol.attestation import sign_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema
from jsonschema import validate
from xscientist.research_cli import main as research_main
from xscientist.research_rollout import _rollout_audit_result

HASH = "sha256:" + "a" * 64
DIMENSIONS = {
    "result_fidelity": 0.8,
    "claim_support": 0.7,
    "implementation_fidelity": 0.9,
    "resource_efficiency": 0.6,
    "scientific_integrity": 1.0,
}
SIGNING_SECRET = "rollout-test-signing-secret"
SIGNING_KEY_ID = "judge-independent-key"
EVALUATOR_IDENTITY = "agent:judge-independent"
PRODUCER_ACTOR_IDS = sorted(
    {
        "coding_executor",
        "codex",
        "executor-v1",
        "research_policy",
        "xscientist.research-policy-tool-delegation.v1",
    }
)


def _episode(**overrides):
    payload = {
        "task_id": "heldout-task-1",
        "task_hash": HASH,
        "split": "holdout",
        "time_budget_seconds": 3600,
        "tool_delegations": [
            {
                "call_id": "call-1",
                "role": "coding_executor",
                "tool": "codex",
                "model": "executor-v1",
                "decision": "execute",
                "outcome": "success",
                "input_hash": HASH,
                "output_hash": HASH,
                "budget_before_seconds": 3600,
                "budget_after_seconds": 3000,
            }
        ],
        "turns": [
            {
                "turn_id": "turn-1",
                "type": "plan",
                "reward_before": 0,
                "reward_after": 0.2,
            },
            {
                "turn_id": "turn-2",
                "type": "execute",
                "reward_before": 0.2,
                "reward_after": 0.8,
            },
        ],
        "evaluations": [
            {
                "sample_id": "sample-1",
                "evaluator_id": "judge-a",
                "scores": DIMENSIONS,
                "evidence_refs": [HASH],
            },
            {
                "sample_id": "sample-2",
                "evaluator_id": "judge-b",
                "scores": {**DIMENSIONS, "claim_support": 0.8},
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_rollout_records_policy_tool_rubric_and_observational_credit():
    rollout = build_research_rollout(_episode())
    assert rollout["policy_contract"]["execution_owner"] == "coding_executor"
    assert rollout["rubric"]["reference_visibility"] == "evaluator_only"
    assert rollout["turn_credit"]["credit_method"] == "post_hoc_positive_delta_v1"
    assert rollout["turn_credit"]["causal_claim_allowed"] is False
    assert rollout["evaluation"]["summary"]["sample_count"] == 2
    assert rollout["quality_claim_allowed"] is False
    assert json.dumps(rollout, allow_nan=False)


def test_rubric_is_locked_to_task_and_rejects_nonfinite_scores():
    rubric = build_replication_rubric(task_id="task-1", task_hash=HASH)
    assert rubric["rubric_hash"].startswith("sha256:")
    with pytest.raises(ResearchRolloutError):
        evaluate_replication_rollout(
            rubric,
            [
                {
                    "sample_id": "s",
                    "evaluator_id": "e",
                    "scores": {**DIMENSIONS, "claim_support": float("nan")},
                }
            ],
        )


def test_nonuniform_rubric_weights_quantize_to_exact_unit_sum():
    raw_weights = {
        "result_fidelity": 0.5,
        "claim_support": 0.5,
        "implementation_fidelity": 0.5,
        "resource_efficiency": 0.5,
        "scientific_integrity": 1.0,
    }
    rubric = build_replication_rubric(
        task_id="heldout-task-1",
        task_hash=HASH,
        split="holdout",
        dimensions={key: {"weight": weight} for key, weight in raw_weights.items()},
    )
    published_weights = [row["weight"] for row in rubric["dimensions"]]
    assert sum(round(weight * 1_000_000) for weight in published_weights) == 1_000_000
    assert sum(published_weights) == pytest.approx(1.0, abs=1e-12)

    perfect_scores = {dimension: 1.0 for dimension in DIMENSIONS}
    evaluation = evaluate_replication_rollout(
        rubric,
        [
            {
                "sample_id": "perfect",
                "evaluator_id": "judge-a",
                "scores": perfect_scores,
            }
        ],
    )
    assert evaluation["summary"]["overall_mean"] == 1.0

    rollout = build_research_rollout(
        _episode(
            rubric=rubric,
            evaluations=[
                {
                    "sample_id": "perfect",
                    "evaluator_id": "judge-a",
                    "scores": perfect_scores,
                }
            ],
        )
    )
    assert rollout["evaluation"]["summary"]["overall_mean"] == 1.0
    validate(rollout, load_schema("research_rollout"))


def test_rehashed_external_rubric_cannot_bypass_unit_weight_sum():
    rubric = build_replication_rubric(
        task_id="heldout-task-1",
        task_hash=HASH,
        split="holdout",
    )
    tampered = json.loads(json.dumps(rubric))
    tampered["dimensions"][0]["weight"] = 0.3
    core_keys = (
        "schema_version",
        "task_id",
        "task_hash",
        "split",
        "reference_visibility",
        "dimensions",
        "scoring_scale",
        "quality_claim_allowed",
    )
    tampered_hash = canonical_content_hash({key: tampered[key] for key in core_keys})
    tampered["rubric_hash"] = tampered_hash
    tampered["rubric_id"] = "rubric-" + tampered_hash.split(":", 1)[1][:16]

    with pytest.raises(ResearchRolloutError, match="weights must sum to 1"):
        evaluate_replication_rollout(tampered, [])
    with pytest.raises(ResearchRolloutError, match="weights must sum to 1"):
        build_research_rollout(_episode(rubric=tampered))


def test_evaluation_rejects_more_than_thirty_two_evidence_refs() -> None:
    refs = [f"sha256:{index:064x}" for index in range(33)]
    with pytest.raises(ResearchRolloutError, match="exceeds 32"):
        build_research_rollout(
            _episode(
                evaluations=[
                    {
                        "sample_id": "too-many-refs",
                        "evaluator_id": "judge-a",
                        "scores": DIMENSIONS,
                        "evidence_refs": refs,
                    }
                ]
            )
        )


def test_tool_calls_are_metadata_only_and_bound_by_hash():
    with pytest.raises(ResearchRolloutError, match="sensitive"):
        build_tool_delegation_trace([{"call_id": "c", "prompt": "do this"}])
    with pytest.raises(ResearchRolloutError, match="output_hash"):
        build_tool_delegation_trace(
            [{"call_id": "c", "tool": "codex", "outcome": "success"}]
        )


def test_missing_reward_trace_does_not_impute_credit():
    summary = build_turn_credit_summary([{"type": "inspect", "outcome": "observed"}])
    assert summary["credit_method"] == "not_available"
    assert summary["turns"][0]["credit"] is None
    assert summary["reward_trace_complete"] is False


def test_tool_swap_requires_same_task_rubric_split_and_budget():
    left = build_research_rollout(_episode())
    right = build_research_rollout(_episode(tool_delegations=[]))
    result = assess_tool_swap_compatibility(left, right)
    assert result["eligible"] is True
    assert result["quality_claim_allowed"] is False
    right["task_hash"] = "sha256:" + "b" * 64
    assert assess_tool_swap_compatibility(left, right)["eligible"] is False


def test_tool_swap_can_bind_harness_resource_and_evaluator_boundary():
    boundary = {
        "harness_id": "replica-harness-v1",
        "resource_fingerprint": HASH,
        "evaluator_protocol_hash": "sha256:" + "b" * 64,
        "starting_artifact_hash": HASH,
        "network_policy": "internet-on",
        "seed_policy": "fixed-per-task",
    }
    assert build_comparison_boundary(boundary)["comparison_scope"] == (
        "harness_resource_evaluator_bound"
    )
    left = build_research_rollout(_episode(comparison_boundary=boundary))
    right = build_research_rollout(_episode(comparison_boundary=boundary))
    result = assess_tool_swap_compatibility(left, right)
    assert result["eligible"] is True
    assert result["comparison_scope"] == "harness_resource_evaluator_bound"
    changed = dict(boundary, network_policy="offline")
    changed_right = build_research_rollout(_episode(comparison_boundary=changed))
    assert (
        "comparison_boundary_mismatch"
        in assess_tool_swap_compatibility(left, changed_right)["reasons"]
    )


def test_strategy_budget_summary_exposes_contiguous_accounting_and_next_action():
    summary = build_strategy_budget_summary(
        time_budget_seconds=100,
        tool_delegations=[
            {
                "call_id": "plan-1",
                "role": "research_policy",
                "tool": "planner",
                "decision": "plan",
                "outcome": "success",
                "output_hash": HASH,
                "budget_before_seconds": 100,
                "budget_after_seconds": 90,
            },
            {
                "call_id": "execute-1",
                "role": "coding_executor",
                "tool": "executor",
                "decision": "execute",
                "outcome": "success",
                "output_hash": HASH,
                "budget_before_seconds": 90,
                "budget_after_seconds": 40,
            },
        ],
    )
    assert summary["budget_accounting"]["status"] == "complete"
    assert summary["budget_accounting"]["observed_consumed_seconds"] == 60.0
    assert summary["budget_accounting"]["remaining_seconds"] == 40.0
    assert summary["next_action"] == "inspect"
    assert summary["ownership"]["status"] == "consistent"
    assert summary["quality_claim_allowed"] is False


def test_strategy_budget_summary_fails_closed_on_budget_gaps():
    summary = build_strategy_budget_summary(
        time_budget_seconds=100,
        tool_delegations=[
            {
                "call_id": "a",
                "tool": "executor",
                "outcome": "success",
                "output_hash": HASH,
                "budget_before_seconds": 100,
                "budget_after_seconds": 80,
            },
            {
                "call_id": "b",
                "tool": "executor",
                "outcome": "success",
                "output_hash": HASH,
                "budget_before_seconds": 70,
                "budget_after_seconds": 60,
            },
        ],
    )
    assert summary["budget_accounting"]["status"] == "inconsistent"
    assert "non_contiguous_budget" in summary["budget_accounting"]["violations"]
    assert summary["budget_accounting"]["within_declared_budget"] is None


def test_strategy_budget_summary_records_explicit_failure_recovery():
    summary = build_strategy_budget_summary(
        time_budget_seconds=100,
        tool_delegations=[
            {
                "call_id": "failed",
                "tool": "executor",
                "outcome": "failed",
                "follow_up_required": True,
                "budget_before_seconds": 100,
                "budget_after_seconds": 70,
            },
            {
                "call_id": "repair",
                "role": "research_policy",
                "tool": "executor",
                "decision": "repair",
                "outcome": "success",
                "output_hash": HASH,
                "budget_before_seconds": 70,
                "budget_after_seconds": 50,
            },
        ],
    )
    recovery = summary["failure_recovery"]
    assert recovery["failure_count"] == 1
    assert recovery["required_recovery_count"] == 1
    assert recovery["recovered_count"] == 1
    assert recovery["unrecovered_required_count"] == 0
    assert recovery["status"] == "satisfied"


def test_rollout_contains_strategy_budget_projection_without_breaking_old_inputs():
    rollout = build_research_rollout(_episode())
    summary = rollout["strategy_budget_summary"]
    assert summary["policy_id"] == "xscientist.research-policy-budget.v1"
    assert summary["budget_accounting"]["status"] == "complete"
    assert summary["strategy_budget_hash"].startswith("sha256:")


def test_old_tool_delegations_may_omit_optional_budget_fields():
    rollout = build_research_rollout(
        _episode(
            tool_delegations=[
                {
                    "call_id": "legacy-call",
                    "role": "coding_executor",
                    "tool": "codex",
                    "model": "executor-v1",
                    "outcome": "success",
                    "output_hash": HASH,
                }
            ]
        )
    )
    call = rollout["tool_delegations"][0]
    assert call["budget_before_seconds"] is None
    assert call["budget_after_seconds"] is None
    assert rollout["strategy_budget_summary"]["budget_accounting"]["status"] == (
        "partial"
    )


def _independence_receipt(
    evaluator_id: str = "judge-independent",
    target_hash: str = HASH,
    *,
    signed: bool = True,
    producer_actor_ids=None,
):
    identity = f"agent:{evaluator_id}"
    producers = sorted(producer_actor_ids or PRODUCER_ACTOR_IDS)
    if signed:
        binding = build_independence_attestation_payload(
            evaluator_id=evaluator_id,
            evaluator_identity=identity,
            target_hashes=[target_hash],
            producer_actor_ids=producers,
        )
        attestation = sign_attestation(
            binding,
            purpose="research_rollout_independent_evaluation",
            identity=identity,
            key_id=SIGNING_KEY_ID,
            key=SIGNING_SECRET.encode("utf-8"),
            issued_at="2026-08-28T00:00:00+00:00",
        )
        return build_independence_receipt(
            evaluator_id=evaluator_id,
            evaluator_identity=identity,
            target_hashes=[target_hash],
            producer_actor_ids=producers,
            attestation=attestation,
        )
    core = {
        "policy": "xscientist.provenance-actor-disjoint.v1",
        "assurance": "declared_actor_disjointness",
        "identity_verified": True,
        "evaluator_id": evaluator_id,
        "target_hashes": [target_hash],
        "producer_actor_ids": producers,
    }
    return {**core, "receipt_hash": canonical_content_hash(core)}


def _trust_store():
    return {
        SIGNING_KEY_ID: {
            "identity": EVALUATOR_IDENTITY,
            "algorithm": "hmac-sha256",
            "key": SIGNING_SECRET,
        }
    }


def _signed_episode(
    *,
    artifact_hash: str = HASH,
    provider: str | None = None,
    model: str = "executor-v1",
    tool_delegations=None,
):
    calls = tool_delegations
    if calls is None:
        calls = [
            {
                "call_id": "call-1",
                "role": "coding_executor",
                "tool": "codex",
                "provider": provider,
                "model": model,
                "decision": "execute",
                "outcome": "success",
                "input_hash": HASH,
                "output_hash": artifact_hash,
                "budget_before_seconds": 3600,
                "budget_after_seconds": 3000,
            }
        ]
    episode = _episode(tool_delegations=calls, evaluations=[])
    draft = build_research_rollout(episode)
    producers = rollout_producer_actor_ids(draft)
    episode["evaluations"] = [
        {
            "sample_id": "sample-1",
            "evaluator_id": "judge-independent",
            "authority": "independent_evaluator",
            "scores": DIMENSIONS,
            "evidence_refs": [artifact_hash],
            "independence_receipt": _independence_receipt(
                target_hash=artifact_hash,
                producer_actor_ids=producers,
            ),
        }
    ]
    return episode


def _ready_rollout(**kwargs):
    return build_research_rollout(_signed_episode(**kwargs))


@pytest.mark.parametrize(
    ("calls", "expected_status"),
    [
        (
            [
                {
                    "call_id": "partial",
                    "role": "coding_executor",
                    "tool": "codex",
                    "model": "executor-v1",
                    "decision": "execute",
                    "outcome": "success",
                    "output_hash": HASH,
                }
            ],
            "partial",
        ),
        (
            [
                {
                    "call_id": "first",
                    "role": "coding_executor",
                    "tool": "codex",
                    "model": "executor-v1",
                    "decision": "execute",
                    "outcome": "success",
                    "output_hash": HASH,
                    "budget_before_seconds": 3600,
                    "budget_after_seconds": 3300,
                },
                {
                    "call_id": "second",
                    "role": "coding_executor",
                    "tool": "codex",
                    "model": "executor-v1",
                    "decision": "execute",
                    "outcome": "success",
                    "output_hash": HASH,
                    "budget_before_seconds": 3200,
                    "budget_after_seconds": 3000,
                },
            ],
            "inconsistent",
        ),
    ],
)
def test_completed_rollout_audit_rejects_incomplete_budget_accounting(
    calls, expected_status
):
    rollout = _ready_rollout(tool_delegations=calls)
    assert rollout["strategy_budget_summary"]["budget_accounting"]["status"] == (
        expected_status
    )

    audit = audit_research_rollout(
        rollout,
        evidence_hashes=[HASH],
        trust_store=_trust_store(),
    )
    assert audit["verification_allowed"] is False
    assert audit["checks"]["strategy_budget_accounting_complete"] is False
    assert "completed_with_incomplete_budget_accounting" in {
        item["code"] for item in audit["blockers"]
    }
    validate(audit, load_schema("research_rollout_audit"))


def test_failed_repair_or_delegate_does_not_count_as_recovery():
    calls = [
        {
            "call_id": "failed-execution",
            "role": "coding_executor",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "execute",
            "outcome": "failed",
            "follow_up_required": True,
            "budget_before_seconds": 3600,
            "budget_after_seconds": 3400,
        },
        {
            "call_id": "failed-repair",
            "role": "research_policy",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "repair",
            "outcome": "failed",
            "budget_before_seconds": 3400,
            "budget_after_seconds": 3300,
        },
        {
            "call_id": "failed-delegation",
            "role": "research_policy",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "delegate",
            "outcome": "failed",
            "budget_before_seconds": 3300,
            "budget_after_seconds": 3200,
        },
        {
            "call_id": "unrelated-success",
            "role": "coding_executor",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "execute",
            "outcome": "success",
            "output_hash": HASH,
            "budget_before_seconds": 3200,
            "budget_after_seconds": 3000,
        },
    ]
    rollout = _ready_rollout(tool_delegations=calls)
    recovery = rollout["strategy_budget_summary"]["failure_recovery"]
    assert recovery["recovered_count"] == 0
    assert recovery["failed_recovery_count"] >= 2
    assert recovery["unrecovered_required_count"] == 1
    assert recovery["status"] == "needs_recovery"

    audit = audit_research_rollout(
        rollout,
        evidence_hashes=[HASH],
        trust_store=_trust_store(),
    )
    assert audit["checks"]["failure_recovery_satisfied"] is False
    assert "completed_with_unrecovered_required_failure" in {
        item["code"] for item in audit["blockers"]
    }


def test_stop_only_resolves_failure_when_it_is_terminal():
    calls = [
        {
            "call_id": "failed",
            "role": "coding_executor",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "execute",
            "outcome": "failed",
            "follow_up_required": True,
            "budget_before_seconds": 3600,
            "budget_after_seconds": 3400,
        },
        {
            "call_id": "premature-stop",
            "role": "research_policy",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "stop",
            "outcome": "skipped",
            "budget_before_seconds": 3400,
            "budget_after_seconds": 3400,
        },
        {
            "call_id": "continued-execution",
            "role": "coding_executor",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "execute",
            "outcome": "success",
            "output_hash": HASH,
            "budget_before_seconds": 3400,
            "budget_after_seconds": 3000,
        },
    ]
    rollout = _ready_rollout(tool_delegations=calls)
    recovery = rollout["strategy_budget_summary"]["failure_recovery"]
    assert recovery["stopped_count"] == 0
    assert recovery["non_terminal_stop_count"] == 1
    assert recovery["unrecovered_required_count"] == 1
    assert recovery["status"] == "needs_recovery"

    audit = audit_research_rollout(
        rollout,
        evidence_hashes=[HASH],
        trust_store=_trust_store(),
    )
    assert audit["verification_allowed"] is False
    assert audit["checks"]["failure_recovery_satisfied"] is False
    assert "completed_with_non_terminal_stop" in {
        item["code"] for item in audit["blockers"]
    }


def test_non_terminal_stop_is_blocked_even_without_a_prior_failure():
    calls = [
        {
            "call_id": "premature-stop",
            "role": "research_policy",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "stop",
            "outcome": "success",
            "output_hash": HASH,
            "budget_before_seconds": 3600,
            "budget_after_seconds": 3500,
        },
        {
            "call_id": "continued-execution",
            "role": "coding_executor",
            "tool": "codex",
            "model": "executor-v1",
            "decision": "execute",
            "outcome": "success",
            "output_hash": HASH,
            "budget_before_seconds": 3500,
            "budget_after_seconds": 3000,
        },
    ]
    rollout = _ready_rollout(tool_delegations=calls)
    recovery = rollout["strategy_budget_summary"]["failure_recovery"]
    assert recovery["failure_count"] == 0
    assert recovery["non_terminal_stop_count"] == 1
    assert recovery["status"] == "needs_recovery"

    audit = audit_research_rollout(
        rollout,
        evidence_hashes=[HASH],
        trust_store=_trust_store(),
    )
    assert audit["verification_allowed"] is False
    assert audit["checks"]["failure_recovery_satisfied"] is False
    assert "completed_with_non_terminal_stop" in {
        item["code"] for item in audit["blockers"]
    }


def test_strict_tool_swap_uses_supplied_audit_resolver_and_trust_store():
    alternate_hash = "sha256:" + "b" * 64
    reference = _ready_rollout()
    candidate = _ready_rollout(
        artifact_hash=alternate_hash,
        model="executor-v2",
    )

    missing_inputs = assess_tool_swap_compatibility(
        reference,
        candidate,
        strict=True,
    )
    assert missing_inputs["eligible"] is False
    assert {
        "reference_rollout_not_verification_ready",
        "candidate_rollout_not_verification_ready",
    }.issubset(missing_inputs["reasons"])

    supplied_inputs = assess_tool_swap_compatibility(
        reference,
        candidate,
        strict=True,
        audit_evidence_hashes=[HASH, alternate_hash],
        audit_trust_store=_trust_store(),
    )
    assert supplied_inputs["eligible"] is True, supplied_inputs
    assert supplied_inputs["reasons"] == []
    assert supplied_inputs["strict"] is True


def test_rollout_audit_bounds_blockers_and_warnings_to_schema_limits():
    blockers = [
        {
            "code": f"blocker-{index:03d}",
            "field": f"rollout.blockers[{index}]",
            "message": "blocked",
        }
        for index in range(80)
    ]
    warnings = [
        {
            "code": f"warning-{index:03d}",
            "field": f"rollout.warnings[{index}]",
            "message": "warning",
        }
        for index in range(80)
    ]
    audit = _rollout_audit_result(
        {"task_id": "bounded-audit", "task_hash": HASH, "outcome": "failed"},
        {"schema_valid": False},
        blockers,
        warnings,
    )
    assert len(audit["blockers"]) == 64
    assert len(audit["warnings"]) == 64
    assert "blockers_truncated" in {item["code"] for item in audit["blockers"]}
    assert "warnings_truncated" in {item["code"] for item in audit["warnings"]}
    validate(audit, load_schema("research_rollout_audit"))


def test_malformed_rollout_identity_fields_are_not_echoed_by_audit():
    dirty_task_id = "x" * 1000
    dirty_task_hash = "not-a-hash-" + "y" * 1000
    dirty_outcome = "completed-" + "z" * 1000
    audit = audit_research_rollout(
        {
            "task_id": dirty_task_id,
            "task_hash": dirty_task_hash,
            "outcome": dirty_outcome,
        }
    )
    assert audit["task_id"] == ""
    assert audit["task_hash"] == ""
    assert audit["outcome"] == ""
    rendered = json.dumps(audit)
    assert dirty_task_id not in rendered
    assert dirty_task_hash not in rendered
    assert dirty_outcome not in rendered
    validate(audit, load_schema("research_rollout_audit"))


def test_display_provider_and_model_names_have_stable_signable_actor_ids():
    episode = _signed_episode(provider="Open AI", model="GPT 5.5 Codex")
    rollout = build_research_rollout(episode)
    actor_ids = rollout_producer_actor_ids(rollout)
    assert actor_ids == rollout_producer_actor_ids(rollout)
    assert len([item for item in actor_ids if item.startswith("sha256:")]) >= 2
    assert all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", item) for item in actor_ids
    )
    receipt = rollout["evaluation"]["evaluations"][0]["independence_receipt"]
    assert receipt["producer_actor_ids"] == actor_ids

    audit = audit_research_rollout(
        rollout,
        evidence_hashes=[HASH],
        trust_store=_trust_store(),
    )
    assert audit["verification_allowed"] is True, audit


def test_rollout_audit_requires_independent_artifact_bound_evaluation():
    incomplete = build_research_rollout(
        _episode(
            evaluations=[
                {
                    "sample_id": "sample-1",
                    "evaluator_id": "judge-independent",
                    "authority": "independent_evaluator",
                    "scores": DIMENSIONS,
                    "evidence_refs": [HASH],
                }
            ]
        )
    )
    blocked = audit_research_rollout(incomplete, evidence_hashes=[HASH])
    assert blocked["verification_allowed"] is False
    assert "completed_without_independent_evaluator" in {
        item["code"] for item in blocked["blockers"]
    }

    declared_only = build_research_rollout(
        _episode(
            evaluations=[
                {
                    "sample_id": "sample-1",
                    "evaluator_id": "judge-independent",
                    "authority": "independent_evaluator",
                    "scores": DIMENSIONS,
                    "evidence_refs": [HASH],
                    "independence_receipt": _independence_receipt(signed=False),
                }
            ]
        )
    )
    declared_audit = audit_research_rollout(
        declared_only,
        evidence_hashes=[HASH],
        trust_store=_trust_store(),
    )
    assert declared_audit["verification_allowed"] is False
    assert declared_audit["checks"]["independent_evaluator_attestation_valid"] is False

    ready_rollout = build_research_rollout(
        _episode(
            evaluations=[
                {
                    "sample_id": "sample-1",
                    "evaluator_id": "judge-independent",
                    "authority": "independent_evaluator",
                    "scores": DIMENSIONS,
                    "evidence_refs": [HASH],
                    "independence_receipt": _independence_receipt(),
                }
            ]
        )
    )
    ready = audit_research_rollout(
        ready_rollout,
        evidence_hashes=[HASH],
        trust_store=_trust_store(),
    )
    assert ready["verification_allowed"] is True, ready
    assert ready["checks"]["executor_artifact_bound"] is True
    assert ready["checks"]["strategy_budget_summary_valid"] is True


def test_rollout_audit_detects_strategy_or_executor_evidence_tampering():
    rollout = build_research_rollout(
        _episode(
            evaluations=[
                {
                    "sample_id": "sample-1",
                    "evaluator_id": "judge-independent",
                    "authority": "independent_evaluator",
                    "scores": DIMENSIONS,
                    "evidence_refs": [HASH],
                    "independence_receipt": _independence_receipt(),
                }
            ]
        )
    )
    rollout["strategy_budget_summary"]["next_action"] = "stop"
    audit = audit_research_rollout(rollout, evidence_hashes=[HASH])
    assert audit["verification_allowed"] is False
    assert "strategy_budget_summary_mismatch" in {
        item["code"] for item in audit["blockers"]
    }

    untouched = build_research_rollout(
        _episode(
            evaluations=[
                {
                    "sample_id": "sample-1",
                    "evaluator_id": "judge-independent",
                    "authority": "independent_evaluator",
                    "scores": DIMENSIONS,
                    "evidence_refs": ["sha256:" + "b" * 64],
                    "independence_receipt": _independence_receipt(
                        target_hash="sha256:" + "b" * 64
                    ),
                }
            ]
        )
    )
    # The builder accepts a structurally valid evaluator record, but the audit
    # must bind it to the successful executor output rather than any arbitrary
    # content-addressed value.
    result = audit_research_rollout(untouched)
    assert result["verification_allowed"] is False
    assert "completed_executor_artifact_not_evaluated" in {
        item["code"] for item in result["blockers"]
    }


def test_rollout_audit_cli_accepts_raw_payload_and_returns_gate_status(
    tmp_path, capsys
):
    rollout = build_research_rollout(
        _episode(
            evaluations=[
                {
                    "sample_id": "sample-1",
                    "evaluator_id": "judge-independent",
                    "authority": "independent_evaluator",
                    "scores": DIMENSIONS,
                    "evidence_refs": [HASH],
                    "independence_receipt": _independence_receipt(),
                }
            ]
        )
    )
    report_path = tmp_path / "rollout.json"
    report_path.write_text(json.dumps(rollout), encoding="utf-8")
    trust_path = tmp_path / "trust-store.json"
    trust_path.write_text(json.dumps(_trust_store()), encoding="utf-8")
    assert (
        research_main(
            [
                "rollout-audit",
                str(report_path),
                "--evidence-hash",
                HASH,
                "--trust-store",
                str(trust_path),
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["verification_allowed"] is True
    assert output["payloads_disclosed"] is False
    validate(output, load_schema("research_rollout_audit"))


def test_rollout_json_wrapper_can_be_read_back_by_rollout_audit(tmp_path, capsys):
    repository = ResearchRepository.init(
        tmp_path / "research",
        name="rollout-wrapper-test",
        question="# Wrapper round trip\n",
        git_user_name="Rollout Test",
        git_user_email="rollout@example.invalid",
    )
    episode_path = tmp_path / "episode.json"
    episode_path.write_text(json.dumps(_signed_episode()), encoding="utf-8")

    assert (
        research_main(
            [
                "rollout",
                str(episode_path),
                "--repo",
                str(repository.path),
                "--json",
            ]
        )
        == 0
    )
    wrapper = json.loads(capsys.readouterr().out)
    assert wrapper["rollout"]["rollout_hash"].startswith("sha256:")

    wrapper_path = tmp_path / "saved-rollout.json"
    wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
    trust_path = tmp_path / "trust-store.json"
    trust_path.write_text(json.dumps(_trust_store()), encoding="utf-8")
    assert (
        research_main(
            [
                "rollout-audit",
                str(wrapper_path),
                "--evidence-hash",
                HASH,
                "--trust-store",
                str(trust_path),
                "--json",
            ]
        )
        == 0
    )
    audit = json.loads(capsys.readouterr().out)
    assert audit["verification_allowed"] is True, audit
    validate(audit, load_schema("research_rollout_audit"))
