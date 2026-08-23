from __future__ import annotations

import json

import pytest

from xscientist import (
    ResearchRolloutError,
    assess_tool_swap_compatibility,
    build_replication_rubric,
    build_research_rollout,
    build_tool_delegation_trace,
    build_turn_credit_summary,
    evaluate_replication_rollout,
)

HASH = "sha256:" + "a" * 64
DIMENSIONS = {
    "result_fidelity": 0.8,
    "claim_support": 0.7,
    "implementation_fidelity": 0.9,
    "resource_efficiency": 0.6,
    "scientific_integrity": 1.0,
}


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
