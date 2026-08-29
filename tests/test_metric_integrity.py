from __future__ import annotations

import json
import hashlib
from unittest import mock

import pytest
import numpy as np

from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.backend import (
    FunctionCallValidationError,
    ResearchDecisionError,
)
from ai_scientist.treesearch.backend.utils import validate_function_call_payload
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.parallel_agent import metric_parse_spec
from ai_scientist.treesearch.utils.metric import MetricValue
from ai_scientist.utils.deterministic_evaluator import evaluate_experiment_data


def _node(
    value: float,
    *,
    provenance: str,
    node_id: str,
) -> Node:
    report = {
        "schema_version": "deterministic_evaluation.v1",
        "evaluator_version": "test",
        "evaluator_hash": "sha256:" + "a" * 64,
        "status": "verified",
        "trust_tier": "deterministic_verified",
        "input": {"sha256": "sha256:" + "b" * 64},
        "metric": float(value),
    }
    report["result_hash"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    return Node(
        id=node_id,
        metric=MetricValue(value, maximize=True),
        metric_provenance=provenance,
        evaluation_report=report,
        is_buggy=False,
        is_buggy_plots=False,
    )


@pytest.mark.parametrize(
    "value",
    [
        {"metric_names": []},
        {
            "metric_names": [
                {
                    "metric_name": "accuracy",
                    "lower_is_better": False,
                    "description": "empty",
                    "data": [],
                }
            ]
        },
        {"dataset": float("nan")},
        float("inf"),
    ],
)
def test_metric_value_rejects_empty_or_nonfinite_evidence(value) -> None:
    with pytest.raises(ValueError):
        MetricValue(value=value)


def test_metric_comparison_rejects_incompatible_contracts_and_bad_best_value() -> None:
    accuracy = MetricValue(
        value={
            "metric_names": [
                {
                    "metric_name": "accuracy",
                    "lower_is_better": False,
                    "description": "accuracy",
                    "data": [
                        {
                            "dataset_name": "test",
                            "final_value": 0.8,
                            "best_value": 0.9,
                        }
                    ],
                }
            ]
        }
    )
    f1 = MetricValue(
        value={
            "metric_names": [
                {
                    "metric_name": "f1",
                    "lower_is_better": False,
                    "description": "f1",
                    "data": [
                        {
                            "dataset_name": "test",
                            "final_value": 0.8,
                            "best_value": 0.9,
                        }
                    ],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="incompatible"):
        _ = accuracy > f1

    with pytest.raises(ValueError, match="maximization"):
        MetricValue(
            value={
                "metric_names": [
                    {
                        "metric_name": "accuracy",
                        "lower_is_better": False,
                        "description": "invalid best",
                        "data": [
                            {
                                "dataset_name": "test",
                                "final_value": 0.9,
                                "best_value": 0.8,
                            }
                        ],
                    }
                ]
            }
        )


def test_agent_reported_metric_cannot_win_or_supply_a_best_node() -> None:
    journal = Journal()
    verified = _node(0.5, provenance="deterministic_verified", node_id="verified")
    advisory = _node(
        999.0,
        provenance="agent_reported_advisory",
        node_id="advisory",
    )
    journal.append(verified)
    journal.append(advisory)

    assert journal.get_best_node_by_metric() is verified

    advisory_only = Journal()
    advisory_only.append(advisory)
    assert advisory_only.get_best_node_by_metric() is None


def test_advisory_metrics_are_separate_from_verified_success_reporting() -> None:
    from ai_scientist.treesearch import journal as journal_module

    journal = Journal()
    verified = _node(0.5, provenance="deterministic_verified", node_id="verified")
    verified.plan = "verified-plan"
    advisory = _node(
        999.0,
        provenance="agent_reported_advisory",
        node_id="advisory",
    )
    advisory.plan = "advisory-plan"
    journal.append(verified)
    journal.append(advisory)
    manager = AgentManager.__new__(AgentManager)

    metrics = manager._gather_stage_metrics(journal)
    with mock.patch.object(journal_module, "query", return_value="summary") as query:
        assert journal.generate_summary(model="test/report-model") == "summary"

    prompt = query.call_args.kwargs["system_message"]
    assert metrics["good_nodes"] == 1
    assert metrics["runnable_nodes"] == 2
    assert metrics["advisory_metric_nodes"] == 1
    assert "verified-plan" in prompt["Deterministically Verified Experiments"]
    assert "advisory-plan" not in prompt["Deterministically Verified Experiments"]
    assert "advisory-plan" in prompt["Unverified Runnable Experiments (advisory only)"]


def test_agent_reported_metric_cannot_complete_initial_stage() -> None:
    journal = Journal()
    journal.append(
        _node(
            999.0,
            provenance="agent_reported_advisory",
            node_id="advisory",
        )
    )
    stage = Stage(
        name="1_initial_implementation_1_first_attempt",
        description="initial",
        goals=["working verified implementation"],
        max_iterations=3,
        num_drafts=1,
        stage_number=1,
    )
    manager = AgentManager.__new__(AgentManager)
    manager.journals = {stage.name: journal}

    complete, _reason = manager._check_stage_completion(stage)

    assert complete is False


def test_metric_tool_contract_binds_valid_flag_to_nonempty_evidence() -> None:
    invalid_payloads = (
        {"valid_metrics_received": True, "metric_names": []},
        {
            "valid_metrics_received": False,
            "metric_names": [
                {
                    "metric_name": "accuracy",
                    "lower_is_better": False,
                    "description": "should be absent",
                    "data": [
                        {
                            "dataset_name": "dataset",
                            "final_value": 1.0,
                            "best_value": 1.0,
                        }
                    ],
                }
            ],
        },
    )
    for payload in invalid_payloads:
        with pytest.raises(FunctionCallValidationError):
            validate_function_call_payload(
                metric_parse_spec,
                function_name="parse_metrics",
                arguments=json.dumps(payload),
            )


def test_advisory_metric_round_trip_remains_explicitly_unverified() -> None:
    node = _node(
        0.9,
        provenance="agent_reported_advisory",
        node_id="advisory",
    )
    node.advisory_metric = {
        "metric_names": [
            {
                "metric_name": "accuracy",
                "lower_is_better": False,
                "description": "model extracted",
                "data": [
                    {
                        "dataset_name": "dataset",
                        "final_value": 0.9,
                        "best_value": 0.9,
                    }
                ],
            }
        ]
    }

    restored = Node.from_dict(node.to_dict())

    assert restored.metric_provenance == "agent_reported_advisory"
    assert restored.advisory_metric == node.advisory_metric


def _comparison_node(path, *, inputs, predictions, node_id: str) -> Node:
    np.save(
        path,
        {
            "test": {
                "evaluation_inputs": inputs,
                "sample_ids": ["a", "b"],
                "ground_truth": [0, 1],
                "predictions": predictions,
            }
        },
    )
    report = evaluate_experiment_data(path, requested_metric="accuracy")
    return Node(
        id=node_id,
        metric=MetricValue(report["metric"]),
        metric_provenance="deterministic_verified",
        evaluation_report=report,
        is_buggy=False,
    )


def test_journal_ranks_only_within_exact_evaluation_contract(tmp_path) -> None:
    journal = Journal()
    first = _comparison_node(
        tmp_path / "first.npy",
        inputs=[[1], [2]],
        predictions=[0, 0],
        node_id="first",
    )
    better = _comparison_node(
        tmp_path / "better.npy",
        inputs=[[1], [2]],
        predictions=[0, 1],
        node_id="better",
    )
    journal.append(first)
    journal.append(better)
    assert journal.get_best_node_by_metric() is better

    incompatible = _comparison_node(
        tmp_path / "other.npy",
        inputs=[[9], [8]],
        predictions=[0, 1],
        node_id="other",
    )
    journal.append(incompatible)
    with pytest.raises(ResearchDecisionError, match="dataset identities"):
        journal.get_best_node_by_metric()
    assert (
        journal.get_best_node_by_metric(
            reference_contract=first.evaluation_comparison_contract
        )
        is better
    )
