from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from ai_scientist.treesearch import log_summarization


class _ReportNode:
    def __init__(self, node_id: str, *, seed: bool = False, parent=None) -> None:
        self.id = node_id
        self.is_seed_node = seed
        self.is_seed_agg_node = False
        self.parent = parent
        self.multi_seed_report = (
            {"stage": "qualified", "receipt_hash": "sha256:receipt", "seeds": []}
            if not seed
            else None
        )
        self.evaluation_report = {
            "verification_scope": "artifact_internal_consistency",
            "ground_truth_authority": "research_agent_artifact",
            "result_hash": "sha256:evaluation",
        }
        self.metric_provenance = "deterministic_verified"

    def to_dict(self) -> dict:
        return {
            "overall_plan": "plan",
            "analysis": "analysis",
            "metric": {"value": 1.0},
        }


class _ReportJournal:
    def __init__(self, stage_index: int) -> None:
        self.qualified = _ReportNode(f"qualified-{stage_index}")
        self.seed = _ReportNode(
            f"seed-{stage_index}",
            seed=True,
            parent=self.qualified,
        )
        self.verified_nodes = [self.qualified, self.seed]

    def get_best_node_by_metric(self):
        return self.qualified


def test_final_summary_routes_through_report_model_only() -> None:
    cfg = SimpleNamespace(
        report=SimpleNamespace(model="report/glm-5.3", temp=0.2),
        agent=SimpleNamespace(
            summary=SimpleNamespace(model="wrong/summary-model", temp=1.0)
        ),
    )
    journals = [(f"{index}_stage", _ReportJournal(index)) for index in range(1, 5)]

    with (
        mock.patch.object(
            log_summarization,
            "get_ai_client",
            return_value=object(),
        ) as get_client,
        mock.patch.object(
            log_summarization,
            "get_stage_summary",
            return_value={"summary": "qualified"},
        ) as summarize,
    ):
        result = log_summarization.overall_summarize(journals, cfg)

    get_client.assert_called_once_with("report/glm-5.3")
    assert summarize.call_args.args[2] == "report/glm-5.3"
    assert result[0] == {"summary": "qualified"}
    assert (
        result[1]["qualified node"]["verification"]["verification_scope"]
        == "artifact_internal_consistency"
    )


def test_final_summary_rejects_unqualified_stage_input() -> None:
    cfg = SimpleNamespace(report=SimpleNamespace(model="report/model", temp=0.2))
    journals = [(f"{index}_stage", _ReportJournal(index)) for index in range(1, 5)]
    journals[2][1].qualified.multi_seed_report = None

    with (
        mock.patch.object(log_summarization, "get_ai_client", return_value=object()),
        mock.patch.object(log_summarization, "get_stage_summary", return_value={}),
        pytest.raises(ValueError, match="gate-qualified"),
    ):
        log_summarization.overall_summarize(journals, cfg)
