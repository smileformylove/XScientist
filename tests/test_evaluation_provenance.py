from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_scientist.protocol import hash_node_payload
from ai_scientist.treesearch.journal import Node
from ai_scientist.treesearch.utils.metric import MetricValue
from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.deterministic_evaluator import (
    evaluate_experiment_data,
    evaluation_hash_binding,
)


class EvaluationProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _report(self) -> dict:
        data_path = self.tmp / "experiment_data.npy"
        np.save(data_path, {"test": {"y_true": [0, 1], "y_pred": [0, 1]}})
        return evaluate_experiment_data(data_path, requested_metric="accuracy")

    def test_node_round_trip_preserves_evaluation_metadata(self) -> None:
        report = self._report()
        node = Node(
            code="print('ok')",
            metric=MetricValue(value=report["metric"]),
            metric_provenance="deterministic_verified",
            evaluation_report=report,
        )
        restored = Node.from_dict(node.to_dict())
        self.assertEqual(restored.metric_provenance, "deterministic_verified")
        self.assertEqual(restored.evaluation_report, report)
        self.assertEqual(restored.metric.value, report["metric"])

    def test_ara_export_binds_and_surfaces_verified_evaluation(self) -> None:
        report = self._report()
        metric = {"value": report["metric"], "maximize": None, "name": None}
        project = self.tmp / "project"
        exp = project / "02_experiments" / "run"
        journal_dir = exp / "logs" / "0-run"
        journal_dir.mkdir(parents=True)
        (exp / "idea.json").write_text(json.dumps({"Name": "eval"}))
        (journal_dir / "journal.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "n1",
                            "step": 0,
                            "code": "print('ok')",
                            "_term_out": [],
                            "metric": metric,
                            "metric_provenance": "deterministic_verified",
                            "evaluation_report": report,
                            "is_buggy": False,
                            "parent_id": None,
                            "children": [],
                        }
                    ],
                    "node2parent": {},
                    "__version": "2",
                }
            )
        )

        result = export_ara(project_dir=project, exp_dir=exp, idea={"Name": "eval"})
        metrics = json.loads(
            (result.root / "nodes" / "n1" / "metrics.json").read_text()
        )
        graph = json.loads((result.root / "exploration_graph.json").read_text())
        [graph_node] = graph["nodes"]
        binding = evaluation_hash_binding(report)
        expected = hash_node_payload(
            code="print('ok')",
            metric=metric,
            extras={"evaluation": binding},
        )

        self.assertEqual(metrics["content_hash"], expected)
        self.assertIn("evaluation", metrics["content_hash_inputs"])
        self.assertEqual(metrics["metric_provenance"], "deterministic_verified")
        self.assertEqual(metrics["evaluation_report"], report)
        self.assertTrue((result.root / "nodes" / "n1" / "evaluation.json").exists())
        self.assertEqual(graph_node["content_hash"], expected)
        self.assertIn("evaluation", graph_node["content_hash_inputs"])
        self.assertEqual(graph_node["metric_provenance"], "deterministic_verified")

    def test_unverified_report_does_not_change_legacy_hash(self) -> None:
        report = {
            "schema_version": "deterministic_evaluation.v1",
            "status": "unsupported",
            "trust_tier": "unverified",
        }
        self.assertIsNone(evaluation_hash_binding(report))
        baseline = hash_node_payload(code="x = 1", metric={"value": 1})
        with_no_binding = hash_node_payload(
            code="x = 1", metric={"value": 1}, extras=None
        )
        self.assertEqual(baseline, with_no_binding)

    def test_tampered_report_is_not_accepted_as_verified(self) -> None:
        report = self._report()
        report["sample_count"] = 999
        self.assertIsNone(evaluation_hash_binding(report))

    def test_evaluation_hashes_change_node_identity(self) -> None:
        first = self._report()
        data_path = self.tmp / "experiment_data.npy"
        np.save(data_path, {"test": {"y_true": [0, 1], "y_pred": [0, 0]}})
        second = evaluate_experiment_data(data_path, requested_metric="accuracy")
        first_hash = hash_node_payload(
            code="print('ok')",
            metric={"value": first["metric"]},
            extras={"evaluation": evaluation_hash_binding(first)},
        )
        second_hash = hash_node_payload(
            code="print('ok')",
            metric={"value": second["metric"]},
            extras={"evaluation": evaluation_hash_binding(second)},
        )
        self.assertNotEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
