from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from ai_scientist.utils.deterministic_evaluator import (
    _array_shape_from_build_state,
    _computed_input_fingerprints,
    evaluate_experiment_data,
    evaluation_hash_binding,
)
from ai_scientist.utils.evaluation_binding import evaluation_comparison_contract


def _metric_value(report: dict) -> float:
    return report["metric"]["metric_names"][0]["data"][0]["final_value"]


class DeterministicEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "experiment_data.npy"

    def test_nested_classification_accuracy_is_verified(self) -> None:
        np.save(
            self.path,
            {
                "validation": {
                    "outputs": {
                        "ground_truth": np.array([0, 1, 1, 0]),
                        "predictions": np.array([0, 1, 0, 0]),
                    }
                }
            },
        )
        report = evaluate_experiment_data(
            self.path, requested_metric="validation accuracy"
        )
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["trust_tier"], "deterministic_verified")
        self.assertTrue(report["safe_for_legacy_parser"])
        self.assertEqual(_metric_value(report), 0.75)
        self.assertEqual(report["datasets"][0]["dataset_name"], "validation/outputs")
        self.assertEqual(report["sample_count"], 4)
        binding = evaluation_hash_binding(report)
        self.assertEqual(binding["input_hash"], report["input"]["sha256"])
        self.assertEqual(report["verification_scope"], "artifact_internal_consistency")
        self.assertEqual(report["ground_truth_authority"], "research_agent_artifact")

    def test_comparison_identity_is_computed_from_actual_evaluation_inputs(
        self,
    ) -> None:
        first_inputs = [[1.0, 2.0], [3.0, 4.0]]
        np.save(
            self.path,
            {
                "test": {
                    "evaluation_inputs": first_inputs,
                    "sample_ids": ["a", "b"],
                    "y_true": [0, 1],
                    "y_pred": [0, 1],
                }
            },
        )
        first = evaluate_experiment_data(self.path, requested_metric="accuracy")
        first_contract = evaluation_comparison_contract(first)
        self.assertIsNotNone(first_contract)

        np.save(
            self.path,
            {
                "test": {
                    "evaluation_inputs": [[9.0, 8.0], [7.0, 6.0]],
                    "sample_ids": ["a", "b"],
                    "y_true": [0, 1],
                    "y_pred": [0, 1],
                }
            },
        )
        changed = evaluate_experiment_data(self.path, requested_metric="accuracy")
        self.assertNotEqual(
            first_contract,
            evaluation_comparison_contract(changed),
        )

        copied_assertions = _computed_input_fingerprints(
            first_inputs,
            samples=2,
            dataset="test",
        )
        np.save(
            self.path,
            {
                "test": {
                    "evaluation_inputs": [[9.0, 8.0], [7.0, 6.0]],
                    "input_fingerprints": copied_assertions,
                    "sample_ids": ["a", "b"],
                    "y_true": [0, 1],
                    "y_pred": [0, 1],
                }
            },
        )
        forged = evaluate_experiment_data(self.path, requested_metric="accuracy")
        self.assertEqual(forged["status"], "invalid")
        self.assertIn("do not match evaluator-computed", forged["reason"])

    def test_regression_metrics_are_computed_directly(self) -> None:
        np.save(
            self.path,
            {"test": {"y_true": [1.0, 2.0, 3.0], "y_pred": [1.0, 2.5, 2.0]}},
        )
        expected = {
            "mse": 1.25 / 3.0,
            "rmse": np.sqrt(1.25 / 3.0),
            "mae": 0.5,
            "r2": 0.375,
        }
        for metric, value in expected.items():
            with self.subTest(metric=metric):
                report = evaluate_experiment_data(self.path, requested_metric=metric)
                self.assertEqual(report["status"], "verified")
                self.assertAlmostEqual(_metric_value(report), value)

    def test_binary_metrics_and_auc_are_supported(self) -> None:
        np.save(
            self.path,
            {
                "test": {
                    "labels": np.array([0, 0, 1, 1]),
                    "predicted": np.array([0, 1, 1, 1]),
                    "scores": np.array([0.1, 0.4, 0.35, 0.8]),
                }
            },
        )
        expected = {
            "precision": 2 / 3,
            "recall": 1.0,
            "f1": 0.8,
            "roc auc": 0.75,
        }
        for metric, value in expected.items():
            with self.subTest(metric=metric):
                report = evaluate_experiment_data(self.path, requested_metric=metric)
                self.assertEqual(report["status"], "verified")
                self.assertAlmostEqual(_metric_value(report), value)

    def test_auc_requires_scores_instead_of_treating_labels_as_scores(self) -> None:
        np.save(self.path, {"y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 1]})
        report = evaluate_experiment_data(self.path, requested_metric="roc auc")
        self.assertEqual(report["status"], "invalid")
        self.assertIn("requires probability scores", report["reason"])

    def test_unsupported_metric_does_not_claim_verification(self) -> None:
        np.save(self.path, {"perplexity": 12.0})
        report = evaluate_experiment_data(self.path, requested_metric="perplexity")
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(report["trust_tier"], "unverified")
        self.assertTrue(report["safe_for_legacy_parser"])
        self.assertIsNone(report["metric"])
        self.assertIsNone(evaluation_hash_binding(report))

    def test_short_metric_alias_does_not_match_inside_another_word(self) -> None:
        np.save(self.path, {"y_true": [0, 1], "y_pred": [0, 1]})
        report = evaluate_experiment_data(
            self.path, requested_metric="Jaccard similarity"
        )
        self.assertEqual(report["status"], "unsupported")

    def test_ambiguous_mismatched_and_nonfinite_inputs_are_rejected(self) -> None:
        cases = [
            {
                "y_true": [0, 1],
                "labels": [0, 1],
                "predictions": [0, 1],
            },
            {"y_true": [0, 1], "predictions": [0]},
            {"y_true": [1.0, np.nan], "predictions": [1.0, 2.0]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                np.save(self.path, payload)
                report = evaluate_experiment_data(
                    self.path, requested_metric="accuracy"
                )
                self.assertEqual(report["status"], "invalid")
                self.assertIsNone(report["metric"])

    def test_ambiguous_positive_label_is_rejected(self) -> None:
        np.save(
            self.path,
            {"y_true": ["cat", "dog"], "y_pred": ["cat", "dog"]},
        )
        report = evaluate_experiment_data(self.path, requested_metric="f1")
        self.assertEqual(report["status"], "invalid")
        self.assertIn("positive class is ambiguous", report["reason"])

    def test_accuracy_rejects_continuous_regression_values(self) -> None:
        np.save(self.path, {"y_true": [0.1, 0.2], "y_pred": [0.1, 0.2]})
        report = evaluate_experiment_data(self.path, requested_metric="accuracy")
        self.assertEqual(report["status"], "invalid")
        self.assertFalse(report["safe_for_legacy_parser"])
        self.assertIn("continuous values", report["reason"])

    def test_restricted_unpickler_rejects_code_execution(self) -> None:
        marker = Path(self._tmp.name) / "executed"

        class Payload:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        np.save(self.path, {"payload": Payload()})
        report = evaluate_experiment_data(self.path, requested_metric="accuracy")
        self.assertEqual(report["status"], "invalid")
        self.assertIn("unsafe pickle global rejected", report["reason"])
        self.assertFalse(report["safe_for_legacy_parser"])
        self.assertFalse(marker.exists())

    def test_size_depth_and_array_limits_fail_closed(self) -> None:
        np.save(self.path, {"y_true": [0, 1], "y_pred": [0, 1]})
        too_small = evaluate_experiment_data(
            self.path, requested_metric="accuracy", max_file_bytes=16
        )
        self.assertEqual(too_small["status"], "invalid")

        nested: dict = {}
        cursor = nested
        for _ in range(8):
            cursor["next"] = {}
            cursor = cursor["next"]
        np.save(self.path, nested)
        too_deep = evaluate_experiment_data(
            self.path, requested_metric="accuracy", max_depth=3
        )
        self.assertEqual(too_deep["status"], "invalid")

        np.save(self.path, {"y_true": np.arange(10), "y_pred": np.arange(10)})
        too_many = evaluate_experiment_data(
            self.path, requested_metric="accuracy", max_array_elements=5
        )
        self.assertEqual(too_many["status"], "invalid")

    def test_declared_pickle_shape_is_checked_before_allocation(self) -> None:
        state = (1, (10_000_000,), np.dtype("float64"), False, b"")
        self.assertEqual(_array_shape_from_build_state(state), (10_000_000,))

    def test_input_and_result_hashes_change_with_data(self) -> None:
        np.save(self.path, {"y_true": [0, 1], "y_pred": [0, 1]})
        first = evaluate_experiment_data(self.path, requested_metric="accuracy")
        np.save(self.path, {"y_true": [0, 1], "y_pred": [0, 0]})
        second = evaluate_experiment_data(self.path, requested_metric="accuracy")
        self.assertNotEqual(first["input"]["sha256"], second["input"]["sha256"])
        self.assertNotEqual(first["result_hash"], second["result_hash"])

    def test_missing_artifact_is_explicit(self) -> None:
        report = evaluate_experiment_data(self.path, requested_metric="accuracy")
        self.assertEqual(report["status"], "missing")
        self.assertEqual(report["trust_tier"], "unverified")

    def test_symlink_artifact_is_rejected(self) -> None:
        target = Path(self._tmp.name) / "target.npy"
        np.save(target, {"y_true": [0, 1], "y_pred": [0, 1]})
        self.path.symlink_to(target)
        report = evaluate_experiment_data(self.path, requested_metric="accuracy")
        self.assertEqual(report["status"], "invalid")
        self.assertIn("symbolic-link", report["reason"])


if __name__ == "__main__":
    unittest.main()
