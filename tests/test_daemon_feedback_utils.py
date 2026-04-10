from __future__ import annotations

import unittest

from ai_scientist.utils.daemon_feedback import (
    build_active_source_feedback_snapshot,
    get_active_source_feedback,
)


class DaemonFeedbackUtilsTests(unittest.TestCase):
    def test_get_active_source_feedback_prefers_source_name(self) -> None:
        status = {
            "active_source": {"name": "topic_a"},
            "active_source_key": "topic::topic_a::topic.md",
            "source_quality_feedback": {
                "topic_a": {"avg_experiment_todo": 2.0},
                "topic::topic_a::topic.md": {"avg_experiment_todo": 3.0},
            },
        }
        feedback = get_active_source_feedback(status)
        self.assertEqual(feedback.get("avg_experiment_todo"), 2.0)

    def test_build_active_source_feedback_snapshot_shape(self) -> None:
        status = {
            "active_source": {"name": "topic_a"},
            "source_quality_feedback": {
                "topic_a": {
                    "source_name": "topic_a",
                    "avg_experiment_todo": 1.5,
                    "avg_experiment_todo_p0": 0.4,
                    "avg_experiment_todo_closure_rate": 0.7,
                    "experiment_todo_pressure_rate": 0.5,
                    "self_review_gate_ready_rate": 0.6,
                    "extra_field_should_not_leak": 1,
                }
            },
        }
        snapshot = build_active_source_feedback_snapshot(status)
        self.assertEqual(snapshot.get("source_name"), "topic_a")
        self.assertEqual(snapshot.get("avg_experiment_todo"), 1.5)
        self.assertEqual(snapshot.get("avg_experiment_todo_p0"), 0.4)
        self.assertEqual(snapshot.get("avg_experiment_todo_closure_rate"), 0.7)
        self.assertEqual(snapshot.get("experiment_todo_pressure_rate"), 0.5)
        self.assertEqual(snapshot.get("self_review_gate_ready_rate"), 0.6)
        self.assertNotIn("extra_field_should_not_leak", snapshot)


if __name__ == "__main__":
    unittest.main()
