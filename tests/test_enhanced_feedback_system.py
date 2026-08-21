#!/usr/bin/env python3
"""
Tests for Enhanced Feedback System
"""

import unittest
import tempfile
import time
import os
import json
import subprocess
import sys
from pathlib import Path

from ai_scientist.enhanced_feedback_system import (
    EnhancedFeedbackSystem,
    LongRunningTaskMonitor,
    FeedbackCategory,
    FeedbackPriority,
)


class TestEnhancedFeedbackSystem(unittest.TestCase):
    """Test enhanced feedback system"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_system = EnhancedFeedbackSystem(
            feedback_dir=Path(self.temp_dir),
            window_size=50,
            trend_window_hours=1,
        )

    def test_add_feedback(self):
        """Test adding feedback items"""
        item = self.feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=FeedbackPriority.HIGH,
            source="test",
            message="Test feedback",
            metrics={"score": 3.5},
            context={"test": True},
        )

        self.assertEqual(item.category, "quality")
        self.assertEqual(item.priority, "high")
        self.assertEqual(item.source, "test")
        self.assertTrue(item.actionable)
        self.assertFalse(item.resolved)
        self.assertTrue((Path(self.temp_dir) / "feedback_history.json").is_file())

        reloaded = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))
        self.assertEqual(len(reloaded.feedback_history), 1)
        self.assertEqual(reloaded.feedback_history[0].message, "Test feedback")
        self.assertIn("score", reloaded.metric_windows)

    def test_empty_report_is_unknown_instead_of_perfect(self):
        report = self.feedback_system.get_health_report()

        self.assertFalse(report["has_data"])
        self.assertEqual(report["health_state"], "unknown")
        self.assertIsNone(report["health_score"])

    def test_corrupted_history_is_reported_instead_of_treated_as_empty(self):
        history = Path(self.temp_dir) / "feedback_history.json"
        history.write_text("{not-json", encoding="utf-8")

        feedback = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))
        report = feedback.get_health_report()

        self.assertEqual(report["health_state"], "corrupted")
        self.assertIsNone(report["health_score"])
        self.assertTrue(report["load_errors"])

        project_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        actions = subprocess.run(
            [
                sys.executable,
                "-m",
                "xscientist",
                "feedback",
                "--feedback-dir",
                self.temp_dir,
                "actions",
            ],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(actions.returncode, 1)
        self.assertIn("history is unreadable", actions.stdout)

    def test_cli_feedback_survives_a_new_process(self):
        feedback_dir = Path(self.temp_dir) / "cli-feedback"
        project_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        base = [
            sys.executable,
            "-m",
            "xscientist",
            "feedback",
            "--feedback-dir",
            str(feedback_dir),
        ]

        empty = subprocess.run(
            [*base, "status"],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertIn("Feedback: unknown (no observations yet)", empty.stdout)
        self.assertIn("Items: 0 total, 0 unresolved, 0 critical", empty.stdout)

        empty_json = subprocess.run(
            [*base, "status", "--json"],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(empty_json.returncode, 0, empty_json.stderr)
        empty_payload = json.loads(empty_json.stdout)
        self.assertEqual(empty_payload["schema"], "xscientist.feedback-status.v1")
        self.assertTrue(empty_payload["ok"])
        self.assertEqual(empty_payload["health_state"], "unknown")

        added = subprocess.run(
            [
                *base,
                "add",
                "--category",
                "error",
                "--priority",
                "critical",
                "--source",
                "test",
                "--message",
                "Autonomous run failed",
                "--metrics",
                "error_rate=1",
                "success_rate=0",
            ],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(added.returncode, 0, added.stderr)

        status = subprocess.run(
            [*base, "status"],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn("Items: 1 total, 1 unresolved, 1 critical", status.stdout)
        self.assertNotIn("100.0/100", status.stdout)

        actions_json = subprocess.run(
            [*base, "actions", "--json"],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(actions_json.returncode, 0, actions_json.stderr)
        actions_payload = json.loads(actions_json.stdout)
        self.assertEqual(actions_payload["schema"], "xscientist.feedback-actions.v1")
        self.assertTrue(actions_payload["actions"])

        concurrent = [
            subprocess.Popen(
                [
                    *base,
                    "add",
                    "--category",
                    "performance",
                    "--priority",
                    "info",
                    "--source",
                    "parallel-test",
                    "--message",
                    f"Concurrent observation {index}",
                ],
                cwd=self.temp_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for index in range(4)
        ]
        for process in concurrent:
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr or stdout)
        reloaded = EnhancedFeedbackSystem(feedback_dir=feedback_dir)
        self.assertEqual(len(reloaded.feedback_history), 5)

        attribution_dir = Path(self.temp_dir) / "attribution-feedback"
        attributed = subprocess.run(
            [
                sys.executable,
                "-m",
                "xscientist",
                "feedback",
                "--feedback-dir",
                str(attribution_dir),
                "add",
                "--category",
                "strategy",
                "--priority",
                "info",
                "--source",
                "evolution",
                "--message",
                "Try candidate 42",
                "--intervention-id",
                "candidate-42",
                "--json",
            ],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(attributed.returncode, 0, attributed.stderr)
        attributed_payload = json.loads(attributed.stdout)
        item_id = attributed_payload["item"]["item_id"]
        linked = subprocess.run(
            [
                sys.executable,
                "-m",
                "xscientist",
                "feedback",
                "--feedback-dir",
                str(attribution_dir),
                "link-outcome",
                "--item-id",
                item_id,
                "--outcome-id",
                "outcome-42",
                "--evaluation-scope",
                "independent",
                "--evaluator-id",
                "reviewer-42",
                "--json",
            ],
            cwd=self.temp_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(linked.returncode, 0, linked.stderr)
        linked_payload = json.loads(linked.stdout)
        self.assertEqual(linked_payload["attribution"]["status"], "independent_paired")

    def test_metric_tracking(self):
        """Test metric tracking and windows"""
        # Add multiple metrics
        for i in range(10):
            self.feedback_system.add_feedback(
                category=FeedbackCategory.PERFORMANCE,
                priority=FeedbackPriority.INFO,
                source="test",
                message=f"Metric {i}",
                metrics={"quality_score": 3.0 + i * 0.1},
            )
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Check metric window
        self.assertIn("quality_score", self.feedback_system.metric_windows)
        self.assertEqual(len(self.feedback_system.metric_windows["quality_score"]), 10)

    def test_trend_analysis(self):
        """Test trend analysis"""
        # Add increasing trend
        for i in range(20):
            self.feedback_system.add_feedback(
                category=FeedbackCategory.PERFORMANCE,
                priority=FeedbackPriority.INFO,
                source="test",
                message=f"Metric {i}",
                metrics={"success_rate": 0.5 + i * 0.02},
            )
            time.sleep(0.01)

        trend = self.feedback_system.analyze_trends("success_rate")

        self.assertNotIn("error", trend)
        self.assertEqual(trend["metric"], "success_rate")
        self.assertGreater(trend["data_points"], 0)
        self.assertEqual(trend["trend_direction"], "increasing")
        self.assertGreater(trend["slope"], 0)

    def test_action_generation(self):
        """Test action generation from feedback"""
        # Add critical feedback
        self.feedback_system.add_feedback(
            category=FeedbackCategory.ERROR,
            priority=FeedbackPriority.CRITICAL,
            source="test",
            message="Critical error detected",
            metrics={"error_rate": 0.5},
        )

        # Add low quality metrics
        for i in range(10):
            self.feedback_system.add_feedback(
                category=FeedbackCategory.QUALITY,
                priority=FeedbackPriority.MEDIUM,
                source="test",
                message="Low quality",
                metrics={"quality_score": 2.0},
            )

        actions = self.feedback_system.generate_actions(max_actions=5)

        self.assertGreater(len(actions), 0)
        # Critical or high priority actions should be first
        self.assertIn(actions[0]["priority"], ["critical", "high"])

    def test_health_report(self):
        """Test health report generation"""
        # Add various feedback
        self.feedback_system.add_feedback(
            category=FeedbackCategory.SUCCESS,
            priority=FeedbackPriority.INFO,
            source="test",
            message="Success",
            metrics={"success_rate": 0.8, "quality_score": 4.0},
        )

        report = self.feedback_system.get_health_report()

        self.assertIn("timestamp", report)
        self.assertIn("feedback_summary", report)
        self.assertIn("health_score", report)
        self.assertGreaterEqual(report["health_score"], 0)
        self.assertLessEqual(report["health_score"], 100)
        self.assertEqual(
            report["feedback_summary"]["attribution"]["status"],
            "unattributed",
        )
        self.assertFalse(
            report["feedback_summary"]["attribution"]["causal_attribution_established"]
        )

    def test_feedback_attribution_is_explicit_and_conservative(self):
        observational = self.feedback_system.add_feedback(
            category=FeedbackCategory.STRATEGY,
            priority=FeedbackPriority.INFO,
            source="experiment",
            message="Candidate strategy observed",
            intervention_id="candidate-1",
        )
        self.feedback_system.add_feedback(
            category=FeedbackCategory.SUCCESS,
            priority=FeedbackPriority.INFO,
            source="evaluator",
            message="Outcome measured",
            intervention_id="candidate-1",
            outcome_id="outcome-1",
            evaluation_scope="independent",
            evaluator_id="reviewer-1",
        )

        summary = self.feedback_system.attribution_summary()
        self.assertEqual(summary["status"], "independent_paired")
        self.assertEqual(summary["independent_paired_items"], 1)
        self.assertEqual(summary["intervention_only_items"], 1)
        self.assertFalse(summary["causal_attribution_established"])
        self.assertFalse(summary["promotion_signal_allowed"])
        report = self.feedback_system.get_health_report()
        self.assertEqual(report["feedback_summary"]["attribution"]["paired_items"], 1)
        self.assertEqual(
            observational.evaluation_scope,
            "observational",
        )

    def test_record_outcome_is_persistent_and_monotonic(self):
        item = self.feedback_system.add_feedback(
            category=FeedbackCategory.STRATEGY,
            priority=FeedbackPriority.HIGH,
            source="evolution",
            message="Try candidate configuration",
            intervention_id="candidate-2",
        )

        linked = self.feedback_system.record_outcome(
            item,
            "outcome-2",
            evaluation_scope="independent",
            evaluator_id="reviewer-2",
        )
        self.assertEqual(linked.outcome_id, "outcome-2")
        self.assertEqual(linked.evaluation_scope, "independent")
        with self.assertRaises(ValueError):
            self.feedback_system.record_outcome(item, "different-outcome")

        reloaded = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))
        restored = reloaded.feedback_history[0]
        self.assertEqual(restored.intervention_id, "candidate-2")
        self.assertEqual(restored.outcome_id, "outcome-2")
        self.assertEqual(restored.evaluator_id, "reviewer-2")

    def test_feedback_resolution(self):
        """Test marking feedback as resolved"""
        item = self.feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=FeedbackPriority.HIGH,
            source="test",
            message="Issue",
        )

        self.assertFalse(item.resolved)

        self.feedback_system.mark_resolved(item, "Fixed the issue")

        self.assertTrue(item.resolved)
        self.assertEqual(item.action_taken, "Fixed the issue")

    def test_stale_writer_cannot_resurrect_resolved_feedback(self):
        item = self.feedback_system.add_feedback(
            category=FeedbackCategory.ERROR,
            priority=FeedbackPriority.HIGH,
            source="test",
            message="Concurrent issue",
        )
        stale_writer = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))

        self.feedback_system.mark_resolved(item, "Fixed")
        stale_writer.add_feedback(
            category=FeedbackCategory.SUCCESS,
            priority=FeedbackPriority.INFO,
            source="test",
            message="Later observation",
        )

        reloaded = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))
        original = next(
            feedback
            for feedback in reloaded.feedback_history
            if feedback.item_id == item.item_id
        )
        self.assertTrue(original.resolved)
        self.assertEqual(original.action_taken, "Fixed")

    def test_stale_writer_rebuilds_trends_from_merged_history(self):
        stale_writer = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))
        self.feedback_system.add_feedback(
            category=FeedbackCategory.PERFORMANCE,
            priority=FeedbackPriority.INFO,
            source="first",
            message="First sample",
            metrics={"error_rate": 0.1},
        )
        stale_writer.add_feedback(
            category=FeedbackCategory.PERFORMANCE,
            priority=FeedbackPriority.INFO,
            source="second",
            message="Second sample",
            metrics={"error_rate": 0.9},
        )

        trend = stale_writer.analyze_trends("error_rate")
        self.assertEqual(trend["data_points"], 2)
        self.assertEqual(trend["mean"], 0.5)

    def test_abandoned_feedback_lock_is_recovered(self):
        lock = Path(self.temp_dir) / ".feedback-history.lock"
        lock.mkdir()
        (lock / "owner.json").write_text(
            json.dumps({"pid": 999_999_999, "created_at": 0, "token": "dead"}),
            encoding="utf-8",
        )

        self.feedback_system.add_feedback(
            category=FeedbackCategory.SUCCESS,
            priority=FeedbackPriority.INFO,
            source="test",
            message="Recovered after abandoned lock",
        )

        self.assertFalse(lock.exists())
        self.assertEqual(len(self.feedback_system.feedback_history), 1)

    def test_clear_resolved(self):
        """Test clearing resolved feedback"""
        # Add multiple items
        item1 = self.feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=FeedbackPriority.HIGH,
            source="test",
            message="Issue 1",
        )

        item2 = self.feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=FeedbackPriority.HIGH,
            source="test",
            message="Issue 2",
        )

        # Resolve one
        self.feedback_system.mark_resolved(item1, "Fixed")

        cleared = self.feedback_system.clear_resolved()
        final_count = len(self.feedback_system.feedback_buffer)

        self.assertEqual(cleared, 1)
        self.assertEqual(final_count, 1)
        self.assertEqual(len(self.feedback_system.feedback_history), 1)


class TestLongRunningTaskMonitor(unittest.TestCase):
    """Test long-running task monitor"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_system = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))
        self.monitor = LongRunningTaskMonitor(
            task_name="test_task",
            feedback_system=self.feedback_system,
            heartbeat_interval=1,
            stall_threshold=5,
        )

    def test_heartbeat(self):
        """Test heartbeat functionality"""
        initial_time = self.monitor.last_heartbeat

        time.sleep(0.1)
        self.monitor.heartbeat(progress=0.5, status="running")

        self.assertGreater(self.monitor.last_heartbeat, initial_time)
        self.assertEqual(self.monitor.progress, 0.5)
        self.assertEqual(self.monitor.status, "running")

    def test_checkpoint(self):
        """Test checkpoint creation"""
        self.monitor.checkpoint("test_checkpoint", metadata={"test": True})

        self.assertEqual(len(self.monitor.checkpoints), 1)
        self.assertEqual(self.monitor.checkpoints[0]["name"], "test_checkpoint")
        self.assertEqual(self.monitor.checkpoints[0]["metadata"]["test"], True)

    def test_status(self):
        """Test status retrieval"""
        self.monitor.heartbeat(progress=0.75, status="processing")

        status = self.monitor.get_status()

        self.assertEqual(status["task_name"], "test_task")
        self.assertEqual(status["status"], "processing")
        self.assertEqual(status["progress"], 0.75)
        self.assertGreaterEqual(status["elapsed_time"], 0)
        self.assertFalse(status["is_stalled"])

    def test_stall_detection(self):
        """Test stall detection"""
        # Set last progress update to past
        self.monitor.last_progress_update = time.time() - 10

        # Heartbeat should detect stall
        self.monitor.heartbeat()

        # Check if stall feedback was added
        stall_feedback = [
            item
            for item in self.feedback_system.feedback_buffer
            if "stalled" in item.message.lower()
        ]

        self.assertGreater(len(stall_feedback), 0)


class TestFeedbackIntegration(unittest.TestCase):
    """Integration tests for feedback system"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_system = EnhancedFeedbackSystem(feedback_dir=Path(self.temp_dir))

    def test_end_to_end_workflow(self):
        """Test complete feedback workflow"""
        # 1. Add various feedback
        for i in range(5):
            self.feedback_system.add_feedback(
                category=FeedbackCategory.QUALITY,
                priority=FeedbackPriority.MEDIUM,
                source="test",
                message=f"Quality issue {i}",
                metrics={"quality_score": 3.0 - i * 0.1},
            )

        # 2. Analyze trends
        trend = self.feedback_system.analyze_trends("quality_score")
        self.assertEqual(trend["trend_direction"], "decreasing")

        # 3. Generate actions
        actions = self.feedback_system.generate_actions()
        self.assertGreater(len(actions), 0)

        # 4. Get health report
        report = self.feedback_system.get_health_report()
        self.assertIn("health_score", report)
        self.assertIn("recommended_actions", report)

        # 5. Export report
        output_path = Path(self.temp_dir) / "report.json"
        self.feedback_system.export_report(output_path)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
