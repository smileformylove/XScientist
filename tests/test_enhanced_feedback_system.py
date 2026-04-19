#!/usr/bin/env python3
"""
Tests for Enhanced Feedback System
"""

import unittest
import tempfile
import time
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

        initial_count = len(self.feedback_system.feedback_buffer)
        self.feedback_system.clear_resolved()
        final_count = len(self.feedback_system.feedback_buffer)

        self.assertEqual(final_count, initial_count - 1)


class TestLongRunningTaskMonitor(unittest.TestCase):
    """Test long-running task monitor"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_system = EnhancedFeedbackSystem(
            feedback_dir=Path(self.temp_dir)
        )
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
            item for item in self.feedback_system.feedback_buffer
            if "stalled" in item.message.lower()
        ]

        self.assertGreater(len(stall_feedback), 0)


class TestFeedbackIntegration(unittest.TestCase):
    """Integration tests for feedback system"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_system = EnhancedFeedbackSystem(
            feedback_dir=Path(self.temp_dir)
        )

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
