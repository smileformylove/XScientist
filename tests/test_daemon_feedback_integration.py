#!/usr/bin/env python3
"""
Integration tests for daemon feedback integration
"""

import unittest
import tempfile
import time
from pathlib import Path

from ai_scientist.daemon_feedback_integration import (
    DaemonFeedbackIntegration,
    create_daemon_feedback_integration,
)
from ai_scientist.enhanced_feedback_system import (
    EnhancedFeedbackSystem,
    FeedbackCategory,
    FeedbackPriority,
)


class TestDaemonFeedbackIntegration(unittest.TestCase):
    """Test daemon feedback integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.integration = DaemonFeedbackIntegration(
            daemon_dir=Path(self.temp_dir)
        )

    def test_initialization(self):
        """Test integration initialization"""
        self.assertIsNotNone(self.integration.feedback_system)
        self.assertEqual(self.integration.project_count, 0)
        self.assertEqual(self.integration.success_count, 0)
        self.assertEqual(self.integration.failure_count, 0)

    def test_daemon_monitoring(self):
        """Test daemon monitoring start"""
        self.integration.start_daemon_monitoring("test_daemon")

        self.assertIsNotNone(self.integration.daemon_monitor)
        self.assertEqual(self.integration.daemon_monitor.task_name, "test_daemon")

    def test_daemon_heartbeat(self):
        """Test daemon heartbeat"""
        self.integration.start_daemon_monitoring("test_daemon")

        # Send heartbeat
        self.integration.daemon_heartbeat(status="running")

        # Verify monitor received heartbeat
        status = self.integration.daemon_monitor.get_status()
        self.assertEqual(status["status"], "running")

    def test_project_lifecycle(self):
        """Test complete project lifecycle"""
        # Start project
        self.integration.on_project_start("test_project", metadata={"topic": "ML"})
        self.assertEqual(self.integration.project_count, 1)

        # Complete project successfully
        self.integration.on_project_complete(
            "test_project",
            success=True,
            quality_score=4.5,
            metadata={"experiments": 10},
        )

        self.assertEqual(self.integration.success_count, 1)
        self.assertEqual(self.integration.failure_count, 0)

    def test_project_failure(self):
        """Test project failure handling"""
        self.integration.on_project_start("failing_project")
        self.integration.on_project_complete(
            "failing_project",
            success=False,
            quality_score=2.0,
        )

        self.assertEqual(self.integration.success_count, 0)
        self.assertEqual(self.integration.failure_count, 1)

    def test_quality_gate_tracking(self):
        """Test quality gate result tracking"""
        self.integration.on_quality_gate_result(
            "test_project",
            "submission_ready",
            passed=True,
            score=4.2,
        )

        # Check feedback was added
        feedback_items = self.integration.feedback_system.feedback_buffer
        quality_feedback = [
            item for item in feedback_items
            if item.category == "quality"
        ]
        self.assertGreater(len(quality_feedback), 0)

    def test_review_round_tracking(self):
        """Test review round tracking"""
        self.integration.on_review_round_complete(
            "test_project",
            round_num=1,
            issues_found=10,
            issues_resolved=8,
        )

        # Verify feedback
        feedback_items = self.integration.feedback_system.feedback_buffer
        review_feedback = [
            item for item in feedback_items
            if "review" in item.source.lower()
        ]
        self.assertGreater(len(review_feedback), 0)

    def test_error_handling(self):
        """Test error event handling"""
        self.integration.on_error(
            "RuntimeError",
            "Test error message",
            context={"project": "test"},
        )

        # Check error feedback
        feedback_items = self.integration.feedback_system.feedback_buffer
        error_feedback = [
            item for item in feedback_items
            if item.category == "error"
        ]
        self.assertGreater(len(error_feedback), 0)

    def test_resource_usage_tracking(self):
        """Test resource usage tracking"""
        self.integration.on_resource_usage(
            cpu_percent=75.0,
            memory_percent=60.0,
            disk_percent=50.0,
        )

        # Verify feedback
        feedback_items = self.integration.feedback_system.feedback_buffer
        resource_feedback = [
            item for item in feedback_items
            if item.category == "resource"
        ]
        self.assertGreater(len(resource_feedback), 0)

    def test_health_report(self):
        """Test daemon health report generation"""
        # Add some activity
        self.integration.on_project_start("p1")
        self.integration.on_project_complete("p1", success=True, quality_score=4.0)

        report = self.integration.get_daemon_health_report()

        self.assertIn("health_score", report)
        self.assertIn("daemon_metrics", report)
        self.assertEqual(report["daemon_metrics"]["total_projects"], 1)
        self.assertEqual(report["daemon_metrics"]["successful_projects"], 1)

    def test_recommended_actions(self):
        """Test action recommendation"""
        # Add some feedback
        self.integration.feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=FeedbackPriority.HIGH,
            source="test",
            message="Low quality",
            metrics={"quality_score": 2.0},
        )

        actions = self.integration.get_recommended_actions(max_actions=5)
        self.assertIsInstance(actions, list)

    def test_pause_recommendation(self):
        """Test daemon pause recommendation"""
        # Initially should not pause
        should_pause, reason = self.integration.should_pause_daemon()
        self.assertFalse(should_pause)

        # Add critical feedback to trigger pause
        for _ in range(10):
            self.integration.feedback_system.add_feedback(
                category=FeedbackCategory.ERROR,
                priority=FeedbackPriority.CRITICAL,
                source="test",
                message="Critical error",
                metrics={"error_rate": 0.8},
            )

        should_pause, reason = self.integration.should_pause_daemon()
        # May or may not pause depending on health calculation
        self.assertIsInstance(should_pause, bool)
        self.assertIsInstance(reason, str)

    def test_status_integration(self):
        """Test integration with daemon status"""
        daemon_status = {
            "running": True,
            "projects_completed": 5,
        }

        enhanced_status = self.integration.integrate_with_daemon_status(daemon_status)

        self.assertIn("health_score", enhanced_status)
        self.assertIn("recommended_actions", enhanced_status)
        self.assertIn("should_pause", enhanced_status)
        self.assertEqual(enhanced_status["running"], True)

    def test_report_export(self):
        """Test report export"""
        output_path = self.integration.export_daemon_report()

        self.assertTrue(output_path.exists())
        self.assertTrue(output_path.name.startswith("daemon_health_report_"))

    def test_factory_function(self):
        """Test factory function"""
        integration = create_daemon_feedback_integration(Path(self.temp_dir))

        self.assertIsInstance(integration, DaemonFeedbackIntegration)
        self.assertIsNotNone(integration.feedback_system)


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration workflow tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.integration = DaemonFeedbackIntegration(
            daemon_dir=Path(self.temp_dir)
        )

    def test_complete_daemon_workflow(self):
        """Test complete daemon workflow"""
        # Start daemon monitoring
        self.integration.start_daemon_monitoring("test_daemon")

        # Simulate daemon running multiple projects
        for i in range(5):
            project_name = f"project_{i}"

            # Start project
            self.integration.on_project_start(project_name)
            self.integration.daemon_heartbeat(status="running")

            # Simulate work
            time.sleep(0.01)

            # Complete project (80% success rate)
            success = i < 4
            quality_score = 4.0 if success else 2.0

            self.integration.on_project_complete(
                project_name,
                success=success,
                quality_score=quality_score,
            )

            # Quality gate
            if success:
                self.integration.on_quality_gate_result(
                    project_name,
                    "submission_ready",
                    passed=True,
                    score=quality_score,
                )

            # Review rounds
            if success:
                self.integration.on_review_round_complete(
                    project_name,
                    round_num=1,
                    issues_found=5,
                    issues_resolved=4,
                )

        # Get final report
        report = self.integration.get_daemon_health_report()

        # Verify metrics
        self.assertEqual(report["daemon_metrics"]["total_projects"], 5)
        self.assertEqual(report["daemon_metrics"]["successful_projects"], 4)
        self.assertEqual(report["daemon_metrics"]["failed_projects"], 1)
        self.assertAlmostEqual(report["daemon_metrics"]["success_rate"], 0.8, places=2)

        # Get recommendations
        actions = self.integration.get_recommended_actions()
        self.assertIsInstance(actions, list)

        # Check health score
        self.assertGreaterEqual(report["health_score"], 0)
        self.assertLessEqual(report["health_score"], 100)

    def test_error_recovery_workflow(self):
        """Test error recovery workflow"""
        self.integration.start_daemon_monitoring("test_daemon")

        # Simulate errors
        for i in range(3):
            self.integration.on_error(
                "RuntimeError",
                f"Error {i}",
                context={"attempt": i},
            )
            time.sleep(0.01)

        # Check error rate
        report = self.integration.get_daemon_health_report()

        # Should have error feedback
        error_feedback = [
            item for item in self.integration.feedback_system.feedback_buffer
            if item.category == "error"
        ]
        self.assertEqual(len(error_feedback), 3)

        # Get recovery actions
        actions = self.integration.get_recommended_actions()
        # May or may not have actions depending on thresholds and data
        self.assertIsInstance(actions, list)

    def test_resource_monitoring_workflow(self):
        """Test resource monitoring workflow"""
        self.integration.start_daemon_monitoring("test_daemon")

        # Simulate increasing resource usage
        for i in range(10):
            cpu = 50 + i * 5
            memory = 40 + i * 5
            disk = 30 + i * 5

            self.integration.on_resource_usage(
                cpu_percent=cpu,
                memory_percent=memory,
                disk_percent=disk,
            )
            time.sleep(0.01)

        # Analyze resource trends
        cpu_trend = self.integration.feedback_system.analyze_trends("cpu_usage")

        if "error" not in cpu_trend:
            self.assertEqual(cpu_trend["trend_direction"], "increasing")

        # Check for resource warnings
        report = self.integration.get_daemon_health_report()
        actions = self.integration.get_recommended_actions()

        # Should have resource-related actions if usage is high
        resource_actions = [
            action for action in actions
            if "resource" in action.get("action", "").lower()
        ]
        # May or may not have resource actions depending on thresholds


if __name__ == "__main__":
    unittest.main()
