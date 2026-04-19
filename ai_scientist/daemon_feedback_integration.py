#!/usr/bin/env python3
"""
Daemon Integration for Enhanced Feedback System

This module provides integration between the continuous research daemon
and the enhanced feedback system for better monitoring and self-improvement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from ai_scientist.enhanced_feedback_system import (
    EnhancedFeedbackSystem,
    LongRunningTaskMonitor,
    FeedbackCategory,
    FeedbackPriority,
)


class DaemonFeedbackIntegration:
    """
    Integration layer between daemon and enhanced feedback system

    Provides:
    - Automatic feedback collection from daemon events
    - Health monitoring for daemon operations
    - Action recommendations based on daemon performance
    - Integration with existing daemon feedback mechanisms
    """

    def __init__(
        self,
        daemon_dir: Path,
        feedback_system: Optional[EnhancedFeedbackSystem] = None,
    ):
        """
        Initialize daemon feedback integration

        Args:
            daemon_dir: Daemon working directory
            feedback_system: Optional existing feedback system instance
        """
        self.daemon_dir = Path(daemon_dir)
        self.feedback_dir = self.daemon_dir / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or use provided feedback system
        if feedback_system is None:
            self.feedback_system = EnhancedFeedbackSystem(
                feedback_dir=self.feedback_dir,
                window_size=100,
                trend_window_hours=24,
            )
        else:
            self.feedback_system = feedback_system

        # Task monitor for daemon
        self.daemon_monitor: Optional[LongRunningTaskMonitor] = None

        # Metrics tracking
        self.project_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.start_time = time.time()

    def start_daemon_monitoring(self, daemon_name: str = "research_daemon"):
        """Start monitoring the daemon as a long-running task"""
        self.daemon_monitor = LongRunningTaskMonitor(
            task_name=daemon_name,
            feedback_system=self.feedback_system,
            heartbeat_interval=300,  # 5 minutes
            stall_threshold=1800,  # 30 minutes
        )
        self.start_time = time.time()

    def daemon_heartbeat(self, status: Optional[str] = None):
        """Send daemon heartbeat"""
        if self.daemon_monitor:
            # Calculate progress based on time or projects
            elapsed = time.time() - self.start_time
            progress = min(0.99, elapsed / (24 * 3600))  # Cap at 99% for 24h
            self.daemon_monitor.heartbeat(progress=progress, status=status)

    def on_project_start(self, project_name: str, metadata: Optional[Dict] = None):
        """Handle project start event"""
        self.project_count += 1

        if self.daemon_monitor:
            self.daemon_monitor.checkpoint(
                f"project_start_{project_name}",
                metadata={"project": project_name, **(metadata or {})},
            )

        self.feedback_system.add_feedback(
            category=FeedbackCategory.SUCCESS,
            priority=FeedbackPriority.INFO,
            source="daemon",
            message=f"Started project: {project_name}",
            metrics={"project_count": self.project_count},
            context={"project": project_name, **(metadata or {})},
            actionable=False,
        )

    def on_project_complete(
        self,
        project_name: str,
        success: bool,
        quality_score: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """Handle project completion event"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Calculate success rate
        total = self.success_count + self.failure_count
        success_rate = self.success_count / total if total > 0 else 0

        # Determine priority based on success
        priority = FeedbackPriority.INFO if success else FeedbackPriority.HIGH

        # Add feedback
        self.feedback_system.add_feedback(
            category=FeedbackCategory.SUCCESS if success else FeedbackCategory.ERROR,
            priority=priority,
            source="daemon",
            message=f"Project {'completed' if success else 'failed'}: {project_name}",
            metrics={
                "success_rate": success_rate,
                "quality_score": quality_score or 0,
                "total_projects": total,
            },
            context={
                "project": project_name,
                "success": success,
                **(metadata or {}),
            },
            actionable=not success,
        )

        # Checkpoint
        if self.daemon_monitor:
            self.daemon_monitor.checkpoint(
                f"project_{'complete' if success else 'failed'}_{project_name}",
                metadata={
                    "project": project_name,
                    "success": success,
                    "quality_score": quality_score,
                },
            )

    def on_quality_gate_result(
        self,
        project_name: str,
        gate_name: str,
        passed: bool,
        score: Optional[float] = None,
    ):
        """Handle quality gate result"""
        priority = FeedbackPriority.MEDIUM if not passed else FeedbackPriority.INFO

        self.feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=priority,
            source="quality_gate",
            message=f"Quality gate '{gate_name}' {'passed' if passed else 'failed'} for {project_name}",
            metrics={"gate_score": score or 0, "gate_passed": 1 if passed else 0},
            context={"project": project_name, "gate": gate_name, "passed": passed},
            actionable=not passed,
        )

    def on_review_round_complete(
        self,
        project_name: str,
        round_num: int,
        issues_found: int,
        issues_resolved: int,
    ):
        """Handle review round completion"""
        resolution_rate = (
            issues_resolved / issues_found if issues_found > 0 else 1.0
        )

        self.feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=FeedbackPriority.INFO,
            source="review_system",
            message=f"Review round {round_num} complete for {project_name}",
            metrics={
                "issues_found": issues_found,
                "issues_resolved": issues_resolved,
                "resolution_rate": resolution_rate,
            },
            context={
                "project": project_name,
                "round": round_num,
            },
            actionable=resolution_rate < 0.5,
        )

    def on_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict] = None,
    ):
        """Handle error event"""
        # Calculate error rate
        total = self.success_count + self.failure_count
        error_rate = self.failure_count / total if total > 0 else 0

        self.feedback_system.add_feedback(
            category=FeedbackCategory.ERROR,
            priority=FeedbackPriority.CRITICAL if error_rate > 0.3 else FeedbackPriority.HIGH,
            source="daemon",
            message=f"{error_type}: {error_message}",
            metrics={"error_rate": error_rate},
            context=context or {},
            actionable=True,
        )

    def on_resource_usage(
        self,
        cpu_percent: Optional[float] = None,
        memory_percent: Optional[float] = None,
        disk_percent: Optional[float] = None,
    ):
        """Handle resource usage update"""
        metrics = {}
        max_usage = 0.0

        if cpu_percent is not None:
            metrics["cpu_usage"] = cpu_percent / 100
            max_usage = max(max_usage, cpu_percent / 100)

        if memory_percent is not None:
            metrics["memory_usage"] = memory_percent / 100
            max_usage = max(max_usage, memory_percent / 100)

        if disk_percent is not None:
            metrics["disk_usage"] = disk_percent / 100
            max_usage = max(max_usage, disk_percent / 100)

        # Determine priority based on usage
        if max_usage > 0.9:
            priority = FeedbackPriority.CRITICAL
        elif max_usage > 0.8:
            priority = FeedbackPriority.HIGH
        elif max_usage > 0.7:
            priority = FeedbackPriority.MEDIUM
        else:
            priority = FeedbackPriority.INFO

        self.feedback_system.add_feedback(
            category=FeedbackCategory.RESOURCE,
            priority=priority,
            source="resource_monitor",
            message=f"Resource usage update (max: {max_usage*100:.1f}%)",
            metrics=metrics,
            context={},
            actionable=max_usage > 0.8,
        )

    def get_daemon_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive daemon health report

        Returns:
            Health report with daemon-specific metrics
        """
        base_report = self.feedback_system.get_health_report()

        # Add daemon-specific metrics
        total_projects = self.success_count + self.failure_count
        success_rate = self.success_count / total_projects if total_projects > 0 else 0
        elapsed_hours = (time.time() - self.start_time) / 3600

        daemon_metrics = {
            "daemon_uptime_hours": round(elapsed_hours, 2),
            "total_projects": total_projects,
            "successful_projects": self.success_count,
            "failed_projects": self.failure_count,
            "success_rate": round(success_rate, 3),
            "projects_per_hour": round(total_projects / elapsed_hours, 2) if elapsed_hours > 0 else 0,
        }

        base_report["daemon_metrics"] = daemon_metrics

        # Add daemon status
        if self.daemon_monitor:
            base_report["daemon_status"] = self.daemon_monitor.get_status()

        return base_report

    def export_daemon_report(self, output_path: Optional[Path] = None):
        """Export daemon health report to file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.daemon_dir / f"daemon_health_report_{timestamp}.json"

        report = self.get_daemon_health_report()

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Daemon health report exported to: {output_path}")
        return output_path

    def get_recommended_actions(self, max_actions: int = 10) -> list[Dict]:
        """Get recommended actions for daemon"""
        actions = self.feedback_system.generate_actions(max_actions=max_actions)

        # Add daemon-specific context
        for action in actions:
            action["daemon_context"] = {
                "uptime_hours": (time.time() - self.start_time) / 3600,
                "total_projects": self.success_count + self.failure_count,
                "success_rate": self.success_count / max(1, self.success_count + self.failure_count),
            }

        return actions

    def should_pause_daemon(self) -> tuple[bool, str]:
        """
        Check if daemon should pause based on health

        Returns:
            (should_pause, reason)
        """
        report = self.get_daemon_health_report()
        health_score = report.get("health_score", 100)

        # Critical health
        if health_score < 30:
            return True, f"Critical health score: {health_score}/100"

        # High error rate
        error_rate_trend = self.feedback_system.analyze_trends("error_rate")
        if "error" not in error_rate_trend:
            if error_rate_trend.get("mean", 0) > 0.5:
                return True, f"High error rate: {error_rate_trend['mean']*100:.1f}%"

        # Resource exhaustion
        for resource in ["cpu_usage", "memory_usage", "disk_usage"]:
            trend = self.feedback_system.analyze_trends(resource)
            if "error" not in trend:
                if trend.get("mean", 0) > 0.95:
                    return True, f"Resource exhaustion: {resource} at {trend['mean']*100:.1f}%"

        return False, ""

    def integrate_with_daemon_status(self, daemon_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate enhanced feedback with existing daemon status

        Args:
            daemon_status: Existing daemon status dict

        Returns:
            Enhanced status with feedback integration
        """
        # Add health score
        health_report = self.get_daemon_health_report()
        daemon_status["health_score"] = health_report["health_score"]
        daemon_status["health_report"] = health_report

        # Add recommended actions
        daemon_status["recommended_actions"] = self.get_recommended_actions(max_actions=5)

        # Add pause recommendation
        should_pause, reason = self.should_pause_daemon()
        daemon_status["should_pause"] = should_pause
        daemon_status["pause_reason"] = reason

        return daemon_status


def create_daemon_feedback_integration(daemon_dir: Path) -> DaemonFeedbackIntegration:
    """
    Factory function to create daemon feedback integration

    Args:
        daemon_dir: Daemon working directory

    Returns:
        Configured DaemonFeedbackIntegration instance
    """
    return DaemonFeedbackIntegration(daemon_dir=daemon_dir)
