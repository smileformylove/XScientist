#!/usr/bin/env python3
"""
Enhanced Feedback System for Long-Running Tasks
Provides robust feedback collection, aggregation, and action generation
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import statistics


class FeedbackPriority(Enum):
    """Feedback priority levels"""
    CRITICAL = "critical"  # Requires immediate action
    HIGH = "high"  # Should be addressed soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # Nice to have
    INFO = "info"  # Informational only


class FeedbackCategory(Enum):
    """Feedback categories"""
    QUALITY = "quality"  # Quality-related feedback
    PERFORMANCE = "performance"  # Performance metrics
    RESOURCE = "resource"  # Resource usage
    ERROR = "error"  # Error and failure feedback
    SUCCESS = "success"  # Success patterns
    STRATEGY = "strategy"  # Strategy adjustments


@dataclass
class FeedbackItem:
    """Structured feedback item"""
    timestamp: str
    category: str
    priority: str
    source: str
    message: str
    metrics: Dict[str, Any]
    context: Dict[str, Any]
    actionable: bool
    action_taken: Optional[str] = None
    resolved: bool = False


class EnhancedFeedbackSystem:
    """
    Enhanced feedback system with:
    - Multi-source feedback aggregation
    - Priority-based action generation
    - Trend analysis
    - Adaptive thresholds
    - Long-running task support
    """

    def __init__(
        self,
        feedback_dir: Path,
        window_size: int = 100,
        trend_window_hours: int = 24,
    ):
        """
        Initialize enhanced feedback system

        Args:
            feedback_dir: Directory for feedback storage
            window_size: Size of rolling window for metrics
            trend_window_hours: Hours to consider for trend analysis
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.trend_window_hours = trend_window_hours

        # Rolling windows for metrics
        self.metric_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Feedback buffer
        self.feedback_buffer: List[FeedbackItem] = []
        self.feedback_history: List[FeedbackItem] = []

        # Adaptive thresholds
        self.thresholds: Dict[str, Dict[str, float]] = {
            "quality_score": {"critical": 2.0, "high": 3.0, "medium": 4.0},
            "success_rate": {"critical": 0.3, "high": 0.5, "medium": 0.7},
            "error_rate": {"critical": 0.3, "high": 0.2, "medium": 0.1},
            "resource_usage": {"critical": 0.9, "high": 0.8, "medium": 0.7},
        }

        # Action templates
        self.action_templates = self._initialize_action_templates()

        # Load existing feedback
        self._load_feedback_history()

    def _initialize_action_templates(self) -> Dict[str, Dict]:
        """Initialize action templates for different feedback scenarios"""
        return {
            "low_quality": {
                "action": "Increase review rounds and strengthen quality gates",
                "priority": FeedbackPriority.HIGH,
                "estimated_impact": 0.7,
            },
            "high_error_rate": {
                "action": "Add error recovery mechanisms and increase timeout buffers",
                "priority": FeedbackPriority.CRITICAL,
                "estimated_impact": 0.8,
            },
            "resource_pressure": {
                "action": "Reduce parallelism and implement resource throttling",
                "priority": FeedbackPriority.HIGH,
                "estimated_impact": 0.6,
            },
            "declining_success": {
                "action": "Analyze recent failures and adjust strategy parameters",
                "priority": FeedbackPriority.HIGH,
                "estimated_impact": 0.7,
            },
            "strategy_drift": {
                "action": "Recalibrate strategy based on recent performance data",
                "priority": FeedbackPriority.MEDIUM,
                "estimated_impact": 0.5,
            },
        }

    def add_feedback(
        self,
        category: FeedbackCategory,
        priority: FeedbackPriority,
        source: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        actionable: bool = True,
    ) -> FeedbackItem:
        """
        Add feedback item

        Args:
            category: Feedback category
            priority: Priority level
            source: Feedback source
            message: Feedback message
            metrics: Associated metrics
            context: Additional context
            actionable: Whether this feedback requires action

        Returns:
            Created feedback item
        """
        item = FeedbackItem(
            timestamp=datetime.now().isoformat(),
            category=category.value,
            priority=priority.value,
            source=source,
            message=message,
            metrics=metrics or {},
            context=context or {},
            actionable=actionable,
        )

        self.feedback_buffer.append(item)
        self.feedback_history.append(item)

        # Update metric windows
        for key, value in (metrics or {}).items():
            if isinstance(value, (int, float)):
                self.metric_windows[key].append((time.time(), value))

        # Auto-save periodically
        if len(self.feedback_buffer) >= 10:
            self._save_feedback_batch()

        return item

    def analyze_trends(self, metric_name: str, hours: Optional[int] = None) -> Dict:
        """
        Analyze trends for a specific metric

        Args:
            metric_name: Name of the metric
            hours: Hours to look back (default: trend_window_hours)

        Returns:
            Trend analysis results
        """
        hours = hours or self.trend_window_hours
        cutoff_time = time.time() - (hours * 3600)

        if metric_name not in self.metric_windows:
            return {"error": f"Metric {metric_name} not found"}

        # Filter recent data
        recent_data = [
            (ts, val)
            for ts, val in self.metric_windows[metric_name]
            if ts >= cutoff_time
        ]

        if len(recent_data) < 2:
            return {"error": "Insufficient data for trend analysis"}

        values = [val for _, val in recent_data]
        timestamps = [ts for ts, _ in recent_data]

        # Calculate statistics
        mean_value = statistics.mean(values)
        median_value = statistics.median(values)
        stdev_value = statistics.stdev(values) if len(values) > 1 else 0

        # Calculate trend (simple linear regression)
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * val for i, val in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        trend_direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"

        # Recent vs historical comparison
        recent_mean = statistics.mean(values[-min(10, len(values)):])
        historical_mean = statistics.mean(values[:max(1, len(values) - 10)])
        change_pct = ((recent_mean - historical_mean) / historical_mean * 100) if historical_mean != 0 else 0

        return {
            "metric": metric_name,
            "data_points": len(values),
            "time_range_hours": (timestamps[-1] - timestamps[0]) / 3600,
            "mean": round(mean_value, 3),
            "median": round(median_value, 3),
            "stdev": round(stdev_value, 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "trend_direction": trend_direction,
            "slope": round(slope, 6),
            "recent_vs_historical_change_pct": round(change_pct, 2),
            "recent_mean": round(recent_mean, 3),
            "historical_mean": round(historical_mean, 3),
        }

    def generate_actions(self, max_actions: int = 5) -> List[Dict]:
        """
        Generate prioritized actions based on feedback

        Args:
            max_actions: Maximum number of actions to generate

        Returns:
            List of recommended actions
        """
        actions = []

        # Analyze critical feedback
        critical_feedback = [
            item for item in self.feedback_buffer
            if item.priority == FeedbackPriority.CRITICAL.value and not item.resolved
        ]

        for item in critical_feedback:
            actions.append({
                "priority": "critical",
                "source": item.source,
                "action": f"Address critical issue: {item.message}",
                "context": item.context,
                "estimated_impact": 0.9,
            })

        # Analyze trends and generate actions
        for metric_name in ["quality_score", "success_rate", "error_rate"]:
            if metric_name in self.metric_windows:
                trend = self.analyze_trends(metric_name)
                if "error" not in trend:
                    action = self._generate_action_from_trend(metric_name, trend)
                    if action:
                        actions.append(action)

        # Analyze feedback patterns
        pattern_actions = self._analyze_feedback_patterns()
        actions.extend(pattern_actions)

        # Sort by priority and impact
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        actions.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 3),
                -x.get("estimated_impact", 0)
            )
        )

        return actions[:max_actions]

    def _generate_action_from_trend(self, metric_name: str, trend: Dict) -> Optional[Dict]:
        """Generate action based on trend analysis"""
        if metric_name == "quality_score":
            if trend["mean"] < self.thresholds["quality_score"]["high"]:
                return {
                    "priority": "high",
                    "source": "trend_analysis",
                    "action": self.action_templates["low_quality"]["action"],
                    "metric": metric_name,
                    "trend": trend["trend_direction"],
                    "current_value": trend["mean"],
                    "estimated_impact": self.action_templates["low_quality"]["estimated_impact"],
                }

        elif metric_name == "success_rate":
            if trend["mean"] < self.thresholds["success_rate"]["high"]:
                if trend["trend_direction"] == "decreasing":
                    return {
                        "priority": "high",
                        "source": "trend_analysis",
                        "action": self.action_templates["declining_success"]["action"],
                        "metric": metric_name,
                        "trend": trend["trend_direction"],
                        "current_value": trend["mean"],
                        "estimated_impact": self.action_templates["declining_success"]["estimated_impact"],
                    }

        elif metric_name == "error_rate":
            if trend["mean"] > self.thresholds["error_rate"]["high"]:
                return {
                    "priority": "critical" if trend["mean"] > self.thresholds["error_rate"]["critical"] else "high",
                    "source": "trend_analysis",
                    "action": self.action_templates["high_error_rate"]["action"],
                    "metric": metric_name,
                    "trend": trend["trend_direction"],
                    "current_value": trend["mean"],
                    "estimated_impact": self.action_templates["high_error_rate"]["estimated_impact"],
                }

        return None

    def _analyze_feedback_patterns(self) -> List[Dict]:
        """Analyze patterns in feedback and generate actions"""
        actions = []

        # Group feedback by category
        category_counts = defaultdict(int)
        for item in self.feedback_buffer:
            if not item.resolved:
                category_counts[item.category] += 1

        # Generate actions for high-frequency categories
        for category, count in category_counts.items():
            if count >= 5:  # Threshold for pattern detection
                actions.append({
                    "priority": "medium",
                    "source": "pattern_analysis",
                    "action": f"Address recurring {category} issues (detected {count} instances)",
                    "category": category,
                    "frequency": count,
                    "estimated_impact": 0.6,
                })

        return actions

    def get_health_report(self) -> Dict:
        """
        Generate comprehensive health report

        Returns:
            Health report with metrics and recommendations
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "feedback_summary": {
                "total_items": len(self.feedback_history),
                "unresolved_items": len([item for item in self.feedback_buffer if not item.resolved]),
                "critical_items": len([item for item in self.feedback_buffer if item.priority == "critical" and not item.resolved]),
            },
            "metrics": {},
            "trends": {},
            "recommended_actions": [],
            "health_score": 0.0,
        }

        # Analyze all tracked metrics
        for metric_name in self.metric_windows.keys():
            trend = self.analyze_trends(metric_name)
            if "error" not in trend:
                report["trends"][metric_name] = trend

        # Generate actions
        report["recommended_actions"] = self.generate_actions(max_actions=10)

        # Calculate health score
        health_score = self._calculate_health_score(report)
        report["health_score"] = round(health_score, 2)

        return report

    def _calculate_health_score(self, report: Dict) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0

        # Penalize for unresolved critical items
        critical_items = report["feedback_summary"]["critical_items"]
        score -= critical_items * 10

        # Penalize for high error rate
        if "error_rate" in report["trends"]:
            error_rate = report["trends"]["error_rate"].get("mean", 0)
            score -= error_rate * 50

        # Reward for high success rate
        if "success_rate" in report["trends"]:
            success_rate = report["trends"]["success_rate"].get("mean", 0)
            score += (success_rate - 0.5) * 20

        # Reward for high quality
        if "quality_score" in report["trends"]:
            quality_score = report["trends"]["quality_score"].get("mean", 0)
            score += (quality_score - 3.0) * 10

        return max(0.0, min(100.0, score))

    def mark_resolved(self, feedback_item: FeedbackItem, action_taken: str):
        """Mark feedback item as resolved"""
        feedback_item.resolved = True
        feedback_item.action_taken = action_taken

    def clear_resolved(self):
        """Clear resolved feedback from buffer"""
        self.feedback_buffer = [item for item in self.feedback_buffer if not item.resolved]

    def _save_feedback_batch(self):
        """Save feedback batch to disk"""
        batch_file = self.feedback_dir / f"feedback_batch_{int(time.time())}.json"
        with open(batch_file, "w") as f:
            json.dump(
                [asdict(item) for item in self.feedback_buffer],
                f,
                indent=2,
                ensure_ascii=False,
            )
        self.feedback_buffer = []

    def _load_feedback_history(self):
        """Load feedback history from disk"""
        for batch_file in sorted(self.feedback_dir.glob("feedback_batch_*.json")):
            try:
                with open(batch_file, "r") as f:
                    items = json.load(f)
                    for item_dict in items:
                        item = FeedbackItem(**item_dict)
                        self.feedback_history.append(item)
            except Exception as e:
                print(f"Warning: Failed to load {batch_file}: {e}")

    def export_report(self, output_path: Path):
        """Export comprehensive report to file"""
        report = self.get_health_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


class LongRunningTaskMonitor:
    """
    Monitor for long-running tasks with:
    - Heartbeat tracking
    - Progress monitoring
    - Stall detection
    - Resource tracking
    """

    def __init__(
        self,
        task_name: str,
        feedback_system: EnhancedFeedbackSystem,
        heartbeat_interval: int = 300,  # 5 minutes
        stall_threshold: int = 1800,  # 30 minutes
    ):
        """
        Initialize long-running task monitor

        Args:
            task_name: Name of the task
            feedback_system: Feedback system instance
            heartbeat_interval: Seconds between heartbeats
            stall_threshold: Seconds before considering task stalled
        """
        self.task_name = task_name
        self.feedback_system = feedback_system
        self.heartbeat_interval = heartbeat_interval
        self.stall_threshold = stall_threshold

        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.last_progress_update = time.time()
        self.progress = 0.0
        self.status = "running"

        self.checkpoints: List[Dict] = []

    def heartbeat(self, progress: Optional[float] = None, status: Optional[str] = None):
        """
        Send heartbeat signal

        Args:
            progress: Current progress (0-1)
            status: Current status
        """
        current_time = time.time()
        self.last_heartbeat = current_time

        if progress is not None:
            if progress > self.progress:
                self.last_progress_update = current_time
            self.progress = progress

        if status is not None:
            self.status = status

        # Check for stalls
        time_since_progress = current_time - self.last_progress_update
        if time_since_progress > self.stall_threshold:
            self.feedback_system.add_feedback(
                category=FeedbackCategory.PERFORMANCE,
                priority=FeedbackPriority.HIGH,
                source=self.task_name,
                message=f"Task appears stalled (no progress for {time_since_progress/60:.1f} minutes)",
                metrics={"stall_duration": time_since_progress, "progress": self.progress},
                context={"status": self.status},
            )

    def checkpoint(self, name: str, metadata: Optional[Dict] = None):
        """
        Create checkpoint

        Args:
            name: Checkpoint name
            metadata: Additional metadata
        """
        checkpoint = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "progress": self.progress,
            "metadata": metadata or {},
        }
        self.checkpoints.append(checkpoint)

        self.feedback_system.add_feedback(
            category=FeedbackCategory.SUCCESS,
            priority=FeedbackPriority.INFO,
            source=self.task_name,
            message=f"Checkpoint reached: {name}",
            metrics={"progress": self.progress, "elapsed_time": checkpoint["elapsed_time"]},
            context=checkpoint,
            actionable=False,
        )

    def get_status(self) -> Dict:
        """Get current task status"""
        current_time = time.time()
        return {
            "task_name": self.task_name,
            "status": self.status,
            "progress": self.progress,
            "elapsed_time": current_time - self.start_time,
            "time_since_heartbeat": current_time - self.last_heartbeat,
            "time_since_progress": current_time - self.last_progress_update,
            "checkpoints": len(self.checkpoints),
            "is_stalled": (current_time - self.last_progress_update) > self.stall_threshold,
        }
