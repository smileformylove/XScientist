# Enhanced Feedback System - Quick Start Guide

## Overview

The Enhanced Feedback System provides comprehensive monitoring, health tracking, and self-improvement capabilities for XScientist.

## Installation

The feedback system is already included in XScientist. No additional installation needed.

## Quick Start

### 1. Basic Usage

```python
from pathlib import Path
from ai_scientist.enhanced_feedback_system import (
    EnhancedFeedbackSystem,
    FeedbackCategory,
    FeedbackPriority,
)

# Initialize feedback system
feedback_system = EnhancedFeedbackSystem(
    feedback_dir=Path("./feedback"),
    window_size=100,
    trend_window_hours=24,
)

# Add feedback
feedback_system.add_feedback(
    category=FeedbackCategory.QUALITY,
    priority=FeedbackPriority.HIGH,
    source="review_system",
    message="Paper needs more evidence",
    metrics={"quality_score": 3.5, "evidence_count": 5},
    context={"project": "my_project"},
)

# Get health report
report = feedback_system.get_health_report()
print(f"Health Score: {report['health_score']}/100")

# Generate actions
actions = feedback_system.generate_actions(max_actions=5)
for action in actions:
    print(f"- [{action['priority']}] {action['action']}")
```

### 2. Long-Running Task Monitoring

```python
from ai_scientist.enhanced_feedback_system import LongRunningTaskMonitor

# Create monitor
monitor = LongRunningTaskMonitor(
    task_name="research_project",
    feedback_system=feedback_system,
    heartbeat_interval=300,  # 5 minutes
    stall_threshold=1800,    # 30 minutes
)

# In your task loop
for i in range(100):
    # Do work
    process_step(i)
    
    # Send heartbeat
    monitor.heartbeat(progress=i/100, status="processing")
    
    # Create checkpoints at milestones
    if i % 10 == 0:
        monitor.checkpoint(f"step_{i}", metadata={"step": i})

# Check if task is stalled
status = monitor.get_status()
if status['is_stalled']:
    print("Task is stalled!")
```

### 3. Daemon Integration

```python
from ai_scientist.daemon_feedback_integration import DaemonFeedbackIntegration

# Create integration
integration = DaemonFeedbackIntegration(daemon_dir=Path("./daemon"))

# Start monitoring
integration.start_daemon_monitoring("my_daemon")

# In your daemon loop
while running:
    # Send heartbeat
    integration.daemon_heartbeat(status="running")
    
    # Track project events
    integration.on_project_start("project_1", metadata={"topic": "ML"})
    
    # ... do work ...
    
    integration.on_project_complete(
        "project_1",
        success=True,
        quality_score=4.2,
    )
    
    # Check if should pause
    should_pause, reason = integration.should_pause_daemon()
    if should_pause:
        print(f"Pausing daemon: {reason}")
        break
    
    # Export periodic reports
    if time_to_report():
        integration.export_daemon_report()
```

## Command-Line Interface

### Check System Status

```bash
python3 feedback_cli.py --feedback-dir ./feedback status
```

Output:
```
======================================================================
FEEDBACK SYSTEM STATUS
======================================================================

Timestamp: 2026-04-20T10:30:00
Health Score: 85/100
Status: 🟢 EXCELLENT

Feedback Items:
  Total: 150
  Unresolved: 12
  Critical: 0

Key Metrics:
  quality_score:
    Mean: 4.200
    Trend: increasing
    Recent change: +5.2%
```

### View Recommended Actions

```bash
python3 feedback_cli.py --feedback-dir ./feedback actions --max-actions 5
```

Output:
```
======================================================================
RECOMMENDED ACTIONS
======================================================================

1. 🟠 [HIGH]
   Increase review rounds and strengthen quality gates
   Estimated Impact: 70%
   Metric: quality_score

2. 🟡 [MEDIUM]
   Address recurring quality issues (detected 8 instances)
   Estimated Impact: 60%
```

### Analyze Trends

```bash
python3 feedback_cli.py --feedback-dir ./feedback trends \
  --metrics quality_score success_rate error_rate \
  --hours 24
```

Output:
```
======================================================================
METRIC TRENDS
======================================================================

📊 quality_score
   Data Points: 45
   Time Range: 23.5 hours
   Mean: 4.200
   Median: 4.300
   Std Dev: 0.450
   Range: [3.200, 4.800]
   Trend: increasing (slope: 0.012000)
   Recent vs Historical: +5.2%
```

### Export Report

```bash
python3 feedback_cli.py --feedback-dir ./feedback report \
  --output health_report.json \
  --show
```

### Add Feedback Manually

```bash
python3 feedback_cli.py --feedback-dir ./feedback add \
  --category quality \
  --priority high \
  --source manual \
  --message "Need more ablation studies" \
  --metrics quality_score=3.5 evidence_count=5
```

### Clear Resolved Feedback

```bash
python3 feedback_cli.py --feedback-dir ./feedback clear
```

## Integration Examples

### Example 1: Research Project with Feedback

```python
from pathlib import Path
from ai_scientist.enhanced_feedback_system import (
    EnhancedFeedbackSystem,
    LongRunningTaskMonitor,
    FeedbackCategory,
    FeedbackPriority,
)

def run_research_project(project_name: str):
    # Initialize feedback system
    feedback_system = EnhancedFeedbackSystem(
        feedback_dir=Path(f"./projects/{project_name}/feedback")
    )
    
    # Create monitor
    monitor = LongRunningTaskMonitor(
        task_name=project_name,
        feedback_system=feedback_system,
    )
    
    try:
        # Stage 1: Ideation
        monitor.checkpoint("ideation_start")
        ideas = generate_ideas()
        monitor.heartbeat(progress=0.2, status="ideation_complete")
        
        feedback_system.add_feedback(
            category=FeedbackCategory.SUCCESS,
            priority=FeedbackPriority.INFO,
            source="ideation",
            message=f"Generated {len(ideas)} ideas",
            metrics={"idea_count": len(ideas)},
        )
        
        # Stage 2: Experiments
        monitor.checkpoint("experiments_start")
        results = run_experiments(ideas[0])
        monitor.heartbeat(progress=0.6, status="experiments_complete")
        
        # Stage 3: Writing
        monitor.checkpoint("writing_start")
        paper = write_paper(results)
        monitor.heartbeat(progress=0.8, status="writing_complete")
        
        # Stage 4: Review
        monitor.checkpoint("review_start")
        review_results = review_paper(paper)
        monitor.heartbeat(progress=1.0, status="complete")
        
        # Add quality feedback
        feedback_system.add_feedback(
            category=FeedbackCategory.QUALITY,
            priority=FeedbackPriority.INFO,
            source="review",
            message="Project completed successfully",
            metrics={
                "quality_score": review_results['score'],
                "issues_found": review_results['issues'],
            },
        )
        
        # Generate final report
        report = feedback_system.get_health_report()
        print(f"Project Health Score: {report['health_score']}/100")
        
        # Get recommendations for next project
        actions = feedback_system.generate_actions()
        print("\nRecommendations for next project:")
        for action in actions[:3]:
            print(f"  - {action['action']}")
        
    except Exception as e:
        feedback_system.add_feedback(
            category=FeedbackCategory.ERROR,
            priority=FeedbackPriority.CRITICAL,
            source="project",
            message=f"Project failed: {str(e)}",
            metrics={"error_rate": 1.0},
        )
        raise
```

### Example 2: Continuous Monitoring

```python
import time
from datetime import datetime

def monitor_daemon_health(feedback_dir: Path, interval: int = 300):
    """Monitor daemon health and alert on issues"""
    feedback_system = EnhancedFeedbackSystem(feedback_dir=feedback_dir)
    
    while True:
        # Get health report
        report = feedback_system.get_health_report()
        health_score = report['health_score']
        
        print(f"[{datetime.now()}] Health Score: {health_score}/100")
        
        # Alert on low health
        if health_score < 50:
            print("⚠️  WARNING: Low health score!")
            actions = feedback_system.generate_actions(max_actions=3)
            print("Recommended actions:")
            for action in actions:
                print(f"  - {action['action']}")
        
        # Check for critical issues
        critical_count = report['feedback_summary']['critical_items']
        if critical_count > 0:
            print(f"🔴 CRITICAL: {critical_count} critical issues!")
        
        # Analyze trends
        for metric in ['error_rate', 'success_rate', 'quality_score']:
            trend = feedback_system.analyze_trends(metric)
            if 'error' not in trend:
                if trend['trend_direction'] == 'decreasing' and metric != 'error_rate':
                    print(f"⚠️  {metric} is decreasing: {trend['mean']:.3f}")
        
        time.sleep(interval)
```

## Best Practices

### 1. Regular Heartbeats

Send heartbeats regularly to detect stalls:

```python
# In long-running loops
for item in items:
    process(item)
    monitor.heartbeat(progress=current/total)
```

### 2. Meaningful Checkpoints

Create checkpoints at important milestones:

```python
monitor.checkpoint("experiments_complete", metadata={
    "experiments_run": 10,
    "success_rate": 0.8,
})
```

### 3. Actionable Feedback

Mark feedback as actionable when it requires intervention:

```python
feedback_system.add_feedback(
    category=FeedbackCategory.QUALITY,
    priority=FeedbackPriority.HIGH,
    source="review",
    message="Missing ablation studies",
    actionable=True,  # Requires action
)
```

### 4. Resolve Feedback

Mark feedback as resolved after taking action:

```python
# After fixing the issue
feedback_system.mark_resolved(feedback_item, "Added 3 ablation studies")

# Periodically clear resolved items
feedback_system.clear_resolved()
```

### 5. Export Reports Regularly

Export health reports for historical tracking:

```python
# Daily reports
if datetime.now().hour == 0:
    feedback_system.export_report(
        Path(f"reports/health_{datetime.now().date()}.json")
    )
```

## Troubleshooting

### Issue: No trends available

**Cause**: Not enough data points

**Solution**: Add more feedback with metrics:
```python
feedback_system.add_feedback(
    category=FeedbackCategory.PERFORMANCE,
    priority=FeedbackPriority.INFO,
    source="monitor",
    message="Performance update",
    metrics={"quality_score": 4.0},  # Include metrics
)
```

### Issue: Health score always 100

**Cause**: No negative feedback

**Solution**: System is healthy! Continue monitoring.

### Issue: Stall detection not working

**Cause**: Not sending heartbeats with progress updates

**Solution**: Update progress in heartbeats:
```python
monitor.heartbeat(progress=0.5)  # Must update progress
```

## Advanced Usage

### Custom Thresholds

```python
feedback_system.thresholds["quality_score"]["critical"] = 1.5
feedback_system.thresholds["error_rate"]["high"] = 0.15
```

### Custom Action Templates

```python
feedback_system.action_templates["custom_issue"] = {
    "action": "Take custom action",
    "priority": FeedbackPriority.HIGH,
    "estimated_impact": 0.8,
}
```

### Trend Analysis for Custom Metrics

```python
# Add custom metrics
feedback_system.add_feedback(
    category=FeedbackCategory.PERFORMANCE,
    priority=FeedbackPriority.INFO,
    source="custom",
    message="Custom metric",
    metrics={"my_custom_metric": 42.0},
)

# Analyze trend
trend = feedback_system.analyze_trends("my_custom_metric")
```

## Next Steps

- Read [LONG_RUNNING_GUIDE.md](../LONG_RUNNING_GUIDE.md) for operational details
- See [ARCHITECTURE.md](../../ARCHITECTURE.md) for system architecture
- Check [test_enhanced_feedback_system.py](../../tests/test_enhanced_feedback_system.py) for more examples

## Support

For issues or questions:
- Check documentation in `docs/`
- Review examples in `examples/`
- Open an issue on GitHub
