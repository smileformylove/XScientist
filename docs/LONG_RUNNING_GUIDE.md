# Long-Running Operations Guide

## Overview

This guide covers best practices for running XScientist in long-running, production-like scenarios, including daemon mode, monitoring, and troubleshooting.

## Table of Contents

- [Daemon Mode](#daemon-mode)
- [Monitoring & Health Checks](#monitoring--health-checks)
- [Feedback Mechanisms](#feedback-mechanisms)
- [Resource Management](#resource-management)
- [Error Recovery](#error-recovery)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Daemon Mode

### Starting the Daemon

The daemon is designed for continuous, autonomous operation:

```bash
python3 continuous_research_daemon.py \
  --source-config configs/sources/stable_source_priority.example.json \
  --duration-hours 24 \
  --enable-rewrite-followup \
  --auto-source-quality-feedback \
  --auto-quality-strategy-feedback \
  --auto-quality-governor \
  --auto-evidence-strategy-feedback \
  --auto-export-submission-dossier \
  --auto-failure-guard \
  --serve-dashboard \
  -- --submission-mode --num-ideas 3
```

### Key Daemon Features

#### 1. Auto Feedback Loops

- `--auto-source-quality-feedback`: Adjusts source priorities based on quality metrics
- `--auto-quality-strategy-feedback`: Adapts quality strategies based on outcomes
- `--auto-evidence-strategy-feedback`: Optimizes evidence collection strategies
- `--auto-quality-governor`: Enforces quality gates dynamically

#### 2. Failure Protection

- `--auto-failure-guard`: Automatic recovery from failures
- Graceful degradation on resource exhaustion
- Checkpoint-based recovery
- Timeout protection at all levels

#### 3. Dashboard & Monitoring

- `--serve-dashboard`: Web dashboard for real-time monitoring
- Access at `http://localhost:8080` (default)
- Real-time metrics, logs, and status

### Daemon Management Commands

```bash
# Check daemon status
bash run_stable_daemon.sh status

# Get brief summary
bash run_stable_daemon.sh brief

# Generate handoff report
bash run_stable_daemon.sh handoff

# View trend reports
bash run_stable_daemon.sh report-trends

# Review source planning
bash run_stable_daemon.sh source-plan
```

---

## Monitoring & Health Checks

### Built-in Health Monitoring

The enhanced feedback system provides comprehensive health monitoring:

```python
from ai_scientist.enhanced_feedback_system import (
    EnhancedFeedbackSystem,
    LongRunningTaskMonitor,
    FeedbackCategory,
    FeedbackPriority,
)

# Initialize feedback system
feedback_system = EnhancedFeedbackSystem(
    feedback_dir=Path("./feedback"),
    window_size=100,
    trend_window_hours=24,
)

# Create task monitor
monitor = LongRunningTaskMonitor(
    task_name="research_project",
    feedback_system=feedback_system,
    heartbeat_interval=300,  # 5 minutes
    stall_threshold=1800,    # 30 minutes
)

# Send heartbeats
monitor.heartbeat(progress=0.5, status="running")

# Create checkpoints
monitor.checkpoint("experiments_complete", metadata={"experiments": 10})

# Get health report
health_report = feedback_system.get_health_report()
print(f"Health Score: {health_report['health_score']}/100")
```

### Key Metrics to Monitor

#### Quality Metrics
- `quality_score`: Overall paper quality (1-5 scale)
- `review_gate_ready_rate`: Percentage passing quality gates
- `self_review_coverage`: Review coverage percentage

#### Performance Metrics
- `success_rate`: Project success rate
- `error_rate`: Error occurrence rate
- `completion_time`: Average project completion time

#### Resource Metrics
- `token_usage`: API token consumption
- `memory_usage`: Memory utilization
- `disk_usage`: Disk space usage

#### Evidence Metrics
- `experiment_todo_closure_rate`: TODO completion rate
- `evidence_depth`: Evidence quality score
- `figure_traceability`: Figure-to-claim binding rate

### Health Score Calculation

The system calculates a health score (0-100) based on:
- Critical unresolved issues (-10 per issue)
- Error rate (-50 * error_rate)
- Success rate (+20 * (success_rate - 0.5))
- Quality score (+10 * (quality_score - 3.0))

**Interpretation**:
- 90-100: Excellent health
- 70-89: Good health
- 50-69: Fair health, monitor closely
- 30-49: Poor health, intervention needed
- 0-29: Critical health, immediate action required

---

## Feedback Mechanisms

### Multi-Source Feedback

The system collects feedback from multiple sources:

1. **Self-Reflection**: Autonomous analysis of own performance
2. **Metrics**: Quantitative performance indicators
3. **External Agents**: Feedback from external systems
4. **Peer Review**: Simulated peer review feedback
5. **Human**: Manual feedback from users

### Feedback Priority Levels

- **CRITICAL**: Requires immediate action (e.g., high error rate)
- **HIGH**: Should be addressed soon (e.g., declining quality)
- **MEDIUM**: Normal priority (e.g., strategy adjustments)
- **LOW**: Nice to have (e.g., minor optimizations)
- **INFO**: Informational only (e.g., checkpoints)

### Feedback Categories

- **QUALITY**: Quality-related feedback
- **PERFORMANCE**: Performance metrics
- **RESOURCE**: Resource usage
- **ERROR**: Error and failure feedback
- **SUCCESS**: Success patterns
- **STRATEGY**: Strategy adjustments

### Adding Custom Feedback

```python
feedback_system.add_feedback(
    category=FeedbackCategory.QUALITY,
    priority=FeedbackPriority.HIGH,
    source="custom_validator",
    message="Paper lacks sufficient ablation studies",
    metrics={"ablation_count": 2, "expected_min": 5},
    context={"paper_id": "proj_123", "section": "experiments"},
    actionable=True,
)
```

### Automated Action Generation

The system automatically generates actions based on feedback:

```python
# Generate prioritized actions
actions = feedback_system.generate_actions(max_actions=5)

for action in actions:
    print(f"Priority: {action['priority']}")
    print(f"Action: {action['action']}")
    print(f"Estimated Impact: {action['estimated_impact']}")
```

---

## Resource Management

### Token Budget Management

Monitor and control API token usage:

```python
from ai_scientist.utils.token_tracker import TokenTracker

tracker = TokenTracker()
tracker.set_budget(max_tokens=1000000, max_cost=100.0)

# Check budget status
status = tracker.get_status()
if status['budget_exceeded']:
    print("Budget exceeded, throttling requests")
```

### Memory Management

- Use streaming for large outputs
- Clear intermediate artifacts periodically
- Implement checkpointing for long experiments

### Disk Space Management

Monitor output directory size:

```bash
# Check output directory size
du -sh $RESEARCH_OUTPUT_DIR

# Clean old artifacts (be careful!)
find $RESEARCH_OUTPUT_DIR -name "*.tmp" -mtime +7 -delete
```

### Parallelism Control

Adjust parallelism based on resources:

```python
# In daemon config
{
  "max_parallel_projects": 2,
  "max_parallel_experiments": 4,
  "max_parallel_reviews": 2
}
```

---

## Error Recovery

### Automatic Recovery Mechanisms

#### 1. Timeout Protection

All operations have timeouts:
- Experiment execution: 3600s (configurable)
- Review rounds: 1800s
- Writing operations: 900s

#### 2. Checkpoint-Based Recovery

Projects save checkpoints at key stages:
- After ideation
- After each experiment
- After each review round
- After quality gates

#### 3. Graceful Degradation

When resources are constrained:
- Reduce parallelism
- Skip non-critical operations
- Use fallback strategies

### Manual Recovery

#### Recovering a Failed Project

```bash
# Check project status
python3 research_manager.py process-board --status failed

# Resume from checkpoint
python3 run_project.py <project_name> \
  --resume-from-checkpoint \
  --checkpoint-stage experiments
```

#### Clearing Stuck Projects

```bash
# Identify stuck projects
python3 research_manager.py process-board --status blocked

# Force cleanup
python3 research_manager.py cleanup-project <project_name>
```

---

## Performance Optimization

### Prompt Caching

Enable prompt caching for repeated contexts:

```python
# Automatically enabled for supported models
# Reduces token usage by 50-90% for repeated prompts
```

### Parallel Execution

Maximize throughput with parallelism:

```bash
# Run multiple projects in parallel
python3 continuous_research_daemon.py \
  --max-parallel-projects 3 \
  --max-parallel-experiments 8
```

### Experiment Optimization

Optimize experiment execution:

```yaml
# In bfts_config.yaml
max_depth: 5  # Reduce for faster experiments
max_iterations: 20  # Limit iterations
timeout_per_step: 300  # Shorter timeouts
```

### Review Optimization

Optimize review rounds:

```bash
# Limit review rounds
python3 run_project.py <project> \
  --max-review-rounds 3 \
  --review-timeout 1200
```

---

## Troubleshooting

### Common Issues

#### 1. High Error Rate

**Symptoms**: Error rate > 20%

**Diagnosis**:
```python
# Check error trends
trend = feedback_system.analyze_trends("error_rate")
print(trend)
```

**Solutions**:
- Increase timeouts
- Add retry logic
- Check API rate limits
- Verify network connectivity

#### 2. Low Quality Scores

**Symptoms**: Quality score < 3.0

**Diagnosis**:
```bash
# Review failed quality gates
python3 research_manager.py process-board --status quality_failed
```

**Solutions**:
- Increase review rounds
- Strengthen quality gates
- Improve writing prompts
- Add more evidence requirements

#### 3. Stalled Tasks

**Symptoms**: No progress for > 30 minutes

**Diagnosis**:
```python
# Check task status
status = monitor.get_status()
if status['is_stalled']:
    print("Task is stalled")
```

**Solutions**:
- Check for deadlocks
- Verify resource availability
- Review logs for errors
- Restart with checkpoint

#### 4. Resource Exhaustion

**Symptoms**: Out of memory, disk full, token budget exceeded

**Diagnosis**:
```bash
# Check resource usage
df -h $RESEARCH_OUTPUT_DIR
free -h
```

**Solutions**:
- Clean old artifacts
- Reduce parallelism
- Increase resource limits
- Implement throttling

### Debug Mode

Enable verbose logging:

```bash
export DEBUG=1
export LOG_LEVEL=DEBUG

python3 continuous_research_daemon.py --verbose
```

### Log Analysis

Key log files:
- `daemon.log`: Daemon operations
- `project_<name>.log`: Per-project logs
- `feedback_batch_*.json`: Feedback history
- `evolution_history.json`: Evolution records

### Getting Help

1. Check documentation: `docs/`
2. Review examples: `examples/`
3. Run preflight checks: `python3 preflight_check.py --strict`
4. Open an issue: https://github.com/YOUR_ORG/ai_scientist/issues

---

## Best Practices

### For Long-Running Daemons

1. **Start with a dry run**: Test configuration with `--duration-hours 1`
2. **Monitor actively**: Check dashboard regularly in first 24 hours
3. **Set conservative budgets**: Start with lower token/cost limits
4. **Enable all feedback loops**: Use all `--auto-*-feedback` flags
5. **Use failure guards**: Always enable `--auto-failure-guard`
6. **Schedule maintenance**: Plan for periodic restarts
7. **Archive regularly**: Export results periodically

### For Production Deployments

1. **Use dedicated output directory**: Set `RESEARCH_OUTPUT_DIR`
2. **Implement backup strategy**: Regular backups of output directory
3. **Monitor resource usage**: Set up alerts for resource thresholds
4. **Rotate logs**: Implement log rotation
5. **Version control configs**: Track configuration changes
6. **Document customizations**: Maintain deployment documentation
7. **Test recovery procedures**: Regularly test checkpoint recovery

### For Optimal Quality

1. **Use submission mode**: Enable `--submission-mode` for best quality
2. **Increase review rounds**: Use `--max-review-rounds 5` or higher
3. **Enable rewrite followup**: Use `--enable-rewrite-followup`
4. **Strengthen evidence requirements**: Configure stricter TODO gates
5. **Use quality governor**: Enable `--auto-quality-governor`
6. **Review playbook regularly**: Check self-evolution insights

---

## Appendix

### Configuration Templates

See `configs/sources/` for source configuration templates.

### Metric Definitions

See `ARCHITECTURE.md` for detailed metric definitions.

### API Reference

See individual module docstrings for API details.

### Performance Benchmarks

Typical performance on standard hardware:
- Ideation: 5-10 minutes
- Experiments: 30-60 minutes
- Writing: 10-20 minutes
- Review: 15-30 minutes per round
- Total per project: 2-4 hours

### Resource Requirements

Minimum recommended:
- CPU: 4 cores
- RAM: 16 GB
- Disk: 100 GB free
- Network: Stable internet connection

Recommended for production:
- CPU: 8+ cores
- RAM: 32+ GB
- Disk: 500+ GB free
- Network: High-bandwidth, low-latency
