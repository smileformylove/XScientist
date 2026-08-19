#!/usr/bin/env python3
"""Inspect and manage durable autonomous-research feedback."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ai_scientist.enhanced_feedback_system import (
    EnhancedFeedbackSystem,
    FeedbackCategory,
    FeedbackPriority,
)


def cmd_status(args: argparse.Namespace) -> int:
    """Show one compact feedback health summary."""
    feedback_system = EnhancedFeedbackSystem(
        feedback_dir=Path(args.feedback_dir),
    )

    report = feedback_system.get_health_report()

    corrupted = report.get("health_state") == "corrupted"
    if args.as_json:
        print(
            json.dumps(
                {
                    "schema": "xscientist.feedback-status.v1",
                    "ok": not corrupted,
                    **report,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1 if corrupted else 0

    score = report["health_score"]
    if corrupted:
        print("Feedback: corrupted (history is unreadable)")
        for error in report.get("load_errors") or []:
            print(f"Error: {error}")
    elif score is None:
        print("Feedback: unknown (no observations yet)")
    else:
        if score >= 90:
            state = "excellent"
        elif score >= 70:
            state = "good"
        elif score >= 50:
            state = "fair"
        elif score >= 30:
            state = "poor"
        else:
            state = "critical"
        print(f"Feedback: {state} ({score}/100)")

    summary = report["feedback_summary"]
    print(
        f"Items: {summary['total_items']} total, "
        f"{summary['unresolved_items']} unresolved, "
        f"{summary['critical_items']} critical"
    )

    if report.get("trends"):
        print("Metrics:")
        for metric, trend in report["trends"].items():
            if "error" not in trend:
                print(
                    f"  {metric}: mean={trend['mean']:.3f}, "
                    f"trend={trend['trend_direction']}, "
                    f"recent_change={trend['recent_vs_historical_change_pct']:.1f}%"
                )
    return 1 if corrupted else 0


def cmd_actions(args: argparse.Namespace) -> int:
    """Show the highest-priority unresolved feedback actions."""
    feedback_system = EnhancedFeedbackSystem(
        feedback_dir=Path(args.feedback_dir),
    )

    actions = feedback_system.generate_actions(max_actions=args.max_actions)

    errors = list(feedback_system.load_errors)
    if args.as_json:
        print(
            json.dumps(
                {
                    "schema": "xscientist.feedback-actions.v1",
                    "ok": not errors,
                    "actions": actions,
                    "errors": errors,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1 if errors else 0

    if errors:
        print("Cannot recommend actions because feedback history is unreadable.")
        for error in errors:
            print(f"Error: {error}")
        return 1

    if not actions:
        if not feedback_system.feedback_history:
            print("Actions: none (no observations yet; health is unknown)")
        else:
            print("Actions: none (no unresolved feedback requires action)")
        return 0

    for i, action in enumerate(actions, 1):
        priority = action.get("priority", "medium").upper()
        print(f"{i}. [{priority}] {action['action']}")
        if "estimated_impact" in action:
            print(f"   Impact: {action['estimated_impact']*100:.0f}%")
        if "metric" in action:
            print(f"   Metric: {action['metric']}")
    return 0


def cmd_trends(args: argparse.Namespace) -> int:
    """Show metric trends"""
    feedback_system = EnhancedFeedbackSystem(
        feedback_dir=Path(args.feedback_dir),
    )

    metrics = args.metrics or [
        "quality_score",
        "success_rate",
        "error_rate",
    ]

    print("=" * 70)
    print("METRIC TRENDS")
    print("=" * 70)
    print()

    if feedback_system.load_errors:
        print("Cannot analyze trends because feedback history is unreadable.")
        for error in feedback_system.load_errors:
            print(f"  Error: {error}")
        return 1

    for metric in metrics:
        trend = feedback_system.analyze_trends(metric, hours=args.hours)

        if "error" in trend:
            print(f"❌ {metric}: {trend['error']}")
            continue

        print(f"📊 {metric}")
        print(f"   Data Points: {trend['data_points']}")
        print(f"   Time Range: {trend['time_range_hours']:.1f} hours")
        print(f"   Mean: {trend['mean']:.3f}")
        print(f"   Median: {trend['median']:.3f}")
        print(f"   Std Dev: {trend['stdev']:.3f}")
        print(f"   Range: [{trend['min']:.3f}, {trend['max']:.3f}]")
        print(f"   Trend: {trend['trend_direction']} (slope: {trend['slope']:.6f})")
        print(
            f"   Recent vs Historical: {trend['recent_vs_historical_change_pct']:+.1f}%"
        )
        print()

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Export comprehensive report"""
    feedback_system = EnhancedFeedbackSystem(
        feedback_dir=Path(args.feedback_dir),
    )

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.feedback_dir) / f"health_report_{timestamp}.json"

    feedback_system.export_report(output_path)
    print(f"✓ Report exported to: {output_path}")

    if args.show:
        with open(output_path, "r") as f:
            report = json.load(f)
        print("\n" + json.dumps(report, indent=2, ensure_ascii=False))

    return 1 if feedback_system.load_errors else 0


def cmd_add(args: argparse.Namespace) -> int:
    """Add feedback item"""
    feedback_system = EnhancedFeedbackSystem(
        feedback_dir=Path(args.feedback_dir),
    )

    # Parse category and priority
    try:
        category = FeedbackCategory[args.category.upper()]
    except KeyError:
        print(f"Error: Invalid category '{args.category}'")
        print(
            f"Valid categories: {', '.join(c.name.lower() for c in FeedbackCategory)}"
        )
        return 1

    try:
        priority = FeedbackPriority[args.priority.upper()]
    except KeyError:
        print(f"Error: Invalid priority '{args.priority}'")
        print(
            f"Valid priorities: {', '.join(p.name.lower() for p in FeedbackPriority)}"
        )
        return 1

    # Parse metrics
    metrics = {}
    if args.metrics:
        for metric_str in args.metrics:
            try:
                key, value = metric_str.split("=", 1)
                metrics[key] = float(value)
            except (ValueError, IndexError):
                print(f"Warning: Invalid metric format '{metric_str}', skipping")

    # Add feedback
    item = feedback_system.add_feedback(
        category=category,
        priority=priority,
        source=args.source,
        message=args.message,
        metrics=metrics,
        context={},
        actionable=not args.not_actionable,
    )

    print(f"✓ Feedback added:")
    print(f"  Category: {item.category}")
    print(f"  Priority: {item.priority}")
    print(f"  Source: {item.source}")
    print(f"  Message: {item.message}")
    print(f"  Actionable: {item.actionable}")

    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """Clear resolved feedback"""
    feedback_system = EnhancedFeedbackSystem(
        feedback_dir=Path(args.feedback_dir),
    )

    cleared = feedback_system.clear_resolved()
    after = len(
        [item for item in feedback_system.feedback_history if not item.resolved]
    )
    print(f"✓ Cleared {cleared} resolved feedback items")
    print(f"  Remaining: {after} unresolved items")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect durable research feedback and its next actions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--feedback-dir",
        type=str,
        default="./feedback",
        help="Feedback directory (default: ./feedback)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show feedback health")
    status_parser.add_argument("--json", action="store_true", dest="as_json")

    # Actions command
    actions_parser = subparsers.add_parser("actions", help="Show recommended actions")
    actions_parser.add_argument("--json", action="store_true", dest="as_json")
    actions_parser.add_argument(
        "--max-actions",
        type=int,
        default=10,
        help="Maximum number of actions to show (default: 10)",
    )

    # Trends command
    trends_parser = subparsers.add_parser("trends", help="Show metric trends")
    trends_parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to analyze (default: quality_score, success_rate, error_rate)",
    )
    trends_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours to look back (default: 24)",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Export comprehensive report")
    report_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: auto-generated)",
    )
    report_parser.add_argument(
        "--show",
        action="store_true",
        help="Show report after exporting",
    )

    # Add command
    add_parser = subparsers.add_parser("add", help="Add feedback item")
    add_parser.add_argument(
        "--category",
        required=True,
        help="Feedback category (quality, performance, resource, error, success, strategy)",
    )
    add_parser.add_argument(
        "--priority",
        required=True,
        help="Priority level (critical, high, medium, low, info)",
    )
    add_parser.add_argument(
        "--source",
        required=True,
        help="Feedback source",
    )
    add_parser.add_argument(
        "--message",
        required=True,
        help="Feedback message",
    )
    add_parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics in key=value format",
    )
    add_parser.add_argument(
        "--not-actionable",
        action="store_true",
        help="Mark as not actionable",
    )

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear resolved feedback")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "status": cmd_status,
        "actions": cmd_actions,
        "trends": cmd_trends,
        "report": cmd_report,
        "add": cmd_add,
        "clear": cmd_clear,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    print(f"Error: Unknown command '{args.command}'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
