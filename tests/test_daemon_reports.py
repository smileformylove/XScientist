from __future__ import annotations

import unittest

from ai_scientist.apps.daemon_reports import (
    _build_cycle_summary_markdown,
    _build_failure_guard_markdown,
    _build_primary_action_queue_markdown,
    _build_report_archive_index_markdown,
    _build_source_health_board_markdown,
)


class DaemonReportTests(unittest.TestCase):
    def test_source_health_board_orders_highest_score_first(self) -> None:
        markdown = _build_source_health_board_markdown(
            [
                {"name": "low", "health_score": 20},
                {"name": "high", "health_score": 90},
            ]
        )

        self.assertLess(markdown.index("high"), markdown.index("low"))

    def test_primary_action_queue_includes_commands(self) -> None:
        markdown = _build_primary_action_queue_markdown(
            [
                {
                    "priority": "P0",
                    "label": "recover",
                    "reason": "health risk",
                    "command": "xscientist daemon recover",
                }
            ]
        )

        self.assertIn("[P0] recover", markdown)
        self.assertIn("`xscientist daemon recover`", markdown)

    def test_cycle_summary_preserves_operational_metrics(self) -> None:
        markdown = _build_cycle_summary_markdown(
            {
                "generated_at": "now",
                "health": {"score": 88, "state": "healthy"},
                "guardrail_phase": "steady",
                "active_source_todo_closure_rate": 0.8,
                "active_source_todo_backlog": 2,
            }
        )

        self.assertIn("Health: 88 (healthy)", markdown)
        self.assertIn("Active source TODO closure: 0.8", markdown)
        self.assertIn("Active source TODO backlog: 2", markdown)

    def test_report_index_links_archived_reports(self) -> None:
        markdown = _build_report_archive_index_markdown(
            {
                "generated_at": "now",
                "counts": {"daily": 1, "handoff": 1},
                "entries": [
                    {
                        "kind": "daily",
                        "name": "daily.md",
                        "path": "/tmp/daily.md",
                    },
                    {
                        "kind": "handoff",
                        "name": "handoff.md",
                        "path": "/tmp/handoff.md",
                    },
                ],
            }
        )

        self.assertIn("Daily reports: 1", markdown)
        self.assertIn("Handoff reports: 1", markdown)
        self.assertIn("/tmp/daily.md", markdown)
        self.assertIn("/tmp/handoff.md", markdown)

    def test_failure_guard_reports_applied_state(self) -> None:
        markdown = _build_failure_guard_markdown(
            {
                "enabled": True,
                "source": "papers",
                "consecutive_failures": 3,
                "threshold": 2,
                "cooldown_cycles": 4,
                "applied": True,
                "reason": "threshold reached",
            }
        )

        self.assertIn("Applied: True", markdown)
        self.assertIn("Reason: threshold reached", markdown)


if __name__ == "__main__":
    unittest.main()
