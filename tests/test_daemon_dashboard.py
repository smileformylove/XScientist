from __future__ import annotations

import unittest

from ai_scientist.apps.daemon_dashboard import (
    _build_live_dashboard_html,
    _build_recent_trend_metrics,
    _html_escape,
)


class DaemonDashboardTests(unittest.TestCase):
    def test_recent_trend_metrics_extracts_numeric_series(self) -> None:
        metrics = _build_recent_trend_metrics(
            [
                {
                    "returncode": 0,
                    "duration_seconds": 12,
                    "views": {"submission_board_items": 2},
                    "active_source_feedback": {
                        "avg_experiment_todo_closure_rate": 0.75,
                        "avg_experiment_todo": 3,
                    },
                },
                {
                    "returncode": 1,
                    "duration_seconds": 18,
                    "views": {"submission_board_items": 4},
                    "active_source_feedback": {
                        "avg_experiment_todo_closure_rate": "unknown"
                    },
                },
            ]
        )

        self.assertEqual(metrics["submission_board_items"], [2.0, 4.0])
        self.assertEqual(metrics["duration_seconds"], [12.0, 18.0])
        self.assertEqual(metrics["returncode"], [0.0, 1.0])
        self.assertEqual(metrics["experiment_todo_closure_rate"], [0.75])
        self.assertEqual(metrics["experiment_todo_backlog"], [3.0])

    def test_html_escape_handles_dashboard_content(self) -> None:
        self.assertEqual(
            _html_escape('<script data-x="1">&</script>'),
            "&lt;script data-x=&quot;1&quot;&gt;&amp;&lt;/script&gt;",
        )

    def test_dashboard_html_escapes_runtime_values(self) -> None:
        html = _build_live_dashboard_html(
            {
                "generated_at": "now",
                "refresh_seconds": 30,
                "daemon_status": {
                    "guardrail_phase": "<unsafe>",
                    "guardrail_mode": "steady",
                    "current_daypart": "day",
                    "control": {},
                },
                "cycle_summary": {},
                "daily_summary": {},
                "operator_brief": {},
                "trend_metrics": {},
            }
        )

        self.assertIn("XScientist Daemon Dashboard", html)
        self.assertIn("phase=&lt;unsafe&gt;", html)
        self.assertNotIn("phase=<unsafe>", html)


if __name__ == "__main__":
    unittest.main()
