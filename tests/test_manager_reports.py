from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.apps.manager import ResearchManager
from ai_scientist.apps.manager_reports import (
    render_repair_board_markdown,
    render_rewrite_board_markdown,
    render_shortlist_markdown,
    render_submission_board_markdown,
)


class ManagerReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paper = {
            "name": "demo-paper",
            "type": "journal",
            "target_venue": "iclr",
            "submission_priority_score": 88,
            "submission_priority_tier": "submit_now",
            "suggested_next_step": "run ablation",
            "repair_id": "R1",
            "project": "demo-project",
        }

    def test_renderers_generate_all_manager_board_formats(self) -> None:
        self.assertIn(
            "# Submission Board",
            render_submission_board_markdown({"iclr": [self.paper]}),
        )
        self.assertIn(
            "Next Step: run ablation",
            render_rewrite_board_markdown([self.paper]),
        )
        self.assertIn(
            "demo-paper :: R1",
            render_repair_board_markdown([self.paper]),
        )
        self.assertIn(
            "# Submission Shortlist",
            render_shortlist_markdown([self.paper]),
        )

    def test_manager_exports_write_exact_renderer_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manager = ResearchManager(td)
            root = Path(td)
            cases = [
                (
                    manager.export_submission_board_markdown,
                    {"iclr": [self.paper]},
                    render_submission_board_markdown({"iclr": [self.paper]}),
                ),
                (
                    manager.export_rewrite_board_markdown,
                    [self.paper],
                    render_rewrite_board_markdown([self.paper]),
                ),
                (
                    manager.export_repair_board_markdown,
                    [self.paper],
                    render_repair_board_markdown([self.paper]),
                ),
                (
                    manager.export_shortlist_markdown,
                    [self.paper],
                    render_shortlist_markdown([self.paper]),
                ),
            ]
            for index, (exporter, payload, expected) in enumerate(cases):
                with self.subTest(exporter=exporter.__name__):
                    output = root / "nested" / f"{index}.md"
                    self.assertEqual(exporter(payload, str(output)), str(output))
                    self.assertEqual(output.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
