from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.treesearch.utils import tree_export


class TreeExportAtomicTests(unittest.TestCase):
    def test_tree_data_failure_preserves_previous_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stage_dir = Path(td) / "logs" / "run" / "stage_demo"
            stage_dir.mkdir(parents=True)
            out_path = stage_dir / "tree_plot.html"
            data_path = stage_dir / "tree_data.json"
            out_path.write_text("previous-html", encoding="utf-8")
            data_path.write_text('{"previous":true}', encoding="utf-8")

            with (
                mock.patch.object(
                    tree_export, "cfg_to_tree_struct", return_value={"new": True}
                ),
                mock.patch.object(
                    tree_export,
                    "atomic_write_json",
                    side_effect=OSError("disk busy"),
                ),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                tree_export.generate(mock.Mock(), mock.Mock(), out_path)

            self.assertEqual(
                data_path.read_text(encoding="utf-8"), '{"previous":true}'
            )
            self.assertEqual(out_path.read_text(encoding="utf-8"), "previous-html")

    def test_html_failure_preserves_previous_html(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stage_dir = Path(td) / "logs" / "run" / "stage_demo"
            stage_dir.mkdir(parents=True)
            out_path = stage_dir / "tree_plot.html"
            out_path.write_text("previous-html", encoding="utf-8")
            real_atomic_write_text = tree_export.atomic_write_text

            def fail_stage_html(path, content, *, encoding="utf-8"):
                if Path(path) == out_path:
                    raise OSError("html disk busy")
                return real_atomic_write_text(path, content, encoding=encoding)

            with (
                mock.patch.object(
                    tree_export, "cfg_to_tree_struct", return_value={"new": True}
                ),
                mock.patch.object(tree_export, "generate_html", return_value="new-html"),
                mock.patch.object(
                    tree_export,
                    "atomic_write_text",
                    side_effect=fail_stage_html,
                ),
                self.assertRaisesRegex(OSError, "html disk busy"),
            ):
                tree_export.generate(mock.Mock(), mock.Mock(), out_path)

            self.assertEqual(out_path.read_text(encoding="utf-8"), "previous-html")

    def test_unified_visualization_uses_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "logs" / "run"
            stage_dir = log_dir / "stage_1_demo"
            stage_dir.mkdir(parents=True)
            stage_viz = stage_dir / "tree_plot.html"
            (stage_dir / "tree_data.json").write_text(
                '{"layout":[],"edges":[]}', encoding="utf-8"
            )

            with mock.patch.object(tree_export, "atomic_write_text") as write_mock:
                tree_export.create_unified_viz(mock.Mock(), stage_viz)

            self.assertEqual(
                Path(write_mock.call_args.args[0]), log_dir / "unified_tree_viz.html"
            )


if __name__ == "__main__":
    unittest.main()
