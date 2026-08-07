from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from ai_scientist.treesearch import bfts_utils


class BftsUtilsAtomicTests(unittest.TestCase):
    def test_markdown_output_format_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "idea.md"
            code_path = root / "seed.py"
            code_path.write_text("print('hello')\n", encoding="utf-8")

            bfts_utils.idea_to_markdown(
                {
                    "research_goal": "Demo",
                    "tags": ["one", "two"],
                    "details": {"Metric": "accuracy"},
                },
                str(output),
                str(code_path),
            )

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "## Research Goal\n\n"
                "Demo\n\n"
                "## Tags\n\n"
                "- one\n"
                "- two\n\n"
                "## Details\n\n"
                "### Metric\n"
                "accuracy\n\n"
                "## Code To Potentially Use\n\n"
                "Use the following code as context for your experiments:\n\n"
                "```python\n"
                "print('hello')\n"
                "\n```\n\n",
            )

    def test_missing_code_preserves_existing_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "idea.md"
            output.write_text("previous", encoding="utf-8")

            with self.assertRaises(AssertionError):
                bfts_utils.idea_to_markdown(
                    {"title": "new"}, str(output), str(Path(td) / "missing.py")
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "previous")

    def test_markdown_atomic_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "idea.md"
            output.write_text("previous", encoding="utf-8")

            with (
                mock.patch.object(
                    bfts_utils,
                    "atomic_write_text",
                    side_effect=OSError("disk busy"),
                ),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                bfts_utils.idea_to_markdown({"title": "new"}, str(output), None)

            self.assertEqual(output.read_text(encoding="utf-8"), "previous")

    def test_config_atomic_failure_leaves_no_partial_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template = root / "template.yaml"
            template.write_text("goal: demo\n", encoding="utf-8")
            idea_dir = root / "idea"
            idea_dir.mkdir()
            idea_path = idea_dir / "idea.json"
            idea_path.write_text("{}", encoding="utf-8")

            with (
                mock.patch.object(
                    bfts_utils,
                    "atomic_write_text",
                    side_effect=OSError("disk busy"),
                ),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                bfts_utils.edit_bfts_config_file(
                    str(template), str(idea_dir), str(idea_path)
                )

            config_dir = idea_dir / ".xscientist" / "configs"
            self.assertEqual(list(config_dir.glob("bfts_config-*.yaml")), [])

    def test_generated_config_remains_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template = root / "template.yaml"
            template.write_text("goal: demo\n", encoding="utf-8")
            idea_dir = root / "idea"
            idea_dir.mkdir()
            idea_path = idea_dir / "idea.json"
            idea_path.write_text("{}", encoding="utf-8")

            config_path = Path(
                bfts_utils.edit_bfts_config_file(
                    str(template), str(idea_dir), str(idea_path)
                )
            )

            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["goal"], "demo")
            self.assertEqual(payload["workspace_dir"], "../..")
            self.assertEqual(payload["desc_file"], "../../idea.json")
            self.assertEqual(payload["data_dir"], "../../data")
            self.assertEqual(payload["log_dir"], "../../logs")
            self.assertIsNone(payload["resume_from"])


if __name__ == "__main__":
    unittest.main()
