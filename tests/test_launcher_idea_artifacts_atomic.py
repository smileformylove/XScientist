from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.utils import launcher_workflow


class LauncherIdeaArtifactsAtomicTests(unittest.TestCase):
    def test_prepare_idea_artifacts_writes_json_and_marks_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idea_dir = Path(td) / "idea"
            idea_dir.mkdir()
            ideas = [{"Name": "Demo", "Description": "测试"}]

            with (
                mock.patch.object(launcher_workflow, "idea_to_markdown"),
                mock.patch.object(launcher_workflow, "mark_stage_complete") as mark,
            ):
                idea, idea_path = launcher_workflow.prepare_idea_artifacts(
                    ideas, 0, str(Path(td) / "ideas.json"), idea_dir
                )

            self.assertIs(idea, ideas[0])
            self.assertEqual(
                json.loads(Path(idea_path).read_text(encoding="utf-8")), ideas[0]
            )
            mark.assert_called_once()

    def test_json_publish_failure_preserves_file_and_does_not_mark_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idea_dir = Path(td) / "idea"
            idea_dir.mkdir()
            idea_path = idea_dir / "idea.json"
            idea_path.write_text("previous", encoding="utf-8")

            with (
                mock.patch.object(launcher_workflow, "idea_to_markdown"),
                mock.patch.object(
                    launcher_workflow,
                    "atomic_write_json",
                    side_effect=OSError("disk busy"),
                ),
                mock.patch.object(launcher_workflow, "mark_stage_complete") as mark,
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                launcher_workflow.prepare_idea_artifacts(
                    [{"Name": "New"}],
                    0,
                    str(Path(td) / "ideas.json"),
                    idea_dir,
                )

            self.assertEqual(idea_path.read_text(encoding="utf-8"), "previous")
            mark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
