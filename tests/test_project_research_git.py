from __future__ import annotations

import argparse
import shutil
import tempfile
import unittest
from pathlib import Path

from ai_scientist.apps.project import (
    _initialize_local_research_git,
    _record_local_research_checkpoint,
)
from xscientist.research_git import research_log


@unittest.skipUnless(shutil.which("git"), "Git is required for research history tests")
class ProjectResearchGitIntegrationTests(unittest.TestCase):
    def _args(self, root: Path, topic: Path, *, policy: str) -> argparse.Namespace:
        return argparse.Namespace(
            project_dir=str(root),
            topic=str(topic),
            research_git="local",
            git_checkpoint_policy=policy,
            research_git_strict=True,
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )

    def test_milestone_policy_records_experiment_but_not_ideation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "project"
            root.mkdir()
            topic = base / "topic.md"
            topic.write_text("# Question\n\nDoes H1 hold?\n", encoding="utf-8")
            args = self._args(root, topic, policy="milestone")
            args._research_git_active = _initialize_local_research_git(args)
            ideas = root / "01_ideas" / "ideas.json"
            ideas.parent.mkdir()
            ideas.write_text("[]\n", encoding="utf-8")

            _record_local_research_checkpoint(
                args,
                stage="ideation",
                subject="record candidates",
                summary="Idea state.",
            )
            metrics = root / "02_experiments" / "run-1" / "metrics.json"
            metrics.parent.mkdir(parents=True)
            metrics.write_text('{"metric":{"value":1.0}}\n', encoding="utf-8")
            _record_local_research_checkpoint(
                args,
                stage="experiment",
                subject="complete run-1",
                summary="Experiment state.",
            )

            stages = [
                (entry["trailers"].get("Research-Stage") or [None])[0]
                for entry in research_log(root)
            ]
            self.assertEqual(stages, ["experiment", "init"])
            self.assertTrue(ideas.exists(), "skipped ideation output must be preserved")

    def test_stage_policy_records_ideation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "project"
            root.mkdir()
            topic = base / "topic.md"
            topic.write_text("# Question\n", encoding="utf-8")
            args = self._args(root, topic, policy="stage")
            args._research_git_active = _initialize_local_research_git(args)
            ideas = root / "01_ideas" / "ideas.json"
            ideas.parent.mkdir()
            ideas.write_text("[]\n", encoding="utf-8")

            _record_local_research_checkpoint(
                args,
                stage="ideation",
                subject="record candidates",
                summary="Idea state.",
            )

            latest = research_log(root)[0]
            self.assertEqual(latest["trailers"]["Research-Stage"], ["ideation"])


if __name__ == "__main__":
    unittest.main()
