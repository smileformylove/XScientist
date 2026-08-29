from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from omegaconf import OmegaConf

from ai_scientist.resources import resolve_bfts_config_path
from ai_scientist.treesearch.agent_manager import AgentManager
from ai_scientist.treesearch import bfts_utils


class BftsUtilsAtomicTests(unittest.TestCase):
    def test_machine_task_descriptor_reaches_agent_with_binding_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "bfts_task.json"
            idea = {
                "Title": "Confounding probe",
                "Abstract": "Test the apparent effect.",
                "Short Hypothesis": "The pooled effect is confounded.",
                "Experiments": ["pooled comparison", "stratified comparison"],
                "Risk Factors and Limitations": ["observational assignment"],
            }
            bfts_utils.write_bfts_task_descriptor(
                idea,
                str(output),
                research_plan={
                    "plan_id": "plan-1",
                    "workflow_mode": "program_driven",
                    "tasks": [{"task_id": "stratify", "owner": "experiment"}],
                    "required_discriminating_tests": ["stratified comparison"],
                    "acceptance_rules": ["do not infer causality from pooling"],
                },
            )

            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
            cfg.log_dir = root / "logs"
            cfg.workspace_dir = root / "workspace"
            manager = AgentManager(
                task_desc=output.read_text(encoding="utf-8"),
                cfg=cfg,
                workspace_dir=root / "workspace",
            )

            prompt = manager._get_task_desc_str()
            self.assertIn("Binding XScientist Research Contract", prompt)
            self.assertIn("stratified comparison", prompt)
            self.assertIn("do not infer causality from pooling", prompt)
            self.assertIn("not optional context", prompt)

    def test_idea_markdown_embeds_binding_research_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "idea.md"
            bfts_utils.idea_to_markdown(
                {"title": "H1"},
                str(output),
                None,
                research_plan={
                    "plan_id": "plan-1",
                    "tasks": [{"task_id": "t1", "metric": "accuracy"}],
                    "required_discriminating_tests": ["negative control"],
                    "acceptance_rules": ["predeclared metric improves"],
                },
            )

            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Binding Research Contract", rendered)
            self.assertIn("negative control", rendered)
            self.assertIn("not optional context", rendered)

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
