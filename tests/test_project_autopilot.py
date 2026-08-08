from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_scientist.apps.project import (
    _completed_resume_results,
    _prepare_project_input,
    _prepare_autopilot_bfts_config,
    _resolve_resume_work,
    _run_autopilot_preflight,
    _save_project_progress,
    main,
)


class ProjectAutopilotTests(unittest.TestCase):
    def test_plain_language_question_materializes_reproducible_topic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dirs = {
                "root": root,
                "ideas": root / "01_ideas",
                "experiments": root / "02_experiments",
                "papers": root / "03_papers",
                "logs": root / "04_logs",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
            args = argparse.Namespace(
                question="Why does a promising mechanism fail out of distribution?",
                topic=None,
                ideas=None,
                resume=False,
                skip_ideation=False,
            )

            _prepare_project_input(args, dirs)

            topic = Path(args.topic)
            self.assertEqual(topic, root / "00_config" / "topic.md")
            self.assertIn("fail out of distribution", topic.read_text(encoding="utf-8"))

    def test_resume_skips_success_and_reuses_only_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = (
                root
                / "02_experiments"
                / "run_1"
                / "logs"
                / "bfts"
                / "stage_0"
                / "checkpoint.json"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")
            _save_project_progress(
                root,
                results=[
                    {"idea_idx": 0, "status": "success"},
                    {
                        "idea_idx": 1,
                        "status": "budget_exhausted",
                        "checkpoint_path": str(checkpoint),
                    },
                    {
                        "idea_idx": 2,
                        "status": "failed",
                        "checkpoint_path": str(root / "missing.json"),
                    },
                ],
                total=3,
                selected_indices=[0, 1, 2],
            )

            pending, prior, checkpoints = _resolve_resume_work(
                root, [0, 1, 2], enabled=True
            )

            self.assertEqual(pending, [1, 2])
            self.assertEqual([item["idea_idx"] for item in prior], [0])
            self.assertEqual(checkpoints, {1: str(checkpoint)})

    def test_resume_refuses_to_change_original_question(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dirs = {
                "root": root,
                "ideas": root / "01_ideas",
                "experiments": root / "02_experiments",
                "papers": root / "03_papers",
                "logs": root / "04_logs",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
            topic = root / "00_config" / "topic.md"
            topic.parent.mkdir(parents=True)
            topic.write_text("# Research question\n\nOriginal\n", encoding="utf-8")
            args = argparse.Namespace(
                question="Changed",
                topic=None,
                ideas=None,
                resume=True,
                skip_ideation=False,
            )

            with self.assertRaisesRegex(ValueError, "differs"):
                _prepare_project_input(args, dirs)

    def test_completed_resume_is_idempotent_and_needs_no_new_model_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _save_project_progress(
                root,
                results=[
                    {"idea_idx": 0, "status": "success", "quality_score": 8.0},
                    {"idea_idx": 2, "status": "success", "quality_score": 7.5},
                ],
                total=2,
                selected_indices=[0, 2],
            )

            completed = _completed_resume_results(root)

            self.assertEqual([item["idea_idx"] for item in completed or []], [0, 2])

    def test_autopilot_preflight_fails_before_execution_when_isolation_is_missing(
        self,
    ) -> None:
        from ai_scientist.apps.preflight import CheckResult

        args = argparse.Namespace(autopilot="balanced", bfts_config="demo.yaml")
        with mock.patch(
            "ai_scientist.apps.preflight.check_bfts_config",
            return_value=[
                CheckResult(
                    label="Experiment isolation",
                    ok=False,
                    severity="error",
                    detail="docker image is unavailable",
                )
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "before any research-model call"):
                _run_autopilot_preflight(args)

    def test_autopilot_derives_finite_isolated_bfts_budget(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "exec": {
                            "backend": "auto",
                            "require_isolation": False,
                            "docker_image": "xscientist-exec:latest",
                        },
                        "llm_budget": {
                            "max_total_tokens": None,
                            "max_wall_time_seconds": None,
                        },
                        "agent": {
                            "steps": 20,
                            "stages": {
                                "stage1_max_iters": 100,
                                "stage2_max_iters": 100,
                                "stage3_max_iters": 100,
                                "stage4_max_iters": 100,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                autopilot="balanced",
                bfts_config=str(source),
                project_dir=str(root / "project"),
            )

            derived_path = _prepare_autopilot_bfts_config(args)
            derived = yaml.safe_load(
                Path(derived_path or "").read_text(encoding="utf-8")
            )

            self.assertTrue(derived["exec"]["require_isolation"])
            self.assertEqual(derived["exec"]["network"], "none")
            self.assertFalse(derived["exec"]["allow_experiment_network"])
            self.assertEqual(derived["llm_budget"]["max_total_tokens"], 800_000)
            self.assertEqual(derived["llm_budget"]["max_wall_time_seconds"], 14_400)
            self.assertEqual(derived["agent"]["steps"], 3)
            self.assertEqual(derived["agent"]["stages"]["stage1_max_iters"], 8)

    def test_derived_autopilot_config_is_accepted_by_bfts_schema(self) -> None:
        from ai_scientist.resources import resolve_bfts_config_path
        from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file
        from ai_scientist.treesearch.utils.config import load_cfg

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = argparse.Namespace(
                autopilot="discovery",
                bfts_config=str(resolve_bfts_config_path("bfts_config.yaml")),
                project_dir=str(root / "project"),
            )
            derived = _prepare_autopilot_bfts_config(args)
            exp_dir = root / "experiment"
            exp_dir.mkdir()
            idea = exp_dir / "idea.md"
            idea.write_text("# Test idea\n", encoding="utf-8")
            run_config = edit_bfts_config_file(str(derived), str(exp_dir), str(idea))

            cfg = load_cfg(Path(run_config))

            self.assertEqual(cfg.llm_budget.max_total_tokens, 1_500_000)
            self.assertTrue(cfg.exec.require_isolation)
            self.assertFalse(cfg.exec.allow_experiment_network)

    def test_completed_autopilot_main_returns_without_credentials_or_model_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            project_root = output_root / "projects" / "demo"
            _save_project_progress(
                project_root,
                results=[{"idea_idx": 0, "status": "success"}],
                total=1,
                selected_indices=[0],
            )
            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                mock.patch(
                    "ai_scientist.apps.project._run_autopilot_preflight"
                ) as preflight,
                mock.patch("ai_scientist.apps.project._export_project_research_dag"),
                mock.patch("builtins.print"),
            ):
                main(
                    [
                        "demo",
                        "--output-root",
                        str(output_root),
                        "--autopilot",
                        "balanced",
                        "--research-vcs",
                        "off",
                    ]
                )

            credentials.assert_not_called()
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
