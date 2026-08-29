from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_scientist.apps.project import (
    _configure_autopilot_project_budget,
    _completed_resume_results,
    _prepare_project_input,
    _prepare_autopilot_bfts_config,
    _resolve_resume_work,
    _run_autopilot_preflight,
    _save_project_progress,
    main,
)
from ai_scientist.utils.data_readiness import prepare_data_contract
from ai_scientist.utils.llm_budget import llm_budget_manager


class ProjectAutopilotTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Autopilot intentionally binds the process-wide budget manager for the
        # duration of a project run.  Tests use temporary project directories,
        # so they must not leave that singleton pointing at a deleted ledger.
        llm_budget_manager.configure(max_total_tokens=None, reset=True)
        llm_budget_manager.export_environment()

    def test_data_gate_hashes_empirical_inputs_without_disclosing_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            data = Path(td) / "private-data"
            (root / "00_config").mkdir(parents=True)
            data.mkdir()
            (data / "observations.csv").write_text("x,y\n1,2\n", encoding="utf-8")

            with mock.patch.dict("os.environ", {}, clear=False):
                contract = prepare_data_contract(root, data_dir=data, required=True)
                snapshot = Path(os.environ["AI_SCIENTIST_PROJECT_DATA_DIR"])

            self.assertTrue(contract["ready"])
            self.assertEqual(contract["mode"], "content_addressed_snapshot_read_only")
            self.assertEqual(contract["file_count"], 1)
            self.assertTrue(contract["files"][0]["sha256"].startswith("sha256:"))
            self.assertEqual(snapshot.parent.name, "datasets")
            (data / "observations.csv").write_text("x,y\n9,9\n", encoding="utf-8")
            self.assertEqual(
                (snapshot / "observations.csv").read_text(encoding="utf-8"),
                "x,y\n1,2\n",
            )
            serialized = (root / "00_config" / "data_manifest.json").read_text()
            self.assertNotIn(str(data), serialized)

    def test_data_gate_requires_explicit_synthetic_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "00_config").mkdir()
            with self.assertRaisesRegex(RuntimeError, "before model calls"):
                prepare_data_contract(root, required=True)

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
            persisted_text = (root / "04_logs" / "progress.json").read_text(
                encoding="utf-8"
            )
            persisted = json.loads(persisted_text)
            self.assertNotIn(str(root.resolve()), persisted_text)
            self.assertEqual(
                persisted["results"][1]["checkpoint_path"],
                "02_experiments/run_1/logs/bfts/stage_0/checkpoint.json",
            )
            self.assertEqual(
                persisted["results"][2]["checkpoint_path"],
                "missing.json",
            )

            pending, prior, checkpoints = _resolve_resume_work(
                root, [0, 1, 2], enabled=True
            )

            self.assertEqual(pending, [1, 2])
            self.assertEqual([item["idea_idx"] for item in prior], [0])
            self.assertEqual(checkpoints, {1: str(checkpoint.resolve())})

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

    def test_ideation_only_autopilot_does_not_require_an_executor(self) -> None:
        args = argparse.Namespace(
            autopilot="discovery",
            bfts_config="demo.yaml",
            skip_experiment=True,
        )
        with mock.patch(
            "ai_scientist.apps.preflight.check_bfts_config"
        ) as check_bfts_config:
            rows = _run_autopilot_preflight(args)

        check_bfts_config.assert_not_called()
        self.assertEqual(rows[0]["severity"], "info")
        self.assertIn("--skip-experiment", rows[0]["detail"])

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
                max_project_tokens=250_000,
                max_project_hours=1,
                max_cost_usd=5,
            )

            derived_path = _prepare_autopilot_bfts_config(args)
            derived = yaml.safe_load(
                Path(derived_path or "").read_text(encoding="utf-8")
            )

            self.assertTrue(derived["exec"]["require_isolation"])
            self.assertEqual(derived["exec"]["network"], "none")
            self.assertFalse(derived["exec"]["allow_experiment_network"])
            self.assertEqual(derived["llm_budget"]["max_total_tokens"], 250_000)
            self.assertEqual(derived["llm_budget"]["max_wall_time_seconds"], 3_600)
            self.assertEqual(derived["llm_budget"]["max_cost_usd"], 5)
            self.assertEqual(derived["agent"]["steps"], 3)
            self.assertEqual(derived["agent"]["stages"]["stage1_max_iters"], 8)

            with mock.patch.dict("os.environ", {}, clear=False):
                budget = _configure_autopilot_project_budget(args)
                self.assertTrue(budget["shared_across_project"])
                self.assertTrue(
                    (root / "project" / "04_logs" / "llm_budget.json").is_file()
                )

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

    @unittest.skipUnless(shutil.which("git"), "Git is required for the golden journey")
    def test_golden_question_to_insight_and_research_dag_journey(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            output_root = base / "outputs"
            project_root = base / "study"

            def generate(project_dir, _topic, _model, _count, _reflections):
                path = Path(project_dir) / "01_ideas" / "generated_ideas.json"
                path.write_text(
                    json.dumps(
                        [
                            {
                                "Name": "falsifiable_mechanism",
                                "Title": "Falsifiable mechanism study",
                                "Experiment": "Compare the mechanism against a null control.",
                                "Interestingness": 8,
                                "Feasibility": 8,
                                "Novelty": 7,
                                "core_hypothesis": "The mechanism improves the target metric.",
                                "failure_criteria": [
                                    "The paired null control performs at least as well."
                                ],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                return str(path)

            def execute(process_args):
                exp_dir = project_root / "02_experiments" / "idea_0"
                exp_dir.mkdir(parents=True, exist_ok=True)
                return {
                    "idea_idx": 0,
                    "exp_dir": str(exp_dir),
                    "status": "success",
                    "quality_score": 8.0,
                    "rigor_score": 8.0,
                    "quality_gate_passed": True,
                    "submission_acceptance_passed": True,
                    "claim_support_score": 0.7,
                    "seed": 42,
                }

            with (
                mock.patch.dict("os.environ", {}, clear=False),
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch("ai_scientist.apps.project.require_model_credentials"),
                mock.patch(
                    "ai_scientist.apps.project._run_autopilot_preflight",
                    return_value=[],
                ),
                mock.patch(
                    "ai_scientist.apps.project.generate_ideas", side_effect=generate
                ),
                mock.patch(
                    "ai_scientist.apps.project.select_ranked_idea_candidates",
                    return_value=([0], []),
                ),
                mock.patch(
                    "ai_scientist.apps.project.process_single_idea",
                    side_effect=execute,
                ),
                mock.patch(
                    "ai_scientist.llm.create_client",
                    side_effect=RuntimeError("offline golden journey"),
                ),
                mock.patch("builtins.print"),
            ):
                main(
                    [
                        str(project_root),
                        "--output-root",
                        str(output_root),
                        "--question",
                        "Does the mechanism improve the target metric?",
                        "--autopilot",
                        "balanced",
                        "--allow-synthetic-data",
                        "--research-vcs-strict",
                    ]
                )

            insight = json.loads(
                (project_root / "04_logs" / "insight_report.json").read_text()
            )
            self.assertEqual(
                insight["epistemic_status"], "machine_synthesized_unverified"
            )
            self.assertTrue(insight["insights"])
            dag = (
                output_root
                / "views"
                / project_root.name
                / "research-dag"
                / "research-dag.html"
            )
            self.assertTrue(dag.is_file())
            self.assertTrue((project_root / "research.yaml").is_file())
            self.assertTrue(
                (project_root / "00_config" / "data_manifest.json").is_file()
            )
            progress_text = (project_root / "04_logs" / "progress.json").read_text(
                encoding="utf-8"
            )
            progress = json.loads(progress_text)
            self.assertNotIn(str(project_root), progress_text)
            self.assertNotIn(str(project_root.resolve()), progress_text)
            self.assertEqual(
                progress["results"][0]["exp_dir"],
                "02_experiments/idea_0",
            )
            summary_text = (
                project_root / "04_logs" / "project_summary.json"
            ).read_text(encoding="utf-8")
            summary = json.loads(summary_text)
            self.assertNotIn(str(project_root), summary_text)
            self.assertNotIn(str(project_root.resolve()), summary_text)
            self.assertEqual(summary["project_dir"], ".")
            self.assertEqual(
                summary["results"][0]["exp_dir"],
                "02_experiments/idea_0",
            )
            shortlist_text = (
                project_root / "04_logs" / "submission_shortlist.md"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(project_root), shortlist_text)
            self.assertNotIn(str(project_root.resolve()), shortlist_text)


if __name__ == "__main__":
    unittest.main()
