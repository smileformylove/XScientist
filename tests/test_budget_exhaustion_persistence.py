from __future__ import annotations

import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from ai_scientist.treesearch.agent_manager import Stage
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    MANAGER_STATE_SCHEMA,
    ExperimentTermination,
    perform_experiments_bfts,
    termination_signal_guard,
    write_json_atomic,
)
from ai_scientist.treesearch.utils.config import save_run
from ai_scientist.treesearch.utils.metric import WorstMetricValue
from ai_scientist.utils.llm_budget import LLMBudgetExceeded
from ai_scientist.utils.launcher_workflow import run_experiment_phase
from ai_scientist.utils.run_index import is_stage_complete, mark_stage_stopped


class BudgetExhaustionPersistenceTests(unittest.TestCase):
    def test_sigterm_guard_raises_and_restores_previous_handler(self) -> None:
        handlers = []
        previous_handler = object()
        with (
            mock.patch("signal.getsignal", return_value=previous_handler),
            mock.patch("signal.signal", side_effect=lambda _sig, handler: handlers.append(handler)),
        ):
            with self.assertRaises(ExperimentTermination) as ctx:
                with termination_signal_guard():
                    handlers[-1](signal.SIGTERM, None)

        self.assertEqual(ctx.exception.signum, signal.SIGTERM)
        self.assertIs(handlers[-1], previous_handler)

    def test_run_status_write_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run_status.json"
            write_json_atomic(path, {"status": "completed"})

            self.assertEqual(json.loads(path.read_text())["status"], "completed")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_sigterm_interruption_payload_uses_signal_exit_code(self) -> None:
        result, status, checkpoint_exists = self._run_manager_stop(
            ExperimentTermination(signal.SIGTERM)
        )

        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(status["failure_error"]["signal"], "SIGTERM")
        self.assertEqual(status["failure_error"]["signal_number"], signal.SIGTERM)
        self.assertTrue(checkpoint_exists)

    def _run_manager_stop(self, exception: BaseException) -> tuple[dict, dict, bool]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = None
            cfg.generate_report = True
            stage = Stage(
                name="1_initial_implementation_1_preliminary",
                description="preliminary",
                goals="goal",
                max_iterations=1,
                num_drafts=1,
                stage_number=1,
            )

            class FakeManager:
                def __init__(self, **_kwargs):
                    self.current_stage = stage
                    self.completed_stages = []
                    self.journals = {stage.name: Journal()}

                def run(self, **_kwargs):
                    raise exception

                def _save_checkpoint(self):
                    path = log_dir / f"stage_{stage.name}" / "checkpoint.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"checkpoint")
                    return path

            with (
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_cfg",
                    return_value=cfg,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_task_desc",
                    return_value='{"Title":"T"}',
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.prep_agent_workspace"
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager",
                    FakeManager,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.overall_summarize"
                ) as report_mock,
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            status = json.loads(Path(result["run_status_path"]).read_text())
            manager_state = json.loads(
                Path(result["manager_state_path"]).read_text(encoding="utf-8")
            )
            checkpoint_exists = Path(result["checkpoint_path"]).is_file()
            self.assertEqual(manager_state["schema"], MANAGER_STATE_SCHEMA)
            self.assertEqual(manager_state["status"], result["status"])
            self.assertEqual(
                status["manager_state_path"], result["manager_state_path"]
            )
            self.assertFalse(
                Path(result["manager_state_path"]).with_suffix(".json.tmp").exists()
            )
            self.assertFalse((log_dir / "manager.pkl").exists())
            report_mock.assert_not_called()
            return result, status, checkpoint_exists

    def test_runtime_failure_is_checkpointed_and_reported(self) -> None:
        result, status, checkpoint_exists = self._run_manager_stop(
            RuntimeError("worker crashed")
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(status["failure_error"]["type"], "RuntimeError")
        self.assertEqual(status["failure_error"]["message"], "worker crashed")
        self.assertTrue(result["resumable"])
        self.assertTrue(checkpoint_exists)

    def test_completed_run_preserves_workspace_and_writes_json_manager_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = None
            cfg.generate_report = False
            stage = Stage(
                name="1_initial_implementation_1_preliminary",
                description="preliminary",
                goals="goal",
                max_iterations=1,
                num_drafts=1,
                stage_number=1,
            )

            class FakeManager:
                def __init__(self, **_kwargs):
                    self.current_stage = stage
                    self.completed_stages = []
                    self.journals = {
                        stage.name: Journal(
                            nodes=[
                                Node(
                                    plan="failed",
                                    code="raise RuntimeError()",
                                    metric=WorstMetricValue(),
                                    is_buggy=True,
                                )
                            ]
                        )
                    }

                def run(self, **kwargs):
                    kwargs["step_callback"](stage, self.journals[stage.name])

            with (
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_cfg",
                    return_value=cfg,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_task_desc",
                    return_value='{"Title":"T"}',
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.prep_agent_workspace"
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager",
                    FakeManager,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
                mock.patch.object(Journal, "generate_summary", return_value="summary"),
                mock.patch("ai_scientist.treesearch.utils.config.tree_export.generate"),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            self.assertEqual(result["status"], "completed")
            self.assertTrue(workspace_dir.is_dir())
            manager_state_path = Path(result["manager_state_path"])
            manager_state = json.loads(manager_state_path.read_text(encoding="utf-8"))
            self.assertEqual(manager_state["schema"], MANAGER_STATE_SCHEMA)
            self.assertEqual(manager_state["status"], "completed")
            self.assertIn(stage.name, manager_state["journals"])
            self.assertFalse((log_dir / "manager.pkl").exists())

    def test_resume_initialization_failure_preserves_existing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            marker = workspace_dir / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = root / "checkpoint.json"
            cfg.generate_report = False

            with (
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_cfg",
                    return_value=cfg,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_task_desc",
                    return_value='{"Title":"T"}',
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager.from_checkpoint",
                    side_effect=ValueError("invalid checkpoint"),
                ),
                self.assertRaisesRegex(ValueError, "invalid checkpoint"),
            ):
                perform_experiments_bfts(root / "config.yaml")

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_manager_state_failure_is_reported_without_losing_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = None
            cfg.generate_report = False

            class FakeManager:
                def __init__(self, **_kwargs):
                    self.current_stage = None
                    self.completed_stages = []
                    self.journals = {}

                def run(self, **_kwargs):
                    return None

            with (
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_cfg",
                    return_value=cfg,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_task_desc",
                    return_value='{"Title":"T"}',
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.prep_agent_workspace"
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager",
                    FakeManager,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.manager_state_payload",
                    side_effect=TypeError("snapshot unavailable"),
                ),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            status = json.loads(Path(result["run_status_path"]).read_text())
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["manager_state_path"])
            self.assertIsNone(status["manager_state_path"])
            self.assertIn(
                "manager_state: TypeError: snapshot unavailable",
                result["persistence_errors"],
            )

    def test_keyboard_interrupt_is_checkpointed_and_reported(self) -> None:
        result, status, checkpoint_exists = self._run_manager_stop(KeyboardInterrupt())

        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(status["failure_error"]["type"], "KeyboardInterrupt")
        self.assertTrue(result["resumable"])
        self.assertTrue(checkpoint_exists)

    @mock.patch("ai_scientist.utils.launcher_workflow.mark_stage_stopped")
    @mock.patch("ai_scientist.utils.launcher_workflow.save_token_tracker")
    @mock.patch("ai_scientist.utils.launcher_workflow.perform_experiments_bfts")
    @mock.patch("ai_scientist.utils.launcher_workflow.edit_bfts_config_file")
    @mock.patch(
        "ai_scientist.utils.launcher_workflow.is_stage_complete", return_value=False
    )
    def test_launcher_marks_runtime_failure_stopped(
        self,
        _stage_complete_mock: mock.Mock,
        edit_config_mock: mock.Mock,
        perform_mock: mock.Mock,
        _save_tracker_mock: mock.Mock,
        mark_stopped_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            idea_path = root / "idea.json"
            idea_path.write_text("{}", encoding="utf-8")
            edit_config_mock.return_value = str(root / "bfts_config.yaml")
            perform_mock.return_value = {
                "status": "failed",
                "resumable": True,
                "checkpoint_path": str(root / "checkpoint.json"),
                "run_status_path": str(root / "run_status.json"),
                "failure_error": {
                    "type": "RuntimeError",
                    "message": "worker crashed",
                },
            }

            result = run_experiment_phase(
                root,
                idea_path,
                "plot-model",
                resume=False,
                logger=lambda _message: None,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            mark_stopped_mock.call_args.kwargs["reason"], "experiment_failed"
        )
        self.assertEqual(
            mark_stopped_mock.call_args.kwargs["metadata"]["failure_error"]["type"],
            "RuntimeError",
        )

    def test_stopped_stage_is_not_completed_by_artifact_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "logs").mkdir()

            mark_stage_stopped(
                run_dir,
                "experiment",
                reason="llm_budget_exhausted",
                metadata={"resumable": True},
            )

            self.assertFalse(is_stage_complete(run_dir, "experiment"))

    def test_budget_exhaustion_maps_to_temporary_failure_exit_code(self) -> None:
        import launch_scientist_bfts

        self.assertEqual(
            launch_scientist_bfts.experiment_budget_exit_code(
                {"status": "budget_exhausted"}
            ),
            75,
        )
        self.assertIsNone(
            launch_scientist_bfts.experiment_budget_exit_code({"status": "completed"})
        )
        self.assertEqual(
            launch_scientist_bfts.experiment_stop_exit_code({"status": "failed"}), 1
        )
        self.assertEqual(
            launch_scientist_bfts.experiment_stop_exit_code(
                {"status": "interrupted"}
            ),
            130,
        )
        self.assertEqual(
            launch_scientist_bfts.experiment_stop_exit_code(
                {
                    "status": "interrupted",
                    "failure_error": {"signal_number": signal.SIGTERM},
                }
            ),
            143,
        )

    def test_save_run_offline_mode_never_calls_llm_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.log_dir = Path(td)
            journal = Journal(
                nodes=[
                    Node(
                        plan="failed",
                        code="raise RuntimeError()",
                        metric=WorstMetricValue(),
                        is_buggy=True,
                    )
                ]
            )
            with (
                mock.patch.object(
                    journal,
                    "get_best_node",
                    side_effect=AssertionError("LLM selection called"),
                ) as llm_selection,
                mock.patch("ai_scientist.treesearch.utils.config.tree_export.generate"),
            ):
                save_run(cfg, journal, stage_name="stage_x", allow_llm_selection=False)

            llm_selection.assert_not_called()
            self.assertTrue((Path(td) / "stage_x" / "journal.json").is_file())
            self.assertTrue((Path(td) / "stage_x" / "config.yaml").is_file())

    @mock.patch("ai_scientist.utils.launcher_workflow.mark_stage_complete")
    @mock.patch("ai_scientist.utils.launcher_workflow.aggregate_plots")
    @mock.patch("ai_scientist.utils.launcher_workflow.write_experiment_report")
    @mock.patch("ai_scientist.utils.launcher_workflow.mark_stage_stopped")
    @mock.patch("ai_scientist.utils.launcher_workflow.save_token_tracker")
    @mock.patch("ai_scientist.utils.launcher_workflow.perform_experiments_bfts")
    @mock.patch("ai_scientist.utils.launcher_workflow.edit_bfts_config_file")
    @mock.patch(
        "ai_scientist.utils.launcher_workflow.is_stage_complete", return_value=False
    )
    def test_experiment_budget_stop_skips_reporting_and_marks_stopped(
        self,
        _stage_complete_mock: mock.Mock,
        edit_config_mock: mock.Mock,
        perform_mock: mock.Mock,
        save_tracker_mock: mock.Mock,
        mark_stopped_mock: mock.Mock,
        write_report_mock: mock.Mock,
        aggregate_plots_mock: mock.Mock,
        mark_complete_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            idea_path = root / "idea.json"
            idea_path.write_text("{}", encoding="utf-8")
            checkpoint = root / "logs" / "0-run" / "stage_x" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            edit_config_mock.return_value = str(root / "bfts_config.yaml")
            perform_mock.return_value = {
                "status": "budget_exhausted",
                "resumable": True,
                "checkpoint_path": str(checkpoint),
                "run_status_path": str(checkpoint.parents[1] / "run_status.json"),
                "budget_error": {"dimension": "tokens"},
            }

            result = run_experiment_phase(
                root,
                idea_path,
                "plot-model",
                resume=False,
                logger=lambda _message: None,
            )

        self.assertEqual(result["status"], "budget_exhausted")
        self.assertTrue(result["resumable"])
        save_tracker_mock.assert_called_once_with(root)
        mark_stopped_mock.assert_called_once()
        write_report_mock.assert_not_called()
        aggregate_plots_mock.assert_not_called()
        mark_complete_mock.assert_not_called()

    @mock.patch("ai_scientist.utils.launcher_workflow.perform_experiments_bfts")
    @mock.patch("ai_scientist.utils.launcher_workflow.edit_bfts_config_file")
    @mock.patch(
        "ai_scientist.utils.launcher_workflow.is_stage_complete", return_value=False
    )
    def test_stopped_experiment_passes_checkpoint_back_to_bfts_config(
        self,
        _stage_complete_mock: mock.Mock,
        edit_config_mock: mock.Mock,
        perform_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "logs" / "0-run" / "stage_x" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            (root / ".workflow_state.json").write_text(
                json.dumps(
                    {
                        "stages": {
                            "experiment": {
                                "status": "stopped",
                                "metadata": {"checkpoint_path": str(checkpoint)},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            idea_path = root / "idea.json"
            idea_path.write_text("{}", encoding="utf-8")
            edit_config_mock.return_value = str(root / "bfts_config.yaml")
            perform_mock.return_value = {
                "status": "budget_exhausted",
                "resumable": True,
                "checkpoint_path": str(checkpoint),
                "run_status_path": str(root / "run_status.json"),
                "budget_error": {"dimension": "tokens"},
            }

            with (
                mock.patch("ai_scientist.utils.launcher_workflow.save_token_tracker"),
                mock.patch("ai_scientist.utils.launcher_workflow.mark_stage_stopped"),
            ):
                run_experiment_phase(
                    root,
                    idea_path,
                    "plot-model",
                    resume=True,
                    logger=lambda _message: None,
                )

        self.assertEqual(
            edit_config_mock.call_args.kwargs["resume_from"], str(checkpoint.resolve())
        )

    @mock.patch(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.prep_agent_workspace"
    )
    @mock.patch(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_task_desc"
    )
    @mock.patch(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.load_cfg"
    )
    def test_budget_stop_writes_status_and_skips_final_report(
        self,
        load_cfg_mock: mock.Mock,
        load_task_desc_mock: mock.Mock,
        _prep_workspace_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = None
            cfg.generate_report = True
            load_cfg_mock.return_value = cfg
            load_task_desc_mock.return_value = '{"Title":"T"}'
            stage = Stage(
                name="1_initial_implementation_1_preliminary",
                description="preliminary",
                goals="goal",
                max_iterations=1,
                num_drafts=1,
                stage_number=1,
            )

            class FakeManager:
                last_instance = None

                def __init__(self, **_kwargs):
                    type(self).last_instance = self
                    self.current_stage = stage
                    self.completed_stages = []
                    self.journals = {
                        stage.name: Journal(
                            nodes=[
                                Node(
                                    plan="failed",
                                    code="raise RuntimeError()",
                                    metric=WorstMetricValue(),
                                    is_buggy=True,
                                )
                            ]
                        )
                    }

                def run(self, **_kwargs):
                    _kwargs["step_callback"](stage, self.journals[stage.name])

                def _save_checkpoint(self):
                    path = log_dir / f"stage_{stage.name}" / "checkpoint.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"checkpoint")
                    return path

            with (
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager",
                    FakeManager,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.overall_summarize"
                ) as report_mock,
                mock.patch("ai_scientist.treesearch.utils.config.tree_export.generate"),
                mock.patch.object(
                    Journal,
                    "generate_summary",
                    side_effect=LLMBudgetExceeded(
                        "tokens", "LLM token budget is exhausted", {"used": {}}
                    ),
                ),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            status = json.loads(Path(result["run_status_path"]).read_text())
            saved_stage_name = next(iter(FakeManager.last_instance.journals))
            actual_stage_dir = (
                Path(result["run_status_path"]).parent / f"stage_{saved_stage_name}"
            )
            checkpoint_exists = Path(result["checkpoint_path"]).is_file()
            persisted_files = list(actual_stage_dir.parent.rglob("*"))
            journal_persisted = (actual_stage_dir / "journal.json").is_file()
            config_persisted = (actual_stage_dir / "config.yaml").is_file()

        self.assertEqual(result["status"], "budget_exhausted")
        self.assertTrue(result["resumable"])
        self.assertEqual(status["status"], "budget_exhausted")
        self.assertEqual(status["budget_error"]["dimension"], "tokens")
        self.assertEqual(status["checkpoint_path"], result["checkpoint_path"])
        self.assertTrue(
            journal_persisted,
            f"run_status={result['run_status_path']}; checkpoint={result['checkpoint_path']}; "
            f"checkpoint_exists={checkpoint_exists}; files={persisted_files}; "
            f"errors={result['persistence_errors']}",
        )
        self.assertTrue(config_persisted)
        report_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
