from __future__ import annotations

import os
import pickle
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from ai_scientist.treesearch.agent_manager import AgentManager
from ai_scientist.treesearch.journal import Node
from ai_scientist.treesearch.utils.metric import MetricValue
from ai_scientist.treesearch.utils.config import prep_cfg
from ai_scientist.utils.llm_budget import llm_budget_manager


class LLMBudgetConfigTests(unittest.TestCase):
    def test_bfts_config_creates_shared_budget_state_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.data_dir = str(root / "data")
            cfg.desc_file = None
            cfg.goal = "test"
            cfg.log_dir = str(root / "logs")
            cfg.workspace_dir = str(root / "workspaces")
            cfg.llm_budget.max_total_tokens = 1234

            loaded = prep_cfg(cfg)
            state_path = loaded.workspace_dir / "llm_budget.json"
            self.assertTrue(state_path.exists())
            self.assertEqual(
                os.environ["AI_SCIENTIST_LLM_BUDGET_STATE"], str(state_path)
            )
            self.assertEqual(
                llm_budget_manager.snapshot()["limits"]["max_total_tokens"],
                1234,
            )
            llm_budget_manager.configure(max_total_tokens=None, reset=True)
            llm_budget_manager.export_environment()

    def test_environment_budget_is_not_cleared_by_null_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.data_dir = str(root / "data")
            cfg.desc_file = None
            cfg.goal = "test"
            cfg.log_dir = str(root / "logs")
            cfg.workspace_dir = str(root / "workspaces")
            old_value = os.environ.get("AI_SCIENTIST_LLM_MAX_TOTAL_TOKENS")
            old_state = os.environ.get("AI_SCIENTIST_LLM_BUDGET_STATE")
            os.environ["AI_SCIENTIST_LLM_MAX_TOTAL_TOKENS"] = "777"
            os.environ.pop("AI_SCIENTIST_LLM_BUDGET_STATE", None)
            try:
                prep_cfg(cfg)
                self.assertEqual(
                    llm_budget_manager.snapshot()["limits"]["max_total_tokens"],
                    777,
                )
            finally:
                if old_value is None:
                    os.environ.pop("AI_SCIENTIST_LLM_MAX_TOTAL_TOKENS", None)
                else:
                    os.environ["AI_SCIENTIST_LLM_MAX_TOTAL_TOKENS"] = old_value
                if old_state is None:
                    os.environ.pop("AI_SCIENTIST_LLM_BUDGET_STATE", None)
                else:
                    os.environ["AI_SCIENTIST_LLM_BUDGET_STATE"] = old_state
                llm_budget_manager.configure(max_total_tokens=None, reset=True)
                llm_budget_manager.export_environment()

    def test_resume_reuses_existing_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data"
            logs_dir = root / "logs"
            workspaces_dir = root / "workspaces"
            run_name = "0-run"
            checkpoint = logs_dir / run_name / "stage_demo" / "checkpoint.pkl"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(pickle.dumps({"placeholder": True}))
            (workspaces_dir / run_name).mkdir(parents=True)
            data_dir.mkdir()

            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.data_dir = str(data_dir)
            cfg.desc_file = None
            cfg.goal = "test"
            cfg.log_dir = str(logs_dir)
            cfg.workspace_dir = str(workspaces_dir)
            cfg.resume_from = str(checkpoint)

            old_state = os.environ.get("AI_SCIENTIST_LLM_BUDGET_STATE")
            os.environ.pop("AI_SCIENTIST_LLM_BUDGET_STATE", None)
            try:
                loaded = prep_cfg(cfg)
                resumed_budget_path = os.environ[
                    "AI_SCIENTIST_LLM_BUDGET_STATE"
                ]
            finally:
                if old_state is None:
                    os.environ.pop("AI_SCIENTIST_LLM_BUDGET_STATE", None)
                else:
                    os.environ["AI_SCIENTIST_LLM_BUDGET_STATE"] = old_state
                llm_budget_manager.configure(max_total_tokens=None, reset=True)
                llm_budget_manager.export_environment()

            self.assertEqual(loaded.exp_name, run_name)
            self.assertEqual(loaded.log_dir, (logs_dir / run_name).resolve())
            self.assertEqual(
                loaded.workspace_dir, (workspaces_dir / run_name).resolve()
            )
            self.assertEqual(loaded.resume_from, checkpoint.resolve())
            self.assertEqual(
                resumed_budget_path,
                str((workspaces_dir / run_name / "llm_budget.json").resolve()),
            )

    def test_agent_manager_checkpoint_round_trip_preserves_stage_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "workspaces" / "0-run"
            log_dir = root / "logs" / "0-run"
            workspace.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace
            manager = AgentManager(
                task_desc='{"Title":"T","Abstract":"A","Short Hypothesis":"H",'
                '"Experiments":[],"Risk Factors and Limitations":[]}',
                cfg=cfg,
                workspace_dir=workspace,
            )
            manager.completed_stages = ["prior_stage"]
            journal = manager.journals[manager.current_stage.name]
            parent = Node(
                plan="parent",
                code="print('parent')",
                metric=MetricValue(0.5, maximize=True),
                is_buggy=False,
            )
            child = Node(
                plan="child",
                code="print('child')",
                metric=MetricValue(0.7, maximize=True),
                is_buggy=False,
                parent=parent,
            )
            journal.append(parent)
            journal.append(child)

            checkpoint = manager._save_checkpoint()
            restored = AgentManager.from_checkpoint(
                checkpoint, cfg=cfg, workspace_dir=workspace
            )

            self.assertEqual(restored.current_stage.name, manager.current_stage.name)
            self.assertEqual(restored.completed_stages, ["prior_stage"])
            self.assertEqual(set(restored.journals), set(manager.journals))
            restored_nodes = restored.journals[restored.current_stage.name].nodes
            self.assertEqual(len(restored_nodes), 2)
            self.assertIs(restored_nodes[1].parent, restored_nodes[0])
            self.assertIn(restored_nodes[1], restored_nodes[0].children)


if __name__ == "__main__":
    unittest.main()
