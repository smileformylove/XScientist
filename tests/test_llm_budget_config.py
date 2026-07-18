from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

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


if __name__ == "__main__":
    unittest.main()
