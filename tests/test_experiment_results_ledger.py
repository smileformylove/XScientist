from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from ai_scientist.treesearch.agent_manager import Stage
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    RESULTS_TSV_COLUMNS,
    perform_experiments_bfts,
    repair_results_tsv,
)
from ai_scientist.treesearch.utils.metric import WorstMetricValue
from ai_scientist.treesearch.utils.serialize import durable_append_text


class ExperimentResultsLedgerTests(unittest.TestCase):
    def _row(
        self,
        node_id: str,
        *,
        kind: str = "main",
        status: str = "ok",
        decision: str = "keep",
    ) -> str:
        values = ["" for _ in RESULTS_TSV_COLUMNS]
        values[0] = "2026-01-01T00:00:00"
        values[1] = "stage"
        values[3] = kind
        values[4] = node_id
        values[6] = status
        values[7] = decision
        values[8] = "1.0"
        values[14] = "1"
        return "\t".join(values) + "\n"

    def test_repair_drops_torn_tail_and_recovers_logged_node_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_bytes((header + self._row("node-1") + "partial\trow").encode())

            logged, stage_best = repair_results_tsv(path)

            self.assertEqual(logged, {"node-1"})
            self.assertEqual(stage_best["stage"]["node_id"], "node-1")
            self.assertEqual(
                path.read_text(encoding="utf-8"), header + self._row("node-1")
            )

    def test_repair_rejects_malformed_middle_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_text(
                header + "bad\trow\n" + self._row("node-2"), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Malformed"):
                repair_results_tsv(path)

            self.assertIn("bad\trow", path.read_text(encoding="utf-8"))

    def test_repair_rejects_complete_malformed_tail_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_text(
                header + self._row("node-1") + "bad\trow\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Malformed"):
                repair_results_tsv(path)

            self.assertTrue(path.read_text(encoding="utf-8").endswith("bad\trow\n"))

    def test_repair_rejects_duplicate_node_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_text(
                header + self._row("node-1") + self._row("node-1"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate node id"):
                repair_results_tsv(path)

    def test_repair_rejects_invalid_ledger_enums(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_text(
                header
                + self._row(
                    "node-1", kind="seed", status="ok", decision="discard"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid.*decision"):
                repair_results_tsv(path)

    def test_repair_rejects_keep_for_failed_node(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_text(
                header
                + self._row(
                    "node-1", kind="main", status="crash", decision="keep"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "keep decision"):
                repair_results_tsv(path)

    def test_durable_append_retries_partial_os_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            real_write = os.write
            calls = 0

            def partial_write(descriptor: int, payload) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    chunk_size = max(1, len(payload) // 2)
                    return real_write(descriptor, payload[:chunk_size])
                return real_write(descriptor, payload)

            with mock.patch(
                "ai_scientist.treesearch.utils.serialize.os.write",
                side_effect=partial_write,
            ):
                durable_append_text(path, "complete-row\n")

            self.assertGreaterEqual(calls, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "complete-row\n")

    def test_durable_append_rolls_back_partial_row_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            path.write_text("previous-row\n", encoding="utf-8")
            real_write = os.write
            calls = 0

            def partial_then_fail(descriptor: int, payload) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(descriptor, payload[:4])
                raise OSError("disk full")

            with (
                mock.patch(
                    "ai_scientist.treesearch.utils.serialize.os.write",
                    side_effect=partial_then_fail,
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                durable_append_text(path, "new-row\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "previous-row\n")

    def test_failed_journal_save_does_not_publish_derived_artifacts(self) -> None:
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

            class FakeManager:
                checkpoint_calls = 0

                def __init__(self, **_kwargs):
                    self.current_stage = stage
                    self.completed_stages = []
                    self.journals = {stage.name: journal}

                def run(self, **kwargs):
                    kwargs["step_callback"](stage, journal)

                def _save_checkpoint(self):
                    type(self).checkpoint_calls += 1
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
                mock.patch.object(Journal, "generate_summary", return_value="summary"),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.save_run",
                    side_effect=OSError("disk full"),
                ),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                (log_dir / "results.tsv").read_text(encoding="utf-8"), header
            )
            self.assertFalse((log_dir / "program.md").exists())
            self.assertEqual(FakeManager.checkpoint_calls, 1)

    def test_failed_checkpoint_does_not_publish_derived_artifacts(self) -> None:
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

            class FakeManager:
                checkpoint_calls = 0

                def __init__(self, **_kwargs):
                    self.current_stage = stage
                    self.completed_stages = []
                    self.journals = {stage.name: journal}

                def run(self, **kwargs):
                    kwargs["step_callback"](stage, journal)

                def _save_checkpoint(self):
                    type(self).checkpoint_calls += 1
                    if type(self).checkpoint_calls == 1:
                        raise OSError("checkpoint disk full")
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
                mock.patch.object(Journal, "generate_summary", return_value="summary"),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.save_run"
                ),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                (log_dir / "results.tsv").read_text(encoding="utf-8"), header
            )
            self.assertFalse((log_dir / "program.md").exists())
            self.assertEqual(FakeManager.checkpoint_calls, 2)


if __name__ == "__main__":
    unittest.main()
