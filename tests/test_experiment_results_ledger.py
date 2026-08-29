from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from omegaconf import OmegaConf

from ai_scientist.resources import resolve_bfts_config_path

from ai_scientist.treesearch.agent_manager import Stage
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    RESULTS_TSV_COLUMNS,
    _encode_dataset_names,
    _gate_binding_payload,
    _multi_seed_gate_meta,
    _validate_ledger_gates_against_checkpoint,
    perform_experiments_bfts,
    repair_results_tsv,
)
from ai_scientist.treesearch.utils.metric import MetricValue, WorstMetricValue
from ai_scientist.treesearch.utils.serialize import durable_append_text


class ExperimentResultsLedgerTests(unittest.TestCase):
    def _row(
        self,
        node_id: str,
        *,
        kind: str = "main",
        status: str = "ok",
        decision: str = "provisional",
        parent_id: str = "",
    ) -> str:
        values = ["" for _ in RESULTS_TSV_COLUMNS]
        values[0] = "2026-01-01T00:00:00"
        values[1] = "stage"
        values[3] = kind
        values[4] = node_id
        values[5] = parent_id
        values[6] = status
        values[7] = decision
        values[8] = "1.0"
        values[9] = "1.0"
        values[10] = "accuracy"
        values[11] = "True"
        values[12] = "[]"
        values[14] = "1"
        return "\t".join(values) + "\n"

    def test_repair_drops_torn_tail_and_recovers_logged_node_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_bytes((header + self._row("node-1") + "partial\trow").encode())

            logged, stage_best, node_stages, _gate_bindings = repair_results_tsv(path)

            self.assertEqual(logged, {"node-1"})
            self.assertEqual(stage_best, {})
            self.assertEqual(node_stages, {"node-1": "stage"})
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
                + self._row("node-1", kind="seed", status="ok", decision="discard"),
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
                    "node-1",
                    kind="main",
                    status="crash",
                    decision="provisional",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "positive decision"):
                repair_results_tsv(path)

    def test_repair_recovers_only_final_gate_as_stage_best(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            gate_id = "gate_" + "a" * 64
            path.write_text(
                header
                + self._row("candidate")
                + self._row(
                    gate_id,
                    kind="gate",
                    status="ok",
                    decision="qualified",
                    parent_id="candidate",
                ),
                encoding="utf-8",
            )

            logged, stage_best, _node_stages, gate_bindings = repair_results_tsv(path)

            self.assertEqual(logged, {"candidate", gate_id})
            self.assertEqual(stage_best["stage"]["node_id"], "candidate")
            self.assertEqual(gate_bindings[gate_id]["parent_id"], "candidate")

    def test_repair_migrates_legacy_keep_to_provisional(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            legacy = self._row("candidate", decision="keep")
            path.write_text(header + legacy, encoding="utf-8")

            _logged, stage_best, _node_stages, _gate_bindings = repair_results_tsv(path)

            self.assertEqual(stage_best, {})
            self.assertIn("\tprovisional\t", path.read_text(encoding="utf-8"))
            self.assertNotIn("\tkeep\t", path.read_text(encoding="utf-8"))

    def test_repair_rejects_unbound_or_duplicate_qualified_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_text(
                header
                + self._row("candidate")
                + self._row(
                    "gate_" + "a" * 64,
                    kind="gate",
                    status="ok",
                    decision="qualified",
                    parent_id="candidate",
                )
                + self._row("candidate-2")
                + self._row(
                    "gate_" + "b" * 64,
                    kind="gate",
                    status="ok",
                    decision="qualified",
                    parent_id="candidate-2",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate qualified gate"):
                repair_results_tsv(path)

            path.write_text(
                header
                + self._row(
                    "gate_" + "c" * 64,
                    kind="gate",
                    status="ok",
                    decision="qualified",
                    parent_id="missing",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unbound gate event"):
                repair_results_tsv(path)

    def test_repair_rejects_qualified_gate_for_discarded_parent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            path.write_text(
                header
                + self._row(
                    "candidate",
                    status="crash",
                    decision="discard",
                )
                + self._row(
                    "gate_" + "a" * 64,
                    kind="gate",
                    status="ok",
                    decision="qualified",
                    parent_id="candidate",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unbound gate event"):
                repair_results_tsv(path)

    def test_repair_rejects_nonfinite_gate_metrics_and_reserved_node_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.tsv"
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            gate_fields = (
                self._row(
                    "gate_" + "a" * 64,
                    kind="gate",
                    status="ok",
                    decision="qualified",
                    parent_id="candidate",
                )
                .rstrip("\n")
                .split("\t")
            )
            gate_fields[8] = "NaN"
            gate_fields[9] = "NaN"
            path.write_text(
                header + self._row("candidate") + "\t".join(gate_fields) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid gate metrics"):
                repair_results_tsv(path)

            path.write_text(
                header + self._row("gate_" + "b" * 64),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Reserved gate event id"):
                repair_results_tsv(path)

    def test_dataset_names_are_canonical_json_escaped(self) -> None:
        encoded = _encode_dataset_names(["line\nbreak", "tab\tname", "plain"])

        self.assertNotIn("\n", encoded)
        self.assertNotIn("\t", encoded)
        self.assertEqual(
            json.loads(encoded),
            ["line\nbreak", "plain", "tab\tname"],
        )

    def test_resume_gate_bindings_must_match_checkpoint_receipts(self) -> None:
        stage = SimpleNamespace(
            name="stage",
            qualified_node_id=None,
            multi_seed_receipt_hash=None,
        )
        node = Node(
            id="candidate",
            metric=MetricValue(0.9, maximize=True),
            multi_seed_attempts=[],
        )
        manager = SimpleNamespace(
            stages=[stage],
            journals={"stage": Journal(nodes=[node])},
        )
        forged_id = "gate_" + "a" * 64
        report = {
            "receipt_hash": "sha256:" + "a" * 64,
            "datasets": {"test": {"mean": 0.75}},
        }
        forged = {
            forged_id: _gate_binding_payload(
                node,
                stage="stage",
                receipt_hash=report["receipt_hash"],
                decision="qualified",
                report=report,
            )
        }

        with self.assertRaisesRegex(ValueError, "do not match"):
            _validate_ledger_gates_against_checkpoint(forged, manager)

        stage.qualified_node_id = "candidate"
        stage.multi_seed_receipt_hash = "sha256:" + "a" * 64
        node.multi_seed_report = report
        _validate_ledger_gates_against_checkpoint(forged, manager)

        altered = {forged_id: {**forged[forged_id], "objective": "0.8"}}
        with self.assertRaisesRegex(ValueError, "do not match"):
            _validate_ledger_gates_against_checkpoint(altered, manager)

    def test_gate_scalar_uses_confirmation_means_not_selection_score(self) -> None:
        node = Node(metric=MetricValue(0.95, maximize=True))
        report = {
            "datasets": {
                "a": {"mean": 0.2},
                "b": {"mean": 0.4},
                "c": {"mean": 0.6},
            }
        }

        metric_mean, objective, _name, maximize = _multi_seed_gate_meta(node, report)

        self.assertAlmostEqual(metric_mean, 0.4)
        self.assertAlmostEqual(objective, 0.4)
        self.assertTrue(maximize)

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
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
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
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
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

    def test_resume_backfills_checkpoint_nodes_missing_from_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            checkpoint = log_dir / "stage_demo" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = checkpoint
            cfg.generate_report = False
            stage = Stage(
                name="1_initial_implementation_1_preliminary",
                description="preliminary",
                goals="goal",
                max_iterations=1,
                num_drafts=1,
                stage_number=1,
            )
            node = Node(
                id="checkpoint-node",
                plan="failed",
                code="raise RuntimeError()",
                metric=WorstMetricValue(),
                is_buggy=True,
            )
            journal = Journal(nodes=[node])

            class FakeManager:
                def __init__(self):
                    self.current_stage = stage
                    self.current_stage_number = 1
                    self.completed_stages = []
                    self.stages = [stage]
                    self.stage_history = []
                    self.journals = {stage.name: journal}
                    self.task_desc = {"Title": "T"}

                def run(self, **_kwargs):
                    return None

            manager = FakeManager()
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
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager.from_checkpoint",
                    return_value=manager,
                ) as restore_mock,
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            logged, _stage_best, node_stages, _gate_bindings = repair_results_tsv(
                log_dir / "results.tsv"
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(logged, {"checkpoint-node"})
            self.assertEqual(node_stages, {"checkpoint-node": stage.name})
            self.assertIn(
                "\tcheckpoint-node\t",
                (log_dir / "results.tsv").read_text(encoding="utf-8"),
            )
            self.assertTrue((log_dir / "program.md").is_file())
            restore_mock.assert_called_once()

    def test_resume_rejects_ledger_nodes_absent_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            checkpoint = log_dir / "stage_demo" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = checkpoint
            cfg.generate_report = False
            stage = Stage(
                name="1_initial_implementation_1_preliminary",
                description="preliminary",
                goals="goal",
                max_iterations=1,
                num_drafts=1,
                stage_number=1,
            )
            manager = mock.Mock(
                current_stage=stage,
                current_stage_number=1,
                completed_stages=[],
                stages=[stage],
                stage_history=[],
                journals={stage.name: Journal()},
                task_desc={"Title": "T"},
            )
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            (log_dir / "results.tsv").write_text(
                header + self._row("newer-node"), encoding="utf-8"
            )

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
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.AgentManager.from_checkpoint",
                    return_value=manager,
                ),
                mock.patch(
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            self.assertEqual(result["status"], "initialization_failed")
            self.assertEqual(result["initialization_phase"], "checkpoint_restore")
            self.assertEqual(
                result["failure_error"]["error_code"],
                "initialization_initialization_failed_checkpoint_restore",
            )
            self.assertNotIn("ahead of the selected checkpoint", json.dumps(result))

    def test_new_run_rejects_existing_populated_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_dir = root / "logs" / "0-run"
            workspace_dir = root / "workspaces" / "0-run"
            log_dir.mkdir(parents=True)
            workspace_dir.mkdir(parents=True)
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
            cfg.exp_name = "0-run"
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace_dir
            cfg.resume_from = None
            cfg.generate_report = False
            header = "\t".join(RESULTS_TSV_COLUMNS) + "\n"
            (log_dir / "results.tsv").write_text(
                header + self._row("old-node"), encoding="utf-8"
            )

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
                    "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager.backend.compile_prompt_to_md",
                    return_value="task",
                ),
            ):
                result = perform_experiments_bfts(root / "config.yaml")

            self.assertEqual(result["status"], "initialization_failed")
            self.assertEqual(result["initialization_phase"], "manager_creation")
            self.assertEqual(
                result["failure_error"]["error_code"],
                "initialization_initialization_failed_manager_creation",
            )
            self.assertNotIn("requires a resume checkpoint", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
