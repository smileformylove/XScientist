from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from ai_scientist.resources import resolve_bfts_config_path

from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.utils import serialize
from ai_scientist.treesearch.utils.config import save_run
from ai_scientist.treesearch.utils.metric import MetricValue


class SaveRunAtomicTests(unittest.TestCase):
    def _journal(self, node_id: str, metric: float, code: str) -> Journal:
        return Journal(
            nodes=[
                Node(
                    id=node_id,
                    code=code,
                    plan="plan",
                    metric=MetricValue(metric, maximize=True),
                    is_buggy=False,
                )
            ]
        )

    def test_atomic_json_failure_preserves_previous_file_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "journal.json"
            path.write_text("previous", encoding="utf-8")
            journal = self._journal("new", 1.0, "print('new')")

            with (
                mock.patch.object(Path, "replace", side_effect=OSError("disk busy")),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                serialize.dump_json(journal, path)

            self.assertEqual(path.read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(path.parent.glob(".journal.json.*.tmp")), [])

    def test_atomic_json_payload_failure_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "status.json"
            path.write_text('{"status":"previous"}', encoding="utf-8")

            with self.assertRaises(TypeError):
                serialize.atomic_write_json(path, {"invalid": object()})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["status"],
                "previous",
            )
            self.assertEqual(list(path.parent.glob(".status.json.*.tmp")), [])

    def test_checkpoint_replace_failure_preserves_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "workspace"
            log_dir = root / "logs"
            workspace.mkdir()
            log_dir.mkdir()
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
            cfg.log_dir = log_dir
            cfg.workspace_dir = workspace
            manager = AgentManager(
                task_desc=(
                    '{"Title":"T","Abstract":"A","Short Hypothesis":"H",'
                    '"Experiments":[],"Risk Factors and Limitations":[]}'
                ),
                cfg=cfg,
                workspace_dir=workspace,
            )
            manager.current_stage = Stage(
                name="demo",
                description="demo",
                goals="demo",
                max_iterations=1,
                num_drafts=1,
                stage_number=1,
            )
            manager.stages = [manager.current_stage]
            checkpoint = log_dir / "stage_demo" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text('{"schema":"previous"}', encoding="utf-8")

            with (
                mock.patch.object(Path, "replace", side_effect=OSError("disk busy")),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                manager._save_checkpoint()

            self.assertEqual(
                json.loads(checkpoint.read_text(encoding="utf-8"))["schema"],
                "previous",
            )
            self.assertEqual(list(checkpoint.parent.glob(".checkpoint.json.*.tmp")), [])

    def test_best_solution_pointer_failure_preserves_previous_solution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
            cfg.log_dir = root
            stage_dir = root / "stage_x"

            with mock.patch(
                "ai_scientist.treesearch.utils.config.tree_export.generate"
            ):
                save_run(cfg, self._journal("old", 1.0, "print('old')"), "stage_x")

            real_atomic_write = serialize.atomic_write_text

            def fail_pointer(path, content, *, encoding="utf-8"):
                if Path(path).name == "best_node_id.txt":
                    raise OSError("pointer write failed")
                return real_atomic_write(path, content, encoding=encoding)

            with (
                mock.patch("ai_scientist.treesearch.utils.config.tree_export.generate"),
                mock.patch(
                    "ai_scientist.treesearch.utils.config.serialize.atomic_write_text",
                    side_effect=fail_pointer,
                ),
            ):
                save_run(cfg, self._journal("new", 2.0, "print('new')"), "stage_x")

            self.assertEqual(
                (stage_dir / "best_node_id.txt").read_text(encoding="utf-8"),
                "old",
            )
            self.assertEqual(
                (stage_dir / "best_solution_old.py").read_text(encoding="utf-8"),
                "print('old')",
            )
            self.assertFalse((stage_dir / "best_solution_new.py").exists())

    def test_successful_save_replaces_best_solution_without_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
            cfg.log_dir = root
            stage_dir = root / "stage_x"

            with mock.patch(
                "ai_scientist.treesearch.utils.config.tree_export.generate"
            ):
                save_run(cfg, self._journal("old", 1.0, "print('old')"), "stage_x")
                save_run(cfg, self._journal("new", 2.0, "print('new')"), "stage_x")

            self.assertEqual(
                (stage_dir / "best_node_id.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertFalse((stage_dir / "best_solution_old.py").exists())
            self.assertEqual(
                (stage_dir / "best_solution_new.py").read_text(encoding="utf-8"),
                "print('new')",
            )
            self.assertEqual(list(stage_dir.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
