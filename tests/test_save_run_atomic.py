from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omegaconf import OmegaConf

from ai_scientist.treesearch.journal import Journal, Node
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

    def test_best_solution_pointer_failure_preserves_previous_solution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.log_dir = root
            stage_dir = root / "stage_x"

            with mock.patch("ai_scientist.treesearch.utils.config.tree_export.generate"):
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
            cfg = OmegaConf.load("bfts_config.yaml")
            cfg.log_dir = root
            stage_dir = root / "stage_x"

            with mock.patch("ai_scientist.treesearch.utils.config.tree_export.generate"):
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
