"""Tests for ara_pipeline — the high-level glue used by run_project.py."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_pipeline import (
    SEED_PROVENANCE_ENV_VAR,
    finalize_ara_for_idea,
    stage_seed_from_cli,
    summarise_finalize,
    summarise_seed_stage,
)
from ai_scientist.utils.ara_seed import SEED_ENV_VAR, clear_active_seed_env


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _seed_parent(tmp: Path) -> tuple[Path, Path, str]:
    project = tmp / "parent"
    exp = project / "02_experiments" / "20260701_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(
        exp / "logs",
        "0-run",
        [
            {
                "id": "seed_node",
                "step": 0,
                "code": "print('parent')",
                "_term_out": ["ok\n"],
                "metric": {"value": 0.5, "maximize": True, "name": "acc"},
                "is_buggy": False,
                "parent_id": None,
                "children": [],
            }
        ],
    )
    result = export_ara(
        project_dir=project,
        exp_dir=exp,
        idea={"Name": "parent"},
        timestamp="20260701",
    )
    return project, result.root, "seed_node"


class StageSeedFromCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_parent(self.tmp)
        self._orig_seed = os.environ.get(SEED_ENV_VAR)
        self._orig_prov = os.environ.get(SEED_PROVENANCE_ENV_VAR)

    def tearDown(self) -> None:
        clear_active_seed_env()
        for var, orig in (
            (SEED_ENV_VAR, self._orig_seed),
            (SEED_PROVENANCE_ENV_VAR, self._orig_prov),
        ):
            if orig is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = orig
        self._tmp.cleanup()

    def test_no_seed_returns_unused_result(self) -> None:
        result = stage_seed_from_cli(
            seed_from_ara=None, seed_node_id=None, project_dir=self.tmp
        )
        self.assertFalse(result.seed_used)
        self.assertNotIn(SEED_ENV_VAR, os.environ)

    def test_seed_from_ara_node_sets_env_vars(self) -> None:
        result = stage_seed_from_cli(
            seed_from_ara=str(self.ara_root),
            seed_node_id=self.node_id,
            project_dir=self.tmp,
        )
        self.assertTrue(result.seed_used)
        self.assertEqual(os.environ[SEED_ENV_VAR], str(result.seed_path))
        prov = json.loads(os.environ[SEED_PROVENANCE_ENV_VAR])
        self.assertEqual(prov["parent_node_id"], self.node_id)
        self.assertTrue(prov["parent_content_hash"].startswith("sha256:"))

    def test_bad_source_reports_error_no_raise(self) -> None:
        result = stage_seed_from_cli(
            seed_from_ara="/definitely/does/not/exist",
            seed_node_id=None,
            project_dir=self.tmp,
        )
        self.assertFalse(result.seed_used)
        self.assertIsNotNone(result.error)
        # Env vars must not have been touched by the failing path.
        self.assertNotIn(SEED_ENV_VAR, os.environ)


class FinalizeARAForIdeaTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # Set up a child experiment ready for its own ARA export.
        self.project = self.tmp / "child"
        self.exp = self.project / "02_experiments" / "20260702_child_idea"
        (self.exp / "logs" / "0-run").mkdir(parents=True)
        _write_journal(
            self.exp / "logs",
            "0-run",
            [
                {
                    "id": "n0",
                    "step": 0,
                    "code": "print('child')",
                    "_term_out": ["ok\n"],
                    "metric": {"value": 0.75, "maximize": True, "name": "acc"},
                    "is_buggy": False,
                    "parent_id": None,
                    "children": [],
                }
            ],
        )
        self._orig_prov = os.environ.get(SEED_PROVENANCE_ENV_VAR)

    def tearDown(self) -> None:
        if self._orig_prov is None:
            os.environ.pop(SEED_PROVENANCE_ENV_VAR, None)
        else:
            os.environ[SEED_PROVENANCE_ENV_VAR] = self._orig_prov
        self._tmp.cleanup()

    def test_finalize_pipes_provenance_from_env(self) -> None:
        os.environ[SEED_PROVENANCE_ENV_VAR] = json.dumps(
            {
                "parent_ara_root": "/tmp/parent-ara",
                "parent_node_id": "seed_node",
                "parent_content_hash": "sha256:" + "a" * 64,
            }
        )
        result = finalize_ara_for_idea(
            project_dir=self.project,
            exp_dir=self.exp,
            idea={"Name": "child_idea"},
            timestamp="20260702",
            bfts_config_path=None,
            model_spec={"writeup": "opus"},
            writing_profile=None,
        )
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.export)
        manifest = json.loads(result.export.manifest_path.read_text())
        self.assertEqual(manifest["provenance"]["parent_node_id"], "seed_node")

    def test_finalize_skips_reexec_when_flag_disabled(self) -> None:
        result = finalize_ara_for_idea(
            project_dir=self.project,
            exp_dir=self.exp,
            idea={"Name": "child_idea"},
            timestamp="20260702",
            bfts_config_path=None,
            model_spec={},
            writing_profile=None,
            run_reexec=False,
        )
        self.assertIsNone(result.reexec_summary)

    def test_bad_provenance_env_is_ignored(self) -> None:
        os.environ[SEED_PROVENANCE_ENV_VAR] = "not json"
        result = finalize_ara_for_idea(
            project_dir=self.project,
            exp_dir=self.exp,
            idea={"Name": "child_idea"},
            timestamp="20260702",
            bfts_config_path=None,
            model_spec={},
            writing_profile=None,
        )
        self.assertIsNone(result.error)
        manifest = json.loads(result.export.manifest_path.read_text())
        # Bad env should not attach a provenance block.
        self.assertNotIn("provenance", manifest)


class SummaryFormattingTest(unittest.TestCase):
    def test_seed_stage_summary_empty_when_unused(self) -> None:
        from ai_scientist.utils.ara_pipeline import SeedStageResult

        self.assertEqual(summarise_seed_stage(SeedStageResult(seed_used=False)), "")

    def test_finalize_summary_reports_missing_export(self) -> None:
        from ai_scientist.utils.ara_pipeline import ARAFinalizeResult

        lines = summarise_finalize(1, ARAFinalizeResult(error="boom"))
        self.assertEqual(len(lines), 1)
        self.assertIn("ARA export 失败", lines[0])


if __name__ == "__main__":
    unittest.main()
