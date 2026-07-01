"""Tests for the ARA A/B harness."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_scientist.experiments.ara_ab import harness
from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_seed import (
    build_seed_manifest_from_ara_node,
    stage_seed_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _seed_parent_project(tmp: Path) -> tuple[Path, str]:
    project = tmp / "parent"
    exp = project / "02_experiments" / "20260701_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(
        exp / "logs",
        "0-run",
        [
            {
                "id": "n1",
                "step": 0,
                "code": "print('seeded from parent')",
                "_term_out": ["ok\n"],
                "metric": {"value": 0.5, "maximize": True, "name": "acc", "description": ""},
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
    return result.root, "n1"


class HashOverlapTest(unittest.TestCase):
    def test_overlap_and_jaccard(self) -> None:
        a = harness.ArmResult(
            arm="baseline", started_at="", finished_at="", duration_seconds=1.0,
            llm_calls=1, used_seed=False,
            node_content_hashes=["sha256:a", "sha256:b"],
        )
        b = harness.ArmResult(
            arm="ara_seed", started_at="", finished_at="", duration_seconds=0.5,
            llm_calls=0, used_seed=True,
            node_content_hashes=["sha256:b", "sha256:c"],
        )
        overlap = harness.compute_hash_overlap(a, b)
        self.assertEqual(overlap["shared_count"], 1)
        self.assertEqual(overlap["shared"], ["sha256:b"])
        self.assertAlmostEqual(overlap["jaccard"], 1 / 3, places=5)

    def test_overlap_empty_when_missing_arm(self) -> None:
        self.assertEqual(harness.compute_hash_overlap(None, None), {})


class VerdictTest(unittest.TestCase):
    def test_seed_saved_llm_calls(self) -> None:
        baseline = harness.ArmResult(
            arm="baseline", started_at="", finished_at="", duration_seconds=1.0,
            llm_calls=1, used_seed=False,
        )
        seeded = harness.ArmResult(
            arm="ara_seed", started_at="", finished_at="", duration_seconds=0.1,
            llm_calls=0, used_seed=True,
        )
        verdict = harness.build_verdict(baseline, seeded)
        self.assertEqual(verdict["conclusion"], "seed_saved_llm_calls")
        self.assertEqual(verdict["llm_calls_saved_on_draft"], 1)
        self.assertGreater(verdict["wall_clock_speedup"], 1.0)

    def test_seed_did_not_short_circuit(self) -> None:
        baseline = harness.ArmResult(
            arm="baseline", started_at="", finished_at="", duration_seconds=1.0,
            llm_calls=1, used_seed=False,
        )
        seeded = harness.ArmResult(
            arm="ara_seed", started_at="", finished_at="", duration_seconds=1.0,
            llm_calls=1, used_seed=False,
        )
        self.assertEqual(
            harness.build_verdict(baseline, seeded)["conclusion"],
            "seed_did_not_short_circuit",
        )


class StubEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        ara_root, node_id = _seed_parent_project(self.tmp)
        manifest = build_seed_manifest_from_ara_node(ara_root=ara_root, node_id=node_id)
        self.seed_path = stage_seed_manifest(manifest, workspace_dir=self.tmp / "ws")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_ab_stub_writes_report_and_short_circuits(self) -> None:
        out_dir = self.tmp / "out"
        report = harness.run_ab_stub(
            seed_manifest_path=self.seed_path,
            out_dir=out_dir,
            idea_hint="stub-idea",
        )
        # Written to disk.
        report_path = out_dir / "ab_report.json"
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text())
        self.assertEqual(payload["schema"], "ara.ab_report.v1")
        self.assertEqual(payload["mode"], "stub")

        # Baseline called the stubbed LLM; seed arm did not.
        self.assertEqual(payload["baseline"]["llm_calls"], 1)
        self.assertFalse(payload["baseline"]["used_seed"])
        self.assertEqual(payload["ara_seed"]["llm_calls"], 0)
        self.assertTrue(payload["ara_seed"]["used_seed"])

        # Verdict must flag the saving.
        self.assertEqual(payload["verdict"]["conclusion"], "seed_saved_llm_calls")
        self.assertEqual(payload["verdict"]["llm_calls_saved_on_draft"], 1)

    def test_cli_entrypoint_exits_zero_on_short_circuit(self) -> None:
        out_dir = self.tmp / "out_cli"
        completed = subprocess.run(
            [
                sys.executable, "-m", "ai_scientist.experiments.ara_ab.harness",
                "stub", "--seed-manifest", str(self.seed_path),
                "--out-dir", str(out_dir),
            ],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("ARA A/B report", completed.stdout)


class RealDryRunTest(unittest.TestCase):
    """`real --dry-run` should print the commands without invoking run_project."""

    def test_dry_run_records_planned_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = harness.run_ab_real(
                project_dir_baseline=tmp_path / "b",
                project_dir_seeded=tmp_path / "s",
                seed_from_ara="/nonexistent/fork",
                seed_node_id=None,
                run_project_args=["--topic", "does_not_exist.md"],
                out_dir=tmp_path / "out",
                dry_run=True,
            )
            self.assertEqual(report.mode, "real_dry_run")
            self.assertEqual(report.baseline.exit_code, 0)
            self.assertEqual(report.ara_seed.exit_code, 0)
            self.assertTrue(any("run_project.py" in n for n in report.baseline.notes))
            self.assertTrue(any("--seed-from-ara" in n for n in report.ara_seed.notes))


if __name__ == "__main__":
    unittest.main()
