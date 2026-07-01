"""Tests for ARA fork CLI + re-execution verifier."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_reexec import (
    reexec_ara,
    reexec_enabled,
    reexec_node,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORK_SCRIPT = REPO_ROOT / "run_ara_fork.py"


def _load_fork_module():
    spec = importlib.util.spec_from_file_location("_ara_fork", FORK_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _seed_project(tmp: Path, *, metric_value: float = 0.42) -> tuple[Path, Path, str]:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260701_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    code = textwrap.dedent(
        f"""
        import json
        result = {{"name": "acc", "value": {metric_value}}}
        print("ARA_METRIC=" + json.dumps(result))
        """
    ).strip()
    _write_journal(
        exp / "logs",
        "0-run",
        [
            {
                "id": "n1",
                "step": 0,
                "code": code,
                "_term_out": [f"ARA_METRIC={{\"name\": \"acc\", \"value\": {metric_value}}}\n"],
                "metric": {"value": metric_value, "maximize": True, "name": "acc", "description": ""},
                "is_buggy": False,
                "parent_id": None,
                "children": [],
            },
        ],
    )
    result = export_ara(
        project_dir=project,
        exp_dir=exp,
        idea={"Name": "idea"},
        timestamp="20260701",
    )
    return project, result.root, "n1"


class RunBundleTest(unittest.TestCase):
    def test_bundle_files_present_and_run_sh_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, ara_root, node_id = _seed_project(Path(tmp))
            node_dir = ara_root / "nodes" / node_id
            self.assertTrue((node_dir / "code.py").exists())
            self.assertTrue((node_dir / "env.json").exists())
            run_sh = node_dir / "run.sh"
            self.assertTrue(run_sh.exists())
            # POSIX permission bit set (best-effort check on unix).
            if os.name == "posix":
                self.assertTrue(bool(run_sh.stat().st_mode & stat.S_IXUSR))
            env = json.loads((node_dir / "env.json").read_text())
            self.assertEqual(env["node_id"], node_id)
            self.assertEqual(env["code_file"], "code.py")


class ForkCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_inspect_prints_key_fields(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(FORK_SCRIPT), "inspect", "--ara", str(self.ara_root), "--node-id", self.node_id],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("Node " + self.node_id, completed.stdout)
        self.assertIn("Metric:", completed.stdout)

    def test_exec_re_runs_and_writes_verify_report(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(FORK_SCRIPT), "exec", "--ara", str(self.ara_root), "--node-id", self.node_id],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        verify_dir = self.ara_root / "verify"
        self.assertTrue(verify_dir.exists())
        reports = list(verify_dir.glob(f"{self.node_id}_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        self.assertEqual(payload["node_id"], self.node_id)
        self.assertTrue(payload["comparison"]["within_tolerance"])

    def test_fork_copies_node_bundle(self) -> None:
        dest = self.tmp / "forked"
        completed = subprocess.run(
            [sys.executable, str(FORK_SCRIPT), "fork", "--ara", str(self.ara_root),
             "--node-id", self.node_id, "--dest", str(dest)],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("forked node", completed.stdout)
        self.assertTrue((dest / "node" / "code.py").exists())
        self.assertTrue((dest / "manifest.json").exists())
        self.assertTrue((dest / "fork.json").exists())


class ParseMetricTest(unittest.TestCase):
    def test_json_marker_wins(self) -> None:
        mod = _load_fork_module()
        stdout = "chatter\nARA_METRIC={\"name\": \"acc\", \"value\": 0.9}\nmore output\n"
        result = mod._parse_metric_from_stdout(stdout)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["value"], 0.9)

    def test_text_tail_fallback(self) -> None:
        mod = _load_fork_module()
        stdout = "training log ...\nmetric: 0.5\n"
        result = mod._parse_metric_from_stdout(stdout)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["value"], 0.5)

    def test_no_metric_returns_unavailable(self) -> None:
        mod = _load_fork_module()
        self.assertFalse(_load_fork_module()._parse_metric_from_stdout("nothing here")["available"])


class ReexecHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reexec_node_returns_within_tolerance(self) -> None:
        verdict = reexec_node(self.ara_root, self.node_id, timeout=60)
        self.assertEqual(verdict["node_id"], self.node_id)
        self.assertEqual(verdict["returncode"], 0)
        self.assertTrue(verdict["comparison"]["within_tolerance"])

    def test_reexec_ara_writes_batch_report(self) -> None:
        result = reexec_ara(self.ara_root, limit=2)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(Path(result["report_path"]).exists())
        payload = json.loads(Path(result["report_path"]).read_text())
        self.assertGreaterEqual(payload["verdict_count"], 1)

    def test_reexec_enabled_respects_env(self) -> None:
        orig = os.environ.get("AI_SCIENTIST_ARA_REEXEC")
        try:
            os.environ.pop("AI_SCIENTIST_ARA_REEXEC", None)
            self.assertFalse(reexec_enabled())
            os.environ["AI_SCIENTIST_ARA_REEXEC"] = "1"
            self.assertTrue(reexec_enabled())
        finally:
            if orig is None:
                os.environ.pop("AI_SCIENTIST_ARA_REEXEC", None)
            else:
                os.environ["AI_SCIENTIST_ARA_REEXEC"] = orig


if __name__ == "__main__":
    unittest.main()
