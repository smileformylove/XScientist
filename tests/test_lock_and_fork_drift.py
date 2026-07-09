"""Regression tests for two related bug classes:

1. Fork-metadata drift after the seed_hash_binding fix (iter-3, 5d2c7b2)
   left the fork's metrics.json advertising the parent's
   content_hash_inputs and the fork's exploration_graph node entry
   missing content_hash_inputs / llm_call_refs entirely.

2. Manifest.lock absent from two producers (`run_ara_fork.py fork` and
   `export_minimal_ara`), leaving those ARAs "unlocked" per the
   immutability layer promised in SPEC.md.

Both classes are treated as one commit so the recipe stays symmetrical:
every producer that writes a manifest.json also writes the lock, and
every fork node advertises the inputs it was actually hashed against.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock
from ai_scientist.utils.ara_minimal import export_minimal_ara


REPO_ROOT = Path(__file__).resolve().parents[1]
FORK_SCRIPT = REPO_ROOT / "run_ara_fork.py"


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


def _fork(ara_root: Path, node_id: str, dest: Path) -> None:
    subprocess.run(
        [sys.executable, str(FORK_SCRIPT), "fork",
         "--ara", str(ara_root), "--node-id", node_id, "--dest", str(dest)],
        capture_output=True, text=True, check=True,
    )


class ForkMetadataDriftTest(unittest.TestCase):
    """Bug class 1: fork's metrics.json + exploration_graph node must match
    the seed-bound hash the fork actually computes."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp)
        self.fork_root = self.tmp / "forked"
        _fork(self.ara_root, self.node_id, self.fork_root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fork_bundle_metrics_content_hash_inputs_declares_seed(self) -> None:
        metrics_path = self.fork_root / "nodes" / self.node_id / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        inputs = metrics.get("content_hash_inputs")
        self.assertIsInstance(inputs, list, msg=f"metrics.json missing content_hash_inputs: {metrics!r}")
        self.assertIn("seed", inputs,
                      msg=f"fork's metrics.json content_hash_inputs must declare 'seed': {inputs!r}")

    def test_fork_bundle_exploration_graph_node_carries_inputs_and_refs(self) -> None:
        graph = json.loads((self.fork_root / "exploration_graph.json").read_text(encoding="utf-8"))
        nodes = graph.get("nodes") or []
        self.assertEqual(len(nodes), 1, msg=f"expected 1 fork node, got {len(nodes)}")
        node = nodes[0]
        self.assertIn("content_hash_inputs", node,
                      msg=f"fork node entry missing content_hash_inputs: {node!r}")
        self.assertIn("seed", node["content_hash_inputs"],
                      msg=f"fork node content_hash_inputs must declare 'seed': {node['content_hash_inputs']!r}")
        # llm_call_refs must be present; empty list is the sanctioned default.
        self.assertIn("llm_call_refs", node,
                      msg=f"fork node entry missing llm_call_refs key: {node!r}")
        self.assertIsInstance(node["llm_call_refs"], list)


class ManifestLockCoverageTest(unittest.TestCase):
    """Bug class 2: every producer that writes manifest.json must also
    write manifest.lock so verify-lock doesn't report state=unlocked."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fork_writes_manifest_lock(self) -> None:
        _, ara_root, node_id = _seed_project(self.tmp)
        fork_root = self.tmp / "forked"
        _fork(ara_root, node_id, fork_root)
        self.assertTrue((fork_root / "manifest.lock").exists(),
                        msg="cmd_fork must write manifest.lock alongside manifest.json")
        report = verify_manifest_lock(fork_root)
        self.assertTrue(report["ok"], msg=f"verify_manifest_lock report: {report!r}")
        self.assertEqual(report["state"], "clean",
                         msg=f"expected state=clean, got {report!r}")

    def test_export_minimal_ara_writes_manifest_lock(self) -> None:
        pdf = self.tmp / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal body")
        result = export_minimal_ara(
            project_dir=self.tmp / "paper_dir",
            manuscript_pdf=pdf,
            idea={"Name": "no-search"},
            timestamp="20260702",
        )
        self.assertTrue((result.root / "manifest.lock").exists(),
                        msg="export_minimal_ara must write manifest.lock")
        report = verify_manifest_lock(result.root)
        self.assertTrue(report["ok"], msg=f"verify_manifest_lock report: {report!r}")
        self.assertEqual(report["state"], "clean",
                         msg=f"expected state=clean, got {report!r}")


if __name__ == "__main__":
    unittest.main()
