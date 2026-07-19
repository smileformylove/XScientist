"""Producers that stamp ``is_seed_node: True`` must fold the seed marker into
their advertised ``content_hash``.

Two known producers besides ``export_ara`` mint seed nodes:

- ``ai_scientist.utils.ara_minimal.export_minimal_ara`` (manuscript-only
  ARAs from ``ai_scientist.apps.batch``).
- ``run_ara_fork.py`` ``cmd_fork`` (the fork command reuses the parent's
  code+metric and marks the fork ``is_seed_node: True``).

Under the seed-role binding rule (see ``hash_node_payload(..., is_seed=True)``)
their emitted ``content_hash`` must equal what a downstream consumer would
recompute for the same code+metric with ``is_seed=True`` — NOT the parent's
original hash.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import hash_node_payload
from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_minimal import export_minimal_ara


REPO_ROOT = Path(__file__).resolve().parents[1]
FORK_COMMAND = [sys.executable, "-m", "ai_scientist.apps.ara"]


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


class ARAMinimalSeedBindingTest(unittest.TestCase):
    def test_ara_minimal_synthetic_node_hash_matches_declared_seed_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "proj"
            project.mkdir()
            result = export_minimal_ara(
                project_dir=project,
                manuscript_pdf=None,
                idea={"Name": "manuscript_only_idea", "Title": "T"},
                timestamp="20260710",
            )
            graph = json.loads((result.root / "exploration_graph.json").read_text())
            self.assertEqual(len(graph["nodes"]), 1)
            node = graph["nodes"][0]
            self.assertTrue(node["is_seed_node"])
            expected = hash_node_payload(code="", metric=None, is_seed=True)
            self.assertEqual(node["content_hash"], expected)

            # metrics.json must agree with the graph node — otherwise diffing
            # the ARA against itself would report a spurious hash mismatch.
            metrics = json.loads(
                (result.root / "nodes" / node["id"] / "metrics.json").read_text()
            )
            self.assertEqual(metrics["content_hash"], expected)


class ForkSeedBindingTest(unittest.TestCase):
    def _seed_project(self, tmp: Path, code: str, metric: dict) -> tuple[Path, str]:
        project = tmp / "project"
        exp = project / "02_experiments" / "20260710_idea"
        (exp / "logs" / "0-run").mkdir(parents=True)
        _write_journal(
            exp / "logs",
            "0-run",
            [
                {
                    "id": "n1",
                    "step": 0,
                    "code": code,
                    "_term_out": [],
                    "metric": metric,
                    "is_buggy": False,
                    "parent_id": None,
                    "children": [],
                },
            ],
        )
        (exp / "idea.json").write_text(
            json.dumps({"Name": "idea", "Title": "T"}), encoding="utf-8"
        )
        result = export_ara(
            project_dir=project,
            exp_dir=exp,
            idea={"Name": "idea", "Title": "T"},
            timestamp="20260710",
        )
        return Path(result.root), "n1"

    def test_cmd_fork_hash_computes_from_parent_with_is_seed_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code = "print('parent')"
            metric = {"value": 0.5, "maximize": True, "name": "acc"}
            ara_root, node_id = self._seed_project(tmp_path, code, metric)

            # Parent's own content_hash: no is_seed binding (regular explored node).
            parent_metrics = json.loads(
                (ara_root / "nodes" / node_id / "metrics.json").read_text()
            )
            parent_hash = parent_metrics["content_hash"]
            expected_plain = hash_node_payload(code=code, metric=metric)
            self.assertEqual(parent_hash, expected_plain)

            dest = tmp_path / "forked"
            subprocess.run(
                [
                    *FORK_COMMAND, "fork",
                    "--ara", str(ara_root),
                    "--node-id", node_id,
                    "--dest", str(dest),
                ],
                capture_output=True, text=True, check=True,
            )
            fork_graph = json.loads((dest / "exploration_graph.json").read_text())
            fork_node = fork_graph["nodes"][0]
            self.assertTrue(fork_node["is_seed_node"])

            expected_fork = hash_node_payload(code=code, metric=metric, is_seed=True)
            # The fork's OWN content_hash must reflect is_seed=True, NOT the
            # parent's original hash.
            self.assertEqual(fork_node["content_hash"], expected_fork)
            self.assertNotEqual(fork_node["content_hash"], parent_hash)

            # Provenance still points at the parent hash — those are references
            # TO the parent, not the fork's own hash.
            fork_meta = json.loads((dest / "fork.json").read_text())
            self.assertEqual(fork_meta["source_content_hash"], parent_hash)
            fork_manifest = json.loads((dest / "manifest.json").read_text())
            self.assertEqual(
                fork_manifest["provenance"]["parent_content_hash"], parent_hash
            )

            # metrics.json in the fork bundle must agree with the graph.
            fork_metrics = json.loads(
                (dest / "nodes" / node_id / "metrics.json").read_text()
            )
            self.assertEqual(fork_metrics["content_hash"], expected_fork)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
