"""End-to-end: journal with llm_call_refs → exported ARA nodes have the
prompt-bound content_hash + content_hash_inputs field."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import hash_node_payload
from ai_scientist.utils.ara_artifact import export_ara


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _minimal_project(tmp: Path, nodes: list[dict]) -> tuple[Path, Path]:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260709_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(exp / "logs", "0-run", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": "idea", "Title": "T"}), encoding="utf-8")
    return project, exp


class LLMRefsFlowIntoExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # Backward compatibility: no llm_call_refs on the node → old hash
    # ------------------------------------------------------------------
    def test_node_without_refs_hashes_as_before(self) -> None:
        project, exp = _minimal_project(self.tmp, [{
            "id": "n1", "step": 0, "code": "print('ok')",
            "_term_out": [], "metric": {"value": 0.5, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        }])
        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea", "Title": "T"})
        ara = Path(result.root)
        node_metrics = json.loads((ara / "nodes" / "n1" / "metrics.json").read_text())

        expected = hash_node_payload(
            code="print('ok')",
            metric={"value": 0.5, "maximize": True, "name": "acc"},
        )
        self.assertEqual(node_metrics["content_hash"], expected)
        self.assertEqual(node_metrics["content_hash_inputs"], ["code", "metric"])

    # ------------------------------------------------------------------
    # Forward compatibility: llm_call_refs present → hash flips, inputs update
    # ------------------------------------------------------------------
    def test_node_with_refs_binds_them_into_hash(self) -> None:
        prompt_hash = "sha256:" + "e" * 64
        project, exp = _minimal_project(self.tmp, [{
            "id": "n1", "step": 0, "code": "print('ok')",
            "_term_out": [], "metric": {"value": 0.5, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
            "llm_call_refs": [prompt_hash],
        }])
        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea", "Title": "T"})
        ara = Path(result.root)
        node_metrics = json.loads((ara / "nodes" / "n1" / "metrics.json").read_text())

        # 1. hash must differ from the unbounded version
        unbounded = hash_node_payload(
            code="print('ok')",
            metric={"value": 0.5, "maximize": True, "name": "acc"},
        )
        self.assertNotEqual(node_metrics["content_hash"], unbounded)
        # 2. hash must equal what we get calling hash_node_payload directly
        expected = hash_node_payload(
            code="print('ok')",
            metric={"value": 0.5, "maximize": True, "name": "acc"},
            llm_call_hashes=[prompt_hash],
        )
        self.assertEqual(node_metrics["content_hash"], expected)
        # 3. inputs must include llm_calls
        self.assertEqual(node_metrics["content_hash_inputs"],
                         ["code", "metric", "llm_calls"])

        # 4. exploration_graph must surface the refs
        graph = json.loads((ara / "exploration_graph.json").read_text())
        n1 = next(n for n in graph["nodes"] if n["id"] == "n1")
        self.assertEqual(n1["llm_call_refs"], [prompt_hash])
        self.assertEqual(n1["content_hash_inputs"], ["code", "metric", "llm_calls"])

    def test_two_nodes_same_code_diff_prompts_hash_differently(self) -> None:
        # This is THE property we care about — "code identical, prompt different"
        # nodes must be distinguishable at the hash layer.
        h_a = "sha256:" + "a" * 64
        h_b = "sha256:" + "b" * 64
        project, exp = _minimal_project(self.tmp, [
            {
                "id": "na", "step": 0, "code": "print('same')",
                "_term_out": [], "metric": {"value": 0.9, "maximize": True, "name": "acc"},
                "is_buggy": False, "parent_id": None, "children": [],
                "llm_call_refs": [h_a],
            },
            {
                "id": "nb", "step": 1, "code": "print('same')",
                "_term_out": [], "metric": {"value": 0.9, "maximize": True, "name": "acc"},
                "is_buggy": False, "parent_id": None, "children": [],
                "llm_call_refs": [h_b],
            },
        ])
        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea", "Title": "T"})
        ara = Path(result.root)
        ma = json.loads((ara / "nodes" / "na" / "metrics.json").read_text())
        mb = json.loads((ara / "nodes" / "nb" / "metrics.json").read_text())
        self.assertNotEqual(ma["content_hash"], mb["content_hash"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
