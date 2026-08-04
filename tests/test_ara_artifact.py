"""Unit tests for the ARA (Agent-Native Research Artifact) exporter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import analyze_exploration_graph
from ai_scientist.utils.ara_artifact import (
    ARA_SCHEMA_VERSION,
    ara_dir_for_idea,
    ara_root_for_project,
    export_ara,
    iter_ara_exports,
    update_manifest_claim_count,
)
from ai_scientist.utils.ara_graph import render_exploration_graph_html
from ai_scientist.utils.claim_registry import (
    _CLAIM_RE,
    scan_tex_for_claims,
    write_claims_into_ara,
)


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _write_serialized_journal(
    logs_dir: Path,
    run_name: str,
    *,
    nodes: list[dict],
    node2parent: dict[str, str],
) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": node2parent, "__version": "2"}),
        encoding="utf-8",
    )


class ARAExporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project_dir = Path(self._tmp.name)
        self.exp_dir = self.project_dir / "02_experiments" / "20260701_abc_idea"
        (self.exp_dir / "logs" / "0-run").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed_journal(self) -> None:
        nodes = [
            {
                "id": "node_root",
                "step": 0,
                "code": "print('draft')",
                "_term_out": ["hello\n"],
                "metric": {"value": 0.1, "maximize": True, "name": "acc", "description": ""},
                "analysis": "baseline drafted",
                "is_buggy": False,
                "execution_backend": "docker",
                "execution_isolation": {
                    "isolated": True,
                    "network": "none",
                    "docker_image": "xscientist-exec@sha256:test",
                },
                "parent_id": None,
                "children": ["node_child"],
                "plots": ["logs/0-run/plot.png"],
                "plot_paths": [str(self.exp_dir / "plot.png")],
            },
            {
                "id": "node_child",
                "step": 1,
                "code": "raise RuntimeError('oops')",
                "_term_out": ["Traceback\n", "RuntimeError: oops\n"],
                "metric": None,
                "analysis": "bug",
                "is_buggy": True,
                "exc_type": "RuntimeError",
                "parent_id": "node_root",
                "children": [],
            },
        ]
        _write_journal(self.exp_dir / "logs", "0-run", nodes)

    def test_export_writes_manifest_nodes_and_graph(self) -> None:
        self._seed_journal()
        (self.exp_dir / "experiment_registry.jsonl").write_text(
            json.dumps({"task": "t1", "status": "planned"}) + "\n", encoding="utf-8"
        )
        (self.exp_dir / "repair_attempts.jsonl").write_text(
            json.dumps({"attempt": 1}) + "\n", encoding="utf-8"
        )

        result = export_ara(
            project_dir=self.project_dir,
            exp_dir=self.exp_dir,
            idea={"Name": "abc_idea", "Title": "ABC"},
            timestamp="20260701",
            bfts_config_path=None,
            model_spec={"writeup": "opus", "review": "sonnet"},
            writing_profile="test-profile",
        )

        self.assertEqual(result.node_count, 2)
        self.assertTrue(result.manifest_path.exists())
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual(manifest["schema_version"], ARA_SCHEMA_VERSION)
        self.assertEqual(manifest["counts"]["nodes"], 2)
        self.assertEqual(manifest["counts"]["buggy_nodes"], 1)
        self.assertEqual(manifest["idea"]["name"], "abc_idea")
        self.assertIn("env", manifest["references"])
        self.assertIn("exploration_graph_visualization", manifest["references"])
        self.assertTrue((result.root / "exploration_graph.html").exists())
        self.assertTrue((result.root / "exploration_graph.summary.json").exists())
        graph = json.loads((result.root / "exploration_graph.json").read_text())
        self.assertTrue(graph["dag"]["is_dag"])
        self.assertEqual(graph["dag"]["topological_order"], ["node_root", "node_child"])
        # Model fingerprint should hash the spec.
        fp = json.loads(
            (result.root / manifest["references"]["env"]["model_fingerprint"]).read_text()
        )
        self.assertIn("fingerprint", fp)
        # Repair history is aggregated.
        self.assertTrue((result.root / "repair_history.jsonl").exists())
        # Every node has code + term_out.
        for node_id in ("node_root", "node_child"):
            self.assertTrue((result.root / "nodes" / node_id / "code.py").exists())
            self.assertTrue((result.root / "nodes" / node_id / "term_out.log").exists())
            self.assertTrue((result.root / "nodes" / node_id / "metrics.json").exists())

        root_metrics = json.loads(
            (result.root / "nodes" / "node_root" / "metrics.json").read_text()
        )
        self.assertEqual(root_metrics["execution_backend"], "docker")
        self.assertTrue(root_metrics["execution_isolation"]["isolated"])
        root_graph_node = {
            node["id"]: node for node in graph["nodes"]
        }["node_root"]
        self.assertEqual(root_graph_node["execution_backend"], "docker")

        # ARA lives under `<project_dir>/ara/`.
        self.assertTrue(str(result.root).startswith(str(ara_root_for_project(self.project_dir))))

    def test_missing_journal_recorded_not_fatal(self) -> None:
        result = export_ara(
            project_dir=self.project_dir,
            exp_dir=self.exp_dir,
            idea={"Name": "abc"},
            timestamp="20260701",
        )
        self.assertEqual(result.node_count, 0)
        self.assertTrue(any("exploration_graph" in note for note in result.missing))
        manifest = json.loads(result.manifest_path.read_text())
        self.assertIn("missing", manifest)

    def test_iter_ara_exports(self) -> None:
        self._seed_journal()
        export_ara(
            project_dir=self.project_dir,
            exp_dir=self.exp_dir,
            idea={"Name": "abc"},
            timestamp="20260701",
        )
        manifests = list(iter_ara_exports(self.project_dir))
        self.assertEqual(len(manifests), 1)
        self.assertTrue(manifests[0].name == "manifest.json")

    def test_ara_dir_for_idea_slug(self) -> None:
        path = ara_dir_for_idea(self.project_dir, "Complex Idea/with weird chars!", timestamp="ts")
        self.assertTrue(path.name.startswith("ts_"))
        self.assertNotIn("/", path.name)
        self.assertNotIn(" ", path.name)

    def test_graph_analyzer_detects_cycles(self) -> None:
        graph = {
            "nodes": [
                {"id": "a", "parent_id": "b", "children": ["b"]},
                {"id": "b", "parent_id": "a", "children": ["a"]},
            ],
            "edges": [{"parent": "a", "child": "b"}, {"parent": "b", "child": "a"}],
        }
        dag = analyze_exploration_graph(graph)
        self.assertFalse(dag["is_dag"])
        self.assertIn("cycle_detected", {issue["code"] for issue in dag["issues"]})

    def test_graph_analyzer_uses_parent_links_when_edges_are_absent(self) -> None:
        graph = {
            "nodes": [
                {"id": "root", "parent_id": None, "children": []},
                {"id": "child", "parent_id": "root", "children": []},
            ],
            "edges": [],
        }
        dag = analyze_exploration_graph(graph)
        self.assertTrue(dag["is_dag"])
        self.assertEqual(dag["edge_count"], 1)
        self.assertEqual(dag["topological_order"], ["root", "child"])

    def test_export_uses_serialized_journal_node2parent_for_dag_edges(self) -> None:
        _write_serialized_journal(
            self.exp_dir / "logs",
            "serialized-run",
            nodes=[
                {
                    "id": "root",
                    "step": 0,
                    "code": "print('root')",
                    "_term_out": [],
                    "metric": {"value": 0.1, "maximize": True, "name": "score"},
                    "is_buggy": False,
                },
                {
                    "id": "child",
                    "step": 1,
                    "code": "print('child')",
                    "_term_out": [],
                    "metric": {"value": 0.2, "maximize": True, "name": "score"},
                    "is_buggy": False,
                },
            ],
            node2parent={"child": "root"},
        )
        result = export_ara(
            project_dir=self.project_dir,
            exp_dir=self.exp_dir,
            idea={"Name": "serialized"},
            timestamp="20260701",
        )
        graph = json.loads((result.root / "exploration_graph.json").read_text())
        by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertNotIn("parent_id", by_id["child"])
        self.assertNotIn("children", by_id["root"])
        self.assertEqual(graph["topology_encoding"], "edges")
        self.assertEqual(graph["edges"], [{"parent": "root", "child": "child", "stage": "serialized-run"}])
        self.assertEqual(graph["dag"]["topological_order"], ["root", "child"])

    def test_export_dedupes_multistage_carryover_nodes_by_id_and_hash(self) -> None:
        root_node = {
            "id": "root",
            "step": 0,
            "code": "print('root')",
            "_term_out": [],
            "metric": {"value": 0.1, "maximize": True, "name": "score"},
            "is_buggy": False,
        }
        _write_serialized_journal(
            self.exp_dir / "logs",
            "stage-1",
            nodes=[root_node],
            node2parent={},
        )
        _write_serialized_journal(
            self.exp_dir / "logs",
            "stage-2",
            nodes=[
                dict(root_node),
                {
                    "id": "child",
                    "step": 1,
                    "code": "print('child')",
                    "_term_out": [],
                    "metric": {"value": 0.2, "maximize": True, "name": "score"},
                    "is_buggy": False,
                },
            ],
            node2parent={"child": "root"},
        )
        result = export_ara(
            project_dir=self.project_dir,
            exp_dir=self.exp_dir,
            idea={"Name": "multistage"},
            timestamp="20260701",
        )
        graph = json.loads((result.root / "exploration_graph.json").read_text())
        self.assertEqual([node["id"] for node in graph["nodes"]], ["root", "child"])
        root = {node["id"]: node for node in graph["nodes"]}["root"]
        self.assertEqual(root["stages"], ["stage-1", "stage-2"])
        self.assertNotIn("children", root)
        self.assertEqual(
            graph["edges"],
            [{"parent": "root", "child": "child", "stage": "stage-2"}],
        )
        self.assertEqual(graph["dag"]["topological_order"], ["root", "child"])

    def test_html_renderer_merges_children_links_like_dag_analyzer(self) -> None:
        html = render_exploration_graph_html(
            {
                "nodes": [
                    {"id": "root", "children": ["child"], "parent_id": None},
                    {"id": "child", "children": [], "parent_id": None},
                ],
                "edges": [],
            }
        )
        self.assertIn("Array.isArray(n.children)", html)
        self.assertIn("explicitEdges.push([String(n.id), String(child)])", html)


class ClaimRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project_dir = Path(self._tmp.name)
        self.exp_dir = self.project_dir / "02_experiments" / "20260701_x"
        (self.exp_dir / "logs" / "0-run").mkdir(parents=True)
        _write_journal(
            self.exp_dir / "logs",
            "0-run",
            [
                {
                    "id": "node_root",
                    "step": 0,
                    "code": "pass",
                    "_term_out": ["ok\n"],
                    "metric": {"value": 0.5, "maximize": True, "name": "f1", "description": ""},
                    "is_buggy": False,
                    "parent_id": None,
                    "children": [],
                },
            ],
        )
        latex_dir = self.exp_dir / "latex"
        latex_dir.mkdir(parents=True)
        self.tex_path = latex_dir / "template.tex"
        self.tex_path.write_text(
            "\\documentclass{article}\n"
            "\\providecommand{\\claimref}[2][]{}\n"
            "\\begin{document}\n"
            "We report F1=0.82\\claimref{node_root}.\n"
            "Ablation shows drop\\claimref[stage=ablation]{node_missing}.\n"
            "Sanity check\\claimref{node_root}. % duplicate node id, different line\n"
            "\\end{document}\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_regex_extracts_claims_and_options(self) -> None:
        text = "cite\\claimref[key=val, flag]{node42} then\\claimref{node_x}"
        matches = list(_CLAIM_RE.finditer(text))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].group("node"), "node42")
        self.assertEqual(matches[1].group("node"), "node_x")

    def test_scan_tex_produces_claims_with_line_info(self) -> None:
        claims = scan_tex_for_claims(self.tex_path, tex_root=self.tex_path.parent)
        self.assertEqual(len(claims), 3)
        ids = [c.node_id for c in claims]
        self.assertIn("node_root", ids)
        self.assertIn("node_missing", ids)
        self.assertEqual(claims[1].options.get("stage"), "ablation")

    def test_write_claims_into_ara_flags_unresolved(self) -> None:
        export = export_ara(
            project_dir=self.project_dir,
            exp_dir=self.exp_dir,
            idea={"Name": "x"},
            timestamp="20260701",
        )
        summary = write_claims_into_ara(
            ara_dir=export.root, tex_files=[self.tex_path]
        )
        self.assertEqual(summary["claim_count"], 3)
        self.assertEqual(summary["resolved_count"], 2)
        self.assertIn("node_missing", summary["unresolved_node_ids"])
        self.assertTrue((export.root / "claims" / "_index.json").exists())
        claim_files = [
            path for path in (export.root / "claims").glob("*.json")
            if not path.name.startswith("_")
        ]
        stored = [json.loads(path.read_text()) for path in claim_files]
        resolved = next(row for row in stored if row["node_id"] == "node_root")
        self.assertNotIn("node", resolved)
        self.assertTrue(resolved["claim_hash"].startswith("sha256:"))
        self.assertEqual(len(resolved["evidence_refs"]), 1)
        self.assertTrue(resolved["source"]["document_hash"].startswith("sha256:"))
        # Manifest updated after.
        update_manifest_claim_count(export.manifest_path, summary["claim_count"])
        manifest = json.loads(export.manifest_path.read_text())
        self.assertEqual(manifest["counts"]["claims"], 3)


if __name__ == "__main__":
    unittest.main()
