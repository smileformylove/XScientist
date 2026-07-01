"""Unit tests for the ARA (Agent-Native Research Artifact) exporter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import (
    ARA_SCHEMA_VERSION,
    ara_dir_for_idea,
    ara_root_for_project,
    export_ara,
    iter_ara_exports,
    update_manifest_claim_count,
)
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
        # Manifest updated after.
        update_manifest_claim_count(export.manifest_path, summary["claim_count"])
        manifest = json.loads(export.manifest_path.read_text())
        self.assertEqual(manifest["counts"]["claims"], 3)


if __name__ == "__main__":
    unittest.main()
