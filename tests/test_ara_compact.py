"""Non-destructive migration tests for compacted ARA successors."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import validate_ara
from ai_scientist.utils.ara_compact import compact_ara
from ai_scientist.utils.ara_manifest_lock import (
    verify_manifest_lock,
    write_manifest_lock,
)


class ARACompactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "source"
        self.source.mkdir()
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "created_at": "2026-08-04T00:00:00Z",
            "source_exp_dir": "/tmp/exp",
            "idea": {"name": "legacy"},
            "counts": {"nodes": 2, "edges": 1, "claims": 1},
            "references": {},
        }
        (self.source / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        write_manifest_lock(self.source, manifest)
        graph = {
            "schema_version": "ara.v1",
            "protocol_kind": "exploration_graph",
            "nodes": [
                {
                    "id": "root",
                    "parent_id": None,
                    "children": ["child"],
                    "content_hash": "sha256:" + "a" * 64,
                    "metric": {"value": 1},
                },
                {
                    "id": "child",
                    "parent_id": "root",
                    "children": [],
                    "content_hash": "sha256:" + "b" * 64,
                    "metric": {"value": 2},
                },
            ],
            "edges": [{"parent": "root", "child": "child"}],
            "counts": {"nodes": 2, "edges": 1},
            "dag": {"is_dag": True},
        }
        (self.source / "exploration_graph.json").write_text(
            json.dumps(graph), encoding="utf-8"
        )
        claims = self.source / "claims"
        claims.mkdir()
        claim = {
            "claim_id": "c1",
            "node_id": "child",
            "tex_file": "paper.tex",
            "line": 4,
            "context": "A claim",
            "options": {},
            "resolved": True,
            "node": graph["nodes"][1],
        }
        (claims / "c1.json").write_text(json.dumps(claim), encoding="utf-8")
        verify = self.source / "verify"
        verify.mkdir()
        report = {
            "schema": "ara.verify.v1",
            "node_id": "child",
            "ara_root": str(self.source),
            "started_at": "a",
            "finished_at": "b",
            "returncode": 0,
            "comparison": {"within_tolerance": True},
            "stdout_tail": "x" * 10000,
            "stderr_tail": "",
        }
        (verify / "child_1.json").write_text(json.dumps(report), encoding="utf-8")
        (self.source / "exploration_graph.html").write_text("derived", encoding="utf-8")

    def test_compaction_preserves_source_and_writes_conformant_successor(self) -> None:
        destination = self.tmp / "compact"
        result = compact_ara(self.source, destination)
        self.assertTrue(result["validation"]["ok"])

        source_claim = json.loads((self.source / "claims" / "c1.json").read_text())
        self.assertIn("node", source_claim)
        compact_claim = json.loads((destination / "claims" / "c1.json").read_text())
        self.assertNotIn("node", compact_claim)
        self.assertTrue(compact_claim["claim_hash"].startswith("sha256:"))
        self.assertEqual(compact_claim["evidence_refs"], ["sha256:" + "b" * 64])

        graph = json.loads((destination / "exploration_graph.json").read_text())
        self.assertEqual(graph["topology_encoding"], "edges")
        self.assertNotIn("parent_id", graph["nodes"][1])
        self.assertNotIn("children", graph["nodes"][0])
        self.assertNotIn("dag", graph)

        report = json.loads((destination / "verify" / "child_1.json").read_text())
        self.assertNotIn("stdout_tail", report)
        self.assertIn("stdout_ref", report)
        self.assertIn("verdict_ref", report)
        self.assertFalse((destination / "exploration_graph.html").exists())
        self.assertTrue((destination / "legacy" / "manifest.lock").exists())
        self.assertEqual(verify_manifest_lock(destination)["state"], "clean")
        self.assertTrue(validate_ara(destination).ok)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
