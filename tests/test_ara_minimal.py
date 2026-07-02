"""Tests for the manuscript-only ARA export path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import validate_ara
from ai_scientist.utils.ara_minimal import export_minimal_ara


class ExportMinimalARATest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _fake_pdf(self, name: str = "paper.pdf") -> Path:
        pdf = self.tmp / name
        pdf.write_bytes(b"%PDF-1.4 fake body")
        return pdf

    def test_returns_conformant_ara(self) -> None:
        pdf = self._fake_pdf()
        result = export_minimal_ara(
            project_dir=self.tmp / "paper_dir",
            manuscript_pdf=pdf,
            idea={"Name": "no-search"},
            timestamp="20260702",
        )
        self.assertTrue(result.manifest_path.exists())
        self.assertEqual(result.node_count, 1)
        report = validate_ara(result.root)
        self.assertTrue(report.ok, msg=[e.__dict__ for e in report.errors])

    def test_manuscript_pdf_copied_into_node_bundle(self) -> None:
        pdf = self._fake_pdf("mine.pdf")
        result = export_minimal_ara(
            project_dir=self.tmp / "paper_dir",
            manuscript_pdf=pdf,
            idea={"Name": "with-pdf"},
        )
        nodes_dirs = list((result.root / "nodes").iterdir())
        self.assertEqual(len(nodes_dirs), 1)
        self.assertTrue((nodes_dirs[0] / "mine.pdf").exists())
        # Manifest ``missing`` calls out the tree-less nature.
        manifest = json.loads(result.manifest_path.read_text())
        self.assertTrue(
            any("no journal.json" in note for note in manifest["missing"])
        )

    def test_no_pdf_still_produces_valid_ara(self) -> None:
        result = export_minimal_ara(
            project_dir=self.tmp / "paper_dir",
            manuscript_pdf=None,
            idea={"Name": "pdf-less"},
        )
        report = validate_ara(result.root)
        self.assertTrue(report.ok, msg=[e.__dict__ for e in report.errors])

    def test_provenance_survives_manifest_write(self) -> None:
        provenance = {
            "parent_ara_root": "/tmp/parent",
            "parent_node_id": "n1",
            "parent_content_hash": "sha256:" + "a" * 64,
            "parents": [
                {"role": "code", "parent_ara_root": "/tmp/parent",
                 "parent_node_id": "n1",
                 "parent_content_hash": "sha256:" + "a" * 64},
            ],
        }
        result = export_minimal_ara(
            project_dir=self.tmp / "child",
            manuscript_pdf=None,
            idea={"Name": "child"},
            provenance=provenance,
        )
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual(manifest["provenance"], provenance)
        # And the whole thing still validates.
        report = validate_ara(result.root)
        self.assertTrue(report.ok, msg=[e.__dict__ for e in report.errors])


if __name__ == "__main__":
    unittest.main()
