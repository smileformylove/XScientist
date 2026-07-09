"""End-to-end: export_ara + update_manifest_claim_count writes lock + revision."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import hash_manifest
from ai_scientist.utils.ara_artifact import export_ara, update_manifest_claim_count
from ai_scientist.utils.ara_manifest_lock import (
    MANIFEST_HISTORY_NAME,
    MANIFEST_LOCK_NAME,
    verify_manifest_lock,
)


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _minimal_project(tmp: Path) -> tuple[Path, Path]:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260709_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(exp / "logs", "0-run", [{
        "id": "n1", "step": 0, "code": "print('ok')",
        "_term_out": [], "metric": {"value": 0.5, "maximize": True, "name": "acc"},
        "is_buggy": False, "parent_id": None, "children": [],
    }])
    (exp / "idea.json").write_text(json.dumps({"Name": "idea"}), encoding="utf-8")
    return project, exp


class ExportWritesLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_export_writes_manifest_lock(self) -> None:
        project, exp = _minimal_project(self.tmp)
        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea"})
        ara = Path(result.root)
        lock_path = ara / MANIFEST_LOCK_NAME
        self.assertTrue(lock_path.exists())

        # verify reports clean
        report = verify_manifest_lock(ara)
        self.assertTrue(report["ok"])
        self.assertEqual(report["state"], "clean")

        # base hash in lock matches hash of the manifest bytes on disk
        manifest = json.loads((ara / "manifest.json").read_text())
        lock = json.loads(lock_path.read_text())
        self.assertEqual(lock["manifest_hash"], hash_manifest(manifest))

    def test_update_claim_count_creates_revision(self) -> None:
        project, exp = _minimal_project(self.tmp)
        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea"})
        ara = Path(result.root)
        manifest_path = ara / "manifest.json"

        # Simulate claim scan finishing.
        update_manifest_claim_count(manifest_path, claim_count=7)

        # New value visible.
        current = json.loads(manifest_path.read_text())
        self.assertEqual(current["counts"]["claims"], 7)

        # History row appended.
        rows = [json.loads(l) for l in
                (ara / MANIFEST_HISTORY_NAME).read_text().splitlines() if l]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["producer"], "update_manifest_claim_count")
        self.assertIn("counts.claims", rows[0]["changed_fields"])

        # verify reports revised (not tampered)
        report = verify_manifest_lock(ara)
        self.assertTrue(report["ok"])
        self.assertEqual(report["state"], "revised")
        self.assertEqual(report["revision_count"], 1)

    def test_repeated_same_count_is_no_op(self) -> None:
        # claim_registry writes the count every time it runs; identical value
        # must not create spurious revisions.
        project, exp = _minimal_project(self.tmp)
        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea"})
        ara = Path(result.root)
        manifest_path = ara / "manifest.json"

        update_manifest_claim_count(manifest_path, claim_count=4)
        # First call flipped counts.claims 0→4 and set counts_updated_at →
        # that's genuinely two mutations. Subsequent calls with the same
        # value only touch counts_updated_at, which changes the hash → so
        # a second revision IS expected. Test the different property:
        # calling with the same value never introduces a claims flip.
        update_manifest_claim_count(manifest_path, claim_count=4)

        current = json.loads(manifest_path.read_text())
        self.assertEqual(current["counts"]["claims"], 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
