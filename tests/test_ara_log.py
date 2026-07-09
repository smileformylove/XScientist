"""Tests for the ARA log engine (revisions + ancestry)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara, update_manifest_claim_count
from ai_scientist.utils.ara_log import ara_log


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _project(tmp: Path, sub: str, nodes: list[dict],
             *, provenance: dict | None = None) -> tuple[Path, Path]:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260709_{sub}"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(exp / "logs", "0-run", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    return project, exp


def _basic_node(nid: str) -> dict:
    return {
        "id": nid, "step": 0, "code": "print('ok')",
        "_term_out": [],
        "metric": {"value": 0.5, "maximize": True, "name": "acc"},
        "is_buggy": False, "parent_id": None, "children": [],
    }


class LogEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # Revision section
    # ------------------------------------------------------------------
    def test_fresh_export_shows_no_revisions(self) -> None:
        p, e = _project(self.tmp, "a", [_basic_node("n1")])
        r = export_ara(project_dir=p, exp_dir=e, idea={"Name": "a"})
        log = ara_log(r.root)
        self.assertIsNotNone(log.lock)
        self.assertEqual(log.revisions, [])
        self.assertEqual(log.verify["state"], "clean")

    def test_revisions_appear_after_update(self) -> None:
        p, e = _project(self.tmp, "a", [_basic_node("n1")])
        r = export_ara(project_dir=p, exp_dir=e, idea={"Name": "a"})
        update_manifest_claim_count(Path(r.root) / "manifest.json", claim_count=3)
        update_manifest_claim_count(Path(r.root) / "manifest.json", claim_count=7)
        log = ara_log(r.root)
        self.assertEqual([rev.revision for rev in log.revisions], [1, 2])
        # Chain integrity: rev 2's base_hash equals rev 1's new_hash.
        self.assertEqual(log.revisions[1].base_hash, log.revisions[0].new_hash)
        self.assertEqual(log.verify["state"], "revised")

    # ------------------------------------------------------------------
    # Ancestry
    # ------------------------------------------------------------------
    def test_root_ara_has_no_ancestors(self) -> None:
        p, e = _project(self.tmp, "root", [_basic_node("n1")])
        r = export_ara(project_dir=p, exp_dir=e, idea={"Name": "root"})
        log = ara_log(r.root)
        self.assertEqual(log.ancestors, [])

    def test_provenance_reachable_and_verified(self) -> None:
        # Build parent ARA.
        pp, ep = _project(self.tmp, "parent", [_basic_node("np")])
        parent = export_ara(project_dir=pp, exp_dir=ep, idea={"Name": "parent"})
        # Find parent node's content_hash
        parent_graph = json.loads((Path(parent.root) / "exploration_graph.json").read_text())
        parent_node_hash = parent_graph["nodes"][0]["content_hash"]

        # Build child ARA that claims parent as its ancestor.
        pc, ec = _project(self.tmp, "child", [_basic_node("nc")])
        child = export_ara(
            project_dir=pc, exp_dir=ec, idea={"Name": "child"},
            provenance={
                "parent_ara_root": str(parent.root),
                "parent_node_id": "np",
                "parent_content_hash": parent_node_hash,
            },
        )
        log = ara_log(child.root)
        self.assertEqual(len(log.ancestors), 1)
        anc = log.ancestors[0]
        self.assertEqual(anc.depth, 1)
        self.assertEqual(anc.ara_root, str(parent.root))
        self.assertTrue(anc.reachable)
        self.assertTrue(anc.hash_verified)

    def test_provenance_unreachable_when_parent_path_missing(self) -> None:
        pc, ec = _project(self.tmp, "orphan", [_basic_node("nc")])
        child = export_ara(
            project_dir=pc, exp_dir=ec, idea={"Name": "orphan"},
            provenance={
                "parent_ara_root": "/nonexistent/parent/path/xyz",
                "parent_node_id": "np",
                "parent_content_hash": "sha256:" + "d" * 64,
            },
        )
        log = ara_log(child.root)
        self.assertEqual(len(log.ancestors), 1)
        anc = log.ancestors[0]
        self.assertFalse(anc.reachable)
        # content_hash still records the parent identity even without a path
        self.assertEqual(anc.content_hash, "sha256:" + "d" * 64)

    def test_ancestry_verify_reports_mismatch_when_parent_node_hash_differs(self) -> None:
        pp, ep = _project(self.tmp, "p", [_basic_node("np")])
        parent = export_ara(project_dir=pp, exp_dir=ep, idea={"Name": "p"})
        pc, ec = _project(self.tmp, "c", [_basic_node("nc")])
        child = export_ara(
            project_dir=pc, exp_dir=ec, idea={"Name": "c"},
            provenance={
                "parent_ara_root": str(parent.root),
                "parent_node_id": "np",
                # Deliberately wrong hash.
                "parent_content_hash": "sha256:" + "0" * 64,
            },
        )
        log = ara_log(child.root)
        anc = log.ancestors[0]
        self.assertTrue(anc.reachable)
        self.assertFalse(anc.hash_verified)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
