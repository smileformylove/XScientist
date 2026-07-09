"""End-to-end: consumed seed manifest is snapshotted into the child ARA."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import ObjectStore, content_hash
from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_seed import SEED_MANIFEST_NAME


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
        "_term_out": ["ok\n"],
        "metric": {"value": 0.5, "maximize": True, "name": "acc"},
        "is_buggy": False, "parent_id": None, "children": [],
    }])
    (exp / "idea.json").write_text(json.dumps({"Name": "idea", "Title": "T"}), encoding="utf-8")
    return project, exp


def _write_seed(project: Path, *, code: str, plan: str = "seed plan",
                consumed: bool = True) -> Path:
    seed_dir = project / ".ara_seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "ara.v1",
        "protocol_kind": "seed",
        "code": code,
        "plan": plan,
        "provenance": {
            "parent_ara_root": "/tmp/parent_ara_fake",
            "parent_node_id": "n_parent",
            "parent_content_hash": "sha256:" + "d" * 64,
        },
    }
    seed_path = seed_dir / SEED_MANIFEST_NAME
    seed_path.write_text(json.dumps(manifest), encoding="utf-8")
    if consumed:
        (seed_dir / f"{SEED_MANIFEST_NAME}.consumed").write_text(
            json.dumps({"consumed_at": "2026-07-09T12:00:00Z"}), encoding="utf-8"
        )
    return seed_path


class SeedInARATests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_consumed_seed_is_snapshotted(self) -> None:
        project, exp = _minimal_project(self.tmp)
        seed_src = _write_seed(project, code="print('seed')")

        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea"})
        ara = Path(result.root)

        # Snapshot appears at the canonical path
        snapshot = ara / "seed" / SEED_MANIFEST_NAME
        self.assertTrue(snapshot.exists())
        self.assertEqual(snapshot.read_bytes(), seed_src.read_bytes())

        # .consumed sidecar also copied (provenance receipt)
        self.assertTrue((ara / "seed" / f"{SEED_MANIFEST_NAME}.consumed").exists())

        # references.seed is set
        manifest = json.loads((ara / "manifest.json").read_text())
        seed_ref = manifest["references"].get("seed")
        self.assertIsNotNone(seed_ref)
        self.assertEqual(seed_ref["path"], f"seed/{SEED_MANIFEST_NAME}")
        self.assertTrue(seed_ref["content_hash"].startswith("sha256:"))

        # CAS blob is retrievable
        store = ObjectStore(ara)
        self.assertTrue(store.exists(seed_ref["content_hash"]))
        self.assertEqual(store.get_bytes(seed_ref["content_hash"]),
                         seed_src.read_bytes())

    def test_seed_hash_lands_in_provenance(self) -> None:
        project, exp = _minimal_project(self.tmp)
        seed_src = _write_seed(project, code="print('bound')")

        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea"})
        ara = Path(result.root)
        manifest = json.loads((ara / "manifest.json").read_text())

        expected_hash = content_hash({"__probe__": None})  # sanity: hash format
        # We can't easily compute the actual seed content_hash from outside,
        # but we CAN assert it matches what the references say.
        seed_ref = manifest["references"]["seed"]
        prov = manifest["provenance"]
        self.assertEqual(prov["seed_hash"], seed_ref["content_hash"])

    def test_unconsumed_seed_is_not_snapshotted(self) -> None:
        # A staged-but-never-used seed (e.g. pipeline aborted before _draft)
        # must NOT appear in the child ARA — it wasn't actually part of the run.
        project, exp = _minimal_project(self.tmp)
        _write_seed(project, code="never used", consumed=False)

        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea"})
        ara = Path(result.root)
        self.assertFalse((ara / "seed").exists())
        manifest = json.loads((ara / "manifest.json").read_text())
        self.assertNotIn("seed", manifest["references"])
        # No seed_hash in provenance either
        prov = manifest.get("provenance") or {}
        self.assertNotIn("seed_hash", prov)

    def test_missing_seed_dir_leaves_manifest_clean(self) -> None:
        # Legacy runs have no .ara_seed at all → export must be a no-op
        # w.r.t. the seed slot.
        project, exp = _minimal_project(self.tmp)

        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea"})
        ara = Path(result.root)
        self.assertFalse((ara / "seed").exists())
        manifest = json.loads((ara / "manifest.json").read_text())
        self.assertNotIn("seed", manifest["references"])

    def test_seed_and_parent_provenance_coexist(self) -> None:
        # When caller supplies provenance AND a seed is consumed, both should
        # end up on the child manifest.
        project, exp = _minimal_project(self.tmp)
        _write_seed(project, code="parented + seeded")

        parent_hash = "sha256:" + "a" * 64
        caller_prov = {
            "parent_ara_root": "/tmp/some_parent",
            "parent_node_id": "n_parent",
            "parent_content_hash": parent_hash,
        }

        result = export_ara(
            project_dir=project, exp_dir=exp,
            idea={"Name": "idea"},
            provenance=caller_prov,
        )
        ara = Path(result.root)
        manifest = json.loads((ara / "manifest.json").read_text())
        prov = manifest["provenance"]
        self.assertEqual(prov["parent_content_hash"], parent_hash)
        self.assertTrue(prov["seed_hash"].startswith("sha256:"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
