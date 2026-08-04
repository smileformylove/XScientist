"""Lifecycle tests for ARA storage inventory, pins, and recoverable GC."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.protocol import ObjectStore
from ai_scientist.apps import ara as ara_app
from ai_scientist.utils.ara_refs import set_ref
from ai_scientist.utils.ara_storage import (
    ARAStorageError,
    apply_gc_plan,
    collect_hash_references,
    create_gc_plan,
    hydrate_objects,
    object_inventory,
    purge_quarantine,
    restore_quarantine,
    storage_report,
)


class ARAStorageLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "ara"
        self.root.mkdir()
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "ara.v1",
                    "created_at": "2026-08-04T00:00:00Z",
                    "source_exp_dir": "/tmp/exp",
                    "idea": {"name": "storage"},
                    "counts": {"nodes": 0},
                    "references": {},
                }
            ),
            encoding="utf-8",
        )
        (self.root / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "schema_version": "ara.v1",
                    "nodes": [],
                    "edges": [],
                    "counts": {"nodes": 0},
                }
            ),
            encoding="utf-8",
        )
        self.store = ObjectStore(self.root)

    def _reference(self, ref) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["references"]["payload"] = ref.to_json()
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _run(self, *argv: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = ara_app.main(list(argv))
        return rc, stdout.getvalue(), stderr.getvalue()

    def test_report_finds_duplicate_logical_bytes_and_unreachable_objects(self) -> None:
        referenced = self.store.put_bytes(b"same payload")
        orphan = self.store.put_bytes(b"orphan")
        self._reference(referenced)
        (self.root / "pipeline").mkdir()
        (self.root / "pipeline" / "mirror.bin").write_bytes(b"same payload")

        report = storage_report(self.root)
        self.assertGreater(report["duplicate_logical_bytes"], 0)
        self.assertEqual(report["objects"]["reachable"], 1)
        self.assertEqual(report["objects"]["unreachable"], 1)
        self.assertEqual(report["objects"]["unreachable_hashes"], [orphan.hash])

    def test_pin_is_a_gc_root(self) -> None:
        orphan = self.store.put_bytes(b"pin me")
        set_ref(self.root, "pins/release", orphan.hash)
        references = collect_hash_references(self.root)
        self.assertIn(orphan.hash, references)
        plan = create_gc_plan(self.root, write=False)
        self.assertEqual(plan["candidate_count"], 0)

    def test_apply_moves_to_quarantine_and_restore_recovers(self) -> None:
        orphan = self.store.put_bytes(b"recoverable")
        original = self.store._resolve(orphan.hash)
        plan = create_gc_plan(self.root, write=True)
        receipt = apply_gc_plan(plan["plan_path"])
        self.assertFalse(original.exists())
        self.assertEqual(receipt["moved_count"], 1)
        restored = restore_quarantine(receipt["receipt_path"])
        self.assertEqual(restored["restored_count"], 1)
        self.assertTrue(original.exists())
        self.assertEqual(self.store.get_bytes(orphan.hash), b"recoverable")

    def test_apply_refuses_when_roots_changed_after_plan(self) -> None:
        orphan = self.store.put_bytes(b"became-live")
        plan = create_gc_plan(self.root, write=True)
        set_ref(self.root, "pins/late", orphan.hash)
        with self.assertRaisesRegex(ARAStorageError, "references changed"):
            apply_gc_plan(plan["plan_path"])
        self.assertTrue(self.store.exists(orphan.hash))

    def test_explicit_purge_keeps_an_audit_receipt(self) -> None:
        self.store.put_bytes(b"purge-after-quarantine")
        plan = create_gc_plan(self.root, write=True)
        receipt = apply_gc_plan(plan["plan_path"])
        result = purge_quarantine(receipt["receipt_path"], grace_seconds=0)
        self.assertFalse(result["recoverable"])
        self.assertTrue(Path(result["receipt_path"]).exists())

    def test_shared_store_materialises_local_hardlink_when_supported(self) -> None:
        shared = Path(self._tmp.name) / ".ara-store"
        linked = ObjectStore(self.root, shared_root=shared)
        ref = linked.put_bytes(b"shared")
        local_path = linked._resolve(ref.hash)
        _, digest = ref.hash.split(":", 1)
        shared_path = shared / "objects" / "sha256" / digest[:2] / digest[2:]
        self.assertTrue(local_path.exists())
        self.assertTrue(shared_path.exists())
        self.assertEqual(linked.get_bytes(ref.hash), b"shared")

    def test_hydrate_restores_missing_local_view_from_project_store(self) -> None:
        project = Path(self._tmp.name) / "project"
        shared = project / ".ara-store"
        linked = ObjectStore(self.root, shared_root=shared)
        ref = linked.put_bytes(b"rehydrate")
        self._reference(ref)
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["project_dir"] = str(project)
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        linked._resolve(ref.hash).unlink()
        self.assertFalse(linked.exists(ref.hash))
        result = hydrate_objects(self.root, hashes=[ref.hash])
        self.assertTrue(result["complete"])
        self.assertEqual(result["restored"], [ref.hash])
        self.assertEqual(linked.get_bytes(ref.hash), b"rehydrate")

    def test_storage_pin_and_gc_cli_round_trip(self) -> None:
        orphan = self.store.put_bytes(b"cli-orphan")
        rc, out, err = self._run("storage-report", "--ara", str(self.root), "--json")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(json.loads(out)["objects"]["unreachable"], 1)

        rc, _, err = self._run(
            "pin",
            "--ara",
            str(self.root),
            "--name",
            "release/test",
            "--set",
            orphan.hash,
        )
        self.assertEqual(rc, 0, msg=err)
        rc, out, err = self._run("pin", "--ara", str(self.root), "--list", "--json")
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(json.loads(out)[0]["target"], orphan.hash)
        rc, _, _ = self._run(
            "pin",
            "--ara",
            str(self.root),
            "--name",
            "release/test",
            "--delete",
        )
        self.assertEqual(rc, 0)

        rc, out, err = self._run("gc", "--ara", str(self.root))
        self.assertEqual(rc, 0, msg=err)
        plan = json.loads(out)
        self.assertEqual(plan["candidate_count"], 1)
        rc, out, err = self._run("gc", "--apply", plan["plan_path"])
        self.assertEqual(rc, 0, msg=err)
        receipt = json.loads(out)
        self.assertEqual(receipt["moved_count"], 1)
        rc, out, err = self._run("gc", "--restore", receipt["receipt_path"])
        self.assertEqual(rc, 0, msg=err)
        self.assertEqual(json.loads(out)["restored_count"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
