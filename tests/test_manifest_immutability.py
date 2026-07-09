"""Tests for the manifest immutability layer (lock + revisions + verify)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import hash_manifest
from ai_scientist.utils.ara_manifest_lock import (
    HISTORY_SUBDIR,
    MANIFEST_HISTORY_NAME,
    MANIFEST_LOCK_NAME,
    append_manifest_revision,
    verify_manifest_lock,
    write_manifest_lock,
)


def _fresh_ara(tmp: Path, *, manifest: dict | None = None) -> tuple[Path, dict]:
    ara = tmp / "ara"
    ara.mkdir()
    payload = manifest or {
        "schema_version": "ara.v1",
        "protocol_kind": "manifest",
        "created_at": "2026-07-09T12:00:00Z",
        "source_exp_dir": "/tmp/exp",
        "idea": {"name": "idea"},
        "counts": {"nodes": 1},
    }
    (ara / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    write_manifest_lock(ara, payload)
    return ara, payload


class HashManifestTests(unittest.TestCase):
    def test_hash_excludes_signatures(self) -> None:
        # Signing must not invalidate the very hash it signs.
        base = {"schema_version": "ara.v1", "counts": {"nodes": 1}}
        signed = {**base, "signatures": [{"algo": "minisign", "key_id": "x",
                                           "signature": "AAA"}]}
        self.assertEqual(hash_manifest(base), hash_manifest(signed))

    def test_hash_sensitive_to_counts(self) -> None:
        m = {"schema_version": "ara.v1", "counts": {"nodes": 1}}
        m2 = {"schema_version": "ara.v1", "counts": {"nodes": 2}}
        self.assertNotEqual(hash_manifest(m), hash_manifest(m2))


class ManifestLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_lock_contains_expected_hash(self) -> None:
        ara, manifest = _fresh_ara(self.tmp)
        lock = json.loads((ara / MANIFEST_LOCK_NAME).read_text())
        self.assertEqual(lock["manifest_hash"], hash_manifest(manifest))
        self.assertEqual(lock["protocol_kind"], "manifest_lock")
        self.assertTrue(lock["hasher"].startswith("hash_manifest."))

    def test_verify_reports_clean_after_export(self) -> None:
        ara, _ = _fresh_ara(self.tmp)
        report = verify_manifest_lock(ara)
        self.assertTrue(report["ok"])
        self.assertEqual(report["state"], "clean")
        self.assertEqual(report["revision_count"], 0)

    def test_verify_reports_unlocked_without_lock(self) -> None:
        ara = self.tmp / "ara"
        ara.mkdir()
        (ara / "manifest.json").write_text("{}", encoding="utf-8")
        report = verify_manifest_lock(ara)
        self.assertFalse(report["ok"])
        self.assertEqual(report["state"], "unlocked")


class AppendManifestRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_revision_writes_history_row_and_archives_prior(self) -> None:
        ara, manifest = _fresh_ara(self.tmp)
        base_hash = hash_manifest(manifest)

        def _mutate(m: dict) -> list[str]:
            m.setdefault("counts", {})["claims"] = 5
            return ["counts.claims"]

        new = append_manifest_revision(
            ara / "manifest.json", _mutate,
            reason="claim scan finished", producer="update_manifest_claim_count",
        )
        self.assertIsNotNone(new)
        self.assertEqual(new["counts"]["claims"], 5)

        # History row
        rows = [json.loads(l) for l in
                (ara / MANIFEST_HISTORY_NAME).read_text().splitlines() if l]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["revision"], 1)
        self.assertEqual(row["base_hash"], base_hash)
        self.assertEqual(row["new_hash"], hash_manifest(new))
        self.assertEqual(row["changed_fields"], ["counts.claims"])
        self.assertEqual(row["reason"], "claim scan finished")

        # Prior revision archived
        _, digest = base_hash.split(":", 1)
        archive = ara / HISTORY_SUBDIR / f"{digest}.json"
        self.assertTrue(archive.exists())
        archived = json.loads(archive.read_text())
        self.assertNotIn("claims", archived.get("counts", {}))

        # New manifest.json matches the new_hash
        current = json.loads((ara / "manifest.json").read_text())
        self.assertEqual(hash_manifest(current), row["new_hash"])

    def test_verify_reports_revised_after_append(self) -> None:
        ara, _ = _fresh_ara(self.tmp)
        append_manifest_revision(
            ara / "manifest.json",
            lambda m: (m.setdefault("counts", {}).__setitem__("claims", 3), ["counts.claims"])[1],
        )
        report = verify_manifest_lock(ara)
        self.assertTrue(report["ok"])
        self.assertEqual(report["state"], "revised")
        self.assertEqual(report["revision_count"], 1)

    def test_no_op_mutation_writes_no_row(self) -> None:
        ara, _ = _fresh_ara(self.tmp)
        append_manifest_revision(
            ara / "manifest.json",
            lambda m: [],  # no changes
        )
        self.assertFalse((ara / MANIFEST_HISTORY_NAME).exists())
        self.assertFalse((ara / HISTORY_SUBDIR).exists())

    def test_multiple_revisions_chain(self) -> None:
        ara, manifest = _fresh_ara(self.tmp)
        for n in (1, 2, 3):
            append_manifest_revision(
                ara / "manifest.json",
                lambda m, k=n: (m.setdefault("counts", {}).__setitem__("claims", k),
                                ["counts.claims"])[1],
            )
        rows = [json.loads(l) for l in
                (ara / MANIFEST_HISTORY_NAME).read_text().splitlines() if l]
        self.assertEqual([r["revision"] for r in rows], [1, 2, 3])
        # Each row's base_hash equals the previous row's new_hash
        for prev, cur in zip(rows, rows[1:]):
            self.assertEqual(cur["base_hash"], prev["new_hash"])
        # verify still reports clean-through-latest
        report = verify_manifest_lock(ara)
        self.assertTrue(report["ok"])
        self.assertEqual(report["state"], "revised")
        self.assertEqual(report["revision_count"], 3)

    def test_missing_manifest_returns_none(self) -> None:
        # Callers used to swallow this silently — preserve that contract.
        result = append_manifest_revision(
            self.tmp / "nope.json", lambda m: None,
        )
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Tamper detection — the main "immutability" promise
    # ------------------------------------------------------------------
    def test_verify_reports_tampered_when_manifest_edited_outside_api(self) -> None:
        ara, _ = _fresh_ara(self.tmp)
        # Bypass the append-only API entirely — write different bytes.
        (ara / "manifest.json").write_text(
            json.dumps({"schema_version": "ara.v1", "sneaky": True}),
            encoding="utf-8",
        )
        report = verify_manifest_lock(ara)
        self.assertFalse(report["ok"])
        self.assertEqual(report["state"], "tampered")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
