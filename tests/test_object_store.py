"""Unit tests for the content-addressable object store."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import ObjectRef, ObjectStore
from ai_scientist.protocol.constants import CONTENT_HASH_ALGO
from ai_scientist.protocol.objects import _GZIP_THRESHOLD


class ObjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = ObjectStore(self.root)

    # ------------------------------------------------------------------
    # Layout / construction
    # ------------------------------------------------------------------
    def test_construction_creates_layout(self) -> None:
        self.assertTrue((self.root / "objects" / CONTENT_HASH_ALGO).is_dir())

    def test_ref_shape_matches_hash_prefix(self) -> None:
        ref = self.store.put_bytes(b"hello")
        self.assertTrue(ref.hash.startswith(f"{CONTENT_HASH_ALGO}:"))
        _, digest = ref.hash.split(":", 1)
        self.assertEqual(len(digest), 64)  # sha256 hex
        self.assertEqual(ref.size, 5)
        self.assertFalse(ref.gzip)

    def test_two_level_sharding(self) -> None:
        ref = self.store.put_bytes(b"payload-abc")
        _, digest = ref.hash.split(":", 1)
        expected = self.root / "objects" / CONTENT_HASH_ALGO / digest[:2] / digest[2:]
        self.assertTrue(expected.exists())

    # ------------------------------------------------------------------
    # Idempotency / dedup
    # ------------------------------------------------------------------
    def test_put_bytes_is_idempotent(self) -> None:
        ref1 = self.store.put_bytes(b"same bytes")
        mtime1 = self.store._resolve(ref1.hash).stat().st_mtime_ns
        ref2 = self.store.put_bytes(b"same bytes")
        mtime2 = self.store._resolve(ref2.hash).stat().st_mtime_ns
        self.assertEqual(ref1, ref2)
        # Second put must not rewrite the file
        self.assertEqual(mtime1, mtime2)

    def test_different_bytes_get_different_hashes(self) -> None:
        r1 = self.store.put_bytes(b"one")
        r2 = self.store.put_bytes(b"two")
        self.assertNotEqual(r1.hash, r2.hash)

    # ------------------------------------------------------------------
    # gzip threshold
    # ------------------------------------------------------------------
    def test_small_payload_not_gzipped(self) -> None:
        small = b"x" * (_GZIP_THRESHOLD - 1)
        ref = self.store.put_bytes(small)
        self.assertFalse(ref.gzip)
        raw = self.store._resolve(ref.hash).read_bytes()
        # not gzip magic
        self.assertNotEqual(raw[:2], b"\x1f\x8b")

    def test_large_payload_is_gzipped(self) -> None:
        big = b"y" * (_GZIP_THRESHOLD + 100)
        ref = self.store.put_bytes(big)
        self.assertTrue(ref.gzip)
        self.assertEqual(ref.size, len(big))  # size is original, pre-gzip
        raw = self.store._resolve(ref.hash).read_bytes()
        self.assertEqual(raw[:2], b"\x1f\x8b")
        self.assertEqual(gzip.decompress(raw), big)

    def test_round_trip_ungzips_transparently(self) -> None:
        big = b"z" * (_GZIP_THRESHOLD + 500)
        ref = self.store.put_bytes(big)
        self.assertEqual(self.store.get_bytes(ref.hash), big)

    def test_round_trip_small(self) -> None:
        ref = self.store.put_bytes(b"tiny")
        self.assertEqual(self.store.get_bytes(ref.hash), b"tiny")

    # ------------------------------------------------------------------
    # Text / JSON convenience
    # ------------------------------------------------------------------
    def test_put_json_is_canonical(self) -> None:
        # Insertion order differs but the two dicts serialise to the same bytes.
        r1 = self.store.put_json({"b": 2, "a": 1})
        r2 = self.store.put_json({"a": 1, "b": 2})
        self.assertEqual(r1.hash, r2.hash)

    def test_get_json_round_trip(self) -> None:
        payload = {"a": [1, 2, 3], "b": "text", "c": None}
        ref = self.store.put_json(payload)
        self.assertEqual(self.store.get_json(ref.hash), payload)

    def test_get_text_round_trip(self) -> None:
        ref = self.store.put_text("hello 世界")
        self.assertEqual(self.store.get_text(ref.hash), "hello 世界")

    # ------------------------------------------------------------------
    # Existence / ref validation
    # ------------------------------------------------------------------
    def test_exists_reports_true_only_after_put(self) -> None:
        # Hash of unwritten content still resolves to a legal path but
        # exists() must be False.
        never_written = f"{CONTENT_HASH_ALGO}:" + "0" * 64
        self.assertFalse(self.store.exists(never_written))
        ref = self.store.put_bytes(b"present")
        self.assertTrue(self.store.exists(ref.hash))

    def test_bad_ref_prefix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.get_bytes("md5:deadbeef")
        with self.assertRaises(ValueError):
            self.store.get_bytes("no-colon-here")

    # ------------------------------------------------------------------
    # ObjectRef.to_json
    # ------------------------------------------------------------------
    def test_object_ref_to_json_shape(self) -> None:
        ref = ObjectRef(hash="sha256:abc", size=3, gzip=False)
        self.assertEqual(ref.to_json(), {"hash": "sha256:abc", "size": 3, "gzip": False})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
