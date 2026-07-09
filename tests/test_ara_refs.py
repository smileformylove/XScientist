"""Tests for the ARA refs helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_refs import (
    REFS_SUBDIR,
    Ref,
    RefError,
    delete_ref,
    get_ref,
    list_refs,
    set_ref,
)


class RefsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ara = Path(self._tmp.name)

    def test_set_and_get_flat_name(self) -> None:
        set_ref(self.ara, "HEAD", "sha256:" + "a" * 64)
        self.assertEqual(get_ref(self.ara, "HEAD"), "sha256:" + "a" * 64)
        self.assertTrue((self.ara / REFS_SUBDIR / "HEAD").exists())

    def test_set_and_get_nested_name(self) -> None:
        set_ref(self.ara, "candidates/best", "sha256:" + "b" * 64)
        self.assertEqual(get_ref(self.ara, "candidates/best"),
                         "sha256:" + "b" * 64)
        self.assertTrue((self.ara / REFS_SUBDIR / "candidates" / "best").exists())

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(get_ref(self.ara, "does/not/exist"))

    def test_update_overwrites(self) -> None:
        set_ref(self.ara, "x", "sha256:" + "c" * 64)
        set_ref(self.ara, "x", "sha256:" + "d" * 64)
        self.assertEqual(get_ref(self.ara, "x"), "sha256:" + "d" * 64)

    def test_delete_returns_true_only_when_existed(self) -> None:
        set_ref(self.ara, "x", "sha256:" + "e" * 64)
        self.assertTrue(delete_ref(self.ara, "x"))
        self.assertFalse(delete_ref(self.ara, "x"))

    def test_delete_cleans_empty_parent_dirs(self) -> None:
        set_ref(self.ara, "a/b/c", "sha256:" + "f" * 64)
        parent = self.ara / REFS_SUBDIR / "a" / "b"
        self.assertTrue(parent.exists())
        delete_ref(self.ara, "a/b/c")
        # After delete the empty parents should also be gone
        self.assertFalse(parent.exists())
        self.assertFalse((self.ara / REFS_SUBDIR / "a").exists())

    def test_list_refs_sorted_and_complete(self) -> None:
        set_ref(self.ara, "z/last", "sha256:" + "1" * 64)
        set_ref(self.ara, "a/first", "sha256:" + "2" * 64)
        set_ref(self.ara, "middle", "sha256:" + "3" * 64)
        refs = list_refs(self.ara)
        self.assertEqual([r.name for r in refs], ["a/first", "middle", "z/last"])

    def test_list_refs_empty_when_no_refs_dir(self) -> None:
        self.assertEqual(list_refs(self.ara), [])

    # ------------------------------------------------------------------
    # Security: name validation is the ONLY boundary. Test it hard.
    # ------------------------------------------------------------------
    def test_absolute_name_rejected(self) -> None:
        with self.assertRaises(RefError):
            set_ref(self.ara, "/etc/passwd", "sha256:" + "0" * 64)

    def test_dotdot_rejected(self) -> None:
        with self.assertRaises(RefError):
            set_ref(self.ara, "../escape", "sha256:" + "0" * 64)
        with self.assertRaises(RefError):
            set_ref(self.ara, "a/../b", "sha256:" + "0" * 64)

    def test_bad_characters_rejected(self) -> None:
        with self.assertRaises(RefError):
            set_ref(self.ara, "space name", "sha256:" + "0" * 64)
        with self.assertRaises(RefError):
            set_ref(self.ara, "with@symbol", "sha256:" + "0" * 64)

    def test_empty_rejected(self) -> None:
        with self.assertRaises(RefError):
            set_ref(self.ara, "", "sha256:" + "0" * 64)

    def test_bad_target_shape_rejected(self) -> None:
        with self.assertRaises(RefError):
            set_ref(self.ara, "ok", "not-a-hash")
        with self.assertRaises(RefError):
            set_ref(self.ara, "ok", "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
