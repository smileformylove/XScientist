from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.utils import safe_files
from ai_scientist.utils.safe_files import (
    BoundedFileError,
    read_bounded_regular_file,
)


class BoundedRegularFileTests(unittest.TestCase):
    def test_rejects_symlink_and_size_limit_without_reading_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root / "outside.json"
            outside.write_bytes(b"secret")
            linked = root / "linked.json"
            try:
                linked.symlink_to(outside)
            except OSError as exc:  # pragma: no cover - platform capability
                self.skipTest(f"file symlinks unavailable: {exc}")

            with self.assertRaises(BoundedFileError) as symlink_error:
                read_bounded_regular_file(linked, maximum=1024, label="artifact")
            self.assertEqual(symlink_error.exception.reason, "symlink_rejected")

            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(1025)
            with self.assertRaises(BoundedFileError) as size_error:
                read_bounded_regular_file(oversized, maximum=1024, label="artifact")
            self.assertEqual(size_error.exception.reason, "too_large")

    def test_rejects_inode_swap_between_lstat_and_open(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "artifact.json"
            target.write_bytes(b"same bytes")
            replacement = root / "replacement.json"
            replacement.write_bytes(b"same bytes")
            real_open = os.open
            swapped = False

            def replace_then_open(path, flags):
                nonlocal swapped
                if not swapped:
                    swapped = True
                    replacement.replace(target)
                return real_open(path, flags)

            with mock.patch.object(
                safe_files.os,
                "open",
                side_effect=replace_then_open,
            ):
                with self.assertRaises(BoundedFileError) as caught:
                    read_bounded_regular_file(
                        target,
                        maximum=1024,
                        label="artifact",
                    )

            self.assertTrue(swapped)
            self.assertEqual(caught.exception.reason, "changed_during_read")


if __name__ == "__main__":
    unittest.main()
