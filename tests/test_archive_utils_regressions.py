from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from ai_scientist.treesearch.utils import extract_archives


class ArchiveUtilsRegressionTests(unittest.TestCase):
    def test_extract_archives_should_handle_nested_zip_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inner_zip = root / "inner.zip"
            with zipfile.ZipFile(inner_zip, "w") as zip_ref:
                zip_ref.writestr("nested.txt", "nested-content")

            outer_zip = root / "outer.zip"
            with zipfile.ZipFile(outer_zip, "w") as zip_ref:
                zip_ref.write(inner_zip, arcname="inner.zip")

            inner_zip.unlink()
            extract_archives(root)

            self.assertFalse((root / "outer.zip").exists())
            self.assertFalse((root / "outer" / "inner.zip").exists())
            self.assertTrue((root / "outer" / "inner" / "nested.txt").exists())

    def test_extract_archives_should_drop_redundant_zip_when_file_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing_file = root / "data"
            existing_file.write_text("same-content", encoding="utf-8")

            redundant_zip = root / "data.zip"
            with zipfile.ZipFile(redundant_zip, "w") as zip_ref:
                zip_ref.writestr("data", "same-content")

            extract_archives(root)

            self.assertTrue(existing_file.exists())
            self.assertFalse(redundant_zip.exists())
            self.assertEqual(existing_file.read_text(encoding="utf-8"), "same-content")


if __name__ == "__main__":
    unittest.main()
