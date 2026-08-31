from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import xscientist.provider_config as provider_config_module
import xscientist.research_git as research_git_module
from ai_scientist.utils.atomic_io import atomic_write_text


class AtomicIOTests(unittest.TestCase):
    def test_shared_text_write_preserves_requested_utf8_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "nested" / "artifact.jsonl"

            atomic_write_text(target, "α\nβ\n")

            self.assertEqual(target.read_bytes(), "α\nβ\n".encode("utf-8"))

    def test_research_text_write_preserves_requested_utf8_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "nested" / "trajectory.jsonl"

            research_git_module._atomic_write_text(target, "α\nβ\n")

            self.assertEqual(target.read_bytes(), "α\nβ\n".encode("utf-8"))

    def test_provider_replace_preserves_requested_utf8_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / ".xscientist" / "providers.json"
            before = provider_config_module._capture_provider_file_state(target)

            after = provider_config_module._atomic_provider_text_replace(
                target,
                "α\nβ\n",
                mode=0o600,
                expected_before=before,
            )

            expected = "α\nβ\n".encode("utf-8")
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(after.content, expected)


if __name__ == "__main__":
    unittest.main()
