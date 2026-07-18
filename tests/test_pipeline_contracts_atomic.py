from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.utils import atomic_io, pipeline_contracts


class PipelineContractsAtomicTests(unittest.TestCase):
    def test_json_serialization_failure_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.json"
            path.write_text('{"status":"previous"}', encoding="utf-8")

            with self.assertRaises(TypeError):
                pipeline_contracts.save_json_artifact(path, {"invalid": object()})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "previous"},
            )

    def test_jsonl_row_failure_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.jsonl"
            path.write_text('{"status":"previous"}\n', encoding="utf-8")

            with self.assertRaises(TypeError):
                pipeline_contracts.save_jsonl_artifact(
                    path,
                    [{"valid": True}, {"invalid": object()}],
                )

            self.assertEqual(
                path.read_text(encoding="utf-8"), '{"status":"previous"}\n'
            )

    def test_text_replace_failure_preserves_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.md"
            path.write_text("previous", encoding="utf-8")

            with (
                mock.patch.object(Path, "replace", side_effect=OSError("disk busy")),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                pipeline_contracts.save_text_artifact(path, "new")

            self.assertEqual(path.read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(path.parent.glob(".artifact.md.*.tmp")), [])

    def test_jsonl_append_rolls_back_partial_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.jsonl"
            path.write_text('{"status":"previous"}\n', encoding="utf-8")
            real_write = os.write
            calls = 0

            def partial_then_fail(descriptor: int, payload) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_write(descriptor, payload[:4])
                raise OSError("disk full")

            with (
                mock.patch.object(
                    atomic_io.os,
                    "write",
                    side_effect=partial_then_fail,
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                pipeline_contracts.append_jsonl_artifact(path, {"status": "new"})

            self.assertEqual(
                path.read_text(encoding="utf-8"), '{"status":"previous"}\n'
            )


if __name__ == "__main__":
    unittest.main()
