from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.treesearch import parallel_agent
from ai_scientist.treesearch.errors import ExperimentCannotContinueError


class ParallelAgentArtifactsAtomicTests(unittest.TestCase):
    def test_publish_source_artifacts_writes_all_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "results"

            parallel_agent._publish_source_artifacts(
                output_dir,
                {
                    "plotting_code.py": "plot()\n",
                    "experiment_code.py": "run()\n",
                },
            )

            self.assertEqual(
                (output_dir / "plotting_code.py").read_text(encoding="utf-8"),
                "plot()\n",
            )
            self.assertEqual(
                (output_dir / "experiment_code.py").read_text(encoding="utf-8"),
                "run()\n",
            )

    def test_publish_failure_preserves_existing_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "results"
            output_dir.mkdir()
            source_path = output_dir / "experiment_code.py"
            source_path.write_text("previous", encoding="utf-8")

            with (
                mock.patch.object(
                    parallel_agent,
                    "atomic_write_text",
                    side_effect=OSError("disk busy"),
                ),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                parallel_agent._publish_source_artifacts(
                    output_dir, {"experiment_code.py": "new"}
                )

            self.assertEqual(source_path.read_text(encoding="utf-8"), "previous")

    def test_plot_publish_never_overwrites_canonical_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "working"
            destination_dir = root / "evidence"
            source_dir.mkdir()
            destination_dir.mkdir()
            canonical = destination_dir / "experiment_data.npy"
            canonical.write_bytes(b"trusted-evidence")
            (source_dir / "experiment_data.npy").write_bytes(b"agent-rewrite")
            (source_dir / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\nplot-bytes")

            published = parallel_agent._publish_plot_artifacts(
                source_dir,
                destination_dir,
            )

            self.assertEqual(canonical.read_bytes(), b"trusted-evidence")
            self.assertEqual(published, [destination_dir / "figure.png"])

    def test_plot_publish_rejects_symlink_without_moving_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_dir = root / "working"
            destination_dir = root / "evidence"
            source_dir.mkdir()
            destination_dir.mkdir()
            canary = root / "canary.png"
            canary.write_bytes(b"\x89PNG\r\n\x1a\ncanary")
            try:
                (source_dir / "leak.png").symlink_to(canary)
            except OSError:
                self.skipTest("symlinks are unavailable on this platform")

            with self.assertRaises(ExperimentCannotContinueError):
                parallel_agent._publish_plot_artifacts(
                    source_dir,
                    destination_dir,
                )

            self.assertTrue(canary.is_file())
            self.assertEqual(canary.read_bytes(), b"\x89PNG\r\n\x1a\ncanary")
            self.assertFalse((destination_dir / "leak.png").exists())


if __name__ == "__main__":
    unittest.main()
