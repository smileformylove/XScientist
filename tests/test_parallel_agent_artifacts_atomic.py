from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.treesearch import parallel_agent


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


if __name__ == "__main__":
    unittest.main()
