from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.bounded_process import ProcessCancelled, run_process_bounded
from xscientist.client import XScientist


class BoundedProcessControlTests(unittest.TestCase):
    def test_cancellation_terminates_and_preserves_bounded_output_flags(self) -> None:
        polls = 0

        def cancel() -> bool:
            nonlocal polls
            polls += 1
            return polls >= 3

        with self.assertRaises(ProcessCancelled) as captured:
            run_process_bounded(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                cancel_check=cancel,
                max_output_chars=100,
                poll_interval=0.01,
            )
        self.assertIn("started", captured.exception.stdout)
        self.assertFalse(captured.exception.stdout_truncated)

    def test_client_maps_cancellation_to_a_bounded_130_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            client = XScientist(work_dir=Path(raw))
            result = client.run_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cancel_check=lambda: True,
                max_output_chars=8,
            )
        self.assertEqual(result.returncode, 130)
        self.assertLessEqual(len(result.stderr), 8)
        self.assertTrue(result.stderr_truncated)


if __name__ == "__main__":
    unittest.main()
