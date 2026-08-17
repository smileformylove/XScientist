from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from xscientist.cli import main as cli_main


@unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
class DemoStatusTests(unittest.TestCase):
    def test_provider_free_demo_creates_a_contested_evidence_journey(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["demo", str(workspace), "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["schema"], "xscientist.demo.v1")
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["network_used"])
            self.assertFalse(payload["provider_used"])
            self.assertEqual(payload["cost_usd"], 0.0)
            self.assertTrue(payload["dag"]["integrity_ok"])
            self.assertEqual(payload["dag"]["closure"], "blocked")
            self.assertGreaterEqual(payload["dag"]["nodes"], 10)
            self.assertGreaterEqual(payload["dag"]["relations"], 10)
            self.assertTrue(Path(payload["dag"]["html"]).is_file())
            self.assertTrue((workspace / "research.yaml").is_file())
            self.assertIn("failed_attempt", payload["objects"])
            self.assertIn("supporting_evidence", payload["objects"])
            self.assertIn("refuting_evidence", payload["objects"])

    def test_demo_refuses_to_replace_an_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            workspace.mkdir()
            sentinel = workspace / "notes.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli_main(["demo", str(workspace), "--json"])

            self.assertEqual(exit_code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["error_code"], "demo_creation_failed")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_status_summarizes_progress_without_starting_a_model_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["demo", str(workspace)]), 0)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["status", str(workspace), "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["schema"], "xscientist.workspace-status.v1")
            self.assertTrue(payload["research"]["initialized"])
            self.assertEqual(payload["research"]["branch"], "main")
            self.assertFalse(payload["run"]["started"])
            self.assertTrue(payload["result"]["dag_html"])
            self.assertTrue(payload["next_steps"])


if __name__ == "__main__":
    unittest.main()
