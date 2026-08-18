from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

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
            self.assertIn("bounded_inference", payload["objects"])
            self.assertEqual(
                payload["guide"]["next_steps"][0]["code"],
                "resolve_contested_claim",
            )

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
            self.assertEqual(
                payload["next_steps"][0]["code"], "resolve_contested_claim"
            )
            self.assertEqual(payload["workspace"], "demo")
            self.assertIsNone(payload["background_run"])
            self.assertTrue(payload["workspace_id"].startswith("ws-"))
            repository_config = yaml.safe_load(
                (workspace / "research.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["workspace_id"], repository_config["repository_id"]
            )

    def test_status_reports_corrupted_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["demo", str(workspace)]), 0)
            logs = workspace / "04_logs"
            logs.mkdir(exist_ok=True)
            (logs / "progress.json").write_text("{not-json", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = cli_main(["status", str(workspace), "--json"])

            self.assertEqual(exit_code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["errors"][0]["code"], "workspace_state_corrupted")
            self.assertEqual(payload["errors"][0]["file"], "04_logs/progress.json")

    def test_human_demo_status_journey_names_the_contested_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            demo_output = io.StringIO()
            status_output = io.StringIO()
            with contextlib.redirect_stdout(demo_output):
                self.assertEqual(cli_main(["demo", str(workspace), "--lang", "en"]), 0)
            with contextlib.redirect_stdout(status_output):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--lang", "en"]), 0
                )

            self.assertIn("Scientific closure: blocked", demo_output.getvalue())
            self.assertIn("the demo itself succeeded", demo_output.getvalue())
            self.assertIn("Next: xscientist status", demo_output.getvalue())
            rendered = status_output.getvalue()
            self.assertIn("Workspace: demo", rendered)
            self.assertIn("Scientific progress: 7/8", rendered)
            self.assertIn("Resolve or narrow the contested claim", rendered)
            self.assertIn(f"--repo {workspace}", rendered)

    def test_missing_workspace_is_an_error_instead_of_an_empty_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "missing"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["status", str(workspace), "--json"])

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["errors"][0]["code"], "workspace_not_found")
            self.assertEqual(payload["next_steps"], [])

    def test_contested_claim_guidance_advances_after_the_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["demo", str(workspace)]), 0)
                self.assertEqual(
                    cli_main(
                        [
                            "research",
                            "plan",
                            "@latest:hypothesis",
                            "Test the contested boundary",
                            "--test",
                            "A held-out result resolves the conflict",
                            "--repo",
                            str(workspace),
                        ]
                    ),
                    0,
                )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--json"]),
                    0,
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(
                payload["next_steps"][0]["code"], "run_resolution_experiment"
            )

    def test_chinese_demo_and_status_render_complete_primary_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            demo_output = io.StringIO()
            status_output = io.StringIO()
            with contextlib.redirect_stdout(demo_output):
                self.assertEqual(cli_main(["demo", str(workspace), "--lang", "zh"]), 0)
            with contextlib.redirect_stdout(status_output):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--lang", "zh"]), 0
                )

            self.assertIn("零 Provider 演示已就绪", demo_output.getvalue())
            self.assertIn("演示成功", demo_output.getvalue())
            rendered = status_output.getvalue()
            self.assertIn("工作区：demo", rendered)
            self.assertIn("科学进度", rendered)
            self.assertIn("下一步", rendered)
            self.assertIn("检验存在争议的边界", rendered)

    def test_autopilot_fixture_populates_runtime_budget_and_insight_contracts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(
                    [
                        "demo",
                        str(workspace),
                        "--autopilot",
                        "--autopilot-profile",
                        "discovery",
                        "--json",
                    ]
                )
            self.assertEqual(exit_code, 0)
            demo = json.loads(output.getvalue())
            self.assertEqual(demo["autopilot_fixture"]["profile"], "discovery")
            self.assertFalse(demo["autopilot_fixture"]["network_used"])
            self.assertFalse(demo["autopilot_fixture"]["generated_code_executed"])

            status_output = io.StringIO()
            with contextlib.redirect_stdout(status_output):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--json"]),
                    0,
                )
            status = json.loads(status_output.getvalue())
            self.assertTrue(status["run"]["started"])
            self.assertEqual(status["run"]["current_stage"], "complete")
            self.assertEqual(status["budget"]["used"]["cost_usd"], 0.0)
            self.assertEqual(
                status["result"]["epistemic_status"],
                "machine_synthesized_unverified",
            )
            self.assertTrue(
                (workspace / "04_logs" / "autopilot_fixture_receipt.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
