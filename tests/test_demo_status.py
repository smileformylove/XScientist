from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from xscientist.cli import _contextual_action, main as cli_main
from xscientist.research_journey import workspace_action_contract


@unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
class DemoStatusTests(unittest.TestCase):
    def test_repository_neutral_template_declares_workspace_cwd(self) -> None:
        action = workspace_action_contract(
            "xscientist research discovery template --output discovery.json"
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["workspace_binding"]["mode"], "cwd")
        self.assertEqual(action["cwd_binding"]["mode"], "workspace_root")
        self.assertEqual(action["cwd_binding"]["template"], "{workspace}")
        self.assertNotIn(".", action["argv_template"])

    def test_empty_workspace_demo_action_is_bound_to_the_workspace_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "empty-workspace"
            workspace.mkdir()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--json"]),
                    0,
                )

            payload = json.loads(output.getvalue())
            step = payload["next_steps"][0]
            action = step["action"]
            self.assertEqual(step["code"], "start_research")
            self.assertEqual(
                action["argv_template"],
                ["xscientist", "demo", "./xscientist-demo"],
            )
            self.assertEqual(action["workspace_binding"]["mode"], "cwd")
            self.assertTrue(action["workspace_binding"]["required"])
            self.assertEqual(action["cwd_binding"]["mode"], "workspace_root")
            self.assertEqual(action["cwd_binding"]["template"], "{workspace}")

    def test_contextual_explore_action_stays_bound_to_inspected_workspace(self) -> None:
        """A copied next-step command must not mutate the caller's directory."""

        workspace = Path("/tmp/xscientist-user-study")
        self.assertEqual(
            _contextual_action("xscientist explore .", workspace),
            "xscientist explore /tmp/xscientist-user-study",
        )
        self.assertEqual(
            _contextual_action("xscientist explore . --lang zh", workspace),
            "xscientist explore /tmp/xscientist-user-study --lang zh",
        )

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
            self.assertTrue((workspace / payload["dag"]["html"]).is_file())
            self.assertEqual(payload["repository"], ".")
            self.assertFalse(payload["privacy"]["host_paths_disclosed"])
            self.assertTrue((workspace / "research.yaml").is_file())
            self.assertIn("failed_attempt", payload["objects"])
            self.assertIn("supporting_evidence", payload["objects"])
            self.assertIn("refuting_evidence", payload["objects"])
            self.assertIn("bounded_inference", payload["objects"])
            self.assertEqual(
                payload["guide"]["next_steps"][0]["code"],
                "resolve_contested_claim",
            )
            self.assertEqual(
                payload["guide"]["next_steps"][1]["code"],
                "strengthen_research_program",
            )
            self.assertEqual(payload["guide"]["program_review"]["gap_count"], 6)
            commands = {
                step["code"]: step["command"] for step in payload["guide"]["next_steps"]
            }
            self.assertIn(
                "program review --repo {workspace}",
                commands["strengthen_research_program"],
            )

            audit_output = io.StringIO()
            with contextlib.redirect_stdout(audit_output):
                self.assertEqual(
                    cli_main(
                        [
                            "research",
                            "audit",
                            "--repo",
                            str(workspace),
                            "--level",
                            "replay",
                            "--json",
                        ]
                    ),
                    0,
                )
            audit = json.loads(audit_output.getvalue())
            self.assertTrue(audit["complete"])
            self.assertTrue(all(row["replay_ready"] for row in audit["claims"]))

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
            self.assertTrue(payload["result"]["dag_current"])
            self.assertTrue(payload["review"]["clean"])
            self.assertEqual(payload["review"]["checks"]["trace"], "pass")
            self.assertEqual(payload["review"]["checks"]["replay"], "pass")
            self.assertEqual(payload["review"]["checks"]["verify"], "pending")
            self.assertEqual(payload["review"]["target_level"], "verify")
            self.assertGreaterEqual(payload["review"]["blocker_count"], 0)
            self.assertTrue(
                set(payload["review"]["blocker_codes"]).issubset(
                    set(payload["review"]["closure_levels"]["verify"]["blocker_codes"])
                )
            )
            self.assertEqual(
                set(payload["review"]["closure_levels"]),
                {"trace", "replay", "verify"},
            )
            self.assertTrue(payload["review"]["closure_levels"]["trace"]["complete"])
            self.assertFalse(payload["review"]["closure_levels"]["verify"]["complete"])
            self.assertFalse(payload["review"]["promotion_ready"])
            self.assertTrue(payload["review"]["commands"]["diff"])
            self.assertTrue(payload["next_steps"])
            self.assertEqual(
                payload["next_steps"][0]["code"], "resolve_contested_claim"
            )
            action = payload["next_steps"][0]["action"]
            self.assertEqual(action["schema_version"], "xscientist.workspace-action.v1")
            self.assertEqual(action["workspace_binding"]["mode"], "argument")
            self.assertIn("{workspace}", action["argv_template"])
            self.assertNotIn("--repo .", payload["next_steps"][0]["command"])
            self.assertEqual(
                payload["workspace_context"]["workspace_source"],
                "workspace argument supplied to this invocation",
            )
            self.assertNotIn(str(workspace), output.getvalue())
            self.assertEqual(payload["workspace"], "demo")
            self.assertIsNone(payload["background_run"])
            self.assertEqual(payload["operational_state"], "scientific_followup")
            self.assertFalse(payload["attention_required"])
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

    def test_status_prioritizes_the_last_readiness_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["demo", str(workspace)]), 0)
            readiness = {
                "schema": "xscientist.doctor.v1",
                "ok": False,
                "configuration_ready": True,
                "runtime_ready": False,
                "remediations": [
                    {
                        "code": "docker_cli_missing",
                        "command": "https://docs.docker.com/get-started/get-docker/",
                        "detail": "Install and start Docker before preparing the executor.",
                        "severity": "error",
                    }
                ],
            }
            (workspace / ".xscientist" / "readiness.json").write_text(
                json.dumps(readiness), encoding="utf-8"
            )
            followups = {
                "active": [
                    {
                        "object_id": "rso-open-gap",
                        "gap": "single_hypothesis_bias",
                        "action": "Record a rival hypothesis.",
                    },
                    {
                        "object_id": "rso-resolved-gap",
                        "gap": "already_resolved",
                        "action": "This stale action must stay hidden.",
                    },
                ]
            }
            (workspace / "04_logs").mkdir(exist_ok=True)
            (workspace / "04_logs" / "research_strategy_followups.json").write_text(
                json.dumps(followups), encoding="utf-8"
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli_main(["status", str(workspace), "--json"]), 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["operational_state"], "needs_attention")
            self.assertTrue(payload["attention_required"])
            self.assertEqual(payload["next_steps"][0]["code"], "docker_cli_missing")
            self.assertEqual(
                payload["next_steps"][1]["code"],
                "inspect_scientific_strategy_followup",
            )
            self.assertEqual(len(payload["strategy_followups"]["queued"]), 1)
            self.assertFalse(payload["readiness"]["runtime_ready"])

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
            self.assertIn("State: run complete; more evidence needed", rendered)
            self.assertIn("Scientific progress: 7/8", rendered)
            self.assertIn("History: main@", rendered)
            self.assertIn("Checks: trace=pass, replay=pass, verify=pending", rendered)
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
            self.assertFalse(payload["result"]["dag_current"])
            self.assertEqual(payload["warnings"][0]["code"], "generated_view_stale")

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
            self.assertIn("状态：运行完成，仍需补充证据", rendered)
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
            self.assertIn(
                "competitive_hypothesis_portfolio",
                demo["autopilot_fixture"]["profile_behavior"],
            )
            self.assertIn("portfolio", demo["autopilot_fixture"]["profile_objects"])
            self.assertGreater(demo["dag"]["nodes"], 16)
            self.assertLessEqual(demo["guide"]["program_review"]["gap_count"], 4)
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
            human_status = io.StringIO()
            with contextlib.redirect_stdout(human_status):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--lang", "en"]),
                    0,
                )
            self.assertIn(
                "State: run complete; more evidence needed",
                human_status.getvalue(),
            )
            self.assertNotIn("Research pipeline:", human_status.getvalue())
            verbose_status = io.StringIO()
            with contextlib.redirect_stdout(verbose_status):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--lang", "en", "--verbose"]),
                    0,
                )
            self.assertIn(
                "Research pipeline: run complete; scientific closure pending",
                verbose_status.getvalue(),
            )
            self.assertTrue(
                (workspace / "04_logs" / "autopilot_fixture_receipt.json").is_file()
            )
            self.assertIn(
                "04_logs/autopilot_fixture_receipt.json",
                demo["runtime_checkpoint"]["staged_paths"],
            )
            research_status = io.StringIO()
            with contextlib.redirect_stdout(research_status):
                self.assertEqual(
                    cli_main(
                        [
                            "research",
                            "status",
                            "--repo",
                            str(workspace),
                            "--json",
                        ]
                    ),
                    0,
                )
            vcs_status = json.loads(research_status.getvalue())
            self.assertEqual(vcs_status["eligible_changes"], [])
            self.assertEqual(vcs_status["staged_paths"], [])

    def test_publication_fixture_adds_independent_review_board_objects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "publication-demo"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    cli_main(
                        [
                            "demo",
                            str(workspace),
                            "--autopilot",
                            "--autopilot-profile",
                            "publication",
                            "--json",
                        ]
                    ),
                    0,
                )

            payload = json.loads(output.getvalue())
            fixture = payload["autopilot_fixture"]
            self.assertIn("multi_role_review", fixture["profile_behavior"])
            self.assertEqual(len(fixture["profile_objects"]["publication_reviews"]), 2)
            self.assertEqual(len(fixture["profile_objects"]["publication_gates"]), 2)
            self.assertEqual(len(fixture["profile_objects"]["decision_contexts"]), 2)
            self.assertGreater(payload["dag"]["nodes"], 16)


if __name__ == "__main__":
    unittest.main()
