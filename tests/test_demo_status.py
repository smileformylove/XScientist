from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from xscientist.cli import _contextual_action, main as cli_main
from xscientist.research_git import init_repository
from xscientist.research_journey import workspace_action_contract
from xscientist.research_vcs import ResearchRepository
from xscientist.workspace_history import save_workspace_checkpoint
from xscientist.workspace_status import (
    _MAX_POINTER_RECORDS,
    _cas_logical_bindings,
    _deliverables_summary,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
class DemoStatusTests(unittest.TestCase):
    def test_project_progress_does_not_hide_recorded_terminal_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary, _warnings = _deliverables_summary(
                Path(td),
                progress={"results": [{"status": "success"}]},
                research={
                    "guide": {"experiment_states": {"completed": 1, "failed": 1}}
                },
                review={"object_counts": {"experiment_attempt": 2}},
                repo_status=None,
            )

        self.assertEqual(
            summary["experiment"],
            {
                "runs": 2,
                "successful": 1,
                "failed": 1,
                "recorded_attempts": 2,
                "terminal_states": {"completed": 1, "failed": 1},
                "registry_status": None,
            },
        )

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

    def test_first_status_is_ready_and_binds_record_to_the_inspected_workspace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            workspace = parent / "my-study"
            previous = Path.cwd()
            try:
                os.chdir(parent)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        cli_main(
                            [
                                "explore",
                                "./my-study",
                                "--idea",
                                "Does walking improve sleep?",
                                "--expect",
                                "Sleep improves.",
                                "--disprove",
                                "Sleep does not improve.",
                                "--test",
                                "Run a held-out comparison.",
                                "--non-interactive",
                            ]
                        ),
                        0,
                    )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(
                        cli_main(["status", "./my-study", "--lang", "en"]),
                        0,
                    )
            finally:
                os.chdir(previous)

            rendered = output.getvalue()
            self.assertIn("State: ready for the next research step", rendered)
            self.assertNotIn("State: run complete", rendered)
            run_line = next(
                line for line in rendered.splitlines() if line.startswith("Run:  ")
            )
            argv = shlex.split(run_line.removeprefix("Run:  "))
            self.assertEqual(argv[:2], ["xscientist", "record"])
            self.assertEqual((parent / argv[2]).resolve(), workspace.resolve())
            self.assertNotEqual(argv[2], ".")

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
            self.assertIn("failure_evidence", payload["objects"])
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
            repository = ResearchRepository(workspace)
            failure_evidence = repository.get(payload["objects"]["failure_evidence"])
            self.assertEqual(failure_evidence["state"], "completed")
            self.assertEqual(
                failure_evidence["payload"]["failure_reason"],
                "The first fixture contained malformed records.",
            )
            self.assertIn(
                {
                    "type": "derived_from",
                    "target": payload["objects"]["failed_attempt"],
                },
                failure_evidence["relations"],
            )
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

    def test_status_surfaces_experiment_paper_and_checkpoint_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["demo", str(workspace)]), 0)
            experiment = workspace / "02_experiments" / "20260901_reviewed"
            experiment.mkdir(parents=True)
            (experiment / "pipeline_manifest.json").write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "experiment_registry": {"status": "ready"},
                            "manuscript_state": {"status": "ready"},
                            "review_state": {"status": "ready"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "experiment_registry.jsonl").write_text(
                '{"status":"completed"}\n', encoding="utf-8"
            )
            (experiment / "paper.tex").write_text(
                "\\documentclass{article}\n", encoding="utf-8"
            )
            (experiment / "manuscript_state.json").write_text(
                json.dumps(
                    {
                        "guardrail_status": "ready",
                        "latex_path": "paper.tex",
                        "evidence_summary": {
                            "claim_count": 2,
                            "supported_claim_count": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "figure_spec.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "figure_count": 2,
                            "ready_figure_count": 1,
                        },
                        "figures": [],
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "review_state.json").write_text(
                json.dumps(
                    {
                        "active_issue_records": [{"issue_id": "RVW-1"}],
                        "repair_metrics": {"active_issue_count": 1},
                        "lane_summaries": {
                            "hostile_critic": {"blocking_issue_count": 1}
                        },
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "repair_plan.json").write_text(
                json.dumps({"summary": {"task_count": 1, "ready_task_count": 1}}),
                encoding="utf-8",
            )
            paper = workspace / "03_papers" / "final.pdf"
            paper.parent.mkdir()
            paper.write_bytes(b"%PDF-1.4\nreviewed draft\n")
            progress_path = workspace / "04_logs" / "progress.json"
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps(
                    {
                        "schema_version": "xscientist.project-progress.v1",
                        "selected_indices": [0, 1],
                        "results": [
                            {
                                "idea_idx": 0,
                                "status": "completed",
                                "research_status": "submission_ready",
                                "pipeline_manifest": (
                                    "02_experiments/20260901_reviewed/"
                                    "pipeline_manifest.json"
                                ),
                                "pdf_path": "03_papers/final.pdf",
                            },
                            {"idea_idx": 1, "status": "failed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            before_output = io.StringIO()
            with contextlib.redirect_stdout(before_output):
                self.assertEqual(cli_main(["status", str(workspace), "--json"]), 0)
            before = json.loads(before_output.getvalue())

            self.assertEqual(before["deliverables"]["experiment"]["successful"], 1)
            self.assertEqual(before["deliverables"]["experiment"]["failed"], 1)
            self.assertEqual(
                before["deliverables"]["paper"]["state"], "revision_needed"
            )
            self.assertEqual(
                before["deliverables"]["paper"]["figures"],
                {"total": 2, "ready": 1},
            )
            self.assertEqual(
                before["deliverables"]["paper"]["review"]["blocking_issues"],
                1,
            )
            self.assertGreater(before["deliverables"]["audit"]["pending"], 0)
            self.assertIn(
                "03_papers/final.pdf",
                before["deliverables"]["audit"]["unbound_paths"],
            )
            self.assertEqual(before["next_steps"][0]["code"], "bind_research_output")
            bind_action = before["next_steps"][0]["action"]
            self.assertTrue(bind_action["executable_after_binding"])
            self.assertEqual(bind_action["workspace_binding"]["mode"], "argument")
            self.assertIn("{workspace}", bind_action["argv_template"])
            self.assertEqual(bind_action["cwd_binding"]["mode"], "workspace_root")
            self.assertEqual(bind_action["cwd_binding"]["template"], "{workspace}")
            self.assertFalse(before["review"]["clean"])
            self.assertNotIn(str(workspace), before_output.getvalue())

            human_output = io.StringIO()
            with (
                contextlib.redirect_stdout(human_output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--lang", "en"]), 0
                )
            history_line = next(
                line
                for line in human_output.getvalue().splitlines()
                if line.startswith("History: ")
            )
            self.assertIn("outputs pending=", history_line)
            self.assertIn("outputs unbound=1", history_line)
            run_line = next(
                line
                for line in human_output.getvalue().splitlines()
                if line.startswith("Run:  ")
            )
            bind_argv = shlex.split(run_line.removeprefix("Run:  "))
            self.assertEqual(bind_argv[:4], ["xscientist", "research", "object", "add"])
            self.assertEqual(Path(bind_argv[4]), paper)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(bind_argv[1:]), 0)
            saved = save_workspace_checkpoint(
                workspace,
                message="checkpoint reviewed manuscript",
            )
            self.assertTrue(saved["checkpoint"]["committed"])

            after_output = io.StringIO()
            with contextlib.redirect_stdout(after_output):
                self.assertEqual(cli_main(["status", str(workspace), "--json"]), 0)
            after = json.loads(after_output.getvalue())
            audit = after["deliverables"]["audit"]
            self.assertEqual(audit["pending"], 0)
            self.assertEqual(audit["unbound"], 0)
            self.assertEqual(audit["checkpointed"], audit["total"])
            self.assertTrue(after["review"]["clean"])

    def test_status_pairs_the_selected_experiment_with_its_progress_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "project"
            older = workspace / "02_experiments" / "idea_1"
            selected = workspace / "02_experiments" / "idea_0"
            older.mkdir(parents=True)
            selected.mkdir(parents=True)
            for experiment in (older, selected):
                (experiment / "pipeline_manifest.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (experiment / "manuscript_state.json").write_text(
                    json.dumps({"guardrail_status": "ready"}), encoding="utf-8"
                )
            selected_pdf = workspace / "03_papers" / "idea_0.pdf"
            older_pdf = workspace / "03_papers" / "idea_1.pdf"
            selected_pdf.parent.mkdir()
            selected_pdf.write_bytes(b"%PDF-1.4\nselected\n")
            older_pdf.write_bytes(b"%PDF-1.4\nolder\n")
            old_ns = 1_700_000_000_000_000_000
            new_ns = old_ns + 1_000_000_000
            for path in (older, *older.iterdir()):
                os.utime(path, ns=(old_ns, old_ns))
            for path in (selected, *selected.iterdir()):
                os.utime(path, ns=(new_ns, new_ns))
            logs = workspace / "04_logs"
            logs.mkdir(parents=True)
            (logs / "progress.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "idea_idx": 0,
                                "status": "completed",
                                "research_status": "manuscript_draft",
                                "pipeline_manifest": (
                                    "02_experiments/idea_0/pipeline_manifest.json"
                                ),
                                "pdf_path": "03_papers/idea_0.pdf",
                            },
                            {
                                "idea_idx": 1,
                                "status": "completed",
                                "research_status": "submission_ready",
                                "pipeline_manifest": (
                                    "02_experiments/idea_1/pipeline_manifest.json"
                                ),
                                "pdf_path": "03_papers/idea_1.pdf",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli_main(["status", str(workspace), "--json"]), 0)
            payload = json.loads(output.getvalue())

            self.assertEqual(
                payload["deliverables"]["experiment_root"],
                "02_experiments/idea_0",
            )
            self.assertEqual(payload["deliverables"]["paper"]["state"], "draft")
            self.assertEqual(
                payload["deliverables"]["paper"]["pdf"],
                "03_papers/idea_0.pdf",
            )
            self.assertIn(
                "03_papers/idea_0.pdf",
                payload["deliverables"]["audit"]["unbound_paths"],
            )
            self.assertNotIn(
                "03_papers/idea_1.pdf",
                payload["deliverables"]["audit"]["unbound_paths"],
            )
            self.assertFalse(payload["research"]["initialized"])
            self.assertFalse(payload["review"]["available"])
            self.assertIsNone(payload["review"]["clean"])
            self.assertEqual(
                payload["next_steps"][0]["code"], "initialize_research_history"
            )
            self.assertEqual(
                payload["next_steps"][0]["action"]["argv_template"],
                ["xscientist", "research", "init", "{workspace}"],
            )

    def test_pointer_status_scan_consumes_at_most_the_record_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            pointer_root = workspace / "research-objects"
            pointer_root.mkdir()

            class FakeEntry:
                def __init__(self, index: int) -> None:
                    self.name = f"ignored-{index}.txt"
                    self.path = str(pointer_root / self.name)

                def is_file(self, *, follow_symlinks: bool) -> bool:
                    self.follow_symlinks = follow_symlinks
                    return True

            class BoundedEntries:
                def __init__(self) -> None:
                    self.count = 0

                def __enter__(self):
                    return self

                def __exit__(self, *_args) -> None:
                    return None

                def __iter__(self):
                    return self

                def __next__(self):
                    if self.count >= _MAX_POINTER_RECORDS:
                        raise AssertionError("pointer scan exceeded its hard bound")
                    self.count += 1
                    return FakeEntry(self.count)

            entries = BoundedEntries()
            with mock.patch(
                "xscientist.workspace_status.os.scandir", return_value=entries
            ):
                bindings = _cas_logical_bindings(
                    workspace,
                    repo_status={"last_checkpoint": {}},
                    pending_paths=set(),
                )

            self.assertEqual(bindings, {})
            self.assertEqual(entries.count, _MAX_POINTER_RECORDS)

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

    def test_status_refuses_symlinked_workspace_state(self) -> None:
        for parent_symlink in (False, True):
            with (
                self.subTest(parent_symlink=parent_symlink),
                tempfile.TemporaryDirectory() as td,
            ):
                base = Path(td)
                workspace = base / "workspace"
                workspace.mkdir()
                external = base / "external-logs"
                external.mkdir()
                (external / "progress.json").write_text(
                    json.dumps({"current_stage": "complete", "results": []}),
                    encoding="utf-8",
                )
                (external / "insight_report.json").write_text(
                    json.dumps({"epistemic_status": "verified"}),
                    encoding="utf-8",
                )
                if parent_symlink:
                    (workspace / "04_logs").symlink_to(
                        external, target_is_directory=True
                    )
                else:
                    logs = workspace / "04_logs"
                    logs.mkdir()
                    (logs / "progress.json").symlink_to(external / "progress.json")
                    (logs / "insight_report.json").symlink_to(
                        external / "insight_report.json"
                    )

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = cli_main(["status", str(workspace), "--json"])

                self.assertEqual(exit_code, 1)
                payload = json.loads(output.getvalue())
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["operational_state"], "invalid")
                self.assertFalse(payload["run"]["started"])
                self.assertIsNone(payload["result"]["epistemic_status"])
                corrupted = {
                    item.get("file")
                    for item in payload["errors"]
                    if item.get("code") == "workspace_state_corrupted"
                }
                self.assertIn("04_logs/progress.json", corrupted)
                self.assertIn("04_logs/insight_report.json", corrupted)
                self.assertNotIn(str(external), output.getvalue())

    def test_status_treats_malformed_state_fields_as_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "workspace"
            logs = workspace / "04_logs"
            logs.mkdir(parents=True)
            (logs / "progress.json").write_text(
                json.dumps({"results": 1}), encoding="utf-8"
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["status", str(workspace), "--json"])

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["operational_state"], "invalid")
            self.assertIn("progress.results must be a JSON array", output.getvalue())

    def test_status_never_crashes_on_malformed_review_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "workspace"
            experiment = workspace / "02_experiments" / "idea_0"
            experiment.mkdir(parents=True)
            (experiment / "pipeline_manifest.json").write_text("{}\n", encoding="utf-8")
            (experiment / "review_state.json").write_text(
                json.dumps({"active_issue_records": 1}), encoding="utf-8"
            )
            logs = workspace / "04_logs"
            logs.mkdir()
            (logs / "progress.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "status": "completed",
                                "pipeline_manifest": (
                                    "02_experiments/idea_0/pipeline_manifest.json"
                                ),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["status", str(workspace), "--json"])

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["attention_required"])
            self.assertEqual(payload["operational_state"], "needs_attention")
            self.assertIn(
                "review_state.active_issue_records must be a JSON array",
                output.getvalue(),
            )

    def test_status_does_not_treat_plain_git_commit_as_research_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["demo", str(workspace)]), 0)
            tracked_path = (
                "02_experiments/offline-autopilot-fixture/pipeline_manifest.json"
            )
            tracked = workspace / tracked_path
            tracked.write_text("{}\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "--", tracked_path],
                cwd=workspace,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "plain git artifact commit"],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["status", str(workspace), "--json"])

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["operational_state"], "invalid")
            self.assertIn(
                "research_head_not_checkpointed",
                {item["code"] for item in payload["errors"]},
            )
            audit = payload["deliverables"]["audit"]
            self.assertFalse(audit["research_checkpoint_head"])
            self.assertEqual(audit["checkpointed"], 0)
            self.assertGreater(audit["unbound"], 0)

    def test_research_init_does_not_retroactively_bind_raw_git_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            paper = workspace / "03_papers" / "paper.pdf"
            paper.parent.mkdir(parents=True)
            paper.write_bytes(b"%PDF-1.4\n%%EOF\n")
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=workspace,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Raw Commit Test"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "raw@example.invalid"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "add", "03_papers/paper.pdf"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "raw paper artifact"],
                cwd=workspace,
                check=True,
                capture_output=True,
            )
            init_repository(
                workspace,
                name="audit-boundary",
                question="# Is this artifact Research Git bound?\n",
                git_user_name="Research Test",
                git_user_email="research@example.invalid",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = cli_main(["status", str(workspace), "--json"])

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            audit = payload["deliverables"]["audit"]
            self.assertTrue(audit["research_checkpoint_head"])
            self.assertNotIn(
                "research_head_not_checkpointed",
                {item["code"] for item in payload["errors"]},
            )
            self.assertEqual(audit["checkpointed"], 0)
            self.assertIn("03_papers/paper.pdf", audit["unbound_paths"])
            self.assertIn(
                "deliverable_artifacts_unbound",
                {item["code"] for item in payload["warnings"]},
            )

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
            self.assertIn("State: experiment recorded; more evidence needed", rendered)
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
            run_step = payload["next_steps"][0]
            self.assertEqual(run_step["command"], "xscientist record {workspace}")
            self.assertFalse(run_step["action"]["executable_after_binding"])
            self.assertEqual(run_step["action"]["input_binding"]["mode"], "interactive")
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
            self.assertIn("状态：已记录实验，仍需补充证据", rendered)
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
                "State: experiment recorded; more evidence needed",
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
