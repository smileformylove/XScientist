from __future__ import annotations

import contextlib
import io
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist import cli as cli_module
from xscientist import research_commands as research_commands_module
from xscientist import research_git as research_git_module
from xscientist.cli import main as cli_main
from xscientist.research_git import (
    ResearchGitError,
    create_checkpoint,
    research_object_origin_checkpoint,
    show_checkpoint,
)
from xscientist.research_journey import build_research_guide
from xscientist.research_vcs import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required for quick-record tests")
class QuickRecordTests(unittest.TestCase):
    def _explore(self, workspace: Path, *, capture: io.StringIO | None = None) -> None:
        output = capture or io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = cli_main(
                [
                    "explore",
                    str(workspace),
                    "--idea",
                    "Does walking improve sleep?",
                    "--expect",
                    "The preregistered sleep score improves.",
                    "--disprove",
                    "The score is unchanged or worse.",
                    "--test",
                    "Compare walking and usual-activity periods.",
                    "--non-interactive",
                ]
            )
        self.assertEqual(exit_code, 0)

    def test_explore_points_directly_to_quick_record(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            output = io.StringIO()

            self._explore(workspace, capture=output)

            self.assertIn(f"xscientist record {workspace}", output.getvalue())

    def test_record_saves_attempt_evidence_artifact_and_reproduction_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "study"
            artifact = root / "result.json"
            artifact.write_text('{"sleep_score": 0.4}\n', encoding="utf-8")
            self._explore(workspace)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "Compared walking and usual-activity periods.",
                        "--status",
                        "completed",
                        "--result",
                        "Sleep score improved by 0.4.",
                        "--metric",
                        "sleep_score=0.4",
                        "--artifact",
                        f"metrics={artifact}",
                        "--reproduce-command",
                        "python run_study.py",
                        "--seed",
                        "1",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(exit_code, 0)
            rendered = output.getvalue()
            self.assertIn("Recorded experiment:", rendered)
            self.assertIn("Recorded evidence:", rendered)
            self.assertIn("Immutable result files: 1 bound", rendered)
            repository = ResearchRepository(workspace)
            attempts = repository.objects(kind="experiment_attempt")
            evidence = repository.objects(kind="evidence")
            self.assertEqual(len(attempts), 1)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(attempts[0]["payload"]["metrics"]["sleep_score"], 0.4)
            self.assertTrue(attempts[0]["payload"]["result_artifact_hashes"])
            latest_checkpoint = json.loads(
                sorted((workspace / "checkpoints").glob("*.json"))[-1].read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                latest_checkpoint["reproduce"]["command"],
                "python run_study.py",
            )
            self.assertEqual(evidence[0]["payload"]["metrics"]["sleep_score"], 0.4)
            guide = build_research_guide(workspace)
            self.assertEqual(guide["progress"]["completed_stages"], 4)
            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(git_status, "")

    def test_record_json_is_portable_and_missing_status_does_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "A run without a declared terminal state.",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 2)
            rendered_error = stderr.getvalue()
            error = json.loads(rendered_error)
            self.assertFalse(error["ok"])
            self.assertNotIn(str(workspace.resolve()), rendered_error)
            self.assertEqual(
                ResearchRepository(workspace).objects(kind="experiment_attempt"), []
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "A declared run.",
                        "--status",
                        "completed",
                        "--result",
                        "The result was bounded.",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            payload = json.loads(rendered)
            self.assertEqual(payload["workspace"], ".")
            self.assertFalse(payload["host_paths_disclosed"])
            self.assertNotIn(str(workspace.resolve()), rendered)
            self.assertEqual(
                payload["next_action"]["argv_template"],
                ["xscientist", "status", "{workspace}"],
            )

    def test_confirmatory_plan_fails_closed_with_portable_advanced_action(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "study"
            split = root / "split.txt"
            split.write_text("fixed split\n", encoding="utf-8")
            started_output = io.StringIO()
            with contextlib.redirect_stdout(started_output):
                self.assertEqual(
                    cli_main(
                        [
                            "research",
                            "start",
                            str(workspace),
                            "--question",
                            "Does walking improve sleep?",
                            "--hypothesis",
                            "Walking improves sleep.",
                            "--falsifier",
                            "Sleep is unchanged or worse.",
                            "--json",
                        ]
                    ),
                    0,
                )
            hypothesis_id = json.loads(started_output.getvalue())["hypothesis_id"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "research",
                            "preregister",
                            hypothesis_id,
                            "--dataset",
                            "sleep-v1",
                            "--metric",
                            "sleep_score",
                            "--baseline",
                            "usual-activity",
                            "--split-file",
                            str(split),
                            "--registered-by",
                            "human:reviewer",
                            "--minimum-seeds",
                            "4",
                            "--repo",
                            str(workspace),
                            "--json",
                        ]
                    ),
                    0,
                )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "Sealed run",
                        "--status",
                        "completed",
                        "--seed",
                        "41",
                        "--seed",
                        "43",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertFalse(payload["partial_success"])
            self.assertIsNone(payload["experiment_id"])
            self.assertEqual(
                payload["next_action"]["argv_template"][:3],
                ["xscientist", "research", "experiment"],
            )
            advanced_argv = payload["next_action"]["argv_template"]
            self.assertEqual(
                advanced_argv[advanced_argv.index("--status") + 1], "completed"
            )
            self.assertNotIn("--failure-class", advanced_argv)
            self.assertEqual(
                [
                    advanced_argv[index + 1]
                    for index, token in enumerate(advanced_argv)
                    if token == "--seed"
                ],
                ["41", "43", "SEED_3", "SEED_4"],
            )
            self.assertEqual(
                {
                    "SEED_3",
                    "SEED_4",
                },
                set(payload["next_action"]["input_binding"]["placeholders"])
                & {"SEED_3", "SEED_4"},
            )
            self.assertIn("{workspace}", payload["next_action"]["argv_template"])
            self.assertEqual(
                ResearchRepository(workspace).objects(kind="experiment_attempt"), []
            )

            failed_stderr = io.StringIO()
            with contextlib.redirect_stderr(failed_stderr):
                failed_code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "Sealed run failed",
                        "--status",
                        "failed",
                        "--seed",
                        "97",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(failed_code, 2)
            failed_payload = json.loads(failed_stderr.getvalue())
            failed_argv = failed_payload["next_action"]["argv_template"]
            self.assertEqual(failed_argv[failed_argv.index("--status") + 1], "failed")
            self.assertEqual(
                failed_argv[failed_argv.index("--failure-class") + 1],
                "FAILURE_CLASS",
            )
            self.assertNotIn("--config", failed_argv)
            self.assertNotIn("--result-artifact", failed_argv)
            self.assertEqual(
                [
                    failed_argv[index + 1]
                    for index, token in enumerate(failed_argv)
                    if token == "--seed"
                ],
                ["97"],
            )

            unknown_stderr = io.StringIO()
            with contextlib.redirect_stderr(unknown_stderr):
                unknown_code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(unknown_code, 2)
            unknown_payload = json.loads(unknown_stderr.getvalue())
            self.assertEqual(
                unknown_payload["next_action"]["argv_template"][:2],
                ["xscientist", "record"],
            )
            self.assertIn(
                "TERMINAL_STATUS",
                unknown_payload["next_action"]["input_binding"]["placeholders"],
            )

    def test_evidence_pending_rejects_attempt_only_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "record",
                            str(workspace),
                            "--summary",
                            "run one",
                            "--status",
                            "failed",
                            "--seed",
                            "1",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            pending_guide = build_research_guide(workspace)
            self.assertEqual(pending_guide["primary_action"]["code"], "bind_evidence")
            self.assertEqual(
                shlex.split(pending_guide["primary_action"]["command"])[:2],
                ["xscientist", "record"],
            )
            self.assertNotIn(
                "research evidence", pending_guide["primary_action"]["command"]
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "run two",
                        "--status",
                        "completed",
                        "--seed",
                        "2",
                        "--result",
                        "run two succeeded",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertFalse(payload["partial_success"])
            repository = ResearchRepository(workspace)
            self.assertEqual(len(repository.objects(kind="experiment_attempt")), 1)
            self.assertEqual(repository.objects(kind="evidence"), [])

    def test_new_attempt_is_not_hidden_by_an_old_completed_evidence_chain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "record",
                            str(workspace),
                            "--summary",
                            "The first experiment.",
                            "--status",
                            "completed",
                            "--result",
                            "The first bounded result.",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )

            repository = ResearchRepository(workspace)
            first_evidence_id = str(repository.objects(kind="evidence")[0]["object_id"])
            first_inference = research_commands_module.save_inference(
                str(workspace),
                statement="The first result supports only the tested setting.",
                premises=[first_evidence_id],
                warrant="The comparison was limited to the tested setting.",
                commit=True,
            )["object"].object_id
            research_commands_module.save_review(
                str(workspace),
                summary="Independent review of the first inference.",
                evaluates=[first_inference],
                verifier_id="human:first-reviewer",
                decision="hold",
                commit=True,
            )
            research_commands_module.save_claim(
                str(workspace),
                statement="A bounded first claim.",
                evidence_ids=[first_inference],
                commit=True,
            )

            plan_id = repository.resolve("@latest:research_plan")
            second_attempt = research_commands_module.save_experiment(
                str(workspace),
                summary="The second experiment.",
                status="completed",
                plan_id=plan_id,
                commit=True,
            )["object"].object_id

            guide = build_research_guide(workspace)
            self.assertEqual(guide["primary_action"]["code"], "bind_evidence")
            self.assertEqual(
                guide["primary_action"]["target_object_id"], second_attempt
            )
            status_output = io.StringIO()
            with contextlib.redirect_stdout(status_output):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--json"]),
                    0,
                )
            status = json.loads(status_output.getvalue())
            status_action = status["research"]["guide"]["next_steps"][0]
            self.assertEqual(status_action["code"], "bind_evidence")
            self.assertEqual(status_action["target_object_id"], second_attempt)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "record",
                            str(workspace),
                            "--result",
                            "The second bounded result.",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            second_evidence = next(
                item
                for item in repository.objects(kind="evidence")
                if any(
                    relation.get("type") == "derived_from"
                    and relation.get("target") == second_attempt
                    for relation in item.get("relations") or []
                )
            )
            guide = build_research_guide(workspace)
            self.assertEqual(guide["primary_action"]["code"], "record_inference")
            self.assertEqual(
                guide["primary_action"]["target_object_id"],
                second_evidence["object_id"],
            )

            second_inference = research_commands_module.save_inference(
                str(workspace),
                statement="The second result has a separate bounded interpretation.",
                premises=[second_evidence["object_id"]],
                warrant="The second attempt has its own recorded evidence.",
                commit=True,
            )["object"].object_id
            guide = build_research_guide(workspace)
            self.assertEqual(guide["primary_action"]["code"], "independent_review")
            self.assertEqual(
                guide["primary_action"]["target_object_id"], second_inference
            )

            research_commands_module.save_review(
                str(workspace),
                summary="Independent review of the second inference.",
                evaluates=[second_inference],
                verifier_id="human:second-reviewer",
                decision="hold",
                commit=True,
            )
            guide = build_research_guide(workspace)
            self.assertEqual(guide["primary_action"]["code"], "state_claim")
            self.assertEqual(
                guide["primary_action"]["target_object_id"], second_inference
            )

            second_claim = research_commands_module.save_claim(
                str(workspace),
                statement="A bounded second claim.",
                evidence_ids=[second_inference],
                commit=True,
            )["object"].object_id
            guide = build_research_guide(workspace)
            self.assertIn(
                guide["primary_action"]["code"],
                {"reproduce", "select_claim_for_reproduction"},
            )
            pending_claims = {
                guide["primary_action"].get("target_object_id"),
                *(guide["primary_action"].get("candidate_object_ids") or []),
            }
            self.assertIn(second_claim, pending_claims)

    def test_multiple_unbound_attempts_require_explicit_evidence_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            repository = ResearchRepository(workspace)
            plan_id = repository.resolve("@latest:research_plan")
            attempts = [
                research_commands_module.save_experiment(
                    str(workspace),
                    summary=f"Concurrent experiment {index}.",
                    status="completed",
                    plan_id=plan_id,
                    commit=True,
                )["object"].object_id
                for index in (1, 2)
            ]

            guide = build_research_guide(workspace)
            action = guide["primary_action"]
            self.assertEqual(action["code"], "select_attempt_for_evidence")
            self.assertEqual(set(action["candidate_object_ids"]), set(attempts))
            self.assertTrue(
                any(
                    warning["code"] == "ambiguous_unbound_experiment_attempt"
                    for warning in guide["warnings"]
                )
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--result",
                        "An ambiguous result.",
                        "--non-interactive",
                        "--json",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("choose", json.loads(stderr.getvalue())["error"].lower())
            self.assertEqual(repository.objects(kind="evidence"), [])

    def test_failed_result_guidance_uses_neutral_bounded_interpretation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "record",
                            str(workspace),
                            "--summary",
                            "The run failed before producing a usable output.",
                            "--status",
                            "failed",
                            "--result",
                            "No usable output was produced.",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )

            guide = build_research_guide(workspace, language="en")
            action = guide["primary_action"]
            self.assertEqual(action["code"], "record_inference")
            self.assertIn("bounded conclusion", action["title"].lower())
            self.assertNotIn("supports the conclusion", action["title"].lower())

            status_output = io.StringIO()
            with contextlib.redirect_stdout(status_output):
                self.assertEqual(
                    cli_main(["status", str(workspace), "--json"]),
                    0,
                )
            status = json.loads(status_output.getvalue())
            experiment = status["deliverables"]["experiment"]
            self.assertEqual(experiment["runs"], 1)
            self.assertEqual(experiment["failed"], 1)
            self.assertEqual(experiment["terminal_states"]["failed"], 1)

    def test_predictable_evidence_privacy_failure_precedes_attempt_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "safe run",
                        "--status",
                        "completed",
                        "--result",
                        "token sk-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertFalse(payload["partial_success"])
            self.assertNotIn("sk-", stderr.getvalue())
            self.assertEqual(
                ResearchRepository(workspace).objects(kind="experiment_attempt"), []
            )

    def test_record_refuses_a_plan_changed_while_inputs_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            repository = ResearchRepository(workspace)
            original_plan = repository.resolve("@latest:research_plan")
            hypothesis_id = repository.resolve("@latest:hypothesis")
            original_inputs = cli_module._interactive_record_inputs

            def advance_plan(parsed, *, needs_attempt, language):
                research_commands_module.save_research_plan(
                    str(workspace),
                    hypothesis_id=hypothesis_id,
                    summary="A concurrently selected replacement plan.",
                    discriminating_tests=["Run the replacement comparison."],
                    commit=True,
                )
                return original_inputs(
                    parsed,
                    needs_attempt=needs_attempt,
                    language=language,
                )

            stderr = io.StringIO()
            with (
                mock.patch.object(
                    cli_module,
                    "_interactive_record_inputs",
                    side_effect=advance_plan,
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "Ran the original comparison.",
                        "--status",
                        "completed",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertIn("research history changed", payload["error"])
            self.assertFalse(payload["partial_success"])
            self.assertNotEqual(
                repository.resolve("@latest:research_plan"), original_plan
            )
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            self.assertEqual(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
            )

    def test_record_refuses_same_head_branch_identity_change_during_input(
        self,
    ) -> None:
        for transition in ("branch", "detached"):
            with (
                self.subTest(transition=transition),
                tempfile.TemporaryDirectory() as td,
            ):
                workspace = Path(td) / "study"
                self._explore(workspace)
                repository = ResearchRepository(workspace)
                selected_status = repository.status()
                original_inputs = cli_module._interactive_record_inputs

                def switch_branch(parsed, *, needs_attempt, language):
                    command = (
                        ["git", "switch", "-c", "concurrent-line"]
                        if transition == "branch"
                        else ["git", "switch", "--detach", "HEAD"]
                    )
                    subprocess.run(
                        command,
                        cwd=workspace,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    return original_inputs(
                        parsed,
                        needs_attempt=needs_attempt,
                        language=language,
                    )

                stderr = io.StringIO()
                with (
                    mock.patch.object(
                        cli_module,
                        "_interactive_record_inputs",
                        side_effect=switch_branch,
                    ),
                    contextlib.redirect_stderr(stderr),
                ):
                    code = cli_main(
                        [
                            "record",
                            str(workspace),
                            "--summary",
                            "Ran against the originally selected research line.",
                            "--status",
                            "completed",
                            "--non-interactive",
                            "--json",
                        ]
                    )

                self.assertEqual(code, 2)
                payload = json.loads(stderr.getvalue())
                self.assertIn("research history changed", payload["error"])
                self.assertFalse(payload["partial_success"])
                current_status = repository.status()
                self.assertEqual(current_status["head"], selected_status["head"])
                self.assertNotEqual(current_status["branch"], selected_status["branch"])
                self.assertEqual(
                    repository.objects(kind="experiment_attempt"),
                    [],
                )
                self.assertEqual(
                    subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=workspace,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout,
                    "",
                )

    def test_record_pins_a_symlinked_workspace_for_the_whole_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original_workspace = root / "original"
            clone_workspace = root / "clone"
            workspace_link = root / "study"
            self._explore(original_workspace)
            shutil.copytree(original_workspace, clone_workspace)
            workspace_link.symlink_to(original_workspace, target_is_directory=True)
            original_inputs = cli_module._interactive_record_inputs

            def switch_workspace(parsed, *, needs_attempt, language):
                workspace_link.unlink()
                workspace_link.symlink_to(clone_workspace, target_is_directory=True)
                return original_inputs(
                    parsed,
                    needs_attempt=needs_attempt,
                    language=language,
                )

            with (
                mock.patch.object(
                    cli_module,
                    "_interactive_record_inputs",
                    side_effect=switch_workspace,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = cli_main(
                    [
                        "record",
                        str(workspace_link),
                        "--summary",
                        "Ran against the originally selected workspace.",
                        "--status",
                        "completed",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(
                len(
                    ResearchRepository(original_workspace).objects(
                        kind="experiment_attempt"
                    )
                ),
                1,
            )
            self.assertEqual(
                ResearchRepository(clone_workspace).objects(kind="experiment_attempt"),
                [],
            )

    def test_record_refuses_an_attempt_changed_before_evidence_binding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "record",
                            str(workspace),
                            "--summary",
                            "Ran the first comparison.",
                            "--status",
                            "completed",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            repository = ResearchRepository(workspace)
            original_attempt = repository.resolve("@latest:experiment_attempt")
            plan_id = repository.resolve("@latest:research_plan")
            original_inputs = cli_module._interactive_record_inputs

            def advance_attempt(parsed, *, needs_attempt, language):
                research_commands_module.save_experiment(
                    str(workspace),
                    summary="A concurrently completed replacement attempt.",
                    status="completed",
                    plan_id=plan_id,
                    commit=True,
                )
                return original_inputs(
                    parsed,
                    needs_attempt=needs_attempt,
                    language=language,
                )

            stderr = io.StringIO()
            with (
                mock.patch.object(
                    cli_module,
                    "_interactive_record_inputs",
                    side_effect=advance_attempt,
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--result",
                        "Result from the original attempt.",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertIn("research history changed", payload["error"])
            self.assertFalse(payload["partial_success"])
            self.assertNotEqual(
                repository.resolve("@latest:experiment_attempt"), original_attempt
            )
            self.assertEqual(len(repository.objects(kind="experiment_attempt")), 2)
            self.assertEqual(repository.objects(kind="evidence"), [])
            self.assertEqual(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
            )

    def test_unavoidable_second_stage_failure_reports_portable_partial_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            stderr = io.StringIO()
            with (
                mock.patch(
                    "xscientist.research_commands.save_evidence",
                    side_effect=ResearchGitError("simulated evidence commit failure"),
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "completed run",
                        "--status",
                        "completed",
                        "--result",
                        "bounded result",
                        "--reproduce-command",
                        "python reproduce.py",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertTrue(payload["partial_success"])
            self.assertTrue(payload["experiment_id"].startswith("rso-"))
            self.assertEqual(
                payload["next_action"]["argv_template"][:3],
                ["xscientist", "record", "{workspace}"],
            )
            self.assertFalse(payload["next_action"]["executable_after_binding"])
            self.assertEqual(
                len(ResearchRepository(workspace).objects(kind="experiment_attempt")),
                1,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                retry_code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--result",
                        "bounded result",
                        "--non-interactive",
                    ]
                )
            self.assertEqual(retry_code, 0)
            self.assertEqual(
                show_checkpoint(workspace)["checkpoint"]["reproduce"]["command"],
                "python reproduce.py",
            )

    def test_concurrent_attempt_uses_explicit_partial_recovery_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            plan_id = ResearchRepository(workspace).resolve("@latest:research_plan")
            real_save_evidence = research_commands_module.save_evidence

            def advance_attempt_then_save_evidence(*args, **kwargs):
                research_commands_module.save_experiment(
                    str(workspace),
                    summary="A concurrent second attempt.",
                    status="completed",
                    plan_id=plan_id,
                    commit=True,
                )
                return real_save_evidence(*args, **kwargs)

            stderr = io.StringIO()
            with (
                mock.patch.object(
                    research_commands_module,
                    "save_evidence",
                    side_effect=advance_attempt_then_save_evidence,
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "The preserved first attempt.",
                        "--status",
                        "completed",
                        "--result",
                        "Result belonging only to the first attempt.",
                        "--reproduce-command",
                        "python reproduce.py",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertTrue(payload["partial_success"])
            preserved_attempt = payload["experiment_id"]
            repository = ResearchRepository(workspace)
            self.assertNotEqual(
                repository.resolve("@latest:experiment_attempt"), preserved_attempt
            )
            recovery_argv = payload["next_action"]["argv_template"]
            self.assertEqual(recovery_argv[:3], ["xscientist", "research", "evidence"])
            self.assertEqual(
                recovery_argv[recovery_argv.index("--attempt") + 1],
                preserved_attempt,
            )
            self.assertEqual(
                recovery_argv[recovery_argv.index("--repo") + 1], "{workspace}"
            )
            self.assertEqual(
                recovery_argv[recovery_argv.index("--reproduce-command") + 1],
                "python reproduce.py",
            )
            self.assertFalse(payload["next_action"]["executable_after_binding"])
            self.assertNotIn(str(workspace.resolve()), stderr.getvalue())

            bound_argv = [
                (
                    str(workspace)
                    if token == "{workspace}"
                    else (
                        "Result belonging only to the first attempt."
                        if token == "WHAT THE RESULT SHOWS"
                        else token
                    )
                )
                for token in recovery_argv
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(bound_argv[1:]), 0)
            evidence = repository.objects(kind="evidence")
            self.assertEqual(len(evidence), 1)
            self.assertEqual(
                [
                    relation["target"]
                    for relation in evidence[0]["relations"]
                    if relation.get("type") == "derived_from"
                ],
                [preserved_attempt],
            )
            self.assertEqual(
                show_checkpoint(workspace)["checkpoint"]["reproduce"]["command"],
                "python reproduce.py",
            )

    def test_second_stage_interrupt_reports_the_preserved_experiment(self) -> None:
        for as_json in (False, True):
            with self.subTest(as_json=as_json), tempfile.TemporaryDirectory() as td:
                workspace = Path(td) / "study"
                self._explore(workspace)
                stderr = io.StringIO()
                argv = [
                    "record",
                    str(workspace),
                    "--summary",
                    "completed run",
                    "--status",
                    "completed",
                    "--result",
                    "bounded result",
                    "--non-interactive",
                ]
                if as_json:
                    argv.append("--json")
                with (
                    mock.patch(
                        "xscientist.research_commands.save_evidence",
                        side_effect=KeyboardInterrupt(),
                    ),
                    contextlib.redirect_stderr(stderr),
                ):
                    code = cli_main(argv)

                self.assertEqual(code, 130)
                rendered = stderr.getvalue()
                repository = ResearchRepository(workspace)
                attempts = repository.objects(kind="experiment_attempt")
                self.assertEqual(len(attempts), 1)
                self.assertEqual(repository.objects(kind="evidence"), [])
                if as_json:
                    payload = json.loads(rendered)
                    self.assertEqual(payload["error_code"], "record_interrupted")
                    self.assertTrue(payload["partial_success"])
                    self.assertEqual(payload["experiment_id"], attempts[0]["object_id"])
                    self.assertEqual(
                        payload["next_action"]["argv_template"][:3],
                        ["xscientist", "record", "{workspace}"],
                    )
                    self.assertNotIn(str(workspace.resolve()), rendered)
                else:
                    self.assertIn("Experiment checkpoint preserved:", rendered)
                    self.assertIn(attempts[0]["object_id"], rendered)
                    self.assertIn("Next: xscientist record", rendered)
                    self.assertIn("{workspace}", rendered)
                    self.assertNotIn(str(workspace.resolve()), rendered)

    def test_evidence_checkpoint_failure_rolls_back_for_immediate_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            real_create_checkpoint = research_commands_module.create_checkpoint
            checkpoint_calls = 0

            def fail_second_checkpoint(*args, **kwargs):
                nonlocal checkpoint_calls
                checkpoint_calls += 1
                if checkpoint_calls == 2:
                    raise ResearchGitError("simulated evidence checkpoint failure")
                return real_create_checkpoint(*args, **kwargs)

            stderr = io.StringIO()
            with (
                mock.patch.object(
                    research_commands_module,
                    "create_checkpoint",
                    side_effect=fail_second_checkpoint,
                ),
                contextlib.redirect_stderr(stderr),
            ):
                code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--summary",
                        "completed run",
                        "--status",
                        "completed",
                        "--result",
                        "bounded result",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertTrue(payload["partial_success"])
            repository = ResearchRepository(workspace)
            self.assertEqual(len(repository.objects(kind="experiment_attempt")), 1)
            self.assertEqual(repository.objects(kind="evidence"), [])
            self.assertEqual(
                build_research_guide(workspace)["primary_action"]["code"],
                "bind_evidence",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                retry_code = cli_main(
                    [
                        "record",
                        str(workspace),
                        "--result",
                        "bounded result",
                        "--non-interactive",
                    ]
                )
            self.assertEqual(retry_code, 0)
            self.assertEqual(len(repository.objects(kind="evidence")), 1)
            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(git_status, "")

    def test_evidence_retry_origin_lookup_is_bounded_by_one_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "record",
                            str(workspace),
                            "--summary",
                            "completed run",
                            "--status",
                            "completed",
                            "--reproduce-command",
                            "python reproduce.py",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            repository = ResearchRepository(workspace)
            attempt_id = repository.resolve("@latest:experiment_attempt")
            for index in range(64):
                repository.record("observation", {"measurement": index})
            create_checkpoint(
                workspace,
                stage="experiment",
                subject="record unrelated observations",
            )

            calls: list[list[str]] = []
            real_run_git = research_git_module._run_git

            def counted_run_git(repo, args, *, check=True):
                calls.append(list(args))
                return real_run_git(repo, args, check=check)

            with mock.patch.object(
                research_git_module,
                "_run_git",
                side_effect=counted_run_git,
            ):
                origin = research_object_origin_checkpoint(
                    workspace, attempt_id, kind="experiment_attempt"
                )

            self.assertEqual(
                origin["checkpoint"]["reproduce"]["command"],
                "python reproduce.py",
            )
            self.assertLessEqual(len(calls), 12)
            object_show_calls = [
                args
                for args in calls
                if args and args[0] == "show" and ".xscientist/objects/" in args[-1]
            ]
            self.assertEqual(len(object_show_calls), 1)

    def test_interactive_eof_and_interrupt_are_short_and_structured(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            self._explore(workspace)
            for error, expected_code, expected_text in (
                (EOFError(), 2, "interactive input ended"),
                (KeyboardInterrupt(), 130, "interrupted"),
            ):
                with self.subTest(error=type(error).__name__):
                    stderr = io.StringIO()
                    with (
                        mock.patch.object(sys.stdin, "isatty", return_value=True),
                        mock.patch("builtins.input", side_effect=error),
                        contextlib.redirect_stderr(stderr),
                    ):
                        code = cli_main(["record", str(workspace)])
                    self.assertEqual(code, expected_code)
                    self.assertIn(expected_text, stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())


class RootCommandHelpTests(unittest.TestCase):
    def test_publication_help_keeps_an_explicit_cost_budget(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["--help"])

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("--autopilot publication --max-cost-usd 10", rendered)
        shortest = rendered.split("Start here:", 1)[0]
        self.assertLess(shortest.index(" explore "), shortest.index(" record "))
        self.assertLess(shortest.index(" record "), shortest.index(" status "))

    def test_unknown_command_suggests_one_short_repair(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = cli_main(["statsu"])

        self.assertEqual(exit_code, 2)
        rendered = stderr.getvalue()
        self.assertIn("Did you mean `status`?", rendered)
        self.assertNotIn("choose from", rendered)


if __name__ == "__main__":
    unittest.main()
