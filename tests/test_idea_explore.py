from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist import ResearchRepository
from xscientist.cli import main as cli_main
from xscientist.onboarding import create_workspace


@unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS")
class IdeaExploreTests(unittest.TestCase):
    def test_one_idea_creates_an_honest_provider_free_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "explore",
                        str(workspace),
                        "--idea",
                        "Does daily walking improve sleep quality?",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(
                payload["schema_version"], "xscientist.idea-exploration.v1"
            )
            self.assertEqual(payload["framing"]["status"], "idea_saved")
            self.assertEqual(
                payload["framing"]["missing"],
                ["expected_observation", "disconfirming_result"],
            )
            self.assertFalse(payload["safety"]["api_key_required"])
            self.assertFalse(payload["safety"]["provider_used"])
            self.assertFalse(payload["safety"]["external_network_used"])
            self.assertFalse(payload["safety"]["evidence_generated"])
            self.assertFalse(payload["safety"]["conclusion_generated"])
            self.assertEqual(
                payload["guide"]["next_steps"][0]["code"], "record_hypothesis"
            )
            self.assertEqual(
                payload["guide"]["next_steps"][0]["command"],
                "xscientist explore {workspace}",
            )
            action = payload["guide"]["next_steps"][0]["action"]
            self.assertEqual(
                action["argv_template"],
                ["xscientist", "explore", "{workspace}"],
            )
            self.assertEqual(action["workspace_binding"]["mode"], "argument")
            self.assertTrue(action["executable_after_binding"])
            self.assertEqual(
                action["workspace_binding"]["source"], "invocation_workspace"
            )
            self.assertEqual(action["cwd_binding"]["mode"], "caller")
            self.assertEqual(
                payload["continue_command"], "xscientist explore {workspace}"
            )
            self.assertEqual(
                payload["continue_action"]["argv_template"],
                ["xscientist", "explore", "{workspace}"],
            )
            self.assertEqual(
                payload["workspace_context"]["workspace_placeholder"], "{workspace}"
            )
            self.assertNotIn(str(workspace), output.getvalue())
            self.assertNotIn("xscientist explore .", output.getvalue())
            self.assertFalse(payload["privacy"]["host_paths_disclosed"])
            repository = ResearchRepository(workspace)
            self.assertEqual(len(repository.objects(kind="question")), 1)
            self.assertEqual(len(repository.objects(kind="research_goal")), 1)
            self.assertEqual(repository.objects(kind="hypothesis"), [])
            self.assertEqual(repository.objects(kind="evidence"), [])
            self.assertEqual(repository.status()["eligible_changes"], [])

    def test_plain_answers_create_a_falsifiable_first_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "explore",
                        str(workspace),
                        "--idea",
                        "Does daily walking improve sleep quality?",
                        "--expect",
                        "Daily walking improves a preregistered sleep score.",
                        "--disprove",
                        "The score is unchanged or worse.",
                        "--test",
                        "Compare walking and usual-activity periods.",
                        "--success-rule",
                        "The score improves beyond measurement noise.",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["framing"]["status"], "planned")
            self.assertEqual(payload["framing"]["missing"], [])
            self.assertEqual(
                payload["guide"]["next_steps"][0]["code"], "run_experiment"
            )
            repository = ResearchRepository(workspace)
            hypothesis = repository.objects(kind="hypothesis")[0]
            plan = repository.objects(kind="research_plan")[0]
            self.assertEqual(
                hypothesis["payload"]["falsifier"],
                "The score is unchanged or worse.",
            )
            self.assertEqual(
                plan["payload"]["success_rule"],
                "The score improves beyond measurement noise.",
            )
            self.assertEqual(hypothesis["actor"]["authority"], "human")
            self.assertEqual(repository.status()["eligible_changes"], [])

    def test_rerun_adds_only_the_missing_step_and_preserves_ambient_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--idea",
                            "Does X change Y?",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            question_file = workspace / "question.md"
            question_file.write_text(
                question_file.read_text(encoding="utf-8") + "Personal note.\n",
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--expect",
                            "X increases Y.",
                            "--disprove",
                            "Y is unchanged or lower.",
                            "--test",
                            "Compare X with a no-X baseline.",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )

            repository = ResearchRepository(workspace)
            self.assertEqual(len(repository.objects(kind="hypothesis")), 1)
            self.assertEqual(len(repository.objects(kind="research_plan")), 1)
            self.assertEqual(repository.status()["eligible_changes"], ["question.md"])
            self.assertIn("Personal note.", question_file.read_text(encoding="utf-8"))

    def test_saved_idea_workspace_can_gain_a_local_model_runtime_safely(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--idea",
                            "Does X change Y?",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            original_question = (workspace / "question.md").read_text(encoding="utf-8")
            original_ignore = (workspace / ".gitignore").read_text(encoding="utf-8")

            created = create_workspace(
                workspace,
                provider="ollama",
                model="ollama/qwen2.5:7b",
                preserve_existing=True,
            )

            self.assertEqual(created["preserved_files"], [".gitignore"])
            self.assertEqual(
                (workspace / "question.md").read_text(encoding="utf-8"),
                original_question,
            )
            self.assertEqual(
                (workspace / ".gitignore").read_text(encoding="utf-8"),
                original_ignore,
            )
            self.assertTrue((workspace / ".xscientist" / "providers.json").is_file())
            self.assertTrue((workspace / "bfts_config.yaml").is_file())

    def test_start_reuses_the_saved_question_when_adding_a_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--idea",
                            "Does X change Y?",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )

            output = io.StringIO()
            with (
                mock.patch.dict(
                    "os.environ",
                    {"AI_SCIENTIST_AUTH_FILE": str(root / "auth.json")},
                ),
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--user",
                        "test-researcher",
                        "--prepare-only",
                        "--skip-credentials",
                        "--non-interactive",
                        "--json",
                    ]
                )

            self.assertIn(code, {0, 1})
            payload = json.loads(output.getvalue())
            self.assertNotEqual(payload["phase"], "input")
            self.assertTrue(payload["phases"]["workspace"]["ok"])
            self.assertFalse(payload["phases"]["research_vcs"]["created"])
            self.assertEqual(
                (workspace / "question.md").read_text(encoding="utf-8"),
                "# Research question\n\nDoes X change Y?\n",
            )
            self.assertEqual(
                ResearchRepository(workspace).status()["eligible_changes"], []
            )

    def test_start_rejects_a_conflicting_existing_question_before_any_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--idea",
                            "Does X change Y?",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            question_before = (workspace / "question.md").read_bytes()
            self.assertFalse((workspace / "topic.md").exists())

            output = io.StringIO()
            with (
                mock.patch("xscientist.onboarding.create_workspace") as create,
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does A change B?",
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("conflicts", json.loads(output.getvalue())["error"])
            create.assert_not_called()
            self.assertEqual((workspace / "question.md").read_bytes(), question_before)
            self.assertFalse((workspace / "topic.md").exists())
            self.assertFalse((workspace / ".xscientist" / "providers.json").exists())

    def test_start_rejects_a_conflicting_topic_without_rewriting_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--idea",
                            "Does X change Y?",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            question_path = workspace / "question.md"
            topic_path = workspace / "topic.md"
            topic_path.write_text(
                "# Research question\n\nDoes A change B?\n",
                encoding="utf-8",
            )
            question_before = question_path.read_bytes()
            topic_before = topic_path.read_bytes()
            output = io.StringIO()
            with (
                mock.patch("xscientist.onboarding.create_workspace") as create,
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X change Y?",
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("topic.md", json.loads(output.getvalue())["error"])
            create.assert_not_called()
            self.assertEqual(question_path.read_bytes(), question_before)
            self.assertEqual(topic_path.read_bytes(), topic_before)

    def test_start_failures_do_not_rewrite_existing_research_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--idea",
                            "Does X change Y?",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )

            question_path = workspace / "question.md"
            topic_path = workspace / "topic.md"
            question_path.write_text(
                "# Research question\n\nDoes   X change Y?\n",
                encoding="utf-8",
            )
            topic_path.write_text(
                "# Research topic\n\nDoes X change   Y?\n",
                encoding="utf-8",
            )
            question_before = question_path.read_bytes()
            topic_before = topic_path.read_bytes()
            unready = {
                "schema": "xscientist.doctor.v1",
                "ok": False,
                "configuration_ready": True,
                "runtime_ready": False,
                "checks": {},
                "next_actions": ["install the configured runtime"],
            }
            ready = {
                **unready,
                "ok": True,
                "runtime_ready": True,
                "next_actions": [],
            }
            auth = (True, "ok", {"username": "test-researcher"})

            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=auth,
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=unready),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                doctor_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        " does x change y? ",
                        "--provider",
                        "ollama",
                        "--model",
                        "ollama/qwen2.5:7b",
                        "--prepare-only",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(doctor_code, 1)
            self.assertEqual(question_path.read_bytes(), question_before)
            self.assertEqual(topic_path.read_bytes(), topic_before)

            with (
                mock.patch(
                    "ai_scientist.utils.auth_session.validate_session",
                    return_value=auth,
                ),
                mock.patch("xscientist.diagnostics.diagnose", return_value=ready),
                mock.patch("xscientist.cli.project_main", return_value=1),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch_code = cli_main(
                    [
                        "start",
                        str(workspace),
                        "--question",
                        "Does X change Y?",
                        "--allow-synthetic-data",
                        "--skip-credentials",
                        "--json",
                    ]
                )

            self.assertEqual(launch_code, 1)
            self.assertEqual(question_path.read_bytes(), question_before)
            self.assertEqual(topic_path.read_bytes(), topic_before)

    def test_interactive_flow_uses_plain_questions_and_can_finish_a_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            answers = [
                "Does X change Y?",
                "X increases Y.",
                "Y is unchanged or lower.",
                "Compare X with no X.",
                "Y exceeds the fixed threshold.",
            ]
            output = io.StringIO()
            with (
                mock.patch("sys.stdin.isatty", return_value=True),
                mock.patch("builtins.input", side_effect=answers) as prompt,
                contextlib.redirect_stdout(output),
            ):
                code = cli_main(["explore", str(workspace), "--lang", "en"])

            self.assertEqual(code, 0)
            self.assertEqual(prompt.call_count, 5)
            self.assertIn("Rigor check:", output.getvalue())
            self.assertIn("no evidence or conclusion was generated", output.getvalue())
            self.assertEqual(
                len(ResearchRepository(workspace).objects(kind="research_plan")), 1
            )

    def test_partial_noninteractive_hypothesis_fails_before_creating_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(
                    [
                        "explore",
                        str(workspace),
                        "--idea",
                        "Does X change Y?",
                        "--expect",
                        "X increases Y.",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("--expect and --disprove", stderr.getvalue())
            self.assertFalse(workspace.exists())

    def test_existing_workspace_refuses_a_different_idea(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cli_main(
                        [
                            "explore",
                            str(workspace),
                            "--idea",
                            "Does X change Y?",
                            "--non-interactive",
                        ]
                    ),
                    0,
                )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli_main(
                    [
                        "explore",
                        str(workspace),
                        "--idea",
                        "Does A change B?",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("different idea", stderr.getvalue())
            self.assertEqual(
                len(ResearchRepository(workspace).objects(kind="question")), 1
            )

    def test_nonempty_destination_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "study"
            workspace.mkdir()
            note = workspace / "notes.txt"
            note.write_text("keep\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli_main(
                    [
                        "explore",
                        str(workspace),
                        "--idea",
                        "Does X change Y?",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(note.read_text(encoding="utf-8"), "keep\n")

    def test_chinese_output_states_the_scientific_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli_main(
                    [
                        "explore",
                        str(Path(td) / "study"),
                        "--idea",
                        "每天散步是否改善睡眠？",
                        "--lang",
                        "zh",
                        "--non-interactive",
                    ]
                )

            self.assertEqual(code, 0)
            rendered = output.getvalue()
            self.assertIn("严谨性检查", rendered)
            self.assertIn("未生成证据或结论", rendered)
            self.assertIn("免 Key 的本地 Ollama", rendered)


if __name__ == "__main__":
    unittest.main()
