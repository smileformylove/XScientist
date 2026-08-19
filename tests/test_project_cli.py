from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from ai_scientist.apps import project
from ai_scientist.apps.project_cli import build_parser
from ai_scientist.utils.workflow_cli import (
    apply_project_autopilot_profile,
    normalize_project_workflow_args,
)


class ProjectCliTests(unittest.TestCase):
    def test_parser_exposes_injected_defaults_and_project_options(self) -> None:
        parser = build_parser(
            default_output_root="/tmp/research",
            default_writing_profile="logic_first",
            writing_profiles=["default", "logic_first"],
            workflow_modes=["adaptive", "review_board"],
        )

        args = parser.parse_args(
            [
                "demo",
                "--ideas",
                "ideas.json",
                "--workflow-mode",
                "review_board",
                "--writeup-type",
                "journal",
                "--no-integrity-forensics",
            ]
        )

        self.assertEqual(args.project_dir, "demo")
        self.assertEqual(args.output_root, "/tmp/research")
        self.assertEqual(args.writing_profile, "logic_first")
        self.assertEqual(args.workflow_mode, "review_board")
        self.assertEqual(args.writeup_type, "journal")
        self.assertFalse(args.integrity_forensics)

    def test_project_main_accepts_argv_and_help_skips_runtime_guards(self) -> None:
        with (
            mock.patch.object(project, "require_login") as require_login,
            mock.patch.object(project, "initialize_runtime") as initialize_runtime,
            self.assertRaises(SystemExit) as raised,
        ):
            project.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        require_login.assert_not_called()
        initialize_runtime.assert_not_called()

    def test_project_help_is_consistently_english(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            project.main(["--help"])

        rendered = output.getvalue()
        self.assertIn("Start from a plain-language question", rendered)
        self.assertNotRegex(rendered, r"[\u4e00-\u9fff]")

    def test_workspace_default_model_applies_to_every_role_and_can_be_overridden(
        self,
    ) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "AI_SCIENTIST_DEFAULT_MODEL": "openai/research-model",
                "AI_SCIENTIST_MODEL_REVIEW": "anthropic/review-model",
            },
            clear=False,
        ):
            parser = build_parser(
                default_output_root="/tmp/research",
                default_writing_profile="default",
                writing_profiles=["default"],
                workflow_modes=["adaptive"],
            )
            args = parser.parse_args(
                ["demo", "--model-citation", "gemini/citation-model"]
            )

        self.assertEqual(args.model_ideation, "openai/research-model")
        self.assertEqual(args.model_agg_plots, "openai/research-model")
        self.assertEqual(args.model_writeup, "openai/research-model")
        self.assertEqual(args.model_writeup_small, "openai/research-model")
        self.assertEqual(args.model_citation, "gemini/citation-model")
        self.assertEqual(args.model_review, "anthropic/review-model")

    def test_plain_language_question_and_autopilot_are_first_class_options(
        self,
    ) -> None:
        parser = build_parser(
            default_output_root="/tmp/research",
            default_writing_profile="default",
            writing_profiles=["default"],
            workflow_modes=[
                "adaptive",
                "agentic_tree",
                "program_driven",
                "multi_agent_board",
            ],
        )
        args = parser.parse_args(
            [
                "demo",
                "--question",
                "Why does the effect fail?",
                "--autopilot",
                "discovery",
            ]
        )
        apply_project_autopilot_profile(args)
        normalize_project_workflow_args(args)

        self.assertEqual(args.question, "Why does the effect fail?")
        self.assertEqual(args.autopilot, "discovery")
        self.assertEqual(args.workflow_mode, "agentic_tree")
        self.assertTrue(args.resume)
        self.assertTrue(args.rank_ideas)
        self.assertTrue(args.parallel)
        self.assertGreaterEqual(args.num_ideas, 5)
        self.assertEqual(args.top_k_ideas, 3)
        self.assertTrue(args.integrity_forensics)

    def test_bare_autopilot_uses_cost_bounded_balanced_profile(self) -> None:
        parser = build_parser(
            default_output_root="/tmp/research",
            default_writing_profile="default",
            writing_profiles=["default"],
            workflow_modes=["adaptive", "program_driven"],
        )
        args = parser.parse_args(["demo", "--autopilot"])
        apply_project_autopilot_profile(args)

        self.assertEqual(args.autopilot, "balanced")
        self.assertEqual(args.workflow_mode, "program_driven")
        self.assertEqual(args.top_k_ideas, 2)


if __name__ == "__main__":
    unittest.main()
