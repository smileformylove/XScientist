from __future__ import annotations

import unittest
from unittest import mock

from ai_scientist.apps import project
from ai_scientist.apps.project_cli import build_parser


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


if __name__ == "__main__":
    unittest.main()
