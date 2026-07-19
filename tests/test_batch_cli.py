from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from ai_scientist.apps import batch
from ai_scientist.apps.batch_cli import build_parser


class BatchCliTests(unittest.TestCase):
    def test_parser_exposes_injected_defaults_and_batch_options(self) -> None:
        parser = build_parser(
            default_research_dir="/tmp/research",
            default_writing_profile="logic_first",
            writing_profiles=["default", "logic_first"],
            workflow_modes=["adaptive", "program_driven"],
        )

        args = parser.parse_args(
            [
                "--ideas",
                "ideas.json",
                "--paper-types",
                "normal",
                "journal",
                "--workflow-mode",
                "program_driven",
                "--no-integrity-forensics",
            ]
        )

        self.assertEqual(args.research_dir, "/tmp/research")
        self.assertEqual(args.writing_profile, "logic_first")
        self.assertEqual(args.paper_types, ["normal", "journal"])
        self.assertEqual(args.workflow_mode, "program_driven")
        self.assertFalse(args.integrity_forensics)

        help_output = io.StringIO()
        with contextlib.redirect_stdout(help_output):
            parser.print_help()
        self.assertIn(
            "python -m xscientist batch \\\n     --topic", help_output.getvalue()
        )

    def test_batch_main_accepts_argv_and_help_skips_runtime_guards(self) -> None:
        with (
            mock.patch.object(batch, "require_login") as require_login,
            mock.patch.object(batch, "initialize_runtime") as initialize_runtime,
            self.assertRaises(SystemExit) as raised,
        ):
            batch.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        require_login.assert_not_called()
        initialize_runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
