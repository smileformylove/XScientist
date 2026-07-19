from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock


class AppCompatibilityTests(unittest.TestCase):
    def test_run_project_module_aliases_internal_application(self) -> None:
        legacy = importlib.import_module("run_project")
        internal = importlib.import_module("ai_scientist.apps.project")

        self.assertIs(legacy, internal)
        self.assertIs(legacy.process_single_idea, internal.process_single_idea)

    def test_legacy_mock_patch_targets_internal_globals(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.project")

        with mock.patch("run_project.idea_to_markdown") as patched:
            self.assertIs(internal.idea_to_markdown, patched)

    def test_project_help_does_not_require_a_login_session(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.project")

        with (
            mock.patch.object(sys, "argv", ["xscientist project", "--help"]),
            mock.patch.object(internal, "require_login") as require_login,
            self.assertRaises(SystemExit) as raised,
        ):
            internal.main()

        self.assertEqual(raised.exception.code, 0)
        require_login.assert_not_called()

    def test_continuous_generator_aliases_internal_batch_application(self) -> None:
        legacy = importlib.import_module("continuous_paper_generator")
        internal = importlib.import_module("ai_scientist.apps.batch")

        self.assertIs(legacy, internal)
        self.assertIs(
            legacy.ContinuousPaperGenerator, internal.ContinuousPaperGenerator
        )

    def test_legacy_batch_mock_patch_targets_internal_globals(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.batch")

        with mock.patch(
            "continuous_paper_generator.record_quality_fallback_if_needed"
        ) as patched:
            self.assertIs(internal.record_quality_fallback_if_needed, patched)

    def test_batch_help_does_not_require_a_login_session(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.batch")

        with (
            mock.patch.object(sys, "argv", ["xscientist batch", "--help"]),
            mock.patch.object(internal, "require_login") as require_login,
            self.assertRaises(SystemExit) as raised,
        ):
            internal.main()

        self.assertEqual(raised.exception.code, 0)
        require_login.assert_not_called()

    def test_research_manager_aliases_internal_manager_application(self) -> None:
        legacy = importlib.import_module("research_manager")
        internal = importlib.import_module("ai_scientist.apps.manager")

        self.assertIs(legacy, internal)
        self.assertIs(legacy.ResearchManager, internal.ResearchManager)

    def test_manager_help_does_not_require_a_login_session(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.manager")

        with (
            mock.patch.object(sys, "argv", ["xscientist manager", "--help"]),
            mock.patch.object(internal, "require_login") as require_login,
            self.assertRaises(SystemExit) as raised,
        ):
            internal.main()

        self.assertEqual(raised.exception.code, 0)
        require_login.assert_not_called()

    def test_research_daemon_aliases_internal_daemon_application(self) -> None:
        legacy = importlib.import_module("continuous_research_daemon")
        internal = importlib.import_module("ai_scientist.apps.daemon")

        self.assertIs(legacy, internal)
        self.assertIs(legacy.build_parser, internal.build_parser)

    def test_legacy_daemon_mock_patch_targets_internal_globals(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.daemon")

        with mock.patch("continuous_research_daemon.ResearchManager") as patched:
            self.assertIs(internal.ResearchManager, patched)

    def test_daemon_help_does_not_require_a_login_session(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.daemon")

        with (
            mock.patch.object(sys, "argv", ["xscientist daemon", "--help"]),
            mock.patch.object(internal, "require_login") as require_login,
            self.assertRaises(SystemExit) as raised,
        ):
            internal.main()

        self.assertEqual(raised.exception.code, 0)
        require_login.assert_not_called()

    def test_auth_cli_aliases_internal_application(self) -> None:
        legacy = importlib.import_module("auth_cli")
        internal = importlib.import_module("ai_scientist.apps.auth")

        self.assertIs(legacy, internal)

    def test_feedback_cli_aliases_internal_application(self) -> None:
        legacy = importlib.import_module("feedback_cli")
        internal = importlib.import_module("ai_scientist.apps.feedback")

        self.assertIs(legacy, internal)

    def test_ara_cli_aliases_internal_application(self) -> None:
        legacy = importlib.import_module("run_ara_fork")
        internal = importlib.import_module("ai_scientist.apps.ara")

        self.assertIs(legacy, internal)

    def test_validate_cli_aliases_internal_application(self) -> None:
        legacy = importlib.import_module("validate_repo")
        internal = importlib.import_module("ai_scientist.apps.validate")

        self.assertIs(legacy, internal)

    def test_validate_help_does_not_require_a_login_session(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.validate")

        with (
            mock.patch.object(sys, "argv", ["xscientist validate", "--help"]),
            mock.patch.object(internal, "require_login") as require_login,
            self.assertRaises(SystemExit) as raised,
        ):
            internal.main()

        self.assertEqual(raised.exception.code, 0)
        require_login.assert_not_called()


if __name__ == "__main__":
    unittest.main()
