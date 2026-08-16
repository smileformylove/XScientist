from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class AppCompatibilityTests(unittest.TestCase):
    def test_legacy_mock_patch_targets_internal_globals(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.project")

        with mock.patch("ai_scientist.apps.project.idea_to_markdown") as patched:
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

    def test_legacy_batch_mock_patch_targets_internal_globals(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.batch")

        with mock.patch(
            "ai_scientist.apps.batch.record_quality_fallback_if_needed"
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

    def test_legacy_daemon_mock_patch_targets_internal_globals(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.daemon")

        with mock.patch("ai_scientist.apps.daemon.ResearchManager") as patched:
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

    def test_validate_help_does_not_require_a_login_session(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.validate")

        with (
            mock.patch.object(sys, "argv", ["xscientist validate", "--help"]),
            self.assertRaises(SystemExit) as raised,
        ):
            internal.main()

        self.assertEqual(raised.exception.code, 0)
        self.assertFalse(hasattr(internal, "require_login"))

    def test_installed_validation_writes_bytecode_outside_package(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.validate")
        package_file = Path("/read-only/site-packages/xscientist/client.py")

        with (
            mock.patch.object(
                internal, "iter_installed_python_files", return_value=[package_file]
            ),
            mock.patch.object(internal.py_compile, "compile") as compile_file,
        ):
            internal.run_installed_py_compile()

        compile_file.assert_called_once()
        self.assertNotEqual(
            Path(compile_file.call_args.kwargs["cfile"]).parent,
            package_file.parent,
        )
        self.assertTrue(compile_file.call_args.kwargs["doraise"])

    def test_installed_validation_discovers_packages_under_site_packages(self) -> None:
        internal = importlib.import_module("ai_scientist.apps.validate")

        with tempfile.TemporaryDirectory() as td:
            package_root = Path(td) / "site-packages"
            source_file = package_root / "xscientist" / "client.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("VALUE = 1\n", encoding="utf-8")
            with mock.patch.object(internal, "PROJECT_ROOT", package_root):
                files = internal.iter_installed_python_files()

        self.assertIn(source_file, files)

    def test_low_level_launcher_help_does_not_require_login(self) -> None:
        for module_name in ("ai_scientist.apps.bfts", "ai_scientist.apps.zhipu"):
            with self.subTest(module=module_name):
                internal = importlib.import_module(module_name)
                with (
                    mock.patch.object(sys, "argv", [module_name, "--help"]),
                    mock.patch.object(internal, "require_login") as require_login,
                    self.assertRaises(SystemExit) as raised,
                ):
                    internal.main()

                self.assertEqual(raised.exception.code, 0)
                require_login.assert_not_called()


if __name__ == "__main__":
    unittest.main()
