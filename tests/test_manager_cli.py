from __future__ import annotations

import unittest
from unittest import mock

from ai_scientist.apps import manager


class ManagerCliTests(unittest.TestCase):
    def test_main_accepts_argv_and_injects_current_manager_dependencies(self) -> None:
        fake_manager = mock.Mock()
        fake_manager.list_papers.return_value = []

        with (
            mock.patch.object(
                manager, "ResearchManager", return_value=fake_manager
            ) as manager_cls,
            mock.patch.object(manager, "require_login") as require_login,
            mock.patch("builtins.print"),
        ):
            result = manager.main(["list-papers"])

        self.assertIsNone(result)
        manager_cls.assert_called_once()
        require_login.assert_called_once_with("研究管理操作(research_manager)")
        fake_manager.list_papers.assert_called_once_with(None, "modified")

    def test_help_exits_before_login_or_manager_construction(self) -> None:
        with (
            mock.patch.object(manager, "ResearchManager") as manager_cls,
            mock.patch.object(manager, "require_login") as require_login,
            self.assertRaises(SystemExit) as raised,
        ):
            manager.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        manager_cls.assert_not_called()
        require_login.assert_not_called()


if __name__ == "__main__":
    unittest.main()
