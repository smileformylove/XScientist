from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_release import validate_release
from tools.engineering_checks import (
    REPOSITORY_ROOT,
    _requirement_names,
    check_action_pinning,
    check_markdown_links,
    run_checks,
)


class EngineeringChecksTests(unittest.TestCase):
    def test_requirement_parser_handles_extras_markers_and_directives(self) -> None:
        names = _requirement_names(
            [
                "coverage[toml]>=7.6",
                "uvicorn[standard]>=0.30; python_version >= '3.10'",
                "-c constraints.txt",
                "# comment",
                "PyYAML>=6 # inline comment",
            ]
        )
        self.assertEqual(names, {"coverage", "uvicorn", "pyyaml"})

    def test_markdown_link_check_reports_missing_local_target(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "docs").mkdir()
            (root / "README.md").write_text("[missing](docs/nope.md)\n")
            (root / "docs" / "README.zh.md").write_text(
                "[remote](https://example.com)\n", encoding="utf-8"
            )
            self.assertEqual(
                check_markdown_links(root),
                ["broken local link in README.md: docs/nope.md"],
            )

    def test_current_repository_passes_engineering_checks(self) -> None:
        errors = {
            name: values
            for name, values in run_checks(REPOSITORY_ROOT).items()
            if values
        }
        self.assertEqual(errors, {})

    def test_release_workflow_requires_history_main_ancestry_and_full_tests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (workflows / "release.yml").write_text(
                "publish: ${{ startsWith(github.ref, 'refs/tags/v') }}\n",
                encoding="utf-8",
            )

            errors = check_action_pinning(root)

        self.assertIn("release checkout must fetch complete Git history", errors)
        self.assertIn(
            "release tags must be verified as reachable from origin/main", errors
        )
        self.assertIn(
            "release build must install test dependencies and run full pytest", errors
        )
        self.assertIn(
            "release tests must install and import the HTTP service dependencies",
            errors,
        )

    def test_release_tag_is_bound_to_package_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "xscientist").mkdir()
            (root / "xscientist" / "_version.py").write_text(
                '__version__ = "0.2.0"\n', encoding="utf-8"
            )
            (root / "CITATION.cff").write_text(
                "cff-version: 1.2.0\nversion: 0.2.0\n", encoding="utf-8"
            )
            (root / "CHANGELOG.md").write_text(
                "# Changelog\n\n## [Unreleased]\n\n"
                "## [0.2.0] - 2026-08-06\n\n- Ready.\n",
                encoding="utf-8",
            )
            self.assertEqual(validate_release("v0.2.0", root), [])
            errors = validate_release("v9.9.9", root)
            self.assertTrue(
                any("release tag must be v0.2.0" in item for item in errors)
            )

    def test_release_rejects_unreleased_changelog_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "xscientist").mkdir()
            (root / "xscientist" / "_version.py").write_text(
                '__version__ = "0.2.0"\n', encoding="utf-8"
            )
            (root / "CITATION.cff").write_text(
                "cff-version: 1.2.0\nversion: 0.2.0\n", encoding="utf-8"
            )
            (root / "CHANGELOG.md").write_text(
                "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- Pending.\n\n"
                "## [0.2.0] - 2026-08-06\n\n- Ready.\n",
                encoding="utf-8",
            )
            errors = validate_release("v0.2.0", root)
            self.assertTrue(any("Unreleased entries" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
