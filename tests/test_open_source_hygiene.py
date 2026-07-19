from __future__ import annotations

import re
import unittest
from pathlib import Path


class OpenSourceHygieneTests(unittest.TestCase):
    def test_repository_root_visible_files_are_intentional(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected = {
            "LICENSE",
            "MANIFEST.in",
            "Makefile",
            "README.md",
            "pyproject.toml",
            "requirements.txt",
            "run_stable_daemon.sh",
            "start_research.sh",
        }
        actual = {
            path.name
            for path in repo_root.iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        self.assertEqual(actual, expected)

    def test_repository_root_has_no_python_modules(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root_modules = sorted(path.name for path in repo_root.glob("*.py"))
        self.assertEqual(
            root_modules,
            [],
            f"root-level Python modules should live under packages or compat/: {root_modules}",
        )

    def test_repository_root_has_no_runtime_yaml_configs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        root_configs = sorted(
            path.name
            for pattern in ("*.yaml", "*.yml")
            for path in repo_root.glob(pattern)
        )
        self.assertEqual(
            root_configs,
            [],
            f"runtime YAML configs should live under configs/: {root_configs}",
        )

    def test_ci_dependency_files_live_under_requirements(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "constraints-ci.txt").exists())
        self.assertFalse((repo_root / "requirements-smoke.txt").exists())
        self.assertTrue((repo_root / "requirements" / "constraints-ci.txt").is_file())
        self.assertTrue((repo_root / "requirements" / "smoke.txt").is_file())

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.readme_path = cls.repo_root / "README.md"
        cls.chinese_readme_path = cls.repo_root / "docs" / "README.zh.md"
        cls.pyproject_path = cls.repo_root / "pyproject.toml"
        cls.gitignore_path = cls.repo_root / ".gitignore"
        cls.workflow_path = cls.repo_root / ".github" / "workflows" / "smoke.yml"
        cls.release_workflow_path = (
            cls.repo_root / ".github" / "workflows" / "release.yml"
        )
        cls.constraints_path = cls.repo_root / "requirements" / "constraints-ci.txt"
        cls.smoke_requirements_path = cls.repo_root / "requirements" / "smoke.txt"
        portable_sources = [
            cls.readme_path,
            *sorted((cls.repo_root / ".github").glob("*.md")),
            *sorted((cls.repo_root / "docs").glob("**/*.md")),
            *sorted((cls.repo_root / "configs").glob("**/*.example.json")),
            *sorted((cls.repo_root / "configs").glob("**/*.example.yaml")),
            *sorted((cls.repo_root / "configs").glob("**/*.example.yml")),
            *sorted((cls.repo_root / "configs").glob("**/*.example.toml")),
        ]
        cls.path_sensitive_sources = [
            path for path in dict.fromkeys(portable_sources) if path.exists()
        ]

    def test_readme_should_not_include_local_absolute_paths(self) -> None:
        text = self.readme_path.read_text(encoding="utf-8")
        forbidden_prefixes = ["/Users/", "C:\\\\", "file://"]
        for prefix in forbidden_prefixes:
            self.assertNotIn(
                prefix,
                text,
                msg=f"README should avoid local absolute paths: found {prefix}",
            )

    def test_readme_local_markdown_links_exist(self) -> None:
        for source in (self.readme_path, self.chinese_readme_path):
            text = source.read_text(encoding="utf-8")
            links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
            for link in links:
                if not link or link.startswith(("http://", "https://", "#")):
                    continue
                target = (source.parent / link).resolve()
                self.assertTrue(
                    target.exists(),
                    msg=f"Broken local link in {source.relative_to(self.repo_root)}: {link}",
                )

    def test_community_documents_use_standard_locations(self) -> None:
        expected = {
            self.repo_root / ".github" / "CONTRIBUTING.md",
            self.repo_root / ".github" / "CODE_OF_CONDUCT.md",
            self.repo_root / ".github" / "SECURITY.md",
            self.repo_root / "docs" / "ARCHITECTURE.md",
            self.repo_root / "docs" / "README.zh.md",
        }
        self.assertTrue(all(path.is_file() for path in expected))
        for old_name in (
            "ARCHITECTURE.md",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "README.en.md",
            "README.zh.md",
            "SECURITY.md",
        ):
            self.assertFalse((self.repo_root / old_name).exists())

    def test_smoke_workflow_seeds_ci_auth_session(self) -> None:
        text = self.workflow_path.read_text(encoding="utf-8")
        self.assertIn(
            "AI_SCIENTIST_AUTH_FILE",
            text,
            msg="Smoke workflow should define AI_SCIENTIST_AUTH_FILE env",
        )
        self.assertIn(
            "Seed login session for CI",
            text,
            msg="Smoke workflow should seed auth session for guarded entrypoints",
        )

    def test_ci_constraints_file_exists_and_is_used(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        self.assertTrue(
            self.constraints_path.exists(),
            msg="requirements/constraints-ci.txt should exist for reproducible CI installs",
        )
        self.assertTrue(
            self.smoke_requirements_path.exists(),
            msg="requirements/smoke.txt should exist for lightweight CI installs",
        )
        self.assertIn(
            "-c requirements/constraints-ci.txt",
            workflow_text,
            msg="Smoke workflow should use requirements/constraints-ci.txt",
        )
        self.assertIn(
            "-r requirements/smoke.txt",
            workflow_text,
            msg="Smoke workflow should install the dedicated smoke dependency set",
        )

    def test_smoke_workflow_should_upload_ci_artifacts(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        self.assertIn(
            "actions/upload-artifact@v7",
            workflow_text,
            msg="Smoke workflow should upload failure artifacts for diagnosis",
        )
        self.assertIn(
            "smoke-checks-artifacts",
            workflow_text,
            msg="Smoke workflow should use a stable artifact name",
        )
        self.assertIn(
            "include-hidden-files: true",
            workflow_text,
            msg="Smoke workflow should upload the hidden .ci-output directory",
        )

    def test_workflows_should_use_node24_actions(self) -> None:
        smoke_text = self.workflow_path.read_text(encoding="utf-8")
        release_text = self.release_workflow_path.read_text(encoding="utf-8")
        for expected in (
            "actions/checkout@v7",
            "actions/setup-python@v6",
            "actions/upload-artifact@v7",
        ):
            self.assertIn(expected, smoke_text)
            self.assertIn(expected, release_text)
        self.assertIn("actions/download-artifact@v8", release_text)

    def test_smoke_workflow_should_run_syntax_checks(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        self.assertIn(
            "python -m compileall -q ai_scientist xscientist compat scripts tools tests",
            workflow_text,
            msg="Smoke workflow should compile Python sources for syntax regressions",
        )
        self.assertIn(
            "bash -n run_stable_daemon.sh",
            workflow_text,
            msg="Smoke workflow should validate run_stable_daemon.sh syntax",
        )
        self.assertIn(
            "bash -n start_research.sh",
            workflow_text,
            msg="Smoke workflow should validate start_research.sh syntax",
        )

    def test_pyproject_should_expose_public_package_and_entrypoints(self) -> None:
        text = self.pyproject_path.read_text(encoding="utf-8")
        self.assertIn('name = "xscientist"', text)
        self.assertIn("[project.scripts]", text)
        self.assertIn('xscientist = "xscientist.cli:main"', text)
        self.assertIn("[project.optional-dependencies]", text)
        self.assertIn("service = [", text)

    def test_release_workflow_should_use_pypi_trusted_publishing(self) -> None:
        text = self.release_workflow_path.read_text(encoding="utf-8")
        self.assertIn("pypa/gh-action-pypi-publish@release/v1", text)
        self.assertIn("id-token: write", text)
        self.assertIn("python -m build --sdist --wheel", text)

    def test_pyproject_should_define_python_floor_and_black_config(self) -> None:
        text = self.pyproject_path.read_text(encoding="utf-8")
        self.assertIn(
            'requires-python = ">=3.10"',
            text,
            msg="pyproject.toml should declare the supported Python floor",
        )
        self.assertIn(
            "[tool.black]",
            text,
            msg="pyproject.toml should centralize Black configuration",
        )
        self.assertIn(
            'target-version = ["py310"]',
            text,
            msg="Black config should align with the repository Python floor",
        )

    def test_gitignore_should_ignore_ci_output_directory(self) -> None:
        text = self.gitignore_path.read_text(encoding="utf-8")
        self.assertIn(
            ".ci-output/",
            text,
            msg=".gitignore should ignore local CI artifact directories",
        )

    def test_portable_sources_should_not_ship_local_machine_paths(self) -> None:
        forbidden_prefixes = ["/Users/", "C:\\\\", "file://"]
        for path in self.path_sensitive_sources:
            text = path.read_text(encoding="utf-8")
            for prefix in forbidden_prefixes:
                self.assertNotIn(
                    prefix,
                    text,
                    msg=f"{path.relative_to(self.repo_root)} should avoid local absolute paths: found {prefix}",
                )


if __name__ == "__main__":
    unittest.main()
