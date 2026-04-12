from __future__ import annotations

import re
import unittest
from pathlib import Path


class OpenSourceHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.readme_path = cls.repo_root / "README.md"
        cls.pyproject_path = cls.repo_root / "pyproject.toml"
        cls.gitignore_path = cls.repo_root / ".gitignore"
        cls.workflow_path = cls.repo_root / ".github" / "workflows" / "smoke.yml"
        cls.constraints_path = cls.repo_root / "constraints-ci.txt"
        cls.smoke_requirements_path = cls.repo_root / "requirements-smoke.txt"
        portable_sources = [
            cls.readme_path,
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
        text = self.readme_path.read_text(encoding="utf-8")
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        for link in links:
            if not link or link.startswith(("http://", "https://", "#")):
                continue
            target = (self.readme_path.parent / link).resolve()
            self.assertTrue(target.exists(), msg=f"Broken local README link: {link}")

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
            msg="constraints-ci.txt should exist for reproducible CI installs",
        )
        self.assertTrue(
            self.smoke_requirements_path.exists(),
            msg="requirements-smoke.txt should exist for lightweight CI installs",
        )
        self.assertIn(
            "-c constraints-ci.txt",
            workflow_text,
            msg="Smoke workflow should install dependencies with constraints-ci.txt",
        )
        self.assertIn(
            "-r requirements-smoke.txt",
            workflow_text,
            msg="Smoke workflow should install the dedicated smoke dependency set",
        )

    def test_smoke_workflow_should_upload_ci_artifacts(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        self.assertIn(
            "actions/upload-artifact@v4",
            workflow_text,
            msg="Smoke workflow should upload failure artifacts for diagnosis",
        )
        self.assertIn(
            "smoke-checks-artifacts",
            workflow_text,
            msg="Smoke workflow should use a stable artifact name",
        )

    def test_smoke_workflow_should_run_syntax_checks(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        self.assertIn(
            "python -m compileall -q ai_scientist *.py tests",
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
