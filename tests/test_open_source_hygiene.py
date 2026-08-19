from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


class OpenSourceHygieneTests(unittest.TestCase):
    def test_repository_root_visible_files_are_intentional(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        expected = {
            "CHANGELOG.md",
            "CITATION.cff",
            "LICENSE",
            "MANIFEST.in",
            "Makefile",
            "mkdocs.yml",
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
            if path.name != "mkdocs.yml"
        )
        self.assertEqual(
            root_configs,
            [],
            f"runtime YAML configs should live under configs/: {root_configs}",
        )
        self.assertTrue((repo_root / "mkdocs.yml").is_file())

    def test_environment_template_lives_under_configs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / ".env.example").exists())
        self.assertTrue(
            (repo_root / "configs" / "environment" / "example.env").is_file()
        )

    def test_root_protocol_files_stay_discoverable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for name in (".gitignore", "MANIFEST.in", "pyproject.toml"):
            self.assertTrue(
                (repo_root / name).is_file(),
                msg=f"{name} must remain at repository root for tool discovery",
            )

    def test_ci_dependency_files_live_under_requirements(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.assertFalse((repo_root / "constraints-ci.txt").exists())
        self.assertFalse((repo_root / "requirements-smoke.txt").exists())
        self.assertTrue((repo_root / "requirements" / "constraints-ci.txt").is_file())
        self.assertTrue((repo_root / "requirements" / "smoke.txt").is_file())

    def test_dependency_manifests_avoid_unused_and_transitive_redundancies(
        self,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (repo_root / "pyproject.toml").open("rb") as handle:
            metadata = tomllib.load(handle)
        full_dependencies = "\n".join(
            metadata["project"]["optional-dependencies"]["full"]
        ).lower()
        root_requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
        smoke_requirements = (repo_root / "requirements" / "smoke.txt").read_text(
            encoding="utf-8"
        )

        for package in ("funcy", "wandb"):
            self.assertNotIn(package, full_dependencies)
            self.assertNotRegex(
                root_requirements,
                rf"(?m)^\s*{re.escape(package)}\s*(?:#.*)?$",
            )
            self.assertNotRegex(
                smoke_requirements,
                rf"(?m)^\s*{re.escape(package)}\s*(?:#.*)?$",
            )
        self.assertNotRegex(
            full_dependencies,
            r"(?m)^\s*botocore(?:[<>=!~].*)?$",
            msg="boto3 already installs botocore; keep only its CI constraint",
        )

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

    def test_readmes_describe_current_installation_status(self) -> None:
        for source in (self.readme_path, self.chinese_readme_path):
            text = source.read_text(encoding="utf-8")
            self.assertIn(
                "git+https://github.com/smileformylove/XScientist.git@main",
                text,
            )
            self.assertIn("PyPI", text)
        english = self.readme_path.read_text(encoding="utf-8")
        chinese = self.chinese_readme_path.read_text(encoding="utf-8")
        self.assertIn("xscientist==0.1.2", english)
        self.assertIn("`0.1.3` release candidate", english)
        self.assertIn("xscientist==0.1.2", chinese)
        self.assertIn("`0.1.3` 候选版", chinese)

    def test_readmes_use_public_workflow_commands(self) -> None:
        forbidden = (
            "run_ara_fork.py",
            "run_project.py",
            "continuous_paper_generator.py",
            "continuous_research_daemon.py",
            "research_manager.py",
        )
        for source in (self.readme_path, self.chinese_readme_path):
            text = source.read_text(encoding="utf-8")
            for legacy_name in forbidden:
                self.assertNotIn(
                    legacy_name,
                    text,
                    msg=f"{source.name} should recommend public xscientist commands",
                )

    def test_chinese_readme_toc_tracks_quick_start_sections(self) -> None:
        text = self.chinese_readme_path.read_text(encoding="utf-8")
        expected_navigation = {
            '<a href="#不配置模型先体验">快速体验</a>': "## 不配置模型先体验",
            '<a href="#运行自主研究">自主研究</a>': "## 运行自主研究",
            '<a href="#检查审计与复现">审计复现</a>': "## 检查、审计与复现",
            '<a href="#安装方式">安装</a>': "## 安装方式",
        }
        for navigation, heading in expected_navigation.items():
            self.assertIn(navigation, text)
            self.assertIn(heading, text)

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
        self.assertRegex(
            workflow_text,
            r"actions/upload-artifact@[0-9a-f]{40}\s+# v7\.",
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

    def test_workflows_pin_actions_to_immutable_commits(self) -> None:
        for path in (self.workflow_path, self.release_workflow_path):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "uses:" not in line:
                    continue
                self.assertRegex(
                    line,
                    r"uses:\s*[^\s@]+@[0-9a-f]{40}(?:\s+#.*)?$",
                    msg=f"{path.name}:{line_number} must pin an immutable action commit",
                )

    def test_smoke_workflow_should_run_syntax_checks(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        self.assertIn(
            "make PYTHON=python syntax",
            workflow_text,
            msg="Smoke workflow should use the canonical syntax target",
        )

    def test_smoke_workflow_has_compatibility_and_coverage_gates(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        for version in ("3.10", "3.11", "3.12"):
            self.assertIn(version, workflow_text)
        for runner in ("ubuntu-latest", "macos-latest", "windows-latest"):
            self.assertIn(runner, workflow_text)
        self.assertIn("python -m coverage run", workflow_text)
        self.assertIn("tools/engineering_checks.py", workflow_text)
        self.assertIn(
            "repository-privacy",
            (self.repo_root / "tools" / "engineering_checks.py").read_text(
                encoding="utf-8"
            ),
        )
        self.assertIn("tools/check_distribution.py", workflow_text)

    def test_pyproject_should_expose_public_package_and_entrypoints(self) -> None:
        text = self.pyproject_path.read_text(encoding="utf-8")
        self.assertIn('name = "xscientist"', text)
        self.assertIn("[project.scripts]", text)
        self.assertIn('xscientist = "xscientist.cli:main"', text)
        self.assertIn("[project.optional-dependencies]", text)
        self.assertIn("service = [", text)

    def test_release_workflow_should_use_pypi_trusted_publishing(self) -> None:
        text = self.release_workflow_path.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r"pypa/gh-action-pypi-publish@[0-9a-f]{40}\s+# release/v1",
        )
        self.assertIn("id-token: write", text)
        self.assertIn("tools/build_distribution.py", text)
        self.assertIn("tools/check_distribution.py", text)
        self.assertIn("tools/check_release.py", text)
        self.assertIn("startsWith(github.ref, 'refs/tags/v')", text)

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
