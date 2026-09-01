from __future__ import annotations

import contextlib
import io
import json
import re
import unittest
from pathlib import Path
from unittest import mock

from xscientist._version import PUBLISHED_VERSION
from xscientist.cli import main as cli_main

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


class OpenSourceHygieneTests(unittest.TestCase):
    def test_executor_json_errors_redact_host_paths(self) -> None:
        stderr = io.StringIO()
        private_path = "/" + "Users" + "/alice/private-lab/token.txt"
        with (
            mock.patch(
                "xscientist.executor_manager.inspect_executor",
                side_effect=OSError(f"cannot read {private_path}"),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = cli_main(["executor", "check", "--workspace", ".", "--json"])

        payload = json.loads(stderr.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["ok"])
        self.assertNotIn(private_path, stderr.getvalue())
        self.assertIn("[REDACTED_PATH]", payload["error"])

    def test_bilingual_docs_define_structured_trajectory_as_git_substrate(
        self,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        english = (repo_root / "README.md").read_text(encoding="utf-8")
        chinese = (repo_root / "docs" / "README.zh.md").read_text(encoding="utf-8")
        protocol = (repo_root / "ai_scientist" / "protocol" / "SPEC.md").read_text(
            encoding="utf-8"
        )

        self.assertRegex(english, r"structured\s+research trajectory")
        self.assertIn("结构化科研轨迹", chinese)
        self.assertIn("Structured Trajectory Invariant", protocol)
        self.assertIn("chain-of-thought", english)
        self.assertIn("隐藏思维链", chinese)
        self.assertIn("chain-of-thought", protocol)

    def test_third_party_code_lineage_is_disclosed_and_machine_readable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        notice = (repo_root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        provenance = json.loads(
            (repo_root / "provenance" / "upstream_sources.json").read_text(
                encoding="utf-8"
            )
        )
        aide_license = (
            repo_root / "third_party" / "licenses" / "AIDE-MIT.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("AI-Scientist-v2", notice)
        self.assertIn("AIDE", notice)
        self.assertIn("maintainer_confirmation_required", json.dumps(provenance))
        self.assertEqual(provenance["schema"], "xscientist.upstream-provenance.v1")
        self.assertIn("Copyright (c) 2024 Weco AI Ltd", aide_license)
        self.assertTrue(
            provenance["sources"][0]["distribution_remediation"][
                "removed_upstream_full_paper_review_examples"
            ]
        )
        self.assertIn("fictional XScientist-authored", notice)

    def test_distribution_license_and_authored_asset_boundary(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (repo_root / "pyproject.toml").open("rb") as handle:
            metadata = tomllib.load(handle)

        self.assertEqual(metadata["project"]["license"], "Apache-2.0 AND MIT")
        self.assertIn(
            "setuptools>=77",
            metadata["build-system"]["requires"],
        )

        retained_templates = {
            repo_root / "ai_scientist" / "blank_icbinb_latex" / "template.tex",
            repo_root / "ai_scientist" / "blank_icml_latex" / "template.tex",
        }
        for template in retained_templates:
            self.assertIn(
                "XScientist-authored generic manuscript seed",
                template.read_text(encoding="utf-8"),
            )

        risky_assets = (
            "natbib.sty",
            "fancyhdr.sty",
            "algorithm.sty",
            "algorithmic.sty",
            "iclr2025.sty",
            "iclr2025.bst",
            "icml2025.sty",
            "icml2025.bst",
        )
        for name in risky_assets:
            self.assertFalse(
                any((repo_root / "ai_scientist").glob(f"blank_*_latex/{name}"))
            )

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
            "THIRD_PARTY_NOTICES.md",
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
        install_spec = f"xscientist=={PUBLISHED_VERSION}"
        self.assertIn(install_spec, english)
        self.assertIn(f"Published `{PUBLISHED_VERSION}`", english)
        self.assertIn(install_spec, chinese)
        self.assertIn(f"正式版 `{PUBLISHED_VERSION}`", chinese)

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
            '<a href="#quick-start">快速开始</a>': (
                "## 从自己的想法开始：不需要 API Key"
            ),
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

    def test_smoke_executes_pytest_native_tests(self) -> None:
        workflow_text = self.workflow_path.read_text(encoding="utf-8")
        smoke_requirements = self.smoke_requirements_path.read_text(encoding="utf-8")

        self.assertRegex(
            smoke_requirements,
            r"(?m)^pytest(?:\s|$)",
            msg="Smoke dependencies must include the runner imported by tests",
        )
        self.assertIn(
            "coverage run -m pytest",
            workflow_text,
            msg="Full Smoke must execute pytest-native functions and fixtures",
        )
        self.assertNotIn(
            "coverage run -m unittest discover",
            workflow_text,
            msg="unittest discovery silently omits pytest-native tests",
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
