from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from tools.build_distribution import GENERATED_TARGETS, REPOSITORY_ROOT
from tools.check_distribution import smoke_install


class DistributionBuildTests(unittest.TestCase):
    def test_isolated_smoke_installs_declared_core_dependencies(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"version":"0.1.1","schemas":23}\n',
            stderr="",
        )
        with mock.patch(
            "tools.check_distribution.subprocess.run",
            side_effect=[completed, completed],
        ) as run:
            smoke_install(Path("xscientist-0.1.1-py3-none-any.whl"))

        install_command = run.call_args_list[0].args[0]
        self.assertNotIn("--no-deps", install_command)
        self.assertIn("--target", install_command)

    def test_clean_build_targets_are_narrow_and_generated(self) -> None:
        self.assertEqual(
            {
                path.relative_to(REPOSITORY_ROOT).as_posix()
                for path in GENERATED_TARGETS
            },
            {"build", "compat/xscientist.egg-info"},
        )

    def test_wheel_contains_public_api_entrypoints_and_runtime_resources(self) -> None:
        if shutil.which(sys.executable) is None:
            self.skipTest("python executable unavailable")
        repo_root = Path(__file__).resolve().parents[1]
        self.addCleanup(
            shutil.rmtree,
            repo_root / "compat" / "xscientist.egg-info",
            ignore_errors=True,
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    str(out_dir),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if (
                completed.returncode != 0
                and "No module named build" in completed.stderr
            ):
                self.skipTest("build frontend not installed")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            sdist = next(out_dir.glob("xscientist-*.tar.gz"))
            with tarfile.open(sdist) as archive:
                sdist_names = set(archive.getnames())
                if sys.version_info >= (3, 12):
                    archive.extractall(out_dir, filter="data")
                else:  # Python 3.10/3.11 compatibility lane.
                    archive.extractall(out_dir)
            sdist_root = next(iter(sdist_names)).split("/", 1)[0]
            extracted_root = out_dir / sdist_root
            wheel_build = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(out_dir),
                ],
                cwd=extracted_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(wheel_build.returncode, 0, wheel_build.stderr)
            wheel = next(out_dir.glob("xscientist-*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                metadata_name = next(
                    name for name in names if name.endswith(".dist-info/METADATA")
                )
                metadata = archive.read(metadata_name).decode("utf-8")
                install_root = out_dir / "wheel-install"
                archive.extractall(install_root)

            required = {
                "xscientist/__init__.py",
                "xscientist/cli.py",
                "xscientist/onboarding.py",
                "xscientist/dependency_profiles.py",
                "xscientist/provider_config.py",
                "xscientist/service.py",
                "xscientist/service_jobs.py",
                "xscientist/research_git.py",
                "xscientist/research_cli.py",
                "xscientist/research_vcs.py",
                "xscientist/research_lifecycle.py",
                "xscientist/research_closure.py",
                "xscientist/research_evolution.py",
                "xscientist/evolution_cli.py",
                "xscientist/research_interop.py",
                "xscientist/research_adapters.py",
                "xscientist/research_dag.py",
                "xscientist/research_journey.py",
                "xscientist/research_tools.py",
                "xscientist/git_support.py",
                "xscientist/research_commands.py",
                "xscientist/system_comparison.py",
                "xscientist/evidence_index.py",
                "run_project.py",
                "ai_scientist/apps/project.py",
                "ai_scientist/apps/project_cli.py",
                "continuous_paper_generator.py",
                "ai_scientist/apps/batch.py",
                "ai_scientist/apps/batch_cli.py",
                "ai_scientist/apps/batch_experiment_artifacts.py",
                "research_manager.py",
                "ai_scientist/apps/manager.py",
                "ai_scientist/apps/manager_cli.py",
                "ai_scientist/apps/manager_ranking.py",
                "ai_scientist/apps/manager_reports.py",
                "continuous_research_daemon.py",
                "ai_scientist/apps/daemon.py",
                "ai_scientist/apps/daemon_control.py",
                "ai_scientist/apps/daemon_sources.py",
                "ai_scientist/apps/daemon_dashboard.py",
                "ai_scientist/apps/daemon_reports.py",
                "auth_cli.py",
                "ai_scientist/apps/auth.py",
                "feedback_cli.py",
                "ai_scientist/apps/feedback.py",
                "run_ara_fork.py",
                "ai_scientist/apps/ara.py",
                "validate_repo.py",
                "ai_scientist/apps/validate.py",
                "launch_scientist_bfts.py",
                "ai_scientist/apps/bfts.py",
                "launch_scientist_zhipu.py",
                "ai_scientist/apps/zhipu.py",
                "preflight_check.py",
                "ai_scientist/apps/preflight.py",
                "ai_scientist/resources/configs/bfts_default.yaml",
                "ai_scientist/resources/configs/bfts_glm53.yaml",
                "ai_scientist/blank_icbinb_latex/template.tex",
                "ai_scientist/treesearch/utils/viz_templates/template.html",
                "ai_scientist/protocol/schemas/manifest.schema.json",
                "ai_scientist/protocol/schemas/research_checkpoint.schema.json",
                "ai_scientist/protocol/schemas/research_closure.schema.json",
                "ai_scientist/protocol/schemas/reproduction_receipt.schema.json",
                "ai_scientist/protocol/schemas/arft_coverage.schema.json",
                "ai_scientist/protocol/schemas/autoresearch_conformance.schema.json",
                "ai_scientist/protocol/schemas/first_run_benchmark.schema.json",
                "ai_scientist/protocol/schemas/benchmark_report_verification.schema.json",
                "ai_scientist/protocol/schemas/exploration_audit.schema.json",
                "ai_scientist/protocol/schemas/research_rollout.schema.json",
                "ai_scientist/protocol/schemas/research_rollout_audit.schema.json",
                "ai_scientist/protocol/schemas/belief_context.schema.json",
                "ai_scientist/protocol/schemas/belief_context_audit.schema.json",
                "ai_scientist/protocol/schemas/process_audit.schema.json",
                "ai_scientist/protocol/schemas/evidence_index.schema.json",
                "ai_scientist/protocol/schemas/system_comparison.schema.json",
                "ai_scientist/protocol/schemas/attestation.schema.json",
                "ai_scientist/protocol/schemas/evolution_artifact.schema.json",
                "ai_scientist/protocol/schemas/benchmark_suite.schema.json",
                "ai_scientist/protocol/schemas/benchmark_run.schema.json",
                "ai_scientist/protocol/schemas/canary_suite.schema.json",
                "ai_scientist/protocol/schemas/canary_run.schema.json",
                "ai_scientist/protocol/schemas/deployment_receipt.schema.json",
                "ai_scientist/protocol/schemas/research_interop.schema.json",
                "ai_scientist/protocol/schemas/research_adapter_receipt.schema.json",
                "ai_scientist/protocol/schemas/research_dag.schema.json",
                "ai_scientist/protocol/schemas/tool_evidence.schema.json",
                "ai_scientist/protocol/schemas/insight_report.schema.json",
                "ai_scientist/utils/insight_synthesis.py",
                "ai_scientist/protocol/canonical_json.py",
                "ai_scientist/protocol/attestation.py",
                "ai_scientist/protocol/conformance/canonical-json-v1.json",
                "ai_scientist/protocol/conformance/verify-canonical-json.mjs",
                "ai_scientist/utils/evolution_artifacts.py",
                "ai_scientist/utils/evolution_runtime.py",
                "ai_scientist/utils/evolution_deployment.py",
                "ai_scientist/fewshot_examples/132_automated_relational.txt",
                "ai_scientist/fewshot_examples/132_automated_relational.json",
                "ai_scientist/fewshot_examples/attention.txt",
                "ai_scientist/fewshot_examples/attention.json",
                "ai_scientist/fewshot_examples/2_carpe_diem.txt",
                "ai_scientist/fewshot_examples/2_carpe_diem.json",
            }
            self.assertFalse(
                required - names, f"missing wheel files: {required - names}"
            )
            self.assertIn(
                'Requires-Dist: tomli>=2.0; python_version < "3.11"',
                metadata,
            )
            for extra in (
                "research",
                "openai",
                "anthropic",
                "zhipu",
                "openai-compatible",
                "bedrock",
                "vertex",
                "plot",
                "pdf",
                "pdf-layout",
                "ml",
                "full",
                "service",
                "trust",
                "dev",
            ):
                self.assertIn(f"Provides-Extra: {extra}", metadata)
            for requirement in (
                'Requires-Dist: openai>=1.40; extra == "openai"',
                'Requires-Dist: anthropic>=0.25; extra == "anthropic"',
                'Requires-Dist: zhipuai>=2.0; extra == "zhipu"',
                'Requires-Dist: openai>=1.40; extra == "openai-compatible"',
                'Requires-Dist: anthropic[bedrock]>=0.25; extra == "bedrock"',
                'Requires-Dist: anthropic[vertex]>=0.25; extra == "vertex"',
                'Requires-Dist: pymupdf4llm>=0.0.20; extra == "pdf-layout"',
                'Requires-Dist: transformers>=4.40; extra == "ml"',
                'Requires-Dist: transformers>=4.40; extra == "full"',
            ):
                self.assertIn(requirement, metadata)
            self.assertFalse(
                any(name.startswith("tests/") for name in names),
                "wheel should not include the repository test suite",
            )
            self.assertFalse(
                any(
                    name.startswith("ai_scientist/fewshot_examples/")
                    and name.endswith(".pdf")
                    for name in names
                ),
                "text-backed review few-shots must not ship duplicate PDFs",
            )
            self.assertFalse(
                any(name.endswith(".pyc") or "/__pycache__/" in name for name in names),
                "wheel should never include interpreter cache artifacts",
            )
            source_only = {
                "scripts/daemon/run_daemon_profile.py",
                "scripts/daemon/run_daemon_rehearsal.py",
            }
            self.assertFalse(
                source_only & names,
                f"wheel should not include source-only ops: {source_only & names}",
            )
            missing_source_ops = {
                path
                for path in source_only
                if f"{sdist_root}/{path}" not in sdist_names
            }
            self.assertFalse(
                missing_source_ops,
                f"sdist is missing source-only ops: {missing_source_ops}",
            )
            self.assertFalse(
                any(
                    name.startswith(("tools/", "docs/", "examples/")) for name in names
                ),
                "wheel should not include source-only maintenance assets",
            )
            self.assertIn(
                f"{sdist_root}/tools/repository_validation.py",
                sdist_names,
                "sdist should retain repository-only validation",
            )
            for source_asset in (
                "CHANGELOG.md",
                "CITATION.cff",
                ".github/CODE_OF_CONDUCT.md",
                ".github/CONTRIBUTING.md",
                ".github/PULL_REQUEST_TEMPLATE.md",
                ".github/SECURITY.md",
                "configs/bfts/bfts_config.yaml",
                "configs/bfts/bfts_config_deep.yaml",
                "configs/bfts/bfts_config_enhanced.yaml",
                "configs/environment/example.env",
                "docs/CONFIG_REFERENCE.md",
                "docs/ARCHITECTURE.md",
                "docs/ENGINEERING.md",
                "docs/OPERATIONS_CHECKLIST.md",
                "docs/README.zh.md",
                "examples/example_topic.md",
                "requirements/constraints-ci.txt",
                "requirements/smoke.txt",
            ):
                self.assertIn(
                    f"{sdist_root}/{source_asset}",
                    sdist_names,
                    f"sdist should retain source operation asset: {source_asset}",
                )

            legacy_import_smoke = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import auth_cli, ai_scientist.apps.batch, "
                        "ai_scientist.apps.daemon, feedback_cli, "
                        "launch_scientist_bfts, launch_scientist_zhipu, "
                        "preflight_check, ai_scientist.apps.manager, run_ara_fork, "
                        "run_project, validate_repo; "
                        "import ai_scientist.apps.project as project; "
                        "from ai_scientist.resources import ("
                        "bfts_config_path, resolve_bfts_config_path); "
                        "assert run_project is project; "
                        "assert resolve_bfts_config_path('bfts_config.yaml') "
                        "== bfts_config_path('default')"
                    ),
                ],
                cwd=out_dir,
                env={
                    **os.environ,
                    "PYTHONPATH": str(install_root),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                legacy_import_smoke.returncode, 0, legacy_import_smoke.stderr
            )
            for legacy_module in ("run_project", "ai_scientist.apps.batch"):
                legacy_help = subprocess.run(
                    [sys.executable, "-m", legacy_module, "--help"],
                    cwd=out_dir,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(install_root),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(legacy_help.returncode, 0, legacy_help.stderr)
                self.assertIn("usage:", legacy_help.stdout.lower())


if __name__ == "__main__":
    unittest.main()
