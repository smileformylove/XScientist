from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from ai_scientist.apps.project import _prepare_autopilot_bfts_config, main
from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ProjectAdmissionStartupOrderTests(unittest.TestCase):
    @staticmethod
    def _required_repository(root: Path) -> ResearchRepository:
        repository = ResearchRepository.init(
            root,
            question="Can startup recovery precede every model access?",
            policy="manual",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )
        config_path = root / "research.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["experiment_admission"] = {"mode": "required"}
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        policy_checkpoint = repository.commit(
            stage="review",
            subject="require experiment admission for startup recovery",
        )
        if not policy_checkpoint.committed:
            raise AssertionError(
                f"required admission policy commit failed: {policy_checkpoint.reason}"
            )
        if repository.experiment_admission_mode() != "required":
            raise AssertionError("required admission policy is not committed")
        ideas_path = root / "01_ideas" / "ideas.json"
        ideas_path.parent.mkdir(parents=True, exist_ok=True)
        ideas_path.write_text(
            json.dumps([{"Name": "recovery", "Experiment": {"seed": 0}}]),
            encoding="utf-8",
        )
        return repository

    @staticmethod
    def _main_argv(
        repository: ResearchRepository,
        output_root: Path,
        *,
        research_vcs: str,
    ) -> list[str]:
        return [
            str(repository.path),
            "--output-root",
            str(output_root),
            "--resume",
            "--skip-ideation",
            "--skip-experiment",
            "--idea-indices",
            "0",
            "--research-vcs",
            research_vcs,
        ]

    @staticmethod
    def _pending_recovery() -> dict:
        return {
            "active": True,
            "committed_results": [],
            "recovered_results": [],
            "execute_indices": [0],
            "batch_indices": [0],
            "pending_batch": True,
        }

    @staticmethod
    def _resume_scope(*, configuration_hash: str, budget_hash: str) -> dict:
        return {
            "selected_indices": [0],
            "ideas_hash": None,
            "configuration_hash": configuration_hash,
            "budget_hash": budget_hash,
        }

    @staticmethod
    def _write_autopilot_derivative(root: Path) -> tuple[Path, str]:
        path = root / "00_config" / "autopilot_bfts.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "# XScientist finite autopilot derivative\n# profile: balanced\n"
        content += yaml.safe_dump(
            {
                "exec": {
                    "backend": "auto",
                    "require_isolation": True,
                    "network": "none",
                    "allow_experiment_network": False,
                },
                "llm_budget": {
                    "max_total_tokens": 1000,
                    "max_wall_time_seconds": 3600,
                },
                "agent": {"steps": 3, "stages": {}},
            },
            sort_keys=False,
        )
        path.write_text(content, encoding="utf-8")
        return path, content

    def test_existing_required_repository_rejects_research_vcs_off_before_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            repository = self._required_repository(output_root / "research")

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                mock.patch("ai_scientist.apps.project.record_model_provider_decisions"),
                mock.patch(
                    "ai_scientist.apps.project._write_project_pipeline_seed_artifacts",
                    return_value=[],
                ),
                mock.patch(
                    "ai_scientist.apps.project._record_local_research_checkpoint"
                ),
                mock.patch(
                    "ai_scientist.apps.project.select_ranked_idea_candidates",
                    return_value=([0], []),
                ),
                self.assertRaises((RuntimeError, SystemExit)),
            ):
                main(
                    self._main_argv(
                        repository,
                        output_root,
                        research_vcs="off",
                    )
                )

            credentials.assert_not_called()

    def test_deleted_required_policy_rejects_research_vcs_off_before_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            repository = self._required_repository(output_root / "research")
            (repository.path / "research.yaml").unlink()

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                self.assertRaisesRegex(
                    RuntimeError,
                    "requires experiment admission.*--research-vcs off",
                ),
            ):
                main(
                    self._main_argv(
                        repository,
                        output_root,
                        research_vcs="off",
                    )
                )

            credentials.assert_not_called()

    def test_recoverable_required_batch_precedes_models_with_no_progress_cache(
        self,
    ) -> None:
        for progress_state in ("missing", "damaged"):
            with (
                self.subTest(progress_state=progress_state),
                tempfile.TemporaryDirectory() as td,
            ):
                output_root = Path(td)
                repository = self._required_repository(output_root / "research")
                progress_path = repository.path / "04_logs" / "progress.json"
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                if progress_state == "damaged":
                    progress_path.write_text("{damaged", encoding="utf-8")

                recovery_result = {
                    "active": True,
                    "committed_results": [
                        {
                            "idea_idx": 0,
                            "status": "failed",
                            "registry_status": "failed",
                            "research_status": "failed",
                            "stage": "recovered_terminal",
                        }
                    ],
                    "recovered_results": [],
                    "execute_indices": [],
                    "batch_indices": [0],
                }
                with (
                    mock.patch("ai_scientist.apps.project.require_login"),
                    mock.patch(
                        "ai_scientist.apps.project.initialize_runtime",
                        return_value=SimpleNamespace(research_root=output_root),
                    ),
                    mock.patch(
                        "ai_scientist.apps.project.require_model_credentials"
                    ) as credentials,
                    mock.patch(
                        "ai_scientist.apps.project.record_model_provider_decisions"
                    ),
                    mock.patch(
                        "ai_scientist.apps.project._write_project_pipeline_seed_artifacts",
                        return_value=[],
                    ),
                    mock.patch(
                        "ai_scientist.apps.project._record_local_research_planning_objects"
                    ),
                    mock.patch(
                        "ai_scientist.apps.project._record_local_research_checkpoint"
                    ),
                    mock.patch(
                        "ai_scientist.apps.project.select_ranked_idea_candidates",
                        return_value=([0], []),
                    ),
                    mock.patch(
                        "ai_scientist.apps.project."
                        "_reconcile_required_experiment_admission_runtime",
                        return_value=recovery_result,
                    ) as reconcile,
                ):
                    main(
                        self._main_argv(
                            repository,
                            output_root,
                            research_vcs="local",
                        )
                    )

                credentials.assert_not_called()
                reconcile.assert_called_once()
                rebuilt = json.loads(progress_path.read_text(encoding="utf-8"))
                self.assertEqual(rebuilt["selected_indices"], [0])
                self.assertEqual(
                    rebuilt["results"], recovery_result["committed_results"]
                )

    def test_unfinished_required_resume_rejects_explicit_config_before_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            repository = self._required_repository(output_root / "research")
            approved = output_root / "approved.yaml"
            changed = output_root / "changed.yaml"
            approved.write_text("agent:\n  steps: 1\n", encoding="utf-8")
            changed.write_text("agent:\n  steps: 2\n", encoding="utf-8")
            scope = self._resume_scope(
                configuration_hash=canonical_content_hash(
                    approved.read_text(encoding="utf-8")
                ),
                budget_hash=canonical_content_hash({"enabled": False}),
            )
            argv = self._main_argv(
                repository,
                output_root,
                research_vcs="local",
            )
            argv.remove("--skip-experiment")
            argv.extend(["--bfts-config", str(changed)])

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project._required_admission_resume_inputs",
                    return_value=scope,
                ),
                mock.patch(
                    "ai_scientist.apps.project."
                    "_reconcile_required_experiment_admission_runtime",
                    return_value=self._pending_recovery(),
                ),
                mock.patch(
                    "ai_scientist.apps.project._prepare_autopilot_bfts_config"
                ) as prepare_config,
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                self.assertRaisesRegex(RuntimeError, "immutable receipt"),
            ):
                main(argv)

            prepare_config.assert_not_called()
            credentials.assert_not_called()

    def test_unfinished_required_resume_rejects_budget_override_before_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            repository = self._required_repository(output_root / "research")
            config_path, config_text = self._write_autopilot_derivative(repository.path)
            scope = self._resume_scope(
                configuration_hash=canonical_content_hash(config_text),
                budget_hash="sha256:" + "a" * 64,
            )
            argv = self._main_argv(
                repository,
                output_root,
                research_vcs="local",
            )
            argv.remove("--skip-experiment")
            argv.extend(["--autopilot", "balanced", "--max-project-tokens", "500"])

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project._required_admission_resume_inputs",
                    return_value=scope,
                ),
                mock.patch(
                    "ai_scientist.apps.project."
                    "_reconcile_required_experiment_admission_runtime",
                    return_value=self._pending_recovery(),
                ),
                mock.patch(
                    "ai_scientist.apps.project._prepare_autopilot_bfts_config"
                ) as prepare_config,
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                self.assertRaisesRegex(
                    RuntimeError,
                    "budget controls would change the immutable configuration",
                ),
            ):
                main(argv)

            prepare_config.assert_not_called()
            credentials.assert_not_called()
            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)

    def test_unfinished_required_resume_rejects_budget_hash_before_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            repository = self._required_repository(output_root / "research")
            config_path, config_text = self._write_autopilot_derivative(repository.path)
            admitted_budget = {
                "enabled": True,
                "shared_across_project": True,
                "state": "04_logs/llm_budget.json",
                "limits": {"max_total_tokens": 1000},
            }
            changed_budget = {
                **admitted_budget,
                "limits": {"max_total_tokens": 999},
            }
            scope = self._resume_scope(
                configuration_hash=canonical_content_hash(config_text),
                budget_hash=canonical_content_hash(admitted_budget),
            )
            argv = self._main_argv(
                repository,
                output_root,
                research_vcs="local",
            )
            argv.remove("--skip-experiment")
            argv.extend(["--autopilot", "balanced"])

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project._required_admission_resume_inputs",
                    return_value=scope,
                ),
                mock.patch(
                    "ai_scientist.apps.project."
                    "_reconcile_required_experiment_admission_runtime",
                    return_value=self._pending_recovery(),
                ),
                mock.patch(
                    "ai_scientist.utils.data_readiness.prepare_data_contract",
                    return_value={"mode": "synthetic"},
                ),
                mock.patch(
                    "ai_scientist.apps.project._configure_autopilot_project_budget",
                    return_value=changed_budget,
                ),
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                self.assertRaisesRegex(
                    RuntimeError,
                    "budget does not match the immutable receipt",
                ),
            ):
                main(argv)

            credentials.assert_not_called()
            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)

    def test_unfinished_required_resume_reuses_exact_autopilot_derivative(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            repository = self._required_repository(output_root / "research")
            config_path, config_text = self._write_autopilot_derivative(repository.path)
            budget = {
                "enabled": True,
                "shared_across_project": True,
                "state": "04_logs/llm_budget.json",
                "limits": {
                    "max_total_tokens": 1000,
                    "max_wall_time_seconds": 3600,
                },
            }
            scope = self._resume_scope(
                configuration_hash=canonical_content_hash(config_text),
                budget_hash=canonical_content_hash(budget),
            )
            argv = self._main_argv(
                repository,
                output_root,
                research_vcs="local",
            )
            argv.remove("--skip-experiment")
            argv.extend(["--autopilot", "balanced"])

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project._required_admission_resume_inputs",
                    return_value=scope,
                ),
                mock.patch(
                    "ai_scientist.apps.project."
                    "_reconcile_required_experiment_admission_runtime",
                    return_value=self._pending_recovery(),
                ),
                mock.patch(
                    "ai_scientist.utils.data_readiness.prepare_data_contract",
                    return_value={"mode": "synthetic"},
                ),
                mock.patch(
                    "ai_scientist.apps.project._configure_autopilot_project_budget",
                    return_value=budget,
                ) as configure_budget,
                mock.patch(
                    "ai_scientist.apps.project._prepare_autopilot_bfts_config",
                    wraps=_prepare_autopilot_bfts_config,
                ) as prepare_config,
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials",
                    side_effect=RuntimeError("credential boundary reached"),
                ) as credentials,
                self.assertRaisesRegex(RuntimeError, "credential boundary reached"),
            ):
                main(argv)

            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)
            prepare_config.assert_called_once()
            self.assertTrue(prepare_config.call_args.kwargs["reuse_existing"])
            configure_budget.assert_called_once()
            credentials.assert_called_once()

    def test_fully_committed_required_resume_ignores_execution_overrides(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            repository = self._required_repository(output_root / "research")
            scope = self._resume_scope(
                configuration_hash="sha256:" + "b" * 64,
                budget_hash="sha256:" + "c" * 64,
            )
            recovery = {
                "active": True,
                "committed_results": [
                    {
                        "idea_idx": 0,
                        "status": "failed",
                        "registry_status": "failed",
                        "research_status": "failed",
                        "stage": "recovered_terminal",
                    }
                ],
                "recovered_results": [],
                "execute_indices": [],
                "batch_indices": [0],
                "pending_batch": False,
            }
            argv = self._main_argv(
                repository,
                output_root,
                research_vcs="local",
            )
            argv.extend(
                [
                    "--autopilot",
                    "balanced",
                    "--bfts-config",
                    str(output_root / "missing.yaml"),
                    "--max-project-tokens",
                    "1",
                ]
            )

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project._required_admission_resume_inputs",
                    return_value=scope,
                ),
                mock.patch(
                    "ai_scientist.apps.project."
                    "_reconcile_required_experiment_admission_runtime",
                    return_value=recovery,
                ),
                mock.patch(
                    "ai_scientist.apps.project._prepare_autopilot_bfts_config"
                ) as prepare_config,
                mock.patch(
                    "ai_scientist.apps.project._configure_autopilot_project_budget"
                ) as configure_budget,
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                mock.patch(
                    "ai_scientist.apps.project.record_model_provider_decisions"
                ) as provider_decisions,
                mock.patch(
                    "ai_scientist.apps.project._write_project_pipeline_seed_artifacts",
                    return_value=[],
                ),
                mock.patch(
                    "ai_scientist.apps.project._record_local_research_planning_objects"
                ),
                mock.patch(
                    "ai_scientist.apps.project._record_local_research_checkpoint"
                ),
            ):
                main(argv)

            prepare_config.assert_not_called()
            configure_budget.assert_not_called()
            credentials.assert_not_called()
            provider_decisions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
