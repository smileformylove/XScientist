from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_scientist.apps.project import (
    _configure_autopilot_project_budget,
    _completed_resume_results,
    _prepare_project_input,
    _prepare_autopilot_bfts_config,
    _publication_gate_failure,
    _resolve_resume_work,
    _run_autopilot_preflight,
    _save_project_progress,
    _upsert_project_results,
    create_project_structure,
    main,
    run_parallel_experiments,
    save_project_summary,
)
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.data_readiness import prepare_data_contract
from ai_scientist.utils.llm_budget import llm_budget_manager


class ProjectAutopilotTests(unittest.TestCase):
    _VALID_PDF = b"%PDF-1.4\n%%EOF\n"

    @staticmethod
    def _write_project_ideas(
        project_root: Path,
        *,
        count: int = 1,
        ideas: list[dict[str, object]] | None = None,
    ) -> Path:
        ideas_path = project_root / "01_ideas" / "ideas.json"
        ideas_path.parent.mkdir(parents=True, exist_ok=True)
        payload = ideas or [
            {"Name": f"idea-{index}", "Experiment": {"seed": index}}
            for index in range(count)
        ]
        ideas_path.write_text(json.dumps(payload), encoding="utf-8")
        return ideas_path

    @staticmethod
    def _project_ideas_hash(project_root: Path) -> str:
        payload = json.loads(
            (project_root / "01_ideas" / "ideas.json").read_text(encoding="utf-8")
        )
        return canonical_content_hash(payload)

    def tearDown(self) -> None:
        # Autopilot intentionally binds the process-wide budget manager for the
        # duration of a project run.  Tests use temporary project directories,
        # so they must not leave that singleton pointing at a deleted ledger.
        llm_budget_manager.configure(max_total_tokens=None, reset=True)
        llm_budget_manager.export_environment()

    def test_data_gate_hashes_empirical_inputs_without_disclosing_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            data = Path(td) / "private-data"
            (root / "00_config").mkdir(parents=True)
            data.mkdir()
            (data / "observations.csv").write_text("x,y\n1,2\n", encoding="utf-8")

            with mock.patch.dict("os.environ", {}, clear=False):
                contract = prepare_data_contract(root, data_dir=data, required=True)
                snapshot = Path(os.environ["AI_SCIENTIST_PROJECT_DATA_DIR"])

            self.assertTrue(contract["ready"])
            self.assertEqual(contract["mode"], "content_addressed_snapshot_read_only")
            self.assertEqual(contract["file_count"], 1)
            self.assertTrue(contract["files"][0]["sha256"].startswith("sha256:"))
            self.assertEqual(snapshot.parent.name, "datasets")
            (data / "observations.csv").write_text("x,y\n9,9\n", encoding="utf-8")
            self.assertEqual(
                (snapshot / "observations.csv").read_text(encoding="utf-8"),
                "x,y\n1,2\n",
            )
            serialized = (root / "00_config" / "data_manifest.json").read_text()
            self.assertNotIn(str(data), serialized)

    def test_data_gate_requires_explicit_synthetic_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "00_config").mkdir()
            with self.assertRaisesRegex(RuntimeError, "before model calls"):
                prepare_data_contract(root, required=True)

    def test_plain_language_question_materializes_reproducible_topic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dirs = {
                "root": root,
                "ideas": root / "01_ideas",
                "experiments": root / "02_experiments",
                "papers": root / "03_papers",
                "logs": root / "04_logs",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
            args = argparse.Namespace(
                question="Why does a promising mechanism fail out of distribution?",
                topic=None,
                ideas=None,
                resume=False,
                skip_ideation=False,
            )

            _prepare_project_input(args, dirs)

            topic = Path(args.topic)
            self.assertEqual(topic, root / "00_config" / "topic.md")
            self.assertIn("fail out of distribution", topic.read_text(encoding="utf-8"))

    def test_resume_requires_one_canonical_persisted_ideas_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            dirs = {
                "root": root,
                "ideas": root / "01_ideas",
                "experiments": root / "02_experiments",
                "papers": root / "03_papers",
                "logs": root / "04_logs",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
            generated = dirs["ideas"] / "generated_ideas.json"
            stable = dirs["ideas"] / "ideas.json"
            generated.write_text(
                json.dumps([{"Name": "old", "Experiment": "generated"}]),
                encoding="utf-8",
            )
            stable.write_text(
                json.dumps([{"Name": "new", "Experiment": "stable"}]),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                question=None,
                topic=None,
                ideas=None,
                resume=True,
                skip_ideation=False,
            )

            with self.assertRaisesRegex(ValueError, "conflicting"):
                _prepare_project_input(args, dirs)

            stable.write_text(generated.read_text(encoding="utf-8"), encoding="utf-8")
            _prepare_project_input(args, dirs)
            self.assertTrue(args.skip_ideation)

    def test_resume_allows_matching_explicit_inputs_and_rejects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "project"
            dirs = {
                "root": root,
                "ideas": root / "01_ideas",
                "experiments": root / "02_experiments",
                "papers": root / "03_papers",
                "logs": root / "04_logs",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
            persisted = [{"Name": "same", "Experiment": {"seed": 7}}]
            (dirs["ideas"] / "ideas.json").write_text(
                json.dumps(persisted, sort_keys=True), encoding="utf-8"
            )
            same_ideas = base / "same.json"
            same_ideas.write_text(
                '[{"Experiment":{"seed":7},"Name":"same"}]', encoding="utf-8"
            )
            changed_ideas = base / "changed.json"
            changed_ideas.write_text(
                json.dumps([{"Name": "changed", "Experiment": {"seed": 7}}]),
                encoding="utf-8",
            )

            matching_args = argparse.Namespace(
                question=None,
                topic=None,
                ideas=str(same_ideas),
                resume=True,
                skip_ideation=False,
            )
            _prepare_project_input(matching_args, dirs)
            self.assertTrue(matching_args.skip_ideation)

            changed_args = argparse.Namespace(
                question=None,
                topic=None,
                ideas=str(changed_ideas),
                resume=True,
                skip_ideation=False,
            )
            with self.assertRaisesRegex(ValueError, "ideas differ"):
                _prepare_project_input(changed_args, dirs)

            topic = root / "00_config" / "topic.md"
            topic.parent.mkdir(parents=True, exist_ok=True)
            topic.write_text("# Same topic\n", encoding="utf-8")
            same_topic = base / "same-topic.md"
            same_topic.write_text("# Same topic\n", encoding="utf-8")
            changed_topic = base / "changed-topic.md"
            changed_topic.write_text("# Changed topic\n", encoding="utf-8")

            matching_topic_args = argparse.Namespace(
                question=None,
                topic=str(same_topic),
                ideas=None,
                resume=True,
                skip_ideation=False,
            )
            _prepare_project_input(matching_topic_args, dirs)
            changed_topic_args = argparse.Namespace(
                question=None,
                topic=str(changed_topic),
                ideas=None,
                resume=True,
                skip_ideation=False,
            )
            with self.assertRaisesRegex(ValueError, "topic differs"):
                _prepare_project_input(changed_topic_args, dirs)

    def test_managed_project_directory_symlinks_fail_before_external_writes(
        self,
    ) -> None:
        managed_names = (
            "00_config",
            "01_ideas",
            "02_experiments",
            "03_papers",
            "04_logs",
        )
        for managed_name in managed_names:
            with (
                self.subTest(managed_name=managed_name),
                tempfile.TemporaryDirectory() as td,
            ):
                base = Path(td)
                root = base / "project"
                outside = base / "outside"
                root.mkdir()
                outside.mkdir()
                sentinel = outside / "sentinel.txt"
                sentinel.write_text("unchanged", encoding="utf-8")
                (root / managed_name).symlink_to(outside, target_is_directory=True)

                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    create_project_structure(str(root))

                self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
                self.assertEqual(
                    sorted(path.name for path in root.iterdir()), [managed_name]
                )

    def test_managed_input_leaf_symlinks_are_never_read_or_overwritten(self) -> None:
        scenarios = (
            ("00_config", "topic.md"),
            ("01_ideas", "ideas.json"),
            ("01_ideas", "generated_ideas.json"),
        )
        for directory, filename in scenarios:
            with (
                self.subTest(directory=directory, filename=filename),
                tempfile.TemporaryDirectory() as td,
            ):
                base = Path(td)
                root = base / "project"
                for name in (
                    "00_config",
                    "01_ideas",
                    "02_experiments",
                    "03_papers",
                    "04_logs",
                ):
                    (root / name).mkdir(parents=True, exist_ok=True)
                outside = base / f"outside-{filename}"
                outside.write_text("external sentinel", encoding="utf-8")
                (root / directory / filename).symlink_to(outside)
                dirs = {
                    "root": root,
                    "ideas": root / "01_ideas",
                    "experiments": root / "02_experiments",
                    "papers": root / "03_papers",
                    "logs": root / "04_logs",
                }
                args = argparse.Namespace(
                    question="replacement" if filename == "topic.md" else None,
                    topic=None,
                    ideas=None,
                    resume=filename != "topic.md",
                    skip_ideation=False,
                )

                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    _prepare_project_input(args, dirs)

                self.assertEqual(
                    outside.read_text(encoding="utf-8"), "external sentinel"
                )

    def test_progress_leaf_symlink_is_never_read_or_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "project"
            for name in (
                "00_config",
                "01_ideas",
                "02_experiments",
                "03_papers",
                "04_logs",
            ):
                (root / name).mkdir(parents=True, exist_ok=True)
            self._write_project_ideas(root)
            outside = base / "external-progress.json"
            external_payload = {
                "selected_indices": [0],
                "results": [{"idea_idx": 0, "status": "success"}],
            }
            outside.write_text(json.dumps(external_payload), encoding="utf-8")
            (root / "04_logs" / "progress.json").symlink_to(outside)

            with (
                mock.patch("ai_scientist.apps.project._safe_load_json") as loader,
                self.assertRaisesRegex(ValueError, "must not be a symlink"),
            ):
                _completed_resume_results(root)
            loader.assert_not_called()

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                _save_project_progress(
                    root,
                    ideas_hash=self._project_ideas_hash(root),
                    results=[{"idea_idx": 0, "status": "success"}],
                    total=1,
                    selected_indices=[0],
                )
            self.assertEqual(
                json.loads(outside.read_text(encoding="utf-8")), external_payload
            )

    def test_resume_progress_is_bound_to_the_canonical_ideas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            ideas_path = self._write_project_ideas(
                root,
                ideas=[{"Name": "original", "Experiment": {"seed": 1}}],
            )
            paper = root / "03_papers" / "paper.pdf"
            paper.parent.mkdir(parents=True)
            paper.write_bytes(self._VALID_PDF)
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=[{"idea_idx": 0, "status": "success", "pdf_path": str(paper)}],
                total=1,
                selected_indices=[0],
            )
            progress = json.loads(
                (root / "04_logs" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertTrue(str(progress.get("ideas_hash", "")).startswith("sha256:"))

            ideas_path.write_text(
                json.dumps([{"Name": "replacement", "Experiment": {"seed": 2}}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "current ideas differ"):
                _completed_resume_results(root)
            self.assertEqual(paper.read_bytes(), self._VALID_PDF)

    def test_progress_save_refuses_ideas_changed_during_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            ideas_path = self._write_project_ideas(
                root,
                ideas=[{"Name": "loaded", "Experiment": {"seed": 1}}],
            )
            run_ideas_hash = self._project_ideas_hash(root)
            _save_project_progress(
                root,
                ideas_hash=run_ideas_hash,
                results=[{"idea_idx": 0, "status": "success"}],
                total=1,
                selected_indices=[0],
            )
            progress_path = root / "04_logs" / "progress.json"
            progress_before = progress_path.read_bytes()
            ideas_path.write_text(
                json.dumps([{"Name": "edited", "Experiment": {"seed": 2}}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "changed during this run"):
                _save_project_progress(
                    root,
                    ideas_hash=run_ideas_hash,
                    results=[{"idea_idx": 0, "status": "failed"}],
                    total=1,
                    selected_indices=[0],
                )

            self.assertEqual(progress_path.read_bytes(), progress_before)

    def test_parallel_progress_save_preserves_run_ideas_identity(self) -> None:
        required_keys = (
            "output_root",
            "bfts_config",
            "model_writeup",
            "model_citation",
            "model_review",
            "model_agg_plots",
            "model_writeup_small",
            "num_cite_rounds",
            "writeup_retries",
            "writeup_type",
            "improvement_rounds",
            "review_reflections",
            "review_ensemble",
            "review_fewshot",
            "review_temperature",
            "review_strategy",
            "high_quality_mode",
            "quality_preset",
            "quality_model",
            "target_venue",
            "quality_threshold",
            "rigor_threshold",
            "quality_rewrite_rounds",
            "autonomous_quality_followup_rounds",
            "require_quality_gate",
            "min_submission_priority",
            "max_submission_blockers",
            "writing_profile",
            "writing_audit_rounds",
            "strict_writing_guardrails",
            "guardrail_repair_rounds",
            "workflow_mode",
            "template_profile",
            "template_capability",
            "strict_fallbacks",
            "integrity_forensics_enabled",
        )
        for mutate_ideas in (False, True):
            with (
                self.subTest(mutate_ideas=mutate_ideas),
                tempfile.TemporaryDirectory() as td,
            ):
                root = Path(td) / "project"
                ideas_path = self._write_project_ideas(root)
                ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
                run_ideas_hash = self._project_ideas_hash(root)
                kwargs = {key: None for key in required_keys}
                kwargs.update(
                    ideas_hash=run_ideas_hash,
                    prior_results=[],
                    progress_total=1,
                    selected_indices=[0],
                )
                future = mock.MagicMock()

                def finish() -> dict[str, object]:
                    if mutate_ideas:
                        ideas_path.write_text(
                            json.dumps([{"Name": "edited"}]), encoding="utf-8"
                        )
                    return {"idea_idx": 0, "status": "success"}

                future.result.side_effect = finish
                executor = mock.MagicMock()
                executor.submit.return_value = future
                with (
                    mock.patch(
                        "ai_scientist.apps.project.ProcessPoolExecutor"
                    ) as executor_factory,
                    mock.patch(
                        "ai_scientist.apps.project.as_completed",
                        return_value=[future],
                    ),
                ):
                    executor_factory.return_value.__enter__.return_value = executor
                    if mutate_ideas:
                        with self.assertRaisesRegex(
                            ValueError, "changed during this run"
                        ):
                            run_parallel_experiments(
                                str(root),
                                str(ideas_path),
                                1,
                                [0],
                                **kwargs,
                            )
                    else:
                        results = run_parallel_experiments(
                            str(root), str(ideas_path), 1, [0], **kwargs
                        )
                        self.assertEqual(
                            results, [{"idea_idx": 0, "status": "success"}]
                        )
                        progress = json.loads(
                            (root / "04_logs" / "progress.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        self.assertEqual(progress["ideas_hash"], run_ideas_hash)
                if mutate_ideas:
                    self.assertFalse((root / "04_logs" / "progress.json").exists())
                self.assertEqual(len(ideas), 1)

    def test_resume_refuses_damaged_progress_before_pipeline_calls(self) -> None:
        scenarios = ("{not-json", "[]")
        for content in scenarios:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as td:
                output_root = Path(td)
                project_root = output_root / "projects" / "damaged"
                self._write_project_ideas(project_root)
                progress = project_root / "04_logs" / "progress.json"
                progress.parent.mkdir(parents=True)
                progress.write_text(content, encoding="utf-8")
                with (
                    mock.patch("ai_scientist.apps.project.require_login"),
                    mock.patch(
                        "ai_scientist.apps.project.initialize_runtime",
                        return_value=SimpleNamespace(research_root=output_root),
                    ),
                    mock.patch(
                        "ai_scientist.apps.project.process_single_idea"
                    ) as pipeline,
                    self.assertRaisesRegex(ValueError, "saved project progress"),
                ):
                    main(
                        [
                            "damaged",
                            "--output-root",
                            str(output_root),
                            "--resume",
                            "--research-vcs",
                            "off",
                        ]
                    )
                pipeline.assert_not_called()

    def test_resume_refuses_legacy_progress_without_an_ideas_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            self._write_project_ideas(root)
            progress = root / "04_logs" / "progress.json"
            progress.parent.mkdir(parents=True)
            progress.write_text(
                json.dumps(
                    {
                        "selected_indices": [0],
                        "results": [{"idea_idx": 0, "status": "success"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "legacy progress"):
                _completed_resume_results(root)

    def test_summary_leaf_symlinks_are_never_overwritten(self) -> None:
        for filename in ("project_summary.json", "submission_shortlist.md"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                root = base / "project"
                for name in (
                    "00_config",
                    "01_ideas",
                    "02_experiments",
                    "03_papers",
                    "04_logs",
                ):
                    (root / name).mkdir(parents=True, exist_ok=True)
                outside = base / f"external-{filename}"
                outside.write_text("external sentinel", encoding="utf-8")
                (root / "04_logs" / filename).symlink_to(outside)

                with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                    save_project_summary(str(root), [])

                self.assertEqual(
                    outside.read_text(encoding="utf-8"), "external sentinel"
                )

    def test_resume_skips_success_and_reuses_only_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = (
                root
                / "02_experiments"
                / "run_1"
                / "logs"
                / "bfts"
                / "stage_0"
                / "checkpoint.json"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}", encoding="utf-8")
            paper = root / "03_papers" / "idea-0.pdf"
            paper.parent.mkdir(parents=True)
            paper.write_bytes(self._VALID_PDF)
            self._write_project_ideas(root, count=3)
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=[
                    {
                        "idea_idx": 0,
                        "status": "success",
                        "pdf_path": str(paper),
                    },
                    {
                        "idea_idx": 1,
                        "status": "budget_exhausted",
                        "checkpoint_path": str(checkpoint),
                    },
                    {
                        "idea_idx": 2,
                        "status": "failed",
                        "checkpoint_path": str(root / "missing.json"),
                    },
                ],
                total=3,
                selected_indices=[0, 1, 2],
            )
            persisted_text = (root / "04_logs" / "progress.json").read_text(
                encoding="utf-8"
            )
            persisted = json.loads(persisted_text)
            self.assertNotIn(str(root.resolve()), persisted_text)
            self.assertEqual(
                persisted["results"][1]["checkpoint_path"],
                "02_experiments/run_1/logs/bfts/stage_0/checkpoint.json",
            )
            self.assertEqual(
                persisted["results"][2]["checkpoint_path"],
                "missing.json",
            )

            pending, prior, checkpoints = _resolve_resume_work(
                root, [0, 1, 2], enabled=True
            )

            self.assertEqual(pending, [1, 2])
            self.assertEqual([item["idea_idx"] for item in prior], [0])
            self.assertEqual(checkpoints, {1: str(checkpoint.resolve())})

    def test_resume_retries_success_with_unfinished_paper_state_or_review_blockers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            papers: dict[int, Path] = {}
            for idea_idx in range(12):
                paper = root / "03_papers" / f"idea-{idea_idx}.pdf"
                paper.parent.mkdir(parents=True, exist_ok=True)
                paper.write_bytes(self._VALID_PDF)
                papers[idea_idx] = paper
            review_state = root / "02_experiments" / "reviewed" / "review_state.json"
            review_state.parent.mkdir(parents=True)
            review_state.write_text(
                json.dumps(
                    {
                        "repair_metrics": {"active_issue_count": 1},
                        "active_issue_records": [{"issue_id": "RVW-1"}],
                    }
                ),
                encoding="utf-8",
            )
            self._write_project_ideas(root, count=12)
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=[
                    # Legacy progress had no research_status and remains compatible.
                    {
                        "idea_idx": 0,
                        "status": "success",
                        "pdf_path": str(papers[0]),
                    },
                    {
                        "idea_idx": 1,
                        "status": "success",
                        "research_status": "revision_needed",
                        "pdf_path": str(papers[1]),
                    },
                    {
                        "idea_idx": 2,
                        "status": "success",
                        "research_status": "manuscript_draft",
                        "pdf_path": str(papers[2]),
                    },
                    {
                        "idea_idx": 3,
                        "status": "success",
                        "research_status": "exploratory_draft",
                        "pdf_path": str(papers[3]),
                    },
                    {
                        "idea_idx": 4,
                        "status": "success",
                        "research_status": "evidence_blocked",
                        "pdf_path": str(papers[4]),
                    },
                    {
                        "idea_idx": 5,
                        "status": "success",
                        "research_status": "quality_gate_failed",
                        "pdf_path": str(papers[5]),
                    },
                    {
                        "idea_idx": 6,
                        "status": "success",
                        "research_status": "submission_ready",
                        "critic_active_issue_count": 1,
                        "pdf_path": str(papers[6]),
                    },
                    {
                        "idea_idx": 7,
                        "status": "success",
                        "research_status": "submission_ready",
                        "blocker_count": 0,
                        "pdf_path": str(papers[7]),
                    },
                    {
                        "idea_idx": 8,
                        "status": "success",
                        "research_status": "submission_ready",
                        "critic_blocking_issue_count": 1,
                        "pdf_path": str(papers[8]),
                    },
                    {
                        "idea_idx": 9,
                        "status": "success",
                        "research_status": "submission_ready",
                        "blocker_count": "unknown",
                        "pdf_path": str(papers[9]),
                    },
                    {
                        "idea_idx": 10,
                        "status": "success",
                        "research_status": "submission_ready",
                        "review_state_file": str(review_state),
                        "pdf_path": str(papers[10]),
                    },
                    {
                        "idea_idx": 11,
                        "status": "success",
                        "research_status": "submission_ready",
                        "critic_blocking_issue_count": False,
                        "pdf_path": str(papers[11]),
                    },
                ],
                total=12,
                selected_indices=list(range(12)),
            )

            pending, prior, checkpoints = _resolve_resume_work(
                root, list(range(12)), enabled=True
            )

            self.assertEqual(pending, [1, 4, 5, 6, 8, 9, 10, 11])
            self.assertEqual([item["idea_idx"] for item in prior], [0, 2, 3, 7])
            self.assertEqual(checkpoints, {})
            self.assertIsNone(_completed_resume_results(root))

    def test_resume_preserves_progress_when_first_pending_result_crashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "outputs"
            project_root = output_root / "projects" / "resume-demo"
            ideas_path = project_root / "01_ideas" / "ideas.json"
            ideas_path.parent.mkdir(parents=True)
            ideas_path.write_text(
                json.dumps([{"Name": "retry", "Experiment": "rerun"}]),
                encoding="utf-8",
            )
            _save_project_progress(
                project_root,
                ideas_hash=self._project_ideas_hash(project_root),
                results=[
                    {
                        "idea_idx": 0,
                        "status": "success",
                        "research_status": "revision_needed",
                        "checkpoint_path": "02_experiments/retry/checkpoint.json",
                    }
                ],
                total=1,
                selected_indices=[0],
            )
            progress_path = project_root / "04_logs" / "progress.json"
            progress_before = progress_path.read_bytes()

            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch("ai_scientist.apps.project.require_model_credentials"),
                mock.patch(
                    "ai_scientist.apps.project.process_single_idea",
                    side_effect=RuntimeError("forced pre-result crash"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "forced pre-result crash"),
            ):
                main(
                    [
                        "resume-demo",
                        "--output-root",
                        str(output_root),
                        "--resume",
                        "--research-vcs",
                        "off",
                    ]
                )

            self.assertEqual(progress_path.read_bytes(), progress_before)

    def test_resume_rejects_malformed_declared_review_issue_lists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results: list[dict[str, object]] = []
            declared_values = ([], 0, {}, False, None)
            for idea_idx, declared in enumerate(declared_values):
                paper = root / "03_papers" / f"idea-{idea_idx}.pdf"
                paper.parent.mkdir(parents=True, exist_ok=True)
                paper.write_bytes(self._VALID_PDF)
                results.append(
                    {
                        "idea_idx": idea_idx,
                        "status": "success",
                        "research_status": "submission_ready",
                        "pdf_path": str(paper),
                        "active_review_issues": declared,
                    }
                )
            self._write_project_ideas(root, count=len(results))
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=results,
                total=len(results),
                selected_indices=list(range(len(results))),
            )

            pending, prior, _checkpoints = _resolve_resume_work(
                root, list(range(len(results))), enabled=True
            )

            self.assertEqual([item["idea_idx"] for item in prior], [0])
            self.assertEqual(pending, [1, 2, 3, 4])

    def test_resume_rejects_malformed_review_state_and_research_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results: list[dict[str, object]] = []
            malformed_review_states = (
                {"active_issue_records": {}},
                {"active_issues": ""},
                {"repair_metrics": []},
                {"lane_summaries": []},
            )
            for idea_idx, review_state in enumerate(malformed_review_states):
                paper = root / "03_papers" / f"idea-{idea_idx}.pdf"
                paper.parent.mkdir(parents=True, exist_ok=True)
                paper.write_bytes(self._VALID_PDF)
                review_path = root / "02_experiments" / f"review-{idea_idx}.json"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                review_path.write_text(json.dumps(review_state), encoding="utf-8")
                results.append(
                    {
                        "idea_idx": idea_idx,
                        "status": "success",
                        "research_status": "submission_ready",
                        "pdf_path": str(paper),
                        "review_state_file": str(review_path),
                    }
                )

            malformed_statuses = ([], {}, True, "", "future_unknown")
            for offset, research_status in enumerate(malformed_statuses, start=4):
                paper = root / "03_papers" / f"idea-{offset}.pdf"
                paper.write_bytes(self._VALID_PDF)
                results.append(
                    {
                        "idea_idx": offset,
                        "status": "success",
                        "research_status": research_status,
                        "pdf_path": str(paper),
                    }
                )

            valid_idx = len(results)
            valid_paper = root / "03_papers" / f"idea-{valid_idx}.pdf"
            valid_paper.write_bytes(self._VALID_PDF)
            results.append(
                {
                    "idea_idx": valid_idx,
                    "status": "success",
                    "research_status": "manuscript_draft",
                    "pdf_path": str(valid_paper),
                }
            )
            self._write_project_ideas(root, count=len(results))
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=results,
                total=len(results),
                selected_indices=list(range(len(results))),
            )

            pending, prior, _checkpoints = _resolve_resume_work(
                root, list(range(len(results))), enabled=True
            )

        self.assertEqual(pending, list(range(valid_idx)))
        self.assertEqual([item["idea_idx"] for item in prior], [valid_idx])

    def test_resume_result_upsert_preserves_unselected_checkpoints(self) -> None:
        existing = [
            {
                "idea_idx": 0,
                "status": "success",
                "research_status": "revision_needed",
                "checkpoint_path": "02_experiments/zero/checkpoint.json",
            },
            {
                "idea_idx": 1,
                "status": "success",
                "checkpoint_path": "02_experiments/one/checkpoint.json",
            },
        ]
        replacement = {
            "idea_idx": 0,
            "status": "success",
            "research_status": "submission_ready",
            "pdf_path": "03_papers/zero.pdf",
        }

        merged = _upsert_project_results(existing, [replacement])

        self.assertEqual(merged, [replacement, existing[1]])

    def test_resume_refuses_to_change_original_question(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dirs = {
                "root": root,
                "ideas": root / "01_ideas",
                "experiments": root / "02_experiments",
                "papers": root / "03_papers",
                "logs": root / "04_logs",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
            topic = root / "00_config" / "topic.md"
            topic.parent.mkdir(parents=True)
            topic.write_text("# Research question\n\nOriginal\n", encoding="utf-8")
            args = argparse.Namespace(
                question="Changed",
                topic=None,
                ideas=None,
                resume=True,
                skip_ideation=False,
            )

            with self.assertRaisesRegex(ValueError, "differs"):
                _prepare_project_input(args, dirs)

    def test_completed_resume_is_idempotent_and_needs_no_new_model_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first_paper = root / "03_papers" / "idea-0.pdf"
            second_paper = root / "03_papers" / "idea-2.pdf"
            first_paper.parent.mkdir(parents=True)
            first_paper.write_bytes(self._VALID_PDF)
            second_paper.write_bytes(self._VALID_PDF)
            self._write_project_ideas(root, count=3)
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=[
                    {
                        "idea_idx": 0,
                        "status": "success",
                        "quality_score": 8.0,
                        "pdf_path": str(first_paper),
                    },
                    {
                        "idea_idx": 2,
                        "status": "success",
                        "quality_score": 7.5,
                        "pdf_path": str(second_paper),
                    },
                ],
                total=2,
                selected_indices=[0, 2],
            )

            completed = _completed_resume_results(root)

            self.assertEqual([item["idea_idx"] for item in completed or []], [0, 2])

    def test_completed_resume_requires_a_valid_pdf_for_every_selected_idea(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paper = root / "03_papers" / "idea-0.pdf"
            paper.parent.mkdir(parents=True)
            paper.write_bytes(self._VALID_PDF)
            self._write_project_ideas(root, count=2)
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=[
                    {
                        "idea_idx": 0,
                        "status": "success",
                        "pdf_path": str(paper),
                    },
                    {
                        "idea_idx": 1,
                        "status": "success",
                        "pdf_path": "03_papers/missing.pdf",
                    },
                ],
                total=2,
                selected_indices=[0, 1],
            )

            self.assertIsNone(_completed_resume_results(root))
            pending, prior, _checkpoints = _resolve_resume_work(
                root, [0, 1], enabled=True
            )
            self.assertEqual(pending, [1])
            self.assertEqual([item["idea_idx"] for item in prior], [0])

    def test_completed_resume_rejects_a_truncated_pdf_header(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paper = root / "03_papers" / "truncated.pdf"
            paper.parent.mkdir(parents=True)
            paper.write_bytes(b"%PDF-")
            self._write_project_ideas(root)
            _save_project_progress(
                root,
                ideas_hash=self._project_ideas_hash(root),
                results=[{"idea_idx": 0, "status": "success", "pdf_path": str(paper)}],
                total=1,
                selected_indices=[0],
            )

            self.assertIsNone(_completed_resume_results(root))
            pending, prior, _checkpoints = _resolve_resume_work(root, [0], enabled=True)
            self.assertEqual(pending, [0])
            self.assertEqual(prior, [])

    def test_autopilot_preflight_fails_before_execution_when_isolation_is_missing(
        self,
    ) -> None:
        from ai_scientist.apps.preflight import CheckResult

        args = argparse.Namespace(
            autopilot="balanced",
            bfts_config="demo.yaml",
            project_dir="study",
        )
        with mock.patch(
            "ai_scientist.apps.preflight.check_bfts_config",
            return_value=[
                CheckResult(
                    label="Experiment isolation",
                    ok=False,
                    severity="error",
                    detail="docker image is unavailable",
                )
            ],
        ) as check_bfts_config:
            with self.assertRaisesRegex(RuntimeError, "before any research-model call"):
                _run_autopilot_preflight(args)
        check_bfts_config.assert_called_once_with("demo.yaml", workspace="study")

    def test_ideation_only_autopilot_does_not_require_an_executor(self) -> None:
        args = argparse.Namespace(
            autopilot="discovery",
            bfts_config="demo.yaml",
            skip_experiment=True,
        )
        with mock.patch(
            "ai_scientist.apps.preflight.check_bfts_config"
        ) as check_bfts_config:
            rows = _run_autopilot_preflight(args)

        check_bfts_config.assert_not_called()
        self.assertEqual(rows[0]["severity"], "info")
        self.assertIn("--skip-experiment", rows[0]["detail"])

    def test_skip_experiment_reports_that_no_paper_was_generated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "outputs"
            ideas_path = Path(td) / "ideas.json"
            ideas_path.write_text(
                json.dumps([{"Name": "existing idea", "Experiment": "unused"}]),
                encoding="utf-8",
            )
            project_root = output_root / "projects" / "skip-demo"
            self._write_project_ideas(
                project_root,
                ideas=[{"Name": "existing idea", "Experiment": "unused"}],
            )
            _save_project_progress(
                project_root,
                ideas_hash=self._project_ideas_hash(project_root),
                results=[{"idea_idx": 0, "status": "success"}],
                total=1,
                selected_indices=[0],
            )
            output = io.StringIO()
            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch("ai_scientist.apps.project.require_model_credentials"),
                mock.patch(
                    "ai_scientist.apps.project.select_ranked_idea_candidates",
                    return_value=([0], []),
                ),
                contextlib.redirect_stdout(output),
            ):
                main(
                    [
                        "skip-demo",
                        "--output-root",
                        str(output_root),
                        "--ideas",
                        str(ideas_path),
                        "--resume",
                        "--skip-experiment",
                        "--research-vcs",
                        "off",
                    ]
                )

            rendered = output.getvalue()
            self.assertIn("--skip-experiment", rendered)
            self.assertIn("experiment → plot → writeup → review", rendered)
            self.assertIn("本次运行未生成论文", rendered)
            self.assertNotIn("🎉 项目完成", rendered)
            self.assertNotIn("📄 论文位置", rendered)
            papers = project_root / "03_papers"
            self.assertEqual(list(papers.iterdir()), [])
            progress = json.loads(
                (project_root / "04_logs" / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                progress["results"], [{"idea_idx": 0, "status": "success"}]
            )

    def test_autopilot_derives_finite_isolated_bfts_budget(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "exec": {
                            "backend": "auto",
                            "require_isolation": False,
                            "docker_image": "xscientist-exec:latest",
                        },
                        "llm_budget": {
                            "max_total_tokens": None,
                            "max_wall_time_seconds": None,
                        },
                        "agent": {
                            "steps": 20,
                            "stages": {
                                "stage1_max_iters": 100,
                                "stage2_max_iters": 100,
                                "stage3_max_iters": 100,
                                "stage4_max_iters": 100,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                autopilot="balanced",
                bfts_config=str(source),
                project_dir=str(root / "project"),
                max_project_tokens=250_000,
                max_project_hours=1,
                max_cost_usd=5,
            )

            derived_path = _prepare_autopilot_bfts_config(args)
            derived = yaml.safe_load(
                Path(derived_path or "").read_text(encoding="utf-8")
            )

            self.assertTrue(derived["exec"]["require_isolation"])
            self.assertEqual(derived["exec"]["network"], "none")
            self.assertFalse(derived["exec"]["allow_experiment_network"])
            self.assertEqual(derived["llm_budget"]["max_total_tokens"], 250_000)
            self.assertEqual(derived["llm_budget"]["max_wall_time_seconds"], 3_600)
            self.assertEqual(derived["llm_budget"]["max_cost_usd"], 5)
            self.assertEqual(derived["agent"]["steps"], 3)
            self.assertEqual(derived["agent"]["stages"]["stage1_max_iters"], 8)

            with mock.patch.dict("os.environ", {}, clear=False):
                budget = _configure_autopilot_project_budget(args)
                self.assertTrue(budget["shared_across_project"])
                self.assertTrue(
                    (root / "project" / "04_logs" / "llm_budget.json").is_file()
                )

    def test_autopilot_binds_original_trusted_executor_workspace(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trusted = root / "trusted"
            external_project = root / "external-output" / "project"
            (trusted / ".xscientist").mkdir(parents=True)
            (trusted / ".xscientist" / "providers.json").write_text(
                "{}", encoding="utf-8"
            )
            (trusted / "Dockerfile.executor").write_text(
                "FROM python:3.11-slim\n", encoding="utf-8"
            )
            source = trusted / "bfts_config.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "exec": {
                            "backend": "docker",
                            "require_isolation": True,
                            "docker_image": "xscientist-exec:test",
                        },
                        "llm_budget": {},
                        "agent": {"stages": {}},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                autopilot="balanced",
                bfts_config=str(source),
                project_dir=str(external_project),
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                derived = _prepare_autopilot_bfts_config(args)
                self.assertEqual(
                    os.environ.get("XSCIENTIST_WORKSPACE"), str(trusted.resolve())
                )
                self.assertEqual(
                    Path(derived or "").resolve(),
                    (external_project / "00_config" / "autopilot_bfts.yaml").resolve(),
                )

    def test_non_autopilot_binds_trusted_config_root_from_nonworkspace_cwd(
        self,
    ) -> None:
        from ai_scientist.treesearch.interpreter import Interpreter, SandboxPolicy

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trusted = root / "trusted"
            external_project = root / "external-output" / "project"
            nonworkspace = root / "unrelated-cwd"
            nonworkspace.mkdir()
            external_project.mkdir(parents=True)
            (trusted / ".xscientist").mkdir(parents=True)
            (trusted / ".xscientist" / "providers.json").write_text(
                "{}", encoding="utf-8"
            )
            source = trusted / "bfts_config.yaml"
            source.write_text("exec: {}\n", encoding="utf-8")
            args = argparse.Namespace(
                autopilot=None,
                bfts_config=str(source),
                project_dir=str(external_project),
            )
            original_cwd = Path.cwd()
            try:
                os.chdir(nonworkspace)
                with mock.patch.dict(os.environ, {}, clear=True):
                    derived = _prepare_autopilot_bfts_config(args)
                    self.assertIsNone(derived)
                    self.assertEqual(
                        os.environ.get("XSCIENTIST_WORKSPACE"),
                        str(trusted.resolve()),
                    )
                    with (
                        mock.patch(
                            "ai_scientist.treesearch.interpreter.docker_is_available",
                            return_value=(True, None),
                        ),
                        mock.patch(
                            "xscientist.executor_manager.inspect_executor",
                            return_value={
                                "ok": True,
                                "image": "xscientist-exec:test",
                                "image_id": "sha256:" + "a" * 64,
                                "error": None,
                            },
                        ) as inspect_executor,
                    ):
                        interpreter = Interpreter(
                            external_project,
                            sandbox_policy=SandboxPolicy(
                                backend="docker",
                                require_isolation=True,
                                docker_image="xscientist-exec:test",
                            ),
                        )
                    self.assertEqual(interpreter.execution_backend, "docker")
                    inspect_executor.assert_called_once_with(trusted.resolve())
            finally:
                os.chdir(original_cwd)

    def test_project_entrypoint_explicit_config_overrides_only_bootstrap_workspace(
        self,
    ) -> None:
        from xscientist.entrypoints import project_main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cwd_workspace = root / "cwd-workspace"
            config_workspace = root / "config-workspace"
            user_workspace = root / "user-workspace"
            for workspace in (cwd_workspace, config_workspace, user_workspace):
                (workspace / ".xscientist").mkdir(parents=True)
                (workspace / ".xscientist" / "providers.json").write_text(
                    "{}", encoding="utf-8"
                )
                (workspace / "Dockerfile.executor").write_text(
                    "FROM python:3.11-slim\n", encoding="utf-8"
                )
                (workspace / "bfts_config.yaml").write_text(
                    "exec: {}\n", encoding="utf-8"
                )
            nested_cwd = cwd_workspace / "notes"
            nested_cwd.mkdir()
            observed_workspaces: list[str | None] = []

            def project_entry() -> int:
                argv = sys.argv[1:]
                config_index = argv.index("--bfts-config") + 1
                args = argparse.Namespace(
                    autopilot=None,
                    bfts_config=argv[config_index],
                    project_dir=str(root / "external-project"),
                )
                _prepare_autopilot_bfts_config(args)
                observed_workspaces.append(os.environ.get("XSCIENTIST_WORKSPACE"))
                return 0

            module = SimpleNamespace(main=project_entry)
            invocation = [
                "demo",
                "--bfts-config",
                str(config_workspace / "bfts_config.yaml"),
            ]
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("xscientist.entrypoints.Path.cwd", return_value=nested_cwd),
                mock.patch(
                    "xscientist.provider_config.load_workspace_environment",
                    return_value={"error": None},
                ),
                mock.patch(
                    "xscientist.entrypoints.importlib.import_module",
                    return_value=module,
                ),
            ):
                self.assertEqual(project_main(invocation), 0)
            with (
                mock.patch.dict(
                    os.environ,
                    {"XSCIENTIST_WORKSPACE": str(user_workspace)},
                    clear=True,
                ),
                mock.patch("xscientist.entrypoints.Path.cwd", return_value=nested_cwd),
                mock.patch(
                    "xscientist.provider_config.load_workspace_environment",
                    return_value={"error": None},
                ),
                mock.patch(
                    "xscientist.entrypoints.importlib.import_module",
                    return_value=module,
                ),
            ):
                self.assertEqual(project_main(invocation), 0)

            self.assertEqual(
                observed_workspaces,
                [str(config_workspace.resolve()), str(user_workspace)],
            )

    def test_autopilot_never_overwrites_explicit_executor_workspace(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trusted = root / "trusted"
            trusted.mkdir()
            (trusted / "Dockerfile.executor").write_text(
                "FROM python:3.11-slim\n", encoding="utf-8"
            )
            source = trusted / "bfts_config.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "exec": {"docker_image": "xscientist-exec:test"},
                        "agent": {"stages": {}},
                    }
                ),
                encoding="utf-8",
            )
            explicit = root / "explicit-invalid-workspace"
            args = argparse.Namespace(
                autopilot="balanced",
                bfts_config=str(source),
                project_dir=str(root / "external-project"),
            )

            with mock.patch.dict(
                os.environ,
                {"XSCIENTIST_WORKSPACE": str(explicit)},
                clear=True,
            ):
                _prepare_autopilot_bfts_config(args)
                self.assertEqual(os.environ.get("XSCIENTIST_WORKSPACE"), str(explicit))
                args.autopilot = None
                self.assertIsNone(_prepare_autopilot_bfts_config(args))
                self.assertEqual(os.environ.get("XSCIENTIST_WORKSPACE"), str(explicit))

    def test_autopilot_generic_config_does_not_invent_executor_workspace(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            standalone = root / "standalone"
            standalone.mkdir()
            source = standalone / "custom.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "exec": {"docker_image": "generic:test"},
                        "agent": {"stages": {}},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                autopilot="balanced",
                bfts_config=str(source),
                project_dir=str(root / "external-project"),
            )

            with mock.patch.dict(os.environ, {}, clear=True):
                _prepare_autopilot_bfts_config(args)
                self.assertNotIn("XSCIENTIST_WORKSPACE", os.environ)
                args.autopilot = None
                self.assertIsNone(_prepare_autopilot_bfts_config(args))
                self.assertNotIn("XSCIENTIST_WORKSPACE", os.environ)

    def test_derived_autopilot_config_is_accepted_by_bfts_schema(self) -> None:
        from ai_scientist.resources import resolve_bfts_config_path
        from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file
        from ai_scientist.treesearch.utils.config import load_cfg

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = argparse.Namespace(
                autopilot="discovery",
                bfts_config=str(resolve_bfts_config_path("bfts_config.yaml")),
                project_dir=str(root / "project"),
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                derived = _prepare_autopilot_bfts_config(args)
            exp_dir = root / "experiment"
            exp_dir.mkdir()
            idea = exp_dir / "idea.md"
            idea.write_text("# Test idea\n", encoding="utf-8")
            run_config = edit_bfts_config_file(str(derived), str(exp_dir), str(idea))

            cfg = load_cfg(Path(run_config))

            self.assertEqual(cfg.llm_budget.max_total_tokens, 1_500_000)
            self.assertTrue(cfg.exec.require_isolation)
            self.assertFalse(cfg.exec.allow_experiment_network)

    def test_completed_autopilot_main_returns_without_credentials_or_model_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            project_root = output_root / "projects" / "demo"
            paper = project_root / "03_papers" / "paper.pdf"
            paper.parent.mkdir(parents=True)
            paper.write_bytes(self._VALID_PDF)
            self._write_project_ideas(project_root)
            _save_project_progress(
                project_root,
                ideas_hash=self._project_ideas_hash(project_root),
                results=[
                    {
                        "idea_idx": 0,
                        "status": "success",
                        "pdf_path": str(paper),
                        "submission_acceptance_passed": True,
                        "quality_gate_passed": True,
                        "submission_priority_score": 10.0,
                        "blocker_count": 0,
                    }
                ],
                total=1,
                selected_indices=[0],
            )
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
                    "ai_scientist.apps.project._run_autopilot_preflight"
                ) as preflight,
                mock.patch("ai_scientist.apps.project._export_project_research_dag"),
                mock.patch("builtins.print"),
            ):
                main(
                    [
                        "demo",
                        "--output-root",
                        str(output_root),
                        "--autopilot",
                        "balanced",
                        "--research-vcs",
                        "off",
                    ]
                )

            credentials.assert_not_called()
            preflight.assert_not_called()

    def test_resume_without_a_pdf_does_not_claim_completion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            project_root = output_root / "projects" / "demo"
            self._write_project_ideas(project_root)
            _save_project_progress(
                project_root,
                ideas_hash=self._project_ideas_hash(project_root),
                results=[{"idea_idx": 0, "status": "success"}],
                total=1,
                selected_indices=[0],
            )
            output = io.StringIO()
            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                contextlib.redirect_stdout(output),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "demo",
                        "--output-root",
                        str(output_root),
                        "--resume",
                        "--research-vcs",
                        "off",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            credentials.assert_called_once()
            rendered = output.getvalue()
            self.assertNotIn("项目完成", rendered)
            self.assertNotIn("论文位置", rendered)
            self.assertNotIn("📄 PDF:", rendered)

    def test_completed_resume_rechecks_current_publication_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            project_root = output_root / "projects" / "demo"
            paper = project_root / "03_papers" / "paper.pdf"
            paper.parent.mkdir(parents=True)
            paper.write_bytes(self._VALID_PDF)
            self._write_project_ideas(project_root)
            _save_project_progress(
                project_root,
                ideas_hash=self._project_ideas_hash(project_root),
                results=[
                    {
                        "idea_idx": 0,
                        "status": "success",
                        "pdf_path": str(paper),
                        "submission_acceptance_passed": False,
                    }
                ],
                total=1,
                selected_indices=[0],
            )
            output = io.StringIO()
            with (
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch(
                    "ai_scientist.apps.project.require_model_credentials"
                ) as credentials,
                contextlib.redirect_stdout(output),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "demo",
                        "--output-root",
                        str(output_root),
                        "--resume",
                        "--require-quality-gate",
                        "--research-vcs",
                        "off",
                    ]
                )

            self.assertEqual(raised.exception.code, 1)
            credentials.assert_not_called()
            rendered = output.getvalue()
            self.assertIn("没有任何项目论文通过", rendered)
            self.assertNotIn("本次流程完成", rendered)

    def test_completed_resume_recomputes_stricter_current_gate_fields(self) -> None:
        scenarios = (
            (
                "quality",
                {"quality_gate_passed": False},
                ["--require-quality-gate"],
            ),
            (
                "priority",
                {"submission_priority_score": 3.0},
                ["--min-submission-priority", "5"],
            ),
        )
        for label, stale_fields, gate_args in scenarios:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                output_root = Path(td)
                project_root = output_root / "projects" / "demo"
                paper = project_root / "03_papers" / "paper.pdf"
                paper.parent.mkdir(parents=True)
                paper.write_bytes(self._VALID_PDF)
                self._write_project_ideas(project_root)
                result = {
                    "idea_idx": 0,
                    "status": "success",
                    "pdf_path": str(paper),
                    "submission_acceptance_passed": True,
                    "quality_gate_passed": True,
                    "submission_priority_score": 9.0,
                    "blocker_count": 0,
                    **stale_fields,
                }
                _save_project_progress(
                    project_root,
                    ideas_hash=self._project_ideas_hash(project_root),
                    results=[result],
                    total=1,
                    selected_indices=[0],
                )
                output = io.StringIO()
                with (
                    mock.patch("ai_scientist.apps.project.require_login"),
                    mock.patch(
                        "ai_scientist.apps.project.initialize_runtime",
                        return_value=SimpleNamespace(research_root=output_root),
                    ),
                    mock.patch(
                        "ai_scientist.apps.project.require_model_credentials"
                    ) as credentials,
                    contextlib.redirect_stdout(output),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(
                        [
                            "demo",
                            "--output-root",
                            str(output_root),
                            "--resume",
                            "--research-vcs",
                            "off",
                            *gate_args,
                        ]
                    )

                self.assertEqual(raised.exception.code, 1)
                credentials.assert_not_called()
                self.assertIn("没有任何项目论文通过", output.getvalue())

    def test_publication_gate_cannot_borrow_quality_from_a_missing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid_pdf = root / "03_papers" / "valid.pdf"
            valid_pdf.parent.mkdir(parents=True)
            valid_pdf.write_bytes(self._VALID_PDF)
            args = argparse.Namespace(
                project_dir=str(root),
                require_quality_gate=True,
                min_submission_priority=None,
                max_submission_blockers=None,
                strict_writing_guardrails=False,
                high_quality_mode=False,
            )

            failure = _publication_gate_failure(
                args,
                [
                    {
                        "status": "success",
                        "pdf_path": "03_papers/missing.pdf",
                        "submission_acceptance_passed": True,
                        "quality_gate_passed": True,
                    },
                    {
                        "status": "success",
                        "pdf_path": str(valid_pdf),
                        "submission_acceptance_passed": False,
                        "quality_gate_passed": False,
                    },
                ],
            )

            self.assertIn("没有任何项目论文通过", failure or "")

            args.require_quality_gate = False
            args.max_submission_blockers = 1
            blocker_failure = _publication_gate_failure(
                args,
                [
                    {
                        "status": "success",
                        "pdf_path": str(valid_pdf),
                        "submission_acceptance_passed": True,
                        "quality_gate_passed": True,
                        "blocker_count": 4,
                    }
                ],
            )
            self.assertIn("没有任何项目论文通过", blocker_failure or "")

    def test_publication_gate_rejects_nonfinite_priority_and_negative_blockers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid_pdf = root / "03_papers" / "valid.pdf"
            valid_pdf.parent.mkdir(parents=True)
            valid_pdf.write_bytes(self._VALID_PDF)
            args = argparse.Namespace(
                project_dir=str(root),
                require_quality_gate=True,
                min_submission_priority=0.0,
                max_submission_blockers=0,
                strict_writing_guardrails=False,
                high_quality_mode=False,
            )
            base_result = {
                "status": "success",
                "pdf_path": str(valid_pdf),
                "submission_acceptance_passed": True,
                "quality_gate_passed": True,
                "blocker_count": 0,
            }

            for priority in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(priority=priority):
                    failure = _publication_gate_failure(
                        args,
                        [{**base_result, "submission_priority_score": priority}],
                    )
                    self.assertIn("没有任何项目论文通过", failure or "")

            failure = _publication_gate_failure(
                args,
                [
                    {
                        **base_result,
                        "submission_priority_score": 10.0,
                        "blocker_count": -1,
                    }
                ],
            )
            self.assertIn("没有任何项目论文通过", failure or "")

    @unittest.skipUnless(shutil.which("git"), "Git is required for the golden journey")
    def test_golden_question_to_insight_and_research_dag_journey(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            output_root = base / "outputs"
            project_root = base / "study"

            def generate(project_dir, _topic, _model, _count, _reflections):
                path = Path(project_dir) / "01_ideas" / "generated_ideas.json"
                path.write_text(
                    json.dumps(
                        [
                            {
                                "Name": "falsifiable_mechanism",
                                "Title": "Falsifiable mechanism study",
                                "Experiment": "Compare the mechanism against a null control.",
                                "Interestingness": 8,
                                "Feasibility": 8,
                                "Novelty": 7,
                                "core_hypothesis": "The mechanism improves the target metric.",
                                "failure_criteria": [
                                    "The paired null control performs at least as well."
                                ],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                return str(path)

            def execute(process_args):
                exp_dir = project_root / "02_experiments" / "idea_0"
                exp_dir.mkdir(parents=True, exist_ok=True)
                paper = project_root / "03_papers" / "idea_0.pdf"
                paper.parent.mkdir(parents=True, exist_ok=True)
                paper.write_bytes(self._VALID_PDF)
                return {
                    "idea_idx": 0,
                    "exp_dir": str(exp_dir),
                    "status": "success",
                    "pdf_path": str(paper),
                    "quality_score": 8.0,
                    "rigor_score": 8.0,
                    "quality_gate_passed": True,
                    "submission_acceptance_passed": True,
                    "claim_support_score": 0.7,
                    "seed": 42,
                }

            with (
                mock.patch.dict("os.environ", {}, clear=False),
                mock.patch("ai_scientist.apps.project.require_login"),
                mock.patch(
                    "ai_scientist.apps.project.initialize_runtime",
                    return_value=SimpleNamespace(research_root=output_root),
                ),
                mock.patch("ai_scientist.apps.project.require_model_credentials"),
                mock.patch(
                    "ai_scientist.apps.project._run_autopilot_preflight",
                    return_value=[],
                ),
                mock.patch(
                    "ai_scientist.apps.project.generate_ideas", side_effect=generate
                ),
                mock.patch(
                    "ai_scientist.apps.project.select_ranked_idea_candidates",
                    return_value=([0], []),
                ),
                mock.patch(
                    "ai_scientist.apps.project.process_single_idea",
                    side_effect=execute,
                ),
                mock.patch(
                    "ai_scientist.llm.create_client",
                    side_effect=RuntimeError("offline golden journey"),
                ),
                mock.patch("builtins.print"),
            ):
                main(
                    [
                        str(project_root),
                        "--output-root",
                        str(output_root),
                        "--question",
                        "Does the mechanism improve the target metric?",
                        "--autopilot",
                        "balanced",
                        "--allow-synthetic-data",
                        "--research-vcs-strict",
                    ]
                )

            insight = json.loads(
                (project_root / "04_logs" / "insight_report.json").read_text()
            )
            self.assertEqual(
                insight["epistemic_status"], "machine_synthesized_unverified"
            )
            self.assertTrue(insight["insights"])
            dag = (
                output_root
                / "views"
                / project_root.name
                / "research-dag"
                / "research-dag.html"
            )
            self.assertTrue(dag.is_file())
            self.assertTrue((project_root / "research.yaml").is_file())
            self.assertTrue(
                (project_root / "00_config" / "data_manifest.json").is_file()
            )
            progress_text = (project_root / "04_logs" / "progress.json").read_text(
                encoding="utf-8"
            )
            progress = json.loads(progress_text)
            self.assertNotIn(str(project_root), progress_text)
            self.assertNotIn(str(project_root.resolve()), progress_text)
            self.assertEqual(
                progress["results"][0]["exp_dir"],
                "02_experiments/idea_0",
            )
            summary_text = (
                project_root / "04_logs" / "project_summary.json"
            ).read_text(encoding="utf-8")
            summary = json.loads(summary_text)
            self.assertNotIn(str(project_root), summary_text)
            self.assertNotIn(str(project_root.resolve()), summary_text)
            self.assertEqual(summary["project_dir"], ".")
            self.assertEqual(
                summary["results"][0]["exp_dir"],
                "02_experiments/idea_0",
            )
            shortlist_text = (
                project_root / "04_logs" / "submission_shortlist.md"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(project_root), shortlist_text)
            self.assertNotIn(str(project_root.resolve()), shortlist_text)


if __name__ == "__main__":
    unittest.main()
