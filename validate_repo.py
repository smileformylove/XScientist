#!/usr/bin/env python3
"""仓库级轻量校验脚本。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import py_compile
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_scientist.utils.auth_session import require_login  # noqa: E402

IGNORED_PATH_PARTS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
    "_outputs",
    "experiments",
    "final_papers",
    "node_modules",
    "results",
    "site-packages",
    "venv",
}

MIN_PYTHON = (3, 10)


def _should_ignore_python_path(path: Path) -> bool:
    parts = path.relative_to(PROJECT_ROOT).parts
    for part in parts:
        if part in IGNORED_PATH_PARTS:
            return True
        if part.startswith(".venv"):
            return True
    return False


def iter_python_files() -> list[Path]:
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if not _should_ignore_python_path(path)
    )


def ensure_supported_python() -> None:
    if sys.version_info >= MIN_PYTHON:
        return
    minimum = ".".join(str(part) for part in MIN_PYTHON)
    current = sys.version.split()[0]
    raise SystemExit(
        f"validate_repo.py requires Python >= {minimum}; current interpreter is {current}"
    )


def run_py_compile() -> None:
    files = iter_python_files()
    for path in files:
        py_compile.compile(str(path), doraise=True)
    print(f"[OK] py_compile passed for {len(files)} Python files")


def run_helper_smoke() -> None:
    from ai_scientist.review_strategies import (
        ReviewStrategy,
        generate_review_instruction,
    )
    from ai_scientist.utils.high_quality_pipeline import (
        VENUE_PRESETS,
        _build_architecture_figure_brief_text,
        _build_experiment_analysis_text,
        _build_experiment_visualization_brief_text,
        _build_figure_caption_guidance_text,
        _build_humanizer_style_notes_text,
        _build_logic_check_text,
        _build_reviewer_gate_report_text,
        _build_table_caption_guidance_text,
        _build_writing_skill_pack_text,
        assess_claim_support,
        assess_claim_alignment,
        assess_experiment_rigor,
        assess_numeric_coverage,
        build_evidence_pack,
        _derive_section_risk_language_guidance,
        build_claim_evidence_ledger,
        build_contribution_map,
        evaluate_submission_acceptance,
        extract_key_results,
        recommend_target_venue_from_idea,
        resolve_submission_acceptance_settings,
        resolve_target_venue,
    )
    from ai_scientist.utils.idea_ranking import rank_ideas
    from ai_scientist.utils.submission_history import (
        recommend_reviewer_risk_mitigation,
        recommend_rewrite_efficiency_controls,
        recommend_rewrite_focus_adjustments,
        recommend_section_reviewer_language_guidance,
        recommend_rewrite_style_preferences,
        recommend_submission_strategy_adjustments,
    )
    from ai_scientist.utils.pipeline_helpers import (
        cleanup_child_processes,
        find_best_pdf_path,
        find_latest_pdf_path,
        get_process_cleanup_capability,
        redirect_stdout_stderr_to_file,
        save_review_artifacts,
        save_token_tracker,
    )
    from ai_scientist.utils.launcher_cli import normalize_common_launcher_args
    from ai_scientist.utils.workflow_cli import (
        normalize_batch_workflow_args,
        normalize_project_workflow_args,
        recommend_default_target_venue,
    )
    from ai_scientist.utils.review_workflow import (
        build_review_execution_plan,
        resolve_review_strategy,
    )
    from ai_scientist.utils.quality_workflow import (
        derive_autonomous_followup_focus,
        execute_quality_workflow,
        execute_quality_workflow_with_followups,
        format_quality_pass_summary,
    )
    from ai_scientist.perform_plotting import (
        build_aggregator_prompt,
        load_quality_plot_guidance,
    )
    from ai_scientist.utils.review_execution import execute_review_pass
    from ai_scientist.utils.writeup_workflow import (
        build_writeup_execution_plan,
        resolve_page_limit,
        resolve_writeup_engine,
    )
    from ai_scientist.utils.workflow_selection import (
        resolve_paper_type_for_venue,
        resolve_paper_types_for_venue,
        select_ranked_idea_candidates,
    )
    from ai_scientist.utils.guardrail_artifacts import load_guardrail_artifacts
    from ai_scientist.utils.guardrail_artifacts import result_passed_writeup_guardrails
    from ai_scientist.writeup_guardrails import (
        build_guardrail_repair_plan,
        collect_guardrail_findings,
        list_blocking_guardrail_reasons,
    )
    from ai_scientist.utils.run_index import (
        infer_run_entry,
        is_stage_complete,
        load_run_index,
        mark_stage_complete,
        rebuild_run_index,
    )
    from ai_scientist.utils.experiment_todo_progress import (
        bootstrap_todo_tasks_from_round_gate,
        evaluate_todo_progress_snapshot,
    )
    from research_manager import ResearchManager
    from continuous_research_daemon import (
        _apply_evidence_strategy_feedback,
        _apply_failure_guard,
        _apply_quality_strategy_feedback,
        _apply_submission_autopilot,
        _build_source_quality_feedback,
        _derive_quality_governor,
        _derive_rewrite_followup_policy,
    )
    from continuous_paper_generator import (
        _build_paper_experiment_todo_tasks,
        _build_batch_experiment_agenda,
        _build_batch_experiment_agenda_markdown,
        _build_batch_experiment_ledger_rows,
        _build_batch_experiment_ledger_tsv,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        save_review_artifacts(root / "review_json", {"score": 1}, {"image": "ok"})
        save_review_artifacts(
            root / "review_text",
            {"score": 2},
            {"image": "ok"},
            text_filename="review_text.txt",
            image_filename="review_img_cap_ref.json",
            text_mode="text_json",
        )
        save_token_tracker(root / "token_usage")

        for name in ["paper.pdf", "reflection_1.pdf", "reflection_final.pdf"]:
            (root / name).write_text("x", encoding="utf-8")

        with redirect_stdout_stderr_to_file(root / "redirect.log"):
            print("redirect-ok")

        assert find_best_pdf_path(root).endswith("reflection_final.pdf")
        assert find_latest_pdf_path(root).endswith("paper.pdf")
        assert (root / "review_json" / "review_text.json").exists()
        assert (root / "review_json" / "review_img.json").exists()
        assert (root / "review_text" / "review_text.txt").exists()
        assert (root / "review_text" / "review_img_cap_ref.json").exists()
        assert (root / "token_usage" / "token_tracker.json").exists()
        assert (root / "token_usage" / "token_tracker_interactions.json").exists()
        review_exec = execute_review_pass(
            paper_dir=root,
            model_review="offline-smoke",
            review_plan={
                "review_reflections": 2,
                "review_fewshot": 1,
                "review_ensemble": 1,
                "review_temperature": 0.2,
                "review_instruction": "smoke-instruction",
            },
            create_client_fn=lambda model: ("client", model),
            load_paper_fn=lambda _path: "paper-content",
            perform_review_fn=lambda *args, **kwargs: {
                "review": {"scores": {"Overall": 7.0}}
            },
            perform_imgs_cap_ref_review_fn=lambda *_args, **_kwargs: {
                "figures": ["ok"]
            },
            pdf_path_resolver=lambda _paper_dir: str(root / "paper.pdf"),
            save_dir=root / "review_exec",
        )
        assert review_exec["found"] is True
        assert review_exec["pdf_path"].endswith("paper.pdf")
        assert (root / "review_exec" / "review_text.json").exists()
        assert (root / "review_exec" / "review_img.json").exists()
        assert (root / "redirect.log").read_text(
            encoding="utf-8"
        ).strip() == "redirect-ok"
        findings_dir = root / "guardrail_exp" / "writing_audits"
        findings_dir.mkdir(parents=True, exist_ok=True)
        (findings_dir / "final_guardrail_report.json").write_text(
            json.dumps(
                {
                    "venue": "iclr",
                    "missing_bibtex_keys": ["missing_key_2024"],
                    "missing_bibtex_key_count": 1,
                    "placeholder_citation_keys": [],
                    "placeholder_citation_key_count": 0,
                    "missing_required_sections": ["Limitations"],
                    "missing_required_section_count": 1,
                    "ai_style_markers": [],
                    "ai_style_marker_count": 0,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        artifact_findings, artifact_reasons = load_guardrail_artifacts(
            str(root / "guardrail_exp")
        )
        assert artifact_findings is not None
        assert any(
            reason.startswith("missing_bibtex_keys=") for reason in artifact_reasons
        )
        (findings_dir / "final_guardrail_report.json").write_text(
            json.dumps(
                {
                    "findings": {
                        "venue": "iclr",
                        "missing_bibtex_keys": [],
                        "missing_bibtex_key_count": 0,
                        "placeholder_citation_keys": [],
                        "placeholder_citation_key_count": 0,
                        "missing_required_sections": [],
                        "missing_required_section_count": 0,
                        "ai_style_markers": [],
                        "ai_style_marker_count": 0,
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (findings_dir / "final_guardrail_failure_reasons.json").write_text(
            json.dumps(
                {"blocking_reasons": "manual_guardrail_block"}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        artifact_findings, artifact_reasons = load_guardrail_artifacts(
            str(root / "guardrail_exp")
        )
        assert artifact_findings is not None
        assert artifact_reasons == ["manual_guardrail_block"]
        (findings_dir / "final_guardrail_report.json").unlink()
        (findings_dir / "final_guardrail_failure_reasons.json").write_text(
            json.dumps(
                {
                    "reasons": ["fallback_reason_from_reasons_file"],
                    "findings": {
                        "venue": "iclr",
                        "missing_bibtex_keys": ["fallback_missing_key"],
                        "missing_bibtex_key_count": 1,
                        "placeholder_citation_keys": [],
                        "placeholder_citation_key_count": 0,
                        "missing_required_sections": [],
                        "missing_required_section_count": 0,
                        "ai_style_markers": [],
                        "ai_style_marker_count": 0,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        artifact_findings, artifact_reasons = load_guardrail_artifacts(
            str(root / "guardrail_exp")
        )
        assert artifact_findings is not None
        assert artifact_findings.get("missing_bibtex_key_count") == 1
        assert artifact_reasons == ["fallback_reason_from_reasons_file"]
        (findings_dir / "final_guardrail_failure_reasons.json").write_text(
            "manual_plaintext_reason",
            encoding="utf-8",
        )
        artifact_findings, artifact_reasons = load_guardrail_artifacts(
            str(root / "guardrail_exp")
        )
        assert artifact_findings is None
        assert artifact_reasons == ["manual_plaintext_reason"]
        assert result_passed_writeup_guardrails({"status": "success"}) is True
        assert (
            result_passed_writeup_guardrails(
                {"status": "failed", "stage": "submission_bar"}
            )
            is True
        )
        assert (
            result_passed_writeup_guardrails({"status": "failed", "stage": "writeup"})
            is False
        )

        launcher_dir = root / "launcher_writeup"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        launcher_captured: dict[str, object] = {}

        def _fake_normal_writeup(**kwargs):
            launcher_captured["writeup_engine"] = "normal"
            launcher_captured["writeup_target_venue"] = kwargs.get("target_venue")
            launcher_captured["writeup_page_limit"] = kwargs.get("page_limit")
            return True

        def _fake_icbinb_writeup(**kwargs):
            raise AssertionError("journal writeup should not route to icbinb writer")

        def _fake_high_quality_pass(*args, **kwargs):
            launcher_captured["hq_target_venue"] = kwargs.get("target_venue")
            return {
                "quality_score_before": 0.0,
                "quality_score_after": 1.0,
                "rewrite_applied": False,
                "quality_gate_passed": True,
                "submission_priority_score": 95.0,
                "blocker_count": 0,
            }

        def _fake_execute_quality_workflow_with_followups(**kwargs):
            launcher_captured["hq_target_venue"] = kwargs.get("target_venue")
            return {
                "quality_result": _fake_high_quality_pass(**kwargs),
                "acceptance": {"accepted": True, "reasons": []},
                "effective_priority_bar": 90.0,
                "effective_blocker_bar": 1,
                "summary": "High-quality pass: before=0.00, after=1.00, rewrite_applied=False, gate_passed=True, priority=95.0, blockers=0",
            }

        def _fake_mark_stage_complete(*args, **kwargs):
            launcher_captured["stage_target_venue"] = (
                kwargs.get("metadata", {}) or {}
            ).get("target_venue")

        launcher_workflow = importlib.import_module(
            "ai_scientist.utils.launcher_workflow"
        )
        # The CI workflow runs unit tests before this validator. Some tests may
        # import lazy loader entrypoints and leave ai_scientist.llm in module cache.
        # Clear it here so the deferred-import smoke checks stay deterministic.
        sys.modules.pop("ai_scientist.llm", None)
        with mock.patch.object(
            launcher_workflow, "find_best_pdf_path", return_value=None
        ), mock.patch.object(
            launcher_workflow, "is_stage_complete", return_value=False
        ), mock.patch.object(
            launcher_workflow, "gather_citations", return_value=""
        ), mock.patch.object(
            launcher_workflow,
            "perform_writeup",
            side_effect=_fake_normal_writeup,
        ), mock.patch.object(
            launcher_workflow,
            "perform_icbinb_writeup",
            side_effect=_fake_icbinb_writeup,
        ), mock.patch.object(
            launcher_workflow,
            "execute_quality_workflow_with_followups",
            side_effect=_fake_execute_quality_workflow_with_followups,
        ), mock.patch.object(
            launcher_workflow, "save_token_tracker", return_value=None
        ), mock.patch.object(
            launcher_workflow,
            "mark_stage_complete",
            side_effect=_fake_mark_stage_complete,
        ):
            writeup_result = launcher_workflow.run_writeup_phase(
                launcher_dir,
                writeup_type="journal",
                writeup_retries=1,
                num_cite_rounds=1,
                model_citation="glm-4-air",
                model_writeup_small="glm-4-air",
                model_writeup="glm-4-plus",
                high_quality_mode=True,
                quality_preset="balanced",
                quality_model="glm-4-plus",
                target_venue=None,
                quality_threshold=None,
                rigor_threshold=None,
                max_quality_rewrites=1,
                autonomous_quality_followup_rounds=1,
                require_quality_gate=True,
                min_submission_priority=None,
                max_submission_blockers=None,
                writing_profile="default",
                writing_audit_rounds=0,
                strict_guardrails=False,
                guardrail_repair_rounds=1,
                research_root=root,
                resume=False,
                logger=lambda *_args, **_kwargs: None,
            )
            assert writeup_result.get("success") is True
        assert launcher_captured.get("writeup_engine") == "normal"
        assert launcher_captured.get("writeup_target_venue") == "nature"
        assert launcher_captured.get("writeup_page_limit") == 12
        assert launcher_captured.get("hq_target_venue") == "nature"
        assert launcher_captured.get("stage_target_venue") == "nature"
        launcher_workflow_smoke = "passed"

        continuous_module = importlib.import_module("continuous_paper_generator")
        continuous_captured: dict[str, object] = {"quality_calls": []}

        def _fake_get_batch_dir(batch_name: str, output_root: str | Path | None = None) -> Path:
            return root / "batch_runs" / batch_name

        def _fake_get_paper_dir(
            idea_name: str,
            paper_type: str,
            timestamp: str,
            output_root: str | Path | None = None,
        ) -> Path:
            return root / "papers" / f"{paper_type}_{idea_name}_{timestamp}"

        def _fake_create_paper_structure(paper_dir: Path) -> dict[str, Path]:
            latex_dir = paper_dir / "latex"
            reviews_dir = paper_dir / "reviews"
            latex_dir.mkdir(parents=True, exist_ok=True)
            reviews_dir.mkdir(parents=True, exist_ok=True)
            return {"root": paper_dir, "latex": latex_dir, "reviews": reviews_dir}

        class _FakeExpertSectionWriter:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def generate_detailed_outline(self, idea, experiment_results):
                return {"title": idea.get("Title", "stub")}

            def write_full_paper(self, *args, **kwargs):
                return "\\section{Introduction}\nStub content.\n"

        class _FakeLearningEngine:
            def recommend_strategy(self, **kwargs):
                return {
                    "success_probability": 0.72,
                    "confidence": 0.61,
                    "similar_papers": [{"paper_id": "demo"}],
                }

        class _FakeKnowledgeBase:
            def generate_learning_summary(self):
                return {
                    "total_papers": 1,
                    "success_rate": 1.0,
                    "success_count": 1,
                    "failure_count": 0,
                }

        class _FakeAdaptiveWriter:
            model = "adaptive-smoke-model"

        def _fake_continuous_quality_workflow(**kwargs):
            continuous_captured["quality_calls"].append(kwargs)
            return {
                "quality_result": {
                    "quality_score_after": 4.2,
                    "quality_gate_passed": True,
                    "submission_priority_score": 90.0,
                    "blocker_count": 0,
                },
                "acceptance": {"accepted": True, "reasons": []},
                "effective_priority_bar": 85.0,
                "effective_blocker_bar": 1,
                "summary": "High-quality pass: before=3.80, after=4.20, rewrite_applied=True, gate_passed=True, priority=90.0, blockers=0",
            }

        with mock.patch.object(
            continuous_module, "_ensure_runtime_imports", return_value=None
        ), mock.patch.object(
            continuous_module, "get_batch_dir", side_effect=_fake_get_batch_dir
        ), mock.patch.object(
            continuous_module, "get_paper_dir", side_effect=_fake_get_paper_dir
        ), mock.patch.object(
            continuous_module,
            "create_paper_structure",
            side_effect=_fake_create_paper_structure,
        ), mock.patch.object(
            continuous_module,
            "ExpertSectionWriter",
            _FakeExpertSectionWriter,
            create=True,
        ), mock.patch.object(
            continuous_module,
            "execute_quality_workflow_with_followups",
            side_effect=_fake_continuous_quality_workflow,
        ), mock.patch.object(
            continuous_module,
            "run_high_quality_pass",
            side_effect=AssertionError(
                "continuous generator should route quality pass through helper"
            ),
        ):
            continuous_generator = continuous_module.ContinuousPaperGenerator(
                research_dir=str(root),
                batch_name="quality_smoke",
                enable_learning=False,
            )
            professional_result = (
                continuous_generator.generate_paper_with_professional_writing(
                    {
                        "Name": "professional_smoke",
                        "Title": "Professional smoke paper",
                    },
                    paper_type="journal",
                    experiment_results={"result": "ok"},
                    model="writer-smoke-model",
                    enable_evaluation=False,
                    target_venue="nature",
                    high_quality_mode=True,
                    quality_preset="balanced",
                )
            )
            continuous_generator.enable_learning = True
            continuous_generator.learning_engine = _FakeLearningEngine()
            continuous_generator.knowledge_base = _FakeKnowledgeBase()
            continuous_generator.adaptive_writer = _FakeAdaptiveWriter()
            adaptive_result = (
                continuous_generator.generate_paper_with_adaptive_learning(
                    {
                        "Name": "adaptive_smoke",
                        "Title": "Adaptive smoke paper",
                    },
                    paper_type="journal",
                    experiment_results={"result": "ok"},
                    enable_evaluation=False,
                    learn_from_result=False,
                    target_venue="nature",
                    high_quality_mode=True,
                    quality_preset="balanced",
                )
            )

        quality_calls = continuous_captured["quality_calls"]
        assert len(quality_calls) == 2
        assert quality_calls[0]["rewrite_model"] == "writer-smoke-model"
        assert quality_calls[1]["rewrite_model"] == "adaptive-smoke-model"
        assert quality_calls[0]["target_venue"] == "nature"
        assert quality_calls[1]["target_venue"] == "nature"
        assert professional_result["quality_result"]["quality_score_after"] == 4.2
        assert adaptive_result["quality_result"]["quality_score_after"] == 4.2
        professional_dir = Path(professional_result["paper_dir"])
        adaptive_dir = Path(adaptive_result["paper_dir"])
        assert (professional_dir / "idea.json").exists()
        assert (professional_dir / "latex" / "outline.json").exists()
        assert (professional_dir / "latex" / "template.tex").exists()
        assert Path(professional_result["latex_path"]).exists()
        assert (adaptive_dir / "idea.json").exists()
        assert (adaptive_dir / "recommendation.json").exists()
        assert (adaptive_dir / "latex" / "outline.json").exists()
        assert (adaptive_dir / "latex" / "template.tex").exists()
        assert Path(adaptive_result["latex_path"]).exists()

        fallback_captured: dict[str, object] = {}

        def _fake_professional_fallback(*args, **kwargs):
            fallback_captured["kwargs"] = kwargs
            return {"status": "fallback"}

        with mock.patch.object(
            continuous_module, "_ensure_runtime_imports", return_value=None
        ), mock.patch.object(
            continuous_module.ContinuousPaperGenerator,
            "generate_paper_with_professional_writing",
            autospec=True,
            side_effect=_fake_professional_fallback,
        ):
            fallback_generator = continuous_module.ContinuousPaperGenerator(
                research_dir=str(root),
                batch_name="quality_smoke_fallback",
                enable_learning=False,
            )
            fallback_result = fallback_generator.generate_paper_with_adaptive_learning(
                {
                    "Name": "fallback_smoke",
                    "Title": "Fallback smoke paper",
                },
                paper_type="journal",
                experiment_results={"result": "ok"},
                enable_evaluation=False,
                learn_from_result=False,
                target_venue="nature",
                high_quality_mode=True,
                quality_preset="publishable",
                writing_profile="logic_first",
            )

        assert fallback_result["status"] == "fallback"
        fallback_kwargs = fallback_captured["kwargs"]
        assert fallback_kwargs["target_venue"] == "nature"
        assert fallback_kwargs["high_quality_mode"] is True
        assert fallback_kwargs["quality_preset"] == "publishable"
        assert fallback_kwargs["writing_profile"] == "logic_first"
        continuous_quality_smoke = "passed"

        launch_args = argparse.Namespace(
            writing_profile="reviewer-strict",
            writing_audit_rounds=-2,
            guardrail_repair_rounds=-4,
            submission_mode=True,
            breakthrough_mode=False,
            target_venue=None,
            writeup_type="journal",
            high_quality_mode=False,
            require_quality_gate=False,
            autonomous_quality_followup_rounds=0,
            auto_best_idea=False,
            fallback_ranked_ideas=False,
            auto_adjust_paper_type=False,
            quality_preset="balanced",
            strict_writing_guardrails=False,
            min_submission_priority=None,
            max_submission_blockers=None,
            max_ranked_candidates=None,
        )
        invalid_profiles: list[str] = []
        normalize_common_launcher_args(
            launch_args,
            invalid_profile_logger=lambda exc: invalid_profiles.append(str(exc)),
        )
        assert launch_args.writing_profile == "reviewer_strict"
        assert launch_args.writing_audit_rounds == 1
        assert launch_args.guardrail_repair_rounds == 1
        assert launch_args.high_quality_mode is True
        assert launch_args.require_quality_gate is True
        assert launch_args.autonomous_quality_followup_rounds == 1
        assert launch_args.target_venue == "nature"
        assert launch_args.max_ranked_candidates == 5
        assert invalid_profiles == []

        breakthrough_args = argparse.Namespace(
            writing_profile="missing-profile",
            writing_audit_rounds=0,
            guardrail_repair_rounds=0,
            submission_mode=False,
            breakthrough_mode=True,
            target_venue=None,
            writeup_type="icbinb",
            high_quality_mode=False,
            require_quality_gate=False,
            autonomous_quality_followup_rounds=0,
            auto_best_idea=False,
            fallback_ranked_ideas=False,
            auto_adjust_paper_type=False,
            quality_preset="balanced",
            strict_writing_guardrails=False,
            min_submission_priority=None,
            max_submission_blockers=None,
            max_ranked_candidates=None,
        )
        normalize_common_launcher_args(
            breakthrough_args,
            invalid_profile_logger=lambda exc: invalid_profiles.append(str(exc)),
        )
        assert breakthrough_args.writing_profile == "default"
        assert breakthrough_args.target_venue == "nature"
        assert breakthrough_args.max_ranked_candidates == 7
        assert breakthrough_args.quality_preset == "publishable"
        assert breakthrough_args.autonomous_quality_followup_rounds == 1
        assert len(invalid_profiles) == 1

        project_args = argparse.Namespace(
            writing_profile="humanized-academic",
            writing_audit_rounds=-3,
            guardrail_repair_rounds=-2,
            submission_mode=True,
            breakthrough_mode=False,
            target_venue=None,
            writeup_type="journal",
            rank_ideas=False,
            fallback_ranked_ideas=False,
            top_k_ideas=None,
            high_quality_mode=False,
            require_quality_gate=False,
            autonomous_quality_followup_rounds=0,
            auto_adjust_paper_type=False,
            quality_preset="balanced",
            strict_writing_guardrails=False,
            min_submission_priority=None,
            max_submission_blockers=None,
        )
        normalize_project_workflow_args(project_args)
        assert project_args.writing_profile == "humanized_academic"
        assert project_args.target_venue == "nature"
        assert project_args.rank_ideas is True
        assert project_args.fallback_ranked_ideas is True
        assert project_args.top_k_ideas == 5
        assert project_args.autonomous_quality_followup_rounds == 1

        batch_args = argparse.Namespace(
            writing_profile="logic-first",
            writing_audit_rounds=0,
            guardrail_repair_rounds=0,
            submission_mode=True,
            breakthrough_mode=False,
            target_venue=None,
            paper_types=["icbinb"],
            all_types=True,
            rank_ideas=False,
            top_k_ideas=None,
            high_quality_mode=False,
            require_quality_gate=False,
            autonomous_quality_followup_rounds=0,
            auto_adjust_paper_type=False,
            quality_preset="balanced",
            strict_writing_guardrails=False,
            min_submission_priority=None,
            max_submission_blockers=None,
        )
        normalize_batch_workflow_args(batch_args)
        assert batch_args.writing_profile == "logic_first"
        assert batch_args.target_venue == "nature"
        assert batch_args.rank_ideas is True
        assert batch_args.top_k_ideas == 5
        assert batch_args.autonomous_quality_followup_rounds == 1
        assert (
            recommend_default_target_venue(
                paper_types=["icbinb"],
                all_types=True,
            )
            == "nature"
        )
        writeup_plan = build_writeup_execution_plan(
            "journal",
            num_cite_rounds=1,
            writeup_retries=1,
            target_venue=None,
            high_quality_mode=True,
            research_root=root,
        )
        assert writeup_plan["target_venue"] == "nature"
        assert writeup_plan["page_limit"] == 12
        assert writeup_plan["writeup_engine"] == "normal"
        assert writeup_plan["num_cite_rounds"] >= 20
        assert writeup_plan["writeup_retries"] >= 4
        assert resolve_writeup_engine("extended") == "icbinb"
        assert resolve_page_limit("extended") == 2
        quality_log: list[str] = []
        quality_pass = execute_quality_workflow(
            run_high_quality_pass_fn=lambda *args, **kwargs: {
                "target_venue": kwargs.get("target_venue"),
                "quality_score_before": 0.4,
                "quality_score_after": 0.92,
                "rewrite_applied": True,
                "quality_gate_passed": True,
                "submission_priority_score": 95.0,
                "submission_priority_tier": "submit_now",
                "blocker_count": 0,
            },
            run_dir=str(root),
            paper_type="journal",
            rewrite_model="glm-4-plus",
            quality_model="glm-4-plus",
            target_venue="nature",
            quality_preset="balanced",
            quality_threshold=None,
            rigor_threshold=None,
            max_quality_rewrites=2,
            require_quality_gate=True,
            min_submission_priority=None,
            max_submission_blockers=None,
            resume=False,
            logger=quality_log.append,
        )
        assert quality_pass["acceptance"]["accepted"] is True
        assert quality_pass["effective_priority_bar"] is not None
        assert "gate_passed=True" in quality_pass["summary"]
        assert "priority=95.0" in format_quality_pass_summary(
            quality_pass["quality_result"]
        )
        followup_calls: list[dict[str, object]] = []
        derived_focus = derive_autonomous_followup_focus(
            quality_result={
                "quality_gate_passed": False,
                "blocker_count": 4,
                "revision_actions": [
                    {
                        "priority": "P0",
                        "focus": "Claim support",
                        "action": "Anchor claims to figures and numbers",
                        "reason": "unsupported claims remain",
                    },
                    {
                        "priority": "P1",
                        "focus": "Numerical specificity",
                        "action": "Inject quantitative deltas into abstract and conclusion",
                        "reason": "numeric coverage below target",
                    },
                ],
                "submission_readiness": {
                    "status": "needs_work",
                    "categories": {"claim": 2, "numeric": 1, "quality": 1},
                },
                "submission_scorecard": {
                    "claim_support": {"gap": 0.7},
                    "numeric_coverage": {"gap": 0.4},
                    "quality": {"gap": 0.3},
                },
                "submission_priority_reasons": [
                    "largest remaining gap: claim_support (0.70)"
                ],
            },
            acceptance={
                "accepted": False,
                "reasons": [
                    "quality gate requirement not met",
                    "unsupported claims still detected: 2",
                ],
            },
        )
        assert "claim_support" in derived_focus["focus_areas"]
        assert "numeric_coverage" in derived_focus["focus_areas"]
        assert derived_focus["frontmatter_required"] is True
        assert derived_focus["candidate_boost"] >= 2
        assert "abstract" in derived_focus["preferred_sections"]
        quality_followup = execute_quality_workflow_with_followups(
            run_high_quality_pass_fn=lambda *args, **kwargs: (
                followup_calls.append(dict(kwargs))
                or {
                    "target_venue": kwargs.get("target_venue"),
                    "quality_score_before": 0.4 if len(followup_calls) == 1 else 0.92,
                    "quality_score_after": 0.78 if len(followup_calls) == 1 else 0.95,
                    "rewrite_applied": True,
                    "quality_gate_passed": False if len(followup_calls) == 1 else True,
                    "submission_priority_score": (
                        70.0 if len(followup_calls) == 1 else 92.0
                    ),
                    "submission_priority_tier": (
                        "revise" if len(followup_calls) == 1 else "submit_now"
                    ),
                    "blocker_count": 3 if len(followup_calls) == 1 else 0,
                    "submission_readiness": {
                        "status": "draft" if len(followup_calls) == 1 else "ready",
                        "categories": (
                            {"claim": 2, "numeric": 1, "quality": 1}
                            if len(followup_calls) == 1
                            else {}
                        ),
                    },
                    "rewrite_effectiveness_summary": {
                        "priority_gain_total": 0.2 if len(followup_calls) == 1 else 1.2
                    },
                    "revision_actions": (
                        [
                            {
                                "priority": "P0",
                                "focus": "Claim support",
                                "action": "Anchor claims to figures and numbers",
                                "reason": "unsupported claims remain",
                            },
                            {
                                "priority": "P1",
                                "focus": "Numerical specificity",
                                "action": "Inject quantitative deltas into abstract and conclusion",
                                "reason": "numeric coverage below target",
                            },
                        ]
                        if len(followup_calls) == 1
                        else []
                    ),
                    "submission_scorecard": (
                        {
                            "claim_support": {"gap": 0.7},
                            "numeric_coverage": {"gap": 0.4},
                            "quality": {"gap": 0.3},
                        }
                        if len(followup_calls) == 1
                        else {}
                    ),
                }
            ),
            run_dir=str(root),
            paper_type="journal",
            rewrite_model="glm-4-plus",
            quality_model="glm-4-plus",
            target_venue="nature",
            quality_preset="balanced",
            quality_threshold=None,
            rigor_threshold=None,
            max_quality_rewrites=1,
            require_quality_gate=True,
            min_submission_priority=None,
            max_submission_blockers=None,
            autonomous_followup_rounds=2,
            resume=False,
            logger=quality_log.append,
        )
        assert len(followup_calls) == 2
        assert followup_calls[1]["max_rewrite_rounds"] >= 2
        assert isinstance(followup_calls[1]["autonomous_followup_focus"], dict)
        assert (
            "claim_support"
            in followup_calls[1]["autonomous_followup_focus"]["focus_areas"]
        )
        assert (
            followup_calls[1]["autonomous_followup_focus"]["frontmatter_required"]
            is True
        )
        assert quality_followup["acceptance"]["accepted"] is True
        assert quality_followup["quality_result"]["autonomous_followup_rounds_run"] == 1
        assert quality_followup["quality_result"]["autonomous_followup_applied"] is True
        assert (
            quality_followup["autonomous_followup_history"][0]["plan"][
                "autonomous_followup_focus"
            ]["candidate_boost"]
            >= 2
        )
        review_plan = build_review_execution_plan(
            "journal",
            target_venue="nature",
            review_reflections=1,
            review_ensemble=1,
            review_fewshot=1,
            review_temperature=0.9,
            high_quality_mode=True,
            research_root=root,
        )
        assert review_plan["strategy"] == ReviewStrategy.NATURE
        assert review_plan["review_reflections"] >= 2
        assert review_plan["review_ensemble"] >= 3
        assert review_plan["review_fewshot"] >= 2
        assert review_plan["review_temperature"] <= 0.65
        assert (
            resolve_review_strategy(
                "normal",
                target_venue="neurips",
                review_strategy="depth",
                high_quality_mode=False,
            )
            == ReviewStrategy.DEPTH
        )

        selection_log: list[str] = []
        helper_ideas = [
            {"Name": "idea_0"},
            {"Name": "idea_1"},
            {"Name": "idea_2"},
            {"Name": "idea_3"},
            {"Name": "idea_4"},
        ]
        selected_indices, helper_rankings = select_ranked_idea_candidates(
            helper_ideas,
            ranking_enabled=True,
            ranking_model="glm-4-air",
            target_venue="nature",
            prioritize_breakthrough=True,
            research_root=root,
            ranking_output_path=root / "rankings.json",
            requested_indices=[4, 2, 99],
            limit=3,
            ranker=lambda ideas, **kwargs: [
                {"idea_idx": 4, "ranking_score": 5.0},
                {"idea_idx": 1, "ranking_score": 4.5},
                {"idea_idx": 2, "ranking_score": 4.0},
                {"idea_idx": 3, "ranking_score": 3.5},
            ],
        )
        assert selected_indices == [4, 2]
        assert helper_rankings[0]["idea_idx"] == 4
        selected_defaults, _ = select_ranked_idea_candidates(
            helper_ideas,
            ranking_enabled=False,
            ranking_model=None,
            default_indices=[3, 1, 3],
        )
        assert selected_defaults == [3, 1]
        resolved_paper_type = resolve_paper_type_for_venue(
            "icbinb",
            "nature",
            auto_adjust=True,
            logger=selection_log.append,
        )
        assert resolved_paper_type == "journal"
        resolved_paper_types = resolve_paper_types_for_venue(
            ["icbinb", "journal", "normal", "journal"],
            "nature",
            auto_adjust=True,
            logger=selection_log.append,
        )
        assert resolved_paper_types == ["journal", "normal"]
        assert len(selection_log) >= 2

        for script_name in ["run_project.py", "continuous_paper_generator.py"]:
            help_result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script_name), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            assert help_result.returncode == 0, help_result.stderr
            assert "usage:" in help_result.stdout.lower()
        for module_name in ["run_project", "continuous_paper_generator"]:
            importlib.import_module(module_name)
            assert "ai_scientist.llm" not in sys.modules
        cli_help_smoke = "passed"

        instruction = generate_review_instruction(ReviewStrategy.DEPTH)
        rigor = assess_experiment_rigor(
            root, latex_content="baseline ablation p-value hyperparameter"
        )
        claim_support = assess_claim_support(
            """\\begin{abstract}We show significant improvements over baselines.\\end{abstract}
            \\section{Conclusion}Our method outperforms prior work; see Figure 1 and Table 1.\\cite{foo}"""
        )
        numeric_coverage = assess_numeric_coverage(
            """\\begin{abstract}Accuracy improved to 84.7 and error dropped by 5.2%.\\end{abstract}
            \\section{Conclusion}We achieved 84.7 with p-value 0.03.\\end{document}"""
        )
        evidence = build_evidence_pack(
            root,
            """\\begin{figure}\\includegraphics{figures/a.png}\\caption{Main result}\\label{fig:main}\\end{figure}
            \\begin{table}\\caption{Ablation}\\label{tab:abl}\\end{table}
            We refer to \\ref{fig:main} and \\ref{tab:abl}.""",
        )
        key_results = extract_key_results(
            root, "Accuracy improved from 81.2 to 84.7 with 5.2% gain and p-value 0.03"
        )
        ledger = build_claim_evidence_ledger(
            """\\begin{abstract}We show significant improvements over baselines.\\end{abstract}
            \\section{Conclusion}Our method outperforms prior work; see Figure 1 and Table 1.\\cite{foo}"""
        )
        contribution_map = build_contribution_map(
            {
                "claim_ledger": ledger,
                "evidence_pack": evidence,
                "key_results": key_results,
                "claim_support": claim_support,
                "rigor": rigor,
            }
        )
        claim_alignment = assess_claim_alignment(ledger)
        assert "审查要求" in instruction
        assert rigor["score"] >= 4
        assert claim_support["score"] >= 3
        assert numeric_coverage["score"] >= 4
        assert evidence["num_figures"] == 1
        assert evidence["num_tables"] == 1
        assert evidence["evidence_density_score"] >= 2
        assert key_results["count"] >= 3
        assert isinstance(key_results.get("structured_results"), list)
        assert len(ledger) >= 1
        assert len(contribution_map) >= 1
        assert claim_alignment["score"] >= 1
        guardrail_findings = collect_guardrail_findings(
            """\\section{Introduction}
We cite \\cite{missing_key_2024} but no matching bib entry exists.
\\section{Experiments}
Results are preliminary.
""",
            "iclr",
        )
        guardrail_reasons = list_blocking_guardrail_reasons(
            guardrail_findings,
            allow_placeholder_citations=False,
            require_venue_sections=True,
        )
        guardrail_plan = build_guardrail_repair_plan(guardrail_findings, "iclr")
        assert any(
            reason.startswith("missing_bibtex_keys=") for reason in guardrail_reasons
        )
        assert "Guardrail repair plan" in guardrail_plan
        assert "missing_key_2024" in guardrail_plan
        assert resolve_target_venue("journal", None) == "nature"
        assert (
            recommend_target_venue_from_idea(
                {"Title": "Real-world climate challenge", "Abstract": "broad impact"},
                "normal",
            )
            == "nature"
        )
        assert "nature" in VENUE_PRESETS
        min_priority, max_blockers = resolve_submission_acceptance_settings("nature")
        acceptance = evaluate_submission_acceptance(
            {
                "quality_gate_passed": True,
                "submission_priority_score": 91.0,
                "submission_priority_tier": "submit_now",
                "blocker_count": 1,
            },
            require_quality_gate=True,
            min_submission_priority=min_priority,
            max_submission_blockers=max_blockers,
        )
        assert acceptance["accepted"] is True
        history_dir = root / "papers" / "normal" / "20240101_hist_idea"
        (history_dir / "quality").mkdir(parents=True, exist_ok=True)
        (history_dir / "idea.json").write_text(
            json.dumps(
                {
                    "Name": "hist_climate",
                    "Title": "Climate-adaptive system for real-world robustness",
                    "Abstract": "A broad-impact climate challenge with strong real-world significance.",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (history_dir / "quality" / "high_quality_result.json").write_text(
            json.dumps(
                {
                    "target_venue": "nature",
                    "quality_gate_passed": True,
                    "submission_priority_score": 91.0,
                    "submission_priority_tier": "submit_now",
                    "submission_status": "ready",
                    "blocker_count": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        rankings = rank_ideas(
            [
                {
                    "Name": "idea_a",
                    "Title": "Climate-adaptive discovery system",
                    "Short Hypothesis": "real-world climate challenge with broad impact",
                },
                {
                    "Name": "idea_b",
                    "Title": "Toy benchmark tweak",
                    "Short Hypothesis": "minor benchmark change",
                },
            ],
            model="offline-smoke",
            target_venue="nature",
            prioritize_breakthrough=True,
            research_root=root,
        )
        strategy_feedback = recommend_submission_strategy_adjustments(
            "nature", research_root=root
        )
        rewrite_focus = recommend_rewrite_focus_adjustments(
            "nature", research_root=root
        )
        reviewer_risks = recommend_reviewer_risk_mitigation(
            "nature", research_root=root
        )
        section_guidance = recommend_section_reviewer_language_guidance(
            "nature", "abstract", research_root=root
        )
        style_preferences = recommend_rewrite_style_preferences(
            "nature", research_root=root
        )
        rewrite_efficiency = recommend_rewrite_efficiency_controls(
            "nature", research_root=root
        )
        local_section_guidance = _derive_section_risk_language_guidance(
            "abstract", {"historical_reviewer_risks": reviewer_risks}
        )
        assert len(rankings) == 2
        assert "ranking_score" in rankings[0]
        assert "historical_acceptance_adjustment" in rankings[0]
        assert isinstance(strategy_feedback.get("effective_defaults"), dict)
        assert isinstance(rewrite_focus.get("dimension_boosts"), dict)
        assert isinstance(reviewer_risks.get("anticipated_objections"), list)
        assert isinstance(section_guidance.get("claim_softening"), list)
        assert isinstance(local_section_guidance.get("claim_softening"), list)
        assert isinstance(style_preferences.get("frontmatter_style_order"), list)
        assert isinstance(rewrite_efficiency.get("preferred_sections"), list)

        artifact_report = {
            "claim_ledger": ledger,
            "claim_support": {
                "unsupported_claims": ["Overstated novelty in abstract"],
            },
            "evidence_pack": {
                "strongest_results": [
                    {
                        "type": "figure",
                        "label": "fig:main",
                        "caption": "Main result",
                        "ref_count": 3,
                    },
                    {
                        "type": "table",
                        "label": "tab:abl",
                        "caption": "Ablation",
                        "ref_count": 2,
                    },
                ],
                "evidence_density_score": 2,
            },
            "key_results": {
                "structured_results": [
                    {"source": "result.json", "path": "accuracy", "value": 84.7}
                ]
            },
            "contribution_map": contribution_map,
            "submission_scorecard": {
                "quality": {"score": 4.2, "threshold": 4.1, "pass": True, "gap": 0.0}
            },
            "historical_reviewer_risks": {
                "anticipated_objections": ["Need stronger ablation justification"],
                "rebuttal_focus": ["Clarify why the main comparison is fair"],
                "claim_softening_advice": [
                    "Prefer measured novelty language in the abstract."
                ],
            },
            "historical_style_preferences": {
                "frontmatter_style_order": ["evidence_first"],
                "section_style_order": ["results_first"],
            },
            "revision_actions": [
                {
                    "priority": "P0",
                    "focus": "experiments",
                    "action": "Add stronger ablation analysis",
                    "reason": "Reviewer may question causal contribution",
                }
            ],
            "target_venue": "nature",
        }
        artifact_result = {
            "target_venue": "nature",
            "claim_alignment_after": 3.4,
            "claim_support_after": 3.8,
            "unsupported_claims_count": 1,
            "submission_readiness": {
                "status": "revise_before_submission",
                "blockers": ["Need clearer causal evidence"],
            },
            "submission_scorecard": artifact_report["submission_scorecard"],
            "submission_priority_score": 88.0,
            "submission_priority_tier": "submit_now",
            "quality_gate_passed": True,
            "rigor_score_after": 4.0,
            "numeric_coverage_after": 4.2,
            "evidence_density_score": 2.0,
            "revision_actions": artifact_report["revision_actions"],
            "historical_reviewer_risks": artifact_report["historical_reviewer_risks"],
            "historical_style_preferences": artifact_report[
                "historical_style_preferences"
            ],
        }
        assert "Logic Check Report" in _build_logic_check_text(
            artifact_report, artifact_result
        )
        assert "Reviewer Gate Report" in _build_reviewer_gate_report_text(
            artifact_report, artifact_result
        )
        assert "Experiment Analysis" in _build_experiment_analysis_text(
            artifact_report, artifact_result
        )
        assert (
            "Experiment Visualization Brief"
            in _build_experiment_visualization_brief_text(
                artifact_report, artifact_result
            )
        )
        assert "Figure Caption Guidance" in _build_figure_caption_guidance_text(
            artifact_report, artifact_result
        )
        assert "Table Caption Guidance" in _build_table_caption_guidance_text(
            artifact_report, artifact_result
        )
        assert "Architecture Figure Brief" in _build_architecture_figure_brief_text(
            artifact_report, artifact_result
        )
        assert "Humanizer Style Notes" in _build_humanizer_style_notes_text(
            artifact_report, artifact_result
        )
        assert "Writing Skill Pack" in _build_writing_skill_pack_text(
            artifact_report, artifact_result
        )
        plot_guidance_dir = root / "plot_guidance_case" / "quality"
        plot_guidance_dir.mkdir(parents=True, exist_ok=True)
        (plot_guidance_dir / "experiment_visualization_brief.md").write_text(
            "# Experiment Visualization Brief\n- Plot the main result first.\n",
            encoding="utf-8",
        )
        from ai_scientist.perform_plotting import (
            build_aggregator_prompt,
            load_quality_plot_guidance,
        )
        from continuous_paper_generator import (
            _build_batch_experiment_agenda,
            _build_batch_experiment_agenda_markdown,
            _build_batch_experiment_ledger_rows,
            _build_batch_experiment_ledger_tsv,
        )

        plot_guidance_text = load_quality_plot_guidance(str(plot_guidance_dir.parent))
        assert "Experiment Visualization Brief" in plot_guidance_text
        aggregator_prompt = build_aggregator_prompt(
            "{}",
            "idea text",
            plot_guidance_text,
            "# Figure Spec\n- Figure 1: Main result.\n",
        )
        assert "QUALITY / VISUALIZATION GUIDANCE" in aggregator_prompt
        assert "Plot the main result first" in aggregator_prompt
        assert "STRUCTURED FIGURE SPEC" in aggregator_prompt

        experiment_report = {
            "completed_papers": [
                {
                    "status": "success",
                    "idea_idx": 0,
                    "idea_name": "quality_source_paper",
                    "paper_type": "journal",
                    "target_venue": "nature",
                    "submission_acceptance_passed": True,
                    "quality_gate_passed": True,
                    "submission_priority_score": 92.0,
                    "blocker_count": 1,
                    "unsupported_claims_count": 0,
                    "evidence_density_score": 2.4,
                    "revision_actions": [
                        {
                            "priority": "P0",
                            "focus": "experiments",
                            "action": "Run a stronger ablation on the key mechanism.",
                            "reason": "Need causal evidence.",
                        }
                    ],
                }
            ],
            "failed_papers": [
                {
                    "status": "failed",
                    "idea_idx": 1,
                    "idea_name": "failed_paper",
                    "paper_type": "journal",
                    "stage": "experiment",
                }
            ],
            "quality_summary": {
                "top_papers": [
                    {
                        "idea_idx": 0,
                        "idea_name": "quality_source_paper",
                        "paper_type": "journal",
                        "target_venue": "nature",
                        "submission_priority_score": 92.0,
                        "blocker_count": 1,
                        "quality_gate_passed": True,
                        "submission_acceptance_passed": True,
                        "unsupported_claims_count": 0,
                        "evidence_density_score": 2.4,
                        "revision_actions": [
                            {
                                "priority": "P0",
                                "focus": "experiments",
                                "action": "Run a stronger ablation on the key mechanism.",
                                "reason": "Need causal evidence.",
                            }
                        ],
                    }
                ]
            },
        }
        experiment_rows = _build_batch_experiment_ledger_rows(experiment_report)
        assert any(row.get("decision") == "keep" for row in experiment_rows)
        assert any(row.get("decision") == "crash" for row in experiment_rows)
        ledger_tsv = _build_batch_experiment_ledger_tsv(experiment_rows)
        assert "decision" in ledger_tsv
        experiment_agenda = _build_batch_experiment_agenda(experiment_report)
        assert experiment_agenda.get("priority_experiments")
        assert "Batch Experiment Agenda" in _build_batch_experiment_agenda_markdown(
            experiment_agenda
        )
        todo_paper_dir = root / "todo_rule_case"
        todo_paper_dir.mkdir(parents=True, exist_ok=True)
        (todo_paper_dir / "self_review_iteration_summary.json").write_text(
            json.dumps(
                {
                    "latest_round_gate": {
                        "ready": False,
                        "score": 74.0,
                        "reasons": [
                            "critical_issues_unresolved",
                            "high_value_coverage_low",
                        ],
                        "next_focus_summaries": ["Need stronger ablation"],
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        todo_tasks = _build_paper_experiment_todo_tasks(
            {
                "paper_dir": str(todo_paper_dir),
                "idea_name": "todo_rule_paper",
                "idea_idx": 2,
                "revision_actions": [
                    {
                        "priority": "P1",
                        "focus": "experiments",
                        "action": "Add ablation over key component.",
                        "reason": "Need stronger evidence for reviewer concerns.",
                    }
                ],
                "unsupported_claims_count": 1,
            }
        )
        assert todo_tasks
        assert all(
            isinstance(task.get("completion_rule"), str) and task.get("completion_rule")
            for task in todo_tasks
        )
        assert any(
            str(task.get("completion_rule")).startswith("gate_reason_cleared:")
            for task in todo_tasks
        )
        bootstrapped_tasks = bootstrap_todo_tasks_from_round_gate(
            {
                "reasons": ["critical_issues_unresolved"],
                "next_focus_summaries": ["Need stronger ablation"],
            },
            prefix="validate",
        )
        assert bootstrapped_tasks
        assert all(task.get("completion_rule") for task in bootstrapped_tasks)
        todo_payload = {
            "tasks": [
                {
                    "task_id": "T01",
                    "priority": "P0",
                    "action": "Resolve unresolved critical issues",
                    "source": "self_review_round_gate",
                    "source_signal": "critical_issues_unresolved",
                    "completion_rule": "gate_reason_cleared:critical_issues_unresolved",
                },
                {
                    "task_id": "T02",
                    "priority": "P1",
                    "action": "Close next focus item",
                    "source": "self_review_next_focus",
                    "source_signal": "Need stronger ablation",
                    "completion_rule": "next_focus_cleared:Need stronger ablation",
                },
                {
                    "task_id": "T03",
                    "priority": "P1",
                    "action": "Raise high-value coverage",
                    "source": "evidence_metrics",
                    "source_signal": "high_value_coverage",
                    "completion_rule": "high_value_coverage_ge:0.8",
                },
            ]
        }
        blocked_todo_snapshot = evaluate_todo_progress_snapshot(
            todo_payload,
            round_gate={
                "ready": False,
                "reasons": ["critical_issues_unresolved"],
                "next_focus_summaries": ["Need stronger ablation"],
                "metrics": {"high_value_coverage_ratio": 0.55},
            },
            issue_progress={"unresolved_critical_count": 1},
            round_index=1,
        )
        assert blocked_todo_snapshot["counts"]["closed_tasks"] == 0
        assert blocked_todo_snapshot["counts"]["unresolved_tasks"] == 3
        cleared_todo_snapshot = evaluate_todo_progress_snapshot(
            todo_payload,
            round_gate={
                "ready": True,
                "reasons": [],
                "next_focus_summaries": [],
                "metrics": {"high_value_coverage_ratio": 0.91},
            },
            issue_progress={"unresolved_critical_count": 0},
            round_index=2,
        )
        assert cleared_todo_snapshot["counts"]["closed_tasks"] == 3
        assert cleared_todo_snapshot["counts"]["unresolved_tasks"] == 0
        assert cleared_todo_snapshot["closure_rate"] == 1.0
        assert cleared_todo_snapshot["p0_closure_rate"] == 1.0

        rewrite_policy_args = argparse.Namespace(
            rewrite_followup_preset="publishable",
            rewrite_followup_quality_threshold=None,
            rewrite_followup_rigor_threshold=None,
            rewrite_followup_max_rounds=None,
            adaptive_rewrite_followup=True,
            rewrite_followup_skip_blocker_threshold=6,
            rewrite_followup_blocker_reduction_threshold=4,
            rewrite_followup_ready_max_rounds=1,
            rewrite_followup_publishable_priority_threshold=85.0,
            rewrite_followup_publishable_gain_threshold=1.25,
        )
        ready_policy = _derive_rewrite_followup_policy(
            rewrite_policy_args,
            {
                "submission_status": "ready",
                "blocker_count": 1,
                "submission_priority_score": 92.0,
            },
        )
        blocker_policy = _derive_rewrite_followup_policy(
            rewrite_policy_args,
            {
                "submission_status": "draft",
                "blocker_count": 6,
                "submission_priority_score": 60.0,
            },
        )
        push_policy = _derive_rewrite_followup_policy(
            rewrite_policy_args,
            {
                "submission_status": "draft",
                "blocker_count": 1,
                "submission_priority_score": 88.0,
                "rewrite_priority_gain_total": 1.4,
            },
        )
        todo_pressure_policy = _derive_rewrite_followup_policy(
            rewrite_policy_args,
            {
                "submission_status": "draft",
                "blocker_count": 2,
                "submission_priority_score": 70.0,
                "experiment_todo_count": 5,
                "experiment_todo_p0_count": 1,
                "experiment_todo_closure_rate": 0.2,
            },
        )
        assert ready_policy["mode"] == "final_polish"
        assert ready_policy["max_rewrite_rounds"] == 1
        assert blocker_policy["skip"] is True
        assert blocker_policy["mode"] == "skip_high_blockers"
        assert push_policy["mode"] == "submission_push"
        assert push_policy["preset_name"] == "publishable"
        assert todo_pressure_policy["mode"] == "evidence_gap_repair"
        assert "experiment TODO backlog" in str(todo_pressure_policy.get("reason"))
        assert int(todo_pressure_policy["max_rewrite_rounds"]) >= 2

        paper_dir = root / "papers" / "paper_20240101_000000_hist_climate_normal"
        (paper_dir / "quality").mkdir(parents=True, exist_ok=True)
        (paper_dir / "idea.json").write_text(
            json.dumps(
                {
                    "Name": "hist_climate",
                    "Title": "Climate-adaptive system for real-world robustness",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (paper_dir / "paper.pdf").write_text("pdf", encoding="utf-8")
        (paper_dir / "quality" / "submission_package.md").write_text(
            "# Submission Package\n", encoding="utf-8"
        )
        (paper_dir / "quality" / "narrative_map.md").write_text(
            "# Narrative Map\n", encoding="utf-8"
        )
        (paper_dir / "quality" / "high_quality_result.json").write_text(
            json.dumps(
                {
                    "target_venue": "nature",
                    "quality_gate_passed": True,
                    "submission_priority_score": 91.0,
                    "submission_priority_tier": "submit_now",
                    "submission_readiness": {"status": "ready"},
                    "rewrite_effectiveness_summary": {"priority_gain_total": 1.4},
                    "blocker_count": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (paper_dir / "process_alignment.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "overall_score": 92.0,
                        "ready_process_count": 7,
                        "blocked_process_count": 0,
                        "needs_attention_process_count": 0,
                        "missing_process_count": 0,
                        "top_process_risks": {},
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (paper_dir / "experiment_todo.json").write_text(
            json.dumps(
                {
                    "counts": {"total_tasks": 2, "p0_tasks": 1},
                    "tasks": [
                        {
                            "task_id": "T01",
                            "priority": "P0",
                            "action": "Close critical evidence gap",
                        },
                        {
                            "task_id": "T02",
                            "priority": "P1",
                            "action": "Expand ablation coverage",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (paper_dir / "experiment_todo_progress.json").write_text(
            json.dumps(
                {
                    "final_snapshot": {
                        "closure_rate": 0.5,
                        "p0_closure_rate": 1.0,
                        "counts": {
                            "closed_tasks": 1,
                            "unresolved_tasks": 1,
                            "p0_closed": 1,
                            "p0_unresolved": 0,
                        },
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        run_index_entry = infer_run_entry(paper_dir)
        assert run_index_entry["experiment_todo_count"] == 2
        assert run_index_entry["experiment_todo_p0_count"] == 1
        assert run_index_entry["experiment_todo_closed_count"] == 1
        assert run_index_entry["experiment_todo_unresolved_count"] == 1
        assert run_index_entry["experiment_todo_closure_rate"] == 0.5
        assert run_index_entry["experiment_todo_p0_closure_rate"] == 1.0
        assert run_index_entry.get("experiment_todo_progress_file")

        submission_daemon_dir = root / "daemon_runs" / "submission_validate"
        submission_daemon_dir.mkdir(parents=True, exist_ok=True)
        submission_manager = ResearchManager(root)
        submission_manager.rebuild_index()
        submission_args = argparse.Namespace(
            auto_export_submission_dossier=True,
            auto_submission_dossier_top_k=1,
            auto_submission_dossier_min_priority=None,
            auto_submission_dossier_max_blockers=None,
            auto_submission_dossier_min_rewrite_gain=None,
            auto_submission_dossier_require_gate=True,
            auto_submission_dossier_require_ready=True,
            shortlist_venue="nature",
            shortlist_min_priority=80.0,
            shortlist_max_blockers=2,
            shortlist_min_rewrite_gain=0.0,
        )
        submission_status = {"cycle": 1}
        submission_brief = _apply_submission_autopilot(
            submission_status,
            submission_daemon_dir,
            submission_manager,
            {},
            submission_args,
        )
        submission_autopilot = submission_brief.get("submission_autopilot") or {}
        assert submission_autopilot.get("exported")
        exported_item = submission_autopilot["exported"][0]
        assert Path(exported_item["output_dir"]).exists()
        assert (Path(exported_item["output_dir"]) / "dossier_manifest.json").exists()

        failure_guard_daemon_dir = root / "daemon_runs" / "failure_guard_validate"
        failure_guard_daemon_dir.mkdir(parents=True, exist_ok=True)
        failure_guard_args = argparse.Namespace(
            auto_failure_guard=True,
            auto_failure_guard_threshold=2,
            auto_failure_guard_cooldown_cycles=3,
        )
        failure_guard_status = {
            "cycle": 3,
            "last_returncode": 1,
            "active_source": {"name": "failing_source"},
            "active_source_key": "topic::failing_source::topic.md",
            "source_runtime": {
                "topic::failing_source::topic.md": {"consecutive_failures": 2}
            },
        }
        failure_guard_status = _apply_failure_guard(
            failure_guard_status,
            failure_guard_daemon_dir,
            failure_guard_args,
        )
        failure_guard_summary = failure_guard_status.get("last_failure_guard") or {}
        assert failure_guard_summary.get("applied") is True
        failure_guard_control = json.loads(
            (failure_guard_daemon_dir / "daemon_control.json").read_text(
                encoding="utf-8"
            )
        )
        cooldown_command = (failure_guard_control.get("source_commands") or {}).get(
            "failing_source"
        )
        assert cooldown_command is not None
        assert cooldown_command.get("cooldown_cycles_once") == 3

        provenance_path = paper_dir / "source_provenance.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "batch_name": "batch_quality_loop",
                    "daemon_name": "quality_loop_daemon",
                    "source_name": "quality_source",
                    "source_key": "topic::quality_source::topic.md",
                    "source_type": "topic",
                    "source_value": "topic.md",
                    "source_target_venue": "nature",
                    "source_paper_types": ["normal"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        submission_manager.rebuild_index()
        papers = submission_manager.list_papers(sort_by="quality")
        matched_paper = next(
            paper for paper in papers if paper.get("folder") == paper_dir.name
        )
        assert matched_paper.get("source_name") == "quality_source"
        feedback_args = argparse.Namespace(
            auto_source_quality_feedback=True,
            source_quality_feedback_min_papers=1,
            source_quality_feedback_max_boost=4.0,
            source_quality_feedback_max_penalty=2.0,
        )
        source_feedback = _build_source_quality_feedback(
            submission_manager, feedback_args
        )
        assert source_feedback.get("quality_source")
        assert source_feedback["quality_source"]["priority_bonus"] > 0
        quality_strategy_args = argparse.Namespace(
            auto_quality_strategy_feedback=True,
            quality_strategy_submission_priority_threshold=85.0,
            quality_strategy_ready_rate_threshold=0.25,
            quality_strategy_exploration_priority_ceiling=65.0,
            quality_strategy_gate_pass_floor=0.25,
            quality_strategy_max_num_ideas_for_strong_sources=2,
            quality_strategy_max_num_ideas_for_weak_sources=6,
            quality_strategy_dominant_venue_rate_threshold=0.5,
            quality_strategy_dominant_paper_type_rate_threshold=0.5,
            guardrail_default_num_ideas=3,
        )
        strong_status = {
            "active_source": {"name": "quality_source"},
            "active_source_key": "topic::quality_source::topic.md",
            "source_quality_feedback": {
                "quality_source": {
                    "avg_priority": 90.0,
                    "ready_rate": 0.5,
                    "gate_pass_rate": 0.5,
                    "best_priority": 92.0,
                    "dominant_venue": "nature",
                    "dominant_venue_rate": 1.0,
                    "dominant_paper_type": "journal",
                    "dominant_paper_type_rate": 1.0,
                }
            },
            "current_daypart": "day",
        }
        strong_args = _apply_quality_strategy_feedback(
            quality_strategy_args,
            strong_status,
            ["--num-ideas", "4", "--paper-types", "normal"],
        )
        assert "--submission-mode" in strong_args
        assert "--rank-ideas" in strong_args
        assert "nature" in strong_args
        assert "journal" in strong_args
        assert strong_status["active_quality_strategy"]["mode"] == "submission_push"
        weak_status = {
            "active_source": {"name": "weak_source"},
            "active_source_key": "topic::weak_source::topic.md",
            "source_quality_feedback": {
                "weak_source": {
                    "avg_priority": 55.0,
                    "ready_rate": 0.0,
                    "gate_pass_rate": 0.0,
                    "best_priority": 58.0,
                }
            },
            "current_daypart": "day",
        }
        weak_args = _apply_quality_strategy_feedback(
            quality_strategy_args,
            weak_status,
            [
                "--num-ideas",
                "3",
                "--target-venue",
                "nature",
                "--paper-types",
                "journal",
            ],
        )
        assert "nature" not in weak_args
        assert weak_status["active_quality_strategy"]["mode"] == "explore_more"
        assert "4" in weak_args
        evidence_args = argparse.Namespace(
            auto_evidence_strategy_feedback=True,
            evidence_strategy_claim_support_floor=3.6,
            evidence_strategy_claim_alignment_floor=3.2,
            evidence_strategy_numeric_coverage_floor=3.8,
            evidence_strategy_evidence_density_floor=2.0,
            evidence_strategy_unsupported_claims_ceiling=1.0,
            evidence_strategy_min_quality_rewrite_rounds=2,
            evidence_strategy_review_strategy="depth",
        )
        evidence_status = {
            "active_source": {"name": "quality_source"},
            "active_source_key": "topic::quality_source::topic.md",
            "source_quality_feedback": {
                "quality_source": {
                    "avg_claim_support": 3.0,
                    "avg_claim_alignment": 2.8,
                    "avg_numeric_coverage": 3.1,
                    "avg_evidence_density": 1.2,
                    "avg_unsupported_claims": 2.0,
                }
            },
            "active_quality_strategy": {"mode": "submission_push"},
        }
        evidence_passthrough = _apply_evidence_strategy_feedback(
            evidence_args,
            evidence_status,
            ["--quality-preset", "balanced", "--quality-rewrite-rounds", "1"],
        )
        assert "--high-quality-mode" in evidence_passthrough
        assert "publishable" in evidence_passthrough
        assert "depth" in evidence_passthrough
        assert "2" in evidence_passthrough
        assert evidence_status["active_evidence_strategy"]["mode"] == "evidence_rebuild"
        governor_args = argparse.Namespace(
            auto_quality_governor=True,
            quality_governor_recent_cycles=6,
            quality_governor_stabilize_health_threshold=55.0,
            quality_governor_exploit_followup_gain=0.5,
            quality_governor_max_rewrite_top_k=2,
            quality_governor_max_dossier_top_k=2,
            quality_governor_max_source_plan_actions=2,
            quality_governor_experiment_todo_closure_floor=0.45,
            quality_governor_experiment_todo_p0_floor=0.5,
            quality_governor_experiment_todo_count_floor=2.5,
            rewrite_followup_top_k=1,
            auto_submission_dossier_top_k=1,
            auto_export_submission_dossier=True,
            auto_source_plan_max_actions=1,
            guardrail_submission_target=4,
            guardrail_min_followup_gain=0.25,
        )
        governor_daemon_dir = root / "daemon_runs" / "governor_validate"
        governor_daemon_dir.mkdir(parents=True, exist_ok=True)
        cycle_history = governor_daemon_dir / "cycle_history.jsonl"
        cycle_history.write_text(
            "\n".join(
                json.dumps(item, ensure_ascii=False)
                for item in [
                    {
                        "cycle": 1,
                        "returncode": 0,
                        "duration_seconds": 100,
                        "views": {
                            "submission_board_items": 4,
                            "rewrite_board_items": 3,
                            "shortlist_items": 2,
                        },
                    },
                    {
                        "cycle": 2,
                        "returncode": 0,
                        "duration_seconds": 120,
                        "views": {
                            "submission_board_items": 5,
                            "rewrite_board_items": 3,
                            "shortlist_items": 2,
                        },
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        governor_status = {
            "last_views": {
                "submission_board_items": 5,
                "rewrite_board_items": 3,
                "shortlist_items": 2,
            },
            "last_returncode": 0,
            "success_count": 3,
            "failure_count": 0,
            "consecutive_low_uplift_cycles": 0,
            "consecutive_empty_rewrite_cycles": 0,
            "consecutive_strong_submission_cycles": 2,
            "guardrail_phase": "hot_polish",
            "guardrail_followup_metrics": {"avg_priority_delta": 0.8},
        }
        governor_summary = _derive_quality_governor(
            governor_status, governor_daemon_dir, governor_args
        )
        assert governor_summary["mode"] == "exploit_quality"
        assert governor_summary["rewrite_followup_top_k_effective"] >= 1
        assert governor_summary["auto_submission_dossier_top_k_effective"] >= 1
        closure_governor_status = {
            "last_views": {
                "submission_board_items": 4,
                "rewrite_board_items": 4,
                "shortlist_items": 2,
            },
            "last_returncode": 0,
            "success_count": 4,
            "failure_count": 0,
            "consecutive_low_uplift_cycles": 0,
            "consecutive_empty_rewrite_cycles": 0,
            "consecutive_strong_submission_cycles": 1,
            "guardrail_phase": "steady_state",
            "guardrail_followup_metrics": {"avg_priority_delta": 0.6},
            "active_source": {"name": "quality_source"},
            "active_source_key": "topic::quality_source::topic.md",
            "source_quality_feedback": {
                "quality_source": {
                    "avg_experiment_todo": 3.2,
                    "avg_experiment_todo_p0": 0.8,
                    "avg_experiment_todo_closure_rate": 0.2,
                }
            },
        }
        closure_governor_summary = _derive_quality_governor(
            closure_governor_status, governor_daemon_dir, governor_args
        )
        assert closure_governor_summary["mode"] == "closure_repair"
        assert closure_governor_summary["auto_submission_dossier_enabled"] is False
        assert closure_governor_summary["rewrite_followup_top_k_effective"] >= 1

        mark_stage_complete(
            root, "prepare", artifacts={"idea_json": str(root / "idea.json")}
        )
        assert is_stage_complete(root, "prepare")
        entry = infer_run_entry(root)
        index = load_run_index()
        cleanup_capability = get_process_cleanup_capability()
        assert isinstance(cleanup_capability["available"], bool)
        assert cleanup_capability["backend"] in {"psutil", "unavailable"}

        sleeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        cleanup_stats = cleanup_child_processes(timeout=1, warn_if_unavailable=False)
        try:
            sleeper.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # psutil may be unavailable in lightweight envs; ensure the test process
            # is always cleaned up even when helper cleanup is a no-op fallback.
            sleeper.terminate()
            try:
                sleeper.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sleeper.kill()
                sleeper.wait(timeout=2)
        assert sleeper.poll() is not None
        assert isinstance(cleanup_stats, dict)
        assert cleanup_stats["available"] is cleanup_capability["available"]
        assert cleanup_stats["backend"] == cleanup_capability["backend"]
        assert cleanup_stats["children_found"] >= 0
        if not cleanup_capability["available"]:
            assert cleanup_stats["missing_module"] == "psutil"
            assert cleanup_stats["children_killed"] == 0
        elif cleanup_stats["children_found"] >= 1:
            assert cleanup_stats["children_killed"] >= 1
        assert isinstance(entry, dict)
        assert isinstance(index, dict)
        rebuilt = rebuild_run_index()
        assert isinstance(rebuilt, dict)

        source_config = root / "source_queue.json"
        source_config.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "name": "validate_topic",
                            "type": "topic",
                            "value": str(PROJECT_ROOT / "examples" / "example_topic.md"),
                            "priority": 10,
                            "time_of_day_preference": "any",
                            "day_target_venue": "nature",
                            "day_paper_types": ["journal"],
                            "day_num_ideas": 2,
                            "night_target_venue": "neurips",
                            "night_paper_types": ["normal"],
                            "night_num_ideas": 5,
                            "submission_mode": True,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        daemon_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "continuous_research_daemon.py"),
                "--source-config",
                str(source_config),
                "--research-dir",
                str(root),
                "--duration-hours",
                "0.1",
                "--dry-run",
                "--serve-dashboard",
                "--dashboard-port",
                "0",
                "--enable-rewrite-followup",
                "--rewrite-followup-top-k",
                "1",
                "--source-rotation",
                "round_robin",
                "--",
                "--num-ideas",
                "1",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "daemon_dir" in daemon_run.stdout
        daemon_info = json.loads(daemon_run.stdout)
        daemon_dir = Path(daemon_info["daemon_dir"])
        dashboard_url = str(daemon_info.get("dashboard_url") or "")
        assert dashboard_url.startswith("http://") or dashboard_url.endswith(
            "latest_live_dashboard.html"
        )
        assert (daemon_dir / "latest_cycle_summary.json").exists()
        assert (daemon_dir / "latest_operator_brief.json").exists()
        assert (daemon_dir / "latest_handoff_report.json").exists()
        assert (daemon_dir / "latest_handoff_report.md").exists()
        assert (daemon_dir / "latest_daily_report.json").exists()
        assert (daemon_dir / "latest_daily_report.md").exists()
        assert (daemon_dir / "reports" / "index.json").exists()
        assert (daemon_dir / "reports" / "index.md").exists()
        assert (daemon_dir / "reports" / "trends.json").exists()
        assert (daemon_dir / "reports" / "trends.md").exists()
        assert (daemon_dir / "latest_primary_action_queue.json").exists()
        assert (daemon_dir / "latest_primary_action_queue.md").exists()
        trends_report = json.loads(
            (daemon_dir / "reports" / "trends.json").read_text(encoding="utf-8")
        )
        primary_queue = json.loads(
            (daemon_dir / "latest_primary_action_queue.json").read_text(
                encoding="utf-8"
            )
        )
        assert "trend_action_label" in trends_report
        assert "trend_action_reason" in trends_report
        assert "trend_action_command" in trends_report
        assert "average_todo_closure_rate" in trends_report
        assert "latest_todo_closure_rate" in trends_report
        assert "todo_closure_delta" in trends_report
        assert "average_todo_backlog" in trends_report
        assert "items" in primary_queue
        archived_handoff_md = next((daemon_dir / "reports" / "handoff").glob("*.md"))
        archived_handoff_json = next(
            (daemon_dir / "reports" / "handoff").glob("*.json")
        )
        assert archived_handoff_md.exists()
        assert archived_handoff_json.exists()
        operator_brief = json.loads(
            (daemon_dir / "latest_operator_brief.json").read_text(encoding="utf-8")
        )
        assert "failure_hotspots" in operator_brief
        assert "rewrite_style_hotspots" in operator_brief
        assert "source_actions" in operator_brief
        assert "recent_control_events" in operator_brief
        assert "source_advisory" in operator_brief
        assert "do_now_actions" in operator_brief
        handoff_report = json.loads(
            (daemon_dir / "latest_handoff_report.json").read_text(encoding="utf-8")
        )
        daily_report = json.loads(
            (daemon_dir / "latest_daily_report.json").read_text(encoding="utf-8")
        )
        assert "do_now_actions" in handoff_report
        assert "recommended_commands" in handoff_report
        assert "attention_label" in handoff_report
        assert "recovery_reason" in handoff_report
        assert "recovery_command" in handoff_report
        assert "report_date" in daily_report
        assert "recommended_commands" in daily_report
        archived_daily_md = (
            daemon_dir / "reports" / "daily" / f"{daily_report.get('report_date')}.md"
        )
        archived_daily_json = (
            daemon_dir / "reports" / "daily" / f"{daily_report.get('report_date')}.json"
        )
        assert archived_daily_md.exists()
        assert archived_daily_json.exists()
        schema = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "daemon"
                / "daemon_control.schema.json"
            ).read_text(encoding="utf-8")
        )
        assert schema.get("title") == "AI Scientist Daemon Control"
        source_schema = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "sources"
                / "source_queue.schema.json"
            ).read_text(encoding="utf-8")
        )
        assert source_schema.get("title") == "AI Scientist Source Queue"
        daemon_profile_schema = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "daemon"
                / "daemon_profile.schema.json"
            ).read_text(encoding="utf-8")
        )
        assert daemon_profile_schema.get("title") == "AI Scientist Daemon Profile"
        assert (PROJECT_ROOT / "configs" / "sources" / "stable_source_priority.example.json").exists()
        assert (PROJECT_ROOT / "configs" / "daemon" / "stable_daemon_profile.local.example.json").exists()
        assert (PROJECT_ROOT / "docs" / "CONFIG_REFERENCE.md").exists()
        profile_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_daemon_profile.py"),
                str(PROJECT_ROOT / "configs" / "daemon" / "daemon_profile.example.json"),
                "--dry-run",
                "--print-command",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "continuous_research_daemon.py" in profile_run.stdout
        stable_profile_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_daemon_profile.py"),
                str(PROJECT_ROOT / "configs" / "daemon" / "stable_daemon_profile.example.json"),
                "--dry-run",
                "--print-command",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        stable_profile_payload = json.loads(stable_profile_run.stdout)
        stable_command = stable_profile_payload["command"]
        assert "stable_source_priority.example.json" in " ".join(stable_command)
        assert "--dashboard-port" in stable_command and "0" in stable_command
        assert "--guardrail-submission-target" in stable_command
        assert "--rewrite-followup-max-rounds" in stable_command
        stable_day_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_daemon_profile.py"),
                str(PROJECT_ROOT / "configs" / "daemon" / "stable_day_daemon_profile.example.json"),
                "--dry-run",
                "--print-command",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "stable_day_polish_daemon" in stable_day_run.stdout
        stable_night_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_daemon_profile.py"),
                str(PROJECT_ROOT / "configs" / "daemon" / "stable_night_daemon_profile.example.json"),
                "--dry-run",
                "--print-command",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "stable_night_generate_daemon" in stable_night_run.stdout
        assert (
            str(PROJECT_ROOT / "configs" / "sources" / "stable_source_priority.example.json")
            in stable_night_run.stdout
        )
        portable_dir = root / "portable_profile"
        portable_dir.mkdir(exist_ok=True)
        portable_source = portable_dir / "portable_source.json"
        portable_source.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "name": "portable",
                            "type": "topic",
                            "value": str(PROJECT_ROOT / "examples" / "example_topic.md"),
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        portable_profile = portable_dir / "portable_profile.json"
        portable_profile.write_text(
            json.dumps(
                {
                    "research_dir": "relative_research",
                    "source_config": "portable_source.json",
                    "duration_hours": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        portable_overlay = portable_dir / "portable_profile.local.json"
        portable_overlay.write_text(
            json.dumps(
                {
                    "sleep_minutes": 17,
                    "dashboard_port": 8123,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        portable_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_daemon_profile.py"),
                str(portable_profile),
                "--dry-run",
                "--print-command",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        portable_topics_dir = portable_dir / "topics"
        portable_topics_dir.mkdir(exist_ok=True)
        portable_topic_file = portable_topics_dir / "portable_topic.md"
        portable_topic_file.write_text(
            "# Portable Topic\n\nA rehearsal topic.\n", encoding="utf-8"
        )
        portable_rehearsal_profile = portable_dir / "portable_rehearsal_profile.json"
        portable_rehearsal_profile.write_text(
            json.dumps(
                {
                    "research_dir": "relative_research",
                    "topic_files": ["topics/portable_topic.md"],
                    "duration_hours": 1,
                    "auto_apply_source_plan": True,
                    "auto_source_plan_max_actions": 1,
                    "auto_source_plan_min_health": 80,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        portable_rehearsal_run = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_daemon_rehearsal.py"),
                "--profile",
                str(portable_rehearsal_profile),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"status": "ok"' in portable_rehearsal_run.stdout
        portable_rehearsal_summary_match = re.search(
            r"\[rehearsal-summary\](.+)", portable_rehearsal_run.stdout
        )
        assert portable_rehearsal_summary_match
        portable_rehearsal_summary = json.loads(
            portable_rehearsal_summary_match.group(1)
        )
        portable_rehearsal_control = (
            portable_rehearsal_summary.get("control_snapshot") or {}
        )
        assert portable_rehearsal_control.get("source_commands")
        auto_command = next(
            iter((portable_rehearsal_control.get("source_commands") or {}).values())
        )
        assert auto_command.get("force_next_cycle") is True
        portable_payload = json.loads(portable_run.stdout)
        portable_command = portable_payload["command"]
        assert str(portable_source.resolve()) in portable_command
        assert str((portable_dir / "relative_research").resolve()) in portable_command
        assert str(portable_overlay.resolve()) in portable_payload["applied_overlays"]
        assert "--sleep-minutes" in portable_command and "17" in portable_command
        assert "--dashboard-port" in portable_command and "8123" in portable_command
        rehearsal_run = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "run_daemon_rehearsal.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"status": "ok"' in rehearsal_run.stdout
        assert (PROJECT_ROOT / "docs" / "OPERATIONS_CHECKLIST.md").exists()
        assert (PROJECT_ROOT / "configs" / "daemon" / "stable_day_daemon_profile.example.json").exists()
        assert (PROJECT_ROOT / "configs" / "daemon" / "stable_night_daemon_profile.example.json").exists()
        assert (PROJECT_ROOT / "run_stable_daemon.sh").exists()
        shellcheck_run = subprocess.run(
            ["bash", "-n", str(PROJECT_ROOT / "run_stable_daemon.sh")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        start_shellcheck_run = subprocess.run(
            ["bash", "-n", str(PROJECT_ROOT / "start_research.sh")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        start_menu_env = dict(os.environ)
        start_menu_env["RESEARCH_OUTPUT_DIR"] = str(root / "menu_research")
        start_menu_env["PYTHON"] = sys.executable
        start_menu_env.pop("ZHIPU_API_KEY", None)
        start_menu_run = subprocess.run(
            ["bash", str(PROJECT_ROOT / "start_research.sh")],
            cwd=str(PROJECT_ROOT),
            env=start_menu_env,
            input="9\n",
            capture_output=True,
            text=True,
            check=True,
        )
        assert "AI Scientist 连续论文生成系统" in start_menu_run.stdout
        assert "退出" in start_menu_run.stdout
        wrapper_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "auto",
                "--dry-run",
                "--print-command",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "continuous_research_daemon.py" in wrapper_run.stdout
        wrapper_overlay = portable_dir / "wrapper_overlay.json"
        wrapper_overlay.write_text(
            json.dumps({"sleep_minutes": 19}, ensure_ascii=False), encoding="utf-8"
        )
        wrapper_overlay_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "balanced",
                "--dry-run",
                "--print-command",
                "--overlay",
                str(wrapper_overlay),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"applied_overlays"' in wrapper_overlay_run.stdout
        assert str(wrapper_overlay.resolve()) in wrapper_overlay_run.stdout
        wrapper_overlay_payload = json.loads(wrapper_overlay_run.stdout)
        assert "daemon_name" in wrapper_overlay_payload
        assert "research_dir" in wrapper_overlay_payload
        target_mode_overlay = portable_dir / "target_mode_overlay.json"
        target_mode_daemon_name = "target_mode_overlay_daemon"
        target_mode_overlay.write_text(
            json.dumps({"daemon_name": target_mode_daemon_name}, ensure_ascii=False),
            encoding="utf-8",
        )
        target_mode_daemon_dir = root / "daemon_runs" / target_mode_daemon_name
        target_mode_daemon_dir.mkdir(parents=True, exist_ok=True)
        (target_mode_daemon_dir / "daemon_status.json").write_text(
            json.dumps({"state": "idle", "cycle": 3}, ensure_ascii=False),
            encoding="utf-8",
        )
        (target_mode_daemon_dir / "latest_operator_brief.md").write_text(
            "# Operator Brief\nready\n", encoding="utf-8"
        )
        wrapper_env = dict(os.environ)
        wrapper_env["RESEARCH_OUTPUT_DIR"] = str(root)
        wrapper_env["PYTHON"] = sys.executable
        status_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "status",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "20",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Daemon directory:" in status_run.stdout
        program_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "program",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "20",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Autonomous Research Program" in program_run.stdout
        experiment_ledger_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "experiment-ledger",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "5",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "cycle" in experiment_ledger_run.stdout.lower()
        target_mode_status_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "status",
                "--target-mode",
                "balanced",
                "--overlay",
                str(target_mode_overlay),
                "--lines",
                "5",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            f"Daemon directory: {target_mode_daemon_dir}"
            in target_mode_status_run.stdout
        )
        invalid_target_mode_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "status",
                "--target-mode",
                "bogus",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert invalid_target_mode_run.returncode != 0
        assert "Unknown target mode: bogus" in invalid_target_mode_run.stderr
        doctor_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "doctor",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert doctor_run.returncode in {0, 1}
        assert '"status": "ok"' in doctor_run.stdout
        assert "AI Scientist preflight" in doctor_run.stdout
        handoff_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "handoff",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "15",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            "Handoff Report" in handoff_run.stdout
            or "falling back to operator brief" in handoff_run.stdout
        )
        daily_report_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "daily-report",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "15",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            "Daily Report" in daily_report_run.stdout
            or "falling back to daily summary" in daily_report_run.stdout
        )
        list_reports_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "list-reports",
                "--daemon-dir",
                str(daemon_dir),
                "--report-kind",
                "all",
                "--top",
                "5",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Entries shown:" in list_reports_run.stdout
        assert (
            "report_date=" in list_reports_run.stdout
            or "No archived reports found." in list_reports_run.stdout
        )
        report_trends_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "report-trends",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "20",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Report Trends" in report_trends_run.stdout
        assert "Trend Action" in report_trends_run.stdout
        next_actions_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "next-actions",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "15",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            "Primary Action Queue" in next_actions_run.stdout
            or "falling back to handoff report" in next_actions_run.stdout
        )
        recover_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "recover",
                "--daemon-dir",
                str(daemon_dir),
                "--print-command",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert recover_run.stdout.strip()
        brief_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "brief",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "10",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            "Current Phase" in brief_run.stdout
            or "Operator Brief" in brief_run.stdout
            or "Top Submission Targets" in brief_run.stdout
        )
        dashboard_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "dashboard",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            "http://" in dashboard_run.stdout
            or "latest_live_dashboard.html" in dashboard_run.stdout
        )
        heartbeat_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "tail-heartbeat",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "5",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            heartbeat_run.stdout.strip()
            or "Heartbeat log not found" in heartbeat_run.stdout
        )
        logs_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "logs",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "5",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            "cycle log" in logs_run.stdout.lower()
            or "launch log" in logs_run.stdout.lower()
        )
        control_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "control",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"paused"' in control_run.stdout
        pause_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "pause",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"paused": true' in pause_run.stdout.lower()
        set_mode_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "set-mode",
                "focus_rewrite",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "focus_rewrite" in set_mode_run.stdout
        stop_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "stop-after-cycle",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "stop_after_cycle" in stop_run.stdout
        history_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "control-history",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "5",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            history_run.stdout.strip()
            or "Control history not found" in history_run.stdout
        )
        disable_source_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "disable-source",
                "validate_topic",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "validate_topic" in disable_source_run.stdout
        set_priority_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "set-source-priority",
                "validate_topic",
                "9",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert (
            "validate_topic" in set_priority_run.stdout
            and "9.0" in set_priority_run.stdout
        )
        force_next_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "source-force-next",
                "validate_topic",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "force_next_cycle" in force_next_run.stdout
        boost_next_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "source-boost-next",
                "validate_topic",
                "4",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "priority_boost_next" in boost_next_run.stdout
        clear_source_command_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "clear-source-command",
                "validate_topic",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "source_commands" in clear_source_command_run.stdout
        enable_source_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "enable-source",
                "validate_topic",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "disabled_sources" in enable_source_run.stdout
        clear_priority_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "clear-source-priority",
                "validate_topic",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "source_priority_overrides" in clear_priority_run.stdout
        source_summary_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "source-summary",
                "--daemon-dir",
                str(daemon_dir),
                "--lines",
                "5",
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Sources shown:" in source_summary_run.stdout
        source_plan_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "source-plan",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Recommendations:" in source_plan_run.stdout
        assert "Suggested Commands:" in source_plan_run.stdout
        assert "run_stable_daemon.sh" in source_plan_run.stdout
        assert "run_stable_daemon.sh --daemon-dir" not in source_plan_run.stdout
        assert "source_summary_command" in operator_brief.get(
            "recommended_commands", {}
        )
        assert operator_brief.get("source_advisory")
        assert (
            "source-force-next" in source_plan_run.stdout
            or "enable-source" in source_plan_run.stdout
        )
        clear_mode_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "clear-mode",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"force_mode": null' in clear_mode_run.stdout.lower()
        resume_run = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "run_stable_daemon.sh"),
                "resume",
                "--daemon-dir",
                str(daemon_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=wrapper_env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert '"paused": false' in resume_run.stdout.lower()
        assert "control_summary" in operator_brief
        assert (daemon_dir / "latest_source_runtime_board.json").exists()
        assert (daemon_dir / "latest_source_health_board.json").exists()
        assert (daemon_dir / "latest_live_dashboard.json").exists()
        assert (daemon_dir / "latest_live_dashboard.html").exists()
        assert "--target-venue" in daemon_info["command"]
        current_hour = datetime.now().hour
        expected_type = "journal" if 8 <= current_hour < 20 else "normal"
        assert expected_type in daemon_info["command"]
        print(
            "[OK] helper smoke passed:",
            json.dumps(
                {
                    "review_pick": find_best_pdf_path(root),
                    "latest_pick": find_latest_pdf_path(root),
                    "cleanup_backend": cleanup_stats["backend"],
                    "cleanup_keys": sorted(cleanup_stats.keys()),
                    "entry_stage": entry["latest_stage"],
                    "review_execution_smoke": review_exec["found"],
                    "cli_help_smoke": cli_help_smoke,
                    "launcher_workflow_smoke": launcher_workflow_smoke,
                    "continuous_quality_smoke": continuous_quality_smoke,
                },
                ensure_ascii=False,
            ),
        )


def run_import_smoke() -> None:
    modules = [
        "ai_scientist.config.paths",
        "ai_scientist.utils.high_quality_pipeline",
        "ai_scientist.utils.idea_ranking",
        "ai_scientist.utils.pipeline_helpers",
        "ai_scientist.utils.review_execution",
        "ai_scientist.utils.launcher_workflow",
        "run_project",
        "continuous_paper_generator",
        "continuous_research_daemon",
        "launch_scientist_bfts",
        "launch_scientist_zhipu",
    ]
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Import smoke failed for module: {module_name}"
            ) from exc
    print(f"[OK] import smoke passed for {len(modules)} modules")


def main() -> int:
    require_login("仓库校验(validate_repo)")

    parser = argparse.ArgumentParser(description="Repository smoke validation")
    parser.add_argument(
        "--full-import-smoke",
        action="store_true",
        help="also import shared runners and entrypoints; requires installed dependencies",
    )
    args = parser.parse_args()

    ensure_supported_python()

    os.environ.setdefault(
        "RESEARCH_OUTPUT_DIR", tempfile.mkdtemp(prefix="ai_scientist_validate_")
    )

    run_py_compile()
    run_helper_smoke()

    if args.full_import_smoke:
        run_import_smoke()
    else:
        print("[SKIP] import smoke not requested")

    print("Validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
