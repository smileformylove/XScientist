from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_scientist.utils.launcher_workflow import run_review_phase, run_writeup_phase
from continuous_paper_generator import (
    StrictFallbackViolation,
    _run_optional_quality_pass,
)


class LauncherWorkflowRegressionTests(unittest.TestCase):
    def _writeup_plan(self) -> dict:
        return {
            "target_venue": "nature",
            "strategy_feedback": {"rationale": []},
            "num_cite_rounds": 12,
            "writeup_retries": 2,
            "page_limit": 8,
            "writeup_engine": "normal",
        }

    @mock.patch("ai_scientist.utils.launcher_workflow.mark_stage_complete")
    @mock.patch("ai_scientist.utils.launcher_workflow.save_token_tracker")
    @mock.patch("ai_scientist.utils.launcher_workflow.execute_quality_workflow_with_followups")
    @mock.patch("ai_scientist.utils.launcher_workflow.build_workflow_execution_policy")
    @mock.patch("ai_scientist.utils.launcher_workflow.perform_writeup")
    @mock.patch("ai_scientist.utils.launcher_workflow.gather_citations")
    @mock.patch("ai_scientist.utils.launcher_workflow.build_writeup_execution_plan")
    @mock.patch("ai_scientist.utils.launcher_workflow.is_stage_complete", return_value=True)
    @mock.patch("ai_scientist.utils.launcher_workflow.find_best_pdf_path")
    def test_run_writeup_phase_should_revalidate_quality_on_resume(
        self,
        find_pdf_mock: mock.Mock,
        _stage_complete_mock: mock.Mock,
        build_plan_mock: mock.Mock,
        gather_citations_mock: mock.Mock,
        perform_writeup_mock: mock.Mock,
        build_policy_mock: mock.Mock,
        execute_quality_mock: mock.Mock,
        _save_token_tracker_mock: mock.Mock,
        _mark_stage_complete_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "paper.pdf"
            pdf_path.write_text("pdf", encoding="utf-8")
            find_pdf_mock.return_value = str(pdf_path)
            build_plan_mock.return_value = self._writeup_plan()
            build_policy_mock.return_value = SimpleNamespace(
                allow_auto_improvement_fallback=False,
                reject_on_auto_improvement_fallback=True,
            )
            execute_quality_mock.return_value = {
                "quality_result": {"quality_gate_passed": True},
                "acceptance": {"accepted": True, "reasons": []},
                "summary": "ok",
            }

            result = run_writeup_phase(
                td,
                writeup_type="normal",
                writeup_retries=2,
                num_cite_rounds=12,
                model_citation="model-citation",
                model_writeup_small="model-small",
                model_writeup="model-big",
                high_quality_mode=True,
                quality_preset="publishable",
                target_venue="nature",
                workflow_mode="review_board",
                submission_mode=True,
                resume=True,
                logger=lambda _msg: None,
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["reused_writeup"])
        execute_quality_mock.assert_called_once()
        gather_citations_mock.assert_not_called()
        perform_writeup_mock.assert_not_called()

    @mock.patch("ai_scientist.utils.launcher_workflow.mark_stage_complete")
    @mock.patch("ai_scientist.utils.launcher_workflow.save_token_tracker")
    @mock.patch("ai_scientist.utils.launcher_workflow.record_quality_fallback_if_needed")
    @mock.patch("ai_scientist.utils.launcher_workflow.execute_quality_workflow_with_followups")
    @mock.patch("ai_scientist.utils.launcher_workflow.build_workflow_execution_policy")
    @mock.patch("ai_scientist.utils.launcher_workflow.perform_writeup", return_value=True)
    @mock.patch("ai_scientist.utils.launcher_workflow.gather_citations", return_value="citations")
    @mock.patch("ai_scientist.utils.launcher_workflow.build_writeup_execution_plan")
    @mock.patch("ai_scientist.utils.launcher_workflow.is_stage_complete", return_value=False)
    @mock.patch("ai_scientist.utils.launcher_workflow.find_best_pdf_path", return_value=None)
    def test_run_writeup_phase_should_fail_closed_on_strict_quality_fallback(
        self,
        _find_pdf_mock: mock.Mock,
        _stage_complete_mock: mock.Mock,
        build_plan_mock: mock.Mock,
        _gather_citations_mock: mock.Mock,
        _perform_writeup_mock: mock.Mock,
        build_policy_mock: mock.Mock,
        execute_quality_mock: mock.Mock,
        record_fallback_mock: mock.Mock,
        _save_token_tracker_mock: mock.Mock,
        mark_stage_complete_mock: mock.Mock,
    ) -> None:
        build_plan_mock.return_value = self._writeup_plan()
        build_policy_mock.return_value = SimpleNamespace(
            allow_auto_improvement_fallback=True,
            reject_on_auto_improvement_fallback=False,
        )
        execute_quality_mock.return_value = {
            "quality_result": {"auto_improvement_fallback_used": True},
            "acceptance": {"accepted": False, "reasons": ["fallback used"]},
            "summary": "bad",
        }
        record_fallback_mock.return_value = {
            "stage": "quality_review",
            "fallback_kind": "auto_improvement_rewrite",
        }

        with tempfile.TemporaryDirectory() as td:
            result = run_writeup_phase(
                td,
                writeup_type="normal",
                writeup_retries=1,
                num_cite_rounds=5,
                model_citation="model-citation",
                model_writeup_small="model-small",
                model_writeup="model-big",
                high_quality_mode=True,
                quality_preset="publishable",
                target_venue="nature",
                workflow_mode="review_board",
                strict_fallbacks=True,
                resume=False,
                logger=lambda _msg: None,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_stage"], "quality_fallback_blocked")
        mark_stage_complete_mock.assert_not_called()

    @mock.patch("ai_scientist.utils.launcher_workflow.evaluate_final_submission_readiness")
    @mock.patch("ai_scientist.utils.launcher_workflow.build_workflow_execution_policy")
    @mock.patch("ai_scientist.utils.launcher_workflow.is_stage_complete", return_value=True)
    @mock.patch("ai_scientist.utils.launcher_workflow.find_best_pdf_path")
    def test_run_review_phase_resume_should_still_evaluate_submission_gate(
        self,
        find_pdf_mock: mock.Mock,
        _stage_complete_mock: mock.Mock,
        build_policy_mock: mock.Mock,
        evaluate_submission_mock: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "review_text.txt").write_text("{}", encoding="utf-8")
            (root / "review_img_cap_ref.json").write_text("{}", encoding="utf-8")
            pdf_path = root / "paper.pdf"
            pdf_path.write_text("pdf", encoding="utf-8")
            find_pdf_mock.return_value = str(pdf_path)
            build_policy_mock.return_value = SimpleNamespace(
                reject_on_auto_improvement_fallback=True
            )
            evaluate_submission_mock.return_value = {
                "accepted": False,
                "reasons": ["review debt remains"],
            }

            result = run_review_phase(
                root,
                model_review="model-review",
                high_quality_mode=True,
                workflow_mode="review_board",
                submission_mode=True,
                resume=True,
            )

        self.assertTrue(result["resumed"])
        self.assertFalse(result["submission_acceptance"]["accepted"])
        evaluate_submission_mock.assert_called_once()


class StrictFallbackEnforcementRegressionTests(unittest.TestCase):
    @mock.patch("continuous_paper_generator.record_quality_fallback_if_needed")
    @mock.patch("continuous_paper_generator.execute_quality_workflow_with_followups")
    def test_quality_override_should_record_fallback_without_raising(
        self,
        execute_quality_mock: mock.Mock,
        record_fallback_mock: mock.Mock,
    ) -> None:
        execute_quality_mock.return_value = {
            "quality_result": {"auto_improvement_fallback_used": True},
            "acceptance": {"accepted": False, "reasons": ["fallback used"]},
            "summary": "bad",
        }
        record_fallback_mock.return_value = {"stage": "quality_review"}

        result = _run_optional_quality_pass(
            enabled=True,
            run_dir="/tmp/demo",
            paper_type="normal",
            rewrite_model="model-writeup",
            quality_model="model-quality",
            target_venue="nature",
            quality_preset="publishable",
            logger=lambda _msg: None,
            reject_on_auto_improvement_fallback=True,
            strict_fallbacks=False,
            workflow_mode="review_board",
        )

        self.assertIsNotNone(result)
        self.assertFalse(record_fallback_mock.call_args.kwargs["strict"])

    @mock.patch("continuous_paper_generator.record_quality_fallback_if_needed")
    @mock.patch("continuous_paper_generator.execute_quality_workflow_with_followups")
    def test_quality_strict_fallback_should_still_raise(
        self,
        execute_quality_mock: mock.Mock,
        record_fallback_mock: mock.Mock,
    ) -> None:
        execute_quality_mock.return_value = {
            "quality_result": {"auto_improvement_fallback_used": True},
            "acceptance": {"accepted": False, "reasons": ["fallback used"]},
            "summary": "bad",
        }
        record_fallback_mock.return_value = {"stage": "quality_review"}

        with self.assertRaises(StrictFallbackViolation):
            _run_optional_quality_pass(
                enabled=True,
                run_dir="/tmp/demo",
                paper_type="normal",
                rewrite_model="model-writeup",
                quality_model="model-quality",
                target_venue="nature",
                quality_preset="publishable",
                logger=lambda _msg: None,
                strict_fallbacks=True,
                workflow_mode="review_board",
            )


if __name__ == "__main__":
    unittest.main()
