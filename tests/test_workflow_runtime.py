from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.critic_workflow import run_independent_critic_pass
from ai_scientist.utils.workflow_runtime import (
    build_workflow_runtime_plan,
    execute_review_suite,
)


class WorkflowRuntimeTests(unittest.TestCase):
    def test_build_workflow_runtime_plan_should_expand_high_quality_final_roles(self) -> None:
        plan = build_workflow_runtime_plan(
            "classic_pipeline",
            high_quality_mode=True,
            target_venue="nature",
        )
        self.assertEqual(plan.improvement_review_roles, ("rigor",))
        self.assertEqual(plan.final_review_roles, ("clarity", "reproducibility"))
        self.assertIn("ideation", plan.stage_sequence)

    def test_execute_review_suite_should_merge_multi_role_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / "paper.pdf"
            pdf_path.write_text("stub", encoding="utf-8")
            review_dir = root / "reviews"

            review_payloads = {
                "novelty": {
                    "review": {
                        "Summary": "Novelty needs to be framed more clearly.",
                        "Weaknesses": ["Novelty delta is underspecified."],
                        "Questions": ["What separates this from prior work?"],
                        "Limitations": ["Contribution may look incremental."],
                        "scores": {"Overall": 3.0, "Novelty": 2.5},
                    }
                },
                "rigor": {
                    "review": {
                        "Summary": "Evaluation lacks a strong baseline.",
                        "Weaknesses": ["Baseline coverage is thin."],
                        "Questions": ["Where is the main ablation?"],
                        "Limitations": ["Only one dataset is reported."],
                        "scores": {"Overall": 4.0, "Rigor": 2.0},
                    }
                },
            }
            image_payloads = {
                "novelty": {
                    "figure_reviews": [
                        {"figure_id": "fig_1", "description": "Clarify the novelty panel."}
                    ]
                },
                "rigor": {
                    "figure_reviews": [
                        {"figure_id": "fig_2", "description": "Add error bars."}
                    ]
                },
            }

            def _fake_execute_review_pass(**kwargs):
                role = kwargs["review_role"]
                save_dir = kwargs.get("save_dir")
                if save_dir is not None:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                return {
                    "found": True,
                    "pdf_path": str(pdf_path),
                    "review_text": review_payloads[role],
                    "review_img": image_payloads[role],
                    "job": {"job_id": f"{role}_job"},
                }

            with patch(
                "ai_scientist.utils.workflow_runtime.execute_review_pass",
                side_effect=_fake_execute_review_pass,
            ):
                result = execute_review_suite(
                    review_roles=["novelty", "rigor"],
                    paper_dir=root,
                    model_review="demo-model",
                    review_plan={
                        "review_instruction": "Review this draft.",
                        "review_reflections": 1,
                        "review_fewshot": 1,
                        "review_ensemble": 1,
                        "review_temperature": 0.5,
                    },
                    create_client_fn=lambda model: (None, model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=lambda *args, **kwargs: None,
                    perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: None,
                    pdf_path_resolver=lambda _: str(pdf_path),
                    save_dir=review_dir,
                    suite_name="unit_suite",
                )

            self.assertTrue(result["found"])
            self.assertEqual(result["review_roles_used"], ["novelty", "rigor"])
            merged_review = result["review_text"]["review"]
            self.assertIn("Novelty delta is underspecified.", merged_review["Weaknesses"])
            self.assertIn("Baseline coverage is thin.", merged_review["Weaknesses"])
            self.assertEqual(merged_review["scores"]["Overall"], 3.5)
            self.assertEqual(merged_review["scores"]["Novelty"], 2.5)
            self.assertEqual(merged_review["scores"]["Rigor"], 2.0)
            merged_image = result["review_img"]
            self.assertEqual(len(merged_image["figure_reviews"]), 2)
            self.assertTrue((review_dir / "review_text.json").exists())
            self.assertTrue((review_dir / "review_img.json").exists())
            self.assertTrue((review_dir / "review_suite.json").exists())
            self.assertTrue((review_dir / "novelty").is_dir())
            self.assertTrue((review_dir / "rigor").is_dir())
            suite_payload = json.loads(
                (review_dir / "review_suite.json").read_text(encoding="utf-8")
            )
            self.assertEqual(suite_payload["job_ids"]["novelty"], "novelty_job")
            self.assertEqual(suite_payload["job_ids"]["rigor"], "rigor_job")

    def test_multi_agent_board_should_include_expanded_hostile_critic_roles(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        self.assertIn("claim_cross_examiner", plan.improvement_review_roles)
        self.assertIn("meta_reviewer", plan.final_review_roles)
        self.assertIn("novelty_executioner", plan.critic_review_roles)
        self.assertIn("stats_sniper", plan.critic_review_roles)
        self.assertIn("desk_reject_editor", plan.critic_review_roles)
        self.assertTrue(plan.requires_independent_critic)

    def test_run_independent_critic_pass_should_support_ablation_toggle(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            with patch.dict(os.environ, {"AI_SCIENTIST_ABLATE_HOSTILE_CRITIC": "1"}):
                result = run_independent_critic_pass(
                    workflow_runtime_plan=plan,
                    paper_dir=project_root,
                    model_review="demo-model",
                    review_plan={},
                    create_client_fn=lambda model: (None, model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=lambda *args, **kwargs: None,
                    perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: None,
                    pdf_path_resolver=lambda _: None,
                    save_dir=project_root / "critic",
                    project_root=project_root,
                )

        self.assertFalse(result["ran"])
        self.assertEqual(result["blocking_issue_count"], 0)
        self.assertIsNone(result["critic_findings_file"])


if __name__ == "__main__":
    unittest.main()
