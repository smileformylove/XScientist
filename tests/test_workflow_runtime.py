from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.critic_workflow import run_independent_critic_pass
from ai_scientist.utils.decision_log import load_decision_log
from ai_scientist.utils.pipeline_contracts import (
    load_contract_artifact,
    save_contract_artifact,
)
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
            decisions = load_decision_log(project_root)

        self.assertFalse(result["ran"])
        self.assertEqual(result["blocking_issue_count"], 0)
        self.assertIsNone(result["critic_findings_file"])
        self.assertEqual(decisions[-1]["category"], "hostile_critic_confirmation")
        self.assertEqual(decisions[-1]["selected"], "skip_independent_confirmation")
        self.assertTrue(decisions[-1]["metadata"]["ablation_enabled"])

    def test_independent_critic_pass_should_run_confirmation_when_primary_clear(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            calls = []

            def _fake_execute_review_suite(**kwargs):
                calls.append(kwargs)
                lane_name = kwargs["lane_name"]
                review_state = load_contract_artifact(
                    project_root,
                    "review_state",
                    default={},
                ) or {}
                lane_summaries = dict(review_state.get("lane_summaries") or {})
                lane_summaries[lane_name] = {
                    "active_issue_count": 0,
                    "blocking_issue_count": 0,
                }
                review_state["lane_summaries"] = lane_summaries
                save_contract_artifact(
                    project_root,
                    "review_state",
                    review_state,
                    producer="test",
                )
                return {
                    "found": True,
                    "review_roles_used": list(kwargs["review_roles"]),
                    "primary_role": "novelty",
                    "passes_by_role": {
                        "novelty": {
                            "review_text": {"review": {"Weaknesses": []}}
                        }
                    },
                    "review_text": {"review": {"Weaknesses": []}},
                    "review_img": {},
                }

            with patch(
                "ai_scientist.utils.critic_workflow.execute_review_suite",
                side_effect=_fake_execute_review_suite,
            ):
                result = run_independent_critic_pass(
                    workflow_runtime_plan=plan,
                    paper_dir=project_root,
                    model_review="demo-model",
                    review_plan={
                        "review_instruction": "Review directly.",
                        "review_reflections": 1,
                        "review_fewshot": 1,
                        "review_ensemble": 1,
                        "review_temperature": 0.5,
                    },
                    create_client_fn=lambda model: (None, model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=lambda *args, **kwargs: None,
                    perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: None,
                    pdf_path_resolver=lambda _: str(project_root / "paper.pdf"),
                    save_dir=project_root / "critic",
                    project_root=project_root,
                )
            decisions = load_decision_log(project_root)

        self.assertEqual([call["lane_name"] for call in calls], [
            "hostile_critic",
            "hostile_critic_confirmation",
        ])
        self.assertTrue(result["critic_confirmed"])
        self.assertEqual(result["blocking_issue_count"], 0)
        confirmation_instruction = calls[1]["review_plan"]["review_instruction"]
        self.assertIn("Independent confirmation pass", confirmation_instruction)
        self.assertEqual(decisions[-1]["category"], "hostile_critic_confirmation")
        self.assertEqual(decisions[-1]["selected"], "run_independent_confirmation")

    def test_confirmation_blocker_should_block_clear_primary_critic(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)

            def _fake_execute_review_suite(**kwargs):
                lane_name = kwargs["lane_name"]
                blocking = 1 if lane_name == "hostile_critic_confirmation" else 0
                review_state = load_contract_artifact(
                    project_root,
                    "review_state",
                    default={},
                ) or {}
                lane_summaries = dict(review_state.get("lane_summaries") or {})
                lane_summaries[lane_name] = {
                    "active_issue_count": blocking,
                    "blocking_issue_count": blocking,
                }
                review_state["lane_summaries"] = lane_summaries
                save_contract_artifact(
                    project_root,
                    "review_state",
                    review_state,
                    producer="test",
                )
                return {
                    "found": True,
                    "review_roles_used": list(kwargs["review_roles"]),
                    "primary_role": "novelty",
                    "passes_by_role": {
                        "novelty": {
                            "review_text": {
                                "review": {
                                    "Weaknesses": (
                                        ["Confirmation found an unsupported claim."]
                                        if blocking
                                        else []
                                    )
                                }
                            }
                        }
                    },
                    "review_text": {
                        "review": {
                            "Weaknesses": (
                                ["Confirmation found an unsupported claim."]
                                if blocking
                                else []
                            )
                        }
                    },
                    "review_img": {},
                }

            with patch(
                "ai_scientist.utils.critic_workflow.execute_review_suite",
                side_effect=_fake_execute_review_suite,
            ):
                result = run_independent_critic_pass(
                    workflow_runtime_plan=plan,
                    paper_dir=project_root,
                    model_review="demo-model",
                    review_plan={
                        "review_instruction": "Review directly.",
                        "review_reflections": 1,
                        "review_fewshot": 1,
                        "review_ensemble": 1,
                        "review_temperature": 0.5,
                    },
                    create_client_fn=lambda model: (None, model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=lambda *args, **kwargs: None,
                    perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: None,
                    pdf_path_resolver=lambda _: str(project_root / "paper.pdf"),
                    save_dir=project_root / "critic",
                    project_root=project_root,
                )

        self.assertFalse(result["critic_confirmed"])
        self.assertEqual(result["blocking_issue_count"], 1)
        self.assertEqual(result["blocking_source"], "ep_independent_eval")
        self.assertEqual(
            result["critic_confirmation"]["reason"],
            "confirmation_blockers",
        )

    def test_primary_blocker_should_not_trigger_confirmation(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            calls = []

            def _fake_execute_review_suite(**kwargs):
                calls.append(kwargs)
                lane_name = kwargs["lane_name"]
                review_state = load_contract_artifact(
                    project_root,
                    "review_state",
                    default={},
                ) or {}
                lane_summaries = dict(review_state.get("lane_summaries") or {})
                lane_summaries[lane_name] = {
                    "active_issue_count": 2,
                    "blocking_issue_count": 2,
                }
                review_state["lane_summaries"] = lane_summaries
                save_contract_artifact(
                    project_root,
                    "review_state",
                    review_state,
                    producer="test",
                )
                return {
                    "found": True,
                    "review_roles_used": list(kwargs["review_roles"]),
                    "review_text": {"review": {}},
                    "review_img": {},
                }

            with patch(
                "ai_scientist.utils.critic_workflow.execute_review_suite",
                side_effect=_fake_execute_review_suite,
            ):
                result = run_independent_critic_pass(
                    workflow_runtime_plan=plan,
                    paper_dir=project_root,
                    model_review="demo-model",
                    review_plan={
                        "review_instruction": "Review directly.",
                        "review_reflections": 1,
                        "review_fewshot": 1,
                        "review_ensemble": 1,
                        "review_temperature": 0.5,
                    },
                    create_client_fn=lambda model: (None, model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=lambda *args, **kwargs: None,
                    perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: None,
                    pdf_path_resolver=lambda _: str(project_root / "paper.pdf"),
                    save_dir=project_root / "critic",
                    project_root=project_root,
                )
            decisions = load_decision_log(project_root)

        self.assertEqual([call["lane_name"] for call in calls], ["hostile_critic"])
        self.assertFalse(result["critic_confirmation"]["ran"])
        self.assertFalse(result["critic_confirmed"])
        self.assertEqual(result["blocking_issue_count"], 2)
        self.assertEqual(result["blocking_source"], "hostile_critic")
        self.assertEqual(decisions[-1]["category"], "hostile_critic_confirmation")
        self.assertEqual(decisions[-1]["selected"], "skip_independent_confirmation")

    def test_role_level_primary_blocker_should_survive_later_clear_role(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            pdf_path = project_root / "paper.pdf"
            pdf_path.write_text("stub", encoding="utf-8")
            review_payloads = [
                {
                    "review": {
                        "Summary": "Blocking issue.",
                        "Weaknesses": ["The main claim is unsupported."],
                        "Questions": [],
                        "Limitations": [],
                        "scores": {"Overall": 2.0},
                    }
                },
                {
                    "review": {
                        "Summary": "No additional issue.",
                        "Weaknesses": [],
                        "Questions": [],
                        "Limitations": [],
                        "scores": {"Overall": 8.0},
                    }
                },
            ]
            calls = []

            def _fake_perform_review(*args, **kwargs):
                calls.append(kwargs.get("review_instruction_form"))
                return review_payloads[len(calls) - 1]

            two_role_plan = plan.__class__(
                workflow_mode=plan.workflow_mode,
                workflow_label=plan.workflow_label,
                stage_sequence=plan.stage_sequence,
                inspirations=plan.inspirations,
                agent_lanes=plan.agent_lanes,
                improvement_review_roles=plan.improvement_review_roles,
                final_review_roles=plan.final_review_roles,
                critic_review_roles=("novelty", "rigor"),
                requires_independent_critic=plan.requires_independent_critic,
                critic_strictness_profile=plan.critic_strictness_profile,
            )
            result = run_independent_critic_pass(
                workflow_runtime_plan=two_role_plan,
                paper_dir=project_root,
                model_review="demo-model",
                review_plan={
                    "review_instruction": "Review directly.",
                    "review_reflections": 1,
                    "review_fewshot": 1,
                    "review_ensemble": 1,
                    "review_temperature": 0.5,
                },
                create_client_fn=lambda model: (None, model),
                load_paper_fn=lambda path: "paper",
                perform_review_fn=_fake_perform_review,
                perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: {},
                pdf_path_resolver=lambda _: str(pdf_path),
                save_dir=project_root / "critic",
                project_root=project_root,
            )

        self.assertEqual(len(calls), 2)
        self.assertFalse(result["critic_confirmation"]["ran"])
        self.assertFalse(result["critic_confirmed"])
        self.assertEqual(result["primary_blocking_issue_count"], 1)
        self.assertEqual(result["blocking_source"], "hostile_critic")

    def test_confirmation_without_materialized_lane_should_block(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            calls = []

            def _fake_execute_review_suite(**kwargs):
                calls.append(kwargs)
                lane_name = kwargs["lane_name"]
                if lane_name == "hostile_critic":
                    save_contract_artifact(
                        project_root,
                        "review_state",
                        {
                            "lane_summaries": {
                                "hostile_critic": {
                                    "active_issue_count": 0,
                                    "blocking_issue_count": 0,
                                }
                            }
                        },
                        producer="test",
                    )
                    return {
                        "found": True,
                        "review_roles_used": list(kwargs["review_roles"]),
                        "primary_role": "novelty",
                        "passes_by_role": {
                            "novelty": {"review_text": {"review": {"Weaknesses": []}}}
                        },
                        "review_text": {"review": {"Weaknesses": []}},
                        "review_img": {},
                    }
                return {
                    "found": True,
                    "review_roles_used": list(kwargs["review_roles"]),
                    "primary_role": "novelty",
                    "passes_by_role": {},
                    "review_text": None,
                    "review_img": {},
                }

            with patch(
                "ai_scientist.utils.critic_workflow.execute_review_suite",
                side_effect=_fake_execute_review_suite,
            ):
                result = run_independent_critic_pass(
                    workflow_runtime_plan=plan,
                    paper_dir=project_root,
                    model_review="demo-model",
                    review_plan={
                        "review_instruction": "Review directly.",
                        "review_reflections": 1,
                        "review_fewshot": 1,
                        "review_ensemble": 1,
                        "review_temperature": 0.5,
                    },
                    create_client_fn=lambda model: (None, model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=lambda *args, **kwargs: None,
                    perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: None,
                    pdf_path_resolver=lambda _: str(project_root / "paper.pdf"),
                    save_dir=project_root / "critic",
                    project_root=project_root,
                )

        self.assertEqual([call["lane_name"] for call in calls], [
            "hostile_critic",
            "hostile_critic_confirmation",
        ])
        self.assertFalse(result["critic_confirmation"]["found"])
        self.assertFalse(result["critic_confirmation"]["confirmed"])
        self.assertEqual(
            result["critic_confirmation"]["reason"],
            "confirmation_missing_artifact",
        )
        self.assertEqual(result["critic_confirmation"]["blocking_issue_count"], 1)
        self.assertEqual(result["blocking_source"], "ep_independent_eval")
        self.assertGreaterEqual(result["active_issue_count"], result["blocking_issue_count"])

    def test_stale_confirmation_lane_summary_must_not_confirm_current_pass(self) -> None:
        plan = build_workflow_runtime_plan(
            "multi_agent_board",
            high_quality_mode=True,
            target_venue="neurips",
        )

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            save_contract_artifact(
                project_root,
                "review_state",
                {
                    "lane_summaries": {
                        "hostile_critic_confirmation": {
                            "active_issue_count": 0,
                            "blocking_issue_count": 0,
                        }
                    }
                },
                producer="test_stale",
            )

            def _fake_execute_review_suite(**kwargs):
                lane_name = kwargs["lane_name"]
                if lane_name == "hostile_critic":
                    return {
                        "found": True,
                        "review_roles_used": list(kwargs["review_roles"]),
                        "primary_role": "novelty",
                        "passes_by_role": {
                            "novelty": {"review_text": {"review": {"Weaknesses": []}}}
                        },
                        "review_text": {"review": {"Weaknesses": []}},
                        "review_img": {},
                    }
                return {
                    "found": True,
                    "review_roles_used": list(kwargs["review_roles"]),
                    "primary_role": "novelty",
                    "passes_by_role": {},
                    "review_text": None,
                    "review_img": {},
                }

            with patch(
                "ai_scientist.utils.critic_workflow.execute_review_suite",
                side_effect=_fake_execute_review_suite,
            ):
                result = run_independent_critic_pass(
                    workflow_runtime_plan=plan,
                    paper_dir=project_root,
                    model_review="demo-model",
                    review_plan={
                        "review_instruction": "Review directly.",
                        "review_reflections": 1,
                        "review_fewshot": 1,
                        "review_ensemble": 1,
                        "review_temperature": 0.5,
                    },
                    create_client_fn=lambda model: (None, model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=lambda *args, **kwargs: None,
                    perform_imgs_cap_ref_review_fn=lambda *args, **kwargs: None,
                    pdf_path_resolver=lambda _: str(project_root / "paper.pdf"),
                    save_dir=project_root / "critic",
                    project_root=project_root,
                )

        self.assertFalse(result["critic_confirmation"]["found"])
        self.assertFalse(result["critic_confirmed"])
        self.assertEqual(
            result["critic_confirmation"]["reason"],
            "confirmation_missing_artifact",
        )
        self.assertEqual(result["critic_confirmation"]["blocking_issue_count"], 1)
        self.assertEqual(result["blocking_source"], "ep_independent_eval")


if __name__ == "__main__":
    unittest.main()
