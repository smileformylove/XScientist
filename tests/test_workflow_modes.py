from __future__ import annotations

import argparse
import unittest

from ai_scientist.utils.workflow_modes import (
    apply_workflow_mode_defaults,
    build_workflow_manifest_metadata,
    resolve_template_mode_for_workflow,
    resolve_workflow_mode,
)


class WorkflowModesTests(unittest.TestCase):
    def test_resolve_workflow_mode_should_choose_program_driven_for_submission_mode(self) -> None:
        spec = resolve_workflow_mode(
            "adaptive",
            submission_mode=True,
            high_quality_mode=False,
            target_venue="neurips",
        )
        self.assertEqual(spec.name, "program_driven")

    def test_apply_workflow_mode_defaults_should_harden_review_board_runs(self) -> None:
        args = argparse.Namespace(
            workflow_mode="review_board",
            submission_mode=False,
            breakthrough_mode=False,
            high_quality_mode=False,
            target_venue="nature",
            require_quality_gate=False,
            strict_writing_guardrails=False,
            review_reflections=1,
            review_ensemble=1,
            review_fewshot=1,
            review_temperature=0.8,
            guardrail_repair_rounds=0,
            review_strategy=None,
            rank_ideas=False,
            top_k_ideas=None,
            fallback_ranked_ideas=False,
        )
        spec = apply_workflow_mode_defaults(
            args,
            rank_flag_attr="rank_ideas",
            candidate_limit_attr="top_k_ideas",
            fallback_flag_attr="fallback_ranked_ideas",
        )
        self.assertEqual(spec.name, "review_board")
        self.assertTrue(args.high_quality_mode)
        self.assertTrue(args.require_quality_gate)
        self.assertTrue(args.strict_writing_guardrails)
        self.assertEqual(args.review_reflections, 2)
        self.assertEqual(args.review_ensemble, 3)
        self.assertEqual(args.review_fewshot, 2)
        self.assertEqual(args.review_temperature, 0.65)
        self.assertEqual(args.review_strategy, "nature")

    def test_manifest_metadata_and_template_mode_should_reflect_workflow(self) -> None:
        metadata = build_workflow_manifest_metadata("agentic_tree")
        template_profile, template_capability = resolve_template_mode_for_workflow(
            "agentic_tree",
            submission_mode=False,
        )
        self.assertEqual(metadata["workflow_mode"], "agentic_tree")
        self.assertIn("SakanaAI/AI-Scientist-v2", metadata["workflow_inspirations"])
        self.assertEqual(template_profile, "open_ended")
        self.assertEqual(template_capability, "agentic_search")

    def test_multi_agent_board_should_enable_submission_grade_defaults(self) -> None:
        args = argparse.Namespace(
            workflow_mode="multi_agent_board",
            submission_mode=False,
            breakthrough_mode=False,
            high_quality_mode=False,
            target_venue="neurips",
            require_quality_gate=False,
            strict_writing_guardrails=False,
            review_reflections=1,
            review_ensemble=1,
            review_fewshot=1,
            review_temperature=0.8,
            quality_rewrite_rounds=0,
            autonomous_quality_followup_rounds=0,
            writing_audit_rounds=0,
            guardrail_repair_rounds=0,
            num_cite_rounds=8,
            writeup_retries=1,
            review_strategy=None,
            rank_ideas=False,
            top_k_ideas=None,
            fallback_ranked_ideas=False,
            quality_preset="balanced",
        )
        spec = apply_workflow_mode_defaults(
            args,
            rank_flag_attr="rank_ideas",
            candidate_limit_attr="top_k_ideas",
            fallback_flag_attr="fallback_ranked_ideas",
        )
        self.assertEqual(spec.name, "multi_agent_board")
        self.assertTrue(args.high_quality_mode)
        self.assertTrue(args.require_quality_gate)
        self.assertTrue(args.strict_writing_guardrails)
        self.assertEqual(args.review_ensemble, 4)
        self.assertEqual(args.review_fewshot, 2)
        self.assertEqual(args.top_k_ideas, 2)
        self.assertEqual(args.quality_preset, "publishable")
        template_profile, template_capability = resolve_template_mode_for_workflow(
            "multi_agent_board",
            submission_mode=False,
        )
        self.assertEqual(template_profile, "program_driven")
        self.assertEqual(template_capability, "multi_agent_board")


if __name__ == "__main__":
    unittest.main()
