from __future__ import annotations

import unittest

from ai_scientist.utils.workflow_execution_policy import (
    build_workflow_execution_policy,
)


class WorkflowExecutionPolicyTests(unittest.TestCase):
    def test_program_driven_policy_should_be_budget_disciplined(self) -> None:
        policy = build_workflow_execution_policy("program_driven")
        self.assertEqual(policy.execution_style, "budgeted_program_execution")
        self.assertEqual(policy.budget["max_retry_per_task"], 1)
        self.assertFalse(policy.allow_auto_improvement_fallback)
        self.assertTrue(policy.reject_on_auto_improvement_fallback)
        self.assertIn(
            "Every task must declare success_criterion and stop_condition before execution.",
            policy.acceptance_rules,
        )

    def test_agentic_tree_policy_should_allow_more_exploration(self) -> None:
        policy = build_workflow_execution_policy("agentic_tree")
        self.assertEqual(policy.evidence_pressure, "exploratory")
        self.assertGreater(policy.budget["max_steps"], 12)
        self.assertTrue(policy.allow_auto_improvement_fallback)
        self.assertIn("branch", " ".join(policy.logging_requirements).lower())

    def test_nature_target_should_force_strict_quality_fallback_policy(self) -> None:
        policy = build_workflow_execution_policy(
            "classic_pipeline",
            high_quality_mode=True,
            target_venue="nature",
        )
        self.assertEqual(policy.quality_fallback_policy, "disallowed")
        self.assertFalse(policy.allow_auto_improvement_fallback)
        self.assertTrue(policy.reject_on_auto_improvement_fallback)

    def test_multi_agent_board_policy_should_require_hostile_critic_discipline(self) -> None:
        policy = build_workflow_execution_policy("multi_agent_board")
        self.assertEqual(policy.execution_style, "multi_agent_submission_board")
        self.assertEqual(policy.evidence_pressure, "adversarial_submission_grade")
        self.assertFalse(policy.allow_auto_improvement_fallback)
        self.assertTrue(policy.reject_on_auto_improvement_fallback)
        self.assertIn(
            "A hostile critic blocker must either trigger repair or block submission readiness.",
            policy.acceptance_rules,
        )


if __name__ == "__main__":
    unittest.main()
