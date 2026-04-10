from __future__ import annotations

import unittest

from ai_scientist.utils.source_planning import (
    build_source_planning_profile,
    infer_source_archetype,
)


class SourcePlanningTests(unittest.TestCase):
    def test_build_source_planning_profile_should_honor_explicit_review_settings(
        self,
    ) -> None:
        source = {
            "name": "review_source",
            "type": "topic",
            "value": "topic.md",
            "source_archetype": "adaptive",
            "day_source_archetype": "review_hardening",
            "batch_profile": "adaptive",
            "day_batch_profile": "review_hardening",
            "workflow_mode": "adaptive",
            "day_workflow_mode": "review_board",
            "workflow_modes": ["program_driven", "review_board"],
            "day_workflow_modes": ["review_board"],
            "alignment_tags": ["review", "camera-ready"],
            "day_alignment_tags": ["clarity", "repro"],
            "planning_notes": "Harden this source before the next run.",
        }

        profile = build_source_planning_profile(
            source,
            daypart="day",
            desired_execution_policy="program_driven",
        )

        self.assertEqual(profile["resolved_workflow_mode"], "review_board")
        self.assertEqual(profile["source_archetype"], "review_hardening")
        self.assertEqual(profile["batch_profile"], "review_hardening")
        self.assertIn("--review-strategy depth", profile["recommended_generator_preview"])
        self.assertIn("--submission-mode", profile["recommended_generator_preview"])
        self.assertEqual(profile["alignment_tags"], ["clarity", "repro"])

    def test_build_source_planning_profile_should_infer_frontier_defaults(
        self,
    ) -> None:
        source = {
            "name": "frontier_source",
            "type": "topic",
            "value": "topic.md",
            "breakthrough_mode": True,
            "submission_mode": False,
            "paper_types": ["normal"],
        }

        profile = build_source_planning_profile(source, daypart="night")

        self.assertEqual(profile["source_archetype"], "frontier_exploration")
        self.assertEqual(profile["resolved_workflow_mode"], "agentic_tree")
        self.assertEqual(profile["batch_profile"], "exploration_sprint")
        self.assertIn("--breakthrough-mode", profile["recommended_generator_preview"])

    def test_infer_source_archetype_should_favor_program_guarded_for_submission(
        self,
    ) -> None:
        source = {
            "submission_mode": True,
            "target_venue": "nature",
            "paper_types": ["journal"],
        }

        self.assertEqual(infer_source_archetype(source), "program_guarded")

    def test_build_source_planning_profile_should_support_multi_agent_board_defaults(
        self,
    ) -> None:
        source = {
            "name": "paper_board_source",
            "type": "topic",
            "value": "topic.md",
            "source_archetype": "paper_hardening_board",
            "batch_profile": "paper_hardening",
            "workflow_mode": "multi_agent_board",
        }

        profile = build_source_planning_profile(source, daypart="day")

        self.assertEqual(profile["resolved_workflow_mode"], "multi_agent_board")
        self.assertEqual(profile["source_archetype"], "paper_hardening_board")
        self.assertEqual(profile["batch_profile"], "paper_hardening")
        self.assertIn("--quality-rewrite-rounds 2", profile["recommended_generator_preview"])


if __name__ == "__main__":
    unittest.main()
