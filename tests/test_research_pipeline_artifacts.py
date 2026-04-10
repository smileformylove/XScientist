from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.figure_spec import build_figure_spec_from_summaries
from ai_scientist.utils.manuscript_state import build_manuscript_state
from ai_scientist.utils.research_planning import (
    build_claim_evidence_graph,
    build_idea_cards,
    build_research_plan,
)


class ResearchPipelineArtifactsTests(unittest.TestCase):
    def test_planning_and_figure_spec_should_produce_ready_manuscript_state(self) -> None:
        ideas = [
            {
                "Name": "Idea Alpha",
                "Title": "Idea Alpha Title",
                "Short Hypothesis": "Structured planning improves scientific workflow quality.",
                "Experiments": [
                    "Compare the new method against a strong baseline on dataset: arxiv-bench with metric: accuracy.",
                    "Run an ablation on the planning module with baseline: no-planning.",
                ],
                "Related Work": "baseline: chain-of-thought pipeline",
            }
        ]
        idea_card = build_idea_cards(
            ideas,
            target_venue="neurips",
            workflow_mode="program_driven",
        )[0]
        research_plan = build_research_plan(idea_card, target_venue="neurips")
        claim_graph = build_claim_evidence_graph(idea_card, research_plan)
        figure_spec = build_figure_spec_from_summaries(
            {
                "baseline_summary": {
                    "exp_results_npy_files": ["results/baseline.npy"],
                },
                "ablation_summary": {
                    "exp_results_npy_files": ["results/ablation.npy"],
                },
            },
            claim_evidence_graph=claim_graph,
        )
        manuscript_state = build_manuscript_state(
            writeup_type="normal",
            target_venue="neurips",
            writing_profile="default",
            skill_pack=["abstract_framing", "results_narrative"],
            claim_evidence_graph=claim_graph,
            figure_spec=figure_spec,
            latex_path="/tmp/template.tex",
        )

        self.assertEqual(idea_card["candidate_datasets"][0], "arxiv-bench with metric")
        self.assertGreaterEqual(len(research_plan["tasks"]), 2)
        self.assertEqual(research_plan["workflow_mode"], "program_driven")
        self.assertEqual(
            research_plan["execution_policy"]["policy_name"],
            "program_driven",
        )
        self.assertEqual(
            research_plan["execution_policy"]["quality_fallback_policy"],
            "disallowed",
        )
        self.assertEqual(research_plan["budget"]["max_retry_per_task"], 1)
        self.assertEqual(research_plan["tasks"][0]["task_kind"], "program_milestone")
        self.assertTrue(research_plan["tasks"][0]["acceptance_checks"])
        self.assertEqual(figure_spec["figure_count"], 2)
        self.assertEqual(manuscript_state["guardrail_status"], "ready")
        self.assertFalse(manuscript_state["missing_evidence"])
        self.assertIn("results", manuscript_state["section_claim_bindings"])
        self.assertIn("claim_0", manuscript_state["section_claim_bindings"]["results"])
        self.assertTrue(manuscript_state["section_figure_bindings"]["results"])
        self.assertEqual(
            figure_spec["summary"]["claim_coverage_ratio"],
            1.0,
        )
        self.assertGreaterEqual(
            figure_spec["summary"]["main_ready_count"],
            2,
        )
        self.assertEqual(
            manuscript_state["claim_figure_bindings"]["claim_0"],
            ["figure_0"],
        )

    def test_manuscript_state_should_block_when_claim_has_no_ready_figure(self) -> None:
        ideas = [
            {
                "Name": "Idea Beta",
                "Short Hypothesis": "Missing evidence should be surfaced.",
                "Experiments": ["Run one comparison study."],
            }
        ]
        idea_card = build_idea_cards(ideas, target_venue="iclr")[0]
        research_plan = build_research_plan(idea_card, target_venue="iclr")
        claim_graph = build_claim_evidence_graph(idea_card, research_plan)
        figure_spec = build_figure_spec_from_summaries(
            {"baseline_summary": {}},
            claim_evidence_graph=claim_graph,
        )

        manuscript_state = build_manuscript_state(
            writeup_type="normal",
            target_venue="iclr",
            writing_profile="default",
            skill_pack=["method_clarity"],
            claim_evidence_graph=claim_graph,
            figure_spec=figure_spec,
        )

        self.assertEqual(manuscript_state["guardrail_status"], "blocked")
        self.assertTrue(manuscript_state["missing_evidence"])

    def test_figure_spec_should_block_when_checked_data_files_are_missing(self) -> None:
        ideas = [
            {
                "Name": "Idea Gamma",
                "Short Hypothesis": "Visual evidence should stay traceable.",
                "Experiments": ["Run one comparison study."],
            }
        ]
        idea_card = build_idea_cards(ideas, target_venue="cvpr", workflow_mode="review_board")[0]
        research_plan = build_research_plan(idea_card, target_venue="cvpr")
        claim_graph = build_claim_evidence_graph(idea_card, research_plan)

        with tempfile.TemporaryDirectory() as td:
            figure_spec = build_figure_spec_from_summaries(
                {
                    "baseline_summary": {
                        "exp_results_npy_files": ["results/missing.npy"],
                    }
                },
                claim_evidence_graph=claim_graph,
                base_folder=td,
            )

        self.assertEqual(figure_spec["figures"][0]["status"], "blocked")
        self.assertIn(
            "missing_data_files",
            figure_spec["figures"][0]["blocking_reasons"],
        )
        self.assertTrue(figure_spec["summary"]["checked_data_file_availability"])
        self.assertEqual(figure_spec["summary"]["ready_missing_data_file_count"], 0)
        self.assertEqual(figure_spec["summary"]["missing_data_file_count"], 1)

    def test_multi_agent_board_plan_should_include_agent_ownership_and_kill_criteria(self) -> None:
        ideas = [
            {
                "Name": "Idea Delta",
                "Short Hypothesis": "A multi-agent board should harden the paper pipeline.",
                "Experiments": [
                    "Compare against baseline: vanilla on dataset: demo-set with metric: accuracy.",
                    "Run a stronger ablation on the coordination mechanism.",
                ],
                "Risk Factors and Limitations": [
                    "If the critic still finds unsupported claims, kill the branch.",
                ],
            }
        ]
        idea_card = build_idea_cards(
            ideas,
            target_venue="neurips",
            workflow_mode="multi_agent_board",
        )[0]
        research_plan = build_research_plan(idea_card, target_venue="neurips")

        self.assertEqual(research_plan["workflow_mode"], "multi_agent_board")
        self.assertTrue(research_plan["agent_plan"]["requires_hostile_critic"])
        self.assertIn(
            "hostile_critic",
            [item["lane"] for item in research_plan["agent_plan"]["lanes"]],
        )
        self.assertEqual(research_plan["tasks"][0]["owner"], "experiment_manager")
        self.assertEqual(research_plan["tasks"][1]["dependencies"], ["task_0"])
        self.assertIn("keep", research_plan["tasks"][0]["branch_outcome_labels"])
        self.assertTrue(research_plan["tasks"][0]["kill_criteria"])
        self.assertEqual(research_plan["tasks"][0]["verifier"], "quality_gate")
        self.assertEqual(research_plan["tasks"][0]["escalation_lane"], "hostile_critic")
        self.assertTrue(research_plan["tasks"][0]["required_inputs"])
        self.assertTrue(research_plan["tasks"][0]["produced_artifacts"])
        self.assertIn(
            "meta_reviewer",
            research_plan["agent_plan"]["review_bundles"]["final_roles"],
        )
        self.assertTrue(research_plan["agent_plan"]["phase_gates"])


if __name__ == "__main__":
    unittest.main()
