from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from continuous_research_daemon import (
    _apply_auto_source_plan,
    _apply_pipeline_contract_feedback,
    _build_source_batch_plan_rows,
    _hydrate_source_next_batch_advisory,
    _build_source_runtime_rows,
    _start_dashboard_server,
)


class DaemonContractFeedbackTests(unittest.TestCase):
    def _parsed(self) -> SimpleNamespace:
        return SimpleNamespace(
            research_dir="/tmp/research",
            guardrail_default_num_ideas=4,
        )

    def test_pipeline_contract_feedback_should_stay_neutral_when_summary_is_healthy(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = ["--num-ideas", "4", "--quality-rewrite-rounds", "1"]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 0,
                "failed_experiment_count": 0,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 0,
                "dominant_execution_policy": "classic_pipeline",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertEqual(updated, passthrough)
        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "contracts_healthy",
        )

    def test_pipeline_contract_feedback_should_intensify_repair_when_blocked(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "5",
            "--quality-rewrite-rounds",
            "1",
            "--guardrail-repair-rounds",
            "1",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 2,
                "failed_experiment_count": 3,
                "blocked_figure_count": 4,
                "budget_exhausted_experiment_count": 0,
                "dominant_execution_policy": "review_board",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertIn("--high-quality-mode", updated)
        self.assertIn("--strict-writing-guardrails", updated)
        self.assertIn("--review-strategy", updated)
        self.assertEqual(
            updated[updated.index("--review-strategy") + 1],
            "depth",
        )
        self.assertEqual(
            updated[updated.index("--quality-rewrite-rounds") + 1],
            "3",
        )
        self.assertEqual(
            updated[updated.index("--guardrail-repair-rounds") + 1],
            "2",
        )
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "2")
        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "review_board_hardening",
        )
        self.assertEqual(
            updated[updated.index("--workflow-mode") + 1],
            "review_board",
        )

    def test_pipeline_contract_feedback_should_switch_to_program_budget_repair(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "5",
            "--workflow-mode",
            "classic_pipeline",
            "--breakthrough-mode",
            "--quality-rewrite-rounds",
            "1",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 0,
                "failed_experiment_count": 2,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 3,
                "dominant_execution_policy": "program_driven",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "program_budget_repair",
        )
        self.assertEqual(
            updated[updated.index("--workflow-mode") + 1],
            "program_driven",
        )
        self.assertIn("--submission-mode", updated)
        self.assertNotIn("--breakthrough-mode", updated)
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "2")
        self.assertEqual(updated[updated.index("--writing-audit-rounds") + 1], "1")

    def test_pipeline_contract_feedback_should_expand_agentic_exploration_when_needed(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "2",
            "--workflow-mode",
            "classic_pipeline",
            "--submission-mode",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 1,
                "failed_experiment_count": 4,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 0,
                "dominant_execution_policy": "agentic_tree",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "agentic_exploration_rebuild",
        )
        self.assertEqual(
            updated[updated.index("--workflow-mode") + 1],
            "agentic_tree",
        )
        self.assertIn("--breakthrough-mode", updated)
        self.assertNotIn("--submission-mode", updated)
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "4")
        self.assertEqual(
            updated[updated.index("--autonomous-quality-followup-rounds") + 1],
            "1",
        )

    def test_pipeline_contract_feedback_should_treat_strict_fallback_debt_as_pressure(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "5",
            "--quality-rewrite-rounds",
            "1",
            "--guardrail-repair-rounds",
            "1",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 0,
                "failed_experiment_count": 0,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 0,
                "fallback_count": 3,
                "strict_fallback_count": 2,
                "fallback_heavy_project_count": 1,
                "dominant_execution_policy": "classic_pipeline",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertIn("--high-quality-mode", updated)
        self.assertIn("--strict-writing-guardrails", updated)
        self.assertEqual(updated[updated.index("--quality-rewrite-rounds") + 1], "3")
        self.assertEqual(updated[updated.index("--guardrail-repair-rounds") + 1], "2")
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "2")
        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "contract_blocker_repair",
        )

    def test_pipeline_contract_feedback_should_treat_blocked_stage_standards_as_pressure(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "5",
            "--quality-rewrite-rounds",
            "1",
            "--guardrail-repair-rounds",
            "1",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 0,
                "stage_blocked_project_count": 2,
                "stage_missing_project_count": 0,
                "stage_attention_project_count": 1,
                "failed_experiment_count": 0,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 0,
                "fallback_count": 0,
                "strict_fallback_count": 0,
                "fallback_heavy_project_count": 0,
                "dominant_execution_policy": "review_board",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertIn("--high-quality-mode", updated)
        self.assertIn("--strict-writing-guardrails", updated)
        self.assertEqual(updated[updated.index("--quality-rewrite-rounds") + 1], "3")
        self.assertEqual(updated[updated.index("--guardrail-repair-rounds") + 1], "2")
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "2")
        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "review_board_hardening",
        )
        self.assertIn("blocked stage standards", status["active_pipeline_contract_strategy"]["reason"])

    def test_pipeline_contract_feedback_should_treat_review_resolution_debt_as_pressure(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "5",
            "--quality-rewrite-rounds",
            "1",
            "--guardrail-repair-rounds",
            "1",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 0,
                "stage_blocked_project_count": 0,
                "stage_missing_project_count": 0,
                "stage_attention_project_count": 0,
                "review_low_resolution_project_count": 2,
                "review_low_binding_project_count": 2,
                "review_low_repair_ready_project_count": 2,
                "review_persistent_issue_count": 4,
                "failed_experiment_count": 0,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 0,
                "fallback_count": 0,
                "strict_fallback_count": 0,
                "fallback_heavy_project_count": 0,
                "dominant_execution_policy": "classic_pipeline",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertIn("--high-quality-mode", updated)
        self.assertIn("--strict-writing-guardrails", updated)
        self.assertEqual(updated[updated.index("--quality-rewrite-rounds") + 1], "3")
        self.assertEqual(updated[updated.index("--guardrail-repair-rounds") + 1], "2")
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "2")
        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "contract_blocker_repair",
        )

    def test_pipeline_contract_feedback_should_rebuild_when_self_evolution_is_blocked(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "5",
            "--workflow-mode",
            "classic_pipeline",
            "--quality-rewrite-rounds",
            "1",
            "--guardrail-repair-rounds",
            "1",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 0,
                "failed_experiment_count": 0,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 0,
                "blocked_self_evolution_project_count": 1,
                "self_evolution_attention_project_count": 0,
                "self_evolution_required_failure_count": 2,
                "dominant_execution_policy": "classic_pipeline",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "self_evolution_rebuild",
        )
        self.assertEqual(
            updated[updated.index("--workflow-mode") + 1],
            "review_board",
        )
        self.assertIn("--high-quality-mode", updated)
        self.assertIn("--strict-writing-guardrails", updated)
        self.assertEqual(
            updated[updated.index("--writing-audit-rounds") + 1],
            "1",
        )
        self.assertEqual(
            updated[updated.index("--quality-rewrite-rounds") + 1],
            "3",
        )
        self.assertEqual(
            updated[updated.index("--guardrail-repair-rounds") + 1],
            "2",
        )
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "2")
        self.assertIn(
            "self-evolution self-checks",
            status["active_pipeline_contract_strategy"]["reason"],
        )

    def test_pipeline_contract_feedback_should_repair_process_alignment_gaps(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {}
        passthrough = [
            "--num-ideas",
            "5",
            "--workflow-mode",
            "classic_pipeline",
            "--quality-rewrite-rounds",
            "1",
        ]

        with mock.patch(
            "continuous_research_daemon.ResearchManager", return_value=object()
        ), mock.patch(
            "continuous_research_daemon._build_pipeline_contract_summary",
            return_value={
                "enabled": True,
                "blocked_project_count": 0,
                "failed_experiment_count": 0,
                "blocked_figure_count": 0,
                "budget_exhausted_experiment_count": 0,
                "process_alignment_blocked_project_count": 1,
                "process_alignment_missing_project_count": 2,
                "dominant_execution_policy": "classic_pipeline",
            },
        ):
            updated = _apply_pipeline_contract_feedback(parsed, status, passthrough)

        self.assertEqual(
            status["active_pipeline_contract_strategy"]["mode"],
            "process_alignment_repair",
        )
        self.assertEqual(
            updated[updated.index("--workflow-mode") + 1],
            "program_driven",
        )
        self.assertIn("--high-quality-mode", updated)
        self.assertIn("--strict-writing-guardrails", updated)
        self.assertEqual(
            updated[updated.index("--quality-rewrite-rounds") + 1],
            "3",
        )
        self.assertEqual(updated[updated.index("--num-ideas") + 1], "2")

    def test_start_dashboard_server_should_fallback_to_static_file_when_bind_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "continuous_research_daemon.ThreadingHTTPServer",
            side_effect=PermissionError("sandbox denied"),
        ):
            daemon_dir = Path(td)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                server, thread, url = _start_dashboard_server(
                    daemon_dir, "127.0.0.1", 0
                )

        self.assertIsNone(server)
        self.assertIsNone(thread)
        self.assertTrue(url.endswith("latest_live_dashboard.html"))
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Dashboard server unavailable", stderr.getvalue())

    def test_source_runtime_rows_should_prefer_program_aligned_sources_under_program_repair(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {
            "cycle": 0,
            "current_daypart": "day",
            "active_pipeline_contract_strategy": {
                "selected_execution_policy": "program_driven"
            },
            "source_quality_feedback": {},
        }
        queue = [
            {
                "name": "explore_topic",
                "type": "topic",
                "value": "topic.md",
                "priority": 1.0,
                "submission_mode": False,
                "breakthrough_mode": True,
                "paper_types": ["normal"],
                "target_venue": "neurips",
                "time_of_day_preference": "any",
            },
            {
                "name": "submission_ideas",
                "type": "ideas",
                "value": "ideas.json",
                "priority": 1.0,
                "submission_mode": True,
                "breakthrough_mode": False,
                "paper_types": ["journal"],
                "target_venue": "nature",
                "time_of_day_preference": "any",
            },
        ]

        with mock.patch(
            "continuous_research_daemon._load_source_queue",
            return_value=queue,
        ):
            rows = _build_source_runtime_rows(parsed, status)

        self.assertEqual(rows[0]["name"], "submission_ideas")
        self.assertEqual(rows[0]["preferred_execution_policy"], "program_driven")
        self.assertEqual(rows[0]["resolved_workflow_mode"], "program_driven")
        self.assertEqual(rows[0]["batch_profile"], "submission_push")
        self.assertGreater(rows[0]["workflow_alignment_score"], rows[1]["workflow_alignment_score"])

    def test_source_runtime_rows_should_prefer_breakthrough_sources_under_agentic_rebuild(
        self,
    ) -> None:
        parsed = self._parsed()
        status = {
            "cycle": 0,
            "current_daypart": "day",
            "active_pipeline_contract_strategy": {
                "selected_execution_policy": "agentic_tree"
            },
            "source_quality_feedback": {},
        }
        queue = [
            {
                "name": "submission_ideas",
                "type": "ideas",
                "value": "ideas.json",
                "priority": 1.0,
                "submission_mode": True,
                "breakthrough_mode": False,
                "paper_types": ["journal"],
                "target_venue": "nature",
                "time_of_day_preference": "any",
            },
            {
                "name": "explore_topic",
                "type": "topic",
                "value": "topic.md",
                "priority": 1.0,
                "submission_mode": False,
                "breakthrough_mode": True,
                "paper_types": ["normal"],
                "target_venue": "neurips",
                "time_of_day_preference": "any",
            },
        ]

        with mock.patch(
            "continuous_research_daemon._load_source_queue",
            return_value=queue,
        ):
            rows = _build_source_runtime_rows(parsed, status)

        self.assertEqual(rows[0]["name"], "explore_topic")
        self.assertEqual(rows[0]["preferred_execution_policy"], "agentic_tree")
        self.assertEqual(rows[0]["resolved_workflow_mode"], "agentic_tree")
        self.assertEqual(rows[0]["batch_profile"], "exploration_sprint")
        self.assertGreater(rows[0]["workflow_alignment_score"], rows[1]["workflow_alignment_score"])

    def test_source_batch_plan_rows_should_prioritize_ready_aligned_sources(self) -> None:
        rows = [
            {
                "name": "aligned_ready",
                "availability_state": "ready",
                "resolved_workflow_mode": "program_driven",
                "source_archetype": "program_guarded",
                "source_archetype_label": "Program Guarded",
                "batch_profile": "submission_push",
                "batch_profile_label": "Submission Push",
                "batch_goal": "Convert to submission-grade artifacts.",
                "workflow_alignment_score": 5,
                "health_score": 96.0,
                "recommended_generator_preview": ["--workflow-mode program_driven"],
                "planning_notes": "Best next source.",
                "alignment_tags": ["budgeted"],
                "archetype_inspirations": ["karpathy/autoresearch"],
            },
            {
                "name": "blocked_source",
                "availability_state": "cooldown",
                "resolved_workflow_mode": "review_board",
                "source_archetype": "review_hardening",
                "source_archetype_label": "Review Hardening",
                "batch_profile": "review_hardening",
                "batch_profile_label": "Review Hardening",
                "batch_goal": "Repair before next pass.",
                "workflow_alignment_score": 2,
                "health_score": 70.0,
                "recommended_generator_preview": ["--workflow-mode review_board"],
                "planning_notes": "",
                "alignment_tags": [],
                "archetype_inspirations": ["ResearAI/DeepReviewer-v2"],
            },
        ]

        plan = _build_source_batch_plan_rows(rows)

        self.assertEqual(plan[0]["tier"], "run-now")
        self.assertEqual(plan[0]["source"], "aligned_ready")
        self.assertIn("best matches", plan[0]["recommendation"])
        self.assertEqual(plan[-1]["tier"], "defer")
        self.assertEqual(plan[-1]["source"], "blocked_source")

    def test_auto_source_plan_should_force_policy_aligned_source_from_mix(self) -> None:
        parsed = SimpleNamespace(
            auto_apply_source_plan=True,
            auto_source_plan_min_health=70,
            auto_source_plan_max_actions=1,
            auto_source_plan_expires_after_cycles=2,
        )
        status = {"cycle": 3, "quality_governor": {}, "control": {}}
        brief = {
            "source_advisory": [],
            "source_runtime_rows": [
                {
                    "name": "program_source",
                    "key": "topic::program_source::topic.md",
                    "availability_state": "ready",
                    "health_score": 93.0,
                    "resolved_workflow_mode": "program_driven",
                    "compatible_workflow_modes": ["program_driven", "review_board"],
                    "source_archetype": "program_guarded",
                    "workflow_alignment_score": 5,
                }
            ],
            "source_mix_advisory": {
                "desired_policy": "program_driven",
                "summary": {
                    "dominant_archetype": "review_hardening",
                    "dominant_workflow_mode": "review_board",
                },
                "recommendations": [],
            },
        }

        with tempfile.TemporaryDirectory() as td:
            updated = _apply_auto_source_plan(status, Path(td), brief, parsed)

        applied = updated["auto_source_plan"]["applied"]
        self.assertEqual(applied[0]["source"], "program_source")
        self.assertEqual(applied[0]["operation"], "source-force-next")
        self.assertEqual(applied[0]["source_plan_origin"], "mix_advisory")
        self.assertEqual(applied[0]["mix_reason"], "desired_policy_alignment")

    def test_auto_source_plan_should_rebalance_mix_when_too_narrow(self) -> None:
        parsed = SimpleNamespace(
            auto_apply_source_plan=True,
            auto_source_plan_min_health=70,
            auto_source_plan_max_actions=1,
            auto_source_plan_expires_after_cycles=2,
        )
        status = {"cycle": 5, "quality_governor": {}, "control": {}}
        brief = {
            "source_advisory": [],
            "source_runtime_rows": [
                {
                    "name": "template_source",
                    "key": "topic::template_source::topic.md",
                    "availability_state": "ready",
                    "health_score": 95.0,
                    "resolved_workflow_mode": "classic_pipeline",
                    "compatible_workflow_modes": ["classic_pipeline"],
                    "source_archetype": "template_first",
                    "workflow_alignment_score": 4,
                },
                {
                    "name": "frontier_source",
                    "key": "topic::frontier_source::topic.md",
                    "availability_state": "ready",
                    "health_score": 82.0,
                    "resolved_workflow_mode": "agentic_tree",
                    "compatible_workflow_modes": ["agentic_tree"],
                    "source_archetype": "frontier_exploration",
                    "workflow_alignment_score": 2,
                },
            ],
            "source_mix_advisory": {
                "desired_policy": None,
                "summary": {
                    "dominant_archetype": "template_first",
                    "dominant_workflow_mode": "classic_pipeline",
                },
                "recommendations": [
                    {
                        "label": "mix_too_narrow",
                        "recommendation": "The source mix is narrow and dominated by template_first.",
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as td:
            updated = _apply_auto_source_plan(status, Path(td), brief, parsed)

        applied = updated["auto_source_plan"]["applied"]
        self.assertEqual(applied[0]["source"], "frontier_source")
        self.assertEqual(applied[0]["operation"], "source-boost-next")
        self.assertEqual(applied[0]["value"], 2)
        self.assertEqual(applied[0]["source_plan_origin"], "mix_advisory")
        self.assertEqual(applied[0]["mix_reason"], "mix_too_narrow")

    def test_hydrate_source_next_batch_advisory_should_attach_runtime_fields(self) -> None:
        advisory = {
            "desired_policy": "program_driven",
            "cadence": {"label": "submission_hardening_loop", "reason": "tighten"},
            "slots": [
                {
                    "lane": "primary_lane",
                    "source": "program_source",
                    "source_key": "topic::program_source::program.md",
                    "source_workflow_mode": "program_driven",
                    "source_archetype": "program_guarded",
                    "source_batch_profile": "submission_push",
                    "share": 0.5,
                    "rationale": "Best source.",
                }
            ],
            "recommendations": [],
        }
        rows = [
            {
                "name": "program_source",
                "key": "topic::program_source::program.md",
                "availability_state": "ready",
                "health_score": 92.0,
                "workflow_alignment_score": 5,
                "recommended_generator_preview": ["--workflow-mode program_driven"],
                "planning_notes": "Submission ready.",
                "alignment_tags": ["budgeted", "submission"],
            }
        ]

        hydrated = _hydrate_source_next_batch_advisory(advisory, rows)

        slot = hydrated["slots"][0]
        self.assertEqual(slot["availability_state"], "ready")
        self.assertEqual(slot["health_score"], 92.0)
        self.assertEqual(slot["workflow_alignment_score"], 5)
        self.assertEqual(
            slot["recommended_generator_preview"],
            ["--workflow-mode program_driven"],
        )
        self.assertEqual(slot["planning_notes"], "Submission ready.")
        self.assertEqual(slot["alignment_tags"], ["budgeted", "submission"])


if __name__ == "__main__":
    unittest.main()
