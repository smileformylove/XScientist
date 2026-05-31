from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.pipeline_contracts import initialize_pipeline_contracts
from ai_scientist.utils.repair_reflection import (
    apply_reflection_to_task,
    reflect_issue_repair_plan,
)


def _make_task() -> dict:
    return {
        "task_id": "repair_task_0",
        "issue_id": "RVW-1",
        "issue_text": "Clarify the new ablation evidence.",
        "role": "rigor",
        "severity": "major",
        "lane": "evidence_followup",
        "blocker_class": "evidence_hole",
        "priority_tier": "p1",
        "review_lane": "review",
        "primary_target_type": "claim",
        "primary_target_id": "claim_0",
        "primary_target_label": "Main Claim",
        "claim_ids": ["claim_0"],
        "figure_ids": [],
        "section_ids": [],
        "depends_on": ["repair_actions"],
    }


def _make_review_state() -> dict:
    return {
        "schema_version": 1,
        "rounds": [{"job_id": "rigor_0"}],
        "role_summaries": {
            "rigor": {
                "scores": {"Soundness": 5, "Quality": 5},
                "weaknesses": ["Ablation is thin."],
                "questions": ["What about variance?"],
                "limitations": [],
            }
        },
    }


class ReflectionFallbackTests(unittest.TestCase):
    def test_llm_exception_falls_back_to_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            def _boom(*args, **kwargs):
                raise RuntimeError("simulated")

            with patch("ai_scientist.utils.repair_reflection.get_response_from_llm", side_effect=_boom):
                reflection = reflect_issue_repair_plan(
                    project_root,
                    _make_task(),
                    review_state=_make_review_state(),
                    client=object(),  # bypass create_client
                )
            self.assertEqual(reflection["status"], "errored")
            self.assertEqual(reflection["accepted_fields"], {})
            self.assertTrue(any("llm_call_failed" in w for w in reflection["warnings"]))

            task = _make_task()
            apply_reflection_to_task(task, reflection)
            self.assertEqual(task["reflection_status"], "errored")
            self.assertEqual(
                set(task["reflection_metadata"]["fields_from_template"]),
                {"execution_steps", "success_criteria", "verifier", "close_condition"},
            )

    def test_malformed_json_falls_back_to_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            with patch(
                "ai_scientist.utils.repair_reflection.get_response_from_llm",
                return_value=("no json here", []),
            ):
                reflection = reflect_issue_repair_plan(
                    project_root,
                    _make_task(),
                    review_state=_make_review_state(),
                    client=object(),
                )
            self.assertEqual(reflection["status"], "empty")
            self.assertEqual(reflection["accepted_fields"], {})

    def test_partial_json_per_field_takeover(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            partial = (
                "```json\n"
                "{\n"
                "  \"execution_steps\": [\"Run new ablation A.\", \"Cross-check variance.\"],\n"
                "  \"verifier\": \"experiment_validation\",\n"
                "  \"confidence\": \"medium\",\n"
                "  \"rationale\": \"prior attempt left variance unaddressed\"\n"
                "}\n"
                "```"
            )
            with patch(
                "ai_scientist.utils.repair_reflection.get_response_from_llm",
                return_value=(partial, []),
            ):
                reflection = reflect_issue_repair_plan(
                    project_root,
                    _make_task(),
                    review_state=_make_review_state(),
                    client=object(),
                )
            self.assertEqual(reflection["status"], "ok")
            self.assertEqual(
                reflection["accepted_fields"]["verifier"], "experiment_validation"
            )
            self.assertEqual(
                reflection["accepted_fields"]["execution_steps"],
                ["Run new ablation A.", "Cross-check variance."],
            )

            task = _make_task()
            apply_reflection_to_task(task, reflection)
            self.assertEqual(task["execution_steps"], ["Run new ablation A.", "Cross-check variance."])
            self.assertEqual(task["verifier"], "experiment_validation")
            # success_criteria + close_condition came from template fallback.
            self.assertIn("success_criteria", task["reflection_metadata"]["fields_from_template"])
            self.assertIn("close_condition", task["reflection_metadata"]["fields_from_template"])
            self.assertEqual(task["reflection_metadata"]["confidence"], "medium")

    def test_invalid_verifier_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            bogus = (
                "```json\n"
                "{\n"
                "  \"execution_steps\": [\"x\"],\n"
                "  \"verifier\": \"nonsense_verifier\",\n"
                "  \"close_condition\": \"Close when fixed.\"\n"
                "}\n"
                "```"
            )
            with patch(
                "ai_scientist.utils.repair_reflection.get_response_from_llm",
                return_value=(bogus, []),
            ):
                reflection = reflect_issue_repair_plan(
                    project_root,
                    _make_task(),
                    review_state=_make_review_state(),
                    client=object(),
                )
            self.assertEqual(reflection["status"], "ok")
            self.assertNotIn("verifier", reflection["accepted_fields"])
            task = _make_task()
            apply_reflection_to_task(task, reflection)
            # verifier should have come from template fallback
            self.assertIn("verifier", task["reflection_metadata"]["fields_from_template"])


if __name__ == "__main__":
    unittest.main()
