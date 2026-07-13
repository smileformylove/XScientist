from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    load_pipeline_manifest,
)
from ai_scientist.utils.research_planning import build_idea_cards, build_research_plan
from ai_scientist.utils.truth_contracts import (
    CHECK_SEVERITIES,
    TRUTH_CONTRACT_CATEGORIES,
    build_hallucination_review,
    save_truth_contract_bundle,
    validate_hallucination_checks,
    validate_truth_contract,
)


class TruthContractTests(unittest.TestCase):
    def test_bundle_persists_contract_checks_and_review_shell(self) -> None:
        ideas = [
            {
                "Name": "Truthful Plan",
                "Short Hypothesis": "Explicit contracts reduce unsupported claims.",
                "Experiments": [
                    "Compare against baseline: no-contract on dataset: demo with metric: accuracy."
                ],
            }
        ]
        idea_card = build_idea_cards(
            ideas,
            target_venue="neurips",
            workflow_mode="multi_agent_board",
        )[0]
        research_plan = build_research_plan(idea_card, target_venue="neurips")

        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="multi_agent_board")
            bundle = save_truth_contract_bundle(
                project_root,
                idea_card=idea_card,
                research_plan=research_plan,
                producer="test_truth_contracts",
            )

            self.assertTrue(bundle["truth_contract_validation"]["passed"])
            self.assertTrue(bundle["hallucination_checks_validation"]["passed"])
            contract = load_contract_artifact(project_root, "truth_contract", default={})
            checks = load_contract_artifact(
                project_root,
                "hallucination_checks",
                default={},
            )
            review = load_contract_artifact(
                project_root,
                "hallucination_review",
                default={},
            )
            manifest = load_pipeline_manifest(project_root)

            self.assertEqual(
                set(contract["categories"].keys()),
                set(TRUTH_CONTRACT_CATEGORIES),
            )
            self.assertGreater(checks["summary"]["blocker_count"], 0)
            self.assertEqual(review["checks_total"], len(checks["checks"]))
            self.assertGreater(review["summary"]["pending_count"], 0)
            self.assertEqual(manifest["artifacts"]["truth_contract"]["status"], "ready")
            self.assertEqual(
                manifest["artifacts"]["hallucination_checks"]["status"],
                "ready",
            )

    def test_validation_rejects_missing_sources_and_bad_check_enums(self) -> None:
        bad_contract = {
            "categories": {
                category: [{"requirement": "x", "source": "s"}]
                for category in TRUTH_CONTRACT_CATEGORIES
            }
        }
        bad_contract["categories"]["objective_facts"][0]["source"] = ""
        self.assertFalse(validate_truth_contract(bad_contract)["passed"])

        bad_checks = {
            "checks": [
                {
                    "category": "objective_facts",
                    "requirement": "x",
                    "prohibited_failure": "y",
                    "severity": "critical",
                    "evidence_source": "source",
                }
            ]
        }
        result = validate_hallucination_checks(bad_checks)
        self.assertFalse(result["passed"])
        self.assertIn("critical", result["errors"][0])

    def test_validation_rejects_missing_high_risk_task_category(self) -> None:
        research_plan = {
            "workflow_mode": "multi_agent_board",
            "tasks": [
                {
                    "task_id": "task_0",
                    "priority": "P0",
                    "escalation_lane": "hostile_critic",
                }
            ],
        }
        checks = {
            "checks": [
                {
                    "check_id": "task_0:objective_facts:0",
                    "task_id": "task_0",
                    "category": "objective_facts",
                    "requirement": "Use the declared dataset.",
                    "prohibited_failure": "Invents another dataset.",
                    "severity": "blocker",
                    "evidence_source": "research_plan.task.dataset",
                }
            ]
        }
        result = validate_hallucination_checks(checks, research_plan=research_plan)
        self.assertFalse(result["passed"])
        self.assertIn(
            "missing_task_category:task_0:physical_constraints",
            result["errors"],
        )

    def test_hallucination_review_requires_all_decisions_before_pass(self) -> None:
        checks = {
            "checks": [
                {
                    "check_id": "task_0:objective_facts:0",
                    "task_id": "task_0",
                    "category": "objective_facts",
                    "requirement": "Use the declared dataset.",
                    "prohibited_failure": "Invents another dataset.",
                    "severity": CHECK_SEVERITIES[0],
                    "evidence_source": "research_plan.task.dataset",
                }
            ]
        }
        pending = build_hallucination_review(checks)
        self.assertFalse(pending["passed"])
        self.assertEqual(pending["summary"]["pending_count"], 1)

        passed = build_hallucination_review(
            checks,
            decisions=[
                {
                    "check_id": "task_0:objective_facts:0",
                    "passed": True,
                    "finding": "Dataset matches the plan.",
                    "evidence": "Matched artifact manifest dataset field.",
                }
            ],
            keyframe_paths={
                "start": "frames/start.png",
                "mid": "frames/mid.png",
                "end": "frames/end.png",
            },
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["results"][0]["start_keyframe_path"], "frames/start.png")

    def test_hallucination_review_rejects_empty_or_evidence_free_decisions(self) -> None:
        empty = build_hallucination_review({"checks": []})
        self.assertFalse(empty["passed"])
        self.assertEqual(empty["checks_total"], 0)

        checks = {
            "checks": [
                {
                    "check_id": "task_0:values_guardrails:0",
                    "task_id": "task_0",
                    "category": "values_guardrails",
                    "requirement": "Do not overstate unsupported claims.",
                    "prohibited_failure": "Overstates a claim.",
                    "severity": "blocker",
                    "evidence_source": "research_plan.execution_policy.acceptance_rules",
                }
            ]
        }
        evidence_free = build_hallucination_review(
            checks,
            decisions=[
                {
                    "check_id": "task_0:values_guardrails:0",
                    "passed": True,
                    "finding": "Looks fine.",
                }
            ],
        )
        self.assertFalse(evidence_free["passed"])
        self.assertEqual(evidence_free["summary"]["evidence_free_decision_count"], 1)


if __name__ == "__main__":
    unittest.main()
