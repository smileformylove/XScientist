from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol.schemas import load_schema
from xscientist import ResearchRepository
from xscientist.opportunity_funnel import (
    FAR_POOL_PROTOCOL,
    build_opportunity_funnel_summary,
    build_opportunity_attempt,
    build_opportunity_pool,
    build_research_direction,
    inspect_opportunity_funnel,
    rank_opportunity_candidates,
    save_opportunity_allocation,
    save_opportunity_attempt,
    save_opportunity_grade,
    save_opportunity_judgment,
    save_opportunity_pool,
    save_research_direction,
)
from xscientist.research_git import ResearchGitError


class OpportunityFunnelPureTests(unittest.TestCase):
    def _candidates(self) -> list[dict[str, object]]:
        return [
            {
                "candidate_id": "q-2",
                "question": "Can the rival mechanism explain the anomaly?",
                "source_refs": ["source-2"],
                "source_status": "open",
                "expected_success_probability": 0.4,
                "expected_importance": 0.9,
            },
            {
                "candidate_id": "q-1",
                "question": "Does the proposed mechanism survive a held-out test?",
                "source_refs": ["source-1"],
                "source_status": "open",
                "expected_success_probability": 0.8,
                "expected_importance": 0.5,
            },
            {
                "candidate_id": "unscored",
                "question": "What boundary condition remains unknown?",
                "source_refs": [],
                "source_status": "unknown",
            },
        ]

    def test_ranking_is_deterministic_and_does_not_impute_missing_probability(self):
        first = rank_opportunity_candidates(
            self._candidates(), objective="importance_yield", max_attempts=1
        )
        second = rank_opportunity_candidates(
            list(reversed(self._candidates())),
            objective="importance_yield",
            max_attempts=1,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_set"][0]["candidate_id"], "q-1")
        self.assertTrue(first["candidate_set"][0]["selected"])
        unscored = next(
            row for row in first["candidate_set"] if row["candidate_id"] == "unscored"
        )
        self.assertIsNone(unscored["allocation_score"])
        self.assertFalse(unscored["selected"])
        self.assertEqual(first["calibration_status"], "declared_inputs_not_calibrated")

    def test_pool_hash_and_summary_keep_unattempted_candidates_visible(self):
        pool = build_opportunity_pool(
            direction_id="direction-1",
            candidates=self._candidates(),
            complete_candidate_set=False,
        )
        summary = build_opportunity_funnel_summary(pool=pool)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(
            summary["unattempted_candidate_ids"], ["q-1", "q-2", "unscored"]
        )
        self.assertFalse(summary["funnel_complete"])
        self.assertFalse(summary["quality_claim_allowed"])

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ResearchGitError):
            rank_opportunity_candidates(self._candidates(), objective="made_up")
        with self.assertRaises(ResearchGitError):
            build_opportunity_pool(
                direction_id="d",
                candidates=[
                    {"candidate_id": "same", "question": "one"},
                    {"candidate_id": "same", "question": "two"},
                ],
            )
        with self.assertRaises(ResearchGitError):
            build_opportunity_attempt(
                pool_id="rso-aaaaaaaaaaaaaaaa",
                candidate_id="q",
                outcome="none",
                summary="Q",
                attempted_at="2026-08-23T00:00:00",
            )
        with self.assertRaises(ResearchGitError):
            build_opportunity_pool(
                direction_id="d",
                candidates=[
                    {
                        "candidate_id": "q",
                        "question": "Q",
                        "source_status": "open",
                        "source_object_ids": ["rso-not-canonical"],
                    }
                ],
            )

    def test_probability_semantics_are_explicit_and_do_not_double_count_joint_p(self):
        joint = rank_opportunity_candidates(
            [
                {
                    "candidate_id": "joint",
                    "question": "Q",
                    "source_status": "open",
                    "expected_success_probability": 0.8,
                    "expected_artifact_probability": 0.25,
                    "expected_importance": 0.5,
                }
            ],
            objective="importance_yield",
            probability_semantics="joint_artifact_probability",
        )
        row = joint["candidate_set"][0]
        self.assertEqual(joint["probability_semantics"], "joint_artifact_probability")
        self.assertEqual(row["probability_formula"], "expected_artifact_probability")
        self.assertEqual(row["allocation_score"], 0.125)
        self.assertFalse(row["artifact_probability_assumed"])
        missing = rank_opportunity_candidates(
            [
                {
                    "candidate_id": "missing-joint",
                    "question": "Q",
                    "source_status": "open",
                    "expected_success_probability": 0.8,
                }
            ],
            probability_semantics="joint_artifact_probability",
        )
        self.assertFalse(missing["candidate_set"][0]["allocation_eligible"])
        self.assertEqual(
            missing["candidate_set"][0]["allocation_reason"],
            "missing_explicit_joint_artifact_probability",
        )

    def test_summary_reports_lineage_and_orphan_ids(self):
        pool = build_opportunity_pool(
            direction_id="direction-1",
            candidates=[
                {
                    "candidate_id": "q-1",
                    "question": "Q",
                    "source_refs": ["external-paper"],
                    "source_status": "open",
                }
            ],
            complete_candidate_set=True,
        )
        summary = build_opportunity_funnel_summary(
            pool={**pool, "object_id": "rso-pool"},
            attempts=[
                {
                    "object_id": "rso-attempt",
                    "pool_id": "rso-pool",
                    "candidate_id": "q-1",
                    "outcome": "new",
                }
            ],
            judgments=[
                {
                    "object_id": "rso-orphan-judgment",
                    "attempt_id": "rso-not-an-attempt",
                    "verdict": "pass",
                }
            ],
            grades=[
                {
                    "object_id": "rso-orphan-grade",
                    "judgment_id": "rso-not-a-judgment",
                }
            ],
        )
        self.assertFalse(summary["lineage_complete"])
        self.assertEqual(summary["candidate_lineage_unbound_ids"], ["q-1"])
        self.assertEqual(summary["orphan_judgment_attempt_ids"], ["rso-not-an-attempt"])
        self.assertEqual(summary["orphan_grade_judgment_ids"], ["rso-not-a-judgment"])
        self.assertFalse(summary["funnel_complete"])
        with self.assertRaises(ResearchGitError):
            build_opportunity_pool(
                direction_id="d",
                candidates=[
                    {
                        "candidate_id": "bad",
                        "question": "q",
                        "expected_success_probability": float("nan"),
                    }
                ],
            )

    def test_public_literature_opportunity_schema_accepts_builder_payloads(self):
        schema = load_schema("literature_opportunity")
        payloads = [
            build_research_direction(direction_id="d", statement="Q", objective="O"),
            build_opportunity_pool(
                direction_id="d",
                candidates=[
                    {"candidate_id": "q", "question": "Q", "source_status": "open"}
                ],
            ),
            # The allocation builder is intentionally independent of a
            # repository; persistence adds pool/budget binding fields later.
            rank_opportunity_candidates(
                [
                    {
                        "candidate_id": "q",
                        "question": "Q",
                        "source_status": "open",
                        "expected_success_probability": 0.5,
                    }
                ]
            ),
        ]
        for payload in payloads:
            validate(payload, schema)

    def test_bounded_metadata_rejects_recursive_policy_and_secret_refs(self):
        policy: dict[str, object] = {}
        policy["self"] = policy
        with self.assertRaises(ResearchGitError):
            build_research_direction(
                direction_id="d",
                statement="Q",
                objective="O",
                candidate_policy=policy,
            )
        with self.assertRaises(ResearchGitError):
            build_opportunity_pool(
                direction_id="d",
                candidates=[
                    {
                        "candidate_id": "q",
                        "question": "Q",
                        "source_status": "open",
                        "source_refs": [{"api_key": "do-not-persist", "id": "p"}],
                    }
                ],
            )


@unittest.skipUnless(shutil.which("git"), "Git is required for Research VCS tests")
class OpportunityFunnelRepositoryTests(unittest.TestCase):
    def _repo(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            name="opportunity-funnel",
            question="Find and test open research questions.",
            git_user_name="Funnel Test",
            git_user_email="funnel@example.invalid",
        )

    def test_find_attempt_judge_grade_and_allocation_are_typed_and_auditable(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repo(Path(td) / "research")
            direction = save_research_direction(
                repository.path,
                direction_id="direction-1",
                statement="Which mechanism explains the observed anomaly?",
                objective="Produce a falsifiable, reproducible result.",
                commit=False,
            )
            pool = save_opportunity_pool(
                repository.path,
                direction_id=direction["object"].object_id,
                candidates=[
                    {
                        "candidate_id": "candidate-1",
                        "question": "Does the mechanism survive a held-out test?",
                        "source_refs": ["paper-1"],
                        "source_status": "open",
                        "expected_success_probability": 0.75,
                        "expected_importance": 0.8,
                    },
                    {
                        "candidate_id": "candidate-2",
                        "question": "Is the effect a data artifact?",
                        "source_refs": ["paper-2"],
                        "source_status": "open",
                    },
                ],
                commit=False,
            )
            attempt = save_opportunity_attempt(
                repository.path,
                pool_id=pool["object"].object_id,
                candidate_id="candidate-1",
                outcome="new",
                summary="The held-out test supports a new mechanism.",
                commit=False,
            )
            judgment = save_opportunity_judgment(
                repository.path,
                attempt_id=attempt["object"].object_id,
                verdict="pass",
                evaluator_id="independent-judge",
                summary="The claimed result is not known from the bound sources.",
                commit=False,
            )
            grade = save_opportunity_grade(
                repository.path,
                judgment_id=judgment["object"].object_id,
                grade="substantial",
                evaluator_id="independent-grader",
                summary="The result clears the declared importance threshold.",
                commit=False,
            )
            allocation = save_opportunity_allocation(
                repository.path,
                pool_id=pool["object"].object_id,
                objective="importance_yield",
                max_attempts=1,
                commit=False,
            )
            inspected = inspect_opportunity_funnel(repository, pool["object"].object_id)
            summary = inspected["summary"]
            self.assertEqual(
                inspected["pool"]["payload"]["protocol_kind"], FAR_POOL_PROTOCOL
            )
            self.assertEqual(summary["attempt_count"], 1)
            self.assertEqual(summary["judgment_count"], 1)
            self.assertEqual(summary["grade_count"], 1)
            self.assertIn("candidate-2", summary["unattempted_candidate_ids"])
            self.assertFalse(summary["quality_claim_allowed"])
            self.assertEqual(allocation["object"].kind, "resource_budget")
            self.assertEqual(grade["object"].kind, "review")

    def test_attempt_cannot_escape_locked_pool(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repo(Path(td) / "research")
            direction = save_research_direction(
                repository.path,
                direction_id="d",
                statement="Question",
                objective="Objective",
                commit=False,
            )
            pool = save_opportunity_pool(
                repository.path,
                direction_id=direction["object"].object_id,
                candidates=[{"candidate_id": "inside", "question": "Q"}],
                commit=False,
            )
            with self.assertRaises(ResearchGitError):
                save_opportunity_attempt(
                    repository.path,
                    pool_id=pool["object"].object_id,
                    candidate_id="outside",
                    outcome="none",
                    summary="not allowed",
                    commit=False,
                )

    def test_stage_gates_require_new_then_pass_or_known(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repo(Path(td) / "research")
            direction = save_research_direction(
                repository.path,
                direction_id="d-gates",
                statement="Question",
                objective="Objective",
                commit=False,
            )
            pool = save_opportunity_pool(
                repository.path,
                direction_id=direction["object"].object_id,
                candidates=[
                    {
                        "candidate_id": "known-candidate",
                        "question": "Q",
                        "source_status": "open",
                    }
                ],
                commit=False,
            )
            attempt = save_opportunity_attempt(
                repository.path,
                pool_id=pool["object"].object_id,
                candidate_id="known-candidate",
                outcome="none",
                summary="No reliable result.",
                commit=False,
            )
            with self.assertRaises(ResearchGitError):
                save_opportunity_judgment(
                    repository.path,
                    attempt_id=attempt["object"].object_id,
                    verdict="fail",
                    evaluator_id="judge",
                    summary="Retrospective review is not a normal FAR transition.",
                    commit=False,
                )
            judgment = save_opportunity_judgment(
                repository.path,
                attempt_id=attempt["object"].object_id,
                verdict="fail",
                evaluator_id="judge",
                summary="Retrospective review is explicitly recorded.",
                allow_stage_override=True,
                override_reason="The source status was reclassified after the attempt.",
                commit=False,
            )
            with self.assertRaises(ResearchGitError):
                save_opportunity_grade(
                    repository.path,
                    judgment_id=judgment["object"].object_id,
                    grade="minor",
                    evaluator_id="grader",
                    summary="Cannot grade a failed judgment without an override.",
                    commit=False,
                )

    def test_allocation_fails_closed_for_incomplete_or_non_open_pool(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repo(Path(td) / "research")
            direction = save_research_direction(
                repository.path,
                direction_id="d-allocation-gates",
                statement="Question",
                objective="Objective",
                commit=False,
            )
            incomplete = save_opportunity_pool(
                repository.path,
                direction_id=direction["object"].object_id,
                candidates=[
                    {"candidate_id": "q", "question": "Q", "source_status": "open"}
                ],
                complete_candidate_set=False,
                commit=False,
            )
            with self.assertRaises(ResearchGitError):
                save_opportunity_allocation(
                    repository.path,
                    pool_id=incomplete["object"].object_id,
                    commit=False,
                )
            non_open = save_opportunity_pool(
                repository.path,
                direction_id=direction["object"].object_id,
                candidates=[
                    {"candidate_id": "q", "question": "Q", "source_status": "unknown"}
                ],
                commit=False,
            )
            with self.assertRaises(ResearchGitError):
                save_opportunity_allocation(
                    repository.path,
                    pool_id=non_open["object"].object_id,
                    commit=False,
                )
