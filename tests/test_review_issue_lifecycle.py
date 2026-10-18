from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    save_contract_artifact,
)
from ai_scientist.utils.repair_verifier import (
    verify_hostile_critic_recheck,
    verify_reviewer_board_recheck,
)
from ai_scientist.utils.review_jobs import begin_review_job, finalize_review_job


class ReviewIssueLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "project"
        self.root.mkdir()
        initialize_pipeline_contracts(self.root)

    def _state(self) -> dict:
        return load_contract_artifact(self.root, "review_state", default={})

    def _review(
        self, role: str, *issues: str, lane: str = "review", suite: str = ""
    ) -> dict:
        return self._finish(
            role,
            {"review": {"Weaknesses": list(issues)}},
            lane=lane,
            suite=suite,
        )

    def _finish(
        self,
        role: str,
        payload: object,
        *,
        lane: str = "review",
        suite: str = "",
        missing_pdf: bool = False,
    ) -> dict:
        job = begin_review_job(
            self.root,
            role=role,
            model_review="test-model",
            review_plan={},
            lane_name=lane,
            suite_name=suite,
        )
        finalize_review_job(
            self.root,
            job=job,
            review_text=payload,
            review_img=None,
            pdf_path=None if missing_pdf else str(self.root / "paper.pdf"),
        )
        return self._state()

    def test_clean_other_role_preserves_findings_and_repair_queue(self) -> None:
        self._review("rigor", "Missing baseline.")
        state = self._review("clarity")
        self.assertEqual(state["active_issues"], ["Missing baseline."])
        self.assertEqual(len(state["repair_queue"]), 1)
        self.assertEqual(state["repair_metrics"]["resolved_issue_count"], 0)
        issue_id = state["active_issue_records"][0]["issue_id"]
        passed, evidence = verify_reviewer_board_recheck(
            {"issue_id": issue_id}, self.root
        )
        self.assertFalse(passed)
        self.assertEqual(evidence["reason"], "issue_still_active")

    def test_native_unwrapped_reviewer_output_is_tracked_and_resolved(self) -> None:
        state = self._finish("rigor", {"Weaknesses": ["Missing baseline."]})
        issue = state["active_issue_records"][0]
        self._finish("rigor", {"Weaknesses": []})
        self.assertTrue(verify_reviewer_board_recheck(issue, self.root)[0])

    def test_same_role_in_other_lane_or_suite_cannot_clear_findings(self) -> None:
        self._review("rigor", "Missing baseline.", suite="standard")
        self._review("rigor", lane="hostile_critic", suite="standard")
        state = self._review("rigor", suite="extended")
        self.assertEqual(len(state["active_issue_records"]), 1)
        self.assertEqual(
            state["lane_summaries"]["hostile_critic"]["active_issue_count"], 0
        )
        critic = load_contract_artifact(self.root, "critic_findings", default={})
        self.assertEqual(critic["findings"], [])
        state = self._review("rigor", suite="standard")
        self.assertEqual(state["active_issue_records"], [])

    def test_shared_finding_requires_every_reporting_scope_to_clear(self) -> None:
        first = self._review("rigor", "Missing baseline.")
        issue_id = first["active_issue_records"][0]["issue_id"]
        state = self._review("reproducibility", "Missing baseline.")
        self.assertEqual(len(state["repair_queue"]), 1)
        self.assertEqual(len(state["active_issue_records"][0]["review_scopes"]), 2)
        state = self._review("rigor")
        self.assertEqual(state["active_issue_records"][0]["role"], "reproducibility")
        self.assertEqual(state["resolved_issue_records"], [])
        state = self._review("reproducibility")
        self.assertEqual(state["active_issue_records"], [])
        passed, evidence = verify_reviewer_board_recheck(
            {"issue_id": issue_id}, self.root
        )
        self.assertTrue(passed)
        self.assertEqual(len(evidence["resolution_job_ids"]), 2)

    def test_reopened_finding_retains_history_but_is_not_resolved(self) -> None:
        first = self._review("rigor", "Missing baseline.")["active_issue_records"][0]
        self._review("rigor")
        state = self._review("rigor", "Missing baseline.")
        record = state["active_issue_records"][0]
        self.assertEqual(record["first_seen_job_id"], first["first_seen_job_id"])
        self.assertEqual(record["appearance_count"], 2)
        self.assertEqual(state["resolved_issue_records"], [])
        self.assertEqual(state["repair_metrics"]["resolved_issue_count"], 0)
        self.assertFalse(verify_reviewer_board_recheck(record, self.root)[0])

    def test_partial_resolution_uses_remaining_scope_severity(self) -> None:
        self._review("rigor", "Unclear baseline.")
        self._finish(
            "clarity",
            {"review": {"Weaknesses": [], "Questions": ["Unclear baseline."]}},
        )
        state = self._review("rigor")
        self.assertEqual(state["active_issue_records"][0]["severity"], "minor")
        self.assertEqual(state["lane_summaries"]["review"]["blocking_issue_count"], 0)

    def test_invalid_or_incomplete_review_never_clears_findings(self) -> None:
        self._review("rigor", "Missing baseline.")
        for payload in (
            None,
            {},
            {"review": {}},
            {"review": {"Weaknesses": None}},
            {"review": {"Weaknesses": [], "Questions": [None]}},
        ):
            with self.subTest(payload=payload):
                state = self._finish("rigor", payload)
                self.assertEqual(state["active_issues"], ["Missing baseline."])
                self.assertFalse(state["rounds"][-1]["review_payload_valid"])
        state = self._finish("rigor", {"review": {"Weaknesses": []}}, missing_pdf=True)
        self.assertEqual(state["active_issues"], ["Missing baseline."])

    def test_lane_counts_exclude_other_lanes_and_keep_shared_issues(self) -> None:
        self._review("rigor", "Ordinary issue.", "Shared issue.")
        state = self._review(
            "claim_cross_examiner",
            "Hostile issue.",
            "Shared issue.",
            lane="hostile_critic",
        )
        self.assertEqual(len(state["active_issue_records"]), 3)
        self.assertEqual(
            state["lane_summaries"]["hostile_critic"]["active_issue_count"], 2
        )
        critic = load_contract_artifact(self.root, "critic_findings", default={})
        self.assertEqual(len(critic["findings"]), 2)
        self.assertNotIn(
            "Ordinary issue.", [item["why_blocking"] for item in critic["findings"]]
        )
        state = self._review("clarity")
        self.assertEqual(state["lane_summaries"]["review"]["active_issue_count"], 2)
        self.assertEqual(
            state["lane_summaries"]["review"]["roles"], ["clarity", "rigor"]
        )

    def test_omitted_issue_field_is_not_an_explicit_clean_result(self) -> None:
        self._finish(
            "rigor",
            {
                "review": {
                    "Weaknesses": ["Missing baseline."],
                    "Limitations": ["Missing ablation."],
                }
            },
        )
        state = self._review("rigor")
        self.assertEqual(state["active_issues"], ["Missing ablation."])
        self.assertEqual(state["resolved_issues"], ["Missing baseline."])
        state = self._finish("rigor", {"review": {"Weaknesses": [], "Limitations": []}})
        self.assertEqual(state["active_issue_records"], [])

    def test_hostile_resolution_can_precede_last_ordinary_resolution(self) -> None:
        self._review("rigor", "Shared issue.")
        state = self._review(
            "claim_cross_examiner", "Shared issue.", lane="hostile_critic"
        )
        issue = state["active_issue_records"][0]
        self.assertEqual(state["repair_queue"][0]["review_lane"], "hostile_critic")
        self.assertFalse(verify_hostile_critic_recheck(issue, self.root)[0])
        self._review("claim_cross_examiner", lane="hostile_critic")
        self.assertFalse(verify_hostile_critic_recheck(issue, self.root)[0])
        self._review("rigor")
        self.assertTrue(verify_hostile_critic_recheck(issue, self.root)[0])

    def test_ordinary_resolution_cannot_substitute_for_hostile_recheck(self) -> None:
        issue = self._review("rigor", "Missing baseline.")["active_issue_records"][0]
        self._review("rigor")
        passed, evidence = verify_hostile_critic_recheck(issue, self.root)
        self.assertFalse(passed)
        self.assertEqual(evidence["reason"], "missing_hostile_recheck")

    def test_re_report_invalidates_prior_scope_resolution(self) -> None:
        issue = self._review("rigor", "Shared issue.")["active_issue_records"][0]
        self._review("clarity", "Shared issue.")
        self._review("rigor")
        state = self._review("rigor", "Shared issue.")
        self.assertEqual(state["active_issue_records"][0]["resolution_checks"], [])
        self._review("clarity")
        self.assertFalse(verify_reviewer_board_recheck(issue, self.root)[0])

    def test_legacy_owned_issue_migrates_to_scoped_resolution(self) -> None:
        state = self._review("rigor", "Missing baseline.")
        issue = state["active_issue_records"][0]
        issue.pop("review_scopes")
        save_contract_artifact(self.root, "review_state", state, producer="test")
        self._review("rigor")
        self.assertTrue(verify_reviewer_board_recheck(issue, self.root)[0])

    def test_missing_or_unknown_issue_is_not_verification(self) -> None:
        self.assertFalse(verify_reviewer_board_recheck({}, self.root)[0])
        self.assertFalse(
            verify_reviewer_board_recheck({"issue_id": "RVW-unknown"}, self.root)[0]
        )
        self._review("rigor")
        self.assertFalse(
            verify_reviewer_board_recheck({"issue_id": "RVW-unknown"}, self.root)[0]
        )

    def test_resolution_requires_matching_completed_valid_job(self) -> None:
        issue = self._review("rigor", "Missing baseline.")["active_issue_records"][0]
        state = self._review("rigor")
        for changes in (
            {"status": "missing_pdf"},
            {"review_payload_valid": False},
            {"role": "clarity"},
            {"suite_name": "other"},
        ):
            with self.subTest(changes=changes):
                modified = {
                    **state,
                    "rounds": state["rounds"][:-1]
                    + [{**state["rounds"][-1], **changes}],
                }
                save_contract_artifact(
                    self.root, "review_state", modified, producer="test"
                )
                self.assertFalse(verify_reviewer_board_recheck(issue, self.root)[0])


if __name__ == "__main__":
    unittest.main()
