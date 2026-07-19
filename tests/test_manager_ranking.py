from __future__ import annotations

import unittest

from ai_scientist.apps.manager import ResearchManager
from ai_scientist.apps.manager_ranking import (
    _passes_submission_filters,
    _rewrite_board_sort_key,
    _submission_priority_sort_key,
    _suggest_rewrite_next_step,
)


class ManagerRankingTests(unittest.TestCase):
    def test_manager_static_methods_preserve_rule_exports(self) -> None:
        self.assertIs(
            ResearchManager._submission_priority_sort_key,
            _submission_priority_sort_key,
        )
        self.assertIs(
            ResearchManager._passes_submission_filters, _passes_submission_filters
        )
        self.assertIs(ResearchManager._rewrite_board_sort_key, _rewrite_board_sort_key)
        self.assertIs(
            ResearchManager._suggest_rewrite_next_step,
            _suggest_rewrite_next_step,
        )

    def test_unbound_reviewer_issue_returns_target_binding_guidance(self) -> None:
        message = _suggest_rewrite_next_step(
            {
                "review_active_issue_count": 2,
                "review_repair_ready_coverage": 1.0,
                "review_unbound_issue_count": 1,
            }
        )

        self.assertIn("not yet mapped", message)
        self.assertIn("bind them first", message)

    def test_submission_filters_enforce_operational_gates(self) -> None:
        paper = {
            "target_venue": "iclr",
            "quality_gate_passed": True,
            "submission_status": "ready",
            "strict_fallback_count": 0,
            "blocked_stage_count": 0,
            "self_evolution_status": "healthy",
            "self_evolution_required_failure_count": 0,
            "process_alignment_blocked_process_count": 0,
        }

        self.assertTrue(
            _passes_submission_filters(
                paper,
                target_venue="iclr",
                require_gate=True,
                require_ready=True,
                max_strict_fallbacks=0,
                max_blocked_stages=0,
                max_self_evolution_required_failures=0,
                max_blocked_processes=0,
            )
        )
        self.assertFalse(
            _passes_submission_filters(
                paper | {"process_alignment_blocked_process_count": 1},
                max_blocked_processes=0,
            )
        )

    def test_rewrite_sort_key_prioritizes_open_gate_debt(self) -> None:
        open_debt = {
            "self_review_round_gate_ready": False,
            "self_review_round_gate_score": 40,
            "review_active_issue_count": 2,
        }
        ready = {
            "self_review_round_gate_ready": True,
            "self_review_round_gate_score": 100,
        }

        self.assertGreater(
            _rewrite_board_sort_key(open_debt), _rewrite_board_sort_key(ready)
        )


if __name__ == "__main__":
    unittest.main()
