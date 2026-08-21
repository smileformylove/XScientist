from __future__ import annotations

import unittest
from unittest import mock

from ai_scientist.professional_writing_system import ProfessionalPaperEvaluator


class ProfessionalWritingQualityTests(unittest.TestCase):
    def _evaluator(self) -> ProfessionalPaperEvaluator:
        evaluator = ProfessionalPaperEvaluator.__new__(ProfessionalPaperEvaluator)
        evaluator.template = {"name": "NeurIPS"}
        evaluator.client = object()
        evaluator.client_model = "test-model"
        return evaluator

    def test_bare_json_reviewer_response_is_parsed(self) -> None:
        evaluator = self._evaluator()
        with mock.patch(
            "ai_scientist.professional_writing_system.get_response_from_llm",
            return_value=('{"score": 4, "analysis": "clear"}', None),
        ):
            result = evaluator._evaluate_dimension("clarity", "paper", {})

        self.assertEqual(result["score"], 4.0)
        self.assertNotIn("evaluation_error", result)

    def test_invalid_reviewer_score_fails_closed(self) -> None:
        evaluator = self._evaluator()
        with mock.patch(
            "ai_scientist.professional_writing_system.get_response_from_llm",
            return_value=('{"analysis": "missing score"}', None),
        ):
            result = evaluator._evaluate_dimension("rigor", "paper", {})

        self.assertEqual(result["score"], 0)
        self.assertEqual(
            result["evaluation_error"], "reviewer_score_missing_or_invalid"
        )

    def test_nonfinite_reviewer_score_fails_closed(self) -> None:
        evaluator = self._evaluator()
        with mock.patch(
            "ai_scientist.professional_writing_system.get_response_from_llm",
            return_value=('{"score": NaN, "analysis": "invalid"}', None),
        ):
            result = evaluator._evaluate_dimension("rigor", "paper", {})

        self.assertEqual(result["score"], 0)
        self.assertEqual(
            result["evaluation_error"], "reviewer_score_missing_or_invalid"
        )


if __name__ == "__main__":
    unittest.main()
