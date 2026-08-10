from __future__ import annotations

import unittest

from ai_scientist.utils.semantic_memory import (
    bounded_semantic_view,
    estimate_text_tokens,
    semantic_overlap,
    truncate_text_to_tokens,
)


class SemanticMemoryTests(unittest.TestCase):
    def test_overlap_is_context_sensitive_for_english_and_chinese(self) -> None:
        query = "当前分支的域偏移 calibration failure"
        relevant = "域偏移导致 calibration 在当前复现实验失败"
        irrelevant = "image segmentation color palette"

        self.assertGreater(
            semantic_overlap(query, relevant),
            semantic_overlap(query, irrelevant),
        )

    def test_projection_is_bounded_and_prioritizes_scientific_risk(self) -> None:
        context = {
            "historical_notes": [f"generic note {index}" for index in range(50)],
            "current": {
                "evidence": "domain shift leakage invalidates the metric",
                "risk": "do not reuse the leaked validation split",
            },
        }
        view = bounded_semantic_view(
            context,
            query="domain shift leakage metric risk",
            budget_tokens=120,
        )

        self.assertLessEqual(estimate_text_tokens(view), 120)
        self.assertIn("current", view[0]["path"])
        self.assertIn("leak", str(view[:3]).lower())

    def test_truncation_respects_conservative_token_budget(self) -> None:
        value = "长期上下文" * 100 + " decisive evidence"
        truncated = truncate_text_to_tokens(value, 24)

        self.assertLessEqual(estimate_text_tokens(truncated), 24)
        self.assertTrue(truncated.endswith("..."))


if __name__ == "__main__":
    unittest.main()
