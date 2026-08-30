from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist import perform_llm_review
from ai_scientist.perform_llm_review import get_review_fewshot_examples

EXPECTED_STEMS = {
    "synthetic_strong",
    "synthetic_borderline",
    "synthetic_negative",
}
EXPECTED_REVIEW_FIELDS = {
    "Summary",
    "Strengths",
    "Weaknesses",
    "Originality",
    "Quality",
    "Clarity",
    "Significance",
    "Questions",
    "Limitations",
    "Ethical Concerns",
    "Soundness",
    "Presentation",
    "Contribution",
    "Overall",
    "Confidence",
    "Decision",
}


class ReviewFewshotAssetTests(unittest.TestCase):
    def test_pdf_layout_dependency_is_optional_at_import_time(self) -> None:
        missing = ModuleNotFoundError("No module named 'pymupdf4llm'")
        missing.name = "pymupdf4llm"
        with mock.patch.object(
            perform_llm_review.importlib,
            "import_module",
            side_effect=missing,
        ):
            self.assertIsNone(perform_llm_review._load_pdf_layout_module())

    def test_synthetic_assets_are_small_complete_and_explicitly_fictional(self) -> None:
        asset_dir = Path(perform_llm_review.dir_path) / "fewshot_examples"
        self.assertEqual(
            {path.stem for path in asset_dir.glob("*")},
            EXPECTED_STEMS,
        )

        overall_scores = []
        for stem in sorted(EXPECTED_STEMS):
            with self.subTest(stem=stem):
                paper_path = asset_dir / f"{stem}.txt"
                review_path = asset_dir / f"{stem}.json"
                paper = paper_path.read_text(encoding="utf-8")
                payload = json.loads(review_path.read_text(encoding="utf-8"))
                review = json.loads(payload["review"])

                self.assertLess(paper_path.stat().st_size, 4096)
                self.assertLess(review_path.stat().st_size, 4096)
                self.assertIn("SYNTHETIC CALIBRATION PAPER", paper)
                self.assertIn("fictional", payload["xscientist_asset_notice"])
                self.assertEqual(set(review), EXPECTED_REVIEW_FIELDS)
                overall_scores.append(review["Overall"])

        self.assertEqual(sorted(overall_scores), [2, 5, 8])

    def test_prompt_loader_uses_only_synthetic_xscientist_examples(self) -> None:
        previous_size = 0
        for count in range(1, 4):
            with self.subTest(count=count):
                prompt = get_review_fewshot_examples(count)
                self.assertIn("XScientist-authored synthetic papers", prompt)
                self.assertEqual(prompt.count("SYNTHETIC CALIBRATION PAPER"), count)
                self.assertGreater(len(prompt), previous_size)
                self.assertNotIn("Attention Is All You Need", prompt)
                previous_size = len(prompt)


if __name__ == "__main__":
    unittest.main()
