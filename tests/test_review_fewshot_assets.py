from __future__ import annotations

import hashlib
import unittest

from ai_scientist.perform_llm_review import get_review_fewshot_examples

EXPECTED_PROMPT_HASHES = {
    1: (64285, "8ef3e83911a19217832789ab120988ea3a84aa7ea07325d9f99dbc5ff9a86537"),
    2: (100491, "86611fbac8342ba64e3e99d6c655e1db8600c788162312e3fbd40d57a806c65c"),
    3: (143953, "d14bc5f9599da84480f01e45697e201bef28c8560aac5632e93dec48cc52c3fc"),
}


class ReviewFewshotAssetTests(unittest.TestCase):
    def test_text_assets_preserve_exact_review_prompts(self) -> None:
        for count, (expected_size, expected_hash) in EXPECTED_PROMPT_HASHES.items():
            with self.subTest(count=count):
                prompt = get_review_fewshot_examples(count).encode("utf-8")
                self.assertEqual(len(prompt), expected_size)
                self.assertEqual(hashlib.sha256(prompt).hexdigest(), expected_hash)


if __name__ == "__main__":
    unittest.main()
