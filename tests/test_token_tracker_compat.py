from __future__ import annotations

import unittest

from ai_scientist.utils.token_tracker import _extract_usage


class _UsageDetails:
    def __init__(self, reasoning_tokens: int = 0, cached_tokens: int = 0) -> None:
        self.reasoning_tokens = reasoning_tokens
        self.cached_tokens = cached_tokens


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 120
        self.completion_tokens = 45
        self.completion_tokens_details = _UsageDetails(reasoning_tokens=17)
        self.prompt_tokens_details = _UsageDetails(cached_tokens=32)


class _Result:
    def __init__(self) -> None:
        self.usage = _Usage()


class TokenTrackerCompatTests(unittest.TestCase):
    def test_extract_usage_should_return_expected_token_tuple(self) -> None:
        result = _Result()
        usage = _extract_usage(result)
        self.assertEqual(usage, (120, 45, 17, 32))


if __name__ == "__main__":
    unittest.main()
