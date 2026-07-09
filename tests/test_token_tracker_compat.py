from __future__ import annotations

import unittest
import warnings
from datetime import datetime

from ai_scientist.utils.token_tracker import (
    _extract_usage,
    _record_token_usage,
    token_tracker,
)


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


class _ResultWithoutCreated:
    """Mimics an LLM response missing the `created` field so the timestamp
    fallback path in `_record_token_usage` is exercised."""

    def __init__(self) -> None:
        self.usage = _Usage()
        self.model = "test-model"


class TokenTrackerCompatTests(unittest.TestCase):
    def test_extract_usage_should_return_expected_token_tuple(self) -> None:
        result = _Result()
        usage = _extract_usage(result)
        self.assertEqual(usage, (120, 45, 17, 32))

    def test_record_token_usage_fallback_timestamp_is_deprecation_free(self) -> None:
        token_tracker.reset()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                _record_token_usage(
                    _ResultWithoutCreated(),
                    prompt="p",
                    system_message="s",
                )

            interactions = token_tracker.get_interactions("test-model")["test-model"]
            self.assertEqual(len(interactions), 1)
            timestamp = interactions[0]["timestamp"]
            self.assertIsInstance(timestamp, datetime)
            self.assertIsNotNone(timestamp.tzinfo)
        finally:
            token_tracker.reset()


if __name__ == "__main__":
    unittest.main()
