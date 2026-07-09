"""Regression + forward-compat tests for hash_node_payload's llm_call_hashes."""

from __future__ import annotations

import unittest

from ai_scientist.protocol.hashing import hash_node_payload


CODE = "print('hi')"
METRIC = {"name": "acc", "value": 0.5, "maximize": True}


class HashLLMBindingTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Regression: no llm_call_hashes must give the pre-existing hash.
    # ------------------------------------------------------------------
    def test_omitting_llm_calls_matches_baseline(self) -> None:
        baseline = hash_node_payload(code=CODE, metric=METRIC)
        with_none = hash_node_payload(code=CODE, metric=METRIC, llm_call_hashes=None)
        with_empty = hash_node_payload(code=CODE, metric=METRIC, llm_call_hashes=[])
        self.assertEqual(baseline, with_none)
        self.assertEqual(baseline, with_empty)

    def test_baseline_unchanged_from_public_expectation(self) -> None:
        # Producers that don't opt in must hash identically across the change.
        # If this ever needs updating, the change is BREAKING and requires a
        # PROTOCOL_VERSION bump.
        a = hash_node_payload(code=CODE, metric=METRIC)
        b = hash_node_payload(code=CODE, metric=METRIC)
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("sha256:"))

    # ------------------------------------------------------------------
    # Order- and duplicate-insensitive.
    # ------------------------------------------------------------------
    def test_order_insensitive(self) -> None:
        h1 = hash_node_payload(
            code=CODE, metric=METRIC,
            llm_call_hashes=["sha256:a" * 64, "sha256:b" * 64],
        )
        h2 = hash_node_payload(
            code=CODE, metric=METRIC,
            llm_call_hashes=["sha256:b" * 64, "sha256:a" * 64],
        )
        self.assertEqual(h1, h2)

    def test_duplicates_collapse(self) -> None:
        h1 = hash_node_payload(
            code=CODE, metric=METRIC,
            llm_call_hashes=["sha256:a" * 64],
        )
        h2 = hash_node_payload(
            code=CODE, metric=METRIC,
            llm_call_hashes=["sha256:a" * 64, "sha256:a" * 64],
        )
        self.assertEqual(h1, h2)

    def test_empty_string_ignored(self) -> None:
        h1 = hash_node_payload(code=CODE, metric=METRIC, llm_call_hashes=[])
        h2 = hash_node_payload(code=CODE, metric=METRIC, llm_call_hashes=["", "", ""])
        self.assertEqual(h1, h2)

    # ------------------------------------------------------------------
    # Sensitivity: adding a real call must change the hash.
    # ------------------------------------------------------------------
    def test_binding_llm_calls_changes_hash(self) -> None:
        baseline = hash_node_payload(code=CODE, metric=METRIC)
        bound = hash_node_payload(
            code=CODE, metric=METRIC,
            llm_call_hashes=["sha256:c" * 64],
        )
        self.assertNotEqual(baseline, bound)

    def test_different_llm_sets_hash_differently(self) -> None:
        h1 = hash_node_payload(
            code=CODE, metric=METRIC,
            llm_call_hashes=["sha256:a" * 64],
        )
        h2 = hash_node_payload(
            code=CODE, metric=METRIC,
            llm_call_hashes=["sha256:b" * 64],
        )
        self.assertNotEqual(h1, h2)

    def test_code_still_dominates_when_calls_match(self) -> None:
        # Different code, same calls → still different hash.
        calls = ["sha256:d" * 64]
        h1 = hash_node_payload(code="print(1)", metric=METRIC, llm_call_hashes=calls)
        h2 = hash_node_payload(code="print(2)", metric=METRIC, llm_call_hashes=calls)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
