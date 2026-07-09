"""Regression: Node.to_dict must round-trip llm_call_refs.

Discovered during the 24h auto-optimization loop's hour 1 review pass.
Without this, every content_hash in a real BFTS run silently reverted to
the pre-Phase-1.5 scheme (code + metric only, no prompt binding) because
llm_call_refs was populated in memory but stripped during the
Node → journal.json → Node round-trip that happens between agent worker
and parent process.

The existing test_node_llm_refs_hashing.py suite hand-crafts journal.json
with llm_call_refs already present, so it never exercised the to_dict()
path.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.treesearch.journal import Node


class NodeSerializationRoundTripTests(unittest.TestCase):
    def test_to_dict_preserves_llm_call_refs(self) -> None:
        n = Node(
            code="print('x')",
            plan="p",
            llm_call_refs=["sha256:" + "a" * 64, "sha256:" + "b" * 64],
        )
        d = n.to_dict()
        self.assertIn("llm_call_refs", d)
        self.assertEqual(d["llm_call_refs"], [
            "sha256:" + "a" * 64, "sha256:" + "b" * 64,
        ])

    def test_from_dict_restores_llm_call_refs(self) -> None:
        n = Node(
            code="c",
            plan="p",
            llm_call_refs=["sha256:" + "c" * 64],
        )
        n2 = Node.from_dict(n.to_dict())
        self.assertEqual(n2.llm_call_refs, ["sha256:" + "c" * 64])

    def test_empty_llm_call_refs_survives_round_trip(self) -> None:
        # Seed nodes and legacy runs have no refs — the field must round-trip
        # as an empty list, not disappear or turn into None.
        n = Node(code="c", plan="p")
        d = n.to_dict()
        self.assertEqual(d.get("llm_call_refs"), [])
        n2 = Node.from_dict(d)
        self.assertEqual(n2.llm_call_refs, [])

    def test_journal_json_serialisable_contains_refs(self) -> None:
        # Full JSON pipeline check — the actual failure mode was that
        # writing to disk and reading back dropped the field.
        n = Node(
            code="c", plan="p",
            llm_call_refs=["sha256:" + "d" * 64],
        )
        serialised = json.dumps(n.to_dict(), default=str)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.json"
            path.write_text(serialised, encoding="utf-8")
            reloaded = json.loads(path.read_text())
        self.assertEqual(reloaded.get("llm_call_refs"),
                         ["sha256:" + "d" * 64])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
