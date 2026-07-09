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

import dataclasses
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


class NodePlotExecSerializationTests(unittest.TestCase):
    """Regression for the plot_* exec fields being dropped by to_dict.

    Same failure class as llm_call_refs (fix in 396a771): the fields
    exist on the dataclass but are never emitted, so the Node round-trip
    through parallel_agent worker → parent and through save_run →
    journal.json → ara_artifact silently loses plot execution info.

    ara_artifact reads plot_term_out / plot_exc_* off the deserialised
    journal, and parallel_agent.py:1788 branches on child_node.plot_exc_type
    to decide whether the plot step succeeded — both are load-bearing.
    """

    _DISTINCTIVE = {
        "plot_term_out": ["plot line 1\n", "plot line 2\n"],
        "plot_exec_time": 12.5,
        "plot_exc_type": "ValueError",
        "plot_exc_info": {"args": ["broken plot"]},
        "plot_exc_stack": [("plot.py", 42, "make_plot", "raise ValueError()")],
    }

    def _make_node(self) -> Node:
        return Node(code="c", plan="p", **self._DISTINCTIVE)

    def test_to_dict_emits_all_plot_exec_fields(self) -> None:
        d = self._make_node().to_dict()
        for key, expected in self._DISTINCTIVE.items():
            self.assertIn(key, d, f"to_dict dropped {key}")
            self.assertEqual(d[key], expected, f"to_dict garbled {key}")

    def test_from_dict_round_trips_plot_exec_fields(self) -> None:
        n2 = Node.from_dict(self._make_node().to_dict())
        for attr, expected in self._DISTINCTIVE.items():
            self.assertEqual(
                getattr(n2, attr), expected,
                f"round-trip lost {attr}",
            )

    def test_plot_exec_fields_survive_json_pipeline(self) -> None:
        serialised = json.dumps(self._make_node().to_dict(), default=str)
        reloaded = json.loads(serialised)
        for key, expected in self._DISTINCTIVE.items():
            # JSON coerces tuples in exc_stack to lists; compare after
            # the same normalisation so the test asserts "the value
            # survived", not "the value stayed a tuple".
            self.assertEqual(
                reloaded.get(key),
                json.loads(json.dumps(expected, default=str)),
            )


class NodeToDictCoverageGuardTests(unittest.TestCase):
    """Mechanical guard: every declared Node field must appear in to_dict().

    This is what would have caught the plot_* omission (and the earlier
    llm_call_refs omission) automatically. Fields with genuinely custom
    serialisation (parent → parent_id) are the only allowed exclusions.
    """

    # `parent` is intentionally renamed to `parent_id` in to_dict; every
    # other declared field must appear as a key.
    _CUSTOM_KEY_RENAMES = {"parent": "parent_id"}

    def test_every_dataclass_field_reachable_in_to_dict(self) -> None:
        d = Node(code="c", plan="p").to_dict()
        missing: list[str] = []
        for f in dataclasses.fields(Node):
            expected_key = self._CUSTOM_KEY_RENAMES.get(f.name, f.name)
            if expected_key not in d:
                missing.append(f"{f.name} (expected key {expected_key!r})")
        self.assertEqual(
            missing, [],
            "Node.to_dict() drops declared fields: "
            + ", ".join(missing),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
