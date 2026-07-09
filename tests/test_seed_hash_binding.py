"""Seed role binds into content_hash.

Two nodes with byte-identical code + metric + llm_call_refs but different
``is_seed_node`` roles must hash differently — the semantic role is part
of the content address, not just an out-of-band label. Also verifies
strict back-compat: callers that don't pass ``is_seed=`` at all get the
exact same hash they always did.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import hash_node_payload
from ai_scientist.utils.ara_artifact import export_ara


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _minimal_project(tmp: Path, nodes: list[dict]) -> tuple[Path, Path]:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260710_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(exp / "logs", "0-run", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": "idea", "Title": "T"}), encoding="utf-8")
    return project, exp


CODE = "print('same')"
METRIC = {"value": 0.9, "maximize": True, "name": "acc"}


class SeedHashBindingUnitTests(unittest.TestCase):
    def test_seed_flag_changes_hash(self) -> None:
        h_plain = hash_node_payload(code=CODE, metric=METRIC)
        h_seed = hash_node_payload(code=CODE, metric=METRIC, is_seed=True)
        self.assertNotEqual(h_plain, h_seed)

    def test_default_call_is_backcompat(self) -> None:
        # The whole point: existing callers that never learned about the
        # is_seed param must get bit-identical hashes.
        h_no_arg = hash_node_payload(code=CODE, metric=METRIC)
        h_explicit_false = hash_node_payload(code=CODE, metric=METRIC, is_seed=False)
        self.assertEqual(h_no_arg, h_explicit_false)

    def test_seed_marker_orthogonal_to_llm_calls(self) -> None:
        # Both flags on: hash must differ from either flag alone.
        prompt_hash = "sha256:" + "a" * 64
        h_llm_only = hash_node_payload(
            code=CODE, metric=METRIC, llm_call_hashes=[prompt_hash]
        )
        h_seed_only = hash_node_payload(code=CODE, metric=METRIC, is_seed=True)
        h_both = hash_node_payload(
            code=CODE, metric=METRIC, llm_call_hashes=[prompt_hash], is_seed=True
        )
        self.assertNotEqual(h_both, h_llm_only)
        self.assertNotEqual(h_both, h_seed_only)
        self.assertNotEqual(h_llm_only, h_seed_only)


class SeedHashBindingExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_two_nodes_same_code_diff_seed_role_hash_differently(self) -> None:
        # Byte-identical journal rows other than is_seed_node — the sole
        # differentiator should still yield distinct content_hash values.
        project, exp = _minimal_project(self.tmp, [
            {
                "id": "seed", "step": 0, "code": CODE,
                "_term_out": [], "metric": METRIC,
                "is_buggy": False, "parent_id": None, "children": [],
                "is_seed_node": True,
            },
            {
                "id": "plain", "step": 1, "code": CODE,
                "_term_out": [], "metric": METRIC,
                "is_buggy": False, "parent_id": None, "children": [],
                "is_seed_node": False,
            },
        ])
        result = export_ara(project_dir=project, exp_dir=exp,
                            idea={"Name": "idea", "Title": "T"})
        ara = Path(result.root)

        seed_m = json.loads((ara / "nodes" / "seed" / "metrics.json").read_text())
        plain_m = json.loads((ara / "nodes" / "plain" / "metrics.json").read_text())

        # 1. Hashes must diverge.
        self.assertNotEqual(seed_m["content_hash"], plain_m["content_hash"])

        # 2. content_hash_inputs must reflect the role split.
        self.assertIn("seed", seed_m["content_hash_inputs"])
        self.assertNotIn("seed", plain_m["content_hash_inputs"])
        self.assertEqual(plain_m["content_hash_inputs"], ["code", "metric"])
        self.assertEqual(seed_m["content_hash_inputs"], ["code", "metric", "seed"])

        # 3. Same properties surface in exploration_graph.
        graph = json.loads((ara / "exploration_graph.json").read_text())
        by_id = {n["id"]: n for n in graph["nodes"]}
        self.assertIn("seed", by_id["seed"]["content_hash_inputs"])
        self.assertNotIn("seed", by_id["plain"]["content_hash_inputs"])
        self.assertNotEqual(
            by_id["seed"]["content_hash"], by_id["plain"]["content_hash"]
        )

        # 4. Export hashes must match a direct hash_node_payload call.
        expected_seed = hash_node_payload(code=CODE, metric=METRIC, is_seed=True)
        expected_plain = hash_node_payload(code=CODE, metric=METRIC)
        self.assertEqual(seed_m["content_hash"], expected_seed)
        self.assertEqual(plain_m["content_hash"], expected_plain)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
