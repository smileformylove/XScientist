from __future__ import annotations

import copy
import unittest

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.observability_binding import build_observability_binding_projection


class ObservabilityBindingProjectionTests(unittest.TestCase):
    def _rehash(self, graph: dict) -> None:
        graph["graph_hash"] = canonical_content_hash(
            {
                key: value
                for key, value in graph.items()
                if key not in {"generated_at", "graph_hash"}
            }
        )

    def _graph(self) -> dict:
        commit = "a" * 40
        manifest_hash = "sha256:" + "1" * 64
        ara_graph_hash = "sha256:" + "2" * 64
        base = {
            "schema_version": "xscientist.research-dag.v1",
            "commit": commit,
            "sources": [
                {"name": "research_vcs", "object_count": 1},
                {
                    "name": "ara:0",
                    "manifest_hash": manifest_hash,
                    "manifest_integrity": {
                        "state": "clean",
                        "ok": True,
                        "current_hash": manifest_hash,
                        "expected_hash": manifest_hash,
                    },
                    "graph_binding": {
                        "state": "verified",
                        "ok": True,
                        "graph_hash": ara_graph_hash,
                        "expected_graph_hash": ara_graph_hash,
                    },
                    "snapshot_ref": "b" * 40,
                    "integrity": {"is_dag": True, "error_count": 0},
                    "bound_object_ids": ["rso-0000000000000001"],
                    "binding_count": 1,
                },
            ],
            "nodes": [
                {"id": "rso-0000000000000001", "source": "research_vcs"},
                {"id": "ara:0:n1", "source": "ara"},
            ],
            "edges": [
                {
                    "source": "ara:0:n1",
                    "target": "rso-0000000000000001",
                    "type": "anchors",
                    "category": "verification",
                }
            ],
        }
        return {
            **base,
            "generated_at": "2026-09-21T00:00:00Z",
            "graph_hash": canonical_content_hash(base),
        }

    def test_verified_history_snapshot_exposes_existing_graph_hash(self) -> None:
        graph = self._graph()
        projection = build_observability_binding_projection(graph)
        self.assertEqual(projection["state"], "verified")
        self.assertEqual(projection["verified_count"], 1)
        row = projection["rows"][0]
        self.assertEqual(row["snapshot_ref"], "b" * 40)
        self.assertNotEqual(row["binding_hash"], graph["graph_hash"])
        self.assertTrue(row["binding_hash"].startswith("sha256:"))
        self.assertEqual(row["anchor_object_ids"], ["rso-0000000000000001"])

    def test_real_seq68_export_is_partial_but_fail_closed(self) -> None:
        from pathlib import Path
        import re
        import json

        path = (
            Path(__file__).resolve().parents[3]
            / "experiments/XScientist/iclr2026/07_audits/20260920_latent_trajectory_contract_v1/preview/seq68-research-dag.html"
        )
        if not path.exists():
            self.skipTest("seq68 preview is not available in this checkout")
        match = re.search(
            r'<script id="dag-data" type="application/json">(.*?)</script>',
            path.read_text(encoding="utf-8"),
        )
        self.assertIsNotNone(match)
        projection = build_observability_binding_projection(json.loads(match.group(1)))
        self.assertEqual(projection["state"], "partial")
        self.assertEqual(projection["verified_count"], 4)
        self.assertEqual(projection["invalid_count"], 0)
        self.assertTrue(
            all(
                row["binding_hash"]
                for row in projection["rows"]
                if row["state"] == "verified"
            )
        )

    def test_missing_anchor_is_unbound_and_has_no_hash(self) -> None:
        graph = self._graph()
        graph["edges"] = []
        graph["sources"][1]["bound_object_ids"] = []
        graph["sources"][1]["binding_count"] = 0
        self._rehash(graph)
        projection = build_observability_binding_projection(graph)
        self.assertEqual(projection["state"], "unbound")
        self.assertIsNone(projection["rows"][0]["binding_hash"])
        self.assertIn(
            "missing_exact_research_object_anchor", projection["rows"][0]["reasons"]
        )

    def test_tampered_hash_and_anchor_endpoint_are_invalid(self) -> None:
        graph = self._graph()
        graph["sources"][1]["graph_binding"]["graph_hash"] = "sha256:" + "3" * 64
        graph["edges"][0]["target"] = "not-a-research-node"
        projection = build_observability_binding_projection(graph)
        self.assertEqual(projection["state"], "invalid")
        self.assertIn("ara_graph_hash_mismatch", projection["rows"][0]["reasons"])
        self.assertIn("anchor_set_mismatch", projection["rows"][0]["reasons"])
        self.assertIsNone(projection["rows"][0]["binding_hash"])

    def test_malformed_anchor_metadata_fails_closed_without_raising(self) -> None:
        graph = self._graph()
        graph["sources"][1]["bound_object_ids"] = "rso-0000000000000001"
        graph["sources"][1]["binding_count"] = "1"
        projection = build_observability_binding_projection(graph)
        self.assertEqual(projection["state"], "invalid")
        self.assertIn("invalid_declared_anchor_set", projection["rows"][0]["reasons"])

    def test_top_level_graph_hash_tamper_is_invalid(self) -> None:
        graph = self._graph()
        graph["commit"] = "c" * 40
        projection = build_observability_binding_projection(graph)
        self.assertEqual(projection["state"], "invalid")
        self.assertIn("research_graph_hash_mismatch", projection["issues"])

    def test_projection_does_not_mutate_input_or_infer_context_anchors(self) -> None:
        graph = self._graph()
        graph["sources"][1]["bound_object_ids"] = []
        graph["sources"][1]["binding_count"] = 0
        graph["edges"][0].update(type="contextualizes", category="context")
        self._rehash(graph)
        original = copy.deepcopy(graph)
        projection = build_observability_binding_projection(graph)
        self.assertEqual(graph, original)
        self.assertEqual(projection["state"], "unbound")
        self.assertEqual(projection["rows"][0]["anchor_count"], 0)

    def test_malformed_counts_and_mixed_ids_never_receive_binding_hash(self) -> None:
        for count, ids in [
            (True, ["a"]),
            (1.0, ["a"]),
            ({}, []),
            (1, [None]),
            (2, ["a", {}]),
            (-1, []),
        ]:
            with self.subTest(count=count, ids=ids):
                graph = self._graph()
                graph["sources"][1].update(binding_count=count, bound_object_ids=ids)
                self._rehash(graph)
                projection = build_observability_binding_projection(graph)
                self.assertEqual(projection["state"], "invalid")
                self.assertIsNone(projection["rows"][0]["binding_hash"])

    def test_metadata_cannot_hide_missing_anchor_or_claim_endpoint(self) -> None:
        for update in ("missing", "wrong_endpoint", "lineage", "duplicate"):
            with self.subTest(update=update):
                graph = self._graph()
                if update == "missing":
                    graph["edges"] = []
                elif update == "wrong_endpoint":
                    graph["nodes"][0]["source"] = "ara"
                elif update == "lineage":
                    graph["edges"][0]["category"] = "lineage"
                else:
                    graph["nodes"].append(copy.deepcopy(graph["nodes"][1]))
                self._rehash(graph)
                projection = build_observability_binding_projection(graph)
                self.assertNotEqual(projection["state"], "verified")
                self.assertIsNone(projection["rows"][0]["binding_hash"])

    def test_external_worktree_and_legacy_lock_only_remain_unbound(self) -> None:
        for state in ("external", "legacy"):
            with self.subTest(state=state):
                graph = self._graph()
                if state == "external":
                    graph["sources"][1]["snapshot_ref"] = "worktree"
                else:
                    graph["sources"][1]["manifest_integrity"][
                        "state"
                    ] = "legacy_lock_only"
                self._rehash(graph)
                projection = build_observability_binding_projection(graph)
                self.assertEqual(projection["state"], "unbound")
                self.assertIsNone(projection["rows"][0]["binding_hash"])


if __name__ == "__main__":
    unittest.main()
