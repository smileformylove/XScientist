from __future__ import annotations

import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from xscientist import ResearchRepository
from xscientist.research_context import (
    build_research_context_snapshot,
    record_research_context_snapshot,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for context DAG tests")
class ResearchContextDagEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = ResearchRepository.init(
            Path(self.temp.name) / "research",
            name="context-edge-fixture",
            question="Synthetic context projection test",
            git_user_name="Context Edge Test",
            git_user_email="context-edge@example.invalid",
        )
        self.source = self.repository.record(
            "hypothesis", {"statement": "Synthetic H1", "falsifier": "not H1"}
        )

    def _incoming(self, graph: dict, object_id: str) -> list[dict]:
        return [edge for edge in graph["edges"] if edge["target"] == object_id]

    def test_only_bound_dependencies_are_context_independent_of_role(self) -> None:
        payload = build_research_context_snapshot(
            self.repository, target_ids=[self.source.object_id]
        )
        incidental = self.repository.record(
            "hypothesis", {"statement": "Unconsumed H2", "falsifier": "not H2"}
        )
        snapshots = []
        for role in ("context_source", "premise", ""):
            source_relation = {"type": "depends_on", "target": self.source.object_id}
            if role:
                source_relation["role"] = role
            snapshots.append(
                self.repository.record(
                    "context_snapshot",
                    payload,
                    state="completed",
                    relations=[
                        source_relation,
                        {
                            "type": "depends_on",
                            "target": incidental.object_id,
                            "role": "context_source",
                        },
                        {
                            "type": "qualifies",
                            "target": self.source.object_id,
                            "role": "context_source",
                        },
                        {
                            "type": "refutes",
                            "target": self.source.object_id,
                            "role": "context_source",
                        },
                    ],
                )
            )
        ordinary = self.repository.record(
            "evidence",
            {"result": "Synthetic fixture", "measurement_hash": "sha256:" + "a" * 64},
            state="completed",
            relations=[
                {
                    "type": "depends_on",
                    "target": self.source.object_id,
                    "role": "context_source",
                }
            ],
        )
        self.repository.commit(stage="context", subject="record context edge controls")
        semantic_edges = None
        for disclose in (True, False):
            with self.subTest(disclose_summaries=disclose):
                graph = self.repository.dag(disclose_summaries=disclose)
                for snapshot in snapshots:
                    actual = {
                        (edge["source"], edge["type"]): edge["category"]
                        for edge in self._incoming(graph, snapshot.object_id)
                    }
                    self.assertEqual(
                        actual,
                        {
                            (self.source.object_id, "depends_on"): "context",
                            (incidental.object_id, "depends_on"): "lineage",
                            (self.source.object_id, "qualifies"): "argument",
                            (self.source.object_id, "refutes"): "challenge",
                        },
                    )
                self.assertEqual(
                    [
                        edge["category"]
                        for edge in self._incoming(graph, ordinary.object_id)
                    ],
                    ["lineage"],
                )
                current = [
                    {key: value for key, value in edge.items() if key != "role"}
                    for edge in graph["edges"]
                ]
                if semantic_edges is not None:
                    self.assertEqual(current, semantic_edges)
                    self.assertTrue(all(edge["role"] == "" for edge in graph["edges"]))
                semantic_edges = current

    def test_invalid_context_identity_cannot_reclassify_dependency(self) -> None:
        payload = build_research_context_snapshot(
            self.repository, target_ids=[self.source.object_id]
        )
        invalid = copy.deepcopy(payload)
        invalid["context_hash"] = "sha256:" + "0" * 64
        snapshot = self.repository.record(
            "context_snapshot",
            invalid,
            state="completed",
            relations=[
                {
                    "type": "depends_on",
                    "target": self.source.object_id,
                    "role": "context_source",
                }
            ],
        )
        self.repository.commit(
            stage="context", subject="record invalid context fixture"
        )
        graph = self.repository.dag()
        node = next(node for node in graph["nodes"] if node["id"] == snapshot.object_id)
        self.assertIn("context_identity_valid", node["proof"]["blockers"])
        self.assertEqual(
            self._incoming(graph, snapshot.object_id)[0]["category"], "lineage"
        )

    def test_identity_valid_incomplete_context_keeps_source_edges(self) -> None:
        snapshot = record_research_context_snapshot(
            self.repository, target_ids=[self.source.object_id], budget_tokens=128
        )
        self.repository.commit(
            stage="context", subject="record incomplete context fixture"
        )
        graph = self.repository.dag()
        node = next(node for node in graph["nodes"] if node["id"] == snapshot.object_id)
        self.assertEqual(node["state"], "rejected")
        self.assertNotIn("context_identity_valid", node["proof"]["blockers"])
        self.assertIn("context_complete", node["proof"]["blockers"])
        self.assertEqual(
            self._incoming(graph, snapshot.object_id)[0]["category"], "context"
        )

    def test_context_chain_and_consumer_relations_keep_distinct_categories(
        self,
    ) -> None:
        first = record_research_context_snapshot(
            self.repository, target_ids=[self.source.object_id]
        )
        second = record_research_context_snapshot(
            self.repository,
            target_ids=[self.source.object_id],
            rationale=["Next decision"],
        )
        decision = self.repository.record(
            "gate_decision",
            {"decision": "hold", "claim_promotion_allowed": False},
            state="rejected",
            relations=[{"type": "uses_context", "target": second.object_id}],
        )
        self.repository.commit(stage="context", subject="record context chain fixture")
        graph = self.repository.dag()
        incoming = self._incoming(graph, second.object_id)
        self.assertEqual(
            {
                edge["type"]: edge["category"]
                for edge in incoming
                if edge["source"] == first.object_id
            },
            {"depends_on": "context", "supersedes": "evolution"},
        )
        payload = self.repository.get(second.object_id)["payload"]
        self.assertEqual(
            {edge["source"] for edge in incoming if edge["category"] == "context"},
            set(payload["source_object_ids"]),
        )
        self.assertEqual(
            self._incoming(graph, decision.object_id),
            [
                {
                    "source": second.object_id,
                    "target": decision.object_id,
                    "type": "uses_context",
                    "role": "",
                    "category": "context",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
