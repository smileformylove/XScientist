from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.epistemic_graph import (
    EpistemicGraphError,
    advance_epistemic_node,
    build_epistemic_graph,
    build_epistemic_node,
    current_epistemic_status,
    seed_scientific_foundation,
    validate_epistemic_graph,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    load_pipeline_manifest,
)
from ai_scientist.utils.science_constitution import build_science_constitution


def _idea_card() -> dict:
    return {
        "idea_id": "idea_0",
        "title": "Mechanistic discovery",
        "core_hypothesis": "Mechanism A causes outcome B under condition C.",
        "failure_criteria": ["Outcome B is absent under preregistered condition C."],
        "literature_queries": ["mechanism A outcome B"],
        "source_idea": {},
    }


class EpistemicGraphTests(unittest.TestCase):
    def test_seed_graph_contains_question_hypothesis_and_relation(self) -> None:
        constitution = build_science_constitution(project_name="demo")
        graph = build_epistemic_graph(
            [_idea_card()],
            constitution=constitution,
            producer="test",
        )

        self.assertTrue(validate_epistemic_graph(graph)["passed"])
        self.assertEqual(
            {node["node_type"] for node in graph["nodes"]},
            {"question", "hypothesis"},
        )
        self.assertEqual(graph["edges"][0]["edge_type"], "addresses")

    def test_claim_like_nodes_require_falsifiers_and_start_speculative(self) -> None:
        with self.assertRaises(EpistemicGraphError):
            build_epistemic_node(
                node_type="claim",
                title="Unsupported claim",
                statement="Always true.",
                created_by="agent",
            )
        with self.assertRaises(EpistemicGraphError):
            build_epistemic_node(
                node_type="claim",
                title="Premature claim",
                statement="Always true.",
                created_by="agent",
                falsifiers=["A counterexample."],
                initial_status="canonical",
            )

    def test_state_advancement_is_evidence_gated_and_hash_chained(self) -> None:
        constitution = build_science_constitution(project_name="demo")
        graph = build_epistemic_graph(
            [_idea_card()],
            constitution=constitution,
            producer="test",
        )
        node_id = next(
            node["node_id"]
            for node in graph["nodes"]
            if node["node_type"] == "hypothesis"
        )
        with self.assertRaises(EpistemicGraphError):
            advance_epistemic_node(
                graph,
                node_id=node_id,
                to_status="grounded",
                actor_id="researcher",
                reason="Literature checked.",
                evidence_refs=[],
            )

        grounded = advance_epistemic_node(
            graph,
            node_id=node_id,
            to_status="grounded",
            actor_id="researcher",
            reason="Literature checked.",
            evidence_refs=["literature:review-1"],
        )
        preregistered = advance_epistemic_node(
            grounded,
            node_id=node_id,
            to_status="preregistered",
            actor_id="planner",
            reason="Protocol locked.",
            evidence_refs=["preregistration:sha256:abc"],
        )

        self.assertEqual(
            current_epistemic_status(preregistered, node_id),
            "preregistered",
        )
        self.assertEqual(
            preregistered["transitions"][1]["previous_event_hash"],
            preregistered["transitions"][0]["event_hash"],
        )
        self.assertTrue(validate_epistemic_graph(preregistered)["passed"])

        preregistered["transitions"][0]["reason"] = "post-hoc rewrite"
        self.assertFalse(validate_epistemic_graph(preregistered)["passed"])

    def test_refuted_nodes_are_terminal_and_preserved(self) -> None:
        constitution = build_science_constitution(project_name="demo")
        graph = build_epistemic_graph(
            [_idea_card()],
            constitution=constitution,
            producer="test",
        )
        node_id = next(
            node["node_id"]
            for node in graph["nodes"]
            if node["node_type"] == "hypothesis"
        )
        refuted = advance_epistemic_node(
            graph,
            node_id=node_id,
            to_status="refuted",
            actor_id="verifier",
            reason="A preregistered counterexample was observed.",
            evidence_refs=["observation:counterexample-1"],
        )
        with self.assertRaises(EpistemicGraphError):
            advance_epistemic_node(
                refuted,
                node_id=node_id,
                to_status="grounded",
                actor_id="researcher",
                reason="Try to revive in place.",
                evidence_refs=["literature:new"],
            )
        self.assertEqual(len(refuted["nodes"]), len(graph["nodes"]))

    def test_foundation_seed_persists_constitution_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            paths = seed_scientific_foundation(
                root,
                [_idea_card()],
                producer="test",
            )

            self.assertTrue(Path(paths["science_constitution"]).exists())
            self.assertTrue(Path(paths["epistemic_graph"]).exists())
            constitution = load_contract_artifact(
                root, "science_constitution", default={}
            )
            graph = load_contract_artifact(root, "epistemic_graph", default={})
            self.assertEqual(
                graph["constitution_hash"], constitution["constitution_hash"]
            )
            manifest = load_pipeline_manifest(root)
            self.assertEqual(
                manifest["artifacts"]["epistemic_graph"]["status"], "ready"
            )


if __name__ == "__main__":
    unittest.main()
