from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash as protocol_content_hash
from ai_scientist.protocol.research_vcs import build_research_object
from xscientist import ResearchRepository
import xscientist.research_argument as research_argument_module
from xscientist.research_argument import (
    ADVISORY_ROUTE_INDEX_SCHEMA,
    ARA_CLAIM_ROLE_BINDING_SCHEMA,
    ResearchArgumentError,
    build_research_argument_view,
    load_advisory_route_index,
    project_advisory_route_index,
    validate_research_argument_view,
)


def _seal_graph(graph: dict) -> dict:
    core = {
        key: value
        for key, value in graph.items()
        if key not in {"generated_at", "graph_hash"}
    }
    graph["graph_hash"] = canonical_content_hash(core)
    return graph


def _seal_closure(base: dict) -> dict:
    return {**base, "content_hash": protocol_content_hash(base)}


@unittest.skipUnless(shutil.which("git"), "Git is required")
class AdvisoryRouteIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "study"
        ResearchRepository.init(
            self.root,
            question="Which latent world-model claim survives?",
            git_user_name="Argument Test",
            git_user_email="argument@example.invalid",
        )
        (self.root / "advisory").mkdir()
        self.artifact = self.root / "advisory" / "witness.md"
        self.contract = self.root / "advisory" / "contract.json"
        self.artifact.write_bytes(b"exact witness bytes\n")
        self.contract.write_bytes(b'{"gate":"W0-W5"}\n')
        self.index_path = self.root / "advisory-index.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _index(self, *, artifact_path: str = "advisory/witness.md") -> dict:
        return {
            "schema_version": ADVISORY_ROUTE_INDEX_SCHEMA,
            "routes": [
                {
                    "route_id": "Q5-XM",
                    "question": "Does the finite transient witness survive W0-W5?",
                    "authority": "UNREGISTERED",
                    "artifact_bindings": [
                        {
                            "path": artifact_path,
                            "raw_sha256": self._digest(self.artifact),
                        }
                    ],
                    "gate_contract_binding": {
                        "path": "advisory/contract.json",
                        "raw_sha256": self._digest(self.contract),
                    },
                    "gate_results": [
                        {"gate_id": "W0", "status": "PASS"},
                        {"gate_id": "W5", "status": "UNREVIEWED"},
                    ],
                    "open_findings": [
                        {"severity": "P1", "statement": "Novelty is not closed."}
                    ],
                    "novelty_disposition": "OPEN",
                    "registration_state": "absent",
                    "permitted_next_action": "Finish independent W5 adjudication.",
                }
            ],
        }

    def _write_index(self, value: dict | str | bytes) -> None:
        if isinstance(value, dict):
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
        else:
            encoded = value
        self.index_path.write_bytes(encoded)

    def test_loads_exact_contract_and_rejects_ambiguous_json(self) -> None:
        self._write_index(self._index())
        loaded = load_advisory_route_index(self.root, self.index_path)
        self.assertEqual(loaded["routes"][0]["route_id"], "Q5-XM")

        duplicate = (
            '{"schema_version":"xscientist.advisory-route-index.v1",'
            '"schema_version":"xscientist.advisory-route-index.v1","routes":[]}'
        )
        for raw in (duplicate, '{"schema_version":NaN,"routes":[]}'):
            with self.subTest(raw=raw):
                self._write_index(raw)
                with self.assertRaises(ResearchArgumentError):
                    load_advisory_route_index(self.root, self.index_path)

        self._write_index(b"\xff")
        with self.assertRaises(ResearchArgumentError):
            load_advisory_route_index(self.root, self.index_path)

    def test_hash_syntax_is_bare_lowercase_only(self) -> None:
        for invalid in (
            "sha256:" + "a" * 64,
            "A" * 64,
            "a" * 63,
        ):
            with self.subTest(invalid=invalid):
                payload = self._index()
                payload["routes"][0]["artifact_bindings"][0]["raw_sha256"] = invalid
                self._write_index(payload)
                with self.assertRaises(ResearchArgumentError):
                    load_advisory_route_index(self.root, self.index_path)

    def test_traversal_symlink_hardlink_and_budgets_fail_closed(self) -> None:
        payload = self._index(artifact_path="../outside.md")
        self._write_index(payload)
        with self.assertRaises(ResearchArgumentError):
            load_advisory_route_index(self.root, self.index_path)

        link = self.root / "advisory" / "linked.md"
        link.symlink_to(self.artifact)
        payload = self._index(artifact_path="advisory/linked.md")
        payload["routes"][0]["artifact_bindings"][0]["raw_sha256"] = self._digest(
            self.artifact
        )
        self._write_index(payload)
        with self.assertRaisesRegex(ResearchArgumentError, "symlink"):
            project_advisory_route_index(self.root, self.index_path)

        hardlink = self.root / "advisory" / "hardlinked.md"
        try:
            os.link(self.artifact, hardlink)
        except OSError:
            hardlink = None
        if hardlink is not None:
            payload = self._index(artifact_path="advisory/hardlinked.md")
            self._write_index(payload)
            with self.assertRaisesRegex(ResearchArgumentError, "hardlink"):
                project_advisory_route_index(self.root, self.index_path)
            hardlink.unlink()

        self._write_index(self._index())
        with mock.patch.object(
            research_argument_module, "_MAX_ADVISORY_INDEX_BYTES", 8
        ):
            with self.assertRaisesRegex(ResearchArgumentError, "exceeds"):
                load_advisory_route_index(self.root, self.index_path)
        with mock.patch.object(
            research_argument_module, "_MAX_ADVISORY_TOTAL_BYTES", 1
        ):
            with self.assertRaisesRegex(ResearchArgumentError, "aggregate"):
                project_advisory_route_index(self.root, self.index_path)

    def test_stale_byte_suppresses_entire_route(self) -> None:
        self._write_index(self._index())
        verified = project_advisory_route_index(self.root, self.index_path)
        self.assertEqual(verified["routes"][0]["binding_state"], "VERIFIED")
        self.assertRegex(verified["index_binding"]["raw_sha256"], r"^[0-9a-f]{64}$")

        self.artifact.write_bytes(b"one changed byte!\n")
        stale = project_advisory_route_index(self.root, self.index_path)["routes"][0]
        self.assertEqual(
            set(stale),
            {
                "route_id",
                "authority",
                "registration_state",
                "binding_state",
                "route_state",
                "binding_failures",
            },
        )
        self.assertEqual(stale["binding_state"], "HASH_MISMATCH")
        self.assertEqual(stale["route_state"], "BLOCKED")
        self.assertNotIn("question", stale)
        self.assertNotIn("permitted_next_action", stale)

    def test_metadata_only_suppresses_advisory_substance(self) -> None:
        self._write_index(self._index())
        route = project_advisory_route_index(
            self.root, self.index_path, disclose_summaries=False
        )["routes"][0]
        for key in (
            "question",
            "gate_results",
            "open_findings",
            "novelty_disposition",
            "permitted_next_action",
        ):
            self.assertNotIn(key, route)
        self.assertEqual(route["authority"], "UNREGISTERED")
        self.assertEqual(route["registration_state"], "absent")


class ResearchArgumentProjectionTests(unittest.TestCase):
    @staticmethod
    def _base_graph(
        *, nodes: list[dict] | None = None, edges: list[dict] | None = None
    ) -> dict:
        return _seal_graph(
            {
                "schema_version": "xscientist.research-dag.v1",
                "ref": "HEAD",
                "commit": "8" * 40,
                "content_disclosed": True,
                "sources": [{"name": "research_vcs", "object_count": 0}],
                "nodes": nodes or [],
                "edges": edges or [],
                "theory_frontier": {
                    "submission_candidate_pool": {
                        "candidate_count": 0,
                        "experiment_authorized": False,
                    }
                },
                "scientific_closure": {"status": "not_applicable", "claim_count": 0},
                "claim_insights": [],
                "generated_at": "2026-09-14T00:00:00+00:00",
            }
        )

    @staticmethod
    def _rebind_closure(graph: dict, closure: dict) -> None:
        closure_core = {
            key: value for key, value in closure.items() if key != "content_hash"
        }
        closure["content_hash"] = protocol_content_hash(closure_core)
        graph["scientific_closure"].update(
            {
                "status": closure["status"],
                "blocker_count": len(closure["blockers"]),
                "warning_count": len(closure["warnings"]),
                "closure_levels": closure["closure_levels"],
                "content_hash": closure["content_hash"],
            }
        )
        _seal_graph(graph)

    @staticmethod
    def _append_research_node(graph: dict, item: dict, *, summary: str) -> None:
        graph["nodes"].append(
            {
                "id": item["object_id"],
                "source": "research_vcs",
                "kind": item["kind"],
                "state": item["state"],
                "summary": summary,
                "actor": dict(item["actor"]),
                "content_hash": item["content_hash"],
                "source_ref": graph["commit"],
            }
        )
        graph["sources"][0]["object_count"] += 1

    @staticmethod
    def _mark_closure_verified(graph: dict, closure: dict) -> None:
        row = closure["claims"][0]
        row.update(
            {
                "trace_complete": True,
                "replay_ready": True,
                "verified": True,
                "complete": True,
                "missing": [],
            }
        )
        closure.update({"status": "complete", "complete": True, "blockers": []})
        closure["closure_levels"] = {
            level: {
                "complete": True,
                "status": "complete",
                "claim_count": 1,
                "complete_claim_count": 1,
                "blocker_count": 0,
                "warning_count": 0,
                "blocker_codes": [],
                "warning_codes": [],
            }
            for level in ("trace", "replay", "verify")
        }
        ResearchArgumentProjectionTests._rebind_closure(graph, closure)

    def test_empty_claim_state_and_exact_denominators_are_advisory_only(self) -> None:
        categories = {
            "argument": 4,
            "challenge": 10,
            "support": 3,
            "verification": 60,
            "context": 91,
            "evolution": 335,
            "lineage": 278,
            "decision": 17,
        }
        edges = [
            {
                "source": f"s-{category}-{index}",
                "target": f"t-{category}-{index}",
                "type": category,
                "role": "",
                "category": category,
            }
            for category, count in categories.items()
            for index in range(count)
        ]
        nodes = [
            {
                "id": f"ara:0:n{index}",
                "source": "ara",
                "kind": "experiment_node",
            }
            for index in range(381)
        ]
        graph = self._base_graph(nodes=nodes, edges=edges)
        original = copy.deepcopy(graph)
        view = build_research_argument_view(graph)

        self.assertEqual(graph, original)
        self.assertEqual(view["registered_claims"], [])
        self.assertEqual(view["counters"]["registered_claims"], 0)
        self.assertEqual(
            view["counters"]["direct_semantic_edges"], {"count": 17, "total": 798}
        )
        self.assertEqual(
            view["counters"]["verification_edges"], {"count": 60, "total": 798}
        )
        self.assertEqual(
            view["counters"]["provenance_context_evolution_edges"],
            {"count": 704, "total": 798},
        )
        self.assertEqual(
            view["counters"]["ara_claim_role_nodes"], {"bound": 0, "total": 381}
        )
        self.assertEqual(sum(view["counters"]["edge_categories"].values()), 798)
        base = {key: value for key, value in view.items() if key != "content_hash"}
        self.assertEqual(view["content_hash"], canonical_content_hash(base))

    @staticmethod
    def _claim_fixture(*, include_role_binding: bool = True) -> tuple[dict, list, dict]:
        manifest_hash = "sha256:" + "3" * 64
        graph_hash = "sha256:" + "4" * 64
        selected = ["node-a"]
        claim = build_research_object(
            kind="claim",
            state="verified",
            payload={
                "statement": "The latent state predicts the intervention.",
                "depth_level": "causal",
            },
            created_at="2026-09-14T00:00:00Z",
        )
        claim_id = claim["object_id"]
        binding = {
            "schema_version": ARA_CLAIM_ROLE_BINDING_SCHEMA,
            "claim_id": claim_id,
            "manifest_hash": manifest_hash,
            "exploration_graph_hash": graph_hash,
            "selected_node_ids": selected,
            "selected_node_ids_hash": canonical_content_hash(selected),
            "role": "support",
            "metric": {
                "name": "held_out_nll",
                "direction": "minimize",
                "threshold": 0.0,
            },
            "control_ids": ["matched-null"],
            "execution_receipt_hash": "sha256:" + "5" * 64,
            "reproduction_state": "NOT_REPRODUCED",
        }
        evidence_payload = {
            "result": "Bound result",
            "ara_manifest_hash": manifest_hash,
            "ara_exploration_graph_hash": graph_hash,
        }
        if include_role_binding:
            evidence_payload["ara_claim_role_bindings"] = [binding]
        evidence = build_research_object(
            kind="evidence",
            state="completed",
            payload=evidence_payload,
            relations=[{"type": "supports", "target": claim_id}],
            created_at="2026-09-14T00:01:00Z",
        )
        evidence_id = evidence["object_id"]
        objects = [claim, evidence]
        graph = ResearchArgumentProjectionTests._base_graph(
            nodes=[
                {
                    "id": claim_id,
                    "source": "research_vcs",
                    "kind": "claim",
                    "state": "verified",
                    "summary": "The latent state predicts the intervention.",
                    "actor": dict(claim["actor"]),
                    "content_hash": claim["content_hash"],
                    "source_ref": "8" * 40,
                },
                {
                    "id": evidence_id,
                    "source": "research_vcs",
                    "kind": "evidence",
                    "state": "completed",
                    "summary": "Bound result",
                    "actor": dict(evidence["actor"]),
                    "content_hash": evidence["content_hash"],
                    "source_ref": "8" * 40,
                },
                {
                    "id": "ara:0:node-a",
                    "source": "ara",
                    "kind": "experiment_node",
                    "state": "completed",
                    "content_hash": "sha256:" + "c" * 64,
                    "proof": {
                        "level": "replayable",
                        "checks": [{"passed": True}],
                        "blockers": [],
                        "contested": False,
                    },
                },
                {
                    "id": "ara:0:matched-null",
                    "source": "ara",
                    "kind": "experiment_node",
                    "state": "completed",
                    "content_hash": "sha256:" + "d" * 64,
                    "proof": {
                        "level": "replayable",
                        "checks": [{"passed": True}],
                        "blockers": [],
                        "contested": False,
                    },
                },
            ],
            edges=[
                {
                    "source": claim_id,
                    "target": evidence_id,
                    "type": "supports",
                    "role": "",
                    "category": "support",
                }
            ],
        )
        graph["sources"] = [
            {"name": "research_vcs", "object_count": 2},
            {
                "name": "ara:0",
                "manifest_hash": manifest_hash,
                "manifest_hashes": [manifest_hash],
                "manifest_integrity": {"ok": True, "state": "clean"},
                "graph_binding": {
                    "ok": True,
                    "state": "verified",
                    "graph_hash": graph_hash,
                },
                "integrity": {"error_count": 0},
            },
        ]
        graph["claim_insights"] = [
            {"claim_id": claim_id, "depth_level": "causal"}
        ]
        closure_row = {
            "claim_id": claim_id,
            "state": "verified",
            "evidence_ids": [evidence_id],
            "experimental_evidence_ids": [evidence_id],
            "passage_evidence_ids": [],
            "source_snapshot_ids": [],
            "search_receipt_ids": [],
            "search_plan_ids": [],
            "argument_ids": [],
            "source_update_ids": [],
            "attempt_ids": [],
            "claim_attempt_ids": [],
            "study_evidence_ids": [],
            "attempt_states": {},
            "plan_ids": [],
            "preregistration_ids": [],
            "priority_ids": [],
            "gate_ids": [],
            "reproduction_ids": [],
            "reproduction_closure_ids": [evidence_id],
            "trace_complete": False,
            "replay_ready": False,
            "verified": False,
            "complete": False,
            "missing": ["missing_passing_gate"],
        }
        closure_levels = {
            level: {
                "complete": False,
                "status": "blocked",
                "claim_count": 1,
                "complete_claim_count": 0,
                "blocker_count": 1,
                "warning_count": 0,
                "blocker_codes": ["missing_passing_gate"],
                "warning_codes": [],
            }
            for level in ("trace", "replay", "verify")
        }
        closure = _seal_closure(
            {
                "schema_version": "xscientist.research-closure.v3",
                "ref": "8" * 40,
                "commit": "8" * 40,
                "target_level": "verify",
                "status": "blocked",
                "complete": False,
                "counts": {"claim": 1, "evidence": 1},
                "superseded_claim_ids": [],
                "claims": [closure_row],
                "blockers": [
                    {
                        "code": "missing_passing_gate",
                        "object_id": claim_id,
                        "message": "claim has no passing independent gate",
                    }
                ],
                "warnings": [],
                "closure_levels": closure_levels,
                "integrity": {"ok": True, "errors": [], "warnings": []},
                "trajectory": {
                    "ok": True,
                    "schema_version": "xscientist.structured-trajectory-projection.v1",
                    "projection_hash": "sha256:" + "7" * 64,
                    "resolved_head": "8" * 40,
                    "complete": True,
                    "truncated": False,
                    "checkpoint_count": 1,
                    "boundary_parent_edges": [],
                    "boundary_rollback_edges": [],
                    "errors": [],
                },
                "payloads_disclosed": False,
            }
        )
        graph["scientific_closure"] = {
            "status": "blocked",
            "claim_count": 1,
            "blocker_count": 1,
            "warning_count": 0,
            "closure_levels": closure_levels,
            "content_hash": closure["content_hash"],
        }
        _seal_graph(graph)
        return graph, objects, closure

    def test_clean_source_level_ara_binding_contributes_zero(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        view = build_research_argument_view(graph, object_rows=objects, closure=closure)
        claim = view["registered_claims"][0]
        self.assertEqual(claim["ara_claim_role_bindings"], [])
        self.assertEqual(
            view["counters"]["ara_claim_role_nodes"], {"bound": 0, "total": 2}
        )

    def test_v1_ara_role_binding_fails_closed_without_typed_authority(self) -> None:
        graph, objects, closure = self._claim_fixture()
        view = build_research_argument_view(graph, object_rows=objects, closure=closure)
        claim = view["registered_claims"][0]
        self.assertEqual(claim["direct_supporting_ids"], [objects[1]["object_id"]])
        self.assertEqual(claim["ara_claim_role_bindings"], [])
        self.assertEqual(claim["ara_binding_rejections"], [])
        self.assertEqual(
            view["counters"]["ara_claim_role_nodes"], {"bound": 0, "total": 2}
        )

    def test_tampered_ara_binding_cannot_bypass_object_identity(self) -> None:
        graph, objects, closure = self._claim_fixture()
        binding = objects[1]["payload"]["ara_claim_role_bindings"][0]
        binding["selected_node_ids"] = ["missing-leaf"]
        binding["selected_node_ids_hash"] = canonical_content_hash(["missing-leaf"])
        with self.assertRaisesRegex(
            ResearchArgumentError, "authoritative identity is invalid"
        ):
            build_research_argument_view(graph, object_rows=objects, closure=closure)

    def test_claim_insight_reachability_is_not_direct_support(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        graph["edges"][0]["category"] = "lineage"
        graph["edges"][0]["type"] = "depends_on"
        graph["claim_insights"] = [
            {
                "claim_id": objects[0]["object_id"],
                "supporting_ids": [objects[1]["object_id"]],
            }
        ]
        _seal_graph(graph)
        with self.assertRaisesRegex(ResearchArgumentError, "support edges disagree"):
            build_research_argument_view(graph, object_rows=objects, closure=closure)

    def test_graph_hash_tampering_is_rejected_before_projection(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        graph["edges"][0]["category"] = "lineage"

        with self.assertRaisesRegex(ResearchArgumentError, "graph_hash mismatch"):
            build_research_argument_view(graph, object_rows=objects, closure=closure)

    def test_research_object_identity_must_match_authoritative_node(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        objects[1]["content_hash"] = "sha256:" + "e" * 64

        with self.assertRaisesRegex(
            ResearchArgumentError, "authoritative identity is invalid"
        ):
            build_research_argument_view(graph, object_rows=objects, closure=closure)

    def test_support_category_without_support_relation_type_is_not_direct(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        graph["edges"][0]["type"] = "depends_on"
        _seal_graph(graph)

        with self.assertRaisesRegex(ResearchArgumentError, "support edges disagree"):
            build_research_argument_view(graph, object_rows=objects, closure=closure)

    def test_superseded_support_cannot_make_claim_decision_ready(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        old_support = objects[1]
        successor = build_research_object(
            kind="evidence",
            state="completed",
            payload={"result": "The earlier support is withdrawn."},
            relations=[
                {"type": "supersedes", "target": old_support["object_id"]}
            ],
            created_at="2026-09-14T00:02:00Z",
        )
        objects.append(successor)
        self._append_research_node(graph, successor, summary="Support withdrawn")
        graph["edges"].append(
            {
                "source": old_support["object_id"],
                "target": successor["object_id"],
                "type": "supersedes",
                "role": "",
                "category": "evolution",
            }
        )
        closure["counts"]["evidence"] = 2
        self._mark_closure_verified(graph, closure)

        view = build_research_argument_view(graph, object_rows=objects, closure=closure)

        claim = view["registered_claims"][0]
        self.assertEqual(claim["direct_supporting_ids"], [])
        self.assertFalse(claim["decision_ready"])
        self.assertIn("direct_semantic_support_missing", claim["gaps"])

    def test_support_for_superseded_claim_alias_binds_active_successor(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        predecessor = objects[0]
        successor = build_research_object(
            kind="claim",
            state="verified",
            payload={
                "statement": "The latent state predicts the intervention.",
                "depth_level": "causal",
            },
            relations=[
                {"type": "supersedes", "target": predecessor["object_id"]}
            ],
            created_at="2026-09-14T00:02:00Z",
        )
        objects.append(successor)
        self._append_research_node(
            graph,
            successor,
            summary="The latent state predicts the intervention.",
        )
        graph["edges"].append(
            {
                "source": predecessor["object_id"],
                "target": successor["object_id"],
                "type": "supersedes",
                "role": "",
                "category": "evolution",
            }
        )
        graph["claim_insights"] = [
            {"claim_id": successor["object_id"], "depth_level": "causal"}
        ]
        graph["scientific_closure"]["claim_count"] = 2
        closure["counts"]["claim"] = 2
        closure["superseded_claim_ids"] = [predecessor["object_id"]]
        row = closure["claims"][0]
        row["claim_id"] = successor["object_id"]
        row["state"] = "verified"
        closure["blockers"][0]["object_id"] = successor["object_id"]
        self._rebind_closure(graph, closure)

        view = build_research_argument_view(graph, object_rows=objects, closure=closure)

        self.assertEqual(len(view["registered_claims"]), 1)
        claim = view["registered_claims"][0]
        self.assertEqual(claim["claim_id"], successor["object_id"])
        self.assertEqual(claim["direct_supporting_ids"], [objects[1]["object_id"]])

    def test_failed_refutation_is_not_an_active_direct_challenge(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        claim_id = objects[0]["object_id"]
        refutation = build_research_object(
            kind="evidence",
            state="failed",
            payload={"result": "Refutation attempt failed."},
            relations=[{"type": "refutes", "target": claim_id}],
            created_at="2026-09-14T00:02:00Z",
        )
        objects.append(refutation)
        self._append_research_node(graph, refutation, summary="Failed refutation")
        graph["edges"].append(
            {
                "source": claim_id,
                "target": refutation["object_id"],
                "type": "refutes",
                "role": "",
                "category": "challenge",
            }
        )
        row = closure["claims"][0]
        row["evidence_ids"].append(refutation["object_id"])
        row["experimental_evidence_ids"].append(refutation["object_id"])
        row["reproduction_closure_ids"].append(refutation["object_id"])
        closure["counts"]["evidence"] = 2
        self._rebind_closure(graph, closure)

        view = build_research_argument_view(graph, object_rows=objects, closure=closure)

        claim = view["registered_claims"][0]
        self.assertEqual(claim["direct_challenging_ids"], [])
        self.assertNotIn("direct_semantic_challenge_active", claim["gaps"])

    def test_payload_tampering_is_rejected_even_when_identity_fields_are_retained(
        self,
    ) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        objects[0]["payload"]["statement"] = "Forged best-paper claim."

        with self.assertRaisesRegex(
            ResearchArgumentError, "authoritative identity is invalid"
        ):
            build_research_argument_view(graph, object_rows=objects, closure=closure)

    def test_closure_rehash_cannot_escape_the_dag_commitment(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        row = closure["claims"][0]
        row["verified"] = True
        row["complete"] = True
        row["missing"] = []
        closure_core = {
            key: value for key, value in closure.items() if key != "content_hash"
        }
        closure["content_hash"] = protocol_content_hash(closure_core)

        with self.assertRaisesRegex(ResearchArgumentError, "DAG commitment"):
            build_research_argument_view(graph, object_rows=objects, closure=closure)

    def test_closure_schema_commit_and_content_hash_are_checked(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        mutations = {
            "schema": ("schema_version", "xscientist.research-closure.v2", "schema"),
            "commit": ("commit", "9" * 40, "commit"),
            "content_hash": ("status", "complete", "content_hash"),
        }
        for label, (field, value, error) in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(closure)
                candidate[field] = value
                if label == "commit":
                    core = {
                        key: item
                        for key, item in candidate.items()
                        if key != "content_hash"
                    }
                    candidate["content_hash"] = protocol_content_hash(core)
                with self.assertRaisesRegex(ResearchArgumentError, error):
                    build_research_argument_view(
                        graph, object_rows=objects, closure=candidate
                    )

    def test_rehashed_sidecar_cannot_forge_authoritative_claim_fields(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        view = build_research_argument_view(graph, object_rows=objects, closure=closure)

        mutations = {
            "claim": lambda candidate: candidate["registered_claims"].append(
                copy.deepcopy(candidate["registered_claims"][0])
            ),
            "decision_ready": lambda candidate: candidate["registered_claims"][0].__setitem__(
                "decision_ready", True
            ),
            "ara_binding": lambda candidate: candidate["registered_claims"][0][
                "ara_claim_role_bindings"
            ].append({"role": "support"}),
            "statement": lambda candidate: candidate["registered_claims"][0].__setitem__(
                "statement", "Forged best-paper claim."
            ),
            "state": lambda candidate: candidate["registered_claims"][0].__setitem__(
                "state", "promoted"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(view)
                mutate(candidate)
                core = {
                    key: value
                    for key, value in candidate.items()
                    if key != "content_hash"
                }
                candidate["content_hash"] = canonical_content_hash(core)
                with self.assertRaisesRegex(
                    ResearchArgumentError, "canonical DAG projection"
                ):
                    validate_research_argument_view(graph, candidate)

    def test_provenance_category_is_counted_in_provenance_counter(self) -> None:
        graph = self._base_graph(
            edges=[
                {
                    "source": "source",
                    "target": "update",
                    "type": "updates",
                    "role": "",
                    "category": "provenance",
                }
            ]
        )

        view = build_research_argument_view(graph)

        self.assertEqual(
            view["counters"]["provenance_context_evolution_edges"],
            {"count": 1, "total": 1},
        )

    def test_requested_disclosure_cannot_upgrade_metadata_only_graph(self) -> None:
        graph, objects, closure = self._claim_fixture(include_role_binding=False)
        graph["content_disclosed"] = False
        _seal_graph(graph)

        view = build_research_argument_view(
            graph,
            object_rows=objects,
            closure=closure,
            disclose_summaries=True,
        )

        claim = view["registered_claims"][0]
        self.assertNotIn("statement", claim)
        self.assertNotIn("scope", claim)
        self.assertFalse(view["research_binding"]["content_disclosed"])


if __name__ == "__main__":
    unittest.main()
