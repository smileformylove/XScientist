from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist import ResearchRepository
from xscientist.research_argument import ResearchArgumentError
from xscientist.research_cli import main as research_main
from xscientist.research_dag import export_research_dag, render_research_dag_html


def _seal_graph(graph: dict) -> dict:
    core = {
        key: value
        for key, value in graph.items()
        if key not in {"generated_at", "graph_hash"}
    }
    graph["graph_hash"] = canonical_content_hash(core)
    return graph


def _research_binding(graph: dict) -> dict:
    return {
        "schema_version": graph["schema_version"],
        "ref": graph["ref"],
        "commit": graph["commit"],
        "graph_hash": graph["graph_hash"],
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "content_disclosed": graph["content_disclosed"],
    }


def _seal_argument(argument: dict) -> dict:
    core = {key: value for key, value in argument.items() if key != "content_hash"}
    argument["content_hash"] = canonical_content_hash(core)
    return argument


def _graph(*, claims: int = 0) -> dict:
    nodes = [
        {
            "id": f"rso-{index + 1:016x}",
            "source": "research_vcs",
            "kind": "claim",
            "state": "completed",
            "phase": 6,
            "layer": "theory",
            "summary": "claim",
            "content_hash": "sha256:" + "1" * 64,
            "actor": {},
            "created_at": "",
            "proof": {
                "level": "recorded",
                "checks": [],
                "blockers": [],
                "warnings": [],
                "contested": False,
            },
            "source_ref": "8" * 40,
        }
        for index in range(claims)
    ]
    return _seal_graph(
        {
            "schema_version": "xscientist.research-dag.v1",
            "ref": "HEAD",
            "commit": "8" * 40,
            "content_disclosed": True,
            "sources": [{"name": "research_vcs", "object_count": claims}],
            "nodes": nodes,
            "edges": [],
            "proof_summary": {"recorded": claims},
            "theory_frontier": {
                "historical_hypothesis_ids": [],
                "contested_hypothesis_ids": [],
                "open_theory_hypothesis_ids": [],
                "submission_candidate_pool": {
                    "candidate_count": 0,
                    "experiment_authorized": False,
                    "status": "reported",
                    "source_gate_authoritative": False,
                },
            },
            "claim_insights": [],
            "strategy_summary": {"open_anomaly_count": 0},
            "scientific_closure": {
                "status": "not_applicable" if not claims else "blocked",
                "claim_count": claims,
            },
            "integrity": {"is_dag": True, "topological_order": []},
            "generated_at": "2026-09-14T00:00:00+00:00",
        }
    )


def _embedded(page: str, element_id: str) -> dict:
    marker = f'<script id="{element_id}" type="application/json">'
    return json.loads(page.split(marker, 1)[1].split("</script>", 1)[0])


class ResearchArgumentHtmlTests(unittest.TestCase):
    def test_argument_option_lanes_counters_and_focusable_cards_are_present(
        self,
    ) -> None:
        graph = _graph()
        argument = _seal_argument(
            {
                "schema_version": "xscientist.research-argument-view.v1",
                "research_binding": _research_binding(graph),
                "advisory_index_binding": None,
                "registered_claims": [],
                "advisory_routes": [],
                "counters": {
                    "registered_claims": 0,
                    "unregistered_routes": 0,
                    "closure": "not_applicable",
                    "submission_candidates": 0,
                    "experiment_authorized": False,
                    "direct_semantic_edges": {"count": 0, "total": 0},
                    "verification_edges": {"count": 0, "total": 0},
                    "provenance_context_evolution_edges": {"count": 0, "total": 0},
                    "ara_claim_role_nodes": {"bound": 0, "total": 0},
                    "edge_categories": {},
                },
            }
        )
        page = render_research_dag_html(graph, argument_view=argument)

        self.assertIn('<option value="argument">Argument</option>', page)
        self.assertIn('id="argument-data"', page)
        self.assertNotIn('<link rel="icon" href="data:,"', page)
        self.assertIn("REGISTERED CLAIM CLOSURE", page)
        self.assertIn("UNREGISTERED SPRINT FRONTIER", page)
        self.assertIn("direct semantic edges", page)
        self.assertIn("ARA exact claim-role nodes", page)
        self.assertIn("NO REGISTERED CLAIMS", page)
        self.assertIn('class="argument-card" tabindex="0"', page)
        self.assertIn("view.value==='argument'", page)
        self.assertEqual(_embedded(page, "argument-data"), argument)

    def test_direct_renderer_with_claims_fails_closed_without_projection(self) -> None:
        page = render_research_dag_html(_graph(claims=1))
        argument = _embedded(page, "argument-data")

        self.assertEqual(argument["projection_state"], "UNAVAILABLE")
        self.assertIsNone(argument["registered_claims"])
        self.assertEqual(argument["counters"]["registered_claims"], 1)
        self.assertIn("ARGUMENT PROJECTION UNAVAILABLE", page)

    def test_advisory_xss_is_encoded_and_never_enters_dag_payload(self) -> None:
        graph = _graph()
        attack = '</script><img src=x onerror="globalThis.pwned=1">'
        argument = _seal_argument(
            {
                "schema_version": "xscientist.research-argument-view.v1",
                "research_binding": _research_binding(graph),
                "scientific_closure_authority": None,
                "summaries_disclosed": True,
                "advisory_index_binding": {"path": "index.json"},
                "registered_claims": [],
                "advisory_routes": [
                    {
                        "route_id": "route-xss",
                        "question": attack,
                        "authority": "UNREGISTERED",
                        "artifact_bindings": [],
                        "gate_contract_binding": {},
                        "gate_results": [],
                        "open_findings": [],
                        "novelty_disposition": attack,
                        "registration_state": "absent",
                        "permitted_next_action": attack,
                        "binding_state": "VERIFIED",
                        "route_state": "ADVISORY",
                    }
                ],
                "counters": {
                    "registered_claims": 0,
                    "unregistered_routes": 1,
                    "closure": "not_applicable",
                    "submission_candidates": 0,
                    "experiment_authorized": False,
                    "direct_semantic_edges": {"count": 0, "total": 0},
                    "verification_edges": {"count": 0, "total": 0},
                    "provenance_context_evolution_edges": {"count": 0, "total": 0},
                    "ara_claim_role_nodes": {"bound": 0, "total": 0},
                    "edge_categories": {},
                },
            }
        )
        page = render_research_dag_html(graph, argument_view=argument)
        dag_payload = page.split('<script id="dag-data" type="application/json">', 1)[
            1
        ].split("</script>", 1)[0]
        argument_payload = page.split(
            '<script id="argument-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]

        self.assertNotIn(attack, page)
        self.assertIn(r"\u003c/script\u003e", argument_payload)
        self.assertNotIn("route-xss", dag_payload)
        self.assertEqual(json.loads(dag_payload), graph)
        self.assertEqual(json.loads(argument_payload), argument)
        self.assertIn("const esc=s=>", page)
        self.assertIn('id="argument-view" class="advisory-only"', page)
        self.assertIn(
            "#argument-view.advisory-only .argument-lanes{grid-template-columns:minmax(0,1fr)}",
            page,
        )

    def test_renderer_rejects_tampered_sidecar_hash(self) -> None:
        graph = _graph()
        argument = _seal_argument(
            {
                "schema_version": "xscientist.research-argument-view.v1",
                "research_binding": _research_binding(graph),
                "advisory_index_binding": None,
                "registered_claims": [],
                "advisory_routes": [],
                "counters": {},
            }
        )
        argument["advisory_routes"].append({"route_state": "ADVISORY"})

        with self.assertRaisesRegex(ResearchArgumentError, "content_hash mismatch"):
            render_research_dag_html(graph, argument_view=argument)

    def test_renderer_rejects_rehashed_sidecar_bound_to_other_graph(self) -> None:
        graph = _graph()
        argument = _seal_argument(
            {
                "schema_version": "xscientist.research-argument-view.v1",
                "research_binding": _research_binding(graph),
                "advisory_index_binding": None,
                "registered_claims": [],
                "advisory_routes": [],
                "counters": {},
            }
        )
        argument["research_binding"]["node_count"] = 1
        _seal_argument(argument)

        with self.assertRaisesRegex(ResearchArgumentError, "DAG binding mismatch"):
            render_research_dag_html(graph, argument_view=argument)

    def test_renderer_disclosure_requires_request_and_disclosed_graph(self) -> None:
        secret = "PRIVATE-ARGUMENT-AND-DAG-BODY"
        for graph_disclosed, requested in ((False, True), (True, False)):
            with self.subTest(
                graph_disclosed=graph_disclosed, requested=requested
            ):
                graph = _graph()
                graph["content_disclosed"] = graph_disclosed
                graph["nodes"] = [
                    {
                        "id": "ara:0:node-a",
                        "source": "ara",
                        "kind": "experiment_node",
                        "state": "completed",
                        "phase": 3,
                        "layer": "execution",
                        "summary": secret,
                        "content_hash": "sha256:" + "2" * 64,
                        "actor": None,
                        "created_at": None,
                        "proof": {
                            "level": "recorded",
                            "checks": [],
                            "blockers": [],
                            "warnings": [],
                            "contested": False,
                        },
                        "source_ref": "ara:0",
                    }
                ]
                graph["edges"] = [
                    {
                        "source": "ara:0:node-a",
                        "target": "ara:0:node-a",
                        "type": "refutes",
                        "role": secret,
                        "category": "challenge",
                    }
                ]
                next_experiment = {
                    "portfolio_id": "portfolio-1",
                    "priority_object_id": "priority-1",
                    "candidate_id": "candidate-1",
                    "design_object_id": "design-1",
                    "summary": secret,
                    "expected_information_gain": 0.5,
                    "utility_score": 0.5,
                }
                graph["theory_frontier"].update(
                    {
                        "open_questions": [{"message": secret}],
                        "next_experiments": [next_experiment],
                        "next_experiment": next_experiment,
                        "portfolio_frontiers": [
                            {
                                "portfolio_id": "portfolio-1",
                                "hypothesis_ids": [],
                                "posterior_update_id": None,
                                "next_experiment": next_experiment,
                            }
                        ],
                    }
                )
                _seal_graph(graph)
                argument = _seal_argument(
                    {
                        "schema_version": "xscientist.research-argument-view.v1",
                        "research_binding": _research_binding(graph),
                        "scientific_closure_authority": None,
                        "summaries_disclosed": graph_disclosed,
                        "advisory_index_binding": None,
                        "registered_claims": [],
                        "advisory_routes": [
                            {"route_id": "route-1", "question": secret}
                        ],
                        "counters": {
                            "registered_claims": 0,
                            "unregistered_routes": 1,
                            "closure": "not_applicable",
                            "submission_candidates": 0,
                            "experiment_authorized": False,
                            "direct_semantic_edges": {"count": 1, "total": 1},
                            "verification_edges": {"count": 0, "total": 1},
                            "provenance_context_evolution_edges": {
                                "count": 0,
                                "total": 1,
                            },
                            "ara_claim_role_nodes": {"bound": 0, "total": 1},
                            "edge_categories": {"challenge": 1},
                        },
                    }
                )

                page = render_research_dag_html(
                    graph,
                    node_details={
                        "ara:0:node-a": {
                            "fields": {"analysis": secret}
                        }
                    },
                    argument_view=argument,
                    disclose_summaries=requested,
                )
                rendered_graph = _embedded(page, "dag-data")
                rendered_argument = _embedded(page, "argument-data")

                self.assertNotIn(secret, page)
                self.assertFalse(rendered_graph["content_disclosed"])
                self.assertEqual(rendered_graph["edges"][0]["role"], "")
                self.assertEqual(
                    rendered_graph["theory_frontier"]["open_questions"], []
                )
                self.assertEqual(
                    rendered_graph["theory_frontier"]["next_experiment"][
                        "summary"
                    ],
                    "",
                )
                self.assertEqual(_embedded(page, "detail-data"), {})
                self.assertEqual(rendered_argument["registered_claims"], [])
                self.assertEqual(rendered_argument["advisory_routes"], [])


@unittest.skipUnless(shutil.which("git"), "Git is required")
class ResearchArgumentExportTests(unittest.TestCase):
    def test_export_writes_sidecar_and_embeds_the_identical_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            ResearchRepository.init(
                root,
                question="Which latent world-model claim survives?",
                git_user_name="Argument Export Test",
                git_user_email="argument-export@example.invalid",
            )
            exported = export_research_dag(root, root / "analysis" / "dag")
            sidecar = json.loads(Path(exported["argument"]).read_text(encoding="utf-8"))
            page = Path(exported["html"]).read_text(encoding="utf-8")
            dag = json.loads(Path(exported["json"]).read_text(encoding="utf-8"))

            self.assertEqual(_embedded(page, "argument-data"), sidecar)
            self.assertEqual(
                sidecar["research_binding"]["graph_hash"], dag["graph_hash"]
            )
            self.assertEqual(sidecar["registered_claims"], [])
            self.assertNotIn("registered_claims", dag)
            self.assertNotIn("advisory_routes", dag)

    def test_cli_requires_output_for_explicit_advisory_index(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            status = research_main(
                ["dag", "--repo", ".", "--advisory-index", "advisory.json"]
            )
        self.assertEqual(status, 2)
        self.assertIn("--advisory-index requires --output", errors.getvalue())

    def test_cli_reports_blocked_advisory_without_failing_dag_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            ResearchRepository.init(
                root,
                question="Which latent world-model claim survives?",
                git_user_name="Argument CLI Test",
                git_user_email="argument-cli@example.invalid",
            )
            advisory = root / "advisory"
            advisory.mkdir()
            witness = advisory / "witness.md"
            contract = advisory / "contract.json"
            witness.write_bytes(b"bound witness\n")
            contract.write_bytes(b'{"gate":"W0"}\n')
            index = {
                "schema_version": "xscientist.advisory-route-index.v1",
                "routes": [
                    {
                        "route_id": "Q5-XM",
                        "question": "Does the witness survive?",
                        "authority": "UNREGISTERED",
                        "artifact_bindings": [
                            {
                                "path": "advisory/witness.md",
                                "raw_sha256": hashlib.sha256(
                                    witness.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                        "gate_contract_binding": {
                            "path": "advisory/contract.json",
                            "raw_sha256": hashlib.sha256(
                                contract.read_bytes()
                            ).hexdigest(),
                        },
                        "gate_results": [],
                        "open_findings": [],
                        "novelty_disposition": "OPEN",
                        "registration_state": "absent",
                        "permitted_next_action": "Repair the byte binding.",
                    }
                ],
            }
            (root / "advisory-index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            witness.write_bytes(b"tampered witness\n")

            human = io.StringIO()
            with redirect_stdout(human):
                human_status = research_main(
                    [
                        "dag",
                        "--repo",
                        str(root),
                        "--output",
                        str(root / "dag-human"),
                        "--advisory-index",
                        "advisory-index.json",
                    ]
                )
            self.assertEqual(human_status, 0)
            self.assertIn("advisory_status=BLOCKED", human.getvalue())
            self.assertIn("blocked_count=1", human.getvalue())

            machine = io.StringIO()
            with redirect_stdout(machine):
                machine_status = research_main(
                    [
                        "dag",
                        "--repo",
                        str(root),
                        "--output",
                        str(root / "dag-json"),
                        "--advisory-index",
                        "advisory-index.json",
                        "--json",
                    ]
                )
            report = json.loads(machine.getvalue())
            self.assertEqual(machine_status, 0)
            self.assertEqual(report["advisory_status"], "BLOCKED")
            self.assertEqual(report["advisory_blocked_count"], 1)
            self.assertTrue(report["graph"]["integrity"]["is_dag"])


if __name__ == "__main__":
    unittest.main()
