from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.node_test_support import NODE_EXECUTABLE
from xscientist import ResearchRepository
from xscientist.research_dag import (
    _RESEARCH_DAG_SEARCH_HELPERS_JS,
    _build_dag_search_index,
    _gate_detail_views,
    _strategy_projection,
    build_research_dag,
    export_research_dag,
    render_research_dag_html,
)


@unittest.skipUnless(shutil.which("git"), "Git is required")
class ResearchDagVisibilityTests(unittest.TestCase):
    def test_terminal_attempt_runtime_bindings_are_visible_and_bounded(self) -> None:
        bindings = {
            "admission_runtime_claim_hash": "sha256:" + "1" * 64,
            "admission_bfts_terminal_hash": "sha256:" + "2" * 64,
            "admission_worker_terminal_hash": "sha256:" + "3" * 64,
            "terminal_receipt_hash": "sha256:" + "4" * 64,
        }
        objects = [{
            "object_id": "attempt",
            "kind": "experiment_attempt",
            "state": "completed",
            "payload": {**bindings, "private_note": "NOT_DISPLAYED"},
        }]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            details = _gate_detail_views(
                objects,
                repository_root=root,
                html_directory=root,
                snapshot_commit="a" * 40,
                disclose=True,
            )
        self.assertEqual(details["attempt"]["fields"], bindings)
        self.assertEqual(details["attempt"]["artifacts"], [])
        self.assertNotIn("NOT_DISPLAYED", json.dumps(details))

    def test_advisory_evidence_assessment_preserves_denials_and_zero_counts(self) -> None:
        payload = {
            "status": "question_only_novelty_hold",
            "authority": "local_advisory_only",
            "scope": "Uniform reset triples only.",
            "result": "A scoped internally reviewed theorem.",
            "limits": ["Not an empirical world-model result."],
            "novelty_cleared": False,
            "claim_promotion_allowed": False,
            "experiment_authorized": False,
            "training_authorized": False,
            "candidate_count": 0,
            "registered_claim_count": 0,
            "best_paper_evidence": False,
            "external_peer_review": False,
            "mathematical_machine_proof": False,
        }
        objects = [{"object_id": "evidence", "kind": "evidence_synthesis",
                    "state": "completed", "payload": {**payload, "private_note": "NOT_DISPLAYED"}}]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            details = _gate_detail_views(
                objects, repository_root=root, html_directory=root,
                snapshot_commit="a" * 40, disclose=True,
            )
            hidden = _gate_detail_views(
                objects, repository_root=root, html_directory=root,
                snapshot_commit="a" * 40, disclose=False,
            )
        self.assertEqual(details["evidence"]["fields"], payload)
        self.assertEqual(details["evidence"]["title"], "Recorded evidence assessment (advisory)")
        self.assertEqual(details["evidence"]["artifacts"], [])
        self.assertEqual(objects[0]["state"], "completed")
        self.assertNotIn("NOT_DISPLAYED", json.dumps(details))
        self.assertEqual(hidden, {})

    def test_direct_evidence_artifacts_are_visible_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = ResearchRepository.init(
                root,
                question="Can direct evidence bindings be inspected?",
                git_user_name="DAG Visibility Test",
                git_user_email="dag-visibility@example.invalid",
            )
            artifact = root / "README.md"
            artifact.write_text("direct artifact binding fixture\n", encoding="utf-8")
            content_hash = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
            evidence = repository.record(
                "evidence_synthesis",
                {
                    "summary": "A directly inspectable evidence packet.",
                    "status": "question_only_novelty_hold",
                    "claim_promotion_allowed": False,
                    "experiment_authorized": False,
                    "artifacts": [
                        {"path": "README.md", "content_hash": content_hash}
                    ],
                },
                state="completed",
            )
            object_path = str(
                Path(evidence.path).resolve().relative_to(root.resolve())
            )
            repository.stage(["README.md", object_path])
            repository.commit(
                stage="theory",
                subject="record direct artifact binding",
            )

            exported = export_research_dag(root, root / "05_analysis" / "dag")
            page = Path(exported["html"]).read_text(encoding="utf-8")
            detail_json = page.split(
                '<script id="detail-data" type="application/json">', 1
            )[1].split("</script>", 1)[0]
            details = json.loads(detail_json)
            artifacts = details[evidence.object_id]["artifacts"]
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]["logical_path"], "README.md")
            self.assertEqual(artifacts[0]["content_hash"], content_hash)
            self.assertIs(artifacts[0]["binding_verified"], True)
            self.assertEqual(details[evidence.object_id]["fields"]["status"], "question_only_novelty_hold")
            self.assertIs(details[evidence.object_id]["fields"]["experiment_authorized"], False)
            self.assertIs(details[evidence.object_id]["fields"]["claim_promotion_allowed"], False)

    def test_metadata_strategy_projection_keeps_shape_without_derived_body(self) -> None:
        secret = "PRIVATE-FRNM-NEXT-EXPERIMENT-BODY"
        objects = {
            "portfolio": {
                "object_id": "portfolio",
                "kind": "hypothesis_portfolio",
                "state": "locked",
                "payload": {
                    "members": [{"hypothesis_id": "hypothesis"}],
                },
                "relations": [],
            },
            "priority": {
                "object_id": "priority",
                "kind": "experiment_priority",
                "state": "locked",
                "payload": {
                    "portfolio_id": "portfolio",
                    "selected_candidate_id": "candidate-1",
                    "candidate_set": [
                        {
                            "candidate_id": "candidate-1",
                            "design_object_id": "design-1",
                            "summary": secret,
                            "expected_information_gain": 0.5,
                            "utility_score": 0.5,
                        }
                    ],
                },
                "relations": [],
            },
            "review": {
                "object_id": "review",
                "kind": "research_review",
                "state": "completed",
                "payload": {
                    "gaps": [{"code": "open", "message": secret}],
                    "review_due": True,
                },
                "relations": [],
            },
            "boundary": {
                "object_id": "boundary",
                "kind": "boundary_condition",
                "state": "draft",
                "payload": {
                    "status": "untested",
                    "dimension": "rank",
                    "condition": secret,
                },
                "relations": [],
            },
            "gate": {
                "object_id": "gate",
                "kind": "gate_decision",
                "state": "completed",
                "actor": {"actor_id": "gate", "authority": "deterministic_gate"},
                "payload": {
                    "candidate_count": 1,
                    "evaluated_candidate_ids": ["candidate-1"],
                    "experiment_authorized": False,
                    "decision": secret,
                    "status": secret,
                },
                "relations": [],
            },
        }
        introduction_order = {
            object_id: index for index, object_id in enumerate(objects, start=1)
        }

        metadata, _claim_insights, _summary = _strategy_projection(
            objects, introduction_order, disclose=False
        )
        disclosed, _claim_insights, _summary = _strategy_projection(
            objects, introduction_order, disclose=True
        )

        self.assertNotIn(secret, json.dumps(metadata))
        self.assertIn(secret, json.dumps(disclosed))
        self.assertEqual(metadata["open_questions"], [])
        self.assertEqual(metadata["next_experiments"][0]["summary"], "")
        self.assertEqual(
            metadata["portfolio_frontiers"][0]["next_experiment"]["summary"],
            "",
        )
        self.assertIsNone(metadata["submission_candidate_pool"]["decision"])
        self.assertIsNone(metadata["submission_candidate_pool"]["gate_status"])

    def test_theory_history_is_separate_from_recorded_submission_pool(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = ResearchRepository.init(
                root,
                question="Which latent world-model claim survives?",
                git_user_name="DAG Visibility Test",
                git_user_email="dag-visibility@example.invalid",
            )
            open_theory = repository.record(
                "hypothesis",
                {
                    "statement": "An open theory remains to be reduced.",
                    "proposal_id": "C22-CSE",
                },
            )
            contested = repository.record(
                "hypothesis",
                {"statement": "A prior candidate failed its theorem gate."},
            )
            gate = repository.record(
                "gate_decision",
                {
                    "decision": "no_candidate",
                    "status": "failed_submission_direction_gate",
                    "candidate_count": 0,
                    "evaluated_candidate_ids": ["C6", "C7", "C8", "C9"],
                    "experiment_authorized": False,
                    "claim_promotion_allowed": False,
                    "authority": "local_advisory_only",
                    "ara_created": False,
                    "hypothesis_id": contested.object_id,
                    "protocol_kind": "xscientist.theorem-reduction-gate.v1",
                    "reentry_requirements": ["Open a new candidate contract."],
                    "verdict_vector": [
                        {
                            "criterion_id": "source_separated_theorem_delta",
                            "required": True,
                            "passed": False,
                        }
                    ],
                },
                state="rejected",
                relations=[{"type": "rejects", "target": contested.object_id}],
                actor={
                    "actor_id": "xscientist",
                    "authority": "research_agent",
                },
            )
            repository.commit(
                stage="decision", subject="record the submission candidate gate"
            )

            graph = build_research_dag(repository.path)
            frontier = graph["theory_frontier"]

            self.assertEqual(
                frontier["historical_hypothesis_ids"],
                sorted([open_theory.object_id, contested.object_id]),
            )
            self.assertEqual(
                frontier["contested_hypothesis_ids"], [contested.object_id]
            )
            self.assertEqual(
                frontier["open_theory_hypothesis_ids"], [open_theory.object_id]
            )
            self.assertEqual(frontier["active_hypothesis_ids"], [open_theory.object_id])
            pool = frontier["submission_candidate_pool"]
            self.assertEqual(pool["candidate_count"], 0)
            self.assertEqual(pool["evaluated_candidate_ids"], ["C6", "C7", "C8", "C9"])
            self.assertIs(pool["experiment_authorized"], False)
            self.assertEqual(pool["source_gate_id"], gate.object_id)
            self.assertEqual(pool["source_gate_authority"], "research_agent")
            self.assertIs(pool["source_gate_authoritative"], False)

            gate_node = next(
                node for node in graph["nodes"] if node["id"] == gate.object_id
            )
            self.assertIn("candidate_count=0", gate_node["summary"])
            self.assertIn(
                'evaluated_candidate_ids=["C6","C7","C8","C9"]',
                gate_node["summary"],
            )
            self.assertIn("experiment_authorized=false", gate_node["summary"])

            page = render_research_dag_html(graph)
            self.assertIn("hypothesis history", page)
            self.assertIn("contested hypotheses", page)
            self.assertIn("submission candidates", page)
            self.assertIn("contested theory history", page)
            self.assertIn('data-presentation-only="true"', page)
            self.assertIn("Rendering legend", page)
            self.assertIn("challenge / contradiction", page)
            self.assertIn("do not add scientific evidence", page)

            exported = export_research_dag(
                repository.path, root / "05_analysis" / "dag"
            )
            exported_page = Path(exported["html"]).read_text(encoding="utf-8")
            detail_json = exported_page.split(
                '<script id="detail-data" type="application/json">', 1
            )[1].split("</script>", 1)[0]
            search_json = exported_page.split(
                '<script id="search-data" type="application/json">', 1
            )[1].split("</script>", 1)[0]
            self.assertIn(
                "c22-cse", json.loads(search_json)[open_theory.object_id].lower()
            )
            gate_fields = json.loads(detail_json)[gate.object_id]["gate"]
            self.assertEqual(gate_fields["candidate_count"], 0)
            self.assertEqual(
                gate_fields["evaluated_candidate_ids"], ["C6", "C7", "C8", "C9"]
            )
            self.assertIs(gate_fields["experiment_authorized"], False)
            self.assertEqual(gate_fields["authority"], "local_advisory_only")
            self.assertIs(gate_fields["ara_created"], False)
            self.assertEqual(gate_fields["hypothesis_id"], contested.object_id)
            self.assertEqual(
                gate_fields["reentry_requirements"],
                ["Open a new candidate contract."],
            )
            self.assertEqual(
                gate_fields["verdict_vector"][0]["criterion_id"],
                "source_separated_theorem_delta",
            )
            self.assertIn("node-jump", exported_page)
            self.assertIn('id="search-data"', exported_page)
            self.assertIn("search-seed", exported_page)
            self.assertIn("search-context", exported_page)
            self.assertIn("source-integrity", exported_page)
            self.assertIn("observability-binding", exported_page)
            self.assertIn("observability-binding-data", exported_page)
            self.assertIn("binding_hash_scope", exported_page)
            self.assertIn("Hashes identify recorded source lineage", exported_page)
            self.assertIn('id="view"', exported_page)
            self.assertIn('id="source"', exported_page)
            self.assertIn('id="relation"', exported_page)
            self.assertIn("e.category||'unknown'", exported_page)
            self.assertIn("Full audit", exported_page)
            self.assertIn("UI aggregate only.", exported_page)
            self.assertIn("authoritativeEdgesForNodes(nodes,allEdges)", exported_page)
            self.assertIn("recorded ${allNodes.length} nodes", exported_page)
            self.assertIn("exact Research Object links", exported_page)
            self.assertIn("Actor authority", exported_page)
            self.assertIn("oneHopIds(seedIds,allEdges)", exported_page)
            self.assertIn("filters apply to direct matches", exported_page)
            self.assertIn('class="control source-control"', exported_page)
            self.assertIn('id="source-integrity-disclosure"', exported_page)
            self.assertNotIn('id="source-integrity-disclosure" open', exported_page)
            self.assertIn(
                ".source-integrity-disclosure>summary{display:list-item",
                exported_page,
            )
            self.assertNotIn(
                ".source-integrity-disclosure>summary{display:none", exported_page
            )
            self.assertIn(
                "sourceIntegrityDisclosure.open=sourceIntegrityHasIssues", exported_page
            )
            self.assertIn("fitSvgLabel(el,boxW-24)", exported_page)
            self.assertIn('class="muted node-summary"', exported_page)
            self.assertNotIn("JSON.stringify(detailViews[n.id]", exported_page)

    def test_search_index_excludes_opaque_provenance_but_keeps_semantics(self) -> None:
        node_id = "rso-4d5be45efe86dc16"
        digest = "sha256:" + "a" * 64
        truncated_hash = "deadbeefc016"
        nodes = [
            {
                "id": node_id,
                "kind": "gate_decision",
                "state": "rejected",
                "layer": "decision_memory",
                "summary": (
                    "The bounded latent theorem remains unresolved at snapshot "
                    f"{truncated_hash}."
                ),
                "actor": {
                    "actor_id": "xscientist:c17_theorem_search",
                    "authority": "local_advisory_only",
                },
            }
        ]
        details = {
            node_id: {
                "gate": {
                    "candidate_id": "C17",
                    "proposal_id": "C22-CSE",
                    "plan_id": "rso-bbbbbbbbbbbbbc16",
                    "decision": "no_candidate",
                    "input_bindings": {
                        "source_id": "rso-aaaaaaaaaaaaac16",
                        "artifact_path": (
                            "07_audits/20260905_c15_c17_hostile_theorem_review.md"
                        ),
                        "content_hash": digest,
                    },
                    "gate_hash": digest,
                },
                "lineage": [
                    {
                        "source_id": "rso-deadbeefc016",
                        "source_kind": "experiment_attempt",
                        "fields": {
                            "summary": "Evidence supports a bounded theorem.",
                            "reference": "snapshot deadbeefc017",
                        },
                    }
                ],
                "artifacts": [
                    {
                        "label": "hostile theorem audit",
                        "logical_path": "07_audits/deadbeefc018_c15_review.md",
                        "content_hash": digest,
                        "source_id": "rso-cccccccccccccc16",
                        "source_kind": "gate_decision",
                    }
                ],
            }
        }

        index = _build_dag_search_index(nodes, details)[node_id].lower()

        self.assertNotIn(node_id, index)
        self.assertNotIn("rso-aaaaaaaaaaaaac16", index)
        self.assertNotIn("rso-bbbbbbbbbbbbbc16", index)
        self.assertNotIn("rso-deadbeefc016", index)
        self.assertNotIn("rso-cccccccccccccc16", index)
        self.assertNotIn("07_audits", index)
        self.assertNotIn("20260905_c15_c17", index)
        self.assertNotIn(truncated_hash, index)
        self.assertNotIn("deadbeefc017", index)
        self.assertNotIn("deadbeefc018", index)
        self.assertNotIn(digest, index)
        self.assertIn("xscientist:c17_theorem_search", index)
        self.assertIn("local_advisory_only", index)
        self.assertIn("bounded latent theorem", index)
        self.assertIn("c17", index)
        self.assertIn("c22-cse", index)
        self.assertIn(
            "c15",
            _build_dag_search_index(
                [{**nodes[0], "summary": "Candidate C15 remains viable."}], details
            )[node_id].lower(),
        )

    def test_search_helpers_use_exact_ids_and_strict_one_hop_context(self) -> None:
        if NODE_EXECUTABLE is None:
            self.skipTest("Node.js is required for browser helper tests")
        script = _RESEARCH_DAG_SEARCH_HELPERS_JS + r"""
const assert = require('node:assert/strict');
const nodeId = 'rso-4d5be45efe86dc16';
const graphNode = {id: nodeId};
const index = {
  [nodeId]: 'gate decision bounded theorem xscientist:c17_theorem_search'
};
assert.equal(isDirectMatch(graphNode, nodeId, index), true);
assert.equal(isDirectMatch(graphNode, 'c16', index), false);
assert.equal(isDirectMatch(graphNode, 'deadbeefc016', index), false);
assert.equal(isDirectMatch(graphNode, 'rso-deadbeefc016', index), false);
assert.equal(isDirectMatch(graphNode, 'xscientist:c17_theorem_search', index), true);
assert.equal(isDirectMatch(graphNode, '07_audits/20260905_c15_c17', index), false);

const edges = [
  {source: 'a', target: 'b'},
  {source: 'b', target: 'c'},
  {source: 'c', target: 'd'}
];
const seed = new Set(['b']);
const sorted = values => [...values].sort();
assert.deepEqual(sorted(oneHopIds(seed, edges)), ['a', 'b', 'c']);
assert.deepEqual(sorted(oneHopIds(seed, [...edges].reverse())), ['a', 'b', 'c']);
assert.equal(oneHopIds(seed, edges).has('d'), false);
assert.equal(sourceNameForNode({source: 'ara', id: 'ara:3:n1'}), 'ara:3');
assert.deepEqual(sourceBindingSummary([
  {name: 'research_vcs', object_count: 80},
  {name: 'ara:0', object_count: 25, binding_count: 1},
  {name: 'ara:1', object_count: 36, binding_count: 0}
]), {
  araSourceCount: 2,
  exactLinkCount: 1,
  sourcesWithExactLinks: 1,
  sourcesWithoutExactLinks: 1,
  nodesWithoutExactLinks: 36
});

const wideGateSummary = {
  value: 'REJECT_Q1_AXIS_AND_Q3_UNIVERSALITY…',
  get textContent() { return this.value; },
  set textContent(value) { this.value = String(value); },
  getComputedTextLength() {
    return [...this.value].reduce(
      (width, character) => width + (/[A-Z_]/.test(character) ? 8 : 6), 0
    );
  }
};
const originalWideSummary = wideGateSummary.textContent;
const fittedWideSummary = fitSvgLabel(wideGateSummary, 196);
assert.ok(wideGateSummary.getComputedTextLength() <= 196);
assert.ok(fittedWideSummary.endsWith('…'));
assert.ok(fittedWideSummary.length < originalWideSummary.length);

const shortSummary = {
  value: 'bounded summary',
  get textContent() { return this.value; },
  set textContent(value) { this.value = String(value); },
  getComputedTextLength() { return this.value.length * 6; }
};
assert.equal(fitSvgLabel(shortSummary, 196), 'bounded summary');
assert.equal(shortSummary.textContent, 'bounded summary');
"""
        result = subprocess.run(
            [NODE_EXECUTABLE, "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_view_helpers_collapse_ara_snapshots_without_synthetic_edges(self) -> None:
        if NODE_EXECUTABLE is None:
            self.skipTest("Node.js is required for browser helper tests")
        script = _RESEARCH_DAG_SEARCH_HELPERS_JS + r"""
const assert = require('node:assert/strict');
const sources = [
  {name: 'research_vcs', object_count: 1},
  {
    name: 'ara:0', object_count: 2, binding_count: 1, context_binding_count: 0,
    repository_path: 'ara/20260903-full-v1/manifest.json', snapshot_ref: 'a'.repeat(40),
    manifest_integrity: {state: 'clean'}, graph_binding: {state: 'verified'},
    integrity: {error_count: 0, warning_count: 0}, bound_object_ids: ['rso-main']
  },
  {
    name: 'ara:1', object_count: 1, binding_count: 0, context_binding_count: 0,
    repository_path: 'ara/20260903-pilot-v0/manifest.json', snapshot_ref: 'a'.repeat(40),
    manifest_integrity: {state: 'clean'}, graph_binding: {state: 'verified'},
    integrity: {error_count: 0, warning_count: 0}, bound_object_ids: []
  }
];
const nodes = [
  {id: 'rso-main', source: 'research_vcs'},
  {id: 'ara:0:root', source: 'ara'},
  {id: 'ara:0:leaf', source: 'ara'},
  {id: 'ara:1:only', source: 'ara'}
];
const edges = [
  {source: 'ara:0:root', target: 'ara:0:leaf', type: 'evolves_to'},
  {source: 'ara:0:leaf', target: 'rso-main', type: 'anchors'}
];

const overview = projectDagView(nodes, sources, 'overview', 'all', '');
assert.deepEqual(
  overview.nodes.map(node => node.id),
  ['rso-main', 'ui:ara-source:ara:0', 'ui:ara-source:ara:1']
);
assert.equal(overview.aggregateCount, 2);
assert.equal(overview.collapsedNodeCount, 3);
assert.deepEqual(authoritativeEdgesForNodes(overview.nodes, edges), []);
const unlinked = overview.nodes.find(node => node.id === 'ui:ara-source:ara:1');
assert.equal(unlinked.ui_aggregate, true);
assert.equal(unlinked.aggregate.no_exact_links, true);
assert.equal(unlinked.state, 'no_exact_research_object_link');
assert.equal(sourceNameForNode(unlinked), 'ara:1');
assert.equal(sourceDisplayName(sources[2]), '20260903-pilot-v0');
assert.equal(isDirectMatch(unlinked, 'pilot-v0', {}), true);

const full = projectDagView(nodes, sources, 'full', 'all', '');
assert.deepEqual(full.nodes, nodes);
assert.equal(full.aggregateCount, 0);
assert.equal(full.collapsedNodeCount, 0);
assert.deepEqual(authoritativeEdgesForNodes(full.nodes, edges), edges);

const researchVcs = projectDagView(nodes, sources, 'overview', 'research_vcs', '');
assert.deepEqual(researchVcs.nodes.map(node => node.id), ['rso-main']);
const selectedAra = projectDagView(nodes, sources, 'overview', 'ara:0', '');
assert.deepEqual(
  selectedAra.nodes.map(node => node.id), ['ara:0:root', 'ara:0:leaf']
);

const exactAra = projectDagView(nodes, sources, 'overview', 'all', 'ARA:0:LEAF');
assert.equal(exactAra.expandedSourceName, 'ara:0');
assert.equal(exactAra.aggregateCount, 1);
assert.equal(exactAra.collapsedNodeCount, 1);
assert.deepEqual(
  exactAra.nodes.map(node => node.id),
  ['rso-main', 'ara:0:root', 'ara:0:leaf', 'ui:ara-source:ara:1']
);

// Semantic search must expand a collapsed source when a hidden ARA node matches.
const semanticIndex = {
  'ara:1:only': 'experiment_node pilot bounded candidate theorem'
};
const semanticAra = projectDagView(
  nodes, sources, 'overview', 'all', 'bounded candidate', semanticIndex
);
assert.equal(semanticAra.expandedSourceName, 'ara:1');
assert.deepEqual(
  semanticAra.nodes.map(node => node.id),
  ['rso-main', 'ara:1:only', 'ui:ara-source:ara:0']
);
"""
        result = subprocess.run(
            [NODE_EXECUTABLE, "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_overview_search_keeps_matches_from_every_ara_snapshot(self) -> None:
        if NODE_EXECUTABLE is None:
            self.skipTest("Node.js is required for browser helper tests")
        script = _RESEARCH_DAG_SEARCH_HELPERS_JS + r"""
const assert = require('node:assert/strict');
const sources = [
  {name: 'research_vcs', object_count: 1},
  {name: 'ara:0', object_count: 1},
  {name: 'ara:1', object_count: 1},
  {name: 'ara:2', object_count: 1}
];
const nodes = [
  {id: 'rso-main', source: 'research_vcs'},
  {id: 'ara:0:match', source: 'ara'},
  {id: 'ara:1:match', source: 'ara'},
  {id: 'ara:2:other', source: 'ara'}
];
const index = {
  'ara:0:match': 'experiment_node regret baseline',
  'ara:1:match': 'experiment_node regret successor',
  'ara:2:other': 'experiment_node unrelated'
};
const directIds = projection => projection.nodes
  .filter(node => isDirectMatch(node, 'regret', index)).map(node => node.id).sort();
const expected = ['ara:0:match', 'ara:1:match'];
for (const ordered of [nodes, [...nodes].reverse()]) {
  const overview = projectDagView(ordered, sources, 'overview', 'all', 'regret', index);
  const full = projectDagView(ordered, sources, 'full', 'all', 'regret', index);
  assert.deepEqual(directIds(overview), expected);
  assert.deepEqual(directIds(overview), directIds(full));
  assert.deepEqual(overview.expandedSourceNames, ['ara:0', 'ara:1']);
  assert.equal(overview.expandedSourceName, null);
  assert.equal(overview.aggregateCount, 1);
  assert.equal(overview.collapsedNodeCount, 1);
  assert.ok(overview.nodes.some(node => node.id === 'ui:ara-source:ara:2'));
}
const selected = projectDagView(nodes, sources, 'overview', 'ara:1', 'regret', index);
assert.deepEqual(directIds(selected), ['ara:1:match']);
const exact = projectDagView(nodes, sources, 'overview', 'all', 'ara:1:match', index);
assert.equal(exact.aggregateCount, 2);
assert.equal(exact.expandedSourceName, 'ara:1');
const absent = projectDagView(nodes, sources, 'overview', 'all', 'missing', index);
assert.equal(absent.aggregateCount, 3);
assert.equal(absent.collapsedNodeCount, 3);
"""
        result = subprocess.run(
            [NODE_EXECUTABLE, "-e", script],
            check=False, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ara_decision_spine_is_bounded_and_presentation_only(self) -> None:
        if NODE_EXECUTABLE is None:
            self.skipTest("Node.js is required for browser helper tests")
        script = _RESEARCH_DAG_SEARCH_HELPERS_JS + r"""
const assert = require('node:assert/strict');
const sources = [
  {name: 'ara:0', object_count: 18, binding_count: 1},
  {name: 'ara:1', object_count: 18, binding_count: 0}
];
const nodes = [];
for (const source of ['ara:0', 'ara:1']) {
  for (let i = 0; i < 18; i++) {
    nodes.push({
      id: `${source}:n${String(i).padStart(2, '0')}`,
      source: 'ara', kind: 'experiment_node',
      state: i === 3 ? 'rejected' : 'completed', phase: i % 6,
      summary: i === 17 ? 'hidden kill candidate' : `budget-${i}`,
      proof: {level: i === 2 ? 'replayable' : 'traceable'}
    });
  }
}
const edges = [
  {source: 'ara:0:n00', target: 'ara:0:n01', type: 'evolves_to'},
  {source: 'ara:0:n16', target: 'ara:0:n17', type: 'evolves_to'}
];
const before = JSON.stringify(nodes);
const spine = projectDagView(nodes, sources, 'ara_spine', 'all', '', {});
assert.equal(spine.presentation, 'ara_decision_spine');
assert.equal(spine.maxNodes, 64);
assert.equal(spine.maxPerSource, 12);
assert.equal(spine.omittedNodeCount, 12);
assert.equal(spine.aggregateCount, 2);
assert.equal(spine.nodes.filter(node => !node.ui_aggregate).length, 24);
assert.equal(spine.nodes.filter(node => node.ui_aggregate).length, 2);
assert.ok(spine.nodes.every(node => node.ui_aggregate || node.source === 'ara'));
const spineEdges = authoritativeEdgesForNodes(spine.nodes, edges);
assert.ok(spineEdges.length <= edges.length);
assert.ok(spineEdges.every(edge => edges.includes(edge)));
assert.ok(spineEdges.every(edge => !String(edge.source).startsWith('ui:') && !String(edge.target).startsWith('ui:')));
assert.equal(JSON.stringify(nodes), before);
const hiddenSearch = projectDagView(
  nodes, sources, 'ara_spine', 'all', 'hidden kill',
  {'ara:0:n17': 'hidden kill candidate'}
);
assert.ok(hiddenSearch.nodes.some(node => node.id === 'ara:0:n17'));
assert.ok(hiddenSearch.nodes.length <= 26);
const selected = projectDagView(nodes, sources, 'ara_spine', 'ara:1', '', {});
assert.equal(selected.nodes.filter(node => !node.ui_aggregate).length, 12);
assert.equal(selected.nodes.filter(node => node.ui_aggregate).length, 1);
"""
        result = subprocess.run(
            [NODE_EXECUTABLE, "-e", script],
            check=False, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
