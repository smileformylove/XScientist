from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema
from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_dag import (
    build_research_dag,
    export_research_dag,
    render_research_dag_html,
)
from xscientist.research_journey import (
    build_research_guide,
    start_guided_research,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for Research DAG tests")
class ResearchDagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo_path = self.root / "study"
        started = start_guided_research(
            self.repo_path,
            question="Does intervention A improve outcome B?",
            hypothesis="Intervention A improves outcome B.",
            falsifier="Outcome B is unchanged or worse.",
            language="en",
            git_user_name="DAG Test",
            git_user_email="dag@example.invalid",
        )
        self.hypothesis_id = started["hypothesis_id"]
        self.repository = ResearchRepository(self.repo_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _record_lineage(
        self, *, ara_manifest_hash: str | None = None
    ) -> dict[str, str]:
        plan = self.repository.record(
            "research_plan",
            {"summary": "Compare A with the fixed baseline."},
            relations=[{"type": "depends_on", "target": self.hypothesis_id}],
        )
        attempt = self.repository.record(
            "experiment_attempt",
            {"status": "completed", "study_phase": "exploratory"},
            state="completed",
            relations=[
                {"type": "depends_on", "target": plan.object_id, "role": "plan"}
            ],
            provenance={
                "environment_hash": "sha256:" + "1" * 64,
                "dependency_lock_hashes": ["sha256:" + "2" * 64],
                "dataset_hashes": ["sha256:" + "3" * 64],
                "code_commit": "4" * 40,
                "seeds": [1, 2, 3],
            },
        )
        evidence_provenance = (
            {"ara_manifest_hash": ara_manifest_hash} if ara_manifest_hash else None
        )
        evidence = self.repository.record(
            "evidence",
            {
                "result": "The result challenges the primary hypothesis.",
                "measurement_hash": "sha256:" + "5" * 64,
            },
            state="completed",
            relations=[
                {"type": "derived_from", "target": attempt.object_id},
                {"type": "refutes", "target": self.hypothesis_id},
            ],
            provenance=evidence_provenance,
        )
        claim = self.repository.record(
            "claim",
            {"statement": "The tested intervention did not improve the outcome."},
            relations=[{"type": "depends_on", "target": evidence.object_id}],
        )
        self.repository.commit(stage="evidence", subject="record challenged lineage")
        return {
            "plan": plan.object_id,
            "attempt": attempt.object_id,
            "evidence": evidence.object_id,
            "claim": claim.object_id,
        }

    def test_guided_start_is_atomic_and_guide_uses_plain_next_step(self) -> None:
        objects = self.repository.objects()
        self.assertEqual({item["kind"] for item in objects}, {"question", "hypothesis"})
        guide = build_research_guide(self.repo_path, language="zh")

        self.assertEqual(guide["progress"]["completed_stages"], 1)
        self.assertEqual(guide["next_steps"][0]["code"], "choose_study_mode")
        self.assertEqual(
            {step["code"] for step in guide["next_steps"]},
            {"choose_study_mode", "preregister_confirmatory"},
        )
        self.assertIn("探索", guide["next_steps"][0]["title"])
        self.assertIn("@latest:hypothesis", guide["next_steps"][0]["command"])
        self.assertTrue(self.repository.fsck()["ok"])

    def test_guided_start_normalizes_plain_human_actor(self) -> None:
        second_repo = self.root / "actor-study"
        start_guided_research(
            second_repo,
            question="Can the workflow preserve actor identity?",
            hypothesis="A plain actor name is normalized.",
            falsifier="The stored actor lacks a human namespace.",
            actor="alice",
            git_user_name="Actor Test",
            git_user_email="actor@example.invalid",
        )

        actors = {
            item["actor"]["actor_id"]
            for item in ResearchRepository(second_repo).objects()
        }
        self.assertEqual(actors, {"human:alice"})

    def test_unified_dag_exposes_challenge_and_layered_proof(self) -> None:
        ids = self._record_lineage()
        graph = build_research_dag(self.repo_path)

        validate(graph, load_schema("research_dag"))
        self.assertTrue(graph["integrity"]["is_dag"], graph["integrity"]["issues"])
        nodes = {item["id"]: item for item in graph["nodes"]}
        self.assertEqual(nodes[self.hypothesis_id]["proof"]["level"], "contested")
        self.assertEqual(nodes[ids["attempt"]]["proof"]["level"], "replayable")
        self.assertEqual(nodes[ids["evidence"]]["proof"]["level"], "replayable")
        self.assertIn("missing_passing_gate", nodes[ids["claim"]]["proof"]["blockers"])
        challenge_edges = [
            edge for edge in graph["edges"] if edge["category"] == "challenge"
        ]
        self.assertEqual(challenge_edges[0]["source"], self.hypothesis_id)
        base = {
            key: value
            for key, value in graph.items()
            if key not in {"generated_at", "graph_hash"}
        }
        self.assertEqual(graph["graph_hash"], canonical_content_hash(base))

    def test_ara_exploration_nodes_link_to_manifest_bound_evidence(self) -> None:
        manifest_hash = "sha256:" + "a" * 64
        ids = self._record_lineage(ara_manifest_hash=manifest_hash)
        ara = self.root / "ara"
        ara.mkdir()
        (ara / "manifest.lock").write_text(
            json.dumps({"manifest_hash": manifest_hash}), encoding="utf-8"
        )
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "baseline",
                            "analysis": "Baseline run",
                            "content_hash": "sha256:" + "b" * 64,
                            "is_buggy": False,
                            "execution_isolation": {"isolated": True},
                        },
                        {
                            "id": "candidate",
                            "parent_id": "baseline",
                            "analysis": "Candidate failed",
                            "content_hash": "sha256:" + "c" * 64,
                            "is_buggy": True,
                            "execution_isolation": {"isolated": True},
                        },
                    ],
                    "edges": [{"parent": "baseline", "child": "candidate"}],
                }
            ),
            encoding="utf-8",
        )

        graph = build_research_dag(self.repo_path, ara_roots=[ara])

        self.assertTrue(graph["integrity"]["is_dag"], graph["integrity"]["issues"])
        self.assertIn("ara:0:candidate", {node["id"] for node in graph["nodes"]})
        self.assertIn(
            ("ara:0:candidate", ids["evidence"], "anchors"),
            {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]},
        )

    def test_invalid_ara_hash_does_not_claim_traceability(self) -> None:
        ara = self.root / "invalid-hash-ara"
        ara.mkdir()
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "unanchored",
                            "content_hash": "sha256:not-a-real-digest",
                            "execution_isolation": {"isolated": True},
                        }
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )

        graph = build_research_dag(self.repo_path, ara_roots=[ara])
        node = next(item for item in graph["nodes"] if item["source"] == "ara")

        self.assertIsNone(node["content_hash"])
        self.assertEqual(node["proof"]["level"], "recorded")
        self.assertIn("node_content_addressed", node["proof"]["blockers"])

    def test_invalid_ara_source_integrity_blocks_unified_dag(self) -> None:
        ara = self.root / "dangling-ara"
        ara.mkdir()
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [{"id": "child", "parent_id": "missing"}],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )

        graph = build_research_dag(self.repo_path, ara_roots=[ara])

        self.assertFalse(graph["integrity"]["is_dag"])
        self.assertIn(
            "ara:0_missing_edge_parent",
            {issue["code"] for issue in graph["integrity"]["issues"]},
        )

    def test_agent_evolution_is_visible_on_the_same_scientific_dag(self) -> None:
        candidate = self.repository.record(
            "agent_candidate",
            {
                "candidate_id": "search-policy-v2",
                "candidate_hash": "sha256:" + "d" * 64,
                "candidate_artifact_hash": "sha256:" + "e" * 64,
            },
            relations=[{"type": "derived_from", "target": self.hypothesis_id}],
        )
        evaluation = self.repository.record(
            "agent_evaluation",
            {"candidate_id": candidate.object_id, "verdict": "promote_to_canary"},
            state="verified",
            relations=[{"type": "evaluates", "target": candidate.object_id}],
            actor={
                "actor_id": "service:independent-evaluator",
                "authority": "independent_evaluator",
            },
        )
        gate = self.repository.record(
            "gate_decision",
            {"decision": "canary", "claim_promotion_allowed": False},
            state="completed",
            relations=[
                {"type": "evaluates", "target": evaluation.object_id},
                {"type": "promotes", "target": candidate.object_id},
            ],
            actor={
                "actor_id": "service:evolution-gate",
                "authority": "deterministic_gate",
            },
        )
        self.repository.commit(stage="evolution", subject="evaluate agent candidate")

        graph = build_research_dag(self.repo_path)
        nodes = {node["id"]: node for node in graph["nodes"]}

        self.assertEqual(nodes[candidate.object_id]["kind"], "agent_candidate")
        self.assertEqual(nodes[evaluation.object_id]["proof"]["level"], "verified")
        self.assertIn(
            (candidate.object_id, gate.object_id, "evolution"),
            {
                (edge["source"], edge["target"], edge["category"])
                for edge in graph["edges"]
            },
        )

    def test_offline_browser_is_searchable_accessible_and_script_safe(self) -> None:
        self._record_lineage()
        graph = build_research_dag(self.repo_path)
        graph["nodes"][0]["summary"] = "</script><script>alert(1)</script>"

        page = render_research_dag_html(graph)

        self.assertNotIn("</script><script>alert", page)
        self.assertIn("\\u003c/script\\u003e", page)
        self.assertIn('aria-label="Graph filters"', page)
        self.assertIn("Verification checks", page)
        self.assertNotIn("https://", page)

        exported = export_research_dag(self.repo_path, self.root / "dag")
        self.assertTrue(Path(exported["json"]).is_file())
        self.assertTrue(Path(exported["html"]).is_file())

    def test_cli_start_guide_and_dag_journey(self) -> None:
        cli_repo = self.root / "cli-study"
        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                [
                    "start",
                    str(cli_repo),
                    "--question",
                    "Can X improve Y?",
                    "--hypothesis",
                    "X improves Y.",
                    "--falsifier",
                    "Y does not improve.",
                    "--git-user-name",
                    "CLI Test",
                    "--git-user-email",
                    "cli@example.invalid",
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        started = json.loads(output.getvalue())
        self.assertTrue(started["hypothesis_id"].startswith("rso-"))

        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                ["guide", "--repo", str(cli_repo), "--lang", "en", "--json"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(output.getvalue())["next_steps"][0]["code"],
            "choose_study_mode",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                [
                    "plan",
                    "@latest:hypothesis",
                    "Compare X with the fixed baseline.",
                    "--test",
                    "Measure Y under identical conditions.",
                    "--repo",
                    str(cli_repo),
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(output.getvalue())["object"]["kind"], "research_plan"
        )

        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                ["guide", "--repo", str(cli_repo), "--lang", "en", "--json"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(output.getvalue())["next_steps"][0]["code"],
            "run_experiment",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                ["dag", "--repo", str(cli_repo), "--metadata-only", "--json"]
            )
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.getvalue())["content_disclosed"])


if __name__ == "__main__":
    unittest.main()
