from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash, hash_manifest
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.ara_manifest_lock import (
    append_manifest_revision,
    write_manifest_lock,
)
from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_dag import (
    build_research_dag,
    export_research_dag,
    render_research_dag_html,
)
from xscientist.research_git import ResearchGitError
from xscientist.research_journey import (
    _command_for_repo,
    build_research_guide,
    public_research_guide_payload,
    start_guided_research,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for Research DAG tests")
class ResearchDagTests(unittest.TestCase):
    def test_repository_context_preserves_repository_neutral_templates(self) -> None:
        self.assertEqual(
            _command_for_repo(
                "xscientist research discovery template --output discovery.json",
                "/tmp/study",
            ),
            "xscientist research discovery template --output discovery.json",
        )
        self.assertEqual(
            _command_for_repo("xscientist research program review", "/tmp/study"),
            "xscientist research program review --repo /tmp/study",
        )

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
        self,
        *,
        ara_manifest_hash: str | None = None,
        ara_graph_hash: str | None = None,
        context_pack_hash: str | None = None,
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
        evidence_provenance = None
        if ara_manifest_hash:
            evidence_provenance = {"ara_manifest_hash": ara_manifest_hash}
            if ara_graph_hash:
                evidence_provenance["ara_exploration_graph_hash"] = ara_graph_hash
            if context_pack_hash:
                evidence_provenance["context_hashes"] = [context_pack_hash]
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
        self.assertEqual(
            {item["kind"] for item in objects},
            {"question", "research_goal", "hypothesis"},
        )
        guide = build_research_guide(self.repo_path, language="zh")

        self.assertEqual(guide["progress"]["completed_stages"], 1)
        self.assertEqual(guide["next_steps"][0]["code"], "record_rival_hypothesis")
        self.assertEqual(
            {step["code"] for step in guide["next_steps"]},
            {
                "record_rival_hypothesis",
                "choose_study_mode",
                "preregister_confirmatory",
                "lock_method_discovery",
            },
        )
        self.assertIn("竞争假设", guide["next_steps"][0]["title"])
        self.assertIn('"竞争假设"', guide["next_steps"][0]["command"])
        self.assertIn("什么结果会推翻竞争假设", guide["next_steps"][0]["command"])
        self.assertIn(f"--repo {self.repo_path}", guide["next_steps"][0]["command"])
        self.assertEqual(guide["primary_action"]["owner"], "researcher")
        self.assertEqual(
            guide["primary_action"]["required_inputs"],
            ["rival_hypothesis", "rival_disconfirming_result"],
        )
        self.assertTrue(self.repository.fsck()["ok"])

    def test_guide_locks_existing_competitors_before_choosing_study_mode(
        self,
    ) -> None:
        rival = self.repository.record(
            "hypothesis",
            {
                "statement": "Intervention A only changes measurement noise.",
                "falsifier": "The effect remains under an independent outcome measure.",
            },
        )
        self.repository.commit(stage="ideation", subject="record competing hypothesis")

        guide = build_research_guide(self.repo_path, language="en")

        self.assertEqual(guide["next_steps"][0]["code"], "lock_hypothesis_portfolio")
        self.assertEqual(guide["next_steps"][1]["code"], "choose_study_mode")
        command = guide["next_steps"][0]["command"]
        self.assertIn(self.hypothesis_id, command)
        self.assertIn(f"--alternative {rival.object_id}", command)
        self.assertIn('--question "RESEARCH QUESTION"', command)
        self.assertEqual(guide["primary_action"]["owner"], "researcher")
        self.assertEqual(
            guide["primary_action"]["required_inputs"],
            ["research_question", "primary_hypothesis", "rival_hypothesis"],
        )
        public_guide = public_research_guide_payload(guide)
        self.assertIn(
            "{workspace}", public_guide["primary_action"]["action"]["argv_template"]
        )
        self.assertFalse(
            public_guide["primary_action"]["action"]["executable_after_binding"]
        )
        self.assertTrue(
            public_guide["primary_action"]["action"]["input_binding"]["required"]
        )
        self.assertIn(
            "RESEARCH QUESTION",
            public_guide["primary_action"]["action"]["input_binding"]["placeholders"],
        )
        self.assertNotIn(str(self.repo_path), json.dumps(public_guide))

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

    def test_guided_start_preserves_unrelated_existing_git_work(self) -> None:
        existing = self.root / "existing-project"
        existing.mkdir()
        subprocess.run(
            ["git", "-C", str(existing), "init", "-b", "main"],
            check=True,
            capture_output=True,
            text=True,
        )
        readme = existing / "README.md"
        readme.write_text("user-owned uncommitted project\n", encoding="utf-8")

        start_guided_research(
            existing,
            question="Can managed research coexist with project work?",
            hypothesis="Only managed research paths enter the checkpoint.",
            falsifier="The unrelated README is committed.",
            git_user_name="Passed Researcher",
            git_user_email="passed@example.invalid",
        )

        committed_paths = set(
            subprocess.run(
                ["git", "-C", str(existing), "ls-tree", "-r", "--name-only", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
        author = subprocess.run(
            ["git", "-C", str(existing), "log", "-1", "--format=%an|%ae"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(existing), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        self.assertNotIn("README.md", committed_paths)
        self.assertIn("?? README.md", status)
        self.assertEqual(author, "Passed Researcher|passed@example.invalid")
        self.assertEqual(
            readme.read_text(encoding="utf-8"), "user-owned uncommitted project\n"
        )

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
        self.assertEqual(
            set(graph["scientific_closure"]["closure_levels"]),
            {"trace", "replay", "verify"},
        )
        self.assertGreater(
            graph["scientific_closure"]["closure_levels"]["verify"]["blocker_count"],
            0,
        )
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
        context_pack_hash = "sha256:" + "d" * 64
        ids = self._record_lineage(
            ara_manifest_hash=manifest_hash,
            context_pack_hash=context_pack_hash,
        )
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
                            "context_pack_refs": [context_pack_hash],
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
            "context_snapshot",
            {node["kind"] for node in graph["nodes"] if node["source"] == "ara"},
        )
        self.assertIn(
            "context",
            {
                edge["category"]
                for edge in graph["edges"]
                if edge["target"] == "ara:0:candidate"
            },
        )
        self.assertIn(
            ("ara:0:candidate", ids["evidence"], "anchors"),
            {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]},
        )
        context_node_id = next(
            node["id"]
            for node in graph["nodes"]
            if node.get("content_hash") == context_pack_hash
        )
        self.assertIn(
            (context_node_id, ids["evidence"], "contextualizes"),
            {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]},
        )
        ara_source = next(
            source for source in graph["sources"] if source["name"] == "ara:0"
        )
        self.assertEqual(ara_source["context_binding_count"], 1)
        self.assertEqual(ara_source["context_bound_object_ids"], [ids["evidence"]])

    def test_revised_ara_keeps_prior_revision_provenance_links(self) -> None:
        ara = self.root / "revised-ara"
        ara.mkdir()
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "counts": {"claims": 0},
        }
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_manifest_lock(ara, manifest)
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "candidate",
                            "content_hash": "sha256:" + "c" * 64,
                            "execution_isolation": {"isolated": True},
                        }
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        base_hash = hash_manifest(manifest)
        append_manifest_revision(
            ara / "manifest.json",
            lambda value: (
                value["counts"].__setitem__("claims", 1) or ["counts.claims"]
            ),
            reason="claim scan",
            producer="test",
        )
        current_manifest = json.loads((ara / "manifest.json").read_text())
        current_hash = hash_manifest(current_manifest)
        ids = self._record_lineage(ara_manifest_hash=base_hash)

        graph = build_research_dag(self.repo_path, ara_roots=[ara])

        source = next(item for item in graph["sources"] if item["name"] == "ara:0")
        self.assertEqual(source["manifest_integrity"]["state"], "revised")
        self.assertEqual(source["graph_binding"]["state"], "unbound_worktree")
        self.assertEqual(source["manifest_hash"], current_hash)
        self.assertEqual(set(source["manifest_hashes"]), {base_hash, current_hash})
        anchor = next(
            edge
            for edge in graph["edges"]
            if (edge["source"], edge["target"], edge["type"])
            == ("ara:0:candidate", ids["evidence"], "anchors")
        )
        self.assertEqual(anchor["category"], "lineage")

    def test_tampered_ara_cannot_create_verification_anchor(self) -> None:
        ara = self.root / "tampered-ara"
        ara.mkdir()
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "counts": {"claims": 0},
        }
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_manifest_lock(ara, manifest)
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "candidate",
                            "content_hash": "sha256:" + "d" * 64,
                            "execution_isolation": {"isolated": True},
                        }
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        base_hash = hash_manifest(manifest)
        manifest["counts"]["claims"] = 99
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        ids = self._record_lineage(ara_manifest_hash=base_hash)

        graph = build_research_dag(self.repo_path, ara_roots=[ara])

        source = next(item for item in graph["sources"] if item["name"] == "ara:0")
        self.assertEqual(source["manifest_integrity"]["state"], "tampered")
        self.assertFalse(graph["integrity"]["is_dag"])
        self.assertNotIn(
            ("ara:0:candidate", ids["evidence"], "anchors"),
            {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]},
        )

    def test_object_graph_hash_prevents_ambiguous_manifest_binding(self) -> None:
        ara = self.root / "graph-bound-ara"
        ara.mkdir()
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "counts": {"nodes": 1},
        }
        graph_payload = {
            "nodes": [
                {
                    "id": "candidate",
                    "content_hash": "sha256:" + "6" * 64,
                    "execution_isolation": {"isolated": True},
                }
            ],
            "edges": [],
        }
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_manifest_lock(ara, manifest)
        (ara / "exploration_graph.json").write_text(
            json.dumps(graph_payload), encoding="utf-8"
        )
        ids = self._record_lineage(
            ara_manifest_hash=hash_manifest(manifest),
            ara_graph_hash="sha256:" + "e" * 64,
        )

        graph = build_research_dag(self.repo_path, ara_roots=[ara])

        self.assertFalse(graph["integrity"]["is_dag"])
        self.assertNotIn(
            ("ara:0:candidate", ids["evidence"], "anchors"),
            {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]},
        )
        issue = next(
            item
            for item in graph["integrity"]["issues"]
            if item["code"] == "unresolved_ara_manifest_binding"
        )
        self.assertIn("sha256:" + "e" * 64, issue["message"])

    def test_historical_ref_auto_discovers_exact_committed_ara_snapshot(self) -> None:
        ara = self.repo_path / "ara" / "demo"
        ara.mkdir(parents=True)
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "counts": {"nodes": 1},
        }
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_manifest_lock(ara, manifest)
        old_graph = {
            "nodes": [
                {
                    "id": "old",
                    "content_hash": "sha256:" + "8" * 64,
                    "execution_isolation": {"isolated": True},
                }
            ],
            "edges": [],
        }
        (ara / "exploration_graph.json").write_text(
            json.dumps(old_graph), encoding="utf-8"
        )
        base_hash = hash_manifest(manifest)
        ids = self._record_lineage(
            ara_manifest_hash=base_hash,
            ara_graph_hash=content_hash(old_graph),
        )
        old_commit = self.repository.show()["commit"]
        first_manifest_ref = self.repository.show()["checkpoint"]["ara_manifests"][0]
        self.assertTrue(
            first_manifest_ref["exploration_graph_hash"].startswith("sha256:")
        )

        current_graph = {
            "nodes": [
                {
                    "id": "old",
                    "content_hash": "sha256:" + "8" * 64,
                    "execution_isolation": {"isolated": True},
                },
                {
                    "id": "new",
                    "parent_id": "old",
                    "content_hash": "sha256:" + "9" * 64,
                    "execution_isolation": {"isolated": True},
                },
            ],
            "edges": [{"parent": "old", "child": "new"}],
        }
        (ara / "exploration_graph.json").write_text(
            json.dumps(current_graph), encoding="utf-8"
        )
        append_manifest_revision(
            ara / "manifest.json",
            lambda value: (value["counts"].__setitem__("nodes", 2) or ["counts.nodes"]),
            reason="extend exploration",
            producer="test",
        )
        current_hash = hash_manifest(json.loads((ara / "manifest.json").read_text()))
        current_evidence = self.repository.record(
            "evidence",
            {
                "result": "The successor ARA adds a second experiment node.",
                "measurement_hash": "sha256:" + "7" * 64,
            },
            state="completed",
            provenance={
                "ara_manifest_hash": current_hash,
                "ara_exploration_graph_hash": content_hash(current_graph),
            },
        )
        self.repository.commit(stage="evidence", subject="extend committed ARA")
        revised_commit = self.repository.show()["commit"]
        self.assertTrue(
            any(
                "ara/demo/history/" in change
                for change in self.repository.diff(old_commit, revised_commit)[
                    "changes"
                ]
            )
        )
        self.repository.record(
            "hypothesis",
            {"statement": "A later checkpoint keeps the ARA binding."},
        )
        self.repository.commit(stage="ideation", subject="advance without changing ARA")
        carried_manifests = self.repository.show()["checkpoint"]["ara_manifests"]
        self.assertEqual(len(carried_manifests), 1)
        self.assertTrue(
            carried_manifests[0]["exploration_graph_hash"].startswith("sha256:")
        )

        historical = build_research_dag(self.repo_path, ref=old_commit)
        current = build_research_dag(self.repo_path)

        self.assertEqual(
            {node["id"] for node in historical["nodes"] if node["source"] == "ara"},
            {"ara:0:old"},
        )
        self.assertEqual(
            {node["id"] for node in current["nodes"] if node["source"] == "ara"},
            {"ara:0:old", "ara:0:new", "ara:1:old"},
        )
        historical_source = next(
            item for item in historical["sources"] if item["name"] == "ara:0"
        )
        current_source = next(
            item for item in current["sources"] if item["name"] == "ara:0"
        )
        self.assertEqual(historical_source["snapshot_ref"], old_commit)
        self.assertEqual(historical_source["manifest_hash"], base_hash)
        self.assertEqual(historical_source["graph_binding"]["state"], "verified")
        self.assertEqual(current_source["manifest_hash"], current_hash)
        self.assertEqual(current_source["manifest_integrity"]["state"], "revised")
        self.assertEqual(current_source["graph_binding"]["state"], "verified")
        anchors = {
            (edge["source"], edge["target"], edge["type"], edge["category"])
            for edge in current["edges"]
            if edge["type"] == "anchors"
        }
        self.assertIn(
            ("ara:0:new", current_evidence.object_id, "anchors", "verification"),
            anchors,
        )
        self.assertIn(
            ("ara:1:old", ids["evidence"], "anchors", "verification"),
            anchors,
        )
        historical_version = next(
            item for item in current["sources"] if item["name"] == "ara:1"
        )
        self.assertEqual(historical_version["snapshot_ref"], old_commit)

    def test_graph_changed_outside_checkpoint_cannot_keep_ara_anchors(self) -> None:
        ara = self.repo_path / "ara" / "tampered-graph"
        ara.mkdir(parents=True)
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "counts": {"nodes": 1},
        }
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_manifest_lock(ara, manifest)
        graph_path = ara / "exploration_graph.json"
        original_graph = {
            "nodes": [
                {
                    "id": "bound",
                    "content_hash": "sha256:" + "4" * 64,
                    "execution_isolation": {"isolated": True},
                }
            ],
            "edges": [],
        }
        graph_path.write_text(json.dumps(original_graph), encoding="utf-8")
        ids = self._record_lineage(
            ara_manifest_hash=hash_manifest(manifest),
            ara_graph_hash=content_hash(original_graph),
        )

        changed_graph = json.loads(graph_path.read_text())
        changed_graph["nodes"][0]["analysis"] = "silently changed after checkpoint"
        graph_path.write_text(json.dumps(changed_graph), encoding="utf-8")
        subprocess.run(
            ["git", "add", "ara/tampered-graph/exploration_graph.json"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "raw graph edit"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )

        with self.assertRaisesRegex(
            ResearchGitError, "is not bound to a research checkpoint"
        ):
            build_research_dag(self.repo_path)
        self.assertFalse(self.repository.fsck()["ok"])
        self.repository.record(
            "hypothesis",
            {
                "statement": "An unrelated checkpoint must not accept the raw graph edit."
            },
        )
        self.repository.commit(stage="ideation", subject="unrelated later checkpoint")
        later = build_research_dag(self.repo_path)
        later_source = next(
            item for item in later["sources"] if item["name"] == "ara:0"
        )
        self.assertEqual(later_source["graph_binding"]["state"], "mismatch")
        self.assertFalse(later["integrity"]["is_dag"])
        self.assertNotIn(
            ("ara:0:bound", ids["evidence"], "anchors"),
            {(edge["source"], edge["target"], edge["type"]) for edge in later["edges"]},
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

    def test_missing_ara_for_bound_research_object_blocks_integrity(self) -> None:
        missing_hash = "sha256:" + "f" * 64
        ids = self._record_lineage(ara_manifest_hash=missing_hash)

        graph = build_research_dag(self.repo_path)

        self.assertFalse(graph["integrity"]["is_dag"])
        issue = next(
            item
            for item in graph["integrity"]["issues"]
            if item["code"] == "unresolved_ara_manifest_binding"
        )
        self.assertIn(ids["evidence"], issue["message"])
        self.assertIn(missing_hash, issue["message"])

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
        self.assertIn('id="layer"', page)
        self.assertIn("Claim reasoning", page)
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
        self.assertNotIn(str(cli_repo), output.getvalue())
        self.assertFalse(Path(started["checkpoint"]["checkpoint_path"]).is_absolute())
        self.assertFalse(started["workspace_context"]["host_path_disclosed"])
        self.assertFalse(started["privacy"]["host_path_disclosed"])

        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                ["guide", "--repo", str(cli_repo), "--lang", "en", "--json"]
            )
        self.assertEqual(code, 0)
        public_guide = json.loads(output.getvalue())
        self.assertEqual(
            public_guide["next_steps"][0]["code"], "record_rival_hypothesis"
        )
        self.assertIn(
            "{workspace}", public_guide["next_steps"][0]["action"]["argv_template"]
        )
        self.assertNotIn("--repo .", public_guide["next_steps"][0]["command"])
        self.assertNotIn(str(cli_repo), output.getvalue())

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
