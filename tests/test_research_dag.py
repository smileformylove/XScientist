from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest import mock

from jsonschema import validate

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash, hash_manifest
from ai_scientist.protocol.research_vcs import (
    RESEARCH_RELATION_TYPES,
    RESEARCH_SEMANTIC_PROFILE_SCHEMA,
)
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.ara_manifest_lock import (
    append_manifest_revision,
    write_manifest_lock,
)
import xscientist.research_dag as research_dag_module
from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_dag import (
    _gate_detail_views,
    _object_ara_bindings,
    build_research_dag,
    export_research_dag,
    render_research_dag_html,
)
from xscientist.research_git import (
    ResearchGitError,
    ResearchObjectResult,
    add_research_object,
)
from xscientist.research_journey import (
    _command_for_repo,
    build_research_guide,
    public_research_guide_payload,
    start_guided_research,
    workspace_action_contract,
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
        self.assertEqual(
            _command_for_repo("xscientist record . --lang zh", "/tmp/study"),
            "xscientist record /tmp/study --lang zh",
        )

    def test_record_action_binds_positional_workspace_and_requires_interaction(
        self,
    ) -> None:
        action = workspace_action_contract(
            "xscientist record /private/host/study --lang zh"
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(
            action["argv_template"],
            ["xscientist", "record", "{workspace}", "--lang", "zh"],
        )
        self.assertEqual(action["workspace_binding"]["mode"], "argument")
        self.assertFalse(action["executable_after_binding"])
        self.assertEqual(action["input_binding"]["mode"], "interactive")
        self.assertTrue(action["input_binding"]["required"])
        self.assertEqual(action["input_binding"]["placeholders"], [])
        self.assertNotIn("/private/host/study", json.dumps(action))

        for control_flag in ("--json", "--non-interactive"):
            with self.subTest(control_flag=control_flag):
                controlled = workspace_action_contract(
                    f"xscientist record /private/host/study {control_flag}"
                )
                assert controlled is not None
                self.assertFalse(controlled["executable_after_binding"])
                self.assertEqual(controlled["input_binding"]["mode"], "arguments")
                self.assertTrue(controlled["input_binding"]["required"])

        scripted = workspace_action_contract(
            'xscientist record /private/host/study --summary "run" '
            "--status completed --non-interactive"
        )
        assert scripted is not None
        self.assertTrue(scripted["executable_after_binding"])
        self.assertFalse(scripted["input_binding"]["required"])

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
        include_paths: Sequence[str] = (),
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
        if include_paths:
            self.repository.stage(include_paths)
            self.repository.stage(all_changes=True)
            self.repository.commit(
                stage="evidence",
                subject="record challenged lineage",
                staged_only=True,
            )
        else:
            self.repository.commit(
                stage="evidence", subject="record challenged lineage"
            )
        return {
            "plan": plan.object_id,
            "attempt": attempt.object_id,
            "evidence": evidence.object_id,
            "claim": claim.object_id,
        }

    def _detail_payload(self, exported: Mapping[str, Any]) -> dict[str, Any]:
        page = Path(exported["html"]).read_text(encoding="utf-8")
        encoded = page.split('<script id="detail-data" type="application/json">', 1)[
            1
        ].split("</script>", 1)[0]
        return json.loads(encoded)

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

    def test_typed_relation_category_cannot_be_overridden_by_role(self) -> None:
        evidence = self.repository.record(
            "evidence",
            {
                "result": "The typed relations remain authoritative.",
                "measurement_hash": "sha256:" + "6" * 64,
            },
            state="completed",
            relations=[
                {
                    "type": "refutes",
                    "target": self.hypothesis_id,
                    "role": "premise",
                },
                {
                    "type": "qualifies",
                    "target": self.hypothesis_id,
                    "role": "context_source",
                },
            ],
        )
        self.repository.commit(stage="evidence", subject="record typed relations")

        graph = build_research_dag(self.repo_path)
        by_type = {
            edge["type"]: edge
            for edge in graph["edges"]
            if edge["target"] == evidence.object_id
        }

        self.assertEqual(by_type["refutes"]["category"], "challenge")
        self.assertEqual(by_type["qualifies"]["category"], "argument")
        self.assertEqual(by_type["refutes"]["role"], "premise")
        self.assertEqual(by_type["qualifies"]["role"], "context_source")

        metadata = build_research_dag(self.repo_path, disclose_summaries=False)
        self.assertTrue(
            all(edge["role"] == "" for edge in metadata["edges"]),
            metadata["edges"],
        )

    def test_unknown_extension_relation_is_explicitly_unclassified(self) -> None:
        profile_core = {
            "schema": RESEARCH_SEMANTIC_PROFILE_SCHEMA,
            "uri": "urn:xscientist:test:extension-profile",
            "version": "1.0.0",
            "kinds": ["evidence"],
            "relations": ["urn:xscientist:test:extension"],
        }
        evidence = self.repository.record(
            "evidence",
            {"result": "An extension relation must not become lineage."},
            state="completed",
            relations=[
                {"type": "urn:xscientist:test:extension", "target": self.hypothesis_id}
            ],
            semantic_profile={
                **profile_core,
                "schema_digest": canonical_content_hash(profile_core),
            },
        )
        self.repository.commit(stage="evidence", subject="record extension relation")

        graph = build_research_dag(self.repo_path)
        edge = next(
            edge
            for edge in graph["edges"]
            if edge["target"] == evidence.object_id
            and edge["type"] == "urn:xscientist:test:extension"
        )
        self.assertEqual(edge["category"], "unknown")
        self.assertTrue(
            any(
                issue["code"] == "unknown_research_relation_type"
                and evidence.object_id in issue["message"]
                for issue in graph["integrity"]["issues"]
            )
        )

    def test_all_native_relations_have_explicit_categories(self) -> None:
        self.assertEqual(
            set(RESEARCH_RELATION_TYPES) - set(research_dag_module._RELATION_CATEGORIES),
            set(),
        )
        for relation in ("derived_by", "consumes", "produces"):
            self.assertEqual(research_dag_module._RELATION_CATEGORIES[relation], "lineage")

    def test_metadata_only_clears_synthesized_ara_edge_roles(self) -> None:
        ara = self.root / "metadata-ara"
        ara.mkdir()
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [{"id": "root"}, {"id": "child"}],
                    "edges": [{"parent": "root", "child": "child"}],
                }
            ),
            encoding="utf-8",
        )

        metadata = build_research_dag(
            self.repo_path, ara_roots=[ara], disclose_summaries=False
        )

        ara_edges = [edge for edge in metadata["edges"] if edge["source"] == "ara:0:root"]
        self.assertTrue(ara_edges, metadata["edges"])
        self.assertTrue(all(edge["role"] == "" for edge in ara_edges), ara_edges)

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

    def test_revised_ara_does_not_bind_prior_manifest_to_current_projection(
        self,
    ) -> None:
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
        self.assertNotIn(
            ("ara:0:candidate", ids["evidence"], "anchors"),
            {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]},
        )
        self.assertFalse(graph["integrity"]["is_dag"])
        issue = next(
            item
            for item in graph["integrity"]["issues"]
            if item["code"] == "unresolved_ara_manifest_binding"
        )
        self.assertIn(base_hash, issue["message"])

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
                    "analysis": "Analysis stored at the original ARA snapshot.",
                    "content_hash": "sha256:" + "8" * 64,
                    "execution_isolation": {"isolated": True},
                    "metric": {"name": "loss", "value": 0.0},
                }
            ],
            "edges": [],
        }
        old_metrics_path = ara / "nodes" / "old" / "metrics.json"
        old_metrics_path.parent.mkdir(parents=True)
        old_metrics_path.write_text(
            json.dumps(
                {
                    "analysis": "Analysis stored at the original ARA snapshot.",
                    "metric": {"name": "loss", "value": 0.0},
                    "content_hash": "sha256:" + "8" * 64,
                }
            ),
            encoding="utf-8",
        )
        self.assertNotEqual(content_hash(old_graph), canonical_content_hash(old_graph))
        (ara / "exploration_graph.json").write_text(
            json.dumps(old_graph), encoding="utf-8"
        )
        base_hash = hash_manifest(manifest)
        ids = self._record_lineage(
            ara_manifest_hash=base_hash,
            ara_graph_hash=content_hash(old_graph),
        )
        old_commit = self.repository.show()["commit"]
        committed_paths = set(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_path),
                    "ls-tree",
                    "-r",
                    "--name-only",
                    old_commit,
                ],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.splitlines()
        )
        self.assertIn("ara/demo/nodes/old/metrics.json", committed_paths)
        first_manifest_ref = self.repository.show()["checkpoint"]["ara_manifests"][0]
        self.assertTrue(
            first_manifest_ref["exploration_graph_hash"].startswith("sha256:")
        )

        current_graph = {
            "nodes": [
                {
                    "id": "old",
                    "analysis": "Analysis stored only in the successor snapshot.",
                    "content_hash": "sha256:" + "8" * 64,
                    "execution_isolation": {"isolated": True},
                    "metric": {"name": "loss", "value": 0.0},
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
        old_metrics_path.write_text(
            json.dumps(
                {
                    "analysis": "Analysis stored only in the successor snapshot.",
                    "metric": {"name": "loss", "value": 0.0},
                    "content_hash": "sha256:" + "8" * 64,
                }
            ),
            encoding="utf-8",
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
        self.repository.stage(all_changes=True)
        self.repository.commit(
            stage="evidence",
            subject="extend committed ARA",
            staged_only=True,
        )
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

        historical_export = export_research_dag(
            self.repo_path, self.root / "historical-dag", ref=old_commit
        )
        current_export = export_research_dag(self.repo_path, self.root / "current-dag")

        self.assertEqual(
            self._detail_payload(historical_export)["ara:0:old"]["fields"]["analysis"],
            "Analysis stored at the original ARA snapshot.",
        )
        current_details = self._detail_payload(current_export)
        self.assertEqual(
            current_details["ara:0:old"]["fields"]["analysis"],
            "Analysis stored only in the successor snapshot.",
        )
        self.assertEqual(
            current_details["ara:1:old"]["fields"]["analysis"],
            "Analysis stored at the original ARA snapshot.",
        )

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
        self._record_lineage(
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
        with self.assertRaisesRegex(ResearchGitError, "uncheckpointed raw Git commit"):
            self.repository.commit(
                stage="ideation", subject="unrelated later checkpoint"
            )
        with self.assertRaisesRegex(
            ResearchGitError, "is not bound to a research checkpoint"
        ):
            build_research_dag(self.repo_path)

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

    def test_explicit_malformed_ara_graph_hash_cannot_use_legacy_wildcard(self) -> None:
        manifest_hash = "sha256:" + "8" * 64
        malformed_values = (None, "", "not-a-valid-content-hash", 7)

        for malformed in malformed_values:
            with self.subTest(malformed=malformed):
                bindings = _object_ara_bindings(
                    {
                        "provenance": {
                            "ara_manifest_hash": manifest_hash,
                            "ara_exploration_graph_hash": malformed,
                        },
                        # A legacy manifest-only duplicate must not hide the
                        # stronger but malformed explicit graph declaration.
                        "payload": {"ara_manifest_hash": manifest_hash},
                    }
                )

                self.assertEqual(bindings, {(manifest_hash, "<invalid>")})
                self.assertNotIn((manifest_hash, None), bindings)

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

    def test_ara_projection_ignores_stale_mirrors_for_edges_encoding(self) -> None:
        ara = self.root / "edges-only-ara"
        ara.mkdir()
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "topology_encoding": "edges",
                    "nodes": [
                        {"id": "root", "parent_id": "child"},
                        {"id": "child", "children": ["root"]},
                    ],
                    "edges": [{"parent": "root", "child": "child"}],
                }
            ),
            encoding="utf-8",
        )

        graph = build_research_dag(self.repo_path, ara_roots=[ara])

        evolution = {
            (edge["source"], edge["target"])
            for edge in graph["edges"]
            if edge["category"] == "evolution" and edge["source"].startswith("ara:0:")
        }
        self.assertEqual(evolution, {("ara:0:root", "ara:0:child")})
        self.assertTrue(graph["integrity"]["is_dag"], graph["integrity"]["issues"])

    def test_ara_projection_preserves_legacy_children_only_topology(self) -> None:
        ara = self.root / "legacy-children-ara"
        ara.mkdir()
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"id": "root", "children": ["child"]},
                        {"id": "child", "children": []},
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )

        graph = build_research_dag(self.repo_path, ara_roots=[ara])

        self.assertIn(
            ("ara:0:root", "ara:0:child"),
            {
                (edge["source"], edge["target"])
                for edge in graph["edges"]
                if edge["category"] == "evolution"
            },
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
        graph["nodes"][0]["metric"] = float("nan")
        graph["strategy_summary"][
            "open_anomaly_count"
        ] = '</span><img src=x onerror="alert(2)">'

        page = render_research_dag_html(
            graph,
            node_details={
                graph["nodes"][0]["id"]: {
                    "fields": {"loss": float("inf"), "lower": float("-inf")}
                }
            },
        )

        self.assertNotIn("</script><script>alert", page)
        self.assertIn("\\u003c/script\\u003e", page)
        self.assertNotIn(":NaN", page)
        self.assertNotIn(":Infinity", page)
        self.assertIn('"metric":"NaN"', page)
        self.assertIn('"loss":"Infinity"', page)
        self.assertIn('"lower":"-Infinity"', page)
        self.assertNotIn("</span><img", page)
        self.assertIn("esc(graph.strategy_summary.open_anomaly_count||0)", page)
        self.assertIn('aria-label="Graph filters"', page)
        self.assertIn('id="layer"', page)
        self.assertIn('id="details" aria-live="polite"', page)
        self.assertIn(
            "renderStats(nodes,edges,seedNodes.length,contextCount,Boolean(q))",
            page,
        )
        self.assertIn("No scientific object matches the current filters.", page)
        self.assertIn("graph.integrity.topological_order", page)
        self.assertIn("<title>${esc(e.type||'relation')}", page)
        self.assertIn("Claim reasoning", page)
        self.assertIn("Verification checks", page)
        self.assertNotIn("https://", page)

        exported = export_research_dag(self.repo_path, self.root / "dag")
        self.assertTrue(Path(exported["json"]).is_file())
        self.assertTrue(Path(exported["html"]).is_file())

    def test_export_reuses_the_dag_object_snapshot_for_gate_details(self) -> None:
        self._record_lineage()
        loader = research_dag_module.list_research_objects_at_ref

        with mock.patch.object(
            research_dag_module,
            "list_research_objects_at_ref",
            wraps=loader,
        ) as observed_loader:
            export_research_dag(self.repo_path, self.root / "dag")

        self.assertEqual(observed_loader.call_count, 1)

    def test_export_includes_ara_scientific_detail_without_changing_graph(self) -> None:
        ara = self.root / "ara-detail"
        ara.mkdir()
        node_hash = "sha256:" + "b" * 64
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "decision",
                            "stage": "decision",
                            "analysis": "Legacy graph analysis.",
                            "plan_excerpt": "Stop this candidate.",
                            "metric": {
                                "name": "continuation_gate_passed",
                                "value": 0,
                                "maximize": True,
                            },
                            "content_hash": node_hash,
                            "is_buggy": False,
                        }
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        metrics_path = ara / "nodes" / "decision" / "metrics.json"
        metrics_path.parent.mkdir(parents=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "analysis": "Candidate failed its scientific gate.",
                    "scientific_outcome": "rejected_hypothesis",
                    "content_hash": node_hash,
                    "metric": {
                        "name": "continuation_gate_passed",
                        "value": 0,
                        "maximize": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        baseline = build_research_dag(self.repo_path, ara_roots=[ara])
        output = self.root / "ara-detail-dag"
        exported = export_research_dag(self.repo_path, output, ara_roots=[ara])
        exported_graph = json.loads(Path(exported["json"]).read_text(encoding="utf-8"))
        page = Path(exported["html"]).read_text(encoding="utf-8")
        detail_json = page.split(
            '<script id="detail-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        details = json.loads(detail_json)

        self.assertEqual(exported_graph["graph_hash"], baseline["graph_hash"])
        self.assertFalse(any("details" in node for node in exported_graph["nodes"]))
        view = details["ara:0:decision"]
        self.assertEqual(view["title"], "ARA experiment detail")
        self.assertEqual(
            view["fields"]["analysis"],
            "Legacy graph analysis.",
        )
        self.assertNotIn("scientific_outcome", view["fields"])
        self.assertEqual(
            view["metrics"][0]["values"],
            {
                "name": "continuation_gate_passed",
                "value": 0,
                "maximize": True,
            },
        )
        projected = next(
            node for node in exported_graph["nodes"] if node["id"] == "ara:0:decision"
        )
        self.assertEqual(projected["state"], "completed")
        self.assertEqual(projected["summary"], "Legacy graph analysis.")
        self.assertIn("const displaySummary=n=>", page)
        self.assertIn("ARA experiment detail", page)

        metadata = export_research_dag(
            self.repo_path,
            self.root / "ara-detail-metadata",
            ara_roots=[ara],
            disclose_summaries=False,
        )
        metadata_page = Path(metadata["html"]).read_text(encoding="utf-8")
        metadata_json = metadata_page.split(
            '<script id="detail-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(metadata_json), {})

    def test_ara_sidecar_requires_matching_node_content_hash(self) -> None:
        ara = self.root / "ara-hash-bound-detail"
        nodes = ara / "nodes"
        nodes.mkdir(parents=True)
        graph_nodes = []
        cases = {
            "mismatch": "sha256:" + "1" * 64,
            "missing": "sha256:" + "2" * 64,
            "malformed": "not-a-content-hash",
        }
        for node_id, node_hash in cases.items():
            graph_nodes.append(
                {
                    "id": node_id,
                    "analysis": f"Trusted legacy {node_id} analysis.",
                    "content_hash": node_hash,
                }
            )
            node_dir = nodes / node_id
            node_dir.mkdir()
            companion = {
                "analysis": f"UNBOUND-{node_id}-ANALYSIS",
                "scientific_outcome": "rejected_hypothesis",
            }
            if node_id == "mismatch":
                companion["content_hash"] = "sha256:" + "f" * 64
            elif node_id == "malformed":
                companion["content_hash"] = node_hash
            (node_dir / "metrics.json").write_text(
                json.dumps(companion), encoding="utf-8"
            )
        graph_nodes.append({"id": "legacy", "stage": "legacy"})
        legacy_dir = nodes / "legacy"
        legacy_dir.mkdir()
        (legacy_dir / "metrics.json").write_text(
            json.dumps({"analysis": "Legacy unhashed companion analysis."}),
            encoding="utf-8",
        )
        (ara / "exploration_graph.json").write_text(
            json.dumps({"nodes": graph_nodes, "edges": []}), encoding="utf-8"
        )

        exported = export_research_dag(
            self.repo_path, self.root / "ara-hash-bound-dag", ara_roots=[ara]
        )
        page = Path(exported["html"]).read_text(encoding="utf-8")
        detail_json = page.split(
            '<script id="detail-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        details = json.loads(detail_json)

        self.assertNotIn("UNBOUND-mismatch-ANALYSIS", page)
        self.assertNotIn("UNBOUND-missing-ANALYSIS", page)
        self.assertNotIn("UNBOUND-malformed-ANALYSIS", page)
        for node_id in cases:
            view = details[f"ara:0:{node_id}"]
            self.assertEqual(
                view["fields"],
                {"analysis": f"Trusted legacy {node_id} analysis."},
            )
            self.assertEqual(view["metrics"], [])
        self.assertEqual(
            details["ara:0:malformed"]["detail_binding"]["state"],
            "graph_only",
        )
        self.assertEqual(
            details["ara:0:legacy"]["fields"]["analysis"],
            "Legacy unhashed companion analysis.",
        )
        self.assertEqual(
            details["ara:0:legacy"]["detail_binding"]["state"], "snapshot_only"
        )

    def test_ara_sidecar_metric_conflict_falls_back_and_receipt_is_recomputable(
        self,
    ) -> None:
        ara = self.root / "ara-metric-conflict"
        metrics_path = ara / "nodes" / "decision" / "metrics.json"
        metrics_path.parent.mkdir(parents=True)
        node_hash = "sha256:" + "b" * 64
        graph_payload = {
            "nodes": [
                {
                    "id": "decision",
                    "analysis": "Graph-bound analysis.",
                    "content_hash": node_hash,
                    "metric": {"name": "loss", "value": 0},
                }
            ],
            "edges": [],
        }
        graph_path = ara / "exploration_graph.json"
        graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
        metrics_path.write_text(
            json.dumps(
                {
                    "analysis": "FORGED-SIDECAR-ANALYSIS",
                    "content_hash": node_hash,
                    "metric": {"name": "loss", "value": 999},
                }
            ),
            encoding="utf-8",
        )

        exported = export_research_dag(
            self.repo_path, self.root / "ara-metric-conflict-dag", ara_roots=[ara]
        )
        detail = self._detail_payload(exported)["ara:0:decision"]

        self.assertEqual(detail["fields"], {"analysis": "Graph-bound analysis."})
        self.assertEqual(detail["metrics"][0]["values"], {"name": "loss", "value": 0})
        self.assertNotIn("FORGED-SIDECAR-ANALYSIS", json.dumps(detail))
        snapshot = detail["source_snapshot"]
        self.assertEqual(snapshot["source_ref"], "worktree")
        self.assertEqual(snapshot["source_path"], "nodes/decision/metrics.json")
        self.assertEqual(
            snapshot["graph_sha256"],
            "sha256:" + hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            snapshot["sidecar_sha256"],
            "sha256:" + hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        )
        receipt_fields = {
            key: value for key, value in snapshot.items() if key != "receipt_hash"
        }
        self.assertEqual(
            snapshot["receipt_hash"], canonical_content_hash(receipt_fields)
        )

    def test_same_graph_revisions_keep_current_and_historical_receipts_separate(
        self,
    ) -> None:
        ara = self.repo_path / "ara" / "stable-graph"
        metrics_path = ara / "nodes" / "candidate" / "metrics.json"
        metrics_path.parent.mkdir(parents=True)
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "counts": {"revisions": 1},
        }
        graph_payload = {
            "nodes": [
                {
                    "id": "candidate",
                    "analysis": "Stable graph analysis.",
                    "content_hash": "sha256:" + "c" * 64,
                    "metric": {"name": "loss", "value": 0},
                }
            ],
            "edges": [],
        }
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_manifest_lock(ara, manifest)
        graph_path = ara / "exploration_graph.json"
        graph_path.write_text(json.dumps(graph_payload), encoding="utf-8")
        metrics_path.write_text(
            json.dumps(
                {
                    "revision_marker": "old-sidecar",
                    "content_hash": "sha256:" + "c" * 64,
                    "metric": {"name": "loss", "value": 0},
                }
            ),
            encoding="utf-8",
        )
        graph_hash = content_hash(graph_payload)
        old_manifest_hash = hash_manifest(manifest)
        old_ids = self._record_lineage(
            ara_manifest_hash=old_manifest_hash,
        )
        old_commit = self.repository.show()["commit"]
        old_sidecar_sha256 = (
            "sha256:" + hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        )

        append_manifest_revision(
            ara / "manifest.json",
            lambda value: (
                value["counts"].__setitem__("revisions", 2) or ["counts.revisions"]
            ),
            reason="sidecar refresh",
            producer="test",
        )
        metrics_path.write_text(
            json.dumps(
                {
                    "revision_marker": "new-sidecar",
                    "content_hash": "sha256:" + "c" * 64,
                    "metric": {"name": "loss", "value": 0},
                }
            ),
            encoding="utf-8",
        )
        current_manifest = json.loads((ara / "manifest.json").read_text())
        current_manifest_hash = hash_manifest(current_manifest)
        current_evidence = self.repository.record(
            "evidence",
            {
                "result": "The same graph was inspected under the new manifest.",
                "measurement_hash": "sha256:" + "d" * 64,
            },
            state="completed",
            provenance={
                "ara_manifest_hash": current_manifest_hash,
                "ara_exploration_graph_hash": graph_hash,
            },
        )
        self.repository.stage(all_changes=True)
        self.repository.commit(
            stage="evidence", subject="refresh sidecar without changing graph"
        )

        exported = export_research_dag(self.repo_path, self.root / "stable-graph-dag")
        details = self._detail_payload(exported)
        sources = exported["graph"]["sources"]
        current_source = next(row for row in sources if row["name"] == "ara:0")
        historical_source = next(row for row in sources if row["name"] == "ara:1")

        self.assertEqual(current_source["manifest_hash"], current_manifest_hash)
        self.assertEqual(historical_source["manifest_hash"], old_manifest_hash)
        self.assertEqual(historical_source["snapshot_ref"], old_commit)
        self.assertEqual(
            current_source["graph_binding"]["graph_hash"],
            historical_source["graph_binding"]["graph_hash"],
        )
        current_snapshot = details["ara:0:candidate"]["source_snapshot"]
        historical_snapshot = details["ara:1:candidate"]["source_snapshot"]
        self.assertNotEqual(
            current_snapshot["sidecar_sha256"],
            historical_snapshot["sidecar_sha256"],
        )
        self.assertEqual(historical_snapshot["sidecar_sha256"], old_sidecar_sha256)
        self.assertNotEqual(
            current_snapshot["receipt_hash"], historical_snapshot["receipt_hash"]
        )
        anchors = {
            (edge["source"], edge["target"], edge["type"])
            for edge in exported["graph"]["edges"]
        }
        self.assertIn(
            ("ara:0:candidate", current_evidence.object_id, "anchors"), anchors
        )
        self.assertIn(("ara:1:candidate", old_ids["evidence"], "anchors"), anchors)

    def test_strict_json_rejects_duplicate_keys_and_non_finite_constants(
        self,
    ) -> None:
        for invalid_value in (
            '{"nodes":[],"nodes":[{"id":"LEAKED-DUPLICATE"}],"edges":[]}',
            '{"nodes":[{"id":"candidate","metric":NaN}],"edges":[]}',
            '{"nodes":[{"id":"candidate","metric":Infinity}],"edges":[]}',
            '{"nodes":[{"id":"candidate","metric":-Infinity}],"edges":[]}',
        ):
            with self.subTest(core_json=invalid_value):
                ara = self.root / ("invalid-core-" + str(abs(hash(invalid_value))))
                ara.mkdir()
                (ara / "exploration_graph.json").write_text(
                    invalid_value, encoding="utf-8"
                )
                with self.assertRaisesRegex(ResearchGitError, "invalid") as raised:
                    build_research_dag(self.repo_path, ara_roots=[ara])
                self.assertNotIn("LEAKED-DUPLICATE", str(raised.exception))

        for invalid_line in (
            '{"revision":1,"revision":2}',
            '{"revision":NaN}',
            '{"revision":Infinity}',
            '{"revision":-Infinity}',
        ):
            with self.subTest(jsonl=invalid_line):
                rows, errors = research_dag_module._read_json_lines(invalid_line)
                self.assertEqual(rows, [])
                self.assertEqual(len(errors), 1)

        for invalid_blob in (
            b'{"nodes":[],"nodes":[{"id":"duplicate"}]}',
            b'{"nodes":[],"revision":NaN}',
        ):
            with self.subTest(committed_json=invalid_blob):
                with mock.patch.object(
                    research_dag_module,
                    "_blob_bytes_at_commit",
                    return_value=invalid_blob,
                ):
                    with self.assertRaisesRegex(ResearchGitError, "invalid"):
                        research_dag_module._json_blob_at_commit(
                            self.repo_path,
                            "a" * 40,
                            "ara/demo/exploration_graph.json",
                        )

        for invalid_manifest in (
            '{"schema_version":"ara.v1","schema_version":"LEAKED"}',
            '{"schema_version":"ara.v1","revision":NaN}',
        ):
            with self.subTest(manifest=invalid_manifest):
                ara = self.root / (
                    "invalid-manifest-" + str(abs(hash(invalid_manifest)))
                )
                ara.mkdir()
                (ara / "exploration_graph.json").write_text(
                    json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
                )
                (ara / "manifest.json").write_text(invalid_manifest, encoding="utf-8")
                with self.assertRaisesRegex(ResearchGitError, "invalid") as raised:
                    build_research_dag(self.repo_path, ara_roots=[ara])
                self.assertNotIn("LEAKED", str(raised.exception))

        for invalid_sidecar in (
            '{"analysis":"safe","analysis":"LEAKED-DUPLICATE"}',
            '{"analysis":"LEAKED-NAN","metric":NaN}',
            '{"analysis":"LEAKED-INFINITY","metric":Infinity}',
        ):
            with self.subTest(sidecar=invalid_sidecar):
                ara = self.root / ("invalid-sidecar-" + str(abs(hash(invalid_sidecar))))
                metrics_path = ara / "nodes" / "candidate" / "metrics.json"
                metrics_path.parent.mkdir(parents=True)
                (ara / "exploration_graph.json").write_text(
                    json.dumps(
                        {
                            "nodes": [
                                {
                                    "id": "candidate",
                                    "analysis": "Graph fallback.",
                                    "metric": {"name": "loss", "value": 0},
                                }
                            ],
                            "edges": [],
                        }
                    ),
                    encoding="utf-8",
                )
                metrics_path.write_text(invalid_sidecar, encoding="utf-8")
                exported = export_research_dag(
                    self.repo_path,
                    self.root
                    / ("invalid-sidecar-dag-" + str(abs(hash(invalid_sidecar)))),
                    ara_roots=[ara],
                )
                detail = self._detail_payload(exported)["ara:0:candidate"]
                self.assertEqual(detail["fields"], {"analysis": "Graph fallback."})
                self.assertNotIn("LEAKED-", json.dumps(detail))

    def test_committed_ara_blob_is_size_checked_before_reading(self) -> None:
        blob_id = "a" * 40
        path = "ara/demo/nodes/decision/metrics.json"
        responses = [
            subprocess.CompletedProcess([], 0, f"100644 blob {blob_id}\t{path}\0", ""),
            subprocess.CompletedProcess(
                [], 0, str(research_dag_module._MAX_ARA_NODE_DETAIL_BYTES + 1), ""
            ),
        ]

        with (
            mock.patch.object(research_dag_module, "_run_git", side_effect=responses),
            mock.patch.object(research_dag_module.subprocess, "run") as read_blob,
        ):
            with self.assertRaisesRegex(ResearchGitError, "exceeds"):
                research_dag_module._json_blob_at_commit(self.repo_path, "b" * 40, path)

        read_blob.assert_not_called()

    def test_pointer_metadata_has_an_aggregate_byte_budget(self) -> None:
        pointers = []
        for index, payload in enumerate((b"first payload", b"second payload")):
            source = self.root / f"pointer-{index}.bin"
            source.write_bytes(payload)
            pointers.append(
                add_research_object(
                    self.repo_path,
                    source,
                    logical_path=f"result-artifacts/pointer-{index}.bin",
                )
            )
        subprocess.run(
            ["git", "add", "research-objects"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add pointer budget fixtures"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        pointer_bytes = sum(item.pointer_path.stat().st_size for item in pointers)
        self.assertGreater(
            pointer_bytes - 1,
            max(item.pointer_path.stat().st_size for item in pointers),
        )

        with mock.patch.object(
            research_dag_module,
            "_MAX_ARTIFACT_POINTER_TOTAL_BYTES",
            pointer_bytes - 1,
        ):
            with self.assertRaisesRegex(ResearchGitError, "aggregate bytes"):
                research_dag_module._strict_pointer_records_at_commit(
                    self.repo_path, "HEAD"
                )

    def test_committed_ara_core_blob_budget_is_aggregate(self) -> None:
        ara = self.repo_path / "ara" / "core-budget"
        ara.mkdir(parents=True)
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "counts": {"nodes": 0},
        }
        (ara / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        write_manifest_lock(ara, manifest)
        (ara / "exploration_graph.json").write_text(
            json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
        )
        subprocess.run(
            ["git", "add", "ara/core-budget"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add core budget fixtures"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        first_two_bytes = sum(
            (ara / name).stat().st_size for name in ("manifest.json", "manifest.lock")
        )
        core_budget = {
            "bytes": 0,
            "files": 0,
            "maximum_bytes": first_two_bytes - 1,
            "maximum_files": 10,
            "label": "committed ARA core metadata",
        }

        with self.assertRaisesRegex(ResearchGitError, "aggregate bytes"):
            research_dag_module._committed_ara_projection(
                self.repo_path,
                "HEAD",
                "ara/core-budget/manifest.json",
                index=0,
                disclose=False,
                expected_manifest_hash=None,
                expected_graph_hash=None,
                required=True,
                core_budget=core_budget,
                load_node_metrics=False,
            )

    def test_ara_sidecar_reader_deduplicates_node_ids(self) -> None:
        ara = self.root / "deduplicated-ara"
        metrics = ara / "nodes" / "same" / "metrics.json"
        metrics.parent.mkdir(parents=True)
        metrics.write_text(json.dumps({"analysis": "once"}), encoding="utf-8")
        original_reader = research_dag_module._read_local_ara_file

        with mock.patch.object(
            research_dag_module,
            "_read_local_ara_file",
            wraps=original_reader,
        ) as reader:
            details = research_dag_module._local_ara_node_metrics(
                ara, {"nodes": [{"id": "same"}, {"id": "same"}]}
            )

        self.assertEqual(details["same"]["analysis"], "once")
        self.assertEqual(reader.call_count, 1)

    def test_ara_sidecar_reader_shares_aggregate_budget(self) -> None:
        first = self.root / "first-ara"
        second = self.root / "second-ara"
        first_metrics = first / "nodes" / "same" / "metrics.json"
        second_metrics = second / "nodes" / "other" / "metrics.json"
        first_metrics.parent.mkdir(parents=True)
        second_metrics.parent.mkdir(parents=True)
        first_metrics.write_text(json.dumps({"analysis": "first"}), encoding="utf-8")
        second_metrics.write_text(json.dumps({"analysis": "second"}), encoding="utf-8")
        shared_budget = {"bytes": 0, "files": 0, "cache": {}}
        original_reader = research_dag_module._read_local_ara_file

        with (
            mock.patch.object(
                research_dag_module,
                "_MAX_ARA_NODE_DETAILS_TOTAL_BYTES",
                first_metrics.stat().st_size,
            ),
            mock.patch.object(
                research_dag_module,
                "_read_local_ara_file",
                wraps=original_reader,
            ) as reader,
        ):
            first_details = research_dag_module._local_ara_node_metrics(
                first, {"nodes": [{"id": "same"}]}, detail_budget=shared_budget
            )
            second_details = research_dag_module._local_ara_node_metrics(
                second,
                {"nodes": [{"id": "other"}]},
                detail_budget=shared_budget,
            )

        self.assertEqual(first_details["same"]["analysis"], "first")
        self.assertEqual(second_details, {})
        self.assertEqual(reader.call_count, 1)

    def test_ara_sidecar_file_budget_counts_rejected_path_attempts(self) -> None:
        budget = {"bytes": 0, "files": 0, "cache": {}}
        with (
            mock.patch.object(research_dag_module, "_MAX_ARA_NODE_DETAIL_FILES", 1),
            mock.patch.object(
                research_dag_module,
                "_read_local_ara_file",
                side_effect=ResearchGitError("unsafe node path"),
            ) as read_node,
        ):
            details = research_dag_module._local_ara_node_metrics(
                self.root,
                {"nodes": [{"id": "first"}, {"id": "second"}]},
                detail_budget=budget,
            )

        self.assertEqual(details, {})
        self.assertEqual(budget["files"], 1)
        self.assertEqual(read_node.call_count, 1)

    def test_local_ara_sidecar_does_not_reuse_stale_worktree_cache(self) -> None:
        ara = self.root / "mutable-sidecar-ara"
        metrics = ara / "nodes" / "legacy" / "metrics.json"
        metrics.parent.mkdir(parents=True)
        graph = {"nodes": [{"id": "legacy"}], "edges": []}
        graph_sha256 = "sha256:" + "a" * 64
        budget = {"bytes": 0, "files": 0, "cache": {}}
        metrics.write_text(json.dumps({"analysis": "sidecar-v1"}), encoding="utf-8")

        first = research_dag_module._local_ara_node_metrics(
            ara,
            graph,
            detail_budget=budget,
            graph_sha256=graph_sha256,
        )["legacy"]
        metrics.write_text(json.dumps({"analysis": "sidecar-v2"}), encoding="utf-8")
        second = research_dag_module._local_ara_node_metrics(
            ara,
            graph,
            detail_budget=budget,
            graph_sha256=graph_sha256,
        )["legacy"]

        self.assertEqual(first["analysis"], "sidecar-v1")
        self.assertEqual(second["analysis"], "sidecar-v2")
        snapshot_key = research_dag_module._ARA_SOURCE_SNAPSHOT_KEY
        self.assertNotEqual(
            first[snapshot_key]["sidecar_sha256"],
            second[snapshot_key]["sidecar_sha256"],
        )

    def test_committed_ara_sidecar_rejects_pathological_json(self) -> None:
        graph = {"nodes": [{"id": "decision"}]}
        with mock.patch.object(
            research_dag_module,
            "_blob_bytes_at_commit",
            return_value=b"1" * 5000,
        ):
            integer_details = research_dag_module._committed_ara_node_metrics(
                self.repo_path,
                "a" * 40,
                "ara/demo",
                graph,
            )
        with (
            mock.patch.object(
                research_dag_module,
                "_blob_bytes_at_commit",
                return_value=b"{}",
            ),
            mock.patch.object(
                research_dag_module.json,
                "loads",
                side_effect=RecursionError,
            ),
        ):
            recursive_details = research_dag_module._committed_ara_node_metrics(
                self.repo_path,
                "a" * 40,
                "ara/demo",
                graph,
            )

        self.assertEqual(integer_details, {})
        self.assertEqual(recursive_details, {})

    def test_ara_sidecar_rejects_symlinked_node_metrics(self) -> None:
        ara = self.root / "ara-symlink-detail"
        node_dir = ara / "nodes" / "decision"
        node_dir.mkdir(parents=True)
        (ara / "exploration_graph.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "decision",
                            "stage": "decision",
                            "content_hash": "sha256:" + "c" * 64,
                        }
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        outside = self.root / "outside-node-metrics.json"
        outside.write_text(
            json.dumps({"analysis": "SHOULD-NOT-LEAK-FROM-SYMLINK"}),
            encoding="utf-8",
        )
        try:
            (node_dir / "metrics.json").symlink_to(outside)
        except OSError as exc:  # pragma: no cover - platform capability
            self.skipTest(f"symlinks unavailable: {exc}")

        exported = export_research_dag(
            self.repo_path, self.root / "ara-symlink-dag", ara_roots=[ara]
        )
        page = Path(exported["html"]).read_text(encoding="utf-8")
        detail_json = page.split(
            '<script id="detail-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]

        self.assertNotIn("SHOULD-NOT-LEAK-FROM-SYMLINK", page)
        self.assertNotIn("ara:0:decision", json.loads(detail_json))

    def test_external_ara_rejects_symlinked_core_files_without_leaking(self) -> None:
        outside = self.root / "outside-core.json"
        outside.write_text(
            '{"nodes":[],"edges":[],"secret":"CORE-SYMLINK-SECRET"}',
            encoding="utf-8",
        )
        for linked_name in (
            "exploration_graph.json",
            "manifest.json",
            "manifest.history.jsonl",
        ):
            with self.subTest(linked_name=linked_name):
                ara = self.root / ("symlink-core-" + linked_name.replace(".", "-"))
                ara.mkdir()
                if linked_name != "exploration_graph.json":
                    (ara / "exploration_graph.json").write_text(
                        json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
                    )
                try:
                    (ara / linked_name).symlink_to(outside)
                except OSError as exc:  # pragma: no cover - platform capability
                    self.skipTest(f"symlinks unavailable: {exc}")

                with self.assertRaises(ResearchGitError) as raised:
                    build_research_dag(self.repo_path, ara_roots=[ara])

                self.assertNotIn("CORE-SYMLINK-SECRET", str(raised.exception))

    def test_external_ara_rejects_symlinked_node_ancestors_without_leaking(
        self,
    ) -> None:
        for link_level in ("nodes", "node"):
            with self.subTest(link_level=link_level):
                ara = self.root / f"symlink-{link_level}-ancestor"
                ara.mkdir()
                (ara / "exploration_graph.json").write_text(
                    json.dumps(
                        {
                            "nodes": [
                                {
                                    "id": "decision",
                                    "analysis": "Graph-only safe analysis.",
                                }
                            ],
                            "edges": [],
                        }
                    ),
                    encoding="utf-8",
                )
                outside = self.root / f"outside-{link_level}"
                outside_metrics = outside / "decision" / "metrics.json"
                if link_level == "node":
                    outside_metrics = outside / "metrics.json"
                outside_metrics.parent.mkdir(parents=True)
                outside_metrics.write_text(
                    json.dumps({"analysis": "NODE-ANCESTOR-SECRET"}),
                    encoding="utf-8",
                )
                try:
                    if link_level == "nodes":
                        (ara / "nodes").symlink_to(outside, target_is_directory=True)
                    else:
                        (ara / "nodes").mkdir()
                        (ara / "nodes" / "decision").symlink_to(
                            outside, target_is_directory=True
                        )
                except OSError as exc:  # pragma: no cover - platform capability
                    self.skipTest(f"symlinks unavailable: {exc}")

                exported = export_research_dag(
                    self.repo_path,
                    self.root / f"symlink-{link_level}-ancestor-dag",
                    ara_roots=[ara],
                )
                page = Path(exported["html"]).read_text(encoding="utf-8")
                detail = self._detail_payload(exported)["ara:0:decision"]

                self.assertEqual(
                    detail["fields"], {"analysis": "Graph-only safe analysis."}
                )
                self.assertNotIn("NODE-ANCESTOR-SECRET", page)

    def test_external_ara_oversized_core_files_fail_closed(self) -> None:
        for target_name in (
            "exploration_graph.json",
            "manifest.json",
            "manifest.lock",
            "manifest.history.jsonl",
        ):
            with self.subTest(target_name=target_name):
                ara = self.root / ("oversized-" + target_name.replace(".", "-"))
                ara.mkdir()
                if target_name != "exploration_graph.json":
                    (ara / "exploration_graph.json").write_text(
                        json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
                    )
                with (ara / target_name).open("wb") as stream:
                    stream.write(b'{"padding":"')
                    stream.truncate(research_dag_module._MAX_ARA_CORE_JSON_BYTES + 1)

                with self.assertRaisesRegex(ResearchGitError, "exceeds"):
                    build_research_dag(self.repo_path, ara_roots=[ara])

    def test_external_ara_archive_reads_are_deduplicated_and_budgeted(self) -> None:
        ara = self.root / "archive-budget-ara"
        history_dir = ara / "history"
        history_dir.mkdir(parents=True)
        (ara / "exploration_graph.json").write_text(
            json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
        )
        first_hash = "sha256:" + "a" * 64
        second_hash = "sha256:" + "b" * 64
        archive_payload = json.dumps({"revision": "snapshot"}).encode("utf-8")
        for digest in (first_hash, second_hash):
            (history_dir / f"{digest.split(':', 1)[1]}.json").write_bytes(
                archive_payload
            )
        rows = [
            {"revision": 1, "base_hash": first_hash, "new_hash": second_hash},
            {"revision": 2, "base_hash": first_hash, "new_hash": second_hash},
        ]
        (ara / "manifest.history.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        original_reader = research_dag_module._read_local_json_at

        with mock.patch.object(
            research_dag_module,
            "_read_local_json_at",
            wraps=original_reader,
        ) as read_json:
            research_dag_module._read_ara(ara, index=0, disclose=False)

        archive_calls = [
            call
            for call in read_json.call_args_list
            if str(call.args[1]).startswith("history/")
        ]
        self.assertEqual(len(archive_calls), 1)

        rows[1]["base_hash"] = second_hash
        (ara / "manifest.history.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        with mock.patch.object(
            research_dag_module,
            "_MAX_ARA_ARCHIVED_MANIFEST_TOTAL_BYTES",
            len(archive_payload),
        ):
            with self.assertRaisesRegex(ResearchGitError, "exceeds"):
                research_dag_module._read_ara(ara, index=0, disclose=False)

    def test_manifest_history_revision_count_is_bounded(self) -> None:
        with mock.patch.object(research_dag_module, "_MAX_ARA_MANIFEST_REVISIONS", 1):
            rows, errors = research_dag_module._read_json_lines(
                '{"revision":1}\n{"revision":2}\n'
            )

        self.assertEqual(rows, [{"revision": 1}])
        self.assertEqual(len(errors), 1)
        self.assertIn("exceeds 1", errors[0])

    def test_committed_ara_manifest_enumeration_is_bounded_before_projection(
        self,
    ) -> None:
        listing = subprocess.CompletedProcess(
            [],
            0,
            "ara/first/manifest.json\0ara/second/manifest.json\0",
            "",
        )
        with (
            mock.patch.object(research_dag_module, "_MAX_COMMITTED_ARA_MANIFESTS", 1),
            mock.patch.object(
                research_dag_module, "_run_git_bounded", return_value=listing
            ) as bounded_git,
            mock.patch.object(
                research_dag_module, "_committed_ara_projection"
            ) as projection,
        ):
            with self.assertRaisesRegex(ResearchGitError, "bounded audit limit"):
                research_dag_module._committed_ara_sources(
                    self.repo_path,
                    "a" * 40,
                    checkpoint={"ara_manifests": []},
                    objects=[],
                    disclose=False,
                )

        projection.assert_not_called()
        enumeration_args = bounded_git.call_args.args[1]
        self.assertIn("-z", enumeration_args)
        self.assertEqual(enumeration_args[-2:], ["--", "ara"])

    def test_committed_ara_history_budget_is_global_and_fail_closed(self) -> None:
        current_commit = "a" * 40
        historical_commit = "b" * 40
        target_manifest = "sha256:" + "c" * 64
        listing = subprocess.CompletedProcess([], 0, "ara/demo/manifest.json\0", "")
        history = subprocess.CompletedProcess(
            [], 0, f"{current_commit}\n{historical_commit}\n", ""
        )

        def bounded_git(
            _repo: Path, args: Sequence[str], **_kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            return listing if args[0] == "ls-tree" else history

        current_projection = (
            [],
            [],
            {
                "manifest_hash": "sha256:" + "d" * 64,
                "manifest_integrity": {"ok": True, "valid_hashes": []},
                "graph_binding": {
                    "ok": True,
                    "graph_hash": "sha256:" + "e" * 64,
                },
                "repository_path": "ara/demo/manifest.json",
            },
        )
        with (
            mock.patch.object(
                research_dag_module, "_MAX_COMMITTED_ARA_HISTORY_COMMITS", 1
            ),
            mock.patch.object(
                research_dag_module, "_run_git_bounded", side_effect=bounded_git
            ),
            mock.patch.object(
                research_dag_module,
                "_committed_ara_projection",
                return_value=current_projection,
            ) as projection,
            mock.patch.object(
                research_dag_module, "_checkpoint_ara_expectation"
            ) as expectation,
        ):
            with self.assertRaisesRegex(ResearchGitError, "history exceeds"):
                research_dag_module._committed_ara_sources(
                    self.repo_path,
                    current_commit,
                    checkpoint={"ara_manifests": []},
                    objects=[{"provenance": {"ara_manifest_hash": target_manifest}}],
                    disclose=False,
                )

        self.assertEqual(projection.call_count, 1)
        expectation.assert_not_called()

    def test_artifact_hashes_must_match_resolved_list_label(self) -> None:
        snapshot_commit = self.repository.show()["commit"]
        artifact_bytes = subprocess.run(
            ["git", "show", f"{snapshot_commit}:question.md"],
            cwd=self.repo_path,
            check=True,
            capture_output=True,
        ).stdout
        artifact_hash = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
        conflicting_hash = "sha256:" + "f" * 64

        def details(mapped_hash: str) -> dict[str, Any]:
            return _gate_detail_views(
                [
                    {
                        "object_id": "gate",
                        "kind": "gate_decision",
                        "payload": {"decision": "hold"},
                        "relations": [{"type": "derived_from", "target": "attempt"}],
                    },
                    {
                        "object_id": "attempt",
                        "kind": "experiment_attempt",
                        "payload": {
                            "result_artifact_hashes": {"summary": mapped_hash},
                            "result_artifacts": [
                                {
                                    "logical_name": "summary",
                                    "path": "question.md",
                                    "content_hash": artifact_hash,
                                }
                            ],
                        },
                        "relations": [],
                    },
                ],
                repository_root=self.repo_path,
                html_directory=self.root / "dag",
                snapshot_commit=snapshot_commit,
                disclose=True,
            )

        matching = details(artifact_hash)["gate"]["artifacts"][0]
        conflicting = details(conflicting_hash)["gate"]["artifacts"][0]

        self.assertTrue(matching["binding_verified"])
        self.assertEqual(matching["binding_reason"], "verified_exact_hash")
        self.assertFalse(conflicting["binding_verified"])
        self.assertEqual(
            conflicting["binding_reason"], "conflicting_content_hash_declarations"
        )
        self.assertEqual(conflicting["label"], "summary")

    def test_artifact_binding_stops_after_verified_path(self) -> None:
        snapshot = self.repository.show()["commit"]
        hashes = {}
        for path in ("question.md", "research.yaml"):
            raw = subprocess.run(
                ["git", "show", f"{snapshot}:{path}"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
            ).stdout
            hashes[path] = "sha256:" + hashlib.sha256(raw).hexdigest()
        budget = {"files": 0, "bytes": 0, "cache": {}}
        with mock.patch.object(research_dag_module, "_MAX_ARTIFACT_BINDING_FILES", 2):
            rows = research_dag_module._artifact_detail_rows(
                {
                    "object_id": "attempt",
                    "kind": "experiment_attempt",
                    "payload": {"artifacts": {
                        "first": {
                            "path": "question.md",
                            "logical_path": "research.yaml",
                            "content_hash": hashes["question.md"],
                        },
                        "second": {
                            "path": "research.yaml",
                            "content_hash": hashes["research.yaml"],
                        },
                    }},
                },
                repository_root=self.repo_path,
                html_directory=self.root / "dag",
                snapshot_commit=snapshot,
                pointer_records={},
                binding_budget=budget,
            )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["binding_verified"] for row in rows), rows)
        self.assertEqual(budget["files"], 2)
        self.assertEqual(rows[0]["binding_checked_path"], "question.md")
        self.assertEqual(rows[1]["binding_checked_path"], "research.yaml")

    def test_artifact_binding_reason_identifies_the_checked_path(self) -> None:
        with mock.patch.object(
            research_dag_module,
            "_local_artifact_binding_status",
            side_effect=[
                (False, "artifact_not_found_at_snapshot"),
                (False, "content_hash_mismatch"),
            ],
        ):
            rows = research_dag_module._artifact_detail_rows(
                {
                    "object_id": "attempt",
                    "kind": "experiment_attempt",
                    "payload": {"artifacts": {"summary": {
                        "path": "missing.md",
                        "logical_path": "question.md",
                        "content_hash": "sha256:" + "f" * 64,
                    }}},
                },
                repository_root=self.repo_path,
                html_directory=self.root / "dag",
                snapshot_commit="a" * 40,
                pointer_records={},
                binding_budget={"files": 0, "bytes": 0, "cache": {}},
            )
        self.assertFalse(rows[0]["binding_verified"])
        self.assertEqual(rows[0]["binding_reason"], "content_hash_mismatch")
        self.assertEqual(rows[0]["binding_checked_path"], "question.md")

    def test_gate_details_follow_plural_payload_references(self) -> None:
        details = _gate_detail_views(
            [
                {
                    "object_id": "gate",
                    "kind": "gate_decision",
                    "payload": {"decision": "hold", "evidence_ids": ["evidence"]},
                    "relations": [],
                },
                {
                    "object_id": "evidence",
                    "kind": "evidence",
                    "payload": {"summary": "plural evidence reference"},
                    "relations": [],
                },
            ],
            repository_root=self.repo_path,
            html_directory=self.root / "dag",
            snapshot_commit=self.repository.show()["commit"],
            disclose=True,
        )

        lineage_ids = {row["source_id"] for row in details["gate"]["lineage"]}
        self.assertIn("evidence", lineage_ids)
        self.assertEqual(details["gate"]["lineage_meta"]["truncated"], False)

    def test_gate_lineage_reports_node_budget_truncation(self) -> None:
        with mock.patch.object(research_dag_module, "_MAX_GATE_LINEAGE_NODES", 1):
            details = _gate_detail_views(
                [
                    {
                        "object_id": "gate",
                        "kind": "gate_decision",
                        "payload": {"decision": "hold"},
                        "relations": [{"type": "derived_from", "target": "evidence"}],
                    },
                    {
                        "object_id": "evidence",
                        "kind": "evidence",
                        "payload": {"summary": "bounded lineage"},
                        "relations": [],
                    },
                ],
                repository_root=self.repo_path,
                html_directory=self.root / "dag",
                snapshot_commit=self.repository.show()["commit"],
                disclose=True,
            )

        meta = details["gate"]["lineage_meta"]
        self.assertTrue(meta["truncated"])
        self.assertEqual(meta["node_count"], 1)
        self.assertEqual(meta["truncation_reasons"], ["node_budget"])

    def test_gate_details_include_lineage_metrics_and_verified_artifact_bindings(
        self,
    ) -> None:
        artifact_source = self.root / "summary.html"
        artifact_source.write_text(
            "<script>document.body.dataset.executed = 'yes'</script>",
            encoding="utf-8",
        )
        pointer = add_research_object(
            self.repo_path,
            artifact_source,
            logical_path="result-artifacts/summary/summary.html",
        )
        artifact_hash = pointer.object_hash
        pointer_path = (
            pointer.pointer_path.resolve()
            .relative_to(self.repo_path.resolve())
            .as_posix()
        )
        attempt = self.repository.record(
            "experiment_attempt",
            {
                "summary": "Run the locked kill test.",
                "status": "completed",
                "boundary_condition": "One-state exploratory model only.",
                "metrics": {
                    "stable_superiority_passed": 0,
                    "maximum_action_mean_difference": 2.22e-16,
                },
                "result_artifact_hashes": {
                    "summary": artifact_hash,
                },
                "result_artifacts": {
                    "summary": {
                        "logical_name": "summary",
                        "logical_path": "result-artifacts/summary/summary.html",
                        "pointer_path": pointer_path,
                        "content_hash": artifact_hash,
                    },
                },
            },
            state="completed",
            relations=[{"type": "depends_on", "target": self.hypothesis_id}],
        )
        evidence = self.repository.record(
            "evidence",
            {
                "result": "The candidate missed its fixed threshold.",
                "measurement_hash": "sha256:" + "b" * 64,
                "metrics": {"successful_positive_budgets_total": 0},
            },
            state="completed",
            relations=[{"type": "derived_from", "target": attempt.object_id}],
        )
        gate = self.repository.record(
            "gate_decision",
            {
                "decision": "kill_candidate",
                "status": "failed_scientific_gate",
                "reason": "The fixed superiority rule failed in both worlds.",
                "scope": "Exploratory one-state model only.",
                "claim_promotion_allowed": False,
                "verification": {
                    "integrity_checks_passed": 5,
                    "manifest_lock_state": "clean",
                },
            },
            state="rejected",
            relations=[{"type": "derived_from", "target": evidence.object_id}],
            actor={"actor_id": "fixed-gate", "authority": "deterministic_gate"},
        )
        self.repository.commit(stage="decision", subject="record failed gate")

        baseline = build_research_dag(self.repo_path)
        output = self.repo_path / "05_analysis" / "dag"
        exported = export_research_dag(self.repo_path, output)
        exported_graph = json.loads(Path(exported["json"]).read_text(encoding="utf-8"))
        page = Path(exported["html"]).read_text(encoding="utf-8")
        detail_json = page.split(
            '<script id="detail-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        details = json.loads(detail_json)

        self.assertEqual(exported_graph["graph_hash"], baseline["graph_hash"])
        self.assertFalse(any("details" in node for node in exported_graph["nodes"]))
        gate_view = details[gate.object_id]
        self.assertEqual(
            gate_view["gate"],
            {
                "decision": "kill_candidate",
                "status": "failed_scientific_gate",
                "reason": "The fixed superiority rule failed in both worlds.",
                "scope": "Exploratory one-state model only.",
                "claim_promotion_allowed": False,
                "verification": {
                    "integrity_checks_passed": 5,
                    "manifest_lock_state": "clean",
                },
            },
        )
        lineage_by_source = {
            row["source_id"]: row["fields"] for row in gate_view["lineage"]
        }
        self.assertEqual(
            lineage_by_source[attempt.object_id]["summary"],
            "Run the locked kill test.",
        )
        self.assertEqual(
            lineage_by_source[attempt.object_id]["boundary_condition"],
            "One-state exploratory model only.",
        )
        metrics_by_source = {
            row["source_id"]: row["values"] for row in gate_view["metrics"]
        }
        self.assertEqual(
            metrics_by_source[evidence.object_id],
            {"successful_positive_budgets_total": 0},
        )
        self.assertEqual(
            metrics_by_source[attempt.object_id]["stable_superiority_passed"], 0
        )
        artifacts = {row["label"]: row for row in gate_view["artifacts"]}
        self.assertTrue(artifacts["summary"]["binding_verified"])
        self.assertIsNone(artifacts["summary"]["href"])
        self.assertFalse((output / "artifact-blobs").exists())
        self.assertNotIn("document.body.dataset.executed", page)
        self.assertEqual(
            artifacts["summary"]["logical_path"],
            "result-artifacts/summary/summary.html",
        )

        metadata_export = export_research_dag(
            self.repo_path,
            self.repo_path / "05_analysis" / "metadata-dag",
            disclose_summaries=False,
        )
        metadata_page = Path(metadata_export["html"]).read_text(encoding="utf-8")
        metadata_detail_json = metadata_page.split(
            '<script id="detail-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(metadata_detail_json), {})
        self.assertNotIn("The fixed superiority rule failed", metadata_page)

        unsafe_details = _gate_detail_views(
            [
                {
                    "object_id": "gate",
                    "kind": "gate_decision",
                    "payload": {"decision": "hold"},
                    "relations": [{"type": "derived_from", "target": "attempt"}],
                },
                {
                    "object_id": "attempt",
                    "kind": "experiment_attempt",
                    "payload": {
                        "result_artifacts": {
                            "unsafe": {
                                "logical_path": "/private/secret.json",
                                "pointer_path": "../outside.json",
                                "content_hash": artifact_hash,
                            }
                        }
                    },
                    "relations": [],
                },
            ],
            repository_root=self.repo_path,
            html_directory=output,
            snapshot_commit=str(baseline["commit"]),
            disclose=True,
        )
        unsafe = unsafe_details["gate"]["artifacts"][0]

        self.assertIsNone(unsafe["href"])
        self.assertFalse(unsafe["binding_verified"])
        self.assertIsNone(unsafe["logical_path"])
        self.assertIsNone(unsafe["pointer_path"])
        unsafe["href"] = "javascript:alert(1)"
        unsafe_page = render_research_dag_html(baseline, node_details=unsafe_details)
        self.assertNotIn("/private/secret.json", unsafe_page)
        self.assertNotIn("../outside.json", unsafe_page)
        self.assertNotIn("javascript:alert(1)", unsafe_page)
        self.assertNotIn(' download rel="noopener"', unsafe_page)
        self.assertNotIn('target="_blank"', unsafe_page)

        mismatched_details = _gate_detail_views(
            [
                {
                    "object_id": "gate",
                    "kind": "gate_decision",
                    "payload": {"decision": "hold"},
                    "relations": [{"type": "derived_from", "target": "attempt"}],
                },
                {
                    "object_id": "attempt",
                    "kind": "experiment_attempt",
                    "payload": {
                        "result_artifact_hashes": {"summary": "sha256:" + "f" * 64},
                        "result_artifacts": {
                            "summary": {
                                "logical_path": "result-artifacts/summary/summary.html",
                                "pointer_path": pointer_path,
                                "content_hash": artifact_hash,
                            }
                        },
                    },
                    "relations": [],
                },
            ],
            repository_root=self.repo_path,
            html_directory=output,
            snapshot_commit=str(baseline["commit"]),
            disclose=True,
        )
        self.assertIsNone(mismatched_details["gate"]["artifacts"][0]["href"])
        self.assertFalse(mismatched_details["gate"]["artifacts"][0]["binding_verified"])

        (self.repo_path / "question.md").write_text(
            "UNCOMMITTED replacement\n", encoding="utf-8"
        )
        dirty_export = export_research_dag(
            self.repo_path,
            self.repo_path / "05_analysis" / "dirty-dag",
        )
        dirty_page = Path(dirty_export["html"]).read_text(encoding="utf-8")
        dirty_detail_json = dirty_page.split(
            '<script id="detail-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        dirty_details = json.loads(dirty_detail_json)
        dirty_artifacts = {
            row["label"]: row for row in dirty_details[gate.object_id]["artifacts"]
        }
        self.assertTrue(dirty_artifacts["summary"]["binding_verified"])
        self.assertIsNone(dirty_artifacts["summary"]["href"])
        self.assertFalse(
            (self.repo_path / "05_analysis" / "dirty-dag" / "artifact-blobs").exists()
        )

    def test_export_clears_legacy_artifact_blobs_for_disclosed_and_metadata_runs(
        self,
    ) -> None:
        self._record_lineage()
        output = self.root / "reused-dag"
        stale = output / "artifact-blobs" / "aa" / ("b" * 62)
        stale.parent.mkdir(parents=True)
        stale.write_text("STALE-DISCLOSED-ARTIFACT", encoding="utf-8")

        export_research_dag(self.repo_path, output)

        self.assertFalse((output / "artifact-blobs").exists())
        stale.parent.mkdir(parents=True)
        stale.write_text("STALE-METADATA-ARTIFACT", encoding="utf-8")

        metadata = export_research_dag(self.repo_path, output, disclose_summaries=False)

        self.assertFalse((output / "artifact-blobs").exists())
        self.assertEqual(self._detail_payload(metadata), {})

    def test_export_rejects_symlinked_artifact_blob_namespace(self) -> None:
        self._record_lineage()
        output = self.root / "symlinked-artifact-output"
        output.mkdir()
        outside = self.root / "outside-artifact-directory"
        outside.mkdir()
        secret = outside / "must-survive.txt"
        secret.write_text("OUTSIDE-ARTIFACT-SECRET", encoding="utf-8")
        try:
            (output / "artifact-blobs").symlink_to(outside, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - platform capability
            self.skipTest(f"symlinks unavailable: {exc}")

        with self.assertRaisesRegex(ResearchGitError, "artifact namespace"):
            export_research_dag(self.repo_path, output)

        self.assertEqual(secret.read_text(encoding="utf-8"), "OUTSIDE-ARTIFACT-SECRET")

    def test_export_rejects_symlinked_destination_components(self) -> None:
        self._record_lineage()
        outside = self.root / "outside-dag-destination"
        outside.mkdir()
        stale = outside / "artifact-blobs" / "must-survive.txt"
        stale.parent.mkdir()
        stale.write_text("OUTSIDE-DAG-SECRET", encoding="utf-8")
        destination_link = self.root / "destination-link"
        ancestor_link = self.root / "ancestor-link"
        try:
            destination_link.symlink_to(outside, target_is_directory=True)
            ancestor_link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:  # pragma: no cover - platform capability
            self.skipTest(f"symlinks unavailable: {exc}")

        for destination in (destination_link, ancestor_link / "nested"):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ResearchGitError, "real directory"):
                    export_research_dag(self.repo_path, destination)
                self.assertEqual(
                    stale.read_text(encoding="utf-8"), "OUTSIDE-DAG-SECRET"
                )
                self.assertFalse((outside / "research-dag.json").exists())
                self.assertFalse((outside / "nested").exists())

    def test_export_keeps_writes_on_open_directory_if_path_is_replaced(self) -> None:
        self._record_lineage()
        output = self.root / "replaceable-dag-output"
        outside = self.root / "replacement-target"
        moved = self.root / "original-output-moved"
        outside.mkdir()
        original_clear = research_dag_module._clear_artifact_blob_namespace

        def clear_then_replace(path: Path, descriptor: int) -> None:
            original_clear(path, descriptor)
            path.rename(moved)
            path.symlink_to(outside, target_is_directory=True)

        with mock.patch.object(
            research_dag_module,
            "_clear_artifact_blob_namespace",
            side_effect=clear_then_replace,
        ):
            with self.assertRaisesRegex(ResearchGitError, "changed during export"):
                export_research_dag(self.repo_path, output)

        self.assertFalse((outside / "research-dag.json").exists())
        self.assertFalse((outside / "research-dag.html").exists())
        self.assertTrue((moved / "research-dag.json").is_file())
        self.assertTrue((moved / "research-dag.html").is_file())

    def test_gate_details_do_not_follow_claim_signal_relations(self) -> None:
        details = _gate_detail_views(
            [
                {
                    "object_id": "gate",
                    "kind": "gate_decision",
                    "payload": {"decision": "kill_candidate"},
                    "relations": [{"type": "derived_from", "target": "evidence"}],
                },
                {
                    "object_id": "evidence",
                    "kind": "evidence",
                    "payload": {"metrics": {"gate_metric": 0}},
                    "relations": [
                        {"type": "derived_from", "target": "attempt"},
                        {"type": "refutes", "target": "claim"},
                    ],
                },
                {
                    "object_id": "attempt",
                    "kind": "experiment_attempt",
                    "payload": {"metrics": {"attempt_metric": 1}},
                    "relations": [],
                },
                {
                    "object_id": "claim",
                    "kind": "claim",
                    "payload": {"metrics": {"claim_metric": 2}},
                    "relations": [
                        {"type": "depends_on", "target": "unrelated-attempt"}
                    ],
                },
                {
                    "object_id": "unrelated-attempt",
                    "kind": "experiment_attempt",
                    "payload": {"metrics": {"unrelated_metric": 3}},
                    "relations": [],
                },
            ],
            repository_root=self.repo_path,
            html_directory=self.repo_path / "05_analysis" / "dag",
            snapshot_commit="a" * 40,
            disclose=True,
        )

        metric_sources = {row["source_id"] for row in details["gate"]["metrics"]}
        self.assertEqual(metric_sources, {"evidence", "attempt"})
        lineage_sources = {row["source_id"] for row in details["gate"]["lineage"]}
        self.assertNotIn("claim", lineage_sources)
        self.assertNotIn("unrelated-attempt", lineage_sources)

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
        guide_payload = json.loads(output.getvalue())
        self.assertEqual(guide_payload["next_steps"][0]["code"], "run_experiment")
        run_step = guide_payload["next_steps"][0]
        self.assertEqual(run_step["command"], "xscientist record {workspace}")
        self.assertFalse(run_step["action"]["executable_after_binding"])
        self.assertEqual(run_step["action"]["input_binding"]["mode"], "interactive")
        self.assertTrue(run_step["action"]["input_binding"]["required"])

        chinese_guide = build_research_guide(cli_repo, language="zh", command_repo=".")
        self.assertEqual(
            chinese_guide["next_steps"][0]["command"],
            "xscientist record . --lang zh",
        )

        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                ["dag", "--repo", str(cli_repo), "--metadata-only", "--json"]
            )
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.getvalue())["content_disclosed"])

    def test_guided_writers_print_copyable_next_without_changing_json(self) -> None:
        recorded = ResearchObjectResult(
            created=True,
            path=self.repo_path / "research-objects" / "fake.json",
            object_id="rso-fake",
            qualified_id=None,
            object_hash="sha256:" + "f" * 64,
            kind="test",
            state="draft",
        )
        result = {"object": recorded, "related": [], "checkpoint": None}
        cases = [
            (
                "xscientist.research_commands.save_hypothesis",
                ["hypothesis", "H", "--falsifier", "not H"],
            ),
            (
                "xscientist.research_commands.save_research_plan",
                ["plan", "rso-h", "Plan"],
            ),
            (
                "xscientist.research_commands.save_experiment",
                ["experiment", "Run", "--status", "completed"],
            ),
            (
                "xscientist.research_commands.save_evidence",
                ["evidence", "Result", "--attempt", "rso-a"],
            ),
            (
                "xscientist.research_commands.save_inference",
                ["infer", "Conclusion", "--premise", "rso-e", "--warrant", "W"],
            ),
            (
                "xscientist.research_commands.save_review",
                [
                    "review",
                    "Review",
                    "--evaluates",
                    "rso-i",
                    "--verifier",
                    "human:reviewer",
                    "--decision",
                    "hold",
                ],
            ),
            (
                "xscientist.research_commands.save_claim",
                ["claim", "Claim", "--evidence", "rso-i"],
            ),
        ]
        expected_next = f"Next: xscientist research guide --repo {self.repo_path}"
        for target, command in cases:
            with (
                self.subTest(command=command[0]),
                mock.patch(target, return_value=result),
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = research_main([*command, "--repo", str(self.repo_path)])
                self.assertEqual(code, 0)
                self.assertIn(expected_next, output.getvalue())

        with mock.patch(
            "xscientist.research_commands.save_hypothesis", return_value=result
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                code = research_main(
                    [
                        "hypothesis",
                        "H-json",
                        "--falsifier",
                        "not H",
                        "--repo",
                        str(self.repo_path),
                        "--json",
                    ]
                )
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(set(payload), {"object", "related_objects", "checkpoint"})


if __name__ == "__main__":
    unittest.main()
