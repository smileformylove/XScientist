from __future__ import annotations

import json
import io
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import redirect_stdout

from ai_scientist.protocol.research_vcs import (
    RESEARCH_SEMANTIC_PROFILE_SCHEMA,
    ResearchObjectError,
    build_research_object,
    research_profile_status,
    validate_research_object,
)
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.research_integrity import (
    build_preregistration,
    lock_preregistration,
)
from xscientist import ResearchLifecycle, ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_git import ResearchGitError
from xscientist.research_git import add_research_object
from xscientist.research_semantics import claim_scope_hash, normalize_claim_scope


class ResearchObjectProtocolTests(unittest.TestCase):
    def test_empty_semantic_payload_is_rejected_before_persistence(self) -> None:
        with self.assertRaisesRegex(ResearchObjectError, "payload must not be empty"):
            build_research_object(kind="hypothesis", payload={})

    def test_identity_is_deterministic_and_relations_are_canonical(self) -> None:
        first = build_research_object(
            kind="claim",
            state="verified",
            payload={"statement": "H1 improves the registered metric."},
            relations=[
                {"type": "supports", "target": "rso-bbbbbbbbbbbbbbbb"},
                {"type": "depends_on", "target": "rso-aaaaaaaaaaaaaaaa"},
                {"type": "supports", "target": "rso-bbbbbbbbbbbbbbbb"},
            ],
            created_at="2026-01-01T00:00:00+00:00",
        )
        second = build_research_object(
            kind="claim",
            state="verified",
            payload={"statement": "H1 improves the registered metric."},
            relations=list(reversed(first["relations"])),
            created_at="2026-02-01T00:00:00+00:00",
        )

        self.assertEqual(first["object_id"], second["object_id"])
        self.assertEqual(first["content_hash"], second["content_hash"])
        self.assertNotEqual(first["created_at"], second["created_at"])
        self.assertEqual(len(first["relations"]), 2)
        self.assertEqual(first["relations"][0]["type"], "depends_on")

    def test_validation_detects_content_tampering(self) -> None:
        payload = build_research_object(
            kind="hypothesis",
            payload={"statement": "H1"},
        )
        payload["payload"]["statement"] = "H2"

        with self.assertRaisesRegex(ResearchObjectError, "hash mismatch"):
            validate_research_object(payload)

    def test_builtin_objects_bind_a_validated_semantic_profile(self) -> None:
        payload = build_research_object(
            kind="inference",
            payload={"statement": "The observed effect supports H1."},
            relations=[{"type": "has_premise", "target": "rso-aaaaaaaaaaaaaaaa"}],
        )

        status = research_profile_status(payload)
        self.assertTrue(status["declared"])
        self.assertTrue(status["validator_available"])
        self.assertEqual(validate_research_object(payload), payload)

    def test_extension_profile_is_storable_but_not_locally_verified(self) -> None:
        core = {
            "schema": RESEARCH_SEMANTIC_PROFILE_SCHEMA,
            "uri": "urn:example:wet-lab-profile:v1",
            "version": "1.0.0",
            "kinds": ["assay_result"],
            "relations": ["urn:example:usesMaterial"],
        }
        profile = {**core, "schema_digest": canonical_content_hash(core)}
        payload = build_research_object(
            kind="assay_result",
            payload={"result": "growth inhibited", "assay": "MIC"},
            semantic_profile=profile,
        )

        self.assertEqual(validate_research_object(payload), payload)
        self.assertFalse(research_profile_status(payload)["validator_available"])
        with self.assertRaisesRegex(ResearchObjectError, "does not declare relation"):
            build_research_object(
                kind="assay_result",
                payload={"result": "growth inhibited"},
                relations=[
                    {"type": "urn:example:other", "target": "rso-aaaaaaaaaaaaaaaa"}
                ],
                semantic_profile=profile,
            )


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ResearchRepositoryTests(unittest.TestCase):
    def _init(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            question="Does the intervention improve the registered metric?",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )

    def test_record_list_load_commit_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)

            question = repository.record(
                "question",
                {"text": "Does the intervention improve the registered metric?"},
            )
            repeated = repository.record(
                "question",
                {"text": "Does the intervention improve the registered metric?"},
            )
            hypothesis = repository.record(
                "hypothesis",
                {"statement": "The intervention increases the metric."},
                relations=[
                    {"type": "depends_on", "target": question.object_id},
                ],
            )

            self.assertTrue(question.created)
            self.assertFalse(repeated.created)
            self.assertEqual(question.object_id, repeated.object_id)
            self.assertEqual(repository.get(question.object_id)["kind"], "question")
            self.assertEqual(
                [item["kind"] for item in repository.objects()],
                ["hypothesis", "question"],
            )
            self.assertEqual(
                repository.objects(kind="hypothesis")[0]["object_id"],
                hypothesis.object_id,
            )

            checkpoint = repository.commit(
                stage="ideation",
                subject="record question and hypothesis",
            )
            verification = repository.fsck()

            self.assertTrue(checkpoint.committed)
            self.assertTrue(verification["ok"], verification["errors"])
            self.assertIn(
                str(question.path.relative_to(repository.path)),
                checkpoint.staged_paths,
            )
            semantic = repository.diff("HEAD~1", "HEAD")["semantic"]
            self.assertEqual(
                {item["object_id"] for item in semantic["research_objects"]["added"]},
                {question.object_id, hypothesis.object_id},
            )
            blame = repository.blame(hypothesis.object_id)
            self.assertEqual(blame["object"]["content_hash"], hypothesis.object_hash)
            self.assertEqual(blame["origin"]["checkpoint_id"], checkpoint.checkpoint_id)

    def test_object_selectors_resolve_latest_kind_and_unique_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")
            first = repository.record("hypothesis", {"statement": "H1"})
            second = repository.record("hypothesis", {"statement": "H2"})

            self.assertEqual(repository.resolve("@latest:hypothesis"), second.object_id)
            self.assertEqual(
                repository.get(first.object_id[:10])["object_id"], first.object_id
            )
            plan = repository.record(
                "research_plan",
                {"summary": "Test latest hypothesis"},
                relations=[{"type": "depends_on", "target": "@latest:hypothesis"}],
            )
            self.assertEqual(
                repository.get(plan.object_id)["relations"][0]["target"],
                second.object_id,
            )

    def test_privacy_gate_rejects_secret_without_persisting_it(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            secret = "sk-" + "q" * 40

            with self.assertRaisesRegex(
                ResearchGitError, "privacy gate refused"
            ) as caught:
                repository.record("evidence", {"credential": secret})

            self.assertNotIn(secret, str(caught.exception))
            self.assertEqual(repository.objects(), [])

    def test_native_stage_commits_only_the_selected_research_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            question = repository.record("question", {"text": "Q1"})
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            question_path = question.path.relative_to(repository.path).as_posix()
            hypothesis_path = hypothesis.path.relative_to(repository.path).as_posix()

            staged = repository.stage([question_path])
            status = repository.status()
            checkpoint = repository.commit(
                stage="ideation",
                subject="record only the selected question",
                staged_only=True,
            )

            self.assertEqual(staged.paths, (question_path,))
            self.assertEqual(status["research_stage"]["paths"], [question_path])
            self.assertIn(question_path, checkpoint.staged_paths)
            self.assertNotIn(hypothesis_path, checkpoint.staged_paths)
            self.assertEqual(repository.status()["research_stage"]["paths"], [])
            self.assertIn(hypothesis_path, repository.status()["eligible_changes"])

    def test_native_stage_detects_content_changed_after_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            result = repository.record("hypothesis", {"statement": "H1"})
            relative = result.path.relative_to(repository.path).as_posix()
            repository.stage([relative])
            result.path.write_text(result.path.read_text(encoding="utf-8") + " ")

            with self.assertRaisesRegex(ResearchGitError, "changed after selection"):
                repository.commit(
                    stage="ideation",
                    subject="must not commit stale selection",
                    staged_only=True,
                )

    def test_native_stage_binds_only_selected_large_object_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            repository = self._init(root)
            first_source = base / "first.bin"
            second_source = base / "second.bin"
            first_source.write_bytes(b"first evidence")
            second_source.write_bytes(b"second evidence")
            first = add_research_object(
                root, first_source, logical_path="data/first.bin"
            )
            second = add_research_object(
                root, second_source, logical_path="data/second.bin"
            )
            first_path = first.pointer_path.relative_to(repository.path).as_posix()
            second_path = second.pointer_path.relative_to(repository.path).as_posix()

            repository.stage([first_path])
            checkpoint = repository.commit(
                stage="evidence",
                subject="bind selected evidence only",
                staged_only=True,
            )
            verification = repository.fsck()

            self.assertIn(
                first.object_hash, repository.show()["checkpoint"]["object_refs"]
            )
            self.assertNotIn(
                second.object_hash, repository.show()["checkpoint"]["object_refs"]
            )
            self.assertIn(first_path, checkpoint.staged_paths)
            self.assertNotIn(second_path, checkpoint.staged_paths)
            self.assertTrue(verification["ok"], verification["errors"])

    def test_research_branches_and_tags_use_scientific_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)

            forked = repository.fork("hypothesis/h1")
            result = repository.record("hypothesis", {"statement": "H1"})
            repository.commit(stage="ideation", subject="explore H1")
            tag = repository.tag("result/h1-v1")
            branches = repository.branches()
            tags = repository.tags()

            self.assertTrue(forked["current"])
            self.assertEqual(repository.status()["branch"], "hypothesis/h1")
            self.assertEqual(
                [item["name"] for item in branches],
                ["hypothesis/h1", "main"],
            )
            self.assertEqual(
                tag["checkpoint_id"], repository.show()["checkpoint"]["checkpoint_id"]
            )
            self.assertEqual(tags[0]["name"], "result/h1-v1")
            self.assertEqual(tags[0]["checkpoint_id"], tag["checkpoint_id"])
            self.assertTrue(result.path.is_file())
            with self.assertRaisesRegex(ResearchGitError, "already exists"):
                repository.tag("result/h1-v1")

    def test_switch_refuses_uncommitted_research(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.fork("alternative", switch=False)
            repository.record("hypothesis", {"statement": "uncommitted"})

            with self.assertRaisesRegex(ResearchGitError, "clean working state"):
                repository.switch("alternative")

    def test_branch_rename_delete_restore_and_semantic_revert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.fork("draft", switch=False)
            renamed = repository.rename_branch("draft", "challenge")
            self.assertEqual(renamed["name"], "challenge")
            deleted = repository.delete_branch("challenge")
            self.assertTrue(deleted["deleted"])

            claim_path = root / "claims" / "result.md"
            claim_path.write_text("version one\n", encoding="utf-8")
            first = repository.commit(stage="claim", subject="record version one")
            claim_path.write_text("version two\n", encoding="utf-8")
            second = repository.commit(stage="claim", subject="record version two")

            restored = repository.restore(first.commit or "HEAD~1", "claims/result.md")
            self.assertEqual(restored["paths"], ["claims/result.md"])
            self.assertEqual(claim_path.read_text(encoding="utf-8"), "version one\n")
            restored_checkpoint = repository.commit(
                stage="restore", subject="restore version one"
            )

            reverted = repository.revert(restored_checkpoint.commit or "HEAD")
            self.assertEqual(reverted["reverted"], restored_checkpoint.commit)
            self.assertTrue(reverted["checkpoint"]["committed"])
            self.assertEqual(claim_path.read_text(encoding="utf-8"), "version two\n")
            self.assertTrue(repository.fsck()["ok"])

    def test_clean_research_merge_retains_both_scientific_parents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.fork("alternative")
            hypothesis = repository.record("hypothesis", {"statement": "H-alt"})
            branch_checkpoint = repository.commit(
                stage="ideation", subject="explore alternative"
            )
            repository.switch("main")

            preview = repository.merge_preview("alternative")
            merged = repository.merge("alternative")
            checkpoint = repository.show()["checkpoint"]
            verification = repository.fsck()

            self.assertTrue(preview["clean"], preview["conflicts"])
            self.assertEqual(merged.target, "main")
            self.assertEqual(len(checkpoint["parent_checkpoint_hashes"]), 2)
            self.assertIn(
                branch_checkpoint.content_hash, checkpoint["parent_checkpoint_hashes"]
            )
            self.assertEqual(repository.get(hypothesis.object_id)["kind"], "hypothesis")
            self.assertTrue(verification["ok"], verification["errors"])

    def test_merge_preflight_blocks_opposed_scientific_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            repository.commit(stage="ideation", subject="record H1")
            repository.fork("challenge")
            repository.record(
                "evidence",
                {"result": "negative"},
                relations=[{"type": "refutes", "target": hypothesis.object_id}],
            )
            repository.commit(stage="evidence", subject="challenge H1")
            repository.switch("main")
            repository.record(
                "evidence",
                {"result": "positive"},
                relations=[{"type": "supports", "target": hypothesis.object_id}],
            )
            repository.commit(stage="evidence", subject="support H1")

            preview = repository.merge_preview("challenge")

            self.assertFalse(preview["clean"])
            self.assertIn(
                "opposed_evidence",
                {item["type"] for item in preview["conflicts"]},
            )
            opposed = next(
                item
                for item in preview["conflicts"]
                if item["type"] == "opposed_evidence"
            )
            self.assertTrue(opposed["conflict_id"].startswith("rvc-"))
            self.assertEqual(opposed["severity"], "blocking")
            self.assertGreaterEqual(len(opposed["resolution"]), 2)
            with self.assertRaisesRegex(ResearchGitError, "conflict resolution"):
                repository.merge("challenge")

            merged = repository.merge("challenge", preserve_conflicts=True)
            self.assertEqual(len(merged.resolution_objects), 1)
            resolution = repository.get(merged.resolution_objects[0])
            self.assertEqual(resolution["kind"], "gate_decision")
            self.assertEqual(resolution["state"], "rejected")
            self.assertEqual(resolution["payload"]["decision"], "hold")
            self.assertFalse(resolution["payload"]["claim_promotion_allowed"])
            self.assertEqual(
                resolution["payload"]["merge_conflict_id"],
                opposed["conflict_id"],
            )
            self.assertTrue(repository.fsck()["ok"])

    def test_merge_does_not_confuse_disjoint_scopes_with_a_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            repository.commit(stage="ideation", subject="record H1")
            repository.fork("children")
            child_scope = normalize_claim_scope({"population": "children"})
            repository.record(
                "evidence",
                {
                    "result": "negative in children",
                    "scope": child_scope,
                    "scope_hash": claim_scope_hash(child_scope),
                },
                relations=[{"type": "refutes", "target": hypothesis.object_id}],
            )
            repository.commit(stage="evidence", subject="challenge H1 in children")
            repository.switch("main")
            adult_scope = normalize_claim_scope({"population": "adults"})
            repository.record(
                "evidence",
                {
                    "result": "positive in adults",
                    "scope": adult_scope,
                    "scope_hash": claim_scope_hash(adult_scope),
                },
                relations=[{"type": "supports", "target": hypothesis.object_id}],
            )
            repository.commit(stage="evidence", subject="support H1 in adults")

            preview = repository.merge_preview("children")

            self.assertTrue(preview["clean"], preview["conflicts"])

    def test_decision_policy_checkpoints_before_fork_without_mutating(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            initial_head = repository.status()["head"]
            repository.record("hypothesis", {"statement": "H2"})

            decision = repository.decide(
                event="hypothesis",
                name="competing mechanism",
                competing_hypothesis=True,
            )

            self.assertEqual(
                [item["action"] for item in decision["actions"]],
                ["checkpoint", "fork"],
            )
            self.assertEqual(
                decision["actions"][1]["branch"],
                "hypothesis/competing-mechanism",
            )
            self.assertFalse(decision["mutates_repository"])
            self.assertTrue(decision["trace_required"])
            self.assertEqual(repository.status()["head"], initial_head)
            self.assertEqual(repository.status()["branch"], "main")

    def test_first_hypothesis_checkpoints_without_unnecessary_fork(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.record("hypothesis", {"statement": "H1"})

            decision = repository.decide(event="hypothesis", name="primary")

            self.assertEqual(
                [item["action"] for item in decision["actions"]],
                ["checkpoint"],
            )

    def test_technology_tree_preserves_relations_without_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            supported = repository.record("hypothesis", {"statement": "H1"})
            open_hypothesis = repository.record("hypothesis", {"statement": "H2"})
            evidence = repository.record(
                "evidence",
                {"result": "positive"},
                state="verified",
                relations=[{"type": "supports", "target": supported.object_id}],
            )

            tree = repository.technology_tree()

            self.assertTrue(tree["integrity"]["ok"])
            self.assertFalse(tree["payloads_disclosed"])
            self.assertEqual(tree["counts"]["nodes"], 3)
            self.assertEqual(tree["counts"]["edges"], 1)
            self.assertTrue(all("payload" not in node for node in tree["nodes"]))
            self.assertEqual(
                tree["edges"][0],
                {
                    "source": evidence.object_id,
                    "target": supported.object_id,
                    "type": "supports",
                    "role": "",
                },
            )
            self.assertEqual(
                tree["frontier"],
                [
                    {
                        "object_id": open_hypothesis.object_id,
                        "kind": "hypothesis",
                        "classification": "open",
                    }
                ],
            )
            self.assertLess(
                tree["topological_order"].index(supported.object_id),
                tree["topological_order"].index(evidence.object_id),
            )

    def test_technology_tree_unifies_all_research_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            shared = repository.record("hypothesis", {"statement": "shared"})
            repository.commit(stage="ideation", subject="record shared hypothesis")
            repository.fork("hypothesis/alternative")
            alternative = repository.record("hypothesis", {"statement": "alternative"})
            repository.commit(stage="ideation", subject="record alternative")
            repository.switch("main")

            tree = repository.technology_tree()
            nodes = {item["object_id"]: item for item in tree["nodes"]}

            self.assertEqual(
                nodes[shared.object_id]["research_lines"],
                ["hypothesis/alternative", "main"],
            )
            self.assertEqual(
                nodes[alternative.object_id]["research_lines"],
                ["hypothesis/alternative"],
            )

    def test_cli_runs_native_record_stage_commit_and_branch_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            commands = [
                [
                    "init",
                    str(root),
                    "--question",
                    "Does H1 improve the metric?",
                    "--git-user-name",
                    "Research Test",
                    "--git-user-email",
                    "research@example.invalid",
                ],
                [
                    "record",
                    "hypothesis",
                    "--repo",
                    str(root),
                    "--data",
                    '{"statement":"H1"}',
                ],
                ["stage", "--repo", str(root), "--all"],
                [
                    "checkpoint",
                    "--repo",
                    str(root),
                    "--stage",
                    "ideation",
                    "--subject",
                    "record H1",
                    "--staged",
                ],
                ["branch", "alternative", "--repo", str(root)],
            ]

            for command in commands:
                with self.subTest(command=command), redirect_stdout(io.StringIO()):
                    self.assertEqual(research_main(command), 0)

            repository = ResearchRepository(root)
            self.assertEqual(
                repository.objects(kind="hypothesis")[0]["payload"], {"statement": "H1"}
            )
            self.assertEqual(
                [item["name"] for item in repository.branches()],
                ["alternative", "main"],
            )

    def test_cli_json_errors_are_structured_and_raw_verified_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            output = io.StringIO()
            with redirect_stdout(output):
                status = research_main(
                    [
                        "record",
                        "claim",
                        "--repo",
                        str(root),
                        "--state",
                        "verified",
                        "--data",
                        '{"statement":"unsafe"}',
                        "--json",
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(status, 2)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["category"], "research_vcs_error")
            self.assertIn("cannot create verified", payload["error"]["message"])

    def test_evidence_gated_lifecycle_reaches_final_manuscript(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            lifecycle = ResearchLifecycle(repository)
            hypothesis_payload = {
                "idea_id": "idea-1",
                "title": "Reliability study",
                "core_hypothesis": "The intervention improves accuracy.",
                "failure_criteria": ["Accuracy does not improve."],
            }
            plan_payload = {
                "plan_id": "plan-1",
                "tasks": [
                    {
                        "task_id": "task-1",
                        "dataset": "benchmark-v1",
                        "metric": "accuracy",
                        "baseline": "baseline-a",
                    }
                ],
            }
            registration = lock_preregistration(
                build_preregistration(hypothesis_payload, plan_payload),
                split_hashes={"task-1": "sha256:" + "a" * 64},
                registered_by="planner",
            )

            planning = lifecycle.planning(
                hypothesis=hypothesis_payload,
                plan=plan_payload,
                preregistration=registration,
            )
            attempt = lifecycle.experiment_attempt(
                {
                    "record_id": "run-1",
                    "status": "success",
                    "study_phase": "confirmatory",
                    "dataset_split_hash": "sha256:" + "a" * 64,
                    "seed": 11,
                },
                preregistration_id=planning["preregistration"].object_id,
                plan_id=planning["plan"].object_id,
            )
            evidence = lifecycle.evidence(
                {"effect": 0.04, "metric": "accuracy"},
                attempt_ids=[attempt["attempt"].object_id],
                supports=[planning["hypothesis"].object_id],
                verified=True,
                verifier_id="evidence-verifier",
            )
            evaluation = lifecycle.evaluation(
                {
                    "status": "verified",
                    "claim_promotion_allowed": True,
                    "required_failures": [],
                    "report_hash": "sha256:" + "b" * 64,
                },
                evaluates=[evidence["evidence"].object_id],
                verifier_id="independent-verifier",
            )
            claim = lifecycle.claim(
                {"statement": "The intervention improves accuracy."},
                evidence_ids=[evidence["evidence"].object_id],
                gate_id=evaluation["gate"].object_id,
                verified=True,
            )
            manuscript = lifecycle.manuscript(
                {"title": "A Confirmatory Reliability Study", "version": 1},
                claim_ids=[claim["claim"].object_id],
                gate_id=evaluation["gate"].object_id,
                final=True,
            )

            self.assertEqual(manuscript["manuscript"].state, "completed")
            self.assertTrue(repository.fsck()["ok"])
            context = repository.get(evaluation["context"].object_id)
            self.assertTrue(context["payload"]["complete"])
            self.assertEqual(
                evaluation["gate"].object_id,
                repository.resolve("@latest:gate_decision"),
            )
            self.assertIn(
                evaluation["context"].object_id,
                {
                    relation["target"]
                    for relation in repository.get(evaluation["gate"].object_id)[
                        "relations"
                    ]
                    if relation.get("role") == "decision_context"
                },
            )
            self.assertEqual(
                {item["kind"] for item in repository.objects()},
                {
                    "claim",
                    "context_snapshot",
                    "evidence",
                    "experiment_attempt",
                    "gate_decision",
                    "hypothesis",
                    "manuscript",
                    "preregistration",
                    "research_plan",
                    "review",
                },
            )

    def test_confirmatory_attempt_requires_locked_registration_and_failures_persist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            lifecycle = ResearchLifecycle(self._init(root))
            hypothesis = {
                "title": "H1",
                "core_hypothesis": "H1",
                "failure_criteria": ["H1 fails"],
            }
            plan = {
                "plan_id": "p1",
                "tasks": [
                    {
                        "task_id": "t1",
                        "dataset": "d1",
                        "metric": "accuracy",
                        "baseline": "b1",
                    }
                ],
            }
            draft = build_preregistration(hypothesis, plan)
            planning = lifecycle.planning(
                hypothesis=hypothesis,
                plan=plan,
                preregistration=draft,
            )

            with self.assertRaisesRegex(ResearchGitError, "locked preregistration"):
                lifecycle.experiment_attempt(
                    {"status": "success", "study_phase": "confirmatory"},
                    preregistration_id=planning["preregistration"].object_id,
                    commit=False,
                )

            timed_out = lifecycle.experiment_attempt(
                {"status": "timeout", "error_class": "deadline"},
                plan_id=planning["plan"].object_id,
            )
            self.assertEqual(timed_out["attempt"].state, "timed_out")
            self.assertEqual(
                lifecycle.repository.get(timed_out["attempt"].object_id)["state"],
                "timed_out",
            )

    def test_damaged_object_is_never_returned_as_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            result = repository.record("metric", {"name": "accuracy", "value": 1.0})
            payload = json.loads(result.path.read_text(encoding="utf-8"))
            payload["payload"]["value"] = 0.0
            result.path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ResearchGitError, "damaged"):
                repository.get(result.object_id)
            with self.assertRaisesRegex(ResearchGitError, "damaged"):
                repository.objects()

    def test_concurrent_idempotent_recording_has_no_loss_or_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)

            with ThreadPoolExecutor(max_workers=8) as pool:
                repeated = list(
                    pool.map(
                        lambda _index: repository.record(
                            "evidence",
                            {"result": "same deterministic evidence"},
                        ),
                        range(24),
                    )
                )
                distinct = list(
                    pool.map(
                        lambda index: repository.record(
                            "metric",
                            {"name": "score", "seed": index, "value": index / 10},
                        ),
                        range(16),
                    )
                )

            self.assertEqual(sum(item.created for item in repeated), 1)
            self.assertEqual(len({item.object_id for item in repeated}), 1)
            self.assertEqual(len({item.object_id for item in distinct}), 16)
            self.assertEqual(len(repository.objects()), 17)


if __name__ == "__main__":
    unittest.main()
