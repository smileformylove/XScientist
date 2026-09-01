from __future__ import annotations

import json
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

import yaml

from ai_scientist.protocol.research_vcs import (
    RESEARCH_OBJECT_IDENTITY_PROFILE,
    RESEARCH_SEMANTIC_PROFILE_SCHEMA,
    ResearchObjectError,
    build_research_object,
    research_profile_status,
    validate_research_object,
)
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash
from ai_scientist.utils.research_integrity import (
    build_preregistration,
    lock_preregistration,
)
from xscientist import ResearchLifecycle, ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_git import (
    ResearchGitError,
    _create_checkpoint_locked,
    research_object_origin_checkpoint,
)
from xscientist.research_git import add_research_object
from xscientist.research_authority import require_independent_evaluator
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

    def test_v2_envelope_blocks_timestamp_tampering_and_profile_downgrade(self) -> None:
        payload = build_research_object(
            kind="hypothesis",
            payload={"statement": "H1"},
        )
        self.assertEqual(
            RESEARCH_OBJECT_IDENTITY_PROFILE,
            "xscientist.research-object-identity.v2",
        )
        tampered_time = {**payload, "created_at": "2030-01-01T00:00:00Z"}
        with self.assertRaisesRegex(ResearchObjectError, "envelope hash mismatch"):
            validate_research_object(tampered_time)

        downgraded = {
            **payload,
            "identity_profile": "xscientist.research-object-identity.v1",
        }
        downgraded.pop("envelope_hash")
        with self.assertRaisesRegex(ResearchObjectError, "content hash mismatch"):
            validate_research_object(downgraded)

        with self.assertRaisesRegex(ResearchObjectError, "UTC"):
            build_research_object(
                kind="hypothesis",
                payload={"statement": "H2"},
                created_at="2026-01-01T00:00:00+08:00",
            )

    def test_legacy_v1_object_remains_readable_without_time_authority(self) -> None:
        legacy = build_research_object(
            kind="hypothesis",
            payload={"statement": "legacy H1"},
        )
        legacy["identity_profile"] = "xscientist.research-object-identity.v1"
        legacy.pop("envelope_hash")
        identity_core = {
            key: value
            for key, value in legacy.items()
            if key
            not in {
                "object_id",
                "qualified_id",
                "identity_profile",
                "created_at",
                "content_hash",
            }
        }
        legacy_hash = content_hash(identity_core)
        legacy["content_hash"] = legacy_hash
        legacy["object_id"] = f"rso-{legacy_hash.split(':', 1)[1][:16]}"
        legacy["qualified_id"] = (
            "urn:xscientist:research-object:sha256:" + legacy_hash.split(":", 1)[1]
        )

        self.assertEqual(validate_research_object(legacy), legacy)

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

    def test_builtin_relation_rejects_arbitrary_target_identifier(self) -> None:
        with self.assertRaisesRegex(
            ResearchObjectError, "built-in relation targets must use"
        ):
            build_research_object(
                kind="inference",
                payload={"statement": "The observation supports H1."},
                relations=[{"type": "has_premise", "target": "not-an-object"}],
            )

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
            relations=[
                {
                    "type": "urn:example:usesMaterial",
                    "target": "external-sample:42",
                }
            ],
            semantic_profile=profile,
        )

        self.assertEqual(validate_research_object(payload), payload)
        self.assertEqual(payload["relations"][0]["target"], "external-sample:42")
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

    def _record_locked_preregistration(
        self,
        repository: ResearchRepository,
        *,
        hypothesis_id: str,
        plan_id: str,
        registration_id: str,
        dataset: str,
        metric: str,
        baseline: str,
    ):
        plan = repository.record(
            "research_plan",
            {"plan_id": plan_id, "hypothesis_id": hypothesis_id},
            state="draft",
            relations=[{"type": "depends_on", "target": hypothesis_id}],
        )
        return repository.record(
            "preregistration",
            {
                "preregistration_id": registration_id,
                "plan_id": plan_id,
                "hypothesis_id": hypothesis_id,
                "hypotheses": {"alternative": hypothesis_id},
                "outcomes": [
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "baseline": baseline,
                    }
                ],
                "analysis_plan": {"method": f"compare against {baseline}"},
                "status": "locked",
            },
            state="locked",
            relations=[{"type": "depends_on", "target": plan.object_id}],
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
            repository.commit(stage="ideation", subject="record H1")
            second = repository.record("hypothesis", {"statement": "H2"})
            repository.commit(stage="ideation", subject="record H2")

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

    def test_latest_selector_rejects_multiple_uncommitted_objects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")
            repository.record("hypothesis", {"statement": "H1"})
            repository.record("hypothesis", {"statement": "H2"})

            with self.assertRaisesRegex(ResearchGitError, "ambiguous"):
                repository.resolve("@latest:hypothesis")

    def test_record_rejects_missing_and_wrong_kind_builtin_relation_targets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")
            hypothesis = repository.record("hypothesis", {"statement": "H1"})

            with self.assertRaisesRegex(ResearchGitError, "references missing object"):
                repository.record(
                    "inference",
                    {"statement": "Missing premise"},
                    relations=[
                        {
                            "type": "has_premise",
                            "target": "rso-ffffffffffffffff",
                        }
                    ],
                )
            with self.assertRaisesRegex(
                ResearchGitError,
                "uses_method requires target kind method; got hypothesis",
            ):
                repository.record(
                    "inference",
                    {"statement": "Wrong method binding"},
                    relations=[{"type": "uses_method", "target": hypothesis.object_id}],
                )

    def test_fsck_rejects_missing_and_wrong_kind_builtin_relation_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            invalid_objects = [
                build_research_object(
                    kind="inference",
                    payload={"statement": "Missing method"},
                    relations=[
                        {
                            "type": "uses_method",
                            "target": "rso-ffffffffffffffff",
                        }
                    ],
                ),
                build_research_object(
                    kind="inference",
                    payload={"statement": "Wrong method kind"},
                    relations=[{"type": "uses_method", "target": hypothesis.object_id}],
                ),
            ]
            for payload in invalid_objects:
                target = (
                    root
                    / ".xscientist"
                    / "objects"
                    / str(payload["kind"])
                    / f"{payload['object_id']}.json"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            repository.commit(stage="analysis", subject="inject invalid relations")

            verification = repository.fsck()

            self.assertFalse(verification["ok"])
            self.assertTrue(
                any(
                    "references missing object: rso-ffffffffffffffff" in error
                    for error in verification["errors"]
                ),
                verification["errors"],
            )
            self.assertTrue(
                any(
                    "uses_method requires target kind method; got hypothesis" in error
                    for error in verification["errors"]
                ),
                verification["errors"],
            )

    def test_blame_resolves_latest_selector_at_requested_historical_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")
            first = repository.record("hypothesis", {"statement": "H1"})
            first_checkpoint = repository.commit(stage="ideation", subject="record H1")
            repository.record("hypothesis", {"statement": "H2"})
            repository.commit(stage="ideation", subject="record H2")

            blame = repository.blame(
                "@latest:hypothesis",
                commit=str(first_checkpoint.commit),
            )

        self.assertEqual(blame["selector"], "@latest:hypothesis")
        self.assertEqual(blame["resolved_object_id"], first.object_id)
        self.assertEqual(blame["object"]["object_id"], first.object_id)

    def test_blame_uses_one_resolved_snapshot_when_branch_moves(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            commit_before_object = str(repository.status()["head"])
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            object_checkpoint = repository.commit(
                stage="ideation",
                subject="record H1",
            )
            subprocess.run(
                ["git", "branch", "moving-audit-ref", str(object_checkpoint.commit)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            from xscientist import research_git

            real_run_git = research_git._run_git
            moved = False

            def move_ref_after_resolve(repo_path, args, *, check=True):
                nonlocal moved
                result = real_run_git(repo_path, args, check=check)
                if not moved and list(args) == [
                    "rev-parse",
                    "--verify",
                    "moving-audit-ref^{commit}",
                ]:
                    moved = True
                    real_run_git(
                        repo_path,
                        [
                            "update-ref",
                            "refs/heads/moving-audit-ref",
                            commit_before_object,
                        ],
                    )
                return result

            with mock.patch.object(
                research_git,
                "_run_git",
                side_effect=move_ref_after_resolve,
            ):
                blame = repository.blame(
                    hypothesis.object_id,
                    commit="moving-audit-ref",
                )

            self.assertTrue(moved)
            self.assertEqual(blame["resolved_object_id"], hypothesis.object_id)
            self.assertEqual(blame["origin"]["commit"], object_checkpoint.commit)
            self.assertEqual(
                subprocess.run(
                    ["git", "rev-parse", "moving-audit-ref"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                commit_before_object,
            )

    def test_blame_rejects_multiple_reachable_origins_after_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.fork("duplicate-origin")
            repository.switch("main")
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            object_bytes = hypothesis.path.read_bytes()
            main_checkpoint = repository.commit(
                stage="ideation",
                subject="introduce H1 on main",
            )

            repository.switch("duplicate-origin")
            hypothesis.path.parent.mkdir(parents=True, exist_ok=True)
            hypothesis.path.write_bytes(object_bytes)
            side_checkpoint = repository.commit(
                stage="ideation",
                subject="independently introduce identical H1",
            )
            repository.switch("main")
            repository.merge("duplicate-origin")

            with self.assertRaisesRegex(
                ResearchGitError,
                "multiple reachable origins",
            ):
                repository.blame(hypothesis.object_id)

            main_blame = repository.blame(
                hypothesis.object_id,
                commit=str(main_checkpoint.commit),
            )
            side_blame = repository.blame(
                hypothesis.object_id,
                commit=str(side_checkpoint.commit),
            )
            self.assertEqual(main_blame["origin"]["commit"], main_checkpoint.commit)
            self.assertEqual(side_blame["origin"]["commit"], side_checkpoint.commit)

    def test_origin_lookup_rejects_linear_revert_and_reintroduction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            object_bytes = hypothesis.path.read_bytes()
            introduced = repository.commit(
                stage="ideation",
                subject="introduce H1",
            )
            repository.revert(introduced.commit or "HEAD")
            hypothesis.path.parent.mkdir(parents=True, exist_ok=True)
            hypothesis.path.write_bytes(object_bytes)
            repository.commit(
                stage="ideation",
                subject="reintroduce H1",
            )

            with self.assertRaisesRegex(
                ResearchGitError,
                "multiple reachable origins",
            ):
                repository.blame(hypothesis.object_id)
            with self.assertRaisesRegex(
                ResearchGitError,
                "multiple reachable origins",
            ):
                research_object_origin_checkpoint(
                    root,
                    hypothesis.object_id,
                    kind="hypothesis",
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

    def test_native_stage_honors_an_explicit_safe_file_outside_default_patterns(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            notes = root / "notes.txt"
            notes.write_text("bounded observation\n", encoding="utf-8")

            staged = repository.stage(["notes.txt"])
            checkpoint = repository.commit(
                stage="observation",
                subject="record explicitly selected notes",
                staged_only=True,
            )

            self.assertEqual(staged.paths, ("notes.txt",))
            self.assertTrue(checkpoint.committed)
            self.assertIn("notes.txt", checkpoint.staged_paths)
            self.assertNotIn("notes.txt", repository.status()["eligible_changes"])

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
            first_parent = subprocess.run(
                ["git", "rev-parse", f"{merged.commit}^1"],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            committed_material = {
                path
                for path in subprocess.run(
                    ["git", "diff", "--name-only", first_parent, merged.commit],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout.splitlines()
                if not path.startswith("checkpoints/")
            }
            self.assertEqual(set(checkpoint["changed_paths"]), committed_material)
            self.assertEqual(repository.get(hypothesis.object_id)["kind"], "hypothesis")
            self.assertTrue(verification["ok"], verification["errors"])

    def test_research_merge_refuses_an_uncheckpointed_source_tip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.fork("unreviewed", switch=True)
            secret = root / ".env"
            secret.write_text(
                "deny-listed fixture path\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "-f", ".env"], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "raw unreviewed change"],
                cwd=root,
                capture_output=True,
                check=True,
            )
            repository.switch("main")
            head_before = repository.status()["head"]

            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                repository.merge_preview("unreviewed")
            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                repository.merge("unreviewed")

            self.assertEqual(repository.status()["head"], head_before)
            self.assertFalse(secret.exists())

    def test_research_merge_refuses_a_source_path_denied_by_target_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.fork("source", switch=True)
            source_path = root / "claims" / "source.md"
            source_path.write_text("source-only result\n", encoding="utf-8")
            repository.commit(stage="evidence", subject="record source result")
            repository.switch("main")
            config_path = root / "research.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["git"]["deny_patterns"].append("claims/source.md")
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            repository.commit(stage="policy", subject="deny source path")
            preexisting_view = root / "local-view.log"
            preexisting_view.write_text(
                "preserve this generated local view\n", encoding="utf-8"
            )
            head_before = repository.status()["head"]

            self.assertTrue(repository.merge_preview("source")["clean"])
            with self.assertRaisesRegex(ResearchGitError, "outside.*safety policy"):
                repository.merge("source")

            self.assertEqual(repository.status()["head"], head_before)
            self.assertFalse(source_path.exists())
            self.assertEqual(
                preexisting_view.read_text(encoding="utf-8"),
                "preserve this generated local view\n",
            )
            self.assertEqual(
                subprocess.run(
                    ["git", "diff", "--cached", "--quiet"], cwd=root
                ).returncode,
                0,
            )
            self.assertEqual(
                subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode,
                0,
            )
            self.assertNotEqual(
                subprocess.run(
                    ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"],
                    cwd=root,
                    capture_output=True,
                ).returncode,
                0,
            )
            self.assertTrue(repository.status()["worktree_clean"])

    def test_strict_checkpoint_rejects_an_undeclared_amended_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.record("hypothesis", {"statement": "H1"})
            repository.commit(stage="ideation", subject="record H1")
            hidden = root / "claims" / "hidden.md"
            hidden.write_text("undeclared material\n", encoding="utf-8")
            subprocess.run(["git", "add", str(hidden)], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=root,
                capture_output=True,
                check=True,
            )

            with self.assertRaisesRegex(ResearchGitError, "changed_paths"):
                repository.show()

    def test_exact_checkpoint_paths_remain_stable_with_git_rename_detection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            subprocess.run(
                ["git", "config", "diff.renames", "true"], cwd=root, check=True
            )
            original = root / "claims" / "original.md"
            renamed = root / "claims" / "renamed.md"
            original.write_text("stable scientific material\n", encoding="utf-8")
            repository.commit(stage="evidence", subject="record original")
            original.rename(renamed)

            repository.commit(stage="evidence", subject="rename material")
            checkpoint = repository.show()["checkpoint"]

            self.assertEqual(
                set(checkpoint["changed_paths"]),
                {"claims/original.md", "claims/renamed.md"},
            )

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

    def test_merge_allows_independent_locked_preregistrations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            first_hypothesis = repository.record("hypothesis", {"statement": "H1"})
            self._record_locked_preregistration(
                repository,
                hypothesis_id=first_hypothesis.object_id,
                plan_id="plan-h1",
                registration_id="prereg-h1",
                dataset="dataset-a",
                metric="accuracy",
                baseline="baseline-a",
            )
            repository.commit(stage="preregister", subject="lock H1 plan")
            repository.fork("independent")
            second_hypothesis = repository.record("hypothesis", {"statement": "H2"})
            self._record_locked_preregistration(
                repository,
                hypothesis_id=second_hypothesis.object_id,
                plan_id="plan-h2",
                registration_id="prereg-h2",
                dataset="dataset-b",
                metric="f1",
                baseline="baseline-b",
            )
            repository.commit(stage="preregister", subject="lock H2 plan")
            repository.switch("main")

            preview = repository.merge_preview("independent")
            merged = repository.merge("independent")

            self.assertTrue(preview["clean"], preview["conflicts"])
            self.assertTrue(merged.commit)

    def test_merge_allows_same_hypothesis_in_disjoint_registered_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            self._record_locked_preregistration(
                repository,
                hypothesis_id=hypothesis.object_id,
                plan_id="shared-plan",
                registration_id="prereg-a",
                dataset="dataset-a",
                metric="accuracy",
                baseline="baseline-a",
            )
            repository.commit(stage="preregister", subject="lock scope A")
            repository.fork("scope-b")
            self._record_locked_preregistration(
                repository,
                hypothesis_id=hypothesis.object_id,
                plan_id="shared-plan",
                registration_id="prereg-b",
                dataset="dataset-b",
                metric="accuracy",
                baseline="baseline-b",
            )
            repository.commit(stage="preregister", subject="lock scope B")
            repository.switch("main")

            preview = repository.merge_preview("scope-b")

            self.assertTrue(preview["clean"], preview["conflicts"])

    def test_merge_blocks_incompatible_registration_for_same_plan_and_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            self._record_locked_preregistration(
                repository,
                hypothesis_id=hypothesis.object_id,
                plan_id="shared-plan",
                registration_id="prereg-a",
                dataset="dataset-a",
                metric="accuracy",
                baseline="baseline-a",
            )
            repository.commit(stage="preregister", subject="lock baseline A")
            repository.fork("incompatible")
            self._record_locked_preregistration(
                repository,
                hypothesis_id=hypothesis.object_id,
                plan_id="shared-plan",
                registration_id="prereg-b",
                dataset="dataset-a",
                metric="accuracy",
                baseline="baseline-b",
            )
            repository.commit(stage="preregister", subject="lock baseline B")
            repository.switch("main")

            preview = repository.merge_preview("incompatible")

            self.assertFalse(preview["clean"])
            self.assertIn(
                "locked_preregistration",
                {item["type"] for item in preview["conflicts"]},
            )

    def test_merge_preflight_detects_base_support_and_branch_refutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            support = repository.record(
                "evidence",
                {"result": "positive"},
                relations=[{"type": "supports", "target": hypothesis.object_id}],
            )
            repository.commit(stage="evidence", subject="record base support")
            repository.fork("challenge")
            refutation = repository.record(
                "evidence",
                {"result": "negative"},
                relations=[{"type": "refutes", "target": hypothesis.object_id}],
            )
            repository.commit(stage="evidence", subject="challenge base support")
            repository.switch("main")

            preview = repository.merge_preview("challenge")

            self.assertFalse(preview["clean"])
            opposed = next(
                item
                for item in preview["conflicts"]
                if item["type"] == "opposed_evidence"
            )
            self.assertEqual(opposed["supporting_evidence"], [support.object_id])
            self.assertEqual(opposed["refuting_evidence"], [refutation.object_id])

    def test_merge_does_not_repeat_a_preexisting_contested_pair(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            repository.record(
                "evidence",
                {"result": "positive"},
                relations=[{"type": "supports", "target": hypothesis.object_id}],
            )
            repository.record(
                "evidence",
                {"result": "negative"},
                relations=[{"type": "refutes", "target": hypothesis.object_id}],
            )
            repository.commit(stage="evidence", subject="retain contested evidence")
            repository.fork("notes")
            repository.record("question", {"text": "Which boundary explains H1?"})
            repository.commit(stage="ideation", subject="add an unrelated question")
            repository.switch("main")

            preview = repository.merge_preview("notes")

            self.assertTrue(preview["clean"], preview["conflicts"])

    def test_merge_preflight_detects_base_metric_redefined_on_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.record(
                "metric",
                {
                    "metric_id": "accuracy",
                    "name": "accuracy",
                    "definition": "top-1 accuracy",
                },
            )
            repository.commit(stage="plan", subject="lock base metric")
            repository.fork("metric-change")
            repository.record(
                "metric",
                {
                    "metric_id": "accuracy",
                    "name": "accuracy",
                    "definition": "balanced accuracy",
                },
            )
            repository.commit(stage="plan", subject="redefine source metric")
            repository.switch("main")

            preview = repository.merge_preview("metric-change")

            self.assertFalse(preview["clean"])
            self.assertIn(
                "metric_definition",
                {item["type"] for item in preview["conflicts"]},
            )

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

    def test_adopted_transition_decision_is_a_context_bound_research_object(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})

            recorded = repository.record_decision(
                event="hypothesis",
                name="primary",
            )

            decision_result = recorded["decision_object"]
            context_result = recorded["context_object"]
            self.assertIsNotNone(context_result)
            decision = repository.get(decision_result.object_id)
            context = repository.get(context_result.object_id)
            core = {
                key: value
                for key, value in decision["payload"].items()
                if key != "decision_hash"
            }
            self.assertEqual(
                decision["payload"]["decision_hash"],
                content_hash(core),
            )
            self.assertEqual(
                decision["payload"]["context_hash"],
                context["payload"]["context_hash"],
            )
            self.assertEqual(
                {item["target"] for item in decision["relations"]},
                {context_result.object_id},
            )
            self.assertIn(hypothesis.object_id, context["payload"]["target_ids"])
            self.assertFalse(repository.status()["worktree_clean"])
            self.assertTrue(repository.fsck()["ok"])

    def test_cli_can_checkpoint_an_adopted_transition_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            output = io.StringIO()

            with redirect_stdout(output):
                status = research_main(
                    [
                        "decide",
                        "hypothesis",
                        "--name",
                        "primary",
                        "--record",
                        "--repo",
                        str(root),
                        "--json",
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(payload["object"]["kind"], "gate_decision")
            self.assertTrue(payload["checkpoint"]["committed"])
            self.assertEqual(
                repository.show(payload["checkpoint"]["commit"])["checkpoint"]["stage"],
                "decision",
            )
            self.assertIn(
                hypothesis.object_id,
                repository.get(payload["context_object"]["object_id"])["payload"][
                    "target_ids"
                ],
            )
            self.assertTrue(repository.status()["worktree_clean"])
            self.assertTrue(repository.fsck()["ok"])

    def test_generic_decision_recording_is_closed_after_confirmatory_freeze(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            repository.record(
                "preregistration",
                {
                    "status": "locked",
                    "registration_id": "registration-1",
                    "hypothesis_id": hypothesis.object_id,
                    "adaptive_state_freeze": {"state_hash": "sha256:" + "a" * 64},
                },
                state="locked",
                relations=[{"type": "depends_on", "target": hypothesis.object_id}],
            )

            with self.assertRaisesRegex(
                ResearchGitError,
                "generic transition decisions cannot be recorded",
            ):
                repository.record_decision(event="evidence")

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

    def test_structured_trajectory_projects_objects_and_checkpoints_without_payloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record(
                "hypothesis",
                {"statement": "SENTINEL_SEMANTIC_PAYLOAD"},
                actor={"actor_id": "planner:alpha", "authority": "research_agent"},
            )
            repository.commit(stage="ideation", subject="record falsifiable hypothesis")
            evidence = repository.record(
                "evidence",
                {"result": "SENTINEL_RESULT_PAYLOAD"},
                state="verified",
                relations=[{"type": "supports", "target": hypothesis.object_id}],
                actor={"actor_id": "executor:beta", "authority": "research_agent"},
            )
            repository.commit(stage="experiment", subject="record sealed evidence")

            trajectory = repository.trajectory()

            self.assertEqual(
                trajectory["schema_version"],
                "xscientist.structured-trajectory-projection.v1",
            )
            self.assertTrue(trajectory["complete"])
            self.assertFalse(trajectory["truncated"])
            self.assertFalse(trajectory["payloads_disclosed"])
            self.assertEqual(trajectory["object_count"], 2)
            hash_payload = {
                key: value
                for key, value in trajectory.items()
                if key not in {"selected_ref", "projection_hash"}
            }
            self.assertEqual(trajectory["projection_hash"], content_hash(hash_payload))
            serialized = json.dumps(trajectory)
            self.assertNotIn("SENTINEL_SEMANTIC_PAYLOAD", serialized)
            self.assertNotIn("SENTINEL_RESULT_PAYLOAD", serialized)
            objects = {
                item["object_id"]: item
                for entry in trajectory["entries"]
                for item in entry["objects"]
            }
            self.assertEqual(objects[hypothesis.object_id]["kind"], "hypothesis")
            self.assertEqual(
                objects[evidence.object_id]["relations"],
                [{"type": "supports", "target": hypothesis.object_id}],
            )
            self.assertEqual(
                objects[evidence.object_id]["actor"],
                {"actor_id": "executor:beta", "authority": "research_agent"},
            )
            self.assertEqual(
                trajectory["entries"][-1]["parent_checkpoint_hashes"],
                [trajectory["entries"][-2]["checkpoint_hash"]],
            )

            bounded = repository.trajectory(limit=1)
            self.assertFalse(bounded["complete"])
            self.assertTrue(bounded["truncated"])
            self.assertEqual(bounded["checkpoint_count"], 1)
            self.assertEqual(len(bounded["boundary_parent_edges"]), 1)
            self.assertEqual(
                bounded["boundary_parent_edges"][0]["child_commit"],
                bounded["resolved_head"],
            )
            with self.assertRaisesRegex(ResearchGitError, "between 1 and"):
                repository.trajectory(limit=0)
            with self.assertRaisesRegex(ResearchGitError, "between 1 and"):
                repository.trajectory(limit=128)

            output = io.StringIO()
            with redirect_stdout(output):
                status = research_main(
                    [
                        "trajectory",
                        "--repo",
                        str(root),
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                json.loads(output.getvalue())["projection_hash"],
                trajectory["projection_hash"],
            )

    def test_structured_trajectory_separates_existing_git_parent_from_research_edges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            root.mkdir()
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Research Test"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "research@example.invalid"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            (root / "ordinary.txt").write_text(
                "ordinary project history\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", "--", "ordinary.txt"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "ordinary project commit"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            ordinary_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            repository = self._init(root)
            trajectory = repository.trajectory()
            initial = trajectory["entries"][0]

            self.assertTrue(trajectory["complete"])
            self.assertEqual(trajectory["checkpoint_count"], 1)
            self.assertEqual(initial["backend_parent_commits"], [ordinary_commit])
            self.assertEqual(initial["parent_commits"], [])
            self.assertEqual(initial["parent_checkpoint_hashes"], [])
            self.assertEqual(trajectory["boundary_parent_edges"], [])
            self.assertNotIn(
                ordinary_commit,
                {entry["commit"] for entry in trajectory["entries"]},
            )

    def test_structured_trajectory_projects_legal_revert_object_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            hypothesis = repository.record(
                "hypothesis",
                {"statement": "SENTINEL_REVERTED_PAYLOAD"},
            )
            recorded = repository.commit(
                stage="ideation",
                subject="record hypothesis for semantic revert",
            )

            reverted = repository.revert(recorded.commit or "HEAD")
            trajectory = repository.trajectory()
            transitions = [
                item
                for entry in trajectory["entries"]
                for item in entry["objects"]
                if item["object_id"] == hypothesis.object_id
            ]

            self.assertTrue(trajectory["complete"])
            self.assertEqual(
                [item["change"] for item in transitions],
                ["added", "removed"],
            )
            self.assertEqual(
                transitions[0]["content_hash"],
                transitions[1]["content_hash"],
            )
            revert_checkpoint = repository.show(str(reverted["revert_commit"]))[
                "checkpoint"
            ]
            self.assertEqual(
                revert_checkpoint["reverts_commit"],
                recorded.commit,
            )
            self.assertEqual(
                trajectory["rollback_edges"],
                [
                    {
                        "revert_commit": reverted["revert_commit"],
                        "revert_checkpoint_hash": revert_checkpoint["content_hash"],
                        "target_commit": recorded.commit,
                        "target_checkpoint_hash": recorded.content_hash,
                    }
                ],
            )
            self.assertNotIn("SENTINEL_REVERTED_PAYLOAD", json.dumps(trajectory))

            output = io.StringIO()
            with redirect_stdout(output):
                status = research_main(["trajectory", "--repo", str(root)])
            self.assertEqual(status, 0)
            self.assertIn(
                f"{hypothesis.object_id} hypothesis [draft; removed]", output.getvalue()
            )
            self.assertIn("checkpoint=sha256:", output.getvalue())
            self.assertIn("parents=", output.getvalue())
            self.assertNotIn("SENTINEL_REVERTED_PAYLOAD", output.getvalue())

            bounded_output = io.StringIO()
            with redirect_stdout(bounded_output):
                bounded_status = research_main(
                    ["trajectory", "--repo", str(root), "--limit", "1"]
                )
            self.assertEqual(bounded_status, 0)
            self.assertIn(
                "Boundary parent edges (projection is truncated):",
                bounded_output.getvalue(),
            )

    def test_ordinary_checkpoint_cannot_modify_or_delete_immutable_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            repository.commit(stage="ideation", subject="record H1")
            original = hypothesis.path.read_bytes()

            hypothesis.path.write_bytes(original.replace(b'"H1"', b'"H2"'))
            with self.assertRaisesRegex(ResearchGitError, "cannot be modified"):
                repository.commit(stage="ideation", subject="tamper H1")
            hypothesis.path.write_bytes(original)
            hypothesis.path.unlink()
            with self.assertRaisesRegex(ResearchGitError, "can only be removed"):
                repository.commit(stage="ideation", subject="delete H1")

    def test_typed_revert_refuses_an_extra_immutable_object_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")
            first = repository.record("hypothesis", {"statement": "H1"})
            target = repository.commit(stage="ideation", subject="record H1")
            second = repository.record("hypothesis", {"statement": "H2"})
            repository.commit(stage="ideation", subject="record H2")
            target_checkpoint = repository.show(str(target.commit))["checkpoint"]
            subprocess.run(
                ["git", "revert", "--no-commit", str(target.commit)],
                cwd=repository.path,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "rm", "--", str(second.path.relative_to(repository.path))],
                cwd=repository.path,
                check=True,
                capture_output=True,
                text=True,
            )

            with self.assertRaisesRegex(ResearchGitError, "outside the exact target"):
                _create_checkpoint_locked(
                    repository.path,
                    stage="revert",
                    subject="malicious extra deletion",
                    allow_backend_stage=True,
                    allow_checkpoint_only=True,
                    reverts_commit=str(target.commit),
                    reverts_checkpoint_hash=str(target_checkpoint["content_hash"]),
                )

            self.assertTrue(first.object_id)

    def test_structured_trajectory_closes_branch_and_merge_parents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            shared = repository.record("hypothesis", {"statement": "shared"})
            base = repository.commit(stage="ideation", subject="record shared base")
            repository.fork("alternative")
            branch_object = repository.record(
                "evidence",
                {"result": "SENTINEL_BRANCH_PAYLOAD"},
                state="verified",
                relations=[{"type": "supports", "target": shared.object_id}],
            )
            branch = repository.commit(
                stage="experiment",
                subject="record branch evidence",
            )
            repository.switch("main")
            repository.record("assumption", {"statement": "main-only assumption"})
            main = repository.commit(stage="planning", subject="record main context")
            merged = repository.merge("alternative")

            trajectory = repository.trajectory()
            by_commit = {entry["commit"]: entry for entry in trajectory["entries"]}
            merge_entry = by_commit[str(merged.commit)]

            self.assertTrue(trajectory["complete"])
            self.assertEqual(trajectory["boundary_parent_edges"], [])
            self.assertEqual(
                merge_entry["parent_commits"],
                [str(main.commit), str(branch.commit)],
            )
            self.assertEqual(
                merge_entry["backend_parent_commits"],
                [str(main.commit), str(branch.commit)],
            )
            self.assertEqual(
                merge_entry["parent_checkpoint_hashes"],
                [str(main.content_hash), str(branch.content_hash)],
            )
            for entry in trajectory["entries"]:
                for parent_commit, parent_hash in zip(
                    entry["parent_commits"],
                    entry["parent_checkpoint_hashes"],
                ):
                    parent = by_commit[parent_commit]
                    self.assertEqual(parent["checkpoint_hash"], parent_hash)
                    self.assertLess(parent["sequence"], entry["sequence"])
            self.assertIn(str(base.commit), by_commit)
            self.assertNotIn("SENTINEL_BRANCH_PAYLOAD", json.dumps(trajectory))
            self.assertIn(
                branch_object.object_id,
                {
                    item["object_id"]
                    for entry in trajectory["entries"]
                    for item in entry["objects"]
                },
            )

            bounded = repository.trajectory(limit=1)
            self.assertTrue(bounded["truncated"])
            self.assertFalse(bounded["complete"])
            self.assertEqual(len(bounded["boundary_parent_edges"]), 2)

    def test_structured_trajectory_ignores_clock_skew_when_closing_merge_dag(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            initial_commit = str(repository.status()["head"])
            repository.fork("clock-skew-side")
            repository.record("evidence", {"result": "side result"})
            with mock.patch.dict(
                os.environ,
                {
                    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
                },
            ):
                side = repository.commit(
                    stage="experiment",
                    subject="record clock-skewed side result",
                )
            repository.switch("main")
            repository.record("hypothesis", {"statement": "main hypothesis"})
            with mock.patch.dict(
                os.environ,
                {
                    "GIT_AUTHOR_DATE": "2030-01-01T00:00:00+00:00",
                    "GIT_COMMITTER_DATE": "2030-01-01T00:00:00+00:00",
                },
            ):
                main = repository.commit(
                    stage="planning",
                    subject="record later-dated main state",
                )
            with mock.patch.dict(
                os.environ,
                {
                    "GIT_AUTHOR_DATE": "2031-01-01T00:00:00+00:00",
                    "GIT_COMMITTER_DATE": "2031-01-01T00:00:00+00:00",
                },
            ):
                merged = repository.merge("clock-skew-side")

            trajectory = repository.trajectory(require_complete=True)
            by_commit = {entry["commit"]: entry for entry in trajectory["entries"]}

            self.assertTrue(trajectory["complete"])
            self.assertEqual(trajectory["boundary_parent_edges"], [])
            self.assertEqual(
                set(by_commit),
                {
                    initial_commit,
                    str(side.commit),
                    str(main.commit),
                    str(merged.commit),
                },
            )
            for entry in trajectory["entries"]:
                for parent in entry["parent_commits"]:
                    self.assertLess(
                        by_commit[parent]["sequence"],
                        entry["sequence"],
                    )

    def test_structured_trajectory_rejects_manual_non_checkpoint_commit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            manual = root / "claims" / "manual.md"
            manual.parent.mkdir(parents=True, exist_ok=True)
            manual.write_text("manual backend commit\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "--", "claims/manual.md"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "manual non-checkpoint commit"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            raw_head = repository.status()["head"]
            checkpoints_before = sorted((root / "checkpoints").iterdir())
            repository.record("hypothesis", {"statement": "must remain uncommitted"})

            with self.assertRaisesRegex(
                ResearchGitError,
                "HEAD is an uncheckpointed raw Git commit",
            ):
                repository.commit(
                    stage="planning",
                    subject="must not launder the raw transition",
                )

            self.assertEqual(repository.status()["head"], raw_head)
            self.assertEqual(
                sorted((root / "checkpoints").iterdir()),
                checkpoints_before,
            )

            with self.assertRaisesRegex(
                ResearchGitError,
                "commit without an exact, hash-valid Research VCS checkpoint",
            ):
                repository.trajectory()

    def test_structured_trajectory_rejects_duplicate_matching_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            shown = repository.show()
            duplicate = root / "checkpoints" / "9999-duplicate.json"
            duplicate.write_text(
                json.dumps(shown["checkpoint"], indent=2) + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "add", "--", duplicate.relative_to(root).as_posix()],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            with self.assertRaisesRegex(
                ResearchGitError,
                "commit without an exact, hash-valid Research VCS checkpoint",
            ):
                repository.trajectory()

    def test_structured_trajectory_rejects_unbounded_checkpoint_candidates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            for index in range(16):
                candidate = root / "checkpoints" / f"noise-{index:02d}.json"
                candidate.write_text("{}\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "--", "checkpoints"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            with self.assertRaisesRegex(
                ResearchGitError,
                "commit without an exact, hash-valid Research VCS checkpoint",
            ):
                repository.trajectory()

    def test_structured_trajectory_uses_only_bounded_content_reads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.record(
                "hypothesis",
                {"statement": "SENTINEL_BOUNDED_PAYLOAD"},
            )
            repository.commit(stage="ideation", subject="record bounded object")

            from xscientist import research_git

            real_run_git = research_git._run_git
            real_run_git_bounded = research_git._run_git_bounded
            bounded_calls: list[list[str]] = []

            def guard_unbounded(repo_path, args, *, check=True):
                if args and args[0] in {
                    "cat-file",
                    "diff",
                    "diff-tree",
                    "ls-tree",
                    "rev-list",
                    "show",
                }:
                    raise AssertionError(
                        f"trajectory used an unbounded Git content read: {args[0]}"
                    )
                return real_run_git(repo_path, args, check=check)

            def observe_bounded(repo_path, args, **kwargs):
                bounded_calls.append([str(item) for item in args])
                return real_run_git_bounded(repo_path, args, **kwargs)

            with (
                mock.patch.object(
                    research_git,
                    "_run_git",
                    side_effect=guard_unbounded,
                ),
                mock.patch.object(
                    research_git,
                    "_run_git_bounded",
                    side_effect=observe_bounded,
                ),
            ):
                trajectory = repository.trajectory()

            self.assertTrue(trajectory["complete"])
            self.assertNotIn("SENTINEL_BOUNDED_PAYLOAD", json.dumps(trajectory))
            self.assertTrue(
                {"cat-file", "diff", "diff-tree", "rev-list", "show"}
                & {call[0] for call in bounded_calls}
            )
            object_diff_calls = [
                call
                for call in bounded_calls
                if call[0] in {"diff", "diff-tree"} and "--name-status" in call
            ]
            self.assertTrue(object_diff_calls)
            for call in object_diff_calls:
                separator = call.index("--")
                self.assertEqual(call[separator + 1 :], [".xscientist/objects"])

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

    def test_local_generic_review_cannot_authorize_verified_claim(self) -> None:
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
                    # This unit exercises the generic evidence lifecycle.  A
                    # dedicated confirmatory suite covers the host-attested
                    # freeze and structured-trajectory publication boundary.
                    "study_phase": "exploratory",
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
            with self.assertRaisesRegex(
                ResearchGitError,
                "verified claim requires a passing gate decision",
            ):
                lifecycle.claim(
                    {"statement": "The intervention improves accuracy."},
                    evidence_ids=[evidence["evidence"].object_id],
                    gate_id=evaluation["gate"].object_id,
                    verified=True,
                )

            review_object = repository.get(evaluation["review"].object_id)
            gate_object = repository.get(evaluation["gate"].object_id)
            self.assertEqual(review_object["state"], "completed")
            self.assertEqual(review_object["actor"]["authority"], "recorder")
            self.assertEqual(
                review_object["payload"]["authority_scope"], "local_advisory"
            )
            self.assertFalse(gate_object["payload"]["claim_promotion_allowed"])
            self.assertEqual(gate_object["state"], "rejected")
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
                    "context_snapshot",
                    "evidence",
                    "experiment_attempt",
                    "gate_decision",
                    "hypothesis",
                    "preregistration",
                    "research_plan",
                    "review",
                },
            )

    def test_attempt_actor_binds_producer_and_role_alias_cannot_fake_independence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")
            lifecycle = ResearchLifecycle(repository)
            attempt = lifecycle.experiment_attempt(
                {
                    "status": "completed",
                    "study_phase": "exploratory",
                    "producer_id": "agent:same-principal",
                }
            )["attempt"]

            saved = repository.get(attempt.object_id)
            self.assertEqual(saved["actor"]["actor_id"], "agent:same-principal")
            self.assertEqual(saved["payload"]["producer_id"], "agent:same-principal")
            with self.assertRaisesRegex(
                ResearchGitError,
                "independent of the complete producer provenance",
            ):
                require_independent_evaluator(
                    repository,
                    evaluator_id="human:same-principal",
                    target_ids=[attempt.object_id],
                    label="alias review",
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

    def test_running_attempt_is_never_persisted_as_an_immutable_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._init(Path(td) / "research")

            with self.assertRaisesRegex(ResearchGitError, "mutable execution state"):
                ResearchLifecycle(repository).experiment_attempt(
                    {"status": "running", "summary": "still executing"},
                    commit=False,
                )
            with self.assertRaisesRegex(ResearchGitError, "must be terminal"):
                repository.record(
                    "experiment_attempt",
                    {"status": "running", "summary": "generic bypass"},
                    state="running",
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_invalid_locked_planning_is_rejected_before_any_object_is_written(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            lifecycle = ResearchLifecycle(repository)
            head = repository.status()["head"]

            with self.assertRaisesRegex(
                ResearchGitError, "failed integrity validation"
            ):
                lifecycle.planning(
                    hypothesis={"core_hypothesis": "H1"},
                    plan={"plan_id": "p1", "tasks": []},
                    preregistration={"status": "locked"},
                )

            self.assertEqual(repository.objects(), [])
            self.assertEqual(repository.status()["head"], head)
            self.assertTrue(repository.status()["worktree_clean"])

    def test_confirmatory_attempt_revalidates_a_declared_locked_registration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            forged = repository.record(
                "preregistration",
                {"status": "locked"},
                state="locked",
            )
            repository.commit(stage="preregister", subject="record invalid fixture")

            with self.assertRaisesRegex(
                ResearchGitError, "failed integrity validation"
            ):
                ResearchLifecycle(repository).experiment_attempt(
                    {"status": "success", "study_phase": "confirmatory"},
                    preregistration_id=forged.object_id,
                    commit=False,
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

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
