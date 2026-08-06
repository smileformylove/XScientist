from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import validate

import xscientist.research_git as research_git_module
from ai_scientist.protocol.schemas import load_schema
from xscientist.research_cli import main as research_main
from xscientist.research_git import (
    ResearchGitError,
    add_research_object,
    create_checkpoint,
    create_research_bundle,
    init_repository,
    repository_status,
    reproduce_checkpoint,
    restore_research_bundle,
    research_diff,
    research_log,
    show_checkpoint,
    verify_research_bundle,
    verify_research_repository,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for research history tests")
class LocalResearchGitTests(unittest.TestCase):
    def _init(self, root: Path, **kwargs):
        return init_repository(
            root,
            question="# Question\n\nDoes H1 improve the metric?\n",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
            **kwargs,
        )

    def _git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    def test_init_creates_serverless_repository_and_initial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"

            result = self._init(root)
            status = repository_status(root)

            self.assertTrue(result.committed)
            self.assertTrue((root / ".git").is_dir())
            self.assertEqual(status["branch"], "main")
            self.assertEqual(status["checkpoint_policy"], "milestone")
            self.assertFalse(status["auto_push"])
            self.assertFalse(self._git(root, "remote"), "init must not create a remote")
            checkpoint = show_checkpoint(root)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "init")
            validate(checkpoint, load_schema("research_checkpoint"))

    def test_checkpoint_stages_only_policy_paths_and_skips_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root, max_file_bytes=2048)
            hypothesis = root / "hypotheses" / "h1.json"
            hypothesis.write_text('{"hypothesis":"H1"}\n', encoding="utf-8")
            (root / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
            (root / "hypotheses" / "large.json").write_text(
                "x" * 4096, encoding="utf-8"
            )

            result = create_checkpoint(
                root,
                stage="preregister",
                subject="lock H1",
                summary="Prospective hypothesis.",
            )
            tree = set(
                self._git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
            )

            self.assertTrue(result.committed)
            self.assertIn("hypotheses/h1.json", tree)
            self.assertNotIn(".env", tree)
            self.assertNotIn("hypotheses/large.json", tree)
            self.assertTrue(any("exceeds" in item for item in result.excluded_paths))

    def test_checkpoint_refuses_a_prepopulated_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            path = root / "hypotheses" / "h1.json"
            path.write_text("{}\n", encoding="utf-8")
            self._git(root, "add", "hypotheses/h1.json")

            with self.assertRaisesRegex(ResearchGitError, "already contains staged"):
                create_checkpoint(root, stage="preregister", subject="lock H1")

    def test_checkpoint_commit_failure_restores_index_and_own_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            existing_checkpoints = set((root / "checkpoints").iterdir())
            hypothesis = root / "hypotheses" / "h1.json"
            hypothesis.write_text('{"hypothesis":"H1"}\n', encoding="utf-8")
            real_run_git = research_git_module._run_git

            def fail_commit(repo, args, *, check=True):
                if args and args[0] == "commit":
                    raise ResearchGitError("injected commit failure")
                return real_run_git(repo, args, check=check)

            with mock.patch.object(
                research_git_module,
                "_run_git",
                side_effect=fail_commit,
            ):
                with self.assertRaisesRegex(ResearchGitError, "injected"):
                    create_checkpoint(root, stage="preregister", subject="lock H1")

            self.assertEqual(set((root / "checkpoints").iterdir()), existing_checkpoints)
            self.assertTrue(hypothesis.is_file())
            self.assertFalse(self._git(root, "diff", "--cached", "--name-only"))
            self.assertIn("hypotheses/", self._git(root, "status", "--porcelain"))

    def test_checkpoint_skips_when_only_ignored_or_untracked_noise_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            (root / "scratch.txt").write_text("noise\n", encoding="utf-8")

            result = create_checkpoint(root, stage="experiment", subject="no result")

            self.assertFalse(result.created)
            self.assertIn("no material research change", result.reason or "")

    def test_object_pointer_bundle_and_reproduction_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "dataset.bin"
            source.write_bytes(b"evidence" * 1024)

            pointer = add_research_object(
                root,
                source,
                logical_path="data/dataset.bin",
            )
            checkpoint = create_checkpoint(
                root,
                stage="evidence",
                subject="bind dataset",
                reproduce_command="python verify.py",
            )
            bundle_path = base / "research.tar.gz"
            bundle = create_research_bundle(root, bundle_path)
            bundle_verification = verify_research_bundle(bundle_path)
            reproduction = reproduce_checkpoint(
                root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "reproduction",
            )
            checkpoint_payload = json.loads(
                checkpoint.checkpoint_path.read_text(encoding="utf-8")
            )

            self.assertIn(pointer.object_hash, checkpoint_payload["object_refs"])
            self.assertTrue(bundle["complete"])
            self.assertTrue(bundle_verification["ok"], bundle_verification["errors"])
            self.assertTrue(reproduction["objects_complete"])
            self.assertEqual(
                (base / "reproduction" / "data" / "dataset.bin").read_bytes(),
                source.read_bytes(),
            )
            with tarfile.open(bundle_path) as archive:
                names = set(archive.getnames())
            self.assertIn("repository.gitbundle", names)
            self.assertIn("bundle.manifest.json", names)
            self.assertIn(
                f"objects/sha256/{pointer.object_hash.split(':', 1)[1]}", names
            )
            validate(
                json.loads(pointer.pointer_path.read_text(encoding="utf-8")),
                load_schema("research_object_pointer"),
            )
            validate(
                {key: value for key, value in bundle.items() if key != "destination"},
                load_schema("research_bundle"),
            )

    def test_bundle_restore_round_trip_is_clean_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"portable evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            checkpoint = create_checkpoint(
                root,
                stage="evidence",
                subject="bind portable evidence",
            )
            bundle_path = base / "research.tar.gz"
            create_research_bundle(root, bundle_path)

            restored = restore_research_bundle(bundle_path, base / "restored")
            inspection = reproduce_checkpoint(
                base / "restored",
                commit=checkpoint.commit or "HEAD",
                destination=base / "restored-run",
            )

            self.assertEqual(restored["commit"], checkpoint.commit)
            self.assertTrue(restored["fsck"]["ok"])
            self.assertEqual(restored["objects_restored"], 1)
            self.assertFalse(self._git(base / "restored", "remote"))
            self.assertFalse(self._git(base / "restored", "status", "--porcelain"))
            self.assertEqual(
                (Path(inspection["worktree"]) / "data" / "evidence.bin").read_bytes(),
                b"portable evidence",
            )
            restored_object = (
                base
                / "restored"
                / ".ara-store"
                / "objects"
                / "sha256"
                / pointer.object_hash.split(":", 1)[1]
            )
            self.assertEqual(restored_object.read_bytes(), b"portable evidence")

    def test_bundle_verification_rejects_tampered_member(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            create_checkpoint(root, stage="evidence", subject="bind data")
            original = base / "research.tar.gz"
            tampered = base / "tampered.tar.gz"
            create_research_bundle(root, original)

            with tarfile.open(original, "r:gz") as source_archive:
                payloads = {
                    member.name: source_archive.extractfile(member).read()
                    for member in source_archive.getmembers()
                    if member.isfile()
                }
            object_name = f"objects/sha256/{pointer.object_hash.split(':', 1)[1]}"
            payloads[object_name] = b"tampered evidence"
            with tarfile.open(tampered, "w:gz") as target_archive:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    target_archive.addfile(info, io.BytesIO(payload))

            verification = verify_research_bundle(tampered)

            self.assertFalse(verification["ok"])
            self.assertTrue(
                any("hash mismatch" in error for error in verification["errors"]),
                verification["errors"],
            )
            with self.assertRaisesRegex(ResearchGitError, "verification failed"):
                restore_research_bundle(tampered, base / "restored")

    def test_cas_snapshot_is_independent_from_mutable_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"immutable evidence")

            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            source.write_bytes(b"changed after registration")

            self.assertFalse(pointer.linked)
            self.assertEqual(pointer.store_path.read_bytes(), b"immutable evidence")

    def test_reproduction_rejects_tampered_cas_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            checkpoint = create_checkpoint(root, stage="evidence", subject="bind data")
            pointer.store_path.write_bytes(b"tampered evidence")

            inspection = reproduce_checkpoint(root, commit=checkpoint.commit or "HEAD")

            self.assertFalse(inspection["objects_complete"])
            self.assertEqual(inspection["damaged_objects"], [pointer.object_hash])
            with self.assertRaisesRegex(ResearchGitError, "missing or damaged"):
                reproduce_checkpoint(
                    root,
                    commit=checkpoint.commit or "HEAD",
                    destination=base / "reproduction",
                )

    def test_fsck_detects_pointer_and_object_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            create_checkpoint(root, stage="evidence", subject="bind data")

            clean = verify_research_repository(root)
            pointer.store_path.write_bytes(b"damaged")
            damaged = verify_research_repository(root)

            self.assertTrue(clean["ok"])
            self.assertFalse(damaged["ok"])
            self.assertTrue(
                any("CAS object" in error for error in damaged["errors"]),
                damaged["errors"],
            )

    def test_reproduction_refuses_tampered_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            checkpoint = create_checkpoint(root, stage="evidence", subject="bind data")
            payload = json.loads(pointer.pointer_path.read_text(encoding="utf-8"))
            payload["size"] += 1
            pointer.pointer_path.write_text(json.dumps(payload), encoding="utf-8")
            self._git(root, "add", f"research-objects/{pointer.pointer_path.name}")
            self._git(root, "commit", "-m", "tamper pointer")

            with self.assertRaisesRegex(ResearchGitError, "cannot validate"):
                reproduce_checkpoint(root, commit="HEAD")
            self.assertTrue(checkpoint.commit)

    def test_log_show_and_diff_expose_scientific_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            first = self._init(root)
            (root / "claims" / "c1.json").write_text(
                '{"claim":"C1"}\n', encoding="utf-8"
            )
            second = create_checkpoint(
                root,
                stage="evidence",
                subject="bind C1",
                claims=["c1"],
            )

            history = research_log(root)
            shown = show_checkpoint(root, second.commit or "HEAD")
            diff = research_diff(
                root,
                first.commit or "HEAD~1",
                second.commit or "HEAD",
                deep=True,
            )

            self.assertEqual(history[0]["trailers"]["Research-Stage"], ["evidence"])
            self.assertEqual(shown["checkpoint"]["claims"], ["c1"])
            self.assertTrue(any("claims/c1.json" in line for line in diff["changes"]))
            self.assertEqual(diff["semantic"]["claims"]["added"], ["c1"])
            self.assertTrue(
                any(
                    change["file"] == "claims/c1.json"
                    for change in diff["semantic"]["structured_changes"]
                )
            )

    def test_environment_receipt_supports_strict_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            (root / "requirements.txt").write_text(
                "jsonschema==4.23.0\n",
                encoding="utf-8",
            )
            checkpoint = create_checkpoint(
                root,
                stage="preregister",
                subject="lock runtime",
            )
            payload = json.loads(checkpoint.checkpoint_path.read_text(encoding="utf-8"))
            receipt = payload["reproduce"]["environment"]

            validate(receipt, load_schema("research_environment"))
            reproduction = reproduce_checkpoint(
                root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "strict-reproduction",
                environment_policy="strict",
            )

            self.assertTrue(reproduction["environment"]["matches"])
            self.assertEqual(
                receipt["dependency_locks"][0]["path"],
                "requirements.txt",
            )

    def test_environment_policy_reports_runtime_drift_and_strictly_refuses_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)

            with mock.patch.object(
                research_git_module.platform,
                "python_version",
                return_value="0.0.0",
            ):
                warning = reproduce_checkpoint(root, environment_policy="warn")
                with self.assertRaisesRegex(ResearchGitError, "environment mismatch"):
                    reproduce_checkpoint(root, environment_policy="strict")

            self.assertFalse(warning["environment"]["matches"])
            self.assertEqual(
                warning["environment"]["mismatches"][0]["field"],
                "python.version",
            )

    def test_strict_reproduction_refuses_dependency_lock_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            lock = root / "requirements.txt"
            lock.write_text("jsonschema==4.23.0\n", encoding="utf-8")
            create_checkpoint(root, stage="preregister", subject="lock dependencies")
            lock.write_text("jsonschema==4.24.0\n", encoding="utf-8")
            self._git(root, "add", "requirements.txt")
            self._git(root, "commit", "-m", "mutate dependency lock without checkpoint")
            destination = base / "strict-reproduction"

            with self.assertRaisesRegex(ResearchGitError, "dependency lock mismatch"):
                reproduce_checkpoint(
                    root,
                    destination=destination,
                    environment_policy="strict",
                )

            self.assertFalse(destination.exists())

    def test_divergent_branches_converge_with_multiple_scientific_parents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            self._git(root, "switch", "-c", "hypothesis/a")
            (root / "claims" / "a.json").write_text("{}\n", encoding="utf-8")
            branch_a = create_checkpoint(
                root,
                stage="experiment",
                subject="test branch A",
            )
            self._git(root, "switch", "main")
            (root / "claims").mkdir(exist_ok=True)
            (root / "claims" / "b.json").write_text("{}\n", encoding="utf-8")
            branch_b = create_checkpoint(
                root,
                stage="experiment",
                subject="test branch B",
            )
            self._git(
                root,
                "merge",
                "--no-ff",
                "hypothesis/a",
                "-m",
                "merge scientific branches",
            )
            (root / "manuscript" / "merge.md").write_text(
                "# Converged evidence\n",
                encoding="utf-8",
            )

            converged = create_checkpoint(
                root,
                stage="review",
                subject="review converged evidence",
            )
            payload = json.loads(converged.checkpoint_path.read_text(encoding="utf-8"))
            fsck = verify_research_repository(root)

            self.assertEqual(payload["sequence"], 3)
            self.assertEqual(
                set(payload["parent_checkpoint_hashes"]),
                {branch_a.content_hash, branch_b.content_hash},
            )
            self.assertEqual(
                payload["previous_checkpoint_hash"],
                payload["parent_checkpoint_hashes"][0],
            )
            self.assertTrue(fsck["ok"], fsck["errors"])

    def test_old_commit_resolves_its_own_pointer_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"old evidence")
            pointer = add_research_object(
                root, source, logical_path="data/evidence.bin"
            )
            bound = create_checkpoint(root, stage="evidence", subject="bind object")
            pointer.pointer_path.unlink()
            (root / "claims" / "c2.json").write_text("{}\n", encoding="utf-8")
            create_checkpoint(root, stage="review", subject="supersede object")

            reproduction = reproduce_checkpoint(root, commit=bound.commit or "HEAD")

            self.assertTrue(reproduction["objects_complete"])
            self.assertFalse(reproduction["missing_objects"])

    def test_reproduction_executes_without_a_shell_in_detached_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            script = root / ".xscientist" / "verify.py"
            script.write_text("print('verified')\n", encoding="utf-8")
            checkpoint = create_checkpoint(
                root,
                stage="review",
                subject="record verifier",
                reproduce_command="python .xscientist/verify.py",
            )

            reproduction = reproduce_checkpoint(
                root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "execute",
                execute=True,
                timeout_seconds=10,
            )

            self.assertEqual(reproduction["returncode"], 0)
            self.assertFalse(reproduction["timed_out"])
            self.assertEqual(reproduction["stdout"].strip(), "verified")

    def test_reproduction_refuses_a_tampered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            checkpoint_path = sorted((root / "checkpoints").glob("*.json"))[-1]
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["summary"] = "tampered"
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
            self._git(root, "add", checkpoint_path.relative_to(root).as_posix())
            self._git(root, "commit", "-m", "tamper checkpoint")

            self.assertFalse(show_checkpoint(root)["checkpoint_hash_valid"])
            with self.assertRaisesRegex(ResearchGitError, "hash verification failed"):
                reproduce_checkpoint(root)

    def test_unified_cli_supports_init_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self.assertEqual(
                research_main(
                    [
                        "init",
                        str(root),
                        "--question",
                        "Does H1 hold?",
                        "--git-user-name",
                        "Research Test",
                        "--git-user-email",
                        "research@example.invalid",
                    ]
                ),
                0,
            )
            self.assertEqual(research_main(["status", "--repo", str(root)]), 0)


if __name__ == "__main__":
    unittest.main()
