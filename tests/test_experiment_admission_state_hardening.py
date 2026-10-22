from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import xscientist.experiment_admission_state as admission_state
from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.experiment_admission_state import (
    begin_experiment_admission_transition,
    clear_experiment_admission_transition,
    experiment_admission_subject_hash,
    load_all_experiment_admission_high_water,
    load_all_experiment_admission_transitions,
    load_experiment_admission_high_water,
    load_experiment_admission_transition,
    record_experiment_admission_high_water,
    recover_experiment_admission_staging,
)
from xscientist.research_git import (
    ResearchGitError,
    _repository_common_git_dir,
    _repository_lock,
)
from xscientist.research_vcs import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ExperimentAdmissionStateHardeningTests(unittest.TestCase):
    @staticmethod
    def _repository(root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            question="Which repository-wide admission decision is current?",
            git_user_name="Admission Hardening Test",
            git_user_email="admission-hardening@example.invalid",
        )

    @staticmethod
    def _git(
        repo: Path,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            raise AssertionError(completed.stderr or completed.stdout)
        return completed

    @staticmethod
    def _gate(*, suffix: str, decision: str, task_id: str) -> dict:
        bindings = {
            "hypothesis": {
                "object_id": "rso-1111111111111111",
                "content_hash": "sha256:" + "1" * 64,
            },
            "plan": {
                "object_id": "rso-2222222222222222",
                "content_hash": "sha256:" + "2" * 64,
            },
            "preregistration": None,
        }
        scope = {"task_id": task_id}
        request_hash = canonical_content_hash(
            {
                "repository_id": "ws-1111111111111111",
                "input_bindings": bindings,
                "scope": scope,
                "decision": decision,
            }
        )
        payload = {
            "repository_id": "ws-1111111111111111",
            "input_bindings": bindings,
            "scope": scope,
            "decision": decision,
            "admission_request_hash": request_hash,
            "authority_receipt": {"receipt_hash": "sha256:" + suffix * 64},
        }
        return {
            "object_id": f"rso-{suffix * 16}",
            "content_hash": "sha256:" + suffix * 64,
            "payload": payload,
        }

    @staticmethod
    def _decisions_root(repository: ResearchRepository) -> Path:
        return (
            _repository_common_git_dir(repository.path)
            / "xscientist"
            / "admission-decisions"
        )

    @classmethod
    def _subject_root(cls, repository: ResearchRepository, gate: dict) -> Path:
        subject_hash = experiment_admission_subject_hash(gate["payload"])
        return cls._decisions_root(repository) / subject_hash.removeprefix("sha256:")

    def test_repository_wide_enumeration_and_pending_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            head = str(repository.status()["head"])
            authorization = self._gate(
                suffix="a", decision="authorize", task_id="task-a"
            )
            denial = self._gate(suffix="b", decision="deny", task_id="task-b")
            pending_gate = self._gate(suffix="c", decision="deny", task_id="task-c")
            blocked_gate = self._gate(
                suffix="d", decision="authorize", task_id="task-d"
            )

            first = record_experiment_admission_high_water(
                repository.path, authorization, checkpoint_commit=head
            )
            second = record_experiment_admission_high_water(
                repository.path, denial, checkpoint_commit=head
            )
            pending = begin_experiment_admission_transition(
                repository.path, pending_gate
            )

            self.assertEqual(
                {
                    row["event_hash"]
                    for row in load_all_experiment_admission_high_water(repository.path)
                },
                {first["event_hash"], second["event_hash"]},
            )
            self.assertEqual(
                load_all_experiment_admission_transitions(repository.path),
                [pending],
            )
            with self.assertRaisesRegex(
                ResearchGitError,
                "another experiment admission high-water transition is pending",
            ):
                begin_experiment_admission_transition(repository.path, blocked_gate)

    def test_linked_worktree_observes_every_subject_and_pending_transition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository = self._repository(root / "research")
            head = str(repository.status()["head"])
            authorization = self._gate(
                suffix="1", decision="authorize", task_id="worktree-a"
            )
            denial = self._gate(suffix="2", decision="deny", task_id="worktree-b")
            pending_gate = self._gate(suffix="3", decision="deny", task_id="worktree-c")
            recorded = {
                record_experiment_admission_high_water(
                    repository.path, gate, checkpoint_commit=head
                )["event_hash"]
                for gate in (authorization, denial)
            }
            pending = begin_experiment_admission_transition(
                repository.path, pending_gate
            )

            linked = root / "linked"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.path),
                    "worktree",
                    "add",
                    "--detach",
                    str(linked),
                    head,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                {
                    row["event_hash"]
                    for row in load_all_experiment_admission_high_water(linked)
                },
                recorded,
            )
            self.assertEqual(
                load_all_experiment_admission_transitions(linked), [pending]
            )

    def test_repository_scan_rejects_malformed_unsafe_and_unknown_entries(
        self,
    ) -> None:
        cases = ("malformed-name", "symlink-subject", "unknown-subject-entry")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                repository = self._repository(root / "research")
                decisions_root = self._decisions_root(repository)
                decisions_root.mkdir(parents=True)
                if case == "malformed-name":
                    (decisions_root / "not-a-subject").mkdir()
                elif case == "symlink-subject":
                    target = root / "outside"
                    target.mkdir()
                    try:
                        (decisions_root / ("a" * 64)).symlink_to(
                            target, target_is_directory=True
                        )
                    except OSError as exc:
                        self.skipTest(f"directory symlinks are unavailable: {exc}")
                else:
                    subject_root = decisions_root / ("b" * 64)
                    (subject_root / "events").mkdir(parents=True)
                    (subject_root / "unexpected").write_text("x", encoding="utf-8")

                with self.assertRaises(ResearchGitError):
                    load_all_experiment_admission_high_water(repository.path)

    def test_4097th_subject_is_rejected_before_its_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(
                suffix="4", decision="authorize", task_id="capacity-target"
            )
            decisions_root = self._decisions_root(repository)
            decisions_root.mkdir(parents=True)
            target_root = self._subject_root(repository, gate)
            existing_names = {
                f"{index:064x}"
                for index in range(admission_state._MAX_DECISION_SUBJECTS)
            }
            self.assertNotIn(target_root.name, existing_names)
            for name in existing_names:
                (decisions_root / name).mkdir()

            with self.assertRaisesRegex(ResearchGitError, "capacity is exhausted"):
                begin_experiment_admission_transition(repository.path, gate)
            self.assertFalse(target_root.exists())
            self.assertEqual(
                len(tuple(decisions_root.iterdir())),
                admission_state._MAX_DECISION_SUBJECTS,
            )

    def test_staging_residue_blocks_reads_until_exact_locked_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(suffix="5", decision="deny", task_id="recovery")
            pending = begin_experiment_admission_transition(repository.path, gate)
            subject_root = self._subject_root(repository, gate)
            pending_staging = subject_root / ".admission-pending-unit.tmp"
            event_staging = subject_root / ".admission-event-unit.tmp"
            pending_staging.write_text("partial", encoding="utf-8")
            event_staging.write_text("partial", encoding="utf-8")

            with self.assertRaisesRegex(
                ResearchGitError, "staging residue requires exact transition retry"
            ):
                load_experiment_admission_high_water(repository.path, gate["payload"])
            self.assertTrue(pending_staging.exists())
            self.assertTrue(event_staging.exists())
            with self.assertRaisesRegex(ResearchGitError, "exclusive repository lock"):
                recover_experiment_admission_staging(repository.path, gate["payload"])

            with _repository_lock(repository.path):
                recover_experiment_admission_staging(repository.path, gate["payload"])

            self.assertFalse(pending_staging.exists())
            self.assertFalse(event_staging.exists())
            self.assertEqual(
                load_experiment_admission_transition(repository.path, gate["payload"]),
                pending,
            )

    def test_locked_recovery_rejects_symlink_staging_residue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository = self._repository(root / "research")
            gate = self._gate(suffix="6", decision="deny", task_id="unsafe-recovery")
            begin_experiment_admission_transition(repository.path, gate)
            subject_root = self._subject_root(repository, gate)
            target = root / "outside.tmp"
            target.write_text("do not remove", encoding="utf-8")
            staging = subject_root / ".admission-event-symlink.tmp"
            try:
                staging.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"file symlinks are unavailable: {exc}")

            with (
                _repository_lock(repository.path),
                self.assertRaisesRegex(ResearchGitError, "staging residue is unsafe"),
            ):
                recover_experiment_admission_staging(repository.path, gate["payload"])

            self.assertTrue(staging.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "do not remove")

    def test_pending_publish_failure_before_link_leaves_no_final_or_staging_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(
                suffix="7", decision="authorize", task_id="pending-pre-link"
            )

            with mock.patch.object(
                admission_state.os,
                "fsync",
                side_effect=OSError("synthetic pending file flush failure"),
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic pending file flush failure"
                ):
                    begin_experiment_admission_transition(repository.path, gate)

            subject_root = self._subject_root(repository, gate)
            self.assertFalse((subject_root / "pending.json").exists())
            self.assertEqual(list(subject_root.glob(".admission-pending-*.tmp")), [])
            self.assertIsNone(
                load_experiment_admission_transition(repository.path, gate["payload"])
            )

    def test_ref_update_failure_remains_pending_and_exact_record_repairs_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(suffix="9", decision="deny", task_id="ref-repair")
            head = str(repository.status()["head"])
            pending = begin_experiment_admission_transition(repository.path, gate)
            real_run_git = admission_state._run_git

            def fail_update_ref(
                repo: Path,
                args: list[str],
                *,
                check: bool = True,
            ) -> subprocess.CompletedProcess[str]:
                if args and args[0] == "update-ref":
                    raise ResearchGitError("synthetic checkpoint ref update failure")
                return real_run_git(repo, args, check=check)

            with mock.patch.object(
                admission_state,
                "_run_git",
                side_effect=fail_update_ref,
            ):
                with self.assertRaisesRegex(
                    ResearchGitError, "synthetic checkpoint ref update failure"
                ):
                    record_experiment_admission_high_water(
                        repository.path, gate, checkpoint_commit=head
                    )

            self.assertEqual(
                load_experiment_admission_transition(repository.path, gate["payload"]),
                pending,
            )
            with self.assertRaisesRegex(
                ResearchGitError, "retention ref does not match"
            ):
                load_all_experiment_admission_high_water(repository.path)

            repaired = record_experiment_admission_high_water(
                repository.path, gate, checkpoint_commit=head
            )
            self.assertEqual(repaired["sequence"], 1)
            self.assertEqual(
                len(list(self._subject_root(repository, gate).glob("events/*.json"))),
                1,
            )
            self.assertEqual(
                load_all_experiment_admission_high_water(repository.path),
                [repaired],
            )
            clear_experiment_admission_transition(repository.path, gate)

    def test_hidden_ref_keeps_checkpoint_after_worktree_branch_delete_and_gc(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository = self._repository(root / "research")
            base = str(repository.status()["head"])
            linked = root / "ephemeral-worktree"
            branch = "admission-ephemeral"
            self._git(
                repository.path,
                "worktree",
                "add",
                "-b",
                branch,
                str(linked),
                base,
            )
            (linked / "ephemeral.txt").write_text(
                "checkpoint retained only by admission state\n",
                encoding="utf-8",
            )
            self._git(linked, "add", "ephemeral.txt")
            self._git(linked, "commit", "-m", "ephemeral admission checkpoint")
            checkpoint = self._git(linked, "rev-parse", "HEAD").stdout.strip()
            gate = self._gate(suffix="a", decision="authorize", task_id="gc-retention")
            event = record_experiment_admission_high_water(
                repository.path,
                gate,
                checkpoint_commit=checkpoint,
            )
            subject_hash = experiment_admission_subject_hash(gate["payload"])
            ref_name = "refs/xscientist/admission/" + subject_hash.removeprefix(
                "sha256:"
            )

            self._git(
                repository.path,
                "worktree",
                "remove",
                "--force",
                str(linked),
            )
            self._git(repository.path, "branch", "-D", branch)
            self._git(repository.path, "reflog", "expire", "--expire=now", "--all")
            self._git(repository.path, "gc", "--prune=now")

            self.assertFalse(linked.exists())
            self.assertNotEqual(
                self._git(
                    repository.path,
                    "show-ref",
                    "--verify",
                    "--quiet",
                    f"refs/heads/{branch}",
                    check=False,
                ).returncode,
                0,
            )
            self._git(repository.path, "cat-file", "-e", f"{checkpoint}^{{commit}}")
            self.assertEqual(
                self._git(
                    repository.path,
                    "rev-parse",
                    "--verify",
                    f"{ref_name}^{{commit}}",
                ).stdout.strip(),
                checkpoint,
            )
            self.assertEqual(
                load_all_experiment_admission_high_water(repository.path),
                [event],
            )

    def test_exact_retry_repeats_event_directory_fsync_until_it_succeeds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(suffix="8", decision="deny", task_id="event-fsync")
            head = str(repository.status()["head"])
            pending = begin_experiment_admission_transition(repository.path, gate)
            events_dir = self._subject_root(repository, gate) / "events"
            real_fsync_directory = admission_state._fsync_directory
            event_fsync_attempts = 0

            def fail_event_directory(path: Path) -> None:
                nonlocal event_fsync_attempts
                if Path(path) == events_dir:
                    event_fsync_attempts += 1
                    raise OSError("synthetic event directory fsync failure")
                real_fsync_directory(path)

            with mock.patch.object(
                admission_state,
                "_fsync_directory",
                side_effect=fail_event_directory,
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic event directory fsync failure"
                ):
                    record_experiment_admission_high_water(
                        repository.path, gate, checkpoint_commit=head
                    )
                with self.assertRaisesRegex(
                    OSError, "synthetic event directory fsync failure"
                ):
                    begin_experiment_admission_transition(repository.path, gate)

            self.assertEqual(event_fsync_attempts, 2)
            self.assertEqual(
                load_experiment_admission_transition(repository.path, gate["payload"]),
                pending,
            )
            self.assertIsNotNone(
                load_experiment_admission_high_water(repository.path, gate["payload"])
            )
            begin_experiment_admission_transition(repository.path, gate)
            clear_experiment_admission_transition(repository.path, gate)
            self.assertIsNone(
                load_experiment_admission_transition(repository.path, gate["payload"])
            )


if __name__ == "__main__":
    unittest.main()
