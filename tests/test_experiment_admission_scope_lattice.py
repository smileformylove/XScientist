from __future__ import annotations

import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml

from tests import test_experiment_admission as admission_helpers
from xscientist import ResearchLifecycle
from xscientist.experiment_admission_state import (
    begin_experiment_admission_transition,
)
from xscientist.research_git import (
    ResearchGitError,
    _repository_lock,
    list_research_objects_at_ref,
)


@unittest.skipUnless(
    admission_helpers.shutil.which("git"), "Git is required for repository tests"
)
class ExperimentAdmissionScopeLatticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = admission_helpers.ExperimentAdmissionTests()
        self.helper.setUp()

    def tearDown(self) -> None:
        self.helper.tearDown()

    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        return completed.stdout.strip()

    @contextmanager
    def _stale_broad_authorization_with_narrow_denial(
        self,
    ) -> Iterator[dict[str, object]]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, hypothesis, plan = self.helper._repository(
                root / "research", admission_mode="required"
            )
            broad = self.helper._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"allowed_seeds": [7, 11]},
            )
            authorization_head = str(repository.status()["head"])
            linked = root / "stale-linked-worktree"
            self._git(
                repository.path,
                "worktree",
                "add",
                "--detach",
                str(linked),
                authorization_head,
            )
            denial = self.helper._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope={"allowed_seeds": [7]},
            )
            linked_ids = {
                str(item["object_id"])
                for item in list_research_objects_at_ref(linked, "HEAD")
            }
            self.assertIn(broad["object_id"], linked_ids)
            self.assertNotIn(denial["object_id"], linked_ids)
            yield {
                "repository": repository,
                "plan": plan,
                "broad": broad,
                "denial": denial,
                "linked": linked,
            }

    def test_stale_linked_worktree_blocks_overlapping_narrow_denial(self) -> None:
        with self._stale_broad_authorization_with_narrow_denial() as fixture:
            lifecycle = ResearchLifecycle(fixture["linked"])

            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                lifecycle.preflight_experiment_admission(
                    plan_id=fixture["plan"]["object_id"],
                    admission_gate_id=fixture["broad"]["object_id"],
                    execution_scope={"seeds": [7]},
                )

    def test_stale_linked_worktree_preserves_nonoverlapping_authorization(self) -> None:
        with self._stale_broad_authorization_with_narrow_denial() as fixture:
            authorized = ResearchLifecycle(
                fixture["linked"]
            ).preflight_experiment_admission(
                plan_id=fixture["plan"]["object_id"],
                admission_gate_id=fixture["broad"]["object_id"],
                execution_scope={"seeds": [11]},
            )

            self.assertEqual(
                authorized["admission_gate_id"], fixture["broad"]["object_id"]
            )

    def test_mixed_seed_batch_is_blocked_by_any_overlapping_denial(self) -> None:
        with self._stale_broad_authorization_with_narrow_denial() as fixture:
            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                ResearchLifecycle(fixture["repository"]).preflight_experiment_admission(
                    plan_id=fixture["plan"]["object_id"],
                    admission_gate_id=fixture["broad"]["object_id"],
                    execution_scope={"seeds": [7, 11]},
                )

    def test_unseen_repository_wide_authorization_is_not_discoverable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, hypothesis, plan = self.helper._repository(
                root / "research", admission_mode="required"
            )
            linked = root / "pre-authorization-worktree"
            self._git(
                repository.path,
                "worktree",
                "add",
                "--detach",
                str(linked),
                str(repository.status()["head"]),
            )
            authorization = self.helper._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "task-1"},
            )
            linked_ids = {
                str(item["object_id"])
                for item in list_research_objects_at_ref(linked, "HEAD")
            }
            self.assertNotIn(authorization["object_id"], linked_ids)

            with self.assertRaises(ResearchGitError):
                ResearchLifecycle(linked).preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    execution_scope={"task_id": "task-1"},
                )

    def test_repository_wide_pending_transition_blocks_all_worktrees(self) -> None:
        with self._stale_broad_authorization_with_narrow_denial() as fixture:
            repository = fixture["repository"]
            denial = repository.get(fixture["denial"]["object_id"])
            with _repository_lock(repository.path):
                begin_experiment_admission_transition(repository.path, denial)

            with self.assertRaisesRegex(ResearchGitError, "transition is pending"):
                ResearchLifecycle(fixture["linked"]).preflight_experiment_admission(
                    plan_id=fixture["plan"]["object_id"],
                    admission_gate_id=fixture["broad"]["object_id"],
                    execution_scope={"seeds": [11]},
                )

    def test_repository_id_drift_invalidates_explicit_and_discovered_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, hypothesis, plan = self.helper._repository(
                root / "research", admission_mode="required"
            )
            authorization = self.helper._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "task-1"},
            )
            config_path = repository.path / "research.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            signed_repository_id = config["repository_id"]
            config["repository_id"] = "ws-ffffffffffffffff"
            self.assertNotEqual(config["repository_id"], signed_repository_id)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            self._git(repository.path, "add", "research.yaml")
            self._git(
                repository.path,
                "commit",
                "-q",
                "-m",
                "change repository identity",
            )
            lifecycle = ResearchLifecycle(repository)

            with self.assertRaisesRegex(ResearchGitError, "repository identity"):
                lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    admission_gate_id=authorization["object_id"],
                    execution_scope={"task_id": "task-1"},
                )
            with self.assertRaisesRegex(ResearchGitError, "admission is required"):
                lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    execution_scope={"task_id": "task-1"},
                )


if __name__ == "__main__":
    unittest.main()
