from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from xscientist.research_git import ResearchGitError, record_research_object
from xscientist.research_vcs import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ExperimentAdmissionPolicyMonotonicityTests(unittest.TestCase):
    def _repository(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            question="Can a committed admission policy be downgraded?",
            git_user_name="Admission Policy Test",
            git_user_email="admission-policy@example.invalid",
        )

    @staticmethod
    def _set_policy(repository: ResearchRepository, mode: str) -> None:
        config_path = repository.path / "research.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["experiment_admission"] = {"mode": mode}
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        result = repository.commit(
            stage="review",
            subject=f"set experiment admission policy to {mode}",
        )
        if not result.committed:
            raise AssertionError(f"policy commit failed: {result.reason}")

    def test_committed_required_policy_cannot_be_downgraded_to_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            self._set_policy(repository, "required")
            self._set_policy(repository, "legacy")

            self.assertEqual(repository.experiment_admission_mode(), "required")
            with self.assertRaisesRegex(
                ResearchGitError, "raw experiment_attempt recording is disabled"
            ):
                repository.record(
                    "experiment_attempt",
                    {"status": "completed", "summary": "downgrade bypass"},
                    state="completed",
                )

    def test_old_linked_worktree_observes_repository_wide_required_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository = self._repository(root / "research")
            linked_path = root / "old-policy-worktree"
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "add",
                    "-b",
                    "old-policy",
                    str(linked_path),
                    "HEAD",
                ],
                cwd=repository.path,
                check=True,
                capture_output=True,
                text=True,
            )
            linked = ResearchRepository(linked_path)

            self._set_policy(repository, "required")
            self.assertEqual(repository.experiment_admission_mode(), "required")
            self._set_policy(repository, "legacy")

            self.assertEqual(linked.experiment_admission_mode(), "required")
            with self.assertRaisesRegex(
                ResearchGitError, "raw experiment_attempt recording is disabled"
            ):
                record_research_object(
                    linked.path,
                    kind="experiment_attempt",
                    payload={
                        "status": "completed",
                        "summary": "stale linked-worktree bypass",
                    },
                    state="completed",
                )


if __name__ == "__main__":
    unittest.main()
