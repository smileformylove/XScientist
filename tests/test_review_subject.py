from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)
from ai_scientist.utils.review_execution import execute_review_pass
from ai_scientist.utils.pipeline_helpers import save_review_artifacts
from ai_scientist.utils.critic_workflow import run_independent_critic_pass
from ai_scientist.utils.workflow_runtime import (
    build_workflow_runtime_plan,
    execute_review_suite,
)
from ai_scientist.utils.review_subject import (
    capture_review_subject,
    check_review_subject,
)


class ReviewSubjectTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.pdf = self.root / "paper.pdf"
        self.pdf.write_bytes(b"%PDF-1.4\nreview fixture\n")
        initialize_pipeline_contracts(self.root)

    def _git(self, *args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                f"core.hooksPath={self.base / 'no-hooks'}",
                "-C",
                str(self.root),
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _repository(self) -> None:
        if not shutil.which("git"):
            self.skipTest("git is unavailable")
        self._git("init")
        (self.root / "experiment.py").write_text("value = 1\n", encoding="utf-8")
        self._git("add", "paper.pdf", "experiment.py")
        self._git("commit", "-m", "fixture")

    def test_subject_is_portable_and_matches_current_bytes(self) -> None:
        subject = capture_review_subject(self.root, self.pdf)
        self.assertEqual(subject["status"], "bound")
        self.assertEqual(subject["pdf"]["path"], "paper.pdf")
        self.assertNotIn(str(self.base), json.dumps(subject))
        self.assertTrue(check_review_subject(self.root, subject)[0])
        copied = self.base / "copied"
        shutil.copytree(self.root, copied)
        self.assertTrue(check_review_subject(copied, subject)[0])

    def test_pdf_change_or_explicit_different_target_invalidates_subject(self) -> None:
        subject = capture_review_subject(self.root, self.pdf)
        other = self.root / "other.pdf"
        shutil.copyfile(self.pdf, other)
        self.assertFalse(check_review_subject(self.root, subject, pdf_path=other)[0])
        self.pdf.write_bytes(b"%PDF-1.4\nchanged fixture\n")
        self.assertEqual(
            check_review_subject(self.root, subject), (False, "stale_review_subject")
        )

    def test_declared_latex_source_is_bound_even_before_pdf_rebuild(self) -> None:
        latex = self.root / "template.tex"
        latex.write_text("original source", encoding="utf-8")
        save_contract_artifact(
            self.root,
            "manuscript_state",
            {"latex_path": "template.tex"},
            producer="test",
        )
        subject = capture_review_subject(self.root, self.pdf)
        self.assertEqual(subject["manuscript_source"]["path"], "template.tex")
        latex.write_text("edited source", encoding="utf-8")
        self.assertFalse(check_review_subject(self.root, subject)[0])
        latex.unlink()
        self.assertEqual(
            capture_review_subject(self.root, self.pdf)["status"], "unavailable"
        )

    def test_missing_empty_external_and_directory_inputs_are_unbound(self) -> None:
        external = self.base / "external.pdf"
        external.write_bytes(b"private fixture")
        empty = self.root / "empty.pdf"
        empty.touch()
        for value in (None, empty, self.root / "missing.pdf", self.root, external):
            with self.subTest(value=value):
                subject = capture_review_subject(self.root, value)
                self.assertEqual(subject["status"], "unavailable")
                self.assertFalse(check_review_subject(self.root, subject)[0])
                self.assertNotIn(str(self.base), json.dumps(subject))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires named pipes")
    def test_special_files_are_rejected_without_blocking(self) -> None:
        fifo = self.root / "pipe.pdf"
        os.mkfifo(fifo)
        self.assertEqual(
            capture_review_subject(self.root, fifo)["status"], "unavailable"
        )

    def test_tampered_and_incomplete_bindings_do_not_pass(self) -> None:
        subject = capture_review_subject(self.root, self.pdf)
        for changed in (
            None,
            {},
            {**subject, "schema": "unknown"},
            {**subject, "pdf": {}},
            {**subject, "subject_hash": "bad"},
        ):
            with self.subTest(subject=changed):
                self.assertFalse(check_review_subject(self.root, changed)[0])

    def test_hashing_limits_fail_closed(self) -> None:
        with patch("ai_scientist.utils.review_subject.MAX_SUBJECT_BYTES", 1):
            self.assertEqual(
                capture_review_subject(self.root, self.pdf)["status"], "unavailable"
            )

    def test_commit_change_invalidates_subject(self) -> None:
        self._repository()
        subject = capture_review_subject(self.root, self.pdf)
        self.assertEqual(
            subject["source_revision"]["commit"], self._git("rev-parse", "HEAD")
        )
        self._git("commit", "--allow-empty", "-m", "next revision")
        self.assertFalse(check_review_subject(self.root, subject)[0])

    def test_uncommitted_tracked_source_edits_and_deletion_are_detected(self) -> None:
        self._repository()
        subject = capture_review_subject(self.root, self.pdf)
        source = self.root / "experiment.py"
        source.write_text("value = 2\n", encoding="utf-8")
        self.assertFalse(check_review_subject(self.root, subject)[0])
        source.unlink()
        self.assertFalse(check_review_subject(self.root, subject)[0])

    def test_new_staged_source_is_detected(self) -> None:
        self._repository()
        subject = capture_review_subject(self.root, self.pdf)
        (self.root / "new.py").write_text("value = 2\n", encoding="utf-8")
        # Untracked files are not part of the reviewed Git source contract.
        self.assertTrue(check_review_subject(self.root, subject)[0])
        self._git("add", "new.py")
        self.assertFalse(check_review_subject(self.root, subject)[0])

    def test_pipeline_outputs_do_not_invalidate_their_own_receipts(self) -> None:
        self._repository()
        save_contract_artifact(
            self.root, "review_state", {"rounds": []}, producer="test"
        )
        self._git("add", "review_state.json")
        self._git("commit", "-m", "track pipeline state")
        subject = capture_review_subject(self.root, self.pdf)
        save_contract_artifact(
            self.root, "review_state", {"rounds": ["new review"]}, producer="test"
        )
        self.assertTrue(check_review_subject(self.root, subject)[0])

    def test_run_directory_does_not_inherit_an_ancestor_repository(self) -> None:
        self._repository()
        nested = self.root / "run"
        nested.mkdir()
        shutil.copyfile(self.pdf, nested / "paper.pdf")
        subject = capture_review_subject(nested, "paper.pdf")
        self.assertEqual(subject["source_revision"], {"status": "not_applicable"})

    def test_unusable_git_metadata_and_source_limits_do_not_pass(self) -> None:
        self._repository()
        with patch("ai_scientist.utils.review_subject.MAX_SOURCE_FILES", 1):
            self.assertEqual(
                capture_review_subject(self.root, self.pdf)["status"], "unavailable"
            )
        with patch(
            "ai_scientist.utils.review_subject._git",
            side_effect=subprocess.TimeoutExpired("git", 10),
        ):
            self.assertEqual(
                capture_review_subject(self.root, self.pdf)["status"], "unavailable"
            )

    def _runtime_args(self, review_fn) -> dict:
        return {
            "paper_dir": self.root,
            "model_review": "test-model",
            "review_plan": {
                "review_instruction": "Review the supplied document.",
                "review_reflections": 1,
                "review_fewshot": 0,
                "review_ensemble": 1,
                "review_temperature": 0.0,
            },
            "create_client_fn": lambda model: (None, model),
            "load_paper_fn": lambda path: Path(path).read_bytes(),
            "perform_review_fn": review_fn,
            "perform_imgs_cap_ref_review_fn": lambda *args: {},
            "pdf_path_resolver": lambda root: str(self.pdf),
        }

    def _execute(self, review_fn) -> dict:
        with patch(
            "ai_scientist.protocol.llm_trace.active_ara_root", return_value=None
        ):
            return execute_review_pass(
                project_root=self.root,
                persist_job=True,
                review_role="rigor",
                **self._runtime_args(review_fn),
            )

    def test_runtime_records_subject_before_reading_and_rejects_mid_review_changes(
        self,
    ) -> None:
        def review(*args, **kwargs):
            self.pdf.write_bytes(b"%PDF-1.4\nreplaced during review\n")
            return {"Weaknesses": []}

        result = self._execute(review)
        self.assertFalse(result["found"])
        self.assertEqual(result["job"]["status"], "unbound_subject")
        self.assertEqual(result["job"]["subject_check"], "stale_review_subject")

    def test_runtime_accepts_unchanged_subject_but_not_an_empty_review(self) -> None:
        result = self._execute(lambda *args, **kwargs: {"Weaknesses": []})
        self.assertTrue(result["found"])
        self.assertTrue(
            check_review_subject(self.root, result["job"]["review_subject"])[0]
        )
        result = self._execute(lambda *args, **kwargs: {})
        self.assertFalse(result["found"])

    def test_multi_role_suite_requires_one_shared_version(self) -> None:
        calls = 0

        def resolver(root):
            nonlocal calls
            calls += 1
            if calls == 3:
                self.pdf.write_bytes(b"%PDF-1.4\nnew version between roles\n")
            return str(self.pdf)

        args = self._runtime_args(lambda *args, **kwargs: {"Weaknesses": []})
        args["pdf_path_resolver"] = resolver
        result = execute_review_suite(
            **args,
            review_roles=["rigor", "clarity"],
            persist_job=True,
            project_root=self.root,
        )
        self.assertFalse(result["found"])
        self.assertEqual(result["subject_check"], "mismatched_review_subjects")
        self.assertTrue(all(row["found"] for row in result["passes_by_role"].values()))

    def test_multi_role_suite_accepts_one_unchanged_version(self) -> None:
        result = execute_review_suite(
            **self._runtime_args(lambda *args, **kwargs: {"Weaknesses": []}),
            review_roles=["rigor", "clarity"],
            persist_job=True,
            project_root=self.root,
        )
        self.assertTrue(result["found"])
        self.assertTrue(check_review_subject(self.root, result["review_subject"])[0])

    def test_aggregate_artifact_does_not_record_a_pass_after_subject_drift(
        self,
    ) -> None:
        suite_dir = self.root / "reviews" / "suite"

        def save(destination, **kwargs):
            save_review_artifacts(destination, **kwargs)
            if Path(destination) == suite_dir:
                self.pdf.write_bytes(b"%PDF-1.4\nchanged during aggregation\n")

        result = execute_review_suite(
            **self._runtime_args(lambda *args, **kwargs: {"Weaknesses": []}),
            review_roles=["rigor", "clarity"],
            persist_job=True,
            project_root=self.root,
            save_dir=suite_dir,
            save_review_artifacts_fn=save,
        )
        summary = json.loads(
            (suite_dir / "review_suite.json").read_text(encoding="utf-8")
        )
        self.assertFalse(result["found"])
        self.assertFalse(summary["found"])
        self.assertEqual(summary["subject_check"], "stale_review_subject")

    def test_runtime_rejects_a_changed_pdf_selection_even_when_old_file_survives(
        self,
    ) -> None:
        other = self.root / "other.pdf"
        shutil.copyfile(self.pdf, other)
        selected = [self.pdf]

        def review(*args, **kwargs):
            selected[0] = other
            return {"Weaknesses": []}

        args = self._runtime_args(review)
        args["pdf_path_resolver"] = lambda root: str(selected[0])
        result = execute_review_pass(**args, project_root=self.root, persist_job=True)
        self.assertFalse(result["found"])
        self.assertEqual(result["job"]["subject_check"], "selected_pdf_changed")

    def test_independent_confirmation_cannot_confirm_a_different_version(self) -> None:
        plan = replace(
            build_workflow_runtime_plan("multi_agent_board"),
            critic_review_roles=("rigor",),
        )

        def run_suite(**kwargs):
            if kwargs["lane_name"] == "hostile_critic_confirmation":
                self.pdf.write_bytes(b"%PDF-1.4\nnew version before confirmation\n")
            return execute_review_suite(**kwargs)

        with patch(
            "ai_scientist.utils.critic_workflow.execute_review_suite",
            side_effect=run_suite,
        ):
            result = run_independent_critic_pass(
                **self._runtime_args(lambda *args, **kwargs: {"Weaknesses": []}),
                workflow_runtime_plan=plan,
                project_root=self.root,
                save_dir=self.root / "reviews" / "critic",
            )
        self.assertFalse(result["critic_confirmed"])
        self.assertEqual(
            result["critic_confirmation"]["reason"], "confirmation_subject_changed"
        )
        self.assertGreaterEqual(result["blocking_issue_count"], 1)


if __name__ == "__main__":
    unittest.main()
