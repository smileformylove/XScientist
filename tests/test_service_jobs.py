from __future__ import annotations

import json
import threading
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist import CommandResult, ProjectRequest
from xscientist.service_jobs import Job, JobStore, WorkspaceBusyError


class ServiceJobTests(unittest.TestCase):
    def test_job_round_trip_preserves_request_and_result(self) -> None:
        job = Job(
            id="job-1",
            request=ProjectRequest(project="demo", topic="topic.md"),
            status="succeeded",
            result=CommandResult(
                command=("python", "run.py"),
                returncode=0,
                stdout="done",
                stderr="",
                started_at="start",
                finished_at="finish",
            ),
        )

        restored = Job.from_dict(job.to_dict())

        self.assertEqual(restored.id, "job-1")
        self.assertEqual(restored.request.project, "demo")
        self.assertEqual(restored.result.stdout, "done")

    def test_store_restores_incomplete_jobs_as_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            (state_dir / "unfinished.json").write_text(
                json.dumps(
                    {
                        "id": "unfinished",
                        "status": "running",
                        "request": ProjectRequest(
                            project="demo", topic="topic.md"
                        ).to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            client = mock.Mock()
            store = JobStore(
                client,
                max_workers=1,
                max_output_chars=20,
                state_dir=state_dir,
            )
            try:
                job = store.get("unfinished")
            finally:
                store.shutdown()

            saved = json.loads(
                (state_dir / "unfinished.json").read_text(encoding="utf-8")
            )

        self.assertEqual(job.status, "interrupted")
        self.assertIn("restarted", job.error)
        self.assertEqual(saved["status"], "interrupted")

    def test_store_truncates_completed_process_output(self) -> None:
        client = mock.Mock()
        client.run_project.return_value = CommandResult(
            command=("python", "run.py"),
            returncode=0,
            stdout="0123456789",
            stderr="abcdefghij",
            started_at="start",
            finished_at="finish",
        )
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=1,
                max_output_chars=4,
                state_dir=td,
            )
            try:
                job = store.submit(ProjectRequest(project="demo", topic="topic.md"))
                store.futures[job.id].result(timeout=5)
                completed = store.get(job.id)
            finally:
                store.shutdown()

        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(completed.result.stdout, "6789")
        self.assertEqual(completed.result.stderr, "ghij")

    def test_state_io_does_not_hold_global_scheduler_lock(self) -> None:
        client = mock.Mock()
        client.run_project.return_value = CommandResult(
            command=("python", "run.py"),
            returncode=0,
            stdout="done",
            stderr="",
            started_at="start",
            finished_at="finish",
        )
        observed_lock_states: list[bool] = []
        store = None

        def write_json(path, payload) -> None:
            if store is not None:
                observed_lock_states.append(store.lock.locked())
            Path(path).write_text(json.dumps(payload), encoding="utf-8")

        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=2,
                max_output_chars=20,
                state_dir=td,
                write_json=write_json,
            )
            try:
                job = store.submit(ProjectRequest(project="demo", topic="topic.md"))
                store.futures[job.id].result(timeout=5)
            finally:
                store.shutdown()

        self.assertEqual(observed_lock_states, [False, False, False])

    def test_same_resolved_workspace_is_rejected_until_completion(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def run_project(_request, **_kwargs):
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return CommandResult(
                command=("python", "run.py"),
                returncode=0,
                stdout="done",
                stderr="",
                started_at="start",
                finished_at="finish",
            )

        client = mock.Mock(output_root=None, env={}, work_dir=None)
        client.run_project.side_effect = run_project
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "output"
            output_root.mkdir()
            (output_root / "nested").mkdir()
            output_alias = output_root / "nested" / ".."
            store = JobStore(
                client,
                max_workers=2,
                max_output_chars=100,
                state_dir=Path(td) / "state",
            )
            try:
                first = store.submit(
                    ProjectRequest(
                        project="demo",
                        topic="topic.md",
                        output_root=output_root,
                    )
                )
                self.assertTrue(entered.wait(timeout=2))

                with self.assertRaises(WorkspaceBusyError) as raised:
                    store.submit(
                        ProjectRequest(
                            project="demo",
                            topic="other-topic.md",
                            output_root=output_alias,
                        )
                    )

                self.assertEqual(raised.exception.code, "workspace_busy")
                self.assertEqual(raised.exception.active_job_id, first.id)
                self.assertEqual(
                    str(raised.exception),
                    "workspace_busy: project workspace already has an active job "
                    f"({first.id})",
                )
                self.assertEqual(len(list((Path(td) / "state").glob("*.json"))), 1)

                release.set()
                store.futures[first.id].result(timeout=5)
                self.assertEqual(store.get(first.id).status, "succeeded")

                second = store.submit(
                    ProjectRequest(
                        project="demo",
                        topic="other-topic.md",
                        output_root=output_alias,
                    )
                )
                store.futures[second.id].result(timeout=5)
                self.assertEqual(store.get(second.id).status, "succeeded")
            finally:
                release.set()
                store.shutdown()

    def test_macos_case_aliases_share_one_workspace_reservation(self) -> None:
        client = mock.Mock(output_root=None, env={}, work_dir=None)
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=1,
                max_output_chars=100,
                state_dir=Path(td) / "state",
            )
            upper = Path(td) / "projects" / "Demo"
            lower = Path(td) / "projects" / "demo"
            try:
                with mock.patch("xscientist.service_jobs.sys.platform", "darwin"):
                    store._reserve_workspace("first", upper)
                    with self.assertRaises(WorkspaceBusyError) as raised:
                        store._reserve_workspace("second", lower)
                    self.assertEqual(raised.exception.active_job_id, "first")
                    store._release_workspace("first")
                    store._reserve_workspace("second", lower)
                    store._release_workspace("second")
            finally:
                store.shutdown()

    def test_windows_trailing_dot_and_space_aliases_share_reservation_key(
        self,
    ) -> None:
        with mock.patch("xscientist.service_jobs.sys.platform", "win32"):
            canonical = JobStore._workspace_reservation_key(
                Path("C:/research/projects/demo")
            )
            dotted = JobStore._workspace_reservation_key(
                Path("C:/research/projects/Demo. ")
            )

        self.assertEqual(canonical, dotted)

    def test_concurrent_same_workspace_submissions_admit_exactly_one(self) -> None:
        submit_barrier = threading.Barrier(3)
        run_entered = threading.Event()
        release = threading.Event()
        outcomes_lock = threading.Lock()
        outcomes: list[Job | WorkspaceBusyError] = []

        def run_project(_request, **_kwargs):
            run_entered.set()
            self.assertTrue(release.wait(timeout=5))
            return CommandResult(
                command=("python", "run.py"),
                returncode=0,
                stdout="done",
                stderr="",
                started_at="start",
                finished_at="finish",
            )

        client = mock.Mock(output_root=None, env={}, work_dir=None)
        client.run_project.side_effect = run_project
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=2,
                max_output_chars=100,
                state_dir=Path(td) / "state",
            )
            request = ProjectRequest(
                project="demo",
                topic="topic.md",
                output_root=td,
            )

            def submit() -> None:
                submit_barrier.wait(timeout=5)
                try:
                    outcome: Job | WorkspaceBusyError = store.submit(request)
                except WorkspaceBusyError as exc:
                    outcome = exc
                with outcomes_lock:
                    outcomes.append(outcome)

            threads = [threading.Thread(target=submit) for _ in range(2)]
            try:
                for thread in threads:
                    thread.start()
                submit_barrier.wait(timeout=5)
                for thread in threads:
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())

                accepted = [item for item in outcomes if isinstance(item, Job)]
                rejected = [
                    item for item in outcomes if isinstance(item, WorkspaceBusyError)
                ]
                self.assertEqual(len(accepted), 1)
                self.assertEqual(len(rejected), 1)
                self.assertEqual(rejected[0].active_job_id, accepted[0].id)
                self.assertTrue(run_entered.wait(timeout=2))
                self.assertEqual(client.run_project.call_count, 1)

                release.set()
                store.futures[accepted[0].id].result(timeout=5)
            finally:
                release.set()
                for thread in threads:
                    thread.join(timeout=5)
                store.shutdown()

    def test_different_workspaces_can_run_in_parallel(self) -> None:
        both_entered = threading.Event()
        release = threading.Event()
        active_lock = threading.Lock()
        active_projects: set[str] = set()

        def run_project(request, **_kwargs):
            with active_lock:
                active_projects.add(request.project)
                if len(active_projects) == 2:
                    both_entered.set()
            self.assertTrue(release.wait(timeout=5))
            return CommandResult(
                command=("python", "run.py"),
                returncode=0,
                stdout=request.project,
                stderr="",
                started_at="start",
                finished_at="finish",
            )

        client = mock.Mock(output_root=None, env={}, work_dir=None)
        client.run_project.side_effect = run_project
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=2,
                max_output_chars=100,
                state_dir=Path(td) / "state",
            )
            try:
                first = store.submit(
                    ProjectRequest(
                        project="alpha",
                        topic="topic.md",
                        output_root=td,
                    )
                )
                second = store.submit(
                    ProjectRequest(
                        project="beta",
                        topic="topic.md",
                        output_root=td,
                    )
                )
                self.assertTrue(both_entered.wait(timeout=2))
                self.assertEqual(active_projects, {"alpha", "beta"})
                release.set()
                store.futures[first.id].result(timeout=5)
                store.futures[second.id].result(timeout=5)
            finally:
                release.set()
                store.shutdown()

    def test_failed_job_releases_workspace(self) -> None:
        attempts = 0

        def run_project(_request, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("boom")
            return CommandResult(
                command=("python", "run.py"),
                returncode=0,
                stdout="recovered",
                stderr="",
                started_at="start",
                finished_at="finish",
            )

        client = mock.Mock(output_root=None, env={}, work_dir=None)
        client.run_project.side_effect = run_project
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=1,
                max_output_chars=100,
                state_dir=Path(td) / "state",
            )
            request = ProjectRequest(
                project="demo",
                topic="topic.md",
                output_root=td,
            )
            try:
                failed = store.submit(request)
                store.futures[failed.id].result(timeout=5)
                self.assertEqual(store.get(failed.id).status, "failed")

                retry = store.submit(request)
                store.futures[retry.id].result(timeout=5)
                self.assertEqual(store.get(retry.id).status, "succeeded")
            finally:
                store.shutdown()

    def test_cancelled_queued_job_releases_workspace(self) -> None:
        holder_entered = threading.Event()
        release_holder = threading.Event()

        def run_project(request, **_kwargs):
            if request.project == "holder":
                holder_entered.set()
                self.assertTrue(release_holder.wait(timeout=5))
            return CommandResult(
                command=("python", "run.py"),
                returncode=0,
                stdout=request.project,
                stderr="",
                started_at="start",
                finished_at="finish",
            )

        client = mock.Mock(output_root=None, env={}, work_dir=None)
        client.run_project.side_effect = run_project
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=1,
                max_output_chars=100,
                state_dir=Path(td) / "state",
            )
            try:
                holder = store.submit(
                    ProjectRequest(project="holder", topic="topic.md", output_root=td)
                )
                self.assertTrue(holder_entered.wait(timeout=2))
                queued = store.submit(
                    ProjectRequest(project="target", topic="topic.md", output_root=td)
                )

                cancelled = store.cancel(queued.id)
                self.assertEqual(cancelled.status, "cancelled")

                replacement = store.submit(
                    ProjectRequest(project="target", topic="topic.md", output_root=td)
                )
                release_holder.set()
                store.futures[holder.id].result(timeout=5)
                store.futures[replacement.id].result(timeout=5)
                self.assertEqual(store.get(replacement.id).status, "succeeded")
            finally:
                release_holder.set()
                store.shutdown()

    def test_running_job_can_be_cancelled_and_resumed(self) -> None:
        entered = threading.Event()

        def run_project(request, **kwargs):
            if request.resume:
                return CommandResult(
                    command=("python", "run.py", "--resume"),
                    returncode=0,
                    stdout="resumed",
                    stderr="",
                    started_at="start-2",
                    finished_at="finish-2",
                )
            entered.set()
            self.assertIn("cancel_check", kwargs)
            self.assertTrue(entered.wait(timeout=1))
            while not kwargs["cancel_check"]():
                threading.Event().wait(0.01)
            kwargs["output_callback"]("partial", "", False, False)
            return CommandResult(
                command=("python", "run.py"),
                returncode=130,
                stdout="partial",
                stderr="RunCancelled",
                started_at="start",
                finished_at="finish",
            )

        client = mock.Mock()
        client.run_project.side_effect = run_project
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(
                client,
                max_workers=1,
                max_output_chars=100,
                state_dir=td,
            )
            try:
                submitted = store.submit(
                    ProjectRequest(project="demo", topic="topic.md")
                )
                self.assertTrue(entered.wait(timeout=2))
                cancelling = store.cancel(submitted.id)
                self.assertEqual(cancelling.status, "cancelling")
                store.futures[submitted.id].result(timeout=5)
                cancelled = store.get(submitted.id)
                self.assertEqual(cancelled.status, "cancelled")
                self.assertEqual(cancelled.stdout_tail, "partial")

                resumed = store.resume(submitted.id)
                self.assertEqual(resumed.resume_of, submitted.id)
                self.assertTrue(resumed.request.resume)
                store.futures[resumed.id].result(timeout=5)
                self.assertEqual(store.get(resumed.id).status, "succeeded")
            finally:
                store.shutdown()


if __name__ == "__main__":
    unittest.main()
