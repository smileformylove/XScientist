from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist import CommandResult, ProjectRequest
from xscientist.service_jobs import Job, JobStore


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


if __name__ == "__main__":
    unittest.main()
