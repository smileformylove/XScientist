"""Wiring test: run_project's process_single_idea activates and deactivates the LLM tracer.

We can't invoke the real process_single_idea (it takes a huge args tuple and
runs the full pipeline). What we CAN prove cheaply is:

* the finally block clears tracer env vars — even when the try body raises
* activate_llm_tracing + record_llm_call actually round-trip through the
  ARA dir that process_single_idea would create on line 826

This is a defense-in-depth test: the individual pieces are covered by
test_activate_llm_tracing / test_llm_trace / test_llm_interceptor_wiring.
Here we make sure they compose the way run_project.py expects.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import ObjectStore, record_llm_call
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_STAGE,
)
from ai_scientist.utils.ara_pipeline import (
    activate_llm_tracing,
    deactivate_llm_tracing,
)


class RunProjectTracingWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = Path(self._tmp.name) / "proj"
        self.project.mkdir()
        self._snap = {k: os.environ.get(k) for k in (ENV_ACTIVE_ROOT, ENV_STAGE, ENV_ENABLED)}
        for k in self._snap:
            os.environ.pop(k, None)
        os.environ[ENV_ENABLED] = "1"
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_finally_deactivates_even_on_exception(self) -> None:
        # Simulate process_single_idea: activate at the top of try,
        # raise mid-body, finally must clear env.
        try:
            activate_llm_tracing(
                project_dir=self.project,
                idea={"Name": "crashing_idea"},
                timestamp="20260709_120000",
                stage="pipeline",
            )
            self.assertIn(ENV_ACTIVE_ROOT, os.environ)
            raise RuntimeError("simulated pipeline failure")
        except RuntimeError:
            pass
        finally:
            deactivate_llm_tracing()

        # Env must be clean; next idea's activate call sees fresh slate.
        self.assertNotIn(ENV_ACTIVE_ROOT, os.environ)
        self.assertNotIn(ENV_STAGE, os.environ)

    def test_llm_calls_land_in_expected_ara_dir(self) -> None:
        activated = activate_llm_tracing(
            project_dir=self.project,
            idea={"Name": "flow_test"},
            timestamp="20260709_120000",
            stage="pipeline",
        )
        try:
            record_llm_call(
                provider="openai",
                model="gpt-4o-mini",
                request_style="openai_chat",
                system_message="s",
                messages=[{"role": "user", "content": "hello"}],
                response_text="hi",
                params={"temperature": 0.7},
            )
        finally:
            deactivate_llm_tracing()

        expected_ara = (self.project.resolve() / "ara" / "20260709_120000_flow_test")
        self.assertEqual(activated, expected_ara)
        rows = self._read_rows(expected_ara)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "pipeline")

        store = ObjectStore(expected_ara)
        self.assertTrue(store.exists(rows[0]["messages_ref"]["hash"]))

    def test_two_ideas_do_not_bleed_into_each_other(self) -> None:
        # First idea
        activate_llm_tracing(
            project_dir=self.project, idea={"Name": "first"},
            timestamp="20260709_100000", stage="pipeline",
        )
        try:
            record_llm_call(
                provider="p", model="m", request_style="r",
                system_message="s", messages=[{"role": "user", "content": "a"}],
                response_text="1",
            )
        finally:
            deactivate_llm_tracing()

        # Second idea — must get its own ARA dir
        activate_llm_tracing(
            project_dir=self.project, idea={"Name": "second"},
            timestamp="20260709_110000", stage="pipeline",
        )
        try:
            record_llm_call(
                provider="p", model="m", request_style="r",
                system_message="s", messages=[{"role": "user", "content": "b"}],
                response_text="2",
            )
        finally:
            deactivate_llm_tracing()

        first = self.project.resolve() / "ara" / "20260709_100000_first"
        second = self.project.resolve() / "ara" / "20260709_110000_second"
        self.assertEqual(len(self._read_rows(first)), 1)
        self.assertEqual(len(self._read_rows(second)), 1)
        # Contents differ (no cross-contamination of message blobs)
        self.assertNotEqual(
            self._read_rows(first)[0]["messages_ref"]["hash"],
            self._read_rows(second)[0]["messages_ref"]["hash"],
        )

    def _read_rows(self, ara_dir: Path) -> list[dict]:
        p = ara_dir / CALLS_JSONL_RELPATH
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
