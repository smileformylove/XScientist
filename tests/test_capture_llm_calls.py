"""Tests for the capture_llm_calls() context manager."""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import threading
import unittest
from pathlib import Path

from ai_scientist.protocol import capture_llm_calls, record_llm_call
from ai_scientist.protocol.llm_trace import (
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_REDACT,
    ENV_STAGE,
)


class _EnvGuard:
    _KEYS = (ENV_ACTIVE_ROOT, ENV_ENABLED, ENV_STAGE, ENV_REDACT)

    def __init__(self) -> None:
        self._snap = {k: os.environ.get(k) for k in self._KEYS}

    def restore(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class CaptureLLMCallsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._env = _EnvGuard()
        self.addCleanup(self._env.restore)
        for k in _EnvGuard._KEYS:
            os.environ.pop(k, None)
        os.environ[ENV_ACTIVE_ROOT] = str(self.root)
        os.environ[ENV_ENABLED] = "1"
        os.environ[ENV_REDACT] = "0"

    def _call(self, prompt: str, response: str = "r") -> None:
        record_llm_call(
            provider="p", model="m", request_style="r",
            system_message="s",
            messages=[{"role": "user", "content": prompt}],
            response_text=response,
        )

    # ------------------------------------------------------------------
    def test_captures_all_calls_in_block(self) -> None:
        with capture_llm_calls() as refs:
            self._call("a")
            self._call("b")
        self.assertEqual(len(refs), 2)
        for h in refs:
            self.assertTrue(h.startswith("sha256:"))

    def test_identical_prompts_return_identical_ref(self) -> None:
        with capture_llm_calls() as refs:
            self._call("same")
            self._call("same")
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0], refs[1])

    def test_no_capture_outside_block(self) -> None:
        # calls before the block do NOT appear in the block's list
        self._call("outside-before")
        with capture_llm_calls() as refs:
            self._call("inside")
        self._call("outside-after")
        self.assertEqual(len(refs), 1)

    def test_reset_between_blocks(self) -> None:
        with capture_llm_calls() as a:
            self._call("a1")
        with capture_llm_calls() as b:
            self._call("b1")
            self._call("b2")
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 2)
        # Inner blocks must not share buffers
        self.assertNotEqual(a is b, True)

    def test_nested_capture_propagates_to_outer(self) -> None:
        with capture_llm_calls() as outer:
            self._call("outer-only")
            with capture_llm_calls() as inner:
                self._call("inner-1")
                self._call("inner-2")
            # inner refs are still visible in outer
        self.assertEqual(len(inner), 2)
        self.assertEqual(len(outer), 3)
        for h in inner:
            self.assertIn(h, outer)

    def test_exception_still_restores_buffer(self) -> None:
        try:
            with capture_llm_calls() as refs:
                self._call("only")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # After the failed block, a new capture must start empty.
        with capture_llm_calls() as fresh:
            self._call("second")
        self.assertEqual(len(fresh), 1)

    def test_no_active_root_yields_empty_but_no_crash(self) -> None:
        # Tracing disabled → capture still yields a list, just stays empty.
        os.environ[ENV_ENABLED] = "0"
        with capture_llm_calls() as refs:
            self._call("noop")
        self.assertEqual(refs, [])

    def test_thread_isolation_via_copy_context(self) -> None:
        # ContextVar is per-Context, not per-Thread; two raw threads that
        # both call _capture_buffer.set() end up racing on the same slot.
        # Callers who need per-thread capture must run the block inside
        # copy_context().run() — this test documents that contract.
        import contextvars
        barrier = threading.Barrier(2)
        results: dict[str, list[str]] = {}

        def worker(tag: str) -> None:
            with capture_llm_calls() as refs:
                self._call(f"{tag}-1")
                barrier.wait()
                self._call(f"{tag}-2")
            results[tag] = list(refs)

        def run_in_own_context(tag: str) -> None:
            contextvars.copy_context().run(worker, tag)

        t1 = threading.Thread(target=run_in_own_context, args=("A",))
        t2 = threading.Thread(target=run_in_own_context, args=("B",))
        t1.start(); t2.start(); t1.join(); t2.join()

        self.assertEqual(len(results["A"]), 2)
        self.assertEqual(len(results["B"]), 2)
        # Refs from A must not appear in B
        self.assertFalse(set(results["A"]) & set(results["B"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
