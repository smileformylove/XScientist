"""Unit tests for activate_llm_tracing / deactivate_llm_tracing."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol.llm_trace import ENV_ACTIVE_ROOT, ENV_STAGE
from ai_scientist.utils.ara_pipeline import (
    activate_llm_tracing,
    deactivate_llm_tracing,
)


class ActivateLLMTracingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project = Path(self._tmp.name) / "proj"
        self.project.mkdir()
        self._snap = {k: os.environ.get(k) for k in (ENV_ACTIVE_ROOT, ENV_STAGE)}
        for k in self._snap:
            os.environ.pop(k, None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_sets_env_and_creates_ara_dir(self) -> None:
        ara_dir = activate_llm_tracing(
            project_dir=self.project,
            idea={"Name": "my_idea"},
            timestamp="20260709T120000",
            stage="ideation",
        )
        self.assertIsNotNone(ara_dir)
        self.assertTrue(ara_dir.is_dir())
        self.assertEqual(os.environ[ENV_ACTIVE_ROOT], str(ara_dir))
        self.assertEqual(os.environ[ENV_STAGE], "ideation")

    def test_returns_none_when_no_name(self) -> None:
        ara_dir = activate_llm_tracing(
            project_dir=self.project, idea={}, timestamp=None
        )
        self.assertIsNone(ara_dir)
        self.assertNotIn(ENV_ACTIVE_ROOT, os.environ)

    def test_deactivate_clears_env(self) -> None:
        activate_llm_tracing(
            project_dir=self.project,
            idea={"Name": "x"},
            timestamp="ts",
            stage="review",
        )
        self.assertIn(ENV_ACTIVE_ROOT, os.environ)
        self.assertIn(ENV_STAGE, os.environ)
        deactivate_llm_tracing()
        self.assertNotIn(ENV_ACTIVE_ROOT, os.environ)
        self.assertNotIn(ENV_STAGE, os.environ)

    def test_deactivate_is_idempotent(self) -> None:
        # Never activated → no crash.
        deactivate_llm_tracing()
        deactivate_llm_tracing()

    def test_exp_dir_without_timestamp_falls_back(self) -> None:
        # exp_dir names without a leading timestamp are common; activation
        # must still succeed, just without a timestamp segment in the ARA path.
        exp = self.project / "02_experiments" / "20260709_my_idea"
        exp.mkdir(parents=True)
        ara_dir = activate_llm_tracing(
            project_dir=self.project,
            idea={"Name": "my_idea"},
            exp_dir=exp,
        )
        self.assertIsNotNone(ara_dir)
        # No timestamp extracted → ARA dir is just <root>/ara/<slug>
        self.assertTrue(str(ara_dir).endswith("/ara/my_idea"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
