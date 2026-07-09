"""Real-world composition: capture_llm_calls() around get_response_from_llm.

This is the integration point parallel_agent._draft uses. If the wrapper
here works, then _draft/_debug/_improve automatically populate
Node.llm_call_refs from real LLM calls."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_scientist import llm as llm_mod
from ai_scientist.protocol import capture_llm_calls
from ai_scientist.protocol.llm_trace import (
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_REDACT,
    ENV_STAGE,
)


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 1
        self.completion_tokens = 1


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _FakeChatCompletions:
    def __init__(self, response: _Resp) -> None:
        self._response = response
    def create(self, **kwargs):
        return self._response


class _FakeClient:
    def __init__(self, response: _Resp) -> None:
        class _Chat: pass
        self.chat = _Chat()
        self.chat.completions = _FakeChatCompletions(response)


class CaptureAroundGetResponseFromLLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._snap = {k: os.environ.get(k) for k in
                      (ENV_ACTIVE_ROOT, ENV_ENABLED, ENV_STAGE, ENV_REDACT)}
        for k in self._snap:
            os.environ.pop(k, None)
        os.environ[ENV_ACTIVE_ROOT] = str(self.root)
        os.environ[ENV_ENABLED] = "1"
        os.environ[ENV_REDACT] = "0"
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_capture_collects_llm_call_ref(self) -> None:
        client = _FakeClient(_Resp("hi back"))
        with capture_llm_calls() as refs:
            llm_mod.get_response_from_llm(
                prompt="ping", client=client, model="gpt-4o-mini",
                system_message="s", temperature=0.5,
            )
        self.assertEqual(len(refs), 1)
        self.assertTrue(refs[0].startswith("sha256:"))

    def test_multiple_llm_calls_in_capture(self) -> None:
        client = _FakeClient(_Resp("resp"))
        with capture_llm_calls() as refs:
            for i in range(3):
                llm_mod.get_response_from_llm(
                    prompt=f"prompt-{i}", client=client, model="gpt-4o-mini",
                    system_message="s", temperature=0.5,
                )
        self.assertEqual(len(refs), 3)

    def test_same_prompt_two_calls_returns_same_ref(self) -> None:
        # This is what makes Node.llm_call_refs de-dupe cleanly across
        # multiple identical draft attempts.
        client = _FakeClient(_Resp("out"))
        with capture_llm_calls() as refs:
            llm_mod.get_response_from_llm(
                prompt="dup", client=client, model="gpt-4o-mini",
                system_message="s", temperature=0.5,
            )
            llm_mod.get_response_from_llm(
                prompt="dup", client=client, model="gpt-4o-mini",
                system_message="s", temperature=0.5,
            )
        self.assertEqual(len(refs), 2)  # two records
        self.assertEqual(refs[0], refs[1])  # same messages_ref

    def test_no_capture_no_side_effect_on_calls_jsonl(self) -> None:
        # Confirms capture is orthogonal to the on-disk journaling.
        client = _FakeClient(_Resp("outside"))
        llm_mod.get_response_from_llm(
            prompt="q", client=client, model="gpt-4o-mini",
            system_message="s", temperature=0.5,
        )
        rows = [json.loads(l) for l in
                (self.root / "llm" / "calls.jsonl").read_text().splitlines() if l]
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
