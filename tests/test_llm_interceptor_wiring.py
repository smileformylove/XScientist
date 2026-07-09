"""End-to-end wiring test: get_response_from_llm records into ARA when active."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

from ai_scientist import llm as llm_mod
from ai_scientist.protocol import ObjectStore
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_REDACT,
    ENV_STAGE,
)


class _FakeUsage:
    def __init__(self, i: int, o: int) -> None:
        self.prompt_tokens = i
        self.completion_tokens = o


class _FakeChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeChoiceMessage(content)


class _FakeResponse:
    def __init__(self, content: str, tokens: tuple[int, int] = (3, 2)) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(*tokens)


class _FakeChatCompletions:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeChat:
    def __init__(self, response: _FakeResponse) -> None:
        self.completions = _FakeChatCompletions(response)


class _FakeOpenAIClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.chat = _FakeChat(response)


class LLMInterceptorWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        # Snapshot + reset relevant env
        self._snap = {k: os.environ.get(k) for k in
                      (ENV_ACTIVE_ROOT, ENV_ENABLED, ENV_STAGE, ENV_REDACT)}
        for k in self._snap:
            os.environ.pop(k, None)
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_records_call_and_populates_cas(self) -> None:
        os.environ[ENV_ACTIVE_ROOT] = str(self.root)
        os.environ[ENV_ENABLED] = "1"
        os.environ[ENV_REDACT] = "0"
        os.environ[ENV_STAGE] = "unit-test"

        client = _FakeOpenAIClient(_FakeResponse("pong", tokens=(7, 3)))
        content, hist = llm_mod.get_response_from_llm(
            prompt="ping",
            client=client,
            model="gpt-4o-mini",
            system_message="be terse",
            temperature=0.1,
        )

        self.assertEqual(content, "pong")
        # Assistant reply must be appended to history
        self.assertEqual(hist[-1], {"role": "assistant", "content": "pong"})

        # ARA jsonl exists and has exactly one row
        log = self.root / CALLS_JSONL_RELPATH
        self.assertTrue(log.exists())
        rows = [json.loads(l) for l in log.read_text().splitlines() if l]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["stage"], "unit-test")
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["model"], "gpt-4o-mini")
        self.assertEqual(row["tokens"], {"input": 7, "output": 3})
        self.assertIsInstance(row["latency_ms"], int)
        self.assertGreaterEqual(row["latency_ms"], 0)

        # CAS blobs must be resolvable and readable
        store = ObjectStore(self.root)
        self.assertTrue(store.exists(row["messages_ref"]["hash"]))
        self.assertTrue(store.exists(row["response_ref"]["hash"]))
        self.assertEqual(store.get_text(row["response_ref"]["hash"]), "pong")
        msg_payload = store.get_json(row["messages_ref"]["hash"])
        self.assertEqual(msg_payload["system"], "be terse")

    def test_no_active_root_leaves_no_trace(self) -> None:
        # Tracer disabled → no side effects at all
        client = _FakeOpenAIClient(_FakeResponse("silent"))
        content, _ = llm_mod.get_response_from_llm(
            prompt="q", client=client, model="gpt-4o-mini",
            system_message="s", temperature=0.0,
        )
        self.assertEqual(content, "silent")
        self.assertFalse((self.root / CALLS_JSONL_RELPATH).exists())
        self.assertFalse((self.root / "objects").exists())

    def test_tracer_failure_does_not_break_llm_call(self) -> None:
        # Point ENV_ACTIVE_ROOT at a file (not a dir). active_ara_root should
        # return None; the LLM call must still succeed.
        bogus = self.root / "not_a_dir.txt"
        bogus.write_text("x")
        os.environ[ENV_ACTIVE_ROOT] = str(bogus)
        os.environ[ENV_ENABLED] = "1"
        client = _FakeOpenAIClient(_FakeResponse("still-fine"))
        content, _ = llm_mod.get_response_from_llm(
            prompt="q", client=client, model="gpt-4o-mini",
            system_message="s", temperature=0.0,
        )
        self.assertEqual(content, "still-fine")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
