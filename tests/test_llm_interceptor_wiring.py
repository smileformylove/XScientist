"""End-to-end wiring test: get_response_from_llm records into ARA when active."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist import llm as llm_mod
from ai_scientist.protocol import ObjectStore
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_REDACT,
    ENV_STAGE,
    ENV_STRICT,
)


class _FakeUsage:
    def __init__(self, i: int, o: int) -> None:
        self.prompt_tokens = i
        self.completion_tokens = o


class _FakeChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, *, finish_reason: str = "stop") -> None:
        self.message = _FakeChoiceMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(
        self,
        content: str,
        tokens: tuple[int, int] = (3, 2),
        *,
        model: str | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.choices = [_FakeChoice(content, finish_reason=finish_reason)]
        self.usage = _FakeUsage(*tokens)
        self.model = model


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
        self._snap = {
            k: os.environ.get(k)
            for k in (
                ENV_ACTIVE_ROOT,
                ENV_ENABLED,
                ENV_STAGE,
                ENV_REDACT,
                ENV_STRICT,
                "OPENAI_COMPAT_API_KEY",
                "OPENAI_COMPAT_BASE_URL",
            )
        }
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
        self.assertTrue(store.exists(row["call_receipt_ref"]["hash"]))
        response_digest = store.get_json(row["response_ref"]["hash"])
        msg_payload = store.get_json(row["messages_ref"]["hash"])
        self.assertFalse(response_digest["payload_recorded"])
        self.assertFalse(msg_payload["payload_recorded"])
        self.assertNotIn("pong", json.dumps(response_digest))
        self.assertNotIn("be terse", json.dumps(msg_payload))

    def test_no_active_root_leaves_no_trace(self) -> None:
        # Tracer disabled → no side effects at all
        client = _FakeOpenAIClient(_FakeResponse("silent"))
        content, _ = llm_mod.get_response_from_llm(
            prompt="q",
            client=client,
            model="gpt-4o-mini",
            system_message="s",
            temperature=0.0,
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
            prompt="q",
            client=client,
            model="gpt-4o-mini",
            system_message="s",
            temperature=0.0,
        )
        self.assertEqual(content, "still-fine")

    def test_custom_route_requires_exact_identity_and_uses_bounded_timeout(
        self,
    ) -> None:
        os.environ[ENV_ACTIVE_ROOT] = str(self.root)
        os.environ[ENV_ENABLED] = "1"
        os.environ[ENV_STRICT] = "1"
        os.environ["OPENAI_COMPAT_API_KEY"] = "compat-test-key"
        os.environ["OPENAI_COMPAT_BASE_URL"] = "https://gateway.example/v1"
        client = _FakeOpenAIClient(_FakeResponse("research-result", model="glm-5.3"))

        content, _ = llm_mod.get_response_from_llm(
            prompt="research-task",
            client=client,
            model="openai_compat/glm-5.3",
            system_message="execute only",
            temperature=0.0,
        )

        self.assertEqual(content, "research-result")
        request = client.chat.completions.calls[0]
        self.assertEqual(request["model"], "glm-5.3")
        self.assertEqual(request["timeout"], llm_mod.OPENAI_COMPAT_CALL_TIMEOUT_SECONDS)
        [row] = [
            json.loads(line)
            for line in (self.root / CALLS_JSONL_RELPATH).read_text().splitlines()
        ]
        self.assertEqual(row["provider"], "openai_compat")
        self.assertEqual(row["model_provenance"]["reported_model"], "glm-5.3")

    def test_custom_route_rejects_missing_alias_and_secret_model_without_trace(
        self,
    ) -> None:
        os.environ[ENV_ACTIVE_ROOT] = str(self.root)
        os.environ[ENV_ENABLED] = "1"
        for reported_model in (
            None,
            "openai_compat/glm-5.3",
            "glm-5.3-alias",
            "sk-" + "Q" * 40,
        ):
            with self.subTest(reported_model=type(reported_model).__name__):
                client = _FakeOpenAIClient(
                    _FakeResponse("untrusted", model=reported_model)
                )
                with self.assertRaisesRegex(
                    llm_mod.LLMResponseContractError,
                    "identity is not exact",
                ):
                    llm_mod.get_response_from_llm(
                        prompt="research-task",
                        client=client,
                        model="openai_compat/glm-5.3",
                        system_message="execute only",
                        temperature=0.0,
                    )

        self.assertFalse((self.root / CALLS_JSONL_RELPATH).exists())

    def test_create_client_preserves_custom_route_contract_end_to_end(self) -> None:
        os.environ["OPENAI_COMPAT_API_KEY"] = "compat-test-key"
        os.environ["OPENAI_COMPAT_BASE_URL"] = "https://gateway.example/v1"
        fake_client = _FakeOpenAIClient(
            _FakeResponse("untrusted", model="glm-5.3-alias")
        )

        with mock.patch.object(llm_mod.openai, "OpenAI", return_value=fake_client):
            client, routed_model = llm_mod.create_client("openai_compat/glm-5.3")

        self.assertEqual(routed_model, "openai_compat/glm-5.3")
        with self.assertRaises(llm_mod.LLMResponseContractError):
            llm_mod.get_response_from_llm(
                prompt="research-task",
                client=client,
                model=routed_model,
                system_message="execute only",
                temperature=0.0,
            )

    def test_text_batch_count_is_bounded_before_provider_call(self) -> None:
        for count in (0, -1, True, 9):
            with self.subTest(count=count):
                client = _FakeOpenAIClient(_FakeResponse("unused"))
                with self.assertRaisesRegex(ValueError, "between 1 and 8"):
                    llm_mod.get_batch_responses_from_llm(
                        prompt="q",
                        client=client,
                        model="gpt-4o-mini",
                        system_message="s",
                        n_responses=count,
                    )
                self.assertEqual(client.chat.completions.calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
