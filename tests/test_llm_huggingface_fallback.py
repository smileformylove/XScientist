from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from ai_scientist import llm


class _IncompatibleClient:
    def __init__(self, error: Exception | None = None) -> None:
        create = mock.Mock(
            side_effect=error
            or TypeError("endpoint does not implement chat completions")
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class HuggingFaceFallbackTests(unittest.TestCase):
    def test_compatibility_fallback_uses_and_records_requested_model(self) -> None:
        response = SimpleNamespace(
            status_code=200,
            json=lambda: [{"generated_text": "fallback result"}],
            usage=None,
        )
        spec = SimpleNamespace(
            provider="huggingface",
            request_style="huggingface_chat",
            client_model="org/private-model",
        )
        with (
            mock.patch.object(llm, "resolve_model_provider", return_value=spec),
            mock.patch.object(llm.requests, "post", return_value=response) as post,
            mock.patch.object(llm, "_record_llm_call_safe") as trace,
            mock.patch.dict(
                os.environ, {"HUGGINGFACE_API_KEY": "demo-key"}, clear=False
            ),
        ):
            content, _history = llm.get_response_from_llm(
                "hello",
                _IncompatibleClient(),
                "huggingface/org/private-model",
                "system",
            )

        self.assertEqual(content, "fallback result")
        self.assertEqual(
            post.call_args.args[0],
            "https://api-inference.huggingface.co/models/org/private-model",
        )
        self.assertEqual(trace.call_args.kwargs["model"], "org/private-model")
        self.assertEqual(trace.call_args.kwargs["provider"], "huggingface_http")
        self.assertEqual(
            trace.call_args.kwargs["request_style"], "huggingface_inference"
        )
        self.assertEqual(
            trace.call_args.kwargs["params"]["actual_model"], "org/private-model"
        )

    def test_authentication_failure_never_triggers_http_fallback(self) -> None:
        client = _IncompatibleClient(RuntimeError("authentication failed"))
        with mock.patch.object(llm.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                llm.get_response_from_llm(
                    "hello",
                    client,
                    "huggingface/org/private-model",
                    "system",
                )
        post.assert_not_called()

    def test_strict_trace_failure_propagates_to_model_caller(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="model response"))
            ],
            usage=None,
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=mock.Mock(return_value=response))
            )
        )
        with (
            mock.patch.object(
                llm, "record_llm_call", side_effect=OSError("trace unavailable")
            ),
            mock.patch.object(llm, "strict_llm_tracing", return_value=True),
        ):
            with self.assertRaisesRegex(OSError, "trace unavailable"):
                llm.get_response_from_llm("hello", client, "gpt-4o-mini", "system")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
