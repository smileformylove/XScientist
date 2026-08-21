from __future__ import annotations

import unittest
from unittest import mock

from ai_scientist.utils.provider_registry import (
    build_openai_compatible_client_kwargs,
    missing_model_credentials,
    probe_openai_compatible_model,
    provider_env_statuses,
    resolve_model_provider,
)


class ProviderRegistryTests(unittest.TestCase):
    def test_resolve_model_provider_should_normalize_prefixed_and_legacy_models(
        self,
    ) -> None:
        openrouter_spec = resolve_model_provider(
            "openrouter/meta-llama/llama-3.1-405b-instruct"
        )
        legacy_spec = resolve_model_provider("llama3.1-405b")
        compat_spec = resolve_model_provider("openai_compat/qwen2.5-72b-instruct")

        self.assertEqual(openrouter_spec.provider, "openrouter")
        self.assertEqual(
            openrouter_spec.client_model, "meta-llama/llama-3.1-405b-instruct"
        )
        self.assertEqual(legacy_spec.provider, "openrouter")
        self.assertEqual(compat_spec.provider, "openai_compat")
        self.assertEqual(compat_spec.client_model, "qwen2.5-72b-instruct")

        custom_spec = resolve_model_provider("custom/gpt-5.6-luna")
        self.assertEqual(custom_spec.provider, "openai_compat")
        self.assertEqual(custom_spec.client_model, "gpt-5.6-luna")

    def test_build_openai_compatible_client_kwargs_should_respect_provider_envs(
        self,
    ) -> None:
        kwargs, model = build_openai_compatible_client_kwargs(
            "gemini/gemini-2.5-pro-preview-03-25",
            env={"GOOGLE_API_KEY": "gem-key"},
            max_retries=3,
        )

        self.assertEqual(model, "gemini-2.5-pro-preview-03-25")
        self.assertEqual(kwargs["api_key"], "gem-key")
        self.assertEqual(
            kwargs["base_url"],
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.assertEqual(kwargs["max_retries"], 3)

    def test_missing_model_credentials_should_report_openai_compat_base_url(
        self,
    ) -> None:
        missing = missing_model_credentials(
            ["openai_compat/custom-model"],
            env={"OPENAI_COMPAT_API_KEY": "compat-key"},
        )
        self.assertEqual(len(missing), 1)
        self.assertIn("OPENAI_COMPAT_BASE_URL | OPENAI_BASE_URL", missing[0]["missing"])

    def test_openai_compat_uses_configurable_neutral_user_agent(self) -> None:
        kwargs, model = build_openai_compatible_client_kwargs(
            "openai_compat/research-model",
            env={
                "OPENAI_COMPAT_API_KEY": "compat-key",
                "OPENAI_COMPAT_BASE_URL": "https://compat.example/v1",
                "OPENAI_COMPAT_USER_AGENT": "research-client/1.0",
            },
        )

        self.assertEqual(model, "research-model")
        self.assertEqual(
            kwargs["default_headers"], {"User-Agent": "research-client/1.0"}
        )

    def test_official_openai_route_does_not_override_user_agent(self) -> None:
        kwargs, _ = build_openai_compatible_client_kwargs(
            "gpt-4.1",
            env={"OPENAI_API_KEY": "openai-key"},
        )

        self.assertNotIn("default_headers", kwargs)

    def test_live_probe_reports_endpoint_model_identity_without_content(self) -> None:
        class Usage:
            prompt_tokens = 7
            completion_tokens = 1
            total_tokens = 8

        class Choice:
            finish_reason = "stop"

        class Response:
            model = "gpt-5.6-luna"
            choices = [Choice()]
            usage = Usage()

        with mock.patch("openai.OpenAI") as client_type:
            client = client_type.return_value
            client.chat.completions.create.return_value = Response()
            result = probe_openai_compatible_model(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["exact_model_match"])
        self.assertEqual(result["reported_model"], "gpt-5.6-luna")
        self.assertFalse(result["response_content_recorded"])
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(
            request["messages"][0]["content"],
            "Reply with exactly OK and no other text.",
        )

    def test_live_probe_marks_model_substitution_unverified(self) -> None:
        class Response:
            model = "gpt-5.4-mini"
            choices = []
            usage = None

        with mock.patch("openai.OpenAI") as client_type:
            client_type.return_value.chat.completions.create.return_value = Response()
            result = probe_openai_compatible_model(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        self.assertFalse(result["ok"])
        self.assertFalse(result["exact_model_match"])
        self.assertEqual(result["reported_model"], "gpt-5.4-mini")

    def test_huggingface_base_url_tracks_requested_model(self) -> None:
        spec = resolve_model_provider("huggingface/org/custom-model")

        self.assertEqual(spec.client_model, "org/custom-model")
        self.assertEqual(
            spec.default_base_url,
            "https://api-inference.huggingface.co/models/org/custom-model",
        )

    def test_provider_env_statuses_should_surface_vendor_matrix(self) -> None:
        statuses = provider_env_statuses(
            {
                "OPENAI_API_KEY": "openai-key",
                "ZHIPU_API_KEY": "zhipu-key",
                "OPENAI_COMPAT_API_KEY": "compat-key",
                "OPENAI_COMPAT_BASE_URL": "https://compat.example/v1",
            }
        )
        by_provider = {row.provider: row for row in statuses}
        self.assertTrue(by_provider["openai"].configured)
        self.assertTrue(by_provider["zhipu"].configured)
        self.assertTrue(by_provider["openai_compat"].configured)
        self.assertFalse(by_provider["deepseek"].configured)


if __name__ == "__main__":
    unittest.main()
