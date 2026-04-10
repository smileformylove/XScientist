from __future__ import annotations

import unittest

from ai_scientist.utils.provider_registry import (
    build_openai_compatible_client_kwargs,
    missing_model_credentials,
    provider_env_statuses,
    resolve_model_provider,
)


class ProviderRegistryTests(unittest.TestCase):
    def test_resolve_model_provider_should_normalize_prefixed_and_legacy_models(self) -> None:
        openrouter_spec = resolve_model_provider(
            "openrouter/meta-llama/llama-3.1-405b-instruct"
        )
        legacy_spec = resolve_model_provider("llama3.1-405b")
        compat_spec = resolve_model_provider("openai_compat/qwen2.5-72b-instruct")

        self.assertEqual(openrouter_spec.provider, "openrouter")
        self.assertEqual(openrouter_spec.client_model, "meta-llama/llama-3.1-405b-instruct")
        self.assertEqual(legacy_spec.provider, "openrouter")
        self.assertEqual(compat_spec.provider, "openai_compat")
        self.assertEqual(compat_spec.client_model, "qwen2.5-72b-instruct")

    def test_build_openai_compatible_client_kwargs_should_respect_provider_envs(self) -> None:
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

    def test_missing_model_credentials_should_report_openai_compat_base_url(self) -> None:
        missing = missing_model_credentials(
            ["openai_compat/custom-model"],
            env={"OPENAI_COMPAT_API_KEY": "compat-key"},
        )
        self.assertEqual(len(missing), 1)
        self.assertIn("OPENAI_COMPAT_BASE_URL | OPENAI_BASE_URL", missing[0]["missing"])

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
