from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from ai_scientist.utils.provider_registry import (
    build_openai_compatible_client_kwargs,
    missing_model_credentials,
    model_identity_status,
    model_provenance,
    probe_live_model,
    probe_openai_compatible_model,
    probe_openai_compatible_tool_call,
    provider_env_statuses,
    resolve_model_provider,
)


def _tool_probe_response(
    *,
    model: str = "gpt-5.6-luna",
    finish_reason: object = "tool_calls",
    function_name: object = "xscientist_capability_check",
    arguments: object = '{"status":"ok"}',
    prompt_tokens: object = 11,
    completion_tokens: object = 5,
    total_tokens: object = 16,
    include_tool_call: bool = True,
) -> SimpleNamespace:
    tool_calls = []
    if include_tool_call:
        tool_calls.append(
            SimpleNamespace(
                type="function",
                function=SimpleNamespace(
                    name=function_name,
                    arguments=arguments,
                ),
            )
        )
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


class ProviderRegistryTests(unittest.TestCase):
    def test_model_identity_status_distinguishes_route_alias_from_substitution(
        self,
    ) -> None:
        self.assertEqual(
            model_identity_status("gpt-5.6-luna", "openai_compat/gpt-5.6-luna"),
            "alias",
        )
        self.assertEqual(model_identity_status("gpt-5.6-luna", "gpt-5.4"), "mismatch")
        self.assertEqual(model_identity_status("gpt-5.6-luna", None), "unavailable")

    def test_custom_route_rejects_empty_or_sensitive_model_suffix(self) -> None:
        for model in (
            "openai_compat/",
            "openai_compat/   ",
            "custom/",
            "openai_compat/" + "sk-" + "R" * 40,
        ):
            with self.subTest(model_type=len(model)):
                with self.assertRaisesRegex(ValueError, "model name is invalid"):
                    resolve_model_provider(model)

    def test_model_provenance_is_secret_free_and_endpoint_stable(self) -> None:
        provenance = model_provenance(
            "openai_compat/gpt-5.6-luna",
            env={
                "OPENAI_COMPAT_API_KEY": "example-key",
                "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1/",
            },
        )
        rendered = str(provenance)
        self.assertNotIn("example-key", rendered)
        self.assertEqual(provenance["endpoint_env"], "OPENAI_COMPAT_BASE_URL")
        self.assertTrue(str(provenance["endpoint_fingerprint"]).startswith("sha256:"))
        self.assertTrue(
            str(provenance["configuration_fingerprint"]).startswith("sha256:")
        )

    def test_live_probe_returns_structured_unsupported_for_anthropic(self) -> None:
        result = probe_live_model(
            "anthropic/claude-3-5-sonnet-20241022",
            env={"ANTHROPIC_API_KEY": "test-key"},
        )
        self.assertFalse(result["ok"])
        self.assertFalse(result["supported"])
        self.assertEqual(result["error_code"], "live_probe_not_supported")
        self.assertFalse(result["response_content_recorded"])

    def test_public_live_probe_keeps_the_text_completion_contract(self) -> None:
        expected = {"ok": True, "capability": "text_completion"}
        environment = {
            "OPENAI_COMPAT_API_KEY": "test-key",
            "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
        }
        with mock.patch(
            "ai_scientist.utils.provider_registry.probe_openai_compatible_model",
            return_value=expected,
        ) as text_probe:
            result = probe_live_model(
                "openai_compat/gpt-5.6-luna",
                timeout=4.0,
                env=environment,
            )

        self.assertIs(result, expected)
        text_probe.assert_called_once_with(
            "openai_compat/gpt-5.6-luna",
            timeout=4.0,
            env=environment,
        )

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
        self.assertIn("OPENAI_COMPAT_BASE_URL", missing[0]["missing"])

    def test_openai_compat_requires_dedicated_key_and_base_url(self) -> None:
        invalid_environments = (
            {
                "OPENAI_API_KEY": "generic-key",
                "OPENAI_BASE_URL": "https://generic.example/v1",
            },
            {"OPENAI_COMPAT_API_KEY": "compat-key"},
            {"OPENAI_COMPAT_BASE_URL": "https://compat.example/v1"},
        )
        for environment in invalid_environments:
            with self.subTest(environment=sorted(environment)):
                with self.assertRaises(ValueError):
                    build_openai_compatible_client_kwargs(
                        "openai_compat/research-model", env=environment
                    )

    def test_openai_compat_rejects_unsafe_endpoint_and_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            build_openai_compatible_client_kwargs(
                "openai_compat/research-model",
                env={
                    "OPENAI_COMPAT_API_KEY": "compat-key",
                    "OPENAI_COMPAT_BASE_URL": "http://remote.example/v1",
                },
            )
        with self.assertRaisesRegex(ValueError, "USER_AGENT"):
            build_openai_compatible_client_kwargs(
                "openai_compat/research-model",
                env={
                    "OPENAI_COMPAT_API_KEY": "compat-key",
                    "OPENAI_COMPAT_BASE_URL": "https://remote.example/v1",
                    "OPENAI_COMPAT_USER_AGENT": "unsafe\r\nheader",
                },
            )

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
        self.assertEqual(result["identity_status"], "mismatch")

    def test_live_probe_never_publishes_untrusted_provider_metadata(self) -> None:
        canary = "sk-" + "P" * 40

        class SensitiveValue:
            def __str__(self) -> str:
                return canary

            __repr__ = __str__

        class Usage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class Choice:
            finish_reason = SensitiveValue()

        class Response:
            model = canary
            choices = [Choice()]
            usage = Usage()

        with mock.patch("openai.OpenAI") as client_type:
            client_type.return_value.chat.completions.create.return_value = Response()
            result = probe_openai_compatible_model(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(canary, rendered)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["reported_model"])
        self.assertEqual(result["error_code"], "provider_metadata_invalid")

    def test_tool_probe_forces_and_validates_one_function_call(self) -> None:
        with mock.patch("openai.OpenAI") as client_type:
            client = client_type.return_value
            client.chat.completions.create.return_value = _tool_probe_response()
            result = probe_openai_compatible_tool_call(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["tool_call_valid"])
        self.assertTrue(result["usage_valid"])
        self.assertEqual(result["identity_status"], "exact")
        self.assertFalse(result["response_content_recorded"])
        self.assertFalse(result["request_content_recorded"])
        self.assertFalse(result["tool_arguments_recorded"])
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(
            request["tool_choice"],
            {
                "type": "function",
                "function": {"name": "xscientist_capability_check"},
            },
        )
        self.assertEqual(
            request["tools"][0]["function"]["parameters"]["required"],
            ["status"],
        )

    def test_tool_probe_fails_when_provider_returns_text_instead_of_tool_call(
        self,
    ) -> None:
        response = _tool_probe_response(
            finish_reason="stop",
            include_tool_call=False,
        )
        with mock.patch("openai.OpenAI") as client_type:
            client_type.return_value.chat.completions.create.return_value = response
            result = probe_openai_compatible_tool_call(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        self.assertFalse(result["ok"])
        self.assertFalse(result["tool_call_valid"])
        self.assertEqual(result["error_code"], "tool_call_contract_failed")

    def test_tool_probe_rejects_inconsistent_usage(self) -> None:
        response = _tool_probe_response(total_tokens=99)
        with mock.patch("openai.OpenAI") as client_type:
            client_type.return_value.chat.completions.create.return_value = response
            result = probe_openai_compatible_tool_call(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["tool_call_valid"])
        self.assertFalse(result["usage_valid"])
        self.assertEqual(result["error_code"], "provider_usage_invalid")
        self.assertEqual(
            result["usage"],
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        )

    def test_tool_probe_rejects_endpoint_model_substitution(self) -> None:
        response = _tool_probe_response(model="different-model")
        with mock.patch("openai.OpenAI") as client_type:
            client_type.return_value.chat.completions.create.return_value = response
            result = probe_openai_compatible_tool_call(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["tool_call_valid"])
        self.assertEqual(result["reported_model"], "different-model")
        self.assertEqual(result["identity_status"], "mismatch")
        self.assertEqual(result["error_code"], "model_identity_mismatch")

    def test_tool_probe_never_publishes_sensitive_provider_metadata(self) -> None:
        canary = "sk-" + "S" * 40
        response = _tool_probe_response(
            model=canary,
            finish_reason=canary,
            function_name=canary,
            arguments=json.dumps({"status": "ok", "secret": canary}),
        )
        with mock.patch("openai.OpenAI") as client_type:
            client_type.return_value.chat.completions.create.return_value = response
            result = probe_openai_compatible_tool_call(
                "openai_compat/gpt-5.6-luna",
                env={
                    "OPENAI_COMPAT_API_KEY": "test-key",
                    "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
                },
            )

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(canary, rendered)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["reported_model"])
        self.assertIsNone(result["finish_reason"])
        self.assertFalse(result["tool_arguments_recorded"])

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

    def test_provider_status_does_not_route_generic_openai_credentials_to_custom(
        self,
    ) -> None:
        statuses = provider_env_statuses(
            {
                "OPENAI_API_KEY": "generic-key",
                "OPENAI_BASE_URL": "https://generic.example/v1",
            }
        )
        by_provider = {row.provider: row for row in statuses}
        self.assertFalse(by_provider["openai_compat"].configured)
        self.assertIn("OPENAI_COMPAT_API_KEY", by_provider["openai_compat"].detail)
        self.assertIn("OPENAI_COMPAT_BASE_URL", by_provider["openai_compat"].detail)


if __name__ == "__main__":
    unittest.main()
