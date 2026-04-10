from __future__ import annotations

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from ai_scientist.utils.optional_dependencies import (
    BackoffFallback,
    MissingOptionalDependencyProxy,
    resolve_exception_types,
)


class OptionalDependencyImportTests(unittest.TestCase):
    def test_llm_module_should_import_without_optional_sdk_errors(self) -> None:
        module = importlib.import_module("ai_scientist.llm")
        self.assertTrue(callable(module.create_client))

    def test_vlm_module_should_import_without_optional_sdk_errors(self) -> None:
        module = importlib.import_module("ai_scientist.vlm")
        self.assertTrue(callable(module.create_client))

    def test_llm_module_should_reload_when_optional_sdks_are_missing(self) -> None:
        self._assert_reload_survives_missing_modules(
            "ai_scientist.llm",
            missing_names={"anthropic", "openai", "backoff"},
        )

    def test_vlm_module_should_reload_when_optional_sdks_are_missing(self) -> None:
        self._assert_reload_survives_missing_modules(
            "ai_scientist.vlm",
            missing_names={"openai", "backoff", "PIL"},
        )

    def test_semantic_scholar_module_should_reload_when_optional_sdks_are_missing(
        self,
    ) -> None:
        self._assert_reload_survives_missing_modules(
            "ai_scientist.tools.semantic_scholar",
            missing_names={"requests", "backoff"},
            expected_attr="search_for_papers",
        )

    def test_treesearch_backend_package_should_reload_when_optional_sdks_are_missing(
        self,
    ) -> None:
        self._assert_reload_survives_missing_modules(
            "ai_scientist.treesearch.backend",
            missing_names={"backoff"},
            expected_attr="get_ai_client",
        )

    def test_treesearch_backend_openai_module_should_reload_when_sdk_is_missing(
        self,
    ) -> None:
        self._assert_reload_survives_missing_modules(
            "ai_scientist.treesearch.backend.backend_openai",
            missing_names={"openai"},
            expected_attr="get_ai_client",
        )

    def test_treesearch_backend_anthropic_module_should_reload_when_sdk_is_missing(
        self,
    ) -> None:
        self._assert_reload_survives_missing_modules(
            "ai_scientist.treesearch.backend.backend_anthropic",
            missing_names={"anthropic"},
            expected_attr="get_ai_client",
        )

    def test_treesearch_backend_zhipu_module_should_reload_when_sdk_is_missing(
        self,
    ) -> None:
        self._assert_reload_survives_missing_modules(
            "ai_scientist.treesearch.backend.backend_zhipu",
            missing_names={"zhipuai"},
            expected_attr="get_ai_client",
        )

    def test_llm_module_should_reload_when_sdk_exception_attrs_are_missing(
        self,
    ) -> None:
        self._assert_reload_survives_missing_exception_attrs("ai_scientist.llm")

    def test_vlm_module_should_reload_when_sdk_exception_attrs_are_missing(
        self,
    ) -> None:
        self._assert_reload_survives_missing_exception_attrs("ai_scientist.vlm")

    def test_create_client_should_raise_with_install_hint_when_sdk_is_missing(
        self,
    ) -> None:
        module = importlib.import_module("ai_scientist.llm")
        missing_anthropic = MissingOptionalDependencyProxy(
            "anthropic",
            install_hint="Install the 'anthropic' package to use Anthropic-backed models.",
        )

        with mock.patch.object(module, "anthropic", missing_anthropic), mock.patch.object(
            module,
            "resolve_model_provider",
            return_value=SimpleNamespace(
                client_family="anthropic",
                display_name="Anthropic",
                client_model="demo-model",
            ),
        ):
            with self.assertRaisesRegex(
                ModuleNotFoundError,
                "Install the 'anthropic' package",
            ):
                module.create_client("demo-model")

    def test_huggingface_http_fallback_should_raise_with_requests_install_hint(
        self,
    ) -> None:
        module = importlib.import_module("ai_scientist.llm")
        missing_requests = MissingOptionalDependencyProxy(
            "requests",
            install_hint="Install the 'requests' package to use HuggingFace HTTP fallback calls.",
        )

        class FailingClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**_kwargs):
                        raise RuntimeError("force http fallback")

        with mock.patch.object(module, "requests", missing_requests), mock.patch.object(
            module,
            "resolve_model_provider",
            return_value=SimpleNamespace(
                provider="huggingface",
                request_style="openai_chat",
                client_model="demo-model",
            ),
        ), mock.patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "demo-key"}, clear=False):
            with self.assertRaisesRegex(
                ModuleNotFoundError,
                "Install the 'requests' package",
            ):
                module.get_response_from_llm(
                    "hello",
                    FailingClient(),
                    "demo-model",
                    "system",
                )

    def test_resolve_exception_types_should_tolerate_missing_exception_attrs(
        self,
    ) -> None:
        module_like = SimpleNamespace(RateLimitError=RuntimeError)

        resolved = resolve_exception_types(
            module_like,
            ("RateLimitError", "APITimeoutError"),
        )

        self.assertEqual(len(resolved), 2)
        self.assertIs(resolved[0], RuntimeError)
        self.assertTrue(issubclass(resolved[1], Exception))

    def test_semantic_scholar_search_should_raise_with_install_hint_when_requests_missing(
        self,
    ) -> None:
        module = importlib.import_module("ai_scientist.tools.semantic_scholar")
        missing_requests = MissingOptionalDependencyProxy(
            "requests",
            install_hint="Install the 'requests' package to use Semantic Scholar search.",
        )
        with mock.patch.object(module, "requests", missing_requests):
            with self.assertRaisesRegex(
                ModuleNotFoundError,
                "Install the 'requests' package",
            ):
                module.search_for_papers("test query")

    def test_treesearch_zhipu_backend_should_raise_with_install_hint_when_sdk_missing(
        self,
    ) -> None:
        module = importlib.import_module("ai_scientist.treesearch.backend.backend_zhipu")
        missing_zhipuai = MissingOptionalDependencyProxy(
            "zhipuai",
            install_hint="Install the 'zhipuai' package to use the treesearch Zhipu backend.",
        )
        with mock.patch.object(module, "zhipuai_sdk", missing_zhipuai):
            with self.assertRaisesRegex(
                ModuleNotFoundError,
                "Install the 'zhipuai' package",
            ):
                module.get_ai_client("glm-4-plus")

    def _assert_reload_survives_missing_modules(
        self,
        module_name: str,
        *,
        missing_names: set[str],
        expected_attr: str = "create_client",
    ) -> None:
        module = importlib.import_module(module_name)
        optional_dependencies = importlib.import_module(
            "ai_scientist.utils.optional_dependencies"
        )
        real_import_module = importlib.import_module

        def fake_import_module(name: str, package: str | None = None):
            if name == "PIL.Image" and "PIL" in missing_names:
                error = ModuleNotFoundError("No module named 'PIL'")
                error.name = "PIL"
                raise error
            if name in missing_names:
                error = ModuleNotFoundError(f"No module named '{name}'")
                error.name = name
                raise error
            return real_import_module(name, package)

        with mock.patch.object(
            optional_dependencies.importlib,
            "import_module",
            side_effect=fake_import_module,
        ):
            reloaded = importlib.reload(module)
            self.assertTrue(callable(getattr(reloaded, expected_attr)))

        importlib.reload(module)

    def _assert_reload_survives_missing_exception_attrs(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        optional_dependencies = importlib.import_module(
            "ai_scientist.utils.optional_dependencies"
        )
        real_import_module = importlib.import_module

        def fake_import_module(name: str, package: str | None = None):
            if name == "backoff":
                return BackoffFallback()
            if name == "openai":
                return SimpleNamespace(OpenAI=object)
            if name == "anthropic":
                return SimpleNamespace(
                    Anthropic=object,
                    AnthropicBedrock=object,
                    AnthropicVertex=object,
                )
            return real_import_module(name, package)

        with mock.patch.object(
            optional_dependencies.importlib,
            "import_module",
            side_effect=fake_import_module,
        ):
            reloaded = importlib.reload(module)
            self.assertTrue(callable(reloaded.create_client))

        importlib.reload(module)


if __name__ == "__main__":
    unittest.main()
