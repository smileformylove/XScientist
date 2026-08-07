from __future__ import annotations

import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from xscientist.dependency_profiles import (
    PROVIDER_EXTRA_BY_NAME,
    capability_installation_spec,
    installation_spec,
    resolve_task_capabilities,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DependencyProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
            cls.extras = tomllib.load(handle)["project"]["optional-dependencies"]

    def test_full_profile_keeps_the_legacy_direct_dependency_set(self) -> None:
        self.assertEqual(
            set(self.extras["full"]),
            {
                "anthropic>=0.25",
                "backoff>=2.0",
                "black>=24.0",
                "openai>=1.40",
                "zhipuai>=2.0",
                "matplotlib>=3.8",
                "pillow>=10",
                "pandas>=2",
                "pypdf>=4",
                "pymupdf>=1.24",
                "pymupdf4llm>=0.0.20",
                "seaborn>=0.13",
                "numpy>=1.26",
                "psutil>=5.9",
                "requests>=2.31",
                "scikit-learn>=1.4",
                "transformers>=4.40",
                "datasets>=2.19",
                "tiktoken>=0.7",
                "tqdm>=4.66",
                "rich>=13",
                "humanize>=4",
                "dataclasses-json>=0.6",
                "genson>=1.2",
                "shutup>=0.2",
                "python-igraph>=0.11",
                "coolname>=2",
                "omegaconf>=2.3",
                "boto3>=1.34",
                "huggingface-hub>=0.23",
            },
        )

    def test_research_profile_does_not_install_provider_or_heavy_ml_clients(
        self,
    ) -> None:
        research = set(self.extras["research"])
        for requirement in (
            "openai>=1.40",
            "anthropic>=0.25",
            "zhipuai>=2.0",
            "boto3>=1.34",
            "transformers>=4.40",
            "datasets>=2.19",
            "pymupdf4llm>=0.0.20",
        ):
            self.assertNotIn(requirement, research)

    def test_provider_profiles_resolve_to_declared_extras(self) -> None:
        for provider, extra in PROVIDER_EXTRA_BY_NAME.items():
            with self.subTest(provider=provider):
                self.assertIn(extra, self.extras)
                self.assertEqual(
                    installation_spec(provider),
                    f"xscientist[research,{extra}]",
                )

    def test_task_resolver_keeps_provider_neutral_workflows_small(self) -> None:
        available = {"jsonschema", "yaml"}
        resolved = resolve_task_capabilities(
            "protocol",
            find_spec=lambda name: object() if name in available else None,
        )

        self.assertTrue(resolved["ready"])
        self.assertFalse(resolved["provider_required"])
        self.assertFalse(resolved["auth_required"])
        self.assertEqual(resolved["capabilities"], [])
        self.assertEqual(
            resolved["install_command"], 'python -m pip install "xscientist"'
        )

    def test_task_resolver_combines_capability_and_provider_extras(self) -> None:
        resolved = resolve_task_capabilities(
            "ml-study",
            provider="openai",
            find_spec=lambda _name: None,
        )

        self.assertFalse(resolved["ready"])
        self.assertEqual(
            capability_installation_spec(
                ("research", "plot", "pdf", "ml"), provider="openai"
            ),
            "xscientist[research,plot,pdf,ml,openai]",
        )
        self.assertEqual(
            resolved["install_command"],
            'python -m pip install "xscientist[research,plot,pdf,ml,openai]"',
        )
        self.assertIn("transformers", resolved["missing_modules"])
        self.assertIn("openai", resolved["missing_provider_modules"])


if __name__ == "__main__":
    unittest.main()
