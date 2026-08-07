from __future__ import annotations

"""Small, dependency-free helpers for optional installation profiles."""

import importlib.util
from collections.abc import Callable, Iterable

PROVIDER_EXTRA_BY_NAME: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "zhipu": "zhipu",
    "deepseek": "openai-compatible",
    "gemini": "openai-compatible",
    "openrouter": "openai-compatible",
    "huggingface": "openai-compatible",
    "ollama": "openai-compatible",
    "openai_compat": "openai-compatible",
    "bedrock": "bedrock",
    "vertex_ai": "vertex",
}

PROVIDER_CLIENT_MODULES: dict[str, tuple[str, ...]] = {
    "openai": ("openai",),
    "anthropic": ("anthropic",),
    # The end-to-end route is OpenAI-compatible while the low-level Zhipu
    # tree-search launcher uses zhipuai directly.
    "zhipu": ("openai", "zhipuai"),
    "deepseek": ("openai",),
    "gemini": ("openai",),
    "openrouter": ("openai",),
    "huggingface": ("openai",),
    "ollama": ("openai",),
    "openai_compat": ("openai",),
    "bedrock": ("anthropic", "boto3"),
    "vertex_ai": ("anthropic", "google.auth"),
}

CAPABILITY_EXTRAS = ("research", "plot", "pdf", "pdf-layout", "ml", "service")


def provider_extra(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    try:
        return PROVIDER_EXTRA_BY_NAME[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported provider dependency profile: {provider!r}"
        ) from exc


def provider_client_modules(provider: str) -> tuple[str, ...]:
    normalized = str(provider or "").strip().lower()
    try:
        return PROVIDER_CLIENT_MODULES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported provider dependency profile: {provider!r}"
        ) from exc


def missing_provider_modules(
    provider: str,
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> list[str]:
    missing: list[str] = []
    for module in provider_client_modules(provider):
        try:
            found = find_spec(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            found = None
        if found is None:
            missing.append(module)
    return missing


def installation_spec(
    provider: str,
    *,
    capabilities: Iterable[str] = ("research",),
    version: str | None = None,
) -> str:
    extras = list(dict.fromkeys([*capabilities, provider_extra(provider)]))
    unknown = (
        set(extras) - set(CAPABILITY_EXTRAS) - set(PROVIDER_EXTRA_BY_NAME.values())
    )
    if unknown:
        raise ValueError(f"unsupported installation extras: {sorted(unknown)}")
    version_suffix = f"=={version}" if str(version or "").strip() else ""
    return f"xscientist[{','.join(extras)}]{version_suffix}"


def installation_command(
    provider: str,
    *,
    capabilities: Iterable[str] = ("research",),
    version: str | None = None,
) -> str:
    return (
        'python -m pip install "'
        + installation_spec(provider, capabilities=capabilities, version=version)
        + '"'
    )


__all__ = [
    "CAPABILITY_EXTRAS",
    "PROVIDER_CLIENT_MODULES",
    "PROVIDER_EXTRA_BY_NAME",
    "installation_command",
    "installation_spec",
    "missing_provider_modules",
    "provider_client_modules",
    "provider_extra",
]
