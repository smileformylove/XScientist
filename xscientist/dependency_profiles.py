from __future__ import annotations

"""Dependency-free provider and research-capability resolution.

The resolver deliberately reports requirements instead of installing them.
This keeps first-run diagnostics deterministic and makes dependency changes an
explicit user action.
"""

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

CORE_MODULES = ("jsonschema", "yaml")

# Modules are import names, not distribution names.  Each capability maps to a
# declared optional dependency extra in ``pyproject.toml``.
CAPABILITY_MODULES: dict[str, tuple[str, ...]] = {
    "research": (
        "backoff",
        "black",
        "coolname",
        "dataclasses_json",
        "genson",
        "huggingface_hub",
        "humanize",
        "igraph",
        "matplotlib",
        "numpy",
        "omegaconf",
        "pandas",
        "PIL",
        "psutil",
        "pymupdf",
        "pypdf",
        "requests",
        "rich",
        "seaborn",
        "shutup",
        "sklearn",
        "tiktoken",
        "tqdm",
    ),
    "plot": ("igraph", "matplotlib", "numpy", "pandas", "PIL", "seaborn"),
    "pdf": ("fitz", "pypdf"),
    "pdf-layout": ("pymupdf4llm",),
    "ml": (
        "datasets",
        "huggingface_hub",
        "numpy",
        "pandas",
        "sklearn",
        "transformers",
    ),
    "service": ("fastapi", "pydantic", "uvicorn"),
}

# A task is a user-facing intent.  Capabilities are composable implementation
# needs.  Provider-neutral tasks can therefore use the small base installation.
TASK_PROFILES: dict[str, dict[str, object]] = {
    "protocol": {
        "description": "Validate protocols and use Research VCS.",
        "capabilities": (),
        "provider_required": False,
        "auth_required": False,
        "runtime_preflight": False,
    },
    "manage": {
        "description": "Inspect existing research outputs and indexes.",
        "capabilities": (),
        "provider_required": False,
        "auth_required": False,
        "runtime_preflight": False,
    },
    "research": {
        "description": "Run a general autonomous research workflow.",
        "capabilities": ("research",),
        "provider_required": True,
        "auth_required": True,
        "runtime_preflight": True,
    },
    "paper": {
        "description": "Run research with plots, PDF reading, and paper output.",
        "capabilities": ("research", "plot", "pdf"),
        "provider_required": True,
        "auth_required": True,
        "runtime_preflight": True,
    },
    "pdf-review": {
        "description": "Review papers with layout-aware PDF extraction.",
        "capabilities": ("research", "pdf", "pdf-layout"),
        "provider_required": True,
        "auth_required": True,
        "runtime_preflight": True,
    },
    "ml-study": {
        "description": "Run an ML study with datasets, models, plots, and PDFs.",
        "capabilities": ("research", "plot", "pdf", "ml"),
        "provider_required": True,
        "auth_required": True,
        "runtime_preflight": True,
    },
    "service": {
        "description": "Expose the local HTTP API without running a study.",
        "capabilities": ("service",),
        "provider_required": False,
        "auth_required": False,
        "runtime_preflight": False,
    },
}


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


def _missing_modules(
    modules: Iterable[str],
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> list[str]:
    missing: list[str] = []
    for module in dict.fromkeys(modules):
        try:
            found = find_spec(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            found = None
        if found is None:
            missing.append(module)
    return missing


def capability_installation_spec(
    capabilities: Iterable[str] = (),
    *,
    provider: str | None = None,
    version: str | None = None,
) -> str:
    extras = list(dict.fromkeys(str(item) for item in capabilities))
    unknown = set(extras) - set(CAPABILITY_EXTRAS)
    if unknown:
        raise ValueError(f"unsupported installation extras: {sorted(unknown)}")
    if provider:
        extras.append(provider_extra(provider))
        extras = list(dict.fromkeys(extras))
    version_suffix = f"=={version}" if str(version or "").strip() else ""
    suffix = f"[{','.join(extras)}]" if extras else ""
    return f"xscientist{suffix}{version_suffix}"


def capability_installation_command(
    capabilities: Iterable[str] = (),
    *,
    provider: str | None = None,
    version: str | None = None,
) -> str:
    return (
        'python -m pip install "'
        + capability_installation_spec(
            capabilities,
            provider=provider,
            version=version,
        )
        + '"'
    )


def resolve_task_capabilities(
    task: str,
    *,
    provider: str | None = None,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> dict[str, object]:
    """Resolve one research intent to exact, locally probed requirements."""

    normalized_task = str(task or "").strip().lower()
    try:
        profile = TASK_PROFILES[normalized_task]
    except KeyError as exc:
        raise ValueError(f"unsupported research task: {task!r}") from exc
    capabilities = tuple(str(item) for item in profile["capabilities"])
    provider_required = bool(profile["provider_required"])
    auth_required = bool(profile["auth_required"])
    runtime_preflight = bool(profile["runtime_preflight"])
    normalized_provider = str(provider or "").strip().lower() or None
    if normalized_provider:
        # Validate before probing so the error is actionable and stable.
        provider_extra(normalized_provider)

    capability_rows: list[dict[str, object]] = []
    all_modules: list[str] = list(CORE_MODULES)
    for capability in capabilities:
        modules = CAPABILITY_MODULES[capability]
        missing = _missing_modules(modules, find_spec=find_spec)
        capability_rows.append(
            {
                "name": capability,
                "extra": capability,
                "modules": list(modules),
                "missing_modules": missing,
                "ready": not missing,
            }
        )
        all_modules.extend(modules)

    missing_core = _missing_modules(CORE_MODULES, find_spec=find_spec)
    missing_provider = (
        missing_provider_modules(normalized_provider, find_spec=find_spec)
        if normalized_provider
        else []
    )
    provider_selected = normalized_provider is not None
    ready = (
        not missing_core
        and all(row["ready"] for row in capability_rows)
        and not missing_provider
        and (provider_selected or not provider_required)
    )
    install = capability_installation_command(
        capabilities,
        provider=normalized_provider,
    )
    return {
        "schema": "xscientist.capability-resolution.v1",
        "task": normalized_task,
        "description": str(profile["description"]),
        "provider_required": provider_required,
        "auth_required": auth_required,
        "runtime_preflight": runtime_preflight,
        "provider": normalized_provider,
        "core_modules": list(CORE_MODULES),
        "missing_core_modules": missing_core,
        "capabilities": capability_rows,
        "required_modules": list(dict.fromkeys(all_modules)),
        "missing_modules": list(
            dict.fromkeys(
                [
                    *missing_core,
                    *(
                        module
                        for row in capability_rows
                        for module in row["missing_modules"]
                    ),
                    *missing_provider,
                ]
            )
        ),
        "provider_modules": (
            list(provider_client_modules(normalized_provider))
            if normalized_provider
            else []
        ),
        "missing_provider_modules": missing_provider,
        "install_command": install,
        "ready": ready,
    }


def installation_spec(
    provider: str,
    *,
    capabilities: Iterable[str] = ("research",),
    version: str | None = None,
) -> str:
    return capability_installation_spec(
        capabilities,
        provider=provider,
        version=version,
    )


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
    "CAPABILITY_MODULES",
    "CAPABILITY_EXTRAS",
    "CORE_MODULES",
    "PROVIDER_CLIENT_MODULES",
    "PROVIDER_EXTRA_BY_NAME",
    "TASK_PROFILES",
    "capability_installation_command",
    "capability_installation_spec",
    "installation_command",
    "installation_spec",
    "missing_provider_modules",
    "provider_client_modules",
    "provider_extra",
    "resolve_task_capabilities",
]
