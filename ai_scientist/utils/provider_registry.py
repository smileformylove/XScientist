from __future__ import annotations

"""Provider and model resolution helpers for multi-vendor LLM access."""

from dataclasses import dataclass
from typing import Iterable, Mapping


OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "ollama",
    "deepseek",
    "huggingface",
    "openrouter",
    "gemini",
    "zhipu",
    "openai_compat",
}

LOCAL_PROVIDER_NAMES = {"ollama", "vertex_ai"}


@dataclass(frozen=True)
class ModelProviderSpec:
    raw_model: str
    provider: str
    display_name: str
    client_family: str
    client_model: str
    request_style: str
    api_key_env_vars: tuple[str, ...] = ()
    required_env_vars: tuple[str, ...] = ()
    optional_env_vars: tuple[str, ...] = ()
    default_base_url: str | None = None
    base_url_env_vars: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderCredentialStatus:
    provider: str
    display_name: str
    configured: bool
    required_envs: tuple[str, ...]
    detail: str
    counts_as_configured_provider: bool = True


_PROVIDER_DISPLAY = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "bedrock": "Amazon Bedrock",
    "vertex_ai": "Vertex AI",
    "ollama": "Ollama",
    "deepseek": "DeepSeek",
    "huggingface": "HuggingFace",
    "openrouter": "OpenRouter",
    "gemini": "Gemini",
    "zhipu": "Zhipu",
    "openai_compat": "OpenAI-Compatible",
}


def _clean_model(model: str) -> str:
    value = str(model or "").strip()
    if not value:
        raise ValueError("Model name cannot be empty")
    return value


def _split_prefixed_model(model: str) -> tuple[str | None, str]:
    if "/" not in model:
        return None, model
    prefix, suffix = model.split("/", 1)
    known_prefixes = {
        "openai",
        "anthropic",
        "bedrock",
        "vertex_ai",
        "ollama",
        "deepseek",
        "huggingface",
        "openrouter",
        "gemini",
        "zhipu",
        "openai_compat",
    }
    if prefix in known_prefixes:
        return prefix, suffix
    return None, model


def _pick_first_env(
    env_names: Iterable[str],
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    source = env or {}
    for env_name in env_names:
        value = str(source.get(env_name) or "").strip()
        if value:
            return env_name, value
    return None, None


def _build_spec(
    raw_model: str,
    provider: str,
    client_family: str,
    client_model: str,
    request_style: str,
    *,
    api_key_env_vars: tuple[str, ...] = (),
    required_env_vars: tuple[str, ...] = (),
    optional_env_vars: tuple[str, ...] = (),
    default_base_url: str | None = None,
    base_url_env_vars: tuple[str, ...] = (),
) -> ModelProviderSpec:
    return ModelProviderSpec(
        raw_model=raw_model,
        provider=provider,
        display_name=_PROVIDER_DISPLAY.get(provider, provider),
        client_family=client_family,
        client_model=client_model,
        request_style=request_style,
        api_key_env_vars=api_key_env_vars,
        required_env_vars=required_env_vars,
        optional_env_vars=optional_env_vars,
        default_base_url=default_base_url,
        base_url_env_vars=base_url_env_vars,
    )


def resolve_model_provider(model: str) -> ModelProviderSpec:
    raw_model = _clean_model(model)
    prefix, suffix = _split_prefixed_model(raw_model)

    if prefix == "anthropic" or raw_model.startswith("claude-"):
        client_model = suffix if prefix == "anthropic" else raw_model
        return _build_spec(
            raw_model,
            "anthropic",
            "anthropic",
            client_model,
            "anthropic_messages",
            api_key_env_vars=("ANTHROPIC_API_KEY",),
        )

    if raw_model.startswith("bedrock/") and "claude" in raw_model:
        return _build_spec(
            raw_model,
            "bedrock",
            "anthropic_bedrock",
            raw_model.split("/", 1)[1],
            "anthropic_messages",
            required_env_vars=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"),
        )

    if raw_model.startswith("vertex_ai/") and "claude" in raw_model:
        return _build_spec(
            raw_model,
            "vertex_ai",
            "anthropic_vertex",
            raw_model.split("/", 1)[1],
            "anthropic_messages",
            optional_env_vars=("GOOGLE_CLOUD_PROJECT", "CLOUD_ML_REGION"),
        )

    if prefix == "ollama" or raw_model.startswith("ollama/"):
        client_model = suffix if prefix == "ollama" else raw_model.split("/", 1)[1]
        return _build_spec(
            raw_model,
            "ollama",
            "openai_compatible",
            client_model,
            "openai_chat",
            optional_env_vars=("OLLAMA_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_HOST"),
            default_base_url="http://localhost:11434/v1",
            base_url_env_vars=("OLLAMA_BASE_URL",),
        )

    if prefix == "openai":
        client_model = suffix
        request_style = "openai_reasoning" if client_model.startswith(("o1", "o3")) else "openai_chat"
        return _build_spec(
            raw_model,
            "openai",
            "openai_compatible",
            client_model,
            request_style,
            api_key_env_vars=("OPENAI_API_KEY",),
        )

    if prefix == "deepseek" or raw_model.startswith("deepseek-") or raw_model in {"deepseek-coder", "deepseek-chat", "deepseek-reasoner"}:
        client_model = suffix if prefix == "deepseek" else raw_model
        if client_model == "deepseek-coder-v2-0724":
            client_model = "deepseek-coder"
        return _build_spec(
            raw_model,
            "deepseek",
            "openai_compatible",
            client_model,
            "openai_chat",
            api_key_env_vars=("DEEPSEEK_API_KEY",),
            default_base_url="https://api.deepseek.com",
        )

    if prefix == "huggingface" or raw_model in {"deepcoder-14b", "agentica-org/DeepCoder-14B-Preview"}:
        client_model = suffix if prefix == "huggingface" else raw_model
        if client_model == "deepcoder-14b":
            client_model = "agentica-org/DeepCoder-14B-Preview"
        return _build_spec(
            raw_model,
            "huggingface",
            "openai_compatible",
            client_model,
            "huggingface_chat",
            api_key_env_vars=("HUGGINGFACE_API_KEY",),
            default_base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
        )

    if prefix == "openrouter" or raw_model in {
        "llama3.1-405b",
        "llama-3-1-405b-instruct",
        "meta-llama/llama-3.1-405b-instruct",
    }:
        client_model = suffix if prefix == "openrouter" else raw_model
        if client_model in {"llama3.1-405b", "llama-3-1-405b-instruct"}:
            client_model = "meta-llama/llama-3.1-405b-instruct"
        return _build_spec(
            raw_model,
            "openrouter",
            "openai_compatible",
            client_model,
            "openai_chat",
            api_key_env_vars=("OPENROUTER_API_KEY",),
            default_base_url="https://openrouter.ai/api/v1",
        )

    if prefix == "gemini" or "gemini" in raw_model:
        client_model = suffix if prefix == "gemini" else raw_model
        return _build_spec(
            raw_model,
            "gemini",
            "openai_compatible",
            client_model,
            "openai_chat",
            api_key_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    if prefix == "zhipu" or raw_model.startswith("glm-"):
        client_model = suffix if prefix == "zhipu" else raw_model
        return _build_spec(
            raw_model,
            "zhipu",
            "openai_compatible",
            client_model,
            "openai_chat",
            api_key_env_vars=("ZHIPU_API_KEY",),
            default_base_url="https://open.bigmodel.cn/api/paas/v4/",
        )

    if prefix == "openai_compat":
        return _build_spec(
            raw_model,
            "openai_compat",
            "openai_compatible",
            suffix,
            "openai_chat",
            api_key_env_vars=("OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY"),
            base_url_env_vars=("OPENAI_COMPAT_BASE_URL", "OPENAI_BASE_URL"),
        )

    if raw_model.startswith(("gpt-", "chatgpt-")) or raw_model.startswith(("o1", "o3")):
        return _build_spec(
            raw_model,
            "openai",
            "openai_compatible",
            raw_model,
            "openai_reasoning" if raw_model.startswith(("o1", "o3")) else "openai_chat",
            api_key_env_vars=("OPENAI_API_KEY",),
        )

    return _build_spec(
        raw_model,
        "openai",
        "openai_compatible",
        raw_model,
        "openai_chat",
        api_key_env_vars=("OPENAI_API_KEY",),
    )


def model_uses_anthropic_client(model: str) -> bool:
    return resolve_model_provider(model).client_family.startswith("anthropic")


def model_uses_openai_chat(model: str) -> bool:
    return resolve_model_provider(model).request_style in {"openai_chat", "huggingface_chat"}


def model_uses_openai_reasoning(model: str) -> bool:
    return resolve_model_provider(model).request_style == "openai_reasoning"


def is_openai_compatible_model(model: str) -> bool:
    return resolve_model_provider(model).provider in OPENAI_COMPATIBLE_PROVIDERS


def _missing_requirements(
    spec: ModelProviderSpec,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    source = env or {}
    missing: list[str] = []
    for env_name in spec.required_env_vars:
        if not str(source.get(env_name) or "").strip():
            missing.append(env_name)
    if spec.api_key_env_vars and _pick_first_env(spec.api_key_env_vars, source)[1] is None:
        missing.append(" | ".join(spec.api_key_env_vars))
    # Some OpenAI-compatible providers (e.g., Ollama) ship a safe default base_url.
    # Only require explicit base_url configuration when no default is available.
    if (
        spec.base_url_env_vars
        and _pick_first_env(spec.base_url_env_vars, source)[1] is None
        and not str(spec.default_base_url or "").strip()
    ):
        missing.append(" | ".join(spec.base_url_env_vars))
    return missing


def describe_model_requirements(
    models: Iterable[str],
    env: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    source = env or {}
    described: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for model in models:
        if not str(model or "").strip():
            continue
        spec = resolve_model_provider(model)
        key = (spec.provider, spec.client_model)
        if key in seen:
            continue
        seen.add(key)
        missing = _missing_requirements(spec, source)
        described.append(
            {
                "model": spec.raw_model,
                "provider": spec.provider,
                "display_name": spec.display_name,
                "client_model": spec.client_model,
                "missing": ", ".join(missing),
            }
        )
    return described


def missing_model_credentials(
    models: Iterable[str],
    env: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    missing_rows: list[dict[str, str]] = []
    for row in describe_model_requirements(models, env=env):
        if row["missing"]:
            missing_rows.append(row)
    return missing_rows


def build_openai_compatible_client_kwargs(
    model: str,
    *,
    env: Mapping[str, str] | None = None,
    max_retries: int | None = None,
) -> tuple[dict[str, object], str]:
    source = env or {}
    spec = resolve_model_provider(model)
    if spec.client_family != "openai_compatible":
        raise ValueError(
            f"Model {model!r} uses {spec.client_family}, not an OpenAI-compatible client"
        )

    kwargs: dict[str, object] = {}
    _, api_key = _pick_first_env(spec.api_key_env_vars, source)
    _, base_url = _pick_first_env(spec.base_url_env_vars, source)
    resolved_base_url = base_url or spec.default_base_url

    if spec.provider == "openai":
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
    else:
        kwargs["api_key"] = api_key or ""
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
    return kwargs, spec.client_model


def provider_env_statuses(
    env: Mapping[str, str] | None = None,
) -> list[ProviderCredentialStatus]:
    source = env or {}
    specs = [
        _build_spec(
            "openai/gpt-4.1",
            "openai",
            "openai_compatible",
            "gpt-4.1",
            "openai_chat",
            api_key_env_vars=("OPENAI_API_KEY",),
        ),
        _build_spec(
            "anthropic/claude-3-5-sonnet-20241022",
            "anthropic",
            "anthropic",
            "claude-3-5-sonnet-20241022",
            "anthropic_messages",
            api_key_env_vars=("ANTHROPIC_API_KEY",),
        ),
        _build_spec(
            "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "bedrock",
            "anthropic_bedrock",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic_messages",
            required_env_vars=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"),
        ),
        _build_spec(
            "vertex_ai/claude-3-5-sonnet@20241022",
            "vertex_ai",
            "anthropic_vertex",
            "claude-3-5-sonnet@20241022",
            "anthropic_messages",
            optional_env_vars=("GOOGLE_CLOUD_PROJECT", "CLOUD_ML_REGION"),
        ),
        _build_spec(
            "ollama/qwen3:8b",
            "ollama",
            "openai_compatible",
            "qwen3:8b",
            "openai_chat",
            optional_env_vars=("OLLAMA_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_HOST"),
            default_base_url="http://localhost:11434/v1",
            base_url_env_vars=("OLLAMA_BASE_URL",),
        ),
        _build_spec(
            "deepseek/deepseek-chat",
            "deepseek",
            "openai_compatible",
            "deepseek-chat",
            "openai_chat",
            api_key_env_vars=("DEEPSEEK_API_KEY",),
            default_base_url="https://api.deepseek.com",
        ),
        _build_spec(
            "huggingface/agentica-org/DeepCoder-14B-Preview",
            "huggingface",
            "openai_compatible",
            "agentica-org/DeepCoder-14B-Preview",
            "huggingface_chat",
            api_key_env_vars=("HUGGINGFACE_API_KEY",),
            default_base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
        ),
        _build_spec(
            "openrouter/meta-llama/llama-3.1-405b-instruct",
            "openrouter",
            "openai_compatible",
            "meta-llama/llama-3.1-405b-instruct",
            "openai_chat",
            api_key_env_vars=("OPENROUTER_API_KEY",),
            default_base_url="https://openrouter.ai/api/v1",
        ),
        _build_spec(
            "gemini/gemini-2.5-pro-preview-03-25",
            "gemini",
            "openai_compatible",
            "gemini-2.5-pro-preview-03-25",
            "openai_chat",
            api_key_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        _build_spec(
            "zhipu/glm-4-plus",
            "zhipu",
            "openai_compatible",
            "glm-4-plus",
            "openai_chat",
            api_key_env_vars=("ZHIPU_API_KEY",),
            default_base_url="https://open.bigmodel.cn/api/paas/v4/",
        ),
        _build_spec(
            "openai_compat/custom-model",
            "openai_compat",
            "openai_compatible",
            "custom-model",
            "openai_chat",
            api_key_env_vars=("OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY"),
            base_url_env_vars=("OPENAI_COMPAT_BASE_URL", "OPENAI_BASE_URL"),
        ),
    ]

    statuses: list[ProviderCredentialStatus] = []
    for spec in specs:
        missing = _missing_requirements(spec, source)
        if spec.provider == "ollama":
            configured = bool(
                str(source.get("OLLAMA_BASE_URL") or "").strip()
                or str(source.get("OLLAMA_HOST") or "").strip()
                or str(source.get("OLLAMA_API_KEY") or "").strip()
            )
            detail = "env optional: OLLAMA_BASE_URL, OLLAMA_API_KEY"
            counts = False
        elif spec.provider == "vertex_ai":
            configured = bool(
                str(source.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
                or str(source.get("GOOGLE_CLOUD_PROJECT") or "").strip()
            )
            detail = "ADC / env optional: GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT"
            counts = configured
        else:
            configured = not missing
            parts = list(spec.required_env_vars)
            if spec.api_key_env_vars:
                parts.append(" | ".join(spec.api_key_env_vars))
            if spec.base_url_env_vars:
                parts.append(" | ".join(spec.base_url_env_vars))
            detail = f"env: {', '.join(parts)}" if parts else "no env required"
            counts = spec.provider not in LOCAL_PROVIDER_NAMES
        statuses.append(
            ProviderCredentialStatus(
                provider=spec.provider,
                display_name=spec.display_name,
                configured=configured,
                required_envs=spec.required_env_vars,
                detail=detail,
                counts_as_configured_provider=counts,
            )
        )
    return statuses
