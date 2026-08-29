from __future__ import annotations

"""Provider and model resolution helpers for multi-vendor LLM access."""

import hashlib
import json
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Iterable, Mapping

from ai_scientist.utils.privacy import redact_sensitive_text

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
        "custom",
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


_MODEL_ROUTE_PREFIXES = frozenset(
    {
        "openai",
        "openai_compat",
        "custom",
        "anthropic",
        "bedrock",
        "vertex_ai",
        "ollama",
        "deepseek",
        "huggingface",
        "openrouter",
        "gemini",
        "zhipu",
    }
)


def _model_identity_variants(value: str) -> set[str]:
    """Return conservative aliases for comparing gateway model identities.

    Gateways frequently echo a route prefix (``openai_compat/foo``) or a
    deployment alias while still serving the requested model.  We only strip
    prefixes that XScientist itself understands; broad substring matching
    would turn a real model substitution into a false pass.
    """

    normalized = str(value or "").strip().lower()
    if not normalized:
        return set()
    variants = {normalized}
    prefix, separator, suffix = normalized.partition("/")
    if separator and prefix in _MODEL_ROUTE_PREFIXES and suffix:
        variants.add(suffix)
    return variants


def model_identity_status(requested_model: str, reported_model: str | None) -> str:
    """Classify a provider-reported model without collapsing aliases and drift.

    Returns one of ``exact``, ``alias``, ``mismatch`` or ``unavailable``.  The
    strict ``ok`` result of a live probe remains tied to ``exact`` so callers
    that require an immutable deployment identity still fail closed, while the
    richer status lets users distinguish a harmless route prefix from a real
    substitution.
    """

    reported = str(reported_model or "").strip()
    requested = str(requested_model or "").strip()
    if not reported:
        return "unavailable"
    if reported == requested:
        return "exact"
    if _model_identity_variants(requested) & _model_identity_variants(reported):
        return "alias"
    return "mismatch"


def safe_provider_metadata_text(
    value: object,
    *,
    allowed: frozenset[str] | None = None,
) -> str | None:
    """Return a bounded provider metadata string without calling ``str``."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or redact_sensitive_text(value) != value
        or (allowed is not None and value not in allowed)
    ):
        return None
    return value


def _fingerprint(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def model_provenance(
    model: str,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a secret-free, hash-stable provider contract for audit records.

    API keys are represented only by the environment variable name that won
    resolution.  Custom endpoints are represented by a content hash rather
    than a URL, which preserves cross-run comparability without leaking private
    hostnames into research artifacts.
    """

    source = os.environ if env is None else env
    spec = resolve_model_provider(model)
    base_name, base_url = _pick_first_env(spec.base_url_env_vars, source)
    endpoint = str(base_url or spec.default_base_url or "").strip().rstrip("/")
    key_name, _key_value = _pick_first_env(spec.api_key_env_vars, source)
    core = {
        "provider": spec.provider,
        "requested_model": spec.raw_model,
        "client_model": spec.client_model,
        "request_style": spec.request_style,
        "endpoint_fingerprint": _fingerprint(endpoint),
    }
    return {
        **core,
        "endpoint_configured": bool(endpoint),
        "endpoint_env": base_name,
        "api_key_env": key_name,
        "configuration_fingerprint": _fingerprint(
            json.dumps(core, sort_keys=True, separators=(",", ":"))
        ),
    }


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
    if prefix is not None and (
        not suffix
        or suffix != suffix.strip()
        or len(suffix) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in suffix)
        or redact_sensitive_text(suffix) != suffix
    ):
        raise ValueError("Route-qualified model name is invalid")

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
            required_env_vars=(
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_REGION_NAME",
            ),
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
        request_style = (
            "openai_reasoning"
            if client_model.startswith(("o1", "o3"))
            else "openai_chat"
        )
        return _build_spec(
            raw_model,
            "openai",
            "openai_compatible",
            client_model,
            request_style,
            api_key_env_vars=("OPENAI_API_KEY",),
        )

    if (
        prefix == "deepseek"
        or raw_model.startswith("deepseek-")
        or raw_model in {"deepseek-coder", "deepseek-chat", "deepseek-reasoner"}
    ):
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

    if prefix == "huggingface" or raw_model in {
        "deepcoder-14b",
        "agentica-org/DeepCoder-14B-Preview",
    }:
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
            default_base_url=(
                "https://api-inference.huggingface.co/models/" + client_model
            ),
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

    if prefix in {"openai_compat", "custom"}:
        return _build_spec(
            raw_model,
            "openai_compat",
            "openai_compatible",
            suffix,
            "openai_chat",
            api_key_env_vars=("OPENAI_COMPAT_API_KEY",),
            base_url_env_vars=("OPENAI_COMPAT_BASE_URL",),
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
    return resolve_model_provider(model).request_style in {
        "openai_chat",
        "huggingface_chat",
    }


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
    if (
        spec.api_key_env_vars
        and _pick_first_env(spec.api_key_env_vars, source)[1] is None
    ):
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
    source = os.environ if env is None else env
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
        if spec.provider == "openai_compat":
            if not api_key:
                raise ValueError(
                    "OPENAI_COMPAT_API_KEY is required for openai_compat models"
                )
            if not resolved_base_url:
                raise ValueError(
                    "OPENAI_COMPAT_BASE_URL is required for openai_compat models"
                )
            if any(ord(char) < 32 or ord(char) == 127 for char in api_key):
                raise ValueError("OPENAI_COMPAT_API_KEY contains control characters")
            if api_key.strip().lower() in {
                "changeme",
                "replace-me",
                "your-api-key",
                "your_api_key_here",
            }:
                raise ValueError("OPENAI_COMPAT_API_KEY is a placeholder")
            parsed = urllib.parse.urlsplit(str(resolved_base_url).strip())
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("OPENAI_COMPAT_BASE_URL is invalid")
            try:
                parsed.port
            except ValueError:
                raise ValueError("OPENAI_COMPAT_BASE_URL has an invalid port") from None
            if parsed.scheme != "https" and parsed.hostname not in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                raise ValueError(
                    "OPENAI_COMPAT_BASE_URL requires HTTPS for remote hosts"
                )
            resolved_base_url = str(resolved_base_url).strip().rstrip("/")
        kwargs["api_key"] = api_key or ""
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url
        if spec.provider == "openai_compat":
            # A few OpenAI-compatible gateways reject the SDK's default
            # ``OpenAI/Python`` user agent even when the request body is valid.
            # Keep the workaround scoped to the explicit compatibility route
            # and allow deployments to choose their own neutral identifier.
            user_agent = str(
                source.get("OPENAI_COMPAT_USER_AGENT") or "xscientist-openai-compatible"
            ).strip()
            if (
                not user_agent
                or len(user_agent) > 128
                or any(ord(char) < 32 or ord(char) == 127 for char in user_agent)
            ):
                raise ValueError("OPENAI_COMPAT_USER_AGENT is invalid")
            kwargs["default_headers"] = {"User-Agent": user_agent}
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
    return kwargs, spec.client_model


def probe_openai_compatible_model(
    model: str,
    *,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Make one explicit minimal call and verify the provider-reported model.

    The response content is intentionally discarded. The probe exposes only
    model identity, latency, finish status, and aggregate token counts so a
    user can detect endpoint-side model substitution without recording prompt
    or completion text.
    """

    source = dict(os.environ if env is None else env)
    spec = resolve_model_provider(model)
    if spec.client_family != "openai_compatible":
        raise ValueError(
            f"live model probing currently supports OpenAI-compatible providers, "
            f"not {spec.provider!r}"
        )
    missing = _missing_requirements(spec, source)
    if missing:
        raise ValueError("missing provider configuration: " + ", ".join(missing))

    provenance = model_provenance(model, env=source)

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ValueError(
            "the OpenAI-compatible client is not installed; install the selected "
            "provider extra first"
        ) from exc

    kwargs, client_model = build_openai_compatible_client_kwargs(
        model,
        env=source,
        max_retries=0,
    )
    if spec.provider == "openai":
        _key_name, api_key = _pick_first_env(spec.api_key_env_vars, source)
        if api_key:
            kwargs["api_key"] = api_key
    started = time.monotonic()
    client = None
    try:
        kwargs["timeout"] = float(timeout)
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=client_model,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly OK and no other text.",
                }
            ],
            max_tokens=8,
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or not 100 <= status_code <= 599
        ):
            status_code = None
        error_type = safe_provider_metadata_text(type(exc).__name__)
        return {
            "ok": False,
            "supported": True,
            "transport_ok": False,
            "provider": spec.provider,
            "requested_model": model,
            "client_model": client_model,
            "reported_model": None,
            "model_identity_verified": False,
            "exact_model_match": False,
            "identity_status": "unavailable",
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "error_code": "live_request_failed",
            "error_type": error_type or "ProviderError",
            "http_status": status_code,
            "response_content_recorded": False,
            "provenance": provenance,
        }
    finally:
        close = getattr(client, "close", None) if client is not None else None
        if callable(close):
            close()

    reported_model = safe_provider_metadata_text(getattr(response, "model", None))
    choices_value = getattr(response, "choices", None)
    choices = choices_value if isinstance(choices_value, list) else []
    finish_reason = (
        safe_provider_metadata_text(
            getattr(choices[0], "finish_reason", None),
            allowed=frozenset({"stop"}),
        )
        if len(choices) == 1
        else None
    )
    usage = getattr(response, "usage", None)
    usage_values = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    usage_valid = all(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0
        for value in usage_values.values()
    )
    envelope_valid = bool(
        reported_model is not None and finish_reason == "stop" and usage_valid
    )
    identity_status = model_identity_status(client_model, reported_model)
    identity_verified = identity_status != "unavailable"
    exact_match = identity_status == "exact"
    return {
        "ok": bool(exact_match and envelope_valid),
        "supported": True,
        "transport_ok": True,
        "provider": spec.provider,
        "requested_model": model,
        "client_model": client_model,
        "reported_model": reported_model,
        "model_identity_verified": identity_verified,
        "exact_model_match": exact_match,
        "identity_status": identity_status,
        "finish_reason": finish_reason,
        "response_envelope_valid": envelope_valid,
        "error_code": None if envelope_valid else "provider_metadata_invalid",
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "usage": {
            key: value if usage_valid else None
            for key, value in usage_values.items()
        },
        "response_content_recorded": False,
        "provenance": provenance,
    }


def probe_live_model(
    model: str,
    *,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run the safest available live probe for a resolved provider.

    OpenAI-compatible routes have a fully implemented, content-free probe.
    Other provider families are returned as a structured unsupported result so
    the CLI can explain the limitation instead of leaking an implementation
    ``ValueError``.  This is deliberately conservative: an unsupported probe
    must never be reported as a successful verification.
    """

    spec = resolve_model_provider(model)
    if spec.client_family == "openai_compatible":
        return probe_openai_compatible_model(model, timeout=timeout, env=env)
    provenance = model_provenance(model, env=env)
    return {
        "ok": False,
        "supported": False,
        "transport_ok": None,
        "provider": spec.provider,
        "requested_model": model,
        "client_model": spec.client_model,
        "reported_model": None,
        "model_identity_verified": False,
        "exact_model_match": False,
        "identity_status": "unavailable",
        "error_code": "live_probe_not_supported",
        "error_type": None,
        "error_message": (
            f"live probing is not implemented for provider family "
            f"{spec.client_family!r}"
        ),
        "response_content_recorded": False,
        "provenance": provenance,
    }


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
            required_env_vars=(
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_REGION_NAME",
            ),
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
            api_key_env_vars=("OPENAI_COMPAT_API_KEY",),
            base_url_env_vars=("OPENAI_COMPAT_BASE_URL",),
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
