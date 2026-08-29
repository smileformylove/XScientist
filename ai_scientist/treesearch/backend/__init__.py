from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from functools import lru_cache
from typing import Any, Mapping

from .utils import (
    FunctionCallValidationError,
    FunctionSpec,
    OutputType,
    PromptType,
    ResearchDecisionError,
    compile_prompt_to_md,
)
from ai_scientist.utils.provider_registry import resolve_model_provider
from ai_scientist.utils.provider_registry import model_provenance
from ai_scientist.utils.privacy import redact_sensitive_payload, redact_sensitive_text
from ai_scientist.protocol.llm_trace import record_llm_call

OPENAI_COMPAT_CALL_TIMEOUT_SECONDS = 30.0


@lru_cache(maxsize=None)
def _load_backend_module(module_name: str):
    return importlib.import_module(f"{__name__}.{module_name}")


def _resolve_backend_module(model: str):
    spec = resolve_model_provider(model)
    if spec.provider == "zhipu":
        return _load_backend_module("backend_zhipu")
    if spec.client_family.startswith("anthropic"):
        return _load_backend_module("backend_anthropic")
    return _load_backend_module("backend_openai")


def get_ai_client(model: str, **model_kwargs):
    """
    Get the appropriate AI client based on the model string.

    Args:
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        **model_kwargs: Additional keyword arguments for model configuration.
    Returns:
        An instance of the appropriate AI client.
    """
    backend_module = _resolve_backend_module(model)
    return backend_module.get_ai_client(model=model, **model_kwargs)


def _safe_reported_model(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 128
        or normalized != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or redact_sensitive_text(value) != value
    ):
        return None
    return normalized


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _trace_messages(
    system_message: PromptType | None,
    user_message: PromptType | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})
    if user_message is not None:
        messages.append({"role": "user", "content": user_message})
    return messages


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    if func_spec is not None and not isinstance(func_spec, FunctionSpec):
        raise TypeError("func_spec must be a FunctionSpec")

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    spec = resolve_model_provider(model)
    # Snapshot the route contract before the provider call so a concurrent
    # environment mutation cannot rewrite the receipt after the fact.
    provider_env_snapshot: dict[str, str] | None = None
    if spec.provider == "openai_compat":
        provider_env_snapshot = {
            name: os.environ.get(name, "")
            for name in (
                *spec.api_key_env_vars,
                *spec.base_url_env_vars,
                "OPENAI_COMPAT_USER_AGENT",
            )
        }
    provenance = model_provenance(model, env=provider_env_snapshot)
    if spec.provider == "openai_compat":
        requested_timeout = model_kwargs.get("timeout")
        if requested_timeout is None:
            model_kwargs["timeout"] = OPENAI_COMPAT_CALL_TIMEOUT_SECONDS
        elif (
            isinstance(requested_timeout, bool)
            or not isinstance(requested_timeout, (int, float))
            or not math.isfinite(float(requested_timeout))
            or float(requested_timeout) <= 0
        ):
            raise ValueError("openai_compat timeout must be a positive finite number")
        else:
            model_kwargs["timeout"] = min(
                float(requested_timeout),
                OPENAI_COMPAT_CALL_TIMEOUT_SECONDS,
            )
    if spec.request_style == "openai_reasoning":
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = max_tokens or 8192
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens or 8192

    query_func = _resolve_backend_module(model).query
    if provider_env_snapshot is not None:
        model_kwargs["_provider_env_snapshot"] = provider_env_snapshot
    # 智谱模型不需要预先编译，backend_zhipu会处理
    # 其他模型需要编译为markdown
    if spec.provider == "zhipu":
        adapter_system_message = system_message
        adapter_user_message = user_message
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            **model_kwargs,
        )
    else:
        adapter_system_message = (
            compile_prompt_to_md(system_message) if system_message else None
        )
        adapter_user_message = (
            compile_prompt_to_md(user_message) if user_message else None
        )
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=adapter_system_message,
            user_message=adapter_user_message,
            func_spec=func_spec,
            **model_kwargs,
        )

    if not isinstance(info, Mapping):
        raise ResearchDecisionError("Provider response metadata is invalid")
    reported_model = _safe_reported_model(info.get("model"))
    if spec.provider == "openai_compat" and reported_model != spec.client_model:
        # Do not log, hash, persist, or interpolate an untrusted reported model.
        raise ResearchDecisionError("Provider-reported model identity is not exact")

    if reported_model is not None:
        provenance["reported_model"] = reported_model
        provenance["reported_model_exact"] = reported_model == spec.client_model
    effective_trace_params = info.get("_trace_params")
    trace_params = (
        dict(effective_trace_params)
        if isinstance(effective_trace_params, Mapping)
        else dict(model_kwargs)
    )
    if func_spec is not None:
        trace_params["tool_schema_sha256"] = _canonical_digest(
            redact_sensitive_payload(
                {
                    "name": func_spec.name,
                    "description": func_spec.description,
                    "json_schema": func_spec.json_schema,
                }
            )
        )
    response_text = (
        output
        if isinstance(output, str)
        else json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    record_llm_call(
        provider=spec.provider,
        model=model,
        request_style=spec.request_style,
        system_message=info.get("_trace_system_message", ""),
        messages=(
            info["_trace_messages"]
            if isinstance(info.get("_trace_messages"), list)
            else _trace_messages(adapter_system_message, adapter_user_message)
        ),
        response_text=response_text,
        params=trace_params,
        tokens={"input": in_tok_count, "output": out_tok_count},
        latency_ms=max(0, round(float(req_time) * 1000)),
        model_provenance=provenance,
    )

    return output
