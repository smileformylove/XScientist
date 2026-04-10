from __future__ import annotations

import importlib
from functools import lru_cache

from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from ai_scientist.utils.provider_registry import resolve_model_provider


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

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    spec = resolve_model_provider(model)
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
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    query_func = _resolve_backend_module(model).query

    # 智谱模型不需要预先编译，backend_zhipu会处理
    # 其他模型需要编译为markdown
    if spec.provider == "zhipu":
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            **model_kwargs,
        )
    else:
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=compile_prompt_to_md(system_message) if system_message else None,
            user_message=compile_prompt_to_md(user_message) if user_message else None,
            func_spec=func_spec,
            **model_kwargs,
        )

    return output
