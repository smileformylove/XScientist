from __future__ import annotations

import json
import logging
import time

from .utils import (
    FunctionSpec,
    OutputType,
    backoff_create,
    opt_messages_to_list,
    summarize_messages_for_log,
    summarize_request_kwargs_for_log,
)
from ai_scientist.utils.provider_registry import (
    build_openai_compatible_client_kwargs,
    resolve_model_provider,
)
from ai_scientist.utils.optional_dependencies import (
    import_optional_module,
    resolve_exception_types,
)

openai = import_optional_module(
    "openai",
    install_hint="Install the 'openai' package to use the treesearch OpenAI-compatible backend.",
    exception_names=(
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    ),
)

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = resolve_exception_types(
    openai,
    (
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    ),
)


def _select_values_notnone(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    kwargs, _ = build_openai_compatible_client_kwargs(
        model,
        max_retries=max_retries,
    )
    return openai.OpenAI(**kwargs)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = model_kwargs.get("model", "")
    spec = resolve_model_provider(model)
    client = get_ai_client(model, max_retries=0)
    filtered_kwargs: dict = _select_values_notnone(model_kwargs)
    filtered_kwargs["model"] = spec.client_model

    # Handle system and user messages
    if spec.provider == "zhipu":
        # Zhipu AI expects standard OpenAI format messages
        # Convert dict/list prompts to string for Zhipu
        from .utils import compile_prompt_to_md

        sys_msg = system_message
        usr_msg = user_message

        # Compile to markdown if dict
        if isinstance(system_message, dict):
            sys_msg = compile_prompt_to_md(system_message)
        elif isinstance(system_message, list):
            # For list messages, convert to string
            sys_msg = "\n".join(system_message) if system_message else None

        if isinstance(user_message, dict):
            usr_msg = compile_prompt_to_md(user_message)
        elif isinstance(user_message, list):
            # For list messages, check if it's multimodal
            if user_message and all(isinstance(item, dict) and "type" in item for item in user_message):
                # Keep multimodal format for Zhipu
                usr_msg = user_message
            else:
                usr_msg = "\n".join(user_message) if user_message else None

        messages = opt_messages_to_list(sys_msg, usr_msg)

        logger.debug(
            "[ZHIPU-OPENAI-COMPAT] messages preview: %s",
            summarize_messages_for_log(messages),
        )
    else:
        messages = opt_messages_to_list(system_message, user_message)

    # Filter out unsupported parameters for Zhipu AI
    if spec.provider == "zhipu":
        # Zhipu AI doesn't support these OpenAI-specific parameters
        unsupported_params = [
            'reasoning_effort',      # OpenAI o1/o3 specific
            'max_completion_tokens',  # OpenAI o1/o3 specific (use max_tokens instead)
            'seed',                   # Not supported by Zhipu
            'top_k',                  # Zhipu uses top_p instead
        ]
        for param in unsupported_params:
            if param in filtered_kwargs:
                logger.debug(
                    "[ZHIPU-OPENAI-COMPAT] removing unsupported param: %s", param
                )
                filtered_kwargs.pop(param, None)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if spec.provider == "zhipu":
        logger.debug(
            "[ZHIPU-OPENAI-COMPAT] request kwargs summary: %s",
            summarize_request_kwargs_for_log(filtered_kwargs),
        )
        logger.debug("[ZHIPU-OPENAI-COMPAT] messages count: %d", len(messages))

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            logger.debug(
                "Function call response received: fn=%s has_tool_calls=%s",
                func_spec.name,
                bool(choice.message.tool_calls),
            )
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
