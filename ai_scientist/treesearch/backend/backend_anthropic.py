from __future__ import annotations

import json
import time
import logging

from .utils import (
    FunctionSpec,
    OutputType,
    backoff_create,
    opt_messages_to_list,
    summarize_request_kwargs_for_log,
)
from ai_scientist.utils.optional_dependencies import (
    import_optional_module,
    resolve_exception_types,
)

anthropic = import_optional_module(
    "anthropic",
    install_hint="Install the 'anthropic' package to use the treesearch Anthropic backend.",
    exception_names=(
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "APIStatusError",
    ),
)

logger = logging.getLogger("ai-scientist")


ANTHROPIC_TIMEOUT_EXCEPTIONS = resolve_exception_types(
    anthropic,
    (
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "APIStatusError",
    ),
)


def _select_values_notnone(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _func_spec_to_anthropic_tool(func_spec: FunctionSpec) -> dict:
    return {
        "name": func_spec.name,
        "description": func_spec.description,
        "input_schema": func_spec.json_schema,
    }


def get_ai_client(model : str, max_retries=2):
    client = anthropic.Anthropic(max_retries=max_retries)
    return client

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)

    # Strip provider prefix from model name before sending to API
    raw_model = model_kwargs.get("model", "")
    if "/" in raw_model:
        model_kwargs["model"] = raw_model.split("/", 1)[1]

    filtered_kwargs: dict = _select_values_notnone(model_kwargs)
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192

    if func_spec is not None:
        filtered_kwargs["tools"] = [_func_spec_to_anthropic_tool(func_spec)]
        filtered_kwargs["tool_choice"] = {
            "type": "tool",
            "name": func_spec.name,
        }

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = backoff_create(
        client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    logger.debug(
        "Anthropic request kwargs summary: %s",
        summarize_request_kwargs_for_log(filtered_kwargs),
    )

    if func_spec is not None:
        tool_use_blocks = [b for b in message.content if b.type == "tool_use"]
        assert tool_use_blocks, f"Expected tool_use response but got: {[b.type for b in message.content]}"
        tool_block = tool_use_blocks[0]
        try:
            output = json.loads(tool_block.input) if isinstance(tool_block.input, str) else tool_block.input
        except (json.JSONDecodeError, TypeError):
            output = tool_block.input
    elif "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        text_blocks = [b for b in message.content if b.type == "text"]
        assert text_blocks, f"Expected text response but got: {[b.type for b in message.content]}"
        output: str = text_blocks[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
