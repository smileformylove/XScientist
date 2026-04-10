from __future__ import annotations

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

def get_ai_client(model : str, max_retries=2) -> anthropic.AnthropicBedrock:
    client = anthropic.AnthropicBedrock(max_retries=max_retries)
    return client

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)

    filtered_kwargs: dict = _select_values_notnone(model_kwargs)
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    if func_spec is not None:
        raise NotImplementedError(
            "Anthropic does not support function calling for now."
        )

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

    if "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output: str = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
