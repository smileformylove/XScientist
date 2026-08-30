# Modified by XScientist contributors from the AI-Scientist-v2/AIDE lineage.
# See THIRD_PARTY_NOTICES.md for provenance and license details.
from __future__ import annotations

import time
import logging

from .utils import (
    FunctionCallValidationError,
    FunctionSpec,
    OutputType,
    ResearchDecisionError,
    backoff_create,
    opt_messages_to_list,
    summarize_request_kwargs_for_log,
    validate_function_call_payload,
)
from ai_scientist.utils.optional_dependencies import (
    import_optional_module,
    resolve_exception_types,
)
from ai_scientist.utils.llm_budget import is_llm_budget_exception

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
    return {key: value for key, value in payload.items() if value is not None}


def _func_spec_to_anthropic_tool(func_spec: FunctionSpec) -> dict:
    return {
        "name": func_spec.name,
        "description": func_spec.description,
        "input_schema": func_spec.json_schema,
    }


def get_ai_client(model: str, max_retries=2):
    client = anthropic.Anthropic(max_retries=max_retries)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    try:
        client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    except Exception as exc:
        if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
            raise
        logger.error("Anthropic client initialization failed: %s", type(exc).__name__)
        raise ResearchDecisionError("Provider client initialization failed") from None

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
    try:
        message = backoff_create(
            client.messages.create,
            ANTHROPIC_TIMEOUT_EXCEPTIONS,
            _budget_model=raw_model,
            _budget_prompt={
                "messages": messages,
                "tools": filtered_kwargs.get("tools"),
            },
            _budget_system_message=system_message,
            _budget_max_output_tokens=filtered_kwargs.get("max_tokens") or 8192,
            messages=messages,
            **filtered_kwargs,
        )
    except Exception as exc:
        if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
            raise
        logger.error("Anthropic request failed: %s", type(exc).__name__)
        raise ResearchDecisionError("Provider request failed") from None
    req_time = time.time() - t0
    if message is False:
        raise ResearchDecisionError("Provider request failed after bounded retries")
    logger.debug(
        "Anthropic request kwargs summary: %s",
        summarize_request_kwargs_for_log(filtered_kwargs),
    )

    try:
        content = message.content
        if not isinstance(content, list):
            raise ResearchDecisionError("Provider response content is invalid")
        expected_stop_reason = "tool_use" if func_spec is not None else "end_turn"
        if getattr(message, "stop_reason", None) != expected_stop_reason:
            raise ResearchDecisionError(
                "Provider response did not terminate with the expected reason"
            )
        if func_spec is not None:
            tool_use_blocks = [
                block for block in content if getattr(block, "type", None) == "tool_use"
            ]
            if len(tool_use_blocks) != 1:
                raise FunctionCallValidationError(
                    "Provider response must contain exactly one tool_use block; "
                    f"received {len(tool_use_blocks)}"
                )
            tool_block = tool_use_blocks[0]
            output = validate_function_call_payload(
                func_spec,
                function_name=getattr(tool_block, "name", None),
                arguments=getattr(tool_block, "input", None),
            )
        elif "thinking" in filtered_kwargs:
            if (
                len(content) != 2
                or getattr(content[0], "type", None) != "thinking"
                or getattr(content[1], "type", None) != "text"
                or not isinstance(getattr(content[1], "text", None), str)
            ):
                raise ResearchDecisionError("Provider thinking response is invalid")
            output = content[1].text
            if not output.strip():
                raise ResearchDecisionError("Provider response content is not text")
        else:
            text_blocks = [
                block for block in content if getattr(block, "type", None) == "text"
            ]
            if len(text_blocks) != 1 or not isinstance(
                getattr(text_blocks[0], "text", None), str
            ):
                raise ResearchDecisionError("Provider text response is invalid")
            output = text_blocks[0].text
            if not output.strip():
                raise ResearchDecisionError("Provider response content is not text")

        in_tokens = message.usage.input_tokens
        out_tokens = message.usage.output_tokens
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (in_tokens, out_tokens)
        ):
            raise ResearchDecisionError("Provider token usage is invalid")

        info = {
            "stop_reason": getattr(message, "stop_reason", None),
            "model": getattr(message, "model", None),
            "_trace_system_message": system_message or "",
            "_trace_messages": messages,
            "_trace_params": {
                key: value
                for key, value in filtered_kwargs.items()
                if key not in {"model", "tools"}
            },
        }
    except Exception as exc:
        if isinstance(exc, ResearchDecisionError):
            raise
        logger.error("Anthropic response validation failed: %s", type(exc).__name__)
        raise ResearchDecisionError("Provider response envelope is invalid") from None

    return output, req_time, in_tokens, out_tokens, info
