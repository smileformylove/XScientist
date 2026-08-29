from __future__ import annotations

import logging
import time
from typing import Mapping

from .utils import (
    FunctionSpec,
    OutputType,
    ResearchDecisionError,
    backoff_create,
    opt_messages_to_list,
    parse_openai_tool_calls,
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
from ai_scientist.utils.llm_budget import is_llm_budget_exception

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
    return {key: value for key, value in payload.items() if value is not None}


def get_ai_client(
    model: str,
    max_retries=2,
    *,
    env: Mapping[str, str] | None = None,
) -> openai.OpenAI:
    kwargs, _ = build_openai_compatible_client_kwargs(
        model,
        env=env,
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
    provider_env_snapshot = model_kwargs.pop("_provider_env_snapshot", None)
    if provider_env_snapshot is not None and not isinstance(
        provider_env_snapshot, Mapping
    ):
        raise ResearchDecisionError("Provider route snapshot is invalid")
    try:
        client = get_ai_client(
            model,
            max_retries=0,
            env=provider_env_snapshot,
        )
    except Exception as exc:
        if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
            raise
        logger.error(
            "OpenAI-compatible client initialization failed: %s",
            type(exc).__name__,
        )
        raise ResearchDecisionError("Provider client initialization failed") from None
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
            if user_message and all(
                isinstance(item, dict) and "type" in item for item in user_message
            ):
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
            "reasoning_effort",  # OpenAI o1/o3 specific
            "max_completion_tokens",  # OpenAI o1/o3 specific (use max_tokens instead)
            "seed",  # Not supported by Zhipu
            "top_k",  # Zhipu uses top_p instead
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
    try:
        completion = backoff_create(
            client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            _budget_model=model,
            _budget_prompt={
                "messages": messages,
                "tools": filtered_kwargs.get("tools"),
            },
            _budget_system_message=system_message,
            _budget_max_output_tokens=(
                filtered_kwargs.get("max_completion_tokens")
                or filtered_kwargs.get("max_tokens")
                or 8192
            ),
            messages=messages,
            **filtered_kwargs,
        )
    except Exception as exc:
        if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
            raise
        logger.error("OpenAI-compatible request failed: %s", type(exc).__name__)
        raise ResearchDecisionError("Provider request failed") from None
    req_time = time.time() - t0

    if completion is False:
        raise ResearchDecisionError("Provider request failed after bounded retries")

    try:
        choices = completion.choices
        if len(choices) != 1:
            raise ResearchDecisionError(
                "Provider response must contain exactly one choice"
            )
        choice = choices[0]
        expected_finish_reason = "tool_calls" if func_spec is not None else "stop"
        if getattr(choice, "finish_reason", None) != expected_finish_reason:
            raise ResearchDecisionError(
                "Provider response did not terminate with the expected reason"
            )

        if func_spec is None:
            output = choice.message.content
            if not isinstance(output, str) or not output.strip():
                raise ResearchDecisionError("Provider response content is not text")
        else:
            output = parse_openai_tool_calls(
                func_spec, getattr(choice.message, "tool_calls", None)
            )
            logger.debug("Validated function call response: fn=%s", func_spec.name)

        in_tokens = completion.usage.prompt_tokens
        out_tokens = completion.usage.completion_tokens
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (in_tokens, out_tokens)
        ):
            raise ResearchDecisionError("Provider token usage is invalid")

        info = {
            "system_fingerprint": getattr(completion, "system_fingerprint", None),
            "model": getattr(completion, "model", None),
            "created": getattr(completion, "created", None),
            "_trace_system_message": "",
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
        logger.error(
            "OpenAI-compatible response validation failed: %s",
            type(exc).__name__,
        )
        raise ResearchDecisionError("Provider response envelope is invalid") from None

    return output, req_time, in_tokens, out_tokens, info
