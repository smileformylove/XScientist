from __future__ import annotations

"""
智谱AI (Zhipu AI) backend implementation
使用智谱API的官方Python SDK
"""
import os
import time

from ai_scientist.utils.provider_registry import resolve_model_provider
from ai_scientist.utils.optional_dependencies import import_optional_module
from ai_scientist.utils.llm_budget import (
    is_llm_budget_exception,
    llm_budget_manager,
)

from .utils import (
    FunctionSpec,
    OutputType,
    ResearchDecisionError,
    compile_prompt_to_md,
    logger,
    parse_openai_tool_calls,
    summarize_request_kwargs_for_log,
)

zhipuai_sdk = import_optional_module(
    "zhipuai",
    install_hint="Install the 'zhipuai' package to use the treesearch Zhipu backend.",
)


def get_ai_client(model: str, max_retries=2):
    """获取智谱AI客户端"""
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        logger.warning("ZHIPU_API_KEY not set, using default empty key")

    client = zhipuai_sdk.ZhipuAI(
        api_key=api_key,
        max_retries=max_retries,
    )
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    使用智谱AI进行查询

    Args:
        system_message: 系统消息
        user_message: 用户消息
        func_spec: 函数调用规范
        **model_kwargs: 模型参数

    Returns:
        (output, req_time, in_tokens, out_tokens, info)
    """
    model = model_kwargs.get("model", "")
    spec = resolve_model_provider(model)
    try:
        client = get_ai_client(model, max_retries=0)
    except Exception as exc:
        if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
            raise
        logger.error("智谱客户端初始化失败: %s", type(exc).__name__)
        raise ResearchDecisionError("Provider client initialization failed") from None

    # 构建消息列表
    messages = []

    if system_message:
        # 如果是dict，转换为markdown
        if isinstance(system_message, dict):
            system_message = compile_prompt_to_md(system_message)
        elif isinstance(system_message, list):
            system_message = "\n".join(system_message)

        messages.append({"role": "system", "content": system_message})

    if user_message:
        # 如果是dict，转换为markdown
        if isinstance(user_message, dict):
            user_message = compile_prompt_to_md(user_message)
        elif isinstance(user_message, list):
            # 检查是否是多模态消息
            if user_message and all(
                isinstance(item, dict) and "type" in item for item in user_message
            ):
                # 保持多模态格式
                pass
            else:
                user_message = "\n".join(user_message)

        if isinstance(user_message, str):
            messages.append({"role": "user", "content": user_message})
        else:
            # 多模态消息
            messages.append({"role": "user", "content": user_message})

    # 🔧 修复：智谱API要求必须有user消息
    # 如果只有system消息，将system消息作为user消息发送
    if len(messages) == 1 and messages[0]["role"] == "system":
        logger.warning("智谱API要求必须有user消息，将system消息转为user消息")
        messages[0]["role"] = "user"

    # 构建请求参数
    request_params = {
        "model": spec.client_model,
        "messages": messages,
    }

    # 添加支持的参数
    if "temperature" in model_kwargs and model_kwargs["temperature"] is not None:
        request_params["temperature"] = model_kwargs["temperature"]

    if "top_p" in model_kwargs and model_kwargs["top_p"] is not None:
        request_params["top_p"] = model_kwargs["top_p"]

    if "max_tokens" in model_kwargs and model_kwargs["max_tokens"] is not None:
        request_params["max_tokens"] = model_kwargs["max_tokens"]

    # 函数调用
    if func_spec is not None:
        request_params["tools"] = [func_spec.as_openai_tool_dict]
        # 对于智谱AI，不强制使用tool_choice，让模型自己决定
        # request_params["tool_choice"] = func_spec.openai_tool_choice_dict

        logger.debug("Zhipu function calling enabled: %s", func_spec.name)

    logger.debug("Zhipu request model selected")
    logger.debug("Zhipu messages count: %d", len(messages))
    logger.debug(
        "Zhipu request params summary: %s",
        summarize_request_kwargs_for_log(request_params),
    )

    # 发送请求
    t0 = time.time()

    try:
        reservation = llm_budget_manager.reserve(
            model=model,
            prompt={
                "messages": messages,
                "tools": request_params.get("tools"),
            },
            system_message=system_message,
            max_output_tokens=request_params.get("max_tokens") or 8192,
        )
        with reservation:
            if reservation.timeout_seconds is not None:
                request_params["timeout"] = reservation.timeout_seconds
            response = client.chat.completions.create(**request_params)
            reservation.settle(response=response)
    except Exception as e:
        if is_llm_budget_exception(e):
            raise
        logger.error("智谱API调用失败: %s", type(e).__name__)
        logger.error(
            "请求参数摘要: %s", summarize_request_kwargs_for_log(request_params)
        )
        raise ResearchDecisionError("Provider request failed") from None

    req_time = time.time() - t0

    try:
        choices = response.choices
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

        # 获取token使用情况
        in_tokens = response.usage.prompt_tokens
        out_tokens = response.usage.completion_tokens
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (in_tokens, out_tokens)
        ):
            raise ResearchDecisionError("Provider token usage is invalid")

        info = {
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "_trace_system_message": "",
            "_trace_messages": messages,
            "_trace_params": {
                key: value
                for key, value in request_params.items()
                if key not in {"model", "messages", "tools"}
            },
        }
    except Exception as exc:
        if isinstance(exc, ResearchDecisionError):
            raise
        logger.error("智谱响应验证失败: %s", type(exc).__name__)
        raise ResearchDecisionError("Provider response envelope is invalid") from None

    return output, req_time, in_tokens, out_tokens, info
