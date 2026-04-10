from __future__ import annotations

"""
智谱AI (Zhipu AI) backend implementation
使用智谱API的官方Python SDK
"""
import os
import time

from ai_scientist.utils.provider_registry import resolve_model_provider
from ai_scientist.utils.optional_dependencies import import_optional_module

from .utils import (
    FunctionSpec,
    OutputType,
    compile_prompt_to_md,
    logger,
    summarize_request_kwargs_for_log,
)

zhipuai_sdk = import_optional_module(
    "zhipuai",
    install_hint="Install the 'zhipuai' package to use the treesearch Zhipu backend.",
)


def get_fallback_response(function_name: str, content: str) -> dict:
    """
    当函数调用失败时，返回合理的默认响应

    Args:
        function_name: 函数名称
        content: 模型返回的原始内容

    Returns:
        默认的函数返回值
    """
    # 根据不同的函数名返回不同的默认值
    if function_name == "submit_review":
        # submit_review 函数期望的格式: {"is_bug": bool, "summary": str}
        return {
            "is_bug": True,  # 默认认为有bug，因为函数调用失败了
            "summary": content[:1000] if content else "无法解析执行结果，需要人工检查"
        }
    elif function_name == "analyze_experiment_plots":
        # analyze_experiment_plots 函数期望的格式
        return {
            "plot_analyses": [{"analysis": content[:500] if content else "无法分析图表"}],
            "valid_plots_received": False,
            "vlm_feedback_summary": "函数调用失败，无法生成VLM反馈"
        }
    elif function_name == "parse_metrics":
        # parse_metrics 函数期望的格式
        return {
            "valid_metrics_received": False,
            "metric_names": []
        }
    elif function_name == "select_plots":
        # select_plots 函数期望的格式
        return {
            "selected_plots": []
        }
    elif "parse_execution_result" in function_name:
        # 对于其他代码执行结果解析，返回一个基本的解析结果
        return {
            "running": False,
            "finished": False,
            "analysis": content[:1000] if content else "无法解析执行结果",  # 限制长度
            "bugs_found": True,
            "error_type": "runtime_error",
            "proposed_fix": "需要人工检查代码"
        }
    elif "improve" in function_name or "write" in function_name:
        # 对于代码改进/写作类函数
        return {
            "success": False,
            "content": content[:2000] if content else "无法生成内容",
            "needs_improvement": True
        }
    else:
        # 通用默认响应
        return {
            "success": False,
            "raw_response": content[:1000] if content else "",
            "fallback": True,
            "note": "函数调用失败，使用fallback响应"
        }


def get_ai_client(model: str, max_retries=2):
    """获取智谱AI客户端"""
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        logger.warning("ZHIPU_API_KEY not set, using default empty key")

    client = zhipuai_sdk.ZhipuAI(
        api_key=api_key,
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
    client = get_ai_client(model, max_retries=0)

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
            if user_message and all(isinstance(item, dict) and "type" in item for item in user_message):
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

    logger.debug("Zhipu model: %s", model)
    logger.debug("Zhipu messages count: %d", len(messages))
    logger.debug(
        "Zhipu request params summary: %s",
        summarize_request_kwargs_for_log(request_params),
    )

    # 发送请求
    t0 = time.time()

    try:
        response = client.chat.completions.create(**request_params)
    except Exception as e:
        logger.error(f"智谱API调用失败: {e}")
        logger.error(
            "请求参数摘要: %s", summarize_request_kwargs_for_log(request_params)
        )
        raise

    req_time = time.time() - t0

    # 解析响应
    choice = response.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        # 函数调用
        if choice.message.tool_calls is None:
            # 智谱AI有时候不会调用函数，而是返回普通文本
            # 这种情况下，记录警告并尝试从content中提取JSON
            logger.warning(f"智谱AI未调用函数，返回了普通文本。尝试解析内容...")
            raw_content = choice.message.content
            content = (
                raw_content
                if isinstance(raw_content, str)
                else str(raw_content) if raw_content is not None else ""
            )
            logger.warning(f"返回的内容: {content[:200]}...")

            # 尝试从文本中提取JSON
            try:
                import json
                # 查找JSON格式的响应
                if "```json" in content:
                    # 提取markdown代码块中的JSON
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                    output = json.loads(json_str)
                elif "{" in content and "}" in content:
                    # 尝试提取第一个JSON对象
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_str = content[json_start:json_end]
                    output = json.loads(json_str)
                else:
                    # 无法解析，抛出异常
                    raise ValueError(
                        f"智谱AI未调用函数且无法从文本中提取JSON。\n"
                        f"期望调用函数: {func_spec.name}\n"
                        f"实际返回: {content[:500]}"
                    )
            except Exception as e:
                logger.error(f"解析函数调用失败: {e}")
                logger.error(f"原始内容: {content}")

                # 🆕 最后的fallback：返回一个默认响应，避免程序崩溃
                logger.warning(f"使用默认fallback响应代替函数调用")
                logger.warning(f"这将允许程序继续运行，但结果可能不准确")

                # 根据函数名返回合理的默认值
                fallback_output = get_fallback_response(func_spec.name, content)

                logger.debug("Using fallback response for %s", func_spec.name)
                output = fallback_output
        else:
            # 正常的函数调用
            assert (
                choice.message.tool_calls[0].function.name == func_spec.name
            ), f"函数名不匹配: 期望 '{func_spec.name}', 实际 '{choice.message.tool_calls[0].function.name}'"
            try:
                import json
                output = json.loads(choice.message.tool_calls[0].function.arguments)
                logger.debug("Function call successful: %s", func_spec.name)
            except json.JSONDecodeError as e:
                logger.error(
                    f"解析函数参数失败: {choice.message.tool_calls[0].function.arguments}"
                )
                raise e

    # 获取token使用情况
    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens

    info = {
        "model": response.model,
        "created": response.created,
    }

    return output, req_time, in_tokens, out_tokens, info
