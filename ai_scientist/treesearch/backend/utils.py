from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable, Union

from ai_scientist.utils.optional_dependencies import (
    import_backoff,
    import_optional_module,
)

backoff = import_backoff()
jsonschema = import_optional_module(
    "jsonschema",
    install_hint="Install the 'jsonschema' package to use treesearch function schema validation.",
)
dataclasses_json = import_optional_module(
    "dataclasses_json",
    install_hint="Install the 'dataclasses-json' package to use treesearch function schema serialization.",
)
try:
    DataClassJsonMixinBase = dataclasses_json.DataClassJsonMixin
except ModuleNotFoundError:
    class DataClassJsonMixinBase:  # type: ignore[too-many-ancestors]
        pass

PromptType = Union[str, dict, list]
FunctionCallType = dict
OutputType = Union[str, FunctionCallType]

logger = logging.getLogger("ai-scientist")


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable,
    retry_exceptions: tuple[type[BaseException], ...],
    *args,
    **kwargs,
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.info(f"Backoff exception: {e}")
        return False


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    """Convert a prompt into markdown format"""
    try:
        logger.debug(f"compile_prompt_to_md input: type={type(prompt)}")
        if isinstance(prompt, (list, dict)):
            logger.debug(f"prompt content: {prompt}")

        if prompt is None:
            return ""

        if isinstance(prompt, str):
            return prompt.strip() + "\n"

        if isinstance(prompt, list):
            # Handle empty list case
            if not prompt:
                return ""
            # Special handling for multi-modal messages
            if all(isinstance(item, dict) and "type" in item for item in prompt):
                # For multi-modal messages, just pass through without modification
                return prompt

            try:
                result = "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])
                return result
            except Exception as e:
                logger.error(f"Error processing list items: {e}")
                logger.error("List contents:")
                for i, item in enumerate(prompt):
                    logger.error(f"  Item {i}: type={type(item)}, value={item}")
                raise

        if isinstance(prompt, dict):
            # Check if this is a single multi-modal message
            if "type" in prompt:
                return prompt

            # Regular dict processing
            try:
                out = []
                header_prefix = "#" * _header_depth
                for k, v in prompt.items():
                    logger.debug(f"Processing dict key: {k}")
                    out.append(f"{header_prefix} {k}\n")
                    out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
                return "\n".join(out)
            except Exception as e:
                logger.error(f"Error processing dict: {e}")
                logger.error(f"Dict contents: {prompt}")
                raise

        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    except Exception as e:
        logger.error("Error in compile_prompt_to_md:")
        logger.error(f"Input type: {type(prompt)}")
        logger.error(f"Input content: {prompt}")
        logger.error(f"Error: {str(e)}")
        raise


def _preview_text(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def summarize_messages_for_log(
    messages: list[dict[str, Any]], max_messages: int = 2, max_chars: int = 160
) -> list[dict[str, Any]]:
    """Return a compact and safe preview of chat messages for debug logging."""
    summary: list[dict[str, Any]] = []
    for msg in messages[:max_messages]:
        entry: dict[str, Any] = {"role": msg.get("role", "<missing-role>")}
        content = msg.get("content")

        if isinstance(content, str):
            entry["content_type"] = "str"
            entry["content_len"] = len(content)
            entry["content_preview"] = _preview_text(content, max_chars=max_chars)
        elif isinstance(content, list):
            entry["content_type"] = "list"
            entry["content_len"] = len(content)
            item_types: list[str] = []
            for item in content[:3]:
                if isinstance(item, dict):
                    item_types.append(str(item.get("type", "dict")))
                else:
                    item_types.append(type(item).__name__)
            entry["item_types"] = item_types
        elif isinstance(content, dict):
            entry["content_type"] = "dict"
            entry["keys"] = list(content.keys())[:8]
        else:
            entry["content_type"] = type(content).__name__
            entry["content_preview"] = _preview_text(str(content), max_chars=max_chars)

        summary.append(entry)

    omitted = len(messages) - len(summary)
    if omitted > 0:
        summary.append({"omitted_messages": omitted})
    return summary


def summarize_request_kwargs_for_log(
    kwargs: dict[str, Any], max_chars: int = 160
) -> dict[str, Any]:
    """Return a compact kwargs summary to keep debug logs readable and safe."""
    redacted_keys = {"api_key", "authorization", "headers"}
    summary: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in redacted_keys:
            summary[key] = "<redacted>"
            continue

        if key == "messages" and isinstance(value, list):
            summary[key] = {
                "count": len(value),
                "preview": summarize_messages_for_log(
                    value, max_messages=2, max_chars=max_chars
                ),
            }
            continue

        if key == "tools" and isinstance(value, list):
            tool_names: list[str] = []
            for tool in value:
                if isinstance(tool, dict):
                    func = tool.get("function")
                    if isinstance(func, dict):
                        tool_names.append(str(func.get("name", "<unknown>")))
                        continue
                tool_names.append(type(tool).__name__)
            summary[key] = {"count": len(value), "tool_names": tool_names[:5]}
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str):
                summary[key] = _preview_text(value, max_chars=max_chars)
            else:
                summary[key] = value
            continue

        if isinstance(value, (list, dict)):
            try:
                text = json.dumps(value, ensure_ascii=False)
                summary[key] = _preview_text(text, max_chars=max_chars)
            except TypeError:
                summary[key] = f"<{type(value).__name__}>"
            continue

        summary[key] = f"<{type(value).__name__}>"

    return summary


@dataclass
class FunctionSpec(DataClassJsonMixinBase):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }
