# Modified by XScientist contributors from the AI-Scientist-v2/AIDE lineage.
# See THIRD_PARTY_NOTICES.md for provenance and license details.
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Union

from ai_scientist.utils.optional_dependencies import (
    import_backoff,
    import_optional_module,
)
from ai_scientist.utils.privacy import redact_sensitive_text
from ai_scientist.utils.llm_budget import llm_budget_manager

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
BACKOFF_MAX_TRIES = 3
_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
_MULTIMODAL_TYPES = frozenset(
    {"text", "image_url", "input_text", "input_image", "audio", "file"}
)


def _safe_metadata_label(value: Any, *, allowed: frozenset[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        return "<invalid>"
    if allowed is not None and value not in allowed:
        return "<invalid>"
    return value if redact_sensitive_text(value) == value else "<redacted>"


@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
    max_tries=BACKOFF_MAX_TRIES,
)
def backoff_create(
    create_fn: Callable,
    retry_exceptions: tuple[type[BaseException], ...],
    *args,
    **kwargs,
):
    budget_model = kwargs.pop("_budget_model", None)
    budget_prompt = kwargs.pop("_budget_prompt", None)
    budget_system_message = kwargs.pop("_budget_system_message", None)
    budget_max_output_tokens = kwargs.pop("_budget_max_output_tokens", None)
    reservation = None
    if budget_model:
        reservation = llm_budget_manager.reserve(
            model=str(budget_model),
            prompt=budget_prompt,
            system_message=budget_system_message,
            max_output_tokens=budget_max_output_tokens,
        )
        if reservation.timeout_seconds is not None:
            # The remaining wall-time budget is an authority boundary.  It
            # must be able to shorten a provider/caller timeout that was
            # already placed in the request kwargs.
            requested_timeout = kwargs.get("timeout")
            kwargs["timeout"] = (
                reservation.timeout_seconds
                if requested_timeout is None
                else min(float(requested_timeout), reservation.timeout_seconds)
            )
    try:
        if reservation is None:
            return create_fn(*args, **kwargs)
        with reservation:
            result = create_fn(*args, **kwargs)
            reservation.settle(response=result)
            return result
    except retry_exceptions as e:
        logger.info("Backoff exception type: %s", type(e).__name__)
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
                logger.error("Error processing list items: %s", type(e).__name__)
                logger.error(
                    "List item types: %s", [type(item).__name__ for item in prompt]
                )
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
                    out.append(f"{header_prefix} {k}\n")
                    out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
                return "\n".join(out)
            except Exception as e:
                logger.error("Error processing dict: %s", type(e).__name__)
                logger.error("Dict key count: %d", len(prompt))
                raise

        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    except Exception as e:
        logger.error("Error in compile_prompt_to_md:")
        logger.error(f"Input type: {type(prompt)}")
        logger.error("Error type: %s", type(e).__name__)
        raise


def summarize_messages_for_log(
    messages: list[dict[str, Any]], max_messages: int = 2, max_chars: int = 160
) -> list[dict[str, Any]]:
    """Return payload-free message metadata for debug logging.

    ``max_chars`` remains part of the public helper signature for compatibility,
    but message text is deliberately never copied into logs.
    """
    del max_chars
    summary: list[dict[str, Any]] = []
    for msg in messages[:max_messages]:
        entry: dict[str, Any] = {
            "role": _safe_metadata_label(
                msg.get("role"),
                allowed=_MESSAGE_ROLES,
            )
        }
        content = msg.get("content")

        if isinstance(content, str):
            entry["content_type"] = "str"
            entry["content_len"] = len(content)
        elif isinstance(content, list):
            entry["content_type"] = "list"
            entry["content_len"] = len(content)
            item_types: list[str] = []
            for item in content[:3]:
                if isinstance(item, dict):
                    item_types.append(
                        _safe_metadata_label(
                            item.get("type"),
                            allowed=_MULTIMODAL_TYPES,
                        )
                    )
                else:
                    item_types.append(type(item).__name__)
            entry["item_types"] = item_types
        elif isinstance(content, dict):
            entry["content_type"] = "dict"
            entry["content_len"] = len(content)
        else:
            entry["content_type"] = type(content).__name__

        summary.append(entry)

    omitted = len(messages) - len(summary)
    if omitted > 0:
        summary.append({"omitted_messages": omitted})
    return summary


def summarize_request_kwargs_for_log(
    kwargs: dict[str, Any], max_chars: int = 160
) -> dict[str, Any]:
    """Return payload-free request metadata to keep debug logs safe."""
    del max_chars
    redacted_keys = {"api_key", "authorization", "headers", "token"}
    summary: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key.lower() in redacted_keys:
            summary[key] = "<redacted>"
            continue

        if key == "messages" and isinstance(value, list):
            summary[key] = {
                "count": len(value),
                "preview": summarize_messages_for_log(value, max_messages=2),
            }
            continue

        if key == "tools" and isinstance(value, list):
            tool_names: list[str] = []
            for tool in value:
                if isinstance(tool, dict):
                    func = tool.get("function")
                    if isinstance(func, dict):
                        tool_names.append(_safe_metadata_label(func.get("name")))
                        continue
                tool_names.append(type(tool).__name__)
            summary[key] = {"count": len(value), "tool_names": tool_names[:5]}
            continue

        if key == "model" and isinstance(value, str):
            # Model identifiers are configuration data, but a malformed caller
            # can still place a credential-shaped value in this field. Never
            # let that value reach debug logs.
            safe_model = redact_sensitive_text(value)
            summary[key] = (
                safe_model
                if safe_model == value
                else {"type": "redacted_model", "length": len(value)}
            )
            continue

        if isinstance(value, str):
            summary[key] = {"type": "str", "length": len(value)}
            continue
        if isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
            continue
        if isinstance(value, list):
            summary[key] = {"type": "list", "count": len(value)}
            continue
        if isinstance(value, dict):
            summary[key] = {"type": "dict", "count": len(value)}
            continue

        summary[key] = f"<{type(value).__name__}>"

    return summary


class ResearchDecisionError(RuntimeError):
    """Raised when a research decision cannot be obtained safely."""


class FunctionCallValidationError(ResearchDecisionError, ValueError):
    """Raised when a provider response violates a requested function contract."""


MAX_FUNCTION_CALL_ARGUMENT_BYTES = 1024 * 1024


def _response_field(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _reject_json_constant(_value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("Duplicate JSON object key")
        payload[key] = value
    return payload


def _mapping_has_only_string_keys(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _mapping_has_only_string_keys(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_mapping_has_only_string_keys(item) for item in value)
    return True


def validate_function_call_payload(
    func_spec: "FunctionSpec",
    *,
    function_name: Any,
    arguments: Any,
) -> dict:
    """Decode and validate one provider function call without leaking its payload.

    Function-call results influence experiment planning and review decisions, so
    malformed output must never be converted into a plausible fallback object.
    Provider adapters may supply either the standard JSON string or an already
    decoded mapping; all other representations fail closed.
    """
    if not isinstance(function_name, str) or function_name != func_spec.name:
        raise FunctionCallValidationError("Tool call function name does not match")

    if isinstance(arguments, str):
        if len(arguments.encode("utf-8")) > MAX_FUNCTION_CALL_ARGUMENT_BYTES:
            raise FunctionCallValidationError("Tool call arguments exceed size limit")
        try:
            payload = json.loads(
                arguments,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
            MemoryError,
        ):
            raise FunctionCallValidationError(
                "Tool call arguments are not valid JSON"
            ) from None
    elif isinstance(arguments, Mapping):
        payload = dict(arguments)
        if not _mapping_has_only_string_keys(payload):
            raise FunctionCallValidationError(
                "Tool call arguments must use string JSON object keys"
            )
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError):
            raise FunctionCallValidationError(
                "Tool call arguments contain a non-JSON value"
            ) from None
    else:
        raise FunctionCallValidationError("Tool call arguments must be a JSON object")

    if not isinstance(payload, dict):
        raise FunctionCallValidationError(
            "Tool call arguments must decode to a JSON object"
        )
    try:
        encoded_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError, MemoryError):
        raise FunctionCallValidationError(
            "Tool call arguments contain a non-JSON value"
        ) from None
    if len(encoded_payload) > MAX_FUNCTION_CALL_ARGUMENT_BYTES:
        raise FunctionCallValidationError("Tool call arguments exceed size limit")

    validator = jsonschema.Draft202012Validator(func_spec.json_schema)
    try:
        errors = sorted(
            validator.iter_errors(payload),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except (TypeError, ValueError, OverflowError, RecursionError, MemoryError):
        raise FunctionCallValidationError(
            "Tool call arguments cannot be safely validated"
        ) from None
    if errors:
        error = errors[0]
        detail = f"rule={error.validator or 'unknown'}"
        if error.validator == "required" and isinstance(
            error.validator_value, Iterable
        ):
            error_instance = (
                error.instance if isinstance(error.instance, Mapping) else {}
            )
            missing = [
                str(field)
                for field in error.validator_value
                if field not in error_instance
            ]
            if missing:
                detail = "missing=" + ",".join(missing)
        raise FunctionCallValidationError(
            f"Tool call arguments violate FunctionSpec schema ({detail})"
        )
    return payload


def parse_openai_tool_calls(func_spec: "FunctionSpec", tool_calls: Any) -> dict:
    """Validate the single tool call returned by an OpenAI-compatible API."""
    if tool_calls is None or isinstance(tool_calls, (str, bytes, Mapping)):
        raise FunctionCallValidationError(
            "Provider response did not contain a tool call"
        )
    try:
        calls = list(tool_calls)
    except TypeError:
        raise FunctionCallValidationError(
            "Provider response did not contain a tool call"
        ) from None
    if len(calls) != 1:
        raise FunctionCallValidationError(
            f"Provider response must contain exactly one tool call; received {len(calls)}"
        )

    function = _response_field(calls[0], "function")
    if function is None:
        raise FunctionCallValidationError("Tool call is missing its function payload")
    return validate_function_call_payload(
        func_spec,
        function_name=_response_field(function, "name"),
        arguments=_response_field(function, "arguments"),
    )


@dataclass
class FunctionSpec(DataClassJsonMixinBase):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or _FUNCTION_NAME_RE.fullmatch(self.name) is None
            or redact_sensitive_text(self.name) != self.name
        ):
            raise ValueError("FunctionSpec name is invalid")
        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or len(self.description) > 4096
        ):
            raise ValueError("FunctionSpec description is invalid")

        def schema_children(node: Mapping[str, Any]):
            for keyword in ("properties", "patternProperties", "definitions", "$defs"):
                children = node.get(keyword)
                if isinstance(children, Mapping):
                    yield from (
                        child
                        for child in children.values()
                        if isinstance(child, Mapping)
                    )
            dependencies = node.get("dependencies")
            if isinstance(dependencies, Mapping):
                yield from (
                    child
                    for child in dependencies.values()
                    if isinstance(child, Mapping)
                )
            dependent_schemas = node.get("dependentSchemas")
            if isinstance(dependent_schemas, Mapping):
                yield from (
                    child
                    for child in dependent_schemas.values()
                    if isinstance(child, Mapping)
                )
            for keyword in (
                "additionalProperties",
                "additionalItems",
                "contains",
                "propertyNames",
                "if",
                "then",
                "else",
                "not",
            ):
                child = node.get(keyword)
                if isinstance(child, Mapping):
                    yield child
            items = node.get("items")
            if isinstance(items, Mapping):
                yield items
            elif isinstance(items, list):
                yield from (child for child in items if isinstance(child, Mapping))
            for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
                children = node.get(keyword)
                if isinstance(children, list):
                    yield from (
                        child for child in children if isinstance(child, Mapping)
                    )

        def require_closed_objects(node: Mapping[str, Any]) -> None:
            node_type = node.get("type")
            object_typed = node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            )
            object_keywords = {
                "properties",
                "patternProperties",
                "required",
                "minProperties",
                "maxProperties",
                "dependencies",
                "dependentRequired",
                "dependentSchemas",
                "propertyNames",
                "unevaluatedProperties",
            }
            if (
                object_typed or any(keyword in node for keyword in object_keywords)
            ) and ("additionalProperties" not in node):
                raise ValueError(
                    "FunctionSpec object schemas must explicitly declare "
                    "additionalProperties"
                )
            for child in schema_children(node):
                require_closed_objects(child)

        # Copy without rewriting schema semantics. In particular, recursively
        # inserting ``additionalProperties: false`` can corrupt allOf/oneOf
        # contracts and data stored under default/enum. Strictness must be an
        # explicit author decision at every actual object-schema node.
        self.json_schema = copy.deepcopy(self.json_schema)

        def reject_ignored_legacy_keywords(node: Mapping[str, Any]) -> None:
            ignored = sorted(
                keyword
                for keyword in ("dependencies", "additionalItems")
                if keyword in node
            )
            if ignored:
                raise ValueError(
                    "FunctionSpec uses keywords ignored by JSON Schema 2020-12: "
                    + ", ".join(ignored)
                )
            for child in schema_children(node):
                reject_ignored_legacy_keywords(child)

        jsonschema.Draft202012Validator.check_schema(self.json_schema)
        if self.json_schema.get("type") != "object":
            raise ValueError("FunctionSpec root schema must have type object")
        reject_ignored_legacy_keywords(self.json_schema)
        require_closed_objects(self.json_schema)

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
