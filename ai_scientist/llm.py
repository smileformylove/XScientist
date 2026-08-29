from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.parse import quote
from ai_scientist.protocol.llm_trace import record_llm_call, strict_llm_tracing
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.utils.llm_budget import LLMBudgetExceeded, llm_budget_manager
from ai_scientist.utils.optional_dependencies import (
    import_backoff,
    import_optional_module,
    resolve_exception_types,
)
from ai_scientist.utils.provider_registry import (
    build_openai_compatible_client_kwargs,
    model_provenance as build_model_provenance,
    model_uses_anthropic_client,
    resolve_model_provider,
)
from ai_scientist.utils.privacy import redact_sensitive_text

backoff = import_backoff()
anthropic = import_optional_module(
    "anthropic",
    install_hint="Install the 'anthropic' package to use Anthropic-backed models.",
    exception_names=("RateLimitError",),
)
openai = import_optional_module(
    "openai",
    install_hint="Install the 'openai' package to use OpenAI-compatible models.",
    exception_names=("RateLimitError", "APITimeoutError", "InternalServerError"),
)
requests = import_optional_module(
    "requests",
    install_hint="Install the 'requests' package to use HuggingFace HTTP fallback calls.",
)
_OPENAI_RETRY_EXCEPTIONS = resolve_exception_types(
    openai,
    ("RateLimitError", "APITimeoutError", "InternalServerError"),
)
_ANTHROPIC_RETRY_EXCEPTIONS = resolve_exception_types(
    anthropic,
    ("RateLimitError",),
)

MAX_NUM_TOKENS = 4096
MAX_RESEARCH_PROVIDER_RETRIES = 3
OPENAI_COMPAT_CALL_TIMEOUT_SECONDS = 30.0
MAX_BATCH_RESPONSES = 8


class LLMResponseContractError(RuntimeError):
    """A provider response cannot safely drive a research action."""


def _safe_reported_model(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if (
        not value
        or value != value.strip()
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or redact_sensitive_text(value) != value
    ):
        return None
    return value


def _validate_openai_compat_response(
    spec: Any,
    response: Any,
    *,
    expected_choices: int = 1,
) -> str | None:
    """Fail closed on model substitution and truncated custom-provider output."""

    reported_model = _safe_reported_model(getattr(response, "model", None))
    if getattr(spec, "provider", None) != "openai_compat":
        return reported_model
    if reported_model != getattr(spec, "client_model", None):
        raise LLMResponseContractError("Provider-reported model identity is not exact")
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or len(choices) != expected_choices:
        raise LLMResponseContractError(
            "Provider response contains an unexpected number of choices"
        )
    for choice in choices:
        if getattr(choice, "finish_reason", None) != "stop":
            raise LLMResponseContractError(
                "Provider response did not terminate normally"
            )
        content = getattr(getattr(choice, "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseContractError(
                "Provider response content is not valid text"
            )
    return reported_model


def _huggingface_http_fallback_url(client_model: str) -> str:
    """Return the legacy inference endpoint for the exact requested model."""

    normalized = str(client_model or "").strip().strip("/")
    if not normalized:
        raise ValueError("HuggingFace fallback requires a concrete model id")
    return "https://api-inference.huggingface.co/models/" + quote(
        normalized, safe="/-_."
    )


def _is_huggingface_protocol_compatibility_error(exc: Exception) -> bool:
    """Limit HTTP fallback to client/protocol incompatibilities.

    Authentication, rate-limit, timeout, and server failures must retain their
    original meaning.  Falling back on those errors can silently change both
    provider behaviour and the scientific provenance of the call.
    """

    if isinstance(exc, (AttributeError, TypeError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return status_code in {404, 405, 415, 422}


def _huggingface_generated_text(response: Any) -> str:
    payload = response.json()
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict) and isinstance(payload.get("generated_text"), str):
        return payload["generated_text"]
    raise ValueError("HuggingFace response did not contain generated_text")


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Local models can be much slower than hosted APIs; keep a conservative default to
# avoid server-side timeouts during long generations. Override per machine if needed.
OLLAMA_MAX_NUM_TOKENS = _env_int("AI_SCIENTIST_OLLAMA_MAX_TOKENS", 1024)

AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # DeepSeek Models
    "deepseek-coder-v2-0724",
    "deepcoder-14b",
    # Llama 3 models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # Google Gemini models
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    # GPT-OSS models via Ollama
    "ollama/gpt-oss:20b",
    "ollama/gpt-oss:120b",
    # Qwen models via Ollama
    "ollama/qwen3:8b",
    "ollama/qwen3:32b",
    "ollama/qwen3:235b",
    "ollama/qwen2.5vl:8b",
    "ollama/qwen2.5vl:32b",
    "ollama/qwen3-coder:70b",
    "ollama/qwen3-coder:480b",
    # Deepseek models via Ollama
    "ollama/deepseek-r1:8b",
    "ollama/deepseek-r1:32b",
    "ollama/deepseek-r1:70b",
    "ollama/deepseek-r1:671b",
    # Zhipu AI models
    "glm-4-flash",
    "glm-4-plus",
    "glm-4-air",
    "glm-4",
    # Provider-prefixed variants for multi-vendor routing
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/o3-mini-2025-01-31",
    "gemini/gemini-2.5-pro-preview-03-25",
    "zhipu/glm-4-plus",
    "openrouter/meta-llama/llama-3.1-405b-instruct",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-coder-v2-0724",
    "huggingface/agentica-org/DeepCoder-14B-Preview",
    "openai_compat/custom-model",
]


def _budgeted_provider_call(
    *,
    model: str,
    prompt: Any,
    system_message: Any,
    max_output_tokens: int,
    create,
    output_multiplier: int = 1,
):
    reservation = llm_budget_manager.reserve(
        model=model,
        prompt=prompt,
        system_message=system_message,
        max_output_tokens=max_output_tokens,
        output_multiplier=output_multiplier,
    )
    timeout_seconds = reservation.timeout_seconds
    if resolve_model_provider(model).provider == "openai_compat":
        timeout_seconds = min(
            float(timeout_seconds or OPENAI_COMPAT_CALL_TIMEOUT_SECONDS),
            OPENAI_COMPAT_CALL_TIMEOUT_SECONDS,
        )
    with reservation:
        response = create(timeout_seconds)
        reservation.settle(response=response)
        return response


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    _OPENAI_RETRY_EXCEPTIONS + _ANTHROPIC_RETRY_EXCEPTIONS,
    max_tries=MAX_RESEARCH_PROVIDER_RETRIES,
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if (
        isinstance(n_responses, bool)
        or not isinstance(n_responses, int)
        or not 1 <= n_responses <= MAX_BATCH_RESPONSES
    ):
        raise ValueError("n_responses must be an integer between 1 and 8")
    msg = prompt
    if msg_history is None:
        msg_history = []
    spec = resolve_model_provider(model)
    direct_batch_call = False
    request_history = msg_history + [{"role": "user", "content": msg}]

    if spec.provider == "ollama":
        direct_batch_call = True
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = _budgeted_provider_call(
            model=model,
            prompt=new_msg_history,
            system_message=system_message,
            max_output_tokens=OLLAMA_MAX_NUM_TOKENS,
            output_multiplier=n_responses,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=OLLAMA_MAX_NUM_TOKENS,
                n=n_responses,
                stop=None,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif spec.request_style == "openai_chat":
        direct_batch_call = True
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = _budgeted_provider_call(
            model=model,
            prompt=new_msg_history,
            system_message=system_message,
            max_output_tokens=MAX_NUM_TOKENS,
            output_multiplier=n_responses,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                stop=None,
                seed=0,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
        _validate_openai_compat_response(
            spec,
            response,
            expected_choices=n_responses,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    # One provider batch is one invocation. Persist one semantic receipt and
    # one aggregate usage record rather than multiplying token accounting by
    # the number of returned choices. The per-choice texts are bound together
    # in the response-array digest.
    if direct_batch_call:
        _record_llm_call_safe(
            spec=spec,
            model=model,
            system_message=system_message,
            messages=request_history,
            response_text=json.dumps(content, ensure_ascii=False),
            params={
                "temperature": temperature,
                "max_tokens": MAX_NUM_TOKENS,
                "n": n_responses,
                "seed": 0,
            },
            response=response,
            client=client,
        )

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    spec = resolve_model_provider(model)
    if spec.provider == "ollama":
        return _budgeted_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
            max_output_tokens=OLLAMA_MAX_NUM_TOKENS,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *prompt,
                ],
                temperature=temperature,
                max_tokens=OLLAMA_MAX_NUM_TOKENS,
                n=1,
                stop=None,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
    elif spec.request_style == "openai_chat":
        return _budgeted_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
            max_output_tokens=MAX_NUM_TOKENS,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *prompt,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
                seed=0,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
    elif spec.request_style == "openai_reasoning":
        return _budgeted_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
            max_output_tokens=MAX_NUM_TOKENS,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "user", "content": system_message},
                    *prompt,
                ],
                temperature=1,
                max_completion_tokens=MAX_NUM_TOKENS,
                n=1,
                seed=0,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )

    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    _OPENAI_RETRY_EXCEPTIONS + _ANTHROPIC_RETRY_EXCEPTIONS,
    max_tries=MAX_RESEARCH_PROVIDER_RETRIES,
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []
    spec = resolve_model_provider(model)
    response = None
    trace_model = model
    trace_provider = getattr(spec, "provider", "unknown")
    trace_request_style = getattr(spec, "request_style", "unknown")
    trace_params: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": MAX_NUM_TOKENS,
        "seed": 0,
    }
    request_history: list[dict[str, Any]] = []
    _t0 = time.perf_counter()

    if model_uses_anthropic_client(model):
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        request_history = list(new_msg_history)
        response = _budgeted_provider_call(
            model=model,
            prompt=new_msg_history,
            system_message=system_message,
            max_output_tokens=MAX_NUM_TOKENS,
            create=lambda timeout: client.messages.create(
                model=spec.client_model,
                max_tokens=MAX_NUM_TOKENS,
                temperature=temperature,
                system=system_message,
                messages=new_msg_history,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
        # response = make_llm_call(client, model, temperature, system_message=system_message, prompt=new_msg_history)
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif spec.provider == "ollama":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_history = list(new_msg_history)
        response = _budgeted_provider_call(
            model=model,
            prompt=new_msg_history,
            system_message=system_message,
            max_output_tokens=OLLAMA_MAX_NUM_TOKENS,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=OLLAMA_MAX_NUM_TOKENS,
                n=1,
                stop=None,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif spec.request_style == "openai_chat" and spec.provider != "huggingface":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_history = list(new_msg_history)
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        _validate_openai_compat_response(spec, response)
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif spec.request_style == "openai_reasoning":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_history = list(new_msg_history)
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        _validate_openai_compat_response(spec, response)
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif spec.provider == "huggingface":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        request_history = list(new_msg_history)
        try:
            response = _budgeted_provider_call(
                model=model,
                prompt=new_msg_history,
                system_message=system_message,
                max_output_tokens=MAX_NUM_TOKENS,
                create=lambda timeout: client.chat.completions.create(
                    model=spec.client_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        *new_msg_history,
                    ],
                    temperature=temperature,
                    max_tokens=MAX_NUM_TOKENS,
                    n=1,
                    stop=None,
                    **({"timeout": timeout} if timeout is not None else {}),
                ),
            )
            content = response.choices[0].message.content
        except LLMBudgetExceeded:
            raise
        except Exception as exc:
            if not _is_huggingface_protocol_compatibility_error(exc):
                raise
            # Some HuggingFace endpoints do not expose the OpenAI-compatible
            # chat contract.  The compatibility path must call the same model
            # and make the transport switch explicit in the trace.
            fallback_url = _huggingface_http_fallback_url(spec.client_model)
            headers = {
                "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
                "Content-Type": "application/json",
            }
            payload = {
                "inputs": {
                    "system": system_message,
                    "messages": [
                        {"role": item["role"], "content": item["content"]}
                        for item in new_msg_history
                    ],
                },
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": MAX_NUM_TOKENS,
                    "return_full_text": False,
                },
            }
            response = _budgeted_provider_call(
                model=model,
                prompt=new_msg_history,
                system_message=system_message,
                max_output_tokens=MAX_NUM_TOKENS,
                create=lambda timeout: requests.post(
                    fallback_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout or 60,
                ),
            )
            if response.status_code == 200:
                content = _huggingface_generated_text(response)
            else:
                raise ValueError(
                    "HuggingFace inference endpoint returned HTTP "
                    f"{response.status_code}"
                )
            trace_model = spec.client_model
            trace_provider = "huggingface_http"
            trace_request_style = "huggingface_inference"
            trace_params.update(
                {
                    "fallback": True,
                    "fallback_from": "openai_compatible",
                    "fallback_reason": type(exc).__name__,
                    "actual_model": spec.client_model,
                }
            )

        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    _record_llm_call_safe(
        spec=spec,
        model=trace_model,
        provider=trace_provider,
        request_style=trace_request_style,
        system_message=system_message,
        messages=request_history,
        response_text=content,
        params=trace_params,
        response=response,
        latency_ms=int((time.perf_counter() - _t0) * 1000),
        client=client,
    )

    return content, new_msg_history


def _record_llm_call_safe(
    *,
    spec,
    model: str,
    system_message: str,
    messages: list[dict[str, Any]],
    response_text: str,
    params: dict[str, Any],
    response: Any = None,
    latency_ms: int | None = None,
    error: str | None = None,
    provider: str | None = None,
    request_style: str | None = None,
    client: Any = None,
) -> None:
    """Wrap ``record_llm_call`` so a broken tracer never breaks the LLM call.

    Also normalises the various SDK usage shapes (Anthropic's
    ``input_tokens/output_tokens`` vs OpenAI's ``prompt_tokens/
    completion_tokens``) into the single ``{input, output}`` shape the
    protocol schema expects.
    """
    try:
        tokens = _extract_tokens(response)
        client_provenance = getattr(client, "_xscientist_model_provenance", None)
        provenance = (
            dict(client_provenance)
            if isinstance(client_provenance, dict)
            and client_provenance.get("requested_model") == model
            else build_model_provenance(model, env=os.environ)
        )
        reported_model = _safe_reported_model(getattr(response, "model", None))
        if reported_model is not None:
            provenance["reported_model"] = reported_model
            provenance["reported_model_exact"] = reported_model == getattr(
                spec, "client_model", None
            )
        record_llm_call(
            provider=provider or getattr(spec, "provider", "unknown"),
            model=model,
            request_style=request_style or getattr(spec, "request_style", "unknown"),
            system_message=system_message or "",
            messages=messages or [],
            response_text=response_text or "",
            params=params,
            tokens=tokens,
            latency_ms=latency_ms,
            error=error,
            model_provenance=provenance,
        )
    except Exception:
        if strict_llm_tracing():
            raise
        # Exploratory tracing must never bring down the caller.
        return


def _extract_tokens(response: Any) -> dict[str, int] | None:
    """Best-effort usage extraction across SDKs; returns None when nothing found."""
    if response is None:
        return None
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None) or getattr(
        usage, "prompt_tokens", None
    )
    output_tokens = getattr(usage, "output_tokens", None) or getattr(
        usage, "completion_tokens", None
    )
    out: dict[str, int] = {}
    if (
        isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and input_tokens >= 0
    ):
        out["input"] = input_tokens
    if (
        isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens >= 0
    ):
        out["output"] = output_tokens
    return out or None


def extract_json_between_markers(llm_output: str) -> dict | None:
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model) -> tuple[Any, str]:
    spec = resolve_model_provider(model)
    if spec.client_family == "anthropic":
        print(f"Using {spec.display_name} API.")
        import httpx as _httpx

        _http_client = _httpx.Client(
            timeout=_httpx.Timeout(60.0, connect=10.0),
            transport=_httpx.HTTPTransport(retries=3),
            limits=_httpx.Limits(
                max_connections=1,
                max_keepalive_connections=0,
            ),
        )
        return (
            anthropic.Anthropic(
                timeout=60.0,
                max_retries=0,
                http_client=_http_client,
            ),
            model,
        )
    if spec.client_family == "anthropic_bedrock":
        print(f"Using {spec.display_name}.")
        return anthropic.AnthropicBedrock(max_retries=0), model
    if spec.client_family == "anthropic_vertex":
        print(f"Using {spec.display_name}.")
        return anthropic.AnthropicVertex(max_retries=0), model
    if spec.client_family == "openai_compatible":
        provider_env = dict(os.environ)
        kwargs, client_model = build_openai_compatible_client_kwargs(
            model, env=provider_env, max_retries=0
        )
        print(f"Using {spec.display_name} API.")
        # Keep the route-qualified identifier for every later call. Returning
        # only the wire model would lose the provider/endpoint contract (for
        # example openai_compat/glm-5.3 would be misclassified as native
        # Zhipu), bypassing identity checks and corrupting provenance.
        client = openai.OpenAI(**kwargs)
        try:
            client._xscientist_model_provenance = build_model_provenance(
                model,
                env=provider_env,
            )
        except (AttributeError, TypeError):
            pass
        return client, model
    raise ValueError(f"Model {model} not supported.")
