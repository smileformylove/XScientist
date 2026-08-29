from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.utils.llm_budget import llm_budget_manager
from ai_scientist.errors import LLMResponseContractError
from ai_scientist.utils.optional_dependencies import (
    import_backoff,
    import_optional_module,
    resolve_exception_types,
)
from ai_scientist.utils.provider_registry import (
    build_openai_compatible_client_kwargs,
    model_provenance as build_model_provenance,
    resolve_model_provider,
)
from ai_scientist.llm import (
    MAX_BATCH_RESPONSES,
    MAX_RESEARCH_PROVIDER_RETRIES,
    OPENAI_COMPAT_CALL_TIMEOUT_SECONDS,
    _record_llm_call_safe,
    _validate_openai_compat_response,
)

backoff = import_backoff()
openai = import_optional_module(
    "openai",
    install_hint="Install the 'openai' package to use OpenAI-compatible VLM models.",
    exception_names=("RateLimitError", "APITimeoutError"),
)
Image = import_optional_module(
    "PIL.Image",
    install_hint="Install the 'Pillow' package to use image-based VLM utilities.",
)
_VLM_RETRY_EXCEPTIONS = resolve_exception_types(
    openai,
    ("RateLimitError", "APITimeoutError"),
)

MAX_NUM_TOKENS = 4096

AVAILABLE_VLMS = [
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "o3-mini",
    # Ollama models
    # llama4
    "ollama/llama4:16x17b",
    # mistral
    "ollama/mistral-small3.2:24b",
    # qwen
    "ollama/qwen2.5vl:32b",
    "ollama/z-uo/qwen2.5vl_tools:32b",
    "openai/gpt-4o-2024-11-20",
    "gemini/gemini-2.0-flash",
    "zhipu/glm-4v-flash",
    "openai_compat/vision-model",
]

SUPPORTED_VLM_PROVIDERS = {
    "openai",
    "ollama",
    "openrouter",
    "gemini",
    "zhipu",
    "openai_compat",
}


def _is_supported_vlm_model(model: str) -> bool:
    spec = resolve_model_provider(model)
    return spec.provider in SUPPORTED_VLM_PROVIDERS


def validate_vlm_response(
    spec: Any,
    response: Any,
    *,
    expected_choices: int = 1,
) -> str | None:
    """Validate the complete custom-provider VLM response envelope."""

    reported_model = _validate_openai_compat_response(
        spec,
        response,
        expected_choices=expected_choices,
    )
    if getattr(spec, "provider", None) != "openai_compat":
        return reported_model

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    values = (prompt_tokens, completion_tokens, total_tokens)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        )
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise LLMResponseContractError("Provider response token usage is invalid")
    return reported_model


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image to base64 string."""
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save to bytes
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")


def budgeted_vlm_provider_call(
    *, model, prompt, system_message, create, max_output_tokens=MAX_NUM_TOKENS
):
    reservation = llm_budget_manager.reserve(
        model=model,
        prompt=prompt,
        system_message=system_message,
        max_output_tokens=max_output_tokens,
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


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    spec = resolve_model_provider(model)
    if spec.provider == "ollama":
        return budgeted_vlm_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
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
    elif spec.request_style == "openai_chat":
        return budgeted_vlm_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
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
        return budgeted_vlm_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
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


@track_token_usage
def make_vlm_call(client, model, temperature, system_message, prompt):
    spec = resolve_model_provider(model)
    if spec.provider == "ollama":
        return budgeted_vlm_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *prompt,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
    elif spec.request_style == "openai_chat":
        return budgeted_vlm_provider_call(
            model=model,
            prompt=prompt,
            system_message=system_message,
            create=lambda timeout: client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *prompt,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                **({"timeout": timeout} if timeout is not None else {}),
            ),
        )
    else:
        raise ValueError(f"Model {model} not supported.")


def prepare_vlm_prompt(
    msg: str, image_paths: str | list[str], max_images: int
) -> list[dict[str, Any]]:
    """Build multimodal user content for VLM calls."""
    if max_images < 0:
        raise ValueError("max_images must be >= 0")

    if isinstance(image_paths, str):
        normalized_paths = [image_paths]
    else:
        normalized_paths = list(image_paths)

    content: list[dict[str, Any]] = [{"type": "text", "text": msg}]
    for image_path in normalized_paths[:max_images]:
        if not isinstance(image_path, str):
            raise TypeError("image_paths must contain strings")
        base64_image = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low",
                },
            }
        )

    return content


@backoff.on_exception(
    backoff.expo,
    _VLM_RETRY_EXCEPTIONS,
    max_tries=MAX_RESEARCH_PROVIDER_RETRIES,
)
def get_response_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_images: int = 25,
) -> tuple[str, list[dict[str, Any]]]:
    """Get response from vision-language model."""
    if msg_history is None:
        msg_history = []
    spec = resolve_model_provider(model)

    if _is_supported_vlm_model(model):
        content = prepare_vlm_prompt(msg, image_paths, max_images)
        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]
        request_history = list(new_msg_history)

        response = make_vlm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )

        validate_vlm_response(spec, response)
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    _record_llm_call_safe(
        spec=spec,
        model=model,
        system_message=system_message,
        messages=request_history,
        response_text=content,
        params={
            "temperature": temperature,
            "max_tokens": MAX_NUM_TOKENS,
        },
        response=response,
        client=client,
    )
    return content, new_msg_history


def create_client(model: str) -> tuple[Any, str]:
    """Create client for vision-language model."""
    if not _is_supported_vlm_model(model):
        raise ValueError(f"Model {model} not supported by VLM client.")
    spec = resolve_model_provider(model)
    provider_env = dict(os.environ)
    kwargs, client_model = build_openai_compatible_client_kwargs(
        model, env=provider_env, max_retries=0
    )
    print(f"Using {spec.display_name} API.")
    client = openai.OpenAI(**kwargs)
    try:
        client._xscientist_model_provenance = build_model_provenance(
            model,
            env=provider_env,
        )
    except (AttributeError, TypeError):
        pass
    return client, model


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


@backoff.on_exception(
    backoff.expo,
    _VLM_RETRY_EXCEPTIONS,
    max_tries=MAX_RESEARCH_PROVIDER_RETRIES,
)
def get_batch_responses_from_vlm(
    msg: str,
    image_paths: str | list[str],
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    n_responses: int = 1,
    max_images: int = 200,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    """Get multiple responses from vision-language model for the same input.

    Args:
        msg: Text message to send
        image_paths: Path(s) to image file(s)
        client: OpenAI client instance
        model: Name of model to use
        system_message: System prompt
        print_debug: Whether to print debug info
        msg_history: Previous message history
        temperature: Sampling temperature
        n_responses: Number of responses to generate

    Returns:
        Tuple of (list of response strings, list of message histories)
    """
    if msg_history is None:
        msg_history = []
    spec = resolve_model_provider(model)

    if _is_supported_vlm_model(model):
        if (
            isinstance(n_responses, bool)
            or not isinstance(n_responses, int)
            or not 1 <= n_responses <= MAX_BATCH_RESPONSES
        ):
            raise ValueError("n_responses must be an integer between 1 and 8")

        content = prepare_vlm_prompt(msg, image_paths, max_images)

        # Construct message with all images
        new_msg_history = msg_history + [{"role": "user", "content": content}]

        reservation = llm_budget_manager.reserve(
            model=model,
            prompt=new_msg_history,
            system_message=system_message,
            max_output_tokens=MAX_NUM_TOKENS,
            output_multiplier=n_responses,
        )
        timeout_seconds = reservation.timeout_seconds
        if spec.provider == "openai_compat":
            timeout_seconds = min(
                float(timeout_seconds or OPENAI_COMPAT_CALL_TIMEOUT_SECONDS),
                OPENAI_COMPAT_CALL_TIMEOUT_SECONDS,
            )
        with reservation:
            response = client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=n_responses,
                seed=0,
                **({"timeout": timeout_seconds} if timeout_seconds is not None else {}),
            )
            reservation.settle(response=response)

        # Extract content from all responses
        validate_vlm_response(
            spec,
            response,
            expected_choices=n_responses,
        )
        contents = [r.message.content for r in response.choices]
        new_msg_histories = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in contents
        ]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        # Just print the first response
        print()
        print("*" * 20 + " VLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_histories[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(contents[0])
        print("*" * 21 + " VLM END " + "*" * 21)
        print()

    _record_llm_call_safe(
        spec=spec,
        model=model,
        system_message=system_message,
        messages=new_msg_history,
        response_text=json.dumps(contents, ensure_ascii=False),
        params={
            "temperature": temperature,
            "max_tokens": MAX_NUM_TOKENS,
            "n": n_responses,
        },
        response=response,
        client=client,
    )
    return contents, new_msg_histories
