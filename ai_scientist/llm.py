from __future__ import annotations

import json
import os
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage
from ai_scientist.utils.optional_dependencies import (
    import_backoff,
    import_optional_module,
    resolve_exception_types,
)
from ai_scientist.utils.provider_registry import (
    build_openai_compatible_client_kwargs,
    model_uses_anthropic_client,
    resolve_model_provider,
)

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


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    _OPENAI_RETRY_EXCEPTIONS + _ANTHROPIC_RETRY_EXCEPTIONS,
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
    msg = prompt
    if msg_history is None:
        msg_history = []
    spec = resolve_model_provider(model)

    if spec.provider == "ollama":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=spec.client_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=OLLAMA_MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif spec.request_style == "openai_chat":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
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

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    spec = resolve_model_provider(model)
    if spec.provider == "ollama":
        return client.chat.completions.create(
            model=spec.client_model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=OLLAMA_MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
    elif spec.request_style == "openai_chat":
        return client.chat.completions.create(
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
        )
    elif spec.request_style == "openai_reasoning":
        return client.chat.completions.create(
            model=spec.client_model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    _OPENAI_RETRY_EXCEPTIONS + _ANTHROPIC_RETRY_EXCEPTIONS,
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
        response = client.messages.create(
            model=spec.client_model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
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
        response = client.chat.completions.create(
            model=spec.client_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=OLLAMA_MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif spec.request_style == "openai_chat" and spec.provider != "huggingface":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif spec.request_style == "openai_reasoning":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif spec.provider == "huggingface":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        try:
            response = client.chat.completions.create(
                model=spec.client_model,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
        except Exception as e:
            # Fallback to direct API call if OpenAI client doesn't work with HuggingFace
            headers = {
                "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": {
                    "system": system_message,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in new_msg_history]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": MAX_NUM_TOKENS,
                    "return_full_text": False
                }
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                content = response.json()["generated_text"]
            else:
                raise ValueError(f"Error from HuggingFace API: {response.text}")

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

    return content, new_msg_history


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
        print(f"Using {spec.display_name} API with model {spec.client_model}.")
        return anthropic.Anthropic(), model
    if spec.client_family == "anthropic_bedrock":
        print(f"Using {spec.display_name} with model {spec.client_model}.")
        return anthropic.AnthropicBedrock(), model
    if spec.client_family == "anthropic_vertex":
        print(f"Using {spec.display_name} with model {spec.client_model}.")
        return anthropic.AnthropicVertex(), model
    if spec.client_family == "openai_compatible":
        kwargs, client_model = build_openai_compatible_client_kwargs(model, env=os.environ)
        print(f"Using {spec.display_name} API with model {client_model}.")
        return openai.OpenAI(**kwargs), client_model
    raise ValueError(f"Model {model} not supported.")
