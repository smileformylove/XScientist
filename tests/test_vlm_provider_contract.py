from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ai_scientist import vlm
from ai_scientist.llm import LLMResponseContractError
from ai_scientist.protocol import capture_llm_calls
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_STRICT,
)


class _Completions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _response(
    contents: list[str],
    *,
    model: object = "glm-5.3",
    finish_reason: object = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
            for content in contents
        ],
        usage=SimpleNamespace(
            prompt_tokens=9,
            completion_tokens=3,
            total_tokens=12,
        ),
    )


def _client(response: object) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(response)))


def _enable_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ACTIVE_ROOT, str(tmp_path))
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_STRICT, "1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-test-key")
    monkeypatch.setenv(
        "OPENAI_COMPAT_BASE_URL",
        "https://gateway.example/v1",
    )


def _prepared_prompt(*_args, **_kwargs) -> list[dict]:
    return [
        {"type": "text", "text": "private-vlm-question-canary"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,private-image-bytes-canary"},
        },
    ]


def test_vlm_custom_route_is_exact_bounded_and_digest_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_trace(tmp_path, monkeypatch)
    client = _client(_response(["private-vlm-response-canary"]))

    with (
        mock.patch.object(vlm, "prepare_vlm_prompt", side_effect=_prepared_prompt),
        capture_llm_calls() as refs,
    ):
        content, _ = vlm.get_response_from_vlm(
            "question",
            ["unused.jpg"],
            client,
            "openai_compat/glm-5.3",
            "vision review",
            temperature=0,
        )

    assert content == "private-vlm-response-canary"
    request = client.chat.completions.calls[0]
    assert request["model"] == "glm-5.3"
    assert request["timeout"] == vlm.OPENAI_COMPAT_CALL_TIMEOUT_SECONDS
    [row] = [
        json.loads(line)
        for line in (tmp_path / CALLS_JSONL_RELPATH).read_text().splitlines()
    ]
    assert refs == [row["call_receipt_ref"]["hash"]]
    persisted = b"".join(
        path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    )
    for forbidden in (
        b"private-vlm-question-canary",
        b"private-image-bytes-canary",
        b"private-vlm-response-canary",
    ):
        assert forbidden not in persisted


@pytest.mark.parametrize(
    "reported_model",
    [None, "openai_compat/glm-5.3", "glm-5.3-alias", "sk-" + "V" * 40],
)
def test_vlm_custom_route_rejects_untrusted_model_without_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_model: object,
) -> None:
    _enable_trace(tmp_path, monkeypatch)
    client = _client(_response(["untrusted"], model=reported_model))

    with (
        mock.patch.object(vlm, "prepare_vlm_prompt", side_effect=_prepared_prompt),
        pytest.raises(LLMResponseContractError, match="identity is not exact"),
    ):
        vlm.get_response_from_vlm(
            "question",
            ["unused.jpg"],
            client,
            "openai_compat/glm-5.3",
            "vision review",
        )

    assert not (tmp_path / CALLS_JSONL_RELPATH).exists()


def test_vlm_batch_is_one_provider_receipt_and_one_usage_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_trace(tmp_path, monkeypatch)
    client = _client(_response(["one", "two"]))

    with (
        mock.patch.object(vlm, "prepare_vlm_prompt", side_effect=_prepared_prompt),
        capture_llm_calls() as refs,
    ):
        contents, _ = vlm.get_batch_responses_from_vlm(
            "question",
            ["unused.jpg"],
            client,
            "openai_compat/glm-5.3",
            "vision review",
            n_responses=2,
        )

    assert contents == ["one", "two"]
    rows = [
        json.loads(line)
        for line in (tmp_path / CALLS_JSONL_RELPATH).read_text().splitlines()
    ]
    assert len(rows) == len(refs) == 1
    assert rows[0]["tokens"] == {"input": 9, "output": 3}
    assert rows[0]["params"]["n"] == 2


def test_vlm_create_client_preserves_route_qualified_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-test-key")
    monkeypatch.setenv(
        "OPENAI_COMPAT_BASE_URL",
        "https://gateway.example/v1",
    )
    with mock.patch.object(vlm.openai, "OpenAI", return_value=object()):
        _client_object, model = vlm.create_client("openai_compat/glm-5.3")

    assert model == "openai_compat/glm-5.3"


@pytest.mark.parametrize("count", [0, -1, True, 9])
def test_vlm_batch_count_is_bounded_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    count: object,
) -> None:
    client = _client(_response(["unused"]))
    with (
        mock.patch.object(vlm, "prepare_vlm_prompt", side_effect=_prepared_prompt),
        pytest.raises(ValueError, match="between 1 and 8"),
    ):
        vlm.get_batch_responses_from_vlm(
            "question",
            [],
            client,
            "openai_compat/glm-5.3",
            "vision review",
            n_responses=count,  # type: ignore[arg-type]
        )
    assert client.chat.completions.calls == []
