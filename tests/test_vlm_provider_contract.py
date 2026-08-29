from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ai_scientist import vlm
from ai_scientist import perform_vlm_review as vlm_review
from ai_scientist.llm import LLMResponseContractError
from ai_scientist.protocol import capture_llm_calls
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_STRICT,
)
from ai_scientist.utils.llm_budget import LLMBudgetManager


@pytest.fixture(autouse=True)
def _isolated_vlm_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider contract tests must not inherit another run's global ledger."""

    monkeypatch.setattr(vlm, "llm_budget_manager", LLMBudgetManager())


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
    assert vlm.OPENAI_COMPAT_CALL_TIMEOUT_SECONDS == 300.0
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


def _stub_duplicate_figure_inputs(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    monkeypatch.setattr(vlm_review, "load_paper", lambda _path: "paper")
    monkeypatch.setattr(
        vlm_review,
        "extract_figure_screenshots",
        lambda *_args, **_kwargs: [
            {
                "img_name": "figure_1",
                "caption": "Figure 1.",
                "images": [b"image-bytes"],
                "main_text_figrefs": [],
            }
        ],
    )
    prepare = mock.Mock(side_effect=_prepared_prompt)
    monkeypatch.setattr(vlm, "prepare_vlm_prompt", prepare)
    return prepare


def test_duplicate_figure_review_uses_wire_model_and_route_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = _stub_duplicate_figure_inputs(monkeypatch)
    _enable_trace(tmp_path, monkeypatch)
    budget = LLMBudgetManager()
    budget.configure(max_wall_time_seconds=5)
    monkeypatch.setattr(vlm, "llm_budget_manager", budget)
    client = _client(_response(["No duplicates found"]))

    with capture_llm_calls() as refs:
        result = vlm_review.detect_duplicate_figures(
            client,
            "openai_compat/glm-5.3",
            str(tmp_path / "paper.pdf"),
        )

    assert result == "No duplicates found"
    request = client.chat.completions.calls[0]
    assert request["model"] == "glm-5.3"
    assert 0 < request["timeout"] <= 5
    assert prepare.call_args.args[2] == 25
    snapshot = budget.snapshot()
    assert set(snapshot["per_model"]) == {"openai_compat/glm-5.3"}
    rows = [
        json.loads(line)
        for line in (tmp_path / CALLS_JSONL_RELPATH).read_text().splitlines()
    ]
    assert len(rows) == len(refs) == 1
    assert rows[0]["model"] == "openai_compat/glm-5.3"
    assert rows[0]["model_provenance"]["reported_model"] == "glm-5.3"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response(["analysis"], model="glm-5.3-alias"), "identity is not exact"),
        (_response([], model="glm-5.3"), "unexpected number of choices"),
        (
            _response(["analysis"], model="glm-5.3", finish_reason="length"),
            "did not terminate normally",
        ),
        (_response(["  "], model="glm-5.3"), "content is not valid text"),
    ],
)
def test_duplicate_figure_review_rejects_invalid_provider_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: object,
    message: str,
) -> None:
    _stub_duplicate_figure_inputs(monkeypatch)
    client = _client(response)

    with pytest.raises(LLMResponseContractError, match=message):
        vlm_review.detect_duplicate_figures(
            client,
            "openai_compat/glm-5.3",
            str(tmp_path / "paper.pdf"),
        )


def test_duplicate_figure_review_rejects_invalid_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_duplicate_figure_inputs(monkeypatch)
    response = _response(["analysis"])
    response.usage.total_tokens = 999

    with pytest.raises(LLMResponseContractError, match="token usage is invalid"):
        vlm_review.detect_duplicate_figures(
            _client(response),
            "openai_compat/glm-5.3",
            str(tmp_path / "paper.pdf"),
        )


def test_duplicate_figure_provider_error_is_sanitized_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_duplicate_figure_inputs(monkeypatch)
    secret = "provider-exception-secret-canary"
    create = mock.Mock(side_effect=RuntimeError(secret))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(
        LLMResponseContractError,
        match="Duplicate-figure VLM review failed",
    ) as exc_info:
        vlm_review.detect_duplicate_figures(
            client,
            "openai_compat/glm-5.3",
            str(tmp_path / "paper.pdf"),
        )

    captured = capsys.readouterr()
    rendered = caplog.text + captured.out + captured.err + str(exc_info.value)
    assert secret not in rendered


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
