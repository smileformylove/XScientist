from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ai_scientist.treesearch.backend import (
    FunctionCallValidationError,
    ResearchDecisionError,
    backend_anthropic,
    backend_openai,
    backend_zhipu,
    query as backend_query,
)
from ai_scientist.treesearch.backend.utils import (
    BACKOFF_MAX_TRIES,
    FunctionSpec,
    MAX_FUNCTION_CALL_ARGUMENT_BYTES,
    backoff_create,
    compile_prompt_to_md,
    summarize_messages_for_log,
    summarize_request_kwargs_for_log,
    validate_function_call_payload,
)
from ai_scientist.protocol import ObjectStore, capture_llm_calls
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_STRICT,
)
from ai_scientist.utils.llm_budget import LLMBudgetManager
from ai_scientist.utils.provider_registry import (
    OPENAI_COMPAT_CALL_TIMEOUT_SECONDS,
)

RESEARCH_DECISION_SPEC = FunctionSpec(
    name="choose_next_experiment",
    description="Choose the next discriminating experiment",
    json_schema={
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "expected_information_gain": {"type": "number"},
        },
        "required": ["hypothesis", "expected_information_gain"],
        "additionalProperties": False,
    },
)


def _tool_call(
    *,
    name: str = RESEARCH_DECISION_SPEC.name,
    arguments: object = '{"hypothesis":"h1","expected_information_gain":0.8}',
) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _openai_response(
    tool_calls: object,
    *,
    content: str | None = None,
    finish_reason: str = "tool_calls",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        system_fingerprint=None,
        model="provider-model",
        created=1,
    )


def _immediate_backoff(create, _retry_exceptions, *args, **kwargs):
    request_kwargs = {
        key: value for key, value in kwargs.items() if not key.startswith("_budget_")
    }
    return create(*args, **request_kwargs)


def _zhipu_budget() -> SimpleNamespace:
    class Reservation:
        timeout_seconds = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def settle(self, response):
            return None

    return SimpleNamespace(reserve=lambda **kwargs: Reservation())


def _query_openai(response: SimpleNamespace) -> dict:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock.Mock(return_value=response))
        )
    )
    with (
        mock.patch.object(backend_openai, "get_ai_client", return_value=client),
        mock.patch.object(
            backend_openai, "backoff_create", side_effect=_immediate_backoff
        ),
    ):
        output, *_ = backend_openai.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model="openai/gpt-4o-mini",
            max_tokens=128,
        )
    return output


def _query_zhipu(response: SimpleNamespace) -> dict:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock.Mock(return_value=response))
        )
    )
    with (
        mock.patch.object(backend_zhipu, "get_ai_client", return_value=client),
        mock.patch.object(backend_zhipu, "llm_budget_manager", _zhipu_budget()),
    ):
        output, *_ = backend_zhipu.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model="zhipu/glm-4-plus",
            max_tokens=128,
        )
    return output


def _query_anthropic(content: list[SimpleNamespace]) -> dict:
    message = SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=13, output_tokens=5),
        stop_reason="tool_use",
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=mock.Mock(return_value=message))
    )
    with (
        mock.patch.object(backend_anthropic, "get_ai_client", return_value=client),
        mock.patch.object(
            backend_anthropic, "backoff_create", side_effect=_immediate_backoff
        ),
    ):
        output, *_ = backend_anthropic.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model="anthropic/claude-test",
            max_tokens=128,
        )
    return output


def test_openai_validates_and_returns_one_schema_bound_tool_call() -> None:
    output = _query_openai(_openai_response([_tool_call()]))

    assert output == {"hypothesis": "h1", "expected_information_gain": 0.8}


@pytest.mark.parametrize(
    "tool_calls",
    [
        None,
        [],
        [_tool_call(name="unrequested_tool")],
        [_tool_call(arguments='{"hypothesis":"h1"}')],
        [_tool_call(arguments="not-json-secret-canary")],
        [_tool_call(arguments="[]")],
        [
            _tool_call(
                arguments=(
                    '{"hypothesis":"h1","hypothesis":"h2",'
                    '"expected_information_gain":0.8}'
                )
            )
        ],
        [_tool_call(arguments=('{"hypothesis":"h1","expected_information_gain":NaN}'))],
        [
            _tool_call(
                arguments=('{"hypothesis":"h1","expected_information_gain":1e999}')
            )
        ],
        [_tool_call(), _tool_call()],
    ],
)
def test_openai_tool_contract_violations_fail_closed(tool_calls: object) -> None:
    with pytest.raises(FunctionCallValidationError):
        _query_openai(_openai_response(tool_calls))


def test_zhipu_plain_text_json_cannot_become_a_research_decision() -> None:
    secret_response = '{"hypothesis":"secret-canary","expected_information_gain":1.0}'

    with pytest.raises(FunctionCallValidationError) as exc_info:
        _query_zhipu(_openai_response(None, content=secret_response))

    assert "secret-canary" not in str(exc_info.value)


@pytest.mark.parametrize(
    "tool_calls",
    [
        [_tool_call(name="unrequested_tool")],
        [_tool_call(arguments='{"hypothesis":"h1"}')],
    ],
)
def test_zhipu_uses_the_same_function_contract_as_openai(tool_calls: object) -> None:
    with pytest.raises(FunctionCallValidationError):
        _query_zhipu(_openai_response(tool_calls))


def test_zhipu_accepts_a_valid_schema_bound_tool_call() -> None:
    output = _query_zhipu(_openai_response([_tool_call()]))

    assert output["hypothesis"] == "h1"


def test_anthropic_checks_tool_name_and_json_schema() -> None:
    wrong_name = SimpleNamespace(
        type="tool_use",
        name="unrequested_tool",
        input={"hypothesis": "h1", "expected_information_gain": 0.8},
    )
    missing_required = SimpleNamespace(
        type="tool_use",
        name=RESEARCH_DECISION_SPEC.name,
        input={"hypothesis": "h1"},
    )

    with pytest.raises(FunctionCallValidationError):
        _query_anthropic([wrong_name])
    with pytest.raises(FunctionCallValidationError):
        _query_anthropic([missing_required])


def test_anthropic_accepts_a_valid_decoded_mapping() -> None:
    tool_use = SimpleNamespace(
        type="tool_use",
        name=RESEARCH_DECISION_SPEC.name,
        input={"hypothesis": "h1", "expected_information_gain": 0.8},
    )

    output = _query_anthropic([tool_use])

    assert output["expected_information_gain"] == 0.8


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_openai_compatible_truncated_tool_decisions_fail_closed(
    finish_reason: str | None,
) -> None:
    with pytest.raises(ResearchDecisionError, match="terminate"):
        _query_openai(_openai_response([_tool_call()], finish_reason=finish_reason))


def test_anthropic_truncated_tool_decision_fails_closed() -> None:
    tool_use = SimpleNamespace(
        type="tool_use",
        name=RESEARCH_DECISION_SPEC.name,
        input={"hypothesis": "h1", "expected_information_gain": 0.8},
    )
    message = SimpleNamespace(
        content=[tool_use],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="max_tokens",
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=mock.Mock(return_value=message))
    )
    with (
        mock.patch.object(backend_anthropic, "get_ai_client", return_value=client),
        mock.patch.object(
            backend_anthropic, "backoff_create", side_effect=_immediate_backoff
        ),
        pytest.raises(ResearchDecisionError, match="terminate"),
    ):
        backend_anthropic.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model="anthropic/claude-test",
            max_tokens=128,
        )


def test_backend_log_summaries_do_not_copy_prompt_or_secret_values() -> None:
    secret = "prompt-secret-canary"
    summary = {
        "messages": summarize_messages_for_log([{"role": "user", "content": secret}]),
        "request": summarize_request_kwargs_for_log(
            {
                "model": "openai/gpt-4o-mini",
                "system": secret,
                "headers": {"Authorization": secret},
                "messages": [{"role": "user", "content": secret}],
            }
        ),
    }

    assert secret not in json.dumps(summary, sort_keys=True)

    credential_shaped_model = "sk-" + "m" * 32
    model_summary = summarize_request_kwargs_for_log({"model": credential_shaped_model})
    assert credential_shaped_model not in json.dumps(model_summary, sort_keys=True)

    metadata_canary = "sk-" + "Z" * 40
    metadata_summary = {
        "messages": summarize_messages_for_log(
            [
                {
                    "role": metadata_canary,
                    "content": [{"type": metadata_canary, "text": "safe"}],
                }
            ]
        ),
        "request": summarize_request_kwargs_for_log(
            {"tools": [{"function": {"name": metadata_canary, "parameters": {}}}]}
        ),
    }
    assert metadata_canary not in json.dumps(metadata_summary, sort_keys=True)


def test_prompt_compilation_errors_do_not_log_prompt_values(caplog) -> None:
    secret = "prompt-error-secret-canary"

    class SensitiveItem:
        def strip(self):
            raise RuntimeError(secret)

    with pytest.raises(RuntimeError, match=secret):
        compile_prompt_to_md([SensitiveItem()])

    assert secret not in caplog.text


def test_function_specs_require_explicitly_closed_object_fields_recursively() -> None:
    with pytest.raises(ValueError, match="additionalProperties"):
        FunctionSpec(
            name="open_decision",
            description="Ambiguous scientific decision",
            json_schema={"type": "object", "properties": {}},
        )

    spec = FunctionSpec(
        name="nested_decision",
        description="Nested scientific decision",
        json_schema={
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {"effect": {"type": "number"}},
                    "required": ["effect"],
                    "additionalProperties": False,
                }
            },
            "required": ["result"],
            "additionalProperties": False,
        },
    )

    assert spec.json_schema["additionalProperties"] is False
    assert spec.json_schema["properties"]["result"]["additionalProperties"] is False
    with pytest.raises(FunctionCallValidationError):
        validate_function_call_payload(
            spec,
            function_name="nested_decision",
            arguments='{"result":{"effect":0.1,"unrequested":"value"}}',
        )


def test_function_spec_lint_does_not_rewrite_schema_data_or_combinators() -> None:
    schema = {
        "type": "object",
        "definitions": {
            "effect": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "choice": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "a"}},
                        "required": ["kind"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"kind": {"const": "b"}},
                        "required": ["kind"],
                        "additionalProperties": False,
                    },
                ],
                "default": {"type": "object", "opaque": "data"},
                "examples": [{"type": "object", "opaque": "example"}],
            },
            "effect": {
                "allOf": [
                    {"$ref": "#/definitions/effect"},
                    {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": True,
                    },
                ]
            },
        },
        "required": ["choice", "effect"],
        "additionalProperties": False,
    }

    spec = FunctionSpec(
        name="composed_decision",
        description="Schema semantics remain unchanged",
        json_schema=schema,
    )

    assert spec.json_schema == schema
    assert validate_function_call_payload(
        spec,
        function_name="composed_decision",
        arguments='{"choice":{"kind":"a"},"effect":{"value":0.1}}',
    ) == {"choice": {"kind": "a"}, "effect": {"value": 0.1}}


def test_function_spec_lint_closes_required_only_branches_but_allows_typed_maps() -> (
    None
):
    with pytest.raises(ValueError, match="additionalProperties"):
        FunctionSpec(
            name="required_only_branch",
            description="Every object-bearing branch must choose its openness",
            json_schema={
                "type": "object",
                "properties": {
                    "choice": {
                        "oneOf": [
                            {"required": ["a"]},
                            {"required": ["b"], "additionalProperties": False},
                        ]
                    }
                },
                "required": ["choice"],
                "additionalProperties": False,
            },
        )

    schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }
    spec = FunctionSpec(
        name="typed_map",
        description="A deliberately open typed map",
        json_schema=schema,
    )
    assert spec.json_schema == schema
    assert validate_function_call_payload(
        spec,
        function_name="typed_map",
        arguments='{"scores":{"dataset-a":0.8}}',
    ) == {"scores": {"dataset-a": 0.8}}


def test_function_spec_enforces_declared_2020_12_dependent_constraints() -> None:
    spec = FunctionSpec(
        name="paired_decision",
        description="A decision whose confidence requires evidence",
        json_schema={
            "type": "object",
            "properties": {
                "confidence": {"type": "number"},
                "evidence": {"type": "string", "minLength": 1},
            },
            "dependentRequired": {"confidence": ["evidence"]},
            "additionalProperties": False,
        },
    )

    with pytest.raises(FunctionCallValidationError):
        validate_function_call_payload(
            spec,
            function_name=spec.name,
            arguments='{"confidence":0.9}',
        )
    assert (
        validate_function_call_payload(
            spec,
            function_name=spec.name,
            arguments='{"confidence":0.9,"evidence":"replicated"}',
        )["evidence"]
        == "replicated"
    )


def test_backend_rejects_legacy_dict_function_specs_before_provider_use() -> None:
    with pytest.raises(TypeError, match="FunctionSpec"):
        backend_query(
            "system",
            None,
            model="openai/gpt-4o-mini",
            func_spec={"name": "legacy"},  # type: ignore[arg-type]
        )


def test_central_backend_verifies_and_traces_exact_custom_model_digest_only(
    tmp_path: Path, monkeypatch
) -> None:
    from ai_scientist.treesearch import backend as backend_module

    secret_key = "sk-" + "K" * 40
    endpoint = "https://private-gateway.example/v1"
    prompt = "private-research-prompt-canary"
    response = "private-research-response-canary"
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", secret_key)
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", endpoint)
    monkeypatch.setenv(ENV_ACTIVE_ROOT, str(tmp_path))
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_STRICT, "1")
    adapter = SimpleNamespace(
        query=mock.Mock(return_value=(response, 0.125, 17, 5, {"model": "glm-5.3"}))
    )

    with (
        mock.patch.object(
            backend_module, "_resolve_backend_module", return_value=adapter
        ),
        capture_llm_calls() as refs,
    ):
        output = backend_module.query(
            system_message=prompt,
            user_message=None,
            model="openai_compat/glm-5.3",
            temperature=0,
            max_tokens=128,
        )

    assert output == response
    [row] = [
        json.loads(line)
        for line in (tmp_path / CALLS_JSONL_RELPATH)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert row["tokens"] == {"input": 17, "output": 5}
    assert row["latency_ms"] == 125
    assert row["model_provenance"]["reported_model"] == "glm-5.3"
    assert row["model_provenance"]["reported_model_exact"] is True
    assert refs == [row["call_receipt_ref"]["hash"]]
    assert (
        adapter.query.call_args.kwargs["timeout"]
        == OPENAI_COMPAT_CALL_TIMEOUT_SECONDS
        == 300.0
    )

    store = ObjectStore(tmp_path)
    message_digest = store.get_json(row["messages_ref"]["hash"])
    response_digest = store.get_json(row["response_ref"]["hash"])
    receipt = store.get_json(row["call_receipt_ref"]["hash"])
    assert message_digest["payload_recorded"] is False
    assert response_digest["payload_recorded"] is False
    assert receipt["response_sha256"] == response_digest["sha256"]

    persisted = b"".join(
        path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    )
    for forbidden in (prompt, response, secret_key, endpoint):
        assert forbidden.encode("utf-8") not in persisted


def test_backend_budget_timeout_shortens_existing_provider_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_scientist.treesearch.backend import utils as backend_utils

    manager = LLMBudgetManager()
    manager.configure(max_wall_time_seconds=4)
    seen: dict[str, object] = {}
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1)
    )

    def create(**kwargs):
        seen.update(kwargs)
        return response

    monkeypatch.setattr(backend_utils, "llm_budget_manager", manager)
    result = backoff_create(
        create,
        (),
        _budget_model="openai_compat/glm-5.3",
        _budget_prompt={"task": "bounded"},
        _budget_max_output_tokens=64,
        timeout=OPENAI_COMPAT_CALL_TIMEOUT_SECONDS,
    )

    assert result is response
    assert 0 < seen["timeout"] <= 4


@pytest.mark.parametrize(
    "reported_model",
    [None, "openai_compat/glm-5.3", "glm-5.3-alias", "sk-" + "S" * 40],
)
def test_central_backend_rejects_untrusted_custom_model_identity_without_trace(
    tmp_path: Path, monkeypatch, reported_model
) -> None:
    from ai_scientist.treesearch import backend as backend_module

    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://private-gateway.example/v1")
    monkeypatch.setenv(ENV_ACTIVE_ROOT, str(tmp_path))
    monkeypatch.setenv(ENV_ENABLED, "1")
    adapter = SimpleNamespace(
        query=mock.Mock(
            return_value=("untrusted-output", 0.1, 1, 1, {"model": reported_model})
        )
    )

    with (
        mock.patch.object(
            backend_module, "_resolve_backend_module", return_value=adapter
        ),
        pytest.raises(
            ResearchDecisionError, match="Provider-reported model identity is not exact"
        ) as exc_info,
    ):
        backend_module.query(
            system_message="research-prompt",
            user_message=None,
            model="openai_compat/glm-5.3",
        )

    assert str(reported_model) not in str(exc_info.value)
    assert not (tmp_path / CALLS_JSONL_RELPATH).exists()


def test_provider_retry_loop_is_finite() -> None:
    attempts = 0

    def return_false() -> bool:
        nonlocal attempts
        attempts += 1
        return False

    with mock.patch("backoff._sync.time.sleep"):
        assert backoff_create(return_false, ()) is False

    assert attempts == BACKOFF_MAX_TRIES


def test_provider_failure_is_sanitized_and_fails_closed(caplog) -> None:
    secret = "provider-secret-canary"
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock.Mock()))
    )
    with (
        mock.patch.object(backend_openai, "get_ai_client", return_value=client),
        mock.patch.object(
            backend_openai, "backoff_create", side_effect=RuntimeError(secret)
        ),
        pytest.raises(
            ResearchDecisionError, match="Provider request failed"
        ) as exc_info,
    ):
        backend_openai.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model="openai/gpt-4o-mini",
            max_tokens=128,
        )

    rendered = caplog.text + str(exc_info.value)
    assert secret not in rendered


@pytest.mark.parametrize(
    ("backend_module", "model"),
    [
        (backend_openai, "openai/gpt-4o-mini"),
        (backend_anthropic, "anthropic/claude-test"),
        (backend_zhipu, "zhipu/glm-4-plus"),
    ],
)
def test_provider_client_initialization_errors_are_sanitized(
    backend_module, model, caplog
) -> None:
    secret = "client-init-secret-canary"
    with (
        mock.patch.object(
            backend_module, "get_ai_client", side_effect=RuntimeError(secret)
        ),
        pytest.raises(
            ResearchDecisionError, match="client initialization failed"
        ) as exc_info,
    ):
        backend_module.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model=model,
            max_tokens=128,
        )

    assert secret not in caplog.text + str(exc_info.value)


def test_exhausted_provider_retries_fail_before_response_parsing() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=mock.Mock()))
    )
    with (
        mock.patch.object(backend_openai, "get_ai_client", return_value=client),
        mock.patch.object(backend_openai, "backoff_create", return_value=False),
        pytest.raises(ResearchDecisionError, match="bounded retries"),
    ):
        backend_openai.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model="openai/gpt-4o-mini",
            max_tokens=128,
        )


def test_malformed_provider_envelope_is_sanitized(caplog) -> None:
    secret = "malformed-envelope-secret-canary"
    malformed = SimpleNamespace(error_detail=secret)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock.Mock(return_value=malformed))
        )
    )
    with (
        mock.patch.object(backend_openai, "get_ai_client", return_value=client),
        mock.patch.object(
            backend_openai, "backoff_create", side_effect=_immediate_backoff
        ),
        pytest.raises(ResearchDecisionError, match="envelope") as exc_info,
    ):
        backend_openai.query(
            "system",
            "user",
            func_spec=RESEARCH_DECISION_SPEC,
            model="openai/gpt-4o-mini",
            max_tokens=128,
        )

    assert secret not in caplog.text + str(exc_info.value)


def test_agent_manager_does_not_replace_failed_research_decision(caplog) -> None:
    from ai_scientist.treesearch import agent_manager as manager_module

    manager = manager_module.AgentManager.__new__(manager_module.AgentManager)
    manager.cfg = SimpleNamespace(
        agent=SimpleNamespace(
            feedback=SimpleNamespace(model="openai/gpt-4o-mini", temp=0)
        )
    )
    secret = "decision-error-secret-canary"

    with (
        mock.patch.object(manager_module, "query", side_effect=RuntimeError(secret)),
        pytest.raises(ResearchDecisionError, match="stage configuration") as exc_info,
    ):
        manager._get_response("private-research-prompt")

    rendered = caplog.text + str(exc_info.value)
    assert secret not in rendered
    assert "private-research-prompt" not in rendered


def test_stage_research_decisions_are_path_safe_bounded_and_coherent() -> None:
    from ai_scientist.treesearch import agent_manager as manager_module

    with pytest.raises(FunctionCallValidationError):
        validate_function_call_payload(
            manager_module.stage_config_spec,
            function_name="generate_stage_config",
            arguments=json.dumps(
                {
                    "name": "../../escape",
                    "description": "unsafe",
                    "goals": ["goal"],
                    "max_iterations": 1,
                }
            ),
        )
    with pytest.raises(FunctionCallValidationError):
        validate_function_call_payload(
            manager_module.stage_config_spec,
            function_name="generate_stage_config",
            arguments=json.dumps(
                {
                    "name": "safe-stage",
                    "description": "bounded",
                    "goals": ["goal"],
                    "max_iterations": manager_module.MAX_AGENT_STAGE_ITERATIONS + 1,
                }
            ),
        )
    with pytest.raises(FunctionCallValidationError, match="conflicts"):
        manager_module._validate_stage_completion_evaluation(
            {
                "is_complete": True,
                "reasoning": "contradictory",
                "missing_criteria": ["still missing"],
            }
        )
    with pytest.raises(ResearchDecisionError, match="declined"):
        manager_module._require_stage_progression({"ready_for_next_stage": False})
    with pytest.raises(ValueError, match="sub-stage"):
        manager_module._parse_stage_name("1_initial_implementation_2_../../escape")


def test_vlm_evidence_binding_follows_selected_plot_order(tmp_path) -> None:
    from ai_scientist.treesearch import parallel_agent as parallel_module
    from ai_scientist.treesearch.journal import Node

    plot_paths = []
    for index in range(11):
        path = tmp_path / f"plot-{index}.png"
        path.write_bytes(b"image")
        plot_paths.append(str(path))
    selected = [plot_paths[7], plot_paths[2]]
    cfg = SimpleNamespace(
        log_dir=str(tmp_path),
        agent=SimpleNamespace(
            feedback=SimpleNamespace(model="openai/gpt-4o-mini", temp=0),
            vlm_feedback=SimpleNamespace(model="openai/gpt-4o-mini", temp=0),
        ),
    )
    agent = parallel_module.MinimalAgent("task", cfg)
    node = Node(
        plot_paths=plot_paths,
        datasets_successfully_tested=["already-bound"],
    )
    responses = [
        {"selected_plots": selected},
        {
            "plot_analyses": [{"analysis": "seven"}, {"analysis": "two"}],
            "valid_plots_received": True,
            "vlm_feedback_summary": "ordered",
        },
    ]

    with mock.patch.object(parallel_module, "query", side_effect=responses):
        agent._analyze_plots_with_vlm(node)

    assert [row["plot_path"] for row in node.plot_analyses] == selected


def test_vlm_evidence_binding_rejects_analysis_count_mismatch(tmp_path) -> None:
    from ai_scientist.treesearch import parallel_agent as parallel_module
    from ai_scientist.treesearch.journal import Node

    paths = []
    for index in range(2):
        path = tmp_path / f"plot-{index}.png"
        path.write_bytes(b"image")
        paths.append(str(path))
    cfg = SimpleNamespace(
        log_dir=str(tmp_path),
        agent=SimpleNamespace(
            vlm_feedback=SimpleNamespace(model="openai/gpt-4o-mini", temp=0)
        ),
    )
    agent = parallel_module.MinimalAgent("task", cfg)
    node = Node(plot_paths=paths, datasets_successfully_tested=["already-bound"])
    response = {
        "plot_analyses": [{"analysis": "only one"}],
        "valid_plots_received": True,
        "vlm_feedback_summary": "incomplete",
    }

    with (
        mock.patch.object(parallel_module, "query", return_value=response),
        pytest.raises(FunctionCallValidationError, match="evidence set"),
    ):
        agent._analyze_plots_with_vlm(node)


def test_tool_arguments_reject_nonfinite_mapping_and_oversized_json() -> None:
    with pytest.raises(FunctionCallValidationError):
        validate_function_call_payload(
            RESEARCH_DECISION_SPEC,
            function_name=RESEARCH_DECISION_SPEC.name,
            arguments={
                "hypothesis": "h1",
                "expected_information_gain": float("inf"),
            },
        )
    with pytest.raises(FunctionCallValidationError, match="size"):
        validate_function_call_payload(
            RESEARCH_DECISION_SPEC,
            function_name=RESEARCH_DECISION_SPEC.name,
            arguments="{" + " " * MAX_FUNCTION_CALL_ARGUMENT_BYTES + "}",
        )


@pytest.mark.parametrize(
    "malformed",
    [
        "plain prose with no code fence",
        "plan\n```python\nx = 1\n```\n```python\ny = 2\n```",
        "",
    ],
)
@pytest.mark.parametrize("agent_class_name", ["MinimalAgent", "ParallelAgent"])
def test_plan_and_code_contract_never_executes_malformed_model_text(
    malformed: str, agent_class_name: str
) -> None:
    from ai_scientist.treesearch import parallel_agent as parallel_module

    agent_class = getattr(parallel_module, agent_class_name)
    fake_agent = SimpleNamespace(
        cfg=SimpleNamespace(
            agent=SimpleNamespace(
                code=SimpleNamespace(model="openai/gpt-4o-mini", temp=0)
            )
        )
    )
    prompt = {"task": "bounded"}
    with (
        mock.patch.object(parallel_module, "query", return_value=malformed) as query,
        pytest.raises(ResearchDecisionError, match="malformed plan/code"),
    ):
        agent_class.plan_and_code_query(fake_agent, prompt, retries=3)

    assert query.call_count == 3


def test_plan_and_code_contract_accepts_one_valid_python_block() -> None:
    from ai_scientist.treesearch import parallel_agent as parallel_module

    fake_agent = SimpleNamespace(
        cfg=SimpleNamespace(
            agent=SimpleNamespace(
                code=SimpleNamespace(model="openai/gpt-4o-mini", temp=0)
            )
        )
    )
    with mock.patch.object(
        parallel_module,
        "query",
        return_value="Test the bounded hypothesis.\n```python\nx = 1\n```",
    ):
        plan, code = parallel_module.MinimalAgent.plan_and_code_query(
            fake_agent, {"task": "bounded"}
        )

    assert plan == "Test the bounded hypothesis."
    assert code.strip() == "x = 1"
