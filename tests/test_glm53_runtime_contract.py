from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from omegaconf import OmegaConf
import pytest

from ai_scientist.resources import bfts_config_path
from ai_scientist.treesearch import agent_manager as manager_module
from ai_scientist.treesearch import backend as backend_module
from ai_scientist.treesearch import journal as journal_module
from ai_scientist.treesearch import parallel_agent as parallel_module
from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.backend import ResearchDecisionError
from ai_scientist.treesearch.backend import backend_openai
from ai_scientist.treesearch.journal import Journal, Node

GLM53_MODEL = "openai_compat/glm-5.3"


def _profile():
    return OmegaConf.load(bfts_config_path("glm53"))


def _agent(cfg):
    return SimpleNamespace(cfg=cfg, task_desc="test a bounded hypothesis")


def _manager(cfg) -> AgentManager:
    manager = object.__new__(AgentManager)
    manager.cfg = cfg
    manager.current_stage = SimpleNamespace(
        name="1_initial_implementation_1_preliminary"
    )
    return manager


def test_glm53_code_limit_reaches_minimal_and_parallel_agent_queries() -> None:
    cfg = _profile()
    completion = "Run the bounded experiment.\n```python\nprint('ok')\n```"

    for agent_class in (parallel_module.MinimalAgent, parallel_module.ParallelAgent):
        with mock.patch.object(
            parallel_module,
            "query",
            return_value=completion,
        ) as query:
            plan, code = agent_class.plan_and_code_query(
                _agent(cfg),
                {"Task": "bounded"},
            )

        assert plan == "Run the bounded experiment."
        assert code.strip() == 'print("ok")'
        assert query.call_args.kwargs["model"] == GLM53_MODEL
        assert query.call_args.kwargs["max_tokens"] == 8192


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "8192"])
def test_stage_max_tokens_rejects_invalid_runtime_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive integer or null"):
        parallel_module._stage_max_tokens(SimpleNamespace(max_tokens=invalid))


def test_glm53_feedback_limit_reaches_node_summary_query() -> None:
    cfg = _profile()
    node = SimpleNamespace(
        code="print('ok')",
        plan="bounded plan",
        term_out="ok",
        analysis="completed",
        metric=None,
        plot_analyses=[],
        vlm_feedback_summary="",
    )

    with mock.patch.object(
        parallel_module,
        "query",
        return_value={"summary": "bounded"},
    ) as query:
        result = parallel_module.MinimalAgent._generate_node_summary(
            _agent(cfg),
            node,
        )

    assert result == {"summary": "bounded"}
    assert query.call_args.kwargs["model"] == GLM53_MODEL
    assert query.call_args.kwargs["max_tokens"] == 4096


def test_glm53_vlm_null_limit_is_forwarded_to_central_router(tmp_path) -> None:
    cfg = _profile()
    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"bounded-image")
    node = SimpleNamespace(
        plot_paths=[str(plot_path)],
        plot_analyses=[],
        datasets_successfully_tested=["dataset-v1"],
        is_buggy_plots=None,
        vlm_feedback_summary="",
        plot_code="plot code",
    )
    response = {
        "plot_analyses": [{"analysis": "bounded evidence"}],
        "vlm_feedback_summary": "bounded evidence",
        "valid_plots_received": True,
    }

    with mock.patch.object(
        parallel_module,
        "query",
        return_value=response,
    ) as query:
        parallel_module.MinimalAgent._analyze_plots_with_vlm(_agent(cfg), node)

    assert query.call_args.kwargs["model"] == GLM53_MODEL
    assert query.call_args.kwargs["max_tokens"] is None


def test_glm53_summary_limit_reaches_journal_query() -> None:
    cfg = _profile()
    journal = Journal(nodes=[Node(is_buggy=True, analysis="failed", exc_type="Error")])

    with mock.patch.object(
        journal_module,
        "query",
        return_value="summary",
    ) as query:
        result = journal.generate_summary(
            model=cfg.agent.summary.model,
            temp=cfg.agent.summary.temp,
            max_tokens=cfg.agent.summary.max_tokens,
        )

    assert result == "summary"
    assert query.call_args.kwargs["model"] == GLM53_MODEL
    assert query.call_args.kwargs["max_tokens"] is None


def test_glm53_agent_manager_feedback_limit_reaches_every_direct_query() -> None:
    cfg = _profile()
    manager = _manager(cfg)
    stage = Stage(
        name="1_initial_implementation_1_preliminary",
        description="bounded stage",
        goals="collect bounded evidence",
        max_iterations=1,
        num_drafts=1,
        stage_number=1,
    )
    best_node = SimpleNamespace(
        id="node-1",
        plot_analyses=[{"analysis": "bounded visual evidence"}],
        vlm_feedback_summary="bounded visual evidence",
    )
    journal = SimpleNamespace(
        nodes=[best_node],
        get_best_node_by_metric=mock.Mock(return_value=best_node),
        get_best_node=mock.Mock(return_value=best_node),
    )
    responses = [
        {
            "is_complete": True,
            "reasoning": "bounded",
            "missing_criteria": [],
        },
        {
            "goals": "run the next bounded experiment",
            "sub_stage_name": "bounded_followup",
        },
        {
            "name": "bounded_followup",
            "description": "run a bounded follow-up",
            "goals": ["collect evidence"],
            "max_iterations": 1,
        },
        {
            "ready_for_next_stage": False,
            "reasoning": "more evidence required",
            "recommendations": ["run the bounded follow-up"],
        },
    ]

    with (
        mock.patch.object(
            manager,
            "_gather_stage_metrics",
            return_value={
                "total_nodes": 1,
                "good_nodes": 1,
                "best_metric": None,
            },
        ),
        mock.patch.object(manager, "_identify_issues", return_value=[]),
        mock.patch.object(
            manager,
            "_analyze_progress",
            return_value={
                "convergence_status": "not_converged",
                "recent_changes": [],
            },
        ),
        mock.patch.object(
            manager_module,
            "query",
            side_effect=responses,
        ) as query,
    ):
        assert manager._check_substage_completion(stage, journal)[0] is True
        assert manager._generate_substage_goal("bounded main goal", journal) == (
            "run the next bounded experiment",
            "bounded_followup",
        )
        assert manager._get_response("configure a bounded stage") == responses[2]
        assert manager._evaluate_stage_progression(stage, {}) == responses[3]

    assert query.call_count == 4
    assert [call.kwargs["func_spec"].name for call in query.call_args_list] == [
        "evaluate_stage_completion",
        "generate_substage_goals",
        "generate_stage_config",
        "evaluate_stage_progression",
    ]
    for call in query.call_args_list:
        assert call.kwargs["model"] == GLM53_MODEL
        assert call.kwargs["max_tokens"] == 4096


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "4096"])
def test_agent_manager_rejects_invalid_feedback_limit_before_wire_call(
    invalid: object,
) -> None:
    cfg = _profile()
    cfg.agent.feedback.max_tokens = invalid
    manager = _manager(cfg)

    with (
        mock.patch.object(manager_module, "query") as query,
        pytest.raises(ResearchDecisionError, match="stage configuration"),
    ):
        manager._get_response("configure a bounded stage")

    query.assert_not_called()


def test_glm53_null_roles_use_router_bounded_global_wire_default(monkeypatch) -> None:
    cfg = _profile()
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=role),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            system_fingerprint=None,
            model="glm-5.3",
            created=1,
        )
        for role in ("vlm", "summary")
    ]
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=mock.Mock(side_effect=responses),
            )
        )
    )
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "test-compat-key")
    monkeypatch.setenv(
        "OPENAI_COMPAT_BASE_URL",
        "https://compat-runtime.test/v1",
    )

    def immediate_backoff(create, _retry_exceptions, **kwargs):
        wire_kwargs = {
            key: value
            for key, value in kwargs.items()
            if not key.startswith("_budget_")
        }
        return create(**wire_kwargs)

    with (
        mock.patch.object(backend_openai, "get_ai_client", return_value=client),
        mock.patch.object(
            backend_openai,
            "backoff_create",
            side_effect=immediate_backoff,
        ),
    ):
        for role in ("vlm_feedback", "summary"):
            role_cfg = cfg.agent[role]
            assert role_cfg.max_tokens is None
            backend_module.query(
                system_message="bounded",
                user_message=None,
                model=role_cfg.model,
                temperature=role_cfg.temp,
                max_tokens=role_cfg.max_tokens,
            )

    assert client.chat.completions.create.call_count == 2
    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs["model"] == "glm-5.3"
        assert call.kwargs["max_tokens"] == 8192


@pytest.mark.parametrize("invalid", [True, 0])
def test_backend_rejects_invalid_output_limit_before_provider_wire(
    invalid: object,
) -> None:
    with (
        mock.patch.object(backend_module, "_resolve_backend_module") as resolve,
        pytest.raises(ValueError, match="positive integer or null"),
    ):
        backend_module.query(
            system_message="bounded",
            user_message=None,
            model=GLM53_MODEL,
            max_tokens=invalid,
        )

    resolve.assert_not_called()


def test_glm53_profile_does_not_invent_select_node_authority() -> None:
    cfg = _profile()
    journal = Journal()
    candidates = [
        SimpleNamespace(
            id=f"candidate-{index}",
            is_seed_node=False,
            metric=str(index),
            analysis="bounded",
            vlm_feedback_summary="",
        )
        for index in (1, 2)
    ]

    with (
        mock.patch.object(
            Journal,
            "verified_nodes",
            new_callable=mock.PropertyMock,
            return_value=candidates,
        ),
        mock.patch.object(
            journal,
            "get_best_node_by_metric",
            return_value=candidates[0],
        ) as deterministic,
        mock.patch.object(journal_module, "query") as query,
    ):
        selected = journal.get_best_node(cfg=cfg)

    assert selected is candidates[0]
    deterministic.assert_called_once_with()
    query.assert_not_called()


def test_explicit_select_node_limit_reaches_query_without_fallback_creation() -> None:
    journal = Journal()
    candidates = [
        SimpleNamespace(
            id=f"candidate-{index}",
            is_seed_node=False,
            metric=str(index),
            analysis="bounded",
            vlm_feedback_summary="",
        )
        for index in (1, 2)
    ]
    cfg = OmegaConf.create(
        {
            "agent": {
                "select_node": {
                    "model": GLM53_MODEL,
                    "temp": 0.2,
                    "max_tokens": 777,
                }
            }
        }
    )

    with (
        mock.patch.object(
            Journal,
            "verified_nodes",
            new_callable=mock.PropertyMock,
            return_value=candidates,
        ),
        mock.patch.object(
            journal_module,
            "query",
            return_value={
                "selected_id": candidates[0].id,
                "reasoning": "bounded",
            },
        ) as query,
    ):
        selected = journal.get_best_node(cfg=cfg)

    assert selected is candidates[0]
    assert query.call_args.kwargs["max_tokens"] == 777
