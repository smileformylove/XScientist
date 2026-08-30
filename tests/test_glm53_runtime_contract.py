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
JUDGMENT_MODEL = "openai/gpt-4o-mini"


@pytest.fixture(autouse=True)
def _isolate_relative_authority_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)


def _profile():
    return OmegaConf.load(bfts_config_path("glm53"))


def _split_profile():
    cfg = _profile()
    for role in ("judgment", "feedback", "vlm_feedback", "summary", "select_node"):
        cfg.agent[role].model = JUDGMENT_MODEL
    return cfg


def _agent(cfg):
    return SimpleNamespace(
        cfg=cfg,
        task_desc="test a bounded hypothesis",
        evaluation_metrics="accuracy",
    )


def _manager(cfg) -> AgentManager:
    manager = object.__new__(AgentManager)
    manager.cfg = cfg
    manager.current_stage = SimpleNamespace(
        name="1_initial_implementation_1_preliminary"
    )
    return manager


def _implementation_spec() -> dict:
    return {
        "objective": "Run the bounded experiment.",
        "hypothesis": "The bounded implementation will produce valid evidence.",
        "implementation_steps": ["Implement the frozen experiment."],
        "locked_parameters": [],
        "required_outputs": ["experiment_data.npy"],
        "prohibited_changes": ["Do not change the metric."],
        "acceptance_checks": ["The program exits successfully."],
        "hyperparameter_trial": None,
        "ablation_contract": None,
    }


def test_packaged_glm53_profile_fails_closed_without_judgment() -> None:
    for agent_class in (parallel_module.MinimalAgent, parallel_module.ParallelAgent):
        with (
            mock.patch.object(parallel_module, "query") as query,
            pytest.raises(ResearchDecisionError, match="judgment"),
        ):
            agent_class.plan_and_code_query(
                _agent(_profile()),
                {"Task": "bounded"},
            )
        query.assert_not_called()


def test_glm53_uses_judgment_then_code_then_conformance() -> None:
    cfg = _split_profile()
    responses = [
        _implementation_spec(),
        "```python\nprint('ok')\n```",
        {"conforms": True, "violations": []},
    ]

    for agent_class in (parallel_module.MinimalAgent, parallel_module.ParallelAgent):
        with mock.patch.object(
            parallel_module,
            "query",
            side_effect=responses,
        ) as query:
            agent = _agent(cfg)
            plan, code = agent_class.plan_and_code_query(
                agent,
                {"Task": "bounded"},
            )

        assert plan.startswith("Run the bounded experiment.")
        assert code.strip() == 'print("ok")'
        assert [call.kwargs["model"] for call in query.call_args_list] == [
            JUDGMENT_MODEL,
            GLM53_MODEL,
            JUDGMENT_MODEL,
        ]
        assert [call.kwargs["max_tokens"] for call in query.call_args_list] == [
            4096,
            8192,
            4096,
        ]
        assert agent._last_locked_spec_hash.startswith("sha256:")
        assert (
            query.call_args_list[1].kwargs["system_message"]["Locked spec hash"]
            == agent._last_locked_spec_hash
        )


def test_glm53_rejects_code_when_judgment_conformance_never_passes(tmp_path) -> None:
    cfg = _split_profile()
    cfg.log_dir = str(tmp_path / "logs")
    responses: list[object] = [_implementation_spec()]
    for attempt in range(3):
        responses.extend(
            [
                f"```python\nprint({attempt})\n```",
                {"conforms": False, "violations": ["changed locked parameter"]},
            ]
        )
    agent = _agent(cfg)
    with (
        mock.patch.object(parallel_module, "query", side_effect=responses) as query,
        pytest.raises(ResearchDecisionError, match="conformance"),
    ):
        parallel_module.MinimalAgent.plan_and_code_query(
            agent,
            {"Task": "bounded"},
        )

    assert query.call_count == 7
    assert getattr(agent, "_last_locked_spec", None) is None
    assert (
        len(
            list(
                (tmp_path / "logs/authority_objects/implementation-spec").glob("*.json")
            )
        )
        == 1
    )
    assert (
        len(list((tmp_path / "logs/authority_objects/conformance").glob("*.json"))) == 3
    )


def test_glm53_hyperparameter_node_binds_concrete_trial() -> None:
    cfg = _split_profile()
    agent = parallel_module.MinimalAgent(
        task_desc="bounded",
        cfg=cfg,
        evaluation_metrics="accuracy",
    )
    parent = Node(code="learning_rate = 0.1\nprint(learning_rate)")
    idea = parallel_module.HyperparamTuningIdea(
        "learning rate",
        "Compare two locked values.",
        parameter="learning_rate",
        control_value="0.1",
        candidate_values=["0.01", "0.001"],
        selection_rule="Select by frozen accuracy on the locked validation split.",
        authority_attempt_ids=["attempt-" + "a" * 32],
        authority_attempt_terminal_hashes={
            "attempt-" + "a" * 32: "sha256:" + "b" * 64,
        },
    )
    with mock.patch.object(
        parallel_module,
        "query",
        side_effect=[
            _implementation_spec(),
            "```python\nlearning_rate = 0.01\nprint(learning_rate)\n```",
            {"conforms": True, "violations": []},
        ],
    ):
        node = agent._generate_hyperparam_tuning_node(parent, idea)

    assert node.implementation_spec_hash.startswith("sha256:")
    assert node.implementation_spec["hyperparameter_trial"] == {
        "parameter": "learning_rate",
        "control_value": "0.1",
        "candidate_values": ["0.01", "0.001"],
        "selection_rule": "Select by frozen accuracy on the locked validation split.",
    }
    assert "attempt-" + "a" * 32 in node.authority_attempt_ids
    assert len(node.authority_attempt_ids) >= 4
    assert set(node.authority_attempt_terminal_hashes) == set(
        node.authority_attempt_ids
    )


@pytest.mark.parametrize("code_model", [GLM53_MODEL, JUDGMENT_MODEL])
def test_improve_does_not_depend_on_hyperparameter_idea(code_model: str) -> None:
    cfg = _split_profile()
    cfg.agent.code.model = code_model
    agent = parallel_module.MinimalAgent(
        task_desc="bounded",
        cfg=cfg,
        evaluation_metrics="accuracy",
    )
    parent = Node(
        code="print('baseline')",
        vlm_feedback_summary="bounded",
        exec_time_feedback="bounded",
    )

    with mock.patch.object(
        agent,
        "plan_and_code_query",
        return_value=("improve safely", "print('improved')"),
    ):
        node = agent._improve(parent)

    assert node.parent is parent
    assert node.plan == "improve safely"
    assert node.code == "print('improved')"


def test_seed_node_does_not_claim_ancestor_authority_attempts() -> None:
    cfg = _split_profile()
    agent = parallel_module.MinimalAgent(
        task_desc="bounded",
        cfg=cfg,
        evaluation_metrics="accuracy",
    )
    attempt_id = "attempt-" + "c" * 32
    parent = Node(
        code="print('baseline')",
        authority_attempt_ids=[attempt_id],
        authority_attempt_terminal_hashes={attempt_id: "sha256:" + "d" * 64},
    )

    seed = agent._generate_seed_node(parent)

    assert seed.parent is parent
    assert seed.authority_attempt_ids == []
    assert seed.authority_attempt_terminal_hashes == {}


def test_glm53_ablation_node_binds_control_component_and_outcome() -> None:
    cfg = _split_profile()
    agent = parallel_module.MinimalAgent(
        task_desc="bounded",
        cfg=cfg,
        evaluation_metrics="accuracy",
    )
    parent = Node(code="use_attention = True\nprint(use_attention)")
    idea = parallel_module.AblationIdea(
        "remove attention",
        "Disable exactly the attention component.",
        "attention",
        "Accuracy should decrease if attention contributes.",
    )
    with mock.patch.object(
        parallel_module,
        "query",
        side_effect=[
            _implementation_spec(),
            "```python\nuse_attention = False\nprint(use_attention)\n```",
            {"conforms": True, "violations": []},
        ],
    ):
        node = agent._generate_ablation_node(parent, idea)

    contract = node.implementation_spec["ablation_contract"]
    assert contract["component"] == "attention"
    assert contract["control_node_id"] == parent.id
    assert contract["control_code_hash"] == parallel_module._semantic_code_hash(
        parent.code
    )
    assert contract["expected_outcome"] == idea.expected_outcome


def test_glm53_debug_inherits_scientific_spec_and_adds_repair_spec() -> None:
    cfg = _split_profile()
    agent = parallel_module.MinimalAgent(
        task_desc="bounded",
        cfg=cfg,
        evaluation_metrics="accuracy",
    )
    spec = {
        "schema": "xscientist.locked-experiment-spec.v1",
        "task_kind": "baseline",
        "primary_metric": "accuracy",
        "scientific_context_hash": "sha256:" + "1" * 64,
        "judgment_model": JUDGMENT_MODEL,
        **_implementation_spec(),
    }
    spec_hash = parallel_module._canonical_spec_hash(spec)
    parent = Node(
        code="raise RuntimeError('bug')",
        implementation_spec=spec,
        implementation_spec_hash=spec_hash,
        _term_out=["RuntimeError: bug"],
    )
    repair = {
        "failure_summary": "The implementation raises before producing output.",
        "repair_steps": ["Remove the unintended exception."],
        "prohibited_changes": ["Keep all scientific inputs fixed."],
        "acceptance_checks": ["The script produces the required output."],
    }
    with mock.patch.object(
        parallel_module,
        "query",
        side_effect=[
            repair,
            "```python\nprint('fixed')\n```",
            {"conforms": True, "violations": []},
        ],
    ):
        child = agent._debug(parent)

    assert child.implementation_spec == spec
    assert child.implementation_spec_hash == spec_hash
    assert child.repair_spec["parent_implementation_spec_hash"] == spec_hash
    assert child.repair_spec_hash.startswith("sha256:")


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "8192"])
def test_stage_max_tokens_rejects_invalid_runtime_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive integer or null"):
        parallel_module._stage_max_tokens(SimpleNamespace(max_tokens=invalid))


def test_judgment_feedback_limit_reaches_node_summary_query() -> None:
    cfg = _split_profile()
    node = SimpleNamespace(
        id="node-summary-test",
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
    assert query.call_args.kwargs["model"] == JUDGMENT_MODEL
    assert query.call_args.kwargs["max_tokens"] == 4096
    assert len(node.authority_attempt_ids) == 1
    assert set(node.authority_attempt_terminal_hashes) == set(
        node.authority_attempt_ids
    )


def test_judgment_vlm_null_limit_is_forwarded_to_central_router(tmp_path) -> None:
    cfg = _split_profile()
    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"bounded-image")
    node = SimpleNamespace(
        id="node-vlm-test",
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

    assert query.call_args.kwargs["model"] == JUDGMENT_MODEL
    assert query.call_args.kwargs["max_tokens"] is None
    assert len(node.authority_attempt_ids) == 1
    assert set(node.authority_attempt_terminal_hashes) == set(
        node.authority_attempt_ids
    )


def test_judgment_summary_limit_reaches_journal_query(tmp_path) -> None:
    cfg = _split_profile()
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
            log_dir=tmp_path / "logs",
        )

    assert result == "summary"
    assert query.call_args.kwargs["model"] == JUDGMENT_MODEL
    assert query.call_args.kwargs["max_tokens"] is None


def test_judgment_agent_manager_limit_reaches_every_direct_query() -> None:
    cfg = _split_profile()
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
        assert call.kwargs["model"] == JUDGMENT_MODEL
        assert call.kwargs["max_tokens"] == 4096


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "4096"])
def test_agent_manager_rejects_invalid_feedback_limit_before_wire_call(
    invalid: object,
) -> None:
    cfg = _split_profile()
    cfg.agent.feedback.max_tokens = invalid
    manager = _manager(cfg)

    with (
        mock.patch.object(manager_module, "query") as query,
        pytest.raises(ResearchDecisionError, match="stage configuration"),
    ):
        manager._get_response("configure a bounded stage")

    query.assert_not_called()


def test_judgment_null_roles_use_router_bounded_global_wire_default(
    monkeypatch,
) -> None:
    cfg = _split_profile()
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
            model="gpt-4o-mini",
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
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

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
        assert call.kwargs["model"] == "gpt-4o-mini"
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


def test_glm53_profile_routes_select_node_to_judgment() -> None:
    cfg = _split_profile()
    journal = Journal()
    candidates = [
        SimpleNamespace(
            id=f"candidate-{index}",
            is_seed_node=False,
            metric=str(index),
            analysis="bounded",
            vlm_feedback_summary="",
            selection_llm_call_refs=[],
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
    assert query.call_args.kwargs["model"] == JUDGMENT_MODEL
    assert query.call_args.kwargs["max_tokens"] == 4096


def test_explicit_select_node_limit_reaches_query_without_fallback_creation() -> None:
    journal = Journal()
    candidates = [
        SimpleNamespace(
            id=f"candidate-{index}",
            is_seed_node=False,
            metric=str(index),
            analysis="bounded",
            vlm_feedback_summary="",
            selection_llm_call_refs=[],
        )
        for index in (1, 2)
    ]
    cfg = OmegaConf.create(
        {
            "log_dir": "logs",
            "agent": {
                "select_node": {
                    "model": JUDGMENT_MODEL,
                    "temp": 0.2,
                    "max_tokens": 777,
                }
            },
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
