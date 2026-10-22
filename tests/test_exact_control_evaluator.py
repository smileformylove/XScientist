from fractions import Fraction as F
from itertools import product

import pytest

from ai_scientist.utils.exact_control_evaluator import (
    FiniteControlModel,
    PublicController,
    evaluate_exact_control,
)


def action_controller(action, horizon=1):
    node = ((1 - action, action), ((0, 0), (0, 0)))
    terminal = ((1 - action, action), ((-1, -1), (-1, -1)))
    return PublicController(
        2,
        2,
        (0, 0) if horizon else (-1, -1),
        tuple([(node,)] * (horizon - 1) + [(terminal,)]) if horizon else (),
    )


def choice_model():
    return FiniteControlModel(
        (F(1, 2), F(1, 2)),
        (((F(1, 4), F(3, 4)),) * 2, ((F(3, 4), F(1, 4)),) * 2),
        ((1, 0), (0, 1)),
        ((0, 1), (0, 1)),
        (0, 0),
    )


def test_exact_suboptimal_action_contribution():
    report = evaluate_exact_control(choice_model(), action_controller(1, 3))
    assert report["policy_value"] == "3/4"
    assert report["public_history_optimal_value"] == "9/4"
    assert report["regret"] == "3/2"
    assert report["action_diagnostics"] == [
        {
            "action": 0,
            "expected_count": "0",
            "regret_contribution": "0",
            "conditional_optimal_action_loss": None,
        },
        {
            "action": 1,
            "expected_count": "3",
            "regret_contribution": "3/2",
            "conditional_optimal_action_loss": "1/2",
        },
    ]


def test_stochastic_policy_and_zero_action_horizon():
    policy = PublicController(2, 2, (0, 0), ((([F(1, 2)] * 2, [[-1] * 2] * 2),),))
    report = evaluate_exact_control(choice_model(), policy)
    assert report["policy_value"] == "1/2"
    assert report["regret"] == "1/4"
    assert [row["expected_count"] for row in report["action_diagnostics"]] == [
        "1/2",
        "1/2",
    ]
    model = choice_model()
    initial_reward = FiniteControlModel(
        model.initial, model.transition, model.emission, model.reward, (0, 1)
    )
    empty = evaluate_exact_control(initial_reward, action_controller(0, 0))
    assert empty["policy_value"] == empty["public_history_optimal_value"] == "1/2"
    assert empty["regret_per_action"] is None


def test_hidden_state_oracle_is_not_used():
    # Initially both hidden states produce observation 0. A matching action
    # reaches the rewarding observation; no public policy can guess the state.
    model = FiniteControlModel(
        (F(1, 2), F(1, 2), 0, 0),
        (
            ((0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1)),
            ((0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        ),
        ((1, 0), (1, 0), (0, 1), (1, 0)),
        ((0, 1), (0, 1)),
        (0, 0),
    )
    report = evaluate_exact_control(model, action_controller(0))
    assert report["policy_value"] == report["public_history_optimal_value"] == "1/2"
    assert report["regret"] == "0"


def exhaustive_path_value(model, policy):
    """Independent enumeration of every hidden/observable/action sequence."""
    horizon = len(policy.layers)
    value, mass = F(0), F(0)
    for hidden in product(range(len(model.initial)), repeat=horizon + 1):
        for observed in product(range(policy.observations), repeat=horizon + 1):
            for actions in product(range(policy.actions), repeat=horizon):
                weight = (
                    model.initial[hidden[0]] * model.emission[hidden[0]][observed[0]]
                )
                reward = model.initial_reward[observed[0]]
                node = policy.initial_nodes[observed[0]]
                for t, action in enumerate(actions):
                    probabilities, successors = policy.layers[t][node]
                    weight *= (
                        probabilities[action]
                        * model.transition[action][hidden[t]][hidden[t + 1]]
                        * model.emission[hidden[t + 1]][observed[t + 1]]
                    )
                    reward += model.reward[action][observed[t + 1]]
                    node = successors[action][observed[t + 1]]
                value += weight * reward
                mass += weight
    assert mass == 1
    return value


def test_all_small_history_policies_against_independent_paths():
    model = FiniteControlModel(
        (F(1, 3), F(2, 3)),
        (
            ((F(3, 4), F(1, 4)), (F(1, 2), F(1, 2))),
            ((F(1, 4), F(3, 4)), (F(2, 3), F(1, 3))),
        ),
        ((F(2, 3), F(1, 3)), (F(1, 4), F(3, 4))),
        ((0, 1), (F(1, 3), F(2, 3))),
        (0, 1),
    )
    values, optima = [], set()
    for choices in product((0, 1), repeat=6):
        first = tuple(
            ((1 - a, a), ((2 * i, 2 * i + 1),) * 2) for i, a in enumerate(choices[:2])
        )
        last = tuple(((1 - a, a), ((-1, -1),) * 2) for a in choices[2:])
        policy = PublicController(2, 2, (0, 1), (first, last))
        report = evaluate_exact_control(model, policy)
        expected = exhaustive_path_value(model, policy)
        assert F(report["policy_value"]) == expected
        assert sum(
            F(row["regret_contribution"]) for row in report["action_diagnostics"]
        ) == F(report["regret"])
        values.append(expected)
        optima.add(F(report["public_history_optimal_value"]))
    assert optima == {max(values)}


@pytest.mark.parametrize("bad", [True, 0.5, float("nan"), "1/2", F(1, 2**65)])
def test_exact_input_contract(bad):
    with pytest.raises(ValueError):
        FiniteControlModel((bad, 1), (((1, 0), (0, 1)),), ((1,), (1,)), ((0,),), (0,))


def test_invalid_unreachable_branch_and_work_cap():
    with pytest.raises(ValueError, match="successor"):
        PublicController(2, 2, (0, 0), ((((1, 0), ((-1, -1), (-1, 0))),),))
    model = FiniteControlModel(
        (1,), (((1,),),) * 4, ((F(1, 4),) * 4,), ((0,) * 4,) * 4, (0,) * 4
    )
    node = ((1, 0, 0, 0), ((0,) * 4,) * 4)
    terminal = ((1, 0, 0, 0), ((-1,) * 4,) * 4)
    policy = PublicController(4, 4, (0,) * 4, ((node,),) * 7 + ((terminal,),))
    with pytest.raises(ValueError, match="tree work"):
        evaluate_exact_control(model, policy)


def test_mutable_inputs_are_frozen_and_no_admission():
    probabilities = [0, 1]
    successors = [[-1, -1], [-1, -1]]
    policy = PublicController(2, 2, [0, 0], [[(probabilities, successors)]])
    before = evaluate_exact_control(choice_model(), policy)
    probabilities[:] = [1, 0]
    successors[0][0] = 400
    assert before == evaluate_exact_control(choice_model(), policy)
    for key in (
        "comparison_ready",
        "experiment_authorized",
        "training_authorized",
        "claim_promotion_allowed",
        "environment_provenance_verified",
    ):
        assert before[key] is False
