"""Independent small-path checks and hostile finite-controller inputs."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import product
import subprocess
import sys
from textwrap import dedent

import pytest

from ai_scientist.utils import exact_control_evaluator as evaluator


def _noisy_model():
    return evaluator.FiniteControlModel(
        (F(2, 5), F(3, 5)),
        (
            ((F(3, 4), F(1, 4)), (F(1, 3), F(2, 3))),
            ((F(1, 5), F(4, 5)), (F(2, 3), F(1, 3))),
        ),
        ((F(4, 5), F(1, 5)), (F(1, 7), F(6, 7))),
        ((-1, F(1, 2)), (F(1, 3), F(-1, 4))),
        (F(1, 6), F(-1, 5)),
    )


def _constant_policy(action=0, horizon=1):
    probabilities = (1 - action, action)
    first = (probabilities, ((0, 0), (0, 0)))
    last = (probabilities, ((-1, -1), (-1, -1)))
    return evaluator.PublicController(
        2,
        2,
        (0, 0) if horizon else (-1, -1),
        ((first,),) * (horizon - 1) + ((last,),) if horizon else (),
    )


def _enumerate_paths_with_counts(model, policy):
    """Sum complete paths, without beliefs, Bellman tables, or mass merging."""
    horizon = len(policy.layers)
    mass, value = F(0), F(0)
    counts = [F(0)] * policy.actions
    for states in product(range(len(model.initial)), repeat=horizon + 1):
        for observations in product(range(policy.observations), repeat=horizon + 1):
            for actions in product(range(policy.actions), repeat=horizon):
                weight = (
                    model.initial[states[0]]
                    * model.emission[states[0]][observations[0]]
                )
                node = policy.initial_nodes[observations[0]]
                total_reward = model.initial_reward[observations[0]]
                for t, action in enumerate(actions):
                    probabilities, successors = policy.layers[t][node]
                    weight *= (
                        probabilities[action]
                        * model.transition[action][states[t]][states[t + 1]]
                        * model.emission[states[t + 1]][observations[t + 1]]
                    )
                    total_reward += model.reward[action][observations[t + 1]]
                    node = successors[action][observations[t + 1]]
                mass += weight
                value += weight * total_reward
                for action in range(policy.actions):
                    counts[action] += weight * actions.count(action)
    assert mass == 1
    return value, counts


def test_stochastic_action_memory_matches_independent_full_paths():
    policy = evaluator.PublicController(
        2,
        2,
        (0, 1),
        (
            (
                ((F(1, 3), F(2, 3)), ((0, 1), (1, 0))),
                ((F(3, 5), F(2, 5)), ((1, 0), (0, 1))),
            ),
            (
                ((F(1, 4), F(3, 4)), ((0, 1), (1, 0))),
                ((F(2, 7), F(5, 7)), ((1, 0), (0, 1))),
            ),
            (
                ((F(1, 7), F(6, 7)), ((-1, -1), (-1, -1))),
                ((F(2, 3), F(1, 3)), ((-1, -1), (-1, -1))),
            ),
        ),
    )
    model = _noisy_model()
    value, counts = _enumerate_paths_with_counts(model, policy)
    report = evaluator.evaluate_exact_control(model, policy)
    diagnostics = report["action_diagnostics"]
    assert F(report["policy_value"]) == value
    assert [F(row["expected_count"]) for row in diagnostics] == counts
    assert sum(counts) == 3
    assert (
        sum(F(row["regret_contribution"]) for row in diagnostics)
        == F(report["public_history_optimal_value"]) - value
    )
    assert all(F(row["regret_contribution"]) >= 0 for row in diagnostics)
    assert sum(
        F(row["expected_count"]) / 3 * F(row["conditional_optimal_action_loss"])
        for row in diagnostics
    ) == F(report["regret_per_action"])


def test_learner_histories_outside_unique_oracle_support_are_evaluated():
    # Oracle stays in observed state 0. Learner enters observed state 1,
    # where its second action has loss 2, unlike loss 1 at the first history.
    model = evaluator.FiniteControlModel(
        (1, 0),
        (((1, 0), (0, 1)), ((0, 1), (0, 1))),
        ((1, 0), (0, 1)),
        ((F(1, 2), 1), (-1, -1)),
        (0, 0),
    )
    report = evaluator.evaluate_exact_control(model, _constant_policy(1, 2))
    assert report["policy_value"] == "-2"
    assert report["public_history_optimal_value"] == "1"
    assert report["regret"] == "3"
    assert report["action_diagnostics"][1]["regret_contribution"] == "3"
    assert report["action_diagnostics"][1]["conditional_optimal_action_loss"] == "3/2"
    assert report["action_diagnostics"][0]["conditional_optimal_action_loss"] is None


def test_public_action_and_observation_recover_information_without_state_oracle():
    # Hidden identity is fixed, but observations reveal whether the preceding
    # guess matched it. Remembering the preceding action is essential.
    model = evaluator.FiniteControlModel(
        (0, F(1, 2), 0, F(1, 2)),
        (
            ((1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 1)),
            ((0, 1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
        ),
        ((0, 1), (1, 0), (0, 1), (1, 0)),
        ((0, 1), (0, 1)),
        (0, 0),
    )
    terminal = tuple(
        ((1 - action, action), ((-1, -1), (-1, -1))) for action in (1, 0, 0, 1)
    )
    policy = evaluator.PublicController(
        2,
        2,
        (0, 0),
        ((((F(1, 3), F(2, 3)), ((0, 1), (2, 3))),), terminal),
    )
    report = evaluator.evaluate_exact_control(model, policy)
    assert report["policy_value"] == report["public_history_optimal_value"] == "3/2"
    assert report["regret"] == "0"
    # A state-observing comparator would obtain 2, which is not this oracle.
    assert F(report["public_history_optimal_value"]) < 2
    assert [F(row["expected_count"]) for row in report["action_diagnostics"]] == [
        F(5, 6),
        F(7, 6),
    ]


def test_zero_horizon_with_initial_observation_reward_and_unused_actions():
    report = evaluator.evaluate_exact_control(_noisy_model(), _constant_policy(0, 0))
    assert (
        report["policy_value"] == report["public_history_optimal_value"] == "-269/5250"
    )
    assert report["regret"] == "0"
    assert report["regret_per_action"] is None
    assert report["observation_horizon"] == 1
    for row in report["action_diagnostics"]:
        assert row["expected_count"] == row["regret_contribution"] == "0"
        assert row["conditional_optimal_action_loss"] is None


@pytest.mark.parametrize(
    "initial_nodes,layers",
    [
        ((0, 1), ((((1, 0), ((-1, -1), (-1, -1))),),)),
        ((0, 0), ((((1, 0), ((-1, -1), (-1, 0))),),)),
        ((0, 0), ((((1, 0), ((-1, -1), (-1, True))),),)),
        ((0, 0), ((((1, 0), ((-1, -1), (-1, -1))), ((-1, 2), ((-1, -1), (-1, -1)))),)),
        (
            (0, 0),
            ((((1, 0), ((-1, -1), (-1, -1))),), (((1, 0), ((-1, -1), (-1, -1))),)),
        ),
    ],
)
def test_unused_and_impossible_controller_branches_still_rejected(
    initial_nodes, layers
):
    with pytest.raises(ValueError):
        evaluator.PublicController(2, 2, initial_nodes, layers)


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_horizon_zero_still_checks_dimensions_and_initial_pointers(bad):
    with pytest.raises(ValueError):
        evaluator.PublicController(bad, 2, (-1, -1), ())
    with pytest.raises(ValueError):
        evaluator.PublicController(2, 2, (-1, bad), ())


def test_tree_work_cap_is_checked_at_exact_boundary(monkeypatch):
    model, policy = _noisy_model(), _constant_policy()
    monkeypatch.setattr(evaluator, "MAX_TREE_WORK", 10)
    assert evaluator.evaluate_exact_control(model, policy)["tree_work_bound"] == 10
    monkeypatch.setattr(evaluator, "MAX_TREE_WORK", 9)
    with pytest.raises(ValueError, match="tree work"):
        evaluator.evaluate_exact_control(model, policy)


def test_node_and_layer_caps_include_unused_nodes(monkeypatch):
    terminal = ((1, 0), ((-1, -1), (-1, -1)))
    monkeypatch.setattr(evaluator, "MAX_NODES", 2)
    evaluator.PublicController(2, 2, (0, 0), ((terminal, terminal),))
    with pytest.raises(ValueError, match="sequence|node"):
        evaluator.PublicController(2, 2, (0, 0), ((terminal, terminal, terminal),))
    nonterminal = ((1, 0), ((0, 0), (0, 0)))
    with pytest.raises(ValueError, match="node limit"):
        evaluator.PublicController(
            2, 2, (0, 0), ((nonterminal, nonterminal), (terminal,))
        )
    with pytest.raises(ValueError, match="sequence"):
        evaluator.PublicController(2, 2, (0, 0), ((terminal,),) * 9)


def test_every_nested_model_input_is_detached_and_frozen():
    initial = [F(1, 2), F(1, 2)]
    transition = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
    emission = [[1, 0], [0, 1]]
    reward = [[-1, 1], [1, -1]]
    initial_reward = [0, 1]
    model = evaluator.FiniteControlModel(
        initial, transition, emission, reward, initial_reward
    )
    before = evaluator.evaluate_exact_control(model, _constant_policy())
    initial[0] = 99
    transition[0][0][0] = 99
    emission[0][0] = 99
    reward[0][0] = 99
    initial_reward[0] = 99
    assert evaluator.evaluate_exact_control(model, _constant_policy()) == before
    with pytest.raises(FrozenInstanceError):
        model.initial = (1, 0)
    with pytest.raises(TypeError):
        model.transition[0][0][0] = 0


@pytest.mark.parametrize("target", ["model", "policy"])
def test_corrupted_frozen_objects_are_revalidated(target):
    model, policy = _noisy_model(), _constant_policy()
    if target == "model":
        object.__setattr__(model, "initial", (1, 1))
    else:
        object.__setattr__(policy, "initial_nodes", (True, 0))
    with pytest.raises(ValueError):
        evaluator.evaluate_exact_control(model, policy)


def test_model_and_controller_alphabet_mismatch_rejected():
    policy = evaluator.PublicController(1, 1, (0,), ((((1,), ((-1,),)),),))
    with pytest.raises(ValueError, match="alphabet"):
        evaluator.evaluate_exact_control(_noisy_model(), policy)


def test_subclassed_numerics_and_unvalidated_object_types_are_rejected():
    class DisguisedInt(int):
        pass

    class DisguisedFraction(F):
        pass

    for bad in (DisguisedInt(1), DisguisedFraction(1)):
        with pytest.raises(ValueError, match="exact integer or Fraction"):
            evaluator.FiniteControlModel((bad,), (((1,),),), ((1,),), ((1,),), (0,))
    with pytest.raises(ValueError, match="validated"):
        evaluator.evaluate_exact_control({}, _constant_policy())
    with pytest.raises(ValueError, match="validated"):
        evaluator.evaluate_exact_control(_noisy_model(), {})


def test_input_fraction_bit_limit_and_reward_range(monkeypatch):
    monkeypatch.setattr(evaluator, "MAX_INPUT_BITS", 3)
    with pytest.raises(ValueError, match="bit limit"):
        evaluator.FiniteControlModel((1,), (((1,),),), ((1,),), ((F(1, 8),),), (0,))
    with pytest.raises(ValueError, match="rewards"):
        evaluator.FiniteControlModel((1,), (((1,),),), ((1,),), ((F(3, 2),),), (0,))


def test_distinct_denominator_cap_deduplicates_and_checks_unused_nodes(monkeypatch):
    model = evaluator.FiniteControlModel(
        (1,), (((1,),), ((1,),)), ((1,),), ((1,), (0,)), (0,)
    )
    thirds = ((F(1, 3), F(2, 3)), ((-1,), (-1,)))
    fifths = ((F(1, 5), F(4, 5)), ((-1,), (-1,)))
    monkeypatch.setattr(evaluator, "MAX_DISTINCT_DENOMINATOR_BITS", 3)
    repeated = evaluator.PublicController(2, 1, (0,), ((thirds,) * 8,))
    assert evaluator.evaluate_exact_control(model, repeated)["policy_value"] == "1/3"
    distinct = evaluator.PublicController(2, 1, (0,), ((thirds, fifths),))

    def expansion_must_not_start(*args):
        raise RuntimeError("preflight allowed arithmetic expansion")

    monkeypatch.setattr(evaluator, "_sum", expansion_must_not_start)
    with pytest.raises(ValueError, match="distinct denominator"):
        evaluator.evaluate_exact_control(model, distinct)


def test_runtime_fraction_cap_checks_denominator_growth(monkeypatch):
    model = evaluator.FiniteControlModel(
        (1,), (((1,),), ((1,),)), ((1,),), ((1,), (0,)), (0,)
    )
    policy = evaluator.PublicController(
        2, 1, (0,), ((((F(1, 3), F(2, 3)), ((-1,), (-1,))),),)
    )
    monkeypatch.setattr(evaluator, "MAX_ARITHMETIC_BITS", 1)
    with pytest.raises(ValueError, match="arithmetic bit"):
        evaluator.evaluate_exact_control(model, policy)


def test_runtime_fraction_cap_checks_numerator_growth(monkeypatch):
    model = evaluator.FiniteControlModel((1,), (((1,),),), ((1,),), ((1,),), (0,))
    node, last = ((1,), ((0,),)), ((1,), ((-1,),))
    policy = evaluator.PublicController(1, 1, (0,), ((node,),) * 3 + ((last,),))
    monkeypatch.setattr(evaluator, "MAX_ARITHMETIC_BITS", 2)
    with pytest.raises(ValueError, match="arithmetic bit"):
        evaluator.evaluate_exact_control(model, policy)


def test_normal_and_optimized_python_match_and_enforce_caps():
    script = dedent("""
        import json
        from fractions import Fraction as F
        from ai_scientist.utils import exact_control_evaluator as e
        model = e.FiniteControlModel(
            (1,), (((1,),), ((1,),)), ((1,),), ((1,), (0,)), (0,)
        )
        policy = e.PublicController(
            2, 1, (0,), ((((F(1, 3), F(2, 3)), ((-1,), (-1,))),),)
        )
        report = e.evaluate_exact_control(model, policy)
        e.MAX_ARITHMETIC_BITS = 1
        try:
            e.evaluate_exact_control(model, policy)
        except ValueError as error:
            if 'arithmetic bit' not in str(error):
                raise
        else:
            raise RuntimeError('arithmetic guard absent')
        e.MAX_ARITHMETIC_BITS = 32768
        e.MAX_DISTINCT_DENOMINATOR_BITS = 1
        try:
            e.evaluate_exact_control(model, policy)
        except ValueError as error:
            if 'distinct denominator' not in str(error):
                raise
        else:
            raise RuntimeError('denominator guard absent')
        try:
            e.PublicController(2, 1, (0,), ((((1, 0), ((-1,), (0,))),),))
        except ValueError:
            pass
        else:
            raise RuntimeError('unused-branch validation absent')
        print(json.dumps(report, sort_keys=True))
        """)
    normal = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    optimized = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert normal.stdout == optimized.stdout


def test_verified_supplied_model_does_not_gain_execution_authority():
    report = evaluator.evaluate_exact_control(_noisy_model(), _constant_policy())
    assert report["identity_verified_exactly"] is True
    assert report["oracle_information"] == "public_history_only"
    assert report["ground_truth_authority"] == "supplied_model_unverified"
    for field in (
        "environment_provenance_verified",
        "comparison_ready",
        "experiment_authorized",
        "training_authorized",
        "claim_promotion_allowed",
    ):
        assert report[field] is False
    assert report["independent_efficacy_endpoints"] == 1
