"""Bounded exact evaluation of supplied finite models and public controllers.

This is model-conditional mathematics, not authenticated environment execution.
The reward contract is restricted to an initial observation reward followed by
reward[action][next_observation]. No hidden-state or extra reward information
is exposed to the controller. There are K actions and K+1 observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path

MAX_DIMENSION = 4
MAX_HORIZON = 8
MAX_NODES = 50_000
MAX_TREE_WORK = 250_000
MAX_INPUT_BITS = 64
MAX_DISTINCT_DENOMINATOR_BITS = 4096
# Also stays below Python's default integer-to-decimal serialization limit.
MAX_ARITHMETIC_BITS = 8192


def _checked(value):
    if (
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        > MAX_ARITHMETIC_BITS
    ):
        raise ValueError("exact arithmetic bit limit exceeded")
    return value


def _sum(values):
    result = F(0)
    for value in values:
        result = _checked(result + _checked(value))
    return result


def _check_denominators(model, policy):
    denominators = set()
    bits = 0
    pending = [
        model.initial,
        model.transition,
        model.emission,
        model.reward,
        model.initial_reward,
    ]
    pending.extend(node[0] for layer in policy.layers for node in layer)
    while pending:
        value = pending.pop()
        if type(value) is tuple:
            pending.extend(value)
        elif type(value) is F and value.denominator not in denominators:
            denominators.add(value.denominator)
            bits += value.denominator.bit_length()
            if bits > MAX_DISTINCT_DENOMINATOR_BITS:
                raise ValueError("distinct denominator bit limit exceeded")


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in [{lower}, {upper}]")
    return value


def _sequence(value, lower, upper, label):
    if type(value) not in (tuple, list) or not lower <= len(value) <= upper:
        raise ValueError(f"{label} has invalid sequence size or type")
    return tuple(value)


def _rational(value):
    if type(value) not in (int, F):
        raise ValueError("exact integer or Fraction required; floats/bools rejected")
    value = F(value)
    if (
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        > MAX_INPUT_BITS
    ):
        raise ValueError("rational input exceeds bit limit")
    return value


def _row(value, size, *, probability=False):
    row = tuple(_rational(x) for x in _sequence(value, size, size, "row"))
    if probability and (any(x < 0 for x in row) or sum(row) != 1):
        raise ValueError("probability row must be nonnegative and sum exactly to one")
    if not probability and any(abs(x) > 1 for x in row):
        raise ValueError("rewards must lie in [-1, 1]")
    return row


@dataclass(frozen=True)
class FiniteControlModel:
    """Homogeneous T[action][state][next_state], E[state][observation]."""

    initial: tuple
    transition: tuple
    emission: tuple
    reward: tuple
    initial_reward: tuple

    def __post_init__(self):
        states = len(_sequence(self.initial, 1, MAX_DIMENSION, "initial"))
        actions = len(_sequence(self.transition, 1, MAX_DIMENSION, "transition"))
        observations = len(
            _sequence(self.initial_reward, 1, MAX_DIMENSION, "initial_reward")
        )
        initial = _row(self.initial, states, probability=True)
        transition = tuple(
            tuple(
                _row(row, states, probability=True)
                for row in _sequence(matrix, states, states, "transition matrix")
            )
            for matrix in self.transition
        )
        emission = tuple(
            _row(row, observations, probability=True)
            for row in _sequence(self.emission, states, states, "emission")
        )
        reward = tuple(
            _row(row, observations)
            for row in _sequence(self.reward, actions, actions, "reward")
        )
        initial_reward = _row(self.initial_reward, observations)
        for name, value in (
            ("initial", initial),
            ("transition", transition),
            ("emission", emission),
            ("reward", reward),
            ("initial_reward", initial_reward),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PublicController:
    """Layered public automaton; nodes are (action probabilities, successors).

    initial_nodes[o0] selects layer zero. successors[action][next_observation]
    selects the next layer; last-layer pointers must all be -1. All nodes and
    branches, even impossible or unused ones, must be well formed. Private
    fresh action randomization is supported; hidden-state input is impossible.
    """

    actions: int
    observations: int
    initial_nodes: tuple
    layers: tuple

    def __post_init__(self):
        actions = _integer(self.actions, 1, MAX_DIMENSION, "actions")
        observations = _integer(self.observations, 1, MAX_DIMENSION, "observations")
        layers = _sequence(self.layers, 0, MAX_HORIZON, "layers")
        total = 0
        for layer in layers:
            total += len(_sequence(layer, 1, MAX_NODES, "layer"))
            if total > MAX_NODES:
                raise ValueError("controller node limit exceeded")
        frozen = []
        for t, layer in enumerate(layers):
            upper = len(layers[t + 1]) - 1 if t + 1 < len(layers) else -1
            lower = 0 if upper >= 0 else -1
            nodes = []
            for node in layer:
                probabilities, successors = _sequence(node, 2, 2, "node")
                probabilities = _row(probabilities, actions, probability=True)
                successors = tuple(
                    tuple(
                        _integer(i, lower, upper, "successor")
                        for i in _sequence(
                            row, observations, observations, "successor row"
                        )
                    )
                    for row in _sequence(successors, actions, actions, "successors")
                )
                nodes.append((probabilities, successors))
            frozen.append(tuple(nodes))
        upper = len(layers[0]) - 1 if layers else -1
        lower = 0 if layers else -1
        initial = tuple(
            _integer(i, lower, upper, "initial node")
            for i in _sequence(
                self.initial_nodes, observations, observations, "initial_nodes"
            )
        )
        object.__setattr__(self, "layers", tuple(frozen))
        object.__setattr__(self, "initial_nodes", initial)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _wire(value):
    if type(value) is F:
        return str(value)
    if type(value) in (tuple, list):
        return [_wire(x) for x in value]
    return value


def evaluate_exact_control(model: FiniteControlModel, policy: PublicController) -> dict:
    """Compute public-history optimum, independent forward return and losses.

    No empirical comparison, experiment permission, or scientific admission is
    granted. Tree-work bounds are checked before any Bellman expansion.
    """
    if type(model) is not FiniteControlModel or type(policy) is not PublicController:
        raise ValueError("validated finite model and public controller required")
    # Reconstruct so even objects corrupted through object.__setattr__ fail closed.
    model = FiniteControlModel(
        model.initial,
        model.transition,
        model.emission,
        model.reward,
        model.initial_reward,
    )
    policy = PublicController(
        policy.actions, policy.observations, policy.initial_nodes, policy.layers
    )
    states, actions, observations = (
        len(model.initial),
        len(model.transition),
        len(model.initial_reward),
    )
    if (actions, observations) != (policy.actions, policy.observations):
        raise ValueError("model/controller alphabet mismatch")
    _check_denominators(model, policy)
    horizon = len(policy.layers)
    expansion = observations * sum(
        (actions * observations) ** t for t in range(horizon + 1)
    )
    if expansion > MAX_TREE_WORK:
        raise ValueError("exact public-history tree work limit exceeded")

    def outcomes(belief, action):
        latent = tuple(
            _sum(belief[s] * model.transition[action][s][z] for s in range(states))
            for z in range(states)
        )
        for observation in range(observations):
            masses = tuple(
                _checked(latent[z] * model.emission[z][observation])
                for z in range(states)
            )
            probability = _sum(masses)
            if probability:
                yield observation, probability, tuple(
                    _checked(x / probability) for x in masses
                )

    @lru_cache(None)
    def optimal(remaining, belief):
        if not remaining:
            return F(0), ()
        q_values = tuple(
            _sum(
                probability
                * (model.reward[a][o] + optimal(remaining - 1, posterior)[0])
                for o, probability, posterior in outcomes(belief, a)
            )
            for a in range(actions)
        )
        return max(q_values), q_values

    initial_branches = []
    initial_value = F(0)
    for o in range(observations):
        masses = tuple(model.initial[s] * model.emission[s][o] for s in range(states))
        probability = sum(masses)
        initial_value = _checked(initial_value + probability * model.initial_reward[o])
        if probability:
            initial_branches.append(
                (o, probability, tuple(x / probability for x in masses))
            )
    optimum = _checked(
        initial_value + _sum(p * optimal(horizon, b)[0] for _, p, b in initial_branches)
    )

    # Physical forward mass propagation never uses Bellman beliefs or value tables.
    occupied = {}
    for o in range(observations):
        for s in range(states):
            key = (policy.initial_nodes[o], s)
            occupied[key] = (
                occupied.get(key, F(0)) + model.initial[s] * model.emission[s][o]
            )
    forward_value = initial_value
    forward_counts = [F(0)] * actions
    for t, layer in enumerate(policy.layers):
        following = {}
        for (node, s), mass in occupied.items():
            probabilities, successors = layer[node]
            for a, selected in enumerate(probabilities):
                forward_counts[a] = _checked(forward_counts[a] + mass * selected)
                for z in range(states):
                    for o in range(observations):
                        weight = _checked(
                            mass
                            * selected
                            * model.transition[a][s][z]
                            * model.emission[z][o]
                        )
                        forward_value = _checked(
                            forward_value + weight * model.reward[a][o]
                        )
                        key = (successors[a][o], z)
                        if weight:
                            following[key] = _checked(following.get(key, F(0)) + weight)
        if _sum(following.values()) != 1:
            raise ArithmeticError("forward probability conservation failed")
        occupied = following

    counts, contributions = [F(0)] * actions, [F(0)] * actions
    visited = 0

    def visit(t, node, belief, occupancy):
        nonlocal visited
        visited += 1
        if t == horizon:
            return
        value, q_values = optimal(horizon - t, belief)
        probabilities, successors = policy.layers[t][node]
        for a, selected in enumerate(probabilities):
            weight = _checked(occupancy * selected)
            if not weight:
                continue
            counts[a] = _checked(counts[a] + weight)
            contributions[a] = _checked(
                contributions[a] + weight * (value - q_values[a])
            )
            for o, probability, posterior in outcomes(belief, a):
                visit(
                    t + 1, successors[a][o], posterior, _checked(weight * probability)
                )

    for o, probability, belief in initial_branches:
        visit(0, policy.initial_nodes[o], belief, probability)
    regret = _checked(optimum - forward_value)
    if (
        counts != forward_counts
        or sum(counts) != horizon
        or sum(contributions) != regret
        or any(c < 0 for c in contributions)
    ):
        raise ArithmeticError("exact performance-difference identity failed")
    model_wire = {
        key: _wire(getattr(model, key))
        for key in ("initial", "transition", "emission", "reward", "initial_reward")
    }
    policy_wire = {
        key: _wire(getattr(policy, key))
        for key in ("actions", "observations", "initial_nodes", "layers")
    }
    report = {
        "schema": "xscientist.exact-control-report.v1",
        "verification_scope": "exact_supplied_model_public_controller",
        "time_contract": "K_actions_K_plus_1_observations_initial_reward_included",
        "action_horizon": horizon,
        "observation_horizon": horizon + 1,
        "model_hash": _hash(model_wire),
        "policy_hash": _hash(policy_wire),
        "evaluator_hash": "sha256:"
        + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_value": str(forward_value),
        "public_history_optimal_value": str(optimum),
        "regret": str(regret),
        "regret_per_action": str(_checked(regret / horizon)) if horizon else None,
        "action_diagnostics": [
            {
                "action": a,
                "expected_count": str(counts[a]),
                "regret_contribution": str(contributions[a]),
                "conditional_optimal_action_loss": (
                    str(_checked(contributions[a] / counts[a])) if counts[a] else None
                ),
            }
            for a in range(actions)
        ],
        "identity_verified_exactly": True,
        "oracle_information": "public_history_only",
        "oracle_belief_subproblems": optimal.cache_info().currsize,
        "learner_history_visits": visited,
        "tree_work_bound": expansion,
        "environment_provenance_verified": False,
        "ground_truth_authority": "supplied_model_unverified",
        "comparison_ready": False,
        "experiment_authorized": False,
        "training_authorized": False,
        "claim_promotion_allowed": False,
        "independent_efficacy_endpoints": 1,
    }
    report["report_hash"] = _hash(report)
    return report
