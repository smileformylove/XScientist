"""Bounded metric recomputation from submitter-asserted realized traces.

The evaluator intentionally consumes *realized* action/reward trajectories.  It
does not simulate an environment, infer hidden state, or authorize an
experiment.  Its purpose is to make a declared control metric reproducible
once a trusted runner has produced the trajectories.

Input schema (``xscientist.control-replay.v1``)::

    {
      "schema": "xscientist.control-replay.v1",
      "protocol_id": "...",
      "horizon": 4,
      "action_space": ["advance", "toggle", "jump"],
      "oracle_reference": {"arm_id": "oracle", "episodes": [...]},
      "zero_data_reference": {"arm_id": "zero_data", "episodes": [...]},
      "arms": [{"arm_id": "candidate", "episodes": [...]}]
    }

Each episode is ``{"episode_id": str, "actions": [...],
"rewards": [finite numbers...]}``.  All arms and both references must have
the same episode IDs in the same order, and every action/reward sequence must
have exactly ``horizon`` entries.

``aligned_step_reward_gap`` is deliberately defined from aligned
realized traces: for each step it subtracts the arm reward from the oracle
reference reward, groups those differences by the action actually selected by
the arm, and weights the per-action means by the arm's realized action
counts. Its scalar equals (V_reference - V_arm) / horizon; it is not the
protocol's unresolved action-occupancy control-regret endpoint or a
counterfactual Q-value estimate. Neither environment nor policy is replayed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "xscientist.control-replay.v1"
REPORT_SCHEMA = "xscientist.control-replay-report.v1"
EVALUATOR_VERSION = "1.0.0"
DEFAULT_REGRET_FLOOR = 1e-6
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_CONTAINER_ITEMS = 100_000
MAX_DEPTH = 16
MAX_STRING_CHARS = 4096
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "protocol_id",
        "horizon",
        "action_space",
        "oracle_reference",
        "zero_data_reference",
        "arms",
    }
)
_REFERENCE_KEYS = frozenset({"arm_id", "episodes"})
_ARM_KEYS = frozenset({"arm_id", "episodes"})
_EPISODE_KEYS = frozenset({"episode_id", "actions", "rewards"})


class ControlEvaluationError(ValueError):
    """A trajectory payload is not safe or unambiguous to evaluate."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by all evaluator hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_payload(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _report_context() -> dict[str, Any]:
    try:
        implementation_hash = (
            "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
    except OSError:
        implementation_hash = None
    return {
        "schema": REPORT_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_hash": implementation_hash,
        "verification_scope": "artifact_internal_consistency",
        "ground_truth_authority": "submitter_asserted_unverified",
        "environment_provenance_verified": False,
        "reference_optimality_verified": False,
        "comparison_ready": False,
        "experiment_authorized": False,
        "training_authorized": False,
        "claim_promotion_allowed": False,
    }


def _validate_json_shape(value: Any) -> None:
    """Bound ordinary JSON values before hashing, copying, or numeric work."""
    pending = [(value, 0)]
    items = 0
    string_chars = 0
    while pending:
        current, depth = pending.pop()
        if depth > MAX_DEPTH:
            raise ControlEvaluationError("input_limit", "input exceeds depth limit")
        if type(current) is dict:
            if any(type(key) is not str for key in current):
                raise ControlEvaluationError("json_type", "object keys must be strings")
            items += len(current) * 2
            if items > MAX_CONTAINER_ITEMS:
                raise ControlEvaluationError("input_limit", "too many container items")
            pending.extend((key, depth + 1) for key in current)
            pending.extend((item, depth + 1) for item in current.values())
        elif type(current) is list:
            items += len(current)
            if items > MAX_CONTAINER_ITEMS:
                raise ControlEvaluationError("input_limit", "too many container items")
            pending.extend((item, depth + 1) for item in current)
        elif type(current) is str:
            string_chars += len(current)
            if len(current) > MAX_STRING_CHARS or string_chars > MAX_JSON_BYTES:
                raise ControlEvaluationError(
                    "input_limit", "input exceeds string limit"
                )
        elif current is not None and type(current) not in (int, float, bool):
            raise ControlEvaluationError("json_type", "only JSON values are accepted")


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _require_mapping(value: Any, code: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlEvaluationError(code, f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(str(item) for item in extra))
        raise ControlEvaluationError(
            "schema_keys", f"{label} has invalid keys ({'; '.join(detail)})"
        )


def _require_nonempty_string(value: Any, code: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlEvaluationError(code, f"{label} must be a non-empty string")
    return value


def _validate_episode(
    value: Any,
    *,
    horizon: int,
    actions: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    episode = _require_mapping(value, "episode_type", label)
    _require_exact_keys(episode, _EPISODE_KEYS, label)
    episode_id = _require_nonempty_string(
        episode["episode_id"], "episode_id", f"{label}.episode_id"
    )
    raw_actions = episode["actions"]
    raw_rewards = episode["rewards"]
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
        raise ControlEvaluationError(
            "actions_type", f"{label}.actions must be an array"
        )
    if isinstance(raw_rewards, (str, bytes)) or not isinstance(raw_rewards, Sequence):
        raise ControlEvaluationError(
            "rewards_type", f"{label}.rewards must be an array"
        )
    if len(raw_actions) != horizon or len(raw_rewards) != horizon:
        raise ControlEvaluationError(
            "horizon_length",
            f"{label} actions/rewards must each have exactly horizon={horizon} entries",
        )
    normalized_actions: list[str] = []
    for index, action in enumerate(raw_actions):
        if not isinstance(action, str) or action not in actions:
            raise ControlEvaluationError(
                "action_domain", f"{label}.actions[{index}] is outside action_space"
            )
        normalized_actions.append(action)
    normalized_rewards: list[float] = []
    for index, reward in enumerate(raw_rewards):
        if not _is_number(reward):
            raise ControlEvaluationError(
                "finite_reward", f"{label}.rewards[{index}] must be finite"
            )
        normalized_rewards.append(float(reward))
    return {
        "episode_id": episode_id,
        "actions": normalized_actions,
        "rewards": normalized_rewards,
    }


def _validate_episode_group(
    value: Any,
    *,
    horizon: int,
    actions: tuple[str, ...],
    label: str,
) -> tuple[str, list[dict[str, Any]]]:
    group = _require_mapping(value, "reference_type", label)
    _require_exact_keys(
        group, _REFERENCE_KEYS if "reference" in label else _ARM_KEYS, label
    )
    arm_id = _require_nonempty_string(group["arm_id"], "arm_id", f"{label}.arm_id")
    raw_episodes = group["episodes"]
    if isinstance(raw_episodes, (str, bytes)) or not isinstance(raw_episodes, Sequence):
        raise ControlEvaluationError(
            "episodes_type", f"{label}.episodes must be an array"
        )
    if not raw_episodes:
        raise ControlEvaluationError(
            "episodes_empty", f"{label}.episodes cannot be empty"
        )
    episodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_episode in enumerate(raw_episodes):
        episode = _validate_episode(
            raw_episode,
            horizon=horizon,
            actions=actions,
            label=f"{label}.episodes[{index}]",
        )
        if episode["episode_id"] in seen:
            raise ControlEvaluationError(
                "duplicate_episode_id",
                f"{label} repeats episode_id={episode['episode_id']}",
            )
        seen.add(episode["episode_id"])
        episodes.append(episode)
    return arm_id, episodes


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ControlEvaluationError(
            "episodes_empty", "at least one episode is required"
        )
    try:
        value = float(math.fsum(values) / len(values))
    except (OverflowError, ValueError):
        raise ControlEvaluationError(
            "nonfinite_metric", "metric aggregation overflowed"
        )
    if not math.isfinite(value):
        raise ControlEvaluationError(
            "nonfinite_metric", "metric aggregation is non-finite"
        )
    return value


def _return_for_episode(episode: Mapping[str, Any], label: str) -> float:
    try:
        value = float(math.fsum(episode["rewards"]))
    except (OverflowError, ValueError):
        raise ControlEvaluationError("return_finite", f"{label} return overflowed")
    if not math.isfinite(value):
        raise ControlEvaluationError("return_finite", f"{label} return is non-finite")
    return value


def _difference(left: float, right: float, label: str) -> float:
    try:
        value = float(left - right)
    except (OverflowError, ValueError):
        raise ControlEvaluationError(
            "nonfinite_metric", f"{label} difference overflowed"
        )
    if not math.isfinite(value):
        raise ControlEvaluationError(
            "nonfinite_metric", f"{label} difference is non-finite"
        )
    return value


def _ratio(numerator: float, denominator: float, label: str) -> float:
    try:
        value = float(numerator / denominator)
    except (OverflowError, ZeroDivisionError, ValueError):
        raise ControlEvaluationError("nonfinite_metric", f"{label} ratio overflowed")
    if not math.isfinite(value):
        raise ControlEvaluationError("nonfinite_metric", f"{label} ratio is non-finite")
    return value


def _invalid_report(code: str, message: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        **_report_context(),
        "status": "invalid",
        "error": {"code": code, "message": message},
    }
    body["report_sha256"] = _sha256_payload(body)
    return body


def evaluate_control_replay(
    payload: Mapping[str, Any],
    *,
    expected_horizon: int | None = None,
    expected_actions: Sequence[str] | None = None,
    expected_protocol_id: str | None = None,
    regret_floor: float = DEFAULT_REGRET_FLOOR,
) -> dict[str, Any]:
    """Validate and deterministically evaluate a control replay payload.

    The function returns a structured ``status=invalid`` report for every
    rejected payload.  It never turns malformed trajectories into metrics.
    Optional expected fields check caller-declared identities only. They do
    not verify a frozen protocol, environment, runner, or reference policy.
    """

    try:
        _validate_json_shape(payload)
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_JSON_BYTES:
            raise ControlEvaluationError("input_limit", "input exceeds byte limit")
        data = _require_mapping(payload, "payload_type", "payload")
        _require_exact_keys(data, _TOP_LEVEL_KEYS, "payload")
        if data["schema"] != SCHEMA:
            raise ControlEvaluationError("schema_version", f"schema must be {SCHEMA}")
        protocol_id = _require_nonempty_string(
            data["protocol_id"], "protocol_id", "protocol_id"
        )
        if expected_protocol_id is not None:
            _require_nonempty_string(
                expected_protocol_id, "protocol_binding", "expected_protocol_id"
            )
            if expected_protocol_id != protocol_id:
                raise ControlEvaluationError(
                    "protocol_binding", "protocol_id does not match"
                )
        horizon = data["horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ControlEvaluationError(
                "horizon", "horizon must be a positive integer"
            )
        if expected_horizon is not None:
            if type(expected_horizon) is not int or expected_horizon <= 0:
                raise ControlEvaluationError(
                    "horizon_binding", "expected_horizon must be a positive integer"
                )
            if horizon != expected_horizon:
                raise ControlEvaluationError(
                    "horizon_binding",
                    f"horizon must equal bound value {expected_horizon}",
                )
        raw_action_space = data["action_space"]
        if isinstance(raw_action_space, (str, bytes)) or not isinstance(
            raw_action_space, Sequence
        ):
            raise ControlEvaluationError(
                "action_space_type", "action_space must be an array"
            )
        if not raw_action_space:
            raise ControlEvaluationError(
                "action_space_empty", "action_space cannot be empty"
            )
        if any(
            not isinstance(action, str) or not action for action in raw_action_space
        ):
            raise ControlEvaluationError(
                "action_space_value", "action_space entries must be strings"
            )
        if len(set(raw_action_space)) != len(raw_action_space):
            raise ControlEvaluationError(
                "action_space_duplicate", "action_space must be unique"
            )
        actions = tuple(raw_action_space)
        if expected_actions is not None:
            if (
                not isinstance(expected_actions, (list, tuple))
                or any(
                    not isinstance(action, str) or not action.strip()
                    for action in expected_actions
                )
                or tuple(expected_actions) != actions
            ):
                raise ControlEvaluationError(
                    "action_space_binding",
                    "action_space does not match expected_actions",
                )
        if not _is_number(regret_floor) or float(regret_floor) != DEFAULT_REGRET_FLOOR:
            raise ControlEvaluationError(
                "regret_floor_binding",
                f"regret_floor is protocol-bound to {DEFAULT_REGRET_FLOOR}",
            )
        floor = float(regret_floor)

        oracle_id, oracle_episodes = _validate_episode_group(
            data["oracle_reference"],
            horizon=horizon,
            actions=actions,
            label="oracle_reference",
        )
        zero_id, zero_episodes = _validate_episode_group(
            data["zero_data_reference"],
            horizon=horizon,
            actions=actions,
            label="zero_data_reference",
        )
        if oracle_id == zero_id:
            raise ControlEvaluationError(
                "arm_id_collision", "reference arm IDs must be distinct"
            )
        oracle_ids = [episode["episode_id"] for episode in oracle_episodes]
        zero_ids = [episode["episode_id"] for episode in zero_episodes]
        if zero_ids != oracle_ids:
            raise ControlEvaluationError(
                "reference_alignment",
                "oracle and zero_data episode IDs/order must match",
            )

        raw_arms = data["arms"]
        if isinstance(raw_arms, (str, bytes)) or not isinstance(raw_arms, Sequence):
            raise ControlEvaluationError("arms_type", "arms must be an array")
        if not raw_arms:
            raise ControlEvaluationError("arms_empty", "at least one arm is required")
        arms: list[tuple[str, list[dict[str, Any]]]] = []
        arm_ids: set[str] = set()
        for index, raw_arm in enumerate(raw_arms):
            arm_id, episodes = _validate_episode_group(
                raw_arm,
                horizon=horizon,
                actions=actions,
                label=f"arms[{index}]",
            )
            if arm_id in arm_ids or arm_id in {oracle_id, zero_id}:
                raise ControlEvaluationError(
                    "arm_id_collision", f"duplicate/reserved arm_id={arm_id}"
                )
            arm_ids.add(arm_id)
            if [episode["episode_id"] for episode in episodes] != oracle_ids:
                raise ControlEvaluationError(
                    "episode_alignment",
                    f"arms[{index}] episode IDs/order do not match references",
                )
            arms.append((arm_id, episodes))

        oracle_by_id = {episode["episode_id"]: episode for episode in oracle_episodes}
        oracle_returns = [
            _return_for_episode(episode, f"oracle_reference/{episode['episode_id']}")
            for episode in oracle_episodes
        ]
        zero_returns = [
            _return_for_episode(episode, f"zero_data_reference/{episode['episode_id']}")
            for episode in zero_episodes
        ]
        oracle_value = _mean(oracle_returns)
        zero_value = _mean(zero_returns)
        denominator = max(
            _difference(oracle_value, zero_value, "oracle-zero_data"), floor
        )

        output_arms: list[dict[str, Any]] = []
        for arm_id, episodes in arms:
            arm_returns = [
                _return_for_episode(episode, f"arm/{arm_id}/{episode['episode_id']}")
                for episode in episodes
            ]
            arm_value = _mean(arm_returns)
            occupancy = {action: 0 for action in actions}
            per_action_deltas = {action: [] for action in actions}
            for episode in episodes:
                oracle_episode = oracle_by_id[episode["episode_id"]]
                for action, arm_reward, oracle_reward in zip(
                    episode["actions"], episode["rewards"], oracle_episode["rewards"]
                ):
                    occupancy[action] += 1
                    per_action_deltas[action].append(
                        _difference(
                            float(oracle_reward),
                            float(arm_reward),
                            f"arm/{arm_id}/{episode['episode_id']}/{action}",
                        )
                    )
            total_steps = horizon * len(episodes)
            if sum(occupancy.values()) != total_steps:
                raise ControlEvaluationError(
                    "occupancy_alignment", "action occupancy does not sum to all steps"
                )
            per_action: dict[str, dict[str, Any]] = {}
            weighted_regret = 0.0
            for action in actions:
                count = occupancy[action]
                if count:
                    action_regret = _mean(per_action_deltas[action])
                    weighted_regret += (count / total_steps) * action_regret
                    if not math.isfinite(weighted_regret):
                        raise ControlEvaluationError(
                            "nonfinite_metric",
                            "occupancy-weighted regret is non-finite",
                        )
                else:
                    action_regret = None
                per_action[action] = {
                    "count": count,
                    "aligned_reward_gap": action_regret,
                }
            output_arms.append(
                {
                    "arm_id": arm_id,
                    "episode_returns": [
                        {"episode_id": episode["episode_id"], "return": value}
                        for episode, value in zip(episodes, arm_returns)
                    ],
                    "action_occupancy": occupancy,
                    "per_action": per_action,
                    "metrics": {
                        "true_environment_value": arm_value,
                        "oracle_normalized_regret": _ratio(
                            _difference(
                                oracle_value, arm_value, f"oracle-arm/{arm_id}"
                            ),
                            denominator,
                            f"oracle-normalized/{arm_id}",
                        ),
                    },
                    "diagnostics": {"aligned_step_reward_gap": float(weighted_regret)},
                }
            )

        body: dict[str, Any] = {
            **_report_context(),
            "status": "verified",
            "bindings": {
                "expected_horizon": expected_horizon,
                "expected_actions": (
                    list(expected_actions) if expected_actions is not None else None
                ),
                "expected_protocol_id": expected_protocol_id,
            },
            "protocol_id": protocol_id,
            "horizon": horizon,
            "action_space": list(actions),
            "episode_ids": oracle_ids,
            "episode_count": len(oracle_ids),
            "references": {
                "oracle_arm_id": oracle_id,
                "zero_data_arm_id": zero_id,
                "oracle_value": oracle_value,
                "zero_data_value": zero_value,
                "normalization_denominator": denominator,
                "regret_floor": floor,
            },
            "arms": output_arms,
            "metric_semantics": {
                "true_environment_value": "mean undiscounted realized reward over horizon",
                "oracle_normalized_regret": "(V_oracle - V_arm) / max(V_oracle - V_zero_data, regret_floor)",
                "aligned_step_reward_gap": "(V_reference - V_arm)/H; descriptive action bins, not counterfactual regret",
            },
            "unsupported_metrics": ["action_occupancy_weighted_regret"],
            "warnings": [
                "Rewards and reference policies are submitter assertions; environment and optimality are unverified.",
                "Matching episode IDs alone does not prove paired worlds, identical inputs, or independent holdouts.",
                "Aligned step reward gap duplicates the mean return gap divided by horizon.",
            ],
            "input_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        }
        if body["evaluator_hash"] is None:
            raise ControlEvaluationError(
                "implementation_hash", "cannot bind evaluator source"
            )
        body["report_sha256"] = _sha256_payload(body)
        return body
    except ControlEvaluationError as exc:
        return _invalid_report(exc.code, exc.message)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        # A malformed non-JSON object must never escape as a metric-bearing result.
        return _invalid_report("invalid_payload", type(exc).__name__)


def evaluate_control_replay_json(
    text: str,
    *,
    expected_horizon: int | None = None,
    expected_actions: Sequence[str] | None = None,
    expected_protocol_id: str | None = None,
    regret_floor: float = DEFAULT_REGRET_FLOOR,
) -> dict[str, Any]:
    """Parse one JSON payload with constants such as NaN rejected."""

    if not isinstance(text, str):
        return _invalid_report("json_type", "JSON input must be text")

    def reject_constant(token: str) -> Any:
        raise ValueError(token)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        if len(text) > MAX_JSON_BYTES or len(text.encode("utf-8")) > MAX_JSON_BYTES:
            return _invalid_report("input_limit", "JSON input exceeds byte limit")
        payload = json.loads(
            text, parse_constant=reject_constant, object_pairs_hook=unique_object
        )
    except (TypeError, ValueError, RecursionError) as exc:
        return _invalid_report("invalid_json", type(exc).__name__)
    return evaluate_control_replay(
        payload,
        expected_horizon=expected_horizon,
        expected_actions=expected_actions,
        expected_protocol_id=expected_protocol_id,
        regret_floor=regret_floor,
    )


__all__ = [
    "ControlEvaluationError",
    "DEFAULT_REGRET_FLOOR",
    "EVALUATOR_VERSION",
    "REPORT_SCHEMA",
    "SCHEMA",
    "canonical_json",
    "evaluate_control_replay",
    "evaluate_control_replay_json",
]
