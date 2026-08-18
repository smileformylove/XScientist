"""Concurrency-safe, pre-call budgets for LLM requests."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - Windows fallback
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

try:  # pragma: no cover - POSIX fallback
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


STATE_VERSION = "llm_budget.v1"
DEFAULT_MAX_OUTPUT_TOKENS = 8192

# USD per one million tokens. Unknown models are not treated as free when a
# cost limit is enabled; users must provide an explicit override.
DEFAULT_MODEL_PRICES_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-2025-04-14": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-2024-12-17": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-preview-2024-09-12": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-mini": {"input": 3.00, "cached_input": 1.50, "output": 12.00},
    "o1-mini-2024-09-12": {"input": 3.00, "cached_input": 1.50, "output": 12.00},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "o3-mini-2025-01-31": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "claude-3-5-sonnet": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "cached_input": 0.03, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "cached_input": 1.50, "output": 75.00},
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input": 0.25,
        "cached_input": 0.03,
        "output": 1.25,
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "input": 15.00,
        "cached_input": 1.50,
        "output": 75.00,
    },
}


class LLMBudgetExceeded(RuntimeError):
    def __init__(self, dimension: str, message: str, snapshot: Mapping[str, Any]):
        super().__init__(message)
        self.dimension = dimension
        self.snapshot = dict(snapshot)

    def __reduce__(self):
        return (type(self), (self.dimension, str(self), self.snapshot))


class UnknownModelPriceError(LLMBudgetExceeded):
    pass


class LLMBudgetStateError(RuntimeError):
    pass


def _find_llm_budget_exception(
    exc: BaseException | None,
) -> LLMBudgetExceeded | None:
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, LLMBudgetExceeded):
            return current
        current = current.__cause__ or current.__context__
    return None


def is_llm_budget_exception(exc: BaseException | None) -> bool:
    """Recognize budget failures, including exceptions crossing process boundaries."""

    if _find_llm_budget_exception(exc) is not None:
        return True
    text = str(exc or "").lower()
    return (
        "llm" in text and "budget" in text and ("exhaust" in text or "exceed" in text)
    ) or "no price is configured for model" in text


def llm_budget_exception_payload(exc: BaseException) -> dict[str, Any]:
    budget_exc = _find_llm_budget_exception(exc)
    if budget_exc is not None:
        return {
            "type": type(budget_exc).__name__,
            "dimension": budget_exc.dimension,
            "message": str(budget_exc),
            "snapshot": dict(budget_exc.snapshot),
        }
    return {
        "type": type(exc).__name__,
        "dimension": "unknown",
        "message": str(exc),
        "snapshot": llm_budget_manager.snapshot(),
    }


def _nonnegative_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("LLM token budgets must be >= 0")
    return parsed


def _nonnegative_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if parsed < 0:
        raise ValueError("LLM cost/wall-time budgets must be >= 0")
    return parsed


def estimate_tokens(value: Any, model: str | None = None) -> int:
    if value is None:
        return 0
    low_detail_images = 0
    unbounded_image_bytes = 0
    model_name = str(model or "").split("/", 1)[-1]

    def normalise(current: Any) -> Any:
        nonlocal low_detail_images, unbounded_image_bytes
        if isinstance(current, dict):
            url = current.get("url")
            if isinstance(url, str) and url.startswith("data:image/"):
                if current.get("detail") == "low":
                    low_detail_images += 1
                else:
                    unbounded_image_bytes += len(url.encode("utf-8", errors="replace"))
                return {**current, "url": "[inline-image]"}
            return {str(key): normalise(item) for key, item in current.items()}
        if isinstance(current, (list, tuple)):
            return [normalise(item) for item in current]
        return current

    if not isinstance(value, str):
        value = json.dumps(
            normalise(value), ensure_ascii=False, sort_keys=True, default=str
        )
    if not value:
        return low_detail_images * 4000 + unbounded_image_bytes
    byte_count = len(value.encode("utf-8", errors="replace"))
    try:
        import tiktoken

        if model_name.startswith(("gpt-", "o1", "o3", "o4")):
            encoding = tiktoken.encoding_for_model(model_name)
            text_tokens = len(encoding.encode(value)) + 128
        else:
            text_tokens = byte_count + 64
    except Exception:
        text_tokens = byte_count + 64
    return text_tokens + low_detail_images * 4000 + unbounded_image_bytes


def extract_usage_tokens(response: Any) -> tuple[int, int, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) if details else 0
    if input_tokens is None and output_tokens is None:
        return None
    return int(input_tokens or 0), int(output_tokens or 0), int(cached_tokens or 0)


def _model_candidates(model: str) -> list[str]:
    raw = str(model or "").strip()
    candidates = [raw]
    if "/" in raw:
        candidates.append(raw.split("/", 1)[1])
    if "@" in raw:
        candidates.append(raw.split("@", 1)[0])
    aliases = (
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
        "o3-mini",
        "o1-mini",
        "o1",
        "claude-3-5-sonnet",
        "claude-3-haiku",
        "claude-3-opus",
        "deepseek-chat",
    )
    for alias in aliases:
        if any(candidate.startswith(alias) for candidate in candidates):
            candidates.append(alias)
    if any("deepseek-coder-v2" in candidate for candidate in candidates):
        candidates.append("deepseek-coder-v2-0724")
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def resolve_model_price(
    model: str,
    *,
    prices_per_million: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float] | None:
    """Resolve a complete, non-negative price without treating unknown models as free."""

    if str(model or "").strip().lower().startswith("ollama/"):
        return {"input": 0.0, "cached_input": 0.0, "output": 0.0}

    prices: dict[str, Mapping[str, Any]] = dict(DEFAULT_MODEL_PRICES_PER_MILLION)
    prices.update(dict(prices_per_million or {}))
    for candidate in _model_candidates(model):
        raw = prices.get(candidate)
        if not isinstance(raw, Mapping):
            continue
        try:
            parsed = {str(key): float(value) for key, value in raw.items()}
        except (TypeError, ValueError):
            continue
        if (
            "input" in parsed
            and "output" in parsed
            and parsed["input"] >= 0
            and parsed["output"] >= 0
            and parsed.get("cached_input", parsed["input"]) >= 0
        ):
            return parsed
    return None


@dataclass(frozen=True)
class LLMBudgetLimits:
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_time_seconds: float | None = None
    prices_per_million: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_total_tokens,
                self.max_cost_usd,
                self.max_wall_time_seconds,
            )
        )


class BudgetReservation:
    def __init__(
        self,
        manager: "LLMBudgetManager",
        reservation_id: str,
        timeout_seconds: float | None,
    ) -> None:
        self.manager = manager
        self.reservation_id = reservation_id
        self.timeout_seconds = timeout_seconds
        self._settled = False

    def settle(
        self,
        *,
        response: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int = 0,
    ) -> None:
        if self._settled:
            return
        usage = extract_usage_tokens(response)
        if usage is not None:
            input_tokens, output_tokens, cached_tokens = usage
        self.manager._settle(
            self.reservation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            charge_reserved_when_unknown=True,
        )
        self._settled = True

    def cancel(self, *, charge_reserved: bool = True) -> None:
        if self._settled:
            return
        self.manager._settle(
            self.reservation_id,
            input_tokens=None,
            output_tokens=None,
            cached_tokens=0,
            charge_reserved_when_unknown=charge_reserved,
        )
        self._settled = True

    def __enter__(self) -> "BudgetReservation":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._settled:
            self.cancel(charge_reserved=True)
        return False


class LLMBudgetManager:
    def __init__(self) -> None:
        self._thread_lock = threading.RLock()
        self._limits = LLMBudgetLimits()
        self._state_path: Path | None = None
        self._memory_state: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self._limits.enabled

    def configure(
        self,
        *,
        max_total_tokens: int | None = None,
        max_cost_usd: float | None = None,
        max_wall_time_seconds: float | None = None,
        prices_per_million: Mapping[str, Mapping[str, Any]] | None = None,
        state_path: str | Path | None = None,
        reset: bool = False,
        allow_limit_increase: bool = False,
        reclaim_active_reservations: bool = False,
    ) -> None:
        custom_prices: dict[str, dict[str, float]] = {}
        for model, prices in (prices_per_million or {}).items():
            parsed = {
                key: float(value)
                for key, value in dict(prices).items()
                if key in {"input", "cached_input", "output"} and float(value) >= 0
            }
            if "input" in parsed and "output" in parsed:
                custom_prices[str(model)] = parsed
        self._limits = LLMBudgetLimits(
            max_total_tokens=_nonnegative_int(max_total_tokens),
            max_cost_usd=_nonnegative_float(max_cost_usd),
            max_wall_time_seconds=_nonnegative_float(max_wall_time_seconds),
            prices_per_million=custom_prices,
        )
        self._state_path = Path(state_path).resolve() if state_path else None
        if self._state_path:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.reset(remove_state=True)
        if self.enabled:
            with self._locked_state() as state:
                self._sync_limits(state, allow_limit_increase=allow_limit_increase)
                if allow_limit_increase:
                    self._resume_wall_clock(state)
                if reclaim_active_reservations:
                    self._reclaim_active_reservations(state)

    def configure_from_env(self) -> None:
        prices: dict[str, Any] = {}
        raw = str(os.environ.get("AI_SCIENTIST_LLM_PRICES_JSON") or "").strip()
        if raw:
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    prices = loaded
            except json.JSONDecodeError:
                pass
        self.configure(
            max_total_tokens=os.environ.get("AI_SCIENTIST_LLM_MAX_TOTAL_TOKENS"),
            max_cost_usd=os.environ.get("AI_SCIENTIST_LLM_MAX_COST_USD"),
            max_wall_time_seconds=os.environ.get(
                "AI_SCIENTIST_LLM_MAX_WALL_TIME_SECONDS"
            ),
            prices_per_million=prices,
            state_path=os.environ.get("AI_SCIENTIST_LLM_BUDGET_STATE"),
        )

    def export_environment(self) -> None:
        values = {
            "AI_SCIENTIST_LLM_MAX_TOTAL_TOKENS": self._limits.max_total_tokens,
            "AI_SCIENTIST_LLM_MAX_COST_USD": self._limits.max_cost_usd,
            "AI_SCIENTIST_LLM_MAX_WALL_TIME_SECONDS": self._limits.max_wall_time_seconds,
            "AI_SCIENTIST_LLM_BUDGET_STATE": (
                str(self._state_path) if self._state_path else None
            ),
            "AI_SCIENTIST_LLM_PRICES_JSON": (
                json.dumps(self._limits.prices_per_million, sort_keys=True)
                if self._limits.prices_per_million
                else None
            ),
        }
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)

    def reset(self, *, remove_state: bool = False) -> None:
        with self._thread_lock:
            self._memory_state = None
            if remove_state and self._state_path:
                self._state_path.unlink(missing_ok=True)

    def _new_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "started_at": time.time(),
            "updated_at": time.time(),
            "limits": {},
            "used": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            },
            "reserved": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            },
            "per_model": {},
            "active": {},
            "denials": [],
        }

    @staticmethod
    def _limit_is_increase(old: Any, new: Any) -> bool:
        if old is None:
            return new is None
        if new is None:
            return True
        return float(new) >= float(old)

    def _sync_limits(
        self, state: dict[str, Any], *, allow_limit_increase: bool = False
    ) -> None:
        expected = {
            "max_total_tokens": self._limits.max_total_tokens,
            "max_cost_usd": self._limits.max_cost_usd,
            "max_wall_time_seconds": self._limits.max_wall_time_seconds,
            "price_schedule_hash": self._price_schedule_hash(),
        }
        existing = state.get("limits") or {}
        if existing and existing != expected:
            can_increase = (
                allow_limit_increase
                and existing.get("price_schedule_hash")
                == expected.get("price_schedule_hash")
                and all(
                    self._limit_is_increase(existing.get(key), expected.get(key))
                    for key in (
                        "max_total_tokens",
                        "max_cost_usd",
                        "max_wall_time_seconds",
                    )
                )
            )
            if not can_increase:
                raise LLMBudgetStateError(
                    "LLM budget policy differs from the existing shared state"
                )
        state["limits"] = expected

    @staticmethod
    def _resume_wall_clock(state: dict[str, Any]) -> None:
        """Exclude time spent stopped between the last state write and resume."""

        elapsed_at_last_update = max(
            0.0,
            float(state.get("updated_at") or time.time()) - float(state["started_at"]),
        )
        state["started_at"] = time.time() - elapsed_at_last_update

    def _reclaim_active_reservations(self, state: dict[str, Any]) -> None:
        """Charge orphaned in-flight calls conservatively before resuming a run."""

        active = list(state.get("active", {}).items())
        for reservation_id, reservation in active:
            state["active"].pop(reservation_id, None)
            for key in ("input_tokens", "output_tokens"):
                state["reserved"][key] -= reservation[key]
                state["used"][key] += reservation[key]
            state["reserved"]["cost_usd"] -= reservation["cost_usd"]
            state["reserved"]["requests"] -= 1
            state["used"]["cost_usd"] += reservation["cost_usd"]
            state["used"]["requests"] += 1
            per_model = state["per_model"].setdefault(
                reservation["model"],
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "requests": 0,
                },
            )
            per_model["input_tokens"] += reservation["input_tokens"]
            per_model["output_tokens"] += reservation["output_tokens"]
            per_model["cost_usd"] += reservation["cost_usd"]
            per_model["requests"] += 1
        if active:
            state["denials"].append(
                {
                    "at": time.time(),
                    "dimension": "resume",
                    "model": "multiple",
                    "reason": (
                        f"charged {len(active)} orphaned active reservation(s) "
                        "while resuming"
                    ),
                }
            )
            del state["denials"][:-100]

    def _price_schedule_hash(self) -> str:
        prices = dict(DEFAULT_MODEL_PRICES_PER_MILLION)
        prices.update(self._limits.prices_per_million)
        encoded = json.dumps(prices, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @contextlib.contextmanager
    def _locked_state(self):
        with self._thread_lock:
            if self._state_path is None:
                if self._memory_state is None:
                    self._memory_state = self._new_state()
                try:
                    yield self._memory_state
                finally:
                    self._memory_state["updated_at"] = time.time()
                return
            lock_path = self._state_path.with_suffix(self._state_path.suffix + ".lock")
            with lock_path.open("a+b") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                elif msvcrt is not None:  # pragma: no cover - exercised on Windows
                    lock_handle.seek(0)
                    if not lock_handle.read(1):
                        lock_handle.write(b"\0")
                        lock_handle.flush()
                    lock_handle.seek(0)
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    try:
                        raw_state = self._state_path.read_text(encoding="utf-8")
                    except FileNotFoundError:
                        state = self._new_state()
                    except OSError as exc:
                        raise LLMBudgetStateError(
                            f"LLM budget state is unreadable: {exc}"
                        ) from exc
                    else:
                        try:
                            state = json.loads(raw_state)
                        except json.JSONDecodeError as exc:
                            raise LLMBudgetStateError(
                                "LLM budget state is corrupted; refusing to reset usage"
                            ) from exc
                        self._validate_state(state)
                    try:
                        yield state
                    finally:
                        state["updated_at"] = time.time()
                        temp = self._state_path.with_suffix(
                            self._state_path.suffix + ".tmp"
                        )
                        temp.write_text(
                            json.dumps(state, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        temp.replace(self._state_path)
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    elif msvcrt is not None:  # pragma: no cover - Windows
                        lock_handle.seek(0)
                        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)

    def _price_for(self, model: str) -> dict[str, float] | None:
        return resolve_model_price(
            model,
            prices_per_million=self._limits.prices_per_million,
        )

    def _validate_state(self, state: Any) -> None:
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            raise LLMBudgetStateError("LLM budget state has an unsupported structure")
        for key in ("used", "reserved", "per_model", "active", "denials"):
            if key not in state:
                raise LLMBudgetStateError(
                    f"LLM budget state is missing required field {key!r}"
                )
        for bucket in ("used", "reserved"):
            if not isinstance(state[bucket], dict) or not all(
                key in state[bucket]
                for key in ("input_tokens", "output_tokens", "cost_usd", "requests")
            ):
                raise LLMBudgetStateError(
                    f"LLM budget state has an invalid {bucket!r} bucket"
                )

    def estimate_cost(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float | None:
        price = self._price_for(model)
        if price is None:
            return None
        cached = min(max(0, cached_tokens), max(0, input_tokens))
        uncached = max(0, input_tokens - cached)
        cached_rate = price.get("cached_input", price["input"])
        return (
            uncached * price["input"]
            + cached * cached_rate
            + max(0, output_tokens) * price["output"]
        ) / 1_000_000

    def reserve(
        self,
        *,
        model: str,
        prompt: Any,
        system_message: Any = None,
        max_output_tokens: int | None = None,
        output_multiplier: int = 1,
    ) -> BudgetReservation:
        if not self.enabled:
            return BudgetReservation(self, "", None)
        input_tokens = estimate_tokens(
            {"system": system_message, "prompt": prompt}, model=model
        )
        output_tokens = max(
            1, int(max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS)
        ) * max(1, int(output_multiplier))
        reserved_cost = self.estimate_cost(
            model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        if self._limits.max_cost_usd is not None and reserved_cost is None:
            raise UnknownModelPriceError(
                "cost",
                "No price is configured for model "
                + repr(model)
                + "; add llm_budget.prices_per_million.",
                self.snapshot(),
            )

        reservation_id = uuid.uuid4().hex
        with self._locked_state() as state:
            self._sync_limits(state)
            elapsed = max(0.0, time.time() - state["started_at"])
            remaining_wall = None
            if self._limits.max_wall_time_seconds is not None:
                remaining_wall = self._limits.max_wall_time_seconds - elapsed
                if remaining_wall <= 0:
                    self._deny(state, "wall_time", model, "wall-time exhausted")
                    raise LLMBudgetExceeded(
                        "wall_time",
                        "LLM wall-time budget is exhausted",
                        self._snapshot(state),
                    )
            projected_tokens = (
                state["used"]["input_tokens"]
                + state["used"]["output_tokens"]
                + state["reserved"]["input_tokens"]
                + state["reserved"]["output_tokens"]
                + input_tokens
                + output_tokens
            )
            if (
                self._limits.max_total_tokens is not None
                and projected_tokens > self._limits.max_total_tokens
            ):
                self._deny(state, "tokens", model, "reservation exceeds limit")
                raise LLMBudgetExceeded(
                    "tokens",
                    "LLM token budget would be exceeded: projected "
                    + str(projected_tokens)
                    + " > "
                    + str(self._limits.max_total_tokens),
                    self._snapshot(state),
                )
            projected_cost = (
                state["used"]["cost_usd"]
                + state["reserved"]["cost_usd"]
                + float(reserved_cost or 0.0)
            )
            if (
                self._limits.max_cost_usd is not None
                and projected_cost > self._limits.max_cost_usd + 1e-12
            ):
                self._deny(state, "cost", model, "reservation exceeds limit")
                raise LLMBudgetExceeded(
                    "cost",
                    "LLM cost budget would be exceeded: projected "
                    + format(projected_cost, ".6f")
                    + " USD > "
                    + format(self._limits.max_cost_usd, ".6f")
                    + " USD",
                    self._snapshot(state),
                )
            reservation = {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": float(reserved_cost or 0.0),
            }
            state["active"][reservation_id] = reservation
            for key in ("input_tokens", "output_tokens"):
                state["reserved"][key] += reservation[key]
            state["reserved"]["cost_usd"] += reservation["cost_usd"]
            state["reserved"]["requests"] += 1
            timeout_seconds = (
                max(0.001, remaining_wall) if remaining_wall is not None else None
            )
        return BudgetReservation(self, reservation_id, timeout_seconds)

    def _settle(
        self,
        reservation_id: str,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_tokens: int,
        charge_reserved_when_unknown: bool,
    ) -> None:
        if not reservation_id or not self.enabled:
            return
        with self._locked_state() as state:
            reservation = state["active"].pop(reservation_id, None)
            if not reservation:
                return
            for key in ("input_tokens", "output_tokens"):
                state["reserved"][key] -= reservation[key]
            state["reserved"]["cost_usd"] -= reservation["cost_usd"]
            state["reserved"]["requests"] -= 1
            if input_tokens is None or output_tokens is None:
                if not charge_reserved_when_unknown:
                    return
                actual_input = reservation["input_tokens"]
                actual_output = reservation["output_tokens"]
                actual_cost = reservation["cost_usd"]
            else:
                actual_input = max(0, int(input_tokens))
                actual_output = max(0, int(output_tokens))
                actual_cost = self.estimate_cost(
                    reservation["model"],
                    input_tokens=actual_input,
                    output_tokens=actual_output,
                    cached_tokens=cached_tokens,
                )
                actual_cost = float(actual_cost or 0.0)
            state["used"]["input_tokens"] += actual_input
            state["used"]["output_tokens"] += actual_output
            state["used"]["cost_usd"] += actual_cost
            state["used"]["requests"] += 1
            per_model = state["per_model"].setdefault(
                reservation["model"],
                {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "requests": 0},
            )
            per_model["input_tokens"] += actual_input
            per_model["output_tokens"] += actual_output
            per_model["cost_usd"] += actual_cost
            per_model["requests"] += 1

    def _deny(
        self, state: dict[str, Any], dimension: str, model: str, reason: str
    ) -> None:
        state["denials"].append(
            {
                "at": time.time(),
                "dimension": dimension,
                "model": model,
                "reason": reason,
            }
        )
        del state["denials"][:-100]

    def _snapshot(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "version": state["version"],
            "elapsed_seconds": max(0.0, time.time() - state["started_at"]),
            "limits": dict(state["limits"]),
            "used": dict(state["used"]),
            "reserved": dict(state["reserved"]),
            "per_model": dict(state["per_model"]),
            "active_reservations": len(state["active"]),
            "denials": list(state["denials"]),
        }

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {"version": STATE_VERSION, "enabled": False}
        with self._locked_state() as state:
            snapshot = self._snapshot(state)
        snapshot["enabled"] = True
        return snapshot


llm_budget_manager = LLMBudgetManager()
llm_budget_manager.configure_from_env()


def configure_llm_budget(
    *,
    max_total_tokens: int | None = None,
    max_cost_usd: float | None = None,
    max_wall_time_seconds: float | None = None,
    prices_per_million: Mapping[str, Mapping[str, Any]] | None = None,
    state_path: str | Path | None = None,
    reset: bool = False,
    allow_limit_increase: bool = False,
    reclaim_active_reservations: bool = False,
) -> LLMBudgetManager:
    llm_budget_manager.configure(
        max_total_tokens=max_total_tokens,
        max_cost_usd=max_cost_usd,
        max_wall_time_seconds=max_wall_time_seconds,
        prices_per_million=prices_per_million,
        state_path=state_path,
        reset=reset,
        allow_limit_increase=allow_limit_increase,
        reclaim_active_reservations=reclaim_active_reservations,
    )
    llm_budget_manager.export_environment()
    return llm_budget_manager
