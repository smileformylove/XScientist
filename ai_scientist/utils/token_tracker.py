# Modified by XScientist contributors from the AI-Scientist-v2/AIDE lineage.
# See THIRD_PARTY_NOTICES.md for provenance and license details.
from functools import wraps
import inspect
from typing import Any, Dict, Optional, List, Tuple
from collections import defaultdict
import asyncio
from datetime import datetime, timezone
import logging

from ai_scientist.utils.llm_budget import (
    DEFAULT_MODEL_PRICES_PER_MILLION,
    llm_budget_manager,
)


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        self.MODEL_PRICES = DEFAULT_MODEL_PRICES_PER_MILLION

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def add_interaction(
        self,
        model: str,
        system_message: Any,
        prompt: Any,
        response: str,
        timestamp: Any,
    ):
        """Record a single interaction with the model."""
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float | None:
        """Calculate the cost for a specific model based on token usage."""
        tokens = self.token_counts[model]
        cost = llm_budget_manager.estimate_cost(
            model,
            input_tokens=tokens["prompt"],
            output_tokens=tokens["completion"],
            cached_tokens=tokens["cached"],
        )
        if cost is None:
            logging.warning(f"Price information not available for model {model}")
            return None
        return cost

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        # return dict(self.token_counts)
        """Get summary of token usage and costs for all models."""
        summary = {}
        for model, tokens in self.token_counts.items():
            cost = self.calculate_cost(model)
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": cost,
                "priced": cost is not None,
            }
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def _extract_prompt_and_system(
    func, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, Any]:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        return bound.arguments.get("prompt"), bound.arguments.get("system_message")
    except (TypeError, ValueError):
        return kwargs.get("prompt"), kwargs.get("system_message")


def _extract_usage(result: Any) -> Optional[Tuple[int, int, int, int]]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0
    ) or 0

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = (
        getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
    ) or 0

    return (
        int(prompt_tokens),
        int(completion_tokens),
        int(reasoning_tokens),
        int(cached_tokens),
    )


def _extract_response_text(result: Any) -> str:
    choices = getattr(result, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "")
        return content if isinstance(content, str) else str(content)

    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    if content is not None:
        return str(content)
    return ""


def _record_token_usage(result: Any, prompt: Any, system_message: Any) -> None:
    usage_tuple = _extract_usage(result)
    if usage_tuple is None:
        return

    model = str(getattr(result, "model", "unknown"))
    timestamp = getattr(result, "created", datetime.now(timezone.utc))
    prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens = usage_tuple

    token_tracker.add_tokens(
        model,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        cached_tokens,
    )
    token_tracker.add_interaction(
        model,
        system_message,
        prompt,
        _extract_response_text(result),
        timestamp,
    )


def track_token_usage(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt, system_message = _extract_prompt_and_system(func, args, kwargs)

        result = await func(*args, **kwargs)
        _record_token_usage(result, prompt=prompt, system_message=system_message)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        prompt, system_message = _extract_prompt_and_system(func, args, kwargs)
        result = func(*args, **kwargs)
        _record_token_usage(result, prompt=prompt, system_message=system_message)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
