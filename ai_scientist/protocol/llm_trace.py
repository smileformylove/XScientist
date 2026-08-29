"""LLM call tracing — writes payload-free provenance rows to ``<ara>/llm/calls.jsonl``.

Callers wrap this around every model invocation. When no ARA root is active
(env var unset or pointing nowhere), :func:`record_llm_call` is a no-op —
so wiring the tracer into ``llm.py`` cannot regress the many code paths that
run outside a pipeline (unit tests, ad-hoc scripts, ``get_response_from_llm``
called from library consumers).

Design points worth flagging for future readers:

* CAS stores only small digest envelopes. Prompt, response, code, chain of
  thought, endpoint, and credentials are never persisted by this tracer.
* Redaction is mandatory *before hashing*. This keeps fingerprints portable
  without turning a trace into an oracle for a known credential-shaped value.
* ``call_receipt_ref`` binds the prompt digest, output digest, requested and
  safely reported model, and semantic request parameters. Node provenance uses
  this receipt rather than the prompt hash alone.
* :func:`_clean_params` only keeps knobs whose values are JSON-serialisable.
  It deliberately drops function references, model client objects, etc. —
  those routinely arrive in the params dict when a caller is lazy, and none
  of them belong in provenance.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import math
import os
import time
import uuid
from typing import Any, Iterator

from ai_scientist.utils.privacy import (
    redact_sensitive_payload,
    redact_sensitive_text,
)

from .objects import ObjectStore

ENV_ACTIVE_ROOT = "AI_SCIENTIST_ARA_ACTIVE_ROOT"
ENV_ENABLED = "AI_SCIENTIST_LLM_TRACE"
ENV_STAGE = "AI_SCIENTIST_LLM_STAGE"
ENV_STRICT = "AI_SCIENTIST_LLM_TRACE_STRICT"
# Deprecated compatibility name.  Redaction is now unconditional; setting the
# variable to ``0`` has no effect because persistent traces must fail closed.
ENV_REDACT = "AI_SCIENTIST_LLM_REDACT"

CALLS_JSONL_RELPATH = os.path.join("llm", "calls.jsonl")

# In-thread/task buffer that collects semantic call-receipt hashes while
# a capture_llm_calls() block is active. Uses a ContextVar so nested captures
# and asyncio tasks each get their own list without leaking into siblings.
# The invariant: when None, we're outside any capture (record_llm_call still
# writes to disk, it just doesn't buffer). When a list, every call written by
# record_llm_call also appends its call_receipt_ref.hash to that list.
_capture_buffer: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_capture_buffer", default=None
)

# Params we consider "hash-worthy" — everything else is dropped as noise.
_KEEP_PARAM_KEYS = frozenset(
    {
        "temperature",
        "timeout",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "seed",
        "top_p",
        "top_k",
        "response_format",
        "tool_choice",
        "stop",
        "reasoning_effort",
        "fallback",
        "fallback_from",
        "fallback_reason",
        "actual_model",
        "tool_schema_sha256",
    }
)


class LLMTraceError(RuntimeError):
    """Raised when a strict provenance trace cannot be persisted."""


def strict_llm_tracing() -> bool:
    return str(os.environ.get(ENV_STRICT, "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def active_ara_root() -> str | None:
    """Return the ARA root the tracer should write into, or None to disable.

    The env var opt-out (``AI_SCIENTIST_LLM_TRACE=0``) beats everything: it
    lets a developer silence the tracer for a single subprocess without
    editing code.
    """
    if str(os.environ.get(ENV_ENABLED, "1")).strip() == "0":
        if strict_llm_tracing():
            raise LLMTraceError("strict LLM tracing cannot be disabled")
        return None
    root = os.environ.get(ENV_ACTIVE_ROOT)
    if not root:
        if strict_llm_tracing():
            raise LLMTraceError("strict LLM tracing requires an active ARA root")
        return None
    if not os.path.isdir(root):
        if strict_llm_tracing():
            raise LLMTraceError("strict LLM tracing requires an existing ARA root")
        return None
    return root


def _strict_json_value(
    value: Any,
    *,
    depth: int = 0,
    item_budget: list[int] | None = None,
) -> Any:
    """Return a JSON-native copy without ever stringifying foreign objects."""

    if depth > 24:
        raise TypeError("LLM trace value exceeds the nesting limit")
    if item_budget is None:
        item_budget = [20_000]
    item_budget[0] -= 1
    if item_budget[0] < 0:
        raise TypeError("LLM trace value exceeds the item limit")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("LLM trace value contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(
                item,
                depth=depth + 1,
                item_budget=item_budget,
            )
            for item in value
        ]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("LLM trace object keys must be strings")
        return {
            key: _strict_json_value(
                item,
                depth=depth + 1,
                item_budget=item_budget,
            )
            for key, item in value.items()
        }
    raise TypeError("LLM trace value is not JSON-native")


def _clean_params(
    params: dict[str, Any] | None,
    *,
    fail_on_unsupported: bool,
) -> dict[str, Any]:
    if not params:
        return {}
    cleaned: dict[str, Any] = {}
    for key in _KEEP_PARAM_KEYS:
        if key not in params:
            continue
        try:
            cleaned[key] = _strict_json_value(params[key])
        except TypeError:
            # Provider client objects and custom response-format classes are
            # common here. Drop them without calling repr/str: either may
            # contain credentials or private research data.
            if fail_on_unsupported:
                raise TypeError(
                    f"LLM trace parameter {key} is not JSON-native"
                ) from None
            continue
    return cleaned


def _redact_string(text: str) -> str:
    return redact_sensitive_text(text)


def _redact(payload: Any) -> Any:
    """Recursively scrub known secret shapes from a JSON-ish payload.

    We do this BEFORE the payload lands in CAS because CAS is append-only:
    a leaked prompt can't be un-hashed.
    """
    return redact_sensitive_payload(payload)


def _redaction_enabled() -> bool:
    """Compatibility helper: persistent LLM trace redaction is always enabled."""

    return True


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest_envelope(
    *, kind: str, payload: Any, item_count: int | None = None
) -> dict[str, Any]:
    encoded = _canonical_bytes(payload)
    envelope: dict[str, Any] = {
        "schema": "ara.llm-payload-digest.v1",
        "kind": kind,
        "sha256": _sha256_bytes(encoded),
        "byte_count": len(encoded),
        "payload_recorded": False,
    }
    if item_count is not None:
        envelope["item_count"] = item_count
    return envelope


def _append_json_line(path: str, row: dict[str, Any]) -> None:
    """Append one bounded row with one OS write for process-safe JSONL records."""

    payload = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(payload) > 64 * 1024:
        raise LLMTraceError("LLM trace row exceeds the append limit")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("partial LLM trace append")
    finally:
        os.close(descriptor)


def record_llm_call(
    *,
    provider: str,
    model: str,
    request_style: str,
    system_message: Any,
    messages: list[dict[str, Any]],
    response_text: str,
    params: dict[str, Any] | None = None,
    tokens: dict[str, int] | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
    stage: str | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> str | None:
    """Append one row to ``<ara>/llm/calls.jsonl``.

    Returns the ``call_id`` on success, or None when tracing is inactive.
    Strict mode raises :class:`LLMTraceError` before publishing a usable call
    receipt when provenance cannot be persisted.
    """
    root = active_ara_root()
    if not root:
        return None

    try:
        store = ObjectStore(root)
        call_id = uuid.uuid4().hex

        for field_name, value in {
            "provider": provider,
            "model": model,
            "request_style": request_style,
            "response_text": response_text,
        }.items():
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
        strict_messages = _strict_json_value(messages)
        if not isinstance(strict_messages, list):
            raise TypeError("messages must be a list")
        strict_system_message = _strict_json_value(system_message)
        strict_provenance = _strict_json_value(model_provenance or {})
        if not isinstance(strict_provenance, dict):
            raise TypeError("model_provenance must be an object")
        strict_tokens = _strict_json_value(tokens or {})
        if not isinstance(strict_tokens, dict) or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in strict_tokens.items()
        ):
            raise TypeError("tokens must contain non-negative integer counts")
        if latency_ms is not None and (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, int)
            or latency_ms < 0
        ):
            raise TypeError("latency_ms must be a non-negative integer")
        if error is not None and not isinstance(error, str):
            raise TypeError("error must be a string or null")
        effective_stage = stage if stage is not None else os.environ.get(ENV_STAGE)
        if effective_stage is not None and not isinstance(effective_stage, str):
            raise TypeError("stage must be a string or null")

        msg_payload = _redact(
            {"system": strict_system_message, "messages": strict_messages}
        )
        safe_response_text = _redact_string(response_text)
        safe_params = _clean_params(
            params,
            fail_on_unsupported=strict_llm_tracing(),
        )
        safe_provenance = _redact(strict_provenance)
        safe_provider = _redact_string(provider)
        safe_model = _redact_string(model)
        safe_request_style = _redact_string(request_style)
        safe_stage = _redact_string(effective_stage) if effective_stage else None
        safe_error = _redact_string(error) if error else None

        message_digest = _digest_envelope(
            kind="messages",
            payload=msg_payload,
            item_count=len(messages or []),
        )
        response_digest = _digest_envelope(
            kind="response",
            payload=safe_response_text,
        )
        messages_ref = store.put_json(message_digest).to_json()
        response_ref = store.put_json(response_digest).to_json()
        call_receipt = _redact(
            {
                "schema": "ara.llm-call-receipt.v1",
                "provider": safe_provider,
                "model": safe_model,
                "request_style": safe_request_style,
                "model_provenance": safe_provenance,
                "params": safe_params,
                "messages_sha256": message_digest["sha256"],
                "response_sha256": response_digest["sha256"],
                "messages_ref_hash": messages_ref["hash"],
                "response_ref_hash": response_ref["hash"],
            }
        )
        call_receipt_ref = store.put_json(call_receipt).to_json()

        row = _redact(
            {
                "schema_version": "ara.v1",
                "protocol_kind": "llm_call",
                "trace_format": "digest_receipt_v1",
                "call_id": call_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage": safe_stage,
                "provider": safe_provider,
                "model": safe_model,
                "request_style": safe_request_style,
                "model_provenance": safe_provenance,
                "params": safe_params,
                "messages_ref": messages_ref,
                "response_ref": response_ref,
                "call_receipt_ref": call_receipt_ref,
                "tokens": strict_tokens,
                "latency_ms": latency_ms,
                "error": safe_error,
            }
        )

        log_path = os.path.join(root, CALLS_JSONL_RELPATH)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        _append_json_line(log_path, row)

        # If a capture block is active in this context, publish the
        # semantic call receipt so the caller (e.g. parallel_agent._draft) can
        # bind it into Node.llm_call_refs.
        buf = _capture_buffer.get()
        if buf is not None:
            buf.append(call_receipt_ref["hash"])

        return call_id
    except Exception as exc:
        if strict_llm_tracing():
            raise LLMTraceError(
                f"strict LLM trace persistence failed: {type(exc).__name__}"
            ) from None
        # Exploratory tracing remains best-effort.
        return None


@contextlib.contextmanager
def capture_llm_calls() -> Iterator[list[str]]:
    """Collect the semantic ``call_receipt_ref.hash`` of every call in this block.

    Usage::

        with capture_llm_calls() as refs:
            plan, code = generate_via_llm(...)
        node.llm_call_refs = list(refs)

    Semantics
    ---------
    * The returned list is mutated in place — callers can read it either
      inside the ``with`` block or after it exits.
    * Nesting is supported: an inner ``capture_llm_calls`` collects only
      calls issued during the inner block; those hashes still also land in
      the outer block's list (a hash issued deep in a stack traces up).
    * Uses ContextVar, so concurrent asyncio tasks or threads-with-copies
      each see their own buffer without cross-contamination.
    * When tracing is disabled (env var unset), the list simply stays
      empty; downstream code that treats it as optional still works.
    """
    fresh: list[str] = []
    parent = _capture_buffer.get()
    token = _capture_buffer.set(fresh)
    try:
        yield fresh
    finally:
        _capture_buffer.reset(token)
        # Propagate the inner block's captures up to the surrounding block
        # so callers stacking captures still get a full transitive list.
        if parent is not None:
            parent.extend(fresh)


__all__ = [
    "ENV_ACTIVE_ROOT",
    "ENV_ENABLED",
    "ENV_STAGE",
    "ENV_STRICT",
    "ENV_REDACT",
    "CALLS_JSONL_RELPATH",
    "active_ara_root",
    "capture_llm_calls",
    "LLMTraceError",
    "record_llm_call",
    "strict_llm_tracing",
]
