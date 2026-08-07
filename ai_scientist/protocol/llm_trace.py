"""LLM call tracing — writes provenance rows to ``<ara>/llm/calls.jsonl``.

Callers wrap this around every model invocation. When no ARA root is active
(env var unset or pointing nowhere), :func:`record_llm_call` is a no-op —
so wiring the tracer into ``llm.py`` cannot regress the many code paths that
run outside a pipeline (unit tests, ad-hoc scripts, ``get_response_from_llm``
called from library consumers).

Design points worth flagging for future readers:

* The redacted messages payload is stored in the CAS. Two calls with
  the same system prompt + conversation history therefore hash to the same
  ``messages_ref`` and store a single blob — the ``calls.jsonl`` row is
  what ties one particular invocation to a wall-clock time and stage.
* Redaction is mandatory. CAS is append-only, so an environment toggle must
  never be able to persist a credential, email address, host path, or machine
  identifier irreversibly.
* :func:`_clean_params` only keeps knobs whose values are JSON-serialisable.
  It deliberately drops function references, model client objects, etc. —
  those routinely arrive in the params dict when a caller is lazy, and none
  of them belong in provenance.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
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
# Deprecated compatibility name.  Redaction is now unconditional; setting the
# variable to ``0`` has no effect because persistent traces must fail closed.
ENV_REDACT = "AI_SCIENTIST_LLM_REDACT"

CALLS_JSONL_RELPATH = os.path.join("llm", "calls.jsonl")

# In-thread/task buffer that collects messages_ref.hash values recorded while
# a capture_llm_calls() block is active. Uses a ContextVar so nested captures
# and asyncio tasks each get their own list without leaking into siblings.
# The invariant: when None, we're outside any capture (record_llm_call still
# writes to disk, it just doesn't buffer). When a list, every call written by
# record_llm_call also appends its messages_ref.hash to that list.
_capture_buffer: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_capture_buffer", default=None
)

# Params we consider "hash-worthy" — everything else is dropped as noise.
_KEEP_PARAM_KEYS = frozenset(
    {
        "temperature",
        "max_tokens",
        "seed",
        "top_p",
        "top_k",
        "response_format",
        "tool_choice",
        "stop",
        "reasoning_effort",
    }
)


def active_ara_root() -> str | None:
    """Return the ARA root the tracer should write into, or None to disable.

    The env var opt-out (``AI_SCIENTIST_LLM_TRACE=0``) beats everything: it
    lets a developer silence the tracer for a single subprocess without
    editing code.
    """
    if str(os.environ.get(ENV_ENABLED, "1")).strip() == "0":
        return None
    root = os.environ.get(ENV_ACTIVE_ROOT)
    if not root:
        return None
    if not os.path.isdir(root):
        return None
    return root


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v, default=str)
        return True
    except Exception:
        return False


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    return {
        k: params[k] for k in _KEEP_PARAM_KEYS if k in params and _jsonable(params[k])
    }


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


def record_llm_call(
    *,
    provider: str,
    model: str,
    request_style: str,
    system_message: str,
    messages: list[dict[str, Any]],
    response_text: str,
    params: dict[str, Any] | None = None,
    tokens: dict[str, int] | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
    stage: str | None = None,
) -> str | None:
    """Append one row to ``<ara>/llm/calls.jsonl``.

    Returns the ``call_id`` on success, or None when tracing is inactive.
    Never raises: a broken tracer must not crash the model call itself.
    """
    root = active_ara_root()
    if not root:
        return None

    try:
        store = ObjectStore(root)
        call_id = uuid.uuid4().hex

        msg_payload = _redact({"system": system_message, "messages": messages})
        response_text = _redact_string(response_text or "")

        messages_ref = store.put_json(msg_payload).to_json()
        response_ref = store.put_text(response_text or "").to_json()

        row = _redact(
            {
                "schema_version": "ara.v1",
                "protocol_kind": "llm_call",
                "call_id": call_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage": stage if stage is not None else os.environ.get(ENV_STAGE),
                "provider": provider,
                "model": model,
                "request_style": request_style,
                "params": _clean_params(params),
                "messages_ref": messages_ref,
                "response_ref": response_ref,
                "tokens": tokens or {},
                "latency_ms": latency_ms,
                "error": error,
            }
        )

        log_path = os.path.join(root, CALLS_JSONL_RELPATH)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        # If a capture block is active in this context, publish the
        # messages_ref.hash so the caller (e.g. parallel_agent._draft) can
        # bind it into Node.llm_call_refs.
        buf = _capture_buffer.get()
        if buf is not None:
            buf.append(messages_ref["hash"])

        return call_id
    except Exception:
        # Tracing is best-effort; never break the real LLM path.
        return None


@contextlib.contextmanager
def capture_llm_calls() -> Iterator[list[str]]:
    """Collect the ``messages_ref.hash`` of every LLM call in this block.

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
    "ENV_REDACT",
    "CALLS_JSONL_RELPATH",
    "active_ara_root",
    "capture_llm_calls",
    "record_llm_call",
]
