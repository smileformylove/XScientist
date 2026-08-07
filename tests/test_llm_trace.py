"""Unit tests for the LLM call tracer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import ObjectStore, load_schema, record_llm_call
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_REDACT,
    ENV_STAGE,
    _redact,
    _redact_string,
    active_ara_root,
)
from ai_scientist.protocol.validator import _validate_against_schema, ValidationReport


class _EnvGuard:
    """Snapshot the tracer env vars for a single test and restore on exit."""

    _KEYS = (ENV_ACTIVE_ROOT, ENV_ENABLED, ENV_STAGE, ENV_REDACT)

    def __init__(self) -> None:
        self._snap = {k: os.environ.get(k) for k in self._KEYS}

    def restore(self) -> None:
        for k, v in self._snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class LLMTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._env = _EnvGuard()
        self.addCleanup(self._env.restore)
        # Clean slate — no ARA active unless the test opts in.
        for k in _EnvGuard._KEYS:
            os.environ.pop(k, None)

    def _enable(self, *, redact: bool = False, stage: str | None = None) -> None:
        os.environ[ENV_ACTIVE_ROOT] = str(self.root)
        os.environ[ENV_ENABLED] = "1"
        os.environ[ENV_REDACT] = "1" if redact else "0"
        if stage is not None:
            os.environ[ENV_STAGE] = stage

    # ------------------------------------------------------------------
    # Off by default: no ARA root → no side effects
    # ------------------------------------------------------------------
    def test_no_active_root_is_noop(self) -> None:
        cid = record_llm_call(
            provider="anthropic",
            model="claude-x",
            request_style="anthropic_messages",
            system_message="s",
            messages=[{"role": "user", "content": "hi"}],
            response_text="hello",
        )
        self.assertIsNone(cid)
        self.assertFalse((self.root / CALLS_JSONL_RELPATH).exists())

    def test_explicit_opt_out_is_noop(self) -> None:
        os.environ[ENV_ACTIVE_ROOT] = str(self.root)
        os.environ[ENV_ENABLED] = "0"
        cid = record_llm_call(
            provider="openai",
            model="gpt-x",
            request_style="openai_chat",
            system_message="s",
            messages=[],
            response_text="",
        )
        self.assertIsNone(cid)
        self.assertIsNone(active_ara_root())

    def test_missing_root_dir_is_noop(self) -> None:
        os.environ[ENV_ACTIVE_ROOT] = "/nonexistent/path/does/not/exist/abcxyz"
        os.environ[ENV_ENABLED] = "1"
        self.assertIsNone(active_ara_root())
        cid = record_llm_call(
            provider="openai",
            model="gpt-x",
            request_style="openai_chat",
            system_message="s",
            messages=[],
            response_text="",
        )
        self.assertIsNone(cid)

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------
    def test_writes_row_and_cas_blobs(self) -> None:
        self._enable(stage="review")
        cid = record_llm_call(
            provider="anthropic",
            model="claude-x",
            request_style="anthropic_messages",
            system_message="be terse",
            messages=[{"role": "user", "content": "ping"}],
            response_text="pong",
            params={"temperature": 0.2, "max_tokens": 128, "client": object()},
            tokens={"input": 4, "output": 1},
            latency_ms=42,
        )
        self.assertIsInstance(cid, str)

        rows = self._read_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["call_id"], cid)
        self.assertEqual(row["stage"], "review")
        self.assertEqual(row["provider"], "anthropic")
        self.assertEqual(row["params"], {"temperature": 0.2, "max_tokens": 128})
        self.assertEqual(row["tokens"], {"input": 4, "output": 1})
        self.assertEqual(row["latency_ms"], 42)

        # Blobs must actually exist in the CAS.
        store = ObjectStore(self.root)
        self.assertTrue(store.exists(row["messages_ref"]["hash"]))
        self.assertTrue(store.exists(row["response_ref"]["hash"]))
        self.assertEqual(store.get_text(row["response_ref"]["hash"]), "pong")
        msg = store.get_json(row["messages_ref"]["hash"])
        self.assertEqual(msg["system"], "be terse")
        self.assertEqual(msg["messages"], [{"role": "user", "content": "ping"}])

    def test_identical_calls_dedup_message_blob(self) -> None:
        self._enable()
        kwargs = dict(
            provider="openai",
            model="gpt-x",
            request_style="openai_chat",
            system_message="sys",
            messages=[{"role": "user", "content": "hi"}],
            response_text="hi back",
        )
        c1 = record_llm_call(**kwargs)
        c2 = record_llm_call(**kwargs)
        self.assertNotEqual(c1, c2)  # unique call_ids
        rows = self._read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["messages_ref"]["hash"], rows[1]["messages_ref"]["hash"]
        )

    def test_stage_override_beats_env(self) -> None:
        self._enable(stage="fallback")
        record_llm_call(
            provider="p",
            model="m",
            request_style="r",
            system_message="s",
            messages=[],
            response_text="",
            stage="review",
        )
        [row] = self._read_rows()
        self.assertEqual(row["stage"], "review")

    def test_row_conforms_to_schema(self) -> None:
        self._enable(stage="planning")
        record_llm_call(
            provider="anthropic",
            model="m",
            request_style="anthropic_messages",
            system_message="s",
            messages=[{"role": "user", "content": "q"}],
            response_text="a",
        )
        [row] = self._read_rows()
        schema = load_schema("llm_call")
        report = ValidationReport()
        _validate_against_schema(row, schema, path="", report=report)
        self.assertTrue(report.ok, msg=[e.format() for e in report.errors])

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------
    def test_redact_string_masks_common_secrets(self) -> None:
        raw = (
            "here is my key sk-abcdefghijklmnop and my token Bearer xyz123 "
            "and email alice@example.com and api_key=letmein"
        )
        out = _redact_string(raw)
        self.assertNotIn("sk-abcdefghijklmnop", out)
        self.assertNotIn("xyz123", out)
        self.assertNotIn("alice@example.com", out)
        self.assertNotIn("letmein", out)

    def test_redact_recurses_into_json_shapes(self) -> None:
        payload = {
            "messages": [{"role": "user", "content": "sk-XXXXXXXXXXXXXXXX"}],
            "meta": {"contact": "bob@x.com"},
        }
        red = _redact(payload)
        self.assertNotIn("sk-XXXXXXXXXXXXXXXX", json.dumps(red))
        self.assertNotIn("bob@x.com", json.dumps(red))

    def test_redaction_actually_applied_before_cas(self) -> None:
        self._enable(redact=True)
        record_llm_call(
            provider="p",
            model="m",
            request_style="r",
            system_message="sys",
            messages=[{"role": "user", "content": "please use sk-supersecretvalue123"}],
            response_text="also secret: alice@example.com",
        )
        [row] = self._read_rows()
        store = ObjectStore(self.root)
        msg_dump = json.dumps(store.get_json(row["messages_ref"]["hash"]))
        resp_dump = store.get_text(row["response_ref"]["hash"])
        self.assertNotIn("sk-supersecretvalue123", msg_dump)
        self.assertNotIn("alice@example.com", resp_dump)

    def test_redaction_cannot_be_disabled_for_persistent_traces(self) -> None:
        self._enable(redact=False)
        record_llm_call(
            provider="p",
            model="m",
            request_style="r",
            system_message="sys",
            messages=[{"role": "user", "content": "sk-plainvalueXXXXXXXX"}],
            response_text="",
        )
        [row] = self._read_rows()
        store = ObjectStore(self.root)
        msg = json.dumps(store.get_json(row["messages_ref"]["hash"]))
        self.assertNotIn("sk-plainvalueXXXXXXXX", msg)

    def test_redaction_covers_host_paths_and_error_rows(self) -> None:
        self._enable(redact=False)
        private_path = "/" + "Users/private-person/research/input.json"
        record_llm_call(
            provider="p",
            model="m",
            request_style="r",
            system_message=private_path,
            messages=[{"role": "user", "content": private_path}],
            response_text=private_path,
            error=f"request failed while reading {private_path}",
        )
        [row] = self._read_rows()
        store = ObjectStore(self.root)
        persisted = json.dumps(row)
        persisted += json.dumps(store.get_json(row["messages_ref"]["hash"]))
        persisted += store.get_text(row["response_ref"]["hash"])
        self.assertNotIn("private-person", persisted)
        self.assertIn("[REDACTED_PATH]", persisted)

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------
    def test_never_raises_on_bad_input(self) -> None:
        self._enable()
        # Non-serialisable object in messages — record should still not raise.
        cid = record_llm_call(
            provider="p",
            model="m",
            request_style="r",
            system_message="s",
            messages=[{"role": "user", "content": "ok"}, {"weird": object()}],
            response_text="x",
        )
        # Either it silently swallows (returns None) or it succeeds — both are fine.
        self.assertTrue(cid is None or isinstance(cid, str))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _read_rows(self) -> list[dict]:
        p = self.root / CALLS_JSONL_RELPATH
        if not p.exists():
            return []
        return [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line
        ]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
