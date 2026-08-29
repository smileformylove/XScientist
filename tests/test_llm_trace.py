"""Unit tests for the LLM call tracer."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.protocol import ObjectStore, load_schema, record_llm_call
from ai_scientist.protocol.llm_trace import (
    CALLS_JSONL_RELPATH,
    ENV_ACTIVE_ROOT,
    ENV_ENABLED,
    ENV_REDACT,
    ENV_STAGE,
    ENV_STRICT,
    LLMTraceError,
    _redact,
    _redact_string,
    active_ara_root,
)
from ai_scientist.protocol.validator import (
    ValidationReport,
    _validate_against_schema,
    _validate_llm_calls,
)


class _EnvGuard:
    """Snapshot the tracer env vars for a single test and restore on exit."""

    _KEYS = (ENV_ACTIVE_ROOT, ENV_ENABLED, ENV_STAGE, ENV_REDACT, ENV_STRICT)

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
        self.assertTrue(store.exists(row["call_receipt_ref"]["hash"]))
        response_digest = store.get_json(row["response_ref"]["hash"])
        message_digest = store.get_json(row["messages_ref"]["hash"])
        receipt = store.get_json(row["call_receipt_ref"]["hash"])
        self.assertFalse(response_digest["payload_recorded"])
        self.assertFalse(message_digest["payload_recorded"])
        self.assertEqual(receipt["response_sha256"], response_digest["sha256"])
        self.assertNotIn("pong", json.dumps(response_digest))
        self.assertNotIn("ping", json.dumps(message_digest))

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

    def test_model_provenance_is_persisted_without_api_key(self) -> None:
        self._enable()
        record_llm_call(
            provider="openai_compat",
            model="openai_compat/research-model",
            request_style="openai_chat",
            system_message="sys",
            messages=[],
            response_text="ok",
            model_provenance={
                "provider": "openai_compat",
                "requested_model": "openai_compat/research-model",
                "client_model": "research-model",
                "endpoint_fingerprint": "sha256:" + "a" * 64,
                "api_key_env": "OPENAI_COMPAT_API_KEY",
                "secret": "marker",
            },
        )
        [row] = self._read_rows()
        self.assertEqual(
            row["model_provenance"]["api_key_env"], "OPENAI_COMPAT_API_KEY"
        )
        self.assertNotIn("marker", json.dumps(row))

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
        self.assertEqual(row["trace_format"], "digest_receipt_v1")

        missing_receipt = dict(row)
        missing_receipt.pop("call_receipt_ref")
        missing_report = ValidationReport()
        _validate_against_schema(
            missing_receipt,
            schema,
            path="",
            report=missing_report,
        )
        self.assertFalse(missing_report.ok)

        legacy_row = dict(missing_receipt)
        legacy_row.pop("trace_format")
        legacy_report = ValidationReport()
        _validate_against_schema(
            legacy_row,
            schema,
            path="",
            report=legacy_report,
        )
        self.assertTrue(
            legacy_report.ok,
            msg=[error.format() for error in legacy_report.errors],
        )

    def test_digest_receipt_object_graph_validates(self) -> None:
        self._write_standard_call()
        report = _validate_llm_calls(self.root)
        self.assertTrue(report.ok, msg=[error.format() for error in report.errors])

    def test_digest_receipt_validation_rejects_missing_object(self) -> None:
        self._write_standard_call()
        [row] = self._read_rows()
        self._object_path(row["response_ref"]["hash"]).unlink()

        report = _validate_llm_calls(self.root)

        self.assertFalse(report.ok)
        self.assertTrue(
            any(
                "referenced object is missing" in item.message for item in report.errors
            )
        )

    def test_digest_receipt_validation_rejects_wrong_envelope_kind(self) -> None:
        self._write_standard_call()
        [row] = self._read_rows()
        store = ObjectStore(self.root)
        envelope = store.get_json(row["messages_ref"]["hash"])
        envelope["kind"] = "response"
        row["messages_ref"] = store.put_json(envelope).to_json()
        receipt = store.get_json(row["call_receipt_ref"]["hash"])
        receipt["messages_ref_hash"] = row["messages_ref"]["hash"]
        row["call_receipt_ref"] = store.put_json(receipt).to_json()
        self._write_rows([row])

        report = _validate_llm_calls(self.root)

        self.assertFalse(report.ok)
        self.assertTrue(
            any("must identify messages" in item.message for item in report.errors)
        )

    def test_digest_receipt_validation_rejects_payload_recording_claim(self) -> None:
        self._write_standard_call()
        [row] = self._read_rows()
        store = ObjectStore(self.root)
        envelope = store.get_json(row["response_ref"]["hash"])
        envelope["payload_recorded"] = True
        row["response_ref"] = store.put_json(envelope).to_json()
        receipt = store.get_json(row["call_receipt_ref"]["hash"])
        receipt["response_ref_hash"] = row["response_ref"]["hash"]
        row["call_receipt_ref"] = store.put_json(receipt).to_json()
        self._write_rows([row])

        report = _validate_llm_calls(self.root)

        self.assertFalse(report.ok)
        self.assertTrue(
            any("False was expected" in item.message for item in report.errors)
        )

    def test_digest_receipt_validation_rejects_inner_digest_mismatch(self) -> None:
        self._write_standard_call()
        [row] = self._read_rows()
        store = ObjectStore(self.root)
        receipt = store.get_json(row["call_receipt_ref"]["hash"])
        receipt["messages_sha256"] = "sha256:" + "0" * 64
        row["call_receipt_ref"] = store.put_json(receipt).to_json()
        self._write_rows([row])

        report = _validate_llm_calls(self.root)

        self.assertFalse(report.ok)
        self.assertTrue(any("messages_sha256" in item.path for item in report.errors))

    def test_digest_receipt_validation_rejects_row_receipt_mismatch(self) -> None:
        self._write_standard_call()
        [row] = self._read_rows()
        row["model"] = "openai_compat/substituted-model"
        row["params"] = {"temperature": 0.9}
        self._write_rows([row])

        report = _validate_llm_calls(self.root)

        self.assertFalse(report.ok)
        paths = {item.path for item in report.errors}
        self.assertTrue(any(path.endswith(".model") for path in paths))
        self.assertTrue(any(path.endswith(".params") for path in paths))

    def test_legacy_row_keeps_schema_only_compatibility(self) -> None:
        self._write_standard_call()
        [row] = self._read_rows()
        row.pop("trace_format")
        row.pop("call_receipt_ref")
        row["messages_ref"]["hash"] = "sha256:" + "1" * 64
        row["response_ref"]["hash"] = "sha256:" + "2" * 64
        self._write_rows([row])

        report = _validate_llm_calls(self.root)

        self.assertTrue(report.ok, msg=[error.format() for error in report.errors])

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
        resp_dump = json.dumps(store.get_json(row["response_ref"]["hash"]))
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
        persisted += json.dumps(store.get_json(row["response_ref"]["hash"]))
        self.assertNotIn("private-person", persisted)
        self.assertNotIn(private_path, persisted)

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

    def test_foreign_param_objects_are_dropped_without_stringification(self) -> None:
        self._enable()
        canary = "foreign-object-secret-canary"

        class SensitiveResponseFormat:
            def __str__(self) -> str:
                return canary

            __repr__ = __str__

        cid = record_llm_call(
            provider="p",
            model="m",
            request_style="r",
            system_message="s",
            messages=[],
            response_text="x",
            params={"response_format": SensitiveResponseFormat()},
        )

        self.assertIsInstance(cid, str)
        persisted = b"".join(
            path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        )
        self.assertNotIn(canary.encode("utf-8"), persisted)
        [row] = self._read_rows()
        self.assertNotIn("response_format", row["params"])

    def test_strict_mode_fails_when_trace_cannot_be_persisted(self) -> None:
        self._enable()
        os.environ[ENV_STRICT] = "1"
        with mock.patch.object(
            ObjectStore, "put_json", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(LLMTraceError, "persistence failed"):
                record_llm_call(
                    provider="p",
                    model="m",
                    request_style="r",
                    system_message="s",
                    messages=[],
                    response_text="x",
                )

    def test_strict_failure_does_not_chain_sensitive_provider_errors(self) -> None:
        self._enable()
        os.environ[ENV_STRICT] = "1"
        canary = "strict-trace-error-secret-canary"
        caught: LLMTraceError | None = None
        with mock.patch.object(
            ObjectStore,
            "put_json",
            side_effect=OSError(canary),
        ):
            try:
                record_llm_call(
                    provider="p",
                    model="m",
                    request_style="r",
                    system_message="s",
                    messages=[],
                    response_text="x",
                )
            except LLMTraceError as exc:
                caught = exc
                rendered = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            else:  # pragma: no cover - assertion helper
                self.fail("strict tracing unexpectedly succeeded")

        self.assertNotIn(canary, rendered)
        self.assertIsNotNone(caught)
        self.assertIsNone(caught.__cause__)

    def test_strict_mode_rejects_unserializable_semantic_parameters(self) -> None:
        self._enable()
        os.environ[ENV_STRICT] = "1"
        with self.assertRaisesRegex(LLMTraceError, "TypeError"):
            record_llm_call(
                provider="p",
                model="m",
                request_style="r",
                system_message="s",
                messages=[],
                response_text="x",
                params={"response_format": object()},
            )
        self.assertFalse((self.root / CALLS_JSONL_RELPATH).exists())

    def test_strict_mode_requires_active_ara_root(self) -> None:
        os.environ[ENV_STRICT] = "1"
        with self.assertRaisesRegex(LLMTraceError, "active ARA root"):
            record_llm_call(
                provider="p",
                model="m",
                request_style="r",
                system_message="s",
                messages=[],
                response_text="x",
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _write_standard_call(self) -> None:
        self._enable(stage="planning")
        call_id = record_llm_call(
            provider="openai_compat",
            model="openai_compat/glm-5.3",
            request_style="openai_chat",
            system_message="system",
            messages=[{"role": "user", "content": "question"}],
            response_text="answer",
            params={"temperature": 0.2},
            model_provenance={
                "provider": "openai_compat",
                "requested_model": "openai_compat/glm-5.3",
                "client_model": "glm-5.3",
            },
        )
        self.assertIsInstance(call_id, str)

    def _object_path(self, digest_ref: str) -> Path:
        digest = digest_ref.removeprefix("sha256:")
        return self.root / "objects" / "sha256" / digest[:2] / digest[2:]

    def _write_rows(self, rows: list[dict]) -> None:
        path = self.root / CALLS_JSONL_RELPATH
        path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

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
