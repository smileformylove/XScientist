from __future__ import annotations

import json
import pickle
import tempfile
import threading
import time
import unittest
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_scientist.utils.llm_budget import (
    LLMBudgetExceeded,
    LLMBudgetManager,
    LLMBudgetStateError,
    UnknownModelPriceError,
    estimate_tokens,
    is_llm_budget_exception,
    llm_budget_exception_payload,
    resolve_model_price,
)


def _response(input_tokens: int, output_tokens: int, cached_tokens: int = 0):
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(usage=usage)


def _raise_budget_error_in_worker() -> None:
    raise LLMBudgetExceeded("tokens", "worker LLM budget exhausted", {"used": {}})


class LLMBudgetTests(unittest.TestCase):
    def test_public_price_resolution_never_treats_unknown_as_free(self) -> None:
        self.assertIsNone(resolve_model_price("unknown-provider/model"))
        self.assertEqual(
            resolve_model_price(
                "custom/local",
                prices_per_million={"custom/local": {"input": 0.0, "output": 0.0}},
            ),
            {"input": 0.0, "output": 0.0},
        )
        self.assertIsNone(
            resolve_model_price(
                "custom/broken",
                prices_per_million={"custom/broken": {"input": 1.0, "output": -1.0}},
            )
        )
        self.assertEqual(
            resolve_model_price("ollama/qwen2.5:7b"),
            {"input": 0.0, "cached_input": 0.0, "output": 0.0},
        )

    def test_budget_exception_survives_pickle_round_trip(self) -> None:
        original = LLMBudgetExceeded(
            "tokens",
            "LLM token budget would be exceeded",
            {"used": {"input_tokens": 10}},
        )

        restored = pickle.loads(pickle.dumps(original))

        self.assertIsInstance(restored, LLMBudgetExceeded)
        self.assertEqual(restored.dimension, "tokens")
        self.assertEqual(restored.snapshot, original.snapshot)
        self.assertTrue(is_llm_budget_exception(restored))

    def test_budget_exception_survives_process_boundary(self) -> None:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_raise_budget_error_in_worker)
            with self.assertRaises(LLMBudgetExceeded) as ctx:
                future.result(timeout=5)

        self.assertEqual(ctx.exception.dimension, "tokens")
        self.assertTrue(is_llm_budget_exception(ctx.exception))

    def test_budget_exception_payload_unwraps_chained_exception(self) -> None:
        budget_error = UnknownModelPriceError(
            "cost",
            "No price is configured for model 'custom'",
            {"limits": {"max_cost_usd": 1.0}},
        )
        try:
            raise RuntimeError("worker failed") from budget_error
        except RuntimeError as wrapped:
            payload = llm_budget_exception_payload(wrapped)

        self.assertEqual(payload["type"], "UnknownModelPriceError")
        self.assertEqual(payload["dimension"], "cost")
        self.assertEqual(payload["snapshot"], budget_error.snapshot)

    def test_budget_detection_does_not_match_unrelated_price_error(self) -> None:
        self.assertFalse(
            is_llm_budget_exception(RuntimeError("market price is configured wrong"))
        )

    def test_reservation_denies_before_provider_call(self) -> None:
        manager = LLMBudgetManager()
        manager.configure(max_total_tokens=50)
        called = False
        with self.assertRaises(LLMBudgetExceeded):
            manager.reserve(model="test", prompt="hello", max_output_tokens=50)
            called = True
        self.assertFalse(called)

    def test_actual_usage_replaces_reservation(self) -> None:
        manager = LLMBudgetManager()
        manager.configure(max_total_tokens=10_000)
        reservation = manager.reserve(
            model="gpt-4o-mini", prompt="hello", max_output_tokens=1000
        )
        reserved = manager.snapshot()["reserved"]["output_tokens"]
        self.assertEqual(reserved, 1000)
        reservation.settle(response=_response(20, 7))
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["reserved"]["requests"], 0)
        self.assertEqual(snapshot["used"]["input_tokens"], 20)
        self.assertEqual(snapshot["used"]["output_tokens"], 7)

    def test_unknown_model_fails_closed_with_cost_limit(self) -> None:
        manager = LLMBudgetManager()
        manager.configure(max_cost_usd=1.0)
        with self.assertRaises(UnknownModelPriceError):
            manager.reserve(
                model="unknown-provider/model",
                prompt="hello",
                max_output_tokens=10,
            )

    def test_custom_model_price_enables_cost_budgeting(self) -> None:
        manager = LLMBudgetManager()
        manager.configure(
            max_cost_usd=1.0,
            prices_per_million={"custom": {"input": 1.0, "output": 2.0}},
        )
        reservation = manager.reserve(
            model="custom", prompt="hello", max_output_tokens=100
        )
        reservation.settle(input_tokens=10, output_tokens=5)
        self.assertAlmostEqual(manager.snapshot()["used"]["cost_usd"], 20 / 1_000_000)

    def test_explicit_zero_price_is_allowed_for_local_models(self) -> None:
        manager = LLMBudgetManager()
        manager.configure(
            max_cost_usd=1.0,
            prices_per_million={"ollama/local": {"input": 0.0, "output": 0.0}},
        )
        reservation = manager.reserve(
            model="ollama/local", prompt="hello", max_output_tokens=100
        )
        reservation.settle(input_tokens=10, output_tokens=5)
        self.assertEqual(manager.snapshot()["used"]["cost_usd"], 0.0)

    def test_wall_time_budget_expires_before_call(self) -> None:
        manager = LLMBudgetManager()
        manager.configure(max_wall_time_seconds=0.01)
        time.sleep(0.02)
        with self.assertRaises(LLMBudgetExceeded) as ctx:
            manager.reserve(model="test", prompt="x", max_output_tokens=1)
        self.assertEqual(ctx.exception.dimension, "wall_time")

    def test_zero_token_limit_blocks_every_request(self) -> None:
        manager = LLMBudgetManager()
        manager.configure(max_total_tokens=0)
        with self.assertRaises(LLMBudgetExceeded):
            manager.reserve(model="test", prompt="", max_output_tokens=1)

    def test_negative_limits_are_rejected(self) -> None:
        manager = LLMBudgetManager()
        with self.assertRaises(ValueError):
            manager.configure(max_cost_usd=-1)

    def test_file_backed_reservations_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            manager_a = LLMBudgetManager()
            manager_b = LLMBudgetManager()
            manager_a.configure(max_total_tokens=300, state_path=state_path)
            manager_b.configure(max_total_tokens=300, state_path=state_path)
            barrier = threading.Barrier(2)
            release = threading.Event()
            outcomes: list[str] = []

            def reserve(manager: LLMBudgetManager) -> None:
                barrier.wait()
                try:
                    reservation = manager.reserve(
                        model="test", prompt="x", max_output_tokens=140
                    )
                    outcomes.append("reserved")
                    release.wait(timeout=2)
                    reservation.cancel(charge_reserved=False)
                except LLMBudgetExceeded:
                    outcomes.append("denied")
                    release.set()

            threads = [
                threading.Thread(target=reserve, args=(manager_a,)),
                threading.Thread(target=reserve, args=(manager_b,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertCountEqual(outcomes, ["reserved", "denied"])

    def test_corrupt_shared_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            state_path.write_text("not-json")
            manager = LLMBudgetManager()
            with self.assertRaises(LLMBudgetStateError):
                manager.configure(max_total_tokens=1000, state_path=state_path)

    def test_structurally_invalid_shared_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            state_path.write_text(json.dumps({"version": "llm_budget.v1"}))
            manager = LLMBudgetManager()
            with self.assertRaises(LLMBudgetStateError):
                manager.configure(max_total_tokens=1000, state_path=state_path)

    def test_disappeared_shared_state_fails_closed_without_reinitializing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "budget.json"
            manager = LLMBudgetManager()
            manager.configure(max_total_tokens=1000, state_path=state_path)
            state_path.unlink()

            with self.assertRaisesRegex(LLMBudgetStateError, "state disappeared"):
                manager.reserve(model="test", prompt="x", max_output_tokens=1)

            self.assertFalse(state_path.exists())

    def test_disappeared_shared_state_directory_raises_budget_state_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "ledger"
            state_path = state_dir / "budget.json"
            manager = LLMBudgetManager()
            manager.configure(max_total_tokens=1000, state_path=state_path)
            state_path.unlink()
            state_path.with_suffix(".json.lock").unlink(missing_ok=True)
            state_dir.rmdir()

            with self.assertRaisesRegex(LLMBudgetStateError, "lock is unavailable"):
                manager.reserve(model="test", prompt="x", max_output_tokens=1)

            self.assertFalse(state_dir.exists())

    def test_file_backed_denial_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            manager = LLMBudgetManager()
            manager.configure(max_total_tokens=10, state_path=state_path)
            with self.assertRaises(LLMBudgetExceeded):
                manager.reserve(model="test", prompt="x", max_output_tokens=10)
            persisted = json.loads(state_path.read_text())
            self.assertEqual(persisted["denials"][-1]["dimension"], "tokens")

    def test_shared_state_rejects_policy_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            first = LLMBudgetManager()
            first.configure(max_total_tokens=100, state_path=state_path)
            second = LLMBudgetManager()
            with self.assertRaises(LLMBudgetStateError):
                second.configure(max_total_tokens=200, state_path=state_path)

    def test_resume_allows_only_budget_increases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            first = LLMBudgetManager()
            first.configure(max_total_tokens=100, state_path=state_path)

            resumed = LLMBudgetManager()
            resumed.configure(
                max_total_tokens=200,
                state_path=state_path,
                allow_limit_increase=True,
            )
            self.assertEqual(resumed.snapshot()["limits"]["max_total_tokens"], 200)

            reduced = LLMBudgetManager()
            with self.assertRaises(LLMBudgetStateError):
                reduced.configure(
                    max_total_tokens=50,
                    state_path=state_path,
                    allow_limit_increase=True,
                )

    def test_resume_wall_time_does_not_charge_stopped_downtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            first = LLMBudgetManager()
            first.configure(max_wall_time_seconds=100, state_path=state_path)
            state = json.loads(state_path.read_text())
            state["started_at"] = 1000.0
            state["updated_at"] = 1010.0
            state_path.write_text(json.dumps(state))

            resumed = LLMBudgetManager()
            resumed.configure(
                max_wall_time_seconds=100,
                state_path=state_path,
                allow_limit_increase=True,
            )

            self.assertLess(resumed.snapshot()["elapsed_seconds"], 20.0)

    def test_resume_reclaims_orphaned_reservations_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            first = LLMBudgetManager()
            first.configure(max_total_tokens=1000, state_path=state_path)
            first.reserve(model="test", prompt="x", max_output_tokens=20)

            resumed = LLMBudgetManager()
            resumed.configure(
                max_total_tokens=1000,
                state_path=state_path,
                reclaim_active_reservations=True,
            )
            snapshot = resumed.snapshot()

            self.assertEqual(snapshot["active_reservations"], 0)
            self.assertEqual(snapshot["reserved"]["requests"], 0)
            self.assertEqual(snapshot["used"]["requests"], 1)
            self.assertEqual(snapshot["denials"][-1]["dimension"], "resume")

    def test_report_retry_rethrows_budget_exhaustion(self) -> None:
        from ai_scientist.treesearch import log_summarization

        budget_error = LLMBudgetExceeded("tokens", "budget exhausted", {})
        with mock.patch.object(
            log_summarization,
            "get_response_from_llm",
            side_effect=budget_error,
        ) as call_mock:
            with self.assertRaises(LLMBudgetExceeded):
                log_summarization.update_summary(
                    {},
                    "stage",
                    SimpleNamespace(good_nodes=[]),
                    {},
                    "model",
                    mock.Mock(),
                )

        call_mock.assert_called_once()

    def test_unbounded_image_estimate_counts_base64_bytes(self) -> None:
        small = estimate_tokens(
            [{"image_url": {"url": "data:image/png;base64," + "a" * 10}}],
            model="gpt-4o-mini",
        )
        large = estimate_tokens(
            [{"image_url": {"url": "data:image/png;base64," + "a" * 100_000}}],
            model="gpt-4o-mini",
        )
        self.assertGreater(large, small)

    def test_low_detail_image_uses_fixed_conservative_reservation(self) -> None:
        small = estimate_tokens(
            [
                {
                    "image_url": {
                        "url": "data:image/png;base64," + "a" * 10,
                        "detail": "low",
                    }
                }
            ],
            model="gpt-4o-mini",
        )
        large = estimate_tokens(
            [
                {
                    "image_url": {
                        "url": "data:image/png;base64," + "a" * 100_000,
                        "detail": "low",
                    }
                }
            ],
            model="gpt-4o-mini",
        )
        self.assertEqual(small, large)

    def test_unknown_visual_model_reserves_inline_image_bytes(self) -> None:
        small = estimate_tokens(
            [{"image_url": {"url": "data:image/png;base64," + "a" * 10}}],
            model="custom-vlm",
        )
        large = estimate_tokens(
            [{"image_url": {"url": "data:image/png;base64," + "a" * 10_000}}],
            model="custom-vlm",
        )
        self.assertGreater(large, small)

    def test_exported_environment_rehydrates_shared_manager(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "budget.json"
            manager = LLMBudgetManager()
            manager.configure(
                max_total_tokens=500,
                max_cost_usd=2.0,
                state_path=state_path,
            )
            snapshot = dict(os.environ)
            try:
                manager.export_environment()
                child = LLMBudgetManager()
                child.configure_from_env()
                self.assertEqual(child.snapshot()["limits"]["max_total_tokens"], 500)
                self.assertEqual(child.snapshot()["limits"]["max_cost_usd"], 2.0)
            finally:
                os.environ.clear()
                os.environ.update(snapshot)


class LLMBudgetWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        from ai_scientist.utils.llm_budget import llm_budget_manager

        self.manager = llm_budget_manager
        self.manager.configure(max_total_tokens=None, reset=True)

    def tearDown(self) -> None:
        self.manager.configure(max_total_tokens=None, reset=True)

    def test_llm_call_is_denied_before_client_method(self) -> None:
        from ai_scientist import llm

        self.manager.configure(max_total_tokens=10)
        client = mock.Mock()
        with self.assertRaises(LLMBudgetExceeded):
            llm.make_llm_call(
                client,
                "gpt-4o-mini",
                0.0,
                "system",
                [{"role": "user", "content": "prompt"}],
            )
        client.chat.completions.create.assert_not_called()

    def test_successful_llm_call_settles_actual_usage(self) -> None:
        from ai_scientist import llm

        self.manager.configure(max_total_tokens=10_000)
        response = _response(12, 4)
        response.choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]
        client = mock.Mock()
        client.chat.completions.create.return_value = response
        llm.make_llm_call(
            client,
            "gpt-4o-mini",
            0.0,
            "system",
            [{"role": "user", "content": "prompt"}],
        )
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["used"]["input_tokens"], 12)
        self.assertEqual(snapshot["used"]["output_tokens"], 4)

    def test_batch_reserves_output_for_every_choice(self) -> None:
        from ai_scientist import llm

        self.manager.configure(max_total_tokens=5000)
        client = mock.Mock()
        with self.assertRaises(LLMBudgetExceeded):
            llm.get_batch_responses_from_llm(
                prompt="p",
                client=client,
                model="gpt-4o-mini",
                system_message="s",
                n_responses=2,
            )
        client.chat.completions.create.assert_not_called()

    def test_treesearch_retry_attempts_reserve_independently(self) -> None:
        from ai_scientist.treesearch.backend import utils

        class RetryError(Exception):
            pass

        self.manager.configure(max_total_tokens=200)
        create = mock.Mock(side_effect=RetryError("retry"))
        result = utils.backoff_create.__wrapped__(
            create,
            (RetryError,),
            _budget_model="test",
            _budget_prompt="p",
            _budget_max_output_tokens=60,
        )
        self.assertFalse(result)
        snapshot = self.manager.snapshot()
        self.assertEqual(snapshot["used"]["requests"], 1)
        with self.assertRaises(LLMBudgetExceeded):
            utils.backoff_create.__wrapped__(
                create,
                (RetryError,),
                _budget_model="test",
                _budget_prompt="p",
                _budget_max_output_tokens=60,
            )

    def test_treesearch_route_uses_bounded_default_output(self) -> None:
        from ai_scientist.treesearch import backend

        captured: dict = {}

        def fake_query(**kwargs):
            captured.update(kwargs)
            return "ok", 0.1, 10, 2, {}

        fake_module = SimpleNamespace(query=fake_query)
        self.manager.configure(max_total_tokens=20_000)
        with mock.patch.object(
            backend, "_resolve_backend_module", return_value=fake_module
        ):
            result = backend.query(
                system_message="s",
                user_message="u",
                model="gpt-4o-mini",
            )
        self.assertEqual(result, "ok")
        self.assertEqual(captured["max_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
