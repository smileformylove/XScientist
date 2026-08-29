from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ai_scientist.protocol.schemas import load_schema
from jsonschema import validate

from xscientist.research_agent_benchmark import (
    ResearchAgentBenchmarkError,
    benchmark_research_agent,
    execute_research_benchmark_tool,
    run_research_agent_episode,
    score_research_agent_episode,
    verify_research_agent_benchmark,
)


def _tool(name: str) -> dict[str, Any]:
    return {"action": "tool", "tool": name, "arguments": {}}


def _decision(
    *,
    evidence: list[str],
    direction: str = "harmful",
    adjusted_effect: float = -0.1,
    analysis_basis: str = "standardized",
    confounding_detected: bool = True,
    confounder: str = "difficulty",
    pooled_result_rejected: bool = True,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "action": "final",
        "decision": {
            "direction": direction,
            "adjusted_effect": adjusted_effect,
            "analysis_basis": analysis_basis,
            "stratum_effects": {"easy": -0.1, "hard": -0.1},
            "pattern": "simpson_reversal",
            "confounding_detected": confounding_detected,
            "confounder": confounder,
            "pooled_result_rejected": pooled_result_rejected,
            "causal_status": "not_identified",
            "recommendation": "reject_pooled_benefit_claim",
            "negative_result": "preserve_and_report",
            "uncertainty": "moderate",
            "limitations": limitations
            or [
                "aggregated_data",
                "observational_assignment",
                "no_causal_identification",
            ],
            "next_experiment": "stratified_randomized_replication",
            "evidence_tool_call_ids": evidence,
        },
    }


class ScriptedTransport:
    def __init__(
        self,
        actions: list[dict[str, Any] | str],
        *,
        reported_model: str = "glm-5.3",
        usage: dict[str, int] | None = None,
    ) -> None:
        self.actions = list(actions)
        self.reported_model = reported_model
        self.usage = usage or {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        messages: Any,
        model: str,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "max_output_tokens": max_output_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        action = self.actions.pop(0)
        content = action if isinstance(action, str) else json.dumps(action)
        return {
            "content": content,
            "reported_model": self.reported_model,
            "usage": self.usage,
        }


class ResearchAgentBenchmarkTests(unittest.TestCase):
    def _passing_transport(self) -> ScriptedTransport:
        return ScriptedTransport(
            [
                _tool("inspect_design"),
                _tool("pooled_effect"),
                _tool("stratified_effect"),
                _tool("standardized_effect"),
                _decision(evidence=["tool-01", "tool-02", "tool-03", "tool-04"]),
            ]
        )

    def test_passing_episode_is_grounded_bounded_and_not_a_quality_claim(self) -> None:
        transport = self._passing_transport()
        report = benchmark_research_agent(transport)

        self.assertTrue(report["ok"])
        self.assertEqual(report["score"]["score"], 100)
        self.assertTrue(report["score"]["benchmark_contract_passed"])
        self.assertTrue(report["claims"]["agent_execution_completed"])
        self.assertTrue(report["claims"]["rollout_audit_complete"])
        for field in (
            "scientific_contract_verified",
            "quality_claim_allowed",
            "causal_claim_allowed",
            "real_world_truth_claim_allowed",
            "generalization_claim_allowed",
            "cross_model_comparison_allowed",
            "cross_system_comparison_allowed",
            "production_promotion_allowed",
            "report_authenticity_verified",
            "live_rollout_verified",
            "research_taste_claim_allowed",
            "independent_scientific_review_completed",
        ):
            self.assertFalse(report["claims"][field])
        execution = report["episode"]["execution"]
        self.assertFalse(execution["network_used_declared"])
        self.assertFalse(execution["provider_used_declared"])
        self.assertFalse(execution["network_use_verified"])
        self.assertFalse(execution["provider_execution_verified"])
        self.assertTrue(execution["model_identity_exact"])
        self.assertEqual(execution["reported_models"], ["glm-5.3"])
        self.assertEqual(execution["usage"]["total_tokens"], 150)
        self.assertEqual(
            report["episode"]["policy_contract"]["decision_owner"],
            "research_policy",
        )
        self.assertEqual(
            report["episode"]["policy_contract"]["execution_owner"],
            "deterministic_statistical_executor",
        )
        validate(report, load_schema("research_agent_benchmark"))
        self.assertTrue(verify_research_agent_benchmark(report)["ok"])

        self.assertEqual(transport.calls[0]["model"], "openai_compat/glm-5.3")
        self.assertEqual(transport.calls[0]["max_output_tokens"], 512)
        self.assertGreater(transport.calls[0]["timeout_seconds"], 0)
        system_prompt = transport.calls[0]["messages"][0]["content"]
        self.assertIn("research_policy", system_prompt)
        self.assertIn("deterministic statistical executor", system_prompt)

    def test_tools_expose_simpson_reversal_without_external_code(self) -> None:
        pooled = execute_research_benchmark_tool("pooled_effect")
        stratified = execute_research_benchmark_tool("stratified_effect")
        standardized = execute_research_benchmark_tool("standardized_effect")

        self.assertAlmostEqual(pooled["rate_difference"], 0.46)
        self.assertAlmostEqual(stratified["effects"]["easy"]["rate_difference"], -0.1)
        self.assertAlmostEqual(stratified["effects"]["hard"]["rate_difference"], -0.1)
        self.assertAlmostEqual(standardized["rate_difference"], -0.1)
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "not allowed"):
            execute_research_benchmark_tool("run_arbitrary_python")

    def test_pooled_only_answer_cannot_pass_even_when_episode_completes(self) -> None:
        transport = ScriptedTransport(
            [
                _tool("pooled_effect"),
                _decision(
                    evidence=["tool-01"],
                    direction="beneficial",
                    adjusted_effect=0.46,
                    analysis_basis="pooled",
                    confounding_detected=False,
                    confounder="none",
                    pooled_result_rejected=False,
                    limitations=["aggregated_data"],
                ),
            ]
        )
        report = benchmark_research_agent(transport)

        self.assertTrue(report["claims"]["agent_execution_completed"])
        self.assertFalse(report["claims"]["benchmark_contract_passed"])
        self.assertFalse(report["ok"])
        gates = report["score"]["hard_gates"]
        self.assertFalse(gates["stratified_analysis_executed"])
        self.assertFalse(gates["standardized_analysis_executed"])
        self.assertFalse(gates["effect_grounded"])
        self.assertFalse(gates["confounding_recognized"])
        self.assertFalse(gates["causal_boundary_respected"])
        self.assertTrue(verify_research_agent_benchmark(report)["ok"])

    def test_invalid_response_is_never_recorded_and_fails_protocol_gate(self) -> None:
        sentinel = "RAW_COMPLETION_SENTINEL"
        transport = ScriptedTransport(
            [
                f"```json\n{sentinel}\n```",
                _tool("inspect_design"),
                _tool("pooled_effect"),
                _tool("stratified_effect"),
                _tool("standardized_effect"),
                _decision(evidence=["tool-01", "tool-02", "tool-03", "tool-04"]),
            ]
        )
        report = benchmark_research_agent(transport)

        self.assertTrue(report["claims"]["agent_execution_completed"])
        self.assertFalse(report["claims"]["benchmark_contract_passed"])
        self.assertIn(
            "invalid_action_schema",
            report["episode"]["execution"]["protocol_violations"],
        )
        self.assertNotIn(sentinel, json.dumps(report))
        self.assertFalse(report["score"]["hard_gates"]["protocol_valid"])

    def test_duplicate_json_keys_fail_closed(self) -> None:
        duplicate = (
            '{"action":"tool","action":"final",'
            '"tool":"inspect_design","arguments":{}}'
        )
        episode = run_research_agent_episode(ScriptedTransport([duplicate, duplicate]))

        self.assertFalse(episode["execution"]["agent_execution_completed"])
        self.assertEqual(
            episode["execution"]["termination_reason"],
            "protocol_violation_limit",
        )
        self.assertEqual(
            episode["execution"]["protocol_violations"],
            ["invalid_action_schema"],
        )

    def test_unknown_tool_is_not_executed_or_fallback_mapped(self) -> None:
        unknown = {"action": "tool", "tool": "python", "arguments": {}}
        episode = run_research_agent_episode(ScriptedTransport([unknown, unknown]))

        self.assertEqual(episode["observations"], [])
        self.assertEqual(
            episode["execution"]["termination_reason"],
            "protocol_violation_limit",
        )

    def test_model_substitution_stops_before_action_execution(self) -> None:
        episode = run_research_agent_episode(
            ScriptedTransport([_tool("inspect_design")], reported_model="another-model")
        )

        self.assertEqual(episode["observations"], [])
        self.assertFalse(episode["execution"]["model_identity_exact"])
        self.assertEqual(
            episode["execution"]["termination_reason"],
            "model_identity_unverified",
        )
        self.assertEqual(
            episode["execution"]["protocol_violations"],
            ["model_identity_not_exact"],
        )

    def test_transport_failure_records_only_safe_exception_type(self) -> None:
        class ExplodingTransport:
            def __call__(self, **_: Any) -> dict[str, Any]:
                raise RuntimeError("secret endpoint and credential material")

        report = benchmark_research_agent(ExplodingTransport())
        rendered = json.dumps(report)

        self.assertNotIn("secret endpoint", rendered)
        self.assertNotIn("credential material", rendered)
        self.assertEqual(
            report["episode"]["execution"]["transport_error_code"],
            "transport_exception",
        )
        self.assertFalse(report["ok"])
        self.assertTrue(verify_research_agent_benchmark(report)["ok"])

    def test_live_mode_is_explicit_and_cannot_be_selected_by_response(self) -> None:
        offline = run_research_agent_episode(
            self._passing_transport(), execution_mode="offline_test"
        )
        live = run_research_agent_episode(
            self._passing_transport(), execution_mode="live_provider"
        )

        self.assertFalse(offline["execution"]["network_used_declared"])
        self.assertTrue(live["execution"]["network_used_declared"])
        self.assertFalse(live["execution"]["network_use_verified"])
        self.assertFalse(live["execution"]["provider_execution_verified"])
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "execution_mode"):
            run_research_agent_episode(self._passing_transport(), execution_mode="auto")

    def test_budget_contract_rejects_unbounded_configuration(self) -> None:
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "max_turns"):
            run_research_agent_episode(self._passing_transport(), max_turns=1000)
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "max_seconds"):
            run_research_agent_episode(
                self._passing_transport(), max_seconds=float("inf")
            )
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "max_response_bytes"):
            run_research_agent_episode(
                self._passing_transport(), max_response_bytes=10_000_000
            )

    def test_score_recomputation_does_not_trust_a_claimed_pass(self) -> None:
        report = benchmark_research_agent(self._passing_transport())
        episode = copy.deepcopy(report["episode"])
        episode["decision"]["evidence_tool_call_ids"] = ["tool-99"]

        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "provenance"):
            score_research_agent_episode(episode)

    def test_verifier_detects_tool_score_claim_and_fingerprint_tampering(self) -> None:
        report = benchmark_research_agent(self._passing_transport())

        tool_tampered = copy.deepcopy(report)
        tool_tampered["episode"]["observations"][3]["result"]["rate_difference"] = 0.9
        checked = verify_research_agent_benchmark(tool_tampered)
        self.assertFalse(checked["ok"])
        self.assertIn("observation_result_mismatch", checked["errors"])

        score_tampered = copy.deepcopy(report)
        score_tampered["score"]["score"] = 99
        checked = verify_research_agent_benchmark(score_tampered)
        self.assertFalse(checked["ok"])
        self.assertIn("score_mismatch", checked["errors"])

        claim_tampered = copy.deepcopy(report)
        claim_tampered["claims"]["quality_claim_allowed"] = True
        checked = verify_research_agent_benchmark(claim_tampered)
        self.assertFalse(checked["ok"])
        self.assertEqual(checked["checks"]["schema"], "failed")

        fingerprint_tampered = copy.deepcopy(report)
        fingerprint_tampered["episode"]["decision"]["uncertainty"] = "high"
        checked = verify_research_agent_benchmark(fingerprint_tampered)
        self.assertFalse(checked["ok"])
        self.assertIn("reproducibility_mismatch", checked["errors"])

        trace_tampered = copy.deepcopy(report)
        trace_tampered["episode"]["trace"][0]["reported_model"] = "other-model"
        checked = verify_research_agent_benchmark(trace_tampered)
        self.assertFalse(checked["ok"])
        self.assertEqual(checked["checks"]["episode_consistency"], "failed")
        self.assertIn("model_identity_summary_mismatch", checked["errors"])

    def test_saved_report_verifies_without_network_or_provider(self) -> None:
        report = benchmark_research_agent(self._passing_transport())
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "research-agent.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            checked = verify_research_agent_benchmark(path)

        self.assertTrue(checked["ok"])
        self.assertFalse(checked["network_used"])
        self.assertFalse(checked["provider_used"])

    def test_live_mode_is_only_a_declaration_not_verified_provenance(self) -> None:
        report = benchmark_research_agent(
            self._passing_transport(), execution_mode="live_provider"
        )

        execution = report["episode"]["execution"]
        self.assertTrue(execution["network_used_declared"])
        self.assertTrue(execution["provider_used_declared"])
        self.assertFalse(execution["network_use_verified"])
        self.assertFalse(execution["provider_execution_verified"])
        self.assertFalse(report["claims"]["live_rollout_verified"])
        self.assertFalse(report["claims"]["report_authenticity_verified"])
        self.assertTrue(verify_research_agent_benchmark(report)["ok"])

    def test_secret_shaped_model_identity_is_rejected_without_digest(self) -> None:
        secret_models = [
            "sk-" + "A" * 48,
            "ghp_" + "B" * 36,
            "xoxb-" + "C" * 32,
            "AKIA" + "D" * 16,
            "AIza" + "E" * 35,
        ]

        for secret_model in secret_models:
            with self.subTest(secret_model=secret_model[:4]):
                report = benchmark_research_agent(
                    ScriptedTransport(
                        [_tool("inspect_design")], reported_model=secret_model
                    )
                )
                rendered = json.dumps(report)

                self.assertNotIn(secret_model, rendered)
                self.assertEqual(report["episode"]["trace"], [])
                self.assertEqual(
                    report["episode"]["execution"]["transport_error_code"],
                    "invalid_transport_response",
                )
                self.assertFalse(report["ok"])
                self.assertTrue(verify_research_agent_benchmark(report)["ok"])

    def test_secret_shaped_requested_model_is_rejected(self) -> None:
        for secret_model in (
            "ghp_" + "B" * 36,
            "xoxb-" + "C" * 32,
            "AKIA" + "D" * 16,
            "AIza" + "E" * 35,
        ):
            with self.subTest(secret_model=secret_model[:4]):
                with self.assertRaisesRegex(
                    ResearchAgentBenchmarkError, "unsupported characters"
                ):
                    benchmark_research_agent(
                        self._passing_transport(), model=secret_model
                    )

    def test_provider_configuration_is_fingerprinted_without_key_or_url(self) -> None:
        secret_key = "sk-" + "B" * 48
        secret_url = "https://private-research-gateway.example/v1"
        report = benchmark_research_agent(
            self._passing_transport(),
            provider_environment={
                "OPENAI_COMPAT_API_KEY": secret_key,
                "OPENAI_COMPAT_BASE_URL": secret_url,
            },
        )
        rendered = json.dumps(report)
        observation = report["episode"]["policy_contract"]["configuration_observation"]

        self.assertNotIn(secret_key, rendered)
        self.assertNotIn(secret_url, rendered)
        self.assertTrue(observation["endpoint_configured"])
        self.assertTrue(observation["endpoint_fingerprint"].startswith("sha256:"))
        self.assertEqual(observation["api_key_env"], "OPENAI_COMPAT_API_KEY")
        self.assertTrue(verify_research_agent_benchmark(report)["ok"])

        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "provenance"):
            benchmark_research_agent(
                self._passing_transport(),
                provider_environment={
                    "OPENAI_API_KEY": secret_key,
                    "OPENAI_BASE_URL": secret_url,
                },
            )

    def test_usage_overflow_and_inconsistent_totals_fail_transport_contract(
        self,
    ) -> None:
        overflow = benchmark_research_agent(
            ScriptedTransport(
                [_tool("inspect_design")],
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": 999_999,
                    "total_tokens": 1_000_000,
                },
            )
        )
        inconsistent = benchmark_research_agent(
            ScriptedTransport(
                [_tool("inspect_design")],
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 999,
                },
            )
        )

        for report in (overflow, inconsistent):
            execution = report["episode"]["execution"]
            self.assertEqual(
                execution["transport_error_code"], "invalid_transport_response"
            )
            self.assertFalse(report["ok"])
            self.assertFalse(report["score"]["hard_gates"]["within_budgets"])
            self.assertTrue(verify_research_agent_benchmark(report)["ok"])

    def test_total_token_budget_terminates_before_an_unbounded_rollout(self) -> None:
        transport = ScriptedTransport(
            [
                _tool("inspect_design"),
                _tool("pooled_effect"),
                _tool("stratified_effect"),
                _tool("standardized_effect"),
                _decision(evidence=["tool-01", "tool-02", "tool-03", "tool-04"]),
            ],
            usage={
                "prompt_tokens": 5_000,
                "completion_tokens": 10,
                "total_tokens": 5_010,
            },
        )
        report = benchmark_research_agent(transport)
        execution = report["episode"]["execution"]

        self.assertEqual(execution["termination_reason"], "token_budget_exhausted")
        self.assertTrue(execution["budget_exceeded"])
        self.assertFalse(report["score"]["hard_gates"]["within_budgets"])
        self.assertFalse(report["ok"])

    def test_missing_usage_can_complete_but_cannot_pass_the_budget_gate(self) -> None:
        transport = self._passing_transport()
        transport.usage = {"prompt_tokens": 20}
        report = benchmark_research_agent(transport)

        self.assertTrue(report["claims"]["agent_execution_completed"])
        self.assertFalse(report["episode"]["execution"]["usage_complete"])
        self.assertFalse(report["score"]["hard_gates"]["usage_accounting_complete"])
        self.assertFalse(report["ok"])

    def test_final_decision_must_be_coherent_with_each_cited_analysis(self) -> None:
        pooled_basis = _decision(
            evidence=["tool-01", "tool-02", "tool-03", "tool-04"],
            analysis_basis="pooled",
        )
        wrong_strata = _decision(evidence=["tool-01", "tool-02", "tool-03", "tool-04"])
        wrong_strata["decision"]["stratum_effects"] = {
            "easy": 0.9,
            "hard": 0.9,
        }
        for decision in (pooled_basis, wrong_strata):
            report = benchmark_research_agent(
                ScriptedTransport(
                    [
                        _tool("inspect_design"),
                        _tool("pooled_effect"),
                        _tool("stratified_effect"),
                        _tool("standardized_effect"),
                        decision,
                    ]
                )
            )
            self.assertFalse(report["score"]["hard_gates"]["decision_coherent"])
            self.assertFalse(report["ok"])

    def test_scientific_restraint_fields_are_non_compensable(self) -> None:
        weak = _decision(
            evidence=["tool-01", "tool-02", "tool-03", "tool-04"],
            limitations=["no_causal_identification"],
        )
        weak["decision"]["uncertainty"] = "low"
        weak["decision"]["next_experiment"] = "none"
        report = benchmark_research_agent(
            ScriptedTransport(
                [
                    _tool("inspect_design"),
                    _tool("pooled_effect"),
                    _tool("stratified_effect"),
                    _tool("standardized_effect"),
                    weak,
                ]
            )
        )

        gates = report["score"]["hard_gates"]
        self.assertFalse(gates["limitations_complete"])
        self.assertFalse(gates["uncertainty_calibrated"])
        self.assertFalse(gates["follow_up_experiment_identified"])
        self.assertFalse(report["ok"])

    def test_public_scorer_rejects_a_handcrafted_source_free_episode(self) -> None:
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "schema"):
            score_research_agent_episode(
                {
                    "schema": "xscientist.research-agent-episode.v1",
                    "execution": {"agent_execution_completed": True},
                    "observations": [],
                    "decision": {},
                    "limits": {},
                }
            )

    def test_public_scorer_applies_full_episode_schema_before_scoring(self) -> None:
        report = benchmark_research_agent(self._passing_transport())
        mutations = (
            ("raw_prompt_recorded", True),
            ("credentials_recorded", True),
            ("tool_results_recorded", False),
            ("structured_decision_recorded", False),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                episode = copy.deepcopy(report["episode"])
                episode["retention"][field] = value
                with self.assertRaisesRegex(ResearchAgentBenchmarkError, "schema"):
                    score_research_agent_episode(episode)

        failed_observation = copy.deepcopy(report["episode"])
        failed_observation["observations"][0]["outcome"] = "failed"
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "schema"):
            score_research_agent_episode(failed_observation)

        extra_trace_payload = copy.deepcopy(report["episode"])
        extra_trace_payload["trace"][0]["raw_response"] = "secret-canary"
        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "schema"):
            score_research_agent_episode(extra_trace_payload)

    def test_public_scorer_rejects_nonterminal_discarded_trace_rows(self) -> None:
        report = benchmark_research_agent(self._passing_transport())
        episode = copy.deepcopy(report["episode"])
        discarded = copy.deepcopy(episode["trace"][0])
        discarded.update(
            {
                "turn": 1,
                "action_kind": "discarded",
                "action_sha256": None,
                "tool": None,
                "tool_call_id": None,
                "observation_sha256": None,
            }
        )
        episode["trace"].insert(0, discarded)
        for turn, row in enumerate(episode["trace"], start=1):
            row["turn"] = turn
        episode["execution"]["turns_observed"] = len(episode["trace"])
        episode["execution"]["usage"] = {
            key: sum(int(row["usage"][key]) for row in episode["trace"])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

        with self.assertRaisesRegex(ResearchAgentBenchmarkError, "provenance"):
            score_research_agent_episode(episode)

    def test_verifier_binds_duration_policy_and_final_action_digest(self) -> None:
        report = benchmark_research_agent(self._passing_transport())

        duration = copy.deepcopy(report)
        duration["episode"]["execution"]["duration_seconds"] = 999_999
        checked = verify_research_agent_benchmark(duration)
        self.assertFalse(checked["ok"])
        self.assertIn("duration_observation_invalid", checked["errors"])

        policy = copy.deepcopy(report)
        policy["episode"]["policy_contract"]["provider"] = "zhipu"
        checked = verify_research_agent_benchmark(policy)
        self.assertFalse(checked["ok"])
        self.assertIn("policy_model_binding_mismatch", checked["errors"])

        final_action = copy.deepcopy(report)
        final_action["episode"]["decision"]["uncertainty"] = "high"
        checked = verify_research_agent_benchmark(final_action)
        self.assertFalse(checked["ok"])
        self.assertIn("final_action_binding_mismatch", checked["errors"])


if __name__ == "__main__":
    unittest.main()
