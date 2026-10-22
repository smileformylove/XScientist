from __future__ import annotations

import copy
import unittest

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.experiment_admission import (
    build_experiment_admission_payload,
    normalize_experiment_admission_payload,
)
from ai_scientist.protocol.schemas import schema_validator
from xscientist.research_git import ResearchGitError
from xscientist.research_lifecycle import ResearchLifecycle


HASH = "sha256:" + "a" * 64


def _bindings() -> tuple[dict, dict]:
    hypothesis = {
        "object_id": "rso-1111111111111111",
        "content_hash": "sha256:" + "1" * 64,
    }
    plan = {
        "object_id": "rso-2222222222222222",
        "content_hash": "sha256:" + "2" * 64,
    }
    return hypothesis, plan


def _payload(scope: dict) -> dict:
    hypothesis, plan = _bindings()
    return build_experiment_admission_payload(
        repository_id="ws-1111111111111111",
        repository_head="a" * 40,
        decision="authorize",
        hypothesis=hypothesis,
        plan=plan,
        preregistration=None,
        execution_scope=scope,
    )


class ObservableScopeTests(unittest.TestCase):
    def test_legacy_scope_remains_unchanged(self) -> None:
        normalized = normalize_experiment_admission_payload(
            _payload({"task_id": "legacy-diagnostic"})
        )
        self.assertEqual(normalized["scope"], {"task_id": "legacy-diagnostic"})
        self.assertNotIn("arm_ids", normalized["scope"])
        self.assertNotIn("resource_contract_hash", normalized["scope"])

    def test_observable_ladder_scope_round_trips_exactly(self) -> None:
        scope = {
            "arm_ids": [
                "latent",
                "observable_full_history",
                "observable_recurrent_2bit",
                "observable_time_aware",
                "suffix_k1",
                "suffix_k2",
            ],
            "resource_contract_hash": HASH,
            "command_hash": HASH,
            "configuration_hash": HASH,
            "budget_hash": HASH,
            "task_id": "observable-ladder-v1",
        }
        payload = normalize_experiment_admission_payload(_payload(scope))
        self.assertEqual(payload["scope"], scope)

    def test_contract_fields_are_jointly_required(self) -> None:
        for scope in (
            {"arm_ids": ["latent"]},
            {"resource_contract_hash": HASH},
        ):
            with self.subTest(scope=scope), self.assertRaisesRegex(
                ResearchGitError, "supplied together"
            ):
                _payload(scope)

    def test_arm_ids_are_sorted_unique_and_strict(self) -> None:
        valid = {"resource_contract_hash": HASH}
        for arm_ids in (
            [],
            ["suffix_k1", "latent"],
            ["latent", "latent"],
            ["latent", "bad arm"],
            ["latent", 1],
            [True],
        ):
            scope = {**valid, "arm_ids": arm_ids}
            with self.subTest(arm_ids=arm_ids), self.assertRaisesRegex(
                ResearchGitError, "arm_ids"
            ):
                _payload(scope)

    def test_resource_contract_and_existing_hash_types_are_strict(self) -> None:
        for field in (
            "resource_contract_hash",
            "budget_hash",
            "command_hash",
            "configuration_hash",
            "environment_hash",
            "ideas_hash",
        ):
            scope = {"arm_ids": ["latent", "suffix_k1"], field: "not-a-hash"}
            with self.subTest(field=field), self.assertRaisesRegex(
                ResearchGitError, "sha256"
            ):
                _payload(scope)

    def test_existing_text_and_list_types_are_strict(self) -> None:
        for field, value in (
            ("candidate_id", {"bad": True}),
            ("task_id", "  padded  "),
            ("study_phase", True),
            ("interventions", []),
            ("interventions", ["ok", 1]),
            ("dataset_hashes", [HASH, "not-a-hash"]),
            ("dataset_hashes", [HASH, 1]),
            ("dataset_hashes", [HASH, {}]),
            ("dataset_hashes", [HASH, HASH]),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(
                ResearchGitError
            ):
                _payload({field: value})

    def test_runtime_scope_matching_rejects_contract_drift(self) -> None:
        signed = {
            "arm_ids": ["latent", "observable_time_aware", "suffix_k1"],
            "resource_contract_hash": HASH,
        }
        self.assertTrue(ResearchLifecycle._admission_scope_matches(signed, signed))
        mutations = (
            {"arm_ids": ["latent", "suffix_k1"], "resource_contract_hash": HASH},
            {"arm_ids": ["latent", "suffix_k1", "suffix_k1"], "resource_contract_hash": HASH},
            {"arm_ids": ["suffix_k1", "observable_time_aware", "latent"], "resource_contract_hash": HASH},
            {"arm_ids": signed["arm_ids"], "resource_contract_hash": "sha256:" + "b" * 64},
            {"arm_ids": signed["arm_ids"]},
            {"arm_ids": signed["arm_ids"], "resource_contract_hash": True},
        )
        for actual in mutations:
            with self.subTest(actual=actual):
                self.assertFalse(ResearchLifecycle._admission_scope_matches(signed, actual))

    def test_runtime_tuple_seeds_keep_legacy_subset_matching(self) -> None:
        signed = {"allowed_seeds": [7, 11]}
        self.assertTrue(
            ResearchLifecycle._admission_scope_matches(signed, {"seeds": (7,)})
        )
        self.assertTrue(
            ResearchLifecycle._admission_scope_overlaps(signed, {"seeds": (7,)})
        )

    def test_runtime_seed_tuple_keeps_legacy_scope_matching(self) -> None:
        signed = {"allowed_seeds": [7, 11]}
        actual = {"seeds": (7,)}
        self.assertTrue(ResearchLifecycle._admission_scope_matches(signed, actual))
        self.assertTrue(ResearchLifecycle._admission_scope_overlaps(signed, actual))

    def test_invalid_signed_scope_is_rejected_before_schema_hashing(self) -> None:
        payload = _payload(
            {"arm_ids": ["latent", "suffix_k1"], "resource_contract_hash": HASH}
        )
        for mutation in (
            {"arm_ids": ["suffix_k1", "latent"]},
            {"arm_ids": ["latent", "latent"]},
            {"resource_contract_hash": "sha256:" + "b" * 64},
        ):
            candidate = copy.deepcopy(payload)
            candidate["scope"].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ResearchGitError):
                normalize_experiment_admission_payload(candidate)

    def test_published_schema_rejects_mutually_exclusive_seed_fields(self) -> None:
        payload = _payload({"allowed_seeds": [7, 11]})
        payload["scope"]["seeds"] = [7]
        payload["scope_hash"] = canonical_content_hash(payload["scope"])
        self.assertFalse(schema_validator("experiment_admission").is_valid(payload))


if __name__ == "__main__":
    unittest.main()
