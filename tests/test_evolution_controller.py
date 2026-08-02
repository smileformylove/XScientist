from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.evolution_controller import (
    assess_evolution_epoch,
    build_evolution_trial,
    evaluate_and_save_evolution_control,
    record_evolution_trial,
    validate_evolution_trial,
)
from ai_scientist.utils.evolution_program import (
    build_evolution_program,
    save_evolution_program,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    load_pipeline_manifest,
)
from ai_scientist.utils.science_constitution import (
    build_science_constitution,
    save_science_constitution,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _program(project_root: str = "/tmp/evolution-controller") -> tuple[dict, dict]:
    constitution = build_science_constitution(project_name="controller-test")
    evolution = {
        "summary": {"status": "ready", "lesson_count": 1},
        "self_check": {"status": "ready", "required_failures": []},
        "lessons": [
            {
                "lesson_id": "lesson:one",
                "source": "review_state",
                "priority_tier": "p0",
                "stage": "experiment",
                "focus": "ablation",
                "risk": "evidence_depth_gap",
                "signal": "The causal mechanism lacks an ablation.",
                "recommended_action": "Test one mechanism with a paired ablation.",
            }
        ],
        "stage_risks": ["experiment: evidence_depth_gap"],
    }
    program = build_evolution_program(
        project_root,
        constitution=constitution,
        self_evolution=evolution,
    )
    return program, constitution


def _trial(
    program: dict,
    index: int,
    *,
    status: str = "held",
    mechanism: str | None = None,
    utility_delta: float | None = 0.0,
    hard_gate_failures: list[str] | None = None,
) -> dict:
    return build_evolution_trial(
        program,
        trial_id=f"trial:{index}",
        intent_id=program["intents"][0]["intent_id"],
        candidate_hash=_digest(f"candidate:{index}"),
        mechanism_hash=_digest(mechanism or f"mechanism:{index}"),
        status=status,
        utility_delta=utility_delta,
        cost_units=1.0,
        hard_gate_failures=hard_gate_failures,
    )


class EvolutionControllerTests(unittest.TestCase):
    def test_empty_epoch_prioritizes_untried_intents_without_stopping(self) -> None:
        program, _ = _program()

        control = assess_evolution_epoch(program, [])

        self.assertEqual(control["status"], "continue")
        self.assertFalse(control["should_stop"])
        self.assertEqual(
            control["next_intent_ids"],
            [program["intents"][0]["intent_id"]],
        )
        self.assertFalse(control["automatic_production_promotion_allowed"])

    def test_repeated_mechanism_is_not_counted_as_search_diversity(self) -> None:
        program, _ = _program()
        trials = [
            _trial(program, 0, mechanism="same", utility_delta=0.02),
            _trial(program, 1, mechanism="same", utility_delta=0.02),
        ]

        control = assess_evolution_epoch(program, trials)

        state = control["intent_states"][0]
        self.assertEqual(state["status"], "needs_new_mechanism")
        self.assertEqual(state["duplicate_mechanism_count"], 1)
        self.assertEqual(control["mechanism_diversity"], 0.5)

    def test_futility_patience_stops_unproductive_intent(self) -> None:
        program, _ = _program()
        control = assess_evolution_epoch(
            program,
            [_trial(program, 0), _trial(program, 1)],
        )

        self.assertEqual(control["status"], "complete")
        self.assertTrue(control["should_stop"])
        self.assertEqual(
            control["intent_states"][0]["status"],
            "stopped_for_futility",
        )

    def test_gate_eligible_trial_reports_discovery_efficiency(self) -> None:
        program, _ = _program()
        control = assess_evolution_epoch(
            program,
            [_trial(program, 0, status="gate_eligible", utility_delta=0.03)],
        )

        self.assertEqual(control["status"], "complete")
        self.assertEqual(control["gate_eligible_intent_count"], 1)
        self.assertEqual(control["discovery_efficiency"], 1.0)

    def test_integrity_or_safety_failure_halts_epoch(self) -> None:
        program, _ = _program()
        control = assess_evolution_epoch(
            program,
            [
                _trial(
                    program,
                    0,
                    hard_gate_failures=["safety_regression"],
                )
            ],
        )

        self.assertEqual(control["status"], "halted")
        self.assertIn("safety_regression", control["epoch_halting_failures"])

    def test_trial_hash_and_epoch_binding_are_validated(self) -> None:
        program, _ = _program()
        trial = _trial(program, 0)
        self.assertTrue(validate_evolution_trial(trial, program=program)["passed"])

        trial["utility_delta"] = 9.0
        self.assertIn(
            "trial_hash_mismatch",
            validate_evolution_trial(trial, program=program)["errors"],
        )

    def test_program_save_initializes_and_updates_control_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            program, constitution = _program(str(root))
            save_science_constitution(root, constitution, producer="test")
            save_evolution_program(
                root,
                program,
                constitution=constitution,
                producer="test",
            )
            initial = load_contract_artifact(root, "evolution_control", default={})
            self.assertEqual(initial["status"], "continue")

            trial = _trial(program, 0, status="gate_eligible", utility_delta=0.03)
            record_evolution_trial(root, trial, program=program)
            updated = evaluate_and_save_evolution_control(
                root,
                program=program,
                producer="test",
            )
            manifest = load_pipeline_manifest(root)

        self.assertEqual(updated["status"], "complete")
        self.assertEqual(manifest["artifacts"]["evolution_control"]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
