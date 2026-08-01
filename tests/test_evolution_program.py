from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ai_scientist.utils.evolution_program import (
    CORE_PROGRAM_POLICY,
    build_evolution_program,
    save_evolution_program,
    validate_evolution_program,
)
from ai_scientist.utils.pipeline_contracts import initialize_pipeline_contracts
from ai_scientist.utils.science_constitution import (
    build_science_constitution,
    save_science_constitution,
)


def _constitution() -> dict:
    return build_science_constitution(project_name="evolution-program-test")


def _self_evolution() -> dict:
    risks = [
        ("evidence_depth_gap", "experiment", "ablation", "p0"),
        ("clarity_gap", "manuscript", "clarity", "p1"),
        ("reproducibility_gap", "experiment", "metadata", "p0"),
        ("verification_path_gap", "review", "verification", "p0"),
        ("stage_standard_blocker", "planning", "budget", "p0"),
        ("persistent_reviewer_debt", "review", "retry", "p1"),
        ("claim_evidence_gap", "planning", "claims", "p1"),
        ("figure_traceability_gap", "figure", "figures", "p2"),
        ("novelty_positioning_gap", "planning", "literature", "p2"),
    ]
    lessons = []
    for index, (risk, stage, focus, priority) in enumerate(risks):
        lessons.append(
            {
                "lesson_id": f"lesson:{index}",
                "source": "review_state",
                "priority_tier": priority,
                "stage": stage,
                "focus": focus,
                "risk": risk,
                "signal": f"Observed {risk} in project evidence.",
                "recommended_action": f"Address {risk} with one bounded change.",
            }
        )
    lessons.append({**lessons[0], "lesson_id": "lesson:duplicate-source"})
    return {
        "schema_version": 2,
        "summary": {"status": "ready", "lesson_count": len(lessons)},
        "self_check": {"status": "ready", "required_failures": []},
        "lessons": lessons,
        "stage_risks": ["experiment: evidence_depth_gap"],
    }


def _playbook() -> dict:
    return {
        "project_count": 4,
        "top_recurring_risks": [
            {"risk": "evidence_depth_gap", "count": 4},
            {"risk": "reproducibility_gap", "count": 3},
        ],
        "top_agentic_defaults": [],
    }


class EvolutionProgramTests(unittest.TestCase):
    def test_program_builds_bounded_quality_diverse_atomic_portfolio(self) -> None:
        program = build_evolution_program(
            "/tmp/evolution-program-test",
            constitution=_constitution(),
            self_evolution=_self_evolution(),
            playbook=_playbook(),
            gate_history=[{"decision": "hold"}, {"decision": "approved"}],
        )

        self.assertLessEqual(
            len(program["intents"]),
            CORE_PROGRAM_POLICY["maximum_active_intents"],
        )
        counts = Counter(item["component_type"] for item in program["intents"])
        self.assertTrue(
            all(
                count <= CORE_PROGRAM_POLICY["maximum_intents_per_component"]
                for count in counts.values()
            )
        )
        self.assertGreaterEqual(len(counts), 3)
        self.assertTrue(
            all(len(item["change_scope"]) == 1 for item in program["intents"])
        )
        self.assertTrue(
            all(
                item["automatic_application_allowed"] is False
                for item in program["intents"]
            )
        )
        evidence_intent = next(
            item
            for item in program["intents"]
            if item["failure_class"] == "evidence_depth_gap"
        )
        self.assertEqual(evidence_intent["search_mode"], "exploit")
        self.assertEqual(evidence_intent["support_count"], 2)
        self.assertEqual(program["archive_summary"]["gate_decision_counts"]["hold"], 1)

    def test_epoch_freezes_utility_and_defers_evaluator_changes(self) -> None:
        program = build_evolution_program(
            "/tmp/evolution-program-test",
            constitution=_constitution(),
            self_evolution=_self_evolution(),
            playbook=_playbook(),
        )
        self.assertEqual(program["epoch"]["utility_mode"], "fixed_within_epoch")
        self.assertEqual(
            program["epoch"]["evaluator_change_window"], "epoch_boundary_only"
        )
        self.assertTrue(program["evaluation_challenges"])
        self.assertTrue(
            all(
                item["automatic_application_allowed"] is False
                for item in program["evaluation_challenges"]
            )
        )
        self.assertTrue(
            validate_evolution_program(program, constitution=_constitution())["passed"]
        )

    def test_repeated_gate_holds_force_a_new_search_branch(self) -> None:
        program = build_evolution_program(
            "/tmp/evolution-program-test",
            constitution=_constitution(),
            self_evolution=_self_evolution(),
            playbook=_playbook(),
            gate_history=[
                {
                    "decision": "hold",
                    "failure_taxonomy_refs": ["failure:evidence-depth-gap"],
                },
                {
                    "decision": "hold",
                    "failure_taxonomy_refs": ["failure:evidence-depth-gap"],
                },
            ],
        )
        intent = next(
            item
            for item in program["intents"]
            if item["failure_class"] == "evidence_depth_gap"
        )
        self.assertEqual(intent["search_mode"], "explore")
        self.assertEqual(intent["prior_gate_outcomes"], {"hold": 2})

    def test_portfolio_reserves_exploration_when_all_signals_recur(self) -> None:
        playbook = _playbook()
        playbook["top_recurring_risks"] = [
            {"risk": lesson["risk"], "count": 5}
            for lesson in _self_evolution()["lessons"]
        ]
        program = build_evolution_program(
            "/tmp/evolution-program-test",
            constitution=_constitution(),
            self_evolution=_self_evolution(),
            playbook=playbook,
        )
        required = 3
        explored = [
            item for item in program["intents"] if item["search_mode"] == "explore"
        ]
        self.assertGreaterEqual(len(explored), required)
        self.assertTrue(
            all(item["mechanism_constraint"] for item in program["intents"])
        )
        self.assertTrue(
            validate_evolution_program(program, constitution=_constitution())["passed"]
        )

    def test_semantic_tampering_is_rejected(self) -> None:
        program = build_evolution_program(
            "/tmp/evolution-program-test",
            constitution=_constitution(),
            self_evolution=_self_evolution(),
            playbook=_playbook(),
        )
        program["intents"][0]["automatic_application_allowed"] = True
        check = validate_evolution_program(program, constitution=_constitution())
        self.assertFalse(check["passed"])
        self.assertTrue(
            any("automatic_application_forbidden" in item for item in check["errors"])
        )

    def test_saved_program_is_append_only_across_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "projects" / "demo"
            root.mkdir(parents=True)
            initialize_pipeline_contracts(root)
            constitution = _constitution()
            save_science_constitution(root, constitution, producer="test")
            first = build_evolution_program(
                root,
                constitution=constitution,
                self_evolution=_self_evolution(),
                playbook=_playbook(),
            )
            save_evolution_program(
                root, first, constitution=constitution, producer="test"
            )
            history = root / "evolution_program_history.jsonl"
            self.assertEqual(len(history.read_text(encoding="utf-8").splitlines()), 1)

            second = build_evolution_program(
                root,
                constitution=constitution,
                self_evolution=_self_evolution(),
                playbook=_playbook(),
                program_history=[
                    {
                        "program_id": first["program_id"],
                        "intents": first["intents"],
                    }
                ],
            )
            save_evolution_program(
                root, second, constitution=constitution, producer="test"
            )
            self.assertEqual(second["epoch"]["epoch_index"], 2)
            self.assertEqual(len(history.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
