from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_pipeline_manifest,
)
from ai_scientist.utils.science_constitution import (
    ScienceConstitutionError,
    build_science_constitution,
    propose_science_constitution_amendment,
    save_science_constitution,
    validate_science_constitution,
)


class ScienceConstitutionTests(unittest.TestCase):
    def test_core_policy_is_locked_and_tamper_evident(self) -> None:
        constitution = build_science_constitution(project_name="grand-discovery")
        self.assertTrue(validate_science_constitution(constitution)["passed"])
        self.assertFalse(
            constitution["core_policy"]["amendment_policy"][
                "automatic_amendment_allowed"
            ]
        )

        constitution["core_policy"]["priority_order"] = ["throughput", "truth"]
        result = validate_science_constitution(constitution)
        self.assertFalse(result["passed"])
        self.assertIn("core_policy_modified", result["errors"])

    def test_amendment_is_a_proposal_and_never_applies_automatically(self) -> None:
        constitution = build_science_constitution(project_name="demo")
        proposed = propose_science_constitution_amendment(
            constitution,
            proposed_by="human:principal-investigator",
            rationale="Add a stricter domain constraint.",
            impact_assessment="No core principle is weakened.",
            proposed_changes={"additional_constraint": "Require two laboratories."},
        )

        self.assertEqual(constitution["amendment_proposals"], [])
        self.assertEqual(len(proposed["amendment_proposals"]), 1)
        self.assertFalse(
            proposed["amendment_proposals"][0]["automatic_application_allowed"]
        )
        self.assertTrue(validate_science_constitution(proposed)["passed"])

    def test_incomplete_amendment_is_rejected(self) -> None:
        with self.assertRaises(ScienceConstitutionError):
            propose_science_constitution_amendment(
                build_science_constitution(project_name="demo"),
                proposed_by="agent",
                rationale="",
                impact_assessment="unknown",
                proposed_changes={},
            )

    def test_save_registers_constitution_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            output = save_science_constitution(
                root,
                build_science_constitution(project_name="project"),
                producer="test",
            )

            self.assertTrue(Path(output).exists())
            manifest = load_pipeline_manifest(root)
            self.assertEqual(
                manifest["artifacts"]["science_constitution"]["status"],
                "ready",
            )


if __name__ == "__main__":
    unittest.main()
