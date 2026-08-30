from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.apps.daemon import build_parser as build_daemon_parser
from ai_scientist.apps.daemon_sources import _validate_source_config
from ai_scientist.apps.project_cli import build_parser as build_project_parser
from ai_scientist.config.venues import DEFAULT_TARGET_VENUE, TARGET_VENUES
from ai_scientist.professional_writing_system import get_template_info
from ai_scientist.review_strategies import ReviewStrategy
from ai_scientist.utils.high_quality_pipeline import (
    VENUE_PRESETS,
    is_paper_type_fit_for_venue,
    recommend_paper_type_for_venue,
    resolve_target_venue,
)
from ai_scientist.utils.review_workflow import build_review_execution_plan
from ai_scientist.utils.workflow_cli import (
    apply_project_autopilot_profile,
    normalize_project_workflow_args,
)
from ai_scientist.utils.writeup_workflow import build_writeup_execution_plan
from xscientist.cli import _build_start_parser
from xscientist.models import ProjectRequest


class TargetVenueContractTests(unittest.TestCase):
    def test_icml_is_accepted_by_public_and_project_parsers(self) -> None:
        start_args = _build_start_parser().parse_args(
            [
                "study",
                "--question",
                "Does the intervention improve robustness?",
                "--autopilot",
                "publication",
                "--target-venue",
                "icml",
                "--allow-synthetic-data",
            ]
        )
        self.assertEqual(start_args.target_venue, "icml")
        self.assertIn(
            "not an acceptance prediction or guarantee",
            " ".join(_build_start_parser().format_help().split()),
        )

        parser = build_project_parser(
            default_output_root="/tmp/research",
            default_writing_profile="default",
            writing_profiles=["default"],
            workflow_modes=["adaptive", "multi_agent_board"],
        )
        project_args = parser.parse_args(
            [
                "study",
                "--question",
                "Does the intervention improve robustness?",
                "--autopilot",
                "publication",
                "--target-venue",
                "icml",
                "--review-strategy",
                "icml",
            ]
        )
        apply_project_autopilot_profile(project_args)
        normalize_project_workflow_args(project_args)

        self.assertEqual(project_args.target_venue, "icml")
        self.assertEqual(project_args.review_strategy, "icml")
        self.assertEqual(project_args.quality_preset, "publishable")
        self.assertEqual(project_args.workflow_mode, "multi_agent_board")

    def test_default_target_remains_neurips(self) -> None:
        self.assertEqual(DEFAULT_TARGET_VENUE, "neurips")
        self.assertEqual(TARGET_VENUES[0], "neurips")
        self.assertEqual(resolve_target_venue("normal"), "neurips")
        parsed = _build_start_parser().parse_args(
            ["study", "--question", "Does X affect Y?", "--prepare-only"]
        )
        self.assertEqual(parsed.target_venue, "neurips")

    def test_sdk_target_venue_resolution_is_normalized_and_fail_closed(self) -> None:
        self.assertEqual(resolve_target_venue("normal", " ICML "), "icml")
        with self.assertRaisesRegex(ValueError, "target_venue must be one of"):
            resolve_target_venue("normal", "neurip")

    def test_icml_quality_writeup_and_review_policies_are_first_class(self) -> None:
        preset = VENUE_PRESETS["icml"]
        self.assertEqual(preset["template"], "icml")
        self.assertEqual(get_template_info("icml")["name"], "ICML")
        self.assertGreaterEqual(preset["rigor_threshold"], 3.8)
        self.assertTrue(
            any("Statistical" in item for item in preset["checklist"]),
            preset["checklist"],
        )
        self.assertTrue(is_paper_type_fit_for_venue("normal", "icml"))
        self.assertEqual(recommend_paper_type_for_venue("icml"), "normal")

        with tempfile.TemporaryDirectory() as td:
            writeup = build_writeup_execution_plan(
                "normal",
                num_cite_rounds=1,
                writeup_retries=1,
                target_venue="icml",
                high_quality_mode=True,
                research_root=td,
            )
            review = build_review_execution_plan(
                "normal",
                target_venue="icml",
                high_quality_mode=True,
                research_root=td,
            )

        self.assertEqual(writeup["target_venue"], "icml")
        self.assertGreaterEqual(writeup["num_cite_rounds"], 20)
        self.assertGreaterEqual(writeup["writeup_retries"], 4)
        self.assertIn(
            "reproducibility", writeup["manuscript_policy"]["required_sections"]
        )
        self.assertEqual(review["strategy"], ReviewStrategy.ICML)
        self.assertGreaterEqual(review["review_ensemble"], 3)

    def test_sdk_and_daemon_validation_share_icml_contract(self) -> None:
        request = ProjectRequest(
            project="study",
            question="Does X affect Y?",
            target_venue="icml",
            autopilot="publication",
        )
        argv = request.to_argv()
        self.assertEqual(argv[argv.index("--target-venue") + 1], "icml")
        with self.assertRaisesRegex(ValueError, "target_venue must be one of"):
            ProjectRequest(
                project="study",
                question="Does X affect Y?",
                target_venue="not-a-venue",
            ).to_argv()

        source_payload = {
            "sources": [
                {
                    "type": "topic",
                    "value": "topic.md",
                    "target_venue": "icml",
                    "day_target_venue": "icml",
                    "night_target_venue": "icml",
                }
            ]
        }
        self.assertEqual(_validate_source_config(source_payload), [])
        daemon_args = build_daemon_parser().parse_args(
            ["--rewrite-board-venue", "icml", "--shortlist-venue", "icml"]
        )
        self.assertEqual(daemon_args.rewrite_board_venue, "icml")
        self.assertEqual(daemon_args.shortlist_venue, "icml")

        root = Path(__file__).resolve().parents[1]
        source_schema = json.loads(
            (root / "configs/sources/source_queue.schema.json").read_text(
                encoding="utf-8"
            )
        )
        daemon_schema = json.loads(
            (root / "configs/daemon/daemon_profile.schema.json").read_text(
                encoding="utf-8"
            )
        )
        source_properties = source_schema["properties"]["sources"]["items"][
            "properties"
        ]
        self.assertIn("icml", source_properties["target_venue"]["enum"])
        self.assertIn("icml", source_properties["day_target_venue"]["enum"])
        self.assertIn("icml", source_properties["night_target_venue"]["enum"])
        self.assertIn(
            "icml", daemon_schema["properties"]["rewrite_board_venue"]["enum"]
        )
        self.assertIn("icml", daemon_schema["properties"]["shortlist_venue"]["enum"])


if __name__ == "__main__":
    unittest.main()
