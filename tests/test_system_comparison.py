from __future__ import annotations

from copy import deepcopy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from jsonschema import ValidationError, validate

import xscientist
from ai_scientist.protocol.schemas import load_schema
from xscientist.cli import main
from xscientist.demo import create_autopilot_demo
from xscientist.system_comparison import (
    CAPABILITY_STATUSES,
    COMPARISON_DIMENSIONS,
    build_system_comparison,
)


class SystemComparisonTests(unittest.TestCase):
    def test_report_is_non_ranking_and_schema_valid(self) -> None:
        report = build_system_comparison()
        validate(report, load_schema("system_comparison"))
        self.assertTrue(report["ok"])
        self.assertFalse(report["official_comparable"])
        self.assertFalse(report["score_claim_allowed"])
        self.assertFalse(report["quality_claim_allowed"])
        self.assertEqual(report["external_rollouts"], 0)
        self.assertEqual(len(report["dimensions"]), len(COMPARISON_DIMENSIONS))
        self.assertGreaterEqual(len(report["systems"]), 16)
        self.assertEqual(report["source_manifest"]["attached_talk"]["page_count"], 107)
        self.assertIn(
            "not an officially supported product",
            report["source_policy"]["google_repository_disclaimer"],
        )
        self.assertIn(
            "CAST example paper",
            report["talk_inventory"]["adjacent_references_not_ranked"],
        )
        self.assertIn(
            "FAR (adjacent discovery/allocation reference; not named in attached talk)",
            report["talk_inventory"]["adjacent_references_not_ranked"],
        )
        self.assertIn(
            "ScientistTwo (slide 105; future concept, not an evaluated system)",
            report["talk_inventory"]["context_only_mentions"],
        )

    def test_far_is_source_audited_without_a_human_or_local_score(self) -> None:
        report = build_system_comparison()
        far = {row["id"]: row for row in report["systems"]}["far"]
        self.assertEqual(far["source_status"], "reported_primary")
        self.assertEqual(far["comparison_status"], "not_measured_here")
        self.assertEqual(far["talk_slides"], [])
        self.assertEqual(far["human_evidence"]["status"], "not_reported")
        self.assertFalse(far["human_evidence"]["same_condition"])
        self.assertIsNone(far["human_evidence"]["score"])
        self.assertIn(
            "No same-condition recruited human task-performance arm",
            far["human_evidence"]["note"],
        )
        urls = {source["url"] for source in far["sources"]}
        self.assertIn("https://arxiv.org/abs/2608.16977", urls)
        self.assertIn("https://github.com/zeyu-zheng/FAR", urls)
        self.assertTrue(
            any(
                "not a benchmark accuracy estimate" in item
                for item in far["limitations"]
            )
        )

    def test_schema_rejects_unrecognized_comparison_status(self) -> None:
        report = deepcopy(build_system_comparison())
        report["systems"][0]["comparison_status"] = "ranked"
        with self.assertRaises(ValidationError):
            validate(report, load_schema("system_comparison"))

    def test_every_row_declares_all_dimensions_and_source_boundary(self) -> None:
        dimension_ids = {item["id"] for item in COMPARISON_DIMENSIONS}
        report = build_system_comparison()
        for row in report["systems"]:
            self.assertEqual(set(row["capabilities"]), dimension_ids, row["id"])
            self.assertTrue(
                set(row["capabilities"].values()).issubset(CAPABILITY_STATUSES),
                row["id"],
            )
            self.assertTrue(row["sources"], row["id"])
            self.assertNotEqual(row["comparison_status"], "ranked")
            self.assertIn(
                row["human_evidence"]["status"],
                {
                    "not_reported",
                    "human_reference_proxy",
                    "human_SOTA_reference",
                    "human_judgment_calibration",
                    "human_agent_process",
                },
            )
            self.assertFalse(row["human_evidence"]["same_condition"])
            self.assertIsNone(row["human_evidence"]["score"])
            self.assertTrue(
                all(
                    1
                    <= slide
                    <= report["source_manifest"]["attached_talk"]["page_count"]
                    for slide in row["talk_slides"]
                ),
                row["id"],
            )

        # These are deliberately useful adjacent references, not invented
        # slide citations from the attached talk.
        rows = {row["id"]: row for row in report["systems"]}
        self.assertEqual(rows["mle_star"]["talk_slides"], [])
        self.assertEqual(rows["ds_star"]["talk_slides"], [])
        self.assertIn(
            "MLE-STAR (adjacent execution-layer reference; not named in attached talk)",
            report["talk_inventory"]["adjacent_references_not_ranked"],
        )

    def test_workspace_process_is_optional_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "demo"
            create_autopilot_demo(root, profile="balanced", language="en")
            report = build_system_comparison(root)
        validate(report, load_schema("system_comparison"))
        local = report["xscientist_local"]
        self.assertEqual(local["status"], "local_process_audit")
        self.assertFalse(local["path_disclosed"])
        self.assertEqual(local["rollouts"], 0)
        self.assertEqual(local["rollout_scope"], "this_audit_only")
        self.assertEqual(local["cost_scope"], "this_audit_only")
        self.assertEqual(local["historical_trajectory_cost"], "unobserved")
        self.assertIsNone(local["score"])
        self.assertNotIn(str(root), json.dumps(report))

    def test_sdk_export_and_cli_surface(self) -> None:
        self.assertEqual(
            xscientist.build_system_comparison.__name__, "build_system_comparison"
        )
        self.assertEqual(
            xscientist.persist_benchmark_report.__name__, "persist_benchmark_report"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["benchmark", "systems"])
        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertIn("qualitative source audit", rendered)
        self.assertIn("no cross-system score", rendered)
