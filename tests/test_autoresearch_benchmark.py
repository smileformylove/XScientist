from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from xscientist.benchmark import (
    _metacognitive_report,
    benchmark_autoresearch_pilot,
    persist_benchmark_report,
)
from xscientist.cli import main
from xscientist.demo import create_autopilot_demo
from xscientist.process_audit import _artifact_row, build_process_summary
import xscientist.process_audit as process_audit_module
from xscientist.research_vcs import ResearchRepository
from ai_scientist.protocol.schemas import load_schema
from jsonschema import validate


class AutoResearchBenchmarkTests(unittest.TestCase):
    def _tasks(self, root: Path) -> Path:
        path = root / "tasks.jsonl"
        path.write_text(
            json.dumps(
                {
                    "task_id": "W-test",
                    "domain": "chemistry",
                    "premise": "A premise",
                    "tension": "A tension",
                    "conclusion_gt": "must never be returned",
                    "source_paper": {"doi": "secret-gold"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_pilot_is_offline_and_does_not_leak_gold_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report = benchmark_autoresearch_pilot(self._tasks(Path(raw)), limit=1)
        self.assertTrue(report["ok"])
        self.assertFalse(report["official_comparable"])
        self.assertEqual(
            report["comparison_context"]["mode"], "qualitative_source_audit"
        )
        self.assertFalse(report["comparison_context"]["external_scores_injected"])
        self.assertFalse(report["execution"]["network_used"])
        self.assertFalse(report["tasks"]["gold_fields_used"])
        self.assertEqual(report["human_baseline"]["status"], "not_reported")
        self.assertEqual(report["human_baseline"]["evidence_class"], "not_reported")
        self.assertFalse(report["human_baseline"]["matched_arm"])
        self.assertIsNone(report["human_baseline"]["score"])
        self.assertEqual(report["human_baseline"]["local_runs"], 0)
        self.assertFalse(report["human_baseline"]["external_scores_injected"])
        retention = report["evidence_retention"]
        self.assertEqual(retention["mode"], "read_only_bounded_index")
        self.assertFalse(retention["task_manifest_copied"])
        self.assertFalse(retention["raw_trajectory_copied"])
        self.assertFalse(retention["ara_snapshot_written"])
        self.assertFalse(retention["cas_payload_copied"])
        self.assertTrue(retention["workspace_artifacts_untouched"])
        self.assertNotIn("must never be returned", json.dumps(report))
        self.assertNotIn("secret-gold", json.dumps(report))
        self.assertFalse(report["quality_claim_allowed"])
        self.assertEqual(
            report["score_semantics"], "task_contract_and_structural_observability_only"
        )
        self.assertEqual(
            report["diagnostics"]["next_required"], "QUALITY.NO_MATCHED_ROLLOUT"
        )
        validate(report, load_schema("autoresearch_conformance"))

    def test_diagnostics_turn_structural_gaps_into_bounded_actions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "study"
            root.mkdir()
            report = benchmark_autoresearch_pilot(
                self._tasks(Path(raw)), workspace=root, limit=1
            )
        diagnostics = report["diagnostics"]
        self.assertFalse(diagnostics["quality_claim_allowed"])
        self.assertTrue(diagnostics["items"])
        self.assertIn(
            "QUALITY.NO_MATCHED_ROLLOUT", [item["id"] for item in diagnostics["items"]]
        )
        self.assertTrue(all("payload" not in item for item in diagnostics["items"]))

    def test_persist_benchmark_report_is_atomic_and_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "nested" / "report.json"
            report = {"schema": "test", "raw_payloads_included": False}
            persisted = persist_benchmark_report(report, destination)
            self.assertEqual(persisted, destination.resolve())
            self.assertEqual(json.loads(destination.read_text()), report)
            with self.assertRaises(ValueError):
                persist_benchmark_report(report, destination.parent)

    def test_pilot_redacts_untrusted_task_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "SECRET_TASK_FILE.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "SECRET_TASK_ID",
                        "domain": "SECRET_DOMAIN",
                        "tension": "SECRET_TENSION",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = benchmark_autoresearch_pilot(path, limit=1)
        rendered = json.dumps(report)
        for secret in (
            "SECRET_TASK_FILE",
            "SECRET_TASK_ID",
            "SECRET_DOMAIN",
            "SECRET_TENSION",
        ):
            self.assertNotIn(secret, rendered)
        self.assertEqual(report["tasks"]["domains"], {"other": 1})

    def test_metacognition_redacts_untrusted_object_identifiers(self) -> None:
        report = _metacognitive_report(
            [
                {
                    "kind": "review",
                    "object_id": "SECRET_OBJECT_IDENTIFIER",
                    "state": "completed",
                    "payload": {"issues": ["SECRET_REVIEW_TEXT"]},
                    "relations": [],
                }
            ]
        )
        rendered = json.dumps(report)
        self.assertNotIn("SECRET_OBJECT_IDENTIFIER", rendered)
        self.assertNotIn("SECRET_REVIEW_TEXT", rendered)
        self.assertEqual(report["unresolved_issues"][0]["source_kind"], "review")

    def test_process_rows_digest_untrusted_identifiers(self) -> None:
        row = _artifact_row(
            {
                "object_id": "SECRET_OBJECT_IDENTIFIER",
                "kind": "hypothesis",
                "state": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "content_hash": "SECRET_CONTENT_HASH",
                "payload": {},
                "relations": [],
                "provenance": {},
            }
        )
        rendered = json.dumps(row)
        self.assertNotIn("SECRET_OBJECT_IDENTIFIER", rendered)
        self.assertNotIn("SECRET_CONTENT_HASH", rendered)
        self.assertTrue(row["object_id"].startswith("sha256:"))

    def test_workspace_stage_coverage_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "study"
            (root / ".xscientist" / "objects" / "question").mkdir(parents=True)
            (root / ".xscientist" / "objects" / "review").mkdir(parents=True)
            (root / "02_experiments").mkdir(parents=True)
            (root / "question.md").write_text("question", encoding="utf-8")
            (root / ".xscientist" / "objects" / "question" / "q.json").write_text("{}")
            (root / ".xscientist" / "objects" / "review" / "r.json").write_text("{}")
            report = benchmark_autoresearch_pilot(
                self._tasks(Path(raw)), workspace=root, limit=1
            )
        stages = report["workspace"]["stages"]
        # Directory names and a single placeholder object are not enough to
        # claim a complete lifecycle stage; the pilot requires its minimum
        # typed evidence and reports the rest as partial/missing.
        self.assertFalse(stages["ideation_planning"]["covered"])
        self.assertFalse(stages["execution_implementation"]["covered"])
        self.assertFalse(stages["self_verification_review"]["covered"])
        self.assertFalse(stages["retrieval_synthesis"]["covered"])
        self.assertEqual(report["workspace"]["stage_coverage"], 0.0)
        self.assertEqual(report["workspace"]["stage_score"], 0.0)
        self.assertEqual(report["workspace"]["object_scan"]["visible_object_count"], 0)
        self.assertEqual(report["workspace"]["object_scan"]["source_object_count"], 2)

    def test_cli_autoresearch_pilot_emits_structured_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tasks = self._tasks(Path(raw))
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    ["benchmark", "autoresearch", "--tasks", str(tasks), "--json"]
                )
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
        self.assertEqual(report["schema"], "xscientist.autoresearch-conformance.v1")
        self.assertEqual(report["execution"]["rollouts_evaluated"], 0)

    def test_cli_can_persist_redacted_report_without_polluting_json_stdout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tasks = self._tasks(Path(raw))
            destination = Path(raw) / "audit" / "report.json"
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "benchmark",
                        "autoresearch",
                        "--tasks",
                        str(tasks),
                        "--output",
                        str(destination),
                        "--json",
                    ]
                )
            stdout_report = json.loads(output.getvalue())
            disk_report = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(stdout_report, disk_report)
        self.assertTrue(stdout_report["report_persistence"]["requested"])
        self.assertFalse(stdout_report["report_persistence"]["raw_payloads_included"])
        validate(stdout_report, load_schema("autoresearch_conformance"))

    def test_cli_show_process_surfaces_structured_intermediate_and_fairness(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "demo"
            create_autopilot_demo(root, profile="balanced", language="en")
            tasks = self._tasks(Path(raw))
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "benchmark",
                        "autoresearch",
                        "--tasks",
                        str(tasks),
                        "--workspace",
                        str(root),
                        "--limit",
                        "1",
                        "--show-process",
                    ]
                )
        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertIn("Branch topology:", rendered)
        self.assertIn("Reasoning trail: artifact-backed signals only", rendered)
        self.assertIn("Artifact scope: current_checkout_only", rendered)
        self.assertIn("Artifacts:", rendered)
        self.assertIn("Attempts:", rendered)
        self.assertIn("Fair branch comparison: NOT VERIFIED", rendered)
        self.assertNotIn("must never be returned", rendered)
        self.assertNotIn(str(root), rendered)

    def test_demo_report_distinguishes_contained_review_debt_from_f4_shipping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "demo"
            create_autopilot_demo(root, profile="balanced", language="en")
            report = benchmark_autoresearch_pilot(
                self._tasks(Path(raw)), workspace=root, limit=1
            )

        workspace = report["workspace"]
        self.assertEqual(workspace["metacognition"]["status"], "contained")
        self.assertGreater(workspace["metacognition"]["unresolved_issue_count"], 0)
        self.assertFalse(
            any(
                row["code"] == "XSCIENTIST.F4_UNCORRECTED_SELF_AWARENESS"
                and row["severity"] == "blocker"
                for row in workspace["metacognitive_signals"]
            )
        )
        self.assertTrue(workspace["closure"]["available"])
        self.assertIn("trace", workspace["closure"]["levels"])

    def test_review_debt_without_containing_gate_is_not_claimed_contained(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "study"
            create_autopilot_demo(root, profile="balanced", language="en")
            objects = root / ".xscientist" / "objects"
            for path in objects.glob("gate_decision/*.json"):
                path.unlink()
            report = benchmark_autoresearch_pilot(
                self._tasks(Path(raw)), workspace=root, limit=1
            )

        metacognition = report["workspace"]["metacognition"]
        self.assertEqual(metacognition["status"], "open")
        self.assertFalse(metacognition["containment_gate_observed"])
        self.assertTrue(
            any(
                row["code"] == "XSCIENTIST.F4_OPEN_REVIEW_DEBT"
                for row in report["workspace"]["metacognitive_signals"]
            )
        )

    def test_process_audit_exposes_branches_and_intermediate_artifacts_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "research"
            repository = ResearchRepository.init(
                root,
                name="process-audit-test",
                question="Does H1 hold?",
                git_user_name="Test",
                git_user_email="test@example.invalid",
            )
            hypothesis = repository.record(
                "hypothesis",
                {"statement": "H1 improves the metric.", "falsifier": "No effect."},
            )
            repository.commit(stage="ideation", subject="record H1")
            repository.fork("challenge")
            repository.record(
                "evidence",
                {"summary": "The challenge result is negative.", "effect": -0.1},
                state="completed",
                relations=[{"type": "refutes", "target": hypothesis.object_id}],
            )
            repository.commit(stage="experiment", subject="run challenge")
            repository.switch("main")
            report = build_process_summary(
                root,
                task_manifest_sha256="sha256:manifest",
                task_count=1,
                task_filter="open-ended",
                task_limit=1,
            )

        self.assertTrue(report["available"])
        self.assertEqual(report["branch_topology"]["branch_count"], 2)
        self.assertTrue(report["branch_topology"]["branching_observed"])
        self.assertGreaterEqual(len(report["commits"]), 3)
        # Checkpoint timestamps can share a second; the process view must use
        # Git parentage so the visible trail remains causal rather than hash
        # sorted.
        stages = [row["stage"] for row in report["commits"]]
        self.assertLess(stages.index("init"), stages.index("ideation"))
        self.assertLess(stages.index("ideation"), stages.index("experiment"))
        self.assertRegex(report["commits"][-1]["short_commit"], r"^[0-9a-f]{7,12}$")
        self.assertGreater(report["intermediate"]["object_count"], 0)
        self.assertEqual(report["fairness"]["task_manifest_sha256"], "sha256:manifest")
        self.assertFalse(report["fairness"]["gold_fields_used"])
        self.assertIn("hidden chain-of-thought", report["reasoning_boundary"])
        rendered = json.dumps(report)
        self.assertNotIn(str(root), rendered)
        self.assertNotIn("H1 improves", rendered)
        validate(report, load_schema("process_audit"))

        forced = build_process_summary(root, gold_fields_used=True)
        self.assertFalse(forced["fairness"]["gold_fields_used"])
        redacted_fairness = build_process_summary(
            root,
            task_manifest_sha256="SECRET_MANIFEST_TEXT",
            task_filter="SECRET_FILTER_TEXT",
        )["fairness"]
        self.assertNotIn("SECRET_MANIFEST_TEXT", json.dumps(redacted_fairness))
        self.assertNotIn("SECRET_FILTER_TEXT", json.dumps(redacted_fairness))
        self.assertEqual(redacted_fairness["filter"], "custom")

    def test_process_audit_schema_covers_unavailable_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report = build_process_summary(Path(raw) / "empty")
        self.assertFalse(report["available"])
        validate(report, load_schema("process_audit"))

    def test_process_audit_fail_closes_invalid_fairness_counts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report = build_process_summary(
                Path(raw) / "empty",
                task_count=-1,
                task_limit=1.5,
            )

        fairness = report["fairness"]
        self.assertIsNone(fairness["task_count"])
        self.assertIsNone(fairness["task_limit"])
        self.assertIsNone(fairness["limit"])
        validate(report, load_schema("process_audit"))

    def test_process_audit_redacts_gold_like_text_and_branch_refs(self) -> None:
        secrets = {
            "SECRET_GOLD_PAYLOAD",
            "SECRET_GOLD_SUBJECT",
            "SECRET_GOLD_DECISION",
            "SECRET_GOLD_BRANCH",
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "research"
            repository = ResearchRepository.init(
                root,
                name="redaction-test",
                question="Q",
                git_user_name="Test",
                git_user_email="test@example.invalid",
            )
            hypothesis = repository.record(
                "hypothesis",
                {
                    "statement": "SECRET_GOLD_PAYLOAD",
                    "falsifier": "counterexample",
                },
            )
            repository.commit(stage="ideation", subject="SECRET_GOLD_SUBJECT")
            repository.fork("SECRET_GOLD_BRANCH")
            repository.record(
                "review",
                {
                    "summary": "SECRET_GOLD_PAYLOAD",
                    "status": "completed",
                    "decision": "SECRET_GOLD_DECISION",
                },
                state="completed",
                relations=[{"type": "refutes", "target": hypothesis.object_id}],
            )
            repository.commit(stage="review", subject="SECRET_GOLD_SUBJECT")
            repository.switch("main")
            report = build_process_summary(
                root,
                task_manifest_sha256="sha256:manifest",
                task_count=1,
                task_filter="open-ended",
                task_limit=1,
            )

        rendered = json.dumps(report, ensure_ascii=False)
        for secret in secrets:
            self.assertNotIn(secret, rendered)
        self.assertTrue(report["redaction"]["free_text"] == "omitted")
        self.assertTrue(report["redaction"]["branch_names"] == "alias_plus_digest")
        self.assertTrue(
            all(
                row["name"].startswith(("current", "alternative-"))
                for row in report["branches"]
            )
        )
        self.assertFalse(
            report["branch_topology"]["fair_branch_comparison"]["eligible"]
        )
        self.assertEqual(report["fairness"]["filter"], "open-ended")
        self.assertEqual(report["fairness"]["limit"], 1)

    def test_process_audit_bounds_reads_and_marks_truncation(self) -> None:
        branches = [
            {
                "name": "main" if index == 0 else f"line-{index}",
                "current": index == 0,
                "commit": f"{index + 1:040x}",
                "stage": "experiment",
                "status": "completed",
                "checkpoint_id": f"rcp-{index:08x}",
            }
            for index in range(5)
        ]
        logs = {
            row["name"]: [
                {
                    "commit": f"{(index + 1) * 100 + offset:040x}",
                    "short_commit": f"{(index + 1) * 100 + offset:040x}"[:12],
                    "authored_at": f"2026-01-01T00:00:{offset:02d}Z",
                    "subject": "SECRET_SUBJECT",
                    "trailers": {},
                }
                for offset in range(6)
            ]
            for index, row in enumerate(branches)
        }
        objects = [
            {
                "object_id": f"rso-{index:016x}",
                "kind": "experiment_attempt" if index < 3 else "review",
                "state": "failed" if index == 0 else "completed",
                "created_at": f"2026-01-01T00:00:{index:02d}Z",
                "content_hash": f"sha256:{index:064x}",
                "payload": {"status": "failed" if index == 0 else "completed"},
                "relations": (
                    [{"type": "repairs", "target": "rso-0000000000000000"}]
                    if index == 3
                    else []
                ),
                "provenance": {},
            }
            for index in range(8)
        ]
        status = {
            "branch": "main",
            "head": "f" * 40,
            "worktree_clean": True,
            "checkpoint_policy": "milestone",
            "staged_paths": [],
            "eligible_changes": [],
        }
        with tempfile.TemporaryDirectory() as raw:
            with (
                mock.patch.object(
                    process_audit_module, "repository_status", return_value=status
                ),
                mock.patch.object(
                    process_audit_module,
                    "list_research_branches",
                    return_value=branches,
                ),
                mock.patch.object(
                    process_audit_module,
                    "research_log",
                    side_effect=lambda _root, *, ref, limit: logs[ref][:limit],
                ),
                mock.patch.object(
                    process_audit_module, "list_research_objects", return_value=objects
                ),
                mock.patch.object(
                    process_audit_module, "show_checkpoint", return_value={}
                ),
            ):
                report = build_process_summary(
                    Path(raw) / "workspace",
                    task_manifest_sha256="sha256:manifest",
                    max_branches=1,
                    max_commits=3,
                    max_artifacts=3,
                    max_decisions=1,
                )

        self.assertTrue(report["available"])
        self.assertEqual(len(report["branches"]), 1)
        self.assertEqual(report["branches"][0]["name"], "current")
        self.assertEqual(len(report["commits"]), 3)
        self.assertEqual(len(report["intermediate"]["artifacts"]), 3)
        self.assertEqual(report["branch_topology"]["source_branch_count"], 5)
        self.assertEqual(report["intermediate"]["source_object_count"], 8)
        self.assertTrue(report["limits"]["truncated"]["branches"])
        self.assertTrue(report["limits"]["truncated"]["commits"])
        self.assertTrue(report["limits"]["truncated"]["artifacts"])
        self.assertTrue(report["limits"]["truncated"]["decisions"])
        # Header counts are computed over the full validated object store even
        # when only three artifact rows are shown in the bounded window.
        self.assertEqual(report["intermediate"]["failed_attempts"], 1)
        self.assertEqual(report["intermediate"]["completed_attempts"], 2)
        self.assertEqual(report["intermediate"]["valid_object_count"], 8)
        self.assertEqual(report["intermediate"]["failed_or_blocked_count"], 1)
        self.assertEqual(report["intermediate"]["recovery_candidates"], 1)
        self.assertEqual(
            report["intermediate"]["statistics_scope"], "all_valid_objects"
        )
        self.assertTrue(report["intermediate"]["attempts_truncated"])
        self.assertFalse(
            report["branch_topology"]["fair_branch_comparison"]["eligible"]
        )

    def test_process_audit_keeps_decision_window_separate_from_artifact_window(
        self,
    ) -> None:
        objects = [
            {
                "object_id": f"rso-{index:016x}",
                "kind": "review" if index == 0 else "hypothesis",
                "state": "completed",
                "created_at": f"2026-01-01T00:00:{index:02d}Z",
                "content_hash": f"sha256:{index:064x}",
                "payload": {"decision": "hold"} if index == 0 else {},
                "relations": [],
                "provenance": {},
            }
            for index in range(6)
        ]
        status = {
            "branch": "main",
            "head": "f" * 40,
            "worktree_clean": True,
            "checkpoint_policy": "milestone",
            "staged_paths": [],
            "eligible_changes": [],
        }
        branches = [
            {
                "name": "main",
                "current": True,
                "commit": "1" * 40,
                "stage": "review",
                "status": "completed",
                "checkpoint_id": None,
            }
        ]
        with tempfile.TemporaryDirectory() as raw:
            with (
                mock.patch.object(
                    process_audit_module, "repository_status", return_value=status
                ),
                mock.patch.object(
                    process_audit_module,
                    "list_research_branches",
                    return_value=branches,
                ),
                mock.patch.object(
                    process_audit_module, "research_log", return_value=[]
                ),
                mock.patch.object(
                    process_audit_module, "list_research_objects", return_value=objects
                ),
                mock.patch.object(
                    process_audit_module, "show_checkpoint", return_value={}
                ),
            ):
                report = build_process_summary(
                    Path(raw) / "workspace",
                    max_artifacts=1,
                    max_decisions=1,
                )
        self.assertEqual(len(report["intermediate"]["artifacts"]), 1)
        self.assertEqual(len(report["intermediate"]["decision_events"]), 1)
        self.assertEqual(
            report["intermediate"]["decision_events"][0]["decision"], "hold"
        )


if __name__ == "__main__":
    unittest.main()
