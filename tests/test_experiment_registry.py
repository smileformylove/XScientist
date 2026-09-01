from __future__ import annotations

import json
import hashlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from ai_scientist.utils import experiment_registry
from ai_scientist.utils.experiment_registry import (
    append_experiment_record,
    build_experiment_record,
    check_experiment_registry_integrity,
    load_experiment_records,
    load_verified_experiment_records,
    normalize_terminal_experiment_status,
    record_terminal_experiment_failure,
    save_experiment_registry,
    summarize_experiment_registry,
)
from ai_scientist.utils.research_integrity import ResearchIntegrityError
from ai_scientist.utils.pipeline_contracts import load_pipeline_manifest
from ai_scientist.utils.decision_log import load_decision_log


class ExperimentRegistryTests(unittest.TestCase):
    def test_terminal_receipts_are_distinct_per_run_and_replay_idempotently(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = {
                "tasks": [
                    {
                        "task_id": "task_0",
                        "dataset": "demo",
                        "metric": "accuracy",
                        "baseline": "base",
                    }
                ]
            }
            first_receipt = {
                "status": "failed",
                "failure_error": {
                    "type": "RuntimeError",
                    "error_code": "experiment_failed",
                    "failure_ref": "receipt-one",
                    "message": "first failure",
                },
            }
            second_receipt = {
                "status": "failed",
                "failure_error": {
                    "type": "RuntimeError",
                    "error_code": "experiment_failed",
                    "failure_ref": "receipt-two",
                    "message": "second failure",
                },
            }

            first = record_terminal_experiment_failure(
                root,
                research_plan=plan,
                experiment_result=first_receipt,
                producer="test.terminal_outcome",
            )
            durable_paths = [
                root / "experiment_registry.jsonl",
                root / "experiment_registry.integrity.json",
                root / "experiment_registry.history.jsonl",
                root / "decision_log.jsonl",
                root / "pipeline_manifest.json",
                root / "stage_standards.json",
            ]
            before_replay = {path: path.read_bytes() for path in durable_paths}
            replay = record_terminal_experiment_failure(
                root,
                research_plan=plan,
                experiment_result=first_receipt,
                producer="test.terminal_outcome",
            )
            self.assertEqual(
                before_replay,
                {path: path.read_bytes() for path in durable_paths},
            )
            second = record_terminal_experiment_failure(
                root,
                research_plan=plan,
                experiment_result=second_receipt,
                producer="test.terminal_outcome",
            )

            rows = load_verified_experiment_records(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["record_id"] for row in rows}), 2)
            self.assertNotEqual(
                first["terminal_receipt_hash"], second["terminal_receipt_hash"]
            )
            self.assertFalse(first["replayed"])
            self.assertTrue(replay["replayed"])
            self.assertFalse(second["replayed"])
            self.assertEqual(len(load_decision_log(root)), 2)

    def test_concurrent_terminal_receipt_replay_records_one_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = {
                "tasks": [
                    {
                        "task_id": "task_0",
                        "dataset": "demo",
                        "metric": "accuracy",
                        "baseline": "base",
                    }
                ]
            }
            receipt = {
                "status": "failed",
                "failure_error": {
                    "failure_ref": "shared-receipt",
                    "message": "same terminal run",
                },
            }
            barrier = threading.Barrier(2)
            real_load = experiment_registry.load_experiment_records

            def synchronize_legacy_precheck(project_root):
                rows = real_load(project_root)
                barrier.wait(timeout=5)
                return rows

            with mock.patch.object(
                experiment_registry,
                "load_experiment_records",
                side_effect=synchronize_legacy_precheck,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(
                        executor.map(
                            lambda _: record_terminal_experiment_failure(
                                root,
                                research_plan=plan,
                                experiment_result=receipt,
                                producer="test.concurrent_terminal_outcome",
                            ),
                            range(2),
                        )
                    )

            self.assertEqual(
                sorted(result["replayed"] for result in results),
                [False, True],
            )
            self.assertEqual(len(load_verified_experiment_records(root)), 1)
            self.assertEqual(len(load_decision_log(root)), 1)

    def test_terminal_failure_is_persisted_before_writeup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project" / "02_experiments" / "run-1"
            checkpoint = root / "logs" / "checkpoint.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("{}\n", encoding="utf-8")
            result = record_terminal_experiment_failure(
                root,
                research_plan={
                    "workflow_mode": "program_driven",
                    "execution_policy": {"policy_name": "program_driven"},
                    "tasks": [
                        {
                            "task_id": "task_0",
                            "dataset": "demo",
                            "metric": "accuracy",
                            "baseline": "base",
                            "acceptance_checks": ["produce a metric"],
                        }
                    ],
                },
                experiment_result={
                    "status": "budget_exhausted",
                    "budget_error": {
                        "message": f"budget exhausted near {root}/private.log"
                    },
                    "resumable": True,
                    "checkpoint_path": str(checkpoint),
                },
                producer="test.terminal_outcome",
            )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["runtime_status"], "budget_exhausted")
            rows = load_verified_experiment_records(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[0]["budget_status"], "budget_exhausted")
            self.assertEqual(
                rows[0]["artifacts"]["checkpoint_path"],
                "logs/checkpoint.json",
            )
            self.assertFalse(rows[0]["entered_storyline"])
            decisions = load_decision_log(root)
            self.assertEqual(decisions[-1]["category"], "experiment_terminal_outcome")
            self.assertEqual(decisions[-1]["selected"], "failed")
            persisted = (root / "experiment_registry.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(str(root), persisted)

    def test_terminal_status_normalization_is_typed(self) -> None:
        self.assertEqual(normalize_terminal_experiment_status("timed_out"), "timed_out")
        self.assertEqual(
            normalize_terminal_experiment_status("interrupted"), "cancelled"
        )
        self.assertEqual(normalize_terminal_experiment_status("locked"), "failed")

    @staticmethod
    def _legacy_v1_history_row(integrity: dict) -> dict:
        core = {
            "record_count": integrity["row_count"],
            "records_hash": integrity["records_hash"],
            "raw_hash": integrity["raw_hash"],
            "chain_tip": integrity["chain_tip"],
        }
        return {
            "version": 1,
            **core,
            "audit_hash": experiment_registry.canonical_hash(core),
        }

    def test_missing_integrity_evidence_blocks_registry_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            output_path = project_root / "experiment_registry.jsonl"
            output_path.write_text('{"status":"previous"}\n', encoding="utf-8")

            with (
                mock.patch.object(
                    experiment_registry, "update_pipeline_artifact"
                ) as update,
                self.assertRaisesRegex(
                    ResearchIntegrityError, "tampered|must all exist"
                ),
            ):
                save_experiment_registry(
                    project_root,
                    [{"record_id": "new", "status": "new"}],
                )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{"status":"previous"}\n',
            )
            update.assert_not_called()

    def test_append_refuses_to_bless_tampered_registry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = build_experiment_record(
                task_id="first",
                dataset="demo",
                metric="accuracy",
                baseline_ref="base",
                status="completed",
            )
            second = build_experiment_record(
                task_id="second",
                dataset="demo",
                metric="accuracy",
                baseline_ref="base",
                status="failed",
            )
            save_experiment_registry(root, [first])
            registry = root / "experiment_registry.jsonl"
            registry.write_bytes(
                registry.read_bytes().replace(b'"completed"', b'"failed"')
            )
            before = {
                path: path.read_bytes()
                for path in (
                    registry,
                    root / "experiment_registry.integrity.json",
                    root / "experiment_registry.history.jsonl",
                )
            }

            with self.assertRaises(ResearchIntegrityError):
                append_experiment_record(root, second)

            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_concurrent_appends_preserve_every_row_and_history_chain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            def record(index: int) -> dict:
                return build_experiment_record(
                    task_id=f"task-{index}",
                    record_id=f"record-{index}",
                    dataset="demo",
                    metric="accuracy",
                    baseline_ref="base",
                    status="completed",
                )

            save_experiment_registry(root, [record(0)])
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(
                    executor.map(
                        lambda index: append_experiment_record(root, record(index)),
                        range(1, 9),
                    )
                )

            loaded = load_experiment_records(root)
            report = check_experiment_registry_integrity(root)
            self.assertEqual(
                {row["record_id"] for row in loaded},
                {f"record-{index}" for index in range(9)},
            )
            self.assertTrue(report["ok"], report["errors"])
            history = [
                json.loads(line)
                for line in (root / "experiment_registry.history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["sequence"] for row in history], list(range(1, 10)))
            self.assertEqual(history[0]["previous_audit_hash"], "GENESIS")
            self.assertEqual(
                [row["previous_audit_hash"] for row in history[1:]],
                [row["audit_hash"] for row in history[:-1]],
            )

    def test_append_and_save_reject_stale_snapshot_after_coherent_commit(
        self,
    ) -> None:
        for operation in ("append", "save"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "target"
                competing_root = Path(td) / "competing"

                def record(record_id: str) -> dict:
                    return build_experiment_record(
                        task_id=record_id,
                        record_id=record_id,
                        dataset="demo",
                        metric="accuracy",
                        baseline_ref="base",
                        status="completed",
                    )

                first = record("A")
                stale_addition = record("B")
                concurrent_addition = record("C")
                save_experiment_registry(root, [first])

                # Prepare a fully coherent A+C triple. The injected replacement
                # deliberately ignores the advisory transaction lock and lands
                # after the public writer observed A but before its low-level
                # commit. A stale A+B writer must fail closed, not adopt A+C as
                # its baseline and erase C.
                save_experiment_registry(competing_root, [first])
                save_experiment_registry(competing_root, [concurrent_addition])
                coherent_by_name = {
                    path.name: path.read_bytes()
                    for path in experiment_registry._registry_paths(competing_root)
                }
                original_write = experiment_registry._write_registry_transaction
                injected = False

                def inject_coherent_commit(project_root, **kwargs):
                    nonlocal injected
                    if not injected:
                        injected = True
                        for path in experiment_registry._registry_paths(root):
                            path.write_bytes(coherent_by_name[path.name])
                    return original_write(project_root, **kwargs)

                with (
                    mock.patch.object(
                        experiment_registry,
                        "_write_registry_transaction",
                        side_effect=inject_coherent_commit,
                    ),
                    self.assertRaisesRegex(ResearchIntegrityError, "concurrently"),
                ):
                    if operation == "append":
                        append_experiment_record(root, stale_addition)
                    else:
                        save_experiment_registry(root, [stale_addition])

                self.assertTrue(injected)
                self.assertEqual(
                    [row["record_id"] for row in load_experiment_records(root)],
                    ["A", "C"],
                )
                report = check_experiment_registry_integrity(root)
                self.assertTrue(report["ok"], report["errors"])
                history = [
                    json.loads(line)
                    for line in (root / "experiment_registry.history.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual([row["sequence"] for row in history], [1, 2])

    def test_save_is_idempotent_append_merge_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = build_experiment_record(
                task_id="first",
                record_id="first",
                dataset="demo",
                metric="accuracy",
                baseline_ref="base",
                status="failed",
            )
            second = build_experiment_record(
                task_id="second",
                record_id="second",
                dataset="demo",
                metric="accuracy",
                baseline_ref="base",
                status="completed",
            )
            save_experiment_registry(root, [first])
            history_path = root / "experiment_registry.history.jsonl"
            first_history = history_path.read_bytes()

            save_experiment_registry(root, [first])
            self.assertEqual(history_path.read_bytes(), first_history)
            rebuilt_first = build_experiment_record(
                task_id="first",
                record_id="first",
                dataset="demo",
                metric="accuracy",
                baseline_ref="base",
                status="failed",
            )
            save_experiment_registry(root, [rebuilt_first])
            self.assertEqual(history_path.read_bytes(), first_history)

            save_experiment_registry(root, [second])
            self.assertEqual(load_experiment_records(root), [first, second])
            self.assertTrue(check_experiment_registry_integrity(root)["ok"])

            changed_first = {**first, "status": "completed"}
            with self.assertRaisesRegex(ResearchIntegrityError, "immutable"):
                save_experiment_registry(root, [changed_first])

    def test_legacy_history_requires_full_prefix_anchor_before_authority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = build_experiment_record(
                task_id="first",
                record_id="A",
                dataset="demo",
                metric="accuracy",
                baseline_ref="base",
                status="completed",
            )
            second = build_experiment_record(
                task_id="second",
                record_id="B",
                dataset="demo",
                metric="accuracy",
                baseline_ref="base",
                status="completed",
            )
            save_experiment_registry(root, [first])
            first_integrity = json.loads(
                (root / "experiment_registry.integrity.json").read_text(
                    encoding="utf-8"
                )
            )
            save_experiment_registry(root, [second])
            second_integrity = json.loads(
                (root / "experiment_registry.integrity.json").read_text(
                    encoding="utf-8"
                )
            )
            legacy_rows = [
                self._legacy_v1_history_row(first_integrity),
                self._legacy_v1_history_row(second_integrity),
            ]
            legacy_raw = "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in legacy_rows
            ).encode("utf-8")
            history_path = root / "experiment_registry.history.jsonl"
            history_path.write_bytes(legacy_raw)

            # Ordinary inspection remains backward compatible, but an
            # unanchored v1 chain must never grant execution/publication trust.
            self.assertEqual(load_experiment_records(root), [first, second])
            report = check_experiment_registry_integrity(root)
            self.assertFalse(report["ok"])
            self.assertIn("legacy_history_unanchored", report["errors"])
            with self.assertRaisesRegex(
                ResearchIntegrityError, "legacy_history_unanchored"
            ):
                load_verified_experiment_records(root)

            # An idempotent save performs a one-time, exact-byte migration even
            # though no experiment record changed.
            save_experiment_registry(root, [first, second])
            migrated_raw = history_path.read_bytes()
            migrated_rows = [
                json.loads(line) for line in migrated_raw.decode("utf-8").splitlines()
            ]
            anchor = migrated_rows[-1]
            self.assertEqual(anchor["version"], 3)
            self.assertEqual(anchor["legacy_history_row_count"], 2)
            self.assertEqual(
                anchor["legacy_history_tip"], legacy_rows[-1]["audit_hash"]
            )
            self.assertEqual(
                anchor["legacy_history_raw_hash"],
                "sha256:" + hashlib.sha256(legacy_raw).hexdigest(),
            )
            self.assertTrue(check_experiment_registry_integrity(root)["ok"])
            self.assertEqual(load_verified_experiment_records(root), [first, second])

            # Removing any anchored legacy prefix changes both the exact prefix
            # digest and row position even though the surviving v1 tip still
            # describes the current A+B registry snapshot.
            history_path.write_bytes(
                b"".join(migrated_raw.splitlines(keepends=True)[1:])
            )
            truncated = check_experiment_registry_integrity(root)
            self.assertFalse(truncated["ok"])
            self.assertIn(
                "registry_history_legacy_raw_hash_mismatch",
                truncated["errors"],
            )

    def test_save_and_summarize_registry_should_update_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            rows = [
                build_experiment_record(
                    task_id="task_0",
                    dataset="demo-ds",
                    metric="accuracy",
                    baseline_ref="baseline_a",
                    status="completed",
                    result_summary={"metric_mean": 0.82},
                    entered_storyline=True,
                    workflow_mode="program_driven",
                    policy_name="program_driven",
                    acceptance_checks=["Keep budget discipline."],
                ),
                build_experiment_record(
                    task_id="task_1",
                    dataset="demo-ds",
                    metric="f1",
                    baseline_ref="baseline_b",
                    status="failed",
                    error_type="timeout",
                    error_message="budget exhausted",
                    workflow_mode="program_driven",
                    policy_name="program_driven",
                    acceptance_checks=["Record budget exhaustion explicitly."],
                ),
            ]
            save_experiment_registry(project_root, rows)

            loaded = load_experiment_records(project_root)
            summary = summarize_experiment_registry(project_root)
            manifest = load_pipeline_manifest(project_root)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(summary["by_status"]["completed"], 1)
            self.assertEqual(summary["by_status"]["failed"], 1)
            self.assertEqual(summary["by_budget_status"]["within_budget"], 1)
            self.assertEqual(summary["by_budget_status"]["budget_exhausted"], 1)
            self.assertEqual(summary["policy_names"]["program_driven"], 2)
            self.assertEqual(summary["storyline_count"], 1)
            self.assertEqual(loaded[0]["workflow_mode"], "program_driven")
            self.assertTrue(loaded[0]["acceptance_checks"])
            self.assertEqual(
                manifest["artifacts"]["experiment_registry"]["status"],
                "ready",
            )


if __name__ == "__main__":
    unittest.main()
