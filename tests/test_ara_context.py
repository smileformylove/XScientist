from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ai_scientist.apps.ara import cmd_catalog, cmd_context
from ai_scientist.protocol import ObjectStore
from ai_scientist.treesearch.journal import Node
from ai_scientist.utils.ara_catalog import (
    CATALOG_RELPATH,
    catalog_status,
    rebuild_semantic_catalog,
)
from ai_scientist.utils.ara_context import (
    ARAContextError,
    MIN_CONTEXT_BUDGET_TOKENS,
    compile_context_pack,
    compile_live_continue_context,
    compile_live_write_context,
    persist_active_context_pack,
    persist_context_pack,
    record_context_consumption,
    render_context_pack_for_prompt,
    validate_context_pack,
)
from ai_scientist.utils.ara_events import bootstrap_event_ledger, iter_events
from ai_scientist.utils.claim_registry import write_claims_into_ara
from ai_scientist.utils.context_receipts import (
    ContextReceiptError,
    validate_context_receipt,
)
from ai_scientist.utils.review_execution import execute_review_pass


class ARAContextTests(unittest.TestCase):
    def _build_ara(self, root: Path) -> Path:
        root.mkdir(parents=True)
        manifest = {
            "schema_version": "ara.v1",
            "protocol_kind": "manifest",
            "created_at": "2026-01-01T00:00:00+00:00",
            "source_exp_dir": str(root / "source"),
            "project_dir": str(root.parent),
            "idea": {"name": "context-demo", "title": "Context Demo", "raw": {}},
            "counts": {"nodes": 3, "edges": 2, "claims": 2},
            "references": {},
            "missing": [],
            "capabilities": {"context": "complete", "reproduce": "partial"},
            "omissions": [],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        graph = {
            "schema_version": "ara.v1",
            "protocol_kind": "exploration_graph",
            "topology_encoding": "edges",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "nodes": [
                {
                    "id": "base",
                    "content_hash": "sha256:" + "1" * 64,
                    "stage": "baseline",
                    "step": 0,
                    "is_buggy": False,
                    "metric": {"name": "accuracy", "value": 0.70, "maximize": True},
                    "plan_excerpt": "establish baseline",
                    "artifacts_dir": "nodes/base",
                },
                {
                    "id": "good",
                    "content_hash": "sha256:" + "2" * 64,
                    "stage": "improve",
                    "step": 1,
                    "is_buggy": False,
                    "metric": {"name": "accuracy", "value": 0.80, "maximize": True},
                    "plan_excerpt": "add calibrated component",
                    "artifacts_dir": "nodes/good",
                },
                {
                    "id": "bad",
                    "content_hash": "sha256:" + "3" * 64,
                    "stage": "improve",
                    "step": 2,
                    "is_buggy": True,
                    "metric": {"name": "accuracy", "value": 0.10, "maximize": True},
                    "plan_excerpt": "unstable high learning rate",
                    "artifacts_dir": "nodes/bad",
                },
            ],
            "edges": [
                {"parent": "base", "child": "good", "stage": "improve"},
                {"parent": "base", "child": "bad", "stage": "improve"},
            ],
            "counts": {"nodes": 3, "edges": 2, "buggy": 1},
        }
        (root / "exploration_graph.json").write_text(
            json.dumps(graph), encoding="utf-8"
        )
        for node_id in ("base", "good", "bad"):
            node_dir = root / "nodes" / node_id
            node_dir.mkdir(parents=True)
            (node_dir / "code.py").write_text(f"print('{node_id}')\n", encoding="utf-8")
            (node_dir / "run.sh").write_text(
                "#!/bin/sh\npython code.py\n", encoding="utf-8"
            )
            (node_dir / "env.json").write_text(
                json.dumps({"expected_cwd": str(node_dir), "python_version": "3.13"}),
                encoding="utf-8",
            )
        claims = root / "claims"
        claims.mkdir()
        (claims / "supported.json").write_text(
            json.dumps(
                {
                    "claim_id": "supported",
                    "claim_hash": "sha256:" + "4" * 64,
                    "node_id": "good",
                    "tex_file": "paper.tex",
                    "line": 10,
                    "context": "The calibrated component improves accuracy.",
                    "resolved": True,
                    "evidence_refs": ["sha256:" + "2" * 64],
                    "source": {"selector": {"type": "line", "value": 10}},
                }
            ),
            encoding="utf-8",
        )
        (claims / "open.json").write_text(
            json.dumps(
                {
                    "claim_id": "open",
                    "claim_hash": "sha256:" + "5" * 64,
                    "node_id": "missing",
                    "tex_file": "paper.tex",
                    "line": 20,
                    "context": "The method transfers to every domain.",
                    "resolved": False,
                    "evidence_refs": [],
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_event_bootstrap_is_idempotent_and_semantic(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            first = bootstrap_event_ledger(ara)
            second = bootstrap_event_ledger(ara)
            events = list(iter_events(ara))
            self.assertEqual(first["created"], 5)
            self.assertEqual(second["created"], 0)
            self.assertEqual(len(events), 5)
            self.assertEqual(
                {event["event_type"] for event in events},
                {"node_completed", "claim_bound", "claim_unresolved"},
            )
            bad = next(event for event in events if event["subject"].get("id") == "bad")
            self.assertEqual(bad["attributes"]["decision"], "discard")

    def test_catalog_is_rebuildable_and_reports_freshness(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            report = rebuild_semantic_catalog(ara)
            self.assertEqual(report["counts"]["nodes"], 3)
            self.assertEqual(report["counts"]["claims"], 2)
            self.assertTrue(catalog_status(ara)["fresh"])
            connection = sqlite3.connect(ara / CATALOG_RELPATH)
            try:
                derived = connection.execute(
                    "SELECT target_id FROM relations WHERE source_id='good' AND relation='derived_from'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(derived[0], "base")
            claim = json.loads((ara / "claims" / "supported.json").read_text())
            claim["context"] = "changed assertion"
            (ara / "claims" / "supported.json").write_text(
                json.dumps(claim), encoding="utf-8"
            )
            self.assertFalse(catalog_status(ara)["fresh"])

    def test_continue_context_prioritizes_target_and_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            pack = compile_context_pack(
                ara,
                intent="continue",
                node_id="good",
                budget_tokens=1500,
            )
            self.assertEqual(pack["consumer"], "experiment_agent")
            self.assertEqual(pack["must_read"][0]["node_id"], "good")
            self.assertIn("base", {item["node_id"] for item in pack["must_read"]})
            self.assertIn("bad", {item["node_id"] for item in pack["failed_attempts"]})
            self.assertTrue(pack["source_closure_hash"].startswith("sha256:"))
            self.assertTrue(pack["memory_snapshot_hash"].startswith("sha256:"))
            self.assertTrue(pack["budget"]["hard_closure_preserved"])
            self.assertIs(validate_context_pack(pack), pack)

    def test_long_failure_history_retains_recent_failure_and_decisive_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            graph_path = ara / "exploration_graph.json"
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            for index in range(80):
                graph["nodes"].append(
                    {
                        "id": f"failure-{index:02d}",
                        "content_hash": "sha256:" + f"{index + 100:064x}"[-64:],
                        "stage": "improve",
                        "step": index + 3,
                        "is_buggy": True,
                        "metric": {
                            "name": "accuracy",
                            "value": 0.1,
                            "maximize": True,
                        },
                        "plan_excerpt": (
                            "latest critical calibration failure"
                            if index == 79
                            else f"historical failed configuration {index}"
                        ),
                        "artifacts_dir": f"nodes/failure-{index:02d}",
                    }
                )
            graph["counts"]["nodes"] = len(graph["nodes"])
            graph["counts"]["buggy"] = 81
            graph_path.write_text(json.dumps(graph), encoding="utf-8")

            pack = compile_context_pack(
                ara,
                intent="continue",
                node_id="good",
                budget_tokens=1000,
            )
            rendered = render_context_pack_for_prompt(pack)

            self.assertTrue(pack["complete"])
            self.assertTrue(pack["budget"]["decision_usable"])
            self.assertLessEqual(pack["budget"]["prompt_estimated_tokens"], 1000)
            self.assertIn("latest critical calibration failure", rendered)
            self.assertIn("accuracy", rendered)
            self.assertGreater(pack["omitted"]["failed_attempts"], 0)
            self.assertIs(validate_context_pack(pack), pack)

    def test_context_budget_has_a_safe_minimum_and_working_memory_is_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            with self.assertRaisesRegex(
                ARAContextError,
                str(MIN_CONTEXT_BUDGET_TOKENS),
            ):
                compile_context_pack(
                    ara,
                    intent="continue",
                    node_id="good",
                    budget_tokens=MIN_CONTEXT_BUDGET_TOKENS - 1,
                )

            pack = compile_context_pack(
                ara,
                intent="continue",
                node_id="good",
                budget_tokens=1000,
            )
            pack["working_memory"]["visible_semantic_lanes"] = []
            with self.assertRaisesRegex(
                ARAContextError,
                "working memory hash mismatch",
            ):
                validate_context_pack(pack)

    def test_incomplete_pack_cannot_be_rendered_as_agent_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            pack = compile_context_pack(
                ara,
                intent="audit",
                budget_tokens=MIN_CONTEXT_BUDGET_TOKENS,
            )
            if pack["complete"]:
                pack["complete"] = False
                pack["blockers"] = ["forced test blocker"]
                identity = {
                    key: value
                    for key, value in pack.items()
                    if key not in {"generated_at", "pack_hash", "persisted_ref"}
                }
                from ai_scientist.protocol import content_hash

                pack["pack_hash"] = content_hash(identity)

            with self.assertRaisesRegex(ARAContextError, "not decision-usable"):
                render_context_pack_for_prompt(pack)
            self.assertIn(
                "ARA task context",
                render_context_pack_for_prompt(pack, allow_incomplete=True),
            )

    def test_write_audit_and_reproduce_are_distinct_consumption_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            write = compile_context_pack(ara, intent="write", claim_id="supported")
            self.assertEqual(len(write["decisive_evidence"]), 1)
            self.assertEqual(write["decisive_evidence"][0]["claim_id"], "supported")

            audit = compile_context_pack(ara, intent="audit")
            self.assertIn(
                "open", {item["claim_id"] for item in audit["open_questions"]}
            )
            self.assertIn("bad", {item["node_id"] for item in audit["failed_attempts"]})

            decide = compile_context_pack(
                ara,
                intent="decide",
                node_id="good",
                decision={"action": "choose next falsification"},
            )
            self.assertTrue(decide["complete"])
            self.assertEqual(decide["decision"]["action"], "choose next falsification")
            self.assertIn(
                "good",
                {
                    item.get("node_id")
                    for item in decide["decisive_evidence"]
                    if isinstance(item, dict)
                },
            )

            reproduce = compile_context_pack(ara, intent="reproduce", node_id="good")
            self.assertTrue(reproduce["complete"])
            self.assertEqual(reproduce["execution"]["run_hook"], "nodes/good/run.sh")
            self.assertEqual(
                reproduce["verification_rules"][0]["type"], "metric_tolerance"
            )

    def test_persisted_pack_is_addressable_and_receipted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            pack = compile_context_pack(ara, intent="reproduce", node_id="good")
            ref = persist_context_pack(ara, pack, consumer="reproduce_agent")
            self.assertEqual(
                ObjectStore(ara).get_json(ref)["pack_hash"], pack["pack_hash"]
            )
            returned = {**pack, "persisted_ref": ref}
            self.assertEqual(validate_context_pack(returned), returned)
            receipts = (ara / "context" / "receipts.jsonl").read_text(encoding="utf-8")
            self.assertIn(ref, receipts)
            self.assertIn("receipt_hash", receipts)
            tampered = dict(pack)
            tampered["memory_refs"] = ["sha256:" + "f" * 64]
            with self.assertRaises(ARAContextError):
                validate_context_pack(tampered)

    def test_tampered_consumption_receipt_is_not_admitted_as_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            pack = compile_context_pack(ara, intent="reproduce", node_id="good")
            ref = persist_context_pack(ara, pack, consumer="reproduce_agent")
            record_context_consumption(
                ara,
                pack_ref=ref,
                consumer="reproduce_agent",
                output_type="verification",
                output_id="verify-1",
            )
            receipts_path = ara / "context" / "receipts.jsonl"
            receipts = [
                json.loads(line)
                for line in receipts_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(receipts), 2)
            for receipt in receipts:
                self.assertEqual(validate_context_receipt(receipt), receipt)
            receipts[1]["output"]["id"] = "tampered"
            with self.assertRaises(ContextReceiptError):
                validate_context_receipt(receipts[1])
            receipts_path.write_text(
                "\n".join(json.dumps(item) for item in receipts) + "\n",
                encoding="utf-8",
            )

            report = bootstrap_event_ledger(ara)
            context_events = [
                event
                for event in iter_events(ara)
                if event["event_type"].startswith("context_")
            ]
            self.assertEqual(report["invalid_context_receipts"], 1)
            self.assertEqual(
                {event["event_type"] for event in context_events},
                {"context_compiled"},
            )

    def test_live_pack_is_injected_via_active_ara_and_node_serializes_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = Path(tmp) / "live"
            ara.mkdir()
            pack = compile_live_continue_context(
                [
                    {
                        "id": "ok",
                        "plan": "baseline",
                        "is_buggy": False,
                        "metric": {"value": 1.0},
                    },
                    {
                        "id": "fail",
                        "plan": "bad retry",
                        "is_buggy": True,
                        "metric": {"value": 0.0},
                    },
                ],
                target_node_id="ok",
                stage="improve",
            )
            with patch.dict(os.environ, {"AI_SCIENTIST_ARA_ACTIVE_ROOT": str(ara)}):
                ref = persist_active_context_pack(pack, consumer="experiment_agent")
            self.assertIsNotNone(ref)
            node = Node(plan="next", code="pass", context_pack_refs=[ref])
            restored = Node.from_dict(node.to_dict())
            self.assertEqual(restored.context_pack_refs, [ref])

    def test_writer_pack_extracts_evidence_and_is_bound_to_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            pack = compile_live_write_context(
                {
                    "RESEARCH_SUMMARY": {
                        "best_node_id": "good",
                        "validation_accuracy": 0.8,
                        "verbose_internal_trace": "not prompt evidence",
                    }
                },
                {"claim_bindings": ["good"]},
            )
            self.assertIn(
                "RESEARCH_SUMMARY.validation_accuracy",
                {item["path"] for item in pack["decisive_evidence"]},
            )
            self.assertNotIn("verbose_internal_trace", json.dumps(pack))
            ref = persist_context_pack(ara, pack, consumer="writing_agent")
            tex = Path(tmp) / "paper.tex"
            tex.write_text("Result improved. \\claimref{good}\n", encoding="utf-8")
            write_claims_into_ara(ara_dir=ara, tex_files=[tex])
            claim_files = [
                path
                for path in (ara / "claims").glob("*.json")
                if path.name
                not in {"_index.json", "coverage.json", "supported.json", "open.json"}
            ]
            payload = json.loads(claim_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["context_pack_refs"], [ref])

    def test_review_agent_receives_audit_context_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            active = Path(tmp) / "active-ara"
            active.mkdir()
            pdf = Path(tmp) / "paper.pdf"
            pdf.write_bytes(b"pdf")
            captured = {}

            def review_fn(*args, **kwargs):
                captured["instruction"] = kwargs["review_instruction_form"]
                return {"ok": True}

            with patch.dict(os.environ, {"AI_SCIENTIST_ARA_ACTIVE_ROOT": str(active)}):
                result = execute_review_pass(
                    paper_dir=tmp,
                    model_review="model",
                    review_plan={
                        "review_instruction": "Review carefully.",
                        "review_reflections": 1,
                        "review_fewshot": 0,
                        "review_ensemble": 1,
                        "review_temperature": 0.2,
                    },
                    create_client_fn=lambda model: (object(), model),
                    load_paper_fn=lambda path: "paper",
                    perform_review_fn=review_fn,
                    perform_imgs_cap_ref_review_fn=lambda *args: {},
                    pdf_path_resolver=lambda path: str(pdf),
                    evidence_refs=["sha256:" + "9" * 64],
                )
            self.assertTrue(result["found"])
            self.assertIn("ARA task context", captured["instruction"])
            self.assertTrue((active / "context" / "receipts.jsonl").is_file())

    def test_cli_catalog_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            ara = self._build_ara(Path(tmp) / "ara")
            self.assertEqual(cmd_catalog(Namespace(ara=str(ara), rebuild=True)), 0)
            args = Namespace(
                ara=str(ara),
                intent="continue",
                node="good",
                claim=None,
                budget=2000,
                receipt=True,
                json=True,
                allow_incomplete=False,
            )
            self.assertEqual(cmd_context(args), 0)
            self.assertTrue((ara / "context" / "receipts.jsonl").is_file())

            decide_args = Namespace(
                ara=str(ara),
                intent="decide",
                node="good",
                claim=None,
                decision_json='{"action":"choose next falsification"}',
                budget=2000,
                receipt=False,
                json=True,
                allow_incomplete=False,
            )
            self.assertEqual(cmd_context(decide_args), 0)


if __name__ == "__main__":
    unittest.main()
