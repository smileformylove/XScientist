"""Conformance tests for the ARA protocol."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import (
    PROTOCOL_VERSION,
    Kind,
    available_schemas,
    content_hash,
    hash_node_payload,
    load_schema,
    validate_ara,
    validate_manifest,
)
from ai_scientist.protocol.hashing import hash_matches
from ai_scientist.utils.ara_artifact import export_ara


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _minimal_project(tmp: Path, *, code: str = "print('ok')") -> Path:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260701_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(
        exp / "logs",
        "0-run",
        [
            {
                "id": "n1",
                "step": 0,
                "code": code,
                "_term_out": ["ok\n"],
                "metric": {"value": 0.5, "maximize": True, "name": "acc", "description": ""},
                "is_buggy": False,
                "parent_id": None,
                "children": [],
            }
        ],
    )
    return exp


class ConstantsTest(unittest.TestCase):
    def test_version_string_is_stable(self) -> None:
        self.assertEqual(PROTOCOL_VERSION, "ara.v1")

    def test_kinds_have_schemas(self) -> None:
        schemas = set(available_schemas())
        for kind in Kind:
            self.assertIn(kind.value, schemas, msg=f"missing schema for {kind}")


class HashingTest(unittest.TestCase):
    def test_hash_is_deterministic(self) -> None:
        a = hash_node_payload(code="x = 1\n", metric={"value": 0.5, "name": "acc"})
        b = hash_node_payload(code="x = 1\n", metric={"value": 0.5, "name": "acc"})
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("sha256:"))

    def test_hash_ignores_unstable_metric_fields(self) -> None:
        a = hash_node_payload(code="x = 1", metric={"value": 0.5, "name": "acc", "description": "run A"})
        b = hash_node_payload(code="x = 1", metric={"value": 0.5, "name": "acc", "description": "run B"})
        self.assertEqual(a, b, msg="description should not affect the hash")

    def test_hash_changes_on_code_edit(self) -> None:
        a = hash_node_payload(code="x = 1", metric={"value": 0.5})
        b = hash_node_payload(code="x = 2", metric={"value": 0.5})
        self.assertNotEqual(a, b)

    def test_hash_matches_algo_prefix(self) -> None:
        a = hash_node_payload(code="x", metric={"value": 1})
        self.assertTrue(hash_matches(a, a))
        self.assertFalse(hash_matches("md5:abc", a))

    def test_content_hash_is_stable_across_orderings(self) -> None:
        a = content_hash({"a": 1, "b": [2, 3]})
        b = content_hash({"b": [2, 3], "a": 1})
        self.assertEqual(a, b)


class SchemaTest(unittest.TestCase):
    def test_all_schemas_loadable(self) -> None:
        for kind in Kind:
            schema = load_schema(kind)
            self.assertIn("$id", schema)
            self.assertEqual(schema.get("type"), "object")


class ValidatorTest(unittest.TestCase):
    def test_minimal_manifest_valid(self) -> None:
        report = validate_manifest(
            {
                "schema_version": PROTOCOL_VERSION,
                "protocol_kind": "manifest",
                "created_at": "2026-07-01T00:00:00Z",
                "source_exp_dir": "/tmp/x",
                "idea": {"name": "abc"},
                "counts": {"nodes": 0},
            }
        )
        self.assertTrue(report.ok, msg=[e.__dict__ for e in report.errors])

    def test_manifest_missing_required_field(self) -> None:
        report = validate_manifest(
            {
                "schema_version": PROTOCOL_VERSION,
                "created_at": "2026-07-01T00:00:00Z",
                "source_exp_dir": "/tmp/x",
                # missing idea
                "counts": {"nodes": 0},
            }
        )
        self.assertFalse(report.ok)
        messages = " ".join(e.message for e in report.errors)
        self.assertIn("idea", messages)

    def test_manifest_wrong_kind_const(self) -> None:
        report = validate_manifest(
            {
                "schema_version": PROTOCOL_VERSION,
                "protocol_kind": "not_a_manifest",
                "created_at": "2026-07-01T00:00:00Z",
                "source_exp_dir": "/tmp/x",
                "idea": {"name": "abc"},
                "counts": {"nodes": 0},
            }
        )
        self.assertFalse(report.ok)


class ARARoundTripTest(unittest.TestCase):
    def test_freshly_exported_ara_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exp_dir = _minimal_project(tmp_path)
            result = export_ara(
                project_dir=tmp_path / "project",
                exp_dir=exp_dir,
                idea={"Name": "idea", "Title": "T"},
                timestamp="20260701",
            )
            report = validate_ara(result.root)
            self.assertTrue(
                report.ok,
                msg=json.dumps(report.to_dict(), indent=2),
            )
            # exploration graph must retain the buggy-included property.
            graph = json.loads((result.root / "exploration_graph.json").read_text())
            self.assertEqual(graph["protocol_kind"], "exploration_graph")
            self.assertEqual(graph["counts"]["nodes"], 1)
            # Each node has a content_hash.
            for node in graph["nodes"]:
                self.assertTrue(node["content_hash"].startswith("sha256:"))

    def test_provenance_survives_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exp_dir = _minimal_project(tmp_path)
            provenance = {
                "parent_ara_root": "/tmp/parent-ara",
                "parent_node_id": "n42",
                "parent_content_hash": "sha256:" + "a" * 64,
            }
            result = export_ara(
                project_dir=tmp_path / "project",
                exp_dir=exp_dir,
                idea={"Name": "child"},
                timestamp="20260701",
                provenance=provenance,
            )
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest.get("provenance"), provenance)
            self.assertTrue(validate_ara(result.root).ok)

    def test_missing_top_level_files_fail_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ara = Path(tmp) / "ara"
            ara.mkdir()
            report = validate_ara(ara)
            self.assertFalse(report.ok)
            messages = " ".join(e.message for e in report.errors)
            self.assertIn("required file missing", messages)


if __name__ == "__main__":
    unittest.main()
