from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol.schemas import load_schema
from xscientist.benchmark import benchmark_autoresearch_pilot
from xscientist.evidence_index import build_evidence_index


class EvidenceIndexTests(unittest.TestCase):
    def test_index_reports_vcs_and_ara_without_payload_or_path_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "workspace-with-sensitive-name"
            (root / ".xscientist" / "objects" / "hypothesis").mkdir(parents=True)
            (
                root / ".xscientist" / "objects" / "hypothesis" / "secret.json"
            ).write_text("SECRET_TYPED_PAYLOAD", encoding="utf-8")
            (root / "ara" / "run-1" / "claims").mkdir(parents=True)
            (root / "ara" / "run-1" / "claims" / "secret.json").write_text(
                "SECRET_ARA_PAYLOAD", encoding="utf-8"
            )
            (root / "ara" / "run-1" / "manifest.json").write_text(
                "SECRET_MANIFEST", encoding="utf-8"
            )
            (root / ".ara-store" / "objects" / "sha256").mkdir(parents=True)
            (root / ".ara-store" / "objects" / "sha256" / "payload").write_bytes(
                b"CAS_SECRET"
            )

            before = sorted(
                path.relative_to(root).as_posix() for path in root.rglob("*")
            )
            report = build_evidence_index(root)
            after = sorted(
                path.relative_to(root).as_posix() for path in root.rglob("*")
            )

        self.assertEqual(before, after)
        self.assertTrue(report["categories"]["research_vcs"]["present"])
        self.assertTrue(report["categories"]["ara"]["present"])
        self.assertEqual(report["ara_contract"]["manifest_count"], 1)
        self.assertEqual(
            report["ara_contract"]["lock_state"], "lock_missing_or_incomplete"
        )
        rendered = json.dumps(report)
        for secret in (
            "SECRET_TYPED_PAYLOAD",
            "SECRET_ARA_PAYLOAD",
            "CAS_SECRET",
            "SECRET_MANIFEST",
            "secret.json",
            str(root),
        ):
            self.assertNotIn(secret, rendered)
        validate(report, load_schema("evidence_index"))

    def test_index_is_bounded_and_marks_partial_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "ara").mkdir()
            (root / "ara" / "large.json").write_bytes(b"x" * 100)
            report = build_evidence_index(root, max_files=1, max_bytes=8)

        category = report["categories"]["ara"]
        self.assertEqual(category["file_count"], 1)
        self.assertEqual(category["byte_count"], 8)
        self.assertEqual(category["digest_scope"], "bounded_prefix")
        self.assertTrue(category["truncated"])
        self.assertTrue(report["truncated"])
        self.assertLessEqual(
            category["byte_count"] + report["ara_contract"]["bytes_read"], 8
        )
        self.assertIn("ara_contract", report)
        validate(report, load_schema("evidence_index"))

    def test_autoresearch_workspace_report_includes_index(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            tasks = parent / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "domain": "physics",
                        "premise": "premise",
                        "tension": "tension",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            workspace = parent / "workspace"
            (workspace / "ara").mkdir(parents=True)
            (workspace / "ara" / "manifest.json").write_text("{}", encoding="utf-8")
            report = benchmark_autoresearch_pilot(tasks, workspace=workspace, limit=1)

        index = report["workspace"]["evidence_index"]
        self.assertEqual(index["schema"], "xscientist.evidence-index.v1")
        self.assertTrue(index["categories"]["ara"]["present"])
        self.assertEqual(index["ara_contract"]["manifest_count"], 1)
        validate(index, load_schema("evidence_index"))

    def test_index_does_not_follow_directory_symlinks_or_accept_bool_limits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "workspace"
            root.mkdir()
            external = Path(raw) / "external"
            external.mkdir()
            (external / "secret.json").write_text("EXTERNAL_SECRET", encoding="utf-8")
            (root / "ara").symlink_to(external, target_is_directory=True)
            report = build_evidence_index(root, max_files=1, max_bytes=8)
            self.assertFalse(report["categories"]["ara"]["present"])
            self.assertNotIn("EXTERNAL_SECRET", json.dumps(report))
            with self.assertRaises(ValueError):
                build_evidence_index(root, max_files=True)
            with self.assertRaises(ValueError):
                build_evidence_index(root, max_bytes=1.5)

    def test_index_marks_parent_symlink_boundary_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "workspace"
            external = Path(raw) / "external"
            (external / "objects" / "hypothesis").mkdir(parents=True)
            (external / "objects" / "hypothesis" / "secret.json").write_text(
                "EXTERNAL_SECRET", encoding="utf-8"
            )
            root.mkdir()
            (root / ".xscientist").symlink_to(external, target_is_directory=True)
            report = build_evidence_index(root)
        category = report["categories"]["research_vcs"]
        self.assertTrue(category["truncated"])
        self.assertFalse(category["source_count_complete"])
        self.assertEqual(category["digest_scope"], "bounded_prefix")
        self.assertNotIn("EXTERNAL_SECRET", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
