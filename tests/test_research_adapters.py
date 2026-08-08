from __future__ import annotations

import json
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from contextlib import redirect_stdout

from jsonschema import validate

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema
from xscientist.research_adapters import (
    FilesystemResearchAdapter,
    ResearchAdapterDescriptor,
    available_research_adapters,
    doctor_research_adapter,
    sync_research_repository,
    validate_research_adapter,
)
from xscientist.research_git import ResearchGitError
from xscientist.research_journey import start_guided_research
from xscientist.research_cli import main as research_main
from xscientist import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required for adapter tests")
class ResearchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "study"
        start_guided_research(
            self.repo,
            question="Does X affect Y?",
            hypothesis="X affects Y.",
            falsifier="Y is unchanged.",
            git_user_name="Adapter Test",
            git_user_email="adapter@example.invalid",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_filesystem_adapter_publishes_hash_bound_interop_package(self) -> None:
        destination = self.root / "published"
        receipt = sync_research_repository(
            self.repo,
            adapter_name="filesystem",
            destination=str(destination),
            formats=["ro-crate", "prov-json"],
        )

        validate(receipt, load_schema("research_adapter_receipt"))
        base = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        self.assertEqual(receipt["receipt_hash"], canonical_content_hash(base))
        self.assertEqual(receipt["result"]["destination_name"], "published")
        self.assertFalse(Path(receipt["result"]["destination_name"]).is_absolute())
        manifest = json.loads((destination / "xscientist-export.json").read_text())
        self.assertFalse(manifest["payloads_included"])
        self.assertTrue((destination / "ro-crate-metadata.json").is_file())
        with self.assertRaisesRegex(ResearchGitError, "already exists"):
            sync_research_repository(
                self.repo,
                adapter_name="filesystem",
                destination=str(destination),
                formats=["ro-crate"],
            )

    def test_discovery_does_not_import_unselected_third_party_adapter(self) -> None:
        entry = mock.Mock()
        entry.name = "remote-platform"
        entry.dist = mock.Mock(name="example-adapter", version="2.0")
        with mock.patch(
            "xscientist.research_adapters._entry_points",
            return_value={"remote-platform": entry},
        ):
            rows = available_research_adapters()

        self.assertIn("remote-platform", {row["name"] for row in rows})
        entry.load.assert_not_called()

    def test_contract_validation_and_doctor_are_versioned(self) -> None:
        adapter = FilesystemResearchAdapter()
        validated = validate_research_adapter(adapter, expected_name="filesystem")
        self.assertIs(validated, adapter)
        doctor = doctor_research_adapter("filesystem")
        self.assertTrue(doctor["ok"])
        self.assertEqual(doctor["adapter"]["api_version"], "1.0")

        invalid = mock.Mock()
        invalid.descriptor = ResearchAdapterDescriptor(
            name="wrong",
            version="1",
            description="",
            capabilities=("publish",),
            destination_kinds=("uri",),
        )
        with self.assertRaisesRegex(ResearchGitError, "name does not match"):
            validate_research_adapter(invalid, expected_name="expected")

    def test_adapter_receipt_refuses_credentials(self) -> None:
        class UnsafeAdapter:
            descriptor = ResearchAdapterDescriptor(
                name="unsafe",
                version="1.0",
                description="unsafe test adapter",
                capabilities=("publish",),
                destination_kinds=("uri",),
            )

            def probe(self):
                return {"ok": True, "requirements": [], "errors": []}

            def publish(self, package_root, destination, *, options):
                del package_root, destination, options
                return {"status": "published", "access_token": "must-not-persist"}

        with mock.patch(
            "xscientist.research_adapters.load_research_adapter",
            return_value=UnsafeAdapter(),
        ):
            with self.assertRaisesRegex(ResearchGitError, "sensitive data"):
                sync_research_repository(
                    self.repo,
                    adapter_name="unsafe",
                    destination="ignored",
                    formats=["ro-crate"],
                )

    def test_external_tool_receipt_enters_dag_as_unverified_evidence(self) -> None:
        repository = ResearchRepository(self.repo)
        hypothesis_id = repository.resolve("@latest:hypothesis")
        plan = repository.record(
            "research_plan",
            {"summary": "External tool test"},
            relations=[{"type": "depends_on", "target": hypothesis_id}],
        )
        attempt = repository.record(
            "experiment_attempt",
            {"status": "completed"},
            state="completed",
            relations=[{"type": "depends_on", "target": plan.object_id}],
        )
        repository.commit(stage="experiment", subject="run external tool")
        receipt_file = self.root / "tool-evidence.json"
        receipt_file.write_text(
            json.dumps(
                {
                    "schema_version": "xscientist.tool-evidence.v1",
                    "tool": {"name": "mlflow", "version": "3.0"},
                    "run_id": "run-123",
                    "result": "Accuracy reached 0.82.",
                    "metrics": {"accuracy": 0.82},
                    "artifact_hashes": ["sha256:" + "f" * 64],
                }
            ),
            encoding="utf-8",
        )
        output = io.StringIO()
        with redirect_stdout(output):
            code = research_main(
                [
                    "ingest",
                    str(receipt_file),
                    "--repo",
                    str(self.repo),
                    "--attempt",
                    attempt.object_id,
                    "--supports",
                    hypothesis_id,
                    "--json",
                ]
            )

        self.assertEqual(code, 0)
        object_id = json.loads(output.getvalue())["object"]["object_id"]
        evidence = repository.get(object_id)
        self.assertEqual(evidence["state"], "completed")
        self.assertEqual(evidence["actor"]["actor_id"], "tool:mlflow")
        self.assertEqual(evidence["actor"]["authority"], "recorder")
        self.assertNotIn(str(receipt_file), json.dumps(evidence))
        self.assertEqual(
            {relation["type"] for relation in evidence["relations"]},
            {"derived_from", "supports"},
        )


if __name__ == "__main__":
    unittest.main()
