from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml

from ai_scientist.protocol.schemas import schema_validator
from xscientist.research_cli import main as research_main
from xscientist.research_commands import save_experiment, save_hypothesis
from xscientist.research_git import ResearchGitError
from xscientist.research_interop import export_research_interop
from xscientist.research_vcs import ResearchRepository


class ResearchInteropTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = tempfile.mkdtemp(prefix="xscientist-interop-")
        self.root = Path(self.raw)
        self.repo = self.root / "study"
        ResearchRepository.init(
            self.repo,
            name="interop-study",
            git_user_name="XScientist Tests",
            git_user_email="tests@example.invalid",
        )
        save_hypothesis(
            str(self.repo),
            statement="A transparent export preserves scientific relations.",
            falsifier="The exported relation graph is incomplete.",
        )
        save_experiment(
            str(self.repo),
            summary="Measure export completeness.",
            status="completed",
            metrics={"coverage": 1.0},
            seeds=[7],
            dataset_hashes=["sha256:" + "a" * 64],
            reproduce_command="python -m pytest tests/test_research_interop.py",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.raw)

    def test_export_all_adapters_is_atomic_and_metadata_safe(self) -> None:
        destination = self.root / "export"
        result = export_research_interop(self.repo, destination)
        schema_validator("research_interop").validate(
            json.loads((destination / "xscientist-export.json").read_text())
        )
        self.assertEqual(
            set(result["formats"]),
            {"ro-crate", "prov-json", "cwl", "dvc", "mlflow"},
        )
        self.assertTrue(result["export_hash"].startswith("sha256:"))
        crate = json.loads((destination / "ro-crate-metadata.json").read_text())
        entities = [
            item
            for item in crate["@graph"]
            if str(item.get("@id", "")).startswith("urn:xscientist:research-object:")
        ]
        self.assertGreaterEqual(len(entities), 2)
        self.assertTrue(all("xscientist:payload" not in item for item in entities))
        prov = json.loads((destination / "research.prov.json").read_text())
        self.assertGreaterEqual(len(prov["entity"]), 2)
        cwl = yaml.safe_load((destination / "research-workflow.cwl").read_text())
        self.assertEqual(cwl["cwlVersion"], "v1.2")
        self.assertGreaterEqual(len(cwl["$graph"]), 2)
        dvc = yaml.safe_load((destination / "dvc.yaml").read_text())
        self.assertTrue(dvc["stages"])
        mlflow = json.loads((destination / "mlflow-runs.json").read_text())
        self.assertEqual(len(mlflow["runs"]), 1)
        with self.assertRaises(ResearchGitError):
            export_research_interop(self.repo, destination)

    def test_payload_export_is_explicit(self) -> None:
        destination = self.root / "payload-export"
        export_research_interop(
            self.repo,
            destination,
            formats=["ro-crate"],
            include_payloads=True,
        )
        crate = json.loads((destination / "ro-crate-metadata.json").read_text())
        entities = [
            item
            for item in crate["@graph"]
            if str(item.get("@id", "")).startswith("urn:xscientist:research-object:")
        ]
        self.assertTrue(any("xscientist:payload" in item for item in entities))

    def test_user_facing_export_command(self) -> None:
        destination = self.root / "cli-export"
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = research_main(
                [
                    "export",
                    "--repo",
                    str(self.repo),
                    "--dest",
                    str(destination),
                    "--format",
                    "ro-crate",
                    "--format",
                    "prov-json",
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["formats"], ["ro-crate", "prov-json"])
        self.assertTrue((destination / "xscientist-export.json").is_file())
