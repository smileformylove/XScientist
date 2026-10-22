from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main


@unittest.skipUnless(shutil.which("git"), "Git is required")
class ResearchDagCliContractTests(unittest.TestCase):
    def test_blank_output_and_advisory_index_are_rejected(self) -> None:
        cases = (
            (["dag", "--output", ""], "--output must not be empty"),
            (
                ["dag", "--output", "dag", "--advisory-index", "   "],
                "--advisory-index must not be empty",
            ),
            (
                ["dag", "--advisory-index", ""],
                "--advisory-index must not be empty",
            ),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                errors = io.StringIO()
                with redirect_stderr(errors):
                    status = research_main(argv)
                self.assertEqual(status, 2)
                self.assertIn(expected, errors.getvalue())

    def test_advisory_json_returns_resolvable_relative_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            ResearchRepository.init(
                root,
                question="Which latent world-model claim survives?",
                git_user_name="DAG CLI Contract Test",
                git_user_email="dag-cli-contract@example.invalid",
            )
            advisory = root / "advisory"
            advisory.mkdir()
            witness = advisory / "witness.md"
            contract = advisory / "contract.json"
            witness.write_bytes(b"bound witness\n")
            contract.write_bytes(b'{"gate":"W0"}\n')
            index = {
                "schema_version": "xscientist.advisory-route-index.v1",
                "routes": [
                    {
                        "route_id": "Q5-XM",
                        "question": "Does the witness survive?",
                        "authority": "UNREGISTERED",
                        "artifact_bindings": [
                            {
                                "path": "advisory/witness.md",
                                "raw_sha256": hashlib.sha256(
                                    witness.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                        "gate_contract_binding": {
                            "path": "advisory/contract.json",
                            "raw_sha256": hashlib.sha256(
                                contract.read_bytes()
                            ).hexdigest(),
                        },
                        "gate_results": [],
                        "open_findings": [],
                        "novelty_disposition": "OPEN",
                        "registration_state": "absent",
                        "permitted_next_action": "Review the route.",
                    }
                ],
            }
            (root / "advisory-index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            destination = root / "exports" / "argument"
            output = io.StringIO()
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                with redirect_stdout(output):
                    status = research_main(
                        [
                            "dag",
                            "--repo",
                            ".",
                            "--output",
                            str(destination),
                            "--advisory-index",
                            "advisory-index.json",
                            "--json",
                        ]
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(status, 0)
            report = json.loads(output.getvalue())
            self.assertNotIn("[REDACTED_PATH]", report["outputs"].values())
            for value in report["outputs"].values():
                self.assertFalse(Path(value).is_absolute())
                self.assertTrue((root / value).is_file())


if __name__ == "__main__":
    unittest.main()
