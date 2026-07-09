"""Regression tests for env/model_fingerprint.json — profile CONTENT hashing.

Before this suite, `writing_profile` was recorded as a bare name. Two runs
with the same profile name but an edited profile body produced identical
fingerprints, silently breaking A/B comparability across profile tweaks.
The fingerprint now stores `{"name": ..., "content_hash": "sha256:..."}` so
downstream diff can detect body changes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _seed(exp_dir: Path) -> None:
    (exp_dir / "logs" / "0-run").mkdir(parents=True, exist_ok=True)
    _write_journal(
        exp_dir / "logs",
        "0-run",
        [
            {
                "id": "n0",
                "step": 0,
                "code": "print('hi')",
                "_term_out": ["hi\n"],
                "metric": {"name": "acc", "value": 0.1, "maximize": True},
                "analysis": "baseline",
                "is_buggy": False,
                "parent_id": None,
                "children": [],
            }
        ],
    )


def _load_fingerprint(project_dir: Path, exp_dir: Path, writing_profile) -> dict:
    result = export_ara(
        project_dir=project_dir,
        exp_dir=exp_dir,
        idea={"Name": "idea_x"},
        timestamp="20260710",
        model_spec={"writeup": "opus"},
        writing_profile=writing_profile,
    )
    manifest = json.loads(result.manifest_path.read_text())
    fp_rel = manifest["references"]["env"]["model_fingerprint"]
    return json.loads((result.root / fp_rel).read_text())


class WritingProfileFingerprintTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.project_dir = Path(self._tmp.name)
        self.exp_dir = self.project_dir / "02_experiments" / "20260710_idea_x"
        self.exp_dir.mkdir(parents=True)
        _seed(self.exp_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_writing_profile_content_hash_present_when_name_resolves(self) -> None:
        fp = _load_fingerprint(self.project_dir, self.exp_dir, "default")
        slot = fp["spec"]["writing_profile"]
        self.assertEqual(slot["name"], "default")
        self.assertIsInstance(slot["content_hash"], str)
        self.assertTrue(slot["content_hash"].startswith("sha256:"))

    def test_writing_profile_content_hash_null_when_name_unknown(self) -> None:
        fp = _load_fingerprint(self.project_dir, self.exp_dir, "__nonexistent__")
        slot = fp["spec"]["writing_profile"]
        self.assertEqual(slot["name"], "__nonexistent__")
        self.assertIsNone(slot["content_hash"])

    def test_writing_profile_content_hash_null_when_name_missing(self) -> None:
        fp = _load_fingerprint(self.project_dir, self.exp_dir, None)
        slot = fp["spec"]["writing_profile"]
        self.assertIsNone(slot["name"])
        self.assertIsNone(slot["content_hash"])

    def test_edited_profile_body_changes_fingerprint_content_hash(self) -> None:
        from ai_scientist import writing_prompt_profiles as wpp

        original = dict(wpp.WRITING_PROFILE_SPECS["default"])
        try:
            fp_before = _load_fingerprint(self.project_dir, self.exp_dir, "default")

            # Mutate the profile body; same name, different content.
            edited = dict(original)
            edited["summary"] = original.get("summary", "") + " EDITED"
            wpp.WRITING_PROFILE_SPECS["default"] = edited

            fp_after = _load_fingerprint(self.project_dir, self.exp_dir, "default")

            before = fp_before["spec"]["writing_profile"]
            after = fp_after["spec"]["writing_profile"]
            self.assertEqual(before["name"], after["name"])
            self.assertNotEqual(
                before["content_hash"],
                after["content_hash"],
                "Editing profile body must alter the writing_profile content_hash",
            )
            # And the top-level fingerprint digest must also change, since
            # writing_profile.content_hash feeds it.
            self.assertNotEqual(fp_before["fingerprint"], fp_after["fingerprint"])
        finally:
            wpp.WRITING_PROFILE_SPECS["default"] = original


if __name__ == "__main__":
    unittest.main()
