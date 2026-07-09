"""End-to-end test: pipeline_contracts artifacts land in ARA + objects/."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol import ObjectStore
from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.pipeline_contracts import ARTIFACT_FILENAMES


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _minimal_project(tmp: Path) -> tuple[Path, Path]:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260709_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(exp / "logs", "0-run", [{
        "id": "n1", "step": 0, "code": "print('ok')",
        "_term_out": ["ok\n"],
        "metric": {"value": 0.5, "maximize": True, "name": "acc", "description": ""},
        "is_buggy": False, "parent_id": None, "children": [],
    }])
    (exp / "idea.json").write_text(json.dumps({"Name": "idea", "Title": "T"}), encoding="utf-8")
    return project, exp


class PipelineArtifactsSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.project, self.exp = _minimal_project(self.tmp)

    def _seed_pipeline_files(self, files: dict[str, str]) -> None:
        """Drop pipeline artifacts into the experiment dir at their canonical names."""
        for kind, content in files.items():
            filename = ARTIFACT_FILENAMES[kind]
            (self.exp / filename).write_text(content, encoding="utf-8")

    def test_pipeline_artifacts_present_are_snapshotted(self) -> None:
        payload_review = json.dumps({"score": 4.2})
        payload_critic = json.dumps({"findings": [{"id": "c1"}]})
        self._seed_pipeline_files({
            "review_state": payload_review,
            "critic_findings": payload_critic,
        })

        result = export_ara(
            project_dir=self.project,
            exp_dir=self.exp,
            idea={"Name": "idea", "Title": "T"},
        )
        ara_dir = Path(result.root)
        manifest = json.loads((ara_dir / "manifest.json").read_text())

        entries = manifest["references"].get("pipeline_artifacts", [])
        kinds = {e["kind"] for e in entries}
        self.assertIn("review_state", kinds)
        self.assertIn("critic_findings", kinds)

        # pipeline/ mirror exists with the same bytes
        review_mirror = ara_dir / "pipeline" / "review_state.json"
        critic_mirror = ara_dir / "pipeline" / "critic_findings.json"
        self.assertTrue(review_mirror.exists())
        self.assertEqual(review_mirror.read_text(), payload_review)
        self.assertTrue(critic_mirror.exists())
        self.assertEqual(critic_mirror.read_text(), payload_critic)

        # CAS blobs exist and hash matches manifest claim
        store = ObjectStore(ara_dir)
        for entry in entries:
            self.assertTrue(store.exists(entry["content_hash"]))
            got = store.get_bytes(entry["content_hash"])
            self.assertEqual(got, (ara_dir / entry["path"]).read_bytes())
            self.assertEqual(entry["size"], len(got))

    def test_absent_artifacts_do_not_bloat_missing(self) -> None:
        # No pipeline files at all → single roll-up note, not 16.
        result = export_ara(
            project_dir=self.project,
            exp_dir=self.exp,
            idea={"Name": "idea", "Title": "T"},
        )
        ara_dir = Path(result.root)
        manifest = json.loads((ara_dir / "manifest.json").read_text())
        pipeline_missing = [m for m in manifest.get("missing", [])
                            if m.startswith("pipeline_artifacts")]
        self.assertEqual(len(pipeline_missing), 1)
        self.assertIn("absent", pipeline_missing[0])

    def test_experiment_dir_wins_over_project_dir(self) -> None:
        # Project-level version should be overridden by experiment-level.
        (self.project / ARTIFACT_FILENAMES["review_state"]).write_text("PROJECT", encoding="utf-8")
        (self.exp / ARTIFACT_FILENAMES["review_state"]).write_text("EXPERIMENT", encoding="utf-8")

        result = export_ara(
            project_dir=self.project,
            exp_dir=self.exp,
            idea={"Name": "idea", "Title": "T"},
        )
        ara_dir = Path(result.root)
        mirror = ara_dir / "pipeline" / ARTIFACT_FILENAMES["review_state"]
        self.assertEqual(mirror.read_text(), "EXPERIMENT")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
