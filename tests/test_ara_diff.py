"""Tests for the ARA diff engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_diff import diff_ara


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _project(tmp: Path, *, sub: str, nodes: list[dict]) -> tuple[Path, Path]:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260709_{sub}"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(exp / "logs", "0-run", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    return project, exp


def _make_node(nid: str, *, code: str = "print('ok')",
               metric_val: float = 0.5,
               llm_refs: list[str] | None = None) -> dict:
    d = {
        "id": nid, "step": 0, "code": code,
        "_term_out": [],
        "metric": {"value": metric_val, "maximize": True, "name": "acc"},
        "is_buggy": False, "parent_id": None, "children": [],
    }
    if llm_refs:
        d["llm_call_refs"] = llm_refs
    return d


class DiffEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # Same-input identical ARAs must diff to "no changes"
    # ------------------------------------------------------------------
    def test_identical_aras_report_no_changes(self) -> None:
        pa, ea = _project(self.tmp, sub="a", nodes=[_make_node("n1")])
        pb, eb = _project(self.tmp, sub="b", nodes=[_make_node("n1")])
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})

        d = diff_ara(ra.root, rb.root)
        self.assertEqual(d.nodes_added, [])
        self.assertEqual(d.nodes_removed, [])
        self.assertEqual(d.nodes_hash_changed, [])
        self.assertEqual(d.nodes_unchanged, 1)
        # Manifest hashes will differ because timestamps/project_dir differ,
        # but the interesting bits (counts / provenance) must not.
        self.assertNotIn("counts", d.manifest.field_changes)

    # ------------------------------------------------------------------
    # Added / removed nodes
    # ------------------------------------------------------------------
    def test_added_and_removed_nodes(self) -> None:
        pa, ea = _project(self.tmp, sub="a",
                          nodes=[_make_node("n1"), _make_node("n2")])
        pb, eb = _project(self.tmp, sub="b",
                          nodes=[_make_node("n2"), _make_node("n3")])
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})

        d = diff_ara(ra.root, rb.root)
        self.assertEqual([n.id for n in d.nodes_added], ["n3"])
        self.assertEqual([n.id for n in d.nodes_removed], ["n1"])
        self.assertEqual(d.nodes_unchanged, 1)

    # ------------------------------------------------------------------
    # Hash-changed nodes get categorized
    # ------------------------------------------------------------------
    def test_hash_changed_code_only(self) -> None:
        pa, ea = _project(self.tmp, sub="a",
                          nodes=[_make_node("n1", code="print('a')")])
        pb, eb = _project(self.tmp, sub="b",
                          nodes=[_make_node("n1", code="print('b')")])
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})

        d = diff_ara(ra.root, rb.root)
        self.assertEqual(len(d.nodes_hash_changed), 1)
        self.assertIn("code", d.nodes_hash_changed[0].changed_categories)

    def test_hash_changed_metric_only(self) -> None:
        pa, ea = _project(self.tmp, sub="a",
                          nodes=[_make_node("n1", metric_val=0.5)])
        pb, eb = _project(self.tmp, sub="b",
                          nodes=[_make_node("n1", metric_val=0.9)])
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})

        d = diff_ara(ra.root, rb.root)
        self.assertEqual(len(d.nodes_hash_changed), 1)
        self.assertIn("metric", d.nodes_hash_changed[0].changed_categories)
        self.assertNotIn("code", d.nodes_hash_changed[0].changed_categories)

    def test_hash_changed_llm_calls_only(self) -> None:
        h1 = "sha256:" + "a" * 64
        h2 = "sha256:" + "b" * 64
        pa, ea = _project(self.tmp, sub="a",
                          nodes=[_make_node("n1", llm_refs=[h1])])
        pb, eb = _project(self.tmp, sub="b",
                          nodes=[_make_node("n1", llm_refs=[h2])])
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})

        d = diff_ara(ra.root, rb.root)
        self.assertEqual(len(d.nodes_hash_changed), 1)
        self.assertIn("llm_calls", d.nodes_hash_changed[0].changed_categories)

    # ------------------------------------------------------------------
    # Prompts (llm/calls.jsonl)
    # ------------------------------------------------------------------
    def test_prompt_delta_when_calls_differ(self) -> None:
        pa, ea = _project(self.tmp, sub="a", nodes=[_make_node("n1")])
        pb, eb = _project(self.tmp, sub="b", nodes=[_make_node("n1")])
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})

        # Fabricate llm/calls.jsonl on each side
        (Path(ra.root) / "llm").mkdir(exist_ok=True)
        (Path(rb.root) / "llm").mkdir(exist_ok=True)
        common = {"messages_ref": {"hash": "sha256:common"}}
        onlya = {"messages_ref": {"hash": "sha256:onlyaXXX"}}
        onlyb = {"messages_ref": {"hash": "sha256:onlybXXX"}}
        (Path(ra.root) / "llm" / "calls.jsonl").write_text(
            json.dumps(common) + "\n" + json.dumps(onlya) + "\n", encoding="utf-8")
        (Path(rb.root) / "llm" / "calls.jsonl").write_text(
            json.dumps(common) + "\n" + json.dumps(onlyb) + "\n", encoding="utf-8")

        d = diff_ara(ra.root, rb.root)
        self.assertEqual(d.prompts.total_a, 2)
        self.assertEqual(d.prompts.total_b, 2)
        self.assertEqual(d.prompts.shared, 1)
        self.assertEqual(d.prompts.only_in_a, 1)
        self.assertEqual(d.prompts.only_in_b, 1)

    # ------------------------------------------------------------------
    # References (pipeline_artifacts)
    # ------------------------------------------------------------------
    def test_pipeline_added_and_hash_changed(self) -> None:
        # Prepare A with review_state present; B with review_state present but
        # different content, plus a new critic_findings entry.
        from ai_scientist.utils.pipeline_contracts import ARTIFACT_FILENAMES

        pa, ea = _project(self.tmp, sub="a", nodes=[_make_node("n1")])
        (ea / ARTIFACT_FILENAMES["review_state"]).write_text(
            json.dumps({"score": 4.0}), encoding="utf-8")
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})

        pb, eb = _project(self.tmp, sub="b", nodes=[_make_node("n1")])
        (eb / ARTIFACT_FILENAMES["review_state"]).write_text(
            json.dumps({"score": 4.5}), encoding="utf-8")
        (eb / ARTIFACT_FILENAMES["critic_findings"]).write_text(
            json.dumps({"findings": []}), encoding="utf-8")
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})

        d = diff_ara(ra.root, rb.root)
        self.assertIn("critic_findings", d.references.pipeline_added)
        kinds_changed = [e["kind"] for e in d.references.pipeline_hash_changed]
        self.assertIn("review_state", kinds_changed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
