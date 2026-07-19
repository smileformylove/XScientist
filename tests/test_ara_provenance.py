"""Smoke tests for the `provenance` verb in run_ara_fork.py.

The verb is a thin fan-out over already-covered readers (manifest.json,
manifest.lock, exploration_graph.json, refs). These tests pin the six
match kinds (manifest / node / llm_call / seed / provenance / ref) and
the empty-result invariant.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.apps import ara as run_ara_fork

from ai_scientist.utils.ara_artifact import export_ara


def _write_journal(logs_dir: Path, nodes: list[dict]) -> None:
    stage_dir = logs_dir / "0-run"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _make_ara(tmp: Path, sub: str, code: str = "print('ok')") -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    _write_journal(exp / "logs", [{
        "id": "n1", "step": 0, "code": code,
        "_term_out": [],
        "metric": {"value": 0.5, "maximize": True, "name": "acc"},
        "is_buggy": False, "parent_id": None, "children": [],
    }])
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    result = export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub})
    return Path(result.root)


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def _node_hash(ara_root: Path, node_id: str) -> str:
    graph = json.loads((ara_root / "exploration_graph.json").read_text(encoding="utf-8"))
    for n in graph.get("nodes") or []:
        if n.get("id") == node_id:
            return n["content_hash"]
    raise AssertionError(f"node {node_id} missing from {ara_root}")


class ProvenanceCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_provenance_finds_node_content_hash(self) -> None:
        ara = _make_ara(self.tmp, "a")
        h = _node_hash(ara, "n1")
        rc, out, _ = _run("provenance", "--hash", h,
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        kinds = {(hit["kind"], (hit.get("detail") or {}).get("node_id")) for hit in hits}
        self.assertIn(("node", "n1"), kinds)

    def test_provenance_finds_manifest_hash(self) -> None:
        ara = _make_ara(self.tmp, "a")
        lock = json.loads((ara / "manifest.lock").read_text(encoding="utf-8"))
        rc, out, _ = _run("provenance", "--hash", lock["manifest_hash"],
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        self.assertTrue(any(h["kind"] == "manifest" for h in hits))

    def test_provenance_finds_ref_target(self) -> None:
        ara = _make_ara(self.tmp, "a")
        target = "sha256:" + "c" * 64
        rc, _, _ = _run("refs", "--ara", str(ara), "--set", "head", target)
        self.assertEqual(rc, 0)
        rc, out, _ = _run("provenance", "--hash", target,
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        ref_hits = [h for h in hits if h["kind"] == "ref"]
        self.assertTrue(ref_hits)
        self.assertEqual(ref_hits[0]["detail"]["ref_name"], "head")

    def test_provenance_finds_llm_call_ref(self) -> None:
        ara = _make_ara(self.tmp, "a")
        graph_path = ara / "exploration_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        target = "sha256:" + "d" * 64
        graph["nodes"][0]["llm_call_refs"] = [target]
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        rc, out, _ = _run("provenance", "--hash", target,
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        self.assertTrue(any(h["kind"] == "llm_call" and
                            (h.get("detail") or {}).get("node_id") == "n1"
                            for h in hits))

    def test_provenance_finds_seed_hash(self) -> None:
        ara = _make_ara(self.tmp, "a")
        manifest_path = ara / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed_hash = "sha256:" + "e" * 64
        manifest["provenance"] = {"seed_hash": seed_hash}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        rc, out, _ = _run("provenance", "--hash", seed_hash,
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        self.assertTrue(any(h["kind"] == "seed" for h in hits))

    def test_provenance_finds_hash_in_parents_array(self) -> None:
        from ai_scientist.protocol.hashing import build_provenance
        ara = _make_ara(self.tmp, "a")
        manifest_path = ara / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        code_hash = "sha256:" + "a" * 64
        env_hash = "sha256:" + "b" * 64
        manifest["provenance"] = build_provenance(parents=[
            {"role": "code", "parent_content_hash": code_hash,
             "parent_node_id": "nc"},
            {"role": "env", "parent_content_hash": env_hash,
             "parent_node_id": "ne"},
        ])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        rc, out, _ = _run("provenance", "--hash", env_hash,
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        parents_hits = [h for h in hits if h["kind"] == "provenance"
                        and (h.get("detail") or {}).get("field") == "parents[]"]
        self.assertEqual(len(parents_hits), 1)
        self.assertEqual(parents_hits[0]["detail"]["role"], "env")
        self.assertEqual(parents_hits[0]["detail"]["parent_node_id"], "ne")

    def test_provenance_finds_top_level_and_parents_array_together(self) -> None:
        from ai_scientist.protocol.hashing import build_provenance
        ara = _make_ara(self.tmp, "a")
        manifest_path = ara / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        code_hash = "sha256:" + "a" * 64
        env_hash = "sha256:" + "b" * 64
        manifest["provenance"] = build_provenance(parents=[
            {"role": "code", "parent_content_hash": code_hash,
             "parent_node_id": "nc"},
            {"role": "env", "parent_content_hash": env_hash,
             "parent_node_id": "ne"},
        ])
        # code_hash will match both the top-level (elected code-role echo)
        # AND the parents[] scan — dedup is a separate concern.
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        rc, out, _ = _run("provenance", "--hash", code_hash,
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        self.assertTrue(any(h["kind"] == "provenance" for h in hits))

    def test_provenance_no_matches_empty_json(self) -> None:
        _make_ara(self.tmp, "a")
        missing = "sha256:" + "f" * 64
        rc, out, err = _run("provenance", "--hash", missing,
                            "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [])
        self.assertIn(missing, err)

    def test_provenance_walks_multiple_aras(self) -> None:
        # Two ARAs under the SAME project (ara/ subtree) so the sweep
        # sees both — different code so the node hashes differ.
        project = self.tmp / "multi"
        exp_a = project / "02_experiments" / "20260710_a"
        exp_b = project / "02_experiments" / "20260710_b"
        _write_journal(exp_a / "logs", [{
            "id": "n1", "step": 0, "code": "print('a')", "_term_out": [],
            "metric": {"value": 0.1, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        }])
        _write_journal(exp_b / "logs", [{
            "id": "n1", "step": 0, "code": "print('b')", "_term_out": [],
            "metric": {"value": 0.2, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        }])
        (exp_a / "idea.json").write_text(json.dumps({"Name": "idea_a"}))
        (exp_b / "idea.json").write_text(json.dumps({"Name": "idea_b"}))
        r_a = export_ara(project_dir=project, exp_dir=exp_a, idea={"Name": "idea_a"})
        r_b = export_ara(project_dir=project, exp_dir=exp_b, idea={"Name": "idea_b"})
        h_a = _node_hash(Path(r_a.root), "n1")
        rc, out, _ = _run("provenance", "--hash", h_a,
                          "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        node_hits = [h for h in hits if h["kind"] == "node"]
        self.assertEqual(len(node_hits), 1)
        self.assertEqual(node_hits[0]["ara_root"], str(r_a.root))
        self.assertNotEqual(node_hits[0]["ara_root"], str(r_b.root))

    def test_provenance_json_shape(self) -> None:
        ara = _make_ara(self.tmp, "a")
        h = _node_hash(ara, "n1")
        rc, out, _ = _run("provenance", "--hash", h,
                          "--project", str(self.tmp / "a"), "--json")
        self.assertEqual(rc, 0)
        hits = json.loads(out)
        self.assertTrue(hits)
        for hit in hits:
            self.assertIn("kind", hit)
            self.assertIn("detail", hit)
            self.assertIn("ara_root", hit)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
