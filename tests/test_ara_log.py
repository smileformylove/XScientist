"""Tests for the ARA log engine (revisions + ancestry)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara, update_manifest_claim_count
from ai_scientist.utils.ara_log import ara_log, walk_node_ancestry

# Load the CLI module under a private name so the log-CLI tests can drive main().
from ai_scientist.apps import ara as _run_ara_fork


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _project(
    tmp: Path, sub: str, nodes: list[dict], *, provenance: dict | None = None
) -> tuple[Path, Path]:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260709_{sub}"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(exp / "logs", "0-run", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    return project, exp


def _basic_node(nid: str) -> dict:
    return {
        "id": nid,
        "step": 0,
        "code": "print('ok')",
        "_term_out": [],
        "metric": {"value": 0.5, "maximize": True, "name": "acc"},
        "is_buggy": False,
        "parent_id": None,
        "children": [],
    }


class LogEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    # ------------------------------------------------------------------
    # Revision section
    # ------------------------------------------------------------------
    def test_fresh_export_shows_no_revisions(self) -> None:
        p, e = _project(self.tmp, "a", [_basic_node("n1")])
        r = export_ara(project_dir=p, exp_dir=e, idea={"Name": "a"})
        log = ara_log(r.root)
        self.assertIsNotNone(log.lock)
        self.assertEqual(log.revisions, [])
        self.assertEqual(log.verify["state"], "clean")

    def test_revisions_appear_after_update(self) -> None:
        p, e = _project(self.tmp, "a", [_basic_node("n1")])
        r = export_ara(project_dir=p, exp_dir=e, idea={"Name": "a"})
        update_manifest_claim_count(Path(r.root) / "manifest.json", claim_count=3)
        update_manifest_claim_count(Path(r.root) / "manifest.json", claim_count=7)
        log = ara_log(r.root)
        self.assertEqual([rev.revision for rev in log.revisions], [1, 2])
        # Chain integrity: rev 2's base_hash equals rev 1's new_hash.
        self.assertEqual(log.revisions[1].base_hash, log.revisions[0].new_hash)
        self.assertEqual(log.verify["state"], "revised")

    # ------------------------------------------------------------------
    # Ancestry
    # ------------------------------------------------------------------
    def test_root_ara_has_no_ancestors(self) -> None:
        p, e = _project(self.tmp, "root", [_basic_node("n1")])
        r = export_ara(project_dir=p, exp_dir=e, idea={"Name": "root"})
        log = ara_log(r.root)
        self.assertEqual(log.ancestors, [])

    def test_provenance_reachable_and_verified(self) -> None:
        # Build parent ARA.
        pp, ep = _project(self.tmp, "parent", [_basic_node("np")])
        parent = export_ara(project_dir=pp, exp_dir=ep, idea={"Name": "parent"})
        # Find parent node's content_hash
        parent_graph = json.loads(
            (Path(parent.root) / "exploration_graph.json").read_text()
        )
        parent_node_hash = parent_graph["nodes"][0]["content_hash"]

        # Build child ARA that claims parent as its ancestor.
        pc, ec = _project(self.tmp, "child", [_basic_node("nc")])
        child = export_ara(
            project_dir=pc,
            exp_dir=ec,
            idea={"Name": "child"},
            provenance={
                "parent_ara_root": str(parent.root),
                "parent_node_id": "np",
                "parent_content_hash": parent_node_hash,
            },
        )
        log = ara_log(child.root)
        self.assertEqual(len(log.ancestors), 1)
        anc = log.ancestors[0]
        self.assertEqual(anc.depth, 1)
        self.assertFalse(Path(anc.ara_root).is_absolute())
        self.assertEqual((Path(child.root) / anc.ara_root).resolve(), parent.root)
        self.assertTrue(anc.reachable)
        self.assertTrue(anc.hash_verified)

    def test_provenance_unreachable_when_parent_path_missing(self) -> None:
        pc, ec = _project(self.tmp, "orphan", [_basic_node("nc")])
        child = export_ara(
            project_dir=pc,
            exp_dir=ec,
            idea={"Name": "orphan"},
            provenance={
                "parent_ara_root": "/nonexistent/parent/path/xyz",
                "parent_node_id": "np",
                "parent_content_hash": "sha256:" + "d" * 64,
            },
        )
        log = ara_log(child.root)
        self.assertEqual(len(log.ancestors), 1)
        anc = log.ancestors[0]
        self.assertFalse(anc.reachable)
        # content_hash still records the parent identity even without a path
        self.assertEqual(anc.content_hash, "sha256:" + "d" * 64)

    def test_ancestry_verify_reports_mismatch_when_parent_node_hash_differs(
        self,
    ) -> None:
        pp, ep = _project(self.tmp, "p", [_basic_node("np")])
        parent = export_ara(project_dir=pp, exp_dir=ep, idea={"Name": "p"})
        pc, ec = _project(self.tmp, "c", [_basic_node("nc")])
        child = export_ara(
            project_dir=pc,
            exp_dir=ec,
            idea={"Name": "c"},
            provenance={
                "parent_ara_root": str(parent.root),
                "parent_node_id": "np",
                # Deliberately wrong hash.
                "parent_content_hash": "sha256:" + "0" * 64,
            },
        )
        log = ara_log(child.root)
        anc = log.ancestors[0]
        self.assertTrue(anc.reachable)
        self.assertFalse(anc.hash_verified)


# ---------------------------------------------------------------------------
# walk_node_ancestry — in-ARA parent_id chain
# ---------------------------------------------------------------------------


def _write_exploration_graph(ara_root: Path, nodes: list[dict]) -> None:
    # Bypass the journal→graph pipeline so ancestry tests can sculpt the
    # chain directly (including a cycle no journal writer would emit).
    ara_root.mkdir(parents=True, exist_ok=True)
    (ara_root / "exploration_graph.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "protocol_kind": "exploration_graph",
                "nodes": nodes,
                "edges": [],
            }
        ),
        encoding="utf-8",
    )


def _node(
    nid: str,
    *,
    parent: str | None = None,
    buggy: bool = False,
    seed: bool = False,
    metric: object = None,
) -> dict:
    return {
        "id": nid,
        "parent_id": parent,
        "is_buggy": buggy,
        "is_seed_node": seed,
        "metric": metric,
        "content_hash": f"sha256:{(nid * 20)[:64]}",
    }


class WalkNodeAncestryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_walk_node_ancestry_leaf_to_root(self) -> None:
        ara = self.tmp / "ara"
        _write_exploration_graph(
            ara,
            [
                _node("n1", metric={"value": 0.1}),
                _node("n2", parent="n1", metric={"value": 0.2}),
                _node("n3", parent="n2", buggy=True, metric={"value": 0.3}),
            ],
        )
        chain = walk_node_ancestry(ara, "n3")
        self.assertEqual([e["id"] for e in chain], ["n3", "n2", "n1"])
        self.assertTrue(chain[0]["is_buggy"])
        self.assertIsNone(chain[-1]["parent_id"])
        self.assertEqual(chain[0]["metric"]["value"], 0.3)

    def test_walk_node_ancestry_root_node(self) -> None:
        ara = self.tmp / "root"
        _write_exploration_graph(ara, [_node("only", seed=True)])
        chain = walk_node_ancestry(ara, "only")
        self.assertEqual(len(chain), 1)
        self.assertTrue(chain[0]["is_seed_node"])
        self.assertIsNone(chain[0]["parent_id"])

    def test_walk_node_ancestry_unknown_id_raises(self) -> None:
        ara = self.tmp / "u"
        _write_exploration_graph(ara, [_node("n1")])
        with self.assertRaises(KeyError):
            walk_node_ancestry(ara, "ghost")

    def test_walk_node_ancestry_cycle_guard(self) -> None:
        ara = self.tmp / "cyc"
        _write_exploration_graph(
            ara, [_node("n1", parent="n2"), _node("n2", parent="n1")]
        )
        chain = walk_node_ancestry(ara, "n1")
        # Terminates; each node appears at most once; note explains truncation.
        self.assertEqual(sorted(e["id"] for e in chain), ["n1", "n2"])
        self.assertIn("cycle", (chain[-1].get("note") or ""))


# ---------------------------------------------------------------------------
# CLI: `log --node <id>` — thin driver around walk_node_ancestry
# ---------------------------------------------------------------------------


def _make_ara(tmp: Path, sub: str, nodes: list[dict]) -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    stage = exp / "logs" / "0-run"
    stage.mkdir(parents=True)
    (stage / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    return Path(export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub}).root)


def _journal_node(
    nid: str, parent: str | None = None, children: list | None = None
) -> dict:
    return {
        "id": nid,
        "step": 0,
        "code": "print('x')",
        "_term_out": [],
        "metric": {"value": 0.5, "maximize": True, "name": "acc"},
        "is_buggy": False,
        "parent_id": parent,
        "children": children or [],
    }


def _run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class LogNodeCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_log_node_json_shape(self) -> None:
        ara = _make_ara(
            self.tmp,
            "chain",
            [
                _journal_node("n1", children=["n2"]),
                _journal_node("n2", parent="n1"),
            ],
        )
        rc, out, _ = _run_cli("log", "--ara", str(ara), "--node", "n2", "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual([e["id"] for e in payload], ["n2", "n1"])
        # JSON output preserves the full sha256 (no ellipsis truncation).
        self.assertGreater(len(payload[0]["content_hash"].split(":", 1)[1]), 16)

    def test_log_node_unknown_returns_rc3(self) -> None:
        ara = _make_ara(self.tmp, "u", [_journal_node("n1")])
        rc, _, err = _run_cli("log", "--ara", str(ara), "--node", "bogus")
        self.assertEqual(rc, 3)
        self.assertIn("bogus", err)

    def test_log_without_node_falls_back_to_ara_log(self) -> None:
        # Default `log --json` still emits the ARALog dict shape.
        ara = _make_ara(self.tmp, "fb", [_journal_node("n1")])
        rc, out, _ = _run_cli("log", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsInstance(payload, dict)
        for key in ("ara_root", "lock", "verify", "revisions", "ancestors"):
            self.assertIn(key, payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
