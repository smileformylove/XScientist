"""Smoke tests for the diff/log/refs CLI verbs in run_ara_fork.py.

Verifies that the CLI wraps its engines correctly — engine correctness is
tested in test_ara_diff.py / test_ara_log.py / test_ara_refs.py.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Load run_ara_fork.py as a module. It's a top-level script, so we do this
# once via importlib to keep imports clean across tests.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_run_ara_fork_cli",
    Path(__file__).resolve().parent.parent / "run_ara_fork.py",
)
run_ara_fork = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(run_ara_fork)  # type: ignore[union-attr]

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
    exp = project / "02_experiments" / f"20260709_{sub}"
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


class CLIDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_diff_json_output_has_expected_keys(self) -> None:
        a = _make_ara(self.tmp, "a", code="print(1)")
        b = _make_ara(self.tmp, "b", code="print(2)")
        rc, out, err = _run("diff", "--ara", str(a), "--other", str(b), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("manifest", payload)
        self.assertIn("nodes_hash_changed", payload)
        self.assertEqual(len(payload["nodes_hash_changed"]), 1)
        cats = payload["nodes_hash_changed"][0]["changed_categories"]
        self.assertIn("code", cats)

    def test_diff_exit_code_on_diff(self) -> None:
        a = _make_ara(self.tmp, "a", code="print(1)")
        b = _make_ara(self.tmp, "b", code="print(2)")
        rc, _, _ = _run("diff", "--ara", str(a), "--other", str(b),
                        "--exit-code-on-diff")
        self.assertEqual(rc, 1)

    def test_diff_human_output_has_headers(self) -> None:
        a = _make_ara(self.tmp, "a", code="print(1)")
        b = _make_ara(self.tmp, "b", code="print(2)")
        rc, out, _ = _run("diff", "--ara", str(a), "--other", str(b))
        self.assertEqual(rc, 0)
        self.assertIn("## manifest", out)
        self.assertIn("## nodes", out)


class CLILogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_log_json_reports_revisions_and_ancestors(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, out, _ = _run("log", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("revisions", payload)
        self.assertIn("ancestors", payload)
        # A freshly exported root ARA has no revisions and no ancestors.
        self.assertEqual(payload["revisions"], [])
        self.assertEqual(payload["ancestors"], [])

    def test_log_human_output_shows_lock(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, out, _ = _run("log", "--ara", str(a))
        self.assertEqual(rc, 0)
        self.assertIn("rev 0 (lock)", out)


class CLIRefsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_refs_set_get_list_delete(self) -> None:
        a = _make_ara(self.tmp, "a")
        target = "sha256:" + "1" * 64

        rc, _, _ = _run("refs", "--ara", str(a), "--set", "HEAD", target)
        self.assertEqual(rc, 0)

        rc, out, _ = _run("refs", "--ara", str(a), "--get", "HEAD")
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), target)

        rc, out, _ = _run("refs", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload, [{"name": "HEAD", "target": target}])

        rc, _, _ = _run("refs", "--ara", str(a), "--delete", "HEAD")
        self.assertEqual(rc, 0)

        rc, _, err = _run("refs", "--ara", str(a), "--get", "HEAD")
        self.assertEqual(rc, 3)
        self.assertIn("not set", err)

    def test_refs_bad_name_returns_error(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, _, err = _run(
            "refs", "--ara", str(a),
            "--set", "../escape", "sha256:" + "1" * 64,
        )
        self.assertEqual(rc, 2)
        self.assertIn("refused", err)

    # --- --prefix filter (list mode) ---------------------------------

    def _seed_refs(self, ara: Path, names: list[str]) -> None:
        for i, name in enumerate(names):
            target = "sha256:" + str(i % 10) * 64
            rc, _, _ = _run("refs", "--ara", str(ara), "--set", name, target)
            self.assertEqual(rc, 0)

    def test_refs_list_prefix_filters_matching(self) -> None:
        a = _make_ara(self.tmp, "a")
        self._seed_refs(a, ["foo/a", "foo/b", "bar/c"])
        rc, out, _ = _run("refs", "--ara", str(a), "--prefix", "foo/")
        self.assertEqual(rc, 0)
        self.assertIn("foo/a", out)
        self.assertIn("foo/b", out)
        self.assertNotIn("bar/c", out)

    def test_refs_list_prefix_json_shape_preserved(self) -> None:
        a = _make_ara(self.tmp, "a")
        self._seed_refs(a, ["foo/a", "foo/b", "bar/c"])
        rc, out, _ = _run("refs", "--ara", str(a), "--json", "--prefix", "foo/")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)
        self.assertEqual({r["name"] for r in payload}, {"foo/a", "foo/b"})

    def test_refs_list_prefix_no_matches_empty(self) -> None:
        a = _make_ara(self.tmp, "a")
        self._seed_refs(a, ["bar/x", "bar/y"])
        rc, out, _ = _run("refs", "--ara", str(a), "--prefix", "nothing/")
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        rc, out, _ = _run("refs", "--ara", str(a), "--json", "--prefix", "nothing/")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [])

    def test_refs_list_prefix_empty_string_matches_all(self) -> None:
        a = _make_ara(self.tmp, "a")
        self._seed_refs(a, ["foo/a", "foo/b", "bar/c"])
        rc, out, _ = _run("refs", "--ara", str(a), "--prefix", "")
        self.assertEqual(rc, 0)
        for name in ("foo/a", "foo/b", "bar/c"):
            self.assertIn(name, out)

    def test_refs_list_prefix_trailing_slash_convention(self) -> None:
        a = _make_ara(self.tmp, "a")
        # A plain `foo` cannot coexist with `foo/bar` on the filesystem
        # (git shares the same restriction), so demonstrate the trailing-slash
        # convention with a sibling that shares the prefix without the slash.
        self._seed_refs(a, ["foobar", "foo/bar"])
        rc, out, _ = _run("refs", "--ara", str(a), "--prefix", "foo/")
        self.assertEqual(rc, 0)
        self.assertIn("foo/bar", out)
        # `foobar` starts with `foo` but not `foo/`, so it must not match.
        names = [ln.split()[0] for ln in out.splitlines() if ln.strip()]
        self.assertNotIn("foobar", names)


class CLIGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_graph_json_reports_dag_and_html_path(self) -> None:
        ara = _make_ara(self.tmp, "graph")
        rc, out, _ = _run("graph", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertTrue(payload["is_dag"])
        self.assertEqual(payload["nodes"], 1)
        self.assertEqual(payload["edges"], 0)
        self.assertTrue(payload["html"].endswith("exploration_graph.html"))

    def test_graph_write_html_regenerates_missing_visualization(self) -> None:
        ara = _make_ara(self.tmp, "graph-write")
        (ara / "exploration_graph.html").unlink()
        rc, out, _ = _run("graph", "--ara", str(ara), "--write-html")
        self.assertEqual(rc, 0)
        self.assertIn("dag:", out)
        self.assertTrue((ara / "exploration_graph.html").exists())


class CLIVerifyLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_clean_ara_exits_zero_with_state_clean(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, out, _ = _run("verify-lock", "--ara", str(a))
        self.assertEqual(rc, 0)
        self.assertIn("state:", out)
        self.assertIn("clean", out)

    def test_tampered_manifest_exits_nonzero_with_state_tampered(self) -> None:
        a = _make_ara(self.tmp, "a")
        manifest = a / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        # Mutate the manifest bytes directly, bypassing the append-only API,
        # so the on-disk hash no longer matches the lock (or any revision).
        payload["__tamper_marker__"] = "unauthorized edit"
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rc, out, _ = _run("verify-lock", "--ara", str(a))
        self.assertEqual(rc, 2)
        self.assertIn("tampered", out)

    def test_missing_lock_exits_nonzero_with_state_unlocked(self) -> None:
        a = _make_ara(self.tmp, "a")
        (a / "manifest.lock").unlink()
        rc, out, _ = _run("verify-lock", "--ara", str(a))
        self.assertEqual(rc, 3)
        self.assertIn("unlocked", out)

    def test_json_output_shape(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, out, _ = _run("verify-lock", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        for key in ("ok", "state", "base_hash", "current_hash",
                    "revision_count", "detail"):
            self.assertIn(key, payload)
        self.assertEqual(payload["state"], "clean")
        self.assertTrue(payload["ok"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
