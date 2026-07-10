"""Smoke tests for the `describe` verb in run_ara_fork.py.

The engine-level pieces (ara_log, verify_manifest_lock, hash_manifest)
are covered elsewhere. This file exercises the CLI glue: field
selection, top-metric direction, null-safe handling, and ancestry
summary aggregation.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_run_ara_fork_cli_describe",
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


def _node(nid: str, *, value: float | None = 0.5, maximize: bool = True,
          is_buggy: bool = False, code: str | None = None) -> dict:
    metric = None if value is None else {
        "value": value, "maximize": maximize, "name": "acc",
    }
    return {
        "id": nid, "step": 0,
        "code": code if code is not None else f"print({nid!r})",
        "_term_out": [],
        "metric": metric,
        "is_buggy": is_buggy, "parent_id": None, "children": [],
    }


def _make_ara(tmp: Path, sub: str, nodes: list[dict],
              provenance: dict | None = None) -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    _write_journal(exp / "logs", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    result = export_ara(
        project_dir=project, exp_dir=exp, idea={"Name": sub},
        provenance=provenance,
    )
    return Path(result.root)


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class DescribeCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_describe_human_output_contains_key_fields(self) -> None:
        a = _make_ara(self.tmp, "human", [
            _node("nA", value=0.7, maximize=True, is_buggy=False),
            _node("nB", value=0.9, maximize=True, is_buggy=True),
        ])
        rc, out, _ = _run("describe", "--ara", str(a))
        self.assertEqual(rc, 0)
        self.assertIn("Idea", out)
        self.assertIn("Nodes:", out)
        self.assertIn("buggy: 1", out)
        self.assertIn("Top metric", out)
        # The non-buggy node id must show up in the top-metric row.
        self.assertIn("nA", out)

    def test_describe_json_output_shape(self) -> None:
        a = _make_ara(self.tmp, "json", [_node("nA")])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        for key in ("ara_root", "idea", "counts", "top_metric_node",
                    "lock", "verify_state", "ancestors"):
            self.assertIn(key, payload)
        self.assertIn("nodes", payload["counts"])
        self.assertIn("buggy", payload["counts"])
        self.assertIn("state", payload["lock"])

    def test_describe_top_metric_picks_highest_when_maximize(self) -> None:
        a = _make_ara(self.tmp, "hi", [
            _node("nLo", value=0.5, maximize=True),
            _node("nHi", value=0.9, maximize=True),
        ])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["top_metric_node"]["id"], "nHi")

    def test_describe_top_metric_picks_lowest_when_minimize(self) -> None:
        a = _make_ara(self.tmp, "lo", [
            _node("nLo", value=0.1, maximize=False),
            _node("nHi", value=0.9, maximize=False),
        ])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["top_metric_node"]["id"], "nLo")

    def test_describe_no_scored_nodes_reports_null(self) -> None:
        a = _make_ara(self.tmp, "empty", [
            _node("nBad", value=0.9, is_buggy=True),
            _node("nNone", value=None),
        ])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsNone(payload["top_metric_node"])

    def test_describe_ancestors_summary_when_parent_reachable(self) -> None:
        parent = _make_ara(self.tmp, "parent", [_node("np", value=0.5)])
        parent_graph = json.loads((parent / "exploration_graph.json").read_text())
        parent_node_hash = parent_graph["nodes"][0]["content_hash"]

        child = _make_ara(
            self.tmp, "child", [_node("nc", value=0.6)],
            provenance={
                "parent_ara_root": str(parent),
                "parent_node_id": "np",
                "parent_content_hash": parent_node_hash,
            },
        )
        rc, out, _ = _run("describe", "--ara", str(child), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["ancestors"]["count"], 1)
        self.assertTrue(payload["ancestors"]["all_reachable"])


class VerifyLockAllTests(unittest.TestCase):
    """`verify-lock --all --project <path>` sweeps <project>/ara/.

    Aggregate rc rule: tampered > unlocked > ok. Empty projects are ok.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _project_with(self, name: str, count: int) -> Path:
        project = self.tmp / name
        for i in range(count):
            sub = f"{name}_ara{i}"
            exp = project / "02_experiments" / f"20260710_{sub}"
            _write_journal(exp / "logs", [_node(f"n{i}")])
            (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
            export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub})
        return project

    @staticmethod
    def _tamper(ara: Path) -> None:
        m = ara / "manifest.json"
        p = json.loads(m.read_text(encoding="utf-8"))
        p["__tamper__"] = "edit"
        m.write_text(json.dumps(p, indent=2), encoding="utf-8")

    def test_verify_lock_all_walks_multiple_aras(self) -> None:
        project = self._project_with("multi", 3)
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 0)
        self.assertEqual(sum(1 for line in out.splitlines() if line.strip()), 4)
        self.assertEqual(out.count("clean"), 3)

    def test_verify_lock_all_reports_tampered(self) -> None:
        project = self._project_with("tamp", 2)
        target = sorted((project / "ara").iterdir())[0]
        self._tamper(target)
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 2)
        self.assertIn("tampered", out)
        self.assertIn(target.name, out)

    def test_verify_lock_all_reports_unlocked(self) -> None:
        project = self._project_with("unlk", 2)
        (sorted((project / "ara").iterdir())[0] / "manifest.lock").unlink()
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 3)
        self.assertIn("unlocked", out)

    def test_verify_lock_all_tampered_beats_unlocked_in_rc(self) -> None:
        project = self._project_with("mix", 3)
        aras = sorted((project / "ara").iterdir())
        self._tamper(aras[1])
        (aras[2] / "manifest.lock").unlink()
        rc, _, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 2)

    def test_verify_lock_all_empty_project_returns_rc_zero(self) -> None:
        project = self.tmp / "empty_project"
        project.mkdir()
        rc, out, err = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 0)
        self.assertIn("no ARAs", err)
        self.assertEqual(out, "")

    def test_verify_lock_all_json_shape(self) -> None:
        project = self._project_with("json", 3)
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 3)
        for entry in payload:
            for key in ("ara_root", "state", "revision_count",
                        "manifest_hash", "detail"):
                self.assertIn(key, entry)
            self.assertEqual(entry["state"], "clean")

    def test_verify_lock_all_json_empty_project_emits_empty_array(self) -> None:
        """`--json` on an empty project must emit `[]` on stdout so CI pipes
        (`... --json | jq`) don't choke on empty input. The human note stays
        on stderr — same invariant every other verb honors."""
        project = self.tmp / "empty_json"
        project.mkdir()
        rc, out, err = _run("verify-lock", "--all", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "[]")
        self.assertEqual(json.loads(out), [])
        self.assertIn("no ARAs", err)

    def test_verify_lock_all_json_includes_manifest_hash_for_clean(self) -> None:
        """Every clean entry must carry a resolvable sha256 manifest_hash so
        downstream tools can compare hashes across ARAs without an extra
        per-ARA verify-lock call."""
        project = self._project_with("mhash", 3)
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 3)
        for entry in payload:
            self.assertEqual(entry["state"], "clean")
            self.assertIsInstance(entry["manifest_hash"], str)
            self.assertTrue(entry["manifest_hash"].startswith("sha256:"))

    def test_verify_lock_all_json_manifest_hash_none_for_unlocked(self) -> None:
        """Unlocked ARAs have no recorded base hash — the field must be
        explicitly null, not missing, so JSON consumers can distinguish
        'no lock' from 'field absent because of a shape regression'."""
        project = self._project_with("unlkjson", 2)
        (sorted((project / "ara").iterdir())[0] / "manifest.lock").unlink()
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project), "--json")
        self.assertEqual(rc, 3)
        payload = json.loads(out)
        unlocked = [e for e in payload if e["state"] == "unlocked"]
        self.assertEqual(len(unlocked), 1)
        self.assertIn("manifest_hash", unlocked[0])
        self.assertIsNone(unlocked[0]["manifest_hash"])

    def test_verify_lock_all_json_matches_single_ara_verify(self) -> None:
        """The sweep entry's manifest_hash must equal the single-mode
        verify-lock's for the same ARA — same underlying report, same
        field."""
        project = self._project_with("match", 2)
        one = sorted((project / "ara").iterdir())[0]

        rc_single, out_single, _ = _run("verify-lock", "--ara", str(one), "--json")
        self.assertEqual(rc_single, 0)
        single = json.loads(out_single)

        rc_all, out_all, _ = _run("verify-lock", "--all", "--project", str(project), "--json")
        self.assertEqual(rc_all, 0)
        sweep = json.loads(out_all)
        # sweep resolves symlinks (e.g. /var → /private/var on macOS); compare
        # resolved paths so the assertion holds regardless of platform.
        one_resolved = one.resolve()
        match = [e for e in sweep if Path(e["ara_root"]).resolve() == one_resolved]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0]["manifest_hash"], single["base_hash"])

    def test_verify_lock_all_and_ara_mutually_exclusive(self) -> None:
        project = self._project_with("mx", 1)
        ara = next(iter((project / "ara").iterdir()))
        with self.assertRaises(SystemExit) as ctx:
            _run("verify-lock", "--ara", str(ara),
                 "--all", "--project", str(project))
        self.assertNotEqual(ctx.exception.code, 0)


class ListCLITests(unittest.TestCase):
    """`list --project <path>` enumerates every ARA under <project>/ara/.

    Mirrors `verify-lock --all --project`'s sweep pattern but emits a
    richer per-ARA row (idea + counts + seed presence + lock state).
    Never fails the sweep — broken ARAs surface as state=error rows.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _project_with_aras(self, name: str, specs: list[tuple[str, list[dict]]]) -> Path:
        project = self.tmp / name
        for sub, nodes in specs:
            exp = project / "02_experiments" / f"20260710_{sub}"
            _write_journal(exp / "logs", nodes)
            (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
            export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub})
        return project

    def test_list_multiple_aras_human_table(self) -> None:
        project = self._project_with_aras("multi_human", [
            ("idea_a", [_node("nA", value=0.5)]),
            ("idea_b", [_node("nB", value=0.7, is_buggy=True)]),
        ])
        rc, out, _ = _run("list", "--project", str(project))
        self.assertEqual(rc, 0)
        self.assertIn("IDEA", out)
        self.assertIn("idea_a", out)
        self.assertIn("idea_b", out)
        # Header + 2 data rows.
        self.assertEqual(sum(1 for line in out.splitlines() if line.strip()), 3)

    def test_list_json_shape(self) -> None:
        project = self._project_with_aras("json_shape", [
            ("only", [_node("nA", value=0.5)]),
        ])
        rc, out, _ = _run("list", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        entry = payload[0]
        for key in ("ara_root", "idea", "nodes", "buggy_nodes",
                    "seed_present", "state", "manifest_hash", "path"):
            self.assertIn(key, entry)
        self.assertEqual(entry["idea"], "only")
        self.assertEqual(entry["state"], "clean")
        self.assertFalse(entry["seed_present"])

    def test_list_empty_project_json_emits_empty_array(self) -> None:
        project = self.tmp / "empty_json"
        project.mkdir()
        rc, out, err = _run("list", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "[]")
        self.assertEqual(json.loads(out), [])
        self.assertIn("no ARAs", err)

    def test_list_empty_project_human_stderr_only(self) -> None:
        project = self.tmp / "empty_human"
        project.mkdir()
        rc, out, err = _run("list", "--project", str(project))
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("no ARAs", err)

    def test_list_survives_broken_ara(self) -> None:
        project = self._project_with_aras("broken", [
            ("good", [_node("nG", value=0.5)]),
            ("bad",  [_node("nB", value=0.5)]),
        ])
        # Corrupt one manifest.json to invalid JSON.
        bad_dir = sorted((project / "ara").iterdir())[0]
        (bad_dir / "manifest.json").write_text("{not-json", encoding="utf-8")

        rc, out, _ = _run("list", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 2)
        broken = next(e for e in payload if e["path"] == bad_dir.name)
        self.assertEqual(broken["state"], "error")
        self.assertEqual(broken["idea"], "?")
        # Human mode still emits a row for the broken ARA.
        rc2, out2, _ = _run("list", "--project", str(project))
        self.assertEqual(rc2, 0)
        self.assertIn(bad_dir.name, out2)
        self.assertIn("error", out2)

    def test_list_reports_seed_present(self) -> None:
        parent_project = self.tmp / "seedproj"
        parent_exp = parent_project / "02_experiments" / "20260710_parent"
        _write_journal(parent_exp / "logs", [_node("np", value=0.5)])
        (parent_exp / "idea.json").write_text(json.dumps({"Name": "parent"}), encoding="utf-8")
        parent_res = export_ara(project_dir=parent_project, exp_dir=parent_exp,
                                idea={"Name": "parent"})
        parent_ara = Path(parent_res.root)

        # Fork the parent's node into <parent_project>/ara/forked_ara so the
        # forked ARA lives inside the sweep target.
        dest = parent_project / "ara" / "forked_ara"
        rc_fork, _, _ = _run("fork", "--ara", str(parent_ara),
                             "--node-id", "np", "--dest", str(dest))
        self.assertEqual(rc_fork, 0)

        rc, out, _ = _run("list", "--project", str(parent_project), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        fork_entry = next(e for e in payload if e["path"] == "forked_ara")
        self.assertTrue(fork_entry["seed_present"])
        parent_entry = next(e for e in payload if e["path"] != "forked_ara")
        self.assertFalse(parent_entry["seed_present"])

    def test_list_skips_subdirs_without_manifest(self) -> None:
        """Non-ARA sibling dirs (_scratch/, __pycache__/, .hidden/) must not
        surface as synthetic entries — matches _verify_lock_all's filter."""
        real = _make_ara(self.tmp, "solo", [_node("nA", value=0.5)])
        ara_base = real.parent
        for junk in ("_scratch", "__pycache__", ".hidden"):
            (ara_base / junk).mkdir()

        rc, out, _ = _run("list", "--project", str(self.tmp / "solo"), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["path"], real.name)
        for junk in ("_scratch", "__pycache__", ".hidden"):
            self.assertNotIn(junk, out)

    def test_list_survives_string_provenance(self) -> None:
        """Legacy/corrupt manifest with `provenance` as a string must not
        crash the sweep — the ARA still surfaces in the output."""
        real = _make_ara(self.tmp, "strprov", [_node("nA", value=0.5)])
        manifest_path = real / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["provenance"] = "parent_ara=/x"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc, out, _ = _run("list", "--project", str(self.tmp / "strprov"), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["path"], real.name)
        self.assertFalse(payload[0]["seed_present"])

    def test_list_survives_string_idea(self) -> None:
        """Legacy/corrupt manifest with `idea` as a bare string must not
        crash the sweep — the ARA still surfaces in the output."""
        real = _make_ara(self.tmp, "stridea", [_node("nA", value=0.5)])
        manifest_path = real / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["idea"] = "just-a-string"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc, out, _ = _run("list", "--project", str(self.tmp / "stridea"), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["path"], real.name)
        self.assertEqual(payload[0]["idea"], "?")

    def test_list_survives_non_dict_counts(self) -> None:
        """Legacy/corrupt manifest with `counts` as a bare scalar must not
        crash the sweep — the ARA still surfaces in the output."""
        real = _make_ara(self.tmp, "badcounts", [_node("nA", value=0.5)])
        manifest_path = real / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["counts"] = 5  # int, not a dict — truthy so `or {}` fallback fails
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc, out, _ = _run(
            "list", "--project", str(self.tmp / "badcounts"), "--json"
        )
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["path"], real.name)
        # nodes/buggy_nodes fall back to None (dict.get default) when counts
        # is coerced back to {}.
        self.assertIsNone(payload[0]["nodes"])
        self.assertIsNone(payload[0]["buggy_nodes"])


class DescribeNonDictCountsTests(unittest.TestCase):
    """Regression: cmd_describe must degrade gracefully when a legacy
    manifest carries `counts` as a non-dict scalar (mirrors cmd_list)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_describe_survives_non_dict_counts(self) -> None:
        a = _make_ara(self.tmp, "descbadcounts", [_node("nA", value=0.5)])
        manifest_path = a / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["counts"] = "bad"  # string, not a dict
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("counts", payload)
        # Falls back to whatever the graph enumerates — 1 node, 0 buggy.
        self.assertEqual(payload["counts"]["nodes"], 1)

    def test_describe_survives_string_provenance(self) -> None:
        """Legacy/corrupt manifest with `provenance` as a bare string must
        not crash cmd_describe — mirrors iter-16's cmd_list guard."""
        a = _make_ara(self.tmp, "descstrprov", [_node("nA", value=0.5)])
        manifest_path = a / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["provenance"] = "not-a-dict"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        # provenance-derived seed payload defaults to null when nothing valid.
        self.assertIsNone(payload.get("seed"))

    def test_describe_survives_string_idea(self) -> None:
        """Legacy/corrupt manifest with `idea` as a bare string must not
        crash cmd_describe."""
        a = _make_ara(self.tmp, "descstridea", [_node("nA", value=0.5)])
        manifest_path = a / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["idea"] = "just-a-string"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        idea = payload.get("idea") or {}
        self.assertIsNone(idea.get("name"))
        self.assertIsNone(idea.get("title"))

    def test_describe_survives_non_numeric_counts_values(self) -> None:
        """Legacy/corrupt manifest with non-numeric `counts.nodes` /
        `counts.edges` must not crash on int() coercion. Falls back to
        the graph's own enumeration (or 0 when the graph is empty)."""
        a = _make_ara(self.tmp, "descbadints", [_node("nA", value=0.5)])
        manifest_path = a / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["counts"] = {"nodes": "not_a_number", "edges": "also_bad"}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        counts = payload["counts"]
        self.assertIsInstance(counts["nodes"], int)
        self.assertIsInstance(counts["edges"], int)
        # Graph has 1 node and no edges — _safe_int falls back to the
        # graph enumeration when the manifest value is unparseable.
        self.assertEqual(counts["nodes"], 1)
        self.assertEqual(counts["edges"], 0)


class EnvCLITests(unittest.TestCase):
    """`env --ara <path>` dumps a JSON summary of the reproducibility fingerprint.

    Focused view alongside `describe` (whole-ARA), `show` (one-node),
    and `hash-check` (per-node integrity) — this one only reads env/*.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_env_all_three_present(self) -> None:
        a = _make_ara(self.tmp, "envfull", [_node("nA", value=0.5)])
        # export_ara already writes model_fingerprint.json; add the other two.
        (a / "env" / "bfts_config.json").write_text(
            json.dumps({"agent": {"steps": 3}}), encoding="utf-8"
        )
        (a / "env" / "requirements.freeze").write_text("", encoding="utf-8")

        rc, out, _ = _run("env", "--ara", str(a))
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertTrue(payload["bfts_config"]["present"])
        self.assertTrue(payload["model_fingerprint"]["present"])
        self.assertTrue(payload["requirements_freeze"]["present"])
        for key in ("bfts_config", "model_fingerprint", "requirements_freeze"):
            self.assertTrue(payload[key]["content_hash"].startswith("sha256:"))
        # bfts_config.json wins over the missing .yaml.
        self.assertEqual(payload["bfts_config"]["path"], "env/bfts_config.json")

    def test_env_missing_dir(self) -> None:
        a = _make_ara(self.tmp, "envmissing", [_node("nA", value=0.5)])
        shutil.rmtree(a / "env")
        rc, out, _ = _run("env", "--ara", str(a))
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsNone(payload["env_dir"])
        self.assertFalse(payload["bfts_config"]["present"])
        self.assertFalse(payload["model_fingerprint"]["present"])
        self.assertFalse(payload["requirements_freeze"]["present"])
        self.assertEqual(payload["other_env_files"], [])

    def test_env_model_fingerprint_lifts_writing_profile(self) -> None:
        # Rebuild fingerprint with the structured writing_profile slot so we
        # exercise the {name, content_hash} lift path independent of whether
        # export_ara received a writing_profile arg in this test env.
        a = _make_ara(self.tmp, "envwp", [_node("nA", value=0.5)])
        fp_path = a / "env" / "model_fingerprint.json"
        fp_path.write_text(json.dumps({
            "fingerprint": "abc123",
            "spec": {
                "models": {"writeup": "gpt-4"},
                "writing_profile": {
                    "name": "profile_a",
                    "content_hash": "sha256:deadbeef",
                },
                "ara_schema_version": "1.0",
            },
        }), encoding="utf-8")
        rc, out, _ = _run("env", "--ara", str(a))
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["model_fingerprint"]["models"], {"writeup": "gpt-4"})
        wp = payload["model_fingerprint"]["writing_profile"]
        self.assertEqual(wp["name"], "profile_a")
        self.assertTrue(wp["content_hash"].startswith("sha256:"))

    def test_env_other_files_listed(self) -> None:
        a = _make_ara(self.tmp, "envother", [_node("nA", value=0.5)])
        (a / "env" / "custom.json").write_text("{}", encoding="utf-8")
        rc, out, _ = _run("env", "--ara", str(a))
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("env/custom.json", payload["other_env_files"])
        # model_fingerprint.json is a known file — must NOT leak into other.
        self.assertNotIn("env/model_fingerprint.json", payload["other_env_files"])

    def test_env_requirements_freeze_line_count_when_small(self) -> None:
        a = _make_ara(self.tmp, "envfreeze", [_node("nA", value=0.5)])
        (a / "env" / "requirements.freeze").write_text(
            "numpy==1.26\nscipy==1.11\ntorch==2.2\n", encoding="utf-8"
        )
        rc, out, _ = _run("env", "--ara", str(a))
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["requirements_freeze"]["line_count"], 3)


    def test_env_requirements_freeze_non_utf8_does_not_crash(self) -> None:
        """Non-UTF-8 requirements.freeze must not break the docstring's
        rc=0 guarantee — presence stays true, line_count is null, and a
        note flags the decode failure."""
        a = _make_ara(self.tmp, "envfreezebin", [_node("nA", value=0.5)])
        (a / "env" / "requirements.freeze").write_bytes(
            b"\xff\xfe\x00\x00binary garbage\n"
        )
        rc, out, _ = _run("env", "--ara", str(a))
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertTrue(payload["requirements_freeze"]["present"])
        self.assertIsNone(payload["requirements_freeze"]["line_count"])
        self.assertIn("UTF-8", payload["requirements_freeze"]["note"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
