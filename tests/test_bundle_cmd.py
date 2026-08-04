"""Smoke tests for the `bundle` verb in run_ara_fork.py.

The `bundle` verb is the "git bundle" analog for an ARA — it packs the
ARA directory into a portable tarball while refusing to package
tampered/unlocked artifacts (unless --no-verify overrides). These tests
exercise the CLI end-to-end using the same driver pattern as
test_run_ara_fork_history.py.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.apps import ara as run_ara_fork

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_manifest_lock import verify_manifest_lock


def _write_journal(logs_dir: Path, nodes: list[dict]) -> None:
    stage_dir = logs_dir / "0-run"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _make_ara(tmp: Path, sub: str) -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    _write_journal(exp / "logs", [{
        "id": "n1", "step": 0, "code": "print('ok')",
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


class BundleCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_bundle_creates_tarball_containing_manifest(self) -> None:
        ara = _make_ara(self.tmp, "a")
        dest = self.tmp / "bundles" / "a.tar.gz"
        rc, out, err = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc, 0, msg=f"stderr: {err}")
        self.assertTrue(dest.exists())
        self.assertIn("[ara-bundle] wrote", out)
        with tarfile.open(dest, "r:gz") as tar:
            names = tar.getnames()
        # Contents must be nested under the ARA root's dir name.
        self.assertTrue(any(n.endswith("/manifest.json") for n in names),
                        msg=f"names: {names[:20]}")
        top_dir = ara.name
        self.assertTrue(any(n.startswith(top_dir) for n in names),
                        msg=f"top dir {top_dir} not in {names[:5]}")

    def test_bundle_extracted_ara_passes_verify_lock(self) -> None:
        ara = _make_ara(self.tmp, "b")
        dest = self.tmp / "b.tar.gz"
        rc, _, err = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc, 0, msg=err)
        extract = self.tmp / "extract"
        extract.mkdir()
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(str(extract))
        # Extracted layout: <extract>/<ara.name>/manifest.json
        extracted_root = extract / ara.name
        self.assertTrue((extracted_root / "manifest.json").exists())
        report = verify_manifest_lock(extracted_root)
        self.assertEqual(report.get("state"), "clean",
                         msg=f"verify report: {report}")

    def test_bundle_refuses_tampered_ara_without_no_verify(self) -> None:
        ara = _make_ara(self.tmp, "c")
        # Rewrite manifest.json outside the append-only API — verify_manifest_lock
        # will label this "tampered".
        (ara / "manifest.json").write_text(
            json.dumps({"schema_version": "ara.v1", "sneaky": True}),
            encoding="utf-8",
        )
        self.assertEqual(verify_manifest_lock(ara).get("state"), "tampered")
        dest = self.tmp / "c.tar.gz"
        rc, _, err = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc, 2)
        self.assertFalse(dest.exists())
        self.assertIn("tampered", err)

    def test_bundle_no_verify_allows_tampered(self) -> None:
        ara = _make_ara(self.tmp, "d")
        (ara / "manifest.json").write_text(
            json.dumps({"schema_version": "ara.v1", "sneaky": True}),
            encoding="utf-8",
        )
        dest = self.tmp / "d.tar.gz"
        rc, out, err = _run(
            "bundle", "--ara", str(ara), "--dest", str(dest), "--no-verify",
        )
        self.assertEqual(rc, 0, msg=f"stderr: {err}")
        self.assertTrue(dest.exists())
        self.assertIn("[ara-bundle] wrote", out)

    def test_bundle_refuses_existing_dest_without_force(self) -> None:
        ara = _make_ara(self.tmp, "e")
        dest = self.tmp / "e.tar.gz"
        rc, _, _ = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc, 0)
        first_size = dest.stat().st_size
        rc2, _, err = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc2, 2)
        self.assertIn("already exists", err)
        # Original tarball is untouched.
        self.assertEqual(dest.stat().st_size, first_size)

    def test_bundle_force_overwrites_existing_dest(self) -> None:
        ara = _make_ara(self.tmp, "f")
        dest = self.tmp / "f.tar.gz"
        rc, _, _ = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc, 0)
        # Corrupt the file so we can detect the overwrite.
        dest.write_bytes(b"XXXX")
        rc2, _, err = _run(
            "bundle", "--ara", str(ara), "--dest", str(dest), "--force",
        )
        self.assertEqual(rc2, 0, msg=f"stderr: {err}")
        # Overwritten bytes must be a valid gzip tar containing manifest.json.
        with tarfile.open(dest, "r:gz") as tar:
            self.assertTrue(
                any(n.endswith("/manifest.json") for n in tar.getnames())
            )

    def test_bundle_prints_size_and_count(self) -> None:
        ara = _make_ara(self.tmp, "g")
        dest = self.tmp / "g.tar.gz"
        rc, out, _ = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc, 0)
        line = out.strip().splitlines()[-1]
        pattern = re.compile(r"^\[ara-bundle\] wrote .+ \(\d+ bytes, \d+ files\)$")
        self.assertRegex(line, pattern)

    def test_bundle_refuses_dest_inside_ara(self) -> None:
        ara = _make_ara(self.tmp, "h")
        dest = ara / "self.tar.gz"
        rc, _, err = _run("bundle", "--ara", str(ara), "--dest", str(dest))
        self.assertEqual(rc, 2)
        self.assertIn("inside ARA", err)
        self.assertFalse(dest.exists())

    def test_index_profile_omits_node_payloads_and_writes_bundle_manifest(self) -> None:
        ara = _make_ara(self.tmp, "index")
        dest = self.tmp / "index.tar.gz"
        rc, _, err = _run(
            "bundle", "--ara", str(ara), "--dest", str(dest),
            "--profile", "index",
        )
        self.assertEqual(rc, 0, msg=err)
        with tarfile.open(dest, "r:gz") as tar:
            names = tar.getnames()
            bundle_name = f"{ara.name}/bundle.manifest.json"
            self.assertIn(bundle_name, names)
            payload = json.loads(tar.extractfile(bundle_name).read())
        self.assertEqual(payload["profile"], "index")
        self.assertFalse(any("/nodes/" in name for name in names))

    def test_fork_profile_refuses_missing_selected_object_unless_allowed(self) -> None:
        ara = _make_ara(self.tmp, "missing-object")
        graph_path = ara / "exploration_graph.json"
        graph = json.loads(graph_path.read_text())
        graph["nodes"][0]["llm_call_refs"] = ["sha256:" + "f" * 64]
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        dest = self.tmp / "missing.tar.gz"
        rc, _, err = _run(
            "bundle", "--ara", str(ara), "--dest", str(dest),
            "--profile", "fork", "--node", "n1",
        )
        self.assertEqual(rc, 2)
        self.assertIn("missing objects", err)
        rc, _, err = _run(
            "bundle", "--ara", str(ara), "--dest", str(dest),
            "--profile", "fork", "--node", "n1", "--allow-incomplete",
        )
        self.assertEqual(rc, 0, msg=err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
