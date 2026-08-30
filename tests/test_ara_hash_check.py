"""Smoke tests for the `hash-check` verb in run_ara_fork.py.

The verb recomputes each node's content_hash from disk using the same
binding rule ``ara_artifact._export_nodes_from_journal`` used at write
time, then reports drift against the stored hash. These tests pin exit
codes, state classification, and the JSON shape.
"""

from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.apps import ara as run_ara_fork
from ai_scientist.protocol import validate_ara

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.authority_attempts import (
    begin_authority_attempt,
    persist_authority_object,
    record_authority_attempt_result,
)
from ai_scientist.treesearch.agent_manager import (
    _validate_restored_authority_ledger,
)
from ai_scientist.treesearch.journal import Journal, Node


def _write_journal(logs_dir: Path, nodes: list[dict]) -> None:
    stage_dir = logs_dir / "0-run"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _make_ara(tmp: Path, sub: str, nodes: list[dict]) -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    _write_journal(exp / "logs", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    result = export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub})
    return Path(result.root)


def _make_ara_in(project: Path, idea_name: str, nodes: list[dict]) -> Path:
    """Build an ARA under an existing project directory (multi-ARA sweep fixture)."""
    exp = project / "02_experiments" / f"20260710_{idea_name}"
    _write_journal(exp / "logs", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": idea_name}), encoding="utf-8")
    result = export_ara(project_dir=project, exp_dir=exp, idea={"Name": idea_name})
    return Path(result.root)


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def _default_node(
    nid: str, *, code: str = "print('ok')", metric_val: float = 0.5
) -> dict:
    return {
        "id": nid,
        "step": 0,
        "code": code,
        "_term_out": [],
        "metric": {"value": metric_val, "maximize": True, "name": "acc"},
        "is_buggy": False,
        "parent_id": None,
        "children": [],
    }


def _create_authority_attempt(
    log_dir: Path,
    *,
    parent_node_id: str | None,
    status: str = "accepted",
) -> tuple[str, str]:
    spec_hash, spec_ref = persist_authority_object(
        log_dir,
        category="implementation-spec",
        payload={"schema": "test.authority-spec.v1", "objective": "bounded"},
    )
    attempt_id = begin_authority_attempt(
        log_dir,
        spec_hash=spec_hash,
        spec_ref=spec_ref,
        parent_node_id=parent_node_id,
        role="execution",
        model="test-executor",
        task_kind="implementation",
    )
    assert attempt_id is not None
    terminal_hash = record_authority_attempt_result(
        log_dir,
        attempt_id,
        status=status,
        result_payload={"produced": parent_node_id} if status != "failed" else None,
        error_type="SyntheticFailure" if status == "failed" else None,
    )
    assert terminal_hash is not None
    return attempt_id, terminal_hash


def _make_authority_ara(tmp: Path, *, metric_val: float = 0.5) -> tuple[Path, Path]:
    project = tmp / "authority_project"
    exp = project / "02_experiments" / "20260710_authority"
    log_dir = exp / "logs" / "0-run"
    node = _default_node("n1", metric_val=metric_val)
    attempt_id, terminal_hash = _create_authority_attempt(
        log_dir,
        parent_node_id="n1",
    )
    node["authority_attempt_ids"] = [attempt_id]
    node["authority_attempt_terminal_hashes"] = {attempt_id: terminal_hash}
    # A failed attempt creates no Node, but remains part of the top-level audit.
    _create_authority_attempt(log_dir, parent_node_id="discarded", status="failed")
    _write_journal(exp / "logs", [node])
    result = export_ara(
        project_dir=project,
        exp_dir=exp,
        idea={"Name": "authority"},
    )
    return Path(result.root), log_dir


class HashCheckCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_hash_check_clean_ara_returns_zero(self) -> None:
        ara = _make_ara(
            self.tmp,
            "clean",
            [
                _default_node("n1"),
                _default_node("n2", metric_val=0.7),
            ],
        )
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual({e["state"] for e in payload}, {"clean"})
        for e in payload:
            self.assertEqual(e["stored_hash"], e["computed_hash"])

    def test_hash_check_detects_code_drift(self) -> None:
        ara = _make_ara(self.tmp, "code_drift", [_default_node("n1")])
        # Mutate the code on disk — stored hash won't match anymore.
        (ara / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "drift")
        self.assertIn("code", entry.get("notes", "").lower())

    def test_hash_check_detects_metric_drift(self) -> None:
        ara = _make_ara(self.tmp, "metric_drift", [_default_node("n1")])
        # Edit metrics.json to change the value — stored hash no longer matches.
        mp = ara / "nodes" / "n1" / "metrics.json"
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["metric"] = {"value": 0.999, "maximize": True, "name": "acc"}
        mp.write_text(json.dumps(m), encoding="utf-8")
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "drift")
        self.assertIn("metric", entry.get("notes", "").lower())

    def test_hash_check_detects_locked_spec_drift(self) -> None:
        spec = {
            "schema": "xscientist.locked-experiment-spec.v1",
            "primary_metric": "accuracy",
            "objective": "bounded",
        }
        encoded = json.dumps(
            spec,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        node = _default_node("n1")
        node["implementation_spec"] = spec
        node["implementation_spec_hash"] = (
            "sha256:" + hashlib.sha256(encoded).hexdigest()
        )
        ara = _make_ara(self.tmp, "spec_drift", [node])
        spec_path = ara / "nodes" / "n1" / "implementation_spec.json"
        tampered = json.loads(spec_path.read_text(encoding="utf-8"))
        tampered["objective"] = "tampered"
        spec_path.write_text(json.dumps(tampered), encoding="utf-8")

        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "drift")

    def test_hash_check_missing_code_returns_rc2(self) -> None:
        ara = _make_ara(self.tmp, "no_code", [_default_node("n1")])
        (ara / "nodes" / "n1" / "code.py").unlink()
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 2, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "missing_code")
        self.assertIsNone(entry["computed_hash"])

    def test_hash_check_empty_code_node_reports_clean(self) -> None:
        # A journal node with empty code — export skips writing code.py but
        # still stamps a content_hash computed with code="". hash-check must
        # recognise the empty-code recompute matches the stored hash and
        # report clean, not a false-positive missing_code.
        ara = _make_ara(self.tmp, "empty_code", [_default_node("n1", code="")])
        self.assertFalse((ara / "nodes" / "n1" / "code.py").exists())
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "clean")
        self.assertEqual(entry["stored_hash"], entry["computed_hash"])

    def test_hash_check_drift_beats_missing_code(self) -> None:
        ara = _make_ara(
            self.tmp,
            "drift_wins",
            [
                _default_node("n1"),
                _default_node("n2"),
            ],
        )
        # n1 gets code drift; n2 gets its code deleted.
        (ara / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        (ara / "nodes" / "n2" / "code.py").unlink()
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)  # drift wins over missing_code
        payload = json.loads(out)
        by_id = {e["node_id"]: e for e in payload}
        self.assertEqual(by_id["n1"]["state"], "drift")
        self.assertEqual(by_id["n2"]["state"], "missing_code")

    def test_hash_check_json_shape(self) -> None:
        ara = _make_ara(self.tmp, "shape", [_default_node("n1")])
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        [entry] = payload
        for key in ("node_id", "state", "stored_hash", "computed_hash"):
            self.assertIn(key, entry)

    def test_hash_check_unhashed_node_is_not_a_failure(self) -> None:
        ara = _make_ara(self.tmp, "legacy", [_default_node("n1")])
        # Simulate a legacy graph entry by stripping the stored hash from
        # exploration_graph.json (metrics.json still has it, but the CLI
        # reads content_hash from the graph entry — matching cmd_show).
        gp = ara / "exploration_graph.json"
        g = json.loads(gp.read_text(encoding="utf-8"))
        for n in g["nodes"]:
            n.pop("content_hash", None)
        gp.write_text(json.dumps(g), encoding="utf-8")
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        [entry] = payload
        self.assertEqual(entry["state"], "unhashed")
        self.assertIsNone(entry["stored_hash"])
        self.assertIsNotNone(entry["computed_hash"])

    def test_hash_check_human_output_prints_table(self) -> None:
        ara = _make_ara(self.tmp, "table", [_default_node("n1")])
        rc, out, err = _run("hash-check", "--ara", str(ara))
        self.assertEqual(rc, 0, msg=err)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        self.assertTrue(lines[0].startswith("NODE"))
        # Truncated hash form uses the ellipsis character.
        self.assertIn("n1", out)

    def test_authority_attempts_are_portable_hash_bound_and_complete(self) -> None:
        ara, _ = _make_authority_ara(self.tmp)
        self.assertTrue(validate_ara(ara).ok)
        artifact_path = ara / "authority_attempts.json"
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes)
        self.assertEqual(
            artifact_bytes,
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )
        self.assertEqual(artifact["schema"], "xscientist.ara.authority-attempts.v1")
        self.assertEqual(artifact["counts"]["attempts"], 2)
        self.assertEqual(
            {row["status"] for row in artifact["attempts"]},
            {"accepted", "failed"},
        )
        graph = json.loads((ara / "exploration_graph.json").read_text())
        [node] = graph["nodes"]
        self.assertEqual(len(node["authority_attempt_ids"]), 1)
        self.assertEqual(
            set(node["authority_attempt_terminal_hashes"]),
            set(node["authority_attempt_ids"]),
        )
        self.assertIn("authority_attempts", node["content_hash_inputs"])
        self.assertEqual(
            graph["authority_attempts"],
            json.loads((ara / "manifest.json").read_text())["references"][
                "authority_attempts"
            ],
        )

        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        entries = json.loads(out)
        self.assertEqual(
            next(row for row in entries if row["node_id"] == "@authority-attempts")[
                "state"
            ],
            "clean",
        )

    def test_hash_check_detects_bundled_authority_event_tamper(self) -> None:
        ara, _ = _make_authority_ara(self.tmp)
        artifact = json.loads((ara / "authority_attempts.json").read_text())
        ledger = ara / artifact["ledgers"][0]["path"]
        attempt_id = artifact["ledgers"][0]["attempt_ids"][0]
        event_path = ledger / "authority_attempts" / attempt_id / "1.json"
        event = json.loads(event_path.read_text())
        event["status"] = "rejected"
        event_path.write_text(json.dumps(event), encoding="utf-8")

        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        entries = json.loads(out)
        authority = next(
            row for row in entries if row["node_id"] == "@authority-attempts"
        )
        self.assertEqual(authority["state"], "drift")
        self.assertIn("invalid", authority["notes"])

    def test_hash_check_detects_missing_authority_artifact(self) -> None:
        ara, _ = _make_authority_ara(self.tmp)
        (ara / "authority_attempts.json").unlink()

        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        entries = json.loads(out)
        authority = next(
            row for row in entries if row["node_id"] == "@authority-attempts"
        )
        self.assertEqual(authority["state"], "drift")
        self.assertIn("missing", authority["notes"])

    def test_hash_check_detects_orphaned_bundled_attempt(self) -> None:
        ara, _ = _make_authority_ara(self.tmp)
        artifact = json.loads((ara / "authority_attempts.json").read_text())
        ledger = ara / artifact["ledgers"][0]["path"]
        orphan = ledger / "authority_attempts" / ("attempt-" + "f" * 32)
        orphan.mkdir(mode=0o700)

        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        entries = json.loads(out)
        authority = next(
            row for row in entries if row["node_id"] == "@authority-attempts"
        )
        self.assertEqual(authority["state"], "drift")
        self.assertIn("orphan", authority["notes"])

    def test_export_refuses_incomplete_authority_ledger(self) -> None:
        project = self.tmp / "incomplete_project"
        exp = project / "02_experiments" / "20260710_incomplete"
        log_dir = exp / "logs" / "0-run"
        spec_hash, spec_ref = persist_authority_object(
            log_dir,
            category="implementation-spec",
            payload={"schema": "test.authority-spec.v1"},
        )
        begin_authority_attempt(
            log_dir,
            spec_hash=spec_hash,
            spec_ref=spec_ref,
            parent_node_id="n1",
            role="execution",
            model="test-executor",
            task_kind="implementation",
        )
        _write_journal(exp / "logs", [_default_node("n1")])

        with self.assertRaisesRegex(ValueError, "incomplete/orphaned/invalid"):
            export_ara(
                project_dir=project,
                exp_dir=exp,
                idea={"Name": "incomplete"},
            )

    def test_authority_bound_node_hash_failure_aborts_export(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict identity hashing failed"):
            _make_authority_ara(self.tmp, metric_val=float("nan"))

    def test_checkpoint_restore_authority_validation_is_fail_closed(self) -> None:
        log_dir = self.tmp / "restore-ledger"
        attempt_id, terminal_hash = _create_authority_attempt(
            log_dir,
            parent_node_id="n1",
        )
        node = Node(
            id="n1",
            code="print('ok')",
            plan="bounded",
            authority_attempt_ids=[attempt_id],
            authority_attempt_terminal_hashes={attempt_id: terminal_hash},
        )
        journal = Journal()
        journal.append(node)
        _validate_restored_authority_ledger({"stage": journal}, log_dir=log_dir)

        event_path = log_dir / "authority_attempts" / attempt_id / "1.json"
        event = json.loads(event_path.read_text())
        event["error_type"] = "Tampered"
        event_path.write_text(json.dumps(event), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ledger"):
            _validate_restored_authority_ledger({"stage": journal}, log_dir=log_dir)


class HashCheckAllCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.project = self.tmp / "proj"
        self.project.mkdir()

    def test_hash_check_all_walks_multiple_aras(self) -> None:
        _make_ara_in(self.project, "idea_a", [_default_node("n1")])
        _make_ara_in(self.project, "idea_b", [_default_node("n1"), _default_node("n2")])
        _make_ara_in(self.project, "idea_c", [_default_node("n1")])
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(len(payload["aras"]), 3)
        self.assertTrue(all(a["state"] == "clean" for a in payload["aras"]))
        self.assertEqual(payload["totals"]["aras"], 3)
        self.assertEqual(payload["totals"]["nodes"], 4)

    def test_hash_check_all_reports_drift(self) -> None:
        _make_ara_in(self.project, "clean_idea", [_default_node("n1")])
        bad = _make_ara_in(self.project, "drift_idea", [_default_node("n1")])
        (bad / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 1, msg=err)
        payload = json.loads(out)
        by_name = {Path(a["ara_root"]).name: a for a in payload["aras"]}
        drift_key = next(k for k in by_name if "drift_idea" in k)
        clean_key = next(k for k in by_name if "clean_idea" in k)
        self.assertEqual(by_name[drift_key]["state"], "drift")
        self.assertEqual(by_name[clean_key]["state"], "clean")
        self.assertEqual(payload["totals"]["drift"], 1)

    def test_hash_check_all_reports_missing_code(self) -> None:
        _make_ara_in(self.project, "clean_idea", [_default_node("n1")])
        bad = _make_ara_in(self.project, "gone_idea", [_default_node("n1")])
        (bad / "nodes" / "n1" / "code.py").unlink()
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 2, msg=err)
        payload = json.loads(out)
        by_name = {Path(a["ara_root"]).name: a for a in payload["aras"]}
        gone_key = next(k for k in by_name if "gone_idea" in k)
        self.assertEqual(by_name[gone_key]["state"], "missing_code")
        self.assertEqual(payload["totals"]["missing_code"], 1)

    def test_hash_check_all_drift_beats_missing_code_in_rc(self) -> None:
        _make_ara_in(self.project, "clean_idea", [_default_node("n1")])
        drift = _make_ara_in(self.project, "drift_idea", [_default_node("n1")])
        (drift / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        gone = _make_ara_in(self.project, "gone_idea", [_default_node("n1")])
        (gone / "nodes" / "n1" / "code.py").unlink()
        rc, _out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 1, msg=err)  # drift wins over missing_code

    def test_hash_check_all_empty_project_returns_rc_zero(self) -> None:
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["aras"], [])
        self.assertIn("no ARAs found", err)

    def test_hash_check_all_json_shape(self) -> None:
        _make_ara_in(self.project, "idea_a", [_default_node("n1")])
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertIn("aras", payload)
        self.assertIn("totals", payload)
        self.assertIsInstance(payload["aras"], list)
        self.assertIsInstance(payload["totals"], dict)
        [entry] = payload["aras"]
        for key in ("ara_root", "nodes", "counts", "state"):
            self.assertIn(key, entry)

    def test_hash_check_all_and_ara_mutually_exclusive(self) -> None:
        ara = _make_ara_in(self.project, "idea_a", [_default_node("n1")])
        with self.assertRaises(SystemExit):
            _run(
                "hash-check",
                "--ara",
                str(ara),
                "--all",
                "--project",
                str(self.project),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
