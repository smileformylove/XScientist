"""Tests for the ARA seed / fork-continue flow.

Verifies that:
  1. A fork directory produced by `run_ara_fork.py fork` can be loaded into a
     seed manifest with provenance intact.
  2. The staged manifest is picked up via env var by `load_active_seed`.
  3. `parallel_agent._draft` short-circuits to the seed without calling the LLM
     (checked by monkey-patching the module — no actual LLM call happens).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_seed import (
    SEED_ENV_VAR,
    build_seed_manifest_from_ara_node,
    build_seed_manifest_from_fork,
    clear_active_seed_env,
    load_active_seed,
    resolve_seed_manifest_from_source,
    stage_seed_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORK_SCRIPT = REPO_ROOT / "run_ara_fork.py"


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _seed_project(tmp: Path, *, code: str, metric_value: float = 0.42) -> tuple[Path, Path, str]:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260701_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    _write_journal(
        exp / "logs",
        "0-run",
        [
            {
                "id": "seed_node",
                "step": 0,
                "code": code,
                "_term_out": ["ok\n"],
                "metric": {"value": metric_value, "maximize": True, "name": "acc", "description": ""},
                "is_buggy": False,
                "parent_id": None,
                "children": [],
            }
        ],
    )
    result = export_ara(
        project_dir=project,
        exp_dir=exp,
        idea={"Name": "seed_idea"},
        timestamp="20260701",
    )
    return project, result.root, "seed_node"


class ForkToSeedManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.code = textwrap.dedent(
            """
            import json
            result = {"name": "acc", "value": 0.42}
            print("ARA_METRIC=" + json.dumps(result))
            """
        ).strip()
        _, self.ara_root, self.node_id = _seed_project(self.tmp, code=self.code)
        self.fork_dir = self.tmp / "forked"
        completed = subprocess.run(
            [
                sys.executable, str(FORK_SCRIPT), "fork",
                "--ara", str(self.ara_root),
                "--node-id", self.node_id,
                "--dest", str(self.fork_dir),
            ],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("forked node", completed.stdout)

    def tearDown(self) -> None:
        clear_active_seed_env()
        self._tmp.cleanup()

    def test_build_seed_manifest_from_fork_carries_provenance(self) -> None:
        manifest = build_seed_manifest_from_fork(self.fork_dir)
        self.assertEqual(manifest["schema_version"], "ara.v1")
        self.assertEqual(manifest["protocol_kind"], "seed")
        self.assertEqual(manifest["code"], self.code)
        provenance = manifest["provenance"]
        self.assertEqual(provenance["parent_ara_root"], str(self.ara_root))
        self.assertEqual(provenance["parent_node_id"], self.node_id)
        self.assertIsNotNone(provenance["parent_content_hash"])
        self.assertTrue(provenance["parent_content_hash"].startswith("sha256:"))

    def test_resolve_from_fork_dir(self) -> None:
        manifest = resolve_seed_manifest_from_source(self.fork_dir)
        self.assertEqual(manifest["code"], self.code)

    def test_resolve_from_ara_root_rejects_without_node_id(self) -> None:
        with self.assertRaises(ValueError):
            resolve_seed_manifest_from_source(self.ara_root)

    def test_build_from_ara_node_direct(self) -> None:
        manifest = build_seed_manifest_from_ara_node(
            ara_root=self.ara_root, node_id=self.node_id
        )
        self.assertEqual(manifest["code"], self.code)
        self.assertEqual(manifest["provenance"]["parent_node_id"], self.node_id)


class StageAndLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp, code="print('seed')")
        self.orig_env = os.environ.get(SEED_ENV_VAR)

    def tearDown(self) -> None:
        clear_active_seed_env()
        if self.orig_env is not None:
            os.environ[SEED_ENV_VAR] = self.orig_env
        self._tmp.cleanup()

    def test_stage_then_load_roundtrip(self) -> None:
        manifest = build_seed_manifest_from_ara_node(
            ara_root=self.ara_root, node_id=self.node_id
        )
        seed_path = stage_seed_manifest(manifest, workspace_dir=self.tmp / "seed_ws")
        self.assertTrue(seed_path.exists())
        # Emulate the pipeline setting the env var before spawning BFTS.
        os.environ[SEED_ENV_VAR] = str(seed_path)
        loaded = load_active_seed()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["code"], "print('seed')")

    def test_load_returns_none_when_env_unset(self) -> None:
        clear_active_seed_env()
        self.assertIsNone(load_active_seed())

    def test_load_returns_none_on_missing_file(self) -> None:
        os.environ[SEED_ENV_VAR] = "/tmp/definitely-does-not-exist-ara-seed.json"
        self.assertIsNone(load_active_seed())

    def test_load_returns_none_when_code_missing(self) -> None:
        seed_path = self.tmp / "seed.json"
        seed_path.write_text(json.dumps({"schema_version": "ara.v1"}), encoding="utf-8")
        os.environ[SEED_ENV_VAR] = str(seed_path)
        self.assertIsNone(load_active_seed())

    def test_consume_once_semantics(self) -> None:
        """After a seed is loaded once, subsequent siblings must NOT re-consume it."""
        manifest = build_seed_manifest_from_ara_node(
            ara_root=self.ara_root, node_id=self.node_id
        )
        seed_path = stage_seed_manifest(manifest, workspace_dir=self.tmp / "ws2")
        os.environ[SEED_ENV_VAR] = str(seed_path)
        first = load_active_seed()
        self.assertIsNotNone(first)
        # The marker should now exist and load_active_seed should refuse it.
        marker = seed_path.with_suffix(seed_path.suffix + ".consumed")
        self.assertTrue(marker.exists())
        self.assertIsNone(load_active_seed())

    def test_idea_binding_gates_load(self) -> None:
        """A seed bound to idea A should not fire when idea B calls load_active_seed."""
        manifest = build_seed_manifest_from_ara_node(
            ara_root=self.ara_root, node_id=self.node_id,
            applies_to_idea_name="idea_A",
        )
        seed_path = stage_seed_manifest(manifest, workspace_dir=self.tmp / "ws3")
        os.environ[SEED_ENV_VAR] = str(seed_path)
        # Mismatched idea: refused, no marker written.
        self.assertIsNone(load_active_seed(current_idea_name="idea_B"))
        marker = seed_path.with_suffix(seed_path.suffix + ".consumed")
        self.assertFalse(marker.exists())
        # Matching idea: accepted, marker appears.
        got = load_active_seed(current_idea_name="idea_A")
        self.assertIsNotNone(got)
        self.assertEqual(got["applies_to_idea_name"], "idea_A")
        self.assertTrue(marker.exists())

    def test_unbound_seed_still_loads_when_idea_name_provided(self) -> None:
        """A seed without `applies_to_idea_name` should not care about the current idea."""
        manifest = build_seed_manifest_from_ara_node(
            ara_root=self.ara_root, node_id=self.node_id
        )
        seed_path = stage_seed_manifest(manifest, workspace_dir=self.tmp / "ws4")
        os.environ[SEED_ENV_VAR] = str(seed_path)
        self.assertIsNotNone(load_active_seed(current_idea_name="any-idea"))


class DraftShortCircuitTest(unittest.TestCase):
    """Prove `_draft` honours the seed without invoking the LLM.

    We import `parallel_agent` and stub `plan_and_code_query` — if the seed
    short-circuit works, that stub is never called.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp, code="print('via seed')")
        manifest = build_seed_manifest_from_ara_node(
            ara_root=self.ara_root, node_id=self.node_id
        )
        self.seed_path = stage_seed_manifest(manifest, workspace_dir=self.tmp / "ws")
        os.environ[SEED_ENV_VAR] = str(self.seed_path)

    def tearDown(self) -> None:
        clear_active_seed_env()
        self._tmp.cleanup()

    def test_draft_bypasses_llm(self) -> None:
        # Import parallel_agent lazily so heavy deps only load on demand.
        import types

        from ai_scientist.treesearch import parallel_agent  # type: ignore

        stub = types.SimpleNamespace(
            calls=0,
            plan_and_code_query=None,
        )

        def _fake_plan_and_code(self, prompt):  # pragma: no cover - would only
            stub.calls += 1                     # execute if seed short-circuit fails
            return "should not run", "should not run"

        # Build a minimal object with the two methods `_draft` needs.
        class Dummy:
            _draft = parallel_agent.MinimalAgent._draft
            plan_and_code_query = _fake_plan_and_code
            cfg = types.SimpleNamespace(agent=types.SimpleNamespace(data_preview=False))
            task_desc = "irrelevant"
            memory_summary = ""
            _prompt_resp_fmt = {}
            _prompt_impl_guideline = {}
            _prompt_environment = {}
            evaluation_metrics = "acc"
            data_preview = ""

        node = Dummy._draft(Dummy())  # type: ignore[arg-type]
        self.assertEqual(node.code, "print('via seed')")
        self.assertEqual(stub.calls, 0, msg="LLM was called even though seed was present")


if __name__ == "__main__":
    unittest.main()
