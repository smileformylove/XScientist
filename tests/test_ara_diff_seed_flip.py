"""Direct-probe seed-flip detection in the ARA diff engine.

Before this change, a seed toggle only surfaced in ``changed_categories``
via the ``content_hash_inputs`` symmetric-difference fallback. That path
depends on producers emitting ``content_hash_inputs`` correctly; a
producer that stamps ``content_hash`` (with seed bound in) but forgets
``content_hash_inputs`` would report ``changed_categories=["unknown"]``.

These tests pin the direct probe: ``_which_categories_flipped`` reads
``is_seed_node`` off the graph entry so seed flips get a real label.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_diff import diff_ara

from tests.test_ara_diff import _make_node, _project


class SeedFlipDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _pair(self, node_a: dict, node_b: dict) -> tuple[Path, Path]:
        pa, ea = _project(self.tmp, sub="a", nodes=[node_a])
        pb, eb = _project(self.tmp, sub="b", nodes=[node_b])
        ra = export_ara(project_dir=pa, exp_dir=ea, idea={"Name": "a"})
        rb = export_ara(project_dir=pb, exp_dir=eb, idea={"Name": "a"})
        return Path(ra.root), Path(rb.root)

    def test_seed_flip_reports_seed_category(self) -> None:
        """Only is_seed_node toggles → changed_categories == ['seed']."""
        a, b = self._pair(
            _make_node("n1", is_seed_node=False),
            _make_node("n1", is_seed_node=True),
        )
        d = diff_ara(a, b)
        self.assertEqual([n.id for n in d.nodes_hash_changed], ["n1"])
        cats = d.nodes_hash_changed[0].changed_categories
        self.assertIn("seed", cats)
        self.assertNotIn("unknown", cats)
        self.assertNotIn("code", cats)
        self.assertNotIn("metric", cats)
        self.assertNotIn("llm_calls", cats)

    def test_seed_and_code_flip_reports_both(self) -> None:
        """Code AND seed differ → both categories surface."""
        a, b = self._pair(
            _make_node("n1", code="print('a')", is_seed_node=False),
            _make_node("n1", code="print('b')", is_seed_node=True),
        )
        d = diff_ara(a, b)
        self.assertEqual(len(d.nodes_hash_changed), 1)
        cats = d.nodes_hash_changed[0].changed_categories
        self.assertIn("code", cats)
        self.assertIn("seed", cats)
        self.assertNotIn("unknown", cats)

    def test_no_seed_toggle_no_seed_category(self) -> None:
        """Identical seed flag on both sides → 'seed' must NOT appear."""
        a, b = self._pair(
            _make_node("n1", code="print('a')", is_seed_node=True),
            _make_node("n1", code="print('b')", is_seed_node=True),
        )
        d = diff_ara(a, b)
        self.assertEqual(len(d.nodes_hash_changed), 1)
        cats = d.nodes_hash_changed[0].changed_categories
        self.assertIn("code", cats)
        self.assertNotIn("seed", cats)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
