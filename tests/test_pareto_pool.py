from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.pareto_pool import (
    SCORE_KEYS,
    add_candidate,
    compute_candidate_key,
    compute_pareto_front,
    describe_merge_pair,
    evict_dominated,
    load_pareto_pool,
    maybe_build_merge_advisory,
    maybe_select_seed_path,
    merge_enabled,
    pareto_pool_enabled,
    select_merge_pair,
    select_seed_for_next_round,
)
from ai_scientist.utils.pipeline_contracts import initialize_pipeline_contracts


def _full_scores(**overrides) -> dict[str, float]:
    base = {key: 6.0 for key in SCORE_KEYS}
    base.update(overrides)
    return base


def _write_latex(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class CandidateKeyTests(unittest.TestCase):
    def test_normalization_stable_across_comment_and_whitespace_edits(self) -> None:
        base = "\\section{Intro}\nHello world.\n"
        with_comment = "\\section{Intro}\n% reviewer note\nHello world.   \n\n\n"
        self.assertEqual(
            compute_candidate_key(base),
            compute_candidate_key(with_comment),
        )

    def test_key_changes_with_real_content_diff(self) -> None:
        a = compute_candidate_key("\\section{Intro}\nHello.")
        b = compute_candidate_key("\\section{Intro}\nHello world.")
        self.assertNotEqual(a, b)


class ParetoFrontTests(unittest.TestCase):
    def test_strict_domination_excludes_inferior_candidate(self) -> None:
        candidates = [
            {"key": "a", "scores": _full_scores(Originality=8, Clarity=8)},
            {"key": "b", "scores": _full_scores(Originality=7, Clarity=7)},
        ]
        self.assertEqual(set(compute_pareto_front(candidates)), {"a"})

    def test_trade_off_candidates_both_on_front(self) -> None:
        candidates = [
            {"key": "a", "scores": _full_scores(Originality=9, Clarity=5)},
            {"key": "b", "scores": _full_scores(Originality=5, Clarity=9)},
        ]
        self.assertEqual(set(compute_pareto_front(candidates)), {"a", "b"})

    def test_ties_do_not_dominate(self) -> None:
        candidates = [
            {"key": "a", "scores": _full_scores()},
            {"key": "b", "scores": _full_scores()},
        ]
        self.assertEqual(set(compute_pareto_front(candidates)), {"a", "b"})


class AddEvictTests(unittest.TestCase):
    def test_add_imputes_missing_dims_and_admits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            latex = _write_latex(project_root / "latex" / "template.tex", "\\section{Intro}\nA.\n")

            outcome = add_candidate(
                project_root,
                latex_path=latex,
                scores={"Originality": 7.0, "Clarity": 5.0, "Quality": 6.0},
                round_index=0,
            )
            self.assertEqual(outcome["status"], "added")

            pool = load_pareto_pool(project_root)
            self.assertEqual(len(pool["candidates"]), 1)
            cand = pool["candidates"][0]
            self.assertEqual(set(cand["scores"].keys()), set(SCORE_KEYS))
            self.assertGreater(len(cand["imputed_dims"]), 0)
            self.assertIn(cand["key"], pool["front_keys"])

    def test_low_signal_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            latex = _write_latex(project_root / "latex" / "template.tex", "\\section{Intro}\nA.\n")

            outcome = add_candidate(
                project_root,
                latex_path=latex,
                scores={"Originality": 7.0},
                round_index=0,
            )
            self.assertEqual(outcome["status"], "skipped_low_signal")
            pool = load_pareto_pool(project_root)
            self.assertEqual(pool["candidates"], [])

    def test_eviction_preserves_front_archives_dominated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            # Two trade-off front members + 5 strictly dominated ones.
            front_a = _write_latex(project_root / "scratch" / "a.tex", "\\section{}\nA")
            front_b = _write_latex(project_root / "scratch" / "b.tex", "\\section{}\nB")
            add_candidate(project_root, latex_path=front_a, scores=_full_scores(Originality=9, Clarity=4), round_index=0)
            add_candidate(project_root, latex_path=front_b, scores=_full_scores(Originality=4, Clarity=9), round_index=0)
            for idx in range(5):
                latex = _write_latex(
                    project_root / "scratch" / f"dom_{idx}.tex",
                    f"\\section{{}}\ncontent {idx}",
                )
                add_candidate(
                    project_root,
                    latex_path=latex,
                    scores=_full_scores(Originality=3, Clarity=3),
                    round_index=idx + 1,
                )

            pool_before = load_pareto_pool(project_root)
            self.assertEqual(len(pool_before["candidates"]), 7)
            front_keys_before = set(pool_before["front_keys"])
            self.assertEqual(len(front_keys_before), 2)

            evicted = evict_dominated(project_root, max_size=3)
            self.assertGreater(evicted, 0)
            pool_after = load_pareto_pool(project_root)
            self.assertLessEqual(len(pool_after["candidates"]), 3)
            front_keys_after = set(pool_after["front_keys"])
            # Front members must survive.
            self.assertTrue(front_keys_before.issubset({c["key"] for c in pool_after["candidates"]}))
            self.assertTrue(front_keys_before.issubset(front_keys_after))

            archived = list((project_root / "pareto_pool" / "archived").glob("*.tex"))
            self.assertGreater(len(archived), 0)


class SeedSelectionTests(unittest.TestCase):
    def test_select_seed_prefers_weak_dim_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            strong = _write_latex(project_root / "scratch" / "strong.tex", "\\section{}\nstrong")
            mixed = _write_latex(project_root / "scratch" / "mixed.tex", "\\section{}\nmixed")
            # Build a genuine trade-off: strong wins Clarity, mixed wins Originality.
            # strong has no weak dims (<7); mixed has one (Clarity=6).
            add_candidate(
                project_root,
                latex_path=strong,
                scores=_full_scores(Originality=9, Clarity=9, Quality=8, Significance=8,
                                    Soundness=8, Presentation=8, Contribution=8),
                round_index=0,
            )
            add_candidate(
                project_root,
                latex_path=mixed,
                scores=_full_scores(Originality=10, Clarity=6, Quality=8, Significance=8,
                                    Soundness=8, Presentation=8, Contribution=8),
                round_index=0,
            )

            pool = load_pareto_pool(project_root)
            self.assertEqual(len(pool["front_keys"]), 2)

            seed = select_seed_for_next_round(project_root)
            self.assertIsNotNone(seed)
            assert seed is not None  # mypy
            self.assertIn("latex_path", seed)
            self.assertTrue(seed["latex_path"].endswith(".tex"))
            mixed_key = compute_candidate_key(mixed.read_text(encoding="utf-8"))
            self.assertEqual(seed["key"], mixed_key)


class MaybeSelectSeedPathTests(unittest.TestCase):
    def test_returns_none_when_pool_flag_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            latex = _write_latex(project_root / "scratch" / "a.tex", "\\section{}\nA")
            add_candidate(project_root, latex_path=latex, scores=_full_scores(), round_index=0)
            os.environ.pop("AI_SCIENTIST_PARETO_POOL", None)
            self.assertFalse(pareto_pool_enabled())
            self.assertIsNone(maybe_select_seed_path(project_root))

    def test_returns_path_when_flag_on_and_front_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            latex = _write_latex(project_root / "scratch" / "a.tex", "\\section{}\nA")
            add_candidate(project_root, latex_path=latex, scores=_full_scores(), round_index=0)
            with patch.dict(os.environ, {"AI_SCIENTIST_PARETO_POOL": "1"}):
                self.assertTrue(pareto_pool_enabled())
                path = maybe_select_seed_path(project_root)
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.endswith(".tex"))
            self.assertTrue(Path(path).exists())

    def test_returns_none_when_pool_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            with patch.dict(os.environ, {"AI_SCIENTIST_PARETO_POOL": "1"}):
                self.assertIsNone(maybe_select_seed_path(project_root))


class MergePairTests(unittest.TestCase):
    def _populate(self, project_root: Path) -> None:
        latex_a = _write_latex(project_root / "scratch" / "a.tex", "\\section{a}\nA")
        latex_b = _write_latex(project_root / "scratch" / "b.tex", "\\section{b}\nB")
        latex_c = _write_latex(project_root / "scratch" / "c.tex", "\\section{c}\nC")
        # A strong on Clarity/Presentation
        add_candidate(
            project_root,
            latex_path=latex_a,
            scores=_full_scores(Clarity=9.0, Presentation=8.5, Soundness=6.0, Quality=6.0),
            round_index=0,
        )
        # B strong on Soundness/Quality (complementary to A)
        add_candidate(
            project_root,
            latex_path=latex_b,
            scores=_full_scores(Clarity=6.0, Presentation=6.0, Soundness=9.0, Quality=8.5),
            round_index=1,
        )
        # C dominated — same as A but weaker everywhere
        add_candidate(
            project_root,
            latex_path=latex_c,
            scores=_full_scores(Clarity=5.0, Presentation=5.0, Soundness=5.0, Quality=5.0),
            round_index=2,
        )

    def test_select_merge_pair_picks_most_complementary_front_members(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            self._populate(project_root)
            pair = select_merge_pair(project_root)
            self.assertIsNotNone(pair)
            assert pair is not None
            a, b = pair
            keys = {a.get("key"), b.get("key")}
            # Both members must be on the front, never the dominated candidate C.
            pool = load_pareto_pool(project_root)
            front_keys = set(pool.get("front_keys") or [])
            self.assertTrue(keys.issubset(front_keys))

    def test_select_merge_pair_returns_none_when_front_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            latex = _write_latex(project_root / "scratch" / "a.tex", "\\section{}\nA")
            add_candidate(project_root, latex_path=latex, scores=_full_scores(), round_index=0)
            self.assertIsNone(select_merge_pair(project_root))

    def test_select_merge_pair_threshold_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            # Two front members with nearly-identical scores — complementarity too low.
            latex_a = _write_latex(project_root / "scratch" / "a.tex", "\\section{a}\nA")
            latex_b = _write_latex(project_root / "scratch" / "b.tex", "\\section{b}\nB")
            add_candidate(
                project_root,
                latex_path=latex_a,
                scores=_full_scores(Clarity=7.0, Soundness=6.0),
                round_index=0,
            )
            add_candidate(
                project_root,
                latex_path=latex_b,
                scores=_full_scores(Clarity=6.0, Soundness=7.0),
                round_index=1,
            )
            # Threshold above the actual gap (=2.0).
            self.assertIsNone(select_merge_pair(project_root, min_complementarity=5.0))

    def test_describe_merge_pair_has_sorted_complementary_dims_and_strengths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            self._populate(project_root)
            pair = select_merge_pair(project_root)
            assert pair is not None
            desc = describe_merge_pair(pair)
            self.assertIn("candidate_a", desc)
            self.assertIn("candidate_b", desc)
            self.assertIn("complementary_dims", desc)
            dims = desc["complementary_dims"]
            self.assertGreater(len(dims), 0)
            gaps = [row["gap"] for row in dims]
            self.assertEqual(gaps, sorted(gaps, reverse=True))
            self.assertEqual(set(desc["candidate_a"]["scores"].keys()), set(SCORE_KEYS))
            self.assertEqual(set(desc["candidate_b"]["scores"].keys()), set(SCORE_KEYS))

    def test_maybe_build_merge_advisory_respects_env_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            self._populate(project_root)
            os.environ.pop("AI_SCIENTIST_PARETO_MERGE", None)
            self.assertFalse(merge_enabled())
            self.assertIsNone(maybe_build_merge_advisory(project_root))
            with patch.dict(os.environ, {"AI_SCIENTIST_PARETO_MERGE": "1"}):
                self.assertTrue(merge_enabled())
                advisory = maybe_build_merge_advisory(project_root)
            self.assertIsNotNone(advisory)
            assert advisory is not None
            self.assertIn("candidate_a", advisory)
            self.assertIn("candidate_b", advisory)


if __name__ == "__main__":
    unittest.main()
