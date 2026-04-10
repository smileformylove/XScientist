from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class LearningEngineRegressionTests(unittest.TestCase):
    def _import_learning_modules(self):
        try:
            from ai_scientist.adaptive_learning_engine import AdaptiveLearningEngine
            from ai_scientist.self_learning_knowledge_base import (
                PatternAnalyzer,
                SelfLearningKnowledgeBase,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"learning modules unavailable in lightweight env: {exc}")
        return AdaptiveLearningEngine, PatternAnalyzer, SelfLearningKnowledgeBase

    def test_save_recommendation_should_write_latest_snapshot_and_history(self) -> None:
        (
            AdaptiveLearningEngine,
            _PatternAnalyzer,
            SelfLearningKnowledgeBase,
        ) = self._import_learning_modules()
        with tempfile.TemporaryDirectory() as td:
            kb = SelfLearningKnowledgeBase(research_dir=td)
            engine = AdaptiveLearningEngine(knowledge_base=kb, research_dir=td)
            recommendation = {
                "paper_type": "neurips",
                "success_probability": 0.72,
                "writing_strategy": {"emphasis_sections": ["Method"]},
            }
            idea = {"Name": "Learning based optimizer", "Field": "ML"}

            engine._save_recommendation(recommendation, idea)

            latest_path = kb.knowledge_dir / "latest_recommendation.json"
            history_path = kb.knowledge_dir / "recommendation_history.jsonl"
            self.assertTrue(latest_path.exists())
            self.assertTrue(history_path.exists())

            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(latest_payload.get("idea_name"), "Learning based optimizer")
            self.assertIn("recommendation", latest_payload)

            history_lines = [
                line
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(history_lines), 1)

    def test_pattern_analyzer_should_extract_writing_patterns(self) -> None:
        (
            _AdaptiveLearningEngine,
            PatternAnalyzer,
            SelfLearningKnowledgeBase,
        ) = self._import_learning_modules()
        with tempfile.TemporaryDirectory() as td:
            kb = SelfLearningKnowledgeBase(research_dir=td)
            kb.success_patterns = [
                {
                    "paper_type": "neurips",
                    "final_scores": {"clarity": 4.4, "rigor": 4.2},
                    "reviews": [
                        {
                            "review": {
                                "scores": {"clarity": 4.5, "innovation": 4.1},
                                "main_issues": ["Need stronger ablation"],
                            }
                        }
                    ],
                    "improvements": [
                        {
                            "strategy": "evidence_first_rewrite",
                            "issue_type": "claim_support",
                            "improvement_score": 0.8,
                        }
                    ],
                },
                {
                    "paper_type": "neurips",
                    "final_scores": {"clarity": 4.1, "rigor": 4.3},
                    "reviews": [
                        {
                            "review": {
                                "scores": {"clarity": 4.2, "rigor": 4.4},
                                "main_issues": ["Need stronger baseline comparison"],
                            }
                        }
                    ],
                    "improvements": [
                        {
                            "strategy": "evidence_first_rewrite",
                            "issue_type": "experiments",
                            "improvement_score": 0.6,
                        }
                    ],
                },
            ]

            analyzer = PatternAnalyzer(kb)
            result = analyzer._analyze_writing_patterns()

            self.assertIn("avg_review_rounds", result)
            self.assertIn("top_review_dimensions", result)
            self.assertIn("top_improvement_strategies", result)
            self.assertIn("average_scores_by_dimension", result)
            self.assertGreater(result["avg_review_rounds"], 0)
            self.assertTrue(result["top_improvement_strategies"])

    def test_recommend_strategy_should_include_self_evolution_guidance(self) -> None:
        (
            AdaptiveLearningEngine,
            _PatternAnalyzer,
            SelfLearningKnowledgeBase,
        ) = self._import_learning_modules()
        with tempfile.TemporaryDirectory() as td:
            kb = SelfLearningKnowledgeBase(research_dir=td)
            playbook_path = kb.knowledge_dir / "self_evolution_playbook.json"
            playbook_path.write_text(
                json.dumps(
                    {
                        "project_count": 2,
                        "status_counts": {"ready": 1, "needs_attention": 1},
                        "top_recurring_risks": [
                            {"risk": "verification_path_gap", "count": 2}
                        ],
                        "top_agentic_defaults": [
                            {
                                "stage": "experiment",
                                "action": "Front-load stronger baseline evidence.",
                                "count": 2,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            engine = AdaptiveLearningEngine(knowledge_base=kb, research_dir=td)
            recommendation = engine.recommend_strategy(
                {"Name": "Repair-aware generator", "Field": "ML"},
                "neurips",
            )

            self.assertEqual(
                recommendation["self_evolution_guidance"]["project_count"],
                2,
            )
            self.assertIn(
                "experiment: Front-load stronger baseline evidence.",
                recommendation["improvement_strategy"]["agentic_defaults"],
            )
            self.assertIn(
                "verification_path_gap",
                recommendation["common_pitfalls"],
            )


if __name__ == "__main__":
    unittest.main()
