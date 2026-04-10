from __future__ import annotations

import importlib
import sys
import types
import unittest

def _load_autonomous_evolution_engine():
    module_name = "ai_scientist.autonomous_evolution"
    if module_name in sys.modules:
        return sys.modules[module_name].AutonomousEvolutionEngine

    injected_modules = {}
    dependencies = {
        "ai_scientist.llm": {
            "create_client": lambda *args, **kwargs: (None, None),
            "get_response_from_llm": lambda *args, **kwargs: ("{}", None),
        },
        "ai_scientist.self_learning_knowledge_base": {
            "SelfLearningKnowledgeBase": type(
                "SelfLearningKnowledgeBase",
                (),
                {"__init__": lambda self, *args, **kwargs: None},
            ),
        },
        "ai_scientist.adaptive_learning_engine": {
            "AdaptiveLearningEngine": type(
                "AdaptiveLearningEngine",
                (),
                {"__init__": lambda self, *args, **kwargs: None},
            ),
        },
    }
    for dependency_name, members in dependencies.items():
        if dependency_name in sys.modules:
            continue
        stub_module = types.ModuleType(dependency_name)
        for member_name, value in members.items():
            setattr(stub_module, member_name, value)
        sys.modules[dependency_name] = stub_module
        injected_modules[dependency_name] = True

    try:
        module = importlib.import_module(module_name)
    finally:
        for dependency_name in injected_modules:
            sys.modules.pop(dependency_name, None)
    return module.AutonomousEvolutionEngine


AutonomousEvolutionEngine = _load_autonomous_evolution_engine()


class AutonomousEvolutionFeedbackTests(unittest.TestCase):
    def _engine_without_init(self) -> AutonomousEvolutionEngine:
        return AutonomousEvolutionEngine.__new__(AutonomousEvolutionEngine)

    def test_integrate_feedback_should_detect_polarity_conflict(self) -> None:
        engine = self._engine_without_init()
        result = engine._integrate_feedback(
            {"issues": ["Need stronger ablation"]},
            [
                {
                    "source": "external_agent",
                    "issues": ["Need stronger ablation"],
                    "strengths": ["Strong baseline comparison"],
                },
                {
                    "source": "peer_review",
                    "strengths": ["Need stronger ablation"],
                    "issues": ["Weak novelty framing"],
                },
            ],
        )
        conflicts = result.get("conflicting_points") or []
        self.assertTrue(conflicts)
        polarity = [item for item in conflicts if item.get("type") == "polarity_conflict"]
        self.assertTrue(polarity)
        self.assertTrue(
            any("need stronger ablation" == item.get("point") for item in polarity)
        )

    def test_integrate_feedback_should_detect_score_conflict(self) -> None:
        engine = self._engine_without_init()
        result = engine._integrate_feedback(
            {"scores": {"clarity": 4.8}},
            [
                {
                    "source": "external_agent",
                    "scores": {"clarity": 2.9},
                },
                {
                    "source": "peer_review",
                    "scores": {"clarity": 4.7},
                },
            ],
        )
        conflicts = result.get("conflicting_points") or []
        score_conflicts = [item for item in conflicts if item.get("type") == "score_conflict"]
        self.assertTrue(score_conflicts)
        clarity_conflict = next(
            item for item in score_conflicts if item.get("dimension") == "clarity"
        )
        self.assertGreaterEqual(float(clarity_conflict.get("score_gap") or 0), 1.5)

    def test_conflicts_should_generate_actionable_priority_actions(self) -> None:
        engine = self._engine_without_init()
        result = engine._integrate_feedback(
            {"strengths": ["Need stronger ablation"]},
            [
                {
                    "source": "external_agent",
                    "issues": ["Need stronger ablation"],
                    "scores": {"novelty": 2.2},
                },
                {
                    "source": "peer_review",
                    "strengths": ["Need stronger ablation"],
                    "scores": {"novelty": 4.5},
                },
            ],
        )
        self.assertTrue(result.get("conflict_actions"))
        priority_actions = result.get("priority_actions") or []
        self.assertTrue(priority_actions)
        self.assertTrue(
            any("resolve scoring disagreement on novelty" in action.lower() for action in priority_actions)
        )
        self.assertTrue(
            any("run focused verification" in action.lower() for action in priority_actions)
        )


if __name__ == "__main__":
    unittest.main()
