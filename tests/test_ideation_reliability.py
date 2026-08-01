from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.tools.semantic_scholar import balanced_rank_papers


class _FakeSearchTool:
    def use_tool(self, **_kwargs):
        return "1: Relevant prior work. Author. Venue, 2025."


class IdeationReliabilityTests(unittest.TestCase):
    def test_balanced_literature_ranking_does_not_reduce_to_citation_count(
        self,
    ) -> None:
        ranked = balanced_rank_papers(
            [
                {"paperId": "recent", "year": 2026, "citationCount": 2},
                {"paperId": "old", "year": 2012, "citationCount": 5000},
            ],
            current_year=2026,
        )
        self.assertEqual(ranked[0]["paperId"], "recent")
        self.assertIn("discovery_score", ranked[0])

    def test_idea_finalization_is_rejected_until_search_succeeds(self) -> None:
        from ai_scientist import perform_ideation_temp_free as module

        idea = {
            "Name": "Evidence First",
            "Title": "Evidence First Discovery",
            "Short Hypothesis": "A grounded mechanism improves discovery.",
            "Mechanism": "Contradiction-guided experiment selection.",
            "Generation Operator": "contradiction",
            "Falsifiers": ["No improvement over random experiment selection."],
            "Related Work": "Relevant prior work.",
            "Abstract": "A testable proposal.",
            "Experiments": "Compare on dataset: demo with metric: accuracy.",
            "Risk Factors and Limitations": "Limited benchmark coverage.",
        }
        responses = [
            "ACTION:\nFinalizeIdea\nARGUMENTS:\n" + json.dumps({"idea": idea}),
            "ACTION:\nSearchSemanticScholar\nARGUMENTS:\n"
            + json.dumps({"query": "contradiction guided scientific discovery"}),
            "ACTION:\nFinalizeIdea\nARGUMENTS:\n" + json.dumps({"idea": idea}),
        ]
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "ideas.json"
            with (
                mock.patch.object(
                    module,
                    "get_response_from_llm",
                    side_effect=[(item, []) for item in responses],
                ),
                mock.patch.dict(
                    module.tools_dict,
                    {"SearchSemanticScholar": _FakeSearchTool()},
                ),
            ):
                ideas = module.generate_temp_free_idea(
                    str(output),
                    client=object(),
                    model="demo-model",
                    workshop_description="Reliable autonomous research",
                    max_num_generations=1,
                    num_reflections=3,
                    reload_ideas=False,
                )

        self.assertEqual(len(ideas), 1)
        metadata = ideas[0]["Literature Search"]
        self.assertEqual(metadata["successful_search_count"], 1)
        self.assertEqual(
            metadata["queries"], ["contradiction guided scientific discovery"]
        )


if __name__ == "__main__":
    unittest.main()
