from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from xscientist.research_cli import main as research_main
from xscientist.research_closure import audit_research_closure
from xscientist.research_dag import build_research_dag
from xscientist.research_vcs import ResearchRepository


class ResearchLiteratureJourneyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = tempfile.mkdtemp(prefix="xscientist-literature-")
        self.root = Path(self.raw)
        self.repo = self.root / "study"
        ResearchRepository.init(
            self.repo,
            name="literature-study",
            git_user_name="XScientist Tests",
            git_user_email="tests@example.invalid",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.raw)

    def _run_json(self, argv: list[str]) -> dict:
        output = StringIO()
        with redirect_stdout(output):
            code = research_main([*argv, "--json"])
        self.assertEqual(code, 0, output.getvalue())
        return json.loads(output.getvalue())

    def test_plain_language_literature_chain_is_trace_complete(self) -> None:
        plan = self._run_json(
            [
                "literature",
                "plan",
                "Does intervention X improve outcome Y?",
                "--query",
                '"intervention X" AND "outcome Y"',
                "--provider",
                "OpenAlex",
                "--include",
                "controlled studies",
                "--repo",
                str(self.repo),
            ]
        )["object"]
        candidates = self.root / "candidates.json"
        candidates.write_text(
            json.dumps(
                [
                    {
                        "title": "A controlled study",
                        "doi": "10.0000/example",
                        "score": 0.91,
                        "selected": True,
                        "raw_response": "must not persist",
                        "api_key": "must not persist",
                    }
                ]
            ),
            encoding="utf-8",
        )
        receipt = self._run_json(
            [
                "literature",
                "receipt",
                plan["object_id"],
                "--provider",
                "OpenAlex",
                "--query",
                '"intervention X" AND "outcome Y"',
                "--results",
                str(candidates),
                "--repo",
                str(self.repo),
            ]
        )["object"]
        stored_receipt = ResearchRepository(self.repo).get(receipt["object_id"])
        serialized_receipt = json.dumps(stored_receipt)
        self.assertNotIn("must not persist", serialized_receipt)
        self.assertEqual(
            stored_receipt["payload"]["profile"], "xscientist.retrieval-receipt.v2"
        )
        self.assertTrue(
            stored_receipt["payload"]["completeness"]["complete_candidate_set"]
        )
        self.assertTrue(stored_receipt["payload"]["request_hash"].startswith("sha256:"))
        self.assertTrue(
            stored_receipt["payload"]["candidate_set_hash"].startswith("sha256:")
        )
        source = self._run_json(
            [
                "literature",
                "source",
                receipt["object_id"],
                "A controlled study",
                "--content-hash",
                "sha256:" + "a" * 64,
                "--doi",
                "10.0000/example",
                "--repo",
                str(self.repo),
            ]
        )["object"]
        passage = self._run_json(
            [
                "literature",
                "passage",
                source["object_id"],
                "The intervention improved the prespecified outcome.",
                "--locator",
                "page=7;section=Results;paragraph=2",
                "--repo",
                str(self.repo),
            ]
        )["object"]
        stored_passage = ResearchRepository(self.repo).get(passage["object_id"])
        self.assertEqual(
            stored_passage["payload"]["selector"]["selectors"][0]["type"],
            "TextQuoteSelector",
        )
        self.assertEqual(
            stored_passage["payload"]["selector"]["selectors"][0]["exact"],
            "The intervention improved the prespecified outcome.",
        )
        claim = self._run_json(
            [
                "claim",
                "Intervention X improves outcome Y in adults.",
                "--evidence",
                passage["object_id"],
                "--population",
                "adults",
                "--intervention",
                "X",
                "--outcome",
                "Y",
                "--repo",
                str(self.repo),
            ]
        )["object"]
        stored_claim = ResearchRepository(self.repo).get(claim["object_id"])
        self.assertTrue(
            stored_claim["qualified_id"].startswith(
                "urn:xscientist:research-object:sha256:"
            )
        )
        self.assertEqual(
            ResearchRepository(self.repo).resolve(stored_claim["qualified_id"]),
            claim["object_id"],
        )
        self.assertEqual(
            stored_claim["payload"]["scope"]["profile"],
            "xscientist.claim-scope.v1",
        )
        self.assertTrue(stored_claim["payload"]["scope_hash"].startswith("sha256:"))
        audit = audit_research_closure(self.repo, level="trace")
        self.assertTrue(audit["complete"], audit["blockers"])
        self.assertEqual(
            audit["claims"][0]["passage_evidence_ids"], [passage["object_id"]]
        )
        dag = build_research_dag(self.repo)
        literature_kinds = {
            node["kind"]
            for node in dag["nodes"]
            if node["kind"]
            in {
                "search_plan",
                "search_receipt",
                "source_snapshot",
                "passage_evidence",
            }
        }
        self.assertEqual(
            literature_kinds,
            {
                "search_plan",
                "search_receipt",
                "source_snapshot",
                "passage_evidence",
            },
        )

        update = self._run_json(
            [
                "literature",
                "update",
                source["object_id"],
                "--status",
                "retracted",
                "--provider",
                "Crossref-Retraction-Watch",
                "--checked-at",
                "2026-08-09T12:00:00Z",
                "--type",
                "retraction",
                "--notice-id",
                "rw-123",
                "--repo",
                str(self.repo),
            ]
        )["object"]
        self.assertEqual(update["kind"], "source_update")
        invalidated = audit_research_closure(self.repo, level="trace")
        self.assertFalse(invalidated["complete"])
        self.assertIn(
            "source_invalidated",
            {item["code"] for item in invalidated["blockers"]},
        )
        self._run_json(
            [
                "literature",
                "update",
                source["object_id"],
                "--status",
                "active",
                "--provider",
                "Crossref-Retraction-Watch",
                "--checked-at",
                "2026-08-10T12:00:00Z",
                "--type",
                "reinstatement",
                "--notice-id",
                "rw-124",
                "--repo",
                str(self.repo),
            ]
        )
        reinstated = audit_research_closure(self.repo, level="trace")
        self.assertTrue(reinstated["complete"], reinstated["blockers"])
