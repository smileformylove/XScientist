from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.research_belief import build_belief_context_projection
from xscientist.research_cli import main as research_main
from xscientist.research_closure import audit_research_closure
from xscientist.research_commands import (
    save_search_plan,
    save_search_receipt,
    save_source_snapshot,
    save_source_update,
)
from xscientist.research_dag import build_research_dag
from xscientist.research_git import ResearchGitError
from xscientist.research_semantics import build_search_receipt_payload
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
                "2026-08-10T08:00:00Z",
                "--type",
                "status_check",
                "--repo",
                str(self.repo),
            ]
        )
        still_invalidated = audit_research_closure(self.repo, level="trace")
        self.assertFalse(still_invalidated["complete"])
        self.assertIn(
            "source_invalidated",
            {item["code"] for item in still_invalidated["blockers"]},
        )
        with self.assertRaisesRegex(ResearchGitError, "provider"):
            save_source_update(
                str(self.repo),
                source_id=source["object_id"],
                status="active",
                provider="Different-Provider",
                checked_at="2026-08-10T10:00:00Z",
                update_type="reinstatement",
                notice_id="different-1",
            )
        reinstatement = self._run_json(
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
        )["object"]
        stored_reinstatement = ResearchRepository(self.repo).get(
            reinstatement["object_id"]
        )
        self.assertIn(
            update["object_id"],
            {
                row["target"]
                for row in stored_reinstatement["relations"]
                if row["type"] == "supersedes"
            },
        )
        reinstated = audit_research_closure(self.repo, level="trace")
        self.assertTrue(reinstated["complete"], reinstated["blockers"])
        objects = ResearchRepository(self.repo).objects()
        historical = build_belief_context_projection(
            objects,
            target_ids=[claim["object_id"]],
            as_of=ResearchRepository(self.repo).get(update["object_id"])["created_at"],
        )
        current = build_belief_context_projection(
            objects, target_ids=[claim["object_id"]]
        )
        self.assertTrue(
            historical["target_assessments"][0]["supporting_signals"][0]["invalidated"]
        )
        self.assertFalse(
            current["target_assessments"][0]["supporting_signals"][0]["invalidated"]
        )

    def test_literature_recording_rejects_unlocked_or_unplanned_retrieval(self) -> None:
        plan_result = save_search_plan(
            str(self.repo),
            question="Which result is admissible?",
            queries=["locked query"],
            providers=["OpenAlex"],
        )
        plan_id = plan_result["object"].object_id
        candidates = [
            {
                "title": "Selected result",
                "doi": "https://doi.org/10.1000/Selected",
                "url": "https://example.org/selected-result",
                "selected": True,
            }
        ]
        with self.assertRaisesRegex(ResearchGitError, "provider"):
            save_search_receipt(
                str(self.repo),
                plan_id=plan_id,
                provider="Crossref",
                query="locked query",
                candidates=candidates,
            )
        with self.assertRaisesRegex(ResearchGitError, "query"):
            save_search_receipt(
                str(self.repo),
                plan_id=plan_id,
                provider="OpenAlex",
                query="different query",
                candidates=candidates,
            )

        repository = ResearchRepository(self.repo)
        draft_core = {
            "question": "draft",
            "queries": ["locked query"],
            "providers": ["OpenAlex"],
            "inclusion_criteria": [],
            "exclusion_criteria": [],
        }
        draft = repository.record(
            "search_plan",
            {
                **draft_core,
                "search_plan_hash": canonical_content_hash(draft_core),
            },
            state="draft",
        )
        repository.commit(
            stage="plan", subject="record draft search plan", status="draft"
        )
        with self.assertRaisesRegex(ResearchGitError, "locked"):
            save_search_receipt(
                str(self.repo),
                plan_id=draft.object_id,
                provider="OpenAlex",
                query="locked query",
                candidates=candidates,
            )

        receipt_result = save_search_receipt(
            str(self.repo),
            plan_id=plan_id,
            provider="OpenAlex",
            query="locked query",
            candidates=candidates,
        )
        receipt = repository.get(receipt_result["object"].object_id)
        plan = repository.get(plan_id)
        self.assertEqual(
            receipt["payload"]["plan_binding"],
            {
                "object_id": plan_id,
                "content_hash": plan["content_hash"],
                "search_plan_hash": plan["payload"]["search_plan_hash"],
            },
        )
        self.assertEqual(
            receipt["payload"]["receipt_hash"],
            canonical_content_hash(
                {
                    key: value
                    for key, value in receipt["payload"].items()
                    if key != "receipt_hash"
                }
            ),
        )
        with self.assertRaisesRegex(ResearchGitError, "selected candidate"):
            save_source_snapshot(
                str(self.repo),
                receipt_id=receipt["object_id"],
                title="Selected result",
                content_hash="sha256:" + "b" * 64,
                doi="10.1000/different",
            )
        with self.assertRaisesRegex(ResearchGitError, "selected candidate"):
            save_source_snapshot(
                str(self.repo),
                receipt_id=receipt["object_id"],
                title="Selected result",
                content_hash="sha256:" + "b" * 64,
                doi="10.1000/selected",
                # A correct DOI must not conceal a second, contradictory
                # persistent identifier supplied by the selected candidate.
                url="https://example.invalid/not-the-selected-result",
            )
        source_result = save_source_snapshot(
            str(self.repo),
            receipt_id=receipt["object_id"],
            title="Selected result",
            content_hash="sha256:" + "b" * 64,
            doi="10.1000/selected",
        )
        source = repository.get(source_result["object"].object_id)
        self.assertEqual(
            source["payload"]["receipt_binding"]["receipt_hash"],
            receipt["payload"]["receipt_hash"],
        )
        self.assertEqual(
            source["payload"]["candidate_binding"]["candidate_hash"],
            canonical_content_hash(receipt["payload"]["candidates"][0]),
        )

    def test_closure_recomputes_conforming_legacy_relation_bindings(self) -> None:
        plan_result = save_search_plan(
            str(self.repo),
            question="What does the legacy search support?",
            queries=["legacy locked query"],
            providers=["OpenAlex"],
        )
        repository = ResearchRepository(self.repo)
        plan_id = plan_result["object"].object_id
        receipt_payload = build_search_receipt_payload(
            provider="OpenAlex",
            query="legacy locked query",
            candidates=[
                {
                    "title": "Legacy selected result",
                    "doi": "10.1000/legacy",
                    "selected": True,
                }
            ],
            retrieved_at="2026-08-01T00:00:00Z",
        )
        legacy_receipt = repository.record(
            "search_receipt",
            receipt_payload,
            state="completed",
            relations=[
                {
                    "type": "depends_on",
                    "target": plan_id,
                    "role": "search_plan",
                }
            ],
        )
        repository.commit(
            stage="evidence", subject="record legacy receipt", status="completed"
        )
        source_core = {
            "title": "Legacy selected result",
            "content_hash": "sha256:" + "d" * 64,
            "metadata_hash": "",
            "doi": "https://doi.org/10.1000/LEGACY",
            "pmid": "",
            "arxiv_id": "",
            "url": "",
            "license": "",
            "retraction_status": "unknown",
        }
        legacy_source = repository.record(
            "source_snapshot",
            {**source_core, "source_hash": canonical_content_hash(source_core)},
            state="completed",
            relations=[{"type": "derived_from", "target": legacy_receipt.object_id}],
        )
        repository.commit(
            stage="evidence", subject="record legacy source", status="completed"
        )
        passage = self._run_json(
            [
                "literature",
                "passage",
                legacy_source.object_id,
                "The legacy source reported a result.",
                "--locator",
                "page=2",
                "--repo",
                str(self.repo),
            ]
        )["object"]
        self._run_json(
            [
                "claim",
                "The legacy source supports this claim.",
                "--evidence",
                passage["object_id"],
                "--repo",
                str(self.repo),
            ]
        )
        audit = audit_research_closure(self.repo, level="trace")
        self.assertTrue(audit["complete"], audit["blockers"])

    def test_closure_recomputes_forged_literature_conformance(self) -> None:
        plan_result = save_search_plan(
            str(self.repo),
            question="What was locked?",
            queries=["locked query"],
            providers=["OpenAlex"],
        )
        repository = ResearchRepository(self.repo)
        plan = repository.get(plan_result["object"].object_id)
        receipt_payload = build_search_receipt_payload(
            provider="Crossref",
            query="unlocked query",
            candidates=[
                {
                    "title": "Selected result",
                    "doi": "10.1000/selected",
                    "url": "https://example.org/selected-result",
                    "selected": True,
                }
            ],
            retrieved_at="2026-08-01T00:00:00Z",
        )
        receipt_core = {
            key: value
            for key, value in receipt_payload.items()
            if key != "receipt_hash"
        }
        forged_receipt = repository.record(
            "search_receipt",
            {
                **receipt_core,
                "receipt_hash": canonical_content_hash(receipt_core),
            },
            state="completed",
            relations=[
                {
                    "type": "depends_on",
                    "target": plan["object_id"],
                    "role": "search_plan",
                }
            ],
        )
        repository.commit(
            stage="evidence", subject="forge retrieval receipt", status="completed"
        )
        stored_receipt = repository.get(forged_receipt.object_id)
        source_core = {
            "title": "Selected result",
            "content_hash": "sha256:" + "c" * 64,
            "metadata_hash": "",
            # One correct persistent identifier cannot conceal a contradictory
            # identifier that is present on both immutable records.
            "doi": "10.1000/selected",
            "pmid": "",
            "arxiv_id": "",
            "url": "https://example.invalid/different-result",
            "license": "",
            "retraction_status": "unknown",
        }
        forged_source = repository.record(
            "source_snapshot",
            {**source_core, "source_hash": canonical_content_hash(source_core)},
            state="completed",
            relations=[
                {
                    "type": "derived_from",
                    "target": stored_receipt["object_id"],
                }
            ],
        )
        repository.commit(
            stage="evidence", subject="forge source selection", status="completed"
        )
        passage = self._run_json(
            [
                "literature",
                "passage",
                forged_source.object_id,
                "A result was reported.",
                "--locator",
                "page=1",
                "--repo",
                str(self.repo),
            ]
        )["object"]
        self._run_json(
            [
                "claim",
                "The selected result supports the claim.",
                "--evidence",
                passage["object_id"],
                "--repo",
                str(self.repo),
            ]
        )
        audit = audit_research_closure(self.repo, level="trace")
        codes = {item["code"] for item in audit["blockers"]}
        self.assertIn("receipt_provider_outside_plan", codes)
        self.assertIn("receipt_query_outside_plan", codes)
        self.assertIn("source_candidate_mismatch", codes)
