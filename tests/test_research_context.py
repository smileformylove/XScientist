from __future__ import annotations

import copy
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.hashing import content_hash
from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_git import ResearchGitError
from xscientist.research_context import (
    _research_prompt_token_estimate,
    build_research_context_snapshot,
    record_research_context_snapshot,
    render_research_context_for_prompt,
    research_context_issues,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for Research Context tests")
class ResearchContextTests(unittest.TestCase):
    def _repository(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            name="context-study",
            question="# Context study\n",
            git_user_name="Context Test",
            git_user_email="context@example.invalid",
        )

    def _lineage(self, repository: ResearchRepository) -> dict[str, str]:
        hypothesis = repository.record(
            "hypothesis", {"statement": "H1", "falsifier": "not H1"}
        )
        plan = repository.record(
            "research_plan",
            {"summary": "test H1"},
            relations=[{"type": "depends_on", "target": hypothesis.object_id}],
        )
        failed = repository.record(
            "experiment_attempt",
            {"status": "failed", "summary": "unstable configuration"},
            state="failed",
            relations=[{"type": "depends_on", "target": plan.object_id}],
        )
        evidence = repository.record(
            "evidence",
            {
                "result": "H1 failed under the unstable configuration",
                "measurement_hash": "sha256:" + "a" * 64,
            },
            state="completed",
            relations=[
                {"type": "derived_from", "target": failed.object_id},
                {"type": "refutes", "target": hypothesis.object_id},
            ],
        )
        gate = repository.record(
            "gate_decision",
            {"decision": "hold", "claim_promotion_allowed": False},
            state="rejected",
            relations=[{"type": "evaluates", "target": evidence.object_id}],
            actor={"actor_id": "legacy-gate", "authority": "deterministic_gate"},
        )
        return {
            "hypothesis": hypothesis.object_id,
            "failed": failed.object_id,
            "evidence": evidence.object_id,
            "gate": gate.object_id,
        }

    def test_context_preserves_negative_memory_and_rejected_alternatives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            ids = self._lineage(repository)

            payload = build_research_context_snapshot(
                repository,
                target_ids=[ids["hypothesis"]],
                decision_kind="next_experiment",
                selected="revise_method",
                options_considered=[
                    {"option": "revise_method", "rejected_because": ""},
                    {
                        "option": "repeat_failed_configuration",
                        "rejected_because": "the retained attempt already failed",
                    },
                ],
                rationale=["use the failure as negative knowledge"],
                memory_refs=["sha256:" + "b" * 64],
                budget_tokens=400,
            )

            self.assertTrue(payload["complete"])
            self.assertIn(ids["failed"], payload["negative_knowledge_ids"])
            self.assertIn(ids["gate"], payload["prior_decision_ids"])
            self.assertIn(ids["evidence"], payload["negative_knowledge_ids"])
            self.assertTrue(payload["budget"]["hard_closure_preserved"])
            belief = payload["belief_context"]
            self.assertTrue(belief["complete"])
            self.assertEqual(
                belief["target_assessments"][0]["belief_state"], "challenged"
            )
            self.assertFalse(belief["scientific_promotion_allowed"])
            receipt = payload["retrieval_receipt"]
            self.assertEqual(
                receipt["profile"], "xscientist.context-retrieval-receipt.v4"
            )
            self.assertTrue(receipt["complete_candidate_set"])
            self.assertEqual(
                {item["object_id"] for item in receipt["candidate_set"]},
                set(payload["source_object_ids"]),
            )
            self.assertFalse(research_context_issues(payload))

            # Versioned receipts remain readable after the v4 belief-context
            # projection is introduced; legacy v3 snapshots did not contain it.
            legacy = copy.deepcopy(payload)
            legacy.pop("belief_context")
            legacy.pop("belief_context_hash")
            legacy["selection_policy"]["version"] = "3.0"
            legacy["selection_policy_hash"] = content_hash(legacy["selection_policy"])
            receipt = legacy["retrieval_receipt"]
            receipt["profile"] = "xscientist.context-retrieval-receipt.v3"
            receipt["algorithm"]["version"] = "3.0"
            receipt["algorithm_hash"] = content_hash(receipt["algorithm"])
            receipt_core = {
                key: value for key, value in receipt.items() if key != "receipt_hash"
            }
            receipt["receipt_hash"] = content_hash(receipt_core)
            legacy_estimate = _research_prompt_token_estimate(legacy)
            legacy["working_set"]["estimated_tokens"] = legacy_estimate
            working_core = {
                key: value
                for key, value in legacy["working_set"].items()
                if key != "working_set_hash"
            }
            legacy["working_set"]["working_set_hash"] = content_hash(working_core)
            legacy["budget"]["working_set_estimated_tokens"] = legacy_estimate
            identity = {
                key: value for key, value in legacy.items() if key != "context_hash"
            }
            legacy["context_hash"] = content_hash(identity)
            self.assertFalse(research_context_issues(legacy))

            unsupported = copy.deepcopy(legacy)
            unsupported["selection_policy"]["version"] = "999.0"
            unsupported["selection_policy_hash"] = content_hash(
                unsupported["selection_policy"]
            )
            unsupported_identity = {
                key: value
                for key, value in unsupported.items()
                if key != "context_hash"
            }
            unsupported["context_hash"] = content_hash(unsupported_identity)
            self.assertIn(
                "context selection policy version is unsupported",
                research_context_issues(unsupported),
            )

    def test_long_history_keeps_current_frontier_and_archives_superseded_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            hypothesis = repository.record(
                "hypothesis",
                {
                    "statement": "calibration improves transfer accuracy",
                    "falsifier": "accuracy does not improve",
                },
            )
            old_refutation = repository.record(
                "evidence",
                {
                    "result": "old pilot appeared to refute calibration",
                    "measurement_hash": "sha256:" + "1" * 64,
                },
                state="completed",
                relations=[
                    {"type": "refutes", "target": hypothesis.object_id},
                ],
            )
            for index in range(48):
                repository.record(
                    "evidence",
                    {
                        "result": f"archival observation {index}",
                        "measurement_hash": "sha256:" + f"{index + 2:064x}"[-64:],
                    },
                    state="completed",
                    relations=[
                        {"type": "supports", "target": hypothesis.object_id},
                    ],
                )
            draft_superseder = repository.record(
                "evidence",
                {
                    "result": "unfinished replication",
                    "measurement_hash": "sha256:" + "e" * 64,
                },
                state="draft",
                relations=[
                    {"type": "supersedes", "target": old_refutation.object_id},
                    {"type": "refutes", "target": hypothesis.object_id},
                ],
                actor={
                    "actor_id": "draft-reviewer",
                    "authority": "independent_evaluator",
                },
            )
            draft_review = repository.record(
                "review",
                {"summary": "unfinished evaluator review"},
                state="draft",
                relations=[
                    {"type": "evaluates", "target": hypothesis.object_id},
                ],
                actor={
                    "actor_id": "draft-gate",
                    "authority": "deterministic_gate",
                },
            )
            before_completion = build_research_context_snapshot(
                repository,
                target_ids=[hypothesis.object_id],
                budget_tokens=400,
            )
            self.assertIn(
                old_refutation.object_id,
                before_completion["negative_knowledge_ids"],
            )
            self.assertNotIn(
                old_refutation.object_id,
                before_completion["archived_history_ids"],
            )
            self.assertNotIn(
                draft_superseder.object_id,
                before_completion["negative_knowledge_ids"],
            )
            draft_source = next(
                item
                for item in before_completion["source_objects"]
                if item["object_id"] == draft_superseder.object_id
            )
            self.assertEqual(draft_source["role"], "lineage")
            draft_review_source = next(
                item
                for item in before_completion["source_objects"]
                if item["object_id"] == draft_review.object_id
            )
            self.assertEqual(draft_review_source["role"], "lineage")
            self.assertNotIn(
                draft_review.object_id,
                before_completion["working_set"]["required_view_ids"],
            )
            draft_challenge = next(
                item
                for item in before_completion["belief_context"]["target_assessments"][
                    0
                ]["challenging_signals"]
                if item["object_id"] == draft_superseder.object_id
            )
            self.assertFalse(draft_challenge["active"])
            current = repository.record(
                "evidence",
                {
                    "result": "current preregistered replication supports calibration",
                    "measurement_hash": "sha256:" + "f" * 64,
                },
                state="verified",
                relations=[
                    {"type": "supports", "target": hypothesis.object_id},
                    {"type": "supersedes", "target": old_refutation.object_id},
                ],
                actor={"actor_id": "replicator", "authority": "independent_evaluator"},
            )

            payload = build_research_context_snapshot(
                repository,
                target_ids=[hypothesis.object_id],
                rationale=["choose the next transfer falsification"],
                budget_tokens=400,
            )
            visible_ids = {item["object_id"] for item in payload["source_views"]}

            self.assertTrue(payload["complete"])
            self.assertTrue(payload["working_set"]["decision_usable"])
            self.assertIn(current.object_id, visible_ids)
            self.assertIn(
                old_refutation.object_id,
                payload["archived_history_ids"],
            )
            self.assertNotIn(
                old_refutation.object_id,
                payload["negative_knowledge_ids"],
            )
            self.assertLessEqual(
                payload["budget"]["working_set_estimated_tokens"],
                400,
            )
            self.assertNotIn(
                "source_objects", render_research_context_for_prompt(payload)
            )
            self.assertIn(
                "candidate_belief_guard",
                render_research_context_for_prompt(payload),
            )
            self.assertFalse(research_context_issues(payload))

    def test_context_closes_over_passage_support_claim_evidence_and_retraction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            hypothesis = repository.record(
                "hypothesis", {"statement": "H", "falsifier": "not H"}
            )
            source_core = {
                "title": "Source A",
                "doi": "10.1000/source-a",
                "content_hash": "sha256:" + "1" * 64,
            }
            source = repository.record(
                "source_snapshot",
                {
                    **source_core,
                    "source_hash": canonical_content_hash(source_core),
                },
                state="completed",
            )
            passage_core = {
                "source_id": source.object_id,
                "locator": "p. 1",
                "quote": "Observed support for H.",
                "quote_hash": canonical_content_hash("Observed support for H."),
            }
            passage = repository.record(
                "passage_evidence",
                {
                    **passage_core,
                    "passage_hash": canonical_content_hash(passage_core),
                },
                relations=[
                    {"type": "quotes", "target": source.object_id},
                    {
                        "type": "qualified_supports",
                        "target": hypothesis.object_id,
                    },
                ],
                state="completed",
            )
            draft_passage_core = {
                "source_id": source.object_id,
                "locator": "p. 2",
                "quote": "Unfinished observation for H.",
                "quote_hash": canonical_content_hash("Unfinished observation for H."),
            }
            draft_passage = repository.record(
                "passage_evidence",
                {
                    **draft_passage_core,
                    "passage_hash": canonical_content_hash(draft_passage_core),
                },
                relations=[
                    {"type": "quotes", "target": source.object_id},
                    {
                        "type": "qualified_supports",
                        "target": hypothesis.object_id,
                    },
                ],
                state="draft",
                actor={
                    "actor_id": "draft-independent-reviewer",
                    "authority": "independent_evaluator",
                },
            )

            hypothesis_context = build_research_context_snapshot(
                repository, target_ids=[hypothesis.object_id]
            )
            self.assertIn(passage.object_id, hypothesis_context["source_object_ids"])
            self.assertIn(source.object_id, hypothesis_context["source_object_ids"])
            passage_view = next(
                item
                for item in hypothesis_context["source_objects"]
                if item["object_id"] == passage.object_id
            )
            self.assertEqual(passage_view["role"], "active_evidence")
            draft_passage_view = next(
                item
                for item in hypothesis_context["source_objects"]
                if item["object_id"] == draft_passage.object_id
            )
            self.assertEqual(draft_passage_view["role"], "lineage")
            self.assertIn(
                passage.object_id,
                hypothesis_context["working_set"]["required_view_ids"],
            )
            self.assertIn(
                passage.object_id,
                {item["object_id"] for item in hypothesis_context["source_views"]},
            )
            hypothesis_assessment = hypothesis_context["belief_context"][
                "target_assessments"
            ][0]
            self.assertEqual(hypothesis_assessment["active_support_count"], 1)
            self.assertEqual(hypothesis_assessment["belief_state"], "supported")
            draft_signal = next(
                item
                for item in hypothesis_assessment["supporting_signals"]
                if item["object_id"] == draft_passage.object_id
            )
            self.assertFalse(draft_signal["active"])

            claim = repository.record(
                "claim",
                {"statement": "Claim H"},
                relations=[{"type": "depends_on", "target": passage.object_id}],
                state="completed",
            )
            claim_context = build_research_context_snapshot(
                repository, target_ids=[claim.object_id]
            )
            claim_assessment = claim_context["belief_context"]["target_assessments"][0]
            self.assertEqual(claim_assessment["active_support_count"], 1)
            self.assertEqual(
                claim_assessment["supporting_signals"][0]["relation_type"],
                "depends_on_evidence",
            )

            update_core = {
                "source_id": source.object_id,
                "status": "retracted",
                "provider": "publisher",
                "checked_at": "2026-08-28T00:00:00+00:00",
                "update_type": "retraction",
            }
            update = repository.record(
                "source_update",
                {
                    **update_core,
                    "update_hash": canonical_content_hash(update_core),
                },
                relations=[
                    {"type": "updates", "target": source.object_id},
                    {"type": "invalidates", "target": source.object_id},
                ],
                state="completed",
            )
            retracted_context = build_research_context_snapshot(
                repository, target_ids=[hypothesis.object_id]
            )
            self.assertIn(update.object_id, retracted_context["source_object_ids"])
            retracted_signal = retracted_context["belief_context"][
                "target_assessments"
            ][0]["supporting_signals"][0]
            self.assertTrue(retracted_signal["invalidated"])
            self.assertFalse(retracted_signal["active"])

    def test_tiny_context_budget_fails_closed_instead_of_hiding_semantics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            ids = self._lineage(repository)

            payload = repository.context(
                target_ids=[ids["hypothesis"]],
                budget_tokens=128,
            )

            self.assertFalse(payload["complete"])
            self.assertFalse(payload["working_set"]["decision_usable"])
            self.assertTrue(payload["blockers"])
            self.assertFalse(research_context_issues(payload))
            with self.assertRaisesRegex(ResearchGitError, "not decision-usable"):
                render_research_context_for_prompt(payload)

    def test_context_chain_stays_bounded_across_repeated_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            ids = self._lineage(repository)
            previous = None
            recorded = []
            for index in range(6):
                current = record_research_context_snapshot(
                    repository,
                    target_ids=[ids["evidence"]],
                    decision_kind="evidence_triage",
                    rationale=[f"decision generation {index}"],
                    budget_tokens=500,
                )
                payload = repository.get(current.object_id)["payload"]
                prior_context_sources = [
                    item
                    for item in payload["source_objects"]
                    if item["kind"] == "context_snapshot"
                ]
                self.assertLessEqual(len(prior_context_sources), 1)
                if previous is not None:
                    self.assertEqual(
                        payload["context_chain"]["previous_context_id"],
                        previous.object_id,
                    )
                    self.assertIn(
                        {
                            "type": "supersedes",
                            "target": previous.object_id,
                            "role": "context_chain",
                        },
                        repository.get(current.object_id)["relations"],
                    )
                previous = current
                recorded.append(payload)

            self.assertEqual(recorded[-1]["context_chain"]["depth"], 5)
            self.assertLess(
                len(str(recorded[-1])),
                len(str(recorded[0])) * 2,
            )
            self.assertFalse(research_context_issues(recorded[-1]))

    def test_recorded_context_is_visible_as_context_edges_on_dag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            ids = self._lineage(repository)
            context = record_research_context_snapshot(
                repository,
                target_ids=[ids["evidence"]],
                decision_kind="evidence_triage",
                selected="hold",
                options_considered=[
                    {"option": "hold", "rejected_because": ""},
                    {
                        "option": "promote",
                        "rejected_because": "the only attempt failed",
                    },
                ],
            )
            repository.commit(stage="context", subject="record decision context")

            graph = repository.dag()
            node = next(
                item for item in graph["nodes"] if item["id"] == context.object_id
            )
            self.assertEqual(node["kind"], "context_snapshot")
            self.assertEqual(node["proof"]["level"], "replayable")
            self.assertIn(
                "context",
                {
                    edge["category"]
                    for edge in graph["edges"]
                    if edge["target"] == context.object_id
                },
            )

    def test_cli_can_inspect_and_record_context_without_raw_git(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            ids = self._lineage(repository)
            repository.commit(stage="evidence", subject="record source lineage")
            output = io.StringIO()
            with redirect_stdout(output):
                status = research_main(
                    [
                        "context",
                        ids["evidence"],
                        "--decision-kind",
                        "triage",
                        "--selected",
                        "hold",
                        "--option",
                        "hold",
                        "--option",
                        "promote=failed attempt is retained",
                        "--repo",
                        str(repository.path),
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn('"context_hash"', output.getvalue())

            prompt_output = io.StringIO()
            with redirect_stdout(prompt_output):
                prompt_status = research_main(
                    [
                        "context",
                        ids["evidence"],
                        "--intent",
                        "continue",
                        "--repo",
                        str(repository.path),
                        "--prompt",
                    ]
                )
            self.assertEqual(prompt_status, 0)
            self.assertIn("bounded, source-bound", prompt_output.getvalue())
            self.assertNotIn("source_objects", prompt_output.getvalue())

    def test_historical_context_resolves_selectors_at_the_requested_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            ids = self._lineage(repository)
            first = repository.commit(stage="evidence", subject="historical evidence")
            later = repository.record(
                "evidence",
                {
                    "result": "later contradictory result",
                    "measurement_hash": "sha256:" + "c" * 64,
                },
                state="completed",
                relations=[
                    {"type": "contradicts", "target": ids["evidence"]},
                    {"type": "refutes", "target": ids["hypothesis"]},
                ],
            )
            repository.commit(stage="evidence", subject="later contradiction")

            historical = build_research_context_snapshot(
                repository,
                target_ids=["@latest:hypothesis"],
                ref=first.commit,
            )
            current = build_research_context_snapshot(
                repository,
                target_ids=["@latest:hypothesis"],
            )

            self.assertEqual(historical["as_of"]["commit"], first.commit)
            self.assertNotIn(later.object_id, historical["source_object_ids"])
            self.assertIn(later.object_id, current["negative_knowledge_ids"])
            self.assertNotEqual(historical["context_hash"], current["context_hash"])

    def test_context_integrity_rejects_relabelled_memory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            ids = self._lineage(repository)
            payload = build_research_context_snapshot(
                repository,
                target_ids=[ids["evidence"]],
            )
            tampered = copy.deepcopy(payload)
            tampered["memory_object_ids"] = []

            self.assertIn(
                "context memory object IDs do not match memory objects",
                research_context_issues(tampered),
            )
            tampered_receipt = copy.deepcopy(payload)
            tampered_receipt["retrieval_receipt"]["candidate_set"].pop()
            self.assertIn(
                "context retrieval candidate set hash mismatch",
                research_context_issues(tampered_receipt),
            )
            tampered_working_set = copy.deepcopy(payload)
            tampered_working_set["working_set"]["source_view_ids"] = []
            self.assertIn(
                "context working-set hash mismatch",
                research_context_issues(tampered_working_set),
            )
            tampered_transform = copy.deepcopy(payload)
            tampered_transform["retrieval_receipt"]["transform_lineage"][0][
                "output_hash"
            ] = ("sha256:" + "0" * 64)
            self.assertIn(
                "context retrieval transform output hash mismatch",
                research_context_issues(tampered_transform),
            )


if __name__ == "__main__":
    unittest.main()
