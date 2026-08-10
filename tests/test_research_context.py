from __future__ import annotations

import copy
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_git import ResearchGitError
from xscientist.research_context import (
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
            receipt = payload["retrieval_receipt"]
            self.assertEqual(
                receipt["profile"], "xscientist.context-retrieval-receipt.v3"
            )
            self.assertTrue(receipt["complete_candidate_set"])
            self.assertEqual(
                {item["object_id"] for item in receipt["candidate_set"]},
                set(payload["source_object_ids"]),
            )
            self.assertFalse(research_context_issues(payload))

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
            self.assertFalse(research_context_issues(payload))

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
