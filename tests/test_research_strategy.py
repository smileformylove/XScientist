from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ai_scientist.protocol.research_vcs import (
    ResearchObjectError,
    build_research_object,
)
from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist import ResearchRepository
from xscientist.research_commands import save_claim
from xscientist.research_cli import main as research_main
from xscientist.research_dag import build_research_dag, render_research_dag_html
from xscientist.research_git import ResearchGitError
from xscientist.research_strategy import (
    EVIDENCE_QUALITY_DOMAINS,
    inspect_claim_depth,
    rank_experiment_candidates,
    review_research_program,
    save_discriminating_prediction,
    save_evidence_quality_assessment,
    save_hypothesis_portfolio,
    save_mechanism_model,
    save_transfer_matrix,
    scan_research_anomalies,
)


@unittest.skipUnless(shutil.which("git"), "Git is required")
class ResearchStrategyTests(unittest.TestCase):
    def _repository(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            name="deep-research",
            question="# What mechanism explains the effect?\n",
            git_user_name="Strategy Test",
            git_user_email="strategy@example.invalid",
        )

    def _hypotheses(self, repository: ResearchRepository):
        primary = repository.record(
            "hypothesis",
            {
                "statement": "mediator M causes the effect",
                "falsifier": "M has no effect",
            },
        )
        rival = repository.record(
            "hypothesis",
            {
                "statement": "data leakage causes the effect",
                "falsifier": "leak-free effect remains",
            },
        )
        null = repository.record(
            "hypothesis",
            {
                "statement": "there is no reliable effect",
                "falsifier": "independent replication succeeds",
            },
        )
        return primary, rival, null

    def test_competing_portfolio_prediction_and_information_value_ranking(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            primary, rival, null = self._hypotheses(repository)
            portfolio_result = save_hypothesis_portfolio(
                repository.path,
                question="Which explanation best predicts held-out behavior?",
                primary_id=primary.object_id,
                alternative_ids=[rival.object_id],
                null_id=null.object_id,
                prior_weights={
                    primary.object_id: 2,
                    rival.object_id: 1,
                    null.object_id: 1,
                },
                commit=False,
            )
            portfolio = repository.get(portfolio_result["object"].object_id)
            self.assertAlmostEqual(
                sum(row["prior_weight"] for row in portfolio["payload"]["members"]),
                1.0,
            )

            prediction = save_discriminating_prediction(
                repository.path,
                portfolio_id=portfolio["object_id"],
                hypothesis_id=primary.object_id,
                when="M is ablated without changing the data split",
                expected_outcome="the effect disappears",
                distinguishes_from=[rival.object_id, null.object_id],
                falsifier="the effect remains unchanged",
                commit=False,
            )
            self.assertEqual(prediction["object"].kind, "discriminating_prediction")

            ranking = rank_experiment_candidates(
                repository.path,
                portfolio_id=portfolio["object_id"],
                candidates=[
                    {
                        "candidate_id": "repeat-score",
                        "summary": "Repeat the same aggregate score.",
                        "predictions": {
                            primary.object_id: "increase",
                            rival.object_id: "increase",
                            null.object_id: "increase",
                        },
                        "novelty": 0,
                        "impact": 1,
                        "transfer_value": 0,
                        "cost": 1,
                        "risk": 0,
                        "redundancy": 4,
                    },
                    {
                        "candidate_id": "ablate-mediator",
                        "summary": "Ablate M under a leak-free held-out split.",
                        "predictions": {
                            primary.object_id: "effect_disappears",
                            rival.object_id: "effect_remains",
                            null.object_id: "no_effect",
                        },
                        "novelty": 3,
                        "impact": 4,
                        "transfer_value": 3,
                        "cost": 2,
                        "risk": 1,
                        "redundancy": 0,
                    },
                ],
                commit=False,
            )
            self.assertEqual(
                ranking["ranking"]["selected_candidate_id"], "ablate-mediator"
            )
            self.assertGreater(
                ranking["ranking"]["candidate_set"][0]["expected_information_gain"],
                0.9,
            )

    def test_anomalies_and_periodic_review_are_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            primary, _, _ = self._hypotheses(repository)
            failed = repository.record(
                "experiment_attempt",
                {"status": "failed", "summary": "numerical instability"},
                state="failed",
                relations=[{"type": "depends_on", "target": primary.object_id}],
            )

            preview = scan_research_anomalies(repository.path)
            self.assertEqual(preview["candidate_count"], 1)
            self.assertEqual(preview["candidates"][0]["source_ids"], [failed.object_id])

            review = review_research_program(
                repository.path,
                record=True,
                commit=False,
            )
            self.assertTrue(review["report"]["review_due"])
            self.assertIn(
                "open_anomalies",
                {item["code"] for item in review["report"]["gaps"]},
            )
            self.assertEqual(review["object"].kind, "research_review")
            self.assertEqual(len(review["related"]), 1)
            self.assertEqual(
                scan_research_anomalies(repository.path)["candidate_count"], 0
            )

    def test_mechanism_quality_boundaries_and_claim_insight(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            primary, rival, _ = self._hypotheses(repository)
            evidence = repository.record(
                "evidence",
                {
                    "result": "intervention removes the effect",
                    "measurement_hash": "sha256:" + "a" * 64,
                },
                state="verified",
                relations=[{"type": "supports", "target": primary.object_id}],
            )
            mechanism = save_mechanism_model(
                repository.path,
                hypothesis_id=primary.object_id,
                statement="M mediates the observed effect",
                mediators=["M"],
                interventions=["do(M=0)"],
                rival_hypothesis_ids=[rival.object_id],
                evidence_ids=[evidence.object_id],
                status="validated",
                commit=False,
            )["object"]
            quality = save_evidence_quality_assessment(
                repository.path,
                evidence_id=evidence.object_id,
                domains={name: "low_risk" for name in EVIDENCE_QUALITY_DOMAINS},
                independent=True,
                assessor_id="independent-reviewer",
                commit=False,
            )["object"]
            claim = repository.record(
                "claim",
                {
                    "statement": "M causally mediates a transferable effect",
                    "depth_level": "transferable",
                },
                relations=[
                    {"type": "depends_on", "target": evidence.object_id},
                    {
                        "type": "depends_on",
                        "target": mechanism.object_id,
                        "role": "mechanism",
                    },
                    {
                        "type": "depends_on",
                        "target": quality.object_id,
                        "role": "quality",
                    },
                ],
            )
            matrix = save_transfer_matrix(
                repository.path,
                claim_id=claim.object_id,
                rows=[
                    {
                        "dimension": "domain",
                        "condition": "development domain",
                        "role": "development",
                        "status": "supported",
                        "evidence_ids": [evidence.object_id],
                    },
                    {
                        "dimension": "domain",
                        "condition": "held-out domain",
                        "role": "transfer",
                        "status": "supported",
                        "evidence_ids": [evidence.object_id],
                    },
                    {
                        "dimension": "scale",
                        "condition": "larger scale",
                        "role": "scale",
                        "status": "supported",
                        "evidence_ids": [evidence.object_id],
                    },
                ],
                commit=False,
            )
            self.assertTrue(matrix["matrix"]["transfer_ready"])
            insight = inspect_claim_depth(repository.path, claim.object_id)
            self.assertIn(mechanism.object_id, insight["mechanism_ids"])
            self.assertIn(quality.object_id, insight["quality_assessment_ids"])
            self.assertIn(matrix["object"].object_id, insight["boundary_ids"])
            self.assertFalse(insight["gaps"])

    def test_strategy_hashes_are_validated(self):
        with self.assertRaisesRegex(ResearchObjectError, "portfolio_hash mismatch"):
            build_research_object(
                kind="hypothesis_portfolio",
                payload={
                    "question": "Which hypothesis?",
                    "members": [
                        {
                            "hypothesis_id": "rso-1111111111111111",
                            "role": "primary",
                            "prior_weight": 0.5,
                        },
                        {
                            "hypothesis_id": "rso-2222222222222222",
                            "role": "alternative",
                            "prior_weight": 0.5,
                        },
                    ],
                    "portfolio_hash": "sha256:" + "0" * 64,
                },
            )

    def test_quality_and_transfer_verdicts_cannot_be_self_declared(self):
        quality_core = {
            "evidence_id": "rso-1111111111111111",
            "domains": {
                name: ("high_risk" if name == "confounding" else "low_risk")
                for name in EVIDENCE_QUALITY_DOMAINS
            },
            "notes": {},
            "overall_grade": "strong",
            "independent": True,
        }
        quality_payload = {
            **quality_core,
            "assessment_hash": canonical_content_hash(quality_core),
        }
        with self.assertRaisesRegex(
            ResearchObjectError, "grade does not match domain verdicts"
        ):
            build_research_object(kind="evidence_quality", payload=quality_payload)

        rows = [
            {
                "dimension": "domain" if index < 2 else "scale",
                "condition": f"condition-{index}",
                "role": "transfer" if index == 1 else "development",
                "status": "supported",
                "evidence_ids": ["rso-1111111111111111"],
            }
            for index in range(3)
        ]
        matrix_core = {
            "claim_id": "rso-2222222222222222",
            "rows": rows,
            "coverage": {"tested_count": 3, "dimension_count": 2},
            "transfer_ready": False,
        }
        matrix_payload = {
            **matrix_core,
            "matrix_hash": canonical_content_hash(matrix_core),
        }
        with self.assertRaisesRegex(ResearchObjectError, "readiness does not match"):
            build_research_object(kind="transfer_matrix", payload=matrix_payload)

    def test_causal_and_transferable_claims_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            primary, rival, _ = self._hypotheses(repository)
            evidence = repository.record(
                "evidence",
                {
                    "result": "intervention evidence",
                    "measurement_hash": "sha256:" + "c" * 64,
                },
                state="verified",
            )
            gate = repository.record(
                "gate_decision",
                {"decision": "pass", "claim_promotion_allowed": True},
                state="verified",
                relations=[{"type": "evaluates", "target": evidence.object_id}],
                actor={"actor_id": "gate", "authority": "deterministic_gate"},
            )
            with self.assertRaisesRegex(
                ResearchGitError,
                "validated intervention-tested mechanism",
            ):
                save_claim(
                    repository.path,
                    statement="M causes the effect",
                    evidence_ids=[evidence.object_id],
                    depth_level="causal",
                    gate_id=gate.object_id,
                    verified=True,
                    commit=False,
                )

            mechanism = save_mechanism_model(
                repository.path,
                hypothesis_id=primary.object_id,
                statement="M mediates the effect",
                mediators=["M"],
                interventions=["do(M=0)"],
                rival_hypothesis_ids=[rival.object_id],
                evidence_ids=[evidence.object_id],
                status="validated",
                commit=False,
            )["object"]
            quality = save_evidence_quality_assessment(
                repository.path,
                evidence_id=evidence.object_id,
                domains={name: "low_risk" for name in EVIDENCE_QUALITY_DOMAINS},
                independent=True,
                assessor_id="quality-reviewer",
                commit=False,
            )["object"]
            draft = repository.record(
                "claim",
                {
                    "statement": "M causes a transferable effect",
                    "depth_level": "transferable",
                },
                relations=[{"type": "depends_on", "target": evidence.object_id}],
            )
            matrix = save_transfer_matrix(
                repository.path,
                claim_id=draft.object_id,
                rows=[
                    {
                        "dimension": "domain",
                        "condition": "development",
                        "role": "development",
                        "status": "supported",
                        "evidence_ids": [evidence.object_id],
                    },
                    {
                        "dimension": "domain",
                        "condition": "heldout",
                        "role": "transfer",
                        "status": "supported",
                        "evidence_ids": [evidence.object_id],
                    },
                    {
                        "dimension": "scale",
                        "condition": "large",
                        "role": "scale",
                        "status": "supported",
                        "evidence_ids": [evidence.object_id],
                    },
                ],
                commit=False,
            )["object"]
            verified = save_claim(
                repository.path,
                statement="M causes a transferable effect",
                evidence_ids=[evidence.object_id],
                depth_level="transferable",
                mechanism_ids=[mechanism.object_id],
                quality_ids=[quality.object_id],
                transfer_ids=[matrix.object_id],
                gate_id=gate.object_id,
                verified=True,
                commit=False,
            )["object"]
            self.assertEqual(verified.state, "verified")
            roles = {
                relation.get("role")
                for relation in repository.get(verified.object_id)["relations"]
            }
            self.assertTrue({"mechanism", "quality", "transfer"} <= roles)

    def test_dag_projects_theory_frontier_and_claim_reasoning(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            primary, rival, _ = self._hypotheses(repository)
            portfolio = save_hypothesis_portfolio(
                repository.path,
                question="Which mechanism survives intervention?",
                primary_id=primary.object_id,
                alternative_ids=[rival.object_id],
                commit=False,
            )["object"]
            ranking = rank_experiment_candidates(
                repository.path,
                portfolio_id=portfolio.object_id,
                candidates=[
                    {
                        "candidate_id": "intervene-m",
                        "summary": "Intervene on M under held-out conditions.",
                        "predictions": {
                            primary.object_id: "effect_disappears",
                            rival.object_id: "effect_remains",
                        },
                        "novelty": 3,
                        "impact": 4,
                        "transfer_value": 3,
                        "cost": 1,
                        "risk": 1,
                        "redundancy": 0,
                    }
                ],
                commit=False,
            )["object"]
            claim = repository.record(
                "claim",
                {
                    "statement": "M explains the effect",
                    "depth_level": "causal",
                },
            )
            repository.commit(stage="review", subject="project strategy frontier")

            graph = build_research_dag(repository.path)
            nodes = {node["id"]: node for node in graph["nodes"]}
            self.assertEqual(nodes[portfolio.object_id]["layer"], "strategy")
            self.assertEqual(nodes[ranking.object_id]["layer"], "strategy")
            self.assertEqual(nodes[claim.object_id]["layer"], "theory")
            self.assertEqual(
                graph["theory_frontier"]["next_experiment"]["candidate_id"],
                "intervene-m",
            )
            insight = next(
                item
                for item in graph["claim_insights"]
                if item["claim_id"] == claim.object_id
            )
            self.assertIn("validated_mechanism_missing", insight["gaps"])
            self.assertIn('id="layer"', render_research_dag_html(graph))
            self.assertIn("Claim reasoning", render_research_dag_html(graph))

    def test_program_cli_template_and_review_are_machine_readable(self):
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            output = io.StringIO()
            with redirect_stdout(output):
                status = research_main(["program", "template", "--json"])
            self.assertEqual(status, 0)
            self.assertIn("experiment_candidates", json.loads(output.getvalue()))

            output = io.StringIO()
            with redirect_stdout(output):
                status = research_main(
                    [
                        "program",
                        "review",
                        "--repo",
                        str(repository.path),
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(output.getvalue())["report"]
            self.assertTrue(report["review_due"])
            self.assertIn(
                "portfolio_missing", {item["code"] for item in report["gaps"]}
            )


if __name__ == "__main__":
    unittest.main()
