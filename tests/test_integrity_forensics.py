from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.integrity_forensics import (
    adjudicate_findings,
    build_integrity_ledger,
    check_numeric_consistency,
    check_presentation_signals,
    check_statistical_consistency,
    run_integrity_forensics,
)

SAMPLE_TEX = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
Our method reaches 85.3\% accuracy.
It improves from 73.1\% to
78.0\% accuracy, a 16.7\% relative improvement.
\end{abstract}

\section{Results}
On an evaluation set of 500 examples, accuracy is 84.7\%.
\begin{table}
\caption{Main results}
\begin{tabular}{lc}
Method & Accuracy \\
Base & 73.1 \\
Ours & 84.7 \\
\end{tabular}
\end{table}

\begin{table}
\caption{Copied results}
\begin{tabular}{lc}
Method & Accuracy \\
Base & 73.1 \\
Ours & 84.7 \\
\end{tabular}
\end{table}

\section{Limitations}
TODO: cite the dataset license. As an AI language model, this sentence should not remain.
\end{document}
"""


class IntegrityForensicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tex = self.root / "paper.tex"
        self.tex.write_text(SAMPLE_TEX, encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_build_integrity_ledger_extracts_span_claims(self) -> None:
        ledger = build_integrity_ledger(paper_id="demo", latex_paths=[self.tex])

        self.assertEqual(ledger["paper_id"], "demo")
        self.assertEqual(ledger["ledger_version"], "xscientist.integrity.ledger.v1")
        self.assertGreaterEqual(len(ledger["claims"]), 8)
        self.assertTrue(
            any(claim["type"] == "table_cell" for claim in ledger["claims"])
        )
        self.assertTrue(any(claim["type"] == "number" for claim in ledger["claims"]))
        self.assertTrue(ledger["source_files"][0]["sha256"])

    def test_numeric_checks_find_delta_and_headline_mismatch(self) -> None:
        ledger = build_integrity_ledger(paper_id="demo", latex_paths=[self.tex])
        findings = check_numeric_consistency(ledger)
        pattern_ids = {finding["pattern_id"] for finding in findings}

        self.assertIn("HP-DELTA-ERROR", pattern_ids)
        self.assertIn("HP-NUM-INFLATE", pattern_ids)
        headline_findings = [
            finding for finding in findings if finding["pattern_id"] == "HP-NUM-INFLATE"
        ]
        self.assertEqual(len(headline_findings), 1)
        self.assertIn("85.3", headline_findings[0]["description"])

    def test_statistical_checks_find_grim_impossible_value(self) -> None:
        ledger = build_integrity_ledger(paper_id="demo", latex_paths=[self.tex])
        findings = check_statistical_consistency(ledger)

        self.assertTrue(
            any(
                finding["pattern_id"] == "HP-GRANULARITY-IMPOSSIBLE"
                for finding in findings
            )
        )

    def test_presentation_checks_find_duplicate_table_and_pipeline_artifact(
        self,
    ) -> None:
        ledger = build_integrity_ledger(paper_id="demo", latex_paths=[self.tex])
        findings = check_presentation_signals(ledger)
        pattern_ids = {finding["pattern_id"] for finding in findings}

        self.assertIn("HP-DUP-TABLE", pattern_ids)
        self.assertIn("HP-PIPELINE-ARTIFACT", pattern_ids)

    def test_wrapped_latex_sentence_still_supports_delta_check(self) -> None:
        ledger = build_integrity_ledger(paper_id="demo", latex_paths=[self.tex])
        delta_findings = [
            finding
            for finding in check_numeric_consistency(ledger)
            if finding["pattern_id"] == "HP-DELTA-ERROR"
        ]

        self.assertEqual(len(delta_findings), 1)
        self.assertIn("73.1", delta_findings[0]["evidence"][0]["span"])
        self.assertIn("78.0", delta_findings[0]["evidence"][0]["span"])
        self.assertIn("16.7", delta_findings[0]["evidence"][0]["span"])

    def test_text_source_pipeline_artifact_is_detected(self) -> None:
        text_path = self.root / "paper.txt"
        text_path.write_text(
            "The final manuscript still says: As an AI language model, I cannot verify this.",
            encoding="utf-8",
        )
        ledger = build_integrity_ledger(
            paper_id="demo-text",
            text_paths=[text_path],
            observability_level=0,
        )
        findings = check_presentation_signals(ledger)

        self.assertTrue(
            any(finding["pattern_id"] == "HP-PIPELINE-ARTIFACT" for finding in findings)
        )

    def test_multifile_table_numbering_keeps_tables_distinct(self) -> None:
        first = self.root / "main.tex"
        second = self.root / "supplement.tex"
        table_tex = r"""
\begin{table}
\begin{tabular}{lc}
Base & 10 \\
Ours & 20 \\
\end{tabular}
\end{table}
"""
        first.write_text(table_tex, encoding="utf-8")
        second.write_text(table_tex.replace("20", "30"), encoding="utf-8")
        ledger = build_integrity_ledger(paper_id="multi", latex_paths=[first, second])
        findings = check_presentation_signals(ledger)

        self.assertFalse(
            any(finding["pattern_id"] == "HP-DUP-TABLE" for finding in findings)
        )

    def test_derived_improvement_is_not_headline_inflate_by_itself(self) -> None:
        tex = self.root / "derived.tex"
        tex.write_text(
            r"""
\begin{abstract}
The method improves from 73.1\% to 78.0\% accuracy, a 6.7\% relative improvement.
\end{abstract}
\begin{table}
\begin{tabular}{lc}
Base & 73.1 \\
Ours & 78.0 \\
\end{tabular}
\end{table}
""",
            encoding="utf-8",
        )
        ledger = build_integrity_ledger(paper_id="derived", latex_paths=[tex])
        findings = check_numeric_consistency(ledger)

        self.assertFalse(
            any(finding["pattern_id"] == "HP-NUM-INFLATE" for finding in findings)
        )

    def test_adjudicator_demotes_unanchored_finding(self) -> None:
        ledger = build_integrity_ledger(paper_id="demo", latex_paths=[self.tex])
        report = adjudicate_findings(
            [
                {
                    "finding_id": "BAD001",
                    "skill": "consistency-audit",
                    "pattern_id": "HP-DELTA-ERROR",
                    "title": "Unanchored",
                    "description": "This should not survive the anchor gate.",
                    "severity": "critical",
                    "observability_level_required": 0,
                    "evidence": [
                        {
                            "claim_id": "C001",
                            "span": "not a real span from the ledger",
                        }
                    ],
                    "false_positive_risk": "low",
                }
            ],
            ledger,
        )

        self.assertEqual(report["overall_verdict"], "CLEAN_GIVEN_EVIDENCE")
        self.assertEqual(report["findings"][0]["_severity_final"], "info")
        self.assertIn("anchor-gate-demoted", report["findings"][0]["_adjudication"])

    def test_run_integrity_forensics_writes_artifacts(self) -> None:
        out_dir = self.root / "integrity"
        result = run_integrity_forensics(
            paper_id="demo",
            latex_paths=[self.tex],
            output_dir=out_dir,
            observability_level=1,
        )

        self.assertEqual(result.report["overall_verdict"], "SOFT_FLAGS")
        self.assertTrue((out_dir / "claims.json").exists())
        self.assertTrue((out_dir / "integrity-findings.json").exists())
        self.assertTrue((out_dir / "report.json").exists())
        self.assertTrue((out_dir / "REPORT.md").exists())
        payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["adjudicator"], "xscientist-deterministic-rules-v1")
        self.assertTrue(
            any("same extracted span" in item for item in payload["limitations"])
        )


if __name__ == "__main__":
    unittest.main()
