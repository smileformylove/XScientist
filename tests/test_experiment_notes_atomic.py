from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_scientist.treesearch import journal as journal_module
from ai_scientist.treesearch.journal import Journal, Node


class _SummaryAgent:
    def __init__(self, summary) -> None:
        self.summary = summary

    def _generate_node_summary(self, _node: Node):
        return self.summary


class ExperimentNotesAtomicTests(unittest.TestCase):
    @staticmethod
    def _config(*, summary=None, report_model: str = "test/report-model"):
        return SimpleNamespace(
            agent={} if summary is None else {"summary": summary},
            report=SimpleNamespace(model=report_model, temp=0.2),
        )

    @staticmethod
    def _journal_with_summary(summary) -> tuple[Journal, Node]:
        journal = Journal()
        node = Node()
        node._agent = _SummaryAgent(summary)
        journal.nodes.append(node)
        return journal, node

    def test_save_experiment_notes_writes_valid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            journal, node = self._journal_with_summary({"finding": "稳定"})

            with mock.patch.object(
                journal_module, "query", return_value="stage summary"
            ) as query:
                journal.save_experiment_notes(td, "stage_demo", self._config())

            self.assertEqual(query.call_args.kwargs["model"], "test/report-model")
            self.assertEqual(query.call_args.kwargs["temperature"], 0.2)

            notes_dir = Path(td) / "experiment_notes"
            node_summary_path = notes_dir / f"stage_demo_node_{node.id}_summary.json"
            self.assertEqual(
                json.loads(node_summary_path.read_text(encoding="utf-8")),
                {"finding": "稳定"},
            )
            self.assertEqual(
                (notes_dir / "stage_demo_summary.txt").read_text(encoding="utf-8"),
                "stage summary",
            )

    def test_node_summary_serialization_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            journal, node = self._journal_with_summary({"invalid": {"set"}})
            notes_dir = Path(td) / "experiment_notes"
            notes_dir.mkdir()
            node_summary_path = notes_dir / f"stage_demo_node_{node.id}_summary.json"
            node_summary_path.write_text("previous", encoding="utf-8")

            with self.assertRaises(TypeError):
                journal.save_experiment_notes(td, "stage_demo", self._config())

            self.assertEqual(node_summary_path.read_text(encoding="utf-8"), "previous")

    def test_stage_summary_write_failure_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            journal, _node = self._journal_with_summary({"finding": "new"})
            notes_dir = Path(td) / "experiment_notes"
            notes_dir.mkdir()
            stage_summary_path = notes_dir / "stage_demo_summary.txt"
            stage_summary_path.write_text("previous", encoding="utf-8")

            with (
                mock.patch.object(journal_module, "query", return_value="new summary"),
                mock.patch.object(
                    journal_module,
                    "atomic_write_text",
                    side_effect=OSError("disk busy"),
                ),
                self.assertRaisesRegex(OSError, "disk busy"),
            ):
                journal.save_experiment_notes(td, "stage_demo", self._config())

            self.assertEqual(stage_summary_path.read_text(encoding="utf-8"), "previous")

    def test_agent_summary_route_takes_precedence_over_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            journal, _node = self._journal_with_summary({"finding": "new"})
            summary_cfg = SimpleNamespace(model="test/summary-model", temp=0.1)

            with mock.patch.object(
                journal_module, "query", return_value="stage summary"
            ) as query:
                journal.save_experiment_notes(
                    td,
                    "stage_demo",
                    self._config(summary=summary_cfg),
                )

            self.assertEqual(query.call_args.kwargs["model"], "test/summary-model")
            self.assertEqual(query.call_args.kwargs["temperature"], 0.1)

    def test_missing_summary_route_fails_before_writing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            journal, _node = self._journal_with_summary({"finding": "new"})

            with (
                mock.patch.dict(
                    journal_module.os.environ,
                    {"ZHIPU_DEFAULT_MODEL": ""},
                ),
                mock.patch.object(journal_module, "query") as query,
                self.assertRaisesRegex(
                    ValueError,
                    "requires cfg.agent.summary, cfg.report, or ZHIPU_DEFAULT_MODEL",
                ),
            ):
                journal.save_experiment_notes(
                    td, "stage_demo", SimpleNamespace(agent={})
                )

            query.assert_not_called()
            self.assertFalse((Path(td) / "experiment_notes").exists())

    def test_explicit_environment_model_is_last_resort(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            journal, _node = self._journal_with_summary({"finding": "new"})

            with (
                mock.patch.dict(
                    journal_module.os.environ,
                    {"ZHIPU_DEFAULT_MODEL": "openai_compat/glm-5.3"},
                ),
                mock.patch.object(
                    journal_module, "query", return_value="stage summary"
                ) as query,
            ):
                journal.save_experiment_notes(
                    td, "stage_demo", SimpleNamespace(agent={})
                )

            self.assertEqual(query.call_args.kwargs["model"], "openai_compat/glm-5.3")
            self.assertEqual(query.call_args.kwargs["temperature"], 0.3)


if __name__ == "__main__":
    unittest.main()
