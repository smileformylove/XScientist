from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.ara_graph import (
    render_exploration_graph_html,
    write_exploration_graph_visualization,
)


class ARAIntermediateStateVisualizationTests(unittest.TestCase):
    def _graph(self) -> dict:
        return {
            "schema_version": "ara.v1",
            "protocol_kind": "exploration_graph",
            "topology_encoding": "edges",
            "nodes": [
                {
                    "id": "root",
                    "stage": "baseline",
                    "step": 0,
                    "is_buggy": False,
                    "plan_excerpt": "establish baseline",
                },
                {
                    "id": "leaf",
                    "stage": "candidate",
                    "step": 1,
                    "is_buggy": False,
                    "plan_excerpt": "evaluate short reset",
                },
            ],
            "edges": [{"parent": "root", "child": "leaf"}],
        }

    def _event(self) -> dict:
        return {
            "schema_version": "ara.v1",
            "protocol_kind": "research_event",
            "event_id": "evt-" + "a" * 64,
            "event_type": "node_completed",
            "timestamp": "2026-09-21T00:00:00+00:00",
            "actor": "experiment_agent",
            "subject": {"type": "node", "id": "leaf"},
            "object_refs": ["sha256:" + "b" * 64],
            "relations": [
                {"type": "derived_from", "target_type": "node", "target_id": "root"}
            ],
            "attributes": {
                "status": "succeeded",
                "decision": "keep_candidate",
                "metric": {"name": "accuracy", "value": 0.8, "maximize": True},
                "private_note": "SHOULD_NOT_BE_EMBEDDED",
            },
        }

    def test_html_exposes_searchable_presentation_state_without_private_attributes(
        self,
    ) -> None:
        html = render_exploration_graph_html(self._graph(), events=[self._event()])

        self.assertIn('id="state-filter"', html)
        self.assertIn('id="frontier-list"', html)
        self.assertIn('id="event-list"', html)
        self.assertIn("Candidate frontier (DAG leaves)", html)
        self.assertIn(
            "Presentation only; this ledger is not a scientific authority", html
        )
        self.assertIn("node_completed", html)
        self.assertIn("keep_candidate", html)
        self.assertNotIn("SHOULD_NOT_BE_EMBEDDED", html)
        self.assertIn(
            "const presentationEvents = Array.isArray(graph.presentation_events)", html
        )
        self.assertIn("stateFilter.addEventListener('input', renderStateLedger)", html)

    def test_write_visualization_reads_event_ledger_but_keeps_dag_summary_canonical(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ara"
            root.mkdir()
            graph = self._graph()
            (root / "exploration_graph.json").write_text(
                json.dumps(graph), encoding="utf-8"
            )
            events = root / "events"
            events.mkdir()
            (events / "research_events.jsonl").write_text(
                json.dumps(self._event()) + "\n", encoding="utf-8"
            )

            refs = write_exploration_graph_visualization(root)
            html = (root / refs["html"]).read_text(encoding="utf-8")
            summary = json.loads((root / refs["summary"]).read_text(encoding="utf-8"))

            self.assertIn("node_completed", html)
            self.assertNotIn("SHOULD_NOT_BE_EMBEDDED", html)
            self.assertEqual(summary["node_count"], 2)
            self.assertEqual(summary["edge_count"], 1)
            self.assertNotIn("presentation_events", summary)
            self.assertEqual(
                json.loads(
                    (root / "exploration_graph.json").read_text(encoding="utf-8")
                ),
                graph,
            )

    def test_html_marks_truncated_event_window(self) -> None:
        events = [
            {
                **self._event(),
                "event_id": "evt-" + f"{index:064x}"[-64:],
                "timestamp": f"2026-09-21T00:00:{index % 60:02d}+00:00",
            }
            for index in range(513)
        ]
        html = render_exploration_graph_html(self._graph(), events=events)

        self.assertIn('"presentation_event_count": 513', html)
        self.assertIn('"presentation_events_truncated": true', html)
        self.assertIn("latest 512 events shown", html)

    def test_state_lists_have_bounded_keyboard_accessible_scroll_regions(self) -> None:
        html = render_exploration_graph_html(self._graph(), events=[self._event()])
        self.assertIn("max-height: 240px", html)
        self.assertIn("overflow-y: auto", html)
        for element in ("frontier-list", "event-list", "latent-list"):
            self.assertIn(
                f'id="{element}" class="state-list" tabindex="0" aria-label=',
                html,
            )

    def test_malformed_event_values_do_not_break_json_or_embed_nested_payloads(
        self,
    ) -> None:
        event = self._event()
        event.update(relations=17, object_refs=False)
        event["attributes"] = {
            "step": float("inf"),
            "status": "s" * 1000,
            "metric": {
                "name": "m" * 1000,
                "value": {"private": "NESTED_VALUE_MUST_NOT_LEAK"},
                "maximize": ["NESTED_MAXIMIZE_MUST_NOT_LEAK"],
            },
        }
        html = render_exploration_graph_html(self._graph(), events=[event])
        match = re.search(
            r'<script id="graph-data" type="application/json">(.*?)</script>',
            html,
            flags=re.S,
        )
        self.assertIsNotNone(match)
        payload_text = match.group(1)
        self.assertNotIn("Infinity", payload_text)
        self.assertNotIn("NESTED_", payload_text)
        payload = json.loads(payload_text)
        row = payload["presentation_events"][0]
        self.assertEqual(row["relation_count"], 0)
        self.assertEqual(row["object_ref_count"], 0)
        self.assertEqual(len(row["attributes"]["status"]), 240)
        self.assertEqual(len(row["attributes"]["metric"]["name"]), 240)
        self.assertNotIn("step", row["attributes"])
        self.assertNotIn("value", row["attributes"]["metric"])


if __name__ == "__main__":
    unittest.main()
