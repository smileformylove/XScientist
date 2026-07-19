from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.apps.daemon import (
    _load_source_config,
    _normalize_source_entry,
    _validate_source_config,
)


class DaemonSourceConfigTests(unittest.TestCase):
    def test_legacy_daemon_exports_load_and_normalize_toml_queue(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        entries = _load_source_config(
            str(repo_root / "configs" / "sources" / "source_queue.example.toml")
        )
        normalized = _normalize_source_entry(entries[0])

        self.assertEqual(normalized["name"], "broad_impact_day")
        self.assertEqual(
            normalized["workflow_modes"], ["program_driven", "review_board"]
        )
        self.assertEqual(normalized["batch_profile"], "submission_push")
        self.assertEqual(normalized["day_paper_types"], ["journal"])
        self.assertEqual(
            normalized["day_generator_args"],
            ["--rank-ideas", "--top-k-ideas", "3"],
        )

    def test_string_aliases_and_generator_args_are_normalized(self) -> None:
        normalized = _normalize_source_entry(
            {
                "ideas": "ideas.json",
                "paper_types": "journal",
                "alignment_tags": "submission-ready",
                "generator_args": '--model "quality model" --rank-ideas',
            }
        )

        self.assertEqual(normalized["type"], "ideas")
        self.assertEqual(normalized["name"], "ideas")
        self.assertEqual(normalized["paper_types"], ["journal"])
        self.assertEqual(normalized["alignment_tags"], ["submission-ready"])
        self.assertEqual(
            normalized["generator_args"],
            ["--model", "quality model", "--rank-ideas"],
        )

    def test_invalid_json_queue_reports_actionable_errors(self) -> None:
        payload = {
            "sources": [
                {
                    "type": "unknown",
                    "value": "topic.md",
                    "target_venue": "unsupported",
                }
            ]
        }
        errors = _validate_source_config(payload)

        self.assertIn("sources[0].type must be 'topic' or 'ideas'", errors)
        self.assertIn("sources[0].target_venue is not a supported venue", errors)

        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "sources.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "invalid source config"):
                _load_source_config(str(config_path))


if __name__ == "__main__":
    unittest.main()
