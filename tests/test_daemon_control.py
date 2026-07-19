from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from continuous_research_daemon import (
    _default_control_payload,
    _load_control_payload,
    _load_recent_control_events,
    _save_control_payload,
)


class DaemonControlTests(unittest.TestCase):
    def test_legacy_daemon_exports_use_persistent_control_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            daemon_dir = Path(td)
            payload = _default_control_payload()
            payload.update(
                {
                    "paused": True,
                    "expires_after_cycles": 2,
                    "source_commands": {
                        "topic": {
                            "force_next_cycle": True,
                            "expires_after_cycles": 1,
                        }
                    },
                }
            )
            _save_control_payload(daemon_dir, payload)

            initialized = _load_control_payload(daemon_dir, current_cycle=3)
            expired = _load_control_payload(daemon_dir, current_cycle=5)

            self.assertEqual(initialized["_expires_after_cycle"], 5)
            self.assertEqual(
                initialized["source_commands"]["topic"]["_expires_after_cycle"],
                4,
            )
            self.assertFalse(expired["paused"])
            self.assertEqual(expired["source_commands"], {})
            self.assertEqual(expired["validation_errors"], [])
            persisted = json.loads(
                (daemon_dir / "daemon_control.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("validation_errors", persisted)
            event_types = {
                row["type"] for row in _load_recent_control_events(daemon_dir)
            }
            self.assertIn("control_override_expired", event_types)
            self.assertIn("source_command_expired", event_types)

    def test_invalid_json_falls_back_to_defaults_and_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            daemon_dir = Path(td)
            control_path = daemon_dir / "daemon_control.json"
            control_path.write_text("{invalid", encoding="utf-8")

            payload = _load_control_payload(daemon_dir)

            self.assertFalse(payload["paused"])
            self.assertIn("not valid JSON", payload["validation_errors"][0])

    def test_non_object_json_reports_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            daemon_dir = Path(td)
            control_path = daemon_dir / "daemon_control.json"
            control_path.write_text("[]", encoding="utf-8")

            payload = _load_control_payload(daemon_dir)

            self.assertFalse(payload["paused"])
            self.assertIn(
                "control payload must be a JSON object",
                payload["validation_errors"],
            )

    def test_saved_control_file_is_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            daemon_dir = Path(td)
            payload = _default_control_payload()
            payload["force_mode"] = "focus_rewrite"

            _save_control_payload(daemon_dir, payload)

            saved = json.loads(
                (daemon_dir / "daemon_control.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved, payload)


if __name__ == "__main__":
    unittest.main()
