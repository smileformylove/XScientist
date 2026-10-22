from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.experiment_admission_state import (
    begin_experiment_admission_transition,
    clear_experiment_admission_transition,
    load_experiment_admission_high_water,
    load_experiment_admission_transition,
    record_experiment_admission_high_water,
    require_current_experiment_admission_decision,
)
from xscientist.research_git import ResearchGitError, _repository_common_git_dir
from xscientist.research_vcs import ResearchRepository


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ExperimentAdmissionStateTests(unittest.TestCase):
    def _repository(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            question="Which signed decision is current?",
            git_user_name="Admission State Test",
            git_user_email="admission-state@example.invalid",
        )

    @staticmethod
    def _gate(*, suffix: str, decision: str, task_id: str = "task-1") -> dict:
        scope = {"task_id": task_id}
        bindings = {
            "hypothesis": {
                "object_id": "rso-1111111111111111",
                "content_hash": "sha256:" + "1" * 64,
            },
            "plan": {
                "object_id": "rso-2222222222222222",
                "content_hash": "sha256:" + "2" * 64,
            },
            "preregistration": None,
        }
        request_hash = canonical_content_hash(
            {
                "repository_id": "ws-1111111111111111",
                "input_bindings": bindings,
                "scope": scope,
                "decision": decision,
            }
        )
        payload = {
            "repository_id": "ws-1111111111111111",
            "input_bindings": bindings,
            "scope": scope,
            "decision": decision,
            "admission_request_hash": request_hash,
            "authority_receipt": {"receipt_hash": "sha256:" + suffix * 64},
        }
        return {
            "object_id": f"rso-{suffix * 16}",
            "content_hash": "sha256:" + suffix * 64,
            "payload": payload,
        }

    def test_latest_decision_survives_checkout_and_rejects_older_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            first_head = str(repository.status()["head"])
            authorization = self._gate(suffix="a", decision="authorize")
            first = record_experiment_admission_high_water(
                repository.path,
                authorization,
                checkpoint_commit=first_head,
            )
            denial = self._gate(suffix="b", decision="deny")
            second = record_experiment_admission_high_water(
                repository.path,
                denial,
                checkpoint_commit=first_head,
            )

            self.assertEqual(first["sequence"], 1)
            self.assertEqual(second["sequence"], 2)
            self.assertEqual(second["previous_event_hash"], first["event_hash"])
            with self.assertRaisesRegex(ResearchGitError, "older than"):
                require_current_experiment_admission_decision(
                    repository.path, authorization
                )
            self.assertEqual(
                require_current_experiment_admission_decision(repository.path, denial)[
                    "decision"
                ],
                "deny",
            )

            subprocess.run(
                ["git", "checkout", "--detach", first_head],
                cwd=repository.path,
                check=True,
                capture_output=True,
                text=True,
            )
            with self.assertRaisesRegex(ResearchGitError, "older than"):
                require_current_experiment_admission_decision(
                    repository.path, authorization
                )

    def test_repeated_record_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(suffix="c", decision="authorize")
            first = record_experiment_admission_high_water(
                repository.path,
                gate,
                checkpoint_commit=str(repository.status()["head"]),
            )
            repeated = record_experiment_admission_high_water(
                repository.path,
                gate,
                checkpoint_commit=str(repository.status()["head"]),
            )
            self.assertEqual(repeated, first)

    def test_historical_decision_cannot_be_replayed_after_newer_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            head = str(repository.status()["head"])
            authorization = self._gate(suffix="1", decision="authorize")
            denial = self._gate(suffix="2", decision="deny")
            record_experiment_admission_high_water(
                repository.path,
                authorization,
                checkpoint_commit=head,
            )
            latest = record_experiment_admission_high_water(
                repository.path,
                denial,
                checkpoint_commit=head,
            )

            with self.assertRaisesRegex(ResearchGitError, "replay is older"):
                record_experiment_admission_high_water(
                    repository.path,
                    authorization,
                    checkpoint_commit=head,
                )
            self.assertEqual(
                load_experiment_admission_high_water(
                    repository.path,
                    denial["payload"],
                ),
                latest,
            )

    def test_failed_atomic_publish_leaves_no_partial_event(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(suffix="3", decision="authorize")
            head = str(repository.status()["head"])

            with mock.patch(
                "xscientist.experiment_admission_state.os.link",
                side_effect=OSError("synthetic publish failure"),
            ):
                with self.assertRaisesRegex(OSError, "synthetic publish failure"):
                    record_experiment_admission_high_water(
                        repository.path,
                        gate,
                        checkpoint_commit=head,
                    )

            self.assertIsNone(
                load_experiment_admission_high_water(
                    repository.path,
                    gate["payload"],
                )
            )
            recorded = record_experiment_admission_high_water(
                repository.path,
                gate,
                checkpoint_commit=head,
            )
            self.assertEqual(recorded["sequence"], 1)

    def test_exact_scopes_have_independent_chains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            first = self._gate(suffix="d", decision="authorize", task_id="task-1")
            second = self._gate(suffix="e", decision="deny", task_id="task-2")
            head = str(repository.status()["head"])
            record_experiment_admission_high_water(
                repository.path, first, checkpoint_commit=head
            )
            record_experiment_admission_high_water(
                repository.path, second, checkpoint_commit=head
            )
            self.assertEqual(
                load_experiment_admission_high_water(repository.path, first["payload"])[
                    "sequence"
                ],
                1,
            )
            self.assertEqual(
                load_experiment_admission_high_water(
                    repository.path, second["payload"]
                )["sequence"],
                1,
            )

    def test_tampered_event_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            gate = self._gate(suffix="f", decision="authorize")
            record_experiment_admission_high_water(
                repository.path,
                gate,
                checkpoint_commit=str(repository.status()["head"]),
            )
            events = list(
                (
                    _repository_common_git_dir(repository.path)
                    / "xscientist"
                    / "admission-decisions"
                ).glob("*/events/*.json")
            )
            self.assertEqual(len(events), 1)
            event = json.loads(events[0].read_text(encoding="utf-8"))
            event["decision"] = "deny"
            events[0].write_text(json.dumps(event), encoding="utf-8")

            with self.assertRaisesRegex(ResearchGitError, "hash does not match"):
                load_experiment_admission_high_water(repository.path, gate["payload"])

    def test_gate_without_high_water_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            with self.assertRaisesRegex(ResearchGitError, "no host high-water"):
                require_current_experiment_admission_decision(
                    repository.path,
                    self._gate(suffix="9", decision="authorize"),
                )

    def test_pending_transition_blocks_old_decision_until_publication(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            head = str(repository.status()["head"])
            authorization = self._gate(suffix="4", decision="authorize")
            denial = self._gate(suffix="5", decision="deny")
            record_experiment_admission_high_water(
                repository.path,
                authorization,
                checkpoint_commit=head,
            )

            pending = begin_experiment_admission_transition(repository.path, denial)
            self.assertEqual(
                begin_experiment_admission_transition(repository.path, denial),
                pending,
            )
            with self.assertRaisesRegex(ResearchGitError, "transition is pending"):
                require_current_experiment_admission_decision(
                    repository.path,
                    authorization,
                )

            record_experiment_admission_high_water(
                repository.path,
                denial,
                checkpoint_commit=head,
            )
            clear_experiment_admission_transition(repository.path, denial)
            self.assertIsNone(
                load_experiment_admission_transition(
                    repository.path,
                    denial["payload"],
                )
            )
            self.assertEqual(
                require_current_experiment_admission_decision(
                    repository.path,
                    denial,
                )["decision"],
                "deny",
            )

    def test_conflicting_pending_transition_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            authorization = self._gate(suffix="6", decision="authorize")
            denial = self._gate(suffix="7", decision="deny")
            begin_experiment_admission_transition(repository.path, authorization)

            with self.assertRaisesRegex(ResearchGitError, "transition is pending"):
                begin_experiment_admission_transition(repository.path, denial)

    def test_full_chain_rejects_transition_before_publishing_pending_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            head = str(repository.status()["head"])
            authorization = self._gate(suffix="8", decision="authorize")
            denial = self._gate(suffix="a", decision="deny")
            record_experiment_admission_high_water(
                repository.path,
                authorization,
                checkpoint_commit=head,
            )

            with (
                mock.patch(
                    "xscientist.experiment_admission_state._MAX_EVENTS_PER_SUBJECT",
                    1,
                ),
                self.assertRaisesRegex(ResearchGitError, "chain is too long"),
            ):
                begin_experiment_admission_transition(repository.path, denial)

            self.assertIsNone(
                load_experiment_admission_transition(
                    repository.path,
                    denial["payload"],
                )
            )


if __name__ == "__main__":
    unittest.main()
