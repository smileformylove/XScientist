from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.apps.project import _revalidate_experiment_admission_preflight
from ai_scientist.protocol.canonical_json import canonical_content_hash
from xscientist.research_git import ResearchGitError


class ExperimentAdmissionWorkerBindingTests(unittest.TestCase):
    def test_worker_rejects_bfts_configuration_changed_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "bfts.yaml"
            approved_text = "agent:\n  model: approved\n"
            config_path.write_text("agent:\n  model: changed\n", encoding="utf-8")
            receipt = {
                "execution_scope": {
                    "configuration_hash": canonical_content_hash(approved_text),
                }
            }
            lifecycle = mock.Mock()
            lifecycle.validate_finalized_experiment_admission_start.return_value = (
                receipt
            )

            with (
                mock.patch(
                    "xscientist.research_lifecycle.ResearchLifecycle",
                    return_value=lifecycle,
                ),
                self.assertRaisesRegex(
                    ResearchGitError,
                    "BFTS configuration changed after parent preflight",
                ),
            ):
                _revalidate_experiment_admission_preflight(
                    root,
                    receipt,
                    bfts_config_path=config_path,
                )

            lifecycle.preflight_experiment_admission.assert_not_called()


if __name__ == "__main__":
    unittest.main()
