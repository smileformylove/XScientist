from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.config.paths import get_idea_path
from ai_scientist.resources import idea_resource_path


class PackagedEntrypointDefaultsTests(unittest.TestCase):
    def test_launcher_defaults_use_packaged_idea_resources(self) -> None:
        import launch_scientist_bfts
        import launch_scientist_zhipu

        with mock.patch("sys.argv", ["launch_scientist_bfts.py"]):
            bfts_args = launch_scientist_bfts.parse_arguments()
        with mock.patch("sys.argv", ["launch_scientist_zhipu.py"]):
            zhipu_args = launch_scientist_zhipu.parse_arguments()

        self.assertEqual(Path(bfts_args.load_ideas), idea_resource_path())
        self.assertEqual(Path(zhipu_args.load_ideas), idea_resource_path())

    def test_default_idea_output_is_outside_installed_package(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = get_idea_path("i_cant_believe_its_not_better", output_root=td)

            self.assertEqual(
                output,
                Path(td) / "ideas" / "i_cant_believe_its_not_better.json",
            )
            self.assertNotEqual(output, idea_resource_path())


if __name__ == "__main__":
    unittest.main()
