from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.apps import batch


class BatchRuntimeImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loaded_before = set(batch._RUNTIME_IMPORTS_LOADED)

    def tearDown(self) -> None:
        batch._RUNTIME_IMPORTS_LOADED.clear()
        batch._RUNTIME_IMPORTS_LOADED.update(self.loaded_before)

    def test_core_import_partition_excludes_learning_and_guidance(self) -> None:
        self.assertFalse(batch._CORE_RUNTIME_ATTRS & batch._LEARNING_RUNTIME_ATTRS)
        self.assertFalse(batch._CORE_RUNTIME_ATTRS & batch._GUIDANCE_RUNTIME_ATTRS)
        self.assertEqual(
            batch._CORE_RUNTIME_ATTRS
            | batch._LEARNING_RUNTIME_ATTRS
            | batch._GUIDANCE_RUNTIME_ATTRS,
            set(batch._RUNTIME_IMPORT_ATTRS),
        )

    def test_core_loader_does_not_resolve_learning_dependencies(self) -> None:
        batch._RUNTIME_IMPORTS_LOADED.clear()
        resolved: list[str] = []

        def fake_loader(module_name: str, attr_name: str):
            resolved.append(attr_name)
            return object()

        with mock.patch.object(batch, "load_module_attr", side_effect=fake_loader):
            batch._ensure_runtime_imports()

        self.assertEqual(batch._RUNTIME_IMPORTS_LOADED, batch._CORE_RUNTIME_ATTRS)
        self.assertEqual(len(resolved), len(batch._CORE_RUNTIME_ATTRS))
        self.assertFalse(batch._RUNTIME_IMPORTS_LOADED & batch._LEARNING_RUNTIME_ATTRS)
        self.assertFalse(batch._RUNTIME_IMPORTS_LOADED & batch._GUIDANCE_RUNTIME_ATTRS)

    def test_learning_disabled_constructor_does_not_require_sklearn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            generator = batch.ContinuousPaperGenerator(
                research_dir=td,
                batch_name="no-learning",
                paper_types=["normal"],
                enable_learning=False,
            )
            self.assertFalse(generator.enable_learning)
            self.assertIsNone(generator.learning_engine)
            self.assertTrue(Path(generator.batch_dir).is_dir())


if __name__ == "__main__":
    unittest.main()
