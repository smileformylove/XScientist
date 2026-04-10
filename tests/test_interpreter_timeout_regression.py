from __future__ import annotations

import tempfile
import unittest


class InterpreterTimeoutRegressionTests(unittest.TestCase):
    def _import_interpreter(self):
        try:
            from ai_scientist.treesearch.interpreter import Interpreter
        except ModuleNotFoundError as exc:
            self.skipTest(f"interpreter dependencies unavailable: {exc}")
        return Interpreter

    def test_timeout_should_not_assert_in_interactive_session(self) -> None:
        Interpreter = self._import_interpreter()
        with tempfile.TemporaryDirectory() as td:
            interpreter = Interpreter(working_dir=td, timeout=1)
            try:
                first = interpreter.run("import time; time.sleep(2)", reset_session=True)
                second = interpreter.run("import time; time.sleep(2)", reset_session=False)
            finally:
                interpreter.cleanup_session()

        allowed = {"TimeoutError", "KeyboardInterrupt"}
        self.assertIn(first.exc_type, allowed)
        self.assertIn(second.exc_type, allowed)
        self.assertGreaterEqual(first.exec_time, 0)
        self.assertGreaterEqual(second.exec_time, 0)


if __name__ == "__main__":
    unittest.main()
