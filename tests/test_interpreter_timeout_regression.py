from __future__ import annotations

import tempfile
import unittest
from unittest import mock


class InterpreterTimeoutRegressionTests(unittest.TestCase):
    def _import_interpreter(self):
        try:
            from ai_scientist.treesearch import interpreter
        except ModuleNotFoundError as exc:
            self.skipTest(f"interpreter dependencies unavailable: {exc}")
        return interpreter

    def test_timeout_should_not_assert_in_interactive_session(self) -> None:
        Interpreter = self._import_interpreter().Interpreter
        with tempfile.TemporaryDirectory() as td:
            interpreter = Interpreter(working_dir=td, timeout=1)
            try:
                first = interpreter.run(
                    "import time; time.sleep(2)", reset_session=True
                )
                second = interpreter.run(
                    "import time; time.sleep(2)", reset_session=False
                )
            finally:
                interpreter.cleanup_session()

        allowed = {"TimeoutError", "KeyboardInterrupt"}
        self.assertIn(first.exc_type, allowed)
        self.assertIn(second.exc_type, allowed)
        self.assertGreaterEqual(first.exec_time, 0)
        self.assertGreaterEqual(second.exec_time, 0)

    def test_process_backend_scrubs_parent_secrets_and_records_fallback(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.dict(
                "os.environ", {"OPENAI_API_KEY": "should-not-leak"}, clear=False
            ),
        ):
            interpreter = module.Interpreter(
                working_dir=td,
                timeout=3,
                sandbox_policy=module.SandboxPolicy(
                    backend="process", require_isolation=False
                ),
            )
            try:
                result = interpreter.run(
                    "import os; print(os.getenv('OPENAI_API_KEY', 'missing'))"
                )
            finally:
                interpreter.cleanup_session()

        self.assertIn("missing", "".join(result.term_out))
        self.assertEqual(result.execution_backend, "process")
        self.assertFalse(result.isolation["isolated"])

    def test_auto_backend_falls_back_with_explicit_reason(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(
                module, "docker_is_available", return_value=(False, "daemon offline")
            ),
        ):
            interpreter = module.Interpreter(
                working_dir=td,
                sandbox_policy=module.SandboxPolicy(
                    backend="auto", require_isolation=False
                ),
            )

        self.assertEqual(interpreter.execution_backend, "process")
        self.assertEqual(
            interpreter.execution_metadata()["fallback_reason"], "daemon offline"
        )

    def test_required_isolation_fails_closed(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(
                module, "docker_is_available", return_value=(False, "daemon offline")
            ),
        ):
            with self.assertRaises(module.SandboxUnavailableError):
                module.Interpreter(
                    working_dir=td,
                    sandbox_policy=module.SandboxPolicy(
                        backend="auto", require_isolation=True
                    ),
                )

    def test_docker_command_applies_security_limits(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(module, "docker_is_available", return_value=(True, None)),
            mock.patch.object(
                module, "docker_image_is_available", return_value=(True, None)
            ),
            mock.patch.object(module.shutil, "which", return_value="/usr/bin/docker"),
        ):
            interpreter = module.Interpreter(
                working_dir=td,
                env_vars={"SAFE_FLAG": "1"},
                sandbox_policy=module.SandboxPolicy(
                    backend="docker",
                    docker_image="example/image@sha256:abc",
                    network="none",
                    memory="2g",
                    cpus=1.5,
                    pids_limit=64,
                    read_only_mounts=(str(module.Path(td).resolve().parent),),
                ),
            )
            command = interpreter._docker_command()

        rendered = " ".join(command)
        self.assertIn("--network none", rendered)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop ALL", rendered)
        self.assertIn(f"--user {module.os.getuid()}:{module.os.getgid()}", rendered)
        self.assertIn("--security-opt no-new-privileges", rendered)
        self.assertIn("--memory 2g", rendered)
        self.assertIn("--cpus 1.5", rendered)
        self.assertIn("--pids-limit 64", rendered)
        self.assertIn("readonly", rendered)
        self.assertNotIn("OPENAI_API_KEY", rendered)

    def test_docker_timeout_forces_container_cleanup(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(module, "docker_is_available", return_value=(True, None)),
            mock.patch.object(
                module, "docker_image_is_available", return_value=(True, None)
            ),
            mock.patch.object(module.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                module.subprocess,
                "run",
                side_effect=module.subprocess.TimeoutExpired("docker", 1),
            ),
            mock.patch.object(
                module.Interpreter, "_force_remove_container"
            ) as remove_mock,
        ):
            interpreter = module.Interpreter(
                working_dir=td,
                timeout=1,
                sandbox_policy=module.SandboxPolicy(backend="docker"),
            )
            result = interpreter._run_docker("print('slow')")

        self.assertEqual(result.exc_type, "TimeoutError")
        remove_mock.assert_called_once()

    def test_auto_backend_falls_back_when_image_is_missing(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(module, "docker_is_available", return_value=(True, None)),
            mock.patch.object(
                module,
                "docker_image_is_available",
                return_value=(False, "image missing"),
            ),
        ):
            interpreter = module.Interpreter(
                working_dir=td,
                sandbox_policy=module.SandboxPolicy(
                    backend="auto", require_isolation=False
                ),
            )

        self.assertEqual(interpreter.execution_backend, "process")
        self.assertEqual(
            interpreter.execution_metadata()["fallback_reason"], "image missing"
        )

    def test_workspace_policy_mounts_data_and_cache_without_api_keys(self) -> None:
        module = self._import_interpreter()
        try:
            from ai_scientist.treesearch import parallel_agent
        except ModuleNotFoundError as exc:
            self.skipTest(f"parallel agent dependencies unavailable: {exc}")

        class ExecConfig:
            backend = "process"
            require_isolation = False
            docker_image = "xscientist-exec:latest"
            network = "none"
            memory = "1g"
            cpus = 1.0
            pids_limit = 32
            read_only_root = True
            read_only_mounts = ()

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.dict(
                "os.environ",
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "OPENAI_API_KEY": "must-not-pass",
                },
                clear=False,
            ),
        ):
            cfg = type(
                "Cfg",
                (),
                {"exec": ExecConfig(), "data_dir": td, "log_dir": td},
            )()
            policy = parallel_agent._sandbox_policy_for_workspace(cfg)
            env = parallel_agent._experiment_execution_env()

        resolved_td = str(module.Path(td).resolve())
        self.assertIn(resolved_td, policy.read_only_mounts)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertIsInstance(policy, module.SandboxPolicy)

    def test_strict_workspace_policy_rejects_experiment_network(self) -> None:
        try:
            from ai_scientist.treesearch import parallel_agent
        except ModuleNotFoundError as exc:
            self.skipTest(f"parallel agent dependencies unavailable: {exc}")

        exec_cfg = type(
            "ExecConfig",
            (),
            {"allow_experiment_network": True, "require_isolation": True},
        )()
        cfg = type("Cfg", (), {"exec": exec_cfg})()
        with self.assertRaises(ValueError):
            parallel_agent._experiment_network_enabled(cfg)

    def test_experiment_network_defaults_off(self) -> None:
        try:
            from ai_scientist.treesearch import parallel_agent
        except ModuleNotFoundError as exc:
            self.skipTest(f"parallel agent dependencies unavailable: {exc}")

        cfg = type("Cfg", (), {"exec": type("ExecConfig", (), {})()})()
        self.assertFalse(parallel_agent._experiment_network_enabled(cfg))


if __name__ == "__main__":
    unittest.main()
