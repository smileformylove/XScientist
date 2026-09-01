from __future__ import annotations

import tempfile
import unittest
from unittest import mock


class InterpreterTimeoutRegressionTests(unittest.TestCase):
    _IMAGE_ID = "sha256:" + "a" * 64

    def _import_interpreter(self):
        try:
            from ai_scientist.treesearch import interpreter
        except ModuleNotFoundError as exc:
            self.skipTest(f"interpreter dependencies unavailable: {exc}")
        return interpreter

    def _available_image(self, _image, *, verified_identity=None, **_kwargs):
        if verified_identity is not None:
            verified_identity["image_id"] = self._IMAGE_ID
        return True, None

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

    def test_docker_probe_errors_do_not_disclose_host_paths(self) -> None:
        module = self._import_interpreter()
        private_docker = "/" + "Users" + "/alice/private-tools/docker"
        with (
            mock.patch.object(module.shutil, "which", return_value=private_docker),
            mock.patch.object(
                module.subprocess,
                "run",
                side_effect=module.subprocess.TimeoutExpired(private_docker, 5),
            ),
        ):
            available, reason = module.docker_is_available()

        self.assertFalse(available)
        self.assertEqual(reason, "docker availability check timed out")
        self.assertNotIn(private_docker, reason or "")

        private_socket = "unix:///" + "Users" + "/alice/.docker/run/docker.sock"
        failed = module.subprocess.CompletedProcess(
            [private_docker, "info"],
            1,
            "",
            f"cannot connect to {private_socket}",
        )
        with (
            mock.patch.object(module.shutil, "which", return_value=private_docker),
            mock.patch.object(module.subprocess, "run", return_value=failed),
        ):
            available, reason = module.docker_is_available()

        self.assertFalse(available)
        self.assertNotIn(private_socket, reason or "")
        self.assertEqual(reason, "docker daemon is unavailable")

    def test_docker_image_probe_errors_do_not_disclose_host_paths(self) -> None:
        module = self._import_interpreter()
        private_docker = "/" + "Users" + "/alice/private-tools/docker"
        private_image = "registry.invalid/private/research:latest"
        with (
            mock.patch.object(module.shutil, "which", return_value=private_docker),
            mock.patch.object(
                module.subprocess,
                "run",
                side_effect=module.subprocess.TimeoutExpired(private_docker, 10),
            ),
        ):
            available, reason = module.docker_image_is_available(private_image)

        self.assertFalse(available)
        self.assertEqual(reason, "docker image inspection timed out")
        self.assertNotIn(private_docker, reason or "")
        self.assertNotIn(private_image, reason or "")

        with (
            mock.patch.object(module.shutil, "which", return_value=private_docker),
            mock.patch.object(
                module.subprocess,
                "run",
                side_effect=OSError(f"cannot execute {private_docker}"),
            ),
        ):
            available, reason = module.docker_image_is_available(private_image)

        self.assertFalse(available)
        self.assertEqual(reason, "docker image inspection failed")
        self.assertNotIn(private_docker, reason or "")
        self.assertNotIn(private_image, reason or "")

    def test_docker_command_applies_security_limits(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(module, "docker_is_available", return_value=(True, None)),
            mock.patch.object(
                module,
                "docker_image_is_available",
                side_effect=self._available_image,
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
        self.assertIn(self._IMAGE_ID, command)
        self.assertNotIn("example/image@sha256:abc", command)
        self.assertEqual(
            interpreter.execution_metadata()["docker_image_id"], self._IMAGE_ID
        )

    def test_docker_timeout_forces_container_cleanup(self) -> None:
        module = self._import_interpreter()
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(module, "docker_is_available", return_value=(True, None)),
            mock.patch.object(
                module,
                "docker_image_is_available",
                side_effect=self._available_image,
            ),
            mock.patch.object(module.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                module,
                "run_process_bounded",
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

    def test_process_backend_caps_output_in_memory(self) -> None:
        module = self._import_interpreter()
        with tempfile.TemporaryDirectory() as td:
            interpreter = module.Interpreter(
                working_dir=td,
                sandbox_policy=module.SandboxPolicy(
                    backend="process",
                    require_isolation=False,
                    max_output_chars=64,
                ),
            )
            try:
                result = interpreter.run("print('x' * 10000)")
            finally:
                interpreter.cleanup_session()

        self.assertTrue(result.output_truncated)
        self.assertNotIn("x" * 100, "".join(result.term_out))

    def test_process_backend_enforces_workspace_quota(self) -> None:
        module = self._import_interpreter()
        with tempfile.TemporaryDirectory() as td:
            interpreter = module.Interpreter(
                working_dir=td,
                sandbox_policy=module.SandboxPolicy(
                    backend="process",
                    require_isolation=False,
                    max_workspace_bytes=128,
                    max_workspace_files=10,
                ),
            )
            try:
                result = interpreter.run(
                    "from pathlib import Path; Path('large.bin').write_bytes(b'x' * 4096)"
                )
            finally:
                interpreter.cleanup_session()

        self.assertEqual(result.exc_type, "ResourceLimitError")
        self.assertIn("workspace size exceeded", "".join(result.term_out))

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

    def test_workspace_docker_backend_rejects_a_stale_executor_identity(self) -> None:
        module = self._import_interpreter()
        with tempfile.TemporaryDirectory() as td:
            workspace = module.Path(td) / "study"
            working_dir = workspace / "02_experiments" / "idea" / "working"
            working_dir.mkdir(parents=True)
            (workspace / "bfts_config.yaml").write_text("exec: {}\n")
            (workspace / "Dockerfile.executor").write_text("FROM python:3.11-slim\n")
            stale = {
                "ok": False,
                "image": "xscientist-exec:test",
                "error": "executor source revision is stale",
            }
            with (
                mock.patch.object(
                    module, "docker_is_available", return_value=(True, None)
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    return_value=stale,
                ) as inspect,
            ):
                with self.assertRaisesRegex(
                    module.SandboxUnavailableError,
                    "source revision is stale",
                ):
                    module.Interpreter(
                        working_dir=working_dir,
                        sandbox_policy=module.SandboxPolicy(
                            backend="docker",
                            require_isolation=True,
                            docker_image="xscientist-exec:test",
                        ),
                    )

        inspect.assert_called_once_with(workspace.resolve())

    def test_explicit_executor_workspace_cannot_be_shadowed_by_working_dir(
        self,
    ) -> None:
        module = self._import_interpreter()
        with tempfile.TemporaryDirectory() as td:
            trusted = module.Path(td) / "trusted"
            nested = trusted / "runs" / "candidate"
            for root in (trusted, nested):
                root.mkdir(parents=True, exist_ok=True)
                (root / "bfts_config.yaml").write_text("exec: {}\n")
                (root / "Dockerfile.executor").write_text("FROM python:3.11-slim\n")
            ready = {
                "ok": True,
                "image": "xscientist-exec:test",
                "image_id": self._IMAGE_ID,
                "error": None,
            }
            with (
                mock.patch.dict(
                    module.os.environ,
                    {"XSCIENTIST_WORKSPACE": str(trusted)},
                    clear=False,
                ),
                mock.patch(
                    "xscientist.executor_manager.inspect_executor",
                    return_value=ready,
                ) as inspect,
            ):
                available, reason = module.docker_image_is_available(
                    "xscientist-exec:test",
                    workspace=nested,
                )

        self.assertTrue(available)
        self.assertIsNone(reason)
        inspect.assert_called_once_with(trusted.resolve())

    def test_invalid_explicit_executor_workspace_fails_closed(self) -> None:
        module = self._import_interpreter()
        with tempfile.TemporaryDirectory() as td:
            invalid = module.Path(td) / "not-initialized"
            invalid.mkdir()
            with (
                mock.patch.dict(
                    module.os.environ,
                    {"XSCIENTIST_WORKSPACE": str(invalid)},
                    clear=False,
                ),
                mock.patch.object(module.subprocess, "run") as run,
            ):
                available, reason = module.docker_image_is_available(
                    "xscientist-exec:test",
                    workspace=module.Path(td),
                )

        self.assertFalse(available)
        self.assertIn("not an initialized executor workspace", reason or "")
        run.assert_not_called()

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
