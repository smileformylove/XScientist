# Modified by XScientist contributors from the AI-Scientist-v2/AIDE lineage.
# See THIRD_PARTY_NOTICES.md for provenance and license details.
"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
- optional Docker isolation for AI-generated code
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Literal

import humanize
from dataclasses_json import DataClassJsonMixin

from ai_scientist.utils.bounded_process import (
    BoundedTextBuffer,
    ProcessResourceLimitExceeded,
    run_process_bounded,
    workspace_limit_checker,
)
from ai_scientist.utils.privacy import redact_sensitive_text

logger = logging.getLogger("ai-scientist")
_DOCKER_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    execution_backend: str = "process"
    isolation: dict[str, Any] | None = None
    output_truncated: bool = False


class SandboxUnavailableError(RuntimeError):
    """Raised when an isolated backend is required but cannot be started."""


@dataclass(frozen=True)
class SandboxPolicy:
    """Execution policy for AI-generated experiment code."""

    backend: Literal["auto", "process", "docker"] = "auto"
    require_isolation: bool = True
    docker_image: str = "xscientist-exec:latest"
    network: Literal["none", "bridge"] = "none"
    memory: str = "4g"
    cpus: float = 2.0
    pids_limit: int = 256
    read_only_root: bool = True
    read_only_mounts: tuple[str, ...] = ()
    max_output_chars: int = 200_000
    max_workspace_bytes: int = 10 * 1024 * 1024 * 1024
    max_workspace_files: int = 100_000

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "process", "docker"}:
            raise ValueError(f"Unsupported execution backend: {self.backend}")
        if self.network not in {"none", "bridge"}:
            raise ValueError(f"Unsupported Docker network mode: {self.network}")
        if self.cpus <= 0:
            raise ValueError("Sandbox CPU limit must be positive")
        if self.pids_limit <= 0:
            raise ValueError("Sandbox PID limit must be positive")
        if self.max_output_chars <= 0:
            raise ValueError("Sandbox output limit must be positive")
        if self.max_workspace_bytes <= 0:
            raise ValueError("Sandbox workspace byte limit must be positive")
        if self.max_workspace_files <= 0:
            raise ValueError("Sandbox workspace file limit must be positive")


def sandbox_policy_from_config(exec_config: Any | None) -> SandboxPolicy:
    """Build a sandbox policy from an OmegaConf/dataclass execution config."""

    if exec_config is None:
        return SandboxPolicy()

    return SandboxPolicy(
        backend=str(getattr(exec_config, "backend", "auto") or "auto").lower(),
        require_isolation=bool(getattr(exec_config, "require_isolation", True)),
        docker_image=str(
            getattr(exec_config, "docker_image", "xscientist-exec:latest")
            or "xscientist-exec:latest"
        ),
        network=str(getattr(exec_config, "network", "none") or "none").lower(),
        memory=str(getattr(exec_config, "memory", "4g") or "4g"),
        cpus=float(getattr(exec_config, "cpus", 2.0) or 2.0),
        pids_limit=int(getattr(exec_config, "pids_limit", 256) or 256),
        read_only_root=bool(getattr(exec_config, "read_only_root", True)),
        read_only_mounts=tuple(
            str(item)
            for item in (getattr(exec_config, "read_only_mounts", ()) or ())
            if str(item).strip()
        ),
        max_output_chars=int(
            getattr(exec_config, "max_output_chars", 200_000) or 200_000
        ),
        max_workspace_bytes=int(
            getattr(exec_config, "max_workspace_bytes", 10 * 1024**3) or 10 * 1024**3
        ),
        max_workspace_files=int(
            getattr(exec_config, "max_workspace_files", 100_000) or 100_000
        ),
    )


def docker_is_available() -> tuple[bool, str | None]:
    """Return whether the Docker daemon is reachable, plus a failure reason."""

    docker = shutil.which("docker")
    if docker is None:
        return False, "docker executable not found"
    try:
        probe = subprocess.run(
            [docker, "info", "--format", "{{json .ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return False, "docker availability check timed out"
    except OSError:
        return False, "docker availability check could not start"
    if probe.returncode != 0:
        return False, "docker daemon is unavailable"
    return True, None


def docker_image_is_available(
    image: str,
    *,
    workspace: str | Path | None = None,
    verified_identity: dict[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Return whether an executor image is present and workspace-compatible."""

    explicit_workspace = str(os.environ.get("XSCIENTIST_WORKSPACE") or "").strip()
    workspace_hints = (
        [explicit_workspace]
        if explicit_workspace
        else ([str(workspace)] if workspace is not None else [])
    )
    if workspace_hints:
        from xscientist.executor_manager import (
            ExecutorManagerError,
            inspect_executor,
            resolve_executor_workspace,
        )

        executor_workspace = next(
            (
                resolved
                for hint in workspace_hints
                if (resolved := resolve_executor_workspace(hint)) is not None
            ),
            None,
        )
        if explicit_workspace and executor_workspace is None:
            return (
                False,
                "explicit XSCIENTIST_WORKSPACE is not an initialized executor "
                "workspace",
            )
        if executor_workspace is not None:
            try:
                status = inspect_executor(executor_workspace)
            except ExecutorManagerError as exc:
                return False, (
                    "executor identity check failed: " + redact_sensitive_text(str(exc))
                )
            except OSError:
                return False, "executor identity check could not start Docker"
            if status.get("image") != image:
                return (
                    False,
                    "configured docker image does not match the initialized "
                    "executor workspace",
                )
            if not status.get("ok"):
                return False, (
                    "executor identity check failed: "
                    + redact_sensitive_text(str(status.get("error") or "unknown error"))
                )
            image_id = str(status.get("image_id") or "").strip().lower()
            if not _DOCKER_IMAGE_ID_PATTERN.fullmatch(image_id):
                return False, "executor identity check returned no immutable image ID"
            if verified_identity is not None:
                verified_identity["image_id"] = image_id
            return True, None

    docker = shutil.which("docker")
    if docker is None:
        return False, "docker executable not found"
    try:
        probe = subprocess.run(
            [docker, "image", "inspect", image, "--format", "{{.Id}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "docker image inspection timed out"
    except OSError:
        return False, "docker image inspection failed"
    if probe.returncode != 0:
        return False, f"docker image is not available locally: {image}"
    image_id = str(probe.stdout or "").strip().lower()
    if not _DOCKER_IMAGE_ID_PATTERN.fullmatch(image_id):
        return False, "docker image check returned no immutable image ID"
    if verified_identity is not None:
        verified_identity["image_id"] = image_id
    return True, None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [l for l in tb_lines if "treesearch/" not in l and "importlib" not in l]
        )

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue, *, chunk_chars: int = 8192):
        self.queue = queue
        self.chunk_chars = chunk_chars

    def write(self, msg):
        text = str(msg or "")
        for offset in range(0, len(text), self.chunk_chars):
            self.queue.put(text[offset : offset + self.chunk_chars])

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str | None] | None = None,
        sandbox_policy: SandboxPolicy | None = None,
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            env_vars: Explicit environment variables to expose to the generated code.
            sandbox_policy: Selects process/Docker execution and isolation limits.
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore
        self.env_vars = {
            str(key): str(value)
            for key, value in (env_vars or {}).items()
            if value is not None
        }
        # Direct Interpreter use is an explicit local API choice. Autonomous
        # workflows pass a config-derived policy, whose default is fail-closed.
        self.sandbox_policy = sandbox_policy or SandboxPolicy(
            backend="process", require_isolation=False
        )
        self.verified_docker_image_id: str | None = None
        self.execution_backend, self.isolation_reason = self._resolve_backend()

    def _resolve_backend(self) -> tuple[str, str | None]:
        policy = self.sandbox_policy
        if policy.backend == "process":
            if policy.require_isolation:
                raise SandboxUnavailableError(
                    "Execution isolation is required, but backend='process' is not isolated."
                )
            return "process", "process backend explicitly selected"

        available, reason = docker_is_available()
        if available:
            verified_identity: dict[str, str] = {}
            image_available, image_reason = docker_image_is_available(
                policy.docker_image,
                workspace=self.working_dir,
                verified_identity=verified_identity,
            )
            if image_available:
                image_id = verified_identity.get("image_id")
                if image_id and _DOCKER_IMAGE_ID_PATTERN.fullmatch(image_id):
                    self.verified_docker_image_id = image_id
                    return "docker", None
                image_reason = "executor verification returned no immutable image ID"
            reason = image_reason
        if policy.backend == "docker" or policy.require_isolation:
            raise SandboxUnavailableError(
                "Docker isolation was requested but is unavailable: " + str(reason)
            )
        logger.warning(
            "Docker sandbox unavailable; falling back to the non-isolated process "
            "backend. Set exec.require_isolation=true to fail closed. Reason: %s",
            reason,
        )
        return "process", reason

    def execution_metadata(self) -> dict[str, Any]:
        policy = self.sandbox_policy
        return {
            "requested_backend": policy.backend,
            "actual_backend": self.execution_backend,
            "isolated": self.execution_backend == "docker",
            "require_isolation": policy.require_isolation,
            "fallback_reason": self.isolation_reason,
            "docker_image": (
                policy.docker_image if self.execution_backend == "docker" else None
            ),
            "docker_image_id": (
                self.verified_docker_image_id
                if self.execution_backend == "docker"
                else None
            ),
            "network": policy.network if self.execution_backend == "docker" else "host",
            "memory": policy.memory if self.execution_backend == "docker" else None,
            "cpus": policy.cpus if self.execution_backend == "docker" else None,
            "pids_limit": (
                policy.pids_limit if self.execution_backend == "docker" else None
            ),
            "read_only_root": (
                policy.read_only_root if self.execution_backend == "docker" else False
            ),
            "read_only_mounts": (
                list(policy.read_only_mounts)
                if self.execution_backend == "docker"
                else []
            ),
            "max_output_chars": policy.max_output_chars,
            "max_workspace_bytes": policy.max_workspace_bytes,
            "max_workspace_files": policy.max_workspace_files,
        }

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()

        safe_process_env = {
            "PATH": os.defpath,
            "HOME": str(self.working_dir),
            "TMPDIR": str(self.working_dir / ".tmp"),
            "LANG": "C.UTF-8",
        }
        (self.working_dir / ".tmp").mkdir(parents=True, exist_ok=True)
        os.environ.clear()
        os.environ.update(safe_process_env)
        os.environ.update(self.env_vars)

        # Default cache locations for faster repeated dataset/model downloads.
        # Do not override user-provided env vars.
        try:
            from ai_scientist.config.paths import apply_cache_env_vars

            apply_cache_env_vars(override=False)
        except Exception:
            # Best-effort: cache env config should never prevent execution.
            pass

        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        self.code_inq = Queue(maxsize=4)
        self.result_outq = Queue(maxsize=256)
        self.event_outq = Queue(maxsize=16)
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _drain_queues(self):
        """Quickly drain all in-flight messages to prevent blocking."""
        while not self.result_outq.empty():
            try:
                self.result_outq.get_nowait()
            except Exception:
                break

        while not self.event_outq.empty():
            try:
                self.event_outq.get_nowait()
            except Exception:
                break

        while not self.code_inq.empty():
            try:
                self.code_inq.get_nowait()
            except Exception:
                break

    def cleanup_session(self):
        if self.process is None:
            return
        # give the child process a chance to terminate gracefully
        self.process.terminate()
        self._drain_queues()
        self.process.join(timeout=2)
        # kill the child process if it's still alive
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self.process.kill()
            self._drain_queues()
            self.process.join(timeout=2)
        # don't wait for gc, clean up immediately
        self.process.close()
        self.process = None  # type: ignore

    def _docker_command(self) -> list[str]:
        return self._docker_command_for_name(None)

    def _docker_command_for_name(self, container_name: str | None) -> list[str]:
        policy = self.sandbox_policy
        docker = shutil.which("docker")
        if docker is None:
            raise SandboxUnavailableError(
                "docker executable disappeared during execution"
            )

        command = [
            docker,
            "run",
            "--rm",
            "--init",
            "--network",
            policy.network,
            "--memory",
            policy.memory,
            "--cpus",
            str(policy.cpus),
            "--pids-limit",
            str(policy.pids_limit),
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--workdir",
            "/workspace",
            "--mount",
            f"type=bind,src={self.working_dir},dst=/workspace,rw",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=1g",
        ]
        if container_name:
            command[3:3] = ["--name", container_name]
        if policy.read_only_root:
            command.append("--read-only")
        mounted_sources = {str(self.working_dir)}
        for raw_mount in policy.read_only_mounts:
            mount_path = Path(raw_mount).expanduser().resolve()
            if not mount_path.exists():
                logger.warning(
                    "Skipping missing read-only sandbox mount: %s", mount_path
                )
                continue
            source = str(mount_path)
            if source in mounted_sources:
                continue
            mounted_sources.add(source)
            command.extend(
                [
                    "--mount",
                    f"type=bind,src={source},dst={source},readonly",
                ]
            )
        container_env = {
            "HOME": "/workspace/.home",
            "TMPDIR": "/tmp",
            "XDG_CACHE_HOME": "/workspace/.cache",
            "MPLCONFIGDIR": "/workspace/.cache/matplotlib",
            "HF_HOME": "/workspace/.cache/huggingface",
            "HF_DATASETS_CACHE": "/workspace/.cache/huggingface/datasets",
            "HF_HUB_CACHE": "/workspace/.cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/workspace/.cache/huggingface/transformers",
            "TORCH_HOME": "/workspace/.cache/torch",
        }
        container_env.update(self.env_vars)
        for key, value in sorted(container_env.items()):
            command.extend(["--env", f"{key}={value}"])
        image_id = self.verified_docker_image_id
        if image_id is None:
            raise SandboxUnavailableError(
                "verified Docker image identity is unavailable"
            )
        command.extend([image_id, "python", "-u", f"/workspace/{self.agent_file_name}"])
        return command

    def _force_remove_container(self, container_name: str) -> None:
        docker = shutil.which("docker")
        if docker is None:
            return
        try:
            subprocess.run(
                [docker, "rm", "-f", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.warning(
                "Failed to remove timed-out Docker container %s", container_name
            )

    def _run_docker(self, code: str) -> ExecutionResult:
        for directory in (
            self.working_dir / ".home",
            self.working_dir / ".cache" / "matplotlib",
            self.working_dir / ".cache" / "huggingface",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        script_path = self.working_dir / self.agent_file_name
        script_path.write_text(code, encoding="utf-8")
        try:
            script_path.chmod(0o664)
        except OSError:
            pass
        started = time.monotonic()
        metadata = self.execution_metadata()
        container_name = f"xscientist-{uuid.uuid4().hex[:16]}"
        limit_check = workspace_limit_checker(
            self.working_dir,
            max_bytes=self.sandbox_policy.max_workspace_bytes,
            max_files=self.sandbox_policy.max_workspace_files,
        )
        try:
            completed = run_process_bounded(
                self._docker_command_for_name(container_name),
                cwd=self.working_dir,
                timeout=self.timeout,
                max_output_chars=self.sandbox_policy.max_output_chars,
                limit_check=limit_check,
            )
            exec_time = time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            self._force_remove_container(container_name)
            output = []
            if exc.stdout:
                output.append(str(exc.stdout))
            if exc.stderr:
                output.append(str(exc.stderr))
            output.append(
                f"TimeoutError: Execution exceeded the time limit of "
                f"{humanize.naturaldelta(self.timeout)}"
            )
            return ExecutionResult(
                term_out=output,
                exec_time=float(self.timeout),
                exc_type="TimeoutError",
                exc_info={"backend": "docker"},
                exc_stack=[],
                execution_backend="docker",
                isolation=metadata,
                output_truncated=(
                    len(str(exc.stdout or "")) >= self.sandbox_policy.max_output_chars
                    or len(str(exc.stderr or ""))
                    >= self.sandbox_policy.max_output_chars
                ),
            )
        except ProcessResourceLimitExceeded as exc:
            self._force_remove_container(container_name)
            output = [value for value in (exc.stdout, exc.stderr) if value]
            output.append(f"ResourceLimitError: {exc.reason}")
            return ExecutionResult(
                term_out=output,
                exec_time=time.monotonic() - started,
                exc_type="ResourceLimitError",
                exc_info={"backend": "docker", "reason": exc.reason},
                exc_stack=[],
                execution_backend="docker",
                isolation=metadata,
                output_truncated=(
                    len(exc.stdout) >= self.sandbox_policy.max_output_chars
                    or len(exc.stderr) >= self.sandbox_policy.max_output_chars
                ),
            )
        except OSError as exc:
            return ExecutionResult(
                term_out=[f"SandboxUnavailableError: {exc}"],
                exec_time=time.monotonic() - started,
                exc_type="SandboxUnavailableError",
                exc_info={"args": [str(exc)], "backend": "docker"},
                exc_stack=[],
                execution_backend="docker",
                isolation=metadata,
            )

        output: list[str] = []
        if completed.stdout:
            output.append(completed.stdout)
        if completed.stderr:
            output.append(completed.stderr)
        exc_type = None if completed.returncode == 0 else "ProcessExitError"
        exc_info = (
            None
            if completed.returncode == 0
            else {"returncode": completed.returncode, "backend": "docker"}
        )
        output.append(
            f"Execution time: {humanize.naturaldelta(exec_time)} seconds "
            f"(time limit is {humanize.naturaldelta(self.timeout)})."
        )
        return ExecutionResult(
            term_out=output,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=[],
            execution_backend="docker",
            isolation=metadata,
            output_truncated=(completed.stdout_truncated or completed.stderr_truncated),
        )

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(
            "Interpreter is executing code (backend=%s, reset_session=%s)",
            self.execution_backend,
            reset_session,
        )

        if self.execution_backend == "docker":
            if not reset_session:
                logger.warning(
                    "Docker backend does not preserve interactive globals; executing in a "
                    "fresh isolated container."
                )
            return self._run_docker(code)

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # wait for child to actually start execution (we don't want interrupt child setup)
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()
        output_buffer = BoundedTextBuffer(self.sandbox_policy.max_output_chars)
        workspace_check = workspace_limit_checker(
            self.working_dir,
            max_bytes=self.sandbox_policy.max_workspace_bytes,
            max_files=self.sandbox_policy.max_workspace_files,
        )
        next_workspace_check = start_time
        resource_limit_reason: str | None = None
        saw_eof = False

        def drain_output_queue() -> bool:
            saw_eof = False
            while True:
                try:
                    line = self.result_outq.get_nowait()
                except queue.Empty:
                    return saw_eof
                if line == "<|EOF|>":
                    saw_eof = True
                else:
                    output_buffer.append(str(line))

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
        # if the child process dies without this flag being set, it's an unexpected termination
        child_in_overtime = False
        timeout_grace_seconds = 60

        while True:
            saw_eof = drain_output_queue() or saw_eof
            now = time.time()
            if now >= next_workspace_check:
                try:
                    resource_limit_reason = workspace_check()
                except Exception as exc:
                    resource_limit_reason = (
                        f"workspace limit check failed: {type(exc).__name__}"
                    )
                if resource_limit_reason:
                    self.cleanup_session()
                    state = (
                        None,
                        "ResourceLimitError",
                        {"reason": resource_limit_reason},
                        [],
                    )
                    exec_time = now - start_time
                    break
                next_workspace_check = now + 0.5
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=0.1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    if not child_in_overtime:
                        try:
                            os.kill(self.process.pid, signal.SIGINT)  # type: ignore
                        except ProcessLookupError:
                            pass
                        child_in_overtime = True
                        if not reset_session:
                            logger.warning(
                                "Timeout occurred in interactive session; resetting interpreter process."
                            )
                    if running_time > self.timeout + timeout_grace_seconds:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break
                    if child_in_overtime and not self.process.is_alive():
                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        if resource_limit_reason is None:
            resource_limit_reason = workspace_check()
            if resource_limit_reason:
                self.cleanup_session()
                state = (
                    None,
                    "ResourceLimitError",
                    {"reason": resource_limit_reason},
                    [],
                )

        e_cls_name, exc_info, exc_stack = state[1:]
        while not saw_eof:
            try:
                line = self.result_outq.get(timeout=0.1)
                if line == "<|EOF|>":
                    saw_eof = True
                    break
                output_buffer.append(str(line))
            except queue.Empty:
                if e_cls_name == "TimeoutError":
                    break
                if not saw_eof and self.process is not None and self.process.is_alive():
                    continue

                break

        output_text, output_truncated = output_buffer.snapshot()
        output = [output_text] if output_text else []

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        elif e_cls_name == "ResourceLimitError":
            output.append(f"ResourceLimitError: {resource_limit_reason}")
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        return ExecutionResult(
            output,
            exec_time,
            e_cls_name,
            exc_info,
            exc_stack,
            execution_backend="process",
            isolation=self.execution_metadata(),
            output_truncated=output_truncated,
        )
