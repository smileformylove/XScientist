from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from unittest import mock

import pytest

from ai_scientist.apps import daemon


def test_timeout_terminates_the_process_group_and_waits_for_reap() -> None:
    process = mock.Mock()
    process.pid = 4242
    process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd=["generator"], timeout=2),
        -signal.SIGKILL,
    ]
    process.poll.return_value = -signal.SIGKILL

    with (
        mock.patch.object(daemon.os, "name", "posix"),
        mock.patch.object(daemon.os, "killpg") as killpg,
    ):
        assert daemon._terminate_cycle_process(process, grace_seconds=0.01)

    assert killpg.call_args_list == [
        mock.call(4242, signal.SIGTERM),
        mock.call(4242, signal.SIGKILL),
    ]
    assert process.wait.call_count == 2


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
def test_timeout_escalates_a_real_sigterm_ignoring_child() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); "
                "time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert select.select([process.stdout], [], [], 3)[0]
        assert process.stdout.readline() == b"ready\n"
        assert daemon._terminate_cycle_process(process, grace_seconds=0.05)
        assert process.poll() == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
def test_timeout_kills_worker_when_group_leader_exits_on_sigterm() -> None:
    worker_script = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    leader_script = (
        "import subprocess, sys, time; "
        f"worker = subprocess.Popen([sys.executable, '-c', {worker_script!r}], "
        "stdout=subprocess.PIPE); "
        "worker.stdout.readline(); "
        "print(worker.pid, flush=True); "
        "time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader_script],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert select.select([process.stdout], [], [], 3)[0]
        worker_pid = int(process.stdout.readline())
        assert daemon._terminate_cycle_process(process, grace_seconds=0.05)
        assert process.poll() == -signal.SIGTERM
        deadline = time.monotonic() + 3
        state = ""
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(worker_pid)],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.01)
        assert not state or state.startswith("Z"), state
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
