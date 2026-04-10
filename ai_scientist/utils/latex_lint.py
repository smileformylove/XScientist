"""Utilities for LaTeX lint tool execution."""

from __future__ import annotations

import subprocess


def run_chktex(writeup_file: str) -> str:
    """Run chktex with stable flags and return diagnostics text."""
    command = ["chktex", writeup_file, "-q", "-n2", "-n24", "-n13", "-n1"]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "chktex command not found; skipping chktex diagnostics."
    except Exception as exc:
        return f"chktex execution failed: {exc}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        return stdout
    if stderr:
        return f"[chktex stderr]\n{stderr}"
    return "No chktex diagnostics."
