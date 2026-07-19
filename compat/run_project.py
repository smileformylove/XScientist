#!/usr/bin/env python3
"""Compatibility alias for :mod:`ai_scientist.apps.project`.

New integrations should use ``xscientist project`` or the public Python SDK.
This module intentionally aliases the implementation module so legacy imports
and ``mock.patch('run_project...')`` continue to target the real globals.
"""

from __future__ import annotations

import importlib
import sys

_implementation = importlib.import_module("ai_scientist.apps.project")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
