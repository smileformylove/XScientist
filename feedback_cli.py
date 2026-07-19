#!/usr/bin/env python3
"""Compatibility alias for :mod:`ai_scientist.apps.feedback`."""

from __future__ import annotations

import importlib
import sys

_implementation = importlib.import_module("ai_scientist.apps.feedback")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
