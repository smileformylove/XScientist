#!/usr/bin/env python3
"""Compatibility alias for :mod:`ai_scientist.apps.daemon`.

New integrations should use ``xscientist daemon``. This module intentionally
aliases the implementation module so existing imports and ``mock.patch``
targets continue to share the real application globals.
"""

from __future__ import annotations

import importlib
import sys

_implementation = importlib.import_module("ai_scientist.apps.daemon")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
