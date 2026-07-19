#!/usr/bin/env python3
"""Compatibility alias for :mod:`ai_scientist.apps.ara`."""

from __future__ import annotations

import importlib
import sys

_implementation = importlib.import_module("ai_scientist.apps.ara")


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
