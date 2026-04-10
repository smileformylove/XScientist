from __future__ import annotations

import importlib
from functools import lru_cache


@lru_cache(maxsize=None)
def load_module(module_name: str):
    return importlib.import_module(module_name)


@lru_cache(maxsize=None)
def load_module_attr(module_name: str, attr_name: str):
    return getattr(load_module(module_name), attr_name)
