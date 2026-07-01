"""Schema registry.

Schemas are plain JSON files under ``schemas/``. They're intentionally *not*
generated from Python dataclasses — the protocol should stay implementable by
non-Python producers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .constants import Kind

_SCHEMAS_DIR = Path(__file__).parent / "schemas"


@lru_cache(maxsize=None)
def load_schema(kind: str | Kind) -> dict:
    """Return the JSON Schema (as a dict) for a given kind.

    Raises FileNotFoundError if the kind isn't backed by a schema — this is a
    real error, not something to swallow.
    """
    key = kind.value if isinstance(kind, Kind) else str(kind)
    path = _SCHEMAS_DIR / f"{key}.schema.json"
    if not path.exists():
        raise FileNotFoundError(f"No ARA schema for kind={key!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def available_schemas() -> tuple[str, ...]:
    if not _SCHEMAS_DIR.exists():
        return ()
    return tuple(sorted(p.stem.replace(".schema", "") for p in _SCHEMAS_DIR.glob("*.schema.json")))
