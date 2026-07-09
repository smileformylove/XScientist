"""git-style refs for ARAs.

A ref is a short human-readable name pointing at a content hash. Stored
under ``<ara>/refs/<name>`` as a single line containing the hash. Refs
never affect the ARA's content_hash or manifest — they're local
bookmarks the user or CI can update freely.

Common uses:
- ``refs/candidates/best`` → the node hash the manuscript chose
- ``refs/ideas/paper_v3`` → the latest ARA hash for an idea
- ``refs/pareto/frontier_2026Q2`` → a curated pareto pick

We use nested directories: passing ``candidates/best`` writes
``<ara>/refs/candidates/best``. The name is normalised to prevent
directory traversal (``..`` / absolute paths / leading slashes).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REFS_SUBDIR = "refs"

_NAME_SEGMENT = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._\-]*$")
_HASH_SHAPE = re.compile(r"^[a-z0-9]+:[0-9a-f]{6,}$")


@dataclass(frozen=True)
class Ref:
    name: str
    target: str  # content_hash-shaped value ("sha256:<hex>")


class RefError(ValueError):
    """Raised for invalid ref names or malformed hash targets."""


def refs_dir(ara_root: str | os.PathLike[str]) -> Path:
    return Path(ara_root) / REFS_SUBDIR


def _validate_name(name: str) -> tuple[str, ...]:
    """Split + validate a ref name; return the path segments to use.

    Guards against absolute paths, ``..``, empty segments, and weird
    characters. This is the entire security boundary for refs — do not
    weaken it.
    """
    if not isinstance(name, str) or not name:
        raise RefError("ref name must be a non-empty string")
    if name.startswith("/") or name.startswith("\\"):
        raise RefError("ref name must not be absolute")
    parts = tuple(seg for seg in name.replace("\\", "/").split("/") if seg)
    if not parts:
        raise RefError("ref name has no valid segments")
    for seg in parts:
        if seg in (".", ".."):
            raise RefError(f"ref name segment {seg!r} not allowed")
        if not _NAME_SEGMENT.match(seg):
            raise RefError(
                f"ref name segment {seg!r} contains disallowed characters; "
                f"allowed: letters, digits, . _ -"
            )
    return parts


def _validate_target(target: str) -> None:
    if not isinstance(target, str) or not _HASH_SHAPE.match(target):
        raise RefError(
            f"ref target {target!r} must look like '<algo>:<hex>' (e.g. sha256:abc123…)"
        )


def _ref_path(ara_root: Path, name: str) -> Path:
    parts = _validate_name(name)
    return ara_root / REFS_SUBDIR / Path(*parts)


def set_ref(ara_root: str | os.PathLike[str], name: str, target: str) -> Path:
    """Create or update a ref. Atomic (via tmp-then-replace)."""
    _validate_target(target)
    root = Path(ara_root)
    dest = _ref_path(root, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(target + "\n", encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def get_ref(ara_root: str | os.PathLike[str], name: str) -> str | None:
    """Resolve a ref to its content hash. Returns None if the ref is absent."""
    root = Path(ara_root)
    path = _ref_path(root, name)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return raw or None


def delete_ref(ara_root: str | os.PathLike[str], name: str) -> bool:
    """Remove a ref; return True iff it existed and was deleted."""
    root = Path(ara_root)
    path = _ref_path(root, name)
    if not path.exists():
        return False
    try:
        path.unlink()
    except OSError:
        return False
    _cleanup_empty_dirs(path.parent, stop_at=root / REFS_SUBDIR)
    return True


def list_refs(ara_root: str | os.PathLike[str]) -> list[Ref]:
    """Yield every ref beneath <ara>/refs/, sorted by name."""
    root = Path(ara_root)
    base = root / REFS_SUBDIR
    if not base.exists():
        return []
    out: list[Ref] = []
    for path in _iter_ref_files(base):
        name = "/".join(path.relative_to(base).parts)
        try:
            target = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not target:
            continue
        out.append(Ref(name=name, target=target))
    return sorted(out, key=lambda r: r.name)


def _iter_ref_files(base: Path) -> Iterator[Path]:
    for p in sorted(base.rglob("*")):
        # Skip our own tmp files and directories.
        if p.is_dir():
            continue
        if p.name.endswith(".tmp"):
            continue
        yield p


def _cleanup_empty_dirs(dir_: Path, *, stop_at: Path) -> None:
    """Remove empty ancestor directories up to (but not including) ``stop_at``.

    Keeps the refs tree tidy after a delete removes the last ref in a subdir.
    """
    cur = dir_.resolve()
    stop = stop_at.resolve()
    while cur != stop and cur.exists():
        try:
            cur.rmdir()  # only removes if empty
        except OSError:
            return
        cur = cur.parent


__all__ = [
    "REFS_SUBDIR",
    "Ref",
    "RefError",
    "delete_ref",
    "get_ref",
    "list_refs",
    "refs_dir",
    "set_ref",
]
