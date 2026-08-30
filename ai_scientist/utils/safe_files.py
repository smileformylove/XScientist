"""Small fail-closed helpers for reading untrusted workspace artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path


class BoundedFileError(ValueError):
    """Describe why a bounded regular-file read was rejected."""

    def __init__(self, label: str, reason: str) -> None:
        self.label = label
        self.reason = reason
        self.code = f"{label}_{reason}"
        super().__init__(self.code)


def _stable_file_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_bounded_regular_file(
    path: str | os.PathLike[str],
    *,
    maximum: int,
    label: str,
) -> bytes:
    """Read a bounded regular file without following its final path component.

    The descriptor identity and mutable metadata are checked both before and
    after the read. This closes the common ``lstat``/path-open race and rejects
    content that changes while the verifier is consuming it.
    """

    if not isinstance(maximum, int) or maximum < 0:
        raise ValueError("maximum must be a non-negative integer")
    target = Path(path)
    try:
        metadata = target.lstat()
    except FileNotFoundError as exc:
        raise BoundedFileError(label, "missing") from exc
    except OSError as exc:
        raise BoundedFileError(label, "unreadable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise BoundedFileError(label, "symlink_rejected")
    if not stat.S_ISREG(metadata.st_mode):
        raise BoundedFileError(label, "not_regular")
    if metadata.st_nlink != 1:
        raise BoundedFileError(label, "hardlink_rejected")
    if metadata.st_size > maximum:
        raise BoundedFileError(label, "too_large")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise BoundedFileError(label, "changed_during_read") from exc
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise BoundedFileError(label, "not_regular")
            if _stable_file_state(before) != _stable_file_state(metadata):
                raise BoundedFileError(label, "changed_during_read")
            if before.st_size > maximum:
                raise BoundedFileError(label, "too_large")

            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65_536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > maximum:
                raise BoundedFileError(label, "too_large")
            after = os.fstat(descriptor)
            if _stable_file_state(after) != _stable_file_state(before):
                raise BoundedFileError(label, "changed_during_read")
            if len(payload) != after.st_size:
                raise BoundedFileError(label, "changed_during_read")
            return payload
        except BoundedFileError:
            raise
        except OSError as exc:
            raise BoundedFileError(label, "unreadable") from exc
    finally:
        os.close(descriptor)
