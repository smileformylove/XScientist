from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_text(
    path: str | Path, content: str, *, encoding: str = "utf-8"
) -> None:
    """Write deterministic encoded bytes without OS newline translation."""

    atomic_write_bytes(path, content.encode(encoding))


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp_path.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    default=None,
    allow_nan: bool = True,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=default,
            allow_nan=allow_nan,
        ),
    )


def durable_append_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode(encoding)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    original_size = os.fstat(descriptor).st_size
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Failed to append artifact content")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
