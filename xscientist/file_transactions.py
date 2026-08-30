"""Fail-closed file identity and rollback primitives for workspace transactions."""

from __future__ import annotations

import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManagedFileState:
    """One descriptor-verified regular-file state used as a rollback CAS token."""

    content: bytes
    mode: int
    device: int
    inode: int
    link_count: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _state_matches_metadata(state: ManagedFileState, metadata: os.stat_result) -> bool:
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_dev == state.device
        and metadata.st_ino == state.inode
        and metadata.st_nlink == state.link_count
        and metadata.st_size == state.size
        and metadata.st_mtime_ns == state.mtime_ns
        and metadata.st_ctime_ns == state.ctime_ns
        and metadata.st_mode & 0o7777 == state.mode
    )


def _same_open_file(first: os.stat_result, second: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_nlink",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(first, field) == getattr(second, field) for field in fields)


def _same_state_after_rename(
    expected: ManagedFileState,
    displaced: ManagedFileState | None,
) -> bool:
    """Compare a quarantined inode while allowing rename-updated ctime."""

    return bool(
        displaced is not None
        and displaced.content == expected.content
        and displaced.mode == expected.mode
        and displaced.device == expected.device
        and displaced.inode == expected.inode
        and displaced.link_count == expected.link_count
        and displaced.size == expected.size
        and displaced.mtime_ns == expected.mtime_ns
    )


def _open_parent(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.parent, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise OSError("managed workspace parent is not a directory")
    return descriptor, metadata


def _parent_still_linked(path: Path, expected: os.stat_result) -> bool:
    try:
        current = path.parent.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISDIR(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
    )


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _write_descriptor(descriptor: int, content: bytes) -> None:
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:  # pragma: no cover - operating-system invariant guard
            raise OSError("managed workspace rollback write made no progress")
        remaining = remaining[written:]


def _capture_at(parent_descriptor: int, name: str) -> ManagedFileState | None:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("managed workspace leaf is not a regular file")
        content = _read_descriptor(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise OSError("managed workspace leaf changed while being read") from exc
    if not _same_open_file(before, after):
        raise OSError("managed workspace leaf changed while being read")
    if (
        not stat.S_ISREG(linked.st_mode)
        or linked.st_dev != after.st_dev
        or linked.st_ino != after.st_ino
    ):
        raise OSError("managed workspace leaf changed while being read")
    if len(content) != after.st_size:
        raise OSError("managed workspace leaf size changed while being read")
    return ManagedFileState(
        content=content,
        mode=after.st_mode & 0o7777,
        device=after.st_dev,
        inode=after.st_ino,
        link_count=after.st_nlink,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def capture_managed_file_state(path: str | os.PathLike[str]) -> ManagedFileState | None:
    """Capture bytes and full file identity without following the leaf or parent."""

    target = Path(path)
    try:
        parent_descriptor, parent_metadata = _open_parent(target)
    except FileNotFoundError:
        return None
    try:
        state = _capture_at(parent_descriptor, target.name)
        if not _parent_still_linked(target, parent_metadata):
            raise OSError("managed workspace parent changed while being read")
        return state
    finally:
        os.close(parent_descriptor)


def _linked_leaf_matches(
    parent_descriptor: int,
    name: str,
    metadata: os.stat_result,
) -> bool:
    try:
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        return False
    return bool(
        stat.S_ISREG(linked.st_mode)
        and linked.st_dev == metadata.st_dev
        and linked.st_ino == metadata.st_ino
    )


def _restore_existing_file(
    path: Path,
    parent_descriptor: int,
    parent_metadata: os.stat_result,
    expected: ManagedFileState,
    original: ManagedFileState,
) -> bool:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except OSError:
        return False
    try:
        before = os.fstat(descriptor)
        if not _state_matches_metadata(expected, before):
            return False
        if _read_descriptor(descriptor) != expected.content:
            return False
        after_read = os.fstat(descriptor)
        if not _same_open_file(before, after_read):
            return False
        if not _linked_leaf_matches(parent_descriptor, path.name, after_read):
            return False
        if expected.content == original.content and expected.mode == original.mode:
            return _parent_still_linked(path, parent_metadata)
        if after_read.st_nlink != 1:
            return False
        _write_descriptor(descriptor, original.content)
        os.fchmod(descriptor, original.mode)
        os.fsync(descriptor)
        restored = os.fstat(descriptor)
        return bool(
            _linked_leaf_matches(parent_descriptor, path.name, restored)
            and _parent_still_linked(path, parent_metadata)
        )
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _restore_displaced_leaf(
    parent_descriptor: int,
    quarantine_name: str,
    destination_name: str,
) -> None:
    """Best-effort no-clobber restoration after a rename exposed a race."""

    try:
        metadata = os.stat(
            quarantine_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISREG(metadata.st_mode):
            os.link(
                quarantine_name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(quarantine_name, dir_fd=parent_descriptor)
            os.symlink(target, destination_name, dir_fd=parent_descriptor)
        else:
            return
        os.unlink(quarantine_name, dir_fd=parent_descriptor)
    except OSError:
        return


def _remove_owned_file(
    path: Path,
    parent_descriptor: int,
    parent_metadata: os.stat_result,
    expected: ManagedFileState,
) -> bool:
    try:
        if _capture_at(parent_descriptor, path.name) != expected:
            return False
    except OSError:
        return False
    quarantine_name = f".{path.name}.rollback-{uuid.uuid4().hex}.tmp"
    try:
        os.rename(
            path.name,
            quarantine_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
    except OSError:
        return False
    try:
        displaced = _capture_at(parent_descriptor, quarantine_name)
    except OSError:
        displaced = None
    if not _same_state_after_rename(expected, displaced):
        _restore_displaced_leaf(
            parent_descriptor,
            quarantine_name,
            path.name,
        )
        return False
    try:
        os.unlink(quarantine_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except OSError:
        return False
    return _parent_still_linked(path, parent_metadata)


def _create_original_file(
    path: Path,
    parent_descriptor: int,
    parent_metadata: os.stat_result,
    original: ManagedFileState,
) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            path.name,
            flags,
            original.mode or 0o600,
            dir_fd=parent_descriptor,
        )
    except OSError:
        return False
    try:
        _write_descriptor(descriptor, original.content)
        os.fchmod(descriptor, original.mode)
        os.fsync(descriptor)
        restored = os.fstat(descriptor)
        return bool(
            _linked_leaf_matches(parent_descriptor, path.name, restored)
            and _parent_still_linked(path, parent_metadata)
        )
    except OSError:
        return False
    finally:
        os.close(descriptor)


def restore_managed_file_state(
    path: str | os.PathLike[str],
    *,
    expected: ManagedFileState | None,
    original: ManagedFileState | None,
) -> bool:
    """Restore only a leaf that still has the transaction-owned post-state.

    Existing files are restored through an already verified descriptor, so a
    later path swap cannot redirect writes or chmod operations. Files created by
    the transaction are first quarantined and identity-checked before deletion.
    The function returns ``False`` on every ownership conflict and preserves the
    competing leaf whenever a no-clobber restoration is possible.
    """

    target = Path(path)
    try:
        parent_descriptor, parent_metadata = _open_parent(target)
    except FileNotFoundError:
        return expected is None and original is None
    except OSError:
        return False
    try:
        if expected is None:
            try:
                current = _capture_at(parent_descriptor, target.name)
            except OSError:
                return False
            if current is not None:
                return False
            if original is None:
                return _parent_still_linked(target, parent_metadata)
            return _create_original_file(
                target,
                parent_descriptor,
                parent_metadata,
                original,
            )
        if original is None:
            return _remove_owned_file(
                target,
                parent_descriptor,
                parent_metadata,
                expected,
            )
        return _restore_existing_file(
            target,
            parent_descriptor,
            parent_metadata,
            expected,
            original,
        )
    finally:
        os.close(parent_descriptor)
