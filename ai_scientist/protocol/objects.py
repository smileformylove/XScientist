"""Content-addressable object store for ARA artifacts.

Anything we want to reference immutably — LLM prompts, model responses,
tool schemas, large prompt fragments, mirrored pipeline artifacts — goes
through :meth:`ObjectStore.put_bytes` (or the JSON/text convenience
wrappers) and is retrieved through :meth:`ObjectStore.get_bytes`. Callers
store only the returned :class:`ObjectRef` (hash + size + gzip flag), never
inline payloads.

Layout on disk::

    <ara_root>/objects/sha256/<hex[:2]>/<hex[2:]>

Two-level sharding keeps ``ls`` responsive even at hundreds of thousands
of blobs. Payloads at or above :data:`_GZIP_THRESHOLD` bytes are stored
gzipped; the on-disk file's first two bytes (``1f 8b``) tell readers which
form was written, so we never need a separate metadata sidecar.

The store is intentionally append-only. ARA artifacts are commit-like:
once written, an object is expected to live for the lifetime of the ARA.
Garbage collection, if it ever ships, will be a separate tool that walks
manifests and prunes unreferenced blobs — not something callers can trip
into by accident.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import CONTENT_HASH_ALGO

# Payloads at or above this size get gzipped on disk. Prompts are typically
# a few KB to tens of KB — compression buys a lot; below the threshold the
# CPU cost outweighs the savings.
_GZIP_THRESHOLD = 4 * 1024  # 4 KiB


@dataclass(frozen=True)
class ObjectRef:
    """Stable pointer to a blob inside an :class:`ObjectStore`.

    ``hash`` uses the ``"<algo>:<hex>"`` shape so downstream consumers can
    detect algorithm mismatches before comparing raw digests. ``size`` is the
    ORIGINAL (pre-gzip) byte count — matching what a consumer will get back
    from :meth:`ObjectStore.get_bytes`. ``gzip`` is bookkeeping only: readers
    detect compression from the file's magic bytes.
    """

    hash: str
    size: int
    gzip: bool

    def to_json(self) -> dict[str, Any]:
        """Serialise into the shape used by ``ObjectRef`` schema references."""
        return {"hash": self.hash, "size": self.size, "gzip": self.gzip}


class ObjectStore:
    """Append-only content-addressable store under ``<ara_root>/objects/``.

    Construction is cheap: it just resolves the target directory and ensures
    it exists. Every ``put_*`` call is idempotent — writing the same bytes
    twice returns the same hash and does not touch the disk on the second
    call. Writes are atomic (``.tmp`` → ``os.replace``) so an interrupted
    process cannot leave half-written blobs.
    """

    def __init__(
        self,
        ara_root: str | os.PathLike[str],
        *,
        shared_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(ara_root) / "objects" / CONTENT_HASH_ALGO
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared_root = (
            Path(shared_root) / "objects" / CONTENT_HASH_ALGO
            if shared_root is not None
            else None
        )
        if self.shared_root is not None:
            self.shared_root.mkdir(parents=True, exist_ok=True)

    # ---- internal helpers ------------------------------------------------

    def _path_for(self, hex_digest: str) -> Path:
        return self.root / hex_digest[:2] / hex_digest[2:]

    def _shared_path_for(self, hex_digest: str) -> Path | None:
        if self.shared_root is None:
            return None
        return self.shared_root / hex_digest[:2] / hex_digest[2:]

    @staticmethod
    def _write_once(target: Path, payload: bytes) -> None:
        """Atomically materialise one encoded object if it is absent."""

        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(payload)
        try:
            os.replace(tmp, target)
        except FileNotFoundError:
            if not target.exists():
                raise
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        """Create a local portable view, sharing blocks when possible."""

        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            # Cross-device filesystems and some network mounts do not support
            # hard links.  Copying preserves the old self-contained ARA
            # contract; profile-aware bundles can still deduplicate later.
            shutil.copy2(source, target)

    # ---- writes ----------------------------------------------------------

    def put_bytes(self, data: bytes) -> ObjectRef:
        """Store raw bytes and return a stable :class:`ObjectRef`.

        Idempotent: identical inputs produce the same on-disk file and the
        same hash. When the target file already exists we skip the write
        entirely (the content is guaranteed identical by construction).

        Concurrency: two threads or processes writing the same content race
        to the atomic ``os.replace`` on the same target path. We use a
        per-call unique tmp filename so no thread can steal another's tmp
        file mid-write; the loser simply sees the winner's file already
        present and moves on.
        """
        digest = hashlib.new(CONTENT_HASH_ALGO, data).hexdigest()
        target = self._path_for(digest)
        shared_target = self._shared_path_for(digest)
        gzipped = len(data) >= _GZIP_THRESHOLD
        if not target.exists() or (
            shared_target is not None and not shared_target.exists()
        ):
            payload = gzip.compress(data) if gzipped else data
            if shared_target is not None:
                self._write_once(shared_target, payload)
                self._link_or_copy(shared_target, target)
            else:
                self._write_once(target, payload)
        return ObjectRef(
            hash=f"{CONTENT_HASH_ALGO}:{digest}", size=len(data), gzip=gzipped
        )

    def put_text(self, text: str) -> ObjectRef:
        """UTF-8 encode ``text`` and delegate to :meth:`put_bytes`."""
        return self.put_bytes((text or "").encode("utf-8"))

    def put_json(self, obj: Any) -> ObjectRef:
        """Canonically-serialise ``obj`` (sorted keys, no whitespace) and store.

        The canonical form mirrors :func:`ai_scientist.protocol.hashing.content_hash`
        so a payload that gets hashed one way and stored another still ends
        up at the same address.
        """
        data = json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return self.put_bytes(data)

    # ---- reads -----------------------------------------------------------

    def _resolve(self, ref: str) -> Path:
        if ":" not in ref:
            raise ValueError(f"object ref missing algo prefix: {ref!r}")
        algo, digest = ref.split(":", 1)
        if algo != CONTENT_HASH_ALGO:
            raise ValueError(f"algo mismatch: expected {CONTENT_HASH_ALGO}, got {algo}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                "object ref digest must be 64 lowercase hexadecimal characters"
            )
        return self._path_for(digest)

    def exists(self, ref: str) -> bool:
        """True iff the blob at ``ref`` is on disk in this store."""
        return self._resolve(ref).exists()

    def get_bytes(self, ref: str) -> bytes:
        """Read, decompress, and verify the blob referenced by ``ref``."""
        path = self._resolve(ref)
        raw = path.read_bytes()
        # gzip magic bytes; guarantees no false positive against raw text
        data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        actual = hashlib.new(CONTENT_HASH_ALGO, data).hexdigest()
        expected = ref.split(":", 1)[1]
        if actual != expected:
            raise ValueError(f"object content hash mismatch: {ref}")
        return data

    def get_text(self, ref: str) -> str:
        return self.get_bytes(ref).decode("utf-8")

    def get_json(self, ref: str) -> Any:
        return json.loads(self.get_bytes(ref).decode("utf-8"))


__all__ = ["ObjectStore", "ObjectRef"]
