"""Manifest immutability layer: lock, revision history, verification.

Before this module existed, ``update_manifest_claim_count`` mutated
manifest.json in place, so any downstream consumer who copied the
manifest at time T1 and re-fetched it at T2 saw a silently different
byte-string with no audit trail. The ARA claimed to be commit-like, but
its top-level pointer was not.

What ships here
---------------
* :func:`write_manifest_lock` — write ``<ara>/manifest.lock`` with the
  content_hash of the base manifest.json. Called from ``export_ara``
  and cmd_fork. The lock is the immutability anchor: anyone re-hashing
  the base manifest and getting a different digest knows the manifest
  was tampered with (or is at a later revision — see history).

  Writes are **byte-idempotent for the same manifest**: if an existing
  lock's stored ``manifest_hash`` matches the freshly computed one, the
  file is left untouched (bytes and mtime unchanged), so re-exports
  don't spuriously rewrite ``created_at``. A malformed existing lock is
  replaced rather than raised on.

* :func:`append_manifest_revision` — replaces in-place mutation. It:
    1. Reads the current manifest.json.
    2. Copies it to ``<ara>/history/<current_hash>.json`` if not already
       there (idempotent — same hash → same path).
    3. Applies the caller's mutations.
    4. Writes the new manifest.json.
    5. Appends a ``manifest_revision`` row to
       ``<ara>/manifest.history.jsonl``.
  The base manifest is always recoverable via manifest.lock → history/,
  and every intermediate state is auditable.

* :func:`verify_manifest_lock` — re-hash the on-disk manifest.json and
  check against manifest.lock. Returns a report explaining whether the
  current bytes match revision 0, a known later revision, or neither
  (which means real tampering).

Why not just refuse in-place writes?
------------------------------------
Legitimate reasons to update the manifest post-export exist — claim
scan finishing after ``export_ara`` returns is the canonical one. We
give those callers a documented append-only API and refuse anything
else. The three functions above are the ONLY sanctioned way to change
a manifest post-write.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from jsonschema import ValidationError

from ..protocol.hashing import hash_manifest
from ..protocol.schemas import schema_validator

logger = logging.getLogger(__name__)


MANIFEST_LOCK_NAME = "manifest.lock"
MANIFEST_HISTORY_NAME = "manifest.history.jsonl"
HISTORY_SUBDIR = "history"
_LOCK_SCHEMA_VERSION = "ara.v1"
_LOCK_HASHER_ID = "hash_manifest.v1"


# ---------------------------------------------------------------------------
# Base lock (revision 0) — written once at export time.
# ---------------------------------------------------------------------------


def write_manifest_lock(
    ara_dir: str | os.PathLike[str],
    manifest: dict[str, Any],
) -> Path:
    """Write ``<ara>/manifest.lock`` and return its path.

    Called from ``export_ara`` (and cmd_fork) after the base
    manifest.json lands on disk. The write is **byte-idempotent for
    the same manifest**: if the lock already exists and its stored
    ``manifest_hash`` equals ``hash_manifest(manifest)``, this function
    returns the existing path without touching the file — bytes and
    mtime are preserved so downstream tools can treat lock mtime as
    "when the base manifest was first stamped".

    A malformed / unparsable existing lock is replaced (not raised on).
    Callers passing a different manifest get a fresh write with a new
    ``created_at``.
    """
    ara_path = Path(ara_dir)
    lock_path = ara_path / MANIFEST_LOCK_NAME
    new_hash = hash_manifest(manifest)
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("manifest_hash") == new_hash:
                return lock_path
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug(
                "write_manifest_lock: replacing malformed lock at %s: %s",
                lock_path,
                exc,
            )
    payload = {
        "schema_version": _LOCK_SCHEMA_VERSION,
        "protocol_kind": "manifest_lock",
        "manifest_hash": new_hash,
        "created_at": _now_iso(),
        "hasher": _LOCK_HASHER_ID,
    }
    _atomic_write_json(lock_path, payload)
    return lock_path


# ---------------------------------------------------------------------------
# Revisions (revision >= 1) — append-only.
# ---------------------------------------------------------------------------


def append_manifest_revision(
    manifest_path: str | os.PathLike[str],
    mutate: Callable[[dict[str, Any]], Iterable[str] | None],
    *,
    reason: str | None = None,
    producer: str | None = None,
) -> dict[str, Any] | None:
    """Atomically evolve manifest.json without losing the old bytes.

    ``mutate`` receives a *copy* of the current manifest and mutates it
    in place. It may return an iterable of ``changed_fields`` (dotted
    paths) for the history row's audit trail; returning None is fine —
    the diff will just say "unknown".

    Behaviour
    ---------
    * If ``mutate`` doesn't actually change the payload, this is a
      no-op: no history row, no history/ file, no rewrite of
      manifest.json. Callers can be conservative without paying for it.
    * Otherwise:
        - the pre-mutation manifest is copied to
          ``history/<base_hash>.json`` (idempotent).
        - manifest.json is rewritten with the new payload.
        - one row is appended to manifest.history.jsonl.

    Returns the new manifest dict, or None if the manifest was missing
    or unreadable (matches the caller-facing contract of the old
    in-place helper).
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "append_manifest_revision: unreadable manifest %s: %s", manifest_path, exc
        )
        return None
    if not isinstance(current, dict):
        return None

    base_hash = hash_manifest(current)

    working = json.loads(json.dumps(current, default=str))  # deep copy via JSON
    hint = mutate(working)
    changed_fields = _normalise_changed_fields(hint)

    new_hash = hash_manifest(working)
    if new_hash == base_hash:
        # Nothing actually changed — callers routinely re-set the same
        # value (claim count of 0 when tex scan is disabled, e.g.).
        return working

    ara_dir = manifest_path.parent
    _archive_previous_revision(ara_dir, base_hash, current)
    _atomic_write_json(manifest_path, working)
    _append_history_row(
        ara_dir,
        base_hash=base_hash,
        new_hash=new_hash,
        changed_fields=changed_fields,
        reason=reason,
        producer=producer,
    )
    return working


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_manifest_lock(ara_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a structured report on whether manifest.json matches its lock chain.

    Result shape::

        {
          "ok": bool,
          "state": "unlocked" | "clean" | "revised" | "tampered",
          "base_hash": <sha256:...> | None,
          "current_hash": <sha256:...> | None,
          "revision_count": int,
          "detail": <human-readable note>,
        }

    ``state`` values
    ----------------
    * ``unlocked``   — manifest.lock is missing entirely. We can't judge.
    * ``clean``      — hash(manifest.json) == base hash from lock.
                       This is a fresh ARA that no one has touched.
    * ``revised``    — hash(manifest.json) differs from base BUT equals
                       the latest ``new_hash`` in manifest.history.jsonl.
                       Legitimate post-export update.
    * ``tampered``   — hash(manifest.json) matches neither base nor any
                       historical new_hash. Someone edited the file
                       outside the append-only API.
    """
    ara_path = Path(ara_dir)
    manifest_path = ara_path / "manifest.json"
    lock_path = ara_path / MANIFEST_LOCK_NAME

    if not manifest_path.exists():
        return _report(False, "unlocked", None, None, 0, "manifest.json missing")
    if not lock_path.exists():
        return _report(False, "unlocked", None, None, 0, "manifest.lock missing")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _report(False, "unlocked", None, None, 0, f"unreadable: {exc}")

    if not isinstance(manifest, dict) or not isinstance(lock, dict):
        return _report(
            False, "tampered", None, None, 0, "manifest or lock is not a JSON object"
        )
    try:
        schema_validator("manifest_lock").validate(lock)
    except ValidationError as exc:
        return _report(
            False,
            "tampered",
            None,
            hash_manifest(manifest),
            0,
            f"manifest.lock is invalid: {exc.message}",
        )
    if lock.get("hasher") not in (None, _LOCK_HASHER_ID):
        return _report(
            False,
            "tampered",
            str(lock.get("manifest_hash")),
            hash_manifest(manifest),
            0,
            f"unsupported manifest hasher: {lock.get('hasher')}",
        )

    current_hash = hash_manifest(manifest)
    base_hash = str(lock["manifest_hash"])
    history, history_issues = _read_history_strict(ara_path)
    if history_issues:
        return _report(
            False,
            "tampered",
            base_hash,
            current_hash,
            len(history),
            "invalid manifest history: " + "; ".join(history_issues),
        )

    expected_base = base_hash
    chain_issues: list[str] = []
    for expected_revision, row in enumerate(history, start=1):
        if row.get("revision") != expected_revision:
            chain_issues.append(
                f"revision {expected_revision} is numbered {row.get('revision')}"
            )
        if row.get("base_hash") != expected_base:
            chain_issues.append(
                f"revision {expected_revision} does not extend the previous hash"
            )
        archive = _history_snapshot_path(ara_path, str(row.get("base_hash") or ""))
        if archive is None or not archive.is_file():
            chain_issues.append(
                f"revision {expected_revision} is missing its base snapshot"
            )
        else:
            try:
                archived = json.loads(archive.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                chain_issues.append(
                    f"revision {expected_revision} base snapshot is unreadable"
                )
            else:
                if not isinstance(archived, dict) or hash_manifest(archived) != row.get(
                    "base_hash"
                ):
                    chain_issues.append(
                        f"revision {expected_revision} base snapshot hash mismatch"
                    )
        expected_base = str(row.get("new_hash") or "")

    if chain_issues:
        return _report(
            False,
            "tampered",
            base_hash,
            current_hash,
            len(history),
            "broken manifest history chain: " + "; ".join(chain_issues),
        )
    if not history:
        if current_hash == base_hash:
            return _report(
                True, "clean", base_hash, current_hash, 0, "manifest matches revision 0"
            )
        return _report(
            False,
            "tampered",
            base_hash,
            current_hash,
            0,
            "manifest differs from revision 0 without a history row",
        )
    if current_hash == expected_base:
        return _report(
            True,
            "revised",
            base_hash,
            current_hash,
            len(history),
            f"manifest matches revision {len(history)}",
        )
    return _report(
        False,
        "tampered",
        base_hash,
        current_hash,
        len(history),
        "current manifest does not match the end of the history chain",
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    os.replace(tmp, path)


def _archive_previous_revision(ara_dir: Path, base_hash: str, payload: dict) -> None:
    hist_dir = ara_dir / HISTORY_SUBDIR
    hist_dir.mkdir(parents=True, exist_ok=True)
    # Filename is the pre-mutation content_hash so multiple mutations of the
    # same base collapse into one on-disk copy (idempotent).
    _, digest = base_hash.split(":", 1)
    archive_path = hist_dir / f"{digest}.json"
    if not archive_path.exists():
        _atomic_write_json(archive_path, payload)


def _append_history_row(
    ara_dir: Path,
    *,
    base_hash: str,
    new_hash: str,
    changed_fields: list[str],
    reason: str | None,
    producer: str | None,
) -> None:
    history_path = ara_dir / MANIFEST_HISTORY_NAME
    existing = _read_history(ara_dir)
    # max(revision)+1 rather than len+1 so a corrupt (skipped) row on
    # disk can't cause a revision number collision with a survivor.
    highest = max(
        (r["revision"] for r in existing if isinstance(r.get("revision"), int)),
        default=0,
    )
    row = {
        "schema_version": _LOCK_SCHEMA_VERSION,
        "protocol_kind": "manifest_revision",
        "revision": max(highest, len(existing)) + 1,
        "ts": _now_iso(),
        "base_hash": base_hash,
        "new_hash": new_hash,
        "changed_fields": changed_fields,
        "reason": reason,
        "producer": producer,
    }
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _read_history(ara_dir: Path) -> list[dict[str, Any]]:
    p = ara_dir / MANIFEST_HISTORY_NAME
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    # Per-line try/except — one corrupt row must not zero the count.
    # Whole-file rejection here caused verify_manifest_lock to mis-report
    # revised ARAs as 'tampered' and reset revision numbers on the next
    # append. Matches ara_log._read_revisions / cmd_history behaviour.
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_history_strict(ara_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read history without hiding malformed or schema-invalid rows."""

    path = ara_dir / MANIFEST_HISTORY_NAME
    if not path.exists():
        return [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"history is unreadable: {exc}"]
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    validator = schema_validator("manifest_revision")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"line {line_number} is invalid JSON")
            continue
        if not isinstance(row, dict):
            issues.append(f"line {line_number} is not an object")
            continue
        errors = sorted(validator.iter_errors(row), key=lambda item: item.message)
        if errors:
            issues.append(f"line {line_number}: {errors[0].message}")
            continue
        rows.append(row)
    return rows, issues


def _history_snapshot_path(ara_dir: Path, manifest_hash: str) -> Path | None:
    if not manifest_hash.startswith("sha256:"):
        return None
    digest = manifest_hash.split(":", 1)[1]
    if len(digest) != 64:
        return None
    return ara_dir / HISTORY_SUBDIR / f"{digest}.json"


def _normalise_changed_fields(hint: Iterable[str] | None) -> list[str]:
    if not hint:
        return []
    return [str(x) for x in hint]


def _report(
    ok: bool,
    state: str,
    base_hash: str | None,
    current_hash: str | None,
    revision_count: int,
    detail: str,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "state": state,
        "base_hash": base_hash,
        "current_hash": current_hash,
        "revision_count": revision_count,
        "detail": detail,
    }


__all__ = [
    "MANIFEST_LOCK_NAME",
    "MANIFEST_HISTORY_NAME",
    "HISTORY_SUBDIR",
    "append_manifest_revision",
    "verify_manifest_lock",
    "write_manifest_lock",
]
