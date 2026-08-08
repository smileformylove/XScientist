"""Fail-closed data contracts for automated research runs."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.atomic_io import atomic_write_json

PROJECT_DATA_ENV = "AI_SCIENTIST_PROJECT_DATA_DIR"
SYNTHETIC_DATA_ENV = "AI_SCIENTIST_ALLOW_SYNTHETIC_DATA"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _snapshot_files(
    source: Path,
    project_root: Path,
    files: list[dict[str, Any]],
) -> tuple[Path, str]:
    """Publish an immutable, content-addressed copy and detect source races."""

    snapshot_id = canonical_content_hash({"files": files})
    store = project_root / ".ara-store" / "datasets"
    destination = store / snapshot_id.removeprefix("sha256:")
    store.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=store))
        try:
            for item in files:
                relative = Path(str(item["path"]))
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / relative, target)
                if _file_hash(target) != item["sha256"]:
                    raise RuntimeError(
                        "data changed while its immutable snapshot was being created"
                    )
            try:
                temporary.replace(destination)
            except OSError:
                if not destination.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    expected_paths = {str(item["path"]) for item in files}
    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise RuntimeError("content-addressed data snapshot has an invalid file set")
    for item in files:
        if _file_hash(destination / str(item["path"])) != item["sha256"]:
            raise RuntimeError(
                "content-addressed data snapshot failed hash verification"
            )
    for path in sorted(destination.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return destination, snapshot_id


def prepare_data_contract(
    project_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    allow_synthetic: bool = False,
    required: bool = False,
) -> dict[str, Any]:
    """Validate and hash user data, or record an explicit synthetic-data choice."""

    if data_dir is not None and allow_synthetic:
        raise ValueError("--data-dir and --allow-synthetic-data are mutually exclusive")
    root = Path(project_dir).expanduser().resolve()
    os.environ.pop(PROJECT_DATA_ENV, None)
    os.environ.pop(SYNTHETIC_DATA_ENV, None)

    files: list[dict[str, Any]] = []
    mode = "unresolved"
    if data_dir is not None:
        source = Path(data_dir).expanduser().resolve()
        if not source.is_dir():
            raise ValueError("--data-dir must point to an existing directory")
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(
                    "--data-dir contains a symbolic link; stage explicit files instead"
                )
            if not path.is_file():
                continue
            files.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_hash(path),
                }
            )
        if not files:
            raise ValueError("--data-dir contains no regular files")
        snapshot, snapshot_id = _snapshot_files(source, root, files)
        mode = "content_addressed_snapshot_read_only"
        os.environ[PROJECT_DATA_ENV] = str(snapshot)
    elif allow_synthetic:
        snapshot_id = None
        mode = "synthetic_explicit"
        os.environ[SYNTHETIC_DATA_ENV] = "1"
    elif required:
        raise RuntimeError(
            "Autopilot data gate stopped before model calls: provide an immutable "
            "input directory with `--data-dir PATH`, or explicitly permit a "
            "synthetic/computational study with `--allow-synthetic-data`."
        )
    else:
        snapshot_id = None

    payload: dict[str, Any] = {
        "schema_version": "xscientist.data-contract.v1",
        "mode": mode,
        "ready": mode != "unresolved",
        "source_path_disclosed": False,
        "snapshot_id": snapshot_id,
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "scientific_boundary": (
            "Synthetic results are exploratory and cannot become independently "
            "verified empirical claims without external evidence."
            if mode == "synthetic_explicit"
            else (
                "Experiments consume a content-addressed read-only snapshot; source "
                "changes after this gate cannot alter the recorded run."
                if mode == "content_addressed_snapshot_read_only"
                else "No automated data contract was selected for this manual run."
            )
        ),
    }
    payload["manifest_hash"] = canonical_content_hash(payload)
    atomic_write_json(
        root / "00_config" / "data_manifest.json",
        payload,
        indent=2,
        ensure_ascii=False,
    )
    return payload


__all__ = [
    "PROJECT_DATA_ENV",
    "SYNTHETIC_DATA_ENV",
    "prepare_data_contract",
]
