"""Stable, opt-in adapter boundary for external research platforms and tools."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.privacy import redact_sensitive_payload
from jsonschema import ValidationError, validate as validate_json

from .research_git import ResearchGitError
from .research_interop import INTEROP_FORMATS, export_research_interop

ADAPTER_API_VERSION = "1.0"
ADAPTER_ENTRYPOINT_GROUP = "xscientist.research_adapters"
ADAPTER_RECEIPT_SCHEMA = "xscientist.research-adapter-receipt.v1"


@dataclass(frozen=True)
class ResearchAdapterDescriptor:
    name: str
    version: str
    description: str
    capabilities: tuple[str, ...]
    destination_kinds: tuple[str, ...]
    source: str = "builtin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "api_version": ADAPTER_API_VERSION,
            "version": self.version,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "destination_kinds": list(self.destination_kinds),
            "source": self.source,
        }


@runtime_checkable
class ResearchAdapter(Protocol):
    descriptor: ResearchAdapterDescriptor

    def probe(self) -> Mapping[str, Any]: ...

    def publish(
        self,
        package_root: Path,
        destination: str,
        *,
        options: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _safe_directory_destination(value: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ResearchGitError("filesystem adapter destination cannot be empty")
    target = Path(raw).expanduser().resolve()
    if target in {Path(target.anchor), Path.home().resolve()}:
        raise ResearchGitError("filesystem adapter refuses a broad destination")
    if target.exists():
        raise ResearchGitError("adapter destination already exists")
    if target.parent.exists() and target.parent.is_symlink():
        raise ResearchGitError("adapter destination parent may not be a symbolic link")
    return target


class FilesystemResearchAdapter:
    """Atomically publish an interoperable research package to a directory."""

    descriptor = ResearchAdapterDescriptor(
        name="filesystem",
        version="1.0",
        description="Copy a hash-bound research exchange package to a local or mounted directory.",
        capabilities=("publish", "offline", "atomic"),
        destination_kinds=("directory", "mounted-volume", "network-share"),
    )

    def probe(self) -> Mapping[str, Any]:
        return {
            "ok": True,
            "adapter": self.descriptor.to_dict(),
            "requirements": [],
        }

    def publish(
        self,
        package_root: Path,
        destination: str,
        *,
        options: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del options
        target = _safe_directory_destination(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.xscientist-", dir=target.parent)
        )
        shutil.rmtree(staging)
        try:
            shutil.copytree(package_root, staging, symlinks=False)
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {
            "status": "published",
            "destination_kind": "directory",
            "destination_name": target.name,
            "file_count": sum(1 for item in target.rglob("*") if item.is_file()),
        }


_BUILTINS: dict[str, type[ResearchAdapter]] = {
    "filesystem": FilesystemResearchAdapter,
}


def _entry_points() -> dict[str, metadata.EntryPoint]:
    discovered = metadata.entry_points()
    selected = (
        discovered.select(group=ADAPTER_ENTRYPOINT_GROUP)
        if hasattr(discovered, "select")
        else discovered.get(ADAPTER_ENTRYPOINT_GROUP, ())
    )
    return {entry.name: entry for entry in selected}


def available_research_adapters() -> list[dict[str, Any]]:
    """List adapter names without importing third-party plugin code."""

    rows = [adapter().descriptor.to_dict() for adapter in _BUILTINS.values()]
    for name, entry in sorted(_entry_points().items()):
        if name in _BUILTINS:
            continue
        distribution = getattr(entry, "dist", None)
        rows.append(
            {
                "name": name,
                "api_version": None,
                "version": getattr(distribution, "version", None),
                "description": "Third-party adapter; inspect explicitly with adapter doctor.",
                "capabilities": [],
                "destination_kinds": [],
                "source": (f"entry-point:{getattr(distribution, 'name', 'unknown')}"),
            }
        )
    return sorted(rows, key=lambda item: str(item["name"]))


def validate_research_adapter(adapter: Any, *, expected_name: str) -> ResearchAdapter:
    """Validate the public cross-package adapter contract without publishing."""

    descriptor = getattr(adapter, "descriptor", None)
    if not isinstance(descriptor, ResearchAdapterDescriptor):
        raise ResearchGitError("adapter descriptor has the wrong type")
    if descriptor.name != expected_name:
        raise ResearchGitError("adapter descriptor name does not match entry point")
    if not descriptor.version or not descriptor.capabilities:
        raise ResearchGitError("adapter descriptor is incomplete")
    if not callable(getattr(adapter, "probe", None)) or not callable(
        getattr(adapter, "publish", None)
    ):
        raise ResearchGitError("adapter must implement probe() and publish()")
    return adapter


def load_research_adapter(name: str) -> ResearchAdapter:
    normalized = str(name or "").strip()
    if normalized in _BUILTINS:
        return validate_research_adapter(
            _BUILTINS[normalized](), expected_name=normalized
        )
    entry = _entry_points().get(normalized)
    if entry is None:
        raise ResearchGitError(f"research adapter not found: {normalized}")
    try:
        loaded = entry.load()
        adapter = loaded() if isinstance(loaded, type) else loaded
    except Exception as exc:
        raise ResearchGitError(
            f"research adapter could not be loaded: {normalized}"
        ) from exc
    return validate_research_adapter(adapter, expected_name=normalized)


def _probe_research_adapter(adapter: ResearchAdapter) -> dict[str, Any]:
    try:
        probe = dict(adapter.probe())
    except Exception as exc:
        return {
            "ok": False,
            "adapter": adapter.descriptor.to_dict(),
            "errors": [f"probe_failed:{type(exc).__name__}"],
        }
    errors = [str(item) for item in probe.get("errors") or []]
    return {
        "ok": probe.get("ok") is True and not errors,
        "adapter": adapter.descriptor.to_dict(),
        "requirements": list(probe.get("requirements") or []),
        "errors": errors,
    }


def doctor_research_adapter(name: str) -> dict[str, Any]:
    return _probe_research_adapter(load_research_adapter(name))


def sync_research_repository(
    repo: str | Path,
    *,
    adapter_name: str,
    destination: str,
    ref: str = "HEAD",
    formats: Sequence[str] = INTEROP_FORMATS,
    include_payloads: bool = False,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export once, then explicitly invoke a selected platform adapter."""

    adapter = load_research_adapter(adapter_name)
    probe = _probe_research_adapter(adapter)
    if not probe["ok"]:
        raise ResearchGitError(
            "research adapter is not ready: " + ", ".join(probe["errors"])
        )
    with tempfile.TemporaryDirectory(prefix="xscientist-adapter-") as raw:
        package = Path(raw) / "exchange"
        exported = export_research_interop(
            repo,
            package,
            ref=ref,
            formats=formats,
            include_payloads=include_payloads,
        )
        try:
            published = dict(
                adapter.publish(
                    package,
                    destination,
                    options=dict(options or {}),
                )
            )
        except ResearchGitError:
            raise
        except Exception as exc:
            raise ResearchGitError(
                f"research adapter publish failed: {adapter_name}"
            ) from exc
    if redact_sensitive_payload(published) != published:
        raise ResearchGitError(
            "research adapter result contains sensitive data and was refused"
        )
    base = {
        "schema_version": ADAPTER_RECEIPT_SCHEMA,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "adapter": adapter.descriptor.to_dict(),
        "ref": ref,
        "export_manifest_hash": exported["export_hash"],
        "include_payloads": bool(include_payloads),
        "result": published,
    }
    try:
        receipt = {**base, "receipt_hash": canonical_content_hash(base)}
    except (TypeError, ValueError) as exc:
        raise ResearchGitError("adapter result must be canonical JSON data") from exc
    try:
        validate_json(receipt, load_schema("research_adapter_receipt"))
    except ValidationError as exc:  # pragma: no cover - implementation contract
        raise ResearchGitError(
            f"adapter produced an invalid receipt: {exc.message}"
        ) from exc
    return receipt


__all__ = [
    "ADAPTER_API_VERSION",
    "ADAPTER_ENTRYPOINT_GROUP",
    "ADAPTER_RECEIPT_SCHEMA",
    "FilesystemResearchAdapter",
    "ResearchAdapter",
    "ResearchAdapterDescriptor",
    "available_research_adapters",
    "doctor_research_adapter",
    "load_research_adapter",
    "sync_research_repository",
    "validate_research_adapter",
]
