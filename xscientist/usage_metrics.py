"""Strictly local, explicit opt-in, payload-free usage counters."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_scientist.utils.atomic_io import atomic_write_text

from ._version import __version__

ALLOWED_EVENTS = frozenset(
    {"demo", "doctor", "start", "status", "upgrade_check", "conformance_check"}
)
ALLOWED_STATUSES = frozenset({"ok", "error", "cancelled"})


def _data_dir() -> Path:
    explicit = str(os.environ.get("XSCIENTIST_METRICS_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = str(os.environ.get("XDG_DATA_HOME") or "").strip()
    if xdg:
        return Path(xdg).expanduser().resolve() / "xscientist"
    return Path.home().resolve() / ".local" / "share" / "xscientist"


def _enabled_path() -> Path:
    return _data_dir() / "usage-metrics.enabled"


def _events_path() -> Path:
    return _data_dir() / "usage-metrics.jsonl"


def metrics_enabled() -> bool:
    env = str(os.environ.get("XSCIENTIST_USAGE_METRICS") or "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return _enabled_path().is_file()


def set_metrics_enabled(enabled: bool) -> dict[str, Any]:
    path = _enabled_path()
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, "enabled\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    else:
        path.unlink(missing_ok=True)
    return metrics_status()


def _duration_bucket(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    value = max(0.0, float(seconds))
    if value < 1:
        return "lt_1s"
    if value < 10:
        return "1_10s"
    if value < 60:
        return "10_60s"
    if value < 600:
        return "1_10m"
    return "gte_10m"


def record_event(
    event: str, *, status: str, duration_seconds: float | None = None
) -> bool:
    """Append a fixed-shape event; arbitrary metadata is intentionally unsupported."""

    if event not in ALLOWED_EVENTS:
        raise ValueError(f"unsupported local metric event: {event}")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"unsupported local metric status: {status}")
    if not metrics_enabled():
        return False
    payload = {
        "schema": "xscientist.local-usage-event.v1",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "version": __version__,
        "event": event,
        "status": status,
        "duration_bucket": _duration_bucket(duration_seconds),
    }
    path = _events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return True


def metrics_status() -> dict[str, Any]:
    path = _events_path()
    count = 0
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
        except OSError:
            count = 0
    return {
        "schema": "xscientist.local-usage-metrics.v1",
        "enabled": metrics_enabled(),
        "event_count": count,
        "storage": "local-only",
        "network_transmission": False,
        "collected_fields": [
            "timestamp",
            "version",
            "event",
            "status",
            "duration_bucket",
        ],
        "excluded_fields": [
            "research question",
            "file paths",
            "credentials",
            "provider/model output",
            "research artifacts",
        ],
    }


def export_metrics() -> dict[str, Any]:
    """Return local rows for user inspection; never upload them."""

    rows: list[dict[str, Any]] = []
    path = _events_path()
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
    return {**metrics_status(), "events": rows}


__all__ = [
    "ALLOWED_EVENTS",
    "export_metrics",
    "metrics_enabled",
    "metrics_status",
    "record_event",
    "set_metrics_enabled",
]
