"""Central login session guard for AI Scientist entrypoints.

All user-facing operations should call `require_login(...)` before executing
business logic.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_utc(raw: str) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _default_auth_file() -> Path:
    custom = str(os.environ.get("AI_SCIENTIST_AUTH_FILE") or "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return Path.home().joinpath(".ai_scientist", "auth", "session.json")


def auth_file_path() -> Path:
    return _default_auth_file()


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_session() -> Optional[Dict[str, Any]]:
    return _safe_read_json(auth_file_path())


def _default_ttl_hours() -> int:
    raw = str(os.environ.get("AI_SCIENTIST_AUTH_TTL_HOURS") or "").strip()
    if not raw:
        return 72
    try:
        value = int(raw)
    except ValueError:
        return 72
    return max(1, min(24 * 365, value))


def create_session(
    *,
    username: str,
    ttl_hours: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    user = str(username or "").strip()
    if not user:
        raise ValueError("username is required")

    ttl = _default_ttl_hours() if ttl_hours is None else int(ttl_hours)
    ttl = max(1, min(24 * 365, ttl))
    issued_at = _now_utc()
    expires_at = issued_at + timedelta(hours=ttl)
    payload: Dict[str, Any] = {
        "username": user,
        "session_id": secrets.token_hex(16),
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "last_seen_at": issued_at.isoformat(),
    }
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra
    _safe_write_json(auth_file_path(), payload)
    return payload


def clear_session() -> bool:
    path = auth_file_path()
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


def validate_session() -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    session = load_session()
    if session is None:
        return False, "未检测到登录会话", None

    username = str(session.get("username") or "").strip()
    if not username:
        return False, "登录会话缺少用户名", session

    expires_at = _parse_iso_utc(str(session.get("expires_at") or ""))
    if expires_at is None:
        return False, "登录会话缺少过期时间", session

    if _now_utc() >= expires_at:
        return False, "登录会话已过期", session

    return True, "ok", session


def session_user() -> Optional[str]:
    ok, _, payload = validate_session()
    if not ok or not isinstance(payload, dict):
        return None
    user = str(payload.get("username") or "").strip()
    return user or None


def touch_session() -> None:
    path = auth_file_path()
    session = _safe_read_json(path)
    if not isinstance(session, dict):
        return
    session["last_seen_at"] = _now_utc().isoformat()
    _safe_write_json(path, session)


def require_login(operation: str = "当前操作") -> Dict[str, Any]:
    ok, reason, session = validate_session()
    if ok and isinstance(session, dict):
        touch_session()
        return session

    auth_file = auth_file_path()
    print("❌ 操作被拒绝：请先登录")
    print(f"   操作: {operation}")
    print(f"   原因: {reason}")
    print(f"   会话文件: {auth_file}")
    print("   登录命令: python3 auth_cli.py login --user <你的用户名>")
    raise SystemExit(1)

