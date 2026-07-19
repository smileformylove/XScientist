#!/usr/bin/env python3
"""Simple login/session CLI for XScientist operations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ai_scientist.utils.auth_session import (
    auth_file_path,
    clear_session,
    create_session,
    session_user,
    validate_session,
)


def _format_time(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "-"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    parser = argparse.ArgumentParser(description="XScientist login session management")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    login_parser = subparsers.add_parser("login", help="登录并创建会话")
    login_parser.add_argument("--user", required=True, help="用户名")
    login_parser.add_argument(
        "--ttl-hours",
        type=int,
        default=None,
        help="会话有效期（小时），默认读取 AI_SCIENTIST_AUTH_TTL_HOURS 或 72",
    )

    subparsers.add_parser("status", help="查看当前登录状态")
    subparsers.add_parser("logout", help="退出登录并删除会话")

    args = parser.parse_args()

    if args.cmd == "login":
        session = create_session(username=args.user, ttl_hours=args.ttl_hours)
        print("✅ 登录成功")
        print(f"   用户: {session.get('username')}")
        print(f"   过期: {_format_time(str(session.get('expires_at') or ''))}")
        print(f"   会话文件: {auth_file_path()}")
        return 0

    if args.cmd == "status":
        ok, reason, session = validate_session()
        if not ok:
            print("❌ 未登录")
            print(f"   原因: {reason}")
            print(f"   会话文件: {auth_file_path()}")
            return 1
        print("✅ 已登录")
        print(f"   用户: {session_user()}")
        print(f"   签发: {_format_time(str(session.get('issued_at') or ''))}")
        print(f"   过期: {_format_time(str(session.get('expires_at') or ''))}")
        print(f"   最近活动: {_format_time(str(session.get('last_seen_at') or ''))}")
        print(f"   会话文件: {auth_file_path()}")
        return 0

    if args.cmd == "logout":
        removed = clear_session()
        if removed:
            print("✅ 已退出登录")
        else:
            print("ℹ️ 当前没有可删除的登录会话")
        print(f"   会话文件: {auth_file_path()}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
