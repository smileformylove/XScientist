#!/usr/bin/env python3
"""Simple login/session CLI for XScientist operations."""

from __future__ import annotations

import argparse
import locale
import sys
from datetime import datetime, timezone

from ai_scientist.utils.auth_session import (
    auth_file_path,
    clear_session,
    create_session,
    session_user,
    validate_session,
)
from ai_scientist.utils.privacy import portable_path


def _auth_location() -> str:
    """Describe the session location without exposing a host-local path."""

    return portable_path(auth_file_path(), base=".")


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


def _language(value: str) -> str:
    if value in {"en", "zh"}:
        return value
    detected = (locale.getlocale()[0] or "en").lower()
    return "zh" if detected.startswith("zh") else "en"


def _reason(value: str, language: str) -> str:
    if language == "zh":
        return value
    return {
        "未检测到登录会话": "no login session was found",
        "登录会话缺少用户名": "the login session has no username",
        "登录会话缺少过期时间": "the login session has no expiration time",
        "登录会话已过期": "the login session has expired",
    }.get(value, value)


def main() -> int:
    parser = argparse.ArgumentParser(description="XScientist login session management")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    login_parser = subparsers.add_parser("login", help="create a local actor session")
    login_parser.add_argument(
        "--user", help="local research actor name (prompted in a terminal)"
    )
    login_parser.add_argument(
        "--ttl-hours",
        type=int,
        default=None,
        help="session lifetime in hours (default: 72)",
    )

    status_parser = subparsers.add_parser(
        "status", help="show the current local actor session"
    )
    logout_parser = subparsers.add_parser(
        "logout", help="delete the current local actor session"
    )
    for command_parser in (login_parser, status_parser, logout_parser):
        command_parser.add_argument(
            "--lang", choices=["auto", "en", "zh"], default="auto"
        )

    args = parser.parse_args()
    language = _language(args.lang)

    if args.cmd == "login":
        username = str(args.user or "").strip()
        if not username and sys.stdin.isatty():
            prompt = (
                "本地科研身份名称："
                if language == "zh"
                else "Local research actor name: "
            )
            username = input(prompt).strip()
        if not username:
            parser.error("--user is required outside an interactive terminal")
        session = create_session(username=username, ttl_hours=args.ttl_hours)
        if language == "zh":
            print("✅ 登录成功")
            print(f"   用户：{session.get('username')}")
            print(f"   过期：{_format_time(str(session.get('expires_at') or ''))}")
            print(f"   会话文件：{_auth_location()}")
        else:
            print("Local research actor session created.")
            print(f"User: {session.get('username')}")
            print(f"Expires: {_format_time(str(session.get('expires_at') or ''))}")
            print(f"Session file: {_auth_location()}")
        return 0

    if args.cmd == "status":
        ok, reason, session = validate_session()
        if not ok:
            if language == "zh":
                print("❌ 未登录")
                print(f"   原因：{reason}")
                print(f"   会话文件：{_auth_location()}")
            else:
                print("No active local research actor session.")
                print(f"Reason: {_reason(reason, language)}")
                print(f"Session file: {_auth_location()}")
            return 1
        if language == "zh":
            print("✅ 已登录")
            print(f"   用户：{session_user()}")
            print(f"   签发：{_format_time(str(session.get('issued_at') or ''))}")
            print(f"   过期：{_format_time(str(session.get('expires_at') or ''))}")
            print(
                f"   最近活动：{_format_time(str(session.get('last_seen_at') or ''))}"
            )
            print(f"   会话文件：{_auth_location()}")
        else:
            print("Local research actor session is active.")
            print(f"User: {session_user()}")
            print(f"Issued: {_format_time(str(session.get('issued_at') or ''))}")
            print(f"Expires: {_format_time(str(session.get('expires_at') or ''))}")
            print(
                f"Last active: {_format_time(str(session.get('last_seen_at') or ''))}"
            )
            print(f"Session file: {_auth_location()}")
        return 0

    if args.cmd == "logout":
        removed = clear_session()
        if language == "zh" and removed:
            print("✅ 已退出登录")
        elif language == "zh":
            print("ℹ️ 当前没有可删除的登录会话")
        elif removed:
            print("Local research actor session deleted.")
        else:
            print("No local research actor session existed.")
        label = "会话文件" if language == "zh" else "Session file"
        print(f"{label}: {_auth_location()}")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
