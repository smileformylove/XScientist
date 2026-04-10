#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_scientist.utils.auth_session import require_login
from run_daemon_profile import _load_profile_with_overlays

REQUIRED_ARTIFACTS = [
    "daemon_status.json",
    "daemon_control.json",
    "latest_cycle_summary.json",
    "latest_daily_summary.json",
    "latest_operator_brief.json",
    "latest_source_runtime_board.json",
    "latest_source_health_board.json",
    "latest_live_dashboard.json",
    "latest_live_dashboard.html",
]


def _print_step(message: str) -> None:
    print(f"[rehearsal] {message}")


def main() -> int:
    require_login("守护进程演练(run_daemon_rehearsal)")

    parser = argparse.ArgumentParser(
        description="Run a short daemon rehearsal and verify key artifacts"
    )
    parser.add_argument(
        "--profile",
        default=str(PROJECT_ROOT / "stable_daemon_profile.example.json"),
    )
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Optional overlay JSON/TOML applied after the base rehearsal profile",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--keep-dir",
        action="store_true",
        help="Keep the temporary rehearsal directory",
    )
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    profile, applied_overlays = _load_profile_with_overlays(
        profile_path,
        explicit_overlays=args.overlay,
    )

    with tempfile.TemporaryDirectory(prefix="ai_scientist_rehearsal_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        daemon_name = "rehearsal_daemon"
        profile["research_dir"] = str(tmpdir_path)
        profile["daemon_name"] = daemon_name

        temp_profile = tmpdir_path / "rehearsal_profile.json"
        temp_profile.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        cmd = [
            args.python,
            str(PROJECT_ROOT / "run_daemon_profile.py"),
            str(temp_profile),
            "--dry-run",
        ]
        _print_step("starting dry-run rehearsal")
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            return completed.returncode

        print(completed.stdout.strip())
        payload = json.loads(completed.stdout)
        daemon_dir = Path(payload["daemon_dir"])
        missing = [
            name for name in REQUIRED_ARTIFACTS if not (daemon_dir / name).exists()
        ]
        if missing:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "missing_artifacts": missing,
                        "daemon_dir": str(daemon_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1

        control_snapshot = json.loads(
            (daemon_dir / "daemon_control.json").read_text(encoding="utf-8")
        )
        summary = {
            "status": "ok",
            "daemon_dir": str(daemon_dir),
            "dashboard_url": payload.get("dashboard_url"),
            "artifacts_checked": REQUIRED_ARTIFACTS,
            "profile_path": str(profile_path),
            "applied_overlays": applied_overlays,
            "control_snapshot": control_snapshot,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[rehearsal-summary]{json.dumps(summary, ensure_ascii=False)}")

        if args.keep_dir:
            target = PROJECT_ROOT / "_outputs" / "rehearsal_last_path.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(daemon_dir) + "\n", encoding="utf-8")
            _print_step(f"recorded daemon dir to {target}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
