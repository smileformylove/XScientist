#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"
resolve_python_bin() {
  local candidates=()
  if [[ -n "${PYTHON:-}" ]]; then
    candidates+=("$PYTHON")
  fi
  candidates+=("python3.11" "python3.10" "python3")

  local candidate
  for candidate in "${candidates[@]}"; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(resolve_python_bin)"; then
  echo "Unable to locate Python >= 3.10 (set PYTHON to override)." >&2
  exit 1
fi

require_login_session() {
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from ai_scientist.utils.auth_session import validate_session
ok, _, _ = validate_session()
raise SystemExit(0 if ok else 1)
PY
  then
    return 0
  fi
  echo "Login required: please run 'python3 auth_cli.py login --user <your_name>' first." >&2
  return 1
}

resolve_research_dir() {
  "$PYTHON_BIN" - <<'PY' "$PROJECT_ROOT"
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project_root))

from ai_scientist.config.paths import resolve_output_path  # noqa: E402

print(resolve_output_path())
PY
}

if ! RESEARCH_DIR="$(resolve_research_dir)"; then
  echo "Unable to resolve the research output directory." >&2
  exit 1
fi
export RESEARCH_OUTPUT_DIR="$RESEARCH_DIR"

if ! require_login_session; then
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./run_stable_daemon.sh [auto|balanced|day|night|rehearsal|status|brief|handoff|daily-report|list-reports|report-trends|next-actions|recover|dashboard|open-dashboard|logs|tail-heartbeat|doctor|program|experiment-ledger|control|control-history|source-summary|source-plan|pause|resume|stop-after-cycle|clear-stop-after-cycle|set-mode|clear-mode|set-phase|clear-phase|set-sleep|clear-sleep|disable-source|enable-source|set-source-priority|clear-source-priority|source-force-next|source-boost-next|source-disable-once|source-cooldown-once|clear-source-command] \
    [--background] [--dry-run] [--print-command] [--overlay PATH] [--daemon-dir PATH] [--target-mode MODE] [--lines N] [--full]

Launch modes:
  auto       Use the day profile from 08:00-19:59 and the night profile otherwise.
  balanced   Use configs/daemon/stable_daemon_profile.example.json.
  day        Use configs/daemon/stable_day_daemon_profile.example.json.
  night      Use configs/daemon/stable_night_daemon_profile.example.json.

Ops commands:
  rehearsal            Run run_daemon_rehearsal.py.
  doctor               Run rehearsal plus strict preflight; with --full also run full import smoke.
  program              Print the latest autonomous research program.
  experiment-ledger    Print recent autonomous experiment ledger rows.
  status               Show the latest daemon status summary and brief excerpt.
  brief                Print the operator brief.
  handoff              Print the latest handoff report.
  daily-report         Print the latest archived daily report.
  list-reports         List recent archived daily and/or handoff reports.
  report-trends        Print the archived report trend summary.
  next-actions         Print the current prioritized action queue.
  recover              Execute the default recovery command from the latest handoff report.
  dashboard            Print the dashboard URL or HTML file path.
  open-dashboard       Open the latest dashboard in the default browser if possible.
  logs                 Show the latest launch log and latest cycle log.
  tail-heartbeat       Show recent heartbeat lines.
  control              Print the current daemon_control.json.
  control-history      Print recent daemon_control_history.jsonl entries.
  source-summary       Print the latest source runtime summary with control overlays.
  source-plan          Print source-level scheduling recommendations for the next cycles.

Global control commands:
  pause                Set paused=true.
  resume               Set paused=false.
  stop-after-cycle     Set stop_after_cycle=true.
  clear-stop-after-cycle  Set stop_after_cycle=false.
  set-mode MODE        Set force_mode to balanced|generate_more|focus_rewrite.
  clear-mode           Clear force_mode.
  set-phase PHASE      Set force_phase to cold_start|steady_state|hot_polish.
  clear-phase          Clear force_phase.
  set-sleep MINUTES    Set sleep_override_minutes.
  clear-sleep          Clear sleep_override_minutes.

Source control commands:
  disable-source NAME          Add source to disabled_sources.
  enable-source NAME           Remove source from disabled_sources.
  set-source-priority NAME P   Set source_priority_overrides[NAME]=P.
  clear-source-priority NAME   Remove source_priority_overrides[NAME].
  source-force-next NAME       Set source_commands[NAME].force_next_cycle=true.
  source-boost-next NAME P     Set source_commands[NAME].priority_boost_next=P.
  source-disable-once NAME     Set source_commands[NAME].disable_once=true.
  source-cooldown-once NAME N  Set source_commands[NAME].cooldown_cycles_once=N.
  clear-source-command NAME    Remove source_commands[NAME].

Options:
  --overlay PATH       Apply an explicit profile overlay; also used when resolving --target-mode daemon dirs.
  --daemon-dir PATH    Inspect or control a specific daemon directory.
  --target-mode MODE   For ops/control commands, target balanced|day|night|auto instead of latest.
  --lines N            Number of lines for brief/log/tail/history output (default: 40).
  --report-date DATE   Select a specific archived daily report date (YYYY-MM-DD).
  --report-kind KIND   Choose `daily`, `handoff`, or `all` for list-reports (default: all).
  --top N              Maximum number of report entries to show (default: 10).
  --full               For doctor, also run validate_repo.py --full-import-smoke.

Auto overlay behavior:
  If a matching `*.local.json` exists next to the chosen profile, it is applied automatically.
EOF
}

pick_profile() {
  local mode="$1"
  case "$mode" in
    balanced) echo "configs/daemon/stable_daemon_profile.example.json" ;;
    day) echo "configs/daemon/stable_day_daemon_profile.example.json" ;;
    night) echo "configs/daemon/stable_night_daemon_profile.example.json" ;;
    auto)
      local hour
      hour="$(date +%H)"
      if ((10#$hour >= 8 && 10#$hour < 20)); then
        echo "configs/daemon/stable_day_daemon_profile.example.json"
      else
        echo "configs/daemon/stable_night_daemon_profile.example.json"
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

profile_for_mode() {
  local mode="$1"
  echo "$PROJECT_ROOT/$(pick_profile "$mode")"
}

profile_daemon_name() {
  local profile_path="$1"
  local overlay_path="${2:-}"
  "$PYTHON_BIN" - <<'PY' "$PROJECT_ROOT" "$profile_path" "$overlay_path"
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
profile_path = Path(sys.argv[2]).resolve()
overlay_path = sys.argv[3].strip()
sys.path.insert(0, str(project_root))
from run_daemon_profile import _load_profile_with_overlays  # noqa: E402

profile, _ = _load_profile_with_overlays(
    profile_path,
    explicit_overlays=[overlay_path] if overlay_path else None,
)
print(profile.get('daemon_name', ''))
PY
}

latest_daemon_dir() {
  local daemon_root="$RESEARCH_DIR/daemon_runs"
  if [[ ! -d "$daemon_root" ]]; then
    return 1
  fi
  ls -td "$daemon_root"/* 2>/dev/null | head -n1 || true
}

latest_launch_log() {
  ls -t "$RESEARCH_DIR"/daemon_launch_*.log 2>/dev/null | head -n1 || true
}

resolve_daemon_dir() {
  local explicit_dir="$1"
  local target_mode="$2"
  local overlay_path="${3:-}"
  if [[ -n "$explicit_dir" ]]; then
    echo "$explicit_dir"
    return 0
  fi
  if [[ -n "$target_mode" ]]; then
    local profile_file
    if ! profile_file="$(pick_profile "$target_mode")"; then
      echo "Unknown target mode: $target_mode" >&2
      return 1
    fi
    local profile_path="$PROJECT_ROOT/$profile_file"
    local daemon_name
    daemon_name="$(profile_daemon_name "$profile_path" "$overlay_path")"
    if [[ -n "$daemon_name" ]]; then
      echo "$RESEARCH_DIR/daemon_runs/$daemon_name"
      return 0
    fi
  fi
  latest_daemon_dir
}

read_status_field() {
  local status_path="$1"
  local key="$2"
  "$PYTHON_BIN" - <<'PY' "$status_path" "$key"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data.get(sys.argv[2], ''))
PY
}

ensure_daemon_dir() {
  local daemon_dir="$1"
  if [[ -z "$daemon_dir" || ! -d "$daemon_dir" ]]; then
    echo "No daemon directory found" >&2
    return 1
  fi
}

show_status() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  echo "Daemon directory: $daemon_dir"
  local status_path="$daemon_dir/daemon_status.json"
  if [[ -f "$status_path" ]]; then
    "$PYTHON_BIN" - <<'PY' "$status_path"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
for key in ["state", "cycle", "guardrail_phase", "guardrail_mode", "next_cycle_at", "dashboard_url"]:
    print(f"{key}: {data.get(key)}")
PY
  fi
  if [[ -f "$daemon_dir/latest_operator_brief.md" ]]; then
    echo
    echo "Operator brief (first ${lines} lines):"
    head -n "$lines" "$daemon_dir/latest_operator_brief.md"
  fi
}

show_brief() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local brief_path="$daemon_dir/latest_operator_brief.md"
  if [[ ! -f "$brief_path" ]]; then
    echo "Operator brief not found: $brief_path"
    return 0
  fi
  head -n "$lines" "$brief_path"
}

show_handoff() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local handoff_path="$daemon_dir/latest_handoff_report.md"
  if [[ -f "$handoff_path" ]]; then
    head -n "$lines" "$handoff_path"
    return 0
  fi
  local brief_path="$daemon_dir/latest_operator_brief.md"
  if [[ -f "$brief_path" ]]; then
    echo "Handoff report not found yet; falling back to operator brief: $brief_path"
    echo
    head -n "$lines" "$brief_path"
    return 0
  fi
  echo "Handoff report not found: $handoff_path"
}

show_program() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local program_path="$daemon_dir/latest_autonomy_program.md"
  if [[ ! -f "$program_path" ]]; then
    echo "Autonomy program not found: $program_path"
    return 0
  fi
  head -n "$lines" "$program_path"
}

show_experiment_ledger() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local ledger_path="$daemon_dir/autonomous_experiment_ledger.tsv"
  if [[ ! -f "$ledger_path" ]]; then
    echo "Experiment ledger not found: $ledger_path"
    return 0
  fi
  tail -n "$lines" "$ledger_path"
}

show_daily_report() {
  local daemon_dir="$1"
  local lines="$2"
  local report_date="$3"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local report_path=""
  if [[ -n "$report_date" ]]; then
    report_path="$daemon_dir/reports/daily/$report_date.md"
  else
    report_path="$daemon_dir/latest_daily_report.md"
  fi
  if [[ -f "$report_path" ]]; then
    head -n "$lines" "$report_path"
    return 0
  fi
  local daily_summary_path="$daemon_dir/latest_daily_summary.md"
  if [[ -z "$report_date" && -f "$daily_summary_path" ]]; then
    echo "Daily report not found yet; falling back to daily summary: $daily_summary_path"
    echo
    head -n "$lines" "$daily_summary_path"
    return 0
  fi
  echo "Daily report not found: $report_path"
}

show_report_index() {
  local daemon_dir="$1"
  local report_kind="$2"
  local top_n="$3"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  "$PYTHON_BIN" - <<'PY' "$daemon_dir" "$report_kind" "$top_n"
from pathlib import Path
import json
import sys

daemon_dir = Path(sys.argv[1])
report_kind = sys.argv[2]
top_n = int(sys.argv[3])
index_path = daemon_dir / 'reports' / 'index.json'
items = []
if index_path.exists():
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    for item in payload.get('entries') or []:
        if report_kind != 'all' and item.get('kind') != report_kind:
            continue
        items.append(item)
else:
    def list_reports(kind: str, report_dir: Path):
        if not report_dir.exists():
            return []
        files = sorted(report_dir.glob('*.md'), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"kind": kind, "path": str(p), "name": p.name, "report_date": None, "health_state": None} for p in files[:top_n]]
    if report_kind in {'all', 'daily'}:
        items.extend(list_reports('daily', daemon_dir / 'reports' / 'daily'))
    if report_kind in {'all', 'handoff'}:
        items.extend(list_reports('handoff', daemon_dir / 'reports' / 'handoff'))
print(f"Daemon directory: {daemon_dir}")
print(f"Report kind: {report_kind}")
print(f"Entries shown: {min(len(items), top_n)}")
for item in items[:top_n]:
    print(f"- [{item.get('kind')}] {item.get('name')} | report_date={item.get('report_date')} | health={item.get('health_state')} :: {item.get('path')}")
if not items:
    print('- No archived reports found.')
PY
}

show_report_trends() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local trends_path="$daemon_dir/reports/trends.md"
  if [[ -f "$trends_path" ]]; then
    head -n "$lines" "$trends_path"
    return 0
  fi
  echo "Report trends not found: $trends_path"
}

show_next_actions() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local queue_path="$daemon_dir/latest_primary_action_queue.md"
  if [[ -f "$queue_path" ]]; then
    head -n "$lines" "$queue_path"
    return 0
  fi
  local handoff_path="$daemon_dir/latest_handoff_report.md"
  if [[ -f "$handoff_path" ]]; then
    echo "Primary action queue not found yet; falling back to handoff report: $handoff_path"
    echo
    head -n "$lines" "$handoff_path"
    return 0
  fi
  echo "Primary action queue not found: $queue_path"
}

run_recover() {
  local daemon_dir="$1"
  local dry_run="$2"
  local print_command="$3"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 1
  fi
  local handoff_json="$daemon_dir/latest_handoff_report.json"
  if [[ ! -f "$handoff_json" ]]; then
    echo "Handoff report not found: $handoff_json" >&2
    echo "bash $PROJECT_ROOT/run_stable_daemon.sh handoff --daemon-dir $daemon_dir" >&2
    return 1
  fi
  local recovery_command
  recovery_command="$($PYTHON_BIN - <<'PY' "$handoff_json"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    payload = json.load(f)
print(payload.get('recovery_command') or '')
PY
)"
  if [[ -z "$recovery_command" ]]; then
    echo "No recovery command available in $handoff_json" >&2
    return 1
  fi
  if [[ "$print_command" == true || "$dry_run" == true ]]; then
    echo "$recovery_command"
    return 0
  fi
  eval "$recovery_command"
}

show_dashboard() {
  local daemon_dir="$1"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local status_path="$daemon_dir/daemon_status.json"
  local dashboard_url=""
  if [[ -f "$status_path" ]]; then
    dashboard_url="$(read_status_field "$status_path" dashboard_url)"
  fi
  if [[ -n "$dashboard_url" ]]; then
    echo "$dashboard_url"
    return 0
  fi
  local html_path="$daemon_dir/latest_live_dashboard.html"
  if [[ -f "$html_path" ]]; then
    echo "$html_path"
    return 0
  fi
  echo "Dashboard not found for daemon directory: $daemon_dir"
}

open_dashboard() {
  local daemon_dir="$1"
  local target
  target="$(show_dashboard "$daemon_dir")"
  echo "$target"
  if [[ "$target" == http://* || "$target" == https://* || -f "$target" ]]; then
    if command -v open >/dev/null 2>&1; then
      open "$target" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$target" >/dev/null 2>&1 || true
    fi
  fi
}

show_logs() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local launch_log
  launch_log="$(latest_launch_log)"
  if [[ -n "$launch_log" && -f "$launch_log" ]]; then
    echo "Launch log: $launch_log"
    tail -n "$lines" "$launch_log"
  else
    echo "No launch logs found under $RESEARCH_DIR"
  fi
  local cycle_log
  cycle_log="$(ls -t "$daemon_dir"/cycles/*.log 2>/dev/null | head -n1 || true)"
  if [[ -n "$cycle_log" && -f "$cycle_log" ]]; then
    echo
    echo "Latest cycle log: $cycle_log"
    tail -n "$lines" "$cycle_log"
  else
    echo
    echo "No cycle logs found under $daemon_dir/cycles"
  fi
}

show_heartbeat() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local heartbeat_path="$daemon_dir/heartbeat.log"
  if [[ ! -f "$heartbeat_path" ]]; then
    echo "Heartbeat log not found: $heartbeat_path"
    return 0
  fi
  tail -n "$lines" "$heartbeat_path"
}

control_path_for() {
  local daemon_dir="$1"
  echo "$daemon_dir/daemon_control.json"
}

history_path_for() {
  local daemon_dir="$1"
  echo "$daemon_dir/daemon_control_history.jsonl"
}

ensure_control_file() {
  local daemon_dir="$1"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 1
  fi
  local control_path
  control_path="$(control_path_for "$daemon_dir")"
  if [[ ! -f "$control_path" ]]; then
    "$PYTHON_BIN" - <<'PY' "$control_path"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
default_payload = {
    "paused": False,
    "stop_after_cycle": False,
    "force_phase": None,
    "force_mode": None,
    "source_priority_overrides": {},
    "disabled_sources": [],
    "source_commands": {},
    "sleep_override_minutes": None,
    "dashboard_refresh_seconds": None,
    "expires_after_cycles": None,
}
path.write_text(json.dumps(default_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
  fi
}

show_control() {
  local daemon_dir="$1"
  if ! ensure_control_file "$daemon_dir"; then
    return 0
  fi
  local control_path
  control_path="$(control_path_for "$daemon_dir")"
  "$PYTHON_BIN" -m json.tool "$control_path"
}

show_control_history() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local history_path
  history_path="$(history_path_for "$daemon_dir")"
  if [[ ! -f "$history_path" ]]; then
    echo "Control history not found: $history_path"
    return 0
  fi
  tail -n "$lines" "$history_path"
}

show_source_summary() {
  local daemon_dir="$1"
  local lines="$2"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local runtime_path="$daemon_dir/latest_source_runtime_board.json"
  local control_path="$daemon_dir/daemon_control.json"
  if [[ ! -f "$runtime_path" ]]; then
    echo "Source runtime board not found: $runtime_path"
    return 0
  fi
  [[ -f "$control_path" ]] || ensure_control_file "$daemon_dir" >/dev/null 2>&1 || true
  "$PYTHON_BIN" - <<'PY' "$runtime_path" "$control_path" "$lines"
import json, sys
from pathlib import Path
runtime_path = Path(sys.argv[1])
control_path = Path(sys.argv[2])
limit = int(sys.argv[3])
runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
control = json.loads(control_path.read_text(encoding='utf-8')) if control_path.exists() else {}
rows = (runtime.get('rows') or [])[:limit]
disabled = set(control.get('disabled_sources') or [])
priority_overrides = control.get('source_priority_overrides') or {}
source_commands = control.get('source_commands') or {}
print(f"Generated: {runtime.get('generated_at')}")
print(f"Sources shown: {len(rows)}")
for row in rows:
    name = row.get('name') or row.get('key')
    extras = []
    if name in disabled:
        extras.append('disabled')
    if name in priority_overrides:
        extras.append(f"priority_override={priority_overrides[name]}")
    if name in source_commands:
        extras.append(f"source_command={source_commands[name]}")
    extra_text = f" | {'; '.join(extras)}" if extras else ''
    print(f"- {name} | health={row.get('health_score')} | state={row.get('availability_state')} | venue={row.get('target_venue')} | ideas={row.get('num_ideas')} | action={row.get('suggested_action')}{extra_text}")
PY
}

show_source_plan() {
  local daemon_dir="$1"
  if ! ensure_daemon_dir "$daemon_dir"; then
    return 0
  fi
  local runtime_path="$daemon_dir/latest_source_runtime_board.json"
  local control_path="$daemon_dir/daemon_control.json"
  if [[ ! -f "$runtime_path" ]]; then
    echo "Source runtime board not found: $runtime_path"
    return 0
  fi
  [[ -f "$control_path" ]] || ensure_control_file "$daemon_dir" >/dev/null 2>&1 || true
  "$PYTHON_BIN" - <<'PY' "$runtime_path" "$control_path" "$daemon_dir" "$PROJECT_ROOT/run_stable_daemon.sh"
import json, shlex, sys
from pathlib import Path
runtime_path = Path(sys.argv[1])
control_path = Path(sys.argv[2])
daemon_dir = Path(sys.argv[3])
wrapper_path = Path(sys.argv[4])
runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
control = json.loads(control_path.read_text(encoding='utf-8')) if control_path.exists() else {}
rows = runtime.get('rows') or []
disabled = set(control.get('disabled_sources') or [])
priority_overrides = control.get('source_priority_overrides') or {}
source_commands = control.get('source_commands') or {}
ready = [row for row in rows if row.get('availability_state') == 'ready']
daypart = [row for row in rows if row.get('availability_state') == 'daypart_mismatch']
blocked = [row for row in rows if row.get('availability_state') in {'blocked', 'quota_exhausted', 'success_budget_reached', 'cooldown'}]
ready.sort(key=lambda row: (-(row.get('health_score') or 0), -(row.get('priority') or 0), row.get('name') or ''))
daypart.sort(key=lambda row: (-(row.get('priority') or 0), -(row.get('health_score') or 0), row.get('name') or ''))
commands = []

def add_command(label: str, *parts: str) -> None:
    command = ' '.join(shlex.quote(str(part)) for part in parts)
    commands.append((label, command))

wrapper_base = ['bash', str(wrapper_path)]
print(f"Generated: {runtime.get('generated_at')}")
print('Recommendations:')
if ready:
    top = ready[0]
    top_name = top.get('name')
    print(f"- Prioritize {top_name} next; it is ready with health={top.get('health_score')} and action='{top.get('suggested_action')}'.")
    if top_name not in disabled and top_name not in source_commands:
        add_command(f"Queue {top_name} for the next cycle", *wrapper_base, 'source-force-next', top_name, '--daemon-dir', str(daemon_dir))
    for row in ready[1:3]:
        name = row.get('name')
        if name in disabled:
            print(f"- Re-enable {name}; it is currently disabled but otherwise ready (health={row.get('health_score')}).")
            add_command(f"Re-enable {name}", *wrapper_base, 'enable-source', name, '--daemon-dir', str(daemon_dir))
        elif name not in priority_overrides and (row.get('health_score') or 0) >= 90:
            print(f"- Consider a temporary boost for {name}; it is ready and strong but has no explicit priority override.")
            add_command(f"Give {name} a one-off priority boost", *wrapper_base, 'source-boost-next', name, '3', '--daemon-dir', str(daemon_dir))
else:
    print('- No sources are currently ready. Focus on time-of-day mismatches, cooldowns, or quota resets.')
if daypart:
    row = daypart[0]
    print(f"- Preserve {row.get('name')} for its preferred {row.get('time_of_day_preference')} window; current mismatch reason: {row.get('availability_reason')}.")
if blocked:
    hotspot = blocked[0]
    print(f"- Watch {hotspot.get('name')}: state={hotspot.get('availability_state')} reason={hotspot.get('availability_reason') or 'n/a'}.")
if source_commands:
    for name, command in list(source_commands.items())[:3]:
        print(f"- Pending source command for {name}: {command}")
else:
    print('- No pending one-shot source commands are queued.')
if disabled:
    print(f"- Disabled sources: {', '.join(sorted(disabled))}")
    for name in sorted(disabled)[:2]:
        add_command(f"Review disabled source {name}", *wrapper_base, 'enable-source', name, '--daemon-dir', str(daemon_dir))
else:
    print('- No sources are globally disabled.')
if not commands:
    add_command('Inspect current source state', *wrapper_base, 'source-summary', '--lines', '10', '--daemon-dir', str(daemon_dir))
print('Suggested Commands:')
for label, command in commands[:5]:
    print(f"- {label}: {command}")
PY
}

update_control() {
  local daemon_dir="$1"
  local operation="$2"
  local value1="${3:-}"
  local value2="${4:-}"
  if ! ensure_control_file "$daemon_dir"; then
    return 1
  fi
  local control_path
  control_path="$(control_path_for "$daemon_dir")"
  "$PYTHON_BIN" - <<'PY' "$control_path" "$operation" "$value1" "$value2"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
operation = sys.argv[2]
value1 = sys.argv[3]
value2 = sys.argv[4]
payload = json.loads(path.read_text(encoding='utf-8'))
payload.setdefault('source_priority_overrides', {})
payload.setdefault('disabled_sources', [])
payload.setdefault('source_commands', {})

if operation == 'pause':
    payload['paused'] = True
elif operation == 'resume':
    payload['paused'] = False
elif operation == 'stop-after-cycle':
    payload['stop_after_cycle'] = True
elif operation == 'clear-stop-after-cycle':
    payload['stop_after_cycle'] = False
elif operation == 'set-mode':
    if value1 not in {'balanced', 'generate_more', 'focus_rewrite'}:
        raise SystemExit('invalid mode: ' + value1)
    payload['force_mode'] = value1
elif operation == 'clear-mode':
    payload['force_mode'] = None
elif operation == 'set-phase':
    if value1 not in {'cold_start', 'steady_state', 'hot_polish'}:
        raise SystemExit('invalid phase: ' + value1)
    payload['force_phase'] = value1
elif operation == 'clear-phase':
    payload['force_phase'] = None
elif operation == 'set-sleep':
    try:
        payload['sleep_override_minutes'] = float(value1)
    except ValueError as exc:
        raise SystemExit('invalid sleep minutes: ' + value1) from exc
elif operation == 'clear-sleep':
    payload['sleep_override_minutes'] = None
elif operation == 'disable-source':
    if value1 and value1 not in payload['disabled_sources']:
        payload['disabled_sources'].append(value1)
elif operation == 'enable-source':
    payload['disabled_sources'] = [item for item in payload['disabled_sources'] if item != value1]
elif operation == 'set-source-priority':
    try:
        payload['source_priority_overrides'][value1] = float(value2)
    except ValueError as exc:
        raise SystemExit('invalid source priority: ' + value2) from exc
elif operation == 'clear-source-priority':
    payload['source_priority_overrides'].pop(value1, None)
elif operation == 'source-force-next':
    command = dict(payload['source_commands'].get(value1) or {})
    command['force_next_cycle'] = True
    payload['source_commands'][value1] = command
elif operation == 'source-boost-next':
    command = dict(payload['source_commands'].get(value1) or {})
    try:
        command['priority_boost_next'] = float(value2)
    except ValueError as exc:
        raise SystemExit('invalid priority boost: ' + value2) from exc
    payload['source_commands'][value1] = command
elif operation == 'source-disable-once':
    command = dict(payload['source_commands'].get(value1) or {})
    command['disable_once'] = True
    payload['source_commands'][value1] = command
elif operation == 'source-cooldown-once':
    command = dict(payload['source_commands'].get(value1) or {})
    try:
        command['cooldown_cycles_once'] = int(value2)
    except ValueError as exc:
        raise SystemExit('invalid cooldown cycles: ' + value2) from exc
    payload['source_commands'][value1] = command
elif operation == 'clear-source-command':
    payload['source_commands'].pop(value1, None)
else:
    raise SystemExit('unknown control operation: ' + operation)

path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({"control_path": str(path), "operation": operation, "value": payload}, ensure_ascii=False, indent=2))
PY
}

run_doctor() {
  local full="$1"
  "$PYTHON_BIN" "$PROJECT_ROOT/run_daemon_rehearsal.py"
  "$PYTHON_BIN" "$PROJECT_ROOT/preflight_check.py" --strict
  if [[ "$full" == true ]]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/validate_repo.py" --full-import-smoke
  fi
}

main() {
  local command="${1:-auto}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
  local background=false
  local dry_run=false
  local print_command=false
  local overlay_path=""
  local daemon_dir_override=""
  local target_mode=""
  local lines=40
  local full=false
  local report_date=""
  local report_kind="all"
  local top_n=10
  local value_one=""
  local value_two=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --background) background=true ;;
      --foreground) background=false ;;
      --dry-run) dry_run=true ;;
      --print-command) print_command=true ;;
      --overlay)
        shift
        [[ $# -gt 0 ]] || { echo "--overlay requires a path" >&2; return 1; }
        overlay_path="$1"
        ;;
      --daemon-dir)
        shift
        [[ $# -gt 0 ]] || { echo "--daemon-dir requires a path" >&2; return 1; }
        daemon_dir_override="$1"
        ;;
      --target-mode)
        shift
        [[ $# -gt 0 ]] || { echo "--target-mode requires a mode" >&2; return 1; }
        target_mode="$1"
        ;;
      --lines)
        shift
        [[ $# -gt 0 ]] || { echo "--lines requires a number" >&2; return 1; }
        lines="$1"
        ;;
      --report-date)
        shift
        [[ $# -gt 0 ]] || { echo "--report-date requires a value" >&2; return 1; }
        report_date="$1"
        ;;
      --report-kind)
        shift
        [[ $# -gt 0 ]] || { echo "--report-kind requires a value" >&2; return 1; }
        report_kind="$1"
        ;;
      --top)
        shift
        [[ $# -gt 0 ]] || { echo "--top requires a value" >&2; return 1; }
        top_n="$1"
        ;;
      --full) full=true ;;
      -h|--help) usage; return 0 ;;
      *)
        case "$command" in
          set-mode|set-phase|set-sleep|disable-source|enable-source|clear-source-priority|source-force-next|source-disable-once|clear-source-command)
            if [[ -z "$value_one" ]]; then
              value_one="$1"
            else
              echo "Unexpected extra value: $1" >&2
              return 1
            fi
            ;;
          set-source-priority|source-boost-next|source-cooldown-once)
            if [[ -z "$value_one" ]]; then
              value_one="$1"
            elif [[ -z "$value_two" ]]; then
              value_two="$1"
            else
              echo "Unexpected extra value: $1" >&2
              return 1
            fi
            ;;
          *)
            echo "Unknown option: $1" >&2
            usage >&2
            return 1
            ;;
        esac
        ;;
    esac
    shift
  done

  case "$command" in
    rehearsal)
      exec "$PYTHON_BIN" "$PROJECT_ROOT/run_daemon_rehearsal.py"
      ;;
    doctor)
      run_doctor "$full"
      return 0
      ;;
    recover)
      local daemon_dir
      daemon_dir="$(resolve_daemon_dir "$daemon_dir_override" "$target_mode" "$overlay_path")"
      run_recover "$daemon_dir" "$dry_run" "$print_command"
      return $?
      ;;
    status|brief|handoff|daily-report|list-reports|report-trends|next-actions|dashboard|open-dashboard|logs|tail-heartbeat|doctor|program|experiment-ledger|control|control-history|source-summary|source-plan)
      local daemon_dir
      daemon_dir="$(resolve_daemon_dir "$daemon_dir_override" "$target_mode" "$overlay_path")"
      case "$command" in
        status) show_status "$daemon_dir" "$lines" ;;
        brief) show_brief "$daemon_dir" "$lines" ;;
        handoff) show_handoff "$daemon_dir" "$lines" ;;
        program) show_program "$daemon_dir" "$lines" ;;
        experiment-ledger) show_experiment_ledger "$daemon_dir" "$lines" ;;
        daily-report) show_daily_report "$daemon_dir" "$lines" "$report_date" ;;
        list-reports) show_report_index "$daemon_dir" "$report_kind" "$top_n" ;;
        report-trends) show_report_trends "$daemon_dir" "$lines" ;;
        next-actions) show_next_actions "$daemon_dir" "$lines" ;;
        dashboard) show_dashboard "$daemon_dir" ;;
        open-dashboard) open_dashboard "$daemon_dir" ;;
        logs) show_logs "$daemon_dir" "$lines" ;;
        tail-heartbeat) show_heartbeat "$daemon_dir" "$lines" ;;
        control) show_control "$daemon_dir" ;;
        control-history) show_control_history "$daemon_dir" "$lines" ;;
        source-summary) show_source_summary "$daemon_dir" "$lines" ;;
        source-plan) show_source_plan "$daemon_dir" ;;
      esac
      return 0
      ;;
    pause|resume|stop-after-cycle|clear-stop-after-cycle|clear-mode|clear-phase|clear-sleep)
      local daemon_dir
      daemon_dir="$(resolve_daemon_dir "$daemon_dir_override" "$target_mode" "$overlay_path")"
      update_control "$daemon_dir" "$command"
      return 0
      ;;
    set-mode|set-phase|set-sleep|disable-source|enable-source|clear-source-priority|source-force-next|source-disable-once|clear-source-command)
      if [[ -z "$value_one" ]]; then
        echo "$command requires a value" >&2
        return 1
      fi
      local daemon_dir
      daemon_dir="$(resolve_daemon_dir "$daemon_dir_override" "$target_mode" "$overlay_path")"
      update_control "$daemon_dir" "$command" "$value_one"
      return 0
      ;;
    set-source-priority|source-boost-next|source-cooldown-once)
      if [[ -z "$value_one" || -z "$value_two" ]]; then
        echo "$command requires two values" >&2
        return 1
      fi
      local daemon_dir
      daemon_dir="$(resolve_daemon_dir "$daemon_dir_override" "$target_mode" "$overlay_path")"
      update_control "$daemon_dir" "$command" "$value_one" "$value_two"
      return 0
      ;;
  esac

  local profile_file
  if ! profile_file="$(pick_profile "$command")"; then
    echo "Unknown command or launch mode: $command" >&2
    usage >&2
    return 1
  fi
  local profile_path="$PROJECT_ROOT/$profile_file"
  if [[ ! -f "$profile_path" ]]; then
    echo "Profile not found: $profile_path" >&2
    return 1
  fi

  local -a cmd=("$PYTHON_BIN" "$PROJECT_ROOT/run_daemon_profile.py" "$profile_path")
  [[ -n "$overlay_path" ]] && cmd+=(--overlay "$overlay_path")
  [[ "$dry_run" == true ]] && cmd+=(--dry-run)
  [[ "$print_command" == true ]] && cmd+=(--print-command)

  if [[ "$background" == true && "$dry_run" == false && "$print_command" == false ]]; then
    mkdir -p "$RESEARCH_DIR"
    local launch_log="$RESEARCH_DIR/daemon_launch_$(date +%Y%m%d_%H%M%S).log"
    nohup "${cmd[@]}" > "$launch_log" 2>&1 &
    local daemon_pid=$!
    echo "Launched stable daemon in background"
    echo "  mode: $command"
    echo "  profile: $profile_file"
    echo "  pid: $daemon_pid"
    echo "  log: $launch_log"
    local overlay_hint=""
    if [[ -n "$overlay_path" ]]; then
      overlay_hint=" --overlay $overlay_path"
    fi
    echo "  dashboard: bash run_stable_daemon.sh dashboard --target-mode $command$overlay_hint"
    echo "  heartbeat: bash run_stable_daemon.sh tail-heartbeat --target-mode $command$overlay_hint"
    echo "  pause: bash run_stable_daemon.sh pause --target-mode $command$overlay_hint"
    return 0
  fi

  exec "${cmd[@]}"
}

main "$@"
