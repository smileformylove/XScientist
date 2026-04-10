# Operations Checklist

This checklist is meant for running the daemon in a practical, repeatable way.

## Before a long run

- Confirm model/API credentials are present.
- Run `python validate_repo.py`.
- Run a dry-run rehearsal:
  - `python run_daemon_rehearsal.py`
- Prefer the conservative preset for a first real 24-hour run:
  - `python run_daemon_profile.py stable_daemon_profile.example.json --dry-run --print-command`
- If you want a simpler operator path, use the wrapper:
  - `bash run_stable_daemon.sh auto --dry-run --print-command`
- If the machine needs local-only settings, copy `stable_daemon_profile.local.example.json` to `stable_daemon_profile.local.json` and edit that file instead of changing the checked-in preset.
- Open the dashboard or generated HTML.
- Check that `source_queue` and `daemon_control` files are valid and match the intended schedule.

## During a run

- Watch `latest_operator_brief.md` for priorities and actions.
- Watch `latest_live_dashboard.html` for:
  - recent failures
  - rewrite uplift trends
  - source availability
  - control status
- If needed, modify `daemon_control.json` instead of restarting the daemon.

## If things stall

- Check `latest_rewrite_board.md` and `latest_source_health_board.md`.
- If rewrite uplift is weak, bias toward generation.
- If strong drafts are accumulating, bias toward rewrite focus.
- If a source is repeatedly failing, temporarily disable or cool it down.

## If things fail hard

- Check `heartbeat.log`.
- Check the latest cycle log under `cycles/`.
- Check `daemon_control_history.jsonl` for recent control events.
- Reduce concurrency/throughput or raise backoff if API or system stability is poor.

## Daily review

- Read `latest_daily_summary.md`.
- Read `latest_operator_brief.md`.
- Export and inspect:
  - `submission-board`
  - `rewrite-board`
- Decide whether to:
  - add more sources
  - raise source priority
  - lower source priority
  - increase rewrite focus
  - reduce rewrite focus

## Handy stable-daemon ops commands

- Open the latest dashboard:
  - `bash run_stable_daemon.sh open-dashboard`
- Tail heartbeat during a long run:
  - `bash run_stable_daemon.sh tail-heartbeat --lines 40`
- Inspect recent logs:
  - `bash run_stable_daemon.sh logs --lines 80`
- Run a preflight doctor check (rehearsal + strict preflight):
  - `bash run_stable_daemon.sh doctor`

## Control shortcuts

- Pause a daemon cleanly:
  - `bash run_stable_daemon.sh pause --target-mode balanced`
- Resume a daemon:
  - `bash run_stable_daemon.sh resume --target-mode balanced`
- Stop after the current cycle:
  - `bash run_stable_daemon.sh stop-after-cycle --target-mode balanced`
- Force rewrite focus for the next cycles:
  - `bash run_stable_daemon.sh set-mode focus_rewrite --target-mode balanced`
- Inspect current control state:
  - `bash run_stable_daemon.sh control --target-mode balanced`

## Source-level control shortcuts

- Temporarily disable a noisy source:
  - `bash run_stable_daemon.sh disable-source broad_impact_day --target-mode balanced`
- Re-enable it later:
  - `bash run_stable_daemon.sh enable-source broad_impact_day --target-mode balanced`
- Force one source to run next:
  - `bash run_stable_daemon.sh source-force-next conference_night_watch --target-mode balanced`
- Give one source a one-off priority boost:
  - `bash run_stable_daemon.sh source-boost-next conference_night_watch 5 --target-mode balanced`
- Apply one-off cooldown to a source:
  - `bash run_stable_daemon.sh source-cooldown-once broad_impact_day 2 --target-mode balanced`

## Source advisory commands

- Review source-level recommendations:
  - `bash run_stable_daemon.sh source-plan --target-mode balanced`
  - If the output includes a good next step, copy the suggested command directly instead of editing `daemon_control.json` by hand.
- Inspect runtime state and active source overrides:
  - `bash run_stable_daemon.sh source-summary --target-mode balanced --lines 10`

## Daily archive review

- Read the latest archived daily report:
  - `bash run_stable_daemon.sh daily-report`
- Read a specific archived day:
  - `bash run_stable_daemon.sh daily-report --report-date YYYY-MM-DD`

## Report archive browsing

- List recent archived reports:
  - `bash run_stable_daemon.sh list-reports --report-kind all --top 10`
- Only list handoff snapshots:
  - `bash run_stable_daemon.sh list-reports --report-kind handoff --top 10`
