# Config Reference

This document describes the two main operator-facing configuration files used by the long-running daemon workflow:

- `daemon_control.json`
- `source_queue.toml` / `source_priority.json`

## `daemon_control.json`

Purpose:
- Change daemon behavior without restarting the process.
- Pause work, force a phase or mode, tune sleep, and issue temporary per-source commands.

Reference files:
- Example: `configs/daemon/daemon_control.example.json`
- Schema: `configs/daemon/daemon_control.schema.json`

### Top-level fields

- `paused`
  - Type: `boolean`
  - Effect: pauses the daemon loop until changed back to `false`.

- `stop_after_cycle`
  - Type: `boolean`
  - Effect: lets the current cycle finish, then stops cleanly.

- `force_phase`
  - Type: `null | cold_start | steady_state | hot_polish`
  - Effect: overrides automatic phase selection.

- `force_mode`
  - Type: `null | balanced | generate_more | focus_rewrite`
  - Effect: overrides automatic mode selection.

- `source_priority_overrides`
  - Type: object mapping source key/name → number
  - Effect: temporarily reprioritizes specific sources.

- `disabled_sources`
  - Type: array of strings
  - Effect: disables matching sources by name or source key.

- `source_commands`
  - Type: object mapping source key/name → object
  - Effect: one-shot source-specific commands that are consumed automatically.

- `sleep_override_minutes`
  - Type: `number | null`
  - Effect: overrides the next sleep duration.

- `dashboard_refresh_seconds`
  - Type: `integer | null`
  - Effect: changes dashboard auto-refresh cadence.

- `expires_after_cycles`
  - Type: `integer | null`
  - Effect: auto-expires temporary global control overrides.

### `source_commands` fields

- `force_next_cycle`
  - Effect: forces a source to the front of the queue for the next cycle.

- `priority_boost_next`
  - Effect: adds a temporary priority boost for the next selection.

- `disable_once`
  - Effect: skips this source one time.

- `cooldown_cycles_once`
  - Effect: applies an additional cooldown after the next selected cycle.

- `expires_after_cycles`
  - Effect: auto-expires the command if it is not consumed quickly.

### Example pattern

Use this when you want to pause briefly, then resume with a stronger focus on one source:

```json
{
  "paused": false,
  "force_mode": "focus_rewrite",
  "source_priority_overrides": {
    "broad_impact_day": 20
  },
  "source_commands": {
    "broad_impact_day": {
      "force_next_cycle": true,
      "priority_boost_next": 8,
      "cooldown_cycles_once": 2,
      "expires_after_cycles": 2
    }
  }
}
```

## `source_queue.toml` / `source_priority.json`

Purpose:
- Define multiple topic or ideas sources for the daemon.
- Attach scheduling hints, venue preferences, quotas, cooldowns, and day/night behavior.

Reference files:
- Examples:
  - `configs/sources/source_queue.example.toml`
  - `configs/sources/source_priority.example.json`
- Schema:
  - `configs/sources/source_queue.schema.json`

### Per-source common fields

- `name`
  - Human-friendly identifier.

- `type`
  - `topic` or `ideas`

- `value`
  - Path to the topic markdown or ideas JSON.

- `priority`
  - Higher values run earlier.

- `target_venue`
  - Optional default venue for this source.

- `paper_types`
  - Optional default paper types.

- `num_ideas`
  - Optional default idea count for this source.

- `submission_mode`
  - If `true`, appends `--submission-mode`.

- `breakthrough_mode`
  - If `true`, appends `--breakthrough-mode`.

- `workflow_mode`
  - Explicit workflow mode for the source.

- `workflow_modes`
  - Compatibility list used by the daemon when it is trying to align a source with the active execution policy.

- `source_archetype`
  - Source persona used for planning. Supported values:
    - `template_first`
    - `frontier_exploration`
    - `program_guarded`
    - `writing_polish`
    - `review_hardening`

- `batch_profile`
  - Batch posture used by the planner. Supported values:
    - `discovery_sprint`
    - `exploration_sprint`
    - `submission_push`
    - `evidence_pack`
    - `review_hardening`

- `alignment_tags`
  - Optional tags that explain why a source exists and what it is good at.

- `planning_notes`
  - Optional operator note that will surface in the runtime and batch planning boards.

- `generator_args`
  - Extra passthrough args for `continuous_paper_generator.py`.

### Scheduling and budget fields

- `cooldown_cycles`
  - Cycles to wait after a successful selection.

- `max_cycles_per_day`
  - Hard daily cap for this source.

- `success_budget`
  - Daily cap on successful runs for this source.

### Day/night fields

- `time_of_day_preference`
  - `any`, `day`, or `night`

- `day_target_venue`, `night_target_venue`
- `day_paper_types`, `night_paper_types`
- `day_num_ideas`, `night_num_ideas`
- `day_submission_mode`, `night_submission_mode`
- `day_breakthrough_mode`, `night_breakthrough_mode`
- `day_workflow_mode`, `night_workflow_mode`
- `day_workflow_modes`, `night_workflow_modes`
- `day_source_archetype`, `night_source_archetype`
- `day_batch_profile`, `night_batch_profile`
- `day_alignment_tags`, `night_alignment_tags`
- `day_generator_args`, `night_generator_args`

These let the same source behave differently depending on local time.

### Workflow-aware source planning

The daemon now treats each source as a planning unit instead of a flat priority row. At runtime it derives:

- `resolved_workflow_mode`
- `source_archetype`
- `batch_profile`
- `batch_goal`
- `recommended_generator_defaults`

These appear in:

- `latest_source_runtime_board.md`
- `latest_source_health_board.md`
- `latest_source_batch_plan.md`
- `latest_source_next_batch.md`
- `progress.json` / `final_report.json` via `source_provenance`
- `research_manager.py source-board`
- `research_manager.py source-mix`
- `research_manager.py source-next-batch`
- `research_manager.py process-board`

This is also where the repository maps the five reference projects into source-level orchestration:

- `template_first` / `classic_pipeline`: [AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
- `frontier_exploration` / `agentic_tree`: [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- `program_guarded` / `program_driven`: [autoresearch](https://github.com/karpathy/autoresearch)
- `writing_polish` / `writing_studio`: [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)
- `review_hardening` / `review_board`: [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2)

In addition to source-level orchestration, the repository now persists `process_alignment.json` for each contracts-driven run. That artifact audits whether the local runtime is actually matching the expected process blueprint across:

- `ideation`
- `program`
- `exploration`
- `experiment`
- `figure`
- `writing`
- `review`
- `evolution`
- `packaging`

Use:

- `python research_manager.py process-board --status blocked`
- `python research_manager.py submission-board --min-process-alignment-score 80`
- `python research_manager.py shortlist --min-process-alignment-score 80`

to make process-level alignment part of everyday operating discipline instead of a README-only claim.

### Example pattern

Use a broad-impact source in the day and a faster exploratory source at night:

```toml
[[sources]]
name = "broad_impact_day"
type = "topic"
value = "examples/example_topic.md"
priority = 10
time_of_day_preference = "day"
source_archetype = "program_guarded"
batch_profile = "submission_push"
workflow_modes = ["program_driven", "review_board"]
day_workflow_mode = "program_driven"
alignment_tags = ["nature-ready", "budgeted", "broad-impact"]
planning_notes = "Use this source when the daemon should converge on submission-grade artifacts with explicit budgets."
day_target_venue = "nature"
day_paper_types = ["journal"]
day_num_ideas = 3
submission_mode = true
cooldown_cycles = 1
max_cycles_per_day = 4
success_budget = 2

[[sources]]
name = "fast_iterate_night"
type = "topic"
value = "examples/example_topic.md"
priority = 7
time_of_day_preference = "night"
source_archetype = "frontier_exploration"
batch_profile = "exploration_sprint"
workflow_modes = ["agentic_tree", "classic_pipeline"]
night_workflow_mode = "agentic_tree"
alignment_tags = ["frontier-search", "high-variance", "night-shift"]
planning_notes = "Use this source for broader agentic exploration and higher-variance idea generation."
night_target_venue = "neurips"
night_paper_types = ["normal"]
night_num_ideas = 6
submission_mode = false
breakthrough_mode = true
cooldown_cycles = 0
max_cycles_per_day = 8
success_budget = 4
```

## Validation strategy

Recommended order before a long run:

1. Validate repo wiring:
   - `python validate_repo.py`
2. Dry-run daemon with your config:
   - `python continuous_research_daemon.py --source-config your_config.json --dry-run --dashboard-port 0`
3. Open the generated `latest_live_dashboard.html` or served URL.
4. Use `bash run_stable_daemon.sh handoff` for a concise shift-handoff view. The handoff report now includes `attention_label`, `recovery_reason`, and a default `recovery_command`.
5. Use `bash run_stable_daemon.sh recover --print-command` to inspect the default recovery action, or `bash run_stable_daemon.sh recover` to execute it.
6. Use `bash run_stable_daemon.sh daily-report` to read the latest daily archive, or `bash run_stable_daemon.sh daily-report --report-date YYYY-MM-DD` to inspect a specific archived day under `reports/daily/`.
7. Use `bash run_stable_daemon.sh list-reports --report-kind all --top 10` to browse recent archived daily and handoff reports.
4. Only then start a long-running session.

## `daemon_profile.json`

Purpose:
- Store a reusable set of daemon arguments for long-running operation.

Reference files:
- Balanced example: `configs/daemon/daemon_profile.example.json`
- Stable example: `configs/daemon/stable_daemon_profile.example.json`
- Stable day example: `configs/daemon/stable_day_daemon_profile.example.json`
- Stable night example: `configs/daemon/stable_night_daemon_profile.example.json`
- Schema: `configs/daemon/daemon_profile.schema.json`
- Launcher: `run_daemon_profile.py`

Typical fields:
- `research_dir`
  - Explicit output-root override for this profile. If omitted by a direct CLI run, `continuous_research_daemon.py` uses the runtime default from `ai_scientist/config/paths.py` (`RESEARCH_OUTPUT_DIR` > `AI_SCIENTIST_OUTPUT_DIR` > sibling `<repo-name>_outputs`).
- `daemon_name`
- `source_config`
- `duration_hours` / `run_forever` / `max_cycles`
- `sleep_minutes` / `failure_backoff_minutes` / `cycle_timeout_minutes`
- `auto_failure_guard` / `auto_failure_guard_threshold` / `auto_failure_guard_cooldown_cycles`
- `auto_source_quality_feedback` / `source_quality_feedback_min_papers` / `source_quality_feedback_max_boost` / `source_quality_feedback_max_penalty`
- `auto_quality_strategy_feedback` / `quality_strategy_submission_priority_threshold` / `quality_strategy_ready_rate_threshold`
- `quality_strategy_exploration_priority_ceiling` / `quality_strategy_gate_pass_floor` / `quality_strategy_max_num_ideas_for_strong_sources` / `quality_strategy_max_num_ideas_for_weak_sources`
- `quality_strategy_dominant_venue_rate_threshold` / `quality_strategy_dominant_paper_type_rate_threshold`
- `auto_evidence_strategy_feedback` / `evidence_strategy_claim_support_floor` / `evidence_strategy_numeric_coverage_floor` / `evidence_strategy_evidence_density_floor`
- `evidence_strategy_claim_alignment_floor` / `evidence_strategy_unsupported_claims_ceiling`
- `evidence_strategy_min_quality_rewrite_rounds` / `evidence_strategy_review_strategy`
- `auto_quality_governor` / `quality_governor_recent_cycles` / `quality_governor_stabilize_health_threshold`
- `quality_governor_exploit_followup_gain` / `quality_governor_max_rewrite_top_k` / `quality_governor_max_dossier_top_k` / `quality_governor_max_source_plan_actions`
- `serve_dashboard` / `dashboard_port` / `dashboard_refresh_seconds`
- `enable_rewrite_followup` / `rewrite_followup_top_k` / `rewrite_followup_max_rounds`
- `adaptive_rewrite_followup` / `rewrite_followup_skip_blocker_threshold` / `rewrite_followup_blocker_reduction_threshold`
- `rewrite_followup_ready_max_rounds` / `rewrite_followup_publishable_priority_threshold` / `rewrite_followup_publishable_gain_threshold`
- `guardrail_submission_target` / `guardrail_min_followup_gain`
- `submission_board_min_priority` / `rewrite_board_min_gain` / `shortlist_min_priority`
- `auto_export_submission_dossier` / `auto_submission_dossier_top_k` / `auto_submission_dossier_require_gate` / `auto_submission_dossier_require_ready`
- `auto_submission_dossier_min_priority` / `auto_submission_dossier_max_blockers` / `auto_submission_dossier_min_rewrite_gain`
- `generator_args`

Quick start:

```bash
python run_daemon_profile.py configs/daemon/daemon_profile.example.json --dry-run
```

Relative path behavior:
- `source_config`, `topic`, `ideas`, `topic_files`, `ideas_files`, and `research_dir` are resolved relative to the profile file first.
- If a relative path is not found there, the launcher falls back to the repository root.
- The checked-in daemon profile examples omit `research_dir` so they use the repository-wide output default. Add `research_dir` only in a local overlay when you want a machine-specific output root.

Local overlay behavior:
- If a sibling local override exists, such as `configs/daemon/stable_daemon_profile.local.json`, it is applied automatically after the base profile.
- You can also pass `--overlay /path/to/override.json` to `run_daemon_profile.py` or `run_stable_daemon.sh`; wrapper ops commands will use it when resolving `--target-mode`.
- Example local override: `configs/daemon/stable_daemon_profile.local.example.json`.

Recommended presets:
- `configs/daemon/daemon_profile.example.json`: balanced example for normal experimentation.
- `configs/daemon/stable_daemon_profile.example.json`: conservative 24-hour profile with slower sleep cadence, smaller rewrite follow-up fan-out, tighter board thresholds, and a dedicated stable source queue.
- `configs/daemon/stable_day_daemon_profile.example.json`: shorter daytime polishing profile with stronger rewrite follow-up and tighter shortlist thresholds.
- `configs/daemon/stable_night_daemon_profile.example.json`: shorter nighttime generation profile with broader idea search and rewrite follow-up disabled by default.

## `source_config` examples

Reference files:
- `configs/sources/source_queue.example.toml`: general source queue example
- `configs/sources/source_priority.example.json`: more aggressive multi-source example
- `configs/sources/stable_source_priority.example.json`: conservative long-run queue with smaller idea counts and daily budgets
- `configs/sources/source_queue.schema.json`: schema for JSON/TOML source queues

## `run_stable_daemon.sh`

Purpose:
- Launch the stable daemon presets without typing long profile commands.

Modes:
- `auto`: choose day or night stable profile from local time
- `balanced`: use the 24-hour stable profile
- `day`: use the daytime polishing profile
- `night`: use the nighttime generation profile
- `rehearsal`: run `run_daemon_rehearsal.py`
- `doctor`: run rehearsal plus `preflight_check.py --strict`, and optionally `validate_repo.py --full-import-smoke` with `--full`
- `program`: print the latest autonomous research program
- `experiment-ledger`: print the latest keep/discard/crash experiment ledger rows
- `status`: inspect the latest daemon run
- `brief`: print the latest operator brief
- `handoff`: print the latest handoff report
- `daily-report`: print the latest archived daily report (or use `--report-date YYYY-MM-DD`)
- `list-reports`: list recent archived daily and/or handoff reports (`--report-kind daily|handoff|all`, `--top N`)
- `recover`: execute the default `recovery_command` from the latest handoff report
- `dashboard`: print the latest dashboard URL or HTML path
- `open-dashboard`: open the latest dashboard in the default browser when possible
- `logs`: print the latest launch log and latest cycle log
- `tail-heartbeat`: print recent heartbeat lines
- `control`: print the current `daemon_control.json`
- `control-history`: print recent `daemon_control_history.jsonl` entries
- `source-summary`: show source runtime state, health, and active control overlays
- `auto_apply_source_plan`: automatically queue the top healthy `source-force-next`/`source-boost-next` action into control after each cycle; it now also consumes source-mix advice for `policy-align` and `mix-rebalance`
- `auto_source_plan_max_actions`: cap how many source-plan actions are auto-queued per cycle
- `auto_source_plan_min_health`: minimum source health score before auto-queueing a source-plan action
- `auto_source_plan_expires_after_cycles`: expire auto-queued source-plan commands after N cycles
- `source-plan`: generate next-step source scheduling recommendations from the latest boards, plus ready-to-run wrapper commands
- `pause` / `resume`: toggle `paused`
- `stop-after-cycle` / `clear-stop-after-cycle`: toggle `stop_after_cycle`
- `set-mode MODE` / `clear-mode`: set or clear `force_mode`
- `set-phase PHASE` / `clear-phase`: set or clear `force_phase`
- `set-sleep MINUTES` / `clear-sleep`: set or clear `sleep_override_minutes`
- `disable-source NAME` / `enable-source NAME`: mutate `disabled_sources`
- `set-source-priority NAME P` / `clear-source-priority NAME`: mutate `source_priority_overrides`
- `source-force-next NAME`: set `source_commands[NAME].force_next_cycle=true`
- `source-boost-next NAME P`: set `source_commands[NAME].priority_boost_next=P`
- `source-disable-once NAME`: set `source_commands[NAME].disable_once=true`
- `source-cooldown-once NAME N`: set `source_commands[NAME].cooldown_cycles_once=N`
- `clear-source-command NAME`: remove `source_commands[NAME]`

Useful flags:
- `--overlay /path/to/override.json`
- `--auto-apply-source-plan`
- `--auto-source-plan-max-actions 1`
- `--auto-source-plan-min-health 88`
- `--auto-source-plan-expires-after-cycles 1`
- `--target-mode balanced|day|night|auto`
- `--daemon-dir /path/to/daemon_runs/<name>`
- `--lines 80`
- `--full` for `doctor`

Examples:

```bash
bash run_stable_daemon.sh auto --dry-run --print-command
bash run_stable_daemon.sh balanced --background
bash run_stable_daemon.sh dashboard --target-mode balanced
bash run_stable_daemon.sh logs --target-mode day --lines 80
bash run_stable_daemon.sh pause --target-mode balanced
bash run_stable_daemon.sh set-mode focus_rewrite --target-mode balanced
bash run_stable_daemon.sh disable-source broad_impact_day --target-mode balanced
bash run_stable_daemon.sh source-force-next conference_night_watch --target-mode balanced
bash run_stable_daemon.sh source-summary --target-mode balanced
bash run_stable_daemon.sh source-plan --target-mode balanced
# source-plan will also print exact follow-up commands such as source-force-next or enable-source.
```
