# DeepReviewer-v2 Alignment (Self-Review Upgrade)

This document maps key DeepReviewer-v2 review-loop ideas to concrete implementation in this repository.

## Point-to-Point Mapping

| DeepReviewer-v2 mechanism | Local implementation | File location |
|---|---|---|
| Structured review state (not just free-form text) | Introduced normalized review payload + issue ledger (`issue_id`, `severity`, `category`, `evidence`, `action_hint`) | `ai_scientist/utils/self_review_optimizer.py` |
| Priority-driven issue triage | Severity/category ranking, de-duplication, and top issue set for rewrite | `ai_scientist/utils/self_review_optimizer.py` |
| Value-driven prioritization (impact vs. effort) | Added per-issue `impact_score`, `value_density`, `priority_tier`, and venue-aware weighting (`nature` stricter weighting) | `ai_scientist/utils/self_review_optimizer.py` |
| Cross-round issue tracking | Added issue progress comparator (`resolved/persistent/new`, `unresolved_critical_count`) | `ai_scientist/utils/self_review_optimizer.py` |
| Persistent issue memory | Added `issue_memory` and `persistence_count` so repeated unresolved issues are automatically escalated in later rounds | `ai_scientist/utils/self_review_optimizer.py` |
| Gate-like completion checks | Added rewrite coverage ratio (`covered_issue_ids / targeted_issue_ids`) and compile rollback guard | `ai_scientist/utils/self_review_optimizer.py` |
| Finalization gate packet per round | Added `round_gate` with hard checks (`critical/persistent/coverage/high-value coverage/round budget`) and numeric gate score | `ai_scientist/utils/self_review_optimizer.py`, `run_project.py`, `ai_scientist/perform_auto_improvement.py` |
| Structured final artifacts each round | Persisted `issue_ledger.json`, `review_structured.md`, `issue_progress.json`, `rewrite_result_round_*.json` | `reviews_round_*` outputs |
| Evented iterative loop, not one-shot rewrite | Replaced single prompt rewrite with issue-driven loop in Step 3 and convergence stopping conditions | `run_project.py` |
| High-value rewrite accounting | Rewrite artifacts now include `high_value_coverage_ratio`, `structured_plan_coverage_ratio`, and unknown addressed issue diagnostics | `ai_scientist/utils/self_review_optimizer.py` |
| End-of-loop regression check | Added final-round issue progress evaluation against latest ledger | `run_project.py` |
| Fallback auto-improvement path uses same loop | Upgraded `perform_auto_improvement.improve_paper_with_review` to same issue-ledger loop and artifacts | `ai_scientist/perform_auto_improvement.py` |
| Review-loop signals feed autopilot scheduling | Added run-index extraction of self-review gate metrics and daemon follow-up policy escalation (`evidence_gap_repair`) | `ai_scientist/utils/run_index.py`, `research_manager.py`, `continuous_research_daemon.py` |
| Evidence-gap findings become executable tasks | Added experiment TODO synthesis from round-gate reasons, next-focus hints, revision actions, and evidence weakness signals | `continuous_paper_generator.py` |
| TODO backlog feeds budget strategy | Daemon now uses per-paper/source TODO pressure to bias rewrite policy, rewrite rounds, review depth, and idea exploration width | `continuous_research_daemon.py` |
| TODO closure telemetry across rounds | Added per-round/final TODO closure snapshots and artifacts (`closure_rate`, `p0_closure_rate`, unresolved backlog) for autonomous control loops | `ai_scientist/utils/experiment_todo_progress.py`, `run_project.py`, `ai_scientist/perform_auto_improvement.py`, `ai_scientist/utils/run_index.py` |
| Governor-level closure repair mode | Quality governor now enters `closure_repair` mode when active-source TODO closure/backlog signals indicate unresolved pressure, and suppresses dossier expansion until closure improves | `continuous_research_daemon.py` |
| TODO closure trend observability | Dashboard/report trends now include TODO closure/backlog trend cards and archive metrics for longitudinal monitoring | `continuous_research_daemon.py` |
| Review issues become agentic repair lanes | Added `repair_queue` -> `repair_plan.json` conversion with lane routing, execution steps, success criteria, and verification-ready tasks | `ai_scientist/utils/review_jobs.py`, `ai_scientist/utils/review_repair_planner.py` |
| Review-driven closure enters operator boards | Added `repair-board`, repair-ready coverage, and repair-plan stats to manager boards and run index | `research_manager.py`, `ai_scientist/utils/run_index.py` |
| Reviewer feedback becomes reusable evolution memory | Added `self_evolution.json` plus `knowledge_base/self_evolution_playbook.json` so repair lessons persist across projects and feed future adaptive recommendations | `ai_scientist/utils/self_evolution.py`, `ai_scientist/adaptive_learning_engine.py`, `research_manager.py` |
| Review repair quality directly affects submission gating and orchestration | `submission-board`, `shortlist`, `readiness-benchmark`, `source-mix`, and daemon contract feedback now penalize or block runs with poor `self_evolution` closure | `research_manager.py`, `ai_scientist/utils/readiness_benchmark.py`, `continuous_research_daemon.py` |
| Review closure becomes part of cross-process open-source alignment | Added `process_alignment.json` and `process-board` so review quality is audited alongside ideation/program/exploration/figure/writing against the five reference repos | `ai_scientist/utils/process_alignment.py`, `research_manager.py`, `continuous_research_daemon.py` |

## New Runtime Artifacts

For each round `reviews_round_<k>/`:
- `issue_ledger.json`
- `review_structured.md`
- `issue_progress.json` (from round 2+)
- `round_gate.json`
- `round_gate_report.md`
- `template_before_round_<k>.tex`
- `template_after_round_<k>.tex`
- `rewrite_response_round_<k>.txt`
- `rewrite_result_round_<k>.json`

For each experiment root:
- `self_review_iteration_summary.json`
- `self_review_final_progress.json` (if final review exists and previous issue ledger exists)
- `review_state.json` with `active_issue_records`, `issue_to_claim/figure/section`, `repair_queue`, and `repair_metrics`
- `repair_plan.json` with lane-oriented repair tasks and success criteria
- `self_evolution.json` with self-check status, agentic lessons, and next-cycle defaults
- `process_alignment.json` with process-level status/score/risks/reference mappings, including the `review` and `evolution` processes
- `knowledge_base/self_evolution_playbook.json` with cross-project recurring risks and top agentic defaults
- `readiness_benchmark` / `submission-board` / `shortlist` now surface `self_evolution_status`, `self_evolution_score`, and `required_failure_count`
- `process-board`, `submission-board`, `shortlist`, and `readiness-benchmark` now also surface `process_alignment` status, blocked counts, and process-level risks
- `improvements/round_*/issue_ledger.json` and companion rewrite artifacts when using `perform_auto_improvement.py`
- `self_review_iteration_summary.json` now includes `latest_round_gate` and `round_gate_ready`
- `experiment_todo.json` / `experiment_todo.md` (batch level)
- `<paper_dir>/experiment_todo.json` / `<paper_dir>/experiment_todo.md` (paper level)
- `<paper_dir>/experiment_todo_progress.json` / `<paper_dir>/experiment_todo_progress.md` (round/final closure telemetry)
- `run_index` now tracks `self_review_round_gate_*`, `self_review_high_value_coverage`, unresolved critical counters, `experiment_todo_*`, and TODO closure signals for scheduling

## Scope Notes

- This upgrade focuses on **self-review loop quality** and **traceable iteration**.
- It does **not** copy DeepReviewer’s full tool orchestration stack (PDF MCP tools, external paper-search service gates, full sectioned final markdown writer), which would require a larger runtime architecture change.
