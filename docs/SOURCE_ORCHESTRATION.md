# Source Orchestration

This repository now treats `source_queue` entries as workflow-aware research units rather than flat priority rows.

## Why this exists

The five reference projects emphasize different strengths:

- [AI-Scientist](https://github.com/SakanaAI/AI-Scientist): stable template-first throughput
- [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2): broader agentic exploration
- [autoresearch](https://github.com/karpathy/autoresearch): explicit research programs and budgets
- [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing): strong evidence-to-writing polish
- [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2): reviewer-facing hardening

Instead of forcing the whole daemon into one fixed pattern, source planning lets each source declare how it should be used.

## Core planning fields

- `workflow_mode`
  - Explicit runtime mode for the source.
- `workflow_modes`
  - Compatibility list when the daemon is matching the active execution policy.
- `source_archetype`
  - Research persona for the source.
- `batch_profile`
  - Planned batch posture for the next cycle.
- `alignment_tags`
  - Human-readable tags that explain fit.
- `planning_notes`
  - Operator-facing rationale.

## Archetype mapping

| Source archetype | Workflow bias | Open-source inspiration | Typical use |
|---|---|---|---|
| `template_first` | `classic_pipeline` | AI-Scientist | Stable throughput, higher success rate |
| `frontier_exploration` | `agentic_tree` | AI-Scientist-v2 | Wider search, higher-variance discovery |
| `program_guarded` | `program_driven` | autoresearch | Budgeted submission push |
| `writing_polish` | `writing_studio` | awesome-ai-research-writing | Evidence packaging and writing polish |
| `review_hardening` | `review_board` | DeepReviewer-v2 | Reviewer-facing repair and hardening |

## Batch profile mapping

| Batch profile | Main goal | Default workflow |
|---|---|---|
| `discovery_sprint` | Quick idea throughput | `classic_pipeline` |
| `exploration_sprint` | Wider experimental branching | `agentic_tree` |
| `submission_push` | Submission-grade convergence | `program_driven` |
| `evidence_pack` | Stronger figures, captions, and writing support | `writing_studio` |
| `review_hardening` | Multi-role review and repair | `review_board` |

## Runtime outputs

Each daemon cycle now exports:

- `latest_source_runtime_board.md`
- `latest_source_health_board.md`
- `latest_source_batch_plan.md`
- `latest_source_next_batch.md`

Each generated batch/run now also persists source lineage into:

- `progress.json`
- `final_report.json`
- `source_provenance.json`
- `run_index`

That makes the same metadata visible from:

- `python research_manager.py batch-summary <batch_name>`
- `python research_manager.py source-board`
- `python research_manager.py source-mix --desired-policy <workflow>`
- `python research_manager.py source-next-batch --desired-policy <workflow>`

The batch plan is the important new artifact. It explains:

- which source should run now
- which sources should queue next
- which workflow mode each source resolves to
- which defaults the daemon will inject
- which reference-style operating posture the source follows

## Mix advisory

`source-board` tells you how each source has performed.

`source-mix` goes one step further and asks whether the current source portfolio is healthy:

- Is one archetype dominating too much?
- Is the desired workflow policy missing from the active source pool?
- Which source should be promoted?
- Which source has consumed cycles without producing ready or gate-passed outcomes?

This turns source planning into a real operating loop instead of a static config file.

When `--auto-apply-source-plan` is enabled, the daemon now consumes this mix advisory too:

- `policy-align`
  - If the current desired execution policy is underrepresented, the daemon can auto-promote the healthiest aligned source with `source-force-next` or `source-boost-next`.
- `mix-rebalance`
  - If one archetype is dominating the portfolio, the daemon can auto-boost a healthy alternative source to widen the next batch mix.

That means the daemon is no longer only replaying `source-plan` queue suggestions. It can also use the portfolio-level signal from `source-mix` to steer the next cycle toward a healthier research mix.

## Next-batch recipe

`source-next-batch` compresses historical source performance and `source-mix` signals into a concrete multi-lane recipe for the next batch:

- `primary_lane`
  - The strongest source to drive the next batch.
- `diversification_lane`
  - A complementary source that keeps the research mix from collapsing into one archetype.
- `hardening_lane`
  - A source reserved for evidence packaging, review hardening, or submission convergence.

The daemon persists the same recommendation to `latest_source_next_batch.md`, so the operator brief, dashboards, and archived cycle summaries all share the same next-batch cadence and lane assignment.
