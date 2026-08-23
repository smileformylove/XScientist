# Research-policy rollouts

Faraday's useful architectural lesson is not a parameter-count comparison. It
is that a research policy should choose experiments and delegate implementation
to a stronger coding tool, while a task-specific evaluator checks the result.
XScientist now records that contract as a metadata-only `research_rollout`
object.

## What is recorded

- a hash-bound task and split (`train`, `test`, `holdout`, or `external`);
- a five-dimension rubric: result fidelity, claim support, implementation
  fidelity, resource efficiency, and scientific integrity;
- tool calls with provider/model fingerprints, input/output hashes, budgets,
  decision type, and outcome — never prompts, stdout, credentials, or raw
  responses;
- turn metadata and observational post-hoc positive reward-delta credit;
- zero or more evaluator samples with per-dimension scores, mean, and
  disagreement;
- explicit `quality_claim_allowed=false` and `causal_claim_allowed=false`.

The evaluator summary is a measurement record, not a ground-truth claim. A
separate independent gate is still required before promoting a scientific
claim. Missing reward traces remain missing; the builder never imputes a
credit assignment.

## CLI

Create a JSON file such as `episode.json` containing `task_id`, a full
`task_hash`, `time_budget_seconds`, and optional `tool_delegations`, `turns`,
and `evaluations`, then record it with:

```bash
xscientist research rollout episode.json --repo ./first-study --json
```

The command is offline and idempotent. It stores only a redacted, content-
addressed Research VCS object and creates the usual experiment checkpoint.

## Tool swaps and comparison boundaries

`assess_tool_swap_compatibility(reference, candidate)` checks task hash, rubric
hash, split, and time budget. An eligible boundary is only a prerequisite for
a controlled comparison; it does not say that one model or tool is better.
The current implementation does not reproduce Faraday's RL training,
three-judge protocol, coding-agent provider, benchmark task set, or reported
scores. The design is an XScientist audit/training-data contract inspired by
the [Faraday paper](https://arxiv.org/abs/2608.13331), not a local replication.
