# Constitution-Bound Self-Evolution

Self-review lessons may propose changes to prompts, tools, routing, search,
resource allocation, failure recovery, or agent scaffolds. They are guidance
only. XScientist does not allow an autonomous process to mutate production
behavior directly.

The upstream organization of lessons into L0/L1/L2 adaptation levels,
fixed-utility epochs, and quality-diverse atomic intents is specified in
[`SELF_EVOLUTION_ARCHITECTURE.md`](SELF_EVOLUTION_ARCHITECTURE.md). This document
covers the downstream admission gate for one candidate.

Every candidate is bound to the project's intact science constitution. The
autonomous mutation whitelist is explicit; the constitution, epistemic
history, raw evidence, sealed benchmarks, evaluation/verification policy,
identity rules, and safety boundaries are a permanent blacklist.

## Promotion sequence

```text
candidate artifact
  -> per-dimension ablation attribution
  -> shadow-only sealed + prospective benchmark
  -> paired regression and confidence checks
  -> bounded real-research canary
  -> content-verified rollback exercise
  -> independent human approval
  -> production promotion
```

The candidate binds its base and proposed artifacts, provenance, scope,
applicability domains, failure-taxonomy references, risk tier, and rollback
reference with SHA-256. Scope must remain under the logical prefix assigned to
the selected mutable component. Every declared change dimension must show a
positive effect when ablated; un-attributed bundles remain shadow-only.

Benchmark tasks compare the baseline and candidate on identical task IDs. Each
task must be opaque, content-addressed, and timestamped as frozen before the
candidate. The suite must contain a resolved prospective task and use at least
two evaluator stacks distinct from the producer stack. Prospective resolution
times are checked against their frozen not-before time. Default metrics cover
objective quality, reproducibility, false-discovery rate, cost, and latency.
Safety, research integrity, and reproducibility are hard gates.

A successful shadow evaluation returns `promote_to_canary`, never direct
production approval. Production requires observations from at least three real
research projects, no error or quality regression beyond policy, no incidents,
long-tail, out-of-distribution, and common-mode failure checks, plus a rollback
receipt proving that the exact baseline artifact was restored. Approvers must
use the `human:` identity namespace and be distinct from the proposer and
canary executor. High-risk changes require at least two independent humans.

Custom policy may only tighten task counts, ablation effects, evaluator
diversity, canary coverage, confidence, regression, and improvement thresholds.
Safety, research-integrity, and reproducibility requirements cannot be
disabled. Gate, canary, rollback, and promotion artifacts are reconstructed
during validation; post-hoc semantic rewrites are rejected even if an outer
hash is recomputed. The executable runtime additionally supports HMAC-SHA256
and optional Ed25519 identity attestations. Actual production deployment
requires signed candidate, independent benchmark, canary, and human approval
evidence. Hardware-backed keys, external timestamps, and physical benchmark
custody remain deployment trust-boundary responsibilities.

## Required evidence records

A raw candidate passed to the CLI includes `change_scope`,
`applicability_domains`, `failure_taxonomy_refs`, `ablation_dimensions`, and
SHA-256 `provenance_hashes` in addition to its version and artifact hashes.
Identity fields use the `agent:`, `service:`, or `human:` namespaces.

Each benchmark row records a task hash, freeze timestamp, evaluation layer,
domain, producer/evaluator stack IDs and hashes, paired metrics, and hard-gate
results. Sealed rows carry a custody attestation; prospective rows also carry
protocol and resolution hashes plus frozen not-before and resolved timestamps.
Ablation rows bind the full and ablated run hashes to their scores.

Use `build_rollback_receipt` and `build_canary_report` to construct canary
evidence. The latter requires a content-addressed run artifact for every real
research project in the canary.

## CLI

Candidate JSON may contain either a complete candidate artifact or the keyword
arguments accepted by `build_evolution_candidate`.

```bash
xscientist evolution-gate \
  --project-root /path/to/research/project \
  --candidate candidate.json \
  --benchmark hidden_benchmark.json \
  --ablation ablation_results.json \
  --policy promotion_policy.json
```

After a canary:

```bash
xscientist evolution-gate \
  --project-root /path/to/research/project \
  --candidate candidate.json \
  --benchmark hidden_benchmark.json \
  --ablation ablation_results.json \
  --canary canary_report.json \
  --approver human:release-controller
```

Repeat `--approver` for a high-risk candidate.

The latest decision is written to `evolution_gate.json`; compact decisions are
appended to `evolution_gate_history.jsonl`. A held or blocked candidate remains
shadow-only.

To build the candidate evidence instead of supplying hand-authored hashes, run
the workflow in [`EVOLUTION_RUNTIME.md`](EVOLUTION_RUNTIME.md).
