# Benchmark-Gated Self-Evolution

Self-review lessons may propose changes to prompts, tools, routing, search, or
agent scaffolds. They are guidance only. XScientist does not allow an
autonomous process to mutate production behavior directly.

## Promotion sequence

```text
candidate artifact
  -> shadow-only hidden benchmark
  -> paired regression and confidence checks
  -> bounded canary
  -> rollback exercise
  -> independent approval
  -> production promotion
```

The candidate binds its base and proposed artifacts with SHA-256 hashes and
must include a rollback reference. Hidden benchmark tasks compare the baseline
and candidate on identical task IDs. Default metrics cover objective quality,
reproducibility, false-discovery rate, cost, and latency. Safety, research
integrity, and reproducibility checks are hard gates.

A successful shadow evaluation returns `promote_to_canary`, never direct
production approval. Production requires enough canary observations, no error
or quality regression beyond policy, no incidents, a tested rollback path,
and an approver different from the proposing agent.

Custom policy may only tighten sample counts, confidence, regression, and
improvement thresholds. Safety, research-integrity, and reproducibility
requirements cannot be disabled. The production step revalidates the candidate and the hash
of every decision-bearing shadow report field, and rejects non-finite canary
measurements. Approver authentication and benchmark secrecy remain deployment
trust-boundary responsibilities; the gate records their identifiers and hashes
for an external identity and artifact service to verify.

## CLI

Candidate JSON may contain either a complete candidate artifact or the keyword
arguments accepted by `build_evolution_candidate`.

```bash
xscientist evolution-gate \
  --project-root /path/to/research/project \
  --candidate candidate.json \
  --benchmark hidden_benchmark.json \
  --policy promotion_policy.json
```

After a canary:

```bash
xscientist evolution-gate \
  --project-root /path/to/research/project \
  --candidate candidate.json \
  --benchmark hidden_benchmark.json \
  --canary canary_report.json \
  --approver release-controller
```

The latest decision is written to `evolution_gate.json`; compact decisions are
appended to `evolution_gate_history.jsonl`. A held or blocked candidate remains
shadow-only.
