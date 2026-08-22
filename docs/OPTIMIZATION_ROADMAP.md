# Benchmark-driven optimization roadmap

This roadmap is a decision aid, not a leaderboard. The local pilot is an
offline conformance audit: it does not run the 100-task/800-trajectory
AutoResearchEval rollout, and it does not import human or competitor scores.
All gaps below are therefore stated as missing evidence or missing capability,
not as claims that another system is universally better.

## What the current pilot actually found

The bundled balanced fixture is a useful integrity test:

| Observation | Evidence | Meaning |
| --- | --- | --- |
| Task contracts | 20/20 open-ended and 20/20 optimization contracts in the recorded run | Framing is deterministic; this is not task-solving quality |
| Lifecycle coverage | 5/6 stages, 83.3% structural coverage | Retrieval artifacts are absent from the fixture; the percentage is not a quality score |
| Closure | trace/replay pass, verify blocked | A held-out conflict and independent reproduction gap remain visible |
| Feedback | `contained`, two unresolved issues, zero shipped-with-issue | The gate blocks release, but containment is not repair |
| Process | 3 commits, 1 branch, 16 typed artifacts | Intermediate decisions are inspectable; exploration diversity is not demonstrated |
| First-run usability | 15.99 s in the latest local run (`benchmark_first_run(max_seconds=30)`) | Provider-free local latency only; do not compare it with model or network latency |
| Fairness | branch fixture remains `NOT VERIFIED` | Same task slice, base, budget, and evaluator are not evidenced |
| External comparison | zero model rollouts and no matched human arm | No honest cross-system or human-vs-agent score exists yet |

The JSON report now emits these as an ordered `diagnostics` backlog. A P0 is a
claim-blocking gap, P1 is evidence/lifecycle debt, and P2 is an exploration or
usability improvement.

## Capability-gap map

| Reference family | What it demonstrates | XScientist gap exposed | Optimization target |
| --- | --- | --- | --- |
| AutoResearchEval / ARFT | Long-horizon rollouts, artifact-aware judging, six-stage failure taxonomy | No official rollout harness, judge, or annotated trajectory package in this repository | A pinned evaluator adapter and trajectory bundle contract |
| ScientistOne / Chain-of-Evidence | Claim provenance and integrity auditing across generated artifacts | Typed objects exist, but a shareable claim-level evidence package is still an explicit export rather than a default scored outcome | Claim → evidence → verifier manifest with independent replay receipts |
| MARS / AdaEvolve / EvoX | Budget-aware search, branches, adaptive exploration | Git-like branches are visible, but per-branch outcome, budget, and evaluator evidence are not yet comparable | Branch experiment manifests and resource-aware scheduling |
| ScholarPeer | Retrieval-assisted challenge and reviewer calibration | Review objects are recorded, but retrieval quality and citation entailment need a task-specific evaluator | Citation/claim entailment checks and adversarial review sets |
| PaperOrchestra / PaperBanana | Specialized writing/figure outputs with dedicated benchmarks | XScientist has broad lifecycle coverage but no component-specific quality score | Plug-in evaluators for manuscript, figure, and citation artifacts |
| Human studies | Matched human arms exist only for some neighboring tasks | No common human protocol for this pilot; external numbers are not interchangeable | Preregister a small matched arm only after the agent evaluator is fixed |

The attached Expo Talk is used for scope discovery only; the primary papers and
repositories in [the system matrix](SYSTEM_COMPARISON.md) remain the evidence
authority. The talk PDF is retained with its SHA-256 in the machine-readable
source manifest.

## 30 / 90 / 180 day plan

### Next 30 days — make the audit unambiguous

- Keep `quality_claim_allowed: false` until a registered evaluator and repeated
  seed policy exist.
- Use `--output` for a durable redacted report and retain the full ARA/VCS bundle
  separately under an explicit, access-controlled export.
- Close the current P0 diagnostics: record task-slice, fork-base, budget, and
  evaluator metadata for every branch; keep `eligible=false` otherwise.
- Add typed-object → ARFT evidence-channel mapping only as an explicit adapter,
  with input errors and `unassessed` states preserved.

Acceptance: a clean report validates its schemas, contains no task/gold text,
and every “pass” has a fixed verification condition.

### Next 90 days — run a fair local benchmark

- Freeze one task manifest, evaluator revision, environment/container, budget,
  and three or more seeds.
- Run the same slice through XScientist and at least one reproducible baseline;
  publish raw hashes, run receipts, failure taxonomy, and cost/time intervals.
- Add per-branch manifests and merge/rejection decisions so exploration can be
  compared without exposing hidden chain-of-thought.
- Add component evaluators: retrieval provenance/entailment, execution receipt,
  uncertainty/negative-result accounting, manuscript claim trace, and
  independent reproduction.

Acceptance: `official_comparable` can become true only when all fairness checks
are true; otherwise the report must remain explicitly unverified.

### Next 180 days — evaluate research quality, not just observability

- Reproduce the official evaluator for a pinned public task subset, with an
  artifact-aware judge and blinded adjudication.
- Add a preregistered human arm only for the same task/tool/budget protocol;
  report participant count, uncertainty, attrition, and raw process receipts.
- Measure feedback evolution: repair success, regression rate, recovery time,
  branch reuse, and whether a held gate actually prevents release.
- Publish bilingual benchmark reports containing both positive and negative
  results, plus a manifest of unavailable evidence. Never aggregate unrelated
  human baselines into one number.

Acceptance: a result is publishable only with a pinned manifest, evaluator,
environment, seeds, evidence bundle, and an explicit uncertainty statement.

## Operational commands

```bash
# Offline structural audit and durable redacted report
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl --workspace ./first-study \
  --limit 20 --kind open-ended --show-process --json \
  --output ./benchmark-evidence/autoresearch-report.json

# Source-audited capability matrix (no external rollout)
xscientist benchmark systems --json --output ./benchmark-evidence/system-matrix.json
```

Neither command produces a scientific leaderboard. They make the boundary,
missing evidence, process branches, and next actions visible so a later,
properly controlled run can be audited rather than retroactively rationalized.
