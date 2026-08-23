# Benchmark-driven optimization status

This status document is a decision aid, not a leaderboard. The local pilot is an
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
| First-run usability | `benchmark_first_run` records the duration for the current machine and run; no stale point estimate is promoted | Provider-free local latency only; do not compare it with model or network latency |
| Fairness | branch fixture remains `NOT VERIFIED` | Same task slice, base, budget, and evaluator are not evidenced |
| External comparison | zero model rollouts and no matched human arm | No honest cross-system or human-vs-agent score exists yet |

The JSON report now emits these as an ordered `diagnostics` backlog. A P0 is a
claim-blocking gap, P1 is evidence/lifecycle debt, and P2 is an exploration or
usability improvement.

## Capability-gap map

| Reference family | What it demonstrates | XScientist current status |
| --- | --- | --- |
| AutoResearchEval / ARFT | Long-horizon rollouts, artifact-aware judging, six-stage failure taxonomy | The repository provides an offline conformance report only. Official rollouts, the artifact-aware judge, and the annotated trajectory package are not present. |
| ScientistOne / Chain-of-Evidence | Claim provenance and integrity auditing across generated artifacts | Typed objects and explicit exports exist. A claim-level evidence package is not a default quality score, and no such score is claimed. |
| MARS / AdaEvolve / EvoX | Budget-aware search, branches, adaptive exploration | Git-like branches are inspectable, but the current fixture has one branch and fairness metadata is unverified. |
| ScholarPeer | Retrieval-assisted challenge and reviewer calibration | Review objects are recordable; retrieval quality and citation entailment are unassessed in the local pilot. |
| PaperOrchestra / PaperBanana | Specialized writing/figure outputs with dedicated benchmarks | XScientist exposes a broad lifecycle, but no component-specific quality score is reported. |
| Human studies | Matched human arms exist only for some neighboring tasks | The local human arm is explicitly `not_reported`; external numbers are kept as contextual evidence and are not substituted. |

The attached Expo Talk is used for scope discovery only; the primary papers and
repositories in [the system matrix](SYSTEM_COMPARISON.md) remain the evidence
authority. The talk PDF is retained with its SHA-256 in the machine-readable
source manifest.

## Completion status and explicit blockers

The repository has implemented the optimizations that can be verified without
claiming an external rollout: schema-validated conformance reports, atomic
redacted output, fixed-vocabulary diagnostics, bounded Git-like process
inspection, fail-closed fairness metadata with fixed unverified reasons, a
bounded read-only evidence/ARA index, exploration-graph counters, deterministic
input fingerprints, offline report verification, conservative feedback
attribution labels, lazy SDK exports, and bilingual documentation. These are
completed repository capabilities, not benchmark quality claims.

The following blockers are intentionally still visible in the report. They are
release conditions, not dated action items:

| Blocker | Current evidence | Status changes only when |
| --- | --- | --- |
| No matched model rollout | `rollouts_evaluated: 0`; `official_comparable: false` | A pinned evaluator, task slice, environment, budget, seed policy, and recorded rollout are all present. |
| No matched human arm | `human_baseline.status: not_reported`; score is `null` | A real human run uses the same task/tool/budget/verifier contract and reports uncertainty. |
| Branch fairness unverified | Current fixture has one branch; task slice/base/budget/evaluator equality is not evidenced | Every compared branch has a machine-readable comparison contract and all required fields verify. |
| Retrieval and independent verification gap | Local fixture is 5/6 structurally covered; `verify` is blocked | Required retrieval artifacts and an independent held-out verification receipt exist. |
| Feedback debt contained, not repaired | Two issues remain `contained`; zero shipped-with-issue | The issues are repaired, regression-tested, and the gate records the new evidence. |
| ARFT adapter absent | `AUDIT.ARFT_ADAPTER_MISSING` remains explicit | A typed adapter and evaluator are implemented with `unassessed`/error states preserved. |

Until these conditions are evidenced, the report must keep
`quality_claim_allowed: false` and must not turn structural coverage into a
scientific score. No timeline or unverified completion date is implied.

## Operational commands

```bash
# Offline structural audit and durable redacted report
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl --workspace ./first-study \
  --limit 20 --kind open-ended --show-process --json \
  --output ./benchmark-evidence/autoresearch-report.json

# Source-audited capability matrix (no external rollout)
xscientist benchmark systems --json --output ./benchmark-evidence/system-matrix.json

# Offline schema/boundary verification of a saved report
xscientist benchmark verify --report ./benchmark-evidence/autoresearch-report.json --json
```

Neither command produces a scientific leaderboard. They expose the current
completion state and explicit blockers so any separately controlled rerun can
be audited without retroactively changing the claim boundary.
