# Deep Research Strategy Protocol

Status: built-in additive profile `research-strategy/v1`

This protocol turns “run another experiment” into an inspectable strategy
loop. It is additive: existing Research VCS objects, profile digests, commits,
and descriptive claims remain valid. Strategy objects use the built-in profile
URI `https://xscientist.io/profiles/research-strategy/v1` and the existing
relation vocabulary, so old profiles are not silently redefined.

## What the protocol is trying to prevent

An autonomous researcher can look productive while repeatedly testing one
favored explanation, optimizing a visible benchmark, or converting correlation
into causal language. The deep-research profile therefore makes five
obligations explicit:

1. maintain at least two competing hypotheses, with an optional null;
2. state outcomes that distinguish those hypotheses before selecting a test;
3. rank a complete candidate set by expected information value and declared
   cost/risk trade-offs;
4. treat failures and contradictions as anomalies requiring explanation;
5. qualify causal and transferable claims with mechanism, quality, and boundary
   evidence.

This improves the structure of exploration. It does not make an LLM, score, or
green DAG node an authority on scientific truth.

## Immutable strategy objects

| Object kind | Required scientific content | Typical state |
| --- | --- | --- |
| `hypothesis_portfolio` | primary, alternative and optional null hypotheses; normalized locked priors | `locked` |
| `discriminating_prediction` | condition, expected outcome, rivals and falsifier | `locked` |
| `experiment_priority` | complete candidate set, per-hypothesis predictions, deterministic ranking and reasons | `locked` |
| `anomaly` | failure/contradiction type, severity, exact source IDs and resolution status | `completed` |
| `research_review` | frontier counts, structural gaps, recommended actions and review cadence | `draft` or `completed` |
| `mechanism_model` | mediators, interventions, rivals, evidence and status | `completed` or `verified` |
| `evidence_quality` | fixed risk domains, notes, grade and assessor independence | `completed` or `verified` |
| `boundary_condition` | dimension, condition, development/held-out/scale role, status and evidence | `completed` or `verified` |
| `transfer_matrix` | all boundary rows, coverage summary and transfer verdict | `completed` or `verified` |

Every payload includes a kind-specific canonical hash. Validation runs before
storage; a hash mismatch or semantically empty required field is rejected.

## Experiment priority policy

Each candidate predicts one discrete outcome for every member of the locked
portfolio. XScientist computes normalized expected information gain as the
entropy reduction induced by that deterministic outcome partition:

```text
EIG = (H(prior) - Σ_outcome P(outcome) H(posterior | outcome)) / H(prior)
```

Researchers also declare auditable integer ratings from 0 to 4. Version 1 uses:

```text
utility = 0.50*EIG
        + 0.15*novelty
        + 0.15*impact
        + 0.10*transfer_value
        - 0.10*cost
        - 0.05*risk
        - 0.05*redundancy
```

Ratings are divided by four before use. Utility is clipped to `[0, 1]`; ties
break by EIG and stable candidate ID. The full set, policy text, policy hash,
selected candidate, rejected candidates, and reasons are stored. This is a
transparent heuristic for experiment selection, not a universal utility
function or a substitute for human safety review.

## Evidence quality and depth gates

Quality assessment uses fixed domains: internal validity, measurement
reliability, confounding, statistical power, multiplicity, preregistration
fidelity, independence, and external validity. Each domain is `low_risk`,
`some_concerns`, `high_risk`, or `not_assessed`. The deterministic aggregate is
`strong`, `moderate`, `weak`, or `critical`. Only an assessment declared
independent reaches `verified`.

Claims declare one of three depth levels:

| Depth | Minimum verified-claim obligation |
| --- | --- |
| `descriptive` | existing evidence and independent gate rules |
| `causal` | descriptive closure plus a validated intervention-tested mechanism tied to selected evidence, and an independent `strong`/`moderate` quality assessment of that evidence |
| `transferable` | causal closure plus a verified transfer matrix with at least three supported tested rows, two dimensions, and a held-out/transfer/scale condition |

The lifecycle API and closure audit both enforce these conditions. A raw draft
may record an ambitious proposition for later work, but it cannot be promoted
as a verified deep claim until the qualification objects exist.

## Periodic review and anomaly handling

`research program review` deterministically scans the current repository for:

- fewer than two hypotheses or a missing competitive portfolio;
- no discriminating prediction or experiment-priority object;
- evidence without structured quality assessment;
- claims without boundary mapping;
- absent mechanism models;
- failed, timed-out, or cancelled attempts; and
- explicit refutation or contradiction relations.

A review is due for the first review, after five new scientific objects, or
when an unrecorded anomaly exists. `--record` appends anomalies and one review
checkpoint. Repeating the scan does not duplicate an anomaly with the same
source set. Resolution is append-only through new evidence, reviews, or
superseding objects; the original surprise remains visible.

## CLI workflow

```bash
xscientist research program template --output deep-research.json
xscientist research program portfolio PRIMARY --alternative RIVAL \
  --question "Which explanation predicts the intervention?"
xscientist research program prediction @latest:hypothesis_portfolio PRIMARY \
  --when "M is removed" --expect "effect disappears" \
  --distinguishes RIVAL --falsifier "effect remains"
xscientist research program prioritize \
  @latest:hypothesis_portfolio deep-research.json
xscientist research program review --record

xscientist research program mechanism PRIMARY "M mediates the effect" \
  --mediator M --intervention "do(M=0)" --rival RIVAL \
  --evidence EVIDENCE --status validated
xscientist research program quality EVIDENCE quality.json \
  --assessor human:independent-reviewer --independent
xscientist research program boundary CLAIM boundaries.json
xscientist research program claim CLAIM
```

Every mutating command checkpoints atomically by default. Use `--no-commit`
only when intentionally assembling several objects into one later checkpoint.
JSON inputs can come from any platform or tool; the public Python functions and
`ResearchRepository.review_program()` / `inspect_claim()` expose the same
semantics without shell coupling.

## DAG projection

The unified DAG adds a non-authoritative projection over the immutable objects:

- six epistemic layers: strategy, execution, evidence, theory, decision memory,
  and evolution;
- a content-hashed theory frontier with active hypotheses, mechanisms, open
  anomalies, open questions, and the next ranked experiment;
- one claim insight row with supporting/refuting IDs, mechanism, quality,
  boundaries, decision readiness, and remaining gaps.

The offline browser can filter these layers and inspect claim reasoning. The
projection is recomputed from the selected Git ref, so reviewing an old commit
cannot silently import current evidence or memory.

## Automation boundary

An agent may generate alternatives, candidate experiments, anomaly summaries,
or review recommendations. Deterministic validators decide whether records are
well-formed and whether local closure rules pass. Independent evaluators,
reproduction, external source status, domain safety controls, and human or
institutional authority remain separate. If context, evidence, or a required
qualification cannot fit or cannot be verified, the system reports an
incomplete/blocked state instead of weakening the rule.
