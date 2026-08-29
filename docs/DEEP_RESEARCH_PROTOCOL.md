# Deep Research Strategy Protocol

Status: built-in additive profile `research-strategy/v2`

This protocol turns “run another experiment” into an inspectable strategy
loop. It is additive: existing Research VCS objects, profile digests, commits,
and descriptive claims remain valid. Strategy objects use the built-in profile
URI `https://xscientist.io/profiles/research-strategy/v2`. The frozen v1
descriptor and validator remain registered, so old commits are not silently
reinterpreted. Newly created strategy objects use v2.

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

The guided next-action policy makes the first obligation concrete. If one
hypothesis exists and no research plan is locked, its first blocking action is
to record a falsifiable rival. If multiple hypotheses exist but no portfolio
does, its first action is to lock their question, alternatives, and priors in a
`hypothesis_portfolio`. Only then does it present exploratory, confirmatory, and
method-discovery planning choices. This ordering is deterministic guidance; it
does not certify that a proposed rival or prior is scientifically adequate.

For the literature-to-opportunity part of a program, see the
[FAR-inspired opportunity funnel](OPPORTUNITY_FUNNEL.md). It records a
research direction, a complete candidate pool, explicit `known`/`new`/`fix`/
`none` attempts, evaluator-disjoint judgments, and declared allocation
estimates. The funnel is a process contract, not a quality or human-baseline
score; it does not alter the information-value policy below or silently impute
primary probabilities. The default conditional allocation mode may record an
explicit neutral `1.0` assumption for a missing conditional artifact factor;
joint-probability mode leaves that row unselected.

For the execution/evaluation part of the loop, see the
[research-policy rollout contract](RESEARCH_ROLLOUTS.md). It records which
outer policy delegated each metadata-only tool call, the task-specific rubric,
observational turn credit, and evaluator disagreement. It deliberately does
not turn an LLM judge into a ground-truth oracle or claim Faraday's scores
locally.

Strict publication and quality-gated runs also require every LLM call trace to
be persisted successfully. A trace storage failure stops the run instead of
silently producing an unverifiable result. Independence receipts currently
prove deterministic disjointness between declared actor IDs in the complete
producer lineage; they explicitly do not claim authenticated real-world
identity separation.

## Literature evidence contract

New literature records form a committed chain rather than a collection of
loosely related URLs:

```text
locked search_plan
  → hash-bound search_receipt for an exact query/candidate set
  → uniquely selected source_snapshot
  → exact passage_evidence
```

The receipt binds the locked plan's object/content/plan hashes, must use an
exact locked query, and—when the plan declares a provider allowlist—must use one
of those providers. The source recomputes the receipt commitment and candidate
set hash, must match exactly one selected candidate by normalized identifiers
or unambiguous title, and stores both receipt and candidate bindings. Closure
audit recomputes the same constraints, so forged relation-only objects do not
become trace-complete. Legacy records remain readable only when their relations
and payloads satisfy the recomputed contract.

Source status is monotonic and append-only. A retraction, withdrawal, or invalid
status keeps invalidating the source; a later ordinary positive status check
cannot erase it. Reinstatement must carry a notice, come from the same provider
as the latest active retraction, have a later checked-at time, and explicitly
`supersedes` that retraction. Historical belief/context decisions apply their
explicit `as_of` cutoff to evidence, source updates, and source lineage. Future
retraction or reinstatement events do not rewrite the past, while a lineage root
that did not yet exist is excluded and surfaced as an action blocker.

## Immutable strategy objects

| Object kind | Required scientific content | Typical state |
| --- | --- | --- |
| `hypothesis_portfolio` | primary, alternative and optional null hypotheses; normalized locked priors | `locked` |
| `discriminating_prediction` | condition, expected outcome, rivals and falsifier | `locked` |
| `experiment_design` | executable candidate bound to one locked prediction per portfolio hypothesis | `locked` |
| `experiment_priority` | candidate design IDs, prior source, deterministic ranking and reasons | `locked` |
| `observation` | observed outcome bound to selected attempt and evidence | `completed` |
| `posterior_update` | prior, likelihoods, normalized posterior and exact design/attempt/observation/evidence IDs | `completed` |
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

Researchers also declare auditable integer ratings from 0 to 4. Version 2 uses:

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
selected candidate, rejected candidates, their locked `experiment_design`
objects, and reasons are stored. A candidate is rejected unless every outcome
matches exactly one previously locked `discriminating_prediction` under the
same condition. An attempt may claim to execute a competitive design only when
it consumes the priority that selected that design. This remains a transparent
heuristic, not a universal utility function or substitute for human review.

After execution, `program posterior` creates an immutable observation and
applies the declared discrete likelihoods using Bayes' rule. The validator
recomputes the posterior, prevents evidence reuse within a portfolio, and uses
the latest non-superseded posterior as the next ranking prior. Its epistemic
status is `agent_computed_draft`; it is not self-promoted to verified.

## Evidence quality and depth gates

Quality assessment uses fixed domains: internal validity, measurement
reliability, confounding, statistical power, multiplicity, preregistration
fidelity, independence, and external validity. Each domain is `low_risk`,
`some_concerns`, `high_risk`, or `not_assessed`. The deterministic aggregate is
`strong`, `moderate`, `weak`, or `critical`. An independent assessment reaches
`verified` only when its assessor is absent from the evidence's complete
producer-provenance closure. The policy, producer actor IDs, traversed object
IDs, and receipt hash are stored; changing a Boolean flag is insufficient.

A validated mechanism must cite verified evidence derived from a completed
attempt. That attempt must bind a locked plan/design containing every claimed
intervention. The validation receipt fixes the evidence, attempt, protocol, and
matched intervention IDs.

Claims declare one of three depth levels:

| Depth | Minimum verified-claim obligation |
| --- | --- |
| `descriptive` | existing evidence and independent gate rules |
| `causal` | descriptive closure plus a validated intervention-tested mechanism tied to selected evidence, and an independent `strong`/`moderate` quality assessment of that evidence |
| `transferable` | causal closure plus a verified matrix with at least three supported rows, two dimensions, a transfer condition, disjoint evidence/attempts, and disjoint development/held-out dataset identities |

The lifecycle API and closure audit both enforce these conditions. A raw draft
may record an ambitious proposition for later work, but it cannot be promoted
as a verified deep claim until the qualification objects exist.

### Complete active claim closure

Verification cannot select only favorable evidence. The active closure contains
all evidence and argument objects reachable from the current semantic claim
identity, active and resolved challenge/refutation objects, the closure nodes
they challenge, and immutable resolution records. At least one independent,
verified review must by itself evaluate that entire set; split partial reviews
cannot be unioned into authority. The deterministic gate must cover the same
closure and bind that review. Its actor-disjointness receipt is a local declared
provenance check, not proof of real-world identity independence.

Any active `refutes`, `qualified_refutes`, `contradicts`, or
`challenges_inference` signal blocks the `verify` level, while `trace` and
`replay` may remain complete and useful. A bare `supersedes` edge cannot make a
challenge disappear: the resolution must be an active immutable object, and a
fresh independent review and gate must cover both records. Superseded reviews,
gates, and reproductions cannot be reused.

A verified reproduction uses receipt v2 to bind four independently recomputed
surfaces: the source Git checkpoint, exact reproduced object hashes, the active
claim closure at that same audit checkpoint, and the execution result. A stale
closure or substituted target therefore remains invalid even if an outer
receipt hash is recomputed. Valid locally generated v1 receipts can be upgraded
before recording, while historical v1 objects remain readable but cannot close
`verify`. These repository hashes do not prove that a declared verifier is a
distinct real-world person or organization.

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
checkpoint. Repeating the scan does not duplicate an open anomaly with the same
type and source set. Different types do not collide. A resolved or superseded
anomaly does not suppress a later recurrence; the new occurrence reopens as a
new immutable object while the original resolution remains visible.

## CLI workflow

```bash
xscientist research program template --output deep-research.json
xscientist research program portfolio PRIMARY --alternative RIVAL \
  --question "Which explanation predicts the intervention?"
xscientist research program prediction @latest:hypothesis_portfolio PRIMARY \
  --when "M is removed" --expect "effect disappears" \
  --distinguishes RIVAL --falsifier "effect remains"
xscientist research program prediction @latest:hypothesis_portfolio RIVAL \
  --when "M is removed" --expect "effect remains" \
  --distinguishes PRIMARY --falsifier "effect disappears"
xscientist research program prioritize \
  @latest:hypothesis_portfolio deep-research.json

xscientist research experiment "mediator ablation" --status completed \
  --plan SELECTED_DESIGN_ID --priority PRIORITY_ID
xscientist research evidence "effect disappeared" --attempt ATTEMPT_ID
xscientist research program posterior PORTFOLIO_ID PRIORITY_ID ATTEMPT_ID EVIDENCE_ID \
  --observed "effect disappeared" \
  --likelihood PRIMARY=0.9 --likelihood RIVAL=0.1
xscientist research program review --record

xscientist research program mechanism PRIMARY "M mediates the effect" \
  --mediator M --intervention "do(M=0)" --rival RIVAL \
  --evidence EVIDENCE --status validated
xscientist research program quality EVIDENCE quality.json \
  --assessor human:independent-reviewer --independent
xscientist research program boundary CLAIM boundaries.json
xscientist research program claim CLAIM
```

Every mutating one-command save checkpoints atomically by default. Commit mode
requires an empty native stage and a clean staged/tracked/research-eligible
worktree, then commits only the exact object paths created by that call plus the
checkpoint records. Use `--no-commit` only when intentionally assembling
several objects into one later explicitly staged checkpoint. JSON inputs can
come from any platform or tool; the public Python functions and
`ResearchRepository.review_program()` / `inspect_claim()` expose the same
semantics without shell coupling.

Portable JSON guidance also separates a recommendation from an executable
command. Each `primary_action`/`next_steps[]` row has a stable action `code`, an
`argv_template` using exact object IDs where available, explicit
workspace/cwd binding, and `input_binding`. `{workspace}` must be bound to the
workspace supplied to the invocation. If a human-input placeholder remains,
`input_binding.required=true` and `executable_after_binding=false`; an agent
must request that input rather than guessing it or executing the template.

## Repository execution and portability boundary

Research-sensitive operations require the exact selected commit to contain a
valid checkpoint JSON/trailer/parent/diff binding. They do not fall back to an
ancestor when a branch tip is an ordinary raw Git commit. Such a tip therefore
blocks inspection as a checkpoint, reproduction, tag, bundle, export, and
semantic merge; copied trailers do not confer authority.

Semantic merge preflight detects backend conflicts, incompatible locked
registrations, metric redefinitions, and newly introduced support/refutation
pairs even when support or refutation already exists at the merge base. The
prepared merge index must exactly match declared paths. The gate scans the
corresponding working-tree paths only after verifying staged/worktree agreement,
then repeats that agreement check after the scan; it does not claim to scan Git
index blobs directly. Opposing evidence may be preserved only under an explicit
hold; this is not conflict resolution or claim promotion.

Reproduction uses an exact detached worktree, verified CAS hydration, a
reduced variable environment/private HOME, no shell, bounded retained output,
and a timeout. The environment control is variables-only and the host filesystem
remains visible. POSIX process-group cleanup is best-effort; Windows terminates
only the parent process, so neither platform provides a process-tree guarantee.
The receipt explicitly reports `isolated=false`, `security_boundary=false`,
`environment_scope=variables_only`, `filesystem=host_visible`, and
`network=host_unrestricted`; it is not an OS security boundary. New v2 receipts
persist this boundary under the receipt hash, and closure audit rejects stronger
or internally mismatched isolation claims. Upgraded v1 receipts use
`legacy_unknown` environment/process values because the old schema did not
retain enough information to reconstruct those controls.
Output hashes bind only the retained bounded tails. New receipts also bind the
capture limit and stdout/stderr truncation flags; upgraded v1 receipts preserve
the absent scope as `legacy_unknown`.
Bundles cover all refs advertised by their embedded Git bundle and include the
historical CAS closure for `reproduce`/`audit`, including old tags and
non-current branches. Verification locally imports the Git bundle and
recomputes that closure before checking pointer and CAS hashes/sizes. These are
local integrity controls, not external custody, signature, or trust proof.

## DAG projection

The unified DAG adds a non-authoritative projection over the immutable objects:

- six epistemic layers: strategy, execution, evidence, theory, decision memory,
  and evolution;
- a content-hashed theory frontier with active hypotheses, per-portfolio
  posterior state, mechanisms, anomalies, questions, and ranked experiments;
- one claim insight row with supporting/refuting IDs, mechanism, quality,
  boundaries, decision readiness, and remaining gaps.

The offline browser can filter these layers and inspect claim reasoning. The
projection is recomputed from effective, non-superseded objects at the selected
Git ref. A claim receives guidance only from portfolios containing a hypothesis
in its evidence/mechanism lineage, so unrelated branches cannot leak a globally
latest experiment into that claim. When a belief projection also declares a
historical `as_of`, evidence, lineage roots, retractions, and reinstatements
created after that logical time are excluded even if they exist at the selected
ref. A future lineage root is reported as unavailable instead of being replaced
with an actor-derived source family.

## v1 to v2 compatibility

- Existing v1 objects retain their original descriptor and legacy semantic
  checks. Historical fsck, DAG views, bundles, and commit hashes remain valid.
- New records default to v2; no object is migrated or rewritten in place.
- Strengthen an old frontier by appending v2 objects and linking replacements
  with `supersedes`.
- A v1 priority remains visible as history but does not satisfy the v2
  prediction→design→attempt→observation→posterior closure.

## Automation boundary

An agent may generate alternatives, candidate experiments, anomaly summaries,
or review recommendations. Deterministic validators decide whether records are
well-formed and whether local closure rules pass. Independent evaluators,
reproduction, external source status, domain safety controls, and human or
institutional authority remain separate. If context, evidence, or a required
qualification cannot fit or cannot be verified, the system reports an
incomplete/blocked state instead of weakening the rule.
